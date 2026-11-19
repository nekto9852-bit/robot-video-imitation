#!/usr/bin/env python3
"""pose_to_trajectories.py - pipeline "video -> 50 robot trajectories".

Stages
------
1.  Human pose extraction from a video via MediaPipe Pose (33 landmarks per
    frame, world coordinates). Alternatives that do not need mediapipe:
    ``--landmarks landmarks.json`` (pre-extracted frames) or ``--demo``
    (synthetic landmarks for a smoke test).
2.  IK retargeting: human arm -> 6-DOF robot arm joint angles. Targets are
    scaled human wrist positions expressed in the robot base frame; solved
    per frame with damped least squares (DLS), joint limits, warm start and
    a nullspace bias towards a rest pose.
3.  Augmentation: N variations of the base trajectory - smoothed Gaussian
    angle noise, playback speed change (time resampling), amplitude scaling.
4.  Metrics per trajectory: ``pose_error`` (mean |dAngle| vs the reference on
    a common time grid), ``smoothness`` (total jerk of the angles), joint
    limit validity and a simplified self-collision check, plus a ``valid``
    flag combining the configured thresholds.
5.  Export: trajectories.json, trajectories.csv (angles per timestep) and
    metrics.json.

Dependencies: numpy + pyyaml (always); mediapipe + opencv only for --video.

Example
-------
    python pose_to_trajectories.py --video input.mp4 --config config.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOG = logging.getLogger("pose2traj")

__all__ = [
    "RobotChain",
    "build_robot",
    "solve_ik_dls",
    "extract_pose_landmarks",
    "load_landmarks_json",
    "make_demo_landmarks",
    "retarget_landmarks_to_angles",
    "generate_variants",
    "resample_speed",
    "resample_to_length",
    "pose_error_deg",
    "jerk_stats",
    "limit_violations",
    "check_collisions",
    "evaluate_trajectory",
    "load_config",
    "default_config",
    "run_pipeline",
    "main",
]

# --------------------------------------------------------------------------- #
# MediaPipe Pose landmark indices (subset). Plain ints keep the retargeting
# path runnable without mediapipe installed.
# --------------------------------------------------------------------------- #
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24

N_LANDMARKS = 33

ARM_SIDES: dict[str, tuple[int, int, int]] = {
    "left": (LM_LEFT_SHOULDER, LM_LEFT_ELBOW, LM_LEFT_WRIST),
    "right": (LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW, LM_RIGHT_WRIST),
}

# --------------------------------------------------------------------------- #
# Default configuration (config.yaml may override any subset of it).
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: dict = {
    "robot": {
        # 6-DOF arm: base yaw (z), shoulder pitch (y), shoulder roll (x),
        # elbow pitch (y), wrist pitch (y), wrist roll (x).
        "link_lengths_m": [0.10, 0.16, 0.14, 0.12, 0.08, 0.06],
        "tool_offset_m": 0.04,
        "joint_limits_deg": [
            [-170.0, 170.0],  # J1 base yaw
            [-170.0, 170.0],  # J2 shoulder pitch
            [-110.0, 110.0],  # J3 shoulder roll
            [-150.0, 150.0],  # J4 elbow
            [-110.0, 110.0],  # J5 wrist pitch
            [-180.0, 180.0],  # J6 wrist roll
        ],
        # Simplified self-collision proxy: keep elbow/wrist/tool clear of a
        # cylinder around the robot base axis ("torso" below the shoulder).
        "collision": {
            "enabled": True,
            "column_radius_m": 0.07,
            "column_height_m": 0.10,
            "min_clearance_m": 0.02,
        },
    },
    "retarget": {
        "side": "auto",  # auto | left | right
        "damping": 0.05,
        "max_iters": 60,
        "tol_m": 1e-4,
        "nullspace_gain": 0.05,
        "rest_pose_deg": [0.0, 0.0, 0.0, -30.0, 0.0, 0.0],
        "min_visibility": 0.5,
        "smoothing_window": 5,
        "target_scale": 0.98,
        "frame": {"depth_sign": 1.0, "mirror_x": False},
    },
    "pose": {
        "model_complexity": 1,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "fps_fallback": 30.0,
        # Only used by mediapipe >= 1.0 (Tasks API): auto-downloaded if missing.
        "landmarker_model": "pose_landmarker_full.task",
    },
    "augmentation": {
        "n_trajectories": 50,
        "speed_range": [0.7, 1.3],
        "amplitude_range": [0.9, 1.1],
        "angle_noise_std_deg": 2.0,
        "noise_std_jitter": [0.5, 1.5],
        "noise_smooth_window": 3,
        "seed": 42,
    },
    "thresholds": {
        "pose_error_max_deg": 8.0,
        "max_collisions": 0,
    },
    "output": {
        "format": "both",  # json | csv | both
        "dir": "output",
        "round_decimals": 4,
    },
}

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def default_config() -> dict:
    """Return a fresh deep copy of the built-in default configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config and merge it over the defaults (deep merge)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    if not isinstance(user, dict):
        raise ValueError(f"config root must be a mapping, got {type(user).__name__}")
    unknown = set(user) - set(DEFAULT_CONFIG)
    if unknown:
        LOG.warning("unknown config sections ignored: %s", sorted(unknown))
    cfg = _deep_merge(default_config(), user)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    au = cfg["augmentation"]
    if int(au["n_trajectories"]) < 1:
        raise ValueError("augmentation.n_trajectories must be >= 1")
    for key in ("speed_range", "amplitude_range", "noise_std_jitter"):
        lo, hi = (float(x) for x in au[key])
        if lo > hi:
            raise ValueError(f"augmentation.{key}: min ({lo}) > max ({hi})")
    if float(au["angle_noise_std_deg"]) < 0:
        raise ValueError("augmentation.angle_noise_std_deg must be >= 0")
    if float(cfg["thresholds"]["pose_error_max_deg"]) <= 0:
        raise ValueError("thresholds.pose_error_max_deg must be > 0")
    build_robot(cfg)  # validates the robot section


# --------------------------------------------------------------------------- #
# Robot kinematics
# --------------------------------------------------------------------------- #


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix around a unit axis."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    c, s = np.cos(angle), np.sin(angle)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def _homogeneous(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = t
    return mat


@dataclass
class RobotChain:
    """Serial 6-DOF arm used for retargeting.

    Joint axes in each joint's local frame: base yaw (z), shoulder pitch (y),
    shoulder roll (x), elbow pitch (y), wrist pitch (y), wrist roll (x).
    After each joint the frame translates along its local z by the link
    length; ``tool_offset`` extends beyond the last joint.
    """

    axes: np.ndarray  # (6, 3) local rotation axes
    link_lengths: np.ndarray  # (6,) meters
    tool_offset: float  # meters
    joint_limits: np.ndarray  # (6, 2) radians, [lo, hi] per joint

    DEFAULT_AXES = np.array(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
        dtype=float,
    )

    @property
    def reach(self) -> float:
        """Maximum distance from the base origin to the tool tip (meters)."""
        return float(self.link_lengths.sum() + self.tool_offset)

    def fk(self, q) -> np.ndarray:
        """Forward kinematics: joint angles (rad) -> tool tip position (m)."""
        q = np.asarray(q, dtype=float)
        t = np.eye(4)
        for i in range(self.axes.shape[0]):
            # Rotate about the joint axis at the current frame origin, then
            # translate along the rotated local z (Rot @ Trans order).
            rot = _axis_rotation(self.axes[i], q[i])
            t = t @ _homogeneous(rot, np.zeros(3))
            t = t @ _homogeneous(np.eye(3), np.array([0.0, 0.0, self.link_lengths[i]]))
        t = t @ _homogeneous(np.eye(3), np.array([0.0, 0.0, self.tool_offset]))
        return t[:3, 3].copy()

    def joint_positions(self, q) -> np.ndarray:
        """Positions of base, all 6 joint origins and the tool tip -> (8, 3)."""
        q = np.asarray(q, dtype=float)
        t = np.eye(4)
        points = [np.zeros(3)]
        for i in range(self.axes.shape[0]):
            rot = _axis_rotation(self.axes[i], q[i])
            t = t @ _homogeneous(rot, np.zeros(3))
            t = t @ _homogeneous(np.eye(3), np.array([0.0, 0.0, self.link_lengths[i]]))
            points.append(t[:3, 3].copy())
        points.append(points[-1] + np.array([0.0, 0.0, self.tool_offset]))
        return np.stack(points, axis=0)

    def jacobian(self, q, eps: float = 1e-6) -> np.ndarray:
        """Numerical position Jacobian (3 x 6) via central forward differences."""
        q = np.asarray(q, dtype=float)
        j = np.zeros((3, 6))
        p0 = self.fk(q)
        for i in range(6):
            qp = q.copy()
            qp[i] += eps
            j[:, i] = (self.fk(qp) - p0) / eps
        return j


def build_robot(cfg: dict) -> RobotChain:
    """Build the robot chain from ``cfg["robot"]`` (validates lengths/limits)."""
    robot_cfg = cfg["robot"]
    links = np.asarray(robot_cfg["link_lengths_m"], dtype=float)
    if links.shape != (6,) or np.any(links <= 0):
        raise ValueError("robot.link_lengths_m must be 6 positive numbers")
    limits_deg = np.asarray(robot_cfg["joint_limits_deg"], dtype=float)
    if limits_deg.shape != (6, 2):
        raise ValueError("robot.joint_limits_deg must be a 6x2 [min, max] list")
    if np.any(limits_deg[:, 0] >= limits_deg[:, 1]):
        raise ValueError("robot.joint_limits_deg: min must be < max for every joint")
    return RobotChain(
        axes=RobotChain.DEFAULT_AXES.copy(),
        link_lengths=links,
        tool_offset=float(robot_cfg["tool_offset_m"]),
        joint_limits=np.deg2rad(limits_deg),
    )


# --------------------------------------------------------------------------- #
# Inverse kinematics (damped least squares)
# --------------------------------------------------------------------------- #


def solve_ik_dls(
    chain: RobotChain,
    target,
    q_init,
    damping: float = 0.05,
    max_iters: int = 60,
    tol_m: float = 1e-4,
    q_rest=None,
    nullspace_gain: float = 0.0,
    max_step_rad: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Position IK via damped least squares (Levenberg-Marquardt style).

    dq = J^T (J J^T + lambda^2 I)^-1 e,  optionally plus a nullspace term
    (I - J^+ J) * gain * (q_rest - q) that pulls redundant DOFs towards a
    rest pose. Joint limits are enforced by clipping after every step.

    Returns (q, final_position_error_m).
    """
    lo = chain.joint_limits[:, 0]
    hi = chain.joint_limits[:, 1]
    q = np.clip(np.asarray(q_init, dtype=float).copy(), lo, hi)
    target = np.asarray(target, dtype=float)
    q_rest = None if q_rest is None else np.asarray(q_rest, dtype=float)
    eye3, eye6 = np.eye(3), np.eye(6)

    for _ in range(int(max_iters)):
        err = target - chain.fk(q)
        if np.linalg.norm(err) <= tol_m:
            break
        jac = chain.jacobian(q)
        jjt_inv_err = np.linalg.solve(jac @ jac.T + damping**2 * eye3, err)
        dq = jac.T @ jjt_inv_err
        if q_rest is not None and nullspace_gain > 0:
            j_pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping**2 * eye3)
            nullspace = eye6 - j_pinv @ jac
            dq = dq + nullspace @ (nullspace_gain * (q_rest - q))
        step = np.linalg.norm(dq)
        if step > max_step_rad:
            dq *= max_step_rad / step
        q = np.clip(q + dq, lo, hi)

    final_err = float(np.linalg.norm(chain.fk(q) - target))
    return q, final_err


# --------------------------------------------------------------------------- #
# Landmark helpers (smoothing, side selection, frame mapping)
# --------------------------------------------------------------------------- #


def _moving_average_cols(arr: np.ndarray, window: int) -> np.ndarray:
    """Moving average along axis 0 with edge replication (window <= 1 -> copy)."""
    arr = np.asarray(arr, dtype=float)
    w = int(window)
    if w <= 1 or arr.shape[0] == 0:
        return arr
    pad = w // 2
    padded = np.pad(arr, [(pad, w - 1 - pad)] + [(0, 0)] * (arr.ndim - 1), mode="edge")
    flat = padded.reshape(padded.shape[0], -1)
    kernel = np.ones(w) / w
    out = np.empty((arr.shape[0], flat.shape[1]))
    for c in range(flat.shape[1]):
        out[:, c] = np.convolve(flat[:, c], kernel, mode="valid")
    return out.reshape(arr.shape)


def pick_arm_side(landmarks: np.ndarray, side: str = "auto") -> tuple[str, tuple[int, int, int]]:
    """Resolve which human arm to retarget. ``auto`` picks the more visible one."""
    if side not in ("auto", "left", "right"):
        raise ValueError(f"side must be auto|left|right, got {side!r}")
    if side == "auto":
        left_vis = float(np.nanmean(landmarks[:, [11, 13, 15], 3]))
        right_vis = float(np.nanmean(landmarks[:, [12, 14, 16], 3]))
        side = "left" if left_vis > right_vis else "right"
    return side, ARM_SIDES[side]


def detect_y_up(landmarks: np.ndarray) -> bool:
    """Detect the vertical axis sign of the landmark coordinate system.

    Compares median shoulder vs hip height, so it works both for MediaPipe
    world landmarks (y up) and image-space conventions (y down).
    """
    shoulders = (landmarks[:, 11, 1] + landmarks[:, 12, 1]) / 2.0
    hips = (landmarks[:, 23, 1] + landmarks[:, 24, 1]) / 2.0
    diff = float(np.nanmedian(shoulders - hips))
    if abs(diff) < 1e-9:
        return True
    return diff > 0


def _map_to_robot_frame(vec: np.ndarray, y_up: bool, depth_sign: float, mirror_x: bool) -> np.ndarray:
    """Human landmark deltas -> robot base frame (x right, y forward, z up)."""
    x = -vec[0] if mirror_x else vec[0]
    y = depth_sign * vec[2]
    z = vec[1] if y_up else -vec[1]
    return np.array([x, y, z])


# --------------------------------------------------------------------------- #
# Retargeting
# --------------------------------------------------------------------------- #


def retarget_landmarks_to_angles(
    landmarks: np.ndarray,
    robot: RobotChain,
    cfg: dict,
    side: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Retarget human arm landmarks to 6 joint angles (degrees) per frame.

    ``landmarks``: (T, 33, >=3) array, world coordinates in meters with
    per-landmark visibility in column 3 when present. The human wrist vector
    relative to the shoulder is scaled to robot proportions and used as the
    IK target; each frame warm-starts from the previous solution.

    Returns (angles_deg (T, 6), info dict).
    """
    rt = cfg["retarget"]
    lm = np.asarray(landmarks, dtype=float)
    if lm.ndim != 3 or lm.shape[1] != N_LANDMARKS or lm.shape[2] < 3:
        raise ValueError(f"landmarks must have shape (T, {N_LANDMARKS}, >=3), got {lm.shape}")

    pos = _moving_average_cols(lm[:, :, :3], int(rt.get("smoothing_window", 1)))
    vis = lm[:, :, 3] if lm.shape[2] >= 4 else np.ones(lm.shape[:2])

    side_name, (i_sho, i_elb, i_wri) = pick_arm_side(lm, side or rt.get("side", "auto"))
    y_up = detect_y_up(lm)
    frame_cfg = rt.get("frame", {})
    depth_sign = float(frame_cfg.get("depth_sign", 1.0))
    mirror_x = bool(frame_cfg.get("mirror_x", False))

    rest = np.deg2rad(np.asarray(rt.get("rest_pose_deg", [0.0] * 6), dtype=float))
    min_vis = float(rt.get("min_visibility", 0.5))
    arm_reach = float(robot.link_lengths[1:].sum() + robot.tool_offset)
    max_target_r = float(rt.get("target_scale", 0.98)) * arm_reach

    n_frames = lm.shape[0]
    angles = np.zeros((n_frames, 6))
    pos_err = np.zeros(n_frames)
    q_prev = np.clip(rest, robot.joint_limits[:, 0], robot.joint_limits[:, 1])
    target: np.ndarray | None = None
    frames_held = 0

    for t in range(n_frames):
        sho, elb, wri = pos[t, i_sho], pos[t, i_elb], pos[t, i_wri]
        if min(vis[t, i_sho], vis[t, i_elb], vis[t, i_wri]) >= min_vis:
            bone = float(np.linalg.norm(elb - sho) + np.linalg.norm(wri - elb))
            if bone > 0.05:  # sane arm geometry
                scale = arm_reach / bone
                target = _map_to_robot_frame(wri - sho, y_up, depth_sign, mirror_x) * scale
                radius = float(np.linalg.norm(target))
                if radius > max_target_r:
                    target *= max_target_r / radius
        if target is None:
            # No usable detection yet: park the tool at a reachable default.
            target = np.array([0.0, 0.0, 0.5 * max_target_r])
            frames_held += 1
        elif min(vis[t, i_sho], vis[t, i_elb], vis[t, i_wri]) < min_vis:
            frames_held += 1

        q_prev, err = solve_ik_dls(
            robot,
            target,
            q_prev,
            damping=float(rt.get("damping", 0.05)),
            max_iters=int(rt.get("max_iters", 60)),
            tol_m=float(rt.get("tol_m", 1e-4)),
            q_rest=rest,
            nullspace_gain=float(rt.get("nullspace_gain", 0.05)),
        )
        angles[t] = q_prev
        pos_err[t] = err

    info = {
        "side": side_name,
        "y_up": y_up,
        "frames_held": int(frames_held),
        "mean_pos_err_m": float(pos_err.mean()),
        "max_pos_err_m": float(pos_err.max()),
        "arm_reach_m": arm_reach,
    }
    return np.rad2deg(angles), info


# --------------------------------------------------------------------------- #
# Pose extraction (MediaPipe) and landmark I/O
# --------------------------------------------------------------------------- #


DEFAULT_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def _ensure_model_file(path: str, url: str) -> None:
    """Download the pose landmarker model once if it is not on disk yet."""
    p = Path(path)
    if p.is_file():
        return
    LOG.info("pose landmarker model not found, downloading: %s", url)
    import urllib.request

    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(p)


class _LegacyPoseDetector:
    """mediapipe 0.10.x Solutions API wrapper."""

    def __init__(self, mp: Any, pose_cfg: dict):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(pose_cfg.get("model_complexity", 1)),
            min_detection_confidence=float(pose_cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(pose_cfg.get("min_tracking_confidence", 0.5)),
        )

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        result = self._pose.process(frame_rgb)
        if result.pose_world_landmarks is None:
            return None
        return np.array(
            [[p.x, p.y, p.z, p.visibility] for p in result.pose_world_landmarks.landmark],
            dtype=float,
        )

    def close(self) -> None:
        self._pose.close()


class _TasksPoseDetector:
    """mediapipe >= 1.0 Tasks API wrapper (VIDEO running mode)."""

    def __init__(self, mp: Any, pose_cfg: dict):
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = str(pose_cfg.get("landmarker_model", "pose_landmarker_full.task"))
        _ensure_model_file(model_path, DEFAULT_LANDMARKER_URL)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=float(pose_cfg.get("min_detection_confidence", 0.5)),
            min_pose_presence_confidence=float(pose_cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(pose_cfg.get("min_tracking_confidence", 0.5)),
        )
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        world = result.pose_world_landmarks
        if not world:
            return None
        return np.array([[p.x, p.y, p.z, p.visibility] for p in world[0]], dtype=float)

    def close(self) -> None:
        self._landmarker.close()


def extract_pose_landmarks(video_path: str | Path, cfg: dict) -> tuple[np.ndarray, float]:
    """Extract per-frame pose landmarks from a video via MediaPipe Pose.

    Supports both the legacy Solutions API (mediapipe 0.10.x) and the Tasks
    API (mediapipe >= 1.0, downloads ``pose.landmarker_model`` on first use).

    Returns (landmarks (T, 33, 4) in world coordinates [x, y, z, visibility],
    fps). Frames without a detection carry the last known pose.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "video input requires 'mediapipe' (pip install mediapipe). "
            "Alternatively use --landmarks or --demo."
        ) from exc

    pose_cfg = cfg["pose"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(cap_fps) if cap_fps and cap_fps > 1.0 else float(pose_cfg.get("fps_fallback", 30.0))

    if hasattr(mp, "solutions"):
        detector: Any = _LegacyPoseDetector(mp, pose_cfg)
    else:
        detector = _TasksPoseDetector(mp, pose_cfg)

    frames: list[np.ndarray] = []
    last: np.ndarray | None = None
    n_missing = 0
    timestamp_ms = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            row = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms)
            timestamp_ms += max(1, int(round(1000.0 / fps)))
            if row is not None:
                last = row
                frames.append(row)
            elif last is not None:
                frames.append(last.copy())
                n_missing += 1
            if len(frames) and len(frames) % 100 == 0:
                LOG.info("processed %d frames...", len(frames))
    finally:
        detector.close()
        cap.release()

    if not frames or last is None:
        raise RuntimeError("no human pose detected in the video")
    LOG.info("pose extraction: %d frames (%d carried from previous)", len(frames), n_missing)
    return np.stack(frames, axis=0), fps


def load_landmarks_json(path: str | Path) -> tuple[np.ndarray, float]:
    """Load pre-extracted landmarks: {"fps": 30, "frames": [{"landmarks": [[x,y,z,(vis)] x33]}]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("landmarks JSON must contain a non-empty 'frames' list")
    rows = []
    for i, frame in enumerate(frames):
        lms = frame.get("landmarks") if isinstance(frame, dict) else None
        if not isinstance(lms, list) or len(lms) != N_LANDMARKS:
            raise ValueError(f"frame {i}: 'landmarks' must be a list of {N_LANDMARKS} entries")
        row = np.zeros((N_LANDMARKS, 4), dtype=float)
        for k, point in enumerate(lms):
            if not isinstance(point, (list, tuple)) or len(point) not in (3, 4):
                raise ValueError(f"frame {i} landmark {k}: expected [x, y, z] or [x, y, z, vis]")
            row[k, : len(point)] = point
            if len(point) == 3:
                row[k, 3] = 1.0
        rows.append(row)
    return np.stack(rows, axis=0), float(data.get("fps", 30.0))


def make_demo_landmarks(n_frames: int = 90, fps: float = 30.0, seed: int = 0) -> tuple[np.ndarray, float]:
    """Synthetic MediaPipe-style world landmarks (y up, meters) of a standing
    person whose right arm traces a smooth arc. Used by --demo and tests."""
    rng = np.random.default_rng(seed)
    n_frames = max(8, int(n_frames))
    lm = np.zeros((n_frames, N_LANDMARKS, 4), dtype=float)
    lm[:, :, 3] = 1.0  # visibility

    static = {
        LM_LEFT_SHOULDER: (-0.18, 1.35, 0.03),
        LM_RIGHT_SHOULDER: (0.18, 1.35, -0.03),
        LM_LEFT_HIP: (-0.10, 0.90, 0.02),
        LM_RIGHT_HIP: (0.10, 0.90, -0.02),
    }
    lm[:, :, :3] = np.array([0.0, 1.05, 0.0])  # neutral fill for unused landmarks
    for idx, xyz in static.items():
        lm[:, idx, :3] = xyz
    # Quiet left arm hanging down.
    lm[:, LM_LEFT_ELBOW, :3] = np.array(static[LM_LEFT_SHOULDER]) + np.array([0.0, -0.28, 0.0])
    lm[:, LM_LEFT_WRIST, :3] = np.array(static[LM_LEFT_SHOULDER]) + np.array([0.0, -0.54, 0.0])

    phase = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    elev = np.deg2rad(55.0 + 30.0 * np.sin(2.0 * phase))  # elevation from vertical up
    azim = np.deg2rad(45.0 * np.sin(phase))  # left-right sweep
    shoulder = np.array(static[LM_RIGHT_SHOULDER])
    u = np.stack(
        [np.sin(elev) * np.sin(azim), np.cos(elev), np.sin(elev) * np.cos(azim)], axis=1
    )
    wrist = shoulder + 0.46 * u
    alpha = np.deg2rad(30.0)
    d1 = np.cos(alpha) * u + np.sin(alpha) * np.array([0.0, -1.0, 0.0])
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
    elbow = shoulder + 0.28 * d1
    lm[:, LM_RIGHT_ELBOW, :3] = elbow
    lm[:, LM_RIGHT_WRIST, :3] = wrist

    lm[:, :, :3] += rng.normal(0.0, 0.001, size=(n_frames, N_LANDMARKS, 3))
    return lm, float(fps)


# --------------------------------------------------------------------------- #
# Trajectory augmentation
# --------------------------------------------------------------------------- #


def resample_to_length(angles: np.ndarray, n_out: int) -> np.ndarray:
    """Linear interpolation of a (T, J) trajectory to exactly ``n_out`` frames."""
    angles = np.asarray(angles, dtype=float)
    n_in = angles.shape[0]
    n_out = max(2, int(n_out))
    if n_in == n_out:
        return angles.copy()
    idx = np.linspace(0.0, n_in - 1.0, n_out)
    out = np.empty((n_out, angles.shape[1]), dtype=float)
    for j in range(angles.shape[1]):
        out[:, j] = np.interp(idx, np.arange(n_in), angles[:, j])
    return out


def resample_speed(angles: np.ndarray, speed: float) -> np.ndarray:
    """Time-resample a trajectory: speed > 1 plays faster (fewer frames)."""
    n_in = np.asarray(angles).shape[0]
    return resample_to_length(angles, max(2, int(round(n_in / float(speed)))))


def scale_amplitude(angles: np.ndarray, factor: float) -> np.ndarray:
    """Scale angular excursions around the per-joint mean by ``factor``."""
    angles = np.asarray(angles, dtype=float)
    mean = angles.mean(axis=0, keepdims=True)
    return mean + (angles - mean) * float(factor)


def add_angle_noise(angles: np.ndarray, std_deg: float, smooth_window: int, rng: np.random.Generator) -> np.ndarray:
    """Add temporally smoothed Gaussian noise (degrees) to joint angles."""
    noise = rng.normal(0.0, float(std_deg), size=np.asarray(angles).shape)
    return np.asarray(angles, dtype=float) + _moving_average_cols(noise, int(smooth_window))


def generate_variants(base_deg: np.ndarray, cfg: dict, seed: int | None = None) -> list[dict]:
    """Generate ``n_trajectories`` variations of the base trajectory.

    Each variant applies: speed resampling, amplitude scaling around the mean
    pose, and smoothed Gaussian angle noise, then clips to joint limits.
    Returns dicts with id, applied parameters, angles_deg and the fraction of
    samples that exceeded the limits before clipping.
    """
    au = cfg["augmentation"]
    n_variants = int(au["n_trajectories"])
    rng = np.random.default_rng(seed if seed is not None else au.get("seed"))
    sp_lo, sp_hi = (float(x) for x in au["speed_range"])
    amp_lo, amp_hi = (float(x) for x in au["amplitude_range"])
    ns_lo, ns_hi = (float(x) for x in au.get("noise_std_jitter", [1.0, 1.0]))
    base_std = float(au["angle_noise_std_deg"])
    noise_win = int(au.get("noise_smooth_window", 1))
    limits = np.asarray(cfg["robot"]["joint_limits_deg"], dtype=float)
    base_deg = np.asarray(base_deg, dtype=float)

    variants = []
    for i in range(n_variants):
        speed = float(rng.uniform(sp_lo, sp_hi))
        amplitude = float(rng.uniform(amp_lo, amp_hi))
        std = base_std * float(rng.uniform(ns_lo, ns_hi))
        angles = resample_speed(base_deg, speed)
        angles = scale_amplitude(angles, amplitude)
        angles = add_angle_noise(angles, std, noise_win, rng)
        exceeded = float(np.mean(np.any((angles < limits[:, 0]) | (angles > limits[:, 1]), axis=1)))
        angles = np.clip(angles, limits[:, 0], limits[:, 1])
        variants.append(
            {
                "id": i + 1,
                "speed": speed,
                "amplitude": amplitude,
                "noise_std_deg": std,
                "angles_deg": angles,
                "aug_exceeded_frac": exceeded,
            }
        )
    return variants


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def pose_error_deg(traj: np.ndarray, ref: np.ndarray) -> float:
    """Mean absolute deviation from the reference (degrees), over all frames/joints."""
    traj = np.asarray(traj, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if traj.shape != ref.shape:
        raise ValueError(f"shape mismatch: {traj.shape} vs {ref.shape} (resample first)")
    return float(np.abs(traj - ref).mean())


def jerk_stats(angles: np.ndarray, dt: float) -> tuple[float, float]:
    """Total and mean absolute jerk (3rd finite difference) of the angles.

    Returns (sum |d3q/dt3|, mean |d3q/dt3|) in deg/s^3.
    """
    angles = np.asarray(angles, dtype=float)
    if angles.shape[0] < 4:
        return 0.0, 0.0
    d3 = np.diff(angles, n=3, axis=0) / float(dt) ** 3
    return float(np.abs(d3).sum()), float(np.abs(d3).mean())


def limit_violations(angles: np.ndarray, limits_deg: np.ndarray) -> tuple[float, float]:
    """Check joint limits: returns (fraction of frames fully within, max violation deg)."""
    angles = np.asarray(angles, dtype=float)
    limits_deg = np.asarray(limits_deg, dtype=float)
    lo, hi = limits_deg[:, 0], limits_deg[:, 1]
    over = float(np.maximum(angles - hi, 0.0).max())
    under = float(np.maximum(lo - angles, 0.0).max())
    within = np.all((angles >= lo) & (angles <= hi), axis=1)
    return float(within.mean()), max(over, under)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ap = p - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float((ap @ ab) / denom)
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def check_collisions(chain: RobotChain, angles_deg: np.ndarray, cfg: dict) -> int:
    """Count frames where elbow/wrist/tool violate the base-cylinder clearance.

    Simplified self-collision proxy: a cylinder of ``column_radius_m`` around
    the base axis, up to ``column_height_m``; a frame collides when any point
    from the elbow onward comes closer than ``min_clearance_m`` to the
    cylinder surface.
    """
    col = cfg["robot"].get("collision", {})
    if not col.get("enabled", True):
        return 0
    radius = float(col.get("column_radius_m", 0.07))
    height = float(col.get("column_height_m", 0.10))
    min_clear = float(col.get("min_clearance_m", 0.02))
    axis_a = np.zeros(3)
    axis_b = np.array([0.0, 0.0, height])

    n_collisions = 0
    for q in np.deg2rad(np.asarray(angles_deg, dtype=float)):
        points = chain.joint_positions(q)
        clearance = min(
            _point_segment_distance(points[i], axis_a, axis_b) for i in range(2, points.shape[0])
        ) - radius
        if clearance < min_clear:
            n_collisions += 1
    return n_collisions


def evaluate_trajectory(
    chain: RobotChain,
    angles_deg: np.ndarray,
    dt: float,
    cfg: dict,
    ref_deg: np.ndarray | None = None,
) -> dict:
    """Compute all metrics for one trajectory and its ``valid`` flag.

    ``ref_deg`` (optional) enables pose_error: the trajectory is resampled to
    the reference length on a normalized time grid before comparison.
    ``valid`` = within limits AND collisions <= threshold AND pose_error <=
    threshold.
    """
    thresholds = cfg["thresholds"]
    limits = np.asarray(cfg["robot"]["joint_limits_deg"], dtype=float)
    frac_within, max_viol = limit_violations(angles_deg, limits)
    jerk_total, jerk_mean = jerk_stats(angles_deg, dt)
    collisions = check_collisions(chain, angles_deg, cfg)
    if ref_deg is not None:
        aligned = resample_to_length(angles_deg, np.asarray(ref_deg).shape[0])
        err = pose_error_deg(aligned, ref_deg)
    else:
        err = 0.0
    valid = (
        max_viol <= 1e-9
        and collisions <= int(thresholds.get("max_collisions", 0))
        and err <= float(thresholds["pose_error_max_deg"])
    )
    return {
        "pose_error_deg": err,
        "jerk_total": jerk_total,
        "jerk_mean": jerk_mean,
        "frac_within_limits": frac_within,
        "max_limit_violation_deg": max_viol,
        "collisions": collisions,
        "valid": bool(valid),
    }


def _aggregate_metrics(variants: list[dict], cfg: dict) -> dict:
    per = [v["metrics"] for v in variants]
    errors = [m["pose_error_deg"] for m in per]
    jerks = [m["jerk_mean"] for m in per]
    n_valid = sum(1 for m in per if m["valid"])
    return {
        "n_trajectories": len(variants),
        "n_valid": n_valid,
        "valid_rate": (n_valid / len(variants)) if variants else 0.0,
        "pose_error_deg": {"mean": float(np.mean(errors)), "max": float(np.max(errors))},
        "jerk_mean": {"mean": float(np.mean(jerks)), "max": float(np.max(jerks))},
        "collisions_total": int(sum(m["collisions"] for m in per)),
        "pose_error_threshold_deg": float(cfg["thresholds"]["pose_error_max_deg"]),
    }


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def _round_list(arr: np.ndarray, decimals: int) -> list:
    return np.round(np.asarray(arr, dtype=float), int(decimals)).tolist()


def save_trajectories_json(
    path: Path, cfg: dict, fps: float, source: str, rt_info: dict, base_deg: np.ndarray, variants: list[dict]
) -> None:
    nd = int(cfg["output"].get("round_decimals", 4))
    payload = {
        "schema": "pose_to_trajectories/v1",
        "meta": {
            "source": source,
            "fps": float(fps),
            "n_joints": 6,
            "side": rt_info["side"],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": cfg,
        },
        "reference": {
            "id": 0,
            "is_reference": True,
            "n_frames": int(base_deg.shape[0]),
            "timestamps_s": [round(i / fps, 4) for i in range(base_deg.shape[0])],
            "angles_deg": _round_list(base_deg, nd),
        },
        "trajectories": [
            {
                "id": v["id"],
                "is_reference": False,
                "speed": v["speed"],
                "amplitude": v["amplitude"],
                "noise_std_deg": v["noise_std_deg"],
                "n_frames": int(v["angles_deg"].shape[0]),
                "timestamps_s": [round(i / fps, 4) for i in range(v["angles_deg"].shape[0])],
                "angles_deg": _round_list(v["angles_deg"], nd),
                "metrics": v["metrics"],
            }
            for v in variants
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def save_trajectories_csv(path: Path, base_deg: np.ndarray, variants: list[dict], fps: float) -> None:
    header = ["traj_id", "is_reference", "t_sec"] + [f"j{j + 1}_deg" for j in range(6)]
    rows = [(0, 1, base_deg)] + [(v["id"], 0, v["angles_deg"]) for v in variants]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for traj_id, is_ref, angles in rows:
            for i in range(angles.shape[0]):
                writer.writerow(
                    [traj_id, is_ref, f"{i / fps:.4f}"] + [f"{x:.4f}" for x in angles[i]]
                )


def save_metrics_json(
    path: Path, source: str, fps: float, rt_info: dict, ref_metrics: dict, variants: list[dict], aggregate: dict
) -> None:
    payload = {
        "schema": "pose_to_trajectories/metrics/v1",
        "source": source,
        "fps": float(fps),
        "retarget": rt_info,
        "reference": ref_metrics,
        "trajectories": [
            {
                "id": v["id"],
                "speed": v["speed"],
                "amplitude": v["amplitude"],
                "noise_std_deg": v["noise_std_deg"],
                "n_frames": int(v["angles_deg"].shape[0]),
                "aug_exceeded_frac": v["aug_exceeded_frac"],
                **v["metrics"],
            }
            for v in variants
        ],
        "aggregate": aggregate,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run_pipeline(
    cfg: dict,
    video: str | None = None,
    landmarks_json: str | None = None,
    demo: bool = False,
    side: str | None = None,
    out_dir: str | None = None,
) -> dict:
    """Run the full pipeline and write outputs. Returns a summary dict."""
    _validate_config(cfg)
    robot = build_robot(cfg)

    if demo:
        landmarks, fps = make_demo_landmarks()
        source = "demo"
    elif landmarks_json:
        landmarks, fps = load_landmarks_json(landmarks_json)
        source = str(landmarks_json)
    elif video:
        landmarks, fps = extract_pose_landmarks(video, cfg)
        source = str(video)
    else:
        raise ValueError("no input: pass --video, --landmarks or --demo")

    dt = 1.0 / float(fps)
    LOG.info("input: %s | frames=%d fps=%.2f", source, landmarks.shape[0], fps)

    base_deg, rt_info = retarget_landmarks_to_angles(
        landmarks, robot, cfg, side=side or cfg["retarget"].get("side", "auto")
    )
    LOG.info("retarget: %s", rt_info)

    ref_metrics = evaluate_trajectory(robot, base_deg, dt, cfg, ref_deg=base_deg)
    variants = generate_variants(base_deg, cfg)
    for v in variants:
        v["metrics"] = evaluate_trajectory(robot, v["angles_deg"], dt, cfg, ref_deg=base_deg)
    aggregate = _aggregate_metrics(variants, cfg)

    out_path = Path(out_dir or cfg["output"].get("dir", "output"))
    out_path.mkdir(parents=True, exist_ok=True)
    fmt = str(cfg["output"].get("format", "both")).lower()
    written: list[Path] = []
    if fmt in ("json", "both"):
        p = out_path / "trajectories.json"
        save_trajectories_json(p, cfg, fps, source, rt_info, base_deg, variants)
        written.append(p)
    if fmt in ("csv", "both"):
        p = out_path / "trajectories.csv"
        save_trajectories_csv(p, base_deg, variants, fps)
        written.append(p)
    p = out_path / "metrics.json"
    save_metrics_json(p, source, fps, rt_info, ref_metrics, variants, aggregate)
    written.append(p)

    return {
        "source": source,
        "n_frames": int(landmarks.shape[0]),
        "fps": float(fps),
        "retarget": rt_info,
        "aggregate": aggregate,
        "reference_metrics": ref_metrics,
        "outputs": [str(w) for w in written],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="video -> 50 robot trajectories (MediaPipe pose + DLS IK retargeting)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default=None, help="input video (mp4/avi/...)")
    parser.add_argument("--landmarks", default=None, help="pre-extracted landmarks JSON (skips MediaPipe)")
    parser.add_argument("--demo", action="store_true", help="run on synthetic landmarks (no video needed)")
    parser.add_argument("--config", default="config.yaml", help="YAML config path")
    parser.add_argument("--out-dir", default=None, help="override output directory from config")
    parser.add_argument("--side", choices=["auto", "left", "right"], default=None, help="override arm side")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        LOG.error("config: %s", exc)
        return 2
    try:
        summary = run_pipeline(
            cfg,
            video=args.video,
            landmarks_json=args.landmarks,
            demo=args.demo,
            side=args.side,
            out_dir=args.out_dir,
        )
    except (RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 1

    agg = summary["aggregate"]
    rt = summary["retarget"]
    print("== pose_to_trajectories ==")
    print(f"source: {summary['source']}  frames: {summary['n_frames']}  fps: {summary['fps']:.1f}")
    print(
        f"retarget: side={rt['side']}  y_up={rt['y_up']}  "
        f"mean_ik_err={rt['mean_pos_err_m'] * 1000:.2f} mm  held_frames={rt['frames_held']}"
    )
    print(f"trajectories: {agg['n_trajectories']}  valid: {agg['n_valid']} ({agg['valid_rate'] * 100:.1f}%)")
    print(
        f"pose_error deg: mean={agg['pose_error_deg']['mean']:.2f}  max={agg['pose_error_deg']['max']:.2f}  "
        f"(threshold {agg['pose_error_threshold_deg']:.1f})"
    )
    print(
        f"jerk deg/s^3: mean={agg['jerk_mean']['mean']:.1f}  max={agg['jerk_mean']['max']:.1f}  "
        f"collisions: {agg['collisions_total']}"
    )
    print("outputs:")
    for out in summary["outputs"]:
        print(f"  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
