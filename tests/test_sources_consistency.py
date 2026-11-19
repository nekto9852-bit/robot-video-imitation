"""Consistency-тесты единого источника (шаг 1).

MJCF — канон кинематики и position-лимитов; limits.yaml — канон
velocity/acceleration/watchdog. Тесты закрепляют согласованность всех
потребителей: MJCF ↔ limits.yaml ↔ URDF ↔ codec ↔ сгенерённый controllers.yaml
↔ sha256 handshake.
"""

import hashlib
import math
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

import mock_worker as mw
import pose_to_trajectories as p2t

ROOT = Path(__file__).resolve().parents[1]
MJCF_PATH = ROOT / "models" / "arm_6dof.xml"
LIMITS_PATH = ROOT / "config" / "limits.yaml"
URDF_PATH = ROOT / "ros2_ws" / "src" / "arm_bringup" / "urdf" / "arm_6dof.urdf.xacro"
GEN_PATH = ROOT / "ros2_ws" / "src" / "arm_bringup" / "config" / "controllers.generated.yaml"

EXPECTED_JOINTS = ["j1_base_yaw", "j2_shoulder_pitch", "j3_shoulder_roll",
                   "j4_elbow", "j5_wrist_pitch", "j6_wrist_roll"]


def _mjcf_joints():
    """{joint_name: (min_rad, max_rad)} из MJCF (канон position-лимитов)."""
    root = ET.parse(MJCF_PATH).getroot()
    out = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        rng = joint.get("range")
        if name and rng and joint.get("type") != "free":
            lo, hi = (float(x) for x in rng.split())
            out[name] = (lo, hi)
    return out


def _limits_yaml():
    return yaml.safe_load(LIMITS_PATH.read_text(encoding="utf-8"))


def _urdf_joints():
    """Revolute-joints URDF/xacro (limits в рад)."""
    root = ET.parse(URDF_PATH).getroot()
    out = {}
    for joint in root.iter("joint"):
        limit = joint.find("limit")
        if limit is not None:
            out[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return out


def _generated_jtc():
    gen = yaml.safe_load(GEN_PATH.read_text(encoding="utf-8"))
    return gen["joint_trajectory_controller"]["ros__parameters"]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mjcf_positions_equal_limits_yaml():
    """Position-лимиты limits.yaml == MJCF range (канон) — для каждого сустава."""
    mjcf = _mjcf_joints()
    limits = _limits_yaml()["limits"]
    assert set(mjcf) == set(limits)
    for name, (lo, hi) in mjcf.items():
        assert limits[name]["position"] == pytest.approx([lo, hi], abs=1e-4), name


def test_joint_names_consistent_across_sources():
    """Имена суставов: MJCF == URDF == codec == limits.yaml == generated JTC."""
    sys_path = str(ROOT / "ros2_ws" / "src" / "pose_trajectory_bridge")
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from pose_trajectory_bridge.trajectory_codec import JOINT_NAMES as codec_names

    assert list(_mjcf_joints()) == EXPECTED_JOINTS
    assert list(_urdf_joints()) == EXPECTED_JOINTS
    assert codec_names == EXPECTED_JOINTS
    assert list(_limits_yaml()["limits"]) == EXPECTED_JOINTS
    assert _generated_jtc()["joints"] == EXPECTED_JOINTS


def test_generated_jtc_params_equal_limits_yaml():
    """JTC joint_limits (сгенерированные) == velocity/acceleration из limits.yaml
    и position == MJCF (тест согласованности с параметрами JTC)."""
    jtc = _generated_jtc()["joint_limits"]
    mjcf = _mjcf_joints()
    limits = _limits_yaml()["limits"]
    for name, (lo, hi) in mjcf.items():
        jl = jtc[name]
        assert jl["min_position"] == pytest.approx(lo, abs=1e-4)
        assert jl["max_position"] == pytest.approx(hi, abs=1e-4)
        assert jl["max_velocity"] == pytest.approx(limits[name]["velocity_max"])
        assert jl["max_acceleration"] == pytest.approx(limits[name]["acceleration_max"])
        assert jl["has_velocity_limits"] and jl["has_acceleration_limits"]


def test_worker_position_limits_from_mjcf():
    """Воркер берёт position-лимиты из MJCF, vel/acc/watchdog — из limits.yaml."""
    src = mw.load_source_limits()
    mjcf = _mjcf_joints()
    limits = _limits_yaml()
    assert src["joint_names"] == EXPECTED_JOINTS
    for i, name in enumerate(EXPECTED_JOINTS):
        lo, hi = mjcf[name]
        assert src["position_min_rad"][i] == pytest.approx(lo, abs=1e-6)
        assert src["position_max_rad"][i] == pytest.approx(hi, abs=1e-6)
        assert src["velocity_max_rad_s"][i] == pytest.approx(
            limits["limits"][name]["velocity_max"])
        assert src["acceleration_max_rad_s2"][i] == pytest.approx(
            limits["limits"][name]["acceleration_max"])
    assert src["watchdog_timeout_ms"] == limits["watchdog_timeout_ms"]

    cfg = mw._load_pipeline_config()
    expected_deg = [x for lo, hi in mjcf.values() for x in (lo * 180 / math.pi, hi * 180 / math.pi)]
    flat = [x for pair in cfg["robot"]["joint_limits_deg"] for x in pair]
    assert flat == pytest.approx(expected_deg, abs=1e-6)


def test_sha256_handshake_matches_files():
    """compute_source_hashes == независимый пересчёт sha256 обоих файлов."""
    hashes = mw.compute_source_hashes()
    assert hashes["mjcf_sha256"] == _sha256(MJCF_PATH)
    assert hashes["limits_sha256"] == _sha256(LIMITS_PATH)
    assert hashes["mjcf_sha256"] != hashes["limits_sha256"]
    assert mw.compute_source_hashes() == hashes  # стабильно


def test_cmaes_generation_within_vel_acc_limits():
    """CMA-ES генерирует траектории уже внутри velocity/acceleration лимитов
    (limits.yaml, в градусах) — не полагаясь на safety filter."""
    cfg = mw._load_pipeline_config()
    robot = p2t.build_robot(cfg)
    landmarks, _ = p2t.make_demo_landmarks()
    base, _ = p2t.retarget_landmarks_to_angles(landmarks, robot, cfg, side="auto")
    job = "consistency-" + uuid.uuid4().hex[:8]
    res = mw.handle_finetune({
        "job_id": job, "iteration": 1,
        "good_trajectories": [{"trajectory_id": i + 1, "angles_deg": base.tolist()}
                              for i in range(6)],
    })
    vel_max_deg = np.asarray(cfg["source_limits"]["velocity_max_deg_s"])
    acc_max_deg = np.asarray(cfg["source_limits"]["acceleration_max_deg_s2"])
    assert res["trajectories"], "finetune вернул пустой список"
    for traj in res["trajectories"]:
        a = np.asarray(traj["angles_deg"], dtype=float)
        vel = np.abs(np.diff(a, axis=0)) / 30.0
        acc = np.abs(np.diff(vel, axis=0)) / 30.0
        assert np.all(vel.max(axis=0) <= vel_max_deg), "velocity превышен лимит"
        assert np.all(acc.max(axis=0) <= acc_max_deg), "acceleration превышен лимит"
