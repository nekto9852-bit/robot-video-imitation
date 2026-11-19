"""Тесты pure-кодека JSON -> JointTrajectory payload (без ROS, идёт на любом pytest).

Каноническое расположение: project tests/; ROS-пакет
ros2_ws/src/pose_trajectory_bridge/test/test_codec_ros.py ре-экспортирует их
для `colcon test`.
"""

import math
import sys
from pathlib import Path

import pytest

ROS2_PKG = Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "pose_trajectory_bridge" / "pose_trajectory_bridge"
try:  # installed (colcon) layout
    from pose_trajectory_bridge import trajectory_codec as codec  # noqa: E402
except ImportError:  # source-tree layout: вложенная папка пакета в sys.path
    sys.path.insert(0, str(ROS2_PKG))
    import trajectory_codec as codec  # noqa: E402

JOINTS = codec.JOINT_NAMES


def _payload(angles_deg, fps=30.0):
    n = len(angles_deg)
    return {"trajectories": [{
        "trajectory_id": 1,
        "n_frames": n,
        "timestamps_s": [round(i / fps, 4) for i in range(n)],
        "angles_deg": angles_deg,
    }]}


def test_valid_payload_structure_and_rad_conversion():
    res = codec.json_to_trajectory(_payload([[180.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [90.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert res["joint_names"] == JOINTS
    assert len(res["points"]) == 2
    p0 = res["points"][0]
    assert p0["positions"][0] == pytest.approx(math.pi)   # 180° -> π rad
    assert p0["positions"][1] == pytest.approx(0.0)
    assert res["duration"] == pytest.approx(1 / 30.0, rel=1e-2)  # timestamps округлены до 1e-4


def test_velocities_are_finite_nonzero_for_motion():
    n = 60
    angles = [[10.0 + 20.0 * (i / n), 0, 0, 0, 0, 0] for i in range(n)]
    res = codec.json_to_trajectory(_payload(angles))
    assert len(res["points"]) == n
    for p in res["points"]:
        for v in p["velocities"]:
            assert math.isfinite(v)
    assert max(p["velocities"][0] for p in res["points"]) > 0.15  # 20° за 2с ≈ 0.17 rad/s


def test_nan_angles_rejected():
    with pytest.raises(ValueError, match="NaN/Inf"):
        codec.json_to_trajectory(_payload([[float("nan"), 0, 0, 0, 0, 0]]))


def test_inf_angles_rejected():
    with pytest.raises(ValueError, match="NaN/Inf"):
        codec.json_to_trajectory(_payload([[float("inf"), 0, 0, 0, 0, 0]]))


def test_non_monotonic_timestamps_rejected():
    data = {
        "trajectories": [{
            "trajectory_id": 1,
            "timestamps_s": [0.1, 0.05],
            "angles_deg": [[0.0] * 6, [0.0] * 6],
        }]
    }
    with pytest.raises(ValueError, match="монотонн"):
        codec.json_to_trajectory(data)


def test_joint_count_mismatch_rejected():
    codec.JOINT_NAMES  # sanity
    with pytest.raises(ValueError, match="суставов"):
        codec.json_to_trajectory(_payload([[0.0, 0.0, 0.0]]))  # 3 вместо 6


def test_missing_trajectories_rejected():
    with pytest.raises(ValueError, match="нет траекторий"):
        codec.json_to_trajectory({"summary": {}})


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="разной длины"):
        codec.json_to_trajectory({
            "trajectories": [{"trajectory_id": 1,
                              "timestamps_s": [0.0, 0.1],
                              "angles_deg": [[0.0] * 6]}]})


def test_direct_trajectory_field_supported():
    """Вариант входа без обёртки trajectories: {"trajectory": {...}}."""
    res = codec.json_to_trajectory({"trajectory": {
        "trajectory_id": 7,
        "timestamps_s": [0.0, 0.1],
        "angles_deg": [[0.0] * 6, [math.degrees(0.5)] * 6],
    }})
    assert res["trajectory_id"] == 7
    assert res["points"][1]["positions"][0] == pytest.approx(0.5)


def test_custom_joint_names():
    res = codec.json_to_trajectory(
        _payload([[0.0] * 6]), joint_names=["a", "b", "c", "d", "e", "f"])
    assert res["joint_names"] == ["a", "b", "c", "d", "e", "f"]
