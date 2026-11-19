"""Unit tests for trajectory metrics and augmentation."""

import numpy as np
import pytest

from pose_to_trajectories import (
    build_robot,
    check_collisions,
    default_config,
    evaluate_trajectory,
    generate_variants,
    jerk_stats,
    limit_violations,
    pose_error_deg,
    resample_speed,
)


@pytest.fixture
def cfg():
    return default_config()


def test_pose_error_zero_for_identical_trajectories():
    a = np.arange(24, dtype=float).reshape(4, 6)
    assert pose_error_deg(a, a.copy()) == pytest.approx(0.0)


def test_pose_error_known_value():
    a = np.zeros((10, 6))
    b = np.full((10, 6), 2.0)
    assert pose_error_deg(a, b) == pytest.approx(2.0)


def test_jerk_zero_for_constant_trajectory():
    a = np.full((50, 6), 15.0)
    total, mean = jerk_stats(a, dt=1.0 / 30.0)
    assert total == 0.0
    assert mean == 0.0


def test_jerk_grows_with_motion_frequency():
    """Analytic jerk of A*sin(w*t) scales as w^3: fast sine must dominate."""
    t = np.linspace(0.0, 2.0 * np.pi, 240)
    dt = t[1] - t[0]
    slow = np.stack([5.0 * np.sin(t)] * 6, axis=1)
    fast = np.stack([5.0 * np.sin(4.0 * t)] * 6, axis=1)
    _, mean_slow = jerk_stats(slow, dt)
    _, mean_fast = jerk_stats(fast, dt)
    assert mean_fast > 8.0 * mean_slow


def test_limit_violations_detect_out_of_range():
    limits = np.tile([-10.0, 10.0], (6, 1))
    good = np.zeros((5, 6))
    frac, viol = limit_violations(good, limits)
    assert frac == pytest.approx(1.0)
    assert viol == pytest.approx(0.0)

    bad = good.copy()
    bad[2, 3] = 12.0  # 2 degrees past the limit on one frame
    frac, viol = limit_violations(bad, limits)
    assert frac == pytest.approx(4.0 / 5.0)
    assert viol == pytest.approx(2.0)


def test_check_collisions_counts_folded_poses():
    cfg = default_config()
    robot = build_robot(cfg)
    straight_up = np.zeros((1, 6))  # arm vertical: elbow far above the base cylinder
    assert check_collisions(robot, straight_up, cfg) == 0
    # Fold the arm down next to the base column -> collision expected.
    folded = np.array([[0.0, 172.0, 0.0, -172.0, 0.0, 0.0]])
    assert check_collisions(robot, folded, cfg) == 1


def test_generate_variants_count_reproducible_and_in_limits():
    cfg = default_config()
    base = np.stack([np.linspace(0.0, 40.0, 100) + 5.0 * j for j in range(6)], axis=1)
    v1 = generate_variants(base, cfg, seed=123)
    v2 = generate_variants(base, cfg, seed=123)
    v3 = generate_variants(base, cfg, seed=124)

    assert len(v1) == cfg["augmentation"]["n_trajectories"] == 50
    assert all(np.allclose(a["angles_deg"], b["angles_deg"]) for a, b in zip(v1, v2))
    # Different seeds may resample to different lengths -> compare shape first.
    assert any(
        a["angles_deg"].shape != b["angles_deg"].shape
        or not np.allclose(a["angles_deg"], b["angles_deg"])
        for a, b in zip(v1, v3)
    )

    limits = np.asarray(cfg["robot"]["joint_limits_deg"], dtype=float)
    for v in v1:
        assert np.all(v["angles_deg"] >= limits[:, 0])
        assert np.all(v["angles_deg"] <= limits[:, 1])
        assert 0.7 <= v["speed"] <= 1.3
        assert 0.9 <= v["amplitude"] <= 1.1


def test_resample_speed_changes_length():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(100, 6))
    fast = resample_speed(a, 1.3)
    slow = resample_speed(a, 0.7)
    assert fast.shape[0] == round(100 / 1.3)  # faster playback -> fewer frames
    assert slow.shape[0] == round(100 / 0.7)  # slower playback -> more frames
    assert np.allclose(fast[0], a[0]) and np.allclose(fast[-1], a[-1])


def test_evaluate_trajectory_flags_invalid(cfg):
    robot = build_robot(cfg)
    base = np.zeros((60, 6))

    ok = evaluate_trajectory(robot, base, 1.0 / 30.0, cfg, ref_deg=base)
    assert ok["valid"]
    assert ok["pose_error_deg"] == pytest.approx(0.0)
    assert ok["frac_within_limits"] == pytest.approx(1.0)
    assert ok["collisions"] == 0

    bad = base + 90.0  # huge deviation -> pose_error above the threshold
    res = evaluate_trajectory(robot, bad, 1.0 / 30.0, cfg, ref_deg=base)
    assert not res["valid"]
    assert res["pose_error_deg"] == pytest.approx(90.0)
