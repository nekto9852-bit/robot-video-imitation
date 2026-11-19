"""Тесты реальной MuJoCo-симуляции (/simulate): физика, лимиты, коллизии, детерминизм."""

import numpy as np
import pytest

import mock_worker as mw
import pose_to_trajectories as p2t


@pytest.fixture(scope="module")
def cfg():
    return mw._load_pipeline_config()


@pytest.fixture(scope="module")
def demo_base(cfg):
    robot = p2t.build_robot(cfg)
    landmarks, _ = p2t.make_demo_landmarks()
    base, _ = p2t.retarget_landmarks_to_angles(landmarks, robot, cfg, side="auto")
    return base.tolist()


def test_mj_model_matches_robot_chain_fk(cfg):
    """MJCF-модель кинематически идентична аналитическому RobotChain FK."""
    mujoco, model = mw._get_mj()
    assert model.nq == 6 and model.nu == 6
    data = mujoco.MjData(model)
    robot = p2t.build_robot(cfg)
    for q in [np.zeros(6),
              np.array([0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0]),
              np.deg2rad([30.0, -40.0, 20.0, 60.0, -30.0, 10.0])]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        assert np.allclose(data.site("site_tool").xpos, robot.fk(q), atol=1e-9)


def test_mj_joint_limits_match_config(cfg):
    _, model = mw._get_mj()
    limits_deg = np.asarray(cfg["robot"]["joint_limits_deg"], dtype=float)
    for i, name in enumerate(["j1_base_yaw", "j2_shoulder_pitch", "j3_shoulder_roll",
                              "j4_elbow", "j5_wrist_pitch", "j6_wrist_roll"]):
        j = model.joint(name)
        assert np.allclose(j.range, np.deg2rad(limits_deg[i]), atol=1e-4)


def test_simulate_deterministic(cfg, demo_base):
    m1, _ = mw.simulate_trajectory_mujoco(demo_base, 30.0, cfg)
    m2, _ = mw.simulate_trajectory_mujoco(demo_base, 30.0, cfg)
    assert m1 == m2  # фиксированное начальное состояние, без случайностей


def test_simulate_real_tracking_error(cfg, demo_base):
    """Реальная физика: ненулевая ошибка слежения (провис/лаг), контактов нет,
    фактические углы в лимитах. Сенситивность: +5° pitch плеча меняет
    гравитационный провис -> другая ошибка слежения (yaw-инвариантности нет)."""
    m1, actual = mw.simulate_trajectory_mujoco(demo_base, 30.0, cfg)
    assert 0.05 < m1["pose_error_deg"] < 8.0  # не ноль (гравитация) и не шум
    assert m1["collisions"] == 0
    assert m1["frac_within_limits"] == 1.0
    assert np.all(np.isfinite(actual))
    other = [[a + (5.0 if j == 1 else 0.0) for j, a in enumerate(row)] for row in demo_base]
    m2, _ = mw.simulate_trajectory_mujoco(other, 30.0, cfg)
    assert abs(m2["pose_error_deg"] - m1["pose_error_deg"]) > 1e-6  # реагирует на вход


def test_simulate_servos_enforce_joint_limits(cfg):
    """Команда за лимитами (j1 = 500°, рука вертикальна — без контактов):
    ctrlrange сервопривода клиппует — фактический угол у лимита."""
    cmd = [[500.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 30
    m, actual = mw.simulate_trajectory_mujoco(cmd, 30.0, cfg)
    assert m["frac_within_limits"] == 1.0
    assert m["max_limit_violation_deg"] == 0.0
    assert np.abs(actual[:, 0] - 170.0).max() < 2.0  # факт у лимита, без контактов


def test_simulate_folded_pose_collides_with_pedestal(cfg):
    """Сложенная вниз рука бьётся о корпус-стойку: реальные penetrating контакты."""
    folded = [[0.0, 172.0, 0.0, -172.0, 0.0, 0.0]] * 30
    m, _ = mw.simulate_trajectory_mujoco(folded, 30.0, cfg)
    assert m["collisions"] >= 10


def test_simulate_jerk_reflects_motion(cfg):
    """jerk фактических углов: постоянная поза (после установки) ~0; движение >> 0."""
    constant = [[10.0, 45.0, 0.0, -30.0, 0.0, 0.0]] * 40
    m_const, _ = mw.simulate_trajectory_mujoco(constant, 30.0, cfg)
    assert abs(m_const["jerk_mean"]) < 100.0
    t = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    sine = np.stack([10.0 + 20.0 * np.sin(t)] * 6, axis=1).tolist()
    m_sine, _ = mw.simulate_trajectory_mujoco(sine, 30.0, cfg)
    assert m_sine["jerk_mean"] > 5.0 * max(abs(m_const["jerk_mean"]), 1e-9)


def test_simulate_nan_injection_knob(cfg, demo_base, monkeypatch):
    """MUJOCO_NAN_RATE — test-harness для ретраев n8n (не физика)."""
    monkeypatch.setattr(mw, "NAN_RATE", 1.0)
    ep = {"episode_id": "nan-test:0", "attempt_no": 0, "angles_deg": demo_base[:8]}
    res = mw.handle_simulate({"episode": ep})
    assert res["ok"] is False
    assert not np.isfinite(res["sim_metrics"]["pose_error_deg"])
    monkeypatch.setattr(mw, "NAN_RATE", 0.0)
    res = mw.handle_simulate({"episode": ep})
    assert res["ok"] is True


def test_simulate_rejects_bad_input(cfg):
    with pytest.raises(ValueError):
        mw.handle_simulate({"episode": {"angles_deg": []}})
    with pytest.raises(ValueError):
        mw.handle_simulate({"episode": {"angles_deg": [[float("nan")] * 6] * 10}})
