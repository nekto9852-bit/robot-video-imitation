"""Тесты реального дообучения (CMA-ES над Фурье-политикой, cost = MuJoCo)."""

import uuid

import numpy as np
import pytest

import mock_worker as mw
import pose_to_trajectories as p2t


@pytest.fixture(scope="module")
def cfg():
    return mw._load_pipeline_config()


@pytest.fixture(scope="module")
def good_trajectories(cfg):
    robot = p2t.build_robot(cfg)
    landmarks, _ = p2t.make_demo_landmarks()
    base, _ = p2t.retarget_landmarks_to_angles(landmarks, robot, cfg, side="auto")
    return [{"trajectory_id": i + 1, "angles_deg": base.tolist()} for i in range(8)]


def _tracked_cost(cfg, angles_deg):
    """Метрики через ту же MuJoCo-симуляцию, что /simulate."""
    m, _ = mw.simulate_trajectory_mujoco(angles_deg, 30.0, cfg)
    return m


def test_finetune_returns_contract(cfg, good_trajectories):
    job = "ft-contract-" + uuid.uuid4().hex[:8]
    res = mw.handle_finetune({"job_id": job, "iteration": 1,
                              "good_trajectories": good_trajectories})
    n = cfg["augmentation"]["n_trajectories"]
    assert len(res["trajectories"]) == n == len(res["episodes"])
    assert res["model_version"].startswith("cmaes-iter-")
    limits = np.asarray(cfg["robot"]["joint_limits_deg"], dtype=float)
    for ep, tr in zip(res["episodes"], res["trajectories"]):
        assert ep["trajectory_id"] == tr["trajectory_id"]
        assert ep["iteration"] == 2 and ep["new_iteration"] is True
        a = np.asarray(tr["angles_deg"], dtype=float)
        assert np.all(a >= limits[:, 0]) and np.all(a <= limits[:, 1])
    # детерминированные episode_id: тот же (job, iteration) -> те же id
    res2 = mw.handle_finetune({"job_id": job, "iteration": 1,
                               "good_trajectories": good_trajectories})
    assert [e["episode_id"] for e in res2["episodes"]] == \
           [e["episode_id"] for e in res["episodes"]]


def test_finetune_smooths_jerk_real_physics(cfg, good_trajectories):
    """Дообучение реально снижает рывок фактических углов (MuJoCo): >= 30%."""
    job = "ft-jerk-" + uuid.uuid4().hex[:8]
    raw = mw.simulate_trajectory_mujoco(good_trajectories[0]["angles_deg"], 30.0, cfg)[0]
    res = mw.handle_finetune({"job_id": job, "iteration": 1,
                              "good_trajectories": good_trajectories})
    tuned = mw.simulate_trajectory_mujoco(res["trajectories"][0]["angles_deg"], 30.0, cfg)[0]
    assert tuned["jerk_mean"] < 0.7 * raw["jerk_mean"], \
        "jerk не снизился: raw %.0f -> tuned %.0f" % (raw["jerk_mean"], tuned["jerk_mean"])
    assert tuned["collisions"] == 0


def test_finetune_improves_across_iterations(cfg, good_trajectories):
    """cost (pe + jerk-штраф) убывает по итерациям: iter3 < iter2 < iter1."""
    job = "ft-conv-" + uuid.uuid4().hex[:8]
    costs = []
    for iteration in (1, 2, 3):
        res = mw.handle_finetune({"job_id": job, "iteration": iteration,
                                  "good_trajectories": good_trajectories})
        m = _tracked_cost(cfg, res["trajectories"][0]["angles_deg"])
        costs.append(m["pose_error_deg"] + mw.JERK_WEIGHT * abs(m["jerk_mean"]) / 1000.0)
    assert costs[2] < costs[0], "cost не улучшился: %s" % [round(c, 3) for c in costs]
    assert costs[1] <= costs[0] * 1.02  # не деградирует между итерациями


def test_finetune_policy_persists_per_job(cfg, good_trajectories):
    job = "ft-state-" + uuid.uuid4().hex[:8]
    assert job not in mw.POLICY_STATE
    mw.handle_finetune({"job_id": job, "iteration": 1, "good_trajectories": good_trajectories})
    assert job in mw.POLICY_STATE
    c = mw.POLICY_STATE[job]["c"]
    assert c.shape == (mw.POLICY_DIM,) and np.any(c != 0.0)


def test_finetune_respects_max_iterations(cfg, good_trajectories):
    job = "ft-max-" + uuid.uuid4().hex[:8]
    max_it = int(mw.TRAINING.get("max_iterations", 5))
    with pytest.raises(ValueError):
        mw.handle_finetune({"job_id": job, "iteration": max_it,
                            "good_trajectories": good_trajectories})
