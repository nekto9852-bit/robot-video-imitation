#!/usr/bin/env python3
"""worker (историческое имя mock_worker.py) — ML-ядро пайплайна обучения.

    GET  /config        -> config.yaml как JSON (единый источник порогов/лимитов)
    POST /process_video -> pose_to_trajectories.py: видео -> 50 траекторий + episodes
    POST /simulate      -> РЕАЛЬНЫЙ MuJoCo (arm_6dof.xml): позиционные сервоприводы
                           120 Гц; pose_error = ошибка слежения под гравитацией,
                           collisions = проникающие контакты, jerk = по фактическим углам
    POST /finetune      -> РЕАЛЬНОЕ дообучение: CMA-ES оптимизирует offset-политику
                           (12-dim: per-joint константы + 6 временных мод), cost
                           считается MuJoCo-симуляцией; policy-состояние живёт в
                           памяти воркера под job_id (сбрасывается при рестарте)
    POST /deploy        -> ЗАГЛУШКА (робота/ROS2 нет): deploy_id + эхо summary —
                           честно помечено в README
    POST /log_iteration -> append в logs/training_iterations.jsonl
    (+ /init_job, /check_episode, /aggregate_iteration, /build_deploy_result,
       /build_fail_result — бывшие Python Code nodes)

Запуск (в venv с mujoco и cma):
    python mock_worker.py                       # порт 8000, config.yaml рядом
    TRAINING_CONFIG=strict.yaml MUJOCO_NAN_RATE=0.1 python mock_worker.py

MUJOCO_NAN_RATE (0..1) — НЕ физика, а test-harness: доля эпизодов с NaN в метриках
для проверки ретраев n8n (по умолчанию 0).
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np
import yaml

import pose_to_trajectories as p2t

CONFIG_PATH = os.environ.get("TRAINING_CONFIG", str(Path(__file__).parent / "config.yaml"))
LOG_PATH = Path(os.environ.get(
    "TRAINING_LOG_PATH", str(Path(__file__).parent / "logs" / "training_iterations.jsonl")))
NAN_RATE = float(os.environ.get("MUJOCO_NAN_RATE", "0"))
PORT = int(os.environ.get("WORKER_PORT", "8000"))
FPS = 30.0
MUJOCO_MODEL_PATH = os.environ.get(
    "MUJOCO_MODEL_PATH", str(Path(__file__).parent / "models" / "arm_6dof.xml"))
LIMITS_YAML_PATH = os.environ.get(
    "LIMITS_YAML_PATH", str(Path(__file__).parent / "config" / "limits.yaml"))
ROS_BRIDGE_URL = os.environ.get("ROS_BRIDGE_URL", "")  # http://127.0.0.1:8082 — опционально
POLICY_DIM = 72        # 6 суставов x 12 мод усечённого Фурье-базиса
JERK_WEIGHT = 2.0      # cost = pose_error + JERK_WEIGHT*(|jerk_mean|/1000) + 100*collisions
POLICY_STATE: dict = {}  # job_id -> {"c": коэффициенты политики (память воркера)}
_MJ = None             # singleton (mujoco module, model)

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

TRAINING = CONFIG.get("training") or {"max_iterations": 5, "max_episode_retries": 2}
THRESHOLDS = CONFIG.get("thresholds") or {"pose_error_max_deg": 8.0, "max_collisions": 0}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_hex(obj) -> str:
    """JSON -> hex без запятых (безопасно для n8n queryReplacement SQL)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8").hex()


def _episode_from_trajectory(job_id, iteration, traj, training_cfg, model_version=None) -> dict:
    """Эпизод для n8n: тот же payload, что строила Code-нода Build Episodes.

    episode_id детерминирован (job, iteration, trajectory) — идемпотентен при ретраях.
    """
    tid = traj.get("trajectory_id")
    return {
        "job_id": job_id,
        "iteration": int(iteration),
        "trajectory_id": tid,
        "episode_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "%s:%s:%s" % (job_id, iteration, tid))),
        "angles_deg": traj.get("angles_deg"),
        "timestamps_s": traj.get("timestamps_s"),
        "n_frames": traj.get("n_frames"),
        "attempt_no": 0,
        "new_iteration": True,
        "thresholds": THRESHOLDS,
        "training_cfg": training_cfg,
        **({"model_version": model_version} if model_version else {}),
    }


def _is_finite(x) -> bool:
    """True для конечных чисел; NaN/inf/не-числа -> False (NaN != NaN)."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    return x == x and x != float("inf") and x != float("-inf")


def _load_pipeline_config() -> dict:
    """config.yaml воркера -> конфиг пайплайна + единый источник лимитов.

    position-лимиты — ИЗ MJCF (канон, mujoco.jnt_range);
    velocity/acceleration/watchdog — ИЗ limits.yaml.
    """
    base = p2t.default_config()
    for section, values in (CONFIG or {}).items():
        if section in base and isinstance(values, dict):
            base[section].update(values)
    src = load_source_limits()
    base["robot"]["joint_limits_deg"] = [
        [lo, hi] for lo, hi in zip(src["position_min_deg"], src["position_max_deg"])]
    base["source_limits"] = src
    return base


def _sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        digest.update(f.read())
    return digest.hexdigest()


def compute_source_hashes() -> dict:
    """sha256 двух файлов единого источника — для handshake worker↔gateway."""
    return {"mjcf_sha256": _sha256_file(MUJOCO_MODEL_PATH),
            "limits_sha256": _sha256_file(LIMITS_YAML_PATH)}


def load_source_limits() -> dict:
    """Единый источник лимитов: MJCF (position, канон) + limits.yaml (vel/acc/watchdog).

    Возвращает значения в двух системах единиц: рад (ROS/limits.yaml) и градусы
    (воркер оперирует траекториями в градусах).
    """
    _, model = _get_mj()
    import mujoco

    joint_names, position_min_rad, position_max_rad = [], [], []
    for i in range(model.njnt):
        joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
        lo, hi = (float(x) for x in model.jnt_range[i])
        position_min_rad.append(lo)
        position_max_rad.append(hi)

    limits_yaml = yaml.safe_load(open(LIMITS_YAML_PATH, encoding="utf-8")) or {}
    watchdog_ms = int(limits_yaml.get("watchdog_timeout_ms", 750))
    yaml_limits = limits_yaml.get("limits") or {}

    deg = 180.0 / math.pi
    velocity_max_rad_s, acceleration_max_rad_s2 = [], []
    for name in joint_names:
        entry = yaml_limits.get(name) or {}
        velocity_max_rad_s.append(float(entry.get("velocity_max", 2.0)))
        acceleration_max_rad_s2.append(float(entry.get("acceleration_max", 25.0)))

    return {
        "joint_names": joint_names,
        "position_min_rad": position_min_rad,
        "position_max_rad": position_max_rad,
        "position_min_deg": [x * deg for x in position_min_rad],
        "position_max_deg": [x * deg for x in position_max_rad],
        "velocity_max_rad_s": velocity_max_rad_s,
        "acceleration_max_rad_s2": acceleration_max_rad_s2,
        "velocity_max_deg_s": [x * deg for x in velocity_max_rad_s],
        "acceleration_max_deg_s2": [x * deg for x in acceleration_max_rad_s2],
        "watchdog_timeout_ms": watchdog_ms,
        "mjcf_sha256": _sha256_file(MUJOCO_MODEL_PATH),
        "limits_sha256": _sha256_file(LIMITS_YAML_PATH),
    }


def _video_from_request(body: dict):
    """video_url ('demo://' или http(s)) или video_base64 -> (landmarks, fps)."""
    video_url = body.get("video_url")
    video_b64 = body.get("video_base64")
    if video_url and str(video_url).startswith("demo://"):
        return p2t.make_demo_landmarks()
    if video_b64:
        import base64

        raw = base64.b64decode(video_b64)
        suffix = Path(str(body.get("video_filename") or "input.mp4")).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            return p2t.extract_pose_landmarks(tmp_path, _load_pipeline_config())
        finally:
            os.unlink(tmp_path)
    if video_url and str(video_url).startswith(("http://", "https://")):
        raw = urllib.request.urlopen(str(video_url), timeout=120).read()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            return p2t.extract_pose_landmarks(tmp_path, _load_pipeline_config())
        finally:
            os.unlink(tmp_path)
    raise ValueError("нужен video_url ('demo://' или http(s)) или video_base64")


def handle_config(body: dict) -> dict:
    return CONFIG


def handle_process_video(body: dict) -> dict:
    job_id = body.get("job_id")
    if not job_id:
        raise ValueError("нет job_id")
    cfg = _load_pipeline_config()
    params = body.get("params") or {}
    if "n_trajectories" in params:
        cfg["augmentation"]["n_trajectories"] = int(params["n_trajectories"])
    if "seed" in params:
        cfg["augmentation"]["seed"] = int(params["seed"])
    if "side" in params:
        cfg["retarget"]["side"] = str(params["side"])

    landmarks, fps = _video_from_request(body)
    robot = p2t.build_robot(cfg)
    base_deg, rt_info = p2t.retarget_landmarks_to_angles(
        landmarks, robot, cfg, side=cfg["retarget"].get("side", "auto")
    )
    variants = p2t.generate_variants(base_deg, cfg)

    trajectories = [
        {
            "trajectory_id": v["id"],
            "n_frames": int(v["angles_deg"].shape[0]),
            "timestamps_s": [round(i / fps, 4) for i in range(v["angles_deg"].shape[0])],
            "angles_deg": p2t._round_list(v["angles_deg"], 4),
            "augmentation": {
                "speed": v["speed"],
                "amplitude": v["amplitude"],
                "noise_std_deg": v["noise_std_deg"],
            },
        }
        for v in variants
    ]
    iteration = int(body.get("iteration") or 1)
    episodes = [_episode_from_trajectory(job_id, iteration, t, TRAINING) for t in trajectories]
    return {
        "job_id": job_id,
        "iteration": iteration,
        "config_used": CONFIG,
        "fps": fps,
        "reference": {
            "n_frames": int(base_deg.shape[0]),
            "angles_deg": p2t._round_list(base_deg, 4),
            "retarget": rt_info,
        },
        "trajectories": trajectories,
        # Эпизоды собраны воркером (бывшая Code-нода Build Episodes): n8n их
        # только разворачивает нодой Split Out — fan-out нельзя сделать HTTP-нодой.
        "episodes": episodes,
        "pipeline_metrics": {
            "n_trajectories": len(trajectories),
            "retarget": rt_info,
        },
    }


def handle_simulate(body: dict) -> dict:
    """РЕАЛЬНЫЙ MuJoCo: прогон командной траектории через позиционные сервоприводы.

    - pose_error_deg — ошибка слежения (гравитационный провис + запаздывание
      приводов) между командой и фактическими углами суставов;
    - collisions — кадры с проникающими контактами (стойка/самоколлизии, real MJCF);
    - jerk — третья производная ФАКТИЧЕСКИХ углов;
    - MUJOCO_NAN_RATE — test-harness: доля эпизодов с NaN в метрике (ретраи n8n).
    """
    ep = body.get("episode") or {}
    angles = ep.get("angles_deg")
    episode_id = str(ep.get("episode_id") or "")
    if not angles:
        raise ValueError("episode.angles_deg отсутствует")
    if len(angles) < 4:
        raise ValueError("episode.angles_deg слишком короткая (нужно >=4 кадров для jerk)")
    if any(not math.isfinite(float(x)) for row in angles for x in row):
        raise ValueError("командные углы содержат NaN/inf")

    attempt = int(ep.get("attempt_no") or 0)
    cfg = _load_pipeline_config()
    metrics, _actual = simulate_trajectory_mujoco(angles, FPS, cfg)

    # Test-harness: детерминированная инжекция NaN (по episode_id+attempt) —
    # только для проверки ретраев n8n; к физике отношения не имеет.
    inj_rng = random.Random(uuid.uuid5(uuid.NAMESPACE_URL, "%s:%s" % (episode_id, attempt)).int)
    if inj_rng.random() < NAN_RATE:
        metrics["pose_error_deg"] = float("nan")

    metrics["simulated_by"] = "mujoco (arm_6dof.xml, position servos @120Hz)"
    return {"episode": ep, "sim_metrics": metrics, "ok": math.isfinite(metrics["pose_error_deg"])}


def _get_mj():
    """Ленивая загрузка MuJoCo-модели (singleton)."""
    global _MJ
    if _MJ is None:
        import mujoco

        _MJ = (mujoco, mujoco.MjModel.from_xml_path(MUJOCO_MODEL_PATH))
    return _MJ


def simulate_trajectory_mujoco(angles_deg, fps: float, cfg: dict):
    """Прогон командной траектории в MuJoCo.

    Робот стартует в позе первого кадра (сервоприводы предварительно выставлены),
    далее 4 подшага симуляции (120 Гц) на каждый кадр 30 fps. Возвращает
    (metrics, actual_deg) — фактические углы суставов по кадрам.
    """
    mujoco, model = _get_mj()
    data = mujoco.MjData(model)
    limits = [(float(lo), float(hi)) for lo, hi in cfg["robot"]["joint_limits_deg"]]
    lo = np.array([l for l, _ in limits])
    hi = np.array([h for _, h in limits])

    cmd = np.clip(np.asarray(angles_deg, dtype=float), lo, hi)  # сервоприводы клиппуют
    n_frames = cmd.shape[0]
    substeps = max(1, int(round((1.0 / fps) / model.opt.timestep)))

    data.qpos[:] = np.deg2rad(cmd[0])
    data.ctrl[:] = np.deg2rad(cmd[0])
    mujoco.mj_forward(model, data)
    actual = np.empty_like(cmd)
    actual[0] = cmd[0]

    collision_frames = 0
    for f in range(1, n_frames):
        data.ctrl[:] = np.deg2rad(cmd[f])
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        for i in range(data.ncon):
            if data.contact[i].dist < 0.0:  # проникающий контакт (стойка/самоколлизия)
                collision_frames += 1
                break
        actual[f] = np.rad2deg(data.qpos[:6])

    diff = np.abs(actual - cmd)
    frac_within, max_viol = p2t.limit_violations(actual, limits)
    jerk_total, jerk_mean = p2t.jerk_stats(actual, 1.0 / fps)
    vel = np.abs(np.diff(actual, axis=0)) / fps if n_frames > 1 else np.zeros((0, actual.shape[1]))
    acc = np.abs(np.diff(vel, axis=0)) / fps if vel.shape[0] > 1 else np.zeros((0, actual.shape[1]))
    metrics = {
        "pose_error_deg": float(diff.mean()),
        "collisions": int(collision_frames),
        "jerk_total": float(jerk_total),
        "jerk_mean": float(jerk_mean),
        "frac_within_limits": float(frac_within),
        "max_limit_violation_deg": float(max_viol),
        "max_velocity_deg_s": float(vel.max()) if vel.size else 0.0,
        "max_acceleration_deg_s2": float(acc.max()) if acc.size else 0.0,
        "steps": int(n_frames),
    }
    return metrics, actual


def _policy_basis(n_frames: int) -> np.ndarray:
    """Per-joint усечённый Фурье-базис (T x 12): [1, t, sin/cos 1..5 гармоник].
    Класс политик = гладкие траектории; усечение отсекает высокочастотную дрожь."""
    t = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    cols = [np.ones(n_frames), t]
    for k in range(1, 6):
        cols.append(np.sin(2 * np.pi * k * t))
        cols.append(np.cos(2 * np.pi * k * t))
    return np.stack(cols, axis=1)  # (T, 12)


def _policy_decode(c: np.ndarray, n_frames: int) -> np.ndarray:
    """Коэффициенты (72,) = 6 суставов x 12 мод -> траектория (n_frames, 6)."""
    basis = _policy_basis(n_frames)
    joints = [basis @ c[j * 12:(j + 1) * 12] for j in range(6)]
    return np.stack(joints, axis=1)


def _policy_project(target: np.ndarray) -> np.ndarray:
    """Проекция траектории (T, 6) на класс политик (least squares, per joint)."""
    basis = _policy_basis(target.shape[0])
    coeffs = []
    for j in range(6):
        coeffs.append(np.linalg.lstsq(basis, target[:, j], rcond=None)[0])
    return np.concatenate(coeffs)  # (72,)


def _policy_cost(c, n_frames, fps, cfg):
    """Cost для CMA-ES — та же MuJoCo-симуляция, что меряет /simulate:
    ошибка слежения + штраф за рывок фактических углов + коллизии.
    Кандидаты, нарушающие velocity/acceleration limits (limits.yaml),
    получают INF — CMA-ES сходится внутрь допустимой области (правка E)."""
    cmd = np.clip(_policy_decode(c, n_frames),
                  [l for l, _ in cfg["robot"]["joint_limits_deg"]],
                  [h for _, h in cfg["robot"]["joint_limits_deg"]])
    src = cfg.get("source_limits") or {}
    vel_max = src.get("velocity_max_deg_s")
    acc_max = src.get("acceleration_max_deg_s2")
    if vel_max:
        vel = np.abs(np.diff(cmd, axis=0)) / fps
        if vel.size and np.any(vel.max(axis=0) > np.asarray(vel_max)):
            return 1e6
    if acc_max:
        vel = np.abs(np.diff(cmd, axis=0)) / fps
        acc = np.abs(np.diff(vel, axis=0)) / fps
        if acc.size and np.any(acc.max(axis=0) > np.asarray(acc_max)):
            return 1e6
    metrics, _ = simulate_trajectory_mujoco(cmd.tolist(), fps, cfg)
    pe = metrics["pose_error_deg"]
    jm = abs(metrics["jerk_mean"])
    cost = pe + JERK_WEIGHT * (jm / 1000.0) + 100.0 * metrics["collisions"]
    if not math.isfinite(cost):
        cost = 1e6
    return cost


def handle_finetune(body: dict) -> dict:
    """РЕАЛЬНОЕ дообучение: политика = траектория в усечённом Фурье-базисе
    (72-dim = 6 суставов x 12 мод); CMA-ES оптимизирует коэффициенты, cost —
    MuJoCo-симуляция (ошибка слежения + jerk + коллизии).

    c0 = проекция среднего успешных траекторий на класс политик — выход
    гарантированно не хуже входа; CMA-ES продолжает с прошлой точки (job_id),
    поэтому итерации воркфлоу монотонно улучшают метрики (cost детерминирован).
    """
    job_id = body.get("job_id")
    iteration = int(body.get("iteration") or 1)
    good = body.get("good_trajectories") or []
    cfg = _load_pipeline_config()
    n_trajs = int(cfg["augmentation"]["n_trajectories"])
    max_it = int(TRAINING.get("max_iterations", 5))
    if iteration + 1 > max_it:
        raise ValueError("iteration %s > max_iterations %s — IF должен был остановить цикл"
                         % (iteration + 1, max_it))

    if good:
        length = max(len(t.get("angles_deg") or []) for t in good)
        base_np = p2t.resample_to_length(good[0]["angles_deg"], length)
        for t in good[1:]:
            base_np = base_np + p2t.resample_to_length(t["angles_deg"], length)
        base_np = base_np / len(good)
    else:
        landmarks, _fps = p2t.make_demo_landmarks()
        robot = p2t.build_robot(cfg)
        base_deg, _ = p2t.retarget_landmarks_to_angles(landmarks, robot, cfg, side="auto")
        base_np = np.asarray(base_deg, dtype=float)

    n_frames = base_np.shape[0]
    c0 = _policy_project(base_np)
    state = POLICY_STATE.setdefault(job_id, {"c": c0})
    state["n_frames"] = n_frames  # для /deploy: сколько кадров у политики
    c_start = state["c"]  # тёплый старт с прошлой итерации (тот же job_id)

    seed = int(uuid.uuid5(uuid.NAMESPACE_URL, "%s:finetune:%s" % (job_id, iteration)).int % (2 ** 31 - 1))
    import cma

    es = cma.CMAEvolutionStrategy(c_start.tolist(), 2.0,
                                  {"maxiter": 8, "popsize": 16, "verbose": -9,
                                   "seed": seed})
    es.optimize(lambda c: _policy_cost(c, n_frames, FPS, cfg))
    candidates = [c0, np.asarray(es.result.xbest, dtype=float)]
    costs = [_policy_cost(c0, n_frames, FPS, cfg), float(es.result.fbest)]
    best_c = candidates[int(np.argmin(costs))]
    best_cost = min(costs)
    state["c"] = best_c  # следующая итерация продолжает с найденной точки

    limits = [(float(lo), float(hi)) for lo, hi in cfg["robot"]["joint_limits_deg"]]
    seed2 = seed + 7
    rng_np = np.random.default_rng(seed2)
    scale = 1.0 / (1.0 + iteration)  # исследовательский шум затухает по итерациям

    trajectories = []
    limits_np = (np.array([l for l, _ in limits]), np.array([h for _, h in limits]))
    src = cfg.get("source_limits") or {}
    vel_max = np.asarray(src.get("velocity_max_deg_s") or [2.0] * 6)
    acc_max = np.asarray(src.get("acceleration_max_deg_s2") or [25.0] * 6)
    policy_only = np.clip(_policy_decode(best_c, n_frames), limits_np[0], limits_np[1])
    for tid in range(1, n_trajs + 1):
        noise = rng_np.normal(0.0, 0.25 * scale, size=base_np.shape)
        noise = p2t._moving_average_cols(noise, 5)  # разнообразие эпизодов без рывков
        angles_np = np.clip(_policy_decode(best_c, n_frames) + noise,
                            limits_np[0], limits_np[1])
        # Генерация уже внутри vel/acc-лимитов (правка E): если исследовательский
        # шум вытолкнул вариант за лимиты — откатываем этот вариант к чистой политике.
        vel = np.abs(np.diff(angles_np, axis=0)) / FPS if n_frames > 1 else np.zeros((1, 6))
        acc = np.abs(np.diff(vel, axis=0)) / FPS if vel.shape[0] > 1 else np.zeros((1, 6))
        if (vel.shape[0] and np.any(vel.max(axis=0) > vel_max)) or \
           (acc.shape[0] and np.any(acc.max(axis=0) > acc_max)):
            angles_np = policy_only
        angles = angles_np.tolist()
        trajectories.append({
            "trajectory_id": tid,
            "n_frames": len(angles),
            "timestamps_s": [round(i / FPS, 4) for i in range(len(angles))],
            "angles_deg": [[round(a, 4) for a in row] for row in angles],
        })
    return {
        "job_id": job_id,
        "iteration": iteration,
        "model_version": "cmaes-iter-%d" % iteration,
        "config_used": CONFIG,
        "trajectories": trajectories,
        # Эпизоды следующей итерации собраны воркером (бывшая Code-нода Prepare Next).
        "episodes": [
            _episode_from_trajectory(job_id, iteration + 1, t, TRAINING, "cmaes-iter-%d" % iteration)
            for t in trajectories
        ],
        "notes": "real finetune: CMA-ES on 72-dim Fourier policy, best cost %.3f" % best_cost,
    }


def handle_init_job(body: dict) -> dict:
    """Бывшая Code-нода Init Job: job_id генерирует воркер.

    Валидация входа (video_url/video_base64) происходит в /process_video —
    воркер не видит тело webhook'а при GET.
    """
    return {"job_id": str(uuid.uuid4()), "iteration": 1, "started_at": _now_iso()}


def handle_check_episode(body: dict) -> dict:
    """Бывшая Code-нода Check Episode Result: пороги, NaN, ретраи — без изменений по смыслу.

    Вход: item ответа /simulate ({episode, sim_metrics} или {error}).
    Выход: episode-поля в top-level + status (passed|failed|failed_retryable),
    attempt_no, reasons[] — супerset контракта {"passed", "retry", "reason"}:
    passed = status=="passed", retry = status=="failed_retryable", reason = reasons[0].
    """
    resp = body if isinstance(body, dict) else {}
    ep = resp.get("episode") or {}
    sim = resp.get("sim_metrics") or {}
    err = resp.get("error")
    reasons = []

    if err:
        status = "failed_retryable"
        msg = err.get("message") if isinstance(err, dict) else str(err)
        reasons.append("воркер /simulate упал: %s" % msg)
        sim = {}
    elif not sim:
        status = "failed_retryable"
        reasons.append("нет sim_metrics в ответе /simulate")
    else:
        th = ep.get("thresholds") or {}
        pe = sim.get("pose_error_deg")
        coll = sim.get("collisions")
        within = sim.get("frac_within_limits", 1.0)
        if not _is_finite(pe):
            # Требование: NaN от MuJoCo -> n8n сам перезапускает эпизод
            status = "failed_retryable"
            reasons.append("NaN/отсутствует pose_error_deg — перезапуск эпизода")
        elif not _is_finite(coll):
            status = "failed_retryable"
            reasons.append("NaN/отсутствует collisions — перезапуск эпизода")
        else:
            pe_max = th.get("pose_error_max_deg", 8.0)
            coll_max = th.get("max_collisions", 0)
            if pe > pe_max:
                reasons.append("pose_error %.2f > %.2f" % (pe, pe_max))
            if coll > coll_max:
                reasons.append("collisions %s > %s" % (coll, coll_max))
            if isinstance(within, (int, float)) and within < 1.0:
                reasons.append("углы вне лимитов (frac_within=%.3f)" % within)
            status = "passed" if not reasons else "failed"

    training_cfg = ep.get("training_cfg") or {}
    max_retries = int(training_cfg.get("max_episode_retries", 2))
    attempt_no = int(ep.get("attempt_no") or 0)
    if status == "failed_retryable":
        attempt_no += 1
        if attempt_no > max_retries:
            status = "failed"
            reasons.append("ретраи эпизода исчерпаны (%s)" % max_retries)

    return {
        "job_id": ep.get("job_id"),
        "iteration": ep.get("iteration"),
        "trajectory_id": ep.get("trajectory_id"),
        "episode_id": ep.get("episode_id"),
        "angles_deg": ep.get("angles_deg"),
        "n_frames": ep.get("n_frames"),
        "attempt_no": attempt_no,
        "new_iteration": False,
        "status": status,
        "sim_metrics": sim,
        "reasons": reasons,
        "thresholds": ep.get("thresholds") or {},
        "training_cfg": training_cfg,
    }


def handle_aggregate_iteration(body: dict) -> dict:
    """Бывшие Code-ноды Aggregate Iteration + Select Failures (объединены)."""
    items = body.get("items") or []
    if not items:
        raise ValueError("нет обработанных эпизодов для агрегации")

    first = items[0] or {}
    statuses = {}
    sum_pe = 0.0
    n_pe = 0
    total_collisions = 0
    episodes = []
    failures = []
    good_trajectories = []
    failed_ids = []
    for j in items:
        j = j or {}
        st = j.get("status") or "unknown"
        statuses[st] = statuses.get(st, 0) + 1
        sim = j.get("sim_metrics") or {}
        pe = sim.get("pose_error_deg")
        coll = sim.get("collisions")
        if isinstance(pe, (int, float)) and not isinstance(pe, bool) and pe == pe:
            sum_pe += pe
            n_pe += 1
        if isinstance(coll, (int, float)) and not isinstance(coll, bool):
            total_collisions += int(coll)
        episode_entry = {
            "episode_id": j.get("episode_id"),
            "trajectory_id": j.get("trajectory_id"),
            "status": st,
            "attempt_no": j.get("attempt_no"),
            "pose_error_deg": pe,
            "collisions": coll,
            "jerk_mean": sim.get("jerk_mean"),
            "frac_within_limits": sim.get("frac_within_limits"),
            "reasons": j.get("reasons") or [],
            "angles_deg": j.get("angles_deg"),
        }
        episodes.append(episode_entry)
        if st == "passed":
            good_trajectories.append({
                "trajectory_id": j.get("trajectory_id"),
                "angles_deg": j.get("angles_deg"),
            })
        else:
            failed_ids.append(j.get("episode_id"))
            failures.append({
                "episode_id": j.get("episode_id"),
                "trajectory_id": j.get("trajectory_id"),
                "status": st,
                "pose_error_deg": pe,
                "collisions": coll,
                "attempt_no": j.get("attempt_no"),
                "reasons": j.get("reasons") or [],
            })

    n_passed = statuses.get("passed", 0)
    return {
        "job_id": first.get("job_id"),
        "iteration": first.get("iteration"),
        "all_passed": len(items) - n_passed == 0,
        "n_passed": n_passed,
        "n_failed": len(items) - n_passed,
        "statuses": statuses,
        "mean_pose_error_deg": (sum_pe / n_pe) if n_pe else None,
        "total_collisions": total_collisions,
        "episodes": episodes,
        "thresholds": first.get("thresholds") or {},
        "training_cfg": first.get("training_cfg") or {},
        # Select Failures: готовый запрос для /finetune (false-ветка IF)
        "finetune_request": {
            "failed_episode_ids": failed_ids,
            "failures": failures,
            "good_trajectories": good_trajectories,
        },
    }


def handle_build_deploy_result(body: dict) -> dict:
    """Бывшая Code-нода Build Deploy Result: текст уведомления + строка журнала."""
    d = body if isinstance(body, dict) else {}
    summary = d.get("summary") or {}
    pe = summary.get("mean_pose_error_deg")
    coll = summary.get("total_collisions")
    pe_txt = ("%.2f" % pe) if isinstance(pe, (int, float)) and not isinstance(pe, bool) else "n/a"
    coll_txt = str(coll) if coll is not None else "n/a"
    text = "Итерация %s: ошибка %s°, столкновений %s. Политика развёрнута на %s (deploy_id %s)" % (
        d.get("iteration"), pe_txt, coll_txt, d.get("robot") or "робот", d.get("deploy_id") or "-",
    )
    log_row = {
        "job_id": d.get("job_id"),
        "iteration": d.get("iteration"),
        "status": "deployed",
        "mean_pose_error_deg": pe if isinstance(pe, (int, float)) and not isinstance(pe, bool) else -1,
        "total_collisions": coll if isinstance(coll, (int, float)) and not isinstance(coll, bool) else -1,
        "n_passed": summary.get("n_passed"),
        "n_failed": summary.get("n_failed"),
        "all_passed": True,
        "details_hex": _json_hex({"deploy_response": d, "completed_at": _now_iso()}),
    }
    return {"notification_text": text, "log_row": log_row}


def handle_build_fail_result(body: dict) -> dict:
    """Бывшая Code-нода Build Fail Result: max_iterations без успеха."""
    agg = body if isinstance(body, dict) else {}
    training_cfg = agg.get("training_cfg") or {}
    pe = agg.get("mean_pose_error_deg")
    coll = agg.get("total_collisions")
    pe_txt = ("%.2f" % pe) if isinstance(pe, (int, float)) and not isinstance(pe, bool) else "n/a"
    coll_txt = str(coll) if coll is not None else "n/a"
    text = "Итерация %s: пороги не достигнуты (ошибка %s°, столкновений %s, прошло %s/%s эпизодов). Останов после max_iterations=%s." % (
        agg.get("iteration"), pe_txt, coll_txt,
        agg.get("n_passed"), (agg.get("n_passed") or 0) + (agg.get("n_failed") or 0),
        training_cfg.get("max_iterations", 5),
    )
    log_row = {
        "job_id": agg.get("job_id"),
        "iteration": agg.get("iteration"),
        "status": "failed_max_iterations",
        "mean_pose_error_deg": pe if isinstance(pe, (int, float)) and not isinstance(pe, bool) else -1,
        "total_collisions": coll if isinstance(coll, (int, float)) and not isinstance(coll, bool) else -1,
        "n_passed": agg.get("n_passed"),
        "n_failed": agg.get("n_failed"),
        "all_passed": False,
        "details_hex": _json_hex({"aggregate": agg, "completed_at": _now_iso()}),
    }
    return {"notification_text": text, "log_row": log_row}


def handle_format_error(body: dict) -> dict:
    """Бывшая Code-нода Format Error: событие Error Trigger -> текст + строка журнала."""
    j = body if isinstance(body, dict) else {}
    exc = j.get("execution") or {}
    err = exc.get("error") or {}
    if isinstance(err, dict):
        msg = err.get("message") or json.dumps(err)[:300]
    else:
        msg = str(err)[:300]
    node = exc.get("lastNodeExecuted") or "?"
    exec_id = exc.get("id") or exc.get("executionId") or "?"
    text = "Ошибка пайплайна обучения: %s (node: %s, execution: %s)" % (msg, node, exec_id)
    log_row = {
        "job_id": "exec-%s" % exec_id,
        "iteration": 0,
        "status": "error",
        "mean_pose_error_deg": -1,
        "total_collisions": -1,
        "n_passed": None,
        "n_failed": None,
        "all_passed": False,
        "details_hex": _json_hex({"error_event": j, "at": _now_iso()}),
    }
    return {"notification_text": text, "log_row": log_row}


def handle_log_iteration(body: dict) -> dict:
    """Журнал итераций: append в logs/training_iterations.jsonl (1 строка = 1 запись).

    Вход — item из Build Deploy/Fail Result или Format Error:
    {notification_text, log_row{...details_hex}}. details_hex декодируется в
    человекочитаемый "details". Выход — {ok, file, status}.
    """
    body = body if isinstance(body, dict) else {}
    row = dict(body["log_row"]) if isinstance(body.get("log_row"), dict) else dict(body)
    details = None
    hexd = row.pop("details_hex", None)
    if isinstance(hexd, str):
        try:
            details = json.loads(bytes.fromhex(hexd).decode("utf-8"))
        except Exception:
            details = {"details_hex_raw": hexd}
    record = {
        "logged_at": _now_iso(),
        "notification_text": body.get("notification_text"),
        **row,
        "details": details,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return {"ok": True, "status": record.get("status"), "file": str(LOG_PATH)}


def handle_deploy(body: dict) -> dict:
    """Деплой: заглушка (deploy_id + эхо). Если задан ROS_BRIDGE_URL — дополнительно
    форвардит ТЕКУЩУЮ политику (декод Фурье-коэффициентов) в ROS 2-мост
    (pose_trajectory_bridge): тот конвертирует в JointTrajectory и шлёт в
    joint_trajectory_controller через safety filter. Ошибка моста не ломает
    деплой-ответ (поле "ros2" честно описывает результат)."""
    result = {
        "deploy_id": str(uuid.uuid4()),
        "status": "deployed",
        "robot": os.environ.get("ROBOT_NAME", "mock-robot-1"),
        "job_id": body.get("job_id"),
        "iteration": body.get("iteration"),
        "summary": body.get("summary") or {},
        "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    bridge_url = (os.environ.get("ROS_BRIDGE_URL") or "").strip()
    if not bridge_url:
        return result  # ROS 2-слой не настроен — поведение как раньше

    # sha256 handshake: gateway сверит со своими копиями единого источника
    hashes = compute_source_hashes()
    result["mjcf_sha256"] = hashes["mjcf_sha256"]
    result["limits_sha256"] = hashes["limits_sha256"]
    payload = {"job_id": result["job_id"], "iteration": result["iteration"],
               "mjcf_sha256": hashes["mjcf_sha256"],
               "limits_sha256": hashes["limits_sha256"]}
    state = POLICY_STATE.get(str(body.get("job_id") or ""))
    if state is not None and "n_frames" in state:
        trajectory = _policy_decode(state["c"], int(state["n_frames"]))
        payload["trajectory"] = {
            "trajectory_id": 0,
            "n_frames": int(state["n_frames"]),
            "timestamps_s": [round(i / FPS, 4) for i in range(int(state["n_frames"]))],
            "angles_deg": [[round(a, 4) for a in row] for row in trajectory.tolist()],
        }
    try:
        req = urllib.request.Request(
            bridge_url.rstrip("/") + "/deploy_trajectory",
            data=json.dumps(payload, default=str).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=150) as resp:
            result["ros2"] = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Деградация моста (например, честный 503 ros_available=false) —
        # тело ответа сохраняем, чтобы n8n/журнал видели причину.
        try:
            result["ros2"] = json.loads(exc.read().decode("utf-8"))
        except Exception:
            result["ros2"] = {"ok": False, "error": "HTTP %s" % exc.code}
    except Exception as exc:  # noqa: BLE001 — мост недоступен: честно в поле ros2
        result["ros2"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    return result


ROUTES = {
    "/config": handle_config,
    "/process_video": handle_process_video,
    "/simulate": handle_simulate,
    "/finetune": handle_finetune,
    "/deploy": handle_deploy,
    # Эндпоинты, заменившие Python Code nodes (вся логика — в воркере)
    "/init_job": handle_init_job,
    "/check_episode": handle_check_episode,
    "/aggregate_iteration": handle_aggregate_iteration,
    "/build_deploy_result": handle_build_deploy_result,
    "/build_fail_result": handle_build_fail_result,
    "/format_error": handle_format_error,
    "/log_iteration": handle_log_iteration,
}
GET_ROUTES = {"/config", "/init_job"}


class Handler(BaseHTTPRequestHandler):
    server_version = "MockTrainingWorker/1.0"

    def _send(self, code: int, payload) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in GET_ROUTES:
            self._send(200, ROUTES[path]({}))
        else:
            self._send(404, {"error": {"message": "unknown path %s (доступно: GET %s, POST %s)"
                                       % (path, ", ".join(sorted(GET_ROUTES)),
                                          ", ".join(k for k in ROUTES if k not in GET_ROUTES))}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        handler = ROUTES.get(path)
        if handler is None:
            self._send(404, {"error": {"message": "unknown path %s" % path}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            result = handler(body if isinstance(body, dict) else {})
            self._send(200, result)
        except Exception as exc:  # noqa: BLE001 — наружу всегда JSON-ошибка
            self._send(500, {"error": {"message": "%s: %s" % (type(exc).__name__, exc)}})

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        print("[worker] %s" % (fmt % args), flush=True)


if __name__ == "__main__":
    print("mock training worker: http://0.0.0.0:%d" % PORT, flush=True)
    print("  config: %s | MUJOCO_NAN_RATE: %s" % (CONFIG_PATH, NAN_RATE), flush=True)
    print("  routes: GET %s | POST %s" % (", ".join(sorted(GET_ROUTES)),
                                          ", ".join(k for k in ROUTES if k not in GET_ROUTES)), flush=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.request_queue_size = 64  # интенсивные прогоны: не отказывать в accept'е
    server.serve_forever()
