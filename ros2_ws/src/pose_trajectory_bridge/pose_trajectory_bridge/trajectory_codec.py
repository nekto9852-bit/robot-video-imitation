"""Pure JSON -> JointTrajectory-payload codec (без ROS-зависимостей).

Этот модуль намеренно не импортирует rclpy/trajectory_msgs: он тестируется
pytest'ом на любой машине (в том числе Windows), а bridge_node оборачивает его
выход в trajectory_msgs/JointTrajectory.

Вход — JSON воркера:
  * ответ /finetune или /process_video: {"trajectories": [...], "episodes": [...]}
  * или уже выбранный {"trajectory": {...}}
Траектория: {"trajectory_id", "n_frames", "timestamps_s", "angles_deg"}.
Углы в ГРАДУСАХ (как весь пайплайн) — конвертируются в радианы.

Выход (payload для FollowJointTrajectory):
  {"joint_names": [...], "points": [{"positions": [rad], "velocities": [rad/s],
    "time_from_start": seconds}], "duration": seconds}
"""

from __future__ import annotations

import math

# Имена суставов ROS — совпадают с MJCF arm_6dof.xml и controllers.yaml.
JOINT_NAMES = [
    "j1_base_yaw",
    "j2_shoulder_pitch",
    "j3_shoulder_roll",
    "j4_elbow",
    "j5_wrist_pitch",
    "j6_wrist_roll",
]


def _select_trajectory(data: dict) -> dict:
    """Выбираетdeploy-траекторию из JSON воркера: trajectory > trajectories[0]."""
    if not isinstance(data, dict):
        raise ValueError("payload должен быть JSON-объектом")
    traj = data.get("trajectory")
    if traj is None:
        trajs = data.get("trajectories")
        if not isinstance(trajs, list) or not trajs:
            raise ValueError("нет траекторий в payload (ожидались trajectories)")
        traj = trajs[0]
    if not isinstance(traj, dict):
        raise ValueError("траектория должна быть JSON-объектом")
    for key in ("angles_deg", "timestamps_s"):
        if not isinstance(traj.get(key), list):
            raise ValueError("trajectories[0].%s отсутствует или не массив" % key)
    angles = traj["angles_deg"]
    stamps = traj["timestamps_s"]
    if not angles or not stamps:
        raise ValueError("пустая траектория")
    if len(angles) != len(stamps):
        raise ValueError("angles_deg и timestamps_s разной длины")
    return traj


def json_to_trajectory(data: dict, joint_names: list[str] | None = None,
                       velocity_window: int = 3) -> dict:
    """JSON воркера -> payload JointTrajectory (позиции в радианах, скорости rad/s).

    Скорости — центральная конечная разность по позициям (сглаженная окном
    velocity_window, чтобы не транслировать шум дискретизации в контроллер).
    Бросает ValueError на NaN/Inf — до ROS-слоя доходит только валидный payload
    (C++ safety filter перепроверяет independently).
    """
    names = list(joint_names) if joint_names else list(JOINT_NAMES)
    traj = _select_trajectory(data)
    angles_deg = traj["angles_deg"]
    stamps = [float(s) for s in traj["timestamps_s"]]

    n_frames = len(angles_deg)
    n_joints = len(angles_deg[0])
    if n_joints != len(names):
        raise ValueError("число суставов в траектории (%d) != %d" % (n_joints, len(names)))

    angles = []
    for row in angles_deg:
        row_r = []
        for x in row:
            x = float(x)
            if not math.isfinite(x):
                raise ValueError("углы содержат NaN/Inf")
            row_r.append(math.radians(x))
        angles.append(row_r)

    times = []
    prev = -1.0
    for s in stamps:
        if not math.isfinite(s) or s < prev:
            raise ValueError("timestamps_s не монотонно/содержит NaN/Inf")
        prev = s
        times.append(s)

    points = []
    for i in range(n_frames):
        vel = [0.0] * n_joints
        if i == 0:
            denom = max(times[1] - times[0], 1e-6) if n_frames > 1 else 1.0
            vel = [(angles[1][j] - angles[0][j]) / denom for j in range(n_joints)] \
                if n_frames > 1 else vel
        elif i == n_frames - 1:
            denom = max(times[i] - times[i - 1], 1e-6)
            vel = [(angles[i][j] - angles[i - 1][j]) / denom for j in range(n_joints)]
        else:
            lo = max(0, i - velocity_window // 2)
            hi = min(n_frames - 1, i + velocity_window // 2)
            denom = max(times[hi] - times[lo], 1e-6)
            vel = [(angles[hi][j] - angles[lo][j]) / denom for j in range(n_joints)]
        if not all(math.isfinite(v) for v in vel):
            raise ValueError("вычисленные скорости содержат NaN/Inf")
        points.append({
            "positions": angles[i],
            "velocities": vel,
            "time_from_start": times[i],
        })

    return {
        "joint_names": names,
        "points": points,
        "duration": times[-1],
        "trajectory_id": traj.get("trajectory_id"),
    }
