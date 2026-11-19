"""Интеграция worker → ROS 2-bridge.

На Windows (без ROS) проверяется «честная деградация»: worker /deploy ->
HTTP -> bridge отвечает 503 {ros_available: false} + payload_preview (кодек
отработал). В ROS-окружении (WSL2/Docker/Linux) идёт полный путь — см.
ros2_ws/src/pose_trajectory_bridge/test/test_bridge_ros.py.
"""

import copy
import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer

import pytest

import mock_worker as mw
import pose_to_trajectories as p2t

ROS2_PKG = pathlib.Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "pose_trajectory_bridge" / "pose_trajectory_bridge"
try:  # installed (colcon) layout
    from pose_trajectory_bridge import bridge_node  # noqa: E402
except ImportError:  # source-tree layout: вложенная папка пакета в sys.path
    sys.path.insert(0, str(ROS2_PKG))
    import bridge_node  # noqa: E402

JOINT_NAMES = ["j1_base_yaw", "j2_shoulder_pitch", "j3_shoulder_roll",
               "j4_elbow", "j5_wrist_pitch", "j6_wrist_roll"]


@pytest.fixture
def bridge_server():
    core = bridge_node.BridgeCore(ros_adapter=None)  # деградированный режим (нет ROS)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge_node.make_http_handler(core))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:%d" % port
    server.shutdown()


@pytest.fixture
def policy_job():
    """Job с готовой политикой: finetune создаёт POLICY_STATE[job]."""
    cfg = mw._load_pipeline_config()
    robot = p2t.build_robot(cfg)
    landmarks, _ = p2t.make_demo_landmarks()
    base, _ = p2t.retarget_landmarks_to_angles(landmarks, robot, cfg, side="auto")
    job = "ros-int-" + uuid.uuid4().hex[:8]
    mw.handle_finetune({
        "job_id": job, "iteration": 1,
        "good_trajectories": [{"trajectory_id": i + 1, "angles_deg": base.tolist()}
                              for i in range(6)],
    })
    return job


def test_deploy_forwards_policy_to_bridge_degraded(bridge_server, policy_job, monkeypatch):
    monkeypatch.setenv("ROS_BRIDGE_URL", bridge_server)
    res = mw.handle_deploy({"job_id": policy_job, "iteration": 2, "summary": {"n_passed": 50}})
    assert res["deploy_id"] and res["status"] == "deployed"  # деплой не ломается
    assert "ros2" in res
    ros = res["ros2"]
    assert ros["ok"] is False and ros["ros_available"] is False  # честная деградация
    preview = ros["payload_preview"]
    assert preview["joint_names"][0] == "j1_base_yaw"
    assert preview["points"] == 90  # n_frames политики (демо: 90 кадров)
    assert preview["duration"] == pytest.approx(89 / 30.0, abs=1e-3)


def test_deploy_without_ros_bridge_env_is_backward_compatible(policy_job, monkeypatch):
    monkeypatch.delenv("ROS_BRIDGE_URL", raising=False)
    res = mw.handle_deploy({"job_id": policy_job, "iteration": 1, "summary": {}})
    assert "ros2" not in res  # прежнее поведение без ROS_BRIDGE_URL


def test_bridge_rejects_invalid_payload(bridge_server):
    body = json.dumps({"trajectories": []}).encode()
    req = urllib.request.Request(
        bridge_server + "/deploy_trajectory", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("ожидали 400")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400
        assert "нет траекторий" in json.loads(exc.read().decode())["error"]


def test_bridge_unknown_path_404(bridge_server):
    req = urllib.request.Request(bridge_server + "/unknown", data=b"{}", method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("ожидали 404")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


def test_bridge_status_get(bridge_server):
    with urllib.request.urlopen(bridge_server, timeout=10) as resp:
        status = json.loads(resp.read().decode())
    assert status["status"]["ros_available"] is False  # деградированный режим
    assert status["joint_names"][0] == "j1_base_yaw"


def test_full_cycle_worker_finetune_deploy_degraded(bridge_server, monkeypatch):
    """Сквозной (в деградации): finetune → deploy → bridge видит политику."""
    monkeypatch.setenv("ROS_BRIDGE_URL", bridge_server)
    cfg = mw._load_pipeline_config()
    robot = p2t.build_robot(cfg)
    landmarks, _ = p2t.make_demo_landmarks()
    base, _ = p2t.retarget_landmarks_to_angles(landmarks, robot, cfg, side="auto")
    job = "ros-full-" + uuid.uuid4().hex[:8]
    good = [{"trajectory_id": i + 1, "angles_deg": base.tolist()} for i in range(6)]
    ft = mw.handle_finetune({"job_id": job, "iteration": 1,
                             "good_trajectories": copy.deepcopy(good)})
    assert len(ft["episodes"]) == 50
    res = mw.handle_deploy({"job_id": job, "iteration": 2, "summary": {}})
    assert res["ros2"]["ros_available"] is False
    assert res["ros2"]["payload_preview"]["joint_names"] == JOINT_NAMES
    assert res["ros2"]["payload_preview"]["points"] == 90
