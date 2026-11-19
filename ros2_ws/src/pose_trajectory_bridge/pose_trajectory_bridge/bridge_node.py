"""Bridge node: HTTP (JSON от воркера) -> FollowJointTrajectory action (JTC).

Архитектура:
  n8n /deploy -> mock_worker /deploy -> HTTP POST /deploy_trajectory (здесь)
  -> trajectory_codec (pure) -> trajectory_msgs/JointTrajectory
  -> safety filter (topic /safety/input_joint_trajectory) -> JTC action.

Без rclpy (например, Windows без ROS) HTTP-сервер стартует в деградированном
режиме и честно отвечает 503 {"ros_available": false} — интеграцию worker->bridge
можно проверить без ROS-окружения.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # установлен как пакет (colcon/pip)
    from .trajectory_codec import JOINT_NAMES, json_to_trajectory
except ImportError:  # запущен как скрипт/из исходного дерева
    from trajectory_codec import JOINT_NAMES, json_to_trajectory

ACTION_NAME = "/arm_controller/follow_joint_trajectory"
SAFETY_TOPIC = "/safety/input_joint_trajectory"


class BridgeCore:
    """ROS-независимое состояние моста + HTTP-логика."""

    def __init__(self, ros_adapter=None):
        self.ros_adapter = ros_adapter  # публикует/шлёт цели в ROS; None = деградация
        self.last_status: dict = {"ros_available": ros_adapter is not None,
                                  "goals_accepted": 0, "goals_rejected": 0}

    def handle_deploy_trajectory(self, body: dict) -> tuple[int, dict]:
        try:
            payload = json_to_trajectory(body, joint_names=JOINT_NAMES)
        except ValueError as exc:
            return 400, {"ok": False, "error": str(exc)}
        if self.ros_adapter is None:
            self.last_status["goals_rejected"] += 1
            return 503, {
                "ok": False,
                "ros_available": False,
                "error": "ROS 2 недоступен: bridge запущен без rclpy "
                         "(запустите в WSL2/Docker/Linux с ROS 2 Jazzy)",
                "payload_preview": {
                    "joint_names": payload["joint_names"],
                    "points": len(payload["points"]),
                    "duration": payload["duration"],
                },
            }
        result = self.ros_adapter.execute(payload)
        if result.get("accepted"):
            self.last_status["goals_accepted"] += 1
        else:
            self.last_status["goals_rejected"] += 1
        return 200, {"ok": bool(result.get("accepted")), "ros_available": True, **result}


class RosAdapter:
    """rclpy-обвязка: publisher в safety filter + FollowJointTrajectory action client.

    Вынесена из BridgeCore, чтобы ядро оставалось тестируемым без ROS.
    """

    def __init__(self, node):
        self.node = node
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        self._jtm = JointTrajectory
        self._jtp = JointTrajectoryPoint
        self.pub = node.create_publisher(JointTrajectory, SAFETY_TOPIC, 10)
        from rclpy.action import ActionClient

        from control_msgs.action import FollowJointTrajectory

        self.action_client = ActionClient(node, FollowJointTrajectory, ACTION_NAME)

    def execute(self, payload: dict) -> dict:
        import time as _time

        from builtin_interfaces.msg import Duration

        from control_msgs.action import FollowJointTrajectory

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = payload["joint_names"]
        for point in payload["points"]:
            p = self._jtp()
            p.positions = [float(x) for x in point["positions"]]
            p.velocities = [float(x) for x in point["velocities"]]
            sec = int(point["time_from_start"])
            p.time_from_start = Duration(sec=sec, nanosec=int(round((point["time_from_start"] - sec) * 1e9)))
            goal.trajectory.points.append(p)

        if not self.action_client.wait_for_server(timeout_sec=5.0):
            return {"accepted": False, "error": "action server %s недоступен" % ACTION_NAME}

        send_future = self.action_client.send_goal_async(goal)
        rclpy = self.node._rclpy_mod  # noqa: SLF001 — прокинуто из main
        goal_handle = rclpy.wait_for_future_or_timeout(self.node, send_future, 10.0)
        if goal_handle is None or not goal_handle.accepted:
            return {"accepted": False, "error": "goal отклонён контроллером"}

        result_future = goal_handle.get_result_async()
        result = rclpy.wait_for_future_or_timeout(self.node, result_future, payload["duration"] + 30.0)
        if result is None:
            return {"accepted": True, "error": "таймаут ожидания результата (исполнение продолжается)"}
        status = result.status
        return {"accepted": goal_handle.accepted, "result_status": int(status),
                "error_code": int(result.result.error_code)}


def wait_for_future_or_timeout(node, future, timeout_s):
    """Помощник: крутит executor до готовности future или таймаута."""
    import time

    start = time.monotonic()
    while not future.done() and time.monotonic() - start < timeout_s:
        rclpy_spin_once(node, timeout_sec=0.05)
    return future.result() if future.done() else None


def rclpy_spin_once(node, timeout_sec):
    import rclpy

    rclpy.spin_once(node, timeout_sec=timeout_sec)


def make_http_handler(core: BridgeCore):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            if self.path.split("?")[0] != "/deploy_trajectory":
                self._send(404, {"ok": False, "error": "unknown path %s" % self.path})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"ok": False, "error": "bad json: %s" % exc})
                return
            code, payload = core.handle_deploy_trajectory(body)
            self._send(code, payload)

        def do_GET(self):  # noqa: N802
            self._send(200, {"ok": True, "status": core.last_status,
                             "joint_names": JOINT_NAMES})

        def log_message(self, fmt, *args):  # тише
            print("[bridge] %s" % (fmt % args), flush=True)

    return Handler


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--http-only", action="store_true",
                        help="деградированный режим: HTTP без ROS (честный 503)")
    opts = parser.parse_args(args)

    ros_adapter = None
    node = None
    if not opts.http_only:
        import rclpy  # noqa: PLC0415 — только когда ROS реально нужен

        rclpy.init(args=None)
        node = rclpy.create_node("pose_trajectory_bridge")
        ros_adapter = RosAdapter(node)
        ros_adapter.node._rclpy_mod = rclpy  # для wait-помощников
        spin_thread = threading.Thread(
            target=lambda: _spin_forever(rclpy, node), daemon=True)
        spin_thread.start()

    core = BridgeCore(ros_adapter=ros_adapter)
    server = ThreadingHTTPServer(("0.0.0.0", opts.port), make_http_handler(core))
    print("[bridge] HTTP on :%d | ROS: %s | action: %s | topic: %s" % (
        opts.port, "on" if ros_adapter else "OFF (degraded)", ACTION_NAME, SAFETY_TOPIC),
        flush=True)
    try:
        server.serve_forever()
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()


def _spin_forever(rclpy, node):
    while True:
        rclpy.spin_once(node, timeout_sec=0.1)


if __name__ == "__main__":
    main()
