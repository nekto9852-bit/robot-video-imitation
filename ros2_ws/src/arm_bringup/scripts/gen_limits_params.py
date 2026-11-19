#!/usr/bin/env python3
"""Генератор JTC joint_limits из ЕДИНОГО источника лимитов (шаг 1).

Источники:
  * models/arm_6dof.xml  — position range (канон, MJCF);
  * config/limits.yaml   — velocity_max / acceleration_max (канон).

Выход:
  ros2_ws/src/arm_bringup/config/controllers.generated.yaml
  (секция joint_limits для joint_trajectory_controller; в launch мержится
  поверх config/controllers.yaml).

Режимы:
  python gen_limits_params.py            # записать сгенерированный файл
  python gen_limits_params.py --check    # exit 1, если файл устарел (для CI)

Чистый Python (yaml + xml.etree), без ROS-зависимостей — идёт на Windows.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
MJCF_PATH = REPO / "models" / "arm_6dof.xml"
LIMITS_PATH = REPO / "config" / "limits.yaml"
OUT_PATH = Path(__file__).resolve().parents[1] / "config" / "controllers.generated.yaml"


def mjcf_joints(mjcf_path: Path) -> dict:
    """{joint_name: (min_rad, max_rad)} из MJCF joint range (канон position)."""
    root = ET.parse(mjcf_path).getroot()
    joints = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        jtype = joint.get("type", "hinge")
        rng = joint.get("range")
        if jtype == "free" or rng is None or name is None:
            continue
        lo, hi = (float(x) for x in rng.split())
        joints[name] = (lo, hi)
    return joints


def build_generated_yaml(mjcf_path: Path = MJCF_PATH, limits_path: Path = LIMITS_PATH) -> dict:
    """Полный controllers-конфиг с joint_limits из единого источника."""
    joints = mjcf_joints(mjcf_path)
    limits = yaml.safe_load(limits_path.read_text(encoding="utf-8")) or {}
    yaml_limits = limits.get("limits") or {}
    watchdog_ms = int(limits.get("watchdog_timeout_ms", 750))

    names = list(joints.keys())
    joint_limits = {}
    for name in names:
        lo, hi = joints[name]
        entry = yaml_limits.get(name) or {}
        joint_limits[name] = {
            "has_position_limits": True,
            "min_position": lo,
            "max_position": hi,
            "has_velocity_limits": True,
            "max_velocity": float(entry.get("velocity_max", 2.0)),
            "has_acceleration_limits": True,
            "max_acceleration": float(entry.get("acceleration_max", 25.0)),
        }
    return {
        "meta": {
            "generated_from": {
                "mjcf": str(mjcf_path), "limits": str(limits_path),
                "watchdog_timeout_ms": watchdog_ms,
            },
            "note": "генерат gen_limits_params.py — не редактировать руками",
        },
        "joint_trajectory_controller": {
            "ros__parameters": {"joints": names, "joint_limits": joint_limits},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="не писать; exit 1, если сгенерированный файл устарел")
    args = parser.parse_args()

    generated = yaml.safe_dump(
        build_generated_yaml(MJCF_PATH, LIMITS_PATH),
        sort_keys=False, allow_unicode=True)

    if args.check:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if current != generated:
            print("controllers.generated.yaml устарел — перегенерируйте", file=sys.stderr)
            return 1
        print("controllers.generated.yaml актуален")
        return 0

    OUT_PATH.write_text(generated, encoding="utf-8")
    print("written:", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
