#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from libero_griparm import GriparmRobosuiteEnv


def main() -> None:
    config = ROOT / "configs/griparm_sorting.yaml"
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    env = GriparmRobosuiteEnv(
        xml_path=ROOT / cfg["scene_xml"],
        physics_hz=int(cfg["physics_hz"]),
        control_hz=int(cfg["control_hz"]),
        conveyor_speed=float(cfg["conveyor_speed"]),
        width=int(cfg["image_width"]),
        height=int(cfg["image_height"]),
        horizon=int(float(cfg["max_episode_seconds"]) * int(cfg["control_hz"])),
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    try:
        obs = env.reset(seed=0)
        print("[LIBERO check] scene loaded")
        print(f"  xml: {ROOT / cfg['scene_xml']}")
        print(f"  robot_state: {obs['robot_state'].shape}")
        print(f"  global_image: {obs['global_image'].shape}")
        print(f"  wrist_image: {obs['wrist_image'].shape}")
        print(f"  place_center: {env.place_center.round(4).tolist()} radius={env.place_radius:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
