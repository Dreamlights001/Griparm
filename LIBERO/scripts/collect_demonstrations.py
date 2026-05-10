#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import shutil
import sys
from pathlib import Path

if "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import glfw
import mujoco.viewer
import numpy as np
import yaml

from libero_griparm import GriparmSortingEnv
from libero_griparm.env import ARM_JOINTS, GRIPPER_JOINT


ARM_STEP = 0.03
GRIPPER_STEP = 0.003


def make_lerobot_dataset(root_dir: Path, fps: int, width: int, height: int):
    from ledataset.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ARM_JOINTS + [GRIPPER_JOINT],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"{j}_cmd" for j in ARM_JOINTS] + ["gripper_cmd"],
        },
        "wrist": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "global": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }
    return LeRobotDataset.create(
        repo_id="Griparm_LIBERO",
        root=root_dir,
        fps=fps,
        robot_type="arm_6dof_claw",
        features=features,
        use_videos=True,
        vcodec="h264",
    )


def key_to_token(keycode: int) -> str | None:
    mapping = {
        glfw.KEY_ESCAPE: "ESC",
        glfw.KEY_ENTER: "ENTER",
        glfw.KEY_KP_ENTER: "ENTER",
        glfw.KEY_KP_DECIMAL: "DISCARD",
        glfw.KEY_UP: "UP",
        glfw.KEY_DOWN: "DOWN",
        glfw.KEY_LEFT: "LEFT",
        glfw.KEY_RIGHT: "RIGHT",
        glfw.KEY_KP_1: "KP_1",
        glfw.KEY_KP_2: "KP_2",
        glfw.KEY_KP_4: "KP_4",
        glfw.KEY_KP_5: "KP_5",
        glfw.KEY_KP_6: "KP_6",
        glfw.KEY_KP_7: "KP_7",
        glfw.KEY_KP_8: "KP_8",
        glfw.KEY_KP_9: "KP_9",
        glfw.KEY_KP_ADD: "KP_ADD",
        glfw.KEY_KP_SUBTRACT: "KP_SUBTRACT",
    }
    return mapping.get(keycode)


def apply_token(action: np.ndarray, token: str, gripper_open: float, gripper_closed: float) -> str | None:
    mapping = {
        "LEFT": (0, +ARM_STEP), "RIGHT": (0, -ARM_STEP),
        "UP": (1, +ARM_STEP), "DOWN": (1, -ARM_STEP),
        "KP_1": (2, -ARM_STEP), "KP_2": (2, +ARM_STEP),
        "KP_4": (3, +ARM_STEP), "KP_6": (3, -ARM_STEP),
        "KP_5": (4, -ARM_STEP), "KP_8": (4, +ARM_STEP),
        "KP_7": (5, +ARM_STEP), "KP_9": (5, -ARM_STEP),
        "KP_ADD": (6, -GRIPPER_STEP),
        "KP_SUBTRACT": (6, +GRIPPER_STEP),
    }
    if token in mapping:
        idx, delta = mapping[token]
        action[idx] += delta
        if idx == 6:
            action[idx] = float(np.clip(action[idx], gripper_open, gripper_closed))
        return None
    if token == "ENTER":
        return "save"
    if token == "DISCARD":
        return "discard"
    if token == "ESC":
        return "exit"
    return None


def add_lerobot_frame(dataset, obs: dict[str, np.ndarray], action: np.ndarray, task: str) -> None:
    dataset.add_frame({
        "observation.state": obs["robot_state"].astype(np.float32),
        "action": action.astype(np.float32),
        "wrist": obs["wrist_image"],
        "global": obs["global_image"],
        "task": task,
    })


def collect_one(env: GriparmSortingEnv, dataset, cfg: dict, seed: int, use_viewer: bool) -> bool:
    obs = env.reset(seed=seed)
    action = np.zeros(7, dtype=np.float64)
    action[:6] = env.data.qpos[env.ids.arm_qpos_adr]
    action[6] = cfg["gripper_open"]
    max_steps = int(cfg["max_episode_seconds"] * cfg["control_hz"])
    key_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
    command = None

    def on_key(keycode: int) -> None:
        token = key_to_token(keycode)
        if token is not None:
            key_queue.put(token)

    viewer_ctx = mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) if use_viewer else None
    try:
        for _ in range(max_steps):
            while True:
                try:
                    token = key_queue.get_nowait()
                except queue.Empty:
                    break
                result = apply_token(action, token, cfg["gripper_open"], cfg["gripper_closed"])
                if result is not None:
                    command = result

            obs, _reward, done, info = env.step(action)
            add_lerobot_frame(dataset, obs, action, cfg["language_instruction"])

            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    command = "exit"
                    break
                viewer_ctx.sync()

            if command == "save" or info["success"]:
                dataset.save_episode()
                return True
            if command in {"discard", "exit"} or info["failure"]:
                dataset.clear_episode_buffer(delete_images=True)
                return False
            if done:
                if info["success"]:
                    dataset.save_episode()
                    return True
                dataset.clear_episode_buffer(delete_images=True)
                return False
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
    dataset.clear_episode_buffer(delete_images=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LIBERO-configured Griparm data in LeRobot format.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/griparm_sorting.yaml")
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output if args.output is not None else ROOT / cfg["dataset_path"]
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    env = GriparmSortingEnv(
        xml_path=(ROOT / cfg["scene_xml"]).resolve(),
        physics_hz=int(cfg["physics_hz"]),
        control_hz=int(cfg["control_hz"]),
        conveyor_speed=float(cfg["conveyor_speed"]),
        width=int(cfg["image_width"]),
        height=int(cfg["image_height"]),
    )
    dataset = make_lerobot_dataset(
        output,
        fps=int(cfg["control_hz"]),
        width=int(cfg["image_width"]),
        height=int(cfg["image_height"]),
    )
    saved = 0
    try:
        for attempt in range(args.num_demos):
            ok = collect_one(env, dataset, cfg, args.seed + attempt, not args.no_viewer)
            if ok:
                saved += 1
            print(f"[LIBERO LeRobot collect] attempt={attempt} saved={ok} total_saved={saved}")
    finally:
        dataset.finalize()
        env.close()
    print(f"[LIBERO LeRobot collect] output={output} saved={saved}/{args.num_demos}")


if __name__ == "__main__":
    main()
