#!/usr/bin/env python3
"""Run MuJoCo expert policy and collect LeRobot-format Sim2Sim data."""

from __future__ import annotations

import argparse
import ctypes
# NOTE: debug_cameras (and thus cv2) MUST be imported before av and mujoco.viewer,
# otherwise cv2 Qt windows will hang after those modules initialize their codecs/GL.
from debug_cameras import get_home_pose_from_model
from debug_cameras import PreviewBackend
from debug_cameras import matrix_to_quat
from debug_cameras import apply_lighting_for_debug

import av
from datetime import datetime
import json
import math
import os
import queue
import random
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Iterable

# Prefer desktop rendering when a display server is present, otherwise use EGL.
if "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import glfw


ARM_JOINTS = ["J_jianbu", "J_dabi", "J_Upper", "J_Lower", "J_wrist", "J_hand"]
GRIPPER_JOINT = "Claw_left"
TASK_TEXT = "pick up the triangular prism anomaly and avoid the cubes"

PHYSICS_HZ = 500
DATA_HZ = 50
SAMPLE_EVERY = PHYSICS_HZ // DATA_HZ
DEFAULT_MAX_DATA_FRAMES = 2500
DEFAULT_CONVEYOR_SPEED = 0.025
WRIST_ROTATE_QUARTER_TURNS_CCW = 0
VISIBLE_SAMPLE_COUNT = 4
CONVEYOR_DRIVE_MAX_Z = 0.12

TABLE_Z = 0.02
CONVEYOR_LENGTH = 1.2
CONVEYOR_BODY = "layout_conveyor"
CONVEYOR_GEOM = "layout_conveyor_geom"
CONVEYOR_COLLISION_GEOM = "layout_conveyor_collision"
CONVEYOR_HALF_LENGTH = CONVEYOR_LENGTH * 0.5
CONVEYOR_HALF_WIDTH = 0.11
CONVEYOR_HALF_HEIGHT = 0.01
OBJECT_SPAWN_S_RANGE = (0.0, CONVEYOR_LENGTH / 6.0)
LATERAL_MARGIN = 0.02
OBJECT_SPAWN_LATERAL_RANGE = (
    -(CONVEYOR_HALF_WIDTH - LATERAL_MARGIN),
    (CONVEYOR_HALF_WIDTH - LATERAL_MARGIN),
)
OBJECT_CLEARANCE = 0.045
OBJECT_SETTLE_STEPS = 140
OBJECT_SETTLE_DROP_HEIGHT = 0.004
OBJECT_CENTER_Z_ON_BELT = CONVEYOR_HALF_HEIGHT + 0.018
RESPAWN_MARGIN = 0.06
DEFAULT_LAYOUT_XML = Path("env_layout_tuned.xml")
DEFAULT_CAMERA_XML = Path("env_camera_tuned.xml")
DEFAULT_GRASP_CALIB_JSON = Path("calib_grasp.json")
HIDDEN_OBJECT_QPOS = np.array([-2.0, -2.0, -1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
TELEOP_ARM_STEP = 0.015
TELEOP_GRIPPER_STEP = 0.0015
TELEOP_JOINT_SPEED = {
    "LEFT": (0, +0.4), "RIGHT": (0, -0.4),
    "UP": (1, +0.75), "DOWN": (1, -0.75),
    "KP_1": (2, -1.0), "KP_2": (2, +1.0),
    "KP_4": (3, +1.0), "KP_6": (3, -1.0),
    "KP_5": (4, -1.0), "KP_8": (4, +1.0),
    "KP_7": (5, +1.0), "KP_9": (5, -1.0),
    "KP_ADD": (-1, -0.02),
    "KP_SUBTRACT": (-1, +0.02),
}
AUTO_J1_RAMP_SPEED = 0.4
X11_KEYSYMS = {
    "LEFT": 0xFF51, "UP": 0xFF52, "RIGHT": 0xFF53, "DOWN": 0xFF54,
    "KP_1": 0xFFB1, "KP_2": 0xFFB2, "KP_4": 0xFFB4, "KP_5": 0xFFB5,
    "KP_6": 0xFFB6, "KP_7": 0xFFB7, "KP_8": 0xFFB8, "KP_9": 0xFFB9,
    "KP_ADD": 0xFFAB, "KP_SUBTRACT": 0xFFAD,
}


class PolicyState(Enum):
    WAITING = auto()
    TRACKING = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT_PLACE = auto()
    DONE = auto()


@dataclass
class EpisodeConfig:
    width: int = 256
    height: int = 256
    conveyor_speed: float = DEFAULT_CONVEYOR_SPEED
    max_data_frames: int = DEFAULT_MAX_DATA_FRAMES
    prediction_time: float = 0.5
    pregrasp_height: float = 0.12
    grasp_height: float = 0.06
    tracking_stable_sec: float = 0.08
    descend_sec: float = 1.0
    max_descend_sec: float = 4.0
    wait_station_s: float = CONVEYOR_LENGTH * 0.5
    reachable_pos_err: float = 0.035
    grasp_hold_sec: float = 0.45
    release_hold_sec: float = 0.30
    gripper_open: float = 0.0     # URDF qpos0: claws apart
    gripper_closed: float = 0.038  # near max: claws together
    grasp_calibration: dict | None = None
    # calib_grasp.json stores a pre-grasp TCP pose above the object. Auto mode
    # first tracks that relative pose, then descends by this amount while
    # preserving the calibrated axis/side offset and following conveyor motion.
    calibration_grasp_drop: float = 0.035

    @property
    def max_physics_steps(self) -> int:
        return self.max_data_frames * SAMPLE_EVERY


@dataclass
class SimContext:
    arm_joint_ids: list[int]
    arm_qpos_adr: np.ndarray
    arm_dof_adr: np.ndarray
    gripper_joint_id: int
    gripper_actuator_id: int
    gripper_right_actuator_id: int
    arm_actuator_ids: list[int]
    tcp_site_id: int
    hand_body_id: int
    anomaly_body_id: int
    normal_body_ids: list[int]
    claw_left_body_id: int
    claw_right_body_id: int
    object_qpos_adr: dict[str, int]
    object_dof_adr: dict[str, int]
    home_qpos: np.ndarray
    conveyor_body_id: int
    place_body_id: int
    conveyor_center: np.ndarray
    conveyor_dir: np.ndarray
    conveyor_lateral: np.ndarray
    conveyor_start: np.ndarray
    place_center: np.ndarray
    place_radius: float
    camera_wrist: str = "wrist"
    camera_global: str = "global"

    @property
    def all_object_names(self) -> list[str]:
        return ["anomaly_0"] + [f"normal_{i}" for i in range(len(self.normal_body_ids))]


@dataclass
class PolicyContext:
    state: PolicyState = PolicyState.WAITING
    lift_place_phase: int = 0
    step_counter_in_state: int = 0
    last_target_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    hold_anomaly_on_conveyor: bool = False
    grasp_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    grip_target: float = 0.0   # gradually ramped gripper target during GRASP
    _prev_claw_q: float = 0.0  # for detecting when claws stop (contact made)
    grasped: bool = False       # true after verified grip → kinematic hold
    target_reached_steps: int = 0
    last_wait_log_step: int = 0
    calib_axis_sign: float = 0.0
    calib_side_sign: float = 0.0
    grasp_tcp_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))


@dataclass
class TeleopState:
    arm_target: np.ndarray
    gripper_target: float
    paused: bool = False
    save_requested: bool = False
    discard_requested: bool = False
    exit_requested: bool = False


class X11KeyPoller:
    """Poll physical key state so MuJoCo teleop supports both hold and tap."""

    def __init__(self, tokens: Iterable[str]):
        self.display = None
        self.keycodes: dict[str, int] = {}
        self.x11 = None
        if not os.environ.get("DISPLAY"):
            return
        try:
            self.x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
            self.x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
            self.x11.XOpenDisplay.restype = ctypes.c_void_p
            self.x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
            self.x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            self.x11.XKeysymToKeycode.restype = ctypes.c_uint
            self.x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char * 32)]
            self.x11.XQueryKeymap.restype = ctypes.c_int
            self.display = self.x11.XOpenDisplay(None)
            if not self.display:
                return
            for token in tokens:
                keysym = X11_KEYSYMS.get(token)
                if keysym is None:
                    continue
                keycode = self.x11.XKeysymToKeycode(self.display, keysym)
                if keycode:
                    self.keycodes[token] = int(keycode)
        except Exception:
            self.close()

    @property
    def available(self) -> bool:
        return bool(self.display and self.keycodes)

    def pressed_tokens(self) -> set[str]:
        if not self.available:
            return set()
        keymap = (ctypes.c_char * 32)()
        if not self.x11.XQueryKeymap(self.display, ctypes.byref(keymap)):
            return set()
        pressed = set()
        for token, keycode in self.keycodes.items():
            byte = keymap[keycode // 8]
            if isinstance(byte, bytes):
                byte = byte[0]
            if byte & (1 << (keycode % 8)):
                pressed.add(token)
        return pressed

    def close(self) -> None:
        if self.display and self.x11:
            self.x11.XCloseDisplay(self.display)
        self.display = None


def _indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def prepare_collection_xml(src: Path) -> Path:
    """Normalize scene visuals and collision setup for collection."""
    tree = ET.parse(src)
    root = tree.getroot()

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib["gravity"] = "0 0 -9.81"
    option.attrib["timestep"] = f"{1.0 / PHYSICS_HZ:.6f}"

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Invalid XML: missing worldbody.")

    for geom in list(worldbody):
        if geom.tag == "geom" and geom.attrib.get("name") == "table":
            worldbody.remove(geom)

    if asset.find(".//texture[@name='collect_checker_tex']") is None:
        ET.SubElement(
            asset,
            "texture",
            name="collect_checker_tex",
            type="2d",
            builtin="checker",
            width="512",
            height="512",
            rgb1="0.20 0.25 0.32",
            rgb2="0.12 0.14 0.18",
        )
    if asset.find(".//material[@name='collect_checker_mat']") is None:
        ET.SubElement(
            asset,
            "material",
            name="collect_checker_mat",
            texture="collect_checker_tex",
            texrepeat="20 20",
            texuniform="true",
            reflectance="0.05",
            shininess="0.1",
            specular="0.1",
        )
    if worldbody.find(".//geom[@name='collect_checker_floor']") is None:
        ET.SubElement(
            worldbody,
            "geom",
            name="collect_checker_floor",
            type="plane",
            pos="0 0 0",
            size="4 4 0.1",
            material="collect_checker_mat",
            friction="1.0 0.005 0.0001",
            condim="3",
            contype="1",
            conaffinity="1",
        )

    conveyor_body = root.find(f".//body[@name='{CONVEYOR_BODY}']")
    if conveyor_body is None:
        raise RuntimeError(f"Invalid XML: missing {CONVEYOR_BODY}.")

    conv_geom = root.find(f".//geom[@name='{CONVEYOR_GEOM}']")
    if conv_geom is not None:
        conv_geom.attrib["type"] = "box"
        conv_geom.attrib["size"] = f"{CONVEYOR_HALF_LENGTH:.6f} {CONVEYOR_HALF_WIDTH:.6f} 0.002"
        conv_geom.attrib["rgba"] = "0.40 0.40 0.40 0.90"
        conv_geom.attrib["contype"] = "0"
        conv_geom.attrib["conaffinity"] = "0"
    if conveyor_body.find(f"./geom[@name='{CONVEYOR_COLLISION_GEOM}']") is None:
        ET.SubElement(
            conveyor_body,
            "geom",
            name=CONVEYOR_COLLISION_GEOM,
            type="plane",
            pos="0 0 0.002",
            size="2 2 0.1",
            rgba="0 0 0 0",
            contype="1",
            conaffinity="1",
            friction="0.8 0.005 0.0001",
            condim="3",
        )

    for geom in root.findall(".//geom"):
        name = geom.attrib.get("name", "")
        if name.startswith("anomaly_") or name.startswith("normal_"):
            geom.attrib.setdefault("contype", "1")
            geom.attrib.setdefault("conaffinity", "1")
            geom.attrib["condim"] = "6"
            geom.attrib["friction"] = "1.0 0.05 0.005"
            geom.attrib["solref"] = "0.01 1"
            geom.attrib["solimp"] = "0.9 0.99 0.001"
        elif name in {"Claw_Link_left", "Claw_Link_right"}:
            geom.attrib["condim"] = "6"
            geom.attrib["friction"] = "1.5 0.2 0.02"
            geom.attrib["solref"] = "0.02 1"
            geom.attrib["solimp"] = "0.9 0.95 0.003"
            geom.attrib["margin"] = "0.0005"

    _indent(root)
    fd, tmp_path = tempfile.mkstemp(prefix="collect_", suffix=".xml")
    os.close(fd)
    out = Path(tmp_path)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def mat_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.array(quat, dtype=np.float64, copy=True)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return np.zeros_like(vec)
    return vec / n


def orientation_error(current_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    # Geometric orientation error in R^3 from SO(3) matrices.
    return 0.5 * (
        np.cross(current_rot[:, 0], target_rot[:, 0])
        + np.cross(current_rot[:, 1], target_rot[:, 1])
        + np.cross(current_rot[:, 2], target_rot[:, 2])
    )


def solve_ik_dls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    arm_qpos_adr: np.ndarray,
    arm_dof_adr: np.ndarray,
    max_iter: int = 100,
    damping: float = 0.01,
    pos_tol: float = 5e-4,
    rot_tol: float = 2e-2,
    max_dq_per_step: float = 0.10,
    return_error: bool = False,
) -> np.ndarray | tuple[np.ndarray, float, float]:
    target_rot = mat_from_quat_wxyz(target_quat)
    q_orig = data.qpos[arm_qpos_adr].copy()
    q = q_orig.copy()

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    i6 = np.eye(6, dtype=np.float64)

    final_pos_err_norm = float("inf")
    final_rot_err_norm = float("inf")
    for _ in range(max_iter):
        data.qpos[arm_qpos_adr] = q
        mujoco.mj_forward(model, data)
        curr_pos = data.site_xpos[site_id].copy()
        curr_rot = data.site_xmat[site_id].reshape(3, 3).copy()

        pos_err = target_pos - curr_pos
        rot_err = orientation_error(curr_rot, target_rot)
        final_pos_err_norm = float(np.linalg.norm(pos_err))
        final_rot_err_norm = float(np.linalg.norm(rot_err))
        if final_pos_err_norm < pos_tol and final_rot_err_norm < rot_tol:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        j = np.vstack([jacp[:, arm_dof_adr], jacr[:, arm_dof_adr]])
        err = np.concatenate([pos_err, rot_err])

        a = j @ j.T + (damping**2) * i6
        dq = j.T @ np.linalg.solve(a, err)
        dq = np.clip(dq, -0.03, 0.03)
        q = q + dq

    data.qpos[arm_qpos_adr] = q
    mujoco.mj_forward(model, data)
    final_pos_err_norm = float(np.linalg.norm(target_pos - data.site_xpos[site_id]))
    final_rot_err_norm = float(np.linalg.norm(orientation_error(data.site_xmat[site_id].reshape(3, 3), target_rot)))

    # Restore original qpos; return a rate-limited target so the arm
    # moves smoothly via actuators instead of jumping kinematically.
    data.qpos[arm_qpos_adr] = q_orig
    mujoco.mj_forward(model, data)

    # Clip per-joint displacement to max_dq_per_step for smooth motion
    dq_total = q - q_orig
    dq_total = np.clip(dq_total, -max_dq_per_step, max_dq_per_step)
    q_target = q_orig + dq_total
    if return_error:
        return q_target, final_pos_err_norm, final_rot_err_norm
    return q_target


def solve_ik_dls_target(*args, **kwargs) -> np.ndarray:
    q_target, _pos_err, _rot_err = solve_ik_dls(*args, return_error=True, **kwargs)
    return q_target


def solve_ik_dls_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    arm_qpos_adr: np.ndarray,
    arm_dof_adr: np.ndarray,
    max_iter: int = 100,
    damping: float = 0.01,
    pos_tol: float = 5e-4,
    rot_tol: float = 2e-2,
    max_dq_per_step: float = 0.10,
    return_error: bool = False,
) -> np.ndarray | tuple[np.ndarray, float, float]:
    target_rot = mat_from_quat_wxyz(target_quat)
    q_orig = data.qpos[arm_qpos_adr].copy()
    q = q_orig.copy()

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    i6 = np.eye(6, dtype=np.float64)
    final_pos_err_norm = float("inf")
    final_rot_err_norm = float("inf")

    for _ in range(max_iter):
        data.qpos[arm_qpos_adr] = q
        mujoco.mj_forward(model, data)
        curr_pos = data.xpos[body_id].copy()
        curr_rot = data.xmat[body_id].reshape(3, 3).copy()

        pos_err = target_pos - curr_pos
        rot_err = orientation_error(curr_rot, target_rot)
        final_pos_err_norm = float(np.linalg.norm(pos_err))
        final_rot_err_norm = float(np.linalg.norm(rot_err))
        if final_pos_err_norm < pos_tol and final_rot_err_norm < rot_tol:
            break

        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        j = np.vstack([jacp[:, arm_dof_adr], jacr[:, arm_dof_adr]])
        err = np.concatenate([pos_err, rot_err])
        dq = j.T @ np.linalg.solve(j @ j.T + (damping**2) * i6, err)
        q = q + np.clip(dq, -0.03, 0.03)

    data.qpos[arm_qpos_adr] = q
    mujoco.mj_forward(model, data)
    final_pos_err_norm = float(np.linalg.norm(target_pos - data.xpos[body_id]))
    final_rot_err_norm = float(np.linalg.norm(orientation_error(data.xmat[body_id].reshape(3, 3), target_rot)))

    data.qpos[arm_qpos_adr] = q_orig
    mujoco.mj_forward(model, data)

    dq_total = np.clip(q - q_orig, -max_dq_per_step, max_dq_per_step)
    q_target = q_orig + dq_total
    if return_error:
        return q_target, final_pos_err_norm, final_rot_err_norm
    return q_target


def ik_position_residual(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    arm_qpos_adr: np.ndarray,
    arm_dof_adr: np.ndarray,
    max_iter: int = 80,
    damping: float = 0.01,
) -> float:
    """Position-only IK probe used for reachable-window gating.

    Full pose IK can reject an otherwise reachable target when the wrist
    orientation is temporarily hard to satisfy. WAITING only needs to know
    whether the calibrated pre-grasp position has entered the arm workspace.
    """
    q_orig = data.qpos[arm_qpos_adr].copy()
    q = q_orig.copy()
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    i3 = np.eye(3, dtype=np.float64)

    for _ in range(max_iter):
        data.qpos[arm_qpos_adr] = q
        mujoco.mj_forward(model, data)
        pos_err = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(pos_err) < 5e-4:
            break
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        j = jacp[:, arm_dof_adr]
        dq = j.T @ np.linalg.solve(j @ j.T + (damping**2) * i3, pos_err)
        q = q + np.clip(dq, -0.04, 0.04)

    data.qpos[arm_qpos_adr] = q
    mujoco.mj_forward(model, data)
    residual = float(np.linalg.norm(target_pos - data.site_xpos[site_id]))
    data.qpos[arm_qpos_adr] = q_orig
    mujoco.mj_forward(model, data)
    return residual


def ik_body_position_residual(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    target_pos: np.ndarray,
    arm_qpos_adr: np.ndarray,
    arm_dof_adr: np.ndarray,
    max_iter: int = 80,
    damping: float = 0.01,
) -> float:
    q_orig = data.qpos[arm_qpos_adr].copy()
    q = q_orig.copy()
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    i3 = np.eye(3, dtype=np.float64)

    for _ in range(max_iter):
        data.qpos[arm_qpos_adr] = q
        mujoco.mj_forward(model, data)
        pos_err = target_pos - data.xpos[body_id]
        if np.linalg.norm(pos_err) < 5e-4:
            break
        mujoco.mj_jacBody(model, data, jacp, None, body_id)
        j = jacp[:, arm_dof_adr]
        dq = j.T @ np.linalg.solve(j @ j.T + (damping**2) * i3, pos_err)
        q = q + np.clip(dq, -0.04, 0.04)

    data.qpos[arm_qpos_adr] = q
    mujoco.mj_forward(model, data)
    residual = float(np.linalg.norm(target_pos - data.xpos[body_id]))
    data.qpos[arm_qpos_adr] = q_orig
    mujoco.mj_forward(model, data)
    return residual


def resolve_context(model: mujoco.MjModel, data: mujoco.MjData) -> SimContext:
    arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
    if any(jid < 0 for jid in arm_joint_ids):
        raise RuntimeError("Arm joints are missing in env.xml.")

    gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
    if gripper_joint_id < 0:
        raise RuntimeError("Claw_left joint missing in env.xml.")

    arm_qpos_adr = np.array([model.jnt_qposadr[jid] for jid in arm_joint_ids], dtype=np.int32)
    arm_dof_adr = np.array([model.jnt_dofadr[jid] for jid in arm_joint_ids], dtype=np.int32)

    arm_actuator_ids = []
    for jn in ARM_JOINTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jn}_pos")
        if aid < 0:
            raise RuntimeError(f"Actuator {jn}_pos missing in env.xml.")
        arm_actuator_ids.append(aid)

    gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Claw_left_pos")
    if gripper_actuator_id < 0:
        raise RuntimeError("Actuator Claw_left_pos missing in env.xml.")
    gripper_right_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Claw_right_pos")
    if gripper_right_actuator_id < 0:
        raise RuntimeError("Actuator Claw_right_pos missing in env.xml.")

    tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    if tcp_site_id < 0:
        raise RuntimeError("tcp_site missing in env.xml.")
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Hand_Link")
    if hand_body_id < 0:
        raise RuntimeError("Hand_Link body missing in env.xml.")

    anomaly_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "anomaly_0")
    if anomaly_body_id < 0:
        raise RuntimeError("anomaly_0 body missing in env.xml.")

    claw_left_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Claw_Link_left")
    claw_right_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Claw_Link_right")
    if claw_left_body_id < 0 or claw_right_body_id < 0:
        raise RuntimeError("Claw_Link_left or Claw_Link_right body missing in env.xml.")

    normal_body_ids = []
    for i in range(5):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"normal_{i}")
        if bid < 0:
            raise RuntimeError(f"normal_{i} body missing in env.xml.")
        normal_body_ids.append(bid)

    object_qpos_adr = {}
    object_dof_adr = {}
    for name in ["anomaly_0"] + [f"normal_{i}" for i in range(5)]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        jnt_adr = model.body_jntadr[body_id]
        jnt_id = jnt_adr
        if model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"{name} is expected to have a freejoint.")
        object_qpos_adr[name] = int(model.jnt_qposadr[jnt_id])
        object_dof_adr[name] = int(model.jnt_dofadr[jnt_id])

    conveyor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "layout_conveyor")
    place_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "layout_place_region")
    if conveyor_body_id < 0 or place_body_id < 0:
        raise RuntimeError("Missing layout_conveyor or layout_place_region in env xml.")

    conveyor_center = model.body_pos[conveyor_body_id].copy()
    conveyor_rot = mat_from_quat_wxyz(model.body_quat[conveyor_body_id])
    conveyor_dir = normalize(conveyor_rot[:, 0].copy())
    conveyor_lateral = normalize(conveyor_rot[:, 1].copy())
    conveyor_start = conveyor_center - conveyor_dir * CONVEYOR_HALF_LENGTH

    place_center = model.body_pos[place_body_id].copy()
    place_radius = 0.15
    geom_adr = int(model.body_geomadr[place_body_id])
    geom_num = int(model.body_geomnum[place_body_id])
    if geom_num > 0:
        place_radius = float(model.geom_size[geom_adr, 0])

    arm_home, gripper_home = get_home_pose_from_model(model)
    home_qpos = data.qpos.copy()
    home_qpos[arm_qpos_adr] = arm_home
    home_qpos[model.jnt_qposadr[gripper_joint_id]] = gripper_home
    return SimContext(
        arm_joint_ids=arm_joint_ids,
        arm_qpos_adr=arm_qpos_adr,
        arm_dof_adr=arm_dof_adr,
        gripper_joint_id=gripper_joint_id,
        gripper_actuator_id=gripper_actuator_id,
        gripper_right_actuator_id=gripper_right_actuator_id,
        arm_actuator_ids=arm_actuator_ids,
        tcp_site_id=tcp_site_id,
        hand_body_id=hand_body_id,
        anomaly_body_id=anomaly_body_id,
        normal_body_ids=normal_body_ids,
        claw_left_body_id=claw_left_body_id,
        claw_right_body_id=claw_right_body_id,
        object_qpos_adr=object_qpos_adr,
        object_dof_adr=object_dof_adr,
        home_qpos=home_qpos,
        conveyor_body_id=conveyor_body_id,
        place_body_id=place_body_id,
        conveyor_center=conveyor_center,
        conveyor_dir=conveyor_dir,
        conveyor_lateral=conveyor_lateral,
        conveyor_start=conveyor_start,
        place_center=place_center,
        place_radius=place_radius,
    )


def enforce_camera_binding(model: mujoco.MjModel) -> None:
    """Ensure camera bindings are correct before collection."""
    global_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "global")
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Hand_Link")
    if global_cam_id < 0 or wrist_cam_id < 0 or hand_body_id < 0:
        raise RuntimeError("Missing required names: global camera / wrist camera / Hand_Link body.")

    # Global camera: fixed in world.
    model.cam_mode[global_cam_id] = int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
    model.cam_targetbodyid[global_cam_id] = -1
    model.cam_bodyid[global_cam_id] = 0

    # Wrist camera: fixed relative to Hand_Link, thus moving with arm.
    model.cam_mode[wrist_cam_id] = int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
    model.cam_targetbodyid[wrist_cam_id] = -1
    model.cam_bodyid[wrist_cam_id] = hand_body_id


def object_axis_frame(data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_rot = data.xmat[body_id].reshape(3, 3)
    axis = body_rot[:, 2].copy()
    axis[2] = 0.0
    if np.linalg.norm(axis) < 1e-8:
        axis = body_rot[:, 0].copy()
        axis[2] = 0.0
    axis = normalize(axis)
    side = normalize(np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), axis))
    if np.linalg.norm(side) < 1e-8:
        side = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return axis, side, up


def desired_gripper_quat_for_object(data: mujoco.MjData, anomaly_body_id: int) -> np.ndarray:
    # Construct gripper orientation from explicit axes:
    #   Z = approach direction (vertical down toward conveyor)
    #   X = grip direction (perpendicular to object major axis, horizontal)
    #   Y = orthogonal to X and Z
    obj_axis, _obj_side, _up = object_axis_frame(data, anomaly_body_id)

    # Grip X axis: perpendicular to object axis, horizontal (claws grip from sides).
    grip_x = np.cross(obj_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if np.linalg.norm(grip_x) < 1e-6:
        grip_x = np.cross(obj_axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    grip_x = normalize(grip_x)

    approach_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # Orthonormal basis: Y = Z × X, then re-orthogonalize X = Y × Z.
    grip_y = normalize(np.cross(approach_z, grip_x))
    grip_x = normalize(np.cross(grip_y, approach_z))

    rot = np.column_stack([grip_x, grip_y, approach_z])
    return matrix_to_quat(rot)


def load_grasp_calibration(path: Path) -> dict | None:
    if not path.exists():
        print(f"[collect_data] grasp calibration not found: {path}; using center-based fallback")
        return None
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if (
        "gripper_body_axis_offset" in raw
        and "gripper_body_side_offset" in raw
        and "gripper_body_height_offset" in raw
    ):
        raw.setdefault("track_frame", "Hand_Link")
        return raw

    if "tcp_axis_offset_abs" in raw and "tcp_side_offset_abs" in raw and "tcp_height_offset" in raw:
        print(
            f"[collect_data] warning: {path} is legacy tcp_site calibration. "
            "Regenerate with calibrate_grasp.py so auto mode tracks Hand_Link."
        )
        return raw

    if "tcp_to_object" in raw and "object_axis_xy" in raw:
        axis = normalize(np.asarray(raw["object_axis_xy"], dtype=np.float64))
        side = normalize(np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), axis))
        rel_obj_to_tcp = -np.asarray(raw["tcp_to_object"], dtype=np.float64)
        raw["schema_version"] = 2
        raw["tcp_from_object"] = rel_obj_to_tcp.tolist()
        raw["object_side_xy"] = side.tolist()
        raw["tcp_axis_offset"] = float(np.dot(rel_obj_to_tcp, axis))
        raw["tcp_side_offset"] = float(np.dot(rel_obj_to_tcp, side))
        raw["tcp_axis_offset_abs"] = abs(float(raw["tcp_axis_offset"]))
        raw["tcp_side_offset_abs"] = abs(float(raw["tcp_side_offset"]))
        raw["tcp_height_offset"] = float(rel_obj_to_tcp[2])
        print(f"[collect_data] upgraded legacy grasp calibration in memory: {path}")
        return raw

    raise ValueError(f"Unsupported grasp calibration format: {path}")


def calibrated_gripper_body_target(
    data: mujoco.MjData,
    ctx: SimContext,
    policy: PolicyContext,
    cfg: EpisodeConfig,
    object_pos: np.ndarray,
    height_delta: float,
) -> np.ndarray:
    calib = cfg.grasp_calibration
    if calib is None:
        return object_pos + np.array([0.0, 0.0, cfg.grasp_height + height_delta], dtype=np.float64)

    axis, side, up = object_axis_frame(data, ctx.anomaly_body_id)
    axis_offset = float(calib.get(
        "gripper_body_axis_offset",
        calib.get("tcp_axis_offset", 0.0),
    ))
    side_offset = float(calib.get(
        "gripper_body_side_offset",
        calib.get("tcp_side_offset", 0.0),
    ))
    # Keep a small positive height above object center so a too-large drop
    # cannot drive the gripper body reference below the workpiece centerline.
    base_height = float(calib.get("gripper_body_height_offset", calib.get("tcp_height_offset", 0.0)))
    height = max(0.015, base_height + height_delta)

    return (
        object_pos
        + axis * axis_offset
        + side * side_offset
        + up * height
    )


def calibrated_tcp_target(*args, **kwargs) -> np.ndarray:
    return calibrated_gripper_body_target(*args, **kwargs)


def calibrated_gripper_body_height(cfg: EpisodeConfig, height_delta: float = 0.0) -> float:
    calib = cfg.grasp_calibration
    if calib is None:
        return cfg.grasp_height + height_delta
    base_height = float(calib.get("gripper_body_height_offset", calib.get("tcp_height_offset", 0.0)))
    return max(0.015, base_height + height_delta)


def wait_station_gripper_body_target(
    data: mujoco.MjData,
    ctx: SimContext,
    cfg: EpisodeConfig,
    object_pos: np.ndarray,
    height_delta: float,
) -> np.ndarray:
    """Safe preposition target for Hand_Link while the moving part is not grabbable.

    This uses the JSON marked height and current object axis orientation, but
    deliberately does not use the signed horizontal axis/side offsets yet.
    The exact horizontal relative relation is applied later in TRACKING.
    """
    rel = object_pos - ctx.conveyor_start
    lat = float(np.dot(rel, ctx.conveyor_lateral))
    lat = float(np.clip(lat, -CONVEYOR_HALF_WIDTH + LATERAL_MARGIN, CONVEYOR_HALF_WIDTH - LATERAL_MARGIN))
    s = float(np.clip(cfg.wait_station_s, 0.0, CONVEYOR_LENGTH))
    station_object_pos = (
        ctx.conveyor_start
        + ctx.conveyor_dir * s
        + ctx.conveyor_lateral * lat
    )
    station_object_pos[2] = object_pos[2]
    return station_object_pos + np.array([0.0, 0.0, calibrated_gripper_body_height(cfg, height_delta)])


def object_quat_laid_down(conveyor_dir: np.ndarray, object_yaw_deg: float) -> np.ndarray:
    conveyor_yaw_deg = math.degrees(math.atan2(conveyor_dir[1], conveyor_dir[0]))
    base = quat_from_euler_xyz(0.0, math.radians(90.0), math.radians(conveyor_yaw_deg + object_yaw_deg))
    flipped = quat_mul(base, quat_from_euler_xyz(0.0, 0.0, math.pi))
    return flipped / (np.linalg.norm(flipped) + 1e-12)


def sample_object_layout(rng: random.Random, count: int) -> list[tuple[float, float, float]]:
    samples: list[tuple[float, float, float]] = []
    attempts = 0
    while len(samples) < count and attempts < 500:
        attempts += 1
        s = rng.uniform(*OBJECT_SPAWN_S_RANGE)
        lateral = rng.uniform(*OBJECT_SPAWN_LATERAL_RANGE)
        if any(math.hypot(s - ps, lateral - pl) < OBJECT_CLEARANCE for ps, pl, _ in samples):
            continue
        samples.append((s, lateral, rng.uniform(0.0, 360.0)))

    while len(samples) < count:
        idx = len(samples)
        frac = idx / max(1, count - 1)
        s = OBJECT_SPAWN_S_RANGE[0] + frac * (OBJECT_SPAWN_S_RANGE[1] - OBJECT_SPAWN_S_RANGE[0])
        lateral = OBJECT_SPAWN_LATERAL_RANGE[0] if idx % 2 == 0 else OBJECT_SPAWN_LATERAL_RANGE[1]
        samples.append((s, lateral, rng.uniform(0.0, 360.0)))
    return samples


def settle_spawned_objects(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctx: SimContext,
    cfg: EpisodeConfig,
    steps: int = OBJECT_SETTLE_STEPS,
) -> None:
    arm_home = ctx.home_qpos[ctx.arm_qpos_adr].copy()
    gripper_home = float(ctx.home_qpos[model.jnt_qposadr[ctx.gripper_joint_id]])
    data.qpos[ctx.arm_qpos_adr] = arm_home
    data.qpos[model.jnt_qposadr[ctx.gripper_joint_id]] = gripper_home
    for i, aid in enumerate(ctx.arm_actuator_ids):
        data.ctrl[aid] = arm_home[i]
    data.ctrl[ctx.gripper_actuator_id] = gripper_home
    data.ctrl[ctx.gripper_right_actuator_id] = -gripper_home

    for _ in range(steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def set_ctrl_from_targets(
    data: mujoco.MjData,
    ctx: SimContext,
    arm_q_target: np.ndarray,
    gripper_target: float,
) -> np.ndarray:
    ctrl = np.zeros(7, dtype=np.float64)
    for i, aid in enumerate(ctx.arm_actuator_ids):
        ctrl[i] = arm_q_target[i]
        data.ctrl[aid] = ctrl[i]
    ctrl[6] = gripper_target
    data.ctrl[ctx.gripper_actuator_id] = gripper_target
    data.ctrl[ctx.gripper_right_actuator_id] = -gripper_target
    return ctrl


def ramp_j1_toward_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctx: SimContext,
    target: float,
    speed: float = AUTO_J1_RAMP_SPEED,
) -> None:
    """Drive the continuous base joint smoothly toward the policy target.

    J1 is modeled as a continuous joint. On this arm model the position
    actuator alone can leave J1 effectively stationary in auto mode, while
    teleop/calibration already use a small qpos ramp. Keep the same bounded
    behavior here instead of teleporting the base angle.
    """
    qadr = int(ctx.arm_qpos_adr[0])
    current = float(data.qpos[qadr])
    err = math.atan2(math.sin(float(target) - current), math.cos(float(target) - current))
    max_step = float(speed) * float(model.opt.timestep)
    data.qpos[qadr] = current + float(np.clip(err, -max_step, max_step))


def teleop_policy(
    data: mujoco.MjData,
    ctx: SimContext,
    state: TeleopState,
) -> np.ndarray:
    return set_ctrl_from_targets(data, ctx, state.arm_target, state.gripper_target)


def glfw_key_to_token(keycode: int) -> str | None:
    mapping = {
        glfw.KEY_ESCAPE: "ESC",
        glfw.KEY_ENTER: "ENTER",
        glfw.KEY_BACKSPACE: "BACKSPACE",
        glfw.KEY_0: "0",
        glfw.KEY_1: "1",
        glfw.KEY_2: "2",
        glfw.KEY_3: "3",
        glfw.KEY_4: "4",
        glfw.KEY_5: "5",
        glfw.KEY_6: "6",
        glfw.KEY_7: "7",
        glfw.KEY_8: "8",
        glfw.KEY_9: "9",
        glfw.KEY_I: "I",
        glfw.KEY_J: "J",
        glfw.KEY_K: "K",
        glfw.KEY_L: "L",
        glfw.KEY_O: "O",
        glfw.KEY_P: "P",
        glfw.KEY_U: "U",
        glfw.KEY_W: "W",
        glfw.KEY_D: "D",
        glfw.KEY_S: "S",
        glfw.KEY_A: "A",
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
        glfw.KEY_KP_DECIMAL: "KP_DECIMAL",
        glfw.KEY_KP_ENTER: "KP_ENTER",
    }
    return mapping.get(keycode)


def claw_anomaly_contact_flags(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctx: SimContext,
) -> tuple[bool, bool]:
    left_contact = False
    right_contact = False
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        other = None
        if b1 == ctx.anomaly_body_id:
            other = b2
        elif b2 == ctx.anomaly_body_id:
            other = b1
        if other == ctx.claw_left_body_id:
            left_contact = True
        elif other == ctx.claw_right_body_id:
            right_contact = True
        if left_contact and right_contact:
            break
    return left_contact, right_contact


def has_two_claw_anomaly_contact(model: mujoco.MjModel, data: mujoco.MjData, ctx: SimContext) -> bool:
    return all(claw_anomaly_contact_flags(model, data, ctx))


def attach_anomaly_to_tcp(data: mujoco.MjData, ctx: SimContext) -> dict[str, np.ndarray]:
    tcp_pos = data.site_xpos[ctx.tcp_site_id].copy()
    tcp_rot = data.site_xmat[ctx.tcp_site_id].reshape(3, 3).copy()
    obj_pos = data.xpos[ctx.anomaly_body_id].copy()
    obj_rot = data.xmat[ctx.anomaly_body_id].reshape(3, 3).copy()
    return {
        "rel_pos": tcp_rot.T @ (obj_pos - tcp_pos),
        "rel_rot": tcp_rot.T @ obj_rot,
    }


def update_attached_anomaly(data: mujoco.MjData, ctx: SimContext, attached: dict[str, np.ndarray] | None) -> None:
    if attached is None:
        return
    tcp_pos = data.site_xpos[ctx.tcp_site_id].copy()
    tcp_rot = data.site_xmat[ctx.tcp_site_id].reshape(3, 3).copy()
    obj_pos = tcp_pos + tcp_rot @ attached["rel_pos"]
    obj_rot = tcp_rot @ attached["rel_rot"]
    qadr = ctx.object_qpos_adr["anomaly_0"]
    dadr = ctx.object_dof_adr["anomaly_0"]
    data.qpos[qadr:qadr + 3] = obj_pos
    data.qpos[qadr + 3:qadr + 7] = matrix_to_quat(obj_rot)
    data.qvel[dadr:dadr + 6] = 0.0


def expert_policy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctx: SimContext,
    policy: PolicyContext,
    cfg: EpisodeConfig,
    anomaly_conveyor_speed: float,
) -> np.ndarray:
    anomaly_pos = data.xpos[ctx.anomaly_body_id].copy()
    # Objects are spawned near conveyor_end and move toward conveyor_start:
    # move_conveyor_objects decreases conveyor coordinate s, i.e. world motion
    # is along -ctx.conveyor_dir.
    predicted_pos = anomaly_pos - ctx.conveyor_dir * (anomaly_conveyor_speed * cfg.prediction_time)
    gripper_body_pos = data.xpos[ctx.hand_body_id].copy()
    pregrasp_height_delta = 0.0 if cfg.grasp_calibration is not None else (cfg.pregrasp_height - cfg.grasp_height)
    grasp_height_delta = -cfg.calibration_grasp_drop if cfg.grasp_calibration is not None else 0.0

    if policy.state == PolicyState.WAITING:
        policy.step_counter_in_state += 1
        # First move to a safe high station around the conveyor work area.
        # Do not apply the signed horizontal calibration offsets until the
        # actual moving part reaches the graspable window.
        target_pos = wait_station_gripper_body_target(data, ctx, cfg, predicted_pos, pregrasp_height_delta)
        target_quat = desired_gripper_quat_for_object(data, ctx.anomaly_body_id)
        exact_target_pos = calibrated_gripper_body_target(data, ctx, policy, cfg, predicted_pos, pregrasp_height_delta)
        pos_err = ik_body_position_residual(
            model,
            data,
            ctx.hand_body_id,
            exact_target_pos,
            ctx.arm_qpos_adr,
            ctx.arm_dof_adr,
            max_iter=100,
            damping=0.01,
        )
        arm_q = solve_ik_dls_body(
            model,
            data,
            ctx.hand_body_id,
            target_pos,
            target_quat,
            ctx.arm_qpos_adr,
            ctx.arm_dof_adr,
            max_iter=80,
            damping=0.01,
            pos_tol=1e-3,
            max_dq_per_step=0.08,
        )
        ctrl = set_ctrl_from_targets(data, ctx, arm_q, cfg.gripper_open)
        if pos_err <= cfg.reachable_pos_err:
            policy.state = PolicyState.TRACKING
            policy.step_counter_in_state = 0
            policy.target_reached_steps = 0
            policy.last_target_quat = target_quat
            print(f"[auto] target reachable: pos_residual={pos_err:.3f}m -> TRACKING")
        else:
            if policy.step_counter_in_state - policy.last_wait_log_step >= int(2.0 * PHYSICS_HZ):
                rel = anomaly_pos - ctx.conveyor_start
                s = float(np.dot(rel, ctx.conveyor_dir))
                print(
                    f"[auto] waiting/preposition: s={s:.3f}m "
                    f"exact_residual={pos_err:.3f}m station_s={cfg.wait_station_s:.3f}m "
                    f"j1={data.qpos[ctx.arm_qpos_adr[0]]:.3f}->{arm_q[0]:.3f}"
                )
                policy.last_wait_log_step = policy.step_counter_in_state
        policy.last_ctrl = ctrl
        return ctrl

    if policy.state == PolicyState.TRACKING:
        policy.step_counter_in_state += 1
        target_pos = calibrated_gripper_body_target(data, ctx, policy, cfg, predicted_pos, pregrasp_height_delta)
        target_quat = desired_gripper_quat_for_object(data, ctx.anomaly_body_id)
        arm_q = solve_ik_dls_body(
            model,
            data,
            ctx.hand_body_id,
            target_pos,
            target_quat,
            ctx.arm_qpos_adr,
            ctx.arm_dof_adr,
            max_iter=100,
            damping=0.01,
            pos_tol=5e-4,
        )
        ctrl = set_ctrl_from_targets(data, ctx, arm_q, cfg.gripper_open)
        track_err = float(np.linalg.norm(gripper_body_pos - target_pos))
        if track_err < 0.012:
            policy.target_reached_steps += 1
        else:
            policy.target_reached_steps = 0
        if policy.target_reached_steps >= int(cfg.tracking_stable_sec * PHYSICS_HZ):
            policy.state = PolicyState.DESCEND
            policy.step_counter_in_state = 0
            policy.target_reached_steps = 0
        policy.last_target_quat = target_quat
        policy.last_ctrl = ctrl
        return ctrl

    if policy.state == PolicyState.DESCEND:
        policy.step_counter_in_state += 1
        descend_steps = max(1, int(cfg.descend_sec * PHYSICS_HZ))
        descend_t = min(1.0, policy.step_counter_in_state / descend_steps)
        target_pos = calibrated_gripper_body_target(data, ctx, policy, cfg, predicted_pos, grasp_height_delta * descend_t)
        target_quat = desired_gripper_quat_for_object(data, ctx.anomaly_body_id)
        arm_q = solve_ik_dls_body(
            model,
            data,
            ctx.hand_body_id,
            target_pos,
            target_quat,
            ctx.arm_qpos_adr,
            ctx.arm_dof_adr,
            max_iter=100,
            damping=0.01,
            pos_tol=5e-4,
        )
        ctrl = set_ctrl_from_targets(data, ctx, arm_q, cfg.gripper_open)
        descend_err = float(np.linalg.norm(gripper_body_pos - target_pos))
        if descend_t >= 1.0 and descend_err < 0.010:
            policy.target_reached_steps += 1
        else:
            policy.target_reached_steps = 0
        if policy.target_reached_steps >= int(cfg.tracking_stable_sec * PHYSICS_HZ):
            policy.state = PolicyState.GRASP
            policy.step_counter_in_state = 0
            policy.target_reached_steps = 0
            # Start closing from current (open) position.
            policy.grip_target = data.qpos[model.jnt_qposadr[ctx.gripper_joint_id]] + 0.001
            policy._prev_claw_q = data.qpos[model.jnt_qposadr[ctx.gripper_joint_id]]
        elif policy.step_counter_in_state >= int(cfg.max_descend_sec * PHYSICS_HZ) and descend_err > 0.025:
            # If the arm cannot stay with the moving target, go back to the
            # calibrated high tracking pose instead of dragging low across the belt.
            policy.state = PolicyState.TRACKING
            policy.step_counter_in_state = 0
            policy.target_reached_steps = 0
        policy.last_target_quat = target_quat
        policy.last_ctrl = ctrl
        return ctrl

    if policy.state == PolicyState.GRASP:
        policy.step_counter_in_state += 1
        # Gradually close gripper — avoid overshoot that pushes the object away.
        grip_speed = 0.0003  # per step (~0.15/s at 500 Hz)
        policy.grip_target = min(cfg.gripper_closed, policy.grip_target + grip_speed)
        target_pos = calibrated_gripper_body_target(data, ctx, policy, cfg, predicted_pos, grasp_height_delta)
        target_quat = desired_gripper_quat_for_object(data, ctx.anomaly_body_id)
        arm_q = solve_ik_dls_body(
            model,
            data,
            ctx.hand_body_id,
            target_pos,
            target_quat,
            ctx.arm_qpos_adr,
            ctx.arm_dof_adr,
            max_iter=50,
            damping=0.01,
            pos_tol=5e-4,
        )
        ctrl = set_ctrl_from_targets(data, ctx, arm_q, policy.grip_target)

        # Track claw motion to detect contact stall.
        claw_q = data.qpos[model.jnt_qposadr[ctx.gripper_joint_id]]
        claw_stalled = abs(claw_q - policy._prev_claw_q) < 1e-7
        policy._prev_claw_q = claw_q

        # Check two-sided contact with anomaly.
        has_left_contact = False
        has_right_contact = False
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            if b1 == ctx.anomaly_body_id and b2 == ctx.claw_left_body_id:
                has_left_contact = True
            elif b2 == ctx.anomaly_body_id and b1 == ctx.claw_left_body_id:
                has_left_contact = True
            elif b1 == ctx.anomaly_body_id and b2 == ctx.claw_right_body_id:
                has_right_contact = True
            elif b2 == ctx.anomaly_body_id and b1 == ctx.claw_right_body_id:
                has_right_contact = True

        hold_steps = int(cfg.grasp_hold_sec * PHYSICS_HZ)
        # Success: claws stalled on object AND both fingers contact the object.
        grasped = claw_stalled and has_left_contact and has_right_contact and policy.grip_target > claw_q + 0.001
        if grasped:
            policy.grip_target = claw_q   # freeze: don't close further
            policy.grasped = True          # enable kinematic hold during transport
            policy.hold_anomaly_on_conveyor = True
            policy.state = PolicyState.LIFT_PLACE
            policy.lift_place_phase = 0
            policy.step_counter_in_state = 0
            policy.grasp_xy = anomaly_pos[:2].copy()
            policy.grasp_tcp_pos = data.xpos[ctx.hand_body_id].copy()
        elif policy.step_counter_in_state >= hold_steps:
            # Timeout — grasp failed (claws closed fully on empty space).
            policy.state = PolicyState.DONE
            policy.hold_anomaly_on_conveyor = False
        policy.last_ctrl = ctrl
        return ctrl

    if policy.state == PolicyState.LIFT_PLACE:
        policy.step_counter_in_state += 1
        target_quat = policy.last_target_quat

        # Interpolated trajectory:
        # Phase 0: lift XY=grasp, Z=grasp→0.20  (0.5s)
        # Phase 1: move XY=grasp→place, Z=0.20  (1.0s)
        # Phase 2: descend XY=place, Z=0.20→0.09 (0.5s)
        # Phase 3: release, keep position  (0.5s)
        lift_steps = int(0.5 * PHYSICS_HZ)
        move_steps = int(1.0 * PHYSICS_HZ)
        descend_steps = int(0.5 * PHYSICS_HZ)
        release_steps = int(0.5 * PHYSICS_HZ)

        if policy.lift_place_phase == 0:
            t = min(1.0, policy.step_counter_in_state / lift_steps)
            start = policy.grasp_tcp_pos if np.linalg.norm(policy.grasp_tcp_pos) > 1e-9 else gripper_body_pos
            x, y = start[0], start[1]
            z = start[2] + t * (0.20 - start[2])
        elif policy.lift_place_phase == 1:
            t = min(1.0, policy.step_counter_in_state / move_steps)
            x = policy.grasp_xy[0] + t * (ctx.place_center[0] - policy.grasp_xy[0])
            y = policy.grasp_xy[1] + t * (ctx.place_center[1] - policy.grasp_xy[1])
            z = 0.20
        elif policy.lift_place_phase == 2:
            t = min(1.0, policy.step_counter_in_state / descend_steps)
            z = 0.20 + t * (0.09 - 0.20)
            x, y = ctx.place_center[0], ctx.place_center[1]
        else:  # phase 3: hold position, gripper open
            x, y, z = ctx.place_center[0], ctx.place_center[1], 0.09

        target_pos = np.array([x, y, z], dtype=np.float64)

        arm_q = solve_ik_dls_body(
            model, data, ctx.hand_body_id, target_pos, target_quat,
            ctx.arm_qpos_adr, ctx.arm_dof_adr,
            max_iter=50, damping=0.01, pos_tol=5e-4,
        )

        # Gripper: hold at grip position during lift/move/descend, open during release.
        gripper_target = cfg.gripper_open if policy.lift_place_phase >= 3 else policy.grip_target
        if policy.lift_place_phase >= 3:
            policy.grasped = False
        ctrl = set_ctrl_from_targets(data, ctx, arm_q, gripper_target)

        # Phase transitions
        phase_steps = {0: lift_steps, 1: move_steps, 2: descend_steps, 3: release_steps}
        dist = np.linalg.norm(data.xpos[ctx.hand_body_id] - target_pos)
        if policy.lift_place_phase < 3 and (
            dist < 0.02 or policy.step_counter_in_state >= phase_steps.get(policy.lift_place_phase, 9999)
        ):
            policy.lift_place_phase += 1
            policy.step_counter_in_state = 0
        elif policy.lift_place_phase >= 3:
            if policy.step_counter_in_state >= release_steps:
                policy.state = PolicyState.DONE
                policy.hold_anomaly_on_conveyor = False
        policy.last_ctrl = ctrl
        return ctrl

    # DONE: keep posture, gripper open, release kinematic hold.
    ctrl = set_ctrl_from_targets(data, ctx, data.qpos[ctx.arm_qpos_adr].copy(), cfg.gripper_open)
    policy.grasped = False
    policy.last_ctrl = ctrl
    return ctrl


def reset_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctx: SimContext,
    rng: random.Random,
    cfg: EpisodeConfig,
) -> set[str]:
    data.qpos[:] = ctx.home_qpos
    data.qvel[:] = 0.0
    if model.na > 0:
        data.act[:] = 0.0

    data.qpos[:] = ctx.home_qpos
    arm_home = ctx.home_qpos[ctx.arm_qpos_adr].copy()
    gripper_home = float(ctx.home_qpos[model.jnt_qposadr[ctx.gripper_joint_id]])
    data.qpos[ctx.arm_qpos_adr] = arm_home
    data.qpos[model.jnt_qposadr[ctx.gripper_joint_id]] = gripper_home

    for i, aid in enumerate(ctx.arm_actuator_ids):
        data.ctrl[aid] = arm_home[i]
    data.ctrl[ctx.gripper_actuator_id] = gripper_home
    data.ctrl[ctx.gripper_right_actuator_id] = -gripper_home

    active_normals = rng.sample([f"normal_{i}" for i in range(5)], k=VISIBLE_SAMPLE_COUNT - 1)
    active = {"anomaly_0", *active_normals}

    visible_names = ["anomaly_0"] + sorted(active_normals)
    local_poses = sample_object_layout(rng, len(visible_names))
    for name, (s, lateral, obj_yaw) in zip(visible_names, local_poses):
        qadr = ctx.object_qpos_adr[name]
        dadr = ctx.object_dof_adr[name]
        conveyor_end = ctx.conveyor_start + ctx.conveyor_dir * CONVEYOR_LENGTH
        pos = conveyor_end - ctx.conveyor_dir * s + ctx.conveyor_lateral * lateral
        pos = pos.copy()
        pos[2] = OBJECT_CENTER_Z_ON_BELT + OBJECT_SETTLE_DROP_HEIGHT
        quat = object_quat_laid_down(ctx.conveyor_dir, obj_yaw)
        data.qpos[qadr : qadr + 3] = pos
        data.qpos[qadr + 3 : qadr + 7] = quat
        data.qvel[dadr : dadr + 6] = 0.0

    for i in range(5):
        name = f"normal_{i}"
        if name in active:
            continue
        qadr = ctx.object_qpos_adr[name]
        dadr = ctx.object_dof_adr[name]
        data.qpos[qadr : qadr + 7] = HIDDEN_OBJECT_QPOS
        data.qvel[dadr : dadr + 6] = 0.0

    settle_spawned_objects(model, data, ctx, cfg)
    mujoco.mj_forward(model, data)
    return active


def move_conveyor_objects(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctx: SimContext,
    active_objects: Iterable[str],
    hold_anomaly: bool,
    rng: random.Random,
    conveyor_speed: float,
) -> None:
    dt = model.opt.timestep
    conveyor_end = ctx.conveyor_start + ctx.conveyor_dir * CONVEYOR_LENGTH
    for name in active_objects:
        if name == "anomaly_0" and hold_anomaly:
            continue
        qadr = ctx.object_qpos_adr[name]
        pos = data.qpos[qadr : qadr + 3].copy()
        rel = pos - ctx.conveyor_start
        s = float(np.dot(rel, ctx.conveyor_dir))
        lat = float(np.dot(rel, ctx.conveyor_lateral))
        on_conveyor_xy = -RESPAWN_MARGIN <= s <= CONVEYOR_LENGTH + RESPAWN_MARGIN and abs(lat) <= CONVEYOR_HALF_WIDTH
        if not on_conveyor_xy or pos[2] > CONVEYOR_DRIVE_MAX_Z:
            continue

        s_new = s - conveyor_speed * dt
        target_pos = ctx.conveyor_start + ctx.conveyor_dir * s_new + ctx.conveyor_lateral * lat
        # Only drive XY (conveyor motion); Z is free for pickup.
        data.qpos[qadr] = target_pos[0]
        data.qpos[qadr + 1] = target_pos[1]

        if s_new < -RESPAWN_MARGIN:
            s_respawn = rng.uniform(*OBJECT_SPAWN_S_RANGE)
            lat_respawn = rng.uniform(*OBJECT_SPAWN_LATERAL_RANGE)
            yaw_respawn = rng.uniform(0.0, 360.0)
            respawn_pos = conveyor_end - ctx.conveyor_dir * s_respawn + ctx.conveyor_lateral * lat_respawn
            respawn_pos[2] = OBJECT_CENTER_Z_ON_BELT + OBJECT_SETTLE_DROP_HEIGHT
            data.qpos[qadr : qadr + 3] = respawn_pos
            data.qpos[qadr + 3 : qadr + 7] = object_quat_laid_down(ctx.conveyor_dir, yaw_respawn)
            dadr = ctx.object_dof_adr[name]
            data.qvel[dadr : dadr + 6] = 0.0


def render_rgb(renderer: mujoco.Renderer, data: mujoco.MjData, camera_name: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name)
    frame = renderer.render()
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def rotate_wrist_frame(frame: np.ndarray) -> np.ndarray:
    k = WRIST_ROTATE_QUARTER_TURNS_CCW % 4
    if k == 0:
        return frame
    return np.ascontiguousarray(np.rot90(frame, k=k, axes=(0, 1)))


def ensure_offscreen_buffer(model: mujoco.MjModel, width: int, height: int) -> None:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))


def make_dataset(root_dir: Path, fps: int, width: int, height: int) -> "LeRobotDataset":
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
        repo_id="Class_Products",
        root=root_dir,
        fps=fps,
        robot_type="arm_6dof_claw",
        features=features,
        use_videos=True,
        vcodec="h264",
    )


def check_success(data: mujoco.MjData, ctx: SimContext) -> bool:
    pos = data.xpos[ctx.anomaly_body_id].copy()
    in_region = np.linalg.norm(pos[:2] - ctx.place_center[:2]) <= ctx.place_radius
    above_table = pos[2] > (TABLE_Z + 0.005)
    return bool(in_region and above_table)


def anomaly_in_place_region(data: mujoco.MjData, ctx: SimContext) -> bool:
    pos = data.xpos[ctx.anomaly_body_id].copy()
    return bool(np.linalg.norm(pos[:2] - ctx.place_center[:2]) <= ctx.place_radius)


def anomaly_in_conveyor_region(data: mujoco.MjData, ctx: SimContext) -> bool:
    pos = data.xpos[ctx.anomaly_body_id].copy()
    rel = pos - ctx.conveyor_start
    s = float(np.dot(rel, ctx.conveyor_dir))
    lat = float(np.dot(rel, ctx.conveyor_lateral))
    return bool(-RESPAWN_MARGIN <= s <= CONVEYOR_LENGTH + RESPAWN_MARGIN and abs(lat) <= CONVEYOR_HALF_WIDTH)


def anomaly_landed(data: mujoco.MjData, ctx: SimContext) -> bool:
    pos = data.xpos[ctx.anomaly_body_id].copy()
    dadr = ctx.object_dof_adr["anomaly_0"]
    near_surface = TABLE_Z <= pos[2] <= (OBJECT_CENTER_Z_ON_BELT + 0.05)
    low_vertical_speed = abs(float(data.qvel[dadr + 2])) < 0.03
    return bool(near_surface and low_vertical_speed)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _trim_video_segment(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    """Trim one episode from a chunk video based on timestamps."""
    if end_s <= start_s:
        raise ValueError(f"Invalid trim range: start={start_s}, end={end_s}")

    with av.open(str(src), mode="r") as in_container:
        in_stream = in_container.streams.video[0]
        src_rate = in_stream.average_rate if in_stream.average_rate is not None else DATA_HZ
        src_fps = float(src_rate)
        start_idx = max(0, int(round(start_s * src_fps)))
        end_idx = max(start_idx + 1, int(round(end_s * src_fps)))

        if dst.exists():
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)

        with av.open(str(dst), mode="w") as out_container:
            out_stream = out_container.add_stream("libx264", rate=src_rate)
            out_stream.width = in_stream.codec_context.width
            out_stream.height = in_stream.codec_context.height
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "23", "preset": "veryfast"}

            frame_idx = 0
            for frame in in_container.decode(video=0):
                if frame_idx < start_idx:
                    frame_idx += 1
                    continue
                if frame_idx >= end_idx:
                    break

                enc_frame = frame.reformat(
                    width=out_stream.width,
                    height=out_stream.height,
                    format="yuv420p",
                )
                for packet in out_stream.encode(enc_frame):
                    out_container.mux(packet)
                frame_idx += 1

            for packet in out_stream.encode():
                out_container.mux(packet)


def _merge_resume_backup(new_root: Path, backup_root: Path) -> None:
    """Merge episodes from a backup dataset into the new dataset root."""
    import pandas as pd
    meta_dir = new_root / "meta" / "episodes"
    new_ep_files = sorted(meta_dir.glob("chunk-*/file-*.parquet"))
    if not new_ep_files:
        max_ep = -1
    else:
        new_eps = pd.concat([pd.read_parquet(p) for p in new_ep_files], ignore_index=True)
        max_ep = int(new_eps["episode_index"].max()) if "episode_index" in new_eps.columns else -1

    backup_meta = backup_root / "meta" / "episodes"
    backup_ep_files = sorted(backup_meta.glob("chunk-*/file-*.parquet"))
    if backup_ep_files:
        old_eps = pd.concat([pd.read_parquet(p) for p in backup_ep_files], ignore_index=True)
        old_eps["episode_index"] = old_eps["episode_index"] + max_ep + 1
        # Write merged episodes into new dataset
        new_chunk = meta_dir / "chunk-000"
        new_chunk.mkdir(parents=True, exist_ok=True)
        out_pq = new_chunk / "file-000.parquet"
        old_eps.to_parquet(out_pq, index=False)

    # Copy old videos with renumbered indices
    for cam in ("wrist", "global"):
        old_videos = sorted((backup_root / "videos" / cam).glob("episode_*.mp4"))
        for vf in old_videos:
            old_idx = int(vf.stem.split("_")[-1])
            new_idx = old_idx + max_ep + 1
            dst = new_root / "videos" / cam / f"episode_{new_idx:06d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vf, dst)

    # Copy old parquet data files
    old_data_dir = backup_root / "data" / "chunk-000"
    if old_data_dir.exists():
        for pf in sorted(old_data_dir.glob("episode_*.parquet")):
            old_idx = int(pf.stem.split("_")[-1])
            new_idx = old_idx + max_ep + 1
            dst = new_root / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pf, dst)

    shutil.rmtree(backup_root, ignore_errors=True)
    print(f"[collect_data] Merged backup episodes into {new_root}")


def export_legacy_layout(dataset_root: Path) -> None:
    """Create compatibility files matching legacy LeRobot-like tree shown by user."""
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        return

    all_df = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
    legacy_chunk = dataset_root / "data" / "chunk-000"
    legacy_chunk.mkdir(parents=True, exist_ok=True)
    for ep in sorted(all_df["episode_index"].unique().tolist()):
        ep_df = all_df[all_df["episode_index"] == ep].copy()
        out_pq = legacy_chunk / f"episode_{int(ep):06d}.parquet"
        ep_df.to_parquet(out_pq, index=False)

    meta_dir = dataset_root / "meta"
    episodes_parquets = sorted((meta_dir / "episodes").glob("chunk-*/file-*.parquet"))
    if episodes_parquets:
        eps_df = pd.concat([pd.read_parquet(p) for p in episodes_parquets], ignore_index=True)
    else:
        eps_df = pd.DataFrame(columns=["episode_index", "length", "tasks"])

    episodes_jsonl = meta_dir / "episodes.jsonl"
    episodes_stats_jsonl = meta_dir / "episodes_stats.jsonl"
    with episodes_jsonl.open("w", encoding="utf-8") as f_ep, episodes_stats_jsonl.open("w", encoding="utf-8") as f_stats:
        for _, row in eps_df.iterrows():
            tasks = row.get("tasks", [])
            if isinstance(tasks, np.ndarray):
                tasks = tasks.tolist()
            ep_idx = int(row.get("episode_index", 0))
            f_ep.write(
                json.dumps(
                    {
                        "episode_index": ep_idx,
                        "length": int(row.get("length", 0)),
                        "tasks": tasks,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            stat_record = {
                "episode_index": ep_idx,
                "observation_state_mean": _jsonable(row.get("stats/observation.state/mean", [])),
                "observation_state_std": _jsonable(row.get("stats/observation.state/std", [])),
                "action_mean": _jsonable(row.get("stats/action/mean", [])),
                "action_std": _jsonable(row.get("stats/action/std", [])),
            }
            f_stats.write(json.dumps(stat_record, ensure_ascii=False) + "\n")

            for cam in ("wrist", "global"):
                chunk = int(row.get(f"videos/{cam}/chunk_index", 0))
                file_index = int(row.get(f"videos/{cam}/file_index", 0))
                from_ts = float(row.get(f"videos/{cam}/from_timestamp", 0.0))
                to_ts = float(row.get(f"videos/{cam}/to_timestamp", 0.0))
                src = dataset_root / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
                dst = dataset_root / "videos" / cam / f"episode_{ep_idx:06d}.mp4"
                if src.exists():
                    try:
                        _trim_video_segment(src, dst, from_ts, to_ts)
                    except Exception:
                        # Fallback to full copy when trimming is unavailable.
                        shutil.copy2(src, dst)

    tasks_jsonl = meta_dir / "tasks.jsonl"
    tasks_parquet = meta_dir / "tasks.parquet"
    if tasks_parquet.exists():
        tdf = pd.read_parquet(tasks_parquet).reset_index()
        with tasks_jsonl.open("w", encoding="utf-8") as f:
            for _, row in tdf.iterrows():
                task_str = str(row.get("index", ""))
                task_index = int(row.get("task_index", 0))
                f.write(json.dumps({"task_index": task_index, "task": task_str}, ensure_ascii=False) + "\n")


def capture_dataset_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    dataset: LeRobotDataset,
    ctx: SimContext,
    ctrl: np.ndarray,
) -> None:
    wrist = rotate_wrist_frame(render_rgb(renderer, data, ctx.camera_wrist))
    global_view = render_rgb(renderer, data, ctx.camera_global)
    obs_state = np.concatenate(
        [data.qpos[ctx.arm_qpos_adr], np.array([data.qpos[model.jnt_qposadr[ctx.gripper_joint_id]]])]
    ).astype(np.float32)
    frame = {
        "observation.state": obs_state,
        "action": ctrl.astype(np.float32),
        "wrist": wrist,
        "global": global_view,
        "task": TASK_TEXT,
    }
    dataset.add_frame(frame)


def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    dataset: LeRobotDataset,
    ctx: SimContext,
    cfg: EpisodeConfig,
    rng: random.Random,
    viewer: object | None = None,
) -> bool:
    active_objects = reset_episode(model, data, ctx, rng, cfg)
    policy = PolicyContext()
    sample_count = 0

    for step in range(cfg.max_physics_steps):
        move_conveyor_objects(
            model, data, ctx, active_objects, policy.hold_anomaly_on_conveyor, rng, cfg.conveyor_speed
        )
        mujoco.mj_forward(model, data)
        ctrl = expert_policy(
            model=model,
            data=data,
            ctx=ctx,
            policy=policy,
            cfg=cfg,
            anomaly_conveyor_speed=cfg.conveyor_speed,
        )
        ramp_j1_toward_target(model, data, ctx, ctrl[0])
        mujoco.mj_forward(model, data)

        # Hold the grasped object only during transport. Release phase must be
        # a real free-body fall so landing can decide save / retry / discard.
        if policy.grasped and policy.state == PolicyState.LIFT_PLACE and policy.lift_place_phase < 3:
            tcp = data.site_xpos[ctx.tcp_site_id].copy()
            tcp_rot = data.site_xmat[ctx.tcp_site_id].reshape(3, 3)
            world_offset = tcp_rot @ np.array([0.0, 0.0, 0.03], dtype=np.float64)
            qadr = ctx.object_qpos_adr["anomaly_0"]
            dadr = ctx.object_dof_adr["anomaly_0"]
            data.qpos[qadr:qadr + 3] = tcp + world_offset
            data.qvel[dadr:dadr + 6] = 0
            mujoco.mj_forward(model, data)  # update contacts with new object pos

        mujoco.mj_step(model, data)

        if viewer is not None and viewer.is_running():
            viewer.sync()

        if step % SAMPLE_EVERY == 0:
            capture_dataset_frame(model, data, renderer, dataset, ctx, ctrl)
            sample_count += 1

        if policy.state == PolicyState.DONE and anomaly_landed(data, ctx):
            if anomaly_in_place_region(data, ctx):
                dataset.save_episode()
                print("[auto] saved: anomaly landed in place region")
                return True
            if anomaly_in_conveyor_region(data, ctx):
                policy = PolicyContext()
                print("[auto] retry: anomaly landed back on conveyor")
                continue
            dataset.clear_episode_buffer(delete_images=True)
            print("[auto] discarded: anomaly landed outside conveyor/place regions")
            return False
        if sample_count >= cfg.max_data_frames:
            break

    dataset.clear_episode_buffer(delete_images=True)
    return False


def handle_teleop_token(
    token: str,
    teleop: TeleopState,
    cfg: EpisodeConfig,
) -> None:
    if token == "8":
        teleop.paused = not teleop.paused
        return
    if token in {"ENTER", "KP_ENTER"}:
        teleop.save_requested = True
        return
    if token in {"0", "KP_DECIMAL"}:
        teleop.discard_requested = True
        return
    if token == "ESC":
        teleop.exit_requested = True


def apply_teleop_motion(active_tokens: Iterable[str], teleop: TeleopState, cfg: EpisodeConfig, dt: float) -> None:
    for token in active_tokens:
        if token not in TELEOP_JOINT_SPEED:
            continue
        idx, speed = TELEOP_JOINT_SPEED[token]
        if idx >= 0:
            teleop.arm_target[idx] += speed * dt
        else:
            teleop.gripper_target += speed * dt
            lo = min(cfg.gripper_open, cfg.gripper_closed)
            hi = max(cfg.gripper_open, cfg.gripper_closed)
            teleop.gripper_target = max(lo, min(hi, teleop.gripper_target))


def clamp_teleop_targets(model: mujoco.MjModel, ctx: SimContext, teleop: TeleopState) -> None:
    for i, jid in enumerate(ctx.arm_joint_ids):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            teleop.arm_target[i] = float(np.clip(teleop.arm_target[i], lo, hi))


def run_teleop_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    dataset: LeRobotDataset,
    ctx: SimContext,
    cfg: EpisodeConfig,
    rng: random.Random,
    viewer: object,
    key_queue: queue.SimpleQueue[str],
    key_poller: X11KeyPoller,
    preview: PreviewBackend,
) -> tuple[bool, bool]:
    active_objects = reset_episode(model, data, ctx, rng, cfg)
    arm_home = ctx.home_qpos[ctx.arm_qpos_adr].copy()
    # URDF qpos0=0 means gripper OPEN (claws apart). Start teleop with gripper open.
    gripper_start = cfg.gripper_open  # = 0.0
    data.qpos[ctx.arm_qpos_adr] = arm_home
    data.qpos[model.jnt_qposadr[ctx.gripper_joint_id]] = gripper_start    # 0 = open
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "claw_right")]] = 0.0
    data.qvel[ctx.arm_dof_adr] = 0.0
    data.qvel[model.jnt_dofadr[ctx.gripper_joint_id]] = 0.0
    for i, aid in enumerate(ctx.arm_actuator_ids):
        data.ctrl[aid] = arm_home[i]
    data.ctrl[ctx.gripper_actuator_id] = gripper_start     # 0 = open
    data.ctrl[ctx.gripper_right_actuator_id] = 0.0          # right also open at 0
    mujoco.mj_forward(model, data)
    teleop = TeleopState(
        arm_target=arm_home.copy(),
        gripper_target=gripper_start,
    )
    sample_count = 0
    last_log = 0.0
    saved = False
    attached: dict[str, np.ndarray] | None = None
    tap_until: dict[str, float] = {}
    tap_hold_sec = 0.08 if key_poller.available else 0.55
    while True:
        try:
            key_queue.get_nowait()
        except queue.Empty:
            break

    print(
        "[teleop] Keys: LEFT/RIGHT(J1)  UP/DOWN(J2)  "
        "numpad 1/2(J3)  numpad 4/6(J4)  numpad 5/8(J5)  numpad 7/9(J6)  numpad -/+(gripper) ; "
        "ENTER=save  numpad .=discard  8=pause  ESC=quit"
    )
    if key_poller.available:
        print("[teleop] key mode: hold + tap (X11 polling enabled)")
    else:
        print("[teleop] key mode: tap/repeat fallback (X11 polling unavailable)")

    for step in range(cfg.max_physics_steps):
        if not viewer.is_running():
            teleop.exit_requested = True
            break

        # Motion keys create a short tap pulse; X11 polling keeps the
        # same token active while the key is physically held.
        while True:
            try:
                token = key_queue.get_nowait()
            except queue.Empty:
                break
            if token in TELEOP_JOINT_SPEED:
                tap_until[token] = time.monotonic() + tap_hold_sec
            else:
                handle_teleop_token(token, teleop, cfg)
        now = time.monotonic()
        active_tokens = key_poller.pressed_tokens()
        for token, until in list(tap_until.items()):
            if until > now:
                active_tokens.add(token)
            else:
                del tap_until[token]
        apply_teleop_motion(active_tokens, teleop, cfg, model.opt.timestep)
        clamp_teleop_targets(model, ctx, teleop)

        j1_err = teleop.arm_target[0] - data.qpos[ctx.arm_qpos_adr[0]]
        data.qpos[ctx.arm_qpos_adr[0]] += np.clip(j1_err, -0.4 * model.opt.timestep, 0.4 * model.opt.timestep)

        if not teleop.paused:
            move_conveyor_objects(
                model, data, ctx, active_objects, attached is not None, rng, cfg.conveyor_speed
            )
        ctrl = teleop_policy(data, ctx, teleop)
        if attached is not None:
            mujoco.mj_forward(model, data)
            update_attached_anomaly(data, ctx, attached)
            mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        two_claw_contact = has_two_claw_anomaly_contact(model, data, ctx)
        if attached is not None and not two_claw_contact:
            attached = None
            print("[teleop grasp] released: lost two-claw contact")
        elif attached is None and teleop.gripper_target > 0.006 and two_claw_contact:
            mujoco.mj_forward(model, data)
            attached = attach_anomaly_to_tcp(data, ctx)
            print("[teleop grasp] attached anomaly_0 to tcp_site: two-claw contact")
        if attached is not None:
            mujoco.mj_forward(model, data)
            update_attached_anomaly(data, ctx, attached)
            mujoco.mj_forward(model, data)

        if step % SAMPLE_EVERY == 0:
            capture_dataset_frame(model, data, renderer, dataset, ctx, ctrl)
            sample_count += 1

        if step % SAMPLE_EVERY == 0:
            img_global = render_rgb(renderer, data, ctx.camera_global)
            img_wrist = rotate_wrist_frame(render_rgb(renderer, data, ctx.camera_wrist))
            overlay = (
                f"teleop paused={teleop.paused} frames={sample_count} "
                f"save[Enter] discard[KP .]"
            )
            preview.show(img_global, img_wrist, overlay)

        viewer.sync()

        if teleop.exit_requested:
            break

        if teleop.save_requested:
            dataset.save_episode()
            saved = True
            break
        if teleop.discard_requested:
            dataset.clear_episode_buffer(delete_images=True)
            return False, False
        if attached is None and anomaly_landed(data, ctx):
            if anomaly_in_place_region(data, ctx):
                dataset.save_episode()
                saved = True
                print("[teleop] auto-saved: anomaly landed in place region")
                break
            if not anomaly_in_conveyor_region(data, ctx):
                dataset.clear_episode_buffer(delete_images=True)
                print("[teleop] discarded: anomaly landed outside conveyor/place regions")
                return False, False

        now = time.time()
        if now - last_log > 2.0:
            last_log = now
            print(
                f"[teleop] frames={sample_count} paused={teleop.paused} "
                f"gripper={teleop.gripper_target:.3f}"
            )

        if sample_count >= cfg.max_data_frames:
            break

    if not saved:
        dataset.clear_episode_buffer(delete_images=True)
    return saved, teleop.exit_requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Sim2Sim data with MuJoCo IK expert.")
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Path to MuJoCo XML scene. If omitted, env_layout_tuned.xml is preferred.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Output dataset directory. If omitted, defaults to Lerobot_datasets/Class_Products_<timestamp>.",
    )
    parser.add_argument("--episodes", type=int, default=30, help="How many attempts to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--width", type=int, default=256, help="Render width.")
    parser.add_argument("--height", type=int, default=256, help="Render height.")
    parser.add_argument(
        "--conveyor-speed",
        type=float,
        default=DEFAULT_CONVEYOR_SPEED,
        help=f"Conveyor speed in m/s. Default: {DEFAULT_CONVEYOR_SPEED}.",
    )
    parser.add_argument(
        "--max-data-frames",
        type=int,
        default=DEFAULT_MAX_DATA_FRAMES,
        help=f"Maximum sampled frames per episode at {DATA_HZ}Hz. Default: {DEFAULT_MAX_DATA_FRAMES}.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "teleop"],
        default="auto",
        help="Collection mode: autonomous expert or manual teleoperation.",
    )
    parser.add_argument("--no-viewer", action="store_true",
                        help="Hide MuJoCo viewer window (auto mode only).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove existing dataset directory before collecting (default).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume appending episodes to an existing dataset directory.")
    parser.add_argument(
        "--preview-backend",
        choices=["auto", "cv2", "matplotlib"],
        default="auto",
        help="Camera preview backend used in teleop mode.",
    )
    parser.add_argument(
        "--grasp-calib",
        type=Path,
        default=DEFAULT_GRASP_CALIB_JSON,
        help=f"Grasp calibration JSON used by auto mode. Default: {DEFAULT_GRASP_CALIB_JSON}.",
    )
    parser.add_argument(
        "--grasp-drop",
        type=float,
        default=EpisodeConfig.calibration_grasp_drop,
        help=(
            "Auto mode descent distance from the calibrated pre-grasp TCP pose "
            f"to the closing pose, in meters. Default: {EpisodeConfig.calibration_grasp_drop}."
        ),
    )
    parser.add_argument(
        "--prediction-time",
        type=float,
        default=EpisodeConfig.prediction_time,
        help=(
            "Lead time used to predict the moving anomaly position during tracking, in seconds. "
            f"Default: {EpisodeConfig.prediction_time}."
        ),
    )
    parser.add_argument(
        "--reachable-pos-err",
        type=float,
        default=EpisodeConfig.reachable_pos_err,
        help=(
            "IK position residual threshold for entering TRACKING from WAITING, in meters. "
            f"Default: {EpisodeConfig.reachable_pos_err}."
        ),
    )
    parser.add_argument(
        "--wait-station-s",
        type=float,
        default=EpisodeConfig.wait_station_s,
        help=(
            "Conveyor coordinate used as the high preposition station in WAITING, in meters from conveyor start. "
            f"Default: {EpisodeConfig.wait_station_s}."
        ),
    )
    return parser.parse_args()


def resolve_default_xml(xml_path: Path | None) -> Path:
    if xml_path is not None:
        return xml_path
    if DEFAULT_LAYOUT_XML.exists():
        return DEFAULT_LAYOUT_XML
    if DEFAULT_CAMERA_XML.exists():
        return DEFAULT_CAMERA_XML
    return Path("env.xml")


def resolve_default_dataset_root(dataset_root: Path | None) -> Path:
    if dataset_root is not None:
        return dataset_root

    base_dir = Path("Lerobot_datasets")
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Class_Products({datetime.now().strftime('%Y%m%d_%H%M%S')})"
    candidate = base_dir / stem
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{stem}_{suffix:02d}"
        suffix += 1
    return candidate


def main() -> None:
    args = parse_args()
    args.xml = resolve_default_xml(args.xml)
    if not args.xml.exists():
        raise FileNotFoundError(f"Scene XML not found: {args.xml}")

    rng = random.Random(args.seed)
    tmp_xml = prepare_collection_xml(args.xml)
    success_count = 0
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_xml))
        model.opt.timestep = 1.0 / PHYSICS_HZ
        apply_lighting_for_debug(model)
        enforce_camera_binding(model)
        ensure_offscreen_buffer(model, args.width, args.height)
        data = mujoco.MjData(model)
        ctx = resolve_context(model, data)
        cfg = EpisodeConfig(
            width=args.width,
            height=args.height,
            conveyor_speed=args.conveyor_speed,
            max_data_frames=args.max_data_frames,
            grasp_calibration=load_grasp_calibration(args.grasp_calib),
            calibration_grasp_drop=args.grasp_drop,
            prediction_time=args.prediction_time,
            reachable_pos_err=args.reachable_pos_err,
            wait_station_s=args.wait_station_s,
        )

        renderer = mujoco.Renderer(model, width=args.width, height=args.height)

        args.dataset_root = resolve_default_dataset_root(args.dataset_root)
        _resume_backup: Path | None = None
        if args.dataset_root.exists():
            if args.resume:
                # Resume: move existing data aside, create fresh, merge back later.
                _resume_backup = args.dataset_root.with_name(args.dataset_root.name + "_resume_backup")
                if _resume_backup.exists():
                    shutil.rmtree(_resume_backup)
                shutil.move(str(args.dataset_root), str(_resume_backup))
                print(f"[collect_data] Resuming from existing dataset (backup at {_resume_backup})")
            else:
                # Default: overwrite.
                shutil.rmtree(args.dataset_root)
                print(f"[collect_data] Removed existing dataset at {args.dataset_root}")
        print(
            f"[collect_data] xml={args.xml} prepared_xml={tmp_xml} "
            f"dataset_root={args.dataset_root} mode={args.mode}"
        )
        print(
            "[collect_data] physics: cone=elliptic impratio=10, condim=6, "
            "gripper_friction=1.0, object_friction=1.0, gripper_kp=400, forcerange=±200N"
        )
        print(
            f"[collect_data] conveyor_speed={cfg.conveyor_speed:.4f}m/s "
            f"prediction_time={cfg.prediction_time:.3f}s "
            f"reachable_pos_err={cfg.reachable_pos_err:.3f}m "
            f"wait_station_s={cfg.wait_station_s:.3f}m "
            f"max_frames={cfg.max_data_frames} duration={cfg.max_data_frames / DATA_HZ:.1f}s"
        )
        if cfg.grasp_calibration is not None:
            pregrasp_height = cfg.grasp_calibration.get(
                "gripper_body_height_offset",
                cfg.grasp_calibration.get("tcp_height_offset", 0.0),
            )
            print(
                f"[collect_data] grasp_calib={args.grasp_calib} "
                f"track_frame={cfg.grasp_calibration.get('track_frame', 'legacy_tcp_site')} "
                f"pregrasp_height={float(pregrasp_height):.4f}m "
                f"grasp_drop={cfg.calibration_grasp_drop:.4f}m"
            )

        dataset = make_dataset(args.dataset_root, fps=DATA_HZ, width=args.width, height=args.height)
        viewer = None
        teleop_key_poller: X11KeyPoller | None = None
        teleop_preview: PreviewBackend | None = None
        try:
            if args.mode == "auto" and not args.no_viewer:
                viewer = mujoco.viewer.launch_passive(
                    model, data, show_left_ui=False, show_right_ui=False
                )
            teleop_key_queue: queue.SimpleQueue[str] | None = None
            if args.mode == "teleop":
                teleop_key_queue = queue.SimpleQueue()

                def on_teleop_key(keycode: int) -> None:
                    token = glfw_key_to_token(keycode)
                    if token is not None:
                        teleop_key_queue.put(token)

                viewer = mujoco.viewer.launch_passive(
                    model,
                    data,
                    key_callback=on_teleop_key,
                    show_left_ui=False,
                    show_right_ui=False,
                )
                teleop_key_poller = X11KeyPoller(TELEOP_JOINT_SPEED.keys())
                teleop_preview = PreviewBackend.create_auto(args.preview_backend)

            for ep in range(args.episodes):
                if args.mode == "auto":
                    success = run_episode(model, data, renderer, dataset, ctx, cfg, rng, viewer)
                    if viewer is not None and not viewer.is_running():
                        break
                else:
                    if (
                        viewer is None
                        or teleop_key_queue is None
                        or teleop_key_poller is None
                        or teleop_preview is None
                    ):
                        raise RuntimeError("Teleop viewer was not initialized")
                    success, should_exit = run_teleop_episode(
                        model,
                        data,
                        renderer,
                        dataset,
                        ctx,
                        cfg,
                        rng,
                        viewer,
                        teleop_key_queue,
                        teleop_key_poller,
                        teleop_preview,
                    )
                    if should_exit or not viewer.is_running():
                        break
                if success:
                    success_count += 1
                print(f"[collect_data] episode={ep:04d} success={success} total_success={success_count}")
        finally:
            if teleop_key_poller is not None:
                teleop_key_poller.close()
            if teleop_preview is not None:
                teleop_preview.close()
            if viewer is not None:
                viewer.close()
            dataset.finalize()
            renderer.close()
            export_legacy_layout(args.dataset_root)

            # Resume: merge old backup episodes into the new dataset.
            if _resume_backup is not None and _resume_backup.exists():
                _merge_resume_backup(args.dataset_root, _resume_backup)
    finally:
        Path(tmp_xml).unlink(missing_ok=True)

    print(
        f"[collect_data] finished attempts={args.episodes}, "
        f"saved={success_count}, dropped={args.episodes - success_count}"
    )


if __name__ == "__main__":
    main()
