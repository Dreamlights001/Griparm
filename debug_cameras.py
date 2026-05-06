#!/usr/bin/env python3
"""Interactive camera tuning tool for MuJoCo scene cameras."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Prefer desktop rendering for interactive Ubuntu sessions.
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import mujoco
import mujoco.viewer


DEFAULT_GLOBAL_POS = np.array([0.0, 0.0, 1.2], dtype=np.float64)
DEFAULT_GLOBAL_LOOKAT = np.array([0.5, 0.0, 0.0], dtype=np.float64)
DEFAULT_WRIST_POS = np.array([0.0, 0.06, 0.1], dtype=np.float64)
DEFAULT_WRIST_RPY_DEG = np.array([180.0, 0.0, 0.0], dtype=np.float64)
WRIST_ROTATE_QUARTER_TURNS_CCW = 0


@dataclass
class CameraState:
    active: str
    global_pos: np.ndarray
    global_lookat: np.ndarray
    wrist_pos: np.ndarray
    wrist_rpy_deg: np.ndarray
    pos_step: float = 0.005
    ang_step_deg: float = 1.0


class PreviewBackend:
    """Display global/wrist images with cv2 or matplotlib."""

    def __init__(self, mode: str):
        self.mode = mode
        self._mpl_queue: list[str] = []
        self._plt = None
        self._fig = None
        self._ax_g = None
        self._ax_w = None
        self._im_g = None
        self._im_w = None

        if mode == "cv2":
            cv2.namedWindow("global_image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("wrist_image", cv2.WINDOW_NORMAL)
            return

        if mode == "matplotlib":
            import matplotlib.pyplot as plt

            self._plt = plt
            self._fig, (self._ax_g, self._ax_w) = plt.subplots(1, 2, figsize=(10, 5))
            self._ax_g.set_title("global_image")
            self._ax_w.set_title("wrist_image")
            self._ax_g.axis("off")
            self._ax_w.axis("off")
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._im_g = self._ax_g.imshow(dummy)
            self._im_w = self._ax_w.imshow(dummy)

            def _on_key(event):
                if event.key is not None:
                    self._mpl_queue.append(event.key)

            self._fig.canvas.mpl_connect("key_press_event", _on_key)
            plt.tight_layout()
            plt.show(block=False)
            return

        raise ValueError(f"Unsupported preview mode: {mode}")

    @classmethod
    def create_auto(cls, mode: str) -> "PreviewBackend":
        if mode == "auto":
            try:
                return cls("cv2")
            except Exception:
                return cls("matplotlib")
        return cls(mode)

    def show(self, img_global: np.ndarray, img_wrist: np.ndarray, overlay_text: str) -> None:
        k = WRIST_ROTATE_QUARTER_TURNS_CCW % 4
        if k != 0:
            img_wrist = np.ascontiguousarray(np.rot90(img_wrist, k=k, axes=(0, 1)))
        if self.mode == "cv2":
            g = cv2.cvtColor(img_global, cv2.COLOR_RGB2BGR)
            w = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)
            cv2.putText(
                g,
                overlay_text,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("global_image", g)
            cv2.imshow("wrist_image", w)
            return

        # matplotlib expects RGB, images are already RGB.
        self._im_g.set_data(img_global)
        self._im_w.set_data(img_wrist)
        self._ax_g.set_xlabel(overlay_text)
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def poll_key(self) -> Optional[str]:
        if self.mode == "cv2":
            code = cv2.waitKeyEx(1)
            if code < 0:
                return None
            arrow_map = {
                2490368: "UP",
                2621440: "DOWN",
                2424832: "LEFT",
                2555904: "RIGHT",
                82: "UP",
                84: "DOWN",
                81: "LEFT",
                83: "RIGHT",
                65362: "UP",
                65364: "DOWN",
                65361: "LEFT",
                65363: "RIGHT",
            }
            if code in arrow_map:
                return arrow_map[code]
            if code == 27:
                return "ESC"
            c = chr(code & 0xFF)
            if c.isprintable():
                return c.lower()
            return None

        if not self._mpl_queue:
            return None
        key = self._mpl_queue.pop(0).lower()
        if key == "escape":
            return "ESC"
        if key in {"up", "down", "left", "right"}:
            return key.upper()
        if key in {"minus", "_"}:
            return "-"
        if key in {"equal", "plus", "+"}:
            return "="
        if len(key) == 1:
            return key
        return None

    def close(self) -> None:
        if self.mode == "cv2":
            cv2.destroyAllWindows()
            return
        if self._plt is not None:
            self._plt.close(self._fig)


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


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.array(quat_wxyz, dtype=np.float64, copy=True)
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


def matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    m = np.array(rot, dtype=np.float64, copy=False)
    trace = np.trace(m)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    return quat


def quat_to_euler_xyz_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    r = quat_to_matrix(quat_wxyz)
    sy = -r[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = math.asin(sy)
    if abs(abs(sy) - 1.0) < 1e-6:
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    else:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


def quat_from_lookat(pos: np.ndarray, lookat: np.ndarray, up_world: np.ndarray | None = None) -> np.ndarray:
    if up_world is None:
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = lookat - pos
    forward /= np.linalg.norm(forward) + 1e-12

    z_axis = -forward  # MuJoCo camera optical axis is -Z.
    x_axis = np.cross(up_world, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(up_world, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12

    rot = np.column_stack([x_axis, y_axis, z_axis])
    return matrix_to_quat(rot)


def format_vec(vec: np.ndarray) -> str:
    return " ".join(f"{v:.6f}" for v in vec)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def rodrigues_rotate(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    if np.linalg.norm(axis) < 1e-12:
        return vec.copy()
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


def rotation_from_forward(forward: np.ndarray) -> np.ndarray:
    f = normalize(forward)
    if np.linalg.norm(f) < 1e-12:
        return np.eye(3, dtype=np.float64)
    z_axis = -f
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(up, z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def rotation_from_forward_up(forward: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Build camera rotation from forward and up hint vectors."""
    f = normalize(forward)
    if np.linalg.norm(f) < 1e-12:
        return np.eye(3, dtype=np.float64)
    up = normalize(up_hint)
    if np.linalg.norm(up) < 1e-12:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = -f  # MuJoCo camera optical axis is -Z.
    x_axis = normalize(np.cross(up, z_axis))
    if np.linalg.norm(x_axis) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = normalize(np.cross(up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def global_basis(state: CameraState) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    forward = normalize(state.global_lookat - state.global_pos)
    dist = max(1e-6, float(np.linalg.norm(state.global_lookat - state.global_pos)))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right = normalize(right)
    up = normalize(np.cross(right, forward))
    return forward, right, up, dist


def move_active_camera(state: CameraState, d_forward: float, d_right: float, d_up: float) -> None:
    if state.active == "global":
        forward, right, up, _ = global_basis(state)
        delta = forward * d_forward + right * d_right + up * d_up
        state.global_pos += delta
        state.global_lookat += delta
        return

    wrist_quat = quat_from_euler_xyz(*np.radians(state.wrist_rpy_deg))
    rot = quat_to_matrix(wrist_quat)
    forward = -rot[:, 2]
    right = rot[:, 0]
    up = rot[:, 1]
    delta = forward * d_forward + right * d_right + up * d_up
    state.wrist_pos += delta


def rotate_active_camera(state: CameraState, pitch_deg: float, yaw_deg: float) -> None:
    if state.active == "global":
        forward, right, up, dist = global_basis(state)
        if abs(pitch_deg) > 0.0:
            forward = normalize(rodrigues_rotate(forward, right, math.radians(pitch_deg)))
        if abs(yaw_deg) > 0.0:
            forward = normalize(rodrigues_rotate(forward, up, math.radians(yaw_deg)))
        state.global_lookat = state.global_pos + forward * dist
        return

    # For wrist camera, rotate in camera-local axes:
    # pitch around camera-right, yaw around camera-up.
    wrist_quat = quat_from_euler_xyz(*np.radians(state.wrist_rpy_deg))
    rot = quat_to_matrix(wrist_quat)
    forward = normalize(-rot[:, 2])
    right = normalize(rot[:, 0])
    up = normalize(rot[:, 1])

    if abs(pitch_deg) > 0.0:
        a = math.radians(pitch_deg)
        forward = normalize(rodrigues_rotate(forward, right, a))
        up = normalize(rodrigues_rotate(up, right, a))
    if abs(yaw_deg) > 0.0:
        a = math.radians(yaw_deg)
        forward = normalize(rodrigues_rotate(forward, up, a))
        right = normalize(rodrigues_rotate(right, up, a))

    new_rot = rotation_from_forward_up(forward, up)
    state.wrist_rpy_deg = quat_to_euler_xyz_deg(matrix_to_quat(new_rot))


def add_marker_geom(
    viewer: mujoco.viewer.Handle,
    gtype: mujoco.mjtGeom,
    size: np.ndarray,
    pos: np.ndarray,
    mat: np.ndarray,
    rgba: np.ndarray,
) -> None:
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        int(gtype),
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(mat, dtype=np.float64).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def update_camera_markers(
    viewer: mujoco.viewer.Handle,
    data: mujoco.MjData,
    wrist_body_id: int,
    state: CameraState,
) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0

    # Global camera world pose.
    g_pos = state.global_pos.copy()
    g_forward = normalize(state.global_lookat - state.global_pos)
    g_mat = rotation_from_forward(g_forward)

    # Wrist camera world pose: attached to Hand_Link body.
    hand_pos = data.xpos[wrist_body_id].copy()
    hand_rot = data.xmat[wrist_body_id].reshape(3, 3).copy()
    w_local_rot = quat_to_matrix(quat_from_euler_xyz(*np.radians(state.wrist_rpy_deg)))
    w_world_rot = hand_rot @ w_local_rot
    w_pos = hand_pos + hand_rot @ state.wrist_pos
    w_forward = -(w_world_rot[:, 2])
    w_mat = rotation_from_forward(w_forward)

    # Colors: global=orange, wrist=cyan.
    global_color = np.array([1.0, 0.45, 0.0, 1.0], dtype=np.float32)
    wrist_color = np.array([0.0, 0.9, 1.0, 1.0], dtype=np.float32)

    # Draw camera points.
    add_marker_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.018, 0.0, 0.0], dtype=np.float64),
        pos=g_pos,
        mat=np.eye(3),
        rgba=global_color,
    )
    add_marker_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.014, 0.0, 0.0], dtype=np.float64),
        pos=w_pos,
        mat=np.eye(3),
        rgba=wrist_color,
    )

    # Draw camera direction arrows.
    add_marker_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([0.006, 0.012, 0.12], dtype=np.float64),
        pos=g_pos + g_forward * 0.06,
        mat=g_mat,
        rgba=global_color,
    )
    add_marker_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([0.0045, 0.009, 0.09], dtype=np.float64),
        pos=w_pos + w_forward * 0.045,
        mat=w_mat,
        rgba=wrist_color,
    )


def freeze_robot(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    arm_home: np.ndarray,
    gripper_home: float,
) -> None:
    for i, name in enumerate(["J_jianbu", "J_dabi", "J_Upper", "J_Lower", "J_wrist", "J_hand"]):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = arm_home[i]
            data.qvel[model.jnt_dofadr[j]] = 0.0
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Claw_left")
    if g >= 0:
        data.qpos[model.jnt_qposadr[g]] = gripper_home
        data.qvel[model.jnt_dofadr[g]] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0


def get_home_pose_from_model(model: mujoco.MjModel) -> tuple[np.ndarray, float]:
    arm_home = np.zeros(6, dtype=np.float64)
    for i, name in enumerate(["J_jianbu", "J_dabi", "J_Upper", "J_Lower", "J_wrist", "J_hand"]):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise RuntimeError(f"Missing arm joint in model: {name}")
        qadr = model.jnt_qposadr[j]
        arm_home[i] = model.qpos0[qadr]

    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Claw_left")
    if g < 0:
        raise RuntimeError("Missing gripper joint in model: Claw_left")
    gripper_home = float(model.qpos0[model.jnt_qposadr[g]])
    return arm_home, gripper_home


def apply_camera_state(
    model: mujoco.MjModel,
    global_cam_id: int,
    wrist_cam_id: int,
    wrist_body_id: int,
    state: CameraState,
) -> tuple[np.ndarray, np.ndarray]:
    global_quat = quat_from_lookat(state.global_pos, state.global_lookat)
    wrist_quat = quat_from_euler_xyz(*np.radians(state.wrist_rpy_deg))

    model.cam_mode[global_cam_id] = int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
    model.cam_targetbodyid[global_cam_id] = -1
    model.cam_bodyid[global_cam_id] = 0
    model.cam_pos[global_cam_id] = state.global_pos
    model.cam_quat[global_cam_id] = global_quat

    model.cam_mode[wrist_cam_id] = int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
    model.cam_targetbodyid[wrist_cam_id] = -1
    model.cam_bodyid[wrist_cam_id] = wrist_body_id
    model.cam_pos[wrist_cam_id] = state.wrist_pos
    model.cam_quat[wrist_cam_id] = wrist_quat
    return global_quat, wrist_quat


def print_help() -> None:
    print(
        "\n[Camera Debug Keys]\n"
        "  1 / 2 : select active camera (global / wrist)\n"
        "  \\     : toggle active camera (global <-> wrist)\n"
        "  Arrow Up/Down    : forward / backward\n"
        "  Arrow Left/Right : strafe left / right\n"
        "  z / x            : move up / down\n"
        "  u / o            : pitch + / -\n"
        "  [ / ]            : yaw left / right\n"
        "  -/=   : position step /2, *2\n"
        "  ,/.   : angle step /2, *2\n"
        "  p     : print current camera XML snippet\n"
        "  m     : save tuned cameras into --save-xml\n"
        "  ESC   : exit\n"
        "  marker: global=orange dot/arrow, wrist=cyan dot/arrow\n"
        "  scene : default preset uses MuJoCo checkerboard floor\n"
        "  note  : matplotlib backend需要预览窗口获得键盘焦点\n"
    )


def apply_lighting_for_debug(model: mujoco.MjModel) -> None:
    # Improve brightness for camera tuning.
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = np.array([0.65, 0.65, 0.65], dtype=np.float32)
    model.vis.headlight.diffuse[:] = np.array([0.65, 0.65, 0.65], dtype=np.float32)
    model.vis.headlight.specular[:] = np.array([0.15, 0.15, 0.15], dtype=np.float32)


def ensure_offscreen_buffer(model: mujoco.MjModel, width: int, height: int) -> None:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))


def remove_named_elements(parent: ET.Element, tag: str, names: set[str]) -> None:
    for child in list(parent):
        if child.tag == tag and child.attrib.get("name") in names:
            parent.remove(child)
            continue
        remove_named_elements(child, tag, names)


def build_preset_xml(src_xml: Path, preset: str) -> tuple[Path, Path | None]:
    if preset == "default":
        return src_xml, None

    tree = ET.parse(src_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Invalid XML: missing worldbody.")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Remove occluders for debug clarity.
    remove_named_elements(root, "geom", {"table"})
    remove_named_elements(root, "geom", {"anomaly_0_geom", "normal_0_geom", "normal_1_geom", "normal_2_geom", "normal_3_geom", "normal_4_geom"})
    remove_named_elements(root, "body", {"anomaly_0", "normal_0", "normal_1", "normal_2", "normal_3", "normal_4"})

    if preset == "checker":
        if asset.find(".//texture[@name='debug_checker_tex']") is None:
            ET.SubElement(
                asset,
                "texture",
                name="debug_checker_tex",
                type="2d",
                builtin="checker",
                width="512",
                height="512",
                rgb1="0.20 0.25 0.32",
                rgb2="0.12 0.14 0.18",
            )
        if asset.find(".//material[@name='debug_checker_mat']") is None:
            ET.SubElement(
                asset,
                "material",
                name="debug_checker_mat",
                texture="debug_checker_tex",
                texrepeat="20 20",
                texuniform="true",
                reflectance="0.05",
                shininess="0.1",
                specular="0.1",
            )
        if worldbody.find(".//geom[@name='debug_floor']") is None:
            ET.SubElement(
                worldbody,
                "geom",
                name="debug_floor",
                type="plane",
                pos="0 0 0",
                size="4 4 0.1",
                material="debug_checker_mat",
                friction="1.0 0.005 0.0001",
                condim="3",
            )

    fd, tmp_path = tempfile.mkstemp(prefix="debug_cam_", suffix=".xml")
    os.close(fd)
    out = Path(tmp_path)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out, out


def print_camera_snippet(state: CameraState) -> None:
    global_quat = quat_from_lookat(state.global_pos, state.global_lookat)
    wrist_quat = quat_from_euler_xyz(*np.radians(state.wrist_rpy_deg))
    print("\n[Camera XML Snippet]")
    print(
        f'<camera name="global" mode="fixed" pos="{format_vec(state.global_pos)}" '
        f'quat="{format_vec(global_quat)}" fovy="45"/>'
    )
    print(
        f'<camera name="wrist" mode="fixed" pos="{format_vec(state.wrist_pos)}" '
        f'quat="{format_vec(wrist_quat)}" fovy="58"/>'
    )
    print(f"[global_lookat] {format_vec(state.global_lookat)}")


def find_named_with_parent(root: ET.Element, tag: str, name: str) -> tuple[ET.Element | None, ET.Element | None]:
    for child in list(root):
        if child.tag == tag and child.attrib.get("name") == name:
            return root, child
        parent, found = find_named_with_parent(child, tag, name)
        if found is not None:
            return parent, found
    return None, None


def save_camera_xml(src_xml: Path, dst_xml: Path, state: CameraState) -> None:
    tree = ET.parse(src_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    hand_body = root.find(".//body[@name='Hand_Link']")
    if worldbody is None or hand_body is None:
        raise RuntimeError("Failed to find worldbody or Hand_Link in XML.")

    global_parent, global_cam = find_named_with_parent(root, "camera", "global")
    wrist_parent, wrist_cam = find_named_with_parent(root, "camera", "wrist")

    if global_cam is None:
        global_cam = ET.SubElement(worldbody, "camera", name="global")
        global_parent = worldbody
    if wrist_cam is None:
        wrist_cam = ET.SubElement(hand_body, "camera", name="wrist")
        wrist_parent = hand_body

    # Enforce expected attachment:
    # - global camera stays in worldbody (fixed in world)
    # - wrist camera stays under Hand_Link (moves with hand)
    if global_parent is not worldbody and global_parent is not None:
        global_parent.remove(global_cam)
        worldbody.append(global_cam)
    if wrist_parent is not hand_body and wrist_parent is not None:
        wrist_parent.remove(wrist_cam)
        hand_body.append(wrist_cam)

    global_quat = quat_from_lookat(state.global_pos, state.global_lookat)
    wrist_quat = quat_from_euler_xyz(*np.radians(state.wrist_rpy_deg))

    global_cam.attrib["mode"] = "fixed"
    global_cam.attrib["pos"] = format_vec(state.global_pos)
    global_cam.attrib["quat"] = format_vec(global_quat)
    global_cam.attrib.pop("target", None)

    wrist_cam.attrib["mode"] = "fixed"
    wrist_cam.attrib["pos"] = format_vec(state.wrist_pos)
    wrist_cam.attrib["quat"] = format_vec(wrist_quat)
    wrist_cam.attrib.pop("target", None)

    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)
    print(f"[debug_cameras] saved tuned xml => {dst_xml}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive camera debug for MuJoCo scene.")
    parser.add_argument("--xml", type=Path, default=Path("env.xml"), help="Scene xml path.")
    parser.add_argument(
        "--save-xml",
        type=Path,
        default=Path("env_camera_tuned.xml"),
        help="Path to save tuned XML when pressing m.",
    )
    parser.add_argument("--width", type=int, default=512, help="Render width for preview windows.")
    parser.add_argument("--height", type=int, default=512, help="Render height for preview windows.")
    parser.add_argument(
        "--preview-backend",
        choices=["auto", "cv2", "matplotlib"],
        default="auto",
        help="Image preview backend. auto: cv2 first, fallback matplotlib.",
    )
    parser.add_argument(
        "--scene-preset",
        choices=["checker", "clean", "default"],
        default="checker",
        help="checker: MuJoCo checkerboard floor (recommended). clean: no floor.",
    )
    parser.add_argument("--show-left-ui", action="store_true", help="Show MuJoCo left UI panel.")
    parser.add_argument("--show-right-ui", action="store_true", help="Show MuJoCo right UI panel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.xml.exists():
        raise FileNotFoundError(f"XML not found: {args.xml}")

    model_xml, temp_xml = build_preset_xml(args.xml, args.scene_preset)
    try:
        model = mujoco.MjModel.from_xml_path(str(model_xml))
        apply_lighting_for_debug(model)
        ensure_offscreen_buffer(model, args.width, args.height)
        data = mujoco.MjData(model)

        global_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "global")
        wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        wrist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Hand_Link")
        if global_cam_id < 0 or wrist_cam_id < 0 or wrist_body_id < 0:
            raise RuntimeError("Missing required names: global camera / wrist camera / Hand_Link body.")

        arm_home, gripper_home = get_home_pose_from_model(model)
        freeze_robot(data, model, arm_home, gripper_home)
        mujoco.mj_forward(model, data)

        state = CameraState(
            active="global",
            global_pos=DEFAULT_GLOBAL_POS.copy(),
            global_lookat=DEFAULT_GLOBAL_LOOKAT.copy(),
            wrist_pos=DEFAULT_WRIST_POS.copy(),
            wrist_rpy_deg=DEFAULT_WRIST_RPY_DEG.copy(),
        )

        print_help()
        preview = PreviewBackend.create_auto(args.preview_backend)
        print(
            "[debug_cameras] Started. "
            f"backend={preview.mode}, scene_preset={args.scene_preset}, "
            f"left_ui={args.show_left_ui}, right_ui={args.show_right_ui}"
        )

        renderer = mujoco.Renderer(model, width=args.width, height=args.height)

        last_log = 0.0
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=args.show_left_ui,
            show_right_ui=args.show_right_ui,
        ) as viewer:
            while viewer.is_running():
                freeze_robot(data, model, arm_home, gripper_home)
                apply_camera_state(model, global_cam_id, wrist_cam_id, wrist_body_id, state)
                mujoco.mj_forward(model, data)
                update_camera_markers(viewer, data, wrist_body_id, state)
                viewer.sync()

                renderer.update_scene(data, camera="global")
                img_global = renderer.render()
                renderer.update_scene(data, camera="wrist")
                img_wrist = renderer.render()

                overlay = f"active={state.active} pos_step={state.pos_step:.4f} ang_step={state.ang_step_deg:.2f}"
                preview.show(img_global, img_wrist, overlay)

                key = preview.poll_key()
                if key is None:
                    key = ""

                if key == "ESC":
                    break
                elif key == "1":
                    state.active = "global"
                elif key == "2":
                    state.active = "wrist"
                elif key == "\\":
                    state.active = "wrist" if state.active == "global" else "global"
                elif key == "-":
                    state.pos_step = max(0.0005, state.pos_step * 0.5)
                elif key == "=":
                    state.pos_step = min(0.05, state.pos_step * 2.0)
                elif key == ",":
                    state.ang_step_deg = max(0.1, state.ang_step_deg * 0.5)
                elif key == ".":
                    state.ang_step_deg = min(45.0, state.ang_step_deg * 2.0)
                elif key == "p":
                    print_camera_snippet(state)
                elif key == "m":
                    save_camera_xml(args.xml, args.save_xml, state)
                elif key == "UP":
                    move_active_camera(state, d_forward=state.pos_step, d_right=0.0, d_up=0.0)
                elif key == "DOWN":
                    move_active_camera(state, d_forward=-state.pos_step, d_right=0.0, d_up=0.0)
                elif key == "LEFT":
                    move_active_camera(state, d_forward=0.0, d_right=-state.pos_step, d_up=0.0)
                elif key == "RIGHT":
                    move_active_camera(state, d_forward=0.0, d_right=state.pos_step, d_up=0.0)
                elif key == "z":
                    move_active_camera(state, d_forward=0.0, d_right=0.0, d_up=state.pos_step)
                elif key == "x":
                    move_active_camera(state, d_forward=0.0, d_right=0.0, d_up=-state.pos_step)
                elif key == "u":
                    rotate_active_camera(state, pitch_deg=state.ang_step_deg, yaw_deg=0.0)
                elif key == "o":
                    rotate_active_camera(state, pitch_deg=-state.ang_step_deg, yaw_deg=0.0)
                elif key == "[":
                    rotate_active_camera(state, pitch_deg=0.0, yaw_deg=state.ang_step_deg)
                elif key == "]":
                    rotate_active_camera(state, pitch_deg=0.0, yaw_deg=-state.ang_step_deg)

                now = time.time()
                if now - last_log > 1.0:
                    last_log = now
                    print(
                        "[debug_cameras] "
                        f"active={state.active}, "
                        f"global_pos={np.round(state.global_pos,4)}, "
                        f"global_lookat={np.round(state.global_lookat,4)}, "
                        f"wrist_pos={np.round(state.wrist_pos,4)}, "
                        f"wrist_rpy_deg={np.round(state.wrist_rpy_deg,2)}"
                    )

        renderer.close()
        preview.close()
        print("[debug_cameras] exited.")
    finally:
        if temp_xml is not None and temp_xml.exists():
            temp_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
