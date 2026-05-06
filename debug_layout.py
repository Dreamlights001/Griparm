#!/usr/bin/env python3
"""Interactive conveyor/place-region layout tool with live camera previews."""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Prefer desktop rendering for interactive Ubuntu sessions.
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import mujoco
import mujoco.viewer

from debug_cameras import (
    PreviewBackend,
    WRIST_ROTATE_QUARTER_TURNS_CCW,
    add_marker_geom,
    apply_lighting_for_debug,
    freeze_robot,
    get_home_pose_from_model,
    quat_from_euler_xyz,
    quat_to_matrix,
    remove_named_elements,
    rotation_from_forward,
)


CONVEYOR_BODY_NAME = "layout_conveyor"
CONVEYOR_GEOM_NAME = "layout_conveyor_geom"
CONVEYOR_START_SITE = "layout_conveyor_start"
CONVEYOR_END_SITE = "layout_conveyor_end"
PLACE_BODY_NAME = "layout_place_region"
PLACE_GEOM_NAME = "layout_place_region_geom"
PLACE_CENTER_SITE = "layout_place_center"
DEFAULT_LAYOUT_XML = Path("env_layout_tuned.xml")
DEFAULT_CAMERA_XML = Path("env_camera_tuned.xml")

CONVEYOR_LENGTH = 1.2
CONVEYOR_HALF_LENGTH = CONVEYOR_LENGTH * 0.5
CONVEYOR_HALF_WIDTH = 0.11
CONVEYOR_HALF_HEIGHT = 0.01
CONVEYOR_DEFAULT_POS = np.array([0.20, 0.00], dtype=np.float64)
CONVEYOR_DEFAULT_YAW_DEG = 0.0

PLACE_RADIUS = 0.15
PLACE_HALF_HEIGHT = 0.002
PLACE_DEFAULT_POS = np.array([0.25, -0.25], dtype=np.float64)
PLACE_DEFAULT_YAW_DEG = 0.0

ANOMALY_PROXY_HALF_SIZE = np.array([0.035, 0.018, 0.018], dtype=np.float64)
NORMAL_PROXY_HALF_SIZE = np.array([0.018, 0.018, 0.018], dtype=np.float64)
OBJECT_RANDOM_X_RANGE = (0.0, CONVEYOR_LENGTH / 6.0)
OBJECT_RANDOM_Y_RANGE = (-0.065, 0.065)
OBJECT_CLEARANCE = 0.045
OBJECT_Z_ON_BELT = CONVEYOR_HALF_HEIGHT + 0.018
VISIBLE_SAMPLE_COUNT = 4
HIDDEN_OBJECT_POS = np.array([-2.0, -2.0, -1.0], dtype=np.float64)


@dataclass
class LayoutState:
    active: str
    conveyor_pos_xy: np.ndarray
    conveyor_yaw_deg: float
    place_pos_xy: np.ndarray
    place_yaw_deg: float
    pos_step: float = 0.01
    ang_step_deg: float = 2.0
    object_seed: int = 7


@dataclass
class LayoutIds:
    conveyor_body_id: int
    place_body_id: int
    anomaly_joint_id: int
    normal_joint_ids: list[int]


def yaw_deg_from_quat(quat_wxyz: np.ndarray) -> float:
    rot = quat_to_matrix(quat_wxyz)
    return math.degrees(math.atan2(rot[1, 0], rot[0, 0]))


def body_quat_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    return quat_from_euler_xyz(0.0, 0.0, math.radians(yaw_deg))


def object_quat_laid_down(conveyor_yaw_deg: float, object_yaw_deg: float) -> np.ndarray:
    # Lay cylinder-like products on their side, then randomize heading in the belt plane.
    return quat_from_euler_xyz(0.0, math.radians(90.0), math.radians(conveyor_yaw_deg + object_yaw_deg))


def body_pose_3d(xy: np.ndarray, z: float) -> np.ndarray:
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def ensure_layout_elements(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Invalid XML: missing worldbody.")

    conveyor_body = root.find(f".//body[@name='{CONVEYOR_BODY_NAME}']")
    if conveyor_body is None:
        conveyor_body = ET.SubElement(
            worldbody,
            "body",
            name=CONVEYOR_BODY_NAME,
            pos="0.20 0.00 0.010",
            quat="1 0 0 0",
        )
        ET.SubElement(
            conveyor_body,
            "geom",
            name=CONVEYOR_GEOM_NAME,
            type="box",
            size=f"{CONVEYOR_HALF_LENGTH:.6f} {CONVEYOR_HALF_WIDTH:.6f} {CONVEYOR_HALF_HEIGHT:.6f}",
            rgba="0.52 0.52 0.52 0.95",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            conveyor_body,
            "site",
            name=CONVEYOR_START_SITE,
            pos=f"{-CONVEYOR_HALF_LENGTH:.6f} 0 0",
            size="0.01",
            rgba="0.6 0.6 0.6 1",
        )
        ET.SubElement(
            conveyor_body,
            "site",
            name=CONVEYOR_END_SITE,
            pos=f"{CONVEYOR_HALF_LENGTH:.6f} 0 0",
            size="0.01",
            rgba="0.6 0.6 0.6 1",
        )

    place_body = root.find(f".//body[@name='{PLACE_BODY_NAME}']")
    if place_body is None:
        place_body = ET.SubElement(
            worldbody,
            "body",
            name=PLACE_BODY_NAME,
            pos="0.25 -0.25 0.002",
            quat="1 0 0 0",
        )
        ET.SubElement(
            place_body,
            "geom",
            name=PLACE_GEOM_NAME,
            type="cylinder",
            size=f"{PLACE_RADIUS:.6f} {PLACE_HALF_HEIGHT:.6f}",
            rgba="1.0 0.35 0.68 0.45",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            place_body,
            "site",
            name=PLACE_CENTER_SITE,
            pos="0 0 0",
            size="0.01",
            rgba="1.0 0.35 0.68 1",
        )


def replace_object_geoms_with_debug_proxies(root: ET.Element) -> None:
    """Use small primitive proxies in the layout tool to avoid oversized STL occlusion."""
    anomaly_geom = root.find(".//geom[@name='anomaly_0_geom']")
    if anomaly_geom is not None:
        anomaly_geom.attrib.clear()
        anomaly_geom.attrib.update(
            {
                "name": "anomaly_0_geom",
                "type": "box",
                "size": "0.035 0.018 0.018",
                "rgba": "1.0 0.58 0.18 1.0",
                "contype": "0",
                "conaffinity": "0",
            }
        )

    for i in range(5):
        geom = root.find(f".//geom[@name='normal_{i}_geom']")
        if geom is None:
            continue
        geom.attrib.clear()
        geom.attrib.update(
            {
                "name": f"normal_{i}_geom",
                "type": "box",
                "size": "0.018 0.018 0.018",
                "rgba": "0.95 0.93 0.82 1.0",
                "contype": "0",
                "conaffinity": "0",
            }
        )


def read_layout_state(root: ET.Element) -> LayoutState:
    ensure_layout_elements(root)

    conveyor_body = root.find(f".//body[@name='{CONVEYOR_BODY_NAME}']")
    place_body = root.find(f".//body[@name='{PLACE_BODY_NAME}']")
    assert conveyor_body is not None
    assert place_body is not None

    conveyor_pos = np.fromstring(conveyor_body.attrib.get("pos", "0.20 0.00 0.010"), sep=" ", dtype=np.float64)
    place_pos = np.fromstring(place_body.attrib.get("pos", "0.25 -0.25 0.002"), sep=" ", dtype=np.float64)

    conveyor_quat = np.fromstring(conveyor_body.attrib.get("quat", "1 0 0 0"), sep=" ", dtype=np.float64)
    place_quat = np.fromstring(place_body.attrib.get("quat", "1 0 0 0"), sep=" ", dtype=np.float64)

    return LayoutState(
        active="conveyor",
        conveyor_pos_xy=conveyor_pos[:2] if conveyor_pos.size >= 2 else CONVEYOR_DEFAULT_POS.copy(),
        conveyor_yaw_deg=yaw_deg_from_quat(conveyor_quat) if conveyor_quat.size == 4 else CONVEYOR_DEFAULT_YAW_DEG,
        place_pos_xy=place_pos[:2] if place_pos.size >= 2 else PLACE_DEFAULT_POS.copy(),
        place_yaw_deg=yaw_deg_from_quat(place_quat) if place_quat.size == 4 else PLACE_DEFAULT_YAW_DEG,
    )


def replace_child_body(worldbody: ET.Element, new_body: ET.Element) -> None:
    name = new_body.attrib.get("name")
    if not name:
        return
    for child in list(worldbody):
        if child.tag == "body" and child.attrib.get("name") == name:
            worldbody.remove(child)
            break
    worldbody.append(copy.deepcopy(new_body))


def merge_layout_from_xml(root: ET.Element, layout_xml: Path | None) -> None:
    if layout_xml is None or not layout_xml.exists():
        return
    layout_root = ET.parse(layout_xml).getroot()
    src_worldbody = layout_root.find("worldbody")
    dst_worldbody = root.find("worldbody")
    if src_worldbody is None or dst_worldbody is None:
        return
    for name in (CONVEYOR_BODY_NAME, PLACE_BODY_NAME):
        body = src_worldbody.find(f".//body[@name='{name}']")
        if body is not None:
            replace_child_body(dst_worldbody, body)


def build_preset_xml(
    src_xml: Path,
    preset: str,
    object_visual: str,
    layout_xml: Path | None,
) -> tuple[Path, Path | None, LayoutState]:
    tree = ET.parse(src_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Invalid XML: missing worldbody.")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    remove_named_elements(root, "geom", {"table"})
    merge_layout_from_xml(root, layout_xml)
    ensure_layout_elements(root)
    if object_visual == "proxy":
        replace_object_geoms_with_debug_proxies(root)
    state = read_layout_state(root)

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

    fd, tmp_path = tempfile.mkstemp(prefix="debug_layout_", suffix=".xml")
    os.close(fd)
    out = Path(tmp_path)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out, out, state


def resolve_layout_ids(model: mujoco.MjModel) -> LayoutIds:
    conveyor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONVEYOR_BODY_NAME)
    place_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PLACE_BODY_NAME)
    anomaly_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "anomaly_0_free")
    normal_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"normal_{i}_free") for i in range(5)]

    if conveyor_body_id < 0 or place_body_id < 0 or anomaly_joint_id < 0 or any(j < 0 for j in normal_joint_ids):
        raise RuntimeError("Missing layout bodies or sample-object freejoints in XML.")

    return LayoutIds(
        conveyor_body_id=conveyor_body_id,
        place_body_id=place_body_id,
        anomaly_joint_id=anomaly_joint_id,
        normal_joint_ids=normal_joint_ids,
    )


def apply_layout_state(model: mujoco.MjModel, state: LayoutState, ids: LayoutIds) -> None:
    model.body_pos[ids.conveyor_body_id] = body_pose_3d(state.conveyor_pos_xy, CONVEYOR_HALF_HEIGHT)
    model.body_quat[ids.conveyor_body_id] = body_quat_from_yaw_deg(state.conveyor_yaw_deg)

    model.body_pos[ids.place_body_id] = body_pose_3d(state.place_pos_xy, PLACE_HALF_HEIGHT)
    model.body_quat[ids.place_body_id] = body_quat_from_yaw_deg(state.place_yaw_deg)


def sample_demo_object_layout(state: LayoutState, count: int) -> list[tuple[float, float, float]]:
    rng = random.Random(state.object_seed)
    samples: list[tuple[float, float, float]] = []
    attempts = 0
    while len(samples) < count and attempts < 500:
        attempts += 1
        lx = rng.uniform(*OBJECT_RANDOM_X_RANGE)
        ly = rng.uniform(*OBJECT_RANDOM_Y_RANGE)
        if any(math.hypot(lx - sx, ly - sy) < OBJECT_CLEARANCE for sx, sy, _ in samples):
            continue
        samples.append((lx, ly, rng.uniform(0.0, 360.0)))

    while len(samples) < count:
        idx = len(samples)
        frac = idx / max(1, count - 1)
        lx = OBJECT_RANDOM_X_RANGE[0] + frac * (OBJECT_RANDOM_X_RANGE[1] - OBJECT_RANDOM_X_RANGE[0])
        ly = OBJECT_RANDOM_Y_RANGE[0] if idx % 2 == 0 else OBJECT_RANDOM_Y_RANGE[1]
        samples.append((lx, ly, rng.uniform(0.0, 360.0)))
    return samples


def set_demo_objects_on_conveyor(model: mujoco.MjModel, data: mujoco.MjData, state: LayoutState, ids: LayoutIds) -> None:
    yaw = math.radians(state.conveyor_yaw_deg)
    rot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    belt_center = np.array([state.conveyor_pos_xy[0], state.conveyor_pos_xy[1], CONVEYOR_HALF_HEIGHT], dtype=np.float64)
    belt_start = belt_center - rot[:, 0] * CONVEYOR_HALF_LENGTH
    object_joint_ids = [ids.anomaly_joint_id] + ids.normal_joint_ids
    visible_joint_ids = [ids.anomaly_joint_id] + ids.normal_joint_ids[: VISIBLE_SAMPLE_COUNT - 1]
    local_poses = sample_demo_object_layout(state, len(visible_joint_ids))
    for joint_id, (s, lateral, obj_yaw) in zip(visible_joint_ids, local_poses):
        qadr = model.jnt_qposadr[joint_id]
        dadr = model.jnt_dofadr[joint_id]
        world = belt_start + rot[:, 0] * s + rot[:, 1] * lateral
        world = world.copy()
        world[2] = OBJECT_Z_ON_BELT
        data.qpos[qadr : qadr + 3] = world
        data.qpos[qadr + 3 : qadr + 7] = object_quat_laid_down(state.conveyor_yaw_deg, obj_yaw)
        data.qvel[dadr : dadr + 6] = 0.0

    hidden_joint_ids = [jid for jid in object_joint_ids if jid not in visible_joint_ids]
    for joint_id in hidden_joint_ids:
        qadr = model.jnt_qposadr[joint_id]
        dadr = model.jnt_dofadr[joint_id]
        data.qpos[qadr : qadr + 3] = HIDDEN_OBJECT_POS
        data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        data.qvel[dadr : dadr + 6] = 0.0


def move_active_layout(state: LayoutState, dx: float, dy: float) -> None:
    if state.active == "conveyor":
        state.conveyor_pos_xy += np.array([dx, dy], dtype=np.float64)
        return
    state.place_pos_xy += np.array([dx, dy], dtype=np.float64)


def rotate_active_layout(state: LayoutState, delta_yaw_deg: float) -> None:
    if state.active == "conveyor":
        state.conveyor_yaw_deg += delta_yaw_deg
        return
    state.place_yaw_deg += delta_yaw_deg


def update_layout_markers(viewer: mujoco.viewer.Handle, state: LayoutState) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0

    conveyor_forward = np.array(
        [math.cos(math.radians(state.conveyor_yaw_deg)), math.sin(math.radians(state.conveyor_yaw_deg)), 0.0],
        dtype=np.float64,
    )
    conveyor_center = body_pose_3d(state.conveyor_pos_xy, CONVEYOR_HALF_HEIGHT)
    conveyor_mat = rotation_from_forward(conveyor_forward)
    conveyor_color = np.array([0.78, 0.78, 0.78, 1.0], dtype=np.float32)

    place_forward = np.array(
        [math.cos(math.radians(state.place_yaw_deg)), math.sin(math.radians(state.place_yaw_deg)), 0.0],
        dtype=np.float64,
    )
    place_center = body_pose_3d(state.place_pos_xy, PLACE_HALF_HEIGHT)
    place_mat = rotation_from_forward(place_forward)
    place_color = np.array([1.0, 0.35, 0.68, 1.0], dtype=np.float32)

    add_marker_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([0.008, 0.016, 0.16], dtype=np.float64),
        pos=conveyor_center + conveyor_forward * 0.08,
        mat=conveyor_mat,
        rgba=conveyor_color,
    )
    add_marker_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.012, 0.0, 0.0], dtype=np.float64),
        pos=place_center,
        mat=np.eye(3),
        rgba=place_color,
    )
    add_marker_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([0.006, 0.012, 0.09], dtype=np.float64),
        pos=place_center + place_forward * 0.045,
        mat=place_mat,
        rgba=place_color,
    )


def print_layout_snippet(state: LayoutState) -> None:
    conveyor_quat = body_quat_from_yaw_deg(state.conveyor_yaw_deg)
    place_quat = body_quat_from_yaw_deg(state.place_yaw_deg)
    print("\n[Layout XML Snippet]")
    print(
        f'<body name="{CONVEYOR_BODY_NAME}" pos="{state.conveyor_pos_xy[0]:.6f} {state.conveyor_pos_xy[1]:.6f} {CONVEYOR_HALF_HEIGHT:.6f}" '
        f'quat="{" ".join(f"{v:.6f}" for v in conveyor_quat)}"> ... </body>'
    )
    print(
        f'<body name="{PLACE_BODY_NAME}" pos="{state.place_pos_xy[0]:.6f} {state.place_pos_xy[1]:.6f} {PLACE_HALF_HEIGHT:.6f}" '
        f'quat="{" ".join(f"{v:.6f}" for v in place_quat)}"> ... </body>'
    )


def save_layout_xml(src_xml: Path, dst_xml: Path, state: LayoutState) -> None:
    tree = ET.parse(src_xml)
    root = tree.getroot()
    ensure_layout_elements(root)

    conveyor_body = root.find(f".//body[@name='{CONVEYOR_BODY_NAME}']")
    place_body = root.find(f".//body[@name='{PLACE_BODY_NAME}']")
    if conveyor_body is None or place_body is None:
        raise RuntimeError("Failed to find layout bodies in XML.")

    conveyor_body.attrib["pos"] = f"{state.conveyor_pos_xy[0]:.6f} {state.conveyor_pos_xy[1]:.6f} {CONVEYOR_HALF_HEIGHT:.6f}"
    conveyor_body.attrib["quat"] = " ".join(f"{v:.6f}" for v in body_quat_from_yaw_deg(state.conveyor_yaw_deg))

    place_body.attrib["pos"] = f"{state.place_pos_xy[0]:.6f} {state.place_pos_xy[1]:.6f} {PLACE_HALF_HEIGHT:.6f}"
    place_body.attrib["quat"] = " ".join(f"{v:.6f}" for v in body_quat_from_yaw_deg(state.place_yaw_deg))

    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)
    print(f"[debug_layout] saved tuned xml => {dst_xml}")


def ensure_offscreen_buffer(model: mujoco.MjModel, width: int, height: int) -> None:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))


def print_help() -> None:
    print(
        "\n[Layout Debug Keys]\n"
        "  \\     : toggle active object (conveyor / place-region)\n"
        "  Arrow Up/Down    : move +X / -X\n"
        "  Arrow Left/Right : move +Y / -Y\n"
        "  z / x            : yaw + / - around world Z\n"
        "  -/=   : position step /2, *2\n"
        "  ,/.   : angle step /2, *2\n"
        "  p     : print current layout XML snippet\n"
        "  m     : save tuned layout into --save-xml\n"
        "  ESC   : exit\n"
        "  colors: conveyor=gray, place-region=pink\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive conveyor/place layout debug.")
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Scene xml path. If omitted, env_layout_tuned.xml is used when available.",
    )
    parser.add_argument(
        "--layout-xml",
        type=Path,
        default=None,
        help="Optional XML to load saved layout_conveyor/layout_place_region from.",
    )
    parser.add_argument(
        "--save-xml",
        type=Path,
        default=Path("env_layout_tuned.xml"),
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
        "--object-visual",
        choices=["stl", "proxy"],
        default="stl",
        help="How to render objects on the conveyor during layout debug.",
    )
    parser.add_argument(
        "--scene-preset",
        choices=["checker", "clean"],
        default="checker",
        help="checker: MuJoCo checkerboard floor. clean: no floor.",
    )
    parser.add_argument("--show-left-ui", action="store_true", help="Show MuJoCo left UI panel.")
    parser.add_argument("--show-right-ui", action="store_true", help="Show MuJoCo right UI panel.")
    return parser.parse_args()


def resolve_xml_args(args: argparse.Namespace) -> None:
    if args.xml is None:
        if DEFAULT_LAYOUT_XML.exists():
            args.xml = DEFAULT_LAYOUT_XML
        elif DEFAULT_CAMERA_XML.exists():
            args.xml = DEFAULT_CAMERA_XML
        else:
            args.xml = Path("env.xml")

    if args.layout_xml is None and DEFAULT_LAYOUT_XML.exists() and args.xml.resolve() != DEFAULT_LAYOUT_XML.resolve():
        args.layout_xml = DEFAULT_LAYOUT_XML


def main() -> None:
    args = parse_args()
    resolve_xml_args(args)
    if not args.xml.exists():
        raise FileNotFoundError(f"XML not found: {args.xml}")

    model_xml, temp_xml, state = build_preset_xml(args.xml, args.scene_preset, args.object_visual, args.layout_xml)
    preview = None
    renderer = None
    try:
        model = mujoco.MjModel.from_xml_path(str(model_xml))
        apply_lighting_for_debug(model)
        ensure_offscreen_buffer(model, args.width, args.height)
        data = mujoco.MjData(model)
        ids = resolve_layout_ids(model)

        arm_home, gripper_home = get_home_pose_from_model(model)
        preview = PreviewBackend.create_auto(args.preview_backend)
        renderer = mujoco.Renderer(model, width=args.width, height=args.height)

        print_help()
        print(
            "[debug_layout] Started. "
            f"xml={args.xml}, layout_xml={args.layout_xml}, "
            f"backend={preview.mode}, scene_preset={args.scene_preset}, "
            f"object_visual={args.object_visual}, "
            f"left_ui={args.show_left_ui}, right_ui={args.show_right_ui}, "
            f"wrist_rotate_k={WRIST_ROTATE_QUARTER_TURNS_CCW}"
        )

        last_log = 0.0
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=args.show_left_ui,
            show_right_ui=args.show_right_ui,
        ) as viewer:
            while viewer.is_running():
                freeze_robot(data, model, arm_home, gripper_home)
                apply_layout_state(model, state, ids)
                set_demo_objects_on_conveyor(model, data, state, ids)
                mujoco.mj_forward(model, data)
                update_layout_markers(viewer, state)
                viewer.sync()

                renderer.update_scene(data, camera="global")
                img_global = renderer.render()
                renderer.update_scene(data, camera="wrist")
                img_wrist = renderer.render()

                overlay = (
                    f"active={state.active} pos_step={state.pos_step:.3f} "
                    f"ang_step={state.ang_step_deg:.1f} "
                    f"conveyor=({state.conveyor_pos_xy[0]:.3f},{state.conveyor_pos_xy[1]:.3f},{state.conveyor_yaw_deg:.1f}) "
                    f"place=({state.place_pos_xy[0]:.3f},{state.place_pos_xy[1]:.3f},{state.place_yaw_deg:.1f})"
                )
                preview.show(img_global, img_wrist, overlay)

                key = preview.poll_key() or ""
                if key == "ESC":
                    break
                elif key == "\\":
                    state.active = "place" if state.active == "conveyor" else "conveyor"
                elif key == "-":
                    state.pos_step = max(0.001, state.pos_step * 0.5)
                elif key == "=":
                    state.pos_step = min(0.10, state.pos_step * 2.0)
                elif key == ",":
                    state.ang_step_deg = max(0.2, state.ang_step_deg * 0.5)
                elif key == ".":
                    state.ang_step_deg = min(45.0, state.ang_step_deg * 2.0)
                elif key == "p":
                    print_layout_snippet(state)
                elif key == "m":
                    save_layout_xml(args.xml, args.save_xml, state)
                elif key == "UP":
                    move_active_layout(state, dx=state.pos_step, dy=0.0)
                elif key == "DOWN":
                    move_active_layout(state, dx=-state.pos_step, dy=0.0)
                elif key == "LEFT":
                    move_active_layout(state, dx=0.0, dy=state.pos_step)
                elif key == "RIGHT":
                    move_active_layout(state, dx=0.0, dy=-state.pos_step)
                elif key == "z":
                    rotate_active_layout(state, delta_yaw_deg=state.ang_step_deg)
                elif key == "x":
                    rotate_active_layout(state, delta_yaw_deg=-state.ang_step_deg)

                now = time.time()
                if now - last_log > 1.0:
                    last_log = now
                    print(
                        "[debug_layout] "
                        f"active={state.active}, "
                        f"conveyor_xy={np.round(state.conveyor_pos_xy,4)}, "
                        f"conveyor_yaw_deg={state.conveyor_yaw_deg:.2f}, "
                        f"place_xy={np.round(state.place_pos_xy,4)}, "
                        f"place_yaw_deg={state.place_yaw_deg:.2f}"
                    )
    finally:
        if renderer is not None:
            renderer.close()
        if preview is not None:
            preview.close()
        print("[debug_layout] exited.")
        if temp_xml is not None and temp_xml.exists():
            temp_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
