#!/usr/bin/env python3
"""Preview the production line: objects flow along conveyor, arm in home pose.

Use this to verify the layout, camera angles, and object flow before
running full data collection with collect_data.py.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import mujoco
import mujoco.viewer

from debug_cameras import (
    PreviewBackend,
    apply_lighting_for_debug,
    freeze_robot,
    get_home_pose_from_model,
    WRIST_ROTATE_QUARTER_TURNS_CCW,
)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
PHYSICS_HZ = 500
CONVEYOR_SPEED = 0.05  # m/s
OBJECT_CLEARANCE = 0.06  # minimum separation between objects on belt
SPAWN_MAX_ATTEMPTS = 200
SCENE_SPAWN_RETRIES = 20
SETTLE_STEPS = 160
SETTLE_DROP_HEIGHT = 0.004

CONVEYOR_BODY = "layout_conveyor"
CONVEYOR_GEOM = "layout_conveyor_geom"
CONVEYOR_COLLISION_GEOM = "layout_conveyor_collision"
CONVEYOR_HALF_LENGTH = 0.6
CONVEYOR_HALF_WIDTH = 0.11
CONVEYOR_HALF_HEIGHT = 0.01
LATERAL_MARGIN = 0.02

BELT_SURFACE_Z = CONVEYOR_HALF_HEIGHT + 0.002  # body Z + thin-plane half-height
OBJECT_Z_ON_BELT = BELT_SURFACE_Z + 0.018  # surface + object half-height

# object spawn region in conveyor-local frame (s = distance along belt)
SPAWN_S_RANGE = (0.0, 0.20)
SPAWN_LATERAL_RANGE = (
    -(CONVEYOR_HALF_WIDTH - LATERAL_MARGIN),
    (CONVEYOR_HALF_WIDTH - LATERAL_MARGIN),
)

ANOMALY_NAME = "anomaly_0"
NORMAL_NAMES = [f"normal_{i}" for i in range(5)]
VISIBLE_COUNT = 4  # anomaly + 3 normals

HIDDEN_POS = np.array([-2.0, -2.0, -1.0], dtype=np.float64)
HIDDEN_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
RESPAWN_MARGIN = 0.06


# ---------------------------------------------------------------------------
# XML scene preparation (checkerboard floor + collision fix)
# ---------------------------------------------------------------------------

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


def prepare_scene_xml(src: Path) -> Path:
    """Load XML, add checkerboard floor, fix collision attributes, write temp."""
    tree = ET.parse(src)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Invalid XML: missing worldbody.")

    # --- checkerboard floor ---
    if asset.find(".//texture[@name='checker_tex']") is None:
        ET.SubElement(
            asset,
            "texture",
            name="checker_tex",
            type="2d",
            builtin="checker",
            width="512",
            height="512",
            rgb1="0.20 0.25 0.32",
            rgb2="0.12 0.14 0.18",
        )
    if asset.find(".//material[@name='checker_mat']") is None:
        ET.SubElement(
            asset,
            "material",
            name="checker_mat",
            texture="checker_tex",
            texrepeat="20 20",
            texuniform="true",
            reflectance="0.05",
            shininess="0.1",
            specular="0.1",
        )
    if worldbody.find(".//geom[@name='checker_floor']") is None:
        ET.SubElement(
            worldbody,
            "geom",
            name="checker_floor",
            type="plane",
            pos="0 0 0",
            size="4 4 0.1",
            material="checker_mat",
            friction="1.0 0.005 0.0001",
            condim="3",
        )

    # --- remove table geom (debug_layout removes it, we do the same) ---
    for geom in list(worldbody):
        if geom.tag == "geom" and geom.attrib.get("name") == "table":
            worldbody.remove(geom)

    # --- conveyor: visual box + separate collision plane for stable preview ---
    conveyor_body = root.find(f".//body[@name='{CONVEYOR_BODY}']")
    conv_geom = root.find(f".//geom[@name='{CONVEYOR_GEOM}']")
    if conv_geom is not None:
        conv_geom.attrib["contype"] = "0"
        conv_geom.attrib["conaffinity"] = "0"
        conv_geom.attrib["size"] = f"{CONVEYOR_HALF_LENGTH} {CONVEYOR_HALF_WIDTH} 0.002"
        conv_geom.attrib["rgba"] = "0.40 0.40 0.40 0.90"
    if conveyor_body is not None and conveyor_body.find(f"./geom[@name='{CONVEYOR_COLLISION_GEOM}']") is None:
        ET.SubElement(
            conveyor_body,
            "geom",
            name=CONVEYOR_COLLISION_GEOM,
            type="plane",
            pos="0 0 0.002",
            size="2.0 2.0 0.1",
            rgba="0 0 0 0",
            contype="1",
            conaffinity="1",
            friction="0.8 0.005 0.0001",
            condim="3",
        )

    # --- object geoms: ensure collision is on ---
    for geom in root.findall(".//geom"):
        name = geom.attrib.get("name", "")
        if name.startswith("anomaly_") or name.startswith("normal_"):
            geom.attrib.setdefault("contype", "1")
            geom.attrib.setdefault("conaffinity", "1")

    _indent(root)
    fd, tmp_path = tempfile.mkstemp(prefix="preview_", suffix=".xml")
    os.close(fd)
    out = Path(tmp_path)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy],
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
    q /= np.linalg.norm(q) + 1e-12
    w, x, y, z = q
    return np.array(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        dtype=np.float64,
    )


def conveyor_frame(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (center, forward, lateral, start, end) in world frame.

    start = center - forward * half_length   (one end)
    end   = center + forward * half_length   (other end)
    Objects spawn near *end* and move toward *start* (-forward).
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONVEYOR_BODY)
    center = model.body_pos[bid].copy()
    rot = mat_from_quat_wxyz(model.body_quat[bid])
    forward = rot[:, 0].copy()
    lateral = rot[:, 1].copy()
    start = center - forward * CONVEYOR_HALF_LENGTH
    end = center + forward * CONVEYOR_HALF_LENGTH
    return center, forward, lateral, start, end


def object_laid_quat(conveyor_forward: np.ndarray, obj_yaw_deg: float) -> np.ndarray:
    conveyor_yaw_deg = math.degrees(math.atan2(conveyor_forward[1], conveyor_forward[0]))
    base = quat_from_euler_xyz(0.0, math.radians(90.0), math.radians(conveyor_yaw_deg + obj_yaw_deg))
    # Flip every part around its own cylinder axis so the defect side is exposed upward.
    flipped = quat_mul(base, quat_from_euler_xyz(0.0, 0.0, math.pi))
    return flipped / (np.linalg.norm(flipped) + 1e-12)


def random_object_layout(rng: np.random.Generator, count: int) -> list[tuple[float, float, float]]:
    """Generate (s, lateral, yaw_deg) with collision avoidance."""
    samples: list[tuple[float, float, float]] = []
    for _ in range(500):
        if len(samples) >= count:
            break
        s = rng.uniform(*SPAWN_S_RANGE)
        lat = rng.uniform(*SPAWN_LATERAL_RANGE)
        if any(math.hypot(s - ps, lat - pl) < OBJECT_CLEARANCE for ps, pl, _ in samples):
            continue
        samples.append((s, lat, rng.uniform(0.0, 360.0)))
    while len(samples) < count:
        idx = len(samples)
        frac = idx / max(1, count - 1)
        s = SPAWN_S_RANGE[0] + frac * (SPAWN_S_RANGE[1] - SPAWN_S_RANGE[0])
        lat = SPAWN_LATERAL_RANGE[0] if idx % 2 == 0 else SPAWN_LATERAL_RANGE[1]
        samples.append((s, lat, rng.uniform(0.0, 360.0)))
    return samples


def set_object_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    jid: int,
    pos: np.ndarray,
    quat: np.ndarray,
    zero_velocity: bool = True,
) -> None:
    qadr = model.jnt_qposadr[jid]
    dadr = model.jnt_dofadr[jid]
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = quat
    if zero_velocity:
        data.qvel[dadr:dadr + 6] = 0.0


def hide_object(model: mujoco.MjModel, data: mujoco.MjData, jid: int) -> None:
    set_object_pose(model, data, jid, HIDDEN_POS, HIDDEN_QUAT, zero_velocity=True)


def body_id_from_joint(model: mujoco.MjModel, jid: int) -> int:
    qadr = model.jnt_qposadr[jid]
    for bid in range(model.nbody):
        jadr = int(model.body_jntadr[bid])
        jnum = int(model.body_jntnum[bid])
        if jnum <= 0:
            continue
        for offset in range(jnum):
            body_jid = jadr + offset
            if int(model.jnt_qposadr[body_jid]) == qadr:
                return bid
    raise RuntimeError(f"Cannot resolve body id from joint id {jid}.")


def object_has_bad_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_body_id: int,
    allowed_static_body_ids: set[int],
    other_object_body_ids: set[int],
) -> bool:
    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = int(model.geom_bodyid[int(contact.geom1)])
        body2 = int(model.geom_bodyid[int(contact.geom2)])
        if object_body_id not in (body1, body2):
            continue
        other = body2 if body1 == object_body_id else body1
        if other == object_body_id:
            continue
        if other in allowed_static_body_ids:
            continue
        if other in other_object_body_ids:
            return True
        return True
    return False


def scene_has_object_overlap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_body_ids: set[int],
    allowed_static_body_ids: set[int],
) -> bool:
    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = int(model.geom_bodyid[int(contact.geom1)])
        body2 = int(model.geom_bodyid[int(contact.geom2)])
        body1_is_object = body1 in object_body_ids
        body2_is_object = body2 in object_body_ids
        if body1_is_object and body2_is_object and body1 != body2:
            return True
        if body1_is_object and body2 not in allowed_static_body_ids and body2 != body1:
            return True
        if body2_is_object and body1 not in allowed_static_body_ids and body1 != body2:
            return True
    return False


def settle_objects(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_home: np.ndarray,
    gripper_home: np.ndarray,
    steps: int = SETTLE_STEPS,
) -> None:
    for _ in range(steps):
        freeze_robot(data, model, arm_home, gripper_home)
        mujoco.mj_step(model, data)
    freeze_robot(data, model, arm_home, gripper_home)
    mujoco.mj_forward(model, data)


def current_layouts_on_conveyor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obj_joints: dict[str, int],
    visible: list[str],
    skip_name: str,
    start: np.ndarray,
    forward: np.ndarray,
    lateral: np.ndarray,
) -> list[tuple[float, float, float]]:
    layouts: list[tuple[float, float, float]] = []
    for name in visible:
        if name == skip_name:
            continue
        jid = obj_joints[name]
        qadr = model.jnt_qposadr[jid]
        pos = data.qpos[qadr:qadr + 3]
        rel = pos - start
        s = float(np.dot(rel, forward))
        lat = float(np.dot(rel, lateral))
        layouts.append((s, lat, 0.0))
    return layouts


def sample_nonoverlap_respawn(
    rng: np.random.Generator,
    occupied_layouts: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    for _ in range(SPAWN_MAX_ATTEMPTS):
        s = rng.uniform(*SPAWN_S_RANGE)
        lat = rng.uniform(*SPAWN_LATERAL_RANGE)
        if any(math.hypot(s - ps, lat - pl) < OBJECT_CLEARANCE for ps, pl, _ in occupied_layouts):
            continue
        return s, lat, rng.uniform(0.0, 360.0)
    return (
        rng.uniform(*SPAWN_S_RANGE),
        rng.uniform(*SPAWN_LATERAL_RANGE),
        rng.uniform(0.0, 360.0),
    )


def sample_valid_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    jid: int,
    object_body_id: int,
    other_object_body_ids: set[int],
    forward: np.ndarray,
    lateral: np.ndarray,
    end: np.ndarray,
    rng: np.random.Generator,
    allowed_static_body_ids: set[int],
    occupied_layouts: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SPAWN_MAX_ATTEMPTS):
        s = rng.uniform(*SPAWN_S_RANGE)
        lat = rng.uniform(*SPAWN_LATERAL_RANGE)
        if any(math.hypot(s - ps, lat - pl) < OBJECT_CLEARANCE for ps, pl, _ in occupied_layouts):
            continue
        yaw_deg = rng.uniform(0.0, 360.0)
        pos = end - forward * s + lateral * lat
        pos[2] = OBJECT_Z_ON_BELT + SETTLE_DROP_HEIGHT
        quat = object_laid_quat(forward, yaw_deg)
        set_object_pose(model, data, jid, pos, quat, zero_velocity=True)
        mujoco.mj_forward(model, data)
        if object_has_bad_contacts(model, data, object_body_id, allowed_static_body_ids, other_object_body_ids):
            continue
        occupied_layouts.append((s, lat, yaw_deg))
        return pos.copy(), quat.copy()
    return None


# ---------------------------------------------------------------------------
# Object management
# ---------------------------------------------------------------------------

def resolve_object_joints(model: mujoco.MjModel) -> dict[str, int]:
    joints = {}
    for name in [ANOMALY_NAME] + NORMAL_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
        if jid >= 0:
            joints[name] = jid
    return joints


def spawn_objects(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obj_joints: dict[str, int],
    rng: np.random.Generator,
    arm_home: np.ndarray,
    gripper_home: np.ndarray,
) -> list[str]:
    """Place visible objects near conveyor *end* (they will flow toward start)."""
    _, forward, lateral, _start, end = conveyor_frame(model)
    visible_names = [ANOMALY_NAME] + NORMAL_NAMES[:VISIBLE_COUNT - 1]
    visible_set = set(visible_names)
    object_body_ids = {name: body_id_from_joint(model, jid) for name, jid in obj_joints.items()}
    conveyor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONVEYOR_BODY)
    world_body_id = 0
    allowed_static_body_ids = {world_body_id}
    if conveyor_body_id >= 0:
        allowed_static_body_ids.add(conveyor_body_id)

    for name, jid in obj_joints.items():
        if name not in visible_set:
            hide_object(model, data, jid)

    for _ in range(SCENE_SPAWN_RETRIES):
        occupied_layouts: list[tuple[float, float, float]] = []
        for name, jid in obj_joints.items():
            if name in visible_set:
                continue
            hide_object(model, data, jid)

        placed_ok = True
        for name in visible_names:
            jid = obj_joints[name]
            other_ids = {object_body_ids[n] for n in visible_names if n != name}
            pose = sample_valid_pose(
                model,
                data,
                jid,
                object_body_ids[name],
                other_ids,
                forward,
                lateral,
                end,
                rng,
                allowed_static_body_ids,
                occupied_layouts,
            )
            if pose is None:
                placed_ok = False
                break

        if not placed_ok:
            continue

        settle_objects(model, data, arm_home, gripper_home)
        visible_body_ids = {object_body_ids[name] for name in visible_names}
        if scene_has_object_overlap(model, data, visible_body_ids, allowed_static_body_ids):
            continue

        freeze_robot(data, model, arm_home, gripper_home)
        mujoco.mj_forward(model, data)
        return visible_names

    layouts = random_object_layout(rng, len(visible_names))
    for name, (s, lat, yaw_deg) in zip(visible_names, layouts):
        jid = obj_joints[name]
        pos = end - forward * s + lateral * lat
        pos[2] = OBJECT_Z_ON_BELT + SETTLE_DROP_HEIGHT
        set_object_pose(model, data, jid, pos, object_laid_quat(forward, yaw_deg), zero_velocity=True)

    for name, jid in obj_joints.items():
        if name not in visible_set:
            hide_object(model, data, jid)
    settle_objects(model, data, arm_home, gripper_home)
    freeze_robot(data, model, arm_home, gripper_home)
    mujoco.mj_forward(model, data)
    return visible_names


def move_conveyor_objects(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obj_joints: dict[str, int],
    visible: list[str],
    dt: float,
    rng: np.random.Generator,
) -> list[str]:
    """Move objects kinematically along the belt without introducing collision walls."""
    _, forward, lateral, start, end = conveyor_frame(model)
    move_dir = -forward  # objects flow from end to start
    updated = list(visible)

    for name in visible:
        jid = obj_joints[name]
        qadr = model.jnt_qposadr[jid]
        pos = data.qpos[qadr:qadr + 3].copy()
        rel = pos - start
        s = float(np.dot(rel, forward))
        lat = float(np.dot(rel, lateral))

        s_new = s - CONVEYOR_SPEED * dt
        target_pos = start + forward * s_new + lateral * lat
        data.qpos[qadr:qadr + 3] = np.array([target_pos[0], target_pos[1], pos[2]], dtype=np.float64)

        # check if object has passed the start end
        obj_pos = data.qpos[qadr:qadr + 3].copy()
        to_obj = obj_pos - start
        s_now = float(np.dot(to_obj, forward))
        if s_now < -RESPAWN_MARGIN:
            # respawn at end
            occupied_layouts = current_layouts_on_conveyor(
                model, data, obj_joints, visible, name, start, forward, lateral
            )
            s_new, lat_new, yaw_new = sample_nonoverlap_respawn(rng, occupied_layouts)
            new_pos = end - forward * s_new + lateral * lat_new
            new_pos[2] = OBJECT_Z_ON_BELT + SETTLE_DROP_HEIGHT
            data.qpos[qadr:qadr + 3] = new_pos
            data.qpos[qadr + 3:qadr + 7] = object_laid_quat(forward, yaw_new)
            dadr = model.jnt_dofadr[jid]
            data.qvel[dadr:dadr + 6] = 0.0
    return updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview production line layout and object flow.")
    p.add_argument(
        "--xml", type=Path, default=None,
        help="Scene XML. Defaults to env_layout_tuned.xml.",
    )
    p.add_argument("--width", type=int, default=512, help="Camera preview width.")
    p.add_argument("--height", type=int, default=512, help="Camera preview height.")
    p.add_argument(
        "--preview-backend", choices=["auto", "cv2", "matplotlib"], default="auto",
        help="Camera preview backend.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for object placement.")
    p.add_argument("--show-left-ui", action="store_true", help="Show MuJoCo left UI panel.")
    p.add_argument("--show-right-ui", action="store_true", help="Show MuJoCo right UI panel.")
    p.add_argument("--no-camera-windows", action="store_true", help="Hide camera preview windows.")
    return p.parse_args()


def print_help() -> None:
    print(
        "\n[Preview Keys]\n"
        "  ESC / Q      : quit\n"
        "  R            : respawn objects with new random layout\n"
        "  Space        : pause / resume conveyor\n"
        "  C            : toggle camera preview windows\n"
        "  . / ,        : speed up / slow down conveyor\n"
        "  Tab          : switch MuJoCo camera (free / global / wrist)\n"
        "  Left-drag    : orbit   Right-drag : pan   Scroll : zoom\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.xml is None:
        args.xml = Path("env_layout_tuned.xml")
    if not args.xml.exists():
        raise FileNotFoundError(f"Scene XML not found: {args.xml}")

    # Build a temp XML with checkerboard floor and collision fixes.
    tmp_xml = prepare_scene_xml(args.xml)
    renderer = None
    preview = None
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_xml))
        model.opt.timestep = 1.0 / PHYSICS_HZ
        apply_lighting_for_debug(model)
        data = mujoco.MjData(model)

        for name in ("Hand_Link", CONVEYOR_BODY, "layout_place_region"):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
                raise RuntimeError(f"Missing body: {name}")

        arm_home, gripper_home = get_home_pose_from_model(model)
        freeze_robot(data, model, arm_home, gripper_home)

        obj_joints = resolve_object_joints(model)
        if ANOMALY_NAME not in obj_joints:
            raise RuntimeError("anomaly_0_free joint not found in XML.")

        rng = np.random.default_rng(args.seed)
        visible = spawn_objects(model, data, obj_joints, rng, arm_home, gripper_home)
        mujoco.mj_forward(model, data)

        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)

        preview = None if args.no_camera_windows else PreviewBackend.create_auto(args.preview_backend)
        renderer = None
        renderer = mujoco.Renderer(model, width=args.width, height=args.height)

        speed_mult = 1.0
        paused = False
        last_log = 0.0

        print_help()
        print(
            f"[preview] xml={args.xml}  seed={args.seed}  "
            f"conveyor_speed={CONVEYOR_SPEED:.3f} m/s  "
            f"visible_objects={VISIBLE_COUNT}  backend={preview.mode if preview else 'none'}"
        )

        with mujoco.viewer.launch_passive(
            model, data,
            show_left_ui=args.show_left_ui,
            show_right_ui=args.show_right_ui,
        ) as viewer:
            while viewer.is_running():
                freeze_robot(data, model, arm_home, gripper_home)

                if not paused:
                    dt = model.opt.timestep * speed_mult
                    move_conveyor_objects(model, data, obj_joints, visible, dt, rng)
                mujoco.mj_forward(model, data)
                viewer.sync()

                if preview is not None:
                    renderer.update_scene(data, camera="global")
                    img_global = renderer.render()
                    renderer.update_scene(data, camera="wrist")
                    img_wrist = renderer.render()
                    k = WRIST_ROTATE_QUARTER_TURNS_CCW % 4
                    if k != 0:
                        img_wrist = np.ascontiguousarray(np.rot90(img_wrist, k=k, axes=(0, 1)))

                    overlay = (
                        f"conveyor={'PAUSED' if paused else 'RUN'}  "
                        f"speed=x{speed_mult:.1f}  "
                        f"visible={len(visible)}"
                    )
                    preview.show(img_global, img_wrist, overlay)

                key = preview.poll_key() if preview is not None else ""
                if key in ("ESC", "q"):
                    break
                elif key == "r":
                    rng = np.random.default_rng()
                    visible = spawn_objects(model, data, obj_joints, rng, arm_home, gripper_home)
                    print("[preview] respawned objects")
                elif key == " ":
                    paused = not paused
                    print(f"[preview] conveyor {'paused' if paused else 'running'}")
                elif key == "c":
                    if preview is not None:
                        preview.close()
                        preview = None
                        print("[preview] camera windows hidden")
                    else:
                        preview = PreviewBackend.create_auto(args.preview_backend)
                        print(f"[preview] camera windows shown (backend={preview.mode})")
                elif key == ".":
                    speed_mult = min(5.0, speed_mult + 0.5)
                    print(f"[preview] speed x{speed_mult:.1f}")
                elif key == ",":
                    speed_mult = max(0.1, speed_mult - 0.5)
                    print(f"[preview] speed x{speed_mult:.1f}")

                now = time.time()
                if now - last_log > 2.0:
                    last_log = now
                    qadr = model.jnt_qposadr[obj_joints[ANOMALY_NAME]]
                    apos = data.qpos[qadr:qadr + 3]
                    print(
                        f"[preview] anomaly_pos=({apos[0]:.3f},{apos[1]:.3f},{apos[2]:.3f}) "
                        f"paused={paused} speed=x{speed_mult:.1f}"
                    )

    finally:
        if renderer is not None:
            renderer.close()
        if preview is not None:
            preview.close()
        if tmp_xml.exists():
            tmp_xml.unlink(missing_ok=True)
        print("[preview] exited.")


if __name__ == "__main__":
    main()
