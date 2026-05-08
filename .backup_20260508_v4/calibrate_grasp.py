#!/usr/bin/env python3
"""Grasp calibration: manually jog the arm, save the grasp pose, test grip.

Key bindings match collect_data.py teleop mode:
  LEFT/RIGHT  J1    numpad 1/2  J3    numpad 5/8  J5    numpad -  close
  UP/DOWN     J2    numpad 4/6  J4    numpad 7/9  J6    numpad +  open
  m  save → calib_grasp.json   g  test grasp   r  reset   i  info   ESC  quit
"""

import json, math, os, sys

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import glfw
import mujoco
import mujoco.viewer
import numpy as np

ARM_JOINTS = ["J_jianbu", "J_dabi", "J_Upper", "J_Lower", "J_wrist", "J_hand"]
GRIPPER_JOINT = "Claw_left"
TELEOP_ARM_STEP = 0.03
TELEOP_GRIP_STEP = 0.003
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.04
PHYSICS_HZ = 500
OUTPUT_FILE = "calib_grasp.json"


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], dtype=np.float64)


def quat_to_matrix(q):
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def apply_lighting(model):
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = [0.65, 0.65, 0.65]
    model.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
    model.vis.headlight.specular[:] = [0.15, 0.15, 0.15]


def resolve_ids(model):
    arm_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
    arm_aids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_pos") for n in ARM_JOINTS]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
    gaid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Claw_left_pos")
    graid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Claw_right_pos")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    abid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "anomaly_0")
    return arm_jids, arm_aids, gid, gaid, graid, sid, abid


def freeze_other_objects(model, data):
    for i in range(5):
        name = f"normal_{i}"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            qadr = model.jnt_qposadr[model.body_jntadr[bid]]
            dadr = model.jnt_dofadr[model.body_jntadr[bid]]
            data.qpos[qadr:qadr + 3] = [-2, -2, -1]
            data.qvel[dadr:dadr + 6] = 0


def place_anomaly_on_conveyor(model, data, rng):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "anomaly_0")
    qadr = model.jnt_qposadr[model.body_jntadr[bid]]
    dadr = model.jnt_dofadr[model.body_jntadr[bid]]
    conv_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "layout_conveyor")
    conv_pos = model.body_pos[conv_bid].copy()
    rot = quat_to_matrix(model.body_quat[conv_bid])
    forward, lateral = rot[:, 0], rot[:, 1]
    half_len = 0.6
    end = conv_pos + forward * half_len
    s = rng.uniform(0.55, 0.65)
    lat = rng.uniform(-0.05, 0.05)
    pos = end - forward * s + lateral * lat
    pos[2] = 0.03
    conveyor_yaw = math.atan2(forward[1], forward[0])
    obj_yaw_deg = rng.uniform(0, 360)
    quat = quat_from_euler_xyz(0.0, math.radians(90.0), math.radians(conveyor_yaw + obj_yaw_deg))
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = quat
    data.qvel[dadr:dadr + 6] = 0


def reset_episode(model, data, arm_jids, arm_aids, gid, gaid, graid, abid, rng):
    mujoco.mj_resetData(model, data)
    arm_target = np.zeros(6, dtype=np.float64)
    for i, jid in enumerate(arm_jids):
        data.qpos[model.jnt_qposadr[jid]] = arm_target[i]
        data.ctrl[arm_aids[i]] = arm_target[i]
    data.qpos[model.jnt_qposadr[gid]] = GRIPPER_OPEN
    data.ctrl[gaid] = GRIPPER_OPEN
    right_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "claw_right")
    data.qpos[model.jnt_qposadr[right_jid]] = 0.0
    data.ctrl[graid] = 0.0
    data.qvel[:] = 0
    freeze_other_objects(model, data)
    place_anomaly_on_conveyor(model, data, rng)
    mujoco.mj_forward(model, data)
    for _ in range(20):
        for i, aid in enumerate(arm_aids):
            data.ctrl[aid] = arm_target[i]
        data.ctrl[gaid] = GRIPPER_OPEN
        data.ctrl[graid] = 0.0
        data.qpos[model.jnt_qposadr[arm_jids[0]]] = arm_target[0]
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return arm_target


def print_help():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              Grasp Calibration Tool                              ║
╠══════════════════════════════════════════════════════════════════╣
║  LEFT/RIGHT   J1     numpad 1/2  J3     numpad 5/8  J5         ║
║  UP/DOWN      J2     numpad 4/6  J4     numpad 7/9  J6         ║
║  numpad -     close       numpad +     open                     ║
║  m  save   g  test grasp   r  reset   i  info   ESC  quit      ║
╚══════════════════════════════════════════════════════════════════╝
""")


def make_key_token_mapping():
    return {
        glfw.KEY_ESCAPE: "ESC", glfw.KEY_Q: "ESC",
        glfw.KEY_UP: "UP", glfw.KEY_DOWN: "DOWN",
        glfw.KEY_LEFT: "LEFT", glfw.KEY_RIGHT: "RIGHT",
        glfw.KEY_KP_1: "KP_1", glfw.KEY_KP_2: "KP_2",
        glfw.KEY_KP_4: "KP_4", glfw.KEY_KP_5: "KP_5",
        glfw.KEY_KP_6: "KP_6", glfw.KEY_KP_7: "KP_7",
        glfw.KEY_KP_8: "KP_8", glfw.KEY_KP_9: "KP_9",
        glfw.KEY_KP_ADD: "KP_ADD", glfw.KEY_KP_SUBTRACT: "KP_SUBTRACT",
        glfw.KEY_M: "m", glfw.KEY_G: "g", glfw.KEY_R: "r", glfw.KEY_I: "i",
    }


JOINT_MAP = {
    "LEFT": (0, +TELEOP_ARM_STEP), "RIGHT": (0, -TELEOP_ARM_STEP),
    "UP":   (1, +TELEOP_ARM_STEP), "DOWN":  (1, -TELEOP_ARM_STEP),
    "KP_1": (2, -TELEOP_ARM_STEP), "KP_2":  (2, +TELEOP_ARM_STEP),
    "KP_4": (3, +TELEOP_ARM_STEP), "KP_6":  (3, -TELEOP_ARM_STEP),
    "KP_5": (4, -TELEOP_ARM_STEP), "KP_8":  (4, +TELEOP_ARM_STEP),
    "KP_7": (5, +TELEOP_ARM_STEP), "KP_9":  (5, -TELEOP_ARM_STEP),
}


def test_grasp(model, data, arm_jids, arm_aids, gid, gaid, graid, abid):
    anomaly_qadr = model.jnt_qposadr[model.body_jntadr[abid]]
    start_z = data.qpos[anomaly_qadr + 2]
    claw_left_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Claw_Link_left")
    claw_right_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Claw_Link_right")

    print("[test] Closing gripper gradually ...")
    grip_target = data.qpos[model.jnt_qposadr[gid]] + 0.002
    has_contact = False
    for step in range(800):
        grip_target = min(GRIPPER_CLOSED, grip_target + 0.0003)
        data.ctrl[gaid] = grip_target
        data.ctrl[graid] = -grip_target
        for aid in arm_aids:
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[arm_jids[arm_aids.index(aid)]]]
        data.qpos[model.jnt_qposadr[arm_jids[0]]] = data.ctrl[arm_aids[0]]
        mujoco.mj_step(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if abid in (b1, b2) and (claw_left_bid in (b1, b2) or claw_right_bid in (b1, b2)):
                has_contact = True
                break
        if has_contact:
            break

    mujoco.mj_forward(model, data)
    grip_q = data.qpos[model.jnt_qposadr[gid]]
    print(f"  gripper at {grip_q:.4f}  contact={has_contact}")
    if not has_contact:
        print("[test] FAILED — no contact")
        return False

    print("[test] Lifting ...")
    for step in range(300):
        data.ctrl[gaid] = grip_q
        data.ctrl[graid] = -grip_q
        for i, aid in enumerate(arm_aids):
            jid = arm_jids[i]
            q = data.qpos[model.jnt_qposadr[jid]]
            if i == 1:
                q += 0.003
            data.qpos[model.jnt_qposadr[jid]] = q
            data.ctrl[aid] = q
        data.qpos[model.jnt_qposadr[arm_jids[0]]] = data.ctrl[arm_aids[0]]
        mujoco.mj_step(model, data)

    end_z = data.qpos[anomaly_qadr + 2]
    lifted = end_z > start_z + 0.005
    print(f"  object Z: {start_z:.4f} → {end_z:.4f}  lifted={lifted}")

    data.ctrl[gaid] = GRIPPER_OPEN
    data.ctrl[graid] = 0.0
    for _ in range(100):
        for aid in arm_aids:
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[arm_jids[arm_aids.index(aid)]]]
        data.qpos[model.jnt_qposadr[arm_jids[0]]] = data.ctrl[arm_aids[0]]
        mujoco.mj_step(model, data)

    return has_contact and lifted


def main():
    from pathlib import Path
    from collect_data import prepare_collection_xml
    tmp_xml = prepare_collection_xml(Path("env_layout_tuned.xml"))
    model = mujoco.MjModel.from_xml_path(str(tmp_xml))
    model.opt.timestep = 1.0 / PHYSICS_HZ
    apply_lighting(model)
    data = mujoco.MjData(model)

    arm_jids, arm_aids, gid, gaid, graid, sid, abid = resolve_ids(model)
    key_token = make_key_token_mapping()
    rng = np.random.default_rng()
    arm_target = reset_episode(model, data, arm_jids, arm_aids, gid, gaid, graid, abid, rng)
    grip_target = GRIPPER_OPEN
    print_help()

    key_queue = []

    def on_key(keycode):
        token = key_token.get(keycode)
        if token:
            key_queue.append(token)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=on_key, show_left_ui=False, show_right_ui=False
    ) as viewer:
        dt = 1.0 / PHYSICS_HZ
        HOLD_STEPS = 30  # ~60ms — bridges GLFW repeat gaps
        hold_until: dict[str, int] = {}
        one_shot = {"ESC", "r", "i", "m", "g"}
        joint_speed = {  # rad/s per token
            "LEFT": (0, +0.4), "RIGHT": (0, -0.4),
            "UP":   (1, +0.75), "DOWN":  (1, -0.75),
            "KP_1": (2, -1.0), "KP_2":  (2, +1.0),
            "KP_4": (3, +1.0), "KP_6":  (3, -1.0),
            "KP_5": (4, -1.0), "KP_8":  (4, +1.0),
            "KP_7": (5, +1.0), "KP_9":  (5, -1.0),
            "KP_ADD":      (-1, -0.02),  # grip open (speed)
            "KP_SUBTRACT": (-1, +0.02),  # grip close (speed)
        }

        step = 0
        while viewer.is_running():
            # Drain key events, refresh hold timers
            while key_queue:
                k = key_queue.pop(0)
                if k in one_shot:
                    if k == "ESC":
                        viewer.close()
                        break
                    elif k == "r":
                        rng = np.random.default_rng()
                        arm_target = reset_episode(model, data, arm_jids, arm_aids, gid, gaid, graid, abid, rng)
                        grip_target = GRIPPER_OPEN
                        print("[reset]")
                    elif k == "i":
                        tcp_pos = data.site_xpos[sid].copy()
                        obj_pos = data.xpos[abid].copy()
                        obj_rot = quat_to_matrix(data.xquat[abid])
                        obj_axis_xy = normalize(np.array([obj_rot[0, 2], obj_rot[1, 2], 0.0]))
                        tcp_rot = data.site_xmat[sid].reshape(3, 3)
                        grip_x_xy = normalize(np.array([tcp_rot[0, 0], tcp_rot[1, 0], 0.0]))
                        perp = abs(np.dot(grip_x_xy, obj_axis_xy))
                        print(f"[info] TCP={np.round(tcp_pos,3)}  obj={np.round(obj_pos,3)}  perp={perp:.4f}")
                    elif k == "m":
                        arm_q = [float(arm_target[i]) for i in range(6)]
                        obj_rot = quat_to_matrix(data.xquat[abid])
                        obj_axis_xy = normalize(np.array([obj_rot[0, 2], obj_rot[1, 2], 0.0]))
                        tcp_pos = data.site_xpos[sid].copy()
                        obj_pos = data.xpos[abid].copy()
                        rel = obj_pos - tcp_pos
                        calib = {"arm_joints": arm_q, "gripper": grip_target,
                                 "tcp_world": tcp_pos.tolist(), "object_world": obj_pos.tolist(),
                                 "object_axis_xy": obj_axis_xy.tolist(), "tcp_to_object": rel.tolist(),
                                 "arm_joint_names": ARM_JOINTS}
                        with open(OUTPUT_FILE, "w") as f:
                            json.dump(calib, f, indent=2)
                        print(f"\n[SAVED] → {OUTPUT_FILE}")
                    elif k == "g":
                        success = test_grasp(model, data, arm_jids, arm_aids, gid, gaid, graid, abid)
                        print(f"[test] grasp {'SUCCESS' if success else 'FAILED'}")
                        rng = np.random.default_rng()
                        arm_target = reset_episode(model, data, arm_jids, arm_aids, gid, gaid, graid, abid, rng)
                        grip_target = GRIPPER_OPEN
                elif k in joint_speed:
                    hold_until[k] = step + HOLD_STEPS

            # Apply continuous motion for all held tokens
            for token, until in list(hold_until.items()):
                if until > step:
                    idx, speed = joint_speed[token]
                    if idx >= 0:
                        arm_target[idx] += speed * dt
                    elif token == "KP_ADD":
                        grip_target = max(GRIPPER_OPEN, grip_target + speed * dt)
                    elif token == "KP_SUBTRACT":
                        grip_target = min(GRIPPER_CLOSED, grip_target + speed * dt)
                else:
                    del hold_until[token]

            # Clamp joints
            for idx, jid in enumerate(arm_jids):
                if model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    arm_target[idx] = float(np.clip(arm_target[idx], lo, hi))

            # Apply via actuators
            mujoco.mj_forward(model, data)
            for i, aid in enumerate(arm_aids):
                data.ctrl[aid] = arm_target[i]
            data.ctrl[gaid] = grip_target
            data.ctrl[graid] = -grip_target
            # J1: micro-step ramp (smooth as J2–J6 actuators)
            j1_err = arm_target[0] - data.qpos[model.jnt_qposadr[arm_jids[0]]]
            data.qpos[model.jnt_qposadr[arm_jids[0]]] += np.clip(j1_err, -0.4 * dt, 0.4 * dt)

            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1

    print("[quit]")


if __name__ == "__main__":
    main()
