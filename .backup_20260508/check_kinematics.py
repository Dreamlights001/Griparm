#!/usr/bin/env python3
"""Check URDF kinematics: keys 1–7 select DOF, ,/. decrease/increase it."""

import os, sys
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import glfw
import mujoco
import mujoco.viewer
import numpy as np

ARM_JOINTS = ["J_jianbu", "J_dabi", "J_Upper", "J_Lower", "J_wrist", "J_hand"]
GRIPPER_JOINT = "Claw_left"
STEP = 0.05  # radians per keypress for arm joints
GRIP_STEP = 0.003  # meters per keypress for gripper

JOINT_LABELS = [
    "1. J1  J_jianbu  (base rot Z)",
    "2. J2  J_dabi    (shoulder)",
    "3. J3  J_Upper   (elbow)",
    "4. J4  J_Lower   (forearm)",
    "5. J5  J_wrist   (wrist rot)",
    "6. J6  J_hand    (hand rot)",
    "7. Grip Claw_left (slide)",
]

model = mujoco.MjModel.from_xml_path("env.xml")
model.opt.timestep = 1.0 / 500
data = mujoco.MjData(model)

# Resolve joint and actuator IDs
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_pos") for j in ARM_JOINTS]
grip_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Claw_left_pos")
grip_right_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Claw_right_pos")
grip_right_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "claw_right")
all_joint_ids = joint_ids + [gripper_id]
all_act_ids = act_ids + [grip_act_id]

active_dof = 0  # 0-6  (0=J1, ..., 5=J6, 6=gripper)

# Freeze all joints at qpos0, but allow active DOF to move
def reset_state():
    mujoco.mj_resetData(model, data)
    # Arm joints to zero
    for i, jid in enumerate(joint_ids):
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = 0.0
        data.ctrl[act_ids[i]] = 0.0
    # Gripper fully open
    data.qpos[model.jnt_qposadr[gripper_id]] = 0.0    # open (URDF default)
    data.qpos[model.jnt_qposadr[grip_right_joint_id]] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[grip_act_id] = 0.0
    data.ctrl[grip_right_act_id] = 0.0
    mujoco.mj_forward(model, data)

reset_state()

# Key handling
keys_pressed = []

def on_key(keycode):
    token = {
        glfw.KEY_1: "1", glfw.KEY_2: "2", glfw.KEY_3: "3",
        glfw.KEY_4: "4", glfw.KEY_5: "5", glfw.KEY_6: "6", glfw.KEY_7: "7",
        glfw.KEY_COMMA: "COMMA",
        glfw.KEY_PERIOD: "PERIOD",
        glfw.KEY_R: "R",  # reset
        glfw.KEY_ESCAPE: "ESC",
        glfw.KEY_Q: "ESC",
    }.get(keycode)
    if token:
        keys_pressed.append(token)

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Kinematics Check — URDF joint-by-joint tester")
    print("=" * 55)
    for label in JOINT_LABELS:
        print(f"  {label}")
    print()
    print("  , / .   decrease / increase active joint")
    print("  R       reset all joints to zero")
    print("  Tab     switch MuJoCo camera view")
    print("  ESC/Q   quit")
    print("=" * 55 + "\n")
    with mujoco.viewer.launch_passive(
        model, data, key_callback=on_key, show_left_ui=False, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            # Process key events
            while keys_pressed:
                k = keys_pressed.pop(0)
                if k in {"ESC"}:
                    viewer.close()
                    break
                elif k == "R":
                    reset_state()
                    print("[reset] all joints → zero")
                elif k in {"1","2","3","4","5","6","7"}:
                    active_dof = int(k) - 1
                    name = (ARM_JOINTS + [GRIPPER_JOINT])[active_dof]
                    val = data.qpos[model.jnt_qposadr[all_joint_ids[active_dof]]]
                    print(f"[select] DOF {active_dof+1} = {name}  current={val:.4f}")
                elif k == "COMMA":
                    jid = all_joint_ids[active_dof]
                    aid = all_act_ids[active_dof]
                    qadr = model.jnt_qposadr[jid]
                    if active_dof < 6:
                        data.qpos[qadr] -= STEP
                        if model.jnt_limited[jid]:
                            lo, hi = model.jnt_range[jid]
                            data.qpos[qadr] = max(lo, data.qpos[qadr])
                        data.ctrl[aid] = data.qpos[qadr]
                    else:
                        lo, hi = model.jnt_range[jid]
                        new_val = max(lo, data.qpos[qadr] - GRIP_STEP)
                        data.qpos[qadr] = new_val
                        data.ctrl[grip_act_id] = new_val
                        data.qpos[model.jnt_qposadr[grip_right_joint_id]] = -new_val
                        data.ctrl[grip_right_act_id] = -new_val
                    mujoco.mj_forward(model, data)
                elif k == "PERIOD":
                    jid = all_joint_ids[active_dof]
                    aid = all_act_ids[active_dof]
                    qadr = model.jnt_qposadr[jid]
                    if active_dof < 6:
                        data.qpos[qadr] += STEP
                        if model.jnt_limited[jid]:
                            lo, hi = model.jnt_range[jid]
                            data.qpos[qadr] = min(hi, data.qpos[qadr])
                        data.ctrl[aid] = data.qpos[qadr]
                    else:
                        lo, hi = model.jnt_range[jid]
                        new_val = min(hi, data.qpos[qadr] + GRIP_STEP)
                        data.qpos[qadr] = new_val
                        data.ctrl[grip_act_id] = new_val
                        data.qpos[model.jnt_qposadr[grip_right_joint_id]] = -new_val
                        data.ctrl[grip_right_act_id] = -new_val
                    mujoco.mj_forward(model, data)

            # Hold arm at current position via position actuators
            for i, (jid, aid) in enumerate(zip(joint_ids, act_ids)):
                data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
            data.ctrl[grip_act_id] = data.qpos[model.jnt_qposadr[gripper_id]]
            data.ctrl[grip_right_act_id] = -data.qpos[model.jnt_qposadr[gripper_id]]

            mujoco.mj_step(model, data)
            viewer.sync()

        print("[quit]")
