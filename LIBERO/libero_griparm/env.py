from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


ARM_JOINTS = ["J_jianbu", "J_dabi", "J_Upper", "J_Lower", "J_wrist", "J_hand"]
GRIPPER_JOINT = "Claw_left"
RIGHT_GRIPPER_JOINT = "claw_right"
CONVEYOR_LENGTH = 1.2
CONVEYOR_HALF_LENGTH = CONVEYOR_LENGTH * 0.5
CONVEYOR_HALF_WIDTH = 0.11
OBJECT_SPAWN_S_RANGE = (0.0, CONVEYOR_LENGTH / 6.0)
OBJECT_SPAWN_LATERAL_RANGE = (-0.09, 0.09)
OBJECT_CENTER_Z_ON_BELT = 0.028
OBJECT_SETTLE_DROP_HEIGHT = 0.004
RESPAWN_MARGIN = 0.06
HIDDEN_OBJECT_QPOS = np.array([-2.0, -2.0, -1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@dataclass
class EnvIds:
    arm_joint_ids: list[int]
    arm_qpos_adr: np.ndarray
    arm_actuator_ids: list[int]
    gripper_joint_id: int
    right_gripper_joint_id: int
    gripper_actuator_id: int
    right_gripper_actuator_id: int
    anomaly_body_id: int
    normal_body_ids: list[int]
    object_qpos_adr: dict[str, int]
    object_dof_adr: dict[str, int]
    conveyor_body_id: int
    place_body_id: int
    tcp_site_id: int


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def mat_from_quat(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def object_quat_laid_down(conveyor_dir: np.ndarray, yaw_deg: float) -> np.ndarray:
    conveyor_yaw = math.atan2(conveyor_dir[1], conveyor_dir[0])
    return quat_from_euler_xyz(0.0, math.radians(90.0), conveyor_yaw + math.radians(yaw_deg))


class GriparmSortingEnv:
    """Standalone MuJoCo environment with LIBERO-style observations and success checks."""

    def __init__(
        self,
        xml_path: Path,
        physics_hz: int = 500,
        control_hz: int = 20,
        conveyor_speed: float = 0.025,
        width: int = 256,
        height: int = 256,
    ) -> None:
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = 1.0 / physics_hz
        self.data = mujoco.MjData(self.model)
        self.physics_hz = physics_hz
        self.control_hz = control_hz
        self.steps_per_control = max(1, int(round(physics_hz / control_hz)))
        self.conveyor_speed = conveyor_speed
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.ids = self._resolve_ids()
        self.conveyor_center = self.model.body_pos[self.ids.conveyor_body_id].copy()
        conveyor_rot = mat_from_quat(self.model.body_quat[self.ids.conveyor_body_id].copy())
        self.conveyor_dir = normalize(conveyor_rot[:, 0].copy())
        self.conveyor_lateral = normalize(conveyor_rot[:, 1].copy())
        self.conveyor_start = self.conveyor_center - self.conveyor_dir * CONVEYOR_HALF_LENGTH
        self.place_center = self.model.body_pos[self.ids.place_body_id].copy()
        geom_adr = int(self.model.body_geomadr[self.ids.place_body_id])
        self.place_radius = float(self.model.geom_size[geom_adr, 0])
        self.active_objects: set[str] = set()
        self.home_qpos = self.data.qpos.copy()

    def close(self) -> None:
        self.renderer.close()

    def _resolve_ids(self) -> EnvIds:
        arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        arm_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_pos") for n in ARM_JOINTS]
        gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
        right_gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_GRIPPER_JOINT)
        gripper_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Claw_left_pos")
        right_gripper_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Claw_right_pos")
        anomaly_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "anomaly_0")
        normal_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"normal_{i}") for i in range(5)]
        conveyor_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "layout_conveyor")
        place_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "layout_place_region")
        tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
        required = arm_joint_ids + arm_actuator_ids + [
            gripper_joint_id, right_gripper_joint_id, gripper_actuator_id,
            right_gripper_actuator_id, anomaly_body_id, conveyor_body_id,
            place_body_id, tcp_site_id,
        ]
        if any(i < 0 for i in required) or any(i < 0 for i in normal_body_ids):
            raise RuntimeError("LIBERO Griparm scene is missing required robot/object names.")
        object_qpos_adr = {}
        object_dof_adr = {}
        for name in ["anomaly_0"] + [f"normal_{i}" for i in range(5)]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jid = self.model.body_jntadr[bid]
            object_qpos_adr[name] = int(self.model.jnt_qposadr[jid])
            object_dof_adr[name] = int(self.model.jnt_dofadr[jid])
        return EnvIds(
            arm_joint_ids=arm_joint_ids,
            arm_qpos_adr=np.array([self.model.jnt_qposadr[j] for j in arm_joint_ids], dtype=np.int32),
            arm_actuator_ids=arm_actuator_ids,
            gripper_joint_id=gripper_joint_id,
            right_gripper_joint_id=right_gripper_joint_id,
            gripper_actuator_id=gripper_actuator_id,
            right_gripper_actuator_id=right_gripper_actuator_id,
            anomaly_body_id=anomaly_body_id,
            normal_body_ids=normal_body_ids,
            object_qpos_adr=object_qpos_adr,
            object_dof_adr=object_dof_adr,
            conveyor_body_id=conveyor_body_id,
            place_body_id=place_body_id,
            tcp_site_id=tcp_site_id,
        )

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home_qpos
        self.data.qpos[self.ids.arm_qpos_adr] = 0.0
        self.data.qpos[self.model.jnt_qposadr[self.ids.gripper_joint_id]] = 0.0
        self.data.qpos[self.model.jnt_qposadr[self.ids.right_gripper_joint_id]] = 0.0
        self.data.qvel[:] = 0.0
        for i, aid in enumerate(self.ids.arm_actuator_ids):
            self.data.ctrl[aid] = self.data.qpos[self.ids.arm_qpos_adr[i]]
        self.data.ctrl[self.ids.gripper_actuator_id] = 0.0
        self.data.ctrl[self.ids.right_gripper_actuator_id] = 0.0
        active_normals = list(rng.choice([f"normal_{i}" for i in range(5)], size=3, replace=False))
        self.active_objects = {"anomaly_0", *active_normals}
        visible = ["anomaly_0"] + sorted(active_normals)
        conveyor_end = self.conveyor_start + self.conveyor_dir * CONVEYOR_LENGTH
        for name in visible:
            qadr = self.ids.object_qpos_adr[name]
            dadr = self.ids.object_dof_adr[name]
            s = rng.uniform(*OBJECT_SPAWN_S_RANGE)
            lateral = rng.uniform(*OBJECT_SPAWN_LATERAL_RANGE)
            pos = conveyor_end - self.conveyor_dir * s + self.conveyor_lateral * lateral
            pos[2] = OBJECT_CENTER_Z_ON_BELT + OBJECT_SETTLE_DROP_HEIGHT
            self.data.qpos[qadr:qadr + 3] = pos
            self.data.qpos[qadr + 3:qadr + 7] = object_quat_laid_down(self.conveyor_dir, rng.uniform(0.0, 360.0))
            self.data.qvel[dadr:dadr + 6] = 0.0
        for i in range(5):
            name = f"normal_{i}"
            if name in self.active_objects:
                continue
            qadr = self.ids.object_qpos_adr[name]
            dadr = self.ids.object_dof_adr[name]
            self.data.qpos[qadr:qadr + 7] = HIDDEN_OBJECT_QPOS
            self.data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, bool]]:
        action = np.asarray(action, dtype=np.float64)
        for i, aid in enumerate(self.ids.arm_actuator_ids):
            self.data.ctrl[aid] = action[i]
        gripper = float(np.clip(action[6], 0.0, 0.04))
        self.data.ctrl[self.ids.gripper_actuator_id] = gripper
        self.data.ctrl[self.ids.right_gripper_actuator_id] = -gripper
        for _ in range(self.steps_per_control):
            self._move_conveyor()
            mujoco.mj_step(self.model, self.data)
        success = self.check_success()
        failure = self.check_failure()
        return self.get_obs(), float(success), bool(success or failure), {"success": success, "failure": failure}

    def _move_conveyor(self) -> None:
        dt = self.model.opt.timestep
        conveyor_end = self.conveyor_start + self.conveyor_dir * CONVEYOR_LENGTH
        for name in self.active_objects:
            qadr = self.ids.object_qpos_adr[name]
            pos = self.data.qpos[qadr:qadr + 3].copy()
            rel = pos - self.conveyor_start
            s = float(np.dot(rel, self.conveyor_dir))
            lat = float(np.dot(rel, self.conveyor_lateral))
            s_new = s - self.conveyor_speed * dt
            target_pos = self.conveyor_start + self.conveyor_dir * s_new + self.conveyor_lateral * lat
            self.data.qpos[qadr] = target_pos[0]
            self.data.qpos[qadr + 1] = target_pos[1]
            if s_new < -RESPAWN_MARGIN:
                pos = conveyor_end - self.conveyor_dir * rng_float(OBJECT_SPAWN_S_RANGE) + self.conveyor_lateral * rng_float(OBJECT_SPAWN_LATERAL_RANGE)
                pos[2] = OBJECT_CENTER_Z_ON_BELT + OBJECT_SETTLE_DROP_HEIGHT
                self.data.qpos[qadr:qadr + 3] = pos

    def check_success(self) -> bool:
        pos = self.data.xpos[self.ids.anomaly_body_id]
        return bool(np.linalg.norm(pos[:2] - self.place_center[:2]) <= self.place_radius and pos[2] > 0.02)

    def check_failure(self) -> bool:
        pos = self.data.xpos[self.ids.anomaly_body_id]
        rel = pos - self.conveyor_start
        s = float(np.dot(rel, self.conveyor_dir))
        lat = float(np.dot(rel, self.conveyor_lateral))
        in_conveyor = -RESPAWN_MARGIN <= s <= CONVEYOR_LENGTH + RESPAWN_MARGIN and abs(lat) <= CONVEYOR_HALF_WIDTH
        in_place = np.linalg.norm(pos[:2] - self.place_center[:2]) <= self.place_radius
        landed = 0.015 <= pos[2] <= 0.08 and abs(float(self.data.qvel[self.ids.object_dof_adr["anomaly_0"] + 2])) < 0.04
        return bool(landed and not in_conveyor and not in_place)

    def get_state(self) -> np.ndarray:
        return self.data.qpos.copy()

    def get_obs(self) -> dict[str, np.ndarray]:
        qpos = np.concatenate([
            self.data.qpos[self.ids.arm_qpos_adr],
            np.array([self.data.qpos[self.model.jnt_qposadr[self.ids.gripper_joint_id]]]),
        ]).astype(np.float32)
        return {
            "robot_state": qpos,
            "global_image": self.render("global"),
            "wrist_image": self.render("wrist"),
        }

    def render(self, camera_name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        frame = self.renderer.render()
        return np.asarray(frame, dtype=np.uint8)


def rng_float(bounds: tuple[float, float]) -> float:
    return float(np.random.uniform(bounds[0], bounds[1]))
