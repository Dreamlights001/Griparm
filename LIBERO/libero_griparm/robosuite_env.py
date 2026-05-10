from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .env import (
    ARM_JOINTS,
    CONVEYOR_HALF_LENGTH,
    CONVEYOR_HALF_WIDTH,
    CONVEYOR_LENGTH,
    GRIPPER_JOINT,
    HIDDEN_OBJECT_QPOS,
    OBJECT_CENTER_Z_ON_BELT,
    OBJECT_SETTLE_DROP_HEIGHT,
    OBJECT_SPAWN_LATERAL_RANGE,
    OBJECT_SPAWN_S_RANGE,
    RESPAWN_MARGIN,
    RIGHT_GRIPPER_JOINT,
    EnvIds,
    mat_from_quat,
    normalize,
    object_quat_laid_down,
)

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/robosuite_numba_cache")

try:
    from robosuite.environments.base import MujocoEnv, register_env
    import robosuite.macros as macros
except ImportError:  # robosuite is an explicit dependency for this backend.
    MujocoEnv = object
    register_env = None
    macros = None
    _ROBOSUITE_IMPORT_ERROR = True
else:
    _ROBOSUITE_IMPORT_ERROR = False


class RawXMLModel:
    """Minimal robosuite model adapter for a fully-authored MuJoCo XML file."""

    def __init__(self, xml_path: Path):
        self.xml_path = Path(xml_path)
        self.mujoco_objects = []

    def get_xml(self) -> str:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        xml_dir = self.xml_path.parent.resolve()
        for elem in root.findall(".//*[@file]"):
            path = Path(elem.get("file"))
            if not path.is_absolute():
                elem.set("file", str((xml_dir / path).resolve()))
        return ET.tostring(root, encoding="unicode")

    def generate_id_mappings(self, sim) -> None:
        return None


@dataclass
class RobosuiteConfig:
    xml_path: Path
    physics_hz: int = 500
    control_hz: int = 20
    conveyor_speed: float = 0.025
    width: int = 256
    height: int = 256
    horizon: int = 1200
    has_renderer: bool = False
    has_offscreen_renderer: bool = True


class GriparmRobosuiteEnv(MujocoEnv):
    """robosuite MujocoEnv wrapper around the tuned Griparm MuJoCo XML scene."""

    def __init__(self, **kwargs):
        if _ROBOSUITE_IMPORT_ERROR:
            raise ImportError(
                "robosuite is required for GriparmRobosuiteEnv. Install it in sim_env with: "
                "pip install robosuite"
            )
        cfg = RobosuiteConfig(**kwargs)
        self.cfg = cfg
        self.xml_path = Path(cfg.xml_path)
        self.width = int(cfg.width)
        self.height = int(cfg.height)
        self.conveyor_speed = float(cfg.conveyor_speed)
        self.active_objects: set[str] = set()
        self.ids: EnvIds | None = None
        self.home_qpos = None
        self.conveyor_center = None
        self.conveyor_dir = None
        self.conveyor_lateral = None
        self.conveyor_start = None
        self.place_center = None
        self.place_radius = None
        self._next_reset_seed: int | None = None
        if macros is not None:
            macros.SIMULATION_TIMESTEP = 1.0 / float(cfg.physics_hz)
        super().__init__(
            has_renderer=cfg.has_renderer,
            has_offscreen_renderer=cfg.has_offscreen_renderer,
            render_camera="global",
            control_freq=cfg.control_hz,
            horizon=cfg.horizon,
            ignore_done=False,
            hard_reset=False,
            renderer="mujoco",
        )

    @property
    def action_spec(self):
        low = np.array([-np.pi] * 6 + [0.0], dtype=np.float32)
        high = np.array([np.pi] * 6 + [0.04], dtype=np.float32)
        return low, high

    @property
    def action_dim(self):
        return 7

    def _load_model(self):
        self.model = RawXMLModel(self.xml_path)

    def _setup_references(self):
        self.ids = self._resolve_ids()
        model = self.sim.model
        self.conveyor_center = self._body_pos(model, self.ids.conveyor_body_id).copy()
        conveyor_rot = mat_from_quat(self._body_quat(model, self.ids.conveyor_body_id).copy())
        self.conveyor_dir = normalize(conveyor_rot[:, 0].copy())
        self.conveyor_lateral = normalize(conveyor_rot[:, 1].copy())
        self.conveyor_start = self.conveyor_center - self.conveyor_dir * CONVEYOR_HALF_LENGTH
        self.place_center = self._body_pos(model, self.ids.place_body_id).copy()
        geom_adr = int(self._body_geomadr(model, self.ids.place_body_id))
        self.place_radius = float(self._geom_size(model, geom_adr)[0])
        self.home_qpos = self.sim.data.qpos.copy()

    def _setup_observables(self):
        return OrderedDict()

    def _reset_internal(self):
        super()._reset_internal()
        seed = self._next_reset_seed
        self._next_reset_seed = None
        self._reset_scene(seed=seed)

    def reset(self, seed: int | None = None):
        self._next_reset_seed = seed
        return super().reset()

    def reset_with_seed(self, seed: int | None = None):
        return self.reset(seed=seed)

    def _reset_scene(self, seed: int | None) -> None:
        if self.ids is None:
            return
        rng = np.random.default_rng(seed)
        data = self.sim.data
        data.qpos[:] = self.home_qpos
        data.qpos[self.ids.arm_qpos_adr] = 0.0
        data.qpos[self._joint_qposadr(self.sim.model, self.ids.gripper_joint_id)] = 0.0
        data.qpos[self._joint_qposadr(self.sim.model, self.ids.right_gripper_joint_id)] = 0.0
        data.qvel[:] = 0.0
        for i, aid in enumerate(self.ids.arm_actuator_ids):
            data.ctrl[aid] = data.qpos[self.ids.arm_qpos_adr[i]]
        data.ctrl[self.ids.gripper_actuator_id] = 0.0
        data.ctrl[self.ids.right_gripper_actuator_id] = 0.0

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
            data.qpos[qadr:qadr + 3] = pos
            data.qpos[qadr + 3:qadr + 7] = object_quat_laid_down(self.conveyor_dir, rng.uniform(0.0, 360.0))
            data.qvel[dadr:dadr + 6] = 0.0
        for i in range(5):
            name = f"normal_{i}"
            if name in self.active_objects:
                continue
            qadr = self.ids.object_qpos_adr[name]
            dadr = self.ids.object_dof_adr[name]
            data.qpos[qadr:qadr + 7] = HIDDEN_OBJECT_QPOS
            data.qvel[dadr:dadr + 6] = 0.0
        self.sim.forward()

    def _pre_action(self, action, policy_step=False):
        action = np.asarray(action, dtype=np.float64)
        for i, aid in enumerate(self.ids.arm_actuator_ids):
            self.sim.data.ctrl[aid] = action[i]
        gripper = float(np.clip(action[6], 0.0, 0.04))
        self.sim.data.ctrl[self.ids.gripper_actuator_id] = gripper
        self.sim.data.ctrl[self.ids.right_gripper_actuator_id] = -gripper
        self._move_conveyor()

    def _post_action(self, action):
        reward = self.reward(action)
        success = self.check_success()
        failure = self.check_failure()
        self.done = ((self.timestep >= self.horizon) and not self.ignore_done) or success or failure
        return reward, self.done, {"success": success, "failure": failure}

    def reward(self, action):
        return float(self.check_success())

    def _check_success(self):
        return self.check_success()

    def _get_observations(self, force_update=False):
        robot_state = np.concatenate([
            self.sim.data.qpos[self.ids.arm_qpos_adr],
            np.array([self.sim.data.qpos[self._joint_qposadr(self.sim.model, self.ids.gripper_joint_id)]]),
        ]).astype(np.float32)
        return OrderedDict(
            robot_state=robot_state,
            global_image=self.render_camera_image("global"),
            wrist_image=self.render_camera_image("wrist"),
        )

    def render_camera_image(self, camera_name: str) -> np.ndarray:
        frame = self.sim.render(camera_name=camera_name, width=self.width, height=self.height)
        return np.asarray(frame, dtype=np.uint8)

    def get_state(self) -> np.ndarray:
        return self.sim.data.qpos.copy()

    @property
    def native_model(self):
        return self.sim.model._model

    @property
    def native_data(self):
        return self.sim.data._data

    @property
    def data(self):
        return self.sim.data

    def _move_conveyor(self) -> None:
        dt = self.sim.model.opt.timestep
        conveyor_end = self.conveyor_start + self.conveyor_dir * CONVEYOR_LENGTH
        for name in self.active_objects:
            qadr = self.ids.object_qpos_adr[name]
            pos = self.sim.data.qpos[qadr:qadr + 3].copy()
            rel = pos - self.conveyor_start
            s = float(np.dot(rel, self.conveyor_dir))
            lat = float(np.dot(rel, self.conveyor_lateral))
            on_belt_xy = -RESPAWN_MARGIN <= s <= CONVEYOR_LENGTH + RESPAWN_MARGIN and abs(lat) <= CONVEYOR_HALF_WIDTH
            if (not on_belt_xy) or pos[2] > 0.12:
                continue
            s_new = s - self.conveyor_speed * dt
            target_pos = self.conveyor_start + self.conveyor_dir * s_new + self.conveyor_lateral * lat
            self.sim.data.qpos[qadr] = target_pos[0]
            self.sim.data.qpos[qadr + 1] = target_pos[1]
            dadr = self.ids.object_dof_adr[name]
            self.sim.data.qvel[dadr:dadr + 3] = -self.conveyor_dir * self.conveyor_speed
            if s_new < -RESPAWN_MARGIN:
                respawn_s = float(np.random.uniform(*OBJECT_SPAWN_S_RANGE))
                respawn_lat = float(np.random.uniform(*OBJECT_SPAWN_LATERAL_RANGE))
                pos = conveyor_end - self.conveyor_dir * respawn_s + self.conveyor_lateral * respawn_lat
                pos[2] = OBJECT_CENTER_Z_ON_BELT + OBJECT_SETTLE_DROP_HEIGHT
                self.sim.data.qpos[qadr:qadr + 3] = pos
                self.sim.data.qvel[dadr:dadr + 6] = 0.0

    def check_success(self) -> bool:
        pos = self.sim.data.xpos[self.ids.anomaly_body_id]
        return bool(np.linalg.norm(pos[:2] - self.place_center[:2]) <= self.place_radius and pos[2] > 0.02)

    def check_failure(self) -> bool:
        pos = self.sim.data.xpos[self.ids.anomaly_body_id]
        rel = pos - self.conveyor_start
        s = float(np.dot(rel, self.conveyor_dir))
        lat = float(np.dot(rel, self.conveyor_lateral))
        in_conveyor = -RESPAWN_MARGIN <= s <= CONVEYOR_LENGTH + RESPAWN_MARGIN and abs(lat) <= CONVEYOR_HALF_WIDTH
        in_place = np.linalg.norm(pos[:2] - self.place_center[:2]) <= self.place_radius
        dadr = self.ids.object_dof_adr["anomaly_0"]
        landed = 0.015 <= pos[2] <= 0.08 and abs(float(self.sim.data.qvel[dadr + 2])) < 0.04
        return bool(landed and not in_conveyor and not in_place)

    def _resolve_ids(self) -> EnvIds:
        model = self.sim.model
        arm_joint_ids = [self._joint_id(model, n) for n in ARM_JOINTS]
        arm_actuator_ids = [self._actuator_id(model, f"{n}_pos") for n in ARM_JOINTS]
        gripper_joint_id = self._joint_id(model, GRIPPER_JOINT)
        right_gripper_joint_id = self._joint_id(model, RIGHT_GRIPPER_JOINT)
        gripper_actuator_id = self._actuator_id(model, "Claw_left_pos")
        right_gripper_actuator_id = self._actuator_id(model, "Claw_right_pos")
        anomaly_body_id = self._body_id(model, "anomaly_0")
        normal_body_ids = [self._body_id(model, f"normal_{i}") for i in range(5)]
        conveyor_body_id = self._body_id(model, "layout_conveyor")
        place_body_id = self._body_id(model, "layout_place_region")
        tcp_site_id = self._site_id(model, "tcp_site")
        object_qpos_adr = {}
        object_dof_adr = {}
        for name in ["anomaly_0"] + [f"normal_{i}" for i in range(5)]:
            bid = self._body_id(model, name)
            jid = int(model.body_jntadr[bid])
            object_qpos_adr[name] = int(model.jnt_qposadr[jid])
            object_dof_adr[name] = int(model.jnt_dofadr[jid])
        return EnvIds(
            arm_joint_ids=arm_joint_ids,
            arm_qpos_adr=np.array([self._joint_qposadr(model, j) for j in arm_joint_ids], dtype=np.int32),
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

    @staticmethod
    def _joint_id(model, name): return model.joint_name2id(name)
    @staticmethod
    def _actuator_id(model, name): return model.actuator_name2id(name)
    @staticmethod
    def _body_id(model, name): return model.body_name2id(name)
    @staticmethod
    def _site_id(model, name): return model.site_name2id(name)
    @staticmethod
    def _joint_qposadr(model, jid): return int(model.jnt_qposadr[jid])
    @staticmethod
    def _body_pos(model, bid): return model.body_pos[bid]
    @staticmethod
    def _body_quat(model, bid): return model.body_quat[bid]
    @staticmethod
    def _body_geomadr(model, bid): return model.body_geomadr[bid]
    @staticmethod
    def _geom_size(model, gid): return model.geom_size[gid]


if register_env is not None:
    register_env(GriparmRobosuiteEnv)
