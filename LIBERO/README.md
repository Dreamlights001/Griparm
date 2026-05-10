# LIBERO Griparm 子项目

这是一个独立的 LIBERO-style 数据采集子项目，但它不使用 LIBERO 的 BDDL 任务体系来驱动仿真。原因是 BDDL 更适合“固定桌面上的静态物体操作”，而本项目有传送带、动态物体、在线成功/失败判定和自定义相机逻辑，必须深入到 robosuite + MuJoCo 的底层仿真循环中定制。

当前实现方式：robosuite `MujocoEnv` 负责环境生命周期，MuJoCo XML 负责场景和物理模型，Python 代码负责传送带动态、reset、相机渲染、成功/失败判定和 LeRobot 数据写入。

## 目录

```text
LIBERO/
├── assets/                         # 本地复制的机器人和产品 STL
├── configs/griparm_sorting.yaml     # robosuite / MuJoCo / LeRobot 配置
├── datasets/                        # LeRobot 输出目录
├── libero_griparm/
│   ├── env.py                       # 直接 MuJoCo 参考封装，不作为默认入口
│   └── robosuite_env.py             # robosuite MujocoEnv 后端
├── scenes/griparm_sorting.xml       # 本地化后的 MuJoCo 场景
└── scripts/
    ├── check_scene.py
    └── collect_demonstrations.py
```

## 后端设计

默认后端是 `GriparmRobosuiteEnv`，位置：

```text
LIBERO/libero_griparm/robosuite_env.py
```

核心设计：

- 继承 `robosuite.environments.base.MujocoEnv`。
- 使用轻量 `RawXMLModel` 适配器直接加载 `scenes/griparm_sorting.xml`。
- 不通过 BDDL 创建任务、不通过 LIBERO 静态任务模板创建场景。
- 在 `_pre_action()` 中按 MuJoCo 物理步推进传送带物体。
- 在 `_reset_scene()` 中重置机械臂、夹爪、瑕疵品和正常品。
- 在 `_get_observations()` 中渲染 `global` 和 `wrist` 两路相机。
- 在 `check_success()` / `check_failure()` 中做动态任务判定。
- `collect_demonstrations.py` 直接把观测和动作写成 LeRobot 格式。

## 安装依赖

推荐创建专用 conda 环境：

```bash
cd /home/wang/Griparm/LIBERO
conda env create -f environment_libero.yml
conda activate griparm_libero
```

如果你想自己手动创建环境，也可以参考：

```bash
conda create -n griparm_libero python=3.10 -y
conda activate griparm_libero
pip install -r requirements.txt
```

注意：采集脚本依赖本地自定义包 `ledataset`，它不是标准 pip 包。完整环境创建说明见 [README_ENVIRONMENT.md](README_ENVIRONMENT.md)。

如果当前机器路径不是 `/home/wang/Griparm`，把命令里的路径替换成实际路径即可。`scenes/griparm_sorting.xml` 使用相对 asset 路径，不依赖父项目绝对路径。

## 检查场景

```bash
cd /home/wang/Griparm/LIBERO
python scripts/check_scene.py
```

成功时会打印：

- robosuite 场景已加载
- `robot_state` 形状
- `global_image` 形状
- `wrist_image` 形状
- 存放区中心和半径

无显示器服务器建议：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/check_scene.py
```

有 Ubuntu 可视化桌面建议：

```bash
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx python scripts/check_scene.py
```

## 采集 LeRobot 数据

默认输出到配置文件中的 `dataset_path`：

```bash
cd /home/wang/Griparm/LIBERO
python scripts/collect_demonstrations.py --num-demos 10
```

指定输出目录并覆盖旧数据：

```bash
python scripts/collect_demonstrations.py \
  --config configs/griparm_sorting.yaml \
  --num-demos 20 \
  --output datasets/my_griparm_lerobot \
  --overwrite
```

无可视化采集：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/collect_demonstrations.py --num-demos 20 --no-viewer
```

有 Ubuntu 桌面并打开 MuJoCo 被动 viewer 遥操作：

```bash
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx python scripts/collect_demonstrations.py --num-demos 5
```

## 遥操作键位

| 按键 | 动作 |
|------|------|
| 方向键 ← / → | J1 |
| 方向键 ↑ / ↓ | J2 |
| 小键盘 1 / 2 | J3 |
| 小键盘 4 / 6 | J4 |
| 小键盘 5 / 8 | J5 |
| 小键盘 7 / 9 | J6 |
| 小键盘 - / + | 夹爪闭合 / 张开 |
| Enter / 小键盘 Enter | 手动保存当前 demo |
| 小键盘 . | 丢弃当前 demo |
| ESC | 退出当前 demo |

## 保存逻辑

每条 demo 在以下情况保存：

- 按 `Enter` / 小键盘 `Enter` 手动保存。
- 瑕疵品进入粉色存放区并满足成功判定。

每条 demo 在以下情况丢弃：

- 按小键盘 `.`。
- 按 `ESC` 退出当前 demo。
- 瑕疵品落到非传送带、非存放区的外部区域并满足失败判定。
- 达到 `max_episode_seconds` 仍未成功。

## LeRobot 输出结构

```text
datasets/griparm_pick_anomaly_lerobot/
├── data/
├── meta/
└── videos/
    ├── global/
    └── wrist/
```

每帧字段：

- `observation.state`：7 维，6 个机械臂关节 + 夹爪位置。
- `action`：7 维，6 个关节控制目标 + 夹爪控制目标。
- `global`：全局相机视频帧。
- `wrist`：腕部相机视频帧。
- `task`：语言指令，来自 `configs/griparm_sorting.yaml`。

## 配置项

主要配置在 `configs/griparm_sorting.yaml`：

- `scene_xml`：robosuite 加载的 MuJoCo XML。
- `dataset_path`：默认 LeRobot 输出目录。
- `physics_hz`：MuJoCo 物理步频，当前 500Hz。
- `control_hz`：动作和采样频率，当前 20Hz。
- `conveyor_speed`：传送带速度。
- `max_episode_seconds`：单条 demo 最大时长，LIBERO 子项目独立配置。
- `camera_names` / `image_width` / `image_height`：保存到 LeRobot 的相机和图像尺寸。

## 与标准 LIBERO 的区别

- 标准 LIBERO 常用 BDDL 描述静态桌面任务；本项目不使用 BDDL 驱动任务。
- 标准 robosuite/Libero demo 常保存 HDF5；本项目最终直接保存 LeRobot 数据格式。
- 传送带运动、物体重置、相机输出、成功/失败判定都在 Python 环境类中实现。
