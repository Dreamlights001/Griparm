# LIBERO Griparm 子项目

这是一个独立的 LIBERO-style 数据采集子项目。它不导入父目录的 `collect_data.py`、`debug_cameras.py` 或其他运行时代码，只移植已经调好的机械臂、产品、相机、传送带和存放区场景。

## 目录

```text
LIBERO/
├── assets/                         # 本地复制的机器人和产品 STL
├── bddl_files/griparm_sorting/      # LIBERO-like 任务描述
├── configs/griparm_sorting.yaml     # 采集配置
├── datasets/                        # LeRobot 输出目录
├── libero_griparm/                  # 独立 MuJoCo 环境封装
├── scenes/griparm_sorting.xml       # 本地化后的 MuJoCo 场景
└── scripts/
    ├── check_scene.py
    └── collect_demonstrations.py
```

## 安装依赖

```bash
cd /home/wang/Griparm/LIBERO
conda activate sim_env
pip install -r requirements.txt
```

如果在当前机器路径是 `/home/dlts/Griparm`，把上面的路径替换成实际路径即可。`scenes/griparm_sorting.xml` 使用相对 asset 路径，不依赖 `/home/wang` 或父项目绝对路径。

## 检查场景

```bash
cd /home/wang/Griparm/LIBERO
python scripts/check_scene.py
```

成功时会打印 robot state、两路图像尺寸和存放区位置。

## 采集 LIBERO 配置的 LeRobot 数据集

```bash
cd /home/wang/Griparm/LIBERO
python scripts/collect_demonstrations.py --num-demos 10
```

默认输出：

```text
LIBERO/datasets/griparm_pick_anomaly_lerobot/
```

也可以指定输出：

```bash
python scripts/collect_demonstrations.py \
  --config configs/griparm_sorting.yaml \
  --num-demos 20 \
  --output datasets/my_griparm_lerobot \
  --overwrite
```

无可视化：

```bash
python scripts/collect_demonstrations.py --num-demos 20 --no-viewer
```

无头机器建议显式使用 EGL：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/collect_demonstrations.py --num-demos 20 --no-viewer
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

## LeRobot 结构

最终输出是 LeRobot 数据目录：

```text
datasets/griparm_pick_anomaly_lerobot/
├── data/
├── meta/
└── videos/
    ├── global/
    └── wrist/
```

每帧字段：

- `observation.state`：7 维，6 个机械臂关节 + 夹爪位置
- `action`：7 维，6 个关节控制目标 + 夹爪控制目标
- `global`：全局相机视频帧
- `wrist`：腕部相机视频帧
- `task`：语言指令，来自 `configs/griparm_sorting.yaml`

## 配置说明

主要配置在 `configs/griparm_sorting.yaml`：

- `scene_xml`：本地 MuJoCo 场景
- `bddl_file`：LIBERO-like 任务描述
- `dataset_path`：默认 LeRobot 输出目录
- `physics_hz`：物理频率
- `control_hz`：控制/采样频率
- `conveyor_speed`：传送带速度
- `max_episode_seconds`：单条 demo 最大时长，独立于父项目采集脚本

## 与父项目的边界

- 本子项目不修改父项目采集脚本。
- 本子项目不导入父项目 Python 模块。
- 父项目只作为资产来源，当前已把需要的 STL 和 XML 场景复制/本地化到 `LIBERO/` 内。
