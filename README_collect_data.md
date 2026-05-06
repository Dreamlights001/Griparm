# collect_data.py 使用说明 / User Guide

## 1. 作用 / Purpose

**中文**

`collect_data.py` 用于运行 MuJoCo 仿真并采集 LeRobot 格式数据集。当前支持两种模式：

- `auto`：专家策略自动采集
- `teleop`：人工遥操作采集

该脚本会：

- 加载场景 XML
- 自动修正采集时需要的物理/可视化设置
- 运行 500Hz 物理仿真
- 按 50Hz 采集图像和动作
- 保存为 LeRobot 数据集格式

**English**

`collect_data.py` runs the MuJoCo simulation and collects data in LeRobot format. It supports two modes:

- `auto`: expert-policy collection
- `teleop`: manual teleoperation-based collection

The script:

- loads the scene XML
- normalizes collection-time physics and visualization settings
- runs physics at 500 Hz
- records images and actions at 50 Hz
- saves the result as a LeRobot dataset

## 2. 采集模式 / Collection Modes

### 2.1 auto

**中文**

`auto` 模式使用内置专家策略：

- 追踪瑕疵品
- 预判目标位置
- IK 求解机械臂动作
- 完成抓取与放置

成功 episode 会保存，失败 episode 会丢弃。

**English**

`auto` mode uses the built-in expert policy to:

- track the anomaly
- predict target motion
- solve robot commands with IK
- perform grasp and place

Successful episodes are saved; failed episodes are discarded.

### 2.2 teleop

**中文**

`teleop` 模式由用户手动控制关节与夹爪，按键保存或丢弃当前 episode。

**English**

`teleop` mode lets the user manually control joints and the gripper, then explicitly save or discard the current episode.

## 3. 启动方式 / How to Run

### 自动采集 / Automatic collection

```bash
conda run -n sim_env python collect_data.py --mode auto --episodes 30
```

### 遥操作采集 / Teleoperation collection

```bash
conda run -n sim_env python collect_data.py --mode teleop --episodes 2
```

### 指定数据集目录 / Specify dataset path

```bash
conda run -n sim_env python collect_data.py \
  --mode auto \
  --episodes 2 \
  --dataset-root Lerobot_datasets/Class_Products418
```

## 4. 主要参数 / Main Arguments

- `--xml`
  - 中文：输入场景 XML，默认优先使用 `env_layout_tuned.xml`
  - English: input scene XML; defaults to `env_layout_tuned.xml` when available

- `--dataset-root`
  - 中文：输出数据集目录；如不指定，则自动生成 `Lerobot_datasets/Class_Products(时间戳)`
  - English: output dataset root; if omitted, a timestamped directory is created automatically

- `--episodes`
  - 中文：尝试回合数
  - English: number of episode attempts

- `--seed`
  - 中文：随机种子
  - English: random seed

- `--width`, `--height`
  - 中文：采集图像分辨率
  - English: capture image resolution

- `--mode {auto,teleop}`
  - 中文：采集模式
  - English: collection mode

- `--preview-backend {auto,cv2,matplotlib}`
  - 中文：`teleop` 模式下的相机预览后端
  - English: preview backend used in `teleop` mode

## 5. teleop 键位 / Teleop Controls

- `1 / q`
  - 中文：第 1 关节正向/反向
  - English: joint 1 positive/negative

- `2 / w`
  - 中文：第 2 关节正向/反向
  - English: joint 2 positive/negative

- `3 / e`
  - 中文：第 3 关节正向/反向
  - English: joint 3 positive/negative

- `4 / r`
  - 中文：第 4 关节正向/反向
  - English: joint 4 positive/negative

- `5 / t`
  - 中文：第 5 关节正向/反向
  - English: joint 5 positive/negative

- `6 / y`
  - 中文：第 6 关节正向/反向
  - English: joint 6 positive/negative

- `o / p`
  - 中文：夹爪闭合/张开
  - English: close/open the gripper

- `Space`
  - 中文：暂停/继续传送带
  - English: pause/resume conveyor motion

- `k`
  - 中文：保存当前 episode
  - English: save the current episode

- `x`
  - 中文：丢弃当前 episode
  - English: discard the current episode

- `ESC`
  - 中文：退出当前回合
  - English: exit the current episode

## 6. 数据输出 / Output Dataset

**中文**

采集结果保存为 LeRobot 风格结构，包括：

- `data/chunk-*/*.parquet`
- `videos/global/*.mp4`
- `videos/wrist/*.mp4`
- `meta/*.json / *.jsonl / *.parquet`

脚本也会额外导出兼容旧目录结构的文件。

**English**

The result is saved in a LeRobot-style structure, including:

- `data/chunk-*/*.parquet`
- `videos/global/*.mp4`
- `videos/wrist/*.mp4`
- `meta/*.json / *.jsonl / *.parquet`

The script also exports compatibility files for the legacy directory layout.

## 7. 物理设置 / Physics Settings

**中文**

采集脚本会自动统一以下设置：

- 重力：`0 0 -9.81`
- 物理频率：`500 Hz`
- 数据采样频率：`50 Hz`
- 传送带摩擦：`0.8 0.005 0.0001`
- 物体与夹爪摩擦：`0.7 0.005 0.0001`
- 传送带碰撞使用独立平面，避免“隐形墙”
- 场景使用 MuJoCo 棋盘格背景

**English**

The collection script normalizes the following settings:

- gravity: `0 0 -9.81`
- physics rate: `500 Hz`
- data sampling rate: `50 Hz`
- conveyor friction: `0.8 0.005 0.0001`
- object and gripper friction: `0.7 0.005 0.0001`
- a dedicated conveyor collision plane is used to avoid invisible collision walls
- the scene uses a MuJoCo checkerboard-style floor

## 8. 推荐工作流 / Recommended Workflow

**中文**

推荐按以下顺序使用脚本：

1. `debug_cameras.py` 调相机
2. `debug_layout.py` 调产线布局
3. `preview.py` 检查整套场景
4. `collect_data.py` 采集数据

**English**

Recommended order:

1. use `debug_cameras.py` to tune cameras
2. use `debug_layout.py` to tune the line layout
3. use `preview.py` to validate the whole scene
4. use `collect_data.py` to collect data
