# Griparm 项目 — 数据采集运行指南

## 1. 环境

```bash
conda activate sim_env
```

确认依赖：
```bash
python -c "import mujoco, ledataset; print(mujoco.__version__)"
```

## 2. 工作流

推荐按顺序执行：

| 步骤 | 脚本 | 作用 |
|------|------|------|
| 1 | `build_env.py` | 从 URDF 生成 `env.xml` |
| 2 | `debug_cameras.py` | 交互式调试相机 → `env_camera_tuned.xml` |
| 3 | `debug_layout.py` | 交互式调试产线布局 → `env_layout_tuned.xml` |
| 4 | `preview.py` | 预览完整场景（物体流动 + 机械臂） |
| 5 | `calibrate_grasp.py` | 标定抓取姿态 → `calib_grasp.json` |
| 6 | `check_kinematics.py` | 逐关节检查运动学 |
| 7 | `collect_data.py` | 采集 LeRobot 数据集 |

## 3. 第一步：构建场景

```bash
python build_env.py --width 256 --height 256
```

## 4. 第二步：调试相机和布局

```bash
python debug_cameras.py --xml env.xml --save-xml env_camera_tuned.xml
python debug_layout.py --xml env_camera_tuned.xml --save-xml env_layout_tuned.xml
```

## 5. 第三步：预览

```bash
python preview.py --xml env_layout_tuned.xml
```

## 6. 第四步：标定抓取

```bash
python calibrate_grasp.py
```

操作机械臂到目标物体正上方 → 按 `m` 保存 → 按 `g` 测试 → 反复调整直到可靠抓取。

## 7. 第五步：采集数据

```bash
# 自动采集
python collect_data.py --mode auto --episodes 100 --dataset-root Lerobot_datasets/my_run

# 接续采集
python collect_data.py --mode auto --episodes 50 --dataset-root Lerobot_datasets/my_run --resume

# 遥操作采集
python collect_data.py --mode teleop --episodes 5 --dataset-root Lerobot_datasets/teleop_run

# 指定传送带速度（默认 0.025 m/s）和最大帧数（默认 2000 帧 = 40 秒）
python collect_data.py --mode teleop --episodes 5 --conveyor-speed 0.025 --max-data-frames 2000
```

遥操作采集说明：

- 默认传送带速度为 `0.025 m/s`，每条 episode 默认最多 `2000` 帧，即 50Hz 下 `40` 秒。
- 键位与 `calibrate_grasp.py` 一致，方向键和小键盘支持长按连续运动，也支持连点微调。
- 夹爪闭合后，只有瑕疵品同时接触左、右两个爪片，才会建立临时 TCP 附着并随夹爪移动。
- 释放不再要求夹爪完全张开；只要双爪同时接触被打破，物体就解除附着并按物理掉落，掉落中再次双爪接触会再次附着。
- anomaly 落入存放区并接近落地稳定时会自动保存当前 episode。
- anomaly 落到非传送带且非存放区的外界区域时会自动丢弃；落回传送带区域则继续采集。

## 8. 项目文件说明

| 文件 | 作用 |
|------|------|
| `build_env.py` | URDF → MJCF 编译，注入场景元素 |
| `debug_cameras.py` | 相机位姿交互调试 |
| `debug_layout.py` | 传送带/放置区布局调试 |
| `preview.py` | 产线预览（物体流动观察） |
| `check_kinematics.py` | 逐关节运动学测试 |
| `calibrate_grasp.py` | 抓取姿态标定 |
| `collect_data.py` | 数据采集（auto / teleop） |
| `env.xml` | 基础场景 |
| `env_camera_tuned.xml` | 调试后的相机 |
| `env_layout_tuned.xml` | 调试后的完整布局（采集用） |
| `calib_grasp.json` | 标定好的抓取姿态 |

## 9. 常见问题

### 场景加载失败（mesh 路径错误）
重新运行 `build_env.py` 生成 `env.xml`，路径会自动适配当前机器。

### 采集时机械臂不动
检查是否使用 `env_layout_tuned.xml`（需要包含 `layout_conveyor` 和 `layout_place_region`）。

### 夹取失败率高
运行 `calibrate_grasp.py` 重新标定抓取姿态；检查 claw 关节参数（damping/frictionloss/kp/forcerange）。

### 画面昏暗
已添加 headlight 照明，最新版采集脚本不会昏暗。如仍有问题，检查 `apply_lighting_for_debug` 是否被调用。
