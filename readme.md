# Griparm 项目总览

本项目围绕一个 6 自由度机械臂 + 双爪夹具的 MuJoCo 仿真系统展开，目标是完成以下工作：

- 从 URDF 构建可运行的 MuJoCo 场景
- 调整腕部相机与全局相机
- 调整传送带与异常品放置区
- 预览产线中物体的生成、姿态与运动
- 检查各关节与夹爪是否正常运动
- 标定抓取姿态并验证夹取是否可靠
- 采集 LeRobot 格式数据
- 使用 robosuite + MuJoCo 底层 API 进行一套独立的 LIBERO 风格采集

## 文档索引

前置与基础：

- [README_build_env.md](README_build_env.md)：`build_env.py` 说明，URDF 转 `env.xml`
- [RUN_COLLECTION_CN.md](RUN_COLLECTION_CN.md)：原始采集运行总指南

主流程文档：

- [README_debug_cameras.md](README_debug_cameras.md)：相机交互调试
- [README_debug_layout.md](README_debug_layout.md)：产线布局交互调试
- [README_preview.md](README_preview.md)：产线预览
- [README_check_kinematics.md](README_check_kinematics.md)：逐关节运动检查
- [README_calibrate_grasp.md](README_calibrate_grasp.md)：抓取姿态标定与抓取验证
- [README_collect_data.md](README_collect_data.md)：主数据采集脚本

LIBERO 独立子项目：

- [LIBERO/README.md](LIBERO/README.md)：LIBERO 子项目总体说明
- [LIBERO/README_ENVIRONMENT.md](LIBERO/README_ENVIRONMENT.md)：LIBERO 专用环境依赖说明

## 推荐流程

严格来说，`build_env.py` 是整个流程的前置步骤；在此之后，推荐按下面的顺序使用脚本。

### 0. 构建基础场景

脚本：

- `build_env.py`

作用：

- 读取 `Arm_6Dof_claw_urdf/urdf/Arm_6Dof.urdf`
- 编译为 MuJoCo XML
- 注入桌面、物体、相机、执行器、接触与摩擦参数
- 输出 `env.xml`

常用命令：

```bash
conda run -n sim_env python build_env.py
```

### 1. 调试相机

脚本：

- `debug_cameras.py`

作用：

- 调整 `global` 全局固定相机
- 调整 `wrist` 腕部相机在 `Hand_Link` 坐标系下的相对位姿
- 同时观察 MuJoCo 主窗口与双相机画面
- 输出 `env_camera_tuned.xml`

说明文档：

- [README_debug_cameras.md](README_debug_cameras.md)

常用命令：

```bash
conda run -n sim_env python debug_cameras.py --xml env.xml --save-xml env_camera_tuned.xml
```

### 2. 调试产线布局

脚本：

- `debug_layout.py`

作用：

- 调整 `layout_conveyor` 传送带的位置与水平朝向
- 调整 `layout_place_region` 异常品存放区域
- 检查双相机是否都能覆盖传送带与存放区
- 输出 `env_layout_tuned.xml`

说明文档：

- [README_debug_layout.md](README_debug_layout.md)

常用命令：

```bash
conda run -n sim_env python debug_layout.py --xml env_camera_tuned.xml --save-xml env_layout_tuned.xml
```

### 3. 预览产线

脚本：

- `preview.py`
- `preview.sh`
- `run_preview.sh`

作用：

- 固定机械臂在初始姿态
- 在传送带上生成固定 4 个产品，其中 1 个为 anomaly
- 让物体沿传送带移动，用于检查起始区域、姿态、速度和相机视野
- 不保存数据集

说明文档：

- [README_preview.md](README_preview.md)

常用命令：

```bash
conda run -n sim_env python preview.py --xml env_layout_tuned.xml
```

### 4. 检查各关节能否正常运动

脚本：

- `check_kinematics.py`

作用：

- 逐个选择 J1~J6 和夹爪
- 通过离散步进检查每个自由度的正反方向、限位和执行器保持效果
- 用最小化场景快速确认“能不能动、方向对不对”

说明文档：

- [README_check_kinematics.md](README_check_kinematics.md)

常用命令：

```bash
conda run -n sim_env python check_kinematics.py
```

### 5. 检查能否正常抓取

脚本：

- `calibrate_grasp.py`

作用：

- 手动把 TCP 调到目标物体合适位置
- 保存 `calib_grasp.json`
- 通过 `g` 键执行闭合与抬升测试
- 验证双爪接触建立附着、失去双爪接触解除附着的逻辑

说明文档：

- [README_calibrate_grasp.md](README_calibrate_grasp.md)

常用命令：

```bash
conda run -n sim_env python calibrate_grasp.py
```

### 6. 数据采集

脚本：

- `collect_data.py`

作用：

- `auto` 模式：专家策略 + IK + 状态机自动抓取
- `teleop` 模式：手动遥操作采集
- 物理 500 Hz，数据采样 50 Hz
- 输出主项目的 LeRobot 数据集到 `Lerobot_datasets/`

说明文档：

- [README_collect_data.md](README_collect_data.md)
- [RUN_COLLECTION_CN.md](RUN_COLLECTION_CN.md)

常用命令：

```bash
python collect_data.py --mode auto --episodes 30
python collect_data.py --mode teleop --episodes 2
```

### 7. 基于 LIBERO 的独立数据采集

目录：

- `LIBERO/`

作用：

- 使用 robosuite `MujocoEnv` + 自定义 MuJoCo XML，而不是 BDDL 静态任务体系
- 独立维护自己的环境依赖、场景 XML 与采集脚本
- 最终仍输出 LeRobot 格式数据

说明文档：

- [LIBERO/README.md](LIBERO/README.md)
- [LIBERO/README_ENVIRONMENT.md](LIBERO/README_ENVIRONMENT.md)

常用命令：

```bash
cd LIBERO
conda activate libero
python scripts/check_scene.py
python scripts/collect_demonstrations.py --num-demos 10
```

## 关节速度与 RPM 换算

下面这张表对应当前主项目连续遥操作逻辑，也就是：

- `calibrate_grasp.py`
- `collect_data.py --mode teleop`

其中 J1~J6 是角速度，夹爪是平动速度。`rpm` 仅适用于旋转关节。

换算关系：

- `deg/s = rad/s × 57.2958`
- `rpm = rad/s × 60 / (2π) ≈ rad/s × 9.5493`
- 夹爪齿轮齿条等效换算：`rpm = 60v / (2πr)`，其中 `v` 为齿条线速度，`r` 为小齿轮节圆半径
- 当前项目模型没有显式建模齿轮半径，下面表格中的夹爪 `rpm` 采用等效假设 `r = 10 mm`

| 关节 | 控制键 | 速度 (rad/s) | 速度 (deg/s) | 速度 (rpm) |
|------|--------|--------------|--------------|------------|
| J1 `J_jianbu` | `← / →` | `0.4` | `22.92` | `3.82` |
| J2 `J_dabi` | `↑ / ↓` | `0.75` | `42.97` | `7.16` |
| J3 `J_Upper` | `KP_1 / KP_2` | `1.0` | `57.30` | `9.55` |
| J4 `J_Lower` | `KP_4 / KP_6` | `1.0` | `57.30` | `9.55` |
| J5 `J_wrist` | `KP_5 / KP_8` | `1.0` | `57.30` | `9.55` |
| J6 `J_hand` | `KP_7 / KP_9` | `1.0` | `57.30` | `9.55` |
| 夹爪 `Claw_left` | `KP_+ / KP_-` | `0.02 m/s` | `20.00 mm/s` | `19.10`（等效，`r=10 mm`） |

补充：

- `check_kinematics.py` 使用的是离散步进，不是连续速度控制：
- J1~J6：`0.05 rad / keypress`
- 夹爪：`0.003 m / keypress`
- `LIBERO/scripts/collect_demonstrations.py` 当前使用的是按键事件步进：
- J1~J6：`0.03 rad / key event`
- 夹爪：`0.003 m / key event`

## 关键文件关系

基础文件：

- `env.xml`：由 `build_env.py` 生成的基础场景
- `env_camera_tuned.xml`：相机调试后的场景
- `env_layout_tuned.xml`：产线布局调试后的场景
- `calib_grasp.json`：抓取姿态标定结果

核心脚本：

- `build_env.py`
- `debug_cameras.py`
- `debug_layout.py`
- `preview.py`
- `check_kinematics.py`
- `calibrate_grasp.py`
- `collect_data.py`

LIBERO 子项目：

- `LIBERO/libero_griparm/robosuite_env.py`
- `LIBERO/scripts/check_scene.py`
- `LIBERO/scripts/collect_demonstrations.py`

## 环境说明

主项目默认使用：

- `conda activate sim_env`

LIBERO 子项目推荐单独环境：

- `conda create -n libero python=3.10 -y`
- `conda activate libero`

如果只关心主项目采集，直接按本目录下说明文档执行即可；如果要跑 robosuite / LIBERO 独立采集，再进入 `LIBERO/` 按对应文档配置环境。
