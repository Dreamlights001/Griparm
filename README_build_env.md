# build_env.py 使用说明 / User Guide

## 1. 作用 / Purpose

**中文**

`build_env.py` 用于从 `Arm_6Dof_claw_urdf/Arm_6Dof.urdf` 构建 MuJoCo 场景 XML，并注入项目所需的场景元素，包括：

- 机械臂 6 个旋转关节与夹爪执行器
- `global` 全局相机
- `wrist` 腕部相机
- 桌面、瑕疵品、正常品
- 抓取稳定性相关的摩擦、接触和执行器参数

输出文件通常是：

- `env.xml`

**English**

`build_env.py` builds a MuJoCo scene XML from `Arm_6Dof_claw_urdf/Arm_6Dof.urdf` and injects all task-specific scene elements, including:

- the 6 arm joints and gripper actuators
- the `global` camera
- the `wrist` camera
- the table, anomaly object, and normal objects
- grasp-related friction, contact, and actuator settings

The typical output file is:

- `env.xml`

## 2. 启动方式 / How to Run

```bash
conda run -n sim_env python build_env.py
```

常用示例 / Common examples:

```bash
conda run -n sim_env python build_env.py --output env.xml
conda run -n sim_env python build_env.py --urdf Arm_6Dof_claw_urdf/urdf/Arm_6Dof.urdf
```

## 3. 主要参数 / Main Arguments

- `--urdf`
  - 中文：输入 URDF 路径
  - English: input URDF path

- `--meshes-dir`
  - 中文：URDF 对应 mesh 目录
  - English: mesh directory used by the URDF

- `--anomaly-stl`
  - 中文：瑕疵品 STL 路径
  - English: anomaly STL path

- `--normal-stl`
  - 中文：正常品 STL 路径
  - English: normal STL path

- `--output`
  - 中文：输出 MJCF/XML 路径
  - English: output MJCF/XML path

## 4. 输出结果 / Output

**中文**

生成后的 `env.xml` 是后续所有调试和采集脚本的基础输入：

- `debug_cameras.py`
- `debug_layout.py`
- `preview.py`
- `check_kinematics.py`
- `calibrate_grasp.py`
- `collect_data.py`

**English**

The generated `env.xml` is the base scene file used by all later scripts:

- `debug_cameras.py`
- `debug_layout.py`
- `preview.py`
- `check_kinematics.py`
- `calibrate_grasp.py`
- `collect_data.py`

## 5. 注意事项 / Notes

**中文**

- 这是整个项目的前置步骤，通常只需在模型或基础场景有变化时重新运行
- 相机和产线布局的精调不在此脚本中完成，而是在后续 `debug_cameras.py` 和 `debug_layout.py` 中完成

**English**

- This is the prerequisite step for the whole project, and usually only needs to be rerun when the robot model or base scene changes
- Final camera and production-line tuning is handled later by `debug_cameras.py` and `debug_layout.py`
