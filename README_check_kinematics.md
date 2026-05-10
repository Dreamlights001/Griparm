# check_kinematics.py 使用说明 / User Guide

## 1. 作用 / Purpose

**中文**

`check_kinematics.py` 用于逐个检查机械臂 6 个关节和夹爪是否能正常运动。它不执行抓取，也不运行产线逻辑，而是一个最小化的关节级联调试工具。

主要用途：

- 验证 `env.xml` 中的关节方向是否正确
- 验证执行器是否能驱动到目标位置
- 快速检查夹爪开合是否正常镜像

**English**

`check_kinematics.py` is a joint-by-joint motion checker for the 6 arm joints and the gripper. It does not run grasping or conveyor logic; it is a minimal kinematic debugging tool.

Main uses:

- verify joint directions in `env.xml`
- verify actuators can hold commanded targets
- verify mirrored gripper motion

## 2. 启动方式 / How to Run

```bash
conda run -n sim_env python check_kinematics.py
```

## 3. 操作键位 / Controls

- `1` ~ `7`
  - 中文：选择当前要测试的自由度
  - English: select the active DOF

- `,`
  - 中文：减小当前关节/夹爪位置
  - English: decrease the selected joint / gripper

- `.`
  - 中文：增大当前关节/夹爪位置
  - English: increase the selected joint / gripper

- `R`
  - 中文：全部关节和夹爪复位到初始状态
  - English: reset all joints and the gripper to the initial pose

- `Tab`
  - 中文：切换 MuJoCo 主视角
  - English: switch the MuJoCo viewer camera

- `ESC / Q`
  - 中文：退出
  - English: quit

## 4. 当前测试步长 / Current Step Size

| 自由度 | 步长 |
|--------|------|
| J1 ~ J6 | `0.05 rad / keypress` |
| 夹爪 | `0.003 m / keypress` |

## 5. 注意事项 / Notes

**中文**

- 该脚本主要检查“能不能动、方向对不对”，不是抓取性能验证
- 它使用离散按键步进，不是长按连续速度控制
- 若某个关节方向与预期相反，应先在这里确认，再回到抓取和采集脚本处理

**English**

- This script is mainly for checking whether each DOF moves and whether the direction is correct
- It uses discrete keypress steps, not continuous hold-based motion
- If a joint moves in the wrong direction, confirm it here before debugging grasping or data collection
