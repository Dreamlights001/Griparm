# preview.py 使用说明 / User Guide

## 1. 作用 / Purpose

**中文**

`preview.py` 用于预览最终产线场景，不进行数据保存。它会：

- 固定机械臂在初始姿态
- 在传送带上生成固定 4 个产品
- 其中 1 个为瑕疵品，3 个为正常品
- 让这些物体按设定方向沿传送带移动
- 同时显示 MuJoCo 主窗口和两个相机画面

主要用途：

- 检查传送带位置、方向和起始区域是否正确
- 检查相机视野是否覆盖目标区域
- 检查产品的出现位置、姿态和移动是否合理
- 在采集前快速确认整套产线布置是否正常

**English**

`preview.py` previews the final production-line scene without saving data. It:

- keeps the robot frozen at its home pose
- spawns a fixed set of 4 products on the conveyor
- uses 1 anomaly and 3 normal parts
- moves them along the conveyor
- shows the MuJoCo main viewer and both camera views

Main uses:

- verify conveyor placement, direction, and spawn region
- verify the camera coverage
- verify product spawn poses and motion
- quickly validate the whole line before running data collection

## 2. 启动方式 / How to Run

```bash
conda run -n sim_env python preview.py --xml env_layout_tuned.xml
```

也可以直接：

```bash
bash preview.sh
```

Common examples:

```bash
conda run -n sim_env python preview.py --xml env_layout_tuned.xml --preview-backend auto
conda run -n sim_env python preview.py --xml env_layout_tuned.xml --no-camera-windows
```

## 3. 主要参数 / Main Arguments

- `--xml`
  - 中文：输入场景 XML，默认优先使用 `env_layout_tuned.xml`
  - English: input scene XML; defaults to `env_layout_tuned.xml` when available

- `--width`, `--height`
  - 中文：相机预览分辨率
  - English: preview render resolution

- `--preview-backend {auto,cv2,matplotlib}`
  - 中文：相机预览后端
  - English: camera preview backend

- `--seed`
  - 中文：物体随机布局种子
  - English: random seed for object layout

- `--show-left-ui`, `--show-right-ui`
  - 中文：显示 MuJoCo 左/右 UI
  - English: show MuJoCo left/right UI panels

- `--no-camera-windows`
  - 中文：仅显示 MuJoCo 主窗口，不显示两个相机图像窗口
  - English: disable the separate camera preview windows

## 4. 操作键位 / Controls

- `ESC / Q`
  - 中文：退出预览
  - English: quit preview

- `R`
  - 中文：重新随机生成当前 4 个物体布局
  - English: respawn the 4 visible objects with a new random layout

- `Space`
  - 中文：暂停/继续传送带运动
  - English: pause/resume conveyor motion

- `C`
  - 中文：显示/隐藏相机窗口
  - English: show/hide camera preview windows

- `.` / `,`
  - 中文：加快/减慢传送带速度倍率
  - English: increase/decrease conveyor speed multiplier

- `Tab`
  - 中文：切换 MuJoCo 主窗口视角
  - English: switch MuJoCo viewer camera

- 鼠标操作 / Mouse:
  - 左键拖拽：环绕
  - 右键拖拽：平移
  - 滚轮：缩放
  - Left drag: orbit
  - Right drag: pan
  - Scroll: zoom

## 5. 输出结果 / Output

**中文**

该脚本不保存数据集文件。它只是运行预览并在终端打印当前状态信息，例如：

- 当前瑕疵品位置
- 传送带是否暂停
- 当前速度倍率

**English**

This script does not save a dataset. It only previews the scene and prints runtime status, such as:

- current anomaly position
- whether the conveyor is paused
- current speed multiplier

## 6. 注意事项 / Notes

**中文**

- `preview.py` 与 `collect_data.py` 分离
- 预览脚本用于检查场景，不执行抓取策略
- 机械臂保持初始姿态不动

**English**

- `preview.py` is separate from `collect_data.py`
- it is meant for scene validation, not policy execution
- the robot remains frozen at its home pose
