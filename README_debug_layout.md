# debug_layout.py 使用说明 / User Guide

## 1. 作用 / Purpose

**中文**

`debug_layout.py` 用于交互式调试两类产线元素：

- `layout_conveyor`：传送带
- `layout_place_region`：异常品放置区域

该脚本会同时显示：

- MuJoCo 主三维窗口
- `global` 相机画面
- `wrist` 相机画面

主要用于：

- 调整传送带的位置与朝向
- 调整放置区域的位置与朝向
- 检查两个相机是否都能看到传送带和目标区域
- 将调好的布局保存到 XML

**English**

`debug_layout.py` is an interactive tool for tuning production-line layout elements:

- `layout_conveyor`: conveyor belt
- `layout_place_region`: anomaly placement region

It shows:

- the MuJoCo 3D viewer
- the `global` camera image
- the `wrist` camera image

It is mainly used to:

- adjust the conveyor position and yaw
- adjust the placement region position and yaw
- verify both cameras can observe the conveyor and target area
- save the tuned layout back to XML

## 2. 启动方式 / How to Run

```bash
conda run -n sim_env python debug_layout.py --xml env_camera_tuned.xml
```

常用示例 / Common examples:

```bash
conda run -n sim_env python debug_layout.py --xml env_camera_tuned.xml --save-xml env_layout_tuned.xml
conda run -n sim_env python debug_layout.py --xml env_camera_tuned.xml --object-visual stl --scene-preset checker
```

## 3. 主要参数 / Main Arguments

- `--xml`
  - 中文：输入场景 XML
  - English: input scene XML

- `--layout-xml`
  - 中文：额外加载已有的布局 XML
  - English: optionally load an existing layout XML

- `--save-xml`
  - 中文：按 `m` 保存时的输出 XML
  - English: output XML path used when pressing `m`

- `--width`, `--height`
  - 中文：相机预览分辨率
  - English: preview image resolution

- `--preview-backend {auto,cv2,matplotlib}`
  - 中文：图像预览后端
  - English: preview backend

- `--object-visual {stl,proxy}`
  - 中文：传送带上物体的显示方式，`stl` 为真实模型，`proxy` 为简化代理几何
  - English: object visualization mode on the conveyor; `stl` uses real meshes, `proxy` uses simplified primitives

- `--scene-preset {checker,clean}`
  - 中文：调试场景风格，推荐 `checker`
  - English: debug scene style; `checker` is recommended

- `--show-left-ui`, `--show-right-ui`
  - 中文：显示 MuJoCo 左/右侧 UI
  - English: show MuJoCo left/right UI panels

## 4. 操作键位 / Controls

- `\`
  - 中文：切换当前编辑对象：传送带 / 放置区域
  - English: toggle active object: conveyor / place region

- `↑ / ↓`
  - 中文：沿世界坐标前后移动当前对象
  - English: move the active object forward/backward in world XY

- `← / →`
  - 中文：沿世界坐标左右移动当前对象
  - English: move the active object left/right in world XY

- `z / x`
  - 中文：绕竖直轴旋转当前对象
  - English: rotate the active object around the vertical axis

- `- / =`
  - 中文：减小/增大平移步长
  - English: decrease/increase translation step

- `, / .`
  - 中文：减小/增大旋转步长
  - English: decrease/increase rotation step

- `p`
  - 中文：在终端打印当前布局片段
  - English: print current layout snippet to the terminal

- `m`
  - 中文：保存当前布局到 `--save-xml`
  - English: save current layout to `--save-xml`

- `ESC`
  - 中文：退出
  - English: quit

## 5. 输出结果 / Output

**中文**

保存后通常得到：

- `env_layout_tuned.xml`

该文件包含：

- 调好的相机参数（如果输入 XML 中已有）
- 调好的传送带
- 调好的放置区域

**English**

After saving, the typical output is:

- `env_layout_tuned.xml`

This file contains:

- tuned cameras, if already present in the input XML
- tuned conveyor layout
- tuned placement region layout

## 6. 注意事项 / Notes

**中文**

- 该脚本主要用于调位置和朝向，不用于抓取动作验证
- 调试时机械臂保持初始姿态
- 推荐先完成相机调试，再做布局调试

**English**

- This tool is for layout tuning, not for grasp-policy validation
- The robot remains frozen at its home pose
- Recommended workflow: finish camera tuning first, then layout tuning
