# debug_cameras.py 使用说明 / User Guide

## 1. 作用 / Purpose

**中文**

`debug_cameras.py` 用于交互式调试 MuJoCo 场景中的两个相机：

- `global`：全局固定相机
- `wrist`：腕部相机，固定在 `Hand_Link` 上，后续会跟随机械臂运动

该脚本主要用于：

- 调整全局相机位置与朝向
- 调整腕部相机相对 `Hand_Link` 的位置与姿态
- 同时查看 MuJoCo 主窗口和两个相机渲染画面
- 将调好的相机参数保存到 XML

**English**

`debug_cameras.py` is an interactive tool for tuning the two cameras in the MuJoCo scene:

- `global`: fixed world camera
- `wrist`: camera attached to `Hand_Link`, intended to move with the robot hand later

It is used to:

- adjust the global camera position and orientation
- adjust the wrist camera pose relative to `Hand_Link`
- view the MuJoCo main viewer and both rendered camera images at the same time
- save the tuned camera configuration back to XML

## 2. 启动方式 / How to Run

```bash
conda run -n sim_env python debug_cameras.py --xml env.xml
```

常用示例 / Common examples:

```bash
conda run -n sim_env python debug_cameras.py --xml env.xml --scene-preset checker
conda run -n sim_env python debug_cameras.py --xml env.xml --save-xml env_camera_tuned.xml
```

## 3. 主要参数 / Main Arguments

- `--xml`
  - 中文：输入场景 XML
  - English: input scene XML

- `--save-xml`
  - 中文：按 `m` 保存时输出的 XML 路径
  - English: output XML path used when pressing `m`

- `--width`, `--height`
  - 中文：两个相机预览窗口的渲染分辨率
  - English: render resolution for the camera preview windows

- `--preview-backend {auto,cv2,matplotlib}`
  - 中文：相机图像预览后端
  - English: backend for the camera preview windows

- `--scene-preset {checker,clean,default}`
  - 中文：调试环境样式，推荐 `checker`
  - English: scene style for debugging; `checker` is recommended

- `--show-left-ui`, `--show-right-ui`
  - 中文：显示 MuJoCo 左/右侧 UI
  - English: show MuJoCo left/right UI panels

## 4. 操作键位 / Controls

- `1`
  - 中文：激活全局相机
  - English: select the global camera

- `2`
  - 中文：激活腕部相机
  - English: select the wrist camera

- `\`
  - 中文：在全局相机与腕部相机之间切换
  - English: toggle between global and wrist cameras

- `↑ / ↓`
  - 中文：沿当前相机前后移动
  - English: move forward/backward in the active camera frame

- `← / →`
  - 中文：沿当前相机左右平移
  - English: move left/right in the active camera frame

- `z / x`
  - 中文：沿当前相机上下移动
  - English: move up/down in the active camera frame

- `u / o`
  - 中文：俯仰旋转
  - English: pitch up/down

- `[` / `]`
  - 中文：左右旋转
  - English: yaw left/right

- `- / =`
  - 中文：减小/增大平移步长
  - English: decrease/increase translation step size

- `, / .`
  - 中文：减小/增大旋转步长
  - English: decrease/increase rotation step size

- `p`
  - 中文：在终端打印当前相机 XML 片段
  - English: print current camera XML snippet to the terminal

- `m`
  - 中文：保存当前相机配置到 `--save-xml`
  - English: save the current camera configuration to `--save-xml`

- `ESC`
  - 中文：退出
  - English: quit

## 5. 输出结果 / Output

**中文**

保存后会得到一个包含调好相机参数的 XML，通常是：

- `env_camera_tuned.xml`

该文件可作为后续：

- 产线布局调试
- 预览
- 数据采集

的输入 XML。

**English**

After saving, you get an XML with tuned camera parameters, typically:

- `env_camera_tuned.xml`

This XML can then be used for:

- layout tuning
- preview
- data collection

## 6. 注意事项 / Notes

**中文**

- 调试时机械臂保持初始姿态不动
- 腕部相机是相对 `Hand_Link` 定义的，因此调好后会在后续脚本中随机械臂运动
- 推荐先调相机，再做产线布局调试

**English**

- The robot is frozen at its initial pose during camera tuning
- The wrist camera is defined relative to `Hand_Link`, so it will move with the robot later
- Recommended workflow: tune cameras first, then tune the production line layout
