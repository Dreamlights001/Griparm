# calibrate_grasp.py — 抓取姿态标定脚本

## 1. 作用

`calibrate_grasp.py` 用于手动标定机械臂抓取姿态。场景中只有一个瑕疵品（anomaly），用户通过遥操作将机械臂移动到正确抓取位置，保存配置并测试抓取是否成功。

标定结果供后续自动采集使用。

## 2. 启动方式

```bash
conda run -n sim_env python calibrate_grasp.py
```

## 3. 操作键位

键位与 `collect_data.py --mode teleop` 一致：

方向键和小键盘支持长按连续运动，也支持连点微调。抓取时先进行正常碰撞仿真，只有 anomaly 同时被左右两个爪片夹到，才临时附着到 TCP；失去双爪同时接触后会立即释放并掉落。

| 按键 | 关节 |
|------|------|
| ← → | J1 底座扭转 |
| ↑ ↓ | J2 肩部抬降 |
| 小键盘 1 / 2 | J3 |
| 小键盘 4 / 6 | J4 |
| 小键盘 5 / 8 | J5 |
| 小键盘 7 / 9 | J6 |
| 小键盘 - / + | 夹爪闭合 / 张开 |

| 功能键 | 作用 |
|--------|------|
| **m** | 保存当前抓取标定到 `calib_grasp.json` |
| **g** | 测试抓取 — 闭合夹爪 → 提升 → 检测物体是否被带起 |
| **r** | 重置场景（随机新物体位置） |
| **i** | 打印 TCP 与物体的相对位置 |
| ESC / q | 退出 |

## 4. 标定流程

```
1. 运行脚本 → 场景中出现一个瑕疵品
2. 用方向键 + 小键盘控制机械臂，将夹爪移动到物体正上方
3. 按 i 查看 TCP 与物体相对位置
4. 调整到位后按 m → 保存到 calib_grasp.json
5. 按 g 测试抓取 → 脚本闭合夹爪并尝试提升
   - 成功：终端出现 `two-claw contact`，物体随夹爪上升 → SUCCESS
   - 失败：没有同时夹到左右两个爪片，或抬升中失去双爪接触 → 调整后重试
6. 反复调整 + 测试，直到可靠抓取
7. 退出
```

## 5. 输出文件 `calib_grasp.json`

当前版本保存的是爪体 `Hand_Link` 相对瑕疵件轴线的局部标定，而不是爪片坐标系，也不是 `tcp_site`。该标定点表示“抓取前预对准位姿”，应当比物体中心高一些；自动采集会先让 `Hand_Link` 到达并跟踪这个相对位姿，再沿竖直方向下降一小段距离完成闭合抓取。

```json
{
  "schema_version": 3,
  "track_frame": "Hand_Link",
  "arm_joints": [0.0, -0.5, 0.3, ...],
  "gripper": 0.0,
  "gripper_body_world": [0.4, 0.05, 0.08],
  "object_world": [0.4, 0.05, 0.03],
  "object_axis_xy": [1.0, 0.0, 0.0],
  "object_side_xy": [0.0, 1.0, 0.0],
  "gripper_body_from_object": [0.0, 0.03, 0.05],
  "gripper_body_axis_offset": 0.0,
  "gripper_body_side_offset": 0.03,
  "gripper_body_height_offset": 0.05,
  "arm_joint_names": ["J_jianbu", "J_dabi", ...]
}
```

| 字段 | 含义 |
|------|------|
| `track_frame` | 自动采集跟踪的坐标框架，当前为 `Hand_Link` |
| `arm_joints` | 6 个关节的目标角度 (rad) |
| `gripper` | 夹爪位置 (0=开, ~0.048=闭) |
| `gripper_body_world` | 爪体 `Hand_Link` 在世界坐标的位置，仅用于调试记录 |
| `object_axis_xy` | 瑕疵件轴线在水平面的方向 |
| `object_side_xy` | 与瑕疵件轴线垂直的水平侧向 |
| `gripper_body_from_object` | 标定时从物体中心指向 `Hand_Link` 的世界坐标向量 |
| `gripper_body_axis_offset` | `Hand_Link` 在物体轴线方向上的带符号相对偏移 |
| `gripper_body_side_offset` | `Hand_Link` 在物体侧向上的带符号相对偏移 |
| `gripper_body_height_offset` | `Hand_Link` 高于物体中心的高度偏移 |
| `arm_joint_names` | 关节名称列表 |

## 6. 注意事项

- 标定时夹爪应张开，将手臂移到目标物体正上方
- `m` 键保存的是抓取前高位，不是最终闭合时的最低抓取位
- 自动采集最终闭合高度由 `collect_data.py --grasp-drop` 控制，默认从标定位向下 0.035 m
- 如果 `collect_data.py` 提示 `legacy tcp_site calibration`，需要重新运行本脚本并按 `m` 保存新版 `Hand_Link` 标定
- 夹爪应朝下（俯角），保持手腕高于传送带
- J6 控制手部旋转，确保爪片开合方向与物体长轴垂直
- 保存后不要移动物体位置，否则相对位置失效
