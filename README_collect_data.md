# collect_data.py — 数据采集脚本

## 1. 作用

MuJoCo 仿真数据采集，输出 LeRobot 格式数据集。支持两种模式：

- **auto**：专家策略（IK + 状态机）自动抓取放置
- **teleop**：键盘遥操作手动控制

物理 500 Hz，数据采样 50 Hz，采集 wrist + global 双路视频。

## 2. 启动方式

```bash
# 自动采集（带 viewer 观察窗）
python collect_data.py --mode auto --episodes 30

# 无头自动采集
python collect_data.py --mode auto --episodes 30 --no-viewer

# 遥操作采集
python collect_data.py --mode teleop --episodes 2

# 指定输出目录（默认覆盖）
python collect_data.py --mode auto --episodes 10 --dataset-root Lerobot_datasets/demo

# 显式覆盖已有目录
python collect_data.py --mode auto --episodes 10 --dataset-root Lerobot_datasets/demo --overwrite

# 续写已有数据集
python collect_data.py --mode auto --episodes 5 --dataset-root Lerobot_datasets/demo --resume
```

## 3. 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--xml` | `env_layout_tuned.xml` | 场景 XML 路径 |
| `--dataset-root` | 自动时间戳目录 | 输出数据集目录 |
| `--episodes` | 30 | 尝试回合数 |
| `--seed` | 42 | 随机种子 |
| `--width` / `--height` | 256 | 采集图像分辨率 |
| `--mode` | auto | `auto` 或 `teleop` |
| `--no-viewer` | false | 隐藏 MuJoCo viewer（auto 模式） |
| `--overwrite` | false | 强制覆盖已有数据集 |
| `--resume` | false | 续写已有数据集 |
| `--preview-backend` | auto | teleop 模式相机预览后端（auto/cv2/matplotlib） |

## 4. teleop 键位

| 按键 | 关节 |
|------|------|
| ← → | J1 底座扭转 |
| ↑ ↓ | J2 肩部抬降 |
| 小键盘 1 / 2 | J3 |
| 小键盘 4 / 6 | J4 |
| 小键盘 5 / 8 | J5 |
| 小键盘 7 / 9 | J6 |
| 小键盘 - | 夹爪闭合 |
| 小键盘 + | 夹爪张开 |

| 功能键 | 作用 |
|--------|------|
| Enter / 小键盘 Enter | 保存当前 episode |
| 小键盘 . | 丢弃当前 episode |
| 8 | 暂停/恢复传送带 |
| ESC | 退出 |

## 5. 自动模式采集流程

专家策略由状态机驱动，物理仿真抓取：

```
TRACKING → DESCEND → GRASP → LIFT_PLACE → DONE
```

| 状态 | 动作 |
|------|------|
| TRACKING | IK 跟踪物体上方预抓取位置 |
| DESCEND | 下降到抓取高度 |
| GRASP | 渐进闭合夹爪，接触检测，冻结夹持位 |
| LIFT_PLACE | 4 阶段插值轨迹：提升 → 横移 → 下降 → 释放 |
| DONE | 保持姿态 |

**夹取策略**：渐进闭合（~0.15/s），检测到爪片停滞 + 接触物体时冻结位置，不再继续闭合。物体靠物理接触力（椭圆摩擦锥 + condim=6 + kp=400）跟随夹爪运动。

## 6. 物理参数

| 参数 | 值 |
|------|-----|
| 重力 | 0 0 -9.81 |
| 物理频率 | 500 Hz |
| 采样频率 | 50 Hz |
| 摩擦锥 | elliptic, impratio=10 |
| 爪片接触 | condim=6, friction=1.0/0.05/0.005 |
| 物体接触 | condim=6, friction=1.0/0.05/0.005 |
| 爪片执行器 | kp=400, forcerange=±200N |
| 爪片阻尼 | damping=100, frictionloss=80N |
| 臂关节阻尼 | damping=15 |

## 7. 输出数据结构

```
dataset_root/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet          # LeRobot v3
│       └── episode_000000.parquet    # 兼容导出
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── tasks.jsonl
│   └── stats.parquet
└── videos/
    ├── wrist/
    │   └── episode_000000.mp4
    └── global/
        └── episode_000000.mp4
```

每帧数据包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `observation.state` | float32[7] | 6 臂关节 + 夹爪位置 |
| `action` | float32[7] | 7 个执行器目标 |
| `wrist` | video | 腕部摄像头 (h264) |
| `global` | video | 全局摄像头 (h264) |
| `task` | string | 任务描述 |

## 8. 注意事项

- 首次运行需先用 `build_env.py` 构建 `env.xml`，用 `debug_cameras.py` / `debug_layout.py` 调试
- 自动模式默认开启 MuJoCo viewer 观察窗（`--no-viewer` 关闭）
- 失败 episode 自动丢弃不保存
- 添加了 headlight 照明，场景不会昏暗
