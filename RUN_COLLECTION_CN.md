# MuJoCo 抓取数据采集运行指南（中文）

本文档说明如何在当前项目中运行：

1. `build_env.py`：从 URDF 生成 MuJoCo 场景 `env.xml`  
2. `collect_data.py`：运行 500Hz 物理仿真 + 专家策略，并以 50Hz 采集 LeRobot 数据

---

## 1. 目录与脚本

请在项目根目录执行（你的目录是 `/home/dlts/Griparm`）：

- `build_env.py`
- `collect_data.py`
- `Arm_6Dof_claw_urdf/urdf/Arm_6Dof.urdf`
- `Sample/Anomaly.STL`
- `Sample/Normal.STL`

---

## 2. 环境准备

推荐使用你已有的 `sim_env` 环境（已包含 MuJoCo / ledataset 相关依赖）：

```bash
conda activate sim_env
```

可选检查：

```bash
python -V
python -c "import mujoco, ledataset; print(mujoco.__version__)"
```

---

## 3. 第一步：生成仿真场景 XML

在项目根目录运行：

```bash
conda run -n sim_env python build_env.py --out env.xml
```

默认会：

- 从 `Arm_6Dof.urdf` 编译并注入场景元素
- 添加 `wrist` 腕部相机和 `global` 全局相机
- 添加 `tcp_site`、执行器、夹爪镜像约束
- 添加 1 个 `anomaly_0` + 5 个 `normal_*` 物体
- 设置摩擦参数（夹爪和物体）

如需修改分辨率（例如 256x256）：

```bash
conda run -n sim_env python build_env.py --out env.xml --width 256 --height 256
```

---

## 4. 第二步：运行采集

## 4.1 最小可运行示例

```bash
conda run -n sim_env python collect_data.py \
  --xml env.xml \
  --dataset-root /tmp/Class_Products_smoke \
  --episodes 1 \
  --seed 5 \
  --width 96 \
  --height 96
```

## 4.2 正式采集示例（与你目标目录一致）

```bash
conda run -n sim_env python collect_data.py \
  --xml env.xml \
  --dataset-root /home/dlts/Griparm/Class_Products \
  --episodes 100 \
  --seed 42 \
  --width 256 \
  --height 256
```

说明：

- `--episodes` 是“尝试回合数”，只有成功回合会保存，失败回合会丢弃。
- 物理频率固定 500Hz，采样频率固定 50Hz（每 10 个物理步采样一次）。
- 单回合最多 1000 帧（约 20 秒），成功放置会提前结束。

---

## 5. 参数说明

- `--xml`：MuJoCo 场景文件路径（通常是 `env.xml`）
- `--dataset-root`：输出数据集目录（必须是不存在的新目录）
- `--episodes`：运行回合数（尝试次数）
- `--seed`：随机种子（控制每回合激活物体数和扰动）
- `--width --height`：相机分辨率（建议训练使用 256x256）

---

## 6. 输出数据结构

脚本会输出两套兼容结构：

1. LeRobot v3 原生结构（主结构）  
2. 你要求的兼容结构（`episode_*.parquet` + `meta/*.jsonl` + `videos/*/episode_*.mp4`）

采集完成后可检查：

```bash
find /home/dlts/Griparm/Class_Products -maxdepth 4 -type f | sort
```

你会看到类似：

- `data/chunk-000/file-000.parquet`（LeRobot v3）
- `data/chunk-000/episode_000000.parquet`（兼容导出）
- `meta/info.json`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`
- `meta/tasks.jsonl`
- `videos/wrist/episode_000000.mp4`
- `videos/global/episode_000000.mp4`

---

## 7. 成功判定与保存逻辑

每回合结束时，脚本检查：

- `anomaly_0` 是否进入目标放置区（右后侧区域）
- `anomaly_0` 的 Z 是否高于桌面阈值

结果处理：

- 成功：`save_episode()` 保存
- 失败：清空缓存并丢弃该回合

---

## 8. 常见问题

## 8.1 `Dataset root already exists`

`--dataset-root` 目录必须不存在。请换新目录，或先删除旧目录后重跑。

## 8.2 EGL / 显卡权限 warning

无头环境可能出现 `libEGL warning`，通常不影响脚本完成。只要进程能跑完并有输出文件即可。

## 8.3 出现 MuJoCo 不稳定警告（QACC）

当前脚本已做了基础抑制（IK 步长限制、控制增益下调等）。若仍频繁出现，可进一步降低控制增益或收紧 IK 更新步长。

---

## 9. 推荐运行流程（复制即用）

```bash
cd /home/dlts/Griparm

conda run -n sim_env python build_env.py --out env.xml --width 256 --height 256

conda run -n sim_env python collect_data.py \
  --xml env.xml \
  --dataset-root /home/dlts/Griparm/Class_Products \
  --episodes 200 \
  --seed 42 \
  --width 256 \
  --height 256
```

完成后检查：

```bash
find /home/dlts/Griparm/Class_Products -maxdepth 4 -type f | sort | sed -n '1,120p'
```

