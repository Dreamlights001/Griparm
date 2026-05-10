# LIBERO 专用采集环境依赖说明

这个文件只说明如何创建专门用于 `LIBERO/` 子项目的数据采集环境。当前 LIBERO 采集仍然使用 `GriparmRobosuiteEnv`，也就是 robosuite + MuJoCo 底层定制，不走 BDDL。

## 推荐方式：conda environment.yml

```bash
cd /home/wang/Griparm/LIBERO
conda env create -f environment_libero.yml
conda activate libero
```

如果环境已经存在：

```bash
conda activate libero
conda env update -f environment_libero.yml --prune
```

## 备选方式：pip requirements.txt

```bash
conda create -n libero python=3.10 -y
conda activate libero
pip install -r requirements.txt
```

## 必须处理的本地包：ledataset

当前采集脚本使用的是本地自定义 LeRobot 数据集封装：

```python
from ledataset.datasets.lerobot_dataset import LeRobotDataset
```

这个 `ledataset` 不是标准 pip 包。你新建环境后必须让 Python 能找到它。可选方式：

```bash
# 方式 1：如果你有 ledataset 源码目录
pip install -e /path/to/ledataset_source

# 方式 2：从旧 sim_env 复制已经可用的包
python -c "import site; print(site.getsitepackages()[0])"
cp -r /home/wang/miniconda3/envs/sim_env/lib/python3.10/site-packages/ledataset \
      /home/wang/miniconda3/envs/libero/lib/python3.10/site-packages/
```

复制路径需要按你机器上的 conda 安装位置调整。

## Ubuntu 24.04 系统依赖

有图形界面、需要 MuJoCo viewer 遥操作时建议安装：

```bash
sudo apt update
sudo apt install -y \
  ffmpeg \
  libgl1 \
  libegl1 \
  libglfw3 \
  libxinerama1 \
  libxcursor1 \
  libxi6 \
  libxrandr2 \
  libxxf86vm1 \
  mesa-utils
```

无头服务器只离屏采集时通常需要：

```bash
sudo apt update
sudo apt install -y ffmpeg libegl1 libgl1 mesa-utils
```

## 环境变量

有 Ubuntu 桌面并需要可视化 viewer：

```bash
export MUJOCO_GL=glfw
export PYOPENGL_PLATFORM=glx
export NUMBA_CACHE_DIR=/tmp/robosuite_numba_cache
```

无头服务器 / 只做离屏渲染：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/robosuite_numba_cache
```

`NUMBA_CACHE_DIR` 用于避免某些 robosuite/numba 安装方式下出现 cache locator 报错。

## 创建后验证

```bash
cd /home/wang/Griparm/LIBERO
conda activate libero
python -c "import mujoco, robosuite, glfw, yaml; import ledataset; print('env ok')"
python scripts/check_scene.py
```

成功时 `check_scene.py` 会打印两路相机图像尺寸和存放区位置。
