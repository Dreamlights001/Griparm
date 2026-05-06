#!/bin/bash
# ============================================================
# 预览脚本：观察产线布置、传送带运料、机械臂与相机视角
# ============================================================
# 用途：在正式采集数据前，快速检查场景是否正常
#   - 瑕疵品和正常品在传送带上循环移动
#   - 机械臂保持在 home 位姿不动
#   - 同时显示 global / wrist 两个相机预览窗口
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-glx}"

echo "[preview] 启动产线预览..."

conda run -n sim_env python preview.py \
  --xml env_layout_tuned.xml \
  --width 512 \
  --height 512 \
  --preview-backend auto \
  "$@"
