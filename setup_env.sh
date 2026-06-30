#!/bin/bash
set -e

# ============================================================
# 2048_cs181 环境搭建脚本 (Linux)
# 基于 README.md 和 requirement.txt
# ============================================================

ENV_NAME="pypy_env"

# 初始化 conda shell 钩子（让 conda activate 在脚本中生效）
eval "$(conda shell.bash hook)"

# 1. 创建 conda 环境 (PyPy 解释器)
echo "[1/3] Creating conda environment '${ENV_NAME}' with PyPy..."
conda create -n ${ENV_NAME} pypy -c conda-forge -y

# 2. 激活环境并安装 pip 包
echo "[2/3] Installing pip dependencies..."
conda activate ${ENV_NAME}

pypy -m ensurepip 2>/dev/null || true

# 使用清华镜像源加速下载
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "  使用镜像源: ${MIRROR}"
pip install -i ${MIRROR} --trusted-host pypi.tuna.tsinghua.edu.cn \
    numpy tqdm pandas scipy matplotlib seaborn

# 3. 设置编码环境变量 (Linux 不需要 chcp，但设置 LANG 确保 UTF-8)
echo "[3/3] Setting UTF-8 encoding..."
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo ""
echo "✅ 环境搭建完成！"
echo ""
echo "激活环境:  conda activate ${ENV_NAME}"
echo "运行实验:  pypy src/phase_1/search/run_search.py"
