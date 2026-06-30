#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate pypy_env

echo "========== 并行启动 3 个实验 =========="

pypy src/phase_1/learning/run_qlearning_new3.py --mode full &
PID1=$!
echo "Q-learning      PID=$PID1"

pypy src/phase_2/run_drift_expectimax.py --mode full --workers 8 &
PID2=$!
echo "Expectimax      PID=$PID2"

pypy src/phase_2/run_drift_mcts1.py --mode full --workers 4 &
PID3=$!
echo "MCTS            PID=$PID3"

echo ""
echo "等待全部完成..."

wait $PID1 $PID2 $PID3

echo ""
echo "========== 三个实验全部完成 =========="
