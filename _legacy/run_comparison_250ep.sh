#!/bin/bash
# AFA-WCA / AFA / APR-S  250ep 比較実験
#   共通条件: ResNet18 / CIFAR-10 / batch=128 / SGD MultiStepLR [60,120,160,190]
#             seed=0 / GPU=0

set -e
GPU=0
SEED=0
EPOCHS=250
AFA_WCA_DIR=/home/kairisasaki/AFA-WCA
WCA_DIR=/home/kairisasaki/WCA
DATA=/home/kairisasaki/data/cifar10
DATA_C=/home/kairisasaki/APR_phase/data

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Run 1: AFA-WCA (WCA基底スワップ補助ストリーム) ───────────────────────
log "=== START: AFA-WCA ==="
cd $AFA_WCA_DIR
conda run -n apr python train_comparison.py \
  --aug afa-wca \
  --max-epoch $EPOCHS --batch-size 128 --lr 0.1 \
  --seed $SEED --gpu $GPU \
  --wca-source haar --wca-target db8 --wca-level 1 --wca-swap-prob 0.2 \
  2>&1 | tee results/afa-wca_ep${EPOCHS}_s${SEED}/train.log
log "=== DONE: AFA-WCA ==="

# ── Run 2: AFA (Fourier基底ノイズ補助ストリーム) ─────────────────────────
log "=== START: AFA ==="
cd $AFA_WCA_DIR
conda run -n apr python train_comparison.py \
  --aug afa \
  --max-epoch $EPOCHS --batch-size 128 --lr 0.1 \
  --seed $SEED --gpu $GPU \
  2>&1 | tee results/afa_ep${EPOCHS}_s${SEED}/train.log
log "=== DONE: AFA ==="

# ── Run 3: APR-S (WCA repo) ──────────────────────────────────────────────
log "=== START: APR-S ==="
cd $WCA_DIR
conda run -n apr python main.py \
  --aug apr-s-orig \
  --data $DATA --data-c $DATA_C --dataset cifar10 \
  --batch-size 128 --max-epoch $EPOCHS --gpu $GPU --seed $SEED \
  --outfolder ./results/apr_s_ep${EPOCHS}_s${SEED} \
  --memo apr_s_ep${EPOCHS}_s${SEED} \
  > ./results/apr_s_ep${EPOCHS}_s${SEED}/train.log 2>&1
conda run -n apr python main.py --eval eval \
  --aug apr-s-orig \
  --data $DATA --data-c $DATA_C --dataset cifar10 \
  --batch-size 256 --gpu $GPU --seed $SEED \
  --outfolder ./results/apr_s_ep${EPOCHS}_s${SEED} \
  --memo apr_s_ep${EPOCHS}_s${SEED} \
  >> ./results/apr_s_ep${EPOCHS}_s${SEED}/train.log 2>&1
log "=== DONE: APR-S ==="

# ── 結果サマリ ────────────────────────────────────────────────────────────
echo ""
log "=== 結果サマリ (${EPOCHS}ep, seed=${SEED}) ==="
printf "%-30s  %10s  %8s\n" "run" "clean_acc" "mCE"

# AFA-WCA
AFA_WCA_RESULT=$AFA_WCA_DIR/results/afa-wca_ep${EPOCHS}_s${SEED}/cifar10c_results.txt
if [ -f "$AFA_WCA_RESULT" ]; then
  CLEAN=$(grep "best_clean_acc" $AFA_WCA_RESULT | awk -F= '{print $2}')
  MCE=$(grep "mCE" $AFA_WCA_RESULT | awk '{print $NF}')
  printf "%-30s  %10s  %8s\n" "AFA-WCA" "$CLEAN" "$MCE"
fi

# AFA
AFA_RESULT=$AFA_WCA_DIR/results/afa_ep${EPOCHS}_s${SEED}/cifar10c_results.txt
if [ -f "$AFA_RESULT" ]; then
  CLEAN=$(grep "best_clean_acc" $AFA_RESULT | awk -F= '{print $2}')
  MCE=$(grep "mCE" $AFA_RESULT | awk '{print $NF}')
  printf "%-30s  %10s  %8s\n" "AFA" "$CLEAN" "$MCE"
fi

# APR-S
APR_LOG=$WCA_DIR/results/apr_s_ep${EPOCHS}_s${SEED}/logs.txt
if [ -f "$APR_LOG" ]; then
  CLEAN=$(grep "clean accuracy" $APR_LOG | tail -1 | awk '{print $NF}')
  MCE=$(grep "Mean Error" $APR_LOG | tail -1 | awk '{print $NF}')
  printf "%-30s  %10s  %8s\n" "APR-S" "$CLEAN" "$MCE"
fi

log "=== 完了 ==="
