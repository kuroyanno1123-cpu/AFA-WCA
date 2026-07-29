#!/bin/bash
# PRIME / AugMix × FBA / WCA  250ep 4 構成比較
#
# 補助の中身 (FBA/WCA) だけが異なり、他は完全同一:
#   batch=256, CosineAnnealing per-step, grad_clip=1.0, seed=0, 250ep
#   ResNet-18, AFA 公式正規化
#
# 注意: 既存の PRIME/AugMix 単独 250ep 値 (VIPAug_phase, batch=128, MultiStepLR)
#       はフレームワーク別のため比較表に並べない。

set -eo pipefail
GPU=0
SEED=0
EPOCHS=250
DIR=/home/kairisasaki/AFA-WCA

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd $DIR

# ── Run 1: PRIME + FBA ────────────────────────────────────────────────────
log "=== START: prime + fba ==="
mkdir -p results/prime_fba_ep${EPOCHS}_s${SEED}
conda run -n apr python train_comparison.py \
  --main prime --aux fba \
  --max-epoch $EPOCHS --batch-size 256 --lr 0.1 \
  --grad-clip 1.0 --seed $SEED --gpu $GPU \
  2>&1 | tee results/prime_fba_ep${EPOCHS}_s${SEED}/train.log
log "=== DONE: prime + fba ==="

# ── Run 2: PRIME + WCA ────────────────────────────────────────────────────
log "=== START: prime + wca ==="
mkdir -p results/prime_wca_ep${EPOCHS}_s${SEED}
conda run -n apr python train_comparison.py \
  --main prime --aux wca \
  --max-epoch $EPOCHS --batch-size 256 --lr 0.1 \
  --grad-clip 1.0 --seed $SEED --gpu $GPU \
  --wca-source haar --wca-target db8 --wca-level 1 --wca-swap-prob 0.2 \
  2>&1 | tee results/prime_wca_ep${EPOCHS}_s${SEED}/train.log
log "=== DONE: prime + wca ==="

# ── Run 3: AugMix + FBA ───────────────────────────────────────────────────
log "=== START: augmix + fba ==="
mkdir -p results/augmix_fba_ep${EPOCHS}_s${SEED}
conda run -n apr python train_comparison.py \
  --main augmix --aux fba \
  --max-epoch $EPOCHS --batch-size 256 --lr 0.1 \
  --grad-clip 1.0 --seed $SEED --gpu $GPU \
  2>&1 | tee results/augmix_fba_ep${EPOCHS}_s${SEED}/train.log
log "=== DONE: augmix + fba ==="

# ── Run 4: AugMix + WCA ───────────────────────────────────────────────────
log "=== START: augmix + wca ==="
mkdir -p results/augmix_wca_ep${EPOCHS}_s${SEED}
conda run -n apr python train_comparison.py \
  --main augmix --aux wca \
  --max-epoch $EPOCHS --batch-size 256 --lr 0.1 \
  --grad-clip 1.0 --seed $SEED --gpu $GPU \
  --wca-source haar --wca-target db8 --wca-level 1 --wca-swap-prob 0.2 \
  2>&1 | tee results/augmix_wca_ep${EPOCHS}_s${SEED}/train.log
log "=== DONE: augmix + wca ==="

# ── 結果サマリ ────────────────────────────────────────────────────────────
echo ""
log "=== 結果サマリ (${EPOCHS}ep, seed=${SEED}) ==="
printf "%-15s  %-6s  %12s  %8s\n" "main" "aux" "clean_acc(%)" "mCE_15(%)"
printf "%-15s  %-6s  %12s  %8s\n" "----" "---" "------------" "---------"

for MAIN in prime augmix; do
  for AUX in fba wca; do
    RESULT=$DIR/results/${MAIN}_${AUX}_ep${EPOCHS}_s${SEED}/cifar10c_results.txt
    if [ -f "$RESULT" ]; then
      CLEAN=$(grep "best_clean_acc" "$RESULT" | awk -F= '{print $2}')
      MCE=$(grep "mCE_15" "$RESULT" | awk -F= '{print $2}')
      printf "%-15s  %-6s  %12s  %8s\n" "$MAIN" "$AUX" "$CLEAN" "$MCE"
    else
      printf "%-15s  %-6s  %12s  %8s\n" "$MAIN" "$AUX" "N/A" "N/A"
    fi
  done
done

echo ""
echo "(注) 既存 PRIME/AugMix 単独値は batch=128/MultiStepLR/VIPAug_phaseリポジトリのため別表で扱うこと"
log "=== 完了 ==="
