#!/bin/bash
# AFA-WCA / AFA / APR-S  200ep 比較実験
#   全3手法を AFA-WCA リポジトリで統一実行 (同一フレームワーク・同一シード)
#   補助ストリームの中身だけが異なる:
#     afa-wca : WaveletBasisSwapOnline (今回の実装)
#     afa     : GeneralFourierOnline   (AFA公式 FBA)
#     apr-s   : APR p=0.6             (AFA公式 APR)
#
#   共通条件: ResNet18 / CIFAR-10 / batch=256 / SGD+CosineAnnealing
#             eta_min=1e-5 / grad_clip=1.0 / seed=0 / 200ep

set -eo pipefail
GPU=0
SEED=0
EPOCHS=250
DIR=/home/kairisasaki/AFA-WCA

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd $DIR

# ── Run 1: AFA-WCA ────────────────────────────────────────────────────────
log "=== START: AFA-WCA ==="
mkdir -p results/afa-wca_ep${EPOCHS}_s${SEED}
conda run -n apr python train_comparison.py \
  --aug afa-wca \
  --max-epoch $EPOCHS --batch-size 256 --lr 0.1 \
  --grad-clip 1.0 --seed $SEED --gpu $GPU \
  --wca-source haar --wca-target db8 --wca-level 1 --wca-swap-prob 0.2 \
  2>&1 | tee results/afa-wca_ep${EPOCHS}_s${SEED}/train.log
log "=== DONE: AFA-WCA ==="

# ── Run 2: AFA (FBA) ──────────────────────────────────────────────────────
log "=== START: AFA ==="
mkdir -p results/afa_ep${EPOCHS}_s${SEED}
conda run -n apr python train_comparison.py \
  --aug afa \
  --max-epoch $EPOCHS --batch-size 256 --lr 0.1 \
  --grad-clip 1.0 --seed $SEED --gpu $GPU \
  2>&1 | tee results/afa_ep${EPOCHS}_s${SEED}/train.log
log "=== DONE: AFA ==="

# ── Run 3: APR-S ─────────────────────────────────────────────────────────
log "=== START: APR-S ==="
mkdir -p results/apr-s_ep${EPOCHS}_s${SEED}
conda run -n apr python train_comparison.py \
  --aug apr-s \
  --max-epoch $EPOCHS --batch-size 256 --lr 0.1 \
  --grad-clip 1.0 --seed $SEED --gpu $GPU \
  2>&1 | tee results/apr-s_ep${EPOCHS}_s${SEED}/train.log
log "=== DONE: APR-S ==="

# ── 結果サマリ ────────────────────────────────────────────────────────────
echo ""
log "=== 結果サマリ (${EPOCHS}ep, seed=${SEED}) ==="
printf "%-12s  %12s  %8s\n" "method" "clean_acc(%)" "mCE_15(%)"
printf "%-12s  %12s  %8s\n" "------" "------------" "---------"

for AUG in afa-wca afa apr-s; do
  RESULT=$DIR/results/${AUG}_ep${EPOCHS}_s${SEED}/cifar10c_results.txt
  if [ -f "$RESULT" ]; then
    CLEAN=$(grep "best_clean_acc" "$RESULT" | awk -F= '{print $2}')
    MCE=$(grep "mCE_15" "$RESULT" | awk -F= '{print $2}')
    printf "%-12s  %12s  %8s\n" "$AUG" "$CLEAN" "$MCE"
  else
    printf "%-12s  %12s  %8s\n" "$AUG" "N/A" "N/A"
  fi
done

log "=== 完了 ==="
