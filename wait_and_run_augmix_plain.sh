#!/bin/bash
# run_remaining_experiments.py (PID 619574) の終了を待って素の AugMix を実行
set -eo pipefail
WAIT_PID=619574
DIR=/home/kairisasaki/AFA-WCA

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Waiting for PID $WAIT_PID (remaining 3 experiments) to finish..."
while kill -0 $WAIT_PID 2>/dev/null; do
    sleep 60
done
log "PID $WAIT_PID done. Starting plain AugMix..."

cd $DIR
conda run -n apr python run_augmix_plain.py 2>&1 | tee run_augmix_plain.log
log "Plain AugMix done."
