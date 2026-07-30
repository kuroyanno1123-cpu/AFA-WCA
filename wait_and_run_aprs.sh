#!/bin/bash
# 4 構成比較 (PID 612351) の終了を待ってから APR-S を実行
set -eo pipefail
WAIT_PID=612351
DIR=/home/kairisasaki/AFA-WCA

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Waiting for PID $WAIT_PID (4-way comparison) to finish..."
while kill -0 $WAIT_PID 2>/dev/null; do
    sleep 60
done
log "PID $WAIT_PID done. Starting APR-S..."

cd $DIR
conda run -n apr python run_aprs.py 2>&1 | tee run_aprs.log
log "APR-S done."
