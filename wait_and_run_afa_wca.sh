#!/bin/bash
# run_comparison.sh (PID 583673) の終了を待ってから 4 実験を起動
set -eo pipefail
WAIT_PID=583673
DIR=/home/kairisasaki/AFA-WCA

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Waiting for PID $WAIT_PID (run_comparison.sh) to finish..."
while kill -0 $WAIT_PID 2>/dev/null; do
    sleep 60
done
log "PID $WAIT_PID done. Starting AFA-WCA 4-way comparison..."

cd $DIR
bash run_afa_wca_comparison.sh
