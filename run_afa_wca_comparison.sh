#!/bin/bash
# AugMix+AFA / AugMix+WCA / PRIME+AFA / PRIME+WCA  公式設定 200ep
# 公式経路: AdvModule + rn18_dubn + ACE loss + DuBN
# AFA↔WCA の切り替えは attack='afa'|'wca' または use_fourier|use_wca の1フラグのみ
set -eo pipefail

DIR=/home/kairisasaki/AFA-WCA
cd $DIR

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== AFA-WCA 公式経路 4 構成比較 ==="
conda run -n apr python run_afa_wca_comparison.py 2>&1 | tee run_afa_wca.log
log "=== 完了 ==="
