#!/bin/bash
# AlpaSim one-time setup, chained behind the torch install (single WiFi pipe — serialize).
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
until grep -q DONE ~/prep_envs.log 2>/dev/null; do sleep 120; done
log 'torch chain done -> installing uv into tanitad-edge'
~/venvs/tanitad-edge/bin/pip install -q uv 2>&1 | tail -1
~/venvs/tanitad-edge/bin/uv --version || { log UV_FAILED; exit 1; }
export PATH=$HOME/venvs/tanitad-edge/bin:$PATH
export HF_TOKEN=$(cat ~/.hftok)
cd ~/alpasim || exit 2
log 'downloading VaVAM example-driver assets (validates the whole download path)'
bash data/download_vavam_assets.sh --model vavam-b > ~/vavam_dl.log 2>&1
log "vavam assets rc=$? (see ~/vavam_dl.log)"
log 'CHAIN_COMPLETE — next: uv run alpasim_wizard deploy=local topology=1gpu driver=vavam'
