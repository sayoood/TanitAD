#!/bin/bash
# AlpaSim setup on Thor — take 2, non-interactive.
#
# WHY TAKE 2: the first chain waited on a DONE marker from the torch install that had ALREADY
# been written before the chain was armed, so it waited forever on a signal in the past. Fixed by
# removing the wait entirely and running the steps directly.
#
# setup_local_env.sh must be SOURCED (it exports env), and it PROMPTS to install rustup when
# cargo is missing — a prompt is a hang under nohup. So rust is installed non-interactively
# FIRST, then the script is sourced with stdin closed.
set -u
log() { echo "[$(date -u +%FT%TZ)] $*"; }

log 'installing rust non-interactively (needed for utils_rs)'
if ! command -v cargo >/dev/null 2>&1; then
  curl -sSf --proto '=https' --tlsv1.2 https://sh.rustup.rs -o /tmp/rustup.sh
  sh /tmp/rustup.sh -y --no-modify-path --profile minimal >/tmp/rustup.log 2>&1
  log "rustup rc=$? ; cargo: $(ls "$HOME/.cargo/bin/cargo" 2>/dev/null || echo MISSING)"
fi
export PATH="$HOME/.cargo/bin:$HOME/venvs/tanitad-edge/bin:$PATH"
cargo --version 2>&1 | head -1

cd "$HOME/alpasim" || exit 2
export HF_TOKEN="$(cat "$HOME/.hftok" 2>/dev/null)"

log 'sourcing setup_local_env.sh (stdin closed so any prompt takes the default)'
source ./setup_local_env.sh < /dev/null > /tmp/alpasim_setup_env.log 2>&1
log "setup_local_env rc=$? (tail below)"
tail -6 /tmp/alpasim_setup_env.log

log 'checking the wizard is callable'
uv run alpasim_wizard --help > /tmp/wizard_help.log 2>&1
log "wizard --help rc=$?"
tail -4 /tmp/wizard_help.log

log 'downloading VaVAM assets for the stock driver smoke'
bash data/download_vavam_assets.sh --model vavam-b > /tmp/vavam_dl.log 2>&1
log "vavam assets rc=$?"
tail -4 /tmp/vavam_dl.log

log SETUP_DONE
