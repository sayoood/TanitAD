#!/usr/bin/env bash
# Lean 4 + Mathlib on the pod (PI go-ahead 2026-11-13). CPU and disk only, niced. Log: /workspace/v8proof/lean/install.log
set -u
L=/workspace/v8proof/lean
mkdir -p "$L" && cd "$L" || exit 2
export ELAN_HOME=$L/elan
export PATH=$ELAN_HOME/bin:$PATH
log(){ echo "[lean $(date -u +%H:%M:%SZ)] $*"; }
if [ ! -x "$ELAN_HOME/bin/elan" ]; then
  log "fetch the elan release from the leanprover GitHub org"
  curl -fsSL -o elan.tar.gz https://github.com/leanprover/elan/releases/latest/download/elan-x86_64-unknown-linux-gnu.tar.gz || { log "STOP fetch"; exit 3; }
  sha256sum elan.tar.gz
  tar xzf elan.tar.gz || { log "STOP untar"; exit 3; }
  ./elan-init -y --no-modify-path --default-toolchain none || { log "STOP elan-init"; exit 3; }
fi
elan --version
if [ ! -d "$L/v8proofs" ]; then
  log "new project with Mathlib"
  lake +leanprover-community/mathlib4:lean-toolchain new v8proofs math || { log "STOP lake new"; exit 4; }
fi
cd "$L/v8proofs" || exit 4
log "mathlib cache"
lake exe cache get || { log "STOP cache get"; exit 5; }
log "build"
lake build || { log "STOP build"; exit 6; }
lean --version
du -sh "$L" | tail -1
log "LEAN_READY"
touch "$L/LEAN_READY.txt"
