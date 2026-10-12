#!/usr/bin/env bash
# G-REWARD-ID on REAL on-policy proposals, through the production CLIs, dev box (RL-S1 part 2).
#   1. refe/onpolicy_relabel_pdm.py  -> pdm_<file>   (NAVSIM's own PDM targets: dac, ...)
#   2. refe/rl_terms_label.py        -> rl_terms_<file>   (D, K, L; gmax for the identity)
#   3. eval/rl_terms_join_identity.py joins the two on the shared key and counts the 2x2 (side files alone).
# Sets: the dev box's own ckpt-4933 dry-run dumps (data/refe_m5/sets, 64 proposals each, A7-repaired executed plans).
#   bash eval/rl_terms_onpolicy_local.sh [FILES_GLOB] [LIMIT_PER_FILE] [WORKERS]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$(dirname "$HERE")"
FILES="${1:-onpolicy_r0_m5w[0-2].jsonl}"
LIMIT="${2:-150}"
WORKERS="${3:-12}"
SETS="${REFE_RLT_SETS:-E:/Projects/TanitAD/data/refe_m5/sets}"
OUT="$HERE/raw/rl_terms/onpolicy_local"
V11="E:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
export PYTHONPATH="$V11;$PKG/refe" PYTHONIOENCODING=utf-8 NUPLAN_MAPS_ROOT="E:/Archive/devbox-C/navsim/data/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0 REFE_SIM_HZ=10 OMP_NUM_THREADS=1
PY="C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
mkdir -p "$OUT"
"$PY" "$PKG/refe/onpolicy_relabel_pdm.py" --sets "$SETS" --files "$FILES" --out "$OUT" --workers "$WORKERS" --limit "$LIMIT" > "$OUT/pdm.log" 2>&1
echo "pdm exit $?" >> "$OUT/pdm.log"
"$PY" "$PKG/refe/rl_terms_label.py" --sets "$SETS" --files "$FILES" --out "$OUT" --workers "$WORKERS" --limit "$LIMIT" > "$OUT/rlt.log" 2>&1
echo "rlt exit $?" >> "$OUT/rlt.log"
"$PY" "$HERE/rl_terms_join_identity.py" --rlt "$OUT" --pdm "$OUT" --out "$HERE/raw/rl_terms/identity_onpolicy.json" > "$OUT/join.log" 2>&1
echo "join exit $?" >> "$OUT/join.log"
