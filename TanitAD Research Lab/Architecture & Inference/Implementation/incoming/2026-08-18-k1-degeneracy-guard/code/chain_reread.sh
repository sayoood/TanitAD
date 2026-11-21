#!/usr/bin/env bash
# CHAIN — re-read the 15 INCUMBENT ladder files under the REPAIRED solve + the
# C97 degeneracy guard. One invocation per arm, identical targets/seeds/estimator
# to the banked run, so old and new are comparable row for row.
#
# ⛔ TWO THINGS THIS CHAIN DOES THAT THE BANKED ONE DID NOT
#   1. --fit-mode unpen  — the repair taken from the MODULE
#      (`ridge_fit(..., intercept_col=-1)`), not re-derived locally. `centred`
#      is algebraically identical on the FULL fit (the z-scored design makes the
#      normal equations block-diagonal) but can differ ~1e-12 on the INNER
#      SPLIT, where the subset's feature mean is not exactly zero.
#   2. The `k1_guard` block on every rung — no repaired PASS is quotable without
#      it (C97).
#
# ⚠️ THE ALPHA GRID IS WIDENED to 1e7, matching the banked `ll_rep_*` runs.
# The banked [1e-2, 1e5] grid is too narrow for a repaired solve — its eval
# optimum sits at 1e6 — and `alpha_at_grid_edge` is emitted per row so a fit
# that ran out of grid says so.
#
# ⛔ THE SCRATCH COPIES ARE SYNCED FROM THE REPO BEFORE ANY RUN. MEASURED
# 2026-08-18: the scratch `pc6_linear_readout.py` still carried the PRE-C92
# `ridge_fit(X, y, alpha)` with no `intercept_col` at all — the local form of the
# pod-drift trap, and a launch from it would have silently re-run the defect.
set -u
ROOT="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
INC="$ROOT/TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
OUT="$INC/2026-08-18-k1-degeneracy-guard/raw"
SCR="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
W="$SCR/ll"; S="$SCR/sp2"; P="$SCR/pc"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
EPS="$S/cache/slotprobe-lead130-w120-256x640cyl"
JOIN="$S/lead130_agents.jsonl"
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTHONPATH="$ROOT/taniteval"
A="--alphas 0.01 0.1 1 10 100 1000 10000 100000 1000000 10000000"
# ⚠️ SEED 0 IS THE VERDICT SEED, and that is why the re-read runs at one seed.
# `ridge_artifact_audit.py` and `reread_table.py` both read `per_seed["0"]`, and
# `fit_one` draws its own `default_rng(seed)` per call — so seed 0's row is
# BIT-IDENTICAL whether or not seeds 1 and 2 also ran. Seed STABILITY under the
# repair is not skipped, it is already MEASURED on the 4 banked `ll_rep_*` arms
# (3 seeds each, `seed_K1_range`), so re-paying for it 15 times buys nothing.
# Override with LLR_SEEDS="0 1 2" if a 3-seed re-read is ever wanted.
SEEDS="${LLR_SEEDS:-0}"
mkdir -p "$OUT/reread"

sync_scratch () {
  cp "$INC/2026-08-17-probe-positive-control/code/pc6_linear_readout.py" "$W/"
  cp "$INC/2026-08-17-latent-linear-ladder/code/ll1_ladder.py"           "$W/"
  rm -rf "$W/__pycache__"
  # ⛔ a real import, not `ls` — the runbook step that catches a stale checkout
  "$PY" -c "
import sys; sys.path.insert(0, r'$W')
from pc6_linear_readout import ridge_fit
import inspect
assert 'intercept_col' in inspect.signature(ridge_fit).parameters, 'STALE pc6'
from taniteval.degeneracy import k1_guard
print('[sync] OK — repaired ridge_fit and k1_guard both importable')" || exit 1
}

run () {  # <tag> <cache> <label> [extra...]
  local tag="$1" cache="$2" label="$3"; shift 3
  "$PY" "$W/ll1_ladder.py" --cache "$cache" --split-json "$S/p3_selection.json" \
    --episodes-dir "$EPS" --join-file "$JOIN" --label "$label" \
    --out "$OUT/reread/llR_$tag.json" --seeds $SEEDS --fit-mode unpen $A "$@" \
    > "$OUT/reread/logR_$tag.txt" 2>&1
  echo "LLR $tag rc=$?"
}

# ⭐ THE REPRODUCTION GATE — before any repaired number is believed, prove the
# EDITED script still reproduces the BANKED incumbent bit-exactly. The guard and
# the `unpen` branch are additive; this is what turns "should be" into MEASURED.
gate () {
  local tag="$1" cache="$2" label="$3"; shift 3
  "$PY" "$W/ll1_ladder.py" --cache "$cache" --split-json "$S/p3_selection.json" \
    --episodes-dir "$EPS" --join-file "$JOIN" --label "$label" \
    --out "$OUT/reread/llGATE_$tag.json" --seeds 0 --fit-mode pc6 "$@" \
    > "$OUT/reread/logGATE_$tag.txt" 2>&1
  echo "GATE $tag rc=$?"
}

case "${1:-all}" in
  sync) sync_scratch ;;
  gate)
    sync_scratch || exit 1
    gate s11250      "$S/cache_s11250/latents.pt"      "v6F-SW-30k@11250"
    gate nullmatched "$S/cache_nullmatched/latents.pt" "RANDOM-LATENT-NULL-MATCHED@11250"
    ;;
  main)
    run s11250      "$S/cache_s11250/latents.pt"      "v6F-SW-30k@11250"
    run nullmatched "$S/cache_nullmatched/latents.pt" "RANDOM-LATENT-NULL-MATCHED@11250"
    run orcdir      "$P/cache_orcdir/latents.pt"      "GT-ORACLE-DIRECT@11250"
    run proxyv0     "$S/cache_s11250/latents.pt"      "C-V0-PROXY@11250" --proxy-v0
    ;;
  ckpt)
    run s09000 "$S/cache_s09000/latents.pt" "v6F-SW-30k@9000"
    run s09250 "$S/cache_s09250/latents.pt" "v6F-SW-30k@9250"
    run s10000 "$S/cache_s10000/latents.pt" "v6F-SW-30k@10000"
    run s02000 "$S/cache_s02000/latents.pt" "v6F-SW-30k@2000"
    ;;
  egoorc)
    for nr in 0.1 1 3 10; do
      run "egoorc_n$nr" "$W/cache_egoorc_n$nr/latents.pt" "EGO-ORACLE-n$nr@11250"
    done
    ;;
  tokens)
    run tok11250     "$S/cache_tok11250/latents.pt" "v6F-SW-30k@11250 TOKENS-MEAN" --features tokens_mean
    run tok11250null "$S/cache_tok11250/latents.pt" "TOKENS-MEAN MATCHED-RANDOM NULL" --features tokens_mean --randomise-features 20260817
    run cells_tokwin "$S/cache_tok11250/latents.pt" "v6F-SW-30k@11250 CELLS on the TOKENS window set"
    ;;
esac
echo "CHAIN_REREAD_DONE_${1:-all}"
