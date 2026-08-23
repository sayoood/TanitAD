#!/usr/bin/env bash
# CHAIN — the C100 165-row re-read, RE-RUN AT THREE SEEDS (C103).
#
# ⛔ WHY THIS EXISTS. C100's inventory was computed at SEED 0 on the strength of
# the ladder's own claim that *"seed spread is exactly zero on 8 of 11 rungs, so
# >=3 seeds supply no uncertainty here"*. C103 falsified that claim: it was
# MEASURED UNDER THE C92 DEFECT, which had FROZEN the alpha sweep (a
# shrunk-to-zero fit makes inner-split MAE insensitive, so alpha selection barely
# moved). Repairing the intercept UN-TRUNCATES the sweep, seed now moves alpha,
# and the arm's own K1B moves 2.516 across seeds.
# ⇒ ROOT-CAUSE CLASS: A STABILITY CLAIM MEASURED UNDER A DEFECT IS NOT INHERITED
#   BY THE REPAIRED INSTRUMENT. A repair changes the estimator's SENSITIVITY,
#   not only its bias.
#
# ⚠️ DERIVED FROM, NOT EDITING, `…/2026-08-18-k1-degeneracy-guard/code/chain_reread.sh`.
# That directory belongs to another stream; this is a copy with exactly three
# things parameterised — OUT, SEEDS and the fit MODE — and everything that could
# move a number (caches, split, join, alpha grid, n_boot, targets) left byte
# identical so the 3-seed rows are comparable to the banked seed-0 rows.
#
# ⛔ THE TWO REPAIR ROUTES ARE RUN SEPARATELY AND NEVER POOLED (C100/C103).
#   route A = `unpen`   — ridge_fit(..., intercept_col=-1), the MODULE's repair.
#                         This is the route C100's 165-row inventory used.
#   route B = `centred` — the locally re-derived repair. This is the route
#                         `ll_rep_*` / LATENT_LINEAR_LADDER §7 was rendered from.
# MEASURED at seed 0 (C103): 44 paired rows, 2 alpha choices differ, 0 verdicts
# differ, but ego_v0's K1 differs by 0.3957 and its K1B by a factor of 8. That
# equivalence was itself measured at ONE SEED under the un-truncated sweep, so
# this chain re-measures it at three — the same discipline C103 demands.
#
# ⚠️ THE ALPHA GRID IS THE WIDENED [1e-2, 1e7] ONE, matching the banked re-read.
# The narrow [1e-2, 1e5] grid is too short for a repaired solve (its eval optimum
# sits at 1e6); `alpha_at_grid_edge` is emitted per row so a fit that ran out of
# grid says so.
#
# ⛔ THE SCRATCH COPIES ARE SYNCED FROM THE REPO AND IMPORT-PROBED BEFORE ANY RUN.
# MEASURED 2026-08-18 (C100): the scratch `pc6_linear_readout.py` was found
# PRE-C92 with no `intercept_col` at all. Staleness is a property of the TARGET;
# the repo being correct proves nothing about the copy the job actually loads.
#
# ⛔ T0-DIAGNOSTIC. A frozen-latent linear readout is a world-model diagnostic,
# never driving performance.
set -u
ROOT="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
INC="$ROOT/TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
OUT="$INC/2026-08-18-ladder-3seed/raw"
SCR="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
W="$SCR/ll"; S="$SCR/sp2"; P="$SCR/pc"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
EPS="$S/cache/slotprobe-lead130-w120-256x640cyl"
JOIN="$S/lead130_agents.jsonl"
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTHONPATH="$ROOT/taniteval"
A="--alphas 0.01 0.1 1 10 100 1000 10000 100000 1000000 10000000"
SEEDS="${LL3_SEEDS:-0 1 2}"
MODE="${LL3_MODE:-unpen}"
DEST="$OUT/reread_$MODE"
mkdir -p "$DEST"

sync_scratch () {
  cp "$INC/2026-08-17-probe-positive-control/code/pc6_linear_readout.py" "$W/"
  cp "$INC/2026-08-17-latent-linear-ladder/code/ll1_ladder.py"           "$W/"
  rm -rf "$W/__pycache__"
  # ⛔ a real import, not `ls` — and the md5s of BOTH files against the repo, so
  # "synced" is a measurement rather than the exit code of `cp`.
  "$PY" -c "
import hashlib, inspect, pathlib, sys
sys.path.insert(0, r'$W')
for a, b in ((r'$W/pc6_linear_readout.py',
              r'$INC/2026-08-17-probe-positive-control/code/pc6_linear_readout.py'),
             (r'$W/ll1_ladder.py',
              r'$INC/2026-08-17-latent-linear-ladder/code/ll1_ladder.py')):
    ha = hashlib.md5(pathlib.Path(a).read_bytes()).hexdigest()
    hb = hashlib.md5(pathlib.Path(b).read_bytes()).hexdigest()
    assert ha == hb, f'STALE {a}: {ha} != {hb}'
    print('[sync] md5 OK', ha, pathlib.Path(a).name)
from pc6_linear_readout import ridge_fit
assert 'intercept_col' in inspect.signature(ridge_fit).parameters, 'STALE pc6'
from taniteval.degeneracy import k1_guard
import ll1_ladder
assert 'centred' in ll1_ladder._solve.__doc__, 'STALE ll1'
print('[sync] OK — repaired ridge_fit, k1_guard and ll1 all importable')" || exit 1
}

run () {  # <tag> <cache> <label> [extra...]
  local tag="$1" cache="$2" label="$3"; shift 3
  "$PY" "$W/ll1_ladder.py" --cache "$cache" --split-json "$S/p3_selection.json" \
    --episodes-dir "$EPS" --join-file "$JOIN" --label "$label" \
    --out "$DEST/ll3_$tag.json" --seeds $SEEDS --fit-mode "$MODE" $A "$@" \
    > "$DEST/log3_$tag.txt" 2>&1
  echo "LL3[$MODE] $tag rc=$?"
}

case "${1:-all}" in
  sync) sync_scratch ;;
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
echo "CHAIN_3SEED_DONE_${MODE}_${1:-all}"
