#!/usr/bin/env bash
# CHAIN — the LAST UNRE-READ ROWS: the four banked `ll_rep_*` arms, refitted
# with the C97 degeneracy guard on every rung.
#
# ⛔ WHY THIS RUNS IN `centred` AND NOT IN `unpen`.
# The 165-row re-read (`…/2026-08-18-k1-degeneracy-guard/`) refitted the fifteen
# INCUMBENT (`fit_mode: pc6`) arms under `--fit-mode unpen`. The four `ll_rep_*`
# files are a DIFFERENT route to the same repair (`--fit-mode centred`), and
# `LATENT_LINEAR_LADDER.md` §7's repair table is rendered from THEM. C100's last
# paragraph forbids pooling the two routes: the full fit agrees to 5e-14 but the
# INNER SPLIT differs by up to 0.74 MAE, enough to flip a near-tied alpha choice
# (K1 +0.4274 -> +0.0317). ⇒ To correct §7 IN PLACE the guard has to be run on
# the SAME route those numbers came from. Running `unpen` here would silently
# substitute one route's numbers for the other's — which is the very error the
# escalation names.
#
# ⛔ AND IT IS ITS OWN REPRODUCTION GATE. Only the guard block and a docstring
# separate today's `ll1_ladder.py` from the one that produced the banked
# `ll_rep_*`. Every non-guard field must therefore come back BIT-EXACT; the
# comparison is `verify_rep_guard.py` and it is not optional, because a silently
# changed producer would make the guard verdicts unattachable to the banked row.
#
# ⛔ SCRATCH IS SYNCED FROM THE REPO AND VERIFIED BY A REAL IMPORT FIRST.
# MEASURED 2026-08-18 (this run): the scratch `ll1_ladder.py` was STALE against
# the repo (md5 b3531424 vs 4b57f4a8) even though the scratch `pc6` copy had
# been refreshed hours earlier by another stream. Staleness is a property of the
# TARGET, not of your own changes, and it is per-file: one synced file proves
# nothing about its sibling.
set -u
ROOT="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
INC="$ROOT/TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
OUT="$INC/2026-08-18-ladder-corrected/raw/rep_guard"
SCR="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
W="$SCR/ll"; S="$SCR/sp2"; P="$SCR/pc"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
EPS="$S/cache/slotprobe-lead130-w120-256x640cyl"
JOIN="$S/lead130_agents.jsonl"
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTHONPATH="$ROOT/taniteval"
# ⚠️ IDENTICAL to the banked `ll_rep_*` invocation (`chain_ladder.sh` case
# `repair`): same alpha grid widened to 1e7, same three seeds, same fit mode.
A="--fit-mode centred --alphas 0.01 0.1 1 10 100 1000 10000 100000 1000000 10000000"
SEEDS="0 1 2"
mkdir -p "$OUT"

sync_scratch () {
  cp "$INC/2026-08-17-probe-positive-control/code/pc6_linear_readout.py" "$W/"
  cp "$INC/2026-08-17-latent-linear-ladder/code/ll1_ladder.py"           "$W/"
  rm -rf "$W/__pycache__"
  "$PY" -c "
import sys, inspect; sys.path.insert(0, r'$W')
from pc6_linear_readout import ridge_fit
assert 'intercept_col' in inspect.signature(ridge_fit).parameters, 'STALE pc6'
from taniteval.degeneracy import k1_guard
import ll1_ladder
src = inspect.getsource(ll1_ladder.fit_one)
assert 'k1_guard(' in src, 'STALE ll1_ladder — no guard call in fit_one'
assert 'K1_PASSES_GUARDED' in src, 'STALE ll1_ladder — no guarded verdict'
print('[sync] OK — repaired ridge_fit, k1_guard, and a guarded ll1_ladder')" || exit 1
}

run () {  # <tag> <cache> <label> [extra...]
  local tag="$1" cache="$2" label="$3"; shift 3
  "$PY" "$W/ll1_ladder.py" --cache "$cache" --split-json "$S/p3_selection.json" \
    --episodes-dir "$EPS" --join-file "$JOIN" --label "$label" \
    --out "$OUT/llrepG_$tag.json" --seeds $SEEDS $A "$@" \
    > "$OUT/logrepG_$tag.txt" 2>&1
  echo "LLREPG $tag rc=$?"
}

sync_scratch || exit 1
run s11250      "$S/cache_s11250/latents.pt"      "v6F-SW-30k@11250 REPAIRED"
run nullmatched "$S/cache_nullmatched/latents.pt" "RANDOM-LATENT-NULL REPAIRED"
run orcdir      "$P/cache_orcdir/latents.pt"      "GT-ORACLE-DIRECT REPAIRED"
run proxyv0     "$S/cache_s11250/latents.pt"      "C-V0-PROXY REPAIRED" --proxy-v0
echo "CHAIN_REP_GUARD_DONE"
