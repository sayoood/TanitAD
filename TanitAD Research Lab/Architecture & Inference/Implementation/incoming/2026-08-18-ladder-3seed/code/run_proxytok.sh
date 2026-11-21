#!/usr/bin/env bash
# ⭐ THE MISSING TRIVIAL-PROXY CONTROL — C-V0 on the **TOKENS** window set.
#
# ⛔ WHY THIS ARM DID NOT EXIST BEFORE. C100's re-read carries ONE C-V0 arm
# (`proxyv0`), fitted on the CELLS cache — n_eval 2221–3023. Three of its 15 arms
# (`tok11250`, `tok11250null`, `cells_tokwin`) are fitted on the TOKENS cache —
# n_eval 1103–1507. Comparing those three against the cells-window C-V0 would be
# an UNPAIRED comparison across different window sets, so C92's mandatory
# trivial-proxy control was simply ABSENT on 33 of the 165 rows.
# ⇒ This arm supplies it: the identical ridge, the identical split, seeds and
# estimator, on the TOKENS windows, with the 2 048-dim latent replaced by the
# single scalar `v0` (`--proxy-v0` reads `rows[i]["v0"]`, so the window set comes
# from the cache and the feature does not).
#
# ⚠️ IT IS A NEW ROW SET, NOT PART OF THE 165. Reported as 11 ADDITIONAL rows and
# never folded into the C100 inventory counts.
#
# ⛔ Run for BOTH repair routes, kept separate, never pooled (C100/C103).
# ⛔ T0-DIAGNOSTIC.
set -u
ROOT="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
INC="$ROOT/TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
OUT="$INC/2026-08-18-ladder-3seed/raw"
SCR="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
W="$SCR/ll"; S="$SCR/sp2"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTHONPATH="$ROOT/taniteval"
A="--alphas 0.01 0.1 1 10 100 1000 10000 100000 1000000 10000000"
SEEDS="${LL3_SEEDS:-0 1 2}"
for MODE in unpen centred; do
  DEST="$OUT/reread_$MODE"; mkdir -p "$DEST"
  "$PY" "$W/ll1_ladder.py" --cache "$S/cache_tok11250/latents.pt" \
    --split-json "$S/p3_selection.json" \
    --episodes-dir "$S/cache/slotprobe-lead130-w120-256x640cyl" \
    --join-file "$S/lead130_agents.jsonl" \
    --label "C-V0-PROXY@11250 TOKENS-WINDOWS" \
    --out "$DEST/ll3_proxytok.json" --seeds $SEEDS --fit-mode "$MODE" $A \
    --proxy-v0 > "$DEST/log3_proxytok.txt" 2>&1
  echo "PROXYTOK[$MODE] rc=$?"
done
echo "PROXYTOK_DONE"
