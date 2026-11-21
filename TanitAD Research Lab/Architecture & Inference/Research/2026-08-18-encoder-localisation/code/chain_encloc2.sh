#!/usr/bin/env bash
# CHAIN 2 - the mechanism-separating control, and the PAIRED deltas.
#
# Pass 1 (chain_encloc.sh) measured the 2x2. Two things it could not do:
#
#  1. SEPARATE "lost field of view" FROM "lost token grid". `refa*` changes BOTH
#     the 51.4 deg crop AND the 16x40 -> 16x16 grid (640 -> 256 tokens, plus a
#     7% anisotropic stretch). `squash1f` keeps the FULL 120 deg field but puts
#     it on the SAME 16x16 square grid, so `refa1f` vs `squash1f` is the field
#     alone and `squash1f` vs `wide1f` is the grid alone. Without it a collapse
#     on `refa1f` is not attributable, and an unattributed collapse is exactly
#     the C104 failure (a mechanism narrated instead of ablated).
#
#  2. PAIR the arms. The ladder's `deltas_vs_p40` only pairs arms inside ONE
#     cache; these arms live in different caches. `--dump-preds` writes the
#     per-row predictions so the paired episode-cluster bootstrap can run across
#     caches - the estimator the pre-registration commits to, and NOT a
#     difference of two independent CIs.
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
REPO="/c/Users/Admin/wt-tanitad-local"
D="$REPO/TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-encoder-localisation"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
IDX="$SP/encloc/row_index.pt"
EPS="$SP/sp2/cache/slotprobe-lead130-w120-256x640cyl"
JOIN="$SP/sp2/lead130_agents.jsonl"
SPLIT="$SP/sp2/p3_selection.json"
export OMP_NUM_THREADS=6 PYTHONUTF8=1

pool_for () { case "$1" in wide*) echo p40 ;; refa*|squash*) echo s16 ;; esac; }

# --- the control arm ----------------------------------------------------
CACHE="$SP/encloc/tok_squash1f.pt"
if [ ! -f "$CACHE" ]; then
  "$PY" "$D/code/encloc_geom_cache.py" --row-index "$IDX" \
    --episodes-dir "$EPS" --out "$CACHE" --arm squash1f \
    > "$D/raw/log_build_squash1f.txt" 2>&1
  echo "BUILD squash1f rc=$?"
fi
"$PY" "$D/code/encloc_ladder.py" --cache "$CACHE" --split-json "$SPLIT" \
  --episodes-dir "$EPS" --join-file "$JOIN" \
  --out "$D/raw/encloc_squash1f.json" \
  --label "E-GEOM squash1f (full 120 deg on the 16x16 square grid)" \
  --arms s16 --targets ego_v0 lead_gap lead_closing --proj-seeds 0 1 2 \
  --dump-preds "$D/raw/preds_squash1f.pkl" \
  > "$D/raw/log_ladder_squash1f.txt" 2>&1
echo "LADDER squash1f rc=$?"

"$PY" "$D/code/encloc_ladder.py" --cache "$CACHE" --split-json "$SPLIT" \
  --episodes-dir "$EPS" --join-file "$JOIN" \
  --out "$D/raw/encloc_pclocal_squash1f.json" \
  --label "PC-LOCAL positive control, squash1f" \
  --arms s16 --targets lead_closing --oracle local --proj-seeds 0 1 \
  > "$D/raw/log_pclocal_squash1f.txt" 2>&1
echo "PCLOCAL squash1f rc=$?"

# --- prediction dumps for the four 2x2 arms ------------------------------
for arm in wide3f wide1f refa3f refa1f; do
  "$PY" "$D/code/encloc_ladder.py" --cache "$SP/encloc/tok_$arm.pt" \
    --split-json "$SPLIT" --episodes-dir "$EPS" --join-file "$JOIN" \
    --out "$D/raw/encloc_$arm.json" --label "E-GEOM $arm" \
    --arms "$(pool_for "$arm")" --targets ego_v0 lead_gap lead_closing \
    --proj-seeds 0 1 2 --dump-preds "$D/raw/preds_$arm.pkl" \
    > "$D/raw/log_ladder_${arm}_preds.txt" 2>&1
  echo "PREDS $arm rc=$?"
done

"$PY" "$D/code/encloc_summarise.py" --raw "$D/raw" \
  --out "$D/raw/encloc_summary.json" --baseline wide3f \
  > "$D/raw/log_summarise.txt" 2>&1
echo "SUMMARISE rc=$?"
echo "CHAIN_ENCLOC2_DONE"
