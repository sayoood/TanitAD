#!/usr/bin/env bash
# CHAIN - E-GEOM, the field-of-view x temporal-content 2x2 (ENCODER_LOCALISATION.md §2).
#
# ZERO training. ZERO Thor. ZERO pod. One banked row index + the dev-box RTX
# 4060 for the DINOv2 forwards and the projections; the ridge is CPU.
#
# THE ORDER IS NOT ARBITRARY. `wide3f` runs FIRST because it is the REPLICATION
# GATE: it is C104's exact condition, and if it does not reproduce lead_gap
# ~0.44997 then the harness is not measuring what C104 measured and nothing
# downstream may be read. The planted POSITIVE CONTROL (PC-LOCAL) runs on every
# arm, because a negative without a positive control taught this programme
# nothing (C79/D1) - and PC-LOCAL, not PC-2OBJ, because C109 measured PC-2OBJ
# INERT at the deployed pooling ratio by construction.
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
REPO="/c/Users/Admin/wt-tanitad-local"
D="$REPO/TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-encoder-localisation"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
IDX="$SP/encloc/row_index.pt"
EPS="$SP/sp2/cache/slotprobe-lead130-w120-256x640cyl"
JOIN="$SP/sp2/lead130_agents.jsonl"
SPLIT="$SP/sp2/p3_selection.json"
# torch spawns ~113 threads PER PROCESS and concurrent arms then make NO
# progress (CLAUDE.md, MEASURED 2026-07-27). Capped here, not remembered.
export OMP_NUM_THREADS=6 PYTHONUTF8=1
mkdir -p "$D/raw" "$SP/encloc"

# arm -> pooling arm on ITS OWN grid. Both give 16 cells: (4,10) on 16x40 and
# (4,4) on 16x16 - so the CELL COUNT the ridge sees is held fixed and only the
# input geometry varies, which is the whole point of the 2x2.
pool_for () { case "$1" in wide*) echo p40 ;; refa*|squash*) echo s16 ;; esac; }

for arm in wide3f wide1f refa3f refa1f; do
  CACHE="$SP/encloc/tok_$arm.pt"
  POOL="$(pool_for "$arm")"
  if [ ! -f "$CACHE" ]; then
    "$PY" "$D/code/encloc_geom_cache.py" --row-index "$IDX" \
      --episodes-dir "$EPS" --out "$CACHE" --arm "$arm" \
      > "$D/raw/log_build_$arm.txt" 2>&1
    echo "BUILD $arm rc=$?"
  else
    echo "BUILD $arm SKIPPED (cache present)"
  fi

  "$PY" "$D/code/encloc_ladder.py" --cache "$CACHE" --split-json "$SPLIT" \
    --episodes-dir "$EPS" --join-file "$JOIN" \
    --out "$D/raw/encloc_$arm.json" --label "E-GEOM $arm (pool $POOL)" \
    --arms "$POOL" --targets ego_v0 lead_gap lead_closing \
    --proj-seeds 0 1 2 > "$D/raw/log_ladder_$arm.txt" 2>&1
  echo "LADDER $arm rc=$?"

  "$PY" "$D/code/encloc_ladder.py" --cache "$CACHE" --split-json "$SPLIT" \
    --episodes-dir "$EPS" --join-file "$JOIN" \
    --out "$D/raw/encloc_pclocal_$arm.json" \
    --label "PC-LOCAL positive control, $arm (pool $POOL)" \
    --arms "$POOL" --targets lead_closing --oracle local \
    --proj-seeds 0 1 > "$D/raw/log_pclocal_$arm.txt" 2>&1
  echo "PCLOCAL $arm rc=$?"
done
echo "CHAIN_ENCLOC_DONE"
