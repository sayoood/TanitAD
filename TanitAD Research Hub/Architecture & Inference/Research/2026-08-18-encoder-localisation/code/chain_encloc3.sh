#!/usr/bin/env bash
# CHAIN 3 - PART 2: rungs 2-3 on the TRAINED refa-dinov2-4b checkpoint.
#
# Order: build (one GPU pass banks all 6 latent arms) -> ladder per arm
# (--arms cells) -> the two DECISION arms re-run at ridge seeds 1/2 for the
# C103 spread (the cells arm has no RP-seed axis; the remaining stochastic
# element is the inner alpha-selection split, seeded by --ridge-seed; the
# rand-init arms carry their own 3-seed axis already).
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
DUR="C:/Users/Admin/tanitad-caches"
REPO="/c/Users/Admin/wt-tanitad-local"
D="$REPO/TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-encoder-localisation"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
EPS="$SP/sp2/cache/slotprobe-lead130-w120-256x640cyl"
JOIN="$SP/sp2/lead130_agents.jsonl"
SPLIT="$SP/sp2/p3_selection.json"
P2="$DUR/encloc-20260818/part2"
export OMP_NUM_THREADS=6 PYTHONUTF8=1
mkdir -p "$P2" "$D/raw"

if [ ! -f "$P2/cells_rung3_h4.pt" ]; then
  "$PY" "$D/code/encloc_part2_latents.py" \
    --ckpt "$DUR/refa-dinov2-4b/ckpt.pt" \
    --row-index "$DUR/encloc-20260818/row_index.pt" \
    --episodes-dir "$EPS" \
    --tok-cache "$DUR/encloc-20260818/tok_refa1f.pt" \
    --out-dir "$P2" > "$D/raw/log_part2_build.txt" 2>&1
  echo "P2BUILD rc=$?"
else
  echo "P2BUILD SKIPPED (caches present)"
fi

for arm in rung2_trained rung2_rand0 rung2_rand1 rung2_rand2 rung3_h1 rung3_h4; do
  "$PY" "$D/code/encloc_ladder.py" --cache "$P2/cells_$arm.pt" \
    --split-json "$SPLIT" --episodes-dir "$EPS" --join-file "$JOIN" \
    --out "$D/raw/encloc_p2_$arm.json" \
    --label "PART2 $arm (trained ckpt latent, cells probe)" \
    --arms cells --targets ego_v0 lead_gap lead_closing \
    --proj-seeds 0 --dump-preds "$D/raw/preds_p2_$arm.pkl" \
    > "$D/raw/log_p2_$arm.txt" 2>&1
  echo "P2LADDER $arm rc=$?"
done

# ridge-seed spread on the two decision arms
for arm in rung2_trained rung3_h1; do
  for rs in 1 2; do
    "$PY" "$D/code/encloc_ladder.py" --cache "$P2/cells_$arm.pt" \
      --split-json "$SPLIT" --episodes-dir "$EPS" --join-file "$JOIN" \
      --out "$D/raw/encloc_p2_${arm}_rs$rs.json" \
      --label "PART2 $arm ridge-seed $rs" \
      --arms cells --targets ego_v0 lead_gap lead_closing \
      --proj-seeds 0 --ridge-seed "$rs" \
      > "$D/raw/log_p2_${arm}_rs$rs.txt" 2>&1
    echo "P2RS $arm rs$rs rc=$?"
  done
done

cd "$DUR/encloc-20260818" && md5sum part2/*.pt >> MANIFEST.md5 && echo "MANIFEST appended"
echo "CHAIN_ENCLOC3_DONE"
