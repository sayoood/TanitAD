#!/bin/bash
# Build realmix union cache dirs on pod3 (hardlinks; pai x2 approximates the
# main run's 0.6 physicalai draw share — actual ~0.56, noted in RUN_README).
set -e
cd /workspace/data
CTRAIN=$(ls -d comma2k19-train-* */comma2k19-train-* 2>/dev/null | head -1)
CVAL=$(ls -d comma2k19-val-* */comma2k19-val-* 2>/dev/null | head -1)
PTRAIN=$(ls -d physicalai/_epcache/physicalai-train-* | head -1)
PVAL=$(ls -d physicalai/_epcache/physicalai-val-* | head -1)
echo "comma: $CTRAIN / $CVAL"; echo "pai:   $PTRAIN / $PVAL"
[ -n "$CTRAIN" ] && [ -n "$PTRAIN" ] || { echo MISSING_CORPUS; exit 1; }
MT=/workspace/data/mix/_epcache/realmix-train-v1
MV=/workspace/data/mix/_epcache/realmix-val-v1
rm -rf /workspace/data/mix; mkdir -p "$MT" "$MV"
i=0; for f in "$CTRAIN"/ep_*.pt; do ln "$f" "$MT/ep_c$(printf %05d $i).pt"; i=$((i+1)); done
echo "comma train linked: $i"
i=0; for f in "$PTRAIN"/ep_*.pt; do
  ln "$f" "$MT/ep_p$(printf %05d $i)a.pt"; ln "$f" "$MT/ep_p$(printf %05d $i)b.pt"; i=$((i+1)); done
echo "pai train linked x2: $i"
i=0; for f in "$CVAL"/ep_*.pt; do ln "$f" "$MV/ep_c$(printf %05d $i).pt"; i=$((i+1)); done
i=0; for f in "$PVAL"/ep_*.pt; do ln "$f" "$MV/ep_p$(printf %05d $i).pt"; i=$((i+1)); done
echo "val linked (comma+pai)"
ls "$MT" | wc -l; ls "$MV" | wc -l
cat > /workspace/experiments/RUN_README_refb.md << 'EON'
refb-30k (launched 2026-07-11 night, Sayed GO): full realmix via union hardlink dirs.
- pai x2 duplication -> draw share ~0.56 vs main run's 0.60 (MixedWindowDataset) — recipe knob, data identical.
- pai cache REBUILT from origin (build_pai_cache.py): 401 train vs pod1 402 (1 clip short) and
  split permutation differs (501- vs 502-clip list) -> cross-model evals MUST use the val-set
  INTERSECTION (episode_id manifests, post-30k) to avoid route contamination.
EON
echo MIX_READY
