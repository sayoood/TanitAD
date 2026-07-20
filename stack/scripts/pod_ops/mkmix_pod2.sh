#!/bin/bash
# Build realmix union dirs on pod2 (comma @ /opt/comma_epcache, pai @ /workspace).
# pai x2 approximates the main run's 0.6 physicalai draw share.
set -e
CTRAIN=$(ls -d /opt/comma_epcache/*train* | head -1)
CVAL=$(ls -d /opt/comma_epcache/*val* | head -1)
PTRAIN=$(ls -d /workspace/data/physicalai/_epcache/*train* | head -1)
PVAL=$(ls -d /workspace/data/physicalai/_epcache/*val* | head -1)
echo "comma: $CTRAIN / $CVAL"; echo "pai: $PTRAIN / $PVAL"
MT=/workspace/data/mix/_epcache/realmix-train-v1
MV=/workspace/data/mix/_epcache/realmix-val-v1
rm -rf /workspace/data/mix; mkdir -p "$MT" "$MV"
i=0; for f in "$CTRAIN"/ep_*.pt; do ln -s "$f" "$MT/ep_c$(printf %05d $i).pt"; i=$((i+1)); done
echo "comma train: $i"
i=0; for f in "$PTRAIN"/ep_*.pt; do
  ln -s "$f" "$MT/ep_p$(printf %05d $i)a.pt"; ln -s "$f" "$MT/ep_p$(printf %05d $i)b.pt"; i=$((i+1)); done
echo "pai train x2: $i"
i=0; for f in "$CVAL"/ep_*.pt; do ln -s "$f" "$MV/ep_c$(printf %05d $i).pt"; i=$((i+1)); done
i=0; for f in "$PVAL"/ep_*.pt; do ln -s "$f" "$MV/ep_p$(printf %05d $i).pt"; i=$((i+1)); done
echo "train files: $(ls "$MT" | wc -l) / val files: $(ls "$MV" | wc -l)"
echo MIX_READY
