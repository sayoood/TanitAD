#!/usr/bin/env bash
# One-command collection of the pod3 scale-up results into the repo pod_artifacts/.
# Run from the dev box if the run finishes after this agent's session ends.
# Usage:  bash collect_results.sh
set -u
SSH="ssh"; SCP="scp"; H=tanitad-pod3
DEST="$(cd "$(dirname "$0")" && pwd)/pod_artifacts"
mkdir -p "$DEST"
echo "collecting from $H -> $DEST"
$SSH $H 'cat /workspace/tmp/yt_scaleup/w*/pointers.jsonl 2>/dev/null > /workspace/tmp/yt_scaleup/pointers_all.jsonl' 2>/dev/null
for f in results/results_scaleup_downstream.json results/harvest_manifest.json results/DONE \
         results/pseudo_labels_w0.json run.log pointers_all.jsonl; do
  b=$(basename "$f"); [ "$b" = "pointers_all.jsonl" ] && b=pointers.jsonl
  $SCP "$H:/workspace/tmp/yt_scaleup/$f" "$DEST/$b" 2>/dev/null && echo "  got $b"
done
echo "--- verdict ---"
if [ -f "$DEST/results_scaleup_downstream.json" ]; then
  python -c "import json;d=json.load(open('$DEST/results_scaleup_downstream.json'));m=d['meta'];a=d['arms_mean_std'];print('clips',m.get('pretrain_youtube_clips'),'seeds',m.get('seeds'));print('floor speed_r2',a['floor']['speed_r2'],'| pseudo_yt',a['pseudo_yt']['speed_r2']);print('beats_all',d.get('pseudo_yt_beats_floor_all_seeds'),'ci_all',d.get('ci_excludes_0_all_seeds'));print('VERDICT',d.get('verdict'))" 2>/dev/null || cat "$DEST/results_scaleup_downstream.json"
else
  echo "results JSON not present yet — run still going. Check: ssh $H 'tail /workspace/tmp/yt_scaleup/run.log; cat /workspace/tmp/yt_scaleup/results/DONE'"
fi
