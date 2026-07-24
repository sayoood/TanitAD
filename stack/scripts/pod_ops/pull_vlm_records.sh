#!/usr/bin/env bash
# Rescue the VLM semantic-labeling records from tanitad-pod3 into the repo.
#
# WHY THIS EXISTS. The previous VLM corpus (595 records) lived only on
# `tanitad-pod3:/workspace` and was flagged as a reconstruction risk in the
# 2026-07-20 repo audit — the program's dominant failure mode is good work
# stranded outside git. The production run here is LONG (hours), so there is a
# real window in which the records exist only on a pod. This makes rescuing
# them a single command that anyone can run at any time, including mid-run:
# the consolidation is idempotent and a partial run yields a valid, smaller
# corpus (the stratified manifest is episode-ordered, and a prefix stays
# stratum-balanced to within ~4 pp at n=300 of 600 — measured).
#
# Usage:  bash stack/scripts/pod_ops/pull_vlm_records.sh [dest_dir]
# Default dest: the 2026-07-21 intake directory.

set -uo pipefail
POD="${POD:-tanitad-pod3}"
DEST="${1:-TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-21-vlm-production-semantic}"
REMOTE=/root/vlmprod

mkdir -p "$DEST"

echo "== consolidating per-window JSON into JSONL on $POD =="
ssh -o ConnectTimeout=60 "$POD" 'cd '"$REMOTE"' && /workspace/venv/bin/python - <<PY
import json, glob, os
SPECS = [("valfull", "val_full"), ("trainstrat", "train_strat"),
         ("phase1", "p1_v1"), ("phase1", "p1_v2"), ("phase1", "p1_v2b"),
         ("phase1", "ab_base"), ("phase1", "ab_dense_early"),
         ("phase1", "ab_wide_cheap"), ("phase1", "ab_dense_hist"),
         ("phase1", "ab_base_randenum"),
         ("probe", "r2_as_written"), ("probe", "r2_right_first")]
for d, tag in SPECS:
    fs = sorted(glob.glob(os.path.join(d, tag, "ep_*_t*.json")))
    if not fs:
        continue
    out = os.path.join(d, tag + ".jsonl")
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for f in fs:
            try:
                r = json.load(open(f, encoding="utf-8"))
            except Exception as e:            # a kill mid-write leaves a stub
                print("SKIP unparseable", f, e)
                continue
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{out}: {n} records")
PY'

echo "== pulling =="
# Every manifest is named windows.json on the pod, so each is pulled to an
# EXPLICIT destination name. `$DEST/windows.json` is reserved for the enum-order
# probe's manifest, because `vlm_compare_score.py` requires that exact filename —
# an earlier version of this script clobbered it and broke the probe's
# reproduction command.
pull() {  # remote-relative-path  dest-name
  scp -q -o ConnectTimeout=120 "$POD:$REMOTE/$1" "$DEST/$2" 2>/dev/null \
    && echo "  ok $2" || echo "  -- absent $1"
}
pull valfull/val_full.jsonl           val_full.jsonl
pull valfull/windows.json             val_full_windows.json
pull valfull/run_val_full.json        run_val_full.json
pull trainstrat/train_strat.jsonl     train_strat.jsonl
pull trainstrat/windows.json          train_strat_windows.json
pull trainstrat/run_train_strat.json  run_train_strat.json
pull phase1/enums.json                enums.json
pull probe/windows.json               windows.json          # the probe manifest
pull probe/windows.json               probe_windows.json
# the kinematic stratum census of the whole 2376-episode train corpus
# (21,393 candidate windows with their v2.1 route labels). ~8 MB, and it
# is what makes re-sampling the train draw a pod-free operation.
pull trainstrat/candidates.json       train_candidate_census.json

for tag in p1_v1 p1_v2 p1_v2b ab_base ab_dense_early ab_wide_cheap \
           ab_dense_hist ab_base_randenum; do
  scp -q -o ConnectTimeout=120 "$POD:$REMOTE/phase1/$tag.jsonl" "$DEST/" 2>/dev/null \
    && echo "  ok $tag"
done

echo "== record counts in $DEST =="
for f in "$DEST"/*.jsonl; do
  [ -f "$f" ] && printf '  %-28s %s\n' "$(basename "$f")" "$(wc -l < "$f")"
done
echo
echo "NEXT: score without the pod —"
echo "  python stack/scripts/vlm_semantic_score.py --out \"$DEST\" --arms val_full --json /tmp/val.json"
echo "  python stack/scripts/vlm_labels_to_lake.py --jsonl \"$DEST/val_full.jsonl\" \\"
echo "      --windows-out \"$DEST/scenario_strata_val.jsonl\" --sidecars-out \"$DEST/sidecars_val\""
