#!/usr/bin/env bash
# Score the CUT-IN TARGETED panel. Runs on the DEV BOX (0 GPU) against rollouts pulled
# from Thor, so it never contends with a pod that is rendering.
#
# ORDER IS DELIBERATE — controls before results:
#   1. DETERMINISM        flagship-v1/objects vs its own repeat. The only true null.
#   2. CROSS-VERSION      starts 75 & 90 of the new `empty` run vs the BANKED scene-2
#                         `empty` run. Proves (or refutes) that the renderer edit which
#                         landed on the pod between the two panels is a no-op on the
#                         default path.
#   3. objects vs empty, per arm       — the experiment.
#   4. flagship vs refc, per condition — the POSITIVE control: the same windows must be
#                                        able to separate SOMETHING, or a null is a dead
#                                        instrument rather than a finding.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
RES="$HERE/results/scene2-cutin-targeted"
BANK="$HERE/results/scene2-realclose"
TRACKS="${TRACKS:?set TRACKS to sequence_tracks.json for scene 7c72937c}"
export PYTHONPATH="$(cygpath -w "$REPO/stack");$(cygpath -w "$REPO/stack/scripts");$(cygpath -w "$REPO/taniteval");$(cygpath -w "$HERE")"
mkdir -p "$RES/metrics" "$RES/verification"
cd "$HERE"

R () { echo "$RES/rollouts/rollouts_$1_$2.json"; }
G="--geometry $BANK/SCENE2_GEOMETRY_renderable.json --tracks $TRACKS \
   --renderable-from $BANK/actor_map_7c72937c.json"

echo "=== 1. NEGATIVE CONTROL: determinism ==="
$PY verify_cutin_panel.py --what determinism \
    --a "$(R flagship-v1 objects)" \
    --b "$RES/rollouts/rollouts_flagship-v1_objects_REPEAT.json" \
    --out "$RES/verification/determinism.json"

echo; echo "=== 2. CROSS-VERSION render equivalence (banked vs new, shared starts) ==="
$PY verify_cutin_panel.py --what cross_version \
    --a "$(R flagship-v1 empty)" \
    --b "$BANK/rollouts/rollouts_flagship-v1_empty.json" \
    --out "$RES/verification/cross_version_render.json"

HASH="$(grep gsplat_renderer "$RES/CODE_MD5.txt" | awk '{print $1}')"
V="--cutin-verdict $RES/CUTIN_IS_REAL.json"

echo; echo "=== 3. objects vs empty, per arm ==="
for ARM in flagship-v1 refc-base; do
  $PY cutin_targeted.py --a "$(R $ARM objects)" --b "$(R $ARM empty)" $G $V \
      --track-hash "$HASH" \
      --label "$ARM objects - empty (cut-in-targeted starts)" \
      --out "$RES/metrics/${ARM}_objects_vs_empty.json"
done

echo; echo "=== 4. POSITIVE CONTROL: flagship vs refc, per condition ==="
for COND in objects empty; do
  $PY cutin_targeted.py --a "$(R flagship-v1 $COND)" --b "$(R refc-base $COND)" $G $V \
      --track-hash "$HASH" \
      --label "flagship-v1 - refc-base within $COND (cut-in-targeted starts)" \
      --out "$RES/metrics/flagship_vs_refc_${COND}.json"
done
echo "CUTIN_TARGETED_SCORING_DONE"
