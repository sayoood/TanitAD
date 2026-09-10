#!/usr/bin/env bash
# The seed pair that makes the >=GT bar's effect readable.
# Seed 0 already has bar-ON and bar-OFF. This adds seed 1 for both, giving a 2x2
# over (bar, seed) -- so the lever is read against THIS RIG'S OWN run-to-run noise,
# not against a single-seed difference.
#
# ⛔ WHY THIS IS NOT OPTIONAL: an arm with ZERO levers moved read "separably worse"
# on 5 of 9 family metrics on this programme's rig, and H-ESTIM-SEED-1 puts the
# false-positive rate for "separated" at 14.3 %. The paired bootstrap resamples
# EPISODES with the models held fixed, so it is structurally blind to training
# variance. A replicate is the only thing that sees it.
set -u
CK="C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt"
WANT_MD5="8f10d6f934f4199e11ddc7352e074939"
WANT_BYTES="1250838325"
OUT="C:/Users/Admin/tanitad-data/rl-pilot"
REPO="C:/Users/Admin/tanitad-rlrun"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/stack"

GOT_B=$(stat -c %s "$CK" 2>/dev/null)
[ "$GOT_B" != "$WANT_BYTES" ] && { echo "ZZGATE-CKPT-BYTES-${GOT_B:-none}ZZ"; exit 2; }
GOT=$(md5sum "$CK" | awk '{print $1}')
[ "$GOT" != "$WANT_MD5" ] && { echo "ZZGATE-MD5-REFUSEDZZ"; exit 3; }

RESOLVED=$("$PY" -c "import tanitad;print(tanitad.__file__)" 2>&1 | tr -d '\r')
case "$RESOLVED" in
  *tanitad-rlrun*stack*tanitad*) echo "[pair] tree OK: $RESOLVED" ;;
  *) echo "ZZGATE-WRONG-TREE-${RESOLVED}ZZ"; exit 5 ;;
esac

# ⛔ The pilot MUST expose --gt-bar, or this whole pair is the old baseline twice.
if ! grep -q -- "--gt-bar" "$REPO/stack/scripts/rl_pilot_refc21.py"; then
  echo "ZZGATE-NO-GTBAR-FLAGZZ"; exit 6
fi

BUSY=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ci "pyth""on")
[ "${BUSY:-1}" -gt 0 ] && { echo "ZZGATE-GPU-BUSY-REFUSEDZZ"; exit 4; }

COMMON=( --ckpt "$CK"
  --train-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74"
  --train-agents "$OUT/pilot_train_agents.jsonl"
  --val-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836"
  --val-agents "$OUT/pilot_val_agents.jsonl"
  --steps 2000 --batch 2 --reward default )

run_arm () {
  local tag="$1"; shift
  echo "[pair] $tag $(date -u +%FT%TZ)"
  timeout 3600 "$PY" stack/scripts/rl_pilot_refc21.py "${COMMON[@]}" --out "$OUT/$tag" "$@" \
    > "$OUT/$tag.log" 2>&1
  local rc=$?
  # ⛔ the ARTIFACT is the evidence, never $?
  if [ -s "$OUT/$tag/pilot_summary.json" ]; then echo "ZZ${tag}-OK-${rc}ZZ"; else echo "ZZ${tag}-NOSUMMARY-${rc}ZZ"; fi
}

run_arm "gtbar-s1"  --seed 1 --gt-bar
run_arm "off-s1"    --seed 1
echo "ZZPAIR-DONEZZ"
