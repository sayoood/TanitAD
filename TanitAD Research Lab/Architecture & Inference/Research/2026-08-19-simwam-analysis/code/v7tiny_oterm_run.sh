#!/usr/bin/env bash
# v7-tiny O-TERM LADDER — "reintroduce them in v7tiny to understand their
# effects" (PI, 2026-08-22).
#
# BASELINE is the banked `v7tiny_fixed` arm: ALL of o1..o6 at v6's own weights,
# RESIDUAL_HEAD_INIT_SCALE=1e-3. Every arm below changes exactly ONE thing
# against it. Same trainer, same data, same seed, same 2,000 steps.
#
# ⭐ THE QUESTION THESE ANSWER. G2 measured the fixed arm at EM -0.019 (h=4) --
# essentially AT hold -- and a fitted LINEAR oracle on the same held-out clips
# tops out near +0.02. So the per-tick latent increment is ~98 % unpredictable
# and the binding constraint is the ENCODER, not the predictor. The obvious
# suspect is O6/SIGReg, which pushes the latent toward an isotropic Gaussian:
# isotropy and temporal smoothness pull against each other.
#
# PRIORITY ORDER (a killed run still yields value): the top suspect runs first.
#   1. no-o6     --w-o6 0                  isotropy pressure removed
#   2. o5-only   everything else 0         a PURE next-latent objective
#   3. no-o1     --w-o1-* 0                O1 is the largest term by magnitude
#
# Each arm is followed IMMEDIATELY by the oracle probe, which is the diagnostic
# that actually answers the question (is dz predictable AT ALL?) -- G2 alone
# cannot distinguish "predictor is bad" from "target is noise".
#
# ⛔ Markers are OPAQUE (ZZ..ZZ) and disjoint from anything grepped, so a
# monitor cannot match its own echoed command.
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
REPO="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
SPD="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
CACHE="$SPD/sp2/cache/slotprobe-lead130-w120-256x640cyl"
HELD="$SPD/sp2/cache/v7tiny-heldout24-w120-256x640cyl"
export PYTHONPATH="$REPO/stack"; export PYTHONIOENCODING=utf-8
cd "$REPO/stack" || exit 1

common="--stage S-W --v2-cache $CACHE --no-require-parity
  --frame-h 256 --frame-w 640 --patch 16
  --enc-dim 128 --enc-depth 3 --enc-heads 4
  --pred-dim 256 --pred-depth 3 --pred-heads 4
  --readout-grid 4 --readout-dim 128
  --window 6 --horizons 1 2 4 --o1-k 4 --o5-k 6
  --d-tac 128 --d-str 64 --steps 2000 --batch 4 --v2-lru 6
  --log-every 100 --seed 0"

run_arm () {                       # $1 = name, rest = the ONE varied flag set
  name="$1"; shift
  echo "ZZARM-${name}-STARTZZ"
  "$PY" scripts/train_v6_staged.py --out "$SPD/v7tiny_${name}" $common "$@" \
    > "$SPD/v7tiny_${name}.log" 2>&1
  rc=$?
  echo "ZZARM-${name}-TRAIN-${rc}ZZ"
  [ $rc -ne 0 ] && return
  "$PY" "$SPD/v7tiny_oracle.py" --arm "${name}" --clips 24 \
    --frames-per-clip 120 --cache "$HELD" \
    --out "$SPD/v7tiny_oracle_${name}.json" \
    > "$SPD/v7tiny_oracle_${name}.log" 2>&1
  echo "ZZARM-${name}-ORACLE-$?ZZ"
}

run_arm no-o6    --w-o6 0
run_arm o5-only  --w-o6 0 --w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0 \
                 --w-o2 0 --w-o3 0
run_arm no-o1    --w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0
echo "ZZLADDER-DONEZZ"
