#!/usr/bin/env bash
# ⭐ O13 WIRING SMOKE — the unit tests validate the LOSS; this validates the WIRING.
#
# ⛔ WHY THIS RUNS BEFORE ANY REAL ARM. The last launch of a freshly-shipped term
# died with `KeyError: ep_idx` after the queue was armed: I had grep-verified the
# marker in the file I shipped but not its DEPENDENCY. A 12-step smoke costs two
# minutes and catches every wiring fault a 30,000-step run would surface at step 1.
#
# TWO ARMS, and the second is the control:
#   ON  --w-o13-ego 0.1  -> the log MUST carry o13_excess / o13_shuffled /
#                           o13_on_z_t, and o13_shuffled must sit near 1.0
#   OFF --w-o13-ego 0    -> the log MUST NOT carry ANY o13 key. The term is
#                           guarded by `if w.o13_ego:`, so weight 0 must be a
#                           no-op rather than "computed and multiplied by zero".
set -u
SPD="$(cd "$(dirname "$0")" && pwd)"
M="C:/Users/Admin/tanitad-mirror/stack"
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
CACHE="$SPD/sp2/cache/v7tiny-heldout24-w120-256x640cyl"
COMMON="--stage S-W --v2-cache $CACHE --frame-h 256 --frame-w 640 --patch 16
  --enc-dim 128 --enc-depth 3 --enc-heads 4 --pred-dim 256 --pred-depth 3
  --pred-heads 4 --readout-grid 4 --readout-grid-w 8 --readout-dim 64
  --window 6 --horizons 1 2 4 --o1-k 4 --o5-k 8 --d-tac 128 --d-str 64
  --steps 12 --batch 4 --v2-lru 4 --log-every 2 --save-every 100000 --seed 0
  --spectrum-accum 43 --sigreg-slices 512 --no-require-parity"

run () {  # $1 = tag, $2 = o13 weight
  local out="$SPD/o13smoke_$1"
  rm -rf "$out"
  echo "===== ARM $1 (--w-o13-ego $2) ====="
  ( cd "$M" && PYTHONPATH="$M" PYTHONIOENCODING=utf-8 \
      "$PY" scripts/train_v6_staged.py --out "$out" $COMMON \
      --w-o13-ego "$2" --o13-k 4 ) 2>&1 | tail -25
  echo "----- o13 keys in the log -----"
  if [ -f "$out/train_log.jsonl" ]; then
    tail -1 "$out/train_log.jsonl" | tr ',' '\n' | grep -i "o13" || echo "(none)"
  else
    echo "NO LOG WRITTEN"
  fi
  echo
}

run on 0.1
run off 0
echo "===== VERDICT ====="
ON="$SPD/o13smoke_on/train_log.jsonl"; OFF="$SPD/o13smoke_off/train_log.jsonl"
# ⛔ COUNT THE TERM LIST, NOT THE WHOLE LINE. The first version grepped the raw
# log for "o13" and matched the CONFIG record (which echoes --o13-k for every
# run, including the control), so the OFF arm scored 1 and the verdict read FAIL
# on a passing control. A filter that matches its own configuration echo is the
# same defect as a monitor grepping for the pattern its command contains.
n_on=$(grep -c '"terms": \[[^]]*"o13"' "$ON" 2>/dev/null | head -1)
n_off=$(grep -c '"terms": \[[^]]*"o13"' "$OFF" 2>/dev/null | head -1)
n_on=${n_on:-0}; n_off=${n_off:-0}
echo "ZZO13-ON=${n_on} O13-OFF=${n_off}ZZ"
if [ "$n_on" -gt 0 ] && [ "$n_off" -eq 0 ]; then
  echo "PASS - the term fires when weighted and is a true no-op at weight 0"
else
  echo "FAIL - on=$n_on (want >0), off=$n_off (want 0)"
fi
