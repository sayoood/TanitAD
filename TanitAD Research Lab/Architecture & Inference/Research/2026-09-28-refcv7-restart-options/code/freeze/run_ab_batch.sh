#!/usr/bin/env bash
# The freeze numerics battery (restart item 2), sequential, CPU only, dev box.
#   U12 / U12b : the LAUNCH code (clean tree), 12 steps, twice -- the determinism control
#   F12        : the FROZEN tree (apply_freeze.py), 12 steps
#   U6         : the launch code, 6 steps, saving ckpt.pt at 6 (a pre-freeze checkpoint)
#   U6R        : launch code, RESUMED from U6's ckpt to 12
#   F6R        : frozen code, RESUMED from U6's ckpt CONVERTED by ckpt_freeze_convert.py, to 12
#   F6N        : frozen code, resumed from the UNCONVERTED ckpt -- must REFUSE loudly
# usage: run_ab_batch.sh <clean tree> <frozen tree> <work dir>
set -u
CLEAN="$1"; FROZEN="$2"; W="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
mkdir -p "$W"
run() {  # name tree steps [extra args...]
  local name="$1" tree="$2" steps="$3"; shift 3
  echo "$(date +%T) START $name" >> "$W/batch.log"
  "$PY" "$HERE/numerics_ab.py" --tree "$tree" --out "$W/$name" --steps "$steps" "$@" > "$W/$name.log" 2>&1
  local rc=$?   # ⛔ captured FIRST: "$(date)" inside the echo resets $? to 0 (MEASURED: F6N logged rc=0)
  echo "$(date +%T) END $name rc=$rc (the verdict is digests.json 'error', never this rc)" >> "$W/batch.log"
}
run U12 "$CLEAN" 12
run U12b "$CLEAN" 12
run F12 "$FROZEN" 12
run U6 "$CLEAN" 6 --save-every 6
# the converter builds the model from U6's OWN argv (its --out pointed at a scratch dir)
"$PY" - "$W/U6/run_spec.json" "$W/conv_argv.json" "$W/conv_scratch" <<'EOF'
import json, sys
spec = json.load(open(sys.argv[1], encoding="utf-8"))
argv = list(spec["argv"])
i = argv.index("--out"); argv[i + 1] = sys.argv[3].replace("\\", "/")
json.dump(argv, open(sys.argv[2], "w", encoding="utf-8"))
EOF
echo "$(date +%T) START convert" >> "$W/batch.log"
"$PY" "$HERE/stack/scripts/refcv7_ckpt_freeze_convert.py" --tree "$FROZEN" --argv-file "$W/conv_argv.json" \
  --ckpt-in "$W/U6/run/ckpt.pt" --ckpt-out "$W/U6_ckpt6.frozen.pt" \
  --record "$W/U6_convert_record.json" > "$W/convert.log" 2>&1
crc=$?
echo "$(date +%T) END convert rc=$crc (the verdict is the ZZCONVERT-OK token in convert.log)" >> "$W/batch.log"
run U6R "$CLEAN" 12 --resume-ckpt "$W/U6/run/ckpt.pt"
run F6R "$FROZEN" 12 --resume-ckpt "$W/U6_ckpt6.frozen.pt"
run F6N "$FROZEN" 12 --resume-ckpt "$W/U6/run/ckpt.pt"
echo "$(date +%T) BATCH DONE" >> "$W/batch.log"
