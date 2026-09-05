#!/bin/sh
# Bank ONE arm's outputs into the research package, md5-verified per file.
# ⛔ Never bank a checkpoint (ckpt.pt is ~230 MB and the repo is not a model store);
# bank the CONFIG, the METRICS and the LOG -- what a reader needs to re-derive the
# numbers and to check the arm actually ran the lever it claims.
# usage: sh bank_arm.sh <arm-name>
set -u
A=$1
W=/c/Users/Admin/run_wbank
# ⚠️ MSYS: the `/g/…` form does NOT resolve for writes on this box (mkdir reports
# "File exists" while a redirect into the same path reports "No such file or
# directory"). Use the drive-letter form, which is what works here.
PKG="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-withheld-bank-panel/raw"
D="$PKG/$A"
mkdir -p "$D"

copy () {
  src=$1; dst=$2
  [ -f "$src" ] || { echo "MISSING $src"; return 1; }
  i=1
  while [ $i -le 6 ]; do
    # ⚠️ `cp` onto this mount can fail with "cannot create regular file: File
    # exists" for a path that DOES NOT EXIST (verified by ls). The redirect
    # succeeds where cp does not, so copy by redirect and check the md5.
    cat "$src" > "$dst" 2>/dev/null
    a=$(md5sum < "$src" | cut -d' ' -f1)
    b=$(md5sum < "$dst" 2>/dev/null | cut -d' ' -f1)
    if [ -n "$b" ] && [ "$a" = "$b" ]; then echo "OK   $(basename "$dst")  md5=$a  bytes=$(wc -c < "$dst")"; return 0; fi
    echo "retry$i $(basename "$dst")"
    i=$((i+1))
  done
  echo "FAILED $dst"; return 1
}

copy "$W/arms/$A/config.json"   "$D/config.json"
copy "$W/arms/$A/metrics.jsonl" "$D/metrics.jsonl"
copy "$W/arms/$A/summary.json"  "$D/summary.json"
copy "$W/arms/$A.log"           "$D/train.log"
# content assertion: the arm is only banked if its metrics are non-trivial
n=$(grep -c '' "$D/metrics.jsonl" 2>/dev/null || echo 0)
echo "ZZBANK-$A-rows${n}-ZZ"
[ "$n" -ge 10 ] || echo "⛔ $A: metrics.jsonl has $n rows -- suspiciously short, do not quote"
