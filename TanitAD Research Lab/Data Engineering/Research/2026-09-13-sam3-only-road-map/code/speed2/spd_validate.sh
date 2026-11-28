#!/bin/bash
# Thor: validation clips for the fp16 measures + the PI's long quality video (2026-09-14). Bars unchanged (SPEC + addendum 3).
#  0) regression check of the camera-name parameter: fast driver, spdF4a flags, 8 frames -> bit-identical to spdF4a
#  1) native7 sequences for two more LiDAR clips (6924358fafe0, 0d90d20036a3)
#  2) streaming driver, three arms per clip set: spdSA = exact reference (today's pipeline), spdS4 = spdF4a flags, spdS5 = spdF5a flags
#     on the 2 LiDAR clips (CAM_FW) and the 8 production-format front clips (native f-theta, ego-path ground, CAM_F0)
#  3) PI checks on the LiDAR clips; bars for spdS4 / spdS5 vs spdSA on all 10 clips
#  4) configuration for the video: spdS5 flags if spdS5 passes on every validation clip (it already passed on the two test clips), else spdS4
#  5) render every clip (the 2 test clips + 10 validation clips), join into one long video
SM=/home/nvidia/sam3map; PY=/home/nvidia/venvs/tanitad-edge/bin/python
PP=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map
LOG=$SM/spd_validate.log
NAT="6924358fafe0 0d90d20036a3"
FRN="1f1f05ca011d b5d9b91e6637 b975bf8ebf95 41f10d46174e 26015e788849 f63e215a546a c1dc66b6ae42 d672fc17a315"
BASE="PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MODE=fast"
echo "start $(date -Is)" >> $LOG
cd $SM/eval
# 0) regression check
env SAM3MAP_ROOT=$SM/native7 $BASE HALF=fp16enc BATCH=19 ASYNC=2 FRAMES=8 $PY sam3map_front_fast.py spdCAMchk 73495082f98b > $SM/spdCAMchk.log 2>&1
EQ=$($PY npz_equal.py $SM/73495082f98b_spdF4araw $SM/73495082f98b_spdCAMchkraw | grep -o "ZZNPZEQ-[0-9-]*ZZ")
echo "ZZVAL-CAMCHK-${EQ}ZZ" >> $LOG
[ "$EQ" = "ZZNPZEQ-8-8-0ZZ" ] || { echo "ZZVAL-ABORT-CAMCHKZZ" >> $LOG; exit 1; }
mv $SM/73495082f98b_spdCAMchkraw $SM/_aside_spdCAMchkraw_$(date +%H%M%S)
# 1) native7 sequences
env PYTHONPATH=$PP $PY build_native7_seq.py 6924358fafe0 0d90d20036a3 > $SM/build_native7_val.log 2>&1
echo "ZZVAL-BUILD-$(ls -d $SM/native7/seq_6924358fafe0/*/ 2>/dev/null | wc -l)-$(ls -d $SM/native7/seq_0d90d20036a3/*/ 2>/dev/null | wc -l)ZZ" >> $LOG
# 2) arms
arm() {   # tag root cam clips flags...
  local T=$1 R=$2 CAM=$3 CL=$4; shift 4
  env SAM3MAP_ROOT=$R SAM3MAP_CAM=$CAM $BASE "$@" $PY sam3map_front_stream.py $T $CL > $SM/front_stream_${T}_$(basename $R).log 2>&1
  echo "ZZVAL-ARM-$T-$(basename $R)-$(grep -c ZZFRONTSTREAM $SM/front_stream_${T}_$(basename $R).log)ZZ" >> $LOG
  cp $SM/front_stream_$T.json $SM/front_stream_${T}_$(basename $R).json
}
arm spdSA $SM/native7 CAM_FW "$NAT" ASYNC=2
arm spdS4 $SM/native7 CAM_FW "$NAT" HALF=fp16enc BATCH=19 ASYNC=2
arm spdS5 $SM/native7 CAM_FW "$NAT" HALF=fp16enc BATCH=19 BB_HALF=fp16 ASYNC=2
arm spdSA $SM/front_native CAM_F0 "$FRN" ASYNC=2
arm spdS4 $SM/front_native CAM_F0 "$FRN" HALF=fp16enc BATCH=19 ASYNC=2
arm spdS5 $SM/front_native CAM_F0 "$FRN" HALF=fp16enc BATCH=19 BB_HALF=fp16 ASYNC=2
# 3) PI checks (LiDAR clips) and bars
for C8 in $NAT; do
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY pi_checks.py $C8 $SM/${C8}_spdSAc $SM/pi_checks_${C8}_val.json 48 -5,40,-15,15 \
      spdSA=$SM/render5_${C8}_spdSAr spdS4=$SM/render5_${C8}_spdS4r spdS5=$SM/render5_${C8}_spdS5r > $SM/pi_checks_${C8}_val.log 2>&1
  echo "ZZVAL-PIC-$C8-$(grep -c ZZPICHECKS $SM/pi_checks_${C8}_val.log)ZZ" >> $LOG
done
SPECS="CAM_FW:6924358fafe0:val CAM_FW:0d90d20036a3:val"
for C8 in $FRN; do SPECS="$SPECS CAM_F0:$C8"; done
$PY spd_eval_clips.py $SM/spd_eval_val.json spdSA spdS4,spdS5 $SPECS > $SM/spd_eval_val.log 2>&1
grep "^spdS" $SM/spd_eval_val.log | sed "s/^/ZZVAL-VERDICT /" >> $LOG
# 4) configuration for the video
if grep -q "^spdS5: NUMERICAL-PASS" $SM/spd_eval_val.log; then FT=spdS5; FT0=spdF5a; FL="fp16 encoder + fp16 backbone + batched prompts + async + streaming"
else FT=spdS4; FT0=spdF4a; FL="fp16 encoder + batched prompts + async + streaming"; fi
echo "ZZVAL-VIDEO-CONFIG-${FT}ZZ" >> $LOG
# 5) bars json for the video, render, join
$PY - <<EOF > $SM/long_bars.log 2>&1
import json, numpy as np
from pathlib import Path
SM = Path("/home/nvidia/sam3map"); out = {}
v = json.loads((SM / "spd_eval_val.json").read_text())
names = {1: "drivable", 2: "line", 3: "crosswalk", 5: "edge"}
for c8, r in v["arms"]["$FT"]["clips"].items():
    b = r["bars"]; out[c8] = {"N1": b["N1"]["value"], "N4": b["N4"]["value"], **{names[int(k)]: x for k, x in b["N2"]["value"].items() if int(k) in names}}
e = json.loads((SM / "spd_eval.json").read_text())
for c8 in ("73495082f98b", "4fbd97b6a4b7"):
    r = e["arms"]["$FT0"][c8]; b = r["bars"]
    out[c8] = {"N1": b["N1"]["value"], "N4": b["N4"]["value"], **{names[int(k)]: x for k, x in b["N2"]["value"].items() if int(k) in names}}
    out[c8]["s_ref"] = e["timing"]["spd0"][c8]["s_per_frame"]; out[c8]["s_fast"] = e["timing"]["$FT0"][c8]["s_per_frame"]
for root in ("native7", "front_native"):                                  # validation clips: combined-arm pace only (the reference arm
    p = SM / f"front_stream_${FT}_{root}.json"                             # here is the exact arm, not today's 3.4 s pipeline)
    if p.exists():
        d = json.loads(p.read_text())
        for c8, cr in d["clips"].items():
            out.setdefault(c8, {})["s_fast"] = cr["extract_s"] / cr["frames"]
(SM / "long_bars.json").write_text(json.dumps(out, indent=1))
print("ZZBARS", len(out))
EOF
LONG=$SM/long_fast; mkdir -p $LONG
ORDER="73495082f98b:native7:CAM_FW:$FT0:spd0:LiDAR 4fbd97b6a4b7:native7:CAM_FW:$FT0:spd0:LiDAR 6924358fafe0:native7:CAM_FW:$FT:spdSA:LiDAR 0d90d20036a3:native7:CAM_FW:$FT:spdSA:LiDAR"
for C8 in $FRN; do ORDER="$ORDER $C8:front_native:CAM_F0:$FT:spdSA:ego-path"; done
N=$(echo $ORDER | wc -w); i=0; jobs_run=0
for item in $ORDER; do
  i=$((i+1)); IFS=: read C8 R CAM FTAG RTAG GR <<< "$item"
  GL=$([ "$GR" = "LiDAR" ] && echo "LiDAR" || echo "ego path, no LiDAR")
  env OMP_NUM_THREADS=2 $PY render_fast_long.py $SM/long_frames/$(printf %02d $i)_$C8 $SM/$R $CAM $C8 $FTAG $RTAG "$FL" $SM/long_bars.json $i $N "$GL" > $SM/long_render_$C8.log 2>&1 &
  jobs_run=$((jobs_run+1))
  if [ $jobs_run -ge 4 ]; then wait -n; jobs_run=$((jobs_run-1)); fi
done
wait
echo "ZZVAL-RENDER-$(grep -l ZZRENDERLONG $SM/long_render_*.log | wc -l)-of-${N}ZZ" >> $LOG
k=0
for d in $(ls -d $SM/long_frames/*/ | sort); do
  for f in $(ls $d/f*.jpg | sort); do ln -sf $f $LONG/f$(printf %05d $k).jpg; k=$((k+1)); done
done
gst-launch-1.0 -e multifilesrc location="$LONG/f%05d.jpg" index=0 caps="image/jpeg,framerate=(fraction)5/1" ! jpegdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/sam3map_fast_long.mp4 > $SM/sam3map_fast_long.gst.log 2>&1
echo "ZZVAL-VIDEO-frames$k-mp4$(stat -c %s $SM/sam3map_fast_long.mp4 2>/dev/null || echo 0)ZZ" >> $LOG
echo "ZZVAL-ENDZZ $(date -Is)" >> $LOG
