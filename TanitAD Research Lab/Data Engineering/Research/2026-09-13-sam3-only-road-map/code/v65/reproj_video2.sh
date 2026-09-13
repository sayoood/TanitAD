#!/bin/bash
# Thor: reprojection comparison video, display v2 (paint mask from SAM3, no boxes).
# Usage: reproj_video2.sh <c8> <version tag> <front npz suffix> <all npz suffix> <pi checks json> <front label> <all label> <footer boxes 0|1>
SM=/home/nvidia/sam3map; PY=/home/nvidia/venvs/tanitad-edge/bin/python
C8=$1; V=$2; FN=$3; AN=$4; PIC=$5; FL=$6; AL=$7; FB=$8
cd $SM/eval
env SAM3MAP_ROOT=$SM/native7 FOOTER_BOXES=$FB OMP_NUM_THREADS=4 $PY compare_gt_reproj_v2.py $C8 $SM/${C8}_$FN $SM/render5_${C8}_${V}f $SM/${C8}_$AN $SM/render5_${C8}_${V}a \
  $SM/reproj2_${V}_$C8 $SM/sam3map_score_${C8}_${FN}_gt_$V.json $SM/sam3map_score_${C8}_${AN}_gt_$V.json $PIC $FL $AL > $SM/reproj2_${V}_$C8.log 2>&1
gst-launch-1.0 -e multifilesrc location="$SM/reproj2_${V}_$C8/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/reproj2_${V}_$C8.mp4 > $SM/reproj2_${V}_$C8.gst.log 2>&1
NP=$(ls $SM/reproj2_${V}_$C8/f*.png 2>/dev/null | wc -l); NM=$(stat -c %s $SM/reproj2_${V}_$C8.mp4 2>/dev/null || echo 0)
echo "ZZREPROJ2-$V-$C8-png$NP-mp4${NM}ZZ" >> $SM/reproj_video2.log
