#!/bin/bash
# Thor: reprojection comparison video (front only vs all 7, GT map drawn into the cameras) for one clip and one map version.
# Usage: reproj_video.sh <c8> <version tag, e.g. v65c> <front npz suffix, e.g. v61sf> <all npz suffix, e.g. v61s> <pi checks json> <front label> <all label>
SM=/home/nvidia/sam3map; PY=/home/nvidia/venvs/tanitad-edge/bin/python
C8=$1; V=$2; FN=$3; AN=$4; PIC=$5; FL=$6; AL=$7
cd $SM/eval
env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY compare_gt_reproj.py $C8 $SM/${C8}_$FN $SM/render5_${C8}_${V}f $SM/${C8}_$AN $SM/render5_${C8}_${V}a \
  $SM/reproj_${V}_$C8 $SM/sam3map_score_${C8}_${FN}_gt_$V.json $SM/sam3map_score_${C8}_${AN}_gt_$V.json $PIC $FL $AL > $SM/reproj_${V}_$C8.log 2>&1
gst-launch-1.0 -e multifilesrc location="$SM/reproj_${V}_$C8/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/reproj_${V}_$C8.mp4 > $SM/reproj_${V}_$C8.gst.log 2>&1
NP=$(ls $SM/reproj_${V}_$C8/f*.png 2>/dev/null | wc -l); NM=$(stat -c %s $SM/reproj_${V}_$C8.mp4 2>/dev/null || echo 0)
echo "ZZREPROJ-$V-$C8-png$NP-mp4${NM}ZZ" >> $SM/reproj_video.log
