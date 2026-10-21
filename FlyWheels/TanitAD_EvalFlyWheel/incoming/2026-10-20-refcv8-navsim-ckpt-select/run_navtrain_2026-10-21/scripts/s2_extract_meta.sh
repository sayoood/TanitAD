#!/bin/bash
# S2: extract the OpenScene trainval metadata (layout located with find, not assumed). Markers: ZZS2_DONE / ZZS2_FAIL_<why>
set -u
N=/home/nvidia/r8ns/navtrain; DL=/home/nvidia/r8ns/navtrain_dl
mkdir -p $N/_meta_x $N/navsim_logs
t0=$(date +%s)
tar -xzf $DL/openscene_metadata_trainval.tgz -C $N/_meta_x 2> $N/s2_tar.stderr
t1=$(date +%s)
echo "tar_secs=$((t1-t0))"
find $N/_meta_x -name '*.pkl' -printf '%h\n' | sort | uniq -c > $N/s2_pkl_dirs.txt
cat $N/s2_pkl_dirs.txt
ndir=$(wc -l < $N/s2_pkl_dirs.txt)
[ "$ndir" = 1 ] || { echo "ZZS2_FAIL_pkl_dirs_$ndir"; exit 2; }
PKL=$(awk '{print $2}' $N/s2_pkl_dirs.txt)
echo "PKLDIR $PKL"
[ -e $N/navsim_logs/trainval ] && { echo ZZS2_FAIL_target_exists; exit 3; }
mv "$PKL" $N/navsim_logs/trainval
n=$(ls $N/navsim_logs/trainval | grep -c '\.pkl$')
du -sb $N/navsim_logs/trainval
echo "ZZS2_DONE npkl=$n secs=$(( $(date +%s)-t0 ))"
