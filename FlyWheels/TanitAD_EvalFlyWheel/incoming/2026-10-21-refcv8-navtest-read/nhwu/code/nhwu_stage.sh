#!/bin/bash
# nhwu_stage.sh (Thor) -- idempotent: (re)stage what the 2026-10-04 backend recipe reads into /dev/shm/navsim, then
# ASSERT per directory that file count AND byte total equal the on-disk source (md5-verified sources: nhwu_assets 2026-10-21
# 10:06; navtest logs/maps verified by the navtest-read builder + 82/82 md5 vs dev-box navsim-crun this session).
# Writes /home/nvidia/r8ns/nhwu/stage_check.json and prints ZZSTG<n_ok>-<n_bad>ZZ.  Never touches the GPU.
set -u
A=/home/nvidia/r8ns/nhwu_assets; N=/dev/shm/navsim; NS=/home/nvidia/r8ns/navtest/nsdata; SH=/home/nvidia/r8ns/nhwu/ship
OUT=/home/nvidia/r8ns/nhwu/stage_check.json
cnt() { find "$1" -type f ! -name 'Intensity*' 2>/dev/null | wc -l; }
byt() { find "$1" -type f ! -name 'Intensity*' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%d", s}'; }
ok=0; bad=0; rec=""
stage() {  # src dst
  local s=$1 d=$2
  if [ "$(cnt $s)" != "$(cnt $d)" ] || [ "$(byt $s)" != "$(byt $d)" ]; then
    rm -rf "$d"; mkdir -p "$d"
    (cd "$s" && find . -type f ! -name 'Intensity*' -print0 | while IFS= read -r -d '' f; do mkdir -p "$d/$(dirname "$f")"; cp -a "$f" "$d/$f"; done)
  fi
  local cs cd bs bd; cs=$(cnt $s); cd=$(cnt $d); bs=$(byt $s); bd=$(byt $d)
  if [ "$cs" = "$cd" ] && [ "$bs" = "$bd" ] && [ "$cs" -gt 0 ]; then ok=$((ok+1)); v=OK; else bad=$((bad+1)); v=BAD; fi
  rec="$rec{\"src\":\"$s\",\"dst\":\"$d\",\"files\":[$cs,$cd],\"bytes\":[$bs,$bd],\"v\":\"$v\"},"
}
mkdir -p $N/exp $N/data/openscene/navsim_logs $N/data/openscene/navhard_two_stage $N/data/openscene/warmup_two_stage $N/seams $N/subsets
stage $A/exp/metric_cache_navhard_two_stage $N/exp/metric_cache_navhard_two_stage
stage $A/exp/metric_cache_warmup_two_stage $N/exp/metric_cache_warmup_two_stage
stage $A/openscene/navhard_two_stage/synthetic_scene_pickles $N/data/openscene/navhard_two_stage/synthetic_scene_pickles
stage $A/openscene/warmup_two_stage/synthetic_scene_pickles $N/data/openscene/warmup_two_stage/synthetic_scene_pickles
stage $NS/openscene/navsim_logs/test $N/data/openscene/navsim_logs/test
stage $NS/maps $N/data/maps
stage $SH/seams $N/seams/step50400_ship
stage $SH/subsets $N/subsets
# the recipe's fixed refcv7 reference-seam paths (fingerprint reference for nhwu_jobs.py)
mkdir -p $N/seams/step50400
cp -a $N/seams/step50400_ship/seam_R7_A1__navhard.npz $N/seams/step50400/; cp -a $N/seams/step50400_ship/seam_R7_A1__warmup.npz $N/seams/step50400/
m1=$(md5sum < $N/seams/step50400/seam_R7_A1__navhard.npz | cut -c1-32); m2=$(md5sum < $N/seams/step50400/seam_R7_A1__warmup.npz | cut -c1-32)
[ "$m1" = dfa0e6e6d794d9c8f4052dd966d05730 ] && [ "$m2" = 8e23b2e00ef9885ce987cf878917ed36 ] && ok=$((ok+1)) || bad=$((bad+1))
shm=$(du -sb $N | cut -f1); avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "{\"dirs\":[${rec%,}],\"refcv7_seam_md5\":[\"$m1\",\"$m2\"],\"shm_bytes\":$shm,\"mem_avail_gb\":$avail,\"ok\":$ok,\"bad\":$bad,\"at\":\"$(date -Is)\"}" > $OUT
echo "ZZSTG${ok}-${bad}ZZ"
