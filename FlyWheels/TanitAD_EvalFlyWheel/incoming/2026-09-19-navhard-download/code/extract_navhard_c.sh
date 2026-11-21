#!/usr/bin/env bash
# ⛔ MSYS GNU tar reads "C:/x" as a REMOTE HOST "C" (rsh syntax) -> "Cannot connect to C: resolve failed";
# --force-local is REQUIRED. MEASURED 2026-09-19: the first run failed exactly so, and only the
# count assertions below revealed it (the wrapper exited 0).
# Extract navhard_two_stage onto C: (NTFS) for the NavSim runtime E1 built at C:/Users/Admin/navsim-crun.
# WHY C: — MEASURED 2026-09-19 by stream E1: the D:-resident (exFAT, 1 MiB clusters) NavSim venv was
# I/O-starved (`import numpy` 23,350 ms on D: vs 324 ms on C:), and extracting the 8,196 small pickle
# files onto D: took ~75 min. The verified ARCHIVES stay on D: as the durable copy; this is a working copy.
# WHAT — the scene pickles in full, and ONLY cam_l0/cam_f0/cam_r0 from the two sensor archives: those
# are the three cameras the TanitAD 120-degree stitch reads (build_navsim_eval.py CAMS); the NavSim
# scorer reads no sensors at all. Counts are asserted on the result, never inferred from the exit code.
set -u
SRC="C:/Users/Admin/navsim/data/navsim-v2"
DST="C:/Users/Admin/navsim-crun/data/openscene"
OUT="$DST/navhard_two_stage"
mkdir -p "$DST"
t0=$(date +%s)
tar --force-local -xzf "$SRC/navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz" -C "$DST"; rc_p=$?
for a in curr_sensors hist_sensors; do
  tar --force-local -xzf "$SRC/navsim_v2.2_navhard_two_stage_${a}.tar.gz" -C "$DST" --wildcards \
      'navhard_two_stage/sensor_blobs/*/CAM_F0/*' \
      'navhard_two_stage/sensor_blobs/*/CAM_L0/*' \
      'navhard_two_stage/sensor_blobs/*/CAM_R0/*'
  echo "rc_${a}=$?"
done
t1=$(date +%s)
np=$(find "$OUT/synthetic_scene_pickles" -type f -name '*.pkl' | wc -l)
nm=$(find "$OUT/openscene_meta_datas" -type f | wc -l)
nf0=$(find "$OUT/sensor_blobs" -type f -path '*/CAM_F0/*' | wc -l)
nl0=$(find "$OUT/sensor_blobs" -type f -path '*/CAM_L0/*' | wc -l)
nr0=$(find "$OUT/sensor_blobs" -type f -path '*/CAM_R0/*' | wc -l)
nz=$(find "$OUT" -type f -size 0 | wc -l)
nlogs=$(find "$OUT/sensor_blobs" -mindepth 1 -maxdepth 1 -type d | wc -l)
bytes=$(du -sb "$OUT" | awk '{print $1}')
cat > "$OUT/EXTRACT_DONE.json" <<JSON
{"what": "navhard_two_stage working copy on C: (pickles full; sensors cam_l0/cam_f0/cam_r0 only)",
 "source_archives_dir": "$SRC",
 "source_sha256": {"scene_pickles": "a2e429a01e9a93afd9d1f753ed306f469c094292627467182d35fae9fb2ecad7",
                   "curr_sensors": "1b95ccf2724ae3aff455cc91a5a6dd13e07937af170abac08bb4170255a19f2b",
                   "hist_sensors": "7482a0d967915d5781bb213afdda0a192d7eb283931043da06aebdf2bdc1ea82"},
 "expected_from_tar_listing": {"synthetic_scene_pickles": 5462, "openscene_meta_datas": 2731},
 "counts": {"synthetic_scene_pickles": $np, "openscene_meta_datas": $nm, "CAM_F0": $nf0, "CAM_L0": $nl0,
            "CAM_R0": $nr0, "sensor_log_dirs": $nlogs, "zero_byte_files": $nz},
 "bytes_on_disk": $bytes, "wall_s": $((t1-t0)), "rc_pickles": $rc_p,
 "ok": $([ "$np" -eq 5462 ] && [ "$nm" -eq 2731 ] && [ "$nz" -eq 0 ] && [ "$nf0" -gt 0 ] && [ "$nf0" -eq "$nl0" ] && [ "$nf0" -eq "$nr0" ] && echo true || echo false)}
JSON
cat "$OUT/EXTRACT_DONE.json"
