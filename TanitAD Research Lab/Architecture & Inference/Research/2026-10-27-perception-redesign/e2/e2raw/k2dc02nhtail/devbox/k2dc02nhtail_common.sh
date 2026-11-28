#!/bin/bash
# k2dc02nhtail_common.sh -- SOURCED by k2dc02nhtail_s6c_pull_score.sh, k2dc02nhtail_verdicts.sh and k2dc02nhtail_ema_verdicts.sh. Never run directly. Template: k2dc02s1tail_common.sh (landed, PREREG_K2DC02S1).
# PREREG_K2DC02NH_2026-11-20.md (sec. 2 to 4). Shared settings and the pinned, blob-verified code export.
#   K2T_D    work root on the dev box (default /f/refcv9_e3/k2dc02nhtail; F: had ~8 GB free when this was written, the s6c step refuses unless the drive of $D keeps >= 6 GB after the pull)
#   K2T_TIP  the pinned commit the scoring code is exported from (default e30e7b6e, the mirror's remote tip when this kit was written: it carries stack/ops/runs.d/r9e2_K2DC02nh.argv.json,
#            stack/ops/r9e2_read/e2read_launch_k2dc02nh.sh and r9_e2_dose_verdict.py with the constants the lever rule imports). The lever rule is k2dc02nhtail_lever.py in THIS kit, so the pinned dose tool needs no K2DC02nh entry in ARM_ARGV.
#   K2T_W    Thor run root of THIS read (default /home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02nh); K2T_W0 = the K2DC02 tail's root, whose rolls are the PINNED comparators (read only)
export PATH="/usr/bin:/bin:$PATH"
W=${K2T_W:-/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02nh}
W0=${K2T_W0:-/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02}
PIN_SHA=3d367ad922e4b250c1782b6204dad7b56ce9c0d83eae5955fe1959c5512f311c      # sha256 of $W0/K2DC02S1_CMP_PIN.sha256 (the K2DC02-tail comparator pin: K0t = cmp/E2_K0_pf, K2DC02t = stageB/E2_K2_pf; verified equal on Thor 2026-11-21)
D=${K2T_D:-/f/refcv9_e3/k2dc02nhtail}
GD=C:/Users/Admin/tanitad-push/.git
TIP=${K2T_TIP:-e30e7b6e910fd85febb9ab342c784f200ceb429b}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
E2P="TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign"
INC="FlyWheels/TanitAD_EvalFlyWheel/incoming"
SPL="$E2P/e0/DEV_HOLD_SPLIT.json"
K2DC02TREAD="$E2P/e2/e2raw/k2dc02tail/result/read"   # the K2DC02-tail banked read (EVAL_K2.json, s6c/) = the K2DC02t read: the comparator in K2's slot
E2TSRC=/f/refcv9_e3/e2tail_stageB/s6c              # E2-TAIL's dev-box pull (K0 slot): read-only source, never moved
LEAD_BLOCK=C:/Users/Admin/r9e2_psync_block/b1_eval_lead_block_k4_dt0p5_psync.npz
LEAD_SHA=85941014426ee1760db2fa5cf79bd44b01854ddd2b1811bc6b196f9f3e356a68
mkdir -p "$D/logs" "$D/sha" 2>/dev/null
W2() { cygpath -w "$1"; }
th() { timeout "${TH_TO:-120}" ssh -n -o ConnectTimeout=20 -o BatchMode=yes tanitad-thor "$1" 2>/dev/null | tr -d '\r'; }
# dev-box commit gate (memory devbox-commit-limit-oom): >= 5632 MB free commit before any python; polls 300 s, 8 h max
gate() { local k=0 CF; while :; do CF=$(powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1KB)" 2>/dev/null | tr -d '\r ')
  [ -n "$CF" ] && [ "$CF" -ge 5632 ] 2>/dev/null && { say "commit free $CF MB >= 5632 ($1)"; return 0; }
  k=$((k + 1)); [ $k -ge 96 ] && return 1; sleep 300; done; }
# export_code: git archive of the PINNED commit + a batch blob check of EVERY exported file (hash-object --no-filters vs ls-tree). Sets C.
export_code() {
  C="$D/code_${TIP:0:12}"
  git --git-dir=$GD cat-file -e "$TIP^{commit}" 2>/dev/null || return 11
  local paths=(stack/ops/r9e2_read stack/ops/runs.d/r9e2_K2DC02.argv.json stack/ops/runs.d/r9e2_K2DC02nh.argv.json stack/scripts stack/tanitad taniteval "$INC/2026-10-25-refcv9-mapsel" "$INC/2026-10-25-refcv9-l2" "$SPL" "$K2DC02TREAD")
  if [ ! -e "$C/.verified" ]; then
    [ -e "$C" ] && return 12   # a half-made export is never reused: the MM deletes it by path
    mkdir -p "$C" && git --git-dir=$GD -c core.autocrlf=false archive "$TIP" "${paths[@]}" | tar --force-local -x -C "$C" || return 13
    git --git-dir=$GD ls-tree -r "$TIP" -- "${paths[@]}" > "$D/sha/code_ls.tsv" || return 14
    cut -f2- "$D/sha/code_ls.tsv" > "$D/sha/code_paths.txt"
    awk -F'[ \t]' '{print $3}' "$D/sha/code_ls.tsv" > "$D/sha/code_want.txt"
    ( cd "$C" && git --git-dir=$GD hash-object --no-filters --stdin-paths < "$D/sha/code_paths.txt" ) > "$D/sha/code_got.txt" || return 15
    cmp -s "$D/sha/code_want.txt" "$D/sha/code_got.txt" || return 16
    echo "$TIP $(wc -l < "$D/sha/code_want.txt") files blob-verified" > "$C/.verified"
  fi
  return 0
}
# lever_tool_guard: the lever rule (k2dc02nhtail_lever.py) imports seed_numbers / decide / BAR_M / BAND_M from the pinned r9_e2_dose_verdict.py and refuses if the constants differ from its own.
# The guard checks the tool defines them, that the nh argv json is in the export, and that it does NOT carry --r9-hist-supervise-geom. Returns 0 or prints why and returns 1.
lever_tool_guard() {
  local t="$C/stack/scripts/r9_e2_dose_verdict.py" a="$C/stack/ops/runs.d/r9e2_K2DC02nh.argv.json"
  [ -s "$t" ] || { echo "no $t in the pinned export"; return 1; }
  grep -qE '^BAR_M = 0[.]0025' "$t" && grep -qE '^BAND_M = 0[.]0010' "$t" && grep -q '^def seed_numbers(' "$t" && grep -q '^def decide(' "$t" || { echo "the pinned r9_e2_dose_verdict.py (tip ${TIP:0:12}) lacks BAR_M = 0.0025 / BAND_M = 0.0010 / seed_numbers / decide: the lever rule would refuse"; return 1; }
  [ -s "$a" ] || { echo "no $a in the pinned export"; return 1; }
  ! grep -q -- '--r9-hist-supervise-geom' "$a" || { echo "$a carries --r9-hist-supervise-geom: that is K2DC02, not K2DC02nh"; return 1; }
  return 0
}
