#!/bin/bash
# k2dc02tail_common.sh -- SOURCED by k2dc02tail_s6c_pull_score.sh and k2dc02tail_verdicts.sh (MM 4a3d05 build agent, 2026-11-20). Never run directly.
# PREREG_K2DC02TAIL_2026-11-20.md + AMENDMENT1. Shared settings and the pinned, blob-verified code export.
#   K2T_D    work root on the dev box (default /c/t/k2dc02tail; keep C: >= 6 GB free)
#   K2T_TIP  the pinned commit the scoring code is exported from (default 387616b3, the commit the kit landed in)
#   K2T_W    Thor run root (default /home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02)
export PATH="/usr/bin:/bin:$PATH"
W=${K2T_W:-/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02}
D=${K2T_D:-/c/t/k2dc02tail}
GD=C:/Users/Admin/tanitad-push/.git
TIP=${K2T_TIP:-387616b3299f751ed730c0a569cd72ad6785d1c0}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
E2P="TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign"
INC="FlyWheels/TanitAD_EvalFlyWheel/incoming"
SPL="$E2P/e0/DEV_HOLD_SPLIT.json"
E2TREAD="$E2P/e2/e2raw/e2tail/stageB/read"        # E2-TAIL's banked read (E2_VERDICT.json, families_s*.json, EVAL_K2.json, s6c/)
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
  local paths=(stack/ops/r9e2_read stack/ops/runs.d/r9e2_K2DC02.argv.json stack/scripts stack/tanitad taniteval "$INC/2026-10-25-refcv9-mapsel" "$INC/2026-10-25-refcv9-l2" "$SPL" "$E2TREAD")
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
