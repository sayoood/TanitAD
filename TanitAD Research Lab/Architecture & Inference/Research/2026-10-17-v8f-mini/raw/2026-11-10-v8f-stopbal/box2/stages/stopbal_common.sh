#!/usr/bin/env bash
# Sourced by every STOPBAL stage. ALL traffic to box 2 goes through "$STOPBAL_SSH" and "$STOPBAL_SFTP": absolute paths of the executables, taken from the environment, NEVER looked up on PATH
# (Git bash resolves /usr/bin first, so a PATH-order stub is silently ignored: that reached the real box 2 once). Fail closed: unset, relative, missing, or the wrong kind -> exit 16 before any step.
#   real launch : STOPBAL_SSH=$(command -v ssh) STOPBAL_SFTP=$(command -v sftp)       (files that do NOT carry the stub marker; STOPBAL_DRY unset)
#   test mode   : STOPBAL_DRY=1 and both variables name the stubs under tests/ (files that carry the marker STOPBAL-STUB-TRANSPORT). Test mode with a real client path is REFUSED, and so is a stub without test mode.
# rc contract of a stage: 0 ok; 11 gate/preflight/pin/token refused; 12 training or transfer failed; 13 disk low; 14 box 2 not idle; 15 pulled copy differs; 16 package incomplete / commit or transport missing;
# 17 ship refused; 18 pilot unreadable; 19 timeout or unreadable probe; 22 12 h rule refused; 23 blocked tag.
set -u
HERE_S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; STOPBAL_HOME_DIR="$(cd "$HERE_S/.." && pwd)"
STOPBAL_HOME_DIR="$(cygpath -m "$STOPBAL_HOME_DIR" 2>/dev/null || echo "$STOPBAL_HOME_DIR")"      # native Windows python must be able to open it
STATE="${STOPBAL_STATE:?STOPBAL_STATE}"; PY="${STOPBAL_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
PKG="${STOPBAL_PKG:-$STOPBAL_HOME_DIR}"; PULLS="${STOPBAL_PULLS:-C:/Users/Admin/t2v8f_pulls}"
B2_ROOT="${B2_ROOT:-D:/tanitad/stopbal}"; STOPBAL_COMMIT="${STOPBAL_COMMIT:-}"
B2_WIN_ROOT="${STOPBAL_B2_WIN_ROOT:-D:\\tanitad}"       # what @ROOT@ means in the job templates (the dry run points it at a fake box folder)
NEED_GB_RUN=5; NEED_GB_SHIP=1                          # DISK REQUIREMENT: constants, not measured (box 2's D: read 11 GB free recently; a run writes one checkpoint + gate output, ESTIMATED <= 1.5 GB); same values as stopbal_check.py
GPU_IDLE_MIB="${STOPBAL_GPU_IDLE_MIB:-1500}"            # the GPU counts as idle below this used memory (MiB) with no python.exe process
NOISE='post-quantum|store now|upgraded|pq.html'
say() { echo "[$(date '+%F %T')] ${STOPBAL_STAGE:-stage}: $*"; }
die() { say "$2"; printf '%s\n' "${STOPBAL_STAGE:-stage}: $2" | head -c 600 > "$STATE/stage_detail.txt"; exit "$1"; }
transport_check() {
  local v p
  for v in STOPBAL_SSH STOPBAL_SFTP; do
    p="${!v:-}"
    [ -n "$p" ] || { echo "REFUSED: $v is unset (set it to the absolute path of the executable; PATH is never searched)" >&2; return 16; }
    case "$p" in /*|[A-Za-z]:/*|[A-Za-z]:\\*) ;; *) echo "REFUSED: $v='$p' is not an absolute path" >&2; return 16;; esac
    [ -f "$p" ] || { echo "REFUSED: $v='$p' does not exist" >&2; return 16; }
    if [ "${STOPBAL_DRY:-0}" = 1 ]; then
      grep -aq 'STOPBAL-STUB-TRANSPORT' "$p" || { echo "REFUSED: STOPBAL_DRY=1 (test mode) but $v='$p' carries no stub marker (it could be the real client)" >&2; return 16; }
    else
      if grep -aq 'STOPBAL-STUB-TRANSPORT' "$p"; then echo "REFUSED: $v='$p' is a stub but STOPBAL_DRY is not 1" >&2; return 16; fi
    fi
  done
  return 0
}
transport_check || exit 16
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20)
b2_ssh() { "$STOPBAL_SSH" "${SSH_OPTS[@]}" tanitad-box2 "$@"; }
b2_sftp() { local o rc; o="$(printf '%s\n' "$1" | "$STOPBAL_SFTP" -b - tanitad-box2 2>&1)"; rc=$?; [ "$rc" -eq 0 ] || printf '%s\n' "$o" | grep -vE "$NOISE" | tail -3 >&2; return "$rc"; }
b2_put() { b2_sftp "put $1 /$2"; }
b2_get() { b2_sftp "get -r /$1 $2"; }
b2_mkdir() { local d; for d in "$@"; do b2_ssh "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force $(printf '%s' "$d" | sed 's#/#\\#g') | Out-Null\"" 2>&1 | grep -vE "$NOISE" > /dev/null; done; }
b2_exists() { [ "$(b2_ssh "powershell -NoProfile -Command \"Test-Path $1\"" 2>&1 | grep -vE "$NOISE" | tr -d '\r' | tail -1)" = True ]; }
b2_cat() { b2_ssh "type $(printf '%s' "$1" | sed 's#/#\\#g')" 2> /dev/null | grep -vE "$NOISE" | tr -d '\r'; }
b2_sha256() { b2_ssh "certutil -hashfile $(printf '%s' "$1" | sed 's#/#\\#g') SHA256" 2>&1 | grep -vE "$NOISE" | tr -d '\r ' | sed -n 2p | tr 'A-F' 'a-f'; }
b2_launch() { b2_ssh "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\tanitad\\launch.ps1 -Job $(printf '%s' "$1" | sed 's#/#\\#g') -Log $(printf '%s' "$2" | sed 's#/#\\#g')" 2>&1 | grep -vE "$NOISE" | tail -2; }
b2_wait() {     # b2_wait <done marker> <fail marker> <max polls>  -> DONE | FAIL | TIMEOUT   (a poll is STOPBAL_POLL_S seconds, default 60)
  local i; for i in $(seq 1 "${3:-1}"); do
    b2_exists "$1" && { echo DONE; return; }
    b2_exists "$2" && { echo FAIL; return; }
    sleep "${STOPBAL_POLL_S:-60}"
  done; echo TIMEOUT; }
b2_free_gb() { b2_ssh "powershell -NoProfile -Command \"[math]::Floor((Get-PSDrive D).Free/1GB)\"" 2>/dev/null | tr -d '\r' | tail -1; }
b2_python_count() { b2_ssh 'tasklist /FI "IMAGENAME eq python.exe" /NH' 2>&1 | grep -vE "$NOISE" | grep -ci 'python.exe'; }
b2_gpu_used_mib() { b2_ssh 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' 2>/dev/null | tr -d '\r ' | head -1; }
need_free_gb() {   # need_free_gb <GB>: assert D: free on box 2 BEFORE a stage; unreadable is a refusal, not a pass
  local free; free="$(b2_free_gb)"; case "$free" in ''|*[!0-9]*) die 19 "box 2 D: free space unreadable (got '$free')";; esac
  [ "$free" -ge "$1" ] || die 13 "box 2 D: free ${free} GB < $1 GB (disk-low)"; say "D: free ${free} GB (need $1)"; }
need_gpu_idle() {  # no python.exe and GPU memory below the idle threshold; unreadable is a refusal
  local n m; n="$(b2_python_count)"; case "$n" in ''|*[!0-9]*) die 19 "box 2 python count unreadable (got '$n')";; esac
  [ "$n" = 0 ] || die 14 "a python process runs on box 2: another job owns the GPU; this chain starts nothing beside it"
  m="$(b2_gpu_used_mib)"; case "$m" in ''|*[!0-9]*) die 19 "box 2 GPU memory unreadable (got '$m')";; esac
  [ "$m" -lt "$GPU_IDLE_MIB" ] || die 14 "box 2 GPU is not idle: ${m} MiB used >= $GPU_IDLE_MIB"; say "GPU idle (${m} MiB used)"; }
need_commit() {
  [ -n "$STOPBAL_COMMIT" ] || die 16 "STOPBAL_COMMIT unset: set it to the full 40-hex sha of the STOPBAL commit (2bca1c52abbc2346d7380d33ae78b324cbda9df7, the code of record of PREREG_V8F_STOPBAL)"
  case "$STOPBAL_COMMIT" in *[!0-9a-f]*) die 16 "STOPBAL_COMMIT is not lowercase hex";; esac
  [ "${#STOPBAL_COMMIT}" -eq 40 ] || die 16 "STOPBAL_COMMIT is not a full 40-hex sha (${#STOPBAL_COMMIT} characters)"; }
need_pkg() { local f miss=""; for f in "$@"; do [ -e "$PKG/$f" ] || miss="$miss $f"; done; [ -z "$miss" ] || die 16 "package incomplete, missing in $PKG:$miss"; }
render() {      # render <template> <out> <K=V>...  (CRLF out; @ROOT@ always)
  local t="$1" o="$2"; shift 2; local a=() kv; for kv in "$@"; do a+=(--set "$kv"); done
  "$PY" -I "$PKG/stopbal_check.py" render --template "$PKG/jobs/$t" --out "$o" --set "ROOT=$B2_WIN_ROOT" "${a[@]}" || die 16 "could not render $t"; }
put_verified() { b2_put "$1" "$2" || die 12 "put $2 failed"; [ "$(b2_sha256 "$2")" = "$(sha256sum "$1" | cut -d' ' -f1)" ] || die 12 "$2 differs from $1 after the put"; }
pull_verified() {  # pull_verified <remote> <local>: up to 3 attempts; the local sha256 equals box 2's own (certutil) or rc 1
  local i want got; want="$(b2_sha256 "$1")"; local HEX64='^[0-9a-f]{64}$'
  [[ "$want" =~ $HEX64 ]] || { say "no sha256 for remote $1"; return 1; }
  for i in 1 2 3; do rm -f "$2"; b2_get "$1" "$2" 2> /dev/null; got="$(sha256sum "$2" 2> /dev/null | cut -d' ' -f1)"; [ "$got" = "$want" ] && return 0; say "pull of $1 attempt $i: sha256 ${got:-none} != $want"; done; return 1; }
c_free_mb() { df -m /c | awk 'NR==2{print $4}'; }
