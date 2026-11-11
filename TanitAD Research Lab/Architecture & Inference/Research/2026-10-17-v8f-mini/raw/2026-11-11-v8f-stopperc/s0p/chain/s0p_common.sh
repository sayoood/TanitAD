#!/usr/bin/env bash
# Sourced by s0p_chain.sh. ALL traffic to box 2 goes through "$S0P_SSH" and "$S0P_SFTP": absolute paths, never looked up on PATH (Git bash resolves /usr/bin first, so a PATH-order stub is silently ignored).
# Fail closed: unset, relative, missing, or the wrong kind -> return 16.  Real launch: S0P_SSH / S0P_SFTP name files WITHOUT the stub marker, S0P_DRY unset.
# Test mode: S0P_DRY=1 and both name the stubs under tests/ (files carrying the marker S0P-STUB-TRANSPORT).  Test mode with a real client is REFUSED, and so is a stub without test mode.
# Everything here lives on box 2 only under D:/tanitad/stopperc (the PI's personal machine: nothing else is read for content, written, or removed).
set -u
HERE_C="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; HERE_C="$(cygpath -m "$HERE_C" 2>/dev/null || echo "$HERE_C")"
S0P_DIR="${S0P_SRC:-$HERE_C/..}"                                  # the folder holding s0p_lib.py / s0p_extract_train.py (shipped to box 2)
PY="${S0P_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"; TOOL="$HERE_C/s0p_chain_tool.py"
B2_S0P="D:/tanitad/stopperc/s0p"; B2_OUT="D:/tanitad/stopperc/s0p_out"; B2_MD="D:/tanitad/stopperc/s0p_jobs"
NOISE='post-quantum|store now|upgraded|pq.html'
say() { echo "[$(date '+%F %T')] ${S0P_STAGE:-stage}: $*"; }
transport_check() {
  local v p
  for v in S0P_SSH S0P_SFTP; do
    p="${!v:-}"
    [ -n "$p" ] || { echo "REFUSED: $v is unset (set it to the absolute path of the executable; PATH is never searched)" >&2; return 16; }
    case "$p" in /*|[A-Za-z]:/*|[A-Za-z]:\\*) ;; *) echo "REFUSED: $v='$p' is not an absolute path" >&2; return 16;; esac
    [ -f "$p" ] || { echo "REFUSED: $v='$p' does not exist" >&2; return 16; }
    if [ "${S0P_DRY:-0}" = 1 ]; then
      grep -aq 'S0P-STUB-TRANSPORT' "$p" || { echo "REFUSED: S0P_DRY=1 (test mode) but $v='$p' carries no stub marker (it could be the real client)" >&2; return 16; }
    else
      if grep -aq 'S0P-STUB-TRANSPORT' "$p"; then echo "REFUSED: $v='$p' is a stub but S0P_DRY is not 1" >&2; return 16; fi
    fi
  done
  return 0
}
transport_check || return 16 2>/dev/null || exit 16
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20)
w() { printf '%s' "$1" | sed 's#/#\\#g'; }                          # D:/a/b -> D:\a\b
b2_ssh() { "$S0P_SSH" "${SSH_OPTS[@]}" tanitad-box2 "$@"; }
b2_sftp() { local o rc; o="$(printf '%s\n' "$1" | "$S0P_SFTP" -b - tanitad-box2 2>&1)"; rc=$?; [ "$rc" -eq 0 ] || printf '%s\n' "$o" | grep -vE "$NOISE" | tail -3 >&2; return "$rc"; }
b2_put() { b2_sftp "put $1 /$2"; }
b2_get() { b2_sftp "get -r /$1 $2"; }
b2_mkdir() { local d; for d in "$@"; do b2_ssh "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force $(w "$d") | Out-Null\"" 2>&1 | grep -vE "$NOISE" > /dev/null; done; }
b2_exists() { [ "$(b2_ssh "powershell -NoProfile -Command \"Test-Path $(w "$1")\"" 2>&1 | grep -vE "$NOISE" | tr -d '\r' | tail -1)" = True ]; }
b2_cat() { b2_ssh "type $(w "$1")" 2> /dev/null | grep -vE "$NOISE" | tr -d '\r\000'; }
b2_sha256() { b2_ssh "certutil -hashfile $(w "$1") SHA256" 2>&1 | grep -vE "$NOISE" | tr -d '\r ' | sed -n 2p | tr 'A-F' 'a-f'; }
b2_launch() { b2_ssh "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\tanitad\\launch.ps1 -Job $(w "$1") -Log $(w "$2")" 2>&1 | grep -vE "$NOISE" | tail -2; }
b2_wait() {     # b2_wait <done marker> <fail marker> <max polls>  -> DONE | FAIL | TIMEOUT   (a poll is S0P_POLL_S seconds, default 60)
  local i; for i in $(seq 1 "${3:-1}"); do
    b2_exists "$1" && { echo DONE; return; }
    b2_exists "$2" && { echo FAIL; return; }
    sleep "${S0P_POLL_S:-60}"
  done; echo TIMEOUT; }
b2_free_bytes() { b2_ssh "powershell -NoProfile -Command \"[string][int64](Get-PSDrive D).Free\"" 2>/dev/null | tr -d '\r' | tail -1; }
b2_dir_bytes() { b2_ssh "powershell -NoProfile -Command \"[string][int64]((Get-ChildItem $(w "$1") -Recurse -File | Measure-Object Length -Sum).Sum)\"" 2>/dev/null | tr -d '\r' | tail -1; }
b2_remove_dir() { b2_ssh "powershell -NoProfile -Command \"Remove-Item -Recurse -Force $(w "$1")\"" 2>&1 | grep -vE "$NOISE" > /dev/null; }
b2_remove_file() { b2_ssh "powershell -NoProfile -Command \"Remove-Item -Force $(w "$1")\"" 2>&1 | grep -vE "$NOISE" > /dev/null; }
put_verified() { b2_put "$1" "$2" || return 1; [ "$(b2_sha256 "$2")" = "$(sha256sum "$1" | cut -d' ' -f1)" ]; }
pull_verified() {  # pull_verified <remote> <local>: up to 3 attempts; the local sha256 equals box 2's own (certutil) or rc 1
  local i want got; want="$(b2_sha256 "$1")"; local HEX64='^[0-9a-f]{64}$'
  [[ "$want" =~ $HEX64 ]] || { say "no sha256 for remote $1"; return 1; }
  for i in 1 2 3; do rm -f "$2"; b2_get "$1" "$2" 2> /dev/null; got="$(sha256sum "$2" 2> /dev/null | cut -d' ' -f1)"; [ "$got" = "$want" ] && return 0; say "pull of $1 attempt $i: sha256 ${got:-none} != $want"; done; return 1; }
c_free_mb() { if [ -n "${S0P_FAKE_C_FREE_MB:-}" ]; then echo "$S0P_FAKE_C_FREE_MB"; else df -m /c | awk 'NR==2{print $4}'; fi; }
