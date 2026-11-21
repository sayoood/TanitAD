#!/usr/bin/env bash
# mapft_thor_run.sh -- DEV-BOX helper: run a LONG Thor-side kit step detached (nohup, no ssh session to lose) and wait for its MARKER.
#   mapft_thor_run.sh start <tag> <script-under-$ROOT/kit> [args...]    e.g.  start bind mapft_prelaunch.sh bind --commit C0
#   mapft_thor_run.sh wait  <tag> <ok-marker> [<fail-marker> ...] [--max-min N]   poll every 60 s; markers are the FULL tokens, e.g. ZZMAPFTBIND-OKZZ
#   mapft_thor_run.sh show  <tag> [N]                                  last N (default 25) lines of the job's output
# The job's output is $ROOT/logs/<tag>.out on Thor.  The verdict of `wait` is the MARKER COUNT read on Thor, never an exit code and never the
# marker text typed into the polling command: each marker is searched with one character bracketed (ZZMAPFTBIND-O[K]ZZ), so the polling
# command's own text cannot match it (CLAUDE.md: a polling filter that contains its pattern reports a failure that never happened).
# `wait` exits 0 only if the OK marker appears and no FAIL marker does; 1 on a FAIL marker, 2 on timeout / the job dying without a marker.
set -u
HOST=tanitad-thor; ROOT="${MAPFT_ROOT:-/home/nvidia/r8ns/mapft}"
rsh() { if [ "${MAPFT_LOCAL:-0}" = 1 ]; then bash -c "$1"; else ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "$1"; fi; }
br() { local m="$1"; local n=${#m}; echo "${m:0:$((n-3))}[${m:$((n-3)):1}]${m:$((n-2))}"; }   # bracket the third-from-last character: ZZ..-OKZZ -> ZZ..-O[K]ZZ
CMD="${1:-}"; shift || true
case "$CMD" in
start)
  TAG="${1:-}"; SCRIPT="${2:-}"; shift 2 || true
  [ -n "$TAG" ] && [ -n "$SCRIPT" ] || { echo "usage: start <tag> <script> [args...]"; exit 2; }
  case "$TAG$SCRIPT" in *[\;\&\|\`\$\(\)\<\>\"\'\ ]*) echo "REFUSED: tag/script must be plain words"; exit 2;; esac
  ARGS=""; for a in "$@"; do case "$a" in *[\;\&\|\`\$\(\)\<\>\"\'\ ]*) echo "REFUSED: argument '$a' has shell metacharacters (pass a plain word / path)"; exit 2;; esac; ARGS="$ARGS $a"; done
  rsh "mkdir -p $ROOT/logs && [ ! -e $ROOT/logs/$TAG.out ] || mv $ROOT/logs/$TAG.out $ROOT/logs/$TAG.out.prev_\$(date +%H%M%S); cd $ROOT && (nohup bash $ROOT/kit/$SCRIPT$ARGS > $ROOT/logs/$TAG.out 2>&1 < /dev/null &) ; sleep 3; test -e $ROOT/logs/$TAG.out && echo STARTED-FILE" | tr -d '\r' | grep -q STARTED-FILE \
    && echo "started $TAG: bash kit/$SCRIPT$ARGS -> $ROOT/logs/$TAG.out" || { echo "REFUSED: could not start $TAG"; exit 2; };;
wait)
  TAG="${1:-}"; OKM="${2:-}"; shift 2 || true
  FAILS=(); MAXM=600
  while [ $# -gt 0 ]; do case "$1" in --max-min) MAXM="$2"; shift 2;; *) FAILS+=("$1"); shift;; esac; done
  [ -n "$TAG" ] && [ -n "$OKM" ] || { echo "usage: wait <tag> <ok-marker> [<fail-marker>...] [--max-min N]"; exit 2; }
  FAILS+=(ZZMAPFTPRELAUNCH-REFUSEDZZ ZZMAPFTBUDGET-STOPZZ ZZGS-REFUSEDZZ ZZMAPFTPOSTRUN-REFUSEDZZ)
  OKB="$(br "$OKM")"; FB=""; for f in "${FAILS[@]}"; do FB="$FB -e $(br "$f")"; done
  t0=$(date +%s)
  while :; do
    # counts are computed THERE and returned as an opaque token QQ<ok>-<fail>QQ
    R="$(rsh "o=\$(grep -c -e '$OKB' $ROOT/logs/$TAG.out 2>/dev/null); f=\$(grep -c $FB $ROOT/logs/$TAG.out 2>/dev/null); echo QQ\${o:-0}-\${f:-0}QQ" | tr -d '\r' | grep -o 'QQ[0-9]*-[0-9]*QQ')"
    case "$R" in
      QQ[1-9]*-0QQ) echo "$TAG: OK marker $OKM seen ($R)"; exit 0;;
      QQ*-[1-9]*QQ) echo "$TAG: a FAIL marker was seen ($R). last lines:"; rsh "tail -n 8 $ROOT/logs/$TAG.out"; exit 1;;
    esac
    # the job died without a marker?  (process count in its OWN ssh call; the bracketed pattern cannot match this command's own text)
    A="$(rsh "ps -eo args | grep -c 'mapft_prelaunc[h].sh\|mapft_postru[n].sh\|make_mapft_arg[v].py'" | tr -d '\r' | tr -dc 0-9)"
    if [ "${A:-0}" = 0 ] && [ -n "$R" ]; then
      sleep 20; R2="$(rsh "o=\$(grep -c -e '$OKB' $ROOT/logs/$TAG.out 2>/dev/null); echo QQ\${o:-0}QQ" | tr -d '\r' | grep -o 'QQ[0-9]*QQ')"
      case "$R2" in QQ0QQ|"") echo "$TAG: no kit process is running and no marker was written (read $ROOT/logs/$TAG.out)"; rsh "tail -n 8 $ROOT/logs/$TAG.out"; exit 2;; esac
    fi
    [ $(( $(date +%s) - t0 )) -gt $(( MAXM * 60 )) ] && { echo "$TAG: timeout after $MAXM min ($R)"; exit 2; }
    sleep 60
  done;;
show) rsh "tail -n ${2:-25} $ROOT/logs/${1:?tag}.out";;
*) sed -n 2,11p "${BASH_SOURCE[0]}"; exit 2;;
esac
