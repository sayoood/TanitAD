#!/bin/bash
# wait_latency.sh -- dev box (script polling, no model turns). Step S0 of COORD_D5_SCOPING_2026-11-19.md.
# Polls the pod every 300 s for the end marker of latency/pod_latency_run.sh (runs/latency_proxy/LATENCY_DONE.txt or LATENCY_STOP.txt; bound 8 h),
# then pulls the small outputs into latency/out_pod/ with sha256 compared on both sides. Starts nothing on the pod, deletes nothing.
# Last line: ZZ LATENCY-WAIT <state>.
set -u
cd /f/TanitAD-scratch/v8proof/latency || exit 9
P=/c/Users/Admin/v8pod/pssh.sh
C=/c/Users/Admin/v8pod/pcat.sh
O=/workspace/v8proof/runs/latency_proxy
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
mkdir -p out_pod
n=0; st=
while :; do
  s=$(pod 60 "[ -e $O/LATENCY_DONE.txt ] && echo DONE; [ -e $O/LATENCY_STOP.txt ] && echo STOP; tail -n 1 $O/latency.log 2>/dev/null | cut -c1-200; echo MARK")
  if echo "$s" | grep -q '^STOP$'; then st=STOP; break; fi
  if echo "$s" | grep -q '^DONE$'; then st=DONE; break; fi
  n=$((n+1)); [ $((n % 6)) = 1 ] && echo "[$(ts)] poll $n: $(echo "$s" | grep -v '^MARK$' | tail -n 1 | cut -c1-200)"
  [ "$n" -gt 96 ] && { echo "ZZ LATENCY-WAIT TIMEOUT $(ts)"; exit 8; }
  sleep 300
done
echo "[$(ts)] latency run ended on the pod: $st"
pod 60 "tail -n 12 $O/latency.log | cut -c1-300; cat $O/LATENCY_STOP.txt 2>/dev/null | cut -c1-300; ls $O/*/ 2>/dev/null | tr '\n' ' '"
for f in $(pod 60 "cd $O && ls */LATENCY_PROXY.json */LATENCY_PROXY.md */SELF_CHECKS_FAILED.json latency.log checks.log 2>/dev/null"); do
  l=$(echo "$f" | tr '/' '_')
  rs=$(pod 60 "sha256sum '$O/$f' 2>/dev/null | cut -c1-64")
  [ -n "$rs" ] || continue
  timeout 300 bash "$C" "$O/$f" "out_pod/$l" < /dev/null > /dev/null 2>&1
  [ "$(sha256sum "out_pod/$l" 2>/dev/null | cut -c1-64)" = "$rs" ] && echo "[$(ts)] pulled $l sha256 ${rs:0:16}" || echo "[$(ts)] ZZ SHA MISMATCH $l"
done
for m in out_pod/*LATENCY_PROXY.md; do [ -f "$m" ] && head -n 40 "$m" | cut -c1-300; done
echo "ZZ LATENCY-WAIT $st $(ts)"
