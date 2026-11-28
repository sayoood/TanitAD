#!/usr/bin/env bash
# MM 2026-11-15: wait for the pod's `gather` to end, then pull /root/q5_416/out/* to F:/TanitAD-scratch/q5_416/run/pod_out/ and verify every file by sha256
# against a list made on the pod. Read-only on the pod. Prints one END line: PULL OK / PULL FAIL / GATHER-FAILED / TIMEOUT.
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
source /c/Users/Admin/lorapod/podenv.sh
q() { pssh "$@" 2>/dev/null | grep -v -i -E 'post-quantum|store now|upgraded' | tr -d '\r'; }
D=/f/TanitAD-scratch/q5_416/run/pod_out
for i in $(seq 1 120); do
  n="$(q 'ps -eo args | grep -c "[q]5_frames_pod.py gather"')"
  [ "$n" = "0" ] && break
  sleep 60
done
[ "$n" = "0" ] || { echo "PULL TIMEOUT gather still running $(date '+%F %T')"; exit 0; }
q 'tail -3 /root/q5_416/gather.out'
q 'grep -c -E "Traceback|Error" /root/q5_416/gather.out' | grep -q '^0$' || { echo "PULL GATHER-FAILED $(date '+%F %T') :: $(q 'tail -c 400 /root/q5_416/gather.out' | tr '\n' ' ')"; exit 0; }
q 'cd /root/q5_416/out && find . -type f | sort | xargs sha256sum' > /f/TanitAD-scratch/q5_416/run/pod_out.sha256.pod
nf=$(wc -l < /f/TanitAD-scratch/q5_416/run/pod_out.sha256.pod)
[ "$nf" -gt 0 ] || { echo "PULL FAIL empty pod sha list"; exit 0; }
mkdir -p "$D"
for d in $(q 'ls /root/q5_416/out'); do
  [ -d "$D/$d" ] && continue
  pscp -r "$POD_TARGET:/root/q5_416/out/$d" "$D/" 2>&1 | grep -v -i -E 'post-quantum|store now|upgraded'
done
cd "$D" && sha256sum -c --quiet /f/TanitAD-scratch/q5_416/run/pod_out.sha256.pod > /f/TanitAD-scratch/q5_416/run/pod_out.verify.txt 2>&1
rc=$?
echo "PULL $([ $rc = 0 ] && echo OK || echo FAIL) files=$nf rc=$rc $(date '+%F %T') :: $(head -c 300 /f/TanitAD-scratch/q5_416/run/pod_out.verify.txt | tr '\n' ' ') :: $(du -sh "$D" | cut -f1)"
