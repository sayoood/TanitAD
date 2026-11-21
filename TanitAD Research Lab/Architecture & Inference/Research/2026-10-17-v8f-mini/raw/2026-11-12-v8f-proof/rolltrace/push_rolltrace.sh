#!/usr/bin/env bash
# push_rolltrace.sh -- dev box: push the kit of the trace export to the pod (new root rolltrace, folder kit), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
# Prints no helper script. Pushes: export_traces.py and podrun_export_traces.sh.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/rolltrace
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
timeout 60 bash $P "mkdir $GP && mkdir $GP/kit && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in export_traces.py podrun_export_traces.sh; do push rolltrace/$f $GP/kit/$f; done
echo "ZZ PUSH-ROLLTRACE bad=$bad"
