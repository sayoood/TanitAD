#!/usr/bin/env bash
# After run_ef.sh (never concurrently: these mutants edit the queue its tests read): the two OACC queue mutants whose anchors were respelled
# (#193, #194; the O-RECOVER stages repeat evaloacc's refusal/skip lines, so the anchors now carry evaloacc's TG / E0 lines), then --check again.
export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
EF=E:/Projects/TanitAD-artifacts/ef
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
export PYTHONPATH="$EF/tree/stack;$EF/tree/taniteval;$EF/tree/stack/scripts"
O="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/tasks/bwvv2mwc9.output"
for i in $(seq 1 360); do grep -q "ZZEF-DONE\|ZZEF-NO-SLOT" "$O" 2>/dev/null && break; sleep 20; done
grep -q ZZEF-DONE "$O" || { echo ZZEF2-RUN_EF-NOT-DONE; exit 3; }
slot() { for i in $(seq 1 720); do n=$("$PY" C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt 2 ] && return 0; sleep 20; done; return 1; }
slot || { echo ZZEF2-NO-SLOT; exit 3; }
mkdir -p $EF/m
cd $EF/tree
M="$PK/raw/2026-10-25-oacc/mutate_oacc.py"
"$PY" "$M" "$EF/tree" "$PY" --check > $EF/out/mutate_oacc_check_respelled.out 2>&1; echo "oacc anchors rc=$? $(tail -1 $EF/out/mutate_oacc_check_respelled.out)"
"$PY" "$M" "$EF/tree" "$PY" --idx 193,194 --tmp $EF/m > $EF/out/mutate_oacc_193_194.out 2>&1
echo "oacc mutants 193,194 rc=$? $(grep -ac '^RED' $EF/out/mutate_oacc_193_194.out) RED; $(tail -2 $EF/out/mutate_oacc_193_194.out | tr '\n' ' ')"
echo ZZEF2-DONE
