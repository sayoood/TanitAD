#!/bin/bash
# pull_and_analyze.sh -- after chain_p2p_thor.sh has written ZZCEND: pull every arm's dump + meta from Thor (md5-verified),
# pull the fan-roll s0 os-only dump (C3 at s0; s1 already in thor_ref/fan_s1), run the pre-registered analysis, render RESULT.md.
L=/c/Users/Admin/refcv8_pf/perc2plan; R=$L/run; mkdir -p $R
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
pull () {  # $1 thor dir  $2 local dir
  mkdir -p "$2"; timeout 900 ssh -n tanitad-thor "cd $1 && tar cf - --exclude=fan_extras.npz --exclude='donor_bank_*.pt' ." | tar xf - -C "$2"
  timeout 300 ssh -n tanitad-thor "cd $1 && find . -type f ! -name fan_extras.npz ! -name 'donor_bank_*.pt' -exec md5sum {} + | sort -k2" > "$2.thor_md5"
  (cd "$2" && find . -type f -exec md5sum {} + | sed 's/ \*/  /' | sort -k2) > "$2.local_md5"
  if diff -q <(awk '{print $1,$2}' "$2.thor_md5") <(awk '{print $1,$2}' "$2.local_md5") >/dev/null && [ -s "$2.thor_md5" ]; then echo "PULLOK $2 $(wc -l < $2.thor_md5)"; else echo "PULLBAD $2"; fi
}
timeout 60 ssh -n tanitad-thor 'cat /home/nvidia/r8ns/perc2plan/out/chain.log' > $R/chain.log
for S in 1 0; do for A in BASE AGT_ZERO BEV_ZERO BOTH_ZERO AGT_SHUF BEV_SHUF; do pull /home/nvidia/r8ns/perc2plan/out/${A}_s$S $R/${A}_s$S; done; done
[ -d $L/thor_ref/fan_s0 ] || pull /home/nvidia/r8ns/fanceil/out/s0/dump $L/thor_ref/fan_s0
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval'
$PY -X utf8 $L/code/perc2plan_analyze.py --root $R --seeds 0,1 --out-json $L/PERC2PLAN.json \
    --fan-dump "s1=$L/thor_ref/fan_s1,s0=$L/thor_ref/fan_s0" --n-boot 2000 > $L/analyze.log 2>&1
tail -1 $L/analyze.log
$PY -X utf8 $L/code/perc2plan_result_md.py --json $L/PERC2PLAN.json --smoke $L/SMOKE_CONTROLS.json --out $L/RESULT.md
