#!/usr/bin/env bash
# early_rl_s0.sh : dev-box side. An EARLY per-base look at the anchored arms of note PREREG_V8F_PROOF_ADD37_N3 at base 0 (chain actcons_c0): RLFT_s0, RLA_s0, RLAm_s0.
# It names nothing (the names of the note need both bases; the waiter wait_actcons_c.sh makes the read of record into actcons/read_rl). Its only use: to know hours earlier
# whether the anchored arm keeps the level, so that the next arm is ready when chain actcons_c1 ends. Pod CPU dumps (the op of the waiter, idempotent: the waiter reuses them),
# pull with sha256, then the reader on one base into actcons/read_rl_early/. Deletes nothing. Last line: ZZ EARLY-RL DONE.
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
H=/f/TanitAD-scratch/v8proof; W=F:/TanitAD-scratch/v8proof; PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
D=$H/imaghead/dump; R=$H/actcons/read_rl_early; mkdir -p "$D" "$R"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
ARMS="RLFT_s0 RLA_s0 RLAm_s0"
for try in 1 2 3 4 5 6; do
  out=$(timeout 1500 bash "$P" "bash -s -- actcons_c0 tree_leadsup $ARMS" < $H/imaghead/ops/pod_imag_dump_arm.sh 2>/dev/null | sed '/hf_/d' | tr -d '\r')
  echo "$out" | cut -c1-330 >> "$R/pod_dump.out"
  echo "$out" | grep -q "DUMP-NOT-RUN" || break
  echo "[$(ts)] pod memory low, dump later"; sleep 300
done
echo "$out" | grep "c2b\|^   c1\|no checkpoint" | cut -c1-300
echo "$out" | grep "^   SHA " | while read -r _ a s1 s2; do
  for pr in "DUMP_$a.npz:$s1" "DUMP_$a.npz.json:$s2"; do f=${pr%%:*}; s=${pr##*:}
    [ -f "$D/$f" ] && [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && continue
    timeout 300 bash $C /workspace/v8proof/runs/imag_z/dump/$f "$D/$f" < /dev/null > /dev/null 2>&1
    [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && echo "[$(ts)] pulled $f" || echo "[$(ts)] ZZ SHA MISMATCH $f"
  done
done
for a in RLFT RLA RLAm; do
  [ -f "$D/DUMP_${a}_s0.npz" ] || { echo "[$(ts)] no dump for $a"; continue; }
  ( cd $H/imaghead && "$PY" imag_read.py read --dump "$W/imaghead/dump/DUMP_${a}_s0.npz" --arm "$a" --seeds 0 --out-dir "$W/actcons/read_rl_early/$a" > "$R/$a.log" 2>&1 )
  echo "[$(ts)] read $a (base 0 only) rc $? : $(tail -n 1 "$R/$a.log" | cut -c1-300)"
done
echo "ZZ EARLY-RL DONE $(ts)"
