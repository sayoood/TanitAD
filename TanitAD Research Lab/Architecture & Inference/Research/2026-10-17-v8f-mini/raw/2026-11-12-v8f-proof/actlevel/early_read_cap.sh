#!/usr/bin/env bash
# early_read.sh <ARM> : dev box. Card PREREG_V8F_PROOF_ADD37, note N6. Reads ONE arm of the level chains (RLVa, RLVb or RLAh at bases 0 and 1) as soon as both of
# its fine-tunes are done, without waiting for the end of the chains: pod CPU imagination dumps (idempotent, memory-guarded on the pod), pull with sha256, reader
# of card ADD36 (controls first), pairs against RLFT and RLAm (reads of record, actcons/read_rl) and against RLA3 (actcons/read_f), then actlevel/level_names.py
# (the frozen rules of note N6 section 4 and note N5 section 3). The same files are written again, unchanged, by wait_actlevel.sh at the end of the chains.
# It starts no GPU job and deletes nothing. One Python job at a time, each only with 2.5 GB of commit memory free. Last line: ZZ EARLY-READ <ARM> DONE | STOP.
ARM=${1:?usage: early_read.sh ARM}
case "$ARM" in RLVa) SPEC=RLVa:level:0.001 ;; RLVb) SPEC=RLVb:level:0.01 ;; RLAh) SPEC=RLAh:dose:0.0003 ;; *) echo "not an arm of note N6: $ARM"; exit 2 ;; esac
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
SEEDS=0,1; S1=0; S2=1
D=$H/imaghead/dump; R=$H/actlevel/read; mkdir -p "$D" "$R"
freemb() { powershell.exe -NoProfile -Command "\$o = Get-CimInstance Win32_OperatingSystem; [int](\$o.FreeVirtualMemory / 1024)" 2>/dev/null | tr -d '\r'; }
gate() { local n=0 f; while true; do f=$(freemb); [ "${f:-0}" -ge 2560 ] && return 0; n=$((n+1)); [ "$n" -gt 30 ] && { echo "ZZ EARLY-READ $ARM STOP: under 2.5 GB commit free for 30 min"; exit 8; }; sleep 60; done; }
dump() { # chain arm... : pod CPU dumps (idempotent), then pull with sha256 from the SHA lines
  local q=$1 out try; shift
  for try in 1 2 3 4 5 6; do
    out=$(timeout 1500 bash "$P" "bash -s -- $q tree_leadsup $*" < $H/imaghead/ops/pod_imag_dump_arm.sh 2>/dev/null | sed '/hf_/d' | tr -d '\r')
    echo "$out" | cut -c1-330 >> "$R/pod_dump.out"
    echo "$out" | grep -q "DUMP-NOT-RUN" || break
    echo "[$(ts)] pod memory low, dump of $* later"; sleep 600
  done
  echo "$out" | grep "c2b\|^   c1\|no checkpoint" | cut -c1-300
  echo "$out" | grep "^   SHA " | while read -r _ a s1 s2; do
    for pr in "DUMP_$a.npz:$s1" "DUMP_$a.npz.json:$s2"; do f=${pr%%:*}; s=${pr##*:}
      [ -f "$D/$f" ] && [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && continue
      timeout 300 bash $C /workspace/v8proof/runs/imag_z/dump/$f "$D/$f" < /dev/null > /dev/null 2>&1; [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && echo "[$(ts)] pulled $f" || echo "[$(ts)] ZZ SHA MISMATCH $f"
    done
  done
}
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
RW=$W/actlevel/read; DW=$W/imaghead/dump; REC=$W/actcons/read_rl; RF=$W/actcons/read_f
dump actlevel_0 ${ARM}_s0
dump actlevel_1 ${ARM}_s1
[ -f "$D/DUMP_${ARM}_s0.npz" ] && [ -f "$D/DUMP_${ARM}_s1.npz" ] || { echo "ZZ EARLY-READ $ARM STOP: the dumps of both bases are not present ($(ts))"; exit 0; }
gate; ( cd $H/imaghead && "$PY" imag_read_r2.py read --dump "$DW/DUMP_${ARM}_s0.npz" "$DW/DUMP_${ARM}_s1.npz" --arm "$ARM" --seeds $SEEDS --out-dir "$RW/$ARM" > "$R/$ARM.log" 2>&1 ); echo "[$(ts)] read $ARM rc $? : $(tail -n 1 "$R/$ARM.log" | cut -c1-260)"
for pr in "RLFT:$REC/RLFT" "RLAm:$REC/RLAm" "RLA3:$RF/RLA3"; do
  b=${pr%%:*}; bd=${pr#*:}
  [ -f "$R/$ARM/RESULT_IMAG_$ARM.json" ] && [ -f "$bd/RESULT_IMAG_$b.json" ] || { echo "[$(ts)] pair $ARM minus $b: a read is missing"; continue; }
  gate; ( cd $H/imaghead && "$PY" imag_read_r2.py pair --a-dir "$RW/$ARM" --b-dir "$bd" --a-arm "$ARM" --b-arm "$b" --out-dir "$RW/PAIR_${ARM}_$b" > "$R/PAIR_${ARM}_$b.log" 2>&1 ); echo "[$(ts)] pair $ARM minus $b rc $? : $(tail -n 1 "$R/PAIR_${ARM}_$b.log" | cut -c1-300)"
done
gate; "$PY" actlevel/level_names.py --read-dir "$RW" --rlft-dir "$REC/RLFT" --rla3-dir "$RF/RLA3" --arms $SPEC --out "$RW/NAMES_$ARM.json" 2>&1 | cut -c1-420
echo "ZZ EARLY-READ $ARM DONE $(ts)"
