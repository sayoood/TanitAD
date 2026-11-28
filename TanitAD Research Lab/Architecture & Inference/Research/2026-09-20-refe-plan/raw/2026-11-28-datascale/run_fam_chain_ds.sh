#!/bin/bash
# DATA-SCALE (PREREG_DATASCALE.md 5 M1-M4): the four families on two seed arms vs C1's pick. MM 2026-11-28: the PROVEN EP-HEAD chain
# (PK/raw/2026-11-27-l2next/run1/scripts/run_fam_chain_eph.sh, ran end-to-end 2026-11-28, every step rc 0) with the arm labels as arguments, nothing else.
# Use this, never PK/raw/2026-11-27-safehead/run_fam_chain.sh (its tactical step uses arm labels where p5c_tactical.py needs the ROLE names).
# The tip export must hold: stack taniteval(-results) PK/eval PK/refe PK/raw/2026-11-12-instruments/fam_ci (families_run + p5c_tactical resolve against --repo).
# CPU only (navsim-crun venv, <= 2 processes). Markers <read dir>/fam/fam.DONE | fam.FAILED -- never a sentinel inside a log.
# UNRUN in this parameterised form (MM 2026-11-28; only the labels became arguments, and the expansion was checked): the read's SMOKE=50 step is its first run.
# usage: SMOKE=50 bash run_fam_chain_ds.sh <read out dir> <tip root> <A0> <A1>    then    bash run_fam_chain_ds.sh <read out dir> <tip root> <A0> <A1>
#        seams expected: <read out dir>/seams/seam_<A0>.npz, seam_<A1>.npz, seam_C1pick.npz
RD="$1"; TIP="$2"; A0="${3:?arm label 0}"; A1="${4:?arm label 1}"
[ -d "$RD/seams" ] || { echo "no $RD/seams (run safehead_read.py read first)"; exit 2; }
EV="$TIP/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval"
PYT=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PYN=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
SD="$RD/seams"
F="$RD/fam${SMOKE:+_smoke$SMOKE}"; mkdir -p "$F/strategic"
SUB=(); [ -n "$SMOKE" ] && SUB=(--subset "$SMOKE")
step() { echo "ZZFAM_STEP $1 start $(date -u +%H:%MZ)"; local nm=$1; shift; "$@" > "$F/log_$nm.txt" 2>&1; local rc=$?; echo "ZZFAM_STEP $nm rc=$rc $(date -u +%H:%MZ)"; return $rc; }
fail() { echo "$1" > "$F/fam.FAILED"; echo "ZZFAM_FAIL $1"; exit 1; }
step convert "$PYT" "$EV/p5c/p5c_read_prep.py" convert --arm "${A0}=$SD/seam_${A0}.npz" "${A1}=$SD/seam_${A1}.npz" "C1pick=$SD/seam_C1pick.npz" "${SUB[@]}" --outdir "$F/plans" || fail convert
step score "$PYT" "$EV/p5c/p5c_read_prep.py" score --plans-dir "$F/plans" --out-dir "$F/pass" --procs 2 --python "$PYN" --no-human || fail score
step families "$PYT" "$EV/p5c/p5c_families.py" --rows "$F/pass/rows_s*.jsonl" --arms ${A0} ${A1} C1pick --pairs ${A0}:C1pick ${A1}:C1pick --out-dir "$F/families" --python "$PYN" --repo "$TIP" || fail families
# MM 2026-11-28: p5c_tactical.py is hard-wired to the P5-C ROLE names C1 / C0 / base (it refused "seam for C1 missing"); the proven form
# (scorerft-read sft_read_p.sh:254) maps roles: C1 = treatment, C0 = control, base = a third arm. One run per seed; decision row = C1 vs C0.
mkdir -p "$F/tactical"
step tactical_s0 "$PYT" "$EV/p5c/p5c_tactical.py" --seam "C1=$SD/seam_${A0}.npz" "C0=$SD/seam_C1pick.npz" "base=$SD/seam_${A1}.npz" ${SMOKE:+--subset-ok} --out "$F/tactical/tactical_${A0}_vs_C1pick.json" || fail tactical_s0
step tactical_s1 "$PYT" "$EV/p5c/p5c_tactical.py" --seam "C1=$SD/seam_${A1}.npz" "C0=$SD/seam_C1pick.npz" "base=$SD/seam_${A0}.npz" ${SMOKE:+--subset-ok} --out "$F/tactical/tactical_${A1}_vs_C1pick.json" || fail tactical_s1
step strategic_s0 "$PYT" "$EV/instruments/strategic_family.py" read --arm "${A0}=$SD/seam_${A0}.npz" --base "C1pick=$SD/seam_C1pick.npz" --out "$F/strategic/strategic_${A0}.json" || fail strategic_s0
step strategic_s1 "$PYT" "$EV/instruments/strategic_family.py" read --arm "${A1}=$SD/seam_${A1}.npz" --base "C1pick=$SD/seam_C1pick.npz" --out "$F/strategic/strategic_${A1}.json" || fail strategic_s1
date -u +%FT%H:%MZ > "$F/fam.DONE"
echo "ZZFAM_DONE $F"
