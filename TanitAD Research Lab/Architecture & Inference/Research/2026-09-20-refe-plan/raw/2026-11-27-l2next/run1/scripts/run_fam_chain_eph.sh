#!/bin/bash
# L2-NEXT EP-HEAD: the four families on the EPH picks (PREREG_EPHEAD.md 5 M4). MM 2026-11-28: run_fam_chain.sh of PK/raw/2026-11-27-safehead with the arm labels SH0/SH1 -> EPH0/EPH1 (card 5), nothing else.  8-pose like-for-like form, the pattern of
# PK/raw/2026-11-20-p5c-next/run_srule_families.sh, on ALL navtest tokens (SMOKE=<n>: a subset of n tokens, plumbing only, no number is read).
# CPU only (navsim-crun venv, <= 2 processes).  Writes markers  <read dir>/fam/fam.DONE | fam.FAILED  -- never a sentinel inside a log.
# usage: SMOKE=50 bash run_fam_chain.sh <read out dir> <exported landed tip root>     then      bash run_fam_chain.sh <read out dir> <tip root>
RD="$1"; TIP="$2"
[ -d "$RD/seams" ] || { echo "no $RD/seams (run safehead_read.py read first)"; exit 2; }
EV="$TIP/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval"
PYT=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PYN=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
SD="$RD/seams"
F="$RD/fam${SMOKE:+_smoke$SMOKE}"; mkdir -p "$F/strategic"
SUB=(); [ -n "$SMOKE" ] && SUB=(--subset "$SMOKE")
step() { echo "ZZFAM_STEP $1 start $(date -u +%H:%MZ)"; local nm=$1; shift; "$@" > "$F/log_$nm.txt" 2>&1; local rc=$?; echo "ZZFAM_STEP $nm rc=$rc $(date -u +%H:%MZ)"; return $rc; }
fail() { echo "$1" > "$F/fam.FAILED"; echo "ZZFAM_FAIL $1"; exit 1; }
step convert "$PYT" "$EV/p5c/p5c_read_prep.py" convert --arm "EPH0=$SD/seam_EPH0.npz" "EPH1=$SD/seam_EPH1.npz" "C1pick=$SD/seam_C1pick.npz" "${SUB[@]}" --outdir "$F/plans" || fail convert
step score "$PYT" "$EV/p5c/p5c_read_prep.py" score --plans-dir "$F/plans" --out-dir "$F/pass" --procs 2 --python "$PYN" --no-human || fail score
step families "$PYT" "$EV/p5c/p5c_families.py" --rows "$F/pass/rows_s*.jsonl" --arms EPH0 EPH1 C1pick --pairs EPH0:C1pick EPH1:C1pick --out-dir "$F/families" --python "$PYN" --repo "$TIP" || fail families
# MM 2026-11-28: p5c_tactical.py is hard-wired to the P5-C ROLE names C1 / C0 / base (it refused "seam for C1 missing"); the proven form
# (scorerft-read sft_read_p.sh:254) maps roles: C1 = treatment, C0 = control, base = a third arm. One run per seed; decision row = C1 vs C0.
mkdir -p "$F/tactical"
step tactical_s0 "$PYT" "$EV/p5c/p5c_tactical.py" --seam "C1=$SD/seam_EPH0.npz" "C0=$SD/seam_C1pick.npz" "base=$SD/seam_EPH1.npz" ${SMOKE:+--subset-ok} --out "$F/tactical/tactical_EPH0_vs_C1pick.json" || fail tactical_s0
step tactical_s1 "$PYT" "$EV/p5c/p5c_tactical.py" --seam "C1=$SD/seam_EPH1.npz" "C0=$SD/seam_C1pick.npz" "base=$SD/seam_EPH0.npz" ${SMOKE:+--subset-ok} --out "$F/tactical/tactical_EPH1_vs_C1pick.json" || fail tactical_s1
step strategic_s0 "$PYT" "$EV/instruments/strategic_family.py" read --arm "EPH0=$SD/seam_EPH0.npz" --base "C1pick=$SD/seam_C1pick.npz" --out "$F/strategic/strategic_EPH0.json" || fail strategic_s0
step strategic_s1 "$PYT" "$EV/instruments/strategic_family.py" read --arm "EPH1=$SD/seam_EPH1.npz" --base "C1pick=$SD/seam_C1pick.npz" --out "$F/strategic/strategic_EPH1.json" || fail strategic_s1
date -u +%FT%H:%MZ > "$F/fam.DONE"
echo "ZZFAM_DONE $F"
