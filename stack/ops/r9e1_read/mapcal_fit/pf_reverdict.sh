#!/usr/bin/env bash
# pf_reverdict.sh <ARM> <REF> -- MM 2026-11-06 (Thor, CPU only). Re-run ONLY the verdict of a pswpedgeF read with the ac90ce1e kit (verdict reports the
# edge F@0.3 m BESIDE rows, PREREG_M1_PEDGEFIT's rows 2/3 selector) on that read's OWN evals / floor / rasters, into <XD>/reverdict_ac90ce1e/.
# Waits for pedgefit_run.sh's 'ARM pswpedgeF <ARM>' line. CHECK: verdict, failing clauses, reads and clause 3 must EQUAL the ff0582d7 verdict's,
# except clause 3's INFORMATIVE list, which by design no longer carries the nf_edge03_* beside rows (v2, 22:30Z: v1 compared clause 3 whole and
# aborted M1 with DIFFER on exactly that; its ABORT line then also stopped the M2a waiter).
export PATH=/usr/bin:/bin:$PATH; set -u
ARM="$1"; REF="$2"
R=/home/nvidia/r8ns/r9e1; RD=$R/read_v3; F=$R/mapcal_fit; K=$R/kit_pedgefit2_ac90ce1e; PY=/home/nvidia/venvs/tanitad-train/bin/python
LV=$RD/LIFT_VALID_M1.npz; FD=$F/floor_thr_pswpedgeF; AD=$RD/arm_$ARM; VS=$K/stack/scripts/r9_e1_verdict.py
PPC="$K/stack:$K/stack/scripts:$K/taniteval:$K/stack/ops/r9e1_read/shim_scipy"
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$F/PEDGEFIT_STATUS.txt"; echo "ZZREVERDICT-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "REVERDICT-ABORT-$1" "$ARM ${*:2}"; exit 3; }
t0=$(date +%s)
until grep -q "ARM pswpedgeF $ARM:" $F/PEDGEFIT_STATUS.txt 2>/dev/null; do
  grep -qE '^[^ ]+ ABORT-' $F/PEDGEFIT_STATUS.txt && die RUN-ABORTED "(read $F/PEDGEFIT_STATUS.txt)"      # pedgefit_run's own ABORT lines only
  [ $(( $(date +%s) - t0 )) -gt 172800 ] && die WAIT "no 'ARM pswpedgeF $ARM' in 48 h"; sleep 60; done
XN=$(grep "ARM pswpedgeF $ARM:" $F/PEDGEFIT_STATUS.txt | tail -1 | grep -oE 'thr_[0-9a-f]{8}_[0-9a-f]{8}' | tail -1)
XD=$AD/$XN; [ -n "$XN" ] && [ -s "$XD/VERDICT_$ARM.json" ] || die NO-XD "$ARM: no verdict dir from the ARM line ($XN)"
DR=$RD/dump_$REF; [ "$REF" = C0 ] || DR=$RD/arm_$REF/dump_$REF
OD=$XD/reverdict_ac90ce1e; mkdir -p $OD
( cd "$AD" && env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN PYTHONPATH="$PPC" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= \
    $PY -X utf8 "$VS" --floor "$FD/FLOOR.json" --floor-eval-c0 "$FD/EVAL_C0.json" --paired "$AD/EVAL_${ARM}_vs_${REF}.json" \
    --eval-arm "$XD/EVAL_$ARM.json" --eval-ref "$XD/EVAL_$REF.json" --arm "$ARM" --ref "$REF" --rast-arm "$AD/dump_$ARM/rast" --rast-ref "$DR/rast" \
    --lift-valid "$LV" --map-thresholds-arm "$F/thr_${ARM}_pswpedgeF.json" --map-thresholds-ref "$F/thr_${REF}_pswpedgeF.json" \
    --boot 2000 --seed 0 --out-dir "$OD" > "$OD/verdict.out" 2> "$OD/verdict.err" ) || die VERDICT-RC "(read $OD/verdict.err)"
CMP=$($PY -I -c 'import json,sys
a=json.load(open(sys.argv[1],encoding="utf-8")); b=json.load(open(sys.argv[2],encoding="utf-8"))
fc=lambda d: sorted(int(c.get("clause", 0)) for c in d.get("failing_clauses", []))
c3a = dict(a["clause3"]); c3b = dict(b["clause3"])
iwa = [r for r in c3a.pop("informative_worse", []) if not str(r.get("metric", "")).startswith("nf_edge03_")]
iwb = c3b.pop("informative_worse", [])
same = a.get("verdict")==b.get("verdict") and fc(a)==fc(b) and a["reads"]==b["reads"] and c3a==c3b and iwa==iwb
bp=b.get("beside_pedgefit") or {}
sel=[r for r in bp.get("rows", []) if r.get("key")=="nf_edge03_f_5_20"]
s=sel[0] if sel else {}
print(("SAME" if same else "DIFFER"), a.get("verdict"), fc(a), "| f_5_20:", s.get("state"), s.get("ref_point"), s.get("arm_point"), s.get("delta"), s.get("ci_lo"), s.get("ci_hi"), s.get("separated"), s.get("reading"))' "$XD/VERDICT_$ARM.json" "$OD/VERDICT_$ARM.json")
[ "${CMP%% *}" = SAME ] || die DIFFER "$CMP"
ev REVERDICT-DONE "$ARM vs $REF $OD: $CMP"
