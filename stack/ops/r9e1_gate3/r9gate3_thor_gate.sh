#!/usr/bin/env bash
# r9gate3_thor_gate.sh -- THOR side: one launch-gate run of the study's profile (R9_STUDY: e1 = refcv9-e1rig, the default; e2 = refcv9-e2) for ONE arm.
#   The study (root, argv prefix, profile) comes from r9gate3_study.sh NEXT TO THIS SCRIPT -- never from a literal here.
#   r9gate3_thor_gate.sh explore <ARM>            non-token exploration of the GPU/data checks (G-HYG,G-DVB,G-LIVE,G-CKPT,G-EVAL,G-CLOCK) -> $R/gate3_explore/thor_<ARM>
#   r9gate3_thor_gate.sh final   <ARM> <SUITE_BOUND_JSON> <DEVBOX_EVIDENCE_DIR>   the real --stage thor run -> $R/gate3/<ARM>/PASS_<c12>.json
# Every verdict is read from an ARTIFACT (token / evidence JSON), never from $?.  The Thor GPU lock is held for the whole gate run
# (G-LIVE / G-CKPT / G-EVAL use the GPU); the GPU must have NO compute app under the lock.   Markers: ZZR9E1G3-<ARM>-<MODE>-{PASS,NOTPASS,REFUSED}ZZ
set -u
MODE="${1:-}"; ARM="${2:-}"; SUITE="${3:-}"; DEVEV="${4:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/r9gate3_study.sh" || { echo "ZZR9E1G3-${ARM}-${MODE}-REFUSEDZZ"; exit 10; }    # R9_ROOT R9_PREFIX R9_PROFILE ... (one definition; refuses an unknown study)
R=$R9_ROOT; T=$R/launch_tree; PY=/home/nvidia/venvs/tanitad-train/bin/python; LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
SHA="$(cat $R/launch_tree.commit)"; C12="${SHA:0:12}"; ARGV=$T/stack/ops/runs.d/${R9_PREFIX}_${ARM}.argv.json
EVO=$T/stack/ops/refcv8_eval_remap_a2_thor.json
REC12="$(cat $R/gate3/REC_C12 2>/dev/null || echo $R9_DEFAULT_REC12)"; OV=$R/ovf_${ARM}_$REC12
say() { echo "[r9e1-gate3:$MODE:$ARM] $(date -u +%H:%M:%SZ) $*"; }
refuse() { say "REFUSED: $*"; echo "ZZR9E1G3-${ARM}-${MODE}-REFUSEDZZ"; exit 10; }
case "$ARM" in ""|*/*|*" "*) refuse "bad arm '$ARM'";; esac   # any arm with a committed argv (checked below)
[ "${#SHA}" = 40 ] || refuse "launch_tree.commit is not a 40-char sha"
R9_PROFILE="$(r9_profile_of_arm "$ARM")"    # e3: a `<V>_d1` arm is gated under the D1 profile; every other arm (and all of e1 / e2) under the study's profile
[ -n "$REC12" ] || refuse "no $R/gate3/REC_C12 stamp and study $R9_STUDY has no default record commit (r9gate3_restart.sh writes it from its third argument)"
[ -f "$ARGV" ] || refuse "no argv $ARGV"; [ -f "$EVO" ] || refuse "no remap override file $EVO"
# >>> E1 record kind (byte-identical in r9gate3_thor_gate.sh and r9gate3_rec_explore.sh; tests/test_r9gate3_record_kind.py runs this block)
# The record KIND follows the arm's argv (E1_OWN_RECORDS), the way r9e1_bind.sh binds it: an argv carrying --r9-map-banded binds its /2-banded map
# record; one carrying an OWN-HEAD --r9-box-* lever (the GATE'S OWN E1_BOX_OWN_FLAGS, no second list) binds its /2-b3 box record. MM 2026-11-05: the
# box half was missing, so B3's bound record ($OV/run_box-b3) was unreachable from this caller and B3 was dropped from the 384e79f restart.
# A classification that cannot run REFUSES: a silent fallback to the inherited box record would be refused later by the gate, under another name.
MREC=$OV/run_map/g_map_overfit.json; MCLO=$OV/map_closure.json; BREC=$OV/run_box/gbo_binding.json; BCLO=$OV/box_closure.json
grep -q '"--r9-map-banded"' "$ARGV" 2>/dev/null && { MREC=$OV/run_map-banded/g_map_overfit.json; MCLO=$OV/map-banded_closure.json; }
_bown="$(PYTHONPATH="$T/stack/scripts" "$PY" -c 'import sys, launch_gate as L; print(" ".join(sorted({f for f, _ in L.flag_pairs(L.load_argv_file(sys.argv[1]))} & set(L.E1_BOX_OWN_FLAGS))))' "$ARGV" 2>/dev/null)" \
  || refuse "could not classify the --r9-box-* flags of $ARGV with the gate's E1_BOX_OWN_FLAGS"
[ -n "$_bown" ] && { BREC=$OV/run_box-b3/gbo_binding.json; BCLO=$OV/box-b3_closure.json; }
# <<< E1 record kind
# >>> E1-REUSE record pick (byte-identical in r9gate3_thor_gate.sh and r9gate3_rec_explore.sh; tests/test_launch_gate_e1rig_reuse.py runs this block)
# An arm bound to a BASE arm's PASSING record (r9e1_bind.sh --reuse-from) has $OV/run_<kind>/reuse_binding.json INSTEAD of a run record; the gate's G-MAP-OVERFIT /
# G-BOX-OVERFIT judges read it through the same --map-overfit-record / --box-overfit-record option. Exactly ONE must exist per kind: both = ambiguous (no silent preference).
for _k in map box; do
  if [ "$_k" = map ]; then _run="$MREC"; else _run="$BREC"; fi; _ru="$OV/run_$_k/reuse_binding.json"
  if [ -e "$_ru" ] && [ -e "$_run" ]; then refuse "both a run record ($_run) and a reuse binding ($_ru) exist for the $_k kind: ambiguous, remove the stale one"; fi
  if [ -e "$_ru" ]; then if [ "$_k" = map ]; then MREC="$_ru"; else BREC="$_ru"; fi
  elif [ ! -e "$_run" ]; then refuse "neither a run record ($_run) nor a reuse binding ($_ru) exists for the $_k kind"; fi
done
# <<< E1-REUSE record pick
for f in $MREC $MCLO $BREC $BCLO; do [ -s "$f" ] || refuse "missing overfit record/closure $f"; done
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
A=(run --profile "$R9_PROFILE" --tree "$T" --commit "$SHA" --argv-file "$ARGV" --eval-kit /home/nvidia --eval-remap-overrides "$EVO" --omp 8
   --map-overfit-record $MREC --map-overfit-closure $MCLO
   --box-overfit-record $BREC --box-overfit-closure $BCLO)
# SPEC 6.2 item 5: a measured s/step above refcv6's +25 % is PI-DECISION unless the PI's RECORDED approval covers it. An arm's approval is committed
# beside its argv (runs.d/<prefix>_<ARM>.pi_cost_approval.json: {max_s_per_step, pi_verbatim, date}) and handed to the gate in BOTH modes (the
# gate binds its sha256 into the G-LIVE evidence). MM 2026-11-05: K0's explore read 9.40 s/step (1.47x) and the e2 chain STOPPED EXPLORE-NOT-PASS
# because this script had no way to pass one. No file = no approval: the cost rule applies unchanged.
PICOST=$T/stack/ops/runs.d/${R9_PREFIX}_${ARM}.pi_cost_approval.json
[ -f "$PICOST" ] && A+=(--pi-cost-approval "$PICOST")
case "$MODE" in
explore) OUT=$R/gate3_explore/${C12}_$ARM; A+=(--checks G-HYG,G-DVB,G-LIVE,G-CKPT,G-EVAL,G-CLOCK --out-dir "$OUT");;
final)   OUT=$R/gate3/$ARM; [ -s "$SUITE" ] || refuse "no suite-bound verdict '$SUITE'"; [ -s "$DEVEV/G-SUITE-PINNED.json" ] || [ -s "$DEVEV/evidence/G-SUITE-PINNED.json" ] || refuse "no G-SUITE-PINNED.json under '$DEVEV'"
         [ -d "$DEVEV/evidence" ] && DEVEV="$DEVEV/evidence"
         A+=(--stage thor --suite-verdict "$SUITE" --import-evidence "$DEVEV" --out-dir "$OUT");;
*) refuse "mode must be explore|final";; esac
mkdir -p "$OUT"; rm -rf "$OUT"/evidence "$OUT"/logs 2>/dev/null
exec 9>"$LOCKF"; say "waiting for the Thor GPU lock (<= ${R9_LOCK_WAIT_S} s)"; flock -w "$R9_LOCK_WAIT_S" 9 || refuse "the Thor GPU lock stayed held for ${R9_LOCK_WAIT_S} s"
NA="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"; [ "$NA" = 0 ] || refuse "the GPU is not empty under the lock ($NA compute app(s))"
say "gate command: $PY $T/stack/scripts/launch_gate.py ${A[*]}"; t0=$(date +%s)
"$PY" "$T/stack/scripts/launch_gate.py" "${A[@]}" > "$OUT.gate.log" 2>&1 9>&- 200>&-
say "gate process ended after $(( $(date +%s) - t0 )) s"
TOK="$OUT/PASS_$C12.json"
V="$($PY -c 'import json,glob,sys; fs=sorted(glob.glob(sys.argv[1]+"/*_"+sys.argv[2]+".json")); d=json.load(open(fs[-1])) if fs else {}; print(d.get("verdict","NO-TOKEN"))' "$OUT" "$C12" 9>&-)"
$PY - "$OUT" <<'PYEOF' 9>&- | tee "$OUT.timings.txt"
import json, glob, sys, datetime as dt
def p(s): return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
for f in sorted(glob.glob(sys.argv[1] + "/evidence/G-*.json")):
    d = json.load(open(f, encoding="utf-8"))
    try: w = int((p(d["finished_utc"]) - p(d["started_utc"])).total_seconds())
    except Exception: w = None
    print("CHECK %-16s %-8s wall_s=%s" % (d.get("check"), d.get("status"), w))
    if d.get("status") != "PASS":
        for r in (d.get("reasons") or [])[:6]: print("   reason:", str(r)[:400])
PYEOF
if [ "$MODE" = final ]; then
  if [ "$V" = PASS ] && [ -s "$TOK" ]; then say "token $TOK is PASS"; echo "ZZR9E1G3-${ARM}-final-PASSZZ"; exit 0; fi
  say "token verdict is '$V', not PASS"; echo "ZZR9E1G3-${ARM}-final-NOTPASSZZ"; exit 1
fi
say "explore verdict (INCOMPLETE expected: no suite): $V"; echo "ZZR9E1G3-${ARM}-explore-DONEZZ"
