#!/usr/bin/env bash
# r9gate3_thor_gate.sh -- THOR side: one launch-gate run of profile refcv9-e1rig for ONE arm.
#   r9gate3_thor_gate.sh explore <ARM>            non-token exploration of the GPU/data checks (G-HYG,G-DVB,G-LIVE,G-CKPT,G-EVAL,G-CLOCK) -> $R/gate3_explore/thor_<ARM>
#   r9gate3_thor_gate.sh final   <ARM> <SUITE_BOUND_JSON> <DEVBOX_EVIDENCE_DIR>   the real --stage thor run -> $R/gate3/<ARM>/PASS_<c12>.json
# Every verdict is read from an ARTIFACT (token / evidence JSON), never from $?.  The Thor GPU lock is held for the whole gate run
# (G-LIVE / G-CKPT / G-EVAL use the GPU); the GPU must have NO compute app under the lock.   Markers: ZZR9E1G3-<ARM>-<MODE>-{PASS,NOTPASS,REFUSED}ZZ
set -u
MODE="${1:-}"; ARM="${2:-}"; SUITE="${3:-}"; DEVEV="${4:-}"
R=/home/nvidia/r8ns/r9e1; T=$R/launch_tree; PY=/home/nvidia/venvs/tanitad-train/bin/python; LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
SHA="$(cat $R/launch_tree.commit)"; C12="${SHA:0:12}"; ARGV=$T/stack/ops/runs.d/r9e1_${ARM}.argv.json
EVO=$T/stack/ops/refcv8_eval_remap_a2_thor.json
OV=$R/ovf_${ARM}_d297ba563d8f
say() { echo "[r9e1-gate3:$MODE:$ARM] $(date -u +%H:%M:%SZ) $*"; }
refuse() { say "REFUSED: $*"; echo "ZZR9E1G3-${ARM}-${MODE}-REFUSEDZZ"; exit 10; }
case "$ARM" in C0|C1) ;; *) refuse "arm must be C0 or C1";; esac
[ "${#SHA}" = 40 ] || refuse "launch_tree.commit is not a 40-char sha"
[ -f "$ARGV" ] || refuse "no argv $ARGV"; [ -f "$EVO" ] || refuse "no remap override file $EVO"
for f in $OV/run_map/g_map_overfit.json $OV/map_closure.json $OV/run_box/gbo_binding.json $OV/box_closure.json; do [ -s "$f" ] || refuse "missing overfit record/closure $f"; done
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
A=(run --profile refcv9-e1rig --tree "$T" --commit "$SHA" --argv-file "$ARGV" --eval-kit /home/nvidia --eval-remap-overrides "$EVO" --omp 8
   --map-overfit-record $OV/run_map/g_map_overfit.json --map-overfit-closure $OV/map_closure.json
   --box-overfit-record $OV/run_box/gbo_binding.json --box-overfit-closure $OV/box_closure.json)
case "$MODE" in
explore) OUT=$R/gate3_explore/${C12}_$ARM; A+=(--checks G-HYG,G-DVB,G-LIVE,G-CKPT,G-EVAL,G-CLOCK --out-dir "$OUT");;
final)   OUT=$R/gate3/$ARM; [ -s "$SUITE" ] || refuse "no suite-bound verdict '$SUITE'"; [ -s "$DEVEV/G-SUITE-PINNED.json" ] || [ -s "$DEVEV/evidence/G-SUITE-PINNED.json" ] || refuse "no G-SUITE-PINNED.json under '$DEVEV'"
         [ -d "$DEVEV/evidence" ] && DEVEV="$DEVEV/evidence"
         A+=(--stage thor --suite-verdict "$SUITE" --import-evidence "$DEVEV" --out-dir "$OUT");;
*) refuse "mode must be explore|final";; esac
mkdir -p "$OUT"; rm -rf "$OUT"/evidence "$OUT"/logs 2>/dev/null
exec 9>"$LOCKF"; say "waiting for the Thor GPU lock (<= 3 h)"; flock -w 10800 9 || refuse "the Thor GPU lock stayed held for 3 h"
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
