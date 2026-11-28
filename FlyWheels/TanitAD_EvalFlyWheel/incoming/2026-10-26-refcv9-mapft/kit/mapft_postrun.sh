#!/usr/bin/env bash
# mapft_postrun.sh -- THOR-SIDE post-run steps for the MAPFT run (PREREG_MAPFT s2).  CPU ONLY, never the GPU, never the supervisor.
#   Lives at $ROOT/kit/ on Thor (shipped by mapft_ship.sh kit).  $ROOT = /home/nvidia/r8ns/mapft.
#
#   mapft_postrun.sh status     READ-ONLY.  Is the run finished?  final step, files, the supervisor's done-marker incl. its zero-gradient record
#   mapft_postrun.sh kfrozen    K-FROZEN (assert_map_only_finetune.py on ckpt.pt = raw+EMA views, and on model_ema.pt) + "at least one changed
#                               tensor in each of the 3 modules" + two controls  -> $ROOT/post/KFROZEN.json
#   mapft_postrun.sh manifest   sha256 + bytes of everything the dev box needs -> $ROOT/post/MANIFEST.sha256   (run AFTER kfrozen)
#   ANY: --root DIR --code DIR --run DIR [--init warm-start-file]
#
# Verdicts are read from ARTIFACTS (KFROZEN.json, summary.json), never from $? and never through a pipe.  Markers are opaque ZZ...ZZ tokens.
# No `pgrep -f`/`pkill -f`; nothing here kills or starts a process.  The EMA file model_ema.pt is the SCORED weights; ckpt.pt's raw view is read beside.
set -u
MODE="${1:-}"; shift || true
_VENV_PY=/home/nvidia/venvs/tanitad-train/bin/python
PY="${PYTHON:-$( [ -x "$_VENV_PY" ] && echo "$_VENV_PY" || echo python3 )}"
export MSYS_NO_PATHCONV=1
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KITN="$KIT"; command -v cygpath >/dev/null 2>&1 && KITN="$(cygpath -m "$KIT")"      # dev-box dry-runs only: a native path for python (no-op on Thor)
ROOT=/home/nvidia/r8ns/mapft; ARM=refcv8-mapft-s0; CODE=""; RUN=""
INIT=/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt
while [ $# -gt 0 ]; do case "$1" in
  --root) ROOT="$2"; shift 2;; --code) CODE="$2"; shift 2;; --run) RUN="$2"; shift 2;; --init) INIT="$2"; shift 2;;
  *) echo "REFUSED: unknown argument $1"; exit 2;; esac; done
CODE="${CODE:-$ROOT/launch_tree}"; RUN="${RUN:-$ROOT/runs/$ARM}"; POSTD="$ROOT/post"; ARGV="$ROOT/argv/$ARM.argv.json"
say() { echo "[mapft-postrun:$MODE] $*"; }
die() { say "REFUSED: $*"; echo "ZZMAPFTPOSTRUN-REFUSEDZZ"; exit 10; }
pyval() { "$PY" -c "$1" "${@:2}" | tr -d '\r'; }
argv_get() { pyval 'import json,sys; d=json.load(open(sys.argv[1])); a=d["argv"] if isinstance(d,dict) else d; print(a[a.index(sys.argv[2])+1] if sys.argv[2] in a else "")' "$1" "$2"; }
[ -n "$MODE" ] || { sed -n 2,11p "${BASH_SOURCE[0]}"; exit 2; }
[ -f "$ARGV" ] || die "no launch argv at $ARGV"
N="$(argv_get "$ARGV" --steps)"; [ -n "$N" ] || die "the launch argv carries no --steps"

case "$MODE" in
status)
  say "run dir $RUN  (N=$N)"
  LAST="$(pyval 'import json,sys
m=-1
try:
    for l in open(sys.argv[1],encoding="utf-8"):
        try: r=json.loads(l)
        except Exception: continue
        if isinstance(r,dict) and isinstance(r.get("step"),int): m=max(m,r["step"])
except Exception: pass
print(m)' "$RUN/metrics.jsonl")"
  say "last metrics step: $LAST of $N"
  for f in ckpt.pt model_ema.pt metrics.jsonl config.json summary.json; do say "  $f: $(stat -c %s "$RUN/$f" 2>/dev/null || echo MISSING) bytes"; done
  [ -e "$RUN/ZEROGRAD_FAIL.json" ] && say "  !! ZEROGRAD_FAIL.json PRESENT: the supervisor's post-run gate rejected the checkpoint (read it; the run's files are still complete)"
  DONE="$(pyval 'import json,sys
try:
    d=json.load(open(sys.argv[1])); zg=d.get("zerograd_gate")
    print("done=%s step=%s zerograd=%s" % (d.get("done"), d.get("final_step", d.get("step")), (zg or {}).get("status", "NO-RECORD")))
except Exception as e: print("unreadable:", type(e).__name__)' "$RUN/summary.json")"
  say "done-marker: $DONE"
  say "GPU compute apps now: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]') (0 = the run released the GPU)"
  if [ "$LAST" -ge $((N-1)) ] && [ -s "$RUN/model_ema.pt" ] && [ -s "$RUN/ckpt.pt" ] && echo "$DONE" | grep -q 'done=True'; then echo "ZZMAPFT-RUN-COMPLETEZZ"; else echo "ZZMAPFT-RUN-NOT-COMPLETEZZ"; fi;;

kfrozen)
  [ -s "$RUN/ckpt.pt" ] && [ -s "$RUN/model_ema.pt" ] || die "ckpt.pt / model_ema.pt missing under $RUN"
  [ -s "$INIT" ] || die "the warm-start file $INIT is missing"
  mkdir -p "$POSTD" 2>/dev/null
  rm -f "$POSTD/KFROZEN.json"
  # CPU only (the GPU may already belong to the next job); low priority; the tool mmaps the checkpoints
  ( cd "$CODE/stack" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$CODE/stack:$CODE/taniteval" nice -n 10 "$PY" "$KITN/mapft_kfrozen.py" run \
      --tree "$CODE" --src "$INIT" --raw "$RUN/ckpt.pt" --ema "$RUN/model_ema.pt" --out-dir "$POSTD" --expect-step "$N" > "$POSTD/kfrozen.out" 2>&1 )
  # the verdict is the ARTIFACT
  V="$(pyval 'import json,sys
try: print(json.load(open(sys.argv[1])).get("K_FROZEN","NONE"))
except Exception: print("NO-ARTIFACT")' "$POSTD/KFROZEN.json")"
  say "K-FROZEN = $V  (record $POSTD/KFROZEN.json, tool output $POSTD/kfrozen.out)"
  [ "$V" = PASS ] && echo "ZZMAPFT-KFROZEN-PASSZZ" || { echo "ZZMAPFT-KFROZEN-FAILZZ"; exit 11; };;

manifest)
  [ "$(pyval 'import json,sys
try: print(json.load(open(sys.argv[1])).get("K_FROZEN","NONE"))
except Exception: print("NO-ARTIFACT")' "$POSTD/KFROZEN.json")" = PASS ] || die "K-FROZEN is not PASS: nothing is shipped to the dev box on a failed freeze control"
  : > "$POSTD/MANIFEST.sha256"
  for f in model_ema.pt ckpt.pt; do   # 400 MB each; the run is over, the box is ours
    H="$(sha256sum "$RUN/$f" | cut -c1-64)"; [ "${#H}" = 64 ] || die "could not sha256 $RUN/$f"
    echo "$H  $(stat -c %s "$RUN/$f")  runs/$ARM/$f" >> "$POSTD/MANIFEST.sha256"
  done
  for f in metrics.jsonl config.json summary.json train.log; do
    [ -s "$RUN/$f" ] || continue
    H="$(sha256sum "$RUN/$f" | cut -c1-64)"; echo "$H  $(stat -c %s "$RUN/$f")  runs/$ARM/$f" >> "$POSTD/MANIFEST.sha256"
  done
  for f in KFROZEN.json; do H="$(sha256sum "$POSTD/$f" | cut -c1-64)"; echo "$H  $(stat -c %s "$POSTD/$f")  post/$f" >> "$POSTD/MANIFEST.sha256"; done
  H="$(sha256sum "$ARGV" | cut -c1-64)"; echo "$H  $(stat -c %s "$ARGV")  argv/$ARM.argv.json" >> "$POSTD/MANIFEST.sha256"
  for need in runs/$ARM/model_ema.pt runs/$ARM/ckpt.pt runs/$ARM/metrics.jsonl runs/$ARM/summary.json post/KFROZEN.json argv/$ARM.argv.json; do
    [ "$(grep -c "  $need\$" "$POSTD/MANIFEST.sha256")" = 1 ] || die "manifest lacks a row for $need"
  done
  cat "$POSTD/MANIFEST.sha256"; say "manifest -> $POSTD/MANIFEST.sha256"; echo "ZZMAPFT-MANIFEST-OKZZ";;
*) echo "usage: mapft_postrun.sh status|kfrozen|manifest [--root D --code D --run D]"; exit 2;;
esac
