#!/usr/bin/env bash
# SPEC_WPB_LADDER_A4 -- the post-chain step on Thor (NOT part of ladder_chain.sh; it never writes into $W or the chain's
# code dir). It WAITS until the chain has logged its final all-done marker AFTER its last start, then:
#   1. runs `ladder_eval_xshuf.py` for arm V-R8, row vmax_xshuf, sampler seeds 0,1 -- under the SAME programme GPU lock,
#      MemAvailable guard, timeout, env and flags as the chain's `eval_arm` (the one forced difference: `--ckpt`, because
#      the chain stripped ckpt.pt to model_final.pt = the same `model` tensors + step, right after its own V-R8 eval);
#   2. scores with `ladder_score_a4.py` into $A4/LADDER_SCORE_A4.json (the chain's LADDER_SCORE.json is never written,
#      and its md5 is asserted unchanged);
#   3. asserts on ARTIFACTS, never on exit codes, and writes ONE status token to $A4/A4_STATUS (read it with `cat`;
#      do not grep a stream for the words your own command contains).
# No GPU before the all-done marker. Every child is launched with the single-instance lock fd closed (200>&-).
# Usage (Thor, after the MM's go):  nohup bash post_chain_a4.sh > $A4/post_chain.out 2>&1 < /dev/null &
set -u
R=${R:-/home/nvidia/refcv8_ladder}
W=${W:-$R/W}
TREE=${TREE:-$R/tree}
A4=${A4:-$R/mm_a4}
CODE_A4=${CODE_A4:-$A4/code}
PY=${PY:-/home/nvidia/venvs/tanitad-train/bin/python}
LOCK=${LOCK:-/home/nvidia/refcv7_post/thor_gpu.lock}
CAP_S=${CAP_S:-3600}
MEM_MIN_GB=${MEM_MIN_GB:-40}
V9TRAIN=${V9TRAIN:-/home/nvidia/refcv8_v9labels/release/v9_labels_train.npz}
V9TRAIN_MD5=${V9TRAIN_MD5:-f63ece410b725febb8a5242cf2b01d3c}
EXPECT_TREE=${EXPECT_TREE:-/home/nvidia/refcv6_run/1df35031db}   # the tree the chain evaluated V-R8 with (MEASURED)
# the donor map xshuf_plan derives from the chain's own V-R8 base grid (MEASURED 2026-10-12 off base_s0/s1.npz input
# arrays, mm_a4/probe2.json); the eval must reproduce it exactly -- a must-read-known-value check on the instrument
EXPECT_DONOR_MAP=${EXPECT_DONOR_MAP:-7e3b86b8179e7fed93ae3459586602a472f6a702997d1ec3d4a3f5925cf5c01f}
POLL_S=${POLL_S:-300}
MAX_WAIT_S=${MAX_WAIT_S:-345600}                                  # 96 h
DRY=${DRY:-0}                                                     # 1 = no GPU job, no scorer (logic test only)
PSEP=${PSEP:-:}
ARM=V-R8
ROW=vmax_xshuf
EV="$A4/eval/$ARM"
mkdir -p "$A4" "$EV"
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$A4/post_chain.log"; }
status() { echo "$1" > "$A4/A4_STATUS"; log "A4M[$1]A4M ${2:-}"; }
stop() { status "$1" "$2"; exit 3; }

# single instance (fd 200, as in the chain; closed on every child below with 200>&-, so no child can inherit the lock -- the CLAUDE.md lock-fd trap)
exec 200>"$A4/post_chain.lock"
flock -n 200 || { echo "another post_chain_a4 holds $A4/post_chain.lock -- exiting"; exit 4; }
status WAITING "W=$W A4=$A4 pid=$$"

# ------------------------------------------------------------------------------------------- wait for the chain
# The marker strings are ASSEMBLED, so no line of this file contains them whole; awk emits an opaque token that is
# parsed here. An unreadable log parses as nothing and counts as NOT done.
M_ALL="ZZ""ALLDONE""ZZ"; M_START="ZZ""START""ZZ"; M_STOP="ZZ""STOP""ZZ"
t_wait0=$(date +%s); last=""
while :; do
  tok=$(awk -v a="$M_ALL" -v s="$M_START" -v p="$M_STOP" 'index($0, s) {ls = NR} index($0, a) {la = NR}
        index($0, p) {lp = NR} END {printf "A4W-%d-%d-%d-A4W", ls, la, lp}' "$W/chain.log" 2>/dev/null 200>&-)
  if [[ "$tok" =~ ^A4W-([0-9]+)-([0-9]+)-([0-9]+)-A4W$ ]]; then
    ls_=${BASH_REMATCH[1]}; la_=${BASH_REMATCH[2]}; lp_=${BASH_REMATCH[3]}
    if [ "$la_" -gt 0 ] && [ "$la_" -gt "$ls_" ]; then log "chain all-done at line $la_ (last start $ls_)"; break; fi
    now="start=$ls_ alldone=$la_ stop=$lp_"
    if [ "$lp_" -gt "$ls_" ] && [ "$lp_" -gt 0 ]; then now="$now (the chain STOPPED after its last start; waiting for a relaunch)"; fi
  else
    now="chain.log unreadable"
  fi
  [ "$now" != "$last" ] && { log "waiting: $now"; last=$now; }
  [ $(( $(date +%s) - t_wait0 )) -gt "$MAX_WAIT_S" ] && stop TIMEOUT-WAITING "no all-done after ${MAX_WAIT_S}s ($now)"
  sleep "$POLL_S" 200>&-
done

# ------------------------------------------------------------------------------------------- preflight (CPU only)
status PREFLIGHT
cd "$A4" || stop BLOCKED-NO-A4DIR "$A4"
[ -e "$A4/taniteval" ] && stop BLOCKED-SHADOW "$A4/taniteval would shadow the taniteval namespace package"
real_tree=$(readlink -f "$TREE" 200>&-)
[ "$real_tree" = "$EXPECT_TREE" ] || stop BLOCKED-TREE-MOVED "tree -> $real_tree, V-R8 base was evaluated on $EXPECT_TREE (pairing needs the same code)"
( cd "$CODE_A4" && md5sum -c --quiet A4_MD5SUMS 200>&- ) >> "$A4/post_chain.log" 2>&1 || stop BLOCKED-CODE-MD5 "$CODE_A4/A4_MD5SUMS does not verify"
export PYTHONPATH="$TREE/stack$PSEP$TREE/taniteval" REFCV6_REPO="$TREE" REFCV6_KIT=/home/nvidia OMP_NUM_THREADS=6
got=$(CUDA_VISIBLE_DEVICES= "$PY" -c 'import sys, tanitad, taniteval.ci; from pathlib import Path as P
t, f = P(sys.argv[1]).resolve(), P(tanitad.__file__).resolve()
print(f.as_posix()); sys.exit(0 if t in f.parents else 7)' "$TREE" 2>&1 200>&-) || stop BLOCKED-IMPORT "tanitad not from $TREE: $got"
N=$(tr -dc 0-9 < "$W/N.txt" 2>/dev/null)
[ -n "$N" ] || stop BLOCKED-NO-N "$W/N.txt"
RUN="$W/arms/$ARM/run"
for s in 0 1; do
  [ -f "$W/eval/$ARM/base_s$s.json" ] && [ -f "$W/eval/$ARM/base_s$s.npz" ] || stop BLOCKED-NO-BASE "$W/eval/$ARM/base_s$s.*"
done
[ -f "$RUN/config.json" ] || stop BLOCKED-NO-CONFIG "$RUN/config.json"
want=$(tr -dc 0-9a-f < "$RUN/model_final.pt.md5" 2>/dev/null)
[ ${#want} -eq 32 ] || stop BLOCKED-NO-MD5 "$RUN/model_final.pt.md5"
CKPT=""
for c in "$RUN/model_final.pt" "$A4/ckpt/$ARM/model_final.pt"; do
  [ -f "$c" ] || continue
  h=$(md5sum "$c" 200>&- | cut -c1-32)
  if [ ${#h} -eq 32 ] && [ "$h" = "$want" ]; then CKPT=$c; break; fi
  log "checkpoint $c md5 ${h:-?} != $want -- not used"
done
[ -n "$CKPT" ] || stop BLOCKED-NO-CKPT "no md5-verified model_final.pt (the dev-box puller deletes Thor's copy): ship D:/refcv8_ladder_pull/$ARM/run/model_final.pt to $A4/ckpt/$ARM/ and re-run"
score_md5_before=$(md5sum "$W/LADDER_SCORE.json" 2>/dev/null 200>&- | cut -c1-32)
log "preflight OK: tree=$real_tree tanitad=$got N=$N ckpt=$CKPT md5=$want chain LADDER_SCORE.json md5=${score_md5_before:-absent}"

# ------------------------------------------------------------------------------------------- the GPU job (chain form)
locked() {   # cmd... -> exit 97 when MemAvailable < MEM_MIN_GB at lock time (ladder_chain.sh `locked`, verbatim logic)
  flock "$LOCK" bash -c 'm=$(awk "/MemAvailable/ {print int(\$2/1048576)}" /proc/meminfo)
    if [ -z "$m" ] || [ "$m" -lt "$0" ]; then exit 97; fi
    exec "$@"' "$MEM_MIN_GB" "$@" 200>&-
}
if [ -f "$EV/${ROW}_s0.json" ] && [ -f "$EV/${ROW}_s1.json" ]; then
  log "the $ROW passes already exist -- not re-run"
elif [ "$DRY" = 1 ]; then
  log "DRY: would run ladder_eval_xshuf.py --run $RUN --ckpt $CKPT --expect-step $N --rows $ROW --seeds 0,1 --out-dir $EV --device cuda"
else
  status EVAL-RUNNING
  while :; do
    locked timeout "$CAP_S" "$PY" "$CODE_A4/ladder_eval_xshuf.py" --run "$RUN" --ckpt "$CKPT" --expect-step "$N" \
        --rows "$ROW" --seeds 0,1 --out-dir "$EV" --device cuda >> "$A4/eval/$ARM.log" 2>&1
    rc=$?
    [ "$rc" != 97 ] && break
    log "MemAvailable < ${MEM_MIN_GB} GB at lock time -- lock released, retry in 5 min"
    sleep 300 200>&-
  done
  log "ladder_eval_xshuf exit $rc (not evidence; the artifacts are)"
fi

# ------------------------------------------------------------------------------------------- assert on the artifacts
[ "$DRY" = 1 ] && stop DRY-DONE "logic test only: nothing evaluated or scored"
rm -f "$A4/A4CHK.txt"
CUDA_VISIBLE_DEVICES= "$PY" - "$EV" "$W/eval/$ARM" "$ROW" "$EXPECT_DONOR_MAP" > "$A4/A4CHK.txt" 2>&1 200>&- <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
ev, base, row, want_map = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
bad = []
for s in (0, 1):
    j, z = ev / f"{row}_s{s}.json", ev / f"{row}_s{s}.npz"
    if not (j.is_file() and z.is_file()):
        bad.append(f"s{s}: missing artifact")
        continue
    r = json.loads(j.read_text(encoding="utf-8"))
    rb = json.loads((base / f"base_s{s}.json").read_text(encoding="utf-8"))
    x = r.get("xshuf") or {}
    if r.get("window_sha12_t_sha256") != rb.get("window_sha12_t_sha256"):
        bad.append(f"s{s}: window digest differs from the chain's base")
    if x.get("selfcheck") != "PASS":
        bad.append(f"s{s}: selfcheck {x.get('selfcheck')!r}")
    if want_map and x.get("donor_map_sha256") != want_map:
        bad.append(f"s{s}: donor map {x.get('donor_map_sha256')!r} != the pre-registered {want_map}")
    if not isinstance(x.get("n_changed"), int) or x["n_changed"] < 1:
        bad.append(f"s{s}: n_changed {x.get('n_changed')!r}")
    vf = np.load(str(z))["v_fed"]
    vb = np.load(str(base / f"base_s{s}.npz"))["v_fed"]
    nd = int((vf != vb).sum())
    print(f"A4CHK s{s}: n_changed {x.get('n_changed')} v_fed differs from base on {nd} / {vf.size}, "
          f"kept same {x.get('n_kept_same')}, no-donor episodes {x.get('n_episodes_no_donor')}, "
          f"donor map {str(x.get('donor_map_sha256'))[:16]}")
    if nd < int(x.get("n_changed") or 1):
        bad.append(f"s{s}: v_fed differs on {nd} < n_changed")
print("A4CHK", "OK" if not bad else "BAD " + "; ".join(bad))
sys.exit(0 if not bad else 6)
EOF
cat "$A4/A4CHK.txt" >> "$A4/post_chain.log" 2>/dev/null
grep -q "^A4CHK OK$" "$A4/A4CHK.txt" 2>/dev/null || stop EVAL-ARTIFACT-BAD "see $A4/A4CHK.txt and $A4/eval/$ARM.log"

# ------------------------------------------------------------------------------------------- score (CPU)
status SCORING
rm -f "$A4/LADDER_SCORE_A4.json" "$A4/A4_SUMMARY.txt"
CUDA_VISIBLE_DEVICES= "$PY" "$CODE_A4/ladder_score_a4.py" score --w "$W" --extra-eval "$A4/eval" \
    --v9-train "$V9TRAIN" --v9-train-md5 "$V9TRAIN_MD5" --out "$A4/LADDER_SCORE_A4.json" >> "$A4/post_chain.log" 2>&1 200>&-
CUDA_VISIBLE_DEVICES= "$PY" - "$A4/LADDER_SCORE_A4.json" > "$A4/A4_SUMMARY.txt" 2>&1 200>&- <<'EOF'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
a4 = r.get("a4") or {}
print("VERDICT", r.get("VERDICT", ""))
print("L4", json.dumps({k: (r.get("rungs") or {}).get("L4", {}).get(k) for k in ("status", "VERDICT", "missing")}))
print("instrument_blind", a4.get("instrument_blind"))
print("vmax_shuf", json.dumps({s: v.get("status") for s, v in (a4.get("vmax_shuf") or {}).items()}))
print("xshuf_control", json.dumps(a4.get("xshuf_control")))
print("x3_c3_reported", (a4.get("x3_c3_reported") or {}).get("reading"))
sys.exit(0 if "a4" in r else 8)
EOF
grep -qE "^instrument_blind (True|False)$" "$A4/A4_SUMMARY.txt" 2>/dev/null || stop SCORE-ARTIFACT-BAD "$A4/LADDER_SCORE_A4.json missing or carries no a4 block"
score_md5_after=$(md5sum "$W/LADDER_SCORE.json" 2>/dev/null 200>&- | cut -c1-32)
[ "${score_md5_before:-absent}" = "${score_md5_after:-absent}" ] || stop CHAIN-SCORE-CHANGED "LADDER_SCORE.json md5 ${score_md5_before:-absent} -> ${score_md5_after:-absent}"
# the A4 copy of the checkpoint is removed once both passes are verified (Thor disk; the chain's own copy is untouched)
[ "$CKPT" = "$A4/ckpt/$ARM/model_final.pt" ] && rm -f "$CKPT" && log "removed the A4 checkpoint copy $CKPT"
if grep -q "^instrument_blind True" "$A4/A4_SUMMARY.txt"; then
  status DONE-INSTRUMENT-BLIND "the vmax_xshuf control did not read its known value: $A4/A4_SUMMARY.txt"
else
  status DONE "$A4/LADDER_SCORE_A4.json + $A4/A4_SUMMARY.txt"
fi
exit 0
