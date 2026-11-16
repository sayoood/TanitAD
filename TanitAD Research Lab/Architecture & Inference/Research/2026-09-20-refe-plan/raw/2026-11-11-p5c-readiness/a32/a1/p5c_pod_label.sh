#!/usr/bin/env bash
# P5-C pod label runner, A3.1 version: the TWO-PASS RJ5 route of PREREG_P5C_A3 section 6 step 2.  FAIL-CLOSED: any unmet precondition prints "ZZP5CL_FAIL <stage>" and exits non-zero.
# Run ON THE POD, AFTER P7's last step (the pod's CPUs and DBs are free then).  WRITTEN AND `bash -n` CHECKED, TESTED IN A STUB SANDBOX ONLY; NEVER RUN BY ITS AUTHOR.
# Usage:  bash p5c_pod_label.sh   (env, all passed by refe/p5c_pod.sh: P5CL_LABEL=p5r-rj5  P5CL_NSHARD  P5CL_SHIP  P5CL_WORK  P5CL_JOIN  P5CL_JOIN_SHA256  P5CL_A31_SHA256  P5CL_PLATFORMS  P5CL_R0_ROWS)
# Flow (A3 section 6 step 2):  guards + pins -> feed -> CONTROL pass (override OFF: the 2,400 P5-R EVAL rows with their BANK trajs) -> control gates (TB-0, TB-1) -> RJ5 pass (all 150,603 rows, the feed bank)
#   -> rekey -> repair (4 runs: rank 0/1 x real bank / feed bank) + repair-combine -> main gates (TB-2..TB-7 + RK0, on the REAL bank) + tb0r -> per-gate classification FROM THE gates.json FILES (never the ZZTB line)
#   -> provenance -> table build --rank-policy rank0 --require-coverage 1.0 --meta-json -> verify -> TB-9 (sidecar keys == table keys) -> ZZP5CL_DONE.
# The control pass runs BEFORE the RJ5 pass and a TB-0/TB-1 failure ends the run there (no 2-3 h pass is spent on a labeller that does not reproduce); every gate verdict is unchanged by the order.
# Verdict lines:  ZZP5CL_GATE <where> <id> <status>   ZZP5CL_CLASS rj5=<pass|stop|fault>   ZZP5CL_TABLE p5r-rj5 rank_policy=rank0 sha256=<hex> table=<path>   ZZP5CL_DONE
#   exit 0 = ZZP5CL_DONE;  exit 3 = ZZP5CL_STOP (a LABEL verdict: RJ5 coverage under 99.0 %, TB-6 rank 0, or RK0: the next lever is A3 section 4's, C0 proceeds);  exit 1 = ZZP5CL_FAIL (a pipeline fault: no launch, fix and rebuild).
# Writes nothing under /workspace/refe-plan-p7 and never reads it.  Resumable only per labelling pass, with the SAME P5CL_NSHARD (the driver skips logs listed in done_<shard>.txt).
set -u

LABEL="${P5CL_LABEL:-}"
NSHARD="${P5CL_NSHARD:-}"
SHIP="${P5CL_SHIP:-/workspace/data/p5c_ship}"          # where the ship list's files land (flat, plus dev_labels/ and p5r_eval/)
WORK="${P5CL_WORK:-/workspace/data/p5c_label}"
JOIN="${P5CL_JOIN:-$SHIP/rj5_bank_join.jsonl.gz}"
JOINSHA="${P5CL_JOIN_SHA256:-}"
A31SHA="${P5CL_A31_SHA256:-}"
PLATFORMS="${P5CL_PLATFORMS:-}"                         # "SRC=devbox|pod ..." declared per join source, never guessed
R0ROWS="${P5CL_R0_ROWS:-}"                              # "SRC=DIR ..." R0 control rows for TB-8
SRC_TREE=/workspace/refe-plan-vdac                      # NEVER /workspace/refe-plan-p7
BANK=/workspace/data/refe_navtrain/train_grow_cmd_3c_p7
DBROOT=/workspace/data/navtrain_dbs
MAPS=/workspace/data/nuplan-maps/nuplan-maps-v1.0
PY=/workspace/venv-teacher/bin/python
EXPECT_ROWS=150603
EVAL_N=2400                                             # P5-R EVAL rows (1,200 per rank): the control pass and TB-0 / TB-1
EXPECT_DEV_RJ5=16133                                    # rows of rj5_labels_local.jsonl.gz (A3 TB-0r)
MIN_COV=0.99                                            # A3 section 3.1: the committed launch bar on the RJ5 coverage of rank-0 keys; a constant, never an env var

# ---- pins (sha256) -----------------------------------------------------------------------------
PIN_BANK_R0=a5690f8186f940eccacf68f45c7b49bb788ad3190966c8fc99c40912b8384a24
PIN_BANK_R1=8b6ccd1b0b85e9f7af12e7d0897efbf8dd9b750f9b50b653d9ac070658f927aa
PIN_P5C_LABELS=77724709088044d19e132cde5eb8a6e549665af37a8a79bcfd245f1ad2a8f933          # p5c_labels.py (reference; the dual labeller copies its Tc20s lines)
PIN_DUAL=07e84b1cfd26d72c4fef7ea5557b07460e39f808a6bbba6a4d0df582ea7bd197                # p5c_dual_labels.py (blob 50b9f0db)
PIN_P5R_LABELS=84e1cac32510fb277ad7d1a876059152dbded5076b4fd1928f5ade330b6c468d          # p5r_labels.py (PREREG_P5R solver)
PIN_P5R_RUN=58faf7a76cb1840f4a8ed76ee6eaa8bb1d8c835f20009cb7a2749990cc7feddb             # p5r_run.py (imported: Sim, sanity)
PIN_LAMBDA=83988ddac582a8a1cd37e9e11745e195abe8012bd093a4dfd72acbb608bdd467              # lambda_star.json (lam* 3.0)
PIN_EVAL_R0=a576bf2a80d6ca7f39217ff58c8e8baa3b179352a6dddbc81ac4aad97679d5fd             # p5r_eval/eval_r0.jsonl
PIN_EVAL_R1=1d2762b19deae138b0ef9906e75d3921ddf43f4c5a949dfb88dc00bfbf8c37b0             # p5r_eval/eval_r1.jsonl
PIN_VERDICT=7ae694a3bb754d8b7ddbd355f0b16928498fefd91cb4d0b71c996d9587a17e9e             # p5r_eval/verdict.json (context for R-TOL-3, not gated)
PIN_P5_TARGET=2b31af64d6f6bcff7ffc0a59d524528b48ad8fceb44033e9cc845b201326c8d0           # eval/p5_target_labels.py (git blob 294644df)
PIN_P4_TRACKER=193c024b107fd2b73fdb9541f805a1000a1c882660f7cd2684a5bc70335609d5          # eval/p4_tracker_comp.py (imported by p5_target_labels)
PIN_TABLE_MOD=ea763fb2bd857f172acba32d1a5a76ee17ae08077801bbc096970069451d77d8           # traj_label_table.py (A2 pin d65e01c4)
PIN_GATES=0c567514a0bb97708cf3d41bbaff5727c59c7fe380353cc9ccefa43654d00d35               # p5c_table_gates.py (A3.2 pin; A1 was 5c2f49cf, with --label p5r and the A3.2 scope flag)
PIN_DEV_S0=7f8d58cc967117ef9d66ed98c6c04d7df044cc98cdc6ea889ae43199bccb5bd3              # dev_labels/labels_s0of2.jsonl (the Tc20s control of the control pass)
PIN_DEV_S1=2dea6e4326ea38feb7ec4ff06c3daaabff5632d700b948ebf3ace8b81083f2a1              # dev_labels/labels_s1of2.jsonl
PIN_RJ5PATH=951d3e1d5027bedc933b2bd766c5df2d98373b3829209ffd71f87441944ae309             # p5c_rj5_path.py (A3 pin 409f4481)
PIN_DEV_RJ5=23f59e5351b41d943e815e2d5847bb3ff936b7fb9fa0a50db1eb0a58f44ae686             # rj5_labels_local.jsonl.gz (the 16,133 dev-box RJ5 records, TB-0r's reference)

fail() { echo "ZZP5CL_FAIL $1 ${2:-}"; exit 1; }
stop() { echo "ZZP5CL_STOP $1 ${2:-}"; exit 3; }
sha_of() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }
is_sha256() { case "$1" in *[!0-9a-f]*|'') return 1 ;; esac; [ "${#1}" = 64 ]; }
pin() {  # pin <file> <expected-sha256> <stage>
  [ -f "$1" ] || fail "$3" "missing $1"
  local got; got="$(sha_of "$1")"
  [ "$got" = "$2" ] || fail "$3" "sha256 of $1 is $got, expected $2"
}
cpu_quota() {          # floor(the cgroup CPU quota); rc 1 when none is readable.  NEVER nproc (128 on the pod that grants 15.3 CPUs).  P5CL_CGROUP_ROOT is for tests.
  local cg="${P5CL_CGROUP_ROOT:-/sys/fs/cgroup}" q="" per=""
  { read -r q per < "$cg/cpu.max"; } 2> /dev/null
  case "$q" in ''|max|*[!0-9]*) ;; *) case "$per" in ''|*[!0-9]*|0) ;; *) echo $(( q / per > 0 ? q / per : 1 )); return 0 ;; esac ;; esac
  q=""; per=""
  { read -r q < "$cg/cpu/cpu.cfs_quota_us"; read -r per < "$cg/cpu/cpu.cfs_period_us"; } 2> /dev/null
  case "$q" in ''|*[!0-9]*|0) return 1 ;; esac
  case "$per" in ''|*[!0-9]*|0) return 1 ;; esac
  echo $(( q / per > 0 ? q / per : 1 ))
}

echo "P5CL start $(date -u +%FT%TZ) label=$LABEL nshard=$NSHARD ship=$SHIP work=$WORK a31_sha256=${A31SHA:0:16} join_sha256=${JOINSHA:0:16}"
case "$SRC_TREE $SHIP $WORK" in *refe-plan-p7*) fail guard_p7_path ;; esac
[ "$LABEL" = p5r-rj5 ] || fail bad_label "P5CL_LABEL is '$LABEL': this runner builds only the A3 label p5r-rj5"
case "$NSHARD" in ''|*[!0-9]*) fail bad_nshard "P5CL_NSHARD='$NSHARD' (set by p5c_pod.sh; no default here)" ;; esac
[ "$NSHARD" -ge 1 ] || fail bad_nshard
QUOTA="$(cpu_quota)" || fail cpu_quota "no readable cgroup CPU quota: the shard count is capped by it, never by nproc"
[ "$NSHARD" -le "$QUOTA" ] || fail nshard_over_quota "P5CL_NSHARD=$NSHARD exceeds floor(cgroup CPU quota)=$QUOTA"
is_sha256 "$JOINSHA" || fail join_sha_unset "P5CL_JOIN_SHA256 is not a sha256: the RJ5 source is pinned before feed runs (A3 section 2)"
is_sha256 "$A31SHA" || fail a31_sha_unset "P5CL_A31_SHA256 is not a sha256: no C1 label is built before Amendment A3.1 is frozen"
[ -n "$PLATFORMS" ] || fail platforms_unset "P5CL_PLATFORMS: each join source's roll platform is declared, never guessed"
[ -n "$R0ROWS" ] || fail r0_rows_unset "P5CL_R0_ROWS: TB-8 needs the R0 control rows"
pin "$JOIN" "$JOINSHA" pin_join

# ---- 1. refuse while a trainer runs (exact argv fields, never pgrep -f / pkill -f) --------------
BUSY="$(ps -eo args= | awk '$1 ~ /(^|\/)python[0-9.]*$/ { for (i = 2; i <= NF; i++) if ($i ~ /(^|\/)(train\.py|scorer_finetune\.py)$/) { print $0; break } }')"
[ -z "$BUSY" ] || fail guard_trainer_running "$(echo "$BUSY" | head -1 | cut -c1-160)"

# ---- 2. pins -------------------------------------------------------------------------------------
[ -x "$PY" ] || fail venv_python "$PY not executable"
[ -d "$BANK" ] || fail bank_dir "$BANK"
[ -d "$DBROOT" ] || fail db_root "$DBROOT"
[ -d "$MAPS" ] || fail maps_root "$MAPS"
[ -d "$SRC_TREE/refe" ] && [ -d "$SRC_TREE/code" ] || fail src_tree "$SRC_TREE/refe or /code missing"
pin "$BANK/targets_rank0.jsonl" "$PIN_BANK_R0" pin_bank_rank0
pin "$BANK/targets_rank1.jsonl" "$PIN_BANK_R1" pin_bank_rank1
pin "$SHIP/p5c_labels.py"                 "$PIN_P5C_LABELS" pin_p5c_labels
pin "$SHIP/p5c_dual_labels.py"            "$PIN_DUAL"       pin_dual_labels
pin "$SHIP/p5r_labels.py"                 "$PIN_P5R_LABELS" pin_p5r_labels
pin "$SHIP/p5r_run.py"                    "$PIN_P5R_RUN"    pin_p5r_run
pin "$SHIP/lambda_star.json"              "$PIN_LAMBDA"     pin_lambda_star
pin "$SHIP/p5r_eval/eval_r0.jsonl"        "$PIN_EVAL_R0"    pin_eval_r0
pin "$SHIP/p5r_eval/eval_r1.jsonl"        "$PIN_EVAL_R1"    pin_eval_r1
pin "$SHIP/p5r_eval/verdict.json"         "$PIN_VERDICT"    pin_verdict
pin "$SHIP/p5_target_labels.py"           "$PIN_P5_TARGET"  pin_p5_target_labels
pin "$SHIP/p4_tracker_comp.py"            "$PIN_P4_TRACKER" pin_p4_tracker_comp
pin "$SHIP/traj_label_table.py"           "$PIN_TABLE_MOD"  pin_table_module
pin "$SHIP/p5c_table_gates.py"            "$PIN_GATES"      pin_gates
pin "$SHIP/dev_labels/labels_s0of2.jsonl" "$PIN_DEV_S0"     pin_dev_labels_s0
pin "$SHIP/dev_labels/labels_s1of2.jsonl" "$PIN_DEV_S1"     pin_dev_labels_s1
pin "$SHIP/p5c_rj5_path.py"               "$PIN_RJ5PATH"    pin_rj5_path
pin "$SHIP/rj5_labels_local.jsonl.gz"     "$PIN_DEV_RJ5"    pin_dev_rj5_labels
[ -f "$SHIP/TB1_control_1018.json" ] || fail ship_missing "$SHIP/TB1_control_1018.json"

# ---- 3. package dir: refe/ + code/ from refe-plan-vdac, eval/ from the shipped, pinned files -----
PKG="$WORK/pkg"
mkdir -p "$WORK" || fail mkdir_work
rm -rf "${PKG:?}" 2>/dev/null          # PKG is this script's own re-creatable copy (literal path under $WORK)
mkdir -p "$PKG/eval" || fail mkdir_pkg
( cd "$SRC_TREE" && tar --exclude='__pycache__' -cf - refe code ) | ( cd "$PKG" && tar -xf - ) || fail pkg_copy
cp "$SHIP/p5_target_labels.py" "$SHIP/p4_tracker_comp.py" "$PKG/eval/" || fail pkg_eval
[ -f "$PKG/refe/navsim_dac.py" ] && [ -f "$PKG/refe/navtrain_scenarios.py" ] || fail pkg_refe_incomplete
pin "$PKG/eval/p5_target_labels.py" "$PIN_P5_TARGET" pin_pkg_p5_target_labels

# THREADS (MM ruling R-C): see the A1 runner: Tc20s runs in the caller's env, the P5-R part single-threaded (threadpoolctl when present, else OPENBLAS/MKL/OMP_NUM_THREADS = 1, which the labeller requires and records).
export P5C_PKG="$PKG" P5C_BANKDIR="$BANK" P5C_DBROOT="$DBROOT" NUPLAN_MAPS_ROOT="$MAPS" REFE_SIM_HZ=10 P5R_DIR="$SHIP" P5R_LAMBDA_JSON="$SHIP/lambda_star.json" P5R_BLAS_THREADS=1
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[ "$OMP_NUM_THREADS/$MKL_NUM_THREADS/$OPENBLAS_NUM_THREADS" = "1/1/1" ] || fail thread_env "OMP/MKL/OPENBLAS_NUM_THREADS must all be 1: $OMP_NUM_THREADS/$MKL_NUM_THREADS/$OPENBLAS_NUM_THREADS"
"$PY" -c "import sys, nuplan, scipy; sys.path[:0] = ['$SHIP', '$PKG/eval', '$PKG/refe', '$PKG/code']; import navsim_dac, navtrain_scenarios, p5_target_labels, p5r_labels, p5r_run
try:
    import threadpoolctl; print('ZZPROBE thread_pin=threadpoolctl')
except ImportError:
    print('ZZPROBE thread_pin=env (threadpoolctl absent; OPENBLAS/MKL/OMP_NUM_THREADS=1 required and exported)')" \
  >"$WORK/import_probe.log" 2>&1 || fail venv_imports "see $WORK/import_probe.log"
grep -h -- ZZPROBE "$WORK/import_probe.log"

RJ5="$SHIP/p5c_rj5_path.py"
GATES_PY="$SHIP/p5c_table_gates.py"
LABELLER="$SHIP/p5c_dual_labels.py"
FEED="$WORK/feed"
CTL="$WORK/control"
OUT="$WORK/out"
GATES="$WORK/gates"
REP="$WORK/repair"
mkdir -p "$GATES/control" "$GATES/main" "$REP" || fail mkdir_work_dirs

# ---- 4. feed: the feed bank (traj := the row's RJ5 traj, else the bank traj) + the per-row map.  Exit 3 = RJ5 coverage under the committed 99.0 % bar (a label verdict), exit 2 = a refusal (a fault) ----------------
FEEDARGS=()
for s in $PLATFORMS; do FEEDARGS+=(--platform "$s"); done
for s in $R0ROWS; do FEEDARGS+=(--r0-rows "$s"); done
"$PY" "$RJ5" feed --bank-dir "$BANK" --join "$JOIN" --expect-join-sha256 "$JOINSHA" "${FEEDARGS[@]}" --min-rj5-coverage "$MIN_COV" --out-dir "$FEED" > "$WORK/feed.log" 2>&1
RC=$?
tail -n 3 "$WORK/feed.log" | cut -c1-400
[ "$RC" != 3 ] || stop coverage "RJ5 coverage of rank-0 keys is under $MIN_COV (A3 section 3.1): no C1 launch; roll the missing logs and rebuild; C0 proceeds ($(grep -h ZZRJ5_REFUSE "$WORK/feed.log" | cut -c1-200))"
[ "$RC" = 0 ] && grep -q "^ZZRJ5_FEED_OK" "$WORK/feed.log" || fail feed "rc $RC: $(grep -h ZZRJ5_REFUSE "$WORK/feed.log" | head -1 | cut -c1-300)"
MAPSHA="$(sha_of "$FEED/feed_map.jsonl.gz")"
is_sha256 "$MAPSHA" || fail feed_map_sha "no $FEED/feed_map.jsonl.gz"
"$PY" -c "import json,sys; m=json.load(open(sys.argv[1])); assert m['feed_map_sha256']==sys.argv[2] and m['join_sha256']==sys.argv[3] and m['n_bank_rows']==int(sys.argv[4]), m['feed_map_sha256']" "$FEED/feed_manifest.json" "$MAPSHA" "$JOINSHA" "$EXPECT_ROWS" || fail feed_manifest "the feed manifest is not about this map / join / bank"
echo "ZZP5CL_FEED map_sha256=$MAPSHA join_sha256=$JOINSHA"

# one labelling pass over a bank dir with N shards of the PINNED labeller, at nice 19.  run_shards <bank dir> <out dir> <expected rows> <tag>
run_shards() {
  local bdir="$1" out="$2" want="$3" tag="$4" i=0 pids="" badrc="" p n f
  mkdir -p "$out" || fail "mkdir_out_$tag"
  while [ "$i" -lt "$NSHARD" ]; do
    P5C_BANKDIR="$bdir" nice -n 19 "$PY" "$LABELLER" label --shard "$i" --nshard "$NSHARD" --out "$out" >"$out/shard_${i}.log" 2>&1 &
    pids="$pids $!"
    i=$((i + 1))
  done
  echo "P5CL $tag pass: launched $NSHARD shards on $bdir: $pids"
  for p in $pids; do wait "$p" || badrc="$badrc $p"; done
  [ -z "$badrc" ] || fail "shard_rc_$tag" "nonzero exit for pids:$badrc (see $out/shard_*.log; re-run resumes)"
  i=0
  while [ "$i" -lt "$NSHARD" ]; do
    grep -q -- "ZZLABEL_DONE" "$out/shard_${i}.log" || fail "shard_not_done_$tag" "shard $i has no ZZLABEL_DONE"
    [ -f "$out/labels_s${i}of${NSHARD}.jsonl" ] || fail "shard_no_output_$tag" "shard $i"
    i=$((i + 1))
  done
  n="$(cat "$out"/labels_s*of"${NSHARD}".jsonl | wc -l | tr -d ' ')"
  [ "$n" = "$want" ] || fail "row_count_$tag" "label records $n != $want (a resumed shard may have duplicates: the gates would catch it, but stop here)"
  for f in "$out"/labels_s*of*.jsonl; do case "$f" in *of"${NSHARD}".jsonl) ;; *) fail "stale_other_nshard_$tag" "$f" ;; esac; done
}

# ---- 5. the CONTROL pass: the pinned labeller with the override OFF on the 2,400 P5-R EVAL rows (their BANK trajs).  TB-0 / TB-1 are judged here (A3 section 4) -------------------------------------------------------
"$PY" - "$BANK" "$SHIP/p5r_eval/eval_r0.jsonl" "$SHIP/p5r_eval/eval_r1.jsonl" "$CTL/bank" "$EVAL_N" > "$WORK/control_bank.log" 2>&1 <<'PYEOF' || fail control_bank "$(tail -n 1 "$WORK/control_bank.log" | cut -c1-300)"
# --- CONTROL-BANK-BEGIN: the control bank = exactly the EVAL rows, with the REAL bank's own lines (bank trajs: the override is OFF); every EVAL key must be found once
import glob, hashlib, json, os, sys
bank, e0, e1, out, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
want = set()
for p in (e0, e1):
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            if "error" not in d:
                want.add((d["log"], d["token"], int(d["rank"])))
if len(want) != n:
    sys.exit("the EVAL files carry %d usable control rows, not %d" % (len(want), n))
os.makedirs(out, exist_ok=True)
found = set()
for rk in (0, 1):
    with open(os.path.join(out, "targets_rank%d.jsonl" % rk), "w", encoding="utf-8", newline="\n") as fo:
        for ln in open(os.path.join(bank, "targets_rank%d.jsonl" % rk), encoding="utf-8"):
            if not ln.strip():
                continue
            r = json.loads(ln)
            k = (r["log_name"], r["token"], int(r["rank"]))
            if k in want:
                if k in found:
                    sys.exit("duplicate bank row %r" % (k,))
                found.add(k)
                fo.write(ln if ln.endswith("\n") else ln + "\n")
if found != want:
    sys.exit("%d EVAL keys are not rows of the real bank (e.g. %r)" % (len(want - found), sorted(want - found)[:2]))
print("ZZCTL_BANK %d rows" % len(found), {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in sorted(glob.glob(os.path.join(out, "targets_rank*.jsonl")))})
# --- CONTROL-BANK-END
PYEOF
grep -h ZZCTL_BANK "$WORK/control_bank.log" | cut -c1-300
run_shards "$CTL/bank" "$CTL/out" "$EVAL_N" control
"$PY" "$GATES_PY" --label p5r --bank "$CTL/bank" --labels "$CTL"/out/labels_s*of"${NSHARD}".jsonl \
  --control "$SHIP/dev_labels/labels_s0of2.jsonl" "$SHIP/dev_labels/labels_s1of2.jsonl" \
  --eval-control "$SHIP/p5r_eval/eval_r0.jsonl" "$SHIP/p5r_eval/eval_r1.jsonl" --eval-n "$EVAL_N" --lambda-star "$SHIP/lambda_star.json" --p5r-verdict "$SHIP/p5r_eval/verdict.json" --tc20s-tol 1e-9 --tc20s-ranks 0 \
  --tb1 "$SHIP/TB1_control_1018.json" --expect-rows "$EVAL_N" --out-dir "$GATES/control" > "$GATES/control/gates_stdout.txt" 2>&1
[ -f "$GATES/control/gates.json" ] || fail gates_control_no_json "see $GATES/control/gates_stdout.txt"
# --- the classifier: one python program, per-gate verdicts read FROM THE gates.json FILES.  The ZZTB line is printed for the log and never read.
cat > "$WORK/classify.py" <<'PYEOF'
# --- CLASSIFY-BEGIN
import json, sys


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def say(where, gid, status, note=""):
    print("ZZP5CL_GATE %s %s %s%s" % (where, gid, status, (" " + note) if note else ""), flush=True)


def control(gates_json):
    g = load(gates_json)["gates"]
    bad = []
    for gid in ("TB-0", "TB-1"):
        st = g.get(gid, {}).get("status", "MISSING")
        say("control", gid, st)
        if st != "PASS":
            bad.append(gid)
    return 1 if bad else 0


def counts_by_source(counts):
    out = {}
    for k, v in counts.items():
        parts = k.split("|")
        kk = "|".join(parts[:3])
        out[kk] = out.get(kk, 0) + v
    return out


def main_(gates_json, tb0r_json, feed_manifest, rekey_manifest, expect_rows, out_json):
    d = load(gates_json)
    g = d["gates"]
    fault, stop = [], []
    for gid in ("TB-2", "TB-3", "TB-4", "TB-5", "TB-7"):
        st = g.get(gid, {}).get("status", "MISSING")
        say("main", gid, st)
        if st != "PASS":
            fault.append(gid)
    for gid in ("TB-0", "TB-1"):
        say("main", gid, g.get(gid, {}).get("status", "MISSING"), "(judged on the control pass: an RJ5 label differs from the bank-traj EVAL labels by construction)")
    t6 = g.get("TB-6", {})
    det = t6.get("detail", {})
    r0ok = det.get("rank0", {}).get("pass") is True
    r1ok = det.get("rank1", {}).get("pass") is True
    say("main", "TB-6", "PASS" if r0ok else "FAIL", "rank0=%s rank1=%s (rank 1 is reported and decides nothing under the forced rank0 policy)" % (r0ok, r1ok))
    if not r0ok:
        stop.append("TB-6 rank 0")
    pol = d.get("rank_policy", {})
    dec = pol.get("decision")
    say("main", "RK0", "STOP" if dec == "STOP" else ("UNDECIDED" if dec not in ("all", "rank0") else "no-stop"), "decision=%s" % dec)
    if dec == "STOP":
        stop.append("RK0 / rank policy STOP")
    elif dec not in ("all", "rank0"):
        fault.append("RK0 policy %r" % (dec,))
    tb0r = load(tb0r_json)
    say("main", "TB-0r", tb0r.get("status", "MISSING"))
    if tb0r.get("status") != "PASS":
        fault.append("TB-0r")
    fm, rm = load(feed_manifest), load(rekey_manifest)
    c_feed, c_rek = fm.get("counts", {}), counts_by_source(rm.get("counts", {}))
    t9 = []
    if c_feed != c_rek:
        t9.append("rekey counts %r != feed counts %r" % (c_rek, c_feed))
    if rm.get("n_records") != fm.get("n_bank_rows") or fm.get("n_bank_rows") != expect_rows:
        t9.append("n_records %r / n_bank_rows %r / expected %r" % (rm.get("n_records"), fm.get("n_bank_rows"), expect_rows))
    if rm.get("feed_map_sha256") != fm.get("feed_map_sha256"):
        t9.append("the rekey manifest is about another feed map")
    nfb, n0 = fm.get("counts", {}).get("bank_fallback|missing|rank0", 0), fm.get("n_bank_rank0_keys", 0)
    if not n0 or nfb > 0.01 * n0:
        t9.append("bank_fallback/missing %r of %r rank-0 keys exceeds the 1.0 %% bar (A3 section 3.1)" % (nfb, n0))
    say("main", "TB-9", "FAIL" if t9 else "PASS", "; ".join(t9))
    if t9:
        fault.append("TB-9")
    verdict = "fault" if fault else ("stop" if stop else "pass")
    json.dump({"verdict": verdict, "fault": fault, "stop": stop, "rank_decision": dec}, open(out_json, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZP5CL_CLASS rj5=%s%s" % (verdict, (" " + "; ".join(fault + stop)) if verdict != "pass" else ""), flush=True)
    return {"pass": 0, "stop": 3, "fault": 1}[verdict]


def sidecar_vs_table(sidecar, table, expect_rows):
    import gzip
    side = set()
    with gzip.open(sidecar, "rt", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                side.add(json.loads(ln)["table_key"])
    with gzip.open(table, "rt", encoding="utf-8") as f:
        keys = set(json.load(f)["table"])
    ok = side == keys and len(keys) == expect_rows
    say("table", "TB-9", "PASS" if ok else "FAIL", "sidecar keys %d, table keys %d (expected %d), equal=%s" % (len(side), len(keys), expect_rows, side == keys))
    return 0 if ok else 1


def force_rank0(meta_in, meta_out):
    m = load(meta_in)
    if m.get("rank_policy") not in ("all", "rank0"):
        print("ZZP5CL_REFUSE the gates meta carries rank_policy %r" % (m.get("rank_policy"),))
        return 1
    m["rank_policy_gates_decision"] = m["rank_policy"]
    m["rank_policy"] = "rank0"
    json.dump(m, open(meta_out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "control":
        sys.exit(control(a[0]))
    if cmd == "main":
        sys.exit(main_(a[0], a[1], a[2], a[3], int(a[4]), a[5]))
    if cmd == "sidecar":
        sys.exit(sidecar_vs_table(a[0], a[1], int(a[2])))
    if cmd == "force-rank0":
        sys.exit(force_rank0(a[0], a[1]))
    sys.exit("unknown command " + cmd)
# --- CLASSIFY-END
PYEOF
echo "ZZP5CL_GATES control (informational, the ZZTB line is never read): $(grep -- '^ZZTB_' "$GATES/control/gates_stdout.txt" | tail -1)"
"$PY" "$WORK/classify.py" control "$GATES/control/gates.json" || fail control_gates "TB-0 / TB-1 on the control pass did not PASS: the pipeline does not reproduce the dev-box labels (A3 section 4: ZZP5CL_FAIL, no C1 launch, fix and rebuild; C0 proceeds)"

# ---- 6. the RJ5 pass: the pinned labeller on the feed bank (all 150,603 rows, <= P5CL_NSHARD shards) --------------------------------------------------------------------------------------------------------------
run_shards "$FEED/bank" "$OUT" "$EXPECT_ROWS" rj5

# ---- 7. rekey: the labeller's records (keyed by the FED traj's sha) -> keyed by the BANK traj's sha, each with its provenance.  The feed-map sha is pinned into it -------------------------------------------------
mkdir -p "$WORK/rekeyed" || fail mkdir_rekeyed
"$PY" "$RJ5" rekey --feed-map "$FEED/feed_map.jsonl.gz" --expect-feed-map-sha256 "$MAPSHA" --labels "$OUT"/labels_s*of"${NSHARD}".jsonl --out "$WORK/rekeyed/labels_rekeyed.jsonl" > "$WORK/rekey.log" 2>&1
RC=$?
[ "$RC" = 0 ] && grep -q "^ZZRJ5_REKEY_OK" "$WORK/rekey.log" || fail rekey "rc $RC: $(grep -h ZZRJ5_REFUSE "$WORK/rekey.log" | head -1 | cut -c1-300)"
REKEYED="$WORK/rekeyed/labels_rekeyed.jsonl"

# ---- 8. the RK0 inputs: the pinned labeller's `repair` on the REAL bank (raw sim8 of the bank traj = C0's target) and on the FEED bank (the RJ5-based labels), per rank; repair-combine joins them --------------------------
PIDS=""; BADRC=""
for k in 0 1; do
  P5C_BANKDIR="$BANK" nice -n 19 "$PY" "$LABELLER" repair --rank "$k" --n 300 --out "$REP/bank_rank${k}.json" >"$REP/bank_rank${k}.log" 2>&1 &
  PIDS="$PIDS $!"
  P5C_BANKDIR="$FEED/bank" nice -n 19 "$PY" "$LABELLER" repair --rank "$k" --n 300 --out "$REP/feed_rank${k}.json" >"$REP/feed_rank${k}.log" 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait "$p" || BADRC="$BADRC $p"; done
[ -z "$BADRC" ] || fail repair_rc "nonzero exit for pids:$BADRC (see $REP/*.log)"
for k in 0 1; do
  for w in bank feed; do
    [ -s "$REP/${w}_rank${k}.json" ] && grep -q -- "ZZREPAIR" "$REP/${w}_rank${k}.log" || fail repair_missing "$w rank $k (see $REP/${w}_rank${k}.log)"
  done
  "$PY" "$RJ5" repair-combine --bank-repair "$REP/bank_rank${k}.json" --feed-repair "$REP/feed_rank${k}.json" --out "$REP/repair_rank${k}_a3.json" > "$REP/combine_rank${k}.log" 2>&1 \
    && grep -q "^ZZRJ5_REPAIR_OK" "$REP/combine_rank${k}.log" || fail repair_combine "rank $k: $(grep -h ZZRJ5_REFUSE "$REP/combine_rank${k}.log" | head -1 | cut -c1-300)"
done

# ---- 9. the main gates, on the REAL bank with the rekeyed records.  No --control / --eval-control: TB-0 and TB-1 are judged on the control pass (they print NOT_RUN here, which the ZZTB line counts as failing: that is why the class is read per gate) -----
"$PY" "$GATES_PY" --label p5r --bank "$BANK" --labels "$REKEYED" --repair0 "$REP/repair_rank0_a3.json" --repair1 "$REP/repair_rank1_a3.json" \
  --expect-rows "$EXPECT_ROWS" --lambda-star "$SHIP/lambda_star.json" --tc20s-tol 1e-9 \
  --code-file "$PKG/eval/p5_target_labels.py" "$SHIP/p5c_dual_labels.py" "$SHIP/p5r_labels.py" "$SHIP/p5r_run.py" "$SHIP/p5c_table_gates.py" "$SHIP/traj_label_table.py" "$SHIP/p5c_rj5_path.py" \
  --out-dir "$GATES/main" > "$GATES/main/gates_stdout.txt" 2>&1
[ -f "$GATES/main/gates.json" ] && [ -f "$GATES/main/table_meta_fields.json" ] || fail gates_main_no_json "see $GATES/main/gates_stdout.txt"
echo "ZZP5CL_GATES main (informational, the ZZTB line is never read): $(grep -- '^ZZTB_' "$GATES/main/gates_stdout.txt" | tail -1)"

# ---- 10. TB-0r: the pod's RJ5-based labels against the dev box's on the same RJ5 inputs (the 16,133 local rows); the pinned gate functions, through p5c_rj5_path.py tb0r -----------------------------------------------
"$PY" "$RJ5" tb0r --gates-module "$GATES_PY" --feed-bank "$FEED/bank" --pod-labels "$OUT"/labels_s*of"${NSHARD}".jsonl --dev-labels "$SHIP/rj5_labels_local.jsonl.gz" --expect-n "$EXPECT_DEV_RJ5" --out "$WORK/tb0r.json" > "$WORK/tb0r.log" 2>&1
RC=$?
[ -f "$WORK/tb0r.json" ] || fail tb0r "rc $RC: $(tail -n 1 "$WORK/tb0r.log" | cut -c1-300)"

# ---- 11. classify per gate, from the gates.json files (A3 section 4's two branches) ------------------------------------------------------------------------------------------------------------------------------------------
"$PY" "$WORK/classify.py" main "$GATES/main/gates.json" "$WORK/tb0r.json" "$FEED/feed_manifest.json" "$WORK/rekeyed/labels_rekeyed.jsonl.manifest.json" "$EXPECT_ROWS" "$WORK/class.json"
RC=$?
case "$RC" in
  0) ;;
  3) stop gates "TB-6 rank 0 FAILED or RK0 fired (A3 section 4): the RJ5-based label is not learnable or is harmful at bank scale; no table; the next lever is P5-T on RJ5 inputs; C0 proceeds ($(tr -d '\n' < "$WORK/class.json" | cut -c1-300))" ;;
  *) fail gates "a gate outside the label verdict failed: $(tr -d '\n' < "$WORK/class.json" 2> /dev/null | cut -c1-300)" ;;
esac

# ---- 12. provenance (the sidecar + the gates' meta with the a3 block), then the table.  The rank policy is FORCED to rank0 (A3 section 3.3): the builder refuses a meta whose rank_policy is another policy, so the gates' own decision is recorded beside it -----
"$PY" "$WORK/classify.py" force-rank0 "$GATES/main/table_meta_fields.json" "$GATES/main/table_meta_fields_rank0.json" || fail force_rank0 "the gates meta carries no all|rank0 policy"
"$PY" "$RJ5" provenance --feed-map "$FEED/feed_map.jsonl.gz" --rekey-manifest "$WORK/rekeyed/labels_rekeyed.jsonl.manifest.json" --gates-meta "$GATES/main/table_meta_fields_rank0.json" \
  --out-meta "$GATES/main/table_meta_fields_a3.json" --out-sidecar "$WORK/C1_provenance.jsonl.gz" \
  --extra "tb0r=$WORK/tb0r.json" --extra "control_gates=$GATES/control/gates.json" --extra "main_gates=$GATES/main/gates.json" --extra "feed_manifest=$FEED/feed_manifest.json" --extra "join=$JOIN" > "$WORK/provenance.log" 2>&1 \
  && grep -q "^ZZRJ5_PROVENANCE_OK" "$WORK/provenance.log" || fail provenance "$(grep -h ZZRJ5_REFUSE "$WORK/provenance.log" | head -1 | cut -c1-300)"
MERGED="$GATES/main/merged_labels_rank0only.jsonl"
[ -f "$MERGED" ] || fail merged_missing "$MERGED"
TABLE="$WORK/P5C_label_table_p5r-rj5_rank0.json.gz"
"$PY" "$SHIP/traj_label_table.py" build --bank-dir "$BANK" --labels "$MERGED" --out "$TABLE" --require-coverage 1.0 --label-name P5R --rank-policy rank0 --meta-json "$GATES/main/table_meta_fields_a3.json" >"$WORK/table_build.log" 2>&1 || fail table_build "see $WORK/table_build.log"
"$PY" "$SHIP/traj_label_table.py" verify --table "$TABLE" --bank-dir "$BANK" >"$WORK/table_verify.json" 2>&1 || fail table_verify "see $WORK/table_verify.json"
"$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['rows']==$EXPECT_ROWS and d['covered']==$EXPECT_ROWS and not d['refused'] and d['rank_policy']=='rank0', d" "$WORK/table_verify.json" || fail table_verify_counts
"$PY" "$WORK/classify.py" sidecar "$WORK/C1_provenance.jsonl.gz" "$TABLE" "$EXPECT_ROWS" || fail tb9_sidecar "the provenance sidecar's keys are not the table's keys (A3 TB-9)"
TSHA="$(sha_of "$TABLE")"
echo "$TSHA  $TABLE" > "$WORK/P5C_LABEL_TABLE_SHA256_p5r-rj5.txt"
echo "ZZP5CL_TABLE p5r-rj5 rank_policy=rank0 sha256=$TSHA table=$TABLE"
echo "P5C_LABEL_TABLE_SHA256=$TSHA  label=p5r-rj5"
echo "$TSHA  label=p5r-rj5" > "$WORK/P5C_LABEL_TABLE_SHA256.txt"
echo "ZZP5CL_NOTE the C1 table is built; the MM writes LABEL_SELECTION.md (label p5r-rj5, the two gates.json, tb0r.json, the join sha) and fills P5C_LABEL_TABLE_SHA256 (A3 section 4)"
echo "ZZP5CL_DONE"
exit 0
