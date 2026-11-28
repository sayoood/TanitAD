#!/bin/bash
# REHEARSAL ONLY.  Builds a MOCK pod (a directory tree under $MOCK, default /tmp/p6mock, inside WSL Ubuntu on the dev box) that has the same paths, markers, file formats and
# commands as the real pod for steps (a)-(e), so that post_merge_to_p6gate.sh can be executed end to end WITHOUT touching the real pod.
#   REAL, from the 3-C package:  cprime_check.py, reroll_bank_3c.py (+ the others), the real mini-bank MERGE OUTPUT (results/controls/pod/merged: 173 rows, 23 kept twins, a real
#                                merge_report.json and audit_all.jsonl), the real input mini bank (bank_i3: 224 rows, 74 twins), `reroll_twins.py check`.
#   REAL, from the P6 tree at the repo tip:  online_labeller.py, p6_prep.py (lane-distinct runs on synthetic parts).
#   STUBS (clearly named, scenario-controlled via $MOCK/ctl/*):  reroll_twins.py run, diag_d7_twin_route.py verify, p6_pod.sh, nvidia-smi, numpy.
# usage (inside WSL):  bash build_mock_pod.sh <package dir as seen from WSL> <p6 refe dir as seen from WSL> [MOCK]
set -eu
PKG_SRC="$1"; P6REFE_SRC="$2"; MOCK="${3:-/tmp/p6mock}"
rm -rf "$MOCK"; mkdir -p "$MOCK"/{ctl,stubs/bin,stubs/numpy}
W="$MOCK/workspace"
mkdir -p "$W/ops/bank3c/code" "$W/ops/bank3c/full" "$W/ops/d7fix/pkg/refe" "$W/ops/d7fix/pkg/code" \
         "$W/data/refe_navtrain/train_grow_cmd" "$W/data/refe_navtrain/train_grow_cmd_3c" "$W/data/refe_navtrain/metric_cache" "$W/data/refe_navtrain/metric_cache_twins_3c" \
         "$W/data/refe_p6/stage" "$W/data/refe_p6/lg" "$W/refe-plan-p6/refe" "$W/data/refe_p2" "$W/data/rls1"
# ---- numpy stub (online_labeller imports numpy at module level and uses it only inside functions the rehearsal does not call)
printf 'class ndarray:\n    pass\nclass generic:\n    pass\nfloat64 = float\nfloat32 = float\n' > "$MOCK/stubs/numpy/__init__.py"   # real classes, so p6_prep.canon()'s isinstance(o, np.ndarray) is False for the synthetic list-valued parts
# ---- nvidia-smi stub: $MOCK/ctl/gpu = idle | busy | unknown
cat > "$MOCK/stubs/bin/nvidia-smi" <<EOF
#!/bin/bash
s=\$(cat $MOCK/ctl/gpu 2>/dev/null || echo idle)
case "\$s" in
  idle) exit 0 ;;
  busy) echo "4242, /workspace/venv-teacher/bin/python, 3384 MiB"; exit 0 ;;
  *) echo "NVIDIA-SMI has failed" >&2; exit 9 ;;
esac
EOF
chmod +x "$MOCK/stubs/bin/nvidia-smi"
# ---- the 3-C code dir: the REAL files, byte for byte, except reroll_twins.py (dispatcher: `check` real, `run` stub) and env_3c.sh (mock)
cp "$PKG_SRC"/code/*.py "$PKG_SRC"/code/*.sh "$W/ops/bank3c/code/"
mv "$W/ops/bank3c/code/reroll_twins.py" "$W/ops/bank3c/code/reroll_twins_real.py"
cat > "$W/ops/bank3c/code/reroll_twins.py" <<'EOF'
# MOCK dispatcher.  `check` -> the REAL reroll_twins.py.  `run` -> a STUB that stands in for the 4 s teacher re-roll (CPU NAVSIM); ctl/c2mode = pass | slow | drop
import json, os, runpy, sys
from pathlib import Path
MOCK = os.environ.get("P6MOCK", "/tmp/p6mock")
if len(sys.argv) > 1 and sys.argv[1] == "check":
    sys.argv[0] = str(Path(__file__).with_name("reroll_twins_real.py"))
    runpy.run_path(sys.argv[0], run_name="__main__")
    raise SystemExit(0)
a = sys.argv[2:]
def opt(n, d=None):
    return a[a.index(n) + 1] if n in a else d
bank, out = opt("--bank"), opt("--out")
i, N = [int(x) for x in opt("--log-shard", "0/1").split("/")]
mode = Path(MOCK, "ctl", "c2mode").read_text().strip() if Path(MOCK, "ctl", "c2mode").exists() else "pass"
kept = {}
for ln in open(f"{MOCK}/workspace/data/refe_navtrain/train_grow_cmd_3c/targets_rank1.jsonl", encoding="utf-8"):
    if '"lane_rank"' in ln:
        r = json.loads(ln)
        if r.get("aug", {}).get("kind") == "lane_rank":
            kept[(r["log_name"], r["token"], int(r["aug"]["value"]))] = r
rows = [json.loads(l) for l in open(Path(bank, "targets_rank1.jsonl"), encoding="utf-8") if l.strip()]
Path(out).mkdir(parents=True, exist_ok=True)
n = 0
with open(Path(out, "targets_reroll.jsonl"), "a" if "--resume" in a else "w", encoding="utf-8") as fo:
    for j, r in enumerate(rows):
        if j % N != i:
            continue
        if mode == "drop" and j % 3 == 0:
            continue                                                         # a shard that silently loses a third of its twins
        k = (r["log_name"], r["token"], int(r["aug"]["value"]))
        t = [list(p) for p in kept[k]["traj"]]
        if mode == "slow":
            t = [[p[0] * 1.5, p[1] * 1.5] + list(p[2:]) for p in t]           # the 4 s held-12 re-roll goes 1.5x as far as the kept twin => kept/base = 0.67 < 0.95
        r2 = dict(r); r2["traj"] = t; fo.write(json.dumps(r2) + "\n"); n += 1
print(f"MOCK reroll_twins run shard {i}/{N}: {n} rows mode={mode}")
EOF
# ---- env_3c.sh (mock of /workspace/ops/bank3c/code/env_3c.sh)
cat > "$W/ops/bank3c/code/env_3c.sh" <<EOF
#!/bin/bash
export P6MOCK=$MOCK
export PKG=$W/ops/d7fix/pkg
export REFE_SIM_HZ=10 NUPLAN_DATA_ROOT=$W/data REFE_NUPLAN_DB_ROOT=$W/data/navtrain_dbs PYTHONUNBUFFERED=1
export NUPLAN_MAPS_ROOT=$W/data/nuplan-maps
export PY=python3
export CODE3C=$W/ops/bank3c/code
export RANK0=$W/data/refe_navtrain/train_grow_cmd/targets_rank0.jsonl
export PYTHONPATH=$MOCK/stubs\${PYTHONPATH:+:\$PYTHONPATH}
export PATH=$MOCK/stubs/bin:\$PATH
run3c() { env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "\$@"; }
EOF
# ---- data: the REAL mini banks
IN="$W/data/refe_navtrain/train_grow_cmd"; OUT="$W/data/refe_navtrain/train_grow_cmd_3c"
cp "$PKG_SRC/results/controls/pod/bank_i3/targets_rank1.jsonl" "$IN/targets_rank1.jsonl"
for f in targets_rank1.jsonl merge_report.json audit_all.jsonl; do cp "$PKG_SRC/results/controls/pod/merged/$f" "$OUT/$f"; done
printf '{"rank":0,"note":"mock rank-0 row"}\n' > "$IN/targets_rank0.jsonl"; cp "$IN/targets_rank0.jsonl" "$OUT/targets_rank0.jsonl"
# ---- 3-C run markers
F="$W/ops/bank3c/full"
echo "ZZ3C_RUN_DONE shards=720" > "$F/RUN_DONE"; : > "$F/PARTS_DONE"
echo '{"now_utc": "mock", "audited": 44409, "total": 44409, "shards_seen": 720, "shards_done": 720}' > "$F/progress.log"
echo "ZZ3C_MERGE_OK lines_in=224 lines_out=173 kept=23 dropped=51 yield_guard=HOLDS -> $OUT" > "$F/merge.out"
# ---- synthetic parts in the REAL format (online_labeller.PARTS_FORMAT), distinct twin/logged centrelines for every twin row of the input bank
python3 - "$IN" "$W/data/refe_navtrain" <<'EOF'
import json, lzma, pickle, sys
from pathlib import Path
inp, nav = Path(sys.argv[1]), Path(sys.argv[2])
def put(p, tag):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(lzma.compress(pickle.dumps({"format": "tanitad.p6parts/1", "parts": (None, None, None, "CENTERLINE_" + tag, ["lane_" + tag], None)}, protocol=4), preset=1))
n = 0
for ln in open(inp / "targets_rank1.jsonl", encoding="utf-8"):
    r = json.loads(ln); a = r.get("aug")
    put(nav / "metric_cache" / r["log_name"] / f"{r['token']}.pkl.xz", "LOGGED_" + r["token"])
    if isinstance(a, dict) and a.get("kind") == "lane_rank":
        put(nav / "metric_cache_twins_3c" / r["log_name"] / f"{r['token']}__lane_rank_{a['value']}.pkl.xz", f"TWIN{a['value']}_" + r["token"]); n += 1
print("synthetic parts: twin parts", n)
EOF
# ---- the P6 tree: the REAL online_labeller.py + p6_prep.py; a MOCK p6_pod.sh; a stub build_metric_cache.py WITHOUT the route-rank fix (like the real shipped tree)
cp "$P6REFE_SRC/online_labeller.py" "$P6REFE_SRC/p6_prep.py" "$W/refe-plan-p6/refe/"
echo "# stub: the shipped tree's build_metric_cache.py has no PDM-side route-rank fix" > "$W/refe-plan-p6/refe/build_metric_cache.py"
echo '{"commit": "1111111111111111111111111111111111111111", "created_utc": "mock", "tree_sha256": "2222222222222222222222222222222222222222222222222222222222222222", "files": {}}' > "$W/refe-plan-p6/SHIP_MANIFEST.json"
cat > "$W/refe-plan-p6/refe/p6_pod.sh" <<'EOF'
#!/bin/bash
# MOCK of refe/p6_pod.sh: a simulator of the chain's launch-gate dry run.  It carries the text markers the orchestrator greps for (the real tree has them after the knob patch):
#   CBANK="${P6_CBANK:-$DATA/refe_navtrain/train_grow_cmd}"   P6_CBANK is set   expandable_segments:True   stop_after launch_gate
# ctl/p6mode = pass | refuse | fallback | wrongbank | rebuild | nopass | train
set -u
MOCK=${P6MOCK:-/tmp/p6mock}; W=$MOCK/workspace; DATA=$W/data; P6=$DATA/refe_p6; L=$P6/chain.log; MC=$DATA/refe_navtrain/metric_cache
CBANK="${P6_CBANK:-$DATA/refe_navtrain/train_grow_cmd}"
mode=$(cat $MOCK/ctl/p6mode 2>/dev/null || echo pass)
log() { echo "$(date -u +%FT%T) $*" >> $L; }
exec 200>"$P6/chain.lock"; flock -n 200 || { echo "ZZP6_REFUSED lock"; exit 1; }
log "chain start (mock)"
[ -n "${HF_TOKEN:-}" ] || { log "REFUSED: HF_TOKEN is not set in the environment"; exit 1; }
if [ -z "${P6_STOP_AFTER:-}" ]; then
  # ---- the REAL launch (no stop point).  ctl/p6launch = ok | refuse | nostep | exit | stepnopre | gone | fail_token
  lm=$(cat $MOCK/ctl/p6launch 2>/dev/null || echo ok); RUN=$DATA/refe_runs/p6_paper_faithful; mkdir -p $RUN
  [ "${P6_D7_FALLBACK:-0}" = 0 ] || { log "REFUSED: mock launch with the fallback on"; exit 1; }
  [ "${P6_CBANK:-}" = "$DATA/refe_navtrain/train_grow_cmd_3c" ] || { log "REFUSED: mock launch without P6_CBANK pointing at the 3-C bank"; exit 1; }
  sleep 1
  case "$lm" in
    refuse)    log "REFUSED: the launch-gate token does not verify for this argv (attempt 1): train.py is NOT started"; exit 1 ;;
    gone)      exit 0 ;;
    stepnopre) printf '{"step": 0, "epoch": 0, "pos": 200, "traj_L1": 12.4, "score": 0.05}\n' >> $RUN/metrics.jsonl; sleep 60; exit 0 ;;
    fail_token) log "ZZEGRESS_FAIL p2r_tick_3 exception:mock"; log "train.py preflight OK; launching"; sleep 60; exit 0 ;;
  esac
  log "train.py preflight OK; launching"
  case "$lm" in
    nostep) sleep 90 ;;
    exit)   log "attempt 1 exited rc 1"; sleep 90 ;;
    ok)     printf '{"event": "start", "at": "mock"}\n' >> $RUN/metrics.jsonl; sleep 2
            printf '{"step": 0, "epoch": 0, "pos": 200, "traj_L1": 12.487878799438477, "score": 0.0573, "winners": 3, "batch": 4}\n' >> $RUN/metrics.jsonl; sleep 60 ;;
  esac
  exit 0
fi
sleep 1
rm -f "$P6/stage/lanes.ok"
for k in 1 2 3; do
  if grep -q "\"value\": $k" $CBANK/targets_rank1.jsonl 2>/dev/null; then
    out=$(cd $W/refe-plan-p6/refe && PYTHONPATH=$MOCK/stubs python3 p6_prep.py lane-distinct --bank $CBANK --cache $MC --k $k --n 40 --min-frac 0.9 2>&1 | tail -n 1)
    echo "$out" > $P6/lanes_distinct_$k.out
    if echo "$out" | grep -q "^ZZD7_DISTINCT_OK "; then log "D7 rank $k: $out"
    elif [ "${P6_D7_FALLBACK:-0}" = 1 ]; then log "D7 FALLBACK (P6_D7_FALLBACK=1): PDM-Closed did not honour the route patch for rank $k"
    else log "REFUSED: D7: rank-$k twins are NOT on their own route"; exit 1; fi
  fi
done
[ "$mode" = fallback ] && log "D7 FALLBACK (mock mode)"
[ "$mode" = rebuild ] && { d=$(ls -d $MC/*/ | head -n 1); echo x > "${d}rebuilt_in_dry_run__lane_rank_1.pkl.xz"; }
log "lanes ok"
log "ARGS sha deadbeefdeadbeef (--yaw-wrap off, --label-workers 7)"
[ "$mode" = refuse ] && { log "REFUSED: the launch gate did not mint a PASS token that verifies for this argv (launch_gate.log, lg/, token_verify.log)"; exit 1; }
[ "$mode" = nopass ] && { log "launch gate PASS (token verified)"; log "stopping after launch_gate (P6_STOP_AFTER)"; exit 0; }
tgt="$CBANK"; [ "$mode" = wrongbank ] && tgt=$DATA/refe_navtrain/train_grow_cmd
python3 - "$P6/lg" "$tgt" "$MC" "$CBANK" <<'PY'
import json, os, sys, time
lg, tgt, mc, cb = sys.argv[1:5]
os.makedirs(lg, exist_ok=True)
nf = sum(len(f) for _, _, f in os.walk(cb)); nm = sum(1 for d in os.listdir(mc) for f in os.listdir(os.path.join(mc, d)) if f.endswith(".pkl.xz"))
argv = ["--backbone", "vitl16", "--targets", tgt, "--metric-cache", mc, "--yaw-wrap", "off", "--nav-mode", "command", "--labeller", "online", "--epochs", "25"]
tok = {"binding": {"argv": argv, "argv_sha256": "a" * 64, "commit": "1111111111111111111111111111111111111111", "tree_sha256": "b" * 64, "profile": "refe_p6",
        "data": {"--targets": {"kind": "dir", "path": tgt, "n_files": nf}, "--metric-cache": {"kind": "dir", "path": mc, "n_files": nm}}},
       "checks": {"G-HYG": {"status": "PASS"}, "G-LIVE": {"status": "PASS"}}, "kind": "launch-gate-token", "reasons": [], "schema": "tanitad.launch_gate/1", "verdict": "PASS"}
json.dump(tok, open(os.path.join(lg, "PASS_mock0001.json"), "w"))
PY
log "launch gate PASS (token verified)"
[ "$mode" = train ] && log "train.py preflight OK; launching"
touch "$P6/stage/launch_gate.ok" "$P6/stage/lanes.ok"
log "stopping after launch_gate (P6_STOP_AFTER)"
exit 0
EOF
# a mock R0d launcher: a live process whose argv contains rls1_pod.sh (the pod-free check scans /proc for it)
cat > "$W/refe-plan-p6/refe/rls1_pod.sh" <<'EOF'
#!/bin/bash
echo $$ > /tmp/p6mock/rls1.pid
sleep ${1:-120}
EOF
echo "P6_DD_GB=16" > "$W/ops/p6_start.env"
touch "$W/data/refe_p6/stage/commands.ok" "$W/data/refe_p6/stage/cache.ok" "$W/data/refe_p6/stage/lanes.ok"
echo "mock-hf-token-never-used" > "$MOCK/hf_token_mock"
# ---- diag_d7_twin_route.py stub (the real one needs NAVSIM): ctl/c3mode = pass | bfail | nosummary
cat > "$W/ops/d7fix/pkg/refe/diag_d7_twin_route.py" <<'EOF'
# MOCK of diag_d7_twin_route.py verify: prints the instrument's own marker lines (c fails, as it does for every real bank: unsatisfiable for this teacher), rc 1 like the real one
import json, os, sys
a = sys.argv[1:]
k = int(a[a.index("--k") + 1]); out = a[a.index("--out") + 1]
mode = open(os.environ.get("P6MOCK", "/tmp/p6mock") + "/ctl/c3mode").read().strip() if os.path.exists(os.environ.get("P6MOCK", "/tmp/p6mock") + "/ctl/c3mode") else "pass"
if mode == "nosummary":
    print("Traceback (most recent call last): mock crash"); sys.exit(1)
s = {"k": k, "n_requested": 17, "n_checked": 17, "n_error": 0, "harness_reproduces_logged_centerline_frac": 1.0,
     "all": {"digest_differs": 1.0, "spec_ok": 0.94 if (mode == "bfail" and k == 2) else 1.0, "traj_closer_to_twin_div_ge_0_5": 0.41, "n_div_ge_0_5": 15},
     "a_ok": True, "b_ok": not (mode == "bfail" and k == 2), "c_ok": False}
open(out, "w").write(json.dumps({"summary": s, "twins": []}))
print("ZZD7V_SUMMARY " + json.dumps(s)); print(f"ZZD7V_FAIL a={s['a_ok']} b={s['b_ok']} c={s['c_ok']}"); sys.exit(1)
EOF
echo pass > "$MOCK/ctl/c2mode"; echo pass > "$MOCK/ctl/c3mode"; echo pass > "$MOCK/ctl/p6mode"; echo idle > "$MOCK/ctl/gpu"; echo ok > "$MOCK/ctl/p6launch"
echo "MOCK POD READY: $MOCK"; find "$W" -type f | wc -l
