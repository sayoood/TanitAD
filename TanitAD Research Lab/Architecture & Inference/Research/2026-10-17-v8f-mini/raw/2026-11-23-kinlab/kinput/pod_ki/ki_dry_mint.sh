#!/bin/bash
# ki_dry_mint.sh -- the BUILD AGENT's CPU dry mint of the KI kit (K-INPUT), run ON THE POD from /root/rollhist/kit_ki/. Writes ONLY under /root/rollhist/kit_ki/dry/ (the results, and the scratch root $ROOT = kit_ki/dry/root, deleted at the end).
# Touches no GPU, no lock (/root/gpu_queue.lock), no marker (/root/N1DW_READY), no process it did not start. Everything runs at nice 19, OMP/MKL 2 threads, CUDA hidden. One selfcheck at a time.
# Guards: the container's total_rss (cgroup, never free) must be < 40 GB before each start (waits up to 30 min, then stops and reports); a selfcheck child is killed BY PID if its RSS passes DRY_MAX_RSS_GB (default 14), it runs longer than
# DRY_MAX_MIN minutes (default 75), or the container reaches 40 GB; the run is then reported, not retried.
# Per seed K (0, 1), with the REAL tree_ki (4097a89e), the REAL banks (links), the REAL init checkpoints:
#   A  the KI argv with `--device cpu` the ONLY difference from the launch argv (ki_argv.py render into the scratch root, then the cpu form); the trainer's --selfcheck-only on it (rh_selfcheck.py): selfcheck.json -> dry/s<K>_selfcheck_cpu_logged.json
#   B  the SAME argv minus the two trailing tokens (`--dagger-base logged`): the trainer's default `own` on the SAME probe batch -> dry/s<K>_selfcheck_cpu_own.json (card section 4: the share of selected rows whose own base differs from the logged base)
#   C  ki_gate_record.py mint + verify on the logged selfcheck (scratch root: the real G-SUITE result, the real pilot / bank check, a PLANTED tree marker, the real stock pair check) and the study-side DRY mint (chk_arm_ki.sh with DRY_GD / DRY_SC): checks.json
# dry/DRY_VALUES.json = per seed the card section 4 values: selected rows (n_selected_at_p1, n_eligible, n_rows) under `logged` and under `own`, rows_changed.base under `own`, the share = rows_changed.base / n_selected_at_p1 under `own`, the exemptions.
# Last line of $DR/dry.log: ZZ KI DRY MINT DONE ZZ or ZZ KI DRY MINT STOP ZZ <why>.
set -u
K=$(cd "$(dirname "$0")" && pwd); DR=$K/dry; ROOT=$DR/root
[ "$K" = /root/rollhist/kit_ki ] || { echo "run from /root/rollhist/kit_ki (got $K)"; exit 1; }
PY=/root/venv/bin/python
LOG=$DR/dry.log
MAXMIN=${DRY_MAX_MIN:-75}; MAXRSS_KB=$(( ${DRY_MAX_RSS_GB:-14} * 1048576 ))
rss() { awk '/^total_rss /{printf "%d", $2}' /sys/fs/cgroup/memory/memory.stat; }
[ ! -e $ROOT ] || { echo "$ROOT exists; refusing (delete it by literal path first)"; exit 1; }
mkdir -p $DR $K/prep_logs
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
stop() { say "ZZ KI DRY MINT STOP ZZ $1"; exit 1; }
[ -d /root/rollhist/tree_ki ] || stop "no tree_ki: the tree prep runs first"
[ -s /root/rollhist/gsuite_result_ki.json ] || stop "no gsuite_result_ki.json: the G-SUITE runs first"
say "start; container total_rss $(( $(rss) / 1048576 )) MB; caps: ${MAXMIN} min, $((MAXRSS_KB/1048576)) GB per selfcheck"
# ---- stage the scratch root (copies of kit files, copies of shared pod files, links to read-only inputs)
mkdir -p $ROOT/runs $ROOT/study $ROOT/pilot $ROOT/done_ki $ROOT/logs
for f in ki_env.sh ki_gate_record.py chk_arm_ki.sh ki_argv.py all_blobs_ki.txt dscale10k_check_ki.py; do cp $K/$f $ROOT/$f || stop "kit file $f missing in $K"; done
cp $K/dscale10k_check_ki.py $ROOT/study/
for f in rh_env.sh rh_evidence.py rh_selfcheck.py blobcheck.py; do cp /root/rollhist/$f $ROOT/$f || stop "no /root/rollhist/$f"; done
cp /root/rollhist/study/dscale10k_merge.py $ROOT/study/; cp /root/rollhist/pilot/PILOT_RH.json $ROOT/pilot/; cp /root/rollhist/gsuite_result_ki.json $ROOT/gsuite_result_ki.json
cp /root/rollhist/RH_READOUT.chosen $ROOT/RH_READOUT.chosen
ln -s /root/rollhist/tree_ki $ROOT/tree_ki; ln -s /root/rollhist/bank $ROOT/bank
for s in 0 1; do
  mkdir -p $ROOT/runs/RH_s$s $ROOT/runs/B0_s$s
  sed "s#/root/rollhist#$ROOT#g" /root/rollhist/runs/RH_s$s/argv.json > $ROOT/runs/RH_s$s/argv.json
  cp /root/rollhist/runs/B0_s$s/argv.json $ROOT/runs/B0_s$s/argv.json
done
date -u +%FT%TZ > $ROOT/done_ki/tree.done          # PLANTED (the scratch root's own; the chain writes the real one under the lock)
export RH_ROOT=$ROOT RH_READOUT=causal_g1x HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
. $ROOT/ki_env.sh
[ "$TREE" = $ROOT/tree_ki ] || stop "TREE is $TREE"
# ---- render (scratch root) and the argv check
$PY $ROOT/ki_argv.py render > $DR/render.log 2>&1 || stop "render: $(tail -c 300 $DR/render.log | tr '\n' ' ')"
say "render: $(tail -n 3 $DR/render.log | tr '\n' ' ' | cut -c1-300)"
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE"
selfcheck() {   # name argv_json outdir result_json
  local nm=$1 AV=$2 OD=$3 RS=$4
  local n=0; until [ "$(rss)" -lt 42949672960 ]; do n=$((n+1)); [ $n -gt 180 ] && stop "$nm: container total_rss >= 40 GB for 30 min before the start"; sleep 10; done
  say "$nm selfcheck start (CPU, nice 19); container total_rss $(( $(rss) / 1048576 )) MB"
  ( RH_TREE=$TREE exec nice -n 19 $PY $ROOT/rh_selfcheck.py $AV $OD $RS > $DR/$nm.log 2>&1 ) &
  local P=$! t0=$(date +%s) last=0
  while kill -0 $P 2>/dev/null; do
    sleep 10
    local el=$(( $(date +%s) - t0 )) pr cr
    pr=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); cr=$(rss)
    if [ $((el - last)) -ge 120 ]; then last=$el; say "$nm selfcheck ${el}s rss $((${pr:-0}/1024)) MB container $((cr/1048576)) MB: $(tail -n 1 $DR/$nm.log | cut -c1-140)"; fi
    if [ "${pr:-0}" -gt $MAXRSS_KB ] || [ $el -gt $((MAXMIN*60)) ] || [ "${cr:-0}" -ge 42949672960 ]; then
      kill -TERM $P 2>/dev/null; sleep 5; kill -KILL $P 2>/dev/null
      stop "$nm selfcheck stopped by the guard at ${el}s (rss ${pr:-0} KB, container ${cr} B); last log line: $(tail -n 1 $DR/$nm.log | cut -c1-200)"
    fi
  done
  wait $P; local rc=$?
  say "$nm selfcheck rc=$rc after $(( $(date +%s) - t0 )) s: $(tail -n 1 $DR/$nm.log | cut -c1-300)"
  [ -s $OD/selfcheck.json ] || stop "$nm selfcheck wrote no selfcheck.json"
}
for s in 0 1; do
  tag=KI_s$s; D=$ROOT/runs/$tag
  $PY $ROOT/ki_argv.py verify --seeds $s > $D/argv_verify.log 2>&1 || stop "argv verify $tag: $(tail -c 300 $D/argv_verify.log | tr '\n' ' ')"
  sha=$($PY -c "import hashlib;print(hashlib.sha256(open('$ROOT/bank/s$s/bank.json','rb').read()).hexdigest())")
  $PY - $D/argv.json $D/argv_nobase.json $D/argv_cpu.json $D/argv_cpu_own.json <<'PYE'
import json, sys
t = json.load(open(sys.argv[1]))["argv"]
i = t.index("--dagger-base")
assert t[i + 1] == "logged" and i + 2 == len(t)
json.dump({"argv": t[:i]}, open(sys.argv[2], "w"), indent=1)
c = list(t)
j = c.index("--device")
assert c[j + 1] == "cuda"
c[j + 1] = "cpu"                                       # the dry form's ONLY difference from the launch argv
json.dump({"argv": c}, open(sys.argv[3], "w"), indent=1)
json.dump({"argv": c[:i]}, open(sys.argv[4], "w"), indent=1)      # the same CPU argv without the base pair: the trainer's default `own`
PYE
  $PY $TREE/stack/scripts/rollhist_pair_check.py --rh $D/argv_nobase.json --b0 $ROOT/runs/B0_s$s/argv.json --bank-sha256 $sha > $D/pair_$tag.log 2>&1 || stop "pair check $tag: $(tail -c 300 $D/pair_$tag.log | tr '\n' ' ')"
  say "$tag pair check: $(tail -n 1 $D/pair_$tag.log)"
  selfcheck ${tag}_logged $D/argv_cpu.json $D/selfcheck $D/selfcheck.result.json
  cp $D/selfcheck/selfcheck.json $DR/s${s}_selfcheck_cpu_logged.json
  selfcheck ${tag}_own $D/argv_cpu_own.json $D/selfcheck_own $D/selfcheck_own.result.json
  cp $D/selfcheck_own/selfcheck.json $DR/s${s}_selfcheck_cpu_own.json
  cp $D/argv.json $DR/s${s}_argv_ki.json; cp $D/argv_cpu.json $DR/s${s}_argv_cpu.json; cp $D/argv_cpu_own.json $DR/s${s}_argv_cpu_own.json
  # ---- gate record (real selfcheck, real G-SUITE, planted tree marker) and the study-side DRY mint
  $PY $ROOT/ki_gate_record.py mint --tag $tag --readout causal_g1x > $D/mint.log 2>&1; mrc=$?
  $PY $ROOT/ki_gate_record.py verify --tag $tag > $D/verify.log 2>&1; vrc=$?
  say "$tag gate record mint rc=$mrc verify rc=$vrc: $(grep -E '^probe|ZZ|PROBLEM' $D/mint.log $D/verify.log | tr '\n' ' ' | cut -c1-500)"
  DRY_GD=$ROOT/dry_gate_s$s DRY_SC=$D/selfcheck/selfcheck.json nice -n 19 bash $ROOT/chk_arm_ki.sh mint $tag > $DR/mint_s$s.log 2>&1; crc=$?
  say "$tag study-side dry mint rc=$crc: $(tail -n 2 $DR/mint_s$s.log | cut -c1-300 | tr '\n' ' ')"
  mkdir -p $DR/$tag
  cp $D/selfcheck.log $D/selfcheck.result.json $D/pair_$tag.log $D/mint.log $D/verify.log $D/argv_verify.log $DR/$tag/ 2>/dev/null
  cp $D/gate/RH_GATE_RECORD.json $D/gate/RH_GATE_REFUSED.json $DR/$tag/ 2>/dev/null
  cp $ROOT/dry_gate_s$s/checks.json $ROOT/dry_gate_s$s/evidence.log $DR/$tag/ 2>/dev/null
  cp $DR/mint_s$s.log $DR/$tag/mint_token_dry.log
done
# ---- the card section 4 values
$PY - $DR <<'PYE'
import json, sys
DR = sys.argv[1]
out = {}
for s in (0, 1):
    lg = json.load(open(f"{DR}/s{s}_selfcheck_cpu_logged.json"))
    ow = json.load(open(f"{DR}/s{s}_selfcheck_cpu_own.json"))
    pl, po = lg["dagger_probe"], ow["dagger_probe"]
    n_own = po["n_selected_at_p1"]
    ch = po["rows_changed"]["base"]
    out[f"s{s}"] = {
        "logged": {"n_rows": pl["n_rows"], "n_eligible": pl["n_eligible"], "n_selected_at_p1": pl["n_selected_at_p1"], "base_source": pl.get("base_source"), "rows_changed": pl["rows_changed"],
                   "selected_rows_bit_identical_exempt": pl.get("selected_rows_bit_identical_exempt"), "base_rows_at_rest_exempt": pl.get("base_rows_at_rest_exempt"),
                   "grad_diff_all": pl["grad_diff_all"], "control_p0_grad_diff": pl["control_p0_grad_diff"], "declared_vs_built": lg.get("declared_vs_built")},
        "own": {"n_rows": po["n_rows"], "n_eligible": po["n_eligible"], "n_selected_at_p1": n_own, "rows_changed": po["rows_changed"],
                "selected_rows_bit_identical_exempt": po.get("selected_rows_bit_identical_exempt"), "base_rows_at_rest_exempt": po.get("base_rows_at_rest_exempt")},
        "same_probe_batch": (pl["n_rows"], pl["n_eligible"], pl["n_selected_at_p1"]) == (po["n_rows"], po["n_eligible"], po["n_selected_at_p1"]),
        "share_selected_rows_whose_own_base_differs": (ch / n_own) if n_own else None,
        "share_numerator_denominator": [ch, n_own],
    }
json.dump(out, open(f"{DR}/DRY_VALUES.json", "w"), indent=1)
print(json.dumps({k: (v["logged"]["n_selected_at_p1"], v["own"]["n_selected_at_p1"], v["share_numerator_denominator"], v["share_selected_rows_whose_own_base_differs"], v["same_probe_batch"]) for k, v in out.items()}))
PYE
cp $DR/dry.log $DR/dry.log.bak 2>/dev/null
rm -f $ROOT/tree_ki $ROOT/bank; rm -rf "${ROOT:?}"; echo "$(date -u +%FT%TZ) $ROOT (dry-mint scratch root: argv copies, selfcheck dirs, gate dirs) deleted by literal path; results kept in $DR" >> $K/prep_logs/deletions.log
rm -f $DR/dry.log.bak
say "ZZ KI DRY MINT DONE ZZ"
