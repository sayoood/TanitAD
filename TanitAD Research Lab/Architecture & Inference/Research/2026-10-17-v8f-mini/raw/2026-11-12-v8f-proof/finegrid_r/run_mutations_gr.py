"""run_mutations_gr.py -- mutation list of the reader and the pod script of card ADD43 (numpy only). Each mutation is a text edit of a temp copy (the old text must occur exactly once); the suite
test_finegrid_r_read.py is run against the copy (FINEGRID_R_READ_FILE / POD_R_SCRIPT) and must turn RED; the unmutated files are run first and last and must be GREEN. Result: MUTATIONS_GR.json.
The mutations of tok_probe_gr.py need torch and are covered by run_checks_pod.py (its g4 check carries the loss-line mutation); they are not in this list.
Run (one Python job): python F:/TanitAD-scratch/v8proof/ops/capmem.py 1000 run_mutations_gr.py
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MUT = os.path.join(HERE, "work", "mut")
READ = os.path.join(HERE, "finegrid_r_read.py")
SCRIPT = os.path.join(HERE, "pod_finegrid_r_run.sh")
PT = os.path.join(HERE, "work", "pt_mut")

MUTATIONS = [
    ("episodes_only_seeds_held_fixed", "read", "    idx = rng.integers(0, n, size=(B, n))                              # SEED-LEVEL: the seed indices with replacement\n",
     "    idx = np.tile(np.arange(n), (B, 1))\n", "not pod_script"),
    ("pair_sign_swapped", "read", 'PAIRS = {"G16": ("F16", "TK16"), "G16w": ("F16", "TK64")}', 'PAIRS = {"G16": ("TK16", "F16"), "G16w": ("TK64", "F16")}', "not pod_script"),
    ("seed_index_misaligned", "read", "        p, d = boot_diff(ctx, measure, outs[arm(a_t, k)], outs[arm(b_t, k)])", "        p, d = boot_diff(ctx, measure, outs[arm(a_t, k)], outs[arm(b_t, 2)])", "not pod_script"),
    ("interval_touching_zero_separates", "read", '    return bool(t["mean"] > 0 and t["ci95"][0] > 0 and t["n_pos"] >= MIN_POS_SEEDS)', '    return bool(t["mean"] > 0 and t["ci95"][0] >= 0 and t["n_pos"] >= MIN_POS_SEEDS)', "not pod_script"),
    ("six_of_eight_clause_lowered_to_four", "read", "MIN_POS_SEEDS = 6", "MIN_POS_SEEDS = 4", "not pod_script"),
    ("mae_clause_dropped", "read", "MAE_FRACTION = 0.10", "MAE_FRACTION = 0.0", "not pod_script"),
    ("p5_of_permuted_arm_reads_the_permuted_part_too", "read", '    keys = ("lead_sup_bce_mean", "lead_sup_gap_mean")', '    keys = ("lead_sup_bce_mean", "lead_sup_gap_mean", "lead_sup_clos_mean")', "not pod_script"),
    ("r1_compares_nothing", "read", '    same = bool(a["idx_hashes"] == b["idx_hashes"] and a["full_hashes"] == b["full_hashes"] and len(a["idx_hashes"]) > 0 and a["n_batches"] == b["n_batches"])', "    same = True", "not pod_script"),
    ("tool_sha_not_compared", "read", '    return {"pass": bool(want and got and want == got),', '    return {"pass": True,', "not pod_script"),
    ("tokens_seen_in_training_not_checked", "read", ' and r.get("n_tok_seen_in_training") == ntok and r.get("D") == ntok * W', ' and r.get("D") == ntok * W', "not pod_script"),
    ("pod_slot_wait_removed_before_gpu_steps", "script", 'gpu() { local step=$1; shift; wait_slot "$step"; run "$@"; }', 'gpu() { local step=$1; shift; run "$@"; }', "pod_script"),
    ("pod_chunk_flag_dropped_from_process_A", "script", '--arms "$ARMS_A" --sampler-chunk-like pool8x20 "${EXP[@]}"', '--arms "$ARMS_A" "${EXP[@]}"', "pod_script"),
    ("pod_noclobber_marker_replaced_by_plain_write", "script", "( set -o noclobber; printf", "( printf", "pod_script"),
]


def pytest(env_extra, k):
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.join(HERE, "test_finegrid_r_read.py"), "--basetemp", PT, "-x", "-k", k]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=1800)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def main():
    os.makedirs(MUT, exist_ok=True)
    res = {"card": "PREREG_V8F_PROOF_ADD43", "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mutations": []}
    rc0, last0 = pytest({}, "not zz_none")
    res["suite_before"] = {"green": rc0 == 0, "last_line": last0}
    for name, which, old, new, k in MUTATIONS:
        src = READ if which == "read" else SCRIPT
        txt = open(src, encoding="utf-8", newline="").read()
        if txt.count(old) != 1:
            res["mutations"].append({"name": name, "error": "old text occurs %d times" % txt.count(old)})
            continue
        dst = os.path.join(MUT, name + (".py" if which == "read" else ".sh"))
        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt.replace(old, new))
        env = {"FINEGRID_R_READ_FILE": dst} if which == "read" else {"POD_R_SCRIPT": dst}
        if which == "read":
            env["FINEGRID_BASE_DIR"] = os.path.normpath(os.path.join(HERE, "..", "finegrid"))
        rc, last = pytest(env, k)
        res["mutations"].append({"name": name, "target": os.path.basename(src), "red": rc != 0, "pytest_rc": rc, "last_line": last})
        print(name, "RED" if rc != 0 else "STILL GREEN (a hole)", flush=True)
    rc1, last1 = pytest({}, "not zz_none")
    res["suite_after"] = {"green": rc1 == 0, "last_line": last1}
    res["n_mutations"] = len(res["mutations"])
    res["n_red"] = sum(1 for m in res["mutations"] if m.get("red"))
    with open(os.path.join(HERE, "MUTATIONS_GR.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print("ZZ MUTATIONS-GR DONE %d of %d red; suite before %s after %s" % (res["n_red"], res["n_mutations"], res["suite_before"]["green"], res["suite_after"]["green"]))
    return 0 if res["n_red"] == res["n_mutations"] and res["suite_before"]["green"] and res["suite_after"]["green"] else 1


if __name__ == "__main__":
    sys.exit(main())
