#!/usr/bin/env python3
"""SO-PICKCON finalizer: run ONCE when both test lanes are done (the watcher does it).  Reads base_summary2.json + patched_summary.json, compares them test by test, copies the TESTED files
(from patched_full) and the logs under out/ mirroring repo paths, writes TEST_RESULTS.md + test_results.json + BASE_BLOBS.json + PATCH_*.diff + paths.json, and fills the <!--TESTS--> block of RESULT.md.
Never commits, never pushes."""
import difflib, hashlib, json, os, shutil, subprocess, sys

ROOT = "C:/Users/Admin/pickcon"
PKREL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
OUT = f"{ROOT}/out"
OUTPK = f"{OUT}/{PKREL}/raw/2026-10-18-so-pickcon"
CHANGED = [f"{PKREL}/refe/{n}" for n in ("scorer_finetune.py", "gate_checks.py", "launch_gate.py", "test_rl_s1b.py", "test_rl_safeobj.py", "test_launch_gate.py", "test_rl_pickcon.py")]
SRC_DIFFS = [f"{PKREL}/refe/{n}" for n in ("scorer_finetune.py", "gate_checks.py", "launch_gate.py")]
MIRROR = "C:/Users/Admin/tanitad-push"
TIP = "0a5958a"


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def blob_of(path):
    b = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def main():
    bs, ps = json.load(open(f"{ROOT}/base_summary2.json")), json.load(open(f"{ROOT}/patched_summary.json"))
    done = bool(bs.get("_done")) and bool(ps.get("_done"))
    rows, regress, same_fail = [], [], []
    for name in sorted(set(bs) | set(ps)):
        if name == "_done":
            continue
        b, p = bs.get(name), ps.get(name)
        tb = (b["tokens"][-1] if b and b["tokens"] else (b["tail"][-1][:90] if b and b["tail"] else "(not run)"))
        tp = (p["tokens"][-1] if p and p["tokens"] else (p["tail"][-1][:90] if p and p["tail"] else "(not run)"))
        if b and p and b["rc"] == 0 and p["rc"] != 0:
            v = "REGRESSION"
            regress.append(name)
        elif b and p and b["rc"] != 0 and p["rc"] != 0:
            v = "FAILS AT BASELINE TOO (environmental)" if (b["tail"][-1:] == p["tail"][-1:]) else "FAILS AT BASELINE AND PATCHED, DIFFERENT TAIL"
            same_fail.append(name)
        elif b and p and b["rc"] == 0 and p["rc"] == 0:
            v = "PASS = PASS"
        elif p and not b:
            v = "NEW TEST" if p["rc"] == 0 else "NEW TEST FAILS"
            if p["rc"] != 0:
                regress.append(name)
        else:
            v = "baseline fails, patched passes" if (p and p["rc"] == 0) else "incomplete"
        rows.append((name, b["rc"] if b else None, tb, p["rc"] if p else None, tp, p["secs"] if p else None, v))
    md = ["| test file (run in full form) | baseline rc | baseline token | patched rc | patched token | patched s | verdict |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | **{r[6]}** |")
    head = f"Lanes complete: {done}. Regressions: {regress or 'none'}. Failing at baseline too: {same_fail or 'none'}."
    table = "\n".join(md)
    os.makedirs(OUTPK, exist_ok=True)
    open(f"{OUTPK}/TEST_RESULTS.md", "w", encoding="utf-8").write(f"# SO-PICKCON test results (baseline = pristine 0a5958a export, patched = the files below)\n\n{head}\n\n{table}\n")
    json.dump({"done": done, "regressions": regress, "same_fail": same_fail, "rows": rows}, open(f"{OUTPK}/test_results.json", "w"), indent=1)
    # the TESTED files, mirrored
    blobs = {}
    for rel in CHANGED:
        src = f"{ROOT}/patched_full/{rel}"
        dst = f"{OUT}/{rel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        base = sh("git", "-C", MIRROR, "rev-parse", f"{TIP}:{rel}")
        blobs[rel] = {"base_blob_at_0a5958a": base if len(base) == 40 else "(new file)", "new_blob": blob_of(dst)}
        if len(base) == 40 and rel in SRC_DIFFS:
            old = sh("git", "-C", MIRROR, "-c", "core.autocrlf=false", "show", f"{TIP}:{rel}").splitlines()
            new = open(dst, encoding="utf-8", newline="").read().splitlines()
            open(f"{OUTPK}/PATCH_{os.path.basename(rel)}.diff", "w", encoding="utf-8", newline="\n").write("\n".join(difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}", lineterm="", n=2)) + "\n")
    json.dump(blobs, open(f"{OUTPK}/BASE_BLOBS.json", "w"), indent=1)
    # logs
    for lane, d in (("baseline", f"{ROOT}/base_logs2"), ("patched", f"{ROOT}/patched_logs")):
        if os.path.isdir(d):
            for f in os.listdir(d):
                os.makedirs(f"{OUTPK}/logs/tests_{lane}", exist_ok=True)
                shutil.copyfile(f"{d}/{f}", f"{OUTPK}/logs/tests_{lane}/{f}")
    for f in ("lane_runner.py", "gen_pickcon_reversals.py", "finalize.py"):
        shutil.copyfile(f"{ROOT}/{f}", f"{OUTPK}/{f}")
    # RESULT.md block
    rp = f"{OUTPK}/RESULT.md"
    if os.path.exists(rp):
        t = open(rp, encoding="utf-8").read()
        a, b = t.find("<!--TESTS-->"), t.find("<!--/TESTS-->")
        if a >= 0 and b > a:
            t = t[:a] + "<!--TESTS-->\n" + head + "\n\n" + table + "\n" + t[b:]
        mp = f"{OUTPK}/pickcon_measure.json"
        a, b = t.find("<!--MEASURE-->"), t.find("<!--/MEASURE-->")
        if os.path.exists(mp) and a >= 0 and b > a:
            m = json.load(open(mp, encoding="utf-8"))
            ra, rb, rc = m["a"], m["b"], m["c"]
            lines = ["(MEASURED by `so_pickcon_measure.py` -> `pickcon_measure.json`, the test's own closures; tiny synthetic scorer, a wiring magnitude, not a driving result)", "",
                     f"* (a) max |constraint grad| on arm B: {ra['max_abs_constraint_grad_T0.1']:.6g} at T 0.1, {ra['max_abs_constraint_grad_T0.02']:.6g} at T 0.02; max |diff| between them {ra['max_abs_diff_of_constraint_grads_0.1_vs_0.02']:.6g} (relative to the T 0.1 max: {ra['relative_diff_to_T0.1_max']:.4g}); arm A's gradient differs by {ra['arm_A_grad_max_abs_diff_0.1_vs_0.02']:.6g}; logged cB_ttc {ra['cB_ttc_T0.1']:.6g} -> {ra['cB_ttc_T0.02']:.6g}; twin (same T twice) max |diff| {ra['twin_same_T_twice_max_abs_diff']:.6g}.",
                     f"* (b) default vs explicit 0.1: max |diff| of total gradient B {rb['default_vs_explicit_0.1_max_abs_diff_of_total_gradient_B']:.6g}, constraint gradient {rb['default_vs_explicit_0.1_max_abs_diff_of_constraint_gradient']:.6g}, gradient A {rb['default_vs_explicit_0.1_max_abs_diff_of_gradient_A']:.6g}; loss bits equal {rb['loss_bits_equal']}, logged parts equal {rb['parts_equal']}; effective T {rb['default_effective_T']} vs {rb['explicit_T']}; zero control (lambda 0) max |constraint grad| {rb['zero_control_lambda0_max_abs_constraint_grad']:.6g}; inheritance at --epo-temp 0.05: effective T {rb['default_inherits_epo_temp_0.05']['effective_T']}, diff vs explicit 0.05 = {rb['default_inherits_epo_temp_0.05']['diff_vs_explicit_0.05']:.6g}, vs explicit 0.1 = {rb['default_inherits_epo_temp_0.05']['diff_vs_explicit_0.1']:.6g}.",
                     "* (c) hard-pick exactness:"]
            for k, v in rc.items():
                lines.append(f"  * {k}: |policy - one-hot| max {v['policy_minus_onehot_max_abs']:.6g}; cA_ttc {v['cA_ttc']:.6g} (pick label mean {v['pickA_mean_ttc_label']:.6g}); cB_ttc {v['cB_ttc']:.6g} (pick label mean {v['pickB_mean_ttc_label']:.6g}); g_ttc {v['g_ttc']:.6g}, equals the hard-pick gap exactly: {v['g_minus_hard_pick_gap_exact']}; g + p1gap (the wedge) {v['wedge_g_plus_p1gap']:.6g}")
            t = t[:a] + "<!--MEASURE-->\n" + "\n".join(lines) + "\n" + t[b:]
        open(rp, "w", encoding="utf-8").write(t)
    # paths.json
    paths = []
    for dp, dn, fn in os.walk(OUT):
        for f in sorted(fn):
            if f == "paths.json":
                continue
            ab = os.path.join(dp, f).replace("\\", "/")
            paths.append([os.path.relpath(ab, OUT).replace("\\", "/"), ab, "raw"])
    json.dump(sorted(paths), open(f"{OUT}/paths.json", "w"), indent=1)
    print(head)
    print(table)
    print("paths.json entries:", len(paths))
    return 0 if (done and not regress) else 1


if __name__ == "__main__":
    sys.exit(main())
