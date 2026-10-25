#!/usr/bin/env python3
"""mutate_hacc.py -- MUTATE, do not inspect: each mutant is a one-or-two-line textual change to v8f_hacc.py, written to mutants/<name>/v8f_hacc.py, and test_v8f_hacc.py is run
against it (HACC_MODULE_DIR).  A mutant must go RED: pytest rc 1, >= 1 failed test, 0 collection errors, and every failure an ASSERTION (AssertionError / 'DID NOT RAISE'),
not an import error or a crash.  The unmutated module is run first and must be GREEN (the harness's own control).  Output: mutation_results.json + mutation.log (stdout).
Sequential, one pytest at a time (the 2-process cap)."""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SRC = open(f"{HERE}/v8f_hacc.py", encoding="utf-8", newline="").read().replace("\r\n", "\n")
MUTANTS = {
    "M1_drop_the_origin_in_D2": [("    for k in range(1, K):\n        full[k - 1, k - 1]", "    for k in range(2, K):\n        full[k - 1, k - 1]")],
    "M2_keep_the_isotonic_step": [("        w[:, a] = res.x\n", "        w[:, a] = H.pav_nondecreasing(res.x)\n")],
    "M3_select_lambda_on_eval_guards_removed": [("    if not isinstance(T, TrainBundle):\n        raise TypeError", "    if False:\n        raise TypeError"),
                                                ("    if T.split != \"train\":\n        raise ValueError(f\"the lambda selection refuses", "    if False:\n        raise ValueError(f\"the lambda selection refuses")],
    "M4_swap_axes": [("stack_system(r[:, :, a], d[:, :, a], lam, D2)", "stack_system(r[:, :, 1 - a], d[:, :, 1 - a], lam, D2)")],
    "M5_skip_the_KKT_check": [("        if not (k <= KKT_TOL):\n            raise KKTError", "        if False:\n            raise KKTError")],
    "M6a_invert_the_SMOOTHER_row": [("    if all(per_seed[s][\"accel_sep_neg\"] and per_seed[s][\"curv_sep_neg\"] for s in seeds):",
                                     "    if not all(per_seed[s][\"accel_sep_neg\"] and per_seed[s][\"curv_sep_neg\"] for s in seeds):")],
    "M6b_DOMINATES_needs_only_one_of_DOM_KEEP": [("    if all(per_seed[s][\"dom\"] and per_seed[s][\"keep\"] for s in seeds):", "    if all(per_seed[s][\"dom\"] or per_seed[s][\"keep\"] for s in seeds):")],
    "M6c_NO_SMOOTHING_reads_all_not_any": [("    if not any(per_seed[s][\"accel_sep_neg\"] for s in seeds):", "    if not all(per_seed[s][\"accel_sep_neg\"] for s in seeds):")],
    "M7_rule_bar_098_becomes_1": [("RULE_RATIO = 0.98\n", "RULE_RATIO = 1.0\n")],
    "M8_ADE_bar_not_strict": [("and r[\"ade2\"] < r[\"p_ade2\"] and r[\"ade6\"] < r[\"p_ade6\"])", "and r[\"ade2\"] <= r[\"p_ade2\"] and r[\"ade6\"] <= r[\"p_ade6\"])")],
    "M9_lambda_enters_unsquared": [("        s = np.sqrt(lam)\n", "        s = lam\n")],
    "M10_grid_changed": [("GRID_M = (0, 1, 4, 16, 64, 256)\n", "GRID_M = (0, 1, 4, 16, 64, 128)\n")],
    "M11_fallback2_is_not_lambda_zero": [("    return rows[0], \"fallback 2", "    return rows[-1], \"fallback 2")],
    "M12_lambda0_not_at_banked_w": [("    s1, s2 = terms_at(T.tau, T.P, T.y, T.banked_w)\n", "    s1, s2 = terms_at(T.tau, T.P, T.y, np.ones((NSTEP, NAX)))\n")],
    "M13_SMOOTHER_reads_accel_only": [("\"curv_sep_neg\": bool(c[\"better\"])}", "\"curv_sep_neg\": bool(a[\"better\"])}")],
}


def run(module_dir, tag):
    env = dict(os.environ, HACC_MODULE_DIR=module_dir, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    bt = f"C:/Users/Admin/pt_hacc/mut_{tag}"
    os.makedirs("C:/Users/Admin/pt_hacc", exist_ok=True)
    t0 = time.time()
    p = subprocess.run([PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", bt, "--tb=line", "-rf", f"{HERE}/test_v8f_hacc.py"],
                       capture_output=True, text=True, env=env, cwd=HERE)
    out = p.stdout + p.stderr
    failed = re.findall(r"^FAILED (\S+)", out, re.M)
    errors = re.findall(r"^ERROR (\S+)", out, re.M)
    lines = [ln for ln in out.splitlines() if re.search(r"\.py:\d+: ", ln)]
    def is_assertion(ln):
        body = re.split(r"\.py:\d+: ", ln, maxsplit=1)[-1].strip()
        return body.startswith("assert") or "AssertionError" in body or "DID NOT RAISE" in body or body.startswith("Failed:")
    assertion_only = bool(lines) and all(is_assertion(ln) for ln in lines)
    m = re.search(r"(\d+) passed", out)
    mf = re.search(r"(\d+) failed", out)
    return {"rc": p.returncode, "n_failed": int(mf.group(1)) if mf else 0, "n_passed": int(m.group(1)) if m else 0, "failed": failed, "errors": errors,
            "failure_lines": lines[:12], "all_failures_are_assertions": assertion_only, "seconds": round(time.time() - t0, 1), "tail": out[-600:]}


def main():
    res = {}
    base = run(HERE, "baseline")
    res["BASELINE_unmutated"] = base
    base["GREEN"] = bool(base["rc"] == 0 and base["n_failed"] == 0 and not base["errors"])
    print(f"BASELINE rc {base['rc']} passed {base['n_passed']} failed {base['n_failed']}  GREEN={base['GREEN']}  {base['seconds']}s", flush=True)
    if not base["GREEN"]:
        json.dump(res, open(f"{HERE}/mutation_results.json", "w"), indent=1)
        print("ZZHACC-MUT-STOP baseline not green", flush=True)
        return 2
    for name, edits in MUTANTS.items():
        s = SRC
        for old, new in edits:
            assert s.count(old) == 1, f"{name}: the snippet must occur exactly once, found {s.count(old)}: {old[:70]!r}"
            s = s.replace(old, new)
        d = f"{HERE}/mutants/{name}"
        os.makedirs(d, exist_ok=True)
        open(f"{d}/v8f_hacc.py", "w", encoding="utf-8", newline="\n").write(s)
        r = run(d, name[:3])
        r["RED"] = bool(r["rc"] == 1 and r["n_failed"] >= 1 and not r["errors"] and r["all_failures_are_assertions"])
        res[name] = r
        print(f"{name:45s} RED={r['RED']}  rc {r['rc']} failed {r['n_failed']} passed {r['n_passed']} assertion_only={r['all_failures_are_assertions']}  {r['seconds']}s  {r['failed'][:3]}", flush=True)
        json.dump(res, open(f"{HERE}/mutation_results.json", "w"), indent=1)
    ok = all(v.get("RED") for k, v in res.items() if k != "BASELINE_unmutated")
    res["ALL_MUTANTS_RED"] = ok
    json.dump(res, open(f"{HERE}/mutation_results.json", "w"), indent=1)
    print(f"ZZHACC-MUT-DONE all_red={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)
    except Exception:
        pass
    sys.exit(main())
