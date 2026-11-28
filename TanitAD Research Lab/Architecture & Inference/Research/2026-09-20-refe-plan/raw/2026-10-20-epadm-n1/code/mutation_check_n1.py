"""Mutation arms for epadm_n1.py / epadm_sim_n1.py: each guard line is rewritten in a COPY and the planted-data suite (test_n1.py) must go RED.  Scratch copies only; the real modules are never touched.
    python mutation_check_n1.py <python> <outjson>      (env EP_ADMISSION_DIR, EP_ADMISSION_R1_DIR, EPADM_C1_DIR as for the tests; MUT_TMP = scratch parent dir)"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# (name, file, old text (must occur EXACTLY once), new text)
MUT = [
    ("N1-CHANGED-DEFINITION", "epadm_n1.py", "return np.asarray(pk_c) != np.asarray(pk_p)", "return np.asarray(pk_c) == np.asarray(pk_p)"),
    ("N1-RESTRICTION-TO-CHANGED", "epadm_n1.py", "out[h] = S.ratio((1.0 - ye) * chf, pred)", "out[h] = S.ratio((1.0 - ye), (1.0 - se))"),
    ("N1-ZERO-CHANGED-REFUSES", "epadm_n1.py", 'if n_ch == 0:\n        raise N1Undefined("zero changed picks', 'if False:\n        raise N1Undefined("zero changed picks'),
    ("N1-ZERO-PREDICTED-MASS-REFUSES", "epadm_n1.py", 'if not float(pred.sum()) > 0.0:\n            raise N1Undefined(f"{h}:', 'if False:\n            raise N1Undefined(f"{h}:'),
    ("N1-LOGVEC-ZERO-DENOMINATOR-REFUSES", "epadm_n1.py", "if not float(np.sum(pred_ch_log)) > 0.0:", "if False:"),
    ("N1-MARGIN-1.05", "epadm_n1.py", "N1_MARGIN = 1.05 ", "N1_MARGIN = 1.00 "),
    ("N1-ANY-SEED-QUANTIFIER", "epadm_n1.py", "if any((c is not None) and not n1_seed_ok(c, margin, heads) for c in per_seed):", "if all((c is not None) and not n1_seed_ok(c, margin, heads) for c in per_seed):"),
    ("N1-UNDEFINED-IS-NOT-A-PASS", "epadm_n1.py", 'if any(c is None for c in per_seed):\n        return "UNDEFINED-N1"', 'if any(c is None for c in per_seed):\n        return "ADMIT-N1"'),
    ("N1-NO-SEED-IS-NOT-A-PASS", "epadm_n1.py", 'if len(per_seed) == 0:\n        return "UNDEFINED-N1"', 'if len(per_seed) == 0:\n        return "ADMIT-N1"'),
    ("N1-ALL-HEADS-MUST-BE-OK", "epadm_n1.py", "return bool(all((h in cis) and A.a2_head_ok_r1(cis[h], margin) for h in heads))", "return bool(any((h in cis) and A.a2_head_ok_r1(cis[h], margin) for h in heads))"),
    ("N1-FROZEN-HEAD-RULE", "epadm_n1.py", "return bool(all((h in cis) and A.a2_head_ok_r1(cis[h], margin) for h in heads))", "return bool(all((h in cis) and cis[h]['point'] <= margin for h in heads))"),
    ("SIM-SALTS-NEW", "epadm_sim_n1.py", "N1_BASE_SALT, N1_OUTER_SALT = 20261401, 20261402", "N1_BASE_SALT, N1_OUTER_SALT = 20261301, 20261302"),
    ("SIM-BASE-C-USES-N1-SALT", "epadm_sim_n1.py", "rng = np.random.default_rng([N1_BASE_SALT, rep + offset])", "rng = np.random.default_rng([S0.S0_BASE_SALT, rep + offset])"),
    ("SIM-OUTER-DRAW-USES-N1-SALT", "epadm_sim_n1.py", "return np.random.default_rng([N1_OUTER_SALT, rep + offset]).integers(0, n, n)", "return np.random.default_rng([S0.S0_OUTER_SALT, rep + offset]).integers(0, n, n)"),
    ("SIM-GENERATOR-P-FIXES-THE-LOGS", "epadm_sim_n1.py", 'if gen == "P":\n        return np.arange(n)', 'if False:\n        return np.arange(n)'),
    ("SIM-CALIBRATE-AT-THE-CHANGED-CLASS", "epadm_sim_n1.py", "k = calib(so, sp) if kfix is None else kfix[(si, j)]", "k = None if kfix is None else kfix[(si, j)]"),
    ("SIM-K0-LEAVES-PREDICTION-UNSCALED", "epadm_sim_n1.py", "return (obs_sum / pred_sum) if (obs_sum > 0.0 and pred_sum > 0.0) else None", "return (obs_sum / pred_sum) if (obs_sum > 0.0 and pred_sum > 0.0) else 0.0"),
    ("SIM-PLANT-DIVIDES-BY-RHO", "epadm_sim_n1.py", "pred = predc * (1.0 if k is None else k) / rho_h", "pred = predc * (1.0 if k is None else k) * rho_h"),
    ("SIM-PLANT-KIND-REACHES-N1", "epadm_sim_n1.py", 'if (kind in ("plant_n1", "plant_a2l") and cond["kw"].get("head") == h) else 1.0', 'if (kind in ("plant_a2l",) and cond["kw"].get("head") == h) else 1.0'),
    ("SIM-UNDEFINED-BASE-IS-FLAGGED", "epadm_sim_n1.py", "if not sp > 0.0:                                               # no changed pick", "if False:                                               # no changed pick"),
    ("SIM-UNDEFINED-DRAWN-LOGS-IS-FLAGGED", "epadm_sim_n1.py", "except N1.N1Undefined:                                         # the redrawn logs hold no changed pick\n                    und[si, j] = True", "except N1.N1Undefined:                                         # the redrawn logs hold no changed pick\n                    pass"),
    ("SIM-IDENTITY-A1-IS-ZERO", "epadm_sim_n1.py", 'd = {"Yp": v["y_p"], "Sp": v["s_p"], "Ye": v["y_p"], "Se": v["s_p"]}', 'd = {"Yp": v["y_p"], "Sp": v["s_p"], "Ye": v["y_e"], "Se": v["s_e"]}'),
    ("SIM-IDENTITY-N1-IS-UNDEFINED", "epadm_sim_n1.py", 'if kind == "identity" else (v[f"OBSc_{h}"], v[f"PREDc_{h}"])', 'if False else (v[f"OBSc_{h}"], v[f"PREDc_{h}"])'),
    ("SIM-A2L-PLANTS-ONLY-THE-CHANGED-CLASS", "epadm_sim_n1.py", "pred_all = (1.0 if ku is None else ku) * pu + (1.0 if kc is None else kc) * pc / rho_h", "pred_all = ((1.0 if ku is None else ku) * pu + (1.0 if kc is None else kc) * pc) / rho_h"),
    ("SIM-SPLIT-PARTITION", "epadm_sim_n1.py", 'out[f"OBSc_{h}"], out[f"PREDc_{h}"] = obs[h] * cf, pred[h] * cf', 'out[f"OBSc_{h}"], out[f"PREDc_{h}"] = obs[h] * (1.0 - cf), pred[h] * (1.0 - cf)'),
    ("SIM-BASEX-REFUSES-WITHOUT-PC9", "epadm_sim_n1.py", 'if not os.path.exists(js) or json.load(open(js, encoding="utf-8")).get("pass") is not True:', "if False:"),
    ("SIM-DESIGN-GUARD", "epadm_sim_n1.py", 'if not os.path.exists(side) or open(side, encoding="utf-8").read().split()[0].lower() != sha256_file(DESIGN):', "if False:"),
    ("SIM-FALSE-REFUSAL-COUNTS-UNDEFINED", "epadm_sim_n1.py", 'nonadmit = np.array([v != "ADMIT-N1" for v in vm])', 'nonadmit = np.array([v == "FAIL-N1" for v in vm])'),
    ("SIM-POWER-COUNTS-ONLY-DEFINED-REFUSALS", "epadm_sim_n1.py", "return any((not und_i[si, j]) and not A.a2_head_ok_r1(S0.ci_of(n1_i[si, j]), margin) for si in range(n1_i.shape[0]))", "return any(not A.a2_head_ok_r1(S0.ci_of(n1_i[si, j]), margin) for si in range(n1_i.shape[0]))"),
    ("SIM-VERDICT-MODULE-UNDEFINED-SEED", "epadm_sim_n1.py", "per_seed.append(None if any(und_i[si, j_of[h]] for h in heads) else", "per_seed.append(None if False else"),
    ("SIM-BAR-MARGIN-1.05", "epadm_sim_n1.py", 'm = cal["N1"]["by_margin"][f"{BAR_MARGIN:.2f}"]', 'm = cal["N1"]["by_margin"]["1.00"]'),
    ("SIM-BAR-POWER-IS-THE-DAC-HEAD", "epadm_sim_n1.py", 'return a1, m["any_head_any_seed"], pm["per_head_any_seed"]["DAC"]', 'return a1, m["any_head_any_seed"], pm["per_head_any_seed"]["NC"]'),
]


def run_pytest(py, d, geo):
    env = dict(os.environ, S0_GEOMETRY=geo)
    return subprocess.run([py, "-I", "-m", "pytest", "test_n1.py", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={os.path.join(d, 'pt')}"], cwd=d, env=env, capture_output=True, text=True, timeout=900)


def main(py, out):
    geo = os.path.abspath(os.path.join(HERE, "..", "geometry_s0.json"))
    files = ("epadm_n1.py", "epadm_sim_n1.py", "epadm_sim_s0.py", "test_n1.py")
    src = {f: open(os.path.join(HERE, f), encoding="utf-8").read() for f in files}
    res = {}
    for name, f, old, new in MUT:
        if src[f].count(old) != 1:
            res[name] = {"status": "ANCHOR-NOT-UNIQUE", "count": src[f].count(old)}
            continue
        d = tempfile.mkdtemp(prefix="mut_n1_", dir=os.environ.get("MUT_TMP"))
        try:
            for g in files:
                open(os.path.join(d, g), "w", encoding="utf-8", newline="\n").write(src[g].replace(old, new) if g == f else src[g])
            r = run_pytest(py, d, geo)
            res[name] = {"status": "CAUGHT" if r.returncode != 0 else "SURVIVED", "rc": r.returncode, "tail": r.stdout[-200:].replace("\n", " ")}
        finally:
            shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp(prefix="mut_n1_", dir=os.environ.get("MUT_TMP"))
    try:                                                                                 # harness sanity: the unmutated copy must pass
        for g in files:
            open(os.path.join(d, g), "w", encoding="utf-8", newline="\n").write(src[g])
        r = run_pytest(py, d, geo)
        res["HARNESS-SANITY-unmutated"] = {"status": "PASS" if r.returncode == 0 else "FAIL", "tail": r.stdout[-120:].replace("\n", " ")}
    finally:
        shutil.rmtree(d, ignore_errors=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v["status"] for k, v in res.items()}, indent=1))
    return 0 if all(v["status"] in ("CAUGHT", "PASS") for v in res.values()) else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
