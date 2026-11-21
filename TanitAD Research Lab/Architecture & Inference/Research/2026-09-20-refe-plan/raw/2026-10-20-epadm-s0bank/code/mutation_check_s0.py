"""Mutation arms for epadm_sim_s0.py: each guard line is rewritten in a COPY and the planted-data suite must go RED.  Scratch copies only; the real module is never touched.
    python mutation_check_s0.py <python> <outjson>      (env EP_ADMISSION_DIR, EP_ADMISSION_R1_DIR, EPADM_C1_DIR as for the tests)"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MUT = [
    ("K0-FIX", 'd[f"PRED_{h}"] = v[f"PRED_{h}"] / rho_h', "pass"),
    ("GEN-T-FIXED-LOGS", 'if gen == "T":\n        return np.arange(n)', 'if False:\n        return np.arange(n)'),
    ("CLONE-CHECK", 'if len(ids) != int(counts[l]) or set(pool["log_idx"][ids].tolist()) != {int(bc["src"][l])}:\n            return False', "if False:\n            return False"),
    ("CLONE-ONE-SOURCE", "ids = ids_by[int(src[l])]", "ids = ids_by[int(rng.integers(0, n_src))]"),
    ("SALTS-NEW", "S0_BASE_SALT, S0_OUTER_SALT = 20261301, 20261302", "S0_BASE_SALT, S0_OUTER_SALT = 20261201, 20261202"),
    ("GEOMETRY-PIN", 'GEOMETRY_SHA256 = "7a2e71e6', 'GEOMETRY_SHA256 = "0a2e71e6'),
    ("GEOMETRY-TOTAL", 'if int(counts.sum()) != TOKENS[name]', 'if False'),
    ("BAR-STATUS", 'return "MET" if x_ok else "BASE-DEPENDENT"', 'return "MET"'),
    ("A1-MODULE", '== "FAIL-A1"\n\n\ndef a1_refuse_inline', '== "ADMIT-A1"\n\n\ndef a1_refuse_inline'),
    ("FAILCLOSED-EXTRA", "return bool(any(k0_i[si, j_of[h]] for si in range(k0_i.shape[0]) for h in heads))", "return False"),
    ("DESIGN-GUARD", 'if not os.path.exists(side) or open(side, encoding="utf-8").read().split()[0].lower() != sha256_file(DESIGN):', "if False:"),
    ("IDENTITY-ZERO", 'd = {"Yp": v["y_p"], "Sp": v["s_p"], "Ye": v["y_p"], "Se": v["s_p"]}', 'd = {"Yp": v["y_p"], "Sp": v["s_p"], "Ye": v["y_e"], "Se": v["s_e"]}'),
]


def main(py, out):
    src = open(os.path.join(HERE, "epadm_sim_s0.py"), encoding="utf-8").read()
    res = {}
    for name, old, new in MUT:
        if src.count(old) != 1:
            res[name] = {"status": "ANCHOR-NOT-UNIQUE", "count": src.count(old)}
            continue
        d = tempfile.mkdtemp(prefix="mut_s0_", dir=os.environ.get("MUT_TMP"))
        try:
            open(os.path.join(d, "epadm_sim_s0.py"), "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
            shutil.copy(os.path.join(HERE, "test_sim_s0.py"), d)
            env = dict(os.environ, S0_GEOMETRY=os.path.abspath(os.path.join(HERE, "..", "geometry_s0.json")))
            r = subprocess.run([py, "-I", "-m", "pytest", "test_sim_s0.py", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={os.path.join(d, 'pt')}"], cwd=d, env=env, capture_output=True, text=True, timeout=900)
            res[name] = {"status": "CAUGHT" if r.returncode != 0 else "SURVIVED", "rc": r.returncode, "tail": r.stdout[-160:].replace("\n", " ")}
        finally:
            shutil.rmtree(d, ignore_errors=True)
    # harness sanity: the unmutated copy must pass
    d = tempfile.mkdtemp(prefix="mut_s0_", dir=os.environ.get("MUT_TMP"))
    try:
        shutil.copy(os.path.join(HERE, "epadm_sim_s0.py"), d)
        shutil.copy(os.path.join(HERE, "test_sim_s0.py"), d)
        env = dict(os.environ, S0_GEOMETRY=os.path.abspath(os.path.join(HERE, "..", "geometry_s0.json")))
        r = subprocess.run([py, "-I", "-m", "pytest", "test_sim_s0.py", "-q", "-p", "no:cacheprovider", f"--basetemp={os.path.join(d, 'pt')}"], cwd=d, env=env, capture_output=True, text=True, timeout=900)
        res["HARNESS-SANITY-unmutated"] = {"status": "PASS" if r.returncode == 0 else "FAIL", "tail": r.stdout[-120:].replace("\n", " ")}
    finally:
        shutil.rmtree(d, ignore_errors=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v["status"] for k, v in res.items()}, indent=1))
    return 0 if all(v["status"] in ("CAUGHT", "PASS") for v in res.values()) else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
