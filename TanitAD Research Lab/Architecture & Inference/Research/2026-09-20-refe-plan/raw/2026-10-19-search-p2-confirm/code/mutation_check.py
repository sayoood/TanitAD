#!/usr/bin/env python3
"""Mutation check of the chain's guard lines: each mutation disables ONE guard in a scratch copy of the package, and `test_spc.py` MUST then fail.  A guard whose removal leaves the suite green is untested.
    CUDA_VISIBLE_DEVICES=-1 python mutation_check.py --out ../logs/mutations.json
Exit 0 only if every mutation is caught (and the unmutated copy passes first: the control)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
MUT = [
    ("spc_chain.py", 'if hashlib.sha256(text).hexdigest() != want:', 'if False:', "G-SEAL: the text's sha256 vs its .sha256"),
    ("spc_chain.py", 'if not m or not m.group(0).startswith("Status: **SEALED"):', 'if False:', "G-SEAL: the first Status line must be SEALED"),
    ("spc_chain.py", 'if miss:\n            self.stop(5, "SEAL_PIN_MISSING"', 'if False:\n            self.stop(5, "SEAL_PIN_MISSING"', "G-SEAL: a required pin missing"),
    ("spc_chain.py", 'if bad:\n            self.stop(5, "SEAL_PIN_MISMATCH"', 'if False:\n            self.stop(5, "SEAL_PIN_MISMATCH"', "G-SEAL: a pin that differs from its file"),
    ("spc_chain.py", 'if f"code:{f}" not in pins:', 'if False:', "G-SEAL: an operative file with no pin"),
    ("spc_chain.py", 'if pins["b_rest_tokens_sha256"] != SL.B_REST_SHA256:', 'if False:', "G-SEAL: the B-REST pin vs the launcher constant"),
    ("spc_chain.py", 'if rec["tokens"] != rest or BR.sha_tokens(rest) != pins["b_rest_tokens_sha256"]:', 'if False:', "G-DATA: raw/b_rest.json vs the recomputation and the pin"),
    ("spc_chain.py", 'if BR.sha_tokens(cid["tokens"]) != SL.CID_SHA256 or len(cid["tokens"]) != 16 or not set(cid["tokens"]) <= set(cb["tokens"]):', 'if False:', "G-DATA: the C-ID sample is the pinned list"),
    ("spc_chain.py", 'if free < self.cfg.min_free_gib:', 'if False:', "disk gate"),
    ("spc_chain.py", 'if not re.search(pat, text, re.M):', 'if False:', "G-BRIDGE / G-BUNDLE markers"),
    ("spc_chain.py", 'if re.search(r"^ZZSTAGE_FAIL", text, re.M):', 'if False:', "GPU stage failure marker"),
    ("spc_chain.py", 'if len(toks) != expect_n or BR.sha_tokens(toks) != want:', 'if False:', "G-BUNDLE: bundle token list vs the pinned read"),
    ("spc_chain.py", 'if cj.get("tokens_sha256") != want or cj.get("n") != expect_n:', 'if False:', "G-BUNDLE: ctx.json vs the pinned read"),
    ("spc_chain.py", 'sha256_file(p) != cj.get("ctx_sha256"):', 'False:', "G-BUNDLE: ctx.npy bytes vs ctx.json"),
    ("spc_chain.py", 'and st.get("bundle_sha256") == sha256_file(os.path.join(self.bundle_dir(which), "bundle.npz"))', 'and True', "G-REACH: the stamp must bind THIS bundle"),
    ("spc_chain.py", 'st.get("verdict") == "PASS"', 'True', "G-REACH: the stamp must be a PASS"),
    ("spc_chain.py", 'if rc != 0 or not self.stamp_ok(which, pins):', 'if False:', "G-REACH: no stamp, no search"),
    ("spc_chain.py", 'if meta.get("arm") != "P2HEAD" or meta.get("head_sha256") is not None:', 'if False:', "G-NOHEAD"),
    ("spc_chain.py", 'if (meta.get("data_clause") or {}).get("carve_sha256") != SL.READS[which]:', 'if False:', "the data clause recorded in the search meta"),
    ("spc_chain.py", 'if meta.get("ckpt_sha256") != pins["ckpt_sha256"] or meta.get("config_sha256") != pins["config_sha256"]:', 'if False:', "ckpt / config recorded in the search meta"),
    ("spc_chain.py", 'if not res["pass"]:\n            self.stop(3, "G_CID_NOT_PASSED"', 'if False:\n            self.stop(3, "G_CID_NOT_PASSED"', "C-ID"),
    ("spc_chain.py", 'if not (j.get("passed") is True and j.get("control_ok") is True):', 'if False:', "G-LABEL: the planted identity and its control"),
    ("spc_chain.py", 'if not toks_ok or not fin:', 'if False:', "G-LABEL: tokens aligned and finite"),
    ("spc_chain.py", 'if self.ram_fn() < c.min_ram_gb:', 'if False:', "RAM gate before the labeller"),
    ("spc_chain.py", 'if miss:\n            self.stop(8, "G_ORDER"', 'if False:\n            self.stop(8, "G_ORDER"', "G-ORDER"),
    ("spc_cid.py", 'bad = [i for i in range(n) if ra[i].tobytes() != rb[i].tobytes()]', 'bad = []', "C-ID: the bitwise comparison"),
    ("spc_cid.py", 'if n_moved < min_moved or n_unmoved < 1:', 'if False:', "C-ID: non-vacuity"),
    ("spc_cid.py", 'if not np.array_equal(np.asarray(banked_ctx[r]), np.asarray(rerun_ctx[j]))', 'if False', "C-ID: the ctx rows"),
    ("spc_cid.py", 'if sorted(stoks) != sorted(btoks):', 'if False:', "C-ID: search and bundle on the same tokens"),
    ("spc_launch.py", 'if any(x == "--head" or', 'if False and any(x == "--head" or', "launcher: no head argument"),
    ("spc_launch.py", 'if ns.arm != "P2HEAD":', 'if False:', "launcher: P2HEAD only"),
    ("spc_launch.py", 'carve_sha256=READS[read])\n\n\ndef cmd_gate', 'carve_sha256=None)\n\n\ndef cmd_gate', "launcher: the pinned data clause reaches the search"),
    ("spc_analyse.py", 'bool(dpdms["point"] >= BAR_DPDMS)', 'bool(dpdms["point"] >= 0.0)', "classifier: the +0.30 point bar"),
    ("spc_analyse.py", 'bool(dpdms["lo"] > 0.0)', 'bool(dpdms["lo"] > -1.0)', "classifier: the lower-CI term"),
    ("spc_analyse.py", 'bool(dpdms["point"] > seed_range)', 'True', "classifier: the seed-range term"),
    ("spc_analyse.py", 'bool(safety_points[k] >= BAR_SAFETY)', 'True', "classifier: the safety terms"),
    ("spc_analyse.py", 'elif dpdms["hi"] < 0.0:', 'elif False:', "classifier: NEGATIVE"),
    ("spc_gpu_stage.py", 'if r.returncode != 0 or not re.search(ok_re, text):', 'if False:', "GPU stage: rc and marker per step"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res, caught = [], 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    for k in range(-1, len(MUT)):
        tmp = tempfile.mkdtemp(prefix="spcmut_")
        try:
            for d in ("code", "raw", "src"):
                shutil.copytree(os.path.join(PKG, d), os.path.join(tmp, d), ignore=shutil.ignore_patterns("__pycache__"))
            label = "CONTROL (unmutated)"
            if k >= 0:
                f, old, new, label = MUT[k]
                p = os.path.join(tmp, "code", f)
                s = open(p, encoding="utf-8").read()
                if s.count(old) != 1:
                    res.append({"mutation": label, "file": f, "status": "MUTATION_NOT_APPLICABLE", "occurrences": s.count(old)})
                    continue
                open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
            r = subprocess.run([sys.executable, "-m", "pytest", "test_spc.py", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={os.path.join(tmp, 'pt')}"], cwd=os.path.join(tmp, "code"), capture_output=True, text=True, env=env)
            ok = r.returncode == 0
            if k < 0:
                res.append({"mutation": label, "status": "CONTROL_GREEN" if ok else "CONTROL_RED", "tail": r.stdout[-300:]})
                if not ok:
                    print("ZZMUT_CONTROL_FAILED", r.stdout[-600:])
                    json.dump(res, open(a.out, "w"), indent=1)
                    return 2
            else:
                failed_tests = [l for l in r.stdout.splitlines() if l.startswith("FAILED")][:2]
                res.append({"mutation": label, "file": MUT[k][0], "status": "CAUGHT" if not ok else "SURVIVED", "first_failing": failed_tests})
                caught += (not ok)
                print("ZZMUT", "CAUGHT  " if not ok else "SURVIVED", MUT[k][0], "|", label, flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    n = len([x for x in res if x["status"] in ("CAUGHT", "SURVIVED", "MUTATION_NOT_APPLICABLE")])
    survived = [x for x in res if x["status"] in ("SURVIVED", "MUTATION_NOT_APPLICABLE")]
    json.dump({"caught": caught, "total": n, "survived_or_inapplicable": survived, "results": res}, open(a.out, "w"), indent=1)
    print(f"ZZMUT_SUMMARY caught {caught} of {n}; survived/inapplicable: {[x['mutation'] for x in survived]}")
    return 0 if not survived else 1


if __name__ == "__main__":
    sys.exit(main())
