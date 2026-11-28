"""Output identity: families6_lowmem.py == the pinned families6.py on the same seam subset (and a regression arm).

Run:  python check_families6_lowmem.py <pinned families6.py> <seam.npz> <inputs.json.gz> <work dir> [n_tokens]
The two outputs must be equal key-for-key after dropping the ``seam`` path; a deliberate-regression arm (poses
shifted 1 m) must DIFFER, or the comparison is blind.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd, log):
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert rc == 0, f"rc {rc}: see {log}"


def load(p):
    d = json.load(open(p, encoding="utf-8"))
    d.pop("seam", None)
    return d


def main():
    f6, seam, inputs, work = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    os.makedirs(work, exist_ok=True)
    z = np.load(seam)
    sel = np.arange(min(n, len(z["token"])))
    keep = {k: z[k] for k in z.files}
    sub = {k: (v[sel] if v.ndim and len(v) == len(z["token"]) else v) for k, v in keep.items()}
    s_sub = os.path.join(work, "sub_seam.npz")
    np.savez(s_sub, **sub)
    bad = dict(sub)
    bad["poses"] = sub["poses"].copy()
    bad["poses"][..., 0] += 1.0
    s_bad = os.path.join(work, "bad_seam.npz")
    np.savez(s_bad, **bad)
    common = ["--inputs", inputs, "--stage", "1", "--label", "IDENTITY", "--n-boot", "200"]
    o_pin, o_low, o_bad = (os.path.join(work, x) for x in ("pinned.json", "lowmem.json", "regression.json"))
    run([PY, f6, "--seam", s_sub, "--out", o_pin, *common], os.path.join(work, "pinned.log"))
    run([PY, os.path.join(HERE, "families6_lowmem.py"), "--families6", f6, "--seam", s_sub, "--out", o_low, *common],
        os.path.join(work, "lowmem.log"))
    run([PY, os.path.join(HERE, "families6_lowmem.py"), "--families6", f6, "--seam", s_bad, "--out", o_bad, *common],
        os.path.join(work, "regression.log"))
    a, b, c = load(o_pin), load(o_low), load(o_bad)
    same = a == b
    blind = c == a
    res = {"n_tokens": a.get("n"), "identical": same, "regression_differs": not blind,
           "PASS": bool(same and not blind and a.get("n", 0) > 0)}
    print(json.dumps(res))
    json.dump(res, open(os.path.join(work, "identity_result.json"), "w"), indent=1)
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
