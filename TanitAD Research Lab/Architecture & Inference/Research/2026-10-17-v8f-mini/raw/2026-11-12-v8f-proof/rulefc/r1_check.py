"""r1_check.py -- control r1 of card ADD144 (amendment 1 point 4): the run H_RLcFCz_s0 (forecast wrapper, a_L forced to 0) must equal H_RLcK_s0 bit for bit.

  python r1_check.py <pulled H_RLcFCz_s0 dir> <pulled H_RLcK_s0 dir> <out json>

Checks: (1) distrule/cmp_dumps.py on the two dump folders (141 files, 0 differing); (2) every numeric column of rule_log, rule_log_mid, p1_first and the
ticks file equal (NaN equals NaN); (3) the readout logs equal apart from paths and tags; (4) the forecast log: zero flag on, every a_L exactly 0, rows at
the tick (q = 0) and at the later look (q = 5), rows with a lead present. Exit 0 only when all pass. Opens no outcome number of the card.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CMP = os.path.join(HERE, "..", "distrule", "cmp_dumps.py")


def npz_equal(a, b):
    with np.load(a, allow_pickle=True) as x, np.load(b, allow_pickle=True) as y:
        ka = {k for k in x.files if x[k].dtype.kind not in "OUS"}
        kb = {k for k in y.files if y[k].dtype.kind not in "OUS"}
        bad = sorted(ka ^ kb)
        n = 0
        for k in sorted(ka & kb):
            if x[k].shape != y[k].shape or not np.array_equal(x[k], y[k], equal_nan=(x[k].dtype.kind == "f")):
                bad.append(k)
            n += 1
    return n, bad


def strip(o, tags):
    if isinstance(o, dict):
        return {k: strip(v, tags) for k, v in o.items()}
    if isinstance(o, list):
        return [strip(v, tags) for v in o]
    if isinstance(o, str):
        for t in tags:
            o = o.replace(t[0], t[1])
        return o
    return o


def main(a_dir, b_dir, out):
    a_arm, b_arm = os.path.basename(a_dir.rstrip("/\\")), os.path.basename(b_dir.rstrip("/\\"))
    res = {"a": a_arm, "b": b_arm, "checks": {}}
    tag_a, tag_b = a_arm + "_m10_causal_heads", b_arm + "_m10_causal_heads"
    p = subprocess.run([sys.executable, CMP, os.path.join(a_dir, "dump_" + tag_a), os.path.join(b_dir, "dump_" + tag_b)], capture_output=True, text=True, encoding="utf-8")
    line = [x for x in p.stdout.splitlines() if x.startswith("CMP ")]
    res["checks"]["dumps"] = {"rc": p.returncode, "line": line[-1] if line else p.stdout[-300:], "pass": p.returncode == 0}
    for kind in ("rule_log_%s.npz", "rule_log_mid_%s.npz", "p1_first_%s.npz"):
        fa, fb = os.path.join(a_dir, "ticks", kind % a_arm), os.path.join(b_dir, "ticks", kind % b_arm)
        n, bad = npz_equal(fa, fb)
        res["checks"][kind.split("_%s")[0]] = {"columns": n, "differing": bad, "pass": n > 0 and not bad}
    n, bad = npz_equal(os.path.join(a_dir, "ticks", "ticks_%s.npz" % tag_a), os.path.join(b_dir, "ticks", "ticks_%s.npz" % tag_b))
    res["checks"]["ticks"] = {"columns": n, "differing": bad, "pass": n > 0 and not bad}
    ra = json.load(open(os.path.join(a_dir, "ticks", "readout_log_%s.json" % a_arm), encoding="utf-8"))
    rb = json.load(open(os.path.join(b_dir, "ticks", "readout_log_%s.json" % b_arm), encoding="utf-8"))
    sa = strip(ra, [(a_arm, "ARM")])
    sb = strip(rb, [(b_arm, "ARM")])
    res["checks"]["readout_log"] = {"pass": sa == sb, "keys_differing": sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))}
    with np.load(os.path.join(a_dir, "ticks", "fc_log_%s.npz" % a_arm)) as z:
        aL, q, pres, zero = z["a_L"], z["q"], z["present"], bool(z["zero"])
        res["checks"]["fc_log"] = {"rows": int(len(aL)), "zero_flag": zero, "all_aL_zero": bool((aL == 0).all()), "q_values": sorted(set(int(x) for x in q)),
                                   "present_rows": int(pres.sum()), "pass": zero and bool((aL == 0).all()) and sorted(set(int(x) for x in q)) == [0, 5] and int(pres.sum()) > 0}
    res["pass"] = all(c["pass"] for c in res["checks"].values())
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
