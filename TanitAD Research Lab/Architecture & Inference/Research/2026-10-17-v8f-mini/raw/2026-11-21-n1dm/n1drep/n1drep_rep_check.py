"""n1drep_rep_check.py -- MM 2026-11-22, card PREREG_V8F_N1DREP (sha256 0c641197), control K-REP-CODE. stdlib only, CPU, sub-second.

  argv  NEW_ARGV_JSON REF_ARGV_JSON NEW_OUT REF_OUT   the two argv token lists are equal except the value after --out (NEW_OUT vs REF_OUT)
  token NEW_TOKEN_JSON REF_TOKEN_JSON                the gate tokens agree on commit, trainer, checker, tree manifest, G-SUITE result, nav reports, data
Prints PASS/FAIL lines; exit 0 only when every check passes.
"""
import json
import sys

FIELDS = ("commit", "trainer_sha256", "checker_sha256", "tree_manifest_sha256", "gsuite_result_sha256", "nav_report_sha256", "nav_rb_report_sha256", "data", "kind", "schema")


def load(p):
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d["argv"] if isinstance(d, dict) and "argv" in d else d


def argv(a, b, new_out, ref_out):
    bad = []
    if not (isinstance(a, list) and isinstance(b, list)):
        return ["argv files are not token lists"]
    if len(a) != len(b):
        bad.append(f"length {len(a)} vs {len(b)}")
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        if i > 0 and a[i - 1] == "--out" and b[i - 1] == "--out" and x == new_out and y == ref_out:
            continue
        bad.append(f"token {i}: {x!r} vs {y!r}")
    if a.count("--out") != 1 or b.count("--out") != 1:
        bad.append("--out must appear exactly once in each")
    elif a[a.index("--out") + 1] != new_out or b[b.index("--out") + 1] != ref_out:
        bad.append("--out values are not the stated ones")
    return bad


def token(a, b):
    return [f"{k}: differs" + ("" if k in a and k in b else " (missing)") for k in FIELDS if a.get(k) != b.get(k) or k not in a]


def main(v):
    if v[1] == "argv" and len(v) == 6:
        bad = argv(load(v[2]), load(v[3]), v[4], v[5])
    elif v[1] == "token" and len(v) == 4:
        bad = token(json.load(open(v[2], encoding="utf-8")), json.load(open(v[3], encoding="utf-8")))
    else:
        print(__doc__)
        return 2
    for m in bad:
        print("PROBLEM", m)
    print("ZZ N1DREP K-REP-CODE", v[1], "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
