"""Verdict for a thor_gate_runner2 run: python gate_verdict.py <local gate dir> (holds res/results/{CAND,TIP}/*.xml).

REGRESSION = the test FAILS on CAND and PASSES on TIP. NEW-FAIL = fails on CAND, absent on TIP.
Also reports files whose rc != 0 without any junit (collection error / crash), and TIP tests MISSING on CAND
(a dropped test is not a pass). Writes <dir>/verdict.json and prints a summary."""
import json
import pathlib
import sys
import xml.etree.ElementTree as ET


def load(r, stage):
    out = {}
    for x in sorted((r / stage).glob("*.xml")):
        for tc in ET.parse(x).getroot().iter("testcase"):
            tid = f"{tc.get('classname')}::{tc.get('name')}"
            f, e = tc.find("failure"), tc.find("error")
            if f is not None or e is not None:
                el = f if f is not None else e
                out[tid] = ("FAIL", (el.get("message") or "")[:240])
            elif tc.find("skipped") is not None:
                out[tid] = ("SKIP", (tc.find("skipped").get("message") or "")[:160])
            else:
                out[tid] = ("PASS", "")
    return out


def main(d):
    r = d / "res" / "results"
    c, t = load(r, "CAND"), load(r, "TIP")
    assert c and t, f"no junit parsed: CAND {len(c)} TIP {len(t)}"
    cnt = lambda o, s: sum(1 for v in o.values() if v[0] == s)  # noqa: E731
    reg = sorted((k, c[k][1]) for k in c if c[k][0] == "FAIL" and t.get(k, ("",))[0] == "PASS")
    newf = sorted((k, c[k][1]) for k in c if c[k][0] == "FAIL" and k not in t)
    pre = sorted(k for k in c if c[k][0] == "FAIL" and t.get(k, ("",))[0] == "FAIL")
    missing = sorted(k for k in t if k not in c)
    tip_skip_cand_fail = sorted(k for k in c if c[k][0] == "FAIL" and t.get(k, ("",))[0] == "SKIP")
    nojunit = []
    for stage in ("CAND", "TIP"):
        js = json.loads((r / f"{stage}.json").read_text(encoding="utf-8"))
        for f in js["files"]:
            if f.get("rc") not in (0, 5) and not f.get("junit"):
                nojunit.append((stage, f["file"], f.get("rc"), (f.get("tail") or "")[-300:]))
    v = {
        "cand": {"pass": cnt(c, "PASS"), "fail": cnt(c, "FAIL"), "skip": cnt(c, "SKIP"), "n": len(c)},
        "tip": {"pass": cnt(t, "PASS"), "fail": cnt(t, "FAIL"), "skip": cnt(t, "SKIP"), "n": len(t)},
        "regressions": reg, "new_fail": newf, "pre_existing_fail": pre, "tip_skip_cand_fail": tip_skip_cand_fail,
        "missing_on_cand": missing, "rc_without_junit": nojunit,
    }
    v["verdict"] = "PASS" if not (reg or newf or tip_skip_cand_fail or nojunit) else "FAIL"
    (d / "verdict.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
    print(json.dumps({k: v[k] for k in ("cand", "tip", "verdict")}))
    for k in ("regressions", "new_fail", "tip_skip_cand_fail", "rc_without_junit"):
        print(f"{k}: {len(v[k])}")
        for row in v[k][:25]:
            print("   ", row)
    print(f"pre_existing_fail: {len(pre)}; missing_on_cand: {len(missing)}")
    for k in missing[:15]:
        print("    MISSING", k)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
