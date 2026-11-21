"""Tip vs tip + batch 1, per TEST, from the per-file junit XMLs of the gate chain.

    python compare_gate.py            -> raw/gate/comparison.json + a printed summary

* REGRESSION = a test that FAILS or ERRORS on FIX and PASSES on TIP (the brief: must be empty, or
  explained per test).
* FIX-only tests (the new files) and TIP-only tests are listed separately.
* A file whose run did not complete (RAM_ABORT / TIMEOUT / no junit) is reported as INCOMPLETE,
  never silently dropped: an absent result is not a pass.
* TIPRED: the new tests on the unfixed tip + the two new modules -- which go RED on the defect.
"""
import json
import pathlib
import xml.etree.ElementTree as ET

G = pathlib.Path(__file__).resolve().parents[1] / "raw" / "gate"


def results(tag: str) -> tuple[dict, list]:
    """{test_id: outcome}, [incomplete files] for one tree tag (FIX / TIP / TIPRED / FIXNEW)."""
    out, incomplete = {}, []
    for part in ("stack", "taniteval"):
        jl = G / f"{tag}_{part}.jsonl"
        if not jl.exists():
            continue
        final = {}
        for ln in jl.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            if r.get("file") != "__header__":
                final[r["file"]] = r
        for f, r in final.items():
            junit = G / f"{tag}_{part}_logs" / (f.replace("/", "__") + ".junit.xml")
            if r.get("status") != "OK" or not junit.exists():
                incomplete.append({"tree": tag, "part": part, "file": f, "status": r.get("status"),
                                   "rc": r.get("rc"), "tail": r.get("tail")})
                if not junit.exists():
                    continue
            root = ET.parse(junit).getroot()
            for tc in root.iter("testcase"):
                tid = f"{part}::{tc.get('classname')}::{tc.get('name')}"
                if tc.find("failure") is not None:
                    o = "failed"
                elif tc.find("error") is not None:
                    o = "error"
                elif tc.find("skipped") is not None:
                    o = "skipped"
                else:
                    o = "passed"
                out[tid] = o
            # a collection error shows as an <error> testcase or as rc 2 with no testcases
            if r.get("rc") == 2 and not list(root.iter("testcase")):
                out[f"{part}::{f}::<collection>"] = "error"
    return out, incomplete


def main() -> None:
    fix, fix_inc = results("FIX")
    tip, tip_inc = results("TIP")
    red, red_inc = results("TIPRED")
    fnew, fnew_inc = results("FIXNEW")
    bad = ("failed", "error")
    regressions = sorted(t for t, o in fix.items() if o in bad and tip.get(t) == "passed")
    fixed = sorted(t for t, o in tip.items() if o in bad and fix.get(t) == "passed")
    both_bad = sorted(t for t, o in fix.items() if o in bad and tip.get(t) in bad)
    only_fix = sorted(t for t in fix if t not in tip)
    only_tip = sorted(t for t in tip if t not in fix)
    count = lambda d: {k: sum(1 for v in d.values() if v == k)
                       for k in ("passed", "failed", "error", "skipped")}
    res = {
        "what": "tip da8400b vs tip + batch 1 (declared-vs-built), per test, from per-file junit",
        "FIX": count(fix), "TIP": count(tip), "TIPRED": count(red), "FIXNEW": count(fnew),
        "regressions_fix_fails_tip_passes": regressions,
        "fixed_tip_fails_fix_passes": fixed,
        "failing_on_both": both_bad,
        "only_in_fix": {"n": len(only_fix), "outcomes": count({t: fix[t] for t in only_fix}),
                        "not_passed": sorted(t for t in only_fix if fix[t] != "passed")},
        "only_in_tip": only_tip,
        "tipred_red": sorted(t for t, o in red.items() if o in bad),
        "tipred_green": sorted(t for t, o in red.items() if o == "passed"),
        "incomplete": fix_inc + tip_inc + red_inc + fnew_inc,
    }
    (G / "comparison.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 40 else f"{len(v)} items")
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
