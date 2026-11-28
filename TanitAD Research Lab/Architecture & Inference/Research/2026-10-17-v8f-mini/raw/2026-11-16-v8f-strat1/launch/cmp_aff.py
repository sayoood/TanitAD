"""Pod-side: for the affected-file rerun on new3, list every failing test and whether it failed in base (pre-existing) or passed in base (a regression)."""
import glob
import json
import sys
import xml.etree.ElementTree as ET


def states(pattern):
    st = {}
    for f in glob.glob(pattern):
        for tc in ET.parse(f).getroot().iter("testcase"):
            tid = (tc.get("classname") or "") + "::" + (tc.get("name") or "")
            if any(ch.tag in ("failure", "error") for ch in tc):
                st[tid] = "fail"
            elif any(ch.tag == "skipped" for ch in tc):
                st[tid] = "skip"
            else:
                st[tid] = "pass"
    return st


base = states("/root/strat1/junit/base/shard*.xml")
new = states(sys.argv[1] if len(sys.argv) > 1 else "/root/strat1/junit/new3_aff/*.xml")
reg = set(json.load(open("/root/strat1/gsuite_result.json"))["regressions"])
out = {"n_new": len(new), "regressions_now_pass": sum(1 for r in reg if new.get(r) == "pass"),
       "regressions_not_pass": sorted(r for r in reg if new.get(r) != "pass"),
       "new_fail_base_pass": sorted(t for t, s in new.items() if s == "fail" and base.get(t) == "pass"),
       "new_fail_not_in_base": sorted(t for t, s in new.items() if s == "fail" and t not in base),
       "new_fail_base_fail": sum(1 for t, s in new.items() if s == "fail" and base.get(t) == "fail"),
       "new_skip_base_pass": sorted(t for t, s in new.items() if s == "skip" and base.get(t) == "pass")}
print(json.dumps(out, indent=1))
