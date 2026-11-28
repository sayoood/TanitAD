"""After the batch-3 dev run: every file with a failure or error on FIX3 is re-run on a byte-faithful
TIP tree (`git archive b4a59b9`), and the per-testcase junit verdicts are compared.

    python b3_tip_control.py <tip tree>

Writes raw/gate_b3_dev/TIP3_control_stack.jsonl (+ logs/junit) and
raw/gate_b3_dev/regressions_FIX3_vs_TIP3.json: a REGRESSION is a testcase that fails / errors on
FIX3 and passes on TIP3. `[]` is the result that lets the failures be read as pre-existing
(environment) rather than batch 3's. The runner's RAM floor applies to the control too.
"""
import json
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

PK = pathlib.Path(__file__).resolve().parents[1]
O = PK / "raw" / "gate_b3_dev"
TIP_TREE = sys.argv[1]
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

final = {}
for ln in (O / "FIX3_stack.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(ln)
    if r.get("file") != "__header__":
        final[r["file"]] = r
bad_files = [f for f, r in final.items()
             if r.get("counts", {}).get("failed") or r.get("counts", {}).get("error")
             or r.get("status") != "OK"]
lst = O / "tip_control_files.txt"
lst.write_text("\n".join(bad_files) + "\n", encoding="utf-8")
print(f"files with a FIX3 failure: {bad_files}")
if bad_files:
    subprocess.run([PY, str(PK / "code" / "run_suite_bounded.py"), TIP_TREE, str(lst),
                    str(O / "TIP3_control_stack.jsonl"), "--rundir", "stack"], check=False)


def verdicts(junit: pathlib.Path) -> dict:
    out = {}
    if not junit.exists():
        return out
    for tc in ET.parse(junit).getroot().iter("testcase"):
        st = "passed"
        for c in tc:
            if c.tag in ("failure", "error", "skipped"):
                st = c.tag
        out[f"{tc.get('classname')}::{tc.get('name')}"] = st
    return out


report = {"tip_tree": TIP_TREE, "files": {}, "regressions": []}
for f in bad_files:
    stem = f.replace("/", "__")
    fx = verdicts(O / "FIX3_stack_logs" / f"{stem}.junit.xml")
    tp = verdicts(O / "TIP3_control_stack_logs" / f"{stem}.junit.xml")
    fails_fix = sorted(k for k, v in fx.items() if v in ("failure", "error"))
    fails_tip = sorted(k for k, v in tp.items() if v in ("failure", "error"))
    regress = sorted(k for k in fails_fix if tp.get(k) == "passed")
    unread = sorted(k for k in fails_fix if k not in tp)
    report["files"][f] = {"fix3_failed": fails_fix, "tip3_failed": fails_tip,
                          "fail_sets_identical": fails_fix == fails_tip,
                          "regressions": regress, "not_run_on_tip": unread}
    report["regressions"] += regress
(O / "regressions_FIX3_vs_TIP3.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=1))
for f, d in report["files"].items():
    print(f"{f}: FIX3 {len(d['fix3_failed'])} failed, TIP3 {len(d['tip3_failed'])} failed, "
          f"identical={d['fail_sets_identical']}, regressions={d['regressions']}, "
          f"not run on TIP3={d['not_run_on_tip']}")
