"""Read the four JUnit XMLs of the battery-test interaction runs and bank one JSON.

The runs (2026-09-26, dev box, CPU): the refcv6 battery's landed `test_yaw_valid.py` and this
package's PROPOSED replacement, each against a clean tip tree whose `refav1_arm.py` is either the
tip blob (c7013107...) or the fixed one (963d98e6...). REFCV6_REPO pointed at that tree.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
J = Path("C:/Users/Admin/ym26/junit")
RUNS = {
    "landed_test_on_TIP_cell": "junit_battery_yaw_TIP.xml",
    "landed_test_on_FIXED_cell": "junit_battery_yaw_FIXED.xml",
    "proposed_test_on_TIP_cell": "junit_battery_proposed_TIP.xml",
    "proposed_test_on_FIXED_cell": "junit_battery_proposed_FIXED.xml",
}
out = {"tool": "2026-09-26-yaw-rate-mask/code/summarize_battery_interaction.py",
       "landed_test": ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/"
                       "battery/code/test_yaw_valid.py (tip blob, unmodified)"),
       "proposed_test": "2026-09-26-yaw-rate-mask/code/proposals/battery_test_yaw_valid.PROPOSED.py",
       "refav1_arm_blobs": {"TIP": "c7013107ab1353c5102fd806d29cd967216b7a3a",
                            "FIXED": "963d98e6c3f1a00df8599c9ea8f39cb538e26e23"},
       "runs": {}}
for nm, f in RUNS.items():
    root = ET.parse(J / f).getroot()
    cases = {}
    for tc in root.iter("testcase"):
        st = "passed"
        for ch in tc:
            if ch.tag in ("failure", "error"):
                st = ch.tag
            elif ch.tag == "skipped":
                st = "skipped"
        cases[tc.get("name")] = st
    out["runs"][nm] = {"junit": f, "cases": cases,
                       "n_passed": sum(v == "passed" for v in cases.values()),
                       "n_failed": sum(v in ("failure", "error") for v in cases.values())}
dst = Path(sys.argv[1])
dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: (v["n_passed"], v["n_failed"]) for k, v in out["runs"].items()}, indent=1))
