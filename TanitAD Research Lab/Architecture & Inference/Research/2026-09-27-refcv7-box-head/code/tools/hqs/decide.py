"""A14.1 decision after the three one-frame tests: the red arm (learned = unanchored) must FAIL, else VOID; the anchored
arms that PASS go to G-BOX-OVERFIT in the Master Mind's order (learned_ref first, then heatmap)."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
res = {}
for q in ("heatmap", "learned_ref", "learned"):
    f = D / f"of_{q}.json"
    try:
        res[q] = json.loads(f.read_text(encoding="utf-8"))["one_frame"]["RESULT"]
    except Exception as exc:  # noqa: BLE001
        res[q] = f"MISSING ({type(exc).__name__})"
if res["learned"] == "PASS":
    order, why = [], "VOID: the unanchored red arm PASSED its one-frame test"
else:
    order = [q for q in ("learned_ref", "heatmap") if res[q] == "PASS"]
    why = "anchored arms that passed, in the Master Mind's order" if order else "no anchored arm passed"
(D / "DECISION.json").write_text(json.dumps({"one_frame": res, "gbo_order": order, "why": why}, indent=1))
print(" ".join(order))
