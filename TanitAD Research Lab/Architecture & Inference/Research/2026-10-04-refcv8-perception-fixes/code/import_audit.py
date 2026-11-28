"""WP-C: assert nothing the fixes import comes from the G: mount (the rule: never touch G:). Run with PYTHONPATH = the extraction's stack, stack/scripts."""
import os
import sys

import tanitad
import refc_v3_train  # noqa: F401
import train_p8_occupancy  # noqa: F401
import refit_perception_thresholds  # noqa: F401
import mine_join_label_defects  # noqa: F401
from tanitad.eval import detection_metrics, detection_nms, detection_zh  # noqa: F401
from tanitad.data import join_label_hygiene  # noqa: F401
from tanitad.models import map_head_hires  # noqa: F401

BS = chr(92)
bad, n, dirs = [], 0, set()
for name, m in list(sys.modules.items()):
    f = getattr(m, "__file__", None)
    if not f:
        continue
    n += 1
    fl = f.replace(BS, "/").lower()
    if fl.startswith("g:/") or "meine ablage" in fl:
        bad.append((name, f))
    if name.split(".")[0] in ("tanitad", "refc_v3_train", "train_p8_occupancy", "refit_perception_thresholds", "mine_join_label_defects"):
        dirs.add(os.path.dirname(f).replace(BS, "/"))
print("tanitad.__file__ =", tanitad.__file__)
print("modules with a file:", n, "| imported from G: or 'Meine Ablage':", len(bad), bad[:5])
print("programme module dirs:", sorted(dirs)[:8])
assert not bad
print("AUDIT PASS: nothing imported from G:")
