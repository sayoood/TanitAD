#!/usr/bin/env python3
"""--nc-at-fault must FAIL LOUDLY without the teacher's at-fault export, and must change NOTHING but the NC key with it.

On ONE real navtest sample (chunk 0's first token of eval/validate_slow_labels_v4.py's queue), in one process:
  RED   label_one(nc_at_fault=True) BEFORE install_teacher_at_fault(): must raise (no silent fallback to the raw
        collision event -- the failure this flag could otherwise hide)
  OFF   label_one(nc_at_fault=False) BEFORE the install: the reference line
  ON    install, then label_one(nc_at_fault=True): every target equals OFF's once the change is undone (raw collision
        back in the NC key), the raw value is kept under teacher_collision.*, and ndiff is unchanged (the new key is
        excluded from the discrimination rule)
  OFF2  label_one(nc_at_fault=False) AFTER the install: identical to OFF -- the wrapper leaves the v3 path's labels alone
Prints ZZNCAF_SELFTEST_OK / _FAIL.
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))
import validate_slow_labels_v4 as V  # noqa: E402


def main() -> int:
    import onpolicy_label as OL
    import onpolicy_label_v4 as L4
    chunk = sorted(glob.glob(os.path.join(V.Q, f"props_r0_{V.CKPT_STEP:06d}_{0:010d}.jsonl*")))[0]
    r = json.loads(open(chunk, encoding="utf-8").readline())
    S = OL.Scorer(0, 2)
    sc = S.scenarios(r["log_name"], [r["token"]])[r["token"]]
    spec = L4.SlowSpec(frac=1.0)
    res = {"token": r["token"]}
    try:
        L4.label_one(S, sc, r, spec, "selftest", nc_at_fault=True)
        res["RED_raised"] = False
    except RuntimeError as exc:
        res["RED_raised"] = "no teacher at-fault event" in str(exc)
    off = json.loads(L4.label_one(S, sc, r, spec, "selftest")["line"])
    print(f"  {L4.install_teacher_at_fault()}", flush=True)
    on = json.loads(L4.label_one(S, sc, r, spec, "selftest", nc_at_fault=True)["line"])
    off2 = json.loads(L4.label_one(S, sc, r, spec, "selftest")["line"])
    strip = lambda d: {k: v for k, v in d.items() if k not in ("sec", "at", "labeller")}  # noqa: E731
    res["ON_targets_equal_OFF_once_undone"] = all(V.as_v3(a) == b for a, b in zip(on["targets"], off["targets"]))
    res["ON_raw_kept"] = all(a.get("teacher_collision.NuPlanCollision.info") == b["collision.NuPlanCollision.info"]
                             for a, b in zip(on["targets"], off["targets"]))
    res["ON_nc_changed_slots"] = sum(a["collision.NuPlanCollision.info"] != b["collision.NuPlanCollision.info"]
                                     for a, b in zip(on["targets"], off["targets"]))
    res["ON_ndiff_equal"] = on["ndiff"] == off["ndiff"]
    res["ON_marked"] = on.get("nc_source") == "teacher_at_fault" and on["label_version"] == 4
    res["OFF2_equals_OFF"] = json.dumps(strip(off2)) == json.dumps(strip(off))
    ok = all(v for k, v in res.items() if k not in ("token", "ON_nc_changed_slots"))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "raw", f"e6_{V.NAME}", "nc_at_fault_selftest.json"), "w"), indent=1)
    print("ZZNCAF_SELFTEST_OK" if ok else "ZZNCAF_SELFTEST_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
