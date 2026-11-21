"""ship_r8_sub_tac.py -- the TACTICAL rows (SHIP-R8 clause 5) of one arm through the registered tac_retime_r8 device, re-pointed.  CPU only.
NEW file (EvalFlyWheel 2026-10-26).  tac_retime_r8.py is IMPORTED and NOT edited: pf_lane.configure re-points the battery modules at the REGISTERED
per-frame table (as the registered lane did), then two module attributes are set -- TAC.DEP = this arm's package directory (written by
ship_r8_sub_forward.py: DEPLOY8_POS_<tree>_s{0,1}.npz under the consumer contract with meta arm = <ARM>, + that run's FORWARD file under the
battery's name for K7, + the registered FORWARD_DEPLOY file for K6) and TAC.SCORED_ARM = <ARM>.  TAC.run then scores exactly as registered.

    python ship_r8_sub_tac.py --arm-dir <out>/arms/<tag>/<ARM> --arm <ARM> --tree step48852ema_pf
For --arm RETIME_DEPLOY8 the identity control is checked: the clause-5 cells, status and controls equal the REGISTERED TAC_<tree>.json exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ship_r8_sub_common as C  # noqa: E402

PFPKG = C.PF_HOLD / "pfpkg"
REG_PF = Path("C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tree", default="step48852ema_pf")
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[sub-tac] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    C.check_amendment_seals()
    C.setup_paths()
    sys.path.insert(0, str(C.PF_HOLD / "code"))
    import pf_lane as PL
    reg = PL.read_registered(REG_PF)
    PL.verify_installed(PFPKG, reg)
    M = PL.mods()
    PL.configure(M, PFPKG, reg)
    TAC, D8 = M["TAC"], M["D8"]
    ad = Path(a.arm_dir)
    for s in (0, 1):
        z = ad / f"DEPLOY8_POS_{a.tree}_s{s}.npz"
        if not z.exists():
            raise SystemExit(f"[sub-tac] {z} does not exist (run ship_r8_sub_forward.py first)")
    reg_tac_path = TAC.DEP / f"TAC_{a.tree}.json"                     # the REGISTERED file (pf_lane points TAC.DEP at the pf package)
    reg_tac = json.loads(reg_tac_path.read_text(encoding="utf-8")) if reg_tac_path.exists() else None
    TAC.DEP, TAC.SCORED_ARM = ad, a.arm
    rc = TAC.run(a.tree)
    out = ad / f"TAC_{a.tree}.json"
    j = json.loads(out.read_text(encoding="utf-8"))                   # assert on the ARTIFACT
    assert j["clause5"]["arm"] == a.arm
    rec = {"arm": a.arm, "tac_file": str(out), "tac_sha256": C.sha256_file(out), "clause5_status": j["clause5"]["status"], "controls_ALL_PASS": j["controls"]["ALL_PASS"],
           "K7_crosscheck": j["controls"].get("K7_crosscheck_vs_FORWARD_DEPLOY8_determinism_only"), "rc": rc}
    if a.arm == "RETIME_DEPLOY8":
        if reg_tac is None:
            rec["identity_vs_registered_TAC"] = {"pass": False, "reason": f"{reg_tac_path} absent"}
        else:
            mism = D8.exact_equal(j["clause5"]["cells"], reg_tac["clause5"]["cells"])
            ctl = {k: j["controls"].get(k) for k in j["controls"] if not k.endswith("_detail") and k != "note"}
            ctl_reg = {k: reg_tac["controls"].get(k) for k in reg_tac["controls"] if not k.endswith("_detail") and k != "note"}
            mism_c = D8.exact_equal(ctl, ctl_reg)
            rec["identity_vs_registered_TAC"] = {"registered_file": str(reg_tac_path), "registered_sha256": C.sha256_file(reg_tac_path),
                                                 "cells_n_mismatches": len(mism), "first_mismatches": mism[:10], "status_equal": j["clause5"]["status"] == reg_tac["clause5"]["status"],
                                                 "controls_n_mismatches": len(mism_c), "controls_first_mismatches": mism_c[:10],
                                                 "pass": bool(not mism and not mism_c and j["clause5"]["status"] == reg_tac["clause5"]["status"])}
    (ad / f"TAC_RECORD_{a.tree}.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "identity_vs_registered_TAC"}), rec.get("identity_vs_registered_TAC", {}).get("pass"), flush=True)
    print(f"ZZSUBTACZZ {a.arm} {j['clause5']['status']} controls={j['controls']['ALL_PASS']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
