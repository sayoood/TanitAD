#!/usr/bin/env python3
"""Tests for s1c_veto.py.  Prints `ZZTEST_PASS <n>` (exit 0) or `ZZTEST_FAIL ...` (exit 1).  Run in the tanitad venv:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_s1c_veto.py

REAL-DATA tests (~35 s per veto read): the raster source is the dry-run forward of S1's banked H1-s0 head (out/dryrun/forward_cpu/head_pred_H1-s0.npz; absent => those tests are skipped and NOT counted).
  * validity 1/0/2/3/4/5 pass, the instrument identity (V-GT, O-DAC vs the banked AUXVETO_READ.json) is exactly 0, the row is the row auxveto_read.decide gives
  * amendment A1's identity: an expected IoU that is off by 1e-4 => the V-S1C row is INVALID (row_if_valid kept), the arm blocks are still reported
  * an all-drivable raster vetoes nothing => its Delta is EXACTLY 0.0 with CI [0, 0]
  * an all-undrivable raster vetoes EVERY proposal; the pick rule then keeps B's pick => Delta exactly 0.0 (the fail-safe), row AV-NULL / AV-CEILING-ONLY
  * SOURCE mutations (mask threshold, the A1 tolerance, the all_pass gate) must go RED
"""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s1c_veto as V  # noqa: E402

AV, R = V.AV, V.R
SRC = (HERE / "s1c_veto.py").read_text(encoding="utf-8")
N_OK, FAILS = 0, []
PRED = HERE / "out" / "dryrun" / "forward_cpu" / "head_pred_H1-s0.npz"
H1S0_P1 = 0.4957035913034702                                 # S1's banked H1-s0 C3 P1 [0, 40) (raw/2026-10-15-s1-read/instrument/perception_navtest.json)


def check(name, cond, detail=""):
    global N_OK
    if cond:
        N_OK += 1
    else:
        FAILS.append(f"{name} {detail}".strip())
        print("  FAIL", name, detail, flush=True)


def mutated(old, new):
    assert SRC.count(old) == 1, f"mutation anchor not unique: {old!r} x{SRC.count(old)}"
    m = types.ModuleType("s1c_veto_mut")
    m.__file__ = str(HERE / "s1c_veto.py")
    sys.modules["s1c_veto_mut"] = m
    exec(compile(SRC.replace(old, new), str(HERE / "s1c_veto.py"), "exec"), m.__dict__)
    return m


def quiet(*a, **k):
    return None


def main():
    if not PRED.is_file():
        print(f"ZZTEST_SKIP real-data tests: {PRED} absent (run the dry run first)")
        return 0
    w, _ = AV.load_world()
    masks, sha = V.load_head_masks(str(PRED), w.token)
    check("load_head_masks_covers_every_token", len(masks) == 1058 and all(m.shape == (40, 32) and m.dtype == bool for m in masks.values()))
    # ---- the valid read
    out = V.run_veto(masks, "V-T", H1S0_P1, log=quiet)
    val = out["validity"]
    check("valid_all_six_checks_pass", val["all_pass"] and all(val[k]["ok"] for k in ("1_B_reproduces_pick", "0_population_identity", "2_A1_mask_iou_equals_harness_P1", "3_zero_control",
                                                                                        "4_odac_bounds_rasters", "5_yflip_mutation_below_vgt")))
    check("a1_identity_is_exact", val["2_A1_mask_iou_equals_harness_P1"]["abs_diff"] <= 1e-12, str(val["2_A1_mask_iou_equals_harness_P1"]))
    check("instrument_identity_is_zero", out["instrument_identity"]["ok"] and out["instrument_identity"]["max_abs_diff"] == 0.0, str(out["instrument_identity"]["max_abs_diff"]))
    ref = AV.decide({"lo": out["arms"]["V-T"]["d_pdms"]["lo"], "hi": out["arms"]["V-T"]["d_pdms"]["hi"], "point": out["arms"]["V-T"]["d_pdms"]["point"]}, {"lo": out["arms"]["V-GT"]["d_pdms"]["lo"]})["row"]
    check("row_equals_auxveto_read_decide", out["decision"]["row"] == ref, f"{out['decision']['row']} vs {ref}")
    check("decision_quotes_the_prereg_cell", out["decision"]["quoted_row"]["next"] == AV.prereg_next_arm(ref))
    check("vgt_block_equals_the_banked_read", all(abs(out["arms"]["V-GT"]["d_pdms"][k] - AV.cj(V.json.load(open(V.BANKED_READ)))["arms"]["V-GT"]["d_pdms"][k]) == 0.0 for k in V.BLOCK_KEYS))
    check("strategic_is_paused_in_the_output", "PAUSED" in out["strategic"])
    check("estimator_names_its_blindness", "TRAINING variance" in out["estimator"]["question_answered"] and out["estimator"]["resamples"] == 10000 and out["estimator"]["seed"] == 20260927)
    # ---- A1: an IoU that does not reproduce => INVALID, recorded
    bad = V.run_veto(masks, "V-T", H1S0_P1 + 1e-4, log=quiet)
    check("a1_identity_miss_is_INVALID", (not bad["validity"]["all_pass"]) and bad["decision"]["row"] == "INVALID" and bad["decision"]["row_if_valid"] == ref
          and bad["decision"]["invalid_checks"] == ["2_A1_mask_iou_equals_harness_P1"], str(bad["decision"].get("invalid_checks")))
    check("a1_identity_miss_still_reports_the_arm_blocks", bad["arms"]["V-T"]["d_pdms"]["point"] == out["arms"]["V-T"]["d_pdms"]["point"])
    # ---- degenerate rasters
    alld = {t: np.ones((40, 32), bool) for t in masks}
    iou_all = AV.iou_pooled(np.stack([alld[t] for t in w.token]), np.stack([AV.gt_mask_to_2m(AV.load_gt_masks(w)[t]) for t in w.token]))
    z = V.run_veto(alld, "V-ALLDRIVABLE", float(iou_all), log=quiet)
    d = z["arms"]["V-ALLDRIVABLE"]["d_pdms"]
    check("all_drivable_raster_vetoes_nothing_delta_exactly_0", z["validity"]["all_pass"] and d["point"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0 and z["arms"]["V-ALLDRIVABLE"]["z_recovery"]["ncdac_count"] == 0,
          str(d))
    check("all_drivable_is_not_AV_GO", z["decision"]["row"] in ("AV-NULL", "AV-CEILING-ONLY"), z["decision"]["row"])
    none_ = {t: np.zeros((40, 32), bool) for t in masks}
    n = V.run_veto(none_, "V-NONE", 0.0, log=quiet)
    dn, bn = n["arms"]["V-NONE"]["d_pdms"], n["arms"]["V-NONE"]
    check("all_undrivable_raster_vetoes_every_proposal_and_B_stands_delta_exactly_0", n["validity"]["all_pass"] and bn["all_vetoed_tokens"] == {"Z": 529, "N": 529} and bn["changed_picks"] == {"Z": 0, "N": 0}
          and dn["point"] == 0.0 and dn["lo"] == 0.0 and dn["hi"] == 0.0, f"{dn['point']} {bn['all_vetoed_tokens']} {bn['changed_picks']}")
    check("all_undrivable_row_is_not_HARM_not_GO", n["decision"]["row"] in ("AV-NULL", "AV-CEILING-ONLY"), n["decision"]["row"])
    # ---- source mutations
    for label, old, new in (("S1 the mask threshold logit > 0 becomes > 3", "AV.pred_mask(lg[idx[t]], None)", "AV.pred_mask(lg[idx[t]] - 3.0, None)"),
                            ("S2 the A1 tolerance becomes 1.0", "IOU_TOL = 1e-6 ", "IOU_TOL = 1.0 "),
                            ("S3 the validity gate is always true", 'val["all_pass"] = all(v["ok"] for v in val.values() if isinstance(v, dict) and "ok" in v)', 'val["all_pass"] = True')):
        try:
            mm = mutated(old, new)
            if label.startswith("S1"):
                m2, _ = mm.load_head_masks(str(PRED), w.token)
                r = mm.run_veto(m2, "V-T", H1S0_P1, log=quiet)
                red = not r["validity"]["all_pass"]
            else:
                r = mm.run_veto(masks, "V-T", H1S0_P1 + 1e-4, log=quiet)
                red = r["decision"]["row"] != "INVALID"             # the mutated gate fails to flag the 1e-4 miss => this suite's INVALID check goes RED
        except Exception:
            red = True
        check(f"mutation_RED: {label}", red)
    if FAILS:
        print(f"ZZTEST_FAIL {len(FAILS)} failed, {N_OK} passed: " + " | ".join(FAILS))
        return 1
    print(f"ZZTEST_PASS {N_OK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
