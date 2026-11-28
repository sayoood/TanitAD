"""Pins `proxy_eval.fam_direction` / `fam_adverse` after the 2026-09-28 defect: the direction rule tested the
SUBSTRING "acc" (meant for "accuracy"), which also matched `accel_mae_mps2`, so a LOWER (better) acceleration error was
flagged as an adverse separation and flipped SPEC Amendment 8 from ADOPT to NOT PROVEN.

Expectations are LITERALS (never an expression over the code under test). The mutation arm re-inserts the historical
defect into a copy of the module and requires this file's checks to FAIL on it.

    python test_fam_direction.py            # runs the checks, then the mutation arm
"""
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# every longitudinal / lateral component families6 emitted on the A8 seams (2026-09-28), with its correct direction
EXPECTED = {
    "accel_mae_mps2": "lower",
    "speed_mae_mps": "lower", "speed_rmse_mps": "lower", "along_mae_m": "lower",
    "speed_bias_mps": "magnitude", "along_bias_m": "magnitude", "along_final_bias_m": "magnitude",
    "ego_progress.progress_error_mean": "lower", "ego_progress.under_progress_rate": "lower",
    "ego_progress.progress_ratio_mean": "closer_to_1", "ego_progress.progress_ratio_median": "closer_to_1",
    "target_speed_acc.within_0.5_mps": "higher", "target_speed_acc.within_1.0_mps": "higher",
    "target_speed_acc.within_2.0_mps": "higher",
    "cross_mae_m": "lower", "cross_final_mae_m": "lower", "curvature_mae_1pm": "lower", "heading_mae_deg": "lower",
    "yaw_rate_mae_degps": "lower", "cross_bias_m": "magnitude", "curvature_bias_1pm": "magnitude",
    # 2026-11-20 POLARITY FIX: lead-agent safety margins are higher-is-safer (any family prefix; the battery's mean_ form too)
    "min_ttc_s": "higher", "headway_min_m": "higher", "time_gap_min_s": "higher",
    "longitudinal.min_ttc_s": "higher", "lead.headway_min_m": "higher", "mean_min_ttc_s": "higher", "mean_time_gap_min_s": "higher",
    "lead_viol": "lower", "lead_at_risk": "lower",           # NOT margins: a violation / at-risk rate is lower-is-better (guards against a substring rule)
}


def fam(components: dict, fam_name: str = "longitudinal") -> dict:
    return {"families": {"refcv6": {fam_name: {"ci": {"components": components}}}}}


def run_checks(PE) -> list:
    bad = [f"direction {k}: got {PE.fam_direction(k)!r}, want {v!r}" for k, v in EXPECTED.items()
           if PE.fam_direction(k) != v]
    # fam_adverse(first, second) flags the SECOND argument when it is entirely on the worse side of the first
    # 1. the A8 case, verbatim intervals: OFF accel [1.1304, 2.5361], ON [0.5995, 0.7825] -> ON is BETTER: not adverse
    r = PE.fam_adverse(fam({"accel_mae_mps2": {"lo": 1.1304, "hi": 2.5361}}),
                       fam({"accel_mae_mps2": {"lo": 0.5995, "hi": 0.7825}}))
    if r["adverse"]:
        bad.append(f"A8 case: a lower accel error was flagged adverse: {r['adverse']}")
    # 2. the reverse must be flagged: the second arm's accel error entirely HIGHER
    r = PE.fam_adverse(fam({"accel_mae_mps2": {"lo": 0.5995, "hi": 0.7825}}),
                       fam({"accel_mae_mps2": {"lo": 1.1304, "hi": 2.5361}}))
    if [x["metric"] for x in r["adverse"]] != ["accel_mae_mps2"]:
        bad.append(f"a higher accel error was NOT flagged adverse: {r['adverse']}")
    # 3. a higher-is-better component: the second arm entirely LOWER is adverse
    r = PE.fam_adverse(fam({"target_speed_acc.within_1.0_mps": {"lo": 0.6, "hi": 0.7}}),
                       fam({"target_speed_acc.within_1.0_mps": {"lo": 0.3, "hi": 0.4}}))
    if [x["metric"] for x in r["adverse"]] != ["target_speed_acc.within_1.0_mps"]:
        bad.append(f"a lower within-band rate was NOT flagged adverse: {r['adverse']}")
    # 3b. the polarity fix: a RISE of a safety margin is NOT adverse, a FALL is (verbatim C1-vs-C0 min_ttc_s interval is a rise)
    for k in ("min_ttc_s", "headway_min_m", "time_gap_min_s"):
        r = PE.fam_adverse(fam({k: {"lo": 13.5, "hi": 13.8}}), fam({k: {"lo": 13.9, "hi": 14.2}}))
        if r["adverse"]:
            bad.append(f"a RISE of {k} was flagged adverse: {r['adverse']}")
        r = PE.fam_adverse(fam({k: {"lo": 13.9, "hi": 14.2}}), fam({k: {"lo": 13.5, "hi": 13.8}}))
        if [x["metric"] for x in r["adverse"]] != [k]:
            bad.append(f"a FALL of {k} was NOT flagged adverse: {r['adverse']}")
    # 4. the skip list must match DOTTED components (properties of the human future are never compared)
    r = PE.fam_adverse(fam({"ego_progress.gt_progress_mean_m": {"lo": 1.0, "hi": 2.0}}),
                       fam({"ego_progress.gt_progress_mean_m": {"lo": 5.0, "hi": 6.0}}))
    if r["components_compared"] != 0 or r["adverse"]:
        bad.append(f"dotted skip-list component was compared: {r}")
    return bad


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, HERE)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    real = os.path.join(HERE, "proxy_eval.py")
    bad = run_checks(load(real, "proxy_eval_under_test"))
    print("REAL module:", "PASS" if not bad else "FAIL", *bad, sep="\n  ")
    src = open(real, encoding="utf-8").read()
    old = 'if "within_" in m or "accuracy" in m or "kappa" in m:'
    assert src.count(old) == 1, "the fixed direction line is not present verbatim"
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "proxy_eval_mutant.py")
        open(mp, "w", encoding="utf-8").write(src.replace(old, 'if "within_" in m or "acc" in m or "kappa" in m:'))
        mbad = run_checks(load(mp, "proxy_eval_mutant"))
    print("MUTANT (the historical 'acc' substring rule):", "RED as required" if mbad else "GREEN -- the test is blind",
          *mbad[:2], sep="\n  ")
    # polarity mutation arm: revert ONE of the three margins -> this file's checks must go RED, for each of the three
    tup = 'SAFETY_MARGIN_HIGHER = ("min_ttc_s", "headway_min_m", "time_gap_min_s")'
    assert src.count(tup) == 1, "the polarity tuple is not present verbatim"
    pol_ok = True
    for drop in ("min_ttc_s", "headway_min_m", "time_gap_min_s"):
        keep = tuple(x for x in ("min_ttc_s", "headway_min_m", "time_gap_min_s") if x != drop)
        with tempfile.TemporaryDirectory() as td:
            mp = os.path.join(td, "proxy_eval_pol_mutant.py")
            open(mp, "w", encoding="utf-8").write(src.replace(tup, "SAFETY_MARGIN_HIGHER = " + repr(keep)))
            pbad = run_checks(load(mp, "proxy_eval_pol_mutant_" + drop))
        print(f"POLARITY MUTANT (revert {drop}):", "RED as required" if pbad else "GREEN -- the test is blind", *pbad[:1], sep="\n  ")
        pol_ok = pol_ok and bool(pbad)
    return 0 if (not bad and mbad and pol_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
