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
    return 0 if (not bad and mbad) else 1


if __name__ == "__main__":
    raise SystemExit(main())
