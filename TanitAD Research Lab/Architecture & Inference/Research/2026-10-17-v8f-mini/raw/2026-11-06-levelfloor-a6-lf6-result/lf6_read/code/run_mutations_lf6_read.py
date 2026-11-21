#!/usr/bin/env python3
"""run_mutations_lf6_read.py -- every mutation of lf6_read.py's rule constants / label logic / table / flags / instruments must turn tests/test_lf6_read.py RED. The mutant is a copy of the reader under F:/TanitAD-scratch/motion_c/lf6_read/pt/mut (it finds its siblings by fixed paths).
Includes one-character changes of the card's constants (the doses, kappa, the smoothing mode, the bound tolerance).
Writes logs/mutations_lf6_read.log ("MUTATIONS ALL RED" on success).   usage: python -I run_mutations_lf6_read.py"""
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MD = "F:/TanitAD-scratch/motion_c/lf6_read/pt/mut"
MUT = {
    "m01_delta_flag_one_char": ('DELTA_FLAG = {"05": "0.5", "10": "1.0"}', 'DELTA_FLAG = {"05": "0.6", "10": "1.0"}'),
    "m02_delta_value_one_char": ('DELTAS = {"05": 0.5, "10": 1.0}', 'DELTAS = {"05": 0.5, "10": 1.1}'),
    "m03_bound_tolerance_one_char": ("BOUND_TOL_M = 1e-6", "BOUND_TOL_M = 1e-5"),
    "m04_kappa_flag_one_char": ('SEEDS, KAPPA, KAPPA_FLAG, BASE_SMOOTH = (0, 1), 1.0, "1.0", "sg72"', 'SEEDS, KAPPA, KAPPA_FLAG, BASE_SMOOTH = (0, 1), 1.0, "1.1", "sg72"'),
    "m05_smooth_mode_one_char": ('SEEDS, KAPPA, KAPPA_FLAG, BASE_SMOOTH = (0, 1), 1.0, "1.0", "sg72"', 'SEEDS, KAPPA, KAPPA_FLAG, BASE_SMOOTH = (0, 1), 1.0, "1.0", "sg73"'),
    "m06_dominant_needs_the_wrong_label": ('hold = [d for d in REPORT_ORDER if lfb.get(d) == "D-HOLDS"]', 'hold = [d for d in REPORT_ORDER if lfb.get(d) == "D-HOLD"]'),
    "m07_harms_case_needs_only_one_dose": ('if all(x == "CEILING-HARMS" for x in labels):', 'if any(x == "CEILING-HARMS" for x in labels):'),
    "m08_larger_delta_named_first": ('REPORT_ORDER = ("05", "10")', 'REPORT_ORDER = ("10", "05")'),
    "m09_delta_05_rollouts_before_10": ('(("RILFB10", "reindexed", "heads", "10"), ("RILFB05", "reindexed", "heads", "05"),', '(("RILFB05", "reindexed", "heads", "05"), ("RILFB10", "reindexed", "heads", "10"),'),
    "m10_composite_first": ("for comp in (False, True):", "for comp in (True, False):"),
    "m11_overshoot_share_inverted": ('"share_vhat_exceeds_logged": float((e > 0).mean())', '"share_vhat_exceeds_logged": float((e < 0).mean())'),
    "m12_signed_error_not_absolute": ('"mae_mps": float(np.abs(e).mean())', '"mae_mps": float(e.mean())'),
    "m13_bound_never_fails": ('"bound_holds": bool(slack is not None and slack <= BOUND_TOL_M)', '"bound_holds": bool(slack is not None and slack <= 1.0)'),
    "m14_sp_strictly_above_s": ('"sp_ge_s": bool((sp >= s).all())', '"sp_ge_s": bool((sp > s).all())'),
    "m15_clears_case_reads_the_wrong_label": ('clears = [d for d in REPORT_ORDER if lfob.get(d) == "CEILING-CLEARS"]', 'clears = [d for d in REPORT_ORDER if lfob.get(d) == "CEILING-SHORT"]'),
    "m16_dterm_share_over_all_ticks": ('float(dt[bind].mean())', 'float(dt.mean())'),
    "m17_mixed_short_harms_not_case_4": ('if all(x in ("CEILING-SHORT", "CEILING-HARMS") for x in labels):', 'if all(x in ("CEILING-SHORT",) for x in labels):'),
    "m18_dose_key_kappa_wrong": ('LFR.KAPPA["B" + d], LFR.KAPPA_FLAG["B" + d] = KAPPA, KAPPA_FLAG', 'LFR.KAPPA["B" + d], LFR.KAPPA_FLAG["B" + d] = 0.75, KAPPA_FLAG'),
    "m19_dterm_unbound_not_checked": ('"dterm_only_on_bound_ticks": bool(not (dt & ~bind).any())', '"dterm_only_on_bound_ticks": True'),
}


def main():
    src = open(f"{PK}/code/lf6_read.py", encoding="utf-8", newline="").read()
    os.makedirs(MD, exist_ok=True)
    L, red = [], 0
    for name, (old, new) in MUT.items():
        assert src.count(old) == 1, (name, src.count(old))
        p = f"{MD}/lf6_read.py"
        open(p, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        env = dict(os.environ, LF6_READER=p, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", "F:/TanitAD-scratch/motion_c/lf6_read/pt/mutrun", f"{PK}/tests/test_lf6_read.py"], env=env, capture_output=True, text=True)
        last = [x for x in r.stdout.splitlines() if x.strip()][-1] if r.stdout.strip() else r.stderr[-200:]
        is_red = r.returncode != 0
        red += is_red
        L.append(f"{'RED' if is_red else 'GREEN (NOT CAUGHT)'}  {name}: {last}")
        print(L[-1], flush=True)
    open(f"{PK}/logs/mutations_lf6_read.log", "w", encoding="utf-8", newline="\n").write(("MUTATIONS ALL RED\n" if red == len(MUT) else "MUTATIONS NOT ALL RED\n") + f"{red} of {len(MUT)} mutations of lf6_read.py turn tests/test_lf6_read.py RED\n" + "\n".join(L) + "\n")
    return 0 if red == len(MUT) else 1


if __name__ == "__main__":
    sys.exit(main())
