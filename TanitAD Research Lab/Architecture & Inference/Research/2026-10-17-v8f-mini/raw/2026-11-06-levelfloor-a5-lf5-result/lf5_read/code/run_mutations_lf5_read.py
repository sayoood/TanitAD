#!/usr/bin/env python3
"""run_mutations_lf5_read.py -- every mutation of lf5_read.py's label logic / table / flags / instruments must turn tests/test_lf5_read.py RED. The mutant is a copy of the reader under F:/TanitAD-scratch/motion_c/lf5_read/pt/mut (it finds its siblings by fixed paths).
Writes logs/mutations_lf5_read.log ("MUTATIONS ALL RED" on success).   usage: python -I run_mutations_lf5_read.py"""
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MD = "F:/TanitAD-scratch/motion_c/lf5_read/pt/mut"
MUT = {
    "m1_sigma_star_is_the_smallest": ("\"sigma_star\": max(ok) if ok else None", "\"sigma_star\": min(ok) if ok else None"),
    "m2_sigma_zero_signals_levers": ("if sstar is not None and sstar >= 0.10:", "if sstar is not None and sstar >= 0.0:"),
    "m3_dominant_needs_the_wrong_label": ("if lf_label == \"D-HOLDS\":", "if lf_label == \"P-HOLDS-G-OR-D-FAILS\":"),
    "m4_harms_signals_levers": ("\"case\": \"OPERATOR-IS-THE-PROBLEM\", \"counted_dominant\": False, \"estimator_levers_signalled\": False", "\"case\": \"OPERATOR-IS-THE-PROBLEM\", \"counted_dominant\": False, \"estimator_levers_signalled\": True"),
    "m5_short_case_renamed": ("PROGRESS-FLOOR-FORM-EXHAUSTED", "OPERATOR-IS-THE-PROBLEM"),
    "m6_composite_first": ("for comp in (False, True):", "for comp in (True, False):"),
    "m7_noise_flags_dropped": ("if source == \"logged\" and sigma and sigma > 0:", "if False:"),
    "m8_heads_arm_last_not_first": ("(\"RILF100\", \"reindexed\", \"heads\", None), (\"RILFO100\", \"reindexed\", \"logged\", 0.0)", "(\"RILFO100\", \"reindexed\", \"logged\", 0.0), (\"RILF100\", \"reindexed\", \"heads\", None)"),
    "m9_signed_error_not_absolute": ("\"mae_mps\": float(np.abs(e).mean())", "\"mae_mps\": float(e.mean())"),
    "m10_overshoot_share_inverted": ("\"share_vhat_exceeds_logged\": float((e > 0).mean())", "\"share_vhat_exceeds_logged\": float((e < 0).mean())"),
    "m11_kappa_flag_075": ("SEEDS, KAPPA, KAPPA_FLAG, BASE_SMOOTH, NOISE_SEED = (0, 1), 1.0, \"1.0\"", "SEEDS, KAPPA, KAPPA_FLAG, BASE_SMOOTH, NOISE_SEED = (0, 1), 1.0, \"0.75\""),
}


def main():
    src = open(f"{PK}/code/lf5_read.py", encoding="utf-8", newline="").read()
    os.makedirs(MD, exist_ok=True)
    L, red = [], 0
    for name, (old, new) in MUT.items():
        assert src.count(old) == 1, (name, src.count(old))
        p = f"{MD}/lf5_read.py"
        open(p, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        env = dict(os.environ, LF5_READER=p, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", "F:/TanitAD-scratch/motion_c/lf5_read/pt/mutrun", f"{PK}/tests/test_lf5_read.py"], env=env, capture_output=True, text=True)
        last = [x for x in r.stdout.splitlines() if x.strip()][-1] if r.stdout.strip() else r.stderr[-200:]
        is_red = r.returncode != 0
        red += is_red
        L.append(f"{'RED' if is_red else 'GREEN (NOT CAUGHT)'}  {name}: {last}")
        print(L[-1], flush=True)
    open(f"{PK}/logs/mutations_lf5_read.log", "w", encoding="utf-8", newline="\n").write(("MUTATIONS ALL RED\n" if red == len(MUT) else "MUTATIONS NOT ALL RED\n") + f"{red} of {len(MUT)} mutations of lf5_read.py turn tests/test_lf5_read.py RED\n" + "\n".join(L) + "\n")
    return 0 if red == len(MUT) else 1


if __name__ == "__main__":
    sys.exit(main())
