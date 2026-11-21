#!/usr/bin/env python3
"""run_mutations_lf5.py -- every mutation of the LF5 tool must turn tests/test_lf5.py RED (>= 1 failing test). The mutant is written into the mutation tree (F:/TanitAD-scratch/motion_c/lf5_build/mut/tree) and tested with LF5_TOOL / LF5_TREE.
Writes logs/MUTATIONS_LF5.txt ("MUTATIONS ALL RED" on success).   usage: python -I run_mutations_lf5.py"""
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
TOOL = "F:/TanitAD-scratch/motion_c/lf5_read/tree/stack/scripts/v8f_t1_eval_ri.py"
MT = "F:/TanitAD-scratch/motion_c/lf5_build/mut/tree"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MUT = {
    "b1_offset_on_vhat": ("v_hat = v_log * np.exp(self.noise * eps)", "v_hat = v_log * np.exp(self.noise * eps) + 0.001"),
    "b2_draw_only_where_finite": ("eps = self.rng.standard_normal(len(v_log))", "eps = np.zeros(len(v_log)); _f = np.isfinite(v_log); eps[_f] = self.rng.standard_normal(int(_f.sum()))"),
    "b3_noise_allowed_on_heads": ('if a.level_floor != "on" or getattr(a, "floor_vhat_source", None) != "logged":', "if False:"),
    "b4_sigma_without_seed_allowed": ("if nz_ is not None and float(nz_) > 0.0 and nzs_ is None:", "if False:"),
    "b5_noise_path_only_when_sigma_positive": ("self.noise_path = self.noise_seed is not None", "self.noise_path = self.noise > 0.0"),
    "b6_header_does_not_record_noise": ('header["floor_vhat_noise"], header["floor_vhat_noise_seed"] = floor.noise, floor.noise_seed', "pass"),
    "b7_one_generator_per_call": ("self.rng = np.random.default_rng(self.noise_seed) if self.noise_path else None", "self.rng = None"),
    "b8_sigma_ignored": ("np.exp(self.noise * eps)", "np.exp(0.1 * eps)"),
}


def main():
    src = open(TOOL, encoding="utf-8", newline="").read()
    L, red = [], 0
    for name, (old, new) in MUT.items():
        assert src.count(old) == 1, (name, src.count(old))
        p = f"{MT}/stack/scripts/v8f_t1_eval_ri.py"
        open(p, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        env = dict(os.environ, LF5_TOOL=p, LF5_TREE=MT, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", "F:/TanitAD-scratch/motion_c/lf5_build/pt/mut", f"{PK}/tests/test_lf5.py"], env=env, capture_output=True, text=True)
        last = [x for x in r.stdout.splitlines() if x.strip()][-1] if r.stdout.strip() else r.stderr[-200:]
        is_red = r.returncode != 0
        red += is_red
        L.append(f"{'RED' if is_red else 'GREEN (NOT CAUGHT)'}  {name}: {last}")
        print(L[-1], flush=True)
    open(f"{PK}/logs/MUTATIONS_LF5.txt", "w", encoding="utf-8", newline="\n").write(("MUTATIONS ALL RED\n" if red == len(MUT) else "MUTATIONS NOT ALL RED\n") + f"{red} of {len(MUT)} mutations of the LF5 tool turn tests/test_lf5.py RED\n" + "\n".join(L) + "\n")
    return 0 if red == len(MUT) else 1


if __name__ == "__main__":
    sys.exit(main())
