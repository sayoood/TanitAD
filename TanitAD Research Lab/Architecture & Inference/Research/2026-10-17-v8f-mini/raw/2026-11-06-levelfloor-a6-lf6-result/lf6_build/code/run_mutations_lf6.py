#!/usr/bin/env python3
"""run_mutations_lf6.py -- every mutation of the LF6 tool must turn tests/test_lf6.py RED (>= 1 failing test). The mutant is written into the mutation tree (F:/TanitAD-scratch/motion_c/lf6_build/mut/tree) and tested with LF6_TOOL / LF6_TREE.
Writes logs/MUTATIONS_LF6.txt ("MUTATIONS ALL RED" on success).   usage: python -I run_mutations_lf6.py"""
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
TOOL = "F:/TanitAD-scratch/motion_c/lf6_read/tree/stack/scripts/v8f_t1_eval_ri.py"
MT = "F:/TanitAD-scratch/motion_c/lf6_build/mut/tree"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MUT = {
    "c1_delta_term_deleted": [("cap_o = v_o + float(delta)", "cap_o = v_o + 1e9")],
    "c2_clip_removed": [("return np.minimum(s_f, np.asarray(S, np.float64)[:, 1:] + float(delta) * t_hi), dterm_k", "return s_f, dterm_k"),
                        ("s_f = np.minimum(np.where(use_b[:, None], s_b, s_ramp), s_k + delta * op.T_KNOTS[None, :])", "s_f = np.where(use_b[:, None], s_b, s_ramp)")],
    "c3_v_o_lagged_by_one_interval": [("v_o = np.diff(np.asarray(S, np.float64), axis=1) / op.DT", "_d = np.diff(np.asarray(S, np.float64), axis=1); v_o = np.concatenate([_d[:, :1], _d[:, :-1]], axis=1) / op.DT")],
    "c4_ramp_term_dropped": [("tr = np.minimum(t, np.maximum(c - v0p, 0.0) / op.A_F)", "tr = np.minimum(t, 0.0 * c)")],
    "c5_dterm_never_true": [("dterm_k = (cap_o < vf_c) & (cap_o < v0p + t_hi)", "dterm_k = (cap_o < vf_c) & (cap_o < 0.0)")],
    "c6_bounded_path_never_used": [("use_b = dterm_k.any(axis=1)", "use_b = np.zeros(len(dterm_k), bool)")],
    "c7_none_seed_key_written_again": [('m["label"], m["floor_vhat_noise"] = "CEILING", self.noise\n            if self.noise_seed is not None:', 'm["label"], m["floor_vhat_noise"] = "CEILING", self.noise\n            if True:')],
    "c8_cap_allowed_with_floor_off": [('        if a.level_floor != "on":\n            raise SystemExit("REFUSED: --floor-lift-cap belongs', '        if False:\n            raise SystemExit("REFUSED: --floor-lift-cap belongs')],
    "c9_negative_cap_allowed": [("float(lc_) >= 0.0", "float(lc_) >= -5.0")],
    "c10_header_does_not_record_cap": [('            header["floor_lift_cap"] = floor.lift_cap', "            pass")],
    "c11_sp_records_s": [('"s": s_k.copy(), "sp": s_new.copy()}', '"s": s_k.copy(), "sp": s_k.copy()}')],
    "c12_default_cap_when_flag_absent": [("self.lift_cap = None if lift_cap is None else float(lift_cap)", "self.lift_cap = 1.0 if lift_cap is None else float(lift_cap)")],
    "c13_delta_scaled_in_v_o_term": [("cap_o = v_o + float(delta)", "cap_o = v_o + 0.5 * float(delta)")],
    "c14_bound_uses_s_f_scalar_vf_uncapped": [("v_fb = np.where(band, v_f, 0.0)", "v_fb = np.where(band, kappa * v_hat, 0.0)")],
}


def main():
    src = open(TOOL, encoding="utf-8", newline="").read()
    L, red = [], 0
    for name, pairs in MUT.items():
        s = src
        for old, new in pairs:
            assert s.count(old) == 1, (name, old[:70], s.count(old))
            s = s.replace(old, new)
        p = f"{MT}/stack/scripts/v8f_t1_eval_ri.py"
        open(p, "w", encoding="utf-8", newline="\n").write(s)
        env = dict(os.environ, LF6_TOOL=p, LF6_TREE=MT, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", "F:/TanitAD-scratch/motion_c/lf6_build/pt/mut", f"{PK}/tests/test_lf6.py"], env=env, capture_output=True, text=True)
        last = [x for x in r.stdout.splitlines() if x.strip()][-1] if r.stdout.strip() else r.stderr[-200:]
        is_red = r.returncode != 0
        red += is_red
        L.append(f"{'RED' if is_red else 'GREEN (NOT CAUGHT)'}  {name}: {last}")
        print(L[-1], flush=True)
    open(f"{PK}/logs/MUTATIONS_LF6.txt", "w", encoding="utf-8", newline="\n").write(("MUTATIONS ALL RED\n" if red == len(MUT) else "MUTATIONS NOT ALL RED\n") + f"{red} of {len(MUT)} mutations of the LF6 tool turn tests/test_lf6.py RED\n" + "\n".join(L) + "\n")
    return 0 if red == len(MUT) else 1


if __name__ == "__main__":
    sys.exit(main())
