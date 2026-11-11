"""SLOWWM beside B1 controls (BESIDE_RULES_SLOWWM.md, B1 (i)-(iii)); each has a red arm that must go RED. Writes ``$BESIDE_OUT/CONTROLS_B1B2.json`` (keys merged with the B2 controls of wmact_controls.py;
``all_pass`` = b1_i, b1_ii, b1_iii and b2 all pass). CPU, 2 threads, tier T0.
  b1_i    Permuted targets (a derangement across train clips) score eval R2 <= 0.02, speed and yaw rate, on the real R35H s0 arrays and on a planted basis arm's.
          Red arm: the same probe scored on its own train rows must read > 0.02.
  b1_ii   Planted target y = z1 . w + 0.1 eps (fixed w, seed 0) recovers eval R2 within 0.02 of its planted value, on both arms. Red arm: train targets rolled by half the rows must miss by > 0.02.
  b1_iii  Adapter route identity: an adapter arm's window latents through the B1 code (``lp_slowwm.prepare_arm_read`` -> ``decod_lib.arrays_from_ctx``) equal ``lp_lib.prepare_arm``'s bit for bit on 64 windows
          (window z1 and the rolled z1 and the targets), and on every window. Red arms: a 1e-6 perturbation is flagged; the rolled z1 at step k+1 differs; a basis model sent down lp_lib's own clip_z1 RAISES.
Train / eval for (i), (ii): the train clips are on box 2, so the controls use an episode-disjoint half split of the eval-141 windows (the control is about the probe machinery); ``decod_run.py`` repeats (i) and (ii)
on each arm's real train / eval arrays before it writes a row.
Usage: python decod_controls.py"""
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SLOWWM_READ_WORK", "F:/TanitAD-scratch/slowwm_beside_build/b1b2/planted")
import decod_lib as DL  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import controls_read_slowwm as CR  # noqa: E402  (planted basis / ubasis / run folder builders of the read package, imported only)

S, LR, LP, SH = DL.S, DL.LR, DL.LP, DL.SH
OUT = os.path.join(DL.OUT_DIR, "CONTROLS_B1B2.json")
R35H_DIR = f"{LR.PULLS}/ladder_R35H_s0"
KEYS = ("b1_i", "b1_ii", "b1_iii", "b2")


def merge(key, val):
    os.makedirs(DL.OUT_DIR, exist_ok=True)
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res.setdefault("tier", "T0")
    res[key] = val
    res["all_pass"] = all(res.get(k, {}).get("pass") is True for k in KEYS)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return res


def planted_slow_ctx():
    """A planted basis arm (R35H s0's checkpoint with wm_basis = slow, a planted basis and planted token means), read through the read package's own route."""
    fr = LR.load_eval_frames(f"{LP.FRAMES}/frames_s0.npz")
    B1 = CR.planted_basis("slow", 1)
    nms = CR.planted_nm(fr)
    ub1 = CR.planted_ubasis_file("1", fr, nms, B1)
    run = CR.planted_run("ok", "slow", "slow", "slow", B1["sha"])
    return S.prepare_arm_read("SLOW_s0", run, B1["path"], ub1)


def b1_i_ii(arms):
    t0 = time.time()
    out, ok = {}, True
    for tag, ctx in arms.items():
        tr, ev = DL.pseudo_split(DL.arrays_from_ctx(ctx, DL.load_tracks()))
        c = DL.controls_on_arrays(tr, ev)
        out[tag] = c
        ok &= c["pass"]
    i = {t: {k: c[k] for k in ("i_permuted_speed", "i_permuted_yaw")} | {"red_arm_score_on_train": c["red_arms"]["i_score_on_train"]} for t, c in out.items()}
    ii = {t: {"ii_planted": c["ii_planted"], "red_arm_misaligned": c["red_arms"]["ii_misaligned"]} for t, c in out.items()}
    pi = all(c["i_permuted_speed"]["ok"] and c["i_permuted_yaw"]["ok"] and (not c["red_arms"]["i_score_on_train"]["ok"]) for c in out.values())
    pii = all(c["ii_planted"]["ok"] and (not c["red_arms"]["ii_misaligned"]["ok"]) for c in out.values())
    return dict(pass_=bool(pi), arms=i, split="episode-disjoint half split of the eval-141 windows (machinery check)", wall_s=DL.stamp(t0)), dict(pass_=bool(pii), arms=ii, wall_s=DL.stamp(t0))


def b1_iii(mine):
    t0 = time.time()
    ref = LP.prepare_arm("R35H_s0")
    tracks, k = DL.load_tracks(), DL.k_steps()
    a, b = DL.arrays_from_ctx(mine, tracks, k), DL.arrays_from_ctx(ref, tracks, k)
    eq = lambda x, y: bool(np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True))
    w64 = bool(torch.equal(mine.Zw[:64], ref.Zw[:64]))
    r64 = eq(a["Z"][:64], b["Z"][:64])
    full = dict(Zw=bool(torch.equal(mine.Zw, ref.Zw)), rolled_z1=eq(a["Z"], b["Z"]), y_speed=eq(a["y_speed"], b["y_speed"]), y_yaw=eq(a["y_yaw"], b["y_yaw"]), m_yaw=eq(a["m_yaw"], b["m_yaw"]))
    # red arms
    z = mine.Zw.clone()
    z[0, 0, 0] += 1e-6
    red_perturb = not bool(torch.equal(z[:64], ref.Zw[:64]))
    r_bad = DL.rolled_z1(mine.wm, mine.Zw[:64, 0], mine.A[:64], k + 1).numpy().astype(np.float64)
    red_k = not eq(r_bad, a["Z"][:64])
    fr0 = LR.load_eval_frames(f"{LP.FRAMES}/frames_s0.npz")
    ctx_basis = planted_slow_ctx()
    try:
        LP.clip_z1(ctx_basis.wm, fr0)               # lp_lib's own route on a basis model: must refuse
        red_basis = False
    except ValueError:
        red_basis = True
    ok = w64 and r64 and all(full.values()) and red_perturb and red_k and red_basis
    return dict(pass_=bool(ok), windows=64, window_z1_bit_equal_64=w64, rolled_z1_bit_equal_64=r64, every_window=full, max_abs_diff_Zw=float((mine.Zw - ref.Zw).abs().max()),
                red_arms=dict(perturbed_flagged=red_perturb, step_k_plus_1_differs=red_k, basis_model_via_lp_lib_route_raises=red_basis), wall_s=DL.stamp(t0))


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    DL.wait_ram()
    try:
        mine = S.prepare_arm_read("R35H_s0", R35H_DIR)
        i, ii = b1_i_ii({"R35H_s0": mine, "planted_SLOW_s0": planted_slow_ctx()})
        for key, v in (("b1_i", i), ("b1_ii", ii), ("b1_iii", b1_iii(mine))):
            v["pass"] = v.pop("pass_")
            res = merge(key, v)
    except BaseException:
        traceback.print_exc()
        res = merge("error_b1", {"pass": False, "text": traceback.format_exc()[-1500:]})
    print({k: res[k].get("pass") for k in KEYS if k in res}, "all_pass", res["all_pass"])
    return 0 if all(res.get(k, {}).get("pass") for k in KEYS[:3]) else 1


if __name__ == "__main__":
    sys.exit(main())
