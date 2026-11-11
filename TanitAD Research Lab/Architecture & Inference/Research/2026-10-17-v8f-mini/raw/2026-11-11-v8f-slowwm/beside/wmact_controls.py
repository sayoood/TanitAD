"""SLOWWM beside B2 control (BESIDE_RULES_SLOWWM.md, B2) and the checks that the port of ``run_wmact.py`` and the basis collection are what they say. Merges key ``b2`` into ``$BESIDE_OUT/CONTROLS_B1B2.json``.
  a  M2b_s0 through the NEW collection (``wmact_arm.collect_arm_z1``, adapter route) reproduces the banked ``wmact/latents.pt`` ``Z1["M2b_s0"]`` bit for bit on 64 windows. Red arms: M2b_s1's model through the same
     collection differs from M2b_s0's banked latents; a 1e-7 perturbation is flagged.
  b  ``wmact_arm.stats_for`` (the per-model body of run_wmact.py) on the banked latents reproduces every row (point, lo, hi) and every sens value of the banked ``RESULT_WMACT.json`` for M2b_s0 within 1e-6 (float32 latents; MEASURED first read 8.1e-8 rows, 3.0e-8 sens), and
     controls 1 and 2 read their known values (identity ~ 0; zeroed embedding: level-1 sens exactly 0). Red arm: M2b_s0's model on M2b_s1's latents differs from the banked M2b_s0 rows.
  c  Basis collection on a planted basis arm: z1 of the first 64 windows (``collect_basis_z1``, live u) equals an independent float64 (nm - mean) @ proj through the model's own ``_z1_post`` within 1e-4; control 4
     (``forward_basis`` against ``stack.loss`` on z1_from_basis) reads 0. Red arm: the OTHER planted basis' expected z1 fails the same equality.
Usage: python wmact_controls.py [a b c]"""
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SLOWWM_READ_WORK", "F:/TanitAD-scratch/slowwm_beside_build/b1b2/planted")
import wmact_arm as A  # noqa: E402
import decod_controls as DC  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DL, W, S, LR, LP, SH, CR = A.DL, A.W, A.S, A.LR, A.LP, A.SH, DC.CR
SB = SH.SB


def control_a(ds, banked):
    t0 = time.time()
    m0, _, _ = W.load_model(W.CKPT["M2b_s0"])
    m1, _, _ = W.load_model(W.CKPT["M2b_s1"])
    Z0, *_ = A.collect_arm_z1(m0, "adapter", None, ds, 64)
    Z1s, *_ = A.collect_arm_z1(m1, "adapter", None, ds, 64)
    ref = banked["Z1"]["M2b_s0"][:64]
    eq = bool(torch.equal(Z0, ref))
    pert = Z0.clone()
    pert[0, 0, 0] += 1e-7
    red = dict(other_model_differs=not bool(torch.equal(Z1s, ref)), perturbation_flagged=not bool(torch.equal(pert, ref)))
    return dict(pass_=bool(eq and all(red.values())), windows=64, bit_equal=eq, max_abs_diff=float((Z0 - ref).abs().max()), red_arms=red, wall_s=DL.stamp(t0))


def _flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict) and "point" in v:
            for s in ("point", "lo", "hi"):
                yield f"{p}{k}/{s}", v[s]
        elif isinstance(v, dict):
            yield from _flat(v, f"{p}{k}/")
        else:
            yield f"{p}{k}", v


def control_b(banked):
    t0 = time.time()
    res = json.load(open(A.BANKED_RESULT, encoding="utf-8"))
    ok_all = banked["ok"].all(1).numpy()
    sel = np.flatnonzero(ok_all)
    eid = np.array(banked["eid"])[sel]
    m0, _, _ = W.load_model(W.CKPT["M2b_s0"])
    mean_sc, _ = W.mean_action_scaled(m0.wm, banked["A_raw"], banked["ok"])
    mine = A.stats_for(m0, banked["Z1"]["M2b_s0"][sel], banked["A_raw"][sel], eid, mean_sc)
    want = dict(_flat(res["rows"]["M2b_s0"]))
    got = dict(_flat(mine["rows"]))
    d_rows = max(abs(got[k] - want[k]) for k in want) if set(got) == set(want) else float("inf")
    ws, gs = res["sens"]["M2b_s0"], mine["sens"]
    d_sens = max(abs(gs[k] - ws[k]) for k in ws) if set(gs) == set(ws) else float("inf")
    c = mine["controls"]
    known = {"identity_sens_max_abs": c["1_identity_sens_max_abs"], "identity_actionuse_max_abs": c["1_identity_actionuse_max_abs"], "zeroed_embed_L1_sens_max_abs": c["2_zeroed_act_embed"]["sens_L1_max_abs"]}
    known_ok = known["identity_sens_max_abs"] < 1e-6 and known["identity_actionuse_max_abs"] < 1e-6 and known["zeroed_embed_L1_sens_max_abs"] < 1e-6
    red = A.stats_for(m0, banked["Z1"]["M2b_s1"][sel], banked["A_raw"][sel], eid, mean_sc)
    d_red = max(abs(dict(_flat(red["rows"]))[k] - want[k]) for k in want)
    ok = d_rows <= 1e-6 and d_sens <= 1e-6 and known_ok and d_red > 1e-6
    return dict(pass_=bool(ok), rows_compared=len(want), max_abs_diff_rows=d_rows, max_abs_diff_sens=d_sens, tolerance=1e-6, controls_known_values=known, controls_known_ok=bool(known_ok),
                red_arm_other_latents_max_abs_diff_rows=d_red, wall_s=DL.stamp(t0))


def control_c(ds):
    t0 = time.time()
    fr = LR.load_eval_frames(f"{LP.FRAMES}/frames_s0.npz")
    B1, B2 = CR.planted_basis("slow", 1), CR.planted_basis("slow", 2)
    nms = CR.planted_nm(fr)
    ub1 = CR.planted_ubasis_file("1", fr, nms, B1)
    run = CR.planted_run("ok", "slow", "slow", "slow", B1["sha"])
    L = S.load_checked("SLOW_s0", run, B1["path"], ub1)
    model, B = L["model"], L["B"]
    Z, *_ = A.collect_arm_z1(model, "slow", B, ds, 64)
    items = [ds.item(ds.windows[i][0], ds.windows[i][1]) for i in range(64)]

    def expect(Bx):
        out = []
        for it in items:
            nm = SB.nm_of(it["feats"])
            u = ((nm.astype(np.float64) - Bx["mean"].astype(np.float64)) @ Bx["proj"].astype(np.float64)).astype(np.float32)
            with torch.no_grad():
                out.append(model.wm._z1_post(torch.as_tensor(u)[None])[0].float())
        return torch.stack(out)
    d_ok, d_swap = float((Z - expect(B1)).abs().max()), float((Z - expect(B2)).abs().max())
    c4 = A.control4(model, "slow", B, ds, 64)
    ok = d_ok <= 1e-4 and d_swap > 1e-3 and c4["max_abs_diff"] == 0.0
    return dict(pass_=bool(ok), z1_vs_independent_float64_max_abs=d_ok, tolerance=1e-4, red_arm_other_basis_max_abs=d_swap, control4_basis_max_abs_diff=c4["max_abs_diff"], wall_s=DL.stamp(t0))


def main(argv):
    want = [a for a in argv if a in ("a", "b", "c")] or ["a", "b", "c"]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    DL.wait_ram()
    parts = {}
    try:
        banked = torch.load(A.BANKED_EVAL, weights_only=False)
        ds, _, _ = W.open_windows(16)
        if "a" in want:
            parts["a_new_collection_vs_banked_latents"] = control_a(ds, banked)
        if "b" in want:
            parts["b_stats_port_vs_banked_result"] = control_b(banked)
        if "c" in want:
            parts["c_basis_collection_planted_arm"] = control_c(ds)
    except BaseException:
        traceback.print_exc()
        parts["error"] = {"pass_": False, "text": traceback.format_exc()[-1500:]}
    for v in parts.values():
        v["pass"] = v.pop("pass_")
    prev = json.load(open(DC.OUT, encoding="utf-8")).get("b2", {}) if os.path.exists(DC.OUT) else {}
    prev.update(parts)
    need = ("a_new_collection_vs_banked_latents", "b_stats_port_vs_banked_result", "c_basis_collection_planted_arm")
    prev["pass"] = all(prev.get(k, {}).get("pass") is True for k in need) and "error" not in prev
    res = DC.merge("b2", prev)
    print({k: v.get("pass") for k, v in prev.items() if isinstance(v, dict)}, "b2", prev["pass"], "all_pass", res["all_pass"])
    return 0 if prev["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
