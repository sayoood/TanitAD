"""cf_run.py -- the nav counterfactual (SELECTION_V2.md s2), held-vs-fresh S, and the controls, on the 4 pre-declared windows, both seeds.
Writes OUT/data/cf_windows.npz (+ cf_windows.json: scalars / strings) and OUT/CONTROLS_V2_part1.json.   CPU only."""
from __future__ import annotations

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_lib import *  # noqa: E402,F401,F403

os.makedirs(f"{OUT}/data", exist_ok=True)


def d6_of(plans: dict) -> dict:
    """plans[cmd] -> [30, 2]; the pre-declared headline D6 and the companions (SELECTION_V2 s2)."""
    f, l, r = (np.asarray(plans[c])[29] for c in ("FOLLOW", "TURN_L", "TURN_R"))
    dl, dr = float(np.linalg.norm(l - f)), float(np.linalg.norm(r - f))
    return {"D6": 0.5 * (dl + dr), "D6_TURN_L": dl, "D6_TURN_R": dr, "lat_sep_LR": float(abs(l[1] - r[1])), "LR_dist": float(np.linalg.norm(l - r))}


def main():
    t0 = time.time()
    res, ctl = {}, {"per_window": {}}
    banked = {s: np.load(f"{WORK}/banked/eval_{s}-kind-own.npz") for s in SEEDS}
    fresh_b = {s: np.load(f"{SC}/sfresh/eval/eval_{s}-sfresh-fresh.npz") for s in SEEDS}
    cyc_b = {s: np.load(f"{SC}/sfresh/eval/eval_{s}-sfresh-cycle.npz") for s in SEEDS}
    for seed in SEEDS:
        model, cfg = load_model(seed)
        ew = EvalWorld(cfg, lru=4)
        zero = copy.deepcopy(model)
        with torch.no_grad():
            zero.e_nav.weight.zero_()                                   # structural zero: the nav embedding becomes a constant (bias only)
        for name in KEYS:
            w = WINDOWS[name]["window"]
            assert ew.clips[ew.wins[w][0]] == WINDOWS[name]["clip"] and ew.wins[w][1] == WINDOWS[name]["t"], (name, ew.wins[w])
            b = ew.batch([w])
            a_t, a_s = int(b["a_t"][0]), int(b["a_s"][0])
            P = f"{seed}/{name}"
            R = {"a_t": a_t, "a_s": a_s, "gt": ew.X["gt"][w], "rc": ew.X["rc"][w], "rc_ok": ew.X["rc_ok"][w], "nav_own_now": b["nav"][0, -1].numpy(),
                 "nav_own_hist": b["nav"][0].numpy(), "nav_d": float(ew.X["nav_d"][w]), "pose_s": b["pose_s"][0].numpy(), "pose_now": b["pose_now"][0].numpy(),
                 "kin_v": float(b["kin3"][0, 0]) if b["kin3"].ndim > 1 else float(b["kin3"][0])}
            # ---- own forward, the chain with no override, and the chain with the window's OWN nav tensor injected at all layers
            o = own_forward(model, b)
            c0 = chain(model, b)
            c_own = chain(model, b, nav=b["nav"].clone(), where=("S", "T", "O"))
            R["own_traj30"] = o["traj30"][0].numpy()
            R["own_ckpt_s"] = o["ckpt"][0].numpy()
            R["own_ckpt_now"] = M.transform_ckpt(o["ckpt"], b["pose_s"], b["pose_now"])[0].numpy()
            R["own_lat_p"] = torch.softmax(o["lat_logits"], -1)[0].numpy()
            R["own_lon_p"] = torch.softmax(o["lon_logits"], -1)[0].numpy()
            diffs = {"chain_vs_forward_traj30": float((c0["traj30"] - o["traj30"]).abs().max()),
                     "chain_vs_forward_ckpt": float((c0["ckpt_s"] - o["ckpt"]).abs().max()),
                     "chain_vs_forward_lat": float((c0["lat_logits"] - o["lat_logits"]).abs().max()),
                     "own_nav_all_layers_vs_forward_traj30": float((c_own["traj30"] - o["traj30"]).abs().max()),
                     "own_nav_all_layers_vs_forward_ckpt": float((c_own["ckpt_s"] - o["ckpt"]).abs().max()),
                     "own_nav_all_layers_vs_forward_lat": float((c_own["lat_logits"] - o["lat_logits"]).abs().max()),
                     "own_nav_all_layers_vs_forward_lon": float((c_own["lon_logits"] - o["lon_logits"]).abs().max())}
            # banked (GPU box-2) comparisons: reported with their actual maxima
            diffs["own_traj30_vs_banked_pred"] = float(np.abs(R["own_traj30"] - banked[seed]["pred"][w]).max())
            # the banked npz stores S's checkpoints in S's TICK frame (the evaluator re-expresses them in NOW for scoring), so compare in that frame, and re-express the banked ones with the model's own transform
            diffs["own_ckpt_Sframe_vs_banked_ckpt"] = float(np.abs(R["own_ckpt_s"] - banked[seed]["ckpt"][w]).max())
            bk_now = M.transform_ckpt(torch.as_tensor(banked[seed]["ckpt"][w:w + 1]), b["pose_s"], b["pose_now"])[0].numpy()
            diffs["own_ckpt_now_vs_banked_ckpt_reexpressed"] = float(np.abs(R["own_ckpt_now"] - bk_now).max())
            diffs["own_ade6_vs_banked_ade6"] = float(abs(np.linalg.norm(R["own_traj30"].astype(np.float64) - R["gt"], axis=-1).mean() - banked[seed]["ade6"][w]))
            # ---- fresh S (a_T = a_S = 0, `--ages fresh`)
            bf = ew.batch([w], fresh=True)
            of = own_forward(model, bf)
            R["fresh_ckpt_now"] = of["ckpt"][0].numpy()                    # fresh: S's tick frame IS NOW
            R["fresh_traj30"] = of["traj30"][0].numpy()
            R["fresh_a"] = np.array([int(bf["a_t"][0]), int(bf["a_s"][0])])
            diffs["fresh_ckpt_vs_sfresh_banked"] = float(np.abs(R["fresh_ckpt_now"] - fresh_b[seed]["ckpt"][w]).max())
            diffs["held_ckpt_Sframe_vs_sfresh_cycle_banked"] = float(np.abs(R["own_ckpt_s"] - cyc_b[seed]["ckpt"][w]).max())
            diffs["fresh_traj30_vs_sfresh_banked_pred"] = float(np.abs(R["fresh_traj30"] - fresh_b[seed]["pred"][w]).max())
            # ---- the counterfactual: 4 injection points x 3 commands
            for cfgn, where in CONFIGS.items():
                plans, ckps, lats, lons = {}, {}, {}, {}
                zplans = {}
                for cmd in CMDS:
                    r = chain(model, b, nav=cmd_tensor(b, cmd), where=where)
                    plans[cmd], ckps[cmd] = r["traj30"][0].numpy(), r["ckpt_now"][0].numpy()
                    lats[cmd], lons[cmd] = torch.softmax(r["lat_logits"], -1)[0].numpy(), torch.softmax(r["lon_logits"], -1)[0].numpy()
                    rz = chain(zero, b, nav=cmd_tensor(b, cmd), where=where)
                    zplans[cmd] = rz["traj30"][0].numpy()
                    R[f"cf/{cfgn}/{cmd}/traj30"], R[f"cf/{cfgn}/{cmd}/ckpt_now"] = plans[cmd], ckps[cmd]
                    R[f"cf/{cfgn}/{cmd}/lat_p"], R[f"cf/{cfgn}/{cmd}/lon_p"] = lats[cmd], lons[cmd]
                dd = d6_of(plans)
                dz = d6_of(zplans)
                for k, v in dd.items():
                    R[f"cf/{cfgn}/{k}"] = v
                for k, v in dz.items():
                    R[f"cfzero/{cfgn}/{k}"] = v
            for k, v in R.items():
                res[f"{P}/{k}"] = np.asarray(v)
            ctl["per_window"][P] = diffs
            print(P, "a_t/a_s", a_t, a_s, {k: f"{v:.2e}" for k, v in diffs.items()}, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(f"{OUT}/data/cf_windows.npz", **res)
    json.dump(ctl, open(f"{OUT}/CONTROLS_V2_part1.json", "w"), indent=1)
    print("wrote", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
