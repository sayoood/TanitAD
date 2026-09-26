#!/usr/bin/env python3
"""refcv7 NEW-1 -- MEASURE the prior against the battery's echo, and answer the
anchor question, on the BANKED refcv6 battery surface. CPU only, no model.

Inputs (all read-only):
  --dump   the battery's banked S2 dump (default: step 30,000, inference seed 0,
           4,754 windows / 139 episodes): per window `ws`, `v0`, `g`, `ha0_ext`,
           `ha`, `os`; per episode `decisions/epNNN.npz` with `ep_poses`,
           `pose_last`, `gt_future_ext`, `gt_future_valid_ext`.
  --cache  the eval139 v2 cache's `_v2manifest.pt` (poses + RECORDED actions
           per clip; metadata only, no frames).
  --repo   the tree under test (its `stack/` and `taniteval/` are imported;
           `tanitad.__file__` is asserted to live under it).

What it measures (every number carries its n and its estimator):
  A  P(mode) vs the BANKED `ha0_ext` at the 2 s slots -- max |diff| in metres,
     and P(ha0_ext) vs the tip's OWN battery functions
     (`refav1_arm.hold_ext_controls` + `refcv3_arm.integrate_select`), called
     per window exactly as `refcv3_arm.py:2125-2128` calls them.
  B  ADE 0-2 s of every prior mode, the banked echo, `ha` and refcv6 `os`
     against `g`, and the paired episode-cluster bootstrap of
     P(mode) - ha0_ext (taniteval.ci, n_boot 2000, seed 0, cluster = clip).
  C  ADE over the 8 model slots (0.5-6 s) on windows whose 60-step future is
     valid, same pairing.
  D  THE ANCHOR QUESTION: oracle-in-vocabulary ADE (min over the 117 anchors)
     of the refcv6 vocabulary rolled (i) ABSOLUTE from v0 -- refcv6's
     `roll_bank` -- and (ii) as RESIDUALS on each prior (`roll_plan`), at 0-2 s
     and 0-6 s, plus the friction-circle census of the composed controls.

Writes raw/measure_prior_vs_echo.json. Prints the headline table.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def _load_by_path(name: str, path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="C:/Users/Admin/ev6_battery/raw/step30000/dump_s0")
    ap.add_argument("--cache", default="D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt")
    ap.add_argument("--anchors", default="D:/refcv6_eval_kit/data/anchors/refc_anchors_6s_v0cond_alat_117.pt")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default=os.path.join(PKG, "raw", "measure_prior_vs_echo.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--battery-fn-episodes", type=int, default=139,
                    help="episodes on which the tip's battery functions are re-called")
    a = ap.parse_args()

    repo = os.path.abspath(a.repo)
    for p in (os.path.join(repo, "stack", "scripts"), os.path.join(repo, "stack"),
              os.path.join(repo, "taniteval")):
        sys.path.insert(0, p)
    import torch
    import tanitad
    if not os.path.abspath(tanitad.__file__).startswith(os.path.join(repo, "stack")):
        raise SystemExit(f"tanitad imported from {tanitad.__file__}, not {repo}")
    from tanitad.models import kinematic_prior as KP
    from taniteval import ci
    import refb_labels
    rc = _load_by_path("refcv3_arm_r7m", os.path.join(repo, "taniteval", "tools", "refcv3_arm.py"))
    ra = rc.ra

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    man = json.load(open(os.path.join(a.dump, "manifest.json"), encoding="utf-8"))
    hz = [int(h) for h in man["model"]["horizons"]]
    grid2 = man["grid"]
    slots2 = list(grid2["slots"])
    W = int(man["grid"]["obs_window"])
    cm = torch.load(a.cache, map_location="cpu", weights_only=False)
    by_clip = {str(c): i for i, c in enumerate(cm["clip_id"])}
    art = torch.load(a.anchors, map_location="cpu", weights_only=False)
    ctrl = art["controls"].float()                          # [117, 2] (a_lon, a_lat)
    meta_units = art.get("control_units") or art.get("meta", {}).get("control_units")
    N = ctrl.shape[0]
    ALAT_FLOOR, KCAP, MU_G = 4.0, 0.12, 0.7 * 9.81
    modes = ["ha0_ext", "ha0_ext_pose", "cv_yawrate"]

    rows = {k: [] for k in ("eid", "g", "echo", "ha", "os", "v0", "g6", "v6", "eid6")}
    P2 = {m: [] for m in modes}
    P8 = {m: [] for m in modes}
    orc = {k: [] for k in ["abs2", "abs6"] + [f"{m}_2" for m in modes] + [f"{m}_6" for m in modes]}
    kamm = {k: [0, 0] for k in ["abs"] + modes}
    bat_maxdiff, bat_n = 0.0, 0
    pose_mismatch = 0
    v0_mismatch_max = 0.0
    t_start = time.time()
    for e in man["episodes"]:
        fi, cid = int(e["file_index"]), str(e["clip_id"])
        z = np.load(os.path.join(a.dump, f"ep{fi:03d}.npz"))
        dz = np.load(os.path.join(a.dump, "decisions", f"ep{fi:03d}.npz"))
        ci_ = by_clip[cid]
        poses = cm["poses"][ci_].float()                    # [T, 4]
        acts = cm["actions"][ci_].float()                   # [T, 2]
        if not torch.equal(poses, torch.as_tensor(dz["ep_poses"]).float()):
            pose_mismatch += 1
        ws = torch.as_tensor(z["ws"]).long()
        n = int(ws.shape[0])
        # the OBSERVED window of every banked window: poses[t0-W+1 .. t0]
        idx = ws[:, None] - (W - 1) + torch.arange(W)[None]  # [n, W]
        if int(idx.min()) < 0:
            raise SystemExit(f"window before clip start on clip {sha12(cid)}")
        ph = poses[idx]                                     # [n, W, 4]
        ah = acts[idx]                                      # [n, W, 2]
        v0 = torch.as_tensor(z["v0"]).float()
        v0_mismatch_max = max(v0_mismatch_max, float((ph[:, -1, 3] - v0).abs().max()))
        for m in modes:
            a0, k0 = KP.prior_controls(m, ph, W, ah if KP.needs_actions(m) else None)
            p = KP.prior_path(a0, k0, v0, hz)               # [n, 8, 2]
            P8[m].append(p)
            P2[m].append(p[:, slots2])
        # the tip's own battery functions, per window, exactly as refcv3_arm calls them
        if fi < a.battery_fn_episodes:
            v_ep, kap_ep = poses[:, 3].float(), acts[:, 0].float()
            n_f = int(grid2["n_frames"])
            for j, t0 in enumerate(ws.tolist()):
                ext = ra.hold_ext_controls(None, v_ep, kap_ep, int(t0), dt=0.1, stride=1)
                hx = rc.integrate_select(ext[None].expand(n_f, 2), float(v0[j]), grid2,
                                         action_units="steer")
                d = float(np.abs(hx[0] - P2["ha0_ext"][-1][j].numpy()).max())
                bat_maxdiff = max(bat_maxdiff, d)
                bat_n += 1
        rows["eid"].append(np.full(n, fi))
        rows["g"].append(z["g"]); rows["echo"].append(z["ha0_ext"])
        rows["ha"].append(z["ha"]); rows["os"].append(z["os"]); rows["v0"].append(z["v0"])
        # 6 s GT on windows with a valid 60-step future (all 8 slots valid)
        fv = torch.as_tensor(dz["gt_future_valid_ext"]) > 0.5
        keep6 = torch.stack([fv[:, h - 1] for h in hz], 1).all(1)
        g8 = refb_labels.waypoint_targets(torch.as_tensor(dz["pose_last"]).float(),
                                          torch.as_tensor(dz["gt_future_ext"]).float(), hz)
        rows["g6"].append(g8[keep6].numpy()); rows["eid6"].append(np.full(int(keep6.sum()), fi))
        rows["v6"].append(keep6.numpy())
        # ---- D: oracle-in-vocabulary, absolute vs residual ---------------------- #
        g2t = torch.as_tensor(z["g"]).float()
        dl = ctrl[None, :, None, :].expand(n, N, len(hz), 2)
        # (i) ABSOLUTE: refcv6's roll_bank semantics (rs.roll_controls on the constant pair)
        from tanitad.refs.refc_sampler import roll_controls
        bank_abs = roll_controls(dl, v0, tuple(hz), control_units="alat",
                                 alat_v_floor=ALAT_FLOOR, kappa_cap=KCAP)   # [n, N, 8, 2]
        ade2 = torch.linalg.norm(bank_abs[:, :, slots2] - g2t[:, None], dim=-1).mean(-1)
        orc["abs2"].append(ade2.min(1).values)
        if keep6.any():
            ade6 = torch.linalg.norm(bank_abs[keep6] - g8[keep6][:, None], dim=-1).mean(-1)
            orc["abs6"].append(ade6.min(1).values)
        # friction census, absolute: |(a_lon, v0^2 kappa)| over mu g
        vv = v0.clamp_min(ALAT_FLOOR) ** 2
        kap_abs = (ctrl[None, :, 1] / vv[:, None]).clamp(-KCAP, KCAP)
        alat_true = kap_abs * v0[:, None] ** 2
        over = torch.sqrt(ctrl[None, :, 0] ** 2 + alat_true ** 2) > MU_G
        kamm["abs"][0] += int(over.sum()); kamm["abs"][1] += int(over.numel())
        for m in modes:
            a0, k0 = KP.prior_controls(m, ph, W, ah if KP.needs_actions(m) else None)
            bank_res = KP.roll_plan(dl, a0, k0, v0, hz, control_units="alat",
                                    alat_v_floor=ALAT_FLOOR, kappa_cap=KCAP)
            ade2r = torch.linalg.norm(bank_res[:, :, slots2] - g2t[:, None], dim=-1).mean(-1)
            orc[f"{m}_2"].append(ade2r.min(1).values)
            if keep6.any():
                ade6r = torch.linalg.norm(bank_res[keep6] - g8[keep6][:, None], dim=-1).mean(-1)
                orc[f"{m}_6"].append(ade6r.min(1).values)
            kr = k0[:, None] + (ctrl[None, :, 1] / vv[:, None]).clamp(-KCAP, KCAP)
            ar = a0[:, None] + ctrl[None, :, 0]
            over_r = torch.sqrt(ar ** 2 + (kr * v0[:, None] ** 2) ** 2) > MU_G
            kamm[m][0] += int(over_r.sum()); kamm[m][1] += int(over_r.numel())
    wall = time.time() - t_start

    cat = {k: np.concatenate(v) for k, v in rows.items()}
    eid, G = cat["eid"], cat["g"].astype(np.float64)

    def ade(P, G_):
        return np.linalg.norm(np.asarray(P, dtype=np.float64) - G_, axis=-1).mean(-1)

    out = {"tool": "measure_prior_vs_echo.py", "repo": repo,
           "tanitad_file": tanitad.__file__, "dump": a.dump,
           "dump_model": {k: man["model"].get(k) for k in ("ckpt", "step")},
           "grid_2s": {k: grid2[k] for k in ("horizons_steps", "slots", "n_windows")},
           "n_windows_2s": int(G.shape[0]), "n_episodes": int(len(np.unique(eid))),
           "cache": a.cache, "anchors": a.anchors, "anchor_units": meta_units,
           "wall_s": round(wall, 1), "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, "
           f"n_boot {a.n_boot}, seed 0, cluster = clip (file_index)"}
    out["alignment"] = {"ep_poses_vs_cache_mismatching_episodes": pose_mismatch,
                        "max_abs_v0_dump_minus_pose_window": v0_mismatch_max}
    # ---- A ---------------------------------------------------------------------
    echo = cat["echo"].astype(np.float64)
    A = {}
    for m in modes:
        P = torch.cat(P2[m]).numpy().astype(np.float64)
        dif = np.abs(P - echo)
        A[m] = {"max_abs_m": float(dif.max()), "mean_abs_m": float(dif.mean()),
                "frac_windows_bit_equal": float((dif.reshape(dif.shape[0], -1).max(1) == 0).mean())}
    A["ha0_ext_vs_tip_battery_functions"] = {"max_abs_m": bat_maxdiff, "n_windows": bat_n}
    out["A_prior_vs_banked_echo_2s"] = A
    # ---- B ---------------------------------------------------------------------
    arms = {"echo_banked": ade(echo, G), "ha_banked": ade(cat["ha"], G), "os_refcv6_30k": ade(cat["os"], G)}
    for m in modes:
        arms[f"P_{m}"] = ade(torch.cat(P2[m]).numpy(), G)
    B = {"ade_mean_m": {k: float(v.mean()) for k, v in arms.items()}}
    for m in modes:
        B[f"P_{m}_minus_echo"] = ci.paired_episode_cluster_bootstrap(
            arms[f"P_{m}"], arms["echo_banked"], list(eid), n_boot=a.n_boot, seed=0)
    B["os_minus_echo"] = ci.paired_episode_cluster_bootstrap(
        arms["os_refcv6_30k"], arms["echo_banked"], list(eid), n_boot=a.n_boot, seed=0)
    out["B_ade_0_2s"] = B
    # ---- C ---------------------------------------------------------------------
    keep6 = cat["v6"].astype(bool)
    G6 = cat["g6"].astype(np.float64)
    eid6 = cat["eid6"]
    C = {"n_windows": int(G6.shape[0]), "n_episodes": int(len(np.unique(eid6)))}
    a6 = {m: ade(torch.cat(P8[m]).numpy()[keep6], G6) for m in modes}
    C["ade_mean_m"] = {f"P_{m}": float(v.mean()) for m, v in a6.items()}
    for m in ("ha0_ext_pose", "cv_yawrate"):
        C[f"P_{m}_minus_P_ha0_ext"] = ci.paired_episode_cluster_bootstrap(
            a6[m], a6["ha0_ext"], list(eid6), n_boot=a.n_boot, seed=0)
    out["C_ade_0_6s_8slots"] = C
    # ---- D ---------------------------------------------------------------------
    D = {"n_windows_2s": int(G.shape[0]), "n_windows_6s": int(G6.shape[0]),
         "definition": "oracle-in-vocabulary = min over the 117 anchors of the window's ADE",
         "vocabulary": "refc_anchors_6s_v0cond_alat_117.pt controls (13 a_lon x 9 a_lat)",
         "absolute": "refcv6 roll_bank: rs.roll_controls(constant pair, v0), kappa = clamp(a_lat/max(v0,4)^2, +-0.12)",
         "residual": "kinematic_prior.roll_plan(pair, prior): a = a0 + a_i, kappa = kappa0 + clamp(a_lat_i/max(v0,4)^2, +-0.12)"}
    o2 = {k: torch.cat(v).numpy().astype(np.float64) for k, v in orc.items() if k.endswith("2")}
    o6 = {k: torch.cat(v).numpy().astype(np.float64) for k, v in orc.items() if k.endswith("6")}
    D["oracle_ade_0_2s_mean_m"] = {k: float(v.mean()) for k, v in o2.items()}
    D["oracle_ade_0_6s_mean_m"] = {k: float(v.mean()) for k, v in o6.items()}
    for m in modes:
        D[f"residual_{m}_minus_absolute_0_2s"] = ci.paired_episode_cluster_bootstrap(
            o2[f"{m}_2"], o2["abs2"], list(eid), n_boot=a.n_boot, seed=0)
        D[f"residual_{m}_minus_absolute_0_6s"] = ci.paired_episode_cluster_bootstrap(
            o6[f"{m}_6"], o6["abs6"], list(eid6), n_boot=a.n_boot, seed=0)
    D["friction_census_over_mu0.7"] = {k: {"n_over": v[0], "n": v[1], "frac": v[0] / max(v[1], 1)}
                                       for k, v in kamm.items()}
    out["D_anchor_question"] = D
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"A": A, "B_ade": B["ade_mean_m"],
                      "C_ade": C["ade_mean_m"], "D2": D["oracle_ade_0_2s_mean_m"],
                      "D6": D["oracle_ade_0_6s_mean_m"], "align": out["alignment"],
                      "wall_s": out["wall_s"]}, indent=1))
    print("WROTE", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
