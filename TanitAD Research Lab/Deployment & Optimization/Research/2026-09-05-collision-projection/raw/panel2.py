#!/usr/bin/env python3
"""H-PROJ-CONTACT-1, part 2 -- the parts the first pass showed were needed.

(1) THE MASS, not only the argmax. `sel_contact` reads 0.0000 in the base, so a
    projection changes nothing about TODAY's emitted plan. The question that
    decides whether the generator defect matters is therefore not "does the argmax
    collide" but "how much probability mass does the model put on the colliding
    set, and is any of it inside the selector's top-k" -- RETRACTION #31 in its
    quantitative form. `rank` and `conf` are banked, so this costs nothing.
(2) THE COMPOSED DECODE. Friction projection THEN contact retraction, with BOTH
    flags re-derived on the same output tensor: the claim under test is that an
    over-friction OR colliding trajectory is SIMULTANEOUSLY unrepresentable.
(3) THE FAR END OF THE MARGIN FRONTIER, where the projection must eventually fail.
    A frontier reported only where it succeeds is not a frontier.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = r"C:\Users\Admin\collproj"
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
sys.path.insert(0, os.path.join(REPO, "taniteval", "tools"))

from tanitad.refs import contact_projection as CP           # noqa: E402
from tanitad.refs import feasible_decode as FD              # noqa: E402
from tanitad.rl.rewards import _collision                    # noqa: E402
import taniteval.ci as CI                                    # noqa: E402
import importlib.util                                        # noqa: E402
_s = importlib.util.spec_from_file_location(
    "fan_safety", os.path.join(REPO, "taniteval", "tools", "fan_safety.py"))
FS = importlib.util.module_from_spec(_s)
sys.modules["fan_safety"] = FS
_s.loader.exec_module(FS)

BANK = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Lab"
        r"\Deployment & Optimization\Research\2026-09-05-veto-only-fan-safety"
        r"\raw\fan_bank_base_240w.npz")
GT = r"C:\Users\Admin\collproj\out\gt_240w.npz"
OUT = r"C:\Users\Admin\collproj\out\contact_projection_panel2.json"
R = CP.CONTACT_R_M
WIDE_MARGINS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0]


def contact_flag(p, l, r=R):
    return (_collision(p, {"lead_path": l.expand_as(p), "ego_radius_m": r / 2.0,
                           "obs_radius_m": r / 2.0}) < 0)


def main():
    z = np.load(BANK, allow_pickle=True)
    gz = np.load(GT, allow_pickle=True)
    fan = torch.tensor(np.asarray(z["fan2"]), dtype=torch.float64)
    lead = torch.tensor(np.asarray(z["lead5"]), dtype=torch.float64)[:, None]
    has = torch.tensor(np.asarray(z["has_lead"]).astype(bool))
    v0 = torch.tensor(np.asarray(z["v0"], dtype=np.float64))
    sel = torch.tensor(np.asarray(z["sel_idx"]).astype(np.int64))
    rank = torch.tensor(np.asarray(z["rank"]), dtype=torch.float32)
    conf = torch.tensor(np.asarray(z["conf"]), dtype=torch.float32)
    gt = torch.tensor(np.asarray(gz["gt"]), dtype=torch.float64)
    B, N, S, _ = fan.shape
    eid = np.asarray(z["eid"]).astype(int).ravel()
    rec = {"_what": "H-PROJ-CONTACT-1 part 2: mass, the composed decode, and the far "
                    "end of the margin frontier",
           "_evidence_class": "MEASURED (ours; rule-based, 0 GPU)", "bank": BANK}

    hit_in = contact_flag(fan, lead) & has[:, None]
    p_rank = torch.softmax(rank.float(), dim=1)
    p_conf = torch.softmax(conf.float(), dim=1)
    order = rank.argsort(dim=1, descending=True)
    x = hit_in.to(torch.float32)

    # ------------------------------------------------------------------ #
    # (1) THE MASS -- what the selector puts on the colliding set          #
    # ------------------------------------------------------------------ #
    mass = {}
    for name, p in (("rank", p_rank), ("conf", p_conf)):
        mm = (p * x).sum(dim=1)                       # [B]
        mass[f"mass_{name}_contact_mean_all"] = float(mm.mean())
        mass[f"mass_{name}_contact_mean_lead"] = float(mm[has].mean())
        mass[f"mass_{name}_contact_max"] = float(mm.max())
        mass[f"mass_{name}_contact_windows_gt_1pct"] = int((mm > 0.01).sum())
        mass[f"mass_{name}_contact_windows_gt_10pct"] = int((mm > 0.10).sum())
    for k in (1, 2, 4, 8, 16, 32, 64):
        kk = min(k, N)
        v = x.gather(1, order[:, :kk]).mean(dim=1)
        mass[f"top{kk}_contact_mean_all"] = float(v.mean())
        mass[f"top{kk}_contact_mean_lead"] = float(v[has].mean())
        mass[f"top{kk}_contact_windows_any"] = int((v > 0).sum())
    mass["sel_contact"] = float(x.gather(1, sel[:, None])[:, 0].mean())
    mass["_note"] = ("`sel_contact` = 0.0000 says the ARGMAX is clear on these 240 "
                     "windows for THIS checkpoint. It is not a property of the "
                     "generator, and every number above it says so.")
    rec["mass_on_the_colliding_set"] = mass
    print("[p2] mass:", {k: v for k, v in mass.items() if "mean_lead" in k or k == "sel_contact"},
          flush=True)

    # ------------------------------------------------------------------ #
    # (2) THE COMPOSED DECODE                                             #
    # ------------------------------------------------------------------ #
    def score(p):
        sc = FS.score_paths(p.float(), v0.float(), lead.float())
        h = contact_flag(p, lead) & has[:, None]
        a = (p[..., 1:, :] - gt[:, None, :, :]).norm(dim=-1).mean(dim=-1)
        return {"fan_contact": float(h.double().mean()),
                "fan_contact_count": int(h.sum()),
                "sel_contact": float(h.gather(1, sel[:, None])[:, 0].double().mean()),
                "fan_envelope": float(sc["envelope"].double().mean()),
                "fan_kamm_over": float(sc["kamm_over"].double().mean()),
                "fan_off_reach": float(sc["off_reach"].double().mean()),
                "fan_peak_g_mean": float(sc["peak_g"].mean()),
                "fan_peak_g_max": float(sc["peak_g"].max()),
                "oracle_in_fan_ade_m": float(a.min(dim=1).values.mean()),
                "sel_ade_m": float(a.gather(1, sel[:, None])[:, 0].mean()),
                "_oracle_per_window": a.min(dim=1).values,
                "_sel_per_window": a.gather(1, sel[:, None])[:, 0]}

    base = score(fan)
    fric = FD.project_feasible(fan, mu=FD.MU_KAMM)
    s_fric = score(fric)
    comp, info_c = CP.project_contact_free(fric, lead, has, margin_m=0.0,
                                           clamp_friction=True)
    s_comp = score(comp)
    cont_only, _ = CP.project_contact_free(fan, lead, has, margin_m=0.0)
    s_cont = score(cont_only)

    def pb(a, b):
        r = CI.paired_episode_cluster_bootstrap(
            np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
            np.asarray(eid), n_boot=4000, seed=11)
        return {k: r[k] for k in ("delta", "lo", "hi", "separated", "n_windows",
                                  "n_episodes", "estimator")}

    arms = {}
    for name, s in (("base", base), ("friction_only", s_fric),
                    ("contact_only", s_cont), ("composed_friction_then_contact", s_comp)):
        d = {k: v for k, v in s.items() if not k.startswith("_")}
        d["paired_oracle_ade_vs_base"] = pb(s["_oracle_per_window"].numpy(),
                                            base["_oracle_per_window"].numpy())
        d["paired_sel_ade_vs_base"] = pb(s["_sel_per_window"].numpy(),
                                         base["_sel_per_window"].numpy())
        arms[name] = d
        print(f"[p2] {name:32s} contact {d['fan_contact']:.6f} envelope "
              f"{d['fan_envelope']:.6f} kamm {d['fan_kamm_over']:.6f} off_reach "
              f"{d['fan_off_reach']:.6f} peak_g {d['fan_peak_g_mean']:.4f} "
              f"orcADE {d['oracle_in_fan_ade_m']:.4f} selADE {d['sel_ade_m']:.4f}",
              flush=True)
    rec["composed_decode"] = arms
    rec["composed_decode"]["_simultaneity_claim"] = {
        "known_value": "envelope == 0 AND kamm_over == 0 AND contact == 0 on the SAME tensor",
        "envelope": arms["composed_friction_then_contact"]["fan_envelope"],
        "kamm_over": arms["composed_friction_then_contact"]["fan_kamm_over"],
        "contact": arms["composed_friction_then_contact"]["fan_contact"],
        "PASS": bool(arms["composed_friction_then_contact"]["fan_envelope"] == 0.0
                     and arms["composed_friction_then_contact"]["fan_kamm_over"] == 0.0
                     and arms["composed_friction_then_contact"]["fan_contact"] == 0.0),
        "_object": "the tensor returned by project_contact_free(project_feasible(fan2))"}

    # ------------------------------------------------------------------ #
    # (3) THE FAR END OF THE FRONTIER -- where it must eventually fail     #
    # ------------------------------------------------------------------ #
    a_in = (fan[..., 1:, :] - gt[:, None, :, :]).norm(dim=-1).mean(dim=-1)
    rows = []
    for m in WIDE_MARGINS:
        out, info = CP.project_contact_free(fan, lead, has, margin_m=m)
        h = contact_flag(out, lead) & has[:, None]
        d_out = CP.min_rel_distance(out, lead.expand_as(out))
        sc = FS.score_paths(out.float(), v0.float(), lead.float())
        a_out = (out[..., 1:, :] - gt[:, None, :, :]).norm(dim=-1).mean(dim=-1)
        moved = ((out - fan).norm(dim=-1).amax(dim=-1) > 0)
        tax = {CP.REASON_NAMES[k]: int((info["reason"] == k).sum())
               for k in CP.REASON_NAMES}
        row = {"margin_m": m, "r_need_m": float(info["r_need_m"]),
               "fan_contact": float(h.double().mean()),
               "fan_contact_count": int(h.sum()),
               "sel_contact": float(h.gather(1, sel[:, None])[:, 0].double().mean()),
               "min_clearance_m": float(d_out[has].min()),
               "taxonomy": tax,
               "n_moved": int(moved.sum()), "n_moved_windows": int(moved.any(1).sum()),
               "sigma_mean_moved": float(info["sigma"][moved].mean()) if bool(moved.any()) else 1.0,
               "n_lateral": tax["lateral"], "n_unavoidable": tax["unavoidable"],
               "margin_rungs_m": [float(x) for x in info["margin_rungs_m"].tolist()],
               "frac_colliders_at_full_margin": (
                   float((info["margin_achieved_m"][moved] >= m - 1e-9).double().mean())
                   if bool(moved.any()) else 1.0),
               "margin_achieved_mean_on_moved": (
                   float(info["margin_achieved_m"][moved].mean())
                   if bool(moved.any()) else float(m)),
               "margin_achieved_min_on_moved": (
                   float(info["margin_achieved_m"][moved].min())
                   if bool(moved.any()) else float(m)),
               "oracle_in_fan_ade_m": float(a_out.min(dim=1).values.mean()),
               "d_oracle_ade_m": float((a_out.min(dim=1).values - a_in.min(dim=1).values).mean()),
               "sel_ade_m": float(a_out.gather(1, sel[:, None])[:, 0].mean()),
               "d_sel_ade_m": float((a_out - a_in).gather(1, sel[:, None])[:, 0].mean()),
               "fan_off_reach": float(sc["off_reach"].double().mean()),
               "fan_peak_g_mean": float(sc["peak_g"].mean()),
               "fan_ttc_below": float(sc["ttc_below"][has].double().mean()),
               "mean_disp_moved_m": float((out - fan).norm(dim=-1).mean(dim=-1)[moved].mean())
               if bool(moved.any()) else 0.0,
               "moved_ade_base_m": float(a_in[moved].mean()) if bool(moved.any()) else 0.0,
               "moved_ade_proj_m": float(a_out[moved].mean()) if bool(moved.any()) else 0.0,
               }
        row["paired_oracle_ade"] = pb(a_out.min(dim=1).values.numpy(),
                                      a_in.min(dim=1).values.numpy())
        rows.append(row)
        print(f"[p2] m={m:5.1f} contact {row['fan_contact']:.6f} ({row['fan_contact_count']:4d}) "
              f"lat {row['n_lateral']:4d} unav {row['n_unavoidable']:4d} moved {row['n_moved']:5d} "
              f"sig {row['sigma_mean_moved']:.3f} fullM {row['frac_colliders_at_full_margin']:.3f} "
              f"dOrcADE {row['d_oracle_ade_m']:+.4f} "
              f"dSelADE {row['d_sel_ade_m']:+.4f} off_reach {row['fan_off_reach']:.4f}",
              flush=True)
    rec["wide_frontier"] = rows

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(f"[p2] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
