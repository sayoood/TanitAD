#!/usr/bin/env python3
"""Diagnostics for the panel's three surprises, each measured rather than reasoned:
(a) the robustness column that did not move with the margin, (b) the 4 non-collider
candidates that were retracted, (c) the EXACTLY-zero oracle/selected ADE delta."""
import os
import sys

import numpy as np
import torch

REPO = r"C:\Users\Admin\collproj"
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from tanitad.refs import contact_projection as CP           # noqa: E402
from tanitad.rl.rewards import _collision                    # noqa: E402

BANK = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Lab"
        r"\Deployment & Optimization\Research\2026-09-05-veto-only-fan-safety"
        r"\raw\fan_bank_base_240w.npz")
GT = r"C:\Users\Admin\collproj\out\gt_240w.npz"
R = CP.CONTACT_R_M

z = np.load(BANK, allow_pickle=True)
gz = np.load(GT, allow_pickle=True)
fan = torch.tensor(np.asarray(z["fan2"]), dtype=torch.float64)
lead = torch.tensor(np.asarray(z["lead5"]), dtype=torch.float64)[:, None]
has = torch.tensor(np.asarray(z["has_lead"]).astype(bool))
sel = torch.tensor(np.asarray(z["sel_idx"]).astype(np.int64))
gt = torch.tensor(np.asarray(gz["gt"]), dtype=torch.float64)
B, N, S, _ = fan.shape


def contact_flag(p, l, r=R):
    return (_collision(p, {"lead_path": l.expand_as(p), "ego_radius_m": r / 2.0,
                           "obs_radius_m": r / 2.0}) < 0)


def ade(p):
    return (p[..., 1:, :] - gt[:, None, :, :]).norm(dim=-1).mean(dim=-1)


hit_in = contact_flag(fan, lead) & has[:, None]
d_in = CP.min_rel_distance(fan, lead.expand_as(fan))
print("=== (a) the robustness column ===")
for m in (0.0, 0.5, 2.0):
    out, info = CP.project_contact_free(fan, lead, has, margin_m=m)
    d_out = CP.min_rel_distance(out, lead.expand_as(out))
    print(f" m={m}: min clearance over LEAD windows {float(d_out[has].min()):.6f}")
    for delta in (0.0, 0.5, 1.0, 2.0):
        h = (d_out < (R + delta)) & has[:, None]
        print(f"    d_out < {R + delta:.2f}: n={int(h.sum())} "
              f"rate_all={float(h.double().mean()):.6f} "
              f"rate_leadpop={float(h[has].double().mean()):.6f}")
    # the honest robustness: RE-SCORE with the scorer against an INFLATED radius
    for delta in (0.5, 2.0):
        hs = contact_flag(out, lead, r=R + delta) & has[:, None]
        print(f"    _collision at r={R + delta:.2f}: n={int(hs.sum())}")

print()
print("=== (b) the 4 retracted non-colliders ===")
out0, info0 = CP.project_contact_free(fan, lead, has, margin_m=0.0)
moved = ((out0 - fan).norm(dim=-1).amax(dim=-1) > 0)
extra = torch.nonzero(moved & ~hit_in)
print(" n extra:", extra.shape[0], " r_need =", float(info0["r_need_m"]))
for b, n in extra.tolist():
    print(f"   row {b} cand {n}: input clearance {float(d_in[b, n]):.6f} "
          f"(scorer flags < {R}); r_need {float(info0['r_need_m']):.4f}; "
          f"in [r, r_need) = {R <= float(d_in[b, n]) < float(info0['r_need_m'])}; "
          f"sigma {float(info0['sigma'][b, n]):.6f}")
print(" ALL 4 lie in [r, r_need) :",
      bool(all(R <= float(d_in[b, n]) < float(info0["r_need_m"]) for b, n in extra.tolist())))
strict = (d_in < float(info0["r_need_m"])) & has[:, None]
print(" moved set == {clearance < r_need} :", bool((moved == strict).all()),
      f"| n_moved {int(moved.sum())} n_strict {int(strict.sum())}")

print()
print("=== (c) the EXACTLY-zero oracle / selected ADE delta ===")
a_in = ade(fan)
orc_idx = a_in.argmin(dim=1)
oracle_moved = moved.gather(1, orc_idx[:, None])[:, 0]
sel_moved = moved.gather(1, sel[:, None])[:, 0]
oracle_collides = hit_in.gather(1, orc_idx[:, None])[:, 0]
sel_collides = hit_in.gather(1, sel[:, None])[:, 0]
print(f" windows whose ORACLE candidate collides : {int(oracle_collides.sum())} / {B}")
print(f" windows whose ORACLE candidate was moved: {int(oracle_moved.sum())} / {B}")
print(f" windows whose SELECTED candidate collides: {int(sel_collides.sum())} / {B}")
print(f" windows whose SELECTED candidate was moved: {int(sel_moved.sum())} / {B}")
a_out = ade(out0)
print(f" max |ade_out - ade_in| over UNMOVED candidates: "
      f"{float((a_out - a_in)[~moved].abs().max()):.3e}")
print(f" ADE on the MOVED candidates: base mean {float(a_in[moved].mean()):.4f} -> "
      f"proj mean {float(a_out[moved].mean()):.4f} "
      f"(delta {float((a_out - a_in)[moved].mean()):+.4f}, "
      f"max {float((a_out - a_in)[moved].max()):+.4f}, "
      f"min {float((a_out - a_in)[moved].min()):+.4f})")
print(f" of the moved, how many got CLOSER to the human: "
      f"{int(((a_out - a_in)[moved] < 0).sum())} / {int(moved.sum())}")
# the fan's coverage of the human, restricted to the 19 collider windows
rows = torch.nonzero(hit_in.any(1)).ravel()
print(f" oracle ADE on the 19 collider windows: base "
      f"{float(a_in[rows].min(dim=1).values.mean()):.4f} -> "
      f"proj {float(a_out[rows].min(dim=1).values.mean()):.4f}")
print(f" oracle ADE if the colliders had simply been DELETED (the veto's ceiling): "
      f"{float(torch.where(hit_in[rows], torch.full_like(a_in[rows], 1e9), a_in[rows]).min(dim=1).values.mean()):.4f}")

print()
print("=== (d) off_reach direction: is the base fan too FAST or too SLOW? ===")
v0 = torch.tensor(np.asarray(z["v0"], dtype=np.float64))
v_mean_in = fan[..., -1, :].norm(dim=-1) / 2.0
v_mean_out = out0[..., -1, :].norm(dim=-1) / 2.0
lo = (v0[:, None] - 5.0).clamp_min(0.0)
hi = v0[:, None] + 5.0
print(f" base : above band {int((v_mean_in > hi).sum())}  below band {int((v_mean_in < lo).sum())}")
print(f" proj : above band {int((v_mean_out > hi).sum())}  below band {int((v_mean_out < lo).sum())}")
