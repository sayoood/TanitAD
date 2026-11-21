#!/usr/bin/env python3
"""H-PROJ-CONTACT-1, part 3 -- the hole the projection found in the PREDICATE, and
the clearability theorem, both measured on the same banked fan.

⛔ THE HOLE. `rewards._collision`'s moving-lead branch slices `rel = lead[..., 1:, :]
- traj[..., 1:, :]` and sweeps THAT polyline. The comment justifies dropping the
POINT at s = 0 ("the ego is at its own origin; the lead is ahead by construction"),
and that justification is sound -- but the SWEPT fix (commit 9765634) inherited the
same slice, so the relative segment from t = 0 to t = 0.5 s is **never tested**. That
is the D-SWEPT-1 corridor again, at the one step where the ego's displacement is
largest, and it is invisible to every number banked under the current predicate.

This file measures the size of the hole and re-runs the projection against the
STRICTER predicate, because a projection that only enforces a predicate with a hole
inherits the hole -- which is exactly what the brief's failure #1 says to avoid.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = r"C:\Users\Admin\collproj"
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from tanitad.refs import contact_projection as CP           # noqa: E402
from tanitad.rl.rewards import _collision                    # noqa: E402
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
OUT = r"C:\Users\Admin\collproj\out\contact_projection_panel3.json"
R = CP.CONTACT_R_M


def main():
    z = np.load(BANK, allow_pickle=True)
    gz = np.load(GT, allow_pickle=True)
    fan = torch.tensor(np.asarray(z["fan2"]), dtype=torch.float64)
    lead = torch.tensor(np.asarray(z["lead5"]), dtype=torch.float64)[:, None]
    has = torch.tensor(np.asarray(z["has_lead"]).astype(bool))
    v0 = torch.tensor(np.asarray(z["v0"], dtype=np.float64))
    sel = torch.tensor(np.asarray(z["sel_idx"]).astype(np.int64))
    rank = torch.tensor(np.asarray(z["rank"]), dtype=torch.float32)
    gt = torch.tensor(np.asarray(gz["gt"]), dtype=torch.float64)
    B, N, S, _ = fan.shape
    rec = {"_what": "H-PROJ-CONTACT-1 part 3: the first-segment hole in the contact "
                    "predicate, and the clearability theorem",
           "_evidence_class": "MEASURED (ours; rule-based, 0 GPU)", "bank": BANK}
    ldx = lead.expand_as(fan)

    def ade(p):
        return (p[..., 1:, :] - gt[:, None, :, :]).norm(dim=-1).mean(dim=-1)

    # ------------------------------------------------------------------ #
    # (1) THE FIRST-SEGMENT HOLE                                          #
    # ------------------------------------------------------------------ #
    d_scorer = CP.min_rel_distance(fan, ldx, skip_first=True)     # what the scorer sees
    d_full = CP.min_rel_distance(fan, ldx, skip_first=False)      # every segment swept
    d0 = (ldx[..., 0, :] - fan[..., 0, :]).norm(dim=-1)           # |rel| at t = 0
    already = (d0 < R)                                            # in contact at t0
    hit_scorer = (d_scorer < R) & has[:, None]
    hit_full = (d_full < R) & has[:, None]
    hole = hit_full & (~hit_scorer) & (~already)
    # the CONTROL that makes the hole a claim about the predicate and not about my
    # arithmetic: the scorer's own function must agree with `d_scorer < R`, exactly.
    hit_rw = (_collision(fan, {"lead_path": ldx, "ego_radius_m": 1.0,
                               "obs_radius_m": 1.0}) < 0) & has[:, None]
    rec["first_segment_hole"] = {
        "_control_predicate_agreement": {
            "known_value": "min_rel_distance(skip_first=True) < r must equal "
                           "rewards._collision exactly",
            "disagreements": int((hit_rw != hit_scorer).sum()),
            "PASS": bool((hit_rw == hit_scorer).all())},
        "n_flagged_by_scorer": int(hit_scorer.sum()),
        "n_flagged_full_sweep": int(hit_full.sum()),
        "n_in_contact_at_t0": int((already & has[:, None]).sum()),
        "n_in_the_hole": int(hole.sum()),
        "hole_rate_all": float(hole.double().mean()),
        "hole_rate_lead_pop": float(hole[has].double().mean()),
        "hole_windows": sorted(torch.nonzero(hole.any(1)).ravel().tolist()),
        "n_hole_windows": int(hole.any(1).sum()),
        "sel_in_the_hole": int(hole.gather(1, sel[:, None])[:, 0].sum()),
        "top32_hole_rate_lead": float(hole.to(torch.float32).gather(
            1, rank.argsort(1, descending=True)[:, :32])[has].mean()),
        "relative_increase_pct": (100.0 * int(hole.sum()) / max(1, int(hit_scorer.sum()))),
        "_what_it_is": ("candidates the CURRENT predicate calls clear, that a full "
                        "sweep including the t=0 -> t=0.5 s relative segment calls "
                        "contact, EXCLUDING the ones already inside the disc at t=0 "
                        "(an initial condition, not a plan defect)"),
        "_first_step_len_m": {"mean": float((fan[..., 1, :].norm(dim=-1))[has].mean()),
                              "p95": float(torch.quantile(
                                  fan[..., 1, :].norm(dim=-1)[has].flatten(), 0.95)),
                              "max": float(fan[..., 1, :].norm(dim=-1)[has].max())},
    }
    h = rec["first_segment_hole"]
    print(f"[p3] HOLE: scorer flags {h['n_flagged_by_scorer']}, full sweep flags "
          f"{h['n_flagged_full_sweep']}, already-in-contact-at-t0 "
          f"{h['n_in_contact_at_t0']}, HOLE {h['n_in_the_hole']} "
          f"(+{h['relative_increase_pct']:.1f} % on the scorer's count) in "
          f"{h['n_hole_windows']} windows; sel in hole {h['sel_in_the_hole']}",
          flush=True)

    # ------------------------------------------------------------------ #
    # (2) THE PROJECTION AGAINST THE STRICTER PREDICATE                   #
    # ------------------------------------------------------------------ #
    a_in = ade(fan)
    out_s, info_s = CP.project_contact_free(fan, lead, has, margin_m=0.0,
                                            skip_first=False)
    d_out_full = CP.min_rel_distance(out_s, lead.expand_as(out_s), skip_first=False)
    d_out_sc = CP.min_rel_distance(out_s, lead.expand_as(out_s), skip_first=True)
    sc = FS.score_paths(out_s.float(), v0.float(), lead.float())
    a_out = ade(out_s)
    moved = ((out_s - fan).norm(dim=-1).amax(dim=-1) > 0)
    tax = {CP.REASON_NAMES[k]: int((info_s["reason"] == k).sum()) for k in CP.REASON_NAMES}
    unav = (info_s["reason"] == CP.REASON_UNAVOIDABLE)
    rec["strict_predicate_arm"] = {
        "predicate": "every relative segment swept, INCLUDING t=0 -> t=0.5 s",
        "fan_contact_scorer_predicate": float(((d_out_sc < R) & has[:, None]).double().mean()),
        "fan_contact_full_sweep": float(((d_out_full < R) & has[:, None]).double().mean()),
        "fan_contact_full_sweep_count": int(((d_out_full < R) & has[:, None]).sum()),
        "taxonomy": tax,
        "n_unavoidable": int(unav.sum()),
        "n_unavoidable_that_are_already_in_contact_at_t0":
            int((unav & already).sum()),
        "n_unavoidable_not_explained_by_t0": int((unav & ~already).sum()),
        "n_moved": int(moved.sum()), "n_moved_windows": int(moved.any(1).sum()),
        "oracle_in_fan_ade_m": float(a_out.min(dim=1).values.mean()),
        "d_oracle_ade_m": float((a_out.min(dim=1).values - a_in.min(dim=1).values).mean()),
        "sel_ade_m": float(a_out.gather(1, sel[:, None])[:, 0].mean()),
        "d_sel_ade_m": float((a_out - a_in).gather(1, sel[:, None])[:, 0].mean()),
        "fan_off_reach": float(sc["off_reach"].double().mean()),
        "fan_peak_g_mean": float(sc["peak_g"].mean()),
        "sigma_mean_moved": float(info_s["sigma"][moved].mean()) if bool(moved.any()) else 1.0,
    }
    s3 = rec["strict_predicate_arm"]
    print(f"[p3] STRICT arm: full-sweep contact {s3['fan_contact_full_sweep']:.6f} "
          f"({s3['fan_contact_full_sweep_count']}) unavoidable {s3['n_unavoidable']} "
          f"(of which already-in-contact-at-t0 "
          f"{s3['n_unavoidable_that_are_already_in_contact_at_t0']}) moved "
          f"{s3['n_moved']} dOrcADE {s3['d_oracle_ade_m']:+.4f} "
          f"dSelADE {s3['d_sel_ade_m']:+.4f} off_reach {s3['fan_off_reach']:.4f}",
          flush=True)

    # ------------------------------------------------------------------ #
    # (3) THE CLEARABILITY THEOREM, checked rather than argued            #
    #     "sigma -> 0 clears every contact with an agent that is outside  #
    #      the ego's disc for the whole window while the ego is at rest"  #
    # ------------------------------------------------------------------ #
    rest = torch.zeros_like(fan)
    d_rest_full = CP.min_rel_distance(rest, ldx, skip_first=False)
    d_rest_sc = CP.min_rel_distance(rest, ldx, skip_first=True)
    rest_clear_sc = (d_rest_sc >= R)
    rest_clear_full = (d_rest_full >= R)
    # every candidate whose window is rest-clear must be clearable, and the module
    # must have found it: reason != UNAVOIDABLE
    viol_sc = (rest_clear_sc & has[:, None] &
               (CP.project_contact_free(fan, lead, has, margin_m=0.0)[1]["reason"]
                == CP.REASON_UNAVOIDABLE))
    viol_full = (rest_clear_full & has[:, None] & unav)
    rec["clearability_theorem"] = {
        "_statement": ("if the agent track stays outside the ego's disc for the whole "
                       "window while the ego does not move, then sigma -> 0 is a "
                       "collision-free member and the search must find one"),
        "n_windows_rest_clear_scorer_predicate": int((rest_clear_sc[:, 0] & has).sum()),
        "n_windows_rest_clear_full_sweep": int((rest_clear_full[:, 0] & has).sum()),
        "n_lead_windows": int(has.sum()),
        "violations_scorer_predicate": int(viol_sc.sum()),
        "violations_full_sweep": int(viol_full.sum()),
        "PASS": bool(int(viol_sc.sum()) == 0 and int(viol_full.sum()) == 0),
        "_corollary": ("the ONLY unclearable contacts are those in which the AGENT's "
                       "own motion enters a stationary ego's disc -- an agent driving "
                       "into a parked car. No candidate set can be constrained out of "
                       "that, and no planner can avoid it; it is a perception / "
                       "prediction problem, not a generation one."),
        "min_rest_distance_over_lead_windows_m": float(d_rest_full[has].min()),
        "windows_not_rest_clear": sorted(
            torch.nonzero((~rest_clear_full[:, 0]) & has).ravel().tolist()),
    }
    ct = rec["clearability_theorem"]
    print(f"[p3] THEOREM: rest-clear windows {ct['n_windows_rest_clear_full_sweep']}"
          f"/{ct['n_lead_windows']} lead windows; violations "
          f"{ct['violations_full_sweep']} (full) / {ct['violations_scorer_predicate']} "
          f"(scorer); PASS={ct['PASS']}", flush=True)

    # ------------------------------------------------------------------ #
    # (4) off_reach ON THE MOVED CANDIDATES -- the trade, priced           #
    # ------------------------------------------------------------------ #
    out0, info0 = CP.project_contact_free(fan, lead, has, margin_m=0.0)
    m0 = ((out0 - fan).norm(dim=-1).amax(dim=-1) > 0)
    sc_in = FS.score_paths(fan.float(), v0.float(), lead.float())
    sc_0 = FS.score_paths(out0.float(), v0.float(), lead.float())
    v_in = fan[..., -1, :].norm(dim=-1) / 2.0
    v_out = out0[..., -1, :].norm(dim=-1) / 2.0
    lo = (v0[:, None] - 5.0).clamp_min(0.0)
    hi = v0[:, None] + 5.0
    rec["off_reach_trade"] = {
        "fan_off_reach_base": float(sc_in["off_reach"].double().mean()),
        "fan_off_reach_proj": float(sc_0["off_reach"].double().mean()),
        "base_above_band": int((v_in > hi).sum()), "base_below_band": int((v_in < lo).sum()),
        "proj_above_band": int((v_out > hi).sum()), "proj_below_band": int((v_out < lo).sum()),
        "off_reach_on_moved_base": float(sc_in["off_reach"].double()[m0].mean()),
        "off_reach_on_moved_proj": float(sc_0["off_reach"].double()[m0].mean()),
        "n_newly_off_reach": int((sc_0["off_reach"] & ~sc_in["off_reach"]).sum()),
        "n_newly_in_reach": int((~sc_0["off_reach"] & sc_in["off_reach"]).sum()),
        "_reading": ("a NEGATIVE net is the opposite of the trade feasible_decode's "
                     "scoping document warned about; it must be reported with its "
                     "mechanism (which side of the S2 band the base fan sat on) and "
                     "never as a free win"),
    }
    ot = rec["off_reach_trade"]
    print(f"[p3] off_reach {ot['fan_off_reach_base']:.4f} -> {ot['fan_off_reach_proj']:.4f} "
          f"| newly OFF {ot['n_newly_off_reach']} newly IN {ot['n_newly_in_reach']} "
          f"| base above/below band {ot['base_above_band']}/{ot['base_below_band']} "
          f"-> proj {ot['proj_above_band']}/{ot['proj_below_band']}", flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(f"[p3] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
