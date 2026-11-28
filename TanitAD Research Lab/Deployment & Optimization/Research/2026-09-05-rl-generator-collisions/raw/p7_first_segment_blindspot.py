"""P7 -- HOW MUCH CONTACT DOES `rewards._collision`'s FIRST-SEGMENT BLIND SPOT HIDE? 0 GPU.

M39 (c9ab82c) names the defect: `_collision` builds its relative path as

    rel = lead[..., 1:, :] - traj[..., 1:, :]

which DROPS index 0 from both sides. The swept test therefore covers the segments t1->t2,
t2->t3, t3->t4 and NEVER t0->t1 -- so a plan that drives THROUGH a car between t0 and t1 can
read CLEAR. M39 states the defect; it does not state its SIZE, and the size is what decides
whether a "structural zero" on this metric is a zero on the road.

This measures it, and separates two things that must not be merged:

  (A) PLAN-CAUSED, currently missed : the t0->t1 segment passes within r while BOTH endpoints
      are outside -- the ego drives through the obstacle inside the first step. This is a real
      collision the metric does not see.
  (B) PRE-EXISTING at t0            : the lead is ALREADY within r at t0. The ego is at its own
      origin by construction, so this is a property of the SCENE, not of the plan; a planner
      cannot un-cause it. Counted and reported separately, never added to (A).

CONTROLS:
  P1 SUPERSET   : the extended flag must be a strict superset of the stock flag (a wider sweep
                  can never detect LESS). 0 candidates may lose their flag.
  P2 REPRODUCE  : restricting the extended test to segments 1..3 must reproduce the stock flag
                  EXACTLY -- it proves the reimplementation is the same geometry, so any
                  difference is attributable to the first segment alone and to nothing else.
  P3 SCENE-ONLY : (B) must be identical for every candidate within a window (it does not depend
                  on the plan). Windows where it varies by candidate would falsify the reading.
"""
import os, sys, json, argparse, importlib.util

_REPO = os.environ.get("TANITAD_REPO") or r"C:\Users\Admin\tanitad-wt"
for _p in (os.path.join(_REPO, "stack"), os.path.join(_REPO, "taniteval"),
           os.path.join(_REPO, "taniteval", "tools")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from tanitad.rl import rewards as RW


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="raw/fan_bank_base_240w.npz")
    ap.add_argument("--out", default="raw/p7_first_segment_blindspot.json")
    a = ap.parse_args()

    FS = _load_by_path("fan_safety_for_p7",
                       os.path.join(_REPO, "taniteval", "tools", "fan_safety.py"))
    z = np.load(a.bank, allow_pickle=True)
    fan2 = torch.from_numpy(z["fan2"]).double()
    lead5 = torch.from_numpy(z["lead5"]).double()
    has_lead = torch.from_numpy(z["has_lead"]).bool()
    W, K = fan2.shape[0], fan2.shape[1]
    lead = lead5[:, None, :, :].expand(W, K, 5, 2)
    ctx = {"dt": FS.DT_S, "lead_len_m": FS.LEAD_LEN_DEFAULT_M, "lead_path": lead}
    r = 2.0
    print("OBJECT %s  fan2%s  r=%.1f m  lead windows %d/%d"
          % (a.bank, tuple(fan2.shape), r, int(has_lead.sum()), W))

    stock = (RW.COMPONENTS["collision"](fan2, ctx) < 0) & has_lead[:, None]

    # the FULL relative path, index 0 included
    rel_full = lead - fan2                                   # [W,K,5,2]
    origin = torch.zeros_like(rel_full[..., :1, :])

    def sweep(rel):
        d_pt = (rel.unsqueeze(-2) - origin.unsqueeze(-3)).norm(dim=-1).amin(-1).amin(-1)
        d = d_pt
        if rel.shape[-2] >= 2:
            d_seg = RW.segment_point_distance(rel[..., :-1, :], rel[..., 1:, :],
                                              origin).amin(-1).amin(-1)
            d = torch.minimum(d, d_seg)
        return d

    # P2 REPRODUCE: segments 1..3 only == the stock geometry
    d_stock_geom = sweep(rel_full[..., 1:, :])
    repro = ((d_stock_geom < r) & has_lead[:, None])
    p2 = int((repro != stock).sum())
    print("P2 REPRODUCE  : my reimplementation on segments 1..3 vs stock -> disagree=%d  %s"
          % (p2, "PASS" if p2 == 0 else "FAIL"))

    # the extended test: include the t0->t1 segment, EXCLUDE the t0 point (scene, not plan)
    seg0 = RW.segment_point_distance(rel_full[..., 0:1, :], rel_full[..., 1:2, :],
                                     origin).amin(-1).amin(-1)
    d_ext = torch.minimum(d_stock_geom, seg0)
    ext = (d_ext < r) & has_lead[:, None]

    p1 = int((stock & ~ext).sum())
    print("P1 SUPERSET   : candidates that LOSE their flag under a wider sweep = %d  %s"
          % (p1, "PASS" if p1 == 0 else "FAIL"))

    # (B) pre-existing: the lead already inside r at t0 (per WINDOW -- the ego is at its origin)
    d_t0 = rel_full[..., 0, :].norm(dim=-1)                  # [W,K]
    pre = (d_t0 < r) & has_lead[:, None]
    p3 = int((pre.any(dim=1) != pre.all(dim=1)).sum())
    print("P3 SCENE-ONLY : windows where the t0 condition varies BY CANDIDATE = %d  %s"
          % (p3, "PASS" if p3 == 0 else "FAIL"))

    newly = ext & ~stock
    n_new = int(newly.sum())
    n_stock = int(stock.sum())
    print()
    print("=== (A) PLAN-CAUSED CONTACT THE METRIC CURRENTLY MISSES ===")
    print("  stock colliders                 : %d  (fan_contact %.6f)"
          % (n_stock, float(stock.double().mean())))
    print("  + first-segment sweep           : %d  (fan_contact %.6f)"
          % (int(ext.sum()), float(ext.double().mean())))
    print("  NEWLY DETECTED                  : %d   = +%.2f%% of the stock count"
          % (n_new, 100.0 * n_new / max(n_stock, 1)))
    print("  windows gaining a collider      : %d (stock had %d collider-bearing windows)"
          % (int(((newly.any(1)) & ~(stock.any(1))).sum()), int(stock.any(1).sum())))
    print()
    print("=== (B) PRE-EXISTING AT t0 -- a SCENE property, NOT a plan defect ===")
    print("  windows with the lead already within %.1f m at t0: %d / %d lead windows"
          % (r, int(pre.any(1).sum()), int(has_lead.sum())))
    print("  (reported separately and NEVER added to (A): the ego is at its own origin at t0,")
    print("   so no plan can un-cause it)")

    ok = (p1 == 0) and (p2 == 0) and (p3 == 0)
    print()
    print("CONTROLS=%s" % ("PASS" if ok else "FAIL"))

    out = {"_tool": "p7_first_segment_blindspot.py",
           "_tier": "T0 readout on the EMITTED fan (never a driving claim)",
           "_evidence_class": "MEASURED (ours)",
           "bank": a.bank, "contact_radius_m": r,
           "defect": "rewards._collision builds rel = lead[...,1:,:] - traj[...,1:,:], dropping "
                     "index 0, so the t0->t1 segment is never swept",
           "controls": {"P1_lost_flags": p1, "P2_reproduce_disagreements": p2,
                        "P3_t0_varies_by_candidate": p3, "ALL_PASS": bool(ok)},
           "stock_colliders": n_stock,
           "stock_fan_contact": float(stock.double().mean()),
           "extended_colliders": int(ext.sum()),
           "extended_fan_contact": float(ext.double().mean()),
           "newly_detected": n_new,
           "newly_detected_pct_of_stock": 100.0 * n_new / max(n_stock, 1),
           "windows_gaining_a_collider": int(((newly.any(1)) & ~(stock.any(1))).sum()),
           "stock_collider_windows": int(stock.any(1).sum()),
           "preexisting_t0_windows": int(pre.any(1).sum()),
           "n_lead_windows": int(has_lead.sum())}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("WROTE %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
