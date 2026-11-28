"""P5 -- THE PRICE OF A COLLISION-FREE FAN, measured directly. 0 GPU.

The PI's question is "does the model emit fewer colliding trajectories, by how much, at what
ADE cost". This measures the CEILING of that trade WITHOUT training anything: for every
colliding candidate, find the smallest longitudinal retreat that clears the 2 m contact disc,
and price it in metres.

WHY BRAKING AND NOT AN ARBITRARY PERTURBATION. `fan_contact` is a property of the candidate
SET, not of the ranking -- no re-weighting can change it, so the generator has to MOVE the
paths. Contact here is with a time-aligned LEAD, and 88.7 % of the programme's measured gap is
longitudinal, so "travel less far along your own path" is the physically meaningful minimal fix.
The path is scaled toward the ego origin by s in [0, 1]: shape preserved, distance reduced.

⚠️ This is an UPPER BOUND ON THE COST and a LOWER BOUND ON THE ACHIEVABLE RATE for this family
of fixes. A cleverer fix (lateral evasion) could be cheaper; a shape-preserving scale is the one
whose cost is unambiguous and whose feasibility is checkable.

CONTROLS:
  Q1 IDENTITY   : s = 1.0 must reproduce the unmodified fan's contact EXACTLY (it IS the fan).
  Q2 TERMINAL   : s = 0.0 is the ego origin held for 2 s. It must clear contact for every
                  candidate -- if it does not, the LEAD is already inside the disc at t0 and
                  that window is UNFIXABLE BY BRAKING; those are counted and reported, never
                  silently scored as fixed.
  Q3 MONOTONE   : contact must be non-increasing in decreasing s within a window; violations
                  are counted (a non-monotone case means the swept test re-enters the disc).
"""
import os, sys, json, argparse, importlib.util

_REPO = os.environ.get("TANITAD_REPO") or r"C:\Users\Admin\tanitad-wt"
for _p in (os.path.join(_REPO, "stack"), os.path.join(_REPO, "taniteval"),
           os.path.join(_REPO, "taniteval", "tools")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default="raw/fan_bank_base_240w.npz")
ap.add_argument("--out", default="raw/p5_collision_price.json")
ap.add_argument("--n-grid", type=int, default=101)
ap.add_argument("--n-boot", type=int, default=4000)
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FS = _load_by_path("fan_safety_for_p5", os.path.join(_REPO, "taniteval", "tools", "fan_safety.py"))
from tanitad.refs import feasible_decode as FD

z = np.load(args.bank, allow_pickle=True)
fan2 = torch.from_numpy(z["fan2"]).double()
v0 = torch.from_numpy(np.asarray(z["v0"])).double()
lead5 = torch.from_numpy(z["lead5"]).double()
has_lead = torch.from_numpy(z["has_lead"]).bool()
eid = np.asarray(z["eid"])
W, K = fan2.shape[0], fan2.shape[1]
print("OBJECT %s  fan2%s  lead windows %d/%d" % (args.bank, tuple(fan2.shape), int(has_lead.sum()), W))


def contact_of(paths):
    sc = FS.score_paths(paths, v0[:, None].expand(W, K),
                        lead5[:, None, :, :].expand(W, K, 5, 2))
    return (sc["contact"] & has_lead[:, None])


base_contact = contact_of(fan2)
print("base fan_contact = %.6f  (%d colliders)" % (base_contact.double().mean(), int(base_contact.sum())))

# ---- sweep s from 1.0 down to 0.0 ----
grid = torch.linspace(1.0, 0.0, args.n_grid).double()
still = torch.ones(W, K, dtype=torch.bool)          # still colliding at the current s
s_clear = torch.full((W, K), float("nan"), dtype=torch.double)
prev = None
nonmono = 0
for gi, s in enumerate(grid):
    c = contact_of(fan2 * s)
    if gi == 0:
        q1_dis = int((c != base_contact).sum())
        print("Q1 IDENTITY  s=1.0 reproduces the fan's contact: disagree=%d -> %s"
              % (q1_dis, "PASS" if q1_dis == 0 else "FAIL"))
    newly = still & ~c
    s_clear[newly] = float(s)
    still = still & c
    if prev is not None:
        nonmono += int((c & ~prev).sum())
    prev = c
q2_left = int(still.sum())
q2_left_of_colliders = int((still & base_contact).sum())
print("Q2 TERMINAL  s=0.0 (hold the origin) still colliding: %d candidates (%d of them were "
      "colliding at s=1) -> %s" % (q2_left, q2_left_of_colliders,
                                   "PASS" if q2_left == 0 else "UNFIXABLE-BY-BRAKING present"))
print("Q3 MONOTONE  re-entries into contact as s decreases: %d -> %s"
      % (nonmono, "PASS" if nonmono == 0 else "non-monotone cases present"))

# ---- the price ----
arclen = fan2[..., 1:, :].norm(dim=-1).mean(dim=-1)          # [W,K] mean |p| over the 4 points
fixable = base_contact & torch.isfinite(s_clear)
cost = torch.zeros(W, K, dtype=torch.double)
cost[fixable] = (1.0 - s_clear[fixable]) * arclen[fixable]   # mean displacement, metres

n_coll = int(base_contact.sum())
n_fix = int(fixable.sum())
n_unfix = n_coll - n_fix
print()
print("=== THE PRICE ===")
print("colliding candidates            : %d" % n_coll)
print("  fixable by braking alone      : %d (%.1f%%)" % (n_fix, 100.0 * n_fix / max(n_coll, 1)))
print("  UNFIXABLE by braking          : %d (%.1f%%)  <- lead already inside the disc at t0"
      % (n_unfix, 100.0 * n_unfix / max(n_coll, 1)))
if n_fix:
    cv = cost[fixable].numpy()
    sv = s_clear[fixable].numpy()
    print("displacement to clear, over FIXABLE colliders (m):")
    print("  mean %.4f  median %.4f  p90 %.4f  max %.4f" %
          (cv.mean(), np.median(cv), np.percentile(cv, 90), cv.max()))
    print("retreat fraction (1-s) needed: mean %.4f  median %.4f  max %.4f"
          % ((1 - sv).mean(), np.median(1 - sv), (1 - sv).max()))

# amortised over the WHOLE fan -- the number that is comparable to an ADE delta
amort_all = float(cost.mean())
amort_lead = float(cost[has_lead].mean())
print()
print("AMORTISED over every emitted candidate (this is the ADE-comparable number):")
print("  all %d windows  : %.6f m   (fan_contact %.6f -> %.6f)"
      % (W, amort_all, float(base_contact.double().mean()),
         float((base_contact & ~fixable).double().mean())))
print("  %d lead windows : %.6f m   (fan_contact %.6f -> %.6f)"
      % (int(has_lead.sum()), amort_lead, float(base_contact[has_lead].double().mean()),
         float((base_contact & ~fixable)[has_lead].double().mean())))

# ---- does the braked path stay feasible? (a fix that breaks the envelope is not a fix) ----
fixed = fan2 * torch.where(torch.isfinite(s_clear), s_clear, torch.ones_like(s_clear))[..., None, None]
sc_fixed = FS.score_paths(fixed, v0[:, None].expand(W, K), lead5[:, None, :, :].expand(W, K, 5, 2))
sc_base = FS.score_paths(fan2, v0[:, None].expand(W, K), lead5[:, None, :, :].expand(W, K, 5, 2))
print()
print("FEASIBILITY OF THE BRAKED FAN (a fix that breaks the envelope is not a fix):")
for f in ("infeasible", "kamm_over", "envelope", "off_reach"):
    print("  %-11s %.6f -> %.6f" % (f, float(sc_base[f].double().mean()), float(sc_fixed[f].double().mean())))


def boot(vals, ep, n_boot, seed):
    rng = np.random.default_rng(seed)
    ue = np.unique(ep)
    idx = [np.nonzero(ep == e)[0] for e in ue]
    o = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.integers(0, len(ue), len(ue))
        o[i] = vals[np.concatenate([idx[j] for j in p])].mean()
    return float(vals.mean()), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


m, lo, hi = boot(cost.mean(dim=1).numpy(), eid, args.n_boot, args.seed)
print()
print("amortised cost, episode-cluster bootstrap: %.6f m [%.6f, %.6f]" % (m, lo, hi))

out = {"_tool": "p5_collision_price.py",
       "_tier": "T0 readout on the EMITTED fan (never a driving claim)",
       "_evidence_class": "MEASURED (ours)",
       "bank": args.bank, "W": W, "K": K, "n_lead_windows": int(has_lead.sum()),
       "contact_definition": "SWEPT relative-frame, radius 2.0 m (rewards._collision)",
       "controls": {"Q1_identity_disagreements": q1_dis,
                    "Q2_still_colliding_at_s0": q2_left,
                    "Q2_unfixable_colliders": q2_left_of_colliders,
                    "Q3_nonmonotone_reentries": nonmono},
       "base_fan_contact": float(base_contact.double().mean()),
       "n_colliders": n_coll, "n_fixable_by_braking": n_fix, "n_unfixable": n_unfix,
       "displacement_to_clear_m": ({"mean": float(cost[fixable].mean()),
                                    "median": float(np.median(cost[fixable].numpy())),
                                    "p90": float(np.percentile(cost[fixable].numpy(), 90)),
                                    "max": float(cost[fixable].max())} if n_fix else None),
       "amortised_cost_m": {"all_windows": amort_all, "lead_windows": amort_lead,
                            "boot_mean": m, "lo": lo, "hi": hi},
       "achievable_fan_contact": {"all_windows": float((base_contact & ~fixable).double().mean()),
                                  "lead_windows": float((base_contact & ~fixable)[has_lead].double().mean())},
       "braked_fan_feasibility": {f: {"before": float(sc_base[f].double().mean()),
                                      "after": float(sc_fixed[f].double().mean())}
                                  for f in ("infeasible", "kamm_over", "envelope", "off_reach")}}
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=1)
print("WROTE %s" % args.out)
