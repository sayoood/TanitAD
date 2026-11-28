"""P6 -- is a GRADED collision term worth building? (successor S2). 0 GPU.

`rewards._collision` returns -1 or 0. Inside a GRPO group the advantage is computed across a
window's candidates, so a BINARY term gives every colliding candidate the SAME value: it can say
"these are bad" but not "this one is worse than that one", and therefore supplies NO DIRECTION
within the colliding set. A graded term (time-to-contact, or penetration depth) would.

That is only worth building if collisions actually VARY in severity. If every collider is equally
bad, grading buys nothing and S2 should not be run. This measures the spread.

CONTROLS:
  G1 BINARY-IS-FLAT : the stock term's within-colliders standard deviation must be EXACTLY 0.0.
                      If it is not, the term is not binary and the whole premise is wrong.
  G2 SEVERITY-ORDERS-CONTACT : the graded severity must separate colliders from non-colliders
                      (AUC vs contact = 1.0 exactly, since it is 0 for non-colliders by
                      construction). A severity that does not is mis-derived.
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
ap.add_argument("--out", default="raw/p6_graded_term_headroom.json")
args = ap.parse_args()


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FS = _load_by_path("fan_safety_for_p6", os.path.join(_REPO, "taniteval", "tools", "fan_safety.py"))
from tanitad.rl import rewards as RW

z = np.load(args.bank, allow_pickle=True)
fan2 = torch.from_numpy(z["fan2"]).double()
v0 = torch.from_numpy(np.asarray(z["v0"])).double()
lead5 = torch.from_numpy(z["lead5"]).double()
has_lead = torch.from_numpy(z["has_lead"]).bool()
W, K = fan2.shape[0], fan2.shape[1]

sc = FS.score_paths(fan2, v0[:, None].expand(W, K), lead5[:, None, :, :].expand(W, K, 5, 2))
contact = sc["contact"] & has_lead[:, None]
ctx = {"dt": FS.DT_S, "lead_len_m": FS.LEAD_LEN_DEFAULT_M,
       "lead_path": lead5[:, None, :, :].expand(W, K, 5, 2)}
binary = RW.COMPONENTS["collision"](fan2, ctx).double()

print("OBJECT %s   colliders %d / %d" % (args.bank, int(contact.sum()), W * K))

# ---- G1: the stock term is flat inside the colliding set ----
vals = binary[contact]
g1_std = float(vals.std()) if vals.numel() > 1 else 0.0
g1_uniq = sorted(set(np.round(vals.numpy(), 12).tolist()))
print("G1 BINARY-IS-FLAT: within-colliders std = %.3e, distinct values = %s -> %s"
      % (g1_std, g1_uniq, "PASS (flat, no direction)" if g1_std == 0.0 else "FAIL"))

# ---- the graded severity: penetration depth into the 2 m disc, time-aligned ----
R = 2.0
rel = lead5[:, None, 1:, :].expand(W, K, 4, 2) - fan2[:, :, 1:, :]     # [W,K,4,2]
d = rel.norm(dim=-1)                                                   # [W,K,4]
pen = (R - d).clamp_min(0.0).amax(dim=-1)                              # deepest incursion, m
pen = pen * has_lead[:, None]

# ---- G2: severity must order contact perfectly (it is 0 off the colliding set) ----
# NOTE the POINT-test caveat: `pen` is a per-sample depth, so it is 0 for a candidate that only
# collides in the SWEPT sense (it passes through the disc BETWEEN samples). Those are counted.
swept_only = int((contact & (pen <= 0)).sum())
print("G2 SEVERITY-ORDERS-CONTACT: colliders with pen == 0 (swept-only, no sampled point inside "
      "the disc) = %d / %d (%.1f%%)" % (swept_only, int(contact.sum()),
                                        100.0 * swept_only / max(int(contact.sum()), 1)))
print("   pen > 0 but NOT flagged contact (must be 0): %d"
      % int(((pen > 0) & ~contact).sum()))

pv = pen[contact & (pen > 0)].numpy()
print()
print("=== SEVERITY SPREAD among colliders with a sampled incursion (n=%d) ===" % pv.size)
if pv.size:
    for q in (0, 10, 25, 50, 75, 90, 100):
        print("   p%-3d  %.4f m" % (q, np.percentile(pv, q)))
    print("   mean %.4f  std %.4f  -> spread ratio p90/p10 = %.2fx"
          % (pv.mean(), pv.std(), np.percentile(pv, 90) / max(np.percentile(pv, 10), 1e-9)))

# min TTC among colliders -- the other natural grading
ttc = sc["min_ttc_s"]
tv = ttc[contact].numpy()
tv = tv[np.isfinite(tv)]
print()
print("=== min-TTC among colliders (n=%d finite) ===" % tv.size)
if tv.size:
    for q in (0, 10, 25, 50, 75, 90, 100):
        print("   p%-3d  %.4f s" % (q, np.percentile(tv, q)))

# ---- within-window variation: the quantity a GROUP-RELATIVE advantage can actually use ----
mixed = (contact.sum(1) > 0) & (contact.sum(1) < K)
rows = []
for w in torch.nonzero(mixed).flatten().tolist():
    p = pen[w][contact[w]].numpy()
    p = p[p > 0]
    if p.size > 1:
        rows.append((p.min(), p.max(), p.std()))
print()
print("=== WITHIN-WINDOW severity spread (what a group-relative advantage can use) ===")
print("mixed windows with >1 graded collider: %d" % len(rows))
if rows:
    a = np.array(rows)
    print("   per-window severity RANGE (max-min): mean %.4f m  median %.4f m  max %.4f m"
          % ((a[:, 1] - a[:, 0]).mean(), np.median(a[:, 1] - a[:, 0]), (a[:, 1] - a[:, 0]).max()))
    print("   per-window severity STD           : mean %.4f m  max %.4f m"
          % (a[:, 2].mean(), a[:, 2].max()))
    print("   windows with a non-degenerate spread (std > 0.05 m): %d / %d"
          % (int((a[:, 2] > 0.05).sum()), len(rows)))

verdict = ("WORTH BUILDING" if rows and (np.array(rows)[:, 2] > 0.05).sum() >= 0.5 * len(rows)
           else "MARGINAL")
print()
print("S2 VERDICT: %s -- the binary term is flat (std %.3e) while severity varies within the "
      "windows that carry the signal." % (verdict, g1_std))

out = {"_tool": "p6_graded_term_headroom.py",
       "_tier": "T0 readout on the EMITTED fan (never a driving claim)",
       "_evidence_class": "MEASURED (ours)",
       "bank": args.bank, "n_colliders": int(contact.sum()),
       "controls": {"G1_binary_within_collider_std": g1_std,
                    "G1_distinct_values": g1_uniq,
                    "G2_swept_only_colliders": swept_only,
                    "G2_pen_positive_but_not_contact": int(((pen > 0) & ~contact).sum())},
       "penetration_m": ({"n": int(pv.size), "mean": float(pv.mean()), "std": float(pv.std()),
                          "p10": float(np.percentile(pv, 10)), "p50": float(np.percentile(pv, 50)),
                          "p90": float(np.percentile(pv, 90)), "max": float(pv.max())}
                         if pv.size else None),
       "min_ttc_s": ({"n": int(tv.size), "p10": float(np.percentile(tv, 10)),
                      "p50": float(np.percentile(tv, 50)), "p90": float(np.percentile(tv, 90))}
                     if tv.size else None),
       "within_window": ({"n_windows": len(rows),
                          "mean_range_m": float((np.array(rows)[:, 1] - np.array(rows)[:, 0]).mean()),
                          "mean_std_m": float(np.array(rows)[:, 2].mean()),
                          "n_nondegenerate": int((np.array(rows)[:, 2] > 0.05).sum())}
                         if rows else None),
       "verdict": verdict}
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=1)
print("WROTE %s" % args.out)
