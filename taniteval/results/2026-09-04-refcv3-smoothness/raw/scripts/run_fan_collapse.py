"""WP-6 item 4: does the DECODE collapse the fan's manoeuvre span?

The reach gate acts on the DECODED fan `x = anchors + offset`. Anchor-side we
measured that turning candidates always exist and always survive. This asks what
the OFFSET does to the span, on the one arm whose decoded fan is banked in-repo.
"""
import sys, numpy as np, torch
sys.path.insert(0, r"C:/Users/Admin/_wp56"); sys.path.insert(0, r"C:/Users/Admin/_wp56/wp56")
import smooth_lib as S
from tanitad.refs.refc import default_anchors
from tanitad.refs.refc_select import reachability_mask

fb = torch.load(r"C:/Users/Admin/_wp56/taniteval/results/fan_refc-base-30k.pt",
                map_location="cpu", weights_only=False)
fan, v0, sel, gt = fb["fan"], fb["v0"], fb["sel"], fb["gt"]
bear = torch.rad2deg(torch.atan2(fan[..., -1, 1], fan[..., -1, 0]))   # [881,128]
gtb = torch.rad2deg(torch.atan2(gt[:, -1, 1], gt[:, -1, 0]))
keep = reachability_mask(fan, v0, accel_max=2.5, horizon_s=2.0)
print("== refc-base DECODED fan, 881 canonical val windows: terminal-bearing span ==")
print(f"  per-window fan bearing span (max-min): p50 {bear.max(1).values.sub(bear.min(1).values).median():7.2f} deg")
print(f"  per-window s.d. of fan bearing        : p50 {bear.std(1).median():7.2f} deg")
for thr in (10, 30, 60):
    n = (bear.abs() > thr).sum(1).float()
    nk = ((bear.abs() > thr) & keep).sum(1).float()
    print(f"  candidates with |bearing| > {thr:2d} deg : mean {n.mean():6.2f}/128 emitted, "
          f"{nk.mean():6.2f} surviving reach_keep; windows with ZERO survivors "
          f"{(nk == 0).float().mean()*100:5.1f} %")
print(f"  GT terminal bearing |deg| : p50 {gtb.abs().median():.2f} p90 "
      f"{gtb.abs().float().quantile(0.9):.2f} max {gtb.abs().max():.2f}")
hard = gtb.abs() > 10
print(f"\n  on the {int(hard.sum())} windows where GT turns > 10 deg:")
selb = bear[torch.arange(len(sel)), sel]
print(f"    GT bearing p50 {gtb[hard].abs().median():.2f} deg ; SELECTED bearing p50 "
      f"{selb[hard].abs().median():.2f} deg  -> the model commits "
      f"{float(selb[hard].abs().median()/gtb[hard].abs().median()):.2f}x of GT's turn")
best = bear[hard].sub(gtb[hard][:, None]).abs().min(1).values
print(f"    BEST-matching candidate in the emitted fan is {best.median():.2f} deg off GT (p50)")
bestk = bear[hard].sub(gtb[hard][:, None]).abs().masked_fill(~keep[hard], 1e9).min(1).values
print(f"    ...restricted to reach_keep SURVIVORS: {bestk.median():.2f} deg off GT (p50)"
      f"   -> the gate costs {float(bestk.median()-best.median()):+.3f} deg")
