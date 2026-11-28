"""Bit-identity + correctness harness for the refcv4-b decoder change.

Run once against the PRISTINE refs and once against the PATCHED refs; the two
runs must produce IDENTICAL tensors on the fixed-vocabulary path. Then the
patched run additionally asserts, by CONTENT, that the v0-conditioned bank is
the same arithmetic as the offline builder's `unicycle_paths` roll.

usage:  python equiv_test.py <out.pt> [--v0]
"""
import sys

import torch

sys.path.insert(0, r"C:\Users\Admin\run_refcv4v\repo\stack")
from tanitad.refs import refc, refc_v3                            # noqa: E402

OUT = sys.argv[1]
WITH_V0 = "--v0" in sys.argv

torch.manual_seed(0)
cfg = refc_v3.refc_v3_smoke_config(hier=True)
cfg.core.anchors.n_anchors = 21          # 3 accel x 7 curvature, 0 in both
if WITH_V0:
    cfg.core.anchors.v0_conditioned = True
    cfg.core.anchors.ref_speed_ms = 10.0
model = refc_v3.RefCV3Model(cfg).eval()

n_anchor = model.core.decoder.anchors.shape[0]
S = model.core.decoder.anchors.shape[1]
print("anchors [%d, %d, 2]  horizons %s  vocab %s"
      % (n_anchor, S, cfg.core.trajectory.horizons,
         getattr(cfg, "tac_vocab_version", "kin3")))

B, W = 3, cfg.core.obs_window if hasattr(cfg.core, "obs_window") else 4
frames = torch.randn(B, 4, 1, 64, 64)
nav = torch.tensor([0, 1, 2])
v0 = torch.tensor([3.0, 12.0, 24.0])

res = {}
if WITH_V0:
    # controls: a deterministic (a, kappa) product grid, 0 present in both axes
    a_g = torch.linspace(-4.0, 3.0, 3)
    a_g = torch.clamp(a_g - a_g[a_g.abs().argmin()], -4.0, 3.0)
    k_g = torch.linspace(-0.06, 0.06, 7)
    assert (a_g == 0).any() and (k_g == 0).any()
    ctrl = torch.stack(torch.meshgrid(a_g, k_g, indexing="ij"), -1).reshape(-1, 2)
    assert ctrl.shape[0] == n_anchor, (ctrl.shape, n_anchor)
    from tanitad.refs.refa_v1_plan import unicycle_paths
    slots = [k - 1 for k in cfg.core.trajectory.horizons]
    ref = unicycle_paths(ctrl[:, None, :].expand(-1, max(cfg.core.trajectory.horizons), -1).contiguous(),
                         torch.tensor(10.0), 0.1, action_units="kappa")[:, slots]
    model.core.decoder.load_anchors(ref.float(), ctrl.float())
    res["controls"] = ctrl

with torch.no_grad():
    out = model(frames, nav_cmd=nav, v0=v0, steps=0)

for k in ("anchor_logits", "anchor_traj", "offset", "sel_score", "traj",
          "sel_idx", "maneuver_logits", "route"):
    if k in out and torch.is_tensor(out[k]):
        res[k] = out[k].detach().clone()
if "anchor_bank" in out:
    res["anchor_bank"] = out["anchor_bank"].detach().clone()
res["anchors_buffer"] = model.core.decoder.anchors.detach().clone()

if WITH_V0:
    from tanitad.refs.refa_v1_plan import unicycle_paths
    ctrl = res["controls"]
    H = max(cfg.core.trajectory.horizons)
    slots = [k - 1 for k in cfg.core.trajectory.horizons]
    bank = res["anchor_bank"]
    print("\n=== v0-conditioned bank, verified by CONTENT against the "
          "offline builder ===")
    worst = 0.0
    for b in range(B):
        exp = unicycle_paths(
            ctrl[:, None, :].expand(-1, H, -1).contiguous(),
            v0[b], 0.1, action_units="kappa")[:, slots]
        d = (bank[b] - exp).abs().max().item()
        worst = max(worst, d)
        print("  row %d  v0=%5.2f  max|bank - unicycle_paths| = %.3e"
              % (b, v0[b], d))
    assert worst < 1e-4, worst
    # the three rows must differ from each other -- a bank that is the SAME for
    # every window is not conditioned on anything.
    d01 = (bank[0] - bank[1]).abs().max().item()
    print("  rows differ (v0 3.0 vs 12.0): max abs diff %.4f m" % d01)
    assert d01 > 1.0, d01
    # kappa = 0 anchors must be straight lines: y == 0 exactly
    straight = (ctrl[:, 1] == 0)
    print("  kappa==0 anchors: %d, max |y| over them = %.3e"
          % (int(straight.sum()), bank[:, straight, :, 1].abs().max().item()))
    assert bank[:, straight, :, 1].abs().max().item() < 1e-6
    # the (a=0, kappa=0) anchor must be the constant-velocity path
    zi = int(((ctrl[:, 0] == 0) & (ctrl[:, 1] == 0)).nonzero()[0, 0])
    t = torch.tensor([k * 0.1 for k in cfg.core.trajectory.horizons])
    for b in range(B):
        cv = torch.stack([v0[b] * t, torch.zeros_like(t)], -1)
        e = (bank[b, zi] - cv).abs().max().item()
        print("  row %d  {a=0,kappa=0} vs (v0*t, 0): max abs err %.3e m" % (b, e))
        assert e < 1e-4, e
    print("  ego_keep guard: withheld rows roll at ref_speed_ms")
    with torch.no_grad():
        bk = model.core.decoder.roll_bank(
            v0, torch.tensor([True, False, True]), B, torch.float32)
    refb = model.core.decoder.roll_bank(
        torch.tensor([10.0, 10.0, 10.0]), None, B, torch.float32)
    same = (bk[1] - refb[1]).abs().max().item()
    diff = (bk[0] - refb[0]).abs().max().item()
    print("    withheld row 1 vs ref-speed bank: %.3e (must be ~0)" % same)
    print("    kept     row 0 vs ref-speed bank: %.4f (must be > 0)" % diff)
    assert same < 1e-5 and diff > 1.0

torch.save(res, OUT)
print("\nwrote %s  keys=%s" % (OUT, sorted(res.keys())))
