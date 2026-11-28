"""test_p4_pool_shortcut.py -- pins the two pieces of P4's injection arithmetic against the run's REAL modules (tree fb47a583763f).

(1) IDENTITY: `_recompute(br, f0, None)` == `PlannerBEVPool(f0)` / `bev_tokens` BIT-FOR-BIT (the oracle switched off = BASE).
(2) SHORTCUT: `_recompute(br, f0, dp)` == the pool applied to f0 + delta, where delta is placed EXPLICITLY at full 0.25 m resolution on the
    planner crop (nearest-upsampled dp, written into rows [0, rows) x cols [col0, col0 + cols) by index -- independent of F.pad's argument
    order, which is what p4_roll uses for the optimisation).  Expected: equal to float tolerance (avg of 4 equal cells = the cell).
(3) PAD ORDER: p4_roll's F.pad placement == the explicit index placement, exactly.
(4) MUTATION: a delta shifted by ONE 0.25 m column must FAIL (2) -- the test can see a misplacement.
Run:  PYTHONPATH=<tree>/stack python -X utf8 test_p4_pool_shortcut.py
"""
import sys
import types

import torch
import torch.nn.functional as F

from tanitad.data.bev_raster import GRID_DEFAULT
from tanitad.data import semantic_map_gt_fine as sem
from tanitad.models.refcv6_perception_branch import PlannerBEVPool


def _recompute(br, f0, dp):                       # == p4_roll._recompute (copied verbatim; a drift is caught by the source check below)
    bp = br.bev_pool
    w = f0[:, :, :bp.rows, bp.col0:bp.col0 + bp.cols]
    p = F.avg_pool2d(w, kernel_size=bp.k, stride=bp.k)
    if dp is not None:
        p = p + dp.to(device=p.device, dtype=p.dtype)[None]
    feats = bp.proj(p)
    return feats, br.bev_tokens(feats)


def main() -> int:
    src = open(__file__.replace("test_p4_pool_shortcut.py", "p4_roll.py"), encoding="utf-8").read()
    body = "    bp = br.bev_pool\n    w = f0[:, :, :bp.rows, bp.col0:bp.col0 + bp.cols]\n    p = F.avg_pool2d(w, kernel_size=bp.k, stride=bp.k)\n"
    assert body in src, "p4_roll._recompute drifted from the pinned copy"
    torch.manual_seed(0)
    ext = sem.MapExtent(100.0, 30.0)
    bp = PlannerBEVPool(64, 96, src_grid=ext.grid(0.25), dst_grid=GRID_DEFAULT).eval()
    with torch.no_grad():
        for prm in bp.parameters():
            prm.normal_(0, 0.2)

    def tokens(feats):                              # PerceptionBranch.bev_tokens: adaptive pool to 30x16 + flatten (zero params)
        from tanitad.refs.refcv6_tactical import bev_feats_to_tokens
        return bev_feats_to_tokens(F.adaptive_avg_pool2d(feats, (30, 16)))
    br = types.SimpleNamespace(bev_pool=bp, bev_tokens=tokens)
    f0 = torch.randn(2, 64, 400, 240)
    res = {}
    with torch.no_grad():
        ref = bp(f0)
        f_id, t_id = _recompute(br, f0, None)
        res["identity_feats_equal"] = bool(torch.equal(f_id, ref))
        res["identity_tokens_equal"] = bool(torch.equal(t_id, tokens(ref)))
        dp = torch.randn(64, bp.rows // 2, bp.cols // 2)
        up = F.interpolate(dp[None], scale_factor=2, mode="nearest")[0]
        full = torch.zeros(64, 400, 240)
        full[:, :bp.rows, bp.col0:bp.col0 + bp.cols] = up             # explicit index placement
        f_sc, _ = _recompute(br, f0, dp)
        f_ex = bp(f0 + full[None])
        res["shortcut_maxdiff"] = float((f_sc - f_ex).abs().max())
        res["shortcut_ok"] = res["shortcut_maxdiff"] < 1e-4
        pad = (bp.col0, 240 - bp.col0 - bp.cols, 0, 400 - bp.rows)       # p4_roll.invert's F.pad
        res["pad_equals_index"] = bool(torch.equal(F.pad(up[None], pad)[0], full))
        bad = torch.zeros_like(full)
        bad[:, :bp.rows, bp.col0 + 1:bp.col0 + bp.cols + 1] = up        # MUTATION: one 0.25 m column off
        res["mutation_maxdiff"] = float((bp(f0 + bad[None]) - f_ex).abs().max())
        res["mutation_detected"] = res["mutation_maxdiff"] > 1e-3
        res["geometry"] = {"rows": bp.rows, "col0": bp.col0, "cols": bp.cols, "k": bp.k,
                           "fine_window": list(sem.MapExtent(bp.rows * 0.25, bp.cols * 0.25 / 2).window_in(ext))}
        res["geometry_ok"] = res["geometry"]["fine_window"] == [0, 600, 140, 460] and (bp.rows, bp.col0, bp.cols, bp.k) == (240, 56, 128, 2)
    ok = all(res[k] for k in ("identity_feats_equal", "identity_tokens_equal", "shortcut_ok", "pad_equals_index", "mutation_detected",
                              "geometry_ok"))
    print(res)
    print("ZZP4UNITPASSZZ" if ok else "ZZP4UNITFAILZZ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
