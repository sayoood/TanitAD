"""`box3d_ap` split into `box3d_match_rows` + `ap_from_rows` (2026-09-19, PREREG_S1 S1A.4).

The split exists so the S1 box read can take the AP's OWN matched pairs (for a velocity error)
and pool rows over RESAMPLED episodes (for an episode-cluster bootstrap), rather than
re-deriving a second matcher. The reference below is a FROZEN LITERAL COPY of the historical
`box3d_ap` body, so "unchanged" is checked against an independent source, not against the
refactor re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tanitad.models.box3d_head import ap_from_rows, box3d_ap, box3d_match_rows  # noqa: E402


def _historical_box3d_ap(pred, tgt, *, dist_thresh_m=2.0, use_z=True, score="presence"):
    """Verbatim body of box3d_ap as it stood before the split (HEAD 362ce88)."""
    key = "box3d" if use_z else "box"
    B = int(pred[key].shape[0])
    rows = []
    n_gt = 0
    with torch.no_grad():
        conf_all = (pred["presence_logit"].sigmoid() if score == "presence"
                    else pred["cls_logits"].softmax(-1).max(-1).values)
        for b in range(B):
            valid = tgt["valid"][b].nonzero(as_tuple=False).flatten()
            n_gt += int(valid.numel())
            if use_z:
                pc = pred["box3d"][b][:, :3]
                tc = torch.stack([tgt["box"][b][valid][:, 0],
                                  tgt["box"][b][valid][:, 1],
                                  tgt["cz"][b][valid]], dim=-1)
            else:
                pc = pred["box"][b][:, :2]
                tc = tgt["box"][b][valid][:, :2]
            conf = conf_all[b]
            order = torch.argsort(conf, descending=True)
            taken = torch.zeros(valid.numel(), dtype=torch.bool)
            for i in order.tolist():
                if valid.numel() == 0:
                    rows.append((float(conf[i]), 0))
                    continue
                d = (tc - pc[i][None, :]).norm(dim=-1)
                d = torch.where(taken.to(d.device), torch.full_like(d, float("inf")), d)
                j = int(torch.argmin(d))
                hit = float(d[j]) <= float(dist_thresh_m)
                if hit:
                    taken[j] = True
                rows.append((float(conf[i]), 1 if hit else 0))
    if n_gt == 0:
        return {"ap": float("nan"), "n_gt": 0, "n_pred": len(rows),
                "precision": [], "recall": []}
    rows.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in rows], dtype=np.float64)
    fp = np.cumsum([1 - r[1] for r in rows], dtype=np.float64)
    rec = tp / float(n_gt)
    prec = tp / np.maximum(tp + fp, 1e-12)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    return {"ap": ap, "n_gt": n_gt, "n_pred": len(rows),
            "precision": prec.tolist(), "recall": rec.tolist()}


def _rand(seed, B=6, N=12, M=9, tie=False):
    g = torch.Generator().manual_seed(seed)
    logit = torch.randn(B, N, generator=g)
    if tie:                                              # force equal confidences
        logit = torch.round(logit)
    pred = {"presence_logit": logit, "cls_logits": torch.randn(B, N, 4, generator=g),
            "box3d": torch.randn(B, N, 3, generator=g) * 4,
            "box": torch.randn(B, N, 4, generator=g) * 4}
    tgt = {"box": torch.randn(B, M, 4, generator=g) * 4, "cz": torch.randn(B, M, generator=g),
           "valid": torch.rand(B, M, generator=g) > 0.4}
    tgt["valid"][0] = False                              # an element with NO target
    return pred, tgt


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("use_z", [True, False])
def test_box3d_ap_is_UNCHANGED_by_the_split(seed, use_z):
    pred, tgt = _rand(seed, tie=seed % 2 == 0)
    new = box3d_ap(pred, tgt, dist_thresh_m=3.0, use_z=use_z)
    old = _historical_box3d_ap(pred, tgt, dist_thresh_m=3.0, use_z=use_z)
    assert new == old


def test_match_rows_carry_a_VALID_one_to_one_matching():
    pred, tgt = _rand(3)
    per, ngt = box3d_match_rows(pred, tgt, dist_thresh_m=3.0)
    for b, rows in enumerate(per):
        hits = [r for r in rows if r[1] == 1]
        gts = [r[3] for r in hits]
        assert len(gts) == len(set(gts)), "a target matched twice"
        assert all(bool(tgt["valid"][b, j]) for j in gts), "matched an invalid slot"
        for conf, hit, i, j in hits:
            tc = torch.stack([tgt["box"][b, j, 0], tgt["box"][b, j, 1], tgt["cz"][b, j]])
            assert float((pred["box3d"][b, i] - tc).norm()) <= 3.0
        assert all(r[3] == -1 for r in rows if r[1] == 0)
        assert len(hits) <= ngt[b]


def test_ap_from_rows_reads_KNOWN_values():
    assert ap_from_rows([(0.9, 1), (0.8, 1)], 2)["ap"] == 1.0          # all hits first
    assert ap_from_rows([(0.9, 0), (0.8, 1)], 1)["ap"] == 0.5          # one FP ahead
    assert np.isnan(ap_from_rows([(0.5, 0)], 0)["ap"])
