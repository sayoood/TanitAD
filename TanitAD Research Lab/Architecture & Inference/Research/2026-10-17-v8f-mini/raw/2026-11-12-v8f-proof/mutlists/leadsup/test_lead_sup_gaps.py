"""test_lead_sup_gaps.py -- tests that close the LEADSUP mutations left STILL-GREEN by the first list (m2, m5).  They call the term function `lead_sup_terms` of the tree directly on hand-built
tensors, with expected values worked out by hand in each docstring (no model, no data).  Place in <tree>/stack/tests/ next to test_v8f_lead_sup.py (same import style); CPU."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.models import v8f_mini as M  # noqa: E402

NAN = float("nan")


def _batch(frame_ok, pres, gap, clos):
    t = lambda x: torch.tensor([x], dtype=torch.float32)
    return {"frame_ok": torch.tensor([frame_ok], dtype=torch.bool), "lead_pres": t(pres), "lead_gapm": t(gap), "lead_clos": t(clos)}


def test_m2_a_window_counts_only_when_all_its_history_frames_are_real():
    """Closes LEADSUP m2 (window mask `all` changed to `any`).  hist = 3, F = 5 frames, 3 end frames (windows of frames 0-2, 1-3, 2-4); frame_ok = [T, T, T, F, T]:
    window 0 all real, window 1 has one unreal frame, window 2 has one unreal frame.  So exactly one window counts (n_pres = 1).  Presence labels 1 everywhere, no gap or closing labels,
    head logits for presence [0, 5, -5]: BCE(logit, 1) = softplus(-logit) = ln 2 for window 0 (the counted one), 0.0067 and 5.0067 for the others.
    Expected lead_sup = ln 2 = 0.693147 (independent formula math.log(2)).  With `any` all three windows count: n_pres = 3 and the term is the mean, (0.6931 + 0.0067 + 5.0067) / 3 = 1.9022."""
    hist = 3
    out = torch.zeros(1, 3, 3)
    out[0, :, 0] = torch.tensor([0.0, 5.0, -5.0])
    b = _batch([True, True, True, False, True], [1.0] * 5, [NAN] * 5, [NAN] * 5)
    T = M.lead_sup_terms(out, b, hist)
    assert float(T["lead_sup_n_pres"]) == 1.0, T["lead_sup_n_pres"]
    assert abs(float(T["lead_sup"]) - math.log(2.0)) < 1e-6, float(T["lead_sup"])
    assert abs(float(T["lead_sup_bce"]) - math.log(2.0)) < 1e-6
    assert float(T["lead_sup_n_gap"]) == 0.0 and float(T["lead_sup_n_clos"]) == 0.0


def test_m5_gap_and_closing_parts_are_means_over_their_own_masks():
    """Closes LEADSUP m5 (gap part divided by the presence count).  hist = 2, F = 5, 4 end frames (frames 1..4), all frames real, so the presence count is 4.
    Gap labels on end frames: [10, NaN, 20, NaN] m  (2 labelled), closing labels [5, NaN, -10, 3] m/s (3 labelled); head outputs all 0.
    Huber (delta d, error e > d) = d * (e - d / 2).  Gap: targets 10/50 = 0.2 and 20/50 = 0.4, d = 0.1: 0.1 * (0.2 - 0.05) = 0.015 and 0.1 * (0.4 - 0.05) = 0.035, sum 0.05, part = 0.05 / 2 = 0.025.
    Closing: targets 0.5, -1.0, 0.3, d = 0.2: 0.2 * 0.4 = 0.08, 0.2 * 0.9 = 0.18, 0.2 * 0.2 = 0.04, sum 0.30, part = 0.30 / 3 = 0.1.  Presence: logit 0, BCE ln 2 on all 4, part = ln 2.
    Expected lead_sup = ln 2 + 0.025 + 0.1.  The mutation divides the gap sum by 4 (the presence count): 0.0125."""
    hist = 2
    out = torch.zeros(1, 4, 3)
    b = _batch([True] * 5, [1.0, 0.0, 1.0, 0.0, 1.0], [NAN, 10.0, NAN, 20.0, NAN], [NAN, 5.0, NAN, -10.0, 3.0])
    T = M.lead_sup_terms(out, b, hist)
    assert (float(T["lead_sup_n_pres"]), float(T["lead_sup_n_gap"]), float(T["lead_sup_n_clos"])) == (4.0, 2.0, 3.0)
    assert abs(float(T["lead_sup_gap"]) - 0.025) < 1e-6, float(T["lead_sup_gap"])
    assert abs(float(T["lead_sup_clos"]) - 0.1) < 1e-6, float(T["lead_sup_clos"])
    assert abs(float(T["lead_sup_bce"]) - math.log(2.0)) < 1e-6
    assert abs(float(T["lead_sup"]) - (math.log(2.0) + 0.025 + 0.1)) < 1e-6
