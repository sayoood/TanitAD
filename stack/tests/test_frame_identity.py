"""E-DEC-18 — the batch must be able to say WHICH clip and WHICH frame it holds.

PSG (PhyLatent's physical-state grounding) supervises a shared head on both the
encoded and the predicted trajectory using our own banked 3D cuboids. Those
labels are keyed ``(clip_id, frame_idx)``; the windowed batch carried neither,
so a label-conditioned term could not be written at all.

⚠️ WHY THE ALIGNMENT MATTERS AS MUCH AS THE PRESENCE. ``t_last`` must be the LAST
OBSERVED frame of the window -- the frame O7/O8/O9 already target via
``frames[:, -1]``. If it were the window START, a PSG term would be supervised
against a scene up to ``window - 1`` ticks in the past while the external-target
terms use the present, and the two would silently disagree. That off-by-window
error is invisible in a loss curve, so it is pinned here.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "train_flagship4b.py"


def _getitem_body() -> str:
    src = SRC.read_text(encoding="utf-8")
    cls = src.split("class FlagshipWindowDataset", 1)[1].split("\ndef _wrap", 1)[0]
    return cls.split("def __getitem__", 1)[1]


def test_emits_episode_and_frame_identity():
    body = _getitem_body()
    assert 'item["ep_idx"]' in body, "the batch must carry the episode index"
    assert 'item["t_last"]' in body, "the batch must carry the frame index"


def test_t_last_is_the_last_observed_frame_not_the_window_start():
    """⛔ ``t + window - 1``, never ``t`` and never ``t + window``."""
    body = _getitem_body()
    m = re.search(r'item\["t_last"\]\s*=\s*torch\.tensor\(\s*([^,]+),', body)
    assert m, "t_last must be assigned from an explicit expression"
    expr = m.group(1).replace(" ", "")
    assert expr == "t+self.window-1", (
        f"t_last is {expr!r}; it must be t + window - 1 so it names the LAST "
        "OBSERVED frame -- the same frame O7/O8/O9 target via frames[:, -1]")


def test_identity_is_int64_so_default_collate_stacks_it():
    body = _getitem_body()
    for key in ("ep_idx", "t_last"):
        m = re.search(r'item\["%s"\]\s*=\s*torch\.tensor\([^)]*\)' % key, body)
        assert m and "torch.long" in m.group(0), (
            f"{key} must be an int64 tensor: a bare Python int collates to a "
            "list on some paths and to a tensor on others")


def test_pose_prev_still_uses_window_minus_two():
    """The neighbouring index arithmetic is easy to 'tidy' into agreement with
    ``t_last``. ``pose_prev`` is the pose at t-1 RELATIVE TO THE LAST OBSERVED
    FRAME, so it is window - 2 and must stay one behind ``t_last``."""
    body = _getitem_body()
    assert "t + self.window - 2" in body, "pose_prev must remain at window - 2"
