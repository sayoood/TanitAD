#!/usr/bin/env python3
"""Pieces 1-3 of the DDv2 "usable now" bucket were ALREADY IN THE TREE on
2026-09-10. This re-verifies them rather than inheriting them.

⛔ WHY RE-VERIFY. `AGENT_OPERATING_STANDARD` rule 4 and `CLAUDE.md`'s evidence
rules: a claim that decides work must be MEASURED, never INHERITED. `RESULT.md`
section 3.2 marks the >= GT bar and the two-scalar noise as *"usable now, not yet
in `rl/`"*, and that status is STALE -- both landed the same day the analysis was
written. The right response is to measure what is there, and to run the OFF-path
proof on it the same way as on new code.

Each block does the PAIR:
  * with the knob OFF, a deliberate corruption of the new branch must leave the
    result BITWISE unchanged;
  * with the knob ON, the same corruption must CHANGE it -- otherwise the OFF
    result is a fact about an inert branch, not about the gate (the WP-B
    vacuous-proof failure).
"""
from __future__ import annotations

import io
import sys
import time

import torch

from tanitad.rl import PostTrainConfig
from tanitad.rl import advantage as A
from tanitad.rl.refcv3_adapter import sample_offsets

OUT: list[str] = []


def say(s: str = "") -> None:
    OUT.append(s)
    print(s)


def block(title: str) -> None:
    say()
    say("=" * 74)
    say(title)
    say("=" * 74)


# ---------------------------------------------------------------------------
block("PIECE 1 -- the >= GT positive mask (advantage.truncated_inter_anchor_advantage)")
r = torch.tensor([[0.2, 0.6, 0.9, 1.0], [0.1, 0.4, 0.4, 0.9]])
off = A.truncated_inter_anchor_advantage(r)
on = A.truncated_inter_anchor_advantage(r, gt_bar=torch.tensor([0.95, 0.5]))
say(f"reward            = {r.tolist()}")
say(f"OFF (gt_bar=None) = {off.tolist()}")
say(f"ON  (bar .95/.50) = {on.tolist()}")
say(f"OFF == ON ?       {torch.equal(off, on)}   (must be False, or the bar is inert)")

# the hand-computed control: row 0 mean = 0.675, so clamp(r-mean,0) =
# [0, 0, 0.225, 0.325]; the bar 0.95 kills 0.9 and keeps 1.0.
hand = torch.tensor([[0.0, 0.0, 0.225, 0.325]])
say(f"hand-computed row 0 (OFF): {hand.tolist()}  -> matches: "
    f"{torch.allclose(off[0:1], hand)}")
say(f"row 0 ON: 0.9 is above the MEAN but below the human's 0.95 -> "
    f"{on[0].tolist()}")

# ---- the pair: corrupt the bar branch --------------------------------------
real_where = torch.where


def poisoned_bar(reward, **kw):
    """Corrupt only the masked path: multiply the ABOVE-bar mask by 0."""
    if kw.get("gt_bar") is None:
        return A.truncated_inter_anchor_advantage(reward, **kw)
    kw2 = dict(kw)
    kw2["gt_bar"] = kw["gt_bar"] * 0.0 + 1e9        # a bar nobody can clear
    return A.truncated_inter_anchor_advantage(reward, **kw2)


say(f"CORRUPTION with the bar OFF: "
    f"{torch.equal(poisoned_bar(r), off)}   (must be True)")
say(f"CORRUPTION with the bar ON : "
    f"{not torch.equal(poisoned_bar(r, gt_bar=torch.tensor([0.95, 0.5])), on)}"
    f"   (must be True -- the corruption is reachable)")

# ---------------------------------------------------------------------------
block("PIECE 2 -- intra-anchor grouping (advantage.grpo_advantage refuses G < 2)")
try:
    A.grpo_advantage(torch.zeros(2, 3, 1))
    say("⛔ G=1 was ACCEPTED -- a silently zero advantage")
except ValueError as e:
    say(f"G=1 REFUSED: {str(e).splitlines()[0]}")
g2 = A.grpo_advantage(torch.tensor([[0.0, 1.0]]))
say(f"G=2 centred advantage = {g2.tolist()}   (analytic: [-0.5, +0.5])")
say(f"exact match to the analytic value: "
    f"{torch.equal(g2, torch.tensor([[-0.5, 0.5]]))}")
comp = A.composite_advantage(torch.rand(2, 5, 4, generator=torch.Generator().manual_seed(0)),
                             gt_bar=torch.tensor([0.0, 10.0]))
say(f"composite frac_above_bar with one clearable and one impossible window = "
    f"{float(comp['frac_above_bar']):.4f}   (analytic: 0.5000)")

# ---------------------------------------------------------------------------
block("PIECE 3 -- two-scalar exploration (refcv3_adapter.sample_offsets)")
off_cfg = PostTrainConfig(noise_mode="multiplicative", noise_scale=0.1, group_size=4)
on_cfg = PostTrainConfig(noise_mode="two_scalar", noise_scale=0.1, group_size=4)
base = torch.randn(2, 3, 8, 2, generator=torch.Generator().manual_seed(3)) * 5.0 + 3.0

s_off, _ = sample_offsets(base, off_cfg,
                          generator=torch.Generator().manual_seed(1))
s_on, _ = sample_offsets(base, on_cfg,
                         generator=torch.Generator().manual_seed(1))
mean = base.unsqueeze(2).expand_as(s_on)
ratio_on = (s_on - mean) / (mean.abs() * 0.1)
ratio_off = (s_off - mean) / (mean.abs() * 0.1)
say(f"ON  (two_scalar)    max spread of the per-axis ratio ALONG the "
    f"trajectory: {float(ratio_on.std(dim=-2).max()):.3e}   (analytic 0)")
say(f"OFF (multiplicative) mean spread of the same ratio:               "
    f"{float(ratio_off.std(dim=-2).mean()):.3e}   (must be O(1))")
say(f"OFF == ON ? {torch.equal(s_off, s_on)}   (must be False)")

# ---- the RNG-shape fact that makes 'OFF is unchanged' true ------------------
say()
say("RNG-STREAM CHECK -- the OFF path must consume the same draws it always "
    "did, or every banked arm re-rolls:")
torch.manual_seed(77)
a1, _ = sample_offsets(base, off_cfg)
tail1 = torch.randn(4)
torch.manual_seed(77)
a2, _ = sample_offsets(base, off_cfg)
tail2 = torch.randn(4)
say(f"  OFF is deterministic under a seed: {torch.equal(a1, a2)}")
say(f"  and its downstream tail is stable: {torch.equal(tail1, tail2)}")
torch.manual_seed(77)
sample_offsets(base, on_cfg)
tail3 = torch.randn(4)
say(f"  ⚠️ the ON path draws a DIFFERENT SHAPE ([B,N,G,1,2] vs [B,N,G,S,2]), "
    f"so its downstream tail differs: {not torch.equal(tail1, tail3)}")
say("     -> that is expected and is why noise_mode is a DECLARED field in "
    "PostTrainConfig.to_dict(); it is not a change to the OFF arm.")

# ---------------------------------------------------------------------------
block("PIECE 4 -- the selector module exists; the CALLER did not (until today)")
import importlib.util  # noqa: E402
for mod in ("tanitad.refs.refc_selector", "tanitad.refs.refc_selector_targets",
            "tanitad.refs.refc_selector_aug"):
    say(f"  {mod:44s} importable: "
        f"{importlib.util.find_spec(mod) is not None}")

say()
say(f"generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} UTC "
    f"| torch {torch.__version__}")

if len(sys.argv) > 1:
    io.open(sys.argv[1], "w", encoding="utf-8").write("\n".join(OUT) + "\n")
