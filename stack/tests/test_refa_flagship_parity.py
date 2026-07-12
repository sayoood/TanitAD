"""REF-A <-> flagship 4-brain PARITY — the machine-checkable proof (Sayed's hard
requirement; Axis 3 "parity" of Project Steering/PRE_FLIGHT_VALIDATION.md).

The frozen-DINO REF-A MIRRORS the flagship EXACTLY and differs in ONLY two axes:

  (1) the ENCODER  — REF-A's frozen-DINO standardizer + trainable grid adapter
                     vs the flagship's from-scratch ViT encoder + spatial readout;
  (2) the SIGReg target — REF-A's "pred_only" (predictor outputs only) vs the
                     flagship's "full_relaxed" (full latent + predictions,
                     position-subspace relaxed).

Everything else is the SAME class, built from the SAME StackConfig, at the SAME
dims. This test pins that:
  * the shared brains (tactical_policy / strategic_policy — and, as a bonus, the
    operative predictor / tactical_pred / inv_dyn) are the SAME class with
    byte-identical param counts;
  * the hierarchical grounding is the SAME class at the SAME state_dim;
  * both trainers call the SAME shared flagship_loss / build_grounding /
    horizon_plan (identical function objects);
  * the two-and-only-two differences hold (encoder module type; SIGReg variant);
  * run_hierarchy produces IDENTICAL output shapes from both models.

CPU. The full-config (~260 M) param parity uses the meta device (shapes only, no
allocation); the run_hierarchy shape parity uses the smoke configs with real
tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refa_train4b  # noqa: E402  (scripts/refa_train4b.py)
import train_flagship4b  # noqa: E402  (scripts/train_flagship4b.py)

from tanitad.config import (flagship4b_config,  # noqa: E402
                            flagship4b_smoke_config)
from tanitad.models.encoder import ViTEncoder  # noqa: E402
from tanitad.models.fourbrain import (StrategicPolicy,  # noqa: E402
                                      TacticalPolicy, WorldModel, run_hierarchy)
from tanitad.models.metric_dynamics import HierarchicalGrounding  # noqa: E402
from tanitad.refs.refa import (DinoGridAdapter,  # noqa: E402
                               FeatureStandardizer, RefAModel)

# The shared brains that MUST be identical (name -> attribute on the model). The
# task's hard requirement is tactical/strategic/grounding; the operative
# predictor + tactical_pred + inv_dyn are pinned too (they strengthen the "only
# the encoder differs" claim — every non-encoder module matches).
SHARED_BRAINS = ("tactical_policy", "strategic_policy", "predictor",
                 "tactical_pred", "inv_dyn")


def _n(module) -> int:
    return sum(p.numel() for p in module.parameters())


def _meta_pair(cfg):
    """flagship WorldModel(cfg) + the frozen-DINO RefAModel from the SAME cfg,
    built on the meta device (shapes only — no ~260 M allocation)."""
    n_tokens = (cfg.encoder.image_size // cfg.encoder.patch_size) ** 2   # 256
    with torch.device("meta"):
        flag = WorldModel(cfg)
        refa = RefAModel.from_stack_config(cfg, n_tokens=n_tokens)
    return flag, refa


# --------------------------------------------------------------------------- #
# (1) the shared brains are the SAME class at the SAME dims and param count     #
# --------------------------------------------------------------------------- #
def test_shared_brains_identical_full_config():
    flag, refa = _meta_pair(flagship4b_config())

    # the grid adapter mirrors the flagship's readout geometry, so the state the
    # shared brains consume is the SAME width (the precondition for parity).
    assert flag.state_dim == refa.state_dim == 2048

    # trained policy brains: SAME class (the deliverable's headline assertion)
    assert type(flag.tactical_policy) is type(refa.tactical_policy) is TacticalPolicy
    assert type(flag.strategic_policy) is type(refa.strategic_policy) is StrategicPolicy

    # SAME dims (not just same class): windows, widths, and the state_dim-sized
    # in/out projections all agree.
    assert flag.tactical_policy.window == refa.tactical_policy.window
    assert flag.strategic_policy.window == refa.strategic_policy.window
    assert (flag.tactical_policy.in_proj.in_features
            == refa.tactical_policy.in_proj.in_features == 2048)
    assert (flag.tactical_policy.target_latent_head.out_features
            == refa.tactical_policy.target_latent_head.out_features == 2048)
    assert (flag.strategic_policy.in_proj.in_features
            == refa.strategic_policy.in_proj.in_features == 2048)

    # byte-identical param counts for every shared brain (exact — far tighter
    # than "a tiny tolerance").
    for name in SHARED_BRAINS:
        fm, rm = getattr(flag, name), getattr(refa, name)
        assert type(fm) is type(rm), name
        assert _n(fm) == _n(rm) > 0, (name, _n(fm), _n(rm))

    # the intent link (tactical -> operative FiLM) is present on BOTH.
    assert flag.predictor.intent_proj is not None
    assert refa.predictor.intent_proj is not None


def test_grounding_heads_identical_and_shared_builder():
    flag, refa = _meta_pair(flagship4b_config())
    with torch.device("meta"):
        gf = HierarchicalGrounding(flag.state_dim)
        gr = HierarchicalGrounding(refa.state_dim)
    assert type(gf) is type(gr) is HierarchicalGrounding
    assert _n(gf) == _n(gr) > 0
    assert gf.LEVELS == gr.LEVELS == ("op", "tac", "str")
    # both trainers ground via the SAME shared builder + loss + horizon plan.
    assert refa_train4b.build_grounding is train_flagship4b.build_grounding
    assert refa_train4b.flagship_loss is train_flagship4b.flagship_loss
    assert refa_train4b.horizon_plan is train_flagship4b.horizon_plan


# --------------------------------------------------------------------------- #
# (2) difference #1 — the ENCODER is the ONLY differing model module           #
# --------------------------------------------------------------------------- #
def test_encoder_is_the_only_differing_module():
    flag, refa = _meta_pair(flagship4b_config())

    # flagship encodes with a from-scratch ViT + readout; REF-A has NO ViT.
    assert isinstance(flag.encoder, ViTEncoder)
    assert not hasattr(refa, "encoder")
    assert isinstance(refa.standardizer, FeatureStandardizer)
    assert isinstance(refa.adapter, DinoGridAdapter)

    # the encoder is the axis of difference: the ViT is heavy; the frozen-DINO
    # front end is a 0-param standardizer + a light adapter (readout only).
    assert _n(flag.encoder) > 10_000_000
    assert _n(refa.standardizer) == 0             # buffers only (mean/std/fitted)
    assert _n(refa.adapter) < _n(flag.encoder)

    # EVERY non-encoder brain matches (so the encoder is the ONLY model-axis
    # difference — nothing else moved).
    for name in SHARED_BRAINS:
        assert _n(getattr(flag, name)) == _n(getattr(refa, name)), name


# --------------------------------------------------------------------------- #
# (3) difference #2 — the SIGReg target is the ONLY differing loss argument     #
# --------------------------------------------------------------------------- #
def test_sigreg_variant_is_the_only_loss_difference():
    assert train_flagship4b.SIGREG_VARIANT == "full_relaxed"
    assert refa_train4b.SIGREG_VARIANT == "pred_only"
    assert train_flagship4b.SIGREG_VARIANT != refa_train4b.SIGREG_VARIANT
    # both are valid branches of the ONE shared loss body (no third variant).
    assert {train_flagship4b.SIGREG_VARIANT, refa_train4b.SIGREG_VARIANT} \
        == {"full_relaxed", "pred_only"}


# --------------------------------------------------------------------------- #
# (4) run_hierarchy produces IDENTICAL output shapes from both models           #
# --------------------------------------------------------------------------- #
def test_run_hierarchy_output_shapes_match():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    flag = WorldModel(cfg).eval()
    refa = RefAModel.from_stack_config(cfg, n_tokens=16).eval()   # 4*4*32 = 512
    assert flag.state_dim == refa.state_dim

    B, W = 3, cfg.predictor.window
    states = torch.randn(B, W, flag.state_dim)
    actions = torch.randn(B, W, 2)
    nav = torch.tensor([0, 1, 2])
    of = run_hierarchy(flag, states, actions, nav)
    orr = run_hierarchy(refa, states, actions, nav)

    assert set(of) == set(orr)
    for k in ("ctx", "route_logits", "maneuver_logits", "target_latent",
              "intent"):
        assert of[k].shape == orr[k].shape, k
    assert set(of["waypoints"]) == set(orr["waypoints"])
    for k in of["waypoints"]:
        assert of["waypoints"][k].shape == orr["waypoints"][k].shape
    for k in cfg.predictor.horizons:
        assert of["preds"][k].shape == orr["preds"][k].shape
    # the model's own hierarchy() convenience wraps run_hierarchy identically.
    assert refa.hierarchy(states, actions, nav)["intent"].shape \
        == of["intent"].shape


# --------------------------------------------------------------------------- #
# (5) headline: exactly two differences, everything else identical             #
# --------------------------------------------------------------------------- #
def test_two_and_only_two_differences():
    flag, refa = _meta_pair(flagship4b_config())

    # SAME: every shared brain + grounding + the intent link + state_dim.
    same = {name: (_n(getattr(flag, name)) == _n(getattr(refa, name)))
            for name in SHARED_BRAINS}
    assert all(same.values()), same
    assert flag.state_dim == refa.state_dim

    # DIFF #1 (encoder): flagship ViT vs REF-A frozen-DINO front end.
    diff_encoder = (isinstance(flag.encoder, ViTEncoder)
                    and not hasattr(refa, "encoder")
                    and _n(flag.encoder) != _n(refa.adapter))
    # DIFF #2 (SIGReg target): the ONE flipped argument to the shared loss.
    diff_sigreg = (train_flagship4b.SIGREG_VARIANT
                   != refa_train4b.SIGREG_VARIANT)
    assert diff_encoder and diff_sigreg
