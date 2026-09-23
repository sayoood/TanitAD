"""refcv6 §2/§6 -- THE PERCEPTION BRANCH, as ONE module a trainer can own.

The PI's instruction (2026-09-16) is to *"train jointly the resnet-trunk, the bev
map (based on the sam2 maps as gt) and a head for 3d bounding boxes extracted
from the resnet trunk"*. Every piece of that existed and **nothing assembled
them**: MEASURED on tip ``c16b7f1``, ``bev_encoder``, ``bev_lift`` and
``box3d_head`` are imported by ``stack/tests/`` and **by no production module at
all** -- ``BEVLift``, ``BEVMapBranch``, ``Box3DMemory`` and ``Box3DSlotDecoder``
are never constructed outside a test. This module is the missing assembly.

WHAT IT IS, AND WHAT IT DELIBERATELY IS NOT
-------------------------------------------
It is a ``nn.Module`` the **trainer** builds and attaches to the model
(``model._perception``), reading ``out["fmap_s16"]`` -- the stride-16 map
``RefCModel.forward`` already returns (``refc.py:4215``) for the LAST OBSERVED
frame of the window, with its graph attached.

⛔ It is NOT an edit to ``refc.py`` / ``refc_v3.py``. Two reasons, and the second
is the load-bearing one:

1. those files are another stream's, and the seam they publish is already
   sufficient -- ``fmap_s16`` needs no new plumbing;
2. **a model that does not build the branch is bit-identical to the tip.** A
   config field would have to be defaulted, stamped, and read on every arm; an
   attribute the trainer attaches only when a weight is positive cannot change
   a default run's parameter set, its ``state_dict``, or its RNG draw order.

WHERE THE SHAPES COME FROM
--------------------------
⛔ Nothing here hard-codes 640 / 1024 / 160 / 40 / 20 or a channel count.

* ``d_image`` = ``encoder.s16_dim``   -- timm's ``feature_info.channels()``;
* ``image_hw`` = ``encoder.s16_shape`` -- derived from the payload's own H x W;
* the BEV grid = :data:`bev_raster.GRID_DEFAULT`, asserted against the SAM3
  label grid by :func:`trunk_shapes.assert_label_grid_unmoved`;
* the camera frame = :func:`trunk_shapes.frame_for_width` on the cache's width.

THE THREE GRADIENT PATHS, NAMED
-------------------------------
``loss_map``   -> MapHead -> BEVEncoder -> BEVLift -> **trunk** (via ``fmap_s16``)
``loss_box3d`` -> Box3DSlotDecoder -> Box3DMemory -> **trunk** (image tokens),
                  and additionally through the BEV features when ``use_bev``.

Both therefore reach the ResNet trunk, which is the PI's "train jointly".
:func:`grad_reach_report` is the instrument that PROVES it per head rather than
asserting it -- the ``tac_goal_tok_head`` precedent (11,286 parameters with
``grad_abs_sum`` exactly 0 for all 40,284 steps) is the failure it exists for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from tanitad.channel_admissibility import ChannelExclusion
from tanitad.data.bev_raster import GRID_DEFAULT, BEVGrid
from tanitad.data.semantic_map_gt import CART_SHAPE, N_CHANNELS
from tanitad.models.bev_encoder import (BEVEncoderConfig, BEVMapBranch,
                                        map_metrics, map_soft_ce)
from tanitad.models.bev_lift import HEIGHTS_M, BEVLift, build_lift_geometry
from tanitad.models.box3d_head import (Box3DMemory, Box3DSlotDecoder,
                                       box3d_set_loss)
from tanitad.models.trunk_shapes import (PERCEPTION_STRIDE,
                                         assert_label_grid_unmoved,
                                         frame_for_width)

__all__ = ["PerceptionBranchConfig", "LiftGeometryBank", "PerceptionBranch",
           "build_perception_branch", "grad_reach_report", "FORWARD_EXCLUSIONS",
           "map_valid_from_lift", "map_loss_row", "box3d_loss_row"]


# --------------------------------------------------------------------------- #
# the RL channel contract — declared HERE because this seam owns the channel   #
# --------------------------------------------------------------------------- #
#: ⛔⛔ PI RULING 2026-09-17 R2 added ``perception_grid`` / ``perception_valid`` to
#: ``RefCV3Model.forward`` (the BEV encoder moved INTO the forward, so this batch's
#: per-clip lift geometry must arrive with it). ``tests/
#: test_rl_forward_keys_cover_signature.py`` DERIVES the RL adapter's requirement from
#: the live signature and went RED on exactly these two — correctly: a channel absent
#: from ``refc_adapter.FORWARD_KEYS`` is **never passed at all**, with no error.
#:
#: ⭐ THEY ARE NOT LABELS. Unlike ``gp_point`` (the goal IS the label) or ``v_max_ms``
#: (read off the ego's realised future), this geometry is RIG CALIBRATION: a mount
#: pose per clip, back-projected through the road plane. Nothing in it comes from the
#: future, so the channel is admissible in principle and the exclusion is TEMPORARY.
#:
#: ⚠️ AND THE ABSENCE CANNOT SILENTLY CORRUPT ANYTHING, which is what makes excluding
#: it safe rather than merely convenient: a model driven through the RL adapter has no
#: perception branch attached, so ``RefCV3Model._bev_hook`` returns ``None`` and the
#: channel is never read; and if one WERE attached, that same hook REFUSES rather than
#: falling back to a nominal camera. There is no third state in which a missing
#: geometry is quietly replaced by a default.
FORWARD_EXCLUSIONS = tuple(
    ChannelExclusion(
        channel=_ch,
        owner="tanitad.models.refcv6_perception_branch (refcv6 §2/§6 BEV lift)",
        reason=(
            "An RL rollout would have to get this from a LiftGeometryBank built "
            "over a PER-CLIP extrinsics table, and the RL adapter plumbs none: "
            "`tanitad.rl.refc_adapter` never constructs a bank and a rollout "
            "carries no clip->mount-pose map. ⛔ The channel is NOT a label — it "
            "is rig calibration and nothing in it comes from the ego's future — "
            "so this is an unplumbed supplier, not an inadmissible one. Feeding "
            "a DEFAULT camera instead is the failure this refuses: MEASURED, the "
            "corpus's mount height spans 1.2131-1.6672 m over 554 distinct "
            "values in 2,400 clips, and the lift back-projects through the road "
            "plane, so one pose for the corpus biases every BEV cell it fills "
            "while every count still looks healthy. ⚠️ Withholding it is safe "
            "because a model driven through this adapter has no `_perception` "
            "attached, so `RefCV3Model._bev_hook` returns None and the channel "
            "is never read — and with a branch attached the same hook RAISES "
            "rather than defaulting."),
        unblock=(
            "Plumb a LiftGeometryBank into the RL rollout: build it from the "
            "same per-clip extrinsics table the trainer uses "
            "(`--agent-rig-extrinsics`), call `for_episodes(batch_episode_ids)` "
            "and pass the (grid, valid) pair through `forward_kwargs`. Then add "
            "both channels to `refc_adapter.FORWARD_KEYS` in signature order and "
            "delete this declaration. No change to this module is needed — the "
            "bank and the refusal are already correct."),
        evidence=(
            "MEASURED 2026-09-17: `Research/2026-09-17-refcv6-bev-tactical/raw/"
            "grad_reach_bev.json` (the branch runs in-forward, four arms) and "
            "the live run's own refusal when a positional episode id was passed "
            "instead of a stable one. PUBLISHED-CODE: `LiftGeometryBank.geometry` "
            "raises SystemExit on a missing clip; `RefCV3Model._bev_hook` raises "
            "when the lift is built and no geometry arrives; "
            "`tanitad.rl.refc_adapter.FORWARD_KEYS` contains no bank."),
        permanent=False,
        refs=("TanitAD Research Lab/Architecture & Inference/Research/"
              "2026-09-17-refcv6-bev-tactical/",),
    )
    for _ch in ("perception_grid", "perception_valid"))


# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PerceptionBranchConfig:
    """What the trainer decides; every SHAPE comes from the payload instead.

    ``w_map`` / ``w_box3d`` are the loss weights. ⛔ **Both 0.0 is not a legal
    config for this object** -- a branch with no live weight must not be BUILT,
    because its parameters would enter ``model.parameters()``, change the
    optimiser's state and the checkpoint, and train on nothing. The trainer
    refuses before constructing; :meth:`__post_init__` refuses again here, so
    the invariant does not depend on one call site remembering it.
    """

    w_map: float
    w_box3d: float
    d_bev: int = 128                  # BEVLift.d_out == BEVEncoderConfig.d_in
    bev_cfg: BEVEncoderConfig = field(default_factory=BEVEncoderConfig)
    n_queries: int = 100              # agent_slots.N_QUERIES_DEFAULT
    d_model: int = 256
    bev_tokens_hw: tuple[int, int] = (30, 16)
    heights_m: tuple[float, ...] = HEIGHTS_M
    stride: int = PERCEPTION_STRIDE
    #: ⛔ SHAPE TESTS ONLY. :class:`Box3DSlotDecoder` enforces §6's
    #: pre-registered 2-4 M parameter band at construction, for the reason its
    #: own docstring gives: a bigger head stops measuring what the LATENT
    #: carries. A tiny test rig is BELOW the band, so a test must be able to
    #: say so BY NAME rather than by silently widening the band -- and no argv
    #: reaches this field.
    enforce_param_band: bool = True
    #: ⛔ derive `occ_logit` from the head's OWN predicted centre instead of
    #: the learned slice (`agent_slots.occ_logit_from_centre`). MEASURED
    #: 2026-09-21 (`6c5fb62`): the learned channel LOSES to a 2-parameter read
    #: of that centre's azimuth by 0.1276 log-loss, CI [-0.1844, -0.0376], and
    #: never wins in any box-error stratum.
    #: ⚠️ DEFAULT OFF -- flipping it changes what the model EMITS at
    #: inference, which is a contract change and the PI's call.
    #: ⛔ STAMPED in `as_dict`: this branch's stamp is what a reader opening
    #: `config.json` in isolation gets, and an `occ` number whose source is not
    #: in the record is unattributable between the learned and derived paths.
    occ_from_geometry: bool = False
    #: ⛔ NOT a knob the operator sets. It is DERIVED: the box head reads BEV
    #: tokens only when a supervised BEV branch exists, i.e. when ``w_map > 0``.
    #: Building the lift + encoder for an unsupervised feature path would put
    #: ~1 M parameters in the optimiser whose only gradient is the box loss's --
    #: legal, but a different experiment, and it must be asked for by name.
    def __post_init__(self) -> None:
        if float(self.w_map) < 0.0 or float(self.w_box3d) < 0.0:
            raise ValueError("perception weights must be >= 0")
        if float(self.w_map) == 0.0 and float(self.w_box3d) == 0.0:
            raise ValueError(
                "PerceptionBranchConfig with BOTH weights 0.0: the branch "
                "would add parameters to the optimiser and the checkpoint "
                "while training on nothing. Do not build it.")
        if int(self.d_bev) != int(self.bev_cfg.d_in):
            raise ValueError(
                f"d_bev {self.d_bev} != bev_cfg.d_in {self.bev_cfg.d_in}: the "
                f"lift's output width IS the BEV encoder's input width")

    @property
    def use_bev(self) -> bool:
        return float(self.w_map) > 0.0

    def as_dict(self) -> dict:
        return {"w_map": float(self.w_map), "w_box3d": float(self.w_box3d),
                "d_bev": int(self.d_bev), "n_queries": int(self.n_queries),
                "d_model": int(self.d_model),
                "bev_tokens_hw": list(self.bev_tokens_hw),
                "heights_m": list(self.heights_m), "stride": int(self.stride),
                "use_bev_in_box_head": bool(self.use_bev),
                "occ_from_geometry": bool(self.occ_from_geometry),
                "bev_encoder": {"d_in": int(self.bev_cfg.d_in),
                                "d_model": int(self.bev_cfg.d_model),
                                "d_out": int(self.bev_cfg.d_out),
                                "dilations": list(self.bev_cfg.dilations)}}


# --------------------------------------------------------------------------- #
# per-clip lift geometry                                                       #
# --------------------------------------------------------------------------- #
class LiftGeometryBank:
    """``episode_id -> (grid, valid)`` for :class:`BEVLift`, built ONCE per clip.

    ⛔ **Per clip, never per run.** The same argument
    :class:`refc_agents.RigCameraBank` carries: MEASURED on the parity corpus,
    mount height spans 1.2131-1.6672 m over **554 distinct values in 2,400
    clips**, and the lift back-projects through the road plane, so one mount
    pose applied to every clip biases the geometry of every cell it fills.

    ⛔ A missing episode REFUSES. Falling back to a default camera is how the
    map head would train against the wrong ground plane while every count looked
    healthy -- the same refusal ``_resolve_rig_cameras`` makes one seam over.
    """

    def __init__(self, extr_by_clip: dict, *, frame, stride: int,
                 heights_m=HEIGHTS_M, grid: BEVGrid = GRID_DEFAULT,
                 equalize_bottom_rows: int = 0):
        from tanitad.data.v2_dataset import stable_episode_id
        self.frame = frame
        self.stride = int(stride)
        self.heights_m = tuple(float(h) for h in heights_m)
        self.grid_spec = grid
        # ⛔ C26: rows the TRUNK zeroes must be rows the LIFT treats as UNOBSERVED.
        # The lift's `observed` mask existed and was never passed (2026-09-22 review);
        # without it a zeroed row reads as observed black road.
        self.equalize_bottom_rows = int(equalize_bottom_rows or 0)
        self._observed = None
        if self.equalize_bottom_rows > 0:
            _o = torch.ones(int(frame.height), int(frame.width), dtype=torch.bool)
            _o[-self.equalize_bottom_rows:, :] = False
            self._observed = _o
        self._extr: dict[int, object] = {}
        self.clip_of: dict[int, str] = {}
        for cid, e in extr_by_clip.items():
            sid = int(stable_episode_id(str(cid)))
            if sid in self._extr:
                raise SystemExit(
                    f"[perception] two clip_ids collide on stable_episode_id "
                    f"{sid}; refusing rather than attaching one clip's mount "
                    f"pose to another's BEV cells")
            self._extr[sid] = e
            self.clip_of[sid] = str(cid)
        self._cache: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self._extr)

    def covers(self, ep_id: int) -> bool:
        return int(ep_id) in self._extr

    def geometry(self, ep_id: int) -> tuple:
        """``(grid [Z,X,Y,2] float32, valid [Z,X,Y] bool)`` for one episode."""
        k = int(ep_id)
        hit = self._cache.get(k)
        if hit is not None:
            return hit
        e = self._extr.get(k)
        if e is None:
            raise SystemExit(
                f"[perception] no rig extrinsics for episode {k} -- the BEV "
                f"lift has no camera for this clip and a default would put the "
                f"road plane in the wrong place. Pass an extrinsics table that "
                f"covers every clip in the cache.")
        g = build_lift_geometry(e, frame=self.frame, stride=self.stride,
                                heights_m=self.heights_m, grid=self.grid_spec,
                                observed=self._observed)
        hit = (g.grid, g.valid)
        self._cache[k] = hit
        return hit

    def for_episodes(self, ep_ids, device=None) -> tuple:
        """Stack one row per episode id -> ``([B,Z,X,Y,2], [B,Z,X,Y])``."""
        ids = [int(x) for x in (ep_ids.tolist() if torch.is_tensor(ep_ids)
                                else list(ep_ids))]
        if not ids:
            raise ValueError("for_episodes: empty batch")
        gs, vs = zip(*(self.geometry(i) for i in ids))
        grid = torch.stack(gs).to(device) if device is not None else torch.stack(gs)
        valid = torch.stack(vs).to(device) if device is not None else torch.stack(vs)
        return grid, valid

    def coverage(self, ep_ids) -> dict:
        ids = [int(x) for x in ep_ids]
        n_ok = sum(1 for i in ids if i in self._extr)
        return {"n": len(ids), "n_covered": n_ok,
                "n_missing": len(ids) - n_ok,
                "frac": (n_ok / len(ids)) if ids else float("nan"),
                "bank_n_clips": len(self._extr)}


# --------------------------------------------------------------------------- #
# the branch                                                                   #
# --------------------------------------------------------------------------- #
class PerceptionBranch(nn.Module):
    """``fmap_s16`` -> map logits and/or 3-D slots. One forward, one graph."""

    def __init__(self, cfg: PerceptionBranchConfig, *, d_image: int,
                 image_hw: tuple[int, int], n_classes: int = N_CHANNELS):
        super().__init__()
        self.cfg = cfg
        self.d_image = int(d_image)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        # ⛔ The SAM3 label grid is asserted, not assumed: if it ever moves, the
        # lift geometry and the loss disagree silently on which cell is which.
        self.map_grid_hw = assert_label_grid_unmoved(CART_SHAPE)

        self.lift: BEVLift | None = None
        self.map_branch: BEVMapBranch | None = None
        if cfg.use_bev:
            self.lift = BEVLift(d_in=self.d_image, d_out=int(cfg.d_bev),
                                n_heights=len(cfg.heights_m),
                                feat_hw=self.image_hw)
            self.map_branch = BEVMapBranch(cfg.bev_cfg, n_classes=n_classes)

        self.box_mem: Box3DMemory | None = None
        self.box_dec: Box3DSlotDecoder | None = None
        if float(cfg.w_box3d) > 0.0:
            self.box_mem = Box3DMemory(
                d_image=self.d_image, d_bev=int(cfg.bev_cfg.d_out),
                d_model=int(cfg.d_model), image_hw=self.image_hw,
                bev_tokens_hw=tuple(cfg.bev_tokens_hw),
                use_bev=bool(cfg.use_bev), perception_stride=int(cfg.stride))
            self.box_dec = Box3DSlotDecoder(
                d_memory=int(cfg.d_model), n_memory=int(self.box_mem.n_tokens),
                n_queries=int(cfg.n_queries), d_model=int(cfg.d_model),
                enforce_band=bool(cfg.enforce_param_band))
            # ⛔ DECLARED IS NOT PLUMBED. A stamped field that never reaches
            # the module it names reads as "the knob does nothing" rather than
            # as a bug -- the refcv6 seam defect verbatim. The test pins THIS
            # line, not the dataclass.
            self.box_dec.occ_from_geometry = bool(cfg.occ_from_geometry)

    # -- refcv6 §4, PI RULING 2026-09-17 R2: BEV TOKENS FOR THE DECODER ----- #
    @property
    def bev_token_dim(self) -> int:
        """Width of one BEV token == :class:`BEVEncoderConfig`.``d_out``.

        ⛔ **DERIVED, never an operator knob.** ``--tac-decoder-d-bev`` must
        equal this or the decoder's ``Linear(d_bev, d_model)`` would be sized
        for a tensor that never arrives -- the units error of C-ANCHOR-UNITS in
        a channel-count costume. The trainer reads THIS property rather than
        re-deriving the number, so there is exactly one spelling of it.
        """
        return int(self.cfg.bev_cfg.d_out)

    @property
    def n_bev_tokens(self) -> int:
        h, w = self.cfg.bev_tokens_hw
        return int(h) * int(w)

    def bev_tokens(self, bev_feats: Tensor) -> Tensor:
        """``[B, d_out, X, Y]`` -> ``[B, P, d_out]``, a FLAT token sequence.

        ⚠️ Pooled to :attr:`PerceptionBranchConfig.bev_tokens_hw` -- the SAME
        pool :class:`Box3DMemory` already applies, and for the same reason its
        docstring gives: nothing downstream of this pool is compared to a
        per-cell label, and a 7,680-key cross-attention would make the decoder
        the experiment rather than the map. ⛔ Deliberately the same field, not
        a second one: two pools that can drift apart is how the box head and
        the tactical decoder would silently read different maps.

        ⭐ **ZERO PARAMETERS.** ``adaptive_avg_pool2d`` + a reshape. That is
        what keeps the bit-identity argument intact: turning the tactical BEV
        path on adds no tensor to ``state_dict`` beyond the decoder's own
        ``bev_in``, which the decoder's ``d_bev`` already accounts for.

        ⛔ **THE FLATTEN IS NOT REIMPLEMENTED HERE.**
        :func:`refcv6_tactical.bev_feats_to_tokens` is *"the ONLY place a BEV
        grid is flattened"* by its own docstring, and it derives the token count
        from the tensor so no grid size is ever typed. A second spelling of a
        transpose is how two call sites end up disagreeing about which axis is
        X — silently, with both shapes valid.
        """
        from tanitad.refs.refcv6_tactical import bev_feats_to_tokens
        if bev_feats.dim() != 4:
            raise ValueError(
                f"[perception] bev_feats must be [B, C, X, Y], got "
                f"{tuple(bev_feats.shape)}")
        if bev_feats.shape[1] != self.bev_token_dim:
            raise ValueError(
                f"[perception] bev_feats are {bev_feats.shape[1]} wide, the "
                f"branch's BEV encoder emits {self.bev_token_dim} "
                f"(BEVEncoderConfig.d_out)")
        t = torch.nn.functional.adaptive_avg_pool2d(
            bev_feats, tuple(int(x) for x in self.cfg.bev_tokens_hw))
        return bev_feats_to_tokens(t)                    # [B, P, d_out]

    # -- reporting ---------------------------------------------------------- #
    def param_breakdown(self) -> dict:
        def n(m):
            return 0 if m is None else int(sum(p.numel() for p in m.parameters()))
        return {"lift": n(self.lift), "bev_encoder": n(
                    None if self.map_branch is None else self.map_branch.encoder),
                "map_head": n(None if self.map_branch is None
                              else self.map_branch.head),
                "box_memory": n(self.box_mem), "box_decoder": n(self.box_dec),
                "total": n(self)}

    def forward(self, fmap_s16: Tensor, grid: Tensor | None = None,
                valid: Tensor | None = None) -> dict:
        """⛔ ONE argument family, all vision. No parameter here can carry a
        label -- the ``AgentSlotDecoder.forward`` audit, extended to the branch.
        """
        if fmap_s16 is None:
            raise ValueError(
                "[perception] fmap_s16 is None: this build's trunk exposes no "
                "stride-16 map. refcv6 perception reads stride 16 (an oracle "
                "on the stride-32 map caps at AP 0.3341 vs 0.4713, "
                "SPEC_REFCV6_V2 section 2) -- use --trunk timm.")
        if fmap_s16.shape[1] != self.d_image:
            raise ValueError(
                f"[perception] fmap_s16 has {fmap_s16.shape[1]} channels, the "
                f"branch was built for {self.d_image} (read from timm's "
                f"feature_info at build time)")
        out: dict = {}
        bev_feats = None
        if self.lift is not None:
            if grid is None or valid is None:
                raise ValueError(
                    "[perception] the BEV branch is built but no lift geometry "
                    "was passed: the map would be predicted from features "
                    "sampled at no camera at all")
            bev = self.lift(fmap_s16, grid, valid)
            mb = self.map_branch(bev)
            bev_feats = mb["bev_feats"]
            out["bev"] = bev
            out["bev_feats"] = bev_feats
            out["map_logits"] = mb["map_logits"]
            # ⭐ THE FIX'S INPUT, EMITTED RATHER THAN RE-DERIVED. The map loss needs
            # "which cells did the camera reach at THIS instant", and the lift already
            # knows -- it is the same `valid.any(dim=1)` it uses to decide where to
            # substitute its `unobserved` embedding. Emitting it costs one reduction and
            # no parameter; re-deriving it at the loss site would be a second spelling of
            # a projection test, which is how two masks drift apart.
            # ⚠️ ADDITIVE KEY ONLY: nothing here changes what an existing arm computes.
            # The behaviour change is at the LOSS, where the caller chooses to pass it.
            out["map_valid"] = map_valid_from_lift(valid)
            # ⭐⭐ PI RULING 2026-09-17 R2/R3. The tactical behaviour decoder's
            # keys and values are "the scene embeddings, for the agent AND THE
            # MAP". This is the map half, and it is emitted ATTACHED: R3
            # authorises the tactical loss to shape the shared trunk, so a
            # `.detach()` here would silently refuse a ruling the PI made.
            out["bev_tokens"] = self.bev_tokens(bev_feats)
        if self.box_dec is not None:
            mem = self.box_mem(fmap_s16, bev_feats)
            out["box_slots"] = self.box_dec(mem)
        return out


# --------------------------------------------------------------------------- #
# factory                                                                      #
# --------------------------------------------------------------------------- #
def build_perception_branch(model, cfg: PerceptionBranchConfig) -> PerceptionBranch:
    """Read the trunk's OWN shapes off the built model and construct the branch.

    ⛔ Every number comes from the encoder object, never from a literal: a
    resnet34 -> resnet101 swap changes ``s16_dim`` 256 -> 1024 and a 256x640 ->
    256x1024 cache changes ``s16_shape`` (16, 40) -> (16, 64). Both arrive here
    through ``timm``'s ``feature_info`` and the payload, so neither is written
    down anywhere in this file.
    """
    enc = model.core.encoder
    if not hasattr(enc, "forward_features") or not hasattr(enc, "s16_dim"):
        raise SystemExit(
            "[perception] this build's encoder exposes no stride-16 map "
            f"({type(enc).__name__}). refcv6 perception reads stride 16; the "
            "legacy REF-C ResNetEncoder returns the stride-32 map only. Pass "
            "--trunk timm.")
    return PerceptionBranch(cfg, d_image=int(enc.s16_dim),
                            image_hw=tuple(enc.s16_shape))


def frame_for_model(model):
    """The :class:`CanonicalFrame` of THIS build's cache geometry."""
    h, w = model.cfg.core.encoder.image_hw()
    return frame_for_width(int(w), int(h))


# --------------------------------------------------------------------------- #
# the losses -- thin, so the arithmetic stays in the tested modules            #
# --------------------------------------------------------------------------- #
def map_valid_from_lift(valid: Tensor) -> Tensor:
    """``valid [B,Z,X,Y]`` -> ``[B,X,Y]`` bool: the cells the camera actually reaches.

    ⭐ **EXACTLY the predicate** :class:`BEVLift` already uses to decide which cells get
    its ``unobserved`` embedding (``bev_lift.py:262``: ``~valid.any(dim=1)``), spelled
    once so the loss's mask and the feature's substitution cannot drift apart.
    """
    if valid.dim() != 4:
        raise ValueError(f"[perception] lift `valid` must be [B,Z,X,Y], got "
                         f"{tuple(valid.shape)}")
    return valid.any(dim=1)


def map_loss_row(logits: Tensor, frac: Tensor, seen: Tensor, *,
                 lift_valid: Tensor | None = None,
                 with_metrics: bool = False) -> dict:
    """:func:`map_soft_ce` plus the per-head COUNT the log row must carry.

    ⭐ ``n_map_cells`` is not decoration. A map loss of 0.0 with 0 seen cells and
    a map loss of 0.0 on 200,000 cells are opposite findings, and without the
    count the log cannot tell them apart -- the ``tac_goal_n_supervised``
    precedent, one head over.

    ⛔⛔ ``lift_valid`` ``[B,X,Y]`` NARROWS ``seen`` TO WHAT THE CAMERA REACHES AT THIS
    INSTANT, AND PASSING IT IS A BEHAVIOUR CHANGE -- stated, not discovered later.
    The SAM3 ``seen`` mask is a **clip-lifetime** mask: the label artifact declares
    itself ``non_causal`` on **135 of 135** files (*"labels use every frame of the
    clip"*), which is legal for a LABEL under the PI's 2026-08-03 ruling and wrong as
    the supervision mask of a vision-only head. MEASURED 2026-09-22
    (``…/2026-09-22-refcv6-review/raw/p4_map_gt_noncausal.json``): of the **590** cells
    per frame that lie outside the rig's +-60 deg at EVERY instant -- an analytic count
    that the census reproduced exactly -- **90.088 % are labelled ``seen``**, against
    90.597 % of the in-field cells. The mask does not separate them at all.

    Against the mask this pipeline ALREADY COMPUTES (``raw/p7_map_seen_vs_lift_valid.json``,
    135 clips, real per-clip extrinsics): **2,170,570 of 19,647,460 supervised cells =
    11.048 %** are cells :meth:`bev_lift.BEVLift.forward` has already zeroed and replaced
    with a learned ``unobserved`` constant. The loss then asks the head to name a class
    from a constant. ⚠️ 11.048 % is a **LOWER** bound: it prices geometry only (azimuth,
    elevation, range), never inter-agent or hood occlusion.

    ⛔ ``map_metrics`` is narrowed by the SAME mask when it is passed. A loss scored on
    one cell set and an IoU on another is two rules, and the IoU is the number a
    collision gate would read.
    """
    m_seen = seen
    n_prefilter = int(seen.sum())
    if lift_valid is not None:
        if tuple(lift_valid.shape) != tuple(seen.shape):
            raise ValueError(
                f"[perception] lift_valid {tuple(lift_valid.shape)} must match seen "
                f"{tuple(seen.shape)} -- [B, X, Y] on the BEV grid. Reduce the lift's "
                f"[B,Z,X,Y] over heights with map_valid_from_lift first.")
        if lift_valid.dtype != torch.bool:
            raise ValueError(
                f"[perception] lift_valid must be bool, got {lift_valid.dtype} -- the "
                f"same rule map_soft_ce's `seen` carries: a float mask would weight "
                f"cells instead of selecting them")
        m_seen = seen & lift_valid.to(seen.device)
    r = map_soft_ce(logits, frac, m_seen)
    row = {"loss": r["loss"], "n_map_cells": float(r["n_cells"]),
           # ⭐ BOTH counts, always. `n_map_cells_seen` - `n_map_cells` IS the
           # supervision the geometric mask removed, on this batch, in the log row --
           # so an arm that claims the fix can be told from one that only stamped it.
           "n_map_cells_seen": float(n_prefilter),
           "n_map_cells_unobserved": float(n_prefilter - int(r["n_cells"]))}
    if with_metrics:
        m = map_metrics(logits.detach(), frac, m_seen)
        row["map_acc"] = float(m["acc"])
        for i, c in enumerate(m["classes"]):
            v = float(m["iou"][i])
            if v == v:                                   # skip NaN (no union)
                row[f"map_iou_{i}"] = v
    return row


def box3d_loss_row(slots: dict, tgt: dict, *, weights: dict | None = None,
                   cls_class_weight=None, visible_filter: bool = True) -> dict:
    """:func:`box3d_set_loss` plus per-term counts, flattened for the log row.

    ``visible_filter`` is forwarded verbatim; see :func:`box3d_set_loss` for what it
    does, what it measured, and why it defaults ON as of 2026-09-23.
    """
    r = box3d_set_loss(slots, tgt, weights=weights, cls_class_weight=cls_class_weight,
                       visible_filter=visible_filter)
    row = {"loss": r["total"]}
    for k, v in r.items():
        if k.startswith("loss_") and torch.is_tensor(v):
            row[f"box3d_{k[5:]}"] = v
    for k, v in (r.get("n") or {}).items():
        row[f"box3d_n_{k}"] = float(v)
    # ⭐ THE ARM, IN THE LOG ROW. `metrics.jsonl` is what a reader opening a finished run
    # in isolation gets, and a `box3d_n_dropped_not_visible` of 0 means two opposite
    # things -- "the filter is off" and "every box was already visible" -- unless the
    # flag travels beside it. The `anchors.pt` units lesson, in a boolean costume.
    row["box3d_visible_filter"] = 1.0 if visible_filter else 0.0
    return row


# --------------------------------------------------------------------------- #
# the instrument that PROVES the gradient reaches the trunk                    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def grad_abs_sum(module: nn.Module) -> tuple[float, int, int]:
    """``(sum |grad|, n_params, n_params_with_a_grad)`` over a module."""
    s, n, g = 0.0, 0, 0
    for p in module.parameters():
        n += int(p.numel())
        if p.grad is not None:
            s += float(p.grad.detach().abs().sum())
            g += int(p.numel())
    return s, n, g


def grad_reach_report(model, branch: PerceptionBranch | None = None) -> dict:
    """Per-head ``grad_abs_sum`` AFTER a backward -- read, never assumed.

    ⛔ Call it after ``.backward()`` and BEFORE ``opt.zero_grad()``. A head with
    parameters and ``grad_abs_sum`` exactly 0 is the ``tac_goal_tok_head``
    class: parsed, stamped, reaching nothing.
    """
    br = branch if branch is not None else getattr(model, "_perception", None)
    rep: dict = {}
    parts = {"trunk": getattr(model.core, "encoder", None)}
    # ⭐⭐ PI RULING 2026-09-17 R3 — THE TRUNK IS NOW OPTIMISED FOUR WAYS:
    # planner, map head, box head AND the tactical behaviour decoder. The
    # ruling's own stated cost is that a trunk improvement can no longer be
    # assigned to one head, and the named mitigation is per-head reach plus the
    # conflict detector. A report that covers three of the four heads cannot
    # perform that mitigation, so the two non-perception heads are named here.
    # ⛔ `None` when the seam is not built -- the loop below skips it, so this
    # cannot add a key to an arm that has no such head.
    parts["planner"] = getattr(model.core, "decoder", None)
    parts["tac_decoder"] = getattr(model, "tac_decoder_v6", None)
    if br is not None:
        parts.update({"lift": br.lift,
                      "bev_encoder": (None if br.map_branch is None
                                      else br.map_branch.encoder),
                      "map_head": (None if br.map_branch is None
                                   else br.map_branch.head),
                      "box_memory": br.box_mem, "box_decoder": br.box_dec})
    for name, m in parts.items():
        if m is None:
            continue
        s, n, g = grad_abs_sum(m)
        rep[name] = {"grad_abs_sum": s, "n_params": n, "n_params_with_grad": g}
    return rep


if __name__ == "__main__":                                # pragma: no cover
    # ⛔ NO LITERAL GEOMETRY, not even in a demo: every shape comes from
    # `TrunkSpec`, which reads timm's `feature_info` and the frame's own width.
    # A demo that writes 256/(16, 40) down is the first place a hard-coded
    # geometry reappears, and `tests/test_refcv6_geometry_agnostic.py` plus
    # `test_refcv6_perception_training.py` both scan this file for exactly that.
    from tanitad.models.trunk_shapes import TrunkSpec
    c = PerceptionBranchConfig(w_map=1.0, w_box3d=1.0)
    for name in ("resnet34.a1_in1k", "resnet101.a1_in1k"):
        for width in (640, 1024):  # geometry-exempt: a __main__ demo must name SOME widths to print a table; nothing importable reads them, and the SHAPES still come from TrunkSpec
            sp = TrunkSpec.from_timm(name, frame_for_width(width))
            b = PerceptionBranch(c, d_image=sp.perception.channels,
                                 image_hw=sp.perception.hw)
            print(f"{name} @ {sp.frame.height}x{sp.frame.width}: "
                  f"{b.param_breakdown()}")
