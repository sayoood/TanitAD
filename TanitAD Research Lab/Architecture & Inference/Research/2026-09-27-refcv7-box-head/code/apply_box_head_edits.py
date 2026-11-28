#!/usr/bin/env python3
"""refcv7 A9 (the REFINED box heads) -- the SHARED-FILE edits, as ANCHORED replacements.

SPEC_REFCV7 §14 (A9) + §14.1, PI verbatim 2026-09-27: *"based on the compariosn with proven refernce heads, refine our
bb head and validate it"*. The NEW modules (``tanitad/models/slot_presence.py``, ``tanitad/data/vis1.py``,
``tanitad/data/vis1_zbuffer.py``, ``tanitad/eval/detection_metrics.py``, ``tanitad/train/box_head_guard.py``,
``scripts/precompute_vis1_sidecar.py``, ``scripts/g_box_overfit.py`` and their tests) are FULL FILES in ``code/new/``.
This script makes the edits to files OTHER streams also edit (landing order: dvb batch 2 -> NEW-2 batch 2 -> dvb
batch 3 -> this), so the rebase onto whatever tip exists then is MECHANICAL:

* every anchor must occur EXACTLY ONCE in the base (or the count named in ``COUNTS``) or the run FAILS LOUDLY naming
  it -- a moved anchor is a rebase conflict to resolve by READING, never by guessing;
* each output keeps its base blob's line endings byte for byte (all targets are CRLF at 68e8ec6, MEASURED);
* the G-DVB registry-count pin is bumped RELATIVE to the base (``+N_NEW_DVB``), so earlier batches' own bumps survive.

usage:
  python apply_box_head_edits.py --base-git C:/Users/Admin/tanitad-push/.git --rev <tip> --out <dir>
  python apply_box_head_edits.py --base-dir <tree> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

AS = "stack/tanitad/models/agent_slots.py"
B3 = "stack/tanitad/models/box3d_head.py"
RA = "stack/tanitad/refs/refc_agents.py"
PB = "stack/tanitad/models/refcv6_perception_branch.py"
TR = "stack/scripts/refc_v3_train.py"
DVB = "stack/tanitad/train/declared_vs_built.py"
DVB_T = "stack/tests/test_declared_vs_built.py"
V6_T = "stack/tests/test_v6_agent_slots.py"
SF_T = "stack/tests/test_refcv6_perception_supervision_fixes.py"
PROV_T = "stack/tests/test_refc_v3_agent_provenance.py"
ARM = "taniteval/tools/refcv3_arm.py"
OCC_T = "stack/tests/test_occ_knob_is_stamped.py"
SAVE_T = "stack/tests/test_refc_v3_save_before_eval.py"
FILES = (AS, B3, RA, PB, TR, DVB, DVB_T, V6_T, SF_T, PROV_T, OCC_T, ARM, SAVE_T)

#: the new G-DVB registry entries this batch adds (the count pin moves by exactly this).
N_NEW_DVB = 5

E: list = []                      # (file, edit_id, old, new)
COUNTS: dict = {}


def edit(path: str, eid: str, old: str, new: str, count: int = 1) -> None:
    E.append((path, eid, old, new))
    if count != 1:
        COUNTS[(path, eid)] = count


# ===================================================================================================== #
# agent_slots.py -- R4 (one spelling, 300), R1 hooks (prior, focal matcher cost), R2 hook (per-layer)   #
# ===================================================================================================== #
edit(AS, "as.n_queries",
     "#: ⛔ THIS IS THE ONLY SPELLING. ``train_v6_staged.py`` imports it for both\n"
     "#: its argparse default and its ``getattr`` fallback, and\n"
     "#: ``tests/test_v6_agent_slots.py`` FAILS if either re-grows a literal.\n"
     "N_QUERIES_DEFAULT: int = 100\n",
     "#: ⛔ THIS IS THE ONLY SPELLING. ``train_v6_staged.py`` imports it for both\n"
     "#: its argparse default and its ``getattr`` fallback, and\n"
     "#: ``tests/test_v6_agent_slots.py`` FAILS if either re-grows a literal.\n"
     "#: ⭐⭐ RE-RULED 2026-09-27 (SPEC_REFCV7 §14 A9, R4 -- M17's zero-drop rule was already BROKEN\n"
     "#: at 100): **300**. The refcv6 corpus line (v7-B1, not the parity join M17 was ruled on)\n"
     "#: reaches **120 targets per window** on the B1 eval clip ``0191487845ef`` (80.3 mean, 146\n"
     "#: targets with no slot -- the literature pass's S-crowd), and DETR's own appendix shows a\n"
     "#: 100-query DETR finds every instance only up to ~50, i.e. ~N/2. The rule is therefore\n"
     "#: **N >= 2 x the observed max**, which 300 meets (2.5 x 120) BEFORE VIS-1 narrows the targets\n"
     "#: (the audit: 26.2 -> 9.0 targets per window at vis >= 0.30 on the crowded pool). The\n"
     "#: nuScenes camera heads run 900. Cost: +200 x d_model = +51,200 parameters per head (the\n"
     "#: box3d decoder 3,806,999 -> 3,858,199 and the agent head 3,822,869 -> 3,874,069, both inside\n"
     "#: PARAM_BAND -- pinned by ``tests/test_refcv7_box_head.py``). ⚠️ A run recorded BEFORE this\n"
     "#: ruling is rebuilt at ITS OWN count: ``refc_v3_train.agent_queries_as_trained`` reads the\n"
     "#: stamp, and the perception branch is rebuilt from its ``n_queries`` stamp.\n"
     "N_QUERIES_DEFAULT: int = 300\n")

edit(AS, "as.presence_costs",
     "#: DETR's ∅-class down-weight: unmatched slots vastly outnumber matched ones,\n"
     "#: and an unweighted BCE simply learns \"always empty\".\n"
     "NO_OBJECT_W: float = 0.1\n",
     "#: DETR's ∅-class down-weight: unmatched slots vastly outnumber matched ones,\n"
     "#: and an unweighted BCE simply learns \"always empty\".\n"
     "NO_OBJECT_W: float = 0.1\n"
     "\n"
     "#: refcv7 A9 R1 -- the PRESENCE term of the Hungarian cost. ``\"sigmoid\"`` (the default,\n"
     "#: every arm before A9, bit-identical): ``MATCH_COST_W[\"presence\"] x (-sigmoid(presence))``.\n"
     "#: ``\"focal\"``: ``slot_presence.FOCAL_MATCH_W`` (2.0) x mmdet's ``FocalLossCost`` on the\n"
     "#: presence logit -- the DETR3D / BEVFormer matcher. The class, centre and size terms are\n"
     "#: unchanged either way (one variable per arm).\n"
     "PRESENCE_COSTS: tuple[str, ...] = (\"sigmoid\", \"focal\")\n")

edit(AS, "as.init_sig",
     "                 ranges: SlotDecodeRanges | None = None,\n"
     "                 enforce_band: bool = True):\n"
     "        super().__init__()\n"
     "        if n_queries < 1:\n",
     "                 ranges: SlotDecodeRanges | None = None,\n"
     "                 enforce_band: bool = True,\n"
     "                 presence_prior: float | None = None):\n"
     "        super().__init__()\n"
     "        if n_queries < 1:\n")

edit(AS, "as.init_attrs",
     "        self.occ_from_geometry: bool = False\n"
     "\n"
     "        self.mem_proj = nn.Linear(self.d_memory, self.d_model)\n",
     "        self.occ_from_geometry: bool = False\n"
     "        #: ⛔ refcv7 A9 R2, OPT-IN, DEFAULT OFF (the ``occ_from_geometry`` idiom: an attribute,\n"
     "        #: set by the builders from a DECLARED config field). When True, :meth:`forward` also\n"
     "        #: reads EVERY decoder layer through the SHARED ``norm`` + ``head`` (0 new parameters) and\n"
     "        #: returns the earlier layers' decodes under ``\"aux\"``; the returned dict's own keys stay\n"
     "        #: the LAST layer's, bit-identical to the off path (pinned by a test).\n"
     "        self.deep_supervision: bool = False\n"
     "        #: refcv7 A9 R1: the presence probability this head was INITIALISED at (the logit bias).\n"
     "        #: ``None`` keeps :attr:`PRESENCE_PRIOR` (0.05), so every pre-A9 build is unchanged.\n"
     "        self.presence_prior: float = float(\n"
     "            self.PRESENCE_PRIOR if presence_prior is None else presence_prior)\n"
     "        if not 0.0 < self.presence_prior < 1.0:\n"
     "            raise ValueError(f\"presence_prior must be in (0, 1), got \"\n"
     "                             f\"{self.presence_prior}\")\n"
     "\n"
     "        self.mem_proj = nn.Linear(self.d_memory, self.d_model)\n")

edit(AS, "as.init_bias",
     "            self.head.bias[SLOT_SLICES[\"presence\"]] = math.log(\n"
     "                self.PRESENCE_PRIOR / (1.0 - self.PRESENCE_PRIOR))\n",
     "            self.head.bias[SLOT_SLICES[\"presence\"]] = math.log(\n"
     "                self.presence_prior / (1.0 - self.presence_prior))\n")

edit(AS, "as.forward",
     "        b = memory.shape[0]\n"
     "        mem = self.mem_proj(memory) + self.mem_pos.to(memory.dtype)\n"
     "        q = self.queries.to(memory.dtype).expand(b, -1, -1)\n"
     "        raw = self.head(self.norm(self.blocks(q, mem)))          # [B, N, W]\n"
     "        return self.decode(raw)\n",
     "        b = memory.shape[0]\n"
     "        mem = self.mem_proj(memory) + self.mem_pos.to(memory.dtype)\n"
     "        q = self.queries.to(memory.dtype).expand(b, -1, -1)\n"
     "        if not getattr(self, \"deep_supervision\", False):\n"
     "            raw = self.head(self.norm(self.blocks(q, mem)))      # [B, N, W]\n"
     "            return self.decode(raw)\n"
     "        # ---- refcv7 A9 R2: every layer through the SHARED norm + head -------- #\n"
     "        # ``nn.TransformerDecoder.forward`` is exactly this loop (no masks, not\n"
     "        # causal, no stack-level norm), so the LAST entry is bit-identical to the\n"
     "        # off path; the earlier ones are the deep-supervision targets.\n"
     "        if self.blocks.norm is not None:\n"
     "            raise RuntimeError(\"deep supervision assumes no stack-level norm\")\n"
     "        x = q\n"
     "        raws = []\n"
     "        for layer in self.blocks.layers:\n"
     "            x = layer(x, mem)\n"
     "            raws.append(self.head(self.norm(x)))\n"
     "        out = self.decode(raws[-1])\n"
     "        out[\"aux\"] = [self.decode(r) for r in raws[:-1]]\n"
     "        return out\n")

edit(AS, "as.match_cost_sig",
     "def _match_cost(pred: dict, tgt: dict, b: int, keep: np.ndarray) -> np.ndarray:\n",
     "def _match_cost(pred: dict, tgt: dict, b: int, keep: np.ndarray,\n"
     "                presence_cost: str = \"sigmoid\") -> np.ndarray:\n")

edit(AS, "as.match_cost_pres",
     "    pres = -pred[\"presence_logit\"][b].sigmoid()[:, None].expand_as(centre)\n"
     "    c = (w[\"centre_m\"] * centre + w[\"size_m\"] * size\n"
     "         + w[\"cls\"] * cls_c + w[\"presence\"] * pres)\n",
     "    if presence_cost == \"focal\":\n"
     "        # refcv7 A9 R1: mmdet FocalLossCost x 2.0 (DETR3D / BEVFormer).\n"
     "        from tanitad.models.slot_presence import FOCAL_MATCH_W, focal_presence_cost\n"
     "        pres = focal_presence_cost(\n"
     "            pred[\"presence_logit\"][b])[:, None].expand_as(centre)\n"
     "        w_pres = FOCAL_MATCH_W\n"
     "    elif presence_cost == \"sigmoid\":\n"
     "        pres = -pred[\"presence_logit\"][b].sigmoid()[:, None].expand_as(centre)\n"
     "        w_pres = w[\"presence\"]\n"
     "    else:\n"
     "        raise ValueError(f\"presence_cost {presence_cost!r} not in {PRESENCE_COSTS}\")\n"
     "    c = (w[\"centre_m\"] * centre + w[\"size_m\"] * size\n"
     "         + w[\"cls\"] * cls_c + w_pres * pres)\n")

edit(AS, "as.match_slots_sig",
     "def match_slots(pred: dict, tgt: dict) -> dict:\n",
     "def match_slots(pred: dict, tgt: dict, *, presence_cost: str = \"sigmoid\") -> dict:\n")

edit(AS, "as.match_slots_doc",
     "    exactly on crowded frames.\n"
     "    \"\"\"\n"
     "    rows, cols, n_t, n_d = [], [], [], []\n",
     "    exactly on crowded frames.\n"
     "\n"
     "    ``presence_cost`` (refcv7 A9 R1): ``\"sigmoid\"`` (default, bit-identical) or\n"
     "    ``\"focal\"`` -- see :data:`PRESENCE_COSTS`.\n"
     "    \"\"\"\n"
     "    if presence_cost not in PRESENCE_COSTS:\n"
     "        raise ValueError(f\"presence_cost {presence_cost!r} not in {PRESENCE_COSTS}\")\n"
     "    rows, cols, n_t, n_d = [], [], [], []\n")

edit(AS, "as.match_slots_call",
     "            c = _match_cost(pred, tgt, b, valid)\n",
     "            c = _match_cost(pred, tgt, b, valid, presence_cost=presence_cost)\n")

# ===================================================================================================== #
# box3d_head.py -- the prior passes through                                                             #
# ===================================================================================================== #
edit(B3, "b3.init",
     "                 z_range_m: float = Z_RANGE_M, h_range_m: float = H_RANGE_M,\n"
     "                 enforce_band: bool = True):\n"
     "        super().__init__(d_memory, n_memory, n_queries=n_queries,\n"
     "                         d_model=d_model, depth=depth, n_heads=n_heads,\n"
     "                         ranges=ranges, enforce_band=False)\n",
     "                 z_range_m: float = Z_RANGE_M, h_range_m: float = H_RANGE_M,\n"
     "                 enforce_band: bool = True,\n"
     "                 presence_prior: float | None = None):\n"
     "        # refcv7 A9 R1: the prior is set on the 2-D head and COPIED below with\n"
     "        # the rest of the 2-D init, so there is one spelling of the bias.\n"
     "        super().__init__(d_memory, n_memory, n_queries=n_queries,\n"
     "                         d_model=d_model, depth=depth, n_heads=n_heads,\n"
     "                         ranges=ranges, enforce_band=False,\n"
     "                         presence_prior=presence_prior)\n")

# ===================================================================================================== #
# refc_agents.py -- the agent head's declared fields, its builder, and the refined dispatch             #
# ===================================================================================================== #
edit(RA, "ra.queries",
     "    #: failure the recipe exists to prevent.\n"
     "    queries: int = 100\n",
     "    #: failure the recipe exists to prevent.\n"
     "    #: ⭐⭐ RE-RULED 2026-09-27 (SPEC_REFCV7 §14 A9, R4): the ONE spelling, now 300 -- see\n"
     "    #: :data:`agent_slots.N_QUERIES_DEFAULT` (>= 2 x the 120-target max on the refcv6 line). A\n"
     "    #: record from before the ruling rebuilds at its stamped count\n"
     "    #: (``refc_v3_train.agent_queries_as_trained``).\n"
     "    queries: int = N_QUERIES_DEFAULT\n")

edit(RA, "ra.fields",
     "    occ_from_geometry: bool = False\n"
     "\n"
     "    def as_dict(self) -> dict:\n",
     "    occ_from_geometry: bool = False\n"
     "    #: ⭐⭐ refcv7 A9 (SPEC_REFCV7 §14) -- the REFINED head. Every default is the pre-A9 arm,\n"
     "    #: so an arm that asks for none of them is bit-identical (``slot_presence.refined_is_legacy``).\n"
     "    #: R1: ``\"focal\"`` = sigmoid focal presence (alpha 0.25, gamma 2, weight 2.0, / n_matched)\n"
     "    #: AND the focal matching cost; ``\"bce\"`` = the historical BCE with NO_OBJECT_W 0.1.\n"
     "    presence_loss: str = \"bce\"\n"
     "    #: R1: the presence logit's init prior (0.01 under A9; 0.05 = AgentSlotDecoder.PRESENCE_PRIOR).\n"
     "    presence_prior: float = 0.05\n"
     "    #: R2: supervise every decoder layer (shared heads, re-matched per layer, summed).\n"
     "    deep_supervision: bool = False\n"
     "    #: R3: VIS-1 targets with IGNORE semantics (``tanitad.data.vis1``); needs the sidecar.\n"
     "    vis1: bool = False\n"
     "\n"
     "    def as_dict(self) -> dict:\n")

edit(RA, "ra.as_dict",
     "            \"occ_from_geometry\": bool(self.occ_from_geometry),\n"
     "            \"n_classes\": int(N_AGENT_CLASSES),\n",
     "            \"occ_from_geometry\": bool(self.occ_from_geometry),\n"
     "            \"presence_loss\": str(self.presence_loss),\n"
     "            \"presence_prior\": float(self.presence_prior),\n"
     "            \"deep_supervision\": bool(self.deep_supervision),\n"
     "            \"vis1\": bool(self.vis1),\n"
     "            \"n_classes\": int(N_AGENT_CLASSES),\n")

edit(RA, "ra.build",
     "        depth=int(cfg.depth), n_heads=int(cfg.n_heads),\n"
     "        enforce_band=bool(cfg.enforce_band))\n",
     "        depth=int(cfg.depth), n_heads=int(cfg.n_heads),\n"
     "        enforce_band=bool(cfg.enforce_band),\n"
     "        presence_prior=float(getattr(cfg, \"presence_prior\",\n"
     "                                     AgentSlotDecoder.PRESENCE_PRIOR)))\n")

edit(RA, "ra.build_plumb",
     "    head.occ_from_geometry = bool(cfg.occ_from_geometry)\n"
     "    return head\n",
     "    head.occ_from_geometry = bool(cfg.occ_from_geometry)\n"
     "    # ⛔ refcv7 A9 R2 -- DECLARED IS NOT PLUMBED, the same rule one line up.\n"
     "    head.deep_supervision = bool(getattr(cfg, \"deep_supervision\", False))\n"
     "    return head\n")

edit(RA, "ra.losses_sig",
     "                 filter_visible: bool = True,\n"
     "                 cls_class_weight=None) -> dict:\n",
     "                 filter_visible: bool = True,\n"
     "                 cls_class_weight=None,\n"
     "                 vis: dict | None = None) -> dict:\n")

edit(RA, "ra.losses_dispatch",
     "    # ⛔ THE FILTER IS ON BY DEFAULT AND THAT IS THE POINT. Without it 61.8 %\n",
     "    # ---- refcv7 A9 (R1-R3): the REFINED path, taken only when asked for ------ #\n"
     "    # ⭐ The historical configuration (BCE, no VIS-1, no per-layer outputs) never\n"
     "    # reaches it, so every pre-A9 arm runs the code below unchanged.\n"
     "    from tanitad.models import slot_presence as _sp\n"
     "    if not _sp.refined_is_legacy(getattr(cfg, \"presence_loss\", \"bce\"),\n"
     "                                 getattr(cfg, \"vis1\", False), slots):\n"
     "        if not filter_visible:\n"
     "            raise ValueError(\n"
     "                \"refcv7 A9: the refined agent loss always applies the field cut \"\n"
     "                \"(VIS-1 is applied AFTER visible_target_filter); filter_visible=False \"\n"
     "                \"is the pre-A9 regression arm and runs only on the legacy path.\")\n"
     "        return _sp.refined_agent_losses(slots, tgt, cfg, cam=cam, weights=weights,\n"
     "                                        cls_class_weight=cls_class_weight, vis=vis)\n"
     "    # ⛔ THE FILTER IS ON BY DEFAULT AND THAT IS THE POINT. Without it 61.8 %\n")

# ===================================================================================================== #
# refcv6_perception_branch.py -- the box head's declared fields, its builder, the loss row             #
# ===================================================================================================== #
edit(PB, "pb.import",
     "from tanitad.models.box3d_head import (Box3DMemory, Box3DSlotDecoder,\n"
     "                                       box3d_set_loss)\n",
     "from tanitad.models.box3d_head import (Box3DMemory, Box3DSlotDecoder,\n"
     "                                       box3d_set_loss)\n"
     "from tanitad.models.agent_slots import N_QUERIES_DEFAULT\n")

edit(PB, "pb.n_queries",
     "    n_queries: int = 100              # agent_slots.N_QUERIES_DEFAULT\n",
     "    #: ⛔ the ONE spelling (refcv7 A9 R4: 300). A run recorded before the ruling is rebuilt from\n"
     "    #: its own ``refcv6_perception.n_queries`` stamp (``refcv3_arm.rebuild_perception_branch``).\n"
     "    n_queries: int = N_QUERIES_DEFAULT\n")

edit(PB, "pb.fields",
     "    occ_from_geometry: bool = False\n",
     "    occ_from_geometry: bool = False\n"
     "    #: ⭐⭐ refcv7 A9 (SPEC_REFCV7 §14) -- the REFINED box head; every default is the pre-A9 arm.\n"
     "    #: R1 presence objective (``\"bce\"`` | ``\"focal\"``), R1 init prior, R2 per-layer\n"
     "    #: supervision, R3 VIS-1 targets. Same four fields, same meaning, as\n"
     "    #: ``refc_agents.AgentSeamConfig`` -- the two heads share ``AgentSlotDecoder``.\n"
     "    presence_loss: str = \"bce\"\n"
     "    presence_prior: float = 0.05\n"
     "    deep_supervision: bool = False\n"
     "    vis1: bool = False\n")

edit(PB, "pb.as_dict",
     "                \"occ_from_geometry\": bool(self.occ_from_geometry),\n",
     "                \"occ_from_geometry\": bool(self.occ_from_geometry),\n"
     "                \"presence_loss\": str(self.presence_loss),\n"
     "                \"presence_prior\": float(self.presence_prior),\n"
     "                \"deep_supervision\": bool(self.deep_supervision),\n"
     "                \"vis1\": bool(self.vis1),\n")

edit(PB, "pb.build_prior",
     "                enforce_band=bool(cfg.enforce_param_band))\n",
     "                enforce_band=bool(cfg.enforce_param_band),\n"
     "                presence_prior=float(cfg.presence_prior))\n")

edit(PB, "pb.build_plumb",
     "            self.box_dec.occ_from_geometry = bool(cfg.occ_from_geometry)\n",
     "            self.box_dec.occ_from_geometry = bool(cfg.occ_from_geometry)\n"
     "            # ⛔ refcv7 A9 R2 -- declared is not plumbed; the test pins THIS line too.\n"
     "            self.box_dec.deep_supervision = bool(cfg.deep_supervision)\n")

edit(PB, "pb.loss_row",
     "def box3d_loss_row(slots: dict, tgt: dict, *, weights: dict | None = None,\n"
     "                   cls_class_weight=None, visible_filter: bool = True) -> dict:\n",
     "def box3d_loss_row(slots: dict, tgt: dict, *, weights: dict | None = None,\n"
     "                   cls_class_weight=None, visible_filter: bool = True,\n"
     "                   presence_loss: str = \"bce\", vis1: bool = False,\n"
     "                   vis: dict | None = None) -> dict:\n")

edit(PB, "pb.loss_row_call",
     "    r = box3d_set_loss(slots, tgt, weights=weights, cls_class_weight=cls_class_weight,\n"
     "                       visible_filter=visible_filter)\n",
     "    # ---- refcv7 A9 (R1-R3): the REFINED path, only when asked for --------------- #\n"
     "    from tanitad.models import slot_presence as _sp\n"
     "    if _sp.refined_is_legacy(presence_loss, vis1, slots):\n"
     "        r = box3d_set_loss(slots, tgt, weights=weights, cls_class_weight=cls_class_weight,\n"
     "                           visible_filter=visible_filter)\n"
     "    else:\n"
     "        if not visible_filter:\n"
     "            raise ValueError(\"refcv7 A9: the refined box loss always applies the field cut; \"\n"
     "                             \"visible_filter=False is the pre-A9 regression arm (legacy path)\")\n"
     "        r = _sp.refined_box3d_losses(slots, tgt, presence_loss=presence_loss, vis1=vis1,\n"
     "                                     vis=vis, visible_filter=True, weights=weights,\n"
     "                                     cls_class_weight=cls_class_weight)\n")

edit(PB, "pb.loss_row_flags",
     "    row[\"box3d_visible_filter\"] = 1.0 if visible_filter else 0.0\n"
     "    return row\n",
     "    row[\"box3d_visible_filter\"] = 1.0 if visible_filter else 0.0\n"
     "    if \"_refine\" in r:                   # refcv7 A9: the arm, in the log row\n"
     "        row[\"box3d_presence_focal\"] = 1.0 if presence_loss == \"focal\" else 0.0\n"
     "        row[\"box3d_vis1\"] = 1.0 if vis1 else 0.0\n"
     "        row[\"box3d_presence_frac_confident\"] = _sp.presence_sanity(\n"
     "            slots[\"presence_logit\"])[\"frac_confident\"]\n"
     "    return row\n")

# ===================================================================================================== #
# refc_v3_train.py -- flags, pins, dataset join, loss wiring, in-run P0, config stamp, as-trained      #
# ===================================================================================================== #
edit(TR, "tr.imports",
     "from tanitad.models import agent_slots as _agent_slots  # noqa: E402\n",
     "from tanitad.models import agent_slots as _agent_slots  # noqa: E402\n"
     "from tanitad.models import slot_presence as _slot_presence  # noqa: E402  (refcv7 A9)\n"
     "from tanitad.data import vis1 as _vis1  # noqa: E402  (refcv7 A9 R3)\n"
     "from tanitad.eval import detection_metrics as _det_metrics  # noqa: E402  (refcv7 A9 P0)\n")

edit(TR, "tr.queries_default",
     "AGENT_QUERIES_DEFAULT = 100\n",
     "#: ⭐⭐ RE-RULED 2026-09-27 (SPEC_REFCV7 §14 A9, R4): the ONE spelling is\n"
     "#: `agent_slots.N_QUERIES_DEFAULT` (300 = 2.5 x the 120-target max on the refcv6 corpus line;\n"
     "#: DETR finds every instance only up to ~N/2). This name stays for its readers. A run\n"
     "#: recorded before the ruling (its argv has no --agent-queries) is rebuilt at its STAMPED\n"
     "#: count by `agent_queries_as_trained`, never at this default.\n"
     "from tanitad.models.agent_slots import N_QUERIES_DEFAULT as AGENT_QUERIES_DEFAULT  # noqa: E402\n")

edit(TR, "tr.agent_cfg",
     "            presence_hard=bool(getattr(args, \"agent_presence_hard\", False)))\n"
     "        core.agents = acfg\n",
     "            presence_hard=bool(getattr(args, \"agent_presence_hard\", False)),\n"
     "            # ⭐⭐ refcv7 A9: the refined head, from the four --slot-* flags\n"
     "            **_slot_refine_kwargs(args))\n"
     "        core.agents = acfg\n")

edit(TR, "tr.pin_call",
     "    _pin_refcv6_perception(cfg, args)\n",
     "    _pin_refcv6_perception(cfg, args)\n"
     "    _pin_slot_refine(cfg, args)\n")

edit(TR, "tr.pin_fn",
     "def _pin_refcv6_perception(cfg, args) -> None:\n",
     "def _slot_refine_kwargs(args) -> dict:\n"
     "    \"\"\"refcv7 A9 -- the four --slot-* flags as the config fields BOTH slot heads declare\n"
     "    (``AgentSeamConfig`` and ``PerceptionBranchConfig``: one spelling of the mapping).\"\"\"\n"
     "    return {\"presence_loss\": str(getattr(args, \"slot_presence_loss\", \"bce\") or \"bce\"),\n"
     "            \"presence_prior\": float(getattr(args, \"slot_presence_prior\", 0.05)),\n"
     "            \"deep_supervision\": bool(getattr(args, \"slot_deep_supervision\", False)),\n"
     "            \"vis1\": bool(getattr(args, \"slot_vis1\", False))}\n"
     "\n"
     "\n"
     "def _slot_refine_active(args) -> bool:\n"
     "    k = _slot_refine_kwargs(args)\n"
     "    return (k[\"presence_loss\"] != \"bce\" or abs(k[\"presence_prior\"] - 0.05) > 1e-12\n"
     "            or k[\"deep_supervision\"] or k[\"vis1\"]\n"
     "            or bool(getattr(args, \"vis1_sidecar\", None)))\n"
     "\n"
     "\n"
     "def _pin_slot_refine(cfg, args) -> None:\n"
     "    \"\"\"⛔ refcv7 A9: refuse every --slot-* / --vis1-sidecar combination that cannot train.\n"
     "\n"
     "    The M18 rule, again: a flag that parses, is STAMPED, and reaches no head is refused\n"
     "    here -- before config.json and before a batch -- never discovered from a flat metric.\n"
     "    \"\"\"\n"
     "    if not _slot_refine_active(args):\n"
     "        return\n"
     "    k = _slot_refine_kwargs(args)\n"
     "    if not 0.0 < k[\"presence_prior\"] < 1.0:\n"
     "        raise SystemExit(\"[v3] ⛔ --slot-presence-prior must be in (0, 1).\")\n"
     "    agent_head = str(getattr(args, \"agents\", \"off\")) == \"head\"\n"
     "    box_head = float(getattr(args, \"w_box3d\", 0.0) or 0.0) > 0.0\n"
     "    if not (agent_head or box_head):\n"
     "        raise SystemExit(\n"
     "            \"[v3] ⛔ --slot-* / --vis1-sidecar set (%s) but NO learned slot head is built \"\n"
     "            \"(--agents head or --w-box3d > 0). The refinement would be STAMPED and train \"\n"
     "            \"nothing (mm-decisions M18).\" % k)\n"
     "    if k[\"vis1\"] and not getattr(args, \"vis1_sidecar\", None):\n"
     "        raise SystemExit(\n"
     "            \"[v3] ⛔ --slot-vis1 without --vis1-sidecar: VIS-1 visibility is PRECOMPUTED \"\n"
     "            \"(scripts/precompute_vis1_sidecar.py) and never guessed.\")\n"
     "    if getattr(args, \"vis1_sidecar\", None) and not k[\"vis1\"]:\n"
     "        raise SystemExit(\n"
     "            \"[v3] ⛔ --vis1-sidecar without --slot-vis1: the sidecar would be loaded, \"\n"
     "            \"joined into every batch and read by no loss (mm-decisions M18).\")\n"
     "    if k[\"vis1\"] and not (getattr(args, \"agent_join\", None)\n"
     "                          and getattr(args, \"join3d\", None)):\n"
     "        raise SystemExit(\n"
     "            \"[v3] ⛔ --slot-vis1 needs --agent-join AND --join3d: the sidecar is keyed by \"\n"
     "            \"the 2-D join's rows and was z-buffered over the 3-D cuboids.\")\n"
     "\n"
     "\n"
     "def agent_queries_as_trained(config: dict, args=None) -> tuple:\n"
     "    \"\"\"``(n, why)`` -- the agent head's query count AS TRAINED, for a loader that re-parses a\n"
     "    RECORDED argv with THIS parser.\n"
     "\n"
     "    ⛔ refcv7 A9 R4 moved the ONE spelling from 100 to 300. A record whose argv never passed\n"
     "    ``--agent-queries`` (refcv6-r101-s0 included) would otherwise be rebuilt at the NEW default and\n"
     "    fail its strict load -- or, worse, be compared at a count it never had. The stamp\n"
     "    ``seams.agents.queries`` is what the run BUILT, so it wins over the parser default; an\n"
     "    explicit argv value wins over both. ``(None, why)`` when the record carries neither.\n"
     "    \"\"\"\n"
     "    argv = list((config or {}).get(\"argv\") or [])\n"
     "    if any(str(x) == \"--agent-queries\" or str(x).startswith(\"--agent-queries=\")\n"
     "           for x in argv):\n"
     "        return (int(getattr(args, \"agent_queries\")) if args is not None else None,\n"
     "                \"argv --agent-queries\")\n"
     "    st = (((config or {}).get(\"seams\") or {}).get(\"agents\") or {})\n"
     "    if \"queries\" in st:\n"
     "        return int(st[\"queries\"]), \"config.json seams.agents.queries (argv silent)\"\n"
     "    return None, \"no --agent-queries in argv and no seams.agents stamp\"\n"
     "\n"
     "\n"
     "def _vis1_batch_block(model, batch: dict, device, keep=None):\n"
     "    \"\"\"refcv7 A9 R3: the batch's VIS-1 block for the loss, or ``None`` when VIS-1 is off.\n"
     "\n"
     "    ⛔ REFUSE, DO NOT SKIP -- the `agent_box` rule: VIS-1 on and no `agent_vis_known` in the\n"
     "    batch means the dataset never joined the sidecar, and a silent fallback would train the\n"
     "    pre-A9 target set while config.json states VIS-1.\n"
     "    \"\"\"\n"
     "    if not bool(getattr(model, \"_vis1\", False)):\n"
     "        return None\n"
     "    miss = [k for k in (\"agent_vis_full\", \"agent_vis_px\", \"agent_vis_known\")\n"
     "            if k not in batch]\n"
     "    if miss:\n"
     "        raise SystemExit(\n"
     "            \"[v3] ⛔ --slot-vis1 but the batch carries no %s: the dataset never joined the \"\n"
     "            \"VIS-1 sidecar (`ds.enable_vis1`).\" % miss)\n"
     "    v = {\"n_full\": batch[\"agent_vis_full\"].to(device),\n"
     "         \"n_vis\": batch[\"agent_vis_px\"].to(device),\n"
     "         \"known\": batch[\"agent_vis_known\"].to(device)}\n"
     "    if keep is not None:\n"
     "        sel = keep.to(device).nonzero(as_tuple=False).flatten()\n"
     "        v = {k2: t.index_select(0, sel) for k2, t in v.items()}\n"
     "    return v\n"
     "\n"
     "\n"
     "def _slot_refine_block(args, model) -> dict | None:\n"
     "    \"\"\"config.json[`slot_refine`]: the refinement AS BUILT, per head; ``None`` for a pre-A9 arm.\"\"\"\n"
     "    if not _slot_refine_active(args):\n"
     "        return None\n"
     "    heads = {}\n"
     "    ah = getattr(getattr(model, \"core\", model), \"agent_head\", None)\n"
     "    if ah is not None and hasattr(ah, \"deep_supervision\"):\n"
     "        heads[\"agent\"] = {\"n_queries\": int(ah.n_queries),\n"
     "                          \"deep_supervision\": bool(ah.deep_supervision),\n"
     "                          \"presence_prior\": float(ah.presence_prior),\n"
     "                          \"n_decoder_layers\": len(ah.blocks.layers),\n"
     "                          \"n_params\": int(ah.n_params)}\n"
     "    br = getattr(model, \"_perception\", None)\n"
     "    bd = getattr(br, \"box_dec\", None) if br is not None else None\n"
     "    if bd is not None:\n"
     "        heads[\"box3d\"] = {\"n_queries\": int(bd.n_queries),\n"
     "                          \"deep_supervision\": bool(bd.deep_supervision),\n"
     "                          \"presence_prior\": float(bd.presence_prior),\n"
     "                          \"n_decoder_layers\": len(bd.blocks.layers),\n"
     "                          \"n_params\": int(bd.n_params)}\n"
     "    k = _slot_refine_kwargs(args)\n"
     "    return {**_slot_presence.refine_stamp(k[\"presence_loss\"], k[\"vis1\"]),\n"
     "            \"presence_prior\": k[\"presence_prior\"],\n"
     "            \"deep_supervision\": k[\"deep_supervision\"], \"heads_built\": heads,\n"
     "            \"g_live_presence_max_confident_frac\":\n"
     "                _slot_presence.G_LIVE_PRESENCE_MAX_CONFIDENT_FRAC,\n"
     "            \"p0_keys\": {h: len(_det_metrics.metric_keys(h)) for h in heads}}\n"
     "\n"
     "\n"
     "def _pin_refcv6_perception(cfg, args) -> None:\n")

edit(TR, "tr.ds_attrs",
     "    join3d_stats: dict | None = None\n",
     "    join3d_stats: dict | None = None\n"
     "    #: ⭐⭐ refcv7 A9 R3 -- the VIS-1 sidecar (`tanitad.data.vis1.VIS1Sidecar`), set by\n"
     "    #: :meth:`enable_vis1`. While it is None the batch carries NO `agent_vis_*` and a\n"
     "    #: `--slot-vis1` loss REFUSES (`_vis1_batch_block`).\n"
     "    vis1_sidecar = None\n"
     "    vis1_stats: dict | None = None\n"
     "    _vis1_sha12: dict | None = None\n")

edit(TR, "tr.ds_item",
     "            item[\"agent_zh_mask\"] = _t3[\"zh_mask\"][0]\n"
     "        return item\n",
     "            item[\"agent_zh_mask\"] = _t3[\"zh_mask\"][0]\n"
     "        # ---- refcv7 A9 R3: VIS-1, joined from the sidecar BY ROW AND TRACK ---- #\n"
     "        if self.vis1_sidecar is not None:\n"
     "            item.update(self._vis1_item(eid, int(f), item, int(n_raw), pad,\n"
     "                                        order if n_raw > pad else None))\n"
     "        return item\n")

edit(TR, "tr.ds_methods",
     "            \"n_clips\": int(getattr(join3d, \"n_clips\", 0))}\n"
     "        return self.join3d_stats\n",
     "            \"n_clips\": int(getattr(join3d, \"n_clips\", 0))}\n"
     "        return self.join3d_stats\n"
     "\n"
     "    # ---- refcv7 A9 R3: the VIS-1 sidecar ------------------------------------------ #\n"
     "    def enable_vis1(self, sidecar, *, split: str) -> dict:\n"
     "        \"\"\"Attach the VIS-1 sidecar so `_agent_item` emits `agent_vis_full/px/known`.\n"
     "\n"
     "        ⛔ REFUSES (never degrades) when the 3-D join or the clip table is absent, or when\n"
     "        ANY clip of THIS dataset is missing from the sidecar -- a missing clip would train\n"
     "        all its targets as IGNORE while config.json states VIS-1. A frame or row the sidecar\n"
     "        lacks refuses at `_agent_item` (`vis1.vis1_block_for_rows`).\n"
     "        \"\"\"\n"
     "        import hashlib as _hl\n"
     "        if self.agent_join is None or self.join3d is None or not self.map_clip_of_ep:\n"
     "            raise SystemExit(\n"
     "                \"[v3] ⛔ enable_vis1 needs the 2-D agent join, the 3-D join and the \"\n"
     "                \"manifest clip table: the sidecar is keyed by the join's rows and was \"\n"
     "                \"z-buffered over the 3-D cuboids.\")\n"
     "        eids = sorted({int(self.episodes[e_i].episode_id) for e_i, _ in self.index})\n"
     "        miss = [e for e in eids if e not in self.map_clip_of_ep]\n"
     "        if miss:\n"
     "            raise SystemExit(f\"[v3] ⛔ enable_vis1: {len(miss)} episodes have no clip id\")\n"
     "        self._vis1_sha12 = {e: _hl.sha256(str(self.map_clip_of_ep[e]).encode())\n"
     "                            .hexdigest()[:12] for e in eids}\n"
     "        sidecar.require_clips(self._vis1_sha12.values(), where=f\"{split} dataset\")\n"
     "        if int(sidecar.meta.get(\"n_stack\", -1)) != int(self.map_n_stack):\n"
     "            raise SystemExit(\n"
     "                f\"[v3] ⛔ the VIS-1 sidecar was built at n_stack \"\n"
     "                f\"{sidecar.meta.get('n_stack')} but this cache is {self.map_n_stack}: the \"\n"
     "                f\"3-D frame offset differs, so the z-buffered cuboids are not these rows.\")\n"
     "        self.vis1_sidecar = sidecar\n"
     "        self.vis1_stats = {\"split\": split, \"n_clips\": len(set(self._vis1_sha12.values())),\n"
     "                           \"sidecar_sha256\": sidecar.sha256,\n"
     "                           \"sidecar_n_clips\": sidecar.n_clips}\n"
     "        return self.vis1_stats\n"
     "\n"
     "    def _vis1_item(self, eid: int, f: int, item: dict, n_raw: int, pad: int,\n"
     "                   order) -> dict:\n"
     "        \"\"\"`agent_vis_full` / `agent_vis_px` (int32 pixel counts) and `agent_vis_known`\n"
     "        (bool) for ONE window, padded like the target block. ⛔ Every key is emitted for a\n"
     "        NO_LABEL window too (all unknown) -- `default_collate` needs one key set per batch.\"\"\"\n"
     "        nf = _np.zeros(int(pad), dtype=_np.int32)\n"
     "        nv = _np.zeros(int(pad), dtype=_np.int32)\n"
     "        kn = _np.zeros(int(pad), dtype=bool)\n"
     "        n = int(item[\"agent_valid\"].sum())\n"
     "        if bool(item[\"agent_label\"]) and n > 0:\n"
     "            tids = self.agent_join.lookup_track_ids(int(eid), int(f))\n"
     "            if tids is None:\n"
     "                raise SystemExit(\"[v3] ⛔ VIS-1: the 2-D join carries no track ids\")\n"
     "            tids = _np.asarray(list(tids), dtype=object)\n"
     "            if order is not None:\n"
     "                tids = tids[order]\n"
     "            nf, nv, kn = _vis1.vis1_block_for_rows(\n"
     "                self.vis1_sidecar, self._vis1_sha12[int(eid)], int(f),\n"
     "                box=item[\"agent_box\"][:n].numpy(), track_ids=list(tids[:n]),\n"
     "                zh_mask=item[\"agent_zh_mask\"][:n].numpy(), pad=int(pad), order=order)\n"
     "        return {\"agent_vis_full\": torch.from_numpy(nf), \"agent_vis_px\": torch.from_numpy(nv),\n"
     "                \"agent_vis_known\": torch.from_numpy(kn)}\n")

edit(TR, "tr.agent_select",
     "            tgt_ag = {k: v.index_select(0, sel) for k, v in tgt_ag.items()}\n"
     "            slots_ag = {k: (v.index_select(0, sel)\n"
     "                            if torch.is_tensor(v) and v.shape[:1] ==\n"
     "                            keep_ag.shape[:1] else v)\n"
     "                        for k, v in out[\"agent_slots\"].items()}\n"
     "        else:\n"
     "            slots_ag = out[\"agent_slots\"]\n",
     "            tgt_ag = {k: v.index_select(0, sel) for k, v in tgt_ag.items()}\n"
     "            # refcv7 A9 R2: `select_slots` is the comprehension this replaced, plus the\n"
     "            # per-layer \"aux\" list (a list passed through unselected would misalign rows).\n"
     "            slots_ag = _slot_presence.select_slots(out[\"agent_slots\"], sel,\n"
     "                                                   int(keep_ag.shape[0]))\n"
     "        else:\n"
     "            slots_ag = out[\"agent_slots\"]\n"
     "        # refcv7 A9 R3: the VIS-1 block, selected with the SAME label mask.\n"
     "        vis_ag = _vis1_batch_block(model, batch, device,\n"
     "                                   keep_ag if (keep_ag is not None and n_lab) else None)\n")

edit(TR, "tr.agent_call",
     "            cls_class_weight=getattr(model, \"_cls_class_weight\", None))\n"
     "        loss = loss + w_agent * ag[\"total\"]\n",
     "            cls_class_weight=getattr(model, \"_cls_class_weight\", None),\n"
     "            vis=vis_ag)\n"
     "        loss = loss + w_agent * ag[\"total\"]\n")

edit(TR, "tr.agent_extras",
     "        extra[\"agent_rows_no_cam\"] = float(ag[\"n\"][\"rows_no_cam\"])\n",
     "        extra[\"agent_rows_no_cam\"] = float(ag[\"n\"][\"rows_no_cam\"])\n"
     "        # ⭐⭐ refcv7 A9: the per-layer terms (R2), the VIS-1 counts (R3), the presence\n"
     "        # saturation read (G-LIVE) and, in EVAL mode, the P0 detection pack. Emitted only\n"
     "        # on the refined path, so a pre-A9 arm's log row is unchanged.\n"
     "        if \"_refine\" in ag:\n"
     "            for _kl, _vl in ag.items():\n"
     "                if _kl.startswith(\"loss_layer\") or _kl.startswith(\"loss_presence_layer\"):\n"
     "                    extra[\"agent_\" + _kl[5:]] = _vl\n"
     "            for _kn in (\"layers\", \"presence_exempt\", \"presence_matched\",\n"
     "                        \"vis1_n_positive\", \"vis1_n_ignore\", \"vis1_n_in_filter\",\n"
     "                        \"vis1_n_in_filter_unknown_vis\"):\n"
     "                if _kn in ag[\"n\"]:\n"
     "                    extra[f\"agent_n_{_kn}\"] = float(ag[\"n\"][_kn])\n"
     "            # G-LIVE-PRES (A10 §15.2): the fraction of slots with sigma >= 0.5, every row.\n"
     "            extra[\"agent_presence_frac_confident\"] = _slot_presence.presence_sanity(\n"
     "                slots_ag[\"presence_logit\"])[\"frac_confident\"]\n"
     "            if vis_ag is not None:\n"
     "                # LOGGING_SPEC_BOX §1 (the cheap census) every row; in EVAL mode the pack\n"
     "                # itself too, POOLED by the eval loop (never a mean of batch ratios).\n"
     "                _pk_ag = _det_metrics.window_packs(\n"
     "                    slots_ag, tgt_ag, vis_ag,\n"
     "                    presence_cost=(\"focal\" if str(core.agents.presence_loss) == \"focal\"\n"
     "                                   else \"sigmoid\"),\n"
     "                    match=ag.get(\"match\"), episode_ids=ep_ag,\n"
     "                    cls_weight=getattr(model, \"_cls_class_weight\", None))\n"
     "                extra.update(_det_metrics.train_row_keys(_pk_ag, \"agent\"))\n"
     "                if not model.training:\n"
     "                    extra[\"_det_pack_agent\"] = _pk_ag\n")

edit(TR, "tr.box3d_select",
     "                _s3 = {k: (v.index_select(0, _sel3)\n"
     "                           if torch.is_tensor(v)\n"
     "                           and v.shape[:1] == _b3_keep.shape[:1] else v)\n"
     "                       for k, v in _pout[\"box_slots\"].items()} \\\n"
     "                    if _b3_keep is not None else _pout[\"box_slots\"]\n",
     "                # refcv7 A9 R2: `select_slots` also narrows the per-layer \"aux\" list.\n"
     "                _s3 = _slot_presence.select_slots(_pout[\"box_slots\"], _sel3,\n"
     "                                                  int(_b3_keep.shape[0])) \\\n"
     "                    if _b3_keep is not None else _pout[\"box_slots\"]\n"
     "                _vis3 = _vis1_batch_block(model, batch, device, _b3_keep)\n")

edit(TR, "tr.box3d_call",
     "                    visible_filter=bool(getattr(model, \"_box3d_visible_filter\", True)))\n"
     "                loss = loss + _w_b3d * _brow[\"loss\"]\n",
     "                    visible_filter=bool(getattr(model, \"_box3d_visible_filter\", True)),\n"
     "                    presence_loss=str(_br.cfg.presence_loss), vis1=bool(_br.cfg.vis1),\n"
     "                    vis=_vis3)\n"
     "                loss = loss + _w_b3d * _brow[\"loss\"]\n"
     "                if _vis3 is not None:\n"
     "                    # ⭐ refcv7 A9 P0: LOGGING_SPEC_BOX §1 census every row; the pack itself in\n"
     "                    # EVAL mode (pooled by the eval loop). The Hungarian set is only needed there.\n"
     "                    _pk_b3 = _det_metrics.window_packs(\n"
     "                        _s3, _t3, _vis3,\n"
     "                        presence_cost=(\"focal\" if str(_br.cfg.presence_loss) == \"focal\"\n"
     "                                       else \"sigmoid\"),\n"
     "                        episode_ids=(batch[\"agent_ep\"].index_select(\n"
     "                            0, _sel3.to(batch[\"agent_ep\"].device))\n"
     "                            if \"agent_ep\" in batch else None),\n"
     "                        cls_weight=getattr(model, \"_cls_class_weight\", None),\n"
     "                        with_match=not model.training)\n"
     "                    extra.update(_det_metrics.train_row_keys(_pk_b3, \"box3d\"))\n"
     "                    if not model.training:\n"
     "                        extra[\"_det_pack_box3d\"] = _pk_b3\n")

edit(TR, "tr.model_attrs",
     "    model._box3d_visible_filter = bool(getattr(args, \"box3d_visible_filter\", True))\n",
     "    model._box3d_visible_filter = bool(getattr(args, \"box3d_visible_filter\", True))\n"
     "    # ⭐⭐ refcv7 A9 R3: VIS-1 travels ON THE MODEL for the same reason -- the loss-time\n"
     "    # batch block (`_vis1_batch_block`) has no `args`. G-DVB reads it back.\n"
     "    model._vis1 = bool(getattr(args, \"slot_vis1\", False))\n")

edit(TR, "tr.pcfg_refine",
     "        model._perception = _perc.build_perception_branch(model, _pcfg).to(device)\n",
     "        # ⭐⭐ refcv7 A9: the box head's four refinement fields, from the SAME mapping the\n"
     "        # agent head uses (`_slot_refine_kwargs`) -- a declared frozen-dataclass replace.\n"
     "        _pcfg = _dc.replace(_pcfg, **_slot_refine_kwargs(args))\n"
     "        model._perception = _perc.build_perception_branch(model, _pcfg).to(device)\n")

edit(TR, "tr.stats_init",
     "    join3d_stats = eval_join3d_stats = None\n",
     "    join3d_stats = eval_join3d_stats = None\n"
     "    vis1_stamp = vis1_stats = eval_vis1_stats = _vis1_sc = None     # refcv7 A9 R3\n"
     "    calib_dl = calib_stamp = None        # refcv7 A10 §15.3: the INFORMATIVE TRAIN P = R gate\n")

edit(TR, "tr.enable_vis1_train",
     "        print(\"[v3] 3-D cuboid join: %s\" % join3d_stats, flush=True)\n",
     "        print(\"[v3] 3-D cuboid join: %s\" % join3d_stats, flush=True)\n"
     "    # ---- refcv7 A9 R3: the VIS-1 sidecar (refused, never degraded) ---------------- #\n"
     "    if getattr(args, \"slot_vis1\", False):\n"
     "        _vis1_sc = _vis1.VIS1Sidecar(args.vis1_sidecar)\n"
     "        vis1_stamp = _vis1_sc.stamp()\n"
     "        vis1_stats = ds.enable_vis1(_vis1_sc, split=\"train\")\n"
     "        print(\"[v3] VIS-1 sidecar: %s sha256 %s (%d clips, %d frames, %d rows)\"\n"
     "              % (Path(args.vis1_sidecar).name, _vis1_sc.sha256[:16], _vis1_sc.n_clips,\n"
     "                 _vis1_sc.n_frames, _vis1_sc.n_rows), flush=True)\n"
     "        # A10 §15.3 (INFORMATIVE): the FIXED TRAIN calibration windows (the audit's 256 / 64\n"
     "        # clips), banked by sha12 + window start. Missing windows are COUNTED, never refused:\n"
     "        # the readout is informative and must not decide whether a run can train.\n"
     "        _cal_pos, _cal_miss = _det_metrics.calib_indices(\n"
     "            ds.index, lambda _e: ds._vis1_sha12[int(ds.episodes[_e].episode_id)])\n"
     "        calib_stamp = {\"artifact\": _det_metrics.CALIB_WINDOWS_FILE,\n"
     "                       \"windows_sha256\": _det_metrics.load_calib_windows()[\"windows_sha256\"],\n"
     "                       \"n_listed\": len(_cal_pos) + int(_cal_miss), \"n_found\": len(_cal_pos),\n"
     "                       \"role\": \"INFORMATIVE P = R gate only (A10 §15.3)\"}\n"
     "        if _cal_pos:\n"
     "            calib_dl = torch.utils.data.DataLoader(\n"
     "                torch.utils.data.Subset(ds, _cal_pos), batch_size=args.batch,\n"
     "                shuffle=False, num_workers=0, drop_last=False)\n"
     "        print(\"[v3] P0 calibration set: %d of %d TRAIN windows found\"\n"
     "              % (len(_cal_pos), len(_cal_pos) + int(_cal_miss)), flush=True)\n")

edit(TR, "tr.enable_vis1_eval",
     "                    args.join3d, len(set(_e_clip.values())), split=\"eval\"))\n",
     "                    args.join3d, len(set(_e_clip.values())), split=\"eval\"))\n"
     "        if getattr(args, \"slot_vis1\", False) and _vis1_sc is not None:\n"
     "            eval_vis1_stats = e_ds.enable_vis1(_vis1_sc, split=\"eval\")\n")

edit(TR, "tr.stamp",
     "        \"agent_join_stats\": ({\"train\": agent_stats, \"eval\": eval_agent_stats}\n"
     "                             if agent_stats is not None else None),\n",
     "        \"agent_join_stats\": ({\"train\": agent_stats, \"eval\": eval_agent_stats}\n"
     "                             if agent_stats is not None else None),\n"
     "        # ⭐⭐ refcv7 A9: the refined slot heads AS BUILT, and the VIS-1 sidecar BY DIGEST.\n"
     "        # ⛔ `None` for a pre-A9 arm, so its record differs by nothing a comparison acts on.\n"
     "        \"slot_refine\": _slot_refine_block(args, model),\n"
     "        \"vis1\": ({\"sidecar\": vis1_stamp, \"rule\": _vis1.vis1_rule_dict(),\n"
     "                  \"train\": vis1_stats, \"eval\": eval_vis1_stats, \"calib\": calib_stamp}\n"
     "                 if vis1_stamp is not None else None),\n")

edit(TR, "tr.eval_acc_init",
     "            acc, nb_e, eval_err = {}, 0, None\n",
     "            acc, nb_e, eval_err = {}, 0, None\n"
     "            det_packs = {}          # refcv7 A9 P0: pooled per-window detection packs\n")

edit(TR, "tr.eval_acc_packs",
     "                        nb_e += 1\n",
     "                        for _hd in _det_metrics.HEADS:\n"
     "                            if el.get(f\"_det_pack_{_hd}\"):\n"
     "                                det_packs.setdefault(_hd, []).extend(\n"
     "                                    el[f\"_det_pack_{_hd}\"])\n"
     "                        nb_e += 1\n")

edit(TR, "tr.eval_row",
     "                erow.update(step=step, eval_batches=nb_e,\n",
     "                # ⭐⭐ refcv7 A9 P0: detection is MEASURED, pooled over the eval windows (AP is a\n"
     "                # joint ranking, never a batch mean). NaN (undefined) is written as null.\n"
     "                for _hd, _pk in det_packs.items():\n"
     "                    for _dk, _dv in _det_metrics.summarise(_pk, _hd).items():\n"
     "                        erow[_dk] = (None if (isinstance(_dv, float) and _dv != _dv)\n"
     "                                     else round(float(_dv), 5))\n"
     "                    if erow.get(f\"eval_{_hd}_conf_ratio_alarm\") == 1.0:\n"
     "                        print(\"[v3:eval] ALARM %s conf_ratio %s outside [0.5, 1.5] (A10 \"\n"
     "                              \"§15.3, a Watch alarm, not a stop)\"\n"
     "                              % (_hd, erow.get(f\"eval_{_hd}_conf_ratio\")), flush=True)\n"
     "                # A10 §15.3 INFORMATIVE: the P = R gate on the FIXED TRAIN calibration windows, in\n"
     "                # eval mode with the RNG isolated (the W-BOOTSTRAP discipline). It can never take\n"
     "                # the run down: a failure is stamped into the row and training continues.\n"
     "                if calib_dl is not None:\n"
     "                    _cpk = {}\n"
     "                    try:\n"
     "                        model.eval()\n"
     "                        with _RngIsolated(device, None), torch.no_grad():\n"
     "                            for _cb in calib_dl:\n"
     "                                _cl = compute_losses_v3(\n"
     "                                    model, _cb, device, mode=args.mode,\n"
     "                                    ablate_frames=args.ablate_frames)\n"
     "                                for _hd in _det_metrics.HEADS:\n"
     "                                    if _cl.get(f\"_det_pack_{_hd}\"):\n"
     "                                        _cpk.setdefault(_hd, []).extend(\n"
     "                                            _cl[f\"_det_pack_{_hd}\"])\n"
     "                        for _hd, _pk in _cpk.items():\n"
     "                            for _dk, _dv in _det_metrics.calib_keys(_pk, _hd).items():\n"
     "                                erow[_dk] = (None if (isinstance(_dv, float) and _dv != _dv)\n"
     "                                             else round(float(_dv), 5))\n"
     "                    except Exception as _cexc:          # noqa: BLE001 (informative)\n"
     "                        import traceback as _tb\n"
     "                        _tb.print_exc()\n"
     "                        erow[\"eval_calib_error\"] = f\"{type(_cexc).__name__}: {_cexc}\"\n"
     "                    finally:\n"
     "                        model.train()\n"
     "                erow.update(step=step, eval_batches=nb_e,\n")

edit(TR, "tr.flags",
     "    g5.add_argument(\"--agent-presence-hard\", action=\"store_true\",\n",
     "    # ---- refcv7 A9 (SPEC_REFCV7 §14): the REFINED slot heads, BOTH of them ---------- #\n"
     "    g5.add_argument(\"--slot-presence-loss\", default=\"bce\", choices=[\"bce\", \"focal\"],\n"
     "                    help=\"refcv7 A9 R1, for BOTH slot heads (box3d + agent). 'focal' = sigmoid \"\n"
     "                         \"focal presence (alpha 0.25, gamma 2, weight 2.0, / n matched GT) AND \"\n"
     "                         \"the focal matching cost -- the 5/5 nuScenes camera-head recipe. 'bce' \"\n"
     "                         \"(default) = the pre-A9 BCE with NO_OBJECT_W 0.1, bit-identical.\")\n"
     "    g5.add_argument(\"--slot-presence-prior\", type=float, default=0.05,\n"
     "                    help=\"refcv7 A9 R1: the presence logit's INIT prior for both slot heads \"\n"
     "                         \"(A9: 0.01; default 0.05 = the pre-A9 AgentSlotDecoder.PRESENCE_PRIOR).\")\n"
     "    g5.add_argument(\"--slot-deep-supervision\", action=\"store_true\",\n"
     "                    help=\"refcv7 A9 R2: supervise EVERY decoder layer of both slot heads \"\n"
     "                         \"(shared norm+head, re-matched per layer, summed; 0 parameters). \"\n"
     "                         \"Inference reads the last layer.\")\n"
     "    g5.add_argument(\"--slot-vis1\", action=\"store_true\",\n"
     "                    help=\"refcv7 A9 R3: VIS-1 targets with IGNORE semantics for both slot heads \"\n"
     "                         \"(POSITIVE = in field AND vis_frac >= 0.30 AND >= 100 px; every other \"\n"
     "                         \"real object IGNORE). Needs --vis1-sidecar, --agent-join, --join3d. \"\n"
     "                         \"Also turns on the P0 detection metrics in the in-run eval.\")\n"
     "    g5.add_argument(\"--vis1-sidecar\", default=None,\n"
     "                    help=\"refcv7 A9 R3: the VIS-1 sidecar (.npz) from \"\n"
     "                         \"scripts/precompute_vis1_sidecar.py, covering EVERY clip of the train \"\n"
     "                         \"and eval caches; its sha256 is stamped in config.json[vis1].\")\n"
     "    g5.add_argument(\"--agent-presence-hard\", action=\"store_true\",\n")

edit(TR, "tr.queries_help",
     "                    help=\"detection queries. 100, ruled by the Master Mind \"\n",
     "                    help=\"detection queries. 300 since SPEC_REFCV7 A9 R4 (>= 2x the \"\n"
     "                         \"120-target max on the refcv6 corpus line); before that 100, \"\n"
     "                         \"ruled by the Master Mind \"\n")

# ===================================================================================================== #
# declared_vs_built.py -- G-DVB entries for the 5 new flags                                             #
# ===================================================================================================== #
edit(DVB, "dvb.entries",
     "_b(\"w_map\", _c_perception)\n",
     "# -- refcv7 A9: the refined slot heads (R1-R3). Each reads BOTH built slot heads -- the\n"
     "#    learned agent head (`core.agent_head`) and the box decoder (`_perception.box_dec`) --\n"
     "#    and the loss config each head's loss reads. A flag set with NO learned head is itself\n"
     "#    a mismatch (declared, built nowhere).\n"
     "def _slot_heads(m):\n"
     "    out = []\n"
     "    ah = getattr(_core(m), \"agent_head\", None)\n"
     "    if ah is not None and hasattr(ah, \"deep_supervision\"):      # learned, not the oracle\n"
     "        out.append((\"agent\", ah, getattr(getattr(_core(m), \"cfg\", None), \"agents\", None)))\n"
     "    br = getattr(m, \"_perception\", None)\n"
     "    bd = getattr(br, \"box_dec\", None) if br is not None else None\n"
     "    if bd is not None:\n"
     "        out.append((\"box3d\", bd, getattr(br, \"cfg\", None)))\n"
     "    return out\n"
     "\n"
     "\n"
     "def _c_slot(dest, default, read, where, tol=None):\n"
     "    def chk(m, a):\n"
     "        want = _a(a, dest, default)\n"
     "        want = type(default)(want) if want is not None else default\n"
     "        heads = _slot_heads(m)\n"
     "        out = []\n"
     "        if not heads and want != default:\n"
     "            out.append(Mismatch(_flag(dest), want, \"no learned slot head built\",\n"
     "                                \"core.agent_head / _perception.box_dec\",\n"
     "                                \"a refinement declared on a model that built no slot head\"))\n"
     "        for name, dec, lcfg in heads:\n"
     "            got = read(dec, lcfg)\n"
     "            if tol is not None:\n"
     "                out += _near(dest, float(want), got, f\"{name}: {where}\", tol=tol)\n"
     "            else:\n"
     "                out += _eq(dest, want, got, f\"{name}: {where}\")\n"
     "        return out\n"
     "    return chk\n"
     "\n"
     "\n"
     "def _c_slot_vis1(m, a):\n"
     "    want = bool(_a(a, \"slot_vis1\", False))\n"
     "    out = _eq(\"slot_vis1\", want, bool(getattr(m, \"_vis1\", False)),\n"
     "              \"model._vis1 (read by the loss-time batch block)\")\n"
     "    return out + _c_slot(\"slot_vis1\", False, lambda d, c: bool(getattr(c, \"vis1\", False)),\n"
     "                         \"loss config .vis1 (read by the loss)\")(m, a)\n"
     "\n"
     "\n"
     "_b(\"slot_presence_loss\", _c_slot(\"slot_presence_loss\", \"bce\",\n"
     "                                  lambda d, c: str(getattr(c, \"presence_loss\", \"<absent>\")),\n"
     "                                  \"loss config .presence_loss (read by the loss)\"))\n"
     "_b(\"slot_presence_prior\", _c_slot(\"slot_presence_prior\", 0.05,\n"
     "                                   lambda d, c: getattr(d, \"presence_prior\", None),\n"
     "                                   \"decoder.presence_prior (its init bias)\", tol=1e-12))\n"
     "_b(\"slot_deep_supervision\", _c_slot(\"slot_deep_supervision\", False,\n"
     "                                     lambda d, c: bool(getattr(d, \"deep_supervision\", False)),\n"
     "                                     \"decoder.deep_supervision (read by the forward)\"))\n"
     "_b(\"slot_vis1\", _c_slot_vis1)\n"
     "register(\"vis1_sidecar\", \"data\", reason=(\n"
     "    \"the VIS-1 visibility sidecar (refcv7 A9 R3); its sha256 is stamped in config.json[vis1] \"\n"
     "    \"and the dataset REFUSES a missing clip, frame, row or a mismatched track/centre\"))\n"
     "_b(\"w_map\", _c_perception)\n")

edit(DVB_T, "dvbt.count", "__COUNT__", "__COUNT__")          # handled by the relative bump below

# ===================================================================================================== #
# tests pinned at 100 -> the ONE spelling                                                               #
# ===================================================================================================== #
edit(V6_T, "v6t.params",
     "PROD_SLOT_PARAMS = 3_228_949\n",
     "#: ⭐ refcv7 A9 R4 (2026-09-27): N_QUERIES_DEFAULT 100 -> 300, so the probe moves by EXACTLY\n"
     "#: 200 x 256 = 51,200 (3,228,949 at 100 -> 3,280,149 at 300): only the query table scales.\n"
     "PROD_SLOT_PARAMS = 3_280_149\n")
edit(V6_T, "v6t.default1",
     "    assert V6Config(tac_vocab_version=\"v6.0\").n_slot_queries \\\n"
     "        == N_QUERIES_DEFAULT == 100\n",
     "    assert V6Config(tac_vocab_version=\"v6.0\").n_slot_queries \\\n"
     "        == N_QUERIES_DEFAULT == 300          # re-ruled by SPEC_REFCV7 A9 R4\n")
edit(V6_T, "v6t.default2",
     "    assert tv6.build_parser().get_default(\"n_slot_queries\") \\\n"
     "        == N_QUERIES_DEFAULT == 100\n",
     "    assert tv6.build_parser().get_default(\"n_slot_queries\") \\\n"
     "        == N_QUERIES_DEFAULT == 300          # re-ruled by SPEC_REFCV7 A9 R4\n")
edit(SF_T, "sft.default",
     "    assert A.N_QUERIES_DEFAULT == 100\n",
     "    assert A.N_QUERIES_DEFAULT == 300          # re-ruled by SPEC_REFCV7 A9 R4\n")
edit(PROV_T, "provt.default",
     "    assert t.AGENT_QUERIES_DEFAULT == 100\n"
     "    assert t.build_parser().get_default(\"agent_queries\") == 100\n",
     "    # re-ruled by SPEC_REFCV7 A9 R4: the ONE spelling is agent_slots.N_QUERIES_DEFAULT (300)\n"
     "    from tanitad.models.agent_slots import N_QUERIES_DEFAULT\n"
     "    assert t.AGENT_QUERIES_DEFAULT is N_QUERIES_DEFAULT or \\\n"
     "        t.AGENT_QUERIES_DEFAULT == N_QUERIES_DEFAULT == 300\n"
     "    assert t.build_parser().get_default(\"agent_queries\") == 300\n")
edit(PROV_T, "provt.cfg",
     "    assert cfg.core.agents.queries == 100\n",
     "    assert cfg.core.agents.queries == 300          # SPEC_REFCV7 A9 R4\n")

edit(OCC_T, "occt.agent_keys",
     "AGENT_SEAM_STAMP_KEYS = {\n",
     "AGENT_SEAM_STAMP_KEYS = {\n"
     "    # refcv7 A9 (SPEC_REFCV7 §14): the refined head's four DECLARED knobs, added on purpose\n"
     "    \"presence_loss\", \"presence_prior\", \"deep_supervision\", \"vis1\",\n")
edit(OCC_T, "occt.perc_keys",
     "PERCEPTION_STAMP_KEYS = {\n",
     "PERCEPTION_STAMP_KEYS = {\n"
     "    # refcv7 A9 (SPEC_REFCV7 §14): the refined box head's four DECLARED knobs, added on purpose\n"
     "    \"presence_loss\", \"presence_prior\", \"deep_supervision\", \"vis1\",\n")

# ===================================================================================================== #
# taniteval/tools/refcv3_arm.py -- a PRE-A9 record rebuilds at ITS OWN query count                     #
# ===================================================================================================== #
edit(ARM, "arm.queries",
     "        cfg = tr._pin_trainer_cfg(base, args)\n",
     "        # ⛔⛔ refcv7 A9 R4 moved the ONE query spelling 100 -> 300. A record whose argv never\n"
     "        # passed --agent-queries (refcv6-r101-s0 included) is rebuilt at its STAMPED count --\n"
     "        # the parser default would build a head its checkpoint does not fit.\n"
     "        _aq = getattr(tr, \"agent_queries_as_trained\", None)\n"
     "        if _aq is not None and str(getattr(args, \"agents\", \"off\")) != \"off\":\n"
     "            _n_q, _why_q = _aq(config, args)\n"
     "            if _n_q is not None and int(_n_q) != int(args.agent_queries):\n"
     "                args.agent_queries = int(_n_q)\n"
     "                src_q = f\"; agent_queries -> {int(_n_q)} ({_why_q})\"\n"
     "            else:\n"
     "                src_q = \"\"\n"
     "        else:\n"
     "            src_q = \"\"\n"
     "        cfg = tr._pin_trainer_cfg(base, args)\n")
edit(ARM, "arm.queries_src",
     "        src = (\"config.json[argv] -> refc_v3_train.build_parser + \"\n"
     "               \"_pin_trainer_cfg (the trainer's own build path)\")\n",
     "        src = (\"config.json[argv] -> refc_v3_train.build_parser + \"\n"
     "               \"_pin_trainer_cfg (the trainer's own build path)\") + src_q\n")
edit(ARM, "arm.perc_fields",
     "    for k in (\"d_bev\", \"n_queries\", \"d_model\", \"stride\", \"bev_source\"):\n",
     "    # ⭐ refcv7 A9: the refined box head's four fields rebuild from the stamp as well (absent in a\n"
     "    # pre-A9 stamp -> the pre-A9 defaults, which is what that run built).\n"
     "    for k in (\"d_bev\", \"n_queries\", \"d_model\", \"stride\", \"bev_source\", \"presence_loss\",\n"
     "              \"presence_prior\", \"deep_supervision\", \"vis1\"):\n")


# ===================================================================================================== #
# test_refc_v3_save_before_eval.py -- the step loop's held-out passes: A10 15.3 adds the calibration pass  #
# ===================================================================================================== #
edit(SAVE_T, "savet.passes",
     "    # ⚠️ TWO held-out passes live in the loop since W-BOOTSTRAP (2026-09-19): the\n"
     "    # aggregate eval and the per-window dump, and EACH must enter eval mode itself.\n"
     "    # The dump once ran WITHOUT its own `model.eval()` -- i.e. in TRAIN mode, right\n"
     "    # after the aggregate's `model.train()` -- which is what turned the old `== 1`\n"
     "    # pin red at HEAD. Behaviour: tests/test_eval_window_dump_mode.py.\n"
     "    assert loop.count(\"model.eval()\") == 2 and loop.count(\"model.train()\") == 2\n",
     "    # ⚠️ THREE held-out passes live in the loop: since W-BOOTSTRAP (2026-09-19) the\n"
     "    # aggregate eval and the per-window dump, and since refcv7 A10 15.3 the INFORMATIVE\n"
     "    # calibration pass on the fixed TRAIN windows (inside the aggregate's success branch;\n"
     "    # it restores train mode in its own `finally`). EACH must enter eval mode itself.\n"
     "    # The dump once ran WITHOUT its own `model.eval()` -- i.e. in TRAIN mode, right\n"
     "    # after the aggregate's `model.train()` -- which is what turned the old `== 1`\n"
     "    # pin red at HEAD. Behaviour: tests/test_eval_window_dump_mode.py.\n"
     "    assert loop.count(\"model.eval()\") == 3 and loop.count(\"model.train()\") == 3\n"
     "    i_cal = loop.index(\"if calib_dl is not None:\")\n"
     "    assert (loop.index(\"model.eval()\", i_cal)\n"
     "            < loop.index(\"compute_losses_v3(\", i_cal)), (\n"
     "        \"the calibration pass forwards before entering eval mode\")\n")


# ===================================================================================================== #
# the applier                                                                                           #
# ===================================================================================================== #
def _read_blob(git_dir: str, rev: str, path: str) -> bytes:
    return subprocess.run(["git", f"--git-dir={git_dir}", "show", f"{rev}:{path}"],
                          check=True, capture_output=True).stdout


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def bump_dvb_count(text: str) -> tuple[str, str]:
    """``assert len(dvb.REGISTRY) == K`` -> ``K + N_NEW_DVB`` (RELATIVE to the base, so an earlier batch's
    own bump is kept)."""
    pat = re.compile(r"(    assert len\(dvb\.REGISTRY\) == )(\d+)([^\n]*)\n")
    m = list(pat.finditer(text))
    if len(m) != 1:
        raise SystemExit(f"ANCHOR FAILED {DVB_T} [dvbt.count]: found {len(m)} registry-count pins, need 1")
    k = int(m[0].group(2))
    new = (f"{m[0].group(1)}{k + N_NEW_DVB}{m[0].group(3)}\n"
           f"    # refcv7 A9 (box head): +{N_NEW_DVB} = --slot-presence-loss, --slot-presence-prior,\n"
           f"    # --slot-deep-supervision, --slot-vis1, --vis1-sidecar ({k} -> {k + N_NEW_DVB})\n")
    return text[:m[0].start()] + new + text[m[0].end():], f"{k} -> {k + N_NEW_DVB}"


def apply(base: bytes, path: str) -> tuple[bytes, list]:
    crlf = base.count(b"\r\n")
    lf = base.count(b"\n")
    if crlf and crlf != lf:
        raise SystemExit(f"{path}: MIXED line endings ({crlf} CRLF of {lf}) -- refusing")
    eol = "\r\n" if crlf else "\n"
    text = base.decode("utf-8").replace("\r\n", "\n")
    done = []
    for f, eid, old, new in E:
        if f != path:
            continue
        if path == DVB_T and eid == "dvbt.count":
            text, how = bump_dvb_count(text)
            done.append(f"{eid} ({how})")
            continue
        want = COUNTS.get((f, eid), 1)
        got = text.count(old)
        if got != want:
            raise SystemExit(f"ANCHOR FAILED {path} [{eid}]: found {got}, need {want}:\n{old[:400]!r}")
        text = text.replace(old, new)
        done.append(eid)
    return text.replace("\n", eol).encode("utf-8"), done


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base-git")
    g.add_argument("--base-dir")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = {"rev": a.rev, "n_new_dvb": N_NEW_DVB, "files": {}}
    for path in FILES:
        if a.base_git:
            if not a.rev:
                raise SystemExit("--rev is required with --base-git")
            base = _read_blob(a.base_git, a.rev, path)
        else:
            base = open(os.path.join(a.base_dir, path), "rb").read()
        new, done = apply(base, path)
        dst = os.path.join(a.out, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(new)
        man["files"][path] = {"base_blob_sha1": git_blob_sha1(base), "out_blob_sha1": git_blob_sha1(new),
                              "base_lines": base.count(b"\n"), "out_lines": new.count(b"\n"),
                              "eol": "CRLF" if base.count(b"\r\n") else "LF", "edits": done}
        print(f"{path}: {len(done)} edits, {base.count(b'\n')} -> {new.count(b'\n')} lines")
    with open(os.path.join(a.out, "EDIT_MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
