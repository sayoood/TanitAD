#!/usr/bin/env python3
"""refcv7 NEW-1 -- the SHARED-FILE edits, as ANCHORED replacements.

Why a script and not three hand-edited files: the fixes agent is editing the
same three files (FIX-3/4/5 + config hygiene) and the brief orders these edits
to be re-based onto the tip AFTER that batch lands. An anchored patch re-applies
mechanically, and every anchor must occur EXACTLY ONCE in the base or the run
FAILS LOUDLY naming it -- a moved anchor is a re-base conflict to resolve by
reading, never by guessing.

usage:
  python apply_residual_prior_edits.py --base-git <GIT_DIR> --rev <commit> \
         --out <dir>        # reads each file's RAW BLOB at <rev> (git show)
  python apply_residual_prior_edits.py --base-dir <tree> --out <dir>

Each output file keeps its base blob's line endings byte for byte (refc.py and
refc_v3_train.py are CRLF blobs, refc_v3.py is LF -- MEASURED at 59f0d46 /
ea9b02d; ddv2_refc_chain.py is CRLF, test_ddv2_refc_chain.py and
channel_admissibility.py are LF -- MEASURED at 12953d2). A manifest
`EDIT_MANIFEST.json` records every base blob sha1, output sha1 and the edit ids
applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

REFC = "stack/tanitad/refs/refc.py"
REFCV3 = "stack/tanitad/refs/refc_v3.py"
TRAIN = "stack/scripts/refc_v3_train.py"

E: list[tuple[str, str, str, str]] = []   # (file, edit_id, old, new)


def edit(path: str, eid: str, old: str, new: str) -> None:
    E.append((path, eid, old, new))


# =========================================================================== #
# refc.py                                                                     #
# =========================================================================== #
edit(REFC, "refc.import",
     "from tanitad.refs import refc_sampler as rs\n",
     "from tanitad.refs import refc_sampler as rs\n"
     "# refcv7 NEW-1: the causal kinematic prior and THE ONE function that turns a\n"
     "# residual into the plan (`kinematic_prior.roll_plan`).\n"
     "from tanitad.models import kinematic_prior as _kp\n")

edit(REFC, "refc.decoder_config_field",
     "    bev_coupling_d_bev: int = 96\n",
     "    bev_coupling_d_bev: int = 96\n"
     "    # ---- refcv7 NEW-1: THE PLAN IS A RESIDUAL ON A CAUSAL KINEMATIC PRIOR --\n"
     "    # A DECLARED field (the refcv5 false-provenance rule; after the config-\n"
     "    # hygiene fix an undeclared attribute is REFUSED at assignment). \"off\"\n"
     "    # (the DEFAULT) reaches no new code, so refcv6 builds and runs bit for\n"
     "    # bit. Any other value (`kinematic_prior.RESIDUAL_PRIOR_MODES`) makes the\n"
     "    # sampler's state and the anchor vocabulary RESIDUALS on the prior P, and\n"
     "    # `kinematic_prior.roll_plan` the one function that turns a residual into\n"
     "    # a path. SPEC_REFCV7 section 1, NEW-1.\n"
     "    residual_prior: str = \"off\"\n")

edit(REFC, "refc.decoder_init_validate",
     "            self.sched = rs.DDIMSchedule()\n        # ---- refcv6 ",
     "            self.sched = rs.DDIMSchedule()\n"
     "        # ---- refcv7 NEW-1: the residual prior, validated at BUILD --------- #\n"
     "        # Refused rather than half-built: each mechanism the prior needs is\n"
     "        # named, and a build lacking one would stamp NEW-1 on a decoder that\n"
     "        # cannot compose it. `off` constructs nothing and draws no RNG.\n"
     "        self.residual_prior = _kp.check_mode(\n"
     "            str(getattr(cfg, \"residual_prior\", _kp.RESIDUAL_PRIOR_OFF)))\n"
     "        if self.residual_prior != _kp.RESIDUAL_PRIOR_OFF:\n"
     "            if self.control_head is None:\n"
     "                raise ValueError(\n"
     "                    f\"refcv7 NEW-1: residual_prior={self.residual_prior!r} \"\n"
     "                    \"needs the control-space DDIM sampler (sampler='ddim'): \"\n"
     "                    \"the residual IS the sampler's state. A classifier \"\n"
     "                    \"build has no state to compose on.\")\n"
     "            if space != \"control\":\n"
     "                raise ValueError(\n"
     "                    \"refcv7 NEW-1: the residual prior composes CONTROLS; the \"\n"
     "                    \"metre-space sampler (the DD-literal regression arm) has \"\n"
     "                    \"no control state, so P + Delta would be undefined.\")\n"
     "            if not self.anchor_v0_cond:\n"
     "                raise ValueError(\n"
     "                    \"refcv7 NEW-1: the prior is rolled per window from the \"\n"
     "                    \"measured v0 and the residual vocabulary with it; a \"\n"
     "                    \"FIXED-path bank carries no controls to compose on. \"\n"
     "                    \"Build with --anchor-v0-conditioned.\")\n"
     "        # ---- refcv6 ")

edit(REFC, "refc.roll_bank_signature",
     "    def roll_bank(self, v_ms: Tensor | None, ego_keep: Tensor | None,\n"
     "                  batch: int, dtype: torch.dtype,\n"
     "                  withheld_speed: Tensor | None = None) -> Tensor:\n",
     "    def roll_bank(self, v_ms: Tensor | None, ego_keep: Tensor | None,\n"
     "                  batch: int, dtype: torch.dtype,\n"
     "                  withheld_speed: Tensor | None = None,\n"
     "                  prior: \"tuple | None\" = None) -> Tensor:\n")

edit(REFC, "refc.roll_bank_body",
     "        dropout regime is genuinely speed-blind.\n"
     "        \"\"\"\n"
     "        n = self.anchors.shape[0]\n"
     "        if not self.anchor_v0_cond:\n",
     "        dropout regime is genuinely speed-blind.\n"
     "\n"
     "        ``prior`` (refcv7 NEW-1) = ``(a0 [B], kappa0 [B], v [B])`` from\n"
     "        :meth:`_residual_prior`: the vocabulary's controls are RESIDUALS on\n"
     "        it and the bank is ``kinematic_prior.roll_plan`` of them, so the\n"
     "        anchor ``(0, 0)`` rolls to ``P`` exactly. REQUIRED on a residual\n"
     "        build and REFUSED on an off build (:meth:`_roll_residual_bank`).\n"
     "        \"\"\"\n"
     "        n = self.anchors.shape[0]\n"
     "        if self.residual_prior != _kp.RESIDUAL_PRIOR_OFF or prior is not None:\n"
     "            return self._roll_residual_bank(prior, batch, dtype)\n"
     "        if not self.anchor_v0_cond:\n")

edit(REFC, "refc.residual_methods",
     "    def _feasible(self, x: Tensor, v_ms: Tensor | None) -> Tensor:\n",
     "    # ---- refcv7 NEW-1: the residual prior's four decoder-side seams ------ #\n"
     "    def _prior_speed(self, v_ms: Tensor, ego_keep: Tensor | None,\n"
     "                     withheld_speed: Tensor | None) -> Tensor:\n"
     "        \"\"\"[B] the ONE speed this forward rolls every row from.\n"
     "\n"
     "        (( !! )) The expression is :meth:`roll_bank`'s own (kept rows at\n"
     "        ``v_ms``, withheld rows at :meth:`_withheld_ref_speed`), computed\n"
     "        ONCE per forward and handed to the bank, the prior AND the sampled\n"
     "        fan. The refcv6 sampler rolls its fan from the RAW ``v_ms`` while\n"
     "        the bank uses the withheld speed (``refc.py`` ``_sample``), so on a\n"
     "        withheld training row the two differ; a residual composed on a\n"
     "        prior rolled at one speed into a fan rolled at another is not\n"
     "        ``P + Delta``. Pinned bit-equal to the legacy bank by\n"
     "        ``test_residual_prior.py`` (a ZERO prior reproduces it).\n"
     "        \"\"\"\n"
     "        v = v_ms.reshape(-1).to(torch.float32)\n"
     "        if ego_keep is not None:\n"
     "            v = torch.where(ego_keep.reshape(-1),\n"
     "                            v, self._withheld_ref_speed(v, withheld_speed))\n"
     "        return v\n"
     "\n"
     "    def _residual_prior(self, residual_prior, v_ms: Tensor | None,\n"
     "                        ego_keep: Tensor | None, batch: int,\n"
     "                        withheld_speed: Tensor | None) -> \"tuple | None\":\n"
     "        \"\"\"``(a0, kappa0)`` from the model -> ``(a0, kappa0, v)`` as this\n"
     "        forward composes it, or ``None`` on an off build.\n"
     "\n"
     "        Withheld rows get the no-information prior (zero controls at the\n"
     "        bank's withheld speed, ``kinematic_prior.withhold``); a forward whose\n"
     "        ``v0`` was not supplied at all (``v_ms is None``) is withheld on\n"
     "        every row. Both directions of the input REFUSE.\n"
     "        \"\"\"\n"
     "        if self.residual_prior == _kp.RESIDUAL_PRIOR_OFF:\n"
     "            if residual_prior is not None:\n"
     "                raise ValueError(\n"
     "                    \"refcv7 NEW-1: a residual prior was handed to a decoder \"\n"
     "                    \"built with residual_prior='off'; it would be silently \"\n"
     "                    \"ignored and the record would not say so.\")\n"
     "            return None\n"
     "        if residual_prior is None:\n"
     "            raise ValueError(\n"
     "                f\"refcv7 NEW-1: this decoder was built with residual_prior=\"\n"
     "                f\"{self.residual_prior!r} but no prior reached the forward. \"\n"
     "                f\"Its vocabulary and its sampler state are RESIDUALS, so \"\n"
     "                f\"without P every emitted path would be Delta presented as \"\n"
     "                f\"the plan. Refusing.\")\n"
     "        if self.anchor_withheld_bank == \"none\":\n"
     "            raise ValueError(\n"
     "                \"refcv7 NEW-1: anchor_withheld_bank='none' rolls EVERY row at \"\n"
     "                \"the reference speed (a speed-blind vocabulary); a prior \"\n"
     "                \"built from the measured state contradicts it. Refusing.\")\n"
     "        a0, k0 = residual_prior\n"
     "        a0 = a0.reshape(-1).to(torch.float32)\n"
     "        k0 = k0.reshape(-1).to(torch.float32)\n"
     "        if a0.shape[0] != batch or k0.shape[0] != batch:\n"
     "            raise ValueError(f\"refcv7 NEW-1: prior rows {a0.shape[0]}/\"\n"
     "                             f\"{k0.shape[0]} != batch {batch}\")\n"
     "        if v_ms is None:\n"
     "            v = a0.new_full((batch,), self.anchor_ref_speed)\n"
     "            return torch.zeros_like(a0), torch.zeros_like(k0), v\n"
     "        v = self._prior_speed(v_ms, ego_keep, withheld_speed)\n"
     "        a0, k0 = _kp.withhold(a0, k0, ego_keep)\n"
     "        return a0, k0, v\n"
     "\n"
     "    def _roll_residual_bank(self, prior, batch: int,\n"
     "                            dtype: torch.dtype) -> Tensor:\n"
     "        \"\"\"[B, N, S, 2] -- the vocabulary as RESIDUALS on ``P``.\"\"\"\n"
     "        if self.residual_prior == _kp.RESIDUAL_PRIOR_OFF:\n"
     "            raise ValueError(\n"
     "                \"refcv7 NEW-1: roll_bank got a prior on a decoder built \"\n"
     "                \"with residual_prior='off' -- refusing to compose silently.\")\n"
     "        if prior is None:\n"
     "            raise ValueError(\n"
     "                \"refcv7 NEW-1: roll_bank on a residual build needs the \"\n"
     "                \"forward's prior; without it the bank would be Delta.\")\n"
     "        a0, k0, v = prior\n"
     "        seq = self.anchor_control_seq(batch, torch.float32)\n"
     "        return _kp.roll_plan(\n"
     "            seq, a0, k0, v, self.anchor_horizons,\n"
     "            control_units=self.anchor_control_units, tick=self.anchor_dt,\n"
     "            alat_v_floor=self.anchor_alat_v_floor,\n"
     "            kappa_cap=self.anchor_kappa_cap).to(dtype)\n"
     "\n"
     "    def _roll_state(self, z: Tensor, v: Tensor, metre: bool,\n"
     "                    prior) -> Tensor:\n"
     "        \"\"\"The sampler's OWN state -> the fan. Off: the legacy\n"
     "        :meth:`_state_to_path`, same call. Residual: ``z`` is Delta and the\n"
     "        fan is ``kinematic_prior.roll_plan(Delta, P)``.\"\"\"\n"
     "        if prior is None:\n"
     "            return self._state_to_path(z, v, metre)\n"
     "        a0, k0, pv = prior\n"
     "        return _kp.roll_plan(\n"
     "            z, a0, k0, pv, self.anchor_horizons,\n"
     "            control_units=self.anchor_control_units, tick=self.anchor_dt,\n"
     "            alat_v_floor=self.anchor_alat_v_floor,\n"
     "            kappa_cap=self.anchor_kappa_cap)\n"
     "\n"
     "    def _export_controls(self, z: Tensor, prior) -> Tensor:\n"
     "        \"\"\"What leaves the decoder as ``u0_hat`` / ``layer_u0_hat``. Off:\n"
     "        ``z`` itself (the SAME object). Residual: the ABSOLUTE controls\n"
     "        (``kinematic_prior.absolute_controls``) -- the residual state never\n"
     "        leaves this class, so no consumer can mistake Delta for the plan.\"\"\"\n"
     "        if prior is None:\n"
     "            return z\n"
     "        a0, k0, pv = prior\n"
     "        return _kp.absolute_controls(\n"
     "            z, a0, k0, pv, control_units=self.anchor_control_units,\n"
     "            alat_v_floor=self.anchor_alat_v_floor)\n"
     "\n"
     "    def _feasible(self, x: Tensor, v_ms: Tensor | None) -> Tensor:\n")

edit(REFC, "refc.state_to_path",
     "    def _state_to_path(self, state: Tensor, v: Tensor,\n"
     "                       metre: bool) -> Tensor:\n"
     "        \"\"\"The sampler's state -> the emitted fan ``[B, N, S, 2]`` in metres.\n",
     "    def _state_to_path(self, state: Tensor, v: Tensor,\n"
     "                       metre: bool, prior=None) -> Tensor:\n"
     "        \"\"\"The sampler's state -> the emitted fan ``[B, N, S, 2]`` in metres.\n"
     "\n"
     "        (( !! )) refcv7 NEW-1: on a RESIDUAL build ``state`` is an EXPORTED\n"
     "        control tensor (``u0_hat`` / ``layer_u0_hat``, ABSOLUTE) and ``prior``\n"
     "        is REQUIRED -- rolling exported controls without it would hand the\n"
     "        caller a path the model never emitted. ``v`` must then be the\n"
     "        prior's own roll speed (``out['residual_prior_v']``). The trainer's F3\n"
     "        cascade is the one production caller. The sampler's INTERNAL rolls\n"
     "        go through :meth:`_roll_state`, never through here.\n")

edit(REFC, "refc.state_to_path_body",
     "        why that arm is pre-registered to fail the flyability gate.\n"
     "        \"\"\"\n"
     "        if metre:\n"
     "            return state\n",
     "        why that arm is pre-registered to fail the flyability gate.\n"
     "        \"\"\"\n"
     "        if self.residual_prior != _kp.RESIDUAL_PRIOR_OFF:\n"
     "            if prior is None:\n"
     "                raise ValueError(\n"
     "                    \"refcv7 NEW-1: _state_to_path on a residual build needs \"\n"
     "                    \"`prior=` (kinematic_prior.prior_from_out(out)); the \"\n"
     "                    \"exported controls rolled without it are not the plan.\")\n"
     "            a0, k0, pv = prior\n"
     "            if not torch.equal(v.reshape(-1).to(torch.float32),\n"
     "                               pv.reshape(-1).to(torch.float32)):\n"
     "                raise ValueError(\n"
     "                    \"refcv7 NEW-1: _state_to_path was handed a speed that is \"\n"
     "                    \"not the prior's roll speed (on a withheld training row \"\n"
     "                    \"the raw v0 differs from the bank's). Pass \"\n"
     "                    \"out['residual_prior_v'].\")\n"
     "            delta = _kp.residual_controls(\n"
     "                state, a0, k0, pv, control_units=self.anchor_control_units,\n"
     "                alat_v_floor=self.anchor_alat_v_floor)\n"
     "            return self._roll_state(delta, pv, metre, prior)\n"
     "        if prior is not None:\n"
     "            raise ValueError(\n"
     "                \"refcv7 NEW-1: _state_to_path got a prior on a decoder built \"\n"
     "                \"with residual_prior='off' -- refusing to compose silently.\")\n"
     "        if metre:\n"
     "            return state\n")

edit(REFC, "refc.sample_signature",
     "                bev: Tensor | None = None\n"
     "                ) -> tuple[Tensor, Tensor, Tensor, dict]:\n",
     "                bev: Tensor | None = None,\n"
     "                prior: \"tuple | None\" = None\n"
     "                ) -> tuple[Tensor, Tensor, Tensor, dict]:\n")

edit(REFC, "refc.sample_speed",
     "        v = (bank.new_full((b,), self.anchor_ref_speed)\n"
     "             if v_ms is None else v_ms.reshape(-1).to(torch.float32))\n",
     "        v = (bank.new_full((b,), self.anchor_ref_speed)\n"
     "             if v_ms is None else v_ms.reshape(-1).to(torch.float32))\n"
     "        if prior is not None:\n"
     "            # refcv7 NEW-1: the fan is rolled from the PRIOR's speed -- the\n"
     "            # one `roll_bank` rolled the bank from (`_prior_speed`).\n"
     "            v = prior[2]\n")

edit(REFC, "refc.sample_xpath",
     "            x_path = self._state_to_path(denorm(x_n), v, metre)\n",
     "            x_path = self._roll_state(denorm(x_n), v, metre, prior)\n")

edit(REFC, "refc.sample_layer_u0",
     "                    layer_u0.append(denorm(x_n + du_i))\n",
     "                    layer_u0.append(self._export_controls(\n"
     "                        denorm(x_n + du_i), prior))\n")

edit(REFC, "refc.sample_fan",
     "        u0_hat = denorm(x0_hat_n)\n"
     "        fan = self._state_to_path(u0_hat, v, metre)\n",
     "        u0_hat = denorm(x0_hat_n)\n"
     "        fan = self._roll_state(u0_hat, v, metre, prior)\n"
     "        # refcv7 NEW-1: the residual never leaves the decoder -- `u0_hat` is\n"
     "        # exported ABSOLUTE (the same object when the prior is off).\n"
     "        u0_hat = self._export_controls(u0_hat, prior)\n")

edit(REFC, "refc.forward_signature",
     "                nav_cmd_sel: Tensor | None = None,\n"
     "                v_limit_ms: Tensor | None = None) -> dict:\n",
     "                nav_cmd_sel: Tensor | None = None,\n"
     "                v_limit_ms: Tensor | None = None,\n"
     "                residual_prior: \"tuple | None\" = None) -> dict:\n")

edit(REFC, "refc.forward_bank",
     "        bank = self.roll_bank(v_ms, ego_keep, b, fmap.dtype,\n"
     "                              withheld_speed=withheld_speed)\n",
     "        # refcv7 NEW-1: the prior, ONCE per forward -- `(a0, kappa0)` withheld\n"
     "        # on dropped rows, and the ONE speed the bank, the prior and the\n"
     "        # sampled fan share. `None` on an off build: nothing below changes.\n"
     "        _rp = self._residual_prior(residual_prior, v_ms, ego_keep, b,\n"
     "                                   withheld_speed)\n"
     "        bank = self.roll_bank(v_ms, ego_keep, b, fmap.dtype,\n"
     "                              withheld_speed=withheld_speed, prior=_rp)\n")

edit(REFC, "refc.forward_sample_call",
     "            x, u0_hat, s_conf, smp_tele = self._sample(\n"
     "                kv, cond, bank, v_ms, steps, agent_tokens, agent_pad,\n"
     "                agent_pos, bev)\n",
     "            # refcv7 NEW-1: `prior=` travels ONLY on a residual build, so an\n"
     "            # off build calls `_sample` exactly as refcv6 did -- a wrapper of\n"
     "            # `_sample` written against the refcv6 signature (the DDv2 chain's\n"
     "            # capture hook, `tanitad/rl/ddv2_refc_chain.py`) is untouched by\n"
     "            # NEW-1 on every refcv6 build, and a wrapper that cannot carry the\n"
     "            # prior fails LOUDLY (TypeError) on a residual build.\n"
     "            _smp_kw = {} if _rp is None else {\"prior\": _rp}\n"
     "            x, u0_hat, s_conf, smp_tele = self._sample(\n"
     "                kv, cond, bank, v_ms, steps, agent_tokens, agent_pad,\n"
     "                agent_pos, bev, **_smp_kw)\n")

edit(REFC, "refc.forward_outputs",
     "        out = {\"anchor_logits\": conf, \"refined_logits\": refined,\n"
     "               \"anchor_traj\": x, \"anchor_bank\": bank,\n"
     "               \"offset\": offset, \"sel_score\": score,\n"
     "               \"traj\": traj, \"sel_idx\": idx, \"sel_tele\": tele}\n",
     "        out = {\"anchor_logits\": conf, \"refined_logits\": refined,\n"
     "               \"anchor_traj\": x, \"anchor_bank\": bank,\n"
     "               \"offset\": offset, \"sel_score\": score,\n"
     "               \"traj\": traj, \"sel_idx\": idx, \"sel_tele\": tele}\n"
     "        if _rp is not None:\n"
     "            # refcv7 NEW-1: the prior this forward composed on, EMITTED so a\n"
     "            # consumer can verify the plan is P + Delta and the launch gate's\n"
     "            # G-LIVE can read 'the prior is non-zero where v0 > 0'. Constants\n"
     "            # of the window (no parameter), so DETACHED.\n"
     "            out[\"residual_prior_ctrl\"] = torch.stack(\n"
     "                [_rp[0], _rp[1]], dim=-1).detach()\n"
     "            out[\"residual_prior_v\"] = _rp[2].detach()\n"
     "            out[\"residual_prior_path\"] = _kp.prior_path(\n"
     "                _rp[0], _rp[1], _rp[2], self.anchor_horizons,\n"
     "                tick=self.anchor_dt).to(x.dtype).detach()\n"
     "            tele[\"residual_prior\"] = self.residual_prior\n")

edit(REFC, "refc.model_init_state",
     "        self._ego_window: tuple | None = None\n",
     "        self._ego_window: tuple | None = None\n"
     "        # refcv7 NEW-1: the observed window's RECORDED actions, popped WITH\n"
     "        # `_ego_window` (a separate attribute, so the 2-tuple every existing\n"
     "        # reader indexes is unchanged).\n"
     "        self._ego_actions: Tensor | None = None\n")

edit(REFC, "refc.model_init_refuse",
     "        # ---- refcv5 WP-6: THE AGENT SEAM (default OFF, builds NOTHING) ----\n",
     "        # ---- refcv7 NEW-1: what the residual prior READS must be built ----- #\n"
     "        # The prior is computed from the OBSERVED ego window, which reaches the\n"
     "        # model only through the ego-history channel, and it is rolled from\n"
     "        # `v_ms`, which the decoder receives only under `sel_reach_clamp`. A\n"
     "        # residual build without either has no prior to compose on.\n"
     "        if self.decoder.residual_prior != _kp.RESIDUAL_PRIOR_OFF:\n"
     "            if self.ego_hist is None:\n"
     "                raise ValueError(\n"
     "                    \"refcv7 NEW-1: residual_prior needs the ego-history \"\n"
     "                    \"channel (--ego-history): the prior reads the observed \"\n"
     "                    \"pose window, and that window reaches the model only \"\n"
     "                    \"through it.\")\n"
     "            if not bool(getattr(cfg, \"sel_reach_clamp\", False)):\n"
     "                raise ValueError(\n"
     "                    \"refcv7 NEW-1: residual_prior needs sel_reach_clamp: the \"\n"
     "                    \"decoder receives the measured speed (`v_ms`) the prior \"\n"
     "                    \"is rolled from only under it.\")\n"
     "        # ---- refcv5 WP-6: THE AGENT SEAM (default OFF, builds NOTHING) ----\n")

edit(REFC, "refc.set_ego_window",
     "    def set_ego_window(self, poses: Tensor, n_past: int) -> None:\n",
     "    def set_ego_window(self, poses: Tensor, n_past: int,\n"
     "                       actions: Tensor | None = None) -> None:\n")

edit(REFC, "refc.set_ego_window_store",
     "        self._ego_window = (poses, int(n_past))\n",
     "        self._ego_window = (poses, int(n_past))\n"
     "        # refcv7 NEW-1 (`ha0_ext` only): the observed window's actions, popped\n"
     "        # with the window by the next forward.\n"
     "        self._ego_actions = actions\n")

edit(REFC, "refc.model_forward_signature",
     "                bev_tokens: Tensor | None = None,\n"
     "                bev_pad: Tensor | None = None) -> dict:\n",
     "                bev_tokens: Tensor | None = None,\n"
     "                bev_pad: Tensor | None = None,\n"
     "                ego_actions: Tensor | None = None) -> dict:\n")

edit(REFC, "refc.model_forward_ego_pop",
     "            if ego_poses is None and self._ego_window is not None:\n"
     "                ego_poses, _n = self._ego_window\n"
     "                self._ego_window = None\n"
     "                ego_n_past = _n if ego_n_past is None else ego_n_past\n",
     "            _win_acts, self._ego_actions = self._ego_actions, None\n"
     "            if ego_poses is None and self._ego_window is not None:\n"
     "                ego_poses, _n = self._ego_window\n"
     "                self._ego_window = None\n"
     "                ego_n_past = _n if ego_n_past is None else ego_n_past\n"
     "                if ego_actions is None:\n"
     "                    ego_actions = _win_acts\n")

edit(REFC, "refc.model_forward_prior",
     "            ego_vec = self.ego_hist(ch)\n",
     "            ego_vec = self.ego_hist(ch)\n"
     "            # refcv7 NEW-1: the prior reads THIS window through the SAME\n"
     "            # channel builder (`kinematic_prior.prior_controls` calls\n"
     "            # `ego_channels_from_poses`), so the prior and the history encoder\n"
     "            # cannot disagree about what t0 was.\n"
     "            if self.decoder.residual_prior != _kp.RESIDUAL_PRIOR_OFF:\n"
     "                _rp_in = self._residual_prior_inputs(ego_poses, n_past,\n"
     "                                                     ego_actions, v0)\n"
     "        if (ego_actions is not None and not (\n"
     "                self.decoder.residual_prior != _kp.RESIDUAL_PRIOR_OFF\n"
     "                and _kp.needs_actions(self.decoder.residual_prior))):\n"
     "            raise ValueError(\n"
     "                \"refcv7 NEW-1: `ego_actions` reached a build whose residual \"\n"
     "                f\"prior ({self.decoder.residual_prior!r}) does not read them \"\n"
     "                \"-- they would be SILENTLY DROPPED.\")\n")

edit(REFC, "refc.model_forward_prior_init",
     "        ego_vec = None\n"
     "        if self.ego_hist is not None:\n",
     "        ego_vec = None\n"
     "        _rp_in = None                       # refcv7 NEW-1 (None when off)\n"
     "        if self.ego_hist is not None:\n")

edit(REFC, "refc.model_forward_decoder_call",
     "                           nav_cmd_sel=(nav_cmd if nav_cmd_given else None),\n"
     "                           v_limit_ms=v_limit_ms)\n",
     "                           nav_cmd_sel=(nav_cmd if nav_cmd_given else None),\n"
     "                           v_limit_ms=v_limit_ms,\n"
     "                           residual_prior=_rp_in)\n")

edit(REFC, "refc.decoder_passthrough",
     "    DECODER_PASSTHROUGH: tuple[str, ...] = (\n"
     "        \"prefinal_logits\", \"reach_keep\", \"layer_u0_hat\", \"layer_logits\")\n",
     "    DECODER_PASSTHROUGH: tuple[str, ...] = (\n"
     "        \"prefinal_logits\", \"reach_keep\", \"layer_u0_hat\", \"layer_logits\",\n"
     "        # refcv7 NEW-1: the prior this forward composed on. The F3 cascade re-roll READS\n"
     "        # `residual_prior_ctrl` / `residual_prior_v` (`kinematic_prior.prior_from_out`) and\n"
     "        # the launch gate's G-LIVE reads `residual_prior_path`. Absent from `dec` on an off\n"
     "        # build, so the loop below copies nothing there (bit-identity).\n"
     "        \"residual_prior_ctrl\", \"residual_prior_v\", \"residual_prior_path\")\n")

edit(REFC, "refc.model_residual_inputs_method",
     "    def encode_pooled(self, frames: Tensor) -> Tensor:\n",
     "    def _residual_prior_inputs(self, ego_poses: Tensor, n_past: int,\n"
     "                               ego_actions: Tensor | None,\n"
     "                               v0: Tensor | None) -> tuple:\n"
     "        \"\"\"refcv7 NEW-1: ``(a0 [B], kappa0 [B])`` of this window's prior.\n"
     "\n"
     "        (( !! )) Consistency, asserted: the pose window must END at the\n"
     "        ``v0`` the plan is rolled from. A window shifted by one frame still\n"
     "        yields a plausible prior -- and a wrong one on every row.\n"
     "        \"\"\"\n"
     "        mode = self.decoder.residual_prior\n"
     "        acts = ego_actions if _kp.needs_actions(mode) else None\n"
     "        a0, k0 = _kp.prior_controls(mode, ego_poses, n_past, acts,\n"
     "                                    dt=float(self.ego_hist.cfg.dt))\n"
     "        if v0 is not None:\n"
     "            vw = ego_poses[:, int(n_past) - 1, 3].to(torch.float32)\n"
     "            dv = float((vw - v0.reshape(-1).to(torch.float32)).abs().max())\n"
     "            if dv > _kp.V0_WINDOW_TOL:\n"
     "                raise ValueError(\n"
     "                    f\"refcv7 NEW-1: the pose window's last speed differs \"\n"
     "                    f\"from v0 by up to {dv:.6f} m/s (> {_kp.V0_WINDOW_TOL}). \"\n"
     "                    f\"The window does not end at the frame the plan starts \"\n"
     "                    f\"from; the prior would be built at the wrong t0.\")\n"
     "        return a0, k0\n"
     "\n"
     "    def encode_pooled(self, frames: Tensor) -> Tensor:\n")

# =========================================================================== #
# refc_v3.py                                                                  #
# =========================================================================== #
edit(REFCV3, "refcv3.forward_signature",
     "                perception_grid: Tensor | None = None,\n"
     "                perception_valid: Tensor | None = None) -> dict:\n",
     "                perception_grid: Tensor | None = None,\n"
     "                perception_valid: Tensor | None = None,\n"
     "                ego_actions: Tensor | None = None) -> dict:\n")

edit(REFCV3, "refcv3.core_call_flat",
     "                             ego_poses=ego_poses, ego_n_past=ego_n_past)\n",
     "                             ego_poses=ego_poses, ego_n_past=ego_n_past,\n"
     "                             ego_actions=ego_actions)\n")

edit(REFCV3, "refcv3.core_call_hier",
     "                        ego_poses=ego_poses, ego_n_past=ego_n_past,\n"
     "                        **_core_kw)\n",
     "                        ego_poses=ego_poses, ego_n_past=ego_n_past,\n"
     "                        ego_actions=ego_actions,\n"
     "                        **_core_kw)\n")

# =========================================================================== #
# refc_v3_train.py                                                            #
# =========================================================================== #
edit(TRAIN, "train.import",
     "from tanitad.models import refcv6_diffusion as _rv6  # noqa: E402  (refcv6 §3)\n",
     "from tanitad.models import refcv6_diffusion as _rv6  # noqa: E402  (refcv6 §3)\n"
     "from tanitad.models import kinematic_prior as _kp  # noqa: E402  (refcv7 NEW-1)\n")

edit(TRAIN, "train.argparse",
     "    ap.add_argument(\"--ego-history-out\", type=int, default=32)\n",
     "    ap.add_argument(\"--ego-history-out\", type=int, default=32)\n"
     "    # ---- refcv7 NEW-1: THE PLAN IS A RESIDUAL ON A CAUSAL KINEMATIC PRIOR -\n"
     "    ap.add_argument(\"--residual-prior\", choices=_kp.RESIDUAL_PRIOR_MODES,\n"
     "                    default=_kp.RESIDUAL_PRIOR_OFF,\n"
     "                    help=\"refcv7 NEW-1 (SPEC_REFCV7 section 1). 'off' (default) \"\n"
     "                         \"is the refcv6 decoder bit for bit. Otherwise the \"\n"
     "                         \"sampler denoises a RESIDUAL on a causal kinematic \"\n"
     "                         \"prior P and the plan is P + Delta. 'ha0_ext' = the \"\n"
     "                         \"battery's echo exactly (reads the recorded steer \"\n"
     "                         \"at t0); 'ha0_ext_pose' = the same construction \"\n"
     "                         \"with the pose-track curvature; 'cv_yawrate' = \"\n"
     "                         \"constant speed + constant yaw rate. Definitions: \"\n"
     "                         \"tanitad/models/kinematic_prior.py.\")\n")

edit(TRAIN, "train.pin_cfg",
     "    # ---- refcv6 §3: the F1..F9 block onto the CORE config ----------------- #\n",
     "    # ---- refcv7 NEW-1: the residual prior onto the DECODER config -------- #\n"
     "    # Pinned onto the CONFIG (a declared `DecoderConfig` field), never only\n"
     "    # onto `args`: `rebuild_config` rebuilds through this helper, so the eval\n"
     "    # loaders replay it from argv and build the identical decoder.\n"
     "    _rp_mode = str(getattr(args, \"residual_prior\", _kp.RESIDUAL_PRIOR_OFF)\n"
     "                   or _kp.RESIDUAL_PRIOR_OFF)\n"
     "    cfg.core.decoder.residual_prior = _kp.check_mode(_rp_mode)\n"
     "    if _rp_mode != _kp.RESIDUAL_PRIOR_OFF and not bool(\n"
     "            getattr(args, \"ego_history\", False)):\n"
     "        raise SystemExit(\n"
     "            \"[v3] --residual-prior %s needs --ego-history: the prior reads \"\n"
     "            \"the observed pose window, which reaches the model only through \"\n"
     "            \"that channel.\" % _rp_mode)\n"
     "    # ---- refcv6 §3: the F1..F9 block onto the CORE config ----------------- #\n")

edit(TRAIN, "train.set_ego_window",
     "        ph = batch[\"pose_hist\"].to(device)\n"
     "        model.core.set_ego_window(ph, int(ph.shape[1]))\n",
     "        ph = batch[\"pose_hist\"].to(device)\n"
     "        # refcv7 NEW-1 (`ha0_ext` only): the observed window's RECORDED\n"
     "        # actions ride with the window; every other mode reads the poses.\n"
     "        _rpm = str(getattr(core.decoder, \"residual_prior\",\n"
     "                           _kp.RESIDUAL_PRIOR_OFF))\n"
     "        _acts = (batch[\"actions\"].to(device)\n"
     "                 if (_rpm != _kp.RESIDUAL_PRIOR_OFF\n"
     "                     and _kp.needs_actions(_rpm)) else None)\n"
     "        model.core.set_ego_window(ph, int(ph.shape[1]), actions=_acts)\n")

edit(TRAIN, "train.cascade_speed",
     "            v_cas = v0.reshape(-1).to(torch.float32)\n",
     "            v_cas = v0.reshape(-1).to(torch.float32)\n"
     "            # refcv7 NEW-1: a residual build rolls every stage from the\n"
     "            # PRIOR's speed and composes it on the prior -- the fan's own roll.\n"
     "            _rp_cas = _kp.prior_from_out(out)\n"
     "            if _rp_cas is not None:\n"
     "                v_cas = _rp_cas[2]\n")

edit(TRAIN, "train.cascade_roll",
     "                p_i = model.core.decoder._state_to_path(\n"
     "                    u_i, v_cas,\n"
     "                    str(core.decoder.sampler_space) == \"metre\")\n",
     "                p_i = model.core.decoder._state_to_path(\n"
     "                    u_i, v_cas,\n"
     "                    str(core.decoder.sampler_space) == \"metre\",\n"
     "                    prior=_rp_cas)\n")

edit(TRAIN, "train.seam_stamp",
     "        \"refcv6\": (_rv6.flag_stamp(core.decoder.refcv6)\n"
     "                   if getattr(core.decoder, \"refcv6\", None) is not None\n"
     "                   else None),\n",
     "        \"refcv6\": (_rv6.flag_stamp(core.decoder.refcv6)\n"
     "                   if getattr(core.decoder, \"refcv6\", None) is not None\n"
     "                   else None),\n"
     "        # refcv7 NEW-1 -- the full definition, not just the mode, so the run\n"
     "        # record says WHICH prior (and whether it IS the battery's echo).\n"
     "        \"residual_prior\": _kp.prior_stamp(str(getattr(\n"
     "            core.decoder, \"residual_prior\", _kp.RESIDUAL_PRIOR_OFF))),\n")

edit(TRAIN, "train.assert_built",
     "    # --- WP-6: the agent seam --------------------------------------------- #\n",
     "    # --- refcv7 NEW-1: the residual prior, BOTH directions ---------------- #\n"
     "    _rp_st = (stamp.get(\"residual_prior\") or {}).get(\"residual_prior\",\n"
     "                                                     _kp.RESIDUAL_PRIOR_OFF)\n"
     "    _rp_built = str(getattr(dec, \"residual_prior\", _kp.RESIDUAL_PRIOR_OFF))\n"
     "    if _rp_st != _rp_built:\n"
     "        bad.append(f\"stamp says residual_prior={_rp_st!r} but the decoder \"\n"
     "                   f\"was BUILT with {_rp_built!r} -- the record would name a \"\n"
     "                   f\"different plan space than the one that trains\")\n"
     "    # --- WP-6: the agent seam --------------------------------------------- #\n")

# =========================================================================== #
# declared_vs_built.py (the fixes agent's G-DVB registry; lands BEFORE NEW-1) #
# =========================================================================== #
DVB = "stack/tanitad/train/declared_vs_built.py"

edit(DVB, "dvb.check_fn",
     "# ============================================================================\n"
     "# THE REGISTRY \u2014 every trainer dest, one entry each (coverage is tested)\n",
     "def _c_residual_prior(m, a):\n"
     "    \"\"\"refcv7 NEW-1 (SPEC_REFCV7 section 1). The mode is read from the BUILT decoder\n"
     "    (``AnchoredDiffusionDecoder.residual_prior``, set at construction and the attribute every\n"
     "    forward branches on) AND from its config; a residual build must also carry the three\n"
     "    mechanisms the prior composes with. Expectation: the LITERAL argv value.\"\"\"\n"
     "    want = str(_a(a, \"residual_prior\", \"off\"))\n"
     "    d = _dec(m)\n"
     "    out = _eq(\"residual_prior\", want, str(getattr(d, \"residual_prior\", \"<absent>\")),\n"
     "              \"core.decoder.residual_prior\")\n"
     "    out += _eq(\"residual_prior\", want, str(getattr(d.cfg, \"residual_prior\", \"<absent>\")),\n"
     "               \"core.decoder.cfg.residual_prior\")\n"
     "    if want != \"off\":\n"
     "        for ok, where, why in (\n"
     "                (bool(getattr(d, \"anchor_v0_cond\", False)), \"core.decoder.anchor_v0_cond\",\n"
     "                 \"the residual vocabulary is rolled per window from v0\"),\n"
     "                (getattr(d, \"time_mlp\", None) is not None, \"core.decoder.time_mlp\",\n"
     "                 \"the residual IS the DDIM sampler's state\"),\n"
     "                (getattr(_core(m), \"ego_hist\", None) is not None, \"core.ego_hist\",\n"
     "                 \"the prior reads the observed pose window\")):\n"
     "            if not ok:\n"
     "                out.append(Mismatch(\"--residual-prior\", want, f\"{where} missing\", where, why))\n"
     "        passthrough = tuple(getattr(type(_core(m)), \"DECODER_PASSTHROUGH\", ()))\n"
     "        # literal: the cascade re-roll reads the first two, G-LIVE the third\n"
     "        for key in (\"residual_prior_ctrl\", \"residual_prior_v\", \"residual_prior_path\"):\n"
     "            if key not in passthrough:\n"
     "                out.append(Mismatch(\"--residual-prior\", f\"{key} in the forward output\",\n"
     "                                    \"absent\", f\"{type(_core(m)).__name__}.DECODER_PASSTHROUGH\",\n"
     "                                    \"the F3 re-roll could not compose on P (the A16 class)\"))\n"
     "    return out\n"
     "\n"
     "\n"
     "# ============================================================================\n"
     "# THE REGISTRY \u2014 every trainer dest, one entry each (coverage is tested)\n")

edit(DVB, "dvb.register",
     "_b(\"ego_history\", _c_ego_history)\n",
     "_b(\"ego_history\", _c_ego_history)\n"
     "# refcv7 NEW-1: the plan is a residual on a causal kinematic prior\n"
     "_b(\"residual_prior\", _c_residual_prior)\n")

DVB_TEST = "stack/tests/test_declared_vs_built.py"

edit(DVB_TEST, "dvb_test.registry_count",
     "    assert len(dvb.REGISTRY) == 202          # the tip's 197 dests + the 5 flags of this batch\n",
     "    # the tip's 197 dests + the 5 flags of the fixes batch + refcv7 NEW-1's --residual-prior\n"
     "    assert len(dvb.REGISTRY) == 203\n")

# =========================================================================== #
# 2026-09-27, after the Thor full-suite gate (TIP b3f7ea6 vs TIP + NEW-1): 21   #
# RL-stack regressions in two groups. (1) the DDv2 chain's FIXED-ARITY capture #
# hook raised TypeError on `_sample`'s new `prior=` (8 tests); (2) the new     #
# forward channel `ego_actions` was declared nowhere, so both RL adapters'      #
# "undeclared channel => REFUSE" guards fired (13 tests). Both guards were      #
# RIGHT; neither is weakened below.                                            #
# =========================================================================== #
CHAIN = "stack/tanitad/rl/ddv2_refc_chain.py"

edit(CHAIN, "chain.import",
     "from tanitad.rl import ddv2_rl as D\n",
     "from tanitad.models import kinematic_prior as _kp\n"
     "from tanitad.rl import ddv2_rl as D\n")

edit(CHAIN, "chain.docstring",
     "Tier: T0 machinery. Evidence class: MEASURED (``tests/test_ddv2_refc_chain.py``).\n",
     "⭐ refcv7 NEW-1 (2026-09-27): ON A RESIDUAL BUILD THE SAMPLER'S STATE IS Delta, NOT THE PLAN.\n"
     "``_sample`` then takes ``prior=(a0, kappa0, v)`` and rolls every state as ``P + Delta``\n"
     "(``kinematic_prior.roll_plan``). The capture forwards that prior VERBATIM and RECORDS it\n"
     "(:attr:`SamplerInputs.prior`), and every roll here composes on it the way ``_sample`` does:\n"
     ":func:`state_to_path` REFUSES a residual build without it, an off build with it, and a speed\n"
     "that is not the prior's. On an off build nothing here changes: ``prior`` is ``None`` and every\n"
     "call is the pre-NEW-1 call.\n"
     "\n"
     "Tier: T0 machinery. Evidence class: MEASURED (``tests/test_ddv2_refc_chain.py``).\n")

edit(CHAIN, "chain.inputs_doc",
     "    \"\"\"What ``_sample`` was called with. ``v`` is the speed ``_sample`` itself derives\n"
     "    (``refc.py:2085-2086``): ``v_ms`` when given, else the anchor reference speed.\"\"\"\n",
     "    \"\"\"What ``_sample`` was called with. ``v`` is the speed ``_sample`` itself derives\n"
     "    (``refc.py:2085-2086``): ``v_ms`` when given, else the anchor reference speed.\n"
     "\n"
     "    ``prior`` (refcv7 NEW-1) is ``_sample``'s ``prior=`` VERBATIM: ``(a0 [B], kappa0 [B],\n"
     "    v [B])`` on a residual build, ``None`` on every other build. When it is set ``v`` IS\n"
     "    ``prior[2]`` -- ``_sample``'s own rule: the fan is rolled from the prior's speed, the one\n"
     "    ``roll_bank`` rolled the bank from, which on a WITHHELD row is not ``v_ms``.\"\"\"\n")

edit(CHAIN, "chain.inputs_field",
     "    rng_cuda: Tensor | None = None\n"
     "\n"
     "    def replay_eps(self, decoder) -> Tensor:\n",
     "    rng_cuda: Tensor | None = None\n"
     "    #: refcv7 NEW-1: the prior ``_sample`` composed on (see the class docstring)\n"
     "    prior: tuple | None = None\n"
     "\n"
     "    def replay_eps(self, decoder) -> Tensor:\n")

edit(CHAIN, "chain.hook_signature",
     "    def hook(kv, cond, bank, v_ms, steps, agents=None, agent_pad=None, agent_pos=None,\n"
     "             bev=None):\n",
     "    def hook(kv, cond, bank, v_ms, steps, agents=None, agent_pad=None, agent_pos=None,\n"
     "             bev=None, prior=None):\n")

edit(CHAIN, "chain.hook_body",
     "        out = orig(kv, cond, bank, v_ms, steps, agents, agent_pad, agent_pos, bev)\n"
     "        b = bank.shape[0]\n"
     "        v = (bank.new_full((b,), decoder.anchor_ref_speed) if v_ms is None\n"
     "             else v_ms.reshape(-1).to(torch.float32))\n"
     "        records.append(SamplerInputs(kv=kv, cond=cond, bank=bank, v_ms=v_ms, v=v,\n"
     "                                     steps=int(steps), out=out,\n"
     "                                     rng_cpu=rng_cpu, rng_cuda=rng_cuda))\n",
     "        # ⛔ refcv7 NEW-1, 2026-09-27. `_sample` gained a TENTH parameter, `prior`, and the\n"
     "        # Thor full-suite gate caught this fixed-arity hook on it (8 TypeErrors). It is\n"
     "        # accepted, forwarded VERBATIM and RECORDED -- never refused like `agents`/`bev`: it\n"
     "        # is not a coupling the refcv5-v2 build lacks, it is the reference frame of the\n"
     "        # sampler's state on a residual build, and a chain that dropped it would roll Delta\n"
     "        # as if it were the plan. Forwarded only when given, so an off build's call is the\n"
     "        # pre-NEW-1 call byte for byte (the decoder passes it only then, too).\n"
     "        out = (orig(kv, cond, bank, v_ms, steps, agents, agent_pad, agent_pos, bev)\n"
     "               if prior is None else\n"
     "               orig(kv, cond, bank, v_ms, steps, agents, agent_pad, agent_pos, bev,\n"
     "                    prior=prior))\n"
     "        b = bank.shape[0]\n"
     "        v = (bank.new_full((b,), decoder.anchor_ref_speed) if v_ms is None\n"
     "             else v_ms.reshape(-1).to(torch.float32))\n"
     "        if prior is not None:\n"
     "            v = prior[2]              # `_sample`'s own rule on a residual build\n"
     "        records.append(SamplerInputs(kv=kv, cond=cond, bank=bank, v_ms=v_ms, v=v,\n"
     "                                     steps=int(steps), out=out,\n"
     "                                     rng_cpu=rng_cpu, rng_cuda=rng_cuda, prior=prior))\n")

edit(CHAIN, "chain.state_to_path",
     "def state_to_path(decoder, x_n: Tensor, v: Tensor) -> Tensor:\n"
     "    \"\"\"Normalised control state ``[B, M, S, 2]`` -> metres ``[B, M, S, 2]`` through the decoder's\n"
     "    own integrator (``refc.py:2025-2040``).\"\"\"\n"
     "    norm = x_n.new_tensor(tuple(decoder.cfg.control_norm))\n"
     "    return decoder._state_to_path(x_n * norm, v, False)\n",
     "def state_to_path(decoder, x_n: Tensor, v: Tensor, prior: tuple | None = None) -> Tensor:\n"
     "    \"\"\"Normalised control state ``[B, M, S, 2]`` -> metres ``[B, M, S, 2]`` through the decoder's\n"
     "    own integrator (``refc.py:2025-2040``).\n"
     "\n"
     "    refcv7 NEW-1: on a RESIDUAL build the chain's state is the sampler's INTERNAL residual\n"
     "    Delta, and its path is ``decoder._roll_state(Delta, v, False, prior)`` -- the roll\n"
     "    ``_sample`` itself makes (``kinematic_prior.roll_plan(Delta, P)``) -- never\n"
     "    ``decoder._state_to_path``, which takes EXPORTED absolute controls. ``prior``\n"
     "    (:attr:`SamplerInputs.prior`) is therefore REQUIRED there, REFUSED on an off build, and\n"
     "    ``v`` must be the prior's own roll speed. Each mismatch raises rather than rolling Delta as\n"
     "    if it were the plan. Off build, ``prior=None``: the pre-NEW-1 call, unchanged.\n"
     "    \"\"\"\n"
     "    norm = x_n.new_tensor(tuple(decoder.cfg.control_norm))\n"
     "    mode = str(getattr(decoder, \"residual_prior\", _kp.RESIDUAL_PRIOR_OFF))\n"
     "    if prior is None:\n"
     "        if mode != _kp.RESIDUAL_PRIOR_OFF:\n"
     "            raise D.Ddv2ConfigError(\n"
     "                f\"refcv7 NEW-1: this decoder was built with residual_prior={mode!r}, so the \"\n"
     "                \"chain's state is the RESIDUAL Delta; rolling it without the prior would hand \"\n"
     "                \"back Delta as the plan. Pass the captured `SamplerInputs.prior`.\")\n"
     "        return decoder._state_to_path(x_n * norm, v, False)\n"
     "    if mode == _kp.RESIDUAL_PRIOR_OFF:\n"
     "        raise D.Ddv2ConfigError(\n"
     "            \"refcv7 NEW-1: a prior was handed to a decoder built with residual_prior='off'; \"\n"
     "            \"composing it would roll a plan this model never emits.\")\n"
     "    if not torch.equal(v.reshape(-1).to(torch.float32),\n"
     "                       prior[2].reshape(-1).to(torch.float32)):\n"
     "        raise D.Ddv2ConfigError(\n"
     "            \"refcv7 NEW-1: the speed handed to the chain is not the prior's roll speed. \"\n"
     "            \"`_sample` rolls a residual build's fan from `prior[2]`, which on a WITHHELD row \"\n"
     "            \"is not the raw v0. Pass `SamplerInputs.v`.\")\n"
     "    return decoder._roll_state(x_n * norm, v, False, prior)\n")

edit(CHAIN, "chain.x0_fn_inputs",
     "    ``M`` may be any multiple of ``N`` (groups): see the module docstring for why that is exact.\n"
     "    \"\"\"\n"
     "    kv, cond, v = inputs.kv, inputs.cond, inputs.v\n",
     "    ``M`` may be any multiple of ``N`` (groups): see the module docstring for why that is exact.\n"
     "    On a residual build (refcv7 NEW-1) the state is Delta and is rolled on ``inputs.prior``\n"
     "    (:func:`state_to_path`), exactly as ``_sample`` rolls it.\n"
     "    \"\"\"\n"
     "    kv, cond, v, prior = inputs.kv, inputs.cond, inputs.v, inputs.prior\n")

edit(CHAIN, "chain.x0_fn_path",
     "        x_path = state_to_path(decoder, x_in, v)\n",
     "        x_path = state_to_path(decoder, x_in, v, prior)\n")

edit(CHAIN, "chain.native_tail",
     "    norm = x0_hat_n.new_tensor(tuple(cfg.control_norm))\n"
     "    u0_hat = x0_hat_n * norm\n"
     "    return decoder._state_to_path(u0_hat, inputs.v, False), u0_hat\n",
     "    norm = x0_hat_n.new_tensor(tuple(cfg.control_norm))\n"
     "    u0_hat = x0_hat_n * norm\n"
     "    if inputs.prior is None:\n"
     "        return decoder._state_to_path(u0_hat, inputs.v, False), u0_hat\n"
     "    # refcv7 NEW-1: `_sample`'s own tail on a residual build -- the fan is Delta rolled on P\n"
     "    # (`_roll_state`) and `u0_hat` leaves ABSOLUTE (`_export_controls`), so parity is checked\n"
     "    # against what the decoder EMITS, never against its internal residual.\n"
     "    return (state_to_path(decoder, x0_hat_n, inputs.v, inputs.prior),\n"
     "            decoder._export_controls(u0_hat, inputs.prior))\n")

CHAIN_TEST = "stack/tests/test_ddv2_refc_chain.py"

edit(CHAIN_TEST, "chain_test.dec_kw",
     "def _dec(seed=0, n=5, horizons=(5, 10, 15, 20)):\n"
     "    torch.manual_seed(seed)\n"
     "    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2, sampler=\"ddim\")\n",
     "def _dec(seed=0, n=5, horizons=(5, 10, 15, 20), **cfg_kw):\n"
     "    torch.manual_seed(seed)\n"
     "    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2, sampler=\"ddim\", **cfg_kw)\n")

edit(CHAIN_TEST, "chain_test.captured_kw",
     "def _captured(dec, seed=5):\n"
     "    fmap, m, v0 = _inputs()\n"
     "    with C.capture_sampler_inputs(dec) as rec:\n"
     "        torch.manual_seed(seed)\n"
     "        with torch.no_grad():\n"
     "            out = dec(fmap, m, steps=2, v_ms=v0)\n",
     "def _captured(dec, seed=5, **fwd_kw):\n"
     "    fmap, m, v0 = _inputs()\n"
     "    with C.capture_sampler_inputs(dec) as rec:\n"
     "        torch.manual_seed(seed)\n"
     "        with torch.no_grad():\n"
     "            out = dec(fmap, m, steps=2, v_ms=v0, **fwd_kw)\n")

edit(CHAIN_TEST, "chain_test.groups_kw",
     "def _group_independence(dec):\n"
     "    inp, _ = _captured(dec)\n",
     "def _group_independence(dec, **fwd_kw):\n"
     "    inp, _ = _captured(dec, **fwd_kw)\n")

edit(CHAIN_TEST, "chain_test.residual_tests",
     "    rates = C.clamp_rates(roll[\"chain\"], roll[\"x0\"])\n"
     "    assert set(rates) >= {\"x0_clamp_frac_a_lon\", \"input_clamp_frac_a_lat\"}\n",
     "    rates = C.clamp_rates(roll[\"chain\"], roll[\"x0\"])\n"
     "    assert set(rates) >= {\"x0_clamp_frac_a_lon\", \"input_clamp_frac_a_lat\"}\n"
     "\n"
     "\n"
     "# --------------------------------------------------------------------------- #\n"
     "# refcv7 NEW-1: the chain on a RESIDUAL build, where the plan is P + Delta.    #\n"
     "# The Thor full-suite gate of 2026-09-27 went RED on every test above that     #\n"
     "# captures: `_sample` gained `prior=` and the fixed-arity capture hook raised  #\n"
     "# TypeError. The hook now forwards the prior VERBATIM and RECORDS it; these    #\n"
     "# pin that the chain composes on it exactly as the deployed sampler does, and  #\n"
     "# that an off build still calls `_sample` the refcv6 way.                      #\n"
     "# --------------------------------------------------------------------------- #\n"
     "import dataclasses  # noqa: E402\n"
     "\n"
     "from tanitad.models import kinematic_prior as KP  # noqa: E402\n"
     "\n"
     "RESIDUAL = {\"residual_prior\": \"ha0_ext_pose\"}\n"
     "\n"
     "\n"
     "def _prior(b=2):\n"
     "    \"\"\"A NON-trivial prior ``(a0 [m/s^2], kappa0 [1/m])``. A zero prior would hide a chain that\n"
     "    ignores it: with ``a0 = kappa0 = 0`` the residual roll IS the vocabulary's own roll.\"\"\"\n"
     "    return torch.tensor([0.8, -1.2][:b]), torch.tensor([0.02, -0.035][:b])\n"
     "\n"
     "\n"
     "def _spy_sample_kwargs(dec, **fwd_kw):\n"
     "    \"\"\"The keyword arguments the decoder's forward passes to ``_sample``.\"\"\"\n"
     "    seen, real = [], dec._sample\n"
     "\n"
     "    def spy(*a, **kw):\n"
     "        seen.append(dict(kw))\n"
     "        return real(*a, **kw)\n"
     "    dec._sample = spy\n"
     "    try:\n"
     "        fmap, m, v0 = _inputs()\n"
     "        with torch.no_grad():\n"
     "            dec(fmap, m, steps=2, v_ms=v0, **fwd_kw)\n"
     "    finally:\n"
     "        del dec._sample\n"
     "    assert len(seen) == 1\n"
     "    return seen[0]\n"
     "\n"
     "\n"
     "def test_NEW1_an_OFF_build_calls_sample_exactly_as_refcv6_did_and_records_no_prior():\n"
     "    \"\"\"OFF parity at the INTERFACE: no ``prior=`` keyword reaches ``_sample`` at all, so a\n"
     "    wrapper written against the refcv6 signature keeps working on every refcv6 build.\"\"\"\n"
     "    dec = _dec()\n"
     "    assert \"prior\" not in _spy_sample_kwargs(dec)\n"
     "    inp, _ = _captured(dec)\n"
     "    assert inp.prior is None\n"
     "    # ...and on a residual build the prior DOES travel, as the one keyword\n"
     "    kw = _spy_sample_kwargs(_dec(**RESIDUAL), residual_prior=_prior())\n"
     "    assert set(kw) == {\"prior\"} and len(kw[\"prior\"]) == 3\n"
     "\n"
     "\n"
     "def test_NEW1_parity_RESIDUAL_the_binding_reproduces_the_deployed_sampler_bitwise():\n"
     "    dec = _dec(**RESIDUAL)\n"
     "    inp, out = _captured(dec, residual_prior=_prior())\n"
     "    # the capture RECORDED the prior exactly as the forward composed it; `v` is its speed\n"
     "    assert inp.prior is not None\n"
     "    for got, want in zip(inp.prior, KP.prior_from_out(out)):\n"
     "        assert torch.equal(got, want)\n"
     "    assert inp.v is inp.prior[2]\n"
     "    fan_ref, u0_ref, _, tele = inp.out\n"
     "    assert tele[\"sampler_ladder\"] == [10, 0]\n"
     "    eps = inp.replay_eps(dec)\n"
     "    with torch.no_grad():\n"
     "        fan, u0 = C.native_sample(dec, inp, eps)\n"
     "    assert torch.equal(u0, u0_ref)\n"
     "    assert torch.equal(fan, fan_ref)\n"
     "    assert torch.equal(out[\"u0_hat\"], u0_ref)         # ABSOLUTE controls, as emitted\n"
     "\n"
     "\n"
     "def test_NEW1_parity_RESIDUAL_holds_on_a_WITHHELD_row_where_the_prior_speed_is_not_v0():\n"
     "    \"\"\"On a withheld row the prior is zeroed and rolled at the bank's withheld speed, NOT the\n"
     "    raw v0 -- the hook's pre-NEW-1 rule (``v = v_ms``) would break parity exactly here.\"\"\"\n"
     "    dec = _dec(**RESIDUAL)\n"
     "    inp, _ = _captured(dec, residual_prior=_prior(), ego_keep=torch.tensor([True, False]))\n"
     "    a0, k0, pv = inp.prior\n"
     "    assert float(a0[1]) == 0.0 and float(k0[1]) == 0.0\n"
     "    assert float(pv[1]) != float(inp.v_ms[1])         # the withheld row is not rolled at v0\n"
     "    with torch.no_grad():\n"
     "        fan, u0 = C.native_sample(dec, inp, inp.replay_eps(dec))\n"
     "    assert torch.equal(u0, inp.out[1]) and torch.equal(fan, inp.out[0])\n"
     "\n"
     "\n"
     "def test_NEW1_REGRESSION_a_chain_that_rolls_Delta_without_the_prior_breaks_parity(monkeypatch):\n"
     "    dec = _dec(**RESIDUAL)\n"
     "    inp, _ = _captured(dec, residual_prior=_prior())\n"
     "    eps = inp.replay_eps(dec)\n"
     "    real_roll = dec._roll_state\n"
     "\n"
     "    def no_prior(z, v, metre, prior):                 # the mutant: Delta rolled as the plan\n"
     "        zero = (torch.zeros_like(prior[0]), torch.zeros_like(prior[1]), prior[2])\n"
     "        return real_roll(z, v, metre, zero)\n"
     "    monkeypatch.setattr(dec, \"_roll_state\", no_prior)\n"
     "    with torch.no_grad():\n"
     "        fan, _ = C.native_sample(dec, inp, eps)\n"
     "    assert not torch.equal(fan, inp.out[0])\n"
     "\n"
     "\n"
     "def test_NEW1_REGRESSION_a_capture_that_loses_the_prior_is_REFUSED_not_rolled():\n"
     "    dec = _dec(**RESIDUAL)\n"
     "    inp, _ = _captured(dec, residual_prior=_prior())\n"
     "    eps = inp.replay_eps(dec)\n"
     "    with pytest.raises(D.Ddv2ConfigError, match=\"RESIDUAL\"):\n"
     "        C.native_sample(dec, dataclasses.replace(inp, prior=None), eps)\n"
     "    wrong_v = dataclasses.replace(inp, v=torch.full_like(inp.v, dec.anchor_ref_speed))\n"
     "    with pytest.raises(D.Ddv2ConfigError, match=\"roll speed\"):\n"
     "        C.native_sample(dec, wrong_v, eps)\n"
     "    off = _dec()\n"
     "    inp_off, _ = _captured(off)\n"
     "    with pytest.raises(D.Ddv2ConfigError, match=\"residual_prior='off'\"):\n"
     "        C.state_to_path(off, C.anchor_state(off, 2), inp_off.v, inp.prior)\n"
     "\n"
     "\n"
     "def test_NEW1_groups_are_independent_queries_on_a_RESIDUAL_build():\n"
     "    whole, whole_others, parts = _group_independence(_dec(**RESIDUAL),\n"
     "                                                     residual_prior=_prior())\n"
     "    assert torch.equal(whole[:, :5], whole_others[:, :5])\n"
     "    assert torch.allclose(whole, parts, rtol=1e-5, atol=1e-6)\n")

ADMISS = "stack/tanitad/channel_admissibility.py"

edit(ADMISS, "admiss.seam_module",
     "    \"tanitad.models.refcv6_perception_branch\",   # perception_grid / _valid\n"
     ")\n",
     "    \"tanitad.models.refcv6_perception_branch\",   # perception_grid / _valid\n"
     "    # ⭐ refcv7 NEW-1 (2026-09-27): the residual prior added `ego_actions` to BOTH\n"
     "    # forwards, and the Thor full-suite gate went RED on it in six RL guard files --\n"
     "    # correctly. It is read ONLY by `--residual-prior ha0_ext`, which SPEC_REFCV7\n"
     "    # section 10 (A5) excludes: the recorded steer at t0 is unruled at inference.\n"
     "    # ⚠️ TEMPORARY and NOT a label -- the route back is in the declaration.\n"
     "    \"tanitad.models.kinematic_prior\",            # ego_actions\n"
     ")\n")

#: files an anchored edit targets that may legitimately be ABSENT at the base (they land
#: in an earlier batch); absence is reported, never silently treated as success.
OPTIONAL_FILES = (DVB, DVB_TEST)

# how many times each (file, id) anchor must occur; default 1
COUNTS: dict = {}


def _read_blob(git_dir: str, rev: str, path: str) -> bytes:
    return subprocess.run(["git", f"--git-dir={git_dir}", "show", f"{rev}:{path}"],
                          check=True, capture_output=True).stdout


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def apply(base: bytes, path: str) -> tuple[bytes, list[str]]:
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
        want = COUNTS.get((f, eid), 1)
        got = text.count(old)
        if got != want:
            raise SystemExit(f"ANCHOR FAILED {path} [{eid}]: found {got}, need {want}:\n{old[:300]!r}")
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
    man = {"rev": a.rev, "files": {}, "skipped_absent": []}
    for path in (REFC, REFCV3, TRAIN, DVB, DVB_TEST, CHAIN, CHAIN_TEST, ADMISS):
        try:
            if a.base_git:
                if not a.rev:
                    raise SystemExit("--rev is required with --base-git")
                base = _read_blob(a.base_git, a.rev, path)
            else:
                base = open(os.path.join(a.base_dir, path), "rb").read()
        except (subprocess.CalledProcessError, FileNotFoundError):
            if path in OPTIONAL_FILES:
                print(f"{path}: ABSENT at this base -- SKIPPED (it lands with the fixes "
                      f"batch; re-run this script on that tip)")
                man["skipped_absent"].append(path)
                continue
            raise
        new, done = apply(base, path)
        dst = os.path.join(a.out, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(new)
        man["files"][path] = {"base_blob_sha1": git_blob_sha1(base),
                              "out_blob_sha1": git_blob_sha1(new),
                              "base_lines": base.count(b"\n"),
                              "out_lines": new.count(b"\n"),
                              "eol": "CRLF" if base.count(b"\r\n") else "LF",
                              "edits": done}
        print(f"{path}: {len(done)} edits, {base.count(b'\n')} -> {new.count(b'\n')} lines")
    with open(os.path.join(a.out, "EDIT_MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
