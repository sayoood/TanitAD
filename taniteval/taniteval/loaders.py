"""TanitEval — model loaders.

Rebuilds each architecture to match its checkpoint (strict load) and returns a
uniform handle: model + the grounded step-readout the rollout engine decodes
with + which episode view it consumes (frames vs frozen features).
  flagship  : WorldModel(flagship4b)          readout = ck['grounding'].step['op']
  refa-plus : RefAModelPlus (temporal, d_dino) readout = ck['step_readout']
  refb      : RefBModel, cfg reconstructed from the run's config.json (the
              stack default drifted from the trained depth-25 encoder)
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import torch

# ⛔ was: sys.path.insert(0, "/root/TanitAD/stack"[/scripts]) — that put a
# possibly PRE-v5 tree IN FRONT of the caller's PYTHONPATH and published a
# plausible wrong number instead of an error (STALE_IMPORT_GUARD.md).
from taniteval.stack_guard import ensure_stack_on_path as _ensure_stack  # noqa: E402
_ensure_stack()
from tanitad.config import flagship4b_config  # noqa: E402


def _apply_overrides(cfg_obj, d: dict):
    """Recursively push a config.json dict onto a (frozen) dataclass tree."""
    for k, v in d.items():
        if not hasattr(cfg_obj, k):
            continue
        cur = getattr(cfg_obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _apply_overrides(cur, v)
        elif isinstance(v, (int, float, bool, str)) or v is None:
            try:
                object.__setattr__(cfg_obj, k, type(cur)(v) if cur is not None
                                   else v)
            except Exception:
                object.__setattr__(cfg_obj, k, v)
        elif isinstance(v, list) and not dataclasses.is_dataclass(cur):
            object.__setattr__(cfg_obj, k, tuple(v) if isinstance(cur, tuple)
                               else v)
    return cfg_obj


def resolve_labels_v2(entry, cfg) -> bool:
    """Which manoeuvre-label family was this arm TRAINED on? ``cfg.v2_labels``.

    ⛔ ONE definition, and the defect that earned it. The trainer's label family
    is ``cfg.v2_labels`` (``StackConfig``; set by ``--v2``, overridden by
    ``--labels-v2`` / ``--no-labels-v2`` — ``train_flagship4b.main``), and the
    loader already rebuilds each arm's ``cfg`` from its OWN ``config.json``. It
    simply never handed it to the eval, so ``taniteval.hierarchy`` hardcoded the
    v1 labeler and mis-scored every ``--v2`` arm's ``maneuver_acc``. This reads
    the arm's own field — it does NOT re-derive the rule.

    A registry entry MAY declare ``labels_v2`` (for arms whose ``cfg`` the
    loader cannot reconstruct, e.g. ``refa-plus``). When it declares one AND the
    rebuilt cfg has one, they must AGREE: a silent disagreement here is the exact
    failure mode this function exists to end, so it raises instead.
    """
    from_cfg = getattr(cfg, "v2_labels", None)
    declared = entry.get("labels_v2")
    if declared is not None and from_cfg is not None \
       and bool(declared) != bool(from_cfg):
        raise ValueError(
            f"labels_v2 disagreement for arm {entry.get('key')!r}: registry "
            f"entry says {bool(declared)}, the run's own config.json says "
            f"{bool(from_cfg)}. Fix the registry — the run config wins on "
            "facts, and a maneuver_acc scored under the wrong family is not a "
            "measurement.")
    return bool(from_cfg if declared is None else declared)


def load(entry, device="cuda"):
    ck = torch.load(entry["ckpt"], map_location="cpu", weights_only=False)
    arch, grounding, step_readout, feed = entry["arch"], None, None, "frames"
    cfg = None

    if arch == "flagship-worldmodel":
        from tanitad.models.fourbrain import WorldModel
        from tanitad.models.metric_dynamics import HierarchicalGrounding
        cfg = flagship4b_config()
        if entry.get("speed_input"):
            object.__setattr__(cfg.predictor, "action_dim", 3)
            if cfg.tactical_pred is not None:
                object.__setattr__(cfg.tactical_pred, "action_dim", 3)
        model = WorldModel(cfg)
        model.load_state_dict(ck["model"])
        grounding = HierarchicalGrounding(model.state_dim)
        grounding.load_state_dict(ck["grounding"])
        grounding = grounding.to(device).eval()
        step_readout = grounding.step["op"]

    elif arch == "flagship-worldmodel-v2":
        # TEMP assess 2026-07-19: v2 arch (anchored tactical decoder, gated
        # intent, ego->planners, route-from-vision, encoder-ego decorr) — the
        # model is rebuilt from the RUN'S OWN config.json (full cfg dump incl.
        # the v2_* flags) so the state_dict loads STRICT. Requires
        # TANITEVAL_STACK_OVERRIDE -> the training pod's stack copy.
        from tanitad.models.fourbrain import WorldModel
        from tanitad.train.flagship_losses import build_grounding
        cfg = flagship4b_config()
        rc = json.loads(Path(entry["run_config"]).read_text())
        _apply_overrides(cfg, json.loads(rc["cfg"]))
        # eval-only: no autograd -> checkpointing is pure warning noise
        object.__setattr__(cfg.encoder, "grad_checkpoint", False)
        model = WorldModel(cfg)
        model.load_state_dict(ck["model"])                    # STRICT
        grounding = build_grounding(model.state_dim)
        grounding.load_state_dict(ck["grounding"])            # STRICT
        grounding = grounding.to(device).eval()
        step_readout = grounding.step["op"]

    elif arch == "refa-plus":
        from refa_plus import RefAModelPlus
        from tanitad.models.metric_dynamics import StepDisplacementReadout
        cfg = flagship4b_config()
        # action_dim = 2 base (steer, accel) + v0 (--speed-input) + yr0
        # (--yaw-input / --dyn-input) — exactly refa_train_plus's adim formula
        # (dyn-input arm = 4). Keep tactical_pred in lockstep.
        adim = 2 + int(bool(entry.get("speed_input"))) + \
            int(bool(entry.get("yaw_input") or entry.get("dyn_input")))
        if adim != cfg.predictor.action_dim:
            object.__setattr__(cfg.predictor, "action_dim", adim)
            if cfg.tactical_pred is not None:
                object.__setattr__(cfg.tactical_pred, "action_dim", adim)
        model = RefAModelPlus.from_stack_config(
            cfg, n_tokens=256, adapter_kind=entry.get("adapter", "temporal"),
            d_dino=entry.get("d_dino", 768))
        model.load_state_dict(ck["model"])
        step_readout = StepDisplacementReadout(model.state_dim)
        step_readout.load_state_dict(ck["step_readout"])
        step_readout = step_readout.to(device).eval()
        feed = entry.get("feat_kind", "dinov2")        # frozen-feature episodes

    elif arch == "refb":
        from tanitad.refs.refb import RefBModel, refb_config
        cfg = refb_config()
        cj = Path(entry["ckpt"]).parent / "config.json"
        if cj.exists():
            _apply_overrides(cfg, json.loads(cj.read_text()).get("cfg", {}))
        model = RefBModel(cfg)
        if entry.get("strict"):
            # TEMP assess 2026-07-19: v2 milestones are fully config-driven
            # (RefBModel(cfg) builds every gated module) -> STRICT load; a
            # mismatch must raise with the exact missing/unexpected keys.
            model.load_state_dict(ck["model"])
        else:
            # trained v1 run added speed_emb/accel_head OUTSIDE the stack
            # build; nothing is missing, so load non-strict, record extras.
            extra = model.load_state_dict(ck["model"], strict=False)
            assert not extra.missing_keys, \
                f"refb missing: {extra.missing_keys[:4]}"
        feed = "frames"                                 # planner: no rollout yet

    elif arch == "refc":
        # REF-C (Anchored-Diffusion-C): a DiffusionDrive-style ANCHORED
        # trajectory decoder (anchor queries cross-attend the conv map -> conf +
        # offset; truncated-diffusion refines the winning modes), NOT a grounded
        # operative rollout. Rebuild from the scale preset named in the entry,
        # then push the run's OWN config.json (the full cfg dump) so every gated
        # graft (imagination / hierarchy / maneuver->anchor) AND the anchor
        # buffer are constructed at the trained shape and the state_dict loads
        # STRICT. step_readout stays None: the trajectory surface is the decoder's
        # own anchor head, evaluated by taniteval.refc_eval (direct decode).
        from tanitad.refs.refc import (RefCModel, refc_config,
                                       refc_small_config, refc_smoke_config,
                                       refc_xl_config)
        _presets = {"small": refc_small_config, "base": refc_config,
                    "xl": refc_xl_config, "smoke": refc_smoke_config}
        cfg = _presets[entry.get("config_preset", "xl")]()
        cj = Path(entry["ckpt"]).parent / "config.json"
        if cj.exists():
            _apply_overrides(cfg, json.loads(cj.read_text()).get("cfg", {}))
        model = RefCModel(cfg)
        model.load_state_dict(ck["model"])                    # STRICT
        feed = "frames"                                       # raw-frame input
    else:
        raise ValueError(arch)

    model = model.to(device).eval()
    return dict(model=model, grounding=grounding, step_readout=step_readout,
                feed=feed, step=ck.get("step"),
                state_dim=getattr(model, "state_dim", None),
                traj_capable=step_readout is not None,
                # The arm's OWN label family, for every panel that scores a
                # manoeuvre head against a derived label (hierarchy.run).
                cfg=cfg, labels_v2=resolve_labels_v2(entry, cfg))
