"""Shared seams for the F-18 slot probe.

⛔ WHY `merge_run_args` EXISTS. The live v6F run started before several CLI
levers were added, so its banked ``config["args"]`` is a STRICT SUBSET of what
today's ``build_stack_from_args`` reads (`AttributeError: 'Namespace' object has
no attribute 'selector'`). Rebuilding from stock defaults alone would be the
error ``v6_probe_trunk`` refuses by design; rebuilding from the RUN's args alone
crashes. The correct composition is **parser defaults UNDERNEATH, the run's own
args ON TOP** — every value the run recorded wins, and only genuinely new levers
fall back to their default. The strict state-dict load is what proves the result
is the trained architecture.
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

STACK = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack")
for _p in (str(STACK), str(STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def parser_defaults(stage: str = "S-W") -> dict:
    """Every CLI lever's DEFAULT, read off the parser's actions.

    ⚠️ Not ``parse_args([...])`` — the v6 trainer's parser has REQUIRED options
    (``--out``, ``--stage``), so parsing an empty argv exits the process with a
    usage message that names *this* script. Reading ``action.default`` gets the
    same mapping with no required-argument surface at all.
    """
    import argparse

    from train_v6_staged import build_parser
    d = {}
    for act in build_parser()._actions:
        if act.dest in ("help",) or act.dest is argparse.SUPPRESS:
            continue
        d[act.dest] = act.default
    d["stage"] = stage
    return d


def merge_run_args(run_args: dict, **overrides) -> Namespace:
    """Parser defaults <- the RUN's recorded args <- explicit overrides."""
    base = parser_defaults(str(run_args.get("stage", "S-W")))
    merged = {**base, **{k: v for k, v in run_args.items() if v is not None},
              **overrides}
    # keys the run recorded as null are still meaningful (e.g. v2_subframe)
    for k, v in run_args.items():
        if v is None and k in base:
            merged[k] = None
    merged.update(overrides)
    return Namespace(**merged)


def read_fp16_snapshot(path: str | Path, config_json: str | Path | None = None):
    """``weights_fp16.pt`` -> ``(state_dict, run_args, step, source)``.

    The layout is the trainer's own snapshot contract
    (``train_v6_staged``: ``{"model": sd, "_meta": {...},
    "_fp16_weights_only": True}``) — the path a rebuilt pod actually takes,
    because the 3.53 GB ``ckpt.pt`` is the artifact that does not travel.

    ⚠️ STATED, never hidden: fp16 round-trips the weights through half
    precision, so this trunk cannot be bit-identical to the fp32 source. The
    trainer's own seam says the same thing ("init_precision": "fp16->fp32
    (lossy)"). Every number produced from it is stamped accordingly.
    """
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    snap = bool(ck.get("_fp16_weights_only")) or (
        "model" in ck and "stack" not in ck and "_meta" in ck)
    meta = (ck.get("_meta") or {}) if snap else ck
    sd = ck["model"] if snap else ck.get("stack", ck)
    args = (meta.get("config") or {}).get("args")
    src = "ckpt['_meta']['config']['args']" if args else None
    if not args and config_json:
        args = json.loads(Path(config_json).read_text("utf-8"))["args"]
        src = f"sidecar {Path(config_json).name}"
    if not args:
        raise SystemExit(
            "[sp] the snapshot carries no run config and no sidecar was given "
            "— refusing to rebuild the architecture from defaults.")
    step = int(meta.get("step", -1))
    return sd, args, step, {
        "layout": "fp16_weights_only_snapshot" if snap else "ckpt",
        "precision": "fp16->fp32 (lossy)" if snap else "fp32",
        "args_source": src,
        "step_source": "ckpt['_meta']['step']" if snap else "ckpt['step']",
    }
