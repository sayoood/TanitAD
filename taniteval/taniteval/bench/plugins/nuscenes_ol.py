"""PLUGIN ``nuscenes_ol`` — W6 (EvalFlyWheel), replacing W1's docstring stub (2026-09-19).

    python -m taniteval.bench nuscenes_ol --ckpt none --split val --device cpu
        env TANITAD_NUSCENES_PROTOCOL = nuScenes_OL_L2_uniad | nuScenes_OL_L2_stp3   (REQUIRED)
        env TANITAD_NUSCENES_ROOT     = <devkit layout root>  (default D:/Archive/devbox-C/nuscenes/data)
        --arms GT,STOP,CV[,A2_vision_pure,A3_ego_nocmd]      (GT/STOP/CV are always added)

⛔ ``claim_bearing`` is FALSE and cannot be set otherwise: nuScenes open-loop planning is
inadmissible as a TanitAD criterion (``H-EVAL-6``) and SKIP claim-bearing (``D-BENCH-PORT``).

ONE convention per run (W1 contract + BUILD_PLAN §1): ``nuScenes_OL_L2_uniad`` = UniAD's pipeline
with the value AT t; ``nuScenes_OL_L2_stp3`` = VAD's pipeline with the mean UP TO t. The protocol
has NO default — a convention the operator did not choose is how two conventions end up in one
column. Until W1's parser grows ``--protocol`` / ``--nuscenes-root`` (integration ask in the W6
COMMS), both come from the environment; ``ctx.args.protocol`` / ``ctx.args.nuscenes_root`` win
when present.

Everything numeric is ``taniteval/adapters/nuscenes_planning.py`` (verbatim kernels + reductions,
pinned; tests ``taniteval/tests/test_nuscenes_planning.py``). This file only maps it onto
:class:`taniteval.bench.contract.BenchContext`.

Arms: ``GT`` (reference — the logged trajectory; its RAW collision is the instrument's floor),
``STOP`` and ``CV`` (floors), and with ``--ckpt`` the model arms ``A2_vision_pure`` (frames only —
the arm our vision-only rule admits; default) and ``A3_ego_nocmd`` (+ measured t0 ego state).
⛔ No command arm: nuScenes' command is the GT future thresholded at +-2 m — refused.
GPU only through ``ctx.gpu_device()``; scoring is CPU.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

DEFAULT_NUSCENES_ROOT = "D:/Archive/devbox-C/nuscenes/data"
ENV_ROOT = "TANITAD_NUSCENES_ROOT"
ENV_PROTOCOL = "TANITAD_NUSCENES_PROTOCOL"
ENV_CONSTRUCTION = "TANITAD_NUSCENES_HISTORY"          # ST (default) | NT | EXACT
ALWAYS_ARMS = ("GT", "STOP", "CV")


def _adapter():
    """``adapters.nuscenes_planning`` — ``<repo>/taniteval`` must be importable as a root."""
    te = Path(__file__).resolve().parents[3]           # plugins -> bench -> taniteval -> taniteval/
    if str(te) not in sys.path:
        sys.path.insert(0, str(te))
    from adapters import nuscenes_planning as NP
    return NP


def _refuse(msg: str):
    from ..navsim.profiles import Refusal
    raise Refusal(msg)


def run_benchmark(ctx) -> None:
    NP = _adapter()
    a = ctx.args
    ctx.set_claim_bearing(NP.CLAIM_BEARING)            # False, recorded even on a refusal
    ctx.rec["claim_bearing_reason"] = NP.CLAIM_BEARING_REASON
    protocol = getattr(a, "protocol", None) or os.environ.get(ENV_PROTOCOL)
    if not protocol:
        _refuse(f"nuscenes_ol runs ONE convention per run and has no default: set {ENV_PROTOCOL}="
                f"nuScenes_OL_L2_uniad (UniAD, value AT t) or nuScenes_OL_L2_stp3 (VAD pipeline, "
                f"mean UP TO t)")
    if protocol not in NP.PROTOCOLS:
        _refuse(f"{protocol!r} is not a nuScenes protocol; closed set {sorted(NP.PROTOCOLS)}")
    ctx.set_protocol(protocol)
    ctx.set_stamps("OPEN-LOOP-L2", {"planner": "OPEN"}, "MEASURED", closed_loop=False)
    root = getattr(a, "nuscenes_root", None) or os.environ.get(ENV_ROOT) or DEFAULT_NUSCENES_ROOT
    try:
        meta, scenes, n_logs = NP.load_split(root, a.split)
    except Exception as e:                              # noqa: BLE001 — NuScenesTermsError / RefusedInput
        _refuse(f"nuScenes data unavailable at {root} ({type(e).__name__}): {str(e)[:1500]}")
    requested = [x for x in (a.arms or [])]
    builtin = list(ALWAYS_ARMS)
    model_names = [x for x in requested if x not in NP.ARM_SPECS]
    for m in model_names:
        if m in NP.REFUSED_MODEL_ARMS:
            _refuse(f"model arm {m} refused: {NP.REFUSED_MODEL_ARMS[m]}")
        if m not in NP.MODEL_ARM_SPECS:
            _refuse(f"unknown arm {m!r}; built-ins {sorted(NP.ARM_SPECS)}, model arms "
                    f"{sorted(NP.MODEL_ARM_SPECS)}")
    if model_names and a.ckpt is None:
        _refuse(f"model arms {model_names} need --ckpt")
    if a.ckpt is not None and not model_names:
        model_names = ["A2_vision_pure"]
    model_arms = {}
    if model_names:
        # CPU inference (W1's default); a GPU forward would go through ctx.gpu_device() - not built
        forward, window, prov = NP.refc_forward(a.ckpt)
        ctx.rec["model"] = dict(prov, device_used="cpu")
        construction = getattr(a, "construction", None) or os.environ.get(ENV_CONSTRUCTION) or "ST"
        for m in model_names:
            model_arms[m] = NP.make_model_arm(m, nuscenes_root=root, forward=forward,
                                              window_rows=window, construction=construction)
    ev = NP.evaluate_arms(meta, scenes, protocol, builtin, model_arms=model_arms)
    out = NP.build_run_outputs(ev, split_name=a.split, n_scenes=len(scenes), n_logs=n_logs)
    ctx.set_devkit(out["devkit"]["repo"], out["devkit"]["sha"], out["devkit"]["patches"],
                   reference_pins=NP.PINS)
    ctx.set_split(a.split, len(scenes), n_logs,
                  **{k: v for k, v in out["split"].items() if k not in ("name", "n_scenes", "n_logs")})
    ctx.rec["protocol_tag"] = out["tag"].to_dict()
    ctx.rec["nuscenes_root"] = str(root).replace(os.sep, "/")
    for arm, o in out["arms"].items():
        m = o["meta"]
        ctx.add_arm(arm, m["kind"], m["declared_inputs"], status="OK",
                    added_by_rule=arm not in requested)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / f"{arm}.csv"
            NP._write_csv(p, o["csv_rows"])
            ctx.write_scores(arm, p)
        ctx.write_artifact(arm, o["artifact"])
        ctx.run_criteria(arm)
    ctx.write_summary(dict(out["summary"], run_id=ctx.run_id))
    ctx.rec["status"] = "COMPLETE"
