#!/usr/bin/env python3
"""v8F-mini training (track M): S -> T -> O on the cached frozen DINOv3-L pooled features. CPU-testable, 8 GB GPU-sized.

Seeded, resumable, writes ``summary.json {"done": true}`` ONLY at the very end (a supervisor that restarts a run whose
done-marker is missing will resurrect it; one that finds the marker leaves it alone).

LAUNCH-GATE DISCIPLINE (the project rule of 2026-09-26): every argparse flag has an entry in :data:`FLAG_CHECKS`; a flag
without one makes ``self_check`` REFUSE the launch (and ``tests/test_v8f_mini.py`` fail). ``declared_vs_built``
compares what argv DECLARES against what the BUILT model/optimiser/data actually are; ``live_check`` runs three
optimiser steps on a real batch and requires every declared loss term finite and every trainable parameter group to
receive a non-zero gradient, against a control (the frozen P10 prior) that must read EXACTLY 0.0.

Arms (PREREG_V8F_MINI.md): M1, M1-navonly, M1-nocons, M2a, M2b are trained arms. ``M1-navshuf`` is an EVALUATION control
(nav taken from another episode): it trains exactly M1 and is accepted here as an alias so the supervisor can name it.
H-DATA-SCALE-1: ``--data-frac {0.25,0.5,1.0}`` and ``--steps-mult {1,2,4}``; S = one epoch of windows at 100 % of the
TRAIN clips at the chosen batch size, the optimiser-step budget is ``S * steps-mult`` regardless of the data fraction.

PREREG_V8F_LORA: ``--encoder lora`` runs the frame encoder (DINOv3 ViT-L/16, frozen, + LoRA r/alpha on q and v of every block) ON THE FLY on the frames
behind the cache (``--frames-root``: ``<clip>.v2ep.pt``) and feeds the heads exactly the tensor the cache would have held (``tanitad.models.v8f_lora_encoder``).
``--encoder cache`` (the default) is every earlier arm, bit for bit: none of the lora code runs, and nothing it imports is imported. ``--accum k`` splits the
ENCODER's work of one effective batch into micro-batches of ``batch / k`` windows; the heads' loss is still evaluated once on the full batch (the gradient-cache
scheme of ``cached_grad_step``), because plain accumulation is not exact for this model (masked-mean normalisers, SIGReg).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import threading
import queue
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

PKG = os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini")

ARM_SPECS = {
    "M1": dict(navonly=False, consistency=True, wm="off"),
    "M1-navshuf": dict(navonly=False, consistency=True, wm="off"),      # trained as M1; the control is at EVAL time
    "M1-navonly": dict(navonly=True, consistency=True, wm="off"),
    "M1-nocons": dict(navonly=False, consistency=False, wm="off"),
    "M2a": dict(navonly=False, consistency=True, wm="a"),
    "M2b": dict(navonly=False, consistency=True, wm="b"),
    # PREREG_V8F_CZ: M2b with ONE lever moved -- T's target for an ABSENT constraint is the 0 O is teacher-forced on
    "M2b-cz": dict(navonly=False, consistency=True, wm="b", cons_absent="zero"),
}

#: dest -> how it is verified. "cfg:<field>" = equals the BUILT config; "lever:<k>" = equals built_levers()[k];
#: "opt:<k>" = read back from the optimiser / scheduler; "data:<k>" = read back from the built data; "run" = a
#: run-control value recorded in summary.json; "path" = an input/output location asserted to exist (or be created).
FLAG_CHECKS = {
    "arm": "lever:arm", "navonly": "lever:navonly", "no_consistency": "lever:consistency", "wm": "lever:wm",
    "cache_dir": "path", "out": "path", "v9_train": "path", "v9_train_md5": "data:v9_md5", "speed_train": "path",
    "lead_train": "path", "join_train": "path", "p10": "path", "grid": "cfg:grid",
    "data_frac": "data:n_clips", "steps_mult": "run", "steps": "run", "batch": "data:batch", "seed": "run",
    "lr": "opt:lr", "wd": "opt:wd", "warmup": "opt:warmup", "device": "run", "amp": "run", "lru": "data:lru",
    "ckpt_every": "run", "log_every": "run", "resume": "run", "no_resume": "run", "max_clips": "data:n_clips", "exclude": "data:n_excluded",
    "hist": "cfg:hist", "hmax": "cfg:hmax", "d": "cfg:d", "n_q": "cfg:n_q", "nb": "cfg:nb",
    "window_start": "lever:window_start",     # PREREG_V8F_HIST Amendment 1: cfg + built_levers + the WINDOW LIST the data built + the dataset's clamp (declared_vs_built)
    "p_gt_start": "cfg:p_gt_start", "p_gt_end": "cfg:p_gt_end", "w_ckpt": "cfg:w_ckpt", "w_tac": "cfg:w_tac",
    "w_cons": "cfg:w_cons", "w_traj": "cfg:w_traj", "w_pen": "cfg:w_pen", "w_pen_self": "cfg:w_pen_self",
    "w_wm": "cfg:w_wm", "detach_upper": "cfg:detach_upper", "synthetic": "run", "selfcheck_only": "run",
    "fit_join": "run", "workers": "run", "clip_grad": "opt:clip_grad", "threads": "run",
    "kin_mode": "cfg:kin_mode", "tracks": "path", "cons_absent": "lever:cons_absent",
    "code_mix": "lever:code_mix",             # PREREG_V8F_MIXO: argv -> make_config -> the BUILT model's levers (declared_vs_built) -> the probe of the code O actually read (code_mix_probe)
    # PREREG_V8F_LORA: the encoder lever. `enc:<k>` = equals the BUILT encoder's lever (enc.built_levers()); with --encoder cache there is no encoder and every
    # one of these must be at its default (a declared flag that builds nothing is REFUSED, not ignored)
    "encoder": "enc:encoder", "lora_r": "enc:lora_r", "lora_alpha": "enc:lora_alpha", "lora_targets": "enc:lora_targets",
    "lora_lr": "opt:lora_lr", "frames_root": "data:frames_missing", "accum": "enc:accum", "enc_ckpt": "enc:checkpointing",
}

LORA_PARSER_DEFAULTS = {"lora_r": 32, "lora_alpha": 32.0, "lora_targets": "q,v", "lora_lr": None, "frames_root": None, "accum": 1, "enc_ckpt": False}
LIVE_PROBE_WINDOWS_LORA = 16          # G-LIVE through the real encoder: windows of the first batch (<= 16 x hmax frames), not all 64


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=sorted(ARM_SPECS), default="M1")
    ap.add_argument("--navonly", action="store_true", default=None, help="override: S and T lose the scene")
    ap.add_argument("--no-consistency", action="store_true", default=None, help="override: penalties at 0")
    ap.add_argument("--wm", choices=("off", "a", "b"), default=None, help="override: H-JEPA auxiliary (a: no IDM, b: IDM)")
    ap.add_argument("--cons-absent", choices=("mask", "zero"), default=None,
                    help="override (PREREG_V8F_CZ): mask = an absent constraint is out of T's loss (every arm but M2b-cz); zero = T's "
                         "target there is the 0 that O reads under teacher forcing")
    ap.add_argument("--code-mix", dest="code_mix", choices=M.CODE_MIX_MODES, default="whole",
                    help="PREREG_V8F_MIXO: the granularity of O's teacher forcing of T's code. whole (default) = ONE GT-or-own switch per sample over the whole code (every earlier arm, bit for bit, "
                         "the RNG stream included); kind = an independent switch per sample AND per unit (the decisions + the six constraint kinds), at the same p_gt")
    ap.add_argument("--cache-dir", default="C:/Users/Admin/refav1_hj_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--v9-train", default="C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_train.npz")
    ap.add_argument("--v9-train-md5", default="f63ece410b725febb8a5242cf2b01d3c")
    ap.add_argument("--speed-train", default="C:/Users/Admin/qland/landed_hold_datacov/raw/refcv6_speed_max_v8_train.jsonl")
    ap.add_argument("--lead-train", default="D:/Projects/TanitAD-artifacts/v9labels/inputs/lead_train.npz")
    ap.add_argument("--join-train", default=None, help="a prebuilt join npz (else built from the v9 release)")
    ap.add_argument("--fit-join", action="store_true", help="(re)build the join even if --join-train exists")
    ap.add_argument("--p10", default=None, help="the frozen P10 prior; default = the one for --kin-mode (yaw: raw/p10_yaw, steer: raw/p10). "
                    "Its declared units must match --kin-mode or the launch is REFUSED")
    ap.add_argument("--kin-mode", choices=D.KIN_MODES, default="yaw",
                    help="PI ruling D-PI-YAWRATE-1 (2026-10-17): yaw (default) = the model gets the MEASURED yaw rate (P10 features, kd_x base and "
                         "the WM action channel 1 from the native pose track); steer = the legacy recorded-steer representation")
    ap.add_argument("--tracks", default=None, help="native_tracks.pt (build_native_tracks.py): REQUIRED for --kin-mode yaw (the cache holds only "
                    "the even frames); the synthetic world writes its own")
    ap.add_argument("--grid", choices=("pool4x10", "pool8x20"), default="pool4x10")
    ap.add_argument("--data-frac", type=float, default=1.0)
    ap.add_argument("--max-clips", type=int, default=0, help="smoke: cap the TRAIN clips (0 = all)")
    ap.add_argument("--exclude", default="", help="A:B -- DROP clips A..B of the TRAIN manifest's sha12 order before anything "
                    "else: the run never sees them, and S is counted without them (PREREG_V8F_MINI Amendment 4: 540:600 = the "
                    "RA-1 60-clip selection holdout, so v8F-mini trains on exactly RA-1 full's 4,512 clips)")
    ap.add_argument("--steps-mult", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="override the step budget (smoke)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip-grad", dest="clip_grad", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (UNVERIFIED on GPU at this commit)")
    ap.add_argument("--lru", type=int, default=0, help="clips held in RAM (0 = auto from the grid size)")
    ap.add_argument("--workers", type=int, default=1, help="1 = a single prefetch thread")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--hist", type=int, default=4)
    ap.add_argument("--window-start", dest="window_start", type=int, default=-1,
                    help="PREREG_V8F_HIST Amendment 1: the first TRAINING window's cache step t of every clip AND the origin of the held-age clamp "
                         "(room = max(t - window_start, 0)). -1 = absent = hist - 1 (every earlier arm, bit for bit); else 0 <= window_start <= hist - 1. "
                         "`--hist 10 --window-start 3` trains B's window list and B's held ages over 10 frames, the ones before the clip padded")
    ap.add_argument("--hmax", type=int, default=0, help="0 = auto: 10 (M1) or 16 (M2)")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n-q", dest="n_q", type=int, default=4)
    ap.add_argument("--nb", type=int, default=12)
    ap.add_argument("--p-gt-start", type=float, default=1.0)
    ap.add_argument("--p-gt-end", type=float, default=0.2)
    ap.add_argument("--w-ckpt", type=float, default=1.0)
    ap.add_argument("--w-tac", type=float, default=1.0)
    ap.add_argument("--w-cons", type=float, default=0.5)
    ap.add_argument("--w-traj", type=float, default=1.0)
    ap.add_argument("--w-pen", type=float, default=0.5)
    ap.add_argument("--w-pen-self", type=float, default=0.25)
    ap.add_argument("--w-wm", type=float, default=0.1)
    ap.add_argument("--detach-upper", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="run on a tiny synthetic world (smoke / CI)")
    ap.add_argument("--selfcheck-only", action="store_true")
    # ---- PREREG_V8F_LORA (M4): LoRA on the frozen DINOv3-L, encoded on the fly. Every flag has its G-DVB entry in FLAG_CHECKS.
    ap.add_argument("--encoder", choices=("cache", "lora"), default="cache",
                    help="cache (default) = the cached frozen pooled tokens, every earlier arm bit for bit; lora = DINOv3-L/16 + LoRA run on the fly on --frames-root")
    ap.add_argument("--lora-r", dest="lora_r", type=int, default=32, help="LoRA rank (PREREG_V8F_LORA: 32)")
    ap.add_argument("--lora-alpha", dest="lora_alpha", type=float, default=32.0, help="LoRA alpha (PREREG_V8F_LORA: 32; scaling = alpha / r)")
    ap.add_argument("--lora-targets", dest="lora_targets", default="q,v", help="comma list among q,k,v,o (PREREG_V8F_LORA: q,v of all 24 blocks)")
    ap.add_argument("--lora-lr", dest="lora_lr", type=float, default=None, help="the LoRA matrices' peak learning rate (default: --lr, the heads')")
    ap.add_argument("--frames-root", dest="frames_root", default=None, help="--encoder lora: directory of <clip>.v2ep.pt (256x640 cylindrical PNG frames) for EVERY train clip")
    ap.add_argument("--accum", type=int, default=1, help="--encoder lora: encoder micro-batches per effective batch (batch / accum windows each); the heads see the full batch")
    ap.add_argument("--enc-ckpt", dest="enc_ckpt", action="store_true", help="--encoder lora: gradient checkpointing inside the encoder (needed for 8 GB)")
    return ap


# ------------------------------------------------------------------------------------------------ declared levers
def declared_levers(a) -> dict:
    spec = dict(ARM_SPECS[a.arm])
    if a.navonly is not None:
        spec["navonly"] = bool(a.navonly)
    if a.no_consistency is not None:
        spec["consistency"] = not a.no_consistency
    if a.wm is not None:
        spec["wm"] = a.wm
    spec.setdefault("cons_absent", "mask")
    if a.cons_absent is not None:
        spec["cons_absent"] = a.cons_absent
    return spec


def make_config(a) -> M.V8FMiniConfig:
    lv = declared_levers(a)
    hmax = a.hmax or (16 if lv["wm"] != "off" else 10)
    return M.V8FMiniConfig(d=a.d, ff=2 * a.d, hist=a.hist, hmax=hmax, window_start=a.window_start, n_q=a.n_q, nb=a.nb, navonly=lv["navonly"],
                           consistency=lv["consistency"], wm=lv["wm"], n_tok=40 if a.grid == "pool4x10" else 160,
                           p_gt_start=a.p_gt_start, p_gt_end=a.p_gt_end, w_ckpt=a.w_ckpt, w_tac=a.w_tac,
                           w_cons=a.w_cons, w_traj=a.w_traj, w_pen=a.w_pen, w_pen_self=a.w_pen_self, w_wm=a.w_wm,
                           detach_upper=a.detach_upper, kin_mode=a.kin_mode, cons_absent=lv["cons_absent"], code_mix=a.code_mix)


def bad_encoder(a, enc, data_info) -> list:
    """The encoder half of the launch gate (PREREG_V8F_LORA proof 3). Empty = pass. With ``--encoder cache`` there must be NO encoder and every lora flag must be
    at its default: a flag that is declared and builds nothing is exactly the class this file's header names."""
    bad = []
    dfl = {k: build_parser().get_default(k) for k in LORA_PARSER_DEFAULTS}
    if a.encoder == "cache":
        if enc is not None:
            bad.append("--encoder cache declared but an encoder was built")
        moved = sorted(k for k in LORA_PARSER_DEFAULTS if getattr(a, k) != dfl[k])
        if moved:
            bad.append(f"--encoder cache declared but {moved} are set: a flag that builds nothing is refused, not ignored")
        return bad
    if enc is None:
        return ["--encoder lora declared but no encoder was built"]
    lv = enc.built_levers()
    tg = parse_targets(a.lora_targets)
    for k, want, got in (("encoder", "lora", lv["encoder"]), ("lora_r", a.lora_r, lv["lora_r"]), ("lora_alpha", a.lora_alpha, lv["lora_alpha"]),
                         ("lora_targets", tg, lv["lora_targets"]), ("lora_dropout", 0.0, lv["lora_dropout"]), ("grid", a.grid[len("pool"):], lv["grid"]),
                         ("checkpointing", bool(a.enc_ckpt), lv["checkpointing"]), ("fp8_ste", True, lv["fp8_ste"])):
        if want != got:
            bad.append(f"encoder {k}: declared {want!r} but built {got!r}")
    if a.accum < 1 or a.batch % a.accum:
        bad.append(f"accum {a.accum} must divide the effective batch {a.batch}")
    # the exact trainable count, from the BUILT modules (r x (in + out) per adapted projection x targets x blocks), and the prereg's literal number
    cfgb = getattr(enc.backbone, "config", None)
    if cfgb is not None:
        want_n = a.lora_r * 2 * cfgb.hidden_size * len(tg) * cfgb.num_hidden_layers
        if lv["lora_blocks"] != cfgb.num_hidden_layers:
            bad.append(f"LoRA wrapped {lv['lora_blocks']} blocks but the backbone has {cfgb.num_hidden_layers}")
        if lv["lora_trainable"] != want_n or lv["lora_expected_trainable"] != want_n:
            bad.append(f"LoRA trainable parameters: built {lv['lora_trainable']} (modules say {lv['lora_expected_trainable']}) != r x 2 x hidden x targets x blocks = {want_n}")
        if (a.lora_r, tuple(tg), cfgb.hidden_size, cfgb.num_hidden_layers) == (32, ("q", "v"), 1024, 24) and lv["lora_trainable"] != 3_145_728:
            bad.append(f"PREREG_V8F_LORA: trainable encoder parameters {lv['lora_trainable']} != 3,145,728")
    try:
        enc.assert_frozen()
    except AssertionError as e:
        bad.append(str(e))
    if not lv["lora_B_zero_at_build"]:
        bad.append("a LoRA B matrix is not zero at build: the identity property (L == B at step 0) is lost")
    if lv["weights_sha256"] is not None and lv["weights_sha256"] != lv["builder_model_sha256"]:
        bad.append(f"encoder weights sha256 {lv['weights_sha256']} is not the cache builder's {lv['builder_model_sha256']}")
    if data_info.get("frames_missing"):
        bad.append(f"--frames-root: {data_info['frames_missing']} train clips have no <clip>.v2ep.pt")
    if not a.frames_root:
        bad.append("--encoder lora needs --frames-root")
    return bad


def lora_lr_of(a) -> float:
    return a.lr if a.lora_lr is None else a.lora_lr


def parse_targets(spec: str) -> list:
    return [t for t in (x.strip() for x in str(spec).split(",")) if t]


def declared_vs_built(a, cfg, model, opt, data_info, enc=None) -> list:
    """Return the list of mismatches (empty = pass). Each flag is checked against what was BUILT, by FLAG_CHECKS kind."""
    bad = bad_encoder(a, enc, data_info)
    parser_dests = {act.dest for act in build_parser()._actions if act.dest != "help"}
    missing = sorted(parser_dests - set(FLAG_CHECKS))
    if missing:
        bad.append(f"flags without a declared-vs-built entry (the gate refuses them): {missing}")
    lv = declared_levers(a)
    built = model.built_levers()
    for k, want in (("navonly", lv["navonly"]), ("consistency", lv["consistency"]), ("wm", lv["wm"]), ("cons_absent", lv["cons_absent"])):
        if built[k] != want:
            bad.append(f"lever {k}: declared {want} but built {built[k]}")
    # PREREG_V8F_MIXO: the teacher-forcing granularity declared on the argv is the one the BUILT model's levers say (argv -> make_config -> cfg -> built_levers); a `kind` declared
    # on a schedule that never draws (p_gt constant 0) or never mixes (p_gt constant 1: every unit is GT, i.e. `whole`) is a lever that is declared and does nothing
    if built.get("code_mix") != a.code_mix:
        bad.append(f"lever code_mix: declared {a.code_mix} but built {built.get('code_mix')}")
    if a.code_mix == "kind" and built["p_gt_start"] == built["p_gt_end"] and built["p_gt_start"] in (0.0, 1.0):
        bad.append(f"code_mix kind declared but the built p_gt is constant {built['p_gt_start']} -> {built['p_gt_end']}: "
                   f"{'O is never teacher-forced' if built['p_gt_start'] == 0.0 else 'every unit is GT, which is whole'} -- a lever that is declared and does nothing")
    if not built["prior_frozen"]:
        bad.append("the P10 prior is not frozen")
    if model.cfg.n_tok != (40 if a.grid == "pool4x10" else 160):
        bad.append(f"grid {a.grid} declared but n_tok built {model.cfg.n_tok}")
    # PI ruling D-PI-YAWRATE-1: the mode the model was built for, the mode the DATA was built in and the mode the PRIOR's units say
    # must be one and the same (a steer prior fed yaw features is a silently wrong model).
    if data_info.get("kin_mode") != a.kin_mode:
        bad.append(f"kin_mode: declared {a.kin_mode} but the data store was built {data_info.get('kin_mode')}")
    if data_info.get("p10_units") not in (None, D.P10_META_UNITS[a.kin_mode]):
        bad.append(f"P10 units {data_info.get('p10_units')} do not match kin_mode {a.kin_mode} (needs {D.P10_META_UNITS[a.kin_mode]})")
    if built["wm"] != "off" and built["wm_ch1_scale"] != M.CH1_SCALE[a.kin_mode]:
        bad.append(f"WM action channel 1 scale {built['wm_ch1_scale']} != {M.CH1_SCALE[a.kin_mode]} for kin_mode {a.kin_mode}")
    for f in ("hist", "hmax", "d", "n_q", "nb", "p_gt_start", "p_gt_end", "w_ckpt", "w_tac", "w_cons", "w_traj", "w_pen",
              "w_pen_self", "w_wm", "detach_upper", "kin_mode"):
        want = {"hist": a.hist, "hmax": cfg.hmax, "d": a.d, "n_q": a.n_q, "nb": a.nb}.get(f, getattr(a, f, None))
        got = getattr(model.cfg, f)
        if want is not None and not (got == want or (isinstance(want, float) and abs(got - want) < 1e-12)):
            bad.append(f"cfg.{f}: declared {want} but built {got}")
    # PREREG_V8F_HIST Amendment 1 (F-3): the BUILT history and batch depth, and the EFFECTIVE first-window step, must be the declared ones -- in the built model's levers, in the
    # window list the data was built from (the measured first t) and in the dataset's held-age clamp. A --window-start the data did not build is REFUSED, and a declared one with no
    # read-back at all is refused too (it would be a flag nobody can see reach anything).
    ws_decl = a.hist - 1 if a.window_start == -1 else a.window_start
    for k, want in (("hist", a.hist), ("hmax", cfg.hmax), ("window_start", ws_decl)):
        if built.get(k) != want:
            bad.append(f"lever {k}: declared {want} but built {built.get(k)}")
    for k, what in (("window_t_min", "the first t of the training window list"), ("ds_window_start", "the dataset's held-age clamp origin")):
        if k in data_info or a.window_start != -1:
            if data_info.get(k) != ws_decl:
                bad.append(f"window_start {a.window_start}: declared {ws_decl} but {what} was built as {data_info.get(k)}")
    # weights that the loss will actually use must be the declared ones (nocons -> 0)
    if not lv["consistency"] and (built["w"]["w_pen"] != 0.0 or built["w"]["w_pen_self"] != 0.0):
        bad.append("M1-nocons declared but penalty weights are non-zero in the built loss")
    if lv["consistency"] and not built["w"]["w_pen"] > 0.0:
        bad.append("consistency penalties declared ON but w_pen is 0 -- a lever that is declared and does nothing")
    if lv["cons_absent"] == "zero" and not (cfg.w_cons > 0.0 and cfg.w_tac > 0.0):
        bad.append("cons_absent zero declared but the constraint loss weight is 0 -- a lever that is declared and does nothing")
    if lv["wm"] != "off" and not built["w"]["w_wm"] > 0.0:
        bad.append("WM auxiliary declared but w_wm is 0 -- a lever that is declared and does nothing")
    if lv["wm"] == "off" and model.wm is not None:
        bad.append("wm declared off but a WM module was built")
    if lv["wm"] != "off" and (model.wm is None or built["wm_idm_gamma"] != (0.0 if lv["wm"] == "a" else cfg.wm_gamma)):
        bad.append(f"wm {lv['wm']} declared but built gamma {built['wm_idm_gamma']}")
    # optimiser read-back
    lrs = {g["lr"] for g in opt.param_groups}
    want_lrs = {a.lr} | ({lora_lr_of(a)} if enc is not None else set())
    if lrs != want_lrs:
        bad.append(f"lr: declared {sorted(want_lrs)} but optimiser holds {sorted(lrs)}")
    if enc is not None:
        lg = [g for g in opt.param_groups if g.get("name") == "lora"]
        if len(lg) != 1 or {id(p) for p in lg[0]["params"]} != {id(p) for p in enc.lora_parameters()}:
            bad.append("the LoRA optimiser group does not hold exactly the encoder's LoRA matrices")
        elif lg[0]["lr"] != lora_lr_of(a):
            bad.append(f"lora_lr: declared {lora_lr_of(a)} but the LoRA group holds {lg[0]['lr']}")
    if {g["weight_decay"] for g in opt.param_groups if g["weight_decay"] > 0} not in ({a.wd}, set()):
        bad.append("weight decay not as declared")
    # data read-back
    if data_info["n_clips"] != data_info["n_clips_expected"]:
        bad.append(f"data: {data_info['n_clips']} clips built, {data_info['n_clips_expected']} declared "
                   f"(data-frac {a.data_frac}, max-clips {a.max_clips})")
    if data_info.get("n_excluded", 0) != data_info.get("n_excluded_declared", 0):
        bad.append(f"exclude {a.exclude!r}: {data_info.get('n_excluded', 0)} clips excluded, "
                   f"{data_info.get('n_excluded_declared', 0)} declared")
    if data_info.get("excluded_in_train", 0):
        bad.append(f"exclude {a.exclude!r}: {data_info['excluded_in_train']} excluded clips are still in the TRAIN set")
    if data_info["batch"] != a.batch:
        bad.append("batch size not as declared")
    if data_info.get("v9_md5") and a.v9_train_md5 and data_info["v9_md5"] != a.v9_train_md5 and not data_info.get("md5_overridden"):
        bad.append(f"v9 release md5 {data_info['v9_md5']} != declared {a.v9_train_md5}")
    return bad


def live_check(model, batch, cfg, steps: int = 3, lr: float = 1e-3) -> dict:
    """G-LIVE: three optimiser steps on a real batch (on a COPY). Every declared loss term finite; every trainable
    parameter receives a non-zero gradient at some step, except the declared-dead set; the frozen prior reads exactly 0."""
    m = copy.deepcopy(model)
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    acc = {n: 0.0 for n, p in m.named_parameters() if p.requires_grad}
    first = None
    terms_seen = None
    for s in range(steps):
        opt.zero_grad(set_to_none=False)
        out = m.loss(batch, p_gt=0.5, gen=torch.Generator().manual_seed(s))
        T = out["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"G-LIVE: loss term {k} is not finite at step {s}: {float(v)}")
        terms_seen = {k: float(v.detach()) for k, v in T.items()}
        T["total"].backward()
        for n, p in m.named_parameters():
            if p.requires_grad and p.grad is not None:
                acc[n] += float(p.grad.abs().sum())
        if s == 0:
            first = {n: float(p.grad.abs().sum()) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}
        opt.step()
    expect = {"ckpt", "goal", "lat", "lon", "cons", "ade", "fde", "total"}
    if cfg.consistency:
        expect |= {f"pen_{r}" for r in M.RULES} | {f"pens_{r}" for r in M.RULES}
    if cfg.wm != "off":
        expect.add("wm")
    missing = sorted(expect - set(terms_seen))
    extra_pen = sorted(k for k in terms_seen if k.startswith("pen") and not cfg.consistency)
    declared_dead = [n for n in acc if cfg.wm == "a" and ".idm_heads." in n]
    dead = sorted(n for n, g in acc.items() if g == 0.0)
    unexpected_dead = sorted(set(dead) - set(declared_dead))
    declared_dead_alive = sorted(n for n in declared_dead if acc[n] != 0.0)
    ctrl = [float(p.grad.abs().sum()) if p.grad is not None else 0.0 for p in m.prior.parameters()]
    head_first = sum(v for n, v in first.items() if n.startswith("head_o."))
    res = {"terms": terms_seen, "missing_terms": missing, "unexpected_terms": extra_pen, "dead_params": dead,
           "declared_dead": declared_dead, "unexpected_dead": unexpected_dead, "declared_dead_but_alive": declared_dead_alive,
           "control_prior_grad_sum": float(sum(ctrl)), "head_o_grad_step0": head_first,
           "n_trainable": len(acc), "groups_grad": {g: float(sum(v for n, v in acc.items() if n.startswith(pre)))
                                                   for g, pre in (("adapter", "adapter"), ("S", "read_s"), ("T", "read_t"),
                                                                  ("O", "read_o"), ("head_s", "head_s"), ("head_o", "head_o"),
                                                                  ("wm", "wm."))}}
    ok = (not missing and not extra_pen and not unexpected_dead and not declared_dead_alive
          and res["control_prior_grad_sum"] == 0.0 and head_first > 0.0)
    res["pass"] = bool(ok)
    if not ok:
        raise AssertionError("G-LIVE failed: " + json.dumps({k: v for k, v in res.items() if k != "groups_grad"}, default=str)[:1500])
    return res


def code_mix_probe(model, batch, cfg, p_gt: float = 0.5, tries: int = 4) -> dict:
    """G-LIVE for PREREG_V8F_MIXO: the code O actually READ in a teacher-forced forward of the BUILT model (on a COPY, no gradient, its own generators) is, per sample and per unit
    of ``CODE_MIX_UNITS``, either the GT code's value or T's own (the code T predicted in that same call), and has the joint structure the declared ``code_mix`` promises:

      whole  every labelled sample is ALL-GT or ALL-OWN (never partial): one switch per sample
      kind   some labelled sample is PARTIAL (>= 1 unit GT and >= 1 unit own) within ``tries`` draws; and each unit is GT on a share of the labelled samples near p_gt

    A unit that matches neither (the code O read is not a mix of the two sources), too few labelled windows to tell, or a joint structure other than the declared one is an
    AssertionError, which the caller turns into a refusal. This is the reachability proof FROM THE ACTUAL CALLER: it reads the tensor the trainer's model hands the O layer."""
    m = copy.deepcopy(model).cpu()
    m.train()
    has = batch["lab_t"]["has"].bool()
    n_lab = int(has.sum())
    if n_lab < 1:
        raise AssertionError("code_mix probe: no labelled window in the probe batch: the joint structure of the code O reads cannot be told")
    units = [torch.as_tensor(idx, dtype=torch.long) for idx in M.CODE_MIX_UNITS.values()]
    cap = {}
    t0 = m.run_t

    def run_t(*a, **k):
        o = t0(*a, **k)
        cap["own"] = m._code_pred(o).detach().clone()                    # T's own code IN THIS CALL (it depends on the checkpoint T was handed)
        return o
    m.run_t = run_t
    n_partial = n_all_gt = n_all_own = 0
    gt_hits = torch.zeros(len(units))
    n_seen = 0
    for k in range(tries):
        with torch.no_grad():
            out = m._forward(batch, p_gt, True, torch.Generator().manual_seed(7919 + k))
            gt = m._code_gt(batch["lab_t"])
        code, own = out["code"], cap["own"]
        is_gt = torch.stack([(code[:, u] == gt[:, u]).all(1) & ~(own[:, u] == gt[:, u]).all(1) for u in units], 1)       # [B, U] unit read from the GT
        is_own = torch.stack([(code[:, u] == own[:, u]).all(1) & ~(own[:, u] == gt[:, u]).all(1) for u in units], 1)     # [B, U] unit read from T's own code
        same = torch.stack([(own[:, u] == gt[:, u]).all(1) for u in units], 1)                                               # [B, U] units where the sources agree (untellable)
        if (same & has[:, None]).any():
            raise AssertionError("code_mix probe: a unit of T's own code equals the GT code on a labelled window of the probe batch: that unit's source cannot be told")
        other = has[:, None] & ~(is_gt | is_own)
        if other.any():
            raise AssertionError(f"code_mix probe: {int(other.sum())} (sample, unit) cells of the code O read are neither the GT value nor T's own: not a mix of the two sources")
        if (is_gt & ~has[:, None]).any():
            raise AssertionError("code_mix probe: an UNLABELLED window read the GT code")
        g, w = is_gt[has], is_own[has]
        n_partial += int((g.any(1) & w.any(1)).sum())
        n_all_gt += int(g.all(1).sum())
        n_all_own += int(w.all(1).sum())
        gt_hits += g.float().sum(0)
        n_seen += int(g.shape[0])
        if cfg.code_mix == "kind" and n_partial > 0 and k >= 1:
            break
    rate = (gt_hits / max(n_seen, 1)).tolist()
    res = {"code_mix": cfg.code_mix, "p_gt": p_gt, "n_labelled": n_lab, "n_draws_samples": n_seen, "n_partial": n_partial, "n_all_gt": n_all_gt, "n_all_own": n_all_own,
           "unit_gt_rate": dict(zip(M.CODE_MIX_UNITS, rate))}
    if cfg.code_mix == "kind":
        if n_partial == 0:
            raise AssertionError(f"code_mix probe: code_mix kind is declared but NO labelled sample read a partly-GT code in {n_seen} draws ({res}): the per-unit mix did not reach the code O reads")
        if any(abs(r - p_gt) > 0.25 for r in rate) and n_seen >= 64:
            raise AssertionError(f"code_mix probe: a unit's GT share {res['unit_gt_rate']} is far from p_gt {p_gt}")
    elif n_partial:
        raise AssertionError(f"code_mix probe: code_mix whole is declared but {n_partial} labelled samples read a PARTLY-GT code ({res}): whole is one switch per sample")
    res["pass"] = True
    return res


# ------------------------------------------------------------------------------------------------ the encoder (PREREG_V8F_LORA)
def build_encoder_raw(a, dev):
    """The ONLY place the real weights are loaded (tests replace THIS, never :func:`make_encoder`): the cache builder's DINOv3 ViT-L/16 (bf16, local HF cache, no
    token, no download) + LoRA."""
    from tanitad.models import v8f_lora_encoder as L
    return L.DinoV3LoRAEncoder.from_pretrained(r=a.lora_r, alpha=a.lora_alpha, targets=parse_targets(a.lora_targets), seed=a.seed,
                                               grid=a.grid[len("pool"):], device=dev, dtype=torch.bfloat16)


def make_encoder(a, dev):
    """The on-the-fly encoder. The global RNG state (CPU and CUDA) is saved and restored around the build, so that nothing the loader or the adapter init draws can
    move the heads' initialisation (``torch.manual_seed(seed)`` then ``V8FMini``): L's heads start exactly where B's did."""
    st = torch.get_rng_state()
    cst = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return build_encoder_raw(a, dev)
    finally:
        torch.set_rng_state(st)
        if cst is not None:
            torch.cuda.set_rng_state_all(cst)


def enc_chunk(a, cfg) -> int:
    """Frames per encoder micro-batch: (batch / accum) windows x hmax frames."""
    return max(1, a.batch // max(1, a.accum)) * cfg.hmax


def lora_step(model, enc, b, frames, fmap, a, dev, amp, p_gt, chunk, step, *, timing: bool = False, on_grads=None) -> tuple:
    """One forward + backward of the full effective batch through heads and encoder (the gradient cache, ``cached_grad_step``), then the two gradient clips
    (heads: B's rule on the heads alone; LoRA: its own norm -- a joint norm would change the heads' update whenever the LoRA gradient dominates, which is
    not B's rule). The optimiser step is the caller's. -> ``(out, heads_grad_norm, lora_grad_norm)``."""
    from tanitad.models import v8f_lora_encoder as L

    def run_loss(feats):
        bb = dict(b)
        bb["feats"] = feats
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            out = model.loss(bb, p_gt=p_gt, gen=None)
        T = out["terms"]
        if not torch.isfinite(T["total"]):
            raise SystemExit(f"ZZNONFINITE-LOSSZZ at step {step}: {({k: float(v) for k, v in T.items()})}")
        return out
    out = L.cached_grad_step(enc, frames, fmap, run_loss, chunk=chunk, device=dev, timing=timing)
    if on_grads is not None:                           # the AR-2 smoke reads the RAW per-group gradients here (before the clips)
        on_grads(model, enc)
    gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.clip_grad)
    gn_l = torch.nn.utils.clip_grad_norm_(enc.lora_parameters(), a.clip_grad)
    return out, gn, gn_l


def identity_probe(enc, probe, dev, chunk) -> dict:
    """The encoder at its INITIAL state (B = 0) on the probe's frames vs the cached tokens of the same windows (the dataset kept them): max / mean abs and
    rel-L2. RECORDED, not gated: tau is the PI's/MM's to commit from the identity control (PREREG_V8F_LORA proof 1)."""
    from tanitad.models import v8f_lora_encoder as L
    with torch.no_grad():
        pu = L.encode_chunked(enc, probe["frames_u8"], chunk, grad=False, device=dev).cpu()
    mine = pu[probe["frame_map"]]
    ref = probe["feats"].float()
    d = (mine - ref).abs()
    return {"n_frames_unique": int(pu.shape[0]), "n_values": int(d.numel()), "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rel_l2": float((mine - ref).norm() / ref.norm().clamp_min(1e-12)), "frac_bitwise_equal": float((mine == ref).float().mean())}


def live_check_lora(model, enc, batch, cfg, a, dev, chunk, steps: int = 3, lr: float = 1e-3) -> dict:
    """G-LIVE through the REAL encoder path (PREREG_V8F_LORA proof 2). Three optimiser steps (heads + LoRA) on a real probe batch, on a COPY of the heads and with
    the LoRA state restored afterwards. Requires: every declared loss term finite; every head parameter non-zero at some step (as ``live_check``); the LoRA B
    matrices non-zero at step 0 while the A matrices read EXACTLY 0.0 there (B = 0 at init, so dL/dA = B^T ... = 0 by construction -- a non-zero would mean B was
    not zero) and non-zero afterwards; the frozen DINOv3 weights (the control) exactly 0.0."""
    from tanitad.models import v8f_lora_encoder as L
    frames, fmap = batch["frames_u8"], batch["frame_map"]
    hb = to_dev({k: v for k, v in batch.items() if k not in ("frames_u8", "frame_map", "feats")}, dev)
    m = copy.deepcopy(model).to(dev)
    m.train()
    sd0 = enc.lora_state_dict()
    heads = [p for p in m.parameters() if p.requires_grad]
    lora = enc.lora_parameters()
    opt = torch.optim.AdamW([{"params": heads}, {"params": lora}], lr=lr)
    acc = {n: 0.0 for n, p in m.named_parameters() if p.requires_grad}
    accL = {n: 0.0 for n, p in enc.named_parameters() if "lora_" in n}
    firstL, first, terms_seen, frozen = None, None, None, 0.0
    try:
        for s_ in range(steps):
            opt.zero_grad(set_to_none=True)

            def run_loss(feats, s_=s_):
                bb = dict(hb)
                bb["feats"] = feats
                out = m.loss(bb, p_gt=0.5, gen=torch.Generator(device=dev).manual_seed(s_))   # the generator lives on the model's device: a CPU one crashed AR-2's cuda smoke
                for k, v in out["terms"].items():
                    if not torch.isfinite(v):
                        raise AssertionError(f"G-LIVE: loss term {k} is not finite at step {s_}: {float(v)}")
                return out
            out = L.cached_grad_step(enc, frames, fmap, run_loss, chunk=chunk, device=dev)
            terms_seen = {k: float(v.detach()) for k, v in out["terms"].items()}
            for n, p in m.named_parameters():
                if p.requires_grad and p.grad is not None:
                    acc[n] += float(p.grad.abs().sum())
            if s_ == 0:
                first = {n: float(p.grad.abs().sum()) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}
            gL = {n: float(p.grad.abs().sum()) for n, p in enc.named_parameters() if "lora_" in n and p.grad is not None}
            if s_ == 0:
                firstL = gL
            for n, g in gL.items():
                accL[n] += g
            frozen += sum(float(p.grad.abs().sum()) for n, p in enc.named_parameters() if "lora_" not in n and p.grad is not None)
            opt.step()
    finally:
        enc.load_lora_state_dict(sd0)
        for p in lora:
            p.grad = None
    expect = {"ckpt", "goal", "lat", "lon", "cons", "ade", "fde", "total"}
    if cfg.consistency:
        expect |= {f"pen_{r}" for r in M.RULES} | {f"pens_{r}" for r in M.RULES}
    if cfg.wm != "off":
        expect.add("wm")
    missing = sorted(expect - set(terms_seen))
    declared_dead = [n for n in acc if cfg.wm == "a" and ".idm_heads." in n]
    dead = sorted(n for n, g in acc.items() if g == 0.0)
    unexpected_dead = sorted(set(dead) - set(declared_dead))
    ctrl = [float(p.grad.abs().sum()) if p.grad is not None else 0.0 for p in m.prior.parameters()]
    A0 = sum(g for n, g in firstL.items() if "lora_A" in n)
    B0 = sum(g for n, g in firstL.items() if "lora_B" in n)
    b_dead = sorted(n for n, g in firstL.items() if "lora_B" in n and g == 0.0)
    a_dead = sorted(n for n, g in accL.items() if "lora_A" in n and g == 0.0)
    res = {"terms": terms_seen, "missing_terms": missing, "dead_params": dead, "unexpected_dead": unexpected_dead,
           "control_prior_grad_sum": float(sum(ctrl)), "control_frozen_dinov3_grad_sum": float(frozen),
           "lora_A_grad_step0": float(A0), "lora_B_grad_step0": float(B0), "lora_B_modules_dead_at_step0": b_dead, "lora_A_modules_dead_after_step1": a_dead,
           "lora_A_grad_total": float(sum(v for n, v in accL.items() if "lora_A" in n)), "lora_B_grad_total": float(sum(v for n, v in accL.items() if "lora_B" in n)),
           "n_lora_matrices": len(accL), "n_trainable_heads": len(acc), "steps": steps}
    ok = (not missing and not unexpected_dead and res["control_prior_grad_sum"] == 0.0 and frozen == 0.0 and A0 == 0.0 and B0 > 0.0
          and not b_dead and not a_dead and res["lora_A_grad_total"] > 0.0)
    res["pass"] = bool(ok)
    if not ok:
        raise AssertionError("G-LIVE (lora) failed: " + json.dumps(res, default=str)[:1800])
    return res


# ------------------------------------------------------------------------------------------------ data
def build_data(a, cfg):
    """-> (store, J, train_windows, clips, info). Joins the v9 release onto the cache grid unless a join npz exists."""
    info = {"batch": a.batch, "md5_overridden": False}
    if a.synthetic:
        root = os.path.join(a.out, "_synthetic")
        w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=a.seed)
        J = D.load_join(os.path.join(root, "join_train.npz"))
        cache = root
        clips_all = w["names"]["train"]
    else:
        cache = a.cache_dir
        man = D.read_manifest(cache, "train")
        clips_all = [c for c, _ in man]
        if a.join_train and os.path.exists(a.join_train) and not a.fit_join:
            J = D.load_join(a.join_train)
        else:
            from tanitad.data.v9_labels import load_v9_release
            rel = load_v9_release(a.v9_train, expect_md5=a.v9_train_md5 or None)
            info["v9_md5"] = rel.md5
            J = D.build_label_join(man, rel, lead_npz=a.lead_train if os.path.exists(a.lead_train) else None,
                                   speed=D.load_speed_sidecar(a.speed_train))
            if a.join_train:
                D.save_join(J, a.join_train)
    excluded, n_ex_declared = [], 0
    if a.exclude:
        xlo, xhi = (int(x) for x in a.exclude.split(":"))
        n_ex_declared = xhi - xlo
        excluded = sorted(clips_all, key=D.sha12)[xlo:xhi]
        ex = set(excluded)
        clips_all = [c for c in clips_all if c not in ex]
    clips = D.select_train_clips(clips_all, a.data_frac)
    if a.max_clips:
        clips = clips[:a.max_clips]
    expected = (max(1, int(round(a.data_frac * len(clips_all)))) if a.data_frac < 1.0 else len(clips_all))
    if a.max_clips:
        expected = min(a.max_clips, expected)
    info.update({"n_clips": len(clips), "n_clips_expected": expected, "n_clips_all": len(clips_all)})
    # the digests are RA-1's (ra1_pooled.clips_sha256: newline-joined, sha12 order), so a v8F-mini run and an RA-1 run can be
    # shown to hold out the SAME clips and train on the SAME clips
    info.update({"n_excluded": len(excluded), "n_excluded_declared": n_ex_declared,
                 "excluded_in_train": len(set(excluded) & set(clips)),
                 "excluded_sha256": hashlib.sha256(chr(10).join(excluded).encode("utf-8")).hexdigest() if excluded else None,
                 "train_sha256": hashlib.sha256(chr(10).join(sorted(clips, key=D.sha12)).encode("utf-8")).hexdigest()})
    lru = a.lru or max(8, int(1.5e9 // (J["n"].max() * cfg.n_tok * 1024 * 2)))
    if a.encoder == "lora":
        lru = a.lru or 50          # clips whose v2ep (~37 MB of PNGs) are held in RAM: >= the sampler's chunk (48) + 2
    info["lru"] = lru
    tracks = None
    if a.kin_mode == "yaw":
        tpath = os.path.join(cache, "native_tracks.pt") if a.synthetic else a.tracks
        if not tpath:
            raise SystemExit("REFUSED: --kin-mode yaw needs --tracks native_tracks.pt (build_native_tracks.py): the cache holds only the even frames")
        tracks = D.load_native_tracks(tpath)
    info["kin_mode"] = a.kin_mode
    store = D.ClipStore(cache, "train", lru=lru, grid=a.grid, kin_mode=a.kin_mode, tracks=tracks)
    allw = D.train_windows(store, J, clips_all, hist=cfg.hist, start=cfg.window_start)
    wins = D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start)
    info.update({"n_windows_all_clips": len(allw), "n_windows": len(wins),
                 "window_start": D.effective_window_start(cfg.hist, cfg.window_start),         # what train_windows was asked for ...
                 "window_t_min": min((w[1] for w in allw), default=None)})                      # ... and what the built list actually starts at (read back)
    return store, J, wins, clips, info


class Prefetch:
    def __init__(self, gen, fn, depth=3):
        self.q, self.fn = queue.Queue(depth), fn
        self.t = threading.Thread(target=self._run, args=(gen,), daemon=True)
        self.t.start()

    def _run(self, gen):
        try:
            for x in gen:
                self.q.put(self.fn(x))
        except Exception as e:                                  # surface in the consumer
            self.q.put(e)
        self.q.put(None)

    def __iter__(self):
        while True:
            x = self.q.get()
            if x is None:
                return
            if isinstance(x, Exception):
                raise x
            yield x


def to_dev(b, dev):
    return {k: ({kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict) else v.to(dev)) for k, v in b.items()}


def argv_sha(argv) -> str:
    return hashlib.sha256(" ".join(argv).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if a.p10 is None:
        a.p10 = os.path.join(PKG, "raw", "p10_yaw" if a.kin_mode == "yaw" else "p10", "p10_prior.pt")
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    cfg = make_config(a)
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    store, J, wins, clips, info = build_data(a, cfg)
    info["p10_units"] = None if a.synthetic else M.p10_meta(a.p10).get("units", "steer")
    if not a.synthetic:
        try:
            M.check_p10_units(a.p10, a.kin_mode)                 # REFUSES a prior whose declared units are not this mode's
        except ValueError as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ {e}")
    fstore = None
    if a.encoder == "lora":
        if not a.frames_root:
            raise SystemExit("ZZDVB-REFUSEDZZ --encoder lora needs --frames-root")
        fstore = D.FrameStore(a.frames_root, lru=info["lru"])
        info["frames_missing"] = len(fstore.missing(clips))
    if a.encoder == "lora":                                          # two NAMED constructions, so test_v8f_window_start's static guard can see both
        ds = D.V8FMiniFrameWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                                   window_start=cfg.window_start)
    else:
        ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                              window_start=cfg.window_start)
    info["ds_window_start"] = ds.window_start                       # the held-age clamp's origin, read back from the BUILT dataset (G-DVB compares it)
    prior = M.load_p10(a.p10) if not a.synthetic else M.P10Prior.synthetic(a.seed)
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, prior, mu, sd).to(dev)
    enc = None
    if a.encoder == "lora":
        enc = make_encoder(a, dev)                     # AFTER the heads: it must not move their initialisation (the global RNG is restored inside)
        enc.set_checkpointing(a.enc_ckpt)
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}]
    if enc is not None:                                # the LoRA matrices (2-D: the same weight-decay rule as the heads), their own peak lr
        groups.append({"params": enc.lora_parameters(), "weight_decay": a.wd, "lr": lora_lr_of(a), "name": "lora", "lr_mult": lora_lr_of(a) / a.lr})
    opt = torch.optim.AdamW(groups, lr=a.lr)
    S = math.ceil(info["n_windows_all_clips"] / a.batch)
    total = a.steps or S * a.steps_mult
    sampler = D.ChunkedSampler(wins, a.batch, seed=a.seed, chunk=max(4, min(48, info["lru"] - 2)),
                               max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    # ---- the launch gate
    bad = declared_vs_built(a, cfg, model, opt, info, enc)
    if bad:
        raise SystemExit("ZZDVB-REFUSEDZZ declared-vs-built: " + "; ".join(bad))
    chunk = enc_chunk(a, cfg)
    ident = None
    if enc is None:
        probe = D.collate([ds.item(*w) for w in next(sampler.epoch(0))])
        live = live_check(model.cpu(), probe, cfg)
    else:
        ds_probe = D.V8FMiniFrameWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                                         window_start=cfg.window_start, keep_cached=True)
        probe_f = D.collate_frames([ds_probe.item(*w) for w in next(sampler.epoch(0))][:LIVE_PROBE_WINDOWS_LORA], fstore)
        probe = {k: v for k, v in probe_f.items() if k not in ("frames_u8", "frame_map")}      # the cached tokens stay in `probe` (admissibility, identity)
        live = live_check_lora(model, enc, probe_f, cfg, a, dev, chunk)
        ident = identity_probe(enc, probe_f, dev, chunk)
    model.to(dev)
    try:
        mix = code_mix_probe(model, probe, cfg)           # PREREG_V8F_MIXO: the joint structure of the code the BUILT model hands O is the declared one (whole: all-or-nothing; kind: partial)
    except AssertionError as e:
        raise SystemExit(f"ZZDVB-REFUSEDZZ declared-vs-built: {e}")
    adm = M.assert_no_situation_classifier_input(model.cpu().eval(), probe)
    model.to(dev).train()
    levers = model.built_levers()
    if enc is not None:
        levers["encoder"] = enc.built_levers()
        levers["accum"] = {"accum": a.accum, "micro_windows": a.batch // a.accum, "chunk_frames": chunk}
    sc = {"declared_vs_built": "PASS", "live": live, "code_mix_probe": mix, "admissibility": adm, "levers": levers,
          "steps_total": total, "S_steps_per_epoch_100pct": S, "info": info, "argv_sha": argv_sha(argv)}
    if ident is not None:
        sc["identity_probe"] = ident
    json.dump(sc, open(os.path.join(a.out, "selfcheck.json"), "w"), indent=1, default=str)
    if a.selfcheck_only:
        print("SELFCHECK PASS", json.dumps({k: sc[k] for k in ("steps_total", "info")}, default=str))
        return 0
    json.dump({"argv": argv, "sha": argv_sha(argv)}, open(os.path.join(a.out, "argv.json"), "w"))

    # ---- resume
    ck_path = os.path.join(a.out, "ckpt_last.pt")
    step, ep, bi = 0, 0, 0
    if os.path.exists(ck_path) and not a.no_resume:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if ck["argv_sha"] != argv_sha(argv):
            raise SystemExit("ZZRESUME-REFUSEDZZ the checkpoint was written by a different argv")
        model.load_state_dict(ck["model"])
        if enc is not None:
            enc.load_lora_state_dict(ck["lora"])
        opt.load_state_dict(ck["opt"])
        step, ep, bi = ck["step"], ck["epoch"], ck["batch_in_epoch"]
        print(f"[resume] step {step} epoch {ep} batch {bi}", flush=True)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((s - a.warmup) / max(total - a.warmup, 1), 1.0))))

    def save(final=False):
        blob = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": ep, "batch_in_epoch": bi,
                "argv_sha": argv_sha(argv), "cfg": cfg.__dict__, "p10_path": os.path.abspath(a.p10) if not a.synthetic else "",
                "grid": a.grid, "arm": a.arm, "seed": a.seed, "cons_mu": mu, "cons_sd": sd,
                "cfg_tuple_fields": {"wm_strides": list(cfg.wm_strides)}}
        if enc is not None:             # `model` keeps B's layout (the unchanged v8f_mini_eval.py loads it); the LoRA matrices travel beside it
            blob["lora"], blob["encoder_levers"] = enc.lora_state_dict(), enc.built_levers()
        torch.save(blob, ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)

    log = open(os.path.join(a.out, "train_log.jsonl"), "a", encoding="utf-8")
    amp = a.amp and dev.type == "cuda"
    t0 = time.time()
    done = step >= total
    while not done:
        gen_batches = sampler.epoch(ep)
        for _ in range(bi):                                    # fast-forward within the epoch (indices only)
            next(gen_batches, None)
        if enc is None:
            pf = Prefetch(gen_batches, lambda idx: D.collate([ds.item(*w) for w in idx]), depth=3)
        else:
            pf = Prefetch(gen_batches, lambda idx: D.collate_frames([ds.item(*w) for w in idx], fstore), depth=2)
        for b in pf:
            fr = (b.pop("frames_u8"), b.pop("frame_map")) if enc is not None else None      # the frames stay on the host; the encoder moves them chunk by chunk
            b = to_dev(b, dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step) * g.get("lr_mult", 1.0)
            p_gt = M.p_gt_schedule(step, total, cfg.p_gt_start, cfg.p_gt_end)
            torch.manual_seed(a.seed * 1_000_003 + step)               # per-step RNG: a resumed run is the same run
            if os.environ.get("V8F_TRAIN_CRASH_AT") == str(step):      # test hook (resume test); never set in a launch
                raise RuntimeError("V8F_TRAIN_CRASH_AT")
            model.train()
            if enc is None:
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                    out = model.loss(b, p_gt=p_gt, gen=None)
                T = out["terms"]
                if not torch.isfinite(T["total"]):
                    raise SystemExit(f"ZZNONFINITE-LOSSZZ at step {step}: {({k: float(v) for k, v in T.items()})}")
                opt.zero_grad(set_to_none=True)
                T["total"].backward()
                gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.clip_grad)
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
                out, gn, gn_l = lora_step(model, enc, b, fr[0], fr[1], a, dev, amp, p_gt, chunk, step)
                T = out["terms"]
                opt.step()
            step += 1
            bi += 1
            if step % a.log_every == 0 or step == total:
                rec = {"step": step, "epoch": ep, "p_gt": p_gt, "lr": lr_at(step - 1), "grad_norm": float(gn),
                       "sec": time.time() - t0, **{k: float(v) for k, v in T.items()}}
                if "viol" in out:
                    rec["viol_any_batch"] = float(torch.stack(list(out["viol"].values())).any(0).float().mean())
                if enc is not None:
                    rec["lora_grad_norm"] = float(gn_l)
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[{step}/{total}] ade {rec['ade']:.3f} total {rec['total']:.3f} p_gt {p_gt:.2f}", flush=True)
            if step % a.ckpt_every == 0:
                save()
            if step >= total:
                done = True
                break
        else:
            ep += 1
            bi = 0
    save(final=True)
    summary = {"done": True, "steps": step, "arm": a.arm, "seed": a.seed, "data_frac": a.data_frac,
               "steps_mult": a.steps_mult, "S": S, "n_clips": info["n_clips"], "n_windows": info["n_windows"],
               "exclude": a.exclude, "excluded_sha256": info.get("excluded_sha256"), "train_sha256": info.get("train_sha256"),
               "seconds": time.time() - t0, "argv_sha": argv_sha(argv), "levers": levers,
               "selfcheck": "selfcheck.json"}
    if enc is not None:
        summary["encoder"] = "lora"
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)      # LAST, atomically meaningful
    print("DONE", json.dumps({k: summary[k] for k in ("steps", "arm", "seed", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
