#!/usr/bin/env python3
"""REF-A v1 trainer — feature prediction on cached frozen DINOv3 fields.

⭐ WHAT IS DIFFERENT FROM `refa_train_plus.py`, IN ONE LINE: the loss is the
FUTURE PATCH FIELD, not a trajectory label. Everything else that made REF-A
stable is kept verbatim (feature-cache training, fit-once standardizer, no
BatchNorm/dropout, adapter-vs-predictor LR groups, the adapter-collapse
monitor).

CACHE CONTRACT (stage 1, separate job — this trainer never touches an image):
    <cache>/<episode_id>.pt  ->  fp16 tensor [T, 640, 1024]
      * DINOv3 ViT-L/16 patch tokens, CLS DISCARDED
      * 256x640 crop at 120 deg HFOV (grid 16x40)
    <cache>/index.json       ->  {"episodes": [...], "parity_key": "...",
                                  "skip_hash": "...", "geometry": {...}}
⛔ The trainer REFUSES a cache whose geometry disagrees with
``refa_v1.DINOV3_GEOMETRY`` — a silently narrower interface is exactly the
defect v1 exists to remove, and it would still train.

Run (smoke, CPU, no cache needed):
    python stack/scripts/refa_v1_train.py --smoke
Run (real):
    python stack/scripts/refa_v1_train.py --cache /path/dinov3_w120 \
        --steps 30000 --bs 8 --out ~/experiments/refa-v1
Run (the precision arm — a NEXT run, never the live one; see --precision):
    python stack/scripts/refa_v1_train.py ... --precision bf16 --tf32
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import DINOV3_GEOMETRY, RefAV1, RefAV1Config


def build_model(args) -> RefAV1:
    cfg = RefAV1Config(
        strategic_cfg=None if args.no_hierarchy else StrategicPolicyConfig(),
        tactical_cfg=None if args.no_hierarchy else TacticalPolicyConfig(),
        w_cf=args.w_cf, cf_negs=args.cf_negs, cf_at_step=args.cf_at_step,
        motion_inject=args.motion_inject,
        target_space=args.target_space,
        w_aux_head=args.w_aux_head, proposal_k=args.proposal_k,
        w_sigreg=args.w_sigreg, var_floor=args.var_floor,
        min_participation=args.min_participation,
        bptt_truncate=args.bptt_truncate,
        ema_targets=args.ema_targets, ema_decay=args.ema_decay,
        ema_decay_end=args.ema_decay_end,
        speed_channel=args.speed_channel, tmix_groups=args.tmix_groups,
        **({} if args.detach_aux is None
           else {"detach_aux_targets": args.detach_aux}),
    )
    if args.smoke:
        cfg.d_enc, cfg.n_tokens, cfg.d_state = 32, 8, 32
        cfg.op_layers, cfg.op_heads, cfg.tac_layers = 1, 2, 1
        cfg.tac_queries, cfg.str_dim, cfg.str_layers = 4, 16, 1
        if not args.no_hierarchy:
            cfg.strategic_cfg = StrategicPolicyConfig(d_model=32, depth=1,
                                                      n_heads=2, d_ctx=16,
                                                      d_cmd=8)
            cfg.tactical_cfg = TacticalPolicyConfig(d_model=32, depth=1,
                                                    n_heads=2, d_intent=16)
    return RefAV1(cfg)


def verify_cache(cache: Path) -> dict:
    """⛔ Geometry is a CONTRACT, not a hint (see module docstring)."""
    idx = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    geo = idx.get("geometry", {})
    for k in ("n_tokens", "d_enc", "hfov_deg"):
        want, got = DINOV3_GEOMETRY[k], geo.get(k)
        if got != want:
            raise SystemExit(
                f"REFUSING this cache: geometry.{k} = {got!r}, v1 requires "
                f"{want!r}. A narrowed visual interface trains happily and is "
                "the defect v1 was built to remove.")
    return idx


# --------------------------------------------------------------------------- #
# PRECISION (Deploy & Optimization FlyWheel, D-REFAV1-STEP-PROFILE 2026-09-02).
# MEASURED on the dev-box 4060: the step IS the 30-step operative token-field
# rollout fwd+bwd (84-92 % of it, GPU busy 98 %), and bf16 autocast makes that
# rollout 3.0x faster (TF32 alone 1.54x, forward). Thor's own ratio is
# UNMEASURED. ⛔ Both switches default OFF: the live run must stay
# reproducible from the repo, and a bf16 arm is a NUMERICS change that needs
# its pre-registered A/B (loss, gnorm, participation, tgt_std_*) before it is
# trusted. The OFF path is byte-identical to the pre-flag trainer — pinned by
# `tests/test_refa_v1_precision.py` against the pre-edit step-1 numbers.
# --------------------------------------------------------------------------- #
def forward_context(precision: str, device_type: str):
    """The context the FORWARD + LOSS run under — and nothing else.

    ``fp32`` -> ``nullcontext`` (no autocast machinery is entered at all).
    ``bf16`` -> ``torch.autocast(bfloat16)`` on ``device_type``. PINNED
    DECISION: on CPU this autocasts ON CPU (torch supports bf16 there) rather
    than refusing — the smoke/test path then exercises the SAME context wiring
    the CUDA run uses, instead of a private path that could drift from it.
    """
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    raise ValueError(f"precision must be 'fp32' or 'bf16', got {precision!r}")


def apply_precision_flags(precision: str, tf32: bool, device) -> dict:
    """Set the process-global numerics BEFORE the model is built; return the
    READ-BACK record that goes into ``config.json`` (the stamp is what the
    process actually has, never an echo of the flag).

    ``--tf32`` off touches NOTHING, so the live run keeps PyTorch's own
    defaults (``matmul.allow_tf32`` False, ``cudnn.allow_tf32`` True) exactly
    as it has them today; on sets both True. Inert on CPU, stamped either way.
    """
    dev = torch.device(device)
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"precision must be 'fp32' or 'bf16', got {precision!r}")
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if (precision == "bf16" and dev.type == "cuda"
            and not torch.cuda.is_bf16_supported()):
        raise SystemExit(
            f"⛔ --precision bf16 on {dev}: this CUDA device reports no bf16 "
            "support. Refusing rather than silently running fp32 under a "
            "launch line that says bf16.")
    return {
        "precision": precision,
        # autocast covers the FORWARD + LOSS only; backward / clip / optimizer
        # / EMA run on the fp32 master weights (autocast backward emits fp32
        # parameter grads). No GradScaler: bf16 keeps fp32's 8-bit exponent,
        # so the gradient UNDERFLOW that fp16 loss-scaling exists to prevent
        # does not arise — bf16's cost is mantissa (8 bits), not range.
        "autocast": ({"device_type": dev.type, "dtype": "bfloat16",
                      "scope": "forward+loss"}
                     if precision == "bf16" else None),
        "grad_scaler": None,
        "master_weights": "fp32",
        "tf32": {"requested": bool(tf32),
                 "effective": bool(tf32) and dev.type == "cuda",
                 "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                 "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                 "float32_matmul_precision":
                     torch.get_float32_matmul_precision()},
        "device": str(dev),
    }


def _jsonable(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def write_config(out: Path, args, cfg, precision: dict) -> Path:
    """``<out>/config.json`` — the launch line, the resolved model config and
    the resolved numerics, so a run's precision is READABLE from its directory
    (the trainer wrote no config record at all before 2026-09-02; the ckpt
    carries ``cfg`` only, and precision is a trainer-side property that is
    deliberately NOT in the model config — a checkpoint loads under either)."""
    rec = {"args": dict(vars(args)), "cfg": dict(vars(cfg)), **precision,
           "torch": torch.__version__,
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = out / "config.json"
    p.write_text(json.dumps(rec, indent=1, default=_jsonable), encoding="utf-8")
    return p


class SmokeData:
    """Random fields with the right shapes — proves the loop, never a number.

    Same dict contract as `RefAV1Windows.batch` (incl. ``v0``), so the smoke
    path exercises the SAME forward-kwarg filter as a real run instead of a
    private tuple path that could drift from it."""

    def __init__(self, cfg: RefAV1Config, bs: int):
        self.cfg, self.bs = cfg, bs

    def batch(self, bs: int | None = None) -> dict:
        c, bs = self.cfg, (self.bs if bs is None else int(bs))
        return {"feats": torch.randn(bs, c.op_window, c.n_tokens, c.d_enc),
                "actions": torch.randn(bs, c.op_steps, c.a_dim) * 0.1,
                "future_feats": torch.randn(bs, c.op_steps, c.n_tokens,
                                            c.d_enc),
                # a "measured" anchor speed in m/s — the loader's `v0` role
                "v0": torch.rand(bs) * 20.0}


# --------------------------------------------------------------------------- #
# RESUME x EMA (Architecture & Inference FlyWheel, D-REFAV1-EMA-RESUME
# 2026-09-03). MEASURED, E-ARCH-TSC-2 R7 (`TanitAD Research Lab/Architecture &
# Inference/Research/2026-09-02-refav1-ema-inflation/raw/R7_resume.log`):
# `--resume --ema-targets` on a checkpoint written WITHOUT `--ema-targets` died
# in 2.8 s — `Missing key(s) in state_dict: "ema.tac_queries",
# "ema.adapter.pos", … "ema.str_read.1.bias"`, 20 keys, strict load, no
# teacher-from-student init. So a run could NOT be switched to EMA targets at a
# checkpoint: register row C-REFAV1-TAC-INFLATION option (b) was blocked by
# exactly this, and the clean-epoch restart only routed around it.
# --------------------------------------------------------------------------- #
def load_resume_state(model: RefAV1, state: dict, step: int) -> dict:
    """Load a resume checkpoint's ``model`` state_dict STRICTLY — with exactly
    one named exception, and a refusal on its mirror image.

    THE EXCEPTION — teacher from student. When the model carries a teacher
    (``--ema-targets``) and the checkpoint lacks EXACTLY the ``ema.*`` keys
    and nothing else, the teacher is INITIALISED FROM THE LOADED STUDENT: a
    copy, not a lerp — what BYOL / I-JEPA do at their own step 0 — so the EMA
    walks from the checkpointed student instead of from a random init. The
    teacher entries are synthesised INTO the state_dict from the student's,
    keyed through ``_EmaTargetPath.pairs`` (the one source of the
    teacher<->student pairing; no hand-written name table to drift from it),
    and the whole dict then goes through ONE ``strict=True`` load. So any
    OTHER missing or unexpected key — a PARTIAL teacher included — still
    refuses with the key named. There is no ``strict=False`` on this path.

    THE MIRROR IMAGE — PINNED DECISION (`tests/test_refa_v1_ema_resume.py`):
    a checkpoint that HAS ``ema.*`` keys under a launch with ``--ema-targets``
    OFF is REFUSED (SystemExit), not silently stripped. Dropping the teacher
    continues the run as a DIFFERENT experiment — the tactical/strategic
    targets fall back from the slow teacher to the live student path — under
    a launch line that says nothing about it, and the run would look healthy.
    That is the stale-manifest relaunch (`supervise_run.sh` replays the
    TRAIN_CMD it booted with) meeting the silently-narrower-experiment class
    this trainer already refuses for --labels and for cache geometry. A
    deliberate teacher-off continuation is a NEW ARM: strip the ``ema.*``
    keys into a new checkpoint on purpose and register it, so the lineage
    shows the choice instead of an omitted flag.

    Returns ``{"ema_init": "student" | "checkpoint" | None, "n_ema_keys":
    int, "step": int}`` — what happened, for the caller and the log.
    """
    want = set(model.state_dict())
    have = set(state)
    ema_want = {k for k in want if k.startswith("ema.")}
    ema_have = {k for k in have if k.startswith("ema.")}
    missing_other = sorted((want - have) - ema_want)
    unexpected_other = sorted((have - want) - ema_have)
    if missing_other or unexpected_other:
        raise RuntimeError(
            "Error(s) in loading state_dict for RefAV1 (--resume): "
            + (f"Missing key(s) in state_dict: {missing_other}. "
               if missing_other else "")
            + (f"Unexpected key(s) in state_dict: {unexpected_other}. "
               if unexpected_other else "")
            + f"[ema.* keys: model wants {len(ema_want)}, checkpoint has "
              f"{len(ema_have)} — the teacher is initialised from the student "
              "ONLY when nothing else is wrong]")
    if ema_have and not ema_want:
        raise SystemExit(
            f"⛔ [refav1] resume REFUSED: the checkpoint carries an EMA "
            f"teacher ({len(ema_have)} ema.* keys, written under "
            "--ema-targets) but this launch has --ema-targets OFF. Dropping "
            "the teacher would continue the run as a DIFFERENT experiment "
            "(tactical/strategic targets fall back from the teacher to the "
            "live student path) under a launch line that says nothing about "
            "it. Pass --ema-targets to continue the run as trained; a "
            "deliberate teacher-off continuation is a NEW arm from a "
            "checkpoint stripped of its ema.* keys on purpose — never an "
            "omitted flag.")
    state = dict(state)                       # never mutate the caller's dict
    init = None
    if ema_want and not ema_have:
        name_of = {id(p): n for n, p in model.named_parameters()}
        added = set()
        for p_ema, p_stu in model.ema.pairs(model):
            k_ema, k_stu = name_of[id(p_ema)], name_of[id(p_stu)]
            state[k_ema] = state[k_stu].detach().clone()
            added.add(k_ema)
        if added != ema_want:
            raise RuntimeError(
                "EMA teacher parameters not covered by _EmaTargetPath.pairs(): "
                f"{sorted(ema_want ^ added)} — refusing to leave a teacher "
                "parameter at its random init")
        init = "student"
    elif ema_want:
        init = "checkpoint"           # strict below: a partial teacher raises
    res = model.load_state_dict(state, strict=True)
    if res.missing_keys or res.unexpected_keys:      # strict=True has raised;
        raise RuntimeError(f"resume load left keys unmatched: {res}")  # belt
    if init == "student":
        print(f"[refav1] resume: EMA targets initialised from the student "
              f"(checkpoint had no ema.* keys; step {step})", flush=True)
    elif init == "checkpoint":
        print(f"[refav1] resume: EMA teacher restored from the checkpoint "
              f"({len(ema_have)} ema.* keys; step {step})", flush=True)
    return {"ema_init": init, "n_ema_keys": len(ema_want), "step": int(step)}


def verify_resume_optimizer(model: RefAV1, opt: torch.optim.Optimizer) -> None:
    """After ``opt.load_state_dict`` on resume: the teacher is FROZEN and
    OUTSIDE the optimizer. The optimizer is built from the ``requires_grad``
    filter in ``main``, so this holds by construction today; it is asserted so
    that the day a refactor hands the teacher to AdamW (a silent weight-decay
    on the target) the resume dies here instead of training on it."""
    if model.ema is None:
        return
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    for n, p in model.ema.named_parameters():
        if p.requires_grad:
            raise RuntimeError(f"EMA teacher parameter ema.{n} has "
                               "requires_grad=True after resume")
        if id(p) in in_opt:
            raise RuntimeError(f"EMA teacher parameter ema.{n} is inside the "
                               "optimizer after resume")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    help="stage-1 DINOv3 feature cache (0.2 s grid)")
    ap.add_argument("--episodes", type=Path,
                    help="the v2ep episode dir (actions/poses at 10 Hz)")
    ap.add_argument("--lru", type=int, default=32,
                    help="episodes held in RAM (each ~130 MB fp16 fields)")
    # ⭐ THE v7.2 JOIN. The loader has accepted these since 2026-09-01 and the
    # model has masked -100 correctly for longer; only the TRAINER could not
    # pass them, so a real run trained the trajectory path with the tactical and
    # strategic heads unsupervised while every component reported itself ready.
    ap.add_argument("--labels", type=Path, default=None,
                    help="v7.2 s2 labels .jsonl.gz — supervises the tactical "
                         "and strategic heads. REQUIRED with a real --cache; "
                         "out-of-band windows emit -100 and the model skips "
                         "that family rather than averaging a NaN")
    ap.add_argument("--nav", type=Path, default=None,
                    help="nav/route source joined on clip_id, fed to all three "
                         "layers. ⛔ must NOT carry the situation classifier's "
                         "output in any form (PI 2026-08-03)")
    ap.add_argument("--allow-unlabelled", action="store_true",
                    help="⛔ deliberate: run a real --cache WITHOUT --labels. "
                         "Only for a diagnostic that does not touch the "
                         "tactical/strategic heads — never for a registered arm")
    ap.add_argument("--out", type=Path, default=Path("./refa_v1_run"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--adapter-lr-mult", type=float, default=0.1,
                    help="REF-A stability item 4: the adapter warms up SLOWER "
                         "than the predictor (10x longer warmup == 0.1x LR).")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--no-hierarchy", action="store_true",
                    help="ablation arm: drop both brains (they are a matched "
                         "set) — the change-#7 control")
    # --- change #10 was UNREACHABLE FROM THE LAUNCH LINE until these existed - #
    # ⛔ The counterfactual term shipped in the model with no flag to turn it
    # on. That is the advertised-but-inert defect one level out: `sanity()`
    # refuses w_cf with zero negatives, and meanwhile NO launch could set w_cf
    # at all. A capability with no switch is not a capability.
    ap.add_argument("--w-cf", type=float, default=0.0,
                    help="weight on the counterfactual-action InfoNCE (change "
                         "#10). 0 = off. Its no-information floor is "
                         "ln(1+cf_negs), so cf_excess > 0 PROVES the predictor "
                         "used the action.")
    ap.add_argument("--cf-negs", type=int, default=3)
    ap.add_argument("--cf-at-step", type=int, default=4)
    # --- PI 2026-08-31: "implement and try 1" ------------------------------- #
    ap.add_argument("--motion-inject", action="store_true",
                    help="add a CROSS-CHANNEL projection of z_t - z_(t-1) to "
                         "the initial rollout state. Default history path is "
                         "the adapter's DEPTHWISE temporal conv only -- each "
                         "channel mixes its own past, so cross-channel motion "
                         "(parallax, an edge crossing patches) has no route "
                         "into the state.")
    # --- ANTI-COLLAPSE (PI 2026-09-02). Each SEPARATELY switchable so its
    # effect stays attributable; five changes in one arm would be the
    # conflation error this programme has already paid for. ---------------- #
    ap.add_argument("--detach-aux-targets", dest="detach_aux", default=None,
                    action="store_true",
                    help="stop-gradient on the tactical/strategic targets "
                         "(SimSiam 2011.10566). DEFAULT ON; --no-detach-aux-"
                         "targets is the deliberate-regression control")
    ap.add_argument("--no-detach-aux-targets", dest="detach_aux",
                    action="store_false")
    ap.add_argument("--w-sigreg", type=float, default=0.0,
                    help="SigReg weight on the adapter output (v6/v7 line). "
                         "0 = off. MEASURED discriminating: 0.477 random vs "
                         "2.61 collapsed. ⛔ DO NOT COPY v6's 0.1: on refav1's "
                         "input scale the raw term reads ~238, so 0.1 "
                         "contributes ~24 against a ~1.3 feature loss — a 20x "
                         "domination. CALIBRATE against the measured term "
                         "before use (a weight near 1e-3 puts it on scale)")
    ap.add_argument("--var-floor", type=float, default=0.0,
                    help="VICReg (2105.04906) per-dim std hinge on the adapter "
                         "output. 0 = off. Punishes std BELOW the floor only")
    ap.add_argument("--min-participation", type=float, default=0.0,
                    help="refuse if participation falls below this (RankMe "
                         "2210.02885 / G-RANK). 0 = MONITOR ONLY, always "
                         "logged. Reference floor 8.56; rank-1 collapse = 1.00")
    ap.add_argument("--bptt-truncate", type=int, default=15,
                    help="detach the carried rollout state every N steps. "
                         "0 = FULL CHAIN (the deliberate-regression control). "
                         "15 = DreamerV3's imagination horizon AND Looped-WM's "
                         "ceil(mu_rec/2) at our K=30 — two independent recipes "
                         "agreeing. Without it: gnorm 3.6e3 / 5.7e7 / inf "
                         "within 300 steps. The FORWARD rollout is unchanged; "
                         "only the gradient path is bounded")
    # --- #5 the speed channel, under the PI ruling of 2026-09-02 ------------ #
    ap.add_argument("--speed-channel", action="store_true",
                    help="third predictor input channel v_k/30 with v_k "
                         "INTEGRATED from the loader's measured anchor speed "
                         "v0 and the action sequence (v_k = v0 + sum_{j<k} "
                         "a_j dt) — never a future GT speed (PI 2026-09-02: "
                         "'velocity as initial measured state at its cycle "
                         "time' only). The planner's control space stays "
                         "(a, kappa). DEFAULT OFF — a next-run arm")
    ap.add_argument("--tmix-groups", type=int, default=None,
                    help="WideAdapter.tmix conv groups: default None = "
                         "d_state (depthwise, the shipped form); 1 = full "
                         "cross-channel temporal mixing (next-run arm)")
    ap.add_argument("--ema-targets", action="store_true",
                    help="tactical/strategic targets from an EMA copy of the "
                         "target path behind a stop-gradient (BYOL 2006.07733, "
                         "I-JEPA 2301.08243). SPEC E-ARCH-TSC-1 §3: 'frozen' "
                         "pins the OPERATIVE target only; these two still "
                         "derive from the trained adapter and shrink with it. "
                         "DEFAULT OFF — a next-run arm; the live run must stay "
                         "reproducible from the repo")
    ap.add_argument("--ema-decay", type=float, default=0.996,
                    help="EMA decay at step 0 (BYOL/I-JEPA 0.996)")
    ap.add_argument("--ema-decay-end", type=float, default=0.999,
                    help="EMA decay at the last step (linear schedule; I-JEPA "
                         "ends at 1.0, pinned at 0.999 so the teacher never "
                         "fully freezes)")
    ap.add_argument("--target-space", choices=("adapter", "frozen"),
                    default="frozen",
                    help="'adapter' = original form, whose primary loss has a "
                         "COLLAPSE MINIMUM (the target passes the trained "
                         "adapter); 'frozen' = predict std(DINOv3) itself -- "
                         "fixed target variance, minimum removed (DINO-WM).")
    # --- MM-E19 carried over: full-chain BPTT needs a tighter clip ---------- #
    ap.add_argument("--clip", type=float, default=1.0,
                    help="grad-norm clip. ⚠️ MM-E19 MEASURED a 60-step "
                         "full-chain rollout diverging (gnorm 2.1e9) at the "
                         "loose default and surviving at 0.5. v1 rolls 30 "
                         "steps with full-chain gradient through an 80 M "
                         "predictor — consider 0.5 for the first real arm.")
    # --- Drive-JEPA-adapted multimodal proposals (2026-09-01) --------------- #
    ap.add_argument("--w-aux-head", type=float, default=0.0,
                    help="imitation weight on the proposal head (WTA over "
                         "proposal-k modes vs the demonstrated (a, kappa)). "
                         "SEEDS the planner only -- behaviour stays planned.")
    ap.add_argument("--proposal-k", type=int, default=1,
                    help="number of proposal modes (Drive-JEPA proposal-set "
                         "idea); >1 adds a score head, and at plan() time all "
                         "modes join the iCEM seed pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/ckpt.pt if present")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    # --- PRECISION (D-REFAV1-STEP-PROFILE, 2026-09-02) --------------------- #
    ap.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                    help="bf16 = torch.autocast(bfloat16) around the FORWARD "
                         "+ LOSS only; backward, clip, optimizer and EMA stay "
                         "on the fp32 master weights, no GradScaler (bf16 has "
                         "fp32's exponent range). MEASURED dev-box 4060: the "
                         "30-step operative rollout fwd+bwd 3.0x faster, and "
                         "that rollout is 84-92 %% of the step. ⛔ DEFAULT "
                         "fp32 — the live run must stay reproducible; a bf16 "
                         "arm is a NUMERICS change with its own A/B")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 tensor-core matmuls/convs (sets "
                         "torch.backends.cuda.matmul.allow_tf32 AND "
                         "torch.backends.cudnn.allow_tf32 True before the "
                         "model is built; off touches nothing). MEASURED "
                         "1.54x on the rollout forward. Inert on CPU; the "
                         "read-back flags are stamped into config.json")
    a = ap.parse_args(argv)

    if not a.smoke and (a.cache is None or a.episodes is None):
        raise SystemExit("--cache AND --episodes are required unless --smoke")
    # ⛔ REFUSE AN UNSUPERVISED REAL RUN — and refuse it HERE, before
    # `verify_cache` touches the disk, so the message is about the mistake and
    # not about whichever file the cache reader happened to open first.
    # The PI made v7.2 tactical/strategic labels and nav-to-all-layers
    # mandatory. Without --labels those heads take NO gradient and the run
    # still LOOKS healthy — losses fall, checkpoints land, the trajectory path
    # learns — so nothing in the log would say the hierarchy was never trained.
    # That is what this refuses: not a crash, a silently narrower experiment.
    if a.cache is not None and not a.labels and not a.allow_unlabelled:
        raise SystemExit(
            "⛔ --cache given without --labels: the tactical and strategic "
            "heads would take NO gradient and the run would still look "
            "healthy. Pass --labels <s2_labels_v7.2_*.jsonl.gz> (and --nav), "
            "or --allow-unlabelled if this is a diagnostic that deliberately "
            "does not touch those heads.")
    if a.cache is not None and not a.nav:
        print("⚠️ [refav1] no --nav: the nav command reaches no layer. "
              "Admissible only for an arm that does not claim route "
              "conditioning.", flush=True)
    if a.cache is not None:
        verify_cache(a.cache)

    # Numerics are resolved BEFORE the model exists: TF32 is a process-global
    # switch that must precede the first matmul, and an unsupported bf16
    # device is refused here, not after the cache is opened.
    precision = apply_precision_flags(a.precision, a.tf32, a.device)
    fwd_device_type = torch.device(a.device).type

    torch.manual_seed(a.seed)
    model = build_model(a).to(a.device)
    cfg = model.cfg
    # ⛔ SANITY BEFORE THE SPEND, NOT AFTER. `sanity()` is what refuses an
    # inexpressible ladder (a rate that is not an integer multiple of op_dt) and
    # an advertised-but-inert counterfactual term. It was never called here, so
    # a 30k-step arm could have run to completion on a config the model itself
    # would have rejected.
    cfg.sanity()
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"[refav1] config -> {write_config(a.out, a, cfg, precision)} "
          f"(precision={precision['precision']} tf32={precision['tf32']})",
          flush=True)

    # Stability item 4: adapter and predictor are SEPARATE param groups.
    # ⛔ `requires_grad` filter: the EMA teacher's parameters (`model.ema.*`,
    # when --ema-targets) are frozen copies moved ONLY by `model.ema_update`;
    # handing them to AdamW would be harmless today (no grad ⇒ skipped) and a
    # silent weight-decay on the teacher the day that changes.
    adapter_p = [p for p in model.adapter.parameters() if p.requires_grad]
    ids = {id(p) for p in adapter_p}
    rest_p = [p for p in model.parameters()
              if p.requires_grad and id(p) not in ids]
    opt = torch.optim.AdamW(
        [{"params": adapter_p, "lr": a.lr * a.adapter_lr_mult},
         {"params": rest_p, "lr": a.lr}], weight_decay=0.01)

    if a.smoke:
        data = SmokeData(cfg, a.bs)
        with torch.no_grad():
            model.std.fit(data.batch()["feats"].to(a.device))
    else:
        # ⭐ THE LOADER GAP IS CLOSED (2026-09-01): real windows over the
        # stage-1 cache + v2ep kinematics. The loader emits (a, kappa) with the
        # MEASURED channel repair (v2ep stores kappa first, r=0.995) and the
        # str-extension pairs.
        # ⭐⭐ AND THE v7.2 LABEL / NAV JOIN IS NOW WIRED (2026-09-02). The line
        # above used to end "labels/nav join is the next increment" — the LOADER
        # had supported `labels_path`/`nav_path` for a day, and the model side
        # had supported `-100` masking for longer, but the TRAINER had no flag
        # to supply either. So a real run would have trained the trajectory path
        # with the tactical/strategic heads unsupervised and nav absent, while
        # every component reported itself ready. A capability that exists at
        # both ends and is not connected in the middle is not a capability.
        from tanitad.data.refav1_loader import RefAV1Windows
        data = RefAV1Windows(a.cache, a.episodes, op_window=cfg.op_window,
                             op_steps=cfg.op_steps, str_dt=cfg.str_dt,
                             str_ext_steps=cfg.str_ext_steps,
                             lru=a.lru, seed=a.seed,
                             labels_path=a.labels, nav_path=a.nav)
        print(f"loader: {len(data)} windows over {len(data.names)} episodes")
        print(f"loader: labels={a.labels or 'NONE'} nav={a.nav or 'NONE'}",
              flush=True)
        with torch.no_grad():
            fit = data.batch(max(a.bs, 8))["feats"]
            model.std.fit(fit.reshape(-1, cfg.d_enc).to(a.device))

    # ⛔ THE AUXILIARY HEAD IS ADVERTISED AND INERT — REFUSED RATHER THAN FAKED.
    # The loop read `out["loss"] + cfg.w_aux_head * torch.zeros(())`: the
    # config carries w_aux_head 0.1, the launch record would show it, and the
    # term contributes EXACTLY nothing because no imitation target is wired.
    # That is the defect `sanity()` refuses for w_cf, sitting in the trainer.
    # ⚠️ Fixing it means supplying a proposal target, which the cache contract
    # does not yet carry — so this REFUSES instead of pretending.
    if cfg.w_aux_head and a.smoke:
        raise SystemExit(
            f"w_aux_head is {cfg.w_aux_head} under --smoke: SmokeData has no "
            "demonstrated actions, so the term would be advertised and fed "
            "noise. With a real --cache the loader's (a, kappa) IS the demo "
            "and the term is live.")

    start_step = 0
    if a.resume and (a.out / "ckpt.pt").exists():
        ck = torch.load(a.out / "ckpt.pt", map_location=a.device,
                        weights_only=False)
        start_step = int(ck["step"])
        # ⭐ Strict, with ONE named exception (teacher initialised from the
        # student when the checkpoint has no ema.* keys) and ONE refusal
        # (teacher present, --ema-targets off) — `load_resume_state`,
        # D-REFAV1-EMA-RESUME. The optimizer state is the student's and loads
        # as before; the teacher never enters it (asserted, not assumed).
        load_resume_state(model, ck["model"], start_step)
        opt.load_state_dict(ck["opt"])
        verify_resume_optimizer(model, opt)
        print(f"resumed from step {start_step}")

    # the set  will actually take — derived from the model, never a
    # hand-maintained list that would drift from it
    import inspect as _inspect
    _FWD_PARAMS = {n for n in _inspect.signature(model.forward).parameters
                   if n != "self"}

    log = (a.out / "train_log.jsonl").open("a", encoding="utf-8")
    t0 = time.time()
    for step in range(start_step + 1, a.steps + 1):
        # One path for smoke and real data: both loaders emit the same dict
        # contract, so the smoke run exercises the SAME kwarg filter (it used
        # to take a private tuple path that could not carry `v0`).
        b = data.batch(a.bs)
        feats = b["feats"].to(a.device)
        actions = b["actions"].to(a.device)
        future = b["future_feats"].to(a.device)
        # ⛔ FILTER BY THE MODEL'S ACTUAL SIGNATURE, AND SAY WHAT WAS
        # DROPPED. This used to forward EVERY batch key, which worked only
        # while the loader emitted exactly what `forward` accepted. Turning
        # on --labels/--nav made the loader emit its validity masks
        # (`nav_valid`, …) and the run died on
        # `TypeError: RefAV1.forward() got an unexpected keyword argument`.
        # A blind pass-through is a contract between two files that nobody
        # checks; it breaks the moment either side grows a field.
        # ⚠️ The report matters as much as the filter: dropping a VALIDITY
        # MASK is correct (the model masks -100 itself), but silently
        # dropping a LABEL would narrow the experiment invisibly — the
        # exact failure the --labels guard exists to prevent. So name them
        # once, and let the reader judge which kind they are.
        kw = {k: (v.to(a.device) if torch.is_tensor(v) else v)
              for k, v in b.items()
              if k in _FWD_PARAMS
              and k not in ("feats", "actions", "future_feats")
              and v is not None}
        if step == start_step + 1:
            dropped = sorted(set(b) - _FWD_PARAMS
                             - {"feats", "actions", "future_feats"})
            print(f"[refav1] forward() consumes {sorted(kw)}", flush=True)
            if dropped:
                print(f"⚠️ [refav1] loader emits but forward() does NOT "
                      f"accept: {dropped} — verify each is a validity mask "
                      f"(safe: the model masks -100 itself) and not a "
                      f"label (which would narrow the run silently)",
                      flush=True)
        # ⭐ PRECISION: the context wraps the FORWARD + LOSS only. `out["loss"]`
        # is assembled inside `forward`, so this one block covers every term;
        # backward / clip / opt.step / ema_update run OUTSIDE it on the fp32
        # master weights. Under fp32 it is a nullcontext — byte-identical to
        # the pre-flag trainer (pinned by tests/test_refa_v1_precision.py).
        # Under bf16 the instruments the model emits are still read at fp32 /
        # fp64: `_chan_std` casts `.float()` BEFORE its reduction, the
        # participation covariance is fp64 (autocast never touches fp64), and
        # every `mse_loss` / `cross_entropy` is on autocast's fp32 list.
        with forward_context(a.precision, fwd_device_type):
            out = model(feats, actions, future_feats=future, **kw)
            loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        # EMA teacher follows the student AFTER the optimizer step, on the
        # linear decay schedule over the whole run (I-JEPA 2301.08243).
        ema_decay = (model.ema_update(step, a.steps)
                     if cfg.ema_targets else None)

        if step % a.log_every == 0 or step == 1:
            with torch.no_grad():
                # THE COLLAPSE MONITOR, carried over from REF-A: adapter output
                # per-dim std. Part 2 measured the trained REF-A adapter at
                # 0.8011 vs 0.220 random-init — i.e. NOT collapsed. If v1 ever
                # drives this toward 0 the run is dead regardless of the loss.
                # Deliberately OUTSIDE the precision context: it reads the fp32
                # master weights' adapter, and `.float()` pins the reduction to
                # fp32 should this block ever move under autocast (a bf16 std
                # keeps ~3 significant digits). No-op under fp32.
                adapter_std = float(model.encode(feats).float()
                                    .std(dim=(0, 1, 2)).mean())
            row = {"step": step, "loss": float(loss.detach()),
                   "precision": a.precision, "tf32": bool(a.tf32),
                   "loss_feat_op": float(out["loss_feat_op"].detach()),
                   "loss_feat_tac": float(out["loss_feat_tac"].detach()),
                   "loss_feat_str": float(out["loss_feat_str"].detach()),
                   "grad_norm": float(gnorm), "adapter_std": adapter_std,
                   # ⭐ the target's own scale — a loss is only interpretable
                   # against the variance of what it predicts (see refa_v1.py
                   # target-scale instrument). In "frozen" these sit at ~1;
                   # in "adapter" they are free to collapse with the loss.
                   "participation": out.get("participation"),
                   "loss_sigreg": (float(out["loss_sigreg"])
                                   if "loss_sigreg" in out else None),
                   "loss_varfloor": (float(out["loss_varfloor"])
                                     if "loss_varfloor" in out else None),
                   "tgt_std_op": out.get("tgt_std_op"),
                   "tgt_std_tac": out.get("tgt_std_tac"),
                   "tgt_std_str": out.get("tgt_std_str"),
                   # None unless --ema-targets: tgt_std_tac/str then read the
                   # TEACHER's targets, and this is the decay that moved it.
                   "ema_decay": ema_decay,
                   "clip": a.clip,
                   # ⭐ THE REALISED LADDER, IN EVERY ROW. The horizon a level
                   # actually trains on is now readable from the log instead of
                   # inferred from the config — which is how a strategic rung
                   # ran at 1.6 s reaching 5.0 s while its config said 1.5/6.0
                   # and every test passed.
                   "tac_target_s": [round((k + 1) * cfg.op_dt, 3)
                                    for k in out["tac_target_idx"]],
                   "str_target_s": [round((k + 1) * cfg.op_dt, 3)
                                    for k in out["str_target_idx"]],
                   "elapsed_s": round(time.time() - t0, 1)}
            if cfg.w_cf:
                # ⭐ cf_excess is measured against a KNOWN floor, ln(1+cf_negs):
                # > 0 proves the predictor used its action, with no baseline to
                # argue about. This is the number the arm exists to move.
                row.update(cf_loss=float(out["cf_loss"].detach()),
                           cf_no_info_floor=out["cf_no_info_floor"],
                           cf_excess=out["cf_excess"])
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row))
            if not math.isfinite(row["loss"]):
                raise SystemExit("non-finite loss — refusing to continue")

        if step % a.save_every == 0 or step == a.steps:
            torch.save({"step": step, "model": model.state_dict(),
                        "opt": opt.state_dict(), "cfg": vars(cfg)},
                       a.out / "ckpt.pt")
    log.close()
    print(f"done: {a.steps} steps, {model.trainable_parameters():,} trainable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
