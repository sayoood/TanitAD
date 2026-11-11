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

from tanitad.data import v8f_dagger as DG  # noqa: E402      PREREG_V8F_DAGGER: the own-state bank + the swap (the DATA library; `--dagger-p 0` = no bank is read and no row is swapped)
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.data import v8f_wm_windows as WW  # noqa: E402      PREREG_V8F_LADDER: the world-model-only window sampler + the pooled path's data half (`--wm-windows train` = it is never called)
from tanitad.data import v8f_t2bank as TB  # noqa: E402       PREREG_T2_DAGGER: the T2 own-state bank + the whole-sample swap (`--t2-bank-p 0` = no bank is read and nothing is applied)
from tanitad.data import v8f_recover as R  # noqa: E402      PREREG_RECOVER_FT: the recovery perturbation + its TRAIN-bank deviation table (`--recover-p 0` = nothing is built and nothing is applied)
from tanitad.models import v8f_mini as M  # noqa: E402

PKG = os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini")
#: PREREG_V8F_DAGGER K-B3: the eval-141 clip list the own-state bank must be disjoint from (the same file the bank generator is given as --eval-grid; the trainer takes no flag for it)
DAGGER_EVAL_GRID = os.path.join(PKG, "raw", "eval_grid_windows.json")

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
    "o_interface": "lever:o_interface",       # PREREG_V8F_FLAT: argv -> make_config -> the BUILT model's levers (declared_vs_built) -> G-LIVE's dead-weight set (live_check) -> the probe of what O is handed (o_interface_probe)
    "w_acc": "cfg:w_acc",                     # PREREG_V8F_OACC: argv -> make_config -> the BUILT config AND built_levers (declared_vs_built) -> acc_probe (the BUILT loss's total moves by exactly w_acc x the acc term, and O's head receives its gradient)
    "launch_intent_w": "cfg:launch_intent_w",  # PREREG_V8F_LAUNCH_INTENT: argv -> make_config -> the BUILT config AND built_levers (declared_vs_built) -> launch_intent_probe (G-LIVE: the mask is the loop's own call `launch_stage`, the BUILT loss's total moves by exactly w_traj x W x term / 3 on launch rows, O's head receives its gradient, an all-False mask reads EXACTLY 0.0) -> `b["launch"] = lmask` in the loop
    # PREREG_V8F_HACC_MC section 7 (O-RECOVER): the ego-state perturbation. `ego:<k>` = equals the BUILT perturber's lever (D.EgoPerturb.built_levers(), the object the training loop calls
    # `.apply` on): argv -> make_ego_perturb -> the perturber -> declared_vs_built (bad_ego_perturb) -> G-LIVE on a perturbed batch + ego_perturb_probe (the perturbed tensors reach the loss
    # gradient, a p = 0 control reads EXACTLY 0.0) -> `b = ego.apply(b, step)` in the loop. With --ego-perturb-p 0 (the default) none of it runs and nothing it imports is imported.
    "ego_perturb_p": "ego:ego_perturb_p", "ego_perturb_vscale": "ego:ego_perturb_vscale", "ego_perturb_vmax": "ego:ego_perturb_vmax", "ego_perturb_adrop": "ego:ego_perturb_adrop",
    "ego_perturb_yawrate_std": "ego:ego_perturb_yawrate_std", "ego_perturb_seed": "ego:ego_perturb_seed",
    # PREREG_V8F_DAGGER (H-V8F-DAGGER-1): the on-policy own-state swap. `dagger:<k>` = equals the BUILT swap's lever (DG.DaggerSwap.built_levers(), the object the training loop calls `.apply` on):
    # argv -> make_dagger (bank load + K-B2 + K-B3 + COVERAGE) -> the swap -> declared_vs_built (bad_dagger) -> G-LIVE on a swapped batch + dagger_probe (the swapped own-state tensors reach the loss
    # gradient, a p = 0 control reads EXACTLY 0.0) -> `b = dag.apply(b, step)` in the loop. With --dagger-p 0 (the default) no bank is read and no row is swapped.
    "dagger_p": "dagger:dagger_p", "dagger_bank": "dagger:dagger_bank", "dagger_bank_sha256": "dagger:dagger_bank_sha256", "dagger_seed": "dagger:dagger_seed",
    # AMEND1 (DAG-ORC re-based on DAG-OACC) section 1: the two levers above composed in ONE arm. `compose:<k>` = equals the BUILT stage's lever (EgoDaggerCompose.built_levers(), the object the loop calls `.apply` on):
    # argv -> make_compose -> the stage -> declared_vs_built (bad_ego_dagger; bad_dagger lifts the exclusivity ONLY for an argv that names it) -> G-LIVE on a COMPOSED batch (ego_dagger_compose_probe: both levers reach the
    # loss gradient, the p = 0 controls read EXACTLY 0.0) -> `b = comp.apply(b, step)` in the loop. Off (the default): the stage is never called and the loop runs its two original lines.
    "ego_perturb_with_dagger": "compose:ego_perturb_with_dagger",
    # PREREG_RECOVER_FT (H-RECOVER-FT-1): the recovery fine-tune. `rec:<k>` = equals the BUILT perturber's lever (R.RecoverPerturb.built_levers(), the object the training loop calls `.apply` on; the bank sha256 it reports is
    # the LOADED manifest's, not the declared string): argv -> make_recover (bank load + K-B3 + the deviation table) -> the perturber -> declared_vs_built (bad_recover) -> G-LIVE on a RECOVERED batch (recover_probe:
    # the perturbed tensors AND the re-expressed target reach the loss gradient, a p = 0 control reads EXACTLY 0.0) -> `b = rec.apply(b, step)` in the loop. With --recover-p 0 (the default) no bank is read and nothing is applied.
    "recover_p": "rec:recover_p", "recover_dev_bank": "rec:recover_dev_bank", "recover_dev_bank_sha256": "rec:recover_dev_bank_sha256", "recover_dev_law": "rec:recover_dev_law",
    "recover_dev_shape": "rec:recover_dev_shape",
    # PREREG_T2_DAGGER: the T2 own-state bank. `t2:<k>` = equals the BUILT swap's lever (TB.T2BankSwap.built_levers(), the object the loop calls `.apply` on; the sha256 it reports is the LOADED manifest's): argv -> make_t2bank
    # (bank load + sha256) -> the swap -> declared_vs_built (bad_t2bank) -> G-LIVE (t2bank_probe: bank rows carry ONLY the target law, every masked term reads EXACTLY 0.0, a p = 0 control is the same batch object) ->
    # `b = t2b.apply(b, step)` in the loop. With --t2-bank-p 0 (the default) no bank is read and nothing is applied.
    "t2_bank_p": "t2:t2_bank_p", "t2_bank": "t2:t2_bank", "t2_bank_sha256": "t2:t2_bank_sha256", "t2_bank_seed": "t2:t2_bank_seed",
    "recover_target": "rec:recover_target", "recover_rejoin_steps": "rec:recover_rejoin_steps", "recover_seed": "rec:recover_seed",
    # PREREG_V8F_LEVEL (lever (a) of DECISION_K2): the vision-only speed-level head. `lever:level_head` = equals built_levers().get("level_head", "off") (the BUILT model: lvl_head + e_level exist iff on); `cfg:w_level` = equals the BUILT config's
    # weight: argv -> make_config -> the BUILT model -> declared_vs_built (bad_level: a head with no loss, a weight with no head, a non-zero e_level at init, a head whose code names a non-vision input are REFUSED) -> level_head_probe (G-LIVE:
    # the BUILT loss moves by exactly w_level x the term, the head receives its gradient, the detach control reads EXACTLY 0.0, the token reaches O's output, vision-only guard) -> `level_stage(b)` BEFORE the ego stages in the loop.
    "level_head": "lever:level_head", "w_level": "cfg:w_level",
    # PREREG_LEVELHEAD_A2 (arm LLR): the lead head + how O reads the heads. `lever:lead_head` = built_levers().get("lead_head", "off"); `cfg:w_lead` = the BUILT config's weight; `lever:aux_cond` = built_levers().get("aux_cond", "add")
    # (the BUILT wiring: e_lead / e_level exist iff add, e_aux iff token and a head is on). argv -> make_config -> the BUILT model -> declared_vs_built (bad_lead) -> lead_head_probe (G-LIVE) -> `lead_stage(b, LL)` BEFORE the ego stages.
    "lead_head": "lever:lead_head", "w_lead": "cfg:w_lead", "aux_cond": "lever:aux_cond",
    # PREREG_V8F_MRATE: `lever:wm_k` = equals built_levers().get("wm_k", 1) (the BUILT modules' window / dilation / causal pad, WMAux.built_k); `lever:wm_only` = the record the WM-only loader read back (bad_wm_only: non-WM tensors bit-equal to the checkpoint,
    # NO WM tensor loaded, the WM re-initialised under --seed, trainable == the WM's parameters exactly, every other lever at its default) -> wm_only_live_check (G-LIVE) + planner_identity_probe -> `model.wm_only_loss(b)` in the loop.
    "wm_k": "lever:wm_k", "wm_only": "lever:wm_only",
    # PREREG_V8F_WMPOOL: `lever:wm_pool` = equals the mode WMAux.built_pool() MEASURES on the BUILT module (the cell map on a planted grid, uniform weights at a zero query, attention under a probe query, ONE zero-initialised [d] parameter), the BUILT dataset's record
    # (data_info["wm_windows"]["pool"]) AND the cell cache's manifest (kind / grid / dtype / adapter sha == the BUILT adapter's) -> bad_wm_pool. Default mean: no module, no key, no batch key.
    "wm_pool": "lever:wm_pool",
    # PREREG_V8F_LFS: `lever:wm_strides` = equals built_levers().get("wm_strides", [2, 2]) AND the BUILT modules (WMAux.built_strides: the LevelConfig strides the loss reads, the macro encoders' stride and the stride their parameter SHAPES carry) -> bad_wm_strides
    # (the gate also refuses a stride that leaves a level fewer than 2 transitions: the arm is NOT RUN, with the count) -> `selfcheck.json levers.wm_strides_readback` (written when the argv NAMES the flag).
    "wm_strides": "lever:wm_strides",
    # PREREG_V8F_LADDER: `lever:wm_rollout_k` = the predictor calls the BUILT loss makes inside HJEPAStack.rollout, per level (WMAux.built_ladder: a forward-hook count while the loss runs, never the config that built it);
    # `lever:wm_rollout_stride` = the free run's anchor count the built loss produces; `lever:wm_rollout_w` = (level loss - its other terms) / the k-step term; `lever:wm_ctx` = the predictor's parameter-shape context (pos.shape[0])
    # AND the context the free run is handed; `data:wm_windows` = the kind, F, first t and pooled flag READ BACK from the BUILT dataset (data_info["wm_windows"]) -> bad_wm_ladder.
    "wm_rollout_k": "lever:wm_rollout_k", "wm_rollout_stride": "lever:wm_rollout_stride", "wm_rollout_w": "lever:wm_rollout_w", "wm_ctx": "lever:wm_ctx",
    "wm_windows": "data:wm_windows", "wm_win_stride": "data:wm_windows", "wm_pooled": "data:wm_windows", "wm_zcache": "data:wm_windows",
    "init_ckpt": "data:init_ckpt",           # O-RECOVER's start state: the MODEL weights of an earlier checkpoint, config-checked, loaded strictly and READ BACK (load_init_ckpt) -> declared_vs_built (bad_init_ckpt)
    # PREREG_V8F_LORA: the encoder lever. `enc:<k>` = equals the BUILT encoder's lever (enc.built_levers()); with --encoder cache there is no encoder and every
    # one of these must be at its default (a declared flag that builds nothing is REFUSED, not ignored)
    "encoder": "enc:encoder", "lora_r": "enc:lora_r", "lora_alpha": "enc:lora_alpha", "lora_targets": "enc:lora_targets",
    "lora_lr": "opt:lora_lr", "frames_root": "data:frames_missing", "accum": "enc:accum", "enc_ckpt": "enc:checkpointing",
    # AMENDMENT_V8F_LORA_5: a resume may change EXACTLY --accum and --enc-ckpt (speed and memory, not the estimator). `perf:` = bad_perf_resume reads it back: the named previous argv, with those two flags
    # (and this flag) removed from both sides, is token for token the argv being launched; at resume the checkpoint's argv_sha must be that previous argv's.
    "resume_perf_levers": "perf:resume_perf_levers",
}

LORA_PARSER_DEFAULTS = {"lora_r": 32, "lora_alpha": 32.0, "lora_targets": "q,v", "lora_lr": None, "frames_root": None, "accum": 1, "enc_ckpt": False}
LIVE_PROBE_WINDOWS_WM = 8            # PREREG_V8F_LADDER: windows of the G-LIVE / identity probe batches in the wmonly mode (<= 8 x hmax frames; the CPU probes run the k-step loss)
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
    ap.add_argument("--o-interface", dest="o_interface", choices=M.O_INTERFACE_MODES, default="full",
                    help="PREREG_V8F_FLAT: what O reads of the layers above it. full (default) = S's checkpoints and T's code (every earlier arm, bit for bit, the RNG stream included); "
                         "cut = a CONSTANT in place of both (T's code = zeros after every override, S's checkpoints = the nominal ck_nom with ok = 0): the capacity-matched FLAT control. "
                         "S and T are still built, trained and run")
    ap.add_argument("--w-acc", dest="w_acc", type=float, default=0.0,
                    help="PREREG_V8F_OACC: the weight of the acceleration-matching term on O's plan (the second difference on the 0.2 s grid, origin knot included, of O's plan minus the GT's, squared, mean over the GT-valid steps). "
                         "0.0 (default) = the term is not in the loss = every earlier arm, bit for bit. > 0 = OACC's X: the value comes from the PREREG's rule (v8f_mini_eval_oacc.py wacc), is written into the argv and never tuned")
    ap.add_argument("--launch-intent-w", dest="launch_intent_w", type=float, default=0.0,
                    help="PREREG_V8F_LAUNCH_INTENT: W, the EXTRA weight on O's along-track distance over knots 1-12 (tau 0.2-2.4 s, the executed horizon at m = 10) for GT-launch TRAIN samples (the LOGGED v0 < 6 m/s and the GT chord speed over the "
                         "last 0.2 s of the 6 s horizon >= v0 + 1 m/s; label-side, nothing of it is read at inference). The term is W x the along-axis, knots 1-12 slice of ADE's own per-knot distance, on launch samples only: on a launch sample that slice "
                         "carries weight 1 + W; a non-launch sample's loss is unchanged. 0.0 (default) = the term is not computed and not in the loss = every earlier arm, bit for bit. > 0 = the value comes from the PREREG's dose rule, is written into the argv and never tuned")
    # ---- PREREG_V8F_LEVEL (lever (a) of DECISION_K2). Both flags default to OFF and have their G-DVB entry in FLAG_CHECKS and their clause in bad_level.
    ap.add_argument("--level-head", dest="level_head", choices=M.LEVEL_HEAD_MODES, default="off",
                    help="PREREG_V8F_LEVEL: off (default) = nothing is built = every earlier arm, bit for bit (state dict AND outputs). on = a head on the mean of the CURRENT tick frame's adapted VISION tokens -> MLP(d, d, 1) -> v_hat (m/s), "
                         "trained with --w-level x Huber(v_hat, the LOGGED speed at NOW) on every window; O reads e_level(v_hat.detach() / 10) (zero-initialised, added to O's ego conditioning token) at every tick, train and inference alike. "
                         "The head reads vision only (labels may use ego, inference may not). Needs --w-level > 0")
    ap.add_argument("--w-level", dest="w_level", type=float, default=0.0,
                    help="PREREG_V8F_LEVEL: the weight of the level term in the total. 0.0 (default) with --level-head off. Must be > 0 with --level-head on (a head with no loss is refused) and exactly 0.0 with it off (a weight with no head is refused). "
                         "The value is fixed by the arm's PREREG (the level loss is ~10 % of the total at step 0 on the smoke), written into the argv and never tuned")
    # ---- PREREG_LEVELHEAD_A2 (arm LLR). All three default to the earlier behaviour and have their G-DVB entry in FLAG_CHECKS and their clause in bad_lead.
    ap.add_argument("--lead-head", dest="lead_head", choices=M.LEAD_HEAD_MODES, default="off",
                    help="PREREG_LEVELHEAD_A2: off (default) = nothing is built = every earlier arm, bit for bit (state dict AND outputs). on = a head on the mean of the CURRENT tick frame's adapted VISION tokens -> MLP(d, d, 2) -> (a presence logit "
                         "for 'a lead within 40 m', gap_hat = sigmoid = gap / 40 in [0, 1]), trained with --w-lead x (BCE(presence) on the frames with a lead label + Huber(gap_hat, min(gap, 40) / 40) on the frames with a lead within 40 m) "
                         "from the lead labels of --lead-train; O reads [sigmoid(logit), gap_hat] DETACHED at every tick, train and inference alike (how: --aux-cond). The head reads vision only. Needs --w-lead > 0")
    ap.add_argument("--w-lead", dest="w_lead", type=float, default=0.0,
                    help="PREREG_LEVELHEAD_A2: the weight of the lead term in the total. 0.0 (default) with --lead-head off. Must be > 0 with --lead-head on (a head with no loss is refused) and exactly 0.0 with it off (a weight with no head is refused). "
                         "The value is fixed by the arm's PREREG (the lead term is ~10 % of the total at step 0: selfcheck.json lead_head_probe.w_lead_for_10pct_share), written into the argv and never tuned")
    ap.add_argument("--aux-cond", dest="aux_cond", choices=M.AUX_COND_MODES, default="add",
                    help="PREREG_LEVELHEAD_A2: how O reads the heads' outputs. add (default) = e_level(v_hat.detach() / 10) and e_lead([sigmoid(logit), gap_hat].detach()) are zero-initialised and ADDED to O's ego conditioning token (the step-0 forward is the "
                         "init's). token = O gets ONE extra conditioning token e_aux(Linear(3, d), default init) of [v_hat / 10 (0 without the level head), sigmoid(logit), gap_hat] (detached), REPLACING both additive paths (the step-0 forward differs "
                         "from the init's by construction). token is refused unless --level-head or --lead-head is on")
    # ---- PREREG_V8F_HACC_MC section 7 (O-RECOVER): an ego-state perturbation of the kinematic inputs; targets and labels are never touched. Every flag defaults to OFF (the defaults are
    # data.EGO_PERTURB_DEFAULTS), and every flag has its G-DVB entry in FLAG_CHECKS and its clause in bad_ego_perturb.
    ap.add_argument("--ego-perturb-p", dest="ego_perturb_p", type=float, default=D.EGO_PERTURB_DEFAULTS["p"],
                    help="O-RECOVER: the share of ELIGIBLE training samples (logged v0 < --ego-perturb-vmax) whose ego state is replaced by a perturbed one before the loss. 0.0 (default) = off = every earlier arm, bit for bit "
                         "(no draw, no copy). PREREG_V8F_HACC_MC section 7: 0.3. Needs --kin-mode yaw")
    ap.add_argument("--ego-perturb-vscale", dest="ego_perturb_vscale", type=float, nargs=2, metavar=("LO", "HI"), default=list(D.EGO_PERTURB_DEFAULTS["vscale"]),
                    help="the speed multiplier s ~ U(LO, HI) (0 < LO <= HI <= 1), one draw per sample, applied to v and to the yaw rate of every real history frame (the same path at the lower speed)")
    ap.add_argument("--ego-perturb-vmax", dest="ego_perturb_vmax", type=float, default=D.EGO_PERTURB_DEFAULTS["vmax"],
                    help="eligibility: only windows whose LOGGED v0 is below this (m/s) can be perturbed")
    ap.add_argument("--ego-perturb-adrop", dest="ego_perturb_adrop", type=float, default=D.EGO_PERTURB_DEFAULTS["adrop"],
                    help="the acceleration offset d ~ U(0, ADROP) m/s^2, subtracted from a on every real history frame")
    ap.add_argument("--ego-perturb-yawrate-std", dest="ego_perturb_yawrate_std", type=float, default=D.EGO_PERTURB_DEFAULTS["yawrate_std_deg"],
                    help="the LATERAL lever: an offset n ~ N(0, STD) deg/s added to the measured yaw rate (and, through it, to kappa, the heading history and the odometry) of the whole window. 0.0 (default) = lateral off")
    ap.add_argument("--ego-perturb-seed", dest="ego_perturb_seed", type=int, default=D.EGO_PERTURB_DEFAULTS["seed"],
                    help="the perturbation's own seed: the draws of step k are a function of (seed, k) only, on a private generator (a resumed run draws what an uninterrupted run drew)")
    # ---- PREREG_V8F_DAGGER: on-policy own states, supervised with the human's labels at the matched route point. Every flag defaults to OFF (the defaults are DG.DAGGER_DEFAULTS), and every flag has its
    # G-DVB entry in FLAG_CHECKS and its clause in bad_dagger. The arm's argv must NAME --dagger-p (DAG-0 names only `--dagger-p 0`: a bank or seed declared while p is 0 is refused).
    ap.add_argument("--dagger-p", dest="dagger_p", type=float, default=DG.DAGGER_DEFAULTS["p"],
                    help="PREREG_V8F_DAGGER: the share of ELIGIBLE training samples (the drawn window has >= 1 own state with r' = t that passed every cut) whose ego tensors are replaced by one of the model's OWN T1-kin "
                         "states, labels unchanged. 0.0 (default) = off = every earlier arm, bit for bit (no bank read, no draw, no copy). The arm's value is fixed by PREREG section 4 / Amendment 1 (0.1), never a default here. Needs --kin-mode yaw")
    ap.add_argument("--dagger-bank", dest="dagger_bank", default="",
                    help="the own-state bank: the merged directory (or its bank.json) written by v8f_dagger_bank.py merge. Loaded, hashed and checked at launch (K-B2 on-policy against --init-ckpt, K-B3 TRAIN-only / window-list "
                         "sha256 / disjoint from eval-141, COVERAGE >= 0.5); its sha256 (the manifest's) is recorded as levers.dagger_bank_sha256 and must equal --dagger-bank-sha256. Default \"\" = none")
    ap.add_argument("--dagger-bank-sha256", dest="dagger_bank_sha256", default="",
                    help="PREREG_V8F_DAGGER section 4: the sha256 (64 hex) of the bank's bank.json that THIS arm declares it trains on. REQUIRED when --dagger-p > 0 (DAG names all four flags) and REFUSED when --dagger-p is 0 "
                         "(a lever that is declared and does nothing). The bank is loaded against it (a bank with another manifest sha256 is refused at load) and the launch gate re-hashes bank.json from disk and refuses if it "
                         "differs. Default \"\" = none")
    ap.add_argument("--dagger-seed", dest="dagger_seed", type=int, default=DG.DAGGER_DEFAULTS["seed"],
                    help="the swap's own seed: the draws of step k are a function of (seed, k) only, on a private generator (a resumed run draws what an uninterrupted run drew)")
    ap.add_argument("--ego-perturb-with-dagger", dest="ego_perturb_with_dagger", action="store_true", default=False,
                    help="AMEND1 section 1: lift the exclusivity of --dagger-p and --ego-perturb-p for THIS arm and compose them. The DAgger swap is drawn first; the ego perturbation is then drawn (its own p, its own seeded "
                         "generator) ONLY on the samples the swap did not replace, so no sample is both swapped and perturbed (the run record carries ego_dagger_overlap_n, exactly 0). REFUSED unless --dagger-p > 0 AND "
                         "--ego-perturb-p > 0 (a lever that is declared and does nothing). Default off = every earlier argv, bit for bit (K-BUILD)")
    # ---- PREREG_RECOVER_FT (H-RECOVER-FT-1): the recovery perturbation. Every flag defaults to OFF (the defaults are R.RECOVER_DEFAULTS), and every flag has its G-DVB entry in FLAG_CHECKS and its clause in bad_recover.
    # The arm's argv must NAME --recover-p (C0 names only `--recover-p 0`: any other recover flag declared while p is 0 is refused). Exclusive with --ego-perturb-p > 0 and --dagger-p > 0.
    ap.add_argument("--recover-p", dest="recover_p", type=float, default=R.RECOVER_DEFAULTS["p"],
                    help="PREREG_RECOVER_FT: the share of ELIGIBLE training samples (NOW is a real kinematic row) whose ego state is offset by ONE (dv, da, dom, dy, dpsi) deviation drawn from the law, with the GT future "
                         "re-expressed in the deviated frame (and blended, --recover-target). 0.0 (default) = off = every earlier arm, bit for bit (no bank read, no draw, no copy). The registered arm value is 0.3. Needs --kin-mode yaw. "
                         "Refused together with --ego-perturb-p > 0 or --dagger-p > 0")
    ap.add_argument("--recover-dev-bank", dest="recover_dev_bank", default="",
                    help="the TRAIN own-state bank the EMPIRICAL deviation table is derived from (the merged directory or its bank.json). Loaded, re-hashed and checked at launch (K-B3: TRAIN-only, window-list sha256, disjoint from "
                         "eval-141); K-B2 is NOT read (the bank's generating checkpoint is not this run's init, by the card). REQUIRED with --recover-p > 0 and the empirical law; refused at p = 0. Default \"\" = none")
    ap.add_argument("--recover-dev-bank-sha256", dest="recover_dev_bank_sha256", default="",
                    help="the sha256 (64 hex) of the bank's bank.json that THIS arm declares it draws deviations from. REQUIRED with --recover-p > 0 and the empirical law; refused at p = 0 (a lever that is declared and does "
                         "nothing). The bank is loaded against it, and the launch gate re-hashes bank.json from disk. Default \"\" = none")
    ap.add_argument("--recover-dev-law", dest="recover_dev_law", choices=list(R.DEV_LAWS), default=R.RECOVER_DEFAULTS["dev_law"],
                    help="`empirical` (registered): one record per selected sample, uniform over the bank-derived table. `apriori`: the card's fixed law (dv U[-3, 0], da U[-1, 0.5], dy N(0, 0.5 m), dpsi N(0, 3 deg), "
                         "yaw rate N(0, 2 deg/s)); no bank is read")
    ap.add_argument("--recover-dev-shape", dest="recover_dev_shape", choices=list(R.DEV_SHAPES), default=R.RECOVER_DEFAULTS["dev_shape"],
                    help="`constant` (default, every earlier arm bit for bit): the drawn record's (dv, da, dom) offsets all history frames alike. `pertick`: history frame k is offset by the record's own per-tick deviation "
                         "(dv_t[k], da_t[k], dom_t[k]) and the pose / nav history moves by the cumulative effect (needs the empirical law; the table is built with per-tick columns; refused with the a-priori law and at --recover-p 0)")
    # ---- PREREG_T2_DAGGER (DT2): whole samples from the T2 own-state bank. Default OFF; the registered arm value is 0.3; exclusive with every other own-state lever.
    ap.add_argument("--t2-bank-p", dest="t2_bank_p", type=float, default=TB.T2_DEFAULTS["p"],
                    help="PREREG_T2_DAGGER: the share of training samples REPLACED by a whole sample of the T2 bank (tokens rendered at the executed poses, executed ego history, RECOVER-FT's target law); the ONLY supervision on a bank row "
                         "is that target (every other loss term is masked). 0.0 (default) = off = the pre-change run, bit for bit. Registered arm value 0.3. Exclusive with --recover-p / --dagger-p / --ego-perturb-p / --encoder lora.")
    ap.add_argument("--t2-bank", dest="t2_bank", default="", help="the bank directory (bank.json + rows.pt). REQUIRED with --t2-bank-p > 0; refused at p = 0")
    ap.add_argument("--t2-bank-sha256", dest="t2_bank_sha256", default="", help="the sha256 (64 hex) of the bank's bank.json that THIS arm declares it draws from. REQUIRED with --t2-bank-p > 0; refused at p = 0")
    ap.add_argument("--t2-bank-seed", dest="t2_bank_seed", type=int, default=TB.T2_DEFAULTS["seed"], help="seed of the swap's private generator (domain-separated hash of (seed, step))")
    ap.add_argument("--recover-target", dest="recover_target", choices=list(R.TARGETS), default=R.RECOVER_DEFAULTS["target"],
                    help="`rejoin` (registered, AMENDMENT_1): the logged future re-expressed in the deviated frame, the first --recover-rejoin-steps knots blended from the logged future onto it. `logged`: the pure frame change")
    ap.add_argument("--recover-rejoin-steps", dest="recover_rejoin_steps", type=int, default=R.RECOVER_DEFAULTS["rejoin_steps"],
                    help="K, the number of knots the rejoin blend spans (1..30). Registered: 10")
    ap.add_argument("--recover-seed", dest="recover_seed", type=int, default=R.RECOVER_DEFAULTS["seed"],
                    help="the perturbation's own seed: the draws of step k are a function of (seed, k) only, on a private generator that never touches the global RNG (a resumed run draws what an uninterrupted run drew)")
    ap.add_argument("--init-ckpt", dest="init_ckpt", default="",
                    help="O-RECOVER's start state: when set and --out holds no ckpt_last.pt, load the MODEL weights of this checkpoint (a v8f_mini_train ckpt, e.g. M2b's) STRICTLY into the freshly built model (its architecture "
                         "config must match the built one, else the launch is REFUSED), at step 0 with a FRESH optimiser and schedule. An existing ckpt_last.pt in --out still resumes as ever (resume wins). The path and the file's "
                         "sha256 are recorded in the levers. Default \"\" = off = every earlier arm, bit for bit")
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
    # ---- PREREG_V8F_MRATE: the multi-rate world model and the WM-only mode. Both default to the earlier behaviour; both have their G-DVB entry in FLAG_CHECKS and their clause in bad_wm_only.
    ap.add_argument("--wm-k", dest="wm_k", type=int, default=1,
                    help="PREREG_V8F_MRATE: how many past states each world-model level reads, spaced at that level's own step. 1 (default) = W0 = every earlier arm, bit for bit (no module and no state-dict key is added). "
                         "K > 1 (W1: 3): L1 reads the adapter tokens of frames t, t-1, .., t-K+1; L2 / L3 read K states of the level below, 0.4 s / 0.8 s apart. Needs --wm b")
    ap.add_argument("--wm-pool", dest="wm_pool", choices=("mean", "att25"), default="mean",
                    help="PREREG_V8F_WMPOOL: how the world model's z1 reads the adapter tokens. mean (default) = the token mean, every earlier arm bit for bit (no module, no state-dict key). att25 = P-ATT: the 4 x 10 grid pooled to 2 x 5 cells (each the mean of a 2 x 2 block), "
                         "ONE zero-initialised learned query attends over the 10 cells (uniform weights at build = the token mean), then proj. Needs --wm-only --wm-windows wmonly and --wm-zcache = a CELL cache (zcells.py; CELL_MANIFEST.json with this adapter's sha); refused otherwise")
    ap.add_argument("--wm-only", dest="wm_only", action="store_true", default=False,
                    help="PREREG_V8F_MRATE: WM-only training. Needs --init-ckpt (a plain M2b s{k}): its adapter, readers and planner load strictly and bit-equal and are FROZEN; its world-model parameters are NOT loaded -- the WM is "
                         "re-initialised under --seed (hmax is not compared: M2b trained at 16, this arm at 28); only the WM loss terms are trained. Every other lever must be at its default (a lever that is declared and does nothing is refused). Default off")
    ap.add_argument("--wm-strides", dest="wm_strides", type=int, nargs=2, default=None, metavar=("S2", "S3"),
                    help="PREREG_V8F_LFS: the world model's level steps, in states of the level below: S2 = L1 states per L2 step, S3 = L2 states per L3 step. Absent (default) = the arm config's (2, 2) = every earlier arm, bit for bit "
                         "(0.2 / 0.4 / 0.8 s on the cache grid). (2, 3) = steps 0.2 / 0.4 / 1.2 s; (3, 3) = 0.2 / 0.6 / 1.8 s. Needs --wm a|b. A flag whose value leaves the strides unchanged must still be NAMED in an LFS argv "
                         "(its selfcheck then records the BUILT strides). Every level must keep >= 2 training transitions at --hmax, else the gate refuses (NOT RUN, with the count)")
    # ---- PREREG_V8F_LADDER: the world-model ladder levers. Every flag defaults to the earlier behaviour and has its G-DVB entry in FLAG_CHECKS and its clause in bad_wm_ladder.
    ap.add_argument("--wm-rollout-k", dest="wm_rollout_k", type=int, nargs=3, default=None, metavar=("K1", "K2", "K3"),
                    help="PREREG_V8F_LADDER: free-run steps of the k-step loss per WM level (through HJEPAStack.rollout, on top of the teacher-forced one-step loss). Absent = (1, 1, 1) = every earlier arm, bit for bit. "
                         "A level with k > 1 needs hmax >= k x (its cumulative stride) + 1 frames. Needs --wm a|b")
    ap.add_argument("--wm-rollout-stride", dest="wm_rollout_stride", type=int, nargs=3, default=None, metavar=("A1", "A2", "A3"),
                    help="PREREG_V8F_LADDER: the free run of level l starts at every A_l-th state of the window. Absent = (1, 1, 1). Used only where that level's k > 1 (a stride > 1 declared on a k = 1 level is refused)")
    ap.add_argument("--wm-rollout-w", dest="wm_rollout_w", type=float, default=1.0,
                    help="PREREG_V8F_LADDER: weight of every level's k-step term in its level loss (constant over training, no schedule). Default 1.0. Only meaningful where some level's k > 1")
    ap.add_argument("--wm-ctx", dest="wm_ctx", type=int, nargs=3, default=None, metavar=("C1", "C2", "C3"),
                    help="PREREG_V8F_LADDER: each WM level's predictor context (states). Absent = (3, 3, 3) = every earlier arm. Changes the predictor's parameter shape (the WM is re-initialised in a --wm-only run)")
    ap.add_argument("--wm-windows", dest="wm_windows", choices=("train", "wmonly"), default="train",
                    help="PREREG_V8F_LADDER: the window list. train (default) = the labelled 6 s-complete windows of every earlier arm. wmonly = the world-model-only sampler: every full hmax-frame window of a clip (t >= hmax - 1), "
                         "no label requirement. Needs --wm-only")
    ap.add_argument("--wm-win-stride", dest="wm_win_stride", type=int, default=1,
                    help="PREREG_V8F_LADDER: anchors of the wmonly sampler are every n-th t (default 1 = all). Used only with --wm-windows wmonly")
    ap.add_argument("--wm-pooled", dest="wm_pooled", choices=("off", "on"), default="off",
                    help="PREREG_V8F_LADDER: the frozen-adapter pooled path (wmonly windows only): the loader hands the token mean of the layer-normalised features and the model applies the frozen adapter after the mean "
                         "(equal to the live adapter mean to 1e-6), so a 91-frame window never builds a [B, F, n_tok, d] tensor. Default off = the live adapter")
    ap.add_argument("--wm-zcache", dest="wm_zcache", default="",
                    help="PREREG_V8F_LADDER: a directory of per-clip <clip>.npz (zmean [T, D] = the frozen adapter's mean token before the WM projection, ok [T], t_s [T]) built from THIS run's init checkpoint. wmonly windows only; every clip "
                         "of the run must have a file (else refused); a window with an ok=False frame is an error. Default empty = the pooled/live path. The adapter is then not read at all")
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
    ap.add_argument("--resume-perf-levers", dest="resume_perf_levers", default=None, metavar="PREV_ARGV_JSON",
                    help="resume a checkpoint written by the argv in this JSON ({'argv': [...]}) although --accum and/or --enc-ckpt differ from it. Nothing else may differ: the two argvs, with "
                         "'--accum <n>', '--enc-ckpt' and this flag removed, must be equal token for token. Writes one perf_resume record into train_log.jsonl. Default: off (a different argv refuses)")
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


WM_STRIDES_DEFAULT = tuple(M.V8FMiniConfig.__dataclass_fields__["wm_strides"].default)       # PREREG_V8F_LFS: the arm config's strides, (2, 2): what an argv that does not name --wm-strides builds


def declared_wm_strides(a) -> tuple:
    """PREREG_V8F_LFS: the strides the argv DECLARES: ``--wm-strides S2 S3``, or the arm config's (2, 2) when the flag is absent."""
    return WM_STRIDES_DEFAULT if a.wm_strides is None else tuple(int(s) for s in a.wm_strides)


WM_LADDER_DEFAULTS = {"wm_rollout_k": (1, 1, 1), "wm_rollout_stride": (1, 1, 1), "wm_ctx": (3, 3, 3)}      # PREREG_V8F_LADDER: what an argv that does not name the flag builds (= the LevelConfig defaults)
WM_LADDER_FLAGS = ("wm_rollout_k", "wm_rollout_stride", "wm_rollout_w", "wm_ctx", "wm_windows", "wm_win_stride", "wm_pooled", "wm_zcache")


def declared_wm_ladder(a) -> dict:
    """PREREG_V8F_LADDER: the ladder levers the argv DECLARES (the flag's value, or the earlier default when the flag is absent)."""
    d = {k: (WM_LADDER_DEFAULTS[k] if getattr(a, k) is None else tuple(int(x) for x in getattr(a, k))) for k in WM_LADDER_DEFAULTS}
    d["wm_rollout_w"] = float(a.wm_rollout_w)
    return d


def wm_ladder_named(a) -> list:
    """The ladder flags the argv moved off their parser defaults."""
    dfl = build_parser()
    return [k for k in WM_LADDER_FLAGS if getattr(a, k) != dfl.get_default(k)]


def make_config(a) -> M.V8FMiniConfig:
    lv = declared_levers(a)
    hmax = a.hmax or (16 if lv["wm"] != "off" else 10)
    return M.V8FMiniConfig(d=a.d, ff=2 * a.d, hist=a.hist, hmax=hmax, window_start=a.window_start, n_q=a.n_q, nb=a.nb, navonly=lv["navonly"],
                           wm_strides=declared_wm_strides(a),                                             # PREREG_V8F_LFS: the ONE line that applies --wm-strides (a mutation deletes it)
                           consistency=lv["consistency"], wm=lv["wm"], n_tok=40 if a.grid == "pool4x10" else 160,
                           p_gt_start=a.p_gt_start, p_gt_end=a.p_gt_end, w_ckpt=a.w_ckpt, w_tac=a.w_tac,
                           w_cons=a.w_cons, w_traj=a.w_traj, w_pen=a.w_pen, w_pen_self=a.w_pen_self, w_wm=a.w_wm, wm_k=a.wm_k,
                           wm_rollout_k=declared_wm_ladder(a)["wm_rollout_k"], wm_rollout_stride=declared_wm_ladder(a)["wm_rollout_stride"],       # PREREG_V8F_LADDER: the lines that apply the ladder levers (a mutation deletes them)
                           wm_rollout_w=declared_wm_ladder(a)["wm_rollout_w"], wm_ctx=declared_wm_ladder(a)["wm_ctx"],
                           wm_pool=a.wm_pool,                                                             # PREREG_V8F_WMPOOL: the ONE line that applies --wm-pool (a mutation deletes it)
                           detach_upper=a.detach_upper, kin_mode=a.kin_mode, cons_absent=lv["cons_absent"], code_mix=a.code_mix, o_interface=a.o_interface, w_acc=a.w_acc, launch_intent_w=a.launch_intent_w,
                           level_head=a.level_head, w_level=a.w_level,
                           lead_head=a.lead_head, w_lead=a.w_lead, aux_cond=a.aux_cond)


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


def make_ego_perturb(a):
    """argv -> the perturber the training loop calls ``.apply`` on (PREREG_V8F_HACC_MC section 7). The ONE place the six flags become an object; ``bad_ego_perturb`` compares it back to the argv."""
    return D.EgoPerturb(p=a.ego_perturb_p, vscale=a.ego_perturb_vscale, vmax=a.ego_perturb_vmax, adrop=a.ego_perturb_adrop,
                        yawrate_std_deg=a.ego_perturb_yawrate_std, seed=a.ego_perturb_seed, kin_mode=a.kin_mode)


def bad_ego_perturb(a, ego) -> list:
    """The O-RECOVER half of the launch gate (G-DVB). Empty = pass. Every ego flag is read back from the BUILT perturber (``ego``); with no perturber built every one must be at its default
    (a flag that builds nothing is refused, not ignored); the built law must be a legal one; and a lever declared without an effect is refused."""
    dfl = {k: build_parser().get_default(k) for k in ("ego_perturb_p", "ego_perturb_vscale", "ego_perturb_vmax", "ego_perturb_adrop", "ego_perturb_yawrate_std", "ego_perturb_seed")}
    declared = {k: getattr(a, k) for k in dfl}
    if ego is None:
        moved = sorted(k for k in dfl if declared[k] != dfl[k])
        return [f"{moved} are set but no ego perturber was built: a flag that builds nothing is refused, not ignored"] if moved else []
    bad = []
    lv = ego.built_levers()
    for k, want in declared.items():
        got = lv.get(k)
        want = list(want) if isinstance(want, (list, tuple)) else want
        if got != want:
            bad.append(f"lever {k}: declared {want!r} but built {got!r}")
    if lv.get("kin_mode") != a.kin_mode:
        bad.append(f"ego perturber kin_mode: declared {a.kin_mode} but built {lv.get('kin_mode')}")
    bad += [f"ego perturber: {m}" for m in ego.problems()]
    return bad


def is_sha256_hex(s) -> bool:
    """A full sha256 as hex: exactly 64 hex digits (either case). A truncated hash copied from a log line is NOT an identity, so it is refused rather than prefix-matched."""
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdefABCDEF" for c in s)


def dagger_bank_manifest(path) -> str:
    """The file whose sha256 IS the bank's identity (``DaggerBank.load``): a merged bank's ``bank.json`` (a directory is resolved to the one inside it); a bare shard names itself."""
    return os.path.join(path, "bank.json") if os.path.isdir(path) else path


def dagger_eval_clips(a, header) -> tuple:
    """K-B3's eval side: the eval-141 clips the own-state bank must be disjoint from -> (sorted clip list, {source: n}). The union of every source that exists, so a bank cannot hide an eval clip behind
    ONE incomplete list: (1) the canonical grid ``eval_grid_windows.json`` (the repo's, or the synthetic world's own for ``--synthetic``) -- REQUIRED, without it the check cannot run and the launch is refused;
    (2) the eval list the bank's own header recorded at generation; (3) the cache's ``eval`` manifest when the cache has one."""
    grid = os.path.join(a.out, "_synthetic", "eval_grid_windows.json") if a.synthetic else DAGGER_EVAL_GRID
    if not os.path.exists(grid):
        raise DG.BankRefused(f"K-B3: the canonical eval-141 grid {grid} does not exist: the eval-disjointness check cannot run, and a check that cannot run is a refusal")
    src = {"grid": [str(c) for c in json.load(open(grid, encoding="utf-8"))["order"]]}
    hdr = [str(c) for c in (header.get("eval") or {}).get("clips") or []]
    if hdr:
        src["bank_header"] = hdr
    root = os.path.join(a.out, "_synthetic") if a.synthetic else a.cache_dir
    if os.path.exists(os.path.join(root, "eval", "manifest.jsonl")):
        src["cache_eval_manifest"] = [c for c, _ in D.read_manifest(root, "eval")]
    if not src["grid"]:
        raise DG.BankRefused(f"K-B3: the eval-141 grid {grid} lists no clip: the eval-disjointness check would pass vacuously")
    return sorted(set().union(*map(set, src.values()))), {k: len(v) for k, v in src.items()}


def load_dagger_bank(a, J, wins, clips, info) -> tuple:
    """``--dagger-bank`` -> (the bank re-indexed to this trainer's join, the controls record). The PREREG's controls, each refusing with its own message (``DG.BankRefused``):

      load      every shard re-hashed against the manifest (``DaggerBank.load``); a bare shard / unmerged bank is refused
      K-B3 (1)  the bank is not PARTIAL and every bank clip is in the arm's TRAIN clip list (``clips``: after --exclude / --data-frac / --max-clips) -- a clip the join lacks is refused again at the re-index
      K-B3 (2)  the TRAIN window-list sha256 the bank recorded == this trainer's ``windows_sha256(J["clip"], wins)``
      K-B3 (3)  bank clips INTERSECT eval-141 = empty (clip id AND stable sid)   -- read FIRST, so an injected eval clip is named as an eval clip
      K-B2      the bank's generating checkpoint sha256 == ``--init-ckpt``'s (the file the model was started from)
      COVERAGE  the eligible fraction of THIS trainer's TRAIN windows >= 0.5, else BLOCKED: the arm does not launch
    """
    bank0 = DG.DaggerBank.load(a.dagger_bank, expect_sha256=a.dagger_bank_sha256 or None)         # the declared --dagger-bank-sha256: a bank whose manifest hashes otherwise is refused here, before any control is read
    if bank0.manifest is None:
        raise DG.BankRefused(f"{a.dagger_bank}: a bare shard, not a merged bank (no bank.json): the bank's identity is the manifest's sha256")
    eval_clips, eval_src = dagger_eval_clips(a, bank0.header)
    wsha = DG.windows_sha256(J["clip"], wins)
    kb3 = bank0.check_train_only(train_clips=clips, eval_clips=eval_clips, train_window_list_sha256=wsha)
    init = info.get("init_ckpt") or {}
    if not init.get("sha256"):
        raise DG.BankRefused("K-B2: --dagger-bank needs --init-ckpt (the bank's generating checkpoint is compared with the file this run starts from)")
    kb2 = bank0.check_on_policy(init["sha256"])
    bank = DG.DaggerBank(bank0.a, bank0.header, clip_names=J["clip"], manifest=bank0.manifest, sha256=bank0.sha256, path=bank0.path)
    cov = bank.coverage(wins)
    if cov["blocked"]:
        raise DG.BankRefused(f"COVERAGE {cov['fraction']:.4f} < {cov['min']} ({cov['n_eligible']} of {cov['n_windows']} TRAIN windows have an eligible own state): the arm does not launch (outcome BLOCKED, PREREG_V8F_DAGGER section 7)")
    ctl = {"bank_path": bank.path, "bank_sha256": bank.sha256, "n_records": bank.n, "reason_counts": bank.reason_counts(), "coverage": cov,
           "kb2": {**kb2, "init_ckpt_sha256": init["sha256"]}, "kb3": {**kb3, "window_list_sha256": wsha, "eval_sources": eval_src, "n_eval_clips_checked": len(eval_clips)}}
    return bank, ctl


def make_dagger(a, J, wins, clips, info):
    """argv -> the swap the training loop calls ``.apply`` on (PREREG_V8F_DAGGER). The ONE place the three flags become an object; ``bad_dagger`` compares it back to the argv. With ``--dagger-p 0`` no bank is
    read (a ``--dagger-bank`` declared at p = 0 builds nothing and is refused by the gate: a lever that is declared and does nothing). A refused bank stops the launch before the gate."""
    bank = None
    if a.dagger_bank and a.dagger_p > 0:
        try:
            bank, ctl = load_dagger_bank(a, J, wins, clips, info)
        except DG.BankRefused as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ dagger-bank {a.dagger_bank}: {e}")
        except Exception as e:                                       # noqa: BLE001  a missing / foreign / corrupt bank is a REFUSAL, not a crash
            raise SystemExit(f"ZZDVB-REFUSEDZZ dagger-bank {a.dagger_bank}: {type(e).__name__}: {str(e)[:600]}")
        info["dagger"] = ctl                                         # selfcheck.json carries the controls that were read
    return DG.DaggerSwap(bank, p=a.dagger_p, seed=a.dagger_seed, kin_mode=a.kin_mode)


def bad_dagger(a, dag, cfg=None, data_info=None) -> list:
    """The DAgger half of the launch gate (G-DVB). Empty = pass. Every flag is read back from the BUILT swap (``dag``); with no swap built every one must be at its default (a flag that builds nothing is
    refused, not ignored). Beyond the flags: the bank's path and sha256 against the declaration (the file on disk is re-hashed), K-B2 / K-B3 / COVERAGE were read (the record exists and says so), the bank's
    generating configuration equals PREREG section 1's and this model's, the exclusivity with --ego-perturb-p, and a lever declared without an effect (DAG-only flags at p = 0) is refused."""
    dfl = {k: build_parser().get_default(k) for k in ("dagger_p", "dagger_bank", "dagger_bank_sha256", "dagger_seed")}
    declared = {k: getattr(a, k) for k in dfl}
    if dag is None:
        moved = sorted(k for k in dfl if declared[k] != dfl[k])
        return [f"{moved} are set but no DAgger swap was built: a flag that builds nothing is refused, not ignored"] if moved else []
    bad = []
    lv = dag.built_levers()
    for k in ("dagger_p", "dagger_seed"):
        if lv.get(k) != declared[k]:
            bad.append(f"lever {k}: declared {declared[k]!r} but built {lv.get(k)!r}")
    if lv.get("kin_mode") != a.kin_mode:
        bad.append(f"DAgger swap kin_mode: declared {a.kin_mode} but built {lv.get('kin_mode')}")
    bad += [f"DAgger swap: {m}" for m in dag.problems()]
    if a.dagger_p > 0 and a.ego_perturb_p > 0 and not a.ego_perturb_with_dagger:
        bad.append(f"--dagger-p {a.dagger_p} together with --ego-perturb-p {a.ego_perturb_p}: the two levers are never combined in one arm (PREREG_V8F_DAGGER section 4; AMEND1 section 1 lifts this only for an argv that names --ego-perturb-with-dagger)")
    if a.dagger_bank_sha256 and not is_sha256_hex(a.dagger_bank_sha256):
        bad.append(f"--dagger-bank-sha256 {a.dagger_bank_sha256!r} is not a sha256 (64 hex digits): a truncated hash identifies nothing")
    if a.dagger_p > 0 and not a.dagger_bank_sha256:
        bad.append(f"--dagger-bank-sha256 is required when --dagger-p is {a.dagger_p} (DAG names all four flags, PREREG_V8F_DAGGER section 4): the arm must declare which bank it trains on")
    if a.dagger_p <= 0 and a.dagger_bank_sha256:
        bad.append(f"--dagger-bank-sha256 is declared while --dagger-p is {a.dagger_p}: a lever that is declared and does nothing (DAG-0 names only --dagger-p 0)")
    if not a.dagger_bank:
        if lv.get("dagger_bank") is not None:
            bad.append(f"lever dagger_bank: no --dagger-bank declared but the swap holds the bank {lv.get('dagger_bank')!r}")
        if a.dagger_bank_sha256 and a.dagger_p > 0:
            bad.append(f"--dagger-bank-sha256 {a.dagger_bank_sha256[:12]} is declared but no --dagger-bank names a bank to hash")
        return bad
    if dag.bank is None:
        bad.append(f"lever dagger_bank: declared {a.dagger_bank!r} but no bank was built (--dagger-p {a.dagger_p}: a declared bank at p = 0 builds nothing)")
        return bad
    want = os.path.normcase(os.path.abspath(dagger_bank_manifest(a.dagger_bank)))
    got = os.path.normcase(os.path.abspath(str(lv.get("dagger_bank"))))
    if want != got:
        bad.append(f"lever dagger_bank: declared {want!r} but built {got!r}")
    disk = DG.sha256_file(dagger_bank_manifest(a.dagger_bank)) if os.path.exists(dagger_bank_manifest(a.dagger_bank)) else None
    if lv.get("dagger_bank_sha256") != disk:
        bad.append(f"lever dagger_bank_sha256: the file {a.dagger_bank!r} hashes to {disk!r} but the swap was built from {lv.get('dagger_bank_sha256')!r}")
    if a.dagger_bank_sha256 and a.dagger_bank_sha256.lower() != disk:
        bad.append(f"lever dagger_bank_sha256: --dagger-bank-sha256 declares {a.dagger_bank_sha256.lower()!r} but the bank manifest {dagger_bank_manifest(a.dagger_bank)!r} hashes to {disk!r}")
    hdr = dag.bank.header
    if hdr.get("schema") != DG.SCHEMA:
        bad.append(f"bank schema {hdr.get('schema')!r} != {DG.SCHEMA!r}")
    if hdr.get("rollout") != DG.ROLLOUT:
        bad.append(f"bank rollout configuration {hdr.get('rollout')!r} != PREREG section 1's {DG.ROLLOUT!r}")
    if hdr.get("constants_sha256") != DG.constants_sha256():
        bad.append(f"bank matching constants sha256 {str(hdr.get('constants_sha256'))[:12]} != this tree's {DG.constants_sha256()[:12]}")
    if cfg is not None:
        for k in ("hist", "hmax", "max_age_t", "max_age_s", "window_start", "kin_mode"):
            if (hdr.get("cfg") or {}).get(k) != getattr(cfg, k):
                bad.append(f"bank was generated with cfg.{k} = {(hdr.get('cfg') or {}).get(k)!r}, this model has {getattr(cfg, k)!r}")
        if lv.get("bank_hmax") != cfg.hmax:
            bad.append(f"bank history length {lv.get('bank_hmax')} != the model's hmax {cfg.hmax}")
    ctl = (data_info or {}).get("dagger")
    if not ctl:
        bad.append("a bank is built but no K-B2 / K-B3 / COVERAGE record was read (the controls did not run)")
    else:
        if ctl.get("bank_sha256") != lv.get("dagger_bank_sha256"):
            bad.append(f"the controls were read on bank {str(ctl.get('bank_sha256'))[:12]} but the swap holds {str(lv.get('dagger_bank_sha256'))[:12]}")
        if (ctl.get("kb2") or {}).get("ckpt_sha256") != hdr.get("ckpt_sha256"):
            bad.append("K-B2 was not read against this bank's generating checkpoint")
        init_sha = ((data_info or {}).get("init_ckpt") or {}).get("sha256")
        if init_sha != hdr.get("ckpt_sha256"):
            bad.append(f"K-B2: the bank was generated by checkpoint {str(hdr.get('ckpt_sha256'))[:12]} but this run started from {str(init_sha)[:12]} (--init-ckpt): the states are not on-policy for this init")
        if (ctl.get("kb3") or {}).get("window_list_sha256") != hdr.get("window_list_sha256") or (ctl.get("kb3") or {}).get("overlap_with_eval") != 0:
            bad.append("K-B3 was not read against this bank's window list / with an empty eval overlap")
        if not isinstance(ctl.get("coverage"), dict) or ctl["coverage"].get("blocked") is not False:
            bad.append(f"COVERAGE is not recorded as passed: {ctl.get('coverage')!r}")
    return bad


class EgoDaggerCompose:
    """AMEND1 section 1 (``--ego-perturb-with-dagger``): the training loop's ego-input stage when BOTH ego levers are on in one arm. ONE object holds the two BUILT levers the loop applies (``ego`` = O-RECOVER's
    perturbation, ``dag`` = the DAgger swap) and the order they run in:

      1. the swap is drawn first (``dag.apply``); its selection mask is ``dag.last["sel"]``;
      2. the perturbation is then drawn, at its own p and from its own seeded generator, ONLY on the samples the swap did not replace (``ego.apply(.., exclude=<the swap's mask>)``);
      3. no sample is both swapped and perturbed: ``overlap`` is MEASURED from the two selection masks after both applies (not inferred from the exclusion), and a nonzero count raises.

    INDEPENDENT STREAMS (Amendment 1 section 1: "at its own p and from its own seeded generator"). The swap and the perturbation seed their generators from ONE formula, ``(seed * 1_000_003 + step) % 2**62``, and the
    arm's registered seeds are equal, so unseparated they were one generator: the selection uniform of the swap was the perturbation's too, and P(perturbed | not swapped) was (p_ego - p_swap) / (1 - p_swap) = 0.222,
    not 0.3 (box 2, 2026-10-28: 19161 / 76898 = 0.2492). When ``on`` is True this stage therefore sets ``ego.stream = D.EGO_PERTURB_COMPOSED_STREAM`` -- a domain-separated seed for the PERTURBATION only; the swap's
    generator, and every non-composed path, is untouched (bit-identical).

    ``on`` False (the flag is not named): the loop never calls this object; it runs its two original lines (``ego.apply`` then ``dag.apply``), so every existing argv is the pre-change run."""

    def __init__(self, ego, dag, on: bool = False):
        self.ego, self.dag, self.on = ego, dag, bool(on)
        if self.on:
            self.ego.stream = D.EGO_PERTURB_COMPOSED_STREAM         # the perturbation's generator is independent of the swap's, whatever the two seeds are (off: the perturber is not touched)
        self.overlap = 0

    def built_levers(self) -> dict:
        return {"ego_perturb_with_dagger": self.on, "order": ["dagger_swap", "ego_perturb"] if self.on else ["ego_perturb", "dagger_swap"], "ego_perturb_stream": self.ego.stream}

    def problems(self) -> list:
        """A composition declared without BOTH built levers on is a lever that is declared and does nothing."""
        bad = []
        if self.on and not self.ego.active:
            bad.append(f"--ego-perturb-with-dagger is declared but the BUILT perturbation is off (ego_perturb_p {self.ego.p}): a lever that is declared and does nothing")
        if self.on and not self.dag.active:
            bad.append(f"--ego-perturb-with-dagger is declared but the BUILT swap is off (dagger_p {self.dag.p}): a lever that is declared and does nothing")
        return bad

    def apply(self, batch: dict, step: int, *, ego_p=None, dag_p=None) -> dict:
        """The composed batch for training step ``step``. ``ego_p`` / ``dag_p`` override either probability for this call only (the launch gate's probes: 1.0 = every eligible sample, 0.0 = the control)."""
        swapped = self.dag.apply(batch, step, p=dag_p)                                  # the swap is drawn FIRST
        sw = self.dag.last.get("sel")
        if sw is None:                                                                  # the swap is off for this call (dag_p = 0): no sample was replaced
            sw = torch.zeros(int(batch["p10f"].shape[0]), dtype=torch.bool)
        out = self.ego.apply(swapped, step, p=ego_p, exclude=sw)                        # the perturbation ONLY on the samples the swap did not replace
        pe = self.ego.last.get("sel")
        self.overlap = 0 if pe is None else int((sw & pe).sum())                        # measured from the two masks, after both applies
        if self.overlap:
            raise RuntimeError(f"ZZEGO-DAGGER-OVERLAPZZ step {step}: {self.overlap} sample(s) are both swapped and perturbed (AMEND1 section 1: exactly 0)")
        return out


def make_compose(a, ego, dag):
    """argv -> the stage the training loop calls ``.apply`` on when ``--ego-perturb-with-dagger`` is named (AMEND1 section 1). The ONE place the flag becomes an object; ``bad_ego_dagger`` compares it back to the argv."""
    return EgoDaggerCompose(ego, dag, on=a.ego_perturb_with_dagger)


def bad_ego_dagger(a, comp, ego, dag) -> list:
    """The composition half of the launch gate (G-DVB). Empty = pass. ``--ego-perturb-with-dagger`` is read back from the BUILT stage; with no stage built it must be at its default (a flag that builds nothing is refused,
    not ignored); the stage holds the very perturber and swap the gate and the probes read back; and a composition whose BUILT levers are not both on (``--dagger-p 0`` or ``--ego-perturb-p 0``) is a lever that is
    declared and does nothing -- refused."""
    declared = bool(a.ego_perturb_with_dagger)
    if comp is None:
        return ["--ego-perturb-with-dagger is declared but no composed stage was built: a flag that builds nothing is refused, not ignored"] if declared else []
    bad = []
    got = comp.built_levers().get("ego_perturb_with_dagger")
    if got != declared:
        bad.append(f"lever ego_perturb_with_dagger: declared {declared!r} but built {got!r}")
    if comp.ego is not ego or comp.dag is not dag:
        bad.append("ego+dagger composition: the stage does not hold the perturber and the swap that the gate reads back and the probes exercise")
    want_stream = D.EGO_PERTURB_COMPOSED_STREAM if declared else 0
    if comp.ego.stream != want_stream:
        bad.append(f"ego+dagger composition: the BUILT perturbation draws from stream {comp.ego.stream} but the declaration needs {want_stream} "
                   f"({'independent of the swap' if declared else 'the original generator'}): the perturbation and the swap would not be the independent draws Amendment 1 section 1 registers")
    bad += [f"ego+dagger composition: {m}" for m in comp.problems()]
    return bad


def bad_init_ckpt(a, data_info) -> list:
    """``--init-ckpt`` read back (G-DVB): declared -> the record exists, names the declared path, carries the file's sha256, and EITHER the model state really IS that checkpoint's (``load_init_ckpt`` compared every
    tensor after the load) OR the load was skipped because ``--out`` already holds a ckpt_last.pt (resume wins); not declared -> nothing was loaded."""
    got = data_info.get("init_ckpt")
    if not a.init_ckpt:
        return ["init_ckpt not declared but a checkpoint was loaded"] if got else []
    if not got:
        return [f"init_ckpt {a.init_ckpt!r} declared but no checkpoint was loaded"]
    bad = []
    if os.path.abspath(a.init_ckpt) != got.get("path"):
        bad.append(f"init_ckpt: declared {os.path.abspath(a.init_ckpt)!r} but built {got.get('path')!r}")
    if not (isinstance(got.get("sha256"), str) and len(got["sha256"]) == 64):
        bad.append(f"init_ckpt: the checkpoint's sha256 is not recorded ({got.get('sha256')!r})")
    if got.get("applied"):
        if got.get("n_mismatch_after_load") != 0 or not got.get("n_tensors"):
            bad.append(f"init_ckpt: after the load {got.get('n_mismatch_after_load')} of {got.get('n_tensors')} tensors differ from the checkpoint's")
    elif got.get("skipped") != "resume":
        bad.append(f"init_ckpt: declared but neither applied nor skipped for a resume ({got.get('skipped')!r})")
    if got.get("applied"):                                  # PREREG_V8F_LEVEL: only the declared lever's keys may be fresh, and a lever-ON argv must have declared them
        if bool(got.get("declared_new_keys")) != (a.level_head == "on" or a.lead_head == "on"):          # PREREG_LEVELHEAD_A2: the lead head adds keys too
            bad.append(f"init_ckpt: level_head {a.level_head} / lead_head {a.lead_head} declared but the load declared new keys {got.get('declared_new_keys')}")
        if set(got.get("fresh_keys", [])) - set(got.get("declared_new_keys", [])):
            bad.append(f"init_ckpt: keys {sorted(set(got.get('fresh_keys', [])) - set(got.get('declared_new_keys', [])))} were left fresh but are not the declared lever's")
    return bad


#: the architecture-defining fields of V8FMiniConfig: a checkpoint may start a fine-tune only for a model of THIS shape and wiring (the training levers -- w_*, p_gt_*, code_mix, o_interface, cons_absent,
#: consistency, detach_upper -- may differ: a fine-tune moves them on purpose)
INIT_CKPT_MATCH = ("d", "heads", "ff", "d_feat", "n_tok", "hist", "hmax", "n_q", "depth_s", "depth_t", "depth_o", "nb", "res_scale_m", "navonly", "wm", "wm_dim", "wm_strides", "wm_beta", "wm_gamma",
                   "kin_mode", "max_age_t", "max_age_s",
                   "o_interface")                      # PREREG_V8F_FLATDAG: a checkpoint trained under the cut must be fine-tuned under the cut (and a full one under full). The key is read only when the checkpoint HAS it:
                                                       # checkpoints written before --o-interface existed (every M2b) carry none and load exactly as before. Without it a FLAT-DAG argv that forgets `--o-interface cut`
                                                       # builds FULL, loads the cut weights and PASSES the gate (MEASURED oaccflat_check/RESULT.md F1); the run is then REFUSED only by C-F3, hours later


def load_init_ckpt(path, model, cfg, apply: bool = True) -> dict:
    """Start state of a fine-tune. Hash the file; REFUSE (ValueError) a checkpoint whose architecture config differs from the BUILT one (``INIT_CKPT_MATCH``); then, if ``apply``, load its MODEL weights into
    ``model`` strictly (keys and shapes) and READ THEM BACK: every tensor of the checkpoint is compared with the model's after the load. ``apply`` is False when ``--out`` already holds a ckpt_last.pt (resume
    wins): the file is still hashed and config-checked, nothing is loaded. Returns the record the levers, selfcheck.json and the checkpoint carry."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck or not isinstance(ck.get("cfg"), dict):
        raise ValueError("not a v8f_mini_train checkpoint (no 'model' / 'cfg')")
    norm = lambda v: tuple(v) if isinstance(v, (list, tuple)) else v          # noqa: E731
    diff = {k: (norm(ck["cfg"][k]), norm(getattr(cfg, k))) for k in INIT_CKPT_MATCH if k in ck["cfg"] and norm(ck["cfg"][k]) != norm(getattr(cfg, k))}
    if diff:
        raise ValueError("config mismatch between the checkpoint and the model this argv builds (checkpoint vs built): " + "; ".join(f"{k}: {v[0]!r} != {v[1]!r}" for k, v in sorted(diff.items())))
    sd = ck["model"]
    rec = {"path": os.path.abspath(path), "sha256": h.hexdigest(), "applied": bool(apply), "skipped": None if apply else "resume", "n_tensors": len(sd), "n_mismatch_after_load": None,
           "source_step": ck.get("step"), "source_arm": ck.get("arm"), "source_seed": ck.get("seed"), "source_argv_sha": ck.get("argv_sha")}
    if not apply:
        return rec
    fresh_ok = sorted(k for k in model.state_dict() if k.startswith(M.NEW_LEVER_KEY_PREFIXES))       # PREREG_V8F_LEVEL / PREREG_LEVELHEAD_A2: the keys the declared levers add (level head, lead head, aux token; none for every earlier argv)
    if not fresh_ok:
        model.load_state_dict(sd, strict=True)                                                   # every earlier argv: the strict load, verbatim
    else:                                                                                        # a lever-ON fine-tune: EVERY key of the checkpoint loads, and EXACTLY the declared new keys are missing (fresh, zero/seeded init)
        res = model.load_state_dict(sd, strict=False)
        if res.unexpected_keys:
            raise ValueError(f"the checkpoint carries keys the built model does not have: {sorted(res.unexpected_keys)[:8]}")
        if set(res.missing_keys) - set(fresh_ok):
            raise ValueError(f"the built model lacks checkpoint values for keys that are not the declared lever's: {sorted(set(res.missing_keys) - set(fresh_ok))[:8]}")
        rec["fresh_keys"], rec["declared_new_keys"] = sorted(res.missing_keys), fresh_ok
    cur = model.state_dict()
    rec["n_mismatch_after_load"] = sum(1 for k in sd if not torch.equal(cur[k].detach().cpu(), sd[k].detach().cpu()))
    return rec


#: PREREG_V8F_MRATE: every other lever a WM-only argv must leave at its parser default (the readers, heads and planner are frozen and never run: a lever on them is declared and does nothing)
WM_ONLY_DEFAULT_DESTS = ("navonly", "no_consistency", "cons_absent", "code_mix", "o_interface", "w_acc", "launch_intent_w", "level_head", "w_level", "lead_head", "w_lead", "aux_cond", "detach_upper",
                         "ego_perturb_p", "ego_perturb_yawrate_std", "dagger_p", "dagger_bank", "dagger_bank_sha256", "ego_perturb_with_dagger", "recover_p", "recover_dev_bank", "recover_dev_bank_sha256",
                         "encoder", "lora_lr", "frames_root", "accum", "enc_ckpt", "window_start")


def is_wm_param(name: str) -> bool:
    return name.startswith("wm.")


def freeze_all_but_wm(model) -> None:
    """PREREG_V8F_MRATE: the WM-only mode trains the world model and nothing else. Called BEFORE the optimiser is built (its decay / nodecay lists filter on requires_grad)."""
    for n, p in model.named_parameters():
        p.requires_grad_(is_wm_param(n))


#: the INIT_CKPT_MATCH keys a WM-ONLY start does not compare: ``hmax`` (it only sizes the WM sequence: M2b 16, MRATE 28) and, PREREG_V8F_LFS, ``wm_strides`` (the WM is re-initialised, none of the checkpoint's WM tensors is loaded, and
#: the strides are the WM's own geometry; M2b's (2, 2) must not block an LFS arm at (2, 3) or (3, 3)). Every other key -- ``wm``, ``wm_dim``, ``wm_beta``, ``wm_gamma``, the readers' shape -- is still compared.
WM_ONLY_NOT_COMPARED = ("hmax", "wm_strides")


def load_wm_only_init(path, model, cfg, seed: int, apply: bool = True) -> dict:
    """Start state of a WM-only run (PREREG_V8F_MRATE, the Master Mind's reading of 2026-11-07 13:09). Hash the file; REFUSE a checkpoint whose architecture config differs from the BUILT one in any ``INIT_CKPT_MATCH`` key EXCEPT
    ``hmax`` (M2b trained at 16, this arm at 28: hmax only sizes the WM sequence); then, if ``apply``, load EVERY non-``wm.`` tensor strictly (the missing keys are exactly the model's ``wm.*`` keys, nothing is unexpected), READ THEM BACK
    (``n_mismatch_after_load``), and replace the world model by a FRESH ``WMAux`` built right after ``torch.manual_seed(seed)`` (none of the checkpoint's WM tensors is loaded; ``wm_params_equal_to_ckpt`` counts the shape-matching float
    parameters that happen to equal the checkpoint's: it must be 0). ``apply`` is False when ``--out`` already holds a ckpt_last.pt (resume wins)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck or not isinstance(ck.get("cfg"), dict):
        raise ValueError("not a v8f_mini_train checkpoint (no 'model' / 'cfg')")
    norm = lambda v: tuple(v) if isinstance(v, (list, tuple)) else v          # noqa: E731
    diff = {k: (norm(ck["cfg"][k]), norm(getattr(cfg, k))) for k in INIT_CKPT_MATCH if k not in WM_ONLY_NOT_COMPARED and k in ck["cfg"] and norm(ck["cfg"][k]) != norm(getattr(cfg, k))}
    if diff:
        raise ValueError("config mismatch between the checkpoint and the model this argv builds (checkpoint vs built; " + " and ".join(WM_ONLY_NOT_COMPARED) + " are not compared): " + "; ".join(f"{k}: {v[0]!r} != {v[1]!r}" for k, v in sorted(diff.items())))
    sd = ck["model"]
    rec = {"path": os.path.abspath(path), "sha256": h.hexdigest(), "applied": bool(apply), "skipped": None if apply else "resume", "n_tensors": len(sd), "n_mismatch_after_load": None,
           "source_step": ck.get("step"), "source_arm": ck.get("arm"), "source_seed": ck.get("seed"), "source_argv_sha": ck.get("argv_sha"), "mode": "wm_only",
           "hmax_ckpt": ck["cfg"].get("hmax"), "hmax_built": cfg.hmax, "wm_strides_ckpt": norm(ck["cfg"].get("wm_strides")), "wm_strides_built": norm(cfg.wm_strides),      # PREREG_V8F_LFS: the WM is fresh, so its strides are the argv's
           "wm_loaded_from_ckpt": 0, "wm_reinit_seed": None, "wm_params_equal_to_ckpt": None}
    if not apply:
        return rec
    rest = {k: v for k, v in sd.items() if not is_wm_param(k)}
    res = model.load_state_dict(rest, strict=False)
    wm_keys = sorted(k for k in model.state_dict() if is_wm_param(k))
    if res.unexpected_keys:
        raise ValueError(f"the checkpoint carries non-WM keys the built model does not have: {sorted(res.unexpected_keys)[:8]}")
    if sorted(res.missing_keys) != wm_keys:
        raise ValueError(f"the built model lacks checkpoint values for keys that are not its WM's: {sorted(set(res.missing_keys) - set(wm_keys))[:8]}")
    cur = model.state_dict()
    rec["n_mismatch_after_load"] = sum(1 for k in rest if not torch.equal(cur[k].detach().cpu(), rest[k].detach().cpu()))
    dev = next(model.parameters()).device
    torch.manual_seed(seed)                                                    # the WM is a function of (seed, cfg) only: W0 and W1 of one seed share every non-level-encoder parameter bit for bit
    model.wm = M.WMAux(cfg).to(dev)
    rec["wm_reinit_seed"] = seed
    old = {k: v for k, v in sd.items() if is_wm_param(k)}
    rec["wm_params_equal_to_ckpt"] = sum(1 for n, q in model.named_parameters() if is_wm_param(n) and n in old and old[n].shape == q.shape and torch.equal(old[n].detach().cpu(), q.detach().cpu()))
    return rec


def bad_wm_only(a, cfg, model, data_info) -> list:
    """G-DVB for ``--wm-k`` and ``--wm-only`` (PREREG_V8F_MRATE). ``--wm-k`` is read back from the BUILT modules (``WMAux.built_k``: window, dilation, causal pad per level). ``--wm-only`` is read back from the loader's record and the
    parameters' ``requires_grad``: the checkpoint's non-WM tensors are in the model bit-equal, no WM tensor was loaded, the WM was re-initialised under the run seed, the trainable set is exactly the WM's parameters, and every other
    lever is at its default."""
    bad = []
    built = model.built_levers()
    if built.get("wm_k", 1) != a.wm_k:
        bad.append(f"lever wm_k: declared {a.wm_k} but built {built.get('wm_k', 1)}")
    if a.wm_k > 1:
        want = {"wm_k_levels": [a.wm_k] * 3, "wm_dilation_levels": [1, *cfg.wm_strides], "wm_causal_pad_levels": [True] * 3}
        for k, v in want.items():
            if built.get(k) != v:
                bad.append(f"lever wm_k {a.wm_k}: built {k} {built.get(k)} != {v}")
    elif model.wm is not None and (model.wm.enc1 is not None or any(k != 1 for k in model.wm.built_k()["k"])):
        bad.append("wm_k 1 declared but a multi-frame world-model module was built")
    rec = data_info.get("wm_only")
    if not a.wm_only:
        if rec:
            bad.append("wm_only not declared but a WM-only start state was loaded")
        return bad
    if model.wm is None:
        return bad + ["wm_only declared but no world model was built (--wm off)"]
    if not a.init_ckpt:
        bad.append("wm_only declared without --init-ckpt: nothing to freeze the readers and the planner at")
    if not rec:
        return bad + ["wm_only declared but no WM-only start state was loaded"]
    if not cfg.w_wm > 0.0:
        bad.append(f"wm_only declared but w_wm is {cfg.w_wm}: the only loss term has weight 0 -- a lever that is declared and does nothing")
    dfl = build_parser()
    for d in WM_ONLY_DEFAULT_DESTS:
        if getattr(a, d) != dfl.get_default(d):
            bad.append(f"wm_only: --{d.replace('_', '-')} is {getattr(a, d)!r} (default {dfl.get_default(d)!r}): the readers, heads and planner are frozen and never run -- a lever that is declared and does nothing")
    if a.arm != "M2b":
        bad.append(f"wm_only: arm {a.arm} (the card trains the world model of M2b)")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    wm_names = [n for n, _ in model.named_parameters() if is_wm_param(n)]
    if not wm_names or sorted(names) != sorted(wm_names):
        bad.append(f"wm_only: trainable parameters are not exactly the WM's ({len(names)} trainable, {len(wm_names)} WM; stray: {sorted(set(names) - set(wm_names))[:4]})")
    if rec.get("applied"):
        if rec.get("wm_loaded_from_ckpt") != 0:
            bad.append(f"wm_only: {rec.get('wm_loaded_from_ckpt')} world-model tensors were loaded from the checkpoint (the card re-initialises the WM)")
        if rec.get("wm_reinit_seed") != a.seed:
            bad.append(f"wm_only: the WM was re-initialised under seed {rec.get('wm_reinit_seed')}, not --seed {a.seed}")
        if rec.get("wm_params_equal_to_ckpt") != 0:
            bad.append(f"wm_only: {rec.get('wm_params_equal_to_ckpt')} WM parameters equal the checkpoint's (the WM must be fresh)")
        if rec.get("mode") != "wm_only":
            bad.append(f"wm_only: the start state was loaded in mode {rec.get('mode')!r}")
    return bad


LFS_MIN_TRANSITIONS = 2        # PREREG_V8F_LFS: a level with fewer than this many training transitions per window is NOT RUN (reported with the count), never run at a different hmax


def wm_strides_readback(a, cfg, model) -> dict:
    """PREREG_V8F_LFS: the record ``selfcheck.json`` carries (``levers.wm_strides_readback``): the strides the argv DECLARED, the strides the BUILT modules use (three readers, ``WMAux.built_strides``) and the number of states and
    transitions each level holds for a window of ``cfg.hmax`` frames, measured by running the BUILT stack's ``encode``."""
    decl = declared_wm_strides(a)
    if model.wm is None:
        return {"declared": list(decl), "built": None, "states_per_level": None, "transitions_per_level": None}
    st = model.wm.level_states(cfg.hmax)
    return {"declared": list(decl), "built": model.wm.built_strides(), "hmax": cfg.hmax, "wm_k": a.wm_k, "states_per_level": st, "transitions_per_level": [n - 1 for n in st]}


def bad_wm_strides(a, cfg, model) -> list:
    """G-DVB for ``--wm-strides`` (PREREG_V8F_LFS). Empty = pass. The declared strides (the argv's, or the arm config's (2, 2) when the flag is absent) are read back from the BUILT modules -- the ``LevelConfig`` strides, the macro
    encoders' attribute and the stride their parameter shapes carry -- and from ``built_levers()``; a flag set whose strides were not applied (the config built at (2, 2)) is refused. A flag NAMED with no world model builds nothing
    and is refused. A stride that leaves a level fewer than ``LFS_MIN_TRANSITIONS`` training transitions at ``cfg.hmax`` is refused with the count (the arm is NOT RUN; it is not run at another hmax)."""
    bad = []
    decl = declared_wm_strides(a)
    if model.wm is None:
        if a.wm_strides is not None:
            bad.append(f"lever wm_strides: --wm-strides {list(decl)} declared but no world model was built (--wm off): a lever that is declared and builds nothing")
        return bad
    if tuple(cfg.wm_strides) != decl:
        bad.append(f"lever wm_strides: declared {list(decl)} but the built config holds {list(cfg.wm_strides)}")
    bs = model.wm.built_strides()
    for k in ("stride", "macro_stride", "macro_shape_stride"):
        if tuple(bs[k]) != decl:
            bad.append(f"lever wm_strides: declared {list(decl)} but the built world model's {k} is {bs[k]}")
    if a.wm_k > 1 and tuple(bs["dilation"]) != decl:
        bad.append(f"lever wm_strides: declared {list(decl)} but the built dilation is {bs['dilation']} (wm_k {a.wm_k}: the dilation of a level's window is its stride)")
    got = tuple(model.built_levers().get("wm_strides", WM_STRIDES_DEFAULT))
    if got != decl:
        bad.append(f"lever wm_strides: declared {list(decl)} but built_levers() says {list(got)}")
    tr = [n - 1 for n in model.wm.level_states(cfg.hmax)]
    if min(tr) < LFS_MIN_TRANSITIONS:
        bad.append(f"lever wm_strides {list(decl)}: NOT RUN -- at --hmax {cfg.hmax} the levels hold {tr} training transitions per window (L1, L2, L3), fewer than {LFS_MIN_TRANSITIONS} at level {tr.index(min(tr)) + 1}; "
                   "the arm is not run at another hmax (a different hmax would change B's windows)")
    return bad


def bad_wm_ladder(a, cfg, model, data_info) -> list:
    """G-DVB for the ladder levers (PREREG_V8F_LADDER). Empty = pass. Everything is read back from what was BUILT, never from the config that built it: ``--wm-rollout-k`` from the predictor calls the BUILT loss makes inside
    ``HJEPAStack.rollout`` (``WMAux.built_ladder``), ``--wm-rollout-stride`` from the free run's measured anchor count, ``--wm-rollout-w`` from the weight the built level loss gives the term, ``--wm-ctx`` from the predictor's
    parameter shape and from the context the free run is handed, ``--wm-windows`` / ``--wm-win-stride`` / ``--wm-pooled`` from the BUILT dataset's record (``data_info["wm_windows"]``). A lever that is declared and builds
    nothing is refused: any ladder flag with no world model, a stride or a weight on levels whose k is 1, wmonly windows without ``--wm-only``, the pooled path without wmonly windows."""
    bad = []
    named = wm_ladder_named(a)
    decl = declared_wm_ladder(a)
    rec = data_info.get("wm_windows")
    if model.wm is None:
        return [f"lever {n}: declared but no world model was built (--wm off): a lever that is declared and builds nothing" for n in named]
    rb = model.wm.built_ladder(cfg.hmax)
    cum = rb["cum_stride"]
    for i in range(3):
        k, st, cx = decl["wm_rollout_k"][i], decl["wm_rollout_stride"][i], decl["wm_ctx"][i]
        if rb["k"][i] != (k if k > 1 else 0):
            bad.append(f"lever wm_rollout_k: level {i + 1} declared k={k} but the built loss runs {rb['k'][i]} free-run predictor steps (0 = no k-step term)")
        if rb["ctx_built"][i] != cx:
            bad.append(f"lever wm_ctx: level {i + 1} declared {cx} but the built predictor's context (pos rows) is {rb['ctx_built'][i]}")
        if k > 1:
            want_anchors = len(range(0, rb["states"][i] - 1, st))
            if rb["anchors"][i] != want_anchors:
                bad.append(f"lever wm_rollout_stride: level {i + 1} declared stride {st} (-> {want_anchors} anchors on {rb['states'][i]} states) but the built free run has {rb['anchors'][i]} anchors")
            w = rb["w_in_loss"][i]
            if w is None or abs(w - decl["wm_rollout_w"]) > 1e-4 * max(1.0, abs(decl["wm_rollout_w"])):
                bad.append(f"lever wm_rollout_w: level {i + 1} declared {decl['wm_rollout_w']} but the built level loss gives the k-step term weight {w}")
            if rb["ctx_passed"][i] != rb["ctx_built"][i]:
                bad.append(f"lever wm_ctx: level {i + 1} free run is handed context {rb['ctx_passed'][i]} but the predictor's context is {rb['ctx_built'][i]}")
            if cfg.hmax < k * cum[i] + 1:
                bad.append(f"lever wm_rollout_k: level {i + 1} k={k} at cumulative stride {cum[i]} needs {k * cum[i] + 1} frames, the window holds {cfg.hmax}: the declared reach could never be trained")
        elif st != 1:
            bad.append(f"lever wm_rollout_stride: level {i + 1} declares stride {st} but k is 1 (no free-run term is built): a lever that is declared and does nothing")
    if "wm_rollout_w" in named and all(k == 1 for k in decl["wm_rollout_k"]):
        bad.append(f"lever wm_rollout_w: declared {decl['wm_rollout_w']} but every level's k is 1 (no k-step term): a lever that is declared and does nothing")
    if decl["wm_rollout_w"] == 0.0 and any(k > 1 for k in decl["wm_rollout_k"]):
        bad.append("lever wm_rollout_w: declared 0.0 with a k-step term built: the term would carry no gradient -- a lever that is declared and does nothing")
    if a.wm_windows == "wmonly":
        if not a.wm_only:
            bad.append("lever wm_windows: wmonly declared without --wm-only (the labelled readers need labelled windows)")
        if not rec or rec.get("kind") != "wmonly":
            bad.append(f"lever wm_windows: wmonly declared but the built dataset is {None if not rec else rec.get('kind')}")
        else:
            if rec.get("F") != cfg.hmax:
                bad.append(f"lever wm_windows: the built dataset's window is {rec.get('F')} frames, hmax is {cfg.hmax}")
            if rec.get("t_min") is None or rec["t_min"] < cfg.hmax - 1:
                bad.append(f"lever wm_windows: the first window t is {rec.get('t_min')}, a full {cfg.hmax}-frame window needs t >= {cfg.hmax - 1}")
            if rec.get("stride") != a.wm_win_stride:
                bad.append(f"lever wm_win_stride: declared {a.wm_win_stride} but the built list's stride is {rec.get('stride')}")
            if (rec.get("zcache") or "") != (a.wm_zcache or ""):
                bad.append(f"lever wm_zcache: declared {a.wm_zcache!r} but the built dataset reads {rec.get('zcache')!r}")
            if rec.get("zcache_missing"):
                bad.append(f"lever wm_zcache: {rec['zcache_missing']} clips of this run have no file in {a.wm_zcache}")
            if rec.get("pooled") != (a.wm_pooled == "on"):
                bad.append(f"lever wm_pooled: declared {a.wm_pooled} but the built dataset pooled={rec.get('pooled')}")
            if not rec.get("n_windows"):
                bad.append("lever wm_windows: the built window list is empty")
    else:
        if rec:
            bad.append("lever wm_windows: train declared but a world-model-only dataset was built")
        if a.wm_zcache:
            bad.append("lever wm_zcache: declared without --wm-windows wmonly: a lever that is declared and builds nothing")
        if a.wm_pooled == "on":
            bad.append("lever wm_pooled: on declared without --wm-windows wmonly: a lever that is declared and builds nothing")
        if a.wm_win_stride != 1:
            bad.append(f"lever wm_win_stride: {a.wm_win_stride} declared without --wm-windows wmonly: a lever that is declared and does nothing")
    return bad


def adapter_state_sha256(state: dict) -> str:
    """PREREG_V8F_WMPOOL: sha256 over name + shape + dtype + bytes of the sorted keys ``adapter.*`` and ``pos`` of a model state dict: the SAME formula as ``theory/ladder/zcache.py adapter_sha``."""
    h = hashlib.sha256()
    for k in sorted(k for k in state if k.startswith("adapter.") or k == "pos"):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode() + str(tuple(t.shape)).encode() + str(t.dtype).encode() + t.numpy().tobytes())
    return h.hexdigest()


def pool_cache_problems(a, adapter_sha: str | None = None) -> list:
    """PREREG_V8F_WMPOOL: why ``--wm-pool att25`` cannot run (empty = fine). Static part (no model): the windows kind and a cell cache directory with a matching ``CELL_MANIFEST.json``; with ``adapter_sha`` (the BUILT model's adapter) also the manifest's adapter sha."""
    if a.wm_pool != "att25":
        return []
    bad = []
    if not a.wm_only or a.wm_windows != "wmonly":
        bad.append("--wm-pool att25 needs --wm-only --wm-windows wmonly (the only path that reads cells): a lever that is declared and builds nothing")
    if not a.wm_zcache:
        bad.append("--wm-pool att25 needs --wm-zcache = a CELL cache directory (zcells.py): none given")
        return bad
    mp = os.path.join(a.wm_zcache, "CELL_MANIFEST.json")
    if not os.path.exists(mp):
        bad.append(f"--wm-pool att25: {mp} does not exist (a mean cache is not a cell cache)")
        return bad
    man = json.load(open(mp, encoding="utf-8"))
    for k, v in (("kind", "zcell"), ("grid", "pool4x10->2x5"), ("n_cells", 10), ("dtype", "float16"), ("d", 256)):
        if man.get(k) != v:
            bad.append(f"--wm-pool att25: CELL_MANIFEST {k} is {man.get(k)!r}, expected {v!r}")
    if adapter_sha is not None and man.get("adapter_state_sha256") != adapter_sha:
        bad.append(f"--wm-pool att25: the cell cache was built from adapter {str(man.get('adapter_state_sha256'))[:12]}, the built model's adapter is {adapter_sha[:12]}")
    return bad


def bad_wm_pool(a, cfg, model, data_info) -> list:
    """G-DVB for ``--wm-pool`` (PREREG_V8F_WMPOOL). Empty = pass. Read back from what was BUILT: the mode and the query from :meth:`WMAux.built_pool` (planted-input measurements), the config, the BUILT dataset's record and the cache manifest vs the BUILT adapter."""
    bad = []
    want = a.wm_pool
    if model.wm is None:
        return [f"lever wm_pool: {want} declared but no world model was built (--wm off)"] if want != "mean" else []
    bp = model.wm.built_pool()
    if bp["mode"] != want:
        bad.append(f"lever wm_pool: declared {want} but the built module behaves as {bp['mode']!r}")
    if model.cfg.wm_pool != want:
        bad.append(f"lever wm_pool: declared {want} but the built config says {model.cfg.wm_pool}")
    rec = data_info.get("wm_windows") or {}
    if want == "mean":
        if rec.get("pool", "mean") != "mean":
            bad.append(f"lever wm_pool: mean declared but the built dataset serves {rec.get('pool')!r}")
        return bad
    if bp["n_cells"] != 10 or not bp["cell_map_ok"] or bp["n_params"] != 256 or bp["q_shape"] != [256] or not bp["q_zero_at_build"] or not bp["uniform_at_build"] or not bp["attends_when_set"]:
        bad.append(f"lever wm_pool: the built pool is {bp}: 10 cells, the 2 x 2 block map, ONE zero-initialised [256] query, uniform weights at build and attention under a probe query were declared")
    if rec.get("pool") != "att25":
        bad.append(f"lever wm_pool: att25 declared but the built dataset serves {rec.get('pool')!r}")
    if rec.get("zcell_shape") != [cfg.hmax, 10, 256]:
        bad.append(f"lever wm_pool: the built dataset's first item has cell shape {rec.get('zcell_shape')}, expected {[cfg.hmax, 10, 256]}")
    bad += pool_cache_problems(a, adapter_state_sha256(model.state_dict()))
    return bad


def wm_pool_readback(a, cfg, model, data_info) -> dict:
    """PREREG_V8F_WMPOOL: the record ``selfcheck.json`` carries (``levers.wm_pool_readback``) when the argv NAMES the flag: declared vs the BUILT measurements."""
    rec = data_info.get("wm_windows") or {}
    return {"declared": a.wm_pool, "built": None if model.wm is None else model.wm.built_pool(), "dataset_pool": rec.get("pool"), "dataset_zcell_shape": rec.get("zcell_shape"), "zcache": a.wm_zcache,
            "cache_problems": pool_cache_problems(a, adapter_state_sha256(model.state_dict()))}


def wm_ladder_readback(a, cfg, model, data_info) -> dict:
    """PREREG_V8F_LADDER: the record ``selfcheck.json`` carries (``levers.wm_ladder_readback``) when the argv NAMES a ladder flag: declared vs the BUILT measurements."""
    decl = declared_wm_ladder(a)
    return {"declared": {k: list(v) if isinstance(v, tuple) else v for k, v in decl.items()}, "named": wm_ladder_named(a), "windows_declared": {"kind": a.wm_windows, "stride": a.wm_win_stride, "pooled": a.wm_pooled},
            "built": None if model.wm is None else model.wm.built_ladder(cfg.hmax), "windows_built": data_info.get("wm_windows"), "hmax": cfg.hmax}


def wm_only_live_check(model, batch, cfg, steps: int = 3, lr: float = 1e-3) -> dict:
    """G-LIVE for the WM-only mode (PREREG_V8F_MRATE control 3): three optimiser steps of ``wm_only_loss`` on a real batch, on a COPY with EVERY parameter thawed (so a missing ``.detach()`` shows up as a gradient on the adapter). Every WM loss term
    finite; every WM parameter receives a non-zero gradient at some step (a=without IDM: the IDM heads are declared dead); the adapter, the readers, the heads and the planner receive EXACTLY 0.0. ``--w-wm 0`` zeroes every WM gradient and FAILS."""
    m = copy.deepcopy(model)
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    wm_params = [p for n, p in m.named_parameters() if is_wm_param(n)]
    opt = torch.optim.AdamW(wm_params, lr=lr)
    acc = {n: 0.0 for n, _ in m.named_parameters()}
    terms_seen = None
    for s in range(steps):
        opt.zero_grad(set_to_none=False)
        for n, p in m.named_parameters():
            if not is_wm_param(n):
                p.grad = None
        torch.manual_seed(s)
        T = m.wm_only_loss(batch)["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"G-LIVE: WM loss term {k} is not finite at step {s}: {float(v)}")
        terms_seen = {k: float(v.detach()) for k, v in T.items()}
        T["total"].backward()
        for n, p in m.named_parameters():
            if p.grad is not None:
                acc[n] += float(p.grad.abs().sum())
        opt.step()
    expect = {"wm", "total"} | {f"wm/{t}/L{l}" for t in ("pred", "sigreg", "idm", "level") for l in (1, 2, 3)} | {f"wm/act_sigreg/L{l}" for l in (2, 3)}
    missing = sorted(expect - set(terms_seen))
    declared_dead = [n for n in acc if cfg.wm == "a" and ".idm_heads." in n]
    dead = sorted(n for n, g in acc.items() if is_wm_param(n) and g == 0.0)
    unexpected_dead = sorted(set(dead) - set(declared_dead))
    groups = {}
    for n, g in acc.items():
        if is_wm_param(n):
            parts = n.split(".")                                           # wm.proj / wm.enc1 / wm.stack.act_embed / wm.stack.<encoders|macro_encoders|predictors|idm_heads>.<level>
            key = ".".join(parts[:4] if parts[1] == "stack" and parts[2] in ("encoders", "macro_encoders", "predictors", "idm_heads") else parts[:3] if parts[1] == "stack" else parts[:2])
            groups[key] = groups.get(key, 0.0) + g
    dead_groups = sorted(k for k, g in groups.items() if g == 0.0 and not (cfg.wm == "a" and ".idm_heads." in k))
    nonwm = {n: g for n, g in acc.items() if not is_wm_param(n)}
    res = {"terms": terms_seen, "missing_terms": missing, "wm_groups_grad": groups, "dead_wm_groups": dead_groups, "dead_params": dead, "declared_dead": declared_dead, "unexpected_dead": unexpected_dead,
           "n_wm_params": len(wm_params), "control_nonwm_grad_sum": float(sum(nonwm.values())), "control_adapter_grad_sum": float(sum(g for n, g in nonwm.items() if n.startswith("adapter.") or n == "pos")),
           "nonwm_with_grad": sorted(n for n, g in nonwm.items() if g != 0.0)[:8], "n_nonwm_params": len(nonwm)}
    ok = (not missing and not dead_groups and not unexpected_dead and res["control_nonwm_grad_sum"] == 0.0 and res["n_wm_params"] > 0 and res["n_nonwm_params"] > 0)
    res["pass"] = bool(ok)
    if not ok:
        raise AssertionError("G-LIVE (wm_only) failed: " + json.dumps({k: v for k, v in res.items() if k != "wm_groups_grad"}, default=str)[:1500])
    return res


def _tensors_of(o, prefix=""):
    if isinstance(o, torch.Tensor):
        yield prefix, o
    elif isinstance(o, dict):
        for k in sorted(o, key=str):
            yield from _tensors_of(o[k], f"{prefix}/{k}")


def planner_identity_probe(model, batch, cfg, steps: int = 3, lr: float = 1e-3) -> dict:
    """PREREG_V8F_MRATE control 1 (planner identity), the in-trainer form: ``forward_infer`` on a fixed batch (eval, CPU, a COPY) is compared tensor by tensor with ``torch.equal`` before and after ``steps`` WM-only optimiser steps. The WM must
    have MOVED (the control: the steps did something) and every planner output must be bit-equal."""
    m = copy.deepcopy(model)
    m.eval()
    with torch.no_grad():
        before = {k: v.clone() for k, v in _tensors_of(m.forward_infer(batch))}
    wm0 = {n: p.detach().clone() for n, p in m.named_parameters() if is_wm_param(n)}
    m.train()
    opt = torch.optim.AdamW([p for n, p in m.named_parameters() if p.requires_grad], lr=lr)
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(s)
        m.wm_only_loss(batch)["terms"]["total"].backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        after = {k: v for k, v in _tensors_of(m.forward_infer(batch))}
    same = sorted(k for k in before if k in after and before[k].shape == after[k].shape and torch.equal(before[k], after[k]))
    moved = sum(1 for n, p in m.named_parameters() if is_wm_param(n) and not torch.equal(p.detach(), wm0[n]))
    res = {"n_tensors": len(before), "n_equal": len(same), "differing": sorted(set(before) - set(same))[:8], "wm_params_moved": moved, "n_wm_params": len(wm0), "steps": steps, "pass": False}
    res["pass"] = bool(len(before) > 0 and len(same) == len(before) and set(before) == set(after) and moved > 0)
    if not res["pass"]:
        raise AssertionError("planner identity failed: " + json.dumps(res, default=str)[:1200])
    return res


class LRSchedule:
    """The loop's learning-rate schedule: linear warmup over ``warmup`` steps to ``lr``, then a cosine to 0.1 x ``lr`` at ``total``. ONE definition: the loop CALLS it and ``bad_schedule`` (G-DVB) READS IT BACK
    -- the declared ``--warmup`` is checked against the schedule that was built, not against the argv it was built from."""

    def __init__(self, lr: float, warmup: int, total: int):
        self.lr, self.warmup, self.total = lr, warmup, total

    def __call__(self, s):
        if s < self.warmup:
            return self.lr * (s + 1) / self.warmup
        return self.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((s - self.warmup) / max(self.total - self.warmup, 1), 1.0))))

    def built_levers(self) -> dict:
        return {"lr": self.lr, "warmup": self.warmup, "total": self.total}


def bad_schedule(a, sched) -> list:
    """G-DVB for ``--warmup`` (and the peak ``--lr``): the BUILT schedule's warmup / peak equal the argv's, AND its behaviour says the same -- the first step is lr / warmup and the last warmup step reaches lr.
    With no schedule built, ``--warmup`` must be at its default (a flag that builds nothing is refused, not ignored)."""
    dfl = build_parser().get_default("warmup")
    if sched is None:
        return [f"warmup {a.warmup} is set but no learning-rate schedule was built: a flag that builds nothing is refused, not ignored"] if a.warmup != dfl else []
    bad = []
    if sched.warmup != a.warmup:
        bad.append(f"lever warmup: declared {a.warmup} but the built schedule's warmup is {sched.warmup}")
    if sched.lr != a.lr:
        bad.append(f"lever lr: declared {a.lr} but the built schedule's peak is {sched.lr}")
    if a.warmup >= 1:
        first, last = sched(0), sched(a.warmup - 1)
        if not (abs(first - a.lr / a.warmup) <= 1e-12 * max(abs(a.lr), 1.0) and abs(last - a.lr) <= 1e-12 * max(abs(a.lr), 1.0)):
            bad.append(f"warmup {a.warmup}: the built schedule reads {first!r} at step 0 and {last!r} at step {a.warmup - 1}, not {a.lr / a.warmup!r} and {a.lr!r}")
    return bad


RECOVER_DESTS = ("recover_p", "recover_dev_bank", "recover_dev_bank_sha256", "recover_dev_law", "recover_dev_shape", "recover_target", "recover_rejoin_steps", "recover_seed")


def load_recover_table(a, J, wins, store, clips, info=None) -> tuple:
    """``--recover-dev-bank`` -> (the empirical deviation table, the controls record). PREREG_RECOVER_FT section 3. The controls, each refusing with its own message (``DG.BankRefused``):

      load      every shard re-hashed against the manifest, the manifest's sha256 == ``--recover-dev-bank-sha256`` (``DaggerBank.load``); a bare shard / unmerged bank is refused
      K-B3      the bank is not PARTIAL, every bank clip is in the arm's TRAIN clip list, none is an eval-141 clip (id AND stable sid, read first), and the window-list sha256 the bank recorded == this trainer's
      table     built deterministically from the bank's ``reason == ok`` records and THIS trainer's own clip store (``R.RecoverDevTable.from_dagger_bank``); an empty table is refused

    K-B2 (on-policy against ``--init-ckpt``) is NOT read: the card draws deviations from the model family's own states (bank2 = DAG's, the init is DAG-OACC), so the bank's generating checkpoint is disclosed in the
    record (``bank_ckpt_sha256`` beside ``init_ckpt_sha256``), never compared."""
    bank0 = DG.DaggerBank.load(a.recover_dev_bank, expect_sha256=a.recover_dev_bank_sha256 or None)
    if bank0.manifest is None:
        raise DG.BankRefused(f"{a.recover_dev_bank}: a bare shard, not a merged bank (no bank.json): the bank's identity is the manifest's sha256")
    eval_clips, eval_src = dagger_eval_clips(a, bank0.header)
    wsha = DG.windows_sha256(J["clip"], wins)
    kb3 = bank0.check_train_only(train_clips=clips, eval_clips=eval_clips, train_window_list_sha256=wsha)
    bank = DG.DaggerBank(bank0.a, bank0.header, clip_names=J["clip"], manifest=bank0.manifest, sha256=bank0.sha256, path=bank0.path)
    table, log_v = R.RecoverDevTable.from_dagger_bank(bank, store, J, per_tick=(a.recover_dev_shape == "pertick"))     # pertick: the table also carries the [N, F] profiles
    if len(table) == 0:
        raise DG.BankRefused("the deviation table is EMPTY (no reason == ok record has a real logged kinematic row): a lever with nothing to draw")
    ctl = {"bank_path": bank.path, "bank_sha256": bank.sha256, "n_records": bank.n, "reason_counts": bank.reason_counts(), "dev_shape": a.recover_dev_shape,
           "kb3": {**kb3, "window_list_sha256": wsha, "eval_sources": eval_src, "n_eval_clips_checked": len(eval_clips)},
           "kb2_read": False, "bank_ckpt_sha256": bank.header.get("ckpt_sha256"), "init_ckpt_sha256": ((info or {}).get("init_ckpt") or {}).get("sha256"),
           "table": table.summary(log_v), "table_quantiles": table.quantiles(), "table_band_3_8": table.band_coverage(log_v)}
    return table, ctl


def make_recover(a, J, wins, store, clips, info):
    """argv -> the perturber the training loop calls ``.apply`` on (PREREG_RECOVER_FT). The ONE place the eight flags become an object; ``bad_recover`` compares it back to the argv. With ``--recover-p 0`` (C0) no bank
    is read and no table is built (a declared bank at p = 0 builds nothing and is refused by the gate: a lever that is declared and does nothing); the a-priori law reads no bank either. A refused bank stops the launch
    before the gate. The built lever ``recover_dev_bank`` is the LOADED manifest's path and ``recover_dev_bank_sha256`` the LOADED table's source hash, so G-DVB compares the declaration with what was read."""
    table, path = None, (a.recover_dev_bank or None)
    if a.recover_p > 0 and a.recover_dev_law == "empirical" and a.recover_dev_bank:
        try:
            table, ctl = load_recover_table(a, J, wins, store, clips, info)      # info: the init checkpoint's sha256 is disclosed beside the bank's generating checkpoint (never compared: no K-B2)
        except DG.BankRefused as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ recover-dev-bank {a.recover_dev_bank}: {e}")
        except Exception as e:                                       # noqa: BLE001  a missing / foreign / corrupt bank is a REFUSAL, not a crash
            raise SystemExit(f"ZZDVB-REFUSEDZZ recover-dev-bank {a.recover_dev_bank}: {type(e).__name__}: {str(e)[:600]}")
        info["recover"] = ctl                                        # selfcheck.json carries the controls and the table's summary
        path = table.path
    return R.RecoverPerturb(p=a.recover_p, dev_law=a.recover_dev_law, table=table, target=a.recover_target, rejoin_steps=a.recover_rejoin_steps, seed=a.recover_seed,
                            kin_mode=a.kin_mode, bank_path=path, bank_sha256=(a.recover_dev_bank_sha256 or None), dev_shape=a.recover_dev_shape)


def bad_recover(a, rec, data_info=None) -> list:
    """The recovery half of the launch gate (G-DVB). Empty = pass. Every flag is read back from the BUILT perturber (``rec``); with no perturber built every one must be at its default (a flag that builds nothing is
    refused, not ignored). Beyond the flags: the bank's path and sha256 against the declaration (the file on disk is re-hashed), K-B3 and the table were read (the record exists and says so), exclusivity with
    --ego-perturb-p / --dagger-p / --ego-perturb-with-dagger, and a lever declared without an effect (any recover flag at p = 0, a bank with the a-priori law) is refused."""
    dfl = {k: build_parser().get_default(k) for k in RECOVER_DESTS}
    declared = {k: getattr(a, k) for k in dfl}
    if rec is None:
        moved = sorted(k for k in dfl if declared[k] != dfl[k])
        return [f"{moved} are set but no recovery perturber was built: a flag that builds nothing is refused, not ignored"] if moved else []
    bad = []
    lv = rec.built_levers()
    for k in ("recover_p", "recover_dev_law", "recover_dev_shape", "recover_target", "recover_rejoin_steps", "recover_seed"):
        if lv.get(k) != declared[k]:
            bad.append(f"lever {k}: declared {declared[k]!r} but built {lv.get(k)!r}")
    if lv.get("kin_mode") != a.kin_mode:
        bad.append(f"recovery perturber kin_mode: declared {a.kin_mode} but built {lv.get('kin_mode')}")
    bad += [f"recovery perturber: {m}" for m in rec.problems()]
    if a.recover_dev_bank_sha256 and not is_sha256_hex(a.recover_dev_bank_sha256):
        bad.append(f"--recover-dev-bank-sha256 {a.recover_dev_bank_sha256!r} is not a sha256 (64 hex digits): a truncated hash identifies nothing")
    if a.recover_p > 0:
        for flag, on in (("--ego-perturb-p", a.ego_perturb_p > 0), ("--dagger-p", a.dagger_p > 0), ("--ego-perturb-with-dagger", bool(a.ego_perturb_with_dagger))):
            if on:
                bad.append(f"--recover-p {a.recover_p} together with {flag}: the recovery lever is exclusive with the DAgger swap and the ego perturbation (PREREG_RECOVER_FT section 4)")
    if a.recover_p > 0 and a.recover_dev_law == "apriori" and (a.recover_dev_bank or a.recover_dev_bank_sha256):
        bad.append("--recover-dev-law apriori with a --recover-dev-bank / --recover-dev-bank-sha256 declared: the bank would do nothing (a lever that is declared and does nothing)")
    if a.recover_p > 0 and a.recover_dev_shape == "pertick" and a.recover_dev_law == "apriori":
        bad.append("--recover-dev-shape pertick with --recover-dev-law apriori: the a-priori law has no per-tick profile to draw (a lever that is declared and does nothing)")
    if a.recover_p > 0 and a.recover_dev_law == "empirical":
        if not a.recover_dev_bank:
            bad.append(f"--recover-dev-bank is required when --recover-p is {a.recover_p} with the empirical law: the arm must declare which bank its deviations come from")
        if not a.recover_dev_bank_sha256:
            bad.append(f"--recover-dev-bank-sha256 is required when --recover-p is {a.recover_p} with the empirical law: the arm must declare which bank it draws from")
        if a.recover_dev_bank and a.recover_dev_bank_sha256:
            if rec.table is None:
                bad.append(f"lever recover_dev_bank: declared {a.recover_dev_bank!r} but no table was built from it")
                return bad
            want = os.path.normcase(os.path.abspath(dagger_bank_manifest(a.recover_dev_bank)))
            got = os.path.normcase(os.path.abspath(str(lv.get("recover_dev_bank"))))
            if want != got:
                bad.append(f"lever recover_dev_bank: declared {want!r} but built {got!r}")
            disk = DG.sha256_file(dagger_bank_manifest(a.recover_dev_bank)) if os.path.exists(dagger_bank_manifest(a.recover_dev_bank)) else None
            if lv.get("recover_dev_bank_sha256") != disk:
                bad.append(f"lever recover_dev_bank_sha256: the file {a.recover_dev_bank!r} hashes to {disk!r} but the table was built from {lv.get('recover_dev_bank_sha256')!r}")
            if a.recover_dev_bank_sha256.lower() != lv.get("recover_dev_bank_sha256"):
                bad.append(f"lever recover_dev_bank_sha256: declared {a.recover_dev_bank_sha256.lower()!r} but built {lv.get('recover_dev_bank_sha256')!r}")
            ctl = (data_info or {}).get("recover")
            if not ctl:
                bad.append("a deviation table is built but no K-B3 / table record was read (the controls did not run)")
            else:
                if ctl.get("bank_sha256") != lv.get("recover_dev_bank_sha256"):
                    bad.append(f"the controls were read on bank {str(ctl.get('bank_sha256'))[:12]} but the table holds {str(lv.get('recover_dev_bank_sha256'))[:12]}")
                kb3 = ctl.get("kb3") or {}
                if kb3.get("window_list_sha256") != (data_info or {}).get("train_window_list_sha256") or kb3.get("overlap_with_eval") != 0:
                    bad.append("K-B3 was not read against this trainer's TRAIN window list / with an empty eval overlap")
                if (ctl.get("table") or {}).get("content_sha256") != rec.table.content_sha256() or not (ctl.get("table") or {}).get("n", 0) > 0:
                    bad.append("the recorded table summary is not the BUILT table's (content sha256) or is empty")
    return bad


def make_t2bank(a, info):
    """argv -> the swap the training loop calls ``.apply`` on (PREREG_T2_DAGGER). The ONE place the four flags become an object; ``bad_t2bank`` compares it back to the argv. With ``--t2-bank-p 0`` nothing is read."""
    bank, path = None, (a.t2_bank or None)
    if a.t2_bank_p > 0 and a.t2_bank:
        try:
            bank = TB.T2Bank.load(a.t2_bank, expect_sha256=a.t2_bank_sha256 or None)
        except (TB.BankRefused, OSError, KeyError, ValueError) as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ t2-bank {a.t2_bank}: {type(e).__name__}: {str(e)[:600]}")
        info["t2bank"] = {"bank_path": bank.path, "bank_sha256": bank.sha256, "n_rows": len(bank), "n_scenes": len(bank.scene_ids), "scene_ids": bank.scene_ids}
        path = bank.path
    return TB.T2BankSwap(p=a.t2_bank_p, bank=bank, target=a.recover_target, rejoin_steps=a.recover_rejoin_steps, seed=a.t2_bank_seed, bank_path=path, bank_sha256=(a.t2_bank_sha256 or None))


def bad_t2bank(a, t2b, data_info=None) -> list:
    """The T2-bank half of the launch gate (G-DVB). Empty = pass. Every flag is read back from the BUILT swap (``t2b``); a flag that builds nothing is refused; the lever is exclusive with the other own-state levers and the LoRA encoder."""
    dfl = {k: build_parser().get_default(k) for k in ("t2_bank_p", "t2_bank", "t2_bank_sha256", "t2_bank_seed")}
    if t2b is None:
        moved = sorted(k for k, d in dfl.items() if getattr(a, k) != d)
        return [f"{moved} are set but no T2 bank swap was built: a flag that builds nothing is refused, not ignored"] if moved else []
    lv, bad = t2b.built_levers(), []
    if lv["t2_bank_p"] != a.t2_bank_p:
        bad.append(f"lever t2_bank_p: declared {a.t2_bank_p!r} but built {lv['t2_bank_p']!r}")
    if lv["t2_bank_seed"] != a.t2_bank_seed:
        bad.append(f"lever t2_bank_seed: declared {a.t2_bank_seed!r} but built {lv['t2_bank_seed']!r}")
    bad += [f"t2 bank swap: {m}" for m in t2b.problems()]
    if a.t2_bank_sha256 and not is_sha256_hex(a.t2_bank_sha256):
        bad.append(f"--t2-bank-sha256 {a.t2_bank_sha256!r} is not a sha256 (64 hex digits): a truncated hash identifies nothing")
    if a.t2_bank_p > 0:
        for flag, on in (("--recover-p", a.recover_p > 0), ("--dagger-p", a.dagger_p > 0), ("--ego-perturb-p", a.ego_perturb_p > 0), ("--ego-perturb-with-dagger", bool(a.ego_perturb_with_dagger)), ("--encoder lora", a.encoder == "lora"), ("--wm-only", bool(a.wm_only))):
            if on:
                bad.append(f"--t2-bank-p {a.t2_bank_p} together with {flag}: the T2 bank lever is exclusive with the other own-state levers and the LoRA encoder (PREREG_T2_DAGGER)")
        if not a.t2_bank or not a.t2_bank_sha256:
            bad.append("--t2-bank and --t2-bank-sha256 are both required when --t2-bank-p > 0: the arm must declare which bank it draws from")
        elif t2b.bank is None:
            bad.append(f"lever t2_bank: declared {a.t2_bank!r} but no bank was built from it")
        else:
            want = os.path.normcase(os.path.abspath(os.path.join(a.t2_bank, "bank.json")))
            got = os.path.normcase(os.path.abspath(str(lv.get("t2_bank"))))
            if want != got:
                bad.append(f"lever t2_bank: declared {want!r} but built {got!r}")
            if lv.get("t2_bank_sha256") != a.t2_bank_sha256.lower():
                bad.append(f"lever t2_bank_sha256: declared {a.t2_bank_sha256.lower()!r} but built {lv.get('t2_bank_sha256')!r}")
    return bad


def t2bank_probe(model, batch, cfg, t2b) -> dict:
    """G-LIVE for PREREG_T2_DAGGER: a batch whose rows are ALL bank rows (p = 1) reads EXACTLY 0.0 on every masked loss term and a finite, non-zero target term (ade) whose gradient reaches the model; a p = 0 control
    returns the SAME batch object. Violations are AssertionErrors (the caller turns them into a refusal). Delete the masking in ``T2BankSwap.apply`` / the ``aux_ok`` gating in ``V8FMini.loss`` and a term reads non-zero."""
    if not t2b.active:
        return {"t2_bank": "off", "pass": True, "skipped": "--t2-bank-p 0: no row is swapped; the default run is the pre-change run"}
    import copy
    ctl = t2b.apply(batch, TB.T2BankSwap.PROBE_STEP, p=0.0)
    assert ctl is batch, "t2 bank probe: a p = 0 call did not return the SAME batch object"
    allb = t2b.apply(batch, TB.T2BankSwap.PROBE_STEP, p=1.0)
    assert int(allb["aux_ok"].sum()) == 0 and int(t2b.last["n_selected"]) == int(batch["feats"].shape[0]), "t2 bank probe: p = 1 did not select every row"
    m = copy.deepcopy(model)
    m.zero_grad(set_to_none=True)
    out = m.loss(allb)
    T = out["terms"]
    masked = {k: float(v) for k, v in T.items() if k in ("ckpt", "goal", "lat", "lon", "cons", "acc", "wm", "launch_along", "level", "lead") or k.startswith("pen_") or k.startswith("pens_")}
    nz = {k: v for k, v in masked.items() if v != 0.0}
    assert not nz, f"t2 bank probe: masked terms are not exactly 0.0 on all-bank rows: {nz}"
    assert math.isfinite(float(T["ade"])) and float(T["ade"]) > 0.0, f"t2 bank probe: the target-law term ade is {float(T['ade'])!r}"
    T["total"].backward()
    gsum = float(sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None))
    assert gsum > 0.0, "t2 bank probe: the target-law term has no gradient"
    return {"t2_bank": "on", "pass": True, "masked_terms": masked, "ade": float(T["ade"]), "grad_abs_sum": gsum}


def bad_level(a, model, data_info=None) -> list:
    """PREREG_V8F_LEVEL, G-DVB: ``--level-head`` / ``--w-level`` read back from the BUILT model. Empty = pass. A head declared on that builds nothing, a head built that was not declared, a head with no loss (w_level 0),
    a weight with no head, a head whose code names a non-vision input and an e_level that is not zero at a fresh start are each refused."""
    bad = []
    if not (math.isfinite(a.w_level) and a.w_level >= 0.0):
        bad.append(f"w_level {a.w_level!r} is not a finite number >= 0")
    built = model.built_levers()
    got = built.get("level_head", "off")
    if got != a.level_head:
        bad.append(f"lever level_head: declared {a.level_head} but built {got}")
    if model.cfg.w_level != a.w_level:
        bad.append(f"cfg.w_level: declared {a.w_level} but built {model.cfg.w_level}")
    has_keys = [k for k in model.state_dict() if k.startswith(M.LEVEL_KEY_PREFIXES)]
    if a.level_head == "off":
        if a.w_level != 0.0:
            bad.append(f"w_level {a.w_level} declared with --level-head off: a loss weight with no head -- a lever that is declared and does nothing")
        if model.lvl_head is not None or model.e_level is not None or has_keys:
            bad.append(f"level_head off declared but a level head was built ({len(has_keys)} state-dict keys)")
        return bad
    if not a.w_level > 0.0:
        bad.append(f"level_head on declared but w_level is {a.w_level}: a head with no loss -- a lever that is declared and does nothing")
    if model.lvl_head is None or (a.aux_cond == "add" and model.e_level is None) or not has_keys:           # PREREG_LEVELHEAD_A2: under --aux-cond token the additive e_level is not built (e_aux replaces it)
        bad.append("level_head on declared but no head / e_level was built")
        return bad
    if not built.get("level_vision_only_source", False):
        bad.append(f"the level head's code names non-vision inputs: {M.level_head_source_problems()}")
    ic = (data_info or {}).get("init_ckpt") or {}
    carried = bool(ic.get("applied")) and not any(k.startswith("e_level.") for k in ic.get("fresh_keys", []))      # a lever-ON checkpoint carries its own (trained) e_level
    if model.e_level is not None and not built.get("level_token_zero_init", False) and not carried:
        bad.append("e_level is not zero-initialised at a fresh start: O's step-0 forward would not be the off model's")
    return bad


#: PREREG_LEVELHEAD_A2: a --lead-train file that labels fewer than this share of the train clips is the wrong file (the eval file, a truncated copy): the launch is refused, and the share is recorded in selfcheck.json
LEAD_MIN_CLIP_COVERAGE = 0.5


def bad_lead(a, model, data_info=None) -> list:
    """PREREG_LEVELHEAD_A2, G-DVB: ``--lead-head`` / ``--w-lead`` / ``--aux-cond`` read back from the BUILT model and the BUILT label arrays. Empty = pass. Refused: a head declared on that builds nothing, a head built that was not declared,
    a head with no loss (w_lead 0), a weight with no head, ``--aux-cond token`` with no head on, a wiring that is not the declared one (token: e_aux built and BOTH additive paths absent; add: no e_aux), a head or aux assembly whose code
    names a non-vision input, an e_lead that is not zero at a fresh start, and a lead head with no lead labels (missing file, no labelled frame, no frame with a lead within 40 m, or too few labelled clips)."""
    bad = []
    if not (math.isfinite(a.w_lead) and a.w_lead >= 0.0):
        bad.append(f"w_lead {a.w_lead!r} is not a finite number >= 0")
    built = model.built_levers()
    any_head = a.level_head == "on" or a.lead_head == "on"
    got = built.get("lead_head", "off")
    if got != a.lead_head:
        bad.append(f"lever lead_head: declared {a.lead_head} but built {got}")
    if model.cfg.w_lead != a.w_lead:
        bad.append(f"cfg.w_lead: declared {a.w_lead} but built {model.cfg.w_lead}")
    if model.cfg.aux_cond != a.aux_cond:
        bad.append(f"cfg.aux_cond: declared {a.aux_cond} but built {model.cfg.aux_cond}")
    if a.aux_cond == "token" and not any_head:
        bad.append("--aux-cond token declared with --level-head and --lead-head both off: there is no head output for the token to carry -- a lever that is declared and does nothing")
    if built.get("aux_cond", "add") != a.aux_cond:
        bad.append(f"lever aux_cond: declared {a.aux_cond} but built {built.get('aux_cond', 'add')}")
    lead_keys = [k for k in model.state_dict() if k.startswith(M.LEAD_KEY_PREFIXES)]
    aux_keys = [k for k in model.state_dict() if k.startswith(M.AUX_KEY_PREFIXES)]
    if a.aux_cond == "token" and any_head:
        if model.e_aux is None or not aux_keys:
            bad.append("aux_cond token declared but no aux token (e_aux) was built")
        if model.e_level is not None or model.e_lead is not None:
            bad.append("aux_cond token declared but an additive path (e_level / e_lead) was built: the token REPLACES both additive paths")
        if built.get("aux_token_dim") != M.AUX_TOKEN_DIM:
            bad.append(f"aux token input width: expected {M.AUX_TOKEN_DIM} but built {built.get('aux_token_dim')}")
        if built.get("aux_vision_only_source") is not True:
            bad.append(f"the aux token's assembly names non-vision inputs: {M.aux_source_problems()}")
    elif model.e_aux is not None or aux_keys:
        bad.append(f"aux_cond {a.aux_cond} declared but an aux token was built ({len(aux_keys)} state-dict keys)")
    if a.lead_head == "off":
        if a.w_lead != 0.0:
            bad.append(f"w_lead {a.w_lead} declared with --lead-head off: a loss weight with no head -- a lever that is declared and does nothing")
        if model.lead_head is not None or model.e_lead is not None or lead_keys:
            bad.append(f"lead_head off declared but a lead head was built ({len(lead_keys)} state-dict keys)")
        return bad
    if not a.w_lead > 0.0:
        bad.append(f"lead_head on declared but w_lead is {a.w_lead}: a head with no loss -- a lever that is declared and does nothing")
    if model.lead_head is None or not lead_keys or (a.aux_cond == "add" and model.e_lead is None):
        bad.append("lead_head on declared but no head / e_lead was built")
        return bad
    if not built.get("lead_vision_only_source", False):
        bad.append(f"the lead head's code names non-vision inputs: {M.lead_head_source_problems()}")
    ic = (data_info or {}).get("init_ckpt") or {}
    carried = bool(ic.get("applied")) and not any(k.startswith("e_lead.") for k in ic.get("fresh_keys", []))      # a lead-ON checkpoint carries its own (trained) e_lead
    if model.e_lead is not None and not built.get("lead_token_zero_init", False) and not carried:
        bad.append("e_lead is not zero-initialised at a fresh start: O's step-0 forward would not be the init's")
    ll = (data_info or {}).get("lead_labels")
    if not ll:
        bad.append("lead_head on declared but no lead labels were built (--lead-train)")
    else:
        if ll.get("path") != os.path.abspath(a.lead_train):
            bad.append(f"lead labels: declared {os.path.abspath(a.lead_train)!r} but built from {ll.get('path')!r}")
        if not (isinstance(ll.get("sha256"), str) and len(ll["sha256"]) == 64):
            bad.append(f"lead labels: the file's sha256 is not recorded ({ll.get('sha256')!r})")
        if not (ll.get("n_has", 0) > 0 and ll.get("n_near", 0) > 0):
            bad.append(f"lead labels: {ll.get('n_has')} labelled frames and {ll.get('n_near')} with a lead within {M.LEAD_RANGE_M:.0f} m: the head has nothing to learn from")
        if ll.get("n_clips_with_labels", 0) < LEAD_MIN_CLIP_COVERAGE * max(ll.get("n_clips", 0), 1):
            bad.append(f"lead labels: only {ll.get('n_clips_with_labels')} of {ll.get('n_clips')} train clips carry labels (< {LEAD_MIN_CLIP_COVERAGE:.0%}): the wrong file")
    return bad


def declared_vs_built(a, cfg, model, opt, data_info, enc=None, ego=None, sched=None, dag=None, comp=None, rec=None, t2b=None) -> list:
    """Return the list of mismatches (empty = pass). Each flag is checked against what was BUILT, by FLAG_CHECKS kind."""
    bad = bad_encoder(a, enc, data_info)
    bad += bad_t2bank(a, t2b, data_info)                                                         # PREREG_T2_DAGGER: the four --t2-bank* flags read back from the BUILT swap (and the LOADED bank's sha256)
    bad += bad_recover(a, rec, data_info)                                                        # PREREG_RECOVER_FT: the eight --recover-* flags read back from the BUILT perturber (and the LOADED bank's sha256)
    bad += bad_ego_perturb(a, ego) + bad_init_ckpt(a, data_info) + bad_schedule(a, sched)          # O-RECOVER (PREREG_V8F_HACC_MC section 7); bad_schedule: --warmup read back from the built schedule
    bad += bad_level(a, model, data_info)                                                        # PREREG_V8F_LEVEL: --level-head / --w-level read back from the BUILT model
    bad += bad_wm_only(a, cfg, model, data_info)                                                 # PREREG_V8F_MRATE: --wm-k read back from the BUILT WM modules, --wm-only from the loader's record and the frozen set
    bad += bad_wm_pool(a, cfg, model, data_info)                                                 # PREREG_V8F_WMPOOL: --wm-pool read back from the BUILT pool module, the BUILT dataset and the cache manifest vs the BUILT adapter
    bad += bad_wm_ladder(a, cfg, model, data_info)                                               # PREREG_V8F_LADDER: the seven ladder flags read back from the BUILT loss / predictor shapes / dataset
    bad += bad_wm_strides(a, cfg, model)                                                         # PREREG_V8F_LFS: --wm-strides read back from the BUILT WM modules (and the >= 2 transitions per level rule)
    bad += bad_lead(a, model, data_info)                                                         # PREREG_LEVELHEAD_A2: --lead-head / --w-lead / --aux-cond read back from the BUILT model and the built label arrays
    bad += bad_dagger(a, dag, model.cfg, data_info)                                              # PREREG_V8F_DAGGER: --dagger-p / --dagger-bank / --dagger-seed read back from the BUILT swap
    bad += bad_ego_dagger(a, comp, ego, dag)                                                     # AMEND1 section 1: --ego-perturb-with-dagger read back from the BUILT stage
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
    # PREREG_V8F_FLAT: the interface declared on the argv is the one the BUILT model's levers say (argv -> make_config -> cfg -> built_levers); o_interface_probe then reads what O is actually handed
    if built.get("o_interface") != a.o_interface:
        bad.append(f"lever o_interface: declared {a.o_interface} but built {built.get('o_interface')}")
    # PREREG_V8F_OACC: the acceleration weight declared on the argv is the one the BUILT config and the BUILT model's levers say (argv -> make_config -> cfg -> built_levers); a NEGATIVE or non-finite weight is
    # refused (a negative weight would REWARD roughness); acc_probe then reads that the BUILT loss's total actually moves by w_acc x the term
    if not (math.isfinite(a.w_acc) and a.w_acc >= 0.0):
        bad.append(f"w_acc {a.w_acc!r} is not a finite number >= 0")
    if built.get("w_acc") != a.w_acc:
        bad.append(f"lever w_acc: declared {a.w_acc} but built {built.get('w_acc')}")
    # PREREG_V8F_LAUNCH_INTENT: the launch weight declared on the argv is the one the BUILT config and the BUILT model's levers say (argv -> make_config -> cfg -> built_levers); a NEGATIVE or non-finite weight is refused
    # (a negative weight would REWARD a launch-sample along-track error); a weight that the position-loss scale zeroes (w_traj 0) is a lever that is declared and does nothing. launch_intent_probe then reads the BUILT loss.
    if not (math.isfinite(a.launch_intent_w) and a.launch_intent_w >= 0.0):
        bad.append(f"launch_intent_w {a.launch_intent_w!r} is not a finite number >= 0")
    if built.get("launch_intent_w") != a.launch_intent_w:
        bad.append(f"lever launch_intent_w: declared {a.launch_intent_w} but built {built.get('launch_intent_w')}")
    if a.launch_intent_w > 0.0 and not model.cfg.w_traj > 0.0:
        bad.append(f"launch_intent_w {a.launch_intent_w} declared but w_traj is {model.cfg.w_traj}: the term enters the total scaled by w_traj -- a lever that is declared and does nothing")
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
              "w_pen_self", "w_wm", "detach_upper", "kin_mode", "w_acc", "launch_intent_w"):
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
    expect = {"ckpt", "goal", "lat", "lon", "cons", "ade", "fde", "acc", "total"}      # + acc: PREREG_V8F_OACC (always logged)
    if cfg.consistency:
        expect |= {f"pen_{r}" for r in M.RULES} | {f"pens_{r}" for r in M.RULES}
    if cfg.wm != "off":
        expect.add("wm")
    if cfg.level_head == "on":
        expect.add("level")                                              # PREREG_V8F_LEVEL
    if cfg.lead_head == "on":
        expect.add("lead")                                               # PREREG_LEVELHEAD_A2
    missing = sorted(expect - set(terms_seen))
    extra_pen = sorted(k for k in terms_seen if k.startswith("pen") and not cfg.consistency)
    declared_dead = [n for n in acc if cfg.wm == "a" and ".idm_heads." in n]
    if cfg.o_interface == "cut":                  # PREREG_V8F_FLAT: O reads a constant code, so e_code.weight receives EXACTLY 0.0 (its bias still trains): declared dead, and G-LIVE REQUIRES the zero
        declared_dead += [n for n in acc if n == "e_code.weight"]
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


def o_interface_probe(model, batch, cfg, shift: float = 2.5) -> dict:
    """G-LIVE for PREREG_V8F_FLAT: what the BUILT model's O is actually HANDED, read from the tensors ``run_o`` receives in a forward of a COPY on the real probe batch (no gradient, its own
    generators, the residual head re-initialised non-zero on the copy: it is zero-init, so at build O's output is P10 whatever it reads and no wire could be seen).

    Perturb everything above O: T's code (a donor window's code, the GT code, a random code: ``code_override``, the evaluator's own route) and S's checkpoints (every one shifted by ``shift``).

      cut   O is handed the constants (code = zeros, checkpoints = ck_nom, ok = 0) under EVERY perturbation, and traj60 is bit-for-bit the unperturbed one (max |d| = 0.0)
      full  O is handed the perturbed values, and traj60 MOVES under every perturbation: the control that proves this probe can see a change

    Two more controls hold under both: the override reached the interface (out["code"] is the override) and T still reads S (T's goal logits move with S's checkpoints). A violation is an
    AssertionError, which the caller turns into a refusal. The probe reads the tensor the trainer's model hands the O layer, so it proves reachability FROM THE ACTUAL CALLER."""
    m = copy.deepcopy(model).cpu()
    m.eval()
    B = int(batch["feats"].shape[0])
    if B < 2:
        if cfg.o_interface == "full":      # the DEFAULT interface is every earlier arm (a smoke run's probe batch can hold one window): nothing to prove there, and the run must not be refused
            return {"o_interface": cfg.o_interface, "shift": shift, "n_windows": B, "perturbations": {}, "pass": True,
                    "skipped": "the probe batch holds a single window (a donor code cannot differ from its own); the full interface is the default and the pre-change model"}
        raise AssertionError("o_interface probe: the probe batch has a single window: a donor code cannot be told from its own")
    g = torch.Generator().manual_seed(2718)
    with torch.no_grad():
        w = m.head_o[2].weight
        w.copy_(torch.randn(w.shape, generator=g) * 0.05)
    cap = {}
    run_o0 = m.run_o

    def run_o(b, adapt, last_o, ck_o, ck_ok, code):
        cap["ck_o"], cap["ck_ok"], cap["code"] = ck_o.detach().clone(), ck_ok.detach().clone(), code.detach().clone()
        return run_o0(b, adapt, last_o, ck_o, ck_ok, code)
    m.run_o = run_o

    def fwd(override=None, shifted=False):
        if shifted:
            run_s0 = type(m).run_s.__get__(m)
            m.run_s = lambda *a, **k: run_s0(*a, **k) + shift
        try:
            with torch.no_grad():
                out = m._forward(batch, 0.0, False, None, override)
        finally:
            m.__dict__.pop("run_s", None)
        return out, dict(cap)
    base, base_cap = fwd()
    own = base["code"].detach().clone()
    perts = {"donor_code": (own.roll(1, 0), False), "gt_code": (m._code_gt(batch["lab_t"]).to(own.dtype), False),
             "random_code": (torch.rand(own.shape, generator=g).to(own.dtype), False), "shifted_checkpoints": (None, True)}
    ck_nom = m.ck_nom[None].expand(B, -1, -1)
    res = {"o_interface": cfg.o_interface, "shift": shift, "n_windows": B, "perturbations": {}}
    for name, (ov, sh) in perts.items():
        out, c = fwd(ov, sh)
        d_traj = float((out["traj60"] - base["traj60"]).abs().max())
        row = {"max_abs_dtraj60": d_traj, "o_read_code_abs_max": float(c["code"].abs().max()), "o_read_ck_minus_nominal_abs_max": float((c["ck_o"] - ck_nom).abs().max()),
               "o_read_ck_ok_abs_max": float(c["ck_ok"].abs().max())}
        if ov is not None and not torch.equal(out["code"], ov.to(out["code"].dtype)):
            raise AssertionError(f"o_interface probe: the {name} override did not reach the interface (out['code'] is not the override): the perturbation is not applied, the probe is blind")
        if sh:
            row["max_abs_d_ckpt"] = float((out["ckpt"] - base["ckpt"]).abs().max())
            row["t_reads_s_max_abs_dgoal_logits"] = float((out["goal_logits"] - base["goal_logits"]).abs().max())
            if row["max_abs_d_ckpt"] == 0.0:
                raise AssertionError("o_interface probe: shifting S's output did not move S's checkpoints: the perturbation is not applied, the probe is blind")
            if not row["t_reads_s_max_abs_dgoal_logits"] > 0.0:
                raise AssertionError("o_interface probe: T's output did not move with S's checkpoints: T does not read S (the interface cut must leave S -> T intact)")
        if cfg.o_interface == "cut":
            if d_traj != 0.0:
                raise AssertionError(f"o_interface probe: cut is declared but traj60 moved by {d_traj} under {name}: O still reads S or T")
            if row["o_read_code_abs_max"] != 0.0 or row["o_read_ck_minus_nominal_abs_max"] != 0.0 or row["o_read_ck_ok_abs_max"] != 0.0:
                raise AssertionError(f"o_interface probe: cut is declared but O was handed a non-constant under {name}: {row}")
        else:
            if not d_traj > 0.0:
                raise AssertionError(f"o_interface probe: full is declared but traj60 did not move under {name}: O does not read what it is declared to read, or the probe cannot see a change")
        res["perturbations"][name] = row
    res["pass"] = True
    return res


def acc_probe(model, batch, cfg, rtol: float = 1e-3) -> dict:
    """G-LIVE for PREREG_V8F_OACC: the acceleration term REACHES THE LOSS THE TRAINER BACKPROPAGATES, with the declared weight, and O's head receives its gradient -- read from the BUILT model's own ``loss`` on the
    real probe batch (a COPY, its own seeds), not from a unit test of the term. The loss is evaluated at the probe weight ``w`` (the declared one; 1.0 when 0.0 is declared, so a B-equivalent run still proves
    the wire) and at 0.0, on the same batch, generator and global seed:

      acc            finite and > 0 (an exactly-zero term would be a plan with the GT's own acceleration), and independent of the weight
      contribution   total(w) - total(0) == w x acc  (to ``rtol`` and float32 rounding): the weight reaches the total. DELETE the line that adds the term and this reads 0.0 -> RED
      O's head       sum |grad(total(w)) - grad(total(0))| over ``head_o.*`` > 0: the gradient the TERM gives O's head
      controls       (a) the same loss twice at w = 0.0 gives gradients that differ by EXACTLY 0.0 (the probe is deterministic, so a difference above is the term's);
                     (b) the frozen P10 prior's gradient difference is EXACTLY 0.0; (c) so is the world-model auxiliary's (it does not feed the plan), when built
      declared 0.0   the contribution at the DECLARED weight is EXACTLY 0.0 (B's total, untouched)

    Returns the record that selfcheck.json carries; any violation is an AssertionError, which the caller turns into a refusal."""
    eps32 = float(torch.finfo(torch.float32).eps)
    w_decl = float(cfg.w_acc)
    w = w_decl if w_decl > 0.0 else 1.0
    m = copy.deepcopy(model).cpu()
    m.train()
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]

    def run(w_):
        m.cfg.w_acc = w_
        for _, p in named:
            p.grad = None
        for p in m.prior.parameters():
            p.grad = None
        torch.manual_seed(20261025)
        out = m.loss(batch, p_gt=0.5, gen=torch.Generator().manual_seed(11))
        T = out["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"acc probe: loss term {k} is not finite at w_acc {w_}: {float(v)}")
        T["total"].backward()
        g = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
        g["__prior__"] = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in m.prior.parameters()])
        return float(T["total"].detach()), float(T["acc"].detach()), g

    t_w, acc_w, g_w = run(w)
    t_0, acc_0, g_0 = run(0.0)
    t_0b, _, g_0b = run(0.0)
    dsum = lambda ga, gb, pre: float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))      # noqa: E731
    contribution = t_w - t_0
    want = w * acc_w
    res = {"w_acc_declared": w_decl, "w_probe": w, "acc": acc_w, "acc_at_zero_weight": acc_0, "total_w": t_w, "total_0": t_0, "contribution": contribution, "w_x_acc": want,
           "head_o_grad_diff": dsum(g_w, g_0, "head_o."), "read_o_grad_diff": dsum(g_w, g_0, "read_o."),
           "control_repeat_total_diff": abs(t_0b - t_0), "control_repeat_grad_diff": dsum(g_0, g_0b, ""),
           "control_prior_grad_diff": float((g_w["__prior__"] - g_0["__prior__"]).abs().sum()),
           "control_wm_grad_diff": dsum(g_w, g_0, "wm.") if m.wm is not None else None}
    if not math.isfinite(acc_w) or not acc_w > 0.0:
        raise AssertionError(f"acc probe: the acceleration term reads {acc_w} on the probe batch (finite and > 0 required)")
    if acc_w != acc_0:
        raise AssertionError(f"acc probe: the logged acceleration term depends on the weight ({acc_w} at {w}, {acc_0} at 0): it is not the term")
    if res["control_repeat_total_diff"] != 0.0 or res["control_repeat_grad_diff"] != 0.0:
        raise AssertionError(f"acc probe: the probe is not deterministic (the same loss twice differs by {res['control_repeat_total_diff']} / {res['control_repeat_grad_diff']}): its differences mean nothing")
    if not want > 64.0 * eps32 * max(abs(t_0), 1.0):
        raise AssertionError(f"acc probe: w x acc = {want} is below float32 visibility against the total {t_0}: the probe cannot see the term; use a larger probe weight")
    if abs(contribution - want) > rtol * want + 4.0 * eps32 * max(abs(t_w), abs(t_0)):
        raise AssertionError(f"acc probe: the total moved by {contribution} when the weight went from 0 to {w}, the term is {acc_w} (w x acc = {want}): the term does not reach the loss with its weight")
    if not res["head_o_grad_diff"] > 0.0:
        raise AssertionError("acc probe: O's head receives no gradient from the acceleration term (the sum of |d total / d head_o| differences is 0.0)")
    if res["control_prior_grad_diff"] != 0.0:
        raise AssertionError(f"acc probe: the frozen P10 prior's gradient moved with the weight ({res['control_prior_grad_diff']}): the control must read exactly 0.0")
    if res["control_wm_grad_diff"] not in (None, 0.0):
        raise AssertionError(f"acc probe: the world-model auxiliary's gradient moved with the weight ({res['control_wm_grad_diff']}): it does not feed the plan, the control must read exactly 0.0")
    if w_decl == 0.0:                                 # the DECLARED weight is 0: the total there is B's, and it reproduces exactly (the repeat control above: 0.0 difference)
        res["declared_zero_total_reproduces_exactly"] = bool(t_0b == t_0)
    res["pass"] = True
    return res


def launch_stage(batch: dict) -> torch.Tensor:
    """PREREG_V8F_LAUNCH_INTENT: the ONE call that turns a COLLATED batch into its GT-launch mask ``[B]`` bool -- the training loop calls it, and ``launch_intent_probe`` (G-LIVE) calls the same function. It reads the LOGGED
    ``kin3[:, 0]`` (v0) and the GT future (``gt``, ``gt_ok``) and must run BEFORE ``ego.apply`` / ``dag.apply`` / ``comp.apply``: a swapped or perturbed sample's ``kin3`` is the OWN or perturbed state, not the label."""
    return M.launch_mask(batch["kin3"][:, 0], batch["gt"], batch["gt_ok"])


def launch_weight_share(n_launch: int, n_eligible: int, w: float) -> float | None:
    """PREREG_V8F_LAUNCH_INTENT K-SHARE: the realised share of the along-track weight that sits on launch samples, from the run record's counts: launch samples carry 1 + W, the other GT-valid samples 1.
    ``(1 + W) n / ((1 + W) n + (eligible - n))``. None when no GT-valid row was seen."""
    if n_eligible <= 0:
        return None
    return ((1.0 + w) * n_launch) / ((1.0 + w) * n_launch + (n_eligible - n_launch))


def launch_intent_probe(model, batch, cfg, rtol: float = 1e-3) -> dict:
    """G-LIVE for PREREG_V8F_LAUNCH_INTENT: the up-weighted along-track term REACHES THE LOSS THE TRAINER BACKPROPAGATES, with the declared weight, ON THE ROWS THE MASK NAMES, and O's head receives its gradient -- read from the BUILT
    model's own ``loss`` on the real probe batch (a COPY, its own seeds). The mask is ``launch_stage(batch)`` (the loop's own call). A probe batch can hold no natural launch row (about 1 in 20 windows is one), so the wire is
    proved on a FORCED mask (the first half of the GT-valid rows) and the natural mask is read beside it (its counts are the record's ``launch_n`` / ``launch_eligible``). With W = 0.0 declared nothing runs:
    {"launch_intent": "off"} (the default run is the pre-change run).

      term           finite and > 0 on the forced mask, and absent at W = 0.0
      contribution   total(W) - total(0) == w_traj x W x term / 3 (to ``rtol`` and float32 rounding). DELETE the line that adds the term and this reads 0.0 -> RED
      O's head       sum |grad(total(W)) - grad(total(0))| over ``head_o.*`` > 0
      natural mask   when it holds >= 1 row, its contribution is > 0 too (the real labels reach the loss)
      controls       (a) the same loss twice at W = 0.0 gives gradients that differ by EXACTLY 0.0; (b) an ALL-FALSE mask at W gives term 0.0, total difference 0.0 and gradient difference EXACTLY 0.0 (a mask that is ignored reads
                     non-zero); (c) the frozen P10 prior's gradient difference is EXACTLY 0.0; (d) so is the world-model auxiliary's, when built

    Returns the record selfcheck.json carries; any violation is an AssertionError, which the caller turns into a refusal."""
    W = float(cfg.launch_intent_w)
    if not W > 0.0:
        return {"launch_intent": "off", "pass": True, "launch_n": 0, "launch_eligible": 0, "skipped": "--launch-intent-w not named: the term is not in the loss (the default run is the pre-change run)"}
    eps32 = float(torch.finfo(torch.float32).eps)
    nat = launch_stage(batch)
    gok = batch["gt_ok"].bool()
    if nat.dtype != torch.bool or tuple(nat.shape) != (int(gok.shape[0]),):
        raise AssertionError(f"launch probe: the mask is {nat.dtype} {tuple(nat.shape)}, a bool [B] is required")
    elig = int(gok.sum())
    if elig < 2:
        raise AssertionError(f"launch probe: only {elig} GT-valid rows in the probe batch; the wire cannot be seen")
    forced = torch.zeros_like(nat)
    forced[gok.nonzero().flatten()[: max(1, elig // 2)]] = True
    none = torch.zeros_like(nat)
    m = copy.deepcopy(model).cpu()
    m.train()
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]

    def run(w_, mask):
        m.cfg.launch_intent_w = w_
        for _, p in named:
            p.grad = None
        for p in m.prior.parameters():
            p.grad = None
        torch.manual_seed(20261028)
        out = m.loss({**batch, "launch": mask}, p_gt=0.5, gen=torch.Generator().manual_seed(11))
        T = out["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"launch probe: loss term {k} is not finite at launch_intent_w {w_}: {float(v)}")
        T["total"].backward()
        g = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
        g["__prior__"] = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in m.prior.parameters()])
        return float(T["total"].detach()), (float(T["launch_along"].detach()) if "launch_along" in T else None), g

    t_w, term_w, g_w = run(W, forced)
    t_0, term_0, g_0 = run(0.0, forced)
    t_0b, _, g_0b = run(0.0, forced)
    t_n, term_n, g_n = run(W, none)
    t_nat, term_nat, g_nat = run(W, nat)
    dsum = lambda ga, gb, pre: float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))      # noqa: E731
    w_traj = float(m.cfg.w_traj)
    want = w_traj * W * (term_w or 0.0) / 3.0
    res = {"launch_intent_w_declared": W, "launch_n": int(nat.sum()), "launch_eligible": elig, "launch_n_forced": int(forced.sum()), "term_forced": term_w, "term_at_zero_weight": term_0,
           "total_w": t_w, "total_0": t_0, "contribution": t_w - t_0, "w_traj_x_W_x_term_over_3": want,
           "head_o_grad_diff": dsum(g_w, g_0, "head_o."), "read_o_grad_diff": dsum(g_w, g_0, "read_o."),
           "natural_contribution": t_nat - t_0, "natural_term": term_nat, "natural_head_o_grad_diff": dsum(g_nat, g_0, "head_o."),
           "control_repeat_total_diff": abs(t_0b - t_0), "control_repeat_grad_diff": dsum(g_0, g_0b, ""),
           "control_allfalse_term": term_n, "control_allfalse_total_diff": abs(t_n - t_0), "control_allfalse_grad_diff": dsum(g_n, g_0, ""),
           "control_prior_grad_diff": float((g_w["__prior__"] - g_0["__prior__"]).abs().sum()),
           "control_wm_grad_diff": dsum(g_w, g_0, "wm.") if m.wm is not None else None}
    if term_w is None or not math.isfinite(term_w) or not term_w > 0.0:
        raise AssertionError(f"launch probe: the up-weighted term reads {term_w} on the forced mask (finite and > 0 required)")
    if term_0 is not None:
        raise AssertionError("launch probe: the term is computed at launch_intent_w 0.0 (a declared-off lever must add no term)")
    if res["control_repeat_total_diff"] != 0.0 or res["control_repeat_grad_diff"] != 0.0:
        raise AssertionError(f"launch probe: the probe is not deterministic (the same loss twice differs by {res['control_repeat_total_diff']} / {res['control_repeat_grad_diff']}): its differences mean nothing")
    if not want > 64.0 * eps32 * max(abs(t_0), 1.0):
        raise AssertionError(f"launch probe: w_traj x W x term / 3 = {want} is below float32 visibility against the total {t_0}: the probe cannot see the term")
    if abs(res["contribution"] - want) > rtol * want + 4.0 * eps32 * max(abs(t_w), abs(t_0)):
        raise AssertionError(f"launch probe: the total moved by {res['contribution']} when the weight went from 0 to {W}, w_traj x W x term / 3 is {want}: the term does not reach the loss with its weight")
    if not res["head_o_grad_diff"] > 0.0:
        raise AssertionError("launch probe: O's head receives no gradient from the launch term (the sum of |d total / d head_o| differences is 0.0)")
    if term_n != 0.0 or res["control_allfalse_total_diff"] != 0.0 or res["control_allfalse_grad_diff"] != 0.0:
        raise AssertionError(f"launch probe: an ALL-FALSE mask must read exactly 0.0 (term {term_n}, total diff {res['control_allfalse_total_diff']}, grad diff {res['control_allfalse_grad_diff']}): the mask is not what selects the rows")
    if res["control_prior_grad_diff"] != 0.0:
        raise AssertionError(f"launch probe: the frozen P10 prior's gradient moved with the weight ({res['control_prior_grad_diff']}): the control must read exactly 0.0")
    if res["control_wm_grad_diff"] not in (None, 0.0):
        raise AssertionError(f"launch probe: the world-model auxiliary's gradient moved with the weight ({res['control_wm_grad_diff']}): it does not feed the plan, the control must read exactly 0.0")
    if res["launch_n"] > 0 and not (res["natural_contribution"] > 0.0 and res["natural_head_o_grad_diff"] > 0.0):
        raise AssertionError(f"launch probe: the natural mask names {res['launch_n']} rows but the total moved by {res['natural_contribution']} (head_o grad diff {res['natural_head_o_grad_diff']}): the real labels do not reach the loss")
    res["pass"] = True
    return res

def level_stage(batch: dict) -> tuple:
    """PREREG_V8F_LEVEL: the ONE call that turns a COLLATED batch into the level head's label ``(level_y [B] float32, level_ok [B] bool)`` -- the LOGGED speed at NOW, ``kin3[:, 0]``. The training loop calls it, and
    ``level_head_probe`` / the main() probe batch call the same function. It must run BEFORE ``ego.apply`` / ``dag.apply`` / ``comp.apply`` / ``rcv.apply`` (a swapped or perturbed sample's ``kin3`` is the own or perturbed
    state, not the label: the head sees the LOGGED frame and is asked for the LOGGED speed). Labels may use ego; the head's input is vision only."""
    y = batch["kin3"][:, 0].float()
    return y, torch.isfinite(y)


def make_lead_labels(a, cfg, J, info):
    """PREREG_LEVELHEAD_A2: the lead head's label arrays, row-aligned to the label join (``D.lead_label_arrays``), or None when the head is off (nothing is read). A missing ``--lead-train`` file is a REFUSAL (the
    default path is a dev-box path: a lead head launched elsewhere must not train on nothing). Records the file's sha256 and the coverage in ``info["lead_labels"]`` for the gate and selfcheck.json."""
    if cfg.lead_head != "on":
        return None
    if not os.path.exists(a.lead_train):
        raise SystemExit(f"ZZDVB-REFUSEDZZ lead-head: --lead-head on needs the lead labels and --lead-train {a.lead_train!r} does not exist")
    LL = D.lead_label_arrays(J, a.lead_train)
    h = hashlib.sha256()
    with open(a.lead_train, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    info["lead_labels"] = {"path": os.path.abspath(a.lead_train), "sha256": h.hexdigest(), **{k: LL[k] for k in ("n_rows", "n_has", "n_near", "n_clips", "n_clips_with_labels")}}
    return LL


def lead_stage(batch: dict, LL: dict) -> tuple:
    """PREREG_LEVELHEAD_A2: the ONE call that turns a COLLATED batch into the lead head's label ``(lead_y [B] float32, lead_has [B] bool, lead_near [B] bool)`` -- looked up by the sample's clip index and cache step
    (``batch["ci"]``, ``batch["t"]``: row ``off[ci] + t`` of ``D.lead_label_arrays``, i.e. the lead file's raw frame ``k = 2 t``). The training loop calls it, and the main() probe batch / ``lead_head_probe`` call the same function. It must
    run BEFORE ``ego.apply`` / ``dag.apply`` / ``comp.apply`` / ``rcv.apply`` (a swapped sample's ``ci`` / ``t`` is the own one, not the label's: the head sees the LOGGED frame and is asked for the LOGGED lead). Labels may use
    anything; the head's input is vision only."""
    rows = LL["off"][batch["ci"].cpu().numpy()] + batch["t"].cpu().numpy()
    return torch.as_tensor(LL["y"][rows]), torch.as_tensor(LL["has"][rows]), torch.as_tensor(LL["near"][rows])


def lead_head_probe(model, batch, cfg, rtol: float = 1e-3) -> dict:
    """G-LIVE for PREREG_LEVELHEAD_A2, read from the BUILT model's own ``loss`` on the real probe batch (a COPY; the zero-init e_lead and O's zero-init head are woken on the copy so every path carries signal). With the head
    off nothing runs: {"lead": "off"}.

      vision only    ``M.assert_lead_head_vision_only`` (and ``M.assert_aux_vision_only`` under --aux-cond token): the source names no ego / nav / limit / code / label input, the head reads only frame_ok, its outputs are bit-identical when every
                     non-vision key changes and move with the features
      labels         the probe batch has frames with a lead label AND frames with a lead within 40 m (else the gap term is invisible)
      term           finite and > 0
      contribution   total(w_lead) - total(0) == w_lead x term (to ``rtol`` and float32 rounding). DELETE the total line and this reads 0.0 -> RED
      head gradient  sum |grad(total(w)) - grad(total(0))| over ``lead_head.*`` > 0, AND over each of the two output rows of ``lead_head.2.weight`` (presence, gap) > 0: a term that reaches one output only is RED
      detach control at w_lead = 0 the head's gradient is EXACTLY 0.0 (O reads the outputs detached: the plan's loss never reaches the head). Remove the detach and this reads > 0 -> RED
      token reaches  at w_lead = 0 e_lead (add) / the lead columns of e_aux (token) receive a non-zero gradient, and zeroing them changes O's plan (the head's output is read)
      controls       the same loss twice at w gives gradients that differ by EXACTLY 0.0; an ALL-FALSE lead_has gives term 0.0, total difference 0.0 and head gradient difference EXACTLY 0.0; the frozen P10 prior's gradient difference is EXACTLY 0.0

    ``w_lead_for_10pct_share`` = (0.1 / 0.9) x (total without the lead term) / (lead term) at STEP 0 (the formula of the level probe; the rest includes the level term when that head is on). Returns the record selfcheck.json carries;
    any violation is an AssertionError, which the caller turns into a refusal."""
    if cfg.lead_head != "on":
        return {"lead": "off", "pass": True, "skipped": "--lead-head not named: nothing is built (the default run is the pre-change run)"}
    if not all(k in batch for k in ("lead_y", "lead_has", "lead_near")):
        raise AssertionError("lead probe: the probe batch carries no lead_y / lead_has / lead_near (the loop's own call is lead_stage)")
    W = float(cfg.w_lead)
    eps32 = float(torch.finfo(torch.float32).eps)
    m = copy.deepcopy(model).cpu()
    m.train()
    vis = M.assert_lead_head_vision_only(m, batch)
    aux_vis = M.assert_aux_vision_only(m, batch) if m.e_aux is not None else None
    fok = batch["frame_ok"][:, -1]
    n_has, n_near = int((batch["lead_has"] & fok).sum()), int((batch["lead_near"] & batch["lead_has"] & fok).sum())
    if n_has == 0 or n_near == 0:
        raise AssertionError(f"lead probe: the probe batch has {n_has} frames with a lead label and {n_near} with a lead within {M.LEAD_RANGE_M:.0f} m: the probe cannot see the lead term (use a larger probe batch or the real corpus)")
    with torch.no_grad():                    # the number the arm's PREREG needs: W such that the lead term is ~10 % of the total at STEP 0 (the copy before it is woken = the model as launched)
        torch.manual_seed(20261105)
        o0 = m.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    T0 = o0["terms"]
    t0_total, t0_term = float(T0["total"]), float(T0["lead"])
    step0 = {"step0_total": t0_total, "step0_lead_term": t0_term, "step0_lead_bce": float(T0["lead_bce"]), "step0_lead_huber": float(T0["lead_huber"]), "step0_total_without_lead": t0_total - W * t0_term,
             "step0_lead_share": W * t0_term / t0_total, "w_lead_for_10pct_share": ((0.1 / 0.9) * (t0_total - W * t0_term) / t0_term) if t0_term > 0.0 else None}
    if "level" in T0:
        step0["step0_level_share"] = float(m.cfg.w_level) * float(T0["level"]) / t0_total
    g = torch.Generator().manual_seed(17)
    with torch.no_grad():
        if m.e_lead is not None:
            m.e_lead.weight.copy_(torch.randn(m.e_lead.weight.shape, generator=g) * 0.5)
            m.e_lead.bias.copy_(torch.randn(m.e_lead.bias.shape, generator=g) * 0.5)
        wo = m.head_o[2].weight
        wo.copy_(torch.randn(wo.shape, generator=g) * 0.05)
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    none = torch.zeros_like(batch["lead_has"])

    def run(w_, has_mask=None):
        m.cfg.w_lead = w_
        for _, p in named:
            p.grad = None
        for p in m.prior.parameters():
            p.grad = None
        torch.manual_seed(20261105)
        out = m.loss({**batch, "lead_has": batch["lead_has"] if has_mask is None else has_mask}, p_gt=0.5, gen=torch.Generator().manual_seed(11))
        T = out["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"lead probe: loss term {k} is not finite at w_lead {w_}: {float(v)}")
        T["total"].backward()
        gr = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
        gr["__prior__"] = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in m.prior.parameters()])
        return float(T["total"].detach()), float(T["lead"].detach()), gr, out["traj60"].detach().clone()

    t_w, term_w, g_w, tr_w = run(W)
    t_0, term_0, g_0, tr_0 = run(0.0)
    t_wb, _, g_wb, _ = run(W)
    t_n, term_n, g_n, _ = run(W, none)
    dsum = lambda ga, gb, pre: float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))      # noqa: E731
    asum = lambda ga, pre: float(sum(ga[n].abs().sum() for n in ga if n.startswith(pre)))                       # noqa: E731
    row = lambda i: float((g_w["lead_head.2.weight"][i] - g_0["lead_head.2.weight"][i]).abs().sum())          # noqa: E731
    if m.e_lead is not None:                                       # --aux-cond add: zero e_lead and read O's plan
        reach_grad = {"e_lead_grad_at_w0": asum(g_0, "e_lead.")}
        saved = (m.e_lead.weight.detach().clone(), m.e_lead.bias.detach().clone())
        with torch.no_grad():
            m.e_lead.weight.zero_()
            m.e_lead.bias.zero_()
        _, _, _, tr_z = run(0.0)
        with torch.no_grad():
            m.e_lead.weight.copy_(saved[0])
            m.e_lead.bias.copy_(saved[1])
    else:                                                          # --aux-cond token: zero the lead columns (1, 2) of e_aux's weight and read O's plan
        reach_grad = {"e_aux_lead_cols_grad_at_w0": float(g_0["e_aux.weight"][:, 1:].abs().sum())}
        saved = m.e_aux.weight.detach().clone()
        with torch.no_grad():
            m.e_aux.weight[:, 1:].zero_()
        _, _, _, tr_z = run(0.0)
        with torch.no_grad():
            m.e_aux.weight.copy_(saved)
    res = {"w_lead_declared": W, "aux_cond": cfg.aux_cond, "vision_only": vis, "aux_vision_only": aux_vis, "probe_n_has": n_has, "probe_n_near": n_near, **step0, "term": term_w, "total_w": t_w, "total_0": t_0,
           "contribution": t_w - t_0, "w_lead_x_term": W * term_w,
           "head_grad_diff": dsum(g_w, g_0, "lead_head."), "head_grad_diff_presence_row": row(0), "head_grad_diff_gap_row": row(1), "head_grad_at_w0_detach_control": asum(g_0, "lead_head."), **reach_grad,
           "plan_moves_with_token": float((tr_0 - tr_z).abs().max()),
           "control_repeat_total_diff": abs(t_wb - t_w), "control_repeat_grad_diff": dsum(g_w, g_wb, ""),
           "control_allfalse_term": term_n, "control_allfalse_total_diff": abs(t_n - t_0), "control_allfalse_head_grad_diff": dsum(g_n, g_0, "lead_head."),
           "control_prior_grad_diff": float((g_w["__prior__"] - g_0["__prior__"]).abs().sum())}
    if not math.isfinite(term_w) or not term_w > 0.0:
        raise AssertionError(f"lead probe: the lead term reads {term_w} (finite and > 0 required)")
    if res["control_repeat_total_diff"] != 0.0 or res["control_repeat_grad_diff"] != 0.0:
        raise AssertionError(f"lead probe: the probe is not deterministic (the same loss twice differs by {res['control_repeat_total_diff']} / {res['control_repeat_grad_diff']}): its differences mean nothing")
    if not res["w_lead_x_term"] > 64.0 * eps32 * max(abs(t_0), 1.0):
        raise AssertionError(f"lead probe: w_lead x term = {res['w_lead_x_term']} is below float32 visibility against the total {t_0}: the probe cannot see the term")
    if abs(res["contribution"] - res["w_lead_x_term"]) > rtol * res["w_lead_x_term"] + 4.0 * eps32 * max(abs(t_w), abs(t_0)):
        raise AssertionError(f"lead probe: the total moved by {res['contribution']} when the weight went from 0 to {W}, w_lead x term is {res['w_lead_x_term']}: the term does not reach the loss with its weight")
    if not res["head_grad_diff"] > 0.0:
        raise AssertionError("lead probe: the lead head receives no gradient from the lead term (the sum of |d total / d lead_head| differences is 0.0)")
    if not (res["head_grad_diff_presence_row"] > 0.0 and res["head_grad_diff_gap_row"] > 0.0):
        raise AssertionError(f"lead probe: the lead term reaches one output of the head only (presence row {res['head_grad_diff_presence_row']}, gap row {res['head_grad_diff_gap_row']}): both the BCE and the Huber must reach it")
    if res["head_grad_at_w0_detach_control"] != 0.0:
        raise AssertionError(f"lead probe: at w_lead 0 the head's gradient is {res['head_grad_at_w0_detach_control']}, not EXACTLY 0.0: the plan's loss reaches the head (O must read the head's outputs detached)")
    if not list(reach_grad.values())[0] > 0.0 or not res["plan_moves_with_token"] > 0.0:
        raise AssertionError(f"lead probe: the lead output does not reach O ({reach_grad}, plan moves by {res['plan_moves_with_token']} when the lead path is zeroed)")
    if term_n != 0.0 or res["control_allfalse_total_diff"] != 0.0 or res["control_allfalse_head_grad_diff"] != 0.0:
        raise AssertionError(f"lead probe: an ALL-FALSE lead_has must read exactly 0.0 (term {term_n}, total diff {res['control_allfalse_total_diff']}, head grad diff {res['control_allfalse_head_grad_diff']}): the mask is not what selects the rows")
    if res["control_prior_grad_diff"] != 0.0:
        raise AssertionError(f"lead probe: the frozen P10 prior's gradient moved with the weight ({res['control_prior_grad_diff']}): the control must read exactly 0.0")
    res["pass"] = True
    return res


def level_head_probe(model, batch, cfg, rtol: float = 1e-3) -> dict:
    """G-LIVE for PREREG_V8F_LEVEL, read from the BUILT model's own ``loss`` on the real probe batch (a COPY; the zero-init e_level and O's zero-init head are woken on the copy so every path carries signal). With the head off
    nothing runs: {"level": "off"} (the default run is the pre-change run).

      vision only    ``M.assert_level_head_vision_only``: the source names no ego / nav / limit / code / label input, ``_level`` reads only frame_ok, v_hat is bit-identical when every non-vision key changes and moves with the features
      term           finite and > 0
      contribution   total(w_level) - total(0) == w_level x term (to ``rtol`` and float32 rounding). DELETE the total line and this reads 0.0 -> RED
      head gradient  sum |grad(total(w)) - grad(total(0))| over ``lvl_head.*`` > 0
      detach control at w_level = 0 the head's gradient is EXACTLY 0.0 (O reads v_hat.detach(): the plan's loss never reaches the head). Remove the detach and this reads > 0 -> RED
      token reaches  at w_level = 0 e_level receives a non-zero gradient, and zeroing e_level changes O's plan (the token is read)
      controls       the same loss twice at w gives gradients that differ by EXACTLY 0.0; an ALL-FALSE level_ok gives term 0.0, total difference 0.0 and head gradient difference EXACTLY 0.0; the frozen P10 prior's gradient difference is EXACTLY 0.0

    Returns the record selfcheck.json carries; any violation is an AssertionError, which the caller turns into a refusal."""
    if cfg.level_head != "on":
        return {"level": "off", "pass": True, "skipped": "--level-head not named: nothing is built (the default run is the pre-change run)"}
    if "level_y" not in batch or "level_ok" not in batch:
        raise AssertionError("level probe: the probe batch carries no level_y / level_ok (the loop's own call is level_stage)")
    W = float(cfg.w_level)
    eps32 = float(torch.finfo(torch.float32).eps)
    m = copy.deepcopy(model).cpu()
    m.train()
    vis = M.assert_level_head_vision_only(m, batch)
    aux_vis = M.assert_aux_vision_only(m, batch) if m.e_aux is not None else None                # PREREG_LEVELHEAD_A2: under --aux-cond token the assembly of O's extra token is vision-only too
    with torch.no_grad():                    # the number the arm's PREREG needs: W such that the level term is ~10 % of the total at STEP 0 (the copy before it is woken = the model as launched)
        torch.manual_seed(20261105)
        o0 = m.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    t0_total, t0_term = float(o0["terms"]["total"]), float(o0["terms"]["level"])
    step0 = {"step0_total": t0_total, "step0_level_term": t0_term, "step0_total_without_level": t0_total - W * t0_term,
             "w_level_for_10pct_share": ((0.1 / 0.9) * (t0_total - W * t0_term) / t0_term) if t0_term > 0.0 else None}
    g = torch.Generator().manual_seed(17)
    with torch.no_grad():
        if m.e_level is not None:                                    # PREREG_LEVELHEAD_A2: absent under --aux-cond token (e_aux has a default, non-zero init)
            m.e_level.weight.copy_(torch.randn(m.e_level.weight.shape, generator=g) * 0.5)
            m.e_level.bias.copy_(torch.randn(m.e_level.bias.shape, generator=g) * 0.5)
        wo = m.head_o[2].weight
        wo.copy_(torch.randn(wo.shape, generator=g) * 0.05)
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    none = torch.zeros_like(batch["level_ok"])

    def run(w_, ok_mask=None):
        m.cfg.w_level = w_
        for _, p in named:
            p.grad = None
        for p in m.prior.parameters():
            p.grad = None
        torch.manual_seed(20261105)
        out = m.loss({**batch, "level_ok": batch["level_ok"] if ok_mask is None else ok_mask}, p_gt=0.5, gen=torch.Generator().manual_seed(11))
        T = out["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"level probe: loss term {k} is not finite at w_level {w_}: {float(v)}")
        T["total"].backward()
        gr = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
        gr["__prior__"] = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in m.prior.parameters()])
        return float(T["total"].detach()), float(T["level"].detach()), gr, out["traj60"].detach().clone()

    t_w, term_w, g_w, tr_w = run(W)
    t_0, term_0, g_0, tr_0 = run(0.0)
    t_wb, _, g_wb, _ = run(W)
    t_n, term_n, g_n, _ = run(W, none)
    dsum = lambda ga, gb, pre: float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))      # noqa: E731
    asum = lambda ga, pre: float(sum(ga[n].abs().sum() for n in ga if n.startswith(pre)))                       # noqa: E731
    if m.e_level is not None:
        saved = (m.e_level.weight.detach().clone(), m.e_level.bias.detach().clone())
        with torch.no_grad():
            m.e_level.weight.zero_()
            m.e_level.bias.zero_()
        _, _, _, tr_z = run(0.0)
        with torch.no_grad():
            m.e_level.weight.copy_(saved[0])
            m.e_level.bias.copy_(saved[1])
        reach_grad = {"e_level_grad_at_w0": asum(g_0, "e_level.")}
    else:                                                            # PREREG_LEVELHEAD_A2 --aux-cond token: the level enters through column 0 of e_aux
        saved = m.e_aux.weight.detach().clone()
        with torch.no_grad():
            m.e_aux.weight[:, 0].zero_()
        _, _, _, tr_z = run(0.0)
        with torch.no_grad():
            m.e_aux.weight.copy_(saved)
        reach_grad = {"e_aux_level_col_grad_at_w0": float(g_0["e_aux.weight"][:, 0].abs().sum())}
    res = {"w_level_declared": W, "vision_only": vis, **step0, "term": term_w, "total_w": t_w, "total_0": t_0, "contribution": t_w - t_0, "w_level_x_term": W * term_w,
           "head_grad_diff": dsum(g_w, g_0, "lvl_head."), "head_grad_at_w0_detach_control": asum(g_0, "lvl_head."), **reach_grad, "aux_vision_only": aux_vis,
           "plan_moves_with_token": float((tr_0 - tr_z).abs().max()),
           "control_repeat_total_diff": abs(t_wb - t_w), "control_repeat_grad_diff": dsum(g_w, g_wb, ""),
           "control_allfalse_term": term_n, "control_allfalse_total_diff": abs(t_n - t_0), "control_allfalse_head_grad_diff": dsum(g_n, g_0, "lvl_head."),
           "control_prior_grad_diff": float((g_w["__prior__"] - g_0["__prior__"]).abs().sum())}
    if not math.isfinite(term_w) or not term_w > 0.0:
        raise AssertionError(f"level probe: the level term reads {term_w} (finite and > 0 required)")
    if res["control_repeat_total_diff"] != 0.0 or res["control_repeat_grad_diff"] != 0.0:
        raise AssertionError(f"level probe: the probe is not deterministic (the same loss twice differs by {res['control_repeat_total_diff']} / {res['control_repeat_grad_diff']}): its differences mean nothing")
    if not res["w_level_x_term"] > 64.0 * eps32 * max(abs(t_0), 1.0):
        raise AssertionError(f"level probe: w_level x term = {res['w_level_x_term']} is below float32 visibility against the total {t_0}: the probe cannot see the term")
    if abs(res["contribution"] - res["w_level_x_term"]) > rtol * res["w_level_x_term"] + 4.0 * eps32 * max(abs(t_w), abs(t_0)):
        raise AssertionError(f"level probe: the total moved by {res['contribution']} when the weight went from 0 to {W}, w_level x term is {res['w_level_x_term']}: the term does not reach the loss with its weight")
    if not res["head_grad_diff"] > 0.0:
        raise AssertionError("level probe: the level head receives no gradient from the level term (the sum of |d total / d lvl_head| differences is 0.0)")
    if res["head_grad_at_w0_detach_control"] != 0.0:
        raise AssertionError(f"level probe: at w_level 0 the head's gradient is {res['head_grad_at_w0_detach_control']}, not EXACTLY 0.0: the plan's loss reaches the head (O must read v_hat.detach())")
    if not list(reach_grad.values())[0] > 0.0 or not res["plan_moves_with_token"] > 0.0:
        raise AssertionError(f"level probe: the level token does not reach O ({reach_grad}, plan moves by {res['plan_moves_with_token']} when its path is zeroed)")
    if term_n != 0.0 or res["control_allfalse_total_diff"] != 0.0 or res["control_allfalse_head_grad_diff"] != 0.0:
        raise AssertionError(f"level probe: an ALL-FALSE level_ok must read exactly 0.0 (term {term_n}, total diff {res['control_allfalse_total_diff']}, head grad diff {res['control_allfalse_head_grad_diff']}): the mask is not what selects the rows")
    if res["control_prior_grad_diff"] != 0.0:
        raise AssertionError(f"level probe: the frozen P10 prior's gradient moved with the weight ({res['control_prior_grad_diff']}): the control must read exactly 0.0")
    res["pass"] = True
    return res


def ego_perturb_probe(model, batch, cfg, ego, exclude=None) -> dict:
    """G-LIVE for O-RECOVER (PREREG_V8F_HACC_MC section 7): the ego perturbation REACHES THE LOSS THE TRAINER BACKPROPAGATES -- read from the BUILT model's own ``loss`` on the real probe batch (a COPY, its own
    seeds), through the SAME ``ego.apply`` the training loop calls. With the lever off (the default) there is nothing to prove and nothing runs (the pre-change model and loss).

      partition    every key ``forward_infer`` reads is in ``EGO_PERTURB_REBUILT`` or ``EGO_PERTURB_KEPT``: no ego tensor the model reads is left unperturbed by omission
      batch        at p = 1 every selected row differs in p10f, base, kin3 and p10_t; every unselected row is byte-identical in every rebuilt tensor; every other key is the SAME object
      law          the selected rows' speed ratio lies in [LO, HI] and the acceleration drop in [0, ADROP]
      G-LIVE       ``live_check`` on the PERTURBED batch: every loss term finite, every trainable group gets a non-zero gradient, the frozen prior reads EXACTLY 0.0
      gradient     sum |grad(total on the perturbed batch) - grad(total on the unperturbed batch)| > 0 and the loss moves; with p1 replaced alone and with base replaced alone the loss moves too
      controls     (a) the same loss twice on the unperturbed batch differs by EXACTLY 0.0 (the probe is deterministic); (b) p = 0 returns the SAME batch object and its loss / gradient differ from the
                   unperturbed ones by EXACTLY 0.0; (c) the frozen P10 prior's gradient difference is EXACTLY 0.0; (d) so is the world-model auxiliary's (``act`` is its transition record and is NOT perturbed)

    ``exclude`` ([B] bool, default None = the pre-change probe): AMEND1 section 1 runs this SAME probe on a COMPOSED batch -- ``batch`` is the DAgger-swapped batch and ``exclude`` the swap's mask -- so that
    the "unselected rows are byte-identical" clause also proves the perturbation never touched a swapped row.

    A violation is an AssertionError, which the caller turns into a refusal. DELETE the line in ``EgoPerturb.apply`` that applies the rebuild, or the ``ego.apply`` line in the loop's input path, and the
    batch / gradient clauses read 0.0 and the gate refuses the launch."""
    if not ego.active:
        return {"ego_perturb": "off", "pass": True, "skipped": "--ego-perturb-p 0: the perturbation is not applied; the default run is the pre-change run"}
    xk = {} if exclude is None else {"exclude": exclude}                          # passed ONLY when given: the pre-change probe calls ``ego.apply`` exactly as before
    m = copy.deepcopy(model).cpu()
    m.train()
    pert = ego.apply(batch, ego.PROBE_STEP, p=1.0, **xk)
    last = dict(ego.last)
    sel = last["sel"]
    n_sel = int(sel.sum())
    if n_sel < 1:
        raise AssertionError(f"ego probe: no ELIGIBLE window (logged v0 < {ego.vmax} m/s) among the {last['n_rows']} probe windows: the perturbation cannot be seen")
    B = int(sel.numel())
    res = {"p_declared": ego.p, "n_rows": B, "n_eligible": last["n_eligible"], "n_selected_at_p1": n_sel}

    m.eval()
    with torch.no_grad():
        keys = set(m.forward_infer(batch)["_keys_read"])
    m.train()
    stray = sorted(keys - set(D.EGO_PERTURB_REBUILT) - set(D.EGO_PERTURB_KEPT))
    if stray:
        raise AssertionError(f"ego probe: forward_infer reads {stray}, which is neither rebuilt nor kept by the perturbation: an input the perturbation leaves unaccounted for")
    res["keys_read"] = sorted(keys)

    def changed(x, y):                                                           # [B] bool: row differs (NaN == NaN)
        same = (x == y) | (torch.isnan(x) & torch.isnan(y)) if x.is_floating_point() else (x == y)
        return (~same).reshape(x.shape[0], -1).any(1)
    res["rows_changed"] = {}
    for k in D.EGO_PERTURB_REBUILT:
        ch = changed(pert[k], batch[k])
        res["rows_changed"][k] = int(ch.sum())
        if bool(ch[~sel].any()):
            raise AssertionError(f"ego probe: {k} changed on a row the perturbation did not select")
        need = sel
        if k == "base":                               # PREREG_V8F_ORECOVER Amendment 4: a selected row AT REST cannot move. Logged v0 = 0 (so s v0 = 0) and a perturbed a0 <= 0 keep
            # the kinematic base identically 0 before and after (MEASURED on box 2's gate batch, 2026-10-26: 1 of 29 rows, v0 = a0 = 0). Exactly those rows are exempt and counted;
            # every other selected row must still change, and at least one must exist.
            at_rest = (sel & (batch["kin3"][:, 0] == 0.0) & (pert["kin3"][:, 1] <= 0.0)
                       & (batch["base"].reshape(B, -1) == 0.0).all(1) & (pert["base"].reshape(B, -1) == 0.0).all(1))
            res["base_rows_at_rest_exempt"] = int(at_rest.sum())
            need = sel & ~at_rest
            if not bool(need.any()):
                raise AssertionError(f"ego probe: all {n_sel} selected rows are at rest: the base cannot show the rebuild")
        if k in ("p10f", "base", "kin3", "p10_t") and not bool(ch[need].all()):
            raise AssertionError(f"ego probe: {k} is unchanged on {int((~ch[need]).sum())} of {int(need.sum())} selected rows: the rebuild did not reach it")
    kept_moved = sorted(k for k in batch if k not in D.EGO_PERTURB_REBUILT and pert[k] is not batch[k])
    if kept_moved:
        raise AssertionError(f"ego probe: {kept_moved} are not the same objects after the perturbation: targets, labels and kept inputs must not be touched")

    k0, k1 = batch["kin3"][sel].double(), pert["kin3"][sel].double()
    pos = k0[:, 0] > 1e-3
    ratio = k1[pos, 0] / k0[pos, 0]
    drop = k0[:, 1] - k1[:, 1]
    lo, hi = ego.vscale
    res["speed_ratio_range"] = [float(ratio.min()), float(ratio.max())] if ratio.numel() else None
    res["accel_drop_range"] = [float(drop.min()), float(drop.max())]
    if ratio.numel() and not bool(((ratio >= lo - 1e-4) & (ratio <= hi + 1e-4)).all()):
        raise AssertionError(f"ego probe: the speed ratio {res['speed_ratio_range']} is outside the declared [{lo}, {hi}]")
    if not bool(((drop >= -1e-4) & (drop <= ego.adrop + 1e-4)).all()):
        raise AssertionError(f"ego probe: the acceleration drop {res['accel_drop_range']} is outside the declared [0, {ego.adrop}]")

    lp = live_check(m, pert, cfg)                                                # G-LIVE on the PERTURBED batch (its own deep copy)
    res["live_perturbed"] = {"pass": lp["pass"], "terms": lp["terms"], "control_prior_grad_sum": lp["control_prior_grad_sum"], "groups_grad": lp["groups_grad"]}

    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]

    def run(b, grad=True):
        for _, p in named:
            p.grad = None
        for p in m.prior.parameters():
            p.grad = None
        torch.manual_seed(20261026)
        with torch.set_grad_enabled(grad):
            T = m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]
            for k, v in T.items():
                if not torch.isfinite(v):
                    raise AssertionError(f"ego probe: loss term {k} is not finite: {float(v)}")
            if not grad:
                return float(T["total"].detach()), None, None
            T["total"].backward()
        g = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
        gp = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in m.prior.parameters()])
        return float(T["total"].detach()), g, gp
    dsum = lambda ga, gb, pre: float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))      # noqa: E731
    t0, g0, gp0 = run(batch)
    t0b, g0b, _ = run(batch)
    t1, g1, gp1 = run(pert)
    b_zero = ego.apply(batch, ego.PROBE_STEP, p=0.0, **xk)
    tz, gz, _ = run(b_zero)
    res.update({"total_unperturbed": t0, "total_perturbed": t1, "loss_diff": t1 - t0, "grad_diff_all": dsum(g1, g0, ""),
                "grad_diff_by_group": {pre: dsum(g1, g0, pre) for pre in ("adapter", "e_ego", "read_s", "read_t", "read_o", "head_s", "head_o", "wm.")},
                "control_repeat_loss_diff": abs(t0b - t0), "control_repeat_grad_diff": dsum(g0, g0b, ""),
                "control_p0_same_object": b_zero is batch, "control_p0_loss_diff": abs(tz - t0), "control_p0_grad_diff": dsum(gz, g0, ""),
                "control_prior_grad_diff": float((gp1 - gp0).abs().sum()), "control_wm_grad_diff": dsum(g1, g0, "wm.") if m.wm is not None else None})
    only = {}
    for name in ("p10f", "base", "pose_t", "pose_s", "kin3", "p10_t"):         # one rebuilt tensor at a time: which of them the loss reads
        only[name] = run({**batch, name: pert[name]}, grad=False)[0] - t0
    res["loss_diff_replacing_only"] = only
    if res["control_repeat_loss_diff"] != 0.0 or res["control_repeat_grad_diff"] != 0.0:
        raise AssertionError(f"ego probe: the probe is not deterministic (the same loss twice differs by {res['control_repeat_loss_diff']} / {res['control_repeat_grad_diff']}): its differences mean nothing")
    if not res["control_p0_same_object"] or res["control_p0_loss_diff"] != 0.0 or res["control_p0_grad_diff"] != 0.0:
        raise AssertionError(f"ego probe: p = 0 is not the identity (same object {res['control_p0_same_object']}, loss diff {res['control_p0_loss_diff']}, grad diff {res['control_p0_grad_diff']}): the control must read exactly 0.0")
    if not (res["loss_diff"] != 0.0 and res["grad_diff_all"] > 0.0):
        raise AssertionError(f"ego probe: the perturbed batch moves the loss by {res['loss_diff']} and the gradient by {res['grad_diff_all']}: the perturbation does not reach the loss the trainer backpropagates")
    for name in ("p10f", "base"):
        if not only[name] != 0.0:
            raise AssertionError(f"ego probe: replacing {name} alone does not move the loss: the model does not read the rebuilt {name}")
    if 0.0 != res["control_prior_grad_diff"]:                   # the same test as the acc probe's, spelled differently: mutate_oacc.py anchors that line once
        raise AssertionError(f"ego probe: the frozen P10 prior's gradient moved with the perturbation ({res['control_prior_grad_diff']}): the control must read exactly 0.0")
    if res["control_wm_grad_diff"] not in (0.0, None):          # the same test as the acc probe's, spelled differently: mutate_oacc.py anchors that line once
        raise AssertionError(f"ego probe: the world-model auxiliary's gradient moved with the perturbation ({res['control_wm_grad_diff']}): act is its transition record and is not perturbed, the control must read exactly 0.0")
    res["pass"] = True
    return res


def recover_probe(model, batch, cfg, rec) -> dict:
    """G-LIVE for PREREG_RECOVER_FT (H-RECOVER-FT-1): the recovery perturbation REACHES THE LOSS THE TRAINER BACKPROPAGATES, in BOTH of its halves -- the perturbed ego tensors AND the re-expressed target ``gt`` -- read from
    the BUILT model's own ``loss`` on the real probe batch (a COPY, its own seeds), through the SAME ``rec.apply`` the training loop calls, at p = 1. With the lever off (p = 0, the default) there is nothing to prove and
    nothing runs (the pre-change model and loss).

      partition    every key ``forward_infer`` reads is rebuilt (``R.RECOVER_REBUILT``) or kept (``EGO_PERTURB_KEPT``)
      batch        at p = 1 every ELIGIBLE row is selected; every selected row differs in kin3, p10f, p10_t (a row whose drawn deviation is exactly zero is exempt and counted; a selected row AT REST may keep its
                   kinematic base, counted); every unselected row is byte-identical in every rebuilt tensor and in gt; every other key is the SAME object; ``lab_s`` differs in rc alone
      target       the batch's ``gt`` on the selected rows equals ``R.rejoin_target`` recomputed from the drawn (dy, dpsi) with the declared mode and K; ``lab_s.rc`` equals its frame change; at least one row moved
      G-LIVE       ``live_check`` on the RECOVERED batch: every loss term finite, every trainable group a non-zero gradient, the frozen prior reads EXACTLY 0.0
      gradient     INPUTS: the loss and the gradient on (recovered inputs, LOGGED target) differ from the logged batch's, and d total / d p10f on the selected rows is non-zero; TARGET: the same on (logged inputs,
                   re-expressed target); and the whole recovered batch moves both
      controls     (a) the same loss twice on the logged batch differs by EXACTLY 0.0 (the probe is deterministic); (b) p = 0 returns the SAME batch object and its loss / gradient differ from the logged ones by EXACTLY 0.0;
                   (c) the frozen P10 prior's gradient difference is EXACTLY 0.0; (d) so is the world-model auxiliary's (``act`` is its transition record and is NOT perturbed)

    A violation is an AssertionError, which the caller turns into a refusal. DELETE the line in ``RecoverPerturb.apply`` that applies the rebuild and the batch / target / gradient clauses read 0 and the gate refuses the
    launch (``tests/test_v8f_recover_wiring.py`` runs that mutation); DELETE the ``rec.apply`` line in the training loop and the loop's own guard (``ZZRECOVER-NOT-APPLIEDZZ``) stops the run at its first step."""
    if not rec.active:
        return {"recover": "off", "pass": True, "skipped": "--recover-p 0: the perturbation is not applied; the default run is the pre-change run"}
    m = copy.deepcopy(model).cpu()
    m.train()
    pert = rec.apply(batch, rec.PROBE_STEP, p=1.0)
    last = dict(rec.last)
    sel = last["sel"]
    n_sel = int(sel.sum())
    if n_sel < 1:
        raise AssertionError(f"recover probe: no ELIGIBLE window (NOW a real kinematic row) among the {last['n_rows']} probe windows: the perturbation cannot be seen")
    B = int(sel.numel())
    res = {"p_declared": rec.p, "n_rows": B, "n_eligible": last["n_eligible"], "n_selected_at_p1": n_sel}
    if n_sel != last["n_eligible"]:
        raise AssertionError(f"recover probe: at p = 1 {n_sel} rows were selected but {last['n_eligible']} are eligible: every eligible row must be selected")

    m.eval()
    with torch.no_grad():
        keys = set(m.forward_infer(batch)["_keys_read"])
    m.train()
    stray = sorted(keys - set(R.RECOVER_REBUILT) - set(D.EGO_PERTURB_KEPT))
    if stray:
        raise AssertionError(f"recover probe: forward_infer reads {stray}, which is neither rebuilt nor kept by the perturbation: an input the perturbation leaves unaccounted for")
    res["keys_read"] = sorted(keys)

    def changed(x, y):                                                           # [B] bool: row differs (NaN == NaN)
        same = (x == y) | (torch.isnan(x) & torch.isnan(y)) if x.is_floating_point() else (x == y)
        return (~same).reshape(x.shape[0], -1).any(1)
    dev = last["dev"]
    zero_dev = sel & (torch.stack([dev[k] == 0.0 for k in R.TABLE_FIELDS], 0).all(0))                 # a drawn deviation of exactly zero moves nothing: exempt, counted
    res["n_zero_deviation_rows_exempt"] = int(zero_dev.sum())
    need = sel & ~zero_dev
    if not bool(need.any()):
        raise AssertionError("recover probe: every selected row drew an all-zero deviation: the perturbation cannot be seen")
    res["rows_changed"] = {}
    for k in R.RECOVER_REBUILT + ("gt",):
        ch = changed(pert[k], batch[k])
        res["rows_changed"][k] = int(ch.sum())
        if bool(ch[~sel].any()):
            raise AssertionError(f"recover probe: {k} changed on a row the perturbation did not select")
        if k == "base":                               # a selected row AT REST cannot move its kinematic base (v0 = 0 stays 0, the base is identically 0 before and after): exempt and counted, as ORC's probe does
            at_rest = (need & (batch["kin3"][:, 0] == 0.0) & (batch["base"].reshape(B, -1) == 0.0).all(1) & (pert["base"].reshape(B, -1) == 0.0).all(1))
            res["base_rows_at_rest_exempt"] = int(at_rest.sum())
            if not bool((need & ~at_rest).any()):
                raise AssertionError(f"recover probe: all {int(need.sum())} selected rows are at rest: the base cannot show the rebuild")
            if not bool(ch[need & ~at_rest].all()):
                raise AssertionError(f"recover probe: base is unchanged on {int((~ch[need & ~at_rest]).sum())} of {int((need & ~at_rest).sum())} selected rows: the rebuild did not reach it")
        elif k in ("p10f", "kin3", "p10_t") and not bool(ch[need].all()):
            raise AssertionError(f"recover probe: {k} is unchanged on {int((~ch[need]).sum())} of {int(need.sum())} selected rows: the rebuild did not reach it")
        elif k == "gt" and not bool(ch.any()):
            raise AssertionError("recover probe: gt is unchanged on every selected row: the target was not re-expressed")
    if rec.dev_shape == "pertick":                                               # the PROFILE reaches the batch: every real history frame of v is max(v + dv_t[k], 0) of the drawn record, and the drawn profiles are not all flat
        prof = last["dev_t"]["dv_t"][sel]
        if not bool((prof[:, :-1] != prof[:, 1:]).any()):
            raise AssertionError("recover probe: every selected row drew a flat dv profile: the per-tick shape cannot be told from the constant one")
        vm_sel = D.EgoPerturb._hist_valid(batch, sel.nonzero(as_tuple=True)[0])
        want = torch.clamp(batch["p10f"][sel][..., 0].double() + prof, min=0.0)
        diff = torch.where(vm_sel, (pert["p10f"][sel][..., 0].double() - want).abs(), torch.zeros_like(want))
        res["pertick_v_max_abs_err"] = float(diff.max())
        if not res["pertick_v_max_abs_err"] <= 1e-4:
            raise AssertionError(f"recover probe: the history speeds of the selected rows differ from max(v + dv_t, 0) of the drawn per-tick profile by {res['pertick_v_max_abs_err']} (> 1e-4): the profile did not reach the batch")
    ls0, ls1 = batch["lab_s"], pert["lab_s"]
    lab_moved = sorted(k for k in ls0 if ls1.get(k) is not ls0[k])
    if lab_moved not in ([], ["rc"]):
        raise AssertionError(f"recover probe: lab_s changed in more than rc ({lab_moved}): rc is the one label tensor re-expressed")
    kept_moved = sorted(k for k in batch if k not in R.RECOVER_REBUILT and k not in ("gt", "lab_s") and pert[k] is not batch[k])
    if kept_moved:
        raise AssertionError(f"recover probe: {kept_moved} are not the same objects after the perturbation: labels, record and kept inputs must not be touched")
    gt_expect = R.rejoin_target(batch["gt"][sel].double(), dev["dy"][sel].double(), dev["dpsi"][sel].double(), mode=rec.target, steps=rec.rejoin_steps)
    res["gt_max_abs_err_vs_rejoin_target"] = float((pert["gt"][sel].double() - gt_expect).abs().max())
    if res["gt_max_abs_err_vs_rejoin_target"] > 1e-5:
        raise AssertionError(f"recover probe: the batch's gt differs from R.rejoin_target({rec.target!r}, K = {rec.rejoin_steps}) of the drawn (dy, dpsi) by {res['gt_max_abs_err_vs_rejoin_target']} (> 1e-5): the target is not the registered one")
    rc_expect = DG.rc_frame_change(ls0["rc"][sel], ls0["rc_ok"][sel], batch["pose_s"][sel], pert["pose_s"][sel])
    res["rc_max_abs_err_vs_frame_change"] = float((torch.nan_to_num(ls1["rc"][sel]) - torch.nan_to_num(rc_expect)).abs().max())
    if res["rc_max_abs_err_vs_frame_change"] > 1e-5 or bool(changed(ls1["rc"], ls0["rc"])[~sel].any()):
        raise AssertionError(f"recover probe: lab_s.rc differs from its frame change by {res['rc_max_abs_err_vs_frame_change']} (> 1e-5) or moved on an unselected row")

    lp = live_check(m, pert, cfg)                                                # G-LIVE on the RECOVERED batch (its own deep copy)
    res["live_recovered"] = {"pass": lp["pass"], "terms": lp["terms"], "control_prior_grad_sum": lp["control_prior_grad_sum"], "groups_grad": lp["groups_grad"]}

    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]

    def run(b, grad=True):
        for _, p in named:
            p.grad = None
        for p in m.prior.parameters():
            p.grad = None
        torch.manual_seed(20261102)
        with torch.set_grad_enabled(grad):
            T = m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]
            for k, v in T.items():
                if not torch.isfinite(v):
                    raise AssertionError(f"recover probe: loss term {k} is not finite: {float(v)}")
            if not grad:
                return float(T["total"].detach()), None, None
            T["total"].backward()
        g = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
        gp = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in m.prior.parameters()])
        return float(T["total"].detach()), g, gp
    dsum = lambda ga, gb, pre: float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))      # noqa: E731
    t0, g0, gp0 = run(batch)
    t0b, g0b, _ = run(batch)
    t1, g1, gp1 = run(pert)                                                      # the whole recovered batch: perturbed inputs AND re-expressed target
    t_in, g_in, _ = run({**pert, "gt": batch["gt"]})                             # INPUTS only: recovered ego tensors, the LOGGED target
    t_tg, g_tg, _ = run({**batch, "gt": pert["gt"]})                             # TARGET only: logged ego tensors, the re-expressed target
    x = pert["p10f"].detach().clone().requires_grad_(True)                       # d total / d p10f on the selected rows: the recovered tensor itself is read by the loss the trainer backpropagates
    for _, p in named:
        p.grad = None
    m.loss({**pert, "p10f": x}, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]["total"].backward()
    d_p10f = float(x.grad[sel].abs().sum()) if x.grad is not None else 0.0
    b_zero = rec.apply(batch, rec.PROBE_STEP, p=0.0)
    tz, gz, _ = run(b_zero)
    res.update({"total_logged": t0, "total_recovered": t1, "loss_diff": t1 - t0, "grad_diff_all": dsum(g1, g0, ""),
                "loss_diff_inputs_only": t_in - t0, "grad_diff_inputs_only": dsum(g_in, g0, ""), "loss_diff_target_only": t_tg - t0, "grad_diff_target_only": dsum(g_tg, g0, ""),
                "grad_wrt_p10f_selected_abs_sum": d_p10f,
                "grad_diff_by_group": {pre: dsum(g1, g0, pre) for pre in ("adapter", "e_ego", "read_s", "read_t", "read_o", "head_s", "head_o", "wm.")},
                "control_repeat_loss_diff": abs(t0b - t0), "control_repeat_grad_diff": dsum(g0, g0b, ""),
                "control_p0_same_object": b_zero is batch, "control_p0_loss_diff": abs(tz - t0), "control_p0_grad_diff": dsum(gz, g0, ""),
                "control_prior_grad_diff": float((gp1 - gp0).abs().sum()), "control_wm_grad_diff": dsum(g1, g0, "wm.") if m.wm is not None else None})
    if res["control_repeat_loss_diff"] != 0.0 or res["control_repeat_grad_diff"] != 0.0:
        raise AssertionError(f"recover probe: the probe is not deterministic (the same loss twice differs by {res['control_repeat_loss_diff']} / {res['control_repeat_grad_diff']}): its differences mean nothing")
    if not res["control_p0_same_object"] or res["control_p0_loss_diff"] != 0.0 or res["control_p0_grad_diff"] != 0.0:
        raise AssertionError(f"recover probe: p = 0 is not the identity (same object {res['control_p0_same_object']}, loss diff {res['control_p0_loss_diff']}, grad diff {res['control_p0_grad_diff']}): the control must read exactly 0.0")
    if not (res["loss_diff"] != 0.0 and res["grad_diff_all"] > 0.0):
        raise AssertionError(f"recover probe: the recovered batch moves the loss by {res['loss_diff']} and the gradient by {res['grad_diff_all']}: the perturbation does not reach the loss the trainer backpropagates")
    if not (res["loss_diff_inputs_only"] != 0.0 and res["grad_diff_inputs_only"] > 0.0 and d_p10f > 0.0):
        raise AssertionError(f"recover probe: the recovered ego tensors do not reach the loss gradient (loss diff {res['loss_diff_inputs_only']}, gradient diff {res['grad_diff_inputs_only']}, d total / d p10f {d_p10f})")
    if not (res["loss_diff_target_only"] != 0.0 and res["grad_diff_target_only"] > 0.0):
        raise AssertionError(f"recover probe: the re-expressed target does not reach the loss gradient (loss diff {res['loss_diff_target_only']}, gradient diff {res['grad_diff_target_only']})")
    if 0.0 != res["control_prior_grad_diff"]:
        raise AssertionError(f"recover probe: the frozen P10 prior's gradient moved with the perturbation ({res['control_prior_grad_diff']}): the control must read exactly 0.0")
    if res["control_wm_grad_diff"] not in (0.0, None):
        raise AssertionError(f"recover probe: the world-model auxiliary's gradient moved with the perturbation ({res['control_wm_grad_diff']}): act is its transition record and is not perturbed, the control must read exactly 0.0")
    res["pass"] = True
    return res


def dagger_probe(model, batch, cfg, dag) -> dict:
    """G-LIVE for DAgger (PREREG_V8F_DAGGER section 6): the swapped own-state tensors REACH THE LOSS THE TRAINER BACKPROPAGATES -- read from the BUILT model's own ``loss`` on the real probe batch (a COPY, its own
    seeds), through the SAME ``dag.apply`` the training loop calls, at p = 1. With the lever off (the default) there is nothing to prove and nothing runs (the pre-change model and loss).

      partition    every key ``forward_infer`` reads is in ``DG.REBUILT`` or ``DG.KEPT``: no ego tensor the model reads is left un-swapped by omission
      batch        the swapped batch holds EXACTLY the keys of the logged one; every selected row is the window's own matched state (the record's clip and r' are the row's (ci, t), its kin3 IS the bank's);
                   every selected row changes in p10f, base, kin3 and p10_t -- except a row whose own ego is BIT-IDENTICAL to the logged one (exempted and counted; at least one non-identical row must
                   exist, ORC A4's rule generalised) and, for ``base`` alone, a row at rest in both (v0 = 0, base = 0: it cannot move); NO unselected row changes in any rebuilt tensor
      kept         every key outside ``DG.REBUILT`` is the SAME object (targets, labels, images, ages, the anchor ``pose_now``, ``act``); ``lab_s`` differs in ``rc`` only, and ``rc`` equals its frame change
                   (``DG.rc_frame_change``, recomputed here from the logged and the swapped ``pose_s``) to 1e-5 on the selected rows and is untouched on the others
      G-LIVE       ``live_check`` on the SWAPPED batch: every loss term finite, every trainable group gets a non-zero gradient, the frozen prior reads EXACTLY 0.0
      gradient     sum |grad(total on the swapped batch) - grad(total on the logged batch)| > 0 and the loss moves; replacing ``p10f`` alone moves the loss too
      controls     (a) the same loss twice on the logged batch differs by EXACTLY 0.0 (the probe is deterministic); (b) p = 0 returns the SAME batch object and its loss / gradient differ from the logged
                   ones by EXACTLY 0.0; (c) the frozen P10 prior's gradient difference is EXACTLY 0.0; (d) so is the world-model auxiliary's (``act`` is its transition record and is KEPT)

    A violation is an AssertionError, which the caller turns into a refusal. DELETE the line in ``DaggerSwap.apply`` that applies the swap, or the ``dag.apply`` line in the loop's input path, and the batch / gradient
    clauses read 0.0 and the gate refuses the launch."""
    if not dag.active:
        return {"dagger": "off", "pass": True, "skipped": "--dagger-p 0: no row is swapped; the default run is the pre-change run"}
    m = copy.deepcopy(model).cpu()
    m.train()
    pert = dag.apply(batch, dag.PROBE_STEP, p=1.0)
    last = dict(dag.last)
    sel = last["sel"]
    n_sel = int(sel.sum())
    if n_sel < 1:
        raise AssertionError(f"dagger probe: no ELIGIBLE window (an own state with r' = t that passed every cut) among the {last['n_rows']} probe windows: the swap cannot be seen")
    B = int(sel.numel())
    res = {"p_declared": dag.p, "n_rows": B, "n_eligible": last["n_eligible"], "n_selected_at_p1": n_sel}

    if set(pert) != set(batch):
        raise AssertionError(f"dagger probe: the swap changed the batch's keys (added {sorted(set(pert) - set(batch))}, removed {sorted(set(batch) - set(pert))}): it may replace the named ego tensors and nothing else")
    m.eval()
    with torch.no_grad():
        keys = set(m.forward_infer(batch)["_keys_read"])
    m.train()
    stray = sorted(keys - set(DG.REBUILT) - set(DG.KEPT))
    if stray:
        raise AssertionError(f"dagger probe: forward_infer reads {stray}, which is neither replaced nor kept by the swap: an input the swap leaves unaccounted for")
    res["keys_read"] = sorted(keys)

    def changed(x, y):                                                           # [B] bool: row differs (NaN == NaN)
        same = (x == y) | (torch.isnan(x) & torch.isnan(y)) if x.is_floating_point() else (x == y)
        return (~same).reshape(x.shape[0], -1).any(1)
    for k in DG.REBUILT:
        if pert[k].shape != batch[k].shape or pert[k].dtype != batch[k].dtype:
            raise AssertionError(f"dagger probe: {k} changed shape / dtype under the swap ({tuple(batch[k].shape)} {batch[k].dtype} -> {tuple(pert[k].shape)} {pert[k].dtype})")
    ch = {k: changed(pert[k], batch[k]) for k in DG.REBUILT}
    res["rows_changed"] = {k: int(v.sum()) for k, v in ch.items()}
    for k in DG.REBUILT:
        if bool(ch[k][~sel].any()):
            raise AssertionError(f"dagger probe: {k} changed on a row the swap did not select")
    idx = np.asarray(last["record_idx"])
    rec = dag.bank.gather(idx)
    sel_rows = sel.nonzero(as_tuple=True)[0]
    if not (np.array_equal(rec["r"].astype(np.int64), batch["t"][sel_rows].numpy()) and np.array_equal(dag.bank.ci[idx], batch["ci"][sel_rows].numpy())
            and bool((dag.bank.reason[idx] == DG.OK).all())):
        raise AssertionError("dagger probe: a selected row was swapped for a record that is not its window's own matched, uncut state (clip, r' = t, reason ok)")
    if not bool((pert["kin3"][sel_rows] == torch.from_numpy(rec["kin3"]).to(batch["kin3"].dtype)).all()):
        raise AssertionError("dagger probe: the swapped kin3 of a selected row is not the bank record's own kin3: the swap does not write the bank's state into the batch")
    identical = sel & ~torch.stack([ch[k] for k in ("p10f", "base", "kin3", "p10_t")]).any(0)          # selected rows whose own ego equals the logged one bit for bit
    need = sel & ~identical
    res["selected_rows_bit_identical_exempt"] = int(identical.sum())
    if not bool(need.any()):
        raise AssertionError(f"dagger probe: all {n_sel} selected rows carry an own ego state bit-identical to the logged one: the swap cannot be seen")
    base_rest = (need & (batch["kin3"][:, 0] == 0.0) & (pert["kin3"][:, 0] == 0.0) & (batch["base"].reshape(B, -1) == 0.0).all(1) & (pert["base"].reshape(B, -1) == 0.0).all(1))
    res["base_rows_at_rest_exempt"] = int(base_rest.sum())
    for k in ("p10f", "base", "kin3", "p10_t"):
        req = need & ~base_rest if k == "base" else need
        if not bool(ch[k][req].all()):
            raise AssertionError(f"dagger probe: {k} is unchanged on {int((~ch[k][req]).sum())} of {int(req.sum())} selected rows: the swap did not reach it")
    kept_moved = sorted(k for k in batch if k not in DG.REBUILT and k != "lab_s" and pert[k] is not batch[k])
    if kept_moved:
        raise AssertionError(f"dagger probe: {kept_moved} are not the same objects after the swap: targets, labels and kept inputs must not be touched")
    ls0, ls1 = batch["lab_s"], pert["lab_s"]
    if set(ls0) != set(ls1) or [k for k in ls0 if k != "rc" and ls1[k] is not ls0[k]]:
        raise AssertionError(f"dagger probe: lab_s changed in more than rc ({sorted(k for k in ls0 if k != 'rc' and ls1.get(k) is not ls0[k])}): rc is the one label tensor re-expressed")
    if bool(changed(ls1["rc"], ls0["rc"])[~sel].any()):
        raise AssertionError("dagger probe: lab_s.rc changed on a row the swap did not select")
    want_rc = DG.rc_frame_change(ls0["rc"][sel_rows], ls0["rc_ok"][sel_rows], batch["pose_s"][sel_rows], pert["pose_s"][sel_rows])
    res["rc_max_abs_err"] = float((ls1["rc"][sel_rows].double() - want_rc.double()).abs().max())
    if not res["rc_max_abs_err"] <= 1e-5:
        raise AssertionError(f"dagger probe: lab_s.rc differs from its frame change by {res['rc_max_abs_err']} (> 1e-5): the labels were not re-expressed in the swapped S frame")

    lp = live_check(m, pert, cfg)                                                # G-LIVE on the SWAPPED batch (its own deep copy)
    res["live_swapped"] = {"pass": lp["pass"], "terms": lp["terms"], "control_prior_grad_sum": lp["control_prior_grad_sum"], "groups_grad": lp["groups_grad"]}

    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]

    def run(b, grad=True):
        for _, p in named:
            p.grad = None
        for p in m.prior.parameters():
            p.grad = None
        torch.manual_seed(20261026)
        with torch.set_grad_enabled(grad):
            T = m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]
            for k, v in T.items():
                if not torch.isfinite(v):
                    raise AssertionError(f"dagger probe: loss term {k} is not finite: {float(v)}")
            if not grad:
                return float(T["total"].detach()), None, None
            T["total"].backward()
        g = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
        gp = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in m.prior.parameters()])
        return float(T["total"].detach()), g, gp
    dsum = lambda ga, gb, pre: float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))      # noqa: E731
    t0, g0, gp0 = run(batch)
    t0b, g0b, _ = run(batch)
    t1, g1, gp1 = run(pert)
    b_zero = dag.apply(batch, dag.PROBE_STEP, p=0.0)
    tz, gz, _ = run(b_zero)
    res.update({"total_logged": t0, "total_swapped": t1, "loss_diff": t1 - t0, "grad_diff_all": dsum(g1, g0, ""),
                "grad_diff_by_group": {pre: dsum(g1, g0, pre) for pre in ("adapter", "e_ego", "read_s", "read_t", "read_o", "head_s", "head_o", "wm.")},
                "control_repeat_loss_diff": abs(t0b - t0), "control_repeat_grad_diff": dsum(g0, g0b, ""),
                "control_p0_same_object": b_zero is batch, "control_p0_loss_diff": abs(tz - t0), "control_p0_grad_diff": dsum(gz, g0, ""),
                "control_prior_grad_diff": float((gp1 - gp0).abs().sum()), "control_wm_grad_diff": dsum(g1, g0, "wm.") if m.wm is not None else None})
    only = {}
    for name in ("p10f", "base", "pose_t", "pose_s", "kin3", "p10_t"):          # one swapped tensor at a time: which of them the loss reads
        only[name] = run({**batch, name: pert[name]}, grad=False)[0] - t0
    res["loss_diff_replacing_only"] = only
    if res["control_repeat_loss_diff"] != 0.0 or res["control_repeat_grad_diff"] != 0.0:
        raise AssertionError(f"dagger probe: the probe is not deterministic (the same loss twice differs by {res['control_repeat_loss_diff']} / {res['control_repeat_grad_diff']}): its differences mean nothing")
    if not res["control_p0_same_object"] or res["control_p0_loss_diff"] != 0.0 or res["control_p0_grad_diff"] != 0.0:
        raise AssertionError(f"dagger probe: p = 0 is not the identity (same object {res['control_p0_same_object']}, loss diff {res['control_p0_loss_diff']}, grad diff {res['control_p0_grad_diff']}): the control must read exactly 0.0")
    if not (res["loss_diff"] != 0.0 and res["grad_diff_all"] > 0.0):
        raise AssertionError(f"dagger probe: the swapped batch moves the loss by {res['loss_diff']} and the gradient by {res['grad_diff_all']}: the swap does not reach the loss the trainer backpropagates")
    if not only["p10f"] != 0.0:
        raise AssertionError("dagger probe: replacing p10f alone does not move the loss: the model does not read the swapped p10f")
    if 0.0 != res["control_prior_grad_diff"]:
        raise AssertionError(f"dagger probe: the frozen P10 prior's gradient moved with the swap ({res['control_prior_grad_diff']}): the control must read exactly 0.0")
    if res["control_wm_grad_diff"] not in (0.0, None):
        raise AssertionError(f"dagger probe: the world-model auxiliary's gradient moved with the swap ({res['control_wm_grad_diff']}): act is its transition record and is kept, the control must read exactly 0.0")
    res["pass"] = True
    return res


COMPOSE_PROBE_DAG_P = 0.5          # the swap's probability in the composed probe: about half of the eligible rows are swapped, the rest are left for the perturbation (at p = 1.0)


def ego_dagger_compose_probe(model, batch, cfg, comp) -> dict:
    """G-LIVE for AMEND1 section 1 (``--ego-perturb-with-dagger``): on a COMPOSED batch -- ``comp.apply``, the SAME call the training loop makes -- BOTH levers reach the loss the trainer backpropagates and the
    p = 0 controls read EXACTLY 0.0. Off (the flag not named) there is nothing to prove and nothing runs.

      composed   the first probe step k < 0 (-1, -2, ..., never a training step) at which the swap (p = COMPOSE_PROBE_DAG_P) AND the perturbation (p = 1.0: every eligible sample the swap did not replace) both
                 show on this batch; the overlap, measured from the two masks, is EXACTLY 0; every ego tensor of a swapped row equals the swap's own value (the perturber never touched it)
      ego        the EXISTING ``ego_perturb_probe`` on the swapped batch with the swap's mask excluded: its batch / law / G-LIVE / gradient / p = 0 clauses, on a composed batch
      gradient   the loss and gradient of {logged, swap only, perturbation only (on the non-swapped rows), composed}: the swap moves them INSIDE the composed batch (composed vs perturbation-only), the
                 perturbation moves them inside it too (composed vs swap-only), and each alone moves them
      controls   (a) the same loss twice on the logged batch differs by EXACTLY 0.0; (b) perturbation p = 0 on the swapped batch: loss and gradient equal the swap-only ones EXACTLY; (c) both p = 0: the SAME batch
                 object back and loss / gradient equal the logged ones EXACTLY; (d) the frozen P10 prior's and (e) the world-model auxiliary's gradient differences are EXACTLY 0.0

    A violation is an AssertionError, which the caller turns into a refusal. DELETE the perturbation's apply on the composed path, or the swap's, or the exclusion, and a clause here reads 0 / a nonzero overlap."""
    if not comp.on:
        return {"ego_dagger_compose": "off", "pass": True, "overlap": 0, "skipped": "--ego-perturb-with-dagger not named: the two levers are not composed (the default run is the pre-change run)"}
    ego, dag = comp.ego, comp.dag
    step = None
    for k in range(-1, -65, -1):
        comp.apply(batch, k, ego_p=1.0, dag_p=COMPOSE_PROBE_DAG_P)
        if int(dag.last["n_selected"]) >= 1 and int(ego.last["n_selected"]) >= 1:
            step = k
            break
    if step is None:
        raise AssertionError("compose probe: no probe step in -1..-64 gives a batch with a swapped row AND a perturbed row (an own state with r' = t, and a logged v0 below vmax, on a row the swap left): "
                             "the composition cannot be seen on this batch")
    swapped = dag.apply(batch, step, p=COMPOSE_PROBE_DAG_P)                       # the swap alone ...
    sw = dag.last["sel"]
    composed = comp.apply(batch, step, ego_p=1.0, dag_p=COMPOSE_PROBE_DAG_P)      # ... and the composition, through the call the loop makes
    pe = ego.last["sel"]
    res = {"probe_step": step, "dagger_p_probe": COMPOSE_PROBE_DAG_P, "ego_p_probe": 1.0, "n_rows": int(sw.numel()), "n_swapped": int(sw.sum()), "n_perturbed": int(pe.sum()),
           "n_ego_eligible_nonswapped": int(ego.last["n_eligible"]), "overlap": int(comp.overlap), "overlap_recomputed": int((sw & pe).sum())}
    if res["overlap"] != 0 or res["overlap_recomputed"] != 0:
        raise AssertionError(f"compose probe: {res['overlap']} / {res['overlap_recomputed']} sample(s) are both swapped and perturbed: the overlap must be exactly 0")
    ego.apply(swapped, step, p=1.0)                                               # the counterfactual: WITHOUT the exclusion, how many swapped rows would the perturbation have hit
    res["n_swapped_rows_the_exclusion_protects"] = int((ego.last["sel"] & sw).sum())
    ego_only = ego.apply(batch, step, p=1.0, exclude=sw)                          # the perturbation alone, on the same non-swapped rows of the logged batch

    def row_same(x, y):                                                           # [B] bool: row equal (NaN == NaN)
        same = (x == y) | (torch.isnan(x) & torch.isnan(y)) if x.is_floating_point() else (x == y)
        return same.reshape(x.shape[0], -1).all(1)
    for k in D.EGO_PERTURB_REBUILT:
        if not bool(row_same(composed[k], swapped[k])[sw].all()):
            raise AssertionError(f"compose probe: {k} of a swapped row differs from the swap's own value: the perturbation touched a swapped sample")
        if not bool(row_same(composed[k], ego_only[k])[~sw].all()):
            raise AssertionError(f"compose probe: {k} of a non-swapped row differs from the perturbation-only value: the swap touched a row it did not select")
    res["ego_perturb_probe_on_composed"] = ego_perturb_probe(model, swapped, cfg, ego, exclude=sw)
    if res["ego_perturb_probe_on_composed"]["n_selected_at_p1"] != res["n_perturbed"]:
        raise AssertionError(f"compose probe: the existing ego probe selected {res['ego_perturb_probe_on_composed']['n_selected_at_p1']} rows on the swapped batch but the composition perturbs {res['n_perturbed']}: "
                             "the probe is not running on the composed selection (the swap's mask was not excluded)")

    m = copy.deepcopy(model).cpu()
    m.train()
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]

    def run(b):
        for _, p in named:
            p.grad = None
        for p in m.prior.parameters():
            p.grad = None
        torch.manual_seed(20261026)
        T = m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"compose probe: loss term {k} is not finite: {float(v)}")
        T["total"].backward()
        g = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
        gp = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in m.prior.parameters()])
        return float(T["total"].detach()), g, gp
    dsum = lambda ga, gb, pre="": float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))      # noqa: E731
    t0, g0, gp0 = run(batch)
    t0b, g0b, _ = run(batch)
    ts, gs, _ = run(swapped)
    te, ge, _ = run(ego_only)
    tc, gc, gpc = run(composed)
    b_ego0 = comp.apply(batch, step, ego_p=0.0, dag_p=COMPOSE_PROBE_DAG_P)       # control: the perturbation off, the swap on
    tz, gz, _ = run(b_ego0)
    b_both0 = comp.apply(batch, step, ego_p=0.0, dag_p=0.0)                      # control: both off
    tzz, gzz, _ = run(b_both0)
    res.update({"loss_logged": t0, "loss_swap_only": ts, "loss_ego_only": te, "loss_composed": tc,
                "grad_diff_swap_alone": dsum(gs, g0), "grad_diff_ego_alone": dsum(ge, g0), "grad_diff_swap_inside_composed": dsum(gc, ge), "grad_diff_ego_inside_composed": dsum(gc, gs),
                "loss_diff_swap_inside_composed": tc - te, "loss_diff_ego_inside_composed": tc - ts,
                "control_repeat_loss_diff": abs(t0b - t0), "control_repeat_grad_diff": dsum(g0, g0b),
                "control_ego_p0_loss_diff": abs(tz - ts), "control_ego_p0_grad_diff": dsum(gz, gs),
                "control_both_p0_same_object": b_both0 is batch, "control_both_p0_loss_diff": abs(tzz - t0), "control_both_p0_grad_diff": dsum(gzz, g0),
                "control_prior_grad_diff": float((gpc - gp0).abs().sum()), "control_wm_grad_diff": dsum(gc, g0, "wm.") if m.wm is not None else None})
    if res["control_repeat_loss_diff"] != 0.0 or res["control_repeat_grad_diff"] != 0.0:
        raise AssertionError(f"compose probe: the probe is not deterministic (the same loss twice differs by {res['control_repeat_loss_diff']} / {res['control_repeat_grad_diff']}): its differences mean nothing")
    if res["control_ego_p0_loss_diff"] != 0.0 or res["control_ego_p0_grad_diff"] != 0.0:
        raise AssertionError(f"compose probe: perturbation p = 0 on the swapped batch is not the swap alone (loss diff {res['control_ego_p0_loss_diff']}, grad diff {res['control_ego_p0_grad_diff']}): the control must read exactly 0.0")
    if not res["control_both_p0_same_object"] or res["control_both_p0_loss_diff"] != 0.0 or res["control_both_p0_grad_diff"] != 0.0:
        raise AssertionError(f"compose probe: both p = 0 is not the identity (same object {res['control_both_p0_same_object']}, loss diff {res['control_both_p0_loss_diff']}, grad diff {res['control_both_p0_grad_diff']}): the control must read exactly 0.0")
    if not (res["grad_diff_swap_inside_composed"] > 0.0 and res["loss_diff_swap_inside_composed"] != 0.0):
        raise AssertionError(f"compose probe: the swap does not move the composed batch's loss ({res['loss_diff_swap_inside_composed']}) / gradient ({res['grad_diff_swap_inside_composed']}): it does not reach the loss the trainer backpropagates")
    if not (res["grad_diff_ego_inside_composed"] > 0.0 and res["loss_diff_ego_inside_composed"] != 0.0):
        raise AssertionError(f"compose probe: the perturbation does not move the composed batch's loss ({res['loss_diff_ego_inside_composed']}) / gradient ({res['grad_diff_ego_inside_composed']}): it does not reach the loss the trainer backpropagates")
    if not (res["grad_diff_swap_alone"] > 0.0 and res["grad_diff_ego_alone"] > 0.0):
        raise AssertionError(f"compose probe: a lever alone does not move the gradient (swap {res['grad_diff_swap_alone']}, perturbation {res['grad_diff_ego_alone']})")
    if 0.0 != res["control_prior_grad_diff"]:
        raise AssertionError(f"compose probe: the frozen P10 prior's gradient moved with the composition ({res['control_prior_grad_diff']}): the control must read exactly 0.0")
    if res["control_wm_grad_diff"] not in (0.0, None):
        raise AssertionError(f"compose probe: the world-model auxiliary's gradient moved with the composition ({res['control_wm_grad_diff']}): act is its transition record and is kept, the control must read exactly 0.0")
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
    expect = {"ckpt", "goal", "lat", "lon", "cons", "ade", "fde", "acc", "total"}      # + acc: PREREG_V8F_OACC (always logged)
    if cfg.consistency:
        expect |= {f"pen_{r}" for r in M.RULES} | {f"pens_{r}" for r in M.RULES}
    if cfg.wm != "off":
        expect.add("wm")
    if cfg.level_head == "on":
        expect.add("level")                                              # PREREG_V8F_LEVEL
    if cfg.lead_head == "on":
        expect.add("lead")                                               # PREREG_LEVELHEAD_A2
    missing = sorted(expect - set(terms_seen))
    declared_dead = [n for n in acc if cfg.wm == "a" and ".idm_heads." in n]
    if cfg.o_interface == "cut":                  # PREREG_V8F_FLAT: O reads a constant code, so e_code.weight receives EXACTLY 0.0 (its bias still trains): declared dead, and G-LIVE REQUIRES the zero
        declared_dead += [n for n in acc if n == "e_code.weight"]
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
    if a.wm_windows == "wmonly":                                                 # PREREG_V8F_LADDER: the world-model-only sampler: every full hmax-frame window, no label requirement
        allw = WW.wm_windows(store, J, clips_all, F=cfg.hmax, stride=a.wm_win_stride)
        wins = WW.wm_windows(store, J, clips, F=cfg.hmax, stride=a.wm_win_stride)
        info.update({"n_windows_all_clips": len(allw), "n_windows": len(wins)})
        info["wm_windows"] = {"kind": "wmonly", "F": cfg.hmax, "stride": a.wm_win_stride, "n_windows": len(wins), "n_windows_all_clips": len(allw),
                              "t_min": min((w[1] for w in allw), default=None), "t_max": max((w[1] for w in allw), default=None), "pooled": None}      # `pooled` is filled from the BUILT dataset in main
    else:
        allw = D.train_windows(store, J, clips_all, hist=cfg.hist, start=cfg.window_start)
        wins = D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start)
        info.update({"n_windows_all_clips": len(allw), "n_windows": len(wins),
                     "window_start": D.effective_window_start(cfg.hist, cfg.window_start),         # what train_windows was asked for ...
                     "window_t_min": min((w[1] for w in allw), default=None)})                      # ... and what the built list actually starts at (read back)
    info["train_window_list_sha256"] = DG.windows_sha256(J["clip"], wins)                       # the TRAIN WINDOW list's digest, recorded on EVERY run (summary.json, beside train_sha256): the bank's `window_list_sha256` is this value, and the reader (K-B3.2) compares the two on both arms
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


# ------------------------------------------------------------------------------------------------ AMENDMENT_V8F_LORA_5: a resume across --accum / --enc-ckpt
def perf_strip(argv) -> list:
    """``argv`` with EXACTLY ``--accum <n>``, ``--enc-ckpt`` and ``--resume-perf-levers <path>`` removed (exact spellings: an abbreviation or ``--accum=n`` stays in and is therefore a difference)."""
    out, i, argv = [], 0, list(argv)
    while i < len(argv):
        if argv[i] in ("--accum", "--resume-perf-levers"):
            i += 2
        elif argv[i] == "--enc-ckpt":
            i += 1
        else:
            out.append(argv[i])
            i += 1
    return out


def perf_resume_plan(a, argv) -> dict:
    """The previous argv named by ``--resume-perf-levers`` against this one. Raises ValueError (or OSError / KeyError / SystemExit from argparse) when it is not a perf-lever-only change."""
    d = json.load(open(a.resume_perf_levers, encoding="utf-8"))
    prev = d["argv"] if isinstance(d, dict) else d
    if not (isinstance(prev, list) and prev and all(isinstance(t, str) for t in prev)):
        raise ValueError("the file holds no argv list")
    ps, ns = perf_strip(prev), perf_strip(argv)
    if ps != ns:
        i = next((k for k in range(min(len(ps), len(ns))) if ps[k] != ns[k]), min(len(ps), len(ns)))
        raise ValueError(f"the argvs differ beyond --accum / --enc-ckpt: previous {ps[max(0, i - 1):i + 2]} vs this {ns[max(0, i - 1):i + 2]} (after removing those flags, token {i})")
    p = build_parser().parse_args(list(prev))
    return {"prev_argv": prev, "prev": {"accum": int(p.accum), "enc_ckpt": bool(p.enc_ckpt)}, "new": {"accum": int(a.accum), "enc_ckpt": bool(a.enc_ckpt)},
            "prev_argv_sha": argv_sha(prev), "new_argv_sha": argv_sha(argv), "stripped_argv_sha": argv_sha(ns)}


def bad_perf_resume(a, argv) -> list:
    """G-DVB for ``--resume-perf-levers`` (empty = pass; the flag off builds nothing and asks nothing)."""
    if a.resume_perf_levers is None:
        return []
    try:
        plan = perf_resume_plan(a, argv)
    except (ValueError, OSError, KeyError, TypeError, SystemExit) as e:
        return [f"--resume-perf-levers {a.resume_perf_levers}: {type(e).__name__}: {str(e)[:400]}"]
    bad = []
    if a.encoder != "lora":
        bad.append("--resume-perf-levers declared but --encoder is not lora: --accum and --enc-ckpt build nothing there")
    if a.no_resume:
        bad.append("--resume-perf-levers declared with --no-resume: it is a lever of a resume")
    if plan["prev"] == plan["new"]:
        bad.append(f"--resume-perf-levers declared but the previous argv has the same --accum / --enc-ckpt {plan['new']}: a flag that is declared and does nothing")
    return bad


def perf_resume_record(a, argv, ck_sha, step):
    """None unless ``--resume-perf-levers`` is set AND the checkpoint was written by exactly the previous argv; then the record that goes into train_log.jsonl."""
    if a.resume_perf_levers is None:
        return None
    try:
        plan = perf_resume_plan(a, argv)
    except (ValueError, OSError, KeyError, TypeError, SystemExit):
        return None
    if ck_sha != plan["prev_argv_sha"]:
        return None
    return {"from_step": step, "prev": plan["prev"], "new": plan["new"], "prev_argv_sha": plan["prev_argv_sha"], "new_argv_sha": plan["new_argv_sha"], "stripped_argv_sha": plan["stripped_argv_sha"],
            "prev_argv_file": os.path.abspath(a.resume_perf_levers)}


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if a.p10 is None:
        a.p10 = os.path.join(PKG, "raw", "p10_yaw" if a.kin_mode == "yaw" else "p10", "p10_prior.pt")
    _pp = pool_cache_problems(a)                                     # PREREG_V8F_WMPOOL: refuse before anything is built (the adapter-sha half runs in G-DVB, where the built model exists)
    if _pp:
        raise SystemExit("ZZDVB-REFUSEDZZ " + "; ".join(_pp))
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    cfg = make_config(a)
    ego = make_ego_perturb(a)                                        # O-RECOVER (PREREG_V8F_HACC_MC section 7): off by default; the object the loop applies, G-DVB reads back and the probes exercise
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
    elif a.wm_windows == "wmonly":                                   # PREREG_V8F_LADDER: the world-model-only dataset (pooled path = the token mean; off = the cached tokens)
        ds = WW.V8FMiniWMWindows(store, J, wins, F=cfg.hmax, pooled=(a.wm_pooled == "on"), lru=info["lru"], zcache=a.wm_zcache, pool=a.wm_pool)
        info["wm_windows"]["zcache"] = ds.zcache
        info["wm_windows"]["zcache_missing"] = len(ds.zcache_missing(clips)) if ds.zcache else 0
        if ds.zcache and not info["wm_windows"]["zcache_missing"]:    # windows holding a frame the cache marks not ok are dropped here, the same list for every arm
            n0 = len(wins)
            wins = ds.ok_windows()
            info["wm_windows"].update({"n_windows": len(wins), "n_windows_before_ok": n0, "n_windows_not_ok": n0 - len(wins)})
            info["n_windows"] = len(wins)
            info["train_window_list_sha256"] = DG.windows_sha256(J["clip"], wins)
        if ds.zcache and not a.selfcheck_only:                        # the action arrays go to RAM at start-up, so no clip file is read during training
            info["wm_windows"]["prewarm_clips"] = ds.prewarm()
        info["wm_windows"]["pooled"] = ds.pooled                     # read back from the BUILT dataset (bad_wm_ladder compares it)
        info["wm_windows"]["pool"] = ds.pool                         # PREREG_V8F_WMPOOL: read back from the BUILT dataset (bad_wm_pool compares it)
        if ds.pool == "att25" and ds.windows and not info["wm_windows"]["zcache_missing"]:
            info["wm_windows"]["zcell_shape"] = list(ds.item(*ds.windows[0])["zcell"].shape)   # PREREG_V8F_WMPOOL: the first REAL item's cell shape
        info["wm_windows"]["F"] = ds.F
    else:
        ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                              window_start=cfg.window_start)
    if a.wm_windows != "wmonly":
        info["ds_window_start"] = ds.window_start                   # the held-age clamp's origin, read back from the BUILT dataset (G-DVB compares it)
    LL = make_lead_labels(a, cfg, J, info)                          # PREREG_LEVELHEAD_A2: the lead head's label arrays (None with the head off: nothing read, nothing built)
    prior = M.load_p10(a.p10) if not a.synthetic else M.P10Prior.synthetic(a.seed)
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, prior, mu, sd).to(dev)
    if a.init_ckpt:                                                  # O-RECOVER's start state: the MODEL weights of an earlier checkpoint (before the optimiser, the probes and the gate); a resume of THIS run wins
        try:
            _apply = not (os.path.exists(os.path.join(a.out, "ckpt_last.pt")) and not a.no_resume)
            if a.wm_only:                                            # PREREG_V8F_MRATE: the non-WM tensors load strictly and bit-equal, hmax is not compared, the WM is re-initialised under --seed
                info["init_ckpt"] = info["wm_only"] = load_wm_only_init(a.init_ckpt, model, cfg, a.seed, apply=_apply)
            else:
                info["init_ckpt"] = load_init_ckpt(a.init_ckpt, model, cfg, apply=_apply)
        except Exception as e:                                       # noqa: BLE001  a missing / foreign / mismatched checkpoint is a REFUSAL, not a crash
            raise SystemExit(f"ZZDVB-REFUSEDZZ init-ckpt {a.init_ckpt}: {type(e).__name__}: {str(e)[:600]}")
    if a.wm_only:
        freeze_all_but_wm(model)                                     # PREREG_V8F_MRATE: BEFORE the optimiser below (its decay / nodecay lists filter on requires_grad); bad_wm_only reads the frozen set back
    dag = make_dagger(a, J, wins, clips, info)                       # PREREG_V8F_DAGGER: off by default (no bank read); with a bank: load + K-B2 + K-B3 + COVERAGE here, a refusal stops the launch before the gate
    rcv = make_recover(a, J, wins, store, clips, info)               # PREREG_RECOVER_FT: off by default (--recover-p 0: no bank read, no table built); with the empirical law: bank load + K-B3 + the deviation table from THIS trainer's TRAIN store here, a refusal stops the launch before the gate
    t2b = make_t2bank(a, info)                                       # PREREG_T2_DAGGER: off by default (--t2-bank-p 0: no bank read); with a bank: load + sha256 here, a refusal stops the launch before the gate
    comp = make_compose(a, ego, dag)                                 # AMEND1 section 1: off by default; with --ego-perturb-with-dagger the loop calls THIS (swap first, then the perturbation on the samples the swap left)
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
    lr_at = LRSchedule(a.lr, a.warmup, total)                        # the loop's schedule, built BEFORE the gate so that G-DVB can read --warmup back from it
    bad = declared_vs_built(a, cfg, model, opt, info, enc, ego, lr_at, dag, comp, rcv, t2b)
    bad += bad_perf_resume(a, argv)                                  # AMENDMENT_V8F_LORA_5: --resume-perf-levers read back (the named previous argv differs from this one in --accum / --enc-ckpt only)
    if bad:
        raise SystemExit("ZZDVB-REFUSEDZZ declared-vs-built: " + "; ".join(bad))
    chunk = enc_chunk(a, cfg)
    ident = None
    probe_wm = None
    if enc is None and a.wm_windows == "wmonly":                      # PREREG_V8F_LADDER: the WM batch for G-LIVE (first 16 windows); the planner identity and the admissibility check need a LABELLED batch, built from the labelled list
        probe_wm = WW.collate_wm([ds.item(*w) for w in next(sampler.epoch(0))][:LIVE_PROBE_WINDOWS_WM])
        ds_reg = D.V8FMiniWindows(store, J, D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start), hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                                  window_start=cfg.window_start)
        probe = D.collate([ds_reg.item(*w) for w in ds_reg.windows[:LIVE_PROBE_WINDOWS_WM]])
    elif enc is None:
        probe = D.collate([ds.item(*w) for w in next(sampler.epoch(0))])
    if enc is None:
        if cfg.launch_intent_w > 0.0:                    # PREREG_V8F_LAUNCH_INTENT: every probe calls model.loss on this batch, and a lever-ON loss reads b["launch"] (the loop's own call, on the LOGGED state)
            probe["launch"] = launch_stage(probe)
        if cfg.level_head == "on":                       # PREREG_V8F_LEVEL: every probe calls model.loss on this batch, and a lever-ON loss reads level_y / level_ok (the loop's own call, on the LOGGED state)
            probe["level_y"], probe["level_ok"] = level_stage(probe)
        if cfg.lead_head == "on":                        # PREREG_LEVELHEAD_A2: as above (the lead labels, looked up by the probe's own clip index and step)
            probe["lead_y"], probe["lead_has"], probe["lead_near"] = lead_stage(probe, LL)
        if a.wm_only:                                    # PREREG_V8F_MRATE: G-LIVE of the WM-only mode + the planner identity (the other probes need gradients on the frozen readers and are not run)
            live = wm_only_live_check(model.cpu(), probe if probe_wm is None else probe_wm, cfg)
            pid = planner_identity_probe(model.cpu(), probe, cfg)
        else:
            live = live_check(model.cpu(), probe, cfg)
    else:
        ds_probe = D.V8FMiniFrameWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                                         window_start=cfg.window_start, keep_cached=True)
        probe_f = D.collate_frames([ds_probe.item(*w) for w in next(sampler.epoch(0))][:LIVE_PROBE_WINDOWS_LORA], fstore)
        if cfg.launch_intent_w > 0.0:                    # PREREG_V8F_LAUNCH_INTENT: as above (the cached tokens stay in `probe`; the mask comes from the logged kin3 and the GT)
            probe_f["launch"] = launch_stage(probe_f)
        if cfg.level_head == "on":                       # PREREG_V8F_LEVEL: as above
            probe_f["level_y"], probe_f["level_ok"] = level_stage(probe_f)
        if cfg.lead_head == "on":                        # PREREG_LEVELHEAD_A2: as above
            probe_f["lead_y"], probe_f["lead_has"], probe_f["lead_near"] = lead_stage(probe_f, LL)
        probe = {k: v for k, v in probe_f.items() if k not in ("frames_u8", "frame_map")}      # the cached tokens stay in `probe` (admissibility, identity)
        live = live_check_lora(model, enc, probe_f, cfg, a, dev, chunk)
        ident = identity_probe(enc, probe_f, dev, chunk)
    model.to(dev)
    if a.wm_only:                                                          # PREREG_V8F_MRATE: the readers, heads and planner are frozen and never run; the probes that need their gradients are skipped (and recorded as skipped)
        mix = ocut = accp = egop = dagp = recp = t2p = lvp = ldp = {"skipped": "wm_only"}
        lip = {"skipped": "wm_only", "launch_n": 0, "launch_eligible": 0}
        cmpp = {"skipped": "wm_only", "overlap": 0}
        adm = M.assert_no_situation_classifier_input(model.cpu().eval(), probe)
        model.to(dev).train()
    else:
        try:
            mix = code_mix_probe(model, probe, cfg)           # PREREG_V8F_MIXO: the joint structure of the code the BUILT model hands O is the declared one (whole: all-or-nothing; kind: partial)
        except AssertionError as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ declared-vs-built: {e}")
        try:
            ocut = o_interface_probe(model, probe, cfg)       # PREREG_V8F_FLAT: what the BUILT model's O is handed under a perturbation of everything above it (cut: constants, traj60 invariant; full: it moves)
        except AssertionError as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ declared-vs-built: {e}")
        try:
            accp = acc_probe(model, probe, cfg)               # PREREG_V8F_OACC: the BUILT loss moves by exactly w_acc x the acceleration term, and O's head receives its gradient (a deleted line reads 0.0 and is refused)
        except AssertionError as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ declared-vs-built: {e}")
        adm = M.assert_no_situation_classifier_input(model.cpu().eval(), probe)
        model.to(dev).train()
        try:                                                  # after the model is back on its device in train mode: the probe sees the state the acc probe saw
            egop = ego_perturb_probe(model, probe, cfg, ego)  # O-RECOVER: G-LIVE on a PERTURBED batch; the perturbed tensors reach the loss gradient, a p = 0 control reads EXACTLY 0.0 (a deleted apply line is refused)
        except AssertionError as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ ego-perturb: {e}")
        try:
            dagp = dagger_probe(model, probe, cfg, dag)       # PREREG_V8F_DAGGER: G-LIVE on a SWAPPED batch; the own-state tensors reach the loss gradient, a p = 0 control reads EXACTLY 0.0 (a deleted apply line is refused)
        except (AssertionError, ValueError) as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ dagger: {e}")
        try:
            cmpp = ego_dagger_compose_probe(model, probe, cfg, comp)      # AMEND1 section 1: G-LIVE on a COMPOSED batch; both levers reach the loss gradient, the p = 0 controls read EXACTLY 0.0, the overlap is exactly 0
        except (AssertionError, ValueError) as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ ego-dagger-compose: {e}")
        try:
            t2p = t2bank_probe(model, probe, cfg, t2b)        # PREREG_T2_DAGGER: G-LIVE on an all-bank batch; every masked term reads EXACTLY 0.0, the target law reaches the gradient, a p = 0 control is the same batch object
            t2b.opened.clear()                                # the probe's draws are not training draws: the opened-scene record starts empty
        except (AssertionError, ValueError) as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ t2-bank: {e}")
        try:
            recp = recover_probe(model, probe, cfg, rcv)      # PREREG_RECOVER_FT: G-LIVE on a RECOVERED batch; the perturbed tensors AND the re-expressed target reach the loss gradient, a p = 0 control reads EXACTLY 0.0 (a deleted apply line is refused)
        except (AssertionError, ValueError) as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ recover: {e}")
        try:
            lip = launch_intent_probe(model, probe, cfg)      # PREREG_V8F_LAUNCH_INTENT: G-LIVE on the loop's own mask (launch_stage); the term reaches the loss on the rows the mask names, O's head gets its gradient, an all-False mask reads EXACTLY 0.0 (off: nothing runs)
        except (AssertionError, ValueError) as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ launch-intent: {e}")
        try:
            lvp = level_head_probe(model, probe, cfg)         # PREREG_V8F_LEVEL: G-LIVE on the loop's own label call (level_stage); the term reaches the loss with its weight, the head gets its gradient, the detach control reads EXACTLY 0.0, the token reaches O, the head is vision-only (off: nothing runs)
        except (AssertionError, ValueError) as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ level-head: {e}")
        try:
            ldp = lead_head_probe(model, probe, cfg)          # PREREG_LEVELHEAD_A2: G-LIVE on the loop's own label call (lead_stage); the term reaches the loss with its weight, BOTH outputs get their gradient, the detach control reads EXACTLY 0.0, the lead output reaches O, the head is vision-only (off: nothing run)
        except (AssertionError, ValueError) as e:
            raise SystemExit(f"ZZDVB-REFUSEDZZ lead-head: {e}")
    levers = model.built_levers()
    levers["ego_perturb"] = ego.built_levers()                        # recorded like every other lever (selfcheck.json, summary.json, the checkpoint)
    levers["dagger"] = dag.built_levers()                             # PREREG_V8F_DAGGER: what was BUILT (the bank's path and sha256 included) ...
    levers["dagger_p"], levers["dagger_bank_sha256"] = dag.p, dag.bank_sha256          # ... and the two values the reader keys on, flat (0.0 / None when the lever is off)
    levers["dagger_bank_sha256_declared"] = a.dagger_bank_sha256 or None              # ... and the hash the argv DECLARED (--dagger-bank-sha256); the gate has just required it to equal the built one (None when the lever is off)
    levers["recover"] = rcv.built_levers()                             # PREREG_RECOVER_FT: what was BUILT (the LOADED bank's path and sha256 included; recover_p 0.0 and every other lever at its default when the lever is off) ...
    if t2b is not None and t2b.active:                                # PREREG_T2_DAGGER: lever-ON launches only (a flag-off selfcheck.json keeps its key set): what the BUILT swap does,
        levers["t2_bank"] = t2b.built_levers()                        # ... read by dt2_check.py gate (p, seed, target, rejoin steps, the loaded bank sha256)
    levers["recover_table"] = (info.get("recover") or {}).get("table")  # ... and the deviation table's build summary (n, content sha256, means, share of records with logged v in [3, 8]); None when the lever is off (the card's KR3 reads it)
    levers["ego_perturb_with_dagger"] = comp.on                        # AMEND1 section 1: what was BUILT (False unless --ego-perturb-with-dagger) ...
    levers["ego_perturb_stream"] = comp.ego.stream                     # ... and the perturbation's generator stream: 0 = the original, D.EGO_PERTURB_COMPOSED_STREAM = domain-separated from the swap's (composed arms only)
    levers["ego_dagger_overlap_n"] = cmpp["overlap"]                   # ... and the composed probe's measured overlap on the probe batch (0 when off; the run total is summary.json's ego_dagger_overlap_n)
    levers["launch_intent"] = {"launch_intent_w": cfg.launch_intent_w, "v0_max": M.LAUNCH_V0_MAX, "dv": M.LAUNCH_DV, "dt": M.LAUNCH_DT, "knots": [M.LAUNCH_KNOT_LO, M.LAUNCH_KNOT_HI], "axis": "along (x of the ego frame at NOW)",
                               "mask_from": "the LOGGED kin3[:, 0] and the GT future, taken before any ego stage"}   # PREREG_V8F_LAUNCH_INTENT: what was BUILT (the definition the mask uses) ...
    levers["launch_n"], levers["launch_eligible"] = lip["launch_n"], lip["launch_eligible"]          # ... and the probe batch's natural launch / GT-valid row counts (0 / 0 when off; the run totals are summary.json's launch_n / launch_eligible, present when the lever is on)
    levers["init_ckpt"] = info.get("init_ckpt")
    if a.wm_strides is not None:                                         # PREREG_V8F_LFS: lever-NAMED launches only (a flag-off selfcheck.json keeps its key set): declared vs the BUILT strides and the transitions per level
        levers["wm_strides_readback"] = wm_strides_readback(a, cfg, model)
    if a.wm_pool != build_parser().get_default("wm_pool"):             # PREREG_V8F_WMPOOL: lever-NAMED launches only
        levers["wm_pool_readback"] = wm_pool_readback(a, cfg, model, info)
    if wm_ladder_named(a):                                              # PREREG_V8F_LADDER: lever-NAMED launches only (a flag-off selfcheck.json keeps its key set): declared vs the BUILT measurements
        levers["wm_ladder_readback"] = wm_ladder_readback(a, cfg, model, info)
    if a.wm_only:                                                      # WM-only arms only (a flag-off run record keeps its key set)
        levers["wm_only"] = {"wm_only": True, "wm_k": a.wm_k, "n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "n_wm": int(sum(p.numel() for n, p in model.named_parameters() if is_wm_param(n))),
                             "init": info.get("wm_only")}
    levers["lr_schedule"] = lr_at.built_levers()                       # the built warmup / peak / total the loop calls (G-DVB read --warmup back from it)
    if enc is not None:
        levers["encoder"] = enc.built_levers()
        levers["accum"] = {"accum": a.accum, "micro_windows": a.batch // a.accum, "chunk_frames": chunk}
    if a.resume_perf_levers is not None:                               # AMENDMENT_V8F_LORA_5: lever-ON launches only (a flag-off selfcheck.json keeps its key set): previous vs BUILT values
        pl = perf_resume_plan(a, argv)
        levers["perf_resume"] = {"prev": pl["prev"], "new_built": {"accum": levers["accum"]["accum"], "enc_ckpt": levers["encoder"]["checkpointing"]}, "prev_argv_sha": pl["prev_argv_sha"],
                                 "stripped_argv_sha": pl["stripped_argv_sha"]}
    sc ={"declared_vs_built": "PASS", "live": live, "code_mix_probe": mix, "o_interface_probe": ocut, "acc_probe": accp, "admissibility": adm, "levers": levers,
          "steps_total": total, "S_steps_per_epoch_100pct": S, "info": info, "argv_sha": argv_sha(argv)}
    sc["ego_perturb_probe"] = egop
    sc["dagger_probe"] = dagp
    sc["ego_dagger_compose_probe"] = cmpp
    sc["recover_probe"] = recp
    sc["t2bank_probe"] = t2p
    sc["launch_intent_probe"] = lip
    if a.wm_only:                                                       # WM-only arms only: a flag-off selfcheck.json keeps its key set
        sc["planner_identity_probe"] = pid
    if cfg.level_head == "on":                                          # lever-ON arms only: a flag-off selfcheck.json keeps its key set
        sc["level_head_probe"] = lvp
    if cfg.lead_head == "on":                                           # PREREG_LEVELHEAD_A2: lever-ON arms only (a flag-off selfcheck.json keeps its key set)
        sc["lead_head_probe"] = ldp
        sc["lead_labels"] = info["lead_labels"]
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
    ego_tot = {"rows": 0, "eligible": 0, "n": 0, "nav_kept": 0}   # O-RECOVER's run totals (samples seen, eligible = logged v0 < vmax, perturbed, nav frames left as logged); carried in the checkpoint so a resumed run keeps counting
    ego_int = {"rows": 0, "eligible": 0, "n": 0, "nav_kept": 0}   # ... and since the last log line
    dag_tot = {"rows": 0, "eligible": 0, "n": 0, "nav_kept": 0}   # PREREG_V8F_DAGGER's run totals (samples seen, eligible = an own state with r' = t, swapped, nav frames left as logged); carried in the checkpoint
    dag_int = {"rows": 0, "eligible": 0, "n": 0, "nav_kept": 0}   # ... and since the last log line
    rec_tot = {"rows": 0, "eligible": 0, "n": 0, "changed": 0}   # PREREG_RECOVER_FT's run totals (samples seen, eligible = NOW a real kinematic row, perturbed, rows whose kin3 the BATCH-level predicate saw differ); exactly 0 at --recover-p 0; carried in the checkpoint of a lever-ON arm only
    t2_tot, t2_int = {"rows": 0, "n": 0}, {"rows": 0, "n": 0}     # PREREG_T2_DAGGER: samples seen / replaced by a bank row (total and since the last log line); exactly 0 at --t2-bank-p 0
    rec_int = {"rows": 0, "eligible": 0, "n": 0, "changed": 0}   # ... and since the last log line
    ov_tot = {"n": 0}                                             # AMEND1 section 1: samples both swapped and perturbed (exactly 0), carried in the checkpoint of a composed arm only
    ov_int = {"n": 0}                                             # ... and since the last log line
    launch_on = cfg.launch_intent_w > 0.0                          # PREREG_V8F_LAUNCH_INTENT: off (the default) = the loop runs no new line
    level_on = cfg.level_head == "on"                              # PREREG_V8F_LEVEL: off (the default) = the loop runs no new line
    lead_on = cfg.lead_head == "on"                                # PREREG_LEVELHEAD_A2: off (the default) = the loop runs no new line
    launch_tot = {"rows": 0, "eligible": 0, "n": 0}                # ... run totals: rows seen, GT-valid rows (the along-track label exists), GT-launch rows; carried in the checkpoint of an arm that has the lever ON only
    launch_int = {"rows": 0, "eligible": 0, "n": 0}                # ... and since the last log line
    perf_rec = None                                                  # AMENDMENT_V8F_LORA_5: set only by a resume across --accum / --enc-ckpt under --resume-perf-levers
    if os.path.exists(ck_path) and not a.no_resume:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if ck["argv_sha"] != argv_sha(argv):
            perf_rec = perf_resume_record(a, argv, ck["argv_sha"], ck["step"])      # None unless the flag is set and the checkpoint is exactly the previous argv's
            if perf_rec is None:
                raise SystemExit("ZZRESUME-REFUSEDZZ the checkpoint was written by a different argv")
        if comp.on and int(ck.get("ego_perturb_stream", 0)) != ego.stream:        # a composed checkpoint from before the stream separation (no key = stream 0) must not be continued on another generator
            raise SystemExit(f"ZZRESUME-REFUSEDZZ the checkpoint drew the perturbation from stream {int(ck.get('ego_perturb_stream', 0))} but this build draws stream {ego.stream}: a resume would switch generators mid-run")
        model.load_state_dict(ck["model"])
        if enc is not None:
            enc.load_lora_state_dict(ck["lora"])
        opt.load_state_dict(ck["opt"])
        step, ep, bi = ck["step"], ck["epoch"], ck["batch_in_epoch"]
        ego_tot.update(ck.get("ego_totals") or {})
        dag_tot.update(ck.get("dagger_totals") or {})
        ov_tot.update(ck.get("ego_dagger_overlap") or {})
        rec_tot.update(ck.get("recover_totals") or {})
        launch_tot.update(ck.get("launch_totals") or {})
        print(f"[resume] step {step} epoch {ep} batch {bi}", flush=True)
    elif a.resume_perf_levers is not None:
        raise SystemExit(f"ZZRESUME-REFUSEDZZ --resume-perf-levers resumes a checkpoint and there is none at {ck_path}")

    def save(final=False):
        blob = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": ep, "batch_in_epoch": bi,
                "argv_sha": argv_sha(argv), "cfg": cfg.__dict__, "p10_path": os.path.abspath(a.p10) if not a.synthetic else "",
                "grid": a.grid, "arm": a.arm, "seed": a.seed, "cons_mu": mu, "cons_sd": sd,
                "cfg_tuple_fields": {"wm_strides": list(cfg.wm_strides)},
                "ego_perturb": ego.built_levers(), "init_ckpt": info.get("init_ckpt"), "ego_totals": dict(ego_tot),
                "dagger": dag.built_levers(), "dagger_totals": dict(dag_tot)}          # O-RECOVER: what was BUILT, beside the config, and the counts so far
        if comp.on:                                                     # composed arms only: a flag-off checkpoint keeps its key set
            blob["ego_perturb_with_dagger"], blob["ego_dagger_overlap"], blob["ego_perturb_stream"] = True, dict(ov_tot), ego.stream
        if rcv.active:                                                  # lever-ON arms only: a flag-off checkpoint keeps its key set
            blob["recover"], blob["recover_totals"] = rcv.built_levers(), dict(rec_tot)
        if launch_on:                                                   # lever-ON arms only: a flag-off checkpoint keeps its key set
            blob["launch_totals"] = dict(launch_tot)
        if a.wm_only:                                                   # WM-only arms only
            blob["wm_only"] = True
        if enc is not None:             # `model` keeps B's layout (the unchanged v8f_mini_eval.py loads it); the LoRA matrices travel beside it
            blob["lora"], blob["encoder_levers"] = enc.lora_state_dict(), enc.built_levers()
        torch.save(blob, ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)

    log = open(os.path.join(a.out, "train_log.jsonl"), "a", encoding="utf-8")
    if perf_rec is not None:                                           # AMENDMENT_V8F_LORA_5: the one record of the lever change (old and new values), with the resume step as its `step`
        log.write(json.dumps({"step": step, "perf_resume": perf_rec}) + "\n"); log.flush()
        print(f"[perf_resume] step {step}: {perf_rec['prev']} -> {perf_rec['new']}", flush=True)
    amp = a.amp and dev.type == "cuda"
    t0 = time.time()
    done = step >= total
    while not done:
        gen_batches = sampler.epoch(ep)
        for _ in range(bi):                                    # fast-forward within the epoch (indices only)
            next(gen_batches, None)
        if enc is None:
            pf = Prefetch(gen_batches, (lambda idx: WW.collate_wm([ds.item(*w) for w in idx])) if a.wm_windows == "wmonly" else (lambda idx: D.collate([ds.item(*w) for w in idx])), depth=3)
        else:
            pf = Prefetch(gen_batches, lambda idx: D.collate_frames([ds.item(*w) for w in idx], fstore), depth=2)
        for b in pf:
            fr = (b.pop("frames_u8"), b.pop("frame_map")) if enc is not None else None      # the frames stay on the host; the encoder moves them chunk by chunk
            if a.wm_windows != "wmonly":                           # PREREG_V8F_LADDER: the label stages read labelled-batch keys (kin3, p10f, ci/t labels); a world-model-only batch carries none (--wm-only forces every one of them off)
                if launch_on:                                          # PREREG_V8F_LAUNCH_INTENT: the GT-launch mask from the LOGGED v0 and the GT future, taken BEFORE any ego stage rebuilds kin3 (a swapped sample's kin3 is the own state)
                    lmask = launch_stage(b)
                if level_on:                                           # PREREG_V8F_LEVEL: the label is the LOGGED speed at NOW, taken BEFORE any ego stage rebuilds kin3 (the head sees the logged frame)
                    lvl_lab = level_stage(b)
                if lead_on:                                            # PREREG_LEVELHEAD_A2: the lead label of the LOGGED frame (clip index and step), taken BEFORE any ego stage (a swapped sample's ci / t is the own one)
                    lead_lab = lead_stage(b, LL)
                if comp.on:                                            # AMEND1 section 1 (--ego-perturb-with-dagger): the swap is drawn first, the perturbation only on the samples the swap did not replace; the overlap is measured (exactly 0)
                    b = comp.apply(b, step)
                    for c_ in (ov_int, ov_tot):
                        c_["n"] += comp.overlap
                else:                                                  # every other argv: the two original lines, in their original order
                    b = ego.apply(b, step)                                 # O-RECOVER: the kinematic inputs of the selected samples are rebuilt from a perturbed ego state; targets and labels are untouched (off: `b` itself)
                    b = dag.apply(b, step)                                 # PREREG_V8F_DAGGER: the ego tensors of the selected samples are the model's OWN T1-kin state at the matched route point; targets and labels stay the human's (off: `b` itself)
                if rcv.active:                                         # PREREG_RECOVER_FT (exclusive with the two stages above, refused otherwise): the ego tensors AND the target of the selected samples; p = 0 runs none of these lines
                    b_logged = b
                    b = rcv.apply(b, step)                                 # the ONE loop line that applies the recovery perturbation (a deleted line is caught by the guard below, by the counts and by the batch-level predicate)
                    if rcv.last is None or rcv.last.get("step") != step or (rcv.last["n_selected"] > 0 and b is b_logged):
                        raise SystemExit(f"ZZRECOVER-NOT-APPLIEDZZ step {step}: --recover-p {rcv.p} is on but the batch the loss receives is the logged one (n_selected {None if rcv.last is None else rcv.last.get('n_selected')}): a lever that is declared and does nothing")
                    for c_ in (rec_int, rec_tot):                      # the control counts: samples seen, eligible (NOW a real kinematic row), perturbed (n_selected), and the batch-level count of rows whose kin3 the loss's batch holds changed
                        c_["rows"] += rcv.last["n_rows"]
                        c_["eligible"] += rcv.last["n_eligible"]
                        c_["n"] += rcv.last["n_selected"]
                        c_["changed"] += int((b["kin3"] != b_logged["kin3"]).reshape(b["kin3"].shape[0], -1).any(1).sum())
                for c_ in (ego_int, ego_tot):                          # the control counts: samples seen, eligible (logged v0 < vmax), perturbed (exactly 0 when the lever is off)
                    c_["rows"] += ego.last["n_rows"]
                    c_["eligible"] += ego.last["n_eligible"]
                    c_["n"] += ego.last["n_selected"]
                    c_["nav_kept"] += ego.last["ego_perturb_nav_kept"]
                for c_ in (dag_int, dag_tot):                          # the control counts: samples seen, eligible (an own state with r' = t), swapped (exactly 0 when the lever is off), nav frames left as logged
                    c_["rows"] += dag.last["n_rows"]
                    c_["eligible"] += dag.last["n_eligible"]
                    c_["n"] += dag.last["n_selected"]
                    c_["nav_kept"] += dag.last["dagger_nav_kept"]
                if launch_on:                                          # ... and attached AFTER the stages (they return their own batch dict): the loss reads b["launch"]
                    b["launch"] = lmask
                    for c_ in (launch_int, launch_tot):                # the control counts: rows seen, GT-valid rows (the along-track label exists), GT-launch rows
                        c_["rows"] += int(lmask.shape[0])
                        c_["eligible"] += int(b["gt_ok"].sum())
                        c_["n"] += int(lmask.sum())
                if level_on:                                           # ... and attached AFTER the stages (they return their own batch dict): the loss reads b["level_y"], b["level_ok"]
                    b["level_y"], b["level_ok"] = lvl_lab
                if lead_on:                                            # PREREG_LEVELHEAD_A2: attached AFTER the stages: the loss reads b["lead_y"], b["lead_has"], b["lead_near"]
                    b["lead_y"], b["lead_has"], b["lead_near"] = lead_lab
                if t2b.active:                                         # PREREG_T2_DAGGER (exclusive with every stage above): whole bank samples replace a share p of the rows AFTER the label stages (it masks launch / level / lead itself); p = 0 runs none of these lines
                    b_pre = b
                    b = t2b.apply(b, step)
                    if t2b.last is None or t2b.last.get("step") != step or (t2b.last["n_selected"] > 0 and b is b_pre) or "aux_ok" not in b:
                        raise SystemExit(f"ZZT2BANK-NOT-APPLIEDZZ step {step}: --t2-bank-p {t2b.p} is on but the batch the loss receives is the logged one (n_selected {None if t2b.last is None else t2b.last.get('n_selected')}): a lever that is declared and does nothing")
                    for c_ in (t2_int, t2_tot):
                        c_["rows"] += t2b.last["n_rows"]
                        c_["n"] += t2b.last["n_selected"]
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
                    out = model.wm_only_loss(b) if a.wm_only else model.loss(b, p_gt=p_gt, gen=None)       # PREREG_V8F_MRATE: ONLY the WM terms (the adapter is detached inside, the rest is frozen and not run)
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
                rec["ego_perturb_n"], rec["ego_perturb_eligible"], rec["ego_perturb_rows"] = ego_int["n"], ego_int["eligible"], ego_int["rows"]      # per log interval (the steps since the last line)
                rec["ego_perturb_nav_kept"] = ego_int["nav_kept"]
                ego_int.update(rows=0, eligible=0, n=0, nav_kept=0)
                if dag.active:
                    rec["dagger_n"], rec["dagger_eligible"], rec["dagger_rows"], rec["dagger_nav_kept"] = dag_int["n"], dag_int["eligible"], dag_int["rows"], dag_int["nav_kept"]          # per log interval
                dag_int.update(rows=0, eligible=0, n=0, nav_kept=0)
                if comp.on:
                    rec["ego_dagger_overlap_n"] = ov_int["n"]               # AMEND1 section 1: per log interval (exactly 0)
                ov_int.update(n=0)
                if t2b.active:                                              # PREREG_T2_DAGGER: per log interval (a flag-off log line keeps its key set)
                    rec["t2_bank_n"], rec["t2_bank_rows"] = t2_int["n"], t2_int["rows"]
                    t2_int.update(rows=0, n=0)
                if rcv.active:                                              # PREREG_RECOVER_FT: per log interval (a flag-off log line keeps its key set)
                    rec["recover_n"], rec["recover_eligible"], rec["recover_rows"], rec["recover_changed"] = rec_int["n"], rec_int["eligible"], rec_int["rows"], rec_int["changed"]
                    rec_int.update(rows=0, eligible=0, n=0, changed=0)
                if launch_on:                                               # PREREG_V8F_LAUNCH_INTENT: per log interval (the steps since the last line)
                    rec["launch_n"], rec["launch_eligible"], rec["launch_rows"] = launch_int["n"], launch_int["eligible"], launch_int["rows"]
                    launch_int.update(rows=0, eligible=0, n=0)
                if enc is not None:
                    rec["lora_grad_norm"] = float(gn_l)
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[{step}/{total}] ade {rec.get('ade', float('nan')):.3f} total {rec['total']:.3f} p_gt {p_gt:.2f}", flush=True)
            if step % a.ckpt_every == 0:
                save()
            if step >= total:
                done = True
                break
        else:
            ep += 1
            bi = 0
    save(final=True)
    if t2b.active:                                                         # PREREG_T2_DAGGER: every scene id the bank opened -- the read refuses a scene that is in its model's bank
        json.dump(t2b.opened_record(), open(os.path.join(a.out, "t2_bank_scenes_opened.json"), "w"), indent=1)
    summary = {"done": True, "steps": step, "arm": a.arm, "seed": a.seed, "data_frac": a.data_frac,
               "steps_mult": a.steps_mult, "S": S, "n_clips": info["n_clips"], "n_windows": info["n_windows"],
               "exclude": a.exclude, "excluded_sha256": info.get("excluded_sha256"), "train_sha256": info.get("train_sha256"),
               "train_window_list_sha256": info.get("train_window_list_sha256"),
               "seconds": time.time() - t0, "argv_sha": argv_sha(argv), "levers": levers,
               "selfcheck": "selfcheck.json",
               "ego_perturb_n": ego_tot["n"], "ego_perturb_eligible": ego_tot["eligible"], "ego_perturb_rows": ego_tot["rows"], "ego_perturb_nav_kept": ego_tot["nav_kept"],
               "dagger_n": dag_tot["n"], "dagger_eligible": dag_tot["eligible"], "dagger_rows": dag_tot["rows"], "dagger_nav_kept": dag_tot["nav_kept"]}      # PREREG_V8F_DAGGER run totals: exactly 0 swapped at p = 0; swapped == eligible at p = 1      # O-RECOVER run totals: exactly 0 perturbed at p = 0; perturbed == eligible at p = 1
    summary["recover_p"], summary["recover_n"], summary["recover_eligible"], summary["recover_rows"], summary["recover_changed"] = rcv.p, rec_tot["n"], rec_tot["eligible"], rec_tot["rows"], rec_tot["changed"]
    summary["t2_bank_p"], summary["t2_bank_n"], summary["t2_bank_rows"] = t2b.p, t2_tot["n"], t2_tot["rows"]
    summary["recover_share"] = (rec_tot["n"] / rec_tot["eligible"]) if rec_tot["eligible"] else None        # PREREG_RECOVER_FT: the card's KR1 reads n_selected / n_eligible: [0.25, 0.35] for R, exactly 0 (n) for C0 / C0rep
    if comp.on:                                                            # AMEND1 section 1: the composed arm's run record (a flag-off summary.json keeps its key set)
        summary["ego_dagger_overlap_n"], summary["ego_perturb_with_dagger"] = ov_tot["n"], True
    if launch_on:                                                          # PREREG_V8F_LAUNCH_INTENT: the lever-ON arm's run record (a flag-off summary.json keeps its key set)
        summary["launch_n"], summary["launch_eligible"], summary["launch_rows"] = launch_tot["n"], launch_tot["eligible"], launch_tot["rows"]
        summary["launch_intent_w"] = cfg.launch_intent_w
        summary["launch_weight_share"] = launch_weight_share(launch_tot["n"], launch_tot["eligible"], cfg.launch_intent_w)
    if level_on:                                                           # PREREG_V8F_LEVEL: the lever-ON arm's run record (a flag-off summary.json keeps its key set)
        summary["level_head"], summary["w_level"] = "on", cfg.w_level
    if lead_on or cfg.aux_cond == "token":                                 # PREREG_LEVELHEAD_A2: the lever-ON arm's run record (a flag-off summary.json keeps its key set)
        summary["aux_cond"] = cfg.aux_cond
    if lead_on:
        summary["lead_head"], summary["w_lead"] = "on", cfg.w_lead
        summary["lead_labels"] = info["lead_labels"]
    if enc is not None:
        summary["encoder"] = "lora"
    if a.wm_only:                                                          # PREREG_V8F_MRATE: the WM-only arm's run record (a flag-off summary.json keeps its key set)
        summary["wm_only"], summary["wm_k"] = True, a.wm_k
    if wm_ladder_named(a):                                                 # PREREG_V8F_LADDER: the ladder arm's run record (a flag-off summary.json keeps its key set)
        summary["wm_ladder"] = {"declared": {k: list(v) if isinstance(v, tuple) else v for k, v in declared_wm_ladder(a).items()}, "windows": info.get("wm_windows"), "wm_windows": a.wm_windows, "wm_pooled": a.wm_pooled}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)      # LAST, atomically meaningful
    print("DONE", json.dumps({k: summary[k] for k in ("steps", "arm", "seed", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
