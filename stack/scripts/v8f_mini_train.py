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
    "init_ckpt": "data:init_ckpt",            # O-RECOVER's start state: the MODEL weights of an earlier checkpoint, config-checked, loaded strictly and READ BACK (load_init_ckpt) -> declared_vs_built (bad_init_ckpt)
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
    ap.add_argument("--o-interface", dest="o_interface", choices=M.O_INTERFACE_MODES, default="full",
                    help="PREREG_V8F_FLAT: what O reads of the layers above it. full (default) = S's checkpoints and T's code (every earlier arm, bit for bit, the RNG stream included); "
                         "cut = a CONSTANT in place of both (T's code = zeros after every override, S's checkpoints = the nominal ck_nom with ok = 0): the capacity-matched FLAT control. "
                         "S and T are still built, trained and run")
    ap.add_argument("--w-acc", dest="w_acc", type=float, default=0.0,
                    help="PREREG_V8F_OACC: the weight of the acceleration-matching term on O's plan (the second difference on the 0.2 s grid, origin knot included, of O's plan minus the GT's, squared, mean over the GT-valid steps). "
                         "0.0 (default) = the term is not in the loss = every earlier arm, bit for bit. > 0 = OACC's X: the value comes from the PREREG's rule (v8f_mini_eval_oacc.py wacc), is written into the argv and never tuned")
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
                           detach_upper=a.detach_upper, kin_mode=a.kin_mode, cons_absent=lv["cons_absent"], code_mix=a.code_mix, o_interface=a.o_interface, w_acc=a.w_acc)


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

    ``on`` False (the flag is not named): the loop never calls this object; it runs its two original lines (``ego.apply`` then ``dag.apply``), so every existing argv is the pre-change run."""

    def __init__(self, ego, dag, on: bool = False):
        self.ego, self.dag, self.on = ego, dag, bool(on)
        self.overlap = 0

    def built_levers(self) -> dict:
        return {"ego_perturb_with_dagger": self.on, "order": ["dagger_swap", "ego_perturb"] if self.on else ["ego_perturb", "dagger_swap"]}

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
    model.load_state_dict(sd, strict=True)
    cur = model.state_dict()
    rec["n_mismatch_after_load"] = sum(1 for k in sd if not torch.equal(cur[k].detach().cpu(), sd[k].detach().cpu()))
    return rec


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


def declared_vs_built(a, cfg, model, opt, data_info, enc=None, ego=None, sched=None, dag=None, comp=None) -> list:
    """Return the list of mismatches (empty = pass). Each flag is checked against what was BUILT, by FLAG_CHECKS kind."""
    bad = bad_encoder(a, enc, data_info)
    bad += bad_ego_perturb(a, ego) + bad_init_ckpt(a, data_info) + bad_schedule(a, sched)          # O-RECOVER (PREREG_V8F_HACC_MC section 7); bad_schedule: --warmup read back from the built schedule
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
              "w_pen_self", "w_wm", "detach_upper", "kin_mode", "w_acc"):
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
    else:
        ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                              window_start=cfg.window_start)
    info["ds_window_start"] = ds.window_start                       # the held-age clamp's origin, read back from the BUILT dataset (G-DVB compares it)
    prior = M.load_p10(a.p10) if not a.synthetic else M.P10Prior.synthetic(a.seed)
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, prior, mu, sd).to(dev)
    if a.init_ckpt:                                                  # O-RECOVER's start state: the MODEL weights of an earlier checkpoint (before the optimiser, the probes and the gate); a resume of THIS run wins
        try:
            info["init_ckpt"] = load_init_ckpt(a.init_ckpt, model, cfg, apply=not (os.path.exists(os.path.join(a.out, "ckpt_last.pt")) and not a.no_resume))
        except Exception as e:                                       # noqa: BLE001  a missing / foreign / mismatched checkpoint is a REFUSAL, not a crash
            raise SystemExit(f"ZZDVB-REFUSEDZZ init-ckpt {a.init_ckpt}: {type(e).__name__}: {str(e)[:600]}")
    dag = make_dagger(a, J, wins, clips, info)                       # PREREG_V8F_DAGGER: off by default (no bank read); with a bank: load + K-B2 + K-B3 + COVERAGE here, a refusal stops the launch before the gate
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
    bad = declared_vs_built(a, cfg, model, opt, info, enc, ego, lr_at, dag, comp)
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
    levers = model.built_levers()
    levers["ego_perturb"] = ego.built_levers()                        # recorded like every other lever (selfcheck.json, summary.json, the checkpoint)
    levers["dagger"] = dag.built_levers()                             # PREREG_V8F_DAGGER: what was BUILT (the bank's path and sha256 included) ...
    levers["dagger_p"], levers["dagger_bank_sha256"] = dag.p, dag.bank_sha256          # ... and the two values the reader keys on, flat (0.0 / None when the lever is off)
    levers["dagger_bank_sha256_declared"] = a.dagger_bank_sha256 or None              # ... and the hash the argv DECLARED (--dagger-bank-sha256); the gate has just required it to equal the built one (None when the lever is off)
    levers["ego_perturb_with_dagger"] = comp.on                        # AMEND1 section 1: what was BUILT (False unless --ego-perturb-with-dagger) ...
    levers["ego_dagger_overlap_n"] = cmpp["overlap"]                   # ... and the composed probe's measured overlap on the probe batch (0 when off; the run total is summary.json's ego_dagger_overlap_n)
    levers["init_ckpt"] = info.get("init_ckpt")
    levers["lr_schedule"] = lr_at.built_levers()                       # the built warmup / peak / total the loop calls (G-DVB read --warmup back from it)
    if enc is not None:
        levers["encoder"] = enc.built_levers()
        levers["accum"] = {"accum": a.accum, "micro_windows": a.batch // a.accum, "chunk_frames": chunk}
    sc = {"declared_vs_built": "PASS", "live": live, "code_mix_probe": mix, "o_interface_probe": ocut, "acc_probe": accp, "admissibility": adm, "levers": levers,
          "steps_total": total, "S_steps_per_epoch_100pct": S, "info": info, "argv_sha": argv_sha(argv)}
    sc["ego_perturb_probe"] = egop
    sc["dagger_probe"] = dagp
    sc["ego_dagger_compose_probe"] = cmpp
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
    ov_tot = {"n": 0}                                             # AMEND1 section 1: samples both swapped and perturbed (exactly 0), carried in the checkpoint of a composed arm only
    ov_int = {"n": 0}                                             # ... and since the last log line
    if os.path.exists(ck_path) and not a.no_resume:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if ck["argv_sha"] != argv_sha(argv):
            raise SystemExit("ZZRESUME-REFUSEDZZ the checkpoint was written by a different argv")
        model.load_state_dict(ck["model"])
        if enc is not None:
            enc.load_lora_state_dict(ck["lora"])
        opt.load_state_dict(ck["opt"])
        step, ep, bi = ck["step"], ck["epoch"], ck["batch_in_epoch"]
        ego_tot.update(ck.get("ego_totals") or {})
        dag_tot.update(ck.get("dagger_totals") or {})
        ov_tot.update(ck.get("ego_dagger_overlap") or {})
        print(f"[resume] step {step} epoch {ep} batch {bi}", flush=True)

    def save(final=False):
        blob = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": ep, "batch_in_epoch": bi,
                "argv_sha": argv_sha(argv), "cfg": cfg.__dict__, "p10_path": os.path.abspath(a.p10) if not a.synthetic else "",
                "grid": a.grid, "arm": a.arm, "seed": a.seed, "cons_mu": mu, "cons_sd": sd,
                "cfg_tuple_fields": {"wm_strides": list(cfg.wm_strides)},
                "ego_perturb": ego.built_levers(), "init_ckpt": info.get("init_ckpt"), "ego_totals": dict(ego_tot),
                "dagger": dag.built_levers(), "dagger_totals": dict(dag_tot)}          # O-RECOVER: what was BUILT, beside the config, and the counts so far
        if comp.on:                                                     # composed arms only: a flag-off checkpoint keeps its key set
            blob["ego_perturb_with_dagger"], blob["ego_dagger_overlap"] = True, dict(ov_tot)
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
            if comp.on:                                            # AMEND1 section 1 (--ego-perturb-with-dagger): the swap is drawn first, the perturbation only on the samples the swap did not replace; the overlap is measured (exactly 0)
                b = comp.apply(b, step)
                for c_ in (ov_int, ov_tot):
                    c_["n"] += comp.overlap
            else:                                                  # every other argv: the two original lines, in their original order
                b = ego.apply(b, step)                                 # O-RECOVER: the kinematic inputs of the selected samples are rebuilt from a perturbed ego state; targets and labels are untouched (off: `b` itself)
                b = dag.apply(b, step)                                 # PREREG_V8F_DAGGER: the ego tensors of the selected samples are the model's OWN T1-kin state at the matched route point; targets and labels stay the human's (off: `b` itself)
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
                rec["ego_perturb_n"], rec["ego_perturb_eligible"], rec["ego_perturb_rows"] = ego_int["n"], ego_int["eligible"], ego_int["rows"]      # per log interval (the steps since the last line)
                rec["ego_perturb_nav_kept"] = ego_int["nav_kept"]
                ego_int.update(rows=0, eligible=0, n=0, nav_kept=0)
                if dag.active:
                    rec["dagger_n"], rec["dagger_eligible"], rec["dagger_rows"], rec["dagger_nav_kept"] = dag_int["n"], dag_int["eligible"], dag_int["rows"], dag_int["nav_kept"]          # per log interval
                dag_int.update(rows=0, eligible=0, n=0, nav_kept=0)
                if comp.on:
                    rec["ego_dagger_overlap_n"] = ov_int["n"]               # AMEND1 section 1: per log interval (exactly 0)
                ov_int.update(n=0)
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
               "train_window_list_sha256": info.get("train_window_list_sha256"),
               "seconds": time.time() - t0, "argv_sha": argv_sha(argv), "levers": levers,
               "selfcheck": "selfcheck.json",
               "ego_perturb_n": ego_tot["n"], "ego_perturb_eligible": ego_tot["eligible"], "ego_perturb_rows": ego_tot["rows"], "ego_perturb_nav_kept": ego_tot["nav_kept"],
               "dagger_n": dag_tot["n"], "dagger_eligible": dag_tot["eligible"], "dagger_rows": dag_tot["rows"], "dagger_nav_kept": dag_tot["nav_kept"]}      # PREREG_V8F_DAGGER run totals: exactly 0 swapped at p = 0; swapped == eligible at p = 1      # O-RECOVER run totals: exactly 0 perturbed at p = 0; perturbed == eligible at p = 1
    if comp.on:                                                            # AMEND1 section 1: the composed arm's run record (a flag-off summary.json keeps its key set)
        summary["ego_dagger_overlap_n"], summary["ego_perturb_with_dagger"] = ov_tot["n"], True
    if enc is not None:
        summary["encoder"] = "lora"
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)      # LAST, atomically meaningful
    print("DONE", json.dumps({k: summary[k] for k in ("steps", "arm", "seed", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
