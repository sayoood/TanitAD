#!/usr/bin/env python3
"""THE REFe LAUNCH GATE: no training step runs until this mints a PASS token for the EXACT (commit, code tree, argv, data) being launched.
CLAUDE.md, PI 2026-09-26 (BINDING).  Route, contract and every regression arm: eval/REFE_LAUNCH_GATE.md.

    # DEV BOX -- ship the exact landed bytes (writes SHIP_MANIFEST.json; refuses a file that is not its blob at the commit)
    python refe/launch_gate.py ship-manifest --pkg <package root> --commit <40-hex> --out <package root>/SHIP_MANIFEST.json
    # POD -- run every check on the REAL argv and REAL data, then mint (the chain does this; a token is never hand-written)
    python launch_gate.py run --profile refe_p2 --pkg /workspace/refe-plan --args "$ARGS" --out-dir <dir> --gate-sets <d> --gate-pdm <d> \
        --gate-bank <d> --p2-gate-json <gate.json>
    # POD -- the chain, before every train.py launch:  exits 0 (ZZGATE-MATCHZZ) or 5 (ZZGATE-REFUSEDZZ)
    python launch_gate.py verify --token <PASS_*.json> --args "$ARGS" --pkg /workspace/refe-plan --profile refe_p2
    # POD, P6 -- refe/p6_pod.sh: --gate-bank carves the smoke bank from rows whose D9 parts exist (gate_fixture.carve_p6_gate_bank), plus the three pod-run inputs
    python launch_gate.py run --profile refe_p6 --pkg <pkg> --args "$ARGS" --out-dir <dir> --gate-bank <d> --gate-lab-json <lab.json>         --gate-mem-metrics <metrics.jsonl> --gate-egr-json <g_egr_result.json>
    # anywhere, instant -- what a profile refuses before a single check runs
    python launch_gate.py refusals --profile refe_p6 --args "<the P6 argv>"
    # RL-S1 (trained by scorer_finetune.py, not train.py): the arm is NAMED (C0 C1 R0 R1 R2); the preflight is the trainer's own `--preflight` log, parsed
    python launch_gate.py run --profile refe_rls1 --arm-name R0 --pkg <pkg> --args "$ARGS" --out-dir <dir> --gate-preflight-json <preflight.jsonl>
    # P2-s1, the P2 training-seed replicate (refe/p2rep_pod.sh): P2's gates on P2's argv PLUS exactly two changes (--out, --seed 1), PINNED token for token
    python launch_gate.py run --profile refe_p2rep --pkg <pkg> --args "$ARGS" --out-dir <dir> --gate-sets <d> --gate-pdm <d> --gate-bank <d> --p2-gate-json <gate.json>

Token: schema `tanitad.launch_gate/1` (gate_token.py), HMAC-SHA256 with a pod-local key; PASS only if every required check has usable PASS
evidence bound to the same (profile, commit, tree_sha256, argv_sha256) whose `inputs_read` fingerprints equal the launch data manifest.
REHEARSAL evidence (tiny config, synthetic images, CPU) never counts toward a token.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_token as GT  # noqa: E402

CHECKS = ("G-HYG", "G-DVB", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE", "G-P2GATE", "G-LIVE-LAB", "G-LIVE-MEM", "G-EGR", "G-PREFLIGHT")
SUITE = [{"script": "test_p2r_chain.py", "token": "ZZTEST_PASS"}, {"script": "test_launch_gate.py", "args": ["--static", "--no-banked"], "token": "ZZLG_STATIC_PASS"}]
FORBIDDEN = ("--synthetic", "--overfit", "--overfit-n", "--overfit-thresh", "--spread", "--break-optimizer", "--halt-after-steps",
             "--declare-change", "--preflight", "--cpu", "--n-cameras", "--no-undistort", "--no-cosine", "--weights")
P2_DATA = ("--targets", "--scorer-targets", "--images", "--calib", "--init-from", "--onpolicy-targets", "--pdm-labels")
P6_TERMS = ("traj_L1", "score", "pos_L1", "yaw_L1", "bce_nc", "bce_dac", "bce_ep", "bce_ttc", "bce_c", "bce_ddc")
# RL-S1 (eval/PREREG_RL_S1.md, DRAFT: section 1 "Lever" + "Recipe", section 3 arms).  ⛔ Every value below is the DRAFT's, so an amendment of the prereg is a
# one-line change HERE (and nowhere else).  Trained by scorer_finetune.py, which is not train.py: the profile names its parser, its source and its own checks.
RLS1_ARM_SEEDS = {"C0": 0, "C1": 1, "R0": 0, "R1": 1, "R2": 0}                       # arm -> its registered seed (naming an arm asserts the seed)
RLS1_ARM_CLASS = {"C0": "C", "C1": "C", "R0": "R", "R1": "R", "R2": "R2"}            # C: BCE-only control; R: BCE + EPO, core reward; R2: R + K + L (D3, PI 2026-10-12)
RLS1_EPO_REGISTERED = {"epo_temp": 0.1, "epo_beta": 0.1, "epo_ramp": 200, "epo_lam": 1.0, "epo_lam_d": 0.15, "epo_w_prog": 0.5}   # effective value (named OR default)
RLS1_R2_REGISTERED = {"epo_lam_k": 0.05, "epo_lam_l": 0.05}
RLS1_RECIPE = {"lr": 1e-4, "warmup": 50, "batch": 8, "accum": 8, "max_updates": 1000,   # the SFT-4 optimiser, 1,000 updates; EVERY arm, so R vs C differ by EPO alone
               "epochs": 4.0}   # TL-1 (PI 2026-10-12, 16,000 scenes): --epochs 2 SILENTLY ends at 500 updates (batch 8 x accum 8); 4 reaches the 1,000
# named-but-inert levers (a flag whose effect is switched off by another registered value is a declared lever the run does not use): the SFT-2 NAVSIM labels would
# mix label sources with the PDM labels; --b-compw / --tau-s belong to other b-modes; --tau-t scales a ListNet term that --lam 0 multiplies by zero.
AUX_HEAD_FLAGS = ("--aux-drv", "--w-aux-drv", "--aux-drv-tap", "--aux-drv-depth", "--aux-drv-feed", "--aux-drv-labels", "--aux-drv-shuffle-seed", "--aux-drv-lr")
RLS1_FORBIDDEN = ("--preflight", "--lane-labels-train", "--lane-labels-heldout", "--b-compw", "--tau-s", "--tau-t") + AUX_HEAD_FLAGS      # RL-S1 is not the head's experiment
RLS1_EPO_TERMS = ("epo", "epo_J", "epo_kl", "epo_lam_pg")                              # the trainer's metric rows must emit these on an --epo arm
RLS1_DATA = ("--ckpt", "--targets", "--images", "--calib", "--onpolicy", "--heldout", "--heldout-logs", "--lane-labels-train", "--lane-labels-heldout",
             "--pdm-labels-train", "--pdm-labels-heldout", "--teacher-lane-train", "--teacher-lane-heldout", "--epo-terms-train", "--epo-terms-heldout", "--aux-drv-labels")
RLS1_SUITE = [{"script": "test_rl_reward.py", "token": "ZZTEST_PASS"}, {"script": "test_rl_epo.py", "token": "ZZTEST_PASS"},
              {"script": "test_launch_gate.py", "args": ["--static", "--no-banked"], "token": "ZZLG_STATIC_PASS"}]
# RL-S1b (eval/PREREG_RL_S1.md section 10, 2026-10-14; raw/2026-10-14-rls1b-attribution/attribution.json + dose.json).  R0 was KILLED by K3/g5; the attribution found the TTC loss is NOT
# the reward's preference (58/59 TTC-worsening pick changes are DISPREFERRED by R0's own reward) but the scorer's generalisation of the EPO shift, and the dose read
# (logit interpolation C0 -> R0) put half the PDMS gain at a quarter of the shift with no TTC loss.  ONE lever moves: the EPO dose, --epo-lam 1.0 -> 0.25 (NAMED in
# the argv; the parser default stays 1.0, so refe_rls1 is untouched).  Same trainer, reward, recipe, data, kills and bar.  C1 is RL-S1's C1 exactly (same class, seed and
# argv).  R2b (lane change at 0.25) is NOT an arm here: it would be a 4th A40 arm beyond the three authorised, so it is a PI decision (section 10).
RLS1B_ARM_SEEDS = {"C1": 1, "R0b": 0, "R1b": 1}
RLS1B_ARM_CLASS = {"C1": "C", "R0b": "R", "R1b": "R"}
RLS1B_EPO_LAM = 0.25                                                                  # a power of two: lam_pg * x is EXACT in binary floating point (test_rl_s1b.py)
RLS1B_EPO_REGISTERED = {**RLS1_EPO_REGISTERED, "epo_lam": RLS1B_EPO_LAM}
RLS1B_NEEDS_PI = {"R2b": "R2b (R0b + --epo-k --epo-l, the lane-change arm at --epo-lam 0.25) is a 4th A40 arm beyond RL-S1's remaining three: it needs a PI decision "
                         "(eval/PREREG_RL_S1.md section 10) before it can be registered"}
RLS1B_SUITE = RLS1_SUITE + [{"script": "test_rl_s1b.py", "token": "ZZTEST_PASS"}]
# Auxiliary drivable-area head, S1 (WIRING_PLAN.md; eval/PREREG_AUX_DRIVABLE_DRAFT.md "Hosts": S1 = the SFT-4 recipe on the dev box, trunk frozen).  ⛔ PROPOSED values: they are
# raw/2026-10-13-aux-head/PREREG_BLANKS_PROPOSED.md's, and the Master Mind's approval of that file is what makes them registered -- an amendment is a one-line change HERE.
# Arms: N = the no-head replicate, H1 = head with the feed off (perception proof; the scorer is bit-identical to N, G12), H2 = head + feed.  One seed per arm name.
AUXS1_SEEDS = (0, 1, 2)
AUXS1_ARM_CLASS = {f"{c}-s{k}": c for c in ("N", "H1", "H2") for k in AUXS1_SEEDS}
AUXS1_ARM_SEED = {f"{c}-s{k}": k for c in ("N", "H1", "H2") for k in AUXS1_SEEDS}
AUXS1_HEAD_REGISTERED = {"w_aux_drv": 0.1, "aux_drv_lr": 2e-4, "aux_drv_tap": "visual", "aux_drv_depth": 2}      # effective value (named OR default), head arms
# the optimiser and loss of EVERY arm (N included), inherited from the SFT-4 run that produced u400 (refe/sft4_pod.sh COMMON); --max-updates 400 is the S1 budget
AUXS1_RECIPE = {"lr": 1e-4, "warmup": 50, "batch": 8, "accum": 8, "max_updates": 400, "eval_every": 400, "b_mode": "expreward", "lam": 1.0, "tau_s": 0.1, "lane_weight": 1.0}
AUXS1_EPO_FLAGS = ("--epo", "--epo-temp", "--epo-beta", "--epo-ramp", "--epo-lam", "--epo-lam-d", "--epo-w-prog", "--epo-k", "--epo-lam-k", "--epo-l", "--epo-lam-l",
                   "--epo-terms-train", "--epo-terms-heldout")
# --aux-drv-shuffle-seed is the label-shuffled CONTROL's lever: a non-control arm never carries it (a dated prereg amendment adds the control arm and relaxes this)
AUXS1_FORBIDDEN = ("--preflight", "--lane-labels-train", "--b-compw", "--tau-t", "--aux-drv-shuffle-seed") + AUXS1_EPO_FLAGS
AUXS1_HEAD_TERMS = ("aux_bce", "aux_huber")                                           # the trainer's metric rows must emit these on a head arm
AUXS1_SUITE = [{"script": "test_rl_reward.py", "token": "ZZTEST_PASS"}, {"script": "test_rl_epo.py", "token": "ZZTEST_PASS"},
               {"script": "test_aux_drivable_head.py", "token": "ZZTEST_PASS"}, {"script": "test_aux_s1_wiring.py", "token": "ZZTEST_PASS"},
               {"script": "test_launch_gate.py", "args": ["--static", "--no-banked"], "token": "ZZLG_STATIC_PASS"}]
# P2-s1, the P2 TRAINING-SEED REPLICATE (eval/PREREG_P2_REPLICATE.md; chain refe/p2rep_pod.sh).  The arm is P2's argv CHARACTER FOR CHARACTER with exactly two changes: `--out` is the
# replicate's own run directory and `--seed 1` is added (P2 ran the parser default, seed 0).  `refe_p2` binds NEITHER (its `required_flags` names 8 of its 20 flags; a missing `--seed 1`, P2's
# own `--out`, or any other lever that moved would still mint), so the replicate has a profile of its own that PINS the whole argv (`pinned_argv`, token for token).
# P2_ARGV is `ARGS` of refe/p2r_pod.sh as it EXPANDS on the pod (DATA=/workspace/data, RUN=$DATA/refe_runs/p2_joint_pdm); test_p2rep_chain.py diffs the three copies (p2r_pod.sh, p2rep_pod.sh, this).
P2_RUN_DIR = "/workspace/data/refe_runs/p2_joint_pdm"
P2REP_RUN_DIR = "/workspace/data/refe_runs/p2_joint_pdm_s1"
P2REP_SEED = 1
P2_ARGV = ("--backbone vitl16 --targets /workspace/data/refe_navtrain/train_grow --scorer-targets /workspace/data/refe_navtrain/train_grow --images /workspace/data/navtrain_pixels "
           "--calib /workspace/data/refe_navtrain/train_grow/calib_table.json --init-from /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --scorer-mode onpolicy "
           "--onpolicy-targets /workspace/data/refe_navtrain/onpolicy/sets --pdm-labels /workspace/data/refe_sft2/train_pdm --pdm-only --scorer-sees-trunk --epochs 3 --epoch-unit scenes "
           "--batch 4 --accum 64 --lr 5e-5 --amp bf16 --tf32 --compile --workers 2 --log-every 1 --ckpt-every-min 20 --out " + P2_RUN_DIR + " --resume")


def p2rep_argv(p2_argv: str = P2_ARGV) -> list:
    """P2's argv with EXACTLY the replicate's two changes: `--out` -> the replicate's run dir, and `--seed 1` inserted before it.  Anything else is P2's."""
    toks = p2_argv.split()
    i = toks.index("--out")
    toks[i + 1] = P2REP_RUN_DIR
    toks[i:i] = ["--seed", str(P2REP_SEED)]
    return toks


P2REP_ARGV = p2rep_argv()
_BASE = dict(trainer="refe/train.py", parser_module="train", trainer_src="train.py", live_terms=("traj_L1", "score"), ckpt_steps=4, smoke_steps=2, smoke_accum=2,
             suite=SUITE, forbidden_flags=FORBIDDEN, output_flags=("--out",), mintable=True, require_commit=True, rehearsal=False,
             checks=("G-HYG", "G-DVB", "G-LIVE", "G-EVAL", "G-CKPT", "G-SUITE"), na_ok=("G-CLOCK",),
             clock_na="P2 has no label clock: its labels are the PDM relabel of a banked on-policy bank, stamped by the relabeller's ckpt_step")

PROFILES = {
    # PREREG_P2 / eval/PREREG_P2_AMENDMENT_1.md: the joint fine-tune p2r_pod.sh launches.  required_flags are its design, asserted not trusted.
    "refe_p2": {**_BASE, "name": "refe_p2", "data_flags": P2_DATA, "checks": _BASE["checks"] + ("G-P2GATE",),
                "required_flags": {"--scorer-mode": "onpolicy", "--pdm-only": None, "--scorer-sees-trunk": None, "--epochs": "3",
                                   "--epoch-unit": "scenes", "--batch": "4", "--accum": "64", "--lr": "5e-5", "--amp": "bf16"},
                "provisional_flags": ()},
    # SPEC_P6_PAPER_FAITHFUL.md section 5.  ⛔ REFUSES TODAY: --yaw-wrap / --nav-mode / --labeller are D5 / D6 / D1 and are NOT built, and the
    # trainer's metrics carry no pos / yaw / per-component BCE terms.  Flag and term NAMES are PROVISIONAL until the implementer lands them.
    "refe_p6": {**_BASE, "name": "refe_p6", "data_flags": ("--targets", "--images", "--calib", "--metric-cache"),
                "suite": SUITE + [{"script": "test_p6_levers.py", "args": ["--fast"], "token": "ZZP6_PASS"},
                                  {"script": "test_p6_chain.py", "token": "ZZP6CHAIN_PASS"}],
                # G-LIVE-LAB / G-LIVE-MEM / G-EGR are POD-RUN: the gate consumes their instruments' output (--gate-lab-json / --gate-mem-metrics / --gate-egr-json)
                "checks": ("G-HYG", "G-DVB", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE", "G-LIVE-LAB", "G-LIVE-MEM", "G-EGR"), "na_ok": (),
                "live_terms": P6_TERMS,
                "forbidden_flags": FORBIDDEN + ("--grow", "--grow-scenes", "--onpolicy-targets", "--init-from", "--pdm-labels", "--pdm-only",
                                                "--scorer-mode", "--label-backend"),
                # --yaw-wrap must be NAMED in the argv AND carry the PI's ruling P6-Y (2026-10-11): "off" = unwrapped yaw L1, the paper's literal reading (on = the wrapped
                # error of P2 / SFT-4). The gate asserts the ruling, not just the presence: `--yaw-wrap on` is refused as a value mismatch.
                "required_flags": {"--scorer-sees-trunk": None, "--yaw-wrap": "off", "--nav-mode": "command", "--labeller": "online",
                                   "--epochs": "25", "--batch": "4", "--accum": "64", "--lr": "2e-4", "--weight-decay": "0.01", "--amp": "bf16",
                                   "--score-w": "0.1"},
                "provisional_flags": (),
                # SPEC_P6 section 0.2 wrote 10,063 steps (= ceil(25 x 103,037 / 256)); the trainer's own definition (train.py: epochs x ceil(epoch_len / (batch x
                # accum)), re-derived independently by G-DVB) gives 25 x 403 = 10,075 -- the final run's measured step count. The profile registers the TRAINER's.
                "expect_schedule": {"epochs": 25, "steps": 10075, "epoch_len": 103037, "n_tuples": 189888}},
    # the dev-box rehearsal: P2's levers on the tiny bank.  Exercises every instrument; its evidence is REHEARSAL and can never mint.
    "refe_rehearsal": {**_BASE, "name": "refe_rehearsal", "data_flags": ("--targets", "--scorer-targets", "--init-from", "--onpolicy-targets",
                                                                          "--pdm-labels"),
                       "forbidden_flags": tuple(f for f in FORBIDDEN if f not in ("--synthetic", "--cpu", "--n-cameras")), "mintable": False,
                       "require_commit": False, "rehearsal": True, "required_flags": {"--scorer-mode": "onpolicy", "--scorer-sees-trunk": None},
                       "provisional_flags": (), "eval_cfg_allowed": ("n_cameras", "cameras")},
    # RL-S1 (eval/PREREG_RL_S1.md, eval/HANDOFF_RL_S1.md finding 5): scorer_finetune.py's EPO arms.  The arm is a CLASS the argv must satisfy (rls1_refusals): C = BCE-only
    # control (no --epo), R = R0/R1 (--epo, core reward), R2 = R + --epo-k + --epo-l.  There is no BUILD/SMOKE instrumentation (that is train.py's): the trainer's own
    # `--preflight` (G1-G11) is CONSUMED, parsed, and its file sha256 is bound into the token; G-DVB is static + the preflight's `data` event.
    "refe_rls1": {**_BASE, "name": "refe_rls1", "runner": "scorer", "arm_rules": "rls1", "trainer": "refe/scorer_finetune.py", "parser_module": "scorer_finetune",
                  "trainer_src": "scorer_finetune.py", "namespace_funcs": ("main", "epo_refusals", "_g_probe", "aux_s1_refusals"), "data_flags": RLS1_DATA,
                  "checks": ("G-HYG", "G-DVB", "G-PREFLIGHT", "G-SUITE"), "na_ok": ("G-CLOCK",), "suite": RLS1_SUITE, "live_terms": ("bceA", "lossB"),
                  "clock_na": "RL-S1 has no label clock of its own: its labels are the PDM relabel of an on-policy bank, stamped by the relabeller's ckpt_step; "
                              "the preflight's G9 bounds the stale share (<= 5 %) and rl_terms count only for the set and ckpt_step they were computed on",
                  "forbidden_flags": RLS1_FORBIDDEN, "output_flags": ("--out",),
                  "required_flags": {"--pdm-labels-train": None, "--pdm-labels-heldout": None, "--require-pdm": None, "--b-mode": "listnet", "--lam": "0"},
                  "provisional_flags": ()},
    # Auxiliary drivable-area head, S1 (see AUXS1_* above): scorer_finetune.py with --aux-drv.  Same runner as refe_rls1 (static checks + the trainer's own --preflight, now with G12);
    # the arm is NAMED (N-s0 .. H2-s2) and is a CLASS the argv must satisfy (auxs1_refusals).  NOT an RL profile: every EPO lever is forbidden.
    "refe_aux_s1": {**_BASE, "name": "refe_aux_s1", "runner": "scorer", "arm_rules": "auxs1", "trainer": "refe/scorer_finetune.py", "parser_module": "scorer_finetune",
                    "trainer_src": "scorer_finetune.py", "namespace_funcs": ("main", "epo_refusals", "_g_probe", "aux_s1_refusals"), "data_flags": RLS1_DATA,
                    "checks": ("G-HYG", "G-DVB", "G-PREFLIGHT", "G-SUITE"), "na_ok": ("G-CLOCK",), "suite": AUXS1_SUITE, "live_terms": ("bceA", "lossB"),
                    "clock_na": "aux S1 has no label clock of its own: its scorer labels are the PDM relabel of an on-policy bank (the preflight's G9 bounds the stale share) and the head's "
                                "labels are keyed by (log_name, token) in a pinned store (G12 + G-DVB bind its sha256 and frame count)",
                    "forbidden_flags": AUXS1_FORBIDDEN, "output_flags": ("--out",),
                    "required_flags": {"--pdm-labels-train": None, "--pdm-labels-heldout": None, "--require-pdm": None, "--lane-head": None, "--teacher-lane-train": None,
                                       "--teacher-lane-heldout": None, "--b-mode": "expreward", "--lam": "1.0", "--tau-s": "0.1", "--max-updates": "400", "--lr": "1e-4"},
                    "provisional_flags": ()},
}


# the replicate = refe_p2 (same checks, same G-P2GATE sub-check, same data flags, same forbidden flags, same P2 levers asserted) + `--seed 1` required + the WHOLE argv pinned + its own chain test in G-SUITE
PROFILES["refe_p2rep"] = {**PROFILES["refe_p2"], "name": "refe_p2rep", "suite": SUITE + [{"script": "test_p2rep_chain.py", "token": "ZZTEST_PASS"}],
                          "required_flags": {**PROFILES["refe_p2"]["required_flags"], "--seed": str(P2REP_SEED)}, "pinned_argv": tuple(P2REP_ARGV)}
# RL-S1b = refe_rls1 (same runner, checks, data flags, forbidden / required flags, recipe, G-DVB, G-PREFLIGHT) with ITS arm table and ONE registered value moved (epo_lam).
# rls1_refusals reads `arm_class` / `arm_seeds` / `epo_registered` from the profile; refe_rls1 carries none of them, so it keeps the module constants above.
PROFILES["refe_rls1b"] = {**PROFILES["refe_rls1"], "name": "refe_rls1b", "suite": RLS1B_SUITE, "arm_class": RLS1B_ARM_CLASS, "arm_seeds": RLS1B_ARM_SEEDS,
                          "epo_registered": RLS1B_EPO_REGISTERED, "arms_need_pi": RLS1B_NEEDS_PI}


def parse_args_string(s: str) -> list:
    return GT.argv_from_string(s)


def flag_present(argv, flag):
    return flag in argv


def pinned_argv_refusals(prof: dict, argv: list) -> list:
    """A profile with `pinned_argv` accepts ONE argv: the same (flag, values) groups, each exactly once, in any order (argparse is order-blind).  A lever that moved, a flag added, a flag dropped,
    a flag twice, or a token before the first flag is a refusal -- the generic `required_flags` rules name only the levers a profile ASSERTS and let every other flag through."""
    pin = prof.get("pinned_argv")
    if not pin:
        return []
    name, out = prof["name"], []
    want, got = {}, {}
    for d, toks in ((want, list(pin)), (got, list(argv))):
        for f, v in GT.flag_pairs(toks):
            d.setdefault(f, []).append(tuple(v))
    if argv and not str(argv[0]).startswith("--"):
        out.append(f"{name}: the argv starts with {argv[0]!r}, which belongs to no flag")
    for f in sorted(want):
        if f not in got:
            out.append(f"{name} PINS {f} {' '.join(want[f][0])}; the argv does not carry it")
        elif len(got[f]) > 1:
            out.append(f"{name}: {f} is carried {len(got[f])} times (it is pinned once)")
        elif got[f][0] != want[f][0]:
            out.append(f"{name} PINS {f} {' '.join(want[f][0])}; the argv has {f} {' '.join(got[f][0])}")
    for f in sorted(set(got) - set(want)):
        out.append(f"{name}: the argv carries {f} {' '.join(got[f][0])}, which the profile does not pin (the replicate is P2's argv plus --out and --seed {P2REP_SEED}, and nothing else)")
    return out


def profile_refusals(prof: dict, argv: list) -> list:
    """Reasons the profile refuses this argv BEFORE any check runs.  (a) a forbidden flag; (b) a required lever missing or different;
    (c) a lever the profile DECLARES that the trainer does not BUILD (flag absent from the parser) -- a launch blocker, never a footnote;
    (d) a declared loss term the trainer never emits.  The trainer is the profile's `parser_module` / `trainer_src` (default train / train.py); a profile with
    `arm_rules` adds its arm rules (refe_rls1: `rls1_refusals`)."""
    import importlib
    import re
    T = importlib.import_module(prof.get("parser_module", "train"))
    out = []
    parser = T.build_parser()
    known = {o for a in parser._actions for o in a.option_strings}
    for f in prof.get("forbidden_flags", ()):
        if f in argv:
            out.append(f"argv carries {f}, which {prof['name']} FORBIDS in a launch")
    for f, want in prof.get("required_flags", {}).items():
        if f not in argv:
            out.append(f"{prof['name']} requires {f}{'' if want is None else ' ' + want}; the argv does not carry it")
            continue
        got = GT.flag_values(argv, f)
        if want is not None:
            try:
                same = got and float(got[0]) == float(want)
            except ValueError:
                same = got and got[0] == want
            if not same:
                out.append(f"{prof['name']} requires {f} {want}; the argv has {f} {' '.join(got or [])}")
    for f in sorted(set(prof.get("provisional_flags", ())) | {f for f in prof.get("required_flags", {}) if f not in known}):
        if f not in known:
            out.append(f"DECLARED, NOT BUILT: {prof['name']} declares {f} but the trainer's parser has no such flag "
                       f"(a config that declares a lever the model does not build is a launch blocker)")
    out += pinned_argv_refusals(prof, argv)
    trainer_src = prof.get("trainer_src", "train.py")
    src = (HERE / trainer_src).read_text(encoding="utf-8")
    terms = tuple(prof.get("live_terms", ()))
    if prof.get("arm_rules") == "rls1":
        rs, extra = rls1_refusals(prof, argv, T, parser)
        out += rs
        terms += extra
    elif prof.get("arm_rules") == "auxs1":
        rs, extra = auxs1_refusals(prof, argv, T, parser)
        out += rs
        terms += extra
    for t in terms:
        if not re.search(r'["\']' + re.escape(t) + r'["\']', src):
            out.append(f"DECLARED, NOT BUILT: {prof['name']} requires the loss term {t!r}; {trainer_src} never emits it")
    return out


# ============================================================================================ RL-S1 arm rules (refe_rls1)
def rls1_arm_class(argv: list) -> str:
    """The arm class an argv DECLARES by what it carries.  No --epo = C (the BCE-only control); --epo alone = R (R0/R1, the core reward); --epo with BOTH
    --epo-k and --epo-l = R2; --epo with exactly one of them = R? (no arm is that)."""
    k, l = "--epo-k" in argv, "--epo-l" in argv
    if "--epo" not in argv:
        return "C"
    return "R2" if (k and l) else ("R?" if (k or l) else "R")


def rls1_control_argv(argv: list) -> list:
    """The matched control of an R argv: the SAME argv minus every EPO lever (flag and value), so R vs C differ by the EPO term alone.  (--out is the
    caller's to change: two arms never share an output directory.)"""
    out, skip = [], False
    for t in argv:
        if t.startswith("--"):
            skip = t == "--epo" or t.startswith("--epo-")
        if not skip:
            out.append(t)
    return out


def rls1_refusals(prof: dict, argv: list, mod, parser) -> tuple:
    """(refusals, extra live terms) of the RL-S1 arm rules.  The arm is DECLARED (`prof['declared_arm']`, the CLI's --arm-name) or, undeclared, read off the
    argv by `rls1_arm_class`; a declared arm is what catches an OMITTED flag (an `--epo` left out of R0 would otherwise read as a clean control).  Beyond
    the generic required/forbidden flags: the arm's own flags, the registered lever values, the recipe, the lane-head pairing, scorer_finetune's OWN refusals,
    and exact flag spelling (argparse accepts unique prefixes; the gate's presence tests are exact tokens)."""
    import gate_checks as C
    out = []
    name = prof["name"]
    arm_class = prof.get("arm_class", RLS1_ARM_CLASS)                    # refe_rls1b carries its own table (RL-S1b); refe_rls1 keeps the module constants
    arm_seeds = prof.get("arm_seeds", RLS1_ARM_SEEDS)
    declared = prof.get("declared_arm")
    if declared is not None and declared not in arm_class:
        out.append(f"{name}: unknown arm {declared!r} (one of {sorted(arm_class)})")
        declared = None
    cls = arm_class[declared] if declared else rls1_arm_class(argv)
    who = declared or f"the {cls} arm class"
    if cls != "R?" and cls not in set(arm_class.values()):
        out.append(f"{name}: the argv is an {cls}-class arm, which {name} does not register (its arms: {sorted(arm_class)})"
                   + (f"; {'; '.join(prof.get('arms_need_pi', {}).values())}" if prof.get("arms_need_pi") else ""))
    ns, err = C.parse_scorer_argv(parser, argv)
    if err:
        out.append(f"{name}: the trainer's parser rejects this argv ({err})")
        return out, ()
    known = {o for x in parser._actions for o in x.option_strings}
    for t in argv:
        if t.startswith("--") and t not in known:
            out.append(f"{name}: {t} is not an exact option of the trainer (argparse accepts an abbreviation; the gate's flag tests are on exact tokens)")
    out += [f"scorer_finetune refuses this argv: {m}" for m in mod.epo_refusals(ns, parser)]

    def how(flag):
        return "named" if flag in argv else "the parser DEFAULT"
    epo_opts = sorted(o for o in known if o == "--epo" or o.startswith("--epo-"))
    named = [o for o in epo_opts if o in argv]
    if cls == "C":
        if named:
            out.append(f"{name}: {who} must NOT carry {named} -- the control is the BCE-only arm, and R vs C must differ by the EPO term alone")
    else:
        for f in ("--epo", "--epo-terms-train"):
            if f not in argv:
                out.append(f"{name} requires {f} for {who}; the argv does not carry it")
        if cls == "R?":
            out.append(f"{name}: exactly one of --epo-k / --epo-l: R0/R1 carry neither (the core reward), R2 carries both")
        if cls == "R":
            for f in ("--epo-k", "--epo-l", "--epo-lam-k", "--epo-lam-l"):
                if f in argv:
                    out.append(f"{name}: {who} (R0/R1) is the CORE reward (PDMS - 0.15 D + guard); {f} belongs to R2 only")
        if cls == "R2":
            for f in ("--epo-k", "--epo-l"):
                if f not in argv:
                    out.append(f"{name} requires {f} for {who}; the argv does not carry it")
        for lam_f, on_f in (("--epo-lam-k", "--epo-k"), ("--epo-lam-l", "--epo-l")):
            if lam_f in argv and on_f not in argv:
                out.append(f"{name}: {lam_f} is named without {on_f}: a weight for a term that is OFF (scorer_finetune would ignore it silently)")
        reg = dict(prof.get("epo_registered", RLS1_EPO_REGISTERED))
        if cls == "R2":
            reg.update(RLS1_R2_REGISTERED)
        for dest, want in reg.items():
            f = "--" + dest.replace("_", "-")
            if not C._close(getattr(ns, dest), want):
                out.append(f"{name} requires {f} {want} (PREREG_RL_S1 lever); the argv resolves to {getattr(ns, dest)} ({how(f)})")
    for dest, want in RLS1_RECIPE.items():
        f = "--" + dest.replace("_", "-")
        if not C._close(getattr(ns, dest), want):
            out.append(f"{name} requires {f} {want} (PREREG_RL_S1 recipe, every arm); the argv resolves to {getattr(ns, dest)} ({how(f)})")
    if declared and ns.seed != arm_seeds[declared]:
        out.append(f"{name}: arm {declared} is registered with --seed {arm_seeds[declared]}; the argv resolves to {ns.seed} ({how('--seed')})")
    if ns.lane_head and not (ns.teacher_lane_train and ns.teacher_lane_heldout):
        out.append(f"{name}: --lane-head needs --teacher-lane-train AND --teacher-lane-heldout (the 7th output would have no label)")
    if (ns.teacher_lane_train or ns.teacher_lane_heldout) and not ns.lane_head:
        out.append(f"{name}: --teacher-lane-* is named without --lane-head: the labels would be loaded and never trained on")
    return out, (RLS1_EPO_TERMS if cls != "C" else ())


# ============================================================================================ aux S1 arm rules (refe_aux_s1)
def auxs1_arm_class(argv: list) -> str:
    """The arm class an argv DECLARES by what it carries: no --aux-drv = N (the no-head replicate); --aux-drv with `--aux-drv-feed on` = H2; otherwise H1."""
    if "--aux-drv" not in argv:
        return "N"
    return "H2" if (GT.flag_values(argv, "--aux-drv-feed") or ["off"])[0] == "on" else "H1"


def auxs1_control_argv(argv: list) -> list:
    """The matched no-head argv of a head arm: the SAME argv minus every head lever (flag and value), so a head arm and N differ by the head alone.  (--out and --seed are the
    caller's: two arms never share an output directory.)"""
    out, skip = [], False
    for t in argv:
        if t.startswith("--"):
            skip = t in AUX_HEAD_FLAGS
        if not skip:
            out.append(t)
    return out


def auxs1_refusals(prof: dict, argv: list, mod, parser) -> tuple:
    """(refusals, extra live terms) of the aux S1 arm rules.  The arm is DECLARED (`prof['declared_arm']`) or, undeclared, read off the argv.  A head arm must NAME every flag whose
    default disables it (--aux-drv, --w-aux-drv, --aux-drv-lr, --aux-drv-labels; --aux-drv-feed on for H2), carry the registered values, and resolve to the declared class and seed;
    N must carry no head lever at all.  Every arm carries the SFT-4 recipe.  scorer_finetune's OWN refusals (epo + aux) are surfaced, and flags are matched as exact tokens."""
    import gate_checks as C
    out = []
    name = prof["name"]
    declared = prof.get("declared_arm")
    if declared is not None and declared not in AUXS1_ARM_CLASS:
        out.append(f"{name}: unknown arm {declared!r} (one of {sorted(AUXS1_ARM_CLASS)})")
        declared = None
    argv_cls = auxs1_arm_class(argv)
    cls = AUXS1_ARM_CLASS[declared] if declared else argv_cls
    who = declared or f"the {cls} arm class"
    ns, err = C.parse_scorer_argv(parser, argv)
    if err:
        out.append(f"{name}: the trainer's parser rejects this argv ({err})")
        return out, ()
    known = {o for x in parser._actions for o in x.option_strings}
    for t in argv:
        if t.startswith("--") and t not in known:
            out.append(f"{name}: {t} is not an exact option of the trainer (argparse accepts an abbreviation; the gate's flag tests are on exact tokens)")
    out += [f"scorer_finetune refuses this argv: {m}" for m in mod.epo_refusals(ns, parser) + mod.AUX.aux_refusals(ns) + mod.aux_s1_refusals(ns)]

    def how(flag):
        return "named" if flag in argv else "the parser DEFAULT"
    if declared and argv_cls != cls:
        out.append(f"{name}: arm {declared} is the {cls} class; the argv resolves to the {argv_cls} class "
                   f"({'--aux-drv is absent' if argv_cls == 'N' else '--aux-drv-feed ' + ('on' if argv_cls == 'H2' else 'is not on')})")
    named = [o for o in AUX_HEAD_FLAGS if o in argv]
    if cls == "N":
        if named:
            out.append(f"{name}: {who} must NOT carry {named} -- the no-head replicate differs from the head arms by the head alone")
    else:
        for f in ("--aux-drv", "--w-aux-drv", "--aux-drv-lr", "--aux-drv-labels"):
            if f not in argv:
                out.append(f"{name} requires {f} for {who}; the argv does not carry it (a flag whose default disables the head must be NAMED, or the arm did not test it)")
        if cls == "H2" and (GT.flag_values(argv, "--aux-drv-feed") or ["off"])[0] != "on":
            out.append(f"{name} requires --aux-drv-feed on for {who}; the argv does not carry it")
        for dest, want in AUXS1_HEAD_REGISTERED.items():
            f = "--" + dest.replace("_", "-")
            if not C._close(getattr(ns, dest), want):
                out.append(f"{name} requires {f} {want} (the registered head lever); the argv resolves to {getattr(ns, dest)} ({how(f)})")
    for dest, want in AUXS1_RECIPE.items():
        f = "--" + dest.replace("_", "-")
        if not C._close(getattr(ns, dest), want):
            out.append(f"{name} requires {f} {want} (the SFT-4 recipe, every arm); the argv resolves to {getattr(ns, dest)} ({how(f)})")
    if declared and ns.seed != AUXS1_ARM_SEED[declared]:
        out.append(f"{name}: arm {declared} is registered with --seed {AUXS1_ARM_SEED[declared]}; the argv resolves to {ns.seed} ({how('--seed')})")
    if ns.lane_head and not (ns.teacher_lane_train and ns.teacher_lane_heldout):
        out.append(f"{name}: --lane-head needs --teacher-lane-train AND --teacher-lane-heldout (the 7th output would have no label)")
    return out, (AUXS1_HEAD_TERMS if cls != "N" else ())


# ============================================================================================ orchestration
def derived_argvs(prof: dict, argv: list, work: Path, gate_sets=None, gate_pdm=None, gate_bank=None) -> tuple:
    """(build_argv, smoke_argv).  BUILD keeps every launch lever (the schedule is observed) and points the on-policy side at the gate's small
    sets; SMOKE additionally swaps the bank for the carved gate bank, cuts the schedule to `smoke_steps`, and zeroes the workers."""
    from gate_checks import _drop, _set
    b = list(argv)
    b = _drop(b, "--resume")
    if gate_sets:
        b = _set(b, "--onpolicy-targets", gate_sets)
    if gate_pdm:
        b = _set(b, "--pdm-labels", gate_pdm)
    b = _set(b, "--out", str(work / "build_run"))
    s = list(b)
    if gate_bank:
        s = _set(s, "--targets", gate_bank)
        if (GT.flag_values(s, "--labeller") or ["bank"])[0] != "online":         # the online labeller reads no scorer-targets file
            s = _set(s, "--scorer-targets", gate_bank)
    for flag in ("--epochs", "--grow-scenes"):
        if flag in s:
            i = s.index(flag)
            del s[i:i + 2]
    s = _set(s, "--steps", str(prof["smoke_steps"]))
    s = _set(s, "--accum", str(min(int((GT.flag_values(s, "--accum") or ["1"])[0]), prof["smoke_accum"])))
    s = _set(s, "--workers", "0")
    s = _set(s, "--log-every", "1")
    s = _set(s, "--ckpt-every-min", "999")
    if prof.get("rehearsal"):
        s = _drop(s, "--compile")
    s = _set(s, "--out", str(work / "smoke_run"))
    return b, s


def write_evidence(out_dir: Path, ev: dict, binding: dict, data: dict) -> Path:
    ev = dict(ev)
    ev["binding"] = binding
    ev["inputs_read"] = {k: v for k, v in data.items()}
    return GT.write_json(out_dir / f"evidence_{ev['check']}.json", ev)


def gate_inputs_of(p2_gate_json=None, lab_json=None, mem_metrics=None, egr_json=None, preflight_json=None) -> dict:
    return {k: v for k, v in (("gate:p2_gate_json", p2_gate_json), ("gate:lab_json", lab_json), ("gate:mem_metrics", mem_metrics),
                              ("gate:egr_json", egr_json), ("gate:preflight_json", preflight_json)) if v}


def run_checks(prof: dict, argv: list, pkg: Path, out_dir: Path, *, only=None, arm=None, gate_sets=None, gate_pdm=None, gate_bank=None,
               p2_gate_json=None, commit=None, lab_json=None, mem_metrics=None, egr_json=None, preflight_json=None) -> dict:
    """Run every check of the profile (or `only`), writing evidence files.  Returns {check: status}."""
    import gate_checks as C
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    tree = GT.tree_manifest(pkg)
    binding = GT.binding_of(prof["name"], commit, GT.tree_digest(tree), argv)
    gate_inputs = gate_inputs_of(p2_gate_json, lab_json, mem_metrics, egr_json, preflight_json)
    data = GT.data_manifest(argv, prof["data_flags"], gate_inputs)
    rehearsal = bool(prof.get("rehearsal"))
    want = [c for c in (prof["checks"] + tuple(prof.get("na_ok", ()))) if (only is None or c in only)]
    status: dict = {}
    refusals = profile_refusals(prof, argv)

    def emit(ev):
        write_evidence(out_dir, ev, binding, data)
        status[ev["check"]] = ev["status"]
        print(f"  {ev['check']:9s} {ev['status']:5s} {('; '.join(ev['reasons']))[:200]}", flush=True)
    if refusals:
        for c in want:
            emit(C.evidence(c, "FAIL", ["profile refusal: " + r for r in refusals], {"refusals": refusals}, arm=arm, rehearsal=rehearsal))
        return status
    if prof.get("runner") == "scorer":                      # scorer_finetune.py: no BUILD/SMOKE capture (that is train.py's) -- static checks + the trainer's own preflight
        pf = C.load_scorer_preflight(preflight_json) if preflight_json else None
        if "G-HYG" in want:
            emit(C.check_hyg(arm=arm, rehearsal=rehearsal, profile=prof))
        if "G-DVB" in want:
            emit(C.check_dvb_scorer(argv, pf, prof, arm=arm, rehearsal=rehearsal))
        if "G-PREFLIGHT" in want:
            emit(consume_scorer_preflight(preflight_json, argv, prof, arm=arm, rehearsal=rehearsal))
        if "G-SUITE" in want:
            emit(C.check_suite(prof, arm=arm, rehearsal=rehearsal))
        if "G-CLOCK" in want:
            emit(C.evidence("G-CLOCK", "NA", [prof.get("clock_na", "no label clock in this profile")], {}, arm=arm, rehearsal=rehearsal))
        return status
    b_argv, s_argv = derived_argvs(prof, argv, work, gate_sets, gate_pdm, gate_bank)
    need_b = any(c in want for c in ("G-DVB", "G-CLOCK"))
    need_s = any(c in want for c in ("G-LIVE", "G-EVAL", "G-DVB", "G-LIVE-LAB"))
    b = C.instrumented_run(b_argv, mode="build", work=work / "build", arm=arm) if need_b else None
    if b is not None:
        _free(b)                      # the BUILD model is not needed again (a second ViT-L would sit on the pod's GPU through the smoke)
    s = C.instrumented_run(s_argv, mode="smoke", work=work / "smoke", arm=arm) if need_s else None
    prof2 = dict(prof)
    if "G-HYG" in want:
        emit(C.check_hyg(arm=arm, rehearsal=rehearsal))
    if "G-DVB" in want:
        emit(C.check_dvb(b_argv, b, s, prof2, arm=arm, rehearsal=rehearsal))
    if "G-LIVE" in want:
        emit(C.check_live(s_argv, s, prof2, arm=arm, rehearsal=rehearsal))
    if "G-CLOCK" in want:
        emit(C.check_clock(b_argv, b, prof2, arm=arm, rehearsal=rehearsal))
    if "G-EVAL" in want:
        emit(C.check_eval(s_argv, s, prof2, arm=arm, rehearsal=rehearsal))
    if "G-CKPT" in want:
        # the round trip runs TWO real trainings in subprocesses: the smoke model, its optimizer and the caching allocator behind them
        # must leave THIS process's GPU first (MEASURED 2026-10-12, P2 pod: this process held 24.11 GiB, U and H1 both died CUDA OOM on 44 GiB)
        if s is not None:
            _free(s, deep=True)
        _log_gpu_reserved("before G-CKPT")
        emit(C.check_ckpt(s_argv, work / "ckpt", prof2, arm=arm, rehearsal=rehearsal))
    if "G-SUITE" in want:
        emit(C.check_suite(prof2, arm=arm, rehearsal=rehearsal))
    if "G-P2GATE" in want:
        emit(consume_p2_gate(p2_gate_json, arm=arm, rehearsal=rehearsal))
    if "G-LIVE-LAB" in want:
        emit(C.check_live_lab(lab_json, s, argv, arm=arm, rehearsal=rehearsal))
    if "G-LIVE-MEM" in want:
        emit(C.check_live_mem(mem_metrics, arm=arm, rehearsal=rehearsal))
    if "G-EGR" in want:
        emit(C.check_egr(egr_json, arm=arm, rehearsal=rehearsal))
    return status


def _free(cap, deep: bool = False) -> None:
    import gc
    cap.model = None
    cap.first_batch = None
    if deep:            # everything else that can pin the model's parameters or its CUDA blocks; the metrics stay (G-LIVE-LAB reads them)
        cap.opt = cap.sched = None
        cap.loaders, cap.samplers = [], []
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                                        # noqa: BLE001 -- best effort
        pass


def _log_gpu_reserved(when: str) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU reserved by the gate process {when}: {torch.cuda.memory_reserved() / 2 ** 30:.2f} GiB", flush=True)
    except Exception:                                                        # noqa: BLE001 -- evidence only
        pass


def consume_p2_gate(path, arm=None, rehearsal=False) -> dict:
    """p2_gate.py's own output (G1 warm start == the planner's load; G2 the analytic pair, incl. the detached arm that must read EXACTLY 0.0;
    G3 pdm-only; G4 the real trainer smoke) is a SUB-CHECK, not duplicated: parsed, never grepped, all four present and ok.  Its file's sha256 is
    bound into the token (`gate:p2_gate_json`)."""
    import gate_checks as C
    if not path or not Path(path).is_file():
        return C.evidence("G-P2GATE", "FAIL", [f"p2_gate.py output missing: {path}"], {}, arm=arm, rehearsal=rehearsal)
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    g = d.get("gates", {})
    bad = [p for p in ("G1", "G2", "G3", "G4") if not any(k.startswith(p + "_") and v.get("ok") is True for k, v in g.items())]
    reasons = []
    if d.get("verdict") != "PASS" or bad:
        reasons.append(f"p2_gate.py is not PASS: verdict={d.get('verdict')} missing_or_not_ok={bad}")
    return C.verdict("G-P2GATE", reasons, {"gates": {k: v.get("ok") for k, v in g.items()}, "file_sha256": GT.sha256_file(path)}, arm=arm, rehearsal=rehearsal)


def consume_scorer_preflight(path, argv, prof, arm=None, rehearsal=False) -> dict:
    """`scorer_finetune.py --preflight`'s own gates (G1 trainable == exactly the scorer; G2 live gradients; G3/G4 the identities; G5 the planner-loader
    round trip; G7 held-out disjoint; G9 PDM targets; G10 the lane head's label; G11 the EPO term live with its zero controls) are a SUB-CHECK, not
    duplicated: PARSED, never grepped; every gate the ARGV obliges must be present and ok, every gate present must be ok, and a gate the argv does NOT
    oblige (G11 on a control) means the preflight ran ANOTHER configuration.  G11 is re-judged from its own fields, not trusted on its `ok`.  The file's
    sha256 is bound into the token (`gate:preflight_json`).  (There is no G6: the trainer numbers 1-5 and 7-11.)"""
    import gate_checks as C
    import math
    pf = C.load_scorer_preflight(path)
    if pf["error"]:
        return C.evidence("G-PREFLIGHT", "FAIL", [pf["error"]], {}, arm=arm, rehearsal=rehearsal)
    mod = importlib.import_module(prof.get("parser_module", "scorer_finetune"))
    ns, err = C.parse_scorer_argv(mod.build_parser(), argv)
    if err:
        return C.evidence("G-PREFLIGHT", "FAIL", [f"the trainer's parser rejects the argv: {err}"], {"file_sha256": pf["sha256"]}, arm=arm, rehearsal=rehearsal)
    row = pf["preflight"]
    if row is None:
        return C.evidence("G-PREFLIGHT", "FAIL", ["the file carries no event=preflight row (not a scorer_finetune --preflight log)"],
                          {"file_sha256": pf["sha256"]}, arm=arm, rehearsal=rehearsal)
    gates = row.get("gates") if isinstance(row.get("gates"), dict) else {}
    reasons = []
    if row.get("verdict") != "PASS":
        reasons.append(f"the preflight verdict is {row.get('verdict')!r}, not PASS")
    req = C.scorer_required_gates(ns)
    prefix = {k: k.split("_", 1)[0] for k in gates}
    missing = [p for p in req if p not in prefix.values()]
    if missing:
        reasons.append(f"required preflight gate(s) ABSENT: {missing}" + (" -- G11 is the EPO term's live/identity gate; an --epo arm without it has not been proved live"
                                                                           if "G11" in missing else ""))
    stray = sorted(k for k, p in prefix.items() if p in C.SCORER_GATES and p not in req)
    if stray:
        reasons.append(f"gate(s) {stray} are present but this argv does not oblige them: the preflight ran a DIFFERENT configuration")
    unknown = sorted(k for k, p in prefix.items() if p not in C.SCORER_GATES)
    if unknown:
        reasons.append(f"gate(s) {unknown} are not known to the gate: the consumer must be updated before it can judge them")
    bad = sorted(k for k, v in gates.items() if not (isinstance(v, dict) and v.get("ok") is True))
    if bad:
        reasons.append(f"preflight gate(s) not ok: {bad}")
    g11 = next((v for k, v in gates.items() if prefix[k] == "G11"), None)
    if isinstance(g11, dict):
        mods = set(C.SCORER_GROUPS)
        on, lam0, off = (g11.get(k) or {} for k in ("per_module_on", "per_module_lam0", "per_module_off"))
        if set(on) != mods or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0.0 for x in on.values()):
            reasons.append(f"G11: the EPO-only backward is not finite and non-zero on every scorer module {sorted(mods)}: {on}")
        if g11.get("controls_exactly_zero") is not True or any(x != 0.0 for d in (lam0, off) for x in d.values()) or not (lam0 and off):
            reasons.append("G11: the zero controls (lam 0 / --epo off) do not read EXACTLY 0.0 -- the probe is not proved able to read zero")
        if g11.get("kl_ok") is not True:
            reasons.append(f"G11: KL(pi_SFT || pi) at step 0 is {g11.get('kl_step0')} (tol {g11.get('kl_tol')}): arm B did not start AT the base")
        if g11.get("mutate_detach"):
            reasons.append("G11 was produced under REFE_RL_MUTATE_EPO_DETACH=1 (a regression arm): it can never count")
    g12 = next((v for k, v in gates.items() if prefix[k] == "G12"), None)
    if g12 is not None:                                      # G12 (--aux-drv) is re-judged from its raw fields too: its own `ok` is a claim, the fields are the evidence
        reasons += C.judge_g12(g12, ns)
    det = {"gates": {k: (v.get("ok") if isinstance(v, dict) else None) for k, v in gates.items()}, "required": req, "verdict": row.get("verdict"),
           "file_sha256": pf["sha256"], "n_preflight_rows": pf["n_preflight"], "has_data_event": pf["data"] is not None,
           "s_per_microbatch": row.get("s_per_microbatch"), "gpu_mem_gb": row.get("gpu_mem_gb")}
    return C.verdict("G-PREFLIGHT", reasons, det, arm=arm, rehearsal=rehearsal)


def collect(out_dir: Path, check: str):
    p = out_dir / f"evidence_{check}.json"
    return GT.read_json(p) if p.is_file() else None


def finalize(prof: dict, argv: list, pkg: Path, out_dir: Path, key: bytes, *, p2_gate_json=None, commit=None, lab_json=None, mem_metrics=None,
             egr_json=None, preflight_json=None):
    """Mint the token from the evidence ON DISK.  Every required check must have usable PASS evidence bound to this launch."""
    tree = GT.tree_manifest(pkg)
    gate_inputs = gate_inputs_of(p2_gate_json, lab_json, mem_metrics, egr_json, preflight_json)
    data = GT.data_manifest(argv, prof["data_flags"], gate_inputs)
    binding = GT.binding_of(prof["name"], commit, GT.tree_digest(tree), argv)
    reasons = profile_refusals(prof, argv) + GT.data_problems(data)
    manifest = GT.read_json(pkg / GT.SHIP_MANIFEST) if (pkg / GT.SHIP_MANIFEST).is_file() else None
    if manifest is not None:
        commit = manifest.get("commit")
        binding = GT.binding_of(prof["name"], commit, GT.tree_digest(tree), argv)
        if manifest.get("files") != tree:
            reasons.append(f"the pod's code tree is not the shipped tree: {GT.tree_diff(manifest.get('files') or {}, tree)}")
    if prof.get("require_commit") and not commit:
        reasons.append("no commit is bound: ship SHIP_MANIFEST.json (launch_gate.py ship-manifest --commit <landed 40-hex>) with the package")
    if not prof.get("mintable", True):
        reasons.append(f"profile {prof['name']} is a REHEARSAL profile: it can never mint a token")
    if os.environ.get("TANITAD_GATE_ARM"):
        reasons.append(f"a regression arm is active ({os.environ['TANITAD_GATE_ARM']}): an arm run can never PASS")
    checks: dict = {}
    for c in prof["checks"] + tuple(prof.get("na_ok", ())):
        ev = collect(out_dir, c)
        if ev is None:
            checks[c] = {"status": "MISSING"}
            if c in prof["checks"]:
                reasons.append(f"{c}: no evidence file")
            continue
        why = []
        if ev.get("binding") != binding:
            why.append("binding differs")
        if ev.get("arm"):
            why.append(f"produced under regression arm {ev['arm']}")
        if ev.get("rehearsal"):
            why.append("REHEARSAL evidence never counts toward a token")
        for f, fp in (ev.get("inputs_read") or {}).items():
            mine = data.get(f)
            if mine is None or GT.fp_key(mine) != GT.fp_key(fp):
                why.append(f"input {f} was read with a different fingerprint")
        ok_status = ("PASS",) if c in prof["checks"] else ("PASS", "NA")
        status = ev.get("status") if not why else "REJECTED"
        checks[c] = {"status": status if status in ok_status or status == "REJECTED" else status, "evidence_sha256": GT.sha256_file(out_dir / f"evidence_{c}.json"),
                     "finished_utc": ev.get("finished_utc"), "reasons": (ev.get("reasons") or [])[:6] + why}
        if status not in ok_status:
            reasons.append(f"{c} {status}: " + " | ".join((ev.get("reasons") or [])[:2] + why))
        if c in prof.get("na_ok", ()) and ev.get("status") == "NA":
            checks[c]["status"] = "PASS"
            checks[c]["na_reason"] = (ev.get("reasons") or [""])[0]
    return GT.mint(profile=prof["name"], commit=commit, tree_man=tree, argv=argv, trainer=prof["trainer"], data=data, checks=checks,
                   reasons=reasons, key=key, out_dir=out_dir, gate_inputs=gate_inputs, gate_self_sha=GT.sha256_file(HERE / "launch_gate.py"))


# ============================================================================================ CLI
def _argv_of(a) -> list:
    if getattr(a, "args_file", None):
        return GT.read_json(a.args_file)
    if getattr(a, "args", None) is not None:
        return parse_args_string(a.args)
    raise SystemExit("--args or --args-file is required")


def _profile_for(a, minting: bool) -> dict:
    """The profile, with the DECLARED arm folded in (rls1_refusals reads `declared_arm`).  A profile with arm rules must be MINTED for one named arm: an arm read
    off the argv cannot notice an omitted flag (an `--epo` left out of R0 reads as a clean control)."""
    prof = PROFILES[a.profile]
    arm_name = getattr(a, "arm_name", None)
    arms = {"rls1": prof.get("arm_class", RLS1_ARM_CLASS), "auxs1": AUXS1_ARM_CLASS}.get(prof.get("arm_rules"), {})
    if arm_name:
        if not prof.get("arm_rules"):
            raise SystemExit(f"--arm-name is only meaningful for a profile with arm rules (refe_rls1, refe_rls1b, refe_aux_s1), not {a.profile}")
        if arm_name in prof.get("arms_need_pi", {}):
            raise SystemExit(f"REFUSED: {prof['arms_need_pi'][arm_name]}")
        if arm_name not in arms:
            raise SystemExit(f"REFUSED: {arm_name!r} is not an arm of {a.profile} (one of {'|'.join(sorted(arms))})")
        prof = {**prof, "declared_arm": arm_name}
    elif prof.get("arm_rules") and minting:
        raise SystemExit(f"REFUSED: {a.profile} mints for ONE named arm: pass --arm-name {'|'.join(sorted(arms))}")
    return prof


def cmd_run(a) -> int:
    prof = _profile_for(a, minting=True)
    argv = _argv_of(a)
    pkg, out = Path(a.pkg), Path(a.out_dir)
    key = GT.load_key(a.key_file, create=True, forbid_under=[pkg])
    man = GT.read_json(pkg / GT.SHIP_MANIFEST) if (pkg / GT.SHIP_MANIFEST).is_file() else None
    commit = man.get("commit") if man else a.commit
    arm = os.environ.get("TANITAD_GATE_ARM")
    if a.gate_bank and not Path(a.gate_bank).exists():
        import gate_fixture as F
        src = (GT.flag_values(argv, "--targets") or [None])[0]
        if (GT.flag_values(argv, "--labeller") or ["bank"])[0] == "online":        # P6: rows whose D9 parts exist (no on-policy sets in this profile)
            mc = (GT.flag_values(argv, "--metric-cache") or [None])[0]
            print("  carved the P6 smoke bank:", F.carve_p6_gate_bank(src, mc, a.gate_bank, a.gate_bank_n), flush=True)
        elif a.gate_sets:
            print("  carved the smoke bank:", F.carve_gate_bank(src, a.gate_sets, a.gate_bank), flush=True)
    run_checks(prof, argv, pkg, out, only=set(a.only.split(",")) if a.only else None, arm=arm, gate_sets=a.gate_sets, gate_pdm=a.gate_pdm,
               gate_bank=a.gate_bank, p2_gate_json=a.p2_gate_json, commit=commit, lab_json=a.gate_lab_json, mem_metrics=a.gate_mem_metrics,
               egr_json=a.gate_egr_json, preflight_json=a.gate_preflight_json)
    verdict, path, tok = finalize(prof, argv, pkg, out, key, p2_gate_json=a.p2_gate_json, commit=commit, lab_json=a.gate_lab_json,
                                  mem_metrics=a.gate_mem_metrics, egr_json=a.gate_egr_json, preflight_json=a.gate_preflight_json)
    for r in tok["reasons"][:8]:
        print("  REASON:", r[:300])
    print(f"verdict {verdict} -> {path}", flush=True)
    print(f"ZZGATE-{verdict}-{(tok['binding']['commit'] or tok['binding']['tree_sha256'])[:12]}ZZ", flush=True)
    return 0 if verdict == "PASS" else 1


def cmd_verify(a) -> int:
    prof = PROFILES.get(a.profile) if a.profile else None
    try:
        key = GT.load_key(a.key_file, create=False, forbid_under=[a.pkg])
        ok, reasons, info = GT.verify_token(a.token, _argv_of(a), a.pkg, key=key, profile=a.profile, data_flags=prof["data_flags"] if prof else (),
                                            commit=a.commit)
    except Exception as e:                                                   # noqa: BLE001 -- refuse, never pass
        ok, reasons, info = False, [f"{type(e).__name__}: {e}"], {}
    for r in reasons:
        print("REFUSED:", r[:400], file=sys.stderr, flush=True)
    if a.json:
        GT.write_json(a.json, {"verdict": "MATCH" if ok else "REFUSED", "reasons": reasons, "info": info, "utc": GT.utc_now()})
    print(f"ZZGATE-{'MATCH' if ok else 'REFUSED'}ZZ", flush=True)
    return 0 if ok else 5


def cmd_ship(a) -> int:
    man = GT.ship_manifest(a.pkg, commit=a.commit, git_dir=a.git_dir, repo_prefix=a.repo_prefix)
    GT.write_json(a.out, man)
    print(f"ZZSHIP_OK {len(man['files'])} files tree {man['tree_sha256'][:16]} commit {a.commit and a.commit[:12]}", flush=True)
    return 0


def cmd_refusals(a) -> int:
    rs = profile_refusals(_profile_for(a, minting=False), _argv_of(a))
    for r in rs:
        print("REFUSED:", r)
    print(f"ZZGATE-{'REFUSED' if rs else 'CLEAN'}ZZ")
    return 5 if rs else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "verify", "refusals"):
        p = sub.add_parser(name)
        p.add_argument("--profile", required=(name != "verify"), choices=sorted(PROFILES))
        p.add_argument("--args")
        p.add_argument("--args-file")
        p.add_argument("--key-file")
        p.add_argument("--pkg", required=(name != "refusals"))
        if name in ("run", "refusals"):
            p.add_argument("--arm-name", choices=sorted(set(RLS1_ARM_CLASS) | set(RLS1B_ARM_CLASS) | set(RLS1B_NEEDS_PI)) + sorted(AUXS1_ARM_CLASS),
                           help="refe_rls1: the arm this argv is for (C0 C1 R0 R1 R2); refe_rls1b: C1 R0b R1b (R2b needs a PI decision); refe_aux_s1: N-s0..H2-s2; "
                                "REQUIRED to mint, optional for `refusals` (undeclared = read off the argv)")
        if name == "run":
            p.add_argument("--out-dir", required=True)
            p.add_argument("--only")
            p.add_argument("--commit")
            p.add_argument("--gate-sets")
            p.add_argument("--gate-pdm")
            p.add_argument("--gate-bank")
            p.add_argument("--gate-bank-n", type=int, default=64, help="refe_p6: rows per targets_rank*.jsonl of the carved smoke bank (rows whose D9 parts exist)")
            p.add_argument("--p2-gate-json")
            p.add_argument("--gate-lab-json", help="refe_p6: diag_p6_labeller_identity.py --out (G-LIVE-LAB)")
            p.add_argument("--gate-mem-metrics", help="refe_p6: metrics.jsonl of a >= 200-micro-batch run on the real card (G-LIVE-MEM)")
            p.add_argument("--gate-egr-json", help="refe_p6: hf_egress.py gate's g_egr_result.json (G-EGR)")
            p.add_argument("--gate-preflight-json", help="refe_rls1: scorer_finetune.py --preflight's preflight.jsonl (G-PREFLIGHT; also G-DVB's built side)")
        if name == "verify":
            p.add_argument("--token", required=True)
            p.add_argument("--commit")
            p.add_argument("--json")
    p = sub.add_parser("ship-manifest")
    p.add_argument("--pkg", required=True)
    p.add_argument("--commit")
    p.add_argument("--git-dir")
    p.add_argument("--repo-prefix", default="")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "verify": cmd_verify, "refusals": cmd_refusals, "ship-manifest": cmd_ship}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
