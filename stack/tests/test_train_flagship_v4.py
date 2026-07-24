"""flagship v4 P4 trainer — scripts/train_flagship_v4.py.

Pins the trainer's tested surface: the joint loss-assembly step is finite and
differentiable across the λ_plan phases, the §17 preflight catches the launch
mistakes it is meant to (O-17 gate-in-ramp, X15 zero-fill, [PM] #2 rollout-k), and
the §16 one-lever reproduction diffs parse. The multi-day training LOOP itself is
not exercised here (it is not launched anywhere — Sayed owns the go/no-go)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_flagship_v4 as T  # noqa: E402


# --------------------------------------------------- the joint loss step ----
def test_smoke_joint_step_is_finite_across_phases():
    out = T.smoke()
    steps = [s for s, _ in out["logs"]]
    assert steps == [0, 4, 8]
    # λ_plan really moves 0 -> ramp -> 1 across Phase A/B/C
    lam = [log["lambda_plan"] for _, log in out["logs"]]
    assert lam[0] == 0.0 and lam[-1] == 1.0 and 0.0 < lam[1] < 1.0
    # every component present and finite
    for _, log in out["logs"]:
        for k in ("total", "wm", "planner", "plan_ade", "oracle_ade"):
            assert k in log and log[k] == log[k]        # not NaN


# ------------------------------------------------------ the §17 preflight ---
def _args(**over):
    argv = ["--print-launch"]
    for k, v in over.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is False:
            argv.append("--no-" + k.replace("_", "-") if k == "dense_plan"
                        else flag)
        else:
            argv += [flag, str(v)]
    return T.build_parser().parse_args(argv)


def test_preflight_clean_default_passes():
    assert T.preflight_asserts(_args()) == []


def test_preflight_catches_o17_gate_inside_ramp():
    problems = T.preflight_asserts(_args(phase_b_steps=12000))   # gate 10000 < 12000
    assert any("O-17" in p for p in problems)


def test_preflight_catches_ego_zero_fill():
    a = T.build_parser().parse_args(["--print-launch", "--ego-zero-fill"])
    problems = T.preflight_asserts(a)
    assert any("X15" in p for p in problems)


def test_preflight_catches_rollout_k_raise():
    problems = T.preflight_asserts(_args(rollout_k=8))
    assert any("rollout-k" in p or "#2" in p for p in problems)


def test_preflight_catches_wrong_effective_batch():
    """⭐ same-as-v1: effective batch must be 64 (16x4). v4.1's accum-1 (=16) is caught."""
    problems = T.preflight_asserts(_args(accum=1))          # 16*1 = 16 != 64
    assert any("effective batch" in p for p in problems)
    assert T.preflight_asserts(_args(accum=4)) == []        # 16*4 = 64 passes clean


# --------------------------------------- ⭐ the v4 FROM-SCRATCH fallback path ---
def test_from_scratch_flag_and_trunk_none_sentinel_detected():
    """--from-scratch AND the --trunk none sentinel both select the random-init path;
    a normal --trunk path does not."""
    assert T._is_from_scratch(T.build_parser().parse_args(["--from-scratch"]))
    assert T._is_from_scratch(T.build_parser().parse_args(["--trunk", "none"]))
    assert T._is_from_scratch(T.build_parser().parse_args(["--trunk", "NONE"]))
    assert not T._is_from_scratch(T.build_parser().parse_args(["--trunk", "/ckpt/v1.pt"]))
    assert not T._is_from_scratch(T.build_parser().parse_args([]))


def test_from_scratch_preflight_clean_without_trunk():
    """The fallback needs NO --trunk and must pass the §17 preflight clean (the
    not-frozen gate is satisfied trivially from random init)."""
    assert T.preflight_asserts(_args(from_scratch=True)) == []
    a = T.build_parser().parse_args(["--print-launch", "--trunk", "none"])
    assert T.preflight_asserts(a) == []


def test_from_scratch_conflicts_with_a_real_trunk():
    """--from-scratch together with a REAL --trunk is ambiguous and must be caught
    before a GPU-day (the trunk would be built then discarded)."""
    a = T.build_parser().parse_args(
        ["--print-launch", "--from-scratch", "--trunk", "/ckpt/v1.pt"])
    problems = T.preflight_asserts(a)
    assert any("from-scratch" in p.lower() for p in problems)


def test_staged_command_from_scratch_emits_flag_and_omits_trunk():
    """The staged launch command carries --from-scratch and NO --trunk; the warm-
    start command is the mirror image."""
    fs = T.build_parser().parse_args(
        ["--print-launch", "--from-scratch",
         "--train-cache", "/x/physicalai-train-e438721ae894/train",
         "--val-cache", "/x/val"])
    cmd = T._staged_command(fs)
    assert "--from-scratch" in cmd and "--trunk" not in cmd

    ws = T.build_parser().parse_args(
        ["--print-launch", "--trunk", "/ckpt/v1.pt",
         "--train-cache", "/x/physicalai-train-e438721ae894/train",
         "--val-cache", "/x/val"])
    cmd2 = T._staged_command(ws)
    assert "--trunk /ckpt/v1.pt" in cmd2 and "--from-scratch" not in cmd2


def test_from_scratch_trunk_is_random_and_passes_not_frozen_gate():
    """⭐ The mission's core invariant: a random-init WorldModel (NO warm-start) has
    EVERY trunk tensor requiring grad and sitting in the AdamW 'trunk' group, so
    _assert_trunk_trainable passes trivially — the not-frozen launch gate is met from
    scratch, and no trunk tensor is frozen."""
    import dataclasses

    import torch

    from tanitad.config import flagship4b_smoke_config
    from tanitad.models.fourbrain import WorldModel

    cfg = flagship4b_smoke_config()
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    world = WorldModel(cfg)                              # RANDOM init — no warm-start
    opt = torch.optim.AdamW(
        [{"params": list(world.parameters()), "lr": 1e-4, "name": "trunk"}],
        weight_decay=0.01)
    report = T._assert_trunk_trainable(world, opt, 1e-4)
    assert report["not_frozen"] is True
    assert report["trunk_tensors_frozen"] == 0
    assert report["trunk_params_requires_grad"] == report["trunk_params_total"]


# ------------------------------------------ the §16 one-lever reproductions -
def test_reproduction_diffs_parse():
    p = T.build_parser()
    # the four attributability diffs of §16 must all be expressible
    assert p.parse_args(["--lambda-plan", "0"]).lambda_plan == "0"
    assert p.parse_args(["--strategic", "off"]).strategic == "off"
    assert p.parse_args(["--long-horizon-k", "0"]).long_horizon_k == 0
    iso = p.parse_args(["--lat-weight", "0", "--lon-weight", "0", "--dist-weight", "0"])
    assert iso.lat_weight == iso.lon_weight == iso.dist_weight == 0.0
    # defaults reproduce the design surface
    d = p.parse_args([])
    assert (d.lambda_plan, d.strategic, d.probe_steps, d.rollout_k) == ("sched", "full", 50, 4)
    assert d.ego_null_row is True                       # P5b default; zero-fill is X15
    # ⭐ v4.2 schedule defaults: lr_trunk 1e-4 (between v4's 3e-4 and v4.1's 3e-5) and
    # the cap-and-hold controller floor 0.25 (the planner is never starved to ~0)
    assert d.lr_trunk == 1e-4
    assert d.lam_mult_floor == 0.25
    # ⭐ same-as-v1 effective batch: micro 16 x accum 4 = 64 (v4.1 ran accum 1 = 16)
    assert d.batch == 16 and d.accum == 4 and d.batch * d.accum == 64


def test_parity_contract_is_pinned():
    assert T.PARITY_KEY == "physicalai-train-e438721ae894"
    assert T.PARITY_SKIP_HASH == "f09e44db"


# ------------------------------------------- the full training LOOP (P4) ----
def test_smoke_loop_proves_loop_checkpoint_controller_archive(tmp_path):
    """The P4 acceptance proof: the real _training_loop on toy episodes across
    phases A/B/C, showing finite loss, a computed canary, the DOWN-ONLY λ_plan
    controller, a milestone archive, and a bit-exact checkpoint save->resume."""
    out = T.smoke_loop(tmp_dir=str(tmp_path))

    # the loop ran across the A/B/C boundaries and finished
    assert out["final_step"] == 5
    # the canary computed on toy data (baseline finite, a trace exists)
    assert out["canary_baseline"] == out["canary_baseline"]        # not NaN
    assert len(out["canary_trace"]) >= 1

    # ⭐ v4.2 CAP-AND-HOLD: the λ_plan controller is DOWN-ONLY under a forced canary
    # regression but HOLDS at the floor (soft breach halves, three hard breaches ->
    # floor), NEVER reaching 0 — so the planner is never starved (the v4.1 bug).
    assert out["controller_down_only"]
    assert out["controller_held_at_floor"]
    assert out["controller_never_zero"]
    assert min(out["mult_trace"]) >= out["mult_floor"] - 1e-9
    assert out["mult_trace"] == sorted(out["mult_trace"], reverse=True)

    # a milestone archive appeared for run_gate.py to score post-hoc
    assert out["milestone_present"]
    assert f"ckpt_step3.pt" in out["milestone_archives"]

    # checkpoint save -> resume is state-consistent: step advances and the
    # controller multiplier is restored BIT-EXACT
    r = out["resume"]
    assert r["step_advances"]
    assert r["mult_bit_exact"]
    assert r["resumed_step"] == r["saved_step"] + 1
