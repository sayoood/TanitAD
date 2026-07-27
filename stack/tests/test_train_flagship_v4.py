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


# ------------------------------------ g_op_fwd_ade_m reaches the log (P8) ----
def test_joint_step_log_carries_g_op_fwd_ade_m():
    """REGRESSION (2026-07-25 v4-gate dry-run): the joint-loss log MUST carry
    ``g_op_fwd_ade_m``.

    ``speed_benefit_recovered_frac`` -- a KILL secondary on
    ``flagship-v4.card.json`` -- reduces exactly this key out of the arm's
    ``train_log.jsonl`` (``tanitad.eval.speed_benefit.GATE_METRIC``). It was
    absent from EVERY v4 train log (MEASURED: flagship-v4.1-10k and
    flagship-v4.2-step4000, 0 occurrences each), so every v4 gate rendered
    INCOMPLETE.

    Root cause: ``grounding_losses`` emits the key ALREADY g-prefixed
    (``metric_dynamics.py:389``), but the joint-step log filtered ``wm_log`` for
    the UNPREFIXED ``op_fwd_ade_m`` and then re-prefixed -- so it matched
    nothing (and would have produced ``g_g_op_fwd_ade_m`` if it had). The
    row-writer had already been patched to forward the key, which made the
    breakage invisible: the fix was inert because this line starved it.

    This test fails on the old code and passes on the fixed line."""
    out = T.smoke()
    for step, log in out["logs"]:
        assert "g_op_fwd_ade_m" in log, (
            f"step {step}: joint-step log lacks g_op_fwd_ade_m -- "
            f"speed_benefit_recovered_frac will read NOT SUPPLIED. "
            f"keys={sorted(log)}")
        assert log["g_op_fwd_ade_m"] == log["g_op_fwd_ade_m"]      # not NaN
        # the double-prefix bug's signature must never come back
        assert "g_g_op_fwd_ade_m" not in log

    # and the key the gate reduces is the one speed_benefit actually asks for
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tanitad.eval import speed_benefit as SB
    assert SB.GATE_METRIC == "g_op_fwd_ade_m"


# ---------------------------- the GROUNDING INSTRUMENT is a PAIR (v5 prep) ----
GROUNDING_PAIR = ("g_op_mid_de_m", "g_op_fwd_ade_m")


def test_joint_step_log_carries_the_GROUNDING_PAIR_not_just_the_imagined_half():
    """v5 prep §1.4 — defect #3 of the last run: all three v4 logs carry NO ``g_*``
    pair, so the real-vs-imagined gap is unmeasurable and v4 is undiagnosable.

    ``g_op_fwd_ade_m`` alone is not the instrument. It is the IMAGINED half; the
    diagnostic quantity is its ratio to ``g_op_mid_de_m``, the REAL-pair half
    (``metric_dynamics.py:386-389``). With one number a rise cannot be attributed
    — encoder drift and predictor drift are indistinguishable. v1 logs both, by
    an unfiltered ``row.update(log)``.
    """
    out = T.smoke()
    for step, log in out["logs"]:
        for k in GROUNDING_PAIR:
            assert k in log, (f"step {step}: the grounding instrument is "
                              f"incomplete — {k} missing. keys={sorted(log)}")
            assert log[k] == log[k]                                  # not NaN
        # the ratio the instrument exists to produce must be computable
        assert log["g_op_fwd_ade_m"] >= 0.0 and log["g_op_mid_de_m"] >= 0.0


def test_the_WRITTEN_ROW_carries_the_pair_and_the_four_selection_diagnostics(
        tmp_path):
    """⭐ The test that would have caught BOTH of the last run's logging defects.

    There are TWO filters between a computed metric and ``train_log.jsonl`` — the
    ``wm_log`` comprehension in ``v4_loss_step`` and the row-writer's key tuple in
    ``_training_loop``. Patching only one leaves the other starved and the fix
    SILENTLY INERT: that is exactly what happened to ``g_op_fwd_ade_m`` (MEASURED
    0 occurrences in flagship-v4.1-10k and v4.2-step4000 *after* the row-writer
    had been 'fixed'). A test on ``v4_loss_step``'s dict alone cannot see it.

    This reads the ACTUAL JSONL the loop wrote.

    It also pins the four selection diagnostics, which were computed **601 times
    per run and discarded** — the exact numbers that would have shown held-out
    selection regressing from ~step 11,000 while every training term improved.
    """
    import json

    out = T.smoke_loop(tmp_dir=str(tmp_path))
    rows = [json.loads(x) for x in
            Path(out["train_log"]).read_text(encoding="utf-8").splitlines() if x]
    step_rows = [r for r in rows if "total" in r]
    assert step_rows, "the loop wrote no per-step rows at all"

    for k in GROUNDING_PAIR:
        n = sum(1 for r in step_rows if k in r)
        assert n == len(step_rows), (
            f"{k} reached the written row in only {n}/{len(step_rows)} rows — "
            f"a starved filter. Both the v4_loss_step comprehension AND the "
            f"row-writer tuple must carry it.")

    for k in ("sel_gap", "rank_acc", "frac_sel_2x_worse_than_oracle"):
        n = sum(1 for r in step_rows if k in r)
        assert n == len(step_rows), (
            f"selection diagnostic {k} reached only {n}/{len(step_rows)} written "
            f"rows; these were computed 601x per run and thrown away.")


# ------------------------------------ the MID-RUN HELD-OUT GATE (v5 prep) -----
def test_preflight_refuses_a_silently_disabled_heldout_gate():
    """Cause #1 made un-repeatable-by-accident: the gate is ON by default and
    turning it off is a visible, preflight-tripping act."""
    assert T.preflight_asserts(_args()) == []                 # ON by default
    a = T.build_parser().parse_args(["--print-launch", "--no-heldout-gate"])
    assert a.heldout_gate is False
    assert any("HELDOUT-GATE" in p and "29.5" in p
               for p in T.preflight_asserts(a))


def test_preflight_catches_a_gate_that_is_present_but_INERT():
    """A gate that can never fire is worse than none — it looks like cover.

    Three ways to build one, each driven here: a cadence longer than the run, a
    run too short to hold incumbent+patience probes, and patience below 2.
    """
    assert any("never fires" in p
               for p in T.preflight_asserts(_args(heldout_every=99999)))
    assert any("consecutive challengers" in p
               for p in T.preflight_asserts(_args(steps=4000,
                                                  heldout_every=2000)))
    assert any("patience must be >= 2" in p
               for p in T.preflight_asserts(_args(heldout_patience=1)))
    assert any("UNPOWERED" in p
               for p in T.preflight_asserts(_args(heldout_episodes=2)))


def test_the_training_LOOP_stops_and_banks_the_best_checkpoint(tmp_path):
    """⭐ The end-to-end proof, on the REAL ``_training_loop``.

    A real :class:`HeldoutGate` is driven with a decaying per-window primary (its
    pseudo-sim stage is replaced so the test needs no renderer), and the loop
    must (a) break early, (b) leave ``ckpt_best.pt`` banked at the incumbent, and
    (c) record the reason in the run's own log.

    The control runs the identical loop with NO gate and shows it runs to the
    end — so the early stop is attributable to the gate and not to the smoke.
    """
    import json

    import numpy as np

    from tanitad.train.heldout_gate import HeldoutGate, HeldoutGateConfig

    eid = [f"ep{e}" for e in range(8) for _ in range(4)]
    series = iter([0.90, 0.55, 0.20, 0.10, 0.05])

    class _Decaying(HeldoutGate):
        def _composite_of(self, pw):                     # no renderer needed
            lvl = next(series)
            rng = np.random.default_rng(0)
            return (lvl + rng.normal(0, 0.01, len(eid)), list(eid),
                    {"grid": None, "traffic_mode": "test", "_estimator": "test"})

        def probe(self, step, world, head, episodes, **kw):
            val, e, node = self._composite_of(None)
            rec = self.observe(step, val, e)
            rec["pseudosim"] = node
            return rec

    g = _Decaying(HeldoutGateConfig(every=1, first_probe_step=1, n_boot=300,
                                    patience=2))
    out = T.smoke_loop(tmp_dir=str(tmp_path), heldout_gate=g)

    assert out["early_stopped"], "the gate did not stop a decaying run"
    assert out["final_step"] < 5, (
        f"the loop ran to step {out['final_step']} despite the gate firing")
    assert out["best_ckpt_present"], "the best checkpoint was not banked"
    assert g.stop_reason and "separated-WORSE" in g.stop_reason

    rows = [json.loads(x) for x in
            Path(out["train_log"]).read_text(encoding="utf-8").splitlines() if x]
    assert any(r.get("EARLY_STOP") for r in rows), (
        "the run's own log does not say why it ended")

    # the control: same loop, no gate -> it runs to the end
    ctl = T.smoke_loop(tmp_dir=str(tmp_path / "ctl"))
    assert not ctl["early_stopped"] and ctl["final_step"] == 5


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
