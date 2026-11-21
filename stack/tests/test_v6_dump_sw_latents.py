"""STEP 1 of the selector-admission recipe — pinned by EXECUTION, not by shape.

⛔ **WHY THIS FILE IS WRITTEN THE WAY IT IS.** The defect that killed SEL-1's
reopening path (C94, `E4_SELECTOR_RESOLUTION.md` §3) was **a fixture that
modelled the CONSUMER'S EXPECTATION instead of the PRODUCER'S OUTPUT**:
`test_v6_chain.py` hand-wrote `{"sigma_2s_m": …}` — the shape the reader wanted
— so the join between instrument and reader was never exercised, and a green
suite certified a connection that did not exist. Name AND nesting level both
differed.

⇒ **Nothing here hand-writes a dump.** Every assertion runs the REAL producer
(`v6_dump_sw_latents.collect_latents`) over a REAL `V6Stack` and a REAL
`FlagshipWindowDataset`, and the headline test hands that output to the REAL
estimator (`e_wc2_sigma_star.run`) and the REAL chain reader
(`v6_chain.read_sw_admission` → `assert_selector_admissible`) and asserts a real
verdict comes out the far end.

⭐ **THE PLANTED-σ ROUND TRIP.** The corpus is built in two passes so σ can be
planted against the encoder's OWN latents without faking either half:

  1. run the producer once → the REAL `pooled`/`ctx` for every window;
  2. rewrite the episodes' POSES so the 2 s ego-frame endpoint is exactly
     ``pooled @ W + N(0, σ)`` per axis;
  3. run the producer AGAIN — unmodified — and push its output through the whole
     chain.

The rewrite touches ONLY pose indices ``last + 20`` (≡ 3 mod 8), while the
reference poses ``last`` (≡ 7) and every other fan waypoint (≡ 4, 1, 6) are
untouched, so nothing but the planted quantity moves. The frames never change,
which is why step 3's latents must come back BIT-IDENTICAL to step 1's — and
that is asserted, because it is what makes the planting non-circular.

NO GPU, NO CORPUS, NO CHECKPOINT — synthetic episodes and a ~1000x-smaller stack
with the identical wiring and the identical seams.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e_wc2_sigma_star as E  # noqa: E402
import v6_chain as C  # noqa: E402
import v6_dump_sw_latents as D  # noqa: E402
from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.data.toy_driving import ToyEpisode  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402
from train_flagship4b import FlagshipWindowDataset  # noqa: E402

#: the pre-registered surface, so the guards are really exercised
N_EP, WIN_PER_EP = 40, 22
T_FRAMES = 8 * WIN_PER_EP + D.WINDOW + D.K_MAX_GRID          # -> WIN_PER_EP windows
CHANNELS = 3


# ============================================================================
# fixtures — a tiny stack and a tiny corpus, both REAL objects
# ============================================================================
def tiny_cfg(**kw) -> V6Config:
    """`tests/test_v6_staged.py::tiny_cfg`, with a narrower readout.

    ``d_readout=4`` puts ``d_op`` at 64 so ``pooled,ctx`` is 80 columns against
    881 rows — an OOF ridge at n/d ~ 11 recovers a planted σ to a few percent,
    where the 128-wide default would inflate it by its own overfit. The wiring,
    the seams and ``plan_steps`` are untouched.
    """
    base = dict(
        encoder=EncoderConfig(in_channels=CHANNELS, image_size=32,
                              image_width=32, patch_size=16, d_model=32,
                              depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=4),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1, 2), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8)
    base.update(kw)
    return V6Config(**base)


def _poses(T: int, *, eid: int) -> torch.Tensor:
    """A smooth, non-degenerate trajectory: turning, and never stationary.

    ⚠️ Non-degeneracy matters — on a parked ego every ±1-row shift also matches,
    which is exactly the vacuous-control failure `refc_dump_latents.
    endpoint_agreement` was hardened against.
    """
    g = torch.Generator().manual_seed(900 + eid)
    yaw_rate = 0.04 * math.sin(0.7 * eid) + 0.01
    x, y, yaw, v, dt = 0.0, 0.0, 0.2 * eid, 6.0 + 0.3 * eid, 0.1
    rows = []
    for t in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(1.0, v + 0.4 * float(torch.rand(1, generator=g)) - 0.2)
    return torch.tensor(rows, dtype=torch.float32)


def _episode(eid: int, T: int = T_FRAMES) -> ToyEpisode:
    g = torch.Generator().manual_seed(100 + eid)
    frames = torch.randint(0, 255, (T, CHANNELS, 32, 32), generator=g,
                           dtype=torch.uint8)
    actions = torch.randn(T, 2, generator=g) * 0.2
    return ToyEpisode(frames=frames, actions=actions, poses=_poses(T, eid=eid),
                      episode_id=eid)


def _dataset(eps) -> FlagshipWindowDataset:
    """The CANONICAL grid: window 8, max_horizon 20 (= K_MAX_GRID)."""
    return FlagshipWindowDataset(eps, window=D.WINDOW,
                                 max_horizon=D.K_MAX_GRID, maneuver_h=20,
                                 channels=CHANNELS)


@pytest.fixture(scope="module")
def stack_no_scorer() -> V6Stack:
    torch.manual_seed(0)
    return V6Stack(tiny_cfg()).eval()


@pytest.fixture(scope="module")
def stack_with_scorer() -> V6Stack:
    torch.manual_seed(1)
    return V6Stack(tiny_cfg(selector="goal")).eval()


@pytest.fixture(scope="module")
def corpus():
    """40 episodes on the canonical grid -> 881 windows, the §5.2 surface."""
    eps = [_episode(e) for e in range(N_EP)]
    eps[0] = _episode(0, T=T_FRAMES + 8)                 # one extra window -> 881
    ds = _dataset(eps)
    grid = D.select_grid(ds, episodes=N_EP)
    return ds, grid


@pytest.fixture(scope="module")
def dump_no_scorer(stack_no_scorer, corpus):
    ds, grid = corpus
    return D.collect_latents(stack_no_scorer, ds, grid, batch=16)


# ============================================================================
# 1. the grid — canonical, and the SAME one the REF-C producer builds
# ============================================================================
def test_the_grid_is_the_refc_producers_grid_element_for_element(corpus):
    ds, grid = corpus
    assert (D.WINDOW, D.STRIDE, D.K_MAX_GRID) == (8, 8, 20)
    # select_grid ASSERTS this internally per episode; re-stated here so the
    # pin is visible rather than buried in the producer.
    for e in range(N_EP):
        want = D.window_starts(int(ds.episodes[e].frames.shape[0]))
        got = [ds.index[i][1] for i in grid if ds.index[i][0] == e]
        assert got == want
    assert len(grid) == D.N_WINDOWS == 881
    assert len({ds.index[i][0] for i in grid}) == D.N_EPISODES == 40


def test_a_model_window_wider_than_the_grid_is_REFUSED(corpus):
    """Widening the grid to fit a model would RE-SELECT WINDOWS. Parity is sacred."""
    ds, grid = corpus
    torch.manual_seed(2)
    wide = V6Stack(tiny_cfg(predictor=PredictorConfig(
        d_model=32, depth=1, n_heads=2, window=12, horizons=(1, 2),
        action_dim=3))).eval()
    with pytest.raises(SystemExit, match="exceeds the canonical"):
        D.collect_latents(wide, ds, grid[:8], batch=4)


# ============================================================================
# 2. the producer's REAL output against the consumer's REAL validator
# ============================================================================
def test_the_producers_own_output_satisfies_the_e_wc2_contract(
        stack_with_scorer, corpus):
    """⭐ The join, at the contract level: a dump this producer really emitted —
    on the FULL pre-registered surface, with a scorer, so every optional key is
    really there — handed to `e_wc2_sigma_star.validate_dump`, must report NO
    problems at all."""
    ds, grid = corpus
    d = D.collect_latents(stack_with_scorer, ds, grid, batch=32)
    assert d["instrument_fail"] == []
    assert E.validate_dump(d) == []
    assert "sel" in d and d["sel"].shape == (881,)
    assert d["controls"]["has_scorer"] is True
    # the shapes the contract names, read off the real tensors
    n = len(d["eid"])
    assert d["pooled"].shape == (n, stack_with_scorer.cfg.d_op)
    assert d["pooled_seq"].shape[:2] == (n, stack_with_scorer.cfg.predictor.window)
    assert d["ctx"].shape == (n, stack_with_scorer.cfg.d_str)
    assert d["z_tac"].shape == (n, stack_with_scorer.cfg.d_tac)
    assert d["gt_endpoint"].shape == (n, 2, 2)
    assert d["endpoint_valid"].shape == (n, 2)
    assert d["endpoint_steps"] == [20, 60]
    assert d["fan"].shape == (n, stack_with_scorer.cfg.n_candidates, 4, 2)
    assert d["gt"].shape == (n, 4, 2)
    assert d["wp_steps"] == [5, 10, 15, 20]


def test_every_recognised_feature_block_is_declared_vision_only(dump_no_scorer):
    """The blocks E-WC2 knows by name must all be VISION_ONLY here — v6 has no
    `measurement` (ego+nav) analogue, and emitting one would be inventing the
    labelled-inadmissible control out of nothing."""
    named = [k for k in E.FEATURE_ADMISSIBILITY if k in dump_no_scorer]
    assert set(named) == {"pooled", "pooled_seq", "ctx", "v0"}
    assert "measurement" not in dump_no_scorer
    for b in ("pooled", "pooled_seq", "ctx"):
        assert E.FEATURE_ADMISSIBILITY[b] == "VISION_ONLY"
    assert E.FEATURE_ADMISSIBILITY["v0"] == "MEASURED_PRESENT"


def test_the_default_recipe_features_build_a_design_matrix(dump_no_scorer):
    """`--features pooled,ctx` is what step 2 of the emitted recipe runs."""
    X, meta = E.build_features(dump_no_scorer, ["pooled", "ctx"],
                               allow_echo=False, declared={})
    assert X.shape == (881, dump_no_scorer["pooled"].shape[1]
                       + dump_no_scorer["ctx"].shape[1])
    assert meta["any_echo"] is False and meta["any_measured_present"] is False
    assert [b["admissibility"] for b in meta["blocks"]] == ["VISION_ONLY"] * 2


# ============================================================================
# 3. VISION-ONLY is MEASURED, and the control can FAIL
# ============================================================================
def test_vision_only_invariance_is_measured_and_not_vacuous(dump_no_scorer):
    ctl = dump_no_scorer["controls"]["vision_only_invariance"]
    assert ctl["vacuous"] is False and ctl["permutation_changed_inputs"] is True
    assert ctl["ok"] is True
    assert set(ctl["blocks"]) == set(D.VISION_ONLY_BLOCKS)
    assert all(ctl["blocks"].values())
    assert dump_no_scorer["instrument_fail"] == []


def test_the_vision_only_control_REPORTS_A_FAILURE_when_a_block_reads_v0():
    """⛔ A control that cannot fail is decoration. Same discipline as E4's
    trilemma: prove each verdict on a constructed input."""
    class _LeakyStack:
        """A stand-in whose `z_op` DOES read v0 — the leak the control exists for."""
        def __call__(self, *, frames, actions, v0):
            b = v0.shape[0]
            z = torch.ones(b, 4) * v0[:, None]
            return {"z_op": z, "z_op_win": z[:, None].expand(b, 2, 4).clone(),
                    "z_str": z[:, :2], "z_tac": z[:, :3],
                    "plan": {"waypoints": torch.zeros(b, 2, 60, 2)}}

    leaky = _LeakyStack()
    frames = torch.zeros(4, 2, CHANNELS, 32, 32)
    acts2 = torch.randn(4, 2, 2)
    v0 = torch.tensor([3.0, 7.0, 11.0, 15.0])
    base = D._forward_latents(leaky, frames, acts2, v0)
    rec = D.vision_only_control(leaky, frames, acts2, v0, base)
    assert rec["ok"] is False and rec["vacuous"] is False
    assert not any(rec["blocks"].values())
    assert "max_abs_diff" in rec


def test_the_vision_only_control_refuses_to_pass_vacuously():
    """b < 2, or a permutation that moves nothing, carries NO evidence."""
    class _Stub:
        def __call__(self, *, frames, actions, v0):
            b = v0.shape[0]
            z = torch.zeros(b, 4)
            return {"z_op": z, "z_op_win": z[:, None].expand(b, 2, 4).clone(),
                    "z_str": z[:, :2], "z_tac": z[:, :3],
                    "plan": {"waypoints": torch.zeros(b, 2, 60, 2)}}

    s = _Stub()
    f, a1 = torch.zeros(1, 2, CHANNELS, 32, 32), torch.zeros(1, 2, 2)
    one = D.vision_only_control(s, f, a1, torch.tensor([5.0]),
                                D._forward_latents(s, f, a1, torch.tensor([5.0])))
    assert one["vacuous"] is True and one["ok"] is False
    # a batch whose v0 AND actions are identical everywhere: no evidence either
    f2, a2, v2 = (torch.zeros(4, 2, CHANNELS, 32, 32), torch.zeros(4, 2, 2),
                  torch.full((4,), 5.0))
    flat = D.vision_only_control(s, f2, a2, v2,
                                 D._forward_latents(s, f2, a2, v2))
    assert flat["vacuous"] is True and flat["ok"] is False


# ============================================================================
# 4. the alignment controls, and that THEY can fail too
# ============================================================================
def test_row_alignment_and_ego_frame_controls_pass_on_a_real_dump(dump_no_scorer):
    ctl = dump_no_scorer["controls"]
    assert ctl["v0_batch_matches_poses"] is True
    assert ctl["endpoint_20_matches_gt"] is True
    assert ctl["reference_horizon_s"] == 2.0
    assert ctl["fan_wp_index_map"] == {"5": 4, "10": 9, "15": 14, "20": 19}
    # the 6 s horizon runs off the end of every episode's tail -> masked, never
    # imputed, and the 2 s horizon is valid everywhere on this grid
    assert ctl["endpoint_valid_frac"]["20"] == 1.0
    assert 0.0 < ctl["endpoint_valid_frac"]["60"] < 1.0


def test_the_row_alignment_control_CATCHES_a_one_window_shift(dump_no_scorer):
    """⛔ A grid off by one regresses every latent onto a NEIGHBOUR's endpoint —
    σ comes back inflated, i.e. a wrong answer that looks like a measurement."""
    d = dict(dump_no_scorer)
    tgt = {"v0_from_poses": torch.roll(d["v0"].clone(), 1)}
    ctl = D.producer_controls(d, tgt)
    assert ctl["v0_batch_matches_poses"] is False
    assert any("NOT the same windows" in f for f in ctl["fails"])


def test_a_short_surface_is_flagged_by_the_producer_not_three_steps_later(
        stack_no_scorer, corpus):
    ds, grid = corpus
    d = D.collect_latents(stack_no_scorer, ds, grid[:40], batch=16)
    assert any("881/40" in f for f in d["instrument_fail"])


def test_the_endpoint_frame_control_CATCHES_a_wrong_ego_frame(dump_no_scorer):
    d = dict(dump_no_scorer)
    d["gt_endpoint"] = d["gt_endpoint"].clone()
    d["gt_endpoint"][:, 0] += 0.5                    # a 0.5 m frame offset
    ctl = D.producer_controls(d, {"v0_from_poses": d["v0"]})
    assert ctl["endpoint_20_matches_gt"] is False
    assert any("not on the fan's rows/frame" in f for f in ctl["fails"])


# ============================================================================
# 5. `sel` is emitted only when it is REAL
# ============================================================================
def test_a_no_scorer_arm_omits_sel_and_records_why(dump_no_scorer):
    """⛔ Fabricating `sel` would manufacture the σ/ADE denominator and with it a
    §5.2 verdict — this session's own root-cause class in a new costume."""
    assert "sel" not in dump_no_scorer
    why = dump_no_scorer["sel_absent_reason"]
    assert "NOT fabricated" in why and "ORACLE" in why
    probs = E.validate_dump(dump_no_scorer)
    assert len(probs) == 1 and "fan`/`gt`/`sel`" in probs[0]
    # ...and E-WC2 therefore refuses its OWN verdict, which is correct
    res = E.run(dump_no_scorer, features=["pooled", "ctx"], n_boot=0)
    assert res["decision"]["verdict"] == "NO_VERDICT"
    assert res["references_and_ratios"].get("sigma_over_ade") is None


def test_a_scorer_arm_records_the_incumbent_argmax_rule(stack_with_scorer, corpus):
    ds, grid = corpus
    d = D.collect_latents(stack_with_scorer, ds, grid[:64], batch=16)
    assert d["sel"].dtype == torch.long
    assert int(d["sel"].max()) < stack_with_scorer.cfg.n_candidates
    assert "sel_absent_reason" not in d


# ============================================================================
# 6. ⭐ THE PLANTED-σ ROUND TRIP — producer -> estimator -> chain reader
# ============================================================================
def _plant_sigma(stack, ds, grid, sigma: float, *, seed: int = 0) -> dict:
    """Rewrite the corpus' 2 s endpoints to ``pooled @ W + N(0, σ)``, then RUN
    THE REAL PRODUCER over it again and return its output.

    Only pose indices ``last + 20`` move; ``last`` (the ego-frame reference),
    ``last - 1`` (the CV baseline) and the other fan waypoints are untouched,
    and only the x/y columns are written so ``v0`` cannot move either.
    """
    probe = D.collect_latents(stack, ds, grid, batch=32, run_control=False)
    pooled = probe["pooled"].double().numpy()
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(pooled.shape[1], 2)) / math.sqrt(pooled.shape[1])
    target = pooled @ w * 20.0 + rng.normal(0.0, sigma, size=(len(grid), 2))

    for r, i in enumerate(grid):
        e_i, t = ds.index[i]
        poses = ds.episodes[e_i].poses
        last = t + D.WINDOW - 1
        yaw = float(poses[last, 2])
        c, s = math.cos(yaw), math.sin(yaw)
        ex, ey = float(target[r, 0]), float(target[r, 1])
        poses[last + 20, 0] = poses[last, 0] + ex * c - ey * s
        poses[last + 20, 1] = poses[last, 1] + ex * s + ey * c
    out = D.collect_latents(stack, ds, grid, batch=32)
    # ⛔ THE PLANTING MUST NOT HAVE REACHED THE MODEL SIDE. The frames never
    # changed, so the producer's latents must come back BIT-IDENTICAL — this is
    # what makes the plant a plant rather than a circular fit.
    assert torch.equal(out["pooled"], probe["pooled"])
    assert torch.equal(out["ctx"], probe["ctx"])
    return out


@pytest.mark.parametrize(("sigma", "verdict"),
                         [(0.30, "FUNDED"), (1.10, "INCONCLUSIVE"),
                          (2.00, "REFUSED")])
def test_PLANTED_SIGMA_travels_producer_to_estimator_to_the_chain_verdict(
        stack_no_scorer, corpus, tmp_path, sigma, verdict):
    """⭐ THE TEST THIS FILE EXISTS FOR — and the one C94 did not have.

    A σ planted in the corpus must come back out of `v6_chain.read_sw_admission`
    as the PRE-REGISTERED verdict, having passed through the real producer, the
    real estimator's real output nesting, and the real resolver. A hand-written
    dump would prove only that this file agrees with itself.
    """
    ds, grid = corpus
    d = _plant_sigma(stack_no_scorer, ds, grid, sigma, seed=int(sigma * 100))

    # --- the REAL estimator, on the REAL dump, at the recipe's own features ---
    res = E.run(d, features=["pooled", "ctx"], n_boot=0)
    got = res["references_and_ratios"]["sigma_perax_2s_m"]
    assert got == pytest.approx(sigma, rel=0.25), (
        f"planted {sigma}, recovered {got}")

    # --- the REAL chain artifact, at the path the chain reads ----------------
    root = tmp_path / "experiments"
    cfg = C.ChainConfig(root=str(root).replace("\\", "/"))
    sw = Path(cfg.path(cfg.sw_dir))
    sw.mkdir(parents=True, exist_ok=True)
    (sw / C.SW_LATENT_ADMISSION["artifact"]).write_text(
        json.dumps(res, indent=1, default=float), encoding="utf-8")

    adm = C.read_sw_admission(cfg)
    assert adm["present"] is True
    assert adm["read_at"] == "references_and_ratios.sigma_perax_2s_m"
    assert adm["verdict"] == verdict, adm
    assert adm["sigma_2s_m"] == pytest.approx(got, abs=1e-9)

    # --- and what the chain then DOES with it --------------------------------
    step = C.step_by_key(C.build_plan(C.ChainConfig(root=cfg.root,
                                                    st_arms=("goal",))),
                         "S-T:goal")
    if verdict == "FUNDED":
        assert C.assert_selector_admissible(step, cfg)["ok"] is True
    else:
        with pytest.raises(C.ChainRefusal):
            C.assert_selector_admissible(step, cfg)


def test_the_admission_gate_reads_sigma_even_though_e_wc2_refuses_its_verdict(
        stack_no_scorer, corpus, tmp_path):
    """⭐ A COUPLING WORTH STATING, MEASURED. On a no-scorer S-W arm σ/ADE is not
    computable, so E-WC2 returns NO_VERDICT — and the S-W admission gate is
    defined on ABSOLUTE metres and reads the σ anyway. Both halves at once, or
    an operator concludes the dump failed."""
    ds, grid = corpus
    d = _plant_sigma(stack_no_scorer, ds, grid, 0.30, seed=7)
    assert "sel" not in d
    res = E.run(d, features=["pooled", "ctx"], n_boot=0)
    assert res["decision"]["verdict"] == "NO_VERDICT"
    assert "sigma_perax_2s_m" in res["references_and_ratios"]

    root = tmp_path / "experiments"
    cfg = C.ChainConfig(root=str(root).replace("\\", "/"))
    sw = Path(cfg.path(cfg.sw_dir))
    sw.mkdir(parents=True, exist_ok=True)
    (sw / C.SW_LATENT_ADMISSION["artifact"]).write_text(
        json.dumps(res, indent=1, default=float), encoding="utf-8")
    assert C.read_sw_admission(cfg)["verdict"] == "FUNDED"


def test_a_dump_with_no_fan_leaves_the_admission_gate_with_nothing_to_read(
        dump_no_scorer, tmp_path):
    """⛔ THE COUPLING THE CONTRACT DOES NOT SPELL OUT. `sigma_perax_2s_m` is
    written only inside `if refs.available` — so a LATENT-ONLY dump produces an
    artifact with no σ at all and the gate stays dead. This is why the fan is
    not optional in this producer."""
    d = {k: v for k, v in dump_no_scorer.items() if k not in ("fan", "gt")}
    res = E.run(d, features=["pooled", "ctx"], n_boot=0)
    assert res["references_and_ratios"]["available"] is False
    assert "sigma_perax_2s_m" not in res["references_and_ratios"]
    root = tmp_path / "experiments"
    cfg = C.ChainConfig(root=str(root).replace("\\", "/"))
    sw = Path(cfg.path(cfg.sw_dir))
    sw.mkdir(parents=True, exist_ok=True)
    (sw / C.SW_LATENT_ADMISSION["artifact"]).write_text(
        json.dumps(res, indent=1, default=float), encoding="utf-8")
    adm = C.read_sw_admission(cfg)
    assert adm["verdict"] is None
    assert "Absence of the number is NOT an admission" in adm["resolve"]["reason"]


# ============================================================================
# 7. preflight — fail in 2 seconds, not 11 minutes in
# ============================================================================
def test_preflight_probes_STEP_TWOs_imports_too():
    """MEASURED 2026-08-11: `t1_eval.py` rolled both arms over all 40 episodes
    and died in `analyze()` on a missing optional module. The expensive part had
    already been paid for."""
    names = [m for m, _ in D.PREFLIGHT_MODULES]
    assert "e_wc2_sigma_star" in names          # step 2's estimator
    assert "tanitad.eval.v6_probe_trunk" in names
    assert "refc_dump_latents" in names
    assert D.preflight(verbose=False) == []


def test_print_contract_mode_needs_no_checkpoint(capsys):
    assert D.main(["--print-contract"]) == 0
    assert "gt_endpoint" in capsys.readouterr().out


def test_preflight_only_mode_runs_before_anything_expensive(capsys):
    """No --ckpt, no --out, no corpus, no CUDA — and it still answers."""
    assert D.main(["--preflight-only"]) == 0
    assert "preflight OK" in capsys.readouterr().out


# ============================================================================
# 7b. the checkpoint seam — the architecture is READ from the run, never typed
# ============================================================================
def _tiny_run_args(tmp_path) -> dict:
    from train_v6_staged import build_parser
    argv = ["--stage", "S-W", "--out", str(tmp_path), "--dry-run",
            "--in-channels", str(CHANNELS), "--frame-h", "32", "--frame-w", "32",
            "--patch", "16", "--enc-dim", "32", "--enc-depth", "1",
            "--enc-heads", "2", "--readout-grid", "4", "--readout-dim", "4",
            "--pred-dim", "32", "--pred-depth", "1", "--pred-heads", "2",
            "--window", "4", "--horizons", "1", "2", "--d-tac", "32",
            "--d-str", "16", "--d-goal-embed", "16", "--adapter-hidden", "32",
            "--n-candidates", "3", "--sigreg-slices", "8"]
    return vars(build_parser().parse_args(argv))


def test_the_stack_is_rebuilt_from_the_checkpoints_OWN_run_args(tmp_path):
    """⛔ The geometry is never typed on this script's command line. It is
    replayed through `build_stack_from_args` and loaded STRICT — a non-strict
    load leaves probe tensors at random init and emits numbers that LOOK like
    results (`v6_probe_trunk`'s own rule)."""
    from train_v6_staged import build_stack_from_args
    args = _tiny_run_args(tmp_path)
    torch.manual_seed(3)
    src = build_stack_from_args(D.Namespace(**args))
    p = tmp_path / "ckpt.pt"
    torch.save({"stack": src.state_dict(), "config": {"args": args},
                "step": 10_000}, p)

    stack, run_args, step, prov = D.load_v6_stack(str(p), device="cpu")
    assert step == 10_000
    assert prov["n_params"] == sum(x.numel() for x in src.parameters())
    assert prov["n_state_keys"] == len(src.state_dict())
    assert prov["has_scorer"] is False and prov["model_window"] == 4
    assert run_args["readout_dim"] == 4
    for (k, a), (_k, b) in zip(stack.state_dict().items(),
                               src.state_dict().items()):
        assert torch.equal(a, b), k
    assert not any(x.requires_grad for x in stack.parameters())


def test_a_weights_only_snapshot_is_readable_with_its_run_record(tmp_path):
    """Thor's `~/ckpt_snaps/v6F_sw_step*.fp16.pt` are weights-only. `--args-from`
    supplies the RUN's record — it is not a place to type a geometry."""
    from train_v6_staged import build_stack_from_args
    args = _tiny_run_args(tmp_path)
    torch.manual_seed(4)
    src = build_stack_from_args(D.Namespace(**args))
    snap = tmp_path / "v6F_sw_step009250.fp16.pt"
    torch.save({k: v.half() for k, v in src.state_dict().items()}, snap)
    rundir = tmp_path / "run"
    rundir.mkdir()
    (rundir / "config.json").write_text(json.dumps({"args": args}),
                                        encoding="utf-8")

    with pytest.raises(SystemExit, match="no run config"):
        D.load_v6_stack(str(snap), device="cpu")
    stack, _ra, _st, prov = D.load_v6_stack(str(snap), device="cpu",
                                            args_from=str(rundir))
    assert prov["normalised_to_stack_layout"] is True
    assert prov["was_v6_layout"] is False
    assert prov["n_state_keys"] == len(src.state_dict())
    # fp16 on disk, fp32 in the rebuilt stack, values equal to half precision
    sd = stack.state_dict()
    for k, v in src.state_dict().items():
        assert sd[k].dtype == v.dtype
        assert torch.allclose(sd[k], v, atol=1e-2, rtol=1e-2), k


# ============================================================================
# 8. the chain's emitted recipe — step 1 is a COMMAND now, not a build item
# ============================================================================
def test_the_recipe_step_1_is_now_a_runnable_command(tmp_path):
    c = C.ChainConfig(root=str(tmp_path).replace("\\", "/"))
    steps = {s["n"]: s for s in C.sw_admission_recipe(c)["steps"]}
    assert len(steps) == 4
    one = steps[1]
    assert one["status"].startswith("✅")
    assert "v6_dump_sw_latents.py" in one["cmd"]
    assert c.val_cache in one["cmd"]                 # the val corpus, not train
    assert "ewc2_sw_dump.pt" in one["cmd"]
    # step 2 consumes exactly what step 1 writes
    assert one["output"] in steps[2]["cmd"]
    for n in (2, 3, 4):
        assert steps[n]["status"].startswith("✅")


def test_the_producer_is_named_in_the_dump_contract():
    txt = json.dumps(E.DUMP_CONTRACT)
    assert "v6_dump_sw_latents.py" in txt            # the v6 producer
    assert "refc_dump_latents.py --endpoint-steps 20,60" in txt   # and REF-C's
