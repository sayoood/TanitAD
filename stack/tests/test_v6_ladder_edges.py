"""The v6 ladder's TRANSITIONS, executed — not read.

Every ladder-edge defect this programme has found was found by RUNNING the
transition (S-S→S-T's missing backward gate, S-W→S-T's `strict=True` refusal of
the designed selector introduction, the selector arm's missing capacity
control). Reading the code found none of them. This file executes the remaining
edges on a tiny CPU stack and pins each outcome — the clean ones so they STAY
clean, the broken ones so they cannot come back.

WHAT WAS MEASURED 2026-08-16, edge by edge:

  * **S-T --init-from-> S-S**  CLEAN. Key-for-key, `missing=[] unexpected=[]
    introduced=[]`. `STAGE_MAY_INTRODUCE["S-S"] == ()` is CORRECT and now
    demonstrated rather than asserted: S-T's checkpoint already carries
    `layer_str` (every stage saves the WHOLE stack), so S-S introduces nothing.
  * **S-S --init-from-> S-J**  CLEAN, same.
  * ⛔ **the `--resume auto` STAGE BOUNDARY**  DEFECTIVE. `load_resume` did a
    strict state-dict load and adopted `ck["step"]` with NO stage check. The
    load SUCCEEDS across the ladder. The only barrier was INCIDENTAL — the
    optimiser's param-group size — and it
      (a) names nothing an operator can act on ("loaded state dict contains a
          parameter group that doesn't match the size of optimizer's group"),
      (b) holds only because the stages happen to train different numbers of
          tensors (S-W 240 · S-T 80 · S-S 54 · S-J 374, MEASURED), and
      (c) is SKIPPED ENTIRELY for a checkpoint with no `opt` key — the exact
          shape of `ops/ckpt_fp16_snapshot.py`, the pod-handover artifact.
  * ⛔ **`--init-from` + `--resume auto` together**  SILENT PROVENANCE LIE.
    config.json recorded `init.trunk_md5_after_load = fbce009a…` while the
    trunk in the model was `326034884…`.
  * ⛔ **`--init-from <fp16 snapshot>`**  REFUSED, blaming the wrong thing. The
    snapshot's state lives under `"model"`; `load_stage_init` looked only for
    `"stack"`, fell through to the wrapper, and reported a 400-key GEOMETRY
    MISMATCH — pointing the operator at the architecture.
  * **freeze x init**  CLEAN, and pinned NON-VACUOUSLY (see the control below).
  * **X3 isolation after `--init-from`**  CLEAN at every stage.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import (  # noqa: E402
    STAGES, V6Config, V6Stack, apply_stage_freeze, stage_trainable_groups)
from train_v6_staged import (  # noqa: E402
    RESUME_CONTRACT, STAGE_MAY_INTRODUCE, ResumeLineageError, V6LossWeights,
    _save_ckpt, assert_resume_lineage, load_resume, load_stage_init,
    read_ckpt_provenance, resume_guard, supersede_init_on_resume,
    synthetic_train_batch, v6_loss_step)


# ============================================================================
# a tiny stack — same wiring, same seams, ~1000x fewer parameters
# ============================================================================
def tiny_cfg(**kw) -> V6Config:
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1, 2), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8)
    base.update(kw)
    return V6Config(**base)


def mk(selector: str = "goal", seed: int = 0) -> V6Stack:
    torch.manual_seed(seed)
    return V6Stack(tiny_cfg(selector=selector))


def _opt_for(stack: V6Stack, stage: str):
    """The optimiser the trainer actually builds: over the TRAINABLE set only
    (train_v6_staged.py — `trainable = [p for p in stack.parameters() if
    p.requires_grad]`), with real moments so the state is not empty."""
    apply_stage_freeze(stack, stage)
    tr = [p for p in stack.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tr, lr=1e-4)
    sum((p * p).sum() for p in tr).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    for p in stack.parameters():
        p.grad = None
    return opt, tr


def write_ckpt(path, stack, stage: str, step: int, *, opt_stage=None):
    """A REAL checkpoint, written by the trainer's own `_save_ckpt`.

    ``opt_stage`` lets a test build the optimiser for a DIFFERENT stage than
    the one the config claims — that is how the "the counts happened to
    collide" case is constructed on purpose.
    """
    opt, _ = _opt_for(stack, opt_stage or stage)
    apply_stage_freeze(stack, stage)
    _save_ckpt(Path(path), stack=stack, opt=opt, step=step,
               cfg_json={"stage": stage, "run": f"v6-staged-{stage}"})
    return Path(path)


@pytest.fixture(scope="module")
def st_ckpt(tmp_path_factory):
    """A real S-T checkpoint: the selector IS built, because S-T is where it is
    built. This is the thing S-S is supposed to stand on."""
    d = tmp_path_factory.mktemp("st")
    return write_ckpt(d / "ckpt.pt", mk("goal", seed=1), "S-T", 30000)


@pytest.fixture(scope="module")
def ss_ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("ss")
    return write_ckpt(d / "ckpt.pt", mk("goal", seed=2), "S-S", 8000)


# ============================================================================
# EDGE 1 & 2 — the forward `--init-from` transitions nobody had executed
# ============================================================================

@pytest.mark.parametrize("prev,stage", [("S-T", "S-S"), ("S-S", "S-J")])
def test_the_upper_ladder_edges_execute_clean(request, prev, stage):
    """S-T -> S-S and S-S -> S-J, RUN rather than reasoned about."""
    ck = request.getfixturevalue("st_ckpt" if prev == "S-T" else "ss_ckpt")
    rep = load_stage_init(mk("goal", seed=50), ck, stage=stage)
    assert rep["missing_keys"] == [], "the successor is missing trunk weights"
    assert rep["unexpected_keys"] == [], "the ckpt carries keys this stack lacks"
    assert rep["introduced_keys"] == [], "nothing should be introduced here"
    assert rep["prev_stage"] == prev
    assert rep["init_source"] == "ckpt" and rep["init_precision"] == "fp32"


@pytest.mark.parametrize("stage", ["S-S", "S-J"])
def test_the_empty_allowance_is_CORRECT_not_merely_declared(st_ckpt, ss_ckpt,
                                                            stage):
    """`STAGE_MAY_INTRODUCE[stage] == ()` is only right if the predecessor
    genuinely carries every module the successor needs. Every stage saves the
    WHOLE stack, so it does — and the test that shows it is the load above
    reporting `introduced_keys == []` while the ALLOWANCE is empty. Declaring
    an empty allowance over a transition that actually needs one is precisely
    the S-W -> S-T defect, one rung up."""
    assert STAGE_MAY_INTRODUCE[stage] == ()
    ck = st_ckpt if stage == "S-S" else ss_ckpt
    rep = load_stage_init(mk("goal", seed=51), ck, stage=stage)
    assert rep["introduced_allowance"] == []
    assert rep["introduced_keys"] == []


def test_a_drifted_selector_flag_is_still_refused_on_the_upper_edges(st_ckpt):
    """Launching S-S with `--selector none` against an S-T ckpt that HAS one is
    a mis-specified arm, not an introduction — and S-S may introduce nothing."""
    with pytest.raises(SystemExit, match="not a valid predecessor"):
        load_stage_init(mk("none", seed=52), st_ckpt, stage="S-S")


# ============================================================================
# EDGE 3 — the stage-boundary `--resume auto`, the one that bit
# ============================================================================

def test_resume_across_the_stage_boundary_is_REFUSED_by_stage(tmp_path):
    """⛔ THE DEFECT. An S-T checkpoint in the directory an S-S run resumes.

    Before the fix this reached `stack.load_state_dict(strict=True)` — which
    SUCCEEDS, because every stage saves the whole stack — and then died in the
    optimiser with a message about param-group sizes.
    """
    ck = write_ckpt(tmp_path / "ckpt.pt", mk("goal", seed=3), "S-T", 30000)
    assert resume_guard(tmp_path, resume="auto",
                        force_rerun=False)["mode"] == "resume"
    with pytest.raises(ResumeLineageError) as e:
        assert_resume_lineage(ck, stage="S-S")
    msg = str(e.value)
    assert "'S-T'" in msg and "'S-S'" in msg, "the message must name BOTH"
    assert "30000" in msg, "and the step it would have silently adopted"
    assert "--init-from" in msg, "and what to do instead"


def test_the_refusal_does_NOT_depend_on_the_optimiser_shape(tmp_path):
    """⭐ THE POINT OF THE FIX. The old barrier was incidental: it held only
    because the stages train different numbers of tensors. Here the checkpoint
    is labelled S-T but its optimiser was built over S-S's trainable set, so
    the param-group sizes MATCH and the accidental guard is defeated — the
    silent wrong-stage resume the old code would have performed."""
    ck = write_ckpt(tmp_path / "ckpt.pt", mk("goal", seed=4), "S-T", 21000,
                    opt_stage="S-S")
    victim = mk("goal", seed=5)
    o_victim, _ = _opt_for(victim, "S-S")
    loaded = torch.load(ck, map_location="cpu", weights_only=False)
    assert (len(loaded["opt"]["param_groups"][0]["params"])
            == len(o_victim.state_dict()["param_groups"][0]["params"])), \
        "this test is only meaningful if the optimiser shapes COLLIDE"
    # the stage check refuses anyway — it never looks at an optimiser
    with pytest.raises(ResumeLineageError, match="'S-T'"):
        load_resume(victim, o_victim, ck, stage="S-S")


def test_the_stage_counts_that_the_old_barrier_leaned_on_are_a_coincidence():
    """The four stages train different numbers of TENSORS today — which is the
    only reason the old accidental barrier ever fired. Pinned so that the day
    a STAGE_GROUPS edit makes two of them collide, this test says so out loud
    instead of the ladder silently losing its guard."""
    counts = {}
    for stg in STAGES:
        s = mk("goal", seed=6)
        apply_stage_freeze(s, stg)
        counts[stg] = sum(1 for p in s.parameters() if p.requires_grad)
    dupes = {(a, b) for a in counts for b in counts
             if a < b and counts[a] == counts[b]}
    assert not dupes, (
        f"stages {dupes} now train the same number of tensors — the OLD "
        f"accidental optimiser barrier would no longer fire for them. The "
        f"stage check in assert_resume_lineage is what actually protects the "
        f"ladder; this is only a note that the fallback is gone. counts="
        f"{counts}")


def test_same_stage_resume_still_works(tmp_path):
    """The regression guard: the fix must not make a NORMAL resume harder."""
    ck = write_ckpt(tmp_path / "ckpt.pt", mk("goal", seed=7), "S-S", 4321)
    prov = assert_resume_lineage(ck, stage="S-S")
    assert prov["stage"] == "S-S" and prov["step"] == 4321
    dst = mk("goal", seed=8)
    dopt, _ = _opt_for(dst, "S-S")
    assert load_resume(dst, dopt, ck, stage="S-S") == 4321


def test_an_UNLABELLED_checkpoint_is_refused(tmp_path):
    """Every checkpoint this trainer writes carries its stage. One that does
    not came from somewhere else, and its lineage cannot be verified."""
    p = tmp_path / "ckpt.pt"
    torch.save({"stack": mk("goal", seed=9).state_dict(), "opt": {},
                "step": 100}, p)
    with pytest.raises(ResumeLineageError, match="no stage label"):
        assert_resume_lineage(p, stage="S-S")


def test_an_UNREADABLE_checkpoint_is_refused_with_a_diagnosis(tmp_path):
    """Not an opaque pickle traceback — the same discipline as every other
    refusal in this file."""
    p = tmp_path / "ckpt.pt"
    p.write_bytes(b"this is not a checkpoint")
    prov = read_ckpt_provenance(p)
    assert prov["readable"] is False and prov["error"]
    with pytest.raises(ResumeLineageError, match="could not read it"):
        assert_resume_lineage(p, stage="S-W")


def test_load_resume_without_a_named_stage_keeps_the_old_behaviour(tmp_path):
    """A check must be ASKED for, never inherited by default — the same
    contract `load_stage_init` uses for its allowance. `train` always names the
    stage; a direct caller that does not gets the pre-fix path, and that is a
    deliberate, tested choice rather than an oversight."""
    ck = write_ckpt(tmp_path / "ckpt.pt", mk("goal", seed=10), "S-S", 77)
    dst = mk("goal", seed=11)
    dopt, _ = _opt_for(dst, "S-S")
    assert load_resume(dst, dopt, ck) == 77          # no stage= -> no check


# ============================================================================
# EDGE 3b — the fp16 snapshot: refused as a RESUME, accepted as an INIT
# ============================================================================

@pytest.fixture
def snapshot_file(tmp_path):
    from ckpt_fp16_snapshot import snapshot
    src = write_ckpt(tmp_path / "full.pt", mk("goal", seed=12), "S-T", 30000)
    dst = tmp_path / "weights_fp16.pt"
    snapshot(str(src), str(dst))
    return dst


def test_the_fp16_snapshot_is_recognised_for_what_it_is(snapshot_file):
    prov = read_ckpt_provenance(snapshot_file)
    assert prov["readable"] and prov["weights_only_snapshot"] is True
    assert prov["has_opt"] is False
    # ⭐ its stage and step survive the snapshot, under `_meta`
    assert prov["stage"] == "S-T" and prov["step"] == 30000


def test_a_weights_only_snapshot_is_REFUSED_as_a_resume(snapshot_file):
    """⛔ It carries no optimiser state BY DESIGN (two thirds of the bytes).
    Half-honouring it would inherit a step and an LR schedule position with
    nothing behind them — and, because the `if "opt" in ck` branch is simply
    skipped, it is the one path where the OLD accidental barrier could not fire
    at all."""
    with pytest.raises(ResumeLineageError, match="NO optimiser state"):
        assert_resume_lineage(snapshot_file, stage="S-T")
    with pytest.raises(ResumeLineageError, match="NO optimiser state"):
        assert_resume_lineage(snapshot_file, stage="S-S")   # even same-stage
    assert "--init-from" in RESUME_CONTRACT["has_optimiser"], \
        "the contract must name the flag that DOES accept this artifact"


def test_init_from_CAN_now_read_the_snapshot_its_docstring_promises(
        snapshot_file):
    """`ops/ckpt_fp16_snapshot.py` states the snapshot "is enough for the
    P-battery, any eval, and --init-from". MEASURED 2026-08-16: it was not —
    the state lives under "model", `load_stage_init` looked for "stack", fell
    through to the wrapper dict and refused with a 400-key GEOMETRY MISMATCH,
    blaming the architecture for an unopened container. This is the handover
    path a rebuilt pod actually takes (the 3.53 GB ckpt.pt never once pushed)."""
    rep = load_stage_init(mk("goal", seed=13), snapshot_file, stage="S-S")
    assert rep["missing_keys"] == [] and rep["unexpected_keys"] == []
    assert rep["prev_stage"] == "S-T" and rep["init_step"] == 30000
    # ⚠️ and it SAYS the weights came through fp16 — a trunk md5 that cannot
    # equal the fp32 source's must not look like a lineage break later.
    assert rep["init_source"] == "fp16_weights_only_snapshot"
    assert "lossy" in rep["init_precision"]


def test_load_resume_on_a_snapshot_names_the_container_not_the_geometry(
        snapshot_file):
    """Direct call, no stage named: it must still not die on `KeyError:
    'stack'`, which is what it did before."""
    dst = mk("goal", seed=14)
    dopt, _ = _opt_for(dst, "S-T")
    with pytest.raises(ResumeLineageError, match="no 'stack' key"):
        load_resume(dst, dopt, snapshot_file)


# ============================================================================
# EDGE 3c — `--init-from` + `--resume auto`: the record must not lie
# ============================================================================

def test_a_resume_supersedes_the_init_report_instead_of_letting_it_lie():
    """⛔ MEASURED 2026-08-16: config.json recorded the INIT's trunk md5 while
    the model held the RESUME's weights (`fbce009a…` vs `326034884…`), because
    `train` loads the init first and the resume second. Nothing warned.

    ⚠️ The fix is NOT to refuse the flag pair: `supervise_run.sh` replays the
    command captured at startup, so the relaunch that resumes necessarily still
    carries the `--init-from` that seeded the run."""
    init = {"init_from": "/w/S-T/ckpt.pt", "init_step": 30000,
            "trunk_md5_after_load": "fbce009ab9fe1b064ab9c44dccf1dc6b",
            "prev_stage": "S-T"}
    out = supersede_init_on_resume(init, "/w/S-S/ckpt.pt")
    assert out["init_from"] is None, \
        "a superseded init must not read as this run's lineage"
    assert out["resumed_from"] == "/w/S-S/ckpt.pt"
    # the evidence is DEMOTED, never deleted — the launch really did do it
    assert out["superseded_by_resume"]["trunk_md5_after_load"] == \
        "fbce009ab9fe1b064ab9c44dccf1dc6b"
    assert "OVERWRITTEN" in out["superseded_by_resume"]["_status"]
    assert json.dumps(out)                    # must survive config.json


def test_supersede_is_a_no_op_when_there_was_no_init():
    rep = {"init_from": None}
    assert supersede_init_on_resume(rep, "/w/x/ckpt.pt") is rep


def test_the_lying_md5_is_reproduced_end_to_end(tmp_path):
    """The measurement itself, pinned: two DIFFERENT ancestors, and the trunk
    md5 recorded by `--init-from` provably does not describe the model once the
    resume has run. This is what `supersede_init_on_resume` exists for."""
    a = write_ckpt(tmp_path / "ancestor.pt", mk("goal", seed=15), "S-S", 30000)
    run = tmp_path / "run"
    run.mkdir()
    b = write_ckpt(run / "ckpt.pt", mk("goal", seed=16), "S-S", 12000)

    s = mk("goal", seed=17)
    rep = load_stage_init(s, a, stage="S-S")
    dopt, _ = _opt_for(s, "S-S")
    load_resume(s, dopt, b, stage="S-S")

    import hashlib
    h = hashlib.md5()
    for n, prm in sorted(s.named_parameters()):
        if s.group_of(n) in ("encoder", "readout", "predictor_op"):
            h.update(n.encode())
            h.update(prm.detach().cpu().numpy().tobytes())
    assert rep["trunk_md5_after_load"] != h.hexdigest(), (
        "the init md5 must differ from the post-resume trunk — if these ever "
        "match, this test has stopped exercising the overwrite")
    assert supersede_init_on_resume(rep, b)["init_from"] is None


# ============================================================================
# EDGE 4 — freeze x init, MEASURED on a real backward (with a vacuity control)
# ============================================================================

def _stage_batch(stack: V6Stack):
    b = synthetic_train_batch(stack, batch=2, k=12, seed=1)
    b["gt_wp"] = torch.randn(2, 10, 2)
    return b


def _grad_census(stack: V6Stack) -> dict:
    got: dict[str, dict] = {}
    for n, p in stack.named_parameters():
        g = stack.group_of(n)
        got.setdefault(g, {"grad": 0, "none": 0})
        got[g]["grad" if p.grad is not None else "none"] += 1
    return got


@pytest.fixture(scope="module")
def reachable_groups() -> set[str]:
    """⚠️ THE VACUITY CONTROL, and the test below is worthless without it.

    "group X received no gradient" proves the freeze only if the SAME loss
    reaches X when X is unfrozen. Otherwise a loss that simply never touches a
    module would certify it as "frozen" — a vacuous pass, and vacuous passes
    are how an isolation guarantee rots (`V6Stack.assert_isolation` makes the
    same point about `requires_grad`).

    MEASURED: with every parameter trainable and the S-J loss, the reachable
    fraction per group is encoder 17/17, readout 2/2, predictor_op 31/35,
    aux 30/30, layer_tac 54/67, layer_str 41/54, planner 2/13 — so every group
    has live parameters and every "no grad" below is a real statement.
    """
    s = mk("goal", seed=60)
    for p in s.parameters():
        p.requires_grad_(True)
    v6_loss_step(s, _stage_batch(s), stage="S-J", weights=V6LossWeights(),
                 o1_k=10, o5_k=10)["loss"].backward()
    census = _grad_census(s)
    reach = {g for g, v in census.items() if v["grad"] > 0}
    assert reach == set(census), \
        f"a group is unreachable by the S-J loss; the freeze test would be " \
        f"vacuous for it: {census}"
    return reach


@pytest.mark.parametrize("stage", list(STAGES))
def test_after_init_from_exactly_the_intended_groups_train(st_ckpt,
                                                           reachable_groups,
                                                           stage):
    """MEASUREMENT, not a reading of the freeze map: initialise stage N+1 from
    a real predecessor checkpoint, apply the stage freeze, run the REAL stage
    loss, and census `.grad` — every out-of-stage group must be `None`, and the
    control above proves each of them was reachable."""
    s = mk("goal", seed=61)
    if stage in ("S-S", "S-J"):
        load_stage_init(s, st_ckpt, stage=stage)
    apply_stage_freeze(s, stage)

    want = set(stage_trainable_groups(stage))
    wrong = [(n, s.group_of(n)) for n, p in s.named_parameters()
             if p.requires_grad != (s.group_of(n) in want)]
    assert not wrong, f"requires_grad disagrees with the stage: {wrong[:5]}"

    v6_loss_step(s, _stage_batch(s), stage=stage, weights=V6LossWeights(),
                 o1_k=10, o5_k=10)["loss"].backward()
    census = _grad_census(s)
    leaked = {g: v for g, v in census.items() if g not in want and v["grad"]}
    assert not leaked, (
        f"stage {stage} put gradient in groups it must not train: {leaked}. "
        f"Every one of these was reachable in the control, so this is a real "
        f"leak, not an artifact of an unreachable loss.")
    assert reachable_groups >= (set(census) - want), \
        "the control does not cover every group this stage freezes"
    assert any(census[g]["grad"] for g in want), \
        f"stage {stage} trained NOTHING — the freeze map and the loss disagree"


# ============================================================================
# EDGE 5 — X3 isolation must survive the transition, not only construction
# ============================================================================

@pytest.mark.parametrize("stage", ["S-S", "S-J"])
def test_X3_isolation_still_holds_AFTER_an_init_from(st_ckpt, stage):
    """X3 is checked at fresh construction elsewhere. A stage does not start
    from fresh construction — it starts from its predecessor's weights, and an
    isolation guarantee that was only ever measured on random init is a
    guarantee about a model nobody trains."""
    s = mk("goal", seed=62)
    load_stage_init(s, st_ckpt, stage=stage)
    apply_stage_freeze(s, stage)
    rep = s.assert_isolation(strict=True)
    assert rep["pass"] is True
    assert rep["n_violations"] == {"planner_to_encoder": 0,
                                   "tactical_to_below": 0,
                                   "strategic_to_below": 0}
    # ⚠️ non-vacuity: the probe must actually have had parameters to look at
    assert all(v > 0 for v in rep["n_probed"].values()), rep["n_probed"]


def test_the_resume_contract_is_stated_as_data():
    """Three requirements, each carrying WHY — the same discipline as
    STAGE_INVALIDATION_MECHANISM. A refusal with no mechanism teaches nobody."""
    assert set(RESUME_CONTRACT) == {"same_stage", "labelled", "has_optimiser"}
    for k, why in RESUME_CONTRACT.items():
        assert len(why) > 80, f"{k} has no mechanism, only a rule"
