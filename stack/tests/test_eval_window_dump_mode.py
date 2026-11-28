"""The W-BOOTSTRAP window dump must be the SAME measurement as the eval it resamples.

⛔⛔ THE DEFECT (found 2026-09-19 while preparing A7, the dump's first real consumer).
`refc_v3_train.py`'s held-out block ends with `model.train()`, and the W-BOOTSTRAP
per-window pass ran AFTER it -- i.e. in TRAIN mode, under `no_grad`. On this model
train mode is not cosmetic:

* `ego_dropout = 0.5` zeroes v0 on half the windows and `route_dropout = 0.5` masks
  the route (`refc.py`), so every dump row is a stochastic, degraded condition --
  not the eval-mode window value the aggregate `eval_traj` averages;
* `compute_losses_v3` gates `update_tactical_prior` on `model.training`
  (C-REFCV3-EVAL-PRIOR-LEAK, fixed 2026-09-02), so in train mode the dump pass EMA'd
  the HELD-OUT labels into `core.lat/lon_log_prior` again -- the closed leak,
  re-opened by a second eval pass;
* the dropout draws, and creating the dump loader's iterator, consume the GLOBAL
  torch RNG, so for `--eval-every < --steps` every later training step ran on a
  different stream than the same run without the dump -- contradicting the dump's
  own contract ("a SEPARATE opt-in pass ... every banked comparison still holds").

MEASURED on the only real run that used the flag (`refcv6-windump-20260919`, 16
held-out windows, 0 GPU): dump-row mean `traj` 14.50351 vs the aggregate
`eval_traj` 14.59615 over the SAME 16 windows; `loss` 178.16631 vs 180.60571.

Pinned here on the synthetic `train()` rig of `test_refc_v3_save_before_eval.py`
(integer episode ids, a faked held-out provider). CPU-only; nothing touches a pod.
The three properties are checked END TO END through `T.train`, the real caller:

(a) every dump-pass forward sees `model.training == False`;
(b) ⭐ the dump rows REPRODUCE the aggregate row: over the same windows, the mean of
    the per-window `traj` equals `eval_traj` -- an independently produced number,
    not the producer re-run;
(c) the tactical prior buffers after the run equal the step-N checkpoint written
    BEFORE both eval passes;
(d) ⭐ the dump changes NOTHING about training: with `--eval-every 1 --steps 2`
    (an eval, then a training step) the final parameters and the global RNG state
    are `torch.equal` between a dump-ON and a dump-OFF run.

Each was shown to go RED against the pre-fix trainer (`mutate_bn_recalib.py`,
mutations M6/M7), and the unmutated control is GREEN.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T                                    # noqa: E402
from tanitad.data import v2_dataset as V2                    # noqa: E402

TRAIN_IDS = (101, 102)
EVAL_IDS = (201, 202)
EVAL_CACHE = "synthetic-held-out"      # never opened: the provider is faked
PRIOR_BUFFERS = ("core.lat_log_prior", "core.lon_log_prior")


def _argv(out, steps, dump):
    a = ["--arm", "hier", "--out", str(out), "--smoke", "--synth-episodes", "2",
         "--steps", str(steps), "--batch", "2", "--device", "cpu",
         "--log-every", "1", "--save-every", "1",
         "--eval-cache", EVAL_CACHE, "--eval-every", "1", "--eval-batches", "2"]
    return a + (["--eval-window-dump", str(dump)] if dump else [])


def _run(out, steps, dump):
    """One real `T.train` on the faked split; returns (calls, model, rng_after).

    `calls` records, for every `compute_losses_v3` call, the batch size and
    `model.training` -- the dump pass is the only one at batch 1."""
    real_synth, seen, calls = T._synth_episodes, {}, []

    def _train_eps(n, cfg, seed=0, min_frames=40):
        seen["core"] = cfg
        eps = real_synth(n, cfg, seed=seed, min_frames=min_frames)
        for ep, sid in zip(eps, TRAIN_IDS):
            ep.episode_id = sid
        return eps

    def _eval_providers(paths, lru_size=6, **kw):
        assert list(paths) == [EVAL_CACHE]
        eps = real_synth(len(EVAL_IDS), seen["core"], seed=77)
        for ep, sid in zip(eps, EVAL_IDS):
            ep.episode_id = sid
        return eps

    real_cl = T.compute_losses_v3

    def _spy(model, batch, *a, **k):
        calls.append({"bs": int(batch["frames"].shape[0]),
                      "training": bool(model.training), "model": model})
        return real_cl(model, batch, *a, **k)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(T, "_synth_episodes", _train_eps)
        m.setattr(V2, "build_v2_providers", _eval_providers)
        m.setattr(T, "compute_losses_v3", _spy)
        T.train(T.build_parser().parse_args(_argv(out, steps, dump)))
    return calls, calls[-1]["model"], torch.get_rng_state().clone()


@pytest.fixture(scope="module")
def one_step(tmp_path_factory):
    d = tmp_path_factory.mktemp("wdm1")
    calls, model, _ = _run(d / "run", 1, d / "windows.jsonl")
    return d, calls, model


# ----------------------------------------------------------------- (a)
def test_the_dump_pass_runs_in_EVAL_mode(one_step):
    _, calls, _ = one_step
    dump = [c for c in calls if c["bs"] == 1]
    assert len(dump) == 4, (
        "expected 4 dump forwards (2 eval batches x batch 2, re-read at batch 1), "
        "got %d -- the rig no longer exercises the dump" % len(dump))
    assert not any(c["training"] for c in dump), (
        "the W-BOOTSTRAP pass ran in TRAIN mode: ego/route dropout fired on the "
        "held-out windows and the tactical-prior gate was open")
    # control: the rig CAN see train mode -- the training step itself
    assert any(c["training"] for c in calls if c["bs"] == 2), (
        "control: no train-mode call was recorded, so (a) would be vacuous")


# ----------------------------------------------------------------- (b)
def _aggregate_from_rows(rows, batch):
    """`eval_traj` rebuilt from the per-window rows by the LOSS'S OWN RULE.

    `loss_traj = sum(|err| * sv) / (2 * sum(sv))` over a batch
    (`refc_v3_train.py`, `denom = (sv.sum() * 2)`): a VALID-SLOT-WEIGHTED mean.
    At batch 1 a row's `traj` is its own window's ratio and `slot_valid_frac` is
    `sv.mean()`, so a batch's value is `sum(traj_w * frac_w) / sum(frac_w)` (the
    slot count cancels), and `eval_traj` is the mean over the eval batches, which
    are consecutive rows because both loaders walk the SAME `perm` in order."""
    vals = []
    for i in range(0, len(rows), batch):
        grp = rows[i:i + batch]
        w = sum(r["slot_valid_frac"] for r in grp)
        vals.append(sum(r["traj"] * r["slot_valid_frac"] for r in grp) / w
                    if w > 0 else 0.0)
    return sum(vals) / len(vals)


def test_the_dump_rows_REPRODUCE_the_aggregate_eval_row(one_step):
    """⭐ An INDEPENDENTLY produced target: the aggregate comes from batch-2 forwards
    and a batch-level loss; the rows from batch-1 forwards. They agree only if the
    dump measured the same model in the same mode on the same windows.

    ⚠️ MEASURED while writing this: the PLAIN mean of the rows is NOT `eval_traj`
    even in eval mode -- the loss weights windows by their valid future slots, and
    the eval pairs them into batches. A bootstrap that equal-weights rows
    therefore estimates a (slightly) different quantity; the A7 analysis uses the
    slot-weighted ratio for exactly this reason."""
    d, _, _ = one_step
    rows = [json.loads(x) for x in
            (d / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
    met = [json.loads(x) for x in
           (d / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    agg = [r for r in met if "eval_traj" in r]
    assert len(agg) == 1 and len(rows) == 4
    rebuilt = _aggregate_from_rows(rows, batch=2)
    assert rebuilt == pytest.approx(agg[0]["eval_traj"], rel=1e-5), (
        "the dump rows do not rebuild the aggregate eval_traj over the SAME "
        "windows (%.6f vs %.6f): the bootstrap would be resampling a different "
        "measurement" % (rebuilt, agg[0]["eval_traj"]))


def test_CONTROL_the_rig_really_has_partial_futures(one_step):
    """Without windows of unequal validity the weighting half of (b) is vacuous."""
    d, _, _ = one_step
    rows = [json.loads(x) for x in
            (d / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({r["slot_valid_frac"] for r in rows}) > 1, (
        "every dump window has the same valid-slot fraction -- the slot-weighting "
        "in (b) is untested; re-read this rig")


# ----------------------------------------------------------------- (c)
def test_the_dump_pass_does_not_touch_the_tactical_prior(one_step):
    d, _, model = one_step
    ck = torch.load(d / "run" / "ckpt.pt", weights_only=False)["model"]
    live = model.state_dict()
    for name in PRIOR_BUFFERS:
        assert torch.equal(ck[name], live[name]), (
            "%s moved after the step-1 checkpoint: a held-out pass EMA'd the "
            "held-out labels into the decode prior (C-REFCV3-EVAL-PRIOR-LEAK)" % name)


# ----------------------------------------------------------------- (d)
def test_the_dump_changes_NOTHING_about_training(tmp_path):
    """⭐ The contract the W-BOOTSTRAP block states in its own comment, made true
    and pinned: eval at step 1, then a training step, with and without the dump."""
    torch.manual_seed(0)
    _, m_off, r_off = _run(tmp_path / "off", 2, None)
    torch.manual_seed(0)
    _, m_on, r_on = _run(tmp_path / "on", 2, tmp_path / "windows.jsonl")
    assert (tmp_path / "windows.jsonl").exists()
    s_off, s_on = m_off.state_dict(), m_on.state_dict()
    assert s_off.keys() == s_on.keys()
    moved = [k for k in s_off if not torch.equal(s_off[k], s_on[k])]
    assert not moved, (
        "turning the window dump ON changed %d tensors after a later training step "
        "(first: %s) -- the dump is not a separate pass" % (len(moved), moved[:3]))
    assert torch.equal(r_off, r_on), (
        "the dump pass consumed training RNG: every later step ran on a different "
        "stream than the same run without it")
