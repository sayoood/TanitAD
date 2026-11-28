"""⛔⛔ A RESUMED RUN MUST CONTINUE THE DATA ORDER, NOT REPLAY IT.

⭐⭐ THE DEFECT, MEASURED 2026-09-23 BY READING THE LOOP before refcv6's first multi-day run.
``refc_v3_train.train()`` resumes from ``<out>/ckpt.pt`` (model, optimiser, step), and its
training loader was ``DataLoader(shuffle=True)`` -- a permutation drawn from the GLOBAL torch
RNG at ``iter(dl)``, which every launch re-seeds from ``--seed``. So a relaunch drew the SAME
permutation as step 0: a run resumed at step k re-trained ``perm[0 : k*B]`` and never reached
``perm[(steps-k)*B : steps*B]``. On a one-epoch budget resumed at its midpoint, half the corpus
is trained twice and half never, with a healthy loss curve and nothing in any log.

⭐ WHAT IS PINNED, and why each piece is independent of the code under test:
(a) the SAMPLER + the loop step (``next_train_batch``), called for real over a real
    ``DataLoader``: a resume from EVERY batch boundary of a multi-epoch sequence continues the
    uninterrupted sequence exactly. The reference is the uninterrupted run itself, not a
    re-derivation of the permutation.
(b) ``train()`` END TO END on the synthetic rig: the windows a 3+3-step resumed run trains on
    in steps 4-6 are the windows a 6-step uninterrupted run trains on in steps 4-6 -- read off
    ``V3Dataset.__getitem__``, i.e. the data the model actually received.
(c) the refusals: a changed ``--batch`` or corpus size refuses, naming the field; a legacy
    checkpoint resumes and SAYS the order restarted.
CPU only, synthetic data.
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

N, B = 10, 3                  # 3 batches per epoch with drop_last, 1 window dropped


def _loader(seed=0):
    ds = torch.utils.data.TensorDataset(torch.arange(N))
    s = T.ResumableEpochSampler(N, seed, B)
    return torch.utils.data.DataLoader(ds, batch_size=B, sampler=s,
                                       drop_last=True), s


def _draw(dl, s, dpos, n_batches):
    out, it = [], iter(dl)
    for _ in range(n_batches):
        it, (b,) = T.next_train_batch(it, dl, s, dpos)
        out.append((tuple(b.tolist()), dict(dpos)))
    return out


def test_a_resume_from_EVERY_boundary_continues_the_uninterrupted_order():
    """⭐⭐ The claim, over 8 batches = 2 full epochs + 2 (so both rollovers are crossed)."""
    dl, s = _loader()
    ref = _draw(dl, s, {"epoch": 0, "batch": 0}, 8)
    for r in range(1, 8):
        dl2, s2 = _loader()
        pos = ref[r - 1][1]                               # what ckpt.pt would carry
        T.resume_data_position(
            {"data_pos": {"scheme": T.ResumableEpochSampler.SCHEME, "epoch": pos["epoch"],
                          "batch": pos["batch"], "n": N, "seed": 0, "batch_size": B}},
            s2, N, B)
        got = _draw(dl2, s2, dict(pos), 8 - r)
        assert [g[0] for g in got] == [x[0] for x in ref[r:]], f"resume at batch {r}"


def test_each_epoch_is_a_PERMUTATION_and_epochs_DIFFER():
    """LITERAL structure, not values: 3 batches of 3 distinct windows per epoch, and the two
    epochs order them differently (a sampler that ignored the epoch would repeat itself)."""
    dl, s = _loader()
    ref = _draw(dl, s, {"epoch": 0, "batch": 0}, 6)
    e0 = [i for b, _ in ref[:3] for i in b]
    e1 = [i for b, _ in ref[3:] for i in b]
    assert len(set(e0)) == 9 and len(set(e1)) == 9
    assert e0 != e1
    assert [p["epoch"] for _, p in ref] == [0, 0, 0, 1, 1, 1]
    assert [p["batch"] for _, p in ref] == [1, 2, 3, 1, 2, 3]


def test_the_HISTORICAL_defect_is_visible_to_this_rig():
    """⛔ THE CONTROL. Re-seeding the global RNG and iterating a `shuffle=True` loader --
    what every relaunch did -- REPLAYS the first batches. If this ever reads 'continues',
    the rig above cannot tell the fix from the defect."""
    ds = torch.utils.data.TensorDataset(torch.arange(N))
    torch.manual_seed(0)
    first = [tuple(b.tolist()) for (b,) in torch.utils.data.DataLoader(
        ds, batch_size=B, shuffle=True, drop_last=True)][:2]
    torch.manual_seed(0)                                   # the relaunch
    again = [tuple(b.tolist()) for (b,) in torch.utils.data.DataLoader(
        ds, batch_size=B, shuffle=True, drop_last=True)][:2]
    assert again == first, "shuffle=True with a re-seeded RNG replays -- the defect"


def _state(pos_batch=1, **over):
    dp = {"scheme": T.ResumableEpochSampler.SCHEME, "epoch": 0, "batch": pos_batch,
          "n": N, "seed": 0, "batch_size": B}
    dp.update(over)
    return {"data_pos": dp}


@pytest.mark.parametrize("field,value", [("batch_size", 4), ("n", 11), ("seed", 1)])
def test_a_resume_with_a_CHANGED_run_REFUSES_and_names_the_field(field, value):
    _, s = _loader()
    with pytest.raises(SystemExit) as e:
        T.resume_data_position(_state(**{field: value}), s, N, B)
    assert field in str(e.value)


def test_a_LEGACY_checkpoint_resumes_and_SAYS_the_order_restarted():
    _, s = _loader()
    st = T.resume_data_position({}, s, N, B)
    assert st["legacy_checkpoint"] is True and st["resumed_from"] is None
    assert (s.epoch, s.skip_batches) == (0, 0)


# --------------------------------------------------------------------------- #
# (b) train() end to end                                                      #
# --------------------------------------------------------------------------- #
def _argv(out, steps):
    return ["--arm", "hier", "--out", str(out), "--smoke", "--synth-episodes", "2",
            "--steps", str(steps), "--batch", "2", "--device", "cpu",
            "--log-every", "1", "--save-every", "100"]


def _recorded(monkeypatch):
    seen: list[int] = []
    real = T.V3Dataset.__getitem__

    def _rec(self, i):
        seen.append(int(i))
        return real(self, i)

    monkeypatch.setattr(T.V3Dataset, "__getitem__", _rec)
    return seen


def test_train_RESUMED_trains_on_the_windows_the_uninterrupted_run_trains_on(
        tmp_path, monkeypatch):
    """⭐⭐ The claim on the real trainer. `--steps 3` saves at step 3 (step == steps); a
    second launch with `--steps 6` into the same --out resumes there. Its six windows must
    be the uninterrupted run's windows for steps 4-6 -- in order."""
    seen = _recorded(monkeypatch)
    T.train(T.build_parser().parse_args(_argv(tmp_path / "u", 6)))
    ref = list(seen)
    assert len(ref) == 12                                 # 6 steps x batch 2, workers 0

    seen.clear()
    T.train(T.build_parser().parse_args(_argv(tmp_path / "r", 3)))
    first = list(seen)
    ck = torch.load(tmp_path / "r" / "ckpt.pt", weights_only=False)
    assert ck["step"] == 3 and ck["data_pos"]["batch_size"] == 2
    seen.clear()
    T.train(T.build_parser().parse_args(_argv(tmp_path / "r", 6)))
    assert first == ref[:6]
    assert seen == ref[6:], "the resumed run did not continue the data order"

    cfg = json.loads((tmp_path / "r" / "config.json").read_text(encoding="utf-8"))
    assert cfg["data_order"]["resumed_at_step"] == 3
    assert cfg["data_order"]["resumed_from"] == {
        "epoch": ck["data_pos"]["epoch"], "batch": ck["data_pos"]["batch"]}
    assert cfg["data_order"]["legacy_checkpoint"] is False
