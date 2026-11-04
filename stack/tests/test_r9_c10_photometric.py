"""refcv9 C10 -- ``--r9-photometric`` (per-sample photometric augmentation, ``tanitad/train/r9_photometric.py``).  CPU, the tiny full model / rig of ``test_map_only_ft``.

THE CLASS ("built, tested, and UNREACHABLE FROM ITS CALLER", CLAUDE.md 2026-09-10/11): the module's own unit tests passing proves nothing about the trainer, so this
file proves each hop and then DELETES the call line and requires the reach check to go RED.

  UNIT        in place (``data_ptr`` unchanged), [0, 1], ONE draw per sample shared by every window row, every stacked-frame triplet and the sample's future frames,
              different draws across samples, (seed, step)-deterministic with no global RNG consumed, the colour matrix keeps grey grey, and a per-sample PEAK
              LIVE-allocation bound measured by a dispatch probe (whose control -- an out-of-place implementation -- reads full-batch size, so it CAN fail)
  OFF         flag off => ``compute_losses_v3`` and ``compute_losses_perc_rig`` are bit-identical to the same functions re-compiled WITHOUT the call line
  ON          train => the trunk input (forward pre-hook on ``core.encoder``) differs from OFF, is in [0, 1], equals ``apply_`` on a clone at (seed, step), the batch's
              frames AND future frames carry the SAME per-sample draw; eval => untouched (bit-identical to OFF); every non-frame batch key byte-identical
  G-DVB       argv ON vs built OFF REFUSED (and the reverse, and a seed mismatch); a registered ``built`` lever the parser covers
  MUTATION    the ``_r9ph.maybe_apply_`` line deleted (in-process re-compile of each loss function AND a temp copy of the trainer file) => the reach check goes RED
  CENSUS      the trainer / loader hops exist exactly where they should, and ``maybe_apply_`` is NOT called by any eval / dump / recalibration path
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import re
import sys
import weakref
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_flatten  # noqa: E402

from tanitad.models.timm_trunk import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402
from tanitad.train import r9_photometric as P  # noqa: E402

TRAINER_PATH = ROOT / "scripts" / "refc_v3_train.py"
TRAINER_SRC = TRAINER_PATH.read_bytes().decode("utf-8").replace("\r\n", "\n")
LOADER_SRC = (ROOT / "tanitad" / "eval" / "refcv7_loader.py").read_bytes().decode("utf-8").replace("\r\n", "\n")
CALL_V3 = "_r9ph.maybe_apply_(model, frames, fut_frames)"
CALL_RIG = "_r9ph.maybe_apply_(model, frames)"
SEED, STEP = 11, 2
_ORIG_MODEL = T.v3.RefCV3Model                                       # the class as imported (a 2nd monkeypatch of the patched name would nest the lambda)


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ===================================================================================================================== #
# helpers                                                                                                               #
# ===================================================================================================================== #
def _args(on=True, seed=SEED):
    return argparse.Namespace(r9_photometric=on, seed=seed)


def _stack(b=4, rows=3, k=3, h=16, w=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, rows, 3 * k, h, w, generator=g)


def _build_rig(monkeypatch):
    monkeypatch.setattr(T.v3, "RefCV3Model", lambda cfg: _ORIG_MODEL(cfg, perc_rig=True))
    return R._build()


def _fresh(batch: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in batch.items()}


def _dict_equal(a: dict, b: dict) -> bool:
    ta = {k: v for k, v in a.items() if torch.is_tensor(v)}
    tb = {k: v for k, v in b.items() if torch.is_tensor(v)}
    return set(ta) == set(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)


def _on(model, seed=SEED, step=STEP):
    P.attach(model, _args(True, seed))
    P.set_step(model, step)
    return model


def _trunk(model, fn, batch, *, train=True, site="core.encoder"):
    """-> (the first args the trunk was called with, the loss dict).  ``batch`` is cloned: with a float CPU batch ``frames_to_device`` ALIASES it.
    ``site``: the full model calls ``core.encoder`` ([B, 9, H, W]); the perception rig calls ``core.encoder.net`` once per RGB triplet ([B*3, 3, H, W])."""
    seen = []
    h = dict(model.named_modules())[site].register_forward_pre_hook(lambda m, a: seen.append(a[0].detach().clone()))
    try:
        model.train(train)
        torch.manual_seed(5)
        out = fn(model, _fresh(batch), "cpu", log_metrics=False)
    finally:
        h.remove()
    if site.endswith(".net"):                    # the rig reaches the timm net AFTER the encoder's ImageNet normalisation: undo it to read [0, 1] frames (to fp32 rounding)
        mean, std = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        seen = [t * std + mean for t in seen]
    return seen, out


def _variant(fn, line):
    """``fn`` re-compiled from ITS OWN SOURCE with the C10 call line deleted (an in-process source mutation; no file is touched)."""
    src = inspect.getsource(fn)
    pat = r"^[ ]*" + re.escape(line) + r"[^\n]*\n"
    src, n = re.subn(pat, "", src, flags=re.M)
    assert n == 1, ("the call line is not unique / not found", fn.__name__, line, n)
    g = dict(T.__dict__)
    exec(compile(src, f"<{fn.__name__} variant>", "exec"), g)
    return g[fn.__name__]


def _reach_problems(fn, build, batch, site="core.encoder", **mk) -> list:
    """The reach check (b): the ON model's trunk input must differ from the OFF model's, stay in [0, 1], and equal ``apply_`` on a clone at (seed, step)."""
    off, on = build(), _on(build(), **mk)
    seen_off, _ = _trunk(off, fn, batch, site=site)
    seen_on, _ = _trunk(on, fn, batch, site=site)
    out = []
    if len(seen_on) != len(seen_off) or not seen_on:
        return ["the trunk was not called the same number of times"]
    if all(torch.equal(a, b) for a, b in zip(seen_on, seen_off)):
        out.append("the trunk input is IDENTICAL to the un-augmented one")
    if any(float(a.min()) < -1e-5 or float(a.max()) > 1.0 + 1e-5 for a in seen_on):
        out.append("the trunk input left [0, 1]")
    return out


class LiveProbe(TorchDispatchMode):
    """Peak LIVE bytes of storages first allocated while the mode is active (tensors created before it are not counted).  A storage dies with its last tensor."""

    def __init__(self):
        super().__init__()
        self.live = self.peak = self.max_single = 0
        self._refs, self._bytes = {}, {}

    def _drop(self, ptr):
        self._refs[ptr] -= 1
        if self._refs[ptr] == 0:
            self.live -= self._bytes[ptr]
            del self._refs[ptr]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat_in = [a for a in tree_flatten((args, kwargs))[0] if isinstance(a, torch.Tensor)]
        out = func(*args, **kwargs)
        for o in tree_flatten(out)[0]:
            if not isinstance(o, torch.Tensor) or any(o is a for a in flat_in):
                continue
            ptr = o.untyped_storage().data_ptr()
            if ptr == 0:
                continue
            if ptr not in self._refs:
                if any(ptr == a.untyped_storage().data_ptr() for a in flat_in):
                    continue                                    # a view of a storage allocated before the mode
                nb = o.untyped_storage().nbytes()
                self._refs[ptr], self._bytes[ptr] = 0, nb
                self.live += nb
                self.peak = max(self.peak, self.live)
                self.max_single = max(self.max_single, nb)
            self._refs[ptr] += 1
            weakref.finalize(o, self._drop, ptr)
        return out


# ===================================================================================================================== #
# UNIT -- the augmentation itself                                                                                       #
# ===================================================================================================================== #
def test_in_place_in_range_and_it_actually_changes_the_frames():
    f, ff = _stack(seed=1), _stack(rows=5, seed=2)
    f0, ff0, pf, pff = f.clone(), ff.clone(), f.data_ptr(), ff.data_ptr()
    frac = P.apply_(f, ff, seed=SEED, step=STEP)
    assert frac == 1.0 and f.data_ptr() == pf and ff.data_ptr() == pff                       # IN PLACE: the same storage
    for t, t0 in ((f, f0), (ff, ff0)):
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0 and bool(torch.isfinite(t).all())
        assert not torch.equal(t, t0) and 0.0 < float((t - t0).abs().mean()) < 0.5          # moved, but mildly (+-0.2 factors)


def test_one_draw_per_sample_shared_by_rows_stack_groups_and_future_frames_and_different_across_samples():
    base = torch.rand(1, 1, 3, 16, 24, generator=torch.Generator().manual_seed(7))
    f = base.expand(5, 3, 3, 16, 24).repeat(1, 1, 3, 1, 1).contiguous()                       # [5, 3, 9, 16, 24]: the SAME triplet in all 3 groups, all 3 rows, all 5 samples
    ff = f.clone()
    P.apply_(f, ff, seed=SEED, step=STEP)
    for i in range(5):
        assert torch.equal(f[i], ff[i])                                                      # frames and future_frames: the SAME draw
        for r in range(1, 3):
            assert torch.equal(f[i, 0], f[i, r])                                              # every window row
        assert torch.equal(f[i, :, 0:3], f[i, :, 3:6]) and torch.equal(f[i, :, 3:6], f[i, :, 6:9])    # every stacked frame of the sample
    assert all(not torch.equal(f[i], f[j]) for i in range(5) for j in range(i + 1, 5))        # different samples: different draws


def test_the_blur_is_drawn_per_sample_and_smooths(monkeypatch):
    f = _stack(b=3, rows=2, seed=3)
    tv = lambda t: float((t[..., 1:, :] - t[..., :-1, :]).abs().mean() + (t[..., :, 1:] - t[..., :, :-1]).abs().mean())  # noqa: E731
    monkeypatch.setattr(P, "BLUR_P", 0.0)
    a = f.clone(); P.apply_(a, None, seed=SEED, step=STEP)
    monkeypatch.setattr(P, "BLUR_P", 1.0)
    b = f.clone(); P.apply_(b, None, seed=SEED, step=STEP)
    assert tv(b) < 0.8 * tv(a) and float(b.min()) >= 0.0 and float(b.max()) <= 1.0           # blurred: less high-frequency energy; the other factors are the same draw
    assert abs(float(a.mean()) - float(b.mean())) < 0.02                                      # the blur moves no colour


def test_deterministic_in_seed_and_step_and_consumes_no_global_rng():
    f = _stack(seed=4)
    torch.manual_seed(123)
    st0 = torch.get_rng_state().clone()
    a, b, c, d = (f.clone() for _ in range(4))
    P.apply_(a, None, seed=SEED, step=STEP)
    P.apply_(b, None, seed=SEED, step=STEP)
    P.apply_(c, None, seed=SEED, step=STEP + 1)
    P.apply_(d, None, seed=SEED + 1, step=STEP)
    assert torch.equal(torch.get_rng_state(), st0)                                            # NO RNG STATE: nothing for a resume to restore
    assert torch.equal(a, b)                                                                  # same (seed, step): same output
    assert not torch.equal(a, c) and not torch.equal(a, d)                                    # step + 1 and seed + 1: another draw
    assert P.draw_params(6, SEED, STEP) == P.draw_params(6, SEED, STEP) != P.draw_params(6, SEED, STEP + 1)
    assert P.draw_params(6, SEED, STEP)["fb"][:3] == P.draw_params(3, SEED, STEP)["fb"]       # sample i's draw does not depend on how many samples follow it


def test_the_ranges_the_colour_matrix_and_the_hue_rule():
    for step in range(200):
        p = P.draw_params(8, 3, step)
        assert all(0.8 <= v <= 1.2 for k in ("fb", "fc", "fs") for v in p[k]) and all(abs(v) <= 0.02 for v in p["hue"])
        assert all(P.BLUR_SIGMA_RANGE[0] <= s <= P.BLUR_SIGMA_RANGE[1] for s in p["sigma"])
    assert torch.allclose(P.color_matrix(1.0, 0.0), torch.eye(3, dtype=torch.float64), atol=1e-6)    # s = 1, h = 0: the identity
    ones = torch.ones(3, dtype=torch.float64)
    for s, h in ((0.8, 0.02), (1.2, -0.02), (1.0, 0.02)):
        assert torch.allclose(P.color_matrix(s, h) @ ones, ones, atol=1e-6)                   # grey stays grey: hue / saturation move CHROMA only
    yiq = torch.tensor(P._YIQ, dtype=torch.float64)
    red = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    before, after = yiq @ red, yiq @ (P.color_matrix(1.0, 0.02) @ red)
    ang = float(torch.atan2(after[2], after[1]) - torch.atan2(before[2], before[1]))
    assert abs(float(after[0] - before[0])) < 1e-9 and abs(ang - 2.0 * 3.141592653589793 * 0.02) < 1e-6      # luma kept; the (I, Q) chroma plane rotated by 0.02 of a turn = 7.2 degrees
    sat = yiq @ (P.color_matrix(1.2, 0.0) @ red)
    assert abs(float(sat[0] - before[0])) < 1e-9 and abs(float(torch.hypot(sat[1], sat[2]) / torch.hypot(before[1], before[2])) - 1.2) < 1e-9     # saturation scales chroma only


def test_peak_live_allocation_is_bounded_per_sample_and_the_probe_can_fail():
    b, rows, k, h, w = 4, 6, 3, 16, 24
    f, ff = _stack(b, rows, k, h, w, seed=5), _stack(b, rows, k, h, w, seed=6)
    group = rows * 3 * h * w * 4                                                             # one sample's ONE stacked-frame triplet, bytes
    sample, batch_bytes = group * k, f.numel() * 4
    ptr = ff.data_ptr()
    with LiveProbe() as pr:
        P.apply_(f, ff, seed=SEED, step=STEP)
    assert ff.data_ptr() == ptr
    assert pr.peak <= group and pr.max_single <= group // 2, (pr.peak, pr.max_single, group)   # never more than ONE triplet-group of one sample, whatever B is
    assert pr.peak < sample < batch_bytes
    with LiveProbe() as ctl:                                                                 # CONTROL: an out-of-place implementation of one op
        g = f.mul(1.1).clamp(0.0, 1.0)
    assert ctl.peak >= batch_bytes and ctl.max_single >= batch_bytes                          # the probe reads full-batch size on it: the bound above is not vacuous
    del g
    big = _stack(b * 2, rows, k, h, w, seed=8)                                               # twice the batch, the SAME per-sample peak
    with LiveProbe() as pr2:
        P.apply_(big, None, seed=SEED, step=STEP)
    assert pr2.peak <= group


def test_refusals():
    with pytest.raises(ValueError, match="float"):
        P.apply_((_stack() * 255).to(torch.uint8), None, seed=0, step=0)
    with pytest.raises(ValueError, match="C % 3"):
        P.apply_(torch.rand(2, 3, 4, 8, 8), None, seed=0, step=0)
    with pytest.raises(ValueError, match="float"):
        P.apply_(torch.rand(2, 9, 8, 8), None, seed=0, step=0)
    with pytest.raises(ValueError, match="does not match"):
        P.apply_(_stack(b=2), _stack(b=3), seed=0, step=0)
    with pytest.raises(ValueError, match="non-negative"):
        P.apply_(_stack(), None, seed=-1, step=0)


# ===================================================================================================================== #
# OFF -- the flag off is the tip                                                                                        #
# ===================================================================================================================== #
def test_OFF_compute_losses_v3_is_bit_identical_to_the_function_without_the_call_line():
    (model, cfg), (model2, _) = R._build(), R._build()                  # two identical builds: a first call mutates a model (buffers, dedicated generators)
    batch = R._batch(cfg)
    assert not hasattr(model, P.MODEL_ATTR) and P.metrics(model) == {}
    ref = _variant(T.compute_losses_v3, CALL_V3)
    _, a = _trunk(model, T.compute_losses_v3, batch)
    _, b = _trunk(model2, ref, batch)
    assert _dict_equal(a, b) and "aug/photometric_frac" not in a


def test_OFF_compute_losses_perc_rig_is_bit_identical_to_the_function_without_the_call_line(monkeypatch):
    (model, cfg), (model2, _) = _build_rig(monkeypatch), _build_rig(monkeypatch)
    batch = R._batch(cfg)
    ref = _variant(T.compute_losses_perc_rig, CALL_RIG)
    _, a = _trunk(model, T.compute_losses_perc_rig, batch, site="core.encoder.net")
    _, b = _trunk(model2, ref, batch, site="core.encoder.net")
    assert _dict_equal(a, b)


# ===================================================================================================================== #
# ON -- training only                                                                                                   #
# ===================================================================================================================== #
def _check_on_full(build):
    model, cfg = build()
    batch = R._batch(cfg)
    # (b) the trunk sees changed frames, in [0, 1]
    off, _ = build()
    seen_off, loss_off = _trunk(off, T.compute_losses_v3, batch)
    seen_on, loss_on = _trunk(_on(model), T.compute_losses_v3, batch)
    return model, batch, seen_off, seen_on, loss_off, loss_on


def test_ON_train_the_trunk_sees_augmented_frames_drawn_at_seed_step_and_the_metric_fires():
    model, batch, seen_off, seen_on, loss_off, loss_on = _check_on_full(R._build)
    assert len(seen_on) == len(seen_off) >= 1
    for a, b in zip(seen_on, seen_off):
        assert not torch.equal(a, b) and float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    assert not torch.equal(loss_on["loss"], loss_off["loss"]) and bool(torch.isfinite(loss_on["loss"]))
    # the batch's frames AND future frames carry exactly ONE per-sample draw at (seed, step): reproduce it on clones
    probe = _fresh(batch)
    model.train()
    T.compute_losses_v3(model, probe, "cpu", log_metrics=False)                              # float CPU batch: frames_to_device ALIASES it, so the batch itself now holds the jitter
    want_f, want_ff = batch["frames"].clone(), batch["future_frames"].clone()
    assert P.apply_(want_f, want_ff, seed=SEED, step=STEP) == 1.0
    assert torch.equal(probe["frames"], want_f) and torch.equal(probe["future_frames"], want_ff)
    assert P.metrics(model) == {"aug/photometric_frac": 1.0}
    # step + 1 is another draw
    P.set_step(model, STEP + 1)
    seen_next, _ = _trunk(model, T.compute_losses_v3, batch)
    assert not torch.equal(seen_next[0], seen_on[0])
    P.set_step(model, STEP)
    seen_again, _ = _trunk(model, T.compute_losses_v3, batch)
    assert torch.equal(seen_again[0], seen_on[0])                                              # same (seed, step): the same trunk input


def test_ON_eval_leaves_everything_untouched():
    model, cfg = R._build()
    batch = R._batch(cfg)
    off, _ = R._build()
    seen_off, loss_off = _trunk(off, T.compute_losses_v3, batch, train=False)
    seen_on, loss_on = _trunk(_on(model), T.compute_losses_v3, batch, train=False)
    assert all(torch.equal(a, b) for a, b in zip(seen_on, seen_off)) and _dict_equal(loss_on, loss_off)
    probe = _fresh(batch)
    model.eval()
    T.compute_losses_v3(model, probe, "cpu", log_metrics=False)
    assert torch.equal(probe["frames"], batch["frames"]) and torch.equal(probe["future_frames"], batch["future_frames"])    # not even in place
    assert P.metrics(model)["aug/photometric_frac"] == 0.0                                    # nothing was jittered by this model yet


def test_ON_train_changes_no_non_frame_batch_key():
    model, cfg = R._build()
    batch = R._batch(cfg)
    before = _fresh(batch)
    probe = _fresh(batch)
    _on(model).train()
    T.compute_losses_v3(model, probe, "cpu", log_metrics=False)
    changed = [k for k in before if k not in ("frames", "future_frames") and torch.is_tensor(before[k]) and
               (probe[k].dtype != before[k].dtype or probe[k].shape != before[k].shape or probe[k].numpy().tobytes() != before[k].numpy().tobytes())]
    assert changed == [] and set(probe) == set(before)                                       # label invariance: every pose / nav / goal / map key is byte-identical
    assert not torch.equal(probe["frames"], before["frames"]) and not torch.equal(probe["future_frames"], before["future_frames"])


def test_ON_the_perception_rig_path_augments_the_newest_row_only_in_training(monkeypatch):
    model, cfg = _build_rig(monkeypatch)
    batch = R._batch(cfg)
    off, _ = _build_rig(monkeypatch)
    seen_off, _ = _trunk(off, T.compute_losses_perc_rig, batch, site="core.encoder.net")
    seen_on, _ = _trunk(_on(model), T.compute_losses_perc_rig, batch, site="core.encoder.net")
    assert not torch.allclose(seen_on[0], seen_off[0], atol=1e-4) and float(seen_on[0].min()) >= -1e-5 and float(seen_on[0].max()) <= 1.0 + 1e-5
    want = batch["frames"][:, -1:].clone()
    P.apply_(want, None, seed=SEED, step=STEP)
    assert torch.allclose(seen_on[0], want[:, 0].reshape(-1, 3, *want.shape[-2:]), atol=1e-5)               # the trunk's input IS the newest row's K RGB triplets at (seed, step)
    assert P.metrics(model) == {"aug/photometric_frac": 1.0}
    seen_ev, _ = _trunk(model, T.compute_losses_perc_rig, batch, train=False, site="core.encoder.net")
    assert torch.equal(seen_ev[0], seen_off[0])                                              # eval: untouched (normalise / un-normalise are the same ops on both sides)


# ===================================================================================================================== #
# G-DVB                                                                                                                 #
# ===================================================================================================================== #
def test_gdvb_entry_refuses_argv_on_with_built_off_and_the_reverse_and_a_seed_mismatch():
    assert D.REGISTRY["r9_photometric"].kind == "built" and "r9_photometric" not in D.coverage(T.build_parser())
    assert any(a.dest == "r9_photometric" and a.default is False for a in T.build_parser()._actions)        # default OFF
    chk = D.REGISTRY["r9_photometric"].check
    off, _ = R._build()
    on = _on(R._build()[0])
    assert chk(off, _args(False)) == [] and chk(on, _args(True)) == []
    bad = chk(off, _args(True))
    assert len(bad) == 1 and bad[0].declared is True and bad[0].built is False and "r9_photometric" in str(bad[0])        # argv ON, built OFF: REFUSED
    assert len(chk(on, _args(False))) == 1                                                    # built ON, argv OFF: REFUSED
    assert len(chk(on, _args(True, seed=SEED + 1))) == 1                                      # another seed than the argv's: REFUSED
    with pytest.raises(SystemExit, match="G-DVB"):
        D.refuse_on_mismatch(off, _args(True), None)                                          # the trainer's own refusal path carries it


def test_the_stamp_names_the_flag_and_is_absent_when_off():
    off, _ = R._build()
    assert P.stamp(off, _args(False)) is None
    st = P.stamp(_on(R._build()[0]), _args(True))
    assert st["flag"] == "--r9-photometric" and st["seed"] == SEED and st["brightness"] == 0.2 and st["hue_turns"] == 0.02
    assert "NOT built" in st["flip"] and "comma2k19.py:633" in st["channel_layout"]


# ===================================================================================================================== #
# MUTATION -- delete the call line, the reach check must go RED                                                         #
# ===================================================================================================================== #
def test_MUTATION_the_reach_check_is_green_on_the_real_functions_and_RED_with_the_call_line_deleted(monkeypatch):
    batch = R._batch(R._build()[1])
    assert _reach_problems(T.compute_losses_v3, lambda: R._build()[0], batch) == []                                    # GREEN control
    red = _reach_problems(_variant(T.compute_losses_v3, CALL_V3), lambda: R._build()[0], batch)
    assert red == ["the trunk input is IDENTICAL to the un-augmented one"]                                           # RED: compute_losses_v3
    build_rig = lambda: _build_rig(monkeypatch)[0]                                                                # noqa: E731
    assert _reach_problems(T.compute_losses_perc_rig, build_rig, batch, site="core.encoder.net") == []
    assert _reach_problems(_variant(T.compute_losses_perc_rig, CALL_RIG), build_rig, batch, site="core.encoder.net") == [
        "the trunk input is IDENTICAL to the un-augmented one"]                                                      # RED: compute_losses_perc_rig


def test_MUTATION_a_temp_copy_of_the_trainer_without_the_call_line_goes_RED(tmp_path):
    assert TRAINER_SRC.count(CALL_V3) == 1
    mut = tmp_path / "refc_v3_train_c10mut.py"
    mut.write_text("".join(ln for ln in TRAINER_SRC.splitlines(keepends=True) if CALL_V3 not in ln), encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location("refc_v3_train_c10mut", mut)
    M = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = M
    try:
        spec.loader.exec_module(M)
    finally:
        sys.modules.pop(spec.name, None)
    assert M.compute_losses_v3 is not T.compute_losses_v3 and CALL_V3 not in inspect.getsource(M.compute_losses_v3)
    batch = R._batch(R._build()[1])
    assert _reach_problems(T.compute_losses_v3, lambda: R._build()[0], batch) == []
    assert _reach_problems(M.compute_losses_v3, lambda: R._build()[0], batch) == ["the trunk input is IDENTICAL to the un-augmented one"]


def test_MUTATION_a_memory_probe_control_a_full_size_temporary_is_caught():
    f = _stack(b=4, rows=6, seed=9)
    group = 6 * 3 * 16 * 24 * 4
    with LiveProbe() as pr:
        f.copy_((f * 1.05).clamp(0.0, 1.0))                                                    # what a naive in-place-looking version does
    assert pr.peak > group                                                                    # the bound test above would go RED on it


# ===================================================================================================================== #
# CENSUS -- where the hops are, and where they are NOT                                                                  #
# ===================================================================================================================== #
def _body(src: str, start: str, end: str) -> str:
    return src[src.index(start):src.index(end, src.index(start) + 10)]


def test_census_the_trainer_and_loader_hops_sit_exactly_where_they_should():
    v3 = _body(TRAINER_SRC, "def compute_losses_v3(", "\ndef ")
    rig = _body(TRAINER_SRC, "def compute_losses_perc_rig(", "\ndef ")
    assert v3.count(CALL_V3) == 1 and rig.count(CALL_RIG) == 1 and TRAINER_SRC.count("_r9ph.maybe_apply_(") == 2
    assert v3.index('fut_frames = frames_to_device(batch["future_frames"], device)') < v3.index(CALL_V3) < v3.index("if ablate_frames:")
    assert rig.index('frames = frames_to_device(batch["frames"][:, -1:], device)') < rig.index(CALL_RIG) < rig.index("out = model(frames")
    # the loop sets the step BEFORE the first loss of the step (the fp32 replay included) and logs the metric
    assert TRAINER_SRC.count("_r9ph.set_step(model, step)") == 1 and TRAINER_SRC.count("losses.update(_r9ph.metrics(model))") == 1
    i_step = TRAINER_SRC.index("_r9ph.set_step(model, step)")
    assert i_step < TRAINER_SRC.index("_gs_replay = _gshare.fp32_replay(") and i_step < TRAINER_SRC.index("losses = compute_losses_v3(model, batch, device, mode=args.mode,")
    assert TRAINER_SRC.count("_r9ph.attach(model, args)") == 1 and LOADER_SRC.count("tr._r9ph.attach(model, args)") == 1
    assert TRAINER_SRC.count("_r9ph.register_dvb_levers(_dvb.register)") == 1
    assert TRAINER_SRC.count("{_r9ph.STAMP_KEY: _r9ph.stamp(model, args)} if _r9ph.is_on(args)") == 1
    assert TRAINER_SRC.count('"--r9-photometric"') == 1 and "store_true" in TRAINER_SRC[TRAINER_SRC.index('"--r9-photometric"'):][:60]


def test_census_no_eval_dump_or_recalibration_path_can_augment():
    # frames_to_device itself, BN recalibration and the dump patch never call the augmentation
    assert "_r9ph" not in _body(TRAINER_SRC, "def frames_to_device(", "\ndef ")
    assert "_r9ph" not in _body(TRAINER_SRC, "def _resolve_rig_cameras(", "\ndef ")
    for name in ("r9_e1_dump.py", "g_box_overfit.py"):
        assert "r9_photometric" not in (ROOT / "scripts" / name).read_text(encoding="utf-8", errors="replace"), name
    # outside the two loss functions the only users are the loop's set_step / metrics, the attach and the stamp
    outside = TRAINER_SRC.replace(_body(TRAINER_SRC, "def compute_losses_v3(", "\ndef "), "").replace(_body(TRAINER_SRC, "def compute_losses_perc_rig(", "\ndef "), "")
    assert "maybe_apply_" not in outside and "r9_photometric.apply_" not in outside and "_r9ph.apply_" not in TRAINER_SRC
    # and the augmentation is inert in training=False whatever the state says
    st = P.PhotometricState(seed=1)
    stub = torch.nn.Module()
    setattr(stub, P.MODEL_ATTR, st)
    f = _stack(seed=12)
    f0 = f.clone()
    stub.eval()
    P.maybe_apply_(stub, f)
    assert torch.equal(f, f0)
    stub.train()
    P.maybe_apply_(stub, f)
    assert not torch.equal(f, f0) and st.frac == 1.0


# ===================================================================================================================== #
# E2E -- the REAL train() loop (CPU, smoke): the step reaches the draw, the metric reaches the log, the stamp reaches config.json  #
# ===================================================================================================================== #
_E2E = ["--arm", "hier", "--smoke", "--synth-episodes", "2", "--steps", "3", "--batch", "2", "--device", "cpu", "--log-every", "1", "--save-every", "100",
        "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k", "--no-trunk-pretrained", "--trunk-in-channels", "3", "--image-hw", "64", "128",
        "--equalize-bottom-rows", "8", "--seed", "5"]


def _run_train(tmp_path, monkeypatch, extra):
    import json
    calls = []
    real = P.apply_

    def spy(frames, fut_frames=None, *, seed, step):
        calls.append((int(seed), int(step), tuple(frames.shape), None if fut_frames is None else tuple(fut_frames.shape)))
        return real(frames, fut_frames, seed=seed, step=step)
    monkeypatch.setattr(P, "apply_", spy)
    out = tmp_path / "run"
    T.train(T.build_parser().parse_args(_E2E + extra + ["--out", str(out)]))
    rows = [json.loads(ln) for ln in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    return calls, rows, json.loads((out / "config.json").read_text(encoding="utf-8"))


def test_E2E_train_loop_the_step_the_metric_and_the_stamp_reach_their_consumers(tmp_path, monkeypatch):
    calls, rows, cfg = _run_train(tmp_path, monkeypatch, ["--r9-photometric"])
    assert [(c[0], c[1]) for c in calls] == [(5, 0), (5, 1), (5, 2)], calls              # one draw per TRAINING step, at (--seed, the loop's step); the in-run eval drew nothing
    assert all(c[2][0] == 2 and c[3] is not None and c[3][0] == 2 for c in calls)         # frames AND future_frames of the batch
    train_rows = [r for r in rows if "loss" in r and "aug/photometric_frac" in r]
    assert len(train_rows) >= 3 and all(r["aug/photometric_frac"] == 1.0 for r in train_rows)
    st = cfg["r9_photometric"]
    assert st["flag"] == "--r9-photometric" and st["seed"] == 5 and st["p_apply"] == 1.0 and "NOT built" in st["flip"]
    assert cfg["declared_vs_built"]["mismatches"] == 0                                    # G-DVB clean through the real train()


def test_E2E_train_loop_OFF_draws_nothing_logs_nothing_stamps_nothing(tmp_path, monkeypatch):
    calls, rows, cfg = _run_train(tmp_path, monkeypatch, [])
    assert calls == [] and not any("aug/photometric_frac" in r for r in rows) and "r9_photometric" not in cfg
    assert cfg["declared_vs_built"]["mismatches"] == 0
