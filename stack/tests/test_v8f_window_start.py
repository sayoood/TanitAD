"""PREREG_V8F_HIST Amendment 1: `--window-start`, the ONE lever that makes `--hist 10` mean what the PREREG says (B's window list, B's held ages, B's 19,164 steps).

THE FINDING THIS FIXES (BUILD.md F-1, pinned by `test_v8f_hist.py::test_FINDING_*`): `--hist 10` alone starts every clip's training windows at t = hist - 1 = 9, so t = 3..8 are not
trained on (27,072 windows, 8.8 %) and the step budget `S x steps_mult` falls from 19,164 to 17,472. Amendment 1: `--window-start` fixes BOTH the first training window of each clip
(`train_windows(start=...)`) AND the origin of the held-age clamp (`V8FMiniWindows.item`: room = max(t - window_start, 0)). ABSENT it is hist - 1, today's behaviour, so every arm that
does not set it is bit-identical. At `--hist 10 --window-start 3` the window list, every held age and the step budget equal `--hist 4`'s EXACTLY, and the readers see 10 frames of which
the ones before the clip are padded (frame_ok False).

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): a flag that the library supports and the CALLER never passes. So the identities below run
from argv, hop by hop, and the negative controls break each hop on purpose:

  (a) DEFAULT BIT-IDENTITY   without the flag, `train_windows` and every `item()` equal the PRE-AMENDMENT functions exactly: the frozen bodies restated below (always on), and the
                             whole pre-amendment modules saved in `raw/2026-10-23-hist/a1/` (when the package is in the checkout): every key of every item, the model's state_dict,
                             `forward_infer`, `built_levers` minus the three new keys.
  (b) hist 10 + window-start 3   the window list equals hist 4's window by window; every `item()` (all keys: the hmax-deep batch does not depend on hist) equals hist 4's under all 15
                             sampler draws; without the flag the same comparison fails (the control).
  (c) the counts on box 2's REAL train manifest: 306,566 windows / 19,164 steps at hist 4 and at hist 10 + window-start 3; 279,494 / 17,472 at hist 10 alone.
  (d) a window at t = 3, hist 10 + window-start 3: the scene gather masks exactly 6 padded frames and reads 4 real ones, padded content cannot move the output, the loss is finite.
  (e) G-DVB                  a declared `--window-start 3` that the model (built_levers), the window list the data built, or the dataset's clamp did not build is REFUSED, each hop
                             on its own; a declared flag with no read-back at all is refused; the trainer run end to end refuses a `train_windows` / dataset / `make_config` that
                             drops the flag; `built_levers` carries hist, hmax and the EFFECTIVE window_start.
  (f) mutations              `raw/2026-10-23-hist/a1/mutate_a1.py` (output beside it): every mutation turns this file RED.
  (g) the evaluators         `v8f_mini_eval.py` and `v8f_t1_eval.py` hand the dataset the CHECKPOINT's window_start (eval-side fix): the real `EV.main run` / `T1.main run` on the synthetic eval world
                             resolve an H checkpoint's held ages (default `--ages cycle`) exactly as a hist 4 checkpoint's at every window; a checkpoint whose saved cfg lacks the key loads with the
                             default and is scored bit for bit as by the saved pre-fix evaluator; a static check that every `V8FMiniWindows(` under stack/scripts/ passes `window_start=`.
Everything that needs the research package (the saved reference modules, box 2's manifest) is SKIPPED, not failing, in a checkout that has only `stack/`."""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
import v8f_mini_eval as EV  # noqa: E402
import v8f_t1_eval as T1  # noqa: E402

torch.set_num_threads(1)
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
A1 = PKG / "raw" / "2026-10-23-hist" / "a1"
REF_DATA_P, REF_MODEL_P = A1 / "pre_a1_v8f_mini_data.py.ref", A1 / "pre_a1_v8f_mini.py.ref"
MANIFEST = Path("C:/Users/Admin/box2_ship/logs/v8f_grid/manifest_train.jsonl")      # box 2's train manifest (4,572 clips), the one measure_hist.py reads
MANIFEST_SHA_PREFIX = "661f5d45"
HMAX = 16                                                                          # the trainer's auto hmax for M2b (a world-model arm)
DRAWS = [(a, a + k) for a in range(3) for k in range(5)]                           # the sampler's 15 equiprobable (a_t, a_s) pairs (ChunkedSampler.epoch)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)


def _load_ref(name, path):
    """A saved pre-amendment module, loaded under a private name. Registered in sys.modules BEFORE it executes (a dataclass with `from __future__ import annotations` looks its module up)."""
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        loader.exec_module(m)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return m


REF_D = _load_ref("v8f_mini_data_pre_a1", REF_DATA_P) if REF_DATA_P.exists() else None
REF_M = _load_ref("v8f_mini_pre_a1", REF_MODEL_P) if REF_MODEL_P.exists() else None
needs_ref_data = pytest.mark.skipif(REF_D is None, reason="raw/2026-10-23-hist/a1 (the saved pre-amendment modules) is not in this checkout")
needs_ref_model = pytest.mark.skipif(REF_M is None, reason="raw/2026-10-23-hist/a1 (the saved pre-amendment modules) is not in this checkout")


# ------------------------------------------------------------------------------------------------ frozen pre-amendment references
def _old_train_windows(J, clips, hist, stride=1):
    """VERBATIM the body of `train_windows` before Amendment 1 (v8f_mini_data.py, `for t in range(hist - 1, T - HORIZON, stride)`)."""
    ci_of = {c: i for i, c in enumerate(J["clip"].tolist())}
    out = []
    for c in clips:
        ci = ci_of[c]
        T = int(J["n"][ci])
        for t in range(hist - 1, T - D.HORIZON, stride):
            out.append((ci, t, 0, 0))
    return out


def _old_ages(t, a_t, a_s, hist, max_age_t=2, max_age_s=6):
    """VERBATIM the two clamp lines of `V8FMiniWindows.item` before Amendment 1 (`room = max(t - (self.hist - 1), 0)`)."""
    room = max(t - (hist - 1), 0)
    a_t = int(min(a_t, room, max_age_t))
    a_s = int(max(a_t, min(a_s, room, max_age_s)))
    return a_t, a_s


def _fake_J(lengths):
    return {"clip": np.array([f"clip{i:04d}" for i in range(len(lengths))]), "n": np.asarray(lengths, np.int64)}


def _same(x, y):
    """Exact equality of two item values (arrays bit for bit with NaN == NaN, scalars, nested dicts)."""
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x, y = np.asarray(x), np.asarray(y)
        return x.shape == y.shape and x.dtype == y.dtype and bool(np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))
    return bool(x == y) or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))


# ------------------------------------------------------------------------------------------------ the synthetic world
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_ws")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    store = D.ClipStore(root, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(root / "native_tracks.pt"))
    return {"root": root, "J": D.load_join(root / "join_train.npz"), "store": store, **w}


def _args(extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *extra])


def _built(a, J, **over):
    """(cfg, model, opt, info) as test_v8f_mini._built does, with the data-side read-backs the trainer would record (what the gate compares)."""
    cfg = TR.make_config(a)
    keep = {k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**cfg.__dict__, **keep, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    ws = cfg.window_start_effective
    info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True, "kin_mode": a.kin_mode, "p10_units": None,
            "window_start": ws, "window_t_min": ws, "ds_window_start": ws}
    return cfg, model, opt, info


# =============================================================================================================================================
# (a) THE DEFAULT IS BIT-IDENTICAL TO EVERY EARLIER ARM
# =============================================================================================================================================
def test_the_flag_defaults_to_absent_everywhere():
    """-1 = absent = hist - 1, at the parser, the config and the effective value -- including at hist 10, where a default of 3 would be invisible only at hist 4 (3 == hist - 1 there)."""
    assert TR.build_parser().parse_args(["--out", "x"]).window_start == -1
    assert M.V8FMiniConfig().window_start == -1 and M.V8FMiniConfig().window_start_effective == 3
    for h, want in ((4, 3), (10, 9), (1, 0), (7, 6)):
        a = _args(["--hist", str(h)])
        cfg = TR.make_config(a)
        assert (a.window_start, cfg.window_start, cfg.window_start_effective) == (-1, -1, want), h
        assert D.effective_window_start(h) == D.effective_window_start(h, None) == D.effective_window_start(h, -1) == want
    assert D.WINDOW_START_ABSENT == -1


@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("hist", [1, 2, 4, 7, 10])
def test_default_train_windows_equal_the_frozen_pre_amendment_function(hist, stride):
    J = _fake_J([20, 31, 33, 45, 60, 96, 101, 101, 7])           # includes clips shorter than hist + HORIZON (no window) and one-window clips
    clips = J["clip"].tolist()
    new = D.train_windows(None, J, clips, hist=hist, stride=stride)
    assert new == _old_train_windows(J, clips, hist, stride)
    assert new == D.train_windows(None, J, clips, hist=hist, stride=stride, start=None) == D.train_windows(None, J, clips, hist=hist, stride=stride, start=-1)
    assert new == D.train_windows(None, J, clips, hist=hist, stride=stride, start=hist - 1), "an explicit hist - 1 IS the default"


@pytest.mark.parametrize("hist", [4, 10])
def test_default_held_ages_equal_the_frozen_clamp_on_every_window_and_draw(world, hist):
    J, store = world["J"], world["store"]
    wins = D.train_windows(store, J, world["names"]["train"], hist=hist)
    ds = D.V8FMiniWindows(store, J, wins, hist=hist, hmax=HMAX)
    assert ds.window_start == hist - 1
    n = 0
    for ci, t, _, _ in wins:
        for a_t, a_s in DRAWS:
            it = ds.item(ci, t, a_t, a_s)
            assert (it["a_t"], it["a_s"]) == _old_ages(t, a_t, a_s, hist), (hist, t, a_t, a_s)
            n += 1
    assert n == len(wins) * 15 and n > 0


@needs_ref_data
@pytest.mark.parametrize("hist", [4, 10])
def test_default_items_and_windows_equal_the_saved_pre_amendment_module_key_for_key(world, hist):
    """The strongest form: the WHOLE saved module (raw/2026-10-23-hist/a1/pre_a1_v8f_mini_data.py.ref = the tip before this change) against the new one -- the window list, and every key of
    every item (features, frame_ok, labels, poses, ages) under all 15 draws. Also at t < hist - 1, which the training list never contains but the eval's item() calls can."""
    J, store = world["J"], world["store"]
    clips = world["names"]["train"]
    wins = D.train_windows(store, J, clips, hist=hist)
    assert wins == REF_D.train_windows(store, J, clips, hist=hist)
    new = D.V8FMiniWindows(store, J, wins, hist=hist, hmax=HMAX)
    old = REF_D.V8FMiniWindows(store, J, wins, hist=hist, hmax=HMAX)
    for ci, t, _, _ in wins[::2] + [(0, 3, 0, 0), (1, 0, 0, 0), (2, 5, 0, 0)]:
        for a_t, a_s in DRAWS[::2] + [(2, 6)]:
            assert _same(new.item(ci, t, a_t, a_s), old.item(ci, t, a_t, a_s)), (hist, ci, t, a_t, a_s)


@needs_ref_model
def test_default_model_is_the_saved_pre_amendment_model_bit_for_bit(world):
    """Same seed, same config fields: identical state_dict, identical forward_infer, identical built_levers except the three keys this amendment adds (hist, hmax, window_start)."""
    J, store = world["J"], world["store"]
    for hist in (4, 10):
        old_f = {f.name for f in dataclasses.fields(REF_M.V8FMiniConfig)}
        new_f = {f.name for f in dataclasses.fields(M.V8FMiniConfig)}
        assert new_f - old_f == {"window_start"} and old_f <= new_f
        kw = dict(TINY, hist=hist, hmax=HMAX, wm="b", kin_mode="yaw")
        mu, sd = M.cons_stats(J)
        torch.manual_seed(3)
        mn = M.V8FMini(M.V8FMiniConfig(**kw), M.P10Prior.synthetic(0), mu, sd)
        torch.manual_seed(3)
        mo = REF_M.V8FMini(REF_M.V8FMiniConfig(**kw), REF_M.P10Prior.synthetic(0), mu, sd)
        sn, so = mn.state_dict(), mo.state_dict()
        assert sn.keys() == so.keys() and all(torch.equal(sn[k], so[k]) for k in sn)
        ln, lo = mn.built_levers(), mo.built_levers()
        assert set(ln) - set(lo) == {"hist", "hmax", "window_start"} and set(lo) <= set(ln)
        assert {k: v for k, v in ln.items() if k in lo} == lo
        assert (ln["hist"], ln["hmax"], ln["window_start"]) == (hist, HMAX, hist - 1)
        wins = D.train_windows(store, J, world["names"]["train"], hist=hist)
        ds = D.V8FMiniWindows(store, J, wins, hist=hist, hmax=HMAX)
        b = D.collate([ds.item(ci, t, 1, 3) for ci, t, _, _ in wins[40:48]])
        mn.eval(), mo.eval()
        with torch.no_grad():
            on, oo = mn.forward_infer(b), mo.forward_infer(b)
        assert on.keys() == oo.keys() and all(torch.equal(on[k], oo[k]) for k in on if isinstance(on[k], torch.Tensor))


# =============================================================================================================================================
# (b) --hist 10 --window-start 3 IS B'S WINDOW LIST AND B'S HELD AGES
# =============================================================================================================================================
def test_hist_10_window_start_3_has_hist_4s_window_list_window_by_window(world):
    J, store = world["J"], world["store"]
    clips = world["names"]["train"]
    w4 = D.train_windows(store, J, clips, hist=4)
    w10 = D.train_windows(store, J, clips, hist=10, start=3)
    assert w10 == w4 and len(w4) == 6 * 27, "T = 60, HORIZON 30: t in [3, 30) per clip"
    w10_alone = D.train_windows(store, J, clips, hist=10)
    assert len(w10_alone) == 6 * 21 and w10_alone == [w for w in w4 if w[1] >= 9], "the trap: without the flag t = 3..8 are gone"
    # an explicit start inside [0, hist - 1] moves the list as asked
    assert D.train_windows(store, J, clips, hist=10, start=0)[0][1] == 0 and len(D.train_windows(store, J, clips, hist=10, start=0)) == 6 * 30


def test_hist_10_window_start_3_items_equal_hist_4s_items_key_for_key_under_every_draw(world):
    """hmax is 16 at both: the batch an item returns does not depend on hist, only the HELD AGES do (through the clamp's origin). So the two datasets must return EQUAL items --
    features, frame_ok, labels at the three tick frames, poses, ages -- for every window and every one of the 15 sampler draws."""
    J, store = world["J"], world["store"]
    w4 = D.train_windows(store, J, world["names"]["train"], hist=4)
    ds4 = D.V8FMiniWindows(store, J, w4, hist=4, hmax=HMAX)
    ds10 = D.V8FMiniWindows(store, J, w4, hist=10, hmax=HMAX, window_start=3)
    assert (ds4.window_start, ds10.window_start) == (3, 3)
    for ci, t, _, _ in w4:
        for a_t, a_s in DRAWS:
            assert _same(ds10.item(ci, t, a_t, a_s), ds4.item(ci, t, a_t, a_s)), (ci, t, a_t, a_s)


def test_without_the_flag_hist_10_held_ages_differ_from_hist_4s_exactly_where_the_clamp_origin_differs(world):
    """THE CONTROL: the equality above is not vacuous. At hist 10 without the flag the origin is 9, so ages are clamped harder for t < 15 (room 0..5 at hist 4 vs 0 at hist 10 for t <= 9)."""
    J, store = world["J"], world["store"]
    w4 = D.train_windows(store, J, world["names"]["train"], hist=4)
    ds4 = D.V8FMiniWindows(store, J, w4, hist=4, hmax=HMAX)
    ds10 = D.V8FMiniWindows(store, J, w4, hist=10, hmax=HMAX)
    differing = set()
    for ci, t, _, _ in w4:
        for a_t, a_s in DRAWS:
            i4, i10 = ds4.item(ci, t, a_t, a_s), ds10.item(ci, t, a_t, a_s)
            if (i4["a_t"], i4["a_s"]) != (i10["a_t"], i10["a_s"]):
                differing.add(t)
                assert (i10["a_t"], i10["a_s"]) == _old_ages(t, a_t, a_s, 10)
    assert differing and min(differing) == 4 and max(differing) < 15, sorted(differing)       # t = 3 has room 0 at both; t >= 15 has room >= 6 at hist 10 too


def test_the_sampler_over_equal_window_lists_draws_equal_batches(world):
    """Equal window lists + equal seed => the SAME epoch order and draws (ChunkedSampler is a function of the list): the held-age stream H trains on is B's."""
    J, store = world["J"], world["store"]
    clips = world["names"]["train"]
    s4 = D.ChunkedSampler(D.train_windows(store, J, clips, hist=4), 8, seed=5, chunk=4)
    s10 = D.ChunkedSampler(D.train_windows(store, J, clips, hist=10, start=3), 8, seed=5, chunk=4)
    assert list(s4.epoch(0)) == list(s10.epoch(0)) and s4.n_batches() == s10.n_batches()


# =============================================================================================================================================
# validation: 0 <= window_start <= hist - 1 when set, else REFUSED
# =============================================================================================================================================
@pytest.mark.parametrize("hist, ws", [(10, 10), (10, 11), (10, -2), (4, 4), (4, -5)])
def test_a_window_start_outside_0_to_hist_minus_1_is_refused_at_every_layer(world, hist, ws):
    J, store = world["J"], world["store"]
    with pytest.raises(ValueError, match="window_start"):
        D.effective_window_start(hist, ws)
    with pytest.raises(ValueError, match="window_start"):
        D.train_windows(store, J, world["names"]["train"], hist=hist, start=ws)
    with pytest.raises(ValueError, match="window_start"):
        D.V8FMiniWindows(store, J, [], hist=hist, hmax=16, window_start=ws)
    with pytest.raises(ValueError, match="window_start"):
        M.V8FMiniConfig(hist=hist, hmax=16, window_start=ws)
    with pytest.raises(ValueError, match="window_start"):
        TR.make_config(_args(["--hist", str(hist), "--window-start", str(ws)]))


@pytest.mark.parametrize("hist, ws", [(10, 0), (10, 3), (10, 9), (4, 0), (4, 3)])
def test_a_window_start_inside_the_range_is_accepted_and_effective(world, hist, ws):
    J, store = world["J"], world["store"]
    cfg = TR.make_config(_args(["--hist", str(hist), "--window-start", str(ws)]))
    assert (cfg.window_start, cfg.window_start_effective) == (ws, ws)
    assert D.effective_window_start(hist, ws) == ws
    assert D.V8FMiniWindows(store, J, [], hist=hist, hmax=16, window_start=ws).window_start == ws
    assert min(w[1] for w in D.train_windows(store, J, world["names"]["train"], hist=hist, start=ws)) == ws


def test_the_trainer_refuses_an_out_of_range_window_start_before_doing_anything(tmp_path):
    with pytest.raises(ValueError, match="window_start"):
        TR.main(["--synthetic", "--arm", "M2b", "--hist", "10", "--window-start", "10", "--out", str(tmp_path / "o"), "--d", "32", "--n-q", "2", "--kin-mode", "yaw",
                 "--selfcheck-only"])


# =============================================================================================================================================
# (c) THE COUNTS ON BOX 2'S REAL TRAIN MANIFEST
# =============================================================================================================================================
def _manifest_clips():
    """The trainer's own clip set for the M2b runs: the manifest's 4,572 clips minus `--exclude 540:600` in sha12 order (v8f_mini_train.py build_data) = 4,512."""
    ents = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                ents[d["clip"]] = int(d["n_frames"])
    excluded = set(sorted(ents, key=D.sha12)[540:600])
    clips = [c for c in ents if c not in excluded]
    J = {"clip": np.array(list(ents)), "n": np.array([ents[c] for c in ents], np.int64)}
    return clips, J


def _budget(n, batch=64, mult=4):
    s = math.ceil(n / batch)                        # v8f_mini_train.py: S = ceil(n_windows_all_clips / batch); total = S x steps_mult (the formula is proven against the trainer below)
    return s, mult * s


@pytest.mark.skipif(not MANIFEST.exists(), reason="box 2's train manifest is not on this disk")
def test_window_counts_and_step_budgets_on_box_2s_real_manifest():
    h = hashlib.sha256()
    with open(MANIFEST, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    assert h.hexdigest().startswith(MANIFEST_SHA_PREFIX), "this is not the manifest BUILD.md measured"
    clips, J = _manifest_clips()
    assert len(clips) == 4512
    w4 = D.train_windows(None, J, clips, hist=4)
    w10 = D.train_windows(None, J, clips, hist=10)
    w10s = D.train_windows(None, J, clips, hist=10, start=3)
    assert (len(w4), _budget(len(w4))) == (306566, (4791, 19164)), "B: the banked M2b (summary.json n_windows 306,566, 19,164 steps)"
    assert (len(w10s), _budget(len(w10s))) == (306566, (4791, 19164)), "H = B + --hist 10 --window-start 3: B's windows and B's budget"
    assert w10s == w4, "...and B's window LIST, window by window"
    assert (len(w10), _budget(len(w10))) == (279494, (4368, 17472)), "the trap: --hist 10 alone loses t = 3..8 of every clip (27,072 windows) and 1,692 steps"
    assert len(w4) - len(w10) == 6 * 4512 and w10 == [w for w in w4 if w[1] >= 9]


def test_the_budget_formula_is_the_trainers_own_on_the_synthetic_world(tmp_path):
    """`_budget` restates v8f_mini_train.py's S and total; prove it equals the trainer's selfcheck at hist 4, hist 10 alone and hist 10 + window-start 3 (no --steps override)."""
    sc = {}
    for tag, extra in (("h4", ["--hist", "4"]), ("h10", ["--hist", "10"]), ("h10ws3", ["--hist", "10", "--window-start", "3"])):
        out = tmp_path / tag
        assert TR.main(["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--arm", "M2b", "--steps-mult", "4", "--out", str(out), "--selfcheck-only",
                        *extra]) == 0
        sc[tag] = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    for tag, n in (("h4", 6 * 27), ("h10", 6 * 21), ("h10ws3", 6 * 27)):
        assert sc[tag]["info"]["n_windows_all_clips"] == sc[tag]["info"]["n_windows"] == n
        assert (sc[tag]["S_steps_per_epoch_100pct"], sc[tag]["steps_total"]) == _budget(n, batch=8), tag
    assert sc["h10ws3"]["steps_total"] == sc["h4"]["steps_total"] > sc["h10"]["steps_total"]
    assert sc["h10ws3"]["info"]["n_windows_all_clips"] == sc["h4"]["info"]["n_windows_all_clips"]


# =============================================================================================================================================
# (d) A WINDOW AT t = 3 WITH hist 10: SIX PADDED FRAMES, FOUR REAL ONES
# =============================================================================================================================================
def _t_batch(world, t, hist, ws, ages=(2, 6)):
    J, store = world["J"], world["store"]
    ds = D.V8FMiniWindows(store, J, [], hist=hist, hmax=HMAX, window_start=ws)
    items = [ds.item(ci, t, *ages) for ci in range(6)]
    return D.collate(items), J


def _spy_scene(model):
    """Every (scene tokens, pad mask, last index) the model's three layers gather, in call order."""
    cap = []
    orig = model._scene

    def scene(adapt, ok, last):
        g, pad = orig(adapt, ok, last)
        cap.append((g.detach().clone(), pad.detach().clone(), last.detach().clone()))
        return g, pad
    model._scene = scene
    return cap


def test_at_t_3_with_hist_10_and_window_start_3_the_gather_masks_six_frames_and_reads_four(world):
    a = _args(["--hist", "10", "--window-start", "3"])
    b, J = _t_batch(world, 3, 10, 3)
    assert b["a_t"].tolist() == [0] * 6 and b["a_s"].tolist() == [0] * 6, "room = max(3 - 3, 0) = 0: no held age can reach a frame before the clip"
    assert tuple(b["frame_ok"].shape) == (6, HMAX) and not bool(b["frame_ok"][:, :12].any()) and bool(b["frame_ok"][:, 12:].all()), "frames t-15..t = -12..3: four real ones"
    cfg, model, opt, info = _built(a, J, hist=10, hmax=HMAX)
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    with torch.no_grad():
        model.head_o[2].weight.normal_(0.0, 0.05)          # the residual head is ZERO-INIT (traj30 == the frozen prior whatever the scene): make the trajectory scene-dependent, or this probe is blind
    cap = _spy_scene(model)
    model.eval()
    with torch.no_grad():
        out = model.forward_infer(b)
    assert len(cap) == 3, "S, T and O each gather their scene"
    for g, pad, last in cap:
        pf = pad.reshape(6, 10, cfg.n_tok)
        assert (pf == pf[:, :, :1]).all(), "a frame is padded as a whole"
        assert pf[:, :, 0].sum(1).tolist() == [6] * 6, "exactly six padded frames per window ..."
        assert pf[:, :6, 0].all() and not pf[:, 6:, 0].any(), "... the six OLDEST ones; the four newest are real"
    assert all(torch.isfinite(v).all() for v in out.values() if isinstance(v, torch.Tensor)), "forward_infer is finite"
    # the padded frames are MASKED, not merely flagged: scrambling every feature before the clip's start moves nothing
    b2 = dict(b, feats=b["feats"].clone())
    b2["feats"][:, :12] = torch.randn_like(b2["feats"][:, :12]) * 50.0
    with torch.no_grad():
        out2 = model.forward_infer(b2)
    checked = [k for k, v in out.items() if isinstance(v, torch.Tensor) and k != "adapt"]       # `adapt` is the per-frame ENCODER output (all hmax frames, before any mask): not a prediction
    assert {"traj30", "traj60"} <= set(checked)
    for k in checked:
        assert torch.equal(out[k], out2[k]), f"{k}: a padded frame changed the output"
    # CONTROL: scrambling a REAL frame does move the output (the probe can see a leak)
    b3 = dict(b, feats=b["feats"].clone())
    b3["feats"][:, 14] = torch.randn_like(b3["feats"][:, 14]) * 50.0
    with torch.no_grad():
        out3 = model.forward_infer(b3)
    assert not torch.equal(out["traj30"], out3["traj30"])
    # the loss is finite and every parameter group trains on a padded-history window
    model.train()
    L = model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(0))
    assert all(torch.isfinite(v) for v in L["terms"].values())
    L["terms"]["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("t", [3, 4, 7, 8, 9, 12])
def test_real_frames_per_layer_are_min_tick_plus_1_and_hist_10(world, t):
    """S reads the tick t - a_s, T the tick t - a_t, O NOW: each gathers the 10 frames ending there, of which min(tick + 1, 10) are real. At window-start 3 the held ages are clamped so that
    no tick lies before t = 3, i.e. every layer has >= 4 real frames (the padded ones are the six oldest at t = 3)."""
    a = _args(["--hist", "10", "--window-start", "3"])
    b, J = _t_batch(world, t, 10, 3)
    cfg, model, opt, info = _built(a, J, hist=10, hmax=HMAX)
    cap = _spy_scene(model)
    model.eval()
    with torch.no_grad():
        model.forward_infer(b)
    assert len(cap) == 3
    for g, pad, last in cap:
        tick = t - (HMAX - 1 - last)                                         # the clip frame the layer's newest scene frame is
        assert (tick >= 3).all(), "no held tick before the window start"
        real = (~pad.reshape(6, 10, cfg.n_tok)[:, :, 0]).sum(1)
        assert real.tolist() == torch.clamp(tick + 1, max=10).tolist(), (t, real.tolist(), tick.tolist())
        assert int(real.min()) >= 4


# =============================================================================================================================================
# (e) G-DVB: A DECLARED --window-start THAT THE MODEL / THE DATA DID NOT BUILD IS REFUSED, HOP BY HOP
# =============================================================================================================================================
def test_the_flag_has_its_declared_vs_built_entry_and_a_flag_without_one_is_refused(world, monkeypatch):
    assert "window_start" in TR.FLAG_CHECKS
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert "window_start" in dests and dests <= set(TR.FLAG_CHECKS) and set(TR.FLAG_CHECKS) <= dests
    a = _args(["--hist", "10", "--window-start", "3"])
    cfg, model, opt, info = _built(a, world["J"], hist=10, hmax=HMAX)
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != "window_start"})
    assert any("window_start" in m and "without a declared-vs-built entry" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))


def test_gdvb_passes_a_built_window_start_and_the_default_and_nothing_else_moves(world):
    J = world["J"]
    for argv, hist in ((["--hist", "10", "--window-start", "3"], 10), (["--hist", "10", "--window-start", "0"], 10), (["--hist", "10"], 10), ([], 4), (["--hist", "4", "--window-start", "3"], 4)):
        a = _args(argv)
        cfg, model, opt, info = _built(a, J, hist=hist, hmax=HMAX)
        assert TR.declared_vs_built(a, cfg, model, opt, info) == [], argv


def test_gdvb_refuses_a_model_that_did_not_build_the_declared_window_start(world):
    """Hop 1: argv -> make_config -> the BUILT model's levers. Declared 3, the model built the default (cfg.window_start -1 = 9)."""
    a = _args(["--hist", "10", "--window-start", "3"])
    cfg, model, opt, info = _built(a, world["J"], hist=10, hmax=HMAX, window_start=-1)
    info = dict(info, window_t_min=3, ds_window_start=3)                  # the data side is right: ONLY the model's lever is wrong
    bad = TR.declared_vs_built(a, cfg, model, opt, info)
    assert bad and all("window_start" in m for m in bad) and any("lever window_start: declared 3 but built 9" in m for m in bad), bad


def test_gdvb_refuses_a_window_list_that_does_not_start_at_the_declared_step(world):
    """Hop 2: the flag -> train_windows. Declared 3, the list the data built starts at 9 (the trap): measured t_min 9."""
    a = _args(["--hist", "10", "--window-start", "3"])
    cfg, model, opt, info = _built(a, world["J"], hist=10, hmax=HMAX)
    bad = TR.declared_vs_built(a, cfg, model, opt, dict(info, window_t_min=9))
    assert bad and all("window_start" in m for m in bad) and any("first t of the training window list was built as 9" in m for m in bad), bad


def test_gdvb_refuses_a_dataset_whose_held_age_clamp_has_another_origin(world):
    """Hop 3: the flag -> V8FMiniWindows. Declared 3, the dataset clamps from hist - 1 = 9 ('room still computed from hist')."""
    a = _args(["--hist", "10", "--window-start", "3"])
    cfg, model, opt, info = _built(a, world["J"], hist=10, hmax=HMAX)
    bad = TR.declared_vs_built(a, cfg, model, opt, dict(info, ds_window_start=9))
    assert bad and all("window_start" in m for m in bad) and any("held-age clamp origin was built as 9" in m for m in bad), bad


@pytest.mark.parametrize("missing", ["window_t_min", "ds_window_start"])
def test_gdvb_refuses_a_declared_window_start_with_no_read_back_from_the_data(world, missing):
    a = _args(["--hist", "10", "--window-start", "3"])
    cfg, model, opt, info = _built(a, world["J"], hist=10, hmax=HMAX)
    info = {k: v for k, v in info.items() if k != missing}
    bad = TR.declared_vs_built(a, cfg, model, opt, info)
    what = {"window_t_min": "first t of the training window list", "ds_window_start": "held-age clamp origin"}[missing]
    assert any(what in m and "was built as None" in m for m in bad), bad


def test_gdvb_does_not_require_the_read_back_when_the_flag_is_absent_but_checks_it_when_present(world):
    a = _args(["--hist", "10"])
    cfg, model, opt, info = _built(a, world["J"], hist=10, hmax=HMAX)
    bare = {k: v for k, v in info.items() if k not in ("window_start", "window_t_min", "ds_window_start")}
    assert TR.declared_vs_built(a, cfg, model, opt, bare) == [], "an info dict that predates the keys is not a lie"
    assert any("window_start" in m for m in TR.declared_vs_built(a, cfg, model, opt, dict(bare, window_t_min=3))), "...but a present key must equal hist - 1 = 9"


def test_gdvb_refuses_a_built_hist_or_hmax_that_is_not_the_declared_one(world):
    a = _args(["--hist", "10", "--window-start", "3"])
    cfg, model, opt, info = _built(a, world["J"], hist=4, hmax=HMAX, window_start=3)
    bad = TR.declared_vs_built(a, cfg, model, opt, info)
    assert any("lever hist: declared 10 but built 4" in m for m in bad), bad
    cfg, model, opt, info = _built(a, world["J"], hist=10, hmax=HMAX, window_start=3)
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    model.cfg = dataclasses.replace(model.cfg, hmax=HMAX + 4)                  # a model whose built batch depth is not the config the gate was handed
    assert any("lever hmax: declared 16 but built 20" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))


@pytest.mark.parametrize("argv, want", [(["--hist", "4"], (4, HMAX, 3)), (["--hist", "10"], (10, HMAX, 9)), (["--hist", "10", "--window-start", "3"], (10, HMAX, 3)),
                                        (["--hist", "10", "--window-start", "0"], (10, HMAX, 0))])
def test_built_levers_carry_hist_hmax_and_the_effective_window_start(world, argv, want):
    a = _args(argv)
    cfg, model, opt, info = _built(a, world["J"], hist=a.hist, hmax=HMAX)
    lv = model.built_levers()
    assert (lv["hist"], lv["hmax"], lv["window_start"]) == want
    assert lv["hist"] == model.time.shape[0], "hist is read from the BUILT per-frame time table, not echoed from the config"


# ---- the trainer, end to end -------------------------------------------------------------------------------------------------------------------
COMMON = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--log-every", "2", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5",
          "--arm", "M2b"]


def _train(out, extra, steps=4):
    """A real `TR.main` run on the synthetic world, with spies on what the trainer hands `train_windows` and `V8FMiniWindows`."""
    cap = {"tw": [], "ds": []}
    tw0, init0 = D.train_windows, D.V8FMiniWindows.__init__

    def tw(*a, **k):
        cap["tw"].append(dict(k))
        return tw0(*a, **k)

    def init(self, *a, **k):
        init0(self, *a, **k)
        cap["ds"].append((self.hist, self.hmax, self.window_start))
    D.train_windows, D.V8FMiniWindows.__init__ = tw, init
    try:
        assert TR.main(["--out", str(out), *COMMON, "--steps", str(steps), *extra]) == 0
    finally:
        D.train_windows, D.V8FMiniWindows.__init__ = tw0, init0
    out = Path(out)
    return {"cap": cap, "out": out, "ck": torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False),
            "sc": json.loads((out / "selfcheck.json").read_text(encoding="utf-8")), "sm": json.loads((out / "summary.json").read_text(encoding="utf-8")),
            "argv": json.loads((out / "argv.json").read_text(encoding="utf-8"))["argv"]}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("ws_runs")
    return {"h4": _train(base / "h4", ["--hist", "4"]), "h10": _train(base / "h10", ["--hist", "10"]),
            "h10ws3": _train(base / "h10ws3", ["--hist", "10", "--window-start", "3"])}


def test_trainer_argv_reaches_train_windows_the_dataset_the_checkpoint_and_the_records(runs):
    r = runs["h10ws3"]
    assert r["argv"].count("--window-start") == 1 and r["argv"][r["argv"].index("--window-start") + 1] == "3"
    assert r["cap"]["tw"] and all(k.get("start") == 3 and k.get("hist") == 10 for k in r["cap"]["tw"]), "both train_windows calls (all clips, selected clips) get start=3"
    assert r["cap"]["ds"] == [(10, HMAX, 3)], "the dataset the trainer trains on clamps from 3"
    assert r["ck"]["cfg"]["window_start"] == 3 and r["ck"]["cfg"]["hist"] == 10, "the checkpoint records the lever"
    info, lv = r["sc"]["info"], r["sc"]["levers"]
    assert (info["window_start"], info["window_t_min"], info["ds_window_start"]) == (3, 3, 3)
    assert (lv["hist"], lv["hmax"], lv["window_start"]) == (10, HMAX, 3) and r["sm"]["levers"]["window_start"] == 3
    assert r["sc"]["declared_vs_built"] == "PASS" and r["sm"]["done"] is True


def test_trainer_h10_window_start_3_run_has_hist_4s_windows_and_budget_and_only_the_time_table_moves(runs):
    r10, r4, r10a = runs["h10ws3"], runs["h4"], runs["h10"]
    assert r10["sc"]["info"]["n_windows"] == r10["sc"]["info"]["n_windows_all_clips"] == r4["sc"]["info"]["n_windows"] == 6 * 27
    assert r10a["sc"]["info"]["n_windows"] == 6 * 21, "the trap, still true without the flag"
    assert r10["sc"]["S_steps_per_epoch_100pct"] == r4["sc"]["S_steps_per_epoch_100pct"] and r10["sc"]["steps_total"] == r4["sc"]["steps_total"]
    lv10, lv4 = r10["sm"]["levers"], r4["sm"]["levers"]
    assert lv10["n_params_trainable"] - lv4["n_params_trainable"] == 6 * 32 == lv10["params"]["adapter"] - lv4["params"]["adapter"]
    skip = {"params", "n_params_trainable", "residual_zero_init", "hist"}
    assert {k: v for k, v in lv10.items() if k not in skip} == {k: v for k, v in lv4.items() if k not in skip}, "window_start (3 == 3) and everything else equal B's"
    assert (lv4["hist"], lv4["window_start"]) == (4, 3) and (runs["h10"]["sm"]["levers"]["hist"], runs["h10"]["sm"]["levers"]["window_start"]) == (10, 9)


def test_trainer_default_run_is_unchanged_by_the_flag_existing(runs):
    """Without the flag the trainer builds exactly what it built before: window_start = hist - 1 recorded, windows from hist - 1, dataset clamp from hist - 1."""
    for tag, hist in (("h4", 4), ("h10", 10)):
        r = runs[tag]
        assert "--window-start" not in r["argv"]
        assert r["cap"]["tw"] and all(k.get("start") == -1 for k in r["cap"]["tw"])
        assert r["cap"]["ds"] == [(hist, HMAX, hist - 1)]
        assert r["ck"]["cfg"]["window_start"] == -1
        assert (r["sc"]["info"]["window_t_min"], r["sc"]["info"]["ds_window_start"]) == (hist - 1, hist - 1)


def _selfcheck_argv(tmp_path, tag, extra):
    return ["--out", str(tmp_path / tag), *COMMON, "--selfcheck-only", *extra]


def test_the_trainer_refuses_a_train_windows_that_drops_the_flag(tmp_path, monkeypatch):
    tw0 = D.train_windows
    monkeypatch.setattr(D, "train_windows", lambda *a, **k: tw0(*a, **{kk: v for kk, v in k.items() if kk != "start"}))
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ.*window_start 3: declared 3 but the first t of the training window list was built as 9"):
        TR.main(_selfcheck_argv(tmp_path, "a", ["--hist", "10", "--window-start", "3"]))


def test_the_trainer_refuses_a_dataset_that_drops_the_flag(tmp_path, monkeypatch):
    init0 = D.V8FMiniWindows.__init__
    monkeypatch.setattr(D.V8FMiniWindows, "__init__", lambda self, *a, **k: init0(self, *a, **{kk: v for kk, v in k.items() if kk != "window_start"}))
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ.*held-age clamp origin was built as 9"):
        TR.main(_selfcheck_argv(tmp_path, "b", ["--hist", "10", "--window-start", "3"]))


def test_the_trainer_refuses_a_make_config_that_drops_the_flag(tmp_path, monkeypatch):
    mc0 = TR.make_config

    def mc(a):
        b = argparse.Namespace(**vars(a))
        b.window_start = -1                                  # the config is built WITHOUT the flag; the gate still holds the declared argv
        return mc0(b)
    monkeypatch.setattr(TR, "make_config", mc)
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ.*lever window_start: declared 3 but built 9"):
        TR.main(_selfcheck_argv(tmp_path, "c", ["--hist", "10", "--window-start", "3"]))


def test_the_trainer_default_still_launches_without_the_flag(tmp_path):
    """The gate must not refuse the arms that never heard of the flag (hist 4 and hist 10 alone)."""
    for tag, extra in (("d4", ["--hist", "4"]), ("d10", ["--hist", "10"])):
        assert TR.main(_selfcheck_argv(tmp_path, tag, extra)) == 0


# =============================================================================================================================================
# (g) THE EVALUATORS (Amendment 1, eval-side fix): the checkpoint's window_start must reach the dataset the EVAL builds
# =============================================================================================================================================
# The class is "built, tested, and unreachable from its caller". `V8FMiniWindows(window_start=...)` was correct and tested; `v8f_mini_eval.py` and `v8f_t1_eval.py` simply never passed it, so an H
# checkpoint (cfg.window_start == 3) was evaluated with the clamp origin 9: held ages near a clip's start (t < 15) were clamped harder than H trained. Below, the REAL `EV.main run` / `T1.main run`
# are driven end to end on the synthetic eval world, with a spy on `V8FMiniWindows.item` that records the origin of the dataset the evaluator built and the ages every call resolved to.


def _eff_ages(t, a_t, a_s, ws, max_age_t=2, max_age_s=6):
    """The clamp of `V8FMiniWindows.item`, re-derived from the origin `ws` (so a test can predict where two origins differ)."""
    room = max(t - ws, 0)
    a_t = int(min(a_t, room, max_age_t))
    return a_t, int(max(a_t, min(a_s, room, max_age_s)))


class _ItemSpy:
    """Records every `V8FMiniWindows.item` call: the dataset's clamp origin and hist, the window, the ages asked for, the ages resolved, the item itself; with `ref` (a saved pre-amendment data
    module) it also builds the PRE-AMENDMENT dataset on the SAME store / join at call time and records whether the two items are identical key for key."""

    def __init__(self, ref=None):
        self.ref, self.calls, self._refds = ref, [], {}

    def __enter__(self):
        spy, item0 = self, D.V8FMiniWindows.item
        self.item0 = item0

        def item(ds, ci, t, a_t=0, a_s=0):
            it = item0(ds, ci, t, a_t, a_s)
            rec = {"origin": ds.window_start, "hist": ds.hist, "ci": ci, "t": t, "req": (int(a_t), int(a_s)), "eff": (it["a_t"], it["a_s"]), "it": it}
            if spy.ref is not None:
                old = spy._refds.get(id(ds))
                if old is None:
                    old = spy._refds[id(ds)] = spy.ref.V8FMiniWindows(ds.store, ds.J, [], hist=ds.hist, hmax=ds.hmax, max_age_t=ds.max_age_t, max_age_s=ds.max_age_s)
                rec["same_as_pre_amendment"] = _same(it, old.item(ci, t, a_t, a_s))
            spy.calls.append(rec)
            return it
        D.V8FMiniWindows.item = item
        return self

    def __exit__(self, *exc):
        D.V8FMiniWindows.item = self.item0


def _eval_argv(root, ck, tag, out, script):
    root = Path(root)
    if script == "mini":
        return ["run", "--ckpt", str(ck), "--tag", tag, "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-kin", str(root / "eval_kin.npz"),
                "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(out), "--n-boot", "20", "--batch", "16", "--lru", "4", "--tracks", str(root / "native_tracks.pt")]
    return ["run", "--ckpt", str(ck), "--tag", tag, "--tracks", str(root / "native_tracks.pt"), "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"),
            "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(out), "--replan-steps", "5", "--batch", "6", "--max-windows", "12", "--n-boot", "20"]


def _eval_run(root, ck, tag, tmp, script, ref=None, module=None):
    """The REAL evaluator CLI (`module` = EV / T1 unless a saved one is given), with the item spy; returns (calls, output dir)."""
    mod = module or (EV if script == "mini" else T1)
    out = Path(tmp) / f"{script}_{tag}"
    with _ItemSpy(ref) as spy:
        assert mod.main(_eval_argv(root, ck, tag, out, script)) == 0
    return spy.calls, out


@pytest.fixture(scope="module")
def evals(runs, world, tmp_path_factory):
    tmp, root = tmp_path_factory.mktemp("ws_evals"), world["root"]
    out = {}
    for tag in ("h4", "h10", "h10ws3"):
        ck = runs[tag]["out"] / "ckpt_last.pt"
        out[tag] = {s: _eval_run(root, ck, tag, tmp, s)[0] for s in ("mini", "t1")}
    for hist, tag in ((4, "old4"), (10, "old10")):                      # a PRE-AMENDMENT checkpoint: its saved cfg has no `window_start` key at all
        ck = torch.load(runs[f"h{hist}"]["out"] / "ckpt_last.pt", map_location="cpu", weights_only=False)
        ck["cfg"] = {k: v for k, v in ck["cfg"].items() if k != "window_start"}
        assert "window_start" not in ck["cfg"] and ck["cfg"]["hist"] == hist
        p = tmp / f"{tag}.pt"
        torch.save(ck, p)
        out[tag] = {"ck": p}
        out[tag]["mini"], out[tag]["mini_dir"] = _eval_run(root, p, tag, tmp, "mini", ref=REF_D)
        if hist == 10:
            out[tag]["t1"], _ = _eval_run(root, p, tag, tmp, "t1", ref=REF_D)
    return out


@pytest.mark.parametrize("script", ["mini", "t1"])
def test_eval_of_an_H_checkpoint_resolves_the_same_held_ages_as_a_hist_4_checkpoint_at_every_window(evals, script):
    """hist 10 + window-start 3 vs hist 4, through the REAL evaluator: the dataset's clamp origin is the checkpoint's (3 for both) and every call resolves to the same ages; for the
    mini evaluator (default `--ages cycle`) every item, all keys, is equal too (the model differs, the items do not)."""
    h4, h10s = evals["h4"][script], evals["h10ws3"][script]
    assert len(h4) == len(h10s) > 0
    assert {c["origin"] for c in h4} == {3} and {c["origin"] for c in h10s} == {3}, "the origin the evaluator built is the CHECKPOINT's"
    assert {c["hist"] for c in h4} == {4} and {c["hist"] for c in h10s} == {10}
    for x, y in zip(h4, h10s):
        assert (x["ci"], x["t"], x["req"]) == (y["ci"], y["t"], y["req"])
        assert x["eff"] == y["eff"], (script, x["t"], x["req"], x["eff"], y["eff"])
        if script == "mini":
            assert _same(x["it"], y["it"]), (x["ci"], x["t"], x["req"])


def test_the_mini_evaluator_applies_the_cycle_rule_by_default_and_the_grid_reaches_the_early_windows(evals):
    """The identity above is not vacuous: the default `--ages cycle` really asks for ages 0..2 / 0..6 at windows with t < 15 (t = 3, 7, 11), where the two clamp origins disagree."""
    calls = evals["h10ws3"]["mini"]
    n = len({(c["ci"], c["t"]) for c in calls[:14]})
    assert n == 14, "2 synthetic eval clips x 7 windows (t = 3, 7, ..., 27)"
    for i, c in enumerate(calls[:14]):                      # the main pass is the first 14 calls, in window order (v8f_mini_eval.py:287-297)
        a_t = i % 3
        assert c["req"] == (a_t, a_t + (i // 3) % 5), i
    early = [c for c in calls[:14] if c["t"] < 15]
    assert {c["t"] for c in early} == {3, 7, 11}
    assert any(_eff_ages(c["t"], *c["req"], 3) != _eff_ages(c["t"], *c["req"], 9) for c in early), "the two origins resolve DIFFERENT ages on some early window"


@pytest.mark.parametrize("script", ["mini", "t1"])
def test_CONTROL_without_the_flag_hist_10_resolves_different_ages_exactly_where_the_origins_differ(evals, script):
    """hist 10 trained WITHOUT the flag has origin 9, and the evaluator must use 9 for it (it is that checkpoint's own training); its ages differ from the H checkpoint's exactly where the
    clamp arithmetic says, and only at t < 15. This is what the unfixed evaluator did to an H checkpoint."""
    a, b = evals["h10"][script], evals["h10ws3"][script]
    assert len(a) == len(b) > 0
    assert {c["origin"] for c in a} == {9} and {c["origin"] for c in b} == {3}
    differ = [(x, y) for x, y in zip(a, b) if x["eff"] != y["eff"]]
    assert differ, "the control must differ somewhere or the identity above proves nothing"
    assert all(x["t"] < 15 for x, y in differ)
    for x, y in zip(a, b):
        assert x["eff"] == _eff_ages(x["t"], *x["req"], 9) and y["eff"] == _eff_ages(y["t"], *y["req"], 3)


@pytest.mark.parametrize("tag, script", [("old4", "mini"), ("old10", "mini"), ("old10", "t1")])
def test_an_old_checkpoint_cfg_without_the_key_loads_with_the_default_and_resolves_todays_ages(evals, tag, script):
    """Every pre-amendment checkpoint (M2b, GRID, PGT: no `window_start` in the saved cfg) loads with the default and is evaluated with the clamp origin hist - 1, the ages of the frozen
    pre-amendment clamp at every call."""
    ck = torch.load(evals[tag]["ck"], map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    hist = ck["cfg"]["hist"]
    assert (cfg.window_start, cfg.window_start_effective) == (-1, hist - 1)
    calls = evals[tag][script]
    assert calls and {c["origin"] for c in calls} == {hist - 1} and {c["hist"] for c in calls} == {hist}
    for c in calls:
        assert c["eff"] == _old_ages(c["t"], *c["req"], hist), (tag, script, c["t"], c["req"])


@needs_ref_data
@pytest.mark.parametrize("tag, script", [("old4", "mini"), ("old10", "mini"), ("old10", "t1")])
def test_an_old_checkpoint_gets_the_SAME_ITEMS_from_the_evaluator_as_from_the_saved_pre_amendment_dataset(evals, tag, script):
    """Every key of every item the evaluator's dataset returned, against the saved pre-amendment data module (which has no window_start at all) built on the same store and join at the same call."""
    calls = evals[tag][script]
    assert calls and all(c["same_as_pre_amendment"] for c in calls), [(c["t"], c["req"]) for c in calls if not c["same_as_pre_amendment"]][:5]


@needs_ref_data
@pytest.mark.skipif(not (A1 / "pre_a2fix_v8f_mini_eval.py.ref").exists(), reason="the saved pre-fix evaluator is not in this checkout")
@pytest.mark.parametrize("tag", ["old4", "old10"])
def test_an_old_checkpoint_scores_bit_for_bit_the_same_through_the_fixed_and_the_saved_pre_fix_evaluator(evals, world, tmp_path, tag):
    """End to end, outputs and all: the SAVED pre-fix `v8f_mini_eval.py` (the tip before this change) and the fixed one write identical `eval_<tag>.npz` arrays and an identical `raw` block for a
    checkpoint whose cfg lacks the key."""
    ref_ev = _load_ref("v8f_mini_eval_pre_a2fix", A1 / "pre_a2fix_v8f_mini_eval.py.ref")
    _, out_old = _eval_run(world["root"], evals[tag]["ck"], tag, tmp_path, "mini", module=ref_ev)
    out_new = evals[tag]["mini_dir"]
    za, zb = np.load(out_old / f"eval_{tag}.npz"), np.load(out_new / f"eval_{tag}.npz")
    assert sorted(za.files) == sorted(zb.files) and za.files
    for k in za.files:
        assert np.array_equal(za[k], zb[k], equal_nan=za[k].dtype.kind == "f"), k
    ja = json.loads((out_old / f"eval_{tag}.json").read_text(encoding="utf-8"))["raw"]
    jb = json.loads((out_new / f"eval_{tag}.json").read_text(encoding="utf-8"))["raw"]
    assert json.dumps(ja, sort_keys=True) == json.dumps(jb, sort_keys=True)


def _constructions_without_window_start(src, name="<src>"):
    """(name, line) of every call of `V8FMiniWindows` (as a bare name or an attribute) that does not pass `window_start=` explicitly. A `**kwargs` splat does not count: it cannot be verified."""
    bad, n = [], 0
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr == "V8FMiniWindows") or (isinstance(f, ast.Name) and f.id == "V8FMiniWindows"):
                n += 1
                if "window_start" not in {k.arg for k in node.keywords}:
                    bad.append((name, node.lineno))
    return bad, n


def test_the_static_check_flags_a_construction_without_the_keyword_and_accepts_one_with_it():
    """The checker's own controls: it can fail (a missing keyword, a splat, a bare-name call) and it can pass."""
    assert _constructions_without_window_start("ds = D.V8FMiniWindows(s, J, [], hist=cfg.hist)") == ([("<src>", 1)], 1)
    assert _constructions_without_window_start("ds = V8FMiniWindows(s, J, [], hist=1, **kw)") == ([("<src>", 1)], 1)
    assert _constructions_without_window_start("ds = D.V8FMiniWindows(s, J, [], hist=cfg.hist, window_start=cfg.window_start)") == ([], 1)
    assert _constructions_without_window_start("x = D.train_windows(s, J, [], hist=1)") == ([], 0)


def test_every_V8FMiniWindows_construction_under_stack_scripts_passes_window_start():
    """(c) the future-caller guard: a script that builds the dataset and forgets the checkpoint's / the arm's `window_start` is exactly the defect Amendment 1's eval fix closed."""
    files = sorted((REPO / "stack" / "scripts").rglob("*.py"))
    assert {p.name for p in files} >= {"v8f_mini_train.py", "v8f_mini_eval.py", "v8f_t1_eval.py"}
    bad, total = [], {}
    for p in files:
        b, n = _constructions_without_window_start(p.read_text(encoding="utf-8"), f"{p.name}")
        bad += b
        if n:
            total[p.name] = n
    assert total.keys() >= {"v8f_mini_train.py", "v8f_mini_eval.py", "v8f_t1_eval.py"}, f"the check must SEE the three known callers, saw {total}"
    assert not bad, f"V8FMiniWindows built without window_start= (the clamp origin silently falls back to hist - 1): {bad}"


@pytest.mark.skipif(not (A1 / "pre_a2fix_v8f_mini_eval.py.ref").exists(), reason="the saved pre-fix evaluators are not in this checkout")
@pytest.mark.parametrize("ref", ["pre_a2fix_v8f_mini_eval.py.ref", "pre_a2fix_v8f_t1_eval.py.ref"])
def test_the_static_check_catches_the_saved_pre_fix_evaluators(ref):
    """The control on real source: the two evaluators as they were BEFORE the fix are flagged (one construction each, no keyword)."""
    bad, n = _constructions_without_window_start((A1 / ref).read_text(encoding="utf-8"), ref)
    assert n == 1 and len(bad) == 1
