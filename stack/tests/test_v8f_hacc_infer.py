"""PREREG_V8F_HACC, the deployed post-map: ``forward_infer`` returns ``P + w * (O - P)`` when the optional buffer ``hacc_w`` [30, 2] is set. CPU only, < 90 s.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_v8f_hacc_infer.py

What is pinned, and the control that must go red for each (the mutants: ``raw/2026-10-26-hacc-infer/mutate_hacc_infer.py``):
  * the buffer ABSENT (the default) changes nothing: ``forward_infer`` is bit-identical to ``_forward`` (what it was before the post-map), the output keys are the same, ``hacc_w`` is not in the
    state dict -- on a synthetic model AND on a real checkpoint (strict load, real eval windows; fixtures-gated). [mutant: apply the map when the buffer is absent; register it with a default]
  * w = 1 gives O's plan EXACTLY, w = 0 gives the P10 prior EXACTLY (both grids); a general w equals an independent float64 formula, on BOTH grids. [mutant: inverted blend; skip the midpoint rule]
  * the axis convention (column 0 = along, column 1 = cross) and the 30 -> 60 weight rule (HBLEND's ``w_midpoints``: odd rows = the knots, even rows = the mean of the neighbours, the first = the
    first knot). [mutants: swap the axes; the loader swaps the columns; nearest-neighbour weights; a mean with the origin]
  * ``traj30`` is ``traj60[:, 1::2]`` (the 0.2 s knots, not the even rows) and every other output is untouched. [mutants: even rows; a map that touches only traj60]
  * the loader reads the weights JSON and REFUSES a wrong shape, a non-finite entry, an entry outside [0, 1], a missing key, ragged lists -- and a refused call leaves the model unchanged.
    [mutants: no range check; no shape check]
  * a state-dict round trip (in memory and through a file) keeps the weights and reproduces ``forward_infer`` bit for bit; a state dict that carries a bad ``hacc_w`` is refused.
    [mutant: no ``_load_from_state_dict`` override]
  * TRAINING is untouched: with the buffer set, ``loss()`` terms, the training-mode plan, the gradient and ``built_levers`` equal the unset model's. [mutant: the map in ``_forward``]
  * inference stays vision + ego-dynamics only: the keys read are inside INFERENCE_INPUT_KEYS, ``forward_diag`` carries the same post-map (its identity with ``forward_infer`` under T's own code).
    [mutant: forward_diag unmapped]
  * the REAL weights (``raw/2026-10-25-hacc/run/weights_HACC_M2b_s{0,1}.json``) are the table that was scored: their canonical hash is pinned, and pushed through ``forward_infer`` with the real
    checkpoints' P and O's banked plan they reproduce HACC's banked per-window plan on all 2,399 eval windows to <= 1e-5 m, while the OTHER seed's weights do not (the test has power).
  * on a real checkpoint's real eval windows the model's own HACC plan agrees with the banked one cell by cell to within the float noise of the two forwards (fixtures-gated).
"""
from __future__ import annotations

import hashlib
import json
import os
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

torch.set_num_threads(1)
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
SEEDS = ("M2b_s0", "M2b_s1")

# ---- repo artifacts (landed with HACC): the scored weights and the per-window plans they produced
HACC_RUN = PKG / "raw" / "2026-10-25-hacc" / "run"
HBLEND_RAW = PKG / "raw" / "2026-10-25-hblend"
WEIGHTS = {s: (HACC_RUN / f"weights_HACC_{s}.json" if (HACC_RUN / f"weights_HACC_{s}.json").exists() else None) for s in SEEDS}
BANKED_HACC = {s: (HACC_RUN / f"score_{s}_perwindow.npz" if (HACC_RUN / f"score_{s}_perwindow.npz").exists() else None) for s in SEEDS}
BANKED_TAU_P = {s: (HBLEND_RAW / f"score_{s}_perwindow.npz" if (HBLEND_RAW / f"score_{s}_perwindow.npz").exists() else None) for s in SEEDS}
#: sha256 of the CANONICAL bytes of each weights table (the parsed [30, 2] float64 C-order array: immune to a CRLF checkout of the JSON)
WEIGHTS_SHA = {"M2b_s0": "5c330a9ba98d66ccdaf3770b518f2a23ffee2612987684776a76a89eb33e9b7f",
               "M2b_s1": "eee64e066792404710eff44009e4eb6e961466060a72ce378a8706d271d4e0a1"}

# ---- fixtures that are NOT in the repo (real checkpoints, the one cached eval clip): gated, skipped with a reason when absent
FIX = Path(os.environ.get("TANITAD_HACC_FIXTURES", "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz/work"))
CKPT = {s: (FIX / "ckpt" / f"{s}.pt" if (FIX / "ckpt" / f"{s}.pt").exists() else None) for s in SEEDS}
_FIX_FILES = [FIX / "inputs" / n for n in ("join_eval141.npz", "eval_grid_windows.json", "native_tracks.pt")] + [FIX / "cache" / "eval" / "manifest.jsonl"]
HAVE_REAL = all(CKPT.values()) and all(p.exists() for p in _FIX_FILES) and all(BANKED_HACC.values()) and all(BANKED_TAU_P.values()) \
    and all((FIX / "banked" / f"eval_{s}-kind-own.npz").exists() for s in SEEDS)
need_weights = pytest.mark.skipif(not all(WEIGHTS.values()), reason="the HACC weights JSONs are not in this tree (raw/2026-10-25-hacc/run/)")
need_banked = pytest.mark.skipif(not (all(WEIGHTS.values()) and all(BANKED_HACC.values()) and all(BANKED_TAU_P.values())),
                                 reason="the banked per-window HACC / HBLEND plans are not in this tree (raw/2026-10-25-{hacc/run,hblend}/)")
need_real = pytest.mark.skipif(not HAVE_REAL, reason=f"real checkpoints / the cached eval clip / banked eval npz not under {FIX} (set TANITAD_HACC_FIXTURES)")


# ------------------------------------------------------------------------------------------------ synthetic model + batch (as test_v8f_mini)
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    return {"root": root, **w}


def make_batch(world, n=8, seed=0, hmax=10):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    items = []
    for i in idx:
        ci, t, _, _ = wins[i]
        items.append(ds.item(ci, t, 1, 3))
    return D.collate(items), J


def tiny_model(J, seed=7, **kw):
    """A tiny V8FMini whose O head is NON-zero (a zero-init head makes O's plan == P10, and then every blend weight gives the same plan)."""
    cfg = M.V8FMiniConfig(**{**TINY, **kw})
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.head_o[2].weight.copy_(torch.randn(model.head_o[2].weight.shape, generator=g) * 0.3)
    return model.eval()


def syn_weights(seed=3):
    """An ASYMMETRIC table (the two axes and the 30 steps all differ) that touches both ends of [0, 1]."""
    w = np.random.default_rng(seed).uniform(0.0, 1.0, (30, 2))
    w[0, 0], w[1, 0], w[5, 1], w[6, 1] = 0.0, 1.0, 0.0, 1.0
    return w


def refuses(fn, *a, **k) -> bool:
    """True iff ``fn`` raises ValueError (a plain bool, so a mutant that stops refusing fails an ``assert``)."""
    try:
        fn(*a, **k)
    except ValueError:
        return True
    return False


def run_infer(model, b):
    with torch.no_grad():
        return model.forward_infer(b)


def tensors_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)


def w_midpoints_ref(w30):
    """HBLEND's ``w_midpoints`` (scratch ``hblend/v8f_hblend.py``), transcribed: the reference the torch rule must equal."""
    w60 = np.zeros((60, 2))
    w60[1::2] = w30
    mid = np.concatenate([w30[:1], 0.5 * (w30[:-1] + w30[1:])], 0)
    w60[0::2] = mid
    return w60


def knots_ref(p30):
    """The prior on the 0.1 s grid, independently of ``upsample_prior``: origin prepended, midpoints between neighbours (numpy)."""
    k = np.concatenate([np.zeros((p30.shape[0], 1, 2)), p30], 1)
    mid = 0.5 * (k[:, :-1] + k[:, 1:])
    out = np.zeros((p30.shape[0], 60, 2))
    out[:, 0::2], out[:, 1::2] = mid, k[:, 1:]
    return out


# ------------------------------------------------------------------------------------------------ 1. the default changes nothing
def test_absent_buffer_forward_infer_is_bit_identical_to_the_plain_forward(world):
    b, J = make_batch(world)
    model = tiny_model(J)
    assert model.hacc_w is None and not model.hacc_active
    assert "hacc_w" not in model.state_dict() and "hacc_w" not in dict(model.named_buffers())
    got = run_infer(model, b)
    with torch.no_grad():
        was = M._KeyGuard(b)
        want = model._forward(was, 0.0, False)                 # forward_infer BEFORE the post-map was exactly this + _keys_read
    want["_keys_read"] = set(was.read)
    assert tensors_equal(got, want), "an absent hacc_w must leave forward_infer's outputs and keys exactly as they were"
    assert "traj60_o" not in got and "traj30_o" not in got
    # CONTROL: the O head is live here, so O's plan is NOT the prior (otherwise every comparison below is vacuous)
    assert (got["traj30"] - got["p10"]).abs().max() > 1e-2


def test_set_then_clear_returns_to_the_plain_plan_bit_for_bit(world):
    b, J = make_batch(world)
    model = tiny_model(J)
    before = run_infer(model, b)
    model.set_hacc_weights(syn_weights())
    assert model.hacc_active and "hacc_w" in model.state_dict()
    assert (run_infer(model, b)["traj30"] - before["traj30"]).abs().max() > 1e-3, "the map must move the plan, or every test below is vacuous"
    model.clear_hacc_weights()
    assert model.hacc_w is None and "hacc_w" not in model.state_dict()
    assert tensors_equal(run_infer(model, b), before)


# ------------------------------------------------------------------------------------------------ 2. the blend
def test_ends_of_the_blend_are_exact_on_both_grids(world):
    b, J = make_batch(world)
    model = tiny_model(J)
    plain = run_infer(model, b)
    model.set_hacc_weights(np.ones((30, 2)))
    one = run_infer(model, b)
    assert torch.equal(one["traj60"], plain["traj60"]) and torch.equal(one["traj30"], plain["traj30"]), "w = 1 must give O's plan exactly"
    model.set_hacc_weights(np.zeros((30, 2)))
    zero = run_infer(model, b)
    assert torch.equal(zero["traj60"], M.upsample_prior(plain["p10"])), "w = 0 must give the P10 prior exactly (all 60 rows)"
    assert torch.equal(zero["traj30"], plain["p10"])
    assert (plain["traj30"] - plain["p10"]).abs().max() > 1e-2          # the two ends are different plans


def test_general_blend_equals_an_independent_float64_formula_on_both_grids(world):
    b, J = make_batch(world)
    model = tiny_model(J)
    plain = run_infer(model, b)
    model.set_hacc_weights(syn_weights())
    got = run_infer(model, b)
    w30 = model.hacc_w.double().numpy()
    P60 = knots_ref(plain["p10"].double().numpy())
    tau60 = plain["traj60"].double().numpy()
    want60 = P60 + w_midpoints_ref(w30)[None] * (tau60 - P60)
    np.testing.assert_allclose(got["traj60"].double().numpy(), want60, atol=2e-5, rtol=0)
    np.testing.assert_allclose(got["traj30"].double().numpy(), want60[:, 1::2], atol=2e-5, rtol=0)
    # the same on the 0.2 s knots by the PREREG's own formula  H = P + w (.) (tau - P)
    p30, t30 = plain["p10"].double().numpy(), plain["traj30"].double().numpy()
    np.testing.assert_allclose(got["traj30"].double().numpy(), p30 + w30[None] * (t30 - p30), atol=2e-5, rtol=0)


def test_axis_convention_column0_is_along_column1_is_cross(world):
    b, J = make_batch(world)
    model = tiny_model(J)
    plain = run_infer(model, b)
    w = np.zeros((30, 2))
    w[:, 0] = 1.0                                   # along = O, cross = P
    model.set_hacc_weights(w)
    got = run_infer(model, b)
    assert torch.equal(got["traj30"][..., 0], plain["traj30"][..., 0]), "axis 0 (along / x) must follow O when column 0 is 1"
    assert torch.equal(got["traj30"][..., 1], plain["p10"][..., 1]), "axis 1 (cross / y) must follow the prior when column 1 is 0"
    assert torch.equal(got["traj60"][..., 0], plain["traj60"][..., 0])
    assert torch.equal(got["traj60"][..., 1], M.upsample_prior(plain["p10"])[..., 1])
    assert (plain["traj30"][..., 1] - plain["p10"][..., 1]).abs().max() > 1e-2          # the cross axes differ, so the line above is a measurement


def test_midpoint_rule_known_values():
    w30 = np.arange(1, 31, dtype=np.float64)[:, None] * np.array([[1.0, 2.0]]) / 62.0           # w[j, a] = (j + 1) (a + 1) / 62: strictly increasing, distinct per axis
    got = M.hacc_w_midpoints(torch.as_tensor(w30)).numpy()
    assert got.shape == (60, 2)
    np.testing.assert_array_equal(got[1::2], w30)                                                # odd rows (tau = 0.2 j) ARE the knot weights
    np.testing.assert_array_equal(got[0], w30[0])                                                # tau = 0.1: the first knot's
    np.testing.assert_allclose(got[2], 0.5 * (w30[0] + w30[1]), atol=1e-15)                      # tau = 0.3: the mean of knots 1 and 2
    np.testing.assert_allclose(got[58], 0.5 * (w30[28] + w30[29]), atol=1e-15)                   # tau = 5.9
    np.testing.assert_array_equal(got[59], w30[29])
    np.testing.assert_allclose(got, w_midpoints_ref(w30), atol=1e-15)                            # HBLEND's rule, verbatim


def test_traj30_is_the_odd_rows_of_traj60_and_every_other_output_is_untouched(world):
    b, J = make_batch(world)
    model = tiny_model(J)
    plain = run_infer(model, b)
    model.set_hacc_weights(syn_weights())
    got = run_infer(model, b)
    assert torch.equal(got["traj30"], got["traj60"][:, 1::2]), "traj30 must be the 0.2 s knots (odd rows) of the mapped traj60"
    assert torch.equal(got["traj60_o"], plain["traj60"]) and torch.equal(got["traj30_o"], plain["traj30"]), "O's own plan is kept under *_o"
    assert set(got) == set(plain) | {"traj60_o", "traj30_o"}
    for k in plain:
        if k not in ("traj60", "traj30"):
            assert (torch.equal(got[k], plain[k]) if torch.is_tensor(plain[k]) else got[k] == plain[k]), f"output {k!r} must not change"
    assert (got["traj60"] - plain["traj60"]).abs().max() > 1e-3 and (got["traj30"] - plain["traj30"]).abs().max() > 1e-3


# ------------------------------------------------------------------------------------------------ 3. the loader
def test_loader_reads_the_json_and_refuses_bad_tables(world, tmp_path):
    b, J = make_batch(world)
    model = tiny_model(J)
    w = syn_weights()
    good = {"w_along_by_step": w[:, 0].tolist(), "w_cross_by_step": w[:, 1].tolist(), "arm": "HACC"}
    pth = tmp_path / "w.json"
    pth.write_text(json.dumps(good), encoding="utf-8")
    np.testing.assert_array_equal(M.read_hacc_weights(pth), w)                    # column 0 = along, column 1 = cross
    np.testing.assert_array_equal(M.read_hacc_weights(good), w)                   # a parsed dict works too
    model.load_hacc_weights(pth)
    np.testing.assert_allclose(model.hacc_w.numpy(), w.astype(np.float32), atol=0, rtol=0)
    assert model.hacc_w.dtype == torch.float32 and tuple(model.hacc_w.shape) == (30, 2) and not model.hacc_w.requires_grad
    model.clear_hacc_weights()

    def bad(**over):
        return {**good, **over}
    cases = {
        "too short": lambda: model.set_hacc_weights(np.zeros((29, 2))),
        "too long (the 60-row grid is not a fitted table)": lambda: model.set_hacc_weights(np.zeros((60, 2))),
        "3 columns": lambda: model.set_hacc_weights(np.zeros((30, 3))),
        "transposed": lambda: model.set_hacc_weights(np.zeros((2, 30))),
        "1-D": lambda: model.set_hacc_weights(np.zeros(30)),
        "below 0": lambda: model.set_hacc_weights(np.where(np.arange(60).reshape(30, 2) == 7, -1e-9, 0.5)),
        "above 1": lambda: model.set_hacc_weights(np.where(np.arange(60).reshape(30, 2) == 7, 1.0 + 1e-9, 0.5)),
        "nan": lambda: model.set_hacc_weights(np.where(np.arange(60).reshape(30, 2) == 7, np.nan, 0.5)),
        "inf": lambda: model.set_hacc_weights(np.where(np.arange(60).reshape(30, 2) == 7, np.inf, 0.5)),
        "strings": lambda: model.set_hacc_weights([["a", "b"]] * 30),
        "json: a key missing": lambda: model.load_hacc_weights({"w_along_by_step": good["w_along_by_step"]}),
        "json: ragged lists": lambda: model.load_hacc_weights(bad(w_cross_by_step=good["w_cross_by_step"][:29])),
        "json: not a dict": lambda: model.load_hacc_weights([0.5] * 60),
        "json: out of range": lambda: model.load_hacc_weights(bad(w_along_by_step=[1.5] + good["w_along_by_step"][1:])),
    }
    for name, fn in cases.items():
        assert refuses(fn), f"the loader must REFUSE: {name}"
        assert model.hacc_w is None, f"a refused call must leave the model unchanged: {name}"
    # the ends of the range are admissible, and a table that is already set survives a refused replacement
    model.set_hacc_weights(np.zeros((30, 2)))
    model.set_hacc_weights(np.ones((30, 2)))
    model.set_hacc_weights(w)
    assert refuses(model.set_hacc_weights, w + 1.0)
    np.testing.assert_allclose(model.hacc_w.numpy(), w.astype(np.float32), atol=0, rtol=0)


# ------------------------------------------------------------------------------------------------ 4. the state dict
def test_state_dict_round_trip_keeps_the_weights_and_the_plan(world, tmp_path):
    b, J = make_batch(world)
    src = tiny_model(J)
    src.set_hacc_weights(syn_weights())
    want = run_infer(src, b)
    sd = src.state_dict()
    assert "hacc_w" in sd and sd["hacc_w"].dtype == torch.float32 and tuple(sd["hacc_w"].shape) == (30, 2)
    dst = tiny_model(J, seed=99)                                               # a different O head, no post-map
    assert dst.hacc_w is None
    res = dst.load_state_dict(sd, strict=False)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == [], res
    assert dst.hacc_w is not None and torch.equal(dst.hacc_w, src.hacc_w)
    assert tensors_equal(run_infer(dst, b), want), "the loaded model must reproduce the HACC plan bit for bit"
    # ... through a file as well
    f = tmp_path / "ck.pt"
    torch.save({"model": src.state_dict()}, f)
    dst2 = tiny_model(J, seed=5)
    res2 = dst2.load_state_dict(torch.load(f, weights_only=False)["model"], strict=False)
    assert list(res2.missing_keys) == [] and list(res2.unexpected_keys) == []
    assert torch.equal(dst2.hacc_w, src.hacc_w)
    # a checkpoint WITHOUT the buffer loads strictly into a model without it, and leaves it absent
    plain = tiny_model(J)
    old_sd = {k: v for k, v in plain.state_dict().items()}
    assert "hacc_w" not in old_sd
    fresh = tiny_model(J, seed=11)
    res3 = fresh.load_state_dict(old_sd, strict=False)
    assert list(res3.missing_keys) == [] and list(res3.unexpected_keys) == [] and fresh.hacc_w is None


def test_state_dict_with_a_bad_table_is_refused(world):
    _, J = make_batch(world)
    src = tiny_model(J)
    src.set_hacc_weights(syn_weights())
    sd = dict(src.state_dict())
    for name, bad in (("out of range", sd["hacc_w"] + 5.0), ("wrong shape", sd["hacc_w"][:20]), ("nan", torch.full((30, 2), float("nan")))):
        dst = tiny_model(J)
        assert refuses(dst.load_state_dict, {**sd, "hacc_w": bad}, strict=False), f"load must refuse a state dict whose hacc_w is {name}"
        assert dst.hacc_w is None, name
        held = tiny_model(J)
        held.set_hacc_weights(syn_weights(5))
        assert refuses(held.load_state_dict, {**sd, "hacc_w": bad}, strict=False), f"... also into a model that already has the buffer: {name}"


# ------------------------------------------------------------------------------------------------ 5. training is untouched
def test_training_path_loss_plan_and_gradient_are_unchanged_by_the_buffer(world):
    b, J = make_batch(world, n=8)
    model = tiny_model(J)
    model.train()
    levers = model.built_levers()

    def train_read():
        torch.manual_seed(1)
        gen = torch.Generator().manual_seed(2)
        model.zero_grad(set_to_none=True)
        out = model.loss(b, p_gt=0.5, gen=gen)
        out["terms"]["total"].backward()
        return out, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    o0, g0 = train_read()
    model.set_hacc_weights(syn_weights())
    o1, g1 = train_read()
    assert model.built_levers() == levers, "built_levers (the G-DVB 'built' side) must not move"
    assert set(o0["terms"]) == set(o1["terms"])
    for k in o0["terms"]:
        assert torch.equal(o0["terms"][k], o1["terms"][k]), f"training loss term {k!r} changed with the post-map set"
    assert torch.equal(o0["traj30"], o1["traj30"]) and torch.equal(o0["traj60"], o1["traj60"]), "the TRAINING forward must read O's raw plan"
    assert "traj60_o" not in o1 and "traj30_o" not in o1
    assert set(g0) == set(g1) and all(torch.equal(g0[n], g1[n]) for n in g0), "gradients must be unchanged"
    assert not model.hacc_w.requires_grad and "hacc_w" not in dict(model.named_parameters())
    assert g0["head_o.2.weight"].abs().sum() > 0                                                  # CONTROL: O's head does get a gradient here
    # while the INFERENCE plan of the very same model does differ (the buffer is live)
    model.eval()
    assert (run_infer(model, b)["traj30"] - o0["traj30"].detach()).abs().max() > 1e-3


# ------------------------------------------------------------------------------------------------ 6. admissibility
def test_inference_stays_admissible_and_forward_diag_carries_the_same_map(world):
    b, J = make_batch(world)
    model = tiny_model(J)
    plain = run_infer(model, b)
    model.set_hacc_weights(syn_weights())
    info = M.assert_no_situation_classifier_input(model, b)                       # source scan + keys read are inside INFERENCE_INPUT_KEYS
    assert set(info["read"]) <= M.INFERENCE_INPUT_KEYS
    mapped = run_infer(model, b)
    assert mapped["_keys_read"] == plain["_keys_read"], "the post-map must read no new batch key"
    assert not any(k.startswith("lab") or k in ("gt", "gt_ok") for k in mapped["_keys_read"])
    with torch.no_grad():
        diag = model.forward_diag(b, mapped["code"])
        other = model.forward_diag(b, mapped["code"].roll(1, 0))
    assert tensors_equal({k: v for k, v in diag.items() if k != "_keys_read"}, {k: v for k, v in mapped.items() if k != "_keys_read"}), \
        "forward_diag under T's own code must stay IDENTICAL to forward_infer (the post-map is in both)"
    assert (other["traj30"] - mapped["traj30"]).abs().max() > 1e-3                # O is code-sensitive here, so the identity above is a measurement


# ------------------------------------------------------------------------------------------------ 7. the REAL weights and the banked plans
@need_weights
def test_real_weights_are_the_scored_table():
    ws = {}
    for s in SEEDS:
        w = M.read_hacc_weights(WEIGHTS[s])
        ws[s] = w
        assert w.shape == (30, 2) and w.min() >= 0.0 and w.max() <= 1.0
        assert hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest() == WEIGHTS_SHA[s], f"{s}: the deployed table is not the one HACC scored"
        meta = json.load(open(WEIGHTS[s], encoding="utf-8"))
        assert meta["arm"] == "HACC" and meta["seed"] == s
    assert np.abs(ws["M2b_s0"] - ws["M2b_s1"]).max() > 0.01, "the two models' tables are different (per-model constants)"


def _fixed_plan_model(P, tau):
    """A tiny model whose ``_forward`` returns FIXED outputs -- the banked prior P and O's banked plan tau on the 0.2 s knots (even rows = the prior's midpoints) -- so that the REAL
    ``forward_infer`` post-map is applied to the 2,399 real eval windows without a feature cache."""
    model = M.V8FMini(M.V8FMiniConfig(**TINY), M.P10Prior.synthetic(0)).eval()
    p = torch.as_tensor(P, dtype=torch.float32)
    t60 = M.upsample_prior(p).clone()
    t60[:, 1::2] = torch.as_tensor(tau, dtype=torch.float32)

    def fixed(g, p_gt, use_labels, gen=None, code_override=None):
        return {"traj60": t60.clone(), "traj30": M.subsample_02(t60.clone()), "p10": p.clone()}
    model._forward = fixed
    return model


@need_banked
@pytest.mark.parametrize("seed", SEEDS)
def test_real_weights_through_forward_infer_reproduce_the_banked_hacc_plan(seed):
    bt = np.load(BANKED_TAU_P[seed])
    hc = np.load(BANKED_HACC[seed])["HACC"]
    assert hc.shape == (2399, 30, 2) and bt["tau"].shape == hc.shape and bt["P"].shape == hc.shape
    model = _fixed_plan_model(bt["P"], bt["tau"])
    model.load_hacc_weights(WEIGHTS[seed])
    with torch.no_grad():
        out = model.forward_infer({})
    err = np.abs(out["traj30"].double().numpy() - hc)
    assert err.max() <= 1e-5, f"{seed}: the model's HACC plan differs from the banked per-window plan by {err.max():.3g} m (> 1e-5)"
    assert torch.equal(out["traj30"], out["traj60"][:, 1::2])
    # POWER: the other seed's table, and no table at all, are both far from the banked plan
    other = SEEDS[1 - SEEDS.index(seed)]
    swapped = _fixed_plan_model(bt["P"], bt["tau"])
    swapped.load_hacc_weights(WEIGHTS[other])
    with torch.no_grad():
        e_other = np.abs(swapped.forward_infer({})["traj30"].double().numpy() - hc).max()
    e_none = np.abs(bt["tau"] - hc).max()
    assert e_other > 1e-3 and e_none > 1e-3, (e_other, e_none)
    # the plan really is a blend: bounded by the prior and O's plan, cell by cell (w in [0, 1])
    lo, hi = np.minimum(bt["P"], bt["tau"]) - 1e-5, np.maximum(bt["P"], bt["tau"]) + 1e-5
    got = out["traj30"].double().numpy()
    assert (got >= lo).all() and (got <= hi).all()


# ------------------------------------------------------------------------------------------------ 8. a real checkpoint, real eval windows (fixtures-gated)
def _real_model(seed):
    ck = torch.load(CKPT[seed], map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    res = model.load_state_dict(ck["model"], strict=False)
    return model.eval(), cfg, ck, res


def _real_clip_batch(cfg):
    """The eval's own batch for every window of the cached eval clip(s) (the same loader chain and the same ``--ages cycle`` as ``v8f_mini_eval.cmd_run``)."""
    inp = FIX / "inputs"
    J = D.load_join(inp / "join_eval141.npz")
    tracks = D.load_native_tracks(inp / "native_tracks.pt")
    store = D.ClipStore(str(FIX / "cache"), "eval", lru=8, grid="pool4x10", kin_mode=cfg.kin_mode, tracks=tracks)
    wins = D.eval_grid_windows(str(inp / "eval_grid_windows.json"), J)
    cached = {p.stem for p in (FIX / "cache" / "eval" / "clips").glob("*.pt")}
    clips = J["clip"].tolist()
    idx = [i for i, w in enumerate(wins) if clips[w[0]] in cached]
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    items = []
    for i in idx:
        ci, t, _, _ = wins[i]
        a_t = i % 3
        items.append(ds.item(ci, t, a_t, a_t + (i // 3) % 5))
    return D.collate(items), np.asarray(idx)


@need_real
@pytest.mark.parametrize("seed", SEEDS)
def test_real_checkpoint_loads_unchanged_and_its_hacc_plan_matches_the_banked_one(seed):
    model, cfg, ck, res = _real_model(seed)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == [], "a checkpoint written before the post-map must load, strictly, unchanged"
    assert set(model.state_dict()) == set(ck["model"]) and model.hacc_w is None
    b, idx = _real_clip_batch(cfg)
    assert len(idx) >= 5
    bank = np.load(FIX / "banked" / f"eval_{seed}-kind-own.npz")["pred"][idx].astype(np.float64)
    tp = np.load(BANKED_TAU_P[seed])
    P_b, hacc_b = tp["P"][idx], np.load(BANKED_HACC[seed])["HACC"][idx]
    plain = run_infer(model, b)
    with torch.no_grad():
        was = M._KeyGuard(b)
        want = model._forward(was, 0.0, False)
    want["_keys_read"] = set(was.read)
    assert tensors_equal(plain, want), "absent hacc_w: forward_infer on the real checkpoint must be what it was before the post-map"
    tau_m = plain["traj30"].double().numpy()
    assert np.abs(tau_m - bank).max() < 5e-4, "harness alignment: this CPU forward must reproduce the banked O plan (its GPU-vs-CPU float noise is ~1e-4)"
    model.load_hacc_weights(WEIGHTS[seed])
    mapped = run_infer(model, b)
    assert set(mapped) == set(plain) | {"traj60_o", "traj30_o"}
    assert torch.equal(mapped["traj30_o"], plain["traj30"]) and torch.equal(mapped["traj30"], mapped["traj60"][:, 1::2])
    w = M.read_hacc_weights(WEIGHTS[seed])[None]
    dtau = np.abs(tau_m - bank)
    dP = np.abs(plain["p10"].double().numpy() - P_b)
    bound = w * dtau + (1.0 - w) * dP + 1e-5                    # HACC is a convex blend: its change is bounded by the change of its two inputs (+ float32 output rounding)
    err = np.abs(mapped["traj30"].double().numpy() - hacc_b)
    assert (err <= bound).all(), f"{seed}: the model's HACC plan leaves the bound set by its own inputs' float noise (max excess {(err - bound).max():.3g} m)"
    assert err.max() < 5e-4
    # POWER: the unmapped plan is far from HACC on these windows, so agreeing with the banked HACC is not agreeing with the banked O
    assert np.abs(tau_m - hacc_b).max() > 1e-2
