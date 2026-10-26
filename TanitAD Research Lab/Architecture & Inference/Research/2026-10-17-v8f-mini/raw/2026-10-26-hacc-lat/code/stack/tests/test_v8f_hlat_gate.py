"""PREREG_V8F_HACC_LAT arm B: the optional speed gate ``hacc_gate_v`` beside ``hacc_w`` in ``V8FMini``. CPU only, < 60 s.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval;<repo>/stack/scripts  pytest -q stack/tests/test_v8f_hlat_gate.py

What is pinned, and the control that must go red for each (the MUTANTS at the bottom are source edits of ``v8f_mini.py`` exec'd as a fresh module; the same battery that passes on the real module must FAIL on each):
  * the gate ABSENT (the default) changes nothing: ``forward_infer`` is HACC's post-map bit for bit (an independent ``hacc_apply`` of the kept O plan), has no ``hacc_gate_fired`` key, and ``hacc_gate_v`` is not in the state dict.
  * v* = 0 (the gate never fires) is bit-identical to the ungated model on every output tensor (control C-B1's model-level form).
  * v* = +inf (the gate always fires): the CROSS-axis plan equals P's cross plan EXACTLY, the ALONG axis is HACC's (control C-B2's model-level form).
  * a mid v*: fired == (v0 < v*) with a STRICT inequality, the cross weight of a fired sample is 0 and the others' are untouched, equal to an independent float64 formula.
  * the gate refuses: no weights, NaN / negative / non-numeric; a state dict carrying the gate without the weights; a set gate with the weights gone (a silent no-op) -- and a refused call leaves the model unchanged.
  * a state-dict round trip (incl. v* = inf) reproduces ``forward_infer`` bit for bit; ``clear_hacc_weights`` takes the gate with it.
  * TRAINING is untouched: ``_forward`` and ``loss`` are equal with and without the gate; ``forward_infer`` still reads only INFERENCE_INPUT_KEYS and keeps its one-argument signature; ``forward_diag`` carries the gate.
  * REACHABILITY: the battery goes RED on a mutant that drops the gate from the comparison, hits the wrong axis, flips the inequality, does not hand ``v0`` over, does not apply the map, drops the gate in
    ``forward_diag`` or in the state-dict load, or reads the gate in ``_forward``.
"""
from __future__ import annotations

import inspect
import sys
import types
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
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_gate")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    return {"root": root, **w}


def make_batch(world, n=8, seed=0):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=10)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    b = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    return b, J


def build(mod, J, seed=7):
    cfg = mod.V8FMiniConfig(**TINY)
    mu, sd = mod.cons_stats(J)
    torch.manual_seed(seed)                                                 # two builds are the SAME model (every random init), not just the same O head
    model = mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.head_o[2].weight.copy_(torch.randn(model.head_o[2].weight.shape, generator=g) * 0.3)
    return model.eval()


def syn_weights(seed=3):
    w = np.random.default_rng(seed).uniform(0.1, 0.9, (30, 2))
    w[0, 0], w[1, 0], w[5, 1], w[6, 1] = 0.0, 1.0, 0.0, 1.0
    return w


def mixed_speed_batch(b, speeds):
    """The batch with the measured speed of sample i set to speeds[i] (p10f[:, -1, 0] -- the feature the prior reads and the gate compares)."""
    b = dict(b)
    p = b["p10f"].clone()
    p[:, -1, 0] = torch.as_tensor(speeds, dtype=p.dtype)
    b["p10f"] = p
    return b


def infer(m, b):
    with torch.no_grad():
        return m.forward_infer(b)


def same(a, b):
    ta, tb = {k: v for k, v in a.items() if torch.is_tensor(v)}, {k: v for k, v in b.items() if torch.is_tensor(v)}
    return set(ta) == set(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)


# ------------------------------------------------------------------------------------------------ the battery (runs on the real module AND on every mutant)
def battery(mod, world) -> dict:
    """name -> bool. An exception inside a check is a False (a mutant may raise where the real module computes)."""
    b0, J = make_batch(world)
    B = b0["p10f"].shape[0]
    speeds = np.linspace(0.0, 14.0, B)
    speeds[3] = 6.0                                                         # EXACTLY at v* = 6: must NOT fire (strict <)
    b = mixed_speed_batch(b0, speeds)
    w = syn_weights()
    res = {}

    def guard(name, fn):
        try:
            res[name] = bool(fn())
        except Exception:                                                   # noqa: BLE001 -- see the docstring
            res[name] = False

    def models():
        m0, mg = build(mod, J), build(mod, J)
        m0.set_hacc_weights(w)
        mg.set_hacc_weights(w)
        return m0, mg

    def c_zero():
        m0, mg = models()
        mg.set_hacc_gate(0.0)
        y0, yg = infer(m0, b), infer(mg, b)
        yg_ = {k: v for k, v in yg.items() if k != "hacc_gate_fired"}
        return same(y0, yg_) and not yg["hacc_gate_fired"].any()

    def c_inf():
        m0, mg = models()
        mg.set_hacc_gate(float("inf"))
        y0, yg = infer(m0, b), infer(mg, b)
        cross_is_P = torch.equal(yg["traj30"][..., 1], yg["p10"][..., 1])
        along_is_hacc = torch.equal(yg["traj30"][..., 0], y0["traj30"][..., 0])
        power = float((y0["traj30"][..., 1] - y0["p10"][..., 1]).abs().max()) > 1e-3         # the ungated map moves the cross axis: the check can see a miss
        return cross_is_P and along_is_hacc and power and bool(yg["hacc_gate_fired"].all())

    def c_mid():
        _, mg = models()
        mg.set_hacc_gate(6.0)
        y = infer(mg, b)
        fired = torch.as_tensor(speeds, dtype=torch.float32) < 6.0
        ok = torch.equal(y["hacc_gate_fired"], fired) and bool(fired.any()) and not bool(fired.all()) and not bool(fired[3])
        tau, P = y["traj30_o"].double().numpy(), y["p10"].double().numpy()
        W = np.broadcast_to(w[None], (B, 30, 2)).copy()
        W[fired.numpy(), :, 1] = 0.0
        ref = P + W * (tau - P)
        return ok and float(np.abs(y["traj30"].double().numpy() - ref).max()) < 1e-5

    def c_roundtrip():
        _, mg = models()
        mg.set_hacc_gate(float("inf"))
        sd = mg.state_dict()
        m2 = build(mod, J)
        m2.load_state_dict(sd)                                              # strict
        return "hacc_gate_v" in sd and m2.hacc_gate_active and same(infer(mg, b), infer(m2, b))

    def c_diag():
        _, mg = models()
        mg.set_hacc_gate(6.0)
        y = infer(mg, b)
        with torch.no_grad():
            yd = mg.forward_diag(b, y["code"])
        return same(y, yd)

    def c_forward_untouched():
        m0, mg = models()
        mg.set_hacc_gate(6.0)
        with torch.no_grad():
            o0, og = m0._forward(dict(b), 0.0, False), mg._forward(dict(b), 0.0, False)
        return set(o0) == set(og) and all(torch.equal(o0[k], og[k]) for k in o0 if torch.is_tensor(o0[k])) and not any("gate" in k for k in og)

    def c_refuse_no_weights():
        m = build(mod, J)
        try:
            m.set_hacc_gate(4.0)
        except ValueError:
            return m.hacc_gate_v is None
        return False

    for name, fn in (("zero_is_ungated", c_zero), ("inf_cross_is_P", c_inf), ("mid_matches_formula", c_mid), ("roundtrip", c_roundtrip), ("diag_carries_gate", c_diag),
                     ("forward_untouched", c_forward_untouched), ("refuses_without_weights", c_refuse_no_weights)):
        guard(name, fn)
    return res


# ------------------------------------------------------------------------------------------------ the real module
def test_battery_is_green_on_the_real_module(world):
    res = battery(M, world)
    assert all(res.values()), res


def test_gate_absent_is_hacc_bit_for_bit(world):
    b, J = make_batch(world)
    m = build(M, J)
    m.set_hacc_weights(syn_weights())
    y = infer(m, b)
    assert m.hacc_gate_v is None and not m.hacc_gate_active and "hacc_gate_v" not in m.state_dict()
    assert "hacc_gate_fired" not in y
    w60 = M.hacc_w_midpoints(m.hacc_w)
    ref = M.hacc_apply(y["traj60_o"], M.upsample_prior(y["p10"]), w60)                  # HACC's own formula on the kept O plan
    assert torch.equal(y["traj60"], ref) and torch.equal(y["traj30"], ref[:, 1::2])
    plain = build(M, J)
    assert torch.equal(infer(plain, b)["traj30"], y["traj30_o"]) and "hacc_w" not in plain.state_dict()


def test_gate_is_strict_and_per_sample_and_cross_only(world):
    b0, J = make_batch(world)
    B = b0["p10f"].shape[0]
    speeds = np.array([0.0, 1.9, 2.0, 2.1, 4.0, 5.99, 6.0, 30.0][:B])
    b = mixed_speed_batch(b0, speeds)
    w = syn_weights()
    m0, mg = build(M, J), build(M, J)
    for m in (m0, mg):
        m.set_hacc_weights(w)
    mg.set_hacc_gate(2.0)
    y0, yg = infer(m0, b), infer(mg, b)
    fired = torch.as_tensor(speeds, dtype=torch.float32) < 2.0
    assert fired.tolist() == [True, True, False, False, False, False, False, False][:B] and torch.equal(yg["hacc_gate_fired"], fired)
    assert torch.equal(yg["traj30"][..., 0], y0["traj30"][..., 0])                          # the along axis is never gated
    assert torch.equal(yg["traj30"][~fired], y0["traj30"][~fired])                          # unfired samples are HACC's, bit for bit
    assert torch.equal(yg["traj30"][fired][..., 1], yg["p10"][fired][..., 1])               # fired samples: the cross axis IS the prior


def test_gate_value_validation_and_refusals(world):
    _, J = make_batch(world)
    m = build(M, J)
    with pytest.raises(ValueError):
        m.set_hacc_gate(3.0)                                                                # no weights
    assert m.hacc_gate_v is None
    m.set_hacc_weights(syn_weights())
    for bad in (float("nan"), -0.5, "fast", None, [1.0]):
        with pytest.raises(ValueError):
            m.set_hacc_gate(bad)
        assert m.hacc_gate_v is None
    m.set_hacc_gate(0)
    m.set_hacc_gate(float("inf"))
    assert float(m.hacc_gate_v) == float("inf")
    assert M.hacc_check_gate(torch.tensor(4.0)) == 4.0
    m.clear_hacc_gate()
    assert m.hacc_gate_v is None and m.hacc_w is not None
    m.set_hacc_gate(4.0)
    m.clear_hacc_weights()
    assert m.hacc_gate_v is None and m.hacc_w is None                                         # the gate goes with the weights
    sd = build(M, J).state_dict()
    sd["hacc_gate_v"] = torch.tensor(3.0)                                                   # a gate with no weights beside it
    with pytest.raises(ValueError):
        build(M, J).load_state_dict(sd)
    m2 = build(M, J)
    m2.set_hacc_weights(syn_weights())
    sd2 = m2.state_dict()
    sd2["hacc_gate_v"] = torch.tensor(-1.0)
    with pytest.raises(ValueError):
        build(M, J).load_state_dict(sd2)


def test_set_gate_without_weights_after_the_fact_is_a_loud_error_not_a_silent_noop(world):
    b, J = make_batch(world)
    m = build(M, J)
    m.set_hacc_weights(syn_weights())
    m.set_hacc_gate(4.0)
    m.hacc_w = None                                                                          # a hand-edited model: the gate would silently do nothing
    with pytest.raises(ValueError):
        infer(m, b)


def test_training_and_inference_surfaces_are_untouched(world):
    b, J = make_batch(world)
    m0, mg = build(M, J), build(M, J)
    for m in (m0, mg):
        m.set_hacc_weights(syn_weights())
    mg.set_hacc_gate(6.0)
    for m in (m0, mg):
        m.train()
    torch.manual_seed(0)
    l0 = m0.loss(b, p_gt=0.5)
    torch.manual_seed(0)
    lg = mg.loss(b, p_gt=0.5)
    assert set(l0) == set(lg) and all(torch.equal(l0[k], lg[k]) for k in l0 if torch.is_tensor(l0[k]))
    assert M.V8FMini.forward_infer.__code__.co_varnames[:2] == ("self", "batch") and list(inspect.signature(M.V8FMini.forward_infer).parameters) == ["self", "batch"]
    mg.eval()
    keys = M.assert_no_situation_classifier_input(mg, b)
    assert set(keys["read"]) <= M.INFERENCE_INPUT_KEYS and "p10f" in keys["read"]
    assert M.scan_source_for_forbidden(inspect.getsource(M)) == []


# ------------------------------------------------------------------------------------------------ the mutants: the battery must go RED
SRC_FIRE = "< self.hacc_gate_v.to(device=plan60.device, dtype=plan60.dtype)"
MUTANTS = {
    "gate_never_fires": (SRC_FIRE, "< -1.0"),
    "flipped_inequality": (SRC_FIRE, "> self.hacc_gate_v.to(device=plan60.device, dtype=plan60.dtype)"),
    "wrong_axis": ("HACC_GATE_AXIS = 1\n", "HACC_GATE_AXIS = 0\n"),
    "forward_infer_drops_v0": ('return self._hacc_post_map(out, None if self.hacc_gate_v is None else g["p10f"][:, -1, 0])\n\n    def _hacc_post_map', "return self._hacc_post_map(out, None)\n\n    def _hacc_post_map"),
    "map_not_applied": ('plan60 = hacc_apply(out["traj60"], prior60, hacc_w_midpoints(w30))', 'plan60 = out["traj60"] + 0.0'),
    "diag_drops_gate": ('return self._hacc_post_map(out, None if self.hacc_gate_v is None else g["p10f"][:, -1, 0])     # the same post-map', "return self._hacc_post_map(out, None)     # the same post-map"),
    "load_drops_gate": ("if gkey in state_dict:", "if False:"),
    "forward_reads_gate": ('out["adapt"] = adapt\n        return out', 'out["adapt"] = adapt\n        out["gate_seen"] = self.hacc_gate_v\n        return out'),
    "gate_not_registered": ('self.register_buffer("hacc_gate_v", torch.tensor(x, dtype=torch.float32, device=self.basis.device))', "pass"),
}


def exec_mutant(old, new):
    src = inspect.getsource(M).replace("\r\n", "\n")
    assert src.count(old) == 1, f"the mutation anchor occurs {src.count(old)}x (the source moved): {old[:70]!r}"
    mod = types.ModuleType("v8f_mini_mutant")
    mod.__file__ = "<mutant>"
    sys.modules["v8f_mini_mutant"] = mod
    try:
        exec(compile(src.replace(old, new), "<mutant>", "exec"), mod.__dict__)
    except Exception:
        sys.modules.pop("v8f_mini_mutant", None)
        raise
    return mod


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_turns_the_battery_red(world, name):
    mod = exec_mutant(*MUTANTS[name])
    try:
        res = battery(mod, world)
    finally:
        sys.modules.pop("v8f_mini_mutant", None)
    assert not all(res.values()), f"mutant {name!r} survived: {res}"
