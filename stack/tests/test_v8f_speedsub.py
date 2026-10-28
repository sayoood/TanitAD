"""PREREG_V8F_SPEEDSUB (DRAFT): synthetic self-test of the `ego` substitution and its instrument. CPU only, no real data, no model checkpoint.

Four things are pinned.
 (A) `ego` is RED on the unmodified SPEEDREAD instrument (golden/v8f_mini_eval_speedread_fd5ce0be.py, sha256-checked) and GREEN on the instrument under test.
 (B) Every SPEEDREAD mode is unchanged: a same-process differential against the pinned original (exact equality: hook output, hook stats, cells, controls, decisions, fitted read-outs,
     the CLI) and a stored golden produced from the original (golden/golden_speedread_modes.json).
 (C) The new code does what the PREREG says: the ego slice reads EGO_READ_KEYS only and equals the ego block of the SPEEDREAD slice; the hook hands O the banked `mlp_ego` cells and
     nothing else; `egoshuf` permutes the registered windows; `decide_sub` is the PREREG's table; every control goes RED on its own break.
 (D) The driver, end to end on a stub harness: subevals -> dumps -> subread -> SUBREAD.json, reaching PASS, HARMS and REFUSED.
SPEEDREAD_MODULE points the module under test elsewhere (the mutation pass and the RED-on-original run use it).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts"))
ORIG_PATH = os.path.join(HERE, "golden", "v8f_mini_eval_speedread_fd5ce0be.py")
ORIG_SHA = "fd5ce0bea37c3e0e608144e041f54a337fd4378b38fb312df65787bc864be4fe"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SR = _load(os.environ.get("SPEEDREAD_MODULE", os.path.join(SCRIPTS, "v8f_mini_eval_speedread.py")), "v8f_mini_eval_speedread_under_test")
ORIG = _load(ORIG_PATH, "v8f_mini_eval_speedread_pinned")

D_FEAT, N_TOK, FN = 16, 40, 10
D_EGO = 8 + 5 + 5 + 1 + 4                                               # the synthetic `limit` is 5 wide
SPEED_IDX = [34 + i for i in (6, 16, 17, 18, 19)]
CI = [i - 34 for i in SPEED_IDX]
SMALL_MLP = {"hidden": 48, "dropout": 0.0, "lr": 3e-3, "wd": 1e-4, "batch": 128, "max_epochs": 12, "patience": 4, "seed": 0}


# ------------------------------------------------------------------------------------------------ the fixture
def make_batch(B=6, seed=0, a_t=None, with_lab=True):
    g = np.random.default_rng(seed)
    ok = np.ones((B, FN), bool)
    ok[0, :8] = False
    bt = {"feats": torch.from_numpy(g.standard_normal((B, FN, N_TOK, D_FEAT)).astype(np.float16)), "frame_ok": torch.from_numpy(ok),
          "p10f": torch.from_numpy(g.standard_normal((B, FN, 8)).astype(np.float32)), "nav": torch.from_numpy(g.standard_normal((B, FN, 5)).astype(np.float32)),
          "limit": torch.from_numpy(g.standard_normal((B, 5)).astype(np.float32)), "limit_ok": torch.from_numpy(g.random(B) > 0.5),
          "a_t": torch.as_tensor(a_t if a_t is not None else [i % 3 for i in range(B)], dtype=torch.long)}
    if with_lab:
        bt["lab_t"] = {"cons": torch.from_numpy(g.standard_normal((B, 26))), "has": torch.ones(B, dtype=torch.bool)}
    return bt


def make_ro(kind="lin", ego_only=False, seed=1, offset=0.0):
    """A deterministic hand-built read-out (no fit): scene PCA 8 + ego 23 -> 5 cells (physical), or ego 23 -> 5."""
    g = np.random.default_rng(seed)
    d_in = D_EGO if ego_only else 8 + D_EGO
    W, b = g.standard_normal((d_in, 5)) * 0.5, np.linspace(6, 10, 5) + offset
    if ego_only:
        return SR_or(kind, ego_only=True, pca_mu=None, pca_V=None, xm=np.zeros(D_EGO, np.float32), xs=np.ones(D_EGO, np.float32), lin=(W, b))
    return SR_or(kind, ego_only=False, pca_mu=np.zeros(14 * D_FEAT, np.float32), pca_V=g.standard_normal((8, 14 * D_FEAT)).astype(np.float32) * 0.1,
                 xm=np.zeros(8 + D_EGO, np.float32), xs=np.ones(8 + D_EGO, np.float32), lin=(W, b))


_MOD_FOR_RO = [None]


def SR_or(kind, **kw):
    m = _MOD_FOR_RO[0] or SR
    return m.ReadOut(kind, kw.pop("pca_mu"), kw.pop("pca_V"), kw.pop("xm"), kw.pop("xs"), lin=kw.pop("lin"), ego_only=kw.pop("ego_only"))


def _fake_world(model_seed=5, speed_fn=None):
    """A stand-in for SPEEDAVG's hook base and the model, enough for the hook's own lines; `_scene` copies V8FMini._scene's pad rule verbatim."""
    class FakeModel:
        def __init__(self):
            self.cons_mu, self.cons_sd = torch.linspace(0, 2.5, 26), torch.linspace(1, 3.5, 26)
            self.hist = 4
            self.Wc = torch.from_numpy(np.random.default_rng(model_seed).standard_normal((8, 60)).astype(np.float32)) * 0.3
            self.speed_fn = speed_fn

        def forward_infer(self, batch):
            code = torch.tanh(batch["p10f"][:, -1, :] @ self.Wc)
            if self.speed_fn is not None:                                  # a stub T that knows the cells up to noise, so that a perfect read-out beats it and a shuffled one does not
                code = code.clone()
                code[:, SPEED_IDX] = ((torch.from_numpy(self.speed_fn(batch)).float() - self.cons_mu[CI]) / self.cons_sd[CI]) + 0.4 * code[:, SPEED_IDX]
            return {"code": code}

        def forward_diag(self, batch, code_override):
            return {"code_read_by_O": code_override}

        def _encode(self, batch):
            return batch["feats"].float()

        def _scene(self, adapt, ok, last):
            pos = last[:, None] - (self.hist - 1 - torch.arange(self.hist))[None]
            pad = ~torch.gather(ok, 1, pos)
            return None, pad[:, :, None].expand(-1, -1, adapt.shape[2]).flatten(1)

        def _code_gt(self, lab):
            code = torch.zeros(lab["cons"].shape[0], 60)
            code[:, SPEED_IDX] = ((lab["cons"][:, CI] - self.cons_mu[CI]) / self.cons_sd[CI]).float()
            return code

    M = types.SimpleNamespace(N_GOAL=22, N_LAT=5, N_LON=7, CODE_DIM=60, V8FMini=FakeModel, CI={})

    def replace_cells(code, src, idx):
        out = code.clone()
        out[:, idx] = src[:, idx]
        return out

    def source_code(mode, own, partner, gt, has):
        if mode == "gt":
            return torch.where(has[:, None], gt, own)
        raise ValueError(mode)

    class SpeedAvgHook:
        def __init__(self, M, mode, other, idx):
            self.M, self.mode, self.idx = M, mode, list(idx)
            self.ci = [i - 34 for i in self.idx]
            self.non = [i for i in range(60) if i not in set(self.idx)]
            self.rows = {k: [] for k in ("own_std", "partner_std", "used_std", "gt_phys", "has")}
            self.nonspeed_dev = self.oracle_dev = 0.0
            self.mu = self.sd = None
            self.n_calls, self.orig = 0, None

        def install(self):
            self.orig = self.M.V8FMini.forward_diag
            hook = self
            self.M.V8FMini.forward_diag = lambda model, batch, code_override: hook(model, batch, code_override)
            return self

        def restore(self):
            self.M.V8FMini.forward_diag = self.orig

        def stats(self):
            return {"mode": self.mode, "n_batches": self.n_calls, "nonspeed_max_abs_dev": self.nonspeed_dev, "oracle_code_max_abs_dev": self.oracle_dev if self.mode == "gt" else None}

        def cells(self):
            out = {k: np.concatenate(v) for k, v in self.rows.items()}
            out["cons_mu"], out["cons_sd"] = self.mu, self.sd
            out["speed_idx"] = np.asarray(self.idx)
            return out

    SA = types.SimpleNamespace(SpeedAvgHook=SpeedAvgHook, replace_cells=replace_cells, source_code=source_code)
    return M, SA, FakeModel()


def _carrier(model, bt, mode):
    own = model.forward_infer(bt)["code"]
    if mode == "own":
        return own
    c = own.clone()
    c[:, SPEED_IDX] = model._code_gt(bt["lab_t"])[:, SPEED_IDX]
    return c


def run_hook(mod, mode, batches, readout=None, extras=None, shuf=None):
    """Drive the module's hook over `batches` on the fake world. -> (list of codes O read, stats, cells)."""
    M, SA, model = _fake_world()
    Hook = mod.make_hook_class(SA)
    h = Hook(M, mode, SPEED_IDX, readout, extras, shuf).install() if shuf is not None else Hook(M, mode, SPEED_IDX, readout, extras).install()
    outs = []
    try:
        for bt in batches:
            outs.append(M.V8FMini.forward_diag(model, bt, _carrier(model, bt, "gt"))["code_read_by_O"].clone())
    finally:
        h.restore()
    return outs, h.stats(), h.cells()


# ------------------------------------------------------------------------------------------------ (B) golden payload and the differential
def golden_payload(mod):
    """What `mod` does on the fixture, for every SPEEDREAD mode (JSON-able). Used for the stored golden (made from ORIG) and compared live against SR."""
    _MOD_FOR_RO[0] = mod
    try:
        bts = [make_batch(6, seed=11), make_batch(4, seed=12, a_t=[2, 1, 0, 2])]
        ro, alt = make_ro("lin", seed=1), make_ro("mlp", seed=2)
        extras = {"read": ro, "alt": alt, "lin_ego": make_ro("lin_ego", ego_only=True, seed=3), "mlp_ego": make_ro("mlp_ego", ego_only=True, seed=4)}
        out = {}
        for mode in mod.MODES:
            r = None if mode in ("own", "identity", "gt") else (ro if mode in ("read", "perm") else alt)
            codes, st, cells = run_hook(mod, mode, bts, r, extras if mode == "identity" else None) if mode != "own" else (None, None, None)
            if mode == "own":
                continue
            out[mode] = {"codes": [c.numpy().astype(np.float64).tolist() for c in codes], "stats": json.loads(json.dumps(st, default=float)),
                         "cells": {k: np.asarray(v, np.float64).tolist() for k, v in cells.items()}}
        sc, eg = mod.window_vectors(bts[0], (4, 10))
        out["window_vectors"] = {"scene": sc.astype(np.float64).tolist(), "ego": eg.astype(np.float64).tolist()}
        out["decide"] = [mod.decide({0: {"read": a, "alt": b}, 1: {"read": a, "alt": b}}, mod.F_M) for a, b in
                         ((_r(-.2, -.3, -.1), _r(-.02, -.06, .02)), (_r(-.05, -.08, -.01), _r(-.02, -.06, .02)), (_r(-.02, -.06, .02), _r(-.02, -.06, .02)),
                          (_r(-.02, -.06, .02), _r(-.05, -.08, -.01)), (_r(.2, .1, .3), _r(.2, .1, .3)))]
        out["constants"] = {"F_M": mod.F_M, "MODES": list(mod.MODES), "READ_KEYS": list(mod.READ_KEYS), "TRAIN_PARITY": mod.TRAIN_PARITY, "PERM_SEED": mod.PERM_SEED, "PCA_SEED": mod.PCA_SEED,
                            "LAMBDAS": list(mod.LAMBDAS), "MLP_CFG": mod.MLP_CFG, "SCHEMA": mod.SCHEMA}
        return out
    finally:
        _MOD_FOR_RO[0] = None


def _r(d, lo, hi):
    return {"delta": d, "lo": lo, "hi": hi}


def _canon(o):
    """dict keys to str (the controls dict mixes int seeds and str names), recursively."""
    if isinstance(o, dict):
        return {str(k): _canon(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_canon(v) for v in o]
    return o


def _close(a, b, atol):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=0, atol=atol, equal_nan=True)


def test_golden_reference_is_the_pinned_speedread_instrument():
    assert hashlib.sha256(open(ORIG_PATH, "rb").read()).hexdigest() == ORIG_SHA


def test_existing_modes_match_the_stored_golden():
    g = json.load(open(os.path.join(HERE, "golden", "golden_speedread_modes.json"), encoding="utf-8"))
    now = golden_payload(SR)
    assert set(now) == set(g)
    for mode in ORIG.MODES:
        if mode == "own":
            continue
        for c1, c2 in zip(g[mode]["codes"], now[mode]["codes"]):
            _close(c1, c2, 1e-6)
        assert g[mode]["stats"] == now[mode]["stats"], mode
        assert set(g[mode]["cells"]) == set(now[mode]["cells"])
        for k in g[mode]["cells"]:
            _close(g[mode]["cells"][k], now[mode]["cells"][k], 1e-6)
    _close(g["window_vectors"]["scene"], now["window_vectors"]["scene"], 1e-6)
    _close(g["window_vectors"]["ego"], now["window_vectors"]["ego"], 0.0)
    assert g["decide"] == now["decide"] and g["constants"] == now["constants"]


def test_existing_modes_identical_to_the_original_in_the_same_process():
    a, b = golden_payload(ORIG), golden_payload(SR)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _synth(n_clips=24, per=30, d_raw=40, seed=0):
    g = np.random.default_rng(seed)
    n = n_clips * per
    clips = np.repeat([f"clip-{seed}-{i:03d}" for i in range(n_clips)], per)
    S = g.standard_normal((n, d_raw)).astype(np.float32)
    S[:, :5] *= 3.0
    E = g.standard_normal((n, 23)).astype(np.float32)
    Y = 5.0 + S[:, :5] / 3.0 + 0.5 * E[:, :5] + 0.2 * g.standard_normal((n, 5))
    return S, E, Y, g.random((n, 5)) > 0.05, clips


def test_existing_fit_and_readouts_identical_to_the_original(tmp_path):
    S, E, Y, M, clips = _synth()
    res = {}
    for nm, mod in (("orig", ORIG), ("sr", SR)):
        res[nm] = mod.fit_all(S[::2], lambda r: S[r], E, Y, M, clips, pca_k=12, mlp_cfg=SMALL_MLP, log=lambda *a: None)
    for k in res["orig"]["readouts"]:
        np.testing.assert_array_equal(res["orig"]["readouts"][k].predict_raw(S, E), res["sr"]["readouts"][k].predict_raw(S, E), err_msg=k)
    assert json.dumps(res["orig"]["record"], sort_keys=True, default=float) == json.dumps(res["sr"]["record"], sort_keys=True, default=float)
    p0, p1 = str(tmp_path / "o.pt"), str(tmp_path / "s.pt")
    ORIG.save_readouts(res["orig"]["readouts"], p0)
    SR.save_readouts(res["sr"]["readouts"], p1)
    lo, ls = ORIG.load_readouts(p0), SR.load_readouts(p1)
    for k in lo:
        np.testing.assert_array_equal(lo[k].predict_raw(S, E), ls[k].predict_raw(S, E), err_msg=k)


def _fake_ev(mod, n=50, seed=0):
    g = np.random.default_rng(seed)
    eid = np.repeat(np.arange(10), 5)
    own = g.random(n) + 2.0
    ev = {}
    for m in mod.MODES:
        ev[m] = {"eid": eid.copy(), "ade6": own.copy() if m in ("own", "identity") else own - 0.3, "pred": np.zeros((n, 30, 2)), "hook": {"nonspeed_max_abs_dev": 0.0, "tick_mirror_ok": True}}
    mu, sd = np.linspace(1, 5, 5), np.full(5, 2.0)
    pr, gt = g.standard_normal((n, 5)) + 8, g.standard_normal((n, 5)) + 8
    gt[3, 1] = np.nan
    ev["identity"]["cells"] = {"cons_mu": mu.astype(np.float32), "cons_sd": sd.astype(np.float32), "pred_read": pr, "pred_const": np.full((n, 5), 7.5), "pred_mlp_ego": pr * 0.9,
                               "own_std": np.zeros((n, 5), np.float32), "gt_phys": gt.astype(np.float32), "has": np.ones(n, bool)}
    ev["read"]["cells"] = {"used_std": ((pr - mu) / sd).astype(np.float32)}
    return ev


def test_existing_controls_decisions_rows_identical_to_the_original():
    fitrec = {"perm_holdout_skill_max": 0.0, "train_parity_ok": True, "eval_train_disjoint": True, "const": [7.5] * 5}
    paired = lambda a, b, eid, n: _r(float(np.mean(a) - np.mean(b)), float(np.mean(a) - np.mean(b)) - 0.01, float(np.mean(a) - np.mean(b)) + 0.01)
    outs = []
    for mod in (ORIG, SR):
        ev = {0: _fake_ev(mod), 1: _fake_ev(mod, seed=1)}
        c = mod.controls(ev, None, fitrec)
        outs.append(json.dumps(_canon({"c": c, "ok": mod.all_pass(c), "rows": mod.read_rows(ev, paired, 10), "cells": mod.cell_table(ev[0]["identity"]["cells"]),
                                       "dec": mod.decide({s: mod.read_rows(ev, paired, 10)[s] for s in (0, 1)}, mod.F_M, True)}), sort_keys=True, default=float))
    assert outs[0] == outs[1]


def test_existing_cli_unchanged():
    def sig(mod):
        ap = mod.build_parser()
        sp = [a for a in ap._actions if isinstance(a, __import__("argparse")._SubParsersAction)][0]
        return {n: sorted((tuple(x.option_strings), x.required, repr(x.default)) for x in p._actions) for n, p in sp.choices.items() if n in ("extract", "fit", "evals", "read")}
    assert sig(ORIG) == sig(SR)
    assert ORIG.MODES == SR.MODES == ("own", "identity", "gt", "read", "alt", "const", "perm")
    assert set(dir(ORIG)) <= set(dir(SR))


# ------------------------------------------------------------------------------------------------ (C) the ego slice, the read-out path, the hook
@pytest.mark.parametrize("a_t", [[0, 1, 2, 0, 1, 2], [2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0]])
def test_ego_slice_equals_the_ego_block_of_the_speedread_slice(a_t):
    bt = make_batch(seed=3, a_t=a_t)
    _, eg = SR.window_vectors(bt, (4, 10))
    np.testing.assert_array_equal(SR.ego_vectors(bt), eg)
    mini = {k: bt[k] for k in SR.EGO_READ_KEYS}
    np.testing.assert_array_equal(SR.ego_vectors(mini), eg)                   # a batch WITHOUT feats / lab_t is enough
    i, e = SR.t_tick_inputs(bt), SR.t_tick_ego(mini)
    for k in ("ok", "p10f", "nav", "limit", "limit_ok"):
        np.testing.assert_array_equal(i[k], e[k], err_msg=k)


def test_ego_slice_reads_only_ego_keys_and_only_the_tick_window():
    assert set(SR.EGO_READ_KEYS) <= set(SR.READ_KEYS) and "feats" not in SR.EGO_READ_KEYS and "lab_t" not in SR.EGO_READ_KEYS
    bt = make_batch(seed=4)
    mini = {k: bt[k] for k in SR.EGO_READ_KEYS}
    base = SR.ego_vectors(mini)
    with pytest.raises(KeyError):
        SR.ego_vectors({k: v for k, v in mini.items() if k != "nav"})
    bt2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in mini.items()}
    for b in range(6):
        last = FN - 1 - int(bt["a_t"][b])
        bt2["p10f"][b, last + 1:] = 99.0
        bt2["nav"][b, :last] = 99.0
    np.testing.assert_array_equal(SR.ego_vectors(bt2), base)
    with pytest.raises(ValueError):
        SR.t_tick_ego({**mini, "frame_ok": mini["frame_ok"][:, :3], "a_t": torch.zeros(6, dtype=torch.long)}, hist=4)


def test_ego_batch_prediction_equals_the_scene_path_and_never_sees_a_pixel():
    bt = make_batch(seed=5)
    for ro in (make_ro("mlp_ego", ego_only=True, seed=7), make_ro("lin_ego", ego_only=True, seed=8)):
        np.testing.assert_array_equal(ro.predict_ego_batch(bt), ro.predict_batch(bt))
        poisoned = dict(bt, feats=torch.full_like(bt["feats"], 6e4), lab_t={"cons": torch.full((6, 26), float("nan")), "has": torch.zeros(6, dtype=torch.bool)})
        np.testing.assert_array_equal(ro.predict_ego_batch(poisoned), ro.predict_ego_batch(bt))
        np.testing.assert_array_equal(ro.predict_ego_batch({k: bt[k] for k in SR.EGO_READ_KEYS}), ro.predict_ego_batch(bt))
    with pytest.raises(ValueError):
        make_ro("lin", seed=1).predict_ego_batch(bt)                             # a scene read-out has no ego-only path


def test_hook_ego_hands_O_the_banked_ego_cells_and_nothing_else():
    bts = [make_batch(6, seed=21), make_batch(3, seed=22, a_t=[1, 2, 0])]
    ro = make_ro("mlp_ego", ego_only=True, seed=9)
    outs, st, cells = run_hook(SR, "ego", bts, ro)
    M, SA, model = _fake_world()
    mu, sd = model.cons_mu[CI].double().numpy(), model.cons_sd[CI].double().numpy()
    for bt, out in zip(bts, outs):
        own = model.forward_infer(bt)["code"]
        non = [i for i in range(60) if i not in SPEED_IDX]
        assert torch.equal(out[:, non], own[:, non])
        np.testing.assert_allclose(out[:, SPEED_IDX].double().numpy() * sd + mu, ro.predict_batch(bt), atol=1e-5)
        assert not torch.equal(out[:, SPEED_IDX], own[:, SPEED_IDX])
    assert st["nonspeed_max_abs_dev"] == 0.0 and st["tick_mirror_ok"] is True and "shuffle" not in st
    assert cells["used_std"].shape == (9, 5)


def test_hook_ego_mirror_catches_a_bad_ego_slice(monkeypatch):
    bts = [make_batch(6, seed=23)]
    ro = make_ro("mlp_ego", ego_only=True, seed=9)
    _, st, _ = run_hook(SR, "ego", bts, ro)
    assert st["tick_mirror_ok"] is True
    real = SR.t_tick_ego
    monkeypatch.setattr(SR, "t_tick_ego", lambda b, hist=SR.HIST: dict(real(b, hist), ok=~real(b, hist)["ok"]))
    _, st2, _ = run_hook(SR, "ego", bts, ro)
    assert st2["tick_mirror_ok"] is False
    _, st3, _ = run_hook(SR, "identity", bts)                                   # the SPEEDREAD modes do not consult the ego slice
    assert st3["tick_mirror_ok"] is True


def test_ego_mode_reads_a_batch_that_has_no_pixels_and_no_label():
    bt = make_batch(seed=6)
    ro = make_ro("mlp_ego", ego_only=True, seed=9)
    mu, sd = np.linspace(0, 2, 5), np.linspace(1, 3, 5)
    mini = {k: bt[k] for k in SR.EGO_READ_KEYS}
    got = SR.new_speed_std("ego", ro, mini, torch.zeros(6, 60), SPEED_IDX, mu, sd)
    np.testing.assert_array_equal(got, SR.phys_to_std(ro.predict_batch(bt), mu, sd))
    assert got.shape == (6, 5)


def test_ego_mode_without_a_readout_is_refused():
    with pytest.raises(ValueError):
        SR.new_speed_std("ego", None, make_batch(), torch.zeros(6, 60), SPEED_IDX, np.zeros(5), np.ones(5))


def _shuf_world(n_batches=(6, 4, 3), seed=31):
    bts = [make_batch(B, seed=seed + i, a_t=[(j + i) % 3 for j in range(B)]) for i, B in enumerate(n_batches)]
    ro = make_ro("mlp_ego", ego_only=True, seed=9)
    rows = np.concatenate([ro.predict_batch(b) for b in bts])
    return bts, ro, rows


def test_egoshuf_hands_window_i_the_cells_of_window_perm_i_across_batches():
    bts, ro, rows = _shuf_world()
    n = len(rows)
    perm = SR.shuffle_perm(n)
    assert sorted(perm.tolist()) == list(range(n)) and not np.array_equal(perm, np.arange(n))
    assert np.array_equal(perm, np.random.default_rng(SR.SHUF_SEED).permutation(n)) and SR.SHUF_SEED == 20261029
    outs, st, cells = run_hook(SR, "egoshuf", bts, ro, shuf={"rows": rows, "perm": perm})
    M, SA, model = _fake_world()
    mu, sd = model.cons_mu[CI].double().numpy(), model.cons_sd[CI].double().numpy()
    got = np.concatenate([o[:, SPEED_IDX].double().numpy() * sd + mu for o in outs])
    np.testing.assert_allclose(got, rows[perm], atol=1e-5)
    s = st["shuffle"]
    assert s["alignment_max_abs_dev"] == 0.0 and s["n_seen"] == n and s["n"] == n and s["seed"] == SR.SHUF_SEED
    assert s["n_fixed"] == int(np.sum(perm == np.arange(n))) and s["perm_sha256"] == hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()


def test_egoshuf_alignment_check_sees_a_source_made_from_other_windows_and_an_overrun_raises():
    bts, ro, rows = _shuf_world()
    bad = rows.copy()
    bad[5] += 0.5
    _, st, _ = run_hook(SR, "egoshuf", bts, ro, shuf={"rows": bad, "perm": SR.shuffle_perm(len(rows))})
    assert st["shuffle"]["alignment_max_abs_dev"] >= 0.5
    with pytest.raises(ValueError):
        run_hook(SR, "egoshuf", bts, ro, shuf={"rows": rows[:-1], "perm": SR.shuffle_perm(len(rows) - 1)})


def test_decision_constants_are_the_preregistered_ones():
    assert SR.F_M == 0.1207 and SR.F_SUB_PASS == pytest.approx(0.06035, abs=1e-12) and SR.F_SUB_PASS == SR.F_M / 2.0
    assert SR.SUB_MODES == ("own", "identity", "gt", "ego", "egoshuf") and SR.MODE_READOUT == {"ego": "mlp_ego", "egoshuf": "mlp_ego"}
    assert SR.EGO_READ_KEYS == ("p10f", "nav", "limit", "limit_ok", "frame_ok", "a_t")
    assert (SR.TOL_OWN_BANKED, SR.TOL_GT_BANKED, SR.TOL_GT_DELTA_4DP) == (1e-5, 1e-4, 5e-4)


# ---- decide_sub
FP = getattr(SR, "F_SUB_PASS", 0.1207 / 2.0)                                # the original has none: its ego tests then fail one by one, they do not abort the collection
REG = _r(5.3, 4.0, 6.7)                                                   # the regression arm, separated-worse
BIG, EDGE, SMALL, NS = _r(-0.20, -0.30, -0.10), _r(-FP, -0.12, -0.001), _r(-0.03, -0.06, -0.001), _r(-0.01, -0.07, 0.05)


def rows_of(e0, e1, g0=REG, g1=REG):
    return {0: {"ego": e0, "egoshuf": g0}, 1: {"ego": e1, "egoshuf": g1}}


@pytest.mark.parametrize("rows,want", [
    (rows_of(BIG, BIG), "PASS"),
    (rows_of(EDGE, BIG), "PASS"),                                           # exactly -F/2 passes
    (rows_of(_r(-FP + 1e-9, -0.12, -0.001), BIG), "NULL"),                  # a hair above the bar does not
    (rows_of(_r(-0.20, -0.30, 0.0), BIG), "NULL"),                          # hi == 0.0 is not separated-better
    (rows_of(SMALL, SMALL), "NULL"),                                        # separated-better at both, below the bar
    (rows_of(BIG, NS), "NULL"),                                             # one seed only
    (rows_of(NS, NS), "NULL"),
    (rows_of(BIG, _r(0.04, 0.0, 0.09)), "NULL"),                            # lo == 0.0 is not separated-worse
    (rows_of(BIG, _r(0.04, 0.01, 0.09)), "HARMS"),                          # separated-worse at ONE seed, the other a PASS-grade win: HARMS
    (rows_of(_r(0.1, 0.02, 0.2), _r(0.1, 0.02, 0.2)), "HARMS"),
    (rows_of(BIG, BIG, g0=_r(0.3, -0.1, 0.5)), "REFUSED"),                  # the regression arm must read separated-worse at both seeds
    (rows_of(BIG, BIG, g1=_r(0.3, 0.0, 0.5)), "REFUSED"),
    ({0: {"ego": BIG, "egoshuf": REG}}, "REFUSED"),
])
def test_decide_sub_table(rows, want):
    assert SR.decide_sub(rows, FP) == want


def test_decide_sub_refuses_on_a_failed_control_and_details_the_null():
    assert SR.decide_sub(rows_of(BIG, BIG), FP, controls_ok=False) == "REFUSED"
    assert SR.null_detail(rows_of(SMALL, SMALL)).startswith("NULL-SMALL")
    assert SR.null_detail(rows_of(BIG, NS)).startswith("NULL-ONE")
    assert SR.null_detail(rows_of(NS, NS)).startswith("NULL-FLAT")
    assert set(SR.NEXT_SUB) == {"PASS", "HARMS", "NULL", "REFUSED"} and "SPEEDCELL" in SR.NEXT_SUB["NULL"] and "T1-kin" in SR.NEXT_SUB["PASS"]


# ---- controls_sub
def _sub_ev(seed=0, n=50):
    g = np.random.default_rng(seed)
    eid = np.repeat(np.arange(10), 5)
    own = g.random(n) + 2.0
    mu, sd = np.linspace(1, 5, 5), np.full(5, 2.0)
    pe = g.standard_normal((n, 5)) + 8
    perm = SR.shuffle_perm(n)
    sha = "a" * 64
    ev = {}
    for m in SR.SUB_MODES:
        ev[m] = {"eid": eid.copy(), "ade6": own.copy() if m in ("own", "identity") else own - 0.3, "pred": np.zeros((n, 30, 2)), "readout_sha256": sha,
                 "hook": {"nonspeed_max_abs_dev": 0.0, "tick_mirror_ok": True}}
    ev["identity"]["cells"] = {"cons_mu": mu.astype(np.float32), "cons_sd": sd.astype(np.float32), "pred_mlp_ego": pe, "pred_const": np.full((n, 5), 7.5)}
    ev["ego"]["cells"] = {"used_std": ((pe - mu) / sd).astype(np.float32), "own_std": np.zeros((n, 5), np.float32)}
    ev["egoshuf"]["cells"] = {"used_std": ((pe[perm] - mu) / sd).astype(np.float32), "own_std": np.zeros((n, 5), np.float32)}
    ev["egoshuf"]["hook"]["shuffle"] = {"seed": SR.SHUF_SEED, "n": n, "n_seen": n, "n_fixed": 0, "alignment_max_abs_dev": 0.0, "perm_sha256": hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()}
    banked = {"own": own.copy(), "gt": own - 0.3, "pred_mlp_ego": pe.copy()}
    return ev, banked


FITREC = {"train_parity_ok": True, "eval_train_disjoint": True, "readout_sha256": "a" * 64,
          "lin": {"holdout_rmse_sd": [.3] * 5}, "mlp": {"holdout_rmse_sd": [.31] * 5}, "lin_ego": {"holdout_rmse_sd": [.29] * 5}, "mlp_ego": {"holdout_rmse_sd": [.25] * 5}}
SHA = "a" * 64


def _ctl(ev, banked, fitrec=FITREC, sha=SHA):
    return SR.controls_sub(ev, banked, fitrec, sha)


def test_controls_sub_pass_on_clean_and_fail_on_each_break():
    e0, b0 = _sub_ev(0)
    e1, b1 = _sub_ev(1)
    assert SR.all_pass(_ctl({0: e0, 1: e1}, {0: b0, 1: b1}))
    breaks = ["identity_ade", "identity_pred", "nonspeed", "mirror", "eid", "own_drift", "gt_drift", "ego_path", "bank_pred", "noop_sub", "shuf_equal_ego", "sha_dump", "sha_expected",
              "shuf_align", "shuf_seen", "shuf_perm", "shuf_fixed", "no_banked", "no_cells", "parity", "disjoint", "selection"]
    for brk in breaks:
        ev, bk, fr, sha = {0: _sub_ev(0)[0], 1: _sub_ev(1)[0]}, {0: _sub_ev(0)[1], 1: _sub_ev(1)[1]}, dict(FITREC), SHA
        if brk == "identity_ade":
            ev[1]["identity"]["ade6"] = ev[1]["identity"]["ade6"] + 1e-9
        elif brk == "identity_pred":
            ev[1]["identity"]["pred"] = ev[1]["identity"]["pred"] + 1e-9
        elif brk == "nonspeed":
            ev[0]["gt"]["hook"]["nonspeed_max_abs_dev"] = 1e-7
        elif brk == "mirror":
            ev[0]["ego"]["hook"]["tick_mirror_ok"] = False
        elif brk == "eid":
            ev[1]["egoshuf"]["eid"] = ev[1]["egoshuf"]["eid"][::-1].copy()
        elif brk == "own_drift":
            ev[0]["own"]["ade6"] = ev[0]["own"]["ade6"] + 2e-5
        elif brk == "gt_drift":
            ev[0]["gt"]["ade6"] = ev[0]["gt"]["ade6"] + 2e-4
        elif brk == "ego_path":
            ev[0]["ego"]["cells"]["used_std"][7, 2] += 1e-3
        elif brk == "bank_pred":
            bk[1]["pred_mlp_ego"][4, 1] += 1e-3
        elif brk == "noop_sub":                                            # the substitution leaves O the model's own cells: S7 still holds, S10 must not
            ev[0]["ego"]["cells"]["own_std"] = ev[0]["ego"]["cells"]["used_std"].copy()
        elif brk == "shuf_equal_ego":
            ev[1]["egoshuf"]["cells"]["used_std"] = ev[1]["ego"]["cells"]["used_std"].copy()
        elif brk == "sha_dump":
            ev[1]["ego"]["readout_sha256"] = "b" * 64
        elif brk == "sha_expected":
            sha = "b" * 64
        elif brk == "shuf_align":
            ev[0]["egoshuf"]["hook"]["shuffle"]["alignment_max_abs_dev"] = 1e-3
        elif brk == "shuf_seen":
            ev[0]["egoshuf"]["hook"]["shuffle"]["n_seen"] = 49
        elif brk == "shuf_perm":
            ev[1]["egoshuf"]["hook"]["shuffle"]["perm_sha256"] = "0" * 64
        elif brk == "shuf_fixed":
            ev[1]["egoshuf"]["hook"]["shuffle"]["n_fixed"] = 25
        elif brk == "no_banked":
            bk = None
        elif brk == "no_cells":
            del ev[1]["ego"]["cells"]
        elif brk == "parity":
            fr["train_parity_ok"] = False
        elif brk == "disjoint":
            fr["eval_train_disjoint"] = False
        elif brk == "selection":
            fr["mlp_ego"] = {"holdout_rmse_sd": [.35] * 5}
        assert not SR.all_pass(SR.controls_sub(ev, bk, fr, sha)), brk


# ------------------------------------------------------------------------------------------------ (D) the driver, end to end on a stub harness
def _boot(a, b, eid, n_boot=200):
    a, b, eid = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(eid)
    eps = np.unique(eid)
    d = a - b
    per = np.array([d[eid == e].mean() for e in eps])
    w = np.array([np.sum(eid == e) for e in eps])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(200):
        i = rng.integers(0, len(eps), len(eps))
        bs.append(float(np.sum(per[i] * w[i]) / np.sum(w[i])))
    return {"delta": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


class StubHarness:
    """Stands in for `v8f_mini_eval.main`: runs the fake model over fixed batches through the (hooked) forward_diag and writes the eval dump the instrument reads."""

    def __init__(self, M, model, batches):
        self.M, self.model, self.batches = M, model, batches

    def main(self, argv):
        tag, out, carrier = argv[argv.index("--tag") + 1], argv[argv.index("--out-dir") + 1], argv[argv.index("--carrier") + 1]
        ade, eid, k = [], [], 0
        for bt in self.batches:
            res = self.M.V8FMini.forward_diag(self.model, bt, _carrier(self.model, bt, "own" if carrier == "own" else "gt"))["code_read_by_O"]
            gts = self.model._code_gt(bt["lab_t"])[:, SPEED_IDX]
            ade.append((1.0 + (res[:, SPEED_IDX] - gts).abs().mean(1) * 2.0).double().numpy())
            eid.append(np.arange(k, k + len(bt["a_t"])) // 5)
            k += len(bt["a_t"])
        ade, eid = np.concatenate(ade), np.concatenate(eid)
        pred = np.zeros((len(ade), 30, 2))
        pred[:, 0, 0] = ade
        np.savez(os.path.join(out, f"eval_{tag}.npz"), eid=eid, ade6=ade, pred=pred)
        json.dump({"raw": {"tier": "T0-open-loop-own-held-outputs (trunk-probe grid)"}}, open(os.path.join(out, f"eval_{tag}.json"), "w", encoding="utf-8"))
        return 0


def _pipeline(tmp_path, monkeypatch, mod, world="pass", tamper=None):
    """subevals for both seeds -> subread. world: pass (ego = GT exactly), harms (ego = GT + 4)."""
    _MOD_FOR_RO[0] = mod
    try:
        bts = [make_batch(B, seed=60 + i, a_t=[(j + i) % 3 for j in range(B)]) for i, B in enumerate((8, 8, 8, 8, 8))]
        ro = make_ro("mlp_ego", ego_only=True, seed=9, offset=0.0 if world == "pass" else -20.0)
        for bt in bts:                                                         # GT cells = the ego read-out's own cells (+ nothing): a perfect read-out
            bt["lab_t"]["cons"][:, CI] = torch.from_numpy(make_ro("mlp_ego", ego_only=True, seed=9).predict_batch(bt))
        const = mod.ReadOut("const", None, None, None, None, const=np.full(5, 8.0))
        d = tmp_path / world
        d.mkdir(parents=True, exist_ok=True)
        rp = str(d / "readout.pt")
        mod.save_readouts({"mlp_ego": ro, "lin_ego": make_ro("lin_ego", ego_only=True, seed=10, offset=3.0), "const": const}, rp)
        sha = mod.sha256_file(rp)
        fit = dict(FITREC, readout_sha256=sha)
        json.dump(fit, open(d / "fit.json", "w", encoding="utf-8"))
        xp = str(d / "ckpt.pt")
        torch.save({"cfg": {"hist": 4}}, xp)
        M, SA, model = _fake_world(speed_fn=lambda b: make_ro("mlp_ego", ego_only=True, seed=9).predict_batch(b))
        E = StubHarness(M, model, bts)
        SA.speed_idx = lambda Mm: list(SPEED_IDX)
        SA.paths_of = lambda a: {"out": a.eval_dir}
        SA.extra_of = lambda a: []
        SA.e_argv = lambda mode, ckpt, tag, paths, extra: ["--carrier", mode, "--tag", tag, "--out-dir", paths["out"]] + list(extra)
        SA._load_modules = lambda: (M, E)
        monkeypatch.setitem(sys.modules, "v8f_mini_eval_speedavg", SA)
        dirs = {}
        for s in (0, 1):
            dirs[s] = str(d / f"eval_s{s}")
            os.makedirs(dirs[s], exist_ok=True)
            a = types.SimpleNamespace(seed=s, x=xp, expect_sha=mod.sha256_file(xp), readout=rp, fit_json=str(d / "fit.json"), expect_readout_sha=sha if tamper != "sha" else "c" * 64,
                                      eval_dir=dirs[s], eval_cache="", tracks="", v9_eval="", lead_eval="", speed_eval="", p10="", eval_grid=None, eval_kin=None, eval_join=None,
                                      n_boot=10, batch=8, lru=1, device="cpu")
            if tamper == "sha":
                with pytest.raises(SystemExit):
                    mod.cmd_subevals(a)
                return None, None
            assert mod.cmd_subevals(a) in (0, None)
        # SPEEDREAD's banked dumps, made from this run's own / gt / identity dumps; its READ.json from the same stub bootstrap
        bd = {}
        rj = {"rows": {}}
        for s in (0, 1):
            bd[s] = str(d / f"banked_s{s}")
            os.makedirs(bd[s], exist_ok=True)
            for m in ("own", "gt"):
                z = np.load(os.path.join(dirs[s], f"eval_M2b-mixo_s{s}-speedsub-{m}.npz"))
                np.savez(os.path.join(bd[s], f"eval_M2b-mixo_s{s}-speedread-{m}.npz"), ade6=z["ade6"], eid=z["eid"])
            ci = np.load(os.path.join(dirs[s], f"speedread_cells_M2b-mixo_s{s}-speedsub-identity.npz"))
            np.savez(os.path.join(bd[s], f"speedread_cells_M2b-mixo_s{s}-speedread-identity.npz"), pred_mlp_ego=ci["pred_mlp_ego"])
            zo, zg = (np.load(os.path.join(dirs[s], f"eval_M2b-mixo_s{s}-speedsub-{m}.npz")) for m in ("own", "gt"))
            rj["rows"][str(s)] = {"gt": _boot(zg["ade6"], zo["ade6"], zo["eid"])}
            if tamper == "readjson":                                          # a banked gt - own row that this run does not reproduce
                rj["rows"][str(s)]["gt"]["delta"] += 0.01
        json.dump(rj, open(d / "READ.json", "w", encoding="utf-8"))
        monkeypatch.setattr(mod, "taniteval_paired", lambda: _boot)
        ns = types.SimpleNamespace(fit_json=str(d / "fit.json"), eval_dir_s0=dirs[0], eval_dir_s1=dirs[1], speedread_dir_s0=bd[0], speedread_dir_s1=bd[1],
                                   speedread_read_json=str(d / "READ.json"), expect_readout_sha=sha, n_boot=200, out=str(d / "out"))
        rc = mod.cmd_subread(ns)
        return rc, json.load(open(d / "out" / "SUBREAD.json", encoding="utf-8"))
    finally:
        _MOD_FOR_RO[0] = None


def test_pipeline_reaches_PASS_when_the_readout_is_perfect(tmp_path, monkeypatch):
    rc, rep = _pipeline(tmp_path, monkeypatch, SR, "pass")
    assert rc == 0 and rep["outcome"] == "PASS" and SR.all_pass(rep["controls"]), rep["controls"]
    for s in ("0", "1"):
        r = rep["rows"][s]
        assert r["ego"]["hi"] < 0 and r["ego"]["delta"] <= -SR.F_SUB_PASS and r["egoshuf"]["lo"] > 0 and r["regression_arm_separated_worse"] is True
    assert rep["schema"] == SR.SUB_SCHEMA and rep["claim"] == "H-V8F-SPEEDSUB-1" and rep["selected_readout"] == "mlp_ego"
    assert open(tmp_path / "pass" / "out" / "OUTCOME.txt").read().splitlines()[0] == "PASS"


def test_pipeline_reaches_HARMS_when_the_readout_is_off(tmp_path, monkeypatch):
    rc, rep = _pipeline(tmp_path, monkeypatch, SR, "harms")
    assert rc == 0 and rep["outcome"] == "HARMS", (rep["outcome"], rep["controls"], rep["rows"])
    assert all(rep["rows"][s]["ego"]["lo"] > 0 for s in ("0", "1"))


def test_pipeline_refuses_a_read_out_that_is_not_the_pinned_one(tmp_path, monkeypatch):
    _pipeline(tmp_path, monkeypatch, SR, "pass", tamper="sha")


def test_pipeline_refuses_when_gt_minus_own_does_not_reproduce_speedreads_row(tmp_path, monkeypatch):
    rc, rep = _pipeline(tmp_path, monkeypatch, SR, "pass", tamper="readjson")
    assert rc == 2 and rep["outcome"] == "REFUSED" and rep["controls"]["S6b_gt_delta_eq_speedread_read_json"] == {"0": False, "1": False}


def test_pipeline_tags_and_hook_blocks_of_the_sub_flow(tmp_path, monkeypatch):
    _pipeline(tmp_path, monkeypatch, SR, "pass")
    d = tmp_path / "pass" / "eval_s0"
    names = sorted(os.listdir(d))
    assert all("-speedsub-" in n for n in names) and not any("-speedread-" in n for n in names)
    js = {m: json.load(open(d / f"eval_M2b-mixo_s0-speedsub-{m}.json", encoding="utf-8"))["speedread"] for m in SR.SUB_MODES}
    assert js["ego"]["readout_key"] == "mlp_ego" and js["egoshuf"]["readout_key"] == "mlp_ego" and js["ego"]["flow"] == SR.SUB_SCHEMA
    assert js["egoshuf"]["tier"] == SR.TIER_REG and js["gt"]["tier"] == SR.TIER_T0 and js["ego"]["tier"] == SR.TIER_ARM
    assert js["own"]["hook"] is None and js["egoshuf"]["hook"]["shuffle"]["n"] == 40
    z = np.load(d / "speedread_cells_M2b-mixo_s0-speedsub-identity.npz")
    assert set(k for k in z.files if k.startswith("pred_")) == {"pred_mlp_ego", "pred_const"}


# ---- (A) RED on the unmodified instrument, GREEN on this one
def test_ego_is_RED_on_the_unmodified_instrument_and_GREEN_on_this_one(tmp_path, monkeypatch):
    for name in ("SUB_MODES", "EGO_MODES", "decide_sub", "controls_sub", "cmd_subevals", "cmd_subread", "t_tick_ego", "ego_vectors", "shuffle_perm", "F_SUB_PASS"):
        assert not hasattr(ORIG, name), name
        assert hasattr(SR, name), name
    # the original's driver runs only its seven SPEEDREAD modes: it never produces an `ego` dump
    for nm, mod in (("orig", ORIG), ("new", SR)):
        _MOD_FOR_RO[0] = mod
        try:
            bts = [make_batch(8, seed=70 + i) for i in range(2)]
            ro = make_ro("mlp_ego", ego_only=True, seed=9)
            for bt in bts:
                bt["lab_t"]["cons"][:, CI] = torch.from_numpy(ro.predict_batch(bt))
            d = tmp_path / nm
            d.mkdir()
            rp = str(d / "readout.pt")
            const = mod.ReadOut("const", None, None, None, None, const=np.full(5, 8.0))
            alls = {k: ro for k in ("read", "alt", "lin", "mlp", "lin_ego", "mlp_ego", "perm")}
            mod.save_readouts(dict(alls, const=const), rp)
            json.dump({"readout_sha256": mod.sha256_file(rp)}, open(d / "fit.json", "w", encoding="utf-8"))
            xp = str(d / "ckpt.pt")
            torch.save({"cfg": {"hist": 4}}, xp)
            M, SA, model = _fake_world()
            E = StubHarness(M, model, bts)
            SA.speed_idx, SA.paths_of, SA.extra_of = (lambda Mm: list(SPEED_IDX)), (lambda a: {"out": a.eval_dir}), (lambda a: [])
            SA.e_argv = lambda mode, ckpt, tag, paths, extra: ["--carrier", mode, "--tag", tag, "--out-dir", paths["out"]]
            SA._load_modules = lambda: (M, E)
            monkeypatch.setitem(sys.modules, "v8f_mini_eval_speedavg", SA)
            (d / "ev").mkdir()
            a = types.SimpleNamespace(seed=0, x=xp, expect_sha=mod.sha256_file(xp), readout=rp, fit_json=str(d / "fit.json"), expect_readout_sha=mod.sha256_file(rp), eval_dir=str(d / "ev"), eval_cache="",
                                      tracks="", v9_eval="", lead_eval="", speed_eval="", p10="", eval_grid=None, eval_kin=None, eval_join=None, n_boot=10, batch=8, lru=1, device="cpu")
            if nm == "orig":
                a.modes, a.flow = ["ego"], "sub"                               # the original ignores both: it has no such options
                mod.cmd_evals(a)
                tags = sorted(f for f in os.listdir(d / "ev") if f.endswith(".json"))
                assert len(tags) == 7 and not any("ego" in t for t in tags) and all("-speedread-" in t for t in tags)
            else:
                mod.cmd_subevals(a)
                tags = sorted(f for f in os.listdir(d / "ev") if f.endswith(".json"))
                assert len(tags) == 5 and any(t.endswith("-ego.json") for t in tags) and any(t.endswith("-egoshuf.json") for t in tags)
        finally:
            _MOD_FOR_RO[0] = None
