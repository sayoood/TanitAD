"""PREREG_V8F_SPEEDREAD (DRAFT): synthetic self-test of the speed read-out probe. CPU only, no real data, no model checkpoint.

Pins: (1) the read-out sees exactly T's tick slice (an independent loop re-implementation, causality, padded frames, no key outside READ_KEYS);
(2) the fit recovers a planted signal, the CONST control reads skill 0.0 exactly, the PERM control reads no skill, and a purely NON-LINEAR target is found by the
MLP and missed by the ridge (a linear negative is not a negative about learnability); (3) the substitution changes ONLY the five speed cells and the hook hands O
the read-out's value (the emitting line), with the tick slice mirrored against the model's own `_scene` pad rule; (4) the outcome table and the controls.
"""
from __future__ import annotations

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
_MOD = os.environ.get("SPEEDREAD_MODULE", os.path.join(SCRIPTS, "v8f_mini_eval_speedread.py"))
_spec = importlib.util.spec_from_file_location("v8f_mini_eval_speedread_under_test", _MOD)
SR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SR)

D_FEAT, N_TOK, FN = 16, 40, 10
SMALL_MLP = {"hidden": 48, "dropout": 0.0, "lr": 3e-3, "wd": 1e-4, "batch": 128, "max_epochs": 60, "patience": 6, "seed": 0}


def make_batch(B=6, seed=0, a_t=None):
    g = np.random.default_rng(seed)
    ok = np.ones((B, FN), bool)
    ok[0, :8] = False                                                   # window 0: only the last two frames are real
    return {"feats": torch.from_numpy(g.standard_normal((B, FN, N_TOK, D_FEAT)).astype(np.float16)), "frame_ok": torch.from_numpy(ok),
            "p10f": torch.from_numpy(g.standard_normal((B, FN, 8)).astype(np.float32)), "nav": torch.from_numpy(g.standard_normal((B, FN, 5)).astype(np.float32)),
            "limit": torch.from_numpy(g.standard_normal((B, 5)).astype(np.float32)), "limit_ok": torch.from_numpy(g.random(B) > 0.5),
            "a_t": torch.as_tensor(a_t if a_t is not None else [i % 3 for i in range(B)], dtype=torch.long)}


def loop_vectors(batch):
    """Independent re-implementation, one window at a time, from `V8FMini.run_t` / `_scene` as written."""
    f, ok = batch["feats"].numpy().astype(np.float32), batch["frame_ok"].numpy()
    sc, eg = [], []
    for b in range(f.shape[0]):
        last = FN - 1 - int(batch["a_t"][b])
        frames = [last - 3 + k for k in range(4)]
        means = [f[b, j].mean(0) * float(ok[b, j]) for j in frames]
        g = (f[b, last] * float(ok[b, last])).reshape(4, 10, D_FEAT)
        pool = [g[2 * r:2 * r + 2, 2 * c:2 * c + 2].mean((0, 1)) for r in range(2) for c in range(5)]
        sc.append(np.concatenate(means + pool))
        eg.append(np.concatenate([batch["p10f"][b, last].numpy(), batch["nav"][b, last].numpy(), batch["limit"][b].numpy(),
                                  [float(batch["limit_ok"][b])], ok[b, frames].astype(np.float32)]))
    return np.stack(sc), np.stack(eg)


# ------------------------------------------------------------------------------------------------ (1) the input slice
def test_tick_slice_matches_independent_loop():
    bt = make_batch()
    sc, eg = SR.window_vectors(bt, (4, 10))
    sc2, eg2 = loop_vectors(bt)
    assert sc.shape == (6, 14 * D_FEAT) and eg.shape == (6, 23)
    np.testing.assert_allclose(sc, sc2, atol=1e-5)
    np.testing.assert_array_equal(eg, eg2.astype(np.float32))


def test_no_frame_after_or_before_the_tick_window_is_read():
    bt = make_batch(seed=1)
    sc, eg = SR.window_vectors(bt, (4, 10))
    bt2 = {k: v.clone() for k, v in bt.items()}
    for b in range(6):
        last = FN - 1 - int(bt["a_t"][b])
        bt2["feats"][b, last + 1:] = 99.0
        bt2["feats"][b, :last - 3] = -99.0
        bt2["p10f"][b, last + 1:] = 99.0
        bt2["nav"][b, :last] = 99.0
    sc2, eg2 = SR.window_vectors(bt2, (4, 10))
    np.testing.assert_array_equal(sc, sc2)
    np.testing.assert_array_equal(eg, eg2)


def test_padded_frame_reads_zero_and_only_read_keys():
    bt = make_batch(seed=2, a_t=[0, 0, 0, 0, 0, 0])
    sc, _ = SR.window_vectors(bt, (4, 10))
    assert np.all(sc[0, :2 * D_FEAT] == 0.0) and np.any(sc[0, 2 * D_FEAT:4 * D_FEAT] != 0.0)       # frames 6, 7 padded, 8, 9 real
    assert set(bt) == set(SR.READ_KEYS)
    poisoned = dict(bt, lab_t={"cons": torch.full((6, 26), float("nan"))}, gt=torch.full((6, 30, 2), float("nan")), pose=torch.zeros(6, FN, 3))
    np.testing.assert_array_equal(SR.window_vectors(poisoned, (4, 10))[0], sc)


# ------------------------------------------------------------------------------------------------ (2) the fit
def synth(n_clips=48, per=40, d_raw=40, kind="linear", seed=0, offset=5.0):
    g = np.random.default_rng(seed)
    n = n_clips * per
    clips = np.repeat([f"clip-{seed}-{i:03d}" for i in range(n_clips)], per)
    S = g.standard_normal((n, d_raw)).astype(np.float32)
    S[:, :5] *= 3.0                                                     # the signal directions carry the variance, so PCA keeps them
    E = g.standard_normal((n, 23)).astype(np.float32)
    u = S[:, :5] / 3.0
    sig = u if kind == "linear" else np.sign(u * np.roll(u, 1, 1)) * 1.5
    Y = offset + sig + 0.2 * g.standard_normal((n, 5))
    M = g.random((n, 5)) > 0.05
    return S, E, Y, M, clips


def fit(S, E, Y, M, clips):
    return SR.fit_all(S[::2], lambda r: S[r], E, Y, M, clips, pca_k=16, mlp_cfg=SMALL_MLP, log=lambda *a: None)


@pytest.fixture(scope="module")
def linear_fit():
    tr = synth(seed=0)
    return tr, fit(*tr), synth(n_clips=16, seed=7)


def test_fit_recovers_a_planted_linear_signal_on_unseen_clips(linear_fit):
    _, res, (S, E, Y, M, clips) = linear_fit
    ro, rec = res["readouts"], res["record"]
    base = rec["const"]
    for k in ("lin", "mlp", "read"):
        assert np.all(SR.skill(ro[k].predict_raw(S, E), Y, M, base) > 0.6), k
    assert np.all(np.abs(SR.skill(ro["lin_ego"].predict_raw(S, E), Y, M, base)) < 0.1)          # the signal is in the scene, not in ego
    assert rec["n_windows"] == 1920 and rec["d_design"] == 16 + 23 and rec["n_clips"] == 48


def test_const_reads_exactly_zero_skill_and_perm_reads_none(linear_fit):
    _, res, (S, E, Y, M, _) = linear_fit
    ro, rec = res["readouts"], res["record"]
    P = ro["const"].predict_raw(S, E)
    assert np.all(SR.skill(P, Y, M, rec["const"]) == 0.0)
    assert np.ptp(P, axis=0).max() == 0.0
    assert rec["perm_holdout_skill_max"] <= 0.02
    assert np.all(SR.skill(ro["perm"].predict_raw(S, E), Y, M, rec["const"]) < 0.05)


def test_permute_targets_breaks_the_link_and_keeps_the_marginal():
    _, _, Y, M, _ = synth(seed=3)
    Yp = SR.permute_targets(Y, M)
    for c in range(5):
        np.testing.assert_allclose(np.sort(Yp[M[:, c], c]), np.sort(Y[M[:, c], c]))
        assert abs(np.corrcoef(Yp[M[:, c], c], Y[M[:, c], c])[0, 1]) < 0.1
    np.testing.assert_array_equal(Yp[~M], Y[~M])


def test_a_nonlinear_target_is_found_by_the_mlp_and_missed_by_the_ridge():
    tr = synth(seed=11, kind="xor", n_clips=60)
    res = fit(*tr)
    S, E, Y, M, _ = synth(seed=12, kind="xor", n_clips=16)
    ro, rec = res["readouts"], res["record"]
    sk_lin = SR.skill(ro["lin"].predict_raw(S, E), Y, M, rec["const"])
    sk_mlp = SR.skill(ro["mlp"].predict_raw(S, E), Y, M, rec["const"])
    assert np.all(sk_lin < 0.1) and np.mean(sk_mlp) > 0.3, (sk_lin, sk_mlp)
    assert rec["primary"] == "mlp" and ro["read"] is ro["mlp"]


def test_selection_uses_heldout_train_only_and_ties_go_linear():
    assert SR.select_primary([0.5] * 5, [0.4] * 5) == "mlp"
    assert SR.select_primary([0.4] * 5, [0.4] * 5) == "lin"
    assert SR.select_primary([0.3] * 5, [0.4] * 5) == "lin"


def test_folds_are_by_clip_and_deterministic():
    clips = np.repeat([f"c{i}" for i in range(50)], 7)
    f = SR.fold_of(clips)
    assert np.array_equal(f, SR.fold_of(clips)) and set(f) <= set(range(5)) and len(set(f)) == 5
    for c in set(clips):
        assert len(set(f[clips == c])) == 1


def test_readout_save_load_roundtrip(linear_fit, tmp_path):
    _, res, (S, E, _, _, _) = linear_fit
    p = str(tmp_path / "readout.pt")
    SR.save_readouts(res["readouts"], p)
    ro2 = SR.load_readouts(p)
    for k in ("lin", "mlp", "lin_ego", "mlp_ego", "const", "perm", "read", "alt"):
        np.testing.assert_array_equal(ro2[k].predict_raw(S, E), res["readouts"][k].predict_raw(S, E))
    assert ro2["read"].kind == res["record"]["primary"]


# ------------------------------------------------------------------------------------------------ (3) the substitution and the hook
SPEED_IDX = [34 + i for i in (6, 16, 17, 18, 19)]


def test_substitute_touches_only_the_speed_cells():
    own = torch.randn(6, 60)
    new = np.arange(30, dtype=np.float64).reshape(6, 5)
    out = SR.substitute(own, new, SPEED_IDX)
    non = [i for i in range(60) if i not in SPEED_IDX]
    assert torch.equal(out[:, non], own[:, non])
    np.testing.assert_allclose(out[:, SPEED_IDX].numpy(), new)
    assert torch.equal(SR.substitute(own, SR.new_speed_std("identity", None, None, own, SPEED_IDX, None, None), SPEED_IDX), own)


def test_phys_std_roundtrip():
    mu, sd = np.array([1.0, 2, 3, 4, 5]), np.array([2.0, 2, 2, 2, 2])
    v = np.random.default_rng(0).standard_normal((4, 5)) * 3 + 7
    np.testing.assert_allclose(SR.phys_to_std(v, mu, sd) * sd + mu, v)


class _ConstReadOut:
    kind = "lin"

    def __init__(self, v):
        self.v = np.asarray(v, np.float64)

    def predict_batch(self, batch):
        SR.window_vectors(batch)                                          # the real slice runs (and would raise on a bad batch)
        return np.broadcast_to(self.v, (batch["feats"].shape[0], 5)).copy()


def _fake_world():
    """A stand-in for SPEEDAVG's hook base and the model, enough for the hook's own lines; `_scene` copies V8FMini._scene's pad rule verbatim."""
    class FakeModel:
        def __init__(self):
            self.cons_mu, self.cons_sd = torch.linspace(0, 2.5, 26), torch.linspace(1, 3.5, 26)
            self.hist = 4

        def forward_infer(self, batch):
            g = torch.Generator().manual_seed(5)
            return {"code": torch.randn(batch["feats"].shape[0], 60, generator=g)}

        def forward_diag(self, batch, code_override):
            return {"code_read_by_O": code_override}

        def _encode(self, batch):
            return batch["feats"].float()

        def _scene(self, adapt, ok, last):
            pos = last[:, None] - (self.hist - 1 - torch.arange(self.hist))[None]
            pad = ~torch.gather(ok, 1, pos)
            return None, pad[:, :, None].expand(-1, -1, adapt.shape[2]).flatten(1)

        def _code_gt(self, lab):
            return torch.zeros(lab["cons"].shape[0], 60)

    M = types.SimpleNamespace(N_GOAL=22, N_LAT=5, N_LON=7, CODE_DIM=60, V8FMini=FakeModel)

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
            return {"mode": self.mode, "nonspeed_max_abs_dev": self.nonspeed_dev}

        def cells(self):
            return {k: np.concatenate(v) for k, v in self.rows.items()}

    SA = types.SimpleNamespace(SpeedAvgHook=SpeedAvgHook, replace_cells=None, source_code=None)
    return M, SA, FakeModel()


def _eval_batch():
    bt = make_batch(seed=4)
    bt["lab_t"] = {"cons": torch.randn(6, 26), "has": torch.ones(6, dtype=torch.bool)}
    return bt


def test_hook_hands_O_the_readout_value_and_nothing_else():
    M, SA, model = _fake_world()
    v = [10.0, 11.0, 12.0, 13.0, 14.0]
    Hook = SR.make_hook_class(SA)
    bt = _eval_batch()
    own = model.forward_infer(bt)["code"]
    h = Hook(M, "read", SPEED_IDX, _ConstReadOut(v), extras={"read": _ConstReadOut(v)}).install()
    try:
        out = M.V8FMini.forward_diag(model, bt, own)["code_read_by_O"]
    finally:
        h.restore()
    mu, sd = model.cons_mu[h.ci].double().numpy(), model.cons_sd[h.ci].double().numpy()
    np.testing.assert_allclose(out[:, SPEED_IDX].double().numpy() * sd + mu, np.broadcast_to(v, (6, 5)), atol=1e-5)
    assert torch.equal(out[:, h.non], own[:, h.non])
    st = h.stats()
    assert st["nonspeed_max_abs_dev"] == 0.0 and st["tick_mirror_ok"] is True
    cells = h.cells()
    assert cells["used_std"].shape == (6, 5) and cells["pred_read"].shape == (6, 5)


def test_hook_identity_is_bit_exact_and_mirror_catches_a_bad_slice():
    M, SA, model = _fake_world()
    Hook = SR.make_hook_class(SA)
    bt = _eval_batch()
    own = model.forward_infer(bt)["code"]
    h = Hook(M, "identity", SPEED_IDX).install()
    try:
        out = M.V8FMini.forward_diag(model, bt, own)["code_read_by_O"]
    finally:
        h.restore()
    assert torch.equal(out, own) and h.mirror_ok is True
    model.hist = 3                                                       # a model whose T slice differs from the read-out's -> the mirror must say so
    h2 = Hook(M, "identity", SPEED_IDX).install()
    try:
        M.V8FMini.forward_diag(model, bt, own)
    finally:
        h2.restore()
    assert h2.mirror_ok is False


# ------------------------------------------------------------------------------------------------ (4) the outcome table and the controls
F = SR.F_M


def r(d, lo, hi):
    return {"delta": d, "lo": lo, "hi": hi}


SEP_BIG, SEP_SMALL, NS = r(-0.20, -0.30, -0.10), r(-0.05, -0.08, -0.01), r(-0.02, -0.06, 0.02)


@pytest.mark.parametrize("rows,want", [
    ({0: {"read": SEP_BIG, "alt": NS}, 1: {"read": SEP_BIG, "alt": NS}}, "INFO IN INPUT"),
    ({0: {"read": r(-F, -0.2, -0.01), "alt": NS}, 1: {"read": SEP_BIG, "alt": NS}}, "INFO IN INPUT"),
    ({0: {"read": r(-F + 1e-9, -0.2, -0.01), "alt": NS}, 1: {"read": SEP_BIG, "alt": NS}}, "PARTIAL"),
    ({0: {"read": SEP_SMALL, "alt": NS}, 1: {"read": SEP_BIG, "alt": NS}}, "PARTIAL"),
    ({0: {"read": NS, "alt": NS}, 1: {"read": NS, "alt": NS}}, "NOT IN INPUT"),
    ({0: {"read": NS, "alt": SEP_SMALL}, 1: {"read": NS, "alt": NS}}, "MIXED"),
    ({0: {"read": SEP_BIG, "alt": NS}, 1: {"read": NS, "alt": NS}}, "MIXED"),
    ({0: {"read": SEP_BIG, "alt": NS}}, "REFUSED"),
])
def test_decide_table(rows, want):
    assert SR.decide(rows, F) == want


def test_decide_refuses_on_a_failed_control():
    assert SR.decide({0: {"read": SEP_BIG, "alt": NS}, 1: {"read": SEP_BIG, "alt": NS}}, F, controls_ok=False) == "REFUSED"


def _fake_ev():
    g = np.random.default_rng(0)
    eid = np.repeat(np.arange(10), 5)
    own = g.random(50) + 2.0
    ev = {}
    for m in SR.MODES:
        ev[m] = {"eid": eid.copy(), "ade6": own.copy() if m in ("own", "identity") else own - 0.3, "pred": np.zeros((50, 30, 2)),
                 "hook": {"nonspeed_max_abs_dev": 0.0, "tick_mirror_ok": True}}
    mu, sd = np.linspace(1, 5, 5), np.full(5, 2.0)
    pr = g.standard_normal((50, 5)) + 8
    gt = g.standard_normal((50, 5)) + 8
    gt[3, 1] = np.nan
    ev["identity"]["cells"] = {"cons_mu": mu.astype(np.float32), "cons_sd": sd.astype(np.float32), "pred_read": pr, "pred_const": np.full((50, 5), 7.5),
                               "gt_phys": gt.astype(np.float32), "has": np.ones(50, bool)}
    ev["read"]["cells"] = {"used_std": ((pr - mu) / sd).astype(np.float32)}
    return ev


def test_controls_pass_on_clean_and_fail_on_each_break():
    fitrec = {"perm_holdout_skill_max": 0.0, "train_parity_ok": True, "eval_train_disjoint": True, "const": [7.5] * 5}
    ev = {0: _fake_ev(), 1: _fake_ev()}
    assert SR.all_pass(SR.controls(ev, None, fitrec))
    for brk in ("identity", "nonspeed", "mirror", "eid", "path", "nocells"):
        ev2 = {0: _fake_ev(), 1: _fake_ev()}
        if brk == "identity":
            ev2[1]["identity"]["ade6"] = ev2[1]["identity"]["ade6"] + 1e-9
        elif brk == "nonspeed":
            ev2[0]["read"]["hook"]["nonspeed_max_abs_dev"] = 1e-7
        elif brk == "mirror":
            ev2[0]["alt"]["hook"]["tick_mirror_ok"] = False
        elif brk == "eid":
            ev2[1]["perm"]["eid"] = ev2[1]["perm"]["eid"][::-1].copy()
        elif brk == "path":
            ev2[0]["read"]["cells"]["used_std"][7, 2] += 1e-3
        else:
            del ev2[1]["identity"]["cells"]
        assert not SR.all_pass(SR.controls(ev2, None, fitrec)), brk
    for k, v in (("perm_holdout_skill_max", 0.05), ("train_parity_ok", False), ("eval_train_disjoint", False)):
        assert not SR.all_pass(SR.controls(ev, None, dict(fitrec, **{k: v}))), k


def test_read_rows_reports_capture_and_perm_guard():
    ev = {0: _fake_ev(), 1: _fake_ev()}
    paired = lambda a, b, eid, n: r(float(np.mean(a) - np.mean(b)), float(np.mean(a) - np.mean(b)) - 0.01, float(np.mean(a) - np.mean(b)) + 0.01)
    rows = SR.read_rows(ev, paired, 10)
    assert rows[0]["capture"] == pytest.approx(1.0) and rows[0]["perm_not_separated_better"] is False


# ------------------------------------------------------------------------------------------------ (A1) the extract follows the data's ego width
@pytest.mark.parametrize("lim_w", [4, 5])
def test_extract_sizes_ego_from_the_data(tmp_path, monkeypatch, lim_w):
    """A1-2026-10-27: the real cache's `limit` is 4-wide, so ego is 22 dims; the draft hard-coded 23 and the first real extract stopped on a
    broadcast error. cmd_extract, run end to end on a stub data layer, must store ego at the data's own width and equal window_vectors."""
    fn, d, n_win = 10, 1024, 7
    g = np.random.default_rng(5)

    def item(c_i, t, a_t, a_s):
        ok = np.ones(fn, bool)
        return {"feats": torch.from_numpy(g.standard_normal((fn, 40, d)).astype(np.float16)), "frame_ok": torch.from_numpy(ok),
                "p10f": torch.from_numpy(g.standard_normal((fn, 8)).astype(np.float32)), "nav": torch.from_numpy(g.standard_normal((fn, 5)).astype(np.float32)),
                "limit": torch.from_numpy(g.standard_normal(lim_w).astype(np.float32)), "limit_ok": torch.tensor(True),
                "a_t": torch.tensor(a_t), "a_s": torch.tensor(a_s), "lab_t": {"cons": torch.from_numpy(g.standard_normal(26)), "has": torch.tensor(True)}}

    def collate(items):
        out = {}
        for k in items[0]:
            if k == "lab_t":
                out[k] = {kk: torch.stack([it[k][kk] for it in items]) for kk in items[0][k]}
            else:
                out[k] = torch.stack([it[k] for it in items])
        return out

    seen = []
    fake_d = types.SimpleNamespace(
        read_manifest=lambda cache, split: [(f"c{i}", None) for i in range(3)], sha12=lambda c: c, select_train_clips=lambda cl, frac: list(cl),
        load_join=lambda p: {"clip": np.array(["c0", "c1", "c2"])}, load_native_tracks=lambda p: {}, ClipStore=lambda *a, **k: None,
        train_windows=lambda store, J, clips, hist: [(i % 3, 5, 0, 0) for i in range(n_win)],
        V8FMiniWindows=lambda *a, **k: types.SimpleNamespace(item=lambda *x: seen.append(item(*x)) or seen[-1]), collate=collate)
    fake_m = types.SimpleNamespace(CI={n: i for i, n in enumerate(SR.SPEED_CELLS)})
    monkeypatch.setattr(SR, "_stack_imports", lambda: (fake_d, fake_m))
    for f in ("join.npz", "tracks.pt"):
        (tmp_path / f).write_bytes(b"x")
    a = types.SimpleNamespace(cache=str(tmp_path), join=str(tmp_path / "join.npz"), tracks=str(tmp_path / "tracks.pt"), grid="pool4x10",
                              out=str(tmp_path / "out"), batch=4, allow_parity_mismatch=True)
    assert SR.cmd_extract(a) == 0
    tr = np.load(tmp_path / "out" / "train_arrays.npz")
    want = 8 + 5 + lim_w + 1 + SR.HIST
    assert tr["ego"].shape == (n_win, want)
    _, eg = SR.window_vectors(collate(seen), (4, 10))
    np.testing.assert_array_equal(tr["ego"], eg)
    rec = json.load(open(tmp_path / "out" / "extract.json", encoding="utf-8"))
    assert rec["d_ego"] == want and rec["parity_ok"] is False
