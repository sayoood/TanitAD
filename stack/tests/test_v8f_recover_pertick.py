"""``--recover-dev-shape {constant, pertick}`` (``tanitad/data/v8f_recover.py``, ``stack/scripts/v8f_mini_train.py``): the recovery perturbation's deviation as a PROFILE over the F history frames.

``constant`` (the default) offsets every history frame by the ONE (dv, da, dom) of the drawn record. ``pertick`` offsets frame k by the record's own per-tick deviation (dv_t[k], da_t[k], dom_t[k]) and moves the pose / nav history by
the cumulative effect. WHAT IS PROVEN, hop by hop (the class "built, tested, and unreachable from its caller", CLAUDE.md 2026-09-10/11):
  (a) ``constant`` is bit-identical to the TIP's module (a frozen snapshot of ``v8f_recover.py`` at 76a0d73c): ``apply`` outputs tensor by tensor, ``content_sha256``, the table build and its counts;
  (b) ``pertick`` with profiles that are constant rows equal to the NOW values equals ``constant`` (within 1e-9: a cumulative sum against a product);
  (c) the per-tick table's NOW column is today's constant law, bit for bit, and a record with an unreal per-tick row is dropped and COUNTED;
  (d) a profile that differs per row: every history row's v / a / yaw rate / pose / nav equals an independent numpy computation of the cumulative rule;
  (e) argv -> make_recover -> the built lever -> declared_vs_built (G-DVB) for both shapes, and the refusals;
  (f) a non-zero gradient reaches the model through a pertick-perturbed batch, and the controls read EXACTLY 0.0;
  (g) MUTATION: the profile replaced by its NOW column (one line, in a copy of the source) turns (d) RED.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", REPO / "stack" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.data import v8f_recover as R  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_ego_perturb import make_batch, world  # noqa: E402,F401  (the synthetic world fixture and the collated-batch builder)
from test_v8f_fast_table import FakeBank, SYN_T, make_world  # noqa: E402
from test_v8f_dagger_train import (COMMON, GATE_ARGV, _selfcheck, _summary, _bank_sha, pack)  # noqa: E402,F401  (the trainer's synthetic world + the generator-written bank)
from test_v8f_recover_wiring import REC, APRIORI, dvb, parts, probe_parts, small_table, table_perturb, _run  # noqa: E402

torch.set_num_threads(1)
RECOVER_P = REPO / "stack" / "tanitad" / "data" / "v8f_recover.py"
TIP_SNAPSHOT = Path(__file__).with_name("v8f_recover_tip_snapshot.txt")
TIP_SNAPSHOT_SHA256 = "73b1ec6eafb50719e1677715e85c2c4b56c952c84edd4465d3078d4b2d7a0aaf"      # v8f_recover.py at tip 76a0d73c, LF
F = 10


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def batch(world):
    b, _, _ = make_batch(world, n=32, seed=0)
    b["lab_s"]["rc_ok"] = torch.ones_like(b["lab_s"]["rc_ok"])
    return b


def now_cols(n=64, seed=0, nonneg=False):
    r = np.random.default_rng(seed)
    dv = r.random(n) * 2.0 if nonneg else -r.random(n) * 3.0 + r.random(n)
    return {"dv": dv, "da": r.normal(0, 0.5, n), "dom": r.normal(0, 0.03, n), "dy": r.normal(0, 0.5, n), "dpsi": r.normal(0, 0.05, n)}


def profile_table(n=64, seed=0, ticks=F, rows="varying", nonneg=False):
    """A table with per-tick columns. ``rows='constant'``: every column row equals the NOW value; ``'varying'``: a different profile per record AND per frame whose last column is the NOW value."""
    c = now_cols(n, seed, nonneg)
    r = np.random.default_rng(seed + 1000)
    prof = {}
    for k, kt, sc in (("dv", "dv_t", 1.0), ("da", "da_t", 0.5), ("dom", "dom_t", 0.03)):
        if rows == "constant":
            prof[kt] = np.repeat(c[k][:, None], ticks, 1)
        else:
            w = np.sort(r.random((n, ticks)), 1)                                   # a ramp the deviation grew along
            m = np.abs(r.normal(0, sc, (n, ticks)))
            m = np.abs(m) if nonneg and k == "dv" else m
            prof[kt] = (c[k][:, None] * w + m * (r.random((n, ticks)) - 0.5))
            prof[kt][:, -1] = c[k]
    return R.RecoverDevTable({**c, **prof}, source_sha256="ab" * 32)


def pert(p=1.0, shape="pertick", table=None, **kw):
    kw.setdefault("bank_path", "bank_x")
    kw.setdefault("bank_sha256", "ab" * 32)
    return R.RecoverPerturb(p=p, dev_shape=shape, table=table if table is not None else profile_table(), **kw)


def same(x, y):
    if isinstance(x, dict):
        return set(x) == set(y) and all(same(x[k], y[k]) for k in x)
    if torch.is_tensor(x):
        return x.dtype == y.dtype and x.shape == y.shape and torch.equal(torch.nan_to_num(x.double(), nan=-7e30) if x.is_floating_point() else x, torch.nan_to_num(y.double(), nan=-7e30) if y.is_floating_point() else y)
    return x == y


# ------------------------------------------------------------------------------------------------ (a) constant == the tip's module, bit for bit
def tip_module(tmp_path):
    raw = TIP_SNAPSHOT.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == TIP_SNAPSHOT_SHA256, "the snapshot of the tip's v8f_recover.py was edited"
    f = tmp_path / "v8f_recover_tip.py"
    f.write_bytes(raw)
    spec = importlib.util.spec_from_file_location("v8f_recover_tip", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m is not R and not hasattr(m, "DEV_SHAPES") and "dev_shape" not in m.RECOVER_DEFAULTS
    return m


@pytest.mark.parametrize("target", ["rejoin", "logged"])
@pytest.mark.parametrize("step, seed, p", [(3, 0, 1.0), (11, 7, 0.5), (12, 7, 0.3)])
def test_a_constant_apply_is_bit_identical_to_the_tips_module(batch, tmp_path, target, step, seed, p):
    T = tip_module(tmp_path)
    c = now_cols(64, 5)
    t_tip = T.RecoverDevTable(c, source_sha256="ab" * 32)
    t_new = R.RecoverDevTable(c, source_sha256="ab" * 32)
    assert t_new.content_sha256() == t_tip.content_sha256(), "a constant table hashes exactly as the tip's"
    old = T.RecoverPerturb(p=p, table=t_tip, bank_path="bank_x", bank_sha256="ab" * 32, seed=seed, target=target)
    new = R.RecoverPerturb(p=p, table=t_new, bank_path="bank_x", bank_sha256="ab" * 32, seed=seed, target=target)
    assert new.dev_shape == "constant" and R.RECOVER_DEFAULTS["dev_shape"] == "constant"
    o0, o1 = old.apply(batch, step), new.apply(batch, step)
    assert set(o0) == set(o1)
    for k in o0:
        assert same(o0[k], o1[k]), f"{k} differs from the tip's"
    assert int(old.last["n_selected"]) == int(new.last["n_selected"]) > 0 or p < 0.4
    assert all(torch.equal(old.last["dev"][k], new.last["dev"][k]) for k in T.TABLE_FIELDS) and torch.equal(old.last["sel"], new.last["sel"])
    assert {k: v for k, v in new.built_levers().items() if k != "recover_dev_shape"} == old.built_levers(), "the built lever record differs from the tip's only by the new key"
    a0, a1 = T.RecoverPerturb(p=p, dev_law="apriori", seed=seed, target=target), R.RecoverPerturb(p=p, dev_law="apriori", seed=seed, target=target)
    assert same(a0.apply(batch, step), a1.apply(batch, step)), "the a-priori law is unchanged"


def test_a_the_constant_table_build_and_its_counts_equal_the_tips(tmp_path):
    T = tip_module(tmp_path)
    names, tracks = make_world(tmp_path / "w", "yaw")
    J, bank = {"clip": np.asarray(names)}, FakeBank(names)
    mk = lambda: D.ClipStore(tmp_path / "w", "train", lru=2, kin_mode="yaw", tracks=tracks)          # noqa: E731
    t0, v0 = T.RecoverDevTable.from_dagger_bank(bank, mk(), J)
    t1, v1 = R.RecoverDevTable.from_dagger_bank(bank, mk(), J)
    t2, v2 = R.RecoverDevTable.from_dagger_bank(bank, mk(), J, per_tick=False)
    for t, v in ((t1, v1), (t2, v2)):
        assert t.counts == t0.counts and "dropped_per_tick_row_unreal" not in t.counts and t.content_sha256() == t0.content_sha256() and not t.has_per_tick and t.ticks is None
        assert all(t.cols[k].tobytes() == t0.cols[k].tobytes() for k in T.TABLE_FIELDS) and v.tobytes() == v0.tobytes()
        assert t.summary(v) == t0.summary(v0) and "per_tick" not in t.summary(v)
    assert len(t0) > 0 and t0.counts["dropped_logged_row_unreal"] > 0


# ------------------------------------------------------------------------------------------------ (c) the per-tick table
class TickBank(FakeBank):
    """``FakeBank`` plus the ``p10f`` the per-tick build reads: [n, F, 8] float32, NaN before the clip start, NOW row = float32 of the bank's kin3 (as the bank writes it)."""

    def __init__(self, names, seed=0, per_clip=24, hmax=F, nan_records=(3, 40, 77)):
        super().__init__(names, seed=seed, per_clip=per_clip, hmax=hmax)
        r = np.random.default_rng(seed + 9)
        n = len(self.ci)
        p10f = r.normal(6.0, 2.0, (n, hmax, 8)).astype(np.float32)
        p10f[:, -1, 0], p10f[:, -1, 1] = self.a["kin3"][:, 0].astype(np.float32), self.a["kin3"][:, 1].astype(np.float32)
        for i in nan_records:                                                  # a record whose own history row is not real (before the clip start)
            p10f[i, 2:4, :] = np.nan
        self.a["p10f"] = p10f
        self.nan_records = tuple(nan_records)


def per_tick_world(tmp_path):
    names, tracks = make_world(tmp_path / "w", "yaw")
    store = D.ClipStore(tmp_path / "w", "train", lru=2, kin_mode="yaw", tracks=tracks)
    return names, store, {"clip": np.asarray(names)}


def test_c_now_column_is_the_constant_law_bit_for_bit_and_unreal_rows_are_counted(tmp_path):
    names, store, J = per_tick_world(tmp_path)
    bank = TickBank(names)
    t_c, v_c = R.RecoverDevTable.from_dagger_bank(bank, store, J)
    t_p, v_p = R.RecoverDevTable.from_dagger_bank(bank, store, J, per_tick=True)
    assert t_p.has_per_tick and t_p.ticks == F and t_p.cols_t["dv_t"].shape == (len(t_p), F) and 0 < len(t_p) < len(t_c)
    assert t_p.counts["records_ok"] == t_c.counts["records_ok"] and t_p.counts["dropped_logged_row_unreal"] == t_c.counts["dropped_logged_row_unreal"]
    assert t_p.counts["kept"] == len(t_p) and t_p.counts["dropped_per_tick_row_unreal"] > 0
    assert t_c.counts["kept"] - t_p.counts["kept"] == t_p.counts["dropped_per_tick_row_unreal"], "every record the constant table keeps and the per-tick table does not is counted"
    for nm, nt in (("dv", "dv_t"), ("da", "da_t"), ("dom", "dom_t")):                  # the invariant: NOW column == the constant law's value, bit for bit
        assert t_p.cols_t[nt][:, -1].tobytes() == t_p.cols[nm].tobytes(), nm
    rows_c = {tuple(r): i for i, r in enumerate(np.stack([t_c.cols[k] for k in R.TABLE_FIELDS], 1))}
    idx = [rows_c[tuple(r)] for r in np.stack([t_p.cols[k] for k in R.TABLE_FIELDS], 1)]       # on the records both keep, the five NOW columns ARE the constant table's rows
    assert len(set(idx)) == len(idx) == len(t_p) and idx == sorted(idx)
    assert np.array_equal(v_p, v_c[idx])
    assert t_p.content_sha256() != t_c.content_sha256()
    cols_t = {k: v.copy() for k, v in t_p.cols_t.items()}
    cols_t["dom_t"][0, 0] += 1e-9
    assert R.RecoverDevTable({**t_p.cols, **cols_t}).content_sha256() != R.RecoverDevTable({**t_p.cols, **t_p.cols_t}).content_sha256(), "the hash covers the per-tick columns"
    assert t_p.summary(v_p)["per_tick"]["ticks"] == F


def test_c_the_per_tick_columns_equal_an_independent_per_record_computation(tmp_path):
    names, store, J = per_tick_world(tmp_path)
    bank = TickBank(names)
    t, _ = R.RecoverDevTable.from_dagger_bank(bank, store, J, per_tick=True)
    wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi            # noqa: E731
    out = []
    for i in np.nonzero(np.asarray(bank.a["reason"]) == 0)[0]:
        d = store.get(str(names[int(bank.ci[i])]))
        jj = int(bank.a["j"][i])
        kin, pg = np.asarray(d["kin3"]), np.asarray(d["pose"], np.float64)
        if jj < F or jj >= len(pg) or jj >= len(kin):
            continue
        rows = list(range(jj - F + 1, jj + 1))
        P = np.asarray(bank.a["pose"][i], np.float64)
        ov = [float(bank.a["p10f"][i, k, 0]) for k in range(F)]
        oa = [float(bank.a["p10f"][i, k, 1]) for k in range(F)]
        ov[-1], oa[-1] = float(bank.a["kin3"][i, 0]), float(bank.a["kin3"][i, 1])
        oom = [None] + [wrap(P[k, 2] - P[k - 1, 2]) / D.DT for k in range(1, F)]
        oom[0] = oom[1]
        lom = [None] + [wrap(pg[r, 2] - pg[r - 1, 2]) / D.DT for r in rows[1:]]
        lom[0] = lom[1]
        lom_ok = math.isfinite(pg[jj - F, 2])
        row = [np.array(ov) - kin[rows, 0], np.array(oa) - kin[rows, 1], np.array(oom) - np.array(lom)]
        if all(np.isfinite(x).all() for x in row) and lom_ok and np.isfinite(kin[jj]).all() and jj >= 2:
            out.append(np.stack(row))
    ref = np.stack(out)
    got = np.stack([t.cols_t["dv_t"], t.cols_t["da_t"], t.cols_t["dom_t"]], 1)
    assert ref.shape == got.shape and ref.shape[0] > 10, (ref.shape, got.shape)
    assert np.abs(ref - got).max() < 1e-12
    assert np.abs(ref[:, :, :-1] - ref[:, :, 1:]).max() > 0.1, "the profiles must really vary along the history"


def test_c_a_table_refuses_partial_misshapen_or_non_finite_per_tick_columns():
    c = now_cols(8)
    with pytest.raises(ValueError, match="all or none"):
        R.RecoverDevTable({**c, "dv_t": np.zeros((8, F))})
    with pytest.raises(ValueError, match="disagree"):
        R.RecoverDevTable({**c, "dv_t": np.zeros((8, F)), "da_t": np.zeros((8, F)), "dom_t": np.zeros((8, F + 1))})
    with pytest.raises(ValueError, match="disagree"):
        R.RecoverDevTable({**c, "dv_t": np.zeros((7, F)), "da_t": np.zeros((7, F)), "dom_t": np.zeros((7, F))})
    bad = np.zeros((8, F))
    bad[2, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite per-tick"):
        R.RecoverDevTable({**c, "dv_t": bad, "da_t": np.zeros((8, F)), "dom_t": np.zeros((8, F))})


# ------------------------------------------------------------------------------------------------ the draw
def test_the_draw_uses_the_same_stream_and_the_same_record_index_in_both_shapes(batch):
    t = profile_table()
    pc, pp = pert(shape="constant", table=t), pert(shape="pertick", table=t)
    for step in (0, 5, 9):
        g0, g1 = torch.Generator().manual_seed(step + 1), torch.Generator().manual_seed(step + 1)
        u0, d0 = pc._draw(32, g0)
        u1, d1 = pp._draw(32, g1)
        assert torch.equal(u0, u1) and torch.equal(g0.get_state(), g1.get_state()), "the same random stream"
        assert torch.equal(d0[3], d1[3]) and torch.equal(d0[4], d1[4]), "dy, dpsi are the NOW row's in both shapes"
        for k in range(3):
            assert d1[k].shape == (32, F) and d0[k].shape == (32,)
            assert torch.equal(d1[k][:, -1], d0[k]), "the NOW column of the profile is the constant draw: the same record"
    oc, op = pc.apply(batch, 4), pp.apply(batch, 4)
    assert torch.equal(pc.last["sel"], pp.last["sel"]) and all(torch.equal(pc.last["dev"][k], pp.last["dev"][k]) for k in R.TABLE_FIELDS)
    assert "dev_t" not in pc.last and set(pp.last["dev_t"]) == set(R.PER_TICK_FIELDS)
    assert torch.equal(oc["kin3"], op["kin3"]) and torch.equal(oc["gt"], op["gt"]), "NOW kinematics and the target are the constant shape's"
    assert not torch.equal(oc["p10f"], op["p10f"]) and not torch.equal(oc["pose"], op["pose"]), "the history is not"


# ------------------------------------------------------------------------------------------------ (b) constant rows == the constant shape
def test_b_pertick_with_constant_rows_equals_constant_within_1e_9(batch):
    t = profile_table(rows="constant", nonneg=True)
    tc = R.RecoverDevTable({k: t.cols[k] for k in R.TABLE_FIELDS}, source_sha256="ab" * 32)
    for step, seed in ((1, 0), (6, 3)):
        a = pert(shape="constant", table=tc, seed=seed).apply(batch, step)
        pp = pert(shape="pertick", table=t, seed=seed)
        b = pp.apply(batch, step)
        assert int(pp.last["n_selected"]) > 5
        for k in R.RECOVER_REBUILT + ("gt",):
            assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype
            x, y = a[k].double(), b[k].double()
            m = torch.isfinite(x) & torch.isfinite(y)
            assert torch.equal(torch.isfinite(x), torch.isfinite(y)) and float((x[m] - y[m]).abs().max()) <= 1e-9, (k, float((x[m] - y[m]).abs().max()))
        assert float((torch.nan_to_num(a["lab_s"]["rc"].double()) - torch.nan_to_num(b["lab_s"]["rc"].double())).abs().max()) <= 1e-9


def test_b_the_clamp_is_read_per_frame_so_a_row_that_clamps_differently_moves_differently(batch):
    """dv_eff_t = max(v + dv_t, 0) - v is a per-frame quantity: with dv = -4 a slow older frame (v 1) clamps to a change of -1 while NOW (v 9) changes by -4. The constant shape reads the NOW row's change for every row."""
    bb = dict(batch)
    bb["p10f"] = batch["p10f"].clone()
    bb["p10f"][:, 4, 0], bb["p10f"][:, -1, 0] = 1.0, 9.0                                    # frame 4 is slow, NOW is fast
    c = {k: np.zeros(16) for k in R.TABLE_FIELDS}
    c["dv"][:] = -4.0
    t = R.RecoverDevTable({**c, **{k: np.repeat(c[n][:, None], F, 1) for k, n in zip(R.PER_TICK_FIELDS, ("dv", "da", "dom"))}}, source_sha256="ab" * 32)
    a = pert(shape="constant", table=R.RecoverDevTable(c, source_sha256="ab" * 32)).apply(bb, 2)
    pp = pert(shape="pertick", table=t)
    b = pp.apply(bb, 2)
    rows = pp.last["sel"].nonzero(as_tuple=True)[0]
    vm = D.EgoPerturb._hist_valid(bb, rows)
    older = vm[:, 0] & vm[:, 4]                                                                # frames older than the clamping frame exist and are real
    assert int(older.sum()) >= 1
    assert float(((a["pose"][rows][older][:, 0] - b["pose"][rows][older][:, 0]).abs()).max()) > 1e-3, "the oldest row moved differently: its cumulative displacement uses frame 4's clamped change"
    assert torch.equal(a["pose"][rows][:, 5:], b["pose"][rows][:, 5:]) or float((a["pose"][rows][:, 5:] - b["pose"][rows][:, 5:]).abs().max()) < 1e-6, "frames newer than the clamp are unchanged"
    assert torch.equal(a["kin3"], b["kin3"]), "NOW is the same"


# ------------------------------------------------------------------------------------------------ (d) per-row profiles vs an independent numpy computation
def numpy_reference(batch, out, pt, r):
    """Row ``r`` of the pertick output recomputed with plain numpy loops from the drawn profiles (``pt.last['dev_t']``) and the NOW-row dy, dpsi (``pt.last['dev']``)."""
    dvt, dat, domt = (pt.last["dev_t"][k][r].double().numpy() for k in R.PER_TICK_FIELDS)
    dy, dpsi = float(pt.last["dev"]["dy"][r]), float(pt.last["dev"]["dpsi"][r])
    rows = torch.tensor([r])
    vm = D.EgoPerturb._hist_valid(batch, rows)[0].numpy()
    v, a, om = (batch["p10f"][r, :, 0].double().numpy(), batch["p10f"][r, :, 1].double().numpy(), batch["act"][r, :, 1].double().numpy())
    v2 = np.maximum(v + dvt, 0.0)
    eff = v2 - v
    Pz = batch["pose"][r].double().numpy()
    psi_dev = Pz[-1, 2] + dpsi
    off = np.array([-math.sin(Pz[-1, 2]) * dy, math.cos(Pz[-1, 2]) * dy])
    nav_ok = D.EgoPerturb._nav_masks(batch, rows)[0][0].numpy()
    nav = batch["nav"][r, :, 3].double().numpy()
    exp_pose, exp_nav = Pz.copy(), nav.copy()
    for k in range(F):
        age = F - 1 - k
        cum = sum(eff[F - 1 - m] for m in range(age))                       # the speed change of the frames at ages 0 .. age - 1
        cum_om = sum(domt[F - 1 - m] for m in range(age))
        ramp = min(max(1.0 - age / R.RAMP_STEPS, 0.0), 1.0)
        if vm[k]:
            exp_pose[k, 0] = Pz[k, 0] + (-cum * D.DT) * math.cos(psi_dev) + ramp * off[0]
            exp_pose[k, 1] = Pz[k, 1] + (-cum * D.DT) * math.sin(psi_dev) + ramp * off[1]
            exp_pose[k, 2] = Pz[k, 2] + ramp * dpsi - cum_om * D.DT
        if nav_ok[k]:
            exp_nav[k] = min(max(max(nav[-1], nav[k] + cum * D.DT / 100.0), 0.0), 3.0)
    return {"vm": vm, "v2": v2, "a2": a + dat, "om2": om + domt, "pose": exp_pose, "nav3": exp_nav}


def check_pertick_against_numpy(batch, out, pt):
    """Raises AssertionError on the first mismatch; returns the number of checked rows and the largest absolute error."""
    sel = pt.last["sel"].nonzero(as_tuple=True)[0].tolist()
    assert len(sel) > 5
    worst = 0.0
    for r in sel:
        e = numpy_reference(batch, out, pt, r)
        vm = e["vm"]
        got_v, got_a = out["p10f"][r, :, 0].double().numpy(), out["p10f"][r, :, 1].double().numpy()
        for name, got, exp in (("v", got_v, e["v2"]), ("a", got_a, e["a2"]), ("pose", out["pose"][r].double().numpy(), e["pose"]), ("nav3", out["nav"][r, :, 3].double().numpy(), e["nav3"])):
            g, x = (got[vm], exp[vm]) if name in ("v", "a") else (got, exp)
            assert not (np.isnan(g) != np.isnan(x)).any(), f"row {r}: {name} differs from the independent numpy computation (NaN pattern)"
            err = float(np.nanmax(np.abs(g - x))) if np.isfinite(g - x).any() else 0.0
            worst = max(worst, err)
            assert err < 2e-4, f"row {r}: {name} differs from the independent numpy computation by {err}"
        assert abs(float(out["kin3"][r, 0]) - e["v2"][-1]) < 2e-6, "kin3 is the NOW row"
    return len(sel), worst


def test_d_every_history_row_follows_its_own_profile_and_the_cumulative_rule(batch):
    t = profile_table(nonneg=False)
    pt = pert(p=0.7, shape="pertick", table=t, seed=2)
    out = pt.apply(batch, 8)
    n, worst = check_pertick_against_numpy(batch, out, pt)
    assert n > 5 and worst < 2e-4
    sel = pt.last["sel"]
    prof = pt.last["dev_t"]["dv_t"][sel]
    assert float((prof[:, :-1] - prof[:, 1:]).abs().max()) > 0.1, "the drawn profiles vary along the history"
    const = pert(p=0.7, shape="constant", table=R.RecoverDevTable({k: t.cols[k] for k in R.TABLE_FIELDS}, source_sha256="ab" * 32), seed=2).apply(batch, 8)
    assert float((const["pose"][sel] - out["pose"][sel]).abs().max()) > 1e-3, "and the constant shape would put the history rows elsewhere (the check is sensitive to the profile)"
    assert float((const["p10f"][sel][..., 0] - out["p10f"][sel][..., 0]).abs().max()) > 0.1
    for k in ("feats", "frame_ok", "act", "limit", "a_t", "a_s", "s_h", "lead_gap_h", "gt_ok", "lab", "lab_t"):
        assert out[k] is batch[k], f"{k} must be the same object"
    unsel = ~sel
    assert int(unsel.sum()) > 3
    for k in R.RECOVER_REBUILT + ("gt",):
        assert not bool((out[k][unsel] != batch[k][unsel]).reshape(int(unsel.sum()), -1).any()), f"{k}: an unselected row moved"


def test_d_p_zero_returns_the_same_batch_and_a_tick_count_mismatch_is_refused(batch):
    assert pert(p=0.0, shape="constant", table=None, bank_path=None, bank_sha256=None).apply(batch, 3) is batch
    with pytest.raises(ValueError, match="another hmax"):
        pert(shape="pertick", table=profile_table(ticks=F + 3)).apply(batch, 3)


# ------------------------------------------------------------------------------------------------ (g) mutation
def test_g_MUTATION_replacing_the_profile_by_its_NOW_column_turns_the_per_row_check_RED(batch, capsys):
    src = RECOVER_P.read_text(encoding="utf-8")
    anchor = [ln for ln in src.splitlines() if "PERTICK-PROFILE-INTAKE" in ln]
    assert len(anchor) == 1 and anchor[0].strip().startswith("dv_t, da_t, dom_t = dv, da, dom")
    indent = anchor[0][: len(anchor[0]) - len(anchor[0].lstrip())]
    mutated = src.replace(anchor[0], indent + "dv_t, da_t, dom_t = (x[:, -1:].expand_as(x) for x in (dv, da, dom))")
    ns: dict = {"__name__": "mutant_recover_pertick"}
    exec(compile(mutated, "mutant_v8f_recover.py", "exec"), ns)
    t = profile_table()
    good = R.RecoverPerturb(p=1.0, dev_shape="pertick", table=t, bank_path="bank_x", bank_sha256="ab" * 32, seed=2)
    mut = ns["RecoverPerturb"](p=1.0, dev_shape="pertick", table=t, bank_path="bank_x", bank_sha256="ab" * 32, seed=2)
    og, om = good.apply(batch, 8), mut.apply(batch, 8)
    n, worst = check_pertick_against_numpy(batch, og, good)
    print(f"CONTROL (unmutated): {n} rows match the independent numpy computation, worst abs err {worst:.3e}")
    with pytest.raises(AssertionError, match="differs from the independent numpy computation") as e:
        check_pertick_against_numpy(batch, om, mut)
    print(f"MUTANT RED: {str(e.value)[:160]}")
    assert not torch.equal(og["pose"], om["pose"]), "the mutant must move the history"
    assert torch.equal(om["kin3"], og["kin3"]), "the mutant keeps the NOW row: only the profile is lost"


def test_g_MUTATION_the_same_NOW_column_mutant_is_caught_by_the_launch_gate_probe(pack):
    """The gate's G-LIVE probe (``recover_probe``) must see a pertick lever whose profile never reaches the batch: the history speeds are checked against max(v + dv_t, 0) of the drawn profile."""
    src = RECOVER_P.read_text(encoding="utf-8")
    anchor = [ln for ln in src.splitlines() if "PERTICK-PROFILE-INTAKE" in ln][0]
    indent = anchor[: len(anchor) - len(anchor.lstrip())]
    ns: dict = {"__name__": "mutant_recover_pertick_probe"}
    exec(compile(src.replace(anchor, indent + "dv_t, da_t, dom_t = (x[:, -1:].expand_as(x) for x in (dv, da, dom))"), "mutant_v8f_recover.py", "exec"), ns)
    model, b, cfg = probe_parts(pack)
    t = profile_table(ticks=int(b["p10f"].shape[1]), seed=4)
    kw = dict(p=0.3, dev_shape="pertick", table=t, bank_path="bank_x", bank_sha256="ab" * 32, seed=1)
    res = TR.recover_probe(model, b, cfg, R.RecoverPerturb(**kw))
    assert res["pass"] is True and res["pertick_v_max_abs_err"] < 1e-4, "the control: the unmutated lever passes the probe"
    with pytest.raises(AssertionError, match="the profile did not reach the batch") as e:
        TR.recover_probe(model, b, cfg, ns["RecoverPerturb"](**kw))
    print(f"GATE PROBE RED ON THE MUTANT: {str(e.value)[:200]}")
    flat = profile_table(ticks=int(b["p10f"].shape[1]), seed=4, rows="constant")
    with pytest.raises(AssertionError, match="flat dv profile"):
        TR.recover_probe(model, b, cfg, R.RecoverPerturb(**{**kw, "table": flat}))


def test_g_the_mutation_anchor_exists_exactly_once():
    assert RECOVER_P.read_text(encoding="utf-8").count("PERTICK-PROFILE-INTAKE") == 1


# ------------------------------------------------------------------------------------------------ (e) argv -> built lever -> G-DVB
def test_e_the_flag_defaults_to_constant_has_a_gate_entry_and_is_in_the_destination_list():
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.recover_dev_shape == "constant" == R.RECOVER_DEFAULTS["dev_shape"] and R.DEV_SHAPES == ("constant", "pertick")
    assert TR.FLAG_CHECKS["recover_dev_shape"] == "rec:recover_dev_shape" and "recover_dev_shape" in TR.RECOVER_DESTS
    with pytest.raises(SystemExit):
        TR.build_parser().parse_args(["--out", "x", "--recover-dev-shape", "zigzag"])
    assert R.RecoverPerturb(p=0.0).built_levers()["recover_dev_shape"] == "constant"


@pytest.mark.parametrize("shape", ["constant", "pertick"])
def test_e_argv_to_built_lever_round_trip_passes_g_dvb(pack, shape):
    extra = [*REC(pack), "--recover-dev-shape", shape] if shape == "pertick" else REC(pack)
    pt = parts(pack, extra)
    a, rcv = pt[0], pt[5]
    assert a.recover_dev_shape == shape and rcv.dev_shape == shape and rcv.built_levers()["recover_dev_shape"] == shape
    assert rcv.table.has_per_tick == (shape == "pertick"), "the table is built with per-tick columns exactly when the shape asks for them"
    assert dvb(pt) == [], "the control: what argv declared is what was built"
    assert pt[4]["recover"]["dev_shape"] == shape
    other = "constant" if shape == "pertick" else "pertick"
    mism = R.RecoverPerturb(p=a.recover_p, dev_law=a.recover_dev_law, table=rcv.table, target=a.recover_target, rejoin_steps=a.recover_rejoin_steps, seed=a.recover_seed, kin_mode=a.kin_mode,
                            bank_path=rcv.bank_path, bank_sha256=rcv.bank_sha256, dev_shape=other)
    bad = dvb(pt, rec=mism)
    assert any("lever recover_dev_shape:" in m and "declared" in m and "built" in m for m in bad), bad


def test_e_a_pertick_table_has_the_profiles_of_the_trainers_own_bank(pack):
    pt = parts(pack, [*REC(pack), "--recover-dev-shape", "pertick"])
    t = pt[5].table
    tc = parts(pack, REC(pack))[5].table
    assert t.ticks == pt[1].hmax and len(t) > 0 and len(tc) >= len(t)
    assert t.counts["kept"] == len(t) and t.counts["records_ok"] == tc.counts["records_ok"]
    assert t.cols_t["dv_t"][:, -1].tobytes() == t.cols["dv"].tobytes() and t.cols_t["dom_t"][:, -1].tobytes() == t.cols["dom"].tobytes()
    assert pt[4]["recover"]["table"]["content_sha256"] == t.content_sha256() != tc.content_sha256()


@pytest.mark.parametrize("kind", ["unknown", "apriori", "no_ticks", "p0"])
def test_e_problems_refuse_the_illegal_shapes(kind):
    if kind == "unknown":
        bad = R.RecoverPerturb(p=0.3, dev_law="apriori", dev_shape="zigzag").problems()
        assert any("recover_dev_shape" in m and "is not one of" in m for m in bad), bad
    elif kind == "apriori":
        bad = R.RecoverPerturb(p=0.3, dev_law="apriori", dev_shape="pertick").problems()
        assert any("pertick with the apriori law" in m for m in bad), bad
        assert R.RecoverPerturb(p=0.3, dev_law="apriori", dev_shape="constant").problems() == []
    elif kind == "no_ticks":
        bad = table_perturb(p=0.3, dev_shape="pertick").problems()
        assert any("no per-tick columns" in m for m in bad), bad
        assert table_perturb(p=0.3, dev_shape="constant").problems() == [] and pert(p=0.3, shape="pertick").problems() == []
        with pytest.raises(ValueError, match="no per-tick columns"):
            table_perturb(p=1.0, dev_shape="pertick").apply({"p10f": torch.zeros(2, F, 8)}, 1)
    else:
        bad = R.RecoverPerturb(p=0.0, dev_shape="pertick").problems()
        assert any("recover_dev_shape" in m and "does nothing" in m for m in bad), bad
        assert R.RecoverPerturb(p=0.0).problems() == []


@pytest.mark.parametrize("extra, what", [
    (["--recover-p", "0.3", "--recover-dev-law", "apriori", "--recover-dev-shape", "pertick"], "pertick with --recover-dev-law apriori"),
    (["--recover-dev-shape", "pertick"], "does nothing"),
])
def test_e_bad_recover_refuses_pertick_with_the_apriori_law_and_at_p0(extra, what):
    a = TR.build_parser().parse_args(["--out", "x", *GATE_ARGV, *extra])
    rcv = TR.make_recover(a, None, None, None, None, {})
    bad = TR.bad_recover(a, rcv)
    assert any(what in m for m in bad), bad
    assert rcv.dev_shape == "pertick"
    a0 = TR.build_parser().parse_args(["--out", "x", *GATE_ARGV, "--recover-p", "0.3", "--recover-dev-law", "apriori"])
    assert TR.bad_recover(a0, TR.make_recover(a0, None, None, None, None, {})) == [], "the control: the a-priori law with the constant shape passes"


def test_e_selfcheck_only_passes_for_a_pertick_arm_and_records_the_shape_the_table_and_the_probe(pack, tmp_path):
    out = tmp_path / "pt"
    assert TR.main(_run(out, [*REC(pack), "--recover-dev-shape", "pertick", "--selfcheck-only"])) == 0
    sc = _selfcheck(out)
    assert sc["declared_vs_built"] == "PASS" and sc["recover_probe"]["pass"] is True
    assert sc["levers"]["recover"]["recover_dev_shape"] == "pertick" and sc["info"]["recover"]["dev_shape"] == "pertick"
    assert sc["levers"]["recover_table"]["per_tick"]["ticks"] > 1 and sc["recover_probe"]["control_p0_grad_diff"] == 0.0 and sc["recover_probe"]["grad_wrt_p10f_selected_abs_sum"] > 0.0


def test_e_a_four_step_pertick_run_applies_the_lever_and_trains_on_something_else_than_constant(pack, tmp_path):
    outs = {}
    for name, extra in (("c", REC(pack, p=1.0)), ("p", [*REC(pack, p=1.0), "--recover-dev-shape", "pertick"])):
        outs[name] = tmp_path / name
        assert TR.main(_run(outs[name], ["--steps", "4", *extra])) == 0
    ck = {k: torch.load(v / "ckpt_last.pt", map_location="cpu", weights_only=False) for k, v in outs.items()}
    assert ck["p"]["recover"]["recover_dev_shape"] == "pertick" and ck["c"]["recover"]["recover_dev_shape"] == "constant"
    assert not all(torch.equal(ck["c"]["model"][n], ck["p"]["model"][n]) for n in ck["c"]["model"]), "the pertick arm trained on a different batch"
    s = _summary(outs["p"])
    assert s["recover_n"] == s["recover_eligible"] > 0 and s["recover_changed"] == s["recover_n"]


# ------------------------------------------------------------------------------------------------ (f) the gradient reaches the model
def _grads(model, b):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for _, p in named:
        p.grad = None
    torch.manual_seed(20261104)
    t = model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]["total"]
    t.backward()
    return float(t.detach()), {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}


def test_f_a_non_zero_gradient_reaches_the_trained_parameters_through_a_pertick_batch_and_the_controls_read_exactly_zero(pack):
    model, b, cfg = probe_parts(pack)
    F16 = int(b["p10f"].shape[1])
    t = profile_table(ticks=F16, seed=4)
    tc = R.RecoverDevTable({k: t.cols[k] for k in R.TABLE_FIELDS}, source_sha256="ab" * 32)
    rp, rc = pert(p=0.3, shape="pertick", table=t, seed=1), pert(p=0.3, shape="constant", table=tc, seed=1)
    res = TR.recover_probe(model, b, cfg, rp)                                   # the G-LIVE probe, on the pertick perturber
    assert res["pass"] is True and res["n_selected_at_p1"] == res["n_eligible"] > 0
    assert res["grad_diff_inputs_only"] > 0.0 and res["loss_diff_inputs_only"] != 0.0 and res["grad_wrt_p10f_selected_abs_sum"] > 0.0 and res["live_recovered"]["pass"] is True
    assert res["control_repeat_grad_diff"] == 0.0 and res["control_p0_grad_diff"] == 0.0 and res["control_p0_loss_diff"] == 0.0 and res["control_p0_same_object"] is True
    assert res["control_prior_grad_diff"] == 0.0 and res["control_wm_grad_diff"] == 0.0
    # the profile itself reaches the loss: the pertick batch and the constant batch of the SAME draw differ in the loss and the gradient; the same batch twice differs by exactly 0.0
    m = model
    m.train()
    bp, bc = rp.apply(b, 9, p=1.0), rc.apply(b, 9, p=1.0)
    assert torch.equal(rp.last["sel"], rc.last["sel"])
    lp, gp = _grads(m, bp)
    lp2, gp2 = _grads(m, bp)
    lc, gc = _grads(m, bc)
    assert lp == lp2 and all(torch.equal(gp[n], gp2[n]) for n in gp), "control: the same batch twice reads exactly 0.0"
    assert lp != lc and sum(float((gp[n] - gc[n]).abs().sum()) for n in gp) > 0.0, "the per-tick profile reaches the trained parameters"
    assert rp.apply(b, 9, p=0.0) is b and rc.apply(b, 9, p=0.0) is b


def test_f_the_probe_still_sees_a_pertick_rebuild_that_is_deleted(pack, monkeypatch):
    model, b, cfg = probe_parts(pack)
    t = profile_table(ticks=int(b["p10f"].shape[1]), seed=4)
    monkeypatch.setattr(R.RecoverPerturb, "_rebuilt_pertick", lambda self, batch, rows, *a: {})
    with pytest.raises(AssertionError, match="unchanged .* selected rows: the rebuild did not reach it"):
        TR.recover_probe(model, b, cfg, pert(p=0.3, shape="pertick", table=t, seed=1))
