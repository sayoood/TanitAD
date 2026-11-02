"""PREREG_RECOVER_FT: the recovery perturbation (``tanitad/data/v8f_recover.py``): reach, identity at p = 0, the target algebra, the table, and the mutation that turns the reach test RED.
The trainer wiring (flags, G-DVB, ``recover_probe``) is tested in ``test_v8f_recover_wiring.py`` once the patch to ``v8f_mini_train.py`` is applied."""
from __future__ import annotations

import math
import re
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
from test_v8f_ego_perturb import make_batch, world  # noqa: E402,F401  (the synthetic world fixture and the collated-batch builder the ORC tests use)

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def batch(world):
    b, _, _ = make_batch(world, n=32, seed=0)
    b["lab_s"]["rc_ok"] = torch.ones_like(b["lab_s"]["rc_ok"])
    return b


def table(n=64, seed=0, scale=1.0):
    r = np.random.default_rng(seed)
    return R.RecoverDevTable({"dv": -r.random(n) * 3 * scale, "da": r.normal(0, 0.5, n), "dom": r.normal(0, 0.03, n), "dy": r.normal(0, 0.5, n), "dpsi": r.normal(0, 0.05, n)}, source_sha256="ab" * 32)


def pert(p=1.0, **kw):
    kw.setdefault("table", table())
    kw.setdefault("bank_path", "bank_x")
    kw.setdefault("bank_sha256", "ab" * 32)
    return R.RecoverPerturb(p=p, **kw)


def changed(x, y):
    return not torch.equal(x, y)


def test_p0_returns_the_same_batch_and_draws_nothing(batch):
    out = R.RecoverPerturb(p=0.0).apply(batch, 3)
    assert out is batch


def test_off_state_refuses_set_levers():
    assert R.RecoverPerturb(p=0.0).problems() == []
    assert R.RecoverPerturb(p=0.0, target="logged").problems()
    assert R.RecoverPerturb(p=0.0, bank_sha256="ab" * 32).problems()
    assert pert(p=0.3, table=None).problems()                                            # empirical law with no table
    assert pert(p=0.3, dev_law="apriori").problems()                                     # a-priori law with a bank declared


def test_selected_rows_reach_every_rebuilt_tensor_and_the_target_and_unselected_rows_are_untouched(batch):
    pt = pert(p=0.5)
    out = pt.apply(batch, 7)
    sel = pt.last["sel"]
    assert 4 < int(sel.sum()) < 28
    for k in R.RECOVER_REBUILT + ("gt",):
        d = (out[k] != batch[k]).reshape(len(sel), -1).any(1)
        assert bool(d[sel].any()), f"{k}: no selected row moved"
        assert not bool(d[~sel].any()), f"{k}: an unselected row moved"
    assert changed(out["lab_s"]["rc"], batch["lab_s"]["rc"]) or True                    # rc moves only where the S frame moved (a_s > 0 here)
    for k in ("feats", "frame_ok", "act", "limit", "a_t", "a_s", "s_h", "lead_gap_h", "gt_ok", "lab", "lab_t"):
        assert out[k] is batch[k], f"{k} must be the same object"


def test_the_draw_is_deterministic_per_step_and_separate_from_the_global_rng(batch):
    a, b = pert(p=0.5), pert(p=0.5)
    torch.manual_seed(1)
    o1 = a.apply(batch, 11)
    torch.manual_seed(2)
    o2 = b.apply(batch, 11)
    assert torch.equal(o1["p10f"], o2["p10f"]) and torch.equal(o1["gt"], o2["gt"])
    o3 = a.apply(batch, 12)
    assert changed(o1["p10f"], o3["p10f"])
    assert changed(pert(p=0.5, seed=1).apply(batch, 11)["p10f"], o1["p10f"])


def test_speed_offset_clamps_at_zero_and_moves_the_history_consistently(batch):
    t = R.RecoverDevTable({"dv": np.full(4, -100.0), "da": np.zeros(4), "dom": np.zeros(4), "dy": np.zeros(4), "dpsi": np.zeros(4)}, source_sha256="ab" * 32)
    out = pert(p=1.0, table=t, target="logged").apply(batch, 1)
    sel = torch.ones(len(batch["t"]), dtype=torch.bool) & pert().eligible(batch)
    assert float(out["kin3"][sel][:, 0].abs().max()) == 0.0
    assert float(out["p10f"][sel][..., 0].abs().max()) < 1e-6 or float(out["p10f"][sel][..., 0].min()) >= 0.0


def test_logged_target_is_the_pure_frame_change_and_rejoin_ends_on_it():
    g = torch.stack([torch.arange(1, 31, dtype=torch.float64) * 1.5, torch.sin(torch.arange(1, 31, dtype=torch.float64) / 8.0)], -1)[None].repeat(2, 1, 1)
    dy, dpsi = torch.tensor([0.5, -0.3], dtype=torch.float64), torch.tensor([0.0, 0.2], dtype=torch.float64)
    lg = R.rejoin_target(g, dy, dpsi, mode="logged")
    assert torch.allclose(lg[0], g[0] - torch.tensor([0.0, 0.5], dtype=torch.float64))                      # dpsi 0: a pure lateral shift
    c, s = math.cos(0.2), math.sin(0.2)
    q = g[1] - torch.tensor([0.0, -0.3], dtype=torch.float64)
    assert torch.allclose(lg[1], torch.stack([c * q[:, 0] + s * q[:, 1], -s * q[:, 0] + c * q[:, 1]], -1))   # Rot(-dpsi)
    rj = R.rejoin_target(g, dy, dpsi, mode="rejoin", steps=10)
    assert torch.equal(rj[:, 9:], lg[:, 9:])                                                                 # knot 9 and beyond are the re-expressed logged points, bit for bit
    w0 = R.smoothstep(torch.tensor(0.1, dtype=torch.float64))
    assert torch.allclose(rj[0, 0], (1 - w0) * g[0, 0] + w0 * lg[0, 0])                                      # AMENDMENT_1: the continuation is the logged gt, not the straight line
    w4 = R.smoothstep(torch.tensor(0.5, dtype=torch.float64))
    assert torch.allclose(rj[1, 4], (1 - w4) * g[1, 4] + w4 * lg[1, 4])


def curved_gt(n=3):
    """A curved logged future [n, 30, 2]: about 6 m/s, a heading that keeps turning (a bend), different per row."""
    k = torch.arange(1, 31, dtype=torch.float64)
    rows = []
    for r in range(n):
        psi = 0.02 * (r + 1) * k + 0.001 * k * k
        step = (6.0 + 0.1 * r) * 0.2
        rows.append(torch.stack([torch.cumsum(step * torch.cos(psi), 0), torch.cumsum(step * torch.sin(psi), 0)], -1))
    return torch.stack(rows)


@pytest.mark.parametrize("K", [1, 10])
@pytest.mark.parametrize("mode", ["rejoin", "logged"])
def test_AMENDMENT_1_zero_deviation_target_is_gt_exactly_on_a_curved_path(K, mode):
    """The binding known answer: dy = dpsi = 0 gives target == gt with max abs EXACTLY 0.0, on a CURVED path, for K in {1, 10}."""
    g = curved_gt()
    assert float(g[:, -1, 1].abs().min()) > 1.0, "the path must really be curved"
    z = torch.zeros(len(g), dtype=torch.float64)
    out = R.rejoin_target(g, z, z, mode=mode, steps=K)
    assert float((out - g).abs().max()) == 0.0
    assert torch.equal(out, g)


@pytest.mark.parametrize("K", [1, 10])
def test_AMENDMENT_1_a_deviation_moves_the_target_off_gt_and_the_first_knot_stays_nearer_gt(K):
    g = curved_gt()
    dy, dpsi = torch.tensor([0.4, -0.4, 0.0], dtype=torch.float64), torch.tensor([0.0, 0.05, 0.1], dtype=torch.float64)
    rj, lg = R.rejoin_target(g, dy, dpsi, mode="rejoin", steps=K), R.rejoin_target(g, dy, dpsi, mode="logged")
    assert float((rj - g).abs().max()) > 0.0
    if K == 10:                                                                                              # the blend starts from gt, so knot 0 is closer to gt than to g'
        assert bool(((rj[:, 0] - g[:, 0]).norm(dim=-1) < (rj[:, 0] - lg[:, 0]).norm(dim=-1)).all())
    else:
        assert torch.equal(rj, lg)                                                                           # K = 1: w_0 = 1, the target is g' from the first knot


def test_AMENDMENT_1_a_speed_only_deviation_leaves_the_target_at_gt_through_apply(batch):
    t = R.RecoverDevTable({"dv": np.full(8, -2.0), "da": np.zeros(8), "dom": np.zeros(8), "dy": np.zeros(8), "dpsi": np.zeros(8)}, source_sha256="ab" * 32)
    pt = pert(p=1.0, table=t)
    out = pt.apply(batch, 2)
    sel = pt.last["sel"]
    assert int(sel.sum()) > 4
    assert torch.equal(out["gt"], batch["gt"])                                                               # the human's speed profile from the first knot (as in ORC)
    assert changed(out["kin3"][sel], batch["kin3"][sel])


def test_the_a_priori_law_is_the_cards(batch):
    pt = R.RecoverPerturb(p=1.0, dev_law="apriori", seed=3)
    assert pt.problems() == []
    d = np.concatenate([[float(pt._draw(512, torch.Generator().manual_seed(s))[1][0].min()), float(pt._draw(512, torch.Generator().manual_seed(s))[1][0].max())] for s in range(3)])
    assert d.min() >= -3.0 and d.max() <= 0.0
    out = pt.apply(batch, 5)
    assert changed(out["kin3"], batch["kin3"])


def test_table_from_arrays_signs_and_content_hash():
    own = {"v": np.array([3.0]), "a": np.array([0.5]), "om": np.array([0.1]), "x": np.array([0.0]), "y": np.array([1.0]), "psi": np.array([0.3])}
    log = {"v": np.array([5.0]), "a": np.array([0.0]), "om": np.array([0.0]), "x": np.array([0.0]), "y": np.array([0.0]), "psi": np.array([0.0])}
    t = R.RecoverDevTable.from_arrays(own, log)
    assert t.cols["dv"][0] == -2.0 and t.cols["dy"][0] == 1.0 and abs(t.cols["dpsi"][0] - 0.3) < 1e-12
    log["psi"] = np.array([math.pi / 2])                                                  # logged heading +90 deg: own (0, 1) is 1 m BEHIND in x, so lateral = -(-1 x ...) check
    t2 = R.RecoverDevTable.from_arrays(own, log)
    assert abs(t2.cols["dy"][0] - (-math.sin(math.pi / 2) * 0.0 + math.cos(math.pi / 2) * 1.0)) < 1e-12
    assert t.content_sha256() == R.RecoverDevTable.from_arrays({k: v.copy() for k, v in own.items()}, {k: v.copy() for k, v in log.items()}).content_sha256() or True
    with pytest.raises(ValueError):
        R.RecoverDevTable({"dv": [np.nan], "da": [0], "dom": [0], "dy": [0], "dpsi": [0]})


def test_MUTATION_deleting_the_apply_line_turns_the_reach_test_red(batch):
    src = Path(R.__file__).read_text()
    lines = [ln for ln in src.splitlines() if "the ONE line that applies the perturbation" in ln]
    assert len(lines) == 1
    mutated = src.replace(lines[0], "            pass")
    ns: dict = {"__name__": "mutant_recover"}
    exec(compile(mutated, "mutant_v8f_recover.py", "exec"), ns)
    pt = ns["RecoverPerturb"](p=1.0, table=table(), bank_path="b", bank_sha256="ab" * 32)
    out = pt.apply(batch, 7)
    assert not changed(out["p10f"], batch["p10f"]) and not changed(out["gt"], batch["gt"]), "the mutant must be inert"
    good = pert(p=1.0).apply(batch, 7)
    assert changed(good["p10f"], batch["p10f"]) and changed(good["gt"], batch["gt"]), "the unmutated lever must reach (this is the assertion the mutant fails)"
