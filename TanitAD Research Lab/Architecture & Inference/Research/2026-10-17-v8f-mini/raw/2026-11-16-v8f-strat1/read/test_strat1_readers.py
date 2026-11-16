"""Unit tests of the STRAT1 readers on SYNTHETIC data (no dumps, no model).  Run: python -m pytest -q -p no:cacheprovider test_strat1_readers.py
Mutations (mutate_check.py) must turn named tests RED."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strat1_lib as L  # noqa: E402
import strat1_strategic as S  # noqa: E402
import strat1_ctx_read as C  # noqa: E402
import strat1_read as R  # noqa: E402
import run_dumps_strat1 as RDS  # noqa: E402
import strat1_k4_eval as K4  # noqa: E402

CI = L.load_ci()
OR = R.OR


def seed_cells(costs=(), gains=()):
    return {c: {"status": "OK", "cost": c in costs, "gain": c in gains} for c in R.CELLS}


A, B_ = R.CELLS[0], R.CELLS[5]


def test_driving_keeps_adds():
    z = seed_cells()
    assert R.driving_outcome({0: z, 1: z}, {})[0] == ["KEEPS"]
    g = seed_cells(gains=(A,))
    assert R.driving_outcome({0: g, 1: g}, {})[0] == ["ADDS"]
    assert R.driving_outcome({0: g, 1: z}, {})[0] == ["KEEPS"]       # a gain at one seed only is not ADDS


def test_driving_costs_nav_heads_disagree():
    c = seed_cells(costs=(A,))
    z = seed_cells()
    assert R.driving_outcome({0: c, 1: c}, {0: c, 1: c})[0] == ["COSTS-NAV"]
    assert R.driving_outcome({0: c, 1: c}, {0: z, 1: z})[0] == ["COSTS-HEADS"]
    assert R.driving_outcome({0: c, 1: c}, {0: c, 1: z})[0] == ["COSTS-HEADS"]      # N1 costs at one seed only: not NAV
    assert R.driving_outcome({0: c, 1: z}, {0: z, 1: z})[0] == ["SEEDS-DISAGREE"]
    two = seed_cells(costs=(A, B_))
    assert R.driving_outcome({0: two, 1: c}, {0: c, 1: c})[0] == ["COSTS-NAV", "SEEDS-DISAGREE"]
    assert R.driving_outcome({0: c}, {})[0] == ["NOT-YET"]


def test_route_cost_label_and_next_steps():
    assert any("ROUTE-COST" in x for x in R.next_steps(["KEEPS"], True))
    assert all(x.split(":")[0] in R.NEXT for x in R.next_steps(["KEEPS", "SEEDS-DISAGREE"], False))


def test_floor_rule_sub_floor_neither_gain_nor_cost():
    rng = np.random.RandomState(0)
    eid = np.repeat(np.arange(40), 10)
    base = rng.randn(400)
    arm = base + 0.05 + 0.001 * rng.randn(400)                       # a tiny, tightly separated worsening
    r_big_floor = OR.cell_delta(CI, arm, base, eid, False, 0.5, 200)
    assert r_big_floor["separated"] and not r_big_floor["cost"] and not r_big_floor["gain"]
    r_small_floor = OR.cell_delta(CI, arm, base, eid, False, 0.01, 200)
    assert r_small_floor["cost"] and not r_small_floor["gain"]
    r_gain = OR.cell_delta(CI, base, arm, eid, False, 0.01, 200)
    assert r_gain["gain"] and not r_gain["cost"]


def test_nan_masking_and_k1_self_pair():
    eid = np.repeat(np.arange(20), 5)
    a = np.random.RandomState(1).randn(100)
    b = a.copy()
    a[::7] = np.nan
    r = L.paired(CI, a, b, eid, 100)
    assert r["n_windows"] == int(np.isfinite(a).sum())
    assert OR.identity_control(CI, eid)["pass"]
    z = L.paired(CI, b, b, eid, 100)
    assert z["delta"] == 0.0 and z["lo"] == 0.0 and z["hi"] == 0.0
    assert L.paired(CI, np.full(5, np.nan), b[:5], eid[:5], 10)["status"] == "NOT-APPLICABLE"


def test_no_pause_string_with_control():
    for f in ("strat1_read.py", "strat1_strategic.py", "strat1_ctx_read.py"):
        txt = open(os.path.join(HERE, f), encoding="utf-8").read()
        assert txt.count("PAUSED") == 0, f
    assert open(os.path.join(HERE, "strat1_read.py"), encoding="utf-8").read().count("route_following") >= 1     # the control literal


# ---------------------------------------------------------------------------------------------------------------- strategic
def fake_G(n=300, seed=0):
    rng = np.random.RandomState(seed)
    side = rng.choice([1, -1], n, p=[0.2, 0.8]).astype(np.int8)
    return {"side": side, "theta": np.full(n, 25.0), "d_text": rng.uniform(0, 40, n), "text_ok": np.ones(n, bool), "rc_ok": np.ones((n, 3), bool)}


def test_k7_k8_controls():
    G = fake_G()
    k = S.k7(G)
    assert k == {"hold": 0.0, "announced_x40deg": 1.0, "mirrored": 0.0}
    k8 = S.k8(G, np.roll(np.arange(len(G["side"])), 37))
    assert k8["real"] == 1.0 and abs(k8["donor"] - k8["expected_donor"]) < 1e-12 and k8["donor"] < 1.0


def block(delta_d, sep_d, delta_b, sep_b, n=100):
    def row(d, s):
        return [{"arc_m": 30, "status": "n/a"}, {"arc_m": 50, "status": "n/a"}, {"arc_m": 80, "status": "OK", "delta": d, "separated": s, "n_windows": n, "n_episodes": 20}]
    return {"real_minus_donor": row(delta_d, sep_d), "real_minus_best_constant": row(delta_b, sep_b)}


def test_route_following_outcomes():
    f, p, i = block(0.3, True, 0.1, True), block(0.3, True, -0.1, True), block(0.0, False, 0.1, True)
    assert S.outcome({0: f, 1: f})[0] == "S-FOLLOWS"
    assert S.outcome({0: f, 1: p})[0] == "S-PARTIAL"
    assert S.outcome({0: p, 1: p})[0] == "S-PARTIAL"
    assert S.outcome({0: f, 1: i})[0] == "S-IGNORES"
    assert S.outcome({0: f})[0] == "NOT-YET"
    na = {"real_minus_donor": [{}, {}, {"status": "n/a", "n_windows": 3, "n_episodes": 1}], "real_minus_best_constant": [{}, {}, {"status": "n/a"}]}
    assert S.outcome({0: na, 1: na})[0] == "NOT-READ"


def test_n_rule_and_contrast_na():
    eid = np.repeat(np.arange(8), 10)                      # 80 windows, 8 episodes: below 10 episodes
    a = np.ones((80, 3))
    assert all(x["n_a"] for x in S.n_block(a, eid))
    rows = S.contrast_rows(CI, a, a * 0, eid)
    assert all(r["status"] == "n/a" and r["n_episodes"] == 8 for r in rows)


def test_best_constant_majority():
    G = fake_G()
    ref = np.ones((len(G["side"]), 3))
    bc, maj = S.best_const_cells(G, ref)
    assert maj[2] == -1 and abs(bc[:, 2].mean() - (G["side"] == -1).mean()) < 1e-12


def test_k9_equal_and_unequal():
    ck = np.random.RandomState(0).randn(4, 3, 3).astype(np.float32)
    ticks = {"tick": np.array([0, 0, 1]), "ci": np.array([0, 1, 1]), "t0": np.array([3, 3, 3]), "s_ckpt": np.stack([ck[0], ck[1], ck[1]])}
    ci_t = {(0, 3): 0, (1, 3): 1}
    assert S.k9(ticks, ck, ci_t)[2] is True
    ticks["s_ckpt"] = ticks["s_ckpt"] + 1e-3
    assert S.k9(ticks, ck, ci_t)[2] is False


def test_ground_refuses_misaligned_tables():
    n = 6
    nt = {"eval_off": np.array([0, 3]), "eval_clip": np.array(["a", "b"]), "eval_nav_d": np.zeros(n)}
    tk = {"eval_off": np.array([0, 4]), "vocab": np.array(["TURN_L", "TURN_R"]), "eval_cls_id": np.zeros(n, int), "eval_flag": np.zeros(n, int), "eval_dist_m": np.zeros(n)}
    jn = {"off": np.array([0, 3]), "clip": np.array(["a", "b"]), "rc": np.zeros((n, 3, 3)), "rc_ok": np.ones((n, 3), bool)}
    with pytest.raises(RuntimeError):
        S.ground(tk, nt, jn, np.array([0]), np.array([0]))


# ---------------------------------------------------------------------------------------------------------------- ctx
def fake_tab(n_clip=4, per=10):
    n = n_clip * per
    rng = np.random.RandomState(0)
    t = {"eval_clip": np.array([f"c{i}" for i in range(n_clip)]), "eval_off": np.arange(n_clip) * per, "eval_n": np.full(n_clip, per), "eval_flag": np.zeros(n, np.uint8)}
    for v in C.CTX_VARS:
        t[f"eval_{v}_id"] = rng.randint(0, 3, n).astype(np.int8)
        t[f"eval_{v}_valid"] = np.ones(n, bool)
    t["eval_illumination_id"][:5] = C.TWILIGHT_ID
    t["eval_flag"][5:8] = 2
    t["eval_weather_valid"][10:12] = False
    return t


def test_label_masking_twilight_flag_valid():
    tab = fake_tab()
    y = C.label_rows(tab, np.arange(40))
    assert (y[:5, 3] == -1).all() and (y[5:8] == -1).all() and (y[10:12, 1] == -1).all() and (y[12:, 1] >= 0).all()
    assert C.table_baselines(tab)["twilight"]["n_steps"] == 5


def test_ctx_controls_k2_k3_k5_k6():
    rng = np.random.RandomState(3)
    n, ne = 600, 30
    eid = np.repeat(np.arange(ne), n // ne)
    y = np.stack([rng.randint(0, k, n) for k in C.CTX_N], 1)
    y[:, 1] = np.where(rng.rand(n) < 0.7, 0, y[:, 1])
    consts = {v: {"constant_class": int(np.bincount(y[:, k], minlength=C.CTX_N[k]).argmax())} for k, v in enumerate(C.CTX_VARS)}
    plat = (eid % 2).astype(int)
    rig = np.where(plat == 0, 3, 5)
    con = C.controls(CI, y, eid, consts, rig, plat, nb=100)
    assert all(d["pass"] for d in con["K2"].values())
    assert all(d["pass"] for d in con["K3"].values())
    assert all(d["pass"] for d in con["K5"].values())
    assert con["K6"]["pass"]


def test_ctx_decide_instrument_fail():
    rows = {v: {"detects": True, "detects_beyond_rig": True} for v in C.CTX_VARS}
    no_k4 = {v: False for v in C.CTX_VARS}
    assert C.decide({0: rows, 1: rows}, no_k4)["country"] == "DETECTS(country) + DETECTS-BEYOND-RIG"
    assert C.decide({0: rows, 1: rows}, {**no_k4, "weather": True})["__all__"] == "INSTRUMENT-FAIL"
    no = {**rows, "weather": {"detects": False}}
    assert C.decide({0: rows, 1: no}, no_k4)["weather"] == "NOT-DETECTED(weather)"


def test_ctx_masked_rows_not_scored():
    y = np.array([0, 1, -1, -1, 2, 0])
    pred = np.array([0, 1, 2, 2, 2, 1])
    eid = np.array([0, 0, 1, 1, 2, 2])
    r = C.var_rows(CI, pred, y, eid, 0, 3, 50)
    assert r["n_windows"] == 4 and abs(r["accuracy"] - 0.75) < 1e-12


def test_misaligned_table_refused():
    tab, join = fake_tab(), {"clip": np.array(["c0", "c1", "c2", "c3"]), "off": np.arange(4) * 10, "n": np.full(4, 10)}
    plat = {"eval_clip": join["clip"]}
    C.check_alignment(tab, plat, join)
    with pytest.raises(RuntimeError):
        C.check_alignment(dict(tab, eval_off=tab["eval_off"] + 1), plat, join)


def test_k4_reinit_changes_head_and_nothing_else():
    torch = pytest.importorskip("torch")
    m = torch.nn.Module()
    m.body = torch.nn.Linear(4, 4)
    m.ctx_head = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU(), torch.nn.Linear(4, 6))
    body0 = m.body.weight.detach().clone()
    head0 = m.ctx_head[2].weight.detach().clone()
    assert K4.reinit_ctx_head(m, 0) == 4
    assert torch.equal(body0, m.body.weight) and not torch.equal(head0, m.ctx_head[2].weight)
    with pytest.raises(RuntimeError):
        K4.reinit_ctx_head(torch.nn.Module(), 0)


# ---------------------------------------------------------------------------------------------------------------- run_dumps / nav args
def test_nav_args_refusals():
    assert RDS.nav_args("S1_s0", "text", "n.npz", "abc") == ["--nav-text-npz", "n.npz", "--nav-text-sha256", "abc"]
    assert RDS.nav_args("C0_s0", "v9", "", "") == []
    with pytest.raises(ValueError):
        RDS.nav_args("C0_s0", "v9", "n.npz", "abc")           # a C0 tag given an npz is refused
    with pytest.raises(ValueError):
        RDS.nav_args("S1_s0", "text", "", "")                  # an S1 tag without one is refused
    with pytest.raises(ValueError):
        RDS.extra_args("STRAT1-C0_s0", "text")                 # the checkpoint's source must match the card's arm
    with pytest.raises(ValueError):
        RDS.extra_args("STRAT1-S1_s0", "v9")


def test_refusal_checks(tmp_path, monkeypatch):
    d = tmp_path / "S1_s0"
    d.mkdir()
    (d / "ckpt_last.pt").write_bytes(b"x")
    monkeypatch.setitem(RDS.ARMS, "STRAT1-S1_s0", (str(d / "ckpt_last.pt"), ("bank", str(d / "BANK_MANIFEST.json"))))
    assert "BANK_MANIFEST" in RDS.refusal("STRAT1-S1_s0")
    good = {"tag": "S1_s0", "commit": L.COMMIT, "files": {"ckpt_last.pt": {"sha256": L.sha256_file(str(d / "ckpt_last.pt"))}}}
    for mod, frag in (({}, None), ({"tag": "N1_s0"}, "names tag"), ({"commit": "0" * 40}, "commit"), ({"files": {"ckpt_last.pt": {"sha256": "0" * 64}}}, "sha256 differs")):
        (d / "BANK_MANIFEST.json").write_text(json.dumps({**good, **mod}))
        got = RDS.refusal("STRAT1-S1_s0")
        assert (got is None) if frag is None else (frag in got)
