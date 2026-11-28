"""AMEND_HWMPLAN_4 tests. Synthetic toy stack (PLANTEST's planted stack, as LON-PLAN's tests) for the shadow, the wrapper and the controls; synthetic numbers for the label rules. CPU only.
Every control has its deliberate-regression arm; every label condition has a mutation that flips a synthetic case. No SCORE window of R35 is planned here."""
import json, os, shutil, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amend4_run as W
import amend4_controls as K
import amend4_read as RD
import amend4_stage as ST
import l2img as L
import lonplan_common as C
import numpy as np
import pytest
import torch
import hwm_plan as H
import run_hwm as R
import test_lonplan as T                       # LON-PLAN's toy world (Env, TOY_GEOM, FAST)

G = T.TOY_GEOM


@pytest.fixture(scope="module")
def env():
    return T.Env()


@pytest.fixture(scope="module")
def pl(env):
    sol = H.SolverCfg(seed_mode="hold", lam=0.0, S=4, n_iters=12, decode_lam=0.0, decode_iters=300, plan_ctx=1)
    return H.HWMPlanner(env.wm, G, sol)


@pytest.fixture(scope="module")
def win(env):
    w = env.windows()
    z = torch.from_numpy(w["z"])
    return dict(w=w, z0=z[:, 0], A0=w["A"][:, 0], goal=H.OracleGoal(z[:, G.goal_frame], G.goal_frame), eid=w["eid"])


def unit(pl, win, mode, idx=None, **kw):
    idx = np.arange(0, 16) if idx is None else idx
    return L.plan_unit(pl, mode, win["goal"], win["z0"], win["A0"], idx, kw.pop("seed", 0), kw.pop("batch0", 0), eids=win["eid"], **kw)


# ------------------------------------------------------------------------------------------ l2img
def test_off_is_the_code_of_record_bit_for_bit_and_default_is_off(pl, win):
    idx = np.arange(0, 16)
    ref = pl.plan_windows("hier", win["goal"], win["z0"], win["A0"], idx=idx, seed=0, batch0=0)
    for m in ("off", None):
        out = unit(pl, win, m)
        assert K.check_off_equal(out, ref)["ok"]
    assert L.DEFAULT_MODE == "off"
    rec = unit(pl, win, "off", record=True)
    assert K.check_off_equal(rec, ref)["ok"] and "m3_b0" in rec


def test_off_mutation_default_prim_goes_red(pl, win):
    ref = pl.plan_windows("hier", win["goal"], win["z0"], win["A0"], idx=np.arange(16), seed=0, batch0=0)
    try:
        L.DEFAULT_MODE = "prim"
        out = unit(pl, win, None)
    finally:
        L.DEFAULT_MODE = "off"
    assert not K.check_off_equal(out, ref)["ok"]


def test_prim_stays_in_the_image_knots_are_p_and_p_is_in_the_box(pl, win, env):
    out = unit(pl, win, "prim")
    r = K.check_prim_id(out, env.st, env.wm, G, "prim")
    assert r["ok"] and r["knots_equal_maxabs"] == 0.0 and r["image_resid_fresh_max"] <= L.IMG_TOL
    lo, hi = env.st.prior_lo_1.numpy(), env.st.prior_hi_1.numpy()
    p = out["l2_prim_raw"] / np.array([H.ACC_SCALE, float(env.wm.ch1_scale)])
    assert (p >= lo - 1e-5).all() and (p <= hi + 1e-5).all()
    assert out["plan_raw"].shape == (16, G.n_traj, 2) and "decode_resid" not in out


def test_prim23_image_at_level_3(pl, win, env):
    out = unit(pl, win, "prim23")
    r = K.check_prim_id(out, env.st, env.wm, G, "prim23")
    assert r["ok"] and r["image_resid3_fresh_max"] <= L.IMG_TOL and "l3_prim_raw" in out


def test_level3_and_generators_untouched_under_prim_and_prim23_and_rs(pl, win):
    off = unit(pl, win, "off", record=True)
    pr, p23, rs = unit(pl, win, "prim"), unit(pl, win, "prim23"), unit(pl, win, "prim", l1_sub="donor")
    assert K.check_l3id(off, pr, "prim")["ok"] and K.check_l3id(off, p23, "prim23")["ok"] and K.check_l3id(off, rs, "prim")["ok"]
    assert not np.array_equal(pr["m3_b0"], p23["m3_b0"])                     # P23 does move level 3


def test_mutation_noise_from_planner_generator_goes_red(pl, win):
    off = unit(pl, win, "off", record=True)
    mut = unit(pl, win, "prim", mutate="noise_from_planner_gen")
    assert not K.check_l3id(off, mut, "prim")["ok"]


def test_mutations_of_prim_id_go_red(pl, win, env):
    a = unit(pl, win, "prim", mutate="no_shadow")
    b = unit(pl, win, "prim", mutate="decode_knots")
    ra, rb = K.check_prim_id(a, env.st, env.wm, G, "prim"), K.check_prim_id(b, env.st, env.wm, G, "prim")
    assert not ra["ok"] and ra["image_resid_fresh_max"] > L.IMG_TOL
    assert not rb["ok"] and rb["knots_equal_maxabs"] > 0.0


def test_reach_gradient_zero_control_and_iterates_and_frozen_mutation(pl, win):
    pr, pm = {}, {}
    unit(pl, win, "prim", probe=pr)
    unit(pl, win, "prim", mutate="frozen_p", probe=pm)
    ok, red = K.check_reach(pr[2]), K.check_reach(pm[2])
    assert ok["ok"] and ok["grad_abs_sum"] > 0 and ok["zero_control"] == 0.0 and ok["iterate_moved_maxabs"] > 0
    assert not red["ok"] and red["zero_control"] == 0.0 and red["iterate_moved_maxabs"] == 0.0


def test_rs_donor_makes_level1_worse_and_the_batch_without_a_shift_is_refused(pl, win):
    pr, rs = unit(pl, win, "prim"), unit(pl, win, "prim", l1_sub="donor")
    assert np.median(rs["cost1"]) > np.median(pr["cost1"])
    dn, d = L.donor_shift(np.array(["a", "a", "b", "b"]))
    assert d == 2 and list(dn) == [2, 3, 0, 1]
    with pytest.raises(RuntimeError):
        L.donor_shift(np.array(["a"] * 4))
    with pytest.raises(ValueError):
        unit(pl, win, "off", l1_sub="donor")


def test_shadow_is_restored_after_a_raise_and_after_return(pl, win, env):
    st = env.st
    base = set(st.__dict__) & {"_optimize", "rollout"}
    assert not base
    boom = lambda idx: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        L.plan_unit(pl, "prim", boom, win["z0"], win["A0"], np.arange(16), 0, 0, eids=win["eid"])
    assert not (set(st.__dict__) & {"_optimize", "rollout"})
    with pytest.raises(RuntimeError):
        L.plan_unit(pl, "prim", boom, win["z0"], win["A0"], np.arange(16), 0, 0, eids=win["eid"], l1_sub="donor")
    assert not (set(st.__dict__) & {"_optimize", "rollout"})
    unit(pl, win, "prim23")
    assert not (set(st.__dict__) & {"_optimize", "rollout"})
    sentinel = lambda *a, **k: None
    with L.instance_attrs(st, _optimize=sentinel):
        pass
    assert "_optimize" not in st.__dict__
    st.__dict__["_optimize"] = sentinel
    with pytest.raises(ValueError):
        with L.instance_attrs(st, _optimize=lambda *a: 0):
            raise ValueError
    assert st.__dict__["_optimize"] is sentinel
    st.__dict__.pop("_optimize")


def test_batch_cap_and_bad_arguments(pl, win):
    with pytest.raises(ValueError):
        unit(pl, win, "off", idx=np.arange(65))
    with pytest.raises(ValueError):
        unit(pl, win, "prim", mutate="nope")
    with pytest.raises(ValueError):
        L.resolve_mode("bogus")


# ------------------------------------------------------------------------------------------ the wrapper (C-L4-IMG, C-READBACK)
@pytest.fixture()
def tr(env, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "prepare", env.prepare)
    wsrc = tmp_path / "windows_src.npz"
    np.savez(wsrc, **env.windows())
    (tmp_path / "ckpt.pt").write_bytes(b"toy-checkpoint"); (tmp_path / "lat.pt").write_bytes(b"toy-latents")
    import types
    t = types.SimpleNamespace(tmp=tmp_path, wsrc=str(wsrc), wsha=C.sha256_file(wsrc), n=0)

    def argv(l2="prim", l1=None, out=None, drop=()):
        t.n += 1
        out = out or str(tmp_path / f"run{t.n}")
        a = ["--tag", "toy", "--ckpt", str(tmp_path / "ckpt.pt"), "--train-latents", str(tmp_path / "lat.pt"), "--train-tag", "toy", "--seed-mode", "hold", "--lam=0", *T.FAST]
        if "l2" not in drop:
            a += ["--l2-image", l2]
        if l1:
            a += ["--l1-sub", l1]
        return a + ["--arms", "hwm_oracle", "--seeds", "0", "--workers", "1", "--out", out]
    t.argv = argv
    return t


def run(tr, l2="prim", l1=None, **kw):
    av = tr.argv(l2, l1)
    W.run_arm(av, l2, tr.wsrc, tr.wsha, l1 or "own", enforce_threads=False, prepare=R.prepare, **kw)
    return av[av.index("--out") + 1]


def plan_of(out, s=0):
    with np.load(os.path.join(out, f"plans_hwm_oracle_s{s}.npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_wrapper_writes_the_layout_stamps_and_solver_json(tr):
    out = run(tr, "prim")
    assert {"windows.npz", "SOLVER.json", "plans_hwm_oracle_s0.npz"} <= set(os.listdir(out))
    sol = json.load(open(f"{out}/SOLVER.json", encoding="utf-8"))
    assert sol["l2_image"] == "prim" and sol["l1_sub"] == "own" and sol["lam"] == 0.0 and {"l2img_py", "amend4_run_py"} <= set(sol["module_sha256"]) and sol["code_sha256"]["hwm_plan.py"]
    p = plan_of(out)
    assert str(p["l2_image_planned"]) == "prim" and str(p["l1_sub_planned"]) == "own" and "l2_prim_raw" in p and "gen_hash" in p and len(p["plan_raw"]) == 80
    assert p["m2_b0"].shape[0] == 64                                         # batch 0 only
    assert W.verify_stamps(out) == {"plans_hwm_oracle_s0.npz": "prim"}


def test_c_l4_img_refusals(tr):
    with pytest.raises(W.RefusedRun, match="not named"):
        W.run_arm(tr.argv("prim", drop=("l2",)), "prim", tr.wsrc, tr.wsha, enforce_threads=False)
    with pytest.raises(W.RefusedRun, match="declared l2_image"):
        W.run_arm(tr.argv("prim"), "off", tr.wsrc, tr.wsha, enforce_threads=False)
    with pytest.raises(W.RefusedRun, match="donor"):
        W.run_arm(tr.argv("off", "donor"), "off", tr.wsrc, tr.wsha, "donor", enforce_threads=False)
    with pytest.raises(W.RefusedRun, match="declared l1_sub"):
        W.run_arm(tr.argv("prim", "donor"), "prim", tr.wsrc, tr.wsha, "own", enforce_threads=False)
    pre = tr.tmp / "pre"
    pre.mkdir()
    np.savez(pre / "plans_hwm_oracle_s0.npz", plan_raw=np.zeros((1, 12, 2)))
    with pytest.raises(W.RefusedRun, match="not empty"):
        W.run_arm(tr.argv("prim", out=str(pre)), "prim", tr.wsrc, tr.wsha, enforce_threads=False)
    with pytest.raises(W.RefusedRun, match="windows source sha256"):
        W.run_arm(tr.argv("prim"), "prim", tr.wsrc, "0" * 64, enforce_threads=False)
    with pytest.raises(W.RefusedRun, match="mutation"):
        W.run_arm(tr.argv("prim"), "prim", tr.wsrc, tr.wsha, enforce_threads=False, mutate="frozen_p")


def test_l4_regressions_helper_and_real_argv_table(tmp_path):
    r = K.l4_regressions(str(tmp_path))
    assert r["ok"] and r["declared_mismatch"]["refused"] and r["flag_not_named"]["refused"] and r["preplanted_plan_file"]["refused"]
    for s in ST.STAGES:
        for ar in ST.ARMS:
            for k in ST.CH:
                assert K.check_argv_declared(s, ar, k)["ok"]
    b = ST.build("score", "RS", "s1")["argv"]
    assert "--l1-sub" in b and b[b.index("--l1-sub") + 1] == "donor" and b[b.index("--l2-image") + 1] == "prim" and "--lam=0" in b and b[b.index("--seeds") + 1:b.index("--seeds") + 2] == ["0"]


def test_c_readback_equal_and_regression_read_at_off_fails(tr):
    out = run(tr, "prim")
    assert W.readback(out, seed=0, prepare=R.prepare)["equal"]
    assert not W.readback(out, seed=0, l2_override="off", prepare=R.prepare)["equal"]
    out2 = run(tr, "prim", "donor")
    assert W.readback(out2, seed=0, prepare=R.prepare)["equal"]
    out3 = run(tr, "off")
    assert W.readback(out3, seed=0, prepare=R.prepare)["equal"]
    assert not W.readback(out3, seed=0, l2_override="prim", prepare=R.prepare)["equal"]


def test_off_through_the_wrapper_equals_the_code_of_record_run(tr, env):
    out = run(tr, "off")
    a = W.W.parse([t for t in tr.argv("off") if t not in ("--l2-image", "off")])
    ref_dir = tr.tmp / "ref"
    os.makedirs(ref_dir)
    shutil.copyfile(tr.wsrc, ref_dir / "windows.npz")
    a.out = str(ref_dir)
    R.cmd_plan(a)
    p, q = plan_of(out), plan_of(str(ref_dir))
    for k in ("plan_raw", "cost1", "a1_raw", "decode_resid", "iters1"):
        assert np.array_equal(p[k], q[k]), k


# ------------------------------------------------------------------------------------------ the other controls, pure parts
def test_c_in_and_regression():
    exp = dict(ckpt="abcd", windows="1234")
    assert K.check_inputs(exp, dict(ckpt="abcdef", windows="123456"))["ok"]
    assert not K.check_inputs(exp, dict(ckpt="bbcdef", windows="123456"))["ok"]


def test_c_train_iso_and_regression(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, eid=np.array(["c1", "c2"])); np.savez(b, eid=np.array(["c3", "c4"]))
    assert K.check_train_iso(str(a), str(b))["ok"]
    assert not K.check_train_iso(str(b), str(b))["ok"]
    np.savez(tmp_path / "c.npz", eid=np.array(["c2", "c9"]))
    assert not K.check_train_iso(str(a), str(tmp_path / "c.npz"))["ok"]


# ------------------------------------------------------------------------------------------ the label rules (synthetic; each outcome; one mutation per rule)
def row(delta, lo, hi):
    return dict(delta=delta, lo=lo, hi=hi)


def base(**o):
    """A chassis input that reads NULL: nothing separated, nothing worse."""
    c = dict(dH=row(0.9, 0.8, 1.0), dH_F=0.01, dH_seed1=0.9, d0=row(-0.001, -0.02, 0.02), d0_F=0.01, d0_seed1=-0.001, gyaw=row(0.0, -0.01, 0.01), gyaw_F=0.005, D=0.5,
             band=dict(d0=row(0.0, -0.1, 0.1), d0_F=0.01, d0_seed1=0.0))
    c.update(o)
    return c


CEIL = dict(dH=row(-0.2, -0.3, -0.1), dH_seed1=-0.2, d0=row(-1.1, -1.2, -1.0), d0_seed1=-1.1)
NARR = dict(d0=row(-0.2, -0.3, -0.1), d0_seed1=-0.2)
HARM = dict(d0=row(0.2, 0.1, 0.3))


def test_each_outcome_row_is_produced():
    assert RD.chassis_label(base()) == "NULL"
    assert RD.chassis_label(base(**NARR)) == "NARROWS"
    assert RD.chassis_label(base(**CEIL)) == "CEILING"
    assert RD.chassis_label(base(**HARM)) == "HARM"
    assert RD.chassis_label(base(gyaw=row(0.05, 0.02, 0.08))) == "HARM"                 # the guard
    assert RD.chassis_label(base(**NARR, gyaw=row(0.05, 0.02, 0.08))) == "HARM"
    assert RD.outcome("CEILING", True) == "CEILING" and RD.outcome("CEILING", False) == "INVALID"
    assert RD.combine(["NARROWS", "NARROWS"])["label"] == "NARROWS"
    c = RD.combine(["CEILING", "NULL"])
    assert c["label"] == "SEEDS-DISAGREE" and c["effective"] == "NULL" and c["seeds_disagree"]
    assert RD.combine(["NARROWS", "HARM"])["effective"] == "HARM"


# (mutated condition, an input that the rule reads as `want` unmutated and as `flip` with that ONE condition dropped)
MUTATIONS = [
    ("ceil_ci", base(dH=row(-0.2, -0.3, 0.1), dH_seed1=-0.2, d0=row(-1.1, -1.2, -1.0), d0_seed1=-1.1), "NARROWS", "CEILING"),
    ("ceil_floor", base(dH=row(-0.005, -0.01, -0.001), dH_seed1=-0.005, d0=row(-1.1, -1.2, -1.0), d0_seed1=-1.1), "NARROWS", "CEILING"),
    ("ceil_seed1", base(dH=row(-0.2, -0.3, -0.1), dH_seed1=0.1, d0=row(-1.1, -1.2, -1.0), d0_seed1=-1.1), "NARROWS", "CEILING"),
    ("ceil_D", base(**CEIL, D=0.05), "NARROWS", "CEILING"),
    ("narrows_ci", base(d0=row(-0.2, -0.3, 0.1), d0_seed1=-0.2), "NULL", "NARROWS"),
    ("narrows_floor", base(d0=row(-0.005, -0.01, -0.001), d0_seed1=-0.005), "NULL", "NARROWS"),
    ("narrows_seed1", base(d0=row(-0.2, -0.3, -0.1), d0_seed1=0.05), "NULL", "NARROWS"),
    ("harm_d0_ci", base(d0=row(0.2, -0.1, 0.3)), "NULL", "HARM"),
    ("harm_d0_floor", base(d0=row(0.005, 0.001, 0.01)), "NULL", "HARM"),
    ("harm_guard", base(gyaw=row(0.05, 0.02, 0.08)), "HARM", "NULL"),
]


@pytest.mark.parametrize("name,c,want,flip", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_label_rule_mutation_flips_a_case(name, c, want, flip):
    assert RD.chassis_label(c) == want
    assert RD.chassis_label(c, drop=(name,)) == flip


def test_label_mutations_leave_the_other_cases_alone():
    assert RD.chassis_label(base(), drop=("ceil_ci",)) == "NULL"
    assert RD.chassis_label(base(**CEIL), drop=("narrows_ci",)) == "CEILING"


def test_band_label_and_base_solver_and_read_labels():
    assert RD.band_label(dict(d0=row(-0.2, -0.3, -0.1), d0_F=0.01, d0_seed1=-0.2)) == "BAND-ADDS"
    assert RD.band_label(dict(d0=row(0.2, 0.1, 0.3), d0_F=0.01, d0_seed1=0.2)) == "BAND-HURTS"
    assert RD.band_label(dict(d0=row(0.0, -0.1, 0.1), d0_F=0.01, d0_seed1=0.0)) == "BAND-NEUTRAL"
    assert RD.base_solver({"P": "CEILING", "P23": "NULL"}, {"P": [0, 0], "P23": [0, 0]}, {"P": [0, 0], "P23": [0, 0]}) == "P"
    assert RD.base_solver({"P": "CEILING", "P23": "CEILING"}, {"P": [-.1, -.1], "P23": [-.2, -.3]}, {"P": [0, 0], "P23": [0, 0]}) == "P23"
    assert RD.base_solver({"P": "CEILING", "P23": "CEILING"}, {"P": [-.1, -.5], "P23": [-.2, -.3]}, {"P": [0, 0], "P23": [0, 0]}) == "P"
    assert RD.base_solver({"P": "NARROWS", "P23": "NULL"}, {"P": [0, 0], "P23": [0, 0]}, {"P": [0, 0], "P23": [0, 0]}) == "P"
    assert RD.base_solver({"P": "NULL", "P23": "NARROWS"}, {"P": [0, 0], "P23": [0, 0]}, {"P": [0, 0], "P23": [0, 0]}) == "P23"
    assert RD.base_solver({"P": "NULL", "P23": "HARM"}, {"P": [0, 0], "P23": [0, 0]}, {"P": [0, 0], "P23": [0, 0]}) == "A0"
    inp = {"P": {"s0": base(), "s1": base()}, "P23": {"s0": base(**CEIL), "s1": base(**NARR)}}
    for x in inp.values():
        for c in x.values():
            c["band"] = dict(d0=row(0.2, 0.1, 0.3), d0_F=0.01, d0_seed1=0.2)
    r = RD.read_labels(inp)
    assert r["P"]["name"] == "L2IMG-NULL" and r["P23"]["label"] == "SEEDS-DISAGREE" and r["P23"]["effective"] == "NARROWS" and r["P23"]["name"] == "L23IMG-SEEDS-DISAGREE"
    assert r["base_solver_for_RA_read"] == "P23" and not r["multiplicity_guard_applies"] and r["P"]["band"]["label"] == "BAND-HURTS"
    inp["P23"]["s1"] = base(**CEIL); inp["P23"]["s1"]["band"] = inp["P"]["s0"]["band"]
    r = RD.read_labels(inp)
    assert r["P23"]["name"] == "L23IMG-CEILING" and r["multiplicity_guard_applies"] and r["base_solver_for_RA_read"] == "P23"


def test_cell_extraction_from_contrast_json_shapes():
    def rows(fam, band, metric, dH, d0, repX, repA):
        return ([dict(kind="contrast", family=fam, band=band, metric=metric, a="A", b="HOLD", **dH), dict(kind="contrast", family=fam, band=band, metric=metric, a="A", b="B", **d0)],
                [dict(kind="inference_seed_replicate", arm="A", family=fam, band=band, metric=metric, delta=repX), dict(kind="inference_seed_replicate", arm="B", family=fam, band=band, metric=metric, delta=repA)])
    R1, P1 = rows("LONGITUDINAL", "0-2s", "acc_mae", row(0.5, 0.4, 0.6), row(-0.4, -0.5, -0.3), 0.01, -0.02)
    R2, P2 = rows("LATERAL", "0-2s", "yaw_mae", row(0.0, -1, 1), row(0.001, -0.002, 0.004), 0.002, 0.001)
    R3, P3 = rows("LONGITUDINAL", "2-6s", "acc_mae", row(0.1, 0.0, 0.2), row(0.3, 0.2, 0.4), 0.0, 0.0)
    js = dict(rows=R1 + R2 + R3, replicate=P1 + P2 + P3)
    c = RD.chassis_input(js, 0.3)
    assert c["dH_F"] == 0.02 and abs(c["d0_seed1"] - (-0.4 + 0.01 + 0.02)) < 1e-12 and abs(c["dH_seed1"] - 0.51) < 1e-12 and c["gyaw_F"] == 0.002 and c["band"]["d0"]["delta"] == 0.3
    assert RD.chassis_label(c) == "NARROWS"


def test_assemble_2_0_and_markdown_skeleton(tmp_path):
    z = dict(plan_raw=np.ones((3, 30, 2)), l2_prim_raw=np.full((3, 30, 2), 2.0))
    np.savez(tmp_path / "p.npz", **z)
    out = RD.assemble_2_0(str(tmp_path / "p.npz"), str(tmp_path / "o.npz"))
    q = np.load(out)["plan_raw"]
    assert (q[:, :10] == 2.0).all() and (q[:, 10:] == 1.0).all()
    c = base()
    c["band"] = dict(d0=row(0.2, 0.1, 0.3), d0_F=0.01, d0_seed1=0.2)
    labels = RD.read_labels({"P": {"s0": c, "s1": c}, "P23": {"s0": c, "s1": c}})
    assert "L2IMG-NULL" in RD.render_md(dict(labels=labels, chassis={}))


def test_p_and_p23_plan_files_carry_l2_prim_raw_with_the_expected_shape(tr):
    for l2 in ("prim", "prim23"):
        p = plan_of(run(tr, l2))
        assert p["l2_prim_raw"].shape == (80, G.h2 * G.c2, 2) and p["l2_prim_raw"].dtype == np.float64 and np.isfinite(p["l2_prim_raw"]).all()
    assert "l2_prim_raw" not in plan_of(run(tr, "off"))
