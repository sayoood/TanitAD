"""AMEND_HWMPLAN_5 build tests: every label rule reached with synthetic inputs and mutated (a dropped condition changes the label), the controls on the toy linear world with their regression arms, the stage rules.
CPU only. Run: run_tests_amend5.cmd (serial, basetemp on F:)."""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pytest
import torch
import amend5_stage as ST
import ra_l1 as R
import amend5_controls as K
import amend5_read as RD

row = lambda d, lo, hi: dict(delta=d, lo=lo, hi=hi)
BETTER = row(-0.10, -0.15, -0.05)
NONE_ = row(-0.01, -0.05, 0.03)
WORSE_ = row(0.10, 0.05, 0.15)


def mk(**o):
    c = dict(dH=NONE_, dH_F=0.01, dH_seed1=-0.01, dB=NONE_, dB_F=0.01, dB_seed1=-0.01, dM=NONE_, dM_F=0.01, dM_seed1=-0.01, gx=NONE_, gx_F=0.01, D=0.5, base="A0")
    c.update(o)
    return c


# ---------------------------------------------------------------------------------------------- label rules, each with a mutation
def test_label_null():
    assert RD.chassis_label(mk())["label"] == "NULL"


def test_label_ceiling_and_each_condition():
    c = mk(dH=BETTER, dH_seed1=-0.1, dB=BETTER, dB_seed1=-0.1)
    assert RD.chassis_label(c)["label"] == "CEILING"
    assert RD.chassis_label(dict(c, dH=row(-0.10, -0.15, 0.01)))["label"] == "NARROWS"                          # CI crosses 0 -> not CEILING
    assert RD.chassis_label(dict(c, dH=row(-0.10, -0.15, 0.01)), drop=("ceil_ci",))["label"] == "CEILING"      # mutation: dropping the CI condition flips it
    assert RD.chassis_label(dict(c, dH_F=0.2))["label"] == "NARROWS"
    assert RD.chassis_label(dict(c, dH_F=0.2), drop=("ceil_floor",))["label"] == "CEILING"
    assert RD.chassis_label(dict(c, dH_seed1=0.01))["label"] == "NARROWS"
    assert RD.chassis_label(dict(c, dH_seed1=0.01), drop=("ceil_seed1",))["label"] == "CEILING"
    assert RD.chassis_label(dict(c, D=0.05))["label"] == "NARROWS"                                              # HOLD-collapsed
    assert RD.chassis_label(dict(c, D=0.05), drop=("ceil_D",))["label"] == "CEILING"


def test_label_narrows_qualifier_and_conditions():
    c = mk(dB=BETTER, dB_seed1=-0.1)
    assert RD.chassis_label(c) == dict(label="NARROWS", qualifier="MEAN-ONLY")
    assert RD.chassis_label(dict(c, dM=BETTER, dM_seed1=-0.1))["qualifier"] == "INFORMATIVE"
    assert RD.chassis_label(dict(c, dM=BETTER, dM_seed1=0.1))["qualifier"] == "MEAN-ONLY"
    assert RD.chassis_label(dict(c, dM=BETTER, dM_seed1=0.1), drop=("inf_seed1",))["qualifier"] == "INFORMATIVE"
    assert RD.chassis_label(dict(c, dM=row(-0.1, -0.2, 0.01)), drop=("inf_ci",))["qualifier"] == "INFORMATIVE"
    assert RD.chassis_label(dict(c, dM=BETTER, dM_seed1=-0.1, dM_F=0.5), drop=())["qualifier"] == "MEAN-ONLY"
    assert RD.chassis_label(dict(c, dM=BETTER, dM_seed1=-0.1, dM_F=0.5), drop=("inf_floor",))["qualifier"] == "INFORMATIVE"
    assert RD.chassis_label(dict(c, dB=row(-0.1, -0.15, 0.01)))["label"] == "NULL"
    assert RD.chassis_label(dict(c, dB=row(-0.1, -0.15, 0.01)), drop=("narrows_ci",))["label"] == "NARROWS"
    assert RD.chassis_label(dict(c, dB_F=0.5))["label"] == "NULL"
    assert RD.chassis_label(dict(c, dB_F=0.5), drop=("narrows_floor",))["label"] == "NARROWS"
    assert RD.chassis_label(dict(c, dB_seed1=0.02))["label"] == "NULL"
    assert RD.chassis_label(dict(c, dB_seed1=0.02), drop=("narrows_seed1",))["label"] == "NARROWS"


def test_label_harm_and_guard():
    assert RD.chassis_label(mk(dB=WORSE_))["label"] == "HARM"
    assert RD.chassis_label(mk(dB=WORSE_), drop=("harm_dB_ci",))["label"] == "HARM"          # ci dropped: still worse by floor
    assert RD.chassis_label(mk(dB=row(0.1, -0.01, 0.2)))["label"] == "NULL"
    assert RD.chassis_label(mk(dB=row(0.1, -0.01, 0.2)), drop=("harm_dB_ci",))["label"] == "HARM"
    assert RD.chassis_label(mk(dB=row(0.005, 0.001, 0.01)))["label"] == "NULL"                # worse CI but inside the floor
    assert RD.chassis_label(mk(dB=row(0.005, 0.001, 0.01)), drop=("harm_dB_floor",))["label"] == "HARM"
    assert RD.chassis_label(mk(gx=WORSE_))["label"] == "HARM"
    assert RD.chassis_label(mk(gx=WORSE_), drop=("harm_guard",))["label"] == "NULL"
    c = mk(dH=BETTER, dH_seed1=-0.1, gx=WORSE_)
    assert RD.chassis_label(c)["label"] == "HARM"                                                 # a worse guard beats CEILING
    assert RD.chassis_label(c, drop=("harm_guard",))["label"] == "CEILING"


def test_combine_and_seeds_disagree():
    assert RD.combine(["NULL", "NULL"]) == dict(label="NULL", effective="NULL", seeds_disagree=False)
    r = RD.combine(["CEILING", "NULL"])
    assert r["label"] == "SEEDS-DISAGREE" and r["effective"] == "NULL"                          # the LESS favourable label's step applies
    assert RD.combine(["NARROWS", "HARM"])["effective"] == "HARM"
    assert RD.combine_qualifier(["INFORMATIVE", "MEAN-ONLY"]) == "MEAN-ONLY" and RD.combine_qualifier([None, "INFORMATIVE"]) == "INFORMATIVE"


def test_ceil_oracle_and_read_limited():
    ok = dict(row=BETTER, F=0.01, seed1=-0.1)
    assert RD.ceil_oracle({"s0": ok, "s1": ok})
    assert not RD.ceil_oracle({"s0": ok, "s1": dict(ok, row=NONE_)})                              # one chassis only -> fails
    assert not RD.ceil_oracle({"s0": ok, "s1": dict(ok, seed1=0.1)})
    assert RD.read_limited("NULL", False) and RD.read_limited("HARM", False) and RD.read_limited("NARROWS", False)
    assert not RD.read_limited("NULL", True) and not RD.read_limited("CEILING", False)


def test_next_step_rows():
    assert "READ-LIMITED" in RD.next_step("NULL", False, "A0")
    assert "RA-L2 under P23" in RD.next_step("NULL", True, "P") and "RA-L2 under P," in RD.next_step("NULL", True, "P23")
    assert "REG" in RD.next_step("NARROWS", True, "A0") and "READ-LIMITED" in RD.next_step("NARROWS", False, "A0")
    assert "wiring" in RD.next_step("HARM", True, "A0") and "CEILING" in RD.next_step("CEILING", True, "A0")


def test_finalize_invalid_and_multiplicity():
    good = {k: mk(dH=BETTER, dH_seed1=-0.1, dB=BETTER, dB_seed1=-0.1) for k in ("s0", "s1")}
    orc = {k: dict(row=BETTER, F=0.01, seed1=-0.1) for k in ("s0", "s1")}
    r = RD.finalize({"RAL1": good}, orc, True)
    assert r["arms"]["RAL1"]["name"] == "RAL1-CEILING" and not r["arms"]["RAL1"]["READ_LIMITED"] and "INVALID" not in r
    r = RD.finalize({"RAL1": good}, orc, False)
    assert r["arms"]["RAL1"]["name"] == "RAL1-INVALID" and "INVALID" in r                           # a RED control reads INVALID whatever the numbers say
    r = RD.finalize({"RAL1": good}, orc, True, extra_invalid=["C-SHUF on SCORE"])
    assert r["arms"]["RAL1"]["name"] == "RAL1-INVALID"
    nul = {k: mk() for k in ("s0", "s1")}
    r = RD.finalize({"RAL2": good, "RAL1": nul}, orc, True)
    assert r["multiplicity"]["guard_applies"] and r["arms"]["RAL2"]["name"] == "RAL2-CEILING"
    r = RD.finalize({"RAL2": nul}, {k: dict(row=NONE_, F=0.01, seed1=0.1) for k in ("s0", "s1")}, True)
    assert r["arms"]["RAL2"]["READ_LIMITED"] and r["arms"]["RAL2"]["effective"] == "NULL"


def _js(rows):
    return dict(rows=[dict(family="L", a="A", b="B", band=b, metric=m, delta=d, lo=lo, hi=hi) for (b, m, d, lo, hi) in rows])


def test_score_identity():
    cells = [(b, m, 0.0, 0.0, 0.0) for b in ("0-2s", "2-6s", "full6s") for m in ("yaw_mae", "head_err")] + [("2-6s", "acc_mae", 0.0, 0.0, 0.0)]
    assert RD.check_score_identity(_js(cells))["ok"]
    bad = list(cells); bad[-1] = ("2-6s", "acc_mae", 1e-9, 0.0, 0.0)
    assert not RD.check_score_identity(_js(bad))["ok"]                                              # a pass that wrote knots 11+ is RED
    assert not RD.check_score_identity(_js(cells[:3]))["ok"]                                        # missing cells are not a pass


# ---------------------------------------------------------------------------------------------- controls on the toy world
@pytest.fixture(scope="module")
def toy():
    ctx = K.toy_ctx()
    win, base = K.toy_world(ctx)
    return ctx, win, base


def test_plant_toy_and_regression(toy):
    ctx, win, base = toy
    g = K.check_plant_toy(ctx, (win, base))
    assert g["ok"] and g["analytic_cost_at_source_actions"] <= 1e-6 and g["analytic_closed_form_read_err"] <= 1e-6
    assert not K.check_plant_toy(ctx, (win, base), mutate="own_actions")["ok"]                      # a pass that cannot move stays at HOLD_c


def test_grad_and_regression(toy):
    ctx, win, base = toy
    z0 = torch.from_numpy(win["z"][:, 0]); hc = ctx.hold_c(win["A"][:, 0])[:, :12]
    src = torch.from_numpy(base["a1_raw"][:, :, 0] / R.ACC)
    t = ctx.ra.read(R.roll(ctx.st, z0, torch.stack([src, hc[:, :, 1]], -1).unsqueeze(1))[:, 0] - z0)
    g = K.grad_of(ctx.st, ctx.ra.read, z0, hc[:, :, 0], hc[:, :, 1], t, ctx.lo1, ctx.hi1)
    assert K.check_grad(g)["ok"]
    gb = K.grad_of(ctx.st, ctx.ra.read, z0, hc[:, :, 0], hc[:, :, 1], t, ctx.lo1, ctx.hi1, read_after=12)
    assert not K.check_grad(gb)["ok"] and K.check_grad(gb)["max_abs_knots_11_12"] > 0                # RA reading after 12 actions: knots 11-12 live


@pytest.mark.parametrize("mode,src", [("l1", "l1"), ("l2", "l2"), ("neg", "l1"), ("shuf", "l1"), ("oracle", "none")])
def test_yaw_band_identity_all_modes(toy, mode, src):
    ctx, win, base = toy
    new, built = R.plan_arrays(ctx, base, win, mode, src, "none", built={})
    assert K.check_yaw_band(new, base)["ok"] and K.check_yaw_band(new, base)["acc_1_10_moved"] > 0
    assert built["mirror"] == (mode == "neg") and built["perm"] == (mode == "shuf")


def test_yaw_band_mutations_red(toy):
    ctx, win, base = toy
    assert not K.check_yaw_band(R.plan_arrays(ctx, base, win, "l1", "l1", mutate="write12")[0], base)["ok"]
    assert not K.check_yaw_band(R.plan_arrays(ctx, base, win, "l1", "l1", mutate="write_yaw")[0], base)["ok"]


def test_off_identity_and_mutation(toy):
    ctx, win, base = toy
    assert K.check_off_equal(R.plan_arrays(ctx, base, win, "off")[0], base)["ok"]
    assert not K.check_off_equal(R.plan_arrays(ctx, base, win, "off", "none", "none", mutate="off_runs_pass")[0], base)["ok"]


def test_mean_zero_and_guard(toy):
    ctx, win, base = toy
    hc = ctx.hold_c(win["A"][:, 0])[:, :10, 0].numpy() * R.ACC
    assert K.check_mean_zero(hc, -1e9, 1e9)["ok"]
    src, logged = base["a1_raw"][:, :10, 0], win["A"][:, :10, 0]
    assert K.check_mean_guard(src, hc, -1e9, 1e9, logged)["ok"]
    assert not K.check_mean_guard(src, hc, -1e9, 1e9, logged, mutate=True)["ok"]                    # a shift taken from the logged accelerations is refused
    new, built = R.plan_arrays(ctx, base, win, "off", "none", "l1", built={})
    assert built["kind"] == "mean" and np.allclose(new["plan_raw"][:, :10, 0].mean(1), np.clip(src.mean(1), -3 * R.ACC, 3 * R.ACC))
    assert K.check_yaw_band(new, base)["ok"]


def test_mean_uses_no_model():
    out = R.mean_assembly(np.full((2, 10), 5.0), np.full((2, 10), 1.0), -2.0, 2.0)
    assert (out == 2.0).all()                                                                       # clipped to B1


def test_l4_argv_and_built(toy):
    ctx, win, base = toy
    b = ST.build("train", "RAL1", "A0", "s0")
    assert K.check_l4(b["argv"], b["declared"])["ok"]
    l2 = list(b["argv"]); l2[l2.index("--ra-l1") + 1] = "l2"
    assert not K.check_l4(l2, b["declared"])["ok"]
    un = [t for i, t in enumerate(b["argv"]) if t != "--ra-l1" and b["argv"][i - 1] != "--ra-l1"]
    assert not K.check_l4(un, b["declared"])["ok"]                                                  # the default `off` is never a silent choice
    dec = dict(b["declared"], readout_sha256="toy")
    assert K.check_l4_built(ctx, base, win, "l1", "l1", dec)["ok"]
    assert not K.check_l4_built(ctx, base, win, "l1", "l1", dec, mutate="wrong_src")["ok"]
    assert not K.check_l4_built(ctx, base, win, "l1", "l1", dict(dec, readout_sha256="other"))["ok"]                # a different readout than declared
    sh = ST.build("train", "SHUF", "A0", "s0", src_for_nl="l1")
    sdec = dict(sh["declared"], readout_sha256="toy")
    assert K.check_l4_built(ctx, base, win, "shuf", "l1", sdec)["ok"]
    assert not K.check_l4_built(ctx, base, win, "l1", "l1", sdec)["ok"]                   # built mode l1 != declared shuf


def test_fresh_out(tmp_path):
    assert K.check_fresh_out_pre(str(tmp_path / "a"), False)["ok"]
    assert not K.check_fresh_out_pre(str(tmp_path / "b"), True)["ok"]


def test_ra_sha(tmp_path):
    p = ST.readout_path("s0")
    assert K.check_ra_sha(p, "s0")["ok"]
    assert not K.check_ra_sha(p, "s0", flip_byte=True, tmp_dir=str(tmp_path))["ok"]


def test_ra_fit_checks():
    rng = np.random.RandomState(0)
    Dr, Yf = rng.randn(300, 6), rng.randn(300, 1)
    fit = R.refit_ridge(Dr, Yf, 0.5, [f"c{i}" for i in range(10)])
    ro = {x: fit[x] for x in ("mu", "sd", "W", "ym")}
    assert R.check_ra_fit(fit, ro, ["s1", "s2"], n_pairs_record=300, n_clips_record=10)["ok"]
    plus = R.refit_ridge(np.vstack([Dr, rng.randn(1, 6)]), np.vstack([Yf, [[3.0]]]), 0.5, fit["clips"])
    assert not R.check_ra_fit(plus, ro, ["s1", "s2"], n_pairs_record=300, n_clips_record=10)["ok"]   # one SCORE window added moves W
    assert not R.check_ra_fit(fit, ro, ["c3"], n_pairs_record=300, n_clips_record=10)["ok"]          # a planted SCORE eid


def test_train_iso_and_in(tmp_path):
    assert K.check_train_iso(["a", "b"], ["x"], "p", "p")["ok"]
    assert not K.check_train_iso(["a", "x"], ["x"], "p", "p")["ok"]
    f = tmp_path / "f.bin"; f.write_bytes(b"abc")
    import hashlib
    h = hashlib.sha256(b"abc").hexdigest()
    assert K.check_in({"f": (str(f), h)})["ok"] and not K.check_in({"f": (str(f), "0" * 64)})["ok"]
    assert K.verdict(dict(ok=True), dict(ok=False))["passed"] and not K.verdict(dict(ok=True), dict(ok=True))["passed"]


def test_shuf_sign():
    assert K.check_shuf_sign(row(0.3, 0.1, 0.5))["ok"] and not K.check_shuf_sign(row(0.3, -0.1, 0.5))["ok"]
    assert K.check_shuf_sign(row(0.3, -0.1, 0.5), separated=False)["ok"] and not K.check_shuf_sign(row(-0.3, -0.5, -0.1), separated=False)["ok"]


def test_plant_real_bar():
    assert K.check_plant_real(np.full(10, 1e-6))["ok"] and not K.check_plant_real(np.full(10, 1e-2))["ok"]


# ---------------------------------------------------------------------------------------------- stage rules
def _bs(tmp_path, txt):
    p = tmp_path / "BASE_SOLVER_AMEND5.md"; p.write_text(txt, encoding="utf-8"); return str(p)


def test_base_solver_file(tmp_path):
    assert ST.read_base_solver(_bs(tmp_path, "x\nBASE_SOLVER: A0\nBRANCH: B3\n"))["base"] == "A0"
    assert ST.read_base_solver(_bs(tmp_path, "- **BASE_SOLVER:** P23\n- BRANCH: B1\n"))["base"] == "P23"
    with pytest.raises(ST.NoBaseSolver):
        ST.read_base_solver(str(tmp_path / "missing.md"))
    with pytest.raises(ST.NoBaseSolver):
        ST.read_base_solver(_bs(tmp_path, "BASE_SOLVER: P\nBRANCH: B3\n"))                          # branch and base disagree
    with pytest.raises(ST.NoBaseSolver):
        ST.read_base_solver(_bs(tmp_path, "BASE_SOLVER: A0\nBASE_SOLVER: P\nBRANCH: B3\n"))


def test_score_refused_without_base_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "BASE_SOLVER_FILE", str(tmp_path / "none.md"))
    b = ST.build("score", "RAL1", "A0", "s0", out=str(tmp_path / "o"))
    a = R.check_declared(b["argv"], b["declared"])
    with pytest.raises(ST.NoBaseSolver):
        R.check_stage_rules(a, b["declared"])
    o = ST.build("score", "OFF", "A0", "s0", out=str(tmp_path / "o2"))                              # the identity arm (C-OFF5) needs no solver
    assert R.check_stage_rules(R.check_declared(o["argv"], o["declared"]), o["declared"]) is None
    m = ST.build("score", "MEANL1", "A0", "s0", out=str(tmp_path / "o3"))
    with pytest.raises(ST.NoBaseSolver):
        R.check_stage_rules(R.check_declared(m["argv"], m["declared"]), m["declared"])


def test_base_must_be_the_solver_and_beside_rule():
    sol = dict(base="A0", branch="B3", path="p", sha256="x")
    b = ST.build("score", "RAL2", "P", "s0", out="o")
    a = R.check_declared(b["argv"], b["declared"])
    with pytest.raises(R.RefusedRun):
        R.check_stage_rules(a, b["declared"], sol)                                                  # P is not the base solver
    a.beside = True
    assert R.check_stage_rules(a, b["declared"], sol) == sol
    sol2 = dict(base="P", branch="B1", path="p", sha256="x")
    with pytest.raises(R.RefusedRun):
        R.check_stage_rules(a, b["declared"], sol2)                                                 # beside is B3 only
    r1 = ST.build("score", "RAL1", "P23", "s0", out="o")
    with pytest.raises(R.RefusedRun):
        R.check_stage_rules(R.check_declared(r1["argv"], r1["declared"]), r1["declared"], sol2)


def test_ral2_not_on_a0():
    with pytest.raises(ValueError):
        ST.build("train", "RAL2", "A0", "s0")
    with pytest.raises(ValueError):
        ST.build("train", "NEG", "A0", "s0")                                                        # NEG needs the committed source


def test_code_sha_pin_is_full_length():
    assert all(len(v) == 64 for v in ST.CODE_SHA_FULL.values()) and ST.CODE_SHA_FULL["hwm_plan.py"].startswith("5a0722719f7c")


# ---------------------------------------------------------------------------------------------- the wrapper end to end on the toy world
def test_run_arm_toy_end_to_end(toy, tmp_path):
    ctx, win, base = toy
    d = tmp_path / "base"; d.mkdir()
    np.savez(d / "plans_hwm_oracle_s0.npz", **base)
    np.savez(tmp_path / "win.npz", **win)
    out = str(tmp_path / "out")
    b = ST.build("train", "RAL1", "A0", "s0", out=out)
    b["declared"] = dict(b["declared"], readout_sha256="toy")
    import hashlib
    ws = str(tmp_path / "win.npz"); wsha = hashlib.sha256(open(ws, "rb").read()).hexdigest()
    sol = R.run_arm(b["argv"], b["declared"], ctx=ctx, windows_src=ws, windows_sha=wsha, enforce_threads=False, base_plan_fn=lambda s: str(d / "plans_hwm_oracle_s0.npz"))
    z = np.load(os.path.join(out, "plans_hwm_oracle_s0.npz"))
    assert str(z["ra_l1_planned"]) == "l1" and K.check_yaw_band(z, base)["ok"] and os.path.exists(os.path.join(out, "SOLVER_RA.json")) and os.path.exists(os.path.join(out, "windows.npz"))
    with pytest.raises(R.RefusedRun):                                                               # a used folder is refused
        R.run_arm(b["argv"], b["declared"], ctx=ctx, windows_src=ws, windows_sha=wsha, enforce_threads=False, base_plan_fn=lambda s: str(d / "plans_hwm_oracle_s0.npz"))
    with pytest.raises(R.RefusedRun):                                                               # a wrong windows sha256 is refused
        R.run_arm(ST.build("train", "RAL1", "A0", "s0", out=str(tmp_path / "o2"))["argv"], b["declared"], ctx=ctx, windows_src=ws, windows_sha="0" * 64, enforce_threads=False, base_plan_fn=lambda s: str(d / "plans_hwm_oracle_s0.npz"))
    o = ST.build("train", "OFF", "A0", "s0", out=str(tmp_path / "off"))
    o["declared"] = dict(o["declared"], readout_sha256="toy")
    R.run_arm(o["argv"], o["declared"], ctx=ctx, windows_src=ws, windows_sha=wsha, enforce_threads=False, base_plan_fn=lambda s: str(d / "plans_hwm_oracle_s0.npz"))
    assert K.check_off_equal(dict(np.load(str(tmp_path / "off" / "plans_hwm_oracle_s0.npz"))), base)["ok"]
    with pytest.raises(R.RefusedRun):                                                               # mutation hooks need a toy ctx and are controls only
        R.run_arm(b["argv"], b["declared"], mutate="write12", windows_src=ws, windows_sha=wsha)


def test_thread_rule(toy, tmp_path, monkeypatch):
    ctx, win, base = toy
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    b = ST.build("train", "RAL1", "A0", "s0", out=str(tmp_path / "o"))
    with pytest.raises(R.RefusedRun):
        R.run_arm(b["argv"], b["declared"], ctx=ctx, windows_src="x", windows_sha="y")
