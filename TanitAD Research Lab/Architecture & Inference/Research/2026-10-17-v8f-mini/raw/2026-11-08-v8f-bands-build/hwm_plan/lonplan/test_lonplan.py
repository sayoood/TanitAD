"""LON-PLAN controls (AMEND_HWMPLAN_3 section 8.2), each with its deliberate-regression arm that must go RED. CPU only. Run through run_tests.cmd / the commands in BUILD_REPORT_LONPLAN.md.
C-L1 reach, C-L2 level-1-only identity, C-L3 limit, C-L4 argv -> built (the run wrapper), C-READBACK, C-REG, C-TRAIN-ISO, the selection procedure (section 3), C-STRAT.
The toy world is synthetic (PLANTEST's planted stack, copied from test_hwm_plan.py: _Cat, _Shim, toy). The run path is the code of record's: `run_hwm.cmd_plan` -> `_init` -> `make_planner` -> `HWMPlanner` -> `plan_windows`;
only `run_hwm.prepare` (model and prior loading) is replaced by the toy stack. No SCORE window and no TRAIN-selection window of R35 is planned here. Tier T0 synthetic."""
import contextlib, dataclasses, json, os, shutil, sys, types
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lonplan_common as C
import numpy as np
import pytest
import torch
import torch.nn as nn
import hwm_plan as H
import hwm_eval as E
import hwm_data as D
import run_hwm as R
import plantest_lib as P
import plantest2_lib as Q
import lonplan_run as W
import lonplan_select as S
import lonplan_cstrat as CS

TOY_GEOM = H.Geometry(2, 2, 3, 3, 3, 12, 6).check()            # level 1 plans 6 steps, goal frame 12, knots 1-6 from level 1 and 7-12 from the level-2 decode
N_OP = TOY_GEOM.n_op
FAST = ["--S", "4", "--n-iters", "12"]
REAL = dict(
    windows="F:/TanitAD-scratch/actsens/read/out/C7_R35_s0/win",
    windows_sha="371c753bcc3f28f1052c0e5e7e4898c0bbbed6140477b3338c8133a3632bdaee")


# ------------------------------------------------------------------------------------------ the toy world (as test_hwm_plan.py)
class _Cat(nn.Module):
    def __init__(self, stride, e):
        super().__init__(); self.stride, self.embed_dim = stride, e

    def forward(self, e):
        return e.reshape(*e.shape[:-2], self.stride * e.shape[-1])


class _Shim:
    def __init__(self, st):
        self.stack, self.ch1_scale = st, 1.0

    def scale_actions(self, a):
        return a


def toy():
    st, _ = P.planted_stack()
    st.macro_encoders[0] = _Cat(2, 2); st.macro_encoders[1] = _Cat(2, 4)
    P.planted_fit_priors(st)
    return _Shim(st)


class Env:
    """The toy stack, its windows and the patched run_hwm.prepare."""
    def __init__(self, n=80, seed=777):
        self.wm = toy(); self.st = self.wm.stack
        g = torch.Generator().manual_seed(seed)
        self.z0 = torch.randn(n, 32, generator=g)
        self.a = P.planted_actions(n, seed + 1)
        self.zT = P.planted_roll(self.st, self.z0, self.a)
        self.A0 = (0.2 * torch.randn(n, 2, generator=g)).numpy().astype(np.float64)
        self.n = n

    def windows(self):
        F = TOY_GEOM.goal_frame + 1
        z = np.zeros((self.n, F, 32), np.float32); z[:, 0] = self.z0.numpy(); z[:, TOY_GEOM.goal_frame] = self.zT.numpy()
        A = np.zeros((self.n, F, 2), np.float64); A[:, 0] = self.A0; A[:, 1:13, :] = self.a.numpy().astype(np.float64)
        return dict(z=z, A=A, eid=np.array([f"ep{i % 8}" for i in range(self.n)], dtype="U36"), now=np.arange(self.n), v0=np.zeros(self.n), donor=np.arange(self.n), meta=json.dumps({"toy": True}))

    def prepare(self, a, ws=None):
        return None, self.wm, self.st, TOY_GEOM


@pytest.fixture(scope="module")
def env():
    return Env()


@pytest.fixture()
def toyrun(env, monkeypatch, tmp_path):
    """A fresh folder with the toy windows, dummy checkpoint and latents files, and run_hwm.prepare replaced by the toy stack."""
    monkeypatch.setattr(R, "prepare", env.prepare)
    wsrc = tmp_path / "windows_src.npz"; np.savez(wsrc, **env.windows())
    (tmp_path / "ckpt.pt").write_bytes(b"toy-checkpoint"); (tmp_path / "lat.pt").write_bytes(b"toy-latents")
    tr = types.SimpleNamespace(env=env, tmp=tmp_path, wsrc=str(wsrc), wsha=C.sha256_file(wsrc), n=0)

    def argv(lam, out=None, extra=(), arms=("hwm_oracle",), name=None):
        tr.n += 1
        out = out or str(tmp_path / (name or f"run{tr.n}"))
        return ["--tag", "toy", "--ckpt", str(tmp_path / "ckpt.pt"), "--train-latents", str(tmp_path / "lat.pt"), "--train-tag", "toy", "--seed-mode", "hold", f"--lam={lam!r}", *FAST,
                "--arms", *arms, "--seeds", "0", "--workers", "1", "--out", out, *extra]

    def wrapper(lam, **kw):
        av = argv(lam, **kw)
        W.run_lambda(av, lam, "hold", tr.wsrc, tr.wsha, enforce_threads=False)
        return av[av.index("--out") + 1]

    def raw(lam, **kw):
        """The code of record's own run path, no wrapper: parse -> windows into the folder -> run_hwm.cmd_plan."""
        av = argv(lam, **kw)
        a = W.parse(av)
        os.makedirs(a.out, exist_ok=True); shutil.copyfile(tr.wsrc, os.path.join(a.out, "windows.npz"))
        R.cmd_plan(a)
        return a.out

    tr.argv, tr.wrapper, tr.raw = argv, wrapper, raw
    return tr


def plan_of(out, arm="hwm_oracle", seed=0):
    with np.load(os.path.join(out, f"plans_{arm}_s{seed}.npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------ C-L1 reach (section 8.2)
def control_l1(tr):
    """lambda = 1 changes level-1 knots against lambda = 0 at the same seed, through run_hwm's own run path; the REG gradient on the level-1 actions is non-zero; lambda = 0 equals the code-of-record plan bit for bit
    and equals a run whose reg_term is patched to 0.0 (here at lambda = 1)."""
    p0, p1 = plan_of(tr.raw(0.0))["plan_raw"], plan_of(tr.raw(1.0))["plan_raw"]
    res = dict(max_abs_diff_level1_knots=float(np.abs(p1[:, :N_OP] - p0[:, :N_OP]).max()))
    res["changes_level1"] = res["max_abs_diff_level1_knots"] > 0
    ref = torch.zeros(4, 6, 2); a = (ref + 0.3).unsqueeze(1).expand(4, 3, 6, 2).clone().requires_grad_(True)
    std = Q.prior_std_orig(tr.env.st)
    try:
        Q.reg_term(a, ref, std, 1.0).sum().backward()
        res["grad_abs_max"] = float(a.grad.abs().max())
    except RuntimeError:
        res["grad_abs_max"] = 0.0                                              # a reg_term with no autograd graph (a patched zero) cannot carry a gradient
    res["grad_nonzero"] = res["grad_abs_max"] > 0
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Q, "reg_term", lambda a, ref, std, lam: torch.zeros(a.shape[:2], dtype=a.dtype))
        pz = plan_of(tr.raw(1.0))["plan_raw"]
    res["lam0_equals_patched_zero_reg"] = bool(np.array_equal(p0, pz))
    rec = H.HWMPlanner(tr.env.wm, TOY_GEOM, H.SolverCfg(seed_mode="hold", lam=0.0, S=4, n_iters=12)).plan_windows(
        "hier", lambda ii: tr.env.zT[torch.as_tensor(ii)], tr.env.z0, tr.env.A0, idx=np.arange(tr.env.n), seed=0, batch0=0)
    res["lam0_equals_code_of_record"] = bool(np.array_equal(p0, rec["plan_raw"]))
    res["ok"] = bool(res["changes_level1"] and res["grad_nonzero"] and res["lam0_equals_patched_zero_reg"] and res["lam0_equals_code_of_record"])
    return res


def test_c_l1_reach_green(toyrun):
    r = control_l1(toyrun)
    print("C-L1", r)
    assert r["ok"], r
    assert r["max_abs_diff_level1_knots"] > 1e-3


def test_c_l1_mutation_solver_from_pins_lam_to_zero_goes_red(toyrun, monkeypatch):
    orig = R.solver_from
    monkeypatch.setattr(R, "solver_from", lambda a: dataclasses.replace(orig(a), lam=0.0))
    r = control_l1(toyrun)
    print("C-L1 mutant (solver_from drops lam)", r["changes_level1"], r["max_abs_diff_level1_knots"])
    assert not r["ok"] and not r["changes_level1"] and r["max_abs_diff_level1_knots"] == 0.0


def test_c_l1_mutation_plan_batch_builds_reg_none_goes_red(toyrun, monkeypatch):
    orig = H.HWMPlanner.plan_batch

    def mutant(self, *a, **k):
        self.solver = dataclasses.replace(self.solver, lam=0.0)               # hop 5: `reg = ... if lam != 0.0 else None` sees 0.0
        return orig(self, *a, **k)
    monkeypatch.setattr(H.HWMPlanner, "plan_batch", mutant)
    r = control_l1(toyrun)
    assert not r["ok"] and not r["changes_level1"]


def test_c_l1_mutation_reg_term_off_goes_red(toyrun, monkeypatch):
    monkeypatch.setattr(Q, "reg_term", lambda a, ref, std, lam: torch.zeros(a.shape[:2], dtype=a.dtype))
    r = control_l1(toyrun)
    assert not r["ok"] and not r["changes_level1"] and not r["grad_nonzero"]


# ------------------------------------------------------------------------------------------ C-L2 level-1-only identity
def _planner(tr, lam, plan_fn=None):
    a = W.parse(tr.argv(lam))
    w = tr.env.windows()
    ws = types.SimpleNamespace(z=torch.from_numpy(w["z"]), A=w["A"], v0=w["v0"], eid=w["eid"], now=w["now"], n=tr.env.n)
    pl = R.make_planner(a, tr.env.wm, tr.env.st, TOY_GEOM, ws)
    if plan_fn is not None:
        pl.plan_fn = plan_fn
    return pl


def _batch(tr, pl):
    ii = np.arange(16)
    hc = pl.hold_c(tr.env.A0)[ii]
    return pl.plan_batch("hier", tr.env.z0[ii], tr.env.zT[ii], hc, P.make_gen(0, 0), ii=ii), ii


def control_l2(tr, plan_fn=None):
    """hier + decode, same seed: knots 7-12 (here the decoded part; 11-30 on R35), level-2 and level-3 actions and costs bit-equal between lambda 1 and 0, knots 1-6 (1-10 on R35) differ."""
    out = {}
    for lam in (0.0, 1.0):
        pl = _planner(tr, lam, plan_fn)
        res, ii = _batch(tr, pl)
        traj = pl.plan_windows("hier", lambda j: tr.env.zT[torch.as_tensor(j)], tr.env.z0, tr.env.A0, idx=ii, seed=0, batch0=0)["plan_raw"]
        out[lam] = (res, traj)
    (r0, t0), (r1, t1) = out[0.0], out[1.0]
    res = dict(level1_knots_differ=bool(np.abs(t1[:, :N_OP] - t0[:, :N_OP]).max() > 0),
               decoded_knots_equal=bool(np.array_equal(t1[:, N_OP:], t0[:, N_OP:])),
               level23_actions_equal=all(bool(torch.equal(r1.level_actions[k], r0.level_actions[k])) for k in (2, 3)),
               level23_costs_equal=all(bool(torch.equal(r1.costs[k], r0.costs[k])) for k in (2, 3)))
    res["ok"] = all(res.values())
    return res


def level2_reg_mutant_plan_fn():
    """plan_topdown2 with REG also applied at level 2 (the mutation): the library source with one added block, executed in its own namespace."""
    src = open(Q.__file__, encoding="utf-8").read()
    marker = "                best_a, best_cost, init_cost, hist, nit = _opt(level, horizon, cost_fn, n_cand)"
    add = ("                if level == 2 and reg is not None and reg[\"lam\"] != 0.0:\n"
           "                    _b2 = cost_fn\n"
           "                    def cost_fn(a, _base=_b2):\n"
           "                        return _base(a) + reg[\"lam\"] * (a - seeds[2].unsqueeze(1)).pow(2).mean(dim=(-1, -2))\n")
    assert src.count(marker) == 1
    ns = {"__file__": Q.__file__, "__name__": "plantest2_lib_l2_mutant"}
    exec(compile(src.replace(marker, add + marker), Q.__file__, "exec"), ns)
    return ns["plan_topdown2"]


def test_c_l2_level1_only_identity_green(toyrun):
    r = control_l2(toyrun)
    print("C-L2", r)
    assert r["ok"], r


def test_c_l2_mutation_reg_also_at_level_2_goes_red(toyrun):
    r = control_l2(toyrun, plan_fn=level2_reg_mutant_plan_fn())
    print("C-L2 mutant", r)
    assert not r["ok"] and not r["level23_actions_equal"]


# ------------------------------------------------------------------------------------------ C-L3 limit
LIMIT_TOL = 1e-4


def control_l3(tr, plan_fn=None):
    """lambda = LAMBDA_LIMIT through HWMPlanner: the level-1 knots equal HOLD_c within 1e-4 prior std, hier and flat."""
    dev = {}
    for mode, arm in (("hier", "hwm_oracle"), ("flat", "flat_oracle")):
        a = W.parse(tr.argv(Q.LAMBDA_LIMIT, arms=(arm,), extra=["--n-iters", "60"]))
        w = tr.env.windows()
        ws = types.SimpleNamespace(z=torch.from_numpy(w["z"]), A=w["A"], v0=w["v0"], eid=w["eid"], now=w["now"], n=tr.env.n)
        pl = R.make_planner(a, tr.env.wm, tr.env.st, TOY_GEOM, ws)
        if plan_fn is not None:
            pl.plan_fn = plan_fn
        ii = np.arange(16)
        hc = pl.hold_c(tr.env.A0)[ii]
        goal = tr.env.zT[ii]
        res = pl.plan_batch(mode, tr.env.z0[ii], goal, hc, P.make_gen(0, 0), ii=ii)
        h = TOY_GEOM.h1 if mode == "hier" else TOY_GEOM.goal_frame
        std = Q.prior_std_orig(tr.env.st)
        dev[mode] = float(((res.actions[:, :h] - hc[:, :h]).abs() / std).max())
    return dict(dev=dev, ok=all(v <= LIMIT_TOL for v in dev.values()))


def test_c_l3_limit_green(toyrun):
    r = control_l3(toyrun)
    print("C-L3", r)
    assert r["ok"], r


def test_c_l3_mutation_reg_reference_zeros_goes_red(toyrun):
    def zero_ref(st, *args, reg=None, **kw):
        if reg is not None:
            reg = dict(reg, ref=torch.zeros_like(reg["ref"]))
        return Q.plan_topdown2(st, *args, reg=reg, **kw)
    r = control_l3(toyrun, plan_fn=zero_ref)
    print("C-L3 mutant", r)
    assert not r["ok"] and all(v > LIMIT_TOL for v in r["dev"].values())


# ------------------------------------------------------------------------------------------ C-L4 argv -> built (the run wrapper)
def test_c_l4_green_records_lam_ckpt_windows_and_every_plan_file_reads_back_its_lam(toyrun):
    for lam in (0.0, 1.0, -1.0):
        out = toyrun.wrapper(lam)
        sol = json.load(open(os.path.join(out, "SOLVER.json"), encoding="utf-8"))
        assert sol["lam"] == lam and sol["seed_mode"] == "hold" and f"--lam={lam!r}" in sol["argv"]
        assert sol["ckpt_sha256"] == C.sha256_file(toyrun.tmp / "ckpt.pt") and sol["windows_sha256"] == toyrun.wsha
        assert sol["train_latents_sha256"] == C.sha256_file(toyrun.tmp / "lat.pt") and "run_hwm.py" in sol["code_sha256"] and sol["code_sha256"]["run_hwm.py"]
        assert "lam=" in sol["solver"] and f"lam={lam}" in sol["solver"].replace(" ", "")
        got = W.verify_plan_lambdas(out)
        assert got and all(v == lam for v in got.values())
        assert W.read_planned_lam(os.path.join(out, "plans_hwm_oracle_s0.npz")) == lam
        assert sol["plans"]["plans_hwm_oracle_s0.npz"]["sha256"] == C.sha256_file(os.path.join(out, "plans_hwm_oracle_s0.npz"))
        assert C.sha256_file(os.path.join(out, "windows.npz")) == toyrun.wsha


def test_c_l4_each_plan_file_records_its_own_lam_not_a_neighbours(toyrun):
    o1, o2 = toyrun.wrapper(1.0), toyrun.wrapper(3.0)
    assert W.read_planned_lam(os.path.join(o1, "plans_hwm_oracle_s0.npz")) == 1.0 and W.read_planned_lam(os.path.join(o2, "plans_hwm_oracle_s0.npz")) == 3.0
    assert not np.array_equal(plan_of(o1)["plan_raw"], plan_of(o2)["plan_raw"])


def test_c_l4_refuses_a_folder_that_already_holds_plans_and_leaves_them_untouched(toyrun):
    out = toyrun.wrapper(1.0)
    f = os.path.join(out, "plans_hwm_oracle_s0.npz"); before = C.sha256_file(f)
    with pytest.raises(W.RefusedRun, match="not empty"):
        W.run_lambda(toyrun.argv(2.0, out=out), 2.0, "hold", toyrun.wsrc, toyrun.wsha, enforce_threads=False)
    assert C.sha256_file(f) == before and W.read_planned_lam(f) == 1.0


def test_c_l4_refuses_a_preplanted_plan_file_in_an_otherwise_new_folder(toyrun):
    out = str(toyrun.tmp / "pre"); os.makedirs(out)
    np.savez(os.path.join(out, "plans_hwm_oracle_s0.npz"), plan_raw=np.zeros((1, 12, 2)), cost1=np.zeros(1))
    with pytest.raises(W.RefusedRun, match="plan files"):
        W.run_lambda(toyrun.argv(1.0, out=out), 1.0, "hold", toyrun.wsrc, toyrun.wsha, enforce_threads=False)


def test_c_l4_mutation_the_skip_trap_is_real_without_the_wrapper(toyrun):
    """The reason for the refusal: the code of record's plan stage skips an existing plans_<arm>_s<seed>.npz whatever its lambda, so a lambda run in a lambda = 0 folder banks nothing new."""
    o = toyrun.raw(0.0)
    a = W.parse(toyrun.argv(1.0, out=o))
    before = plan_of(o)["plan_raw"].copy()
    R.cmd_plan(a)
    assert np.array_equal(before, plan_of(o)["plan_raw"]), "run_hwm plan re-planned over an existing file"


def test_c_l4_refuses_a_declared_lam_that_the_argv_does_not_build(toyrun):
    av = toyrun.argv(0.0)
    with pytest.raises(W.RefusedRun, match="declared lam"):
        W.run_lambda(av, 1.0, "hold", toyrun.wsrc, toyrun.wsha, enforce_threads=False)           # argv says 0.0, declared 1.0
    with pytest.raises(W.RefusedRun, match="seed_mode"):
        W.run_lambda(toyrun.argv(0.0), 0.0, "prior", toyrun.wsrc, toyrun.wsha, enforce_threads=False)


def test_c_l4_refuses_an_argv_that_does_not_name_lam_or_seed_mode(toyrun):
    av = [t for t in toyrun.argv(1.0) if not t.startswith("--lam")]
    with pytest.raises(W.RefusedRun, match="--lam is not named"):
        W.run_lambda(av, 0.0, "hold", toyrun.wsrc, toyrun.wsha, enforce_threads=False)
    av = toyrun.argv(1.0); i = av.index("--seed-mode"); del av[i:i + 2]
    with pytest.raises(W.RefusedRun, match="--seed-mode is not named"):
        W.run_lambda(av, 1.0, "hold", toyrun.wsrc, toyrun.wsha, enforce_threads=False)


def test_c_l4_mutation_solver_from_drops_lam_is_caught_by_the_wrapper(toyrun, monkeypatch):
    orig = R.solver_from
    monkeypatch.setattr(R, "solver_from", lambda a: dataclasses.replace(orig(a), lam=0.0))
    with pytest.raises(W.RefusedRun, match="declared lam"):
        toyrun.wrapper(1.0)


def test_c_l4_refuses_wrong_windows_sha_wrong_ckpt_sha_levers_off_default_and_wrong_threads(toyrun, monkeypatch):
    with pytest.raises(W.RefusedRun, match="windows source sha256"):
        W.run_lambda(toyrun.argv(1.0), 1.0, "hold", toyrun.wsrc, "0" * 64, enforce_threads=False)
    with pytest.raises(W.RefusedRun, match="checkpoint sha256"):
        W.run_lambda(toyrun.argv(1.0), 1.0, "hold", toyrun.wsrc, toyrun.wsha, ckpt_sha256="1" * 64, enforce_threads=False)
    with pytest.raises(W.RefusedRun, match="levers off"):
        W.run_lambda(toyrun.argv(1.0, extra=["--stitch", "l1_full"]), 1.0, "hold", toyrun.wsrc, toyrun.wsha, enforce_threads=False)
    with pytest.raises(W.RefusedRun, match="arms"):
        W.run_lambda(toyrun.argv(1.0, arms=("hwm_plan",)), 1.0, "hold", toyrun.wsrc, toyrun.wsha, enforce_threads=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    with pytest.raises(W.RefusedRun, match="OMP_NUM_THREADS"):
        W.run_lambda(toyrun.argv(1.0), 1.0, "hold", toyrun.wsrc, toyrun.wsha)


def test_c_l4_refuses_prior_seed_mode_with_nonzero_lam(toyrun):
    av = toyrun.argv(1.0); av[av.index("--seed-mode") + 1] = "prior"
    with pytest.raises(W.RefusedRun, match="HOLD seed"):
        W.run_lambda(av, 1.0, "prior", toyrun.wsrc, toyrun.wsha, enforce_threads=False)


def test_c_l4_a_plan_file_without_a_recorded_lam_fails_the_readback_of_the_record(toyrun):
    out = toyrun.wrapper(1.0)
    f = os.path.join(out, "plans_hwm_oracle_s0.npz"); d = plan_of(out); d.pop("lam_planned")
    np.savez(f, **d)
    assert W.read_planned_lam(f) is None
    with pytest.raises(W.RefusedRun, match="records lam None"):
        W.verify_plan_lambdas(out)


# ------------------------------------------------------------------------------------------ C-READBACK
def test_c_readback_green_batch_0_replans_bit_equal(toyrun):
    out = toyrun.wrapper(1.0)
    r = W.readback(out, "hwm_oracle", 0)
    print("C-READBACK", r)
    assert r["equal"] and r["n"] == 64 and r["lam_used"] == 1.0 and r["plan_equal"] and r["cost1_equal"]


def test_c_readback_regression_the_file_read_back_at_lam_zero_fails(toyrun):
    out = toyrun.wrapper(1.0)
    r = W.readback(out, "hwm_oracle", 0, lam_override=0.0)
    print("C-READBACK regression", r)
    assert not r["equal"] and r["lam_used"] == 0.0


def test_c_readback_regression_a_tampered_plan_file_fails(toyrun):
    out = toyrun.wrapper(1.0)
    f = os.path.join(out, "plans_hwm_oracle_s0.npz"); d = plan_of(out)
    d["plan_raw"] = d["plan_raw"].copy(); d["plan_raw"][3, 2, 0] += 1e-9
    np.savez(f, **d)
    assert not W.readback(out, "hwm_oracle", 0)["equal"]


def test_c_readback_regression_a_different_thread_count_is_refused(toyrun):
    out = toyrun.wrapper(1.0)
    sol = json.load(open(os.path.join(out, "SOLVER.json"), encoding="utf-8")); sol["torch_threads"] = int(torch.get_num_threads()) + 1
    C.dump_json(sol, os.path.join(out, "SOLVER.json"))
    with pytest.raises(W.RefusedRun, match="torch threads"):
        W.readback(out, "hwm_oracle", 0)


def test_c_readback_all_covers_every_plan_file(toyrun):
    out = toyrun.wrapper(1.0, arms=("hwm_oracle", "flat_oracle"))
    res = W.readback_all(out)
    assert sorted(res) == ["plans_flat_oracle_s0.npz", "plans_hwm_oracle_s0.npz"] and all(v["equal"] for v in res.values())


# ------------------------------------------------------------------------------------------ C-REG, sign check
def test_c_reg_check_green_and_regression_arms():
    good = dict(lo=0.01, label="worse")
    assert S.check_creg(good, 1.2, 0.8)
    assert not S.check_creg(dict(lo=-0.01, label="none"), 1.2, 0.8), "a non-separated R - A0 must fail C-REG"
    assert not S.check_creg(dict(lo=0.01, label="worse"), 0.7, 0.8), "D(R) <= D(A0) must fail C-REG"
    assert not S.check_creg(dict(lo=-0.5, label="better"), 1.2, 0.8)


def test_negative_lam_pushes_plans_farther_from_hold_on_the_toy_run_path(toyrun):
    """The direction C-REG reads on SCORE: D(R, lam < 0) > D(A0, lam = 0) > D(A1, lam > 0), through the wrapper."""
    w = toyrun.env.windows()
    std = Q.std_raw_of(toyrun.env.st, 1.0)
    d = {lam: S.distance_D(plan_of(toyrun.wrapper(lam))["plan_raw"], w["A"], std)[0] for lam in (-1.0, 0.0, 3.0)}
    print("D by lam (toy, knots 1-10 of 12)", d)
    assert d[-1.0] > d[0.0] > d[3.0]


# ------------------------------------------------------------------------------------------ the selection procedure (section 3), pure logic
def fake(table):
    """table {lam: (C, D)} -> evaluate(lam)."""
    return lambda lam: dict(C=table[lam][0], D=table[lam][1], D_yaw=0.0)


BASE = (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def grid(c_by_lam, d=0.5):
    t = {l: (c_by_lam.get(l, 9.0), d) for l in (*BASE, -1.0, 300.0, 1000.0, 3000.0, 0.1, 0.03, 0.01)}
    return t


def test_select_interior_optimum_runs_no_extension():
    r = S.select(fake(grid({0.0: 1.3, 0.3: 1.2, 1.0: 1.1, 3.0: 1.0, 10.0: 1.05, 30.0: 1.1, 100.0: 1.2, -1.0: 1.5})))
    assert r["lam_star"] == 3.0 and r["label"] == "SELECTED" and sorted(r["runs"]) == sorted((-1.0, *BASE)) and r["sign_check"]["ok"]


def test_select_extends_up_while_the_largest_is_chosen_and_stops_at_first_worse():
    c = {0.0: 1.3, -1.0: 1.5, 0.3: 1.2, 1.0: 1.1, 3.0: 1.0, 10.0: 0.9, 30.0: 0.8, 100.0: 0.7, 300.0: 0.6, 1000.0: 0.65, 3000.0: 0.1}
    r = S.select(fake(grid(c)))
    assert r["lam_star"] == 300.0 and 1000.0 in r["runs"] and 3000.0 not in r["runs"], sorted(r["runs"])
    assert any("extension up stops" in t.get("note", "") for t in r["trace"])


def test_select_extension_up_is_at_most_three_extra_values():
    c = {0.0: 2.0, -1.0: 3.0, 0.3: 1.9, 1.0: 1.8, 3.0: 1.7, 10.0: 1.6, 30.0: 1.5, 100.0: 1.4, 300.0: 1.3, 1000.0: 1.2, 3000.0: 1.1}
    r = S.select(fake(grid(c)))
    assert r["lam_star"] == 3000.0 and [t["lam"] for t in r["trace"] if "extension up" in t.get("why", "")] == [300.0, 1000.0, 3000.0]


def test_select_extension_up_stops_when_the_plan_collapses():
    c = {0.0: 2.0, -1.0: 3.0, 0.3: 1.9, 1.0: 1.8, 3.0: 1.7, 10.0: 1.6, 30.0: 1.5, 100.0: 1.4, 300.0: 1.3, 1000.0: 1.2, 3000.0: 1.1}
    r = S.select(fake({**grid(c), 1000.0: (1.2, 0.05)}))
    assert r["lam_star"] == 300.0 and 3000.0 not in r["runs"], "a collapsed value (D < 0.10) ends the extension, and it is not admissible"


def test_select_extends_down_from_0p3_and_stops_at_first_worse():
    c = {0.0: 2.0, -1.0: 3.0, 0.3: 1.0, 1.0: 1.1, 3.0: 1.2, 10.0: 1.3, 30.0: 1.4, 100.0: 1.5, 0.1: 0.9, 0.03: 0.95, 0.01: 0.1}
    r = S.select(fake(grid(c)))
    assert r["lam_star"] == 0.1 and 0.03 in r["runs"] and 0.01 not in r["runs"], sorted(r["runs"])


def test_select_tie_goes_to_the_smaller_lambda_after_rounding_to_4_decimals():
    c = {0.0: 2.0, -1.0: 3.0, 0.3: 1.5, 1.0: 1.00004, 3.0: 1.00001, 10.0: 1.00003, 30.0: 1.5, 100.0: 1.6}
    r = S.select(fake(grid(c)))
    assert r["lam_star"] == 1.0, "1.00004, 1.00001 and 1.00003 all round to 1.0000: the smaller lambda wins"


def test_select_inadmissible_low_c_is_never_chosen_and_zero_and_negative_are_never_candidates():
    c = {0.0: 0.01, -1.0: 0.01, 0.3: 1.5, 1.0: 1.2, 3.0: 0.5, 10.0: 1.3, 30.0: 1.4, 100.0: 1.6}
    r = S.select(fake({**grid(c), 3.0: (0.5, 0.05)}))
    assert r["lam_star"] == 1.0, "lambda 3 has the lowest C among lambda > 0 but D < 0.10 (collapsed)"
    assert r["sign_check"]["ok"] is False, "C(-1) = C(0) is not > C(0)"


def test_select_no_admissible_runs_the_downward_extension_then_labels_collapse():
    allc = grid({l: 1.0 for l in BASE}, d=0.01)
    r = S.select(fake(allc))
    assert r["lam_star"] is None and r["label"] == "LAMBDA-COLLAPSE" and [t["lam"] for t in r["trace"] if "extension down" in t.get("why", "")] == [0.1, 0.03, 0.01]
    ok = {**allc, 0.03: (0.8, 0.2)}
    r2 = S.select(fake(ok))
    assert r2["lam_star"] == 0.03 and 0.01 not in r2["runs"] and r2["label"] == "SELECTED"


def test_select_mutations_each_change_the_answer(monkeypatch):
    base = {0.0: 0.01, -1.0: 0.01, 0.3: 1.5, 1.0: 1.2, 3.0: 0.5, 10.0: 1.3, 30.0: 1.4, 100.0: 1.6}
    tab = {**grid(base), 3.0: (0.5, 0.05)}
    assert S.select(fake(tab))["lam_star"] == 1.0
    monkeypatch.setattr(S, "admissible", lambda lam, rec: lam > 0)                       # mutation: admissibility dropped
    assert S.select(fake(tab))["lam_star"] == 3.0
    monkeypatch.undo()
    tie = {0.0: 2.0, -1.0: 3.0, 0.3: 1.5, 1.0: 1.00004, 3.0: 1.00001, 10.0: 1.5, 30.0: 1.5, 100.0: 1.6}
    assert S.select(fake(grid(tie)))["lam_star"] == 1.0
    monkeypatch.setattr(S, "c4", lambda x: float(x))                                    # mutation: no rounding, the raw minimum wins
    assert S.select(fake(grid(tie)))["lam_star"] == 3.0
    monkeypatch.undo()
    up = {0.0: 2.0, -1.0: 3.0, 0.3: 1.9, 1.0: 1.8, 3.0: 1.7, 10.0: 1.6, 30.0: 1.5, 100.0: 1.4, 300.0: 1.3, 1000.0: 1.2, 3000.0: 1.1}
    assert S.select(fake(grid(up)))["lam_star"] == 3000.0
    monkeypatch.setattr(S, "EXT_UP", ())                                                # mutation: no extension up
    assert S.select(fake(grid(up)))["lam_star"] == 100.0


def test_select_never_picks_a_non_positive_lambda_even_if_the_sign_run_has_the_lowest_c():
    c = {0.0: 1.0, -1.0: 0.001, 0.3: 1.5, 1.0: 1.2, 3.0: 1.3, 10.0: 1.4, 30.0: 1.4, 100.0: 1.6}
    assert S.select(fake(grid(c)))["lam_star"] == 1.0


# ------------------------------------------------------------------------------------------ the criterion pieces
def test_criterion_c_is_band_metrics_acc_mae_on_30_knot_data():
    rng = np.random.default_rng(3)
    plan, A = rng.normal(size=(40, 30, 2)), rng.normal(size=(40, 31, 2))
    ref = E.band_metrics(plan, A, np.full(40, 8.0), None, None)[0]["0-2s"]["acc_mae"].mean()
    assert S.criterion_C(plan, A) == float(ref)
    assert S.criterion_C(plan + 0.01, A) != S.criterion_C(plan, A)


def test_distance_d_in_prior_std_and_hold_is_zero():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(20, 31, 2)); hold = H.hold_plan(A[:, 0], 30)
    std = np.array([2.0, 0.5])
    assert S.distance_D(hold, A, std) == (0.0, 0.0)
    plan = hold.copy(); plan[:, :10, 0] += 0.4; plan[:, :10, 1] -= 0.1; plan[:, 10:, :] += 9.0                    # knots 11-30 are not read
    d = S.distance_D(plan, A, std)
    assert abs(d[0] - 0.2) < 1e-12 and abs(d[1] - 0.2) < 1e-12


def test_dir_acc_signs_and_na_for_hold():
    rng = np.random.default_rng(5)
    n = 60; A = rng.normal(size=(n, 31, 2)); eid = np.array([f"e{i % 6}" for i in range(n)])
    hold = H.hold_plan(A[:, 0], 30)
    assert np.isnan(S.dir_acc(hold, A, eid, 20)["r"]) and "reason" in S.dir_acc(hold, A, eid, 20)
    follow = hold.copy(); follow[:, :10, 0] = A[:, :10, 0]
    assert S.dir_acc(follow, A, eid, 50)["r"] > 0.99
    away = hold.copy(); away[:, :10, 0] = 2 * A[:, 0:1, 0] - A[:, :10, 0]
    assert S.dir_acc(away, A, eid, 50)["r"] < -0.99
    assert S.dir_acc(follow, A, eid, 50) == S.dir_acc(follow, A, eid, 50), "bootstrap seed 0 is deterministic"


# ------------------------------------------------------------------------------------------ C-TRAIN-ISO and the selection windows
def synth_latents(path, n_clips=6, n_frames=70, d=32, tag="toy", seed=0, bad_ok=True):
    """A TRAIN latents file in the layout of latents_train_R35_s*.pt: overlapping 16-frame windows (start every 4 frames), frame content a function of (clip, frame) only."""
    g = torch.Generator().manual_seed(seed)
    eids = [f"clip{i}" for i in range(n_clips)]
    fz = {e: torch.randn(n_frames, d, generator=g) for e in eids}
    fa = {e: torch.randn(n_frames, 2, generator=g) for e in eids}
    fo = {e: torch.ones(n_frames, dtype=torch.bool) for e in eids}
    if bad_ok:
        fo[eids[0]][20] = False                                                           # a missing frame splits that clip's valid positions
    Z, A, OK, T, EID = [], [], [], [], []
    for e in eids:
        for t in range(3, n_frames - 15, 4):
            Z.append(fz[e][t:t + 16]); A.append(fa[e][t:t + 16]); OK.append(fo[e][t:t + 16]); T.append(t); EID.append(e)
    torch.save(dict(Z1={tag: torch.stack(Z)}, A_raw=torch.stack(A), ok=torch.stack(OK), t=torch.tensor(T), eid=EID), path)
    return eids


def test_selection_windows_per_clip_positions_frames_and_source_equality(tmp_path):
    p = tmp_path / "lat.pt"; eids = synth_latents(p, n_frames=75)
    w = S.build_selection_windows(str(p), "toy", n_frames=13)
    L = torch.load(p, weights_only=False)
    assert w["z"].shape[1:] == (13, 32) and w["A"].shape[1:] == (13, 2) and w["meta"]["max_abs_dz_repeated"] == 0.0 and w["meta"]["max_abs_da_repeated"] == 0.0
    for e in eids:
        got = sorted(int(g) for g in w["now"][w["eid"] == e])
        assert 1 <= len(got) <= 4
    assert S.windows_equal_source(w, L, "toy") == (0.0, 0.0)
    assert (w["v0"] == 0).all() and w["meta"]["n"] == len(w["eid"])
    ok_frames = {(e, f) for e in eids for f in range(75)}                                  # every selected frame is ok
    tab, _, _ = S.frame_table(L, "toy")
    assert all(tab[(str(e), int(g) + i)][2] for e, g in zip(w["eid"], w["now"]) for i in range(13))
    valid = S.valid_positions(tab, ["clip0"], 13)["clip0"]
    assert all(not (g <= 20 < g + 13) for g in valid), "windows spanning the not-ok frame are not valid positions"


def test_pick_positions_is_unique_round_linspace():
    v = list(range(100, 134))
    assert S.pick_positions(v) == [v[i] for i in np.unique(np.round(np.linspace(0, 33, 4)).astype(int))]
    assert S.pick_positions(v[:2]) == v[:2] and S.pick_positions([]) == []


def test_c_train_iso_mutation_a_perturbed_window_is_not_source_equal(tmp_path):
    p = tmp_path / "lat.pt"; synth_latents(p, n_frames=75)
    w = S.build_selection_windows(str(p), "toy", n_frames=13); L = torch.load(p, weights_only=False)
    w["z"][2, 5, 7] += 1e-3
    assert S.windows_equal_source(w, L, "toy")[0] > 0


def test_c_train_iso_disjointness_assertion_and_its_mutation():
    assert S.assert_disjoint(["a", "b"], ["c", "d"]) == 0
    with pytest.raises(AssertionError, match="1 TRAIN clips are eval clips"):
        S.assert_disjoint(["a", "b"], ["b", "d"])


def _selection_cfg(tmp_path, latents):
    (tmp_path / "ckpt.pt").write_bytes(b"toy-checkpoint")
    return dict(tag="toy", ckpt=str(tmp_path / "ckpt.pt"), latents=str(latents), latents_sha=None)


def test_selection_end_to_end_on_the_toy_never_reads_an_eval_window_and_lam0_equals_run_hwm_plan(env, monkeypatch, tmp_path):
    """C-TRAIN-ISO: the whole selection runs with hwm_data.eval_windows_from_cache and wmact_lib.open_windows patched to raise; its lambda = 0 plans equal `run_hwm plan`'s on the same windows bit for bit;
    the record holds the grid as run, C, D, D_yaw, the rule trace, lambda*, every argv and the input and code sha256s."""
    monkeypatch.setattr(R, "prepare", env.prepare)
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("an eval window loader was called during selection"))
    monkeypatch.setattr(D, "eval_windows_from_cache", boom)
    import wmact_lib
    monkeypatch.setattr(wmact_lib, "open_windows", boom)
    lat = tmp_path / "lat.pt"; synth_latents(lat, n_clips=6, n_frames=75)
    evl = tmp_path / "evl.pt"; torch.save(dict(eid=["eval_a", "eval_b"]), evl)
    cfg = _selection_cfg(tmp_path, lat)
    monkeypatch.setattr(S, "plan_argv", lambda cfg_, out, lam, workers: ["--tag", cfg_["tag"], "--ckpt", cfg_["ckpt"], "--train-latents", cfg_["latents"], "--train-tag", cfg_["tag"], "--seed-mode", "hold", f"--lam={lam!r}", *FAST,
                                                                      "--arms", "hwm_oracle", "--seeds", "0", "--workers", str(workers), "--out", out])
    root = str(tmp_path / "sel")
    rec = S.run_selection("s0", root, workers=1, n_boot=20, cfg=cfg, eval_latents=str(evl), enforce_threads=False, n_frames=13)
    f = os.path.join(root, "LAMBDA_SELECTION_R35_s0.json")
    assert os.path.exists(f)
    j = json.load(open(f, encoding="utf-8"))
    for k in ("lam_star", "label", "grid_as_run", "sign_check", "trace", "runs", "windows", "inputs_sha256", "code_of_record_sha256", "rule", "prior_std_raw", "beside_no_bar", "train_clips_in_eval"):
        assert k in j, k
    assert j["train_clips_in_eval"] == 0 and j["label"] in ("SELECTED", "LAMBDA-COLLAPSE")
    for r in j["runs"].values():
        for k in ("C", "D", "D_yaw", "argv", "dir_acc", "plan_sha256"):
            assert k in r
    assert set(map(float, j["runs"])) >= {-1.0, 0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0}
    assert j["code_of_record_sha256"]["run_hwm.py"] and j["inputs_sha256"]["ckpt"] == C.sha256_file(cfg["ckpt"])
    sw = os.path.join(root, "selection_windows.npz")
    assert j["windows"]["sha256"] == C.sha256_file(sw)
    a = W.parse(S.plan_argv(cfg, str(tmp_path / "direct"), 0.0, 1))
    os.makedirs(a.out); shutil.copyfile(sw, os.path.join(a.out, "windows.npz"))
    R.cmd_plan(a)
    p_sel, p_dir = plan_of(S.lam_dir(root, 0.0)), plan_of(a.out)
    assert np.array_equal(p_sel["plan_raw"], p_dir["plan_raw"]) and np.array_equal(p_sel["cost1"], p_dir["cost1"])
    for lam in (0.0, 1.0, -1.0):
        assert W.read_planned_lam(os.path.join(S.lam_dir(root, lam), "plans_hwm_oracle_s0.npz")) == lam
    with pytest.raises(AssertionError, match="eval window loader"):                          # the mutation: a selection that did read an eval window goes red
        D.eval_windows_from_cache(None, None, None, None)
    with pytest.raises(W.RefusedRun, match="not empty"):
        S.run_selection("s0", root, cfg=cfg, eval_latents=str(evl), enforce_threads=False, n_frames=13)


def test_selection_refuses_overlapping_eval_clip_and_a_changed_latents_file(env, monkeypatch, tmp_path):
    monkeypatch.setattr(R, "prepare", env.prepare)
    lat = tmp_path / "lat.pt"; synth_latents(lat, n_clips=3, n_frames=75)
    evl = tmp_path / "evl.pt"; torch.save(dict(eid=["clip1"]), evl)
    with pytest.raises(AssertionError, match="TRAIN clips are eval clips"):
        S.run_selection("s0", str(tmp_path / "r1"), cfg=_selection_cfg(tmp_path, lat), eval_latents=str(evl), enforce_threads=False, n_frames=13)
    cfg = _selection_cfg(tmp_path, lat); cfg["latents_sha"] = "0" * 64
    with pytest.raises(W.RefusedRun, match="phase-1 record"):
        S.run_selection("s0", str(tmp_path / "r2"), cfg=cfg, eval_latents=str(evl), enforce_threads=False, n_frames=13)


# ------------------------------------------------------------------------------------------ C-STRAT (real phase-1 windows, one pair of banked plan files; skipped when the inputs are absent)
HAVE_REAL = os.path.exists(os.path.join(REAL["windows"], "plans_hwm_oracle_s0.npz")) and os.path.exists(CS.RECORD_SCRIPT) and os.path.exists(os.path.join(HERE, "reader_src", "hwm_eval.py"))


@pytest.fixture(scope="module")
def cstrat(tmp_path_factory):
    d = tmp_path_factory.mktemp("cstrat"); w = REAL["windows"]
    assert C.sha256_file(os.path.join(w, "windows.npz")) == REAL["windows_sha"], "the windows are not the phase-1 R35_s0 SCORE windows"
    kw = dict(rep_a=f"{w}/plans_flat_oracle_s1.npz", rep_b=f"{w}/plans_hwm_oracle_s1.npz")
    a, b = f"{w}/plans_flat_oracle_s0.npz", f"{w}/plans_hwm_oracle_s0.npz"
    res = {}
    for name, script, extra in (("record", CS.RECORD_SCRIPT, {}), ("strat", CS.STRAT_SCRIPT, {}), ("regression", CS.STRAT_SCRIPT, dict(eval_src=os.path.join(C.CODE_ROOT, "hwm_plan")))):
        out = str(d / f"{name}.json")
        rc, so, se = CS.run(script, w, a, b, out, py=sys.executable, **kw, **extra)
        assert rc == 0, (name, so, se)
        res[name] = json.load(open(out, encoding="utf-8"))
    return res


@pytest.mark.skipif(not HAVE_REAL, reason="phase-1 R35_s0 windows / plan files, the record script or reader_src absent")
def test_c_strat_record_script_is_the_banked_blob():
    import subprocess
    r = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "hash-object", "--no-filters", CS.RECORD_SCRIPT], capture_output=True, text=True)
    assert r.stdout.strip() == CS.RECORD_SCRIPT_GIT_BLOB


@pytest.mark.skipif(not HAVE_REAL, reason="phase-1 R35_s0 windows / plan files, the record script or reader_src absent")
def test_c_strat_non_strategic_cells_bit_equal_and_strategic_computed(cstrat):
    r = CS.compare(cstrat["record"], cstrat["strat"])
    print("C-STRAT", json.dumps({k: v for k, v in r.items() if k != "strategic_cells"}), r["strategic_cells"])
    assert r["rows_equal"] and r["replicate_equal"] and r["tactical_equal"] and r["counts_equal"], r
    assert r["n_rows"] == 117 and r["n_replicate"] == 72 and r["n_windows"] == 994 and r["n_episodes"] == 71
    assert not r["paused"] and r["strategic_is_block"] and r["strategic_cells_have_n_or_reason"] and r["pass"]
    assert set(r["strategic_cells"]) == {"s_adh", "s_adh_deg", "s_false"}
    assert r["strategic_cells"]["s_false"]["n_windows"] > 0
    assert cstrat["strat"]["strategic"]["state"].startswith("REPORTED IN FULL")


@pytest.mark.skipif(not HAVE_REAL, reason="phase-1 R35_s0 windows / plan files, the record script or reader_src absent")
def test_c_strat_regression_arm_code_of_record_import_reads_paused(cstrat):
    r = CS.compare(cstrat["record"], cstrat["regression"])
    print("C-STRAT regression", {k: v for k, v in r.items() if k in ("paused", "strategic_is_block", "pass", "non_strategic_equal")})
    assert r["paused"] and not r["strategic_is_block"] and not r["pass"], "the code-of-record import must read PAUSED and fail C-STRAT"
    assert cstrat["regression"]["strategic_computed"] is False and "PAUSED" in cstrat["regression"]["strategic"]
    assert not any(x["family"] == "STRATEGIC" for x in cstrat["regression"]["rows"])


def test_c_strat_compare_flags_a_changed_non_strategic_cell_and_missing_strategic_n():
    rows = [dict(family="LATERAL", band="0-2s", metric="yaw_mae", delta=0.1)]
    srow = dict(kind="contrast", family="STRATEGIC", band="full6s", metric="s_false", a="A", b="B", n_windows=5, n_episodes=2, label="none")
    rec = dict(rows=rows, replicate=[], tactical_kappa={}, n_windows=3, n_episodes=1, strategic="PAUSED")
    ok = dict(rows=rows + [srow], replicate=[], tactical_kappa={}, n_windows=3, n_episodes=1, strategic={"state": "REPORTED IN FULL"})
    assert CS.compare(rec, ok)["pass"]
    bad = json.loads(json.dumps(ok)); bad["rows"][0]["delta"] = 0.1000000001
    assert not CS.compare(rec, bad)["pass"] and not CS.compare(rec, bad)["rows_equal"]
    nan = json.loads(json.dumps(ok)); nan["rows"][1].update(n_windows=0)
    assert not CS.compare(rec, nan)["pass"], "a STRATEGIC cell with n = 0 and no reason is neither computed nor a reasoned n/a"
    nan["rows"][1]["reason"] = "n = 0: no window finite in both rows"
    assert CS.compare(rec, nan)["pass"]
    assert not CS.compare(rec, dict(ok, strategic="PAUSED (PI item 25)"))["pass"]
