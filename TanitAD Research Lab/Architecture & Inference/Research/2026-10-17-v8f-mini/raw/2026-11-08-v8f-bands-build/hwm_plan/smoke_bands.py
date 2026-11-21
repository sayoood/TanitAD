"""CPU smoke of the band levers on the frozen first ladder world model (R35_s0.pt), 16 windows. It proves the paths RUN on a real (3,5) model: shapes, finite flags, bit-equality of the defaults, seconds. It prints NO metric: no error, no cost,
no loss value. The priors are fitted on the smoke windows themselves because no TRAIN latents exist for a ladder checkpoint (smoke only, never a result). Usage: smoke_bands.py <ckpt> <n> (env_hwm.cmd for the environment)."""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import hwm_plan as H
import hwm_data as D
import goal_head as G
import plantest_lib as P
import tanitad

CK = sys.argv[1] if len(sys.argv) > 1 else "F:/TanitAD-scratch/hwm_bands/ckpt/R35_s0.pt"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
torch.set_num_threads(2)
assert tanitad.__file__.replace("\\", "/").startswith("F:/TanitAD-scratch/ladder_tree/"), tanitad.__file__
m, cfg, ck = D.load_model(CK)
wm, st = m.wm, m.wm.stack
g2 = H.Geometry.from_stack(st, wm)
g4 = H.Geometry.from_stack(st, wm, goal_steps=4)
gl = H.Geometry.from_stack(st, wm, stitch="l1_full")
print("geometry T3=2:", g2.T3, g2.goal_frame, "K", g2.K1, g2.K2, "| T3=4:", g4.T3, g4.goal_frame, g4.goal_frame * H.DT, "s | l1_full: K1", gl.K1, "h1", gl.h1, flush=True)

eids, nows = D.plantest_pairs()
sel = np.unique(np.linspace(0, len(eids) - 1, N * 3).astype(int))
cnt = D.goal_frame_counts(eids[sel], nows[sel], [g2.goal_frame, g4.goal_frame])
print("windows with the true-future frame, of", cnt["n_pairs"], "pairs:", cnt["counts"], flush=True)
ws = D.make_windowset(m, eids[sel], nows[sel], 31)
keep = np.arange(min(N, ws.n))
ws.z, ws.A, ws.eid, ws.now, ws.v0 = ws.z[:len(keep)], ws.A[:len(keep)], ws.eid[:len(keep)], ws.now[:len(keep)], ws.v0[:len(keep)]
n = len(keep)
print("plan windows (31 frames):", n, "of", len(sel), flush=True)
D.fit_priors(st, wm, ws.z, torch.from_numpy(ws.A).float())                    # SMOKE ONLY
ro = H.fit_kin_readout(st, wm, ws.z, torch.from_numpy(ws.A).float())
print("kin readout fitted on the smoke windows, finite:", bool(np.isfinite(ro.r2).all()), flush=True)

SOLV = dict(S=4, n_iters=6, decode_iters=40)
Zn, A0 = ws.z[:, 0], ws.A[:, 0]
stub = H.PredictedGoal(H.stub_goal_fn, g2.goal_frame, Zn)
tab = []


def run(name, mode="hier", geom=g2, solver=None, goal=None, **kw):
    t = time.time()
    pl = H.HWMPlanner(wm, geom, solver or H.SolverCfg(**SOLV), **kw)
    r = pl.plan_windows(mode, goal or stub, Zn, A0, idx=np.arange(n))
    ok = bool(np.isfinite(r["plan_raw"]).all()) and r["plan_raw"].shape == (n, 30, 2)
    tab.append((name, r["plan_raw"].shape, ok, round(time.time() - t, 1)))
    print(f"{name}: shape {r['plan_raw'].shape} finite {ok} {time.time() - t:.1f}s", flush=True)
    return r


base = run("default (T3=2, ctx 1, decode, stub goal)")
again = run("plan_ctx=1 explicit", solver=H.SolverCfg(**SOLV, plan_ctx=1))
print("default is bit-identical to plan_ctx=1:", bool(np.array_equal(base["plan_raw"], again["plan_raw"])), flush=True)
c3 = run("plan_ctx=3", solver=H.SolverCfg(**SOLV, plan_ctx=3))
print("plan_ctx=3 reaches the plan:", bool(not np.array_equal(base["plan_raw"], c3["plan_raw"])), flush=True)
run("l1_full", geom=gl)
run("T3=4, stub goal", geom=g4, goal=H.PredictedGoal(H.stub_goal_fn, g4.goal_frame, Zn))
voc = run("vocabulary seeds (all 9 classes)", vocab=H.Vocabulary.default(g2.n_hold))
print("vocabulary seeds reach the plan:", bool(not np.array_equal(base["plan_raw"], voc["plan_raw"])), flush=True)
v0 = np.asarray(ws.v0, np.float64)
cons0 = H.Constraints(v0, v0 + 5.0, np.full(n, 3.0), np.full(n, 4.0), 0.0, ro)
c0 = run("constraints, weight 0", constraints=cons0)
print("constraints weight 0 is bit-identical to none:", bool(np.array_equal(base["plan_raw"], c0["plan_raw"])), flush=True)
c1 = run("constraints, weight 1", constraints=H.Constraints(v0, v0 + 5.0, np.full(n, 3.0), np.full(n, 4.0), 1.0, ro))
print("constraints weight 1 reaches the plan:", bool(not np.array_equal(base["plan_raw"], c1["plan_raw"])), flush=True)
run("flat with constraints", mode="flat", constraints=H.Constraints(v0, v0 + 5.0, np.full(n, 3.0), np.full(n, 4.0), 1.0, ro))

# the goal head: build data from the eval cache, selfcheck, 3 steps of head-only training, wire as the PredictedGoal
import wmact_lib as W
store = W.open_windows(16)[0].store
sel2 = np.unique(np.linspace(0, len(eids) - 1, 96).astype(int))
gd = G.build_goal_data(m, store, eids[sel2], nows[sel2], g2.c3, 6)
print("goal-head data rows:", len(gd["eid"]), "valid steps per level-3 step:", gd["valid"].sum(0).tolist(), flush=True)
gd["stok"] = np.random.default_rng(0).normal(size=(len(gd["eid"]), 8)).astype(np.float32)                    # synthetic token: the real dump comes from stream C
gd["nav"] = np.zeros((len(gd["eid"]), 5), np.float32)
nb = G.NpzBatches(gd, None, 32, 0)
head = G.GoalHead(int(st.cfg[0].latent_dim), int(st.cfg[2].latent_dim), 6, ctx_dim=8, hidden=64)
fp0 = G.wm_fingerprint(st)
res = G.train(head, st, nb.get, 3, lr=1e-3, log=lambda s: None)
print("goal head: selfcheck red =", res["selfcheck"]["red"], "| world model bit-equal after the run:", res["wm_fingerprint_after"] == fp0 == G.wm_fingerprint(st), flush=True)
fn = G.make_goal_fn(head, st, 2)
gp = H.PredictedGoal(fn, g2.goal_frame, Zn, nav=torch.zeros(n, 5), ego=torch.from_numpy(ws.A[:, :1]).float(), ctx=torch.randn(n, 8), level=3)
gh_r = run("goal head as PredictedGoal (level 3)", goal=gp)
print("trained-goal plan differs from the stub plan:", bool(not np.array_equal(base["plan_raw"], gh_r["plan_raw"])), flush=True)
print("SMOKE DONE", len(tab), "runs, all finite:", all(t[2] for t in tab), flush=True)
