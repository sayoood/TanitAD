"""HWM-PLAN runner. Three stages, each resumable and CPU only:
  windows  build the WindowSet of a checkpoint (its own adapter + proj on the local eval cache) for PLANTEST's window list, save windows.npz
  plan     plan the arms x inference seeds in units of batches over N workers (OMP threads per worker from the environment), save plans_<arm>_s<seed>.npz
  report   metrics per band and family, paired episode-cluster bootstrap, inference-seed replicate -> RESULT_HWMPLAN_<tag>.json / .md
Example (M2b smoke):  python run_hwm.py windows --tag M2b_s0 --ckpt <ckpt> --train-latents <pt> --out <dir> --n 16
Windows lacking F frames inside the clip are dropped and counted (same set for every arm). The goal of the *_plan arms is the stub (G-STAY) until a goal head is supplied with --goal-module mod:fn."""
import argparse, importlib, json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from types import SimpleNamespace
import numpy as np
import torch
import hwm_plan as H
import hwm_data as D
import hwm_eval as E
import plantest_lib as P

_CTX = {}


def solver_from(a):
    pc = tuple(a.plan_ctx) if len(a.plan_ctx) == 3 else int(a.plan_ctx[0])
    return H.SolverCfg(seed_mode=a.seed_mode, lam=a.lam, S=a.S, n_iters=a.n_iters, decode_lam=a.decode_lam, decode_iters=a.decode_iters, plan_ctx=pc)


def add_common(p):
    p.add_argument("--tag", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True)
    p.add_argument("--train-latents", default=D.TRAIN16, help="pt with Z1[tag], A_raw, ok for the prior fit (TRAIN windows only); a ladder checkpoint needs its own")
    p.add_argument("--train-tag", default=None)
    p.add_argument("--seed-mode", default="hold", choices=["hold", "prior"]); p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--S", type=int, default=P.S_CAND); p.add_argument("--n-iters", type=int, default=P.N_ITERS)
    p.add_argument("--decode-lam", type=float, default=0.0); p.add_argument("--decode-iters", type=int, default=300)
    p.add_argument("--goal-steps", type=int, default=None, help="T3, level-3 steps to the goal (G2): 4 on the ladder is 12 s (arm of record), 2 is 6 s (control); default K2")
    p.add_argument("--goal-module", default=None, help="module:function -> fn(GoalInputs) -> (B, D) goal latent (a level-1 goal; set fn.level = 3 for a level-3 goal)")
    p.add_argument("--goal-head", default=None, help="goal_head.py checkpoint (head-only trained): the predicted level-3 goal at step T3; takes precedence over --goal-module")
    p.add_argument("--goal-aux", default=None, help="npz with eid, now and optional nav (n, 5), stok (n, C), vocab_scores (n, 9): joined to the windows by (eid, now); the harness windows carry no nav and no strategic token themselves")
    p.add_argument("--plan-ctx", type=int, nargs="+", default=[1], help="G1: rollout context of the planner, one value (all levels) or three; 1 = H-JEPA's planning context (today), 3 = the context the ladder trained with")
    p.add_argument("--stitch", default="decode", choices=["decode", "l1_full"], help="G3: decode (L1 knots 1-10, L2 decoded 11-30) or l1_full (level 1 plans all 30 knots against all level-2 sub-goals)")
    p.add_argument("--vocab", default="off", choices=["off", "all"], help="G6: level-2 candidates seeded from the tactical vocabulary templates (all classes unless --goal-aux holds vocab_scores)")
    p.add_argument("--constr-weight", type=float, default=0.0, help="G7: weight of T's constraints in the level-2 and level-1 cost; 0.0 = today, bit for bit")
    p.add_argument("--constr-v-margin", type=float, default=5.0, help="speed cap = v0 + margin, m/s (placeholder, ESTIMATED; the Master Mind sets it at freeze)")
    p.add_argument("--constr-alat", type=float, default=3.0); p.add_argument("--constr-brake", type=float, default=4.0)
    p.add_argument("--no-goal-frame", action="store_true", help="windows need only n_traj + 1 frames (the predicted-goal arms); the oracle arms are refused")
    p.add_argument("--smoke-prior-from-windows", action="store_true", help="SMOKE ONLY: fit the priors on the windows themselves (no TRAIN latents exist for a ladder checkpoint yet); never for a result")
    p.add_argument("--t3-counts", type=int, nargs="*", default=[2, 4], help="windows: report the windows that have the true-future frame at these T3 values")


def prepare(a, ws=None):
    """Model, priors, geometry. The priors are fitted on TRAIN latents (--train-latents); --smoke-prior-from-windows (smoke only) fits them on the window set ws instead. Z, A are kept in a.ZA for the constraint readout."""
    m, cfg, ck = D.load_model(a.ckpt)
    wm, st = m.wm, m.wm.stack
    if a.smoke_prior_from_windows:
        assert ws is not None, "the smoke prior needs the windows"
        Z, A = ws.z, torch.from_numpy(np.asarray(ws.A)).float()
    else:
        Z, A = D.trainset_from_latents_pt(a.train_latents, a.train_tag or a.tag)
    D.fit_priors(st, wm, Z, A)
    a.ZA = (Z, A)
    geom = H.Geometry.from_stack(st, wm, goal_steps=a.goal_steps, stitch=a.stitch)
    return m, wm, st, geom


def aux_for(a, ws):
    """The optional --goal-aux arrays joined to the windows by (eid, now); a missing key is None, a window without a row is refused (no silent zero)."""
    if not a.goal_aux:
        return {}
    d = np.load(a.goal_aux, allow_pickle=False)
    key = {(str(e), int(t)): i for i, (e, t) in enumerate(zip(d["eid"], d["now"]))}
    row = np.array([key.get((str(e), int(t)), -1) for e, t in zip(ws.eid, ws.now)])
    assert (row >= 0).all(), f"{int((row < 0).sum())} windows have no row in {a.goal_aux}"
    return {k: d[k][row] for k in ("nav", "stok", "vocab_scores") if k in d.files}


def make_planner(a, wm, st, geom, ws):
    """The planner with the G6 / G7 extras the flags ask for; with the defaults it is the plain planner."""
    aux = aux_for(a, ws)
    vocab = H.Vocabulary.default(geom.n_hold) if a.vocab == "all" else None
    cons = None
    if a.constr_weight != 0.0:
        v0 = np.asarray(ws.v0, np.float64)
        cons = H.Constraints(v0, v0 + a.constr_v_margin, np.full_like(v0, a.constr_alat), np.full_like(v0, a.constr_brake), float(a.constr_weight), H.fit_kin_readout(st, wm, *a.ZA))
    return H.HWMPlanner(wm, geom, solver_from(a), vocab=vocab, vocab_scores=aux.get("vocab_scores"), constraints=cons)


def cmd_windows(a):
    os.makedirs(a.out, exist_ok=True)
    m, wm, st, geom = prepare_geometry(a)
    F = (geom.n_traj if a.no_goal_frame else max(geom.goal_frame, geom.n_traj)) + 1
    eids, nows = D.plantest_pairs()
    sel = np.arange(len(eids)) if not a.n else np.unique(np.linspace(0, len(eids) - 1, a.n * 2).astype(int))
    c3 = geom.c3
    t3c = D.goal_frame_counts(eids[sel], nows[sel], [t * c3 for t in a.t3_counts])
    print("goal-frame windows (T3 -> frame -> windows with the true-future frame, of", t3c["n_pairs"], "pairs):", {t: t3c["counts"][t * c3] for t in a.t3_counts}, flush=True)
    ws = D.make_windowset(m, eids[sel], nows[sel], F)
    if a.n and ws.n > a.n:
        keep = np.arange(a.n)
        ws.z, ws.A, ws.eid, ws.now, ws.v0 = ws.z[:a.n], ws.A[:a.n], ws.eid[:a.n], ws.now[:a.n], ws.v0[:a.n]
        from tanitad.models.hjepa import donor_indices
        codes = {x: i for i, x in enumerate(sorted(set(ws.eid)))}
        ws.donor = donor_indices(a.n, episode_ids=torch.tensor([codes[x] for x in ws.eid]), seed=0).numpy(); ws.n = a.n
    np.savez(f"{a.out}/windows.npz", z=ws.z.numpy(), A=ws.A, eid=ws.eid, now=ws.now, v0=ws.v0, donor=ws.donor,
             meta=json.dumps(dict(tag=a.tag, F=F, no_goal_frame=bool(a.no_goal_frame), t3_goal_frame_windows={str(t): t3c["counts"][t * c3] for t in a.t3_counts}, t3_pairs=t3c["n_pairs"], n=ws.n, n_input=ws.n_input, n_dropped=ws.n_dropped, n_dropped_length=ws.n_dropped_length, n_dropped_notok=ws.n_dropped_notok, geometry=vars(geom) if hasattr(geom, "__dict__") else str(geom))))
    print("windows", ws.n, "of", ws.n_input, "dropped", ws.n_dropped, "F", F, "geometry", geom, flush=True)


def prepare_geometry(a):
    """Model and geometry only (the windows stage fits no prior)."""
    m, cfg, ck = D.load_model(a.ckpt)
    wm, st = m.wm, m.wm.stack
    return m, wm, st, H.Geometry.from_stack(st, wm, goal_steps=a.goal_steps, stitch=a.stitch)


def load_ws(out):
    d = np.load(f"{out}/windows.npz", allow_pickle=False)
    return SimpleNamespace(z=torch.from_numpy(d["z"]), A=d["A"], eid=d["eid"], now=d["now"], v0=d["v0"], donor=d["donor"], n=len(d["eid"]), meta=json.loads(str(d["meta"])))


def goal_fn_for(arm, ws, geom, a, st=None):
    kind = E.ARMS[arm][1]
    if kind == "oracle":
        assert not ws.meta.get("no_goal_frame"), "these windows were built without the true-future goal frame: the oracle arms are refused"
        return H.OracleGoal(ws.z[:, geom.goal_frame], geom.goal_frame)
    fn = H.stub_goal_fn
    aux = aux_for(a, ws)
    t = lambda k: None if k not in aux else torch.from_numpy(np.asarray(aux[k])).float()
    ego = torch.from_numpy(np.asarray(ws.A[:, :1])).float()
    if a.goal_head:
        import goal_head as G
        head, _ = G.load_head(a.goal_head)
        fn = G.make_goal_fn(head, st, geom.T3)
    elif a.goal_module:
        mod, f = a.goal_module.split(":")
        fn = getattr(importlib.import_module(mod), f)
    return H.PredictedGoal(fn, geom.goal_frame, ws.z[:, 0], nav=t("nav"), ego=ego, ctx=t("stok"), level=int(getattr(fn, "level", 1)))


def _init(a_dict):
    a = SimpleNamespace(**a_dict)
    ws = load_ws(a.out)
    m, wm, st, geom = prepare(a, ws)
    _CTX.update(a=a, planner=make_planner(a, wm, st, geom, ws), ws=ws, geom=geom, st=st)


def _work(unit):
    arm, seed, b0, b1 = unit
    c = _CTX
    ws, geom, a = c["ws"], c["geom"], c["a"]
    idx = np.arange(b0 * P.BATCH, min(ws.n, b1 * P.BATCH))
    t = time.time()
    r = c["planner"].plan_windows(E.ARMS[arm][0], goal_fn_for(arm, ws, geom, a, c["st"]), ws.z[:, 0], ws.A[:, 0], idx=idx, seed=seed, batch0=b0)
    r["wall_s"] = np.array(time.time() - t)
    return unit, r


def cmd_plan(a):
    ws = load_ws(a.out)
    nb = (ws.n + P.BATCH - 1) // P.BATCH
    todo, merged = [], {}
    for arm in a.arms:
        for seed in a.seeds:
            f = f"{a.out}/plans_{arm}_s{seed}.npz"
            if not os.path.exists(f):
                todo += [(arm, seed, b, b + 1) for b in range(nb)]
    print("units", len(todo), flush=True)
    if not todo:
        return
    res = []
    ad = {k: v for k, v in vars(a).items() if k != "fn"}
    if a.workers <= 1:
        _init(ad)
        for u in todo:
            res.append(_work(u)); print("unit", u, round(float(res[-1][1]["wall_s"]), 1), "s", flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(ad,), maxtasksperchild=8) as pool:
            for u, r in pool.imap_unordered(_work, todo):
                res.append((u, r)); print("unit", u, round(float(r["wall_s"]), 1), "s", flush=True)
    by = {}
    for (arm, seed, b0, _), r in res:
        by.setdefault((arm, seed), []).append((b0, r))
    for (arm, seed), lst in by.items():
        lst.sort(key=lambda x: x[0])
        keys = [k for k in lst[0][1] if k != "wall_s"]
        out = {k: np.concatenate([r[k] for _, r in lst]) for k in keys}
        out["wall_s"] = np.array(sum(float(r["wall_s"]) for _, r in lst))
        np.savez(f"{a.out}/plans_{arm}_s{seed}.npz", **out)


def cmd_report(a):
    ws = load_ws(a.out)
    ws.lead = D.lead_for(ws.eid, ws.now)
    ws.lead6 = D.lead_for(ws.eid, ws.now, D.LEAD_BLOCK6)
    print("lead join: states", {k: int((ws.lead["state"] == k).sum()) for k in ("LEAD", "NO_LEAD", "NOT_STRAIGHT", "NO_LABEL", "NO_ROW")}, "v0 agreement (median, max, n)", D.lead_join_check(ws.lead, ws.v0), flush=True)
    m, wm, st, geom = prepare(a, ws)
    pl = make_planner(a, wm, st, geom, ws)
    plans, seeds_M = {}, {}
    for arm in E.ARMS:
        for seed in a.seeds:
            f = f"{a.out}/plans_{arm}_s{seed}.npz"
            if os.path.exists(f):
                met = E.band_metrics(np.load(f)["plan_raw"], ws.A, ws.v0, ws.lead, ws.lead6)[0]
                seeds_M.setdefault(arm, {})[seed] = met
                if seed == 0:
                    plans[arm] = met
    for k, v in E.run_refs(pl, ws).items():
        plans[k] = E.band_metrics(v, ws.A, ws.v0, ws.lead, ws.lead6)[0]
    rep = E.build_report(plans, ws, n_boot=a.n_boot, seeds_M=seeds_M)
    rep["meta"] = dict(tag=a.tag, geometry=str(geom), solver=str(solver_from(a)), windows=ws.meta, bands=dict(T3=geom.T3, plan_ctx=a.plan_ctx, stitch=a.stitch, vocab=a.vocab, constr_weight=a.constr_weight),
                       goal=(a.goal_head or a.goal_module or "stub (G-STAY) for *_plan arms: NOT a HWM-PLAN result"))
    json.dump(rep, open(f"{a.out}/RESULT_HWMPLAN_{a.tag}.json", "w"), indent=1, default=float)
    open(f"{a.out}/RESULT_HWMPLAN_{a.tag}.md", "w", encoding="utf-8", newline="\n").write(E.to_markdown(rep))
    print("report rows", len(rep["rows"]), len(rep["replicate"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("windows", "plan", "report"):
        p = sp.add_parser(name); add_common(p)
        p.add_argument("--n", type=int, default=0)
        p.add_argument("--arms", nargs="+", default=list(E.ARMS)); p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
        p.add_argument("--workers", type=int, default=2); p.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    {"windows": cmd_windows, "plan": cmd_plan, "report": cmd_report}[a.cmd](a)
