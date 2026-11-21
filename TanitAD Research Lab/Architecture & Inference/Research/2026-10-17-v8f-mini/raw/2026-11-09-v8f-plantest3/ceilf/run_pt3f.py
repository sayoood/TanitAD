"""PREREG_V8F_PT3F: READ-END ceilings with readers fitted on every trained clip (N4565). No world model rollout, no planner. CPU only. Every write goes under this folder. The .cmd wrapper writes PT3F.done last."""
import hashlib, json, os, sys, time, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PT3 = os.path.abspath(os.path.join(HERE, ".."))
MLP4 = os.path.join(PT3, "mlp4")
sys.path.insert(0, MLP4)
import numpy as np
import torch
import pt3m_lib as PM
from pt3m_lib import DN, DL, P, THEORY
import plantest3_real as R3
sys.path.insert(0, os.path.join(THEORY, "decinfo", "decn", "fulln"))
import run_fulln as RF

RUNS = os.path.join(HERE, "runs")
FULLN_RUNS = os.path.join(THEORY, "decinfo", "decn", "fulln", "runs")
TAGS = ("M2b_s0", "M2b_s1")
LR = 1e-3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def guard_mutation():
    bad = 0
    for args in ((["a"], ["a"], [], ["e"], ["t"]), (["a"], ["b"], [], ["t"], ["t"]), (["a"], ["t"], [], ["e"], ["t"]), (["a"], ["b"], ["x"], ["e"], ["t"])):
        try:
            DN.split_guard_n(*args)
        except AssertionError:
            bad += 1
    assert bad == 4, "control 4 mutation: the guard let a violation through"


def full_order():
    first, cf = RF.lists()
    order = first + list(cf["clips"])
    assert len(order) == 4512 and len(set(order)) == 4512, "control 5: clip list"
    assert all(os.path.exists(RF.shard(c)) for c in order), "control 5: shards"
    return order


def assemble(tag, order, shard):
    key = f"z1_{tag}"
    T = DN.load_tune_npz()
    xs, ci, vs, ys = [], [], [], []
    for i, c in enumerate(order):
        z = np.load(shard(c))
        a = z[key]
        xs.append(a); ci += [i] * len(a); vs.append(z["v"]); ys.append(z["yaw"])
    X_tr, ci_tr = np.concatenate(xs), np.array(ci)
    del xs
    D, X, ids_tr = DN.assemble(T, key, X_tr, ci_tr, order, len(order), (np.concatenate(vs), np.concatenate(ys)))
    del X_tr
    ev = {f[:-3] for f in os.listdir(f"{DL.W.SF}/cache/eval/clips") if f.endswith(".pt")}
    DN.split_guard_n(set(map(str, D.eid[:D.nf])), set(map(str, D.eid[D.nf:])), DL.es_ids_of(set(map(str, D.eid[:D.nf]))), ev, ids_tr)
    Xs = DL.stack_ch(X, PM.M)
    yS = {k: v[:, PM.M - 1:] for k, v in DL.targets(D).items()}
    del X
    return D, Xs, yS


def fit_ridge(D, Xs, y):
    xf, yf = DL._flat(Xs, y, np.arange(D.nf))
    W, b, sd = PM.ridge_weights(xf, yf)
    xh, yh = DL._flat(Xs, y, np.arange(D.nf, D.n))
    del xf, yf
    p = (torch.as_tensor(xh, dtype=torch.double) @ W + b).numpy() * sd
    return dict(W=W, b=b, sd=sd), DL.r2(yh, p)


def fit_mlp(D, Xs, y):
    (xa, ya), (xv, yv), (xh, yh) = DL._parts(D, Xs, y, y, False)
    DL.wait_ram()
    m = PM.train_net(xa, ya, xv, yv, lr=LR)
    return m, DL.r2(yh.numpy().astype(np.float64), PM.mlp_predict(m, xh).astype(np.float64))


def ridge_apply(r, x):
    return (x.double() @ r["W"] + r["b"]).numpy() * r["sd"]


def fulln_row(tag, reader, tgt):
    rows = json.load(open(f"{FULLN_RUNS}/R0_{tag}_N4565.json", encoding="utf-8"))["rows"]
    return [q["r2"] for q in rows if q["reader"] == reader and q["m"] == 4 and q["target"] == tgt][0]


def n1053_yaw(tag):
    """PT3M's fit set, yaw-rate MLP at 1e-3."""
    c141, ce = DN.clip_lists()
    order = list(c141["clips"]) + list(ce["clips"])
    assert len(order) == 1000
    D, Xs, yS = assemble(tag, order, lambda c: f"{DN.LAT}/{c}.npz")
    m, r = fit_mlp(D, Xs, yS["yaw"])
    del D, Xs, yS
    return m, r


def one(tag):
    t0 = time.time()
    out = f"{RUNS}/pt3f_{tag}.npz"
    pre = f"{RUNS}/pre_{tag}.json"
    if os.path.exists(out) and os.path.exists(pre):
        return json.load(open(pre, encoding="utf-8"))
    DL.wait_ram()
    order = full_order()
    D, Xs, yS = assemble(tag, order, RF.shard)
    log(tag, "assembled: fit windows", D.nf, "fit episodes", D.n_fit_episodes, "held episodes", D.n_held_episodes, f"{time.time() - t0:.0f}s")
    rs, r_rs = fit_ridge(D, Xs, yS["speed"])
    log(tag, f"speed ridge {r_rs:+.3f}")
    ms, r_ms = fit_mlp(D, Xs, yS["speed"])
    log(tag, f"speed mlp {r_ms:+.3f} epochs {ms['epochs']}")
    my, r_my = fit_mlp(D, Xs, yS["yaw"])
    log(tag, f"yaw mlp {r_my:+.3f} epochs {my['epochs']}")
    fit_ids = set(map(str, D.eid[:D.nf]))
    del D, Xs, yS
    m1053, r_1053 = n1053_yaw(tag)
    log(tag, f"N1053 yaw mlp lr 1e-3 {r_1053:+.3f} epochs {m1053['epochs']}")
    torch.save(dict(ridge_speed=rs, mlp_speed=ms, mlp_yaw=my, n1053_yaw=m1053), f"{RUNS}/readers_{tag}.pt")

    ctx = R3.prepare3(tag)
    sc = np.asarray(ctx.score_idx)
    assert not (set(map(str, np.asarray(ctx.eid)[sc])) & fit_ids), "control 4: a SCORE episode is in the fit set"
    v0, w0 = ctx.vk[sc, 0], ctx.A[sc, 0, 1]
    X12, X3 = PM.stack_knots(ctx.Z1[sc], 12).float(), PM.stack_knots(ctx.Z1[sc], 3).float()
    v_r12, v_m12 = ridge_apply(rs, X12), PM.mlp_predict(ms, X12).astype(np.float64)
    v_r3, v_m3 = ridge_apply(rs, X3), PM.mlp_predict(ms, X3).astype(np.float64)
    w12, w1053 = PM.mlp_predict(my, X12).astype(np.float64), PM.mlp_predict(m1053, X12).astype(np.float64)
    use_mlp = bool(r_ms > r_rs)
    vd12, vd3, vo12 = (v_m12, v_m3, v_r12) if use_mlp else (v_r12, v_r3, v_m12)
    ceil = np.load(f"{MLP4}/runs/ceil_{tag}.npz")
    assert np.array_equal(ceil["idx"], sc), "SCORE order"
    rd = PM.load_reader(f"{MLP4}/runs/reader_{tag}.pt")
    y12 = rd.read_raw(X12)
    plans = dict(oracle=PM.end_plan(v0, ctx.vk[sc, 12], w0, ctx.yk[sc, 12]), hold=P.hold_plan(ctx.A[sc]), pt3m=PM.end_plan(v0, y12[:, 0], w0, y12[:, 1]),
                 f_decide=PM.end_plan(v0, vd12, w0, w12), f_other=PM.end_plan(v0, vo12, w0, w12), n1053_lr1e3=PM.end_plan(v0, y12[:, 0], w0, w1053))
    DT = P.DT
    acc_diff = (vd12 - vd3) / (9 * DT)
    plans["f_decide_readdiff"] = PM.end_plan(v0, v0 + acc_diff * 12 * DT, w0, w12)
    np.savez(out, idx=sc, v12_decide=vd12, w12_decide=w12, v12_true=ctx.vk[sc, 12], w12_true=ctx.yk[sc, 12], **plans)
    t = {"speed_ridge_m4": (r_rs, fulln_row(tag, "ridge", "speed")), "speed_mlp_m4": (r_ms, fulln_row(tag, "mlp", "speed")), "yaw_mlp_m4": (r_my, fulln_row(tag, "mlp", "yaw"))}
    rec = dict(tag=tag, n_clips=len(order), speed_form_deciding="mlp" if use_mlp else "ridge", held_r2=dict(speed_ridge=r_rs, speed_mlp=r_ms, yaw_mlp=r_my, n1053_yaw_mlp_lr1e3=r_1053),
               epochs=dict(speed_mlp=ms["epochs"], yaw_mlp=my["epochs"], n1053_yaw=m1053["epochs"]), control1_tie_fulln={k: dict(mine=a, fulln=b, abs_diff=abs(a - b), passed=bool(abs(a - b) <= 0.01)) for k, (a, b) in t.items()}, wall_s=time.time() - t0)
    json.dump(rec, open(pre, "w", encoding="utf-8"), indent=1)
    log(tag, "done", json.dumps(rec["control1_tie_fulln"]), f"{time.time() - t0:.0f}s")
    return rec


def shas():
    out = {}
    for p in [f"{MLP4}/tune_choice.json"] + [f"{MLP4}/runs/ceil_{t}.npz" for t in TAGS]:
        out[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


if __name__ == "__main__":
    os.makedirs(RUNS, exist_ok=True)
    status = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"))
    rc = 0
    try:
        guard_mutation()
        base = shas()
        for t in TAGS:
            rec = one(t)
            status[t] = dict(tie=all(v["passed"] for v in rec["control1_tie_fulln"].values()), speed=rec["speed_form_deciding"])
        status["isolation"] = bool(shas() == base)
    except Exception:
        log("FAILED", traceback.format_exc()); rc = 1
        status["failed"] = traceback.format_exc()[-800:]
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(status, open(f"{HERE}/run_pt3f.status.json", "w", encoding="utf-8"), indent=1)
    sys.exit(rc)
