"""PREREG_V8F_PT3N: PLANTEST3's FLAT1 DEC-END with the decoder fitted on N1053 (53 TUNE inner-fit episodes + DECN's 1,000 streamed train clips). The PLANTEST3 code is imported, not copied; one injection point: ctx.dec.
Usage: python run_pt3n.py <n_workers>. Every write goes under this folder (control 6). Tier T0, CPU only. The .cmd wrapper writes PT3N.done last."""
import hashlib, json, os, sys, time, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PT3 = os.path.abspath(os.path.join(HERE, ".."))
THEORY = os.path.abspath(os.path.join(PT3, ".."))
sys.path.insert(0, PT3)
import numpy as np
import torch
import run_plantest3 as RP
import plantest3_real as R3
import plantest3_lib as L3
import plantest_lib as P

RP.HERE = HERE                                     # choose_from_runs writes tune_choice.json next to this file, never PLANTEST3's
RP.RUNS = f"{HERE}/runs"
RP.ARMS = ("end",)
RUNS = RP.RUNS
TAGS = RP.TAGS
LAT = "F:/TanitAD-scratch/decn_scratch/run1/lat"
DECN_RUNS = f"{THEORY}/decinfo/decn/runs"
PT3_RUNS = f"{PT3}/runs"
_CTX = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def clip_order():
    c141 = json.load(open(f"{THEORY}/wmact/train_stream/clips141.json", encoding="utf-8"))
    ce = json.load(open(f"{THEORY}/decinfo/decn/clips_extra.json", encoding="utf-8"))
    order = list(c141["clips"]) + list(ce["clips"])
    assert len(order) == 1000 and len(set(order)) == 1000, "control 5: clip list"
    return order


def split_guard(fit_eval_ids, held_ids, score_ids, train_ids, eval_ids):
    """Control 4: the train additions are not eval clips; the eval part of the fit reads no SCORE and no inner-held-out episode."""
    fe, h, s, tr, ev = (set(map(str, x)) for x in (fit_eval_ids, held_ids, score_ids, train_ids, eval_ids))
    assert not (tr & ev), "a train clip is an eval clip"
    assert not (fe & s), "the decoder fit reads a SCORE episode"
    assert not (fe & h), "the decoder fit reads an inner-held-out episode"
    return True


def guard_mutation():
    """A mutated id set must raise, or the guard does not exist."""
    bad = 0
    for args in ((["a"], ["b"], ["a"], ["t"], ["a", "b"]), (["a"], ["a"], ["s"], ["t"], ["a", "s"]), (["a"], ["b"], ["s"], ["a"], ["a", "b", "s"])):
        try:
            split_guard(*args)
        except AssertionError:
            bad += 1
    assert bad == 3, "control 4 mutation: the guard let a violation through"


def load_dec(path_, expect=None):
    d = torch.load(path_, weights_only=False)
    dec = L3.Decoder(d["W1"], d["b1"], d["W2"], d["b2"], d["ystd"], n_samples=d.get("n_samples"))
    assert dec.digest() == (expect or d["sha256"]), "decoder digest"
    return dec


def inner_rows(ctx):
    e = np.asarray(ctx.eid)
    fit_ids, held_ids = L3.inner_split(e[ctx.tune_idx])
    ifit = np.array([i for i in ctx.tune_idx if e[i] in fit_ids])
    iheld = np.array([i for i in ctx.tune_idx if e[i] in held_ids])
    assert set(e[ifit]).isdisjoint(set(e[iheld])) and set(ifit) | set(iheld) == set(ctx.tune_idx)
    return ifit, iheld, fit_ids, held_ids


def fit_tag(tag):
    """Fit the N1053 decoder for one training seed, write decoder_<tag>.pt and pre_<tag>.json (the gate with controls 1 and 2)."""
    t0 = time.time()
    ctx = R3.prepare3(tag)
    ifit, iheld, fit_ids, held_ids = inner_rows(ctx)
    e = np.asarray(ctx.eid)
    order = clip_order()
    split_guard(e[ifit], held_ids, e[ctx.score_idx], order, e)
    assert len(fit_ids) == 53 and len(held_ids) == 17 and len(iheld) == 238
    Zs, vs, ys = [], [], []
    for c in order:
        z = np.load(f"{LAT}/{c}.npz")
        Zs.append(z[f"z1_{tag}"]); vs.append(z["v"]); ys.append(z["yaw"])
    Z = torch.cat([ctx.Z1[ifit].float(), torch.as_tensor(np.concatenate(Zs)).float()])
    v = np.concatenate([ctx.vk[ifit], np.concatenate(vs)])
    y = np.concatenate([ctx.yk[ifit], np.concatenate(ys)])
    dec = L3.fit_decoder(ctx.st, Z, v, y)
    torch.save(dict(W1=dec.W1, b1=dec.b1, W2=dec.W2, b2=dec.b2, ystd=dec.ystd, n_samples=dec.n_samples, sha256=dec.digest()), f"{RUNS}/decoder_{tag}.pt")
    st = ctx.st
    Zh, vh, yh = ctx.Z1[iheld], ctx.vk[iheld], ctx.yk[iheld]
    rec = dict(tag=tag, n_tune_windows=int(len(ctx.tune_idx)), n_inner_fit_windows=int(len(Z)), n_inner_held_windows=int(len(iheld)), n_inner_fit_episodes=1053, n_inner_held_episodes=len(held_ids), ystd_speed=float(dec.ystd[0]), ystd_yaw=float(dec.ystd[1]), ridge=L3.RIDGE,
               tune_speed_samples_missing=int(np.isnan(ctx.vk[ctx.tune_idx]).sum()), tune_samples_total=int(ctx.vk[ctx.tune_idx].size), fit_set="N1053")
    for lv in (1, 2):
        rs, ry, ns, ny = L3.heldout_r2(st, dec, lv, Zh, vh, yh)
        rec[f"enc_L{lv}"] = dict(r2_speed=rs, r2_yaw=ry, n_speed=ns, n_yaw=ny)
    rs, ry, ns, ny = L3.heldout_r2(st, dec, 1, Zh[:, 12:13], vh[:, 12:13], yh[:, 12:13])
    rec["enc_L1_k12"] = dict(r2_speed=rs, r2_yaw=ry, n_speed=ns, n_yaw=ny)
    with torch.no_grad():
        a_sc = ctx.wm.scale_actions(torch.as_tensor(ctx.A[iheld][:, :12]).float())
        z12 = st.rollout(1, ctx.Z1[iheld][:, 0], a_sc)[:, -1:]
    rs, ry, ns, ny = L3.heldout_r2(st, dec, 1, z12, vh[:, 12:13], yh[:, 12:13])
    rec["rollout_L1_k12_logged_actions"] = dict(r2_speed=rs, r2_yaw=ry, n_speed=ns, n_yaw=ny)
    rec["DECODER_WEAK_this_seed"] = bool(rec["enc_L1"]["r2_speed"] < L3.R2_SPEED_MIN or rec["enc_L1"]["r2_yaw"] < L3.R2_YAW_MIN)
    rec["gate_decoder_sha256"] = rec["planning_decoder_sha256"] = dec.digest()
    rec["planning_ystd"] = [float(dec.ystd[0]), float(dec.ystd[1])]
    # control 1: tie to DECN's ridge m = 1 at N1053
    rows = json.load(open(f"{DECN_RUNS}/R0_{tag}_N1053.json", encoding="utf-8"))["rows"]
    tie = {}
    for tgt, mine in (("speed", rec["enc_L1"]["r2_speed"]), ("yaw", rec["enc_L1"]["r2_yaw"])):
        them = [r["r2"] for r in rows if r["reader"] == "ridge" and r["m"] == 1 and r["target"] == tgt][0]
        tie[tgt] = dict(mine=mine, decn=them, abs_diff=abs(mine - them), passed=bool(abs(mine - them) <= 0.01))
    # control 2: the N53 fit through this code against PLANTEST3's gate
    d53 = R3.fit_rows(ctx, ifit)
    pre = json.load(open(f"{PT3_RUNS}/pre_{tag}.json", encoding="utf-8"))
    r53s, r53y, _, _ = L3.heldout_r2(st, d53, 1, Zh, vh, yh)
    ref = dict(speed=dict(mine=r53s, plantest3=pre["enc_L1"]["r2_speed"]), yaw=dict(mine=r53y, plantest3=pre["enc_L1"]["r2_yaw"]))
    for k in ref:
        ref[k]["abs_diff"] = abs(ref[k]["mine"] - ref[k]["plantest3"]); ref[k]["passed"] = bool(ref[k]["abs_diff"] <= 0.001)
    rec["control1_tie_decn"] = tie
    rec["control2_reference_n53"] = ref
    rec["wall_s"] = time.time() - t0
    json.dump(rec, open(f"{RUNS}/pre_{tag}.json", "w", encoding="utf-8"), indent=1)
    log("fit", tag, f"speed {rec['enc_L1']['r2_speed']:+.3f} yaw {rec['enc_L1']['r2_yaw']:+.3f}", "tie", {k: v["passed"] for k, v in tie.items()}, "ref", {k: v["passed"] for k, v in ref.items()}, f"{time.time() - t0:.0f}s")
    return all(v["passed"] for v in tie.values()) and all(v["passed"] for v in ref.values())


def path(job):
    if job[0] == "kin":
        return f"{RUNS}/kin_{job[1]}_rand.npz"
    if job[0] == "repro":
        return f"{RUNS}/repro_{job[1]}.json"
    return RP.path(job)


def get_ctx(tag):
    if tag not in _CTX:
        _CTX.clear()
        ctx = R3.prepare3(tag)
        ctx.dec = load_dec(f"{RUNS}/decoder_{tag}.pt")
        _CTX[tag] = ctx
    return _CTX[tag]


def work(job):
    if os.path.exists(path(job)):
        return job, 0.0, "skip"
    try:
        RP.wait_ram()
        ctx = get_ctx(job[1])
        t = time.time()
        if job[0] == "kin":
            for donor in (False, True):
                kp = R3.kin_unit(ctx, ctx.score_idx, donor=donor)
                np.savez(f"{RUNS}/kin_{job[1]}_{'rand' if donor else 'true'}.npz", plan_raw=kp, idx=np.asarray(ctx.score_idx))
            return job, time.time() - t, "ok"
        if job[0] == "repro":
            keep = ctx.dec
            try:
                ctx.dec = load_dec(f"{PT3_RUNS}/decoder_{job[1]}.pt")
                r = R3.unit3(ctx, "flat1", "true", ctx.score_idx, 1.0, "end", "plain", seed=0, collect=True)
            finally:
                ctx.dec = keep
            old = np.load(f"{PT3_RUNS}/score_{job[1]}_flat1_end_plain_true_L1_p0.npz")
            same = bool(np.array_equal(r["plan_raw"], old["plan_raw"]) and np.array_equal(r["idx"], old["idx"]))
            json.dump(dict(unit=f"{job[1]} flat1 end plain true L1 p0", identical=same, max_abs_diff=float(np.abs(r["plan_raw"] - old["plan_raw"]).max()), wall_s=time.time() - t), open(path(job), "w", encoding="utf-8"), indent=1)
            return job, time.time() - t, "ok" if same else "ERR control 3: plans differ"
        if job[0] == "tune":
            _, _, arm, form, lam = job
            r = R3.unit3(ctx, "flat1", "true", ctx.tune_idx, lam, arm, form, seed=0, collect=False)
            m = P.window_metrics(r["plan_raw"], ctx.A[ctx.tune_idx], ctx.v0[ctx.tune_idx])
            r["acc_mae_mean"], r["yaw_mae_mean"] = np.array(m["acc_mae"].mean()), np.array(m["yaw_mae"].mean())
        else:
            _, _, kind, arm, form, goal, lam, pseed = job
            r = R3.unit3(ctx, kind, goal, ctx.score_idx, lam, arm, form, seed=pseed, collect=True)
        tmp = path(job) + ".tmp.npz"
        np.savez(tmp, **r)
        os.replace(tmp, path(job))
        return job, time.time() - t, "ok"
    except Exception:
        return job, 0.0, "ERR " + traceback.format_exc()[-800:]


def run_stage(pool, J, label):
    bad = 0
    for job, dt, st in pool.imap_unordered(work, J, chunksize=1):
        log(label, job, round(dt, 1), "s", st[:900])
        bad += st.startswith("ERR")
    return bad


def tune_sha():
    return hashlib.sha256(open(f"{PT3}/tune_choice.json", "rb").read()).hexdigest()


if __name__ == "__main__":
    import multiprocessing as mp
    n = int(sys.argv[1])
    os.makedirs(RUNS, exist_ok=True)
    guard_mutation()
    base = open(f"{HERE}/tune_choice_orig.sha256", encoding="utf-8").read().strip()
    assert tune_sha() == base, "control 6: PLANTEST3's tune_choice.json moved before this run"
    status = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"))
    ok = True
    for t in TAGS:
        if os.path.exists(f"{RUNS}/pre_{t}.json"):
            rec = json.load(open(f"{RUNS}/pre_{t}.json", encoding="utf-8"))
            ok &= all(v["passed"] for v in rec["control1_tie_decn"].values()) and all(v["passed"] for v in rec["control2_reference_n53"].values())
        else:
            RP.wait_ram()
            ok &= fit_tag(t)
    if not ok:
        log("controls 1 or 2 failed: no planning")
        json.dump(dict(status, failed="controls 1 or 2"), open(f"{HERE}/run_pt3n.status.json", "w", encoding="utf-8"))
        sys.exit(2)
    errs = 0
    with mp.get_context("spawn").Pool(n, maxtasksperchild=4) as pool:
        errs += run_stage(pool, [("repro", "M2b_s0")] + [("kin", t) for t in TAGS], "PRE")
        if errs == 0:
            ea = run_stage(pool, RP.jobs_a(), "A")
            errs += ea
            if ea == 0:
                choice, tab = RP.choose_from_runs()
                log("CHOICE", json.dumps(choice))
                B1, _, B3 = RP.jobs_b(choice, tab)
                errs += run_stage(pool, B1, "B1")
                errs += run_stage(pool, [j for j in B3 if j[2] == "flat1"], "B3")
    status.update(errors=errs, finished=time.strftime("%Y-%m-%d %H:%M:%S"), tune_choice_sha_unchanged=bool(tune_sha() == base))
    json.dump(status, open(f"{HERE}/run_pt3n.status.json", "w", encoding="utf-8"), indent=1)
    assert status["tune_choice_sha_unchanged"], "control 6"
    sys.exit(1 if errs else 0)
