"""holdread.py -- PREREG_V8F_PROOF_ADD15 (HOLDREAD). CPU, zero training, zero rollouts.
Evidence class MEASURED; tier T1-kin (kinematic rollout on the model's own actions, logged frames): a diagnostic, never driving performance.
The hold rule is an EMULATION on banked rollouts, not a rollout. STRATEGIC family: PAUSED (PI item 25).

Usage: python holdread.py run [--out-dir DIR] [--nboot 2000]
Runs controls c1..c6 first, then Q1, Q2, Q3 for pair 1 and pair 2, both arms, seeds 0 and 1; writes RESULT_HOLDREAD.json and VERDICT_HOLDREAD.md.
Exit 0 = every control reads its known value; 2 = a control failed (the read is then not valid; the numbers are still printed).
Definitions and reading choices: HOLD_DEFS.md.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
sys.path.insert(0, "F:/TanitAD-scratch/v8proof/domprem")
sys.path.insert(0, "F:/TanitAD-scratch/v8proof/hier")
import domprem as DP  # noqa: E402  (sets up the pinned tool tree through hierwhere, loads nothing)

H = DP.H
np = H.np
H.torch.set_num_threads(2)
FF = H.FF

CARD = "F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD15.md"
CARD_SHA = "168ad1a7ddc98fabdfb080fbfdc5ea171a114f93f674a92d1bfeac1add777f8b"
PAIRS = DP.PAIRS
SEEDS = (0, 1)
ARMS = ("H", "F")
ARM_NAME = {"H": "hierarchy", "F": "flat"}
REST_V0 = 2.0
TOL = 2.0
GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
NFOLD = 5
NB = 2000
DT = 0.2
KEYS = ("g", "cl", "ol", "ws", "eid", "clip_index", "v0")
DOMPREM_REST = (211, 29)


# ------------------------------------------------------------------------------------------------ loading
def load_dump(d):
    """all keys of the dump folder, sorted file order (as hierwhere.load_arm), plus `ol`."""
    fs = sorted(glob.glob(d + "/ep*.npz"))
    assert len(fs) == 141, (d, len(fs))
    out = {k: [] for k in KEYS}
    for f in fs:
        with np.load(f) as z:
            for k in KEYS:
                out[k].append(np.asarray(z[k]))
    return {k: np.concatenate(v) for k, v in out.items()}


def load_arm_checked(d):
    mine = load_dump(d)
    ref = H.load_arm(d)
    for k in ("g", "cl", "ws", "eid", "clip_index", "v0"):
        assert np.array_equal(mine[k], ref[k]), f"{d}: {k} differs from hierwhere.load_arm"
    return mine


# ------------------------------------------------------------------------------------------------ pure functions (tested)
def norm(x):
    return np.linalg.norm(x, axis=-1)


def ade6(x, g):
    """mean over the 30 knots of ||x - g||, [n]."""
    return norm(np.asarray(x, np.float64) - np.asarray(g, np.float64)).mean(axis=1)


def endpoint_hit(x, g, tol=TOL):
    """Hit: ||x[29] - g[29]|| <= tol, as 0/1 float [n]."""
    return (norm(np.asarray(x, np.float64)[:, 29] - np.asarray(g, np.float64)[:, 29]) <= tol).astype(np.float64)


def fold_assign(eid_rest, seed=0, k=NFOLD):
    """episodes of the REST context, sorted, shuffled with np.random.default_rng(seed).permutation, dealt round-robin. -> dict episode -> fold."""
    eps = np.unique(np.asarray(eid_rest))
    perm = np.random.default_rng(seed).permutation(len(eps))
    return {int(eps[perm[i]]): i % k for i in range(len(eps))}


def select_theta(T_fit, ade_cl_fit, ade_stand_fit, grid=GRID):
    """theta of the grid with the lowest mean ADE6 of the rule over the fit windows (standing where T_ol < theta); ties: the smaller theta."""
    best, bv = None, None
    for th in sorted(grid):
        v = float(np.where(T_fit < th, ade_stand_fit, ade_cl_fit).mean())
        if bv is None or v < bv:
            best, bv = th, v
    return best, bv


def replace_rule(cl, T_ol, rest, theta, invert=False):
    """Hold rule: on REST windows with T_ol < theta (invert: T_ol >= theta) the track is standing at the origin for all 30 knots; everything else keeps cl.
    theta may be a scalar or a per-window array."""
    hold = rest & ((T_ol >= theta) if invert else (T_ol < theta))
    out = np.array(cl, np.float64, copy=True)
    out[hold] = 0.0
    return out, hold


def crossfit(cl, g, T_ol, rest, eid, grid=GRID, seed=0, k=NFOLD):
    """5-fold cross-fitting of theta by episode. -> dict(theta_w [n] per-window theta (NaN outside REST), folds, per-fold chosen theta + fit ADE, fold_of_window [n] (-1 outside REST))."""
    fmap = fold_assign(eid[rest], seed, k)
    fold_w = np.full(len(eid), -1, np.int64)
    for i in np.flatnonzero(rest):
        fold_w[i] = fmap[int(eid[i])]
    a_cl = ade6(cl, g)
    a_st = norm(g).mean(axis=1)                       # ADE6 of standing at the origin: mean over knots of ||0 - g||
    theta_w = np.full(len(eid), np.nan)
    per_fold, splits = [], []
    for f in range(k):
        fit = rest & (fold_w != f) & (fold_w >= 0)
        held = rest & (fold_w == f)
        th, v = select_theta(T_ol[fit], a_cl[fit], a_st[fit], grid)
        theta_w[held] = th
        per_fold.append({"fold": f, "theta": th, "fit_mean_ade6_m": v, "n_fit_windows": int(fit.sum()), "n_heldout_windows": int(held.sum()),
                         "n_heldout_episodes": int(len(np.unique(eid[held])))})
        splits.append((set(np.unique(eid[fit]).tolist()), set(np.unique(eid[held]).tolist())))
    return {"theta_w": theta_w, "fold_w": fold_w, "per_fold": per_fold, "splits": splits, "fmap": fmap}


def auroc(score, label):
    """AUROC of `score` for label True (positive) against False; ties count half. NaN if a class is empty."""
    score = np.asarray(score, np.float64)
    label = np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def cross_track(x, g):
    """|component of x[29] - g[29] orthogonal to the logged heading of the last valid logged step|, [n]; windows with fewer than 2 valid logged steps
    use the ego-frame lateral axis (|y| of the endpoint error). Also returns the boolean fallback mask. A step is valid when its length exceeds FF.MIN_DS_MPS * DT."""
    x = np.asarray(x, np.float64)
    g = np.asarray(g, np.float64)
    p = np.concatenate([np.zeros_like(g[:, :1]), g], axis=1)
    d = p[:, 1:] - p[:, :-1]
    ds = norm(d)
    valid = ds > FF.MIN_DS_MPS * DT
    nv = valid.sum(axis=1)
    e = x[:, 29] - g[:, 29]
    out = np.abs(e[:, 1]).copy()
    fall = nv < 2
    for i in np.flatnonzero(~fall):
        last = np.flatnonzero(valid[i])[-1]
        h = d[i, last] / ds[i, last]
        out[i] = abs(h[0] * e[i, 1] - h[1] * e[i, 0])
    return out, fall


def fam_values(x, g):
    """per-window values of the three card families for track x (STRATEGIC: PAUSED, not computed)."""
    ct, fall = cross_track(x, g)
    return {"LON_speed_mae_mps": H.comps(x, g)["speed_mae_mps"], "LAT_cross_track_6s_m": ct,
            "TAC_cell_right": (H.joint(*H.modes(x)) == H.joint(*H.modes(g))).astype(np.float64), "fallback": fall}


# ------------------------------------------------------------------------------------------------ bootstrap statistics
def auroc_stat(score, go, rest):
    def f(sel):
        sub = sel[rest[sel]]
        return auroc(score[sub], go[sub])
    return f


def quant_stat(v, mask, q):
    def f(sel):
        sub = sel[mask[sel]]
        return float(np.percentile(v[sub], q)) if len(sub) else float("nan")
    return f


def fr(r, nd=4, sign=False):
    s = "{:+.%df}" % nd if sign else "{:.%df}" % nd
    return f"{s.format(r['point'])} [{s.format(r['lo'])}, {s.format(r['hi'])}]"


def row(r):
    return {"point": float(r["point"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"])}


# ------------------------------------------------------------------------------------------------ one (pair, seed)
def build(pair, s):
    dh = load_arm_checked(DP.PAIRS[pair]["H"].format(s=s))
    df = load_arm_checked(DP.PAIRS[pair]["F"].format(s=s))
    for k in ("g", "ws", "eid", "clip_index", "v0"):
        assert np.array_equal(dh[k], df[k]), f"{pair} s{s}: arms differ in {k}"
    g = np.asarray(dh["g"], np.float64)
    v0 = np.asarray(dh["v0"], np.float64)
    rest = v0 < REST_V0
    stay = rest & (norm(g[:, 29]) <= TOL)
    go = rest & ~stay
    W = {"g": g, "v0": v0, "eid": dh["eid"], "rest": rest, "stay": stay, "go": go}
    for arm, d in (("H", dh), ("F", df)):
        W[arm] = {"cl": np.asarray(d["cl"], np.float64), "ol": np.asarray(d["ol"], np.float64)}
        W[arm]["T_cl"] = norm(W[arm]["cl"][:, 29])
        W[arm]["T_ol"] = norm(W[arm]["ol"][:, 29])
    return W


def arm_rule(W, arm):
    """cross-fitted rule, the inverted rule, the privileged rule for one arm; per-window ADE6 / hit / family values."""
    A = W[arm]
    g, rest, eid = W["g"], W["rest"], W["eid"]
    cf = crossfit(A["cl"], g, A["T_ol"], rest, eid)
    ruled, hold = replace_rule(A["cl"], A["T_ol"], rest, cf["theta_w"])
    inv, hold_inv = replace_rule(A["cl"], A["T_ol"], rest, cf["theta_w"], invert=True)
    priv = np.array(A["cl"], copy=True)
    priv[W["stay"]] = 0.0
    return {"cf": cf, "ruled": ruled, "hold": hold, "inv": inv, "hold_inv": hold_inv, "priv": priv}


def read_world(W, nb):
    g, rest, stay, go, eid = W["g"], W["rest"], W["stay"], W["go"], W["eid"]
    allw = np.ones(len(eid), bool)
    stats, aux = {}, {}
    n_rest, ep_rest = int(rest.sum()), int(len(np.unique(eid[rest])))
    # ---- Q2: v0 alone
    stats["Q2|v0"] = auroc_stat(W["v0"], go, rest)
    for arm in ARMS:
        A = W[arm]
        R = arm_rule(W, arm)
        aux[arm] = R
        # ---- Q1
        for nm, v, q in (("median", A["T_cl"], 50), ("p90", A["T_cl"], 90)):
            stats[f"Q1|{arm}|{nm}"] = quant_stat(v, stay, q)
        hit_b = endpoint_hit(A["cl"], g)
        stats[f"Q1|{arm}|hit"] = H.mean_over(hit_b, stay)
        # ---- Q2
        stats[f"Q2|{arm}|auroc"] = auroc_stat(A["T_ol"], go, rest)
        # ---- Q3
        a_b = ade6(A["cl"], g)
        fam_b = fam_values(A["cl"], g)
        for rule in ("ruled", "priv"):
            x = R[rule]
            d_ade = ade6(x, g) - a_b
            d_hit = endpoint_hit(x, g) - hit_b
            stats[f"Q3|{arm}|{rule}|ade6_rest"] = H.mean_over(d_ade, rest)
            stats[f"Q3|{arm}|{rule}|ade6_all"] = H.total_over(d_ade, rest)
            stats[f"Q3|{arm}|{rule}|hit_rest"] = H.mean_over(d_hit, rest)
            stats[f"Q3|{arm}|{rule}|hit_all"] = H.total_over(d_hit, rest)
            fam_x = fam_values(x, g)
            for fk in ("LON_speed_mae_mps", "LAT_cross_track_6s_m", "TAC_cell_right"):
                stats[f"Q3|{arm}|{rule}|{fk}"] = H.mean_over(fam_x[fk] - fam_b[fk], rest)
        aux[arm]["fam_b"] = fam_b
    t0 = time.time()
    res = H.boot(eid, stats, nb)
    print(f"   boot: {len(stats)} statistics x {nb} draws in {time.time() - t0:.0f}s", flush=True)
    out = {"n_rest": n_rest, "ep_rest": ep_rest, "n_stay": int(stay.sum()), "ep_stay": int(len(np.unique(eid[stay]))), "n_go": int(go.sum()),
           "ep_go": int(len(np.unique(eid[go]))), "n_all": int(len(eid)), "ep_all": int(len(np.unique(eid))), "v0_alone_auroc": row(res["Q2|v0"])}
    for arm in ARMS:
        A, R = W[arm], aux[arm]
        a_b = ade6(A["cl"], g)
        hit_b = endpoint_hit(A["cl"], g)
        e = {"Q1": {"median_T_cl_m": row(res[f"Q1|{arm}|median"]), "p90_T_cl_m": row(res[f"Q1|{arm}|p90"]), "hit_rate": row(res[f"Q1|{arm}|hit"]),
                    "n_stay": int(stay.sum()), "ep_stay": out["ep_stay"]},
             "Q2": {"auroc_T_ol": row(res[f"Q2|{arm}|auroc"])},
             "Q3": {"theta_per_fold": R["cf"]["per_fold"], "n_held": int(R["hold"].sum()), "n_held_stay": int((R["hold"] & stay).sum()),
                    "n_held_go": int((R["hold"] & go).sum()), "banked_mean_ade6_rest_m": float(a_b[rest].mean()),
                    "ruled_mean_ade6_rest_m": float(ade6(R["ruled"], g)[rest].mean()), "privileged_mean_ade6_rest_m": float(ade6(R["priv"], g)[rest].mean()),
                    "banked_hit_rest": float(hit_b[rest].mean()), "ruled_hit_rest": float(endpoint_hit(R["ruled"], g)[rest].mean()),
                    "privileged_hit_rest": float(endpoint_hit(R["priv"], g)[rest].mean()),
                    "cross_track_fallback_windows_rest": int(R["fam_b"]["fallback"][rest].sum())}}
        for rule in ("ruled", "priv"):
            e["Q3"][rule] = {k: row(res[f"Q3|{arm}|{rule}|{k}"]) for k in ("ade6_rest", "ade6_all", "hit_rest", "hit_all", "LON_speed_mae_mps", "LAT_cross_track_6s_m", "TAC_cell_right")}
        out[arm] = e
    return out, aux


# ------------------------------------------------------------------------------------------------ controls
def run_controls(data):
    C = []

    def add(cid, what, observed, known, ok):
        C.append({"id": cid, "what": what, "observed": observed, "known": known, "PASS": bool(ok)})
        print(f"  {cid:22s} {'PASS' if ok else 'FAIL'}  {what}: observed {observed}  known {known}", flush=True)
    for (pair, s), D in data.items():
        W = D["W"]
        g, rest, stay, go, eid = W["g"], W["rest"], W["stay"], W["go"], W["eid"]
        n_ep = int(len(np.unique(eid[rest])))
        add(f"c5_{pair}_s{s}", "REST windows, REST episodes", [int(rest.sum()), n_ep], list(DOMPREM_REST), (int(rest.sum()), n_ep) == DOMPREM_REST)
        for arm in ARMS:
            A = W[arm]
            assert np.all(np.isfinite(A["T_ol"]))
            # c1
            x0, h0 = replace_rule(A["cl"], A["T_ol"], rest, 0.0)
            mx = float(np.abs(x0 - A["cl"]).max())
            add(f"c1_{pair}_s{s}_{arm}", "H(0): max abs difference to the banked track; windows held", [mx, int(h0.sum())], [0.0, 0], mx == 0.0 and int(h0.sum()) == 0)
            # c2
            xi, hi = replace_rule(A["cl"], A["T_ol"], rest, float("inf"))
            hit = endpoint_hit(xi, g)
            nrest, nstay = int(rest.sum()), int(stay.sum())
            outside = float(np.abs(xi[~rest] - A["cl"][~rest]).max()) if (~rest).any() else 0.0
            ok = (int(hit[rest].sum()) == nstay) and (float(hit[rest].mean()) == nstay / nrest) and outside == 0.0 and int(hi.sum()) == nrest
            add(f"c2_{pair}_s{s}_{arm}", "H(inf) on REST: hit rate; share of STAY among REST; max change outside REST", [float(hit[rest].mean()), outside], [nstay / nrest, 0.0], ok)
            # c3
            R = arm_rule(W, arm)
            hp = endpoint_hit(R["priv"], g)
            go_change = float(np.abs(R["priv"][~stay] - A["cl"][~stay]).max())
            ok = bool(np.all(hp[stay] == 1.0)) and float(hp[stay].mean()) == 1.0 and go_change == 0.0
            add(f"c3_{pair}_s{s}_{arm}", "privileged rule: hit rate on STAY; max change on non-STAY windows", [float(hp[stay].mean()), go_change], [1.0, 0.0], ok)
            # c4
            ov = [len(a & b) for a, b in R["cf"]["splits"]]
            cover = sorted(sum((list(b) for _, b in R["cf"]["splits"]), []))
            ok = all(o == 0 for o in ov) and cover == sorted(np.unique(eid[rest]).tolist())
            add(f"c4_{pair}_s{s}_{arm}", "episodes in both fit and held-out fold, per split; held-out folds partition the REST episodes", ov, [0] * NFOLD, ok)
    # c6 (pair 1): hierarchy arm decides; the flat arm is printed
    for arm in ARMS:
        for s in SEEDS:
            W = data[("pair1", s)]["W"]
            R = arm_rule(W, arm)
            rest, g = W["rest"], W["g"]
            m_rule, m_inv = float(ade6(R["ruled"], g)[rest].mean()), float(ade6(R["inv"], g)[rest].mean())
            ok = m_inv > m_rule
            tag = "c6" if arm == "H" else "c6info"
            add(f"{tag}_pair1_s{s}_{arm}", f"inverted rule mean ADE6 on REST above the rule as defined ({ARM_NAME[arm]})", [round(m_inv, 4), round(m_rule, 4)], "inverted > rule", ok if arm == "H" else True)
            C[-1]["observed_inverted_gt_rule"] = bool(ok)
    return C


# ------------------------------------------------------------------------------------------------ outcomes (code)
def label_q1(R):
    return "CREEP" if all(R["pair1"][f"s{s}"]["H"]["Q1"]["median_T_cl_m"]["point"] > 2.0 for s in SEEDS) else "HOLDS"


def label_q2(R, pair="pair1"):
    r = [R[pair][f"s{s}"]["H"]["Q2"]["auroc_T_ol"] for s in SEEDS]
    if all(x["lo"] > 0.80 for x in r):
        return "CUE-PRESENT"
    if all(x["hi"] < 0.65 for x in r):
        return "CUE-ABSENT"
    return "CUE-WEAK"


def label_q3(R, pair="pair1"):
    r = [R[pair][f"s{s}"]["H"]["Q3"]["ruled"]["ade6_rest"] for s in SEEDS]
    if all(x["point"] < 0 and x["hi"] < 0 for x in r):
        return "HOLD-HELPS"
    if all(x["point"] > 0 and x["lo"] > 0 for x in r):
        return "HOLD-COSTS"
    return "HOLD-UNDECIDED"


def follows_text(q3, q2):
    if q3 == "HOLD-HELPS":
        return ("HOLD-HELPS: the next step is the rule inside a real T1-kin rollout (the tooling of RULESHIELD), theta fitted on training clips, with the four "
                "families and the six seeds of HIERREP as the replicate.")
    if q3 in ("HOLD-UNDECIDED", "HOLD-COSTS") and q2 in ("CUE-ABSENT", "CUE-WEAK"):
        return (f"{q3} with {q2}: the rule lacks a cue; the ceiling row says what a cue would be worth, and the lever is a go / stay cue at rest from perception, not the rule.")
    if q3 == "HOLD-COSTS" and q2 == "CUE-PRESENT":
        return "HOLD-COSTS with CUE-PRESENT: the emulation is the suspect; the rollout version decides."
    return (f"{q3} with {q2}: the card commits no text for this combination (it commits HOLD-HELPS; HOLD-UNDECIDED or HOLD-COSTS with CUE-ABSENT or CUE-WEAK; HOLD-COSTS with CUE-PRESENT). "
            "The numbers above are printed as they are and no next step is read from the card.")


# ------------------------------------------------------------------------------------------------ verdict
def write_verdict(res, path):
    R, O, C = res["results"], res["outcomes"], res["controls"]
    L = []
    w = L.append
    w("# VERDICT_HOLDREAD (PREREG_V8F_PROOF_ADD15)")
    w("")
    w(f"Card sha256 {res['card_sha256']}; holdread.py sha256 {res['holdread_sha256']}.")
    w("Evidence class MEASURED. Tier T1-kin (kinematic rollout on the model's own actions, logged frames), zero training, zero rollouts: a diagnostic read, never driving performance.")
    w("**The hold rule is an EMULATION on banked rollouts, not a rollout.** A held car in a T1-kin rollout decides again at 2 s and 4 s; here it stands for the whole window.")
    w("STRATEGIC family: PAUSED (PI item 25), reported as PAUSED, not as a pass.")
    w("")
    w("## Controls (run first)")
    w("")
    w(f"Read valid: **{'YES, every control reads its known value' if res['controls_all_pass'] else 'NO, a control failed; the numbers below are printed and are not a valid read'}**.")
    w("")
    w("| control | what | observed | known | |")
    w("|---|---|---|---|---|")
    for c in C:
        w(f"| {c['id']} | {c['what']} | {c['observed']} | {c['known']} | {'PASS' if c['PASS'] else 'FAIL'} |")
    w("")
    w("## Card outcome")
    w("")
    w(f"* Q1 label (pair 1): **{O['pair1']['Q1']}**; Q2 label (pair 1, hierarchy): **{O['pair1']['Q2']}**; Q3 card outcome (pair 1, hierarchy, ADE6 on REST): **{O['pair1']['Q3']}**.")
    w(f"* Same labels on pair 2 (printed beside, decide nothing): Q1 {O['pair2']['Q1']}; Q2 {O['pair2']['Q2']}; Q3 {O['pair2']['Q3']}.")
    w("")
    for pair in ("pair1", "pair2"):
        w(f"### Counts, {pair}")
        w("")
        w("| seed | REST windows (episodes) | STAY | GO | all windows (episodes) |")
        w("|---|---|---|---|---|")
        for s in SEEDS:
            r = R[pair][f"s{s}"]
            w(f"| {s} | {r['n_rest']} ({r['ep_rest']}) | {r['n_stay']} ({r['ep_stay']}) | {r['n_go']} ({r['ep_go']}) | {r['n_all']} ({r['ep_all']}) |")
        w("")
    for pair in ("pair1", "pair2"):
        w(f"## Q1, creep on STAY windows: {pair}")
        w("")
        w("T_cl = ||cl[29]|| in metres; interval: episode-cluster bootstrap (B 2,000, seed 0). Hit: ||cl[29] - g[29]|| <= 2.0 m.")
        w("")
        w("| seed | arm | STAY windows (episodes) | median T_cl [95 %] | 90th percentile T_cl [95 %] | hit rate [95 %] |")
        w("|---|---|---|---|---|---|")
        for s in SEEDS:
            for arm in ARMS:
                q = R[pair][f"s{s}"][arm]["Q1"]
                w(f"| {s} | {ARM_NAME[arm]} | {q['n_stay']} ({q['ep_stay']}) | {fr(q['median_T_cl_m'], 3)} | {fr(q['p90_T_cl_m'], 3)} | {fr(q['hit_rate'], 3)} |")
        w("")
    w(f"Q1 label, pair 1: **{O['pair1']['Q1']}** (CREEP if the hierarchy's median T_cl on STAY windows is above 2.0 m at both seeds of pair 1, else HOLDS).")
    w("")
    for pair in ("pair1", "pair2"):
        w(f"## Q2, does the plan at the window start separate GO from STAY at rest: {pair}")
        w("")
        w("AUROC of T_ol = ||ol[29]|| for GO against STAY on REST windows, ties counted half; beside it the AUROC of `v0` alone (the label comes from the logged motion, so `v0` partly reads its source).")
        w("")
        w("| seed | arm | AUROC of T_ol [95 %] | AUROC of v0 alone [95 %] | GO / STAY windows |")
        w("|---|---|---|---|---|")
        for s in SEEDS:
            r = R[pair][f"s{s}"]
            for arm in ARMS:
                w(f"| {s} | {ARM_NAME[arm]} | {fr(r[arm]['Q2']['auroc_T_ol'], 3)} | {fr(r['v0_alone_auroc'], 3)} | {r['n_go']} / {r['n_stay']} |")
        w("")
    w(f"Q2 label, pair 1, hierarchy: **{O['pair1']['Q2']}** (CUE-PRESENT if the lower bound is above 0.80 at both seeds, CUE-ABSENT if the upper bound is below 0.65 at both, else CUE-WEAK).")
    w("")
    for pair in ("pair1", "pair2"):
        w(f"## Q3, the hold rule (EMULATION), cross-fitted: {pair}")
        w("")
        w("Differences are ruled arm minus the arm as banked; the interval is the paired episode-cluster bootstrap with the cross-fitted tracks held fixed. ADE6 in metres; hit-rate change in share points of windows.")
        w("")
        w("| seed | arm | REST: ADE6 change [95 %] | all windows: ADE6 change [95 %] | REST: hit-rate change [95 %] | all windows: hit-rate change [95 %] | banked / ruled ADE6 on REST | banked / ruled hit on REST | windows held (STAY / GO) |")
        w("|---|---|---|---|---|---|---|---|---|")
        for s in SEEDS:
            for arm in ARMS:
                q = R[pair][f"s{s}"][arm]["Q3"]
                r = q["ruled"]
                w(f"| {s} | {ARM_NAME[arm]} | {fr(r['ade6_rest'], 4, True)} | {fr(r['ade6_all'], 4, True)} | {fr(r['hit_rest'], 4, True)} | {fr(r['hit_all'], 4, True)} | "
                  f"{q['banked_mean_ade6_rest_m']:.3f} / {q['ruled_mean_ade6_rest_m']:.3f} | {q['banked_hit_rest']:.3f} / {q['ruled_hit_rest']:.3f} | {q['n_held']} ({q['n_held_stay']} / {q['n_held_go']}) |")
        w("")
        w(f"Chosen theta per fold ({pair}); grid {{0.5, 1, 2, 3, 4, 6, 8, 12}} m, lowest mean ADE6 on the four fit folds, ties to the smaller theta:")
        w("")
        w("| seed | arm | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |")
        w("|---|---|---|---|---|---|---|")
        for s in SEEDS:
            for arm in ARMS:
                pf = R[pair][f"s{s}"][arm]["Q3"]["theta_per_fold"]
                w(f"| {s} | {ARM_NAME[arm]} | " + " | ".join(f"{p['theta']:g} (held-out {p['n_heldout_windows']})" for p in pf) + " |")
        w("")
        w(f"Privileged ceiling ({pair}): the rule with the logged STAY label in place of the cue (a label, not an input), same differences:")
        w("")
        w("| seed | arm | REST: ADE6 change [95 %] | all windows: ADE6 change [95 %] | REST: hit-rate change [95 %] | all windows: hit-rate change [95 %] | ceiling ADE6 on REST |")
        w("|---|---|---|---|---|---|---|")
        for s in SEEDS:
            for arm in ARMS:
                q = R[pair][f"s{s}"][arm]["Q3"]
                r = q["priv"]
                w(f"| {s} | {ARM_NAME[arm]} | {fr(r['ade6_rest'], 4, True)} | {fr(r['ade6_all'], 4, True)} | {fr(r['hit_rest'], 4, True)} | {fr(r['hit_all'], 4, True)} | {q['privileged_mean_ade6_rest_m']:.3f} |")
        w("")
    w("## Four families, ruled arm minus the arm as banked, REST windows (cross-fitted rule; privileged ceiling in brackets)")
    w("")
    w("LONGITUDINAL = mean absolute speed error over the 30 knots (m/s); LATERAL = mean absolute cross-track error at 6 s (m); TACTICAL = share of windows whose joint cell equals the logged cell (share points); STRATEGIC = PAUSED (PI item 25).")
    w("")
    w("| pair | seed | arm | LONGITUDINAL change [95 %] | LATERAL change [95 %] | TACTICAL change [95 %] | STRATEGIC |")
    w("|---|---|---|---|---|---|---|")
    for pair in ("pair1", "pair2"):
        for s in SEEDS:
            for arm in ARMS:
                q = R[pair][f"s{s}"][arm]["Q3"]
                cells = []
                for k in ("LON_speed_mae_mps", "LAT_cross_track_6s_m", "TAC_cell_right"):
                    cells.append(f"{fr(q['ruled'][k], 4, True)} ({q['priv'][k]['point']:+.4f})")
                w(f"| {pair} | {s} | {ARM_NAME[arm]} | " + " | ".join(cells) + " | PAUSED |")
    w("")
    fb = {f"{p}|s{s}|{a}": R[p][f"s{s}"][a]["Q3"]["cross_track_fallback_windows_rest"] for p in ("pair1", "pair2") for s in SEEDS for a in ARMS}
    w(f"Windows on REST that used the ego-frame lateral axis because the logged future has fewer than 2 valid steps (counted, as the card requires): {json.dumps(fb)}.")
    w("")
    w("## What follows (the paragraph the card commits for the outcome reached)")
    w("")
    w(res["follows"])
    w("")
    w("## Limits (card section 6)")
    w("")
    w("* An emulation: a held car in a T1-kin rollout decides again at 2 s and 4 s; here it stands for the whole window. The rollout version is the committed next step and is the number that would count.")
    w("* theta is fitted on evaluation episodes by cross-fitting, not on training clips (no training rollouts are banked on this box).")
    w("* 211 windows of 29 episodes: wide intervals are expected. STAY is defined with the logged future, so it is a label, not an input.")
    w("* One training run per arm per seed: a separated interval is necessary and not sufficient.")
    w("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--nboot", type=int, default=NB)
    a = ap.parse_args(argv)
    t0 = time.time()
    sha = hashlib.sha256(open(CARD, "rb").read()).hexdigest()
    assert sha == CARD_SHA, f"card hash differs: {sha}"
    me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    print("card sha256 ok", sha, "holdread.py sha256", me, flush=True)
    data = {}
    for pair in PAIRS:
        for s in SEEDS:
            data[(pair, s)] = {"W": build(pair, s)}
            W = data[(pair, s)]["W"]
            print(f"loaded {pair} s{s}: {len(W['eid'])} windows, REST {int(W['rest'].sum())}, STAY {int(W['stay'].sum())}", flush=True)
    print("CONTROLS", flush=True)
    controls = run_controls(data)
    cok = all(c["PASS"] for c in controls)
    print(f"controls: {'ALL PASS' if cok else 'FAILED'} ({time.time() - t0:.0f}s)", flush=True)
    R = {}
    for pair in PAIRS:
        R[pair] = {}
        for s in SEEDS:
            print(f"READ {pair} s{s}", flush=True)
            R[pair][f"s{s}"], _ = read_world(data[(pair, s)]["W"], a.nboot)
            r = R[pair][f"s{s}"]
            for arm in ARMS:
                print(f"   {arm}: Q1 median {r[arm]['Q1']['median_T_cl_m']['point']:.3f}  Q2 auroc {fr(r[arm]['Q2']['auroc_T_ol'], 3)}  Q3 ade6 rest {fr(r[arm]['Q3']['ruled']['ade6_rest'], 4, True)}", flush=True)
    O = {p: {"Q1": label_q1(R) if p == "pair1" else ("CREEP" if all(R[p][f"s{s}"]["H"]["Q1"]["median_T_cl_m"]["point"] > 2.0 for s in SEEDS) else "HOLDS"),
             "Q2": label_q2(R, p), "Q3": label_q3(R, p)} for p in PAIRS}
    fol = follows_text(O["pair1"]["Q3"], O["pair1"]["Q2"])
    print("OUTCOMES", json.dumps(O), flush=True)
    res = {"card": CARD, "card_sha256": sha, "holdread_sha256": me,
           "evidence": "MEASURED; tier T1-kin (kinematic rollout on the model's own actions, logged frames); zero training; the hold rule is an EMULATION on banked rollouts; never driving performance",
           "STRATEGIC": "PAUSED (PI item 25)", "nboot": a.nboot, "bootstrap_seed": H.BSEED, "rest_v0_m_s": REST_V0, "tol_m": TOL, "theta_grid_m": list(GRID), "n_folds": NFOLD,
           "dump_folders": {pair: {s: {arm: PAIRS[pair][arm].format(s=s) for arm in ARMS} for s in SEEDS} for pair in PAIRS},
           "n_windows_episodes": {f"{p}|s{s}": [int(len(data[(p, s)]["W"]["eid"])), int(len(np.unique(data[(p, s)]["W"]["eid"])))] for p in PAIRS for s in SEEDS},
           "controls": controls, "controls_all_pass": cok, "results": R, "outcomes": O, "follows": fol, "exit_code": 0 if cok else 2}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(a.out_dir + "/RESULT_HOLDREAD.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    write_verdict(res, a.out_dir + "/VERDICT_HOLDREAD.md")
    print(f"done {time.time() - t0:.0f}s", flush=True)
    return 0 if cok else 2


if __name__ == "__main__":
    sys.exit(main())
