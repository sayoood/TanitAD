"""imag_read.py: the DEV-SIDE READER of the tool `imaghead` (card PREREG_V8F_PROOF_ADD36, sections 3 to 5). Tier T0: a world-model diagnostic on logged frames,
never a driving number.   STRATEGIC: PAUSED (PI item 25).

  python imag_read.py read --dump DUMP_s0.npz DUMP_s1.npz --arm LS --seeds 0,1 --out-dir DIR [--floor-oof F_s0.npz F_s1.npz] [--B 2000]
  python imag_read.py pair --a-dir DIR_RL --b-dir DIR_LS --out-dir DIR_PAIR [--a-arm RL --b-arm LS]

read: H1, H1e, H1c, H1ec (calibrated statistics of note ADD18_N3 with the head reads in place of the probe reads: five folds by episode, linear fit of the label on the named reads
out of fold; fold = episode index of the floor file's `uep` modulo 5, exactly as pool_rb.py), the mean absolute errors in metres, the slope b of H2 (through the origin, windows
with label gap 5 to 50 m and ego speed >= 5 m/s at t) per seed and pooled, the per-offset mean responses, the presence response, the slope on gaps below 20 m, the names of
card section 4, the controls c1 to c5. Two-level bootstrap as pool_rb.pooled: seeds with replacement, then episodes inside each drawn seed, B 2,000, rng seed 0.
All controls must pass for an arm, else no name is printed.  Writes RESULT_IMAG_<ARM>.json, VERDICT_IMAG_<ARM>.md and SUFF_IMAG_<ARM>.npz (sufficient statistics for `pair`)."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

TIER = "T0 (a world-model diagnostic on logged frames, never driving performance)"
STRAT = "STRATEGIC: PAUSED (PI item 25)"
FAMILIES = "LONGITUDINAL, LATERAL, TACTICAL: not measured by this read"
OFFS_ACT = (-2.0, -1.0, 1.0)             # the offsets that enter the slope (d = 0 is the control c1)
T_NAMED, T_REPORTED = 2.0, 1.0
T_IDX = {2.0: 1, 1.0: 0}                 # axis 2 of roll_*: 0 = 1.0 s, 1 = 2.0 s
H2_GAP, H2_V = (5.0, 50.0), 5.0
CAL = {"c_ii": ("ii",), "c_i2": ("ii", "i"), "c_t2": ("ii", "iii"), "c_iif": ("ii", "fl"), "c_iifi": ("ii", "fl", "i"), "c_iift": ("ii", "fl", "iii")}
H1_DEF = {"H1": ("c_i2", "c_ii"), "H1e": ("c_iifi", "c_iif"), "H1c": ("c_t2", "c_ii"), "H1ec": ("c_iift", "c_iif")}   # statistic = R2 of the first minus R2 of the second
SK = ("i", "ii", "iii", "fl") + tuple(CAL)
OOF_DIRS = ("C:/Users/Admin/v8proof_c/ctxroll/rb", "F:/TanitAD-scratch/v8proof/spatial/q2/out")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ statistics
def r2_pooled(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return float(1.0 - ((y[m] - p[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum())


def r2_of(t):                                           # t [..., 4] = n, sum y, sum y^2, SSE
    sst = t[..., 2] - t[..., 1] ** 2 / np.maximum(t[..., 0], 1)
    return 1.0 - t[..., 3] / sst


def ep_stats_r2(epi, y, p, n_ep):
    m = np.isfinite(y) & np.isfinite(p)
    e, yy, pp = epi[m], y[m], p[m]
    return np.stack([np.bincount(e, minlength=n_ep).astype(np.float64), np.bincount(e, weights=yy, minlength=n_ep), np.bincount(e, weights=yy ** 2, minlength=n_ep),
                     np.bincount(e, weights=(yy - pp) ** 2, minlength=n_ep)], 1)


def calibrate(yk, epi, P, n_fold=5):
    """Out-of-fold linear fits of the label on the named reads (pool_rb.calibrate with the keys of this tool); fold = episode index modulo 5."""
    m = np.isfinite(yk) & np.isfinite(P["i"]) & np.isfinite(P["ii"]) & np.isfinite(P["iii"]) & np.isfinite(P["fl"])
    out = {k: np.full(len(yk), np.nan) for k in CAL}
    fold = epi % n_fold
    for f in range(n_fold):
        tr, te = m & (fold != f), m & (fold == f)
        for k, cols in CAL.items():
            X = np.column_stack([np.ones(int(tr.sum()))] + [P[c][tr] for c in cols])
            w = np.linalg.lstsq(X, yk[tr], rcond=None)[0]
            out[k][te] = np.column_stack([np.ones(int(te.sum()))] + [P[c][te] for c in cols]) @ w
    return out


def h1_seed(yk, epi, n_ep, P):
    """P: dict of reads i, ii, iii, fl over the windows (NaN where absent). -> per-seed record: R2 of every named read, H1 family points, sufficient stats, mean absolute errors."""
    P = dict(P)
    P.update(calibrate(yk, epi, P))
    m = np.isfinite(yk) & np.isfinite(P["i"]) & np.isfinite(P["ii"]) & np.isfinite(P["iii"]) & np.isfinite(P["fl"])
    S = {k: ep_stats_r2(epi, yk, P[k], n_ep) for k in SK}
    r2 = {k: r2_of(S[k].sum(0)) for k in SK}
    pts = {q: float(r2[x] - r2[y]) for q, (x, y) in H1_DEF.items()}
    mae = {k: float(np.abs(yk[m] - P[k][m]).mean()) if m.any() else float("nan") for k in ("i", "ii", "iii")}
    return {"S": S, "r2": r2, "pt": pts, "mae_m": mae, "n": int(m.sum())}


def slope_stats(r, e, epi, n_ep, wmask):
    """r, e [n, n_off]: response and kinematic expectation per window and offset. -> per-episode sums (Sre, See) over the windows of wmask (and finite)."""
    ok = wmask & np.isfinite(r).all(1) & np.isfinite(e).all(1)
    return np.bincount(epi[ok], weights=(r[ok] * e[ok]).sum(1), minlength=n_ep), np.bincount(epi[ok], weights=(e[ok] ** 2).sum(1), minlength=n_ep), int(ok.sum())


def boot_plan(K, n_ep, B, rng_seed=0):
    """The draw of pool_rb.pooled: seeds with replacement, then episodes inside each drawn seed (shared by every statistic and by both arms of a pair)."""
    rng = np.random.default_rng(rng_seed)
    pick = rng.integers(0, K, size=(B, K))
    C = rng.multinomial(n_ep, np.full(n_ep, 1.0 / n_ep), size=(B, K)).astype(np.float64)
    return pick, C


def pool_h1(recs, plan):
    pick, C = plan
    B, K = pick.shape
    St = {k: np.stack([r["S"][k] for r in recs]) for k in SK}
    out = {q: np.zeros(B) for q in H1_DEF}
    for j in range(K):
        k, c = pick[:, j], C[:, j, :]
        rr = {x: r2_of(np.einsum("be,bef->bf", c, St[x][k])) for x in SK}
        for q, (x, y) in H1_DEF.items():
            out[q] += rr[x] - rr[y]
    return {q: v / K for q, v in out.items()}


def pool_slope(sre, see, plan):
    """sre, see [K, n_ep]: the slope b = sum(r e) / sum(e^2) of every drawn seed, averaged over the K slots (as pooled r2)."""
    pick, C = plan
    B, K = pick.shape
    out = np.zeros(B)
    for j in range(K):
        k, c = pick[:, j], C[:, j, :]
        num, den = (c * sre[k]).sum(1), (c * see[k]).sum(1)
        out += num / np.where(den > 0, den, np.nan)
    return out / K


def ci(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def seed_slope_ci(sre, see, n_ep, B, rng_seed=0):
    """One seed: episode bootstrap of the slope (interval beside the pooled one)."""
    rng = np.random.default_rng(rng_seed)
    C = rng.multinomial(n_ep, np.full(n_ep, 1.0 / n_ep), size=B).astype(np.float64)
    den = C @ see
    return ci((C @ sre) / np.where(den > 0, den, np.nan))


# ------------------------------------------------------------------------------------------------ loading
def load_dump(path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["eid"] = d["eid"].astype(str)
    with open(path + ".json", encoding="utf-8") as fh:
        d["meta"] = json.load(fh)
    d["path"] = path.replace(os.sep, "/")
    return d


def find_floor(seed):
    for dd in OOF_DIRS:
        p = os.path.join(dd, "OOF_RB_G_s%d.npz" % seed)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no OOF_RB_G_s%d.npz (give --floor-oof)" % seed)


def load_floor(path):
    z = np.load(path, allow_pickle=True)
    fl_pref = None
    return {"eid": z["w_eid"].astype(str), "g0": z["w_g0"].astype(np.int64), "yk": z["yk"].astype(np.float64), "epi": z["w_epi"].astype(np.int64), "uep": [str(e) for e in z["uep"]],
            "iv": z["p_iv"].astype(np.float64), "ivb": z["p_ivb"].astype(np.float64), "path": path.replace(os.sep, "/"), "sha256": sha256_file(path)}


def h2_arrays(d, T):
    """r [n, 3], e [n, 3] for the offsets of OFFS_ACT at the read time T: r = head gap under the offset minus head gap under the logged actions [m]; e = -0.5 d T^2."""
    offs = [float(x) for x in d["offsets"]]
    ix = [offs.index(o) for o in OFFS_ACT]
    t = T_IDX[T]
    gl = d["roll_logged"][:, t, 1].astype(np.float64)
    r = np.stack([d["roll_off"][:, i, t, 1].astype(np.float64) - gl for i in ix], 1)
    e = np.stack([np.full(len(gl), -0.5 * OFFS_ACT[j] * T * T) for j in range(len(ix))], 1)
    return r, e


def h2_window_mask(d, lt20=False):
    g, v = d["gap_t"].astype(np.float64), d["v_t"].astype(np.float64)
    m = np.isfinite(g) & (g >= H2_GAP[0]) & (g <= H2_GAP[1]) & np.isfinite(v) & (v >= H2_V)
    return m & (g < 20.0) if lt20 else m


def seed_record(d, floor, n_ep, epi_of, label_perm=None, mutate=None):
    """Everything of one seed: the H1 record on the floor file's windows, the slope sufficient statistics per episode, per-offset means. `mutate` (tests only) = 'no_present_as_roll'."""
    key = {(e, int(g)): i for i, (e, g) in enumerate(zip(d["eid"], d["g0"]))}
    nF = len(floor["eid"])
    row = np.array([key.get((e, int(g)), -1) for e, g in zip(floor["eid"], floor["g0"])])
    have = row >= 0
    def pick(col, idx):
        out = np.full(nF, np.nan)
        out[have] = d[col][row[have]][..., idx].astype(np.float64) if col in ("present", "true2") else d[col][row[have], 1, idx].astype(np.float64)
        return out
    gap_t, gap_t2 = np.full(nF, np.nan), np.full(nF, np.nan)
    gap_t[have], gap_t2[have] = d["gap_t"][row[have]], d["gap_t2"][row[have]]
    yk = np.where(np.isfinite(gap_t) & np.isfinite(gap_t2), gap_t2, np.nan)         # lead at t and at t + 2.0 s
    if label_perm is not None:
        m = np.isfinite(yk)
        yk = yk.copy()
        yk[m] = yk[m][label_perm.permutation(int(m.sum()))]
    P = {"i": pick("roll_logged", 1), "ii": pick("present", 1), "iii": pick("true2", 1)}
    r2s = {k: r2_pooled(yk, floor[k]) for k in ("iv", "ivb")}             # the floor rule of pool_rb.load_seed: the probe with the larger R2 on the seed's own labels
    P["fl"] = floor["ivb"] if r2s["ivb"] > r2s["iv"] else floor["iv"]
    P["floor_name"] = "ivb" if r2s["ivb"] > r2s["iv"] else "iv"
    fname = P.pop("floor_name")
    return P, yk, fname, have


def build_seed(d, floor, n_ep, B_seed, rng_seed=0, label_perm=None, present_as_roll=False, future_as_roll=False, no_label_check=False):
    P, yk, fname, have = seed_record(d, floor, n_ep, None, label_perm)
    if present_as_roll:
        P["i"] = P["ii"].copy()
    if future_as_roll:
        P["i"] = P["iii"].copy()
    h1 = h1_seed(yk, floor["epi"], n_ep, P)
    h1["floor"] = fname
    h1["n_lead"] = int(np.isfinite(yk).sum())
    # label check against the banked label of the floor file (control c2c)
    both = np.isfinite(yk) & np.isfinite(floor["yk"])
    h1["label_vs_banked"] = {"n_both": int(both.sum()), "n_mine_only": int((np.isfinite(yk) & ~np.isfinite(floor["yk"])).sum()), "n_banked_only": int((~np.isfinite(yk) & np.isfinite(floor["yk"])).sum()),
                             "max_abs": float(np.abs(yk[both] - floor["yk"][both]).max()) if both.any() else float("nan")}
    # H2 on the dump's own windows (episode index = the floor's uep order; episodes outside it get a new index)
    uep = floor["uep"]
    idx = {e: i for i, e in enumerate(uep)}
    epi_d = np.array([idx.get(e, -1) for e in d["eid"]])
    h2 = {}
    for T in (T_NAMED, T_REPORTED):
        r, e = h2_arrays(d, T)
        for tag, lt in (("", False), ("_lt20", True)):
            wm = h2_window_mask(d, lt) & (epi_d >= 0)
            sre, see, nw = slope_stats(r, e, np.where(epi_d >= 0, epi_d, 0), n_ep, wm)
            h2["T%.1f%s" % (T, tag)] = {"sre": sre, "see": see, "n_windows": nw, "b": float(sre.sum() / see.sum()) if see.sum() > 0 else float("nan")}
    r2, _ = h2_arrays(d, T_NAMED)
    r1, _ = h2_arrays(d, T_REPORTED)
    wm = h2_window_mask(d)
    mean_resp = {}
    for T, rr in ((T_NAMED, r2), (T_REPORTED, r1)):
        mean_resp["T%.1f" % T] = {"%+.0f" % o: float(np.nanmean(rr[wm, j])) if wm.any() else float("nan") for j, o in enumerate(OFFS_ACT)}
        mean_resp["T%.1f" % T]["expected"] = {"%+.0f" % o: float(-0.5 * o * T * T) for o in OFFS_ACT}
    offs = [float(x) for x in d["offsets"]]
    pres = {}
    for T in (T_NAMED, T_REPORTED):
        t = T_IDX[T]
        pl = d["roll_logged"][:, t, 0].astype(np.float64)
        pres["T%.1f" % T] = {"%+.0f" % o: {"logit": float(np.nanmean((d["roll_off"][:, offs.index(o), t, 0] - pl)[wm])) if wm.any() else float("nan"),
                                              "prob": float(np.nanmean((1 / (1 + np.exp(-d["roll_off"][:, offs.index(o), t, 0].astype(np.float64))) - 1 / (1 + np.exp(-pl)))[wm])) if wm.any() else float("nan")}
                             for o in OFFS_ACT}
    return {"h1": h1, "h2": h2, "mean_response_m": mean_resp, "presence_response": pres, "n_windows_dumped": int(len(d["eid"])), "n_h2_windows": int(wm.sum()),
            "n_h2_windows_lt20": int(h2_window_mask(d, True).sum())}


# ------------------------------------------------------------------------------------------------ controls and names
def controls(dumps, floors, n_ep, B, plan):
    c = {}
    # c1: the offset 0 response is exactly 0.0 (same code path), every output
    m1 = [float(np.abs(d["roll_off"][:, [float(x) for x in d["offsets"]].index(0.0)] - d["roll_logged"]).max()) for d in dumps]
    c["c1"] = {"max_abs_response_offset0": m1, "bar": "exactly 0.0", "pass": bool(all(x == 0.0 for x in m1))}
    # c2: at dump time (two code paths of the same model), banked labels at read time
    at = [d["meta"]["controls_at_dump"]["c2"] for d in dumps]
    labs = [build_seed(d, f, n_ep, B)["h1"]["label_vs_banked"] for d, f in zip(dumps, floors)]
    c["c2"] = {"at_dump": at, "label_vs_banked": labs,
               "pass": bool(all(a["pass"] for a in at) and all(l["n_mine_only"] == 0 and l["n_banked_only"] == 0 and l["max_abs"] <= 1e-6 for l in labs)),
               "note": "rollout equality is measured at dump time against the trainer's roll_pairs on the same model; the banked-file comparison is 'banked' in at_dump only if --banked-roll was given"}
    # c4: the zero-action-input copy responds exactly 0.0 to every offset, and the real model responds (the control reads something)
    z = [float(max(np.abs(d["z_roll_off"][:, i] - d["z_roll_logged"]).max() for i in range(d["z_roll_off"].shape[1]))) for d in dumps]
    rr = [d["meta"]["controls_at_dump"]["c4"]["max_abs_response_real_model_nonzero_offsets"] for d in dumps]
    c["c4"] = {"max_abs_response_zero_action_copy": z, "max_abs_response_real_model": rr, "pass": bool(all(x == 0.0 for x in z))}
    # c3: planted known positive through the tool's own slope code (no model needed)
    c["c3"] = c3_planted()
    # c5: calibrated read
    c["c5"] = c5_calibrated(dumps, floors, n_ep, B)
    return c


def c3_planted():
    """A planted response that equals the kinematic expectation exactly, run through slope_stats / pool_slope: slope 1.0 to within 1e-6. The model-level version (planted predictor and head through compute_reads) is in test_imaghead.py."""
    rng = np.random.default_rng(5)
    n, n_ep = 600, 40
    epi = rng.integers(0, n_ep, n)
    r, e = np.zeros((n, 3)), np.zeros((n, 3))
    for j, o in enumerate(OFFS_ACT):
        e[:, j] = -0.5 * o * T_NAMED ** 2
        r[:, j] = e[:, j]
    sre, see, _ = slope_stats(r, e, epi, n_ep, np.ones(n, bool))
    b = float(sre.sum() / see.sum())
    pooled = pool_slope(np.stack([sre, sre]), np.stack([see, see]), boot_plan(2, n_ep, 50))
    return {"slope": b, "pooled_draw_max_abs_dev": float(np.nanmax(np.abs(pooled - 1.0))), "pass": bool(abs(b - 1.0) <= 1e-6 and np.nanmax(np.abs(pooled - 1.0)) <= 1e-6)}


def c5_calibrated(dumps, floors, n_ep, B, mutate=None):
    """labels permuted inside each seed: H1 and H1e interval not above 0; present read given as the rollout: 0 to 1e-9; true future given as the rollout: H1 above 0.
    mutate (tests): 'identity_perm' leaves the labels unpermuted; 'no_present_as_roll' skips the substitution."""
    K = len(dumps)
    plan = boot_plan(K, n_ep, B)
    rng = np.random.default_rng(0)
    out = {}
    perm = [None if mutate == "identity_perm" else np.random.default_rng(100 + s) for s in range(K)]
    rec = [build_seed(d, f, n_ep, B, label_perm=p)["h1"] for d, f, p in zip(dumps, floors, perm)]
    pp = pool_h1(rec, plan)
    out["permuted"] = {q: {"point": float(np.mean([r["pt"][q] for r in rec])), "ci": ci(pp[q])} for q in ("H1", "H1e")}
    ok_perm = all(out["permuted"][q]["ci"][0] <= 0.0 for q in ("H1", "H1e"))
    rec = [build_seed(d, f, n_ep, B, present_as_roll=(mutate != "no_present_as_roll"))["h1"] for d, f in zip(dumps, floors)]
    out["present_as_rollout"] = {q: [float(r["pt"][q]) for r in rec] for q in ("H1", "H1e")}
    ok_pres = all(abs(v) <= 1e-9 for q in ("H1", "H1e") for v in out["present_as_rollout"][q])
    rec = [build_seed(d, f, n_ep, B, future_as_roll=True)["h1"] for d, f in zip(dumps, floors)]
    out["true_future_as_rollout"] = {"H1": [float(r["pt"]["H1"]) for r in rec]}
    ok_fut = all(v > 1e-9 for v in out["true_future_as_rollout"]["H1"])      # "above 0": a numerical zero (iii equal to ii) is not above 0
    out["pass"] = bool(ok_perm and ok_pres and ok_fut)
    out["parts"] = {"permuted_not_above_0": bool(ok_perm), "present_as_rollout_is_0": bool(ok_pres), "true_future_as_rollout_above_0": bool(ok_fut)}
    return out


def name_h1(pts_by_seed, ci_h1, ci_h1e):
    h1 = all(p["H1"] > 0 for p in pts_by_seed) and ci_h1[0] > 0
    if not h1:
        return "IMAG-HEAD-FLAT"
    e = all(p["H1e"] > 0 for p in pts_by_seed) and ci_h1e[0] > 0
    return "IMAG-HEAD-ADDS-LEAD" if e else "IMAG-HEAD-ADDS-EGO-ONLY"       # IMAG-HEAD-ADDS holds with either suffix


def name_h2(b_by_seed, ci_b):
    if all(b >= 0.5 for b in b_by_seed) and ci_b[0] > 0:
        name = "IMAG-RESPONDS"
    elif all(b > 0 for b in b_by_seed) and ci_b[0] > 0:
        name = "IMAG-RESPONDS-WEAKLY"
    else:
        name = "IMAG-DEAF"
    over = [i for i, b in enumerate(b_by_seed) if b > 1.5]
    return name, over


def name_pair(h1a, h1b, ci_dh1, ba, bb, ci_db):
    names = []
    if all(x - y > 0 for x, y in zip(h1a, h1b)) and ci_dh1[0] > 0:
        names.append("RL-IMAGINES-BETTER")
    if all(abs(x - 1) < abs(y - 1) for x, y in zip(ba, bb)) and ci_db[1] < 0:
        names.append("RL-RESPONDS-BETTER")
    return names


# ------------------------------------------------------------------------------------------------ commands
def cmd_read(a):
    seeds = [int(s) for s in a.seeds.split(",")]
    if len(a.dump) != len(seeds):
        raise SystemExit("REFUSED: one dump per seed")
    dumps = [load_dump(p) for p in a.dump]
    floor_paths = a.floor_oof or [find_floor(s) for s in seeds]
    floors = [load_floor(p) for p in floor_paths]
    n_ep = len(floors[0]["uep"])
    if any(f["uep"] != floors[0]["uep"] for f in floors):
        raise SystemExit("REFUSED: the floor files list different episodes")
    ck = {d["meta"]["ckpt_sha256"] for d in dumps}
    if len(ck) != len(dumps):
        raise SystemExit("REFUSED: two seeds with the same checkpoint sha256")
    plan = boot_plan(len(dumps), n_ep, a.B)
    recs = [build_seed(d, f, n_ep, a.B) for d, f in zip(dumps, floors)]
    ctrl = controls(dumps, floors, n_ep, a.B, plan)
    all_pass = all(v["pass"] for v in ctrl.values())
    h1 = [r["h1"] for r in recs]
    pool = pool_h1(h1, plan)
    res = {"arm": a.arm, "seeds": seeds, "tier": TIER, "strategic": STRAT, "families": FAMILIES, "card": "PREREG_V8F_PROOF_ADD36", "B": a.B, "estimator": "two-level bootstrap (seeds with replacement, then episodes), rng seed 0",
           "n_episodes": n_ep, "controls": ctrl, "controls_pass": bool(all_pass), "dumps": [{"path": d["path"], "sha256": sha256_file(d["path"]), "ckpt_sha256": d["meta"]["ckpt_sha256"], "n": d["meta"]["n_windows_dumped"]} for d in dumps],
           "floors": [{"path": f["path"], "sha256": f["sha256"], "chosen": r["h1"]["floor"]} for f, r in zip(floors, recs)], "H1": {}, "H2": {}}
    for q in H1_DEF:
        res["H1"][q] = {"per_seed": [r["pt"][q] for r in h1], "pooled": float(np.mean([r["pt"][q] for r in h1])), "ci": ci(pool[q])}
    res["H1"]["n_windows_with_lead_t_and_t2"] = [r["n"] for r in h1]
    res["H1"]["r2"] = [r["r2"] for r in h1]
    res["H1"]["mae_m"] = {"per_seed": [r["mae_m"] for r in h1], "mean": {k: float(np.mean([r["mae_m"][k] for r in h1])) for k in ("i", "ii", "iii")},
                           "note": "i = head on rolled-out states (logged actions), ii = head on the present states, iii = head on the true future states; label = lead gap at t + 2.0 s; no calibration"}
    res["H2"]["windows"] = {"per_seed": [r["n_h2_windows"] for r in recs], "gap_lt_20": [r["n_h2_windows_lt20"] for r in recs], "rule": "label gap 5 to 50 m, ego speed >= 5 m/s at t"}
    for key in ("T2.0", "T1.0", "T2.0_lt20", "T1.0_lt20"):
        sre = np.stack([r["h2"][key]["sre"] for r in recs])
        see = np.stack([r["h2"][key]["see"] for r in recs])
        pb = pool_slope(sre, see, plan)
        bs = [r["h2"][key]["b"] for r in recs]
        res["H2"]["b_" + key] = {"per_seed": bs, "per_seed_ci_episodes": [seed_slope_ci(sre[k], see[k], n_ep, a.B) for k in range(len(recs))], "pooled": float(np.mean(bs)), "ci_two_level": ci(pb),
                                 "n_windows": [r["h2"][key]["n_windows"] for r in recs]}
    res["H2"]["mean_response_m"] = [r["mean_response_m"] for r in recs]
    res["H2"]["presence_response"] = [r["presence_response"] for r in recs]
    names = {}
    if all_pass:
        n1 = name_h1([r["pt"] for r in h1], res["H1"]["H1"]["ci"], res["H1"]["H1e"]["ci"])
        names["H1"] = ("IMAG-HEAD-ADDS + " + n1) if n1 != "IMAG-HEAD-FLAT" else n1
        n2, over = name_h2(res["H2"]["b_T2.0"]["per_seed"], res["H2"]["b_T2.0"]["ci_two_level"])
        names["H2"] = n2
        names["H2_overshoot_seeds"] = over
    res["names"] = names if all_pass else "NOT GIVEN: a control failed"
    np.savez(os.path.join(a.out_dir, "SUFF_IMAG_%s.npz" % a.arm), **{"S_%s_%d" % (k, s): recs[s]["h1"]["S"][k] for s in range(len(recs)) for k in SK},
             **{"sre_%s_%d" % (k, s): recs[s]["h2"][k]["sre"] for s in range(len(recs)) for k in recs[s]["h2"]}, **{"see_%s_%d" % (k, s): recs[s]["h2"][k]["see"] for s in range(len(recs)) for k in recs[s]["h2"]},
             n_ep=np.int64(n_ep), K=np.int64(len(recs)), B=np.int64(a.B))
    with open(os.path.join(a.out_dir, "RESULT_IMAG_%s.json" % a.arm), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(a.out_dir, "VERDICT_IMAG_%s.md" % a.arm), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(verdict(res))
    print(("ZZ IMAG DONE %s controls PASS names %s" % (a.arm, res["names"])) if all_pass else ("ZZ IMAG CONTROL-FAILED %s %s" % (a.arm, [k for k, v in ctrl.items() if not v["pass"]])))
    return 0 if all_pass else 2


def verdict(r):
    L = ["# VERDICT_IMAG_%s (card PREREG_V8F_PROOF_ADD36)" % r["arm"], "", "Tier: %s. %s. %s." % (TIER, STRAT, r["families"]), "",
         "## Controls (first)", ""]
    for k, v in r["controls"].items():
        L.append("* %s: %s" % (k, "PASS" if v["pass"] else "FAIL"))
    L.append("")
    if not r["controls_pass"]:
        L += ["**A control failed: no name is given for this arm.**", ""]
    L += ["## Names", "", "%s" % (json.dumps(r["names"]) if r["controls_pass"] else "not given"), "", "## H1 (head on imagined states; label = lead gap at t + 2.0 s; seeds %s)" % r["seeds"], ""]
    for q in H1_DEF:
        v = r["H1"][q]
        L.append("* %s: per seed %s, pooled %+.4f, two-level 95%% interval [%+.4f, %+.4f]" % (q, ["%+.4f" % x for x in v["per_seed"]], v["pooled"], v["ci"][0], v["ci"][1]))
    L.append("* mean absolute error [m]: rollout (i) %.3f, present (ii) %.3f, true future (iii) %.3f (windows %s)" % (r["H1"]["mae_m"]["mean"]["i"], r["H1"]["mae_m"]["mean"]["ii"], r["H1"]["mae_m"]["mean"]["iii"], r["H1"]["n_windows_with_lead_t_and_t2"]))
    L += ["", "## H2 (response of the imagined gap to the ego's offset acceleration; windows %s)" % r["H2"]["windows"]["per_seed"], ""]
    for key in ("T2.0", "T1.0", "T2.0_lt20", "T1.0_lt20"):
        v = r["H2"]["b_" + key]
        L.append("* b %s%s: per seed %s, pooled %.3f, two-level interval [%.3f, %.3f]" % (key, " (named)" if key == "T2.0" else "", ["%.3f" % x for x in v["per_seed"]], v["pooled"], v["ci_two_level"][0], v["ci_two_level"][1]))
    L += ["", "Limits: two seeds, one training run per seed; the offsets give action sequences the predictor has rarely seen; the kinematic expectation assumes a lead that keeps its motion; a response in imagination is a property of the world model, not a driving result.", ""]
    return "\n".join(L)


def cmd_pair(a):
    A, Bq = np.load(os.path.join(a.a_dir, "SUFF_IMAG_%s.npz" % a.a_arm)), np.load(os.path.join(a.b_dir, "SUFF_IMAG_%s.npz" % a.b_arm))
    ra = json.load(open(os.path.join(a.a_dir, "RESULT_IMAG_%s.json" % a.a_arm), encoding="utf-8"))
    rb = json.load(open(os.path.join(a.b_dir, "RESULT_IMAG_%s.json" % a.b_arm), encoding="utf-8"))
    if not (ra["controls_pass"] and rb["controls_pass"]):
        raise SystemExit("REFUSED: a control failed for an arm of the pair")
    K, n_ep, B = int(A["K"]), int(A["n_ep"]), int(A["B"])
    if (K, n_ep, B) != (int(Bq["K"]), int(Bq["n_ep"]), int(Bq["B"])):
        raise SystemExit("REFUSED: the two arms were read with different K, episodes or B")
    plan = boot_plan(K, n_ep, B)                                # the SAME draw for both arms: paired on the same windows
    def recs(Z):
        return [{"S": {k: Z["S_%s_%d" % (k, s)] for k in SK}} for s in range(K)]
    pa, pb = pool_h1(recs(A), plan), pool_h1(recs(Bq), plan)
    r2 = lambda Z, s, k: r2_of(Z["S_%s_%d" % (k, s)].sum(0))   # noqa: E731
    h1a = [float(r2(A, s, "c_i2") - r2(A, s, "c_ii")) for s in range(K)]
    h1b = [float(r2(Bq, s, "c_i2") - r2(Bq, s, "c_ii")) for s in range(K)]
    d_h1 = pa["H1"] - pb["H1"]
    def slopes(Z):
        sre = np.stack([Z["sre_T2.0_%d" % s] for s in range(K)])
        see = np.stack([Z["see_T2.0_%d" % s] for s in range(K)])
        return sre, see
    (ra_, ea_), (rb_, eb_) = slopes(A), slopes(Bq)
    pick, C = plan
    dist = np.zeros(B)
    for j in range(K):
        k, c = pick[:, j], C[:, j, :]
        ba = (c * ra_[k]).sum(1) / (c * ea_[k]).sum(1)
        bb = (c * rb_[k]).sum(1) / (c * eb_[k]).sum(1)
        dist += np.abs(ba - 1) - np.abs(bb - 1)
    dist /= K
    bsa = [float(ra_[s].sum() / ea_[s].sum()) for s in range(K)]
    bsb = [float(rb_[s].sum() / eb_[s].sum()) for s in range(K)]
    res = {"a": a.a_arm, "b": a.b_arm, "tier": TIER, "strategic": STRAT, "H1_a": h1a, "H1_b": h1b, "H1_a_minus_b": {"pooled": float(np.mean(h1a) - np.mean(h1b)), "ci": ci(d_h1)},
           "b_a": bsa, "b_b": bsb, "abs_dist_from_1_a_minus_b": {"pooled": float(np.mean([abs(x - 1) for x in bsa]) - np.mean([abs(x - 1) for x in bsb])), "ci": ci(dist)},
           "names": name_pair(h1a, h1b, ci(d_h1), bsa, bsb, ci(dist)) or "nothing is named for the pair"}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "PAIR_IMAG_%s_minus_%s.json" % (a.a_arm, a.b_arm)), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["names"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("read")
    r.add_argument("--dump", nargs="+", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--seeds", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--floor-oof", nargs="+", default=None)
    r.add_argument("--B", type=int, default=2000)
    p = sp.add_parser("pair")
    p.add_argument("--a-dir", required=True)
    p.add_argument("--b-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--a-arm", default="RL")
    p.add_argument("--b-arm", default="LS")
    a = ap.parse_args(argv)
    if a.cmd == "read":
        os.makedirs(a.out_dir, exist_ok=True)
        return cmd_read(a)
    return cmd_pair(a)


if __name__ == "__main__":
    sys.exit(main())
