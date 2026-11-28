"""q5_read.py -- AMENDMENT 7 (Q5), section 2.4 decision logic, the section 4 controls, and the paired bootstrap read.

Order of work (committed): (1) controls are evaluated FIRST and any miss REFUSES the read, (2) one `stat_cluster_bootstrap` call per sample over all its score arrays (as s0p_fit.read_all does),
(3) the outcome code.  The decision functions are pure, so the C10 tests exercise them without data.

Predicates (both arms of every difference come from the SAME model family f and the SAME sample s):
    R(g,f,s) : paired dAUC(Vg@416+E - Vg@256p+E) point >= +0.020 AND CI lower bound > 0   (the card's RES-ADDS bar)
    A(g,f,s) : paired dAUC(Vg@416+E - E)                          CI lower bound > 0
Outcomes: RES416-ADDS-40 | RES416-ADDS-160 | SAMPLES-DISAGREE | RES416-FLAT.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

RES_MIN = 0.020            # card RES-ADDS bar
HU_SCALE = 0.037           # printed beside A, never a gate
GRIDS = (40, 160)
FAMILIES = ("linear", "mlp")
SAMPLES = ("S1", "S2", "S3")
NB, BSEED = 2000, 0


# ------------------------------------------------------------------------------------------------------------ predicates
def pred_R(r):
    """r = {"point", "lo"} of the PAIRED dAUC(V416 - V256p)."""
    return bool(r["point"] >= RES_MIN and r["lo"] > 0)


def pred_A(a):
    """a = {"point", "lo"} of the PAIRED dAUC(V416 - E)."""
    return bool(a["lo"] > 0)


def pred_R_unpaired(r):
    """THE WRONG READING, kept only so the C10 test can show the production code does not use it: the probe's own CI lower bound against the comparator's point.
    `r` must carry probe_lo and comparator_point."""
    return bool(r["probe_lo"] - r["comparator_point"] >= RES_MIN and r["probe_lo"] > r["comparator_point"])


def holds(cell, use_A=True, reader_R=pred_R):
    """cell = {"R": ..., "A": ...} for ONE (g, f, s)."""
    return bool(reader_R(cell["R"]) and (pred_A(cell["A"]) if use_A else True))


def decide(inp, use_A=True, reader_R=pred_R):
    """inp[s][g][f] = {"R": {...}, "A": {...}} with s in S1, S2 (and S3 only when it was built), g in 40/160, f in linear/mlp.  -> (code, detail)."""
    have = [s for s in SAMPLES if s in inp]
    assert "S1" in have and "S2" in have, "S1 and S2 are both required"
    need = 2 if "S3" in have else None      # None: both S1 and S2

    def row_holds(g):
        hit = []
        for f in FAMILIES:
            flags = {s: holds(inp[s][g][f], use_A, reader_R) for s in have if g in inp[s] and f in inp[s][g]}   # same family for R and A, by construction
            ok = (sum(flags.values()) >= 2) if need == 2 else (len(flags) == 2 and all(flags.values()))
            if ok:
                hit.append(f)
        return hit

    h40, h160 = row_holds(40), row_holds(160)
    if h40:
        return "RES416-ADDS-40", {"families": h40}
    if h160:
        return "RES416-ADDS-160", {"families": h160}
    if need is None:
        for g in GRIDS:
            for f in FAMILIES:
                fl = {s: holds(inp[s][g][f], use_A, reader_R) for s in ("S1", "S2") if g in inp[s] and f in inp[s][g]}
                if sum(fl.values()) == 1:
                    return "SAMPLES-DISAGREE", {"g": g, "family": f, "holds_at": [s for s, v in fl.items() if v], "next": "build S3 (seed 20261116); there is no fourth sample"}
    return "RES416-FLAT", {}


# ------------------------------------------------------------------------------------------------------------ controls (evaluated first; any miss refuses the read)
def evaluate_controls(c):
    """c: measured values, keys as below (a missing key is a MISS).  -> (list[(id, ok, detail)], all_ok)."""
    out = []

    def chk(cid, ok, detail):
        out.append((cid, bool(ok), detail))

    g = c.get
    chk("C0", g("c0_differing_bytes") == 0, g("c0_differing_bytes"))
    chk("C1", g("c1_dauc") == 0.0 and g("c1_ci") == [0.0, 0.0] and g("c1_outcome") not in (None, "RES416-ADDS-40", "RES416-ADDS-160"), (g("c1_dauc"), g("c1_ci"), g("c1_outcome")))
    chk("C2", g("c2_auroc") is not None and g("c2_auroc") >= 0.99 and g("c2_outcome") == "RES416-ADDS-40", (g("c2_auroc"), g("c2_outcome")))
    chk("C3a", g("c3a_leak_auroc") is not None and g("c3a_leak_auroc") >= 0.999, g("c3a_leak_auroc"))
    chk("C3b", g("c3b_limit_max_abs_diff") == 0.0, g("c3b_limit_max_abs_diff"))
    chk("C3c", g("c3c_E_windows_changed") is not None and g("c3c_E_windows_changed") >= 1, g("c3c_E_windows_changed"))
    chk("C3d", g("c3d_shared_clips") == 0, g("c3d_shared_clips"))
    chk("C3e", g("c3e_E_y_clip_ws_eid_bitwise") is True, g("c3e_E_y_clip_ws_eid_bitwise"))
    chk("C4", g("c4_E_bitwise_equal_C0") is True, g("c4_E_bitwise_equal_C0"))
    chk("C5", g("c5_real_auroc") is not None and g("c5_null_max") is not None and g("c5_real_auroc") > g("c5_null_max"), (g("c5_real_auroc"), g("c5_null_max")))
    chk("C6a", g("c6a_max_abs_diff") is not None and g("c6a_max_abs_diff") <= 1, g("c6a_max_abs_diff"))      # AMENDMENT 7a: <= 1 count
    chk("C6b", g("c6b_mae_255") is not None and g("c6b_mae_255") <= 1.0, g("c6b_mae_255"))
    chk("C6c", g("c6c_bitwise") is True, g("c6c_bitwise"))
    chk("C7", g("c7_ok") is True, g("c7_ok"))
    chk("C8", g("c8_bitwise") is True, g("c8_bitwise"))
    chk("C9", g("c9_counts") == {"eval_n": 2365, "eval_pos": 131, "eval_episodes": 139, "train_n": 16000, "train_pos": 4000}, g("c9_counts"))
    return out, all(ok for _, ok, _ in out)


# ------------------------------------------------------------------------------------------------------------ the paired bootstrap, one call per sample
def score_names(families=FAMILIES):
    names, diffs = [], {}
    for f in families:
        names += [f"E|{f}"] + [f"V{g}@416+E|{f}" for g in GRIDS] + [f"V{g}@256p+E|{f}" for g in GRIDS]
        for g in GRIDS:
            diffs[f"R|{g}|{f}"] = (f"V{g}@416+E|{f}", f"V{g}@256p+E|{f}")
            diffs[f"A|{g}|{f}"] = (f"V{g}@416+E|{f}", f"E|{f}")
    return names, diffs


def bootstrap_sample(scores, yev, eid, ci_mod, n_boot=NB, seed=BSEED, extra_diffs=None):
    """scores: {name: eval score array}; one stat_cluster_bootstrap call over every array and every paired difference of this sample (the SAME resample for all).
    -> (B, cells) with cells[g][f] = {"R": {point, lo, hi}, "A": {...}}."""
    from_auc = _auroc
    yev, eid = np.asarray(yev, bool), np.asarray(eid, np.int64)
    _, diffs = score_names()
    diffs = {**diffs, **(extra_diffs or {})}
    diffs = {k: v for k, v in diffs.items() if v[0] in scores and v[1] in scores}

    def stat(sel):
        sel = np.asarray(sel)
        y = yev[sel]
        a = {k: from_auc(v[sel], y) for k, v in scores.items()}
        d = {f"D|{k}": a[x] - a[w] for k, (x, w) in diffs.items()}
        return {**{f"A|{k}": v for k, v in a.items()}, **d}

    B = ci_mod.stat_cluster_bootstrap(list(range(len(yev))), eid, stat, n_boot=n_boot, seed=seed)
    cells = {g: {} for g in GRIDS}
    for g in GRIDS:
        for f in FAMILIES:
            kr, ka = f"D|R|{g}|{f}", f"D|A|{g}|{f}"
            if kr in B and ka in B:
                cells[g][f] = {"R": dict(B[kr]), "A": dict(B[ka])}
    return B, cells


def _auroc(score, label):
    from scipy.stats import rankdata
    s, y = np.asarray(score, float), np.asarray(label, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s, method="average")
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


# ------------------------------------------------------------------------------------------------------------ C1 from the scores (identity)
def c1_identity(sc256, yev, eid, ci_mod, n_boot=200):
    """C1 deliberate regression: 'V40@416' (and V160) IS the 256p arrays.  -> (max |paired dAUC|, [min lo, max hi] over the R cells, outcome code).  Expected 0.0, [0.0, 0.0], not-ADDS."""
    sc = {k: v for k, v in sc256.items()}
    for f in FAMILIES:
        for g in GRIDS:
            sc[f"V{g}@416+E|{f}"] = sc[f"V{g}@256p+E|{f}"]
    _, cells = bootstrap_sample(sc, yev, eid, ci_mod, n_boot=n_boot)
    pts = [cells[g][f]["R"]["point"] for g in GRIDS for f in FAMILIES]
    los = [cells[g][f]["R"]["lo"] for g in GRIDS for f in FAMILIES]
    his = [cells[g][f]["R"]["hi"] for g in GRIDS for f in FAMILIES]
    code, _ = decide({"S1": cells, "S2": cells})
    return float(max(abs(p) for p in pts)), [float(min(los)), float(max(his))], code


# ------------------------------------------------------------------------------------------------------------ beside arms (PRINTED, never enter the outcome code)
HU_MIN = 0.037                     # AMENDMENT 2 rule, as s0p_fit.decide_outcomes
T_TAGS = ("BAL-NULL_s0", "BAL-NULL_s1")


def _npy(d, arm, f):
    p = os.path.join(d, f"fit_{arm}_{f}.npy") if d else None
    return p if p and os.path.exists(p) else None


def gather_beside(d, ev_for_T, families=FAMILIES, t_scores_fn=None):
    """d: one sample's config.  -> (extra scores {name: array}, extra diffs {key: (a, b)}, blocked {arm: reason}, groups {arm: [diff keys]}).
    A missing input is BLOCKED with its exact path; nothing is invented.  `t_scores_fn(ev)` -> {tag: scores}; default s0p_fit.t_scores."""
    sc, df, blocked, groups = {}, {}, {}, {}

    def add(group, key, a, b):
        df[key] = (a, b)
        groups.setdefault(group, []).append(key)

    w416, w256, w256b = d.get("work416"), d.get("work256p"), d.get("work256b")
    # 1. route term: V*@256b (box-2 tokens, S0-P's banked scores) against the pod's V*@256p
    miss = []
    for f in families:
        for g in GRIDS:
            p = _npy(w256b, f"V{g}+E", f)
            if p is None:
                miss.append(os.path.join(w256b or "<work256b unset>", f"fit_V{g}+E_{f}.npy"))
                continue
            sc[f"V{g}@256b+E|{f}"] = np.load(p)
            add("route", f"ROUTE|{g}|{f}", f"V{g}@256p+E|{f}", f"V{g}@256b+E|{f}")
    if miss:
        blocked["route V*@256b"] = "missing " + ", ".join(miss) + " (S1: S0-P's banked s0p/run scores; S2: C0's run on box 2 with S2's seed, then s0p_fit prep + fit)"
    # 2. M@416 vs M@256p
    for f in families:
        a, b = _npy(w416, "M+E", f), _npy(w256, "M+E", f)
        if a and b:
            sc[f"M@416+E|{f}"], sc[f"M@256p+E|{f}"] = np.load(a), np.load(b)
            add("M", f"M|{f}", f"M@416+E|{f}", f"M@256p+E|{f}")
        else:
            blocked["M@416 vs M@256p"] = f"missing fit_M+E_{f}.npy in {w416} and/or {w256} (s0p_fit.py fit --arm M+E)"
    # 3. P-B (full-field pool) vs P-A
    for f in families:
        p = _npy(d.get("work416pb"), "V40+E", f)
        if p:
            sc[f"V40@416pb+E|{f}"] = np.load(p)
            add("P-B", f"PB|{f}", f"V40@416pb+E|{f}", f"V40@416+E|{f}")
        else:
            blocked["P-B vs P-A"] = (f"missing {d.get('work416pb') or '<work416pb unset>'}/fit_V40+E_{f}.npy: the pod build (q5_frames_pod.py rows) stores only P-A pooled arrays, so no P-B v40 exists. "
                                     "Producing it needs a P-B variant of q5_now_encode.pool_pa (cell edges over the full field with the 0.2 patch rows at each end, SCOPE_416 section 4) and a re-encode "
                                     "of the 416 NOW frames for S1 and eval (~0.6 h pod GPU) into an S0-P-format dir, then s0p_fit prep + fit there")
    # 4. V40@416+(E-limit) vs E-limit
    for f in families:
        a, b = _npy(w416, "V40+E-limit", f), _npy(w416, "E-limit", f)
        if a and b:
            sc[f"V40@416+E-limit|{f}"], sc[f"E-limit|{f}"] = np.load(a), np.load(b)
            add("E-limit", f"ELIM|{f}", f"V40@416+E-limit|{f}", f"E-limit|{f}")
        else:
            blocked["V40@416+(E-limit) vs E-limit"] = f"missing fit_V40+E-limit_{f}.npy and/or fit_E-limit_{f}.npy in {w416} (q5_fit_extra.py for the first, s0p_fit.py fit --arm E-limit for the second)"
    # 5. V416tok+E, 13x32 native grid, linear only, vs V40@416+E
    p = _npy(d.get("work416tok"), "V416tok+E", "linear")
    if p:
        sc["V416tok+E|linear"] = np.load(p)
        add("V416tok", "TOK|linear", "V416tok+E|linear", "V40@416+E|linear")
    else:
        blocked["V416tok+E (13x32) vs V40@416+E"] = (f"missing {d.get('work416tok') or '<work416tok unset>'}/fit_V416tok+E_linear.npy: no 13x32 token dump exists (q5_frames_pod.py rows hold only v40 / v160 / mean). Needs the sibling "
                                                     "encoder's raw 416-token field kept for each NOW frame ([N,416,1024] fp8, ~6.8 GB per 16,000 windows) written as an S0-P-format dir with a 416-token "
                                                     "grid, plus an arm in a wrapper like q5_fit_extra.py (PCA per token via s0p_fit.fit_pca, linear only)")
    # 6. HEAD-UNDERUSES at 416: V40@416+E - T (Amendment 2 rule), T = BAL-NULL_s0 / _s1 goal_prob
    try:
        if t_scores_fn is None:
            F, _ = _s0p()
            t_scores_fn = F.t_scores
        for t, s in t_scores_fn(ev_for_T).items():
            sc[f"T|{t}"] = np.asarray(s, float)
            for f in families:
                add("HEAD-UNDERUSES", f"HU|{f}|{t}", f"V40@416+E|{f}", f"T|{t}")
    except Exception as e:                    # noqa: BLE001  (a missing STOPDEC npz or a window mismatch is a BLOCK, not a crash)
        blocked["HEAD-UNDERUSES"] = f"{type(e).__name__}: {str(e)[:200]} (needs stopdec_BAL-NULL_s0.npz and _s1 under s0p_fit.STOPDEC_OUT, aligned to the eval windows by (ci, ws))"
    return sc, df, blocked, groups


def _s0p():
    sys.path.insert(0, HERE)
    import q5_controls as C
    return C.s0p_modules()


def beside_report(B, groups, blocked):
    """Pure formatting of the bootstrap output B: every row is a paired difference on the SAME resample as the outcome inputs.  -> dict (printed; never read by decide)."""
    out = {"NOTE": "beside only: nothing here enters the outcome code", "arms": {}, "BLOCKED": dict(blocked)}
    for grp, keys in groups.items():
        rows = {}
        for k in keys:
            r = B.get(f"D|{k}")
            if r is not None:
                rows[k] = {"point": r["point"], "lo": r.get("lo"), "hi": r.get("hi")}
        out["arms"][grp] = rows
    hu = out["arms"].get("HEAD-UNDERUSES", {})
    for f in FAMILIES:
        per = [hu.get(f"HU|{f}|{t}") for t in T_TAGS]
        if all(per):
            out.setdefault("HEAD-UNDERUSES_rule", {})[f] = bool(all(p["point"] >= HU_MIN and p["lo"] > 0 for p in per))
    return out


# ------------------------------------------------------------------------------------------------------------ the read
def read(cfg_path):
    """cfg json: {"tree", "eval": <eval npz used for y / eid>, "eval416", "eval256p", "sidecar_limit_eval" (optional .npy), "out",
    "samples": {"S1": {"work416", "work256p", "train416", "train256p", optional "work256b", "work416pb", "work416tok"}, "S2": ...},
    "controls": {externally measured: c0, c4, c6a/b/c, c7, c8, c9, and c1/c2 if not computed here}}.
    Each work dir holds s0p_fit.py's fit_<arm>_<model>.npy eval scores.  C3 a/b/c/d/e and C5 (and C1 when absent) are computed here and OVERRIDE a same-named key in cfg["controls"]."""
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    sys.path.insert(0, cfg["tree"])
    sys.path.insert(0, cfg["tree"] + "/taniteval")
    from taniteval import ci as CI
    controls = dict(cfg.get("controls", {}))
    detail = {}
    ev = np.load(cfg["eval"], allow_pickle=False)
    if "eval416" in cfg:
        sys.path.insert(0, HERE)
        import q5_controls as C
        comp, detail = C.compute_fit_side(cfg, ci_mod=CI)
        controls.update(comp)
    s1 = next(iter(cfg["samples"].values()))
    if "c1_dauc" not in controls:
        sc0 = {}
        for f in FAMILIES:
            sc0[f"E|{f}"] = np.load(os.path.join(s1["work416"], f"fit_E_{f}.npy"))
            for g in GRIDS:
                sc0[f"V{g}@256p+E|{f}"] = np.load(os.path.join(s1["work256p"], f"fit_V{g}+E_{f}.npy"))
        d1, ci1, o1 = c1_identity(sc0, ev["y"], ev["eid"], CI)
        controls.update(c1_dauc=d1, c1_ci=ci1, c1_outcome=o1)
    ctl, ok = evaluate_controls(controls)
    res = {"controls": [{"id": i, "ok": o, "detail": d} for i, o, d in ctl], "controls_detail": detail}
    if not ok:
        res["outcome"] = "REFUSED: control miss " + ",".join(i for i, o, _ in ctl if not o)
        json.dump(res, open(cfg["out"], "w", encoding="utf-8"), indent=1, default=str)
        return res
    inp, boots, beside = {}, {}, {}
    for s, d in cfg["samples"].items():
        sc = {}
        for f in FAMILIES:
            sc[f"E|{f}"] = np.load(os.path.join(d["work416"], f"fit_E_{f}.npy"))
            for g in GRIDS:
                sc[f"V{g}@416+E|{f}"] = np.load(os.path.join(d["work416"], f"fit_V{g}+E_{f}.npy"))
                sc[f"V{g}@256p+E|{f}"] = np.load(os.path.join(d["work256p"], f"fit_V{g}+E_{f}.npy"))
        ev_T = None
        try:
            F, _ = _s0p()
            ev_T = F.load_any(cfg.get("eval416", cfg["eval"]))
        except Exception:                                   # noqa: BLE001
            pass
        xs, xd, blocked, groups = gather_beside(d, ev_T)
        sc.update(xs)
        B, cells = bootstrap_sample(sc, ev["y"], ev["eid"], CI, extra_diffs=xd)             # ONE call per sample: outcome inputs and beside rows share the resample
        inp[s] = cells
        boots[s] = {k: B[k] for k in B if k.startswith("D|")}
        beside[s] = beside_report(B, groups, blocked)
    code, detail_o = decide(inp)                            # beside rows are not an argument of decide(): they cannot move the code
    res.update(outcome=code, detail=detail_o, inputs=inp, bootstrap_diffs=boots, beside=beside,
               estimator=f"taniteval.ci.stat_cluster_bootstrap, B {NB}, seed {BSEED}, by episode; one call per sample ('another draw of episodes' only; S2 = another draw of the training sample)",
               families={"TACTICAL": "stop-point AUROC, n 2,365 / 131 / 139", "LONGITUDINAL": "n = 0", "LATERAL": "n = 0", "STRATEGIC": "PAUSED (PI item 25)"},
               tier="frozen-feature probe: a diagnostic of what the inputs carry, never driving performance", auroc_floor_scale=HU_SCALE)
    json.dump(res, open(cfg["out"], "w", encoding="utf-8"), indent=1, default=str)
    return res


if __name__ == "__main__":
    print(json.dumps(read(sys.argv[1]), indent=1, default=str)[:4000])
