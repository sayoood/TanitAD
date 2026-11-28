#!/usr/bin/env python3
"""native40_score.py -- NATIVE40 scorer (PREREG_NATIVE40.md + the MM's map control). Reads dump_s0/s1.npz (native40_dump.py) and LS_REF40_WINDOWS.json; writes NATIVE40_RESULT.json / .md.
Every number: MEASURED (ours), tier T0 (teacher-forced head read on logged frames, no rollout), estimator = episode-cluster bootstrap B=2000 seed 0 (answers another draw of EPISODES only).
MAP CONTROL (binding, runs before any head number): the coordinator's window (eid, g0, t, gap_t) is mapped to OUR label by the source-derived rule
  our window = eval-grid window (ci of eid, cache step t0 = t)  [eval_grid_windows: (ci, t, 0, 0), t = the grid's cache step; v8f_mini_data.py:622-626; the head reads the frame at cache step t0 = ds.item(ci, j=t0), v8f_t1_eval_ri.build_item]
  label raw frame k = 2 * t0  [v8f_mini_data.lead_label_arrays: kj = 2*arange(n), line 295]
and our label gap there must equal gap_t within 1e-3 m. HIGH_SHARE = 0.95 (fixed here, before any head number): below it the read is NOT-COMPUTABLE; otherwise only matched windows are scored, n reported.
usage: native40_score.py --d0 dump_s0.npz --d1 dump_s1.npz --windows LS_REF40_WINDOWS.json --out-dir DIR"""
import argparse
import hashlib
import json
import sys

import numpy as np

B_BOOT, SEED = 2000, 0
BAR_MAE, BAR_AUROC, HIGH_SHARE, MAP_TOL = 5.80, 0.85, 0.95, 1e-3
REF_MAE = 5.04


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def auroc(score, pos, w=None):
    """weighted AUROC (ties 0.5). score [n], pos [n] bool, w [n] or [B, n] weights. Returns scalar or [B]."""
    score = np.asarray(score, np.float64)
    order = np.argsort(score, kind="stable")
    s, p = score[order], np.asarray(pos)[order]
    n = len(s)
    W = np.ones((1, n)) if w is None else np.atleast_2d(np.asarray(w, np.float64))[:, order]
    wp, wn = W * p, W * (~p)
    grp_start = np.r_[True, s[1:] != s[:-1]]
    gid = np.cumsum(grp_start) - 1
    G = gid[-1] + 1
    gn = np.zeros((W.shape[0], G))
    gp = np.zeros((W.shape[0], G))
    for b in range(W.shape[0]):
        gn[b] = np.bincount(gid, weights=wn[b], minlength=G)
        gp[b] = np.bincount(gid, weights=wp[b], minlength=G)
    below = np.cumsum(gn, 1) - gn
    num = (gp * (below + 0.5 * gn)).sum(1)
    den = gp.sum(1) * gn.sum(1)
    r = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return r[0] if w is None else r


def ci(vals):
    v = np.asarray(vals, np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def draw_counts(E, B=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    M = np.zeros((B, E))
    for b in range(B):
        M[b] = np.bincount(rng.integers(0, E, E), minlength=E)
    return M


def gap_stats(pred, y, epi, M):
    """MAE, R2 over windows; bootstrap by episode weights M[B, E] (epi: window -> episode index)."""
    err = np.abs(pred - y)
    def pt(w):
        W = w.sum(-1)
        mae = (w * err).sum(-1) / W
        mu = (w * y).sum(-1) / W
        sst = (w * (y - mu[..., None] if np.ndim(mu) else y - mu) ** 2).sum(-1)
        sse = (w * (pred - y) ** 2).sum(-1)
        return mae, 1 - sse / sst
    mae, r2 = pt(np.ones(len(y)))
    Wb = M[:, epi]                                   # [B, n]
    mb, rb = pt(Wb)
    return {"mae_m": float(mae), "mae_ci95": ci(mb), "r2": float(r2), "r2_ci95": ci(rb), "n": int(len(y))}, mb


def amendment_map(WJ, D, res):
    """LABEL-ONLY diagnostic + an INFORMATIONAL gap read, both OUTSIDE the card: the coordinator's gap_t equals our label at raw frame k = 2*g0 (the window START, cache step g0), not at the read frame t = g0 + 11.
    If the dumps carry the head read at cache step g0 (x_* arrays), the gap numbers at those frames are reported here, marked as needing an MM amendment. Not a label of the card."""
    d = D[0]
    out = {"k2g0_share": None, "note": "the card's read is NOT-COMPUTABLE under its own map; this block exists so the MM can decide on an amendment in one call"}
    if "x_t0" not in d.files:
        out["status"] = "dumps carry no head read at cache step g0"
        return out
    pos = {(str(e), int(t)): i for i, (e, t) in enumerate(zip(d["x_eid"], d["x_t0"]))}
    sel, ok = [], []
    for e, g0, gt in zip(WJ["eid"], WJ["g0"], WJ["gap_t"]):
        j = pos.get((e, int(g0)))
        good = j is not None and bool(d["x_has"][j]) and d["x_lead"][j] == 1 and abs(float(d["x_gap"][j]) - float(gt)) <= MAP_TOL
        ok.append(good)
        if good:
            sel.append(j)
    out["k2g0_share"] = float(np.mean(ok))
    sel = np.array(sel)
    ue = np.unique(d["eid"])
    epmap = {e: i for i, e in enumerate(ue)}
    M = draw_counts(len(ue))
    epi = np.array([epmap[e] for e in d["x_eid"][sel]])
    rng = np.random.default_rng(SEED)
    pg = rng.permutation(len(sel))
    out["n_matched"] = int(len(sel))
    out["seeds"] = {}
    for s in (0, 1):
        z = D[s]
        y = z["x_gap"][sel].astype(np.float64)
        raw = z["x_lead_gap_hat"][sel].astype(np.float64)
        pred = np.clip(raw * 40.0, 0, 40)
        g, _ = gap_stats(pred, y, epi, M)
        sh = gap_stats(pred[pg], y, epi, M)[0]
        un = gap_stats(np.clip(raw * 100.0, 0, 40), y, epi, M)[0]
        lab = gap_stats(y, y, epi, M)[0]
        out["seeds"][f"s{s}"] = {"gap": g, "shuffle_mae_m": sh["mae_m"], "shuffle_r2": sh["r2"], "units_x100_mae_m": un["mae_m"], "label_arm_mae_m": lab["mae_m"], "label_arm_r2": lab["r2"],
                                 "n_frame_not_ok": int((~z["x_frame_ok"][sel]).sum())}
    out["mean_gap_mae_m"] = float(np.mean([out["seeds"][f"s{s}"]["gap"]["mae_m"] for s in (0, 1)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", required=True)
    ap.add_argument("--d1", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--read-at", choices=("t", "g0"), default="g0", help="AMENDMENT_1 (MM, before any head number): the coordinator's windows are read at cache step g0 (raw frame 2*g0); 't' is the frozen card's first map")
    a = ap.parse_args()
    WJ = json.load(open(a.windows, encoding="utf-8"))
    wsha = sha256(a.windows)
    res = {"read": "NATIVE40", "evidence_class": "MEASURED", "tier": "T0", "estimator": "episode-cluster bootstrap B=2000 seed 0 (another draw of EPISODES only)",
           "windows_file": a.windows, "windows_sha256": wsha, "bars": {"mean_gap_mae_m_max": BAR_MAE, "auroc_min_each_seed": BAR_AUROC, "reference_mae_m_INHERITED": REF_MAE}, "seeds": {}, "controls": {}}
    D = {s: np.load(p, allow_pickle=True) for s, p in ((0, a.d0), (1, a.d1))}
    res["dump_meta"] = {f"s{s}": json.loads(str(D[s]["meta"])) for s in D}
    for s in D:
        res["dump_meta"][f"s{s}"]["dump_sha256"] = sha256(a.d0 if s == 0 else a.d1)
    # ---- MAP CONTROL (before any head number)
    d = D[0]
    assert np.array_equal(d["eid"], D[1]["eid"]) and np.array_equal(d["t0"], D[1]["t0"]), "the two dumps disagree on windows"
    pos = {(str(e), int(t)): i for i, (e, t) in enumerate(zip(d["eid"], d["t0"]))}
    n_w = len(WJ["eid"])
    midx, mok = np.full(n_w, -1), np.zeros(n_w, bool)
    in_grid = 0
    if a.read_at == "g0":
        assert "x_t0" in d.files, "AMENDMENT_1 needs the head read at cache step g0 (x_* arrays) in the dumps"
        P = "x_"
        pos = {(str(e), int(t)): i for i, (e, t) in enumerate(zip(d["x_eid"], d["x_t0"]))}
        key_t = WJ["g0"]
    else:
        P = ""
        key_t = WJ["t"]
    HAS, LEAD, GAP, HAT, FOK, EID = (P + "has", P + "lead", P + "gap", P + "lead_gap_hat", ("x_frame_ok" if P else "frame_ok_now"), (P + "eid" if P else "eid"))
    for i, (e, t, gt) in enumerate(zip(WJ["eid"], key_t, WJ["gap_t"])):
        j = pos.get((e, int(round(t))))
        if j is None:
            continue
        in_grid += 1
        midx[i] = j
        mok[i] = bool(d[HAS][j]) and bool(d[LEAD][j] == 1) and abs(float(d[GAP][j]) - float(gt)) <= MAP_TOL
    share = float(mok.mean())
    eps_ours, eps_theirs = set(d["eid"].tolist()), set(WJ["eid"])
    res["map_control"] = {"rule": ("AMENDMENT_1: coordinator window read at (eid, cache step g0); our label raw frame k = 2*g0; head read at cache step g0 (x_* arrays); |our gap - gap_t| <= 1e-3 m" if a.read_at == "g0" else "our window (ci of eid, cache step t0 = t); label raw frame k = 2*t0; |our gap - gap_t| <= 1e-3 m"), "read_at": a.read_at,
                          "source": "eval_grid_windows v8f_mini_data.py:622-626 (cache step t); lead_label_arrays kj = 2*arange v8f_mini_data.py:295; ds.item(ci, j=t0) in v8f_t1_eval_ri.build_item",
                          "n_coordinator_windows": n_w, "n_found_in_our_grid": in_grid, "n_matched": int(mok.sum()), "match_share": share, "high_share_threshold": HIGH_SHARE,
                          "coordinator_episodes": len(eps_theirs), "coordinator_episodes_in_our_141": len(eps_theirs & eps_ours), "our_episodes": len(eps_ours)}
    out_json, md = f"{a.out_dir}/NATIVE40_RESULT.json", f"{a.out_dir}/NATIVE40_RESULT.md"
    if share < HIGH_SHARE:
        res["label"] = "NOT-COMPUTABLE"
        res["reason"] = f"the source-derived window map matches {share:.4f} of the coordinator's windows (< {HIGH_SHARE}); no head number of the card's read is computed"
        res["map_diagnostic_labels_only"] = amendment_map(WJ, D, res)
        json.dump(res, open(out_json, "w", encoding="utf-8", newline="\n"), indent=1)
        open(md, "w", encoding="utf-8", newline="\n").write(f"# NATIVE40 result\n\nLabel: **NOT-COMPUTABLE**. {res['reason']}.\n\nMEASURED, T0. Map control: {json.dumps(res['map_control'])}\n\nDiagnostic (outside the card; it decides nothing until the MM amends the map): {json.dumps(res['map_diagnostic_labels_only'])}\n")
        print("NOT-COMPUTABLE", share)
        return 0
    gi = midx[mok]                                                     # indices into our window arrays, matched gap windows
    ueps = np.unique(d["eid"])
    ep_of = {e: i for i, e in enumerate(ueps)}
    epi_all = np.array([ep_of[e] for e in d["eid"]])
    M = draw_counts(len(ueps))
    rng = np.random.default_rng(SEED)
    perm_g = rng.permutation(len(gi))
    # presence set: every window with a label
    hm = d["has"].astype(bool)
    pos_m = d["near"].astype(bool)
    pres_idx = np.where(hm)[0]
    n_pos, n_neg = int(pos_m[pres_idx].sum()), int((~pos_m[pres_idx]).sum())
    res["presence_set"] = {"n_windows_labelled": int(len(pres_idx)), "n_positive": n_pos, "n_negative": n_neg, "n_episodes": int(len(np.unique(d["eid"][pres_idx]))),
                           "n_frame_not_ok_in_set": int((~d["frame_ok_now"][pres_idx]).sum())}
    epi_g = np.array([ep_of[e] for e in d[EID][gi]])
    res["gap_set"] = {"n_windows": int(len(gi)), "n_episodes": int(len(np.unique(d[EID][gi]))), "n_frame_not_ok": int((~d[FOK][gi]).sum())}
    mae_boot = {}
    ctrl_fail = []
    for s in (0, 1):
        z = D[s]
        y = z[GAP][gi].astype(np.float64)
        raw = z[HAT][gi].astype(np.float64)
        pred = np.clip(raw * 40.0, 0.0, 40.0)
        g, mb = gap_stats(pred, y, epi_g, M)
        mae_boot[s] = mb
        sc, pp = z["lead_logit"][pres_idx].astype(np.float64), pos_m[pres_idx]
        epi_p = epi_all[pres_idx]
        au = float(auroc(sc, pp))
        aub = auroc(sc, pp, M[:, epi_p])
        # controls
        sh = gap_stats(pred[perm_g], y, epi_g, M)[0]
        au_sh = float(auroc(sc[rng.permutation(len(sc))], pp))
        un = gap_stats(np.clip(raw * 100.0, 0.0, 40.0), y, epi_g, M)[0]
        un_noclip = float(np.abs(raw * 100.0 - y).mean())
        lab = gap_stats(y, y, epi_g, M)[0]
        c = {"shuffle": {"mae_m": sh["mae_m"], "r2": sh["r2"], "mae_exceeds_unshuffled": sh["mae_m"] > g["mae_m"], "r2_le_0": sh["r2"] <= 0, "presence_auroc_shuffled": au_sh,
                         "auroc_in_0.45_0.55": 0.45 <= au_sh <= 0.55},
             "units_x100": {"mae_m_clipped": un["mae_m"], "mae_m_unclipped": un_noclip, "changes_mae": abs(un["mae_m"] - g["mae_m"]) > 1e-9},
             "label_arm": {"mae_m": lab["mae_m"], "r2": lab["r2"], "mae_is_0_r2_is_1": lab["mae_m"] == 0.0 and abs(lab["r2"] - 1.0) < 1e-12}}
        okc = c["shuffle"]["mae_exceeds_unshuffled"] and c["shuffle"]["r2_le_0"] and c["shuffle"]["auroc_in_0.45_0.55"] and c["units_x100"]["changes_mae"] and c["label_arm"]["mae_is_0_r2_is_1"]
        c["all_pass"] = bool(okc)
        if not okc:
            ctrl_fail.append(f"s{s}")
        res["controls"][f"s{s}"] = c
        res["seeds"][f"s{s}"] = {"gap": g, "presence": {"auroc": au, "auroc_ci95": ci(aub), "n_pos": n_pos, "n_neg": n_neg},
                                 "head_read_summary": {"presence_prob_mean_all_windows": float((1 / (1 + np.exp(-z["lead_logit"]))).mean()), "gap_hat_mean_all_windows": float(z["lead_gap_hat"].mean()),
                                                       "gap_read_unclipped_min_max_m_on_gap_windows": [float(raw.min() * 40), float(raw.max() * 40)]}}
    mean_mae = float(np.mean([res["seeds"][f"s{s}"]["gap"]["mae_m"] for s in (0, 1)]))
    mean_ci = ci((mae_boot[0] + mae_boot[1]) / 2)
    res["mean_over_seeds_gap_mae_m"] = {"point": mean_mae, "ci95": mean_ci}
    gap_ok = mean_mae <= BAR_MAE
    pres_ok = all(res["seeds"][f"s{s}"]["presence"]["auroc"] >= BAR_AUROC for s in (0, 1))
    admissible = not ctrl_fail
    res["controls_admissible"] = admissible
    if not admissible:
        res["label"] = "NOT-ADMISSIBLE (a control did not read as stated: %s)" % ", ".join(ctrl_fail)
    elif gap_ok and pres_ok:
        res["label"] = "NATIVE-OK"
    else:
        res["label"] = "+".join(l for l, f in (("NATIVE-GAP-FAIL", not gap_ok), ("NATIVE-PRES-FAIL", not pres_ok)) if f)
    json.dump(res, open(out_json, "w", encoding="utf-8", newline="\n"), indent=1)
    S = res["seeds"]
    L = ["# NATIVE40 result: T0 read of the native lead head (LLR_s0, LLR_s1)", "",
         f"**Label: {res['label']}**   (evidence MEASURED, tier T0; the interval is the episode-cluster bootstrap B=2000 seed 0 and answers another draw of EPISODES only)", "",
         "Bars (card): mean gap MAE <= 5.80 m AND presence AUROC >= 0.85 at both seeds. Reference 5.04 m is INHERITED (LS_REF40.json), used only as the bar's base.", "",
         f"Map control (before any head number): {res['map_control']['n_matched']} of {n_w} coordinator windows match our label within 1e-3 m (share {share:.4f}); rule: {res['map_control']['rule']}.", "",
         "| seed | gap MAE m [95% CI] | R2 [95% CI] | n gap | AUROC [95% CI] | n pos / n neg |", "|---|---|---|---|---|---|"]
    for s in (0, 1):
        x = S[f"s{s}"]
        L.append(f"| s{s} | {x['gap']['mae_m']:.3f} [{x['gap']['mae_ci95'][0]:.3f}, {x['gap']['mae_ci95'][1]:.3f}] | {x['gap']['r2']:.3f} [{x['gap']['r2_ci95'][0]:.3f}, {x['gap']['r2_ci95'][1]:.3f}] | {x['gap']['n']} | "
                 f"{x['presence']['auroc']:.4f} [{x['presence']['auroc_ci95'][0]:.4f}, {x['presence']['auroc_ci95'][1]:.4f}] | {n_pos} / {n_neg} |")
    L += ["", f"Mean over seeds gap MAE: {mean_mae:.3f} m [{mean_ci[0]:.3f}, {mean_ci[1]:.3f}] (bar 5.80): {'holds' if gap_ok else 'fails'}. AUROC at both seeds >= 0.85: {'holds' if pres_ok else 'fails'}.", "",
          "## Controls", "", "| seed | shuffle MAE / R2 / AUROC | units x100 MAE (clipped / unclipped) | label arm MAE / R2 | all pass |", "|---|---|---|---|---|"]
    for s in (0, 1):
        c = res["controls"][f"s{s}"]
        L.append(f"| s{s} | {c['shuffle']['mae_m']:.3f} / {c['shuffle']['r2']:.3f} / {c['shuffle']['presence_auroc_shuffled']:.3f} | {c['units_x100']['mae_m_clipped']:.3f} / {c['units_x100']['mae_m_unclipped']:.3f} | "
                 f"{c['label_arm']['mae_m']:.3f} / {c['label_arm']['r2']:.3f} | {c['all_pass']} |")
    L += ["", f"Windows file sha256 {wsha}. Dumps: " + "; ".join(f"s{s} {res['dump_meta'][f's{s}']['dump_sha256'][:16]} ckpt {res['dump_meta'][f's{s}']['ckpt_sha256'][:16]}" for s in (0, 1)) + ".",
          "Scope: a head choice only; no LONGITUDINAL, LATERAL, TACTICAL or STRATEGIC family is claimed from this read (card section 6)."]
    open(md, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print(res["label"], json.dumps({k: S[k]["gap"]["mae_m"] for k in S}), json.dumps({k: S[k]["presence"]["auroc"] for k in S}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
