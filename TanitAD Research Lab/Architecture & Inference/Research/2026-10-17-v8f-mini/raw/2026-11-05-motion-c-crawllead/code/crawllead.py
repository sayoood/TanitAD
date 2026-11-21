"""crawllead: do crawl states (v8F T1-kin bank rollouts) sit in scenes with a lead agent?
Frozen rules: ../PREREG_CRAWLLEAD.md (sha256 in ../FROZEN_SHA.txt). numpy + torch (cache cross-check only). CPU only.
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I crawllead.py
"""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

OUT = Path("E:/Projects/TanitAD-artifacts/dagland/motion_c/crawllead")
RAW_BANK = Path("E:/Projects/TanitAD-artifacts/dagland/bank_presteps/raw")
LEAD = Path("F:/TanitAD-scratch/motion_c/crawllead/in/lead_train4572.npz")
CACHE = Path("F:/TanitAD-scratch/speedread/cache/train/clips")
OK = 0
B, SEED = 2000, 0


def sha12(cid):
    return hashlib.sha256(cid.encode("utf-8")).hexdigest()[:12]      # v8f_mini_data.py:76-77


def boot_picks(n, B=B, seed=SEED, chunk=200):
    rng = np.random.default_rng(seed)
    done = 0
    while done < B:
        m = min(chunk, B - done)
        yield rng.choice(n, size=(m, n), replace=True)
        done += m


def pct(a, lo=2.5, hi=97.5):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, lo)), float(np.percentile(a, hi)), int(len(a))


def clip_sums(cid, ncl, *vals):
    return [np.bincount(cid, weights=np.asarray(v, np.float64), minlength=ncl) for v in vals]


def share_ci(num, den):
    """num/den per-clip sums -> point, lo, hi (clip-cluster bootstrap)."""
    point = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    bs = []
    for idx in boot_picks(len(num)):
        d = den[idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bs.append(np.where(d > 0, num[idx].sum(1) / d, np.nan))
    bs = np.concatenate(bs)
    lo, hi, nok = pct(bs)
    return {"point": point, "lo": lo, "hi": hi, "n_boot_ok": nok}


def diff_ci(na, da, nb, db):
    point = float(na.sum() / da.sum() - nb.sum() / db.sum())
    bs = []
    for idx in boot_picks(len(na)):
        with np.errstate(invalid="ignore", divide="ignore"):
            bs.append(na[idx].sum(1) / da[idx].sum(1) - nb[idx].sum(1) / db[idx].sum(1))
    bs = np.concatenate(bs)
    lo, hi, nok = pct(bs)
    return {"point": point, "lo": lo, "hi": hi, "n_boot_ok": nok}


def quart(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return None
    q = np.percentile(a, [5, 25, 50, 75, 95])
    return {"n": int(len(a)), "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4])}


def build_lead_arrays(clip_names):
    """Per bank clip: lead flag (-1 = no row), gap, ego v, tg on raw frames 0..K-1 (K = 210)."""
    z = np.load(LEAD, allow_pickle=True)
    sh, k = z["sha12"], z["k"].astype(np.int64)
    K = int(k.max()) + 2
    s12 = np.array([sha12(c) for c in clip_names.tolist()])
    order = {s: i for i, s in enumerate(s12.tolist())}
    ci = np.array([order.get(s, -1) for s in sh.tolist()], np.int64)
    keep = ci >= 0
    n = len(clip_names)
    L = np.full((n, K), -1, np.int8)
    G = np.full((n, K), np.nan)
    V = np.full((n, K), np.nan)
    TG = np.full((n, K), np.nan)
    L[ci[keep], k[keep]] = z["lead"][keep].astype(np.int8)
    G[ci[keep], k[keep]] = z["gap"][keep]
    V[ci[keep], k[keep]] = z["v"][keep]
    TG[ci[keep], k[keep]] = z["tg"][keep]
    # one row per (clip, k): assert uniqueness
    keyc = ci[keep] * K + k[keep]
    assert len(np.unique(keyc)) == int(keep.sum()), "duplicate (clip, k) rows in the lead file"
    return {"L": L, "G": G, "V": V, "TG": TG, "K": K, "rows_total": int(len(sh)), "rows_in_bank_clips": int(keep.sum()),
            "clips_in_lead_file": int(len(np.unique(sh))), "bank_clips_with_rows": int((L >= 0).any(1).sum()),
            "meta": str(z["meta"])}


def main():
    lg = np.load(RAW_BANK / "logged_cache.npz")
    names = lg["names"].astype(str)
    ncl = len(names)
    lead = build_lead_arrays(names)
    L, G, LV, TG, K = lead["L"], lead["G"], lead["V"], lead["TG"], lead["K"]
    res = {"frozen_sha256": open(OUT / "FROZEN_SHA.txt").read().split()[0],
           "lead_file": {k: v for k, v in lead.items() if k in ("rows_total", "rows_in_bank_clips", "clips_in_lead_file", "bank_clips_with_rows", "meta")},
           "n_bank_clips": ncl, "seeds": {}, "controls": {}}

    # C5: speedread cache v vs logged_cache V on a seeded 150-clip sample
    import torch
    rng = np.random.default_rng(12345)
    samp = rng.choice(ncl, size=150, replace=False)
    md, nfin, nbad = 0.0, 0, 0
    for c in samp.tolist():
        o = torch.load(CACHE / f"{names[c]}.pt", weights_only=True, map_location="cpu")["v"].double().numpy()
        T = int(lg["Ts"][c])
        a = lg["V"][lg["offs"][c]: lg["offs"][c] + T]
        if len(o) != T:
            nbad += 1
            continue
        m = np.isfinite(a) & np.isfinite(o)
        nfin += int(m.sum())
        md = max(md, float(np.abs(a[m] - o[m]).max()))
    res["controls"]["C5_cache_vs_logged"] = {"n_clips": 150, "n_clips_len_mismatch": nbad, "n_values": nfin, "max_abs_diff_mps": md, "PASS": bool(nbad == 0 and md <= 0.01)}

    for seed in ("s0", "s1"):
        z = np.load(RAW_BANK / f"bank_table_{seed}.npz")
        tab = {k: z[k] for k in z.files}
        assert (tab["clip_names"].astype(str) == names).all()
        cid = tab["clip_g"].astype(np.int64)
        r = tab["r"].astype(np.int64)
        tick = tab["tick"]
        T = lg["Ts"][cid]
        okr = (r >= 0) & (r < T)
        vg = np.full(len(r), np.nan)
        vg[okr] = lg["V"][lg["offs"][cid[okr]] + r[okr]]
        vo = tab["v_own"]
        usable = tab["reason"] == OK
        fin = np.isfinite(vo) & np.isfinite(vg)
        crawl = usable & fin & (vo < 0.5 * vg) & (vg >= 3.0)
        mok = usable & fin & (vg >= 3.0) & (vg <= 8.0) & (vo >= 0.8 * vg)
        assert not (crawl & mok).any()                                    # C6
        k2 = 2 * r
        assert k2[usable & fin].max() < K
        lflag = np.where(okr, L[cid, np.clip(k2, 0, K - 1)], -1)          # -1 unlabelled
        gap = np.where(okr, G[cid, np.clip(k2, 0, K - 1)], np.nan)
        lab = lflag >= 0
        within = lambda d: (lflag == 1) & (gap <= d)                      # NaN gap -> False
        out = {"n_records": int(len(r)), "n_usable": int(usable.sum()), "n_finite_usable": int((usable & fin).sum())}

        # C3 / C4 clock alignment of the join over labelled usable records
        sel = usable & fin & lab
        dv = np.abs(LV[cid[sel], k2[sel]] - vg[sel])
        k2b = np.clip(k2[sel] + 2, 0, K - 1)
        dv_off = np.abs(LV[cid[sel], k2b] - vg[sel])
        dvf, dvof = dv[np.isfinite(dv)], dv_off[np.isfinite(dv_off)]
        c3 = {"n": int(len(dvf)), "median_abs_dv": float(np.median(dvf)), "p95_abs_dv": float(np.percentile(dvf, 95)),
              "off_by_one_step_median_abs_dv": float(np.median(dvof)), "off_by_one_step_p95": float(np.percentile(dvof, 95))}
        c3["C3_PASS"] = bool(c3["median_abs_dv"] <= 0.05 and c3["p95_abs_dv"] <= 0.25)
        c3["C4_PASS"] = bool(c3["off_by_one_step_median_abs_dv"] > c3["median_abs_dv"])
        res["controls"][f"C3C4_clock_{seed}"] = c3

        sets = {"CRAWL": crawl, "MOVING_OK": mok}
        # counts
        cl_crawl = np.unique(cid[crawl])
        out["sets"] = {nm: {"n": int(m.sum()), "n_labelled": int((m & lab).sum()), "n_unlabelled": int((m & ~lab).sum()),
                            "n_clips": int(len(np.unique(cid[m])))} for nm, m in sets.items()}
        out["sets"]["CRAWL_tick"] = {str(t): {"n": int((crawl & (tick == t)).sum()), "n_labelled": int((crawl & (tick == t) & lab).sum())} for t in (1, 2)}

        # shares + CI
        for d in (20, 40, 60):
            out[f"L{d}"] = {}
            w = within(d)
            for nm, m in sets.items():
                num, den = clip_sums(cid, ncl, m & lab & w, m & lab)
                out[f"L{d}"][nm] = share_ci(num, den)
                out[f"L{d}"][nm].update(n_num=int((m & lab & w).sum()), n_den=int((m & lab).sum()))
            na, da = clip_sums(cid, ncl, crawl & lab & w, crawl & lab)
            nb, db = clip_sums(cid, ncl, mok & lab & w, mok & lab)
            out[f"L{d}"]["DIFF"] = diff_ci(na, da, nb, db)

        # speed-matched: CRAWL restricted to v_GT in [3,8] vs MOVING_OK
        sm = crawl & (vg <= 8.0)
        w40 = within(40)
        num, den = clip_sums(cid, ncl, sm & lab & w40, sm & lab)
        out["L40_CRAWL_vgt_3_8"] = share_ci(num, den)
        out["L40_CRAWL_vgt_3_8"].update(n_den=int((sm & lab).sum()))
        na, da = clip_sums(cid, ncl, sm & lab & w40, sm & lab)
        nb, db = clip_sums(cid, ncl, mok & lab & w40, mok & lab)
        out["DIFF40_speed_matched"] = diff_ci(na, da, nb, db)
        # tick split
        out["L40_CRAWL_by_tick"] = {}
        for t in (1, 2):
            m = crawl & (tick == t)
            num, den = clip_sums(cid, ncl, m & lab & w40, m & lab)
            out["L40_CRAWL_by_tick"][str(t)] = dict(share_ci(num, den), n_den=int((m & lab).sum()))

        # v_GT distribution of CRAWL
        vgc = vg[crawl]
        out["vgt_CRAWL"] = dict(quart(vgc), shares={"[3,5)": float(((vgc >= 3) & (vgc < 5)).mean()), "[5,8]": float(((vgc >= 5) & (vgc <= 8)).mean()),
                                                    "(8,15]": float(((vgc > 8) & (vgc <= 15)).mean()), ">15": float((vgc > 15).mean())})
        out["vgt_MOVING_OK"] = quart(vg[mok])
        out["vown_over_vgt_CRAWL"] = quart(vo[crawl] / vg[crawl])

        # among CRAWL with a lead (any gap)
        cw = crawl & (lflag == 1)
        tgm = np.where(okr, TG[cid, np.clip(k2, 0, K - 1)], np.nan)
        out["crawl_with_lead"] = {"n": int(cw.sum()), "share_of_labelled_crawl": float(cw.sum() / (crawl & lab).sum()),
                                  "gap_m": quart(gap[cw]), "tg_file_s": quart(tgm[cw]),
                                  "tg_model_speed_s": quart(np.maximum(gap[cw], 0) / np.maximum(vo[cw], 1.0))}
        mw = mok & (lflag == 1)
        out["moving_ok_with_lead"] = {"n": int(mw.sum()), "gap_m": quart(gap[mw]), "tg_file_s": quart(tgm[mw])}

        # lead speed ESTIMATE (PREREG beside item 6)
        kp, km = np.clip(k2 + 2, 0, K - 1), np.clip(k2 - 2, 0, K - 1)
        lp, lm = L[cid, kp], L[cid, km]
        gp, gm = G[cid, kp], G[cid, km]
        ve = LV[cid, np.clip(k2, 0, K - 1)]
        valid = cw & (lp == 1) & (lm == 1) & (k2 + 2 < K) & (k2 - 2 >= 0) & (np.abs(gp - gap) <= 4.0) & (np.abs(gap - gm) <= 4.0)
        vlead = ve + (gp - gm) / 0.4
        vv = valid & np.isfinite(vlead)
        out["lead_speed_est"] = {"EVIDENCE_CLASS": "ESTIMATED", "n_valid": int(vv.sum()), "share_valid_of_crawl_with_lead": float(vv.sum() / max(cw.sum(), 1)),
                                 "v_lead_est": quart(vlead[vv]),
                                 "share_lead_slower_than_vgt": float((vlead[vv] < vg[vv]).mean()) if vv.any() else None,
                                 "share_lead_slower_than_vown": float((vlead[vv] < vo[vv]).mean()) if vv.any() else None,
                                 "share_lead_slower_than_half_vgt": float((vlead[vv] < 0.5 * vg[vv]).mean()) if vv.any() else None}
        # same estimate for MOVING_OK with lead
        mwv = mw & (lp == 1) & (lm == 1) & (k2 + 2 < K) & (k2 - 2 >= 0) & (np.abs(gp - gap) <= 4.0) & (np.abs(gap - gm) <= 4.0) & np.isfinite(vlead)
        out["lead_speed_est_MOVING_OK"] = {"n_valid": int(mwv.sum()), "v_lead_est": quart(vlead[mwv]),
                                           "share_lead_slower_than_vgt": float((vlead[mwv] < vg[mwv]).mean()) if mwv.any() else None}

        # sensitivity to unlabelled
        sens = {}
        for nm, m in sets.items():
            n_all = int(m.sum())
            n_w = int((m & lab & w40).sum())
            n_u = int((m & ~lab).sum())
            sens[nm] = {"trainer_semantics_unlabelled_is_no_lead": n_w / n_all, "lower_bound_unlabelled_all_no_lead": n_w / n_all,
                        "upper_bound_unlabelled_all_lead": (n_w + n_u) / n_all}
        sens["DIFF_trainer_semantics"] = sens["CRAWL"]["trainer_semantics_unlabelled_is_no_lead"] - sens["MOVING_OK"]["trainer_semantics_unlabelled_is_no_lead"]
        out["sensitivity_unlabelled"] = sens

        # controls C1, C2
        c1 = crawl.copy()
        na, da = clip_sums(cid, ncl, mok & lab & w40, mok & lab)
        nb, db = clip_sums(cid, ncl, mok & lab & w40, mok & lab)
        d_c1 = float(na.sum() / da.sum() - nb.sum() / db.sum())
        ones = np.where(lab, 1, -1)
        w_ones = (ones == 1)
        l_one = {nm: float((m & lab & w_ones).sum() / (m & lab).sum()) for nm, m in sets.items()}
        l_zero = {nm: float(0 / (m & lab).sum()) for nm, m in sets.items()}
        out["controls"] = {"C1_diff_when_crawl_is_moving_ok": d_c1, "C2_all_ones_L40": l_one, "C2_all_zeros_L40": l_zero,
                           "PASS": bool(d_c1 == 0.0 and all(v == 1.0 for v in l_one.values()) and all(v == 0.0 for v in l_zero.values())),
                           "C6_disjoint": True}

        # verdict
        Lc, Df = out["L40"]["CRAWL"]["point"], out["L40"]["DIFF"]["point"]
        if Lc >= 0.50 and Df >= 0.10:
            v = "AGENT-TARGET"
        elif Lc < 0.25:
            v = "NO-AGENT-TARGET"
        else:
            v = "PARTIAL"
        straddle = []
        c = out["L40"]["CRAWL"]
        for th in (0.25, 0.50):
            if c["lo"] <= th <= c["hi"]:
                straddle.append(f"L40(CRAWL) CI [{c['lo']:.4f},{c['hi']:.4f}] contains {th}")
        dd = out["L40"]["DIFF"]
        if dd["lo"] <= 0.10 <= dd["hi"]:
            straddle.append(f"DIFF CI [{dd['lo']:.4f},{dd['hi']:.4f}] contains 0.10")
        out["verdict"] = v
        out["threshold_inside_CI"] = straddle
        res["seeds"][seed] = out
        print(seed, v, "L40 CRAWL", round(Lc, 4), "DIFF", round(Df, 4), flush=True)

    vs = {s: res["seeds"][s]["verdict"] for s in res["seeds"]}
    res["overall_verdict"] = vs["s0"] if vs["s0"] == vs["s1"] else "PARTIAL"
    res["seeds_agree"] = vs["s0"] == vs["s1"]
    ctl_ok = (res["controls"]["C5_cache_vs_logged"]["PASS"]
              and all(res["controls"][f"C3C4_clock_{s}"]["C3_PASS"] and res["controls"][f"C3C4_clock_{s}"]["C4_PASS"] for s in ("s0", "s1"))
              and all(res["seeds"][s]["controls"]["PASS"] for s in ("s0", "s1")))
    res["controls_all_pass"] = bool(ctl_ok)
    (OUT / "raw").mkdir(exist_ok=True)
    with open(OUT / "raw" / "crawllead_result.json", "w", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("overall", res["overall_verdict"], "controls_all_pass", ctl_ok)


if __name__ == "__main__":
    main()
