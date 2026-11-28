"""H-ST-REGRET zero-training pre-step (deliverable 1). Reads bank tables + logged cache (numpy only). Frozen rules: FROZEN_CARDS.md R1-R6.
Usage: python -I regret_prestep.py   ->  ../regret_prestep.json, ../regret_prestep.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import stats as S  # noqa: E402

CRAWL_FRAC, CRAWL_GT_MIN = 0.5, 3.0          # R2
DIE_AT, LOW_AT, MULT_MIN = 0.15, 0.03, 5.0   # the card's thresholds


def analyse(seed):
    tab = C.attach_logged(C.load_table(seed), C.load_logged())
    n = len(tab["tick"])
    cid = tab["clip_g"].astype(np.int64)
    ncl = len(tab["clip_names"])
    vo, vg = tab["v_own"], tab["v_gt_r"]
    usable = tab["reason"] == C.OK
    fin = np.isfinite(vo) & np.isfinite(vg)
    crawl = usable & fin & (vo < CRAWL_FRAC * vg) & (vg >= CRAWL_GT_MIN)
    tick = tab["tick"]
    res = {"seed": seed, "n_records": int(n), "n_clips": int(ncl), "bank_sha256": str(tab["bank_sha256"]),
           "n_nonfinite_v_own": int((~np.isfinite(vo)).sum()), "n_nonfinite_v_gt_r": int((~np.isfinite(vg)).sum()),
           "usable_by_tick": {str(t): int((usable & (tick == t)).sum()) for t in (1, 2)}}
    # CONTROLS (must read known values; they gate the output, they enter no number)
    def crawl_of(v_own_, v_gt_):
        f = np.isfinite(v_own_) & np.isfinite(v_gt_)
        return usable & f & (v_own_ < CRAWL_FRAC * v_gt_) & (v_gt_ >= CRAWL_GT_MIN)

    rng = np.random.default_rng(0)
    gt_ok = usable & np.isfinite(vg) & (vg >= CRAWL_GT_MIN)
    vg_shuf = vg.copy()
    idx_u = np.flatnonzero(usable)
    vg_shuf[idx_u] = vg[rng.permutation(idx_u)]
    c1 = int(crawl_of(vg.copy(), vg).sum())                                   # v_own := v_GT  -> exactly 0 crawl records
    c2 = int(crawl_of(0.4 * vg, vg).sum())                                    # v_own := 0.4 v_GT -> every usable record with v_GT >= 3 is a crawl state
    c3 = float(crawl_of(vo, vg_shuf).sum() / usable.sum())                    # v_GT shuffled across usable records -> the share must move
    c4 = int(crawl_of(vo, vg).sum() - crawl.sum())                            # the detector is a pure function: recompute equals the main mask
    res["controls"] = {"v_own_eq_v_gt_crawl_count": c1, "v_own_eq_v_gt_expected": 0, "v_own_04_vgt_crawl_count": c2,
                       "v_own_04_vgt_expected": int(gt_ok.sum()), "shuffled_vgt_share": c3, "recompute_diff": c4,
                       "PASS": bool(c1 == 0 and c2 == int(gt_ok.sum()) and c4 == 0 and abs(c3 - crawl.sum() / usable.sum()) > 0.005)}
    assert res["controls"]["PASS"], res["controls"]
    # R3: record-level crawl share among usable records, pooled / tick 1 / tick 2, clip-cluster bootstrap
    res["share"] = {}
    for name, m in (("pooled", np.ones(n, bool)), ("tick1", tick == 1), ("tick2", tick == 2)):
        num, den = S.cluster_sums(cid, (crawl & m), (usable & m), n_clusters=ncl)
        b = S.ratio_boot(num, den)
        b.pop("boots")
        b.update(n_usable=int((usable & m).sum()), n_crawl=int((crawl & m).sum()))
        res["share"][name] = b
    # beside: crawl share among ALL records, and among the cut records that the recovery pool would add (off_lat / off_head)
    cut = (tab["reason"] == C.OFF_LAT) | (tab["reason"] == C.OFF_HEAD)
    crawl_any = fin & (vo < CRAWL_FRAC * vg) & (vg >= CRAWL_GT_MIN)
    res["beside"] = {"crawl_share_all_records": float(crawl_any.sum() / n),
                     "crawl_share_cut_off_records": float((crawl_any & cut).sum() / max(int(cut.sum()), 1)),
                     "n_cut_off_records": int(cut.sum()),
                     "crawl_share_by_reason": {C.REASONS[i]: float((crawl_any & (tab["reason"] == i)).sum() / max(int((tab["reason"] == i).sum()), 1))
                                               for i in range(5)}}
    # R4: sampler-effective share (window uniform, then uniform among that window's candidates)
    t1 = tick == 1
    wclip, wt0 = cid[t1], tab["t0"][t1].astype(np.int64)
    assert len(wclip) == C.N_WINDOWS, len(wclip)
    ukey = cid[usable] * (1 << 20) + tab["r"][usable].astype(np.int64)
    uniq, inv = np.unique(ukey, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    ccnt = np.bincount(inv, weights=crawl[usable].astype(np.float64))
    wkey = wclip * (1 << 20) + wt0
    pos = np.minimum(np.searchsorted(uniq, wkey), len(uniq) - 1)
    hit = uniq[pos] == wkey
    p = np.where(hit, ccnt[pos] / np.maximum(cnt[pos], 1.0), 0.0)
    num, den = S.cluster_sums(wclip, p, hit.astype(float), n_clusters=ncl)
    b = S.ratio_boot(num, den)
    b.pop("boots")
    b.update(n_windows=int(len(wkey)), n_eligible_windows=int(hit.sum()), eligible_fraction=float(hit.mean()),
             n_distinct_usable_keys=int(len(uniq)), n_distinct_crawl_keys=int((ccnt > 0).sum()))
    res["share_sampler_effective"] = b
    # R5: multiplier of a 50:50 deficit-weighted mixture
    d = np.where(usable & fin, np.maximum(0.0, vg - vo), 0.0)
    cr_n, us_n, d_cr, d_us = S.cluster_sums(cid, crawl, usable, np.where(crawl, d, 0.0), d, n_clusters=ncl)

    def fn(cr, us, dcr, dus):
        return (0.5 * (cr / us) + 0.5 * (dcr / dus)) / (cr / us)

    res["multiplier_M_def"] = S.joint_boot({"cr": cr_n, "us": us_n, "dcr": d_cr, "dus": d_us}, fn)
    res["multiplier_M_def"]["p_mix_crawl_point"] = float(0.5 * cr_n.sum() / us_n.sum() + 0.5 * d_cr.sum() / d_us.sum())
    res["multiplier_M_def"]["p_def_crawl_point"] = float(d_cr.sum() / d_us.sum())
    sh = res["share"]["pooled"]["point"]
    res["multiplier_M_max_point"] = float(0.5 + 0.5 / sh) if sh > 0 else float("inf")
    # R6: expected crawl count in the uniform p = 0.1 own-state draw
    N = C.DAGGER_N[seed]
    sp = res["share"]["pooled"]
    se = res["share_sampler_effective"]
    res["expected_crawl_in_draw"] = {
        "dagger_n": N, "source": f"v8f-dagoacc/M2b_{seed}_dagoacc/summary.json dagger_n (MEASURED)",
        "record_share": {"point": N * sp["point"], "lo": N * sp["lo"], "hi": N * sp["hi"]},
        "sampler_effective": {"point": N * se["point"], "lo": N * se["lo"], "hi": N * se["hi"]},
        "card_quote": "~450-600 crawl samples in the ~15-19k own-state draws (card H-ST-REGRET, INHERITED)"}
    return res


def verdict(r0, r1):
    sh = [r["share"]["pooled"] for r in (r0, r1)]
    md = [r["multiplier_M_def"] for r in (r0, r1)]
    dies = all(s["point"] >= DIE_AT for s in sh)
    low = all(s["point"] < LOW_AT for s in sh)
    mult_ok = all(m["point"] >= MULT_MIN for m in md)
    if dies:
        v, why = "CARD DIES", "pooled crawl share >= 15 % at both seeds"
        strad = any(s["lo"] < DIE_AT for s in sh)
    elif low and mult_ok:
        v, why = ">=5x BRANCH (weighting concentrates >= 5x)", "pooled crawl share < 3 % at both seeds and M_def >= 5 at both"
        strad = any(s["hi"] >= LOW_AT for s in sh)
    else:
        v, why = "MIDDLE BAND (report the multiplier; card neither dies nor is confirmed)", "shares / multipliers fall between the card's branches or the seeds split"
        strad = False
    return {"verdict": v, "reason": why, "ci_straddles_threshold": bool(strad), "pooled_share_points": [s["point"] for s in sh],
            "M_def_points": [m["point"] for m in md], "low_branch_share_ok": bool(low), "low_branch_mult_ok": bool(mult_ok), "dies_branch": bool(dies)}


def write_md(out, path):
    L = []
    L.append("# H-ST-REGRET zero-training pre-step (deliverable 1)")
    L.append(f"\nFrozen cards: `FROZEN_CARDS.md` sha256 `{out['frozen_cards_sha256']}`; amendment 1 sha256 `{out['frozen_amendment_1_sha256']}`.")
    L.append("Evidence class MEASURED (ours; artifact `regret_prestep.json`, inputs `raw/bank_table_s{0,1}.npz` built from `v8f-dagger/bank_s{0,1}` and "
             "`sfresh_inputs/inputs/native_tracks.pt` sha256 1db431f1... = the bank header's). Tier T1-kin rollout states on TRAIN (the DAgger bank), not eval, not driving performance.")
    L.append("Estimator: clip-cluster bootstrap (4,512 clips, B = 2000, seed 0, 95 % percentile, construction of `taniteval/ci.py`). It answers 'another draw of CLIPS', not 'another training run' (`H-ESTIM-SEED-1`).\n")
    v = out["verdict"]
    L.append(f"## Verdict (frozen R3): **{v['verdict']}**  -- {v['reason']}; CI straddles threshold: {v['ci_straddles_threshold']}\n")
    rows = []
    for s in ("s0", "s1"):
        r = out["seeds"][s]
        for nm in ("pooled", "tick1", "tick2"):
            b = r["share"][nm]
            rows.append([s, nm, b["n_usable"], b["n_crawl"], f"{100 * b['point']:.3f} %", f"[{100 * b['lo']:.3f}, {100 * b['hi']:.3f}] %"])
    L.append(C.md_table(rows, ["seed", "scope", "usable records", "crawl records", "crawl share", "95 % CI"]))
    L.append("\nCrawl state = own speed < 0.5 x logged speed at the matched row r' AND logged speed >= 3 m/s; usable = reason ok. Decision statistic = pooled row.\n")
    rows = []
    for s in ("s0", "s1"):
        r = out["seeds"][s]
        e = r["share_sampler_effective"]
        m = r["multiplier_M_def"]
        x = r["expected_crawl_in_draw"]
        rows.append([s, f"{100 * e['point']:.3f} % [{100 * e['lo']:.3f}, {100 * e['hi']:.3f}]", f"{m['point']:.2f} [{m['lo']:.2f}, {m['hi']:.2f}]",
                     f"{r['multiplier_M_max_point']:.1f}", x["dagger_n"],
                     f"{x['record_share']['point']:.0f} [{x['record_share']['lo']:.0f}, {x['record_share']['hi']:.0f}]",
                     f"{x['sampler_effective']['point']:.0f} [{x['sampler_effective']['lo']:.0f}, {x['sampler_effective']['hi']:.0f}]"])
    L.append(C.md_table(rows, ["seed", "sampler-effective share", "M_def (50:50 deficit-weighted)", "M_max ceiling", "own-state draws (dagger_n)",
                               "expected crawl in draw (record share)", "(sampler-effective)"]))
    L.append("\nBeside (not decisive): " + "; ".join(
        f"{s}: crawl share over ALL records {100 * out['seeds'][s]['beside']['crawl_share_all_records']:.2f} %, over the "
        f"{out['seeds'][s]['beside']['n_cut_off_records']} off_lat/off_head records {100 * out['seeds'][s]['beside']['crawl_share_cut_off_records']:.2f} %"
        for s in ("s0", "s1")))
    L.append("\nThe card's own prediction (INHERITED, card text): share < 3 % and ~450-600 crawl samples in ~15-19k draws.")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    out = {"deliverable": "H-ST-REGRET pre-step", "frozen_cards_sha256": C.FROZEN_SHA, "frozen_amendment_1_sha256": C.AMEND_SHA,
           "tier": "T1-kin rollout states on TRAIN (DAgger bank)", "evidence_class": "MEASURED", "seeds": {}}
    for s in ("s0", "s1"):
        out["seeds"][s] = analyse(s)
    out["verdict"] = verdict(out["seeds"]["s0"], out["seeds"]["s1"])
    (C.PKG / "regret_prestep.json").write_text(json.dumps(out, indent=1))
    write_md(out, C.PKG / "regret_prestep.md")
    print(json.dumps({"verdict": out["verdict"], "share_pooled": {s: out["seeds"][s]["share"]["pooled"] for s in out["seeds"]}}, indent=1))


if __name__ == "__main__":
    main()
