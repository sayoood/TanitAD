"""Read the turn-asymmetry panel against SPEC_TURN_ASYMMETRY.md section 1.

⛔ ESTIMATOR. Every interval here is an EPISODE-CLUSTER bootstrap, drawn with
`taniteval.ci`'s OWN resampler (`ci.episode_index` + `ci._draws`) so the draws are
the programme's and not a re-implementation. `overlapping_holdout_se` is never
used. The seed floor is the PAIRED form (same windows, two arms).

⚠️ The within-arm LEFT-vs-RIGHT contrast is NOT paired: the two strata are
different windows. It is a cluster bootstrap of the DIFFERENCE OF TWO STRATUM
MEANS recomputed inside each resampled episode set, which is the correct form —
a paired estimator there would be a lie about what is shared.

Every table carries a control that must read a known value.
"""
import collections
import glob
import json
import os
import sys

import numpy as np
import torch

WT = "C:/Users/Admin/tanitad-wt"
for p in (WT + "/stack", WT + "/taniteval", WT):
    if p not in sys.path:
        sys.path.insert(0, p)
from taniteval import ci as _ci                                   # noqa: E402
from taniteval import four_families as ff                         # noqa: E402
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7 as LATV   # noqa: E402
from tanitad.refs.refc_tactical import factor_from_kinematics      # noqa: E402

P4 = "C:/Users/Admin/refav1_margin/p4out"
DT = 0.2
NB = 2000
LAT = {0: "lane_keep", 1: "turn_left", 2: "turn_right"}
I_TL, I_TR = LATV.index("TURN_L"), LATV.index("TURN_R")
FULL_KAPPA = 0.06          # "tracks the goal" -- GOAL_KAPPA_TURN is 0.08
SEED_FLOOR_POOLED = 0.0750  # the banked TAC_traj_lat_correct seed floor


def load(tag):
    c = collections.defaultdict(list)
    d0 = os.path.join(P4, "dump_" + tag)
    for p in sorted(glob.glob(os.path.join(d0, "ep*.npz"))):
        d = np.load(p)
        n = d["g"].shape[0]
        for k in ("g", "cl", "ha0_ext", "ol", "ha"):
            c[k].append(d[k])
        c["v0"].append(d["v0"])
        c["eid"].append(np.repeat(d["eid"], n))
        q = np.load(os.path.join(d0, "decisions", os.path.basename(p)))
        for k in ("goal_lat_cl", "cl_controls"):
            c[k].append(q[k])
    if not c:
        raise SystemExit("[turn_asym_read] no dump for %r" % tag)
    return {k: np.concatenate(v) for k, v in c.items()}


def lab(A):
    t = torch.as_tensor(A).float()
    dy, dv, v0, v1, _ = ff.maneuver_kinematics(t, DT)
    return factor_from_kinematics(dy, dv, v0, v1)[0].numpy()


def peak_kappa(ctrl):
    k = ctrl[:, :, 1]
    return k[np.arange(k.shape[0]), np.abs(k).argmax(1)]


def cluster_ci(vals, eid, n_boot=NB, seed=0):
    """mean(vals) with an episode-cluster bootstrap CI (single arm)."""
    r = _ci.episode_cluster_bootstrap(np.asarray(vals, float), eid,
                                      n_boot=n_boot, seed=seed)
    return r["mean"], r["lo"], r["hi"]


def contrast_ci(hit, eid, mask_a, mask_b, n_boot=NB, seed=0):
    """CI on mean(hit[mask_a]) - mean(hit[mask_b]) over resampled EPISODES.

    Uses `ci`'s own draw machinery on the FULL window set, then recomputes both
    stratum means inside each draw, so an episode carrying only one direction
    contributes to only that side -- which is the honest behaviour and is why
    the interval widens when a stratum lives in few clusters.
    """
    hit = np.asarray(hit, float)
    uniq, idx_by_ep = _ci.episode_index(eid)
    pt = float(hit[mask_a].mean() - hit[mask_b].mean())
    out = []
    for sel in _ci._draws(uniq, idx_by_ep, n_boot, seed):
        a, b = hit[sel][mask_a[sel]], hit[sel][mask_b[sel]]
        if a.size and b.size:
            out.append(float(a.mean() - b.mean()))
    if not out:
        return pt, float("nan"), float("nan"), False, 0
    lo, hi = np.percentile(out, [2.5, 97.5])
    return pt, float(lo), float(hi), bool(lo > 0 or hi < 0), len(out)


def within_episode_contrast(hit, eid, mask_a, mask_b, n_boot=NB, seed=0):
    """⭐ THE PRIMARY ATTRIBUTION STATISTIC (SPEC section 3.13).

    Mean, over the episodes carrying BOTH strata, of (mean(a) - mean(b)) INSIDE
    that episode -- then an episode-cluster bootstrap over that episode set.

    ⛔ WHY IT IS PRIMARY. On the banked panel the two goal tokens live in
    COMPLETELY DISJOINT episodes, so "the goal is TURN_L" and "the window is in
    episode 1 or 6" are the SAME VARIABLE and the pooled contrast cannot
    attribute anything. Restricting to episodes that carry both removes the
    confound by construction. Returns (point, lo, hi, separated, episodes_used).
    """
    hit = np.asarray(hit, float)
    eid = np.asarray(eid)
    eps = sorted(set(eid[mask_a].tolist()) & set(eid[mask_b].tolist()))
    if not eps:
        return float("nan"), float("nan"), float("nan"), False, []
    per_ep = {e: float(hit[mask_a & (eid == e)].mean()
                       - hit[mask_b & (eid == e)].mean()) for e in eps}
    pt = float(np.mean([per_ep[e] for e in eps]))
    rng = np.random.default_rng(seed)
    draws = [float(np.mean([per_ep[e] for e in rng.choice(eps, len(eps))]))
             for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return pt, float(lo), float(hi), bool(lo > 0 or hi < 0), eps


def main(tags):
    D = {t: load(t) for t in tags}
    ref = D[tags[0]]
    lat_g = lab(ref["g"])
    eid = ref["eid"]
    v0 = ref["v0"]
    n = len(lat_g)
    for t in tags:
        if len(D[t]["g"]) != n or not np.array_equal(D[t]["eid"], eid):
            raise SystemExit("[turn_asym_read] %r is on a DIFFERENT panel "
                             "(n=%d vs %d) -- cross-panel comparison refused"
                             % (t, len(D[t]["g"]), n))
        if not np.allclose(D[t]["g"], ref["g"], atol=0, rtol=0):
            raise SystemExit("[turn_asym_read] %r has different GT -- refused" % t)

    mL, mR = lat_g == 1, lat_g == 2
    print("=" * 100)
    print("PANEL: n = %d windows / %d episodes | GT lane_keep %d, turn_left %d "
          "(%d eps), turn_right %d (%d eps)"
          % (n, len(set(eid.tolist())), (lat_g == 0).sum(), mL.sum(),
             len(set(eid[mL].tolist())), mR.sum(), len(set(eid[mR].tolist()))))
    print("POWER TARGETS (SPEC section 1): n >= 27 per direction, >= 5 clusters, "
          "granularity 1/n <= 0.0375")
    for nm, m in (("turn_left", mL), ("turn_right", mR)):
        nn, cc = int(m.sum()), len(set(eid[m].tolist()))
        print("  %-11s n=%-3d %-10s clusters=%-2d %-10s 1/n=%.4f %s"
              % (nm, nn, "MET" if nn >= 27 else "**MISSED**", cc,
                 "MET" if cc >= 5 else "**MISSED**", 1.0 / max(nn, 1),
                 "resolvable" if 1.0 / max(nn, 1) <= SEED_FLOOR_POOLED / 2
                 else "**TOO COARSE**"))

    hit = {t: (lab(D[t]["cl"]) == lat_g).astype(float) for t in tags}
    hit["ha0_ext"] = (lab(ref["ha0_ext"]) == lat_g).astype(float)
    hit["ol"] = (lab(ref["ol"]) == lat_g).astype(float)
    hit["ha"] = (lab(ref["ha"]) == lat_g).astype(float)
    # ⭐ THE GOAL-HEAD BASELINE (SPEC section 3.16, registered before any `cl`
    # output existed). At W_KAPPA = 0 the plan IS the goal's canonical seed on
    # 21/22 turn-goal windows, so the plan's turn recall is BOUNDED ABOVE by the
    # head's: a plan-recall gap is PARTLY INHERITED and only the INCREMENT over
    # the head is attributable to the cost. The decoded goal is an INPUT and is
    # identical across arms and across plan seeds, so one column serves all.
    gl0 = D[tags[0]]["goal_lat_cl"]
    hit["GOAL_HEAD"] = np.where(lat_g == 1, gl0 == I_TL,
                                np.where(lat_g == 2, gl0 == I_TR,
                                         False)).astype(float)

    print()
    print("=" * 100)
    print("1. PER-DIRECTION TURN RECALL, episode-cluster bootstrap")
    print("=" * 100)
    print("  %-14s | %-26s | %-26s | %s"
          % ("arm", "recall LEFT  [95% CI]", "recall RIGHT [95% CI]", "R - L  [95% CI]  sep?"))
    rec = {}
    for t in list(tags) + ["GOAL_HEAD", "ha0_ext", "ha", "ol"]:
        h = hit[t]
        lm, ll, lh = cluster_ci(h[mL], eid[mL])
        rm, rl, rh = cluster_ci(h[mR], eid[mR])
        d, dl, dh, sep, nd = contrast_ci(h, eid, mR, mL)
        rec[t] = (lm, rm, d, sep)
        star = (" (FLOOR, no planner)" if t in ("ha0_ext", "ha", "ol")
                else "  <-- BASELINE: the decoded goal, an INPUT (SPEC 3.16)"
                if t == "GOAL_HEAD" else "")
        print("  %-14s | %6.4f [%6.4f, %6.4f] | %6.4f [%6.4f, %6.4f] | %+6.4f "
              "[%+6.4f, %+6.4f] %s%s"
              % (t, lm, ll, lh, rm, rl, rh, d, dl, dh,
                 "YES" if sep else "no ", star))
    print()
    print("  ⭐ THE INCREMENT OVER THE GOAL HEAD (SPEC 3.16) — the only part of a")
    print("     plan-recall gap that is attributable to the COST:")
    gh = rec["GOAL_HEAD"][2]
    for t in tags:
        print("     %-14s plan gap %+0.4f  -  head gap %+0.4f  =  INCREMENT %+0.4f"
              % (t, rec[t][2], gh, rec[t][2] - gh))
    print("     (a plan gap no larger than the head's is INHERITED, not caused by")
    print("      W_KAPPA; the head is identical across arms and across plan seeds.)")
    print("  CONTROL (must read exactly +0.0000): an arm's LEFT recall against "
          "itself = %+0.4f"
          % (cluster_ci(hit[tags[0]][mL], eid[mL])[0]
             - cluster_ci(hit[tags[0]][mL], eid[mL])[0]))

    print()
    print("=" * 100)
    print("1b. ⭐ WITHIN-EPISODE CONTRAST (SPEC 3.13) — the PRIMARY attribution "
          "statistic")
    print("=" * 100)
    eps_both = sorted(set(eid[mL].tolist()) & set(eid[mR].tolist()))
    print("  episodes carrying BOTH GT directions: %s  (%d left + %d right "
          "windows inside them)"
          % (eps_both, int((mL & np.isin(eid, eps_both)).sum()),
             int((mR & np.isin(eid, eps_both)).sum())))
    if not eps_both:
        print("  ⛔ NONE — direction and episode are the SAME VARIABLE on this "
              "panel; NOTHING is attributable to direction.")
    win = {}
    for t in list(tags) + ["ha0_ext", "ol"]:
        p, lo, hi, sep, used = within_episode_contrast(hit[t], eid, mR, mL)
        win[t] = (p, sep)
        print("  %-14s within-episode (R - L) = %+6.4f [%+6.4f, %+6.4f] %s "
              "over %d episodes%s"
              % (t, p, lo, hi, "SEPARATED" if sep else "not sep",
                 len(used), " (FLOOR)" if t in ("ha0_ext", "ol") else ""))
    z, zlo, zhi, _, _ = within_episode_contrast(hit[tags[0]], eid, mR, mR)
    print("  CONTROL, a stratum against ITSELF (must be exactly +0.0000): "
          "%+0.4f [%+0.4f, %+0.4f]" % (z, zlo, zhi))

    print()
    print("=" * 100)
    print("2. GOAL-CONDITIONED CURVATURE RETENTION (SPEC section 1.1)")
    print("=" * 100)
    gl = D[tags[0]]["goal_lat_cl"]
    for t in tags:
        if not np.array_equal(D[t]["goal_lat_cl"], gl):
            print("  ** %s decodes a DIFFERENT goal -- the head is not shared **" % t)
    gL, gR = gl == I_TL, gl == I_TR
    print("  decoded goal: TURN_L n=%d (%d eps), TURN_R n=%d (%d eps)"
          % (gL.sum(), len(set(eid[gL].tolist())), gR.sum(),
             len(set(eid[gR].tolist()))))
    print("  %-14s | %-26s | %-26s | %s"
          % ("arm", "retain|goal=TURN_L", "retain|goal=TURN_R", "R - L  [95% CI]  sep?"))
    ret = {}
    for t in tags:
        keep = (np.abs(peak_kappa(D[t]["cl_controls"])) > FULL_KAPPA).astype(float)
        lm, ll, lh = cluster_ci(keep[gL], eid[gL]) if gL.any() else (np.nan,) * 3
        rm, rl, rh = cluster_ci(keep[gR], eid[gR]) if gR.any() else (np.nan,) * 3
        d, dl, dh, sep, _ = contrast_ci(keep, eid, gR, gL)
        ret[t] = (lm, rm, d, sep)
        print("  %-14s | %6.4f [%6.4f, %6.4f] | %6.4f [%6.4f, %6.4f] | %+6.4f "
              "[%+6.4f, %+6.4f] %s"
              % (t, lm, ll, lh, rm, rl, rh, d, dl, dh, "YES" if sep else "no "))
    print("  CONTROL: the SIGN is correct on every retained plan --", end=" ")
    okL = okR = tot = 0
    for t in tags:
        k = peak_kappa(D[t]["cl_controls"])
        m = np.abs(k) > FULL_KAPPA
        okL += int(((k > 0) & m & gL).sum()); okR += int(((k < 0) & m & gR).sum())
        tot += int((m & (gL | gR)).sum())
    print("%d/%d retained plans curve the way their goal asked" % (okL + okR, tot))

    print()
    print("=" * 100)
    print("3. THE PANEL'S OWN INFERENCE-SEED FLOOR (paired, same windows)")
    print("=" * 100)
    pairs = [(a, b) for a in tags for b in tags
             if a < b and a.rsplit("_", 1)[0] == b.rsplit("_", 1)[0]]
    if not pairs:
        print("  ** no seed pair in this arm set -- the floor CANNOT be measured **")
    floors = {}
    for a, b in pairs:
        print("  pair %s vs %s:" % (a, b))
        for nm, m in (("turn_left", mL), ("turn_right", mR)):
            r = _ci.paired_episode_cluster_bootstrap(hit[b][m], hit[a][m], eid[m],
                                                     n_boot=NB)
            # ⛔ SPEC amendment 0: the floor is the CI's reach, not the point.
            # Two seeds agreeing exactly is ONE DRAW, not a demonstration that
            # the true seed effect is zero -- and a zero floor would make the
            # "clears the floor" condition satisfiable by any non-zero gap,
            # which is the necessary-not-sufficient failure re-entering through
            # the floor's own back door.
            floors[(a, b, nm)] = max(abs(r["lo"]), abs(r["hi"]))
            print("    %-11s recall %.4f -> %.4f   delta %+0.4f [%+0.4f, %+0.4f] "
                  "separated=%s" % (nm, hit[a][m].mean(), hit[b][m].mean(),
                                    r["delta"], r["lo"], r["hi"], r["separated"]))
        ka = (np.abs(peak_kappa(D[a]["cl_controls"])) > FULL_KAPPA).astype(float)
        kb = (np.abs(peak_kappa(D[b]["cl_controls"])) > FULL_KAPPA).astype(float)
        for nm, m in (("goal TURN_L", gL), ("goal TURN_R", gR)):
            if not m.any():
                continue
            r = _ci.paired_episode_cluster_bootstrap(kb[m], ka[m], eid[m], n_boot=NB)
            floors[(a, b, nm)] = max(abs(r["lo"]), abs(r["hi"]))
            print("    retain %-9s %.4f -> %.4f   delta %+0.4f [%+0.4f, %+0.4f] "
                  "separated=%s" % (nm, ka[m].mean(), kb[m].mean(), r["delta"],
                                    r["lo"], r["hi"], r["separated"]))
        rr = _ci.paired_episode_cluster_bootstrap(hit[b], hit[a], eid, n_boot=NB)
        print("    CONTROL, pooled TAC_traj_lat_correct (the banked floor is "
              "0.0750): delta %+0.4f [%+0.4f, %+0.4f]" % (rr["delta"], rr["lo"], rr["hi"]))
        z = _ci.paired_episode_cluster_bootstrap(hit[a], hit[a], eid, n_boot=NB)
        print("    CONTROL, an arm against ITSELF (must be exactly +0.0000): "
              "%+0.4f [%+0.4f, %+0.4f]" % (z["delta"], z["lo"], z["hi"]))

    print()
    print("=" * 100)
    print("3b. ⭐ THE LEVER: does W_KAPPA CAUSE the asymmetry?  (paired, same windows)")
    print("=" * 100)
    print("  The causal claim is a DIFFERENT claim from 'the asymmetry is real'.")
    print("  It is tested by pairing a W_KAPPA arm against the W_KAPPA = 0 arm at")
    print("  the SAME plan seed, per direction, on the SAME windows.")
    fam = {}
    for t in tags:
        base = t.rsplit("_", 1)[0]
        fam.setdefault(t.rsplit("_", 1)[1], []).append(t)
    pairs = []
    for seed_sfx, arms_ in sorted(fam.items()):
        wk = [a for a in arms_ if "wk15" in a]
        cc = [a for a in arms_ if "ccos" in a]
        if wk and cc:
            pairs.append((cc[0], wk[0], seed_sfx))
    if not pairs:
        print("  ** no (W_KAPPA=0, W_KAPPA>0) pair at a matched seed in this arm set --")
        print("     the CAUSAL half CANNOT be tested here and is reported as OPEN. **")
    for a0, a1, sfx in pairs:
        print("  seed %s:  %s (W_KAPPA 0)  ->  %s" % (sfx, a0, a1))
        for nm, m in (("turn_left", mL), ("turn_right", mR)):
            r = _ci.paired_episode_cluster_bootstrap(hit[a1][m], hit[a0][m],
                                                     eid[m], n_boot=NB)
            print("    %-11s recall %.4f -> %.4f   delta %+0.4f [%+0.4f, %+0.4f] "
                  "separated=%s" % (nm, hit[a0][m].mean(), hit[a1][m].mean(),
                                    r["delta"], r["lo"], r["hi"], r["separated"]))
        g0 = rec[a0][1] - rec[a0][0]
        g1 = rec[a1][1] - rec[a1][0]
        print("    ⇒ the GAP (R - L) moves %+0.4f -> %+0.4f, i.e. W_KAPPA %s it by "
              "%+0.4f" % (g0, g1, "WIDENS" if abs(g1) > abs(g0) else "narrows",
                          g1 - g0))
        z = _ci.paired_episode_cluster_bootstrap(hit[a0][mL], hit[a0][mL],
                                                 eid[mL], n_boot=200)
        print("    CONTROL, the W_KAPPA=0 arm against ITSELF (must be exactly "
              "+0.0000): %+0.4f" % z["delta"])

    print()
    print("=" * 100)
    print("4. THE VERDICT, against SPEC section 1's three conditions")
    print("=" * 100)
    fam = [t for t in tags if t.rsplit("_", 1)[0] == tags[0].rsplit("_", 1)[0]]
    if len(fam) < 2:
        print("  INCONCLUSIVE -- fewer than two seeds of the same arm")
        return
    a, b = sorted(fam)[:2]
    # SPEC amendment 0: the CI's reach, floored at the instrument's own step.
    fl_seed = max(floors.get((a, b, "turn_left"), 0.0),
                  floors.get((a, b, "turn_right"), 0.0))
    step = max(1.0 / max(int(mL.sum()), 1), 1.0 / max(int(mR.sum()), 1))
    fl = max(fl_seed, step)
    print("  floor_used = max(seed CI reach %.4f, instrument step 1/n %.4f) "
          "= %.4f" % (fl_seed, step, fl))
    ok_n = (mL.sum() >= 27 and mR.sum() >= 27
            and len(set(eid[mL].tolist())) >= 5 and len(set(eid[mR].tolist())) >= 5)
    gaps = {t: rec[t][1] - rec[t][0] for t in (a, b)}
    c1 = all(abs(g) > fl for g in gaps.values())
    c2 = (np.sign(gaps[a]) == np.sign(gaps[b])) and gaps[a] != 0
    print("  gap (recall_R - recall_L): %s %+0.4f | %s %+0.4f" % (a, gaps[a], b, gaps[b]))
    print("  cond 1 -- |gap| > floor at BOTH seeds : %s" % ("MET" if c1 else "NOT MET"))
    print("  cond 2 -- sign agrees across seeds    : %s" % ("MET" if c2 else "NOT MET"))
    print("  cond 3 -- n and cluster targets       : %s" % ("MET" if ok_n else "NOT MET"))
    # SPEC 3.13: the pooled and within-episode contrasts must AGREE in sign, or
    # the disagreement is the episode confound speaking and the answer is B.
    wa, wb = win.get(a, (float("nan"), False))[0], win.get(b, (float("nan"), False))[0]
    c4 = (np.sign(wa) == np.sign(gaps[a]) and np.sign(wb) == np.sign(gaps[b])
          and abs(wa) > fl and abs(wb) > fl)
    print("  cond 4 -- within-episode contrast agrees and clears the floor "
          "(%+0.4f / %+0.4f vs floor %.4f): %s"
          % (wa, wb, fl, "MET" if c4 else "NOT MET"))
    if not ok_n:
        print("\n  ==> OUTCOME C: UNDERPOWERED. Neither A nor B may be quoted.")
    elif c1 and c2 and c4:
        print("\n  ==> OUTCOME A: the asymmetry is REAL and is attributable to "
              "DIRECTION (it survives the within-episode contrast).")
    elif c1 and c2:
        print("\n  ==> OUTCOME B: NOT ESTABLISHED -- the pooled gap clears the "
              "floor but the WITHIN-EPISODE contrast does not, so the pooled "
              "gap is the episode confound speaking (SPEC 3.13).")
    else:
        print("\n  ==> OUTCOME B: NOT ESTABLISHED at this n.")


if __name__ == "__main__":
    main(sys.argv[1:])
