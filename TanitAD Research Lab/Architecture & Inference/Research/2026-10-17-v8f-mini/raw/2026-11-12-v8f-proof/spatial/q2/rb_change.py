r"""rb_change.py -- the change form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md): how much of the CHANGE of the lead gap over 2.0 s the
rolled-out state of an arm holds.  Tier T0: a world-model diagnostic on logged frames, never driving performance.  CPU, no fit, no model pass.

  python rb_change.py run --oof out/OOF_RB_G_s0.npz [--oof ...] [--pair A:B ...] --out out/CHANGE_RB.md

Inputs: the out-of-fold predictions rb_trained.py saved (p_i rolled-out state, p_ii present state, p_iii true future state, yk = gap at +2.0 s on
the windows with a lead at both times) and the bank (gap at the window start).  Target d = gap(t + 2.0 s) - gap(t).  Predictions of the change:
rollout = p_i - p_ii, ceiling = p_iii - p_ii; the copy predicts 0.  Skill = 1 - sum (d - prediction)^2 / sum d^2 (skill over persistence),
episode-cluster bootstrap (the counts of resprobe.boot_counts, B 2,000, seed 0); a pair A:B is the difference of the rollout skills on the same windows.
Controls: k1 the copy in the place of the rollout reads exactly 0; k2 the target shuffled across windows (fixed seed) reads at or below 0.02;
k3 the target set equal to the ceiling prediction reads exactly 1.  Exit 0 valid, 2 a control failed, 3 refused.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

ROOT = "F:/TanitAD-scratch/v8proof"
if ROOT + "/spatial/s0" not in sys.path:
    sys.path.insert(0, ROOT + "/spatial/s0")
import s0_read as S  # noqa: E402

RP = S.RP
TIER = "T0 (a world-model diagnostic on logged frames, never driving performance)"
FAMILIES = "LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)"


def load_arm(path, gap0_of):
    z = np.load(path, allow_pickle=True)
    eid = np.array([str(e) for e in z["w_eid"]])
    g0 = z["w_g0"].astype(np.int64)
    yk = z["yk"].astype(np.float64)
    gap0 = np.array([gap0_of.get((e, int(g)), np.nan) for e, g in zip(eid, g0)])
    P = {k: z["p_" + k].astype(np.float64) for k in ("i", "ii", "iii")}
    ok = np.isfinite(yk) & np.isfinite(gap0) & np.isfinite(P["i"]) & np.isfinite(P["ii"]) & np.isfinite(P["iii"])
    uep = [str(e) for e in z["uep"]]
    return {"eid": eid, "g0": g0, "d": yk - gap0, "roll": P["i"] - P["ii"], "ceil": P["iii"] - P["ii"], "ok": ok, "epi": z["w_epi"].astype(np.int64), "uep": uep}


def skill_boot(d, pred, epi, n_ep, C):
    a = np.bincount(epi, weights=(d - pred) ** 2, minlength=n_ep)
    b = np.bincount(epi, weights=d ** 2, minlength=n_ep)
    pt = 1.0 - a.sum() / b.sum()
    num, den = C @ a, C @ b
    bs = 1.0 - num / np.where(den > 0, den, np.nan)
    return float(pt), bs


def ci(bs):
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return [float(lo), float(hi)]


def cmd_run(a):
    Z = np.load(S.BANK)
    beid = Z["beid"].astype(str)
    bg = Z["bg"].astype(np.int64)
    gap = Z["lead_gap"].astype(np.float64)
    gap0_of = {(e, int(g)): float(v) for e, g, v in zip(beid, bg, gap)}
    arms = {}
    for p in a.oof:
        name = os.path.basename(p)[7:-4]
        arms[name] = load_arm(p, gap0_of)
    res = {"tier": TIER, "families": FAMILIES, "B": a.n_boot, "arms": {}, "pairs": {}, "controls": {}}
    ctl_ok = True
    for name, A in arms.items():
        ok = A["ok"]
        d, roll, ceil, epi = A["d"][ok], A["roll"][ok], A["ceil"][ok], A["epi"][ok]
        n_ep = len(A["uep"])
        C = RP.boot_counts(n_ep, a.n_boot, S.SEED).astype(np.float64)
        s_r, b_r = skill_boot(d, roll, epi, n_ep, C)
        s_c, b_c = skill_boot(d, ceil, epi, n_ep, C)
        k1, _ = skill_boot(d, np.zeros_like(d), epi, n_ep, C)
        rng = np.random.default_rng(7)
        dsh = d[rng.permutation(len(d))]
        k2r, _ = skill_boot(dsh, roll, epi, n_ep, C)
        k2c, _ = skill_boot(dsh, ceil, epi, n_ep, C)
        k3, _ = skill_boot(ceil, ceil, epi, n_ep, C)
        cr, cc = ci(b_r), ci(b_c)
        res["arms"][name] = {"n_windows": int(ok.sum()), "n_episodes_with_windows": int(len(set(epi.tolist()))), "target_rms_m": float(np.sqrt(np.mean(d ** 2))),
                             "rollout": {"skill": s_r, "ci95": cr, "above_0": bool(cr[0] > 0)}, "ceiling": {"skill": s_c, "ci95": cc, "above_0": bool(cc[0] > 0)},
                             "ceiling_minus_rollout": {"diff": s_c - s_r, "ci95": ci(b_c - b_r)}}
        c = {"k1_copy_is_0": {"pass": bool(k1 == 0.0), "value": k1}, "k2_shuffled_target": {"pass": bool(k2r <= 0.02 and k2c <= 0.02), "rollout": k2r, "ceiling": k2c, "bar": 0.02},
             "k3_identity_is_1": {"pass": bool(k3 == 1.0), "value": k3}}
        res["controls"][name] = c
        ctl_ok &= all(v["pass"] for v in c.values())
    for pr in a.pair or []:
        na, nb = pr.split(":")
        A, Bz = arms[na], arms[nb]
        if not (np.array_equal(A["eid"], Bz["eid"]) and np.array_equal(A["g0"], Bz["g0"])):
            print("REFUSED: windows differ between %s and %s" % (na, nb))
            return 3
        ok = A["ok"] & Bz["ok"]
        if not np.allclose(A["d"][ok], Bz["d"][ok]):
            print("REFUSED: targets differ between %s and %s" % (na, nb))
            return 3
        n_ep = len(A["uep"])
        C = RP.boot_counts(n_ep, a.n_boot, S.SEED).astype(np.float64)
        sa, ba = skill_boot(A["d"][ok], A["roll"][ok], A["epi"][ok], n_ep, C)
        sb, bb = skill_boot(Bz["d"][ok], Bz["roll"][ok], Bz["epi"][ok], n_ep, C)
        c_ = ci(ba - bb)
        res["pairs"][na + "_minus_" + nb] = {"rollout_skill_diff": sa - sb, "ci95": c_, "separated_pos": bool(c_[0] > 0), "separated_neg": bool(c_[1] < 0), "n_windows": int(ok.sum())}
    res["controls_pass"] = bool(ctl_ok)
    L = ["# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)", "", "Tier " + TIER + ". " + FAMILIES + ".",
         "Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B %d." % a.n_boot,
         "This read names nothing; the outcome names of the cards are decided on the level read.", "", "## Controls (first)", ""]
    for name, c in res["controls"].items():
        L.append("- %s: %s" % (name, "; ".join("%s %s %s" % (k, "PASS" if v["pass"] else "FAIL", json.dumps({x: y for x, y in v.items() if x != "pass"})) for k, v in c.items())))
    L += ["- controls_pass: %s" % res["controls_pass"], "", "## Change skill", "", "| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |", "|---|---|---|---|---|---|"]
    for name, r in res["arms"].items():
        L.append("| %s | %d | %.2f | %+.3f [%+.3f, %+.3f] | %+.3f [%+.3f, %+.3f] | %+.3f [%+.3f, %+.3f] |" % (
            name, r["n_windows"], r["target_rms_m"], r["rollout"]["skill"], *r["rollout"]["ci95"], r["ceiling"]["skill"], *r["ceiling"]["ci95"],
            r["ceiling_minus_rollout"]["diff"], *r["ceiling_minus_rollout"]["ci95"]))
    if res["pairs"]:
        L += ["", "## Pairs (rollout skill, same windows)", ""]
        for k, p in res["pairs"].items():
            L.append("- %s: %+.3f [%+.3f, %+.3f]%s" % (k.replace("_", " "), p["rollout_skill_diff"], p["ci95"][0], p["ci95"][1],
                     " (separated above 0)" if p["separated_pos"] else (" (separated below 0)" if p["separated_neg"] else "")))
    L += ["", "Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.", ""]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    with open(a.out[:-3] + ".json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("\n".join(L))
    return 0 if res["controls_pass"] else 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--oof", action="append", required=True)
    r.add_argument("--pair", action="append")
    r.add_argument("--out", required=True)
    r.add_argument("--n-boot", type=int, default=S.B_BOOT)
    return cmd_run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
