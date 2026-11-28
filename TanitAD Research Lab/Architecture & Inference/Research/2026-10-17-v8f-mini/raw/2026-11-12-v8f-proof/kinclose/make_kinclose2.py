"""make_kinclose2.py -- writes kinclose2.py (card PREREG_V8F_PROOF_ADD89, KINCLOSE2) from kinclose.py (card ADD88 with its note N1) by counted edits.
Changes: control k1 in the form of card ADD89 (five planted faults of the ego path, two of them scale faults; the pair B_25 against BS_25 is an outcome), control k5 and the planted
gap-noise arms, the error of the reader's gap against the label gap, the names of card ADD89. Nothing of the slopes, the arms K, S, E, B, BS, the scored frames or the bootstrap changes."""
import hashlib, io, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_SHA = "7cd329bd3ecaf8ee"

NEW_FUNCS = '''def plant_noise(ctx, sigma, kind):
    """noise [clips, T] of one planted arm: one draw per (clip id, kind, sigma), the same for both reader seeds; white, or a stationary first-order autoregression of PLANT_AR per frame."""
    T = ctx["T"]
    out = np.zeros((len(ctx["clips"]), T))
    mult = 2.0 if "plant_sd_double" in FAULTS else 1.0
    for i, c in enumerate(ctx["clips"]):
        h = int(hashlib.sha256(("%s|%s|%.4f|%d" % (c, kind, sigma, PLANT_SEED)).encode("utf-8")).hexdigest()[:8], 16)
        z = np.random.default_rng(h).standard_normal(T)
        if kind == "smooth":
            e = np.zeros(T)
            e[0] = z[0]
            for t in range(1, T):
                e[t] = PLANT_AR * e[t - 1] + np.sqrt(1.0 - PLANT_AR ** 2) * z[t]
            z = e
        out[i] = sigma * mult * z
    return out


def planted(ctx, W, sigma, kind):
    """slope_W(label gap + planted noise) at every scored row; a value only where the label gap is present on all W frames (the frames of BS_W)."""
    sl, cnt = slope_rows(ctx["LG"] + plant_noise(ctx, sigma, kind), ctx["LM"], ctx["ci"], ctx["fr"], W, ctx["dt"])
    return np.where(cnt == W, sl, np.nan)


def control_k5(ctxs):
    """planted arms: sigma 0 equals BS_W; the realised spread is within 10 percent of sigma on the frames with a label gap; lag-1 of the smooth noise in 0.85..0.95, of the white noise within 0.05 of zero."""
    out, ok, worst = {"levels": {}}, True, 0.0
    for W in PLANT_WS:
        for c in ctxs:
            a, b = planted(c, W, 0.0, "white"), arm(c, "BS", W)
            fa, fb = np.isfinite(a), np.isfinite(b)
            ok = ok and bool((fa == fb).all()) and bool(fa.any())
            if (fa & fb).any():
                worst = max(worst, float(np.abs(a[fa & fb] - b[fa & fb]).max()))
    out["sigma0_max_abs_diff"] = worst
    ok = ok and bool(worst <= K5_TOL)
    c = ctxs[0]
    LM = c["LM"]
    pr = LM[:, 1:] & LM[:, :-1]
    for kind in ("white", "smooth"):
        for sg, code in SIGMAS:
            nz = plant_noise(c, sg, kind)
            sd = float(nz[LM].std())
            l1 = float(CF.pearson(nz[:, :-1][pr], nz[:, 1:][pr]))
            good = bool(abs(sd / sg - 1.0) <= 0.10 and ((0.85 <= l1 <= 0.95) if kind == "smooth" else abs(l1) <= 0.05))
            out["levels"]["%s_%s" % (kind, code)] = {"sd": sd, "sigma": sg, "lag1": l1, "pass": good}
            ok = ok and good
    out["pass"] = bool(ok)
    return out


def gap_error(ctxs):
    """error of the reader's raw gap read against the label gap on the frames of the scored windows (the 25 frames that end at a scored row) where both exist; per reader seed."""
    out = []
    for c in ctxs:
        U = np.zeros(c["G"].shape, bool)
        for k in range(H_SCORE):
            f = c["fr"] - k
            U[c["ci"][f >= 0], f[f >= 0]] = True
        m = U & c["RM"] & c["LM"]
        e = np.where(m, c["G"] - c["LG"], 0.0)
        pr = m[:, 1:] & m[:, :-1]
        n = int(m.sum())
        out.append({"n": n, "mean_m": float(e[m].mean()) if n else float("nan"), "sd_m": float(e[m].std()) if n else float("nan"),
                    "mean_abs_m": float(np.abs(e[m]).mean()) if n else float("nan"),
                    "lag1": float(CF.pearson(e[:, :-1][pr], e[:, 1:][pr])) if int(pr.sum()) >= 3 else float("nan")})
    return out


# ============================================================================================ names
def tri(p, pos, neg, neither):
    if p and p["ci"][1] < 0:
        return pos
    if p and p["ci"][0] > 0:
        return neg
    return neither


def names_from(arms, paired_):
    names = [tri(paired_["B25-BS25"], "KINCLOSE2-EGO-TERM-ON-THE-LABEL-GAP-HELPS", "KINCLOSE2-EGO-TERM-ON-THE-LABEL-GAP-HURTS", "KINCLOSE2-EGO-TERM-ON-THE-LABEL-GAP-NOT-SEPARATED"),
             tri(paired_["K25-S25"], "KINCLOSE2-EGO-TERM-HELPS", "KINCLOSE2-EGO-TERM-HURTS", "KINCLOSE2-EGO-TERM-NOT-SEPARATED")]
    kb = paired_["K25-L25"]
    r_ok = bool(arms.get("K25")) and all(np.isfinite(x) and x >= BAR_R_SEED for x in arms["K25"]["r_per_seed"])
    names.append("KINCLOSE2-K25-BETTER-THAN-OUTPUTS" if (kb and kb["ci"][1] < 0 and r_ok) else "KINCLOSE2-K25-NOT-BETTER")
    lows = [arms[n]["r_ci"][0] for n in ("K25", "S25") if arms.get(n)]
    alls = [arms.get("%s%d" % (k, w)) for k in ("K", "S") for w in WS]
    if any(x >= REQ_R for x in lows):
        names.append("KINCLOSE2-READER-GAP-REACHES-THE-REQUIREMENT")
    elif all(a and a["r_ci"][1] < REQ_R for a in alls):
        names.append("KINCLOSE2-READER-GAP-BELOW-THE-REQUIREMENT")
    else:
        names.append("KINCLOSE2-READER-GAP-NOT-DECIDED")
    for W in PLANT_WS:
        b = arms.get("BS%d" % W)
        if b and b["r_ci"][0] >= REQ_R:
            names.append("KINCLOSE2-WINDOW-%d-REACHES-THE-REQUIREMENT" % W)
            best = None
            for sg, code in SIGMAS:
                a = arms.get("PS%d_%s" % (W, code))
                if a and a["r_ci"][0] >= REQ_R:
                    best = code
                else:
                    break
            names.append(("KINCLOSE2-GAP-ERROR-W%d-UP-TO-%s" % (W, best)) if best else ("KINCLOSE2-GAP-ERROR-W%d-BELOW-S025" % W))
        elif b and b["r_ci"][1] < REQ_R:
            names.append("KINCLOSE2-WINDOW-%d-BELOW-THE-REQUIREMENT" % W)
        else:
            names.append("KINCLOSE2-WINDOW-%d-NOT-DECIDED" % W)
    return names
'''

EDITS = [
    ('CARD = "PREREG_V8F_PROOF_ADD88"\n',
     'CARD = "PREREG_V8F_PROOF_ADD89"\nSIGMAS = ((0.25, "S025"), (0.5, "S050"), (1.0, "S100"), (2.0, "S200"))\nPLANT_WS = (10, 25)\nPLANT_AR = 0.9\nPLANT_SEED = 89\nK5_TOL = 1e-9\n'),
    ('    if pathmod == "flip":\n        s = -s\n',
     '    if "ego_kmh" in FAULTS:\n        s = s * 3.6\n    if pathmod == "flip":\n        s = -s\n    elif pathmod == "scale_up":\n        s = s * 3.6\n    elif pathmod == "scale_down":\n        s = s / 3.6\n'),
    ('    return {"G": G, "RM": RM,', '    return {"clips": [str(c) for c in D["clips"]], "G": G, "RM": RM,'),
    ('    mods = {m: [arm(c, "B", 25, m) for c in ctxs] for m in ("shift+", "shift-", "flip")}\n'
     '    masks = [fin(base[i], bs[i], mods["shift+"][i], mods["shift-"][i], mods["flip"][i]) for i in range(len(ctxs))]\n',
     '    mods = {m: [arm(c, "B", 25, m) for c in ctxs] for m in ("shift+", "shift-", "flip", "scale_up", "scale_down")}\n'
     '    masks = [fin(base[i], bs[i], *[mods[m][i] for m in mods]) for i in range(len(ctxs))]\n'),
    ('    pr = paired(base, bs, rl, masks, B, BOOT_SEED)\n    out["B25_lower_than_BS25_separated"] = bool(pr["diff_abs_err"] < 0 and pr["ci"][1] < 0)\n',
     '    out["form"] = "card ADD89: five planted faults of the ego path; the pair B_25 against BS_25 is an outcome of the card and is not judged here"\n'),
    ('    est["L25"] = cell_est(ctxs, est_paths)\n',
     '    est["L25"] = cell_est(ctxs, est_paths)\n    for W in PLANT_WS:\n        for sg, code in SIGMAS:\n            est["PW%d_%s" % (W, code)] = [planted(c, W, sg, "white") for c in ctxs]\n'
     '            est["PS%d_%s" % (W, code)] = [planted(c, W, sg, "smooth") for c in ctxs]\n'),
    ('("B25-BS25", "B25", "BS25")] + [', '("B25-BS25", "B25", "BS25"), ("S25-L25", "S25", "L25")] + ['),
    ('"k4": control_k4()}', '"k4": control_k4(), "k5": control_k5(ctxs)}'),
    ('    A, P = all_arms(ctxs, ps["est"], a.boot)\n', '    A, P = all_arms(ctxs, ps["est"], a.boot)\n    GE = gap_error(ctxs)\n'),
    ('        res.update({"arms": A, "paired": P, "names": names_from(A, P)})\n', '        res.update({"arms": A, "paired": P, "gap_error": GE, "names": names_from(A, P)})\n'),
    ('        res.update({"names": [], "withheld": {"arms": A, "paired": P}})\n', '        res.update({"names": [], "withheld": {"arms": A, "paired": P, "gap_error": GE}})\n'),
    ('    print("KINCLOSE (card ADD88, tier T0)")\n', '    print("KINCLOSE2 (card ADD89, tier T0)")\n'),
    ('v["ci"][0], v["ci"][1], v["n_frames"])))\n    return 0\n',
     'v["ci"][0], v["ci"][1], v["n_frames"])))\n    for i, g in enumerate(R.get("gap_error", [])):\n'
     '        print("gap error of the reader against the label gap, reader seed index %d: n %d mean %+.2f m sd %.2f m mean abs %.2f m lag1 %+.3f" % (i, g["n"], g["mean_m"], g["sd_m"], g["mean_abs_m"], g["lag1"]))\n'
     '    return 0\n'),
    ('chk("signal world: rc 0 and all five controls pass (k4 planted included)", rc == 0 and C["controls_all_pass"] and C["n_passed"] == 5)',
     'chk("signal world: rc 0 and all six controls pass (k4 and k5 planted included)", rc == 0 and C["controls_all_pass"] and C["n_passed"] == 6)'),
    ('"KINCLOSE-K25-BETTER-THAN-OUTPUTS" in R["names"] and "KINCLOSE-EGO-TERM-HELPS" in R["names"])',
     '"KINCLOSE2-K25-BETTER-THAN-OUTPUTS" in R["names"] and "KINCLOSE2-EGO-TERM-HELPS" in R["names"])'),
    ('        chk("k4 planted world: signal reads better with K, noise stays at E", C["controls"]["k4"]["pass"])\n',
     '        chk("k4 planted world: signal reads better with K, noise stays at E", C["controls"]["k4"]["pass"])\n'
     '        chk("k5: the planted arms sit on the frames of BS and their noise has the planted spread and time course", C["controls"]["k5"]["pass"])\n'
     '        chk("names: one window name per planted window, the label-gap ego name and a reader-gap name", sum(1 for n in R["names"] if "-WINDOW-" in n) == 2 and sum(1 for n in R["names"] if "ON-THE-LABEL-GAP" in n) == 1 and sum(1 for n in R["names"] if "-READER-GAP-" in n) == 1)\n'
     '        chk("planted ladder: with white noise over 10 frames the largest gap noise reads worse than the smallest (r and error spread)", R["arms"]["PW10_S025"]["r_pooled"] > R["arms"]["PW10_S200"]["r_pooled"] and R["arms"]["PW10_S200"]["err_sd"] > R["arms"]["PW10_S025"]["err_sd"])\n'
     '        chk("gap error of the reader is in the result, one record per reader seed", len(R["gap_error"]) == 2 and all(g["n"] > 0 for g in R["gap_error"]))\n'),
    ('(("ego_sign", "k1"),("ego_shift", "k1"),', '(("ego_sign", "k1"),("ego_shift", "k1"), ("ego_kmh", "k1"), ("plant_sd_double", "k5"),'),
]


def main():
    raw = io.open(os.path.join(HERE, "kinclose.py"), "rb").read().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest()[:16] != PARENT_SHA:
        sys.exit("REFUSED: parent hash is not %s" % PARENT_SHA)
    s = raw.decode("utf-8")
    for old, new in EDITS:
        if s.count(old) != 1:
            sys.exit("REFUSED: %d matches of %r" % (s.count(old), old[:70]))
        s = s.replace(old, new)
    s, n = re.subn(r"# =+ names\ndef names_from\(arms, paired_\):.*?\n    return names\n", lambda m: NEW_FUNCS, s, flags=re.S)
    if n != 1:
        sys.exit("REFUSED: %d matches of the names block" % n)
    if s.count('"""kinclose.py') != 1:
        sys.exit("REFUSED: the parent's docstring start is not found once")
    s = s.replace('"""kinclose.py', '"""kinclose2.py -- card PREREG_V8F_PROOF_ADD89 (KINCLOSE2), written by make_kinclose2.py from kinclose.py %s of card ADD88: control k1 as five planted faults of the ego path,\n'
                  'control k5 and the planted gap-noise arms, the error of the reader gap, the names of card ADD89. Text of the parent follows.\nkinclose.py' % PARENT_SHA, 1)
    out = os.path.join(HERE, "kinclose2.py")
    io.open(out, "w", encoding="utf-8", newline="\n").write(s)
    print("wrote", out, hashlib.sha256(s.encode("utf-8")).hexdigest()[:16])


if __name__ == "__main__":
    main()
