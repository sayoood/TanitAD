"""make_planted_joins.py -- the six planted joins of card PREREG_V8F_PROOF_ADD94 (GAPRULE10K). numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).
Usage: python make_planted_joins.py --eval-join NPZ --out-dir NEWDIR      writes G0, GR1, GR2, GW05, GC10, GC20 as <ARM>.npz and PLANTED_JOINS_RECORD.json
       python make_planted_joins.py --selftest                            synthetic join; the clean build passes control j1 and each planted fault turns it red
A planted join is the evaluation join with only `lead_gap` changed, and only on the frames that are valid for the rule's label source (lead_valid == 1, finite, 0 <= gap <= 60 m: the test of
LabelSource._ok in distrule/v8f_t1_eval_rule.py). Unit noise (card section 2): one draw per clip and kind, seeded by sha256 of "<clip>|<kind>|94", shifted and scaled to mean 0 and standard
deviation 1 EXACTLY over all valid frames. W white; R first-order autoregression 0.94 per frame within a clip (over all frames of the clip, kept where valid); C one value per clip.
Clips are seeded by clip NAME when the join has the array `clip` (the evaluation join has it), else by clip index (the text of the index). Arms: G0 = label gap (no change, no limits);
GR1 = gap + 1 m R; GR2 = gap + 2 m R; GW05 = gap + 0.5 m W; GC10 = gap * (1 + 0.10 C); GC20 = gap * (1 + 0.20 C). Limits (not for G0): below 0.5 m -> 0.5 m, above 60 m -> 60 m; the number of
values changed by the limits is recorded per arm. Refuses an existing out dir. The dtype of every key is kept (lead_gap stays as stored)."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

SEED = 94
GAP_MAX_M = 60.0
GAP_MIN_M = 0.5
AR_COEF = 0.94
ARMS = (("G0", "none", None, 0.0), ("GR1", "R", "absolute", 1.0), ("GR2", "R", "absolute", 2.0), ("GW05", "W", "absolute", 0.5), ("GC10", "C", "proportional", 0.10), ("GC20", "C", "proportional", 0.20))
TOL_STAT = 1e-9
LAG1 = {"W": ("abs_below", 0.05), "R": ("between", (0.89, 0.97)), "C": ("at_least", 0.999)}
MAX_LIMITED_FRACTION = 0.02


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def valid_mask(J):
    g = np.asarray(J["lead_gap"], np.float64)
    return (np.asarray(J["lead_valid"]) == 1) & np.isfinite(g) & (g >= 0.0) & (g <= GAP_MAX_M)


def clip_names(J):
    n = len(J["n"])
    if "clip" in J:
        return [str(c) for c in J["clip"]], "clip name (array clip of the join)"
    return [str(i) for i in range(n)], "clip index (the join has no array clip)"


def unit_noise(J, kind, names, faults=()):
    """[rows] unit noise of one kind, mean 0 and standard deviation 1 over all valid frames (exactly, up to rounding)."""
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    out = np.zeros(len(np.asarray(J["lead_gap"])))
    for i, nm in enumerate(names):
        h = int(hashlib.sha256(("%s|%s|%d" % (nm, kind, SEED)).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        o, m = int(off[i]), int(n[i])
        if m <= 0:
            continue
        if kind == "C":
            out[o:o + m] = rng.standard_normal()
        elif kind == "W":
            out[o:o + m] = rng.standard_normal(m)
        else:
            z = rng.standard_normal(m)
            e = np.zeros(m)
            e[0] = z[0]
            for t in range(1, m):
                e[t] = AR_COEF * e[t - 1] + np.sqrt(1.0 - AR_COEF ** 2) * z[t]
            out[o:o + m] = e
    v = valid_mask(J)
    out = (out - out[v].mean()) / out[v].std()
    if "no_normalise" in faults:
        out = out * 1.1
    return out


def lag1(N, J, v):
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    same = np.zeros(len(N), bool)                        # frame r and r-1 in the same clip
    for o, m in zip(off, n):
        if m > 1:
            same[int(o) + 1:int(o) + int(m)] = True
    pr = np.zeros(len(N), bool)
    pr[1:] = same[1:] & v[1:] & v[:-1]
    a, b = N[:-1][pr[1:]], N[1:][pr[1:]]
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None


def stat(x, v):
    return {"mean": float(np.mean(x[v])), "sd": float(np.std(x[v]))}


def equal_key(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind == "f" else bool(np.array_equal(a, b))


def build(J, out_dir, faults=()):
    """writes the six joins and the record. J: the evaluation join as a dict of arrays. -> record."""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    os.makedirs(out_dir)
    names, seeding = clip_names(J)
    v = valid_mask(J)
    gap = np.asarray(J["lead_gap"], np.float64)
    dt = np.asarray(J["lead_gap"]).dtype
    noise = {k: unit_noise(J, k, names, faults) for k in ("W", "R", "C")}
    rec = {"card": "PREREG_V8F_PROOF_ADD94", "seed": SEED, "seeding": seeding, "n_clips": int(len(J["n"])), "n_rows": int(len(gap)), "n_valid_frames": int(v.sum()), "gap_min_m": GAP_MIN_M,
           "gap_max_m": GAP_MAX_M, "valid_rule": "lead_valid == 1, finite, 0 <= lead_gap <= 60", "unit_noise": {}, "arms": {}, "faults_for_selftest": sorted(faults)}
    for k in ("W", "R", "C"):
        rec["unit_noise"][k] = dict(stat(noise[k], v), lag1=lag1(noise[k], J, v))
    for arm, kind, typ, lvl in ARMS:
        if kind == "none":
            pre = gap.copy()
            err = np.zeros(len(gap))
            rel = np.zeros(len(gap))
        else:
            N = noise[kind]
            if typ == "absolute":
                pre = gap + lvl * N
            elif "prop_as_abs" in faults:
                pre = gap + lvl * N
            else:
                pre = gap * (1.0 + lvl * N)
            err = pre - gap
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(gap > 0.0, err / gap, lvl * N)          # where the label gap is 0 the ratio is undefined: the noise term itself
        post = pre.copy()
        n_lim = 0
        if kind != "none":
            lim = np.clip(pre, GAP_MIN_M, GAP_MAX_M)
            n_lim = int(((lim != pre) & v).sum())
            post = np.where(v, lim, gap)
        new = np.where(v, post, gap).astype(dt)
        K = dict(J)
        K["lead_gap"] = new
        if "other_field" in faults and arm == "GR2":
            K["lead_valid"] = np.asarray(J["lead_valid"]).copy()
            K["lead_valid"][0] = 1 - K["lead_valid"][0]
        path = os.path.join(out_dir, arm + ".npz")
        np.savez(path, **{k: x for k, x in K.items() if isinstance(x, np.ndarray)}, n_clips_missing_labels=np.int64(J.get("n_clips_missing_labels", 0)))
        with np.load(path, allow_pickle=False) as z:
            back = {k: z[k] for k in z.files}
        eq = {k: equal_key(back[k], J[k]) for k in J if isinstance(J[k], np.ndarray) and k != "lead_gap"}
        outside = int((np.asarray(back["lead_gap"], np.float64)[~v] != gap[~v]).sum() - (np.isnan(gap[~v]) & np.isnan(np.asarray(back["lead_gap"], np.float64)[~v])).sum())
        e = rel if typ == "proportional" else err
        rec["arms"][arm] = {"kind": kind, "type": typ or "none", "level": lvl, "planted_error_before_limits": stat(e, v) if kind != "none" else {"mean": 0.0, "sd": 0.0},
                            "unit_noise_lag1": rec["unit_noise"][kind]["lag1"] if kind != "none" else None, "n_valid_frames": int(v.sum()), "n_changed_by_limits": n_lim,
                            "fraction_changed_by_limits": float(n_lim / max(int(v.sum()), 1)), "n_frames_changed_in_file": int((np.asarray(back["lead_gap"], np.float64)[v] != gap[v]).sum()),
                            "n_changed_outside_valid_frames": outside, "other_keys_equal_to_input": eq, "other_keys_all_equal": bool(all(eq.values())), "file": arm + ".npz", "sha256": sha_file(path)}
    rec["j1"] = j1_check(rec)
    with open(os.path.join(out_dir, "PLANTED_JOINS_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def j1_check(rec):
    """control j1 of the card from a record: -> {"PASS": bool, "failures": [text]}"""
    bad = []
    try:
        for arm, kind, typ, lvl in ARMS:
            if kind == "none":
                continue
            a = rec["arms"][arm]
            st = a["planted_error_before_limits"]
            if not abs(st["sd"] - lvl) <= TOL_STAT:
                bad.append("%s: standard deviation %.12g is not the level %g within 1e-9" % (arm, st["sd"], lvl))
            if not abs(st["mean"]) <= TOL_STAT:
                bad.append("%s: mean %.3g is not 0 within 1e-9" % (arm, st["mean"]))
            l1 = a["unit_noise_lag1"]
            mode, bound = LAG1[kind]
            ok = l1 is not None and ((mode == "abs_below" and abs(l1) < bound) or (mode == "between" and bound[0] <= l1 <= bound[1]) or (mode == "at_least" and l1 >= bound))
            if not ok:
                bad.append("%s: lag-1 autocorrelation %s of the unit noise (%s) is outside its band" % (arm, l1, kind))
            if a["other_keys_all_equal"] is not True or a["n_changed_outside_valid_frames"] != 0:
                bad.append("%s: another field or an invalid frame differs from the evaluation join" % arm)
            if not a["fraction_changed_by_limits"] <= MAX_LIMITED_FRACTION:
                bad.append("%s: %.4f of the valid frames are changed by the limits (bar 0.02)" % (arm, a["fraction_changed_by_limits"]))
        g0 = rec["arms"]["G0"]
        if g0["n_frames_changed_in_file"] != 0 or g0["other_keys_all_equal"] is not True:
            bad.append("G0: differs from the evaluation join")
    except (KeyError, TypeError) as ex:
        bad.append("record incomplete: %s: %s" % (type(ex).__name__, ex))
    return {"PASS": not bad, "failures": bad}


def synthetic(seed=3, n_clips=14):
    rng = np.random.default_rng(seed)
    n = rng.integers(60, 140, n_clips).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64)
    T = int(n.sum())
    valid = (rng.random(T) < 0.8).astype(np.int8)
    gap = rng.uniform(4.0, 40.0, T).astype(np.float32)
    gap[rng.random(T) < 0.02] = np.nan
    return {"clip": np.array(["c%02d_%s" % (i, "ab" * (i % 3 + 1)) for i in range(n_clips)]), "off": off, "n": n, "lead_valid": valid, "lead_gap": gap,
            "speed": rng.normal(10, 2, T).astype(np.float32), "has": np.ones(T, bool), "n_clips_missing_labels": 0}


def selftest(here):
    base = os.path.join(here, "tmp_selftest_%d" % os.getpid())
    os.makedirs(base)
    J = synthetic()
    res, ok = {}, True
    r = build(J, os.path.join(base, "clean"))
    res["clean"] = r["j1"]
    ok = ok and r["j1"]["PASS"]
    print("SELFTEST clean build: j1 %s (valid frames %d)" % ("PASS" if r["j1"]["PASS"] else "FAIL " + str(r["j1"]["failures"]), r["n_valid_frames"]))
    for f in ("no_normalise", "prop_as_abs", "other_field"):
        r = build(J, os.path.join(base, "f_" + f), faults=(f,))
        caught = not r["j1"]["PASS"]
        ok = ok and caught
        print("SELFTEST fault %-13s caught by j1: %s" % (f, caught))
    again = None
    try:
        build(J, os.path.join(base, "clean"))
    except SystemExit as ex:
        again = str(ex)
    ok = ok and again is not None and again.startswith("REFUSED")
    print("SELFTEST existing out dir refused: %s" % (again is not None))
    print("SELFTEST %s (temp folder %s)" % ("PASS" if ok else "FAIL", base))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--eval-join")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(os.path.dirname(os.path.abspath(__file__)))
    if not a.eval_join or not a.out_dir:
        sys.exit("REFUSED: --eval-join and --out-dir are required")
    with np.load(a.eval_join, allow_pickle=False) as z:
        J = {k: z[k] for k in z.files}
    J["n_clips_missing_labels"] = int(J.get("n_clips_missing_labels", 0))
    rec = build(J, a.out_dir)
    rec["eval_join_sha256"] = sha_file(a.eval_join)
    with open(os.path.join(a.out_dir, "PLANTED_JOINS_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    print("ZZ PLANTED-JOINS j1 %s" % ("PASS" if rec["j1"]["PASS"] else "FAIL"))
    for f in rec["j1"]["failures"]:
        print("   " + f)
    return 0 if rec["j1"]["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
