"""read_gapspread_t0.py -- card PREREG_V8F_PROOF_ADD132 (GAPSPREAD-T0): does the disagreement of the three gap fits of card ADD106 place a margin better than a flat one?

Tier T0 (recorded frames, no rollout; a diagnostic of a perception read and not driving performance). CPU only.
The tool of card ADD131 (make_joined_gapfiles.py) is imported unchanged for the row map and the label rows of its T0 read.

  controls  -> GAPSPREAD_CONTROLS.json, lines 'ctl s<i> PASS/FAIL'; prints no value of the card
  name      -> refuses unless the controls file is all-pass for the same input files; GAPSPREAD_RESULT.json and the name
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import io
import json
import math
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD132"
MJ_PREFIX = "91e599330bae7ca9"
TIER_T0 = "T0 (recorded frames, no rollout; a diagnostic of a perception read and not driving performance)"
DOSE = 0.90
FLOOR = 0.60
BAR = 0.90
SEED = 132
N_BOOT = 2000
FIT_NAMES = ("VGAPL_s2", "VGAPL_s3", "VGAPL_s4")
BASE_NAME = "VGAPLE"
SAME_KEYS = ("clip_index", "frame", "has_out", "clips")
MEAN_TOL = 1e-3
DOSE_TOL = 1e-6
ROW_TOL = 1e-9
EXPECT = {"rows": 3841, "clips": 141, "e_base": 3.132}
E_TOL = 1e-3
PLANT_ADD = 0.01
PERM_NEAR = 0.03
FAMILY_LINES = (
    "LONGITUDINAL: the long part and the short part of the gap input against the label gap, on recorded frames.",
    "LATERAL: not read by this card (no rollout, no plan; n = 0).",
    "TACTICAL: not read by this card (no rollout, no plan; n = 0).",
    "STRATEGIC: not read by this card (no rollout, no route; n = 0).",
)


def load_mj():
    m = sys.modules.get("make_joined_gapfiles")
    if m is not None and getattr(m, "__file__", None) and os.path.isfile(m.__file__):
        return _checked(m, m.__file__)
    for p in (os.environ.get("JOINED_GAPFILES_TOOL"), "/workspace/v8proof/boxfuse10k/kit_gap/make_joined_gapfiles.py", "F:/TanitAD-scratch/v8proof/boxfuse10k/gap/make_joined_gapfiles.py"):
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("make_joined_gapfiles", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["make_joined_gapfiles"] = m
            spec.loader.exec_module(m)
            return _checked(m, p)
    raise SystemExit("REFUSED: make_joined_gapfiles.py not found (set JOINED_GAPFILES_TOOL)")


def _checked(m, p):
    if not m.sha_file(p).startswith(MJ_PREFIX):
        raise SystemExit("REFUSED: %s is not the tool of card ADD131 (%s)" % (p, MJ_PREFIX))
    return m


MJ = load_mj()
RB, BS = MJ.RB, MJ.BS


# ============================================================================================ the arms
def spread_of(fits, m):
    """fits [3, rows], m [rows] -> relative spread (largest - smallest) / m; not a number where m is not a positive number"""
    f = np.asarray(fits, np.float64)
    m = np.asarray(m, np.float64)
    s = np.full(m.shape, np.nan)
    ok = np.isfinite(m) & (m > 0) & np.isfinite(f).all(axis=0)
    s[ok] = (f[:, ok].max(axis=0) - f[:, ok].min(axis=0)) / m[ok]
    return s


def factor_of(s, k):
    return np.maximum(FLOOR, 1.0 - k * np.asarray(s, np.float64))


def solve_k(s_out):
    """the one k with mean(max(FLOOR, 1 - k s)) = DOSE over the rows given (bisection); None when no k reaches it"""
    s = np.asarray(s_out, np.float64)
    if s.size == 0 or not np.isfinite(s).all() or (s < 0).any():
        return None
    pos = s[s > 0]
    if pos.size == 0:
        return None
    hi = (1.0 - FLOOR) / float(pos.min()) * 2.0 + 1.0
    if float(factor_of(s, hi).mean()) > DOSE:
        return None
    lo = 0.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if float(factor_of(s, mid).mean()) > DOSE:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sp_gap(m, s, k, fault=None):
    g = np.asarray(m, np.float64) * factor_of(s, k)
    if fault == "shift":
        g = g.copy()
        g[np.flatnonzero(np.isfinite(g))[0]] += 0.01
    return g


def sp_gap_rows(m, s, k):
    """a second routine, row by row"""
    out = np.full(len(m), np.nan)
    for i in range(len(m)):
        if math.isfinite(float(m[i])) and math.isfinite(float(s[i])):
            f = 1.0 - k * float(s[i])
            out[i] = float(m[i]) * (f if f > FLOOR else FLOOR)
    return out


def long_part(g, lab):
    return np.maximum(g - lab, 0.0)


def short_part(g, lab):
    return np.maximum(lab - g, 0.0)


def clip_sums(x, ci, nc):
    return np.bincount(ci, weights=np.asarray(x, np.float64), minlength=nc)


def ratio_iv(num_rows, den_rows, ci, nc):
    n, d = clip_sums(num_rows, ci, nc), clip_sums(den_rows, ci, nc)
    r = float(n.sum() / d.sum()) if d.sum() > 0 else float("nan")
    return r, BS.boot_ratio(n, d, N_BOOT, SEED)


def rank_avg(x):
    u, inv, cnt = np.unique(np.asarray(x, np.float64), return_inverse=True, return_counts=True)
    top = np.cumsum(cnt).astype(np.float64)
    return (top - (cnt - 1) / 2.0)[inv]


def rank_corr(a, b):
    ra, rb = rank_avg(a), rank_avg(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def rank_corr_iv(a, b, ci, nc):
    rows_of = [np.flatnonzero(ci == c) for c in range(nc)]
    idx = BS.boot_idx(nc, N_BOOT, SEED)
    out = []
    for d in idx:
        r = np.concatenate([rows_of[c] for c in d])
        if r.size > 2:
            v = rank_corr(a[r], b[r])
            if math.isfinite(v):
                out.append(v)
    if len(out) < N_BOOT // 2:
        return None
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def name_of(rl, iv_l, iv_s):
    """-> (name, bar name or None)"""
    if iv_l is None or iv_s is None:
        return "GAPSPREAD-T0-ALLOCATION-NOT-SEPARATED", None
    if iv_l[1] < 1.0:
        if iv_s[0] > 1.0:
            return "GAPSPREAD-T0-ALLOCATION-TRADES", None
        return "GAPSPREAD-T0-ALLOCATION-BETTER", ("GAPSPREAD-T0-BAR-MET" if rl <= BAR else "GAPSPREAD-T0-BAR-NOT-MET")
    if iv_l[0] > 1.0:
        return "GAPSPREAD-T0-ALLOCATION-WORSE", None
    return "GAPSPREAD-T0-ALLOCATION-NOT-SEPARATED", None


# ============================================================================================ the world
class World:
    def __init__(self, x, a):
        self.paths = {BASE_NAME: x.base}
        for nm, p in zip(FIT_NAMES, x.fit):
            self.paths[nm] = p
        self.files = {k: MJ.load(p) for k, p in self.paths.items()}
        self.sha = {k: MJ.sha_file(p) for k, p in self.paths.items()}
        B = self.files[BASE_NAME]
        self.has = np.asarray(B["has_out"], bool)
        self.m = np.asarray(B["gap"], np.float64)
        self.fits = np.stack([np.asarray(self.files[k]["gap"], np.float64) for k in FIT_NAMES])
        self.s = spread_of(self.fits, self.m)
        J = BS.load_npz(a.eval_join)
        L = {k: BS.lay_dump(J, BS.load_npz(p)) for k, p in RB.dump_paths(a).items()}
        RM = MJ.RowMap(J, B)
        t3 = BS.t3_rows(L)
        sel = (RM.join_row >= 0) & self.has
        sel[sel] &= t3[RM.join_row[sel]]
        self.rows = np.flatnonzero(sel)
        self.lab = np.asarray(L["D_0"]["lab_gap"], np.float64)[RM.join_row[self.rows]]
        jc = {c: i for i, c in enumerate(RM.jclips)}
        self.ci = np.array([jc[RM.clip_of_row[r]] for r in self.rows], np.int64)
        self.nc = len(RM.jclips)
        self.n_clips_with_rows = int(len(set(self.ci.tolist())))
        self.unknown = int(RM.unknown)

    def arms(self, s=None, fault=None):
        """-> (k, gaps on all rows {A90, SP90}); s defaults to the real spread"""
        s = self.s if s is None else s
        k = solve_k(s[self.has])
        if k is None:
            return None, None
        return k, {"A90": DOSE * self.m, "SP90": sp_gap(self.m, s, k, fault)}

    def rl_rs(self, gaps):
        a, p = gaps["A90"][self.rows], gaps["SP90"][self.rows]
        rl, iv_l = ratio_iv(long_part(p, self.lab), long_part(a, self.lab), self.ci, self.nc)
        rs, iv_s = ratio_iv(short_part(p, self.lab), short_part(a, self.lab), self.ci, self.nc)
        return rl, iv_l, rs, iv_s


# ============================================================================================ controls
def ctl_s0(W, rec106):
    bad = []
    files = (rec106 or {}).get("files") or {}
    for k in (BASE_NAME,) + FIT_NAMES:
        want = (files.get(k) or {}).get("file_sha256")
        if not want or want != W.sha[k]:
            bad.append("%s does not carry the hash of the record of card ADD106" % k)
    B = W.files[BASE_NAME]
    for k in FIT_NAMES:
        for key in SAME_KEYS:
            if key not in W.files[k] or not np.array_equal(np.asarray(W.files[k][key]), np.asarray(B[key])):
                bad.append("%s differs from the base file in %s" % (k, key))
    dev = None
    if not bad:
        mm = W.fits[:, W.has].mean(axis=0)
        if not np.isfinite(mm).all() or not np.isfinite(W.m[W.has]).all():
            bad.append("a row with an output has a gap that is not a number")
        else:
            dev = float(np.abs(mm - W.m[W.has]).max())
            if dev > MEAN_TOL:
                bad.append("the mean of the three fits is not the base gap (largest deviation %.3g m)" % dev)
    if W.unknown:
        bad.append("%d rows of the base file are not rows of the join" % W.unknown)
    return {"pass": not bad, "failures": bad, "max_abs_dev_mean_minus_base_m": dev, "n_rows": int(len(W.m)), "n_rows_with_output": int(W.has.sum())}


def ctl_s1(W, expect):
    bad = []
    e = float(np.abs(W.m[W.rows] - W.lab).mean()) if len(W.rows) else float("nan")
    if len(W.rows) != expect["rows"]:
        bad.append("label rows %d, expected %d" % (len(W.rows), expect["rows"]))
    if W.nc != expect["clips"]:
        bad.append("clips %d, expected %d" % (W.nc, expect["clips"]))
    if not (abs(e - expect["e_base"]) <= E_TOL):
        bad.append("the mean absolute error of the base file is not the one of row 200")
    return {"pass": not bad, "failures": bad, "n_rows": int(len(W.rows)), "n_clips": W.nc, "n_clips_with_rows": W.n_clips_with_rows, "expected": expect}


def ctl_s2(W, fault=None):
    bad = []
    k, G = W.arms(fault=fault)
    if k is None:
        return {"pass": False, "failures": ["no k reaches the dose with the real spread"]}
    dose = float(factor_of(W.s[W.has], k).mean())
    if abs(dose - DOSE) > DOSE_TOL:
        bad.append("the mean factor is %.8f" % dose)
    dev2 = float(np.nanmax(np.abs(G["SP90"][W.has] - sp_gap_rows(W.m, W.s, k)[W.has])))
    if not dev2 <= ROW_TOL:
        bad.append("the second routine differs by %.3g m" % dev2)
    c = np.where(W.has, 0.05, np.nan)
    kc, Gc = W.arms(s=c)
    devc = None
    if kc is None:
        bad.append("no k reaches the dose with a constant spread")
    else:
        devc = float(np.abs(Gc["SP90"][W.has] - Gc["A90"][W.has]).max())
        if not devc <= ROW_TOL:
            bad.append("with a constant spread SP90 differs from A90 by %.3g m" % devc)
    return {"pass": not bad, "failures": bad, "mean_factor": dose, "max_abs_dev_second_routine_m": dev2, "max_abs_dev_constant_spread_m": devc}


def planted_spread(W):
    s = np.where(W.has, PLANT_ADD, np.nan)
    s[W.rows] = np.maximum((W.m[W.rows] - W.lab) / W.lab, 0.0) + PLANT_ADD
    return s


def permuted_spread(W):
    rng = np.random.RandomState(SEED)
    s = W.s.copy()
    s[W.rows] = W.s[W.rows][rng.permutation(len(W.rows))]
    other = np.flatnonzero(W.has)
    other = other[~np.isin(other, W.rows)]
    s[other] = W.s[other][rng.permutation(len(other))]
    return s


def ctl_s3(W):
    bad = []
    out = {}
    kp, Gp = W.arms(s=planted_spread(W))
    if kp is None:
        bad.append("no k reaches the dose with the planted spread")
    else:
        rl = W.rl_rs(Gp)[0]
        out["planted_reads_below_the_bar"] = bool(rl < BAR)
        if not rl < BAR:
            bad.append("the planted spread is not read (the ratio of the long part is not below the bar)")
    km, Gm = W.arms(s=permuted_spread(W))
    if km is None:
        bad.append("no k reaches the dose with the permuted spread")
    else:
        rl, iv = W.rl_rs(Gm)[:2]
        ok = (iv is not None and iv[0] <= 1.0 <= iv[1]) or abs(rl - 1.0) <= PERM_NEAR
        out["permuted_reads_no_mark"] = bool(ok)
        if not ok:
            bad.append("the permuted spread reads as a mark")
    out.update({"pass": not bad, "failures": bad})
    return out


def ctl_s4(paths):
    bad = []
    seen = []
    for p in paths:
        t = BS.read_text(p)
        if t is None:
            bad.append("test output %s is missing" % os.path.basename(p))
            continue
        last = [ln for ln in t.replace("\r", "").split("\n") if ln.strip()][-1:] or [""]
        seen.append(last[0][:80])
        if " passed" not in last[0] or "failed" in t or " error" in last[0]:
            bad.append("test output %s does not end in a pass" % os.path.basename(p))
    if len(paths) < 2:
        bad.append("two test outputs are needed (dev box and pod)")
    return {"pass": not bad, "failures": bad, "last_lines": seen}


def inputs_of(W, x):
    return {"files": W.sha, "tool": MJ.sha_file(os.path.abspath(__file__)), "make_joined_gapfiles.py": MJ.sha_file(MJ.__file__),
            "rec106": MJ.sha_file(x.rec106) if x.rec106 and os.path.isfile(x.rec106) else None}


def run_controls(x, a, expect=None, fault=None):
    W = World(x, a)
    ctl = {"s0": ctl_s0(W, BS.try_json(x.rec106)), "s1": ctl_s1(W, expect or EXPECT), "s2": ctl_s2(W, fault), "s3": ctl_s3(W), "s4": ctl_s4(x.tests_out)}
    rec = {"card": CARD, "tier": TIER_T0, "controls": ctl, "inputs": inputs_of(W, x), "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    for k in sorted(ctl):
        rec["pass_" + k] = bool(ctl[k]["pass"])
    rec["all_pass"] = all(rec["pass_" + k] for k in ctl)
    os.makedirs(x.out_dir, exist_ok=True)
    BS.wt(RB.fresh(os.path.join(x.out_dir, "GAPSPREAD_CONTROLS.json")), json.dumps(rec, indent=1, default=BS._jdefault))
    for k in sorted(ctl):
        c = ctl[k]
        print("ctl %s PASS" % k if c["pass"] else "ctl %s FAIL %s" % (k, (c["failures"][0] if c["failures"] else "")[:150]))
    print("ZZ GAPSPREAD-T0 controls all_pass %s" % rec["all_pass"])
    return rec


# ============================================================================================ the name
def run_name(x, a):
    crec = BS.try_json(os.path.join(x.out_dir, "GAPSPREAD_CONTROLS.json")) or {}
    W = World(x, a)
    if crec.get("card") != CARD or crec.get("all_pass") is not True or crec.get("inputs") != json.loads(json.dumps(inputs_of(W, x))):
        print("REFUSED: the controls file is not all-pass for these input files: no value is opened")
        return 3
    k, G = W.arms()
    rl, iv_l, rs, iv_s = W.rl_rs(G)
    name, bar = name_of(rl, iv_l, iv_s)
    lab = W.lab
    env = W.fits.min(axis=0)
    arms = {"base": W.m, "A90": G["A90"], "SP90": G["SP90"], "ENV": env}
    rel = (W.m[W.rows] - lab) / lab
    s_lab = W.s[W.rows]
    fac = factor_of(W.s[W.has], k)
    bins = []
    for lo, hi in RB.GAP_BINS:
        kk = BS.gap_bin_mask(lab, lo, hi)
        ent = {"bin": "%d-%d m" % (lo, hi), "n": int(kk.sum())}
        if kk.any():
            a9, sp = G["A90"][W.rows][kk], G["SP90"][W.rows][kk]
            dl, ds = float(long_part(a9, lab[kk]).mean()), float(short_part(a9, lab[kk]).mean())
            ent["RL"] = BS.fnum(float(long_part(sp, lab[kk]).mean()) / dl) if dl > 0 else None
            ent["RS"] = BS.fnum(float(short_part(sp, lab[kk]).mean()) / ds) if ds > 0 else None
        bins.append(ent)
    res = {"card": CARD, "tier": TIER_T0, "name": name, "bar_name": bar, "families": list(FAMILY_LINES),
           "n_rows": int(len(W.rows)), "n_clips": W.nc, "n_rows_with_output": int(W.has.sum()),
           "RL": BS.fnum(rl), "RL_interval": iv_l, "RS": BS.fnum(rs), "RS_interval": iv_s, "n_boot": N_BOOT, "seed": SEED, "dose": DOSE, "floor": FLOOR, "bar": BAR,
           "beside": {"k": BS.fnum(k), "mean_spread_rows_with_output": BS.fnum(W.s[W.has].mean()), "mean_spread_label_rows": BS.fnum(s_lab.mean()),
                      "share_factor_below_dose": BS.fnum((fac < DOSE).mean()), "share_factor_at_floor": BS.fnum((fac <= FLOOR).mean()),
                      "rank_corr_spread_vs_signed_rel_error": BS.fnum(rank_corr(s_lab, rel)), "rank_corr_interval": rank_corr_iv(s_lab, rel, W.ci, W.nc),
                      "mean_factor_ENV": BS.fnum((env[W.has] / W.m[W.has]).mean()),
                      "long_part_m": {n: BS.fnum(long_part(g[W.rows], lab).mean()) for n, g in arms.items()},
                      "short_part_m": {n: BS.fnum(short_part(g[W.rows], lab).mean()) for n, g in arms.items()},
                      "abs_error_m": {n: BS.fnum(np.abs(g[W.rows] - lab).mean()) for n, g in arms.items()}, "bins": bins},
           "controls_sha256": MJ.sha_file(os.path.join(x.out_dir, "GAPSPREAD_CONTROLS.json")), "inputs": inputs_of(W, x)}
    BS.wt(RB.fresh(os.path.join(x.out_dir, "GAPSPREAD_RESULT.json")), json.dumps(res, indent=1, default=BS._jdefault))
    print("ZZ GAPSPREAD-T0 NAME %s%s" % (name, (" " + bar) if bar else ""))
    fi = lambda v: "None" if v is None else "[%.4f, %.4f]" % (v[0], v[1])  # noqa: E731
    print("label rows %d of %d clips: RL %.4f %s, RS %.4f %s" % (len(W.rows), W.nc, rl, fi(iv_l), rs, fi(iv_s)))
    b = res["beside"]
    print("beside: k %.4f, mean spread %.4f, rank correlation %.4f %s, share of rows with a factor below the dose %.4f, at the floor %.4f, mean factor of the envelope %.4f"
          % (k, b["mean_spread_rows_with_output"], b["rank_corr_spread_vs_signed_rel_error"], fi(b["rank_corr_interval"]), b["share_factor_below_dose"], b["share_factor_at_floor"], b["mean_factor_ENV"]))
    for n in ("base", "A90", "SP90", "ENV"):
        print("  %-5s long %.3f m, short %.3f m, abs error %.3f m" % (n, b["long_part_m"][n], b["short_part_m"][n], b["abs_error_m"][n]))
    for e in bins:
        print("  bin %s n %d RL %s RS %s" % (e["bin"], e["n"], e.get("RL"), e.get("RS")))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("mode", choices=("controls", "name"))
    ap.add_argument("--base", default="")
    ap.add_argument("--fit", action="append", default=[])
    ap.add_argument("--rec106", default="")
    ap.add_argument("--rb-args", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tests-out", action="append", default=[])
    x = ap.parse_args(argv)
    if not x.base or len(x.fit) != 3 or not x.rb_args or not x.out_dir:
        print("REFUSED: --base, three --fit, --rb-args and --out-dir are required")
        return 2
    a = MJ.rb_args(x.rb_args, x.out_dir + "_scratch")
    if x.mode == "controls":
        return 0 if run_controls(x, a)["all_pass"] else 1
    return run_name(x, a)


if __name__ == "__main__":
    sys.exit(main())
