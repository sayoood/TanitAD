"""read_gapspreadb_t0.py -- card PREREG_V8F_PROOF_ADD133 (GAPSPREADB-T0): the read of card ADD132 with the permuted spread as its own arm.

Tier T0 (recorded frames, no rollout; a diagnostic of a perception read and not driving performance). CPU only.
The tool of card ADD132 (read_gapspread_t0.py) is imported unchanged. This card runs only when control s3 of card ADD132 has failed in its run of record.

  controls  -> GAPSPREADB_CONTROLS.json, lines 'ctl s<i> PASS/FAIL'; prints no value of the card
  name      -> refuses unless the controls file is all-pass for the same input files; GAPSPREADB_RESULT.json and the names
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import io
import json
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD133"
CARD_A = "PREREG_V8F_PROOF_ADD132"
GS_PREFIX = "af05b17a90410836"
SEED = 133
SEED_2 = 134
N_BOOT = 2000
SORT_TOL = 1e-12


def _sha(p):
    import hashlib
    h = hashlib.sha256()
    with io.open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_gs():
    m = sys.modules.get("read_gapspread_t0")
    p = getattr(m, "__file__", None) if m is not None else None
    if not (p and os.path.isfile(p)):
        for p in (os.environ.get("GAPSPREAD_TOOL"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "read_gapspread_t0.py")):
            if p and os.path.isfile(p):
                break
        else:
            raise SystemExit("REFUSED: read_gapspread_t0.py not found (set GAPSPREAD_TOOL)")
        spec = importlib.util.spec_from_file_location("read_gapspread_t0", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["read_gapspread_t0"] = m
        spec.loader.exec_module(m)
    if not _sha(p).startswith(GS_PREFIX):
        raise SystemExit("REFUSED: %s is not the tool of card ADD132 (%s)" % (p, GS_PREFIX))
    return m


GS = load_gs()
MJ, RB, BS = GS.MJ, GS.RB, GS.BS


def permuted(W, s, seed):
    """the spread permuted among the label rows and among the other rows with an output"""
    rng = np.random.RandomState(seed)
    out = np.array(s, np.float64, copy=True)
    out[W.rows] = s[W.rows][rng.permutation(len(W.rows))]
    other = np.flatnonzero(W.has)
    other = other[~np.isin(other, W.rows)]
    out[other] = s[other][rng.permutation(len(other))]
    return out


def ratio_iv(num_rows, den_rows, W):
    n, d = GS.clip_sums(num_rows, W.ci, W.nc), GS.clip_sums(den_rows, W.ci, W.nc)
    r = float(n.sum() / d.sum()) if d.sum() > 0 else float("nan")
    return r, BS.boot_ratio(n, d, N_BOOT, SEED)


def gap_of(W, s, k):
    return GS.sp_gap(W.m, s, k)


def pair(W, g_num, g_den):
    """-> (RL, interval, RS, interval) of two gaps on the label rows"""
    a, b = g_num[W.rows], g_den[W.rows]
    rl, il = ratio_iv(GS.long_part(a, W.lab), GS.long_part(b, W.lab), W)
    rs, is_ = ratio_iv(GS.short_part(a, W.lab), GS.short_part(b, W.lab), W)
    return rl, il, rs, is_


def flat_name(rl, il, is_):
    if il is None or is_ is None:
        return "GAPSPREADB-T0-PLACED-NOT-SEPARATED", None
    if il[1] < 1.0:
        if is_[0] > 1.0:
            return "GAPSPREADB-T0-PLACED-TRADES", None
        return "GAPSPREADB-T0-PLACED-BETTER-THAN-FLAT", ("GAPSPREADB-T0-BAR-MET" if rl <= GS.BAR else "GAPSPREADB-T0-BAR-NOT-MET")
    if il[0] > 1.0:
        return "GAPSPREADB-T0-PLACED-WORSE-THAN-FLAT", None
    return "GAPSPREADB-T0-PLACED-NOT-SEPARATED", None


def mark_name(il):
    if il is not None and il[1] < 1.0:
        return "GAPSPREADB-T0-SPREAD-MARKS-LONG-READS"
    if il is not None and il[0] > 1.0:
        return "GAPSPREADB-T0-SPREAD-MARKS-THE-OTHER-WAY"
    return "GAPSPREADB-T0-SPREAD-DOES-NOT-MARK"


def ctl_s3(W):
    bad, out = [], {}
    sp = GS.planted_spread(W)
    kp = GS.solve_k(sp[W.has])
    if kp is None:
        bad.append("no k reaches the dose with the planted spread")
    else:
        g = gap_of(W, sp, kp)
        rl = pair(W, g, GS.DOSE * W.m)[0]
        ilp = pair(W, g, gap_of(W, permuted(W, sp, SEED), kp))[1]
        out["planted_below_the_bar"] = bool(rl < GS.BAR)
        out["planted_placement_read"] = bool(ilp is not None and ilp[1] < 1.0)
        if not rl < GS.BAR:
            bad.append("the planted spread is not read against the flat margin")
        if not (ilp is not None and ilp[1] < 1.0):
            bad.append("the planted spread is not read against its permuted form")
    k = GS.solve_k(W.s[W.has])
    if k is None:
        bad.append("no k reaches the dose with the real spread")
    else:
        r, iv = pair(W, gap_of(W, permuted(W, W.s, SEED), k), gap_of(W, permuted(W, W.s, SEED_2), k))[:2]
        ok = (iv is not None and iv[0] <= 1.0 <= iv[1]) or abs(r - 1.0) <= GS.PERM_NEAR
        out["two_permutations_agree"] = bool(ok)
        if not ok:
            bad.append("two permutations of the real spread do not read alike")
    out.update({"pass": not bad, "failures": bad})
    return out


def ctl_s4(paths):
    c = GS.ctl_s4(paths)
    if len(paths) < 4:
        c["pass"] = False
        c["failures"].append("four test outputs are needed (two tools, dev box and pod)")
    return c


def ctl_s5(W, fault=None):
    bad = []
    k = GS.solve_k(W.s[W.has])
    if k is None:
        return {"pass": False, "failures": ["no k reaches the dose with the real spread"]}
    sp = permuted(W, W.s, SEED)
    if fault == "other-k":
        k_pm = k * 1.01
    else:
        k_pm = GS.solve_k(sp[W.has])
    if k_pm is None or abs(k_pm - k) > 1e-9 * max(1.0, abs(k)):
        bad.append("the permuted arm does not have the k of SP90")
    f0, f1 = GS.factor_of(W.s, k), GS.factor_of(sp, k)
    other = np.flatnonzero(W.has)
    other = other[~np.isin(other, W.rows)]
    for nm, idx in (("label rows", W.rows), ("other rows", other)):
        if len(idx) and float(np.abs(np.sort(f0[idx]) - np.sort(f1[idx])).max()) > SORT_TOL:
            bad.append("the factors of the permuted arm on the %s are not those of SP90" % nm)
    return {"pass": not bad, "failures": bad, "n_label_rows": int(len(W.rows)), "n_other_rows": int(len(other))}


def inputs_of(W, x):
    d = GS.inputs_of(W, x)
    d["tool_b"] = _sha(os.path.abspath(__file__))
    d["add132_controls"] = _sha(x.add132_controls) if x.add132_controls and os.path.isfile(x.add132_controls) else None
    return d


def order_ok(x):
    """this card runs only after a run of record of card ADD132 whose control s3 failed"""
    r = BS.try_json(x.add132_controls) or {}
    return r.get("card") == CARD_A and r.get("pass_s3") is False


def run_controls(x, a, expect=None, fault=None):
    if not order_ok(x):
        print("REFUSED: no controls file of card ADD132 with a failed control s3: this card is not run")
        return None
    W = GS.World(x, a)
    ctl = {"s0": GS.ctl_s0(W, BS.try_json(x.rec106)), "s1": GS.ctl_s1(W, expect or GS.EXPECT), "s2": GS.ctl_s2(W), "s3": ctl_s3(W), "s4": ctl_s4(x.tests_out), "s5": ctl_s5(W, fault)}
    rec = {"card": CARD, "tier": GS.TIER_T0, "controls": ctl, "inputs": inputs_of(W, x), "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    for k in sorted(ctl):
        rec["pass_" + k] = bool(ctl[k]["pass"])
    rec["all_pass"] = all(rec["pass_" + k] for k in ctl)
    os.makedirs(x.out_dir, exist_ok=True)
    BS.wt(RB.fresh(os.path.join(x.out_dir, "GAPSPREADB_CONTROLS.json")), json.dumps(rec, indent=1, default=BS._jdefault))
    for k in sorted(ctl):
        c = ctl[k]
        print("ctl %s PASS" % k if c["pass"] else "ctl %s FAIL %s" % (k, (c["failures"][0] if c["failures"] else "")[:150]))
    print("ZZ GAPSPREADB-T0 controls all_pass %s" % rec["all_pass"])
    return rec


def run_name(x, a):
    crec = BS.try_json(os.path.join(x.out_dir, "GAPSPREADB_CONTROLS.json")) or {}
    W = GS.World(x, a)
    if crec.get("card") != CARD or crec.get("all_pass") is not True or crec.get("inputs") != json.loads(json.dumps(inputs_of(W, x))):
        print("REFUSED: the controls file is not all-pass for these input files: no value is opened")
        return 3
    k = GS.solve_k(W.s[W.has])
    g = {"base": W.m, "A90": GS.DOSE * W.m, "SP90": gap_of(W, W.s, k), "PM": gap_of(W, permuted(W, W.s, SEED), k), "ENV": W.fits.min(axis=0)}
    rl, il, rs, is_ = pair(W, g["SP90"], g["A90"])
    rlp, ilp, rsp, isp = pair(W, g["SP90"], g["PM"])
    pl, pil, ps, pis = pair(W, g["PM"], g["A90"])
    name, bar = flat_name(rl, il, is_)
    mark = mark_name(ilp)
    lab = W.lab
    rel = (W.m[W.rows] - lab) / lab
    s_lab = W.s[W.rows]
    fac = GS.factor_of(W.s[W.has], k)
    bins = []
    for lo, hi in RB.GAP_BINS:
        kk = BS.gap_bin_mask(lab, lo, hi)
        ent = {"bin": "%d-%d m" % (lo, hi), "n": int(kk.sum())}
        if kk.any():
            for nm, den in (("", "A90"), ("p", "PM")):
                dl = float(GS.long_part(g[den][W.rows][kk], lab[kk]).mean())
                ds = float(GS.short_part(g[den][W.rows][kk], lab[kk]).mean())
                ent["RL" + nm] = BS.fnum(float(GS.long_part(g["SP90"][W.rows][kk], lab[kk]).mean()) / dl) if dl > 0 else None
                ent["RS" + nm] = BS.fnum(float(GS.short_part(g["SP90"][W.rows][kk], lab[kk]).mean()) / ds) if ds > 0 else None
        bins.append(ent)
    res = {"card": CARD, "tier": GS.TIER_T0, "name": name, "bar_name": bar, "mark_name": mark, "families": list(GS.FAMILY_LINES),
           "n_rows": int(len(W.rows)), "n_clips": W.nc, "n_rows_with_output": int(W.has.sum()), "n_boot": N_BOOT, "seed": SEED, "dose": GS.DOSE, "floor": GS.FLOOR, "bar": GS.BAR,
           "RL": BS.fnum(rl), "RL_interval": il, "RS": BS.fnum(rs), "RS_interval": is_, "RLp": BS.fnum(rlp), "RLp_interval": ilp, "RSp": BS.fnum(rsp), "RSp_interval": isp,
           "beside": {"k": BS.fnum(k), "mean_spread_rows_with_output": BS.fnum(W.s[W.has].mean()), "mean_spread_label_rows": BS.fnum(s_lab.mean()),
                      "mean_factor_SP90_label_rows": BS.fnum(GS.factor_of(s_lab, k).mean()), "share_factor_below_dose": BS.fnum((fac < GS.DOSE).mean()),
                      "share_factor_at_floor": BS.fnum((fac <= GS.FLOOR).mean()), "rank_corr_spread_vs_signed_rel_error": BS.fnum(GS.rank_corr(s_lab, rel)),
                      "rank_corr_interval": GS.rank_corr_iv(s_lab, rel, W.ci, W.nc), "mean_factor_ENV": BS.fnum((g["ENV"][W.has] / W.m[W.has]).mean()),
                      "PM_over_A90": {"RL": BS.fnum(pl), "RL_interval": pil, "RS": BS.fnum(ps), "RS_interval": pis},
                      "long_part_m": {n: BS.fnum(GS.long_part(v[W.rows], lab).mean()) for n, v in g.items()},
                      "short_part_m": {n: BS.fnum(GS.short_part(v[W.rows], lab).mean()) for n, v in g.items()},
                      "abs_error_m": {n: BS.fnum(np.abs(v[W.rows] - lab).mean()) for n, v in g.items()}, "bins": bins},
           "controls_sha256": _sha(os.path.join(x.out_dir, "GAPSPREADB_CONTROLS.json")), "inputs": inputs_of(W, x)}
    BS.wt(RB.fresh(os.path.join(x.out_dir, "GAPSPREADB_RESULT.json")), json.dumps(res, indent=1, default=BS._jdefault))
    fi = lambda v: "None" if v is None else "[%.4f, %.4f]" % (v[0], v[1])  # noqa: E731
    print("ZZ GAPSPREADB-T0 NAMES %s%s %s" % (name, (" " + bar) if bar else "", mark))
    print("label rows %d of %d clips, against the flat margin: RL %.4f %s, RS %.4f %s" % (len(W.rows), W.nc, rl, fi(il), rs, fi(is_)))
    print("against the permuted arm: RLp %.4f %s, RSp %.4f %s; permuted against flat: RL %.4f %s, RS %.4f %s" % (rlp, fi(ilp), rsp, fi(isp), pl, fi(pil), ps, fi(pis)))
    b = res["beside"]
    print("beside: k %.4f, mean spread %.4f (label rows %.4f), mean factor on label rows %.4f, rank correlation %.4f %s, share below the dose %.4f, at the floor %.4f, mean factor of the envelope %.4f"
          % (k, b["mean_spread_rows_with_output"], b["mean_spread_label_rows"], b["mean_factor_SP90_label_rows"], b["rank_corr_spread_vs_signed_rel_error"], fi(b["rank_corr_interval"]),
             b["share_factor_below_dose"], b["share_factor_at_floor"], b["mean_factor_ENV"]))
    for n in ("base", "A90", "SP90", "PM", "ENV"):
        print("  %-5s long %.3f m, short %.3f m, abs error %.3f m" % (n, b["long_part_m"][n], b["short_part_m"][n], b["abs_error_m"][n]))
    for e in bins:
        print("  bin %s n %d RL %s RS %s RLp %s RSp %s" % (e["bin"], e["n"], e.get("RL"), e.get("RS"), e.get("RLp"), e.get("RSp")))
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
    ap.add_argument("--add132-controls", default="")
    x = ap.parse_args(argv)
    if not x.base or len(x.fit) != 3 or not x.rb_args or not x.out_dir or not x.add132_controls:
        print("REFUSED: --base, three --fit, --rb-args, --out-dir and --add132-controls are required")
        return 2
    a = MJ.rb_args(x.rb_args, x.out_dir + "_scratch")
    if x.mode == "controls":
        rec = run_controls(x, a)
        return 3 if rec is None else (0 if rec["all_pass"] else 1)
    return run_name(x, a)


if __name__ == "__main__":
    sys.exit(main())
