r"""read_dscale_pres.py -- the reader of card PREREG_V8F_PROOF_ADD126 (DSCALE-PRES-T0). Tier T0. CPU only, numpy only.

TIER: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance.

The reader of card ADD125 (read_dscale_eqp.py) is IMPORTED UNCHANGED (its sha256 prefix is asserted at import; path settable with the environment variable EQP_READER). Through it this file reuses the probe
table of the 18 probes, the probe loading (the banked chain's per-probe read), the input hashes, the test clause and the two-level bootstrap (two_level) of the banked chain. This file adds the presence AUROC with
ties counted half, its episode-weighted form on every draw of the chain, the measure A of the card, the beside differences and tables, the controls p0 to p5 and the names.

  python read_dscale_pres.py controls <run arguments of read_dscale_eqp.py> --eqp-result F --out-dir D --test-out F --test-before EPOCH [--B 2000]
  python read_dscale_pres.py names    (the same arguments)

Mode controls: writes DSCALE_PRES_CONTROLS.json and prints one line per control. It computes and prints no value of A, no beside value and no per-probe share.
Mode names: refuses (exit 3) unless the controls file of the same out-dir says all_pass with the same input hashes; then writes DSCALE_PRES_RESULT.json and DSCALE_PRES_VERDICT.md and prints the name line.
Exit codes: 0 all pass, 2 a control failed, 3 refused. A result file is never overwritten (a second call writes ..._2).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
EQP_SHA_PREFIX = "7ffbc27367203f58"
EQP_PATH = os.environ.get("EQP_READER") or os.path.abspath(os.path.join(HERE, "..", "..", "add125", "read", "read_dscale_eqp.py")).replace(os.sep, "/")
EQP_RESULT_SHA_PREFIX = "927b06ff7e4a69e9"          # DSCALE_EQP_RESULT.json of card ADD125 (card ADD126 section 5, p0)


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _load_eqp():
    if not os.path.exists(EQP_PATH):
        raise AssertionError("REFUSED: the reader of card ADD125 is not found at %s (set EQP_READER)" % EQP_PATH)
    got = _sha_file(EQP_PATH)
    if not got.startswith(EQP_SHA_PREFIX):
        raise AssertionError("REFUSED: sha256 of %s begins %s, the card's reader begins %s" % (EQP_PATH, got[:16], EQP_SHA_PREFIX))
    spec = importlib.util.spec_from_file_location("read_dscale_eqp_base", EQP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["read_dscale_eqp_base"] = mod
    spec.loader.exec_module(mod)
    return mod


EQ = _load_eqp()
X, D, N, RP = EQ.X, EQ.D, EQ.N, EQ.RP
SEEDS = EQ.SEEDS
CONTROL_NAMES = tuple("p%d" % i for i in range(6))
NAMES = ("LEAD-PRES-EQP-SCALES", "LEAD-PRES-EQP-SMALL", "LEAD-PRES-EQP-WORSE", "LEAD-PRES-EQP-FLAT")
MEAN_BAR = 0.020
MIN_POS, MIN_NEG, MIN_EPISODES, N_EPISODES = 50, 50, 20, 48
P1_TOL = 1e-9
LOGIT_AT_HALF = 0.0                  # presence_logit is the raw first output of the head (leadsup_read_n1.head_out_units); probability 0.5 is logit 0
GAP_BINS = ((0.0, 20.0, "gap_0_20_m"), (20.0, 40.0, "gap_20_40_m"), (40.0, float("inf"), "gap_40_plus_m"))
CONTROLS_FILE, RESULT_FILE, VERDICT_FILE = "DSCALE_PRES_CONTROLS.json", "DSCALE_PRES_RESULT.json", "DSCALE_PRES_VERDICT.md"
TIER_LINE = D.TIER_LINE
FAMILY_LINES = D.FAMILY_LINES
LIMITS = ("One frozen encoder; pooled tokens; the evaluation windows come from the corpus only; four seeds, and the two-level interval treats them as the draw of seeds. The point values were seen "
          "before the card (card section 2). An interval above 0 on four seeds is necessary for a name and not sufficient for a claim about training runs at large. A probe is not a model.")


def lab(kind, k):
    return "P16%s_s%d" % (kind, k)


# ============================================================================================ the presence AUROC, ties counted half, episode weights
def _prep(ep, y, s):
    m = np.isfinite(y) & np.isfinite(s) & (ep >= 0)
    e, yy, ss = ep[m], y[m], s[m]
    o = np.argsort(ss, kind="mergesort")
    e, yy, ss = e[o], yy[o], ss[o]
    n = len(ss)
    if n == 0:
        return e, yy, ss, np.zeros(0, int), np.zeros(0, int)
    starts = np.flatnonzero(np.concatenate(([True], ss[1:] != ss[:-1])))
    ends = np.concatenate((starts[1:], [n])) - 1
    return e, yy, ss, starts, ends


def auc_weighted(ep, y, s, C, chunk=100):
    """Weighted AUROC for every row of C ([draws, n_episodes] episode counts as case weights). A positive scored above a negative counts 1, a tie counts 1/2. y in {0, 1}; non-finite rows and ep < 0 are dropped."""
    e, yy, ss, starts, ends = _prep(ep, y, s)
    C = np.atleast_2d(np.asarray(C, float))
    if len(ss) == 0:
        return np.full(len(C), np.nan)
    pos, neg = (yy == 1), (yy == 0)
    out = []
    for i in range(0, len(C), chunk):
        W = C[i:i + chunk][:, e]
        wp, wn = W * pos, W * neg
        cum = np.cumsum(wn, axis=1)
        gn = np.add.reduceat(wn, starts, axis=1)
        gp = np.add.reduceat(wp, starts, axis=1)
        before = cum[:, ends] - gn
        num = (gp * (before + 0.5 * gn)).sum(1)
        den = wp.sum(1) * wn.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(num / den)
    return np.concatenate(out)


def auc_point(ep, y, s):
    n_ep = int(ep.max()) + 1 if len(ep) and ep.max() >= 0 else 1
    return float(auc_weighted(ep, y, s, np.ones((1, n_ep)))[0])


def n_tied(ep, y, s):
    e, yy, ss, starts, ends = _prep(ep, y, s)
    return int(len(ss) - len(starts))


def pres_arrays(ctx):
    """The windows, labels and episode draw of the chain's own M4 (leadsup_read_n1.eval_read): pres0, ep_all, n_all, C_all."""
    return ctx["pres0"], ctx["ep_all"], ctx["n_all"], ctx["C_all"]


def boot_of(ctx, out):
    y, ep, n, C = pres_arrays(ctx)
    return auc_point(ep, y, out["presence_logit"]), auc_weighted(ep, y, out["presence_logit"], C)


def pair_two_level(boots, a_kind, b_kind):
    pts, Ds = [], []
    for k in SEEDS:
        pa, va = boots[lab(a_kind, k)]
        pb, vb = boots[lab(b_kind, k)]
        pts.append(pa - pb)
        Ds.append(va - vb)
    return D.two_level(pts, np.array(Ds), 0, "shared")


# ============================================================================================ beside, descriptive
def descriptive(ctx, out):
    """Per probe, at the score threshold 0.5 (logit 0): the share of windows with a lead scored below it, in all and by the label gap at S0 (ctx G15 column S0F), and the share of windows without a lead scored at or above it."""
    y = ctx["pres0"]
    s = out["presence_logit"]
    g = ctx["G15"][:, N.S0F]
    ok = np.isfinite(y) & np.isfinite(s)
    lead, nolead = ok & (y == 1), ok & (y == 0)
    below = s < LOGIT_AT_HALF
    r = {"n_with_lead": int(lead.sum()), "n_without_lead": int(nolead.sum()),
         "share_lead_below_half_all": float(below[lead].mean()) if lead.any() else None, "by_gap": {}}
    binned = np.zeros(len(y), bool)
    for lo, hi, name in GAP_BINS:
        mb = lead & np.isfinite(g) & (g >= lo) & (g < hi)
        binned |= mb
        r["by_gap"][name] = {"n": int(mb.sum()), "share_below_half": float(below[mb].mean()) if mb.any() else None}
    r["lead_windows_without_a_finite_gap_label"] = int((lead & ~binned).sum())
    r["share_no_lead_at_or_above_half"] = float((~below)[nolead].mean()) if nolead.any() else None
    return r


def window_counts(ctx):
    y, ep, n, C = pres_arrays(ctx)
    m = np.isfinite(y) & (ep >= 0)
    return {"windows": int(m.sum()), "windows_with_a_lead": int((y[m] == 1).sum()), "windows_without_a_lead": int((y[m] == 0).sum()), "episodes": int(len(np.unique(ep[m]))),
            "n_episodes_all_windows": int(n), "n_episodes_scored_windows": int(ctx["n10"])}


# ============================================================================================ controls p0 to p5
def p0_inputs(a, specs, eqp_result_path):
    mine_args = argparse.Namespace(**{**vars(a), "test_out": None})
    mine = {k.replace("\\", "/"): v for k, v in EQ.input_hashes(mine_args, specs).items()}
    row = {"eqp_result_found": False, "eqp_result_sha_prefix_matches": False, "input_hashes_in_result": False, "every_read_input_equals_the_banked_hash": False, "no_read_input_missing": False,
           "inputs_only_in_the_result_at_most_one": False}
    if not (eqp_result_path and os.path.exists(eqp_result_path)):
        return {"pass": False, **row}
    row["eqp_result_found"] = True
    row["eqp_result_sha_prefix_matches"] = bool(_sha_file(eqp_result_path).startswith(EQP_RESULT_SHA_PREFIX))
    prev = (X._json(eqp_result_path) or {}).get("input_hashes")
    if not isinstance(prev, dict):
        return {"pass": False, **row}
    row["input_hashes_in_result"] = True
    prev_n = {}
    for k, v in prev.items():
        k2 = k.replace("\\", "/")
        for old, new in (getattr(a, "eqp_prefix_map", None) or []):
            if k2.startswith(old):
                k2 = new + k2[len(old):]
                break
        prev_n[k2] = v
    row["every_read_input_equals_the_banked_hash"] = bool(mine and all(prev_n.get(k) == v for k, v in mine.items()))
    row["no_read_input_missing"] = bool(all(v != "MISSING" for v in mine.values()) and all(v != "MISSING" for v in prev_n.values()))
    row["inputs_only_in_the_result_at_most_one"] = bool(len([k for k in prev_n if k not in mine]) <= 1)
    return {"pass": bool(all(row.values())), **row, "n_inputs_read": len(mine), "n_inputs_in_result": len(prev_n)}


def p1_identity(ctx, P):
    y, ep, n, C = pres_arrays(ctx)
    worst, rows, ok = 0.0, {}, True
    for label, p in P.items():
        good = False
        if p["read"] and p["R"].get("M4") and "auroc" in p["R"]["M4"]:
            mine = auc_point(ep, y, p["out"]["presence_logit"])
            d = abs(mine - float(p["R"]["M4"]["auroc"]))
            good = bool(np.isfinite(d) and d <= P1_TOL)
            worst = max(worst, d) if np.isfinite(d) else float("inf")
            rows[label] = {"pass": good, "tied_windows": n_tied(ep, y, p["out"]["presence_logit"])}
        else:
            worst = float("inf")
            rows[label] = {"pass": False, "tied_windows": None}
        ok = ok and good
    return {"pass": bool(ok and len(P) == 18), "probes_read": int(sum(1 for p in P.values() if p["read"])), "max_abs_difference": float(worst), "tolerance": P1_TOL, "per_probe": rows}


def p2_estimator(ctx):
    """The estimator forms on the labels, episodes and draw of the chain's M4, with synthetic scores (no probe value is touched)."""
    y, ep, n, C = pres_arrays(ctx)
    m = np.isfinite(y) & (ep >= 0)
    r = np.random.default_rng(2026)
    base = np.where(y == 1, 0.8, 0.0) + r.normal(size=len(y))          # scores with signal
    seeds_s = [base + 0.3 * r.normal(size=len(y)) for _ in SEEDS]
    seeds_t = [base + 0.6 * r.normal(size=len(y)) for _ in SEEDS]

    def read(sa, sb):
        pts, Ds = [], []
        for a_s, b_s in zip(sa, sb):
            pa, va = auc_point(ep, y, a_s), auc_weighted(ep, y, a_s, C)
            pb, vb = auc_point(ep, y, b_s), auc_weighted(ep, y, b_s, C)
            pts.append(pa - pb)
            Ds.append(va - vb)
        return D.two_level(pts, np.array(Ds), 0, "shared")
    selfp = read(seeds_s, seeds_s)
    pair, swap = read(seeds_s, seeds_t), read(seeds_t, seeds_s)
    labels_as_scores = auc_point(ep, y, np.where(m, y, np.nan))
    lab_boot = auc_weighted(ep, y, np.where(m, y, np.nan), C)
    perm = np.random.default_rng(7).permutation(len(base))
    pp = auc_point(ep, y, base[perm])
    pci = RP.ci(auc_weighted(ep, y, base[perm], C))
    row = {"self_pair_mean_is_exactly_0": bool(selfp["mean"] == 0.0), "self_pair_interval_is_0_0": bool(selfp["ci95"] == [0.0, 0.0]),
           "swapped_pair_mean_is_the_negative": bool(abs(swap["mean"] + pair["mean"]) <= 1e-12 and pair["mean"] != 0.0),
           "swapped_pair_interval_is_the_mirror": bool(abs(swap["ci95"][0] + pair["ci95"][1]) <= 1e-12 and abs(swap["ci95"][1] + pair["ci95"][0]) <= 1e-12),
           "labels_as_scores_read_1": bool(labels_as_scores == 1.0 and bool(np.all(lab_boot == 1.0))),
           "permuted_scores_interval_includes_0.5": bool(pci[0] <= 0.5 <= pci[1])}
    return {"pass": bool(all(row.values())), **row}


def p3_instrument(ctx, P):
    y, ep, n, C = pres_arrays(ctx)
    c = window_counts(ctx)
    least_pos, least_neg = c["windows_with_a_lead"], c["windows_without_a_lead"]
    least_ep = c["episodes"]
    for p in P.values():
        if p["read"]:
            s = p["out"]["presence_logit"]
            m = np.isfinite(y) & np.isfinite(s) & (ep >= 0)
            least_pos, least_neg = min(least_pos, int((y[m] == 1).sum())), min(least_neg, int((y[m] == 0).sum()))
            least_ep = min(least_ep, int(len(np.unique(ep[m]))))
    ok = bool(least_pos >= MIN_POS and least_neg >= MIN_NEG and least_ep >= MIN_EPISODES)
    return {"pass": ok, "counts": c, "least_over_probes": {"windows_with_a_lead": least_pos, "windows_without_a_lead": least_neg, "episodes": least_ep},
            "bars": {"windows_with_a_lead": MIN_POS, "windows_without_a_lead": MIN_NEG, "episodes": MIN_EPISODES}}


def p4_tests(a):
    t = X.tests_clause(a.test_out, a.test_before)
    return {"pass": bool(t["pass"]), "tests": t}


def p5_one_set(ctx, P):
    q6 = X.q6_one_evalset(ctx, P)
    same_draw = bool(ctx["n10"] == ctx["n_all"] and np.array_equal(ctx["C10"], ctx["C_all"]))
    row = {"all_18_probes_read": bool(len(P) == 18 and all(p["read"] for p in P.values())), "same_ids_same_order_same_count": bool(q6["pass"]),
           "scored_windows_come_from_48_episodes": bool(ctx["n10"] == N_EPISODES)}
    # note N1 of the card: the window set of M4 spans every evaluation episode with a window (more than the 48 episodes of the scored gap windows); its episode count is recorded, not judged
    return {"pass": bool(all(row.values())), **row, "q6": q6, "n_episodes_all_windows": int(ctx["n_all"]), "n_episodes_scored_windows": int(ctx["n10"]),
            "recorded_all_windows_come_from_48_episodes": bool(ctx["n_all"] == N_EPISODES), "recorded_draw_C10_equals_draw_C_all": same_draw}


def run_controls(a, ctx, P, specs):
    c = {"p0": p0_inputs(a, specs, a.eqp_result), "p1": p1_identity(ctx, P), "p2": p2_estimator(ctx), "p3": p3_instrument(ctx, P), "p4": p4_tests(a), "p5": p5_one_set(ctx, P)}
    return c


def why(c, path=""):
    out = []
    if isinstance(c, dict):
        for k, v in c.items():
            if k in ("pass", "q6", "tests") and path == "":
                continue
            if v is False:
                out.append(path + str(k))
            elif isinstance(v, dict) and k not in ("per_probe", "counts", "least_over_probes", "bars", "q6"):
                out += why(v, path + str(k) + ".")
    return out


def short_reason(c):
    w = why(c)
    if c.get("tests") and c["tests"].get("pass") is False:
        w = ["tests"] + w
    return (", ".join(w[:3]) + (" (+%d)" % (len(w) - 3) if len(w) > 3 else "")) if w else "see the controls file"


# ============================================================================================ A and the names
def decide(A):
    above = bool(A["ci95"][0] > 0)
    below = bool(A["ci95"][1] < 0)
    n_pos, n_neg, mean = A["n_pos"], A["n_neg"], A["mean"]
    if above and n_pos >= 3 and mean >= MEAN_BAR:
        name = NAMES[0]
    elif above and n_pos >= 3:
        name = NAMES[1]
    elif below and n_neg >= 3:
        name = NAMES[2]
    else:
        name = NAMES[3]
    return {"name": name, "interval_above_0": above, "interval_below_0": below, "seeds_above_0": int(n_pos), "seeds_below_0": int(n_neg), "mean_of_A": float(mean), "mean_bar": MEAN_BAR}


def measures(ctx, P):
    boots = {l: boot_of(ctx, p["out"]) for l, p in P.items()}
    A = pair_two_level(boots, "E", "C")
    beside = {"Q_minus_C": pair_two_level(boots, "Q", "C"), "U_minus_C": pair_two_level(boots, "U", "C")}
    desc = {l: descriptive(ctx, p["out"]) for l, p in P.items()}
    probes = {l: {"auroc": boots[l][0], "chain_M4_auroc": float(p["R"]["M4"]["auroc"]) if p["R"].get("M4") else None} for l, p in P.items()}
    return {"A": A, "beside": {**beside, "per_probe_descriptive": desc}, "probes": probes, "counts": window_counts(ctx)}


# ============================================================================================ writing
def _ci(c):
    return "[%+.4f, %+.4f]" % (c[0], c[1])


def _row(t):
    return "mean %+.4f %s; seeds above 0: %d of 4; per seed %s" % (t["mean"], _ci(t["ci95"]), t["n_pos"], ", ".join("%+.4f" % x for x in t["per_seed"]))


def _f(x):
    return "n/a" if x is None else "%.3f" % x


def write_verdict(path, res):
    n, M = res["names"], res["measures"]
    L = ["# DSCALE_PRES_VERDICT: card PREREG_V8F_PROOF_ADD126 (DSCALE-PRES-T0)", "", TIER_LINE, ""] + ["- " + x for x in FAMILY_LINES]
    L += ["", "Controls p0 to p5: all passed before this file was written (DSCALE_PRES_CONTROLS.json). B %d; two-level bootstrap (seed indices, then episodes; rng 0); AUROC computed again on every episode draw." % res["B"],
          "", "## Name: **%s**" % n["name"], ""]
    for k in ("interval_above_0", "interval_below_0", "seeds_above_0", "seeds_below_0"):
        L.append("- %s: %s" % (k, n[k]))
    L.append("- mean of A %+.4f (bar %.3f)" % (n["mean_of_A"], n["mean_bar"]))
    L += ["", "## A (presence AUROC of P16E minus P16C, per seed, same windows)", "", "- A: " + _row(M["A"]), "", "## Beside (names nothing)", "",
          "- P16Q minus P16C presence AUROC: " + _row(M["beside"]["Q_minus_C"]), "- P16U minus P16C presence AUROC: " + _row(M["beside"]["U_minus_C"]),
          "- windows: %d, with a lead %d, without %d, in %d episodes" % (M["counts"]["windows"], M["counts"]["windows_with_a_lead"], M["counts"]["windows_without_a_lead"], M["counts"]["episodes"]), "",
          "## Per probe (AUROC; share of lead windows scored below 0.5 in all and by label gap; share of no-lead windows scored at 0.5 or more)", ""]
    for lbl, d in M["beside"]["per_probe_descriptive"].items():
        g = d["by_gap"]
        L.append("- %s: AUROC %.3f; below 0.5: all %s, 0-20 m %s (n %d), 20-40 m %s (n %d), 40+ m %s (n %d); no-lead at or above 0.5: %s" % (
            lbl, M["probes"][lbl]["auroc"], _f(d["share_lead_below_half_all"]), _f(g["gap_0_20_m"]["share_below_half"]), g["gap_0_20_m"]["n"], _f(g["gap_20_40_m"]["share_below_half"]),
            g["gap_20_40_m"]["n"], _f(g["gap_40_plus_m"]["share_below_half"]), g["gap_40_plus_m"]["n"], _f(d["share_no_lead_at_or_above_half"])))
    L += ["", "## Limits", "", LIMITS, ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


# ============================================================================================ main
def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("controls", "names"))
    for nm in ("banked-run-dir", "banked-in-dir", "banked-done", "q-run-dir", "q-in-dir", "q-done", "new-run-dir", "new-in-dir", "new-done", "out-dir", "eqp-result"):
        ap.add_argument("--" + nm, required=True)
    ap.add_argument("--pack-dir", default=D.TKR.READ)
    ap.add_argument("--test-out", default=None)
    ap.add_argument("--test-before", type=float, default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--controls-json", default=None)
    ap.add_argument("--eqp-prefix-map", nargs="*", default=[], help="OLD=NEW path prefixes applied to the keys of the previous card's input_hashes (when the previous read ran under other paths)")
    a = ap.parse_args(argv)
    a.eqp_prefix_map = [tuple(x.split("=", 1)) for x in a.eqp_prefix_map if "=" in x]
    return a


def hashes_of(a, specs):
    h = EQ.input_hashes(a, specs)
    h[a.eqp_result] = X.sha256_file(a.eqp_result) if os.path.exists(a.eqp_result) else "MISSING"
    return h


def main(argv=None):
    a = parse(argv)
    t0 = time.time()
    specs = EQ.probe_specs()
    for d in (a.banked_run_dir, a.banked_in_dir, a.q_run_dir, a.q_in_dir, a.new_run_dir, a.new_in_dir):
        if not os.path.isdir(d):
            print("ZZ DSCALE-PRES REFUSED missing folder %s" % d)
            return 3
    for f in (a.banked_done, a.q_done, a.new_done):
        if not os.path.exists(f):
            print("ZZ DSCALE-PRES REFUSED missing done file %s" % f)
            return 3
    os.makedirs(a.out_dir, exist_ok=True)
    hashes = hashes_of(a, specs)
    cp = None
    if a.mode == "names":
        cp = a.controls_json or X.latest_path(a.out_dir, CONTROLS_FILE)
        cj = X._json(cp) if cp else None
        if not (cj and cj.get("all_pass") is True and cj.get("input_hashes") == hashes and cj.get("mode") == "controls"):
            print("ZZ DSCALE-PRES REFUSED no passing controls file for these inputs")
            return 3
    ctx = EQ.build_ctx(a)
    P = EQ.load_all(a, ctx, specs)
    if a.mode == "controls":
        c = run_controls(a, ctx, P, specs)
        failed = [k for k in CONTROL_NAMES if not c[k]["pass"]]
        res = {"mode": "controls", "card": "PREREG_V8F_PROOF_ADD126", "tier": TIER_LINE, "controls": c, "all_pass": not failed, "failed": failed, "input_hashes": hashes, "B": a.B,
               "unread_probes": sorted(l for l, p in P.items() if not p["read"]), "elapsed_s": round(time.time() - t0, 1)}
        with open(X.unique_path(a.out_dir, CONTROLS_FILE), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=X._jd)
        for k in CONTROL_NAMES:
            print("ctl %s PASS" % k if c[k]["pass"] else "ctl %s FAIL %s" % (k, short_reason(c[k])[:100]))
        cn = c["p3"]["counts"]
        print("beside p3 windows with a lead %d, without %d, episodes %d" % (cn["windows_with_a_lead"], cn["windows_without_a_lead"], cn["episodes"]))
        print("beside p1 largest absolute difference %.3e" % c["p1"]["max_abs_difference"])
        print("ZZ DSCALE-PRES CONTROLS ALL-PASS" if not failed else "ZZ DSCALE-PRES CONTROLS FAILED")
        return 0 if not failed else 2
    if any(not p["read"] for p in P.values()):
        print("ZZ DSCALE-PRES REFUSED a probe could not be read: " + ", ".join(sorted(l for l, p in P.items() if not p["read"])))
        return 3
    M = measures(ctx, P)
    nm = decide(M["A"])
    res = {"mode": "names", "card": "PREREG_V8F_PROOF_ADD126", "tier": TIER_LINE, "families": list(FAMILY_LINES), "evidence_class": "MEASURED once run", "B": a.B, "names": nm,
           "measures": M, "probes": M["probes"], "counts": M["counts"], "controls_file": cp, "input_hashes": hashes, "elapsed_s": round(time.time() - t0, 1)}
    with open(X.unique_path(a.out_dir, RESULT_FILE), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=X._jd)
    write_verdict(X.unique_path(a.out_dir, VERDICT_FILE), res)
    print("ZZ DSCALE-PRES NAME %s" % nm["name"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
