"""P3 (THEORY_CRAWL.md frozen 6e83451c): is every banked downstream lever with a measured (M1 closure, guard) pair on or below ONE monotone frontier?
Fails if one lever dominates another on BOTH axes by more than its interval.  Raw reader JSONs only (paths below); zero training, CPU.
closure = -dM1(X - Y) / dM1(Y - P10)   (share of the base's M1 crawl excess over P10 that the lever closes; the readers' `share_closed`)
guard cost 1 = dADE6(X - Y) in metres (point, reader's paired episode-cluster 95 % interval); guard cost 2 = number of non-crawl rows separated-worse in X - Y.
Interval of closure: the M1 interval of X - Y divided by the same denominator (the denominator is held at its point value: DISCLOSED, it ignores the base-excess uncertainty).
"""
import json, sys, itertools
import numpy as np

A = "E:/Projects/TanitAD-artifacts/dagland"
M = A + "/motion_c"
HERE = M + "/theory/wsp_read"
LOWER_BETTER_EXCEPT = {"TAC-LAT", "TAC-LON"}
rec = []
unplaced = []

def worse_rows(rows):
    out = []
    for name, r in rows.items():
        if name == "M1 crawl share":
            continue
        if not r.get("separated"):
            continue
        d = r["delta"]
        worse = (d < 0) if name in LOWER_BETTER_EXCEPT else (d > 0)
        if worse:
            out.append(name)
    return out

def add(lever, kind, base, seed, xy, yp, src, note=""):
    m1 = xy["M1 crawl share"]; den = yp["M1 crawl share"]["delta"]
    if not den or den <= 0:
        unplaced.append((lever, base, seed, f"base M1 excess over P10 is {den}: closure undefined"))
        return
    cl = -m1["delta"] / den
    rec.append({"lever": lever, "kind": kind, "base": base, "seed": seed, "closure": cl, "closure_lo": -m1["hi"] / den, "closure_hi": -m1["lo"] / den,
                "m1_separated": bool(m1["separated"]), "m1_delta": m1["delta"], "base_m1_excess": den,
                "ade6": xy["ADE6"]["delta"], "ade6_lo": xy["ADE6"]["lo"], "ade6_hi": xy["ADE6"]["hi"], "ade6_separated": bool(xy["ADE6"]["separated"]),
                "worse_rows": worse_rows(xy), "n_worse": len(worse_rows(xy)), "src": src, "note": note})

# ---- LF family (base C0, T1-kin reindexed, bare arm = S then F; X - Y = RILF - RIC0)
d = json.load(open(M + "/lf_result_pkg/lf_read/raw/LF_READ.json"))
for dose in ("050", "075"):
    for s in ("s0", "s1"):
        sd = d["doses"][dose]["seeds"][s]
        add(f"LF{dose}", "ADMISSIBLE", "C0", s, sd["RILF_minus_RIC0"], sd["RIC0_minus_P10"], "lf_result_pkg/lf_read/raw/LF_READ.json")
        rep = d["doses"][dose]["rules"]["P"]["per_seed"][s]["share_closed"]
        assert abs(rep - rec[-1]["closure"]) < 1e-9, (dose, s, rep, rec[-1]["closure"])        # control: my closure equals the reader's share_closed
d = json.load(open(M + "/lfo_result_pkg/lfo_read/raw/LFO_READ.json"))
for s in ("s0", "s1"):
    sd = d["arm"]["seeds"][s]
    add("A4 LFO075", "CEILING", "C0", s, sd["RILFO_minus_RIC0"], sd["RIC0_minus_P10"], "lfo_result_pkg/lfo_read/raw/LFO_READ.json")
d = json.load(open(M + "/lf5_result_pkg/lf5_read/raw/LF5_READ.json"))
for s in ("s0", "s1"):
    sd = d["LF100"]["seeds"][s]
    add("A5 LF100", "ADMISSIBLE", "C0", s, sd["RILF_minus_RIC0"], sd["RIC0_minus_P10"], "lf5_result_pkg/lf5_read/raw/LF5_READ.json")
    assert abs(d["LF100"]["rules"]["P"]["per_seed"][s]["share_closed"] - rec[-1]["closure"]) < 1e-9
for arm in ("LFO100", "LFN100x10", "LFN100x20"):
    for s in ("s0", "s1"):
        sd = d["ceiling"][arm]["seeds"][s]
        add(f"A5 {arm}", "CEILING", "C0", s, sd["RILFO_minus_RIC0"], sd["RIC0_minus_P10"], "lf5_result_pkg/lf5_read/raw/LF5_READ.json")
d = json.load(open(M + "/lf6_result_pkg/lf6_read/raw/LF6_READ.json"))
for arm in ("LFB05", "LFB10"):
    for s in ("s0", "s1"):
        sd = d["LFB"][arm]["seeds"][s]
        add(f"A6 {arm}", "ADMISSIBLE", "C0", s, sd["RILF_minus_RIC0"], sd["RIC0_minus_P10"], "lf6_result_pkg/lf6_read/raw/LF6_READ.json")
for arm in ("LFOB05", "LFOB10"):
    for s in ("s0", "s1"):
        sd = d["ceiling"][arm]["seeds"][s]
        add(f"A6 {arm}", "CEILING", "C0", s, sd["RILFO_minus_RIC0"], sd["RIC0_minus_P10"], "lf6_result_pkg/lf6_read/raw/LF6_READ.json")
# ---- decode levers on DAG-OACC and SOUP (X - Y with Y = the same base after S)
def decode(path, arms_filter=None, single=False):
    dd = json.load(open(path))
    for key, a in dd["arms"].items():
        if single:
            base_s, lever = key, "curvature box"
        else:
            base_s, lever = key.split(":")
        base, s = base_s.split("_")
        base = {"CTRL": "DAG-OACC"}.get(base, base)
        add(lever, "DECODE", base, s, a["rows"]["X_minus_Y"], a["rows"]["Y_minus_P10"], path.replace(A + "/", ""))
decode(A + "/kappa_t1/raw/KAPPA_T1.json", single=True)
decode(A + "/feas2_t1/raw/FEAS2_T1.json")
decode(A + "/feas2_t1/raw/FEAS3_T1.json")

# ---- named, not placed
unplaced.append(("DAG-OACC / SOUP (as levers)", "-", "-", "they are the BASES of the decode levers (Y of X - Y), not downstream levers; their own point is closure 0, guard 0 by construction"))
unplaced.append(("A5/A6 'C' composites (S -> F -> kbox)", "-", "-", "composites add the curvature box; read beside, not placed (the bare arm is the deciding contrast in every reader)"))

# ---- dominance test
def dom_strict(a, b):     # a dominates b: closure intervals disjoint upward AND ADE6 intervals disjoint downward
    return a["closure_lo"] > b["closure_hi"] and a["ade6_hi"] < b["ade6_lo"]
def dom_point(a, b):
    return a["closure"] > b["closure"] and a["ade6"] < b["ade6"]
def dom_strict_cnt(a, b):  # closure disjoint upward AND strictly fewer separated-worse rows
    return a["closure_lo"] > b["closure_hi"] and a["n_worse"] < b["n_worse"]
res = {"records": rec, "unplaced": unplaced, "tests": {}}
def pairs(scope):
    out = {"strict_ade6": [], "point_ade6": [], "strict_count": []}
    for s in ("s0", "s1"):
        R = [r for r in rec if r["seed"] == s and scope(r)]
        for a, b in itertools.permutations(R, 2):
            tag = (f"{a['lever']}[{a['base']}] > {b['lever']}[{b['base']}]", s)
            if dom_strict(a, b): out["strict_ade6"].append(tag)
            if dom_point(a, b): out["point_ade6"].append(tag)
            if dom_strict_cnt(a, b): out["strict_count"].append(tag)
    return out
scopes = {"all": lambda r: True, "C0-based (LF, A4, A5, A6)": lambda r: r["base"] == "C0", "admissible only": lambda r: r["kind"] == "ADMISSIBLE",
          "decode levers": lambda r: r["kind"] == "DECODE", "NON-CEILING (ADMISSIBLE + DECODE)": lambda r: r["kind"] != "CEILING", "C0-based admissible": lambda r: r["base"] == "C0" and r["kind"] == "ADMISSIBLE"}
for nm, sc in scopes.items():
    p = pairs(sc)
    both = {}
    for kind, lst in p.items():
        c = {}
        for t, s in lst:
            c.setdefault(t, set()).add(s)
        both[kind] = sorted(t for t, ss in c.items() if ss == {"s0", "s1"})
    res["tests"][nm] = {"n_records": sum(1 for r in rec if sc(r)), **{k + "_n_pairs_by_seed": len(v) for k, v in p.items()}, **{k + "_pairs_at_BOTH_seeds": both[k] for k in both},
                        **{k + "_pairs_s0": [t for t, s in v if s == "s0"][:40] for k, v in p.items() if k == "strict_ade6"},
                        **{k + "_pairs_s1": [t for t, s in v if s == "s1"][:40] for k, v in p.items() if k == "strict_ade6"}}
# ---- monotone check: Spearman between guard cost and closure over placed points (does closure buy guard?)
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])
res["spearman_closure_vs_ade6"] = {}
for s in ("s0", "s1"):
    for nm, sc in (("all", lambda r: True), ("C0-based", lambda r: r["base"] == "C0"), ("admissible", lambda r: r["kind"] == "ADMISSIBLE"), ("decode", lambda r: r["kind"] == "DECODE"), ("non-ceiling", lambda r: r["kind"] != "CEILING")):
        R = [r for r in rec if r["seed"] == s and sc(r)]
        res["spearman_closure_vs_ade6"][f"{nm}.{s}"] = {"n": len(R), "rho_closure_ade6": spearman([r["closure"] for r in R], [r["ade6"] for r in R]),
                                                          "rho_closure_nworse": spearman([r["closure"] for r in R], [r["n_worse"] for r in R])}
json.dump(res, open(HERE + "/P3_RESULT.json", "w"), indent=1)
# ---- digest
print(f"{'lever':22s} {'kind':11s} {'base':9s} s  closure [lo,hi]       ADE6 [lo,hi]            n_worse")
for r in sorted(rec, key=lambda r: (r["base"], r["seed"], r["closure"])):
    print(f"{r['lever']:22s} {r['kind']:11s} {r['base']:9s} {r['seed']} {r['closure']:+.3f} [{r['closure_lo']:+.3f},{r['closure_hi']:+.3f}]  {r['ade6']:+.4f} [{r['ade6_lo']:+.3f},{r['ade6_hi']:+.3f}]  {r['n_worse']:2d}")
for nm, t in res["tests"].items():
    print(nm, {k: v for k, v in t.items() if "n_pairs" in k or "BOTH" in k or k == "n_records"})
print(res["spearman_closure_vs_ade6"])
print("UNPLACED", unplaced)

# ---- plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = {"ADMISSIBLE": "#1f6feb", "CEILING": "#d1242f", "DECODE": "#6e7781"}
MK = {"C0": "o", "DAG-OACC": "s", "SOUP": "^"}
for yaxis, fn, lab in (("ade6", "P3_FRONTIER_ADE6.png", "guard cost 1: ADE6 X - Y (m)"), ("n_worse", "P3_FRONTIER_COUNT.png", "guard cost 2: non-crawl rows separated-worse (count)")):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, s in zip(axs, ("s0", "s1")):
        for r in [r for r in rec if r["seed"] == s]:
            y = r[yaxis]
            xerr = [[r["closure"] - r["closure_lo"]], [r["closure_hi"] - r["closure"]]]
            yerr = [[y - r["ade6_lo"]], [r["ade6_hi"] - y]] if yaxis == "ade6" else None
            ax.errorbar(r["closure"], y, xerr=xerr, yerr=yerr, fmt=MK[r["base"]], color=COL[r["kind"]], ms=6, alpha=0.85, elinewidth=0.7, capsize=0)
            if r["kind"] != "DECODE" or r["closure"] > 0.03:
                ax.annotate(r["lever"].replace("A5 ", "").replace("A6 ", "").replace("A4 ", ""), (r["closure"], y), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        if yaxis == "ade6":
            ax.axhline(0.0604, color="k", lw=0.6, ls="--"); ax.text(-0.19, 0.075, "CL guard +0.0604 m", fontsize=7)
        ax.axvline(0, color="k", lw=0.4); ax.axvline(0.5, color="k", lw=0.6, ls=":"); ax.set_title(f"seed {s}: M1 closure vs guard cost"); ax.set_xlabel("M1 closure (share of the base's crawl excess over P10)")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel(lab)
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker="o", color=COL[k], ls="", label=k) for k in COL] + [Line2D([0], [0], marker=MK[b], color="k", ls="", label="base " + b) for b in MK]
    axs[1].legend(handles=h, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(HERE + "/" + fn, dpi=130); plt.close(fig)
print("plots written")
