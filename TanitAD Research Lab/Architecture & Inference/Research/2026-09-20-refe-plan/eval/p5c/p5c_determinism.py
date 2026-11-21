#!/usr/bin/env python3
"""P5-C read instrument B10: the inference-determinism subset pass (PREREG_P5C.md section 5, "Inference run").

Card text: REFe inference is deterministic. This is ASSERTED: a second pass of each arm over a seeded 10 % subset (1,215 tokens, seed 20261112)
must be bit-identical in per-token PDMS under both scorers. A mismatch is a defect, investigated before any read.
Section 11: the pass is an identity check on the same model and is NOT counted in the K ledger.

What is implemented (nothing is decided beyond the card; the two gaps are card questions Q-7 / Q-8, see HANDOFF):
  select   --plans-dir D --out subset.json [--frac 0.10 --seed 20261112]
             the subset: k = ceil(frac * n) tokens (12,146 -> 1,215, the card's number), drawn by numpy default_rng(seed).choice(n, k, replace=False)
             over the SORTED token list, returned sorted. The same plans_*.npz token set is required for every arm (else REFUSED).
  plans    --plans-dir D --subset subset.json --out-dir D2
             D2/plans_<arm>.npz = every plans_*.npz restricted to the subset (every array whose first axis is the token axis). Feed D2 to
             `p5c_read_prep.py score` = the second pass (same scorer, same flags, same cache route as the first).
  compare  --rows1 "<pass-1 rows glob>" --rows2 "<pass-2 rows glob>" --subset subset.json --arms C1 C0 base [--human human] --out result.json
             per arm x scorer (tracked, transparent) x metric (nc dac ddc ttc ep c pdms): max |pass1 - pass2| and the number of tokens over the tolerance
             (default 0.0 = bit-identical). PASS iff full coverage, 0 error rows, 0 tokens over tolerance on EVERY metric (the card names PDMS; a sub-score
             that moves is non-determinism too, so this reading is the stricter one, Q-7), and pass 2 is a genuinely separate pass.
             The result JSON carries a boolean `pass`, which `p5c_read.py --controls` consumes, and `control: "B10_determinism"`, which `--require-kinds` reads.
  Guards against a vacuous PASS (each has a RED test): identical row-file paths; pass 2 rows that are a byte copy of pass 1 (every row's `t_s` equal);
  a subset that does not match the declared (n, seed, rule); a token set in pass 2 that is not exactly the subset.
Evidence class of anything this prints: MEASURED on the passes supplied. It says nothing about P5-C until real arms are supplied.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FRAC, SEED = 0.10, 20261112
SUBS = ("nc", "dac", "ddc", "ttc", "ep", "c")
METRICS = SUBS + ("pdms",)
SCORERS = ("tracked", "transparent")
KIND = "B10_determinism"
RULE = "k = ceil(frac * n); default_rng(seed).choice(n, k, replace=False) over the sorted tokens; returned sorted"


def tokens_sha256(tokens) -> str:
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def choose_subset(tokens, frac: float = FRAC, seed: int = SEED) -> list:
    toks = sorted(tokens)
    n = len(toks)
    if n == 0:
        raise ValueError("no tokens")
    k = min(n, max(1, math.ceil(frac * n - 1e-9)))
    idx = np.sort(np.random.default_rng(seed).choice(n, k, replace=False))
    return [toks[i] for i in idx]


def plan_files(plans_dir: str) -> dict:
    fs = sorted(glob.glob(os.path.join(plans_dir, "plans_*.npz")))
    if not fs:
        raise SystemExit(f"REFUSED: no plans_*.npz in {plans_dir}")
    return {os.path.basename(f)[len("plans_"):-len(".npz")]: f for f in fs}


def plans_tokens(plans_dir: str) -> list:
    ref = None
    for arm, f in plan_files(plans_dir).items():
        t = [str(x) for x in np.load(f, allow_pickle=False)["tokens"]]
        if ref is None:
            ref = sorted(t)
        elif sorted(t) != ref:
            raise SystemExit(f"REFUSED: arm {arm} has a different token set from the others (n {len(t)} vs {len(ref)}): the arms are not like-for-like")
    return ref


def make_subset_doc(tokens, frac: float = FRAC, seed: int = SEED) -> dict:
    sub = choose_subset(tokens, frac, seed)
    return {"kind": "B10_subset", "rule": RULE, "frac": frac, "seed": seed, "n_total": len(tokens), "n_subset": len(sub),
            "population_sha256": tokens_sha256(sorted(tokens)), "tokens_sha256": tokens_sha256(sub), "tokens": sub}


def check_subset_doc(doc: dict, population=None) -> list:
    """-> list of problems (empty = the subset is exactly what the declared rule gives)."""
    bad = []
    toks = doc.get("tokens") or []
    if doc.get("tokens_sha256") != tokens_sha256(toks):
        bad.append("tokens_sha256 does not match the tokens (the file was edited)")
    if sorted(toks) != toks or len(set(toks)) != len(toks):
        bad.append("tokens are not sorted-unique")
    if doc.get("n_subset") != len(toks):
        bad.append("n_subset does not match the token list")
    if population is not None:
        if tokens_sha256(sorted(population)) != doc.get("population_sha256"):
            bad.append("population_sha256: the subset was drawn from another token set than the plans supplied")
        elif choose_subset(population, doc.get("frac", FRAC), doc.get("seed", SEED)) != toks:
            bad.append("the tokens are not what the declared (frac, seed, rule) gives on this population")
    return bad


def read_rows(globs) -> tuple:
    rows, errs = {}, {}
    paths = [p for g in ([globs] if isinstance(globs, str) else globs) for p in sorted(glob.glob(g))]
    for fp in paths:
        for ln in open(fp, encoding="utf-8"):
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if "error" in r:
                errs[r.get("token")] = str(r["error"])[:200]
            else:
                rows[r["token"]] = r
                errs.pop(r["token"], None)
    return rows, errs, [os.path.abspath(p) for p in paths]


def compare(rows1: dict, rows2: dict, subset_tokens, arms, errs1=None, errs2=None, tol: float = 0.0, paths1=None, paths2=None) -> dict:
    """pure comparison of two passes' rows over the subset. See the module docstring for PASS."""
    exp = sorted(subset_tokens)
    problems = []
    if paths1 and paths2 and set(paths1) & set(paths2):
        problems.append(f"pass 1 and pass 2 share row files ({sorted(set(paths1) & set(paths2))[:2]}): not two passes")
    miss1 = [t for t in exp if t not in rows1]
    miss2 = [t for t in exp if t not in rows2]
    extra2 = sorted(set(rows2) - set(exp))
    if miss1:
        problems.append(f"{len(miss1)} subset tokens missing from pass 1 (first {miss1[0]})")
    if miss2:
        problems.append(f"{len(miss2)} subset tokens missing from pass 2 (first {miss2[0]})")
    if extra2:
        problems.append(f"pass 2 holds {len(extra2)} tokens outside the subset (first {extra2[0]}): not the declared subset")
    if errs1 or errs2:
        problems.append(f"error rows: pass 1 {len(errs1 or {})}, pass 2 {len(errs2 or {})}")
    both = [t for t in exp if t in rows1 and t in rows2]
    ts1 = [rows1[t].get("t_s") for t in both]
    ts2 = [rows2[t].get("t_s") for t in both]
    copy_suspect = bool(both) and all(isinstance(x, (int, float)) for x in ts1 + ts2) and ts1 == ts2
    if copy_suspect:
        problems.append("every row's t_s is identical in the two passes: pass 2 is a copy of pass 1, not a second run")
    out_arms, over_tokens, max_all = {}, set(), 0.0
    for a in arms:
        out_arms[a] = {}
        for s in SCORERS:
            mx = {m: 0.0 for m in METRICS}
            n_over = {m: 0 for m in METRICS}
            n_cmp = 0
            for t in both:
                try:
                    x, y = rows1[t]["arms"][a][s], rows2[t]["arms"][a][s]
                except KeyError:
                    problems.append(f"arm {a} / scorer {s} absent on token {t}")
                    break
                n_cmp += 1
                for m in METRICS:
                    d = abs(float(x[m]) - float(y[m]))
                    if not math.isfinite(d):
                        d = float("inf")
                    mx[m] = max(mx[m], d)
                    if d > tol:
                        n_over[m] += 1
                        over_tokens.add(t)
            max_all = max(max_all, max(mx.values()))
            out_arms[a][s] = {"n_compared": n_cmp, "max_abs_diff": mx, "n_tokens_over_tolerance_by_metric": n_over,
                              "n_pdms_over_tolerance": n_over["pdms"]}
    ok = (not problems) and not over_tokens and len(both) == len(exp) and len(exp) > 0
    return {"control": KIND, "instrument": "p5c_determinism.py B10", "card": "PREREG_P5C.md section 5 (inference run)",
            "tolerance": tol, "n_subset": len(exp), "n_compared": len(both), "arms": out_arms,
            "n_tokens_over_tolerance": len(over_tokens), "over_tolerance_first10": sorted(over_tokens)[:10], "max_abs_diff_all": max_all,
            "problems": problems, "pass": bool(ok), "verdict": "PASS" if ok else "FAIL",
            "not_counted_in_K": True, "evidence": "MEASURED on the two passes supplied"}


def cmd_select(a) -> int:
    pop = plans_tokens(a.plans_dir)
    doc = make_subset_doc(pop, a.frac, a.seed)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    print("ZZP5C_B10_SELECT", doc["n_subset"], "of", doc["n_total"], "seed", doc["seed"], doc["tokens_sha256"][:12])
    return 0


def cmd_plans(a) -> int:
    doc = json.load(open(a.subset, encoding="utf-8"))
    pop = plans_tokens(a.plans_dir)
    bad = check_subset_doc(doc, pop)
    if bad:
        print("REFUSED:", "; ".join(bad), file=sys.stderr)
        return 2
    keep = set(doc["tokens"])
    os.makedirs(a.out_dir, exist_ok=True)
    for arm, f in plan_files(a.plans_dir).items():
        z = np.load(f, allow_pickle=False)
        tok = [str(t) for t in z["tokens"]]
        sel = np.array([i for i, t in enumerate(tok) if t in keep])
        out = {k: (np.asarray(z[k])[sel] if np.asarray(z[k]).ndim >= 1 and len(z[k]) == len(tok) else np.asarray(z[k])) for k in z.files}
        np.savez(os.path.join(a.out_dir, f"plans_{arm}.npz"), **out)
    print("ZZP5C_B10_PLANS", len(keep), "tokens x", len(plan_files(a.plans_dir)), "arms ->", a.out_dir)
    return 0


def cmd_compare(a) -> int:
    doc = json.load(open(a.subset, encoding="utf-8"))
    bad = check_subset_doc(doc)
    if bad:
        print("REFUSED:", "; ".join(bad), file=sys.stderr)
        return 2
    r1, e1, p1 = read_rows(a.rows1)
    r2, e2, p2 = read_rows(a.rows2)
    arms = list(a.arms) + ([a.human] if a.human else [])
    res = compare(r1, r2, doc["tokens"], arms, e1, e2, a.tolerance, p1, p2)
    res["subset"] = {k: doc[k] for k in ("frac", "seed", "rule", "n_total", "n_subset", "population_sha256", "tokens_sha256")}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("ZZP5C_B10", res["verdict"], "subset", res["n_subset"], "tokens over tolerance", res["n_tokens_over_tolerance"], "max |diff|", res["max_abs_diff_all"])
    return 0 if res["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select")
    s.add_argument("--plans-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--frac", type=float, default=FRAC)
    s.add_argument("--seed", type=int, default=SEED)
    p = sp.add_parser("plans")
    p.add_argument("--plans-dir", required=True)
    p.add_argument("--subset", required=True)
    p.add_argument("--out-dir", required=True)
    c = sp.add_parser("compare")
    c.add_argument("--rows1", nargs="+", required=True)
    c.add_argument("--rows2", nargs="+", required=True)
    c.add_argument("--subset", required=True)
    c.add_argument("--arms", nargs="+", required=True)
    c.add_argument("--human", default=None)
    c.add_argument("--tolerance", type=float, default=0.0)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"select": cmd_select, "plans": cmd_plans, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
