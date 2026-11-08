#!/usr/bin/env python3
"""M-SHUF (WIRING_PLAN section 4): build the shuffled behaviour-label set, and the label-level pre-check.  NOT RUN on real data (no label set exists yet); tested on synthetic files (test_mshuf.py).

  shuffle   python refe/mshuf_labels.py shuffle --in <behaviour dir> --out <dir> [--seed 20261108]
            For every side file and every LOG, the (q, act) payloads of the log's records are permuted across the log's (log, token, step, rank) keys by a fixed-seed DERANGEMENT (no record keeps its own
            payload), so the per-term marginals (activation rate, mean q, the multiset of rows) are preserved EXACTLY and only the label-to-scene link is destroyed.  A log with ONE record has no derangement
            inside the log: such records are deranged ACROSS logs (cyclic shift over the sorted singleton keys of the whole set; a single singleton overall stays and is COUNTED).  Keys, ckpt_step, label_version
            and terms_version are untouched, so load_beh_terms joins the shuffled set exactly as it joins the true one.  Writes <out>/behaviour_*.jsonl + <out>/MSHUF_MANIFEST.json (seed, counts, fixed
            points = must be 0 bar the single-singleton case, marginals equal, sha256 of every input and output file).  Refuses --out == --in.
  precheck  python refe/mshuf_labels.py precheck --beh <behaviour dir> --onpolicy <dir> --pdm <dir> [--cols 0,1,2,4]
            The cheap label-level check, no training: within-set Spearman rho between each term's q (over its ACTIVE entries) and agg_v1(tg[..., :6]) (the PDM score of the same proposals), per term and for the
            listwise product q_prod; a term with |mean rho| > 0.95 is a function of PDMS alone and is DROPPED before any GPU hour.  Prints/writes the report; exit 3 if any term is flagged.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SEED = 20261108
RHO_DROP = 0.95
BEH_GLOB = "behaviour_*.jsonl"


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """a permutation p of range(n) with p[i] != i for all i (n >= 2): a random permutation re-drawn until it has no fixed point (expected ~e draws)"""
    if n < 2:
        raise ValueError("a derangement needs n >= 2")
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def shuffle_records(recs: list, seed: int = SEED) -> tuple:
    """recs: list of record dicts (all files). -> (new records, stats).  Deterministic given the sorted keys and the seed."""
    order = sorted(range(len(recs)), key=lambda i: (recs[i]["key"][0], str(recs[i]["key"][1]), int(recs[i]["key"][2]), int(recs[i]["key"][3]), int(recs[i]["ckpt_step"])))
    by_log: dict = {}
    for i in order:
        by_log.setdefault(recs[i]["key"][0], []).append(i)
    rng = np.random.default_rng(seed)
    payload = {i: (recs[i]["q"], recs[i]["act"]) for i in range(len(recs))}
    new_payload = {}
    singles = []
    for log in sorted(by_log):
        idx = by_log[log]
        if len(idx) == 1:
            singles.append(idx[0])
            continue
        p = derangement(len(idx), rng)
        for j, i in enumerate(idx):
            new_payload[i] = payload[idx[int(p[j])]]
    if len(singles) >= 2:
        for j, i in enumerate(singles):
            new_payload[i] = payload[singles[(j + 1) % len(singles)]]
    elif singles:
        new_payload[singles[0]] = payload[singles[0]]
    out, fixed = [], 0
    for i, r in enumerate(recs):
        q, act = new_payload[i]
        fixed += int(q is payload[i][0])
        out.append(dict(r, q=q, act=act))
    return out, {"records": len(recs), "logs": len(by_log), "singleton_logs": len(singles), "fixed_points": fixed}


def marginals(recs: list) -> dict:
    q = np.asarray([r["q"] for r in recs], np.float64)
    a = np.asarray([r["act"] for r in recs], bool)
    return {"act_rate": a.mean((0, 1)).round(12).tolist(), "q_mean_active": [round(float(q[..., c][a[..., c]].mean()), 12) if a[..., c].any() else None for c in range(q.shape[-1])]}


def cmd_shuffle(args) -> int:
    src, dst = Path(args.inp).resolve(), Path(args.out).resolve()
    if src == dst:
        raise SystemExit("--out must not be --in")
    files = sorted(glob.glob(str(src / BEH_GLOB)))
    if not files:
        raise SystemExit(f"no {BEH_GLOB} in {src}")
    recs, where = [], []
    for fp in files:
        for ln in open(fp, encoding="utf-8"):
            if not ln.strip():
                continue
            d = json.loads(ln)
            if d.get("kind") == "behaviour":
                recs.append(d); where.append(fp)
    new, st = shuffle_records(recs, args.seed)
    dst.mkdir(parents=True, exist_ok=True)
    outs = {}
    for fp in files:
        outs[fp] = open(dst / os.path.basename(fp), "w", encoding="utf-8")
    for r, fp in zip(new, where):
        outs[fp].write(json.dumps(r) + "\n")
    for fh in outs.values():
        fh.close()
    m0, m1 = marginals(recs), marginals(new)
    man = {"seed": args.seed, **st, "marginals_equal": m0 == m1, "marginals": m0,
           "inputs": {os.path.basename(f): _sha(f) for f in files}, "outputs": {os.path.basename(f): _sha(dst / os.path.basename(f)) for f in files}}
    (dst / "MSHUF_MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("seed", "records", "logs", "singleton_logs", "fixed_points", "marginals_equal")}))
    ok = man["marginals_equal"] and (man["fixed_points"] == 0 or (man["fixed_points"] == 1 and man["singleton_logs"] == 1))
    print("ZZMSHUF_" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 3


def spearman(x: np.ndarray, y: np.ndarray):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx, ry = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float), np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else None


def rho_report(terms: dict, get_pdm, cols=(0, 1, 2, 4), names=("q_free", "q_gap", "q_osc", "q_stall", "q_comf")) -> dict:
    """terms: {key: (q [M,5], act [M,5])}; get_pdm(key) -> pdm score [M] (agg_v1 of the six PDM labels) or None.  -> per term and q_prod: mean within-set Spearman rho, n sets, flagged."""
    acc = {n: [] for n in names}
    acc["q_prod"] = []
    for k, (q, act) in terms.items():
        s = get_pdm(k)
        if s is None:
            continue
        s = np.asarray(s, np.float64)
        for c, n in enumerate(names):
            m = act[:, c]
            r = spearman(q[m, c].astype(np.float64), s[m]) if m.sum() >= 3 else None
            if r is not None:
                acc[n].append(r)
        prod = np.where(act[:, list(cols)], q[:, list(cols)], 1.0).prod(-1)
        r = spearman(prod.astype(np.float64), s)
        if r is not None:
            acc["q_prod"].append(r)
    rep = {}
    for n, v in acc.items():
        mu = float(np.mean(v)) if v else None
        rep[n] = {"mean_rho": mu, "sets": len(v), "flag_function_of_pdms": bool(mu is not None and abs(mu) > RHO_DROP)}
    return rep


def cmd_precheck(args) -> int:
    """UNVERIFIED on real data: the bank loader (train.OnPolicyBank, as scorer_finetune.main builds it) needs the on-policy bank + PDM labels, which are not on this machine."""
    import torch
    import scorer_finetune as S
    from model import REFeConfig
    cfg = REFeConfig()
    n_prop = args.n_proposals or cfg.n_proposals
    op = S.T.OnPolicyBank(args.onpolicy, n_prop, args.horizon or cfg.horizon_steps, pdm_labels=args.pdm)
    terms, st = S.load_beh_terms(args.beh, op.by, n_prop)

    def get_pdm(k):
        got = op.get(k[0], k[1], k[2], k[3])
        return None if got is None else S.agg_v1(torch.as_tensor(got[1])[..., :6].float()).numpy()
    cols = tuple(int(c) for c in args.cols.split(","))
    rep = rho_report(terms, get_pdm, cols)
    rep["_join"] = st
    print(json.dumps(rep, indent=1))
    flagged = [n for n, v in rep.items() if not n.startswith("_") and v["flag_function_of_pdms"]]
    print("ZZMSHUF_PRECHECK_" + ("FLAGGED " + ",".join(flagged) if flagged else "OK"))
    return 3 if flagged else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shuffle")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=SEED)
    p = sub.add_parser("precheck")
    p.add_argument("--beh", required=True); p.add_argument("--onpolicy", required=True); p.add_argument("--pdm", required=True); p.add_argument("--cols", default="0,1,2,4"); p.add_argument("--n-proposals", type=int, default=0); p.add_argument("--horizon", type=int, default=0)
    a = ap.parse_args()
    return cmd_shuffle(a) if a.cmd == "shuffle" else cmd_precheck(a)


if __name__ == "__main__":
    sys.exit(main())
