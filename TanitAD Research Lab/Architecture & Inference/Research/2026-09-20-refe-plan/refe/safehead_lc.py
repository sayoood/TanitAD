#!/usr/bin/env python3
"""DATA-SCALE M6 (PREREG_DATASCALE.md 5): the LEARNING CURVE.  Descriptive, it decides nothing.  CPU, no GPU, no trunk, writes NO head.

Nested fit sets of n logs (default 35, 70, 110, N).  The first set is EXACTLY the parent's logs, given by --parent-logs (a list file).  Each larger set adds the other logs in
sha256(log name) order.  Every set runs 5-fold log CV WITHIN ITSELF (folds = sorted log name, index mod 5, over the set's logs) with the trainer's own recipe and `cv_select`
(so the per-step OOF table, the `cv5m` margin and the guard are the trainer's code, not a copy), and reports the OOF gain(s) with the log-cluster CI at every saved step.

CTRL-PARENT: with --parent-cv-select <the parent's banked cv_select.json>, the n = |parent logs| set's no-margin argmax step and its OOF PDMS must match the banked table to
--atol (1e-9).  A mismatch means the rebuilt pipeline is not the parent's: the run exits 1 and writes lc.json with `ctrl_parent.ok = false`.

    python safehead_lc.py --parent-logs <parent_logs.txt> --sizes 35,70,110,N [--parent-cv-select <cv_select.json>] \\
        <the trainer's argv: --cache-index ... --init-from ... --heldout-logs ... --navtest-logs ... --out <empty dir> --seed 0 --steps 2000 --lr 0.0003 --batch 32 --gamma 2.0
         --rows 2|0,1,3,5 --save-steps 0,250,500,1000,2000 --select cv5m>
It saves no head, but it TRAINS heads (T.cv_select), so it is a training launch for CLAUDE.md "NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS" (PI 2026-09-26): like the trainer's
`main`, it refuses under a regression arm and calls T.require_token on the trainer argv (the argv left after its own flags), with no bypass.  DATA-SCALE CODE AMENDMENT 1.
It also refuses a navtest log or token in the fit sets (CTRL-LEAK) and a feature mismatch (CTRL-FEAT).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import safehead_data as D  # noqa: E402
import safehead_train as T  # noqa: E402


def nested_sets(all_logs, parent_logs, sizes) -> dict:
    """{n: sorted list of logs}.  `sizes` are ints (the largest may be len(all_logs)).  The smallest must be >= the parent's logs; a set contains the parent's logs, plus the
    first (n - |parent|) of the OTHER logs in sha256(log name) order, so the sets are nested."""
    all_logs, parent = sorted(set(all_logs)), sorted(set(parent_logs))
    if len(parent) != len(parent_logs):
        raise SystemExit("REFUSED: the parent log list has a duplicate log")
    missing = sorted(set(parent) - set(all_logs))
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} parent log(s) are not in the cache (first {missing[:3]})")
    others = sorted(set(all_logs) - set(parent), key=lambda lg: hashlib.sha256(lg.encode("utf-8")).hexdigest())
    out = {}
    for n in sorted(set(sizes)):
        if n < len(parent) or n > len(all_logs):
            raise SystemExit(f"REFUSED: set size {n} is outside [{len(parent)} (the parent's logs), {len(all_logs)} (all logs)]")
        out[n] = sorted(parent + others[: n - len(parent)])
    return out


def best_step(sel: dict) -> dict:
    """The set's best step by OOF gain among the saved steps > 0 (descriptive; the `cv5m` choice is `chosen_step`)."""
    steps = sorted(int(s) for s in sel["margin"] if int(s) > 0)
    top = max(steps, key=lambda s: (sel["margin"][str(s)]["gain"]["point"], -s))
    return {"step": top, "gain": sel["margin"][str(top)]["gain"]}


def ctrl_parent(sel: dict, banked: dict, atol: float) -> dict:
    """The n = |parent| set's no-margin argmax and its OOF PDMS (every saved step) against the banked parent cv_select.json."""
    bs = sorted(int(s) for s in banked["per_step"])
    diffs = {str(s): abs(sel["per_step"][str(s)]["oof_pdms"] - banked["per_step"][str(s)]["oof_pdms"]) if str(s) in sel["per_step"] else None for s in bs}
    worst = max((v for v in diffs.values() if v is not None), default=None)
    ok = (sel["no_margin_argmax_step"] == banked["chosen_step"] and all(v is not None and v <= atol for v in diffs.values()) and worst is not None
          and sel["n_logs"] == banked.get("n_logs", sel["n_logs"]) and sel["n_tokens"] == banked.get("n_tokens", sel["n_tokens"]))
    return {"ok": bool(ok), "atol": atol, "banked_chosen_step": banked["chosen_step"], "our_no_margin_argmax_step": sel["no_margin_argmax_step"],
            "oof_pdms_abs_diff_per_step": diffs, "max_abs_diff": worst, "banked_n_logs": banked.get("n_logs"), "banked_n_tokens": banked.get("n_tokens"),
            "our_n_logs": sel["n_logs"], "our_n_tokens": sel["n_tokens"]}


def run(argv) -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent-logs", required=True, help="the parent's fit logs, one per line (the n = |list| set is exactly these)")
    ap.add_argument("--sizes", default="35,70,110,N", help="comma list of set sizes; N = all navtrain logs in the cache")
    ap.add_argument("--parent-cv-select", default=None, help="CTRL-PARENT: the parent's banked cv_select.json (lever B: EPH_s0)")
    ap.add_argument("--atol", type=float, default=1e-9)
    a, rest = ap.parse_known_args(argv)
    cfg = T.build_config(T.build_parser().parse_args(rest))
    if cfg.select != "cv5m" or cfg.permute_labels_seed >= 0 or cfg.smoke_steps or cfg.build_only:
        raise SystemExit("REFUSED: the learning curve runs --select cv5m, with no --permute-labels-seed, --smoke-steps or --build-only")
    if 0 not in cfg.save_steps:
        raise SystemExit("REFUSED: --save-steps must include 0")
    if os.environ.get(T.ARM_ENV):                                                                 # AMENDMENT 1: the trainer's own rule (safehead_train.run), no bypass
        raise SystemExit(f"REFUSED: a regression arm is active ({os.environ[T.ARM_ENV]}): a training launch never runs under one")
    tok = T.require_token(rest)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"REFUSED: --out {out} is not empty")
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    W0, b0 = D.c1_head_from_checkpoint(cfg.init_from)
    c = D.load_cache(cfg.cache_index, splits=D.SPLITS_NAVTRAIN, require_sealed=True)
    leak = D.leak_problems(c, D.read_log_list(cfg.heldout_logs), D.read_log_list(cfg.navtest_logs))
    if leak:
        raise SystemExit("REFUSED (CTRL-LEAK): " + " | ".join(leak))
    dl = np.abs(D.direct_logits(c["s"], W0, b0) - c["logits"])
    if dl.max() > T.FEAT_MAX or np.percentile(dl, 99) > T.FEAT_P99:
        raise SystemExit(f"REFUSED (CTRL-FEAT): C1's head on the cached s is not the cached logits (max {float(dl.max())})")
    all_logs = sorted(set(map(str, c["log"])))
    sizes = [len(all_logs) if x.strip().upper() == "N" else int(x) for x in a.sizes.split(",") if x.strip()]
    parent_logs = D.read_log_list(a.parent_logs)
    sets = nested_sets(all_logs, parent_logs, sizes)
    rows = list(cfg.rows)
    clog = np.array(c["log"])
    res = {}
    for n, logs in sets.items():
        pool = np.where(np.isin(clog, logs))[0]
        fold_of_log = D.cv_folds(clog[pool], T.N_FOLDS)
        tok_fold = np.array([fold_of_log[lg] for lg in clog[pool]])
        for k in range(T.N_FOLDS):
            if (tok_fold != k).sum() < cfg.batch or (tok_fold == k).sum() == 0:
                raise SystemExit(f"REFUSED: n = {n}: CV fold {k} has {(tok_fold == k).sum()} test / {(tok_fold != k).sum()} train tokens (batch {cfg.batch})")
        Y = np.stack([c[D.ROW_LABEL_ALL[r]][pool] for r in rows], -1).astype(np.float32)
        S, YT = torch.from_numpy(np.ascontiguousarray(c["s"][pool])), torch.from_numpy(Y)
        extra_row = 2 if 2 not in rows else 0
        EXTRA = torch.from_numpy(c[D.ROW_LABEL_ALL[extra_row]][pool].astype(np.float32))
        t1 = time.time()
        sel = T.cv_select(c, pool, tok_fold, S, YT, EXTRA, cfg, None, W0, b0, sorted(cfg.save_steps), cfg.steps, None)
        res[str(n)] = {"n_logs": sel["n_logs"], "n_tokens": sel["n_tokens"], "per_step": sel["per_step"], "margin": sel["margin"], "per_log": sel["per_log"],
                       "chosen_step": sel["chosen_step"], "no_margin_argmax_step": sel["no_margin_argmax_step"], "best_step": best_step(sel), "seconds": round(time.time() - t1, 1)}
        print(f"n={n} logs={sel['n_logs']} tokens={sel['n_tokens']} cv5m_chosen={sel['chosen_step']} no_margin_argmax={sel['no_margin_argmax_step']} "
              f"best_gain={res[str(n)]['best_step']['gain']['point']:.4f}@{res[str(n)]['best_step']['step']} ({res[str(n)]['seconds']} s)", flush=True)
    ns = sorted(int(k) for k in res)
    pts = [res[str(n)]["best_step"]["gain"]["point"] for n in ns]
    summary = {"sizes": ns, "best_step_gain_point": pts, "monotone_strictly_increasing": all(y > x for x, y in zip(pts, pts[1:])),
               "monotone_nondecreasing": all(y >= x for x, y in zip(pts, pts[1:])), "point_at_largest_exceeds_point_at_smallest": bool(pts[-1] > pts[0]),
               "definition": "best-step gain = max over saved steps > 0 of the OOF gain point (x100) of that set; descriptive, it decides nothing (PREREG_DATASCALE 5, M6)"}
    cp = None
    if a.parent_cv_select:
        banked = json.loads(Path(a.parent_cv_select).read_text(encoding="utf-8"))
        if len(parent_logs) not in sets:
            raise SystemExit(f"REFUSED: CTRL-PARENT needs the n = {len(parent_logs)} set among --sizes")
        cp = ctrl_parent(res[str(len(parent_logs))], banked, a.atol)
    rep = {"argv": list(argv), "evidence": "MEASURED (out-of-fold navtrain, rebuilt-cache PDMS; descriptive, NOT navtest)", "parent_logs": len(parent_logs), "sets": res,
           "summary": summary, "ctrl_parent": cp, "token": tok, "seconds": round(time.time() - t0, 1),
           "set_logs_sha256": {str(n): hashlib.sha256("\n".join(lg).encode("utf-8")).hexdigest() for n, lg in sets.items()}}
    (out / "lc.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    if cp is not None and not cp["ok"]:
        print(f"ZZSHLC_FAIL CTRL-PARENT: {json.dumps({k: cp[k] for k in ('banked_chosen_step', 'our_no_margin_argmax_step', 'max_abs_diff')})}", flush=True)
        return 1
    print(f"ZZSHLC_DONE sets {ns} ctrl_parent {None if cp is None else 'PASS'} ({rep['seconds']} s)", flush=True)
    return 0


def main(argv=None) -> int:
    return run(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
