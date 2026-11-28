"""K-CLIP (PREREG_V8F_N1DW section 6, R4): the share of LOGGED optimiser steps on which gradient clipping was active.

The trainer clips at ``--clip-grad`` (default 1.0, v8f_mini_train.py ``ap.add_argument("--clip-grad", ... default=1.0)``) with ``torch.nn.utils.clip_grad_norm_``, which returns the PRE-clip total norm; the loop logs it as
``grad_norm`` (``rec = {..., "grad_norm": float(gn), ...}`` at every ``--log-every`` step). A step is clipped iff its logged grad_norm > clip_grad. The share is over the LOGGED steps only (the count is printed).

usage: python kclip.py <train_log.jsonl> [--clip-grad X] [--json]
``clip_grad`` is read from ``summary.json`` beside the log (run-control value) when it is there, else from ``--clip-grad``, else 1.0 (the parser default; the source of the value is printed)."""
import argparse
import json
import math
import os
import sys


def kclip(log_path, clip_grad=None):
    rows = []
    with open(log_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    src = "argument"
    if clip_grad is None:
        sm = os.path.join(os.path.dirname(os.path.abspath(log_path)), "summary.json")
        if os.path.isfile(sm):
            try:
                v = json.load(open(sm, encoding="utf-8")).get("clip_grad")
                if v is not None:
                    clip_grad, src = float(v), "summary.json"
            except (OSError, ValueError):
                pass
    if clip_grad is None:
        clip_grad, src = 1.0, "parser default (--clip-grad 1.0)"
    g = [float(r["grad_norm"]) for r in rows if "grad_norm" in r and math.isfinite(float(r["grad_norm"]))]
    n_bad = sum(1 for r in rows if "grad_norm" in r) - len(g)
    n = len(g)
    c = sum(1 for x in g if x > clip_grad)
    return {"log": log_path, "clip_grad": clip_grad, "clip_grad_source": src, "logged_steps": n, "clipped_steps": c, "share_clipped": (c / n) if n else None,
            "non_finite_grad_norm_rows": n_bad, "grad_norm_median": sorted(g)[n // 2] if n else None, "grad_norm_max": max(g) if n else None, "first_step": rows[0].get("step") if rows else None,
            "last_step": rows[-1].get("step") if rows else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--clip-grad", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    r = kclip(a.log, a.clip_grad)
    if a.json:
        print(json.dumps(r))
    else:
        sh = "n/a" if r["share_clipped"] is None else f"{r['share_clipped']:.4f}"
        print(f"logged steps {r['logged_steps']}  clip_grad {r['clip_grad']} ({r['clip_grad_source']})  clipped {r['clipped_steps']}  share {sh}  grad_norm median {r['grad_norm_median']} max {r['grad_norm_max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
