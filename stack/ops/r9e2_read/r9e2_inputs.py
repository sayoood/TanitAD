#!/usr/bin/env python3
"""r9e2_inputs.py -- writes INPUTS.json for ``stack/scripts/r9_e2_verdict.py`` (the refcv9 E2 judge) from the files the E2 roll wrote.  NEW file (E2-ROLL-2, 2026-11-04).

    python r9e2_inputs.py --rd <read dir> --out <read dir>/INPUTS.json [--k2-label K2 --k0-label K0] [--s6c-k2 <json>] [--s6c-k0 <json>] [--seeds 0,1]

Per inference seed S (the judge REQUIRES exactly seeds 0 AND 1) the five judge inputs are, by file name inside ``--rd``::

    eval_k2  = EVAL_K2.json        eval_k0  = EVAL_K0.json            (r9_e1_eval.py single-arm evals; perception is deterministic => the same file for both seeds)
    families = families_s<S>.json  camblind = camblind_s<S>.json      (r9_e2_families.py --mode families / --mode camblind, ONE call per seed)
    s6c_k2   = <--s6c-k2>          s6c_k0   = <--s6c-k0>              (S6c-DEV planner sidewalk rate)

S6c is NOT BUILT for E2 (no instrument): ``s6c_k2`` is ``null`` unless a real file is passed, and the judge then reads clause (c) NOT-COMPUTABLE (=> NO-GO that names it).
A placeholder number is never written.  ``s6c_k0`` is written only when a real file is passed (the shell passes ``R9E2_S6C_K0`` when set).  A REQUIRED file that is
missing REFUSES (rc 2, nothing written): the judge would read it NOT-COMPUTABLE anyway, but a missing file after a Thor roll is a harness fault to see, not a verdict.
The file records its own provenance: sha256 of every referenced file and the list of keys deliberately left null (``not_built``).  The judge ignores unknown keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "r9e2-inputs-v1"
JUDGE_SEEDS = ("0", "1")
SEED_KEYS = ("eval_k2", "eval_k0", "s6c_k2", "s6c_k0", "families", "camblind")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_inputs(rd, seeds=JUDGE_SEEDS, s6c_k2=None, s6c_k0=None, k2_label="K2", k0_label="K0") -> dict:
    """-> the INPUTS.json dict.  REFUSES (SystemExit) on a seed set other than {0, 1}, a missing required file, or a given S6c path that is not a file."""
    rd = Path(rd)
    if set(map(str, seeds)) != set(JUDGE_SEEDS):
        raise SystemExit(f"[r9e2_inputs] REFUSED: the judge needs inference seeds 0 AND 1, got {list(seeds)}")
    for lab, p in (("--s6c-k2", s6c_k2), ("--s6c-k0", s6c_k0)):
        if p and not Path(p).is_file():
            raise SystemExit(f"[r9e2_inputs] REFUSED: {lab} {p} is not a file (an S6c input is a real file or absent -- never a placeholder)")
    required = {"eval_k2": rd / "EVAL_K2.json", "eval_k0": rd / "EVAL_K0.json"}
    per_seed = {s: {"families": rd / f"families_s{s}.json", "camblind": rd / f"camblind_s{s}.json"} for s in JUDGE_SEEDS}
    missing = [str(p) for p in required.values() if not p.is_file()] + [str(p) for d in per_seed.values() for p in d.values() if not p.is_file()]
    if missing:
        raise SystemExit("[r9e2_inputs] REFUSED: required judge input(s) missing: " + ", ".join(missing))
    seeds_out, shas = {}, {}
    not_built = []
    for s in JUDGE_SEEDS:
        row = {"eval_k2": str(required["eval_k2"]), "eval_k0": str(required["eval_k0"]), "s6c_k2": str(s6c_k2) if s6c_k2 else None,
               "s6c_k0": str(s6c_k0) if s6c_k0 else None, "families": str(per_seed[s]["families"]), "camblind": str(per_seed[s]["camblind"])}
        assert tuple(row) == SEED_KEYS
        seeds_out[s] = row
        not_built += [f"seed{s}.{k}" for k in ("s6c_k2", "s6c_k0") if row[k] is None]
        for k, v in row.items():
            if v:
                shas[v] = _sha256(Path(v))
    return {"schema": SCHEMA, "k2_label": k2_label, "k0_label": k0_label, "seeds": seeds_out, "not_built": not_built,
            "provenance": {"sha256": shas, "note": "s6c_* null = S6c-DEV is not built; the judge reads clause (c) NOT-COMPUTABLE"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rd", required=True, help="the read dir holding EVAL_K2/K0.json, families_s*.json, camblind_s*.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k2-label", default="K2")
    ap.add_argument("--k0-label", default="K0")
    ap.add_argument("--s6c-k2", default="")
    ap.add_argument("--s6c-k0", default="")
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args(argv)
    inputs = build_inputs(a.rd, [s for s in a.seeds.split(",") if s != ""], a.s6c_k2 or None, a.s6c_k0 or None, a.k2_label, a.k0_label)
    Path(a.out).write_text(json.dumps(inputs, indent=1) + "\n", encoding="utf-8")
    print(f"[r9e2_inputs] wrote {a.out}; not built (judge reads NOT-COMPUTABLE): {inputs['not_built']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
