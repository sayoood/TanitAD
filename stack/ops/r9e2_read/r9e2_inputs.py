#!/usr/bin/env python3
"""r9e2_inputs.py -- writes INPUTS.json for ``stack/scripts/r9_e2_verdict.py`` (the refcv9 E2 judge) from the files the E2 roll wrote.  NEW file (E2-ROLL-2, 2026-11-04).

    python r9e2_inputs.py --rd <read dir> --out <read dir>/INPUTS.json [--k2-label K2 --k0-label K0] [--s6c-dir <dir>] [--seeds 0,1]

Per inference seed S (the judge REQUIRES exactly seeds 0 AND 1) the five judge inputs are, by file name inside ``--rd``::

    eval_k2  = EVAL_K2.json        eval_k0  = EVAL_K0.json            (r9_e1_eval.py single-arm evals; perception is deterministic => the same file for both seeds)
    families = families_s<S>.json  camblind = camblind_s<S>.json      (r9_e2_families.py --mode families / --mode camblind, ONE call per seed)
    s6c_k2   = S6C_<k2 label>_s<S>.json   s6c_k0 = S6C_<k0 label>_s<S>.json   inside ``--s6c-dir`` (s6c_dev.py's flat JSON, ONE FILE PER ARM PER SEED)

S6c (the planner sidewalk rate, ``s6c_dev.py`` on that arm's MAPSEL roll at that inference seed) is PER SEED AND PER ARM: seeds 0 and 1 are two separate rolls (H-ESTIM-SEED-1), so
one file is NEVER stamped on both seeds.  A per-seed file that is absent leaves ``null`` (the judge reads clause (c) NOT-COMPUTABLE); a placeholder is never written.  A file that
exists must record ``infer_seed`` == the seed slot it fills and the arm's own ``E2_<label>_mapsel`` trace path, else the writer REFUSES (a seed swap or a K0/K2 swap would read as
evidence).  The old single-file ``--s6c-k2`` / ``--s6c-k0`` flags are REFUSED.  A REQUIRED file that is missing REFUSES (rc 2, nothing written): the judge would read it
NOT-COMPUTABLE anyway, but a missing file after a Thor roll is a harness fault to see, not a verdict.
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


def s6c_name(label, seed) -> str:
    return f"S6C_{label}_s{seed}.json"


def _s6c_file(s6c_dir, label, seed):
    """-> the per-seed, per-arm S6c file or None (absent).  REFUSES a file whose own record names another seed or another arm's trace."""
    if s6c_dir is None:
        return None
    p = Path(s6c_dir) / s6c_name(label, seed)
    if not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SystemExit(f"[r9e2_inputs] REFUSED: {p} is not JSON ({e})") from None
    if not isinstance(rec, dict) or str(rec.get("infer_seed")) != str(seed):
        raise SystemExit(f"[r9e2_inputs] REFUSED: {p} records infer_seed={rec.get('infer_seed') if isinstance(rec, dict) else None!r}, it fills the seed {seed} slot (seeds are separate runs)")
    if f"E2_{label}_mapsel" not in str(rec.get("trace", "")).replace("\\", "/"):
        raise SystemExit(f"[r9e2_inputs] REFUSED: {p} was scored from trace {rec.get('trace')!r}, not from E2_{label}_mapsel (an arm swap would read as evidence)")
    return p


def build_inputs(rd, seeds=JUDGE_SEEDS, s6c_dir=None, k2_label="K2", k0_label="K0") -> dict:
    """-> the INPUTS.json dict.  REFUSES (SystemExit) on a seed set other than {0, 1}, a missing required file, an ``s6c_dir`` that is not a directory, or an S6c file of the wrong seed / arm."""
    rd = Path(rd)
    if set(map(str, seeds)) != set(JUDGE_SEEDS):
        raise SystemExit(f"[r9e2_inputs] REFUSED: the judge needs inference seeds 0 AND 1, got {list(seeds)}")
    if s6c_dir is not None and not Path(s6c_dir).is_dir():
        raise SystemExit(f"[r9e2_inputs] REFUSED: --s6c-dir {s6c_dir} is not a directory (an S6c input is a real per-seed file or absent -- never a placeholder)")
    required = {"eval_k2": rd / "EVAL_K2.json", "eval_k0": rd / "EVAL_K0.json"}
    per_seed = {s: {"families": rd / f"families_s{s}.json", "camblind": rd / f"camblind_s{s}.json"} for s in JUDGE_SEEDS}
    missing = [str(p) for p in required.values() if not p.is_file()] + [str(p) for d in per_seed.values() for p in d.values() if not p.is_file()]
    if missing:
        raise SystemExit("[r9e2_inputs] REFUSED: required judge input(s) missing: " + ", ".join(missing))
    seeds_out, shas = {}, {}
    not_built = []
    for s in JUDGE_SEEDS:
        row = {"eval_k2": str(required["eval_k2"]), "eval_k0": str(required["eval_k0"]), "s6c_k2": _str(_s6c_file(s6c_dir, k2_label, s)),
               "s6c_k0": _str(_s6c_file(s6c_dir, k0_label, s)), "families": str(per_seed[s]["families"]), "camblind": str(per_seed[s]["camblind"])}
        assert tuple(row) == SEED_KEYS
        seeds_out[s] = row
        not_built += [f"seed{s}.{k}" for k in ("s6c_k2", "s6c_k0") if row[k] is None]
        for k, v in row.items():
            if v:
                shas[v] = _sha256(Path(v))
    return {"schema": SCHEMA, "k2_label": k2_label, "k0_label": k0_label, "seeds": seeds_out, "not_built": not_built,
            "provenance": {"sha256": shas, "note": "s6c_* null = that arm/seed has no S6c-DEV file; the judge reads clause (c) NOT-COMPUTABLE"}}


def _str(p):
    return None if p is None else str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rd", required=True, help="the read dir holding EVAL_K2/K0.json, families_s*.json, camblind_s*.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k2-label", default="K2")
    ap.add_argument("--k0-label", default="K0")
    ap.add_argument("--s6c-dir", default="", help="holds S6C_<label>_s<seed>.json, one per arm per seed (s6c_dev.py --flat-out)")
    ap.add_argument("--s6c-k2", default="", help="REFUSED: one file would be stamped on both seeds")
    ap.add_argument("--s6c-k0", default="", help="REFUSED: one file would be stamped on both seeds")
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args(argv)
    if a.s6c_k2 or a.s6c_k0:
        raise SystemExit("[r9e2_inputs] REFUSED: --s6c-k2 / --s6c-k0 name ONE file for both inference seeds; seeds are separate runs -- use --s6c-dir with one file per arm per seed")
    inputs = build_inputs(a.rd, [s for s in a.seeds.split(",") if s != ""], a.s6c_dir or None, a.k2_label, a.k0_label)
    Path(a.out).write_text(json.dumps(inputs, indent=1) + "\n", encoding="utf-8")
    print(f"[r9e2_inputs] wrote {a.out}; no S6c file (judge reads NOT-COMPUTABLE): {inputs['not_built']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
