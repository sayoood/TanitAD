#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv7 restart options, item 3 -- the ARGV-ONLY levers as ready-to-gate argv files.

Each variant is the CANONICAL launch argv (`stack/ops/runs.d/refcv7-r101-s0.argv.json` at the
launch commit, 154 tokens, gate sha256 6402d33d...) with exactly the named flag values changed IN
PLACE (token order kept -- the gate hashes the ORDERED list), written as the same
`tanitad.launch_argv/1` object with a `changes_vs_launch` block, and its gate sha256 printed.

⛔ An argv edit is a NEW gate run and a new token (`verify_token`), AND it voids both overfit
records, which bind the LAUNCH argv sha256 -- see RESTART_OPTIONS.md. Nothing here is gated.

Usage: make_argv_variants.py --canonical <argv.json> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

#: name -> [(flag, launch value, new value, why)]
VARIANTS = {
    "conflict50": [("--conflict-every", "10", "50",
                    "instrument cadence: 1 grad-conflict reading per 50 steps instead of 10 "
                    "(observational probe; training numerics bit-identical, "
                    "tests/test_refcv6_grad_conflict.py)")],
    "conflict100": [("--conflict-every", "10", "100", "as conflict50, 1 reading per 100 steps")],
    "conflict50_ckptoff": [
        ("--conflict-every", "10", "50", "as conflict50"),
        ("--map-hires-grad-ckpt", "on", "off",
         "no gradient checkpoint in the 10 cm branch: +1.31 GB/sample MEASURED (dev box), "
         "~+21 GB at b16 against the live 24.8 GB peak; equivalent numerics by construction "
         "(the recompute is the same deterministic forward) -- NOT MEASURED bit-identical")],
    "conflict50_eval1000": [
        ("--conflict-every", "10", "50", "as conflict50"),
        ("--eval-every", "500", "1000",
         "in-training T0 monitor every 1,000 steps (the Training Watch cadence halves)")],
}


def gate_sha(argv: list[str]) -> str:
    """launch_gate.argv_sha256: sha256 of the ORDERED list as compact JSON, ensure_ascii False."""
    return hashlib.sha256(json.dumps(argv, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    doc = json.loads(Path(a.canonical).read_text(encoding="utf-8"))
    base = list(doc["argv"])
    base_sha = gate_sha(base)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"canonical": str(a.canonical), "canonical_gate_sha256": base_sha,
               "n_tokens": len(base), "variants": {}}
    for name, edits in VARIANTS.items():
        argv = list(base)
        for flag, old, new, _why in edits:
            idx = [i for i, t in enumerate(argv) if t == flag]
            if len(idx) != 1 or argv[idx[0] + 1] != old:
                raise SystemExit(f"{name}: {flag} is not exactly once with value {old!r}")
            argv[idx[0] + 1] = new
        assert len(argv) == len(base)
        v = copy.deepcopy(doc)
        v["argv"] = argv
        v["arm"] = doc["arm"]
        v["what"] = (f"RESTART-OPTION VARIANT '{name}' of the refcv7 launch argv -- NOT GATED. "
                     + doc.get("what", ""))
        v["changes_vs_launch"] = [{"flag": f, "launch": o, "variant": n, "why": w}
                                  for f, o, n, w in edits]
        v["launch_gate_sha256"] = base_sha
        p = out / f"refcv7-r101-s0.{name}.argv.json"
        p.write_text(json.dumps(v, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["variants"][name] = {"file": p.name, "gate_sha256": gate_sha(argv),
                                     "changes": v["changes_vs_launch"]}
        print(f"{name:<22} {gate_sha(argv)}")
    (out / "argv_variants.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("canonical", base_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
