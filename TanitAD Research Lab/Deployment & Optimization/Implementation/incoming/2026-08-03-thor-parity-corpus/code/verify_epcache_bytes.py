#!/usr/bin/env python3
"""Destination-side integrity check for a transferred epcache — BY SIZE AND BY
LOADING, never by exit code.

WHY BOTH, AND WHY NEITHER ALONE
-------------------------------
* ``parity.assert_parity_corpus`` proves the SET OF NAMES is the canonical episode
  set. It explicitly does NOT hash tensor bytes (``parity.py``'s own docstring). A
  perfectly parity-passing directory can be full of truncated files.
* A SIZE check catches truncation, which is the failure this programme actually has:
  Thor's ``valdata/.../ep_00028.pt`` sat at 92,299,264 B against a true 117,383,256 B
  — right name, right place, 21.4 % of the episode missing, and every tool that
  looked at the directory listing said it was fine.
* A ``sha256`` check catches silent corruption a size check cannot see.
* A ``torch.load`` catches a file that is byte-perfect but structurally wrong for
  this loader — and it is the only check that exercises the code path training will.

⚠️ EXIT CODES ARE NOT EVIDENCE. Silent truncation with exit 0 has bitten this
programme three times in one day. Every verdict here is a counted, printed fact.

The expected sizes/digests come from ``hf_expected.json``, minted by
``mint_hf_expected.py`` off the HF LFS metadata of the SOURCE repo — so this
compares the destination against the source's own record of itself, not against a
number someone typed.

Usage (on the destination host):
    python verify_epcache_bytes.py --cache ~/epcache/.../physicalai-train-e438721ae894 \
        --expected hf_expected_train.json --mode strict --sha256 all --load 8 \
        --out verify_train.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _import_parity():
    """Find ``tanitad.data.parity`` whether this script sits in the repo (6 levels
    under the root) or has been copied to a bare directory on a pod / on Thor.

    ⚠️ ``parents[6]`` raises ``IndexError`` — not a graceful miss — when the file is
    shallower than that, which is exactly what happens on the destination host this
    script exists to run on. Guarded, not assumed."""
    here = Path(__file__).resolve()
    cands = [Path.home() / "TanitAD" / "stack", Path("/workspace/TanitAD/stack")]
    if len(here.parents) > 6:
        cands.insert(0, here.parents[6] / "stack")
    env = os.environ.get("PYTHONPATH", "")
    cands += [Path(x) for x in env.split(os.pathsep) if x]
    for cand in cands:
        if (cand / "tanitad" / "data" / "parity.py").exists():
            sys.path.insert(0, str(cand))
            break
    else:
        raise SystemExit(
            f"cannot locate tanitad/data/parity.py — tried {[str(c) for c in cands]}. "
            f"Set PYTHONPATH to the stack root.")
    from tanitad.data import parity
    return parity


def sha256_file(p: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--expected", default=None,
                    help="hf_expected_*.json: {name: {size, sha256}}")
    ap.add_argument("--mode", choices=("strict", "subset"), default="strict",
                    help="parity check mode; 'subset' for a deliberate sorted "
                         "PREFIX of the canonical corpus")
    ap.add_argument("--sha256", default="0",
                    help="'all', or an integer sample size (evenly spaced)")
    ap.add_argument("--load", type=int, default=4,
                    help="how many episodes to actually torch.load")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    parity = _import_parity()
    d = Path(a.cache).expanduser()
    rec: dict = {"cache": str(d), "host": os.uname().nodename if hasattr(os, "uname")
                 else "?", "mode": a.mode, "t_start": time.strftime("%FT%TZ",
                                                                    time.gmtime())}

    # -- 1. the NAME-SET check (parity) -------------------------------------- #
    try:
        rec["parity"] = parity.assert_parity_corpus(
            d, label="thor epcache", mode=a.mode)
        rec["parity_verdict"] = "PASS"
    except SystemExit as e:
        rec["parity_verdict"] = "REFUSED"
        rec["parity_refusal"] = str(e)

    eps = sorted(d.glob("ep_*.pt"))
    rec["n_episodes_on_disk"] = len(eps)
    rec["bytes_on_disk"] = sum(p.stat().st_size for p in eps)

    # -- 2. the SIZE check (catches truncation) ------------------------------ #
    exp = json.loads(Path(a.expected).read_text()) if a.expected else {}
    exp_files = exp.get("files", exp)
    size_bad, missing_exp = [], []
    for p in eps:
        e = exp_files.get(p.name)
        if e is None:
            missing_exp.append(p.name)
            continue
        got = p.stat().st_size
        if got != int(e["size"]):
            size_bad.append({"name": p.name, "on_disk": got,
                             "expected": int(e["size"]),
                             "short_by": int(e["size"]) - got})
    rec["size_checked"] = len(eps) - len(missing_exp)
    rec["size_mismatches"] = size_bad
    rec["no_expectation_for"] = missing_exp[:20]
    rec["n_no_expectation"] = len(missing_exp)

    # -- 3. the SHA256 check (catches corruption a size cannot see) ---------- #
    if a.sha256 == "all":
        pick = eps
    elif int(a.sha256) > 0:
        n = min(int(a.sha256), len(eps))
        step = max(len(eps) // n, 1)
        pick = eps[::step][:n]
    else:
        pick = []
    sha_bad = []
    t0 = time.time()
    for p in pick:
        e = exp_files.get(p.name)
        if not e or not e.get("sha256"):
            continue
        got = sha256_file(p)
        if got != e["sha256"]:
            sha_bad.append({"name": p.name, "got": got, "expected": e["sha256"]})
    rec["sha256_checked"] = len(pick)
    rec["sha256_seconds"] = round(time.time() - t0, 1)
    rec["sha256_mismatches"] = sha_bad

    # -- 4. the LOAD check (the only one that runs the training code path) --- #
    load_bad, load_ok = [], []
    if a.load > 0 and eps:
        import torch
        step = max(len(eps) // a.load, 1)
        for p in eps[::step][:a.load]:
            try:
                o = torch.load(p, map_location="cpu", weights_only=False)
                fr = o.get("frames_u8", o.get("frames"))
                load_ok.append({"name": p.name, "keys": sorted(o.keys()),
                                "frames": list(fr.shape), "dtype": str(fr.dtype),
                                "poses": list(o["poses"].shape)})
            except Exception as ex:                              # noqa: BLE001
                load_bad.append({"name": p.name, "error": f"{type(ex).__name__}: {ex}"})
    rec["load_ok"] = load_ok
    rec["load_failures"] = load_bad

    rec["VERDICT"] = ("PASS" if rec.get("parity_verdict") == "PASS"
                      and not size_bad and not sha_bad and not load_bad
                      and not missing_exp else "FAIL")
    print(json.dumps({k: v for k, v in rec.items() if k != "parity_refusal"},
                     indent=2, default=str))
    if rec.get("parity_refusal"):
        print(rec["parity_refusal"], file=sys.stderr)
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=2, default=str),
                               encoding="utf-8")
    return 0 if rec["VERDICT"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
