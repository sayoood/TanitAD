#!/usr/bin/env python3
"""nt_heldout -- build the H-NT-DECIDE-1 held-out guard file (``tanitad.nt_heldout/1``) and re-export the guard.

``build``: the 38 selection-scan logs (``NAVTRAIN_SAMPLE_TOKENS.json`` ``per_log``, every log the scan READ) + the
2,089 selection-sample tokens (token-level belt) + every navtest log (``navtest.yaml`` ``log_names``). Each source
is recorded by path + sha256, and the registered sample's token sha (9dd68698...) is RE-DERIVED from its token list
and must match its own stamp -- a guard built from a different sample refuses here, not at launch.

The trainer-side reader/refusal is ``stack/tanitad/train/nt_guard.py`` (the NT tree); this module re-exports it so
the label/join builders apply the SAME check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NT_STACK = Path(os.environ.get("NT_TREE") or (HERE.parent / "nt_tree")) / "stack"
if str(NT_STACK) not in sys.path:
    sys.path.insert(0, str(NT_STACK))
from tanitad.train.nt_guard import (SCHEMA, NTGuardError, assert_disjoint, load_guard,  # noqa: E402,F401
                                    load_token_table, refuse_heldout_in_train)

#: PREREG_NT_DECIDE / D-R8-CKPT-RULE: the registered sample's token sha (literal, never re-derived from itself)
SAMPLE_SHA256_SORTED = "9dd68698659ec73c131b677120c45fa6f385ea058d4a055f7378f4c13f46d707"
N_SELECTION_SCAN_LOGS = 38
N_SAMPLE_TOKENS = 2089
NAVTEST_YAML_MD5 = "5f3cfd20fe184fe921e1921083647aae"


def _sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def build(sample_path: str, navtest_yaml: str, out: str) -> dict:
    import yaml
    s = json.load(open(sample_path, encoding="utf-8"))
    toks = sorted(str(t) for t in s["tokens"])
    sha = hashlib.sha256("\n".join(toks).encode("utf-8")).hexdigest()
    sha_alt = hashlib.sha256(json.dumps(toks).encode("utf-8")).hexdigest()
    if s.get("sha256_sorted_tokens") != SAMPLE_SHA256_SORTED:
        raise NTGuardError(f"[nt-heldout] ⛔ sample stamp {s.get('sha256_sorted_tokens')} != registered "
                           f"{SAMPLE_SHA256_SORTED}")
    rederived = SAMPLE_SHA256_SORTED in (sha, sha_alt)
    if len(toks) != N_SAMPLE_TOKENS or int(s["count"]) != N_SAMPLE_TOKENS:
        raise NTGuardError(f"[nt-heldout] ⛔ sample holds {len(toks)} tokens (count {s['count']}), registered "
                           f"{N_SAMPLE_TOKENS}")
    sel = [str(r["log"]) for r in sorted(s["per_log"], key=lambda r: int(r["rank"]))]
    if len(sel) != N_SELECTION_SCAN_LOGS or len(set(sel)) != len(sel):
        raise NTGuardError(f"[nt-heldout] ⛔ {len(sel)} selection-scan logs, registered {N_SELECTION_SCAN_LOGS}")
    tok_logs = {str(v) for v in s["token_log"].values()}
    if not tok_logs <= set(sel):
        raise NTGuardError("[nt-heldout] ⛔ a sample token sits on a log outside per_log")
    ym = _md5(navtest_yaml)
    if ym != NAVTEST_YAML_MD5:
        raise NTGuardError(f"[nt-heldout] ⛔ navtest.yaml md5 {ym} != registered {NAVTEST_YAML_MD5}")
    y = yaml.safe_load(open(navtest_yaml, encoding="utf-8"))
    nt = sorted({str(x) for x in (y.get("log_names") or [])})
    if not nt:
        raise NTGuardError("[nt-heldout] ⛔ navtest.yaml carries no log_names")
    held = sorted(set(sel) | set(nt))
    d = {"schema": SCHEMA,
         "what": "H-NT-DECIDE-1 held-out logs: NEVER train on these (PREREG_NT_DECIDE sec. 2, control K3)",
         "selection_scan_logs": sel, "navtest_logs": nt, "heldout_logs": held,
         "selection_sample_tokens": toks,
         "counts": {"selection_scan_logs": len(sel), "navtest_logs": len(nt), "heldout_logs": len(held),
                    "selection_sample_tokens": len(toks), "overlap_selection_navtest": len(set(sel) & set(nt))},
         "sources": {"selection_sample": {"path": str(sample_path), "sha256": _sha256(sample_path),
                                          "sha256_sorted_tokens_stamp": s.get("sha256_sorted_tokens"),
                                          "stamp_rederived_from_token_list": bool(rederived)},
                     "navtest_yaml": {"path": str(navtest_yaml), "md5": ym, "sha256": _sha256(navtest_yaml),
                                      "n_tokens": len(y.get("tokens") or [])}},
         "tool": "nt_heldout.py build"}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    g = load_guard(out)                      # the trainer's own reader must accept what we wrote
    return {"path": out, "sha256": g["sha256"], **d["counts"], "stamp_rederived": bool(rederived)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sample", required=True)
    b.add_argument("--navtest-yaml", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build(a.sample, a.navtest_yaml, a.out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
