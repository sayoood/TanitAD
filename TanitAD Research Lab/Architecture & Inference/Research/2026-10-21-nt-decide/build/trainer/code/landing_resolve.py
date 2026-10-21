#!/usr/bin/env python3
"""Resolve the 3 conflict regions ``landing_check.py`` found merging the NT change set onto
origin/agent/arch-inf-20260803 @ d0f87190b (2026-10-19 21:39), and write the resolved files beside the clean ones
(``<landing>/landed/<rel>``, each in its committed EOL convention) with their expected git blob ids.

Every region is "both sides ADDED a block at the same anchor" -- no line was edited by both:
  * launch_gate.py region 1: HEAD adds the v8F constants (V8F_P10_* ... _V8F_STEER_OPEN_ITEMS), nt_tree adds the
    refcv8-nt constants (_NT_ROOT ... _NT_OPEN_ITEMS); both blocks close on the SAME shared ``)`` line that follows.
    Union = HEAD's block, its own ``)``, the section separator, then nt_tree's block (which keeps the shared ``)``).
  * launch_gate.py region 2: HEAD adds ``_v8f_profile`` + PROFILES["v8f"/"v8f_steer"], nt_tree adds ``_nt_b1`` +
    PROFILES["refcv8-nt"]. Union = HEAD's block, two blank lines, nt_tree's block.
  * test_declared_vs_built.py: HEAD 276 -> 279 (+p10_prior, +p10_prior_sha256, +plan_vmax_clamp), nt_tree 276 -> 282
    (+6 NT flags). Union = HEAD's comment lines, the NT comment re-based (279 -> 285), ``== 285``, the NT kind asserts.
The resolver REFUSES unless each region has exactly the expected first/last lines (the shape it was written for)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

NUL = bytes([0])


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + NUL + data).hexdigest()


def regions(lines):
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<< "):
            j = next(k for k in range(i, len(lines)) if lines[k] == "=======\n")
            e = next(k for k in range(j, len(lines)) if lines[k].startswith(">>>>>>> "))
            out.append((i, j, e))
            i = e + 1
        else:
            i += 1
    return out


def splice(lines, resolved_per_region):
    rs = regions(lines)
    assert len(rs) == len(resolved_per_region), (len(rs), len(resolved_per_region))
    out, last = [], 0
    for (i, j, e), res in zip(rs, resolved_per_region):
        out += lines[last:i] + res
        last = e + 1
    return out + lines[last:]


def must(cond, msg):
    if not cond:
        raise SystemExit("REFUSED: " + msg)


def resolve_gate(lines):
    rs = regions(lines)
    must(len(rs) == 2, f"launch_gate.py: expected 2 regions, got {len(rs)}")
    (i1, j1, e1), (i2, j2, e2) = rs
    h1, n1 = lines[i1 + 1:j1], lines[j1 + 1:e1]
    must(h1[0].startswith("# v8F rung alpha") and h1[-1].rstrip().endswith('plumb actions"},')
         and any(l.startswith("_V8F_STEER_OPEN_ITEMS = (") for l in h1), "region 1 HEAD shape")
    must(n1[0].startswith("# refcv8-nt: H-NT-DECIDE-1") and n1[-1].lstrip().startswith('"owner"'), "region 1 nt shape")
    must(lines[e1 + 1] == ")\n" and lines[e1 + 2].startswith('_R8 = PROFILES["refcv8"]'), "region 1 shared tail")
    sep = lines[i1 - 1]
    must(sep.startswith("# ------"), "region 1 separator")
    r1 = h1 + [")\n", "\n", "\n", sep] + n1
    h2, n2 = lines[i2 + 1:j2], lines[j2 + 1:e2]
    must(h2[0].startswith("def _v8f_profile(") and h2[-1].startswith('PROFILES["v8f_steer"]'), "region 2 HEAD shape")
    must(n2[0].startswith("def _nt_b1(") and n2[-1] == ")\n", "region 2 nt shape")
    r2 = h2 + ["\n", "\n"] + n2
    return splice(lines, [r1, r2])


def resolve_dvb_test(lines):
    rs = regions(lines)
    must(len(rs) == 1, f"test_declared_vs_built.py: expected 1 region, got {len(rs)}")
    i, j, e = rs[0]
    h, n = lines[i + 1:j], lines[j + 1:e]
    must(h[-1].strip() == "assert len(dvb.REGISTRY) == 279", "test HEAD count literal")
    must(n[2].strip() == "assert len(dvb.REGISTRY) == 282", "test nt count literal")
    must("(276 -> 282)" in n[1], "test nt comment")
    nt_comment = [n[0], n[1].replace("(276 -> 282)", "(279 -> 285)")]
    res = h[:-1] + nt_comment + ["    assert len(dvb.REGISTRY) == 285\n"] + n[3:]
    return splice(lines, [res])


RESOLVERS = {"stack/scripts/launch_gate.py": resolve_gate,
             "stack/tests/test_declared_vs_built.py": resolve_dvb_test}


def main() -> int:
    land = Path(sys.argv[1])
    rep = json.loads((land / "LANDING_CHECK.json").read_text(encoding="utf-8"))
    out = {}
    for rel, fn in RESOLVERS.items():
        row = rep["files"][rel]
        must(row.get("merge_conflicts", 0) > 0, f"{rel} is not conflicted in this landing check")
        merged = (land / "merge" / rel.replace("/", "__") / "merged").read_bytes().decode("utf-8")
        lines = merged.splitlines(keepends=True)
        res = "".join(fn(lines))
        must("<<<<<<<" not in res and ">>>>>>>" not in res and "\n=======\n" not in res, f"{rel}: markers left")
        b = res.encode("utf-8")
        if row.get("head_blob_crlf"):
            b = b.replace(b"\n", b"\r\n")
        dst = land / "landed" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b)
        out[rel] = {"landed_blob_expected": git_blob(b), "crlf": bool(row.get("head_blob_crlf")),
                    "n_lines": res.count("\n")}
        row["landed_blob_expected"] = out[rel]["landed_blob_expected"]
        row["landed_blob_lf"] = git_blob(b.replace(bytes([13, 10]), bytes([10])))
        row["resolved_by"] = "landing_resolve.py (union of two added blocks; see its docstring)"
    rep["n_conflicted_after_resolve"] = 0
    (land / "LANDING_CHECK.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    (land / "RESOLUTION.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
