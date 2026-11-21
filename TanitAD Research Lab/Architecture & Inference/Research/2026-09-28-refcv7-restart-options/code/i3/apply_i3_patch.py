#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv7 restart item 5 (I3, lowest priority) -- `refcv6_loader.build_model` rebuilds an OLD
(pre-A9, 100-query) record at the STAMPED query count instead of the post-A9 default 300.

THE EXPOSURE (MEASURED by the I3 agent, `…/2026-09-27-loader-stamped-queries/RESULT.md` sec. 4, and
re-measured here): a pre-A9 record rebuilt through the vendored loader on the launch tree dies on its
strict load, on BOTH heads -- `core.agent_head.queries` [1,100,256] vs [1,300,256] and
`_perception.box_dec.queries` [1,100,256] vs [1,300,256]. `taniteval/tools/refcv3_arm.load_model`
already applies the stamped rule (the reference); this ports its two lines:
  * the AGENT head: `refc_v3_train.agent_queries_as_trained(config, args)` after the re-parse (an
    explicit argv `--agent-queries` wins, then the stamp `seams.agents.queries`); a tree whose
    trainer predates the helper keeps its own parser default (it IS the pre-A9 default);
  * the BOX head: `n_queries` from the record's `refcv6_perception.n_queries` stamp.

⛔ CLOSURE-TOUCHING (box binding): `stack/tanitad/eval/refcv6_loader.py` (14450c78) and
`stack/scripts/g_box_overfit.py` (ad88b61e) are in `gbo_closure.json`; the vendored loader is pinned by
`g_box_overfit.AUDIT_LOADER_BLOB` (the audit body, 11808258) and a literal in
`tests/test_g_box_overfit.py`. All three are re-pinned together; the A11 contract text in the
provenance block says the body is now "the audit file + the I3 patch". PACKAGE-ONLY.
(The EvalFlyWheel battery copy and the audit copy are banked evidence and are NOT edited: the battery
runs on its own pre-A9 `ev6` tree, where the default is 100 -- not exposed, MEASURED by the I3 agent.)

Usage: python apply_i3_patch.py <tree-root> [--check] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

L_REL = "stack/tanitad/eval/refcv6_loader.py"
G_REL = "stack/scripts/g_box_overfit.py"
T_REL = "stack/tests/test_g_box_overfit.py"
BASES = {L_REL: "14450c78e880401339a09376a2d1da2337b6d484",
         G_REL: "ad88b61ed3d0580da0a971285b36c72665c8a318",
         T_REL: "cb6812e16fe02f2bfddd3356d3bc808d5ecb8016"}
OLD_AUDIT = "11808258cd5c90647fae03f5859774b5541e9300"
VENDOR_END = "== END VENDOR PROVENANCE ==\n"
RESULTS: dict[str, str] | None = {    # pinned 2026-09-28 (raw/i3: RED 3 of 4 on the old loader, GREEN 6/6 patched)
    L_REL: "e66b4bc046dafd3477145cca45d60a025629af04",
    G_REL: "99108358c91ff30db9b9897cb50a79bdad76c3fc",
    T_REL: "7cfba30d9887f3141fb36ddd9788bef2b2f787b9"}

L_EDITS = [
    ("  contract    everything after the END line below is the audit file VERBATIM (this block is the only\n"
     "              addition). tests/test_g_box_overfit.py strips the block and re-hashes the rest to the audit blob;\n"
     "              the harness records the same digest in every record (`loader.audit_blob`).\n",
     "  contract    everything after the END line below is the audit file (blob 11808258) PLUS the I3 stamped-\n"
     "              queries patch (refcv7 restart package, 2026-09-28: `_build_model` rebuilds an old record's\n"
     "              agent and box heads at the query counts it STAMPED, as refcv3_arm.load_model does).\n"
     "              tests/test_g_box_overfit.py strips this block and re-hashes the rest to\n"
     "              g_box_overfit.AUDIT_LOADER_BLOB; the harness records the same digest in every record.\n"),
    ('    rec: dict = {"argv_local": argv, "argv_remap": arec, "departures": []}\n',
     '    rec: dict = {"argv_local": argv, "argv_remap": arec, "departures": []}\n'
     '    # ⛔ I3 (refcv7 restart package 2026-09-28): the AGENT head at the query count AS TRAINED.\n'
     '    # refcv7 A9 R4 moved the parser default 100 -> 300; a record whose argv never passed\n'
     '    # --agent-queries (refcv6-r101-s0) would be rebuilt at 300 and fail its strict load. The\n'
     '    # trainer\'s own helper (explicit argv wins, then the seams.agents.queries stamp); a tree whose\n'
     '    # trainer predates it keeps its parser default, which IS the pre-A9 100.\n'
     '    _aq = getattr(tr, "agent_queries_as_trained", None)\n'
     '    if _aq is not None and str(getattr(args, "agents", "off")) != "off":\n'
     '        _n_q, _why_q = _aq(config, args)\n'
     '        rec["agent_queries"] = {"parser": int(getattr(args, "agent_queries")),\n'
     '                                "as_trained": None if _n_q is None else int(_n_q), "why": _why_q}\n'
     '        if _n_q is not None and int(_n_q) != int(args.agent_queries):\n'
     '            args.agent_queries = int(_n_q)\n'
     '            rec["departures"].append(f"--agent-queries: parser default "\n'
     '                                     f"{rec[\'agent_queries\'][\'parser\']} -> the record\'s {int(_n_q)}"\n'
     '                                     f" ({_why_q})")\n'),
    ('        _pcfg = _perc.PerceptionBranchConfig(w_map=model._w_map, w_box3d=model._w_box3d)\n',
     '        # ⛔ I3: the BOX head at the query count the record STAMPED (refcv3_arm\'s rule); a record\n'
     '        # without the stamp keeps the dataclass default (recorded).\n'
     '        _nq = (config.get("refcv6_perception") or {}).get("n_queries")\n'
     '        _pkw = {} if _nq is None else {"n_queries": int(_nq)}\n'
     '        rec["box_queries"] = {"stamp": _nq, "source": "refcv6_perception.n_queries"\n'
     '                              if _nq is not None else "dataclass default (no stamp)"}\n'
     '        _pcfg = _perc.PerceptionBranchConfig(w_map=model._w_map, w_box3d=model._w_box3d, **_pkw)\n'),
]


def git_blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(b) + b).hexdigest()


def audit_body_blob(src: str) -> str:
    """g_box_overfit.vendored_audit_blob, re-implemented: '\"\"\"' + everything after the END line."""
    return git_blob(('"""' + src[src.index(VENDOR_END) + len(VENDOR_END):]).encode("utf-8"))


def _sub(txt: str, old: str, new: str, rel: str) -> str:
    n = txt.count(old)
    if n != 1:
        raise SystemExit(f"[i3] {rel}: anchor found {n} times, expected 1: {old[:90]!r}")
    return txt.replace(old, new)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.tree)
    raw = {r: (root / r).read_bytes() for r in BASES}
    for r, b in BASES.items():
        if git_blob(raw[r]) != b:
            raise SystemExit(f"[i3] {r}: base blob {git_blob(raw[r])} != launch blob {b} -- REFUSED")
    lsrc = raw[L_REL].decode("utf-8")
    if audit_body_blob(lsrc) != OLD_AUDIT:
        raise SystemExit("[i3] the vendored loader's body is not the audit blob -- REFUSED")
    for old, new in L_EDITS:
        lsrc = _sub(lsrc, old, new, L_REL)
    new_audit = audit_body_blob(lsrc)
    gsrc = _sub(raw[G_REL].decode("utf-8"),
                f'AUDIT_LOADER_BLOB = "{OLD_AUDIT}"   # 2026-09-26-box-head-audit/code/refcv6_loader.py\n',
                f'AUDIT_LOADER_BLOB = "{new_audit}"   # the audit body (11808258) + the I3 stamped-queries '
                f'patch (refcv7 restart package 2026-09-28)\n', G_REL)
    tsrc = _sub(raw[T_REL].decode("utf-8"),
                f'assert G.vendored_audit_blob(src) == G.AUDIT_LOADER_BLOB == "{OLD_AUDIT}"\n',
                f'assert G.vendored_audit_blob(src) == G.AUDIT_LOADER_BLOB == "{new_audit}"\n', T_REL)
    out = {L_REL: lsrc.encode("utf-8"), G_REL: gsrc.encode("utf-8"), T_REL: tsrc.encode("utf-8")}
    blobs = {r: git_blob(b) for r, b in out.items()}
    if RESULTS is not None and blobs != RESULTS:
        raise SystemExit(f"[i3] result blobs {blobs} != the verified {RESULTS} -- REFUSED")
    print(f"[i3] audit body blob {OLD_AUDIT[:12]} -> {new_audit}")
    for r in BASES:
        print(f"[i3] {r}: {BASES[r][:12]} -> {blobs[r]}")
    if not a.check:
        for r, b in out.items():
            dst = (Path(a.out_dir) / r) if a.out_dir else (root / r)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
