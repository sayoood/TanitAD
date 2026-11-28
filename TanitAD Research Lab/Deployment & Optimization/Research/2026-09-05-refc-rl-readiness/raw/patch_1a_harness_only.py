"""Patch 1a: ONLY the harness inert-buffer tolerance (patch 1 part A), applied to a
given refcv3_arm.py — used to re-apply the fix onto the G: worktree's NEWER copy
(md5 44b74a01…, moved by another agent after the clone was taken) instead of
overwriting it with the clone's version. Idempotent: refuses if already applied.
Usage: python patch_1a_harness_only.py <path/to/refcv3_arm.py>"""
import sys

path = sys.argv[1]
s = open(path, encoding="utf-8").read()
if "tolerated_inert_buffers" in s:
    print("already applied:", path)
    raise SystemExit(0)
OLD_A = '''    strict_rep = {"missing_keys": list(res.missing_keys),
                  "unexpected_keys": list(res.unexpected_keys)}
    if (res.missing_keys or res.unexpected_keys) and not allow_nonstrict:
        raise SystemExit(f"[refcv3_arm] ⛔ NON-STRICT LOAD: missing "
                         f"{list(res.missing_keys)[:8]} unexpected "
                         f"{list(res.unexpected_keys)[:8]} — the checkpoint and "
                         f"the rebuilt model disagree. Pass --allow-nonstrict "
                         f"only for a deliberate diagnostic, and say so.")
'''
NEW_A = '''    strict_rep = {"missing_keys": list(res.missing_keys),
                  "unexpected_keys": list(res.unexpected_keys)}
    # ⭐ INERT-BUFFER TOLERANCE (2026-09-05, REF-C RL-readiness WP). refcv4-b
    # added the PERSISTENT buffer `core.decoder.anchor_controls` (refc.py:1168),
    # which `roll_bank` reads ONLY when the decoder is `v0_conditioned`. A
    # checkpoint trained before it existed (refcv3 @ 40,284, the published HF
    # weights) is therefore MISSING a key the rebuilt model never reads, and the
    # strict refusal was a FALSE refusal — MEASURED: STAGE 0 of the RL chain died
    # on it with the md5-verified base. Tolerated only when (a) the decoder is NOT
    # v0-conditioned, (b) nothing else is missing and nothing is unexpected — and
    # RECORDED in the provenance, never silent.
    _dec = getattr(model.core, "decoder", None)
    inert = sorted(k for k in res.missing_keys
                   if k.endswith(".anchor_controls") and _dec is not None
                   and not bool(getattr(_dec, "anchor_v0_cond", False)))
    strict_rep["tolerated_inert_buffers"] = inert
    real_missing = [k for k in res.missing_keys if k not in inert]
    if (real_missing or res.unexpected_keys) and not allow_nonstrict:
        raise SystemExit(f"[refcv3_arm] ⛔ NON-STRICT LOAD: missing "
                         f"{real_missing[:8]} unexpected "
                         f"{list(res.unexpected_keys)[:8]} — the checkpoint and "
                         f"the rebuilt model disagree. Pass --allow-nonstrict "
                         f"only for a deliberate diagnostic, and say so.")
'''
n = s.count(OLD_A)
assert n == 1, f"anchor found {n} times in {path}"
s = s.replace(OLD_A, NEW_A)
open(path, "w", encoding="utf-8", newline="\n").write(s)
print("patched", path)
