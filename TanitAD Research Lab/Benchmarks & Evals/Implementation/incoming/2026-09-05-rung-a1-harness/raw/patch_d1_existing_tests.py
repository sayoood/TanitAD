"""RUNG A1 / DELIVERABLE 1 — update the EXISTING refcv3_arm pins for `ha0_ext`.

Two assertions in ``stack/tests/test_refcv3_arm.py`` pin the pre-Rung-A1 arm
space exactly, which is what a good pin does — they FAILED, correctly, the
moment the arm was added. This updates them to the new contract and makes each
one assert the NEW fact rather than merely tolerating it.

Idempotent, exact-match, read-back verified. Same reasons as
``patch_d1_ha0_ext.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EDITS: list[tuple[str, str, str]] = [
    (
        "arm space",
        '''    for arm in ("os", "ha", "ha0", "os_navshuf", "os_navzero", "oracle_sel"):
        assert d[arm].shape == (N, 4, 2), arm
    # ⛔ the dump's key space IS the arm space: metadata must not be an arm
    assert d["v0"].shape == (N,) and d["eid"].tolist() == [0]
    assert set(rec["arms"]) == {"os", "ha", "ha0", "os_navshuf", "os_navzero",
                                "oracle_sel"}
    assert rec["tiers"] == {"os": "T1", "ha": "T1", "ha0": "T1",
                            "os_navshuf": "T1", "os_navzero": "T1",
                            "oracle_sel": "T0"}''',
        '''    # ⭐ `ha0_ext` JOINED THIS LIST AT RUNG A1 (2026-09-05) and it is NOT
    # optional: refcv5's acceptance bar is "beat BOTH `ha` and `ha0_ext`", and
    # before the port this harness computed neither, so half the bar was
    # unreadable on the whole REF-C surface.
    for arm in ("os", "ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero",
                "oracle_sel"):
        assert d[arm].shape == (N, 4, 2), arm
    # ⛔ the dump's key space IS the arm space: metadata must not be an arm
    assert d["v0"].shape == (N,) and d["eid"].tolist() == [0]
    assert set(rec["arms"]) == {"os", "ha", "ha0", "ha0_ext", "os_navshuf",
                                "os_navzero", "oracle_sel"}
    assert rec["tiers"] == {"os": "T1", "ha": "T1", "ha0": "T1",
                            "ha0_ext": "T1", "os_navshuf": "T1",
                            "os_navzero": "T1", "oracle_sel": "T0"}''',
    ),
    (
        "action units",
        '''    assert au["applies_to"] == ["ha"]
    assert any("ha0" in s for s in au["does_not_apply_to"])''',
        '''    # ⭐ `ha0_ext` holds a RECORDED channel-1 value, so the steer->kappa
    # conversion applies to it exactly as it does to `ha`. `ha0` is exactly
    # zero and is therefore unit-free — that is what makes it, and only it,
    # bit-comparable across refav1 and refcv3.
    assert au["applies_to"] == ["ha", "ha0_ext"]
    assert any("ha0" in s for s in au["does_not_apply_to"])''',
    ),
    (
        "paired blocks",
        '''    assert "paired_os_navzero_minus_ha0" in fp
    assert "paired_os_minus_navzero" in fp''',
        '''    assert "paired_os_navzero_minus_ha0" in fp
    assert "paired_os_minus_navzero" in fp
    # ⭐⭐ RUNG A1: THE OTHER HALF OF refcv5's ACCEPTANCE BAR. `ha0` alone is
    # not the bar — stack/tanitad/eval/echo_gate.py retracts that reading by
    # name. An arm that beats `ha` while TYING `ha0_ext` has echoed its own
    # ego state, not read the scene, and only this block can see it.
    assert "paired_os_minus_ha0ext" in fp, (
        "the os - ha0_ext margin is half of refcv5's acceptance bar")
    assert "paired_os_navzero_minus_ha0ext" in fp, (
        "the DEPLOYMENT-relevant form of the same read (no oracle nav)")
    assert fp["paired_os_minus_ha0ext"]["direction"] == "os - ha0_ext"''',
    ),
]


def _apply(path: Path, edits, label: str) -> str:
    raw = path.read_bytes()
    if b"\r" in raw:
        raise SystemExit(f"[{label}] REFUSING: {path} contains CR")
    src = path.read_text(encoding="utf-8")
    done, todo = [], []
    for name, old, new in edits:
        if new in src:
            done.append(name)
        elif old in src:
            todo.append((name, old, new))
        else:
            raise SystemExit(f"[{label}] REFUSING: anchor {name!r} not found "
                             f"and its replacement is not present either")
    if not todo:
        return f"[{label}] already applied ({len(done)}/{len(edits)})"
    for name, old, new in todo:
        if src.count(old) != 1:
            raise SystemExit(f"[{label}] REFUSING: anchor {name!r} occurs "
                             f"{src.count(old)} times, need exactly 1")
        src = src.replace(old, new, 1)
    path.write_text(src, encoding="utf-8", newline="")
    back = path.read_text(encoding="utf-8")
    missing = [n for n, _o, nw in edits if nw not in back]
    if missing:
        raise SystemExit(f"[{label}] read-back FAILED for {missing}")
    return (f"[{label}] applied {[n for n, _, _ in todo]}, "
            f"{len(done)} already present; read-back OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    print(_apply(Path(a.root) / "stack" / "tests" / "test_refcv3_arm.py",
                 EDITS, "test_refcv3_arm"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
