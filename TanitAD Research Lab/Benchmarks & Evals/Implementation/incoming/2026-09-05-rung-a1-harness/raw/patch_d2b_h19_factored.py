"""RUNG A1 / D2 follow-up — H19-OFF on a FACTORED build.

⛔ A REAL DISCREPANCY IN THE PRE-REGISTRATION, MEASURED 2026-09-05 while
implementing the switch. ``PREREG_REFCV4B_HIERARCHY_EVAL.md`` §3 registers
H19-OFF as ``decoder.maneuver_to_anchor = None``. On every REF-C **v3/v4**
build — which is what ``refcv4b-b1-v72-40k`` is — ``core.factored_maneuver`` is
True (``refc_v3.py:437``, ``:615``), and ``refc.py:1214-1221`` then builds
``lat_to_anchor`` + ``lon_to_anchor`` and leaves ``maneuver_to_anchor`` **None**.

⇒ The literal registered mechanism would have found nothing to remove on the
checkpoint it is registered for. The switch therefore removes whichever anchor-
prior heads the build ACTUALLY carries, names them in the record, and refuses
only when the build carries none.

Idempotent, exact-match, read-back verified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EDITS = [
    (
        "h19 registry text",
        '''    "h19_off": {
        "prereg_arm": "H19-OFF",
        "mechanism": ("`decoder.maneuver_to_anchor = None`, removing the "
                      "kin3-derived 5-way prior from the anchor confidence "
                      "(refc.py:1572 gates on it)"),
        "seen_in_training": "no",
        "tests": "the model's own tactical prediction improves its selection",
        "applies_at": "model",
        "requires": "a decoder built with graft_maneuver=True",
    },''',
        '''    "h19_off": {
        "prereg_arm": "H19-OFF",
        "mechanism": ("the kin3-derived anchor prior removed from the "
                      "confidence: whichever of `decoder.maneuver_to_anchor` / "
                      "`decoder.lat_to_anchor` / `decoder.lon_to_anchor` the "
                      "build carries is set to None, so refc.py:1572-1581's "
                      "`terms` loses them"),
        "seen_in_training": "no",
        "tests": "the model's own tactical prediction improves its selection",
        "applies_at": "model",
        "requires": "a decoder carrying at least one anchor-prior head",
        "⛔ prereg_correction": (
            "PREREG_REFCV4B_HIERARCHY_EVAL.md §3 registers this arm as "
            "`decoder.maneuver_to_anchor = None`. MEASURED 2026-09-05: on "
            "every REF-C v3/v4 build — including refcv4b — "
            "`core.factored_maneuver` is True (refc_v3.py:437, :615), so "
            "refc.py:1214-1221 builds the FACTORED pair "
            "(lat_to_anchor + lon_to_anchor) and leaves `maneuver_to_anchor` "
            "None. The literal registered mechanism would have found nothing "
            "to remove on the very checkpoint it is registered for. This "
            "switch removes the heads that EXIST and names them in the "
            "record; the prereg needs an erratum, not the code a workaround."),
    },''',
    ),
    (
        "h19 implementation",
        '''        elif n == "h19_off":
            if dec is None or getattr(dec, "maneuver_to_anchor", None) is None:
                raise SystemExit(
                    "[refcv3_arm] ablation 'h19_off' needs a decoder built "
                    "with graft_maneuver=True; `maneuver_to_anchor` is None "
                    "here, so the prior it removes is already absent.")
            ev["removed"] = type(dec.maneuver_to_anchor).__name__
            dec.maneuver_to_anchor = None''',
        '''        elif n == "h19_off":
            # ⛔ NOT ONLY `maneuver_to_anchor` — see the registry's
            # `prereg_correction`. A v3/v4 build carries the FACTORED pair and
            # leaves the unfactored head None, so the prereg's literal
            # mechanism would remove nothing on refcv4b itself.
            heads = ("maneuver_to_anchor", "lat_to_anchor", "lon_to_anchor")
            present = [h for h in heads
                       if dec is not None and getattr(dec, h, None) is not None]
            if not present:
                raise SystemExit(
                    "[refcv3_arm] ablation 'h19_off': this decoder carries NO "
                    f"anchor-prior head ({', '.join(heads)} are all None), so "
                    f"the prior it removes is already absent. Refusing rather "
                    f"than reporting an unchanged arm as a knockout.")
            ev["removed"] = {h: type(getattr(dec, h)).__name__ for h in present}
            ev["factored"] = bool(getattr(cfg.core, "factored_maneuver", False))
            for h in present:
                setattr(dec, h, None)
            _p(f"  [ablation] h19_off removed {present} "
               f"(factored_maneuver={ev['factored']})")''',
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
                             f"{src.count(old)} times")
        src = src.replace(old, new, 1)
    path.write_text(src, encoding="utf-8", newline="")
    back = path.read_text(encoding="utf-8")
    missing = [n for n, _o, nw in edits if nw not in back]
    if missing:
        raise SystemExit(f"[{label}] read-back FAILED for {missing}")
    return f"[{label}] applied {[n for n, _, _ in todo]}; read-back OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    print(_apply(Path(a.root) / "taniteval" / "tools" / "refcv3_arm.py",
                 EDITS, "refcv3_arm"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
