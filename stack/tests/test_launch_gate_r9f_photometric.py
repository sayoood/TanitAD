"""refcv9 C10 -- `--r9-photometric` is CLASSIFIED in the refcv9-full family (admitted: no own record; gated by G-DVB + its attach needle).

  P1  the flag hits NO open item in refcv9-e2 / refcv9-full; a near-miss name (`--r9-photometric-v2`) is still R9F-UNCLASSIFIED-LEVER (exact match)
  P2  RED: the classification removed from the item's `when_except_exact` -> R9F-UNCLASSIFIED-LEVER again
  P3  the attach needle holds on the real trainer; RED: its attach line deleted in a COPY -> lever_attach_reasons names the flag; no flag -> inert
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

E2, FULL = (LG.PROFILES[n] for n in ("refcv9-e2", "refcv9-full"))
K0 = json.loads((ROOT / "ops" / "runs.d" / "r9e2_K0.argv.json").read_text(encoding="utf-8"))["argv"]
TRAINER = (ROOT / "scripts" / "refc_v3_train.py").read_bytes().decode("utf-8").replace("\r\n", "\n")
FLAG = "--r9-photometric"


def _hit(prof, flags, items=None):
    return sorted(i["id"] for i in (items or prof["open_items"]) if LG.open_item_hits(i, list(K0) + flags) not in (None, []))


def test_P1_the_flag_is_classified_and_only_its_exact_name():
    for prof in (E2, FULL):
        assert _hit(prof, []) == [] and _hit(prof, [FLAG]) == []
        assert _hit(prof, [FLAG + "-v2"]) == ["R9F-UNCLASSIFIED-LEVER"]


def test_P2_RED_without_the_classification_the_flag_is_unclassified_again():
    items = [dict(i, when_except_exact=tuple(f for f in (i.get("when_except_exact") or ()) if f != FLAG))
             if i["id"] == "R9F-UNCLASSIFIED-LEVER" else i for i in FULL["open_items"]]
    assert _hit(FULL, [FLAG], items) == ["R9F-UNCLASSIFIED-LEVER"]


def test_P3_the_attach_needle_holds_and_its_mutation_is_RED(tmp_path):
    a = list(K0) + [FLAG]
    assert FULL["lever_attach_needles"][FLAG] == "_r9ph.attach(model, args)"
    assert LG.lever_attach_reasons(FULL, a, TRAINER)[0] == []
    line = next(l for l in TRAINER.split("\n") if l.strip().startswith("_r9ph.attach(model, args)"))
    copy = tmp_path / "refc_v3_train_mut.py"
    copy.write_text(TRAINER.replace(line + "\n", ""), encoding="utf-8")
    r, d = LG.lever_attach_reasons(FULL, a, copy.read_text(encoding="utf-8"))
    assert len(r) == 1 and FLAG in r[0] and d["checked"][FLAG] is False, r
    assert LG.lever_attach_reasons(FULL, K0, copy.read_text(encoding="utf-8"))[0] == []
