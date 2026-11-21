"""K2DC's argv is COMPOSED by r9_e2_make_argv.py, never hand-written (MM 4a3d05, 2026-11-13; PREREG_K2DC_2026-11-12.md, PI: "Approve K2DC").

K2DC = K2M1N20's composition (map arm M1N20, T1v, initial --w-map-hires 25, the same W-COST critic weight) + the four map-dose controller flags,
appended after every other token. These tests regenerate the committed file byte-for-byte and pin the delta against K2M1N20 to exactly those flags.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r9_e2_make_argv as MK  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
DOSE = ["--map-hires-share-target", "0.10", "--map-hires-share-every", "250", "--map-hires-share-group", "trunk", "--map-hires-w-bounds", "1,25"]


def _doc(v):
    return json.loads((RUNS / f"r9e2_{v}.argv.json").read_text(encoding="utf-8"))


def test_K2DC_regenerates_byte_equal_from_the_composer():
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    d = _doc("K2DC")
    r = float(d["w_cost_ratio"])
    w = MK.w_from_ratio(r)
    rec = MK.variant_doc(json.loads(k0b.decode("utf-8")), k0b, "K2DC", w, provisional=False,
                         w_source="0.1 / %g (--w-cost-ratio: the K2 explore smoke's W-COST line)" % r)
    got = MK.dump(MK.stamp_w_cost(rec, r, d["w_cost_source"])).encode("utf-8")
    assert got == (RUNS / "r9e2_K2DC.argv.json").read_bytes()
    assert d["arm"] == "r9e2_K2DC" and d["profile"] == "refcv9-e2" and d["steps"] == 6000 and d["derived_from"] == "r9e2_K0"
    assert d["base_sha256"] == hashlib.sha256(k0b).hexdigest()


def test_K2DC_is_K2M1N20_plus_exactly_the_four_controller_flags():
    a, m = _doc("K2DC"), _doc("K2M1N20")
    assert a["argv"][-len(DOSE):] == DOSE
    assert a["argv"][:-len(DOSE)] == [x.replace("r9e2-K2M1N20", "r9e2-K2DC") for x in m["argv"]]
    assert sum(x != y for x, y in zip(a["argv"], m["argv"])) == 1          # only --out's value differs inside the shared prefix
    assert a["w_cost_ratio"] == m["w_cost_ratio"] and a["w_cost_source"] == m["w_cost_source"]
    assert a["lever"]["dose_ctl"] == DOSE and a["lever"]["w_map_hires"] == 25.0


def test_the_other_variants_carry_no_controller_flag():
    for v in ("K2", "K2M1N20", "K2M3", "K2M3N20"):
        argv = MK.compose(_doc("K0"), v, 0.1)
        assert not any(t.startswith("--map-hires-share-") or t == "--map-hires-w-bounds" for t in argv), v
