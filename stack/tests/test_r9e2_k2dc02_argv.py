"""K2DC02's argv is COMPOSED by r9_e2_make_argv.py, never hand-written (MM 4a3d05, 2026-11-17).

K2DC02 = K2DC with ``--map-hires-share-target 0.02`` and ``--map-hires-w-bounds 0.1,25``; every other token is K2DC's. It is the pre-registered next arm if E2-TAIL
fails clause (d) (PREREG_E2TAIL_2026-11-16.md section 5); the floor moves from 1 to 0.1 because at 0.02 a floor of 1 would bind on about half the updates
(NOTE_K2DC_SHARE002_BOUNDS_2026-11-17.md). PI-approved 2026-11-17, CONDITIONAL on E2-WISE pointing to the trunk: r9e2_K2DC02.pi_cost_approval.json carries the PI's words. No scipy, no GPU.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import r9_e2_make_argv as MK  # noqa: E402
import launch_gate as LG  # noqa: E402
import refc_v3_train as T  # noqa: E402
from tanitad.train import map_dose_ctl as M  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
FILE = RUNS / "r9e2_K2DC02.argv.json"


def _doc(v):
    return json.loads((RUNS / f"r9e2_{v}.argv.json").read_text(encoding="utf-8"))


def test_K2DC02_regenerates_byte_equal_from_the_composer():
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    d = _doc("K2DC02")
    r = float(d["w_cost_ratio"])
    rec = MK.variant_doc(json.loads(k0b.decode("utf-8")), k0b, "K2DC02", MK.w_from_ratio(r), provisional=False,
                         w_source="0.1 / %g (--w-cost-ratio: the K2 explore smoke's W-COST line)" % r)
    got = MK.dump(MK.stamp_w_cost(rec, r, d["w_cost_source"])).encode("utf-8")
    assert got == FILE.read_bytes()
    assert d["arm"] == "r9e2_K2DC02" and d["profile"] == "refcv9-e2" and d["steps"] == 6000 and d["derived_from"] == "r9e2_K0"
    assert d["base_sha256"] == hashlib.sha256(k0b).hexdigest()
    assert "PI-approved 2026-11-17, CONDITIONAL" in d["lever"]["dose_ctl_provenance"] and "PENDING" not in d["lever"]["dose_ctl_provenance"]
    ap = json.loads((RUNS / "r9e2_K2DC02.pi_cost_approval.json").read_text(encoding="utf-8"))
    assert ap["pi_verbatim"].startswith("1 i approve it") and ap["date"] == "2026-11-17" and ap["max_s_per_step"] == 12.3
    assert ap["status"].startswith("PI-APPROVED, CONDITIONAL")


def test_K2DC02_is_K2DC_except_the_out_and_the_two_dose_values():
    a, c = _doc("K2DC02")["argv"], _doc("K2DC")["argv"]
    assert len(a) == len(c)
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, c)) if x != y]
    assert [(x, y) for _, x, y in diff] == [("/home/nvidia/r8ns/r9e2/runs/r9e2-K2DC02", "/home/nvidia/r8ns/r9e2/runs/r9e2-K2DC"), ("0.02", "0.10"), ("0.1,25", "1,25")]
    assert a[diff[0][0] - 1] == "--out" and a[diff[1][0] - 1] == "--map-hires-share-target" and a[diff[2][0] - 1] == "--map-hires-w-bounds"
    assert _doc("K2DC02")["w_cost_ratio"] == _doc("K2DC")["w_cost_ratio"]
    assert _doc("K2DC02")["lever"]["w_map_hires"] == 25.0


def test_K2DC02_parses_target_and_bounds():
    argv = _doc("K2DC02")["argv"]
    ns = T.build_parser().parse_args(argv)
    assert ns.map_hires_share_target == 0.02 and ns.map_hires_w_bounds == "0.1,25"
    assert M.parse_bounds("0.1,25") == (0.1, 25.0)


def test_K2DC02_gate_static_reasons_empty_and_24_firings():
    argv = _doc("K2DC02")["argv"]
    assert LG.dose_controller_static_reasons(argv)[0] == []
    assert len(LG.dc_reading_rows(argv)) == 24


def test_K2DC_is_unchanged_by_the_new_variant():
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    d = _doc("K2DC")
    rec = MK.variant_doc(json.loads(k0b.decode("utf-8")), k0b, "K2DC", MK.w_from_ratio(float(d["w_cost_ratio"])), provisional=False,
                         w_source="0.1 / %g (--w-cost-ratio: the K2 explore smoke's W-COST line)" % float(d["w_cost_ratio"]))
    assert MK.dump(MK.stamp_w_cost(rec, float(d["w_cost_ratio"]), d["w_cost_source"])).encode("utf-8") == (RUNS / "r9e2_K2DC.argv.json").read_bytes()
