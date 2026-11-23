"""Tests for r9_e2_dose_verdict.py -- the pre-registered K2DC dose-hypothesis reader (PREREG_K2DC_2026-11-12.md, "Read -> Dose hypothesis").  CPU only.

  * the decision table: synthetic per-seed numbers reach each of SUPPORTED / REFUTED / INCONCLUSIVE, and the bars' boundaries (-2.5 mm inclusive, +-1 mm inclusive) read as written;
  * the in-training validity check (>= 2 probes at step >= 3,000 inside [0.05, 0.15]) turns the outcome into DOSE-VOID, and a missing --metrics reads UNCHECKED-VOID, never a quiet pass;
  * END TO END on synthetic refcv3_arm-form dumps (the real reader: r9_e2_families.build + the jitter bootstrap) for the three outcomes; K2 vs K2 reads REFUTED-shaped zeros;
  * DELIBERATE-REGRESSION MUTANTS of the reader's own source (the -2.5 mm bar flipped, the jitter clause dropped, the K2DC-K0 clause dropped, the band widened, `all` seeds -> `any`):
    each must make the decision-table check RED.  To see one red by hand: R9_DOSE_UNDER_TEST=<mutated copy> pytest stack/tests/test_r9_e2_dose_verdict.py -k "not mutant".
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (the dump helper needs it)
import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[0] / "scripts"
for p in (str(SCRIPTS), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
SRC = Path(os.environ.get("R9_DOSE_UNDER_TEST") or SCRIPTS / "r9_e2_dose_verdict.py")


def _load(path, name="_dose_under_test"):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load(SRC)


# ---------------------------------------------------------------------------------------------------------------- #
# the decision table                                                                                                #
# ---------------------------------------------------------------------------------------------------------------- #
def row(delta, sep=False):
    return {"delta": delta, "lo": delta - 0.001, "hi": delta + 0.001, "separated": sep}


def seed(k0=(0.001, False), cross=-0.004, final=-0.004, jit=-0.0005):
    return {"dc_vs_k0": {"LAT_cross_mae_m": row(*k0), "LAT_cross_abs_final_m": row(0.0)},
            "dc_vs_k2": {"LAT_cross_mae_m": row(cross), "LAT_cross_abs_final_m": row(final)},
            "jitter_dlag_dc_vs_k2": row(jit)}


def both(**kw):
    return {"0": seed(**kw), "1": seed(**kw)}


def table(mod):
    """(label, per_seed numbers, expected outcome): the pre-registered table.  Raises AssertionError on the first disagreement."""
    mixed = {"0": seed(), "1": seed(cross=-0.0010, final=-0.0010)}                      # seed 0 supports, seed 1 is only -1 mm
    refut_one_row = both(cross=0.0004, final=0.0030)                                      # cross inside the band, final-waypoint row outside
    cases = [
        ("supported", both(), "SUPPORTED"),
        ("supported at the bar, -2.5 mm inclusive", both(cross=-0.0025), "SUPPORTED"),
        ("just short of the bar", both(cross=-0.0024), "INCONCLUSIVE"),
        ("K2DC-K0 separated BETTER is still not worse", both(k0=(-0.002, True)), "SUPPORTED"),
        ("K2DC-K0 separated worse kills it", both(k0=(0.004, True)), "INCONCLUSIVE"),
        ("K2DC-K0 worse but not separated is fine", both(k0=(0.004, False)), "SUPPORTED"),
        ("jitter not negative kills it (bar met)", both(jit=0.0001), "INCONCLUSIVE"),
        ("jitter exactly 0 is not negative", both(jit=0.0), "INCONCLUSIVE"),
        ("one seed supports only", mixed, "INCONCLUSIVE"),
        ("refuted", both(cross=0.0002, final=-0.0003, jit=0.0), "REFUTED"),
        ("refuted at the band, +-1 mm inclusive", both(cross=0.0010, final=-0.0010), "REFUTED"),
        ("just outside the band", both(cross=0.0011, final=0.0), "INCONCLUSIVE"),
        ("one lateral row outside the band", refut_one_row, "INCONCLUSIVE"),
        ("K2DC worse than K2 by 4 mm", both(cross=0.004, final=0.004), "INCONCLUSIVE"),
        ("refuted needs BOTH seeds", {"0": seed(cross=0.0, final=0.0), "1": seed(cross=0.003, final=0.003)}, "INCONCLUSIVE"),
    ]
    for label, ps, want in cases:
        got = mod.decide(ps)["outcome_if_valid"]
        assert got == want, f"{label}: read {got}, the PREREG says {want}"


def test_decision_table():
    table(D)


def test_missing_seed_is_refused():
    with pytest.raises(SystemExit):
        D.decide({"0": seed()})


def test_bar_and_band_constants_are_the_prereg_numbers():
    assert D.BAR_M == 0.0025 and D.BAND_M == 0.0010 and D.SHARE_BAND == (0.05, 0.15) and D.SHARE_MIN_STEP == 3000 and D.SHARE_MIN_PROBES == 2


# ---------------------------------------------------------------------------------------------------------------- #
# the in-training validity check                                                                                    #
# ---------------------------------------------------------------------------------------------------------------- #
def metrics(tmp_path, shares):
    p = tmp_path / "metrics.jsonl"
    p.write_text("\n".join(json.dumps({"step": s, "loss": 1.0, **({"dc_share_read": v} if v is not None else {})}) for s, v in shares) + "\n", encoding="utf-8")
    return p


def test_void_check_held_needs_two_late_probes_inside_the_band(tmp_path):
    held = D.void_check(metrics(tmp_path, [(250, 0.4), (3000, 0.09), (3250, 0.11), (3500, 0.30)]))
    assert held["valid"] is True and held["n_inside_band"] == 2
    one = D.void_check(metrics(tmp_path, [(250, 0.10), (2750, 0.10), (3000, 0.09), (3250, 0.30)]))      # steps < 3000 do not count
    assert one["valid"] is False and one["n_inside_band"] == 1
    out = D.void_check(metrics(tmp_path, [(3000, 0.04), (3250, 0.16), (3500, 0.80)]))
    assert out["valid"] is False and out["status"] == "VOID"
    edge = D.void_check(metrics(tmp_path, [(3000, 0.05), (3250, 0.15)]))                                  # the band is inclusive
    assert edge["valid"] is True


def test_void_check_without_probes_or_file(tmp_path):
    assert D.void_check(metrics(tmp_path, [(3000, None), (3250, None)]))["valid"] is False           # the controller never fired
    assert D.void_check(None)["valid"] is None
    with pytest.raises(SystemExit):
        D.void_check(tmp_path / "nope.jsonl")


def test_void_overrides_and_unchecked_is_labelled():
    nums = {"0": seed(), "1": seed()}
    voided = D.build(nums, {"valid": False, "status": "VOID"})
    assert voided["outcome"] == "DOSE-VOID" and voided["outcome_if_valid"] == "SUPPORTED"           # still computed, never silently dropped
    unchecked = D.build(nums, {"valid": None, "status": "NOT-CHECKED"})
    assert unchecked["outcome"].startswith("SUPPORTED") and "UNCHECKED-VOID" in unchecked["outcome"]
    held = D.build(nums, {"valid": True, "status": "HELD"})
    assert held["outcome"] == "SUPPORTED"
    assert held["STRATEGIC"]["status"] == "NOT-SUPPLIED" and "PAUSED" not in json.dumps(held)        # never omitted, never a pass, never PAUSED (PI 2026-11-16)


# ---------------------------------------------------------------------------------------------------------------- #
# end to end on synthetic dumps                                                                                     #
# ---------------------------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fam_mod():
    import test_r9_e2_families as T
    return T


def make_pf(T, root, noise, seeds=(11, 12)):
    """<root>/dump_s0, dump_s1: sparse (dt 0.5) refcv3_arm-form dumps over DEV + HOLD clips; the same seed => the same draw, scaled by `noise`."""
    clips = T.DEV_CLIPS + T.HOLD_CLIPS
    for s, sd in zip((0, 1), seeds):
        T.write_dump(root / f"dump_s{s}", clips, noise, sd, sparse=True)
    return root


def split_file(T, tmp_path):
    p = tmp_path / "DEV_HOLD_SPLIT.json"
    p.write_text(json.dumps({"DEV": {"sha12": [T.sha12(c) for c in T.DEV_CLIPS]}, "HOLD": {"sha12": [T.sha12(c) for c in T.HOLD_CLIPS]}}), encoding="utf-8")
    return p


def e2e(fam_mod, tmp_path, k2dc, k2, k0):
    T = fam_mod
    pf = {n: make_pf(T, tmp_path / n, z) for n, z in (("k2dc", k2dc), ("k2", k2), ("k0", k0))}
    nums = {s: D.seed_numbers(pf["k2dc"], pf["k2"], pf["k0"], split_file(T, tmp_path), s, n_boot=300) for s in ("0", "1")}
    return D.build(nums, {"valid": True, "status": "HELD"}), nums


def test_e2e_supported(fam_mod, tmp_path):
    doc, nums = e2e(fam_mod, tmp_path, k2dc=0.010, k2=0.040, k0=0.010)        # K2DC == K0, far better than K2
    assert doc["outcome"] == "SUPPORTED", json.dumps(doc["per_seed"], indent=1)
    for s in ("0", "1"):
        assert nums[s]["dc_vs_k2"]["LAT_cross_mae_m"]["delta"] <= -0.0025 and nums[s]["jitter_dlag_dc_vs_k2"]["delta"] < 0
        assert set(nums[s]["dc_vs_k2_by_family"]) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"}      # all four families, STRATEGIC NOT-SUPPLIED (no dump rows)
        assert nums[s]["dc_vs_k2_by_family"]["STRATEGIC"]["status"] == "NOT-SUPPLIED" and nums[s]["dc_vs_k0_by_family"]["STRATEGIC"]["rows"] == {}
    assert doc["STRATEGIC"]["status"] == "NOT-SUPPLIED" and "openloop_suite.py:943" in doc["STRATEGIC"]["reason"]
    for text in (json.dumps(doc, default=float), D.md(doc)):                                                  # (a) PAUSED appears nowhere
        assert "PAUSED" not in text and "paused" not in text.lower() and "ITEM 25" not in text
    assert "NOT-SUPPLIED" in D.md(doc)


def _strict(doc):
    """Everything the decision and the validity read, i.e. the document minus the STRATEGIC report and the seeds' per-family tables."""
    return {k: v for k, v in doc.items() if k not in ("STRATEGIC", "seeds", "families_note")}


def _with_strategic(nums, rows):
    import copy
    n2 = copy.deepcopy(nums)
    for s in n2:
        for tag in ("dc_vs_k2_by_family", "dc_vs_k0_by_family"):
            n2[s][tag]["STRATEGIC"] = D.strategic_entry(rows)
    return n2


def test_strategic_separated_worse_is_report_only_and_never_moves_the_decision(fam_mod, tmp_path):
    """(b) a supplied strategic row separated WORSE: REPORT-ONLY, separated True in the JSON, visible in the md; outcome / per_seed / validity identical to the same input without it."""
    doc, nums = e2e(fam_mod, tmp_path, k2dc=0.010, k2=0.040, k0=0.010)
    worse = {"route_acc": {"delta": 0.3, "lo": 0.2, "hi": 0.4, "separated": True, "better": "lower", "n": 100}}
    doc2 = D.build(_with_strategic(nums, worse), {"valid": True, "status": "HELD"})
    st = doc2["seeds"]["0"]["dc_vs_k2_by_family"]["STRATEGIC"]
    assert st["status"] == "REPORT-ONLY" and st["rows"]["route_acc"]["separated"] is True and st["rows"]["route_acc"]["separated_worse"] is True
    assert doc2["STRATEGIC"]["status"] == "REPORT-ONLY" and doc2["STRATEGIC"]["per_seed"]["0"]["dc_vs_k2"]["separated_worse_rows"] == ["route_acc"]
    m = D.md(doc2)
    assert "STRATEGIC (report-only)" in m and "route_acc" in m and "WORSE" in m and "PAUSED" not in m
    assert _strict(doc2) == _strict(doc) and doc2["outcome"] == doc["outcome"] == "SUPPORTED"


def test_strategic_separated_better_cannot_rescue_a_failing_outcome(fam_mod, tmp_path):
    """(c) a strategic row separated BETTER by a mile cannot turn an INCONCLUSIVE (K2DC separated worse than K0) into SUPPORTED."""
    doc, nums = e2e(fam_mod, tmp_path, k2dc=0.010, k2=0.040, k0=0.002)
    assert doc["outcome"] == "INCONCLUSIVE"
    better = {"route_acc": {"delta": -9.0, "lo": -9.1, "hi": -8.9, "separated": True, "better": "lower", "n": 100}}
    doc2 = D.build(_with_strategic(nums, better), {"valid": True, "status": "HELD"})
    assert doc2["outcome"] == "INCONCLUSIVE" and _strict(doc2) == _strict(doc)
    assert doc2["seeds"]["0"]["dc_vs_k2_by_family"]["STRATEGIC"]["separated_worse_rows"] == []


def test_strategic_stamp_dict_carries_no_rows():
    assert D.strategic_entry({"status": "PAUSED", "reason": "x"})["status"] == "NOT-SUPPLIED" and D.strategic_entry(None)["status"] == "NOT-SUPPLIED"


def test_e2e_refuted_and_k2_vs_k2_reads_zeros(fam_mod, tmp_path):
    doc, nums = e2e(fam_mod, tmp_path, k2dc=0.020, k2=0.020, k0=0.010)        # K2DC is K2 (identical dumps): the dry-run shape
    assert doc["outcome"] == "REFUTED", json.dumps(doc["per_seed"], indent=1)
    for s in ("0", "1"):
        assert nums[s]["dc_vs_k2"]["LAT_cross_mae_m"]["delta"] == 0 and nums[s]["jitter_dlag_dc_vs_k2"]["delta"] == 0
        assert nums[s]["dc_vs_k0"]["LAT_cross_mae_m"]["delta"] > 0            # and it is worse than K0, as K2 is


def test_e2e_inconclusive(fam_mod, tmp_path):
    doc, nums = e2e(fam_mod, tmp_path, k2dc=0.0180, k2=0.0200, k0=0.0180)    # a small gain: between the bars
    d0 = nums["0"]["dc_vs_k2"]["LAT_cross_mae_m"]["delta"]
    assert -0.0025 < d0 < -0.0010, d0                                          # the fixture really sits between the bars
    assert doc["outcome"] == "INCONCLUSIVE"


def test_e2e_k2dc_worse_than_k0_blocks_supported(fam_mod, tmp_path):
    doc, _ = e2e(fam_mod, tmp_path, k2dc=0.010, k2=0.040, k0=0.002)           # beats K2 by a mile but is separated worse than K0
    assert doc["per_seed"]["0"]["i_dc_k0_cross_not_separated_worse"] is False and doc["outcome"] == "INCONCLUSIVE"


def test_cli_writes_json_md_and_exit_code(fam_mod, tmp_path):
    T = fam_mod
    pf = {n: make_pf(T, tmp_path / n, z) for n, z in (("k2dc", 0.010), ("k2", 0.040), ("k0", 0.010))}
    m = metrics(tmp_path, [(3000, 0.10), (3250, 0.09)])
    out = tmp_path / "out"
    rc = D.main(["--k2dc-pf", str(pf["k2dc"]), "--k2-pf", str(pf["k2"]), "--k0-pf", str(pf["k0"]), "--dev-split", str(split_file(T, tmp_path)), "--metrics", str(m),
                 "--boot", "200", "--out-dir", str(out)])
    assert rc == 0
    doc = json.loads((out / "DOSE_VERDICT.json").read_text(encoding="utf-8"))
    assert doc["outcome"] == "SUPPORTED" and doc["dose_validity"]["status"] == "HELD" and "STRATEGIC" in (out / "DOSE_VERDICT.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------- #
# deliberate-regression mutants: each must turn the decision table RED                                              #
# ---------------------------------------------------------------------------------------------------------------- #
MUTANTS = {
    "bar flipped (>= -2.5 mm)": ('bar = float(d["delta"]) <= -BAR_M + EPS', 'bar = float(d["delta"]) >= -BAR_M - EPS'),
    "jitter clause dropped": ('"supported": bool(not_worse and bar and jit)', '"supported": bool(not_worse and bar)'),
    "K2DC-K0 clause dropped": ('"supported": bool(not_worse and bar and jit)', '"supported": bool(bar and jit)'),
    "band widened to 10 mm": ("BAND_M = 0.0010", "BAND_M = 0.0100"),
    "any seed instead of both (supported)": ('sup = all(cl[s]["supported"] for s in SEEDS)', 'sup = any(cl[s]["supported"] for s in SEEDS)'),
    "any seed instead of both (refuted)": ('ref = all(cl[s]["refuted"] for s in SEEDS)', 'ref = any(cl[s]["refuted"] for s in SEEDS)'),
    "jitter sign flipped": ("jit = jd < 0.0", "jit = jd > 0.0"),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_caught(name, tmp_path):
    src = (SCRIPTS / "r9_e2_dose_verdict.py").read_text(encoding="utf-8")
    old, new = MUTANTS[name]
    assert src.count(old) == 1, f"the mutation site for {name!r} is not unique / not there"
    p = tmp_path / "mutant_dose.py"
    p.write_text(src.replace(old, new), encoding="utf-8")
    mut = _load(p, "_dose_mutant")
    with pytest.raises((AssertionError, SystemExit)):      # a wrong table value, or the reader's own overlap refusal
        table(mut)
    table(_load(SCRIPTS / "r9_e2_dose_verdict.py", "_dose_pristine"))                                  # and the pristine source passes the same table
