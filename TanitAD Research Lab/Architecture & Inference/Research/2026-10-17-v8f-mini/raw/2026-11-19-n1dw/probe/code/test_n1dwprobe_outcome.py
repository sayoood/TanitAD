"""unit tests of the committed outcome logic of PROBE-N1D-FEAT (n1dwprobe_outcome.py) on synthetic contrasts: one per label, the multi-label case, the one-seed case, bar edges, incomplete units."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n1dwprobe_outcome as OC  # noqa: E402

NULL = {"delta": 0.0, "lo": -0.01, "hi": 0.01, "separated": False}


def c(delta, half=0.01):
    return {"delta": delta, "lo": delta - half, "hi": delta + half, "separated": (delta - half > 0) or (delta + half < 0)}


def con(**kw):
    d = {k: dict(NULL) for k in OC.NEED}
    d.update(kw)
    return d


def units(per):
    """four identical units (both N1D seeds x both probe seeds)"""
    return {f"N1D_s{s}|tree|seed{p}": per for s in (0, 1) for p in (0, 1)}


def holds(comb):
    return [l for l in OC.LABELS if comb["labels"][l]["status"] == "HOLDS"]


def test_lost_at_token():
    cm = OC.combine(units(con(**{"NAV-TEXT": c(-0.10)})))
    assert "PROBE-LOST-AT-TOKEN" in holds(cm)
    assert "PROBE-LOST-AFTER-TOKEN" not in holds(cm)           # at-token excludes after-token
    assert "PROBE-PRESENT-UNUSED" in holds(cm)                 # PLAN-TEXT is null: not separated below


def test_lost_after_token():
    cm = OC.combine(units(con(**{"PLAN-NAV": c(-0.10), "PLAN-TEXT": c(-0.12)})))
    assert holds(cm) == ["PROBE-LOST-AFTER-TOKEN"]


def test_present_unused_only():
    cm = OC.combine(units(con()))
    assert holds(cm) == ["PROBE-PRESENT-UNUSED"]
    h = OC.headline(cm)
    assert h["holds"] == ["PROBE-PRESENT-UNUSED"] and "T0-against-T1" in h["next_lever_after_RBW_NO_GAIN"][0]


def test_vision_via_either_contrast():
    a = OC.combine(units(con(**{"PLANTEXT-PLAN": c(0.06)})))
    b = OC.combine(units(con(**{"VIS-TEXT": c(0.06)})))
    assert "PROBE-VISION" in holds(a) and "PROBE-VISION" in holds(b)
    assert "PROBE-VISION" not in holds(OC.combine(units(con())))


def test_multi_label():
    cm = OC.combine(units(con(**{"NAV-TEXT": c(-0.10), "VIS-TEXT": c(0.08)})))
    assert set(holds(cm)) == {"PROBE-LOST-AT-TOKEN", "PROBE-PRESENT-UNUSED", "PROBE-VISION"}
    assert [l for l in OC.LABELS if l in holds(cm)] == ["PROBE-LOST-AT-TOKEN", "PROBE-PRESENT-UNUSED", "PROBE-VISION"]      # the table's row order


def test_bar_edges():
    assert not OC.below(c(-0.0382))                              # inside the bar
    assert OC.below({"delta": -0.0383, "lo": -0.05, "hi": -0.01, "separated": True})
    assert not OC.below({"delta": -0.10, "lo": -0.2, "hi": 0.01, "separated": False})        # large but not separated
    assert OC.above({"delta": 0.0383, "lo": 0.01, "hi": 0.06, "separated": True})
    assert not OC.above({"delta": 0.10, "lo": -0.01, "hi": 0.2, "separated": False})


def test_one_seed_case():
    per = units(con())
    per["N1D_s1|tree|seed0"] = con(**{"NAV-TEXT": c(-0.10)})
    per["N1D_s1|tree|seed1"] = con(**{"NAV-TEXT": c(-0.10)})
    cm = OC.combine(per)                                        # LOST-AT-TOKEN at N1D_s1 only
    st = cm["labels"]["PROBE-LOST-AT-TOKEN"]
    assert st["status"] == "ONE-SEED" and st["units_holding"] == ["N1D_s1|tree|seed0", "N1D_s1|tree|seed1"] and len(st["units_not"]) == 2
    assert "PROBE-LOST-AT-TOKEN" not in OC.headline(cm)["holds"] and "PROBE-LOST-AT-TOKEN" in OC.headline(cm)["one_seed"]
    assert cm["per_unit"]["N1D_s1|tree|seed0"] == ["PROBE-LOST-AT-TOKEN", "PROBE-PRESENT-UNUSED"]
    assert cm["per_unit"]["N1D_s0|tree|seed0"] == ["PROBE-PRESENT-UNUSED"]
    assert cm["labels"]["PROBE-PRESENT-UNUSED"]["status"] == "HOLDS"


def test_not_held_and_incomplete():
    cm = OC.combine(units(con(**{"PLAN-TEXT": c(-0.2), "PLAN-NAV": c(0.0)})))
    assert cm["labels"]["PROBE-PRESENT-UNUSED"]["status"] == "NOT-HELD"
    per = units(con())
    per["N1D_s0|tree|seed0"] = {**con(), "VIS-TEXT": None}
    cm2 = OC.combine(per)
    assert all(v["status"] == "INCOMPLETE" for v in cm2["labels"].values())
    assert cm2["per_unit"]["N1D_s0|tree|seed0"] is None and OC.headline(cm2)["holds"] == []
