"""``closedloop.py`` must route its HEADLINE through the decision-grade estimator.

THE DEFECT THIS PINS (MEASURED, 2026-07-25 code audit)
------------------------------------------------------
The closed-loop headline ADE/FDE, the compounding-error deltas and the
divergence rate were aggregated by ``_agg`` / ``_jack``: the mean ±
1.96·std/√8 of 8 OVERLAPPING random 20 % episode holdouts. That statistic is
``overlapping_holdout_se`` — the one the program formally deprecated as
**1.28–2.06× too narrow** (``Project Steering/CI_RECOMPUTE_2026-07-20.json``).
``driving.py`` was migrated to ``ci.episode_cluster_bootstrap``; this module —
the *more* decision-relevant axis, since closed-loop is where the v4
imagination thesis is judged — was not, even though the correct estimator was
already imported and used for the imagination A/B forty lines below.

This file is the ``test_driving_gate_block.py`` analogue for the closed loop.

WHAT IS PINNED
  * every emitted interval names a DECISION-GRADE estimator, and the guard
    actually bites when a deprecated one is present;
  * the bootstrap supplies the interval and never moves the point estimate;
  * the migrated headline REPRODUCES ``CI_RECOMPUTE_2026-07-20.json`` for
    flagship-30k ([0.3675, 0.4871]) — the same pin ``driving.py`` carries;
  * the migrated headline agrees EXACTLY with the imagination block, which was
    already using the right estimator (so the two cannot drift);
  * intervals got WIDER — the whole point — and the deprecated block is
    quarantined, labelled, and excluded from the guard;
  * the keys ``closedloop_report.py`` / ``run_and_save`` print still resolve.

THE FIXTURE IS REAL, NOT SYNTHETIC. Two committed, window-aligned arm dumps
over the SAME 881 windows / 40 val episodes — flagship-30k (ADE 0.4271) as the
"open-loop" side and flagship-nospeed (ADE 3.0175) as the degraded "closed-loop"
side — so the error magnitudes, the within-episode correlation and the episode
clustering that drive the interval are all real. Only the path *labels* are
assigned; no statistic is computed on invented data.

CPU-only, no GPU, no pod. Standalone: ``python taniteval/tests/test_closedloop_ci.py``.
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # taniteval/
sys.path.insert(0, "/root/taniteval")              # pod layout

from taniteval import closedloop as CL  # noqa: E402

RES = _HERE.parents[1] / "results"
GOOD = RES / "windows_flagship-30k.pt"          # ade_0_2s 0.4271 (registry §1.2)
BAD = RES / "windows_flagship-nospeed.pt"       # ade_0_2s 3.0175 (no-speed control)

# Project Steering/CI_RECOMPUTE_2026-07-20.json, arm flagship-30k — the SAME
# triple test_driving_gate_block.py pins. Reproducing it here proves the closed
# loop and the driving panel now share one estimator, not two lookalikes.
CI_RECOMPUTE = (0.4271, 0.3675, 0.4871)

_CACHE = {}


def _win():
    """A closed-loop window set built from two REAL, window-aligned arm dumps."""
    for p in (GOOD, BAD):
        if not p.exists():
            raise AssertionError(f"{p} missing — the committed dump is required")
    a = torch.load(GOOD, map_location="cpu", weights_only=False)
    b = torch.load(BAD, map_location="cpu", weights_only=False)
    assert a["eid"] == b["eid"], "dumps must be window-aligned to be paired"
    assert torch.equal(a["gt"], b["gt"])
    n, k = a["pred"].shape[0], CL.K_MAX
    return {
        "eid": a["eid"], "gt": a["gt"], "cv": a["cv"], "speed": a["speed"],
        "closed_bike": b["pred"], "closed_grnd": b["pred"],
        "open_grnd": a["pred"], "open_bike": a["pred"],
        "open_plan_bike": a["pred"], "plan_direct": a["cv"],
        # comfort inputs only — no interval is attached to these
        "steer": torch.zeros(n, k), "accel": torch.zeros(n, k),
        "vseq": a["speed"][:, None].expand(n, k).contiguous(),
    }


def _res():
    if "res" not in _CACHE:
        w = _win()
        _CACHE["win"] = w
        _CACHE["res"] = CL.analyze(w)
    return _CACHE["res"]


def _heldout():
    return _res()["closedloop_ade_fde"]["heldout"]


def _legacy():
    return _res()[CL.LEGACY_BLOCK]


# --------------------------------------------------------------------------- #
# 1. the estimator policy, enforced                                            #
# --------------------------------------------------------------------------- #
def test_every_emitted_interval_names_a_decision_grade_estimator():
    assert CL.assert_no_deprecated_estimator(_res()) is True


def test_headline_block_is_the_episode_cluster_bootstrap():
    for path, suite in _heldout().items():
        for metric, node in suite.items():
            assert node["estimator"] == "episode_cluster_bootstrap", (path, metric)
            for f in ("mean", "lo", "hi", "ci95", "n_windows", "n_episodes",
                      "n_boot"):
                assert f in node, (path, metric, f)
            # the resampling unit is the EPISODE, not the 881 correlated windows
            assert node["n_episodes"] == 40, (path, metric)
            assert node["lo"] <= node["mean"] <= node["hi"], (path, metric)


def test_compounding_deltas_are_paired_not_two_single_arm_intervals():
    for blk in ("compounding_error_grounded", "compounding_error_bicycle"):
        b = _res()[blk]
        for h in ("delta@0.5s", "delta@1s", "delta@1.5s", "delta@2s"):
            node = b[h]
            assert node["estimator"] == "paired_episode_cluster_bootstrap", (blk, h)
            assert "separated" in node and "direction" in node
            assert node["mean"] == node["delta"], "back-compat alias must hold"


def test_divergence_rate_is_migrated():
    node = _res()["stability"]["divergence_rate_gt5m@2s"]
    assert node["estimator"] == "episode_cluster_bootstrap"
    assert node["n_episodes"] == 40


def test_the_guard_actually_bites_on_a_deprecated_interval():
    """A guard that never fires is decoration. The quarantined legacy block is
    a real deprecated interval, so the UNfiltered walk must reject it."""
    import pytest
    from taniteval import driving as D
    with pytest.raises(ValueError) as ei:
        D.assert_no_deprecated_estimator(_res())
    assert CL.DEPRECATED_ESTIMATOR in str(ei.value)


# --------------------------------------------------------------------------- #
# 2. the bootstrap supplies the interval — it must not move the mean           #
# --------------------------------------------------------------------------- #
def test_point_estimate_is_the_full_set_value():
    ho = _heldout()                     # also populates _CACHE["win"]
    win = _CACHE["win"]
    for path, suite in ho.items():
        full = CL._suite(win[path], win["gt"])
        for metric, exact in full.items():
            assert abs(suite[metric]["mean"] - round(exact, 4)) <= 1e-4, \
                (path, metric, suite[metric]["mean"], exact)


def test_migrated_headline_reproduces_ci_recompute_2026_07_20():
    """flagship-30k's committed windows sit in the ``open_grnd`` slot, so its
    interval must be the published episode-cluster bootstrap to 4 decimals —
    the same triple ``driving.py`` is pinned against."""
    mean, lo, hi = CI_RECOMPUTE
    node = _heldout()["open_grnd"]["ade_0_2s"]
    assert (node["mean"], node["lo"], node["hi"]) == (mean, lo, hi), node


def test_headline_agrees_exactly_with_the_already_correct_imagination_block():
    """``heldout.closed_bike.ade_0_2s`` and ``imagination.B_closed_bike_ade@2s``
    are the SAME quantity computed in two places. The imagination block was
    already decision-grade before the migration, so exact agreement is the
    strongest available check that the migrated headline is right — and it
    stops the two from ever drifting apart again."""
    ic = _res()["imagination_comparison"]
    assert _heldout()["closed_bike"]["ade_0_2s"] == ic["B_closed_bike_ade@2s"]
    assert _res()["stability"]["divergence_rate_gt5m@2s"] == \
        ic["B_divergence_rate_gt5m@2s"]


# --------------------------------------------------------------------------- #
# 3. the point of the change: the intervals got WIDER                          #
# --------------------------------------------------------------------------- #
def test_intervals_are_wider_than_the_deprecated_estimator():
    """MEASURED on this fixture: 1.30-2.17x wider, straddling the program-wide
    1.28-2.06x finding. If a future edit ever makes them NARROWER again, the
    anti-conservative estimator is back and this fails."""
    ho, lho = _heldout(), _legacy()["heldout"]
    for path, suite in ho.items():
        for metric, node in suite.items():
            old = lho[path][metric]["ci95"]
            assert node["ci95"] > old, (path, metric, node["ci95"], old)


def test_compounding_and_divergence_intervals_are_wider_too():
    lg = _legacy()
    for blk in ("compounding_error_grounded", "compounding_error_bicycle"):
        for h in ("delta@0.5s", "delta@1s", "delta@1.5s", "delta@2s"):
            assert _res()[blk][h]["ci95"] > lg[blk][h]["ci95"], (blk, h)
    assert (_res()["stability"]["divergence_rate_gt5m@2s"]["ci95"] >
            lg["divergence_rate_gt5m@2s"]["ci95"])


def test_emitted_width_ratios_are_consistent_with_the_blocks():
    """The artifact carries its own proof; it must not be able to lie."""
    wr = _legacy()["ci_width_ratio_new_over_legacy"]
    ho, lho = _heldout(), _legacy()["heldout"]
    for name, (blk, metric) in {
            "closed_bike_ade@2s": ("closed_bike", "ade_0_2s"),
            "closed_bike_fde@2s": ("closed_bike", "fde@2s"),
            "open_grnd_ade@2s": ("open_grnd", "ade_0_2s")}.items():
        want = ho[blk][metric]["ci95"] / lho[blk][metric]["ci95"]
        assert abs(wr[name] - round(want, 3)) < 1e-9, name
        assert wr[name] > 1.0, f"{name} did not widen"


def test_deprecated_block_also_moved_the_point_estimate():
    """Not just the ± — averaging 8 random-subset means is not the full-set
    mean either. MEASURED here: the legacy headline was off by 1.5-5.9 %.
    Documented so nobody 'restores' the old means as a sanity reference."""
    win, ho, lho = _CACHE["win"], _heldout(), _legacy()["heldout"]
    drift = {p: abs(lho[p]["ade_0_2s"]["mean"] - ho[p]["ade_0_2s"]["mean"])
             for p in ho}
    assert max(drift.values()) > 0.01, drift          # they really do differ
    for p in ho:                                       # ...and the NEW one is right
        assert abs(ho[p]["ade_0_2s"]["mean"] -
                   round(CL._suite(win[p], win["gt"])["ade_0_2s"], 4)) <= 1e-4


# --------------------------------------------------------------------------- #
# 4. quarantine + backward compatibility                                       #
# --------------------------------------------------------------------------- #
def test_legacy_block_is_quarantined_and_self_labelling():
    lg = _legacy()
    assert lg["_estimator"] == CL.DEPRECATED_ESTIMATOR
    for path, suite in lg["heldout"].items():
        for metric, node in suite.items():
            assert node["estimator"] == CL.DEPRECATED_ESTIMATOR, (path, metric)
            assert node["deprecated"] is True, (path, metric)
    # ...and it is NOT the headline: `summary` quotes the migrated block only
    s = _res()["summary"]
    assert s["closed_bike_ade@2s"] == _heldout()["closed_bike"]["ade_0_2s"]["mean"]
    assert s["closed_bike_ade@2s_ci95"] == \
        _heldout()["closed_bike"]["ade_0_2s"]["ci95"]


def test_report_and_runner_consumer_keys_still_resolve():
    """The exact reads in closedloop_report.py + closedloop.run_and_save. A
    migration that silently renames a key breaks the report at print time,
    which is the worst possible moment to find out."""
    res = _res()
    ho = res["closedloop_ade_fde"]["heldout"]
    for p in ("closed_bike", "open_grnd"):
        assert isinstance(ho[p]["ade_0_2s"]["mean"], float)
        assert isinstance(ho[p]["ade_0_2s"]["ci95"], float)
    assert isinstance(ho["closed_bike"]["fde@2s"]["mean"], float)
    for blk in ("compounding_error_grounded", "compounding_error_bicycle"):
        d2 = res[blk]["delta@2s"]
        assert isinstance(d2["mean"], float) and isinstance(d2["ci95"], float)
    assert isinstance(res["stability"]["divergence_rate_gt5m@2s"]["mean"], float)
    for k in ("closed_bike_ade@2s", "closed_minus_open_grnd_de@2s",
              "divergence_rate_gt5m@2s", "imagination_verdict"):
        assert k in res["summary"], k
    for lab, row in res["speed_stratified"]["by_speed"].items():
        assert "n" in row and "divergence_rate_gt5m@2s" in row, lab


def test_provenance_stamp_and_gate_readable_block():
    """Mirrors driving.tier0: `primary_ci` + a `cluster_bootstrap['model']`
    block in the shape ``run_gate._read_eval_metric`` reads, so a gate on the
    closed-loop axis can never fall back to the deprecated estimator (the
    ⭐ v4 gate bug, 2026-07-22)."""
    res = _res()
    assert res["primary_ci"] == "episode_cluster_bootstrap"
    est = res["estimator"]
    assert est["interval"] == "episode_cluster_bootstrap"
    assert est["delta"] == "paired_episode_cluster_bootstrap"
    assert est["resampling_unit"] == "val episode"
    assert est["deprecated_and_refused"] == CL.DEPRECATED_ESTIMATOR
    assert est["legacy_block"] == CL.LEGACY_BLOCK
    model = res["cluster_bootstrap"]["model"]
    assert model == res["closedloop_ade_fde"]["heldout"]["closed_bike"]
    for k in ("ade_0_2s", "fde@2s"):
        assert model[k]["estimator"] == "episode_cluster_bootstrap", k
    assert "episode_cluster_bootstrap" in res["protocol"]["ci"]
    assert CL.DEPRECATED_ESTIMATOR not in res["protocol"]["ci"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:                                    # noqa: BLE001
            bad += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"==== {len(fns) - bad}/{len(fns)} passed ====")
    sys.exit(1 if bad else 0)
