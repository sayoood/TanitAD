"""Tests for ``taniteval/taniteval/per_window.py`` and the opt-in ``per_window`` flag of ``adapters.navsim.four_families_block``.

WHAT IS PROVEN (SYNTHETIC NavSim-shaped windows; no dataset, no model).

  1. BYTE-IDENTICAL WHEN OFF.  The block ``four_families_block`` returns with the default flag, with ``per_window=False`` and by the function as it was BEFORE the flag existed (the source is rebuilt
     by removing exactly the lines the flag added, and the removal is asserted to have happened) are the same bytes -- on a window with a ground truth and on the no-ground-truth refusal branch.
     With the flag ON the block gains exactly ONE key (``_per_window``) and every other byte is unchanged.
  2. THE EXPORT IS THE INSTRUMENT'S OWN ARRAYS.  For EVERY exported LONGITUDINAL / LATERAL component, ``per_window.point`` (the reduction: mean, rms, ratio of sums, median) reproduces the number the block
     PRINTS, to its printed rounding -- on the dense grid (``four_families_block``) and on the sparse grid (``all_families(prefer_dense=False)``).  The TACTICAL class ids reproduce the block's accuracy and
     kappa; the goal-setting arrays reproduce its goal block.
  3. THE 21 COMPONENTS of PREREG_P6HN_SCREEN.md section 10 are all there (14 LONGITUDINAL + 7 LATERAL) beside the declared extras (last-pose heading) and the two properties of the human future, and the three
     distance-keeping components are UNAVAILABLE with a reason and n when there is no lead; with a lead their arrays are exported.
  4. The yaw rate (and heading, curvature) are MASKED at standstill (RETR-2026-09-26-YAWMASK): a stationary window contributes no step to the numerator or the denominator.
  5. DELIBERATE REGRESSION ARMS: each edit below breaks one reduction in ``per_window.py`` and must turn the block-reduction check RED.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))       # <repo>/taniteval/tests
_TE = os.path.dirname(_HERE)                             # <repo>/taniteval
_REPO = os.path.dirname(_TE)                             # <repo>
for _p in (os.path.join(_REPO, "stack"), os.path.join(_REPO, "stack", "scripts"), _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from adapters import navsim as NS                        # noqa: E402
from taniteval import four_families as FF                # noqa: E402
from taniteval import per_window as PW                   # noqa: E402

N, K, DT = 90, 8, 0.5
N_STOPPED_PRED, N_STOPPED_HUMAN = 6, 5                   # windows whose PLAN never moves / whose HUMAN never moves


def make_win(seed=0, with_lead=False):
    """a NavSim-shaped window: human arcs at varied speed and curvature (the first N_STOPPED_HUMAN windows never move), a noisy plan (the next N_STOPPED_PRED windows never move)"""
    rng = np.random.default_rng(seed)
    t = DT * np.arange(1, K + 1)
    v = rng.uniform(1.0, 14.0, N)
    v[:N_STOPPED_HUMAN] = 0.0
    curv = rng.normal(0, 0.03, N)
    x = v[:, None] * t[None]
    human = np.zeros((N, K, 3))
    human[:, :, 0] = x
    human[:, :, 1] = 0.5 * curv[:, None] * x ** 2
    human[:, :, 2] = np.arctan2(np.gradient(human[:, :, 1], axis=1), np.gradient(human[:, :, 0], axis=1) + 1e-9)
    pred = human + rng.normal(0, 0.3, human.shape)
    pred[N_STOPPED_HUMAN:N_STOPPED_HUMAN + N_STOPPED_PRED] = 0.0
    logs = [f"log{i % 9}" for i in range(N)]
    toks = [f"t{i}" for i in range(N)]
    win = NS.scenes_to_win(pred, human, frame="ego", origin_included=False, dt_s=DT, scene_tokens=toks, ego_speed_mps=v, log_names=logs)
    if with_lead:
        lead_x = 25.0 + 0.6 * v[:, None] * t[None]
        leads = np.stack([lead_x, np.zeros_like(lead_x)], -1)
        leads[::3] = np.nan                                              # a third of the windows have no lead
        win["lead"] = {"leads": leads, "lead_lens": np.full(N, 4.5), "speeds": v, "eid": logs}
    return win


def dumps(o):
    return json.dumps(o, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


def dotted(d, path):
    """a block value by component name: names carry dots of their own (`target_speed_acc.within_0.5_mps`), so a whole-key hit wins before the first dot is split"""
    if path in d:
        return d[path]
    head, _, tail = path.partition(".")
    return dotted(d[head], tail)


# --------------------------------------------------------------------------- 1. byte identity when the flag is off
_ADDED = [
    # signature
    (", per_window: bool = False) -> dict:", ") -> dict:"),
    # block-refusal branch
    ("        if per_window:\n            from taniteval import per_window as _pw\n            block[\"_per_window\"] = _pw.export(win, None)\n", ""),
    # main branch
    ("    if per_window:\n        from taniteval import per_window as _pw\n        fam[\"_per_window\"] = _pw.export(win, fam)\n", ""),
]


def preflag_module():
    """adapters/navsim.py with exactly the lines the flag added removed -> the function as it was before the flag existed"""
    src = open(NS.__file__, "rb").read().decode("utf-8").replace("\r\n", "\n")
    a = src.index("    ``per_window`` (default ``False``) is the OPT-IN per-window export")
    b = src.index("against the pre-flag source).\n", a) + len("against the pre-flag source).\n")
    src = src[:a].rstrip("\n") + "\n" + src[b:]                                                       # the docstring paragraph the flag added
    for old, new in _ADDED:
        assert src.count(old) == 1, f"the flag's lines are not exactly where the test expects them: {old[:60]!r}"
        src = src.replace(old, new)
    assert "per_window" not in src, "the pre-flag reconstruction still mentions the flag"
    mod = types.ModuleType("navsim_preflag")
    mod.__file__ = NS.__file__
    sys.modules["navsim_preflag"] = mod
    exec(compile(src, "navsim_preflag.py", "exec"), mod.__dict__)                                    # noqa: S102 -- the module under test, reconstructed
    return mod


def test_flag_off_is_byte_identical_to_the_function_before_the_flag():
    pre = preflag_module()
    for win in (make_win(0), make_win(1, with_lead=True)):
        b_pre = pre.four_families_block(win, tier="T1", n_boot=30)
        b_def = NS.four_families_block(win, tier="T1", n_boot=30)
        b_off = NS.four_families_block(win, tier="T1", n_boot=30, per_window=False)
        assert dumps(b_pre) == dumps(b_def) == dumps(b_off)
    # the no-ground-truth refusal branch
    w0 = NS.scenes_to_win(np.cumsum(np.ones((4, K, 2)), 1), None, frame="ego", origin_included=False, dt_s=DT, log_names=["a", "a", "b", "b"])
    assert dumps(pre.four_families_block(w0, tier="T1")) == dumps(NS.four_families_block(w0, tier="T1")) == dumps(NS.four_families_block(w0, tier="T1", per_window=False))
    on0 = NS.four_families_block(w0, tier="T1", per_window=True)
    assert on0["_per_window"]["status"] == "UNAVAILABLE" and "reason" in on0["_per_window"] and on0["_per_window"]["n"] == 4
    assert {k: v for k, v in on0.items() if k != "_per_window"} == NS.four_families_block(w0, tier="T1")


def test_flag_on_adds_exactly_one_key_and_changes_no_value():
    win = make_win(0)
    off = NS.four_families_block(win, tier="T1", n_boot=30)
    on = NS.four_families_block(win, tier="T1", n_boot=30, per_window=True)
    assert set(on) - set(off) == {"_per_window"} and set(off) <= set(on)
    assert dumps({k: v for k, v in on.items() if k != "_per_window"}) == dumps(off)
    assert on["_per_window"]["status"] == "OK" and on["_per_window"]["strategic"]["status"] == "PAUSED"


def test_the_preflag_reconstruction_would_see_a_changed_value():
    """the control that the byte comparison can fail: a block with one number moved differs"""
    pre = preflag_module()
    win = make_win(0)
    a = pre.four_families_block(win, tier="T1", n_boot=30)
    b = json.loads(dumps(a))
    b["longitudinal"]["speed_mae_mps"] += 1e-4
    assert dumps(a) != dumps(b)


# --------------------------------------------------------------------------- 2. the export reduces to the block's printed numbers
NOT_A_BLOCK_COMPONENT = ("last_pose_heading_err_deg",)


def mismatches(mod, win, fam, prefer_dense):
    """-> list of (family.component, point, printed) the reduction does NOT reproduce; plus the number of components checked"""
    exp = mod.export(win, fam, prefer_dense=prefer_dense)
    assert exp["status"] == "OK"
    bad, n = [], 0
    for f in ("longitudinal", "lateral"):
        for name, c in exp[f].items():
            if name in NOT_A_BLOCK_COMPONENT:
                continue                                                  # declared extra, checked separately
            n += 1
            try:
                printed = dotted(fam[f], name)
            except KeyError:
                bad.append((f"{f}.{name}", mod.point(c), "NO PRINTED VALUE IN THE BLOCK"))
                continue
            pt = mod.point(c)
            if not (isinstance(printed, (int, float)) and abs(pt - printed) <= 0.5 * 10.0 ** (-c["dp"]) * 1.02 + 1e-9):
                bad.append((f"{f}.{name}", pt, printed))
    return bad, n


def test_every_exported_component_reduces_to_the_blocks_printed_value_dense():
    win = make_win(0)
    fam = NS.four_families_block(win, tier="T1", n_boot=30)
    bad, n = mismatches(PW, win, fam, True)
    assert not bad, bad
    assert n == 21 + 2                                                    # the 21 of section 10 + the two properties of the human future


def test_every_exported_component_reduces_to_the_blocks_printed_value_sparse():
    win = make_win(2)
    fam = FF.all_families(win, prefer_dense=False, tactical_from_traj=True, strategic_no_label=True, strategic_corpus="navsim", tier="T1", n_boot=30)
    assert fam["_grid"]["used"] == "sparse"
    bad, n = mismatches(PW, win, fam, False)
    assert not bad, bad
    assert n == 21 + 2
    assert PW.export(win, fam, prefer_dense=False)["grid"] == "sparse"


def test_tactical_class_ids_reproduce_the_blocks_accuracy_and_kappa():
    win = make_win(3)
    fam = NS.four_families_block(win, tier="T1", n_boot=30)
    exp = PW.export(win, fam)
    t = exp["tactical"]
    assert t["status"] == "OK"
    for d in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
        g, p, k = t[d]["gt"], t[d]["pred"], len(t[d]["classes"])
        blk = fam["tactical"][d]
        assert round(float((g == p).mean()), 4) == blk["accuracy"]
        kap = FF._kappa_k(g, p, k)
        assert (kap is None and blk["kappa"] is None) or round(kap, 4) == blk["kappa"]
        assert t[d]["classes"] == blk["classes"] and g.shape == (N,) and p.shape == (N,)
    assert len({int(x) for x in t["maneuver_5way_collapsed"]["gt"]}) >= 2          # the fixture really has more than one executed manoeuvre


def test_goal_setting_arrays_reproduce_the_blocks_goal_block():
    win = make_win(3)
    fam = NS.four_families_block(win, tier="T1", n_boot=30)
    g = PW.export(win, fam)["tactical"]["goal_setting"]
    blk = fam["tactical"]["goal_setting"]
    for name in ("goal_point_error_m", "goal_long_bias_m", "goal_lat_bias_m", "goal_bearing_mae_deg", "goal_range_ratio"):
        assert abs(PW.point(g[name]) - blk[name]) <= 0.5e-4 * 1.02 + 1e-9, (name, PW.point(g[name]), blk[name])


# --------------------------------------------------------------------------- 3. the 21 components, the declared extras, the refusals
def test_the_21_components_of_section_10_are_exported_with_the_declared_extras():
    win = make_win(0)
    fam = NS.four_families_block(win, tier="T1", n_boot=30)
    exp = PW.export(win, fam)
    lon_expected = (set(FF.LONGITUDINAL_COMPONENTS) - {c for c in FF.LONGITUDINAL_COMPONENTS if c.startswith("distance_keeping.")} - set(PW.HUMAN_ONLY))
    lon = {k for k, c in exp["longitudinal"].items() if not c.get("human_only")}
    lat = {k for k, c in exp["lateral"].items() if not c.get("extra_to_the_four_family_block")}
    assert lon == lon_expected and len(lon) == 14
    assert lat == set(FF.LATERAL_COMPONENTS) and len(lat) == 7
    assert {k for k, c in exp["longitudinal"].items() if c.get("human_only")} == set(PW.HUMAN_ONLY)
    assert "last_pose_heading_err_deg" in exp["lateral"] and exp["lateral"]["last_pose_heading_err_deg"]["extra_to_the_four_family_block"] is True
    assert exp["n"] == N and exp["tokens"] == [f"t{i}" for i in range(N)] and exp["eid"] == [f"log{i % 9}" for i in range(N)]
    for c in exp["longitudinal"].values():
        for key in ("value", "num", "den"):
            if key in c:
                assert c[key].shape == (N,) or c["kind"] == "median"


def test_last_pose_heading_error_is_the_wrapped_yaw_difference_of_the_last_pose():
    win = make_win(4)
    exp = PW.export(win, None)
    d = win["_navsim"]["navsim_pred_heading_rad"][:, -1] - win["_navsim"]["navsim_gt_heading_rad"][:, -1]
    want = np.abs((d + np.pi) % (2 * np.pi) - np.pi) * 180.0 / math.pi
    assert np.allclose(exp["lateral"]["last_pose_heading_err_deg"]["value"], want, atol=1e-12)


def test_distance_keeping_is_unavailable_with_reason_and_n_without_a_lead():
    exp = PW.export(make_win(0), None)
    for nm in ("mean_headway_min_m", "mean_time_gap_min_s", "mean_min_ttc_s"):
        u = exp["unavailable"][f"longitudinal.distance_keeping.{nm}"]
        assert u["status"] == "UNAVAILABLE" and u["reason"] and "n" in u


def test_with_a_lead_the_distance_keeping_arrays_are_exported_and_reduce_to_the_blocks_value():
    win = make_win(1, with_lead=True)
    fam = NS.four_families_block(win, tier="T1", n_boot=30, per_window=True)
    exp = fam["_per_window"]
    assert fam["longitudinal"]["distance_keeping"]["status"] == "OK"
    for src, nm in (("mean_headway_min_m", "distance_keeping.mean_headway_min_m"), ("mean_min_ttc_s", "distance_keeping.mean_min_ttc_s")):
        assert nm in exp["longitudinal"], sorted(exp["longitudinal"])
        assert abs(PW.point(exp["longitudinal"][nm]) - fam["longitudinal"]["distance_keeping"][src]) <= 5e-4, (nm, PW.point(exp["longitudinal"][nm]), fam["longitudinal"]["distance_keeping"][src])
    assert not any(k.startswith("longitudinal.distance_keeping.mean_headway") for k in exp["unavailable"])


def test_a_standstill_window_contributes_no_step_to_the_masked_components_RETR_2026_09_26_YAWMASK():
    exp = PW.export(make_win(0), None)
    stopped = list(range(N_STOPPED_HUMAN)) + list(range(N_STOPPED_HUMAN, N_STOPPED_HUMAN + N_STOPPED_PRED))      # the human never moved, or the plan never moved
    for name in ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "curvature_bias_1pm"):
        c = exp["lateral"][name]
        assert c["kind"] == "ratio" and np.all(c["den"][stopped] == 0) and np.all(c["num"][stopped] == 0), name
    assert exp["lateral"]["yaw_rate_mae_degps"]["den"].sum() > 0


def test_export_is_deterministic_and_does_not_touch_the_window():
    win = make_win(0)
    before = dumps({k: win[k] for k in ("pred", "gt", "pred_dense", "gt_dense", "eid")})
    a, b = PW.export(win, None), PW.export(win, None)
    assert dumps(a) == dumps(b)
    assert before == dumps({k: win[k] for k in ("pred", "gt", "pred_dense", "gt_dense", "eid")})


# --------------------------------------------------------------------------- 4. deliberate regression arms
MUTATIONS = [
    ("the yaw rate is not masked at standstill", [('(P["yaw_rate"] - G["yaw_rate"]).abs(), both_pair, _DEG, 4,', '(P["yaw_rate"] - G["yaw_rate"]).abs(), torch.ones_like(both_pair), _DEG, 4,')], "lateral.yaw_rate_mae_degps"),
    ("heading is a mean of per-window means (macro) instead of a ratio of sums (micro)", [("            lat[name] = _ratio(s, cnt, scale, dp)\n", "            lat[name] = _mean(np.where(cnt > 0, scale * s / np.maximum(cnt, 1.0), 0.0), dp)\n")], "lateral.heading_mae_deg"),
    ("the rms reads as a mean square (no root)", [('return float(math.sqrt(float(np.mean(c["value"]))))', 'return float(np.mean(c["value"]))')], "longitudinal.speed_rmse_mps"),
    ("the progress median is taken over every window, invalid ones as 0", [('lon["ego_progress.progress_ratio_median"] = _median(w["ratio"][v], rows, 4)', 'lon["ego_progress.progress_ratio_median"] = _median(np.nan_to_num(w["ratio"]), np.arange(len(v)), 4)')], "longitudinal.ego_progress.progress_ratio_median"),
    ("the along-track bias is the MAE", [('lon["along_bias_m"] = _mean(_f64(al_err.mean(1)), 4)', 'lon["along_bias_m"] = _mean(_f64(al_err.abs().mean(1)), 4)')], "longitudinal.along_bias_m"),
    ("the cross-track final error reads the first step", [('lat["cross_final_mae_m"] = _mean(_f64(ct_err[:, -1].abs()), 4)', 'lat["cross_final_mae_m"] = _mean(_f64(ct_err[:, 0].abs()), 4)')], "lateral.cross_final_mae_m"),
]


def load_mutant(edits):
    src = open(PW.__file__, "r", encoding="utf-8", newline="").read().replace("\r\n", "\n")
    for old, new in edits:
        assert src.count(old) == 1, f"mutation snippet occurs {src.count(old)} times: {old[:70]!r}"
        src = src.replace(old, new)
    mod = types.ModuleType("taniteval.per_window_mutant")
    mod.__package__ = "taniteval"
    mod.__file__ = PW.__file__
    exec(compile(src, "per_window_mutant.py", "exec"), mod.__dict__)                                # noqa: S102 -- the module under test, deliberately broken
    return mod


@pytest.mark.parametrize("name,edits,component", MUTATIONS, ids=[m[0][:40] for m in MUTATIONS])
def test_regression_arm_is_caught_by_the_block_reduction_check(name, edits, component):
    win = make_win(0)
    fam = NS.four_families_block(win, tier="T1", n_boot=30)
    assert mismatches(PW, win, fam, True)[0] == []                         # the unmutated module is clean
    bad, _n = mismatches(load_mutant(edits), win, fam, True)
    assert component in [b[0] for b in bad], (name, [b[0] for b in bad])


def test_regression_arm_the_tactical_decisions_swapped_is_caught():
    win = make_win(3)
    fam = NS.four_families_block(win, tier="T1", n_boot=30)
    mut = load_mutant([('"lateral_decision": dec(lat_g, lat_p, LAT_CLASSES)', '"lateral_decision": dec(lon_g, lon_p, LAT_CLASSES)')])
    t = mut.export(win, fam)["tactical"]["lateral_decision"]
    assert round(float((t["gt"] == t["pred"]).mean()), 4) != fam["tactical"]["lateral_decision"]["accuracy"]
