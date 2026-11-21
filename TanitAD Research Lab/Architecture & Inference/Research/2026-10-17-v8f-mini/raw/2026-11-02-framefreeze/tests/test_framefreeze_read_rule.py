"""The pure layer of ``code/framefreeze_read.py`` (KFZ3, KFZ4, the production KFZ1 receipt, the unit-test verdict, the (window, tick) row matching), on known answers; no file access beyond tmp_path.
Run: CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q tests/test_framefreeze_read_rule.py   (from the package; the reader is loaded by file path, never through sys.path)."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("framefreeze_read", str(HERE.parent / "code" / "framefreeze_read.py"))
FR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FR)


def ticks(n_win=6, n_tick=3, seed=0):
    rng = np.random.default_rng(seed)
    win = np.repeat(np.arange(n_win), n_tick)
    tick = np.tile(np.arange(n_tick), n_win)
    return {"win": win, "tick": tick, "ci": win // 2, "t0": win * 5, "plan_raw": rng.normal(size=(n_win * n_tick, 30, 2))}


def cell(better, share):
    return {"better": better, "share_closed": share, "xy_delta": -0.1, "yp_delta": 0.2, "yp_separated_worse": True}


# ------------------------------------------------------------------------------------------------ KFZ3
def test_kfz3_identical_plans_read_zero_and_fail():
    t = ticks()
    r = FR.kfz3(t, {k: v.copy() for k, v in t.items()})
    assert r["PASS"] is False and r["max_dev_ticks_ge1_m"] == 0.0 and r["max_dev_tick0_m"] == 0.0 and r["n_rows_ticks_ge1"] == 12


def test_kfz3_threshold_is_1e3_m_at_ticks_ge1_only():
    t = ticks()
    for dev, want in ((2e-3, True), (5e-4, False), (9.9e-4, False)):
        y = {k: v.copy() for k, v in t.items()}
        y["plan_raw"][t["tick"] == 1, 4, 1] += dev
        assert FR.kfz3(t, y)["PASS"] is want, dev
    y = {k: v.copy() for k, v in t.items()}
    y["plan_raw"][t["tick"] == 0] += 1.0                                  # a tick-0 difference alone is not the gate (ticks >= 1), but it is reported
    r = FR.kfz3(t, y)
    assert r["PASS"] is False and r["max_dev_tick0_m"] == pytest.approx(1.0) and r["frac_rows_tick0_over_tol"] == 1.0


def test_kfz3_matches_rows_by_window_and_tick_not_by_position():
    t = ticks()
    perm = np.random.default_rng(1).permutation(len(t["win"]))
    y = {k: v[perm] for k, v in t.items()}
    assert FR.kfz3(t, y)["max_dev_ticks_ge1_m"] == 0.0                     # the same content in another order reads zero
    y["plan_raw"] = y["plan_raw"] + 0.5
    assert FR.kfz3(t, y)["PASS"] is True
    with pytest.raises(ValueError):
        FR.kfz3(t, {k: v[:-1] for k, v in t.items()})
    z = {k: v.copy() for k, v in t.items()}
    z["t0"] = z["t0"] + 1
    with pytest.raises(ValueError):
        FR.kfz3(t, z)


# ------------------------------------------------------------------------------------------------ KFZ4
def test_kfz4_needs_both_seeds_separated_better_and_mean_share_ge_bar():
    assert FR.kfz4({"s0": cell(True, 0.6), "s1": cell(True, 0.55)})["PASS"] is True
    assert FR.kfz4({"s0": cell(True, 0.9), "s1": cell(False, 0.9)})["PASS"] is False          # one seed not separated-better
    assert FR.kfz4({"s0": cell(True, 0.3), "s1": cell(True, 0.4)})["PASS"] is False           # mean share below the bar
    r = FR.kfz4({"s0": cell(True, 0.5), "s1": cell(True, 0.5)})
    assert r["PASS"] is True and r["mean_share_closed"] == 0.5 and r["each_seed_share_ge_bar"] is True       # the bar is inclusive
    r = FR.kfz4({"s0": cell(True, 0.9), "s1": cell(True, 0.1)})
    assert r["PASS"] is True and r["each_seed_share_ge_bar"] is False                          # mean 0.5 passes; the stricter per-seed reading is reported beside it
    r = FR.kfz4({"s0": cell(True, None), "s1": cell(True, 0.8)})
    assert r["PASS"] is False and r["undefined_share_seeds"] == ["s0"] and r["mean_share_closed"] == pytest.approx(0.4)   # undefined counts 0
    with pytest.raises(ValueError):
        FR.kfz4({"s0": cell(True, 1.0)})


# ------------------------------------------------------------------------------------------------ KFZ1 production receipt
def hdr(**over):
    h = {"scene": "hold_now", "camera_motion": False, "strict": {"camera": False}, "frame_freeze": {"items_frozen": 14394, "max_row_dev": 0.0, "n_unequal_elements": 0, "card_ref": "x"}}
    for k, v in over.items():
        if k == "ff":
            h["frame_freeze"].update(v)
        else:
            h[k] = v
    return h


def test_freeze_receipt_pass_and_each_failure():
    assert FR.freeze_receipt(hdr(), 7197)["PASS"] is True
    assert FR.freeze_receipt(hdr(ff={"items_frozen": 7197}), 7197)["PASS"] is True
    assert FR.freeze_receipt(hdr(ff={"items_frozen": 7000}), 7197)["PASS"] is False        # fewer items than ticks rows
    assert FR.freeze_receipt(hdr(ff={"items_frozen": 14395}), 7197)["PASS"] is False       # not a multiple
    assert FR.freeze_receipt(hdr(ff={"max_row_dev": 1e-3}), 7197)["PASS"] is False
    assert FR.freeze_receipt(hdr(ff={"n_unequal_elements": 3}), 7197)["PASS"] is False
    assert FR.freeze_receipt(hdr(camera_motion=True), 7197)["PASS"] is False
    assert FR.freeze_receipt(hdr(scene="logged"), 7197)["PASS"] is False
    assert FR.freeze_receipt({"scene": "hold_now"}, 7197)["PASS"] is False                 # no receipt at all


def test_pytest_verdict_reads_the_summary_line(tmp_path):
    p = tmp_path / "t.log"
    p.write_text("....\n7 passed in 64.19s (0:01:04)\n", encoding="utf-8")
    assert FR.pytest_verdict(str(p))["PASS"] is True
    p.write_text("..F\n1 failed, 6 passed in 64.19s\n", encoding="utf-8")
    assert FR.pytest_verdict(str(p))["PASS"] is False
    p.write_text("", encoding="utf-8")
    assert FR.pytest_verdict(str(p))["PASS"] is False
    r = FR.pytest_verdict(str(tmp_path / "absent.log"))
    assert r["PASS"] is False and "UNVERIFIED" in r["summary_line"]


# ------------------------------------------------------------------------------------------------ KC1 (flags reached the runs)
def tick_pair(freeze=True, hs="cv_logged"):
    t = ticks()
    fy = {k: v.copy() for k, v in t.items()}
    fh = {k: v.copy() for k, v in t.items()}
    fh["v0"] = np.full(len(t["tick"]), 5.0)
    fh["v0_in"] = np.where(t["tick"] == 0, 5.0, 6.0)
    if freeze:
        fy["scene"] = np.asarray("hold_now")
        fh["scene"] = np.asarray("hold_now")
    if hs:
        fh["histswap"] = np.asarray(hs)
    return fy, fh


def test_kc1_cell_reads_the_flags_and_the_oracle_level():
    fy, fh = tick_pair()
    assert FR.kc1_cell(fy, fh)["PASS"] is True
    assert FR.kc1_cell(*tick_pair(freeze=False))["PASS"] is False                        # no scene key: the freeze flag did not reach the ticks file
    assert FR.kc1_cell(*tick_pair(freeze=False), dry_run=True)["PASS"] is True            # a dry run has no scene keys by construction
    assert FR.kc1_cell(*tick_pair(hs="cv_exec"))["PASS"] is False
    fy2, fh2 = tick_pair()
    fy2["histswap"] = np.asarray("cv_logged")                                            # FZ-Y must carry no histswap key
    assert FR.kc1_cell(fy2, fh2)["PASS"] is False
    fy3, fh3 = tick_pair()
    fh3["v0_in"] = fh3["v0"].copy()                                                      # the oracle level never differs: it did not reach the network
    assert FR.kc1_cell(fy3, fh3)["PASS"] is False
    fy4, fh4 = tick_pair()
    fh4["v0_in"] = np.where(fh4["tick"] == 0, 5.5, 6.0)                                  # differs at tick 0: the oracle must equal the executed speed there
    assert FR.kc1_cell(fy4, fh4)["PASS"] is False


def test_kc1_cell_on_the_real_smoke_ticks_if_present():
    """B / C of smoke_fz.py (32 real windows): the production branch of the reader's KC1 reads PASS on real ticks files."""
    b = Path("F:/TanitAD-scratch/framefreeze/smoke/B_hold/ticks/ticks_SMOKEFZ_B_hold.npz")
    c = Path("F:/TanitAD-scratch/framefreeze/smoke/C_hold_cv/ticks/ticks_SMOKEFZ_C_hold_cv.npz")
    if not (b.exists() and c.exists()):
        pytest.skip("smoke ticks not present")
    ld = lambda p: {k: v for k, v in np.load(p, allow_pickle=True).items() if k not in ("tag", "ckpt")}  # noqa: E731
    r = FR.kc1_cell(ld(b), ld(c))
    assert r["PASS"] is True and r["FZ-HLVL_scene"] == "hold_now" and r["FZ-Y_scene"] == "hold_now" and r["FZ-Y_has_histswap_key"] is False
