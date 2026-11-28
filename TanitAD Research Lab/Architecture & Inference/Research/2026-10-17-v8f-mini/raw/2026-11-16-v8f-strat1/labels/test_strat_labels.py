import numpy as np, pytest
import build_strat_labels as B

def synth(n=101, dt=0.2013333):
    now = np.r_[np.nan, 0.3431 + dt * np.arange(n - 1)]
    J = {"clip": np.array(["c"]), "off": np.array([0]), "n": np.array([n]), "now_s": now}
    t = np.arange(101) * 0.2
    lab = {"c": {"t_s": t, "frame_idx": np.arange(101, dtype=np.int16), **{v: (np.arange(101) // 10).astype(np.int8) for v in B.VARS}}}
    return J, lab

def expected_idx(j, dt=0.2013333): return min(int(round(j * dt / 0.2)), 100)

def test_alignment_matches_rule():
    J, lab = synth()
    T = B.build(J, lab, {})
    exp = np.array([expected_idx(j) for j in range(101)])
    assert (T["frame_idx"] == exp).all()
    assert (T["flag"] == 0).all() and T["weather_valid"].all()

def test_shifted_time_base_breaks_alignment():
    J, lab = synth()
    exp = np.array([expected_idx(j) for j in range(101)])
    T = B.build(J, lab, {}, t_shift=1.0)     # deliberate regression: +1 s time base error
    assert not (T["frame_idx"] == exp).all()
    assert (T["frame_idx"][:90] != exp[:90]).all()

def test_tolerance_and_hold_and_absent():
    J, lab = synth(n=104)
    T = B.build(J, lab, {})
    assert T["flag"][100] == 0 and T["frame_idx"][100] == 100          # held at the last frame (0.13 s past)
    assert T["flag"][103] == 2 and T["frame_idx"][103] == -1           # 0.74 s past: invalid
    T2 = B.build(J, {}, {})
    assert (T2["flag"] == 1).all()

def test_unknown_is_invalid():
    J, lab = synth()
    lab["c"]["weather"][:] = -1
    T = B.build(J, lab, {})
    assert not T["weather_valid"].any() and T["surface_valid"].all()

def test_row_identity_with_navtok_table():
    nt = np.load("F:/TanitAD-scratch/datascale/navtok/navtok_table.npz")
    st = np.load("F:/TanitAD-scratch/strat1/labels/strat_labels_table.npz")
    for s in ("train", "eval"):
        for k in ("clip", "off", "n"):
            assert np.array_equal(nt[f"{s}_{k}"], st[f"{s}_{k}"])
        assert len(nt[f"{s}_flag"]) == len(st[f"{s}_flag"]) == len(st[f"{s}_country_id"])
