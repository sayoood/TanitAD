"""test_r9e3_s6c_hold.py -- E3-S6C-HOLD: s6c_hold.py against FAKE FlyWheels modules (no GPU, no Thor, no real data).  Rates are hand-computed; every control has a deliberate-regression arm."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "ops" / "r9e2_read"), str(ROOT / "ops" / "r9e3_read")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import s6c_hold as SH  # noqa: E402

UU = [f"clip-{i}" for i in range(6)]
DEV_I, HOLD_I = (0, 2, 4), (1, 3, 5)
sha = lambda i: hashlib.sha256(UU[i].encode()).hexdigest()[:12]  # noqa: E731
OFF = np.array([1., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.])        # per S6 row (2 rows per clip, clip order 0..5); row 6 is clip 3 etc.
OFF[6] = 1.
COMP = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], bool)
CLIP = np.repeat(np.arange(6), 2)
WS = np.tile([10, 20], 6)
ALL_RATE = OFF[COMP].mean()
HOLD_RATE = OFF[COMP & np.isin(CLIP, HOLD_I)].mean()
DEV_RATE = OFF[COMP & np.isin(CLIP, DEV_I)].mean()


def split_json(tmp_path, dev=DEV_I, hold=HOLD_I):
    j = {"DEV": {"sha12": [sha(i) for i in dev]}, "HOLD": {"sha12": [sha(i) for i in hold]}}
    p = tmp_path / "split.json"
    p.write_text(json.dumps(j))
    return p


def fake_env(monkeypatch, tmp_path, *, uu=None, bad_base=False):
    uu = UU if uu is None else uu
    C, S2R, PR = types.ModuleType("l2_common"), types.ModuleType("l2_oracle_s2_rows"), types.ModuleType("pair_read_r8")
    base = lambda seed: {"seed": seed, "clip": CLIP.copy(), "ws": WS.copy(), "g": OFF.copy(), "v0": OFF.copy(), "os": OFF.copy(), "r7": OFF.copy(), "eid": CLIP.copy(),
                         "compliant": COMP.copy(), "plan8": OFF.copy(), "prior8": OFF.copy(), "stage_banked": OFF.copy(), "rows": np.arange(12), "logits": OFF.copy(),
                         "lon_v7": OFF.copy(), "lat_v7": OFF.copy(), "v0_extras": OFF.copy(), "logits_s2": OFF.copy(), "traj_s2": OFF.copy(), "sel_s2": OFF.copy(),
                         "ex_ws": WS.copy(), "ex_clip": CLIP.copy(), "dt": 0.5, "checks": {}}
    C.load_eval_pf = base
    S2R.load_panel = lambda seed: {k: v for k, v in base(seed).items() if k in SH.SD.PANEL_ROW_KEYS or k in ("dt", "checks")} | {"bat_os": OFF.copy(), "echo": OFF.copy()}
    PR.paired_cell = lambda x, b, eid, *a, **k: {"fake": True}
    for m in (C, S2R, PR):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    M = types.ModuleType("fake_mapsel_score")
    M.__file__ = str(tmp_path / "mapsel_score.py")
    (tmp_path / "mapsel_score.py").write_text("# fake")
    M.D = types.SimpleNamespace(clip_uuids=lambda: (uu, None))

    def load_trace(path, require_final=True):
        d = {"e_i": CLIP.copy(), "ws": WS.copy(), "sel": np.arange(12), "fan": np.arange(24.).reshape(12, 2)}
        d["idx"] = {(int(e), int(w)): j for j, (e, w) in enumerate(zip(d["e_i"], d["ws"]))}
        d["meta"], d["n"] = {"final": True}, 12
        return d
    M.load_trace = load_trace

    def main():
        import l2_common as C_
        import l2_oracle_s2_rows as P_
        import pair_read_r8 as PR_
        tr = M.load_trace("x")
        E, P = C_.load_eval_pf(0), P_.load_panel(0)
        have = {(int(c), int(w)) for c, w in zip(tr["e_i"], tr["ws"])}
        if not all((int(c), int(w)) in have for c, w in zip(E["clip"], E["ws"])) or not all((int(c), int(w)) in have for c, w in zip(P["clip"], P["ws"])):
            raise SystemExit("windows missing from the trace")
        cm = E["compliant"]
        x = np.where(cm, E["g"], np.nan)
        PR_.paired_cell(x, x, list(E["eid"]))
        rate = float(E["g"][cm].mean()) if cm.any() else None
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        res = {"controls": {"K_BASE_arm_equals_emitted_pick": {"pass": not bad_base, "share": 1.0}, "K_RAST_all_windows_have_a_raster": {"pass": True}}, "hard_controls_pass": False,
               "map_compliance": {"S6c": {"n_windows_compliant": int(cm.sum()), "n_windows_with_gt": int(cm.sum()), "arms": {"BASE": {"off_drivable": {"rate": rate, "n": int(cm.sum())}}}}}}
        (out / "MAPSEL_RESULT.json").write_text(json.dumps(res))
        return 0
    M.main = main
    return M, C, S2R, PR


def run(monkeypatch, tmp_path, subset, *, uu=None, split=None, **kw):
    M, C, S2R, PR = fake_env(monkeypatch, tmp_path, uu=uu, **kw)
    sp = split or split_json(tmp_path)
    dev, hold = SH.SD.load_dev_hold(sp)
    rc, census = SH.run_scorer(M, C, S2R, PR, dev=dev, hold=hold, subset=subset, argv=["--out", str(tmp_path / f"out_{subset}")])
    res = json.loads((tmp_path / f"out_{subset}" / "MAPSEL_RESULT.json").read_text())
    return SH.summarize(res, census, subset=subset, ci=SH.load_ci()), census


def assert_hold_only(census):
    """The HOLD result must hold HOLD clips only."""
    assert set(np.unique(census["_s6_clip"]).tolist()) <= set(HOLD_I), sorted(set(np.unique(census["_s6_clip"]).tolist()))


def test_hand_computed_hold_dev_all(monkeypatch, tmp_path):
    h, ch = run(monkeypatch, tmp_path, "hold")
    d, cd = run(monkeypatch, tmp_path, "dev")
    a, ca = run(monkeypatch, tmp_path, "all")
    assert h["rate"] == pytest.approx(HOLD_RATE) and h["n"] == int(COMP[np.isin(CLIP, HOLD_I)].sum()) and h["n_episodes"] == 3
    assert d["rate"] == pytest.approx(DEV_RATE) and a["rate"] == pytest.approx(ALL_RATE) and a["n"] == int(COMP.sum())
    assert len({round(h["rate"], 9), round(d["rate"], 9)}) == 2           # the control only means something if HOLD != DEV on the fixture
    assert_hold_only(ch)
    assert ch["trace_windows_all"] == 12 and ch["trace_windows_kept"] == 6 and ch["S6_rows_kept"] == 6 and ch["S2_rows_kept"] == 6
    b = h["bootstrap"]
    assert b["mean"] == pytest.approx(h["rate"], abs=1e-6) and b["n_episodes"] == 3 and b["n_boot"] == 2000 and b["lo"] <= b["mean"] <= b["hi"]


def test_dev_window_reaching_hold_result_is_caught(monkeypatch, tmp_path):        # RED arm: the unrestricted run lets DEV windows in; the HOLD-only assertion must go red
    _, c_all = run(monkeypatch, tmp_path, "all")
    with pytest.raises(AssertionError):
        assert_hold_only(c_all)
    a, _ = run(monkeypatch, tmp_path, "all")
    h, _ = run(monkeypatch, tmp_path, "hold")
    assert a["rate"] != pytest.approx(h["rate"])


def test_window_in_neither_split_refuses(monkeypatch, tmp_path):
    sp = split_json(tmp_path, dev=(0, 2), hold=HOLD_I)                           # clip 4 is in neither
    for subset in ("hold", "dev", "all"):
        with pytest.raises(SystemExit, match="NEITHER"):
            run(monkeypatch, tmp_path, subset, split=sp)
    run(monkeypatch, tmp_path, "hold")                                           # RED arm of the refusal: with every clip assigned it does not refuse


def test_empty_set_refuses(monkeypatch, tmp_path):
    sp = split_json(tmp_path, dev=(0, 1, 2, 3, 4, 5)[:5], hold=(5,))             # HOLD = clip 5 only ...
    uu = UU
    with pytest.raises(SystemExit):
        SH.subset_mask(np.array([0, 2]), uu, *SH.SD.load_dev_hold(sp), "hold")   # ... but the windows given hold none of it: empty
    d, h = SH.SD.load_dev_hold(sp)
    assert SH.subset_mask(np.array([0, 5]), uu, d, h, "hold").tolist() == [False, True]     # RED arm: a non-empty HOLD is kept
    res = {"controls": {k: {"pass": True} for k in SH.GATING}, "map_compliance": {"S6c": {"arms": {"BASE": {"off_drivable": {"rate": None, "n": 0}}}}}}
    with pytest.raises(SystemExit, match="empty"):
        SH.summarize(res, {}, subset="hold")


def test_capture_must_reproduce_scorer(monkeypatch, tmp_path):                   # RED arm: a captured vector that disagrees with the scorer's rate refuses
    h, _ = run(monkeypatch, tmp_path, "hold")
    res = {"controls": {k: {"pass": True} for k in SH.GATING},
           "map_compliance": {"S6c": {"arms": {"BASE": {"off_drivable": {"rate": 0.5, "n": 2}}}}}}
    cen = {"base_vec": np.array([1., 0.]), "base_eid": np.array([0, 1])}
    assert SH.summarize(res, cen, subset="hold", ci=SH.load_ci())["rate"] == 0.5
    cen["base_vec"] = np.array([1., 1.])
    with pytest.raises(SystemExit, match="captured vector"):
        SH.summarize(res, cen, subset="hold", ci=SH.load_ci())


def test_broken_k_base_refuses(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="K_BASE"):
        run(monkeypatch, tmp_path, "hold", bad_base=True)


def test_real_data_control_function(tmp_path):                                   # the ALL read must equal the banked MAPSEL_RESULT.json exactly; a one-ulp-off read is red
    banked = tmp_path / "MAPSEL_RESULT.json"
    banked.write_text(json.dumps({"map_compliance": {"S6c": {"arms": {"BASE": {"off_drivable": {"rate": 0.17917, "n": 3198}}}}}}))
    assert SH.real_data_control({"rate": 0.17917, "n": 3198}, banked)["pass"] is True
    assert SH.real_data_control({"rate": 0.17917 + 1e-12, "n": 3198}, banked)["pass"] is False
    assert SH.real_data_control({"rate": 0.17917, "n": 3197}, banked)["pass"] is False


def test_dry_run_switches_refused():
    for sw in ("--allow-partial", "--controls-only"):
        with pytest.raises(SystemExit, match="dry-run"):
            SH.main(["--mapsel-code", "x", "--split", "x", "--subset", "hold", "--trace", "t", "--rast", "r", "--out", "o", "--read-out", "r", sw])


def test_zero_byte_raster_drop_is_control_only(monkeypatch, tmp_path):             # the drop is for dev / all CONTROL modes; HOLD refuses it (RED arm: hold without the flag runs)
    rast = tmp_path / "rast"
    rast.mkdir()
    for e, w in zip(CLIP, WS):
        (rast / f"e{int(e):03d}_w{int(w):04d}.npz").write_bytes(b"x")
    (rast / "e000_w0010.npz").write_bytes(b"")                                       # clip 0 (DEV), ws 10: 0 bytes
    M, C, S2R, PR = fake_env(monkeypatch, tmp_path)
    dev, hold = SH.SD.load_dev_hold(split_json(tmp_path))
    with pytest.raises(SystemExit, match="never drops"):
        SH.run_scorer(M, C, S2R, PR, dev=dev, hold=hold, subset="hold", argv=["--out", str(tmp_path / "o1")], rast_dir=rast)
    rc, cen = SH.run_scorer(M, C, S2R, PR, dev=dev, hold=hold, subset="all", argv=["--out", str(tmp_path / "o2")], rast_dir=rast)
    res = json.loads((tmp_path / "o2" / "MAPSEL_RESULT.json").read_text())
    assert cen["dropped_zero_byte_raster_windows"] == [(0, 10)] and cen["trace_windows_kept"] == 11 and cen["S6_rows_kept"] == 11 and cen["S2_rows_kept"] == 11
    assert res["map_compliance"]["S6c"]["arms"]["BASE"]["off_drivable"]["n"] == int(COMP.sum()) - 1                  # row 0 was compliant
    rc, cen = SH.run_scorer(M, C, S2R, PR, dev=dev, hold=hold, subset="hold", argv=["--out", str(tmp_path / "o3")])
    assert "dropped_zero_byte_raster_windows" not in cen and cen["trace_windows_kept"] == 6
