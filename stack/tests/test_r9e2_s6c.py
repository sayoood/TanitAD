"""test_r9e2_s6c.py -- E2-S6C: mapsel_roll_r9_e2.py + s6c_dev.py against FAKE FlyWheels modules (no GPU, no Thor, no real data).  Rates are hand-computed."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# the modules under test live in stack/ops/r9e2_read, which no runner puts on sys.path: the file must set it up itself (it passed only when
# collected after a module that did -- the Thor G-SUITE runs each file ALONE and failed it at collection, 6f9f857)
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "ops" / "r9e2_read")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mapsel_roll_r9_e2 as MR  # noqa: E402
import run_battery_r9_e2 as R9  # noqa: E402
import s6c_dev as SD  # noqa: E402

UU = [f"clip-{i}" for i in range(6)]
DEV_I, HOLD_I = (0, 2, 4), (1, 3, 5)
sha = lambda i: hashlib.sha256(UU[i].encode()).hexdigest()[:12]  # noqa: E731


def split_json(tmp_path, wpe=None):
    j = {"DEV": {"sha12": [sha(i) for i in DEV_I], "windows_per_episode": {sha(i): [2] for i in DEV_I}, "n_windows_s2": 6},
         "HOLD": {"sha12": [sha(i) for i in HOLD_I]}}
    p = tmp_path / "split.json"
    p.write_text(json.dumps(j))
    return p


# ---- the fake scorer: same SHAPE as mapsel_score.main (loaders resolved at call time inside main, DEV-filtered rows, K_BASE control, S6c BASE rate)
OFF = np.array([1., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.])        # per S6 row (2 rows per clip, clip order 0..5)
COMP = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], bool)
CLIP = np.repeat(np.arange(6), 2)
WS = np.tile([10, 20], 6)
HAND_ALL = OFF[COMP].mean()                                              # all clips, compliant rows: 4/10
HAND_DEV = OFF[COMP & np.isin(CLIP, DEV_I)].mean()


def fake_env(monkeypatch, tmp_path, *, bad_base=False, hold_in_trace=False):
    D = types.SimpleNamespace(clip_uuids=lambda: (UU, None))
    C, S2R = types.ModuleType("l2_common"), types.ModuleType("l2_oracle_s2_rows")
    base = lambda seed: {"seed": seed, "clip": CLIP.copy(), "ws": WS.copy(), "g": OFF.copy(), "v0": OFF.copy(), "os": OFF.copy(), "r7": OFF.copy(), "eid": CLIP.copy(),
                                   "compliant": COMP.copy(), "plan8": OFF.copy(), "prior8": OFF.copy(), "stage_banked": OFF.copy(), "rows": np.arange(12), "logits": OFF.copy(),
                                   "lon_v7": OFF.copy(), "lat_v7": OFF.copy(), "v0_extras": OFF.copy(), "logits_s2": OFF.copy(), "traj_s2": OFF.copy(), "sel_s2": OFF.copy(),
                                   "ex_ws": WS.copy(), "ex_clip": CLIP.copy(), "dt": 0.5, "checks": {}}
    C.load_eval_pf = base
    S2R.load_panel = lambda seed: {k: v for k, v in base(seed).items() if k in SD.PANEL_ROW_KEYS or k in ("dt", "checks")} | {"bat_os": OFF.copy(), "echo": OFF.copy()}
    for m in (C, S2R):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    M = types.ModuleType("fake_mapsel_score")
    M.__file__ = str(tmp_path / "mapsel_score.py")
    (tmp_path / "mapsel_score.py").write_text("# fake")
    M.D = D
    sel = np.arange(12)

    def load_trace(path, require_final=True):
        keep = np.ones(12, bool) if hold_in_trace else np.isin(CLIP, DEV_I)
        return {"e_i": CLIP[keep], "ws": WS[keep], "sel": sel[keep], "n": int(keep.sum())}
    M.load_trace = load_trace

    def main():
        import l2_common as C_
        import l2_oracle_s2_rows as P_
        tr = M.load_trace("x")
        E, P = C_.load_eval_pf(0), P_.load_panel(0)
        have = {(int(c), int(w)) for c, w in zip(tr["e_i"], tr["ws"])}
        if not all((int(c), int(w)) in have for c, w in zip(E["clip"], E["ws"])) or not all((int(c), int(w)) in have for c, w in zip(P["clip"], P["ws"])):
            raise SystemExit("windows missing from the trace")
        cm = E["compliant"]
        rate = float(E["g"][cm].mean()) if cm.any() else None
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        base = {"pass": not bad_base, "share": 0.5 if bad_base else 1.0}
        res = {"controls": {"K_BASE_arm_equals_emitted_pick": base, "K_RAST_all_windows_have_a_raster": {"pass": True}}, "hard_controls_pass": False,
               "map_compliance": {"S6c": {"n_windows_compliant": int(cm.sum()), "n_windows_with_gt": int(cm.sum()), "arms": {"BASE": {"off_drivable": {"rate": rate, "n": int(cm.sum())}}}}}}
        (out / "MAPSEL_RESULT.json").write_text(json.dumps(res))
        return 0
    M.main = main
    return M, C, S2R


def run(monkeypatch, tmp_path, **kw):
    M, C, S2R = fake_env(monkeypatch, tmp_path, **kw)
    sp = split_json(tmp_path)
    dev, hold = SD.load_dev_hold(sp)
    rc, census = SD.run_scorer(M, C, S2R, dev=dev, hold=hold, argv=["--out", str(tmp_path / "out")])
    res = json.loads((tmp_path / "out" / "MAPSEL_RESULT.json").read_text())
    return SD.build_flat(res, census=census, split_path=sp, seed=0), census


def test_hand_computed_dev_rate(monkeypatch, tmp_path):
    flat, census = run(monkeypatch, tmp_path)
    # DEV rows (clips 0,2,4) = idx 0,1,4,5,8,9; compliant idx 0,1,4,5,9; off there = 1,0,1,1,1 -> 4/5
    assert flat["rate"] == pytest.approx(0.8) == pytest.approx(HAND_DEV) and flat["n"] == 5
    assert flat["tier"] == "T1 self-action open loop" and "point" in flat["estimator"] and flat["better"] == "lower" and flat["infer_seed"] == 0
    assert census["S6_rows_all"] == 12 and census["S6_rows_dev"] == 6 and census["S2_rows_dev"] == 6


def test_unfiltered_score_differs_from_dev(monkeypatch, tmp_path):                # RED arm: without the DEV wraps the rate is a different number (all clips: 4/10 = 0.4, n 10)
    M, C, S2R = fake_env(monkeypatch, tmp_path, hold_in_trace=True)
    sys.argv = ["x", "--out", str(tmp_path / "u")]
    M.main()
    un = json.loads((tmp_path / "u" / "MAPSEL_RESULT.json").read_text())["map_compliance"]["S6c"]["arms"]["BASE"]["off_drivable"]
    assert un["rate"] == pytest.approx(0.4) and un["n"] == 10 and un["rate"] != pytest.approx(HAND_DEV)


def test_hold_clip_in_trace_refuses(monkeypatch, tmp_path):                      # RED arm
    with pytest.raises(SystemExit, match="HOLD"):
        run(monkeypatch, tmp_path, hold_in_trace=True)


def test_broken_k_base_refuses(monkeypatch, tmp_path):                           # RED arm
    with pytest.raises(SystemExit, match="K_BASE"):
        run(monkeypatch, tmp_path, bad_base=True)
    res = {"controls": {"K_BASE_arm_equals_emitted_pick": {"pass": True, "share": 0.99}, "K_RAST_all_windows_have_a_raster": {"pass": True}}}
    with pytest.raises(SystemExit, match="share"):
        SD.build_flat(res, census={}, split_path=__file__, seed=0)


def test_empty_and_unknown_sets_refuse():
    keep = np.zeros(4, bool)
    d = {"ws": np.arange(4), "x": np.arange(4)}
    with pytest.raises(SystemExit, match="empty"):
        SD.restrict_rows(d, keep, ("ws", "x"))
    with pytest.raises(SystemExit, match="unknown"):
        SD.restrict_rows(d, np.ones(4, bool), ("ws",))
    res = {"controls": {k: {"pass": True, "share": 1.0} for k in SD.GATING}, "map_compliance": {"S6c": {"n_windows_with_gt": 0, "arms": {"BASE": {"off_drivable": {"rate": None, "n": 0}}}}}}
    with pytest.raises(SystemExit, match="empty"):
        SD.build_flat(res, census={}, split_path=__file__, seed=0)


def test_dry_run_switches_refused(tmp_path):
    for sw in ("--allow-partial", "--controls-only"):
        with pytest.raises(SystemExit, match="dry-run"):
            SD.main(["--mapsel-code", "x", "--split", "x", "--trace", "t", "--rast", "r", "--out", "o", sw])


# ---- the roll wrapper
def test_argv_seed_and_windows(tmp_path):
    own, pre, child = MR.split_argv(["--mapsel-code", "D", "--infer-seed", "1", "--trace-out", "T", "--", "--seed", "1", "--os-only"])
    assert own["--infer-seed"] == 1 and pre == ["--trace-out", "T"] and MR.pin_child_seed(child, 1) == child
    assert MR.pin_child_seed(["--os-only"], 1)[-2:] == ["--seed", "1"]
    with pytest.raises(SystemExit, match="conflicts"):
        MR.pin_child_seed(["--seed", "0"], 1)
    with pytest.raises(SystemExit, match="required"):
        MR.split_argv(["--mapsel-code", "D", "--", "--os-only"])
    sp = split_json(tmp_path)
    dev, hold, split = R9.load_dev_hold(sp)
    w = {UU[i]: [10, 20] for i in range(6)}
    (tmp_path / "w.json").write_text(json.dumps(w))
    cen = {}
    ch = MR.dev_windows_arg(["--windows-json", str(tmp_path / "w.json")], tmp_path / "t" / "trace.npz", dev, hold, split, cen)
    kept = json.loads(Path(ch[1]).read_text())
    assert sorted(kept) == sorted(UU[i] for i in DEV_I) and cen["n_clips_dropped"] == 3
    (tmp_path / "w2.json").write_text(json.dumps({UU[0]: [10], UU[2]: [10, 20], UU[4]: [10, 20]}))        # a DEV clip short of the split's window count
    with pytest.raises(SystemExit, match="differ"):
        MR.dev_windows_arg(["--windows-json", str(tmp_path / "w2.json")], tmp_path / "t" / "trace.npz", dev, hold, split, {})


def test_remap_pin_ignores_the_devbox_assignment():
    m = types.ModuleType("fake_repoint")
    MR.pin_module_attr(m, "REMAP_OVERRIDES", "E2")
    m.REMAP_OVERRIDES = "DEVBOX"
    m.OTHER = 1
    assert m.REMAP_OVERRIDES == "E2" and m.OTHER == 1 and m._r9e2_attempts == [("REMAP_OVERRIDES", "DEVBOX")]


def test_missing_staging_refuses_at_first_call():                                # RED arm
    class Plain:
        pass

    class Proxy(Plain):
        def __call__(self, *a, **k):
            return "ran"
    M = types.SimpleNamespace(_Base=Plain, MapSelProxy=Proxy)
    stager = R9.Stager([("_bw", None), ("_bc5", None)])
    with pytest.raises(SystemExit, match="not the staging proxy"):
        MR.require_staging(M, stager)
    M2 = types.SimpleNamespace(_Base=R9.make_staging_proxy_cls(Plain, stager), MapSelProxy=Proxy)
    MR.require_staging(M2, stager)
    with pytest.raises(SystemExit, match="order|need"):
        MR.require_staging(M2, R9.Stager([("_bc5", None), ("_bw", None)]))
    MR.guard_first_call(M)
    p = Proxy()
    p.__dict__["model"] = object()
    with pytest.raises(SystemExit, match="first forward"):
        p()
    ok = Proxy()
    ok.__dict__["model"] = R9.StagedModel(object(), None, stager)
    assert ok() == "ran"
