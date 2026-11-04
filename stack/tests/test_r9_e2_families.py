"""Tests for r9_e2_families.py (the paired four-family readers for the E2 judge) on SYNTHETIC refcv3_arm-form dumps.  CPU only.

  * the emitted files VALIDATE against the judge (``r9_e2_verdict``: imported from stack/scripts when landed, else the parked hold's code/ path);
  * RED arms: a flipped direction, a window-set mismatch (refused), a GT disagreement (refused), a non-DEV window (dropped / refused), a missing family
    (the judge must read NOT-COMPUTABLE), K2 == camera-blind (W4 must NOT pass), pairing by (clip, ws) and not by file index.
"""
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for p in (SCRIPTS,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import r9_e2_families as F  # noqa: E402


def _judge():
    for d in (SCRIPTS,):
        if (d / "r9_e2_verdict.py").is_file():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            return importlib.import_module("r9_e2_verdict")
    pytest.fail("r9_e2_verdict.py not found in stack/scripts: the contract this reader produces for cannot be checked")


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


K, NW = 20, 10
DEV_CLIPS = [f"clipD{i}" for i in range(8)]
HOLD_CLIPS = [f"clipH{i}" for i in range(3)]


def gt_windows(e, n=NW):
    """[n,K,2] ego-frame waypoints: speed / curvature / accel vary per window and episode."""
    out = np.zeros((n, K, 2))
    for i in range(n):
        v, yaw, a, h = 5.0 + 1.5 * e + 0.4 * i, 0.03 * (e - 3) * (1 if i % 2 else -1), (0.4 if (e + i) % 3 == 0 else -0.5 if (e + i) % 3 == 1 else 0.0), 0.0
        x = y = 0.0
        for k in range(K):
            v = max(v + a * 0.1, 0.0)
            h += yaw * 0.1
            x += v * 0.1 * math.cos(h)
            y += v * 0.1 * math.sin(h)
            out[i, k] = (x, y)
    return out


def write_dump(d, clips, noise, seed, order=None, drop_last_of=None, gt_shift=0.0, stationary_clip=None):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    eps = []
    for fi, c in enumerate(order or clips):
        e = int(c[-1]) + (0 if c.startswith("clipD") else 20)
        G = gt_windows(e)
        if c == stationary_clip:
            G = np.zeros_like(G)
        P = G + rng.normal(0.0, noise, G.shape) if noise else G.copy()
        ws = np.arange(NW) + 7
        if drop_last_of == c:
            G, P, ws = G[:-1], P[:-1], ws[:-1]
        np.savez_compressed(d / f"ep{fi:03d}.npz", g=(G + gt_shift).astype(np.float32), os=P.astype(np.float32), ws=ws, eid=np.array([fi]), clip_index=np.array([e]),
                            v0=np.ones(len(ws), np.float32))
        eps.append({"file_index": fi, "episode_index": e, "clip_id": c, "episode_id": e, "n_windows": int(len(ws))})
    (d / "manifest.json").write_text(json.dumps({"episodes": eps, "grid": {"n_windows": sum(x["n_windows"] for x in eps)}}), encoding="utf-8")
    return d


@pytest.fixture()
def dev_split(tmp_path):
    p = tmp_path / "DEV_HOLD_SPLIT.json"
    p.write_text(json.dumps({"DEV": {"sha12": [sha12(c) for c in DEV_CLIPS]}, "HOLD": {"sha12": [sha12(c) for c in HOLD_CLIPS]}}), encoding="utf-8")
    return p


@pytest.fixture()
def dumps(tmp_path):
    allc = DEV_CLIPS + HOLD_CLIPS
    return {"good": write_dump(tmp_path / "good", allc, 0.02, 1), "bad": write_dump(tmp_path / "bad", allc, 0.6, 2), "good2": write_dump(tmp_path / "good2", allc, 0.02, 1)}


def fam(dumps, dev_split, a="good", b="bad", **kw):
    return F.build(dumps[a], dumps[b], mode=kw.pop("mode", "families"), dev_split=dev_split, n_boot=300, **kw)


def test_families_json_validates_against_judge(dumps, dev_split):
    J = _judge()
    doc = fam(dumps, dev_split, "good", "bad")                      # K2 = good, K0 = bad -> nothing separated worse
    assert set(doc["rows"]) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"}
    c, rep = J.families_clause(json.loads(json.dumps(doc)))
    assert c["status"] == "PASS", c
    for f in F.FAMILIES:
        assert rep[f]["rows"] and all(r["ok"] for r in rep[f]["rows"].values())
    assert rep["STRATEGIC"]["status"] == "PAUSED"                    # never a pass, never omitted
    assert doc["STRATEGIC"]["status"] == "PAUSED" and doc["rows"]["STRATEGIC"]["status"] == "PAUSED"
    assert doc["n_windows_paired"] == len(DEV_CLIPS) * NW and doc["a"]["n_non_dev_dropped"] == len(HOLD_CLIPS) * NW
    # K2 beats K0 on every distance row: delta < 0 (lower better) and separated
    assert doc["rows"]["LATERAL"]["LAT_cross_mae_m"]["delta"] < 0 and doc["rows"]["LATERAL"]["LAT_cross_mae_m"]["separated"]


def test_row_directions_are_the_registered_table(dumps, dev_split):
    registered = {"ade_m": "lower", "fde_m": "lower", "LON_speed_mae_mps": "lower", "LON_along_mae_m": "lower", "LON_along_abs_final_m": "lower",
                  "LON_speed_within_0p5": "higher", "LON_accel_mae_mps2": "lower", "LAT_heading_mae_deg": "lower", "LAT_curvature_mae_1pm_valid": "lower",
                  "LAT_yaw_rate_mae_radps_valid": "lower", "LAT_cross_mae_m": "lower", "LAT_cross_abs_final_m": "lower", "TAC_traj_lat_correct": "higher",
                  "TAC_traj_lon_correct": "higher"}                  # pair_read_r8.METRICS, copied literally
    doc = fam(dumps, dev_split)
    got = {rid: r["better"] for f in F.FAMILIES for rid, r in doc["rows"][f].items()}
    got["ade_m"], got["fde_m"] = doc["ade_m"]["better"], doc["beside_ade_fde"]["better"]
    for rid, d in registered.items():
        assert got[rid] == d, (rid, got[rid], d)


@pytest.mark.parametrize("how", ["swap_arms", "flip_better"])
def test_RED_flipped_direction_is_read_separated_worse(dumps, dev_split, how):
    J = _judge()
    if how == "swap_arms":
        doc = fam(dumps, dev_split, "bad", "good")                   # K2 = the worse arm
    else:
        doc = copy.deepcopy(fam(dumps, dev_split, "good", "bad"))    # the right data with every direction flipped
        for f in F.FAMILIES:
            for r in doc["rows"][f].values():
                r["better"] = "higher" if r["better"] == "lower" else "lower"
    c, _ = J.families_clause(json.loads(json.dumps(doc)))
    assert c["status"] == "FAIL" and c["failing_rows"], c


def test_RED_window_set_mismatch_is_refused_not_intersected(dumps, dev_split, tmp_path):
    short = write_dump(tmp_path / "short", DEV_CLIPS + HOLD_CLIPS, 0.6, 2, drop_last_of="clipD3")
    with pytest.raises(SystemExit, match="window-set mismatch"):
        F.build(dumps["good"], short, mode="families", dev_split=dev_split, n_boot=50)
    with pytest.raises(SystemExit, match="window-set mismatch"):
        F.build(short, dumps["good"], mode="families", dev_split=dev_split, n_boot=50)


def test_RED_gt_disagreement_is_refused(dumps, dev_split, tmp_path):
    shifted = write_dump(tmp_path / "shift", DEV_CLIPS + HOLD_CLIPS, 0.6, 2, gt_shift=0.5)
    with pytest.raises(SystemExit, match="label clocks"):
        F.build(dumps["good"], shifted, mode="families", dev_split=dev_split, n_boot=50)


def test_pairing_is_by_clip_and_ws_not_file_index(dumps, dev_split, tmp_path):
    rev = write_dump(tmp_path / "rev", DEV_CLIPS + HOLD_CLIPS, 0.6, 2, order=list(reversed(DEV_CLIPS + HOLD_CLIPS)))
    ref = write_dump(tmp_path / "ref", DEV_CLIPS + HOLD_CLIPS, 0.6, 2)           # same clips, same noise stream per clip? (rng differs by order) -> compare pairing only
    d1 = F.build(dumps["good"], rev, mode="families", dev_split=dev_split, n_boot=50)
    assert d1["n_windows_paired"] == len(DEV_CLIPS) * NW and d1["pairing"]["n_only_a"] == 0 and d1["pairing"]["n_only_b"] == 0
    assert F.load_dump(rev)["keys"][0][0] == "clipH2" and F.load_dump(ref)["keys"][0][0] == "clipD0"      # file index 0 is a different clip in the two dumps


def test_non_dev_windows_dropped_by_default_and_refusable(dumps, dev_split, tmp_path):
    doc = fam(dumps, dev_split)
    assert doc["a"]["n_non_dev_dropped"] == doc["b"]["n_non_dev_dropped"] == len(HOLD_CLIPS) * NW and doc["rows"]["LATERAL"]["LAT_cross_mae_m"]["n"] == len(DEV_CLIPS) * NW
    with pytest.raises(SystemExit, match="not DEV windows"):
        fam(dumps, dev_split, non_dev="refuse")
    hold_only = write_dump(tmp_path / "ho", HOLD_CLIPS, 0.1, 3)
    with pytest.raises(SystemExit, match="no DEV window"):
        F.build(hold_only, hold_only, mode="families", dev_split=dev_split, n_boot=50)
    # a changed DEV list changes the result: the restriction is applied, not decorative
    sub = tmp_path / "dev_sub.json"
    sub.write_text(json.dumps({"DEV": {"sha12": [sha12(c) for c in DEV_CLIPS[:4]]}}), encoding="utf-8")
    d = F.build(dumps["good"], dumps["bad"], mode="families", dev_split=sub, n_boot=50)
    assert d["n_windows_paired"] == 4 * NW and d["a"]["n_non_dev_dropped"] == (len(DEV_CLIPS) - 4 + len(HOLD_CLIPS)) * NW


def test_RED_missing_family_reads_not_computable(dumps, dev_split):
    J = _judge()
    doc = json.loads(json.dumps(fam(dumps, dev_split)))
    for f in F.FAMILIES:
        broken = copy.deepcopy(doc)
        del broken["rows"][f]
        c, rep = J.families_clause(broken)
        assert c["status"] == "NOT-COMPUTABLE" and f in c["missing_families"], (f, c)
        broken["rows"][f] = {}
        assert J.families_clause(broken)[0]["status"] == "NOT-COMPUTABLE"


def test_stopped_windows_drop_from_the_heading_row_and_are_counted(dumps, dev_split, tmp_path):
    allc = DEV_CLIPS + HOLD_CLIPS
    a = write_dump(tmp_path / "sa", allc, 0.02, 1, stationary_clip="clipD0")
    b = write_dump(tmp_path / "sb", allc, 0.6, 2, stationary_clip="clipD0")
    doc = F.build(a, b, mode="families", dev_split=dev_split, n_boot=50)
    h = doc["rows"]["LATERAL"]["LAT_heading_mae_deg"]
    assert h["n_dropped_nonfinite"] >= NW and h["n"] + h["n_dropped_nonfinite"] == len(DEV_CLIPS) * NW


def test_camblind_rows_validate_w4_and_equal_arms_do_not_pass(dumps, dev_split):
    J = _judge()
    cb = fam(dumps, dev_split, "good", "bad", mode="camblind")      # K2 = good, "blind" = bad -> separated better on every primary row
    assert set(cb["rows"]) == {"ade_m", "LONGITUDINAL", "LATERAL", "TACTICAL"}
    assert cb["primary_rows"] == F.PRIMARY and cb["primary_cite"] and "all_rows" in cb and cb["STRATEGIC"]["status"] == "PAUSED"
    c, rep = J.w4_clause(json.loads(json.dumps(cb)))
    assert c["status"] == "PASS" and all(rep[k]["separated_better"] for k in ("ade_m",) + F.FAMILIES), (c, rep)
    same = fam(dumps, dev_split, "good", "good2", mode="camblind")  # K2 identical to its "camera-blind": the W4 bar must NOT be read as passed
    c2, _ = J.w4_clause(json.loads(json.dumps(same)))
    assert c2["status"] != "PASS", c2
    worse = fam(dumps, dev_split, "bad", "good", mode="camblind")   # K2 worse than blind
    assert J.w4_clause(json.loads(json.dumps(worse)))[0]["status"] == "FAIL"
    ov = fam(dumps, dev_split, "good", "bad", mode="camblind", primary={"LATERAL": "LAT_cross_mae_m"})
    assert ov["primary_rows"]["LATERAL"] == "LAT_cross_mae_m" and ov["rows"]["LATERAL"]["better"] == "lower"


def test_cli_files_feed_the_judge_end_to_end(dumps, dev_split, tmp_path):
    J = _judge()
    out = tmp_path / "o"
    assert F.main(["--mode", "families", "--a-dump", str(dumps["good"]), "--b-dump", str(dumps["bad"]), "--dev-split", str(dev_split), "--boot", "200",
                   "--out", str(out / "families_s0.json")]) == 0
    assert F.main(["--mode", "camblind", "--a-dump", str(dumps["good"]), "--b-dump", str(dumps["bad"]), "--dev-split", str(dev_split), "--boot", "200",
                   "--out", str(out / "camblind_s0.json")]) == 0
    r = J.judge_seed({"families": str(out / "families_s0.json"), "camblind": str(out / "camblind_s0.json")})
    st = {c["id"]: c["status"] for c in r["clauses"]}
    assert st["d"] == "PASS" and st["e"] == "PASS"
    assert st["a"] == st["b"] == st["c"] == "NOT-COMPUTABLE"          # the other clauses' inputs are absent here -> NOT-COMPUTABLE, never dropped
    # a refused pairing returns rc 2 and writes nothing
    assert F.main(["--mode", "families", "--a-dump", str(dumps["good"]), "--b-dump", str(tmp_path / "nope"), "--dev-split", str(dev_split), "--out", str(out / "x.json")]) == 2
    assert not (out / "x.json").exists()


def _t1_eval():
    import taniteval
    p = Path(taniteval.__file__).resolve().parents[1] / "tools" / "t1_eval.py"
    spec = importlib.util.spec_from_file_location("t1_eval_for_famroll", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_per_window_rows_reproduce_t1_eval_pooled_points(dumps):
    """The reader's per-window rows are the SAME geometry as t1_eval.analyze (not a re-derivation): the mean of each row over the windows = analyze's pooled point."""
    d = F.load_dump(dumps["bad"])
    rows = F.window_rows(d["P"], d["G"])
    out = _t1_eval().analyze(sorted(str(p) for p in Path(dumps["bad"]).glob("ep*.npz")), tiers={"os": "T1"}, n_boot=50)
    m = out["arms"]["os"]["intervals"]["metrics"]
    pairs = {"ade_dense_m": "ade_m", "fde_last_m": "fde_m", "LON_speed_mae_mps": "LON_speed_mae_mps", "LON_along_mae_m": "LON_along_mae_m",
             "LAT_cross_mae_m": "LAT_cross_mae_m", "LAT_heading_mae_deg": "LAT_heading_mae_deg"}
    for tk, rk in pairs.items():
        pt = m[tk].get("point", m[tk].get("mean"))
        mine = rows[rk][np.isfinite(rows[rk])].mean()
        assert pt is not None and abs(float(pt) - float(mine)) < 5e-3 * max(1.0, abs(float(pt))), (tk, pt, mine)
