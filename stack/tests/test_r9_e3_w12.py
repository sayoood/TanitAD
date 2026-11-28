"""test_r9_e3_w12.py -- the W1 / W2 input producer for the refcv9 E3 judge (``stack/scripts/r9_e3_w12.py``).  CPU only, no model, no GPU.

SYNTHETIC fixtures (hand-computed rates) + refcv8 data only (the banked refcv8 + H2cn baseline in the battery archive, skipped with a stated reason when the archive is absent).
No refcv9 arm's output on eval139 is read anywhere here.

Controls (each is a function ``controls(mod)`` entry that must read GREEN on the real producer and RED under a mutation of it):
  C-HAND   FD / MD rates and populations equal the hand-computed constants           RED under: all-windows mask, FD <-> MD swap
  C-ID     baseline vs itself -> delta exactly 0.0, CI [0, 0]                         RED under: misaligned pairing
  C-EID    every row carries one episode id per window (8 distinct), row_read accepts RED under: dropped episode ids
  C-JUDGE  the document through ``r9_e3_verdict.row_read`` + ``planner_bars``: one PASS and one FAIL for W1 and for W2
  C-REPRO  (real baseline) FD / MD reproduce the banked SHIP-R8 H2CN_DEPLOY8 cells    RED under: all-windows mask, FD <-> MD swap
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import json
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[1] / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W = _load("r9_e3_w12", SCRIPTS / "r9_e3_w12.py")
V = _load("r9_e3_verdict", SCRIPTS / "r9_e3_verdict.py")
ARCHIVE = Path(W.DEFAULT_ARCHIVE)
PKG_NPZ = W.REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-26-refcv8-ship-r8-h2cn-factored/out/arms/h2cn/H2CN_DEPLOY8"

try:                                            # the FD / MD definitions are IMPORTED from the refcv8 battery code under FlyWheels/; a tree without it
    _ORIG_INDICATOR, HAVE_BATTERY = W._defs()[1].indicator, True    # (Thor's launch tree is stack + taniteval only) skips this file, never errors in collection
except W.Refused:
    _ORIG_INDICATOR, HAVE_BATTERY = None, False
_ORIG_ALIGN, _ORIG_ROW = W._align, W._row       # the real functions, captured before any mutation
pytestmark = pytest.mark.skipif(not HAVE_BATTERY, reason="the refcv8 battery code (" + W.BATTERY_REL + ") is not in this tree: "
                                "the FD / MD definitions are imported from it, never re-implemented, so the producer cannot be exercised here")
NEP, NWIN = 8, 6
GT_CLS = "CCDACD"                  # per episode: window 0..5 ground-truth class (C cruise / D decel / A accel)
PLAN = {"good": "CCDACD",         # a == GT: no false decel, no missed decel
        "base": "CDDACC",         # baseline b: one false decel (w1) and one missed decel (w5) per episode
        "fd_bad": "DDDADC",       # more false decels than b (cruise windows 0,1,4 -> decel at 0,1,4)
        "md_bad": "CCCACC"}       # misses every decel
# hand-computed (GT: cruise windows 0,1,4 -> 24 windows; decel windows 2,5 -> 16 windows)
HAND = {"FD": {"n": 24, "a": 0.0, "b": 1 / 3}, "MD": {"n": 16, "a": 0.0, "b": 0.5}}


def _path(v0, v_last):
    """[6,2] positions whose LAST chord is v_last (dt 1 s) and whose speed ramps v0 -> v_last."""
    sp = np.linspace(v0, v_last, 6)
    return np.stack([np.cumsum(sp), np.zeros(6)], axis=-1)


def _v_last(cls, v0):
    return {"C": v0, "D": v0 - 3.0, "A": v0 + 3.0}[cls]


def write_dump(root: Path, name: str, plan_cls: str, *, seed=0, tier="T1", nep=NEP, extra=None, drop_seed=False, grid=None, shift_ws=0, drop_last_window=False):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    eps = []
    for e in range(nep):
        v0 = np.array([8.0 + e + 0.5 * w for w in range(NWIN)])
        G = np.stack([_path(v0[w], _v_last(GT_CLS[w], v0[w])) for w in range(NWIN)])
        P = np.stack([_path(v0[w], _v_last(plan_cls[w], v0[w])) for w in range(NWIN)])
        ws = np.arange(NWIN) + 10 * e + shift_ws
        if drop_last_window and e == nep - 1:
            G, P, ws, v0 = G[:-1], P[:-1], ws[:-1], v0[:-1]
        np.savez(d / f"ep{e:03d}.npz", g=G, os=P, ws=ws, v0=v0)
        eps.append({"file_index": e, "episode_index": e, "clip_id": f"clip-{e:03d}", "n_windows": int(len(ws))})
    man = {"tiers": {"os": tier}, "grid": grid or {"name": "6s", "dt_s": 1.0, "k": 6}, "episodes": eps}
    if not drop_seed:
        man["inference_seed"] = {"seed": seed}
    man.update(extra or {})
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return d


@pytest.fixture()
def dumps(tmp_path):
    return {k: write_dump(tmp_path, k, v) for k, v in PLAN.items()}


def build(mod, a, b, **kw):
    kw.setdefault("seed", 0)
    kw.setdefault("expect_episodes", NEP)
    kw.setdefault("population", "all")          # the synthetic dumps carry no compliant mask; the compliant default is tested below with a mask npz
    return mod.build(a, b, **kw)


# ------------------------------------------------------------------------------------------------------------------ the controls
def controls(mod, dumps, tmp_path) -> dict:
    """name -> bool (True = the control reads as required).  Run on the real module (all True) and on mutated modules (the targeted one False)."""
    out = {}
    try:
        doc = build(mod, dumps["good"], dumps["base"])
    except mod.Refused:
        return {"C-HAND": False, "C-ID": False, "C-EID": False}       # a producer that cannot pair the arms fails every control
    li = doc["lon_indicators"]
    out["C-HAND"] = all(li[k]["n_windows"] == HAND[k]["n"] and abs(li[k]["rate_a"] - HAND[k]["a"]) < 1e-12 and abs(li[k]["rate_b"] - HAND[k]["b"]) < 1e-12 for k in ("FD", "MD"))
    try:
        idd = build(mod, dumps["base"], dumps["base"])
        ok = True
        for k in ("FD", "MD"):
            r = V.row_read(idd["rows"][k])
            ok &= bool(r["ok"] and r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and not r["separated"])
    except mod.Refused:
        ok = False                              # a producer that cannot even pair the baseline with itself fails the identity control
    out["C-ID"] = bool(ok)
    ok = True
    for k in ("FD", "MD"):
        row = doc["rows"][k]
        ok &= "eid" in row and len(row["eid"]) == len(row["a"]) == len(row["b"]) and len(set(row["eid"])) == NEP
        ok &= bool(V.row_read(row)["ok"]) and V.row_read(row)["n_episodes"] == NEP
    out["C-EID"] = bool(ok)
    return out


def judge_status(doc):
    res, _ = V.planner_bars("0", {"w12": doc, "_prov": {}})
    return res["W1"][0]["status"], res["W2"][0]["status"]


def test_controls_green(dumps, tmp_path):
    c = controls(W, dumps, tmp_path)
    assert c == {"C-HAND": True, "C-ID": True, "C-EID": True}, c


# ---- mutations: each one must turn its control RED --------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def mutated(indicator=None, **patch):
    """Temporarily replace the imported definitions' indicator (or any producer function) ON THE REAL MODULE, so every internal call sees the mutation."""
    saved = {k: getattr(W, k) for k in patch}
    saved_defs = W._defs
    try:
        if indicator is not None:
            S6, CB, d = W._defs()
            fake = types.SimpleNamespace(indicator=indicator)
            W._defs = lambda code_dir=None: (S6, fake, d)
        for k, v in patch.items():
            setattr(W, k, v)
        yield W
    finally:
        W._defs = saved_defs
        for k, v in saved.items():
            setattr(W, k, v)


def _all_windows_indicator(pc, gt, mask, kind):
    """MUTATION: the population mask is ALL windows (GT class ignored)."""
    if kind == "false_decel":
        return np.where(mask, (pc == "decel").astype(np.float64), np.nan)
    return np.where(mask, (pc != "decel").astype(np.float64), np.nan)


def _swapped_indicator(pc, gt, mask, kind):
    """MUTATION: FD and MD swapped."""
    return _ORIG_INDICATOR(pc, gt, mask, "missed_decel" if kind == "false_decel" else "false_decel")


def test_mutation_all_windows_mask_is_red(dumps, tmp_path):
    with mutated(_all_windows_indicator) as m:
        c = controls(m, dumps, tmp_path)
    assert c["C-HAND"] is False, c


def test_mutation_fd_md_swap_is_red(dumps, tmp_path):
    with mutated(_swapped_indicator) as m:
        c = controls(m, dumps, tmp_path)
    assert c["C-HAND"] is False, c


def test_mutation_dropped_episode_ids_is_red(dumps, tmp_path):
    def _row_no_eid(a_vals, b_vals, eids, better):
        r = _ORIG_ROW(a_vals, b_vals, eids, better)
        r.pop("eid")
        return r
    with mutated(_row=_row_no_eid, paired_summary=lambda row, **k: {}) as mut:
        doc = mut.build(dumps["good"], dumps["base"], seed=0, expect_episodes=NEP, population="all")
    for k in ("FD", "MD"):
        assert "eid" not in doc["rows"][k]
        with pytest.raises(V.Refused):                              # the judge REFUSES a row with no episode ids (no estimator can be named), never a pass
            V.row_read(doc["rows"][k])
    # and the C-EID control, applied to the mutated document, reads RED
    ok = all("eid" in doc["rows"][k] for k in ("FD", "MD"))
    assert ok is False


def test_mutation_misaligned_pairing_is_red(dumps, tmp_path):
    def _misaligned(a, b, **k):
        ia, ib = _ORIG_ALIGN(a, b, **k)
        return ia, np.roll(ib, 1)
    with mutated(_align=_misaligned) as m:
        c = controls(m, dumps, tmp_path)
    assert c["C-ID"] is False, c


def test_definitions_are_imported_not_reimplemented():
    src = (SCRIPTS / "r9_e3_w12.py").read_text(encoding="utf-8")
    assert "def plan_lon_class" not in src and "def indicator" not in src
    S6, CB, d = W._defs()
    assert inspect.getsourcefile(S6.plan_lon_class).endswith("s6_lon_decomp.py")
    assert inspect.getsourcefile(CB.indicator).endswith("cruise_brake_retime_r8.py")
    assert "2026-10-15-refcv8-milestone-battery" in str(d).replace("\\", "/")


def test_indicator_equals_ship_l2_common_formula(dumps):
    """The SHIP-R8 verdict's own ``l2_common.lon_indicators`` (re-written here, independently of the import under test)."""
    A = W.load_dump(dumps["base"])
    ind = W.window_indicators(A["P"], A["G"], A["v0"], A["dt"])
    S6, _, _ = W._defs()
    gt, pc = S6.plan_lon_class(A["G"], A["v0"], 1.0), S6.plan_lon_class(A["P"], A["v0"], 1.0)
    mask = np.ones(len(gt), bool)
    fd = np.where(mask & (gt == "cruise"), (pc == "decel").astype(np.float64), np.nan)
    md = np.where(mask & (gt == "decel"), (pc != "decel").astype(np.float64), np.nan)
    assert np.array_equal(ind["FD"], fd, equal_nan=True) and np.array_equal(ind["MD"], md, equal_nan=True)


# ------------------------------------------------------------------------------------------------------------------ through the judge
def test_judge_w1_w2_pass_and_fail_cases(tmp_path):
    """Through ``row_read`` and the W1 / W2 block of ``planner_bars``: W1 PASS + FAIL, W2 PASS + FAIL."""
    base = write_dump(tmp_path, "base", PLAN["base"])
    got = {}
    for name in ("good", "fd_bad", "md_bad"):
        a = write_dump(tmp_path, f"a_{name}", PLAN[name])
        got[name] = judge_status(W.build(a, base, seed=0, expect_episodes=NEP, population="all"))
    assert got["good"] == ("PASS", "PASS")      # FD 0 vs 1/3 separated lower; MD 0 vs 0.5 not worse
    assert got["fd_bad"][0] == "FAIL"           # W1 FAIL: FD 1.0 vs 1/3, separated WORSE
    assert got["md_bad"][1] == "FAIL"           # W2 FAIL: MD 1.0 vs 0.5, separated worse
    assert got["md_bad"][0] == "PASS"           # W1 PASS on its own: a never decelerates (FD 0)
    assert got["fd_bad"][1] == "PASS"           # W2 PASS on its own: MD 0 vs 0.5


def test_identity_through_judge_is_not_a_pass(dumps):
    doc = build(W, dumps["base"], dumps["base"])
    assert judge_status(doc) == ("FAIL", "PASS")            # not separated lower is not W1; not separated worse is W2
    for k in ("FD", "MD"):
        s = doc["rows"][k]["summary"]
        assert (s["delta"], s["lo"], s["hi"]) == (0.0, 0.0, 0.0)


def test_document_shape_for_the_judge(dumps):
    doc = build(W, write_dump(dumps["good"].parent, "g1", PLAN["good"], seed=1), write_dump(dumps["good"].parent, "b1", PLAN["base"], seed=1), seed=1)
    assert doc["split"] == "all" and doc["tier"].startswith("T1") and doc["infer_seed"] == 1 and set(doc["rows"]) == {"FD", "MD"}
    assert doc["rows"]["FD"]["better"] == "lower" and doc["rows"]["MD"]["better"] == "lower"
    # the judge's own loader accepts it in seed slot 1 and refuses it in slot 0
    p = dumps["good"].parent / "w12_s1.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    d, _ = V.load_doc(p, "w12", "all", "T1", seed="1")
    assert d is not None
    with pytest.raises(V.Refused):
        V.load_doc(p, "w12", "all", "T1", seed="0")


# ------------------------------------------------------------------------------------------------------------------ refusals
def _refuses(fn, text=None):
    with pytest.raises(W.Refused) as ei:
        fn()
    if text:
        assert text in str(ei.value)


def test_refuses_dev_and_hold_split(dumps):
    for s in ("dev", "hold"):
        _refuses(lambda s=s: build(W, dumps["good"], dumps["base"], split=s), "ALL of eval139")


def test_refuses_dev_stamped_manifest(tmp_path):
    a = write_dump(tmp_path, "a", PLAN["good"], extra={"split": "dev"})
    b = write_dump(tmp_path, "b", PLAN["base"])
    _refuses(lambda: build(W, a, b), "DEV / HOLD")


def test_refuses_partial_episode_set_by_default(dumps):
    _refuses(lambda: W.build(dumps["good"], dumps["base"], seed=0, population="all"), "139")


def test_refuses_non_t1_and_unstamped_tier(tmp_path):
    b = write_dump(tmp_path, "b", PLAN["base"])
    _refuses(lambda: build(W, write_dump(tmp_path, "a0", PLAN["good"], tier="T0"), b), "T1")
    a = write_dump(tmp_path, "a1", PLAN["good"])
    man = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    man.pop("tiers")
    (a / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    _refuses(lambda: build(W, a, b), "no tier stamp")


def test_refuses_missing_or_wrong_seed(tmp_path):
    b = write_dump(tmp_path, "b", PLAN["base"])
    _refuses(lambda: build(W, write_dump(tmp_path, "a0", PLAN["good"], drop_seed=True), b), "inference_seed")
    _refuses(lambda: build(W, write_dump(tmp_path, "a1", PLAN["good"], seed=1), b, seed=0), "inference_seed")
    _refuses(lambda: build(W, write_dump(tmp_path, "a2", PLAN["good"]), b, seed=None), "--seed")


def test_refuses_unaligned_windows(tmp_path):
    b = write_dump(tmp_path, "b", PLAN["base"])
    _refuses(lambda: build(W, write_dump(tmp_path, "a0", PLAN["good"], drop_last_window=True), b), "window-set mismatch")
    _refuses(lambda: build(W, write_dump(tmp_path, "a1", PLAN["good"], shift_ws=1), b), "window-set mismatch")


def test_refuses_gt_disagreement(tmp_path):
    a = write_dump(tmp_path, "a", PLAN["good"])
    b = write_dump(tmp_path, "b", PLAN["base"])
    z = dict(np.load(b / "ep000.npz"))
    z["g"] = z["g"] + 0.5
    np.savez(b / "ep000.npz", **z)
    _refuses(lambda: build(W, a, b), "GT")


def test_refuses_non_s6_grid(tmp_path):
    b = write_dump(tmp_path, "b", PLAN["base"])
    _refuses(lambda: build(W, write_dump(tmp_path, "a", PLAN["good"], grid={"dt_s": 0.5, "k": 4}), b), "S6 grid")


def test_cli_exit_codes(dumps, tmp_path):
    out = tmp_path / "w12_s0.json"
    base = ["--a-dump", str(dumps["good"]), "--b-dump", str(dumps["base"]), "--seed", "0", "--out", str(out), "--expect-episodes", str(NEP)]
    assert W.main(base + ["--population", "all"]) == 0 and json.loads(out.read_text(encoding="utf-8"))["schema"] == W.SCHEMA
    out.unlink()
    assert W.main(base) == 2 and not out.exists()                   # the compliant default with no --b-plan-npz (no S6_compliant mask) is REFUSED
    assert W.main(base + ["--population", "all", "--split", "dev"]) == 2 and not out.exists()
    assert W.main(["--a-dump", str(dumps["good"]), "--b-dump", str(dumps["base"]), "--seed", "0", "--out", str(out), "--population", "all"]) == 2      # 8 episodes != 139


def test_plan_npz_alignment_and_seed(dumps, tmp_path):
    B = W.load_dump(dumps["base"])
    ws = np.asarray([w for _, w in B["keys"]], np.int64)

    def npz(name, seed, ws_=ws):
        p = tmp_path / name
        np.savez(p, S6_pos=B["P"].astype(np.float32), S6_ws=ws_, S6_clip=B["clip_index"], S6_compliant=np.ones(len(ws), bool), meta_json=json.dumps({"inference_seed": seed}))
        return p
    ok = W.apply_plan_npz(B, npz("ok.npz", 0), 0)
    assert np.allclose(ok["P"], B["P"], atol=1e-4)
    _refuses(lambda: W.apply_plan_npz(B, npz("seed.npz", 1), 0), "seed")
    _refuses(lambda: W.apply_plan_npz(B, npz("ws.npz", 0, ws + 1), 0), "do not align")


# ------------------------------------------------------------------------------------------------------------------ the compliant population (the default)
# The mask drops window 0 of every episode (a GT-cruise window; the baseline plans cruise there, so it is never a false decel).
# Hand-computed on the compliant set: FD on cruise windows 1, 4 -> n 16, b = 1/2 (w1 decel, w4 cruise), a = 0;
# MD on decel windows 2, 5 -> n 16, b = 1/2 (w2 decel, w5 cruise), a = 0.  Beside rows (all windows) keep HAND.
HAND_COMP = {"FD": {"n": 16, "a": 0.0, "b": 0.5}, "MD": {"n": 16, "a": 0.0, "b": 0.5}}


def _mask_npz(tmp_path, base_dir, seed=0, drop_w=0, size=None, drop_ws=None):
    B = W.load_dump(base_dir)
    ws = np.asarray([w for _, w in B["keys"]], np.int64)
    comp = ((ws % 10) != drop_w) if drop_ws is None else ~np.isin(ws, drop_ws)
    if size is not None:
        comp = comp[:size]
    p = tmp_path / f"mask_s{seed}_{drop_w}_{size}.npz"
    np.savez(p, S6_pos=B["P"], S6_ws=ws, S6_clip=B["clip_index"], S6_compliant=comp, meta_json=json.dumps({"inference_seed": seed}))
    return p


def _comp_ok(doc) -> bool:
    li = doc["lon_indicators"]
    ok = doc["population"] == "compliant" and doc["n_compliant"] == NEP * (NWIN - 1) and set(doc["rows"]) == {"FD", "MD", "FD_all", "MD_all"}
    for k in ("FD", "MD"):
        ok &= li[k]["n_windows"] == HAND_COMP[k]["n"] and abs(li[k]["rate_a"] - HAND_COMP[k]["a"]) < 1e-12 and abs(li[k]["rate_b"] - HAND_COMP[k]["b"]) < 1e-12
        ok &= li[k + "_all"]["n_windows"] == HAND[k]["n"] and abs(li[k + "_all"]["rate_b"] - HAND[k]["b"]) < 1e-12
    return bool(ok)


def test_compliant_population_is_the_default_and_applies_the_baseline_mask(dumps, tmp_path):
    doc = W.build(dumps["good"], dumps["base"], seed=0, expect_episodes=NEP, b_plan_npz=_mask_npz(tmp_path, dumps["base"]))
    assert _comp_ok(doc), doc["lon_indicators"]
    assert doc["rows"]["FD"]["role"].startswith("PRIMARY") and doc["rows"]["FD_all"]["role"].startswith("BESIDE")
    assert judge_status(doc) == ("PASS", "PASS")                    # the judge reads FD / MD (the compliant rows) and ignores the beside rows
    assert len(doc["compliant_mask_sha256"]) == 64


def test_compliant_mask_follows_the_window_keys_not_the_row_order(dumps, tmp_path):
    """a's windows in another file order: the mask must be applied by (clip_id, ws), not by row position."""
    a = write_dump(tmp_path, "a_rev", PLAN["good"])
    man = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    man["episodes"] = man["episodes"][::-1]
    (a / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    doc = W.build(a, dumps["base"], seed=0, expect_episodes=NEP, b_plan_npz=_mask_npz(tmp_path, dumps["base"]))
    assert _comp_ok(doc), doc["lon_indicators"]


def test_compliant_without_a_mask_is_refused(dumps, tmp_path):
    _refuses(lambda: W.build(dumps["good"], dumps["base"], seed=0, expect_episodes=NEP), "S6_compliant")
    _refuses(lambda: W.apply_plan_npz(W.load_dump(dumps["base"]), _mask_npz(tmp_path, dumps["base"], size=5), 0), "S6_compliant has 5")
    _refuses(lambda: W.build(dumps["good"], dumps["base"], seed=0, expect_episodes=NEP, population="half"), "--population")


def test_mutation_mask_ignored_is_red(dumps, tmp_path):
    real = W.window_indicators
    with mutated(window_indicators=lambda P, G, v0, dt, mask=None, code_dir=None: real(P, G, v0, dt, mask=None, code_dir=code_dir)) as m:
        doc = m.build(dumps["good"], dumps["base"], seed=0, expect_episodes=NEP, b_plan_npz=_mask_npz(tmp_path, dumps["base"]))
    assert _comp_ok(doc) is False


def test_mutation_mask_by_row_position_is_red(dumps, tmp_path):
    """MUTATION: the mask applied in b's row order instead of through the pairing index -- caught when a's file order differs."""
    a = write_dump(tmp_path, "a_rev2", PLAN["good"])
    man = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    man["episodes"] = man["episodes"][::-1]
    (a / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    real = W.window_indicators
    npz = _mask_npz(tmp_path, dumps["base"], drop_ws=[1])           # drop ONLY episode 0's w1 (ws 1): a per-episode mask, so position vs key differ
    B = W.apply_plan_npz(W.load_dump(dumps["base"]), npz, 0)
    raw = np.asarray(B["compliant"], bool)
    with mutated(window_indicators=lambda P, G, v0, dt, mask=None, code_dir=None: real(P, G, v0, dt, mask=None if mask is None else raw, code_dir=code_dir)) as m:
        bad = m.build(a, dumps["base"], seed=0, expect_episodes=NEP, b_plan_npz=npz)
    good = W.build(a, dumps["base"], seed=0, expect_episodes=NEP, b_plan_npz=npz)
    assert good["lon_indicators"]["FD"]["n_windows"] == 23 and abs(good["lon_indicators"]["FD"]["rate_b"] - 7 / 23) < 1e-12
    assert good["rows"]["FD"]["eid"].count("clip-000") == 2 and good["rows"]["FD"]["eid"].count("clip-007") == 3
    assert bad["rows"]["FD"]["eid"].count("clip-000") == 3 and bad["rows"]["FD"]["eid"].count("clip-007") == 2     # the wrong episode lost its window


# ------------------------------------------------------------------------------------------------------------------ the real baseline (refcv8 + H2cn)
needs_archive = pytest.mark.skipif(not (ARCHIVE / "s6_s0").is_dir() or not (PKG_NPZ / "DEPLOY8_POS_step48852ema_pf_s0.npz").is_file(),
                                   reason="C-REPRO UNVERIFIED here: the refcv8 battery archive (dev box) or the DEPLOY8 npz is absent")


@needs_archive
@pytest.mark.parametrize("seed", [0, 1])
def test_c_repro_banked_ship_h2cn_numbers(seed):
    """Banked: BARS_4C.json seeds.<s>.cells['H2CN_DEPLOY8-RETIME_DEPLOY8|FD' / '|MD'] keys mean_new / n_windows (compliant S6 windows)."""
    r = W.verify_baseline(ARCHIVE / f"s6_s{seed}", PKG_NPZ / f"DEPLOY8_POS_step48852ema_pf_s{seed}.npz", seed)
    assert r["ok"], r
    assert abs(r["cells"]["FD"]["mine"]["mean_new"] - 0.0603448275862069) < 1e-12 and r["cells"]["FD"]["mine"]["n_windows"] == 1856
    assert r["cells"]["MD"]["mine"]["n_windows"] == 639


@needs_archive
def test_c_repro_is_red_under_mutations():
    A, N = ARCHIVE / "s6_s0", PKG_NPZ / "DEPLOY8_POS_step48852ema_pf_s0.npz"
    for mut in (_all_windows_indicator, _swapped_indicator):
        with mutated(mut) as m:
            assert m.verify_baseline(A, N, 0)["ok"] is False
    assert W.verify_baseline(A, N, 0)["ok"] is True                  # the patch is undone


@needs_archive
def test_identity_on_real_baseline_is_exact_zero():
    A, N = ARCHIVE / "s6_s0", PKG_NPZ / "DEPLOY8_POS_step48852ema_pf_s0.npz"
    doc = W.build(A, A, seed=0, b_plan_npz=None, population="all")  # the full 139-episode default; identical dumps (no npz = no mask: the all-window read)
    assert doc["n_episodes"] == 139 and doc["lon_indicators"]["FD"]["n_windows"] > 0
    for k in ("FD", "MD"):
        s = doc["rows"][k]["summary"]
        assert (s["delta"], s["lo"], s["hi"]) == (0.0, 0.0, 0.0)
    doc2 = W.build(A, A, seed=0, b_plan_npz=N, arm_a="os")          # refcv8 os vs refcv8 + H2cn: a real, non-trivial contrast must read as such
    assert doc2["lon_indicators"]["FD"]["rate_a"] != doc2["lon_indicators"]["FD"]["rate_b"]
    # the compliant default reads the registered population: BARS_4C FD n 1,856 / MD n 639, and b's rates are the banked H2CN_DEPLOY8 cells
    li = doc2["lon_indicators"]
    assert doc2["population"] == "compliant" and li["FD"]["n_windows"] == 1856 and li["MD"]["n_windows"] == 639, li
    assert abs(li["FD"]["rate_b"] - 0.0603448275862069) < 1e-12
    assert li["FD_all"]["n_windows"] > 1856 and li["MD_all"]["n_windows"] > 639
