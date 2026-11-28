"""E5b gates (CPU): DK rows from scene tracks, S-POS@30/50/80, the bootstrap over scenes, and the outcome logic of PREREG_T2_CRAWL section 6.
Every `check_*` takes `mut=None` (or a mutated function); `code/run_mutations_e5b.py` runs each with deliberate regressions that must raise AssertionError (RED).
The heavy readers bind their own tree, so the DK / rc / S-POS / bootstrap checks run in the scorer's subprocess (`t2v8f_score.run_sub`).
Run from t2v8f/: CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_e5b.py -q"""
from __future__ import annotations

import inspect
import itertools
import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import t2v8f_score as S  # noqa: E402

ART = S.ART
RIC0_ARR = f"{ART}/dagland/motion_c/ri_read/raw/arrays/RIC0_s0.npz"
RIC0_DUMP = "F:/TanitAD-scratch/motion_c/ri/out/RIC0_s0/dump_RIC0_s0"
FLOOR_DUMP = f"{ART}/v8f-dagoacc-t1/dump_DAGOACC_s0_m10_causal_heads"
HEADS_TICKS = f"{ART}/v8f-dagoacc-t1/ticks/ticks_DAGOACC_s0_m10_causal_heads.npz"
DK_CLIPS = ["126b3a97-0879-4ff3-9f4a-ea846f8a70ef", "b05195bf-36d7-4233-87bf-518708baa307", "f19cffbf-5621-4be5-83a4-7a096f6b7c39", "91e90a95-a129-486a-a1d4-cc1a2915793f"]
RC_CLIPS = ["126b3a97-0879-4ff3-9f4a-ea846f8a70ef", "af6f5964-2634-46e3-bbb1-ac2b194aec5f", "c8a39711-ce08-421f-b675-6533f89ad8a3", "e66979e0-6f5a-4e27-b497-c4e8a6d26534",
            "45a9f7ec-991f-4bf1-ae0e-41d9a233853e", "a0d95df6-499b-45ee-a424-5954f6e65f77"]
TOL_ADAPTER_M = 1e-9


# ------------------------------------------------------------------------------------------------ DK
def check_dk(mut=None):
    """Layer 1: lead_block on the banked clips' own cuboids reproduces the banked block rows and the reader's join dict, bit for bit, on all four states. Layer 2: the same cuboids as
    WORLD-frame scene tracks through the adapter give the same states and the same leads to 1e-9 m. Layer 3: DK-VIOL / DK-HW / DK-TG on the banked RIC0_s0 windows equal the reader's arrays."""
    r = S.run_sub("dk_pin", clips=DK_CLIPS, arr=RIC0_ARR, dump=RIC0_DUMP, floor_dump=FLOOR_DUMP, mut=mut)
    seen = {}
    for c, v in r["clips"].items():
        l1, l2 = v["layer1"], v["layer2"]
        bad = [k for k in ("state", "leads", "lead_lens", "speeds", "is_lead") if not l1[k]]
        assert not bad and l1["counts_equal"], f"DK layer 1: {c[:8]} differs from the banked block in {bad} (counts_equal {l1['counts_equal']})"
        for s, n in l1["counts"].items():
            seen[s] = seen.get(s, 0) + n
        assert l2["state_equal"], f"DK layer 2: {c[:8]} adapter changes a window state"
        assert l2["max_dev_leads_m"] <= TOL_ADAPTER_M and l2["max_dev_lens_m"] <= TOL_ADAPTER_M, f"DK layer 2: {c[:8]} lead track dev {l2['max_dev_leads_m']:.3g} m, length dev {l2['max_dev_lens_m']:.3g}"
        assert l2["max_dev_obs_xy_m"] <= TOL_ADAPTER_M, f"DK layer 2: {c[:8]} round-trip of the cuboids deviates by {l2['max_dev_obs_xy_m']:.3g} m"
    assert all(seen.get(s, 0) > 0 for s in ("LEAD", "NO_LEAD", "NOT_STRAIGHT", "NO_LABEL")), f"DK pin: not all four states exercised {seen}"
    for tag, tol in (("layer1", 0.0), ("layer2", TOL_ADAPTER_M)):
        for lab, v in r["layer3"][tag].items():
            if lab == "state_counts":
                continue
            assert v["n_finite"] >= 10, f"DK rows {tag} {lab}: only {v['n_finite']} finite windows"
            assert v["max_abs_dev"] <= tol, f"DK rows {tag} {lab}: deviates from the reader's banked array by {v['max_abs_dev']:.3g} (tol {tol})"
            if tag == "layer1":
                assert v["bit_identical"], f"DK rows layer 1 {lab}: not bit-identical"
    return {"clips": len(r["clips"]), "states": seen, "layer3_windows": r["layer3_n_windows"], "adapter_max_dev_m": max(v["layer2"]["max_dev_leads_m"] for v in r["clips"].values()),
            "layer3_layer1_bit_identical": {k: v["bit_identical"] for k, v in r["layer3"]["layer1"].items() if k != "state_counts"},
            "layer3_layer2_max_dev": {k: v["max_abs_dev"] for k, v in r["layer3"]["layer2"].items() if k != "state_counts"}}


# ------------------------------------------------------------------------------------------------ the GT of S-POS
def check_rc(mut=None):
    """The v9 builder's functions on a clip's EGO LOG reproduce the join's rc / rc_ok bit for bit, including clips whose route does not reach 30 / 50 / 80 m (rc_ok False: NaN)."""
    r = S.run_sub("rc_pin", clips=RC_CLIPS, mut=mut)
    bad = {c[:8]: v for c, v in r.items() if not (v["ok_equal"] and v["rc_bit_identical"])}
    assert not bad, f"rc pin: {bad}"
    nn = sum(v["n_not_ok_cells"] for v in r.values())
    assert nn > 0 and any(v["n_not_ok_cells"] == 0 for v in r.values()), "rc pin: the NaN path (and its complement) must both be exercised"
    return {"clips": len(r), "steps": sum(v["n_steps"] for v in r.values()), "not_ok_cells": nn}


# ------------------------------------------------------------------------------------------------ S-POS
def check_spos(mut=None):
    """`spos_errors` (T2-record-shaped inputs) against the reader's `analyze_run` on the banked DAGOACC_s0 heads ticks (2399 windows): the per-window S and CTRV errors are bit-identical,
    and the pooled S-POS@30/50/80 delta, lo and hi equal the banked read's (DAGOACC_READ.json) to the 4 dp it publishes."""
    r = S.run_sub("spos_pin", ticks=HEADS_TICKS, mut=mut)
    keys = [k for k in r if k.endswith("bit_identical")] + ["gt_ok_equal"]
    bad = {k: r.get(k.replace("bit_identical", "max_dev")) for k in keys if not r[k] and k != "gt_ok_equal"}
    assert not bad and r["gt_ok_equal"], f"S-POS: per-window arrays differ from the reader: {bad}, gt_ok_equal {r['gt_ok_equal']}"
    assert r["n_windows"] == 2399
    for lab, v in r["pooled"].items():
        got = [round(v["delta"], 4), round(v["lo"], 4), round(v["hi"], 4)]
        assert got == v["banked_B2000"], f"S-POS pooled {lab} {got} != banked {v['banked_B2000']}"
        assert v["n_windows"] == v["reader_n_windows"]
    return {"windows": r["n_windows"], "pooled": {k: (v["delta"], v["lo"], v["hi"], v["n_windows"]) for k, v in r["pooled"].items()}, "n_scored": r["info"]["n_scored"],
            "off_route_cells": r["info"]["off_route_cells"]}


# ------------------------------------------------------------------------------------------------ the bootstrap over scenes
def check_x12(mut=None):
    r = S.run_sub("x12", mut=mut)
    for k in ("X1", "X2"):
        assert r[k]["n_episodes"] == 42, f"{k}: the bootstrap clusters {r[k]['n_episodes']} units; the launch scenes are 42 (eid = scene)"
        assert r[k]["n_windows"] == 84, f"{k}: {r[k]['n_windows']} windows; 42 launch scenes x 2 starts = 84 (the mask selects launch scenes)"
        ref = r["ref_" + k]
        assert all(r[k][q] == ref[q] for q in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")), f"{k} differs from the reader's own paired(): {r[k]} vs {ref}"
    assert abs(r["X1"]["delta"] - r["true_X1"]) < 0.005 and abs(r["X2"]["delta"] - r["true_X2"]) < 0.005, "the planted effects are not recovered"
    assert r["X1"]["lo"] < r["X1"]["delta"] < r["X1"]["hi"] and r["X1"]["separated"] and r["X2"]["separated"]
    assert r["deterministic"], "the bootstrap is not deterministic (seed 0)"
    assert r["guard_refuses_misaligned"], "arms with different (scene, f0) lists were contrasted"
    return {"X1": (r["X1"]["delta"], r["X1"]["lo"], r["X1"]["hi"]), "X2": (r["X2"]["delta"], r["X2"]["lo"], r["X2"]["hi"]), "n_scenes": r["X1"]["n_episodes"]}


# ------------------------------------------------------------------------------------------------ outcome(): card section 6
H = S.H_SEED
c = lambda d, lo, hi: {"delta": d, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0)}            # noqa: E731
NEG = c(-0.04, -0.06, -0.02)                       # X2 separated-better
NOSEP = c(-0.01, -0.05, 0.03)                      # X2 not separated
POS = c(0.04, 0.02, 0.06)                          # X2 separated, the wrong sign
BSEP, BNOT = c(0.05, 0.02, 0.08), c(0.01, -0.02, 0.04)
# name -> (x1[0], x1[1], x2[0], x2[1], bl or None, expected outcome, expected branch)
CASES = {
    "closes": (c(0.0, -0.02, 0.02), c(0.005, -0.02, 0.025), NEG, NEG, None, "CRAWL-CLOSES", None),
    "closes_negative_x1": (c(-0.05, -0.08, -0.02), c(-0.04, -0.07, -0.01), NEG, NEG, None, "CRAWL-CLOSES", None),
    "closes_hi_equals_h_at_s0": (c(0.0, -0.02, H[0]), c(0.005, -0.02, 0.025), NEG, NEG, None, "MIXED", None),
    "closes_x2_unseparated_at_s1": (c(0.0, -0.02, 0.02), c(0.005, -0.02, 0.025), NEG, NOSEP, None, "MIXED", None),
    "closes_x2_wrong_sign_at_s0": (c(0.0, -0.02, 0.02), c(0.005, -0.02, 0.025), POS, NEG, None, "MIXED", None),
    "persists_b": (c(0.06, 0.03, 0.09), c(0.07, 0.04, 0.10), NOSEP, NOSEP, (BSEP, BSEP), "CRAWL-PERSISTS", "b"),
    "persists_a": (c(0.06, 0.03, 0.09), c(0.07, 0.04, 0.10), NOSEP, NOSEP, (BNOT, BNOT), "CRAWL-PERSISTS", "a"),
    "persists_blind_split": (c(0.06, 0.03, 0.09), c(0.07, 0.04, 0.10), NOSEP, NOSEP, (BSEP, BNOT), "CRAWL-PERSISTS", None),
    "persists_no_blind": (c(0.06, 0.03, 0.09), c(0.07, 0.04, 0.10), NOSEP, NOSEP, None, "CRAWL-PERSISTS", None),
    "persists_delta_equals_h": (c(H[0], 0.01, 0.04), c(H[1], 0.01, 0.05), NOSEP, NOSEP, (BSEP, BSEP), "CRAWL-PERSISTS", "b"),
    "persists_lo_equals_zero_at_s0": (c(0.06, 0.0, 0.09), c(0.07, 0.04, 0.10), NOSEP, NOSEP, None, "MIXED", None),
    "persists_fails_seed_gap": (c(0.03, 0.01, 0.05), c(0.12, 0.09, 0.15), NOSEP, NOSEP, None, "MIXED", None),
    "persists_gap_sign": (c(0.12, 0.09, 0.15), c(0.03, 0.01, 0.05), NOSEP, NOSEP, None, "MIXED", None),
    "mixed_separated_below_h": (c(0.02, 0.005, 0.04), c(0.025, 0.006, 0.05), NOSEP, NOSEP, None, "MIXED", None),
    "mixed_upper_bound_only": (c(0.0, -0.02, 0.02), c(0.0, -0.03, 0.025), NOSEP, NOSEP, None, "MIXED", None),
    "mixed_one_seed_closes_one_persists": (c(0.0, -0.02, 0.02), c(0.08, 0.05, 0.11), NEG, NOSEP, None, "MIXED", None),
    "underpowered": (c(0.02, -0.03, 0.05), c(0.03, -0.02, 0.06), NOSEP, NOSEP, None, "UNDERPOWERED", None),
    "underpowered_bounds_equal_h_and_zero": (c(0.01, 0.0, H[0]), c(0.01, 0.0, H[1]), NOSEP, NOSEP, None, "UNDERPOWERED", None),
    "mixed_one_seed_decides": (c(0.02, -0.03, 0.05), c(0.0, -0.02, 0.02), NOSEP, NOSEP, None, "MIXED", None),
}


def _run(fn, name, gates_ok=True):
    x1a, x1b, x2a, x2b, bl, want, br = CASES[name]
    out = fn({0: x1a, 1: x1b}, {0: x2a, 1: x2b}, None if bl is None else {0: bl[0], 1: bl[1]}, gates_ok=gates_ok)
    return out, want, br


def _ref(x1, x2, bl, h):
    """An independent restatement of section 6 as per-seed predicates (used only for the exhaustive comparison)."""
    gap = abs(x1[1]["delta"] - x1[0]["delta"])
    A = [x1[s]["hi"] < h[s] for s in (0, 1)]
    B = [x1[s]["lo"] > 0 for s in (0, 1)]
    X2 = [bool(x2[s]["separated"]) and x2[s]["delta"] < 0 for s in (0, 1)]
    if all(A[s] and X2[s] for s in (0, 1)):
        return "CRAWL-CLOSES", None
    if all(B[s] and x1[s]["delta"] >= h[s] and x1[s]["delta"] > gap for s in (0, 1)):
        if bl is None:
            return "CRAWL-PERSISTS", None
        sep = [bool(bl[s]["separated"]) for s in (0, 1)]
        return "CRAWL-PERSISTS", ("b" if all(sep) else "a" if not any(sep) else None)
    return ("MIXED" if (any(A) or any(B)) else "UNDERPOWERED"), None


def check_outcome(mut=None, fn=None):
    """Every branch of section 6 on synthetic contrasts (named cases incl. boundaries), the gate-fail branch, input validation, and an exhaustive comparison with a restatement."""
    fn = fn or S.outcome
    for name in CASES:
        out, want, br = _run(fn, name)
        assert out["outcome"] == want and out["branch"] == br, f"outcome[{name}]: got {out['outcome']}/{out['branch']}, card says {want}/{br}"
    out, _, _ = _run(fn, "persists_blind_split")
    assert out["notes"], "a BLIND split across seeds must leave a note (the card names no sub-branch)"
    g, _, _ = _run(fn, "persists_b", gates_ok=False)
    assert g["outcome"] == "GATE-FAIL" and g["branch"] is None, "section 6 item 5: a failed gate reads no contrast"
    nan = {0: c(float("nan"), -1.0, 1.0), 1: CASES["closes"][1]}
    assert fn(nan, {0: NEG, 1: NEG}, None, gates_ok=False)["outcome"] == "GATE-FAIL", "a failed gate must not even look at the contrasts"
    for bad in (nan, {0: {"delta": 0.0, "lo": -1.0, "hi": 1.0}, 1: CASES["closes"][1]}):
        try:
            fn(bad, {0: NEG, 1: NEG}, None)
        except ValueError:
            continue
        raise AssertionError("an unreadable contrast (NaN or a missing key) was read instead of refused")
    pool = [c(d, lo, hi) for d, lo, hi in [(0.0, -0.02, 0.02), (0.0267, 0.01, 0.0267), (0.03, 0.001, 0.05), (0.02, 0.005, 0.04), (0.08, 0.05, 0.11), (-0.05, -0.08, -0.02),
                                         (0.02, -0.03, 0.05), (0.0298, 0.0, 0.0298), (0.0, -0.02, 0.0267)]]
    n = 0
    for a, b in itertools.product(pool, pool):
        for x2a, x2b in itertools.product((NEG, NOSEP, POS), (NEG, NOSEP)):
            for bl in (None, (BSEP, BSEP), (BNOT, BNOT), (BSEP, BNOT)):
                x1, x2, blc = {0: a, 1: b}, {0: x2a, 1: x2b}, None if bl is None else {0: bl[0], 1: bl[1]}
                got = fn(x1, x2, blc)
                want = _ref(x1, x2, blc, H)
                assert (got["outcome"], got["branch"]) == want, f"outcome differs from the card restatement on {x1} {x2} {bl}: {got['outcome']}/{got['branch']} vs {want}"
                n += 1
    return {"named_cases": len(CASES), "exhaustive_combinations": n}


def mutate_outcome(old, new):
    """The `outcome` function with ONE source edit (a deliberate regression); the edit must apply exactly once."""
    src = inspect.getsource(S.outcome)
    assert src.count(old) == 1, f"mutation target not unique/absent: {old!r}"
    ns = {"math": math, "H_SEED": S.H_SEED, "_SEEDS": S._SEEDS, "_need": S._need}
    exec(compile(src.replace(old, new), "<mutant outcome>", "exec"), ns)
    return ns["outcome"]


OUTCOME_MUTATIONS = [
    ("CLOSES", 'x1[s]["hi"] < h[s] and x2[s]["separated"]', 'x1[s]["hi"] <= h[s] and x2[s]["separated"]'),
    ("CLOSES", ' and x2[s]["separated"] and x2[s]["delta"] < 0 for s in _SEEDS)\n    persists', ' and x2[s]["delta"] < 0 for s in _SEEDS)\n    persists'),
    ("CLOSES", 'x2[s]["delta"] < 0 for s in _SEEDS)\n    persists', 'x2[s]["delta"] > 0 for s in _SEEDS)\n    persists'),
    ("CLOSES", 'closes = all(x1[s]["hi"]', 'closes = any(x1[s]["hi"]'),
    ("PERSISTS", 'x1[s]["delta"] >= h[s]', 'x1[s]["delta"] > h[s]'),
    ("PERSISTS", ' and x1[s]["delta"] > seed_gap', ''),
    ("PERSISTS", 'x1[s]["lo"] > 0 and x1[s]["delta"] >= h[s]', 'x1[s]["lo"] >= 0 and x1[s]["delta"] >= h[s]'),
    ("PERSISTS", 'seed_gap = abs(x1[1]["delta"] - x1[0]["delta"])', 'seed_gap = x1[1]["delta"] - x1[0]["delta"]'),
    ("PERSISTS sub-branch", 'branch = "b"', 'branch = "a"'),
    ("PERSISTS sub-branch", 'if all(sep):', 'if any(sep):'),
    ("PERSISTS sub-branch", 'elif not any(sep):\n                branch = "a"', 'elif not any(sep):\n                branch = "b"'),
    ("MIXED", ' or x1[s]["lo"] > 0 for s in _SEEDS)', ' for s in _SEEDS)'),
    ("MIXED", 'x1[s]["hi"] < h[s] or x1[s]["lo"] > 0 for s in _SEEDS)', 'x1[s]["lo"] > 0 for s in _SEEDS)'),
    ("MIXED", 'decides = any(', 'decides = all('),
    ("UNDERPOWERED", 'x1[s]["hi"] < h[s] or x1[s]["lo"] > 0 for s in _SEEDS)', 'x1[s]["hi"] <= h[s] or x1[s]["lo"] > 0 for s in _SEEDS)'),
    ("UNDERPOWERED", '"MIXED" if decides else "UNDERPOWERED"', '"MIXED"'),
    ("GATE-FAIL", 'if not gates_ok:', 'if False:'),
    ("GATE-FAIL", '"outcome": "GATE-FAIL"', '"outcome": "UNDERPOWERED"'),
    ("validation", '        _need(x1[s], f"x1[{s}]")\n', ''),
]


def test_outcome():
    print(check_outcome())


def test_outcome_mutations():
    missed = []
    for branch, old, new in OUTCOME_MUTATIONS:
        try:
            check_outcome(fn=mutate_outcome(old, new))
        except AssertionError:
            continue
        missed.append((branch, old))
    assert not missed, f"mutations NOT caught: {missed}"


def test_dk():
    print(check_dk())


def test_rc():
    print(check_rc())


def test_spos():
    print(check_spos())


def test_x12():
    print(check_x12())


# ------------------------------------------------------------------------------------------------ one deliberate regression per link, each must go RED
MUTATIONS = [("DK", check_dk, ["no_rotation", "ego_time_shift", "width_as_length", "stale", "ts_spacing"]),
             ("rc", check_rc, ["psi_deg", "x_shift"]),
             ("S-POS", check_spos, ["kap_sign", "gt_frame", "no_onroute", "tick0_in", "s_frame"]),
             ("bootstrap", check_x12, ["eid_window", "mask_ignored", "no_guard"])]


def test_mutations_heavy():
    missed = []
    for name, fn, muts in MUTATIONS:
        for m in muts:
            try:
                fn(mut=m)
            except AssertionError as e:
                print(f"RED   {name:10s} mutation {m:16s}: {str(e)[:110]}", flush=True)
                continue
            print(f"GREEN {name:10s} mutation {m:16s}: NOT CAUGHT", flush=True)
            missed.append((name, m))
    assert not missed, f"mutations NOT caught: {missed}"


# ------------------------------------------------------------------------------------------------ the whole path on real scenes (oracle records)
SCENES = ["F:/TanitAD-scratch/t2_scenes/9e17ce05-b221-40a7-8ded-08d3cb018147", "F:/TanitAD-scratch/t2_scenes/e848c843-d275-4f60-a194-61cc7f699124"]
SMOKE_F0S = [20, 40, 60, 80, 100, 120]


def check_records(mut=None):
    """Real downloaded scenes (rig + sequence_tracks) -> oracle rollout records (the car executes the logged track; S emits the v9 route checkpoints of its tick) -> `rows_for_records`.
    Every row that has a perfect value must read it (errors 0, TAC 1, DK gaps 0); S-POS must be finite on some windows, NaN where the route is shorter than the arc (counted), and <= 0
    (an oracle S beats CTRV); the DK join must carry LEAD windows (the scene with a lead) and the row set must be the 19 of the reader."""
    r = S.run_sub("records_smoke", scenes=SCENES, f0s=SMOKE_F0S, mut=mut)
    assert not r["missing"], f"rows not computed: {r['missing']}"
    assert len(r["rows_present"]) == 19, f"{len(r['rows_present'])} rows, expected the reader's 19"
    for k, v in r["dev_from_perfect"].items():
        assert v is not None, f"{k}: no finite window"
        assert v <= 1e-6, f"{k}: the oracle reads {v:.3g} from its perfect value"
    for k, v in r["spos"].items():
        assert v["n_finite"] > 0, f"{k}: no window scored"
        assert v["max"] <= 1e-3, f"{k}: an oracle S reads worse than CTRV by {v['max']:.3g} m"
    n = r["n_valid"]
    assert n["S-POS@80"] <= n["S-POS@50"] <= n["S-POS@30"] <= r["n_records"], f"S-POS counts must shrink with the arc (the route does not reach 80 m from every start): {n}"
    assert r["state_counts"]["LEAD"] > 0, f"no LEAD window in the join: {r['state_counts']}"
    return {"records": r["n_records"], "n_valid": n, "states": r["state_counts"], "spos_max": {k: v["max"] for k, v in r["spos"].items()}, "dev": r["dev_from_perfect"]}


def test_records():
    print(check_records())


def test_records_mutations():
    missed = []
    for m in ("ckpt_swapped_xy", "exec_offset"):
        try:
            check_records(mut=m)
        except AssertionError as e:
            print(f"RED   records    mutation {m:16s}: {str(e)[:110]}", flush=True)
            continue
        missed.append(m)
    assert not missed, f"mutations NOT caught: {missed}"
