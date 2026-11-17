"""test_c120_read.py -- tests of the CKPT120 read (c120_lib / c120_run / resolve_reach / c120_read).  CPU only.  Run one job at a time (wait_commit first); TMP / basetemp on F:.
Groups: PURE labels (every committed outcome reached by synthetic rows, one mutation per rule that must flip a planted case); K-RE against the BANKED STRAT1 results; the c120_run flags-off smoke on ONE banked comparison checkpoint
(a code check, never a result); the end-to-end reader smoke on synthetic far files + banked dumps under CK120 names; the bank / reach refusals; the banned-word check of the rendered markdown.
"""
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import c120_lib as C  # noqa: E402
import c120_read as R  # noqa: E402
import c120_run as RUN  # noqa: E402

SL = C.SL


# ---------------------------------------------------------------------------------------------------------------------------- helpers
def row(delta, sep=True, n=300, ne=27, status="OK"):
    if status != "OK":
        return {"status": status, "n_windows": 5, "n_episodes": 2}
    return {"status": "OK", "delta": delta, "lo": delta - 0.1 if sep else -0.05, "hi": delta + 0.1 if sep else 0.05, "separated": sep, "n_windows": n, "n_episodes": ne}


def blk(rd=0.3, bal=0.4, raw=0.2, rd_sep=True, bal_sep=True, raw_sep=False, raw_status="OK"):
    return {"real_minus_donor": row(rd, rd_sep), "balanced_minus_half": row(bal, bal_sep), "raw_real_minus_best_constant": row(raw, raw_sep, status=raw_status)}


FL = {"rd": 0.05, "bal": 0.05, "raw": 0.05}


def lab(per, floors=FL):
    return R.route_label(per, floors)[0]


# ---------------------------------------------------------------------------------------------------------------------------- route label: every outcome
def test_route_follows_and_strong():
    assert lab({0: blk(), 1: blk()}) == "S120-FOLLOWS"
    assert lab({0: blk(raw_sep=True), 1: blk(raw_sep=True)}) == "S120-FOLLOWS-STRONG"
    assert lab({0: blk(raw_sep=True), 1: blk(raw_sep=False)}) == "S120-FOLLOWS"          # STRONG needs both seeds


def test_route_partial_each_failing_condition():
    assert lab({0: blk(bal_sep=False), 1: blk()}) == "S120-PARTIAL"
    assert lab({0: blk(raw=-0.1), 1: blk()}) == "S120-PARTIAL"
    assert lab({0: blk(bal=-0.2), 1: blk()}) == "S120-PARTIAL"                          # balanced separated but BELOW 0.5 is not following


def test_route_raw_point_must_clear_f120_mm_ruling():
    """MM ruling 2026-11-17: the card's 'every effect also > F120' binds the raw point too (a raw point above 0 but not above F120 is PARTIAL)."""
    assert lab({0: blk(raw=0.03), 1: blk()}) == "S120-PARTIAL"
    assert lab({0: blk(raw=0.06), 1: blk()}) == "S120-FOLLOWS"
    assert lab({0: blk(raw=0.03), 1: blk()}, {"rd": 0.05, "bal": 0.05}) == "S120-FOLLOWS (FLOOR-PENDING)"    # a missing raw floor is pending, never a pass of the rule


def test_route_ignores():
    assert lab({0: blk(rd_sep=False), 1: blk()}) == "S120-IGNORES"
    assert lab({0: blk(rd=-0.3), 1: blk()}) == "S120-IGNORES"                            # separated on the wrong side
    assert lab({0: blk(rd=0.04), 1: blk()}) == "S120-IGNORES"                            # separated but not above F120


def test_route_not_yet_not_read_and_floor_pending():
    assert lab({0: blk()}) == "NOT-YET"
    assert lab({0: blk(raw_status="n/a"), 1: blk()}) == "NOT-READ"
    assert lab({0: blk(), 1: blk()}, None) == "S120-FOLLOWS (FLOOR-PENDING)"
    assert lab({0: blk(rd=0.04), 1: blk()}, None) == "S120-FOLLOWS (FLOOR-PENDING)"      # with no floor the floor test is skipped and the label says so


def test_route_mutations_flip_planted_cases(monkeypatch):
    """one mutation per rule: the planted case must change its label"""
    floor_case, sign_case = {0: blk(rd=0.04), 1: blk()}, {0: blk(rd=-0.3), 1: blk()}
    assert lab(floor_case) == "S120-IGNORES" and lab(sign_case) == "S120-IGNORES"
    monkeypatch.setattr(R, "_ok", lambda r, f, need_pos=True: bool(r and r.get("status") == "OK" and r.get("separated") and (not need_pos or r["delta"] > 0)))       # drops the floor comparison
    assert lab(floor_case) != "S120-IGNORES"
    monkeypatch.setattr(R, "_ok", lambda r, f, need_pos=True: bool(r and r.get("status") == "OK" and r.get("separated") and (f is None or abs(r["delta"]) > f)))       # drops the sign test
    assert lab(sign_case) != "S120-IGNORES"
    monkeypatch.undo()
    raw_case = {0: blk(raw=-0.1), 1: blk()}
    assert lab(raw_case) == "S120-PARTIAL"
    bal_case = {0: blk(bal_sep=False), 1: blk()}
    assert lab(bal_case) == "S120-PARTIAL"                                               # balanced - 0.5 must separate; a read that dropped it would label this FOLLOWS (checked by the end-to-end mutation below)


def test_far_label():
    ok = row(-3.0)
    assert R.far_label({0: ok, 1: ok}, 1.0) == "FAR-SETS"
    assert R.far_label({0: ok, 1: row(-0.5)}, 1.0) == "FAR-NO-GAIN"                      # not above F120
    assert R.far_label({0: ok, 1: row(3.0)}, 1.0) == "FAR-NO-GAIN"                       # separated on the wrong side
    assert R.far_label({0: ok, 1: row(-3.0, sep=False)}, 1.0) == "FAR-NO-GAIN"
    assert R.far_label({0: ok}, 1.0) == "NOT-YET"
    assert R.far_label({0: ok, 1: ok}, None) == "FAR-SETS (FLOOR-PENDING)"


def cellset(cost=(), gain=()):
    return {**{c: {"cost": True, "gain": False} for c in cost}, **{c: {"cost": False, "gain": True} for c in gain}, "x:none": {"cost": False, "gain": False}}


def test_driving_label_every_row():
    assert R.driving_label(cellset(), cellset())[0] == ["KEEPS"]
    assert R.driving_label(cellset(gain=["a"]), cellset(gain=["a"]))[0] == ["ADDS"]
    assert R.driving_label(cellset(gain=["a"]), cellset(gain=["b"]))[0] == ["KEEPS"]     # a gain must be on the SAME cell at both seeds
    assert R.driving_label(cellset(cost=["a"]), cellset(cost=["a"]))[0] == ["COSTS-FAR"]
    assert R.driving_label(cellset(cost=["a"]), cellset())[0] == ["SEEDS-DISAGREE"]
    labs = R.driving_label(cellset(cost=["a", "b"]), cellset(cost=["a"]))[0]
    assert labs == ["COSTS-FAR", "SEEDS-DISAGREE"]                                       # both print
    assert R.driving_label(None, cellset())[0] == ["NOT-YET"]
    assert R.route_cost_cells({0: cellset(cost=["pos_err_m@80"]), 1: cellset(cost=["pos_err_m@80"])}) == ["pos_err_m@80"]
    assert R.route_cost_cells({0: cellset(cost=["pos_err_m@80"]), 1: cellset()}) == []
    assert R.detect_cost_labels({"weather": {"cost": True}, "surface": {"cost": True}}, {"weather": {"cost": True}, "surface": {"cost": False}}, {"weather": True, "surface": True}) == ["weather"]
    assert R.detect_cost_labels({"weather": {"cost": True}}, {"weather": {"cost": True}}, {"weather": False}) == []


def test_driving_mutation_cost_needs_both_seeds(monkeypatch):
    base = R.driving_label(cellset(cost=["a"]), cellset())[0]
    assert base == ["SEEDS-DISAGREE"]
    monkeypatch.setattr(R, "cost_set", lambda cells: set(cells))                          # a mutated cost rule counts every cell: the planted KEEPS case flips
    assert R.driving_label(cellset(), cellset())[0] != ["KEEPS"]


def test_floors_over_and_echo():
    a, b = {"c": np.array([1.0, 2.0])}, {"c": np.array([1.5, 2.5])}
    f, used = R.floors_over({"p": (a, b), "q": (None, b)}, ("c",))
    assert f["c"] == pytest.approx(0.5) and used == ["p"]
    f, used = R.floors_over({"q": (None, b)}, ("c",))
    assert f["c"] is None and used == []
    e = R.echo_share(np.array([1.0, 1.0, 0.0, np.nan]), np.array([True, False, True, True]))
    assert e["hits"] == 2 and e["hits_on_token_windows"] == 1 and e["share"] == 0.5 and "UNVERIFIED" in e["status"]


# ---------------------------------------------------------------------------------------------------------------------------- the cells of the tree + K-RE against banked STRAT1
@pytest.fixture(scope="module")
def world():
    CK = R.load_tree_modules()
    G, eid, ci, t, kl = R.ground120(CK)
    return CK, G, eid, ci, t, kl


def test_k_label_matches_the_card(world):
    CK, G, eid, ci, t, kl = world
    assert all(kl["identity_with_truth_npz"].values()), kl
    assert (kl["n_windows"], kl["n_episodes"], kl["left"], kl["right"]) == (341, 27, 79, 262)
    assert kl["matches_card"] is True


def test_k7_k8_identities_and_planted_follower_bars(world):
    CK, G, eid, ci, t, kl = world
    assert CK.k7(G["side"], G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"]) == {"hold": 0.0, "announced": 1.0, "mirrored": 0.0}
    import torch
    from tanitad.models.hjepa import donor_indices
    d = donor_indices(len(eid), episode_ids=torch.as_tensor(eid), seed=0).numpy()
    real, donor, exp = CK.k8(G["side"], d, G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"])
    assert real == 1.0 and abs(donor - exp) < 1e-12
    cp = CK.s_ckpt120(CK.planted(G["side"], "announced"), G["side"], G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"])
    bc, maj = CK.best_constant(cp, G["side"])
    assert maj == -1                                                                      # right (-1) is the majority: 262 of 341
    kp = CK.paired_mean(cp, bc, eid, 2000)
    assert (round(kp["delta"], 3), round(kp["lo"], 3), round(kp["hi"], 3), kp["separated"]) == (0.232, 0.090, 0.406, True)     # K-PERFECT, card section 7
    cn = CK.s_ckpt120(CK.planted(G["side"] * G["token_on"], "announced"), G["side"], G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"])
    bn, _ = CK.best_constant(cn, G["side"])
    kn = CK.paired_mean(cn, bn, eid, 2000)
    assert round(kn["delta"], 3) == -0.126 and not kn["separated"]                       # K-NAVINF raw


def test_s_ckpt120_at_80m_reproduces_the_audited_strat1_cell(world):
    """K-RE: the 120 m cell function at arc 80 equals strat1_strategic.route_cells (strategic_cells.ckpt_cell) bit for bit on a banked STRAT1 T0 pass"""
    CK, G, eid, ci, t, kl = world
    z = dict(np.load(C.banked_t0("S1_s0", "real"), allow_pickle=True))
    banked = R.SS.route_cells(z["ckpt"], G)[:, 2]
    mine = CK.s_ckpt120(np.stack([np.zeros(len(t)), np.zeros(len(t)), z["ckpt"][:, 2, 2]], -1).astype(np.float64), G["side"], G["theta"], G["d_text"], G["text_ok"], G["rc_ok"][:, 2], arc=80.0)
    assert np.array_equal(np.isfinite(banked), np.isfinite(mine)) and np.array_equal(banked[np.isfinite(banked)], mine[np.isfinite(mine)])


def test_k_re_banked_strat1_route_levels_reproduce(world):
    CK, G, eid, ci, t, kl = world
    CI = SL.load_ci()
    banked = json.load(open(SL.HERE + "/out/STRAT1_STRATEGIC.json", encoding="utf-8"))
    for tag in ("S1_s0", "S1_s1"):
        cells = {v: R.SS.route_cells(dict(np.load(C.banked_t0(tag, v), allow_pickle=True))["ckpt"], G) for v in R.VARIANTS}
        blk80 = R.SS.arm_block(CI, cells, G, eid, None)
        assert blk80["level"]["real"][2] == pytest.approx(banked["arms"][tag]["block"]["level"]["real"][2], abs=1e-12)
        assert blk80["real_minus_donor"][2]["delta"] == pytest.approx(banked["arms"][tag]["block"]["real_minus_donor"][2]["delta"], abs=1e-12)


def test_k9b_function_and_strata_partition(world):
    CK, G, eid, ci, t, kl = world
    a = {"ckpt": np.zeros((4, 3, 3)), "ckpt_far": np.zeros((4, 3))}
    assert CK.k9b(a, a)["pass"] is True
    b = {"ckpt": np.zeros((4, 3, 3)) + 2e-6, "ckpt_far": np.zeros((4, 3))}
    assert CK.k9b(a, b)["pass"] is False
    assert CK.k9b(a, {"ckpt": a["ckpt"]})["pass"] is False                              # a key on one side only fails
    st = CK.strata(G["token_on"], G["d_text"], G["v_now"])
    cp = CK.s_ckpt120(CK.planted(G["side"], "announced"), G["side"], G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"])
    ok = np.isfinite(cp)
    assert (st["token_reaches_S"] & ok).sum() == 219 and (st["token_does_not"] & ok).sum() == 122          # card 2.2
    assert (st["near_d<=60"] & ok).sum() == 244 and (st["far_d_60_100"] & ok).sum() == 97
    assert (st["t_6_30_s"] & ok).sum() == 304 and (st["t_above_30_s"] & ok).sum() == 37


# ---------------------------------------------------------------------------------------------------------------------------- refusals
def test_bank_and_reach_refusals(tmp_path):
    assert "checkpoint missing" in C.bank_refusal("H1_s0", str(tmp_path))
    d = tmp_path / "H1_s0"
    d.mkdir()
    (d / "ckpt_last.pt").write_bytes(b"abc")
    assert "BANK_MANIFEST.json missing" in C.bank_refusal("H1_s0", str(tmp_path))
    man = {"tag": "H1_s0", "commit": C.COMMIT, "files": {"ckpt_last.pt": {"sha256": "0" * 64}}}
    (d / "BANK_MANIFEST.json").write_text(json.dumps(man))
    assert "differs from its bank manifest" in C.bank_refusal("H1_s0", str(tmp_path))
    man["files"]["ckpt_last.pt"]["sha256"] = C.sha256_file(str(d / "ckpt_last.pt"))
    man["commit"] = "deadbeef"
    (d / "BANK_MANIFEST.json").write_text(json.dumps(man))
    assert "code of record" in C.bank_refusal("H1_s0", str(tmp_path))
    man["commit"], man["tag"] = C.COMMIT, "H2_s0"
    (d / "BANK_MANIFEST.json").write_text(json.dumps(man))
    assert "names tag" in C.bank_refusal("H1_s0", str(tmp_path))
    man["tag"] = "H1_s0"
    (d / "BANK_MANIFEST.json").write_text(json.dumps(man))
    assert C.bank_refusal("H1_s0", str(tmp_path)) is None
    assert C.reach_verdict("H1_s0", str(tmp_path))[0] == "MISSING"
    (tmp_path / "H1_s0.json").write_text(json.dumps({"verdict": "PASS"}))
    assert C.reach_verdict("H1_s0", str(tmp_path))[0] == "PASS"
    (tmp_path / "H1_s0.json").write_text("{not json")
    assert C.reach_verdict("H1_s0", str(tmp_path))[0] == "MISSING"                       # unreadable is never a PASS
    assert RUN.arm_problem("nope") == "unknown arm nope"


def test_run_refuses_head_dead_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "bank_refusal", lambda arm, ckd=None, commit=C.COMMIT: None)
    monkeypatch.setattr(C, "reach_verdict", lambda arm, reach_dir=None: ("HEAD-DEAD", "p"))
    assert "HEAD-DEAD" in RUN.arm_problem("H1_s0")


def test_argv_templates_name_the_levers():
    a = RUN.t0_argv("H2_s0", "donor", "X", ckpt="K")
    assert a[1].endswith("v8f_mini_eval_far.py") and "--nav-shuffle" in a and "--nav-strat" in a and a[a.index("--far-out") + 1] == "on"
    assert a[a.index("--nav-strat-sha256") + 1] == C.PINS["navtext_nav_s.npz"] and a[a.index("--ckpt") + 1] == "K" and a[a.index("--tag") + 1] == "CK120-H2_s0_donor"
    z = RUN.t0_argv("H1_s0", "zero", "X", ckpt="K")
    assert "--nav-withheld" in z and "--nav-strat" not in z and "--nav-shuffle" not in z
    r = RUN.t0_argv("H1_s0", "real", "X", ckpt="K")
    assert "--nav-withheld" not in r and "--nav-shuffle" not in r
    t, od, tag = RUN.t1_argv("H1_s0", "k9b", ckpt="K")
    assert "--limit" in t and t[t.index("--limit") + 1] == "logged" and t[t.index("--batch") + 1] == "64" and tag == "CK120K9B-H1_s0" and "--dump-nav" in t
    t, od, tag = RUN.t1_argv("H2_s1", "causal", ckpt="K")
    assert t[t.index("--limit") + 1] != "logged" and "--nav-strat" in t and tag == "CK120-H2_s1_m10_causal_heads"


def test_resolve_reach_refuses_unbanked_arm(tmp_path):
    import resolve_reach as RR
    v, p = RR.resolve("H1_s0", ckd=str(tmp_path), reach_dir=str(tmp_path / "reach"))
    assert v == "INCONCLUSIVE" and "REFUSED" in json.load(open(p))["reasons"][0]


def test_reach_controls_on_banked_checkpoint():
    """K-REACH-e control: a trained head reads PASS, a deleted-state head and a zero-state head read HEAD-DEAD (the gate can see a planted dead head)"""
    import resolve_reach as RR
    r = RR.controls(out=str(C.HERE + "/reach/CONTROLS_TEST.json"))
    assert r["trained_head_reads"] == "PASS" and r["planted_deleted_state"] == "HEAD-DEAD" and r["planted_zero_state"] == "HEAD-DEAD" and r["pass"]
    os.remove(C.HERE + "/reach/CONTROLS_TEST.json")


# ---------------------------------------------------------------------------------------------------------------------------- c120_run smoke: ONE banked checkpoint, flags off, a small window subset (a code check, never a result)
@pytest.fixture(scope="module")
def smoke_t0(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("c120smoke"))
    grid = f"{SL.INP}/grid_smoke.json"
    if not os.path.exists(grid):
        pytest.skip("inputs/grid_smoke.json missing")
    a = RUN.t0_argv("S1_s0", "real", out, grid=grid, ckpt=f"{SL.HERE}/ckpts/S1_s0/ckpt_last.pt", far=False)
    rc = RUN._run(a, out + "/log.txt")
    return rc, out, grid


def test_run_t0_flags_off_equals_banked_rows(smoke_t0):
    rc, out, grid = smoke_t0
    assert rc == 0, open(out + "/log.txt", encoding="utf-8").read()[-1500:]
    got = dict(np.load(f"{out}/eval_CK120-S1_s0_real.npz", allow_pickle=True))
    ref = dict(np.load(C.banked_t0("S1_s0", "real"), allow_pickle=True))
    pos = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(ref["win_ci"], ref["win_t"]))}
    idx = [pos[(int(c), int(t))] for c, t in zip(got["win_ci"], got["win_t"])]
    assert len(idx) >= 8
    for k in ("ckpt", "ctx_logits", "pred"):
        d = float(np.max(np.abs(got[k].astype(np.float64) - ref[k][idx].astype(np.float64))))
        # the 48-window subset batches differently from the 2,399-window pass: float32 BLAS noise (~1e-7), not bit identity
        assert d <= 1e-4, f"{k}: the CKPT120 tree's flags-off T0 pass differs from the banked STRAT1 pass by {d}"


def test_run_t1_flags_off_smoke_tick0_equals_banked(tmp_path, monkeypatch):
    """the T1 path of c120_run on ONE banked checkpoint, flags off, 6 windows: it runs, writes the tick dump, and tick 0 equals the banked S1_s0 tick 0 of the same windows (within float32 noise)"""
    grid = f"{SL.INP}/grid_smoke.json"
    monkeypatch.setattr(C, "T1", str(tmp_path / "t1"))
    a, od, tag = RUN.t1_argv("S1_s0", "causal", grid=grid, max_windows=6, ckpt=f"{SL.HERE}/ckpts/S1_s0/ckpt_last.pt", far=False)
    os.makedirs(od + "/ticks", exist_ok=True)
    rc = RUN._run(a, od + "/log.txt")
    assert rc == 0, open(od + "/log.txt", encoding="utf-8").read()[-1500:]
    got = dict(np.load(f"{od}/ticks/ticks_{tag}.npz", allow_pickle=True))
    ref = dict(np.load(f"{SL.HERE}/t1/ticks/ticks_STRAT1-S1_s0_m10_causal_heads.npz", allow_pickle=True))
    key = lambda z, i: (int(z["ci"][i]), int(z["t0"][i]), int(z["tick"][i]))  # noqa: E731
    pos = {key(ref, i): i for i in range(len(ref["tick"]))}
    sel = [i for i in range(len(got["tick"])) if int(got["tick"][i]) == 0]
    assert len(sel) >= 4 and all(key(got, i) in pos for i in sel)
    d = max(float(np.max(np.abs(got["s_ckpt"][i].astype(np.float64) - ref["s_ckpt"][pos[key(got, i)]].astype(np.float64)))) for i in sel)
    assert d <= 1e-4, d


# ---------------------------------------------------------------------------------------------------------------------------- end-to-end reader smoke
SPEC ={"H1_s0": "C0_s0", "H1_s1": "C0_s1", "H2_s0": "N1_s0", "H2_s1": "N1_s1", "H1_rep": "C0_rep", "R1_s0": "C0_s0", "R2_s0": "N1_s0"}      # banked dumps lent under CK120 names (different recipes: non-zero deltas)


def make_far_files(out, G, CK, eid):
    import torch
    from tanitad.models.hjepa import donor_indices
    os.makedirs(out + "/t0", exist_ok=True)
    d = donor_indices(len(eid), episode_ids=torch.as_tensor(eid), seed=0).numpy()
    rng = np.random.default_rng(0)
    ref = {v: dict(np.load(C.banked_t0("S1_s0", v), allow_pickle=True)) for v in R.VARIANTS}
    rc = G["rc120"]
    for arm in C.NEW_ARMS:
        for v in R.VARIANTS:
            side = G["side"].astype(float)
            s = {"real": side, "donor": side[d], "zero": np.zeros_like(side)}[v]
            if arm.startswith("R2"):
                s = s * G["token_on"]
            psi = 0.5 * s
            xy = np.array(rc[:, :2], copy=True)
            if arm == "R1_s0":
                psi = rng.choice([-0.5, 0.5, 0.0], size=len(side))
                xy = np.roll(xy, 500, axis=0)
            far = np.concatenate([xy, psi[:, None]], -1).astype(np.float32)
            far[~np.isfinite(far).all(1)] = 0.0
            np.savez(C.new_t0(arm, v), **ref[v])
            np.savez(C.new_t0_far(arm, v), win_ci=ref[v]["win_ci"], win_t=ref[v]["win_t"], ckpt_far=far, ckpt_far_now=far)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory, world):
    CK, G, eid, ci, t, kl = world
    global E2E_BASE
    base = E2E_BASE = str(tmp_path_factory.mktemp("c120e2e"))
    old = (C.OUT, C.T1)
    C.OUT, C.T1 = base + "/out", base + "/t1"
    make_far_files(C.OUT, G, CK, eid)
    reach = base + "/reach"
    os.makedirs(reach)
    for arm in C.NEW_ARMS:
        json.dump({"verdict": "PASS"}, open(f"{reach}/{arm}.json", "w"))
    lent = lambda arm: C.banked_t1_dump(SPEC[arm])  # noqa: E731
    old_dump = C.new_t1_dump
    C.new_t1_dump = lent
    try:
        R.main(["--out-dir", base + "/res", "--n-boot", "100", "--no-bank-check", "--reach-dir", reach])
    finally:
        C.new_t1_dump = old_dump
        C.OUT, C.T1 = old
    return json.load(open(base + "/res/RESULT_CKPT120.json", encoding="utf-8")), open(base + "/res/RESULT_CKPT120.md", encoding="utf-8").read()


E2E_BASE = None


def test_e2e_head_dead_arm_is_invalid_and_skipped(e2e, monkeypatch):
    """a K-REACH-e HEAD-DEAD arm is INVALID: nothing is read from it, its rung reads HEAD-DEAD, the other arms still read; a MISSING verdict is not a PASS either"""
    base = E2E_BASE
    monkeypatch.setattr(C, "OUT", base + "/out")
    monkeypatch.setattr(C, "T1", base + "/t1")
    reach = base + "/reach_dead"
    os.makedirs(reach, exist_ok=True)
    json.dump({"verdict": "HEAD-DEAD"}, open(f"{reach}/H1_s0.json", "w"))
    json.dump({"verdict": "PASS"}, open(f"{reach}/H1_s1.json", "w"))                      # H1_rep has NO file: MISSING
    import argparse
    a = argparse.Namespace(out_dir=base + "/res2", n_boot=50, smoke=False, smoke_episodes=141, no_bank_check=True, arms=["H1_s0", "H1_s1", "H1_rep"], ckpts=None, reach_dir=reach)
    monkeypatch.setattr(C, "new_t1_dump", lambda arm: C.banked_t1_dump(SPEC[arm]))
    rep = R.read(a)
    assert rep["arm_status"]["H1_s0"].startswith("HEAD-DEAD") and rep["arm_status"]["H1_rep"].startswith("NOT-READ") and rep["arm_status"]["H1_s1"].startswith("READ")
    assert "H1_s0" not in rep["route_120"] and "H1_s1" in rep["route_120"]
    assert rep["route_labels"]["H1"]["label"] == "HEAD-DEAD"
    assert rep["floors_120"]["available"] is False                                       # no H1_rep: FLOOR-PENDING, said so


def test_e2e_labels_reach_their_planted_outcomes(e2e):
    rep, md = e2e
    assert rep["route_labels"]["H1"]["label"] == "S120-FOLLOWS-STRONG"                   # the planted perfect follower
    assert rep["route_labels"]["H2"]["label"] == "S120-FOLLOWS-STRONG"
    assert rep["far_labels"]["H1"]["label"].startswith("FAR-SETS")                       # far xy = RC-A120 itself: closer than EXTRAP at both seeds
    assert rep["floors_120"]["available"] is True and rep["floors_120"]["rd"] == pytest.approx(0.0)       # H1_rep is the same planted follower
    assert rep["controls"]["R1_R2"]["R1_pass"] is True and rep["controls"]["R1_R2"]["R2_pass"] is True
    assert rep["PROVISIONAL"] == [] and rep["verdict_state"] in ("FINAL", "BROKEN READ")
    for k in ("H1_s0", "H2_s1", "R1_s0"):
        assert rep["arm_status"][k] == "READ"


def test_e2e_driving_and_families_present(e2e):
    rep, md = e2e
    assert set(rep["driving_labels"]) == {"H1", "H2"}
    for r in ("H1", "H2"):
        assert rep["driving_labels"][r]["labels"][0] in ("KEEPS", "ADDS", "COSTS-FAR", "SEEDS-DISAGREE")
    assert set(rep["families"]) >= {"LONGITUDINAL", "LATERAL", "TACTICAL", "driving_absent_items"}
    cells = rep["families"]["LONGITUDINAL"]["cells"]
    assert any(c.endswith("dk_hw_abs_err_m") for c in cells) and any(c.endswith("dk_tg_abs_err_s") for c in cells)         # headway / time-gap are in LONGITUDINAL
    assert any(c.endswith("curvature_mae_1pm") for c in rep["families"]["LATERAL"]["cells"]) and any(c.endswith("yaw_rate_mae_degps") for c in rep["families"]["LATERAL"]["cells"])
    assert rep["strategic_family"]["derived_goals"]["status"] == "not in this arm"
    assert rep["detected_info"].get("rows") and "H1_s0-S1_s0" in rep["detected_info"]["rows"]
    assert rep["controls"]["K1_driving_self_pair"]["pass"] and rep["controls"]["K7_pass"] and rep["controls"]["K8"]["pass"]
    assert rep["controls"]["same_estimator_ci_sha256"] is True


def test_e2e_markdown_has_no_banned_words_and_the_wording(e2e):
    rep, md = e2e
    low = md.lower()
    for w in R.BANNED:
        assert w not in low, w
    assert R.WORDING in md and "UNVERIFIED" in md and "Estimator" in md


def test_e2e_mutated_read_is_caught(e2e, monkeypatch):
    """a read whose label function ignores the balanced contrast would call a planted balanced-failing arm FOLLOWS; the real function does not"""
    per = {0: blk(bal_sep=False), 1: blk(bal_sep=False)}
    assert lab(per) == "S120-PARTIAL"
    real_ok = R._ok
    monkeypatch.setattr(R, "_ok", lambda r, f, need_pos=True: True if (r is per[0]["balanced_minus_half"] or r is per[1]["balanced_minus_half"]) else real_ok(r, f, need_pos))
    assert lab(per) == "S120-FOLLOWS"


def test_banned_word_sanitiser(e2e):
    rep, md = e2e
    rep2 = dict(rep, problems=["a correction was withdrawn and it is no longer a closed loop void"])
    out = R.render_md(rep2).lower()
    for w in R.BANNED:
        assert w not in out
