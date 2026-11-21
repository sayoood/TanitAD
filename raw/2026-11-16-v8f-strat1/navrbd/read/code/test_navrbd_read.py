"""Unit tests of the NAVRBD read on SYNTHETIC planted rows (no dumps, no model).  Run whole: python -m pytest -q -p no:cacheprovider test_navrbd_read.py   (CUDA_VISIBLE_DEVICES=-1)
Mutations (mutate_navrbd.py) must turn the named tests RED."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrbd_read as R  # noqa: E402
import strat1_lib as L  # noqa: E402

CI = L.load_ci()
F_HO = 0.042364016736401666


def eps(n_ep, per):
    return np.repeat(np.arange(n_ep), per)


def row(delta, lo, hi, bar=R.bar_of(F_HO)):
    return {"status": "OK", "delta": delta, "lo": lo, "hi": hi, "bar": bar, "gain": bool(delta >= bar and lo > 0), "cost": bool(delta <= -bar and hi < 0)}


GOOD = {s: row(0.06, 0.03, 0.09) for s in (0, 1)}
ZY = {0: True, 1: True}
CLEAN = {0: set(), 1: set()}
NOGAIN = {"gain": False}


# ------------------------------------------------------------------ floor, bar, label, pairing
def test_bar_is_the_larger_of_the_rig_floor_and_f_ho():
    assert R.bar_of(F_HO) == F_HO
    assert round(R.bar_of(F_HO), 4) == 0.0424
    assert R.bar_of(0.01) == 0.0383
    assert R.bar_of(F_HO, 0.05) == 0.05                    # a re-dumped larger F_HO becomes the bar


def test_f_ho_is_the_absolute_mean_difference_of_the_pair():
    a0, a1 = np.array([1, 0, 0, 1.0]), np.array([1, 1, 0, 1.0])
    assert R.f_ho_of(a0, a1) == 0.25 == R.f_ho_of(a1, a0)


def test_pairing_is_same_seed():
    assert R.pair_ref(0) == "N1R_s0" and R.pair_ref(1) == "N1R_s1"
    assert R.d_arm(1) == "N1D_s1"


def test_kinematic_label_accuracy_not_five_way():
    g = np.array([0, 1, 2, 2])
    p = np.array([0, 1, 1, 2])
    assert R.kin_acc(g, p).tolist() == [1.0, 1.0, 0.0, 1.0]
    assert set(R.CL) == {"lane_keep", "turn_left", "turn_right"} and len(R.CL) == 3
    with pytest.raises(AssertionError):                    # the 5-way v9 lat_cls is refused as a label vector
        R.check_three_class(np.array([0, 1, 4]))


def test_gain_needs_the_bar_and_a_positive_lower_bound():
    bar = R.bar_of(F_HO)
    a = np.ones(400)
    b = np.zeros(400)
    e = eps(40, 10)
    r = R.d1_ho(CI, a, b, e, bar, nb=200)
    assert r["gain"] and r["delta"] == 1.0
    assert not row(0.040, 0.01, 0.07)["gain"]               # between 0.0383 and the HO bar 0.0424: NOT a gain (the 0.0383 mutation turns it into one)
    assert row(0.0425, 0.01, 0.07)["gain"]
    assert not row(0.06, -0.01, 0.12)["gain"]               # lo not above 0
    assert row(-0.05, -0.08, -0.01)["cost"] and not row(-0.04, -0.08, -0.01)["cost"]


def test_d1_ho_gain_rule_on_planted_windows():
    e = eps(40, 10)
    b = np.zeros(400)
    a = np.zeros(400)
    for ep in range(40):
        a[ep * 10] = 1.0                                       # +0.10 in every episode (delta 0.10): a gain
    assert R.d1_ho(CI, a, b, e, 0.0424, nb=200)["gain"]
    a2 = np.zeros(400)
    for ep in range(40):
        a2[ep * 10] = 1.0 if ep < 16 else 0.0                  # delta 0.04 in 16 of 40 episodes: 0.04 between 0.0383 and 0.0424
    r = R.d1_ho(CI, a2, b, e, 0.0424, nb=200)
    assert abs(r["delta"] - 0.04) < 1e-12 and not r["gain"]
    a3 = np.zeros(400)
    a3[:60] = 1.0                                               # three episodes carry all of it: delta 0.15 but the interval reaches 0
    for ep in range(40):
        pass
    noisy_a = np.zeros(400)
    noisy_b = np.zeros(400)
    for ep in range(40):
        if ep < 20:
            noisy_a[ep * 10: ep * 10 + 6] = 1.0                # +0.6
        else:
            noisy_b[ep * 10: ep * 10 + 5] = 1.0                # -0.5
    rn = R.d1_ho(CI, noisy_a, noisy_b, e, 0.0424, nb=300)
    assert rn["delta"] >= 0.0424 and rn["lo"] <= 0 and not rn["gain"]


def test_d1_ho_uses_the_bar_it_is_given_and_reads_a_self_pair_as_zero():
    e = eps(20, 5)
    a = (np.arange(100) % 2).astype(float)
    r = R.d1_ho(CI, a, a, e, 0.0424, nb=100)
    assert r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and not r["gain"]
    assert R.k1_self(CI, a, e, nb=100)["pass"]
    assert not R.k1_self(CI, a, e, nb=100, ref=1.0 - a)["pass"]                               # a non-self pair is not a K1 pass


def test_k_const_and_k_floor():
    g = np.array([0] * 5 + [1] * 3 + [2] * 2)
    k = R.k_const(g)
    assert k["majority_class"] == "lane_keep" and k["constant_accuracy"] == 0.5 and k["pass"]
    assert not R.k_const(g, pred=np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))["pass"]            # a non-constant predictor is not the K-CONST identity
    assert R.k_floor(F_HO, F_HO)["pass"] and not R.k_floor(F_HO, F_HO + 1e-9)["pass"]


def test_z_eval_exact_zero_and_mask():
    real = {"c": np.array([1.0, 2.0, 3.0, 4.0]), "d": np.array([1.0, np.nan, 3.0, 4.0])}
    zero = {"c": np.array([1.0, 2.0, 9.0, 4.0]), "d": np.array([1.0, np.nan, 9.0, 4.0])}
    assert R.z_eval(real, zero, np.array([1, 1, 0, 1], bool), ["c", "d"])["pass"]
    assert not R.z_eval(real, zero, np.array([1, 1, 1, 1], bool), ["c", "d"])["pass"]           # a difference inside the mask
    assert not R.z_eval(real, zero, np.zeros(4, bool), ["c", "d"])["pass"]                       # an empty mask never passes
    nan_b = {"c": real["c"], "d": np.array([1.0, 5.0, 3.0, 4.0])}
    assert not R.z_eval(real, nan_b, np.array([1, 1, 0, 1], bool), ["c", "d"])["pass"]          # NaN vs finite


def test_d2_ho_rows():
    g = np.array([0, 0, 1, 1, 2, 2] * 10)
    pa = np.array([0, 1, 1, 0, 2, 1] * 10)
    pr = np.array([0, 0, 1, 1, 2, 2] * 10)
    r = R.d2_ho(CI, g, pa, pr, eps(12, 5), nb=100)
    assert r["turn_recall"]["arm"] == 0.75 and r["turn_recall"]["ref"] == 1.0
    assert r["false_turn_share"]["arm"] == 0.5 and r["false_turn_share"]["ref"] == 0.0
    assert r["side_given_turn_arm"]["n_windows"] == 30


# ------------------------------------------------------------------ the outcome table (card 7)
def out(d1=GOOD, zuse=ZY, guard=CLEAN, n1dx=NOGAIN, route=False, f=F_HO):
    return R.outcome(f, d1, zuse, guard, n1dx, route)


def test_outcome_gain_needs_every_condition():
    o = out()
    assert o["labels"] == ["RBD-GAIN"] and o["status"] == "FINAL"
    assert "RBD-GAIN" not in out(zuse={0: True, 1: False})["labels"]
    assert "RBD-GAIN" not in out(guard={0: {"LONG:x"}, 1: set()})["labels"]
    assert "RBD-GAIN" not in out(guard={0: set(), 1: {"LONG:x"}})["labels"]
    assert "RBD-GAIN" not in out(n1dx=None)["labels"]
    assert "RBD-ONE-SEED" not in out()["labels"]
    assert "RBD-GAIN" not in out(route=True)["labels"]
    assert out(n1dx={"gain": True})["labels"] == ["N1DX-REGRESSION-GAINS"]                      # the regression arm gains: no N1D label stands
    assert "RBD-GAIN" in out(n1dx={"gain": True})["labels_if_N1Dx_ignored"]


def test_outcome_gain_pending_strategic_without_contrast():
    o = out(route=None)
    assert "RBD-GAIN-PENDING-STRATEGIC" in o["labels"] and o["status"] == "NOT-YET"


def test_outcome_one_seed_not_a_gain():
    d1 = {0: row(0.06, 0.03, 0.09), 1: row(0.01, -0.02, 0.04)}
    o = out(d1=d1)
    assert "RBD-ONE-SEED" in o["labels"] and "RBD-GAIN" not in o["labels"]


def test_outcome_no_gain_unread_costs_floor_high():
    d1 = {0: row(0.01, -0.02, 0.04), 1: row(0.0, -0.03, 0.03)}
    o = out(d1=d1)
    assert "RBD-NO-GAIN" in o["labels"] and "RBD-FLOOR-HIGH" in o["labels"]                     # both labels print when both hold
    assert "RBD-UNREAD" in out(d1=d1, zuse={0: False, 1: False})["labels"] and "RBD-NO-GAIN" not in out(d1=d1, zuse={0: False, 1: False})["labels"]
    assert "RBD-NO-GAIN" not in out(d1=d1, zuse={0: True, 1: False})["labels"]
    assert "RBD-UNREAD" not in out(d1=d1, zuse={0: True, 1: False})["labels"]                  # unread at both seeds needed (reading r1)                  # needs Z-USE at both seeds
    cost = {0: row(-0.06, -0.09, -0.03), 1: row(-0.05, -0.08, -0.02)}
    assert "RBD-COSTS" in out(d1=cost)["labels"]
    assert "RBD-COSTS" in out(guard={0: {"LATERAL:a"}, 1: {"LATERAL:a"}})["labels"]
    assert "RBD-COSTS" not in out(guard={0: {"LATERAL:a"}, 1: {"LATERAL:b"}})["labels"]          # the same cell at both seeds
    assert "RBD-FLOOR-HIGH" not in out()["labels"]                                              # a gain clears the floor


def test_outcome_floor_high_needs_f_above_the_rig_floor():
    d1 = {0: row(0.0, -0.03, 0.03, 0.0383), 1: row(0.0, -0.03, 0.03, 0.0383)}
    assert "RBD-FLOOR-HIGH" not in R.outcome(0.03, d1, ZY, CLEAN, NOGAIN, False)["labels"]


def test_outcome_missing_input_is_not_yet_never_a_pass():
    o = R.outcome(F_HO, {0: GOOD[0], 1: None}, {0: True, 1: None}, {0: set(), 1: None}, None, None)
    assert o["status"] == "NOT-YET" and o["labels"] == ["NOT-YET"] and "D1-HO N1D_s1" in o["missing"]
    o2 = R.outcome(F_HO, GOOD, {0: None, 1: None}, CLEAN, NOGAIN, False)
    assert o2["status"] == "NOT-YET" and "RBD-GAIN" not in o2["labels"] and "Z-USE N1D_s0" in o2["missing"]
    assert "N1Dx_s0" in R.outcome(F_HO, GOOD, ZY, CLEAN, None, False)["missing"]


def test_outcome_uses_the_bar_from_f_ho_not_the_rig_floor():
    d = {s: dict(row(0.040, 0.01, 0.07, 0.0383), gain=True) for s in (0, 1)}                     # a planted row that cleared only 0.0383
    o = R.outcome(F_HO, {s: {**d[s], "gain": False} for s in (0, 1)}, ZY, CLEAN, NOGAIN, False)
    assert "RBD-GAIN" not in o["labels"] and o["detail"]["bar"] == F_HO


# ------------------------------------------------------------------ files, aliases, card
def test_alias_and_paths():
    al = R.parse_alias(["N1D_s0=N1R_s1"])
    assert R.ho_dir("N1D_s0", al).endswith("dump_NAVRBD-HO-N1R_s1_m10_causal_heads")
    assert R.ho_dir("N1D_s1").replace("\\", "/").endswith("/read/dumps/dump_NAVRBD-HO-N1D_s1_m10_causal_heads")
    assert R.ho_dir("N1D_s0Z").replace("\\", "/").endswith("/read/dumps/dump_NAVRBD-HO-N1D_s0_zero_m10_causal_heads")      # dump_rbd.sh names the zero variant TAG_zero
    assert R.ho_dir("N1R_s0").replace("\\", "/").endswith("/step0/t1/dump_NAVRBD-HO-N1R_s0_m10_causal_heads")
    assert R.e141_dir("N1D_s0Z").replace("\\", "/").endswith("/dumps141/dump_NAVRBD-N1D_s0Z_m10_causal_heads")


def test_card_sha_prefix_matches():
    ok, sha = R.card_check()
    assert ok, sha


def test_banked_f_ho_file_is_the_card_value():
    import json
    b = json.load(open(R.F_HO_BANKED_FILE, encoding="utf-8"))
    assert b["F_HO"] == F_HO and round(b["F_HO"], 4) == 0.0424


# ------------------------------------------------------------------ the eval-141 dump wrapper (pure parts)
def test_t1_argv_follow_and_zero_twin():
    import navrbd_dumps as DU
    assert DU.variant_of("NAVRBD-N1D_s0") == "follow" and DU.variant_of("NAVRBD-N1D_s0Z") == "zero"
    a = DU.extra_args("NAVRBD-N1D_s1Z")
    assert a[a.index("--nav-rb-feat") + 1] == "exitdist"                  # the 15-column feat is named in argv (a default would disable the channel read)
    assert a[a.index("--nav-rb-variant") + 1] == "zero"
    assert a[a.index("--nav-rb-sha256") + 1] == DU.RB_SHA256 and a[a.index("--nav-rb-npz") + 1].endswith("navtext_nav_rbd.npz")
    assert DU.extra_args("NAVRBD-N1D_s0")[DU.extra_args("NAVRBD-N1D_s0").index("--nav-rb-variant") + 1] == "follow"


def test_dump_refusals(tmp_path):
    import json as _j
    import navrbd_dumps as DU
    assert "missing" in DU.refusal_ck("N1D_s0", str(tmp_path))
    d = tmp_path / "N1D_s0"
    d.mkdir()
    (d / "ckpt_last.pt").write_bytes(b"abc")
    assert DU.refusal_ck("N1D_s0", str(tmp_path)) is None
    (d / "BANK_MANIFEST.json").write_text(_j.dumps({"files": {"ckpt_last.pt": {"sha256": "0" * 64}}}))
    assert "differs" in DU.refusal_ck("N1D_s0", str(tmp_path))
    assert "unknown" in DU.refusal_ck("N1R_s0", str(tmp_path))


def test_dump_wrapper_uses_the_rbd_script_and_refuses_a_wrong_rb_table(monkeypatch, tmp_path):
    import navrbd_dumps as DU
    src = open(DU.__file__, encoding="utf-8").read()
    assert "v8f_t1_eval_rbd.py" in src
    p = tmp_path / "t.npz"
    p.write_bytes(b"x")
    monkeypatch.setattr(DU, "RB_NPZ", str(p))
    with pytest.raises(RuntimeError):
        DU.rb_npz_checked()


def test_driver_reads_the_kinematic_label():
    src = open(R.__file__, encoding="utf-8").read()
    assert 'gl = check_three_class(geom.labels(ref["g"])[0])' in src and 'pred = {arm: check_three_class(geom.labels(x["cl"])[0])' in src
    assert 'ref["lat_cls"]' not in src
