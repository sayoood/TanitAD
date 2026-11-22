"""test_presroll_read.py -- card ADD120 (PRESROLL10K), reader kit (helper-written): the generated reader and driver against their generator, the tables against the card, the names on
hand-made payloads (every branch of section 4, and the agreement of the two name routines on random payloads), the controls e0, e1, e2 and e8 on tiny synthetic files (each clause
passing and failing), and the refusals. The source kit of card ADD117 is taken from ../../gapmargin/read (same relative places in the pod stage as far as the generator goes). No pod result
file is opened; every array here is a few rows."""
import argparse
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "make_presroll_reader.py")
READER = os.path.join(HERE, "read_presroll10k.py")
DRIVER = os.path.join(HERE, "podread_presroll10k.sh")
CARD_FILE = os.path.join(HERE, "..", "..", "PREREG_V8F_PROOF_ADD120.md")
BASH = shutil.which("bash")
sys.path.insert(0, HERE)
import read_presroll10k as RP  # noqa: E402

P = "PRESROLL10K-"
WP = "b2a1d5f7242c3700" + "0" * 48
WSRC = "28bc31f21ecb4ccb" + "1" * 48
LINE = ("arm %s form RR replan steps 10 rule every %s readout %s kappa 0.2 preroll 0 exec plan nav deadreckon grid eval_grid_windows.json model seed %d reader seed %d "
        "ckpt sha256 " + "2f" * 32 + " wrapper sha256 " + "b2" * 32)


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


# ============================================================================================ generator, tables
@pytest.mark.parametrize("kind,path", [("reader", READER), ("driver", DRIVER)])
def test_generated_file_is_what_the_generator_prints(kind, path):
    r = subprocess.run([sys.executable, GEN, "--print", kind], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.replace("\r\n", "\n") == rd(path)
    assert "\r" not in rd(path)


def test_tables():
    assert RP.CARD == "PREREG_V8F_PROOF_ADD120" and RP.PRE == "PRESROLL10K" and RP.SETTINGS == ("CONV", "K0") and RP.SEEDS == (0, 1)
    assert RP.ARMS == ("PO", "POA", "GEc", "A90", "PL", "RLc", "RR")
    assert RP.SPEC["CONV"]["runs"] == {"PO": "RQ50Hb20gGEcPO", "POA": "RQ50Hb20gA90PO", "GEc": "RQ50Hb20gGEc", "A90": "RQ50Hb20gA90", "PL": "RLcHb20pLE", "RLc": "RLcHb20", "RR": "RQ50Hb20"}
    assert RP.SPEC["K0"]["runs"] == {"PO": "RQ50KF5Hb20gGEcPO", "POA": "RQ50KF5Hb20gA90PO", "GEc": "RQ50KF5Hb20gGEc", "A90": "RQ50KF5Hb20gA90", "PL": "RLcKF5Hb20pLE", "RLc": "RLcKF5Hb20", "RR": "RQ50KF5Hb20"}
    assert RP.SPEC["CONV"]["every"] is None and RP.SPEC["CONV"]["readout"] == "pinned" and RP.SPEC["K0"]["every"] == 5 and RP.SPEC["K0"]["readout"] == "K"
    assert all(RP.SPEC[S]["P"] == 0 and RP.SPEC[S]["grid"] == "full" and RP.SPEC[S]["grid_file"] == "pinned" for S in RP.SETTINGS)
    assert RP.NEW_ARMS == {"CONV": ("PO", "POA"), "K0": ("PO", "POA")}
    assert sorted(RP.new_keys()) == sorted("%s_%s_s%d" % (S, a, m) for S in ("CONV", "K0") for a in ("PO", "POA") for m in (0, 1))
    assert RP.PAIRS == {"PO-VS-GEc": ("PO", "GEc"), "PO-VS-PL": ("PO", "PL"), "POA-VS-A90": ("POA", "A90"), "POA-VS-GEc": ("POA", "GEc"), "POA-VS-RLc": ("POA", "RLc"), "GEc-VS-RLc": ("GEc", "RLc")}
    assert RP.EXPECTED["CONV"] == {"GEc": (0.0016, 0.0569, 0.2329), "A90": (0.0008, 0.0463, 0.2539), "PL": (0.0008, 0.0489, 0.2399), "RLc": (0.0016, 0.0320, 0.2243), "RR": (0.0280, 0.0783, 0.2593)}
    assert RP.EXPECTED["K0"] == {"GEc": (0.0000, 0.0498, 0.1651), "A90": (0.0000, 0.0374, 0.1729), "PL": (0.0000, 0.0489, 0.1667), "RLc": (0.0000, 0.0365, 0.1550), "RR": (0.0296, 0.0738, 0.1682)}
    assert RP.ref_arm("CONV", "PO") == "GEc" and RP.ref_arm("K0", "PO") == "GEc" and RP.ref_arm("CONV", "POA") == "A90" and RP.ref_arm("K0", "POA") == "A90"
    assert RP.CLOSE_TOL == 0.0045 and RP.IDP_RUN == "RQ50Hb20gGEcIDP" and RP.IDP_SEED == 0 and RP.EXPECT_DUMPS == 141
    assert RP.WRAPPER_P_SHA == "b2a1d5f7242c3700" and RP.WRAPPER_P_SRC == "28bc31f21ecb4ccb" and RP.WRAPPER_SHA == "28bc31f21ecb4ccb" and RP.CHECK_P_LAST == "ZZ CHECK-WRAPPER-P PASS"
    assert RP.DONE_RE.match("ZZ PRESROLL10K DONE 2026-11-22T16:00:00Z all steps done") and RP.CONTROLS == tuple("e%d" % i for i in range(9))
    assert RP.GAP_ARM == {"PO": None, "POA": "A90"} and RP.MARGIN_FILES == {"A90": "VGAPLE_A90"}


@pytest.mark.skipif(not os.path.isfile(CARD_FILE), reason="the card file is not beside the kit on this machine")
def test_tables_against_the_card_text():
    card = rd(CARD_FILE)
    for S in RP.SETTINGS:
        for arm, nm in RP.SPEC[S]["runs"].items():
            if arm in ("PO", "POA"):
                assert "`I10%s`" % nm in card, nm
    assert "`I10RQ50Hb20gGEcIDP_s0`" in card and "I10RQ50Hb20gGEc_s0" in card
    for S in RP.SETTINGS:
        for arm, v in RP.EXPECTED[S].items():
            assert "%s %.4f / %.4f / %.4f" % (arm, v[0], v[1], v[2]) in card, (S, arm)
    for w in ("0.0045", "b2a1d5f7242c3700", "28bc31f21ecb4ccb", "141 dump files", "2,000 draws"):
        assert w in card, w
    for pair in ("PO − GEc", "PO − PL", "POA − A90", "POA − GEc", "POA − RLc", "GEc − RLc"):
        assert pair in card, pair


# ============================================================================================ names
def payload(ivs=None, u=None):
    """ivs {(S, pair, measure): (lo, hi)}; u {(S, arm): pooled unsafe level}; default: nothing separated, every arm at 0.05, the label form at 0.0320"""
    ivs, u = ivs or {}, u or {}
    st = {}
    for S in RP.SETTINGS:
        pairs = {pn: {mk: {"interval": list(ivs.get((S, pn, mk), (-0.01, 0.01)))} for mk in "UCF"} for pn in RP.PAIRS}
        levels = {arm: {"C": 0.0, "U": u.get((S, arm), 0.0320 if arm == "RLc" else 0.05), "F": 0.2} for arm in RP.ARMS}
        st[S] = {"pairs": pairs, "levels": levels}
    return {"settings": st}


def test_names_when_nothing_is_separated():
    nm = RP.all_names(payload())
    assert len(nm) == 16
    for S in RP.SETTINGS:
        assert nm[S + "_OVERALL"] == P + S + "-BOTH-LEVERS-DO-NOT-REACH-LABEL-FORM"
        assert nm[S + "_POA_LABEL"] == P + S + "-BOTH-LEVERS-UNSAFE-AT-LABEL-FORM"
        for mw in ("UNSAFE", "CONTACT", "FALLING-BEHIND"):
            assert nm["%s_PO_%s" % (S, mw)] == P + "%s-PRESENCE-%s-NOT-SEPARATED" % (S, mw)
            assert nm["%s_POA_%s" % (S, mw)] == P + "%s-BOTH-LEVERS-%s-NOT-SEPARATED" % (S, mw)


def test_names_presence_and_both_levers_per_measure():
    iv = {("CONV", "PO-VS-GEc", "U"): (-0.03, -0.002), ("CONV", "PO-VS-GEc", "C"): (0.001, 0.01), ("CONV", "PO-VS-GEc", "F"): (0.004, 0.03),
          ("K0", "POA-VS-GEc", "U"): (0.001, 0.02), ("K0", "POA-VS-GEc", "C"): (-0.02, -0.001), ("K0", "POA-VS-GEc", "F"): (-0.02, 0.0)}
    nm = RP.all_names(payload(iv))
    assert nm["CONV_PO_UNSAFE"] == P + "CONV-PRESENCE-LOWERS-UNSAFE" and nm["CONV_PO_CONTACT"] == P + "CONV-PRESENCE-RAISES-CONTACT" and nm["CONV_PO_FALLING-BEHIND"] == P + "CONV-PRESENCE-RAISES-FALLING-BEHIND"
    assert nm["K0_POA_UNSAFE"] == P + "K0-BOTH-LEVERS-RAISE-UNSAFE" and nm["K0_POA_CONTACT"] == P + "K0-BOTH-LEVERS-LOWER-CONTACT" and nm["K0_POA_FALLING-BEHIND"] == P + "K0-BOTH-LEVERS-FALLING-BEHIND-NOT-SEPARATED"
    assert nm["CONV_POA_UNSAFE"] == P + "CONV-BOTH-LEVERS-UNSAFE-NOT-SEPARATED" and nm["K0_PO_UNSAFE"] == P + "K0-PRESENCE-UNSAFE-NOT-SEPARATED"


def test_names_label_form_against_poa():
    nm = RP.all_names(payload({("CONV", "POA-VS-RLc", "U"): (0.002, 0.02), ("K0", "POA-VS-RLc", "U"): (-0.02, -0.002)}))
    assert nm["CONV_POA_LABEL"] == P + "CONV-BOTH-LEVERS-UNSAFE-ABOVE-LABEL-FORM" and nm["K0_POA_LABEL"] == P + "K0-BOTH-LEVERS-UNSAFE-BELOW-LABEL-FORM"
    assert RP.all_names(payload({("CONV", "POA-VS-RLc", "U"): (-0.02, 0.0)}))["CONV_POA_LABEL"] == P + "CONV-BOTH-LEVERS-UNSAFE-AT-LABEL-FORM"
    assert RP.all_names(payload({("CONV", "POA-VS-RLc", "U"): (0.0, 0.02)}))["CONV_POA_LABEL"] == P + "CONV-BOTH-LEVERS-UNSAFE-AT-LABEL-FORM"


def test_names_reach_branches():
    low = {("CONV", "POA-VS-GEc", "U"): (-0.03, -0.002)}
    assert RP.all_names(payload(low, {("CONV", "POA"): 0.0350}))["CONV_OVERALL"] == P + "CONV-BOTH-LEVERS-REACH-LABEL-FORM-AT-NO-SEPARATED-PRICE"
    assert RP.reaches(payload(low, {("CONV", "POA"): 0.0350}), "CONV") is True and RP.reaches(payload(low, {("CONV", "POA"): 0.0350}), "K0") is False
    pay = {**low, ("CONV", "POA-VS-RLc", "F"): (0.002, 0.02)}
    assert RP.all_names(payload(pay, {("CONV", "POA"): 0.0350}))["CONV_OVERALL"] == P + "CONV-BOTH-LEVERS-REACH-LABEL-FORM-AT-A-PRICE"
    lower_f = {**low, ("CONV", "POA-VS-RLc", "F"): (-0.02, -0.002)}
    assert RP.all_names(payload(lower_f, {("CONV", "POA"): 0.0350}))["CONV_OVERALL"] == P + "CONV-BOTH-LEVERS-REACH-LABEL-FORM-AT-NO-SEPARATED-PRICE"
    touch_f = {**low, ("CONV", "POA-VS-RLc", "F"): (0.0, 0.02)}                      # an interval whose lower end is 0 is not above 0
    assert RP.all_names(payload(touch_f, {("CONV", "POA"): 0.0350}))["CONV_OVERALL"] == P + "CONV-BOTH-LEVERS-REACH-LABEL-FORM-AT-NO-SEPARATED-PRICE"
    # the bound
    assert RP.all_names(payload(low, {("CONV", "POA"): 0.0320 + 0.0045}))["CONV_OVERALL"] == P + "CONV-BOTH-LEVERS-REACH-LABEL-FORM-AT-NO-SEPARATED-PRICE"
    assert RP.all_names(payload(low, {("CONV", "POA"): 0.0366}))["CONV_OVERALL"] == P + "CONV-BOTH-LEVERS-DO-NOT-REACH-LABEL-FORM"
    not_lower = {("CONV", "POA-VS-GEc", "U"): (-0.03, 0.0)}                                # near the label form but the interval reaches 0
    assert RP.all_names(payload(not_lower, {("CONV", "POA"): 0.0300}))["CONV_OVERALL"] == P + "CONV-BOTH-LEVERS-DO-NOT-REACH-LABEL-FORM"
    po_only = {("CONV", "PO-VS-GEc", "U"): (-0.03, -0.002)}                                # presence alone lowering the rate does not make POA reach
    assert RP.all_names(payload(po_only, {("CONV", "PO"): 0.0300, ("CONV", "POA"): 0.0300}))["CONV_OVERALL"] == P + "CONV-BOTH-LEVERS-DO-NOT-REACH-LABEL-FORM"


def test_the_two_name_routines_agree_on_random_payloads():
    rng = random.Random(120)
    ends = [-0.05, -0.01, -1e-9, 0.0, 1e-9, 0.01, 0.05]
    lv = [0.0300, 0.0320, 0.0365 - 1e-12, 0.0320 + 0.0045, 0.0365 + 1e-9, 0.0500]
    for _ in range(400):
        iv, u = {}, {}
        for S in RP.SETTINGS:
            for pn in RP.PAIRS:
                for mk in "UCF":
                    lo, hi = sorted((rng.choice(ends), rng.choice(ends)))
                    iv[(S, pn, mk)] = (lo, hi)
            for arm in ("PO", "POA", "RLc"):
                u[(S, arm)] = rng.choice(lv)
        pl = payload(iv, u)
        got = RP.names_of(pl)
        assert got == RP.derive_all(pl) and RP.all_names(pl) == got and len(got) == 16


def test_all_names_refuses_when_the_routines_part(monkeypatch):
    monkeypatch.setattr(RP, "derive_all", lambda pl: dict(RP.names_of(pl), CONV_OVERALL="other"))
    with pytest.raises(SystemExit):
        RP.all_names(payload())


# ============================================================================================ e0 wrapper
def e0_args(**ch):
    keys = RP.new_keys() + [RP.IDP_KEY]
    a = dict(wp_sha=WP, rec={"pass": True, "variant_sha256": WP, "source_sha256": WSRC}, check_p_text="a\nZZ CHECK-WRAPPER-P PASS\n", wsha="28bc31f21ecb4ccb", check_text="x\nZZ CHECK-WRAPPER-V PASS\n",
             hashes_text="28bc31f21ecb4ccb file", check_files_present=True, run_shas={k: WP for k in keys})
    a.update(ch)
    return a


def test_e0_passes_and_each_clause_fails():
    assert RP.ctl_e0(**e0_args())["pass"] is True
    keys = RP.new_keys() + [RP.IDP_KEY]
    bads = [dict(wp_sha="c2a1d5f7242c3700" + "0" * 48), dict(wp_sha=None), dict(rec={"pass": False, "variant_sha256": WP, "source_sha256": WSRC}), dict(rec={"variant_sha256": WP, "source_sha256": WSRC}),
            dict(rec={"pass": True, "variant_sha256": "b2a1d5f7242c3700" + "9" * 48, "source_sha256": WSRC}), dict(rec={"pass": True, "variant_sha256": "d" * 64, "source_sha256": WSRC}),
            dict(rec={"pass": True, "variant_sha256": WP, "source_sha256": "e" * 64}), dict(rec=None), dict(check_p_text="a\nZZ CHECK-WRAPPER-P FAIL\n"), dict(check_p_text=""),
            dict(check_p_text="ZZ CHECK-WRAPPER-P PASS\nmore\n"), dict(wsha="0" * 16), dict(check_files_present=False), dict(check_text="ZZ CHECK-WRAPPER-V FAIL"), dict(hashes_text="no hash"),
            dict(run_shas={k: WP for k in keys[1:]}), dict(run_shas=dict({k: WP for k in keys}, **{keys[0]: "28bc31f21ecb4ccb" + "0" * 48})), dict(run_shas=dict({k: WP for k in keys}, **{RP.IDP_KEY: None}))]
    for ch in bads:
        r = RP.ctl_e0(**e0_args(**ch))
        assert r["pass"] is False and r["failures"], ch


# ============================================================================================ e1 identity run
def write_dump(d, nm, **arr):
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, nm), **arr)


def make_identity(tmp, n_dump=3, nan=True):
    pull = os.path.join(str(tmp), "pull")
    for nm in (RP.IDP_RUN, RP.IDP_BASE):
        d = os.path.join(pull, "I10%s_s0" % nm)
        for i in range(n_dump):
            x = np.arange(6, dtype=np.float32).reshape(2, 3)
            if nan:
                x[0, 0] = np.nan
            write_dump(RP.dump_dir(pull, nm, 0), "ep%03d.npz" % i, plan=x + i, flag=np.array([True, False]), names=np.array(["a", "b"]))
        os.makedirs(os.path.join(d, "ticks"), exist_ok=True)
        log = dict(ci=np.array([0, 0, 1]), j=np.array([0, 1, 0]), present=np.array([True, False, True]), g_src=np.array([5.0, np.nan, 7.0]), tag=np.array(nm), meta=np.array(1.0 if nm == RP.IDP_RUN else 2.0),
                   vis_gap_file=np.array("/x/" + nm), vis_closing=np.array(1))
        np.savez(os.path.join(d, "ticks", "rule_log_I10%s_s0.npz" % nm), **log)
        np.savez(os.path.join(d, "ticks", "rule_log_I10%s_s0_rd.npz" % nm), p=np.array([0.1, 0.2, 0.9]), ok=np.array([True, True, True]))
    return argparse.Namespace(pull=pull, expect_dumps=n_dump)


def patch(path, **ch):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for k, v in ch.items():
        if v is None:
            d.pop(k)
        else:
            d[k] = v
    np.savez(path, **d)


def ticks(pull, nm, f):
    return os.path.join(pull, "I10%s_s0" % nm, "ticks", f)


def test_e1_identity_passes_with_nan_and_text_keys_left_out(tmp_path):
    a = make_identity(tmp_path)
    bad, vals = RP.identity_read(a)
    assert bad == [] and vals["dump_files_identity_run"] == 3 and vals["rule_logs_compared"] == 1
    assert RP.ctl_e1(bad, vals)["pass"] is True


def test_e1_identity_fails_on_each_clause(tmp_path):
    pull = lambda a: a.pull
    # a different number of dump files from 141 (here 3 expected)
    a = make_identity(tmp_path / "n")
    assert RP.identity_read(argparse.Namespace(pull=a.pull, expect_dumps=4))[0]
    # an extra dump file in the identity run: the file names differ
    a = make_identity(tmp_path / "x")
    write_dump(RP.dump_dir(a.pull, RP.IDP_RUN, 0), "ep009.npz", plan=np.zeros(1))
    assert any("file names differ" in x for x in RP.identity_read(a)[0])
    # a missing dump file in the identity run
    a = make_identity(tmp_path / "m")
    os.remove(os.path.join(RP.dump_dir(a.pull, RP.IDP_RUN, 0), "ep001.npz"))
    assert RP.ctl_e1(*RP.identity_read(a))["pass"] is False
    # one value of one array of one dump differs
    a = make_identity(tmp_path / "v")
    p = os.path.join(RP.dump_dir(a.pull, RP.IDP_RUN, 0), "ep002.npz")
    plan = np.load(p)["plan"].copy()
    plan[1, 2] += 1.0
    patch(p, plan=plan)
    r = RP.identity_read(a)[0]
    assert any("ep002" in x and "plan" in x for x in r)
    # a different dtype
    a = make_identity(tmp_path / "d")
    p = os.path.join(RP.dump_dir(a.pull, RP.IDP_RUN, 0), "ep000.npz")
    patch(p, plan=np.load(p)["plan"].astype(np.float64))
    assert RP.identity_read(a)[0]
    # an extra key in a dump
    a = make_identity(tmp_path / "k")
    patch(os.path.join(RP.dump_dir(a.pull, RP.IDP_RUN, 0), "ep000.npz"), extra=np.zeros(1))
    assert any("keys differ" in x for x in RP.identity_read(a)[0])
    # a numeric rule log array differs
    a = make_identity(tmp_path / "l")
    patch(ticks(a.pull, RP.IDP_RUN, "rule_log_I10%s_s0.npz" % RP.IDP_RUN), g_src=np.array([5.0, np.nan, 7.5]))
    assert any("rule log 0" in x and "g_src" in x for x in RP.identity_read(a)[0])
    # a rule log key missing
    a = make_identity(tmp_path / "q")
    patch(ticks(a.pull, RP.IDP_RUN, "rule_log_I10%s_s0.npz" % RP.IDP_RUN), g_src=None)
    assert any("keys differ" in x for x in RP.identity_read(a)[0])
    # the side file has the column p_used
    a = make_identity(tmp_path / "s")
    patch(ticks(a.pull, RP.IDP_RUN, "rule_log_I10%s_s0_rd.npz" % RP.IDP_RUN), p_used=np.zeros(3))
    assert any("p_used" in x for x in RP.identity_read(a)[0])
    # no side file at all
    a = make_identity(tmp_path / "z")
    os.remove(ticks(a.pull, RP.IDP_RUN, "rule_log_I10%s_s0_rd.npz" % RP.IDP_RUN))
    assert RP.ctl_e1(*RP.identity_read(a))["pass"] is False
    # no run at all
    assert RP.ctl_e1(*RP.identity_read(argparse.Namespace(pull=str(tmp_path / "nothing"), expect_dumps=3)))["pass"] is False
    assert pull(a)


def test_compare_arrays_exclusion_is_text_and_meta_only():
    x = {"a": np.array([np.nan, 1.0]), "t": np.array("one"), "meta": np.array(1), "o": np.array([1, "x"], dtype=object)}
    y = {"a": np.array([np.nan, 1.0]), "t": np.array("two"), "meta": np.array(2), "o": np.array([2, "y"], dtype=object)}
    assert RP.compare_arrays(x, y, logs=True) == []
    assert RP.compare_arrays({"a": np.array([1, 2])}, {"a": np.array([1, 3])}, logs=True)
    assert RP.compare_arrays({"vis_closing": np.array(1)}, {"vis_closing": np.array(0)}, logs=True)
    assert RP.compare_arrays({"t": np.array("one")}, {"t": np.array(1.0)}, logs=True)
    assert RP.compare_arrays({"meta": np.array(1)}, {"meta": np.array(2)}, logs=False)


# ============================================================================================ e2 presence files
def e2_args(**ch):
    files = {m: {"sha": ("%d" % (m + 1)) * 64, "meta": {"dump_sha256": ("%d" % (m + 1)) * 64, "seed": m}} for m in RP.SEEDS}
    runs = {}
    for k in RP.new_keys():
        m = int(k.rsplit("_s", 1)[1])
        runs[k] = {"line": {"pres_sha": files[m]["sha"], "pres_mode": "or"}, "meta_sha": files[m]["sha"], "meta_mode": "or"}
    a = dict(files=files, runs=runs)
    a.update(ch)
    return a


def test_e2_passes_and_each_clause_fails():
    assert RP.ctl_e2(**e2_args())["pass"] is True
    base = e2_args()

    def mod(path, val):
        a = json.loads(json.dumps(base, default=str))
        a["files"] = {int(k): v for k, v in a["files"].items()}
        a["runs"] = a["runs"]
        d = a
        for p in path[:-1]:
            d = d[p]
        if val is None:
            d.pop(path[-1])
        else:
            d[path[-1]] = val
        return a
    cases = [(("files", 0, "sha"), "9" * 64), (("files", 0, "sha"), None), (("files", 1, "meta", "dump_sha256"), "8" * 64), (("files", 1, "meta", "seed"), 0), (("files", 1, "meta", "seed"), None),
             (("files", 0, "meta"), None), (("runs", "CONV_PO_s0", "line", "pres_sha"), None), (("runs", "CONV_PO_s0", "line", "pres_sha"), "2" * 64),            # seed 0 ran with the file of head seed 1
             (("runs", "K0_POA_s1", "line", "pres_sha"), "1" * 64), (("runs", "K0_PO_s0", "line", "pres_mode"), "head"), (("runs", "K0_PO_s0", "line", "pres_mode"), None),
             (("runs", "CONV_POA_s1", "meta_sha"), "1" * 64), (("runs", "CONV_POA_s1", "meta_sha"), None), (("runs", "CONV_POA_s1", "meta_mode"), "head"), (("runs", "CONV_POA_s1", "meta_mode"), None)]
    for path, val in cases:
        r = RP.ctl_e2(**mod(path, val))
        assert r["pass"] is False and r["failures"], (path, val)
    short = e2_args()
    short["runs"].pop("K0_PO_s1")
    assert RP.ctl_e2(**short)["pass"] is False
    assert RP.ctl_e2({}, {})["pass"] is False


# ============================================================================================ e8 rule logs, and the synthetic world of the run files
NC, NF = 2, 8
OFF, NN = np.array([0, NF]), np.array([NF, NF])


def pres_world(seed):
    p = np.full(NC * NF, np.nan)
    for c in range(NC):
        for f in range(4, NF):
            p[c * NF + f] = [0.9, 0.2, 0.8, 0.1, 0.95, 0.4][(f + c + seed) % 6]
    p[5] = 0.9                                                                                # clip 0 frame 5: a raised frame (reader p 0.2) for every seed
    has = np.isfinite(p)
    has[NF + 7] = False                                                                       # a finite p the file does not mark: not usable
    return dict(p=p, has_out=has, clips=np.array(["c0", "c1"]), off=OFF, n=NN, seed=np.int64(seed))


def reader_p():
    p = np.full((NC, NF), 0.3)
    p[0, 1], p[0, 2], p[0, 5], p[1, 3], p[1, 6], p[1, 7], p[0, 7] = 0.8, 0.9, 0.2, 0.1, 0.7, 0.6, 0.55
    return p


def gap_value(c, f, scale):
    return (10.0 + f + 0.5 * c) * scale


def sha_of(path):
    return RP.sha_file(path)


def rows_of_world(seed, with_pres, scale, gap_has):
    """-> arrays of one look (all clips, all frames) for a run: ci, j, p, p_used, ok, present, g_src, c_src"""
    pw = pres_world(seed)
    ci = np.repeat(np.arange(NC), NF)
    j = np.tile(np.arange(NF), NC)
    p = reader_p().reshape(-1)
    pf, have = RP.pres_at(pw, ci, j)
    pu = np.where(have, np.fmax(p, pf), p) if with_pres else p.copy()
    ok = np.ones(len(ci), bool)
    ok[4] = False
    src_bad = np.zeros(len(ci), bool)
    present = (pu >= 0.5) & ok & ~src_bad
    g = np.array([gap_value(c, f, scale) if gap_has[c, f] else 7.0 for c, f in zip(ci, j)])
    return dict(ci=ci, j=j, p=p, p_used=pu, ok=ok, src_bad=src_bad, present=present, g_src=np.where(present, g, 0.0), c_src=np.where(present, 1.0, 0.0), pu=pu)


def make_run_world(tmp):
    """a tiny world with the eight new runs, the five reference runs RR, the gap files and the presence files; every clause of e8 and e2 reads true. -> Namespace a"""
    root = str(tmp)
    pull, gd, pd_ = os.path.join(root, "pull"), os.path.join(root, "gap"), os.path.join(root, "pres")
    os.makedirs(gd), os.makedirs(pull)
    gap_has = np.ones((NC, NF), bool)
    gap_has[0, 5] = False                                                                     # a raised frame with no gap-file output and no lead in RR
    gap_has[1, 0] = False
    files = {}
    for nm, scale in (("VGAPLE", 1.0), ("VGAPLE_A90", 0.9)):
        ci = np.repeat(np.arange(NC), NF)
        fr = np.tile(np.arange(NF), NC)
        g = np.array([gap_value(c, f, scale) for c, f in zip(ci, fr)])
        has = np.array([gap_has[c, f] for c, f in zip(ci, fr)])
        np.savez(os.path.join(gd, nm + ".npz"), clip_index=ci, frame=fr, gap=np.where(has, g, np.nan), has_out=has)
    pres_sha = {}
    for m in RP.SEEDS:
        d = os.path.join(pd_, "D_H5_s%d" % m)
        os.makedirs(d)
        np.savez(os.path.join(d, "pres.npz"), **pres_world(m))
        pres_sha[m] = sha_of(os.path.join(d, "pres.npz"))
        json.dump({"dump_sha256": pres_sha[m], "seed": m}, open(os.path.join(d, "pres_meta.json"), "w"))
    a = argparse.Namespace(pull=pull, margin_dir=gd, gap_file=os.path.join(gd, "VGAPLE.npz"), pres_dir=pd_)
    for S in RP.SETTINGS:
        for arm in RP.ARMS:
            if arm not in ("PO", "POA", "RR"):
                continue
            for m in RP.SEEDS:
                nm = RP.run_name(S, arm)
                new = arm in ("PO", "POA")
                scale = 0.9 if arm == "POA" else 1.0
                w = rows_of_world(m, new, scale, gap_has)
                if arm == "RR":
                    gap_sha = ""
                else:
                    gap_sha = sha_of(RP.gap_file_of(a, arm))
                n_logs = 2 if RP.SPEC[S]["every"] is not None else 1
                rows = raised = with_file = 0
                logs = []
                for i in range(n_logs):
                    logs.append(w)
                    rows += len(w["ci"])
                    pf, have = RP.pres_at(pres_world(m), w["ci"], w["j"])
                    with_file += int(have.sum())
                    raised += int(((w["p"] < 0.5) & (w["p_used"] >= 0.5)).sum())
                counts = {"rows": rows, "with_file": with_file, "no_file": rows - with_file, "raised": raised, "lowered": 0}
                d = os.path.join(pull, "I10%s_s%d" % (nm, m), "ticks")
                os.makedirs(d)
                for i, wl in enumerate(logs):
                    base = "rule_log_I10%s_s%d.npz" % (nm, m) if i == 0 else "rule_log_mid_I10%s_s%d.npz" % (nm, m)
                    log = {k: wl[k] for k in ("ci", "j", "present", "src_bad", "g_src", "c_src")}
                    log["p_src"] = wl["p_used"]
                    log["vis_gap_file_sha256"] = np.array(gap_sha)
                    log["vis_closing"] = np.array(1)
                    if new:
                        log.update(vis_pres_counts=np.array(json.dumps(counts, sort_keys=True)), vis_pres_file_sha256=np.array(pres_sha[m]), vis_pres_mode=np.array("or"))
                    np.savez(os.path.join(d, base), **log)
                    side = {"p": wl["p"], "ok": wl["ok"]}
                    if new:
                        side["p_used"] = wl["p_used"]
                    np.savez(os.path.join(d, base[:-4] + "_rd.npz"), **side)
                line = LINE % ("I10%s_s%d" % (nm, m), RP.SPEC[S]["every"], "causal_g", m, m)
                if new:
                    line += " vis-pres-file /x/pres.npz vis pres sha256 %s vis-pres-mode or" % pres_sha[m]
                io.open(os.path.join(d, "run_I10%s_s%d.txt" % (nm, m)), "w", encoding="utf-8", newline="\n").write(line + "\nrc 0 finished\n")
    return a


def ticks_of(a, S, arm, m, mid=False, side=False):
    nm = RP.run_name(S, arm)
    base = ("rule_log_mid_I10%s_s%d" if mid else "rule_log_I10%s_s%d") % (nm, m)
    return os.path.join(a.pull, "I10%s_s%d" % (nm, m), "ticks", base + ("_rd.npz" if side else ".npz"))


def test_the_world_itself_has_raised_rows_and_every_clause_passes(tmp_path):
    a = make_run_world(tmp_path)
    per = RP.rule_checks(a)
    assert sorted(per) == sorted(RP.new_keys())
    for k, v in per.items():
        assert v["PASS"] is True, (k, v["failures"])
    assert per["CONV_PO_s0"]["raised"] > 0 and per["CONV_PO_s0"]["lowered"] == 0 and per["K0_POA_s1"]["rows"] == 2 * NC * NF
    assert per["CONV_PO_s0"]["n_raised_rows_without_reference"] >= 1 and per["CONV_POA_s0"]["with_file"] > 0
    r = RP.ctl_e8(per)
    assert r["pass"] is True and set(r["values"]["rollout_shares"]) == set(RP.new_keys())
    assert r["values"]["rollout_shares"]["CONV_PO_s0"]["raised"] == per["CONV_PO_s0"]["raised"]
    files, runs = RP.pres_info(a)
    assert RP.ctl_e2(files, runs)["pass"] is True


def first_failure_of(a, key):
    return " | ".join(RP.rule_checks(a)[key]["failures"])


def test_e8_each_clause_fails(tmp_path):
    # (a) p_used not the larger of p and the file's value
    a = make_run_world(tmp_path / "a")
    sp = ticks_of(a, "CONV", "PO", 0, side=True)
    pu = np.load(sp)["p_used"].copy()
    pu[1] = 0.1
    patch(sp, p_used=pu)
    assert "q8 (a)" in first_failure_of(a, "CONV_PO_s0")
    # (a) the file has no value (has_out false at clip 1 frame 7): p_used must be p, not a value of the file
    a = make_run_world(tmp_path / "a2")
    sp = ticks_of(a, "CONV", "PO", 1, side=True)
    pu = np.load(sp)["p_used"].copy()
    pu[NF + 7] = 0.99
    patch(sp, p_used=pu)
    assert "q8 (a)" in first_failure_of(a, "CONV_PO_s1")
    # (b) present differs from p_used >= 0.5 and ok and not src_bad
    a = make_run_world(tmp_path / "b")
    lp = ticks_of(a, "K0", "POA", 0)
    pr = np.load(lp)["present"].copy()
    pr[2] = ~pr[2]
    patch(lp, present=pr)
    assert "q8 (b)" in first_failure_of(a, "K0_POA_s0")
    # (b) a row with ok false that the log marks present
    a = make_run_world(tmp_path / "b2")
    lp = ticks_of(a, "CONV", "POA", 1)
    pr = np.load(lp)["present"].copy()
    pr[4] = True
    patch(lp, present=pr)
    assert "q8 (b)" in first_failure_of(a, "CONV_POA_s1")
    # (c) p_src not p_used
    a = make_run_world(tmp_path / "c")
    lp = ticks_of(a, "K0", "PO", 1, mid=True)
    ps = np.load(lp)["p_src"].copy()
    ps[3] += 0.01
    patch(lp, p_src=ps)
    assert "q8 (c)" in first_failure_of(a, "K0_PO_s1")
    # (d) the gap the rule received differs from the run's own gap file on a lead row with a file output (PO: base file; POA: the A90 file)
    a = make_run_world(tmp_path / "d")
    lp = ticks_of(a, "CONV", "POA", 0)
    g = np.load(lp)["g_src"].copy()
    i = int(np.flatnonzero(np.load(lp)["present"])[0])
    g[i] += 0.5
    patch(lp, g_src=g)
    assert "q8 (d)" in first_failure_of(a, "CONV_POA_s0")
    a = make_run_world(tmp_path / "d2")                                                       # POA logged with the base gap file's values: the A90 file is the reference
    lp = ticks_of(a, "CONV", "POA", 0)
    z = np.load(lp)
    patch(lp, g_src=np.where(z["present"], z["g_src"] / 0.9, 0.0))
    assert "q8 (d)" in first_failure_of(a, "CONV_POA_s0")
    # (d) the meta names another gap file
    a = make_run_world(tmp_path / "d3")
    patch(ticks_of(a, "CONV", "PO", 0), vis_gap_file_sha256=np.array("ab" * 32))
    assert "gap file hash" in first_failure_of(a, "CONV_PO_s0")
    # (e) the counts differ
    for key, val in (("rows", 1), ("with_file", 0), ("no_file", 0), ("raised", 0)):
        a = make_run_world(tmp_path / ("e_" + key))
        lp = ticks_of(a, "CONV", "PO", 0)
        rec = json.loads(str(np.load(lp)["vis_pres_counts"]))
        rec[key] = val
        patch(lp, vis_pres_counts=np.array(json.dumps(rec, sort_keys=True)))
        assert "q8 (e)" in first_failure_of(a, "CONV_PO_s0") and key in first_failure_of(a, "CONV_PO_s0")
    a = make_run_world(tmp_path / "e_nometa")
    patch(ticks_of(a, "CONV", "PO", 0), vis_pres_counts=None)
    assert "q8 (e)" in first_failure_of(a, "CONV_PO_s0")
    # (e) a lowered row: the record says 1 lowered
    a = make_run_world(tmp_path / "e_low")
    lp = ticks_of(a, "CONV", "PO", 0)
    rec = json.loads(str(np.load(lp)["vis_pres_counts"]))
    rec["lowered"] = 1
    patch(lp, vis_pres_counts=np.array(json.dumps(rec, sort_keys=True)))
    assert "lowered" in first_failure_of(a, "CONV_PO_s0")
    # the real recompute of a lowered row: p_used under p on a row with p >= 0.5
    a = make_run_world(tmp_path / "e_low2")
    sp = ticks_of(a, "CONV", "PO", 0, side=True)
    pu = np.load(sp)["p_used"].copy()
    pu[2] = 0.1
    patch(sp, p_used=pu)
    assert "lowered rows: recomputed 1" in first_failure_of(a, "CONV_PO_s0")


def test_e8_structure_failures(tmp_path):
    a = make_run_world(tmp_path / "s")
    os.remove(ticks_of(a, "K0", "PO", 0, mid=True, side=True))                               # no side file of the mid log
    assert "no side file" in first_failure_of(a, "K0_PO_s0")
    a = make_run_world(tmp_path / "s2")
    sp = ticks_of(a, "CONV", "POA", 1, side=True)
    patch(sp, p_used=None)                                                                   # no column p_used
    assert "side p_used" in first_failure_of(a, "CONV_POA_s1")
    a = make_run_world(tmp_path / "s3")
    sp = ticks_of(a, "CONV", "PO", 0, side=True)
    patch(sp, p=np.zeros(3), p_used=np.zeros(3), ok=np.ones(3, bool))                        # other number of rows than the log
    assert "different numbers of rows" in first_failure_of(a, "CONV_PO_s0")
    a = make_run_world(tmp_path / "s4")
    os.remove(a.gap_file)
    assert all("no gap file" in " ".join(v["failures"]) for k, v in RP.rule_checks(a).items() if "_PO_" in k)
    a = make_run_world(tmp_path / "s5")
    shutil.rmtree(os.path.join(a.pres_dir, "D_H5_s1"))
    per = RP.rule_checks(a)
    assert all("no presence file" in " ".join(v["failures"]) for k, v in per.items() if k.endswith("_s1")) and per["CONV_PO_s0"]["PASS"] is True
    assert RP.ctl_e8(per)["pass"] is False and RP.ctl_e8({})["pass"] is False
    one = RP.rule_checks(make_run_world(tmp_path / "s6"))
    one.pop("K0_POA_s1")
    assert RP.ctl_e8(one)["pass"] is False


def test_the_seed_of_the_presence_file_is_the_seed_of_the_run(tmp_path):
    a = make_run_world(tmp_path)
    for m in (0, 1):                                                                         # swap the two files: each run now meets the other seed's file
        pass
    d0, d1 = (os.path.join(a.pres_dir, "D_H5_s%d" % m, "pres.npz") for m in (0, 1))
    z0, z1 = dict(np.load(d0)), dict(np.load(d1))
    np.savez(d0, **z1)
    np.savez(d1, **z0)
    per = RP.rule_checks(a)
    assert any(v["PASS"] is False for v in per.values()) and "q8 (a)" in " ".join(per["CONV_PO_s0"]["failures"])
    files, runs = RP.pres_info(a)
    assert RP.ctl_e2(files, runs)["pass"] is False


def test_pres_at_has_no_value_outside_the_file():
    F = pres_world(0)
    pf, have = RP.pres_at(F, [0, 0, 1, 2, -1, 0, 1], [4, 3, 7, 4, 4, 8, 6])
    assert list(have) == [True, False, False, False, False, False, True]                      # frame 3 not finite, clip 1 frame 7 has_out false, clip 2 and -1 outside, frame 8 outside
    assert np.isnan(pf[1]) and pf[0] == F["p"][4] and pf[6] == F["p"][NF + 6]


# ============================================================================================ refusals, driver, wording
def test_refusals(tmp_path):
    assert RP.main(["--selftest"]) == 2
    assert RP.main(["--stage", "names"]) == 2
    base = {"card": RP.CARD, "all_pass": True, "payload": payload(), "controls": {k: {"pass": True} for k in RP.CONTROLS}}
    for label, ch in (("one control failed", {"controls": dict(base["controls"], e8={"pass": False})}), ("missing control", {"controls": {k: v for k, v in base["controls"].items() if k != "e8"}}),
                      ("withheld", {"all_pass": False}), ("no payload", {"payload": None}), ("other card", {"card": "PREREG_V8F_PROOF_ADD117"})):
        cj = dict(base, **ch)
        if cj.get("payload") is None:
            cj.pop("payload")
        cp = str(tmp_path / ("c_%s.json" % label.replace(" ", "_")))
        out = str(tmp_path / ("o_%s.json" % label.replace(" ", "_")))
        json.dump(cj, open(cp, "w"))
        assert RP.main(["--stage", "names", "--controls", cp, "--out", out]) == 3, label
        assert not os.path.exists(out)
    eight = dict(base, controls={k: {"pass": True} for k in RP.CONTROLS if k != "e8"})                   # the eight controls of the source kit: not enough
    json.dump(eight, open(str(tmp_path / "c8.json"), "w"))
    assert RP.main(["--stage", "names", "--controls", str(tmp_path / "c8.json"), "--out", str(tmp_path / "o8.json")]) == 3


def test_driver_text():
    t = rd(DRIVER)
    r = subprocess.run([BASH, "-n", DRIVER.replace("\\", "/")], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    assert "GE=$V/presroll; K=$GE/kit_read; OUT=$GE/read; LT=$GE/lead_tests;" in t and "GFM=$V/gapmargin/gapfiles" in t and "PRD=$V/preshead2/dump" in t
    assert 'WRAPP=$GE/kit_wrap/v8f_t1_eval_reader_rule_rzgfhqetjvp.py; WREC=$GE/kit_wrap/WRAPPER_P_GEN_RECORD.json; CHKP=$GE/check_p/CHECK_WRAPPER_P.out' in t
    assert '"$PY" "$K/read_presroll10k.py" --stage controls' in t and '"$PY" "$K/read_presroll10k.py" --stage names' in t
    assert '--wrapper-p-file "$WRAPP" --wrapper-p-record "$WREC" --wrapper-p-check "$CHKP" --pres-dir "$PRD"' in t
    for x in ("I10RQ50Hb20gGEcPO", "I10RQ50Hb20gA90PO", "I10RQ50KF5Hb20gGEcPO", "I10RQ50KF5Hb20gA90PO", "I10RLcHb20pLE", "I10RLcKF5Hb20pLE", "I10R0Hb20", "I10RQ50Hb20gGEcIDP_s0"):
        assert x in t, x
    assert '[ -f "$GE/PRESROLL10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"' in t and '[ -e "$GE/PRESROLL10K_STOP.txt" ] && stop "QUEUE-STOPPED"' in t and "PRESROLL10K_START.txt" in t
    assert "PRESROLL10K_CONTROLS.json" in t and "PRESROLL10K_RESULT.json" in t and "read_names.log" in t and "read_names.out" in t
    assert "lead_pytest_queue.out" in t and "lead_pytest_read.out" in t and "ZZ PODREAD-PRESROLL10K" in t and "e[0-8]" in t
    for bad in ("GAPMARGIN", "VGAPLE_A80", "--margin-record", "grid_p30"):
        assert bad not in t
    code = "\n".join(x for x in t.splitlines() if not x.lstrip().startswith("#"))
    assert not re.search(r"(^|[;&|]\s*|\s)(rm|mv|rmdir)\s", code)


def test_pull_script_text():
    p = os.path.join(HERE, "pull_presroll.sh")
    t = rd(p)
    r = subprocess.run([BASH, "-n", p.replace("\\", "/")], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    assert "ROOT=/workspace/v8proof/presroll" in t and "D=/f/TanitAD-scratch/v8proof/presroll" in t and 'pod_add120' in t and "tmp_pull_add120.txt" in t and "tmp_pull_add120.tar" in t
    for f in ("read/*.json", "read/*.log", "read/*.out", "read/*.txt", "presroll_queue.log", "small_pr/*", "check_p/*.out", "PRESROLL10K_START.txt", "PRESROLL10K_DONE.txt", "PRESROLL10K_STOP.txt",
              "lead_tests/lead_pytest_queue.out", "lead_tests/lead_pytest_read.out"):
        assert f in t, f
    assert "add117" not in t and "gapmargin" not in t
    code = "\n".join(x for x in t.splitlines() if not x.lstrip().startswith("#"))
    assert not re.search(r"(^|[;&|]\s*|\s)(rm|mv|rmdir)\s", code)


def test_wording_of_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("correc", "tion"), ("with", "drawn"), ("no lon", "ger"), ("closed", " loop")]
    for fn in os.listdir(HERE):
        p = os.path.join(HERE, fn)
        if os.path.isfile(p) and fn.endswith((".py", ".sh", ".md")):
            low = rd(p).lower()
            for a, b in parts:
                assert (a + b) not in low, (fn, a + b)


def test_print_sections_prints_names_levels_pairs_and_the_rollout_shares():
    pl = payload()
    for S in RP.SETTINGS:
        d = pl["settings"][S]
        d["counts"] = {"n_windows": 10, "n_lead_windows": 6, "n_episodes_with_lead": 3}
        d["instrument_bar_met"] = False
        d["levels_per_seed"] = {arm: {"C": [0.0, 0.0], "U": [0.1, 0.1], "F": [0.2, 0.2]} for arm in RP.ARMS}
        for pn in RP.PAIRS:
            for mk in "UCF":
                d["pairs"][pn][mk].update(mean=0.0)
        d["families"], d["strategic"] = {}, {"pairs": {}}
    pl["control_values"] = {"e8": {"rollout_shares": {k: {"share_of_rule_steps_with_a_lead": 0.3, "raised": 4, "rows": 40} for k in RP.new_keys()}}}
    lines = RP.print_sections(pl, RP.all_names(pl))
    text = "\n".join(lines)
    assert lines[0].startswith("TIER ") and sum(1 for x in lines if x.startswith("rollout ")) == 8 and '"raised": 4' in text and "level CONV PO pooled" in text and "level K0 POA pooled" in text
    assert "CONV PO-VS-PL:" in text and "K0 POA-VS-RLc:" in text and sum(1 for x in lines if x.startswith(P)) == 16
