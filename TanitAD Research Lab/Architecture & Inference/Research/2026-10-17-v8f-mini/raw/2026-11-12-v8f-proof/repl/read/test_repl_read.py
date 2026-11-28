"""test_repl_read.py -- card ADD121 (REPL10K), reader kit (helper-written): the generated reader and driver against their generator, the refusals of the generator, the tables against the card, the
names on hand-made payloads (every branch of section 4, and the agreement of the two name routines on random payloads), the controls e0, e2 and e6 on tiny synthetic files and texts (each
clause passing and failing), the composed e1, the driver and the pull script, and the wording of the kit files. The source kit of card ADD117 is taken from ../../gapmargin/read. No pod result
file is opened; every array and every text here is a few rows."""
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "make_repl_reader.py")
READER = os.path.join(HERE, "read_repl10k.py")
DRIVER = os.path.join(HERE, "podread_repl10k.sh")
PULL = os.path.join(HERE, "pull_repl.sh")
CARD_FILE = os.path.join(HERE, "..", "..", "PREREG_V8F_PROOF_ADD121.md")
SRC_KIT = os.path.join(HERE, "..", "..", "gapmargin", "read")
BASH = shutil.which("bash")
sys.path.insert(0, HERE)
import read_repl10k as RP  # noqa: E402

P = "REPL10K-"
H64 = "ab" * 32
LINE = ("arm %s form RR replan steps 10%s readout causal_g kappa 0.2%s model seed %d reader seed %d ckpt sha256 %s wrapper sha256 " + "28bc31f21ecb4ccb" + "0" * 48)


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


# ============================================================================================ generator
@pytest.mark.parametrize("kind,path", [("reader", READER), ("driver", DRIVER)])
def test_generated_file_is_what_the_generator_prints(kind, path):
    r = subprocess.run([sys.executable, GEN, "--print", kind], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.replace("\r\n", "\n") == rd(path)
    assert "\r" not in rd(path)


def _gen_world(tmp_path, change_source=False, existing_output=False):
    """a copy of the generator in tmp/x/y and copies of the two source files in tmp/gapmargin/read"""
    gdir = tmp_path / "x" / "y"
    gdir.mkdir(parents=True)
    sdir = tmp_path / "gapmargin" / "read"
    sdir.mkdir(parents=True)
    shutil.copy(GEN, str(gdir / "make_repl_reader.py"))
    for nm in ("read_gapmargin10k.py", "podread_gapmargin10k.sh"):
        t = open(os.path.join(SRC_KIT, nm), "rb").read()
        if change_source and nm.endswith(".py"):
            t = t + b"# one more line\n"
        (sdir / nm).write_bytes(t)
    if existing_output:
        (gdir / "read_repl10k.py").write_text("x")
    return str(gdir / "make_repl_reader.py")


def test_generator_runs_on_a_copy_and_writes_the_same_files(tmp_path):
    g = _gen_world(tmp_path)
    r = subprocess.run([sys.executable, g], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    d = os.path.dirname(g)
    assert rd(os.path.join(d, "read_repl10k.py")) == rd(READER) and rd(os.path.join(d, "podread_repl10k.sh")) == rd(DRIVER)


def test_generator_refuses_a_changed_source(tmp_path):
    g = _gen_world(tmp_path, change_source=True)
    r = subprocess.run([sys.executable, g], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode != 0 and "source sha256 prefix" in (r.stderr + r.stdout)
    assert not os.path.exists(os.path.join(os.path.dirname(g), "read_repl10k.py"))


def test_generator_refuses_an_existing_output(tmp_path):
    g = _gen_world(tmp_path, existing_output=True)
    r = subprocess.run([sys.executable, g], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode != 0 and "exists" in (r.stderr + r.stdout)
    assert not os.path.exists(os.path.join(os.path.dirname(g), "podread_repl10k.sh"))


# ============================================================================================ tables
def test_tables():
    assert RP.CARD == "PREREG_V8F_PROOF_ADD121" and RP.PRE == "REPL10K" and RP.SETTINGS == ("CONV", "K0", "K30")
    assert RP.SEEDS == (2, 3) and RP.FIRST_SEEDS == (0, 1) and RP.ALL_SEEDS == (0, 1, 2, 3) and RP.READER_SEED_OF == {2: 0, 3: 1}
    assert RP.ARMS == ("RLc", "RR", "GEc", "A90") and RP.SPREAD == "R0Hb20"
    assert RP.SPEC["CONV"]["runs"] == {"RLc": "RLcHb20", "RR": "RQ50Hb20", "GEc": "RQ50Hb20gGEc", "A90": "RQ50Hb20gA90"}
    assert RP.SPEC["K0"]["runs"] == {"RLc": "RLcKF5Hb20", "RR": "RQ50KF5Hb20", "GEc": "RQ50KF5Hb20gGEc", "A90": "RQ50KF5Hb20gA90"}
    assert RP.SPEC["K30"]["runs"] == {"RLc": "RLcKF5P30Hb20", "RR": "RQ50KF5P30Hb20", "GEc": "RQ50KF5P30Hb20gGEc", "A90": "RQ50KF5P30Hb20gA90"}
    assert RP.SPEC["CONV"]["every"] is None and RP.SPEC["CONV"]["readout"] == "pinned" and RP.SPEC["K0"]["every"] == 5 and RP.SPEC["K30"]["every"] == 5
    assert RP.SPEC["K30"]["P"] == 30 and RP.SPEC["K30"]["grid"] == "30" and RP.SPEC["K30"]["grid_file"] == "grid_p30.json"
    assert all(RP.SPEC[S]["P"] == 0 and RP.SPEC[S]["grid"] == "full" and RP.SPEC[S]["grid_file"] == "pinned" for S in ("CONV", "K0"))
    assert RP.NEW_ARMS == {S: RP.ARMS for S in RP.SETTINGS}
    assert sorted(RP.new_keys()) == sorted("%s_%s_s%d" % (S, a, m) for S in RP.SETTINGS for a in RP.ARMS for m in (2, 3)) and len(RP.new_keys()) == 24
    assert len(RP.all_run_keys()) == 26 and RP.spread_keys() == ["CONV_R0_s2", "CONV_R0_s3"] and len(RP.rule_keys()) == 12
    assert RP.PAIRS == {"A90-VS-GEc": ("A90", "GEc"), "GEc-VS-RLc": ("GEc", "RLc"), "A90-VS-RLc": ("A90", "RLc"), "GEc-VS-RR": ("GEc", "RR")}
    assert RP.EXPECTED["CONV"] == {"GEc": (0.0016, 0.0569, 0.2329), "A90": (0.0008, 0.0463, 0.2539), "RLc": (0.0016, 0.0320, 0.2243), "RR": (0.0280, 0.0783, 0.2593)}
    assert RP.EXPECTED["K0"] == {"GEc": (0.0000, 0.0498, 0.1651), "A90": (0.0000, 0.0374, 0.1729), "RLc": (0.0000, 0.0365, 0.1550), "RR": (0.0296, 0.0738, 0.1682)}
    assert RP.EXPECTED["K30"] == {"GEc": (0.0062, 0.0839, 0.5154), "A90": (0.0031, 0.0571, 0.5246), "RLc": (0.0000, 0.0714, 0.4908), "RR": (0.0954, 0.1339, 0.4954)}
    assert RP.EXPECT_LEADS == {"CONV": 642, "K0": 642, "K30": 325} and RP.EXPECT_DUMPS == 141 and RP.CLOSE_TOL == 0.0045 and RP.LEVEL_TOL == 1e-4
    assert RP.WRAPPER_SHA == "28bc31f21ecb4ccb" and RP.MARGIN_ARMS == ("A90",) and RP.MARGIN_FILES == {"A90": "VGAPLE_A90"} and RP.RULE_ARMS == ("GEc", "A90")
    assert RP.DONE_RE.match("ZZ REPL10K DONE 2026-11-22T16:00:00Z all steps done") and not RP.DONE_RE.match("ZZ REPL10K DONE x done")
    assert RP.CONTROLS == tuple("e%d" % i for i in range(8)) and RP.ref_arm("K0", "A90") == "A90"
    assert [(k, nm, S, m) for k, nm, S, m in RP.run_list()][-2:] == [("CONV_R0_s2", "R0Hb20", "CONV", 2), ("CONV_R0_s3", "R0Hb20", "CONV", 3)]


def test_claims_and_sides():
    assert RP.CLAIMS == (("M-K0", "K0", "A90-VS-GEc", "U", -1), ("M-K30", "K30", "A90-VS-GEc", "U", -1), ("P-K0", "K0", "A90-VS-GEc", "F", 1), ("D-CONV", "CONV", "GEc-VS-RLc", "U", 1),
                         ("D-K0", "K0", "GEc-VS-RLc", "U", 1), ("C-K0", "K0", "GEc-VS-RR", "C", -1), ("C-CONV", "CONV", "GEc-VS-RR", "C", -1))
    assert RP.FIRST_PAIR_CLAIMS == ("C-K0", "C-CONV") and RP.NOT_A_CLAIM == "not a claim"
    for X, S, pr, mk, side in RP.CLAIMS:
        assert pr in RP.PAIRS and S in RP.SETTINGS and mk in "UCF"


@pytest.mark.skipif(not os.path.isfile(CARD_FILE), reason="the card file is not beside the kit on this machine")
def test_tables_against_the_card_text():
    card = rd(CARD_FILE)
    for S in RP.SETTINGS:
        for arm, nm in RP.SPEC[S]["runs"].items():
            assert "`I10%s`" % nm in card, nm
    assert "`I10R0Hb20`" in card
    row = re.search(r"\* \*\*r3 identity\*\*.*", card).group(0)
    for S, label in (("CONV", "CONV"), ("K0", "K0"), ("K30", "K30")):
        seg = re.search(label + r" (GEc .*?)(?:; K|\.$|$)", row.replace("**", ""))
        assert seg, label
        for arm in ("GEc", "A90", "RLc", "RR"):
            m = re.search(arm + r" ([0-9.]+) / ([0-9.]+) / ([0-9.]+)", seg.group(1))
            assert m and tuple(float(x) for x in m.groups()) == RP.EXPECTED[S][arm], (S, arm)
    for X, S, pr, mk, side in RP.CLAIMS:
        assert re.search(r"\| %s \|" % re.escape(X), card), X
    assert "642, 642, 325" in card and "141 dump files" in card and "0.0045" in card
    assert "2,000 draws" in card


# ============================================================================================ names
def iv_of(kind):
    return {"lo": [-0.02, -0.005], "hi": [0.005, 0.02], "zero": [-0.01, 0.01], "edge_lo": [-0.01, 0.0], "edge_hi": [0.0, 0.01]}[kind]


def mk_payload(new=None, first=None, u_a90=0.04, u_rlc=0.04):
    """a payload for the names: new / first {(setting, pair, measure): kind}; every other interval is 'zero' (it includes 0)"""
    st = {}
    for S in RP.SETTINGS:
        est = {}
        for en, over in (("new", new or {}), ("first", first or {}), ("four", {})):
            est[en] = {nm: {m: {"interval": iv_of(over.get((S, nm, m), "zero"))} for m in "CUF"} for nm in RP.PAIRS}
        st[S] = {"est": est, "levels": {arm: {"C": 0.0, "U": 0.05, "F": 0.2} for arm in RP.ARMS}, "levels_first": {arm: {"C": 0.0, "U": 0.05, "F": 0.2} for arm in RP.ARMS}}
        st[S]["levels"]["A90"]["U"], st[S]["levels"]["RLc"]["U"] = u_a90, u_rlc
    return {"settings": st}


ALL_CLAIM_SIDE = {"M-K0": ("lo", "hi"), "M-K30": ("lo", "hi"), "P-K0": ("hi", "lo"), "D-CONV": ("hi", "lo"), "D-K0": ("hi", "lo"), "C-K0": ("lo", "hi"), "C-CONV": ("lo", "hi")}


def claim_payload(want, first_ok=True):
    """want {claim: 'rep'|'not'|'opp'}: the interval on the new pair of each claim; first_ok: the first-pair interval of the C claims is on the claimed side"""
    new, first = {}, {}
    for X, S, pr, mk, side in RP.CLAIMS:
        on, opp = ALL_CLAIM_SIDE[X]
        new[(S, pr, mk)] = {"rep": on, "opp": opp, "not": "zero"}[want.get(X, "not")]
        first[(S, pr, mk)] = on if first_ok else "zero"
    return mk_payload(new, first)


def test_names_when_nothing_is_separated():
    nm = RP.all_names(claim_payload({}))
    for X, *_ in RP.CLAIMS:
        assert nm[X] == P + X + "-NOT-SEPARATED-ON-THE-NEW-PAIR"
    assert nm["MARGIN"] == P + "MARGIN-NOT-REPLICATED" and len(nm) == 8


def test_names_replicated_and_opposite_per_claim():
    for X, S, pr, mk, side in RP.CLAIMS:
        assert RP.all_names(claim_payload({X: "rep"}))[X] == P + X + "-REPLICATED"
        assert RP.all_names(claim_payload({X: "opp"}))[X] == P + X + "-OPPOSITE"


def test_names_not_a_claim_for_the_two_contact_claims():
    for X in ("C-K0", "C-CONV"):
        for w in ("rep", "not", "opp"):
            assert RP.all_names(claim_payload({X: w}, first_ok=False))[X] == "not a claim"
    nm = RP.all_names(claim_payload({"M-K0": "rep"}, first_ok=False))
    assert nm["M-K0"] == P + "M-K0-REPLICATED"                                      # only the contact claims depend on the first pair
    assert nm["D-K0"] == P + "D-K0-NOT-SEPARATED-ON-THE-NEW-PAIR"


def test_margin_names_four_branches():
    assert RP.all_names(claim_payload({"M-K0": "rep", "M-K30": "rep"}))["MARGIN"] == P + "MARGIN-REPLICATED"
    assert RP.all_names(claim_payload({"M-K0": "rep"}))["MARGIN"] == P + "MARGIN-REPLICATED-AT-K0-ONLY"
    assert RP.all_names(claim_payload({"M-K30": "rep"}))["MARGIN"] == P + "MARGIN-REPLICATED-AT-K30-ONLY"
    assert RP.all_names(claim_payload({"M-K0": "opp", "M-K30": "not"}))["MARGIN"] == P + "MARGIN-NOT-REPLICATED"


def test_interval_edges_at_zero_are_not_separated():
    pl = mk_payload({("K0", "A90-VS-GEc", "U"): "edge_lo", ("K30", "A90-VS-GEc", "U"): "edge_hi", ("K0", "GEc-VS-RLc", "U"): "edge_hi"})
    nm = RP.names_of(pl)
    assert nm["M-K0"].endswith("NOT-SEPARATED-ON-THE-NEW-PAIR") and nm["M-K30"].endswith("NOT-SEPARATED-ON-THE-NEW-PAIR") and nm["D-K0"].endswith("NOT-SEPARATED-ON-THE-NEW-PAIR")


def test_only_the_new_pair_decides_and_the_four_seed_estimate_carries_no_name():
    pl = claim_payload({"M-K0": "not"})
    for S in RP.SETTINGS:
        for nm in RP.PAIRS:
            pl["settings"][S]["est"]["four"][nm]["U"]["interval"] = [-0.5, -0.4]
    assert RP.all_names(pl)["M-K0"].endswith("NOT-SEPARATED-ON-THE-NEW-PAIR")


def test_bound_statement_is_not_a_name():
    assert RP.bound_k0(mk_payload(u_a90=0.0414, u_rlc=0.0370)) is True
    assert RP.bound_k0(mk_payload(u_a90=0.0416, u_rlc=0.0370)) is False
    assert not any("BOUND" in v.upper() for v in RP.all_names(mk_payload()).values())


def test_the_two_name_routines_agree_on_random_payloads_and_every_branch_is_reached():
    rnd = random.Random(121)
    kinds = ("lo", "hi", "zero", "edge_lo", "edge_hi")
    seen = set()
    for _ in range(600):
        new = {(S, pr, mk): rnd.choice(kinds) for S in RP.SETTINGS for pr in RP.PAIRS for mk in "CUF"}
        first = {k: rnd.choice(kinds) for k in new}
        pl = mk_payload(new, first, u_a90=rnd.choice((0.03, 0.05)), u_rlc=0.04)
        a, b = RP.names_of(pl), RP.derive_all(pl)
        assert a == b and RP.all_names(pl) == a
        seen.update(a.values())
    want = {P + X + "-" + w for X, *_ in RP.CLAIMS for w in ("REPLICATED", "NOT-SEPARATED-ON-THE-NEW-PAIR", "OPPOSITE")}
    want |= {"not a claim"} | {P + "MARGIN-" + w for w in ("REPLICATED", "REPLICATED-AT-K0-ONLY", "REPLICATED-AT-K30-ONLY", "NOT-REPLICATED")}
    assert want <= seen
    assert seen <= want


def test_all_names_refuses_when_the_routines_part(monkeypatch):
    pl = claim_payload({"M-K0": "rep"})
    monkeypatch.setattr(RP, "derive_all", lambda p: dict(RP.names_of(p), MARGIN="x"))
    with pytest.raises(SystemExit) as ex:
        RP.all_names(pl)
    assert "REFUSED" in str(ex.value)
    monkeypatch.setattr(RP, "derive_all", lambda p: {k: v for k, v in RP.names_of(p).items() if k != "P-K0"})
    with pytest.raises(SystemExit):
        RP.all_names(pl)


# ============================================================================================ e0 models
def run_text(key_run_m, ckpt):
    return LINE % (key_run_m[0], " rule every 5", "", key_run_m[1], key_run_m[1] - 2, ckpt)


def e0_inputs():
    rec = {"models": {"UE_s%d" % m: {"ckpt_sha256": H64 if m == 2 else "cd" * 32, "gate_pass": True, "recipe_same_but_seed": True, "record": "text %d" % m} for m in (2, 3)}, "note_sha256": "ef" * 32}
    shas = {2: H64, 3: "cd" * 32}
    texts = {k: run_text((nm, m), rec["models"]["UE_s%d" % m]["ckpt_sha256"]) for k, nm, S, m in RP.run_list()}
    return rec, shas, texts


def test_e0_passes():
    r = RP.ctl_e0(*e0_inputs())
    assert r["pass"] is True and r["failures"] == [] and len(e0_inputs()[2]) == 26


def test_e0_each_clause_fails():
    def fails(mut, frag):
        rec, shas, texts = e0_inputs()
        mut(rec, shas, texts)
        r = RP.ctl_e0(rec, shas, texts)
        assert r["pass"] is False and any(frag in f for f in r["failures"]), (frag, r["failures"])

    fails(lambda r, s, t: s.update({2: "00" * 32}), "UE_s2: the checkpoint on disk has sha256")
    fails(lambda r, s, t: s.update({3: None}), "UE_s3: no checkpoint file on disk")
    fails(lambda r, s, t: r["models"]["UE_s2"].update(gate_pass=False), "gate_pass")
    fails(lambda r, s, t: r["models"]["UE_s3"].update(gate_pass="true"), "gate_pass")
    fails(lambda r, s, t: r["models"]["UE_s3"].update(recipe_same_but_seed=False), "recipe_same_but_seed")
    fails(lambda r, s, t: r["models"]["UE_s2"].update(record=""), "no text")
    fails(lambda r, s, t: r["models"]["UE_s2"].update(ckpt_sha256="zz" * 32), "not 64 hex")
    fails(lambda r, s, t: r.update(note_sha256="ef" * 31), "note_sha256")
    fails(lambda r, s, t: r["models"].pop("UE_s3"), "UE_s3: no entry")
    fails(lambda r, s, t: r.pop("models"), "no entry")
    fails(lambda r, s, t: t.update({"K0_A90_s3": run_text(("RQ50KF5Hb20gA90", 3), "11" * 32)}), "K0_A90_s3: the first line names the checkpoint")
    fails(lambda r, s, t: t.update({"CONV_R0_s2": t["CONV_R0_s2"].replace("ckpt sha256", "ckpt hash")}), "CONV_R0_s2: the first line names no ckpt sha256")
    fails(lambda r, s, t: t.pop("K30_RR_s2"), "run texts read")


def test_e0_prefix_of_16_agrees_and_15_does_not():
    rec, shas, texts = e0_inputs()
    texts["K0_RR_s2"] = run_text(("x", 2), H64[:16])
    assert RP.ctl_e0(rec, shas, texts)["pass"] is True
    texts["K0_RR_s2"] = run_text(("x", 2), H64[:15])
    assert RP.ctl_e0(rec, shas, texts)["pass"] is False


def test_e0_a_record_that_is_no_dict_fails():
    for bad in (None, [], "x", {"models": []}):
        assert RP.ctl_e0(bad, {2: H64, 3: H64}, e0_inputs()[2])["pass"] is False


# ============================================================================================ e2 arguments
def d2_line(S, name, m, ckpt, drop=(), over=None, reader=None):
    ev = {"CONV": "", "K0": " rule every 5", "K30": " rule every 5"}[S]
    grid = {"CONV": "eval_grid_windows.json", "K0": "eval_grid_windows.json", "K30": "grid_p30.json"}[S]
    pre = " preroll %d exec plan nav deadreckon grid %s" % (RP.SPEC[S]["P"], grid)
    f = {"preroll": "", "grid": "", "nav": ""}
    ln = "arm I10%s_s%d form RR replan steps 10%s readout causal_g kappa 0.2%s model seed %d reader seed %d ckpt sha256 %s wrapper sha256 %s" % (
        name, m, ev, pre, m, ((m - 2) if m >= 2 else m) if reader is None else reader, ckpt, "28bc31f21ecb4ccb" + "0" * 48)
    for k in drop:
        ln = re.sub(r" %s \S+" % k, "", ln)
    for k, v in (over or {}).items():
        ln = re.sub(r"%s \S+" % k, "%s %s" % (k, v), ln)
    return ln + "\nrc 0 finished x"


def e2_lines():
    out = {}
    for k, nm, S, m in RP.run_list():
        out[k] = (d2_line(S, nm, m, "aa" * 32), d2_line(S, nm, m - 2, "bb" * 32))
    return out


def test_e2_passes_with_other_model_seed_and_checkpoint():
    r = RP.ctl_e2(e2_lines())
    assert r["pass"] is True and r["failures"] == [], r["failures"][:3]


def test_e2_fails_when_the_new_line_names_the_checkpoint_of_the_banked_run():
    L = e2_lines()
    key = "K0_GEc_s2"
    n, b = L[key]
    L[key] = (n.replace("aa" * 32, "bb" * 32), b)
    r = RP.ctl_e2(L)
    assert r["pass"] is False and any(f.startswith(key) and "ckpt: the new line names the checkpoint hash of the banked run" in f for f in r["failures"]), r["failures"][:3]


def test_key_shift_names_the_runs_of_the_card_at_the_second_pair_and_restores(monkeypatch):
    import types
    stub = types.SimpleNamespace(key_of=lambda run, k: "I10%s_s%d" % (run, k))
    monkeypatch.setattr(RP, "R", stub)
    first = stub.key_of
    with RP.KeyShift(["R0Hb20", "RQ50KF5Hb20gA90"], 2):
        assert stub.key_of("R0Hb20", 0) == "I10R0Hb20_s2" and stub.key_of("RQ50KF5Hb20gA90", 1) == "I10RQ50KF5Hb20gA90_s3" and stub.key_of("R0LrCarry", 0) == "I10R0LrCarry_s0"
    assert stub.key_of is first and stub.key_of("R0Hb20", 0) == "I10R0Hb20_s0"
    with RP.KeyShift(["R0Hb20"], 0):
        assert stub.key_of("R0Hb20", 1) == "I10R0Hb20_s1"


def test_e2_each_clause_fails():
    def fails(mut, frag, key="K0_GEc_s2"):
        L = e2_lines()
        n, b = L[key]
        L[key] = mut(n, b)
        r = RP.ctl_e2(L)
        assert r["pass"] is False and any(frag in f and f.startswith(key) for f in r["failures"]), (frag, r["failures"][:4])

    fails(lambda n, b: (n.replace("kappa 0.2", "kappa 0.3"), b), "kappa: new 0.3")
    fails(lambda n, b: (n.replace("replan steps 10", "replan steps 5"), b), "replan: new 5")
    fails(lambda n, b: (n.replace("rule every 5", "rule every 10"), b), "every: new 10")
    fails(lambda n, b: (n, b.replace("readout causal_g", "readout K")), "readout: new causal_g")
    fails(lambda n, b: (n.replace("model seed 2", "model seed 3"), b), "model seed: the new line reads 3, expected 2")
    fails(lambda n, b: (n.replace("reader seed 0", "reader seed 2"), b), "reader seed: the new line reads 2, expected 0")
    fails(lambda n, b: (n, b.replace("reader seed 0", "reader seed 1")), "reader_seed: the banked line reads 1")
    fails(lambda n, b: (n, b.replace("model seed 0", "model seed 5")), "model_seed: the banked line reads 5")
    fails(lambda n, b: (re.sub(r" model seed \S+", "", n), b), "the new line carries no model_seed")
    fails(lambda n, b: (re.sub(r" reader seed \S+", "", n), b), "the new line carries no reader_seed")
    fails(lambda n, b: (re.sub(r" kappa \S+", "", n), b), "the banked line carries it (0.2), the new line does not")
    fails(lambda n, b: (n.replace("exec plan", "exec other"), b), "exec: new other")
    fails(lambda n, b: (n, "\n"), "no run log of the banked run")
    fails(lambda n, b: ("\n", b), "no run log of the new run")


def test_e2_fields_both_carry_must_agree_but_ckpt_and_model_seed_may_differ():
    L = e2_lines()
    n, b = L["K30_RR_s3"]
    assert RP.ctl_e2(L)["pass"] is True
    L["K30_RR_s3"] = (n.replace("ckpt sha256 " + "aa" * 32, "ckpt sha256 " + "cc" * 32), b)
    assert RP.ctl_e2(L)["pass"] is True


def test_e2_defaults_where_the_banked_line_lacks_the_fields():
    L = e2_lines()
    n, b = L["K0_A90_s3"]
    L["K0_A90_s3"] = (n, re.sub(r" preroll \S+ exec plan nav deadreckon grid \S+", "", b))                 # banked line without the optional fields
    assert RP.ctl_e2(L)["pass"] is True
    L["K0_A90_s3"] = (n.replace("preroll 0", "preroll 30"), L["K0_A90_s3"][1])
    r = RP.ctl_e2(L)
    assert r["pass"] is False and any("the new line has to read preroll 0" in f for f in r["failures"])
    L = e2_lines()
    n, b = L["K30_GEc_s2"]
    bare = re.sub(r" preroll \S+ exec plan nav deadreckon grid \S+", "", b)                              # at K30 the default is the pre-roll of the setting and its own grid
    L["K30_GEc_s2"] = (n, bare)
    assert RP.ctl_e2(L)["pass"] is True
    L["K30_GEc_s2"] = (n.replace("preroll 30", "preroll 0").replace("grid_p30.json", "eval_grid_windows.json"), bare)
    r = RP.ctl_e2(L)
    assert r["pass"] is False and any("has to read preroll 30" in f for f in r["failures"]) and any("has to read grid grid_p30.json" in f for f in r["failures"])
    L["K30_GEc_s2"] = (re.sub(r" preroll \S+ exec plan nav deadreckon grid \S+", "", n), bare)
    r = RP.ctl_e2(L)
    assert any("has to read preroll 30" in f for f in r["failures"]) and any("has to read nav deadreckon" in f for f in r["failures"])


def test_e2_run_set_is_checked():
    L = e2_lines()
    L.pop("CONV_R0_s3")
    r = RP.ctl_e2(L)
    assert r["pass"] is False and any("run lines read" in f for f in r["failures"])


# ============================================================================================ e6 counts
def e6_inputs():
    done = {k: True for k in RP.all_run_keys()}
    nd = {k: 141 for k in RP.all_run_keys()}
    leads = {S: {m: RP.EXPECT_LEADS[S] for m in (2, 3)} for S in RP.SETTINGS}
    return done, nd, leads


def test_e6_passes():
    d, n, l_ = e6_inputs()
    r = RP.ctl_e6(d, n, 141, "x\nZZ REPL10K DONE 2026-11-22T20:00:00Z all steps done\n", l_)
    assert r["pass"] is True and r["failures"] == []
    assert RP.ctl_e6(d, n, 141, None, l_)["pass"] is True


def test_e6_each_clause_fails():
    d, n, l_ = e6_inputs()
    ok_marker = "ZZ REPL10K DONE 2026-11-22T20:00:00Z all steps done"
    d2 = dict(d)
    d2["K30_A90_s3"] = False
    r = RP.ctl_e6(d2, n, 141, ok_marker, l_)
    assert r["pass"] is False and any("K30_A90_s3: no T1_DONE" in f for f in r["failures"])
    d2 = dict(d)
    d2.pop("CONV_R0_s2")
    assert any("CONV_R0_s2: no T1_DONE" in f for f in RP.ctl_e6(d2, n, 141, ok_marker, l_)["failures"])
    n2 = dict(n)
    n2["K0_RR_s2"] = 140
    assert any("K0_RR_s2: 140 dump files" in f for f in RP.ctl_e6(d, n2, 141, ok_marker, l_)["failures"])
    n2 = dict(n)
    n2["CONV_R0_s3"] = None
    assert any("CONV_R0_s3: None dump files" in f for f in RP.ctl_e6(d, n2, 141, ok_marker, l_)["failures"])
    for bad in ("ZZ GAPMARGIN10K DONE x all steps done", "ZZ REPL10K DONE x finished", ""):
        assert any("done marker" in f for f in RP.ctl_e6(d, n, 141, bad, l_)["failures"]), bad
    for S, want in (("CONV", 642), ("K0", 642), ("K30", 325)):
        l2 = {s: dict(v) for s, v in l_.items()}
        l2[S][3] = want + 1
        r = RP.ctl_e6(d, n, 141, ok_marker, l2)
        assert r["pass"] is False and any(("%s model seed 3: %d windows" % (S, want + 1)) in f for f in r["failures"])
    l2 = {s: dict(v) for s, v in l_.items()}
    l2["K30"][2] = 642
    assert any("K30 model seed 2" in f for f in RP.ctl_e6(d, n, 141, ok_marker, l2)["failures"])
    l2 = {"CONV": l_["CONV"], "K0": l_["K0"]}
    assert any("K30 model seed 2: None" in f for f in RP.ctl_e6(d, n, 141, ok_marker, l2)["failures"])


# ============================================================================================ e1 and the kept controls
def test_e1_needs_both_parts():
    ok = {"pass": True, "failures": [], "values": {"n": 1}}
    assert RP.ctl_e1(ok, dict(ok, wrapper_sha16="28bc31f21ecb4ccb"))["pass"] is True
    r = RP.ctl_e1({"pass": False, "failures": ["gone"]}, ok)
    assert r["pass"] is False and r["failures"] == ["gap files: gone"]
    r = RP.ctl_e1(ok, {"pass": False, "failures": ["other"]})
    assert r["pass"] is False and r["failures"] == ["wrapper: other"]
    assert RP.ctl_e1({"pass": False, "failures": []}, ok)["pass"] is False


def test_kept_gap_file_control_reads_a90_only():
    rec = {"pass_k0": True, "scale_one_gives_the_base_arrays": True, "base": {"sha256": "55de2f7c" + "0" * 56},
           "clauses": {"VGAPLE_A90": {"a": True, "b": True}}, "files": {"VGAPLE_A90": {"sha256": "ab" * 32, "a": 0.9, "n_rows_base_gap_above_60_and_scaled_at_most_60": 3}}, "counts": {}}
    r = RP.ctl_gap(rec, "55de2f7c" + "0" * 56, {"A90": "ab" * 32})
    assert r["pass"] is True and r["values"]["n_rows_newly_in_range"] == {"A90": 3}
    assert RP.ctl_gap(rec, "55de2f7c" + "0" * 56, {"A90": "cd" * 32})["pass"] is False


def test_kept_identity_control_covers_the_three_settings():
    lv = {S: {arm: dict(zip("CUF", v)) for arm, v in RP.EXPECTED[S].items()} for S in RP.SETTINGS}
    assert RP.ctl_e3(lv)["pass"] is True
    lv["K30"]["A90"]["U"] += 0.0002
    r = RP.ctl_e3(lv)
    assert r["pass"] is False and any("K30 A90 U" in f for f in r["failures"])
    lv["K30"]["A90"]["U"] -= 0.0002
    lv.pop("CONV")
    assert RP.ctl_e3(lv)["pass"] is False


# ============================================================================================ seeds of the pod part, printing, refusals
def test_the_seed_context_sets_and_restores_the_engine_globals():
    class Fake(object):
        SEEDS = (0, 1)
    old_r = RP.R
    try:
        RP.R = Fake
        with RP.SeedsCtx((2, 3)):
            assert Fake.SEEDS == (2, 3)
            with RP.SeedsCtx((0, 1, 2, 3)):
                assert Fake.SEEDS == (0, 1, 2, 3)
            assert Fake.SEEDS == (2, 3)
        assert Fake.SEEDS == (0, 1)
    finally:
        RP.R = old_r


def test_print_sections_prints_names_bound_levels_three_estimates_and_the_spread():
    pl = claim_payload({"M-K0": "rep", "M-K30": "rep"})
    for S in RP.SETTINGS:
        d = pl["settings"][S]
        d["levels_four"] = d["levels"]
        d["levels_per_seed"] = {arm: {c: [0.1, 0.2, 0.3, 0.4] for c in "CUF"} for arm in RP.ARMS}
        d["levels"]["GEc"]["U"] = 0.07
        d["counts"] = {"n_windows": 1, "n_lead_windows": RP.EXPECT_LEADS[S], "n_episodes_with_lead": 3}
        d["est"]["new"]["A90-VS-GEc"]["U"].update(per_seed=[0, 0], pooled=-0.01)
        for en in ("new", "first", "four"):
            for nm in RP.PAIRS:
                for m in "CUF":
                    d["est"][en][nm][m].setdefault("per_seed", [0.0, 0.0])
                    d["est"][en][nm][m].setdefault("pooled", 0.0)
        d["pairs"] = d["est"]["new"]
        d["families"], d["strategic"] = {}, {"pairs": {}}
    pl["control_values"] = {"e7": {"rollout_shares": {"K0_GEc_s2": {"n_rows": 5}}}}
    names = RP.all_names(pl)
    L = RP.print_sections(pl, names)
    txt = "\n".join(L)
    assert L[0].startswith("TIER T1-kin/hb0.2") and P + "MARGIN-REPLICATED" in L
    assert "bound, no name: U(A90) <= U(RLc) + 0.0045 at K0 on the new pair: True" in txt
    for S in RP.SETTINGS:
        for nm in RP.PAIRS:
            for en in ("new", "first", "four"):
                assert "%s %s %s:" % (S, nm, en) in txt
        for arm in RP.ARMS:
            assert "spread between pairs %s %s" % (S, arm) in txt
    assert "spread between pairs CONV GEc, pooled level of the new pair minus that of the first pair, no name: C +0.0000 U +0.0200 F +0.0000" in txt
    assert "rollout K0_GEc_s2" in txt and "pre-roll of 30 steps (6 s)" in txt and "no pre-roll" in txt


def test_refusals_of_the_reader(tmp_path):
    base = [sys.executable, READER]
    r = subprocess.run(base + ["--selftest"], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 2 and "REFUSED" in r.stdout
    r = subprocess.run(base + ["--stage", "controls"], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 2 and "REFUSED" in r.stdout
    out = tmp_path / "o.json"
    out.write_text("x")
    r = subprocess.run(base + ["--stage", "names", "--out", str(out), "--model-reader", ""], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 2 and "exists" in r.stdout
    ctl = tmp_path / "c.json"
    ctl.write_text(json.dumps({"card": RP.CARD, "all_pass": True, "payload": {}, "controls": {"e%d" % i: {"pass": True} for i in range(7)}}))
    r = subprocess.run(base + ["--stage", "names", "--out", str(tmp_path / "n.json"), "--controls", str(ctl)], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 3 and "ZZ READ-REPL10K names refused" in r.stdout and "e0 to e7" in r.stdout
    assert not (tmp_path / "n.json").exists()
    ctl.write_text(json.dumps({"card": RP.CARD, "all_pass": True, "payload": {}, "controls": dict({"e%d" % i: {"pass": True} for i in range(7)}, e7={"pass": False})}))
    r = subprocess.run(base + ["--stage", "names", "--out", str(tmp_path / "n.json"), "--controls", str(ctl)], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 3


# ============================================================================================ driver, pull script, wording
@pytest.mark.skipif(BASH is None, reason="no bash")
@pytest.mark.parametrize("path", [DRIVER, PULL])
def test_bash_syntax(path):
    r = subprocess.run([BASH, "-n", path], capture_output=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr[-300:]


def test_driver_text():
    t = rd(DRIVER)
    assert t.startswith("#!/usr/bin/env bash") and "GE=$V/repl; K=$GE/kit_read; OUT=$GE/read; LT=$GE/lead_tests; GFM=$V/gapmargin/gapfiles" in t
    for must in ("REPL10K_START.txt", "REPL10K_DONE.txt", "REPL10K_STOP.txt", "REPL10K_CONTROLS.json", "REPL10K_RESULT.json", "read_names.log", "read_names.out", "ZZ PODREAD-REPL10K",
                 "--models-record \"$MREC\"", "MREC=$GE/models/REPL_MODELS_RECORD.json", "for s in 0 1 2 3; do", "--grid-dir \"$W/longroll\"", "read_repl10k.py", "$LT/lead_pytest_queue.out",
                 "$LT/lead_pytest_read.out", "grep -E '^e[0-7] (PASS|FAIL)|^ZZ READ-REPL10K'", "I10R0Hb20", "lead_pytest_read.out"):
        assert must in t, must
    run_line = re.search(r"  for x in (.*); do", t).group(1).split()
    want = []
    for S in RP.SETTINGS:
        want += ["I10" + RP.SPEC[S]["runs"][a] for a in RP.ARMS]
    assert sorted(run_line) == sorted(want + ["I10R0Hb20"]) and len(run_line) == 13
    assert "A80" not in t and "GAPMARGIN10K" not in t and "READ-GAPMARGIN" not in t
    assert t.count("REPL10K_START.txt") >= 2 and "head -n 1 \"$GE/REPL10K_START.txt\" | awk '{print $1}'" in t


def test_pull_script_text():
    t = rd(PULL)
    assert "D=/f/TanitAD-scratch/v8proof/repl" in t and "ROOT=/workspace/v8proof/repl" in t and "pod_add121" in t and "tmp_pull_add121.txt" in t and "tmp_pull_add121.tar" in t
    assert 'FILES="read/*.json read/*.log read/*.out read/*.txt repl_queue.log small_rp/* models/*.json REPL10K_START.txt REPL10K_DONE.txt REPL10K_STOP.txt lead_tests/lead_pytest_queue.out lead_tests/lead_pytest_read.out repl_queue.out"' in t
    assert "*.npz|*.pt) continue" in t and "sed '/hf_/d'" in t and "mkdir \"$D/pod_add121\"" in t and "rm " not in t and "mv " not in t and "pull_add121_" in t


def test_wording_of_the_kit_files():
    banned = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lon" + "ger", "closed" + " loop"]
    names = [f for f in os.listdir(HERE) if os.path.isfile(os.path.join(HERE, f)) and f.split(".")[-1] in ("py", "sh", "md", "out")]
    assert {"make_repl_reader.py", "read_repl10k.py", "podread_repl10k.sh", "pull_repl.sh", "test_repl_read.py"} <= set(names)
    for f in names:
        t = rd(os.path.join(HERE, f)).lower()
        for w in banned:
            assert w not in t, (f, w)
    for f in names:
        assert not re.match(r"^(aux|con|prn|nul)(\.|$)", f.lower())
        assert len(os.path.join(HERE, f)) <= 200
    for f in ("read_repl10k.py", "podread_repl10k.sh", "make_repl_reader.py", "pull_repl.sh"):
        assert "\r" not in rd(os.path.join(HERE, f))
