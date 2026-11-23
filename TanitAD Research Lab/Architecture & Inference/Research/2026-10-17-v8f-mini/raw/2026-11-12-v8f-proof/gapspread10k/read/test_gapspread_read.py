"""test_gapspread_read.py -- card ADD134 (GAPSPREAD10K), reader kit (helper-written): the generated reader and driver against their generator, the refusals of the generator, the tables against the
card, the names on hand-made payloads (every branch of section 4 and the agreement of the name routines with a third derivation written here), the controls p0 to p7 on tiny synthetic worlds (each
clause passing and failing, every control failing alone), the stages of the reader as child processes, the driver and the pull script, and the wording of the kit files. The source kit of card ADD121
is taken from ../../repl/read (tests that need it are skipped where it is absent). The working folder of the child processes is HERE, or the folder named by the environment variable
GAPSPREAD_TEST_CWD. No pod result file is opened; every array and every text here is a few rows."""
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
CWD = os.environ.get("GAPSPREAD_TEST_CWD") or HERE
GEN = os.path.join(HERE, "make_gapspread_reader.py")
READER = os.path.join(HERE, "read_gapspread10k.py")
DRIVER = os.path.join(HERE, "podread_gapspread10k.sh")
PULL = os.path.join(HERE, "pull_gapspread10k.sh")
CARD_FILE = os.path.join(HERE, "..", "..", "PREREG_V8F_PROOF_ADD134.md")
SRC_KIT = os.path.join(HERE, "..", "..", "repl", "read")
HAVE_SRC = all(os.path.isfile(os.path.join(SRC_KIT, f)) for f in ("read_repl10k.py", "podread_repl10k.sh"))
BASH = shutil.which("bash")
sys.path.insert(0, HERE)
import make_gapspread_reader as MK  # noqa: E402
import read_gapspread10k as GS  # noqa: E402

P = "GAPSPREAD10K-K0-"
WRAP = "28bc31f21ecb4ccb" + "0" * 48
need_src = pytest.mark.skipif(not HAVE_SRC, reason="the source kit of card ADD121 is not beside this kit")
need_md = pytest.mark.skipif(GS.MD is None, reason="the model reader is not found")


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def run_py(args, **kw):
    return subprocess.run([sys.executable] + args, capture_output=True, encoding="utf-8", timeout=180, cwd=CWD, **kw)


# ============================================================================================ generator
@need_src
@pytest.mark.parametrize("kind,path", [("reader", READER), ("driver", DRIVER)])
def test_generated_file_is_what_the_generator_prints(kind, path):
    r = run_py([GEN, "--print", kind])
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.replace("\r\n", "\n") == rd(path)
    assert "\r" not in rd(path)


def _gen_world(tmp_path, change_source=False, existing_output=False):
    """a copy of the generator in tmp/x/y and copies of the two source files in tmp/repl/read"""
    gdir = tmp_path / "x" / "y"
    gdir.mkdir(parents=True)
    sdir = tmp_path / "repl" / "read"
    sdir.mkdir(parents=True)
    shutil.copy(GEN, str(gdir / "make_gapspread_reader.py"))
    for nm in ("read_repl10k.py", "podread_repl10k.sh"):
        t = open(os.path.join(SRC_KIT, nm), "rb").read()
        if change_source and nm.endswith(".py"):
            t = t + b"# one more line\n"
        (sdir / nm).write_bytes(t)
    if existing_output:
        (gdir / "read_gapspread10k.py").write_text("x")
    return str(gdir / "make_gapspread_reader.py")


@need_src
def test_generator_runs_on_a_copy_and_writes_the_same_files(tmp_path):
    g = _gen_world(tmp_path)
    r = run_py([g])
    assert r.returncode == 0, r.stderr[-500:]
    d = os.path.dirname(g)
    assert rd(os.path.join(d, "read_gapspread10k.py")) == rd(READER) and rd(os.path.join(d, "podread_gapspread10k.sh")) == rd(DRIVER)


@need_src
def test_generator_refuses_a_changed_source(tmp_path):
    g = _gen_world(tmp_path, change_source=True)
    r = run_py([g])
    assert r.returncode != 0 and "source sha256 prefix" in (r.stderr + r.stdout)
    assert not os.path.exists(os.path.join(os.path.dirname(g), "read_gapspread10k.py"))


@need_src
def test_generator_refuses_an_existing_output(tmp_path):
    g = _gen_world(tmp_path, existing_output=True)
    r = run_py([g])
    assert r.returncode != 0 and "exists" in (r.stderr + r.stdout)
    assert not os.path.exists(os.path.join(os.path.dirname(g), "podread_gapspread10k.sh"))


def test_substitution_helper_refuses_an_anchor_that_is_not_found_once():
    with pytest.raises(SystemExit) as ex:
        MK.Sub("abc").once("x", "y")
    assert "expected once, found 0" in str(ex.value)
    with pytest.raises(SystemExit) as ex:
        MK.Sub("abab").once("ab", "y")
    assert "expected once, found 2" in str(ex.value)
    with pytest.raises(SystemExit):
        MK.Sub("b a").span("a", "b", "z")                                           # anchors in the wrong order
    with pytest.raises(SystemExit):
        MK.Sub("a a b").span("a", "b", "z")                                         # an anchor twice
    with pytest.raises(SystemExit):
        MK.Sub("abc").line_from("q", "z")
    s = MK.Sub("one\ntwo\nthree")
    s.line_from("two", "2")
    assert s.t == "one\n2\nthree"


# ============================================================================================ tables
def test_tables():
    assert GS.CARD == "PREREG_V8F_PROOF_ADD134" and GS.PRE == "GAPSPREAD10K" and GS.SETTINGS == ("K0",) and GS.SEEDS == (0, 1, 2, 3) and GS.SPREAD == "R0Hb20"
    assert GS.ARMS == ("SP90", "A90", "GEc", "RLc", "RR") and GS.NEW_ARMS == {"K0": ("SP90",)} and GS.NEW_FILES == {"SP90": "VGAPSP90"}
    assert GS.SPEC["K0"]["runs"] == {"SP90": "RQ50KF5Hb20gSP90", "A90": "RQ50KF5Hb20gA90", "GEc": "RQ50KF5Hb20gGEc", "RLc": "RLcKF5Hb20", "RR": "RQ50KF5Hb20"}
    assert GS.SPEC["K0"]["every"] == 5 and GS.SPEC["K0"]["readout"] == "K" and GS.SPEC["K0"]["P"] == 0 and GS.SPEC["K0"]["grid"] == "full" and GS.SPEC["K0"]["grid_file"] == "pinned"
    assert GS.PAIRS == {"SP90-VS-A90": ("SP90", "A90"), "SP90-VS-GEc": ("SP90", "GEc"), "SP90-VS-RLc": ("SP90", "RLc"), "A90-VS-GEc": ("A90", "GEc")}
    assert GS.NAMED == (("VS-FLAT", "SP90-VS-A90"), ("VS-CANDIDATE", "SP90-VS-GEc")) and sorted(GS.STRAT_PAIRS) == ["SP90-VS-A90", "SP90-VS-GEc"]
    assert GS.MEASURES == (("U", "UNSAFE"), ("C", "CONTACT"), ("F", "FALLING-BEHIND")) and GS.HALF_BOUND == 0.0058
    assert GS.EXPECTED == {"K0": {"GEc": (0.0000, 0.0480, 0.1589), "A90": (0.0000, 0.0365, 0.1651), "RLc": (0.0004, 0.0369, 0.1460), "RR": (0.0300, 0.0703, 0.1589)}}
    assert GS.EXPECTED_PAIR == {"A90-VS-GEc": {"U": (-0.0116, -0.0175, -0.0063), "F": (0.0062, 0.0019, 0.0112)}}
    assert GS.LEVEL_TOL == 1e-4 and GS.EXPECT_DUMPS == 141 and GS.WRAPPER_SHA == "28bc31f21ecb4ccb"
    assert GS.CONTROLS == tuple("p%d" % i for i in range(8))
    assert GS.DONE_RE.match("ZZ GAPSPREAD10K DONE 2026-11-23T16:00:00Z all steps done") and not GS.DONE_RE.match("ZZ REPL10K DONE x all steps done")
    assert [k for k, nm, S, m in GS.run_list()] == ["K0_SP90_s0", "K0_SP90_s1", "K0_SP90_s2", "K0_SP90_s3"] and GS.new_keys() == GS.all_run_keys() == GS.rule_keys()
    assert [nm for k, nm, S, m in GS.run_list()] == ["RQ50KF5Hb20gSP90"] * 4 and GS.ref_arm("K0", "SP90") == "A90"
    assert GS.TIER.startswith("T1-kin/hb0.2") and "driving performance" not in GS.TIER


@pytest.mark.skipif(not os.path.isfile(CARD_FILE), reason="the card file is not beside the kit on this machine")
def test_tables_against_the_card_text():
    card = rd(CARD_FILE)
    for arm, nm in GS.SPEC["K0"]["runs"].items():
        assert ("`I10%s_s0`" % nm in card) if arm == "SP90" else ("`I10%s`" % nm in card), nm
    row = re.search(r"\* \*\*p3 identity\*\*.*", card).group(0).replace("**", "")
    for arm in ("GEc", "A90", "RLc", "RR"):
        m = re.search(arm + r" ([0-9.]+) / ([0-9.]+) / ([0-9.]+)", row)
        assert m and tuple(float(x) for x in m.groups()) == GS.EXPECTED["K0"][arm], arm
    assert "unsafe -0.0116 [-0.0175, -0.0063]" in row and "falling behind +0.0062 [+0.0019, +0.0112]" in row
    assert "+0.0058" in card and "0.0116" in card and "2,000 draws" in card and "141 dump files" in card and "PREREG_V8F_PROOF_ADD134" in card
    for tail in ("GAPSPREAD10K-K0-PLACED-GIVES-UP-SAFETY", "GAPSPREAD10K-K0-PLACED-SAME-SAFETY-AT-A-LOWER-PRICE", "GAPSPREAD10K-K0-PLACED-SAFER-AT-NO-HIGHER-PRICE",
                 "GAPSPREAD10K-K0-PLACED-NOT-SEPARATED-FROM-FLAT"):
        assert tail in card, tail
    for w in ("-LOWERS-<M>", "-RAISES-<M>", "-<M>-NOT-SEPARATED", "VS-FLAT", "VS-CANDIDATE", "GAPSPREAD10K-K0-<P>"):
        assert w in card, w


# ============================================================================================ names
KINDS = {"lo": [-0.02, -0.005], "hi": [0.005, 0.02], "zero": [-0.01, 0.01], "edge_lo": [-0.01, 0.0], "edge_hi": [0.0, 0.01]}


def mk_payload(over=None):
    """a payload for the names: over {(pair, measure): kind name or an interval}; every other interval is 'zero' (it includes 0)"""
    pairs = {}
    for nm in GS.PAIRS:
        pairs[nm] = {m: {"interval": list(KINDS["zero"]), "pooled": 0.0, "per_seed": [0.0] * 4} for m in "CUF"}
    for (pk, mk), iv in (over or {}).items():
        pairs[pk][mk]["interval"] = list(KINDS[iv] if isinstance(iv, str) else iv)
    return {"settings": {"K0": {"pairs": pairs}}}


def ov(u=None, c=None, f=None, pair="SP90-VS-A90"):
    return {(pair, m): v for m, v in (("U", u), ("C", c), ("F", f)) if v is not None}


def merge(*ds):
    out = {}
    for d in ds:
        out.update(d)
    return out


def overall(**kw):
    return GS.all_names(mk_payload(ov(**kw)))["OVERALL"]


def test_pair_names_per_measure_and_pair():
    for P_, pk in GS.NAMED:
        for mk, mw in GS.MEASURES:
            assert GS.all_names(mk_payload({(pk, mk): "lo"}))["%s_%s" % (P_, mw)] == P + P_ + "-LOWERS-" + mw
            assert GS.all_names(mk_payload({(pk, mk): "hi"}))["%s_%s" % (P_, mw)] == P + P_ + "-RAISES-" + mw
            assert GS.all_names(mk_payload({(pk, mk): "zero"}))["%s_%s" % (P_, mw)] == P + P_ + "-" + mw + "-NOT-SEPARATED"
    assert GS.all_names(mk_payload())["VS-FLAT_UNSAFE"] == "GAPSPREAD10K-K0-VS-FLAT-UNSAFE-NOT-SEPARATED"
    assert GS.all_names(mk_payload(ov(u="lo")))["VS-FLAT_UNSAFE"] == "GAPSPREAD10K-K0-VS-FLAT-LOWERS-UNSAFE"


def test_every_name_has_seven_entries_and_the_other_pairs_carry_no_name():
    nm = GS.all_names(mk_payload(ov(u="lo", pair="SP90-VS-RLc")))
    assert len(nm) == 7 and sorted(nm) == sorted(["VS-FLAT_UNSAFE", "VS-FLAT_CONTACT", "VS-FLAT_FALLING-BEHIND", "VS-CANDIDATE_UNSAFE", "VS-CANDIDATE_CONTACT", "VS-CANDIDATE_FALLING-BEHIND", "OVERALL"])
    assert nm == GS.all_names(mk_payload())
    nm2 = GS.all_names(mk_payload(merge(ov(u="hi", c="hi", f="lo", pair="A90-VS-GEc"), ov(u="lo", pair="SP90-VS-RLc"))))
    assert nm2 == nm


def test_overall_clause_1_safety_given_up():
    assert overall(u="hi") == P + "PLACED-GIVES-UP-SAFETY"
    assert overall(c="hi") == P + "PLACED-GIVES-UP-SAFETY"
    assert overall(u="hi", c="hi", f="lo") == P + "PLACED-GIVES-UP-SAFETY"                # clause 1 comes first although falling behind is lower
    assert overall(u=[0.0001, 0.02], f="lo") == P + "PLACED-GIVES-UP-SAFETY"
    assert overall(u="lo", c="hi", f="lo") == P + "PLACED-GIVES-UP-SAFETY"                 # a raised contact rate gives up safety whatever the unsafe rate does


def test_overall_clause_2_same_safety_at_a_lower_price():
    same = P + "PLACED-SAME-SAFETY-AT-A-LOWER-PRICE"
    assert overall(u=[-0.01, 0.0058], f="lo") == same                                       # the bound itself is inside
    assert overall(u=[-0.01, 0.00579], f="lo") == same
    assert overall(u=[-0.01, 0.005], f="lo") == same
    assert overall(u="lo", f="lo") == same                                                  # clause 2 comes before clause 3
    assert overall(u="edge_lo", c="edge_hi", f="lo") == same
    assert overall(u=[-0.01, 0.0058], c="zero", f="lo") == same


def test_overall_clause_2_needs_every_part():
    nsf = P + "PLACED-NOT-SEPARATED-FROM-FLAT"
    assert overall(u=[-0.01, 0.00581], f="lo") == nsf                                       # just above the bound
    assert overall(u=[-0.01, 0.0058], f="zero") == nsf                                      # falling behind not below 0
    assert overall(u=[-0.01, 0.0058], f="edge_lo") == nsf                                   # its upper end is 0.0: not below 0
    assert overall(u=[-0.01, 0.0058], f="hi") == nsf
    assert overall(u="zero", f="lo") == nsf                                                 # an interval of +-0.01 has an upper end above 0.0058


def test_overall_clause_3_safer_at_no_higher_price():
    safer = P + "PLACED-SAFER-AT-NO-HIGHER-PRICE"
    assert overall(u="lo") == safer
    assert overall(u="lo", f="zero") == safer
    assert overall(u="lo", f="edge_hi") == safer                                            # a lower end of 0.0 is not above 0
    assert overall(u="lo", f="hi") == P + "PLACED-NOT-SEPARATED-FROM-FLAT"                  # falling behind above 0: clause 3 does not hold
    assert overall(u="lo", f=[0.0001, 0.02]) == P + "PLACED-NOT-SEPARATED-FROM-FLAT"


def test_overall_clause_4_and_edges_at_zero():
    nsf = P + "PLACED-NOT-SEPARATED-FROM-FLAT"
    assert overall() == nsf
    assert overall(u="edge_hi") == nsf                                                      # a lower end of 0.0 is not above 0
    assert overall(c="edge_hi") == nsf
    assert overall(u="edge_lo") == nsf                                                      # an upper end of 0.0 is not below 0 and the falling-behind interval includes 0
    assert overall(f="hi") == nsf and overall(f="lo") == nsf                               # falling behind alone never decides: the unsafe interval of +-0.01 is above the bound
    assert overall(u="zero", c="zero", f="hi") == nsf


def test_the_overall_name_reads_sp90_minus_a90_only():
    base = ov(u="lo", f="lo")
    other = merge(*[ov(u="hi", c="hi", f="hi", pair=pk) for pk in ("SP90-VS-GEc", "SP90-VS-RLc", "A90-VS-GEc")])
    assert GS.all_names(mk_payload(merge(base, other)))["OVERALL"] == P + "PLACED-SAME-SAFETY-AT-A-LOWER-PRICE"
    assert GS.all_names(mk_payload(merge(base, other)))["VS-CANDIDATE_UNSAFE"] == P + "VS-CANDIDATE-RAISES-UNSAFE"


def oracle(pl):
    """a third derivation, written in the test from the text of the card: the first clause that holds, on the intervals of SP90 - A90"""
    pr = pl["settings"]["K0"]["pairs"]
    (ulo, uhi), (clo, chi), (flo, fhi) = (pr["SP90-VS-A90"][m]["interval"] for m in "UCF")
    out = {}
    for P_, pk in (("VS-FLAT", "SP90-VS-A90"), ("VS-CANDIDATE", "SP90-VS-GEc")):
        for m, w in (("U", "UNSAFE"), ("C", "CONTACT"), ("F", "FALLING-BEHIND")):
            lo, hi = pr[pk][m]["interval"]
            out[(P_, w)] = (P + P_ + "-LOWERS-" + w) if hi < 0 else ((P + P_ + "-RAISES-" + w) if lo > 0 else (P + P_ + "-" + w + "-NOT-SEPARATED"))
    clauses = [("PLACED-GIVES-UP-SAFETY", ulo > 0 or clo > 0),
               ("PLACED-SAME-SAFETY-AT-A-LOWER-PRICE", uhi <= 0.0058 and not clo > 0 and fhi < 0),
               ("PLACED-SAFER-AT-NO-HIGHER-PRICE", uhi < 0 and not flo > 0),
               ("PLACED-NOT-SEPARATED-FROM-FLAT", True)]
    out["OVERALL"] = P + next(name for name, ok in clauses if ok)
    return out


def test_the_name_routines_agree_with_a_third_derivation_and_every_name_is_reached():
    rnd = random.Random(134)
    cands = [[-0.02, -0.005], [0.005, 0.02], [-0.01, 0.01], [-0.01, 0.0], [0.0, 0.01], [-0.01, 0.0058], [-0.01, 0.00581], [-0.004, 0.0057], [-0.03, -0.0001], [0.0001, 0.03]]
    seen = set()
    for _ in range(3000):
        over = {(pk, mk): rnd.choice(cands) for pk in GS.PAIRS for mk in "CUF"}
        pl = mk_payload(over)
        a, b = GS.names_of(pl), GS.derive_all(pl)
        o = oracle(pl)
        assert a == b and GS.all_names(pl) == a
        assert a["OVERALL"] == o["OVERALL"] and all(a["%s_%s" % k] == v for k, v in o.items() if k != "OVERALL")
        seen.update(a.values())
    want = {P + P_ + "-LOWERS-" + w for P_ in ("VS-FLAT", "VS-CANDIDATE") for _m, w in GS.MEASURES}
    want |= {P + P_ + "-RAISES-" + w for P_ in ("VS-FLAT", "VS-CANDIDATE") for _m, w in GS.MEASURES}
    want |= {P + P_ + "-" + w + "-NOT-SEPARATED" for P_ in ("VS-FLAT", "VS-CANDIDATE") for _m, w in GS.MEASURES}
    want |= {P + t for t in ("PLACED-GIVES-UP-SAFETY", "PLACED-SAME-SAFETY-AT-A-LOWER-PRICE", "PLACED-SAFER-AT-NO-HIGHER-PRICE", "PLACED-NOT-SEPARATED-FROM-FLAT")}
    assert len(want) == 22
    assert seen == want


def test_all_names_refuses_when_the_routines_part(monkeypatch):
    pl = mk_payload(ov(u="lo"))
    monkeypatch.setattr(GS, "derive_all", lambda p: dict(GS.names_of(p), OVERALL="x"))
    with pytest.raises(SystemExit) as ex:
        GS.all_names(pl)
    assert "REFUSED" in str(ex.value)
    monkeypatch.setattr(GS, "derive_all", lambda p: {k: v for k, v in GS.names_of(p).items() if k != "OVERALL"})
    with pytest.raises(SystemExit):
        GS.all_names(pl)


# ============================================================================================ p0, p2, p3, p4, p5, p6
H64 = "ab" * 32


def rec_ok(**kw):
    r = {"card": "PREREG_V8F_PROOF_ADD134", "pass_p0": True, "files": {"VGAPSP90": {"sha256": H64}}, "counts": {"n_rows_out_of_range": 7}}
    r.update(kw)
    return r


def test_p0_passes_and_each_clause_fails():
    r = GS.ctl_p0(rec_ok(), {"VGAPSP90": H64})
    assert r["pass"] is True and r["failures"] == [] and r["values"]["gap_file_sha16"] == {"VGAPSP90": H64[:16]}
    assert GS.ctl_p0(rec_ok(), {"VGAPSP90": H64.upper()})["pass"] is True

    def fails(rec, shas, frag):
        r_ = GS.ctl_p0(rec, shas)
        assert r_["pass"] is False and any(frag in f for f in r_["failures"]), (frag, r_["failures"])

    fails(rec_ok(card="PREREG_V8F_PROOF_ADD133"), {"VGAPSP90": H64}, "names the card")
    fails(rec_ok(pass_p0=False), {"VGAPSP90": H64}, "pass_p0 False")
    fails(rec_ok(pass_p0="true"), {"VGAPSP90": H64}, "pass_p0 true")
    fails(rec_ok(pass_p0=None), {"VGAPSP90": H64}, "pass_p0 None")
    fails(rec_ok(), {"VGAPSP90": "cd" * 32}, "on disk has sha256")
    fails(rec_ok(), {"VGAPSP90": None}, "on disk has sha256 none")
    fails(rec_ok(), {}, "on disk has sha256 none")
    fails(rec_ok(files={"VGAPSP90": {"sha256": "ab" * 31}}), {"VGAPSP90": H64}, "64 hex")
    fails(rec_ok(files={}), {"VGAPSP90": H64}, "64 hex")
    fails(rec_ok(files={"VGAPLE": {"sha256": H64}}), {"VGAPSP90": H64}, "64 hex")
    for bad in (None, [], "x"):
        assert GS.ctl_p0(bad, {"VGAPSP90": H64})["pass"] is False


def line(name, m, kappa="0.2", ckpt=None, optional=True, extra="", model_seed=None, reader=None):
    ckpt = ckpt or (("%02x" % (m + 1)) * 32)
    ln = "arm I10%s_s%d form RR replan steps 10 rule every 5 readout causal_g kappa %s" % (name, m, kappa)
    if optional:
        ln += " preroll 0 exec plan nav deadreckon grid eval_grid_windows.json"
    ln += " model seed %d reader seed %d ckpt sha256 %s wrapper sha256 %s started 2026-11-23T10:00:00Z%s" % (m if model_seed is None else model_seed, (m if m < 2 else m - 2) if reader is None else reader, ckpt, WRAP, extra)
    return ln + "\nrc 0 finished x"


def p2_lines():
    return {"K0_SP90_s%d" % m: (line("RQ50KF5Hb20gSP90", m, extra=" vis-gap-file /w/VGAPSP90.npz vis-closing 1"), line("RQ50KF5Hb20gA90", m)) for m in GS.SEEDS}


def test_p2_passes_and_ignores_fields_only_the_new_line_carries():
    r = GS.ctl_p2(p2_lines())
    assert r["pass"] is True and r["failures"] == [], r["failures"][:3]


def test_p2_each_clause_fails():
    def fails(mut, frag, key="K0_SP90_s1"):
        L = p2_lines()
        n, b = L[key]
        L[key] = mut(n, b)
        r = GS.ctl_p2(L)
        assert r["pass"] is False and any(frag in f and f.startswith(key) for f in r["failures"]), (frag, r["failures"][:4])

    fails(lambda n, b: (n.replace("kappa 0.2", "kappa 0.3"), b), "kappa: new 0.3")
    fails(lambda n, b: (n, b.replace("kappa 0.2", "kappa 0.3")), "kappa: new 0.2")
    fails(lambda n, b: (n.replace("replan steps 10", "replan steps 5"), b), "replan: new 5")
    fails(lambda n, b: (n.replace("rule every 5", "rule every 10"), b), "every: new 10")
    fails(lambda n, b: (n, b.replace("readout causal_g", "readout K")), "readout: new causal_g")
    fails(lambda n, b: (n.replace("form RR", "form RLc"), b), "form: new RLc")
    fails(lambda n, b: (n.replace("preroll 0", "preroll 30"), b), "preroll: new 30")
    fails(lambda n, b: (n.replace("exec plan", "exec other"), b), "exec: new other")
    fails(lambda n, b: (n.replace("nav deadreckon", "nav other"), b), "nav: new other")
    fails(lambda n, b: (n.replace("grid eval_grid_windows.json", "grid grid_p30.json"), b), "grid: new grid_p30.json")
    fails(lambda n, b: (n.replace("model seed 1", "model seed 3"), b), "model seed: the new line reads 3, expected 1")
    fails(lambda n, b: (n, b.replace("model seed 1", "model seed 0")), "model_seed: new 1, banked 0")
    fails(lambda n, b: (n.replace("reader seed 1", "reader seed 0"), b), "reader_seed: new 0, banked 1")
    fails(lambda n, b: (n.replace("ckpt sha256 " + "02" * 32, "ckpt sha256 " + "99" * 32), b), "ckpt: new")
    fails(lambda n, b: (re.sub(r" kappa \S+", "", n), b), "the new line carries no kappa")
    fails(lambda n, b: (n, re.sub(r" kappa \S+", "", b)), "the banked line carries no kappa")
    fails(lambda n, b: (re.sub(r" ckpt sha256 \S+", "", n), b), "ckpt: the banked line carries it")
    fails(lambda n, b: (re.sub(r" exec \S+", "", n), b), "exec: the banked line carries it")
    fails(lambda n, b: (re.sub(r" reader seed \S+", "", n), b), "the new line carries no reader_seed")
    fails(lambda n, b: (n, "\n"), "no run log of the banked run")
    fails(lambda n, b: ("\n", b), "no run log of the new run")


def test_p2_checkpoint_agrees_by_a_prefix_of_16_and_not_15():
    L = p2_lines()
    n, b = L["K0_SP90_s2"]
    L["K0_SP90_s2"] = (n.replace("ckpt sha256 " + "03" * 32, "ckpt sha256 " + "03" * 8), b)
    assert GS.ctl_p2(L)["pass"] is True
    L["K0_SP90_s2"] = (n.replace("ckpt sha256 " + "03" * 32, "ckpt sha256 " + "030303030303030"), b)
    assert GS.ctl_p2(L)["pass"] is False


def test_p2_defaults_where_the_banked_line_lacks_the_optional_fields():
    L = p2_lines()
    n, b = L["K0_SP90_s3"]
    bare = re.sub(r" preroll \S+ exec plan nav deadreckon grid \S+", "", b)
    L["K0_SP90_s3"] = (n, bare)
    assert GS.ctl_p2(L)["pass"] is True
    L["K0_SP90_s3"] = (n.replace("preroll 0", "preroll 30"), bare)
    r = GS.ctl_p2(L)
    assert r["pass"] is False and any("the new line has to read preroll 0" in f for f in r["failures"])
    L["K0_SP90_s3"] = (re.sub(r" preroll \S+ exec plan nav deadreckon grid \S+", "", n), bare)
    r = GS.ctl_p2(L)
    assert any("has to read nav deadreckon" in f for f in r["failures"]) and any("has to read grid eval_grid_windows.json" in f for f in r["failures"])


def test_p2_run_set_is_checked():
    L = p2_lines()
    L.pop("K0_SP90_s3")
    r = GS.ctl_p2(L)
    assert r["pass"] is False and any("run lines read" in f for f in r["failures"])


def lv_ok():
    lv = {"K0": {arm: dict(zip("CUF", v)) for arm, v in GS.EXPECTED["K0"].items()}}
    lv["K0"]["SP90"] = {"C": 0.0, "U": 0.04, "F": 0.16}
    return lv


def pair_ok():
    return {"U": {"pooled": -0.0116, "interval": [-0.0175, -0.0063]}, "F": {"pooled": 0.0062, "interval": [0.0019, 0.0112]}, "C": {"pooled": 0.0, "interval": [0.0, 0.0]}}


def test_p3_passes_within_the_tolerance_and_each_cell_fails_outside():
    assert GS.ctl_p3(lv_ok(), pair_ok())["pass"] is True
    lv = lv_ok()
    lv["K0"]["A90"]["U"] += 0.00009
    pr = pair_ok()
    pr["U"]["interval"][0] -= 0.00009
    assert GS.ctl_p3(lv, pr)["pass"] is True
    for arm in ("GEc", "A90", "RLc", "RR"):
        for c in "CUF":
            lv = lv_ok()
            lv["K0"][arm][c] += 0.0002
            r = GS.ctl_p3(lv, pair_ok())
            assert r["pass"] is False and any(("K0 %s %s:" % (arm, c)) in f for f in r["failures"]), (arm, c)
    for mk in "UF":
        for key in ("pooled", "lo", "hi"):
            pr = pair_ok()
            if key == "pooled":
                pr[mk]["pooled"] += 0.0002
            else:
                pr[mk]["interval"][0 if key == "lo" else 1] += 0.0002
            r = GS.ctl_p3(lv_ok(), pr)
            assert r["pass"] is False and any(("A90-VS-GEc %s" % mk) in f for f in r["failures"]), (mk, key)


def test_p3_missing_values_fail_and_the_failure_text_holds_no_value():
    assert GS.ctl_p3({}, pair_ok())["pass"] is False
    assert GS.ctl_p3(lv_ok(), None)["pass"] is False
    assert GS.ctl_p3(lv_ok(), {})["pass"] is False
    lv = lv_ok()
    lv["K0"]["GEc"]["U"] = float("nan")
    assert GS.ctl_p3(lv, pair_ok())["pass"] is False
    lv = lv_ok()
    lv["K0"]["GEc"]["U"], lv["K0"]["RR"]["F"] = 0.123456, 0.987654
    pr = pair_ok()
    pr["U"]["pooled"], pr["F"]["interval"] = -0.5432, [0.3333, 0.4444]
    r = GS.ctl_p3(lv, pr)
    txt = " ".join(r["failures"])
    assert r["pass"] is False and not re.search(r"\d\.\d", txt) and "123456" not in txt and "5432" not in txt, txt
    assert GS.ctl_p3(lv_ok(), pair_ok())["failures"] == []


def p4(**kw):
    a = dict(text_queue="....\n12 passed in 3s\n", text_read="== 30 passed, 2 warnings in 9s ==", mtime_queue=1000.0, mtime_read=1010.0, queue_start=1100.0, controls_start=1200.0,
             path_queue="/w/gapspread10k/lead_tests/lead_pytest_queue.out", path_read="/w/gapspread10k/lead_tests/lead_pytest_read.out")
    a.update(kw)
    return GS.ctl_p4(a["text_queue"], a["text_read"], a["mtime_queue"], a["mtime_read"], a["queue_start"], a["controls_start"], a["path_queue"], a["path_read"])


def test_p4_passes_and_each_clause_fails():
    assert p4()["pass"] is True
    assert p4(mtime_queue=1100.0, mtime_read=1200.0)["pass"] is True                                         # equal times are not later
    for kw, frag in (({"text_queue": "3 passed, 1 failed"}, "queue kit"), ({"text_read": "1 error"}, "reader kit"), ({"text_read": "no tests ran"}, "reader kit"),
                     ({"text_queue": "0 passed"}, "queue kit"), ({"mtime_queue": 1101.0}, "later than the first start"), ({"mtime_read": 1201.0}, "later than the start of the controls"),
                     ({"queue_start": None}, "a time is missing"), ({"controls_start": "x"}, "a time is missing"),
                     ({"path_queue": "/workspace/v8proof/gapspread10k/kit_queue/lead_pytest_queue.out"}, "inside the kit folder kit_queue"),
                     ({"path_read": "C:\\w\\gapspread10k\\kit_read\\lead_pytest_read.out"}, "inside the kit folder kit_read")):
        r = p4(**kw)
        assert r["pass"] is False and any(frag in f for f in r["failures"]), (kw, r["failures"])
    assert p4(path_queue="", path_read="")["pass"] is True
    assert p4(path_read="/w/gapspread10k/kit_read_out/x.out")["pass"] is True                                 # a folder that only begins with the kit name is not a kit folder


def test_p5_needs_grid_and_both_strategic_stages():
    ok = {"K0": {"grid_all_pass": True, "strategic_a": True, "strategic_b": True}}
    if GS.MD is None:
        pytest.skip("no model reader")
    assert GS.ctl_p5(ok)["pass"] is True
    for k in ("grid_all_pass", "strategic_a", "strategic_b"):
        bad = {"K0": dict(ok["K0"], **{k: False})}
        r = GS.ctl_p5(bad)
        assert r["pass"] is False and r["failures"]
    r = GS.ctl_p5({"K0": dict(ok["K0"], strategic_b=False, strategic_reason="boom")})
    assert "boom" in r["failures"][0]
    assert GS.ctl_p5({})["pass"] is False and GS.ctl_p5(dict(ok, K30=ok["K0"]))["pass"] is False


def test_p6_counts():
    done = {k: True for k in GS.all_run_keys()}
    nd = {k: 141 for k in GS.all_run_keys()}
    marker = "x\nZZ GAPSPREAD10K DONE 2026-11-23T20:00:00Z all steps done\n"
    r = GS.ctl_p6(done, nd, 141, marker)
    assert r["pass"] is True and r["failures"] == [] and GS.ctl_p6(done, nd, 141, None)["pass"] is True
    d2 = dict(done, K0_SP90_s3=False)
    assert any("K0_SP90_s3: no T1_DONE" in f for f in GS.ctl_p6(d2, nd, 141, marker)["failures"])
    d2 = dict(done)
    d2.pop("K0_SP90_s0")
    assert any("K0_SP90_s0: no T1_DONE" in f for f in GS.ctl_p6(d2, nd, 141, marker)["failures"])
    n2 = dict(nd, K0_SP90_s1=140)
    assert any("K0_SP90_s1: 140 dump files" in f for f in GS.ctl_p6(done, n2, 141, marker)["failures"])
    n2 = dict(nd, K0_SP90_s2=None)
    assert any("K0_SP90_s2: None dump files" in f for f in GS.ctl_p6(done, n2, 141, marker)["failures"])
    for bad in ("ZZ REPL10K DONE x all steps done", "ZZ GAPSPREAD10K DONE x finished", ""):
        assert any("done marker" in f for f in GS.ctl_p6(done, nd, 141, bad)["failures"]), bad


# ============================================================================================ the whole of the controls on tiny worlds
def _write_run(pull, name, m, first, ci, fr, present, gaps, clos=None, sha=None, done=True, ndump=3, meta=True):
    d = os.path.join(pull, "I10%s_s%d" % (name, m))
    os.makedirs(os.path.join(d, "ticks"), exist_ok=True)
    h = len(ci) // 2
    for nm, sl in (("rule_log_I10%s_s%d.npz", slice(0, h)), ("rule_log_mid_I10%s_s%d.npz", slice(h, len(ci)))):
        z = dict(ci=ci[sl], j=fr[sl], present=present[sl], g_src=gaps[sl].copy())
        z["c_src"] = clos[sl] if clos is not None else np.zeros(len(ci[sl]))
        if meta:
            z.update(vis_gap_file_sha256=np.array(sha or "none"), vis_closing=np.array(1))
        np.savez(os.path.join(d, "ticks", nm % (name, m)), **z)
    io.open(os.path.join(d, "ticks", "run_I10%s_s%d.txt" % (name, m)), "w", encoding="utf-8", newline="\n").write(first + "\nrc 0 finished x\n")
    if done:
        io.open(os.path.join(d, "T1_DONE"), "w").write("")
    dd = os.path.join(d, "dump_I10%s_s%d_m10_causal_heads" % (name, m))
    os.makedirs(dd, exist_ok=True)
    for i in range(ndump):
        np.savez(os.path.join(dd, "ep%03d.npz" % i), x=np.zeros(1))


class NS(object):
    pass


def make_world(base, fault=None):
    """a tiny world (3 clips of 12 frames, four model seeds) in which every control passes (fault None), or exactly the one control named by `fault` (its first two characters) fails. -> (a, m, wsha)"""
    base = str(base)
    pull = os.path.join(base, "pull")
    os.makedirs(pull, exist_ok=True)
    nk = 12
    ci, fr = np.repeat(np.arange(3), nk), np.tile(np.arange(nk), 3)
    n = len(ci)
    has = np.ones(n, bool)
    has[[2, 14, 30]] = False
    gp = 15.0 + 0.9 * fr + 0.3 * ci
    gfile = os.path.join(base, "gapfiles", "VGAPSP90.npz")
    os.makedirs(os.path.dirname(gfile), exist_ok=True)
    np.savez(gfile, clip_index=ci, frame=fr, gap=np.where(has, gp, np.nan), has_out=has, clips=np.array(["c0", "c1", "c2"]), seed=np.int64(-1))
    gsha = GS.sha_file(gfile)
    rec = {"card": GS.CARD, "pass_p0": fault != "p0", "files": {"VGAPSP90": {"sha256": gsha if fault != "p0_sha" else "1" * 64}}, "counts": {"n_rows_out_of_range": 0}}
    grec = os.path.join(base, "gapfiles", "GAPSPREAD10K_GAPFILE_RECORD.json")
    json.dump(rec, io.open(grec, "w", encoding="utf-8"))
    reader_gap = 20.0 + 0.2 * np.arange(n)
    clos = np.where(np.arange(n) % 2 == 0, 0.5, 0.0)
    ones = np.ones(n, bool)
    for m in GS.SEEDS:
        _write_run(pull, "RQ50KF5Hb20", m, line("RQ50KF5Hb20", m), ci, fr, ones, reader_gap + 1e-5, clos, meta=False)
        _write_run(pull, "RQ50KF5Hb20gA90", m, line("RQ50KF5Hb20gA90", m), ci, fr, ones, reader_gap + 1e-5, clos, meta=False)
        g = np.where(has, gp, reader_gap) + 1e-5
        c2 = clos
        sha = gsha
        if fault == "p7" and m == 1:
            g[3] += 0.5
        if fault == "p7_closing" and m == 2:
            c2 = np.zeros(n)
        if fault == "p7_sha" and m == 3:
            sha = "2" * 64
        first = line("RQ50KF5Hb20gSP90", m, extra=" vis-gap-file /w/VGAPSP90.npz gap file sha256 %s vis-closing 1" % gsha)
        if fault == "p2" and m == 1:
            first = first.replace("kappa 0.2", "kappa 0.3")
        if fault == "p1" and m == 0:
            first = first.replace("28bc31f21ecb4ccb", "e" * 16)
        _write_run(pull, "RQ50KF5Hb20gSP90", m, first, ci, fr, ones, g, c2, sha=sha, done=not (fault == "p6" and m == 3), ndump=3 if not (fault == "p6_dump" and m == 0) else 2)
    chk = os.path.join(base, "CHECK_WRAPPER_V_DONE.txt")
    io.open(chk, "w", encoding="utf-8").write("a\n" + (GS.CHECK_LAST if fault != "p1_check" else "ZZ CHECK-WRAPPER-V FAIL") + "\n\n")
    hs = os.path.join(base, "CHECK_WRAPPER_V.hashes")
    io.open(hs, "w", encoding="utf-8").write("sha %s\n" % GS.WRAPPER_SHA)
    tdir = os.path.join(base, "kit_read" if fault == "p4_kit" else "lead_tests")
    os.makedirs(tdir, exist_ok=True)
    tq, tr = os.path.join(tdir, "lead_pytest_queue.out"), os.path.join(tdir, "lead_pytest_read.out")
    io.open(tq, "w", encoding="utf-8").write("....\n12 passed in 3.1s\n")
    io.open(tr, "w", encoding="utf-8").write("....\n== 30 passed, 2 warnings in 9.0s ==\n")
    dm = os.path.join(base, "GAPSPREAD10K_DONE.txt")
    io.open(dm, "w", encoding="utf-8").write("ZZ GAPSPREAD10K DONE 2026-11-23T11:00:00Z all steps done\n")
    a = NS()
    a.pull, a.gap_record, a.gap_files = pull, grec, {"VGAPSP90": gfile}
    a.wrapper_check, a.wrapper_hashes, a.tests_queue, a.tests_read, a.done_marker, a.expect_dumps = chk, hs, tq, tr, dm, 3
    a.tests_mtime_queue, a.tests_mtime_read, a.queue_start, a.controls_start = 1000.0, 1010.0 if fault != "p4" else 1300.0, 1100.0, 1200.0
    lv = lv_ok()
    if fault == "p3":
        lv["K0"]["RR"]["U"] += 0.001
    pr = pair_ok()
    if fault == "p3_pair":
        pr["F"]["interval"][1] += 0.001
    st = {"K0": {"levels": lv["K0"], "pairs": {"A90-VS-GEc": pr}, "grid_all_pass": True, "strategic_a": True, "strategic_b": fault != "p5"}}
    return a, {"settings": st}, GS.WRAPPER_SHA


@need_md
def test_every_control_passes_in_the_world_and_a_fault_fails_exactly_its_own(tmp_path):
    a, m, w = make_world(tmp_path / "ok")
    c = GS.compute_controls(a, m, w)
    assert sorted(c) == sorted(GS.CONTROLS)
    assert {k: v["pass"] for k, v in c.items()} == {k: True for k in GS.CONTROLS}, {k: v.get("failures") for k, v in c.items() if not v["pass"]}
    assert sorted(c["p7"]["values"]["rollout_shares"]) == GS.rule_keys()
    for fault in ("p0", "p0_sha", "p1", "p1_check", "p2", "p3", "p3_pair", "p4", "p4_kit", "p5", "p6", "p6_dump", "p7", "p7_closing", "p7_sha"):
        a, m, w = make_world(tmp_path / ("w_" + fault), fault)
        c = GS.compute_controls(a, m, w)
        flags = {k: v["pass"] for k, v in c.items()}
        assert flags == {k: (k != fault[:2]) for k in GS.CONTROLS}, (fault, flags, c[fault[:2]]["failures"][:3])


@need_md
def test_p7_names_the_rollout_and_the_reason(tmp_path):
    a, m, w = make_world(tmp_path / "w7", "p7")
    r = GS.ctl_p7(GS.rule_checks(a))
    assert r["pass"] is False and any(f.startswith("K0_SP90_s1: ") and "differ from the gap of the file" in f for f in r["failures"])
    a, m, w = make_world(tmp_path / "w7s", "p7_sha")
    r = GS.ctl_p7(GS.rule_checks(a))
    assert any(f.startswith("K0_SP90_s3: ") and "meta names the gap file hash" in f for f in r["failures"])
    a, m, w = make_world(tmp_path / "w7c", "p7_closing")
    assert any(f.startswith("K0_SP90_s2: ") and "closing input is 0" in f for f in GS.ctl_p7(GS.rule_checks(a))["failures"])
    a, m, w = make_world(tmp_path / "w7n")
    a.gap_files = {}
    assert any("no gap file" in f for f in GS.ctl_p7(GS.rule_checks(a))["failures"])
    assert GS.ctl_p7({})["pass"] is False


def test_gap_file_arguments():
    assert GS.parse_gap_files(["VGAPSP90=/a/b.npz"]) == {"VGAPSP90": "/a/b.npz"} and GS.parse_gap_files([]) == {}
    for bad in (["VGAPSP90"], ["VGAPSP90="], ["OTHER=/a"], ["VGAPSP90=/a", "VGAPSP90=/b"]):
        with pytest.raises(SystemExit):
            GS.parse_gap_files(bad)


@need_md
def test_assemble_controls_withholds_the_payload_unless_all_pass(tmp_path):
    a, m, w = make_world(tmp_path / "ok")
    c = GS.compute_controls(a, m, w)
    r = GS.assemble_controls(c, {"settings": {}}, {"files": {}})
    assert r["all_pass"] is True and "payload" in r and "withheld" not in r and r["card"] == GS.CARD
    a, m, w = make_world(tmp_path / "bad", "p6")
    r = GS.assemble_controls(GS.compute_controls(a, m, w), {"settings": {}}, {"files": {}})
    assert r["all_pass"] is False and "withheld" in r and "payload" not in r


def test_controls_lines_print_only_pass_fail_and_a_reason():
    ok = {k: {"pass": True, "failures": []} for k in GS.CONTROLS}
    L = GS.controls_lines({"controls": ok, "all_pass": True})
    assert L == ["ctl p%d PASS" % i for i in range(8)] + ["ZZ READ-GAPSPREAD10K controls all_pass True"]
    bad = dict(ok, p3={"pass": False, "failures": ["K0 A90 U: not within the tolerance of the banked level", "second"]}, p6={"pass": False, "failures": []})
    L = GS.controls_lines({"controls": bad, "all_pass": False})
    assert L[3] == "ctl p3 FAIL K0 A90 U: not within the tolerance of the banked level" and L[6] == "ctl p6 FAIL not computed" and L[-1] == "ZZ READ-GAPSPREAD10K controls all_pass False"
    assert len(L) == 9 and all(x.startswith("ctl p") or x.startswith("ZZ READ-GAPSPREAD10K") for x in L)
    L = GS.controls_lines({"controls": {}, "all_pass": False})
    assert L[0] == "ctl p0 FAIL not computed" and len(L) == 9


# ============================================================================================ the names stage as a child process
def full_payload():
    pl = mk_payload(ov(u="lo", f="lo"))
    d = pl["settings"]["K0"]
    d["levels"] = {arm: {"C": 0.0, "U": 0.05, "F": 0.2} for arm in GS.ARMS}
    d["levels_per_seed"] = {arm: {c: [0.1, 0.2, 0.3, 0.4] for c in "CUF"} for arm in GS.ARMS}
    d["counts"] = {"n_windows": 9, "n_lead_windows": 6, "n_episodes_with_lead": 3}
    d["instrument_bar_met"] = True
    d["families"] = {nm: {"lateral": "x"} for nm in GS.PAIRS}
    d["strategic"] = {"pairs": {nm: {"reading": "strategic-x"} for nm in GS.STRAT_PAIRS}}
    pl["control_values"] = {"p7": {"rollout_shares": {"K0_SP90_s0": {"n_rows": 5}}}}
    return pl


def write_controls(path, files, all_pass=True, drop=None, fail=None):
    ctl = {"p%d" % i: {"pass": True} for i in range(8)}
    if drop:
        ctl.pop(drop)
    if fail:
        ctl[fail] = {"pass": False}
    inputs = {"files": {lab: {"path": p, "sha256": (GS.sha_file(p) if os.path.isfile(p) else None)} for lab, p in files.items()}}
    io.open(path, "w", encoding="utf-8").write(json.dumps({"card": GS.CARD, "all_pass": all_pass, "payload": full_payload(), "controls": ctl, "inputs": inputs}))


def test_refusals_of_the_reader(tmp_path):
    base = [READER]
    r = run_py(base + ["--selftest"])
    assert r.returncode == 2 and "REFUSED" in r.stdout and "no selftest" in r.stdout
    r = run_py(base + ["--stage", "controls"])
    assert r.returncode == 2 and "REFUSED" in r.stdout
    out = tmp_path / "o.json"
    out.write_text("x")
    r = run_py(base + ["--stage", "names", "--out", str(out), "--model-reader", ""])
    assert r.returncode == 2 and "exists" in r.stdout
    inp = tmp_path / "in.txt"
    inp.write_text("input")
    ctl = tmp_path / "c.json"
    for kw in ({"drop": "p7"}, {"fail": "p7"}, {"all_pass": False}):
        write_controls(str(ctl), {"in": str(inp)}, **kw)
        r = run_py(base + ["--stage", "names", "--out", str(tmp_path / "n.json"), "--controls", str(ctl)])
        assert r.returncode == 3 and "ZZ READ-GAPSPREAD10K names refused" in r.stdout and "p0 to p7" in r.stdout, kw
    assert not (tmp_path / "n.json").exists()
    write_controls(str(ctl), {"in": str(inp)})
    inp.write_text("input changed")
    r = run_py(base + ["--stage", "names", "--out", str(tmp_path / "n.json"), "--controls", str(ctl)])
    assert r.returncode == 3 and "an input is not as the controls stage hashed it" in r.stdout and "in: sha256" in r.stdout and not (tmp_path / "n.json").exists()
    ctl.write_text(json.dumps({"card": GS.CARD, "all_pass": True, "payload": {}, "controls": {"p%d" % i: {"pass": True} for i in range(8)}}))
    r = run_py(base + ["--stage", "names", "--out", str(tmp_path / "n.json"), "--controls", str(ctl)])
    assert r.returncode == 3 and "no input hashes" in r.stdout


def test_the_names_stage_writes_result_log_and_verdict_once(tmp_path):
    inp = tmp_path / "in.txt"
    inp.write_text("input")
    ctl = tmp_path / "c.json"
    write_controls(str(ctl), {"in": str(inp)})
    out = tmp_path / "RES.json"
    r = run_py([READER, "--stage", "names", "--out", str(out), "--controls", str(ctl)])
    assert r.returncode == 0, (r.stdout[-400:], r.stderr[-400:])
    lines = r.stdout.splitlines()
    assert lines[-1] == "ZZ READ-GAPSPREAD10K names done" and lines[0].startswith("TIER T1-kin/hb0.2")
    res = json.loads(rd(str(out)))
    assert res["names"]["OVERALL"] == P + "PLACED-SAME-SAFETY-AT-A-LOWER-PRICE" and res["controls_sha256"] == GS.sha_file(str(ctl))
    vp = tmp_path / "GAPSPREAD10K_VERDICT.md"
    assert vp.exists() and (tmp_path / "RES_names.log").exists()
    v = rd(str(vp))
    assert "not driving performance" in v and P + "PLACED-SAME-SAFETY-AT-A-LOWER-PRICE" in v and "Controls: p0 to p7 all pass" in v
    for nm in GS.PAIRS:
        assert "- %s:" % nm in v
    r2 = run_py([READER, "--stage", "names", "--out", str(tmp_path / "RES2.json"), "--controls", str(ctl)])
    assert r2.returncode == 2 and "exists" in r2.stdout and not (tmp_path / "RES2.json").exists()
    r3 = run_py([READER, "--stage", "names", "--out", str(out), "--controls", str(ctl)])
    assert r3.returncode == 2


def test_print_sections_prints_names_levels_pairs_and_families():
    pl = full_payload()
    for nm in GS.PAIRS:
        for m in "CUF":
            pl["settings"]["K0"]["pairs"][nm][m].setdefault("pooled", 0.0)
    names = GS.all_names(pl)
    L = GS.print_sections(pl, names)
    txt = "\n".join(L)
    assert L[0].startswith("TIER T1-kin/hb0.2") and names["OVERALL"] in L
    for arm in GS.ARMS:
        assert "level K0 %s pooled over four model seeds" % arm in txt
    for nm in GS.PAIRS:
        assert "K0 %s (" % nm in txt
    assert "K0 SP90-VS-A90 (named)" in txt and "K0 SP90-VS-RLc (no name)" in txt and "K0 A90-VS-GEc (no name)" in txt
    assert "families K0 SP90-VS-A90" in txt and "families K0 SP90-VS-GEc" in txt and "families K0 A90-VS-GEc" not in txt
    assert "rollout K0_SP90_s0" in txt and "no pre-roll" in txt


# ============================================================================================ driver, pull script, wording
@pytest.mark.skipif(BASH is None, reason="no bash")
@pytest.mark.parametrize("path", [DRIVER, PULL])
def test_bash_syntax(path):
    r = subprocess.run([BASH, "-n", path], capture_output=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr[-300:]


def test_driver_text():
    t = rd(DRIVER)
    assert t.startswith("#!/usr/bin/env bash") and "GE=$V/gapspread10k; K=$GE/kit_read; OUT=$GE/read_out; LT=$GE/lead_tests; GFM=$GE/gapfiles" in t
    for must in ("GAPSPREAD10K_START.txt", "GAPSPREAD10K_DONE.txt", "GAPSPREAD10K_STOP.txt", "GAPSPREAD10K_CONTROLS.json", "GAPSPREAD10K_RESULT.json", "GAPSPREAD10K_VERDICT.md", "read_names.log", "read_names.out",
                 "ZZ PODREAD-GAPSPREAD10K", "GS=$GFM/VGAPSP90.npz", "GREC=$GFM/GAPSPREAD10K_GAPFILE_RECORD.json", "--gap-file VGAPSP90=\"$GS\" --gap-record \"$GREC\"", "for s in 0 1 2 3; do",
                 "--grid-dir \"$W/longroll\"", "read_gapspread10k.py", "$LT/lead_pytest_queue.out", "$LT/lead_pytest_read.out", "grep -E '^ctl p[0-7] (PASS|FAIL)|^ZZ READ-GAPSPREAD10K'", "I10R0Hb20",
                 "ZZ PODREAD-GAPSPREAD10K STOP", "--verdict \"$OUT/GAPSPREAD10K_VERDICT.md\"", "READER_HASHES.txt"):
        assert must in t, must
    run_line = re.search(r"  for x in (.*); do", t).group(1).split()
    want = ["I10" + GS.SPEC["K0"]["runs"][a] for a in GS.ARMS] + ["I10R0Hb20"]
    assert sorted(run_line) == sorted(want) and len(run_line) == 6
    assert "A80" not in t and "REPL" not in t and "GAPMARGIN" not in t and "--models-record" not in t and "VGAPLE" not in t
    assert t.count("GAPSPREAD10K_START.txt") >= 2 and "head -n 1 \"$GE/GAPSPREAD10K_START.txt\" | awk '{print $1}'" in t
    assert "/workspace/v8proof/gapspread10k" not in t.replace("$V/gapspread10k", "") and "rm " not in t and "mv " not in t


def test_pull_script_text():
    t = rd(PULL)
    for must in ("D=/f/TanitAD-scratch/v8proof/gapspread10k", "ROOT=/workspace/v8proof/gapspread10k", "pod_add134", "tmp_pull_add134.txt", "tmp_pull_add134.tar", "pull_add134_",
                 "*.npz|*.pt) continue", "sed '/hf_/d'", "mkdir \"$D/pod_add134\"", "GAPSPREAD10K_START.txt", "GAPSPREAD10K_DONE.txt", "GAPSPREAD10K_STOP.txt", "lead_tests/lead_pytest_queue.out",
                 "lead_tests/lead_pytest_read.out", "read_out/*.json", "read_out/*.md", "small_gapspread10k/*", "gapspread10k_queue.log", "gapfiles/*.json"):
        assert must in t, must
    assert "rm " not in t and "mv " not in t and ".npz " not in t.split("FILES=")[1].split("\n")[0]


def test_wording_of_the_kit_files():
    banned = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lon" + "ger", "closed" + " loop"]
    names = [f for f in os.listdir(HERE) if os.path.isfile(os.path.join(HERE, f)) and f.split(".")[-1] in ("py", "sh", "md", "out")]
    assert {"make_gapspread_reader.py", "read_gapspread10k.py", "podread_gapspread10k.sh", "pull_gapspread10k.sh", "test_gapspread_read.py"} <= set(names)
    for f in names:
        t = rd(os.path.join(HERE, f)).lower()
        for w in banned:
            assert w not in t, (f, w)
    for f in names:
        assert not re.match(r"^(aux|con|prn|nul)(\.|$)", f.lower())
        assert len(os.path.join(HERE, f)) <= 200
    for f in ("read_gapspread10k.py", "podread_gapspread10k.sh", "make_gapspread_reader.py", "pull_gapspread10k.sh"):
        assert "\r" not in rd(os.path.join(HERE, f))
    for f in ("read_gapspread10k.py", "podread_gapspread10k.sh", "pull_gapspread10k.sh", "GAPSPREAD10K_READ_DEFS.md"):
        t = rd(os.path.join(HERE, f))
        for m_ in re.finditer(r"driving performance", t):
            assert t[max(0, m_.start() - 4):m_.start()] == "not ", (f, t[max(0, m_.start() - 30):m_.end()])
