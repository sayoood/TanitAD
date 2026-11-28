"""test_preswrap.py -- tests of the kit of study "preswrap": the generator of the wrapper variant rzgfhqetjvp (presence file flags), its check and the helpers. Synthetic data only; nothing runs on a pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest preswrap/test_preswrap.py -q -p no:cacheprovider --basetemp F:/TanitAD-scratch/v8proof/preswrap_tmp/pt"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("DISTRULE_DIR", str(HERE.parent / "distrule"))
import check_wrapper_p as CW  # noqa: E402
import make_wrapper_p as MWP  # noqa: E402

SOURCE = HERE.parent / "visrule" / "v8f_t1_eval_reader_rule_rzgfhqetjv.py"
VARIANT = HERE / "v8f_t1_eval_reader_rule_rzgfhqetjvp.py"
RECORD = HERE / "WRAPPER_P_GEN_RECORD.json"
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
needs_dist = pytest.mark.skipif(not (Path(os.environ["DISTRULE_DIR"]) / "v8f_t1_eval_rule_mix.py").is_file(), reason="the distrule folder is not there")


def stext():
    return SOURCE.read_bytes().decode("utf-8")


# ------------------------------------------------------------------------------------------------ the generator
def test_the_source_has_the_asserted_hash_and_is_lf_text():
    raw = SOURCE.read_bytes()
    assert b"\r" not in raw
    assert hashlib.sha256(raw).hexdigest()[:16] == "28bc31f21ecb4ccb" == MWP.PARENT_SHA16


def test_generator_is_deterministic_and_the_text_on_disk_is_what_it_builds():
    a = MWP.build(stext())
    b = MWP.build(stext())
    assert a[0] == b[0]
    assert a[0] == VARIANT.read_bytes().decode("utf-8")
    assert len(a[1]) == 12 and [n.split(" ")[0] for n, _ in a[1]] == ["P%d" % i for i in range(1, 13)]
    assert a[0].count("\n") > stext().count("\n")


def test_each_edit_text_must_occur_exactly_once_in_the_source():
    s = stext()
    with pytest.raises(AssertionError):
        MWP.build(s.replace('ap.add_argument("--vis-closing", action="store_true")\n', 'ap.add_argument("--vis-closing", action="store_true")\n    ap.add_argument("--vis-closing", action="store_true")\n'))
    with pytest.raises(AssertionError):
        MWP.build(s.replace('pe, ge, oke = _reader_at(st, rd, int(e_off) - d)', "pass"))
    with pytest.raises(AssertionError):
        MWP.build(s.replace('    present, used = feed(STATE["form"], p, g, ok, STATE["xs"], STATE["ys"])\n    vis_c = None\n', "    vis_c = None\n"))


def test_main_writes_the_variant_and_the_record_and_refuses_a_changed_source_and_an_existing_output(tmp_path, capsys):
    out, rec = tmp_path / "v.py", tmp_path / "rec.json"
    assert MWP.main(["--out", str(out), "--record", str(rec)]) == 0
    assert out.read_bytes() == VARIANT.read_bytes()
    r = json.loads(rec.read_text(encoding="utf-8"))
    assert r["source_sha256"][:16] == MWP.PARENT_SHA16 and r["pass"] is True
    assert r["variant_sha256"] == hashlib.sha256(VARIANT.read_bytes()).hexdigest()
    assert len(r["substitutions"]) == 12 and all(x["label"] and x["id"].startswith("P") for x in r["substitutions"])
    with pytest.raises(SystemExit) as e:
        MWP.main(["--out", str(out), "--record", str(tmp_path / "rec2.json")])
    assert "exists" in str(e.value)
    with pytest.raises(SystemExit) as e:
        MWP.main(["--out", str(tmp_path / "v2.py"), "--record", str(rec)])
    assert "exists" in str(e.value)
    changed = tmp_path / "changed.py"
    changed.write_bytes(SOURCE.read_bytes() + b"# x\n")
    with pytest.raises(SystemExit) as e:
        MWP.main(["--parent", str(changed), "--out", str(tmp_path / "v3.py"), "--record", str(tmp_path / "rec3.json")])
    assert "REFUSED" in str(e.value) and not (tmp_path / "v3.py").exists()
    crlf = tmp_path / "crlf.py"
    crlf.write_bytes(SOURCE.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(SystemExit) as e:
        MWP.main(["--parent", str(crlf), "--out", str(tmp_path / "v4.py"), "--record", str(tmp_path / "rec4.json")])
    assert "REFUSED" in str(e.value)


def test_the_variant_compiles_and_the_record_on_disk_matches_it():
    compile(VARIANT.read_text(encoding="utf-8"), str(VARIANT), "exec")
    r = json.loads(RECORD.read_text(encoding="utf-8"))
    assert r["variant_sha256"] == hashlib.sha256(VARIANT.read_bytes()).hexdigest() and r["pass"] is True


def test_static_check_passes_and_each_planted_fault_turns_one_fact_red(tmp_path):
    assert all(ok for _, ok, _ in CW.static_check(str(SOURCE), str(VARIANT), str(RECORD)))
    src = VARIANT.read_text(encoding="utf-8")
    faults = {"S1": ('ap.add_argument("--vis-pres-file", default="")', 'ap.add_argument("--vis-pres-file", default="x")'),
              "S2": ('choices=("head", "or"), default=""', 'choices=("head", "or", "and"), default=""'),
              "S3": ('(mine.vis_pres_file or mine.vis_pres_mode) and mine.form != "RR"', '(mine.vis_pres_file or mine.vis_pres_mode) and mine.form != "RRc"'),
              "S6": ("        p = pres_tick(p)\n", "        pass\n"),
              "S8": ("pe = pres_look(st, int(e_off) - d, pe, False)", "pe = pe"),
              "S9": ('if STATE.get("vis") is not None or STATE.get("pres") is not None:', 'if STATE.get("vis") is not None:'),
              "S5": ("if mine.vis_pres_file else None", "if mine.vis_pres_file else 5"),
              "S10": ('STATE["cols"].setdefault("p_used", []).append(np.asarray(p))', "pass"),
              "S11": ("**pres_meta(mine.vis_pres_file, mine.vis_pres_mode,", "**dict(a=mine.vis_pres_file, b=mine.vis_pres_mode,")}
    for k, (old, new) in faults.items():
        assert src.count(old) == 1, k
        res = dict((n.split(" ")[0], c) for n, c in CW.static_facts(src.replace(old, new)))
        assert res[k] is False, k
    p = tmp_path / "mut.py"
    p.write_text(src.replace("# x", "") + "\n", encoding="utf-8", newline="\n")
    res = CW.static_check(str(SOURCE), str(p), str(RECORD))
    assert [n.split(" ")[0] for n, ok, _ in res if not ok][:1] == ["T2"]            # the text is not what the generator builds


# ------------------------------------------------------------------------------------------------ the helpers
def world(tmp_path):
    join, pf = CW._world()
    return join, pf, CW._write(str(tmp_path / "good.npz"), join, pf)


def test_replace_in_both_modes_with_edge_cases(tmp_path):
    ns = MWP.helper_ns()
    join, pf, good = world(tmp_path)
    ci = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, -1, 2])
    fr = np.array([0, 1, 2, 3, 0, 1, 2, 4, 3, 0, 0])
    p = np.array([0.6, 0.4, 0.8, 0.5, 0.2, 0.9, 0.1, 0.7, 0.7, 0.4, 0.4])
    S = ns["load_pres"](good, "head", join)
    assert np.array_equal(ns["pres_replace"](S, ci, fr, p), [0.1, 0.9, 0.8, 0.5, 0.7, 0.3, 0.1, 0.7, 0.7, 0.4, 0.4])      # idx 2, 6: NaN in the file; 7, 8: frame outside; 9, 10: clip outside
    S = ns["load_pres"](good, "or", join)
    assert np.array_equal(ns["pres_replace"](S, ci, fr, p), [0.6, 0.9, 0.8, 0.5, 0.7, 0.9, 0.1, 0.7, 0.7, 0.4, 0.4])
    # p exactly 0.5 is present on the tick test of the source (p >= 0.5): the file value 0.5 keeps it present, 0.4999 does not
    pf2 = pf.copy()
    pf2[0], pf2[1] = 0.5, 0.4999
    S = ns["load_pres"](CW._write(str(tmp_path / "edge.npz"), join, pf2), "head", join)
    out = ns["pres_replace"](S, [0, 0], [0, 1], [0.1, 0.9])
    assert out[0] >= 0.5 and out[1] < 0.5
    S = ns["load_pres"](CW._write(str(tmp_path / "edge2.npz"), join, pf2), "or", join)
    out = ns["pres_replace"](S, [0, 0], [0, 1], [0.5, 0.4])
    assert out[0] == 0.5 and out[1] == 0.4999 and np.all(np.isfinite(out))
    # the input is not changed, the output is float64
    p_in = np.array([0.2, 0.2], np.float32)
    ns["pres_replace"](S, [0, 0], [0, 1], p_in)
    assert p_in.dtype == np.float32 and np.array_equal(p_in, np.array([0.2, 0.2], np.float32))
    assert ns["pres_replace"](None, [0], [0], p_in).dtype == np.float64


def test_a_frame_before_the_first_head_output_and_a_negative_frame_have_no_file_value(tmp_path):
    ns = MWP.helper_ns()
    join, pf, good = world(tmp_path)
    S = ns["load_pres"](good, "head", join)
    pv, have = ns["pres_values"](S, [0, 0, 1, 1], [-1, 4, 3, 2])
    assert have.tolist() == [False, False, False, False] and np.isnan(pv).all()
    pv, have = ns["pres_values"](S, [0, 1], [3, 2])
    assert have.tolist() == [True, False] and pv[0] == 0.5 and np.isnan(pv[1])


def test_a_usable_value_needs_the_output_flag_too(tmp_path):
    ns = MWP.helper_ns()
    join, pf, _ = world(tmp_path)
    has = np.isfinite(pf).copy()
    has[1] = False                                         # finite p, but the file says no output on that row
    S = ns["load_pres"](CW._write(str(tmp_path / "flag.npz"), join, pf, has_out=has), "head", join)
    pv, hv = ns["pres_values"](S, [0, 0], [0, 1])
    assert hv.tolist() == [True, False]


def test_counts_for_both_modes_and_the_closing_frames_are_not_counted_twice(tmp_path):
    ns = MWP.helper_ns()
    join, pf, good = world(tmp_path)
    ci = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1])
    fr = np.array([0, 1, 2, 3, 0, 1, 2, 4, 3])
    p = np.array([0.6, 0.4, 0.8, 0.5, 0.2, 0.9, 0.1, 0.7, 0.7])
    for mode, raised, lowered in (("head", 2, 2), ("or", 2, 0)):
        cnt = {"rows": 0, "with_file": 0, "no_file": 0, "raised": 0, "lowered": 0}
        ns["pres_replace"](ns["load_pres"](good, mode, join), ci, fr, p, cnt)
        assert cnt == {"rows": 9, "with_file": 5, "no_file": 4, "raised": raised, "lowered": lowered}
    S = ns["load_pres"](good, "head", join)
    ns["STATE"]["pres"] = S

    class St:
        def __init__(self, c, j):
            self.ci, self.j = c, j
    st = [St(0, 0), St(0, 1)]
    ns["pres_look"](st, 0, np.array([0.6, 0.4]))
    ns["pres_look"](st, 1, np.array([0.6, 0.4]), False)
    assert ns["STATE"]["pres_counts"]["rows"] == 2
    ns["STATE"]["vis_states"] = st
    ns["pres_tick"](np.array([0.6, 0.4]))
    assert ns["STATE"]["pres_counts"]["rows"] == 4
    with pytest.raises(RuntimeError):
        ns["pres_tick"](np.array([0.6, 0.4, 0.1]))


@pytest.mark.parametrize("label,kw", [
    ("array p missing", {"p": None}), ("array has_out missing", {"has_out": None}), ("array clips missing", {"clips": None}), ("array off missing", {"off": None}), ("array n missing", {"n": None}),
    ("has_out length", {"has_out": np.ones(6, bool)}), ("off differs", {"off": np.array([0, 5])}), ("n differs", {"n": np.array([4, 2])}),
    ("clips differ", {"clips": np.array(["a", "c"])}), ("clips length", {"clips": np.array(["a"])}),
    ("output where p is not finite", {"has_out": np.ones(7, bool)}),
])
def test_loader_refusals_on_the_file(tmp_path, label, kw):
    ns = MWP.helper_ns()
    join, pf, _ = world(tmp_path)
    path = CW._write(str(tmp_path / "bad.npz"), join, pf, **kw)
    with pytest.raises(SystemExit) as e:
        ns["load_pres"](path, "head", join)
    assert "REFUSED" in str(e.value)


def test_loader_refusals_on_values_shape_mode_and_join(tmp_path):
    ns = MWP.helper_ns()
    join, pf, good = world(tmp_path)

    def refused(fn):
        with pytest.raises(SystemExit) as e:
            fn()
        assert "REFUSED" in str(e.value)
    refused(lambda: ns["load_pres"](good, "avg", join))
    refused(lambda: ns["load_pres"](good, "", join))
    refused(lambda: ns["load_pres"](str(tmp_path / "none.npz"), "head", join))
    refused(lambda: ns["load_pres"]("", "head", join))
    refused(lambda: ns["load_pres"](good, "head", None))
    refused(lambda: ns["load_pres"](CW._write(str(tmp_path / "hi.npz"), join, np.where(np.arange(7) == 1, 1.0001, pf)), "head", join))
    refused(lambda: ns["load_pres"](CW._write(str(tmp_path / "lo.npz"), join, np.where(np.arange(7) == 1, -0.0001, pf)), "head", join))
    refused(lambda: ns["load_pres"](CW._write(str(tmp_path / "rows.npz"), join, pf[:6]), "head", join))
    refused(lambda: ns["load_pres"](CW._write(str(tmp_path / "dim.npz"), join, pf.reshape(7, 1), has_out=np.isfinite(pf).reshape(7, 1)), "head", join))
    refused(lambda: ns["load_pres"](CW._write(str(tmp_path / "empty.npz"), join, np.zeros(0), clips=np.array([], dtype="<U1"), off=np.zeros(0, np.int64), n=np.zeros(0, np.int64)), "head",
                                    {"clip": np.array([], dtype="<U1"), "off": np.zeros(0, np.int64), "n": np.zeros(0, np.int64)}))
    # the join the run uses decides, not the file: a join with other clip names or another n is refused
    refused(lambda: ns["load_pres"](good, "head", dict(join, clip=np.array(["a", "z"]))))
    refused(lambda: ns["load_pres"](good, "head", dict(join, n=np.array([4, 4]))))
    # the exact bounds 0 and 1 are admitted
    pv = pf.copy()
    pv[0], pv[1] = 0.0, 1.0
    ns["load_pres"](CW._write(str(tmp_path / "bounds.npz"), join, pv), "or", join)


def test_meta_and_first_line_fields_exist_only_with_the_flag():
    ns = MWP.helper_ns()
    assert ns["pres_meta"]("", "", "") == {}
    ns["STATE"]["pres_counts"]["rows"] = 3
    m = ns["pres_meta"]("f.npz", "or", "abc")
    assert sorted(m) == ["vis_pres_counts", "vis_pres_file", "vis_pres_file_sha256", "vis_pres_mode"] and m["vis_pres_mode"] == "or" and m["vis_pres_file_sha256"] == "abc"
    assert json.loads(m["vis_pres_counts"]) == {"lowered": 0, "no_file": 0, "raised": 0, "rows": 3, "with_file": 0}
    assert ns["pres_fields"]("f.npz", "abc", "or") == " vis-pres-file f.npz vis pres sha256 abc vis-pres-mode or"
    t = VARIANT.read_text(encoding="utf-8")
    assert t.count("pres_fields(") == 2                                              # the definition and the one call
    i = t.index('print("pres run-line fields:"')
    assert 'if STATE["pres"] is not None:' in t[i - 120:i]                           # printed only with the flag
    assert 'STATE["pres"] = None ' in t


# ------------------------------------------------------------------------------------------------ the variant against the source (flags absent, flags given)
@needs_dist
def test_identity_checks_pass_on_the_generated_wrapper():
    res = CW.identity_checks(str(SOURCE), str(VARIANT))
    assert all(ok for _, ok, _ in res), [(n, d) for n, ok, d in res if not ok]
    assert len(res) == 8


def mutant(tmp_path, name, old, new):
    t = VARIANT.read_text(encoding="utf-8")
    assert t.count(old) == 1, old
    p = tmp_path / ("mut_%s.py" % name)
    p.write_text(t.replace(old, new), encoding="utf-8", newline="\n")
    return p


@needs_dist
@pytest.mark.parametrize("name,old,new,red", [
    ("tick", "        p = pres_tick(p)\n", "        pass\n", "I3"),
    ("earlier", "pe = pres_look(st, int(e_off) - d, pe, False)", "pe = pe", "I5"),
    ("ownp", '(("p", p_rd), ("g_read", g), ("g_used", used), ("ok", ok), ("c_read", c))', '(("p", p), ("g_read", g), ("g_used", used), ("ok", ok), ("c_read", c))', "I3"),
    ("ormode", 'pf if S["mode"] == "head" else np.fmax(p, pf)', 'pf if S["mode"] == "head" else np.minimum(p, pf)', "I4"),
    ("fallback", "p2 = np.where(have, pf if", "p2 = np.where(np.isfinite(pf) | True, pf if", "I2"),
    ("refusal", '(mine.vis_pres_file or mine.vis_pres_mode) and mine.form != "RR"', '(mine.vis_pres_file or mine.vis_pres_mode) and mine.form != "RRc"', "I6"),
])
def test_planted_faults_in_the_variant_turn_the_identity_checks_red(tmp_path, name, old, new, red):
    p = mutant(tmp_path, name, old, new)
    try:
        res = CW.identity_checks(str(SOURCE), str(p))
    except Exception:                                      # a fault that breaks the run is red as well
        assert name == "refusal"                           # (the refusal fault lets the call run on into the pinned tool loader)
        return
    failed = [n.split(" ")[0] for n, ok, _ in res if not ok]
    assert red in failed, (name, failed)


@needs_dist
def test_the_new_flags_are_refused_outside_form_rr_and_one_without_the_other():
    m = CW.load_module(str(VARIANT), "wp_flags")
    for args in (["--form", "RRc", "--vis-pres-file", "x.npz", "--vis-pres-mode", "head"], ["--form", "RLb", "--vis-pres-mode", "or"], ["--form", "RR", "--vis-pres-file", "x.npz"], ["--form", "RR", "--vis-pres-mode", "head"]):
        with pytest.raises(SystemExit) as e:
            m._main_inner(["--tree", "x"] + args)
        assert "REFUSED" in str(e.value) and "vis-pres" in str(e.value)
    assert m.STATE["pres"] is None and m.pres_meta("", "", "") == {}


@needs_dist
def test_flags_absent_the_source_and_the_variant_parse_alike_and_the_state_is_unset():
    s, v = CW.load_module(str(SOURCE), "wp_s"), CW.load_module(str(VARIANT), "wp_v")
    assert v.STATE["pres"] is None and "pres" not in s.STATE
    assert v.STATE["pres_counts"] == {"rows": 0, "with_file": 0, "no_file": 0, "raised": 0, "lowered": 0}
    st, vt = SOURCE.read_text(encoding="utf-8"), VARIANT.read_text(encoding="utf-8")
    assert vt.count('ap.add_argument("--vis-pres-') == 2 and st.count("--vis-pres") == 0
    for flag in ('"--vis-gap-file"', '"--vis-closing"', '"--rule-join"'):
        assert st.count("ap.add_argument(%s" % flag) == vt.count("ap.add_argument(%s" % flag) == 1


@needs_dist
def test_the_check_main_ends_with_pass(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_wrapper_p.py"])
    assert CW.main() == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last == "ZZ CHECK-WRAPPER-P PASS"


def test_selftest_passes_and_each_planted_fault_of_the_helpers_is_seen(monkeypatch):
    ok, bad = CW.selftest()
    assert ok and not bad
    broken = MWP.PRES_HELPER.replace("np.fmax(p, pf)", "np.minimum(p, pf)")
    assert broken != MWP.PRES_HELPER
    monkeypatch.setattr(MWP, "PRES_HELPER", broken)
    ok, bad = CW.selftest()
    assert not ok and any("or" in b for b in bad)
    monkeypatch.setattr(MWP, "PRES_HELPER", MWP.PRES_HELPER.replace("(p[fin] > 1.0).any()", "False"))
    ok, bad = CW.selftest()
    assert not ok


def test_the_texts_of_the_kit_hold_no_banned_words_and_no_carriage_return():
    for f in ("make_wrapper_p.py", "check_wrapper_p.py", "test_preswrap.py", "v8f_t1_eval_reader_rule_rzgfhqetjvp.py", "WRAPPER_P_GEN_RECORD.json", "PRESWRAP_DEFS.md"):
        raw = (HERE / f).read_bytes()
        assert b"\r" not in raw, f
        t = raw.decode("utf-8")
        assert not [w for w in BANNED if w in t], (f, [w for w in BANNED if w in t])
        assert len(str(HERE / f)) <= 200
