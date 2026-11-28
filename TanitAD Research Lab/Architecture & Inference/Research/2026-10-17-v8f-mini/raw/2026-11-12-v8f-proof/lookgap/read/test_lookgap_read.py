"""test_lookgap_read.py -- tests of the read kit of card PREREG_V8F_PROOF_ADD108 (LOOKGAP-T0): the selftest, every control red on its planted fault, the four names and the agreement of the two derivations,
the bootstrap against a slow reference loop, bin edges, the m6 parsing, the controls and names stages on a tiny pod-like tree, the shell scripts, wording and path rules. Synthetic data only; no file above 5 MB.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest lookgap/read/test_lookgap_read.py -q -p no:cacheprovider --basetemp lookgap/read/tmp_pt/<new name>"""
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
V8 = HERE.parent.parent
LOOKAT = V8 / "lookat" / "read_lookat.py"
sys.path.insert(0, str(HERE))
import read_lookgap as RG       # noqa: E402

LK, LK_SHA, _ = RG.load_lookat(str(LOOKAT))
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
PODREAD, PUSH, PULL = HERE / "podread_lookgap.sh", HERE / "push_lookgap_read.sh", HERE / "pull_lookgap.sh"


def posix(p):
    return Path(p).as_posix()


def bash_exe():
    return shutil.which("bash")


def run_main(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = RG.main(argv)
    return rc, buf.getvalue()


# ------------------------------------------------------------------------------------------------ selftest and the imported reader
def test_selftest_passes_and_reader_of_card_add107_is_imported_unchanged():
    rc, out = run_main(["--selftest", "--lookat-reader", str(LOOKAT)])
    assert rc == 0 and out.strip().splitlines()[-1] == "ZZ READ-LOOKGAP SELFTEST OK", out[-500:]
    assert LK_SHA.startswith("8ed6c6256fd5149b")


# ------------------------------------------------------------------------------------------------ bootstrap, bins
def test_bootstrap_is_reproducible_and_equals_a_slow_reference_loop():
    num, den = np.array([1.0, 2.0, 3.0, 4.5, 2.0, 0.5]), np.array([2.0, 2.5, 2.0, 2.0, 3.0, 1.0])
    a, b = RG.boot_ratio(num, den, 300, 108), RG.boot_ratio(num, den, 300, 108)
    assert a == b
    idx = np.random.RandomState(108).randint(0, 6, size=(300, 6))
    v = sorted(sum(num[j] for j in row) / sum(den[j] for j in row) for row in idx)
    lo = v[0] + 0.0
    pos = lambda q: (len(v) - 1) * q / 100.0                       # noqa: E731
    ref = [v[int(pos(q))] + (v[min(int(pos(q)) + 1, len(v) - 1)] - v[int(pos(q))]) * (pos(q) - int(pos(q))) for q in (2.5, 97.5)]
    assert lo <= ref[0] and np.allclose(a, ref, atol=1e-12) and np.allclose(a, np.percentile([sum(num[j] for j in r) / sum(den[j] for j in r) for r in idx], [2.5, 97.5]), atol=1e-12)
    assert RG.boot_ratio(num, den, 300, 109) != a
    assert RG.boot_ratio(np.zeros(6), np.zeros(6), 300, 108) is None                 # fewer than half of the draws finite: no interval


def test_default_draws_are_2000_with_seed_108():
    assert RG.N_BOOT == 2000 and RG.BOOT_SEED == 108


def test_bin_edges_follow_the_card_and_read_lookat():
    g = np.array([0.0, 9.99, 10.0, 19.99, 20.0, 39.99, 40.0, 60.0, 60.01])
    assert [int(RG.gap_bin_mask(g, lo, hi).sum()) for lo, hi in RG.GAP_BINS] == [2, 2, 2, 2]           # 0-10 [0,10), 10-20, 20-40, 40-60 closed at 60; 60.01 in none
    az = np.array([0.0, 4.0, 4.01, 8.0, 8.01, 16.0, 16.01, -30.0])
    assert [int(RG.az_bin_mask(az, lo, hi).sum()) for lo, hi in RG.AZ_BINS] == [2, 2, 2, 2]
    st, _, _, _ = LK._stats(np.zeros(8), np.zeros(8), g[:8], az)                                         # same bins as the imported reader
    assert [b["n"] for b in st["by_az"]] == [int(RG.az_bin_mask(az, lo, hi).sum()) for lo, hi in RG.AZ_BINS]
    assert [b["n"] for b in st["by_gap"]] == [int(RG.gap_bin_mask(g[:8], lo, hi).sum()) for lo, hi in RG.GAP_BINS]


# ------------------------------------------------------------------------------------------------ values
def test_values_on_the_synthetic_world_match_hand_arithmetic():
    P = RG.syn_payload(LK, "cuts")
    S = P["sums"]
    r = np.array(S["r"])
    assert int(r.sum()) == P["n_rows"] and P["n_label_fields_differ"] == 0 and P["n_base"] >= P["n_rows"]
    for k in RG.MAIN_FITS:
        assert abs(sum(S["abs"][k]) / r.sum() - P["E"][k]) < 1e-12
    assert abs(P["R"] - np.mean([P["E"][k] for k in RG.P_FITS]) / np.mean([P["E"][k] for k in ("D0", "D1", "D2")])) < 1e-12
    mD, mP, mB = (np.mean([P["E"][k] for k in ks]) for ks in (("D0", "D1"), ("P0", "P1"), ("B0", "B1")))
    assert abs(P["share_of_bound"] - (mD - mP) / (mD - mB)) < 1e-12
    assert P["perm"]["ratio"] > 1.0 and P["perm"]["iv"][0] > 1.0
    assert set(P["cross"]) == {"X0", "X1"} and P["cross"]["X0"]["n"] == P["n_rows"]
    assert [b["bin"] for b in P["bins_gap"]["P0"]] == ["0-10 m", "10-20 m", "20-40 m", "40-60 m"]
    assert [b["bin"] for b in P["bins_az"]["D0"]] == ["0-4 deg", "4-8 deg", "8-16 deg", "above 16 deg"]
    assert sum(b["n"] for b in P["bins_gap"]["P0"]) == P["n_rows"]


def test_a_label_field_that_differs_between_dumps_is_counted_and_missing_rows_leave_the_row_set():
    J, dumps, bx, dr = RG.syn_world("cuts")
    nrow = len(bx["az"])
    box = {"has": np.ones(nrow, bool), "az": bx["az"], "el": bx["el"]}
    dirs = {"has": np.ones(nrow, bool), "az": dr["az"], "el": dr["el"]}
    base_n = RG.compute_values(LK, J, {k: RG.lay(LK, J, d) for k, d in dumps.items()}, box, dirs, 200)["n_rows"]
    d = dict(dumps["B1"])
    g = d["gap"].copy()
    g[np.flatnonzero(d["lab_present"])[:5]] = np.nan                                  # B1 lacks five rows
    d["gap"] = g
    lg = d["lab_gap"].copy()
    lg[np.flatnonzero(d["lab_present"])[7]] += 0.5                                    # and one label differs
    d["lab_gap"] = lg
    P = RG.compute_values(LK, J, {k: RG.lay(LK, J, v) for k, v in dict(dumps, B1=d).items()}, box, dirs, 200)
    assert P["n_rows"] == base_n - 5 and P["missing_in_base"]["B1"] == 5 and P["missing_in_base"]["P0"] == 0 and P["n_label_fields_differ"] == 1
    lines = RG.print_sections(P, RG.name_of(P))
    assert any(ln.startswith("row set is not 3841") and "B1 5" in ln for ln in lines)


# ------------------------------------------------------------------------------------------------ names
def test_each_name_branch_is_reached_and_the_two_derivations_agree():
    want = {"cuts": "LOOKGAP-T0-CUTS", "notfit": "LOOKGAP-T0-CUTS-NOT-EACH-FIT", "worse": "LOOKGAP-T0-WORSE", "sep": "LOOKGAP-T0-NOT-SEPARATED", "nointerval": None}
    for kind, w in want.items():
        P = RG.syn_payload(LK, kind)
        n1, n2, agree = RG.derive_name(P)
        assert n1 == w and n2 == w and agree, kind
    P = RG.syn_payload(LK, "nointerval")
    assert P["iv"] is None
    assert "name: none (no interval)" in RG.print_sections(P, None)


def test_name_rules_on_planted_values():
    base = {"iv": [0.5, 0.9], "E": {"P0": 1.0, "P1": 1.0, "P2": 1.0, "D0": 4.0, "D1": 3.9, "D2": 4.1, "D3": 4.2, "D4": 4.0}}
    assert RG.name_of(base) == "LOOKGAP-T0-CUTS"
    assert RG.name_of(dict(base, E=dict(base["E"], P2=3.9))) == "LOOKGAP-T0-CUTS-NOT-EACH-FIT"          # P2 equal to the smallest: not below it
    assert RG.name_of(dict(base, iv=[0.5, 1.0])) == "LOOKGAP-T0-NOT-SEPARATED"                          # upper end 1.0 is not below 1
    assert RG.name_of(dict(base, iv=[1.0, 1.2])) == "LOOKGAP-T0-NOT-SEPARATED"                          # lower end 1.0 is not above 1
    assert RG.name_of(dict(base, iv=[1.01, 1.2])) == "LOOKGAP-T0-WORSE"
    assert RG.name_of(dict(base, iv=None)) is None


def test_names_stage_refuses_unless_all_seven_pass_and_writes_the_result_and_log(tmp_path):
    ns = RG.write_tree(LK, tmp_path / "w")
    argv = tree_argv(ns)
    out = tmp_path / "c.json"
    rc, text = run_main(["--stage", "controls", "--out", str(out)] + argv)
    lines = text.strip().splitlines()
    assert rc == 0 and lines[:7] == ["m%d PASS" % i for i in range(7)] and lines[-1] == "ZZ READ-LOOKGAP controls all_pass True" and len(lines) == 8     # no value on stdout
    cj = json.loads(out.read_text(encoding="utf-8"))
    assert cj["all_pass"] is True and sorted(cj["controls"]) == list(RG.CONTROLS) and all(v["pass"] is True for v in cj["controls"].values()) and "payload" in cj and "inputs" in cj
    assert cj["inputs"]["read_lookat.py"] == LK_SHA and len(cj["inputs"]["dump_D0"]) == 64
    rc, text = run_main(["--stage", "controls", "--out", str(out)] + argv)                                  # never reads twice into one file
    assert rc == 2 and "REFUSED" in text
    res, log = tmp_path / "r.json", tmp_path / "r.log"
    rc, text = run_main(["--stage", "names", "--controls", str(out), "--out", str(res), "--log", str(log)])
    assert rc == 0 and res.is_file() and log.is_file() and text.strip().splitlines()[-1] == "ZZ READ-LOOKGAP names rc 0"
    assert json.loads(res.read_text(encoding="utf-8"))["name"] == "LOOKGAP-T0-CUTS"
    sec = log.read_text(encoding="utf-8")
    for needle in ("tier: T0", "families:", "rows: base rows", "E (mean absolute", "R = ", "name: LOOKGAP-T0-CUTS", "share of the bound", "correlation", "by label gap P0", "by label gap D0", "by label gap B0",
                   "by |label azimuth| P0", "cross read X0", "cross read X1", "XDUMP_IDENT.json: pass True", "m3 ratio", "directions on the"):
        assert needle in sec, needle
    assert sec.index("rows: base rows") < sec.index("E (mean absolute") < sec.index("R = ") < sec.index("share of the bound") < sec.index("correlation") < sec.index("by label gap P0") < sec.index("cross read X0") < sec.index("m3 ratio") < sec.index("directions on the")
    rc, text = run_main(["--stage", "names", "--controls", str(out), "--out", str(res), "--log", str(log)])
    assert rc == 2 and "REFUSED" in text
    # a controls file with one control failed: no name
    bad = RG.write_tree(LK, tmp_path / "w2", fault="m1")
    out2 = tmp_path / "c2.json"
    rc, text = run_main(["--stage", "controls", "--out", str(out2)] + tree_argv(bad))
    assert "m1 FAIL" in text and text.strip().splitlines()[-1] == "ZZ READ-LOOKGAP controls all_pass False"
    cj2 = json.loads(out2.read_text(encoding="utf-8"))
    assert "payload" not in cj2 and "withheld" in cj2
    rc, text = run_main(["--stage", "names", "--controls", str(out2), "--out", str(tmp_path / "r2.json"), "--log", str(tmp_path / "r2.log")])
    assert rc == 3 and "REFUSED" in text and not (tmp_path / "r2.json").exists()


def tree_argv(ns):
    a = ["--lookat-reader", str(LOOKAT), "--lookgap-root", posix(ns.lookgap_root), "--run1", posix(ns.run1), "--run-tok-b", posix(ns.run_tok_b), "--box-dir", posix(ns.box_dir), "--eval-join", posix(ns.eval_join),
         "--expect-n-train", "8", "--expect-n-eval", "12", "--expect-rows", str(ns.expect["rows"]), "--expect-fit", "6", "--expect-sel", "2",
         "--ref-e", ",".join("%s=%s" % kv for kv in ns.ref_e.items()), "--ref-s", ",".join("%s=%s" % kv for kv in ns.ref_s.items()), "--n-boot", "300"]
    for k, p in ns.dump_paths.items():
        a += ["--dump", "%s=%s" % (k, posix(p))]
    return a


# ------------------------------------------------------------------------------------------------ controls: one planted fault each
def clean_m0():
    return {"check_dirs_all_pass": True, "n_train": 500, "n_eval": 141, "n_checked": 641, "missing": [], "keys_bad": [], "frames_bad": [], "lead_bad": []}


def clean_m2():
    rec = {"arm": "B", "head": "H5", "seed": 0, "permute_labels": False, "argv": ["f.py", "--root", "/w/run"], "tool_sha256": RG.TOOL_SHA16 + "ab", "train_list_sha256": "T", "eval_list_sha256": "E",
           "clips_read": {"fit": ["a"] * 450, "sel": ["b"] * 50}, "hparams_equal_card": True}
    return {"recs": {"s0": dict(rec), "s1": dict(rec, seed=1), "s2": dict(rec, seed=2), "perm": dict(rec, permute_labels=True)}, "train_sha": "T", "eval_sha": "E", "run_root": "/w/run", "tok_p_real": "/w/tok/P",
            "live_tok_b_real": "/w/tok/P", "runroot": {"tok_b_realpath": "/w/tok/P", "train_list_sha256": "T", "eval_list_sha256": "E", "pilot_train_list_sha256": "T", "pilot_eval_list_sha256": "E"}}


def clean_p():
    return {"E": {"D0": 3.960, "D1": 3.913, "D2": 4.112, "B0": 2.137, "B1": 1.910}, "n_rows": 3841, "n_label_fields_differ": 0, "perm": {"iv": [1.2, 1.9]},
            "dirs": {"n_base": 3841, "n_no_box": 0, "n_no_dir": 0, "S4": 0.8748, "S8_out": 0.5844}}


def test_every_control_is_green_on_clean_facts_and_red_on_its_planted_faults():
    assert RG.ctl_m0(clean_m0())["pass"] and RG.ctl_m1({"pass_m1": True, "clauses": {"a": {"pass": True}}})["pass"] and RG.ctl_m2(clean_m2())["pass"]
    assert RG.ctl_m3(clean_p())["pass"] and RG.ctl_m4(clean_p())["pass"] and RG.ctl_m5(clean_p())["pass"] and RG.ctl_m6({k: "x\n3 passed in 1s\n" for k in "abcd"})["pass"]
    m0 = [RG.ctl_m0(dict(clean_m0(), check_dirs_all_pass=False)), RG.ctl_m0(dict(clean_m0(), check_dirs_all_pass=None)), RG.ctl_m0(dict(clean_m0(), missing=["c1"])), RG.ctl_m0(dict(clean_m0(), keys_bad=["c1"])),
          RG.ctl_m0(dict(clean_m0(), frames_bad=["c1"])), RG.ctl_m0(dict(clean_m0(), lead_bad=["c1"])), RG.ctl_m0(dict(clean_m0(), n_checked=640)), RG.ctl_m0(dict(clean_m0(), n_train=499))]
    m1 = [RG.ctl_m1(None), RG.ctl_m1({"pass_m1": False, "clauses": {"a": {"pass": True}}}), RG.ctl_m1({"pass_m1": True, "clauses": {"a": {"pass": True}, "b": {"pass": False}}}), RG.ctl_m1({"pass_m1": True, "clauses": []}),
          RG.ctl_m1({"pass_m1": True, "clauses": [{"pass": True}, {"pass": False}]})]

    def rec_with(k, **kw):
        f = clean_m2()
        f["recs"][k] = dict(f["recs"][k], **kw)
        return RG.ctl_m2(f)
    m2 = [rec_with("s1", seed=0), rec_with("s0", head="H4"), rec_with("s0", arm="D"), rec_with("s2", permute_labels=True), rec_with("perm", permute_labels=False), rec_with("perm", seed=1), rec_with("s0", hparams_equal_card=False),
          rec_with("s0", tool_sha256="0" * 64), rec_with("s1", train_list_sha256="X"), rec_with("s1", eval_list_sha256="X"), rec_with("s2", clips_read={"fit": ["a"] * 449, "sel": ["b"] * 50}),
          rec_with("s2", clips_read={"fit": ["a"] * 450, "sel": ["b"] * 49}), rec_with("s0", argv=["f.py", "--root", "/other"]), rec_with("s0", argv=["f.py"])]
    f = clean_m2()
    f["recs"].pop("s2")
    m2.append(RG.ctl_m2(f))
    for key, val in (("tok_b_realpath", "/else"), ("train_list_sha256", "X"), ("pilot_eval_list_sha256", "X")):
        f = clean_m2()
        f["runroot"][key] = val
        m2.append(RG.ctl_m2(f))
    f = clean_m2()
    f["live_tok_b_real"] = "/w/tok/B"
    m2.append(RG.ctl_m2(f))
    f = clean_m2()
    del f["runroot"]["eval_list_sha256"]
    m2.append(RG.ctl_m2(f))
    m3 = [RG.ctl_m3(dict(clean_p(), perm={"iv": [0.9, 1.9]})), RG.ctl_m3(dict(clean_p(), perm={"iv": [1.0, 1.9]})), RG.ctl_m3(dict(clean_p(), perm={"iv": None}))]
    m4 = [RG.ctl_m4(dict(clean_p(), E=dict(clean_p()["E"], D0=3.962))), RG.ctl_m4(dict(clean_p(), E=dict(clean_p()["E"], B1=1.9111))), RG.ctl_m4(dict(clean_p(), n_rows=3840)), RG.ctl_m4(dict(clean_p(), n_label_fields_differ=1)),
          RG.ctl_m4(dict(clean_p(), E={k: v for k, v in clean_p()["E"].items() if k != "D2"}))]
    m5 = [RG.ctl_m5(dict(clean_p(), dirs=dict(clean_p()["dirs"], S4=0.8750))), RG.ctl_m5(dict(clean_p(), dirs=dict(clean_p()["dirs"], S8_out=0.5846))), RG.ctl_m5(dict(clean_p(), dirs=dict(clean_p()["dirs"], n_base=3840))),
          RG.ctl_m5(dict(clean_p(), dirs=dict(clean_p()["dirs"], S8_out=None)))]
    ok = {k: "x\n3 passed in 1s\n" for k in "abcd"}
    m6 = [RG.ctl_m6(dict(ok, a="3 passed, 1 failed in 1s\n")), RG.ctl_m6(dict(ok, b="1 error in 0.1s\n")), RG.ctl_m6(dict(ok, c=None)), RG.ctl_m6(dict(ok, d="")), RG.ctl_m6({k: v for k, v in ok.items() if k != "d"})]
    for name, rs in (("m0", m0), ("m1", m1), ("m2", m2), ("m3", m3), ("m4", m4), ("m5", m5), ("m6", m6)):
        assert all(not r["pass"] and r["failures"] for r in rs), (name, [r["pass"] for r in rs])
    for r in m0 + m1 + m2 + m3 + m4 + m5 + m6:
        assert set(r) >= {"pass", "failures"} and isinstance(r["pass"], bool)


def test_m4_and_m5_tolerances_are_inclusive_of_the_card_values():
    P = clean_p()
    assert RG.ctl_m4(dict(P, E=dict(P["E"], D0=3.9609)))["pass"] and RG.ctl_m5(dict(P, dirs=dict(P["dirs"], S4=0.87485)))["pass"]


def test_m6_reads_the_last_non_empty_line_only():
    ok = {k: "5 failed earlier\n== 55 passed, 1 warning in 3.2s ==\n\n  \n" for k in "abcd"}
    assert RG.ctl_m6(ok)["pass"]
    assert not RG.ctl_m6(dict(ok, a="55 passed in 1s\n\nno tests ran\n"))["pass"]
    assert not RG.ctl_m6(dict(ok, a="55 passed in 1s\n3 failed\n"))["pass"]


def test_planted_faults_on_a_tiny_tree_flip_exactly_the_named_control(tmp_path):
    for k in RG.CONTROLS:
        ns = RG.write_tree(LK, tmp_path / ("f" + k), fault=k)
        res = RG.run_controls(ns, LK)
        assert sorted(c for c, v in res["controls"].items() if not v["pass"]) == [k] and "payload" not in res, k
    res = RG.run_controls(RG.write_tree(LK, tmp_path / "ok"), LK)
    assert res["all_pass"] and "payload" in res


def test_m2_compares_the_live_token_real_path_and_m0_reads_frame_counts_from_the_token_header(tmp_path):
    ns = RG.write_tree(LK, tmp_path / "w")
    other = tmp_path / "w" / "lookgap" / "tok" / "B"
    other.mkdir(parents=True)
    ns.run_tok_b = str(other)
    res = RG.run_controls(ns, LK)
    assert [c for c, v in res["controls"].items() if not v["pass"]] == ["m2"]
    ns = RG.write_tree(LK, tmp_path / "w3")
    np.save(os.path.join(ns.tok_p, "t2.npy"), np.zeros((RG.SYN_NFR + 1, 2), np.float32))
    res = RG.run_controls(ns, LK)
    assert [c for c, v in res["controls"].items() if not v["pass"]] == ["m0"]


# ------------------------------------------------------------------------------------------------ shell scripts
@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_scripts_parse_and_podread_with_a_bad_stage_name_stops():
    for f in (PODREAD, PUSH, PULL):
        assert subprocess.run([bash_exe(), "-n", str(f)]).returncode == 0, f
    r = subprocess.run([bash_exe(), str(PODREAD), "bogus"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 3 and r.stdout.strip().splitlines()[-1] == "ZZ PODREAD-LOOKGAP STOP usage: controls|names"
    r = subprocess.run([bash_exe(), str(PODREAD)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 3 and "STOP" in r.stdout


def pod_world(tmp_path, mem_free_gb=30):
    ns = RG.write_tree(LK, tmp_path / "w")
    lg = Path(ns.lookgap_root)
    kit = lg / "kit_read"
    shutil.copy(str(HERE / "read_lookgap.py"), str(kit / "read_lookgap.py"))
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text(str((mem_free_gb + 1) * 1073741824) + "\n", encoding="utf-8")
    (cg / "memory.stat").write_text("anon 1073741824\nfile 5\n", encoding="utf-8")
    (lg / "DONE_FIT.txt").write_text("done\n", encoding="utf-8")
    env = {"LG_ROOT": posix(lg), "LG_PY": posix(sys.executable), "LG_LOOKAT": posix(LOOKAT), "LG_CG": posix(cg), "LG_EXTRA": " ".join(tree_argv(ns)[4:])}
    return ns, lg, env


def run_pod(env, stage, cwd):
    r = subprocess.run([bash_exe(), str(PODREAD), stage], env=dict(os.environ, **env), cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=600)
    return r.returncode, r.stdout


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_podread_controls_then_names_on_a_tiny_tree_prints_only_the_card_lines(tmp_path):
    ns, lg, env = pod_world(tmp_path)
    (lg / "queue" / "logs").mkdir(parents=True)
    (lg / "queue" / "logs" / "score_train.log").write_text("noise\nSCORE train a 1\n", encoding="utf-8")
    (lg / "queue" / "logs" / "score_eval.log").write_text("SCORE eval b 2\nmore noise\n", encoding="utf-8")
    rc, out = run_pod(env, "names", tmp_path)
    assert rc == 3 and out.strip() == "ZZ PODREAD-LOOKGAP STOP NO-CONTROLS-FILE"
    rc, out = run_pod(env, "controls", tmp_path)
    lines = out.strip().splitlines()
    assert rc == 0 and lines[0] == "ZZ READ-LOOKGAP SELFTEST OK" and lines[1:8] == ["m%d PASS" % i for i in range(7)] and lines[8] == "ZZ READ-LOOKGAP controls all_pass True"
    assert lines[9].startswith("ZZ PODREAD-LOOKGAP controls DONE rc 0") and len(lines) == 10, out
    rc, out = run_pod(env, "controls", tmp_path)
    assert rc == 3 and "ALREADY-READ controls" in out
    rc, out = run_pod(env, "names", tmp_path)
    lines = out.strip().splitlines()
    assert rc == 0 and lines[0] == "ZZ READ-LOOKGAP names rc 0" and lines[1].startswith("ZZ PODREAD-LOOKGAP names DONE rc 0"), out
    log = (lg / "read" / "read_names.log").read_text(encoding="utf-8").splitlines()
    i = log.index(RG.STAGE_A_HEADING)
    assert log[i + 1:] == ["SCORE train a 1", "SCORE eval b 2"] and (lg / "read" / "LOOKGAP_RESULT.json").is_file()
    rc, out = run_pod(env, "names", tmp_path)
    assert rc == 3 and "ALREADY-READ names" in out


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_podread_stops_on_stop_marker_missing_done_marker_and_low_memory(tmp_path):
    ns, lg, env = pod_world(tmp_path)
    (lg / "STOP_FIT.txt").write_text("x\n", encoding="utf-8")
    assert "QUEUE-STOPPED" in run_pod(env, "controls", tmp_path)[1]
    os.rename(str(lg / "STOP_FIT.txt"), str(lg / "gone.txt"))
    os.rename(str(lg / "DONE_FIT.txt"), str(lg / "gone2.txt"))
    assert "QUEUE-NOT-DONE" in run_pod(env, "controls", tmp_path)[1]
    os.rename(str(lg / "gone2.txt"), str(lg / "DONE_FIT.txt"))
    (tmp_path / "cg" / "memory.max").write_text(str(10 * 1073741824) + "\n", encoding="utf-8")
    rc, out = run_pod(env, "controls", tmp_path)
    assert rc == 3 and "MEMORY free" in out
    assert not (lg / "read" / "LOOKGAP_CONTROLS.json").exists()


def test_scripts_never_delete_move_or_stop_and_push_pull_are_as_the_model():
    for f in (PODREAD, PUSH, PULL):
        t = f.read_text(encoding="utf-8")
        assert not re.search(r"(^|[\s;&|(])(rm|mv|kill|pkill|rmdir)\s", t, re.M), f.name
    push, pull, pod = PUSH.read_text(encoding="utf-8"), PULL.read_text(encoding="utf-8"), PODREAD.read_text(encoding="utf-8")
    assert "set -C" in push and "/workspace/v8proof/lookgap" in push and "kit_read" in push and "read_lookgap.py podread_lookgap.sh lead_pytest.out" in push and "sha256sum" in push
    assert "pod_add108" in pull and '--exclude=\\"*.pt\\"' in pull and '--exclude=\\"*.npz\\"' in pull and '--exclude=\\"*.npy\\"' in pull and "sed '/hf[_]/d'" in pull and "sha256sum" in pull
    for must in ("FIT_RECORD.json", "RUNROOT.json", "ENC_M1.json", "ENC_M1_IDENTITY.json", "XDUMP_IDENT.json", "CHECK_DIRS.json", "START_", "DONE_", "STOP_", "times"):
        assert must in pull, must
    assert not re.search(r"\.pt\b[^\"]*\.tar|\*\.json.*\.npz", pull.replace("--exclude", ""))
    assert not re.search(r"pod_direct|root@|\bssh\b", pull + push)                                       # no pod address anywhere in the scripts
    assert "LOOKGAP_CONTROLS.json" in pod and "stage A scores (frames with a label lead and a full history; not the row set of this card):" in pod and "ALREADY-READ" in pod
    assert "DONE_FIT.txt" in pod and "STOP_FIT.txt" in pod and "score_train.log" in pod and "score_eval.log" in pod and "CUDA_VISIBLE_DEVICES=-1" in pod


# ------------------------------------------------------------------------------------------------ wording, paths, encodings
def test_files_follow_the_wording_path_and_line_end_rules():
    for f in HERE.iterdir():
        if not f.is_file() or f.suffix not in (".py", ".sh", ".md"):
            continue
        t = f.read_text(encoding="utf-8")
        low = t.lower()
        for b in BANNED:
            assert b not in low, (f.name, b)
        if f.suffix == ".sh":
            assert b"\r" not in f.read_bytes(), f.name
        assert len(str(f)) <= 200 and len(f.relative_to(V8).as_posix()) <= 200 and f.stem not in ("aux", "con", "prn", "nul")
        assert ("hf" + "_") not in t.replace("hf[_]", "") and ("Keys" + ".txt") not in t


def test_reader_is_numpy_only_and_subprocess_calls_name_the_encoding():
    t = (HERE / "read_lookgap.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+(torch|scipy|pandas)", t, re.M)
    for m in re.finditer(r"subprocess\.\w+\(", t):
        assert "encoding=" in t[m.start():m.start() + 400]
