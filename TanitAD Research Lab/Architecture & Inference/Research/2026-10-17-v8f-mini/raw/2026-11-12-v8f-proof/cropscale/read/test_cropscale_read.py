"""pytest of the reader kit of card ADD106 (CROPSCALE10K): synthetic data only (tiny worlds, no file above a few hundred KB). One test per control clause with planted faults, per T0 name branch, per T1 name rule
(count rules with ties and mixed directions), and the refusal of the names stage with a control failed or missing."""
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_cropscale10k as RE                 # noqa: E402

MSG, ENCC = RE.MSG, RE.ENCC
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
W16 = "28bc31f21ecb4ccb"


def bash_exe():
    return shutil.which("bash")


def posix(p):
    """an MSYS path (/f/dir/...): GNU tar reads `F:/...` as a remote host"""
    s = Path(p).as_posix()
    return ("/" + s[0].lower() + s[2:]) if (len(s) > 2 and s[1] == ":") else s


class Ns:
    pass


# ------------------------------------------------------------------------------------------------ the gap-file world (g2, g3)
def make_world(base):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    J, DL, DM, P = MSG.synthetic_world(str(base / "world"))
    F = MSG.collect(P)
    ej = base / "join.npz"
    np.savez(str(ej), **J)
    out = base / "gap"
    ejs = RE.sha_file(str(ej))
    rec = MSG.build(J, DL, DM, F, str(out), eval_join_sha=ejs, expect_clips=len(J["clip"]), inputs={})
    a = Ns()
    a.fit_root, a.large_list, a.pilot_list, a.eval_list, a.old_fits, a.mid_root, a.rec104 = (P["fit_root"], P["large_list"], P["pilot_list"], P["eval_list"], P["old_fits"], P["mid_root"], P["rec104"])
    a.eval_join, a.gap_dir, a.expect_clips = str(ej), str(out), len(J["clip"])
    w = Ns()
    w.a, w.J, w.DL, w.DM, w.P, w.rec, w.ejs, w.base = a, J, DL, DM, P, rec, ejs, base
    return w


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return make_world(tmp_path_factory.mktemp("w"))


def run23(w, rec=None, ejs=None):
    rec = w.rec if rec is None else rec
    F, now = RE.g23_now(w.a, rec)
    return RE.ctl_g2(rec, F), RE.ctl_g3(rec, w.a.gap_dir, w.ejs if ejs is None else ejs, F, now)


def test_clean_world_passes_g2_and_g3_read_again_from_disk(world):
    c2, c3 = run23(world)
    assert c2["PASS"], c2
    assert c3["PASS"], c3
    assert all(c3["now_parts"].values()) and sorted(c3["now_parts"]) == sorted(RE.G3_PARTS)


def edit_npz(path, fn):
    z = MSG.load_npz(path)
    fn(z)
    np.savez(path, **z)


def test_g2_and_g3_planted_faults_turn_them_red(tmp_path):
    def fit(w, s):
        return Path(MSG.fit_dir(os.path.join(w.P["fit_root"], "fits"), s))

    def dump(w, s, root="fit_root"):
        return Path(w.P[root]) / "dump" / "D_H5" / ("reader_s%d_score.npz" % s)

    def gap_edit(name, delta):
        def f(w):
            edit_npz(str(Path(w.a.gap_dir) / (name + ".npz")), lambda z: z.__setitem__("gap", np.where(np.isfinite(z["gap"]), z["gap"] + delta, z["gap"])))
        return f

    def dump_edit(s, root):
        def f(w):
            edit_npz(str(dump(w, s, root)), lambda z: z.__setitem__("gap", np.where(np.arange(len(z["gap"])) == 20, 31.0, z["gap"])))
        return f

    def meta_edit(w):
        p = Path(w.P["fit_root"]) / "dump" / "D_H5" / "reader_s3_meta.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        m["head_pt_sha256"] = "0" * 64
        p.write_text(json.dumps(m), encoding="utf-8")

    def head_edit(w):
        with open(str(fit(w, 3) / "head.pt"), "ab") as fh:
            fh.write(b"x")

    faults = {
        "a head file of a large fit changed on disk": (head_edit, "g2"),
        "a single gap file edited": (gap_edit("VGAPL_s2", 0.5), "g3"),
        "the mean file of the large fits edited by 1e-3": (gap_edit("VGAPLE", 1e-3), "g3"),
        "the mean file of the 500-clip fits edited": (gap_edit("VGAPSE3", 0.5), "g3"),
        "a large dump changed after the build": (dump_edit(4, "fit_root"), "g3"),
        "a 500-clip dump changed after the build": (dump_edit(2, "mid_root"), "g3"),
        "a dump meta names another head file": (meta_edit, "g3"),
    }
    for i, (name, (fn, which)) in enumerate(faults.items()):
        w = make_world(tmp_path / ("p%d" % i))
        assert all(c["PASS"] for c in run23(w)), name
        fn(w)
        c2, c3 = run23(w)
        assert not (c2 if which == "g2" else c3)["PASS"], name
    w = make_world(tmp_path / "recflags")
    r = copy.deepcopy(w.rec)
    r["g2"]["PASS"] = False
    assert not run23(w, rec=r)[0]["PASS"]
    r = copy.deepcopy(w.rec)
    r["g3"]["PASS"] = False
    assert not run23(w, rec=r)[1]["PASS"]
    r = copy.deepcopy(w.rec)
    r["g3"]["parts"]["mean_mid"] = False
    assert not run23(w, rec=r)[1]["PASS"]
    r = copy.deepcopy(w.rec)
    r["fits"]["large"]["s2"]["record_sha256"] = "0" * 64
    assert not run23(w, rec=r)[0]["PASS"]
    assert not run23(w, ejs="0" * 64)[1]["PASS"]                                             # a record for another evaluation join
    assert not RE.ctl_g2(None, {})["PASS"] and not RE.ctl_g3(None, "", None, {}, {})["PASS"]


# ------------------------------------------------------------------------------------------------ g0 and g1
def enc_world(tmp):
    d = str(tmp)
    os.makedirs(d, exist_ok=True)
    nmap = ENCC.make_fixture(d)
    P = lambda *x: os.path.join(d, *x)                                                       # noqa: E731

    def disk0():
        return ENCC.g0(P("pl", "eligible.txt"), P("pl", "eval.txt"), P("pl", "pilot_final.txt"), P("lists", "train_final.txt"), P("lists", "LISTS_LARGE_RECORD.json"), n_sel=1)

    def ident():
        return ENCC.g1_identity(["c0", "c1", "c2"], P("pilot", "tok", "D"), P("check_tok", "tok", "D"))
    return d, P, nmap, disk0, ident


def enc_json(disk0, ident):
    g1 = {c: {"pass": True} for c in RE.G1_CLAUSES}
    g1["pass"] = True
    return {"card": "ADD106", "controls": {"g0": disk0, "g1": g1}, "pass_g0": disk0["pass"], "pass_g1": True, "pass_all": bool(disk0["pass"])}


def test_g0_clean_and_each_clause_planted(tmp_path):
    d, P, nmap, disk0, ident = enc_world(tmp_path / "clean")
    assert disk0()["pass"]
    assert RE.ctl_g0(enc_json(disk0(), ident()), disk0())["PASS"]

    def rewrite(P, fn):
        p = P("lists", "train_final.txt")
        rows = ENCC.read_rows(p)
        rows = fn(rows)
        body = "".join("%s %s\n" % x for x in rows)
        open(p, "w").write(body)
        rec = json.load(open(P("lists", "LISTS_LARGE_RECORD.json")))
        import hashlib
        rec["sha256"]["train_final"] = hashlib.sha256(body.encode()).hexdigest()
        rec["n_large_list"], rec["n_fit"], rec["n_sel"] = len(rows), sum(1 for _, r in rows if r == "fit"), sum(1 for _, r in rows if r == "sel")
        json.dump(rec, open(P("lists", "LISTS_LARGE_RECORD.json"), "w"))

    def rec_edit(P, fn):
        p = P("lists", "LISTS_LARGE_RECORD.json")
        r = json.load(open(p))
        fn(r)
        json.dump(r, open(p, "w"))
    planted = {
        "a clip outside the eligible list": (lambda P: rewrite(P, lambda r: r + [("zz", "fit")]), "g0_all_in_eligible"),
        "an evaluation clip in the large list": (lambda P: rewrite(P, lambda r: r + [("e0", "fit")]), "g0_none_is_evaluation"),
        "the pilot's lines reordered": (lambda P: rewrite(P, lambda r: [r[1], r[0]] + r[2:]), "g0_pilot_500_unchanged"),
        "a pilot role changed": (lambda P: rewrite(P, lambda r: [(r[0][0], "sel")] + r[1:]), "g0_pilot_500_unchanged"),
        "a new clip with the role sel": (lambda P: rewrite(P, lambda r: r[:-1] + [(r[-1][0], "sel")]), "g0_selection_is_pilot_50"),
        "a left-out clip not listed": (lambda P: rec_edit(P, lambda r: r.__setitem__("left_out", r["left_out"][:1])), "g0_left_out_listed"),
        "a left-out clip with no reason": (lambda P: rec_edit(P, lambda r: r.__setitem__("left_out", [{"clip": "c6", "reason": " "}, {"clip": "c7", "reason": "x"}])), "g0_left_out_listed"),
        "a duplicated line": (lambda P: rewrite(P, lambda r: r + [r[-1]]), "g0_no_duplicates"),
        "a record with another hash": (lambda P: rec_edit(P, lambda r: r.__setitem__("sha256", {"train_final": "0" * 64})), "g0_record_matches_file"),
    }
    for i, (name, (fn, clause)) in enumerate(planted.items()):
        d, P, nmap, disk0, ident = enc_world(tmp_path / ("p%d" % i))
        enc = enc_json(disk0(), ident())                                                      # the encode kit's own record still says PASS: the clause is read again from the files
        fn(P)
        assert disk0()[clause]["pass"] is False, name
        assert not RE.ctl_g0(enc, disk0())["PASS"], name
    d, P, nmap, disk0, ident = enc_world(tmp_path / "flags")
    e = enc_json(disk0(), ident())
    for mut in (lambda e: e.__setitem__("pass_g0", False), lambda e: e["controls"]["g0"].__setitem__("pass", False), lambda e: e["controls"]["g0"].pop("g0_no_duplicates"),
                lambda e: e["controls"]["g0"]["g0_left_out_listed"].__setitem__("pass", False)):
        e2 = copy.deepcopy(e)
        mut(e2)
        assert not RE.ctl_g0(e2, disk0())["PASS"]
    assert not RE.ctl_g0(None, disk0())["PASS"] and not RE.ctl_g0(e, None)["PASS"] and not RE.ctl_g0(e, {"pass": True})["PASS"]


def test_g1_flags_and_the_identity_clause_read_again(tmp_path):
    d, P, nmap, disk0, ident = enc_world(tmp_path / "w")
    e = enc_json(disk0(), ident())
    assert RE.ctl_g1(e, ident())["PASS"]
    for mut in (lambda e: e.__setitem__("pass_g1", False), lambda e: e["controls"]["g1"].__setitem__("pass", False), lambda e: e["controls"]["g1"].pop("g1_pilot_unchanged"),
                lambda e: e["controls"]["g1"]["g1_identity_three_clips"].__setitem__("pass", False), lambda e: e.__setitem__("pass_all", False)):
        e2 = copy.deepcopy(e)
        mut(e2)
        assert not RE.ctl_g1(e2, ident())["PASS"]
    assert not RE.ctl_g1(None, ident())["PASS"] and not RE.ctl_g1(e, None)["PASS"]
    p = P("check_tok", "tok", "D", "c1.npy")                                                  # a check token off by 0.5: the clause read again from disk is red though the record says PASS
    a = np.load(p)
    np.save(p, (a.astype(np.float32) + 0.5).astype(np.float16))
    assert ident()["pass"] is False and not RE.ctl_g1(e, ident())["PASS"]


# ------------------------------------------------------------------------------------------------ T0
def planted_dumps(J, errs, nan_rows=None):
    """dumps of the eight heads whose absolute error is exactly errs[head] on every row with a lead in the label and a frame >= 4; elsewhere 20 m. nan_rows {head: [(clip, frame)]} gives rows with no output."""
    D = {}
    for i, h in enumerate(RE.HEADS):
        d = MSG.synthetic_dump(J, i, rng_seed=11)
        lab, labp = d["lab_gap"], d["lab_present"]
        g = np.where(labp & np.isfinite(lab) & (d["frame"] >= 4), lab + errs[h], 20.0)
        g = np.where(d["frame"] >= 4, g, np.nan)
        for (c, f) in (nan_rows or {}).get(h, []):
            g[(d["cid"] == c) & (d["frame"] == f)] = np.nan
        d["gap"] = g
        D[h] = d
    return D


ERRS = {"O0": 3.96, "O1": 3.913, "M2": 4.112, "M3": 4.079, "M4": 4.056, "L2": 3.0, "L3": 3.1, "L4": 3.2}


def test_t0_errors_ratio_and_rows_against_a_plain_loop(tmp_path):
    J = MSG.synthetic_join(n_clips=6, seed=2)
    D = {h: MSG.synthetic_dump(J, i, rng_seed=40 + i) for i, h in enumerate(RE.HEADS)}
    t = RE.t0_read(J, D, n_boot=200, seed=7)
    where = {}
    for h in RE.HEADS:
        d = D[h]
        for r in range(len(d["cid"])):
            where.setdefault((int(d["cid"][r]), int(d["frame"][r])), {})[h] = (float(d["gap"][r]), float(d["lab_gap"][r]), bool(d["lab_present"][r]))
    rows = [v for v in where.values() if len(v) == 8 and all(x[2] and np.isfinite(x[1]) and np.isfinite(x[0]) for x in v.values())]
    assert t["n_rows"] == len(rows) and len(rows) > 10
    for h in RE.HEADS:
        want = sum(abs(v[h][0] - v[h][1]) for v in rows) / len(rows)
        assert abs(t["errors_m"][h] - want) < 1e-9, h
        gs, ls = [v[h][0] for v in rows], [v[h][1] for v in rows]
        assert abs(t["correlation_with_label_gap"][h] - np.corrcoef(gs, ls)[0, 1]) < 1e-9
    l = sum(t["errors_m"][h] for h in RE.HEADS_L) / 3.0
    m = sum(t["errors_m"][h] for h in ("M2", "M3", "M4")) / 3.0
    assert abs(t["ratio"] - l / m) < 1e-12 and t["n_label_fields_differ"] == 0
    assert t["interval"][0] <= t["interval"][1] and t["n_boot"] == 200


def test_t0_row_set_needs_an_output_of_all_eight_heads_and_a_lead_in_the_label():
    J = MSG.synthetic_join(n_clips=6, seed=2)
    base = RE.t0_read(J, planted_dumps(J, ERRS), n_boot=50)
    rows = [(c, f) for c in range(6) for f in range(4, 9)]
    one = RE.t0_read(J, planted_dumps(J, ERRS, nan_rows={"L3": rows}), n_boot=50)
    assert 0 < one["n_rows"] < base["n_rows"]
    five = RE.t0_five(J, planted_dumps(J, ERRS, nan_rows={"L3": rows}))                       # the five-fit measure does not need the large heads
    assert five["n_rows"] > one["n_rows"] and five["n_rows"] == RE.t0_five(J, planted_dumps(J, ERRS))["n_rows"]
    D = planted_dumps(J, ERRS)
    D["M4"]["lab_gap"] = D["M4"]["lab_gap"] + 1.0                                              # label fields that differ between dumps are counted
    assert RE.t0_read(J, D, n_boot=50)["n_label_fields_differ"] > 0


def test_t0_with_constant_errors_the_interval_is_the_point_and_the_name_follows():
    J = MSG.synthetic_join(n_clips=6, seed=2)
    t = RE.t0_read(J, planted_dumps(J, ERRS), n_boot=300, seed=1)
    for h, e in ERRS.items():
        assert abs(t["errors_m"][h] - e) < 1e-9
    r = (3.0 + 3.1 + 3.2) / 3.0 / ((4.112 + 4.079 + 4.056) / 3.0)
    assert abs(t["ratio"] - r) < 1e-9 and abs(t["interval"][0] - r) < 1e-9 and abs(t["interval"][1] - r) < 1e-9
    assert RE.t0_name(t["interval"], [t["errors_m"][h] for h in RE.HEADS_L], [t["errors_m"][h] for h in RE.HEADS_MID]) == "CROPSCALE-T0-GAP-ERROR-CUT"
    e3 = dict(ERRS, L4=3.95, L2=2.9)                                                           # 3.95 is above 3.913, the smallest 500-clip error: the second clause fails
    t3 = RE.t0_read(J, planted_dumps(J, e3), n_boot=300)
    assert t3["interval"][1] < 1.0 and RE.t0_name(t3["interval"], [t3["errors_m"][h] for h in RE.HEADS_L], [t3["errors_m"][h] for h in RE.HEADS_MID]) == "CROPSCALE-T0-GAP-ERROR-CUT-WITHIN-FIT-NOISE"


def test_t0_boot_ratio_equals_a_plain_loop_with_the_same_draws_and_is_seeded():
    sn, sd = np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 2.0])
    rng = np.random.RandomState(9)
    idx = rng.randint(0, 3, size=(400, 3))
    r = []
    for row in idx:
        n = sum(sn[i] for i in row)
        d = sum(sd[i] for i in row)
        r.append(n / d)
    want = [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    assert np.allclose(RE.boot_ratio(sn, sd, 400, 9), want, rtol=0, atol=1e-12)
    assert RE.boot_ratio(sn, sd, 400, 9) == RE.boot_ratio(sn, sd, 400, 9)
    rng = np.random.RandomState(5)
    a, b = rng.rand(60) + 1, rng.rand(60) + 1
    assert RE.boot_ratio(a, b, 300, 1) != RE.boot_ratio(a, b, 300, 2)
    assert RE.boot_ratio(np.ones(5), np.zeros(5), 50, 1) is None                              # no finite draw: no interval


def test_t0_name_branches_ties_and_the_second_clause():
    mid = [3.96, 3.913, 4.112, 4.079, 4.056]
    f = RE.t0_name
    assert f([0.80, 0.99], [3.0, 3.1, 3.2], mid) == "CROPSCALE-T0-GAP-ERROR-CUT"
    assert f([0.80, 0.99], [3.0, 3.1, 3.913], mid) == "CROPSCALE-T0-GAP-ERROR-CUT-WITHIN-FIT-NOISE"          # a large error equal to the smallest is not below it
    assert f([0.80, 0.99], [3.0, 3.1, 3.95], mid) == "CROPSCALE-T0-GAP-ERROR-CUT-WITHIN-FIT-NOISE"
    assert f([0.80, 1.0], [3.0, 3.1, 3.2], mid) == "CROPSCALE-T0-GAP-ERROR-NOT-SEPARATED"                    # an upper end at 1.0 is not below
    assert f([1.0, 1.2], [3.0, 3.1, 3.2], mid) == "CROPSCALE-T0-GAP-ERROR-NOT-SEPARATED"                     # a lower end at 1.0 is not above
    assert f([1.001, 1.2], [3.0, 3.1, 3.2], mid) == "CROPSCALE-T0-MORE-GAP-ERROR"
    assert f([0.9, 1.1], [3.0, 3.1, 3.2], mid) == "CROPSCALE-T0-GAP-ERROR-NOT-SEPARATED"
    assert f([1.2, 1.3], [4.0, 4.1, 4.2], mid) == "CROPSCALE-T0-MORE-GAP-ERROR"
    assert f([0.5, 0.9], [3.0, 3.1, 3.2], [9.0, 9.0, 3.05, 9.0, 9.0]) == "CROPSCALE-T0-GAP-ERROR-CUT-WITHIN-FIT-NOISE"   # the smallest of all five counts, not of three


def test_control_g9_on_the_five_fit_measure():
    ok = {"errors_m": dict(RE.T0_EXPECTED), "n_rows": 3841, "n_label_fields_differ": 0}
    assert RE.ctl_g9(ok)["PASS"]
    assert RE.ctl_g9(dict(ok, errors_m=dict(RE.T0_EXPECTED, O0=3.9608)))["PASS"]
    for h in RE.HEADS_MID:
        bad = dict(RE.T0_EXPECTED)
        bad[h] += 0.002
        assert not RE.ctl_g9(dict(ok, errors_m=bad))["PASS"], h
    assert not RE.ctl_g9(dict(ok, n_rows=3840))["PASS"] and not RE.ctl_g9(dict(ok, n_label_fields_differ=2))["PASS"]
    assert not RE.ctl_g9(None)["PASS"] and not RE.ctl_g9(dict(ok, errors_m=dict(RE.T0_EXPECTED, M3=float("nan"))))["PASS"]
    J = MSG.synthetic_join(n_clips=6, seed=2)
    five = RE.t0_five(J, planted_dumps(J, dict(ERRS, **RE.T0_EXPECTED)))
    for h in RE.HEADS_MID:
        assert abs(five["errors_m"][h] - RE.T0_EXPECTED[h]) < 1e-9
    assert not RE.ctl_g9(five)["PASS"]                                                         # the synthetic world has not 3,841 rows


def test_control_g10_the_permuted_fit_has_more_error_than_l2():
    J = MSG.synthetic_join(n_clips=6, seed=2)
    D = planted_dumps(J, ERRS)

    def perm_with(e):
        d = MSG.synthetic_dump(J, 9, rng_seed=11)
        lab, labp = d["lab_gap"], d["lab_present"]
        d["gap"] = np.where(d["frame"] >= 4, np.where(labp & np.isfinite(lab), lab + e, 20.0), np.nan)
        return d
    pr = RE.perm_ratio(J, perm_with(6.0), D["L2"], n_boot=300)
    assert abs(pr["ratio"] - 2.0) < 1e-9 and pr["interval"][0] > 1.0 and RE.ctl_g10(pr)["PASS"]
    for e in (2.9, 2.0):                                                                       # less error: the lower end is not above 1.0
        pr = RE.perm_ratio(J, perm_with(e), D["L2"], n_boot=300)
        assert not RE.ctl_g10(pr)["PASS"], e
    same = RE.perm_ratio(J, copy.deepcopy(D["L2"]), D["L2"], n_boot=300)                       # the same dump: ratio exactly 1.0, the lower end is not above 1.0
    assert same["ratio"] == 1.0 and not RE.ctl_g10(same)["PASS"]
    pr = RE.perm_ratio(J, perm_with(6.0), D["L2"], n_boot=300)
    pr_diff = dict(pr, n_label_fields_differ=3)
    assert not RE.ctl_g10(pr_diff)["PASS"] and not RE.ctl_g10(None)["PASS"]
    # a random permuted-label fit on noisy data: the interval is read, not assumed
    rng = np.random.RandomState(77)                                                            # not the generator of the join (seed 2): its labels would be correlated with the draws
    d = perm_with(0.0)
    d["gap"] = np.where(np.isfinite(d["gap"]), rng.uniform(0, 60, len(d["gap"])), np.nan)
    pr2 = RE.perm_ratio(J, d, D["L2"], n_boot=300)
    assert pr2["interval"] is not None and pr2["ratio"] > 1.0


def perm_facts(w):
    f = copy.deepcopy(MSG.collect(w.P)["large"]["s2"])
    f.update({"permute_labels": True, "argv": list(f["argv"]) + ["--permute-labels"], "head_disk_sha256": "9" * 64, "head_pt_sha256_record": "9" * 64})
    return f


def test_control_g10_clauses_on_the_permuted_fit_record(world):
    pr = {"ratio": 2.0, "interval": [1.5, 2.5], "n_label_fields_differ": 0}
    l2 = MSG.collect(world.P)["large"]["s2"]
    f = perm_facts(world)
    meta = {"head_pt_sha256": "9" * 64, "seed": 2, "permute_labels": True}
    others = {"1" * 64, "2" * 64}
    assert RE.ctl_g10(pr, f, meta, l2, others)["PASS"]
    muts = {
        "labels not permuted": lambda f, m: f.__setitem__("permute_labels", False),
        "no flag in the call": lambda f, m: f.__setitem__("argv", [x for x in f["argv"] if x != "--permute-labels"]),
        "another seed": lambda f, m: f.__setitem__("seed", 3),
        "another head": lambda f, m: f.__setitem__("head", "H4"),
        "other fit clips": lambda f, m: f.__setitem__("fit_clips", f["fit_clips"][:-1]),
        "other selection clips": lambda f, m: f.__setitem__("sel_clips", f["sel_clips"][1:]),
        "head file differs from its record": lambda f, m: f.__setitem__("head_disk_sha256", "8" * 64),
        "dump meta names another head": lambda f, m: m.__setitem__("head_pt_sha256", "7" * 64),
        "dump meta not permuted": lambda f, m: m.__setitem__("permute_labels", False),
        "dump meta another seed": lambda f, m: m.__setitem__("seed", 4),
        "unreadable record": lambda f, m: f.__setitem__("error", "OSError: x"),
    }
    for name, mut in muts.items():
        f2, m2 = copy.deepcopy(f), dict(meta)
        mut(f2, m2)
        assert not RE.ctl_g10(pr, f2, m2, l2, others)["PASS"], name
    assert not RE.ctl_g10(pr, f, meta, l2, others | {"9" * 64})["PASS"]                       # the head file is the same file as another fit's
    assert not RE.ctl_g10(dict(pr, interval=[1.0, 2.5]), f, meta, l2, others)["PASS"] and not RE.ctl_g10(dict(pr, interval=[0.8, 2.5]), f, meta, l2, others)["PASS"]


# ------------------------------------------------------------------------------------------------ g4 to g8
def test_control_g4_g6_g7_g8():
    shas = {"%s_s%d" % (a, s): W16 + "0" * 48 for a in RE.ALLARMS for s in (0, 1)}
    assert len(shas) == 10 and RE.ctl_g4(W16, "x\nZZ CHECK-WRAPPER-V PASS\n", "abc " + W16 + " v8f\n", shas)["PASS"]
    assert not RE.ctl_g4("0" * 16, "ZZ CHECK-WRAPPER-V PASS\n", W16, shas)["PASS"]
    assert not RE.ctl_g4(W16, "ZZ CHECK-WRAPPER-V FAIL\n", W16, shas)["PASS"]
    assert not RE.ctl_g4(W16, "ZZ CHECK-WRAPPER-V PASS\n", "no hash here", shas)["PASS"]
    assert not RE.ctl_g4(W16, "ZZ CHECK-WRAPPER-V PASS\n", W16, dict(shas, G3_s0="1" * 64))["PASS"]
    assert not RE.ctl_g4(W16, "ZZ CHECK-WRAPPER-V PASS\n", W16, dict(shas, E3c_s1=None))["PASS"]
    assert not RE.ctl_g4(W16, "ZZ CHECK-WRAPPER-V PASS\n", W16, {k: v for k, v in shas.items() if k != "GEc_s1"})["PASS"]      # nine run texts of ten
    lv = dict(RE.C_EXPECTED)
    assert RE.ctl_g6(lv)["PASS"] and RE.ctl_g6(dict(lv, RR=0.09544, F3=0.01846))["PASS"]
    for k, v in (("RR", 0.0953), ("VD5c", 0.0177), ("F2", 0.0139), ("F3", 0.0186), ("F4", 0.0124), ("E5c", 0.0139), ("RLc", 0.0005), ("E5c", None)):
        assert not RE.ctl_g6(dict(lv, **{k: v}))["PASS"], k
    assert not RE.ctl_g6({k: v for k, v in lv.items() if k != "F4"})["PASS"]
    assert RE.C_EXPECTED == {"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169, "F2": 0.0138, "F3": 0.0185, "F4": 0.0123, "E5c": 0.0138}
    ok = "12 passed in 3s"
    assert RE.ctl_g7({"kit_enc": ok, "kit_fit": "24 passed, 1 warning in 3s", "kit_read": ok})["PASS"]
    for k in ("kit_enc", "kit_fit", "kit_read"):
        for bad in ("11 passed, 1 failed in 3s", "", "1 passed, 2 error in 1s", "0 passed in 1s"):
            t = {"kit_enc": ok, "kit_fit": ok, "kit_read": ok}
            t[k] = bad
            assert not RE.ctl_g7(t)["PASS"], (k, bad)
        assert not RE.ctl_g7({kk: ok for kk in ("kit_enc", "kit_fit", "kit_read") if kk != k})["PASS"]
    assert RE.ctl_g8(True, True, True)["PASS"] and not RE.ctl_g8(True, True, False)["PASS"] and not RE.ctl_g8(False, True, True)["PASS"] and not RE.ctl_g8(None, True, True)["PASS"] and not RE.ctl_g8(True, False, True)["PASS"]


def rows(ci, j, pres, g, c=None):
    ci, j, pres, g = np.asarray(ci), np.asarray(j), np.asarray(pres, bool), np.asarray(g, float)
    return (ci.astype(np.int64), j.astype(np.int64), pres, g, np.full(len(g), -0.4) if c is None else np.asarray(c, float))


def test_control_g5_on_planted_rows():
    ref = {(0, 4): 10.0, (0, 5): 11.0, (0, 3): None, (1, 4): 20.0}
    v0 = {(0, 3): [7.0], (0, 4): [9.0], (0, 5): [9.5, 9.5004], (1, 4): [19.0]}
    K = ([0, 0, 0, 1], [3, 4, 5, 4], [1, 1, 1, 1])
    good = rows(*K, [7.0, 10.0, 11.0005, 20.0])
    r = RE.w2_one(good, ref, v0)
    assert r["PASS"] and r["n_lead_rows_with_head_output"] == 3 and r["n_lead_rows_without_head_output"] == 1
    assert not RE.w2_one(rows(*K, [7.0, 10.0, 11.002, 20.0]), ref, v0)["PASS"]
    assert not RE.w2_one(rows(*K, [7.1, 10.0, 11.0, 20.0]), ref, v0)["PASS"]
    assert not RE.w2_one(rows(*K, [7.0, 9.0, 11.0, 20.0]), ref, v0)["PASS"]
    assert not RE.w2_one(rows(*K, [7.0, 10.0, 11.0, 20.0], c=[0, 0, 0, 0]), ref, v0)["PASS"]
    assert RE.w2_one(rows(*K, [7.0, 10.0, 11.0, 20.0], c=[0, 0, 0, 0.5]), ref, v0)["PASS"]
    assert not RE.w2_one(rows([0, 0], [3, 4], [1, 1], [7.0, 10.0]), ref, {(0, 4): [9.0]})["PASS"]
    assert not RE.w2_one(rows([0, 0], [3, 4], [0, 0], [7.0, 10.0]), ref, v0)["PASS"]
    exp = {"vis_gap_file_sha256": "a" * 64, "vis_closing": 1}
    assert RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "a" * 64, "vis_closing": 1})["PASS"]
    assert not RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "b" * 64, "vis_closing": 1})["PASS"]
    assert not RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "a" * 64, "vis_closing": 0})["PASS"]


def write_logs(pull, run, m, tick, mid, meta):
    d = Path(pull) / ("I10%s_s%d" % (run, m)) / "ticks"
    d.mkdir(parents=True, exist_ok=True)
    for nm, r in (("rule_log_I10%s_s%d.npz" % (run, m), tick), ("rule_log_mid_I10%s_s%d.npz" % (run, m), mid)):
        np.savez(str(d / nm), ci=r[0], j=r[1], present=r[2], g_src=r[3], c_src=r[4], **{k: np.asarray(v) for k, v in meta.items()})
    (d / ("run_I10%s_s%d.txt" % (run, m))).write_text("arm x wrapper sha256 %s started t\n" % (W16 + "0" * 48), encoding="utf-8")


def synth_pull(tmp_path, world, spoil=None, closing_zero=None, meta_bad=None, wrong_file=None):
    pull = tmp_path / "pull"
    J, n = world.J, np.asarray(world.J["n"])
    refs = {nm: RE.gapfile_ref(str(Path(world.a.gap_dir) / (nm + ".npz"))) for nm in RE.GAPFILE.values()}
    for m in (0, 1):
        for arm in ("V0",) + RE.ALLARMS:
            run = RE.V0_RUN if arm == "V0" else RE.RUN[arm]
            fn = None if arm == "V0" else RE.GAPFILE[arm]
            if wrong_file and wrong_file[0] == (arm, m):
                fn = wrong_file[1]
            cols = {0: [[], [], [], []], 1: [[], [], [], []]}
            for c in range(len(n)):
                for j in range(int(n[c])):
                    own = np.float32(20.0 + 0.1 * j + c)
                    h = refs[fn].get((c, j)) if fn else None
                    g = np.float32(h) if h is not None else own
                    t = cols[j % 2]
                    t[0].append(c), t[1].append(j), t[2].append(g)
                    t[3].append(0.0 if arm == "V0" else -0.5)
                    if arm == "V0" and j % 2 == 0:
                        t[0].append(c), t[1].append(j), t[2].append(np.float32(own + 0.0004)), t[3].append(0.0)
            if spoil and spoil[0] == (arm, m):
                cols[0][2][spoil[1]] = cols[0][2][spoil[1]] + 0.01
            if closing_zero == (arm, m):
                cols[0][3] = [0.0] * len(cols[0][3])
                cols[1][3] = [0.0] * len(cols[1][3])
            exp = "" if arm == "V0" else world.rec["files"][RE.GAPFILE[arm]]["file_sha256"]
            if meta_bad == (arm, m):
                exp = "f" * 64
            meta = {"vis_gap_file_sha256": exp, "vis_closing": 0 if arm == "V0" else 1}
            tick, mid = [rows(cols[k][0], cols[k][1], [True] * len(cols[k][0]), cols[k][2], cols[k][3]) for k in (0, 1)]
            write_logs(pull, run, m, tick, mid, meta)
    return pull


def test_control_g5_end_to_end_with_the_five_arms_and_planted_faults(world, tmp_path):
    a = Ns()
    a.gap_dir = world.a.gap_dir
    a.pull = str(synth_pull(tmp_path / "a", world))
    per = RE.g5_from_files(a, world.rec)
    c5 = RE.ctl_g5(per)
    assert c5["PASS"], c5
    assert sorted(per) == sorted("%s_s%d" % (x, m) for x in RE.ALLARMS for m in (0, 1)) and len(per) == 10
    assert all(v["n_lead_rows_with_head_output"] > 0 and v["share_of_lead_steps_with_nonzero_closing"] > 0 for v in per.values())
    assert sorted(c5["values"]["rollout_shares"]) == sorted(per)
    for label, kw in (("spoil_g3", dict(spoil=(("G3", 1), 5))), ("spoil_gec", dict(spoil=(("GEc", 0), 7))), ("closing_zero", dict(closing_zero=("G4", 1))), ("meta", dict(meta_bad=("G2", 0))),
                      ("e3c_meta", dict(meta_bad=("E3c", 1))), ("gec_fed_the_other_mean", dict(wrong_file=(("GEc", 0), "VGAPSE3"))), ("g2_fed_g3", dict(wrong_file=(("G2", 0), "VGAPL_s3")))):
        a.pull = str(synth_pull(tmp_path / label, world, **kw))
        assert not RE.ctl_g5(RE.g5_from_files(a, world.rec))["PASS"], label
    a.pull = str(synth_pull(tmp_path / "recwrong", world))
    wrong = copy.deepcopy(world.rec)
    wrong["files"]["VGAPLE"]["file_sha256"] = "e" * 64
    assert not RE.ctl_g5(RE.g5_from_files(a, wrong))["PASS"]
    assert not RE.ctl_g5({k: v for k, v in per.items() if k != "E3c_s1"})["PASS"] and not RE.ctl_g5({})["PASS"]


# ------------------------------------------------------------------------------------------------ the names
def iv(lo, hi):
    return {"interval": [lo, hi], "pooled": 0.5 * (lo + hi)}


def t0_payload(errs=None, interval=(0.70, 0.80)):
    errs = dict(errs or ERRS)
    ratio = (sum(errs[h] for h in RE.HEADS_L) / 3.0) / (sum(errs[h] for h in ("M2", "M3", "M4")) / 3.0)
    return {"tier": "T0", "n_rows": 5000, "n_label_fields_differ": 0, "n_boot": 2000, "seed": 106, "interval_level": "two-sided 95 % percentile", "errors_m": errs, "ratio": ratio, "interval": list(interval),
            "smallest_500_clip_error_m": min(errs[h] for h in RE.HEADS_MID), "each_large_below_smallest_500_clip": {h: errs[h] < min(errs[x] for x in RE.HEADS_MID) for h in RE.HEADS_L},
            "correlation_with_label_gap": {h: 0.5 for h in RE.HEADS}}


def payload(rr=0.0954, rlc=0.0, arms=None, scale=None, mean=None, t0=None):
    """arms {name: interval of C(X) vs RLc}; scale {arm: {"C": (lo, hi), "U": .., "F": ..}} for the pairs of the same fit seed; mean the same for GEc against E3c"""
    arms = arms or {a: (0.0, 0.004) for a in RE.XARMS}
    flat = {"C": (-0.01, 0.01), "U": (-0.01, 0.01), "F": (-0.02, 0.03)}
    pairs = {}
    for a, r in arms.items():
        pairs[a + "-VS-RR"] = {"C": iv(r[0] - rr, r[1] - rr), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
        pairs[a + "-VS-RLc"] = {"C": iv(*r), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
        pairs[a + "-VS-VD5c"] = {"C": iv(r[0] - 0.0169, r[1] - 0.0169), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
        s = (mean if a == "GEc" else (scale or {}).get(a)) or flat
        pairs[a + "-VS-" + RE.SCALE_PAIR[a]] = {m: iv(*s.get(m, flat[m])) for m in ("C", "U", "F")}
    pairs["E3c-VS-RLc"] = {"C": iv(0.0, 0.01), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
    return {"levels": {"RR": {"C": rr}, "RLc": {"C": rlc}}, "levels_per_seed": {}, "pairs": pairs, "families": {}, "strategic": {"pairs": {}}, "t0": t0 or t0_payload(), "fit_selection": {}, "large_list": {},
            "falling_behind_against_RLc": {}}


def test_arm_name_rules_on_planted_intervals_and_their_edges():
    n = RE.names_of("G2", [-0.05, -0.01], [0.0, 0.004], [0.001, 0.02], [0.001, 0.03], 0.0954, 0.0)
    assert n["G2_CONTACT"] == "CROPSCALE10K-G2-CUTS-CONTACT" and n["G2_HALF"] == "CROPSCALE10K-G2-WITHIN-HALF-THE-GAP" and n["G2_VD5C"] == "CROPSCALE10K-G2-MORE-CONTACT-THAN-VD5c"
    assert n["G2_PRICE"] == "CROPSCALE10K-G2-FALLS-BEHIND-MORE"
    n = RE.names_of("GEc", [0.001, 0.01], [0.05, 0.06], [-0.02, -0.001], [-0.03, -0.01], 0.0954, 0.0)
    assert n["GEc_CONTACT"].endswith("-GEc-MORE-CONTACT") and n["GEc_HALF"].endswith("-GEc-HALF-THE-GAP-NOT-REACHED") and n["GEc_VD5C"].endswith("-GEc-LESS-CONTACT-THAN-VD5c") and n["GEc_PRICE"].endswith("-GEc-FALLS-BEHIND-LESS")
    n = RE.names_of("G4", [-0.04, 0.01], [0.01, 0.06], [-0.01, 0.01], [-0.02, 0.01], 0.0954, 0.0)
    assert n["G4_CONTACT"].endswith("CONTACT-NOT-SEPARATED-FROM-RR") and n["G4_HALF"].endswith("HALF-THE-GAP-NOT-DECIDED") and n["G4_VD5C"].endswith("CONTACT-NOT-SEPARATED-FROM-VD5c") and n["G4_PRICE"].endswith("FALLING-BEHIND-NOT-SEPARATED")
    assert RE.names_of("G2", [0, 0], [0.0, 0.0477], [0, 0], [0, 0], 0.0954, 0.0)["G2_HALF"].endswith("WITHIN-HALF-THE-GAP")
    assert RE.names_of("G2", [0, 0], [0.0477, 0.06], [0, 0], [0, 0], 0.0954, 0.0)["G2_HALF"].endswith("NOT-DECIDED")
    assert RE.names_of("G2", [0, 0], [0, 0], [0.0, 0.01], [0, 0], 0.0954, 0.0)["G2_VD5C"].endswith("CONTACT-NOT-SEPARATED-FROM-VD5c")
    assert RE.names_of("G2", [-0.05, 0.0], [0, 0], [0, 0], [0, 0], 0.0954, 0.0)["G2_CONTACT"].endswith("CONTACT-NOT-SEPARATED-FROM-RR")


def test_the_count_rule_over_g2_g3_g4_word_for_word_and_the_fourth_arm_does_not_count():
    P = RE.PRE
    C, M, N = "CUTS-CONTACT", "MORE-CONTACT", "CONTACT-NOT-SEPARATED-FROM-RR"
    W, X, D = "WITHIN-HALF-THE-GAP", "HALF-THE-GAP-NOT-REACHED", "HALF-THE-GAP-NOT-DECIDED"

    def names(c, h, gec=(N, X)):
        out = {}
        for i, a in enumerate(RE.ARMS3):
            out[a + "_CONTACT"], out[a + "_HALF"] = "%s-%s-%s" % (P, a, c[i]), "%s-%s-%s" % (P, a, h[i])
        out["GEc_CONTACT"], out["GEc_HALF"] = "%s-GEc-%s" % (P, gec[0]), "%s-GEc-%s" % (P, gec[1])
        return out
    cases = [([C, C, C], [W, W, W], "HOLDS"), ([C, C, C], [W, W, D], "HOLDS"), ([C, C, C], [W, D, D], "NOT-DECIDED"), ([C, C, C], [D, D, D], "NOT-DECIDED"), ([C, C, C], [W, W, X], "DOES-NOT-HOLD"),
             ([C, C, C], [X, D, D], "DOES-NOT-HOLD"), ([C, C, N], [W, W, W], "DOES-NOT-HOLD"), ([C, M, C], [W, W, W], "DOES-NOT-HOLD"), ([N, N, N], [D, D, D], "DOES-NOT-HOLD"), ([C, C, N], [W, W, D], "DOES-NOT-HOLD"),
             ([C, C, C], [X, X, X], "DOES-NOT-HOLD"), ([C, C, C], [W, X, W], "DOES-NOT-HOLD")]
    for c, h, want in cases:
        assert RE.holds_name(names(c, h)) == P + "-" + want, (c, h)                          # GEc carries NOT-SEPARATED and NOT-REACHED in every case: it is not in the count


def test_scale_names_count_rule_with_ties_edges_and_mixed_directions():
    f = RE.scale_name
    L, R_, Z = (-0.03, -0.01), (0.01, 0.03), (-0.01, 0.02)
    assert f("CONTACT", [L, L, L]) == "CROPSCALE10K-SCALE-LOWERS-CONTACT"
    assert f("CONTACT", [L, L, Z]) == "CROPSCALE10K-SCALE-LOWERS-CONTACT"
    assert f("CONTACT", [Z, L, L]) == "CROPSCALE10K-SCALE-LOWERS-CONTACT"
    assert f("CONTACT", [L, Z, Z]) == "CROPSCALE10K-SCALE-CONTACT-NOT-SEPARATED"                 # one pair is not enough
    assert f("UNSAFE", [L, L, R_]) == "CROPSCALE10K-SCALE-UNSAFE-NOT-SEPARATED"                  # mixed directions: two below and one above
    assert f("UNSAFE", [R_, R_, L]) == "CROPSCALE10K-SCALE-UNSAFE-NOT-SEPARATED"
    assert f("UNSAFE", [R_, R_, R_]) == "CROPSCALE10K-SCALE-RAISES-UNSAFE"
    assert f("FALLING-BEHIND", [R_, Z, R_]) == "CROPSCALE10K-SCALE-RAISES-FALLING-BEHIND"
    assert f("FALLING-BEHIND", [Z, Z, Z]) == "CROPSCALE10K-SCALE-FALLING-BEHIND-NOT-SEPARATED"
    assert f("CONTACT", [(-0.03, 0.0), (-0.03, 0.0), L]) == "CROPSCALE10K-SCALE-CONTACT-NOT-SEPARATED"   # an upper end at 0 is not below 0
    assert f("CONTACT", [(0.0, 0.03), (0.0, 0.03), R_]) == "CROPSCALE10K-SCALE-CONTACT-NOT-SEPARATED"   # a lower end at 0 is not above 0
    assert f("CONTACT", [L, L, (0.0, 0.03)]) == "CROPSCALE10K-SCALE-LOWERS-CONTACT"               # the third has a lower end at 0, which is not above 0
    assert f("CONTACT", [L, R_, Z]) == "CROPSCALE10K-SCALE-CONTACT-NOT-SEPARATED"                 # one each way
    m = RE.mean_name
    assert m("CONTACT", [-0.02, -0.001]) == "CROPSCALE10K-MEAN-LOWERS-CONTACT" and m("UNSAFE", [0.001, 0.02]) == "CROPSCALE10K-MEAN-RAISES-UNSAFE"
    assert m("FALLING-BEHIND", [-0.02, 0.0]) == "CROPSCALE10K-MEAN-FALLING-BEHIND-NOT-SEPARATED" and m("CONTACT", [0.0, 0.02]) == "CROPSCALE10K-MEAN-CONTACT-NOT-SEPARATED"


def test_all_names_cover_the_card_and_the_second_derivation_agrees():
    p = payload()
    names = RE.all_names(p)
    assert names["CARD"] == "CROPSCALE10K-HOLDS" and names["T0"] == "CROPSCALE-T0-GAP-ERROR-CUT"
    assert len(names) == 4 * 4 + 1 + 3 + 3 + 1 == 24
    assert sorted(k for k in names if k.startswith("SCALE_")) == ["SCALE_CONTACT", "SCALE_FALLING-BEHIND", "SCALE_UNSAFE"] and sorted(k for k in names if k.startswith("MEAN_")) == ["MEAN_CONTACT", "MEAN_FALLING-BEHIND", "MEAN_UNSAFE"]
    assert all(k.split("_")[0] in RE.XARMS for k in names if k.endswith(("_CONTACT", "_HALF", "_VD5C", "_PRICE")) and not k.startswith(("SCALE", "MEAN")))
    assert RE.derive_all(p) == names and RE.card_check(p, names)["PASS"]
    p2 = payload(arms={"G2": (0.0, 0.004), "G3": (0.0, 0.06), "G4": (0.0, 0.07), "GEc": (0.0, 0.004)})
    assert RE.all_names(p2)["CARD"] == "CROPSCALE10K-NOT-DECIDED"
    p3 = payload(arms={"G2": (0.0, 0.004), "G3": (0.0, 0.004), "G4": (0.05, 0.06), "GEc": (0.0, 0.004)})
    assert RE.all_names(p3)["CARD"] == "CROPSCALE10K-DOES-NOT-HOLD" and RE.all_names(p3)["G4_HALF"].endswith("HALF-THE-GAP-NOT-REACHED")
    p4 = payload(arms={"G2": (0.09, 0.1), "G3": (0.0, 0.004), "G4": (0.0, 0.004), "GEc": (0.0, 0.004)})
    assert RE.all_names(p4)["CARD"] == "CROPSCALE10K-DOES-NOT-HOLD"
    p5 = payload(arms={"G2": (0.0, 0.004), "G3": (0.0, 0.004), "G4": (0.0, 0.004), "GEc": (0.1, 0.12)})        # the fourth arm has names but does not move the card name
    assert RE.all_names(p5)["CARD"] == "CROPSCALE10K-HOLDS" and RE.all_names(p5)["GEc_CONTACT"].endswith("GEc-MORE-CONTACT")
    sc = {a: {"C": (-0.02, -0.01), "U": (0.01, 0.02), "F": (-0.02, 0.02)} for a in ("G2", "G3", "G4")}
    n6 = RE.all_names(payload(scale=sc, mean={"C": (-0.02, -0.01), "U": (-0.02, 0.0), "F": (0.0, 0.02)}))
    assert n6["SCALE_CONTACT"].endswith("SCALE-LOWERS-CONTACT") and n6["SCALE_UNSAFE"].endswith("SCALE-RAISES-UNSAFE") and n6["SCALE_FALLING-BEHIND"].endswith("SCALE-FALLING-BEHIND-NOT-SEPARATED")
    assert n6["MEAN_CONTACT"].endswith("MEAN-LOWERS-CONTACT") and n6["MEAN_UNSAFE"].endswith("MEAN-UNSAFE-NOT-SEPARATED") and n6["MEAN_FALLING-BEHIND"].endswith("MEAN-FALLING-BEHIND-NOT-SEPARATED")
    h = RE.h_sensitivity(p)
    assert h["H_card"] == 0.0477 and abs(h["H_point_values"] - 0.0477) < 1e-12 and set(RE.XARMS) <= set(h)


def test_card_check_refuses_a_name_that_the_intervals_do_not_give(monkeypatch):
    p = payload()
    names = RE.all_names(p)
    for k, v in (("CARD", "CROPSCALE10K-DOES-NOT-HOLD"), ("T0", "CROPSCALE-T0-MORE-GAP-ERROR"), ("SCALE_UNSAFE", "CROPSCALE10K-SCALE-LOWERS-UNSAFE"), ("MEAN_CONTACT", "CROPSCALE10K-MEAN-RAISES-CONTACT"),
                 ("G3_HALF", "CROPSCALE10K-G3-HALF-THE-GAP-NOT-REACHED")):
        bad = dict(names, **{k: v})
        assert not RE.card_check(p, bad)["PASS"], k
    assert not RE.card_check(p, dict(names, EXTRA="x"))["PASS"] and not RE.card_check(p, {k: v for k, v in names.items() if k != "G2_PRICE"})["PASS"]
    p2 = copy.deepcopy(p)
    p2["t0"]["ratio"] = p2["t0"]["ratio"] + 0.01                                               # a point ratio that is not the ratio of the errors
    assert not RE.card_check(p2, names)["PASS"]
    monkeypatch.setattr(RE, "holds_name", lambda names: "CROPSCALE10K-DOES-NOT-HOLD")
    with pytest.raises(SystemExit):
        RE.all_names(p)
    monkeypatch.undo()
    monkeypatch.setattr(RE, "scale_name", lambda label, ivs: "CROPSCALE10K-SCALE-LOWERS-" + label)
    with pytest.raises(SystemExit):
        RE.all_names(p)


def test_every_t0_branch_through_all_names():
    e = dict(ERRS)
    assert RE.all_names(payload(t0=t0_payload(e, (0.7, 0.99))))["T0"] == "CROPSCALE-T0-GAP-ERROR-CUT"
    assert RE.all_names(payload(t0=t0_payload(dict(e, L4=3.95), (0.7, 0.99))))["T0"] == "CROPSCALE-T0-GAP-ERROR-CUT-WITHIN-FIT-NOISE"
    assert RE.all_names(payload(t0=t0_payload(dict(e, L2=4.5, L3=4.5, L4=4.5), (1.05, 1.2))))["T0"] == "CROPSCALE-T0-MORE-GAP-ERROR"
    assert RE.all_names(payload(t0=t0_payload(e, (0.9, 1.1))))["T0"] == "CROPSCALE-T0-GAP-ERROR-NOT-SEPARATED"


# ------------------------------------------------------------------------------------------------ controls file and the names stage
def all_pass():
    return {k: {"PASS": True} for k in RE.CONTROLS}


def test_assemble_controls_withholds_with_one_control_failed_and_moves_the_values():
    assert RE.CONTROLS == tuple("g%d" % i for i in range(11))
    ok = all_pass()
    ok["g9"] = {"PASS": True, "values": {"errors_m": {"O0": 3.96}}}
    r = RE.assemble_controls(ok, payload())
    assert r["controls_all_pass"] is True and "payload" in r and "withheld" not in r and r["payload"]["control_values"]["g9"] == {"errors_m": {"O0": 3.96}}
    assert "values" not in r["controls"]["g9"]
    for k in RE.CONTROLS:
        bad = all_pass()
        bad[k] = {"PASS": False, "values": {"x": 1}}
        r = RE.assemble_controls(bad, payload())
        assert r["controls_all_pass"] is False and "payload" not in r and "withheld" in r and "values" not in r["controls"][k]
    r = RE.assemble_controls({k: v for k, v in all_pass().items() if k != "g4"}, payload())
    assert r["controls_all_pass"] is False and "withheld" in r


def test_names_stage_refuses_with_a_control_failed_or_missing_and_writes_the_result_and_the_log(tmp_path):
    p = payload()
    out = tmp_path / "res.json"
    for k in RE.CONTROLS:
        bad = all_pass()
        bad[k] = {"PASS": False}
        cf = tmp_path / ("bad_%s.json" % k)
        cf.write_text(json.dumps(RE.assemble_controls(bad, p)), encoding="utf-8")
        assert RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)]) == 3 and not out.exists(), k
        miss = {kk: vv for kk, vv in all_pass().items() if kk != k}
        cm = tmp_path / ("miss_%s.json" % k)
        cm.write_text(json.dumps(RE.assemble_controls(miss, p)), encoding="utf-8")
        assert RE.main(["--stage", "names", "--controls", str(cm), "--out", str(out)]) == 3 and not out.exists(), k
    forged = RE.assemble_controls(all_pass(), p)
    forged["controls"]["g3"]["PASS"] = False                                                   # a file that says all-pass but carries a failed control
    cf = tmp_path / "forged.json"
    cf.write_text(json.dumps(forged), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)]) == 3
    withheld = RE.assemble_controls(dict(all_pass(), g1={"PASS": False}), p)
    withheld["controls_all_pass"] = True                                                       # flag edited by hand, payload under `withheld`
    cw = tmp_path / "w.json"
    cw.write_text(json.dumps(withheld), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cw), "--out", str(out)]) == 3
    cf2 = tmp_path / "ok.json"
    cf2.write_text(json.dumps(RE.assemble_controls(all_pass(), p)), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cf2), "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["controls_sha256"] == RE.sha_file(str(cf2)) and res["names"]["CARD"] == "CROPSCALE10K-HOLDS" and res["names"]["T0"].startswith("CROPSCALE-T0") and "G4_VD5C" in res["names"]
    log = (tmp_path / "res_names.log").read_text(encoding="utf-8")
    assert "CROPSCALE10K-HOLDS" in log and "T0 ratio" in log and log.strip().splitlines()[-1] == "ZZ READ-CROPSCALE10K names rc 0 controls_all_pass True"
    assert RE.main(["--stage", "names", "--controls", str(cf2), "--out", str(out)]) == 2        # an existing result is not written twice


# ------------------------------------------------------------------------------------------------ the controls stage through main(), with a stand-in for the engine
def full_world(tmp_path, monkeypatch, wrapper_ok=True):
    """every file the controls stage reads, synthetic; the engine's grid read is replaced by a function that returns the payload's pairs. -> (argv, world, out path)"""
    import hashlib
    import types
    w = make_world(tmp_path / "w")
    J, base = w.J, w.base
    run1 = Path(w.P["pilot_list"]).parent.parent
    dumps = {s: MSG.synthetic_dump(J, s, rng_seed=100) for s in (0, 1)}
    for s, d in dumps.items():
        p = run1 / "dump" / "D_H5" / ("reader_s%d_score.npz" % s)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(p), **d)
    large = ENCC.read_rows(w.P["large_list"])
    elig = [c for c, _ in large] + ["xx1", "xx2"]
    (base / "eligible.txt").write_text("".join(c + "\n" for c in elig), encoding="utf-8")
    body = open(w.P["large_list"], encoding="utf-8").read()
    lr = base / "LISTS_LARGE_RECORD.json"
    lr.write_text(json.dumps({"n_large_list": len(large), "n_fit": sum(1 for _, r in large if r == "fit"), "n_sel": sum(1 for _, r in large if r == "sel"), "sha256": {"train_final": hashlib.sha256(body.encode()).hexdigest()},
                              "left_out": [{"clip": "xx1", "stage": "did not stage", "reason": "did not stage: no obstacle source"}, {"clip": "xx2", "stage": "staged", "reason": "did not encode: x"}],
                              "left_out_counts": {"did not stage": 1, "did not encode": 1}}), encoding="utf-8")
    ids = [c for c, _ in large[:3]]
    (run1 / "lists" / "train_final_ids.txt").write_text("".join(c + "\n" for c in ids), encoding="utf-8")
    for c in ids:
        for d in (run1 / "tok" / "D", base / "check_tok" / "tok" / "D"):
            d.mkdir(parents=True, exist_ok=True)
            np.save(str(d / (c + ".npy")), np.ones((2, 3), np.float16))
    disk0 = ENCC.g0(str(base / "eligible.txt"), w.P["eval_list"], w.P["pilot_list"], w.P["large_list"], str(lr))
    enc = enc_json(disk0, {"pass": True})
    (base / "ENC_CONTROLS.json").write_text(json.dumps(enc), encoding="utf-8")
    fr = Path(w.P["fit_root"])
    lf = [c for c, r in large if r == "fit"]
    ps = [c for c, r in ENCC.read_rows(w.P["pilot_list"]) if r == "sel"]
    pd_ = fr / "fits" / "D_H5_s2_perm"
    MSG._fit_record(str(pd_), 2, lf, ps, MSG.sha_file(w.P["large_list"]), MSG.sha_file(w.P["eval_list"]), b"perm head " * 20, argv_extra=("--permute-labels",))
    rj = json.loads((pd_ / "FIT_RECORD.json").read_text(encoding="utf-8"))
    rj["permute_labels"] = True
    (pd_ / "FIT_RECORD.json").write_text(json.dumps(rj), encoding="utf-8")
    pdump = MSG.synthetic_dump(J, 2, rng_seed=100)
    pdump["gap"] = pdump["gap"] + 30.0
    pdir = fr / "dump" / "D_H5_perm"
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez(str(pdir / "reader_s2_score.npz"), **pdump)
    (pdir / "reader_s2_meta.json").write_text(json.dumps({"head_pt_sha256": MSG.sha_file(str(pd_ / "head.pt")), "seed": 2, "permute_labels": True}), encoding="utf-8")
    for k in ("kit_enc", "kit_fit", "kit_read"):
        (base / k).mkdir()
        (base / k / "lead_pytest.out").write_text("5 passed in 1s\n", encoding="utf-8")
    wrap = base / "wrapper.py"
    wrap.write_text("wrapper text\n", encoding="utf-8")
    w16 = RE.sha_file(str(wrap))[:16]
    monkeypatch.setattr(RE, "WRAPPER_SHA", w16)
    pull = synth_pull(base / "pullroot", w)
    for f in pull.glob("I10*/ticks/run_*.txt"):
        f.write_text("arm x wrapper sha256 %s started t\n" % (w16 + "0" * 48 if wrapper_ok else "1" * 64), encoding="utf-8")
    (base / "chk.txt").write_text("a\n" + RE.CHECK_LAST + "\n", encoding="utf-8")
    (base / "chk.hashes").write_text("%s v8f\n" % w16, encoding="utf-8")
    five = RE.t0_five(J, {"O0": dumps[0], "O1": dumps[1], "M2": w.DM[2], "M3": w.DM[3], "M4": w.DM[4]})
    monkeypatch.setattr(RE, "T0_EXPECTED", dict(five["errors_m"]))
    monkeypatch.setattr(RE, "T0_ROWS", five["n_rows"])
    pl = payload()

    def fake_measure(a, E, log):
        lv = {k: {"C": v, "U": 0.0, "F": 0.0} for k, v in RE.C_EXPECTED.items()}
        return {"levels": lv, "levels_per_seed": {k: {"C": [v, v], "U": [0, 0], "F": [0, 0]} for k, v in RE.C_EXPECTED.items()}, "pairs": pl["pairs"], "families": {}, "strategic": {"pairs": {}},
                "grid_all_pass": True, "strategic_a": True, "strategic_b": True}
    monkeypatch.setattr(RE, "measure", fake_measure)
    monkeypatch.setattr(RE, "R", types.SimpleNamespace(engine=lambda a: None))
    out = base / "CONTROLS.json"
    argv = ["--stage", "controls", "--pull", str(pull), "--code", "c", "--grid-dir", "g", "--w", "w", "--tree", "t", "--wrapper-file", str(wrap), "--wrapper-check", str(base / "chk.txt"),
            "--wrapper-hashes", str(base / "chk.hashes"), "--eval-join", w.a.eval_join, "--gap-dir", w.a.gap_dir, "--fit-root", w.P["fit_root"], "--large-list", w.P["large_list"], "--lists-record", str(lr),
            "--pilot-list", w.P["pilot_list"], "--eval-list", w.P["eval_list"], "--eligible", str(base / "eligible.txt"), "--old-fits", w.P["old_fits"], "--mid-root", w.P["mid_root"], "--rec104", w.P["rec104"],
            "--enc-controls", str(base / "ENC_CONTROLS.json"), "--tests-enc", str(base / "kit_enc" / "lead_pytest.out"), "--tests-fit", str(base / "kit_fit" / "lead_pytest.out"),
            "--tests-read", str(base / "kit_read" / "lead_pytest.out"), "--pilot-run", str(run1), "--cs-root", str(base), "--expect-clips", str(len(J["clip"])), "--out", str(out)]
    return argv, w, out


def test_the_controls_stage_end_to_end_with_all_controls_passing_then_the_names_stage(tmp_path, monkeypatch, capsys):
    argv, w, out = full_world(tmp_path, monkeypatch)
    assert RE.main(argv) == 0
    printed = capsys.readouterr().out.strip().splitlines()
    assert printed[:-1] == ["g%d PASS" % i for i in range(11)], printed
    assert printed[-1] == "ZZ READ-CROPSCALE10K controls rc 0 controls_all_pass True"
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["controls_all_pass"] is True and "payload" in res and "withheld" not in res and sorted(res["controls"]) == sorted(RE.CONTROLS)
    assert all("values" not in v for v in res["controls"].values()) and set(res["payload"]["control_values"]) >= {"g5", "g9", "g10"}
    assert res["payload"]["t0"]["n_rows"] > 0 and len(res["payload"]["large_list"]["left_out"]) == 2 and "P2perm" in res["payload"]["fit_selection"] and "L2" in res["payload"]["fit_selection"]
    assert (out.parent / "CONTROLS.json.engine.log").is_file()
    assert RE.main(argv) == 2                                                                  # a controls file is not written twice
    n_out = out.parent / "RESULT.json"
    assert RE.main(["--stage", "names", "--controls", str(out), "--out", str(n_out)]) == 0
    r = json.loads(n_out.read_text(encoding="utf-8"))
    assert r["controls_sha256"] == RE.sha_file(str(out)) and len(r["names"]) == 24 and r["names"]["T0"].startswith("CROPSCALE-T0-")
    assert RE.card_check(r["values"], r["names"])["PASS"]


def test_the_controls_stage_with_one_control_failed_withholds_every_value(tmp_path, monkeypatch, capsys):
    argv, w, out = full_world(tmp_path, monkeypatch, wrapper_ok=False)                          # the run texts name another wrapper: g4 fails
    assert RE.main(argv) == 0
    printed = capsys.readouterr().out.strip().splitlines()
    assert "g4 FAIL" in printed and sum(1 for ln in printed if ln.endswith("PASS")) == 10 and printed[-1].endswith("controls_all_pass False")
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["controls_all_pass"] is False and "payload" not in res and "withheld" in res
    assert RE.main(["--stage", "names", "--controls", str(out), "--out", str(out.parent / "RESULT.json")]) == 3 and not (out.parent / "RESULT.json").exists()


# ------------------------------------------------------------------------------------------------ files and texts
def test_reader_source_carries_the_references_the_strategic_pairs_and_the_engine_bootstrap_setting():
    t = (HERE / "read_cropscale10k.py").read_text(encoding="utf-8")
    assert "a.n_boot_control = 300" in t and t.index("a = ap.parse_args(argv)") < t.index("a.n_boot_control = 300")
    assert "R.strategic_grid(" in t and "S.pair_read(" in t and 'pairs_def[x + "-VS-VD5c"]' in t and 'pairs_def[x + "-VS-" + SCALE_PAIR[x]]' in t and "SCALE_PAIR[x], runs[SCALE_PAIR[x]]" in t
    assert 'RUN = {"G2": "RQ50KF5P30Hb20gG2"' in t and '"E3c": "RQ50KF5P30Hb20gE3c"' in t and 'B104 = {"F2": "RQ50KF5P30Hb20fF2"' in t and 'B105 = {"E5c": "RQ50KF5P30Hb20eE5c"}' in t
    assert 'GAPFILE = {"G2": "VGAPL_s2"' in t and '"GEc": "VGAPLE"' in t and '"E3c": "VGAPSE3"' in t
    assert "BOOT_SEED" in t and "N_BOOT_T0 = 2000" in t and "contextlib.redirect_stdout" in t


def test_arm_names_match_the_queue_of_the_fit_kit():
    q = HERE.parent / "fit" / "pod_cropscale10k_queue.sh"
    if not q.is_file():
        pytest.skip("the queue file is not beside the reader kit")
    t = q.read_text(encoding="utf-8")
    for arm in RE.ALLARMS:
        assert ('VGAP="$GF/%s.npz" VCLOSE=1 rollout I10%s RR "$k" "$k"' % (RE.GAPFILE[arm], RE.RUN[arm])) in t, arm


def test_podread_text():
    t = (HERE / "podread_cropscale10k.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=-1" in t and "CROPSCALE10K_CONTROLS.json" in t and "CROPSCALE10K_RESULT.json" in t and "ALREADY-READ" in t and "QUEUE-STOPPED" in t and "ZZ PODREAD-CROPSCALE10K" in t
    for x in ("gG2", "gG3", "gG4", "gGEc", "gE3c", "vV0", "vVD5c", "fF2", "fF3", "fF4", "eE5c", "D_H5_perm", "D_H5_s2_perm", "--fit-root", "--wrapper-hashes", "--wrapper-file", "CROPSCALE_GAPFILES_RECORD.json",
              "ENC_CONTROLS.json", "--tests-enc", "--tests-fit", "--tests-read", "lead_pytest.out", "DONE_$st.txt", "CROPSCALE10K_DONE_s0.txt", "read_names.log", "meta.json"):
        assert x in t, x
    for bad in (" rm ", " mv ", "kill", "pkill", "visrep10k"):
        assert bad not in t, bad
    last = [ln for ln in t.splitlines() if ln.startswith('  echo "ZZ PODREAD-CROPSCALE10K') or ln.startswith('echo "ZZ PODREAD-CROPSCALE10K')]
    assert last and all("DONE rc" in ln for ln in last)
    if bash_exe():
        assert subprocess.run([bash_exe(), "-n", str(HERE / "podread_cropscale10k.sh")]).returncode == 0


def test_push_script():
    pu = (HERE / "push_cropscale_read.sh").read_text(encoding="utf-8")
    assert "set -C" in pu and "cropscale/read/lead_pytest.out" in pu and "/workspace/v8proof/cropscale" in pu and "kit_read" in pu and "ZZ PUSH-CROPSCALE-READ" in pu and "REFUSED" in pu
    for f in ("read_cropscale10k.py", "podread_cropscale10k.sh", "lead_pytest.out"):
        assert f in pu
    for f in ("read_cropscale10k.py", "podread_cropscale10k.sh"):
        assert (HERE / f).is_file()
    if bash_exe():
        assert subprocess.run([bash_exe(), "-n", str(HERE / "push_cropscale_read.sh")]).returncode == 0


def pull_variant(tmp_path, name, d, root, ps):
    t = (HERE / "pull_cropscale.sh").read_text(encoding="utf-8")
    s = t.replace("D=/f/TanitAD-scratch/v8proof/cropscale", "D=" + posix(d)).replace("PS=/c/Users/Admin/v8pod/pssh.sh", "PS=" + posix(ps)).replace("ROOT=/workspace/v8proof/cropscale", "ROOT=" + posix(root))
    assert s != t and posix(d) in s and posix(ps) in s and posix(root) in s
    p = tmp_path / name
    with open(str(p), "w", newline="\n", encoding="utf-8") as fh:
        fh.write(s)
    return p


def test_pull_script_text_and_what_it_pulls():
    pl = (HERE / "pull_cropscale.sh").read_text(encoding="utf-8")
    assert "pod_add106" in pl and "pull_add106_" in pl and "date -u +%Y%m%dT%H%M%SZ" in pl and "CROPSCALE_GAPFILES_RECORD.json" in pl and "FIT_RECORD.json" in pl and "_meta.json" in pl and "ENC_CONTROLS.json" in pl
    assert "head.pt" not in pl and "score.npz" not in pl and "tok/" not in pl and "stage" not in pl.split("FILES=")[1].split("[ -e")[0] and "REFUSED" in pl
    assert 'if [ -e "$f" ]; then' in pl
    if bash_exe():
        assert subprocess.run([bash_exe(), "-n", str(HERE / "pull_cropscale.sh")]).returncode == 0


@pytest.mark.skipif(not shutil.which("bash"), reason="no bash")
def test_pull_script_builds_its_tar_from_existing_members_only_with_a_time_stamp_name(tmp_path):
    root = tmp_path / "root"
    (root / "read").mkdir(parents=True)
    (root / "lists").mkdir()
    (root / "ENC_CONTROLS.json").write_text("{}", encoding="utf-8")
    (root / "read" / "x.json").write_text("{}", encoding="utf-8")
    (root / "lists" / "LISTS_LARGE_RECORD.json").write_text("{}", encoding="utf-8")                  # most of the listed members do not exist
    (root / "unlisted_big.bin").write_bytes(b"0" * 10)
    ps = tmp_path / "fakepssh.sh"
    with open(str(ps), "w", newline="\n") as fh:
        fh.write('#!/bin/bash\nbash -c "$1"\n')
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir(), d2.mkdir()

    def run(p):
        return subprocess.run([bash_exe(), str(p)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=120)
    r = run(pull_variant(tmp_path, "pull1.sh", d1, root, ps))
    assert r.returncode == 0 and "PULL-OK" in r.stdout, r.stdout
    got = sorted(str(p.relative_to(d1 / "pod_add106")).replace("\\", "/") for p in (d1 / "pod_add106").rglob("*") if p.is_file())
    assert got == ["ENC_CONTROLS.json", "lists/LISTS_LARGE_RECORD.json", "read/x.json"]
    r = run(pull_variant(tmp_path, "pull1b.sh", d1, root, ps))                                   # an existing pod_add106 is refused
    assert r.returncode == 2 and "REFUSED" in r.stdout
    time.sleep(1.2)
    r = run(pull_variant(tmp_path, "pull2.sh", d2, root, ps))                                    # a second pull does not collide with the tar of the first
    assert r.returncode == 0 and "PULL-OK" in r.stdout, r.stdout
    assert len(sorted(root.glob("pull_add106_*.tar"))) == 2
    empty = tmp_path / "empty_root"
    empty.mkdir()
    d3 = tmp_path / "d3"
    d3.mkdir()
    r = run(pull_variant(tmp_path, "pull3.sh", d3, empty, ps))                                   # no member at all: failed pull, nothing made
    assert r.returncode == 1 and "PULL-FAILED" in r.stdout and not (d3 / "pod_add106").exists()


def test_texts_hold_no_banned_words_and_paths_are_short():
    for f in ("read_cropscale10k.py", "podread_cropscale10k.sh", "push_cropscale_read.sh", "pull_cropscale.sh", "test_cropscale_read.py", "CROPSCALE_READ_DEFS.md"):
        p = HERE / f
        if p.is_file():
            t = p.read_text(encoding="utf-8")
            assert not [w for w in BANNED if w in t], f
            if f.endswith((".sh", ".py")):
                assert "\r" not in t
            assert len(str(p)) < 200 and len("cropscale/read/" + f) < 200
    for f in ("read_cropscale10k.py", "podread_cropscale10k.sh", "push_cropscale_read.sh", "pull_cropscale.sh"):                  # no pod address and no reading of the pod tool's folder
        t = (HERE / f).read_text(encoding="utf-8")
        assert "v8pod/" not in t.replace("/c/Users/Admin/v8pod/pssh.sh", "")
    for f in ("read_cropscale10k.py", "podread_cropscale10k.sh"):
        assert "pod_add" not in (HERE / f).read_text(encoding="utf-8")


def test_selftest_runs():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(HERE / "read_cropscale10k.py"), "--selftest"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
    assert r.stdout.strip().splitlines()[-1] == "ZZ SELFTEST-READ-CROPSCALE10K ok: True", r.stdout[-400:]
