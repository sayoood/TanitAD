"""make_kit_b.py -- card PREREG_V8F_PROOF_ADD95 (GAPRULE10KB): writes the variants of the kit of card ADD94 that the new card needs, each by asserted edits of the file of card ADD94.
  make_planted_joins_b.py   the limit clause of control j1 (2 percent below 40 m, 5 percent over all valid frames), the count below 40 m in the record, --same-as DIR (arrays equal to an earlier build)
  read_gaprule10kb.py       the card name, the name prefix GAPRULE10KB, the join module of this card, and j1 requires the comparison with the earlier build
  test_gaprule_b.py         the tests of the kit on the two variants plus the tests of the changed clause
  pod_check_wrapper_jb.sh   the pod check of control j2 in the root of this card
Refuses to write over an existing file. Usage: python make_kit_b.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(name):
    with open(os.path.join(HERE, name), encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def sub(src, old, new, count=1, what=""):
    n = src.count(old)
    if n != count:
        sys.exit("REFUSED: %s: expected %d of [%s], found %d" % (what, count, old[:60], n))
    return src.replace(old, new)


def wr(name, text):
    p = os.path.join(HERE, name)
    if os.path.exists(p):
        sys.exit("REFUSED: %s exists" % name)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote", name, len(text.splitlines()), "lines")


SAME_AS = '''def same_as(out_dir, ref_dir):
    """every array of each of the six joins against the same file of an earlier build -> {"dir", "arms": {arm: {...}}, "all_equal"}"""
    res = {"dir": ref_dir, "arms": {}}
    for arm, _k, _t, _l in ARMS:
        pa, pb = os.path.join(out_dir, arm + ".npz"), os.path.join(ref_dir, arm + ".npz")
        if not os.path.isfile(pa) or not os.path.isfile(pb):
            res["arms"][arm] = {"all_equal": False, "why": "a file is missing"}
            continue
        with np.load(pa, allow_pickle=False) as za, np.load(pb, allow_pickle=False) as zb:
            keys_eq = sorted(za.files) == sorted(zb.files)
            eq = {k: equal_key(za[k], zb[k]) for k in za.files if k in zb.files}
            worst = None
            if "lead_gap" in za.files and "lead_gap" in zb.files:
                x, y = np.asarray(za["lead_gap"], np.float64), np.asarray(zb["lead_gap"], np.float64)
                if x.shape == y.shape:
                    m = np.isfinite(x) & np.isfinite(y)
                    worst = float(np.max(np.abs(x[m] - y[m]))) if m.any() else 0.0
        res["arms"][arm] = {"all_equal": bool(keys_eq and len(eq) > 0 and all(eq.values())), "keys_equal": bool(keys_eq), "arrays_compared": len(eq),
                            "lead_gap_largest_abs_difference": worst, "sha256_earlier": sha_file(pb)}
    res["all_equal"] = bool(all(x["all_equal"] for x in res["arms"].values()))
    return res


'''

NEW_TESTS = '''

# ------------------------------------------------------------------------------------------------ the changed clause of control j1 (card ADD95)
import copy          # noqa: E402


def _rec_with(clean, arm, **kw):
    r = copy.deepcopy(clean)
    r["arms"][arm].update(kw)
    return r


def test_b_limit_share_between_the_bars_passes(joins):
    r = _rec_with(joins["clean"], "GC20", fraction_changed_by_limits=0.0296, fraction_changed_by_limits_below_40m=0.0)
    assert MPJ.j1_check(r)["PASS"] is True


def test_b_limit_share_above_the_overall_bar_is_red(joins):
    c = MPJ.j1_check(_rec_with(joins["clean"], "GC20", fraction_changed_by_limits=0.0501))
    assert c["PASS"] is False and any("bar 0.05" in x for x in c["failures"])


def test_b_limit_share_below_40m_above_its_bar_is_red(joins):
    c = MPJ.j1_check(_rec_with(joins["clean"], "GR1", fraction_changed_by_limits_below_40m=0.0201))
    assert c["PASS"] is False and any("below 40 m" in x for x in c["failures"])


def test_b_far_gaps_trip_only_the_overall_bar(tmp_path):
    J = MPJ.synthetic()
    J["lead_gap"] = np.random.default_rng(5).uniform(52.0, 60.0, len(J["lead_gap"])).astype(np.float32)
    rec = MPJ.build(J, str(tmp_path / "far"))
    a = rec["arms"]["GC20"]
    assert a["fraction_changed_by_limits"] > 0.05 and a["n_valid_frames_below_40m"] == 0 and a["fraction_changed_by_limits_below_40m"] == 0.0
    assert rec["j1"]["PASS"] is False and all("below 40 m" not in x for x in rec["j1"]["failures"])


def test_b_short_gaps_trip_the_bar_below_40m(tmp_path):
    J = MPJ.synthetic()
    J["lead_gap"] = np.random.default_rng(6).uniform(0.5, 1.5, len(J["lead_gap"])).astype(np.float32)
    rec = MPJ.build(J, str(tmp_path / "short"))
    assert rec["arms"]["GR2"]["fraction_changed_by_limits_below_40m"] > 0.02
    assert rec["j1"]["PASS"] is False and any("below 40 m" in x for x in rec["j1"]["failures"])


def test_b_same_as_equal_and_different(tmp_path):
    J = MPJ.synthetic()
    MPJ.build(J, str(tmp_path / "a"))
    rec = MPJ.build(J, str(tmp_path / "b"))
    sa = MPJ.same_as(str(tmp_path / "b"), str(tmp_path / "a"))
    assert sa["all_equal"] is True and all(x["lead_gap_largest_abs_difference"] == 0.0 for x in sa["arms"].values())
    rec["same_as"] = sa
    assert MPJ.j1_check(rec)["PASS"] is True
    J2 = dict(J)
    J2["lead_gap"] = (np.asarray(J["lead_gap"]) + np.float32(0.25)).astype(np.float32)
    MPJ.build(J2, str(tmp_path / "c"))
    sb = MPJ.same_as(str(tmp_path / "b"), str(tmp_path / "c"))
    assert sb["all_equal"] is False
    rec["same_as"] = sb
    assert MPJ.j1_check(rec)["PASS"] is False
    assert MPJ.same_as(str(tmp_path / "b"), str(tmp_path / "nowhere"))["all_equal"] is False


def test_b_reader_j1_needs_the_comparison(joins):
    r = copy.deepcopy(joins["clean"])
    assert RG.ctl_j1(r, str(joins["base"] / "clean"), None)["PASS"] is True
    r.pop("same_as")
    c = RG.ctl_j1(r, str(joins["base"] / "clean"), None)
    assert c["PASS"] is False and any("earlier build" in x for x in c["failures"])


def test_b_names_carry_the_prefix_of_this_card():
    assert RG.CARD == "PREREG_V8F_PROOF_ADD95" and RG.PRE == "GAPRULE10KB"


def test_b_main_refuses_without_same_as(tmp_path):
    with pytest.raises(SystemExit) as ex:
        MPJ.main(["--eval-join", "x.npz", "--out-dir", str(tmp_path / "o")])
    assert "REFUSED" in str(ex.value)
'''


def main():
    # ---- make_planted_joins_b.py
    s = rd("make_planted_joins.py")
    s = sub(s, '"""make_planted_joins.py -- the six planted joins of card PREREG_V8F_PROOF_ADD94 (GAPRULE10K).',
            '"""make_planted_joins_b.py -- the six planted joins of card PREREG_V8F_PROOF_ADD95 (GAPRULE10KB): the tool of card ADD94 with the limit clause of control j1 changed (2 percent below 40 m,\n'
            '5 percent over all valid frames), the count below 40 m in the record and --same-as DIR (arrays equal to an earlier build, required). Generated by make_kit_b.py.', 1, "head")
    s = sub(s, "MAX_LIMITED_FRACTION = 0.02\n", "MAX_LIMITED_FRACTION = 0.02          # on the frames with a label gap below NEAR_M\nMAX_LIMITED_FRACTION_ALL = 0.05\nNEAR_M = 40.0\n", 1, "constants")
    s = sub(s, 'rec = {"card": "PREREG_V8F_PROOF_ADD94"', 'rec = {"card": "PREREG_V8F_PROOF_ADD95"', 1, "record card")
    s = sub(s, "        n_lim = 0\n", "        n_lim = 0\n        n_lim_near = 0\n        near = v & (gap < NEAR_M)\n", 1, "near mask")
    s = sub(s, "            n_lim = int(((lim != pre) & v).sum())\n", "            n_lim = int(((lim != pre) & v).sum())\n            n_lim_near = int(((lim != pre) & near).sum())\n", 1, "near count")
    s = sub(s, '"fraction_changed_by_limits": float(n_lim / max(int(v.sum()), 1)),',
            '"fraction_changed_by_limits": float(n_lim / max(int(v.sum()), 1)), "n_valid_frames_below_40m": int(near.sum()), "n_changed_by_limits_below_40m": n_lim_near,\n'
            '                            "fraction_changed_by_limits_below_40m": float(n_lim_near / max(int(near.sum()), 1)),', 1, "record fields")
    s = sub(s, '            if not a["fraction_changed_by_limits"] <= MAX_LIMITED_FRACTION:\n'
               '                bad.append("%s: %.4f of the valid frames are changed by the limits (bar 0.02)" % (arm, a["fraction_changed_by_limits"]))\n',
            '            if not a["fraction_changed_by_limits_below_40m"] <= MAX_LIMITED_FRACTION:\n'
            '                bad.append("%s: %.4f of the valid frames with a label gap below 40 m are changed by the limits (bar 0.02)" % (arm, a["fraction_changed_by_limits_below_40m"]))\n'
            '            if not a["fraction_changed_by_limits"] <= MAX_LIMITED_FRACTION_ALL:\n'
            '                bad.append("%s: %.4f of the valid frames are changed by the limits (bar 0.05)" % (arm, a["fraction_changed_by_limits"]))\n', 1, "j1 clause")
    s = sub(s, '            bad.append("G0: differs from the evaluation join")\n',
            '            bad.append("G0: differs from the evaluation join")\n'
            '        sa = rec.get("same_as")\n'
            '        if sa is not None and sa.get("all_equal") is not True:\n'
            '            bad.append("the arrays of the joins differ from the earlier build")\n', 1, "j1 same_as")
    s = sub(s, "def synthetic(seed=3, n_clips=14):\n", SAME_AS + "def synthetic(seed=3, n_clips=14):\n", 1, "same_as function")
    s = sub(s, '    ap.add_argument("--out-dir")\n', '    ap.add_argument("--out-dir")\n    ap.add_argument("--same-as")\n', 1, "argument")
    s = sub(s, '    if not a.eval_join or not a.out_dir:\n        sys.exit("REFUSED: --eval-join and --out-dir are required")\n',
            '    if not a.eval_join or not a.out_dir or not a.same_as:\n        sys.exit("REFUSED: --eval-join, --out-dir and --same-as are required")\n', 1, "required")
    s = sub(s, '    rec["eval_join_sha256"] = sha_file(a.eval_join)\n',
            '    rec["eval_join_sha256"] = sha_file(a.eval_join)\n    rec["same_as"] = same_as(a.out_dir, a.same_as)\n    rec["j1"] = j1_check(rec)\n', 1, "main same_as")
    wr("make_planted_joins_b.py", s)

    # ---- read_gaprule10kb.py
    r = rd("read_gaprule10k.py")
    r = sub(r, '"""read_gaprule10k.py -- read of card PREREG_V8F_PROOF_ADD94 (GAPRULE10K):',
            '"""read_gaprule10kb.py -- read of card PREREG_V8F_PROOF_ADD95 (GAPRULE10KB; the reader of card ADD94 with the card name, the name prefix, the join module of the new card and a j1\n'
            'control that requires the comparison with the earlier build; generated by make_kit_b.py):', 1, "reader head")
    r = sub(r, 'CARD, PRE = "PREREG_V8F_PROOF_ADD94", "GAPRULE10K"', 'CARD, PRE = "PREREG_V8F_PROOF_ADD95", "GAPRULE10KB"', 1, "reader card")
    r = sub(r, 'if os.path.isfile(os.path.join(_p, "make_planted_joins.py")):', 'if os.path.isfile(os.path.join(_p, "make_planted_joins_b.py")):', 1, "reader path")
    r = sub(r, "import make_planted_joins as MPJ", "import make_planted_joins_b as MPJ", 1, "reader import")
    r = sub(r, '    c = MPJ.j1_check(rec)\n    bad = list(c["failures"])\n',
            '    c = MPJ.j1_check(rec)\n    bad = list(c["failures"])\n'
            '    sa = rec.get("same_as")\n'
            '    if not isinstance(sa, dict) or sa.get("all_equal") is not True:\n'
            '        bad.append("no comparison with the earlier build in the record, or it differs")\n', 1, "reader j1")
    wr("read_gaprule10kb.py", r)

    # ---- test_gaprule_b.py
    t = rd("test_gaprule.py")
    t = sub(t, "import make_planted_joins as MPJ ", "import make_planted_joins_b as MPJ ", 1, "test import 1")
    t = sub(t, "import read_gaprule10k as RG ", "import read_gaprule10kb as RG ", 1, "test import 2")
    t = sub(t, '    return {"J": J, "clean": MPJ.build(J, str(base / "clean")), "base": base}\n',
            '    clean = MPJ.build(J, str(base / "clean"))\n'
            '    MPJ.build(J, str(base / "earlier"))\n'
            '    clean["same_as"] = MPJ.same_as(str(base / "clean"), str(base / "earlier"))\n'
            '    clean["j1"] = MPJ.j1_check(clean)\n'
            '    return {"J": J, "clean": clean, "base": base}\n', 1, "fixture")
    wr("test_gaprule_b.py", t + NEW_TESTS)

    # ---- pod_check_wrapper_jb.sh
    c = rd("pod_check_wrapper_j.sh")
    c = sub(c, "# pod_check_wrapper_j.sh -- card ADD94 (GAPRULE10K), control j2:", "# pod_check_wrapper_jb.sh -- card ADD95 (GAPRULE10KB), control j2 (pod_check_wrapper_j.sh with the root of this card):", 1, "check head")
    c = sub(c, "# Writes only into /workspace/v8proof/gaprule10k.", "# Writes only into /workspace/v8proof/gaprule10kb.", 1, "check comment")
    c = sub(c, "GP=/workspace/v8proof/gaprule10k;", "GP=/workspace/v8proof/gaprule10kb;", 1, "check root")
    wr("pod_check_wrapper_jb.sh", c)
    print("ZZ MAKE-KIT-B DONE")


if __name__ == "__main__":
    main()
