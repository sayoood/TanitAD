"""Tests of dscale_precheck.py (card PREREG_V8F_PROOF_ADD45 with note N1, controls d1, d2, d2j, counts, hashes). numpy only, tiny synthetic npz files (230 corpus clips + 210 new clips, 6 frames each).
Evidence class: MEASURED (synthetic). NOT RUN by the builder.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider test_dscale_precheck.py --basetemp <NEW folder two levels deep>
The checker under test is dscale_precheck.py next to this file, or the file named by DSCALE_PRECHECK_FILE (run_mutations_u.py points it at a mutated copy).
Every test holds at least one arm that must FAIL (a planted wrong value): a planted shared evaluation id, a planted gap difference, a dropped row, a planted join difference, a wrong count.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PFILE = Path(os.environ.get("DSCALE_PRECHECK_FILE", str(HERE / "dscale_precheck.py")))
spec = importlib.util.spec_from_file_location("dscale_precheck_under_test", str(PFILE))
P = importlib.util.module_from_spec(spec)
sys.modules["dscale_precheck_under_test"] = P
spec.loader.exec_module(P)

N_CORPUS, N_NEW, M = 230, 210, 6                                                     # clips in the corpus list, new clips (d2n needs at least 200), frames per clip (k = 0, 2, ..., 10)


def make_rows(cid_index, rng):
    k = (2 * np.arange(M)).astype(np.int64)
    lead = (rng.random(M) < 0.6).astype(np.int64)
    gap = np.where(lead == 1, rng.uniform(5, 55, M), np.where(rng.random(M) < 0.5, 7.0, np.nan))      # a finite gap on some rows WITHOUT a lead: the join must read NaN there
    return k, lead, gap.astype(np.float64)


def write_lead(path, rows):
    sh, ks, ld, gp = [], [], [], []
    for cid, (k, lead, gap) in rows.items():
        sh += [P.sha12(cid)] * len(k)
        ks.append(k)
        ld.append(lead)
        gp.append(gap)
    np.savez(str(path), sha12=np.array(sh), k=np.concatenate(ks), lead=np.concatenate(ld), gap=np.concatenate(gp))


def write_join(path, rows, order, edit=None):
    off, n, lv, lg = [], [], [], []
    pos = 0
    for cid in order:
        k, lead, gap = rows[cid]
        off.append(pos)
        n.append(len(k))
        pos += len(k)
        lv.append(lead)
        lg.append(np.where(lead == 1, gap, np.nan))
    lg = np.concatenate(lg)
    if edit:
        edit(lg)
    np.savez(str(path), clip=np.array(order), off=np.array(off, np.int64), n=np.array(n, np.int64), lead_valid=np.concatenate(lv), lead_gap=lg)


def build(tmp, mod_union=None, mod_join=None, eval_ids=("e00", "e01", "e02", "e03"), union_has_eval=False, new_no_gap=False):
    """Writes the cache manifest, the corpus list, both lead files, the join file and the evaluation manifest. Returns the argv of dscale_precheck.main (without --out/--plant)."""
    tmp = Path(tmp)
    (tmp / "cache" / "train").mkdir(parents=True)
    corpus = ["c%03d" % i for i in range(N_CORPUS)]
    new = ["n%03d" % i for i in range(N_NEW)]
    union = corpus + new
    rng = np.random.default_rng(7)
    rows = {cid: make_rows(i, rng) for i, cid in enumerate(union)}
    if new_no_gap:                                                                         # new clips with a presence label on every frame and no gap label at all (join and lead file alike)
        for cid in new:
            k, lead, gap = rows[cid]
            rows[cid] = (k, np.ones_like(lead), np.full_like(gap, np.nan))
    ids_in_manifest = union + (["e00"] if union_has_eval else [])
    (tmp / "cache" / "train" / "manifest.jsonl").write_text("\n".join(json.dumps({"clip": c}) for c in ids_in_manifest) + "\n", encoding="utf-8")
    (tmp / "bset.txt").write_text("\n".join(corpus) + "\n", encoding="utf-8")
    (tmp / "eval_manifest.jsonl").write_text("\n".join(json.dumps({"clip": c}) for c in eval_ids) + "\n", encoding="utf-8")
    write_join(tmp / "join.npz", rows, union, mod_join)
    urows = {c: tuple(a.copy() for a in v) for c, v in rows.items()}
    if mod_union:
        mod_union(urows, corpus)
    write_lead(tmp / "lead_union.npz", urows)
    write_lead(tmp / "lead_corpus.npz", {c: rows[c] for c in corpus})
    return ["--union-cache", str(tmp / "cache"), "--bset-list", str(tmp / "bset.txt"), "--lead-union", str(tmp / "lead_union.npz"), "--lead-corpus", str(tmp / "lead_corpus.npz"),
            "--eval-manifest", str(tmp / "eval_manifest.jsonl"), "--join-union", str(tmp / "join.npz"), "--expect-union", str(N_CORPUS + N_NEW + (1 if union_has_eval else 0)), "--expect-corpus", str(N_CORPUS)]


def go(tmp, argv, extra=()):
    out = Path(tmp) / "PRECHECK.json"
    rc = P.main(list(argv) + ["--out", str(out)] + list(extra))
    return rc, (json.load(open(out, encoding="utf-8")) if out.exists() else None)


# ------------------------------------------------------------------------------------------------ the clean world
def test_clean_world_passes_every_check_and_hashes_every_input(tmp_path):
    argv = build(tmp_path)
    rc, r = go(tmp_path, argv, ["--hash", "extra=%s" % (tmp_path / "bset.txt")])
    assert rc == 0 and r["pass"] is True
    c = r["checks"]
    assert c["counts"]["pass"] and c["d1_no_leak"]["pass"] and c["d2_labels_one_clock"]["pass"] and c["d2j_join_matches_lead_file"]["pass"]
    assert c["d2_labels_one_clock"]["n_clips"] == 200 and c["d2_labels_one_clock"]["rows_compared"] == 200 * M and c["d2_labels_one_clock"]["rows_in_one_file_only"] == 0
    assert r["input_sha256"]["bset_list"] == hashlib.sha256((tmp_path / "bset.txt").read_bytes()).hexdigest() == r["input_sha256"]["extra"]
    assert r["input_sha256"]["lead_union"] == hashlib.sha256((tmp_path / "lead_union.npz").read_bytes()).hexdigest()
    assert r["input_sha256"]["lead_union"] != r["input_sha256"]["lead_corpus"]           # the hashes are of the content, one per file


# ------------------------------------------------------------------------------------------------ d1
def test_d1_a_planted_shared_evaluation_id_fails_and_so_does_a_real_one(tmp_path):
    argv = build(tmp_path)
    rc, r = go(tmp_path, argv, ["--plant"])
    assert rc == 2 and r["checks"]["d1_no_leak"]["pass"] is False and r["checks"]["d1_no_leak"]["n_shared_with_union"] == 1 and r["checks"]["d1_no_leak"]["planted"] is True
    t2 = tmp_path / "w2"
    t2.mkdir()
    argv2 = build(t2, union_has_eval=True)                                                # a union that really holds an evaluation clip
    rc2, r2 = go(t2, argv2)
    assert rc2 == 2 and r2["checks"]["d1_no_leak"]["pass"] is False and r2["checks"]["d1_no_leak"]["first_shared"] == ["e00"]
    t3 = tmp_path / "w3"
    t3.mkdir()
    ev = t3 / "ids.txt"
    ev.write_text("e09\nc010\n", encoding="utf-8")                                        # the extra id file (--eval-ids) is read too
    rc3, r3 = go(t3, build(t3), ["--eval-ids", str(ev)])
    assert rc3 == 2 and r3["checks"]["d1_no_leak"]["first_shared"] == ["c010"]


# ------------------------------------------------------------------------------------------------ d2
def _edit_gap(urows, corpus):
    for c in corpus:
        k, lead, gap = urows[c]
        i = int(np.flatnonzero(lead == 1)[0]) if (lead == 1).any() else 0
        gap[i] = gap[i] + 1.0                                                              # 1 m: far above the 1e-4 tolerance
        if not (lead == 1).any():
            gap[0] = 3.0


def _edit_lead(urows, corpus):
    for c in corpus:
        urows[c][1][0] = 1 - urows[c][1][0]


def _drop_last_row(urows, corpus):
    for c in corpus:
        k, lead, gap = urows[c]
        urows[c] = (k[:-1], lead[:-1], gap[:-1])


def test_d2_a_planted_gap_difference_a_flag_difference_and_a_dropped_row_each_fail(tmp_path):
    for i, (name, mod, key) in enumerate((("gap", _edit_gap, "rows_mismatching"), ("lead flag", _edit_lead, "rows_mismatching"), ("dropped row", _drop_last_row, "rows_in_one_file_only"))):
        t = tmp_path / ("w%d" % i)
        t.mkdir()
        rc, r = go(t, build(t, mod_union=mod))
        d2 = r["checks"]["d2_labels_one_clock"]
        assert rc == 2 and d2["pass"] is False and d2[key] > 0, name


def test_d2_tolerance_a_difference_below_1e_4_passes_one_above_fails(tmp_path):
    def small(urows, corpus):
        for c in corpus:
            urows[c][2][np.flatnonzero(urows[c][1] == 1)[:1]] += 5e-5
    t = tmp_path / "a"
    t.mkdir()
    rc, r = go(t, build(t, mod_union=small))
    assert rc == 0 and r["checks"]["d2_labels_one_clock"]["pass"] is True
    t2 = tmp_path / "b"
    t2.mkdir()
    rc2, r2 = go(t2, build(t2, mod_union=_edit_gap))
    assert rc2 == 2


def test_d2_refuses_fewer_than_200_clips_and_samples_with_a_fixed_seed(tmp_path):
    argv = build(tmp_path)
    rc, r = go(tmp_path, argv, ["--n-clips", "100"])
    assert rc == 2 and r["checks"]["d2_labels_one_clock"]["pass"] is False and "REFUSED" in r["checks"]["d2_labels_one_clock"]["reason"]
    lu, lc = P.LeadFile(str(tmp_path / "lead_union.npz")), P.LeadFile(str(tmp_path / "lead_corpus.npz"))
    corpus = P.read_ids(str(tmp_path / "bset.txt"))
    a = P.check_d2(lu, lc, corpus, 200, 45, 1e-4)["sampled"]
    b = P.check_d2(lu, lc, corpus, 200, 45, 1e-4)["sampled"]
    c = P.check_d2(lu, lc, corpus, 200, 46, 1e-4)["sampled"]
    assert a == b and a != c and len(set(a)) == 200
    assert not set(a) & {"n%03d" % i for i in range(N_NEW)}                               # only corpus clips are compared by d2


# ------------------------------------------------------------------------------------------------ d2j
def test_d2j_join_labels_agree_with_the_lead_file_only_where_the_flag_is_one(tmp_path):
    rc, r = go(tmp_path, build(tmp_path))
    assert r["checks"]["d2j_join_matches_lead_file"]["pass"] is True and r["checks"]["d2j_join_matches_lead_file"]["rows_compared"] == 200 * M

    def join_gap_off(lg):                                                                  # every row of the join moved by 1 m
        lg += 1.0
    t = tmp_path / "w1"
    t.mkdir()
    rc1, r1 = go(t, build(t, mod_join=join_gap_off))
    assert rc1 == 2 and r1["checks"]["d2j_join_matches_lead_file"]["pass"] is False and r1["checks"]["d2j_join_matches_lead_file"]["rows_mismatching"] > 0

    def join_gap_without_flag(lg):                                                         # a finite join gap on a frame the lead file marks lead 0
        lg[np.isnan(lg)] = 7.0
    t2 = tmp_path / "w2"
    t2.mkdir()
    rc2, r2 = go(t2, build(t2, mod_join=join_gap_without_flag))
    assert rc2 == 2 and r2["checks"]["d2j_join_matches_lead_file"]["pass"] is False


def test_d2j_a_sampled_clip_missing_from_the_join_fails_and_the_block_is_absent_without_a_join_file(tmp_path):
    argv = build(tmp_path)
    z = np.load(str(tmp_path / "join.npz"))
    keep = np.arange(len(z["clip"])) != 5
    # rewrite the join with clip 5 renamed: its rows then cannot be found by id
    clips = np.array([c if i != 5 else "zzz" for i, c in enumerate(z["clip"])])
    np.savez(str(tmp_path / "join.npz"), clip=clips, off=z["off"], n=z["n"], lead_valid=z["lead_valid"], lead_gap=z["lead_gap"])
    lu = P.LeadFile(str(tmp_path / "lead_union.npz"))
    r = P.check_d2j(str(tmp_path / "join.npz"), [], lu, ["c005", "c006"], 1e-4)
    assert r["pass"] is False and r["clips_absent"] == 1 and keep.sum() == len(clips) - 1
    i = argv.index("--join-union")
    rc, rr = go(tmp_path, argv[:i] + argv[i + 2:])
    assert "d2j_join_matches_lead_file" not in rr["checks"] and "d2n_join_matches_lead_file_new_clips" not in rr["checks"] and "d2g_join_labels_per_group" not in rr["checks"]


# ------------------------------------------------------------------------------------------------ counts, missing inputs
def test_counts_a_wrong_expected_number_fails_and_a_missing_input_exits_3(tmp_path):
    argv = build(tmp_path)
    j = argv.index("--expect-union")
    bad = list(argv)
    bad[j + 1] = str(N_CORPUS + N_NEW + 1)
    rc, r = go(tmp_path, bad)
    assert rc == 2 and r["checks"]["counts"]["pass"] is False and r["checks"]["counts"]["union_manifest_clips"] == N_CORPUS + N_NEW
    k = argv.index("--lead-corpus")
    gone = list(argv)
    gone[k + 1] = str(tmp_path / "no_such.npz")
    out = tmp_path / "x.json"
    assert P.main(gone + ["--out", str(out)]) == 3 and not out.exists()
    # a corpus id that is not in the union manifest fails the counts
    (tmp_path / "bset.txt").write_text("\n".join(["c%03d" % i for i in range(N_CORPUS - 1)] + ["unknown_clip"]) + "\n", encoding="utf-8")
    rc2, r2 = go(tmp_path, argv)
    assert r2["checks"]["counts"]["pass"] is False and r2["checks"]["counts"]["corpus_ids_missing_from_union"] == 1


def test_the_lead_file_must_carry_its_four_keys(tmp_path):
    argv = build(tmp_path)
    z = np.load(str(tmp_path / "lead_union.npz"))
    np.savez(str(tmp_path / "lead_union.npz"), sha12=z["sha12"], k=z["k"], lead=z["lead"])        # no gap
    with pytest.raises(SystemExit) as e:
        go(tmp_path, argv)
    assert "gap" in str(e.value)


def test_the_clock_check_file_is_recorded_with_its_sha256_and_never_recomputed(tmp_path):
    cc = tmp_path / "clock.json"
    cc.write_text('{"shift0": true}', encoding="utf-8")
    rc, r = go(tmp_path, build(tmp_path), ["--clock-check-json", str(cc)])
    assert r["clock_check_of_the_tree_owner_session"]["sha256"] == hashlib.sha256(cc.read_bytes()).hexdigest()
    assert r["clock_check_of_the_tree_owner_session"]["sha256"] != hashlib.sha256(b"x").hexdigest()


def test_the_d2_columns_are_the_flag_and_the_gap():
    assert P.D2_COLUMNS == ("lead", "gap")
    a = {"k": np.array([0, 2]), "lead": np.array([1, 1]), "gap": np.array([10.0, 20.0])}
    b = {"k": np.array([0, 2]), "lead": np.array([1, 1]), "gap": np.array([10.0, 21.0])}
    assert P.compare_clip(a, b, 1e-4) == (2, 0, 1)
    assert P.compare_clip(a, a, 1e-4) == (2, 0, 0)
    assert P.compare_clip(a, {"k": np.array([0, 4]), "lead": a["lead"], "gap": a["gap"]}, 1e-4)[1] == 2


# ------------------------------------------------------------------------------------------------ d2n and d2g (the NEW clips: the join file must carry labels there)
def test_d2n_and_d2g_pass_on_the_clean_world_and_report_per_group_counts(tmp_path):
    rc, r = go(tmp_path, build(tmp_path))
    c = r["checks"]
    assert rc == 0 and c["d2n_join_matches_lead_file_new_clips"]["pass"] and c["d2n_join_matches_lead_file_new_clips"]["sampled"] == 200 and c["d2n_join_matches_lead_file_new_clips"]["new_clips"] == N_NEW
    g = c["d2g_join_labels_per_group"]
    assert g["pass"] and g["new"]["clips"] == 200 and g["corpus"]["clips"] == 200 and g["new"]["gap_label_frames"] > 0 and g["new"]["presence_frames"] > 0 and g["corpus"]["gap_label_frames"] > 0
    assert g["new"]["frames"] == 200 * M


def test_d2n_a_join_that_is_wrong_on_the_new_clips_only_fails_while_d2j_on_the_corpus_passes(tmp_path):
    def new_only(lg):
        lg[-N_NEW * M:] += 1.0                                                                # the last N_NEW clips of the union order are the new ones
    rc, r = go(tmp_path, build(tmp_path, mod_join=new_only))
    c = r["checks"]
    assert rc == 2 and c["d2j_join_matches_lead_file"]["pass"] is True
    assert c["d2n_join_matches_lead_file_new_clips"]["pass"] is False and c["d2n_join_matches_lead_file_new_clips"]["rows_mismatching"] > 0


def test_d2g_new_clips_with_presence_but_no_gap_label_fail_the_group_check(tmp_path):
    rc, r = go(tmp_path, build(tmp_path, new_no_gap=True))
    c = r["checks"]
    assert rc == 2 and c["d2n_join_matches_lead_file_new_clips"]["pass"] is True                  # join and lead file agree (both without a gap) ...
    g = c["d2g_join_labels_per_group"]
    assert g["pass"] is False and g["new"]["presence_frames"] > 0 and g["new"]["gap_label_frames"] == 0 and g["new"]["pass"] is False and g["corpus"]["pass"] is True   # ... and the group check sees it


def test_d2n_refuses_when_fewer_than_200_new_clips_exist(tmp_path):
    z = tmp_path / "few"
    z.mkdir()
    argv = build(z)
    i = argv.index("--lead-union")
    # shrink the union lead file to 150 of the new clips (rows of the other new clips removed): the pool of new clips with rows falls under 200
    lu = np.load(argv[i + 1])
    drop = {P.sha12("n%03d" % j) for j in range(60)}
    keep = np.array([s not in drop for s in lu["sha12"]])
    np.savez(argv[i + 1], sha12=lu["sha12"][keep], k=lu["k"][keep], lead=lu["lead"][keep], gap=lu["gap"][keep])
    rc, r = go(z, argv)
    assert rc == 2 and r["checks"]["d2n_join_matches_lead_file_new_clips"]["pass"] is False and "fewer than" in r["checks"]["d2n_join_matches_lead_file_new_clips"]["reason"]
