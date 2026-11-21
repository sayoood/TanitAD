"""Tests of ``dscale10k_merge.py`` and ``dscale10k_check.py`` (PREREG_V8F_DSCALE10K). Synthetic fixtures in the REAL formats (v9 release npz + manifest, lead npz, speed jsonl,
native tracks .pt, cache manifest.jsonl). Every check has a deliberate-regression arm that must go RED.

Run (serially, dev box):  CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="<tree>/stack;<tree>/stack/scripts;<tree>;<this folder>" python -m pytest test_dscale10k.py --basetemp F:/q/ds10k_tmp/pt
The tests that import the code of record (``tanitad``, the trainer) skip when it is not on the path.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "tree", "stack"), os.path.join(HERE, "tree", "stack", "scripts"), os.path.join(HERE, "tree")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
import dscale10k_check as C  # noqa: E402
import dscale10k_merge as M  # noqa: E402

try:
    from tanitad.data import v8f_mini_data as D  # noqa: E402
    from tanitad.data import v9_labels as V9  # noqa: E402
except Exception:                                      # the code of record is not on the path
    D = V9 = None
needs_record = pytest.mark.skipif(D is None, reason="the code of record (cdbff971) is not on PYTHONPATH")

CONS = list(D.CONS_FIELDS) if D else ["lat_theta_deg", "lc_t_cross_s", "lc_d_cross_m", "v_a_ms"]
RCV = ("A30", "A50", "A80")
GEOM = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "STOP_POINT", "SPEED_BAND")
PROV = {g: ("geometry" if g in GEOM else "vlm") for g in C.GOAL22}
PROV["LANE_CHANGE_L"] = PROV["LANE_CHANGE_R"] = "sam3|vlm"
GEOM_MASK = sum(1 << i for i, g in enumerate(C.GOAL22) if g in GEOM)
TL_MASK = sum(1 << i for i, g in enumerate(C.GOAL22) if g.startswith("TRAFFIC_LIGHT"))
FIELDS = (["k", "now_s", "nav_token", "nav_side_next", "nav_args_valid", "nav_d_next_m", "nav_d_end_m", "nav_dyaw_next_deg", "nav_lookahead_m", "nav_t_next_s", "nav_token_ttime",
           "goal_y", "goal_w", "lat_cls_a", "lon_cls", "lat_allowed_a", "lon_allowed", "lead_valid"] + CONS + [f"rc_{v}_{c}" for v in RCV for c in ("x", "y", "psi", "valid")])
NAMES = {"corpus": [f"corpus-{i:03d}" for i in range(40)], "tenk": [f"tenk-{i:03d}" for i in range(30)]}
HOLD = (5, 8)


def _shape(i):
    return (20, 0) if i % 2 == 0 else (24, 0)


def make_release(path, cids, seed, vlm_corpus=False):
    """A v9 release in the real format: ``row__<field>`` + ``clip__{sid,row0,n_rows,k0}`` + ``<stem>.manifest.json``."""
    rng = np.random.default_rng(seed)
    rows = {f: [] for f in FIELDS}
    tab = []
    pos = 0
    for i, cid in enumerate(cids):
        n, k0 = _shape(i)
        tab.append((M.stable_episode_id(cid), pos, n, k0))
        pos += n
        k = k0 + np.arange(n)
        for f in FIELDS:
            if f == "k":
                v = k.astype(np.int32)
            elif f == "now_s":
                v = k * 0.1
            elif f == "nav_token":
                v = rng.integers(0, 3, n).astype(np.int8)
            elif f in ("goal_y", "goal_w"):
                v = (rng.integers(0, 1 << 22, n).astype(np.uint64) & np.uint64(GEOM_MASK))
                if vlm_corpus and f == "goal_w":
                    v = v | (np.uint64(TL_MASK) * (rng.random(n) < 0.3).astype(np.uint64))
            elif f in ("lat_cls_a", "lon_cls"):
                v = rng.integers(0, 5, n).astype(np.int16)
            elif f in ("lat_allowed_a", "lon_allowed"):
                v = rng.integers(0, 255, n).astype(np.uint8)
            elif f == "lead_valid":
                v = rng.integers(0, 2, n).astype(np.int8)
            elif f in ("nav_args_valid",) or f.endswith("_valid"):
                v = np.ones(n)
            else:
                v = rng.normal(0, 10, n)
                if f in ("lc_t_cross_s", "lc_d_cross_m"):
                    v[:] = np.nan
            rows[f].append(v)
    arr = {"row__" + f: np.concatenate(rows[f]) for f in FIELDS}
    arr["clip__sid"] = np.array([t[0] for t in tab], np.int64)
    arr["clip__row0"] = np.array([t[1] for t in tab], np.int64)
    arr["clip__n_rows"] = np.array([t[2] for t in tab], np.int64)
    arr["clip__k0"] = np.array([t[3] for t in tab], np.int64)
    M.write_npz_deterministic(path, arr)
    man = {"schema": M.SCHEMA, "split": "train", "n_clips": len(cids), "n_rows": int(pos), "row_key": "(sid, k)", "clock_rule": "now_s = k * dt", "vocab": {"GOAL22": list(C.GOAL22), "GOAL_PROV": PROV},
           "literals": {"grid_dt_s": 0.1}, "fields": FIELDS, "npz_md5": M.md5_file(path), "wall_s": 1.0}
    json.dump(man, open(os.path.splitext(str(path))[0] + ".manifest.json", "w", encoding="utf-8"), sort_keys=True)


def make_side(dirp, tag, cids, seed):
    rng = np.random.default_rng(seed + 1)
    with open(os.path.join(dirp, f"manifest_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for i, c in enumerate(cids):
            f.write(json.dumps({"clip": c, "n_frames": _shape(i)[0] // 2}) + "\n")
    sh, ks, ld, gp = [], [], [], []
    for c in cids:
        k = np.arange(0, 20, 2)
        sh += [M.sha12(c)] * len(k); ks.append(k); ld.append(rng.integers(0, 2, len(k)).astype(np.int8)); gp.append(rng.random(len(k)).astype(np.float32) * 50)
    np.savez(os.path.join(dirp, f"lead_{tag}.npz"), sha12=np.array(sh), k=np.concatenate(ks), lead=np.concatenate(ld), gap=np.concatenate(gp))
    with open(os.path.join(dirp, f"speed_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for c in cids:
            f.write(json.dumps({"clip_id": c, "limit_kmh": 50, "one_hot": [0, 1, 0, 0], "valid": 1}) + "\n")
    import torch
    torch.save({"tracks": {c: torch.tensor(rng.normal(0, 1, (22, 4)), dtype=torch.float32) for c in cids}, "meta": {"schema": "native-tracks-v1"}}, os.path.join(dirp, f"tracks_{tag}.pt"))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("world"))
    make_release(os.path.join(d, "v9_corpus.npz"), NAMES["corpus"], 1, vlm_corpus=True)
    make_release(os.path.join(d, "v9_tenk.npz"), NAMES["tenk"], 2)
    make_side(d, "corpus", NAMES["corpus"], 1)
    make_side(d, "tenk", NAMES["tenk"], 2)
    return d


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def run_merge(world, out, hold=HOLD, patch=True):
    w = lambda n: os.path.join(world, n)
    a = NS(corpus_manifest=w("manifest_corpus.jsonl"), tenk_manifest=w("manifest_tenk.jsonl"), holdout=hold, v9=(w("v9_corpus.npz"), w("v9_tenk.npz")),
           lead=(w("lead_corpus.npz"), w("lead_tenk.npz")), speed=(w("speed_corpus.jsonl"), w("speed_tenk.jsonl")), tracks=(w("tracks_corpus.pt"), w("tracks_tenk.pt")), out_dir=out)
    return M.merge_all(a)


@pytest.fixture
def pinned(world, monkeypatch):
    """The real pins are the production files' md5s; the fixtures stand in for them."""
    monkeypatch.setattr(M, "CORPUS_V9_MD5", M.md5_file(os.path.join(world, "v9_corpus.npz")))
    monkeypatch.setattr(M, "TENK_V9_MD5", M.md5_file(os.path.join(world, "v9_tenk.npz")))
    monkeypatch.setattr(M, "EXCLUDED_SHA256", M.holdout_digest(M.holdout_ids(os.path.join(world, "manifest_corpus.jsonl"), *HOLD)))
    return world


@pytest.fixture
def merged(pinned, tmp_path):
    out = str(tmp_path / "m")
    rec = run_merge(pinned, out)
    return pinned, out, rec


# ------------------------------------------------------------------------------------------------ merge tool
def test_ids_mirror_the_code_of_record():
    for c in ("a", "clip-0001", "ñ"):
        if D is not None:
            assert M.sha12(c) == D.sha12(c) and M.stable_episode_id(c) == D.stable_episode_id(c)
    assert M.sha12("a") == hashlib.sha256(b"a").hexdigest()[:12]


def test_holdout_is_the_trainers_exclude_540_600(tmp_path):
    cl = [f"c{i:04d}" for i in range(700)]
    p = tmp_path / "manifest.jsonl"
    p.write_text("".join(json.dumps({"clip": c, "n_frames": 10}) + "\n" for c in cl), encoding="utf-8")
    ids_ = M.holdout_ids(str(p))
    assert len(ids_) == 60 and ids_ == sorted(cl, key=M.sha12)[540:600]
    assert M.holdout_digest(ids_) == hashlib.sha256("\n".join(ids_).encode()).hexdigest()


def test_merge_passes_and_counts(merged):
    w, out, rec = merged
    assert rec["c_merge_pass"] and rec["holdout"]["n"] == 3
    assert rec["v9"]["n_clips"] == 40 - 3 + 30 and rec["manifest"]["n_union"] == 67
    assert rec["lead"]["n_corpus_rows_dropped"] == 3 * 10 and rec["speed"]["n_corpus_dropped"] == 3 and rec["tracks"]["n_corpus_dropped"] == 3
    man = json.load(open(os.path.join(out, "v9_labels_union.manifest.json"), encoding="utf-8"))
    assert man["npz_md5"] == rec["v9"]["md5"] and man["n_clips"] == 67


def test_merge_is_deterministic(pinned, tmp_path):
    a, b = run_merge(pinned, str(tmp_path / "a")), run_merge(pinned, str(tmp_path / "b"))
    assert {k: v["sha256"] for k, v in a["outputs"].items()} == {k: v["sha256"] for k, v in b["outputs"].items()}


@needs_record
def test_union_release_loads_in_the_trainers_loader(merged):
    w, out, rec = merged
    rel = V9.load_v9_release(os.path.join(out, "v9_labels_union.npz"), expect_md5=rec["v9"]["md5"])
    assert len(rel.clips) == 67 and rel.n_rows == rec["v9"]["n_rows"]
    assert M.stable_episode_id("corpus-005") not in rel.clips or "corpus-005" not in set(M.holdout_ids(os.path.join(w, "manifest_corpus.jsonl"), *HOLD))


def _rewrite(path, fn):
    Z = M.load_npz(path)
    fn(Z)
    M.write_npz_deterministic(path, Z)


def test_red_a_changed_corpus_byte_fails_c_merge(merged):
    w, out, rec = merged
    ids_ = M.holdout_ids(os.path.join(w, "manifest_corpus.jsonl"), *HOLD)
    v = lambda: M.verify_v9(os.path.join(w, "v9_corpus.npz"), os.path.join(w, "v9_tenk.npz"), os.path.join(out, "v9_labels_union.npz"), ids_)
    assert v() == []
    sid0 = M.stable_episode_id([c for c in NAMES["corpus"] if c not in ids_][0])
    def flip(Z):
        j = int(np.where(Z["clip__sid"] == sid0)[0][0])
        Z["row__goal_y"][int(Z["clip__row0"][j]) + 3] ^= np.uint64(1)
    _rewrite(os.path.join(out, "v9_labels_union.npz"), flip)
    assert v() and "goal_y" in " ".join(v())


def test_red_tenk_row_and_each_other_file(merged):
    w, out, rec = merged
    ids_ = M.holdout_ids(os.path.join(w, "manifest_corpus.jsonl"), *HOLD)
    f = lambda n: os.path.join(w, n)
    # lead
    assert M.verify_lead(f("lead_corpus.npz"), f("lead_tenk.npz"), os.path.join(out, "lead_union.npz"), ids_) == []
    _rewrite(os.path.join(out, "lead_union.npz"), lambda Z: Z["gap"].__setitem__(0, Z["gap"][0] + 1))
    assert M.verify_lead(f("lead_corpus.npz"), f("lead_tenk.npz"), os.path.join(out, "lead_union.npz"), ids_)
    # speed
    assert M.verify_speed(f("speed_corpus.jsonl"), f("speed_tenk.jsonl"), os.path.join(out, "speed_union.jsonl"), ids_) == []
    sp = os.path.join(out, "speed_union.jsonl")
    b = open(sp, "rb").read()
    open(sp, "wb").write(b.replace(b'"limit_kmh": 50', b'"limit_kmh": 51', 1))
    assert M.verify_speed(f("speed_corpus.jsonl"), f("speed_tenk.jsonl"), sp, ids_)
    # tracks
    import torch
    assert M.verify_tracks(f("tracks_corpus.pt"), f("tracks_tenk.pt"), os.path.join(out, "native_tracks_union.pt"), ids_) == []
    o = torch.load(os.path.join(out, "native_tracks_union.pt"), weights_only=True)
    o["tracks"]["tenk-000"][0, 0] += 1.0
    torch.save(o, os.path.join(out, "native_tracks_union.pt"))
    assert M.verify_tracks(f("tracks_corpus.pt"), f("tracks_tenk.pt"), os.path.join(out, "native_tracks_union.pt"), ids_)


def test_red_a_present_holdout_clip_fails(merged):
    w, out, rec = merged
    ids_ = M.holdout_ids(os.path.join(w, "manifest_corpus.jsonl"), *HOLD)
    # (a) verification: the union built WITHOUT dropping the holdout is refused
    M.merge_v9(os.path.join(w, "v9_corpus.npz"), os.path.join(w, "v9_tenk.npz"), [], os.path.join(out, "nodrop.npz"))
    prob = M.verify_v9(os.path.join(w, "v9_corpus.npz"), os.path.join(w, "v9_tenk.npz"), os.path.join(out, "nodrop.npz"), ids_)
    assert prob and any("holdout" in x for x in prob)
    # (b) a holdout clip inside the 10k files is refused by the merge
    make_release(os.path.join(out, "tenk_bad.npz"), NAMES["tenk"][:5] + [ids_[0]], 3)
    with pytest.raises(M.MergeError, match="holdout"):
        M.merge_v9(os.path.join(w, "v9_corpus.npz"), os.path.join(out, "tenk_bad.npz"), ids_, os.path.join(out, "x.npz"))
    # (c) a clip in BOTH releases is refused
    make_release(os.path.join(out, "tenk_dup.npz"), NAMES["tenk"][:5] + [NAMES["corpus"][20]], 3)
    with pytest.raises(M.MergeError, match="share"):
        M.merge_v9(os.path.join(w, "v9_corpus.npz"), os.path.join(out, "tenk_dup.npz"), ids_, os.path.join(out, "y.npz"))
    # (d) holdout ids that the release does not hold: wrong manifest
    with pytest.raises(M.MergeError, match="holdout"):
        M.merge_v9(os.path.join(w, "v9_corpus.npz"), os.path.join(w, "v9_tenk.npz"), ["nope"], os.path.join(out, "z.npz"))


def test_merge_refuses_wrong_pinned_md5(world, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "CORPUS_V9_MD5", "0" * 32)
    with pytest.raises(M.MergeError, match="md5"):
        run_merge(world, str(tmp_path / "q"))


def test_merge_refuses_wrong_holdout_digest(world, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "CORPUS_V9_MD5", M.md5_file(os.path.join(world, "v9_corpus.npz")))
    monkeypatch.setattr(M, "TENK_V9_MD5", M.md5_file(os.path.join(world, "v9_tenk.npz")))
    with pytest.raises(M.MergeError, match="excluded_sha256"):
        M.merge_all(NS(**{**_ns(world, str(tmp_path / "q")), "holdout": (M.HOLD_LO, M.HOLD_HI)}))


def _ns(world, out):
    w = lambda n: os.path.join(world, n)
    return dict(corpus_manifest=w("manifest_corpus.jsonl"), tenk_manifest=w("manifest_tenk.jsonl"), holdout=HOLD, v9=(w("v9_corpus.npz"), w("v9_tenk.npz")),
                lead=(w("lead_corpus.npz"), w("lead_tenk.npz")), speed=(w("speed_corpus.jsonl"), w("speed_tenk.jsonl")), tracks=(w("tracks_corpus.pt"), w("tracks_tenk.pt")), out_dir=out)


def test_c_merge_rebuild_and_edit_detected(merged):
    w, out, rec = merged
    ok, md5 = C.c_merge_problems(os.path.join(out, "merge_record.json"))
    assert ok == [] and md5 == rec["v9"]["md5"]
    _rewrite(os.path.join(out, "v9_labels_union.npz"), lambda Z: Z["row__goal_y"].__setitem__(0, Z["row__goal_y"][0] ^ np.uint64(1)))
    bad, _ = C.c_merge_problems(os.path.join(out, "merge_record.json"))
    assert bad


# ------------------------------------------------------------------------------------------------ R_s0
def test_derange_properties_and_determinism(merged):
    w, out, rec = merged
    U, T = os.path.join(out, "v9_labels_union.npz"), os.path.join(w, "v9_tenk.npz")
    r1 = M.derange_v9(U, T, os.path.join(out, "r1.npz"))
    r2 = M.derange_v9(U, T, os.path.join(out, "r2.npz"))
    assert r1["md5"] == r2["md5"] and r1["n_deranged_clips"] == 30 and all(int(s) != int(d) for s, d in r1["donor"].items())
    assert M.verify_derange(U, os.path.join(out, "r1.npz"), T, r1) == []
    r3 = M.derange_v9(U, T, os.path.join(out, "r3.npz"), seed=1)
    assert r3["md5"] != r1["md5"]
    Zu, Zr = M.load_npz(U), M.load_npz(os.path.join(out, "r1.npz"))
    tsid = set(M.load_npz(T)["clip__sid"].tolist())
    for sid, r0, n in zip(Zu["clip__sid"].tolist(), Zu["clip__row0"].tolist(), Zu["clip__n_rows"].tolist()):
        for c in Zu:
            if not c.startswith("row__"):
                continue
            same = Zu[c][r0:r0 + n].tobytes() == Zr[c][r0:r0 + n].tobytes()
            if sid not in tsid:
                assert same, f"corpus row changed: {c}"                     # corpus rows bit-equal
            elif M.is_own_column(c):
                assert same, f"own column moved: {c}"                       # k, now_s, nav_* stay
    assert M.is_own_column("row__nav_token") and M.is_own_column("k") and not M.is_own_column("row__goal_y")
    # the donor's block really arrived
    s0 = next(iter(r1["donor"]))
    j0, jd = int(np.where(Zu["clip__sid"] == int(s0))[0][0]), int(np.where(Zu["clip__sid"] == r1["donor"][s0])[0][0])
    n = int(Zu["clip__n_rows"][j0])
    assert Zr["row__goal_y"][int(Zu["clip__row0"][j0]):][:n].tobytes() == Zu["row__goal_y"][int(Zu["clip__row0"][jd]):][:n].tobytes()


def test_red_derangement_with_a_fixed_point_or_changed_corpus(merged):
    w, out, rec = merged
    U, T = os.path.join(out, "v9_labels_union.npz"), os.path.join(w, "v9_tenk.npz")
    r = M.derange_v9(U, T, os.path.join(out, "r.npz"))
    # a "derangement" that is the identity: every 10k clip keeps its own block
    shutil_copy(U, os.path.join(out, "id.npz")); shutil_copy(os.path.splitext(U)[0] + ".manifest.json", os.path.join(out, "id.manifest.json"))
    assert M.verify_derange(U, os.path.join(out, "id.npz"), T, r)
    # a corpus row changed
    shutil_copy(os.path.join(out, "r.npz"), os.path.join(out, "rc.npz"))
    def flip(Z):
        Z["row__goal_y"][0] ^= np.uint64(1)                                  # row 0 is a corpus row
    _rewrite(os.path.join(out, "rc.npz"), flip)
    assert any("corpus clip" in x for x in M.verify_derange(U, os.path.join(out, "rc.npz"), T, r))
    # a changed own column
    shutil_copy(os.path.join(out, "r.npz"), os.path.join(out, "ro.npz"))
    _rewrite(os.path.join(out, "ro.npz"), lambda Z: Z["row__nav_token"].__setitem__(-1, (Z["row__nav_token"][-1] + 1) % 3))
    assert any("own column" in x for x in M.verify_derange(U, os.path.join(out, "ro.npz"), T, r))
    # a recorded fixed point
    r["donor"][next(iter(r["donor"]))] = int(next(iter(r["donor"])))
    assert any("fixed point" in x for x in M.verify_derange(U, os.path.join(out, "r.npz"), T, r))


def man_for(src_manifest, npz):
    """A manifest next to ``npz`` (same schema / vocab as ``src_manifest``), naming the new file's md5."""
    m = json.load(open(src_manifest, encoding="utf-8"))
    m["npz_md5"] = M.md5_file(npz)
    json.dump(m, open(os.path.splitext(npz)[0] + ".manifest.json", "w", encoding="utf-8"))


def shutil_copy(a, b):
    import shutil
    shutil.copyfile(a, b)


def test_derange_refuses_singleton_groups(merged):
    w, out, rec = merged
    make_release(os.path.join(out, "t1.npz"), ["tenk-solo"], 9)                       # one clip of (20, 0) alone... plus a (24, 0) group of one
    names = NAMES["tenk"][:4] + ["tenk-solo"]
    make_release(os.path.join(out, "t2.npz"), names, 9)
    M.merge_v9(os.path.join(w, "v9_corpus.npz"), os.path.join(out, "t2.npz"), [], os.path.join(out, "u2.npz"))
    # shapes alternate (20,24,20,24,20): all groups >= 2 -> fine; build a lone-shape group by hand
    Z = M.load_npz(os.path.join(out, "t2.npz"))
    with pytest.raises(M.MergeError, match="groups of one"):
        make_release(os.path.join(out, "t3.npz"), ["tenk-a", "tenk-b", "tenk-c"], 5)    # shapes 20, 24, 20: the (24, 0) group has one clip
        M.merge_v9(os.path.join(w, "v9_corpus.npz"), os.path.join(out, "t3.npz"), [], os.path.join(out, "u3.npz"))
        M.derange_v9(os.path.join(out, "u3.npz"), os.path.join(out, "t3.npz"), os.path.join(out, "r3.npz"))
    r = M.derange_v9(os.path.join(out, "u3.npz"), os.path.join(out, "t3.npz"), os.path.join(out, "r3.npz"), max_singletons=1)
    assert r["n_singletons_left"] == 1 and r["n_deranged_clips"] == 2


# ------------------------------------------------------------------------------------------------ argv diff
M2B = ["--arm", "M2b", "--seed", "0", "--grid", "pool4x10", "--out", "/r/M2b_s0", "--cache-dir", "/r/cache_pod", "--v9-train", "/r/v9/v9_labels_train.npz", "--v9-train-md5", "83e30bd3791f0767707eade5f4671759",
       "--lead-train", "/r/v9/lead_train4572.npz", "--speed-train", "/r/v9/speed.jsonl", "--join-train", "/r/join_train.npz", "--p10", "/r/p10_prior.pt", "--kin-mode", "yaw", "--tracks", "/r/native_tracks.pt",
       "--exclude", "540:600", "--batch", "64", "--steps-mult", "4", "--device", "cuda"]
STAGE_B = M2B + ["--init-ckpt", "/r/B_s0/ckpt_last.pt", "--steps", "3000", "--lr", "1e-4", "--dagger-p", "0.1", "--dagger-bank", "/r/bank_B_s0", "--dagger-bank-sha256", "ab", "--dagger-seed", "0"]


def setflag(toks, flag, val=None, drop=False):
    t = list(toks)
    if flag in t:
        i = t.index(flag)
        j = i + 1
        while j < len(t) and not t[j].startswith("--"):
            j += 1
        t[i:j] = [] if drop else ([flag] + ([val] if val is not None else []))
    elif not drop:
        t += [flag] + ([val] if val is not None else [])
    return t


def derive(base, arm, ue=None, new_md5="U" * 0 + "a" * 32):
    spec = C.ARMS[arm]
    t = setflag(base, "--out", "/r/" + arm)
    t = setflag(t, "--seed", str(spec["seed"]))
    if spec["set"] in ("U", "R", "X"):
        t = setflag(t, "--exclude", drop=True)
        t = setflag(t, "--cache-dir", "/r/cache_union")
        t = setflag(t, "--v9-train", "/r/union/v9.npz")
        t = setflag(t, "--v9-train-md5", new_md5)
        t = setflag(t, "--lead-train", "/r/union/lead.npz")
        t = setflag(t, "--speed-train", "/r/union/speed.jsonl")
        t = setflag(t, "--tracks", "/r/union/tracks.pt")
        t = setflag(t, "--join-train", "/r/union/join.npz")
        t = setflag(t, "--fit-join")
    if spec["steps"] == "fixed":
        t = setflag(t, "--steps", str(C.RUN_STEPS_A))
    if spec["steps"] == "ue":
        t = setflag(t, "--steps", str(ue))
    return t


@pytest.mark.parametrize("arm", ["B_s0", "U_s0", "B_rep", "R_s0", "B_s1", "U_s1", "UE_s0", "BL_s0", "X_s0", "B_s3", "U_s4"])
def test_argv_diff_passes_for_every_arm(arm):
    assert C.argv_diff_problems(M2B, derive(M2B, arm, ue=61600), arm, "A", 61600) == []


@pytest.mark.parametrize("arm", ["B_s0", "U_s0", "B_rep", "U_s1", "X_s0"])
def test_argv_diff_stage_b_passes(arm):
    t = derive(STAGE_B, arm)
    t = setflag(t, "--steps", "3000")                                                  # stage B has its own --steps 3000 on every arm
    t = setflag(t, "--dagger-seed", str(C.ARMS[arm]["dagger_seed"]))
    t = setflag(t, "--init-ckpt", f"/r/{arm}/ckpt_last.pt")
    assert C.argv_diff_problems(STAGE_B, t, arm, "B") == []


def test_red_argv_extra_difference_is_refused():
    u = derive(M2B, "U_s0")
    assert C.argv_diff_problems(M2B, setflag(u, "--batch", "32"), "U_s0")                        # a changed lever
    assert C.argv_diff_problems(M2B, setflag(u, "--w-acc", "3.0"), "U_s0")                       # an added lever
    assert C.argv_diff_problems(M2B, setflag(u, "--p10", "/r/other.pt"), "U_s0")                 # the shared P10
    assert C.argv_diff_problems(M2B, setflag(u, "--kin-mode", drop=True), "U_s0")                # a removed flag
    assert C.argv_diff_problems(M2B, setflag(derive(M2B, "B_s0"), "--grid", "x"), "B_s0")


def test_red_exclude_on_a_union_argv_g_r3():
    u = setflag(derive(M2B, "U_s0"), "--exclude", "540:600")
    p = C.argv_diff_problems(M2B, u, "U_s0")
    assert any("G-R3" in x for x in p)
    b = setflag(derive(M2B, "B_s0"), "--exclude", drop=True)
    assert any("C-EXCL" in x for x in C.argv_diff_problems(M2B, b, "B_s0"))
    assert any("C-EXCL" in x for x in C.argv_diff_problems(M2B, setflag(derive(M2B, "B_s0"), "--exclude", "540:601"), "B_s0"))


def test_red_steps_seeds_and_joins():
    assert C.argv_diff_problems(M2B, setflag(derive(M2B, "U_s0"), "--steps", "19163"), "U_s0")
    assert C.argv_diff_problems(M2B, setflag(derive(M2B, "B_rep"), "--seed", "0"), "B_rep")
    assert C.argv_diff_problems(M2B, setflag(derive(M2B, "B_s1"), "--seed", "2"), "B_s1")
    assert C.argv_diff_problems(M2B, setflag(derive(M2B, "U_s0"), "--fit-join", drop=True), "U_s0")
    assert C.argv_diff_problems(M2B, setflag(derive(M2B, "U_s0"), "--join-train", "/r/join_train.npz"), "U_s0")      # the corpus join path
    assert C.argv_diff_problems(M2B, setflag(derive(M2B, "B_s0"), "--steps", "19164"), "B_s0")                      # --steps on a matched-epochs arm
    assert C.argv_diff_problems(M2B, derive(M2B, "UE_s0", ue=61600), "UE_s0", "A", 61601)
    assert C.argv_diff_problems(M2B, derive(M2B, "UE_s0", ue=61600), "UE_s0", "A", None)
    assert C.argv_diff_problems(M2B, derive(M2B, "B_s0"), "no_such_arm")
    assert C.argv_diff_problems(M2B, ["--seed", "0", "--seed", "1"], "B_s0")
    t = setflag(derive(STAGE_B, "B_s1"), "--dagger-seed", "0")
    assert C.argv_diff_problems(STAGE_B, t, "B_s1", "B")


# ------------------------------------------------------------------------------------------------ md5 set
def test_md5_set_and_regressions(tmp_path):
    f = tmp_path / "v9.npz"
    f.write_bytes(b"union-bytes")
    md5 = M.md5_file(str(f))
    acc = C.accepted_md5_set(True, md5)
    u = setflag(setflag(derive(M2B, "U_s0"), "--v9-train", str(f)), "--v9-train-md5", md5)
    assert C.md5_problems("U_s0", u, acc) == []
    assert C.md5_problems("U_s0", setflag(u, "--v9-train-md5", "e" * 32), acc)                                # an unknown md5 is refused
    assert C.md5_problems("U_s0", u, C.accepted_md5_set(False, md5))                                           # the union md5 without a passing C-MERGE is refused
    assert C.md5_problems("U_s0", setflag(u, "--v9-train-md5", M.CORPUS_V9_MD5), acc)                          # the corpus release on a union arm
    f.write_bytes(b"edited-bytes")
    assert C.md5_problems("U_s0", u, acc)                                                                      # the file no longer has the declared md5
    assert C.md5_problems("R_s0", setflag(u, "--v9-train-md5", md5), acc)                                      # a union release where the R release is required
    assert set(C.accepted_md5_set()) == {M.CORPUS_V9_MD5, M.TENK_V9_MD5}


# ------------------------------------------------------------------------------------------------ counts
def test_steps_nclips_chunk_trainsha():
    rec = {"S": 4791, "S_U": 15450, "n_clips": 4512, "lru": 50}
    assert C.steps_problems("B_s0", M2B, rec) == []                                            # 4791 x 4 = 19,164
    assert C.steps_problems("U_s0", derive(M2B, "U_s0"), rec) == []
    assert C.steps_problems("U_s0", setflag(derive(M2B, "U_s0"), "--steps", "19163"), rec)     # the C-STEPS regression arm
    assert C.steps_problems("UE_s0", derive(M2B, "UE_s0", ue=4 * 15450), rec, 4 * 15450) == []
    assert C.steps_problems("UE_s0", derive(M2B, "UE_s0", ue=4 * 15450 + 1), rec)
    # MM ruling 2026-11-17 (block 3): BL_s0 = UE_s0's total, read from the UNION record (U_s0's dry record), never from BL's own B record
    b_rec = {"S": 4791, "S_U": 4791, "n_clips": 4512, "lru": 50}                                 # what dry_build writes for a B-set arm
    u_rec = {"S": 15392, "S_U": 15392, "n_clips": 14512, "lru": 50}                              # U_s0's dry record
    ue_t = 4 * 15392                                                                             # 61,568
    assert C.steps_problems("UE_s0", derive(M2B, "UE_s0", ue=ue_t), u_rec, ue_t) == []
    assert C.steps_problems("BL_s0", derive(M2B, "BL_s0", ue=ue_t), b_rec, ue_t, "A", u_rec) == []   # the card's BL: the same total as UE
    assert C.steps_problems("BL_s0", derive(M2B, "BL_s0", ue=19164), b_rec, ue_t, "A", u_rec)        # regression: BL at 19,164 reads REFUSED
    assert C.steps_problems("BL_s0", derive(M2B, "BL_s0", ue=ue_t + 1), b_rec, ue_t, "A", u_rec)     # regression: BL at 61,569
    assert C.steps_problems("UE_s0", derive(M2B, "UE_s0", ue=ue_t + 1), u_rec, ue_t)                 # regression: UE at 61,569
    assert C.steps_problems("BL_s0", derive(M2B, "BL_s0", ue=ue_t), b_rec, ue_t)                     # regression: no union record = a target from the B record, REFUSED
    assert C.steps_problems("BL_s0", derive(M2B, "BL_s0", ue=4 * 4791), b_rec, 4 * 4791, "A", b_rec)  # regression: the B record passed as the union record, target 19,164 with ue_steps 19,164: still a 19,164 BL
    assert C.steps_problems("BL_s0", derive(M2B, "BL_s0", ue=ue_t), b_rec, ue_t + 4, "A", u_rec)     # evidence ue_steps disagrees with 4 x S_U
    assert C.steps_problems("B_s0", STAGE_B, rec, stage="B") == [] and C.steps_problems("B_s0", M2B, rec, stage="B")
    assert C.nclips_problems("B_s0", rec) == [] and C.nclips_problems("U_s0", rec) and C.nclips_problems("X_s0", {"n_clips": 10000}) == []
    assert C.chunk_problems(rec) == [] and C.chunk_problems({"lru": 40}) and C.chunk_problems({"lru": 305}) == [] and C.chunk_problems({})
    cl = [f"clip-{i}" for i in range(20)]
    good = {"train_clips": cl, "n_clips": 20, "train_sha256": M.train_sha256(cl)}
    assert C.train_sha_problems("U_s0", good) == []
    assert C.train_sha_problems("U_s0", {**good, "train_sha256": "0" * 64}) and C.train_sha_problems("U_s0", {**good, "n_clips": 21}) and C.train_sha_problems("U_s0", {"n_clips": 1})
    assert C.train_sha_problems("B_s0", good)                                                   # a B arm must carry the M2b value


def test_disjoint_and_holdout_regressions():
    tr, ev = [f"t{i}" for i in range(50)], [f"e{i}" for i in range(10)]
    assert C.disjoint_problems(tr, ev) == []
    assert any("G-R1" in x for x in C.disjoint_problems(tr + ["e3"], ev))                      # one eval-141 id in a train manifest
    assert C.disjoint_problems(tr, ev, eval_sha12=[M.sha12("t7")])                             # a digest-list hit
    assert C.disjoint_problems(tr, [])                                                          # an empty eval list reads nothing
    assert C.holdout_problems(tr, ["h1", "h2"]) == [] and C.holdout_problems(tr + ["h1"], ["h1", "h2"]) and C.holdout_problems(tr, [])
    assert C.sess_report(tr, ev)["computable"] is False
    s = {**{c: "d1" for c in tr[:3]}, **{c: "d1" for c in ev[:1]}}
    assert C.sess_report(tr, ev, s)["n_shared"] == 3


def test_load_id_list(tmp_path):
    (tmp_path / "a.txt").write_text("x\ny\n\n")
    (tmp_path / "b.json").write_text(json.dumps({"k": ["p", "q"], "n": 3}))
    assert C.load_id_list(str(tmp_path / "a.txt")) == ["x", "y"] and C.load_id_list(str(tmp_path / "b.json")) == ["p", "q"]


# ------------------------------------------------------------------------------------------------ C-MASK
def test_mask_synthetic_and_regression():
    m = C.vlm_mask_from_manifest({"vocab": {"GOAL_PROV": PROV}})
    me = C.vlm_mask_from_manifest({"vocab": {"GOAL_PROV": PROV}}, exact=True)
    assert m & TL_MASK == TL_MASK and (m & GEOM_MASK) == 0 and me != m
    gw = np.zeros(10, np.uint64)
    gw[:5] = GEOM_MASK
    gw[:2] |= np.uint64(TL_MASK)                                                              # corpus rows 0,1 carry VLM bits
    is_t = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], bool)
    assert C.mask_problems(gw, is_t, m) == []
    g2 = gw.copy()
    g2[7] |= np.uint64(1 << C.GOAL22.index("TRAFFIC_LIGHT_REACT_RED"))                        # G-R2: one VLM bit set on a 10k row
    p = C.mask_problems(g2, is_t, m)
    assert p and "G-R2" in p[0] and "TRAFFIC_LIGHT_REACT_RED" in p[0]
    assert C.mask_problems(np.where(is_t, gw, np.uint64(0)), is_t, m)                         # the corpus control reads 0: the probe reads nothing


def test_mask_on_the_merged_release(merged):
    w, out, rec = merged
    U, T = os.path.join(out, "v9_labels_union.npz"), os.path.join(w, "v9_tenk.npz")
    assert C.mask_problems_release(U, T) == []
    Z = M.load_npz(U)
    tsid = set(M.load_npz(T)["clip__sid"].tolist())
    j = next(i for i, s in enumerate(Z["clip__sid"].tolist()) if s in tsid)
    Z["row__goal_w"][int(Z["clip__row0"][j])] |= np.uint64(TL_MASK)
    M.write_npz_deterministic(os.path.join(out, "bad.npz"), Z)
    shutil_copy(os.path.splitext(U)[0] + ".manifest.json", os.path.join(out, "bad.manifest.json"))
    assert C.mask_problems_release(os.path.join(out, "bad.npz"), T)


# ------------------------------------------------------------------------------------------------ C-JOIN-EQ
def _join(world_out, npz, names):
    rel = V9.load_v9_release(npz)
    return D.build_label_join([(c, _shape(i)[0] // 2) for i, c in enumerate(names)], rel)


@needs_record
def test_join_eq_passes_and_perturbation_goes_red(merged):
    w, out, rec = merged
    ids_ = M.holdout_ids(os.path.join(w, "manifest_corpus.jsonl"), *HOLD)
    corpus_kept = [c for c in NAMES["corpus"] if c not in ids_]
    rc = V9.load_v9_release(os.path.join(w, "v9_corpus.npz"))
    ru = V9.load_v9_release(os.path.join(out, "v9_labels_union.npz"))
    nf = lambda c: _shape(NAMES["corpus"].index(c) if c in NAMES["corpus"] else NAMES["tenk"].index(c))[0] // 2
    Jc = D.build_label_join([(c, nf(c)) for c in corpus_kept], rc)
    Ju = D.build_label_join([(c, nf(c)) for c in corpus_kept + NAMES["tenk"]], ru)
    assert C.join_eq_problems(Ju, Jc) == []
    # a union release with one perturbed corpus nav row
    Z = M.load_npz(os.path.join(out, "v9_labels_union.npz"))
    Z["row__nav_token"][4] = (Z["row__nav_token"][4] + 1) % 3
    M.write_npz_deterministic(os.path.join(out, "pert.npz"), Z)
    man_for(os.path.join(out, "v9_labels_union.manifest.json"), os.path.join(out, "pert.npz"))
    Jp = D.build_label_join([(c, nf(c)) for c in corpus_kept + NAMES["tenk"]], V9.load_v9_release(os.path.join(out, "pert.npz")))
    p = C.join_eq_problems(Jp, Jc)
    assert p and any("nav" in x for x in p)
    assert C.join_eq_problems({**Ju, "clip": Ju["clip"][1:], "off": Ju["off"][1:], "n": Ju["n"][1:]}, Jc)         # a corpus clip missing from the union join
    rep = C.coverage_report(Ju, NAMES["tenk"])
    assert rep["corpus"]["n_clips"] == len(corpus_kept) and rep["tenk"]["n_clips"] == 30 and 0.0 < rep["tenk"]["nav_has_row_frac"] <= 1.0


# ------------------------------------------------------------------------------------------------ G-LIVE reach table, G-CLOCK, G-HYG, C-INIT, C-CACHE
def _reach(stage_b=False):
    terms = C.STAGE_A_TERMS + (C.STAGE_B_TERMS if stage_b else ())
    r = {k: {t: {"finite": True, "value": 1.0, "grad": 0.5} for t in terms} for k in ("corpus", "tenk", "mixed")}
    r["tenk"]["goal_vlm"] = {"finite": True, "value": 0.0, "grad": 0.0}
    r["tenk"]["cons_lc"] = {"finite": True, "value": 0.0, "grad": 0.0}
    r["control"] = {"goal_vlm_grad": 0.3}
    return r


def test_reach_table_and_regressions():
    assert C.reach_problems(_reach()) == [] and C.reach_problems(_reach(True), True) == []
    r = _reach(); r["tenk"]["goal_vlm"]["grad"] = 1e-9; assert C.reach_problems(r)                         # the VLM logits must read EXACTLY 0.0 on a 10k batch
    r = _reach(); r["tenk"]["cons_lc"]["value"] = 0.1; assert C.reach_problems(r)
    r = _reach(); r["tenk"]["goal_geom"]["grad"] = 0.0; assert C.reach_problems(r)                         # goal_w forced to 0: the goal term reads dead -> FAIL
    r = _reach(); r["mixed"]["traj"]["grad"] = 0.0; assert C.reach_problems(r)
    r = _reach(); r["corpus"]["wm"]["finite"] = False; assert C.reach_problems(r)
    r = _reach(); r["control"]["goal_vlm_grad"] = 0.0; assert C.reach_problems(r)                          # the zero control
    r = _reach(); del r["mixed"]; assert C.reach_problems(r)
    assert C.reach_problems(_reach(), True)                                                                # stage B terms absent
    r = _reach(); del r["tenk"]["pen"]; assert C.reach_problems(r)


def test_clock_hyg_init_cache():
    rng = np.random.default_rng(0)
    corpus, tenk = rng.normal(0, 0.01, 2000), rng.normal(0, 0.01, 2000)
    assert C.clock_problems(tenk, corpus) == [] and C.clock_problems(tenk + 0.2, corpus) and C.clock_problems([], corpus)
    base = {"n_trainable": C.BASE_PARAMS, "cfg": {"a": 1}}
    assert C.hyg_problems(dict(base), base) == [] and C.hyg_problems({**base, "n_trainable": C.BASE_PARAMS + 1}, base) and C.hyg_problems({**base, "cfg": {"a": 2}}, base)
    import torch
    sd = {"w": torch.arange(4.0), "b": torch.ones(2)}
    assert C.state_dict_sha256(sd) == C.state_dict_sha256({"b": torch.ones(2), "w": torch.arange(4.0)})
    assert C.state_dict_sha256(sd) != C.state_dict_sha256({"w": torch.arange(4.0) + 1, "b": torch.ones(2)})
    assert C.cache_problems({"a": "1", "b": "2"}, {"a": "1", "b": "2"}) == [] and C.cache_problems({"a": "1"}, {"a": "9"}) and C.cache_problems({"a": "1"}, {})


# ------------------------------------------------------------------------------------------------ the trainer's own build_data (real code of record, synthetic world)
@needs_record
def test_dry_build_against_the_trainer(tmp_path):
    pytest.importorskip("torch")
    root = os.path.join(HERE, "tree")
    if not os.path.isfile(os.path.join(root, "stack", "scripts", "v8f_mini_train.py")) or not os.path.isdir(os.path.join(root, "taniteval")):
        pytest.skip("trainer export not present")
    toks = ["--arm", "M2b", "--synthetic", "--out", str(tmp_path / "o"), "--batch", "4", "--steps-mult", "4", "--kin-mode", "yaw", "--exclude", "1:2"]
    rec = C.dry_build(toks, root)
    assert rec["n_clips"] == 5 and rec["n_excluded"] == 1 and rec["excluded_in_train"] == 0
    assert rec["train_sha256"] == M.train_sha256(rec["train_clips"]) and C.train_sha_problems("U_s0", {**rec}) == []
    assert rec["sampler_chunk"] == 48 and C.chunk_problems(rec) == [] and rec["total_steps"] == rec["S"] * 4
    assert C.total_steps(toks, rec) == rec["total_steps"]


# ------------------------------------------------------------------------------------------------ the token
def _evidence(stage_b=False):
    ok = {"pass": True}
    rng = np.random.default_rng(1)
    base = {"n_trainable": C.BASE_PARAMS, "cfg": {"a": 1}}
    return {"hyg": {"arm": dict(base), "base": dict(base)}, "reach": _reach(stage_b), "clock": {"tenk": rng.normal(0, .01, 500).tolist(), "corpus": rng.normal(0, .01, 500).tolist()},
            "G-EVAL": ok, "G-CKPT": ok, "G-SUITE": ok, "c_init": {"B_s0": "x", "U_s0": "x"}}


@pytest.fixture
def gate_world(merged, tmp_path, monkeypatch):
    """Everything a U_s0 / stage-A mint needs, in the fixture world."""
    w, out, mrec = merged
    root = tmp_path / "root"
    (root / "stack" / "scripts").mkdir(parents=True)
    (root / "stack" / "scripts" / "v8f_mini_train.py").write_text("# trainer\n")
    (root / "stack" / "tanitad").mkdir()
    (root / "stack" / "tanitad" / "x.py").write_text("x = 1\n")
    (root / "stack" / "tests").mkdir()
    (root / "taniteval").mkdir()
    ids_ = M.holdout_ids(os.path.join(w, "manifest_corpus.jsonl"), *HOLD)
    union_ids = [c for c in NAMES["corpus"] if c not in ids_] + NAMES["tenk"]
    (tmp_path / "eval.txt").write_text("\n".join(f"eval-{i}" for i in range(10)))
    for n in ("p10.pt", "cache"):
        pass
    p10 = tmp_path / "p10.pt"
    p10.write_bytes(b"p10")
    cache = tmp_path / "cache"
    (cache / "train").mkdir(parents=True)
    shutil_copy(os.path.join(out, "manifest.jsonl"), str(cache / "train" / "manifest.jsonl"))
    base = setflag(M2B, "--p10", str(p10))
    u = derive(base, "U_s0", new_md5=mrec["v9"]["md5"])
    for fl, fn in (("--v9-train", "v9_labels_union.npz"), ("--lead-train", "lead_union.npz"), ("--speed-train", "speed_union.jsonl"), ("--tracks", "native_tracks_union.pt")):
        u = setflag(u, fl, os.path.join(out, fn))
    u = setflag(u, "--cache-dir", str(cache))
    u = setflag(u, "--join-train", str(tmp_path / "union_join.npz"))
    rec = {"S": 4791, "S_U": 15000, "n_clips": 67 if False else len(union_ids), "lru": 50, "train_clips": sorted(union_ids, key=M.sha12), "train_sha256": M.train_sha256(union_ids)}
    monkeypatch.setattr(C, "NCLIPS", {**C.NCLIPS, "U": len(union_ids)})
    monkeypatch.setattr(M, "B_TRAIN_SHA256", "x")
    json.dump(rec, open(tmp_path / "dry.json", "w"))
    json.dump(_evidence(), open(tmp_path / "ev.json", "w"))
    json.dump(base, open(tmp_path / "base.json", "w"))
    json.dump(u, open(tmp_path / "argv.json", "w"))
    args = dict(arm="U_s0", stage="A", argv=str(tmp_path / "argv.json"), base_argv=str(tmp_path / "base.json"), root=str(root), commit="cdbff971", gate_dir=str(tmp_path / "gate"),
                evidence=str(tmp_path / "ev.json"), dry=str(tmp_path / "dry.json"), merge_record=os.path.join(out, "merge_record.json"), union_release=os.path.join(out, "v9_labels_union.npz"),
                tenk_v9=os.path.join(w, "v9_tenk.npz"), r_release="", corpus_join="", union_join="", eval_ids=str(tmp_path / "eval.txt"), digest_json=None,
                holdout_manifest=os.path.join(w, "manifest_corpus.jsonl"), holdout=HOLD, py=sys.executable, gpu="0")
    return args, u, tmp_path


def test_mint_token_and_refusals(gate_world):
    args, u, tmp = gate_world
    bad = C.mint(NS(**args))
    # the C-JOIN-EQ evidence (two joins) is pod-side and absent here: the token is REFUSED, never minted on a partial read
    assert bad and any("C-JOIN-EQ" in x for x in bad) and not os.path.exists(os.path.join(args["gate_dir"], "gate_token.json"))
    checks = json.load(open(os.path.join(args["gate_dir"], "checks.json")))
    for k in ("C-ARGV", "C-MERGE", "C-MD5", "C-STEPS", "C-NCLIPS", "C-CHUNK", "C-TRAIN-SHA", "C-DISJ", "C-HOLD", "C-MASK", "G-HYG", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE"):
        assert checks[k]["pass"], (k, checks[k])
    assert checks["C-MD5"]["accepted_roles"] == ["corpus", "tenk", "union"]


@needs_record
def test_mint_succeeds_with_the_joins_then_binding_refuses_every_edit(gate_world, monkeypatch):
    args, u, tmp = gate_world
    w = os.path.dirname(args["tenk_v9"])
    out = os.path.dirname(args["union_release"])
    ids_ = M.holdout_ids(os.path.join(w, "manifest_corpus.jsonl"), *HOLD)
    kept = [c for c in NAMES["corpus"] if c not in ids_]
    nf = lambda c: _shape(NAMES["corpus"].index(c) if c in NAMES["corpus"] else NAMES["tenk"].index(c))[0] // 2
    Jc = D.build_label_join([(c, nf(c)) for c in kept], V9.load_v9_release(os.path.join(w, "v9_corpus.npz")))
    Ju = D.build_label_join([(c, nf(c)) for c in kept + NAMES["tenk"]], V9.load_v9_release(args["union_release"]))
    D.save_join(Jc, str(tmp / "join_corpus.npz"))
    D.save_join(Ju, str(tmp / "union_join.npz"))
    monkeypatch.setattr(C, "CORPUS_JOIN_MD5_PREFIX", M.md5_file(str(tmp / "join_corpus.npz"))[:8])
    args.update(corpus_join=str(tmp / "join_corpus.npz"), union_join=str(tmp / "union_join.npz"))
    assert C.mint(NS(**args)) == []
    gate = os.path.join(args["gate_dir"], "gate_token.json")
    extra = {k: v for k, v in {"--union-release": args["union_release"], "tenk_v9": args["tenk_v9"], "merge_record": args["merge_record"], "dry_record": args["dry"], "evidence": args["evidence"]}.items()}
    tp = lambda toks=u, **kw: C.token_problems(gate, "U_s0", "A", toks, args["root"], args.get("commit", "cdbff971") if "commit" not in kw else kw["commit"], extra)
    assert tp() == []
    # --- each edit must be refused
    assert tp(setflag(u, "--seed", "5"))                                                  # a different argv
    assert tp(commit="deadbeef")                                                           # a different commit
    open(os.path.join(args["root"], "stack", "scripts", "v8f_mini_train.py"), "a").write("# edit\n")
    assert tp()                                                                            # the trainer (and the tree) changed
    open(os.path.join(args["root"], "stack", "scripts", "v8f_mini_train.py"), "w").write("# trainer\n")
    assert tp() == []
    tok = json.load(open(gate))
    tok["checks"]["C-MASK"]["pass"] = False
    json.dump(tok, open(gate, "w"))
    assert any("edited" in x or "not PASS" in x for x in tp())                             # an edited token
    # (a token re-sealed by hand with the same content is the same token: the seal is tamper-EVIDENT, as strat1's; what it binds is below)
    tok["checks"]["C-MASK"]["pass"] = True
    tok["body_sha256"] = C.body_sha(tok)
    json.dump(tok, open(gate, "w"))
    assert tp() == []
    sp = os.path.join(out, "speed_union.jsonl")
    keep = open(sp, "rb").read()
    open(sp, "ab").write(b'{"clip_id": "x"}' + bytes([10]))
    assert tp()                                                                            # a bound data file edited after the mint
    open(sp, "wb").write(keep)
    assert tp() == []
    assert C.token_problems(str(tmp / "no_such_token.json"), "U_s0", "A", u, args["root"], "cdbff971", extra) == ["no gate token"]


def test_runner_refuses_without_a_token(gate_world, tmp_path):
    args, u, tmp = gate_world
    assert C.run_arm(NS(**{k: args[k] for k in ("arm", "stage", "argv", "root", "commit", "gate_dir", "py", "gpu")})) == 2


@needs_record
def test_mint_refuses_regressions(gate_world, monkeypatch):
    args, u, tmp = gate_world
    base = lambda **kw: NS(**{**args, **kw})
    # an eval-141 id in the train manifest (G-R1)
    dry = json.load(open(args["dry"]))
    dry["train_clips"] = dry["train_clips"] + ["eval-3"]; dry["n_clips"] += 1; dry["train_sha256"] = M.train_sha256(dry["train_clips"])
    json.dump(dry, open(tmp / "dry_bad.json", "w"))
    monkeypatch.setattr(C, "NCLIPS", {**C.NCLIPS, "U": dry["n_clips"]})
    bad = C.mint(base(dry=str(tmp / "dry_bad.json")))
    assert any("C-DISJ" in x for x in bad)
    monkeypatch.undo()
    # --exclude on the union argv
    json.dump(setflag(u, "--exclude", "540:600"), open(tmp / "argv_ex.json", "w"))
    assert any("C-ARGV" in x for x in C.mint(base(argv=str(tmp / "argv_ex.json"))))
    # no merge record: the union md5 is not accepted
    bad = C.mint(base(merge_record=""))
    assert any("C-MERGE" in x for x in bad) and any("C-MD5" in x for x in bad)
    # no dry build
    assert any("C-TRAIN-SHA" in x for x in C.mint(base(dry="")))


def test_cli_merge_and_derange(pinned, tmp_path, capsys):
    w = lambda n: os.path.join(pinned, n)
    out = str(tmp_path / "cli")
    rc = M.main(["merge", "--corpus-manifest", w("manifest_corpus.jsonl"), "--tenk-manifest", w("manifest_tenk.jsonl"), "--holdout", "5:8", "--v9", w("v9_corpus.npz"), w("v9_tenk.npz"),
                 "--lead", w("lead_corpus.npz"), w("lead_tenk.npz"), "--speed", w("speed_corpus.jsonl"), w("speed_tenk.jsonl"), "--tracks", w("tracks_corpus.pt"), w("tracks_tenk.pt"), "--out-dir", out])
    assert rc == 0 and json.loads(capsys.readouterr().out)["c_merge_pass"] is True
    rc = M.main(["derange", "--union", os.path.join(out, "v9_labels_union.npz"), "--tenk-v9", w("v9_tenk.npz"), "--out", os.path.join(out, "v9_labels_union_R.npz")])
    assert rc == 0 and os.path.isfile(os.path.join(out, "v9_labels_union_R.derangement.json"))
    assert M.main(["merge", "--corpus-manifest", w("manifest_corpus.jsonl"), "--tenk-manifest", w("manifest_tenk.jsonl"), "--holdout", "5:8", "--v9", w("v9_tenk.npz"), w("v9_corpus.npz"),
                   "--lead", w("lead_corpus.npz"), w("lead_tenk.npz"), "--speed", w("speed_corpus.jsonl"), w("speed_tenk.jsonl"), "--tracks", w("tracks_corpus.pt"), w("tracks_tenk.pt"),
                   "--out-dir", str(tmp_path / "cli2")]) == 2                              # swapped inputs: the pinned md5s refuse
