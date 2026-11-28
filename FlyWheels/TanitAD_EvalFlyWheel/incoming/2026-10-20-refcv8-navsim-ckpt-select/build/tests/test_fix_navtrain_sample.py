"""LITERAL tests for ``code/fix_navtrain_sample.py`` + ``navtrain_common.py`` (any python; real-data controls need the D: navtest fixtures).

The registered sample definition (D-R8-CKPT-RULE (1) + AMENDMENT 1): logs sorted by log_name; ALL readable tokens of the first logs;
stop at the FIRST log whose cumulative count reaches >= target; < min -> UNDERPOWERED; readable = metadata window AND 4 history frames x
{CAM_L0, CAM_F0, CAM_R0}.  Synthetic world (``shards.py``): 6 logs of 20 frames, filter tokens at positions 3..9 (7 per log); the
``current`` archive holds frames 3..9 and the ``history`` archive frames 0..8; ``e_log`` has no metadata pickle.  Target 15, min 10:

    sorted: a, b, c, d, e, f.   a -> 7, b -> 14 (< 15, continue), c -> 21 (>= 15, STOP).   tokens 03a..09a, 03b..09b, 03c..09c  (token = "<position><log letter>", so sorted order interleaves the logs).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle

import pytest

import fix_navtrain_sample as F
import navtrain_common as N
import shards as SH

A = [f"{i:02d}a" for i in range(3, 10)]          # tokens are "<position><log letter>": 03a .. 09a
B = [f"{i:02d}b" for i in range(3, 10)]
C = [f"{i:02d}c" for i in range(3, 10)]
D = [f"{i:02d}d" for i in range(3, 10)]


def run_fix(w, *, target=15, min_tokens=10, sources=None):
    flt = N.parse_scene_filter(w["yaml"])
    srcs = sources or N.discover_shards(w["shards"])
    avail, _ = F.inventory(srcs)
    return F.fix_sample(flt, avail, F.make_loader(w["meta"]), target=target, min_tokens=min_tokens)


def sha_of(tokens):                                   # an INDEPENDENT derivation: the registered gate-G1 formula, on a literal list
    return hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()


@pytest.fixture()
def world(tmp_path):
    return SH.standard_world(tmp_path)


# -------------------------------------------------------------------------------------------------------------------------------
def test_sorted_logs_all_readable_tokens_stop_at_the_first_log_reaching_the_target(world):
    r = run_fix(world)
    assert r["status"] == "FIXED" and r["count"] == 21 and r["target_reached"] is True
    assert r["log_names"] == ["a_log", "b_log", "c_log"]                       # the filter lists b, a, d, c, e, f
    assert r["tokens"] == sorted(A + B + C)                                    # sorted, and the STOP log is taken whole (21, not 15)
    assert r["n_logs"] == 3


def test_a_log_that_lands_exactly_on_the_target_stops_the_sample(world):
    r = run_fix(world, target=14)                  # a (7) -> b (14 >= 14): STOP at b -- a strict ">" would walk on into c
    assert r["count"] == 14 and r["log_names"] == ["a_log", "b_log"] and r["tokens"] == sorted(A + B)


def test_the_sha256_is_over_the_sorted_token_list(world):
    r = run_fix(world)
    assert r["tokens"] == sorted(r["tokens"]) and r["tokens"][:4] == ["03a", "03b", "03c", "04a"]
    assert r["sha256_sorted_tokens"] == sha_of(A + B + C)
    assert len(r["sha256_sorted_tokens"]) == 64


def test_token_log_and_per_log_accounting(world):
    r = run_fix(world)
    assert r["token_log"]["05b"] == "b_log" and r["token_log"]["09c"] == "c_log" and len(r["token_log"]) == 21
    assert [p["cumulative"] for p in r["per_log"]] == [7, 14, 21]
    assert r["per_log"][0] == {"log": "a_log", "rank": 0, "status": "READ", "n_navtrain_tokens": 7, "n_readable": 7,
                               "n_unreadable_no_window": 0, "n_unreadable_frames_missing": 0, "cumulative": 7}
    assert r["map_locations"] == {"us-nv-las-vegas-strip": 21}


def test_a_log_without_metadata_is_skipped_and_recorded(tmp_path):
    w = SH.standard_world(tmp_path, logs=("e_log", "a_log", "b_log"), with_pickle=["a_log", "b_log"])
    r = run_fix(w, target=5, min_tokens=3)
    assert r["per_log"][0]["status"] == "READ" and r["per_log"][0]["log"] == "a_log"
    assert r["log_names"] == ["a_log"] and r["count"] == 7          # a_log alone reaches 5 (7 >= 5) -> stop; e_log is never reached
    r = run_fix(w, target=9, min_tokens=3)                          # a (7) < 9 -> b: 14
    assert r["log_names"] == ["a_log", "b_log"]
    w2 = SH.standard_world(tmp_path / "w2", logs=("a_log", "b_log"), with_pickle=["b_log"])
    r2 = run_fix(w2, target=5, min_tokens=3)
    assert r2["per_log"][0] == {"log": "a_log", "rank": 0, "status": "NO_METADATA"} and r2["tokens"] == sorted(B)


def test_one_missing_history_jpg_of_one_camera_removes_exactly_the_tokens_that_need_it(tmp_path):
    # b_log/CAM_R0/02b is a history frame of tokens 03b, 04b, 05b (positions 3..5) only
    w = SH.standard_world(tmp_path, drop_rels=(SH.frame_name("b_log", 2, "CAM_R0"),))
    r = run_fix(w)
    assert r["tokens"] == sorted(A + ["06b", "07b", "08b", "09b"] + C) and r["count"] == 18
    assert r["per_log"][1]["n_readable"] == 4 and r["per_log"][1]["n_unreadable_frames_missing"] == 3


def test_front_camera_only_is_not_readable(tmp_path):
    """AMENDMENT 1a: 'CAM_F0 history' alone is too narrow -- the bridge stitches L0 + F0 + R0."""
    w = SH.standard_world(tmp_path, cams=("CAM_F0",))
    r = run_fix(w)
    assert r["count"] == 0 and r["status"] == "UNDERPOWERED" and r["tokens"] == [] and r["log_names"] == []


def test_history_frames_are_required_not_just_the_current_frame(tmp_path):
    """Only the `current` archive (frames 3..9): a token needs frames p-3..p, so only positions 6..9 are readable (4 per log)."""
    w = SH.standard_world(tmp_path, history_frames=[])
    r = run_fix(w)
    assert r["tokens"] == sorted(f"{i:02d}{l}" for l in "abcd" for i in range(6, 10)) and r["count"] == 16
    assert r["log_names"] == ["a_log", "b_log", "c_log", "d_log"]


def test_a_window_the_devkit_drops_is_not_readable(tmp_path):
    """a_log frame 7 has no roadblock_ids (has_route drops the token 07a); 12a sits too close to the log end (no 14-frame window) even
    though its jpgs are in the archives."""
    w = SH.standard_world(tmp_path, extra_filter_tokens=("12a",), current_frames=list(range(3, 13)), history_frames=list(range(0, 13)))
    frames = w["frames_by_log"]["a_log"]
    frames[7]["roadblock_ids"] = []
    SH.write_pickles(w["meta"], {"a_log": frames})
    r = run_fix(w, target=1, min_tokens=1)
    assert r["tokens"] == [t for t in A if t != "07a"] and r["count"] == 6
    assert r["per_log"][0]["n_unreadable_no_window"] == 2 and r["per_log"][0]["n_navtrain_tokens"] == 8


def test_fewer_than_min_tokens_is_underpowered(tmp_path):
    w = SH.standard_world(tmp_path, logs=("a_log", "b_log"))
    r = run_fix(w, target=100, min_tokens=15)                        # 14 readable < 15
    assert r["status"] == "UNDERPOWERED" and r["count"] == 14 and r["target_reached"] is False
    r = run_fix(w, target=100, min_tokens=14)                        # exactly min is enough
    assert r["status"] == "FIXED" and r["count"] == 14 and r["target_reached"] is False


def test_target_not_reached_but_above_min_is_fixed_with_the_flag(tmp_path):
    w = SH.standard_world(tmp_path, logs=("a_log", "b_log"))
    r = run_fix(w, target=2000, min_tokens=10)
    assert r["status"] == "FIXED" and r["count"] == 14 and r["target_reached"] is False


def test_the_registered_defaults_are_literals():
    assert N.TARGET_TOKENS == 2000 and N.MIN_TOKENS == 1000
    assert N.CAMS == ("CAM_L0", "CAM_F0", "CAM_R0")
    assert N.NAVTRAIN_FILTER_SHA256 == "c37fea567a0cfdbc29076cca893d4f5dd32db59baec18ae214527206d6b64e6f"
    assert F.MUTATIONS == ("no_sort_logs", "stop_strict", "f0_only", "t0_frame_only", "truncate_to_target", "hash_unsorted",
                           "underpowered_off", "no_overwrite_guard", "keep_unwindowed")


def test_an_extracted_directory_gives_the_same_sample_as_the_archive(tmp_path):
    w = SH.standard_world(tmp_path)
    cur_dir, his_dir = f"{w['tmp']}/dirs/navtrain_current_1", f"{w['tmp']}/dirs/navtrain_history_1"
    SH.write_dir(cur_dir, w["cur_rels"])
    SH.write_dir(his_dir, w["his_rels"])
    r_dir = run_fix(w, sources=[cur_dir, his_dir])
    r_tgz = run_fix(w)
    assert r_dir["tokens"] == r_tgz["tokens"] == sorted(A + B + C)


def test_non_front_cameras_and_directories_in_the_archive_are_ignored(world):
    avail, stats = F.inventory(N.discover_shards(world["shards"]))
    assert all(r.split("/")[1] in ("CAM_L0", "CAM_F0", "CAM_R0") for r in avail)
    assert len(avail) == 6 * 10 * 3                               # 6 logs x frames 0..9 x 3 cameras (CAM_B0 etc. dropped)
    assert [s["name"] for s in stats] == ["navtrain_current_1.tgz", "navtrain_history_1.tgz"]
    assert stats[0]["n_cam_jpgs_L0F0R0"] == 6 * 7 * 3 and stats[1]["n_cam_jpgs_L0F0R0"] == 6 * 9 * 3


# ---- the CLI: availability is fixed ONCE ---------------------------------------------------------------------------------------
def _cli(world, out, extra=()):
    return F.main(["--filter-yaml", world["yaml"], "--filter-sha256", hashlib.sha256(open(world["yaml"], "rb").read()).hexdigest(),
                   "--metadata-dir", world["meta"], "--shards-dir", world["shards"], "--out", out, "--target", "15",
                   "--min-tokens", "10", *extra])


@pytest.mark.e2e
def test_cli_writes_the_sample_the_launch_gate_reads(world, tmp_path, capsys):
    out = str(tmp_path / "NAVTRAIN_SAMPLE_TOKENS.json")
    assert _cli(world, out) == 0
    assert capsys.readouterr().out.startswith(f"ZZSAMPLE status=FIXED count=21 logs=3 sha256={sha_of(A + B + C)}")
    d = json.load(open(out, encoding="utf-8"))
    # the G1 gate of r8ns_navtrain_select_LAUNCH.sh, restated:  status FIXED, count == len(tokens), unique, sha256 re-derives
    assert d["status"] == "FIXED" and d["count"] == len(d["tokens"]) == 21 and len(set(d["tokens"])) == 21
    assert hashlib.sha256("\n".join(sorted(d["tokens"])).encode()).hexdigest() == d["sha256_sorted_tokens"]
    assert d["no_score_read"] is True and d["shard_sha256_verified_against_hf_lfs_oid"] is False
    assert d["unpickler_stubbed_globals"] == [] and d["n_metadata_pickles_read"] == 3


def test_an_existing_fixed_sample_is_never_replaced_with_different_content(world, tmp_path, capsys):
    out = str(tmp_path / "S.json")
    doc = run_fix(world)
    assert F.write_guarded(out, doc) == 0
    first = open(out, encoding="utf-8").read()
    assert F.write_guarded(out, run_fix(world)) == 0                # the same run again: idempotent
    only_current = run_fix(world, sources=[f"{world['shards']}/navtrain_current_1.tgz"])      # less availability -> a DIFFERENT sample
    assert only_current["sha256_sorted_tokens"] != doc["sha256_sorted_tokens"]
    capsys.readouterr()
    assert F.write_guarded(out, only_current) == 2
    assert "ZZSAMPLE_REFUSED existing FIXED sample" in capsys.readouterr().out
    assert open(out, encoding="utf-8").read() == first


@pytest.mark.e2e
def test_cli_second_run_is_idempotent_and_a_changed_availability_is_refused(world, tmp_path, capsys):
    out = str(tmp_path / "S.json")
    assert _cli(world, out) == 0
    first = open(out, encoding="utf-8").read()
    assert _cli(world, out) == 0
    os.remove(f"{world['shards']}/navtrain_history_1.tgz")
    capsys.readouterr()
    assert _cli(world, out) == 2 and "ZZSAMPLE_REFUSED existing FIXED sample" in capsys.readouterr().out
    assert open(out, encoding="utf-8").read() == first


@pytest.mark.e2e
def test_a_not_fixed_placeholder_is_overwritten(world, tmp_path):
    out = str(tmp_path / "S.json")
    json.dump({"status": "NOT_FIXED__NO_READABLE_NAVTRAIN_TOKEN__ACQUISITION_NEEDS_PI_APPROVAL", "count": 0, "tokens": []},
              open(out, "w", encoding="utf-8"))
    assert _cli(world, out) == 0
    assert json.load(open(out, encoding="utf-8"))["status"] == "FIXED"


@pytest.mark.e2e
def test_cli_refuses_a_filter_that_is_not_the_registered_file(world, tmp_path, capsys):
    out = str(tmp_path / "S.json")
    rc = F.main(["--filter-yaml", world["yaml"], "--metadata-dir", world["meta"], "--shards-dir", world["shards"], "--out", out])
    assert rc == 2 and "ZZSAMPLE_REFUSED filter sha256" in capsys.readouterr().out and not os.path.exists(out)


@pytest.mark.e2e
def test_cli_refuses_a_shard_whose_sha256_is_not_the_hf_oid(world, tmp_path, capsys):
    tree = [{"type": "file", "path": "navsim/navtrain_current_1.tgz", "lfs": {"oid": "0" * 64}},
            {"type": "file", "path": "navsim/navtrain_history_1.tgz",
             "lfs": {"oid": hashlib.sha256(open(f"{world['shards']}/navtrain_history_1.tgz", "rb").read()).hexdigest()}}]
    tj = str(tmp_path / "tree.json")
    json.dump(tree, open(tj, "w"))
    out = str(tmp_path / "S.json")
    rc = _cli(world, out, ("--verify-sha256-from", tj))
    assert rc == 2 and "ZZSAMPLE_REFUSED shard integrity: navtrain_current_1.tgz" in capsys.readouterr().out and not os.path.exists(out)


@pytest.mark.e2e
def test_cli_accepts_shards_whose_sha256_is_the_hf_oid(world, tmp_path):
    tree = [{"type": "file", "path": f"navsim/{n}", "lfs": {"oid": hashlib.sha256(open(f"{world['shards']}/{n}", "rb").read()).hexdigest()}}
            for n in ("navtrain_current_1.tgz", "navtrain_history_1.tgz")]
    tj = str(tmp_path / "tree.json")
    json.dump(tree, open(tj, "w"))
    out = str(tmp_path / "S.json")
    assert _cli(world, out, ("--verify-sha256-from", tj)) == 0
    assert json.load(open(out, encoding="utf-8"))["shard_sha256_verified_against_hf_lfs_oid"] is True


@pytest.mark.e2e
def test_cli_refuses_while_a_mutation_is_set(world, tmp_path, monkeypatch):
    monkeypatch.setattr(F, "MUTATION", "f0_only")
    with pytest.raises(SystemExit):
        _cli(world, str(tmp_path / "S.json"))


# ---- the pickle reader: third-party data ----------------------------------------------------------------------------------------
def test_an_untrusted_pickle_executes_nothing(tmp_path):
    marker = tmp_path / "PWNED.txt"

    class Evil:
        def __reduce__(self):
            return (os.system, (f'echo owned > "{marker}"',))

    p = tmp_path / "evil.pkl"
    pickle.dump([{"token": "x", "evil": Evil()}], open(p, "wb"))
    frames, stubbed = N.load_log_pickle(str(p))
    assert not marker.exists()
    assert frames[0]["token"] == "x" and len(stubbed) == 1 and stubbed[0].endswith(".system")


def test_numpy_arrays_and_paths_survive_the_tolerant_unpickler(tmp_path):
    import numpy as np
    import pathlib
    p = tmp_path / "ok.pkl"
    pickle.dump([{"token": "t", "a": np.arange(6, dtype=np.float32).reshape(2, 3), "pth": pathlib.PurePosixPath("x/y")}], open(p, "wb"))
    frames, stubbed = N.load_log_pickle(str(p))
    assert frames[0]["a"].tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]] and str(frames[0]["pth"]) == "x/y" and stubbed == []


def test_member_rel_literals():
    assert N.member_rel("navtrain_current_25/2021.06.14.14.25.15_veh-26_03592_03664/CAM_F0/1e79add73a655426.jpg") == \
        "2021.06.14.14.25.15_veh-26_03592_03664/CAM_F0/1e79add73a655426.jpg"
    assert N.member_rel("navtrain_current_25/2021.06.14.14.25.15_veh-26_03592_03664/CAM_B0/13f9efbdd8fc5fc2.jpg") is None   # not stitched
    assert N.member_rel("navtrain_current_25/2021.06.14.14.25.15_veh-26_03592_03664/CAM_F0") is None                        # a directory
    assert N.member_rel("navtrain_current_25") is None
    assert N.member_rel("log/CAM_L0/x.jpg", strip_root=False) == "log/CAM_L0/x.jpg"


# ---- real-data controls (D: navtest fixtures; skipped when absent) -------------------------------------------------------------
NAVTEST_LOGS = "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
NAVTEST_EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs_smoke4logs.json.gz"
NAVTEST_YAML = ("D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896/navsim/planning/script/config/common/"
                "train_test_split/scene_filter/navtest.yaml")
needs_real = pytest.mark.skipif(not all(os.path.exists(p) for p in (NAVTEST_LOGS, NAVTEST_EXPORT, NAVTEST_YAML)),
                                reason="D: navtest fixtures absent")


@needs_real
def test_the_window_derivation_reproduces_w3s_export_on_334_real_navtest_tokens():
    """INDEPENDENT of the code under test: W3's exported ``frame_tokens`` came from the devkit's own SceneLoader."""
    import gzip
    doc = json.load(gzip.open(NAVTEST_EXPORT, "rt", encoding="utf-8"))
    flt = N.parse_scene_filter(NAVTEST_YAML)
    navtok = set(flt["tokens"])
    assert doc["n_tokens"] == 334 and flt["header"]["num_history_frames"] == 4 and flt["header"]["has_route"] is True
    n_checked, all_stubs = 0, set()
    for log, n_exp in doc["tokens_per_log"].items():
        frames, stubs = N.load_log_pickle(f"{NAVTEST_LOGS}/{log}.pkl")
        all_stubs |= set(stubs)
        pos = {f["token"]: i for i, f in enumerate(frames)}
        mine = [t for t in (f["token"] for f in frames) if t in navtok]
        valid = [t for t in mine if N.scene_window(frames, pos[t], 4, 10, True) is not None]
        assert len(valid) == n_exp == sum(1 for r in doc["tokens"].values() if r["log_name"] == log)
        for t in valid:
            win = N.scene_window(frames, pos[t], 4, 10, True)
            assert [f["token"] for f in win[:4]] == doc["tokens"][t]["frame_tokens"] and win[3]["token"] == t
            assert N.history_rels(win, 4)["CAM_F0"] == doc["tokens"][t]["cams"]["cam_f0"]
            assert N.history_rels(win, 4)["CAM_L0"] == doc["tokens"][t]["cams"]["cam_l0"]
            assert N.history_rels(win, 4)["CAM_R0"] == doc["tokens"][t]["cams"]["cam_r0"]
            n_checked += 1
    assert n_checked == 334
    assert all_stubs == set()                         # the allow-list is enough for REAL OpenScene pickles: nothing was stubbed


@needs_real
def test_the_parsed_navtest_filter_matches_the_known_counts():
    flt = N.parse_scene_filter(NAVTEST_YAML)
    assert len(set(flt["log_names"])) == 136 and len(set(flt["tokens"])) == 12146
