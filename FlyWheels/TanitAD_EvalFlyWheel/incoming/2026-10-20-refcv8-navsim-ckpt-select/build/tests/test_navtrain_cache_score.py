"""Literal tests for ``navtrain_cache_score.py`` (the navtrain metric-cache + scorer step).

Pure part (any python with numpy): overrides, the command-line guards, the chunker, the asset checks (maps / log pickles), the cache
verifier, the binding builder and its fit to ``select_ckpt_r8.load_arm``, the banked-label rule.
``devkit`` part: the REAL devkit.  (1) the devkit's own Hydra configs are COMPOSED with the driver's overrides and must resolve to the
navtrain split; (2) an end-to-end control on the 20 NAVTEST smoke tokens W3 banked, with the model-free STOP agent -- cache build, scorer,
binding -- whose per-token scores must equal W3's banked ``STOP_smoke20.csv`` exactly; (3) the Windows chunked build + the devkit's own
``save_cache_metadata`` must reproduce the same cache.  NAVTEST and model-free only: no navtrain data and no navtrain score exist here.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys

import pytest

import navtrain_cache_score as S
import select_ckpt_r8 as SEL
import shards as SH
import worlds as W

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.dirname(HERE)
SCRIPT = os.path.join(BUILD, "code", "navtrain_cache_score.py")

NAVSIM_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
W3P = "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest"
SMOKE_TOKENS = f"{W3P}/raw/smoke_tokens.json"
BANKED_STOP = f"{W3P}/raw/STOP_smoke20/STOP_smoke20.csv"
SMOKE_CACHE = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_smoke20"


def hex_tokens(n):
    return [f"{i:016x}" for i in range(n)]


def sample_doc(n=1000, logs=4, status="FIXED"):
    toks = hex_tokens(n)
    tl = {t: f"log_{i % logs}" for i, t in enumerate(toks)}
    return {"status": status, "count": n, "tokens": sorted(toks), "token_log": tl,
            "sha256_sorted_tokens": hashlib.sha256("\n".join(sorted(toks)).encode()).hexdigest()}


def write_sample(tmp_path, **kw):
    doc = sample_doc(**{k: v for k, v in kw.items() if k in ("n", "logs", "status")})
    doc.update({k: v for k, v in kw.items() if k not in ("n", "logs", "status")})
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


def small_sample(tmp_path, sizes):
    toks, tl = [], {}
    for li, n in enumerate(sizes):
        for j in range(n):
            t = f"{li:02d}{j:02d}".ljust(16, "0")
            toks.append(t)
            tl[t] = f"log_{chr(97 + li)}"
    doc = {"status": "FIXED", "count": len(toks), "tokens": sorted(toks), "token_log": tl,
           "sha256_sorted_tokens": hashlib.sha256("\n".join(sorted(toks)).encode()).hexdigest()}
    return doc


# ------------------------------------------------------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------------------------------------------------------
def test_constants_are_the_registered_literals():
    assert S.LABEL_PREFIX == "r8sel_" and S.MIN_TOKENS == 1000
    assert S.MAP_CITIES == ("sg-one-north", "us-ma-boston", "us-nv-las-vegas-strip", "us-pa-pittsburgh-hazelwood")
    assert S.MAP_VERSION == "nuplan-maps-v1.0"
    assert S.SPLIT_DIR == {"navtrain": "trainval", "navtest": "test"}
    assert S.PROFILES["thor"]["exp"] == "/dev/shm/navsim/exp/r8sel_navtrain"      # never W3's w3_navtest_v1 scratch


# ------------------------------------------------------------------------------------------------------------------------------
# the sample
# ------------------------------------------------------------------------------------------------------------------------------
def test_a_fixed_sample_loads_with_its_own_sha(tmp_path):
    s = S.load_sample(write_sample(tmp_path), "navtrain")
    assert s["_sha256_sorted_tokens"] == s["sha256_sorted_tokens"] and len(s["tokens"]) == 1000


def test_an_unfixed_sample_refuses(tmp_path):
    with pytest.raises(SystemExit) as e:
        S.load_sample(write_sample(tmp_path, status="NOT_FIXED__NO_READABLE_NAVTRAIN_TOKEN"), "navtrain")
    assert "not FIXED" in str(e.value)
    with pytest.raises(SystemExit) as e:
        S.load_sample(write_sample(tmp_path, status="UNDERPOWERED"), "navtrain")
    assert "not FIXED" in str(e.value)


def test_fewer_than_1000_tokens_refuses(tmp_path):
    with pytest.raises(SystemExit) as e:
        S.load_sample(write_sample(tmp_path, n=999), "navtrain")
    assert "1000" in str(e.value)


def test_a_sample_whose_hash_does_not_rederive_refuses(tmp_path):
    with pytest.raises(SystemExit) as e:
        S.load_sample(write_sample(tmp_path, sha256_sorted_tokens="0" * 64), "navtrain")
    assert "re-derive" in str(e.value) or "sha256" in str(e.value)


def test_the_navtest_control_is_not_held_to_fixed(tmp_path):
    s = S.load_sample(write_sample(tmp_path, n=20, status="CONTROL_ONLY"), "navtest")
    assert len(s["tokens"]) == 20


# ------------------------------------------------------------------------------------------------------------------------------
# overrides
# ------------------------------------------------------------------------------------------------------------------------------
def test_overrides_name_the_navtrain_split_and_exactly_the_samples_logs_and_tokens(tmp_path):
    doc = small_sample(tmp_path, [2, 1])
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    ov, toks = S.navtrain_split_overrides(str(p))
    assert ov == ["train_test_split=navtrain",
                  "train_test_split.scene_filter.log_names=['log_a','log_b']",
                  "train_test_split.scene_filter.tokens=['0000000000000000','0001000000000000','0100000000000000']"]
    assert toks == ["0000000000000000", "0001000000000000", "0100000000000000"]


def test_the_navtest_control_names_navtest(tmp_path):
    doc = small_sample(tmp_path, [1])
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert S.navtrain_split_overrides(str(p), "navtest")[0][0] == "train_test_split=navtest"


def test_a_run_without_a_sample_file_refuses():
    with pytest.raises(SystemExit) as e:
        S.navtrain_split_overrides(None)
    assert "103,288" in str(e.value)


# ------------------------------------------------------------------------------------------------------------------------------
# command-line guards
# ------------------------------------------------------------------------------------------------------------------------------
def test_windows_refuses_a_2000_token_line_and_accepts_1000(tmp_path):
    big = ["train_test_split=navtrain", "train_test_split.scene_filter.tokens=[" + ",".join(f"'{t}'" for t in hex_tokens(2000)) + "]"]
    with pytest.raises(SystemExit) as e:
        S.check_cmdline(big, "nt")
    assert "32,767" in str(e.value)
    ok = ["train_test_split=navtrain", "train_test_split.scene_filter.tokens=[" + ",".join(f"'{t}'" for t in hex_tokens(1000)) + "]"]
    S.check_cmdline(ok, "nt")
    S.check_cmdline(big, "posix")                                  # ~38 KB in one argument: Linux accepts it


def test_linux_refuses_one_argument_over_131072_bytes():
    huge = ["x=" + "a" * 130000]
    with pytest.raises(SystemExit) as e:
        S.check_cmdline(huge, "posix")
    assert "131,072" in str(e.value)
    S.check_cmdline(["x=" + "a" * 100000], "posix")


# ------------------------------------------------------------------------------------------------------------------------------
# chunking (Windows cache build)
# ------------------------------------------------------------------------------------------------------------------------------
def test_chunks_are_whole_logs_disjoint_and_exhaustive():
    doc = small_sample(None, [5, 3, 4, 6, 2])                      # logs a..e
    ch = S.chunk_logs(doc, 8)
    assert [c["logs"] for c in ch] == [["log_a", "log_b"], ["log_c"], ["log_d", "log_e"]]
    assert [len(c["tokens"]) for c in ch] == [8, 4, 8]
    assert sorted(t for c in ch for t in c["tokens"]) == doc["tokens"]
    for c in ch:
        assert {doc["token_log"][t] for t in c["tokens"]} == set(c["logs"])


def test_a_log_larger_than_the_chunk_gets_a_chunk_to_itself():
    doc = small_sample(None, [10, 2])
    ch = S.chunk_logs(doc, 8)
    assert [c["logs"] for c in ch] == [["log_a"], ["log_b"]] and [len(c["tokens"]) for c in ch] == [10, 2]


def test_a_log_is_never_split_across_chunks():
    doc = small_sample(None, [5, 5, 5])
    ch = S.chunk_logs(doc, 7)
    seen = {}
    for i, c in enumerate(ch):
        for lg in c["logs"]:
            assert lg not in seen, f"{lg} appears in chunk {seen[lg]} and {i}"
            seen[lg] = i
    assert [c["logs"] for c in ch] == [["log_a"], ["log_b"], ["log_c"]]


# ------------------------------------------------------------------------------------------------------------------------------
# assets
# ------------------------------------------------------------------------------------------------------------------------------
def make_maps(root, cities=S.MAP_CITIES, versions=None):
    versions = versions or {"sg-one-north": "9.17.1964", "us-ma-boston": "9.12.1817", "us-nv-las-vegas-strip": "9.15.1915",
                            "us-pa-pittsburgh-hazelwood": "9.17.1937"}
    os.makedirs(root, exist_ok=True)
    json.dump({c: {"version": v, "layers": {}} for c, v in versions.items()}, open(os.path.join(root, "nuplan-maps-v1.0.json"), "w"))
    for c in cities:
        d = os.path.join(root, c, versions[c])
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "map.gpkg"), "wb").write(b"gpkg")
    return str(root)


def test_four_cities_present_is_clean(tmp_path):
    assert S.check_maps(make_maps(tmp_path / "maps")) == []


def test_a_maps_root_with_only_las_vegas_is_not_enough(tmp_path):
    """Thor's measured state: ``navsim_maps/nuplan-maps-v1.0`` holds Las Vegas only.  GPKGMapsDB loads all four at construction."""
    root = make_maps(tmp_path / "maps", cities=("us-nv-las-vegas-strip",))
    probs = S.check_maps_for_sample(root, ["us-nv-las-vegas-strip"])
    joined, r = " ".join(probs).replace("\\", "/"), root.replace("\\", "/")
    assert len(probs) == 3
    for c in ("sg-one-north", "us-ma-boston", "us-pa-pittsburgh-hazelwood"):
        assert f"MAP_GPKG_MISSING:{c}:{r}/{c}/" in joined


def test_a_maps_json_that_lacks_a_city_is_a_problem(tmp_path):
    root = make_maps(tmp_path / "maps")
    meta = json.load(open(os.path.join(root, "nuplan-maps-v1.0.json")))
    del meta["us-ma-boston"]
    json.dump(meta, open(os.path.join(root, "nuplan-maps-v1.0.json"), "w"))
    assert "MAPS_JSON_LACKS:us-ma-boston" in S.check_maps(root)


def test_an_absent_maps_root_is_a_problem(tmp_path):
    (p,) = S.check_maps(str(tmp_path / "nowhere"))
    assert p.startswith("MAPS_JSON_UNREADABLE:")


def make_logs(data, logs, split="trainval", empty=()):
    d = os.path.join(data, "navsim_logs", split)
    os.makedirs(d, exist_ok=True)
    for ln in logs:
        open(os.path.join(d, f"{ln}.pkl"), "wb").write(b"" if ln in empty else b"pkl")


def test_a_missing_or_empty_log_pickle_is_a_problem(tmp_path):
    doc = small_sample(None, [2, 2, 2])
    doc["_sha256_sorted_tokens"] = doc["sha256_sorted_tokens"]
    make_logs(str(tmp_path / "d"), ["log_a", "log_c"])
    assert S.check_logs(str(tmp_path / "d"), doc, "navtrain") == ["LOG_PICKLE_MISSING:log_b"]
    make_logs(str(tmp_path / "e"), ["log_a", "log_b", "log_c"], empty=("log_c",))
    assert S.check_logs(str(tmp_path / "e"), doc, "navtrain") == ["LOG_PICKLE_MISSING:log_c"]
    assert S.check_logs(str(tmp_path / "d"), doc, "navtest") == ["LOG_PICKLE_MISSING:log_a", "LOG_PICKLE_MISSING:log_b",
                                                                 "LOG_PICKLE_MISSING:log_c"]       # the test/ directory is another split


def test_preflight_is_ok_only_when_nothing_is_missing(tmp_path):
    doc = small_sample(None, [2, 2])
    doc["_sha256_sorted_tokens"] = doc["sha256_sorted_tokens"]
    for k in ("py", "wrap", "w3", "tree"):
        (tmp_path / k).mkdir(exist_ok=True)
    prof = {"nv_py": str(tmp_path / "py"), "wrap": str(tmp_path / "wrap"), "w3_code": str(tmp_path / "w3"),
            "v11_tree": str(tmp_path / "tree"), "exp": str(tmp_path / "exp"), "data": str(tmp_path / "data"), "maps": make_maps(tmp_path / "maps")}
    make_logs(prof["data"], ["log_a", "log_b"])
    rep = S.preflight(prof, doc, "navtrain")
    assert rep["ok"] is True and rep["problems"] == [] and rep["n_tokens"] == 4 and rep["n_logs"] == 2
    make_logs(prof["data"], ["log_a"], empty=("log_a",))
    prof["maps"] = make_maps(tmp_path / "maps2", cities=("us-nv-las-vegas-strip",))
    rep = S.preflight(prof, doc, "navtrain")
    assert rep["ok"] is False and "LOG_PICKLE_MISSING:log_a" in rep["problems"] and len(rep["problems"]) == 4


# ------------------------------------------------------------------------------------------------------------------------------
# the cache verifier
# ------------------------------------------------------------------------------------------------------------------------------
def make_cache(root, tokens, csv_tokens=None, win=True, extra_pkl=(), n_csv=1, absent=()):
    os.makedirs(os.path.join(root, "metadata"), exist_ok=True)
    for t in list(tokens) + list(extra_pkl):
        if t in absent:
            continue
        d = os.path.join(root, "log_a", "unknown", t)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "metric_cache.pkl"), "wb").write(b"mc")
    csv_tokens = tokens if csv_tokens is None else csv_tokens
    sep = "\\" if win else "/"
    for k in range(n_csv):
        with open(os.path.join(root, "metadata", f"cache_metadata_node_{k}.csv"), "w", encoding="utf-8") as fh:
            fh.write("file_name\n")
            for t in csv_tokens:
                fh.write(sep.join(["Z:", "elsewhere", "cache", "log_a", "unknown", t, "metric_cache.pkl"]) + "\n")
    return str(root)


@pytest.mark.parametrize("win", [True, False])
def test_a_complete_cache_verifies_whatever_os_built_it(tmp_path, win):
    toks = ["t1", "t2", "t3"]
    rep = S.verify_cache(make_cache(tmp_path / "c", toks, win=win), toks)
    assert rep["ok"] is True and rep["problems"] == [] and rep["n_rows"] == 3 and rep["n_rows_file_exists"] == 3 and rep["n_pickles"] == 3


def test_a_row_missing_from_the_csv_is_reported(tmp_path):
    rep = S.verify_cache(make_cache(tmp_path / "c", ["t1", "t2", "t3"], csv_tokens=["t1", "t2"]), ["t1", "t2", "t3"])
    assert rep["ok"] is False and "ROW_TOKENS_DIFFER:missing=1:unexpected=0" in rep["problems"]


def test_a_swapped_token_with_the_same_count_is_reported(tmp_path):
    """3 rows, 3 pickles, 3 expected -- and the SET is wrong.  A count-only check passes this."""
    rep = S.verify_cache(make_cache(tmp_path / "c", ["t1", "t2", "t9"]), ["t1", "t2", "t3"])
    assert rep["ok"] is False
    assert "ROW_TOKENS_DIFFER:missing=1:unexpected=1" in rep["problems"] and "PICKLE_TOKENS_DIFFER:missing=1:unexpected=1" in rep["problems"]


def test_a_pickle_on_disk_that_the_sample_does_not_name_is_reported(tmp_path):
    rep = S.verify_cache(make_cache(tmp_path / "c", ["t1", "t2"], extra_pkl=["t7"]), ["t1", "t2"])
    assert rep["ok"] is False and "PICKLE_TOKENS_DIFFER:missing=0:unexpected=1" in rep["problems"]


def test_a_row_whose_file_is_absent_is_reported(tmp_path):
    rep = S.verify_cache(make_cache(tmp_path / "c", ["t1", "t2", "t3"], absent=("t2",)), ["t1", "t2", "t3"])
    assert rep["ok"] is False and "ROW_FILE_ABSENT" in rep["problems"]


def test_two_metadata_csvs_are_reported(tmp_path):
    rep = S.verify_cache(make_cache(tmp_path / "c", ["t1"], n_csv=2), ["t1"])
    assert rep["ok"] is False and rep["problems"] == ["METADATA_CSV_COUNT_2"]


def test_a_duplicated_row_is_reported(tmp_path):
    rep = S.verify_cache(make_cache(tmp_path / "c", ["t1", "t2"], csv_tokens=["t1", "t2", "t2"]), ["t1", "t2"])
    assert rep["ok"] is False and "ROW_TOKENS_DUPLICATED" in rep["problems"]


def test_finalize_refuses_unless_the_pickles_are_exactly_the_sample(tmp_path):
    c = make_cache(tmp_path / "c", ["t1", "t2"])
    rep = S.finalize_metadata(c, ["t1", "t2", "t3"])
    assert rep["status"] == "REFUSED_PICKLES_ARE_NOT_THE_SAMPLE" and rep["n_missing"] == 1 and rep["n_unexpected"] == 0
    rep = S.finalize_metadata(c, ["t1"])
    assert rep["status"] == "REFUSED_PICKLES_ARE_NOT_THE_SAMPLE" and rep["n_unexpected"] == 1


# ------------------------------------------------------------------------------------------------------------------------------
# the binding
# ------------------------------------------------------------------------------------------------------------------------------
def world_files(tmp_path, arm_label="r8sel_ref_STOP"):
    w = W.make_world(n_logs=13, per_log=80)
    d = tmp_path / arm_label
    d.mkdir()
    table = w.tables[("ema48852", 0)]
    W.write_csv(str(d / f"{arm_label}.csv"), table)
    counts = {"status": "PASS", "failures": [], "n_expected": len(w.tokens), "csv_valid_rows": len(w.tokens), "C3_missing": 0, "C3_unexpected": 0}
    (d / f"{arm_label}.counts.json").write_text(json.dumps(counts), encoding="utf-8")
    sp = tmp_path / "sample.json"
    sp.write_text(json.dumps(w.sample), encoding="utf-8")
    sample = S.load_sample(str(sp), "navtrain")
    return w, sample, str(sp), str(d / f"{arm_label}.csv"), str(d / f"{arm_label}.counts.json")


def bind(sample, sp, csv_path, counts_path, split="navtrain", seam=None, official="STOP"):
    return S.build_binding(label="r8sel_ref_STOP", split=split, sample=sample, sample_path=sp, seam=seam,
                           seam_sha=(S.sha256_file(seam) if seam else None), official=official, csv_path=csv_path, counts_path=counts_path,
                           cache_dir="/dev/shm/c", cache_ver={"metadata_csv_sha256": "abc", "n_rows": 1040}, prof_name="thor", prov={"x": 1})


def test_a_binding_carries_the_four_things_the_decision_script_requires(tmp_path):
    w, sample, sp, csvp, cp = world_files(tmp_path)
    b = bind(sample, sp, csvp, cp)
    assert b["split"] == "navtrain" and b["control"] is False
    assert b["sample_sha256_sorted_tokens"] == hashlib.sha256("\n".join(sorted(w.tokens)).encode()).hexdigest()
    assert b["csv_sha256"] == hashlib.sha256(open(csvp, "rb").read()).hexdigest()
    assert b["seam_sha256"] is None and b["official_agent"] == "STOP" and b["n_tokens"] == 1040 and b["counts_status"] == "PASS"


def test_a_binding_from_here_is_accepted_by_the_decision_scripts_loader(tmp_path):
    w, sample, sp, csvp, cp = world_files(tmp_path)
    bp = tmp_path / "r8sel_ref_STOP" / "r8sel_ref_STOP.binding.json"
    bp.write_text(json.dumps(bind(sample, sp, csvp, cp)), encoding="utf-8")
    table, info, problems = SEL.load_arm({"csv": csvp, "counts": cp, "binding": str(bp)}, ckpt="STOP", seed=0,
                                         sample_sha=sample["_sha256_sorted_tokens"], n_sample=1040, kind="reference")
    assert problems == [] and len(table) == 1040


def test_a_navtest_control_binding_is_rejected_by_the_decision_script(tmp_path):
    w, sample, sp, csvp, cp = world_files(tmp_path)
    b = bind(sample, sp, csvp, cp, split="navtest")
    assert b["split"] == "navtest" and b["control"] is True
    bp = tmp_path / "r8sel_ref_STOP" / "r8sel_ref_STOP.binding.json"
    bp.write_text(json.dumps(b), encoding="utf-8")
    _, _, problems = SEL.load_arm({"csv": csvp, "counts": cp, "binding": str(bp)}, ckpt="STOP", seed=0,
                                  sample_sha=sample["_sha256_sorted_tokens"], n_sample=1040, kind="reference")
    assert "STOP:BINDING_SPLIT_navtest" in problems


def test_counts_that_are_not_pass_refuse_a_binding(tmp_path):
    w, sample, sp, csvp, cp = world_files(tmp_path)
    c = json.load(open(cp))
    c["status"] = "FAIL"
    json.dump(c, open(cp, "w"))
    with pytest.raises(ValueError) as e:
        bind(sample, sp, csvp, cp)
    assert "counts status" in str(e.value)


def _rewrite_csv(csvp, drop=(), add=()):
    rows = list(csv.reader(open(csvp, encoding="utf-8")))
    head, body = rows[0], rows[1:]
    body = [r for r in body if r[1] not in drop]
    for t in add:
        body.insert(0, ["9", t, "True"] + body[0][3:])
    with open(csvp, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows([head] + body)


def test_a_csv_missing_a_sample_token_refuses_a_binding(tmp_path):
    w, sample, sp, csvp, cp = world_files(tmp_path)
    _rewrite_csv(csvp, drop=(w.tokens[0],))
    with pytest.raises(ValueError) as e:
        bind(sample, sp, csvp, cp)
    assert "missing 1" in str(e.value)


def test_a_csv_with_an_extra_token_refuses_a_binding(tmp_path):
    w, sample, sp, csvp, cp = world_files(tmp_path)
    _rewrite_csv(csvp, add=("zzz_not_in_sample",))
    with pytest.raises(ValueError) as e:
        bind(sample, sp, csvp, cp)
    assert "unexpected 1" in str(e.value)


# ------------------------------------------------------------------------------------------------------------------------------
# a label already bound
# ------------------------------------------------------------------------------------------------------------------------------
def test_a_label_bound_to_different_input_is_never_rescored(tmp_path):
    bp = tmp_path / "b.json"
    bp.write_text(json.dumps({"seam_sha256": "aaa", "official_agent": None, "sample_sha256_sorted_tokens": "sss"}), encoding="utf-8")
    assert S.banked_decision(str(tmp_path / "absent.json"), "aaa", None, "sss") == "RUN"
    assert S.banked_decision(str(bp), "aaa", None, "sss") == "SKIP"
    assert S.banked_decision(str(bp), "bbb", None, "sss") == "REFUSE"          # another seam under the same label
    assert S.banked_decision(str(bp), "aaa", None, "ttt") == "REFUSE"          # another sample
    assert S.banked_decision(str(bp), "aaa", "STOP", "sss") == "REFUSE"        # an official agent under a seam's label


def test_the_operator_can_force_a_rescore(tmp_path, monkeypatch):
    bp = tmp_path / "b.json"
    bp.write_text(json.dumps({"seam_sha256": "aaa", "official_agent": None, "sample_sha256_sorted_tokens": "sss"}), encoding="utf-8")
    monkeypatch.setenv("R8CS_FORCE_RESCORE", "1")
    assert S.banked_decision(str(bp), "bbb", None, "sss") == "RUN"


# ------------------------------------------------------------------------------------------------------------------------------
# CLI refusals that need no devkit
# ------------------------------------------------------------------------------------------------------------------------------
def run_cmd(argv):
    """The command a user reaches through ``main`` -- minus ``main``'s own refusal under a mutation env, so a mutated run fails for the
    right reason."""
    a = S.build_parser().parse_args(argv)
    return {"preflight": S.cmd_preflight, "cache": S.cmd_cache, "score": S.cmd_score}[a.cmd](a)


def test_a_label_without_the_prefix_refuses(tmp_path):
    with pytest.raises(SystemExit) as e:
        run_cmd(["score", "--profile", "devbox", "--sample", write_sample(tmp_path), "--out", str(tmp_path / "o"), "--label", "w3_STOP",
                 "--official", "STOP"])
    assert "r8sel_" in str(e.value)
    with pytest.raises(SystemExit) as e:
        run_cmd(["cache", "--profile", "devbox", "--sample", write_sample(tmp_path), "--out", str(tmp_path / "o"), "--label", "cache_x"])
    assert "r8sel_" in str(e.value)


def test_score_needs_exactly_one_of_seam_and_official(tmp_path):
    with pytest.raises(SystemExit) as e:
        run_cmd(["score", "--profile", "devbox", "--sample", write_sample(tmp_path), "--out", str(tmp_path / "o"), "--label", "r8sel_x"])
    assert "exactly one" in str(e.value)


def test_preflight_on_an_unfixed_sample_refuses_before_touching_a_path(tmp_path):
    with pytest.raises(SystemExit) as e:
        run_cmd(["preflight", "--profile", "thor", "--sample", write_sample(tmp_path, status="NOT_FIXED__X")])
    assert "not FIXED" in str(e.value)


@pytest.mark.e2e
def test_main_refuses_while_a_mutation_env_is_set(tmp_path):
    env = dict(os.environ, R8CS_MUTATION="split_navtest", PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, "-X", "utf8", SCRIPT, "preflight", "--sample", write_sample(tmp_path)], capture_output=True, text=True, env=env)
    assert p.returncode != 0 and "R8CS_MUTATION" in (p.stdout + p.stderr)


# ------------------------------------------------------------------------------------------------------------------------------
# the REAL devkit (navtest control, model-free; no navtrain data and no navtrain score)
# ------------------------------------------------------------------------------------------------------------------------------
NEEDED = [NAVSIM_PY, S.PROFILES["devbox"]["v11_tree"], S.PROFILES["devbox"]["w3_code"], S.PROFILES["devbox"]["maps"], SMOKE_TOKENS,
          BANKED_STOP, SMOKE_CACHE]
devkit = pytest.mark.skipif(any(not os.path.exists(p) for p in NEEDED), reason="NAVSIM venv / devkit / W3 navtest smoke artifacts absent")


def run_cli(args, timeout=1800):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4")
    env.pop("R8CS_MUTATION", None)
    return subprocess.run([NAVSIM_PY, "-X", "utf8", SCRIPT, *args], capture_output=True, text=True, env=env, timeout=timeout)


@pytest.mark.devkit
@pytest.mark.e2e
@devkit
def test_the_devkits_own_configs_compose_to_the_navtrain_split(tmp_path):
    sp = write_sample(tmp_path)                                         # 1,000 synthetic tokens in 4 synthetic logs (composition reads no data)
    p = run_cli(["compose-check", "--profile", "devbox", "--sample", sp, "--data", "/x/openscene", "--exp", "/x/exp", "--maps", "/x/maps"])
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    got = json.loads(p.stdout.split("ZZCOMPOSE ", 1)[1].splitlines()[0])
    want_sha = hashlib.sha256("\n".join(sorted(hex_tokens(1000))).encode()).hexdigest()
    for sub in ("pdm_scoring", "metric_caching"):
        g = got[sub]
        assert g["data_split"] == "trainval" and g["navsim_log_path"] == "/x/openscene/navsim_logs/trainval"
        assert g["n_tokens"] == 1000 and g["tokens_sha256_sorted"] == want_sha and g["logs"] == ["log_0", "log_1", "log_2", "log_3"]
        assert (g["num_history_frames"], g["num_future_frames"], g["has_route"], g["frame_interval"], g["max_scenes"]) == (4, 10, True, 1, None)
    assert got["pdm_scoring"]["sensor_blobs_path"] == "/x/openscene/sensor_blobs/trainval"


def _banked_rows():
    rows = list(csv.DictReader(open(BANKED_STOP, encoding="utf-8")))
    return {r["token"]: r for r in rows if r["token"] != "average"}


def _control_sample(tmp_path):
    d = json.load(open(SMOKE_TOKENS))
    toks = sorted(d["tokens"])
    sp = tmp_path / "smoke20_sample.json"
    sp.write_text(json.dumps({"status": "CONTROL_ONLY", "count": len(toks), "tokens": toks, "token_log": d["token_log"]}), encoding="utf-8")
    return str(sp), toks


@pytest.fixture(scope="module")
def e2e_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("cs_e2e")


@pytest.mark.devkit
@pytest.mark.e2e
@devkit
def test_end_to_end_cache_stop_score_and_binding_equal_w3s_banked_stop_smoke20(e2e_dir):
    sp, toks = _control_sample(e2e_dir)
    exp = str(e2e_dir / "exp").replace("\\", "/")
    common = ["--profile", "devbox", "--split", "navtest", "--sample", sp, "--exp", exp, "--cache-name", "smoke20c", "--out", str(e2e_dir / "out")]
    pf = run_cli(["preflight", "--profile", "devbox", "--split", "navtest", "--sample", sp, "--exp", exp])
    assert pf.returncode == 0 and "ZZPREFLIGHT_OK problems=0 tokens=20 logs=4" in pf.stdout, pf.stdout[-1000:]
    c = run_cli(["cache", *common, "--label", "r8sel_ctl_cache_smoke20"])
    assert c.returncode == 0 and "ZZCACHE_DONE rows=20 expected=20 problems=0" in c.stdout, c.stdout[-2500:]
    s = run_cli(["score", *common, "--label", "r8sel_ctl_STOP_smoke20", "--official", "STOP"])
    assert s.returncode == 0 and "ZZSCORE_DONE label=r8sel_ctl_STOP_smoke20 tokens=20" in s.stdout, s.stdout[-2500:]
    odir = e2e_dir / "out" / "scores" / "r8sel_ctl_STOP_smoke20"
    got = {r["token"]: r for r in csv.DictReader(open(odir / "r8sel_ctl_STOP_smoke20.csv", encoding="utf-8")) if r["token"] != "average"}
    want = _banked_rows()
    assert set(got) == set(want) == set(toks)
    for t in toks:
        for col in ("valid", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
                    "driving_direction_compliance", "score"):
            assert got[t][col] == want[t][col], (t, col, got[t][col], want[t][col])           # bit-for-bit the same CSV cells
    b = json.load(open(odir / "r8sel_ctl_STOP_smoke20.binding.json"))
    assert b["split"] == "navtest" and b["control"] is True and b["official_agent"] == "STOP" and b["n_tokens"] == 20
    assert b["csv_sha256"] == hashlib.sha256(open(odir / "r8sel_ctl_STOP_smoke20.csv", "rb").read()).hexdigest()
    assert b["provenance"]["split"] == "navtest" and b["provenance"]["w3_run_v1_sha256"] == S.sha256_file(f"{S.PROFILES['devbox']['w3_code']}/run_v1.py")
    again = run_cli(["score", *common, "--label", "r8sel_ctl_STOP_smoke20", "--official", "STOP"])
    assert "ZZSCORE_SKIPPED_banked" in again.stdout and again.returncode == 0                  # a banked label is not rescored
    other = run_cli(["score", *common, "--label", "r8sel_ctl_STOP_smoke20", "--official", "CV"])
    assert "ZZSCORE_REFUSED_label_bound_to_a_different_input" in other.stdout and other.returncode == 3


@pytest.mark.devkit
@pytest.mark.e2e
@devkit
def test_a_score_with_no_cache_refuses_before_the_devkit_runs(e2e_dir):
    sp, _ = _control_sample(e2e_dir)
    s = run_cli(["score", "--profile", "devbox", "--split", "navtest", "--sample", sp, "--exp", str(e2e_dir / "exp_empty").replace("\\", "/"),
                 "--cache-name", "none", "--out", str(e2e_dir / "out2"), "--label", "r8sel_ctl_x", "--official", "STOP"])
    assert s.returncode == 2 and "ZZSCORE_REFUSED_preconditions" in s.stdout and "METADATA_CSV_COUNT_0" in s.stdout


@pytest.mark.devkit
@pytest.mark.e2e
@devkit
def test_the_windows_chunked_build_with_the_devkits_own_metadata_writer_reproduces_the_cache(e2e_dir):
    sp, toks = _control_sample(e2e_dir)
    exp = str(e2e_dir / "exp_chunk").replace("\\", "/")
    common = ["--profile", "devbox", "--split", "navtest", "--sample", sp, "--exp", exp, "--cache-name", "smoke20k", "--out", str(e2e_dir / "outk")]
    c = run_cli(["cache", *common, "--label", "r8sel_ctl_cache_chunked", "--chunk-tokens", "5"])
    assert c.returncode == 0 and "ZZCACHE_DONE rows=20 expected=20 problems=0" in c.stdout, c.stdout[-2500:]
    chunks = json.load(open(e2e_dir / "outk" / "cache" / "r8sel_ctl_cache_chunked_chunks" / "chunks.json"))
    assert [(x["n_tokens"], x["n_logs"], x["rc"]) for x in chunks] == [(5, 1, 0)] * 4
    ver = json.load(open(e2e_dir / "outk" / "cache" / "r8sel_ctl_cache_chunked.verify.json"))
    assert ver["ok"] is True and ver["n_rows"] == 20 and ver["n_pickles"] == 20 and ver["n_rows_file_exists"] == 20
    ref = S.verify_cache(SMOKE_CACHE, toks)                                                   # W3's banked smoke20 cache, built in one devkit run
    assert ref["ok"] is True and ref["n_rows"] == 20
    # the same pickles, FIELD for field (everything except file_path, the cache's own location): the chunked build is the single-run build
    cmp_ = subprocess.run([NAVSIM_PY, "-X", "utf8", os.path.join(HERE, "cmp_metric_caches.py"), f"{exp}/metric_cache_smoke20k", SMOKE_CACHE,
                           S.PROFILES["devbox"]["v11_tree"], S.PROFILES["devbox"]["maps"]], capture_output=True, text=True)
    r = json.loads(cmp_.stdout.split("ZZCMP ", 1)[1].splitlines()[0])
    assert r["diff"] == {} and (r["equal"], r["total"]) == (20, 20)
    assert r["fields"] == ["file_path", "trajectory", "ego_state", "observation", "centerline", "route_lane_ids", "drivable_area_map"]
