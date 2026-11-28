"""test_dscale_evalview.py -- tests of dscale_evalview.py (note N3 of card PREREG_V8F_PROOF_ADD45). Standard library and pytest only; needs a file system with links.

The file under test comes from DSCALE_EVALVIEW_FILE when set (the mutation runner points it at a mutated copy), else from this folder.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name, env, default):
    p = Path(os.environ.get(env, str(HERE / default)))
    spec = importlib.util.spec_from_file_location(name, str(p))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


V = _load("dscale_evalview_under_test", "DSCALE_EVALVIEW_FILE", "dscale_evalview.py")
TS = _load("dscale_tokens_same_for_evalview", "DSCALE_TOKENS_SAME_FILE", "dscale_tokens_same.py")

STUDY = ["c%02d" % i for i in range(6)]                                              # the study's order
HERE_ORDER = ["c03", "c00", "c05", "c01", "c04", "c02"]                              # the same ids in another order


def _line(c, tag="pod"):
    return json.dumps({"clip": c, "sha12": "s" + c, "file": "clips/%s.pt" % c, "md5": tag + c, "content_md5": tag + "content" + c, "n_frames": 101})


def make_cache(root, order, tag="pod"):
    ev = root / "eval"
    (ev / "clips").mkdir(parents=True)
    for c in order:
        (ev / "clips" / (c + ".pt")).write_bytes(("tokens of " + c).encode())
    (ev / "manifest.jsonl").write_bytes(("\n".join(_line(c, tag) for c in order) + "\n").encode("utf-8"))
    (ev / "meta.json").write_text("{}", encoding="utf-8")
    (ev / ".tmp").mkdir()
    return root


def make_tsv(path, order):
    path.write_text("".join("%s\tstudycontent%s\n" % (c, c) for c in order), encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def box(tmp_path):
    try:
        os.symlink(str(tmp_path), str(tmp_path / "probe_link"))
    except (OSError, NotImplementedError):
        pytest.skip("this file system makes no links")
    return {"src": make_cache(tmp_path / "src", HERE_ORDER), "tsv": make_tsv(tmp_path / "study.tsv", STUDY), "view": tmp_path / "view", "out": tmp_path / "EVALVIEW.json", "tmp": tmp_path}


def run_view(box, src=None, tsv=None, expect=6):
    rc = V.main(["--eval-cache", str(src or box["src"]), "--study-eval-tsv", str(tsv or box["tsv"]), "--view", str(box["view"]), "--expect-eval", str(expect), "--out", str(box["out"])])
    return rc, json.load(open(box["out"], encoding="utf-8"))


def view_ids(box):
    return [json.loads(l)["clip"] for l in open(box["view"] / "eval" / "manifest.jsonl", encoding="utf-8") if l.strip()]


def test_view_holds_the_source_lines_in_the_study_order(box):
    rc, rec = run_view(box)
    assert rc == 0 and rec["pass"] is True
    assert view_ids(box) == STUDY
    src = {json.loads(l)["clip"]: l for l in (box["src"] / "eval" / "manifest.jsonl").read_bytes().split(b"\n") if l.strip()}
    got = [l for l in (box["view"] / "eval" / "manifest.jsonl").read_bytes().split(b"\n") if l.strip()]
    assert got == [src[c] for c in STUDY]                                            # byte for byte the pod's own lines (its md5 values), only the order is the study's
    assert rec["source_in_study_order"] is False and rec["positions_that_differ"] == 5 and rec["view_in_study_order"] is True and rec["lines_unchanged"] is True
    assert rec["sha256_ids_view"] == rec["sha256_ids_study"] == TS.sha_ids(STUDY) and rec["sha256_ids_source"] == TS.sha_ids(HERE_ORDER)


def test_a_source_in_study_order_gives_the_same_manifest_bytes(box):
    src = make_cache(box["tmp"] / "src2", STUDY)
    rc, rec = run_view(box, src=src)
    assert rc == 0 and rec["source_in_study_order"] is True and rec["positions_that_differ"] == 0
    assert (box["view"] / "eval" / "manifest.jsonl").read_bytes() == (src / "eval" / "manifest.jsonl").read_bytes()


def test_clips_and_other_entries_are_reached_through_links_and_dot_names_are_left_out(box):
    rc, rec = run_view(box)
    assert rc == 0 and rec["links"] == ["clips", "meta.json"]
    ev = box["view"] / "eval"
    assert os.path.islink(ev / "clips") and not os.path.islink(ev / "manifest.jsonl") and not os.path.lexists(ev / ".tmp")
    for line in open(ev / "manifest.jsonl", encoding="utf-8"):
        d = json.loads(line)
        assert (ev / d["file"]).read_bytes() == ("tokens of " + d["clip"]).encode()


def _d2e(box, cache, plant=False):
    (box["tmp"] / "union").mkdir(exist_ok=True)
    (box["tmp"] / "union" / "manifest.jsonl").write_text(_line("c00") + "\n", encoding="utf-8")
    (box["tmp"] / "bset.txt").write_text("c00\n", encoding="utf-8")
    make_tsv(box["tmp"] / "train.tsv", ["c00"])
    argv = ["--eval-cache", str(cache), "--union-cache", str(box["tmp"] / "union"), "--bset-list", str(box["tmp"] / "bset.txt"), "--study-eval-tsv", str(box["tsv"]),
            "--study-train-tsv", str(box["tmp"] / "train.tsv"), "--expect-eval", "6", "--out", str(box["tmp"] / "EVALSET.json")]
    return TS.main(argv + (["--plant-swap"] if plant else []))


def test_control_d2e_fails_on_the_source_passes_on_the_view_and_its_planted_swap_still_fails(box):
    assert _d2e(box, box["src"]) == 2
    assert run_view(box)[0] == 0
    assert _d2e(box, box["view"]) == 0
    assert _d2e(box, box["view"], plant=True) == 2


def test_refuses_another_id_set(box):
    src = make_cache(box["tmp"] / "src3", ["c03", "c00", "c05", "c01", "c04", "x99"])
    rc, rec = run_view(box, src=src)
    assert rc == 2 and rec["pass"] is False and "id sets differ" in " ".join(rec["refused"])
    assert not (box["view"] / "eval" / "manifest.jsonl").exists()


def test_refuses_a_doubled_id_and_a_wrong_count(box):
    src = make_cache(box["tmp"] / "src4", HERE_ORDER)
    with open(src / "eval" / "manifest.jsonl", "ab") as fh:
        fh.write((_line("c03") + "\n").encode())
    rc, rec = run_view(box, src=src)
    assert rc == 2 and "twice" in " ".join(rec["refused"])
    rc, rec = run_view(box, expect=141)
    assert rc == 2 and "counts" in " ".join(rec["refused"])


def test_a_second_call_finds_the_view_equal_and_writes_nothing(box):
    rc, rec = run_view(box)
    assert rc == 0 and rec["manifest_written_by_this_call"] is True
    first = (box["view"] / "eval" / "manifest.jsonl").read_bytes()
    rc, rec = run_view(box)
    assert rc == 0 and rec["pass"] is True and rec["manifest_written_by_this_call"] is False
    assert (box["view"] / "eval" / "manifest.jsonl").read_bytes() == first


def test_refuses_an_existing_view_manifest_that_differs(box):
    assert run_view(box)[0] == 0
    vman = box["view"] / "eval" / "manifest.jsonl"
    lines = [l for l in vman.read_bytes().split(b"\n") if l.strip()]
    lines[0], lines[1] = lines[1], lines[0]
    vman.write_bytes(b"\n".join(lines) + b"\n")
    rc, rec = run_view(box)
    assert rc == 2 and rec["pass"] is False and "differs" in " ".join(rec["refused"])
    assert [json.loads(l)["clip"] for l in vman.read_bytes().split(b"\n") if l.strip()][:2] == ["c01", "c00"]      # the refused view is left as found


def test_refuses_a_view_entry_that_is_not_the_link(box):
    (box["view"] / "eval" / "clips").mkdir(parents=True)                             # a real folder where the link must be
    rc, rec = run_view(box)
    assert rc == 2 and "not the link" in " ".join(rec["refused"])


def test_refuses_the_source_folder_as_the_view(box):
    rc = V.main(["--eval-cache", str(box["src"]), "--study-eval-tsv", str(box["tsv"]), "--view", str(box["src"]), "--expect-eval", "6", "--out", str(box["out"])])
    assert rc == 2 and "source folder" in " ".join(json.load(open(box["out"], encoding="utf-8"))["refused"])
    assert [json.loads(l)["clip"] for l in open(box["src"] / "eval" / "manifest.jsonl", encoding="utf-8") if l.strip()] == HERE_ORDER      # the source is not touched


def test_missing_input_is_exit_1(box):
    rc, rec = run_view(box, tsv=box["tmp"] / "no_such.tsv")
    assert rc == 1 and rec["pass"] is False and rec["missing"]
