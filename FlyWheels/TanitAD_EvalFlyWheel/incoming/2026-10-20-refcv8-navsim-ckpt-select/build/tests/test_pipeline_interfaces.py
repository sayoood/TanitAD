"""CHAIN tests: the artifact one stage writes is the artifact the next stage accepts -- on a synthetic OpenScene world (no navtrain data).

Every stage of the navtrain selection was tested alone against fixtures written by the same author, and a fixture that imitates a schema can
share the defect it was meant to catch (CLAUDE.md).  Here the consumers read what the PRODUCERS wrote:

    fix_navtrain_sample.main  ->  NAVTRAIN_SAMPLE_TOKENS.json  ->  { export_navtrain_inputs.load_sample, navtrain_cache_score.load_sample,
                                                                     select_ckpt_r8.load_sample, build_navtrain_frames.build_bank,
                                                                     navtrain_cache_score.navtrain_split_overrides / chunk_logs }
    export_navtrain_inputs.build_record  ->  the inputs doc  ->  build_navtrain_frames.build_bank   (the REAL record builder, no hand-made export)

and the B7 property that makes the hard refusal safe: the tokens the sample fixer EXCLUDES for a missing jpg are exactly the tokens the bank
builder would REFUSE for it.

World: 26 logs (distinct initials, so the synthetic tokens ``<position><initial>`` are unique) of 60 frames; the devkit keeps positions 3..49
(47 tokens per log); target 1,000 -> the sample stops at the 22nd log: 22 x 47 = 1,034 tokens.
"""
from __future__ import annotations

import hashlib
import json
import types

import numpy as np
import pytest

import build_navtrain_frames as B
import export_navtrain_inputs as X
import fix_navtrain_sample as F
import make_score_jobs as J            # noqa: F401  (imported to prove the module loads next to the others)
import navtrain_cache_score as CS
import navtrain_common as N
import select_ckpt_r8 as SEL
import shards as SH

LOGS = tuple(f"{chr(97 + i)}_log" for i in range(26))
POSITIONS = list(range(3, 50))
N_FRAMES = 60


@pytest.fixture(autouse=True)
def small_frames(monkeypatch):
    monkeypatch.setattr(B, "H", 8)
    monkeypatch.setattr(B, "W", 16)


def world(tmp, drop=()):
    return SH.standard_world(tmp, logs=LOGS, n_frames=N_FRAMES, with_pickle=list(LOGS), current_frames=range(3, 50),
                             history_frames=range(0, 49), drop_rels=drop)


def fix(w, out):
    return F.main(["--filter-yaml", w["yaml"], "--filter-sha256", hashlib.sha256(open(w["yaml"], "rb").read()).hexdigest(),
                   "--metadata-dir", w["meta"], "--shards-dir", w["shards"], "--out", out, "--target", "1000", "--min-tokens", "1000"])


def es():
    return types.SimpleNamespace(ego_pose=np.zeros(3, np.float32), ego_velocity=np.zeros(2, np.float32),
                                 ego_acceleration=np.zeros(2, np.float32), driving_command=np.array([0, 1, 0, 0], np.float32))


def export_doc(w, logs, positions=POSITIONS):
    """The inputs doc built by the PRODUCTION record builder (``export_navtrain_inputs.build_record``), one record per devkit-valid token."""
    toks = {}
    for lg in logs:
        frames = w["frames_by_log"][lg]
        for p in positions:
            win = frames[p - 3:p + 11]
            r = X.build_record(win[3]["token"], win, 4, [es()] * 4, np.zeros((8, 3)), (8, 0.5), np.zeros((8, 3)), f"fp_{lg}_{p}")
            toks[win[3]["token"]] = r
    return {"split": "navtrain", "n_tokens": len(toks), "tokens": toks}


@pytest.fixture()
def fixed(tmp_path):
    w = world(tmp_path)
    out = str(tmp_path / "NAVTRAIN_SAMPLE_TOKENS.json")
    assert fix(w, out) == 0
    return w, out, json.load(open(out, encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------------------------------
def test_the_fixer_writes_the_sample_every_consumer_accepts(fixed):
    w, out, s = fixed
    assert s["status"] == "FIXED" and s["count"] == 1034 and s["log_names"] == [f"{chr(97 + i)}_log" for i in range(22)]
    assert X.load_sample(out, "navtrain")["count"] == 1034
    assert CS.load_sample(out, "navtrain")["_sha256_sorted_tokens"] == s["sha256_sorted_tokens"]
    sel_sample, sel_sha, problems = SEL.load_sample(out)
    assert problems == [] and sel_sha == s["sha256_sorted_tokens"] and len(sel_sample["tokens"]) == 1034


def test_the_cache_overrides_name_exactly_the_fixed_sample(fixed):
    w, out, s = fixed
    ov, toks = CS.navtrain_split_overrides(out)
    assert toks == s["tokens"] and ov[0] == "train_test_split=navtrain"
    assert ov[1] == "train_test_split.scene_filter.log_names=[" + ",".join(f"'{x}'" for x in s["log_names"]) + "]"
    chunks = CS.chunk_logs(CS.load_sample(out, "navtrain"), 300)
    assert sorted(t for c in chunks for t in c["tokens"]) == s["tokens"] and all(len(c["tokens"]) <= 329 for c in chunks)


def test_a_sample_the_fixer_calls_readable_is_buildable_from_the_real_record_builder(fixed, tmp_path):
    w, out, s = fixed
    doc = export_doc(w, s["log_names"])
    assert set(doc["tokens"]) >= set(s["tokens"])
    sources = N.discover_shards(w["shards"])
    rep = B.build_bank(doc, s, sources, w["meta"], str(tmp_path / "bank"), str(tmp_path / "scratch"), SH.FakeE2(), progress=lambda *_: None)
    assert rep["n_tokens_built"] == 1034 and rep["n_frames"] == 22 * 50
    idx = json.load(open(tmp_path / "bank" / "index.json", encoding="utf-8"))
    assert len(idx["tokens"]) == 1034 and all(len(v["rows"]) == 4 for v in idx["tokens"].values())


def test_what_the_fixer_excludes_for_a_missing_jpg_is_exactly_what_the_bank_builder_refuses(tmp_path):
    gone = SH.frame_name("a_log", 10, "CAM_L0")                       # a history frame of the tokens at positions 10, 11, 12, 13
    w = world(tmp_path, drop=(gone,))
    out = str(tmp_path / "S.json")
    assert fix(w, out) == 0
    s = json.load(open(out, encoding="utf-8"))
    excluded = {"10a", "11a", "12a", "13a"}
    assert not (excluded & set(s["tokens"]))
    assert s["per_log"][0]["log"] == "a_log" and s["per_log"][0]["n_readable"] == 43               # 47 devkit-valid - 4 unreadable
    assert s["count"] == 43 + 21 * 47 == 1030 and s["n_logs"] == 22                                 # the 22nd log is still the one that reaches 1,000
    # force them back into a sample: the bank builder must refuse, and name exactly those four tokens
    forced = sorted(set(s["tokens"]) | excluded)
    s2 = dict(s, tokens=forced, count=len(forced), sha256_sorted_tokens=hashlib.sha256("\n".join(forced).encode()).hexdigest())
    doc = export_doc(w, s2["log_names"])
    with pytest.raises(B.Refused) as e:
        B.build_bank(doc, s2, N.discover_shards(w["shards"]), w["meta"], str(tmp_path / "bank"), str(tmp_path / "scratch"), SH.FakeE2(),
                     progress=lambda *_: None)
    assert e.value.why == "missing_frames"
    assert e.value.detail["missing_first50"] == [gone] and set(e.value.detail["tokens_first50"]) == excluded
