"""``internal_t1`` — the REF-C T1 harness (``taniteval/tools/refcv3_arm.py``) mapped into the
run-dir contract (W1). The mapping is a PURE function of the tool's record, so it is pinned with
literals; the tool itself is not re-run here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from taniteval.bench import contract as C
from taniteval.bench import internal_t1 as T


def _ade(mean, lo, hi, n_win, n_ep, boot=2000):
    return {"mean": mean, "lo": lo, "hi": hi, "n_windows": n_win, "n_episodes": n_ep, "n_boot": boot,
            "estimator": "episode_cluster_bootstrap"}


def _rec(n_ep=40, n_win=800):
    fam = {"longitudinal": {"speed_mae_mps": 1.5, "along_mae_m": 2.0, "n_windows": n_win,
                            "distance_keeping": {"status": "UNAVAILABLE", "reason": "no lead block", "n": 0}},
           "lateral": {"cross_mae_m": 0.4, "heading_mae_deg": 3.0, "curvature_mae_1pm": 0.01,
                       "yaw_rate_mae_degps": 2.0, "n_windows": n_win},
           "tactical": {"status": "OK", "n": n_win, "lateral_decision": {"accuracy": 0.7, "kappa": 0.2},
                        "longitudinal_decision": {"accuracy": 0.6, "kappa": 0.1},
                        "goal_setting": {"goal_point_error_m": 3.3}},
           "strategic": {"status": "UNAVAILABLE", "reason": "nav labels off", "n": n_win}}
    return {"n_episodes": n_ep, "n_windows": n_win, "mode": "rollout+analyze", "ckpt": "D:/x/ckpt.pt",
            "_provenance": {"verdict": "REAL_CHECKPOINT_ON_REAL_CORPUS"},
            "arms": {
                "os": {"tier": "T1", "four_families": dict(fam),
                       "intervals": {"metrics": {"ade_dense_m": _ade(1.2345, 1.0, 1.5, n_win, n_ep)}}},
                "ha0": {"tier": "T1", "four_families": dict(fam),
                        "intervals": {"metrics": {"ade_dense_m": _ade(2.0, 1.8, 2.2, n_win, n_ep)}}},
                "ha": {"tier": "T1", "four_families": dict(fam),
                       "intervals": {"metrics": {"ade_dense_m": _ade(1.9, 1.7, 2.1, n_win, n_ep)}}}},
            "refcv3": {"families_paired": {"paired_os_minus_ha0": {"direction": "os - ha0", "families": {"ADE": {
                "ade_m": {"delta": -0.7655, "lo": -0.9, "hi": -0.6, "separated": True, "n_windows": n_win,
                          "n_episodes": n_ep, "estimator": "paired_episode_cluster_bootstrap"}}}}}}}


#: what run_benchmark supplies in production: the triple built from the TOOL's own manifest, with a
#: sha256 computed from the checkpoint file. ⛔ NOT optional -- a model arm without it is an
#: anonymous leaderboard row and validate_summary refuses it (W4, 2026-09-20).
FAKE_CKPT = {"path": "C:/fake/run/ckpt.pt", "sha256": "b" * 64,
             "registry_key_status": "NOT NAMED by the operator (--registry-key)"}


def test_summary_shape_and_literals():
    s = T.summarize_record(_rec(), "20260920T000000Z-internal_t1-none-abcdef", "v2ep-eval124", ckpt=FAKE_CKPT)
    assert C.validate_summary(s) == [], C.validate_summary(s)
    assert s["protocol"] == "TanitAD_T1_refc_physicalai" and s["benchmark"] == "internal_t1"
    assert s["headline_metric"]["higher_is_better"] is False and s["headline_metric"]["column"] == "ade_dense_m"
    assert s["floors"] == ["ha0"]
    assert s["arms"]["os"]["kind"] == "model" and s["arms"]["ha0"]["kind"] == "floor"
    assert s["arms"]["os"]["headline"]["value"] == 1.2345
    assert s["arms"]["os"]["paired"]["ha0"]["headline_delta"] == -0.7655
    assert s["arms"]["os"]["paired"]["ha0"]["interval"]["separated"] is True
    assert s["arms"]["ha0"]["paired"]["ha0"] == {"status": "SELF"}
    assert s["stamps"]["tier"] == "T1*"                      # the ruling is OPEN for a model with no action input


def test_interval_is_admissible_only_above_the_rg14_floor():
    ok = T.summarize_record(_rec(n_ep=40), "r", "s")["arms"]["os"]["interval"]
    assert ok["status"] == "OK" and ok["n_clusters"] == 40 and ok["cluster_unit"] == "episode"
    few = T.summarize_record(_rec(n_ep=2), "r", "s")["arms"]["os"]["interval"]
    assert few["status"] == "UNAVAILABLE" and few["n"] == 2 and "RG-14" in few["reason"]


def test_families_travel_per_family_with_reason_and_n():
    s = T.summarize_record(_rec(), "r", "s")
    f = s["arms"]["os"]["families"]
    assert set(f) == {"longitudinal", "lateral", "tactical", "strategic"}
    assert f["longitudinal"]["status"] == "PARTIAL" and "distance-keeping" in f["longitudinal"]["reason"]
    assert f["lateral"]["status"] == "OK" and f["lateral"]["metrics"]["cross_mae_m"] == 0.4
    assert f["strategic"]["status"] == "UNAVAILABLE" and f["strategic"]["reason"]


def test_paired_falls_back_to_a_difference_of_means_when_the_record_has_no_paired_block():
    rec = _rec()
    rec["refcv3"]["families_paired"] = {}
    s = T.summarize_record(rec, "r", "s")
    p = s["arms"]["os"]["paired"]["ha0"]
    assert p["status"] == "OK" and p["headline_delta"] == pytest.approx(1.2345 - 2.0)
    assert "no paired CI" in p["direction"]


def test_cmd_is_the_tool_unmodified(tmp_path):
    import argparse
    a = argparse.Namespace(ckpt="C:/x/ckpt.pt", episodes="C:/cache", labels="C:/labels.jsonl.gz", config=None,
                           grid="2s", episodes_n=3, window_stride=40, analyze_only=None, n_boot=2000, seed=0,
                           t1_args="--no-navshuf")
    cmd = T.build_cmd(a, tmp_path, "cpu", tmp_path / "dump")
    assert cmd[1].replace("\\", "/").endswith("taniteval/tools/refcv3_arm.py")
    for flag in ("--ckpt", "--episodes", "--labels", "--dump-dir", "--device", "--grid", "--out", "--no-navshuf"):
        assert flag in cmd, flag
    assert cmd[cmd.index("--device") + 1] == "cpu"
    a.analyze_only = str(tmp_path / "banked_dump")
    cmd = T.build_cmd(a, tmp_path, "cpu", tmp_path / "dump")
    assert "--analyze-only" in cmd and "--ckpt" not in cmd        # 0 GPU: re-analyse a banked dump


E9 = Path("C:/Users/Admin/AppData/Local/Temp/claude/D--Projects-TanitAD/"
          "bbcd6d8b-6831-4ca4-8774-3051e3caa1c6/scratchpad/e9_refcv3arm.json")


@pytest.mark.skipif(not E9.exists(), reason="NO_TREE: E9's banked refcv3_arm record is not on this machine")
def test_real_refcv3_arm_record_maps_and_validates():
    """E9's REAL record (refcv6 tiny @ 2,000 steps, 9 windows / 2 episodes) — shape, not capability."""
    rec = json.loads(E9.read_text(encoding="utf-8"))
    s = T.summarize_record(rec, "20260920T000000Z-internal_t1-a3-abcdef", "v2ep-eval124clean-416x1024cyl-halfB",
                           ckpt=T.ckpt_triple(rec, sha256="c" * 64))
    assert C.validate_summary(s) == [], C.validate_summary(s)
    assert s["arms"]["os"]["headline"]["value"] == 2.9098          # the tool's own printed ADE
    assert s["arms"]["ha0"]["headline"]["value"] == 0.5417
    assert s["arms"]["os"]["paired"]["ha0"]["headline_delta"] == 2.3682
    assert s["arms"]["os"]["interval"]["status"] == "UNAVAILABLE"  # 2 episodes < RG-14's 8
    assert s["arms"]["os"]["tier"] == "T1"


# --- the checkpoint a MODEL arm names (W4, 2026-09-20) ----------------------------------------

def test_the_real_ckpt_path_comes_from_the_TOOL_manifest_not_rec_ckpt():
    """⛔ MEASURED: `rec["ckpt"]` is None on every run of this tool while the path it actually loaded
    sits at refcv3.manifest.model.ckpt. Reading the wrong field is how an internal_t1 run published
    a MODEL arm with a null checkpoint -- an anonymous row -- with the answer one file down."""
    rec = {"ckpt": None, "refcv3": {"manifest": {"model": {"ckpt": "C:/x/run/ckpt.pt"}}}}
    assert T.ckpt_path_from_record(rec) == "C:/x/run/ckpt.pt"
    assert rec["ckpt"] is None, "the misleading field is still there; we simply do not read it first"
    assert T.ckpt_path_from_record({"ckpt": "C:/fallback.pt"}) == "C:/fallback.pt"
    assert T.ckpt_path_from_record({}) is None


def test_a_model_summary_WITHOUT_a_sha256_is_refused():
    """The guarantee itself: no sha256 => the row cannot identify its checkpoint => contract error.
    ⭐ This is the state the suite was ACTUALLY in before today, so it must be RED-able."""
    s = T.summarize_record(_rec(), "20260920T000000Z-internal_t1-none-abcdef", "v2ep-eval124")
    assert any(a.get("kind") == "model" for a in s["arms"].values()), "control: there must BE a model arm"
    errs = C.validate_summary(s)
    assert any("/provenance/ckpt/sha256" in e for e in errs), errs


def test_registry_key_absence_must_be_STATED_not_silent():
    rec = {"refcv3": {"manifest": {"model": {"ckpt": "C:/x.pt"}}}}
    t = T.ckpt_triple(rec, sha256="d" * 64)
    assert t["registry_key"] is None and "NOT NAMED" in t["registry_key_status"]
    t2 = T.ckpt_triple(rec, sha256="d" * 64, registry_key="refcv6-tiny-2k")
    assert t2["registry_key"] == "refcv6-tiny-2k" and "registry_key_status" not in t2
