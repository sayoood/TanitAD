"""Refusal tests of run_dumps_stopbal.py with PLANTED fake banks (tmp dirs, tiny files; nothing is dumped, no model is loaded, subprocess.run is made to raise).
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_run_dumps_stopbal.py"""
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_dumps_stopbal as R  # noqa: E402
import stopbal_read as S  # noqa: E402


def plant(root, tag, ckpt=b"fake-checkpoint", levers=None, commit=None, manifest=True, selfcheck=True, sha_override=None, dvb="PASS"):
    b = os.path.join(root, f"stopbal_{tag}")
    os.makedirs(b)
    if ckpt is not None:
        open(os.path.join(b, "ckpt_last.pt"), "wb").write(ckpt)
    if manifest:
        sha = sha_override or hashlib.sha256(ckpt or b"").hexdigest()
        m = {"tag": tag, "files": {"ckpt_last.pt": {"sha256": sha}}}
        if commit:
            m["commit"] = commit
        json.dump(m, open(os.path.join(b, "BANK_MANIFEST.json"), "w"))
    if selfcheck:
        g, l = S.CARD_WEIGHTS[S.kind_of(tag)] if levers is None else levers
        json.dump({"declared_vs_built": dvb, "levers": {"goal_pos_weight_stop": g, "lon_class_weight_stop": l}}, open(os.path.join(b, "selfcheck.json"), "w"))


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "PULL", str(tmp_path / "pull"))
    monkeypatch.setattr(R, "OUT", str(tmp_path / "t1"))
    os.makedirs(R.PULL)

    def boom(*a, **k):
        raise AssertionError("a refused arm must launch nothing")
    monkeypatch.setattr(R.subprocess, "run", boom)
    return R.PULL


@pytest.mark.parametrize("tag", list(R.ARMS))
def test_planted_good_bank_passes_the_refusal_gate(box, tag):
    plant(box, tag)
    why, want = R.refusal(tag)
    assert why is None and want == hashlib.sha256(b"fake-checkpoint").hexdigest()


def test_missing_checkpoint_refused(box):
    plant(box, "BAL-G_s0", ckpt=None)
    assert "checkpoint missing" in R.run("BAL-G_s0")["status"]
    assert not os.path.exists(R.OUT)                    # nothing was created, nothing launched


def test_absent_bank_refused(box):
    assert "checkpoint missing" in R.run("BAL-L_s1")["status"]


def test_missing_manifest_and_selfcheck_refused(box):
    plant(box, "BAL-NULL_s0", manifest=False)
    assert "BANK_MANIFEST.json missing" in R.run("BAL-NULL_s0")["status"]
    plant(box, "BAL-NULL_s1", selfcheck=False)
    assert "selfcheck.json missing" in R.run("BAL-NULL_s1")["status"]


@pytest.mark.parametrize("tag,levers,needle", [
    ("BAL-G_s0", (1.0, 1.0), "did not test it"),             # BAL-G whose build read 1.0 for its own lever
    ("BAL-G_s1", (4.5616, 4.6991), "lon_class_weight_stop reads 4.6991"),
    ("BAL-L_s0", (1.0, 1.0), "did not test it"),
    ("BAL-L_s1", (4.5616, 4.6991), "goal_pos_weight_stop reads 4.5616"),
    ("BAL-NULL_s0", (4.5616, 1.0), "goal_pos_weight_stop reads 4.5616"),
    ("BAL-NULL_rep", (1.0, 4.6991), "lon_class_weight_stop reads 4.6991"),
    ("BAL-G_s0", (4.56, 1.0), "reads 4.56"),
])
def test_control1_lever_mismatch_refused(box, tag, levers, needle):
    plant(box, tag, levers=levers)
    s = R.run(tag)["status"]
    assert s.startswith("REFUSED: control 1 fails") and needle in s, s


def test_declared_vs_built_not_pass_refused(box):
    plant(box, "BAL-G_s0", dvb="FAIL")
    assert "G-DVB" in R.run("BAL-G_s0")["status"]


def test_wrong_commit_and_sha_mismatch_refused(box):
    plant(box, "BAL-G_s0", commit="2bca1c52abbc2346d7380d33ae78b324cbda9df7")
    assert "code of record" in R.run("BAL-G_s0")["status"]
    plant(box, "BAL-G_s1", sha_override="0" * 64)
    assert "sha256 differs" in R.run("BAL-G_s1")["status"]


def test_done_file_skips(box):
    os.makedirs(R.OUT)
    open(os.path.join(R.OUT, "done_BAL-G_s0_m10_causal_heads.txt"), "w").write("{}")
    assert R.run("BAL-G_s0")["status"].startswith("skip")


def test_unknown_tag_refused():
    assert R.refusal("BAL-X_s0")[0].startswith("unknown tag")


def test_finalize_requires_complete_outputs(box):
    plant(box, "BAL-G_s0")
    os.makedirs(R.OUT + "/dump_BAL-G_s0_m10_causal_heads")
    assert R.finalize("BAL-G_s0", {"rc": 0}).startswith("FAILED: outputs incomplete")
    assert not os.path.exists(R.OUT + "/done_BAL-G_s0_m10_causal_heads.txt")


def test_stopdec_arm_substitution_is_exact():
    import stopdec_arm as A
    src = open(os.path.join(A.OTB, "topk_read.py"), encoding="utf-8").read()
    out = A.patched_topk_source(src, "X:/tree", "X:/t1")
    assert 'TREE = "X:/tree"' in out.replace("'", '"') and out.count("otband_read/tree") < src.count("otband_read/tree")
    with pytest.raises(AssertionError):
        A.patched_topk_source("TREE = 1", "a", "b")
