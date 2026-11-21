"""Tests of the gate patch of dscale10k_check.py (2026-11-17): batch kinds of the reach table, stage-B required checks, C-BANK, G-DAGGER. Each check has its deliberate-regression arm (RED)."""
import copy
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dscale10k_check as C  # noqa: E402


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _bank(tmp_path, n_shards=4, partial=False, frac=0.77, kb4=True, kb0_mismatch=0, ckpt=None):
    d = tmp_path / "bank"
    d.mkdir()
    shards = []
    for k in range(n_shards):
        f = d / f"shard_{k:03d}of{n_shards:03d}.npz"
        f.write_bytes(b"shard%d" % k)
        shards.append({"k": k, "file": f.name, "file_sha256": _sha(f), "n_records": 1, "content_sha256": "x"})
        (d / f"shard_{k:03d}of{n_shards:03d}.provenance.json").write_text(json.dumps({"kb0_tick0_identity": {"checked": 5, "mismatch": kb0_mismatch if k == 1 else 0}, "kb1_replay": {"mismatch_chunks": 0}}))
    ck = tmp_path / "ck.pt"
    ck.write_bytes(b"ckpt")
    bj = {"schema": "v8f-dagger-bank-v1", "n_shards": n_shards, "partial": partial, "shards": shards, "coverage": {"fraction": frac, "blocked": frac < 0.5},
          "common": {"partial": partial, "ckpt_sha256": _sha(ck) if ckpt is None else ckpt, "eval": {"n_clips": 141, "overlap": 0},
                     "train": {"n_clips": 4512, "n_windows": 306566, "train_sha256": "t" * 64}}}
    (d / "bank.json").write_text(json.dumps(bj))
    (d / "PROVENANCE.json").write_text(json.dumps({"controls": {"K-B4": {"PASS": kb4}, "K-B5": {"PASS": True}}, "bank_sha256": _sha(d / "bank.json")}))
    dry = {"n_clips": 4512, "n_windows": 306566, "train_sha256": "t" * 64}
    return str(d), str(ck), dry


def test_bank_pass_and_regressions(tmp_path):
    d, ck, dry = _bank(tmp_path)
    assert C.bank_problems(d, ck, dry) == []
    assert C.bank_problems(d, ck, dict(dry, n_windows=1))                       # a bank over another window list
    assert C.bank_problems(d, ck, None)
    assert C.bank_problems(str(tmp_path / "nowhere"), ck, dry)
    other = tmp_path / "other.pt"
    other.write_bytes(b"another checkpoint")
    assert C.bank_problems(d, str(other), dry)                                  # K-B2: not this arm's init checkpoint


def test_bank_one_shard_partial_coverage_controls(tmp_path):
    for i, kw in enumerate((dict(n_shards=1), dict(partial=True), dict(frac=0.3), dict(kb4=False), dict(kb0_mismatch=2))):
        sub = tmp_path / f"r{i}"
        sub.mkdir()
        d, ck, dry = _bank(sub, **kw)
        assert C.bank_problems(d, ck, dry), kw


def test_bank_shard_file_tampered(tmp_path):
    d, ck, dry = _bank(tmp_path)
    open(os.path.join(d, "shard_002of004.npz"), "ab").write(b"x")
    assert any("shard 2" in p for p in C.bank_problems(d, ck, dry))


def _sc():
    return {"declared_vs_built": "PASS", "acc_probe": {"pass": True, "w_acc_declared": 49.6969, "head_o_grad_diff": 4.2},
            "dagger_probe": {"pass": True, "loss_diff": 0.3, "grad_diff_all": 7.0, "control_repeat_loss_diff": 0.0, "control_repeat_grad_diff": 0.0, "control_p0_same_object": True,
                             "control_p0_loss_diff": 0.0, "control_p0_grad_diff": 0.0, "control_prior_grad_diff": 0.0, "control_wm_grad_diff": 0.0},
            "info": {"dagger": {"coverage": {"fraction": 0.77, "blocked": False}}}}


ARGV = ["--dagger-p", "0.1", "--w-acc", "49.6969"]


def test_dagger_pass_and_mutations():
    assert C.dagger_problems(_sc(), ARGV) == []
    s = _sc(); s["dagger_probe"] = {"dagger": "off", "pass": True, "skipped": "--dagger-p 0"}
    assert C.dagger_problems(s, ARGV)                                           # lever off
    s = _sc(); s["dagger_probe"]["grad_diff_all"] = 0.0
    assert C.dagger_problems(s, ARGV)                                           # a swap that never reaches the loss
    s = _sc(); s["dagger_probe"]["control_p0_grad_diff"] = 1e-9
    assert C.dagger_problems(s, ARGV)                                           # the exact-0.0 control
    s = _sc(); s["dagger_probe"]["control_prior_grad_diff"] = 1e-12
    assert C.dagger_problems(s, ARGV)
    s = _sc(); s["acc_probe"]["w_acc_declared"] = 0.0
    assert C.dagger_problems(s, ARGV)
    s = _sc(); s["info"] = {}
    assert C.dagger_problems(s, ARGV)
    assert C.dagger_problems(_sc(), ["--dagger-p", "0"])


def test_required_checks_stage_b():
    a = C.required_checks("B_s0", "A")
    b = C.required_checks("B_s0", "B")
    assert "C-BANK" not in a and "G-DAGGER" not in a
    assert set(b) - set(a) == {"C-BANK", "G-DAGGER"}
    assert "C-MERGE" in C.required_checks("U_s0", "A") and "C-MERGE" not in b


def _kind(grad=1.0):
    t = {k: {"finite": True, "value": 1.0, "grad": grad} for k in C.STAGE_A_TERMS}
    t["goal_vlm"] = {"finite": True, "value": 0.0, "grad": 0.0}
    t["cons_lc"] = {"finite": True, "value": 0.0, "grad": 0.0}
    return t


def test_reach_kinds_b_set_needs_only_corpus():
    corpus = _kind()
    corpus["goal_vlm"] = {"finite": True, "value": 0.2, "grad": 3.0}
    rec = {"corpus": corpus, "control": {"goal_vlm_grad": 5.0}}
    assert C.reach_problems(rec, False, ("corpus",)) == []
    assert C.reach_problems(rec)                                                # the union default still demands tenk and mixed
    rec2 = copy.deepcopy(rec)
    rec2["corpus"]["wm"]["grad"] = 0.0
    assert C.reach_problems(rec2, False, ("corpus",))                           # a dead term on the corpus batch
    rec3 = copy.deepcopy(rec)
    rec3["control"] = {"goal_vlm_grad": 0.0}
    assert C.reach_problems(rec3, False, ("corpus",))                           # the control must read > 0


def test_token_binds_the_extra_files_and_a_runner_must_pass_them(tmp_path):
    root = tmp_path / "r"
    (root / "stack" / "scripts").mkdir(parents=True)
    (root / "stack" / "scripts" / "v8f_mini_train.py").write_text("x")
    for d in ("tanitad", "tests"):
        (root / "stack" / d).mkdir()
    (root / "taniteval").mkdir()
    ev = tmp_path / "ev.json"
    ev.write_text("{}")
    toks = ["--seed", "0"]
    b, bad = C.binding("B_s0", "A", toks, str(root), "c", {"evidence": str(ev)})
    assert not bad
    tp = tmp_path / "tok.json"
    C.write_token(str(tp), b, {c: {"pass": True} for c in C.required_checks("B_s0", "A")}, "B_s0")
    assert C.token_problems(str(tp), "B_s0", "A", toks, str(root), "c", {"evidence": str(ev)}) == []
    assert C.token_problems(str(tp), "B_s0", "A", toks, str(root), "c", None)             # a runner that does not re-bind the extras refuses a good token
    ev.write_text('{"x": 1}')
    assert C.token_problems(str(tp), "B_s0", "A", toks, str(root), "c", {"evidence": str(ev)})   # edited evidence
    # a stage-B token needs C-BANK and G-DAGGER
    b2, _ = C.binding("B_s0", "B", toks, str(root), "c", {})
    try:
        C.write_token(str(tmp_path / "t2.json"), b2, {c: {"pass": True} for c in C.required_checks("B_s0", "A")}, "B_s0")
        raise AssertionError("a stage-B token was written without C-BANK / G-DAGGER")
    except ValueError as e:
        assert "C-BANK" in str(e)
