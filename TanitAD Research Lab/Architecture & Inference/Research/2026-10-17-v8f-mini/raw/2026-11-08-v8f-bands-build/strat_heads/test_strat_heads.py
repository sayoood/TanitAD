"""Tests for the strategic heads. CPU, synthetic clips of the real schemas, the frozen first ladder world model (F:/TanitAD-scratch/hwm_bands/ckpt/R35_s0.pt)."""
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strat_heads as S  # noqa: E402
import synth_clips  # noqa: E402
import train_strat_heads as TR  # noqa: E402

WM = os.environ.get("STRAT_WM_CKPT", "F:/TanitAD-scratch/hwm_bands/ckpt/R35_s0.pt")
pytestmark = pytest.mark.skipif(not os.path.exists(WM), reason="frozen ladder world model checkpoint not present")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("strat") / "d" / "e")
    return synth_clips.make(root)


def argv(data, out, extra=()):
    d = data
    return ["--wm-ckpt", WM, "--zcache-train", d["train"]["zcache"], "--zcache-eval", d["eval"]["zcache"], "--release-train", d["train"]["release"], "--release-eval", d["eval"]["release"],
            "--geom-train", d["train"]["geom"], "--geom-eval", d["eval"]["geom"], "--side-train", d["train"]["side"], "--side-eval", d["eval"]["side"], "--out", str(out),
            "--batch", "32", "--steps", "60", "--log-every", "20", "--save-every", "30", "--warmup", "10", "--lr", "3e-3", "--n-boot", "20", "--eval-stride", "10", *extra]


def test_vocab_pinned():
    TR.import_code()
    from tanitad.models import vocab_v7
    assert tuple(vocab_v7.STRATEGIC_GOAL_TOKENS_V7) == S.STR_TOKENS
    assert S.CTX_DIMS == (25, 4, 4, 3, 3)


def test_constraint_rule():
    y, m = S.constraint_rule(np.array([[0, 2, 1, 2], [-1, -1, -1, -1], [3, 0, 0, 0]]))
    assert m.tolist() == [True, False, True]
    assert y[2].tolist() == [0.0, 0.0] and y[0, 0] > 0 and y[0, 1] > 0 and y[0, 0] <= 0.4


def test_source_scan_detects():
    def bad(self, batch, z3_now, ctx_p):
        return batch["situation"]
    assert S.source_problems(bad, S.GSTR_FORBIDDEN) == ["situation"]
    assert S.source_problems(S.GstrHead.forward, S.GSTR_FORBIDDEN) == []
    assert S.source_problems(S.CtxHead.forward, S.CTX_FORBIDDEN) == []
    assert S.source_problems(S.CkptHead.forward, S.CKPT_FORBIDDEN) == []


def test_chain_matches_encode(data):
    fz, _, _ = TR.load_wm(WM)
    clips = S.load_clipset(data["train"]["zcache"], S.Labels(data["train"]["release"], data["train"]["geom"]), data["train"]["side"])
    r = TR.check_chain(fz, clips[0])
    assert r["max_abs_state"] < 1e-4 and r["step_s"] == pytest.approx(3.0)


def test_selfcheck_token_train_eval_dump(data, tmp_path, capsys):
    out = tmp_path / "o" / "p"
    os.makedirs(out)
    a = argv(data, out)
    assert TR.main(a + ["--selfcheck-only"]) == 0
    sc = json.load(open(out / "selfcheck.json"))
    assert sc["pass"] and sc["levers"]["built_equal"] and sc["detach_measured"]
    m = sc["grad_matrix"]["matrix"]
    assert m["g"]["ctx"] == 0.0 and m["ctx"]["gstr"] == 0.0 and m["g"]["wm"] == 0.0 and m["g"]["gstr"] > 0 and m["ctx"]["ctx"] > 0 and m["ck"]["ckpt"] > 0
    assert set(sc["mutations"]) == {"no_detach", "ctx_ego_named", "ctx_ego_hidden", "gstr_situation", "ctx_edge_cut"}
    assert all(v.startswith("RED as required") for v in sc["mutations"].values())
    assert sc["edges_live"]["ctx_swap"] > 0 and sc["edges_live"]["nav"] > 0
    tok = str(out / "SELFCHECK_TOKEN.json")
    # the token binds the argv: a changed lever is refused
    with pytest.raises(SystemExit) as e:
        TR.main(argv(data, out, ["--w-g", "2.0"]) + ["--token", tok])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        TR.main(a)                                                         # no token
    assert e.value.code == 2
    assert TR.main(a + ["--token", tok]) == 0
    done = json.load(open(out / "train_done.json"))
    assert done["wm_bit_equal"] and done["steps"] == 60
    ev = json.load(open(out / "eval_strat_heads.json"))
    assert ev["strategic_status"].startswith("PAUSED") and "T0" in ev["tier"]
    cap = capsys.readouterr().out
    assert "STRATEGIC PAUSED (PI item 25)" in cap
    assert ev["ctx"]["country"]["n_frames"] > 0 and len(ev["ctx"]["country"]["confusion_true_by_pred"]) == 25
    assert ev["gstr"]["swap_effect_mean_l1_token_probs"] > 0.0                  # the context swap moves the goal head
    assert "plain" in ev["gstr"] and "swap" in ev["gstr"]
    assert ev["ckpt"]["zero"]["6s"]["n"] > 0 and "delta_head_minus_cv_m" in ev["ckpt"]["zero"]["6s"]
    # dump
    dd = tmp_path / "dump" / "x"
    assert TR.main(a + ["--eval-only", "--heads", str(out / "heads_last.pt"), "--dump-dir", str(dd)]) == 0
    f = np.load(dd / "eval" / sorted(os.listdir(dd / "eval"))[0], allow_pickle=False)
    keys = json.loads(str(f["stok_keys"]))
    assert int(f["stok_dim"]) == f["stok"].shape[1] == 78 and keys[-1][2] == 78 and f["stok"].shape[0] == 101
    assert np.isfinite(f["stok"]).all()
    ctx_block = f["stok"][1:, 0:25].sum(1)
    assert np.allclose(ctx_block, 1.0, atol=1e-4)                               # country posteriors are a softmax


def test_lever_mismatch_refused(data, tmp_path):
    out = tmp_path / "m" / "n"
    os.makedirs(out)
    args = TR.make_parser().parse_args(argv(data, out))
    fz, cfg, info = TR.load_wm(WM)
    train, _ = TR.load_data(args)
    heads, opt, smp, wts, gw, mix = TR.build(args, fz, train)
    args.ctx_hidden = 128                                                       # declared 128, built 256
    with pytest.raises(SystemExit) as e:
        TR.selfcheck(args, heads, fz, opt, smp, wts, gw, mix, train, info)
    assert e.value.code == 2


def test_ctx_head_mutation_probe_independent(data):
    """The invariance probe alone catches a hidden-name ego read; the unmutated head passes."""
    torch.manual_seed(0)
    head = S.CtxHead()
    b = {"zhist": torch.randn(4, 4, 256), "hist_ok": torch.ones(4, 4, dtype=torch.bool), "act_f": torch.randn(4, 150, 2), "nav": torch.randn(4, 8)}
    assert S.assert_ctx_vision_only(head, b)["vision_only"]
    head.__class__ = TR._EgoCtxHidden
    with pytest.raises(AssertionError):
        S.assert_ctx_vision_only(head, b)


def test_wm_untouched_by_optimizer_over_heads(data):
    fz, _, _ = TR.load_wm(WM)
    s0 = fz.sha()
    assert all(not p.requires_grad for p in fz.wm.parameters())
    assert fz.sha() == s0
