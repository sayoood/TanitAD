"""Tests of v8f_t1_eval_rule_qhead.py (card ADD42, section 2 and controls h1 / h3 plumbing). The banked tools of cards ADD26 / ADD29 / ADD31 are imported from DISTRULE_DIR, read only.
The rollout itself (pinned tool, model, caches) is not run here: the pod control h1 does that."""
import json
import os
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

import qhead_testlib as TL  # noqa: E402  (sets the source path first)
import qhead_model as QM  # noqa: E402
import v8f_t1_eval_rule_qhead as T  # noqa: E402

W = T.W
D, HIST, NF = 16, 5, 7


class Banked3(nn.Module):
    """The banked LEADSUP head shape (3 outputs), small state width."""

    def __init__(self):
        super().__init__()
        self.hist = HIST
        self.frame = nn.Linear(D, 64)
        self.mlp = nn.Sequential(nn.Linear(HIST * 64, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, z):
        e = self.frame(z)
        w = e.unfold(1, self.hist, 1).permute(0, 1, 3, 2)
        return self.mlp(w.flatten(2)).float()


def fake_model(seed=0):
    torch.manual_seed(seed)
    m = types.SimpleNamespace()
    m.wm = types.SimpleNamespace(z1=lambda x: x * 1.0, lead_head=Banked3().eval())
    m.adapter = nn.Linear(D, D).eval()
    m.pos = torch.randn(D) * 0.1
    return m


def feats_batch(B=9, seed=1, float8=True):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(B, NF, D, generator=g)
    f = f.to(torch.float8_e4m3fn) if float8 else f
    ok = torch.ones(B, NF, dtype=torch.bool)
    ok[1, -2] = False
    return f, ok


@pytest.fixture()
def qh(tmp_path, monkeypatch):
    """A fitted-looking quantile head file with its record, module state reset, W.head_values restored after the test."""
    monkeypatch.setattr(W, "head_values", W.head_values)
    monkeypatch.setattr(T, "STATE", {"mode": "head", "head": None, "head_dev": None, "cols": {k: [] for k in T.SIDE_KEYS}})
    torch.manual_seed(5)
    head = QM.QHead(D, HIST)
    with torch.no_grad():
        head.mlp[2].bias.fill_(0.05)                         # so that q is below g for part of the rows and above it for the rest
    hp = str(tmp_path / "X_s0.qhead.npz")
    QM.save_head(hp, head)
    rec = {"states_record": {"ckpt_sha256": "cd" * 32}, "head_file_sha256": {hp: QM.sha256_file(hp)}}
    json.dump(rec, open(hp[:-4] + ".json", "w"))
    T.STATE["head"] = QM.load_head(hp)
    T.STATE["head_dev"] = "cpu"
    return head, hp


def independent_z1(model, feats):
    f = feats[:, -HIST:].float()
    return (model.adapter(f) + model.pos[None, None]).detach()


def test_gap_used_is_min_of_head_read_and_quantile_read(qh):
    head, hp = qh
    model = fake_model()
    f, ok = feats_batch()
    p, g, c, okk = W.head_values(model, f, ok) if False else T._ORIG_HEAD_VALUES(model, f, ok)
    p2, used, c2, ok2 = T.head_values_qhead(model, f, ok)
    z1 = independent_z1(model, f).numpy()
    q_ind = QM.functional_forward(T.STATE["head"].tensors(), z1)[:, 0] * 50.0
    g_ind = QM.functional_forward({k: v.detach().numpy() for k, v in zip(QM.TENSOR_KEYS, [model.wm.lead_head.frame.weight, model.wm.lead_head.frame.bias, model.wm.lead_head.mlp[0].weight,
                                  model.wm.lead_head.mlp[0].bias, model.wm.lead_head.mlp[2].weight, model.wm.lead_head.mlp[2].bias])}, z1)[:, 1] * 50.0
    assert np.allclose(T.STATE["cols"]["q_head"][0], q_ind, atol=1e-3) and np.allclose(g, g_ind, atol=1e-3)
    assert np.array_equal(used, np.minimum(g, T.STATE["cols"]["q_head"][0]))
    assert np.array_equal(p, p2) and np.array_equal(c, c2) and np.array_equal(okk, ok2) and not okk[1]


def test_used_never_above_head_and_cuts_somewhere(qh):
    head, hp = qh
    model = fake_model()
    f, ok = feats_batch(B=64)
    _, g, _, _ = T._ORIG_HEAD_VALUES(model, f, ok)
    _, used, _, _ = T.head_values_qhead(model, f, ok)
    assert np.all(used <= g) and np.any(used < g) and np.any(used == g)


def test_q_reads_the_last_five_frames_not_others(qh):
    head, hp = qh
    model = fake_model()
    f, ok = feats_batch(B=8, float8=False)
    f2 = f.clone()
    f2[:, :NF - HIST] = 100.0                                  # frames before the window: no influence
    a = T.head_values_qhead(model, f, ok)[1]
    b = T.head_values_qhead(model, f2, ok)[1]
    assert np.array_equal(a, b)
    f3 = f.clone()
    f3[:, -1] = f3[:, -1] + 1.0                                # the end frame: influence
    c = T.head_values_qhead(model, f3, ok)[1]
    assert not np.array_equal(a, c)


def test_equals_head_mode_gives_head_read_bit_for_bit(qh):
    head, hp = qh
    T.STATE["mode"] = "equals-head"
    model = fake_model()
    f, ok = feats_batch(B=33)
    _, g, _, _ = T._ORIG_HEAD_VALUES(model, f, ok)
    _, used, _, _ = T.head_values_qhead(model, f, ok)
    assert np.array_equal(used, g) and used.dtype == g.dtype
    assert np.array_equal(T.STATE["cols"]["q_head"][0], g)


def test_nan_gap_stays_and_nan_quantile_leaves_head_gap():
    g = np.array([np.nan, 10.0, 5.0, 8.0], np.float32)
    q = np.array([3.0, 4.0, np.nan, 20.0], np.float32)
    u = QM.used_gap(g, q)
    assert np.isnan(u[0]) and u[1] == 4.0 and u[2] == 5.0 and u[3] == 8.0 and u.dtype == np.float32


def test_float8_cache_dtype_runs_through_and_matches_float32_path(qh):
    head, hp = qh
    model = fake_model()
    f8, ok = feats_batch(float8=True)
    f32 = f8.float()
    a = T.head_values_qhead(model, f8, ok)[1]
    b = T.head_values_qhead(model, f32, ok)[1]
    assert np.array_equal(a, b)


def test_side_columns_are_logged_per_call_and_saved(qh, tmp_path):
    head, hp = qh
    model = fake_model()
    for s in (1, 2):
        f, ok = feats_batch(B=4 + s, seed=s)
        T.head_values_qhead(model, f, ok)
    out = str(tmp_path / "x_qh.npz")
    T.save_side(out, {"q_mode": "head", "card": "PREREG_V8F_PROOF_ADD42"})
    z = np.load(out)
    assert len(z["g_head"]) == 5 + 6 == len(z["q_head"]) == len(z["g_used"]) == len(z["p_head"]) == len(z["ok"])
    assert np.array_equal(z["g_used"], np.minimum(z["g_head"], z["q_head"]))


def run_main(argv, monkeypatch, tmp_path):
    seen = {}

    def fake_x_main(rest):
        seen["rest"] = list(rest)
        seen["p_min"] = W.HEAD_P_MIN
        seen["hv"] = W.head_values
        model = fake_model()
        f, ok = feats_batch(B=3)
        seen["used"] = W.head_values(model, f, ok)[1]
        return 0
    monkeypatch.setattr(T.X, "main", fake_x_main)
    return T.main(argv), seen


def test_main_installs_the_read_and_writes_side_file(qh, tmp_path, monkeypatch):
    head, hp = qh
    rl = str(tmp_path / "rule_log_X.npz")
    rc, seen = run_main(["--head-p-min", "0.4", "--qhead", hp, "--tree", "T", "--mix-presence", "head", "--mix-gap", "head", "--mix-closing", "head", "--rule-log", rl, "run"], monkeypatch, tmp_path)
    assert rc == 0 and seen["p_min"] == 0.4 and seen["hv"] is T.head_values_qhead and "--head-p-min" not in seen["rest"] and "--qhead" not in seen["rest"]
    z = np.load(str(tmp_path / "rule_log_X_qh.npz"))
    assert len(z["g_used"]) == 3 and str(z["q_mode"]) == "head" and float(z["head_p_min"]) == 0.4 and str(z["qhead_sha256"]) == QM.sha256_file(hp)


def refusal(argv, monkeypatch, tmp_path, match):
    with pytest.raises(SystemExit, match=match):
        run_main(argv, monkeypatch, tmp_path)


MIXH = ["--mix-presence", "head", "--mix-gap", "head", "--mix-closing", "head"]


def test_main_refusals(qh, tmp_path, monkeypatch):
    head, hp = qh
    refusal(["--head-p-min", "1.5", "--qhead", hp] + MIXH, monkeypatch, tmp_path, "strictly between")
    refusal(["--head-p-min", "0.5"] + MIXH, monkeypatch, tmp_path, "exactly one")
    refusal(["--head-p-min", "0.5", "--qhead", hp, "--q-equals-head"] + MIXH, monkeypatch, tmp_path, "exactly one")
    refusal(["--head-p-min", "0.5", "--qhead", hp, "--mix-presence", "head", "--mix-gap", "head"], monkeypatch, tmp_path, "name the mix")
    refusal(["--head-p-min", "0.5", "--qhead", hp, "--mix-presence", "label", "--mix-gap", "head", "--mix-closing", "head"], monkeypatch, tmp_path, "all-head mix only")
    refusal(["--head-p-min", "0.5", "--qhead", str(tmp_path / "nope.npz")] + MIXH, monkeypatch, tmp_path, "no head file")
    ev = str(tmp_path / "LS_s0_eval.qhead.npz")
    open(ev, "wb").write(open(hp, "rb").read())
    refusal(["--head-p-min", "0.5", "--qhead", ev] + MIXH, monkeypatch, tmp_path, "evaluation split")


def test_main_refuses_checkpoint_mismatch_and_tampered_head(qh, tmp_path, monkeypatch):
    head, hp = qh
    ck = str(tmp_path / "ckpt_last.pt")
    open(ck, "wb").write(b"not the checkpoint the head was fitted on")
    refusal(["--head-p-min", "0.5", "--qhead", hp, "--ckpt", ck] + MIXH, monkeypatch, tmp_path, "fitted on states of checkpoint")
    rec = json.load(open(hp[:-4] + ".json"))
    rec["states_record"]["ckpt_sha256"] = QM.sha256_file(ck)
    json.dump(rec, open(hp[:-4] + ".json", "w"))
    run_main(["--head-p-min", "0.5", "--qhead", hp, "--ckpt", ck] + MIXH, monkeypatch, tmp_path)           # matches now
    QM.save_head(hp, QM.QHead(D, HIST))                                                                      # a different head under the recorded sha256
    refusal(["--head-p-min", "0.5", "--qhead", hp] + MIXH, monkeypatch, tmp_path, "does not match the sha256")


def test_main_equals_head_mode_needs_no_head_file(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "head_values", W.head_values)
    monkeypatch.setattr(T, "STATE", {"mode": None, "head": None, "head_dev": None, "cols": {k: [] for k in T.SIDE_KEYS}})
    rc, seen = run_main(["--head-p-min", "0.5", "--q-equals-head"] + MIXH, monkeypatch, tmp_path)
    assert rc == 0 and seen["p_min"] == 0.5 and T.STATE["mode"] == "equals-head"
    model = fake_model()
    f, ok = feats_batch(B=3)
    _, g, _, _ = T._ORIG_HEAD_VALUES(model, f, ok)
    assert np.array_equal(seen["used"], g)
