"""CPU tests for arm_logits_pass.py with a tiny stand-in for REFe (same scorer call sequence), no nuplan, no real checkpoint.
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_arm_logits_pass.py   (CUDA_VISIBLE_DEVICES=-1)"""
import copy
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm_logits_pass as A  # noqa: E402

M, T, D = 5, 8, 3


class Blk(nn.Module):
    def __init__(self, w=8):
        super().__init__()
        self.q = nn.Linear(w, w)
        self.kv = nn.Linear(w, w)

    def forward(self, s, ctx):
        a = torch.softmax(self.q(s) @ self.kv(ctx).transpose(-1, -2), -1) @ ctx
        return s + a


class TinyREFe(nn.Module):
    """trunk -> ctx, queries -> traj; the scorer is score_q_mlp -> score_dec -> score_head exactly as REFe.forward calls it."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.manual_seed(seed)
        self.backbone = nn.Linear(4, 8)
        self.traj_head = nn.Linear(8, T * D)
        self.score_q_mlp = nn.Sequential(nn.Linear(T * D, 8), nn.ReLU(), nn.Linear(8, 8))
        self.score_dec = nn.ModuleList([Blk(), Blk()])
        self.score_head = nn.Linear(8, 6)
        self.queries = nn.Parameter(torch.randn(M, 8))

    def forward(self, img):
        ctx = self.backbone(img)                                    # [1, P, 8]
        traj = self.traj_head(self.queries[None]).view(1, M, T, D)
        s = self.score_q_mlp(traj.detach().flatten(2))
        for b in self.score_dec:
            s = b(s, ctx)
        return traj, self.score_head(s)


class StubPlanner:
    def __init__(self, model):
        self.model = model

    def infer(self, ego, img):
        with torch.no_grad():
            traj, sc = self.model(img)
        return traj[0], sc[0], int(sc[0].sum(-1).argmax())

    def executed(self, traj, k):
        return traj[k]


def to_navsim(p):                                                    # any deterministic [T,3] -> [8,3]
    return np.asarray(p, dtype=np.float64)[:8]


def state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def scorer_perturbed(st, eps=0.05, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = dict(st)
    for k, v in st.items():
        if A.is_scorer_key(k):
            out[k] = v + eps * torch.randn(v.shape, generator=g)
    return out


def items(n, seed=7):
    g = torch.Generator().manual_seed(seed)
    imgs = [torch.randn(1, 6, 4, generator=g) for _ in range(n)]
    return [(f"tok{i:03d}", None, imgs[i]) for i in range(n)]


def make_ref(model, toks_imgs, perturb=0.0):
    pl, tap = StubPlanner(model), A.Tap(model)
    ctl = A.ScorerArm(A.CONTROL, model)
    tk, pr, lg = [], [], []
    for tok, ego, img in toks_imgs:
        p, l, d, k, _ = A.process_token(pl, ego, img, {A.CONTROL: ctl}, tap, to_navsim)
        tk.append(tok); pr.append(p); lg.append(l[A.CONTROL] + perturb)
    return tk, np.stack(pr), np.stack(lg)


def write_ref(path, tk, pr, lg):
    np.savez(path, token=np.array(tk), proposals=pr, logits=lg)


def run(tmp_path, model, arms, n=7, chunk=3, ref=None, name="out", fac_hook=None):
    its = items(n)
    toks = [t for t, _, _ in its]
    if ref is None:
        ref = tmp_path / "ref.npz"
        write_ref(ref, *make_ref(model, its))
    st = A.Store(str(tmp_path / name), toks, "basesha", {k: f"sha_{k}" for k in arms if k != A.CONTROL}, chunk)

    def fac(rem):
        for t, e, i in its:
            if t in rem:
                if fac_hook:
                    fac_hook(t)
                yield t, e, i
    pl = StubPlanner(model)
    return A.run_pass(pl, arms, fac, toks, st, A.Reference(str(ref)), to_navsim, 0.0, log=lambda m: None), st


def test_arm_equal_to_base_matches_control_and_reference(tmp_path):
    m = TinyREFe()
    arms = {A.CONTROL: A.ScorerArm(A.CONTROL, m), "SAME": A.ScorerArm("SAME", m, state(m))}
    (verdict, rep), st = run(tmp_path, m, arms)
    assert verdict == "ZZARMPASS_OK", verdict
    assert rep["internal_control_max_abs_diff"] == 0.0 and rep["reference_control"]["max_logit_diff"] == 0.0
    c = np.load(st.out_path(A.CONTROL)); s = np.load(st.out_path("SAME"))
    assert np.array_equal(c["logits"], s["logits"]) and np.array_equal(c["proposals"], s["proposals"])
    assert c["logits"].shape == (7, M, 6) and c["logits"].dtype == np.float32


def test_arm_logits_equal_a_standalone_forward_of_that_arm(tmp_path):
    """the shared pass gives the arm the logits a full forward of the arm's own model would"""
    m = TinyREFe()
    st_arm = scorer_perturbed(state(m))
    arms = {A.CONTROL: A.ScorerArm(A.CONTROL, m), "S1E": A.ScorerArm("S1E", m, st_arm)}
    (verdict, _), st = run(tmp_path, m, arms)
    assert verdict == "ZZARMPASS_OK"
    full = TinyREFe(); full.load_state_dict(st_arm)
    its = items(7)
    want = np.stack([full(i)[1][0].detach().numpy() for _, _, i in its])
    got = np.load(st.out_path("S1E"))["logits"]
    assert np.array_equal(got, want.astype(np.float32))
    assert not np.array_equal(got, np.load(st.out_path(A.CONTROL))["logits"])      # and it really differs from the control


def test_swapped_arm_names_land_in_the_right_directory(tmp_path):
    m = TinyREFe()
    a1, a2 = scorer_perturbed(state(m), seed=1), scorer_perturbed(state(m), seed=2)
    arms = {A.CONTROL: A.ScorerArm(A.CONTROL, m), "A": A.ScorerArm("A", m, a1), "B": A.ScorerArm("B", m, a2)}
    arms_sw = {A.CONTROL: A.ScorerArm(A.CONTROL, m), "A": A.ScorerArm("A", m, a2), "B": A.ScorerArm("B", m, a1)}
    (v1, _), s1 = run(tmp_path, m, arms, name="o1", ref=None)
    (v2, _), s2 = run(tmp_path, m, arms_sw, name="o2", ref=tmp_path / "ref.npz")
    assert v1 == v2 == "ZZARMPASS_OK"
    la1, lb1 = np.load(s1.out_path("A"))["logits"], np.load(s1.out_path("B"))["logits"]
    la2, lb2 = np.load(s2.out_path("A"))["logits"], np.load(s2.out_path("B"))["logits"]
    assert not np.array_equal(la1, la2) and np.array_equal(la1, lb2) and np.array_equal(lb1, la2)


def test_changed_non_scorer_key_is_refused_naming_the_key():
    m = TinyREFe()
    base = state(m)
    arm = scorer_perturbed(base)
    arm["traj_head.weight"] = arm["traj_head.weight"] + 1e-7
    with pytest.raises(A.ArmRefused, match=r"non-scorer key traj_head\.weight"):
        A.verify_arm("X", base, arm, {})
    ok = scorer_perturbed(base)
    A.verify_arm("X", base, ok, {"n_score_components": 6})                         # scorer-only change passes
    with pytest.raises(A.ArmRefused, match="output count"):
        A.verify_arm("X", base, ok, {"n_score_components": 7})
    extra = dict(ok); extra["zzz.extra"] = torch.zeros(1)
    with pytest.raises(A.ArmRefused, match="zzz.extra"):
        A.verify_arm("X", base, extra, {})
    miss = {k: v for k, v in ok.items() if k != "queries"}
    with pytest.raises(A.ArmRefused, match="queries"):
        A.verify_arm("X", base, miss, {})


def test_perturbed_control_fails_the_reference_comparison(tmp_path):
    m = TinyREFe()
    its = items(7)
    ref = tmp_path / "ref_bad.npz"
    write_ref(ref, *make_ref(m, its, perturb=1e-6))                               # a reference that differs by 1e-6
    arms = {A.CONTROL: A.ScorerArm(A.CONTROL, m)}
    (verdict, _), _ = run(tmp_path, m, arms, ref=ref)
    assert verdict.startswith("ZZARMPASS_FAIL") and "logits" in verdict


def test_perturbed_control_scorer_fails_the_internal_control(tmp_path):
    """a control path whose scorer is not the model's own scorer cannot pass, even with a matching-looking reference"""
    m = TinyREFe()
    bad = A.ScorerArm(A.CONTROL, m)
    with torch.no_grad():
        bad.mods["score_head"].bias += 1e-4
    arms = {A.CONTROL: bad}
    (verdict, _), _ = run(tmp_path, m, arms)
    assert verdict.startswith("ZZARMPASS_FAIL internal control")


def test_proposal_mismatch_fails(tmp_path):
    m = TinyREFe()
    its = items(7)
    tk, pr, lg = make_ref(m, its)
    pr = pr.copy(); pr[3, 0, 0, 0] += 1e-6
    write_ref(tmp_path / "ref.npz", tk, pr, lg)
    (verdict, _), _ = run(tmp_path, m, {A.CONTROL: A.ScorerArm(A.CONTROL, m)}, ref=tmp_path / "ref.npz")
    assert verdict.startswith("ZZARMPASS_FAIL") and "proposals" in verdict


def test_resume_after_kill_gives_identical_output(tmp_path):
    m = TinyREFe()
    st_arm = scorer_perturbed(state(m))
    mk = lambda: {A.CONTROL: A.ScorerArm(A.CONTROL, m), "S1E": A.ScorerArm("S1E", m, st_arm)}

    class Kill(Exception):
        pass

    def die(t):
        if t == "tok005":
            raise Kill()
    with pytest.raises(Kill):
        run(tmp_path, m, mk(), name="r", fac_hook=die)                           # chunk=3: tokens 0-2 banked, 3-4 lost
    seen = []
    (verdict, _), st = run(tmp_path, m, mk(), name="r", ref=tmp_path / "ref.npz", fac_hook=seen.append)
    assert verdict == "ZZARMPASS_OK"
    assert seen[0] == "tok003" and "tok000" not in seen                          # restarted by token
    (v2, _), st2 = run(tmp_path, m, mk(), name="fresh", ref=tmp_path / "ref.npz")
    for n in (A.CONTROL, "S1E"):
        a, b = np.load(st.out_path(n)), np.load(st2.out_path(n))
        assert np.array_equal(a["logits"], b["logits"]) and np.array_equal(a["token"], b["token"])


def test_existing_output_with_other_tokens_or_sha_is_refused(tmp_path):
    m = TinyREFe()
    arms = {A.CONTROL: A.ScorerArm(A.CONTROL, m), "S1E": A.ScorerArm("S1E", m, state(m))}
    (v, _), st = run(tmp_path, m, arms)
    assert v == "ZZARMPASS_OK"
    toks = [t for t, _, _ in items(7)]
    with pytest.raises(A.ArmRefused):                                              # same out dir, different token set
        A.Store(str(tmp_path / "out"), toks[:5], "basesha", {"S1E": "sha_S1E"}, 3)
    with pytest.raises(A.ArmRefused):                                              # same tokens, different arm checkpoint
        A.Store(str(tmp_path / "out"), toks, "basesha", {"S1E": "OTHER"}, 3)
    A.Store(str(tmp_path / "out"), toks, "basesha", {"S1E": "sha_S1E"}, 3)         # identical run: accepted


def test_parse_arms():
    assert A.parse_arms(["C0=a/b.pt", "S1E=c.pt"]) == {"C0": "a/b.pt", "S1E": "c.pt"}
    for bad in (["C0=a", "C0=b"], [f"{A.CONTROL}=a"], ["noeq"]):
        with pytest.raises(SystemExit):
            A.parse_arms(bad)
