"""CPU tests for fit_arm_logits.py and make_arm_pack.py: a tiny stand-in for REFe (same scorer call sequence), synthetic data, no pod, no GPU.
Deliberate-regression cases must go RED.  One real-checkpoint load test (C0 on CPU) skips if the local checkpoints are absent.
Run: CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_arm_fit.py --basetemp F:/tanitad_scratch/p6armfit/pt"""
import copy
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_arm_logits as F  # noqa: E402
import make_arm_pack as MP  # noqa: E402

A = F.A
M, T, D, W = 64, 20, 3, 8


class Blk(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(W, W)
        self.kv = nn.Linear(W, W)

    def forward(self, s, ctx):
        return s + torch.softmax(self.q(s) @ self.kv(ctx).transpose(-1, -2), -1) @ ctx


class Tiny(nn.Module):
    def __init__(self, seed=0, bias=0.0):
        super().__init__()
        torch.manual_seed(seed)
        self.bias = bias
        self.backbone = nn.Linear(4, W)
        self.queries = nn.Parameter(torch.randn(M, W))
        self.traj_head = nn.Linear(W, T * D)
        self.score_q_mlp = nn.Sequential(nn.Linear(T * D, W), nn.GELU(), nn.Linear(W, W))
        self.score_dec = nn.ModuleList([Blk(), Blk()])
        self.score_head = nn.Linear(W, 6)

    def forward(self, img, ego=None, goal=None, calib=None):
        ctx = self.backbone(img)[:, None, :].repeat(1, 3, 1)
        q = self.queries[None] + ctx.mean(1, keepdim=True)
        traj = self.traj_head(q).view(img.shape[0], M, T, D) * torch.tensor([10.0, 10.0, 0.5])
        s = self.score_q_mlp(traj.detach().flatten(2))
        for blk in self.score_dec:
            s = blk(s, ctx)
        return traj, self.score_head(s) + self.bias


class FakeDS:
    def __init__(self, keys, imgs):
        self.rows = [{"log_name": k[0], "token": k[1], "step": k[2], "rank": k[3]} for k in keys]
        self.imgs = imgs

    def __getitem__(self, i):
        return (self.imgs[i],)


class Seam:
    def __init__(self, base_state, keys, imgs, bias=0.0, drop=0):
        self.base_state, self.bias, self.drop = base_state, bias, drop
        self.keys, self.imgs = keys, imgs

    def load_model(self):
        m = Tiny(bias=self.bias)
        m.load_state_dict(self.base_state)
        return m.eval()

    def load_dataset(self, keys):
        ks = self.keys[self.drop:]
        return FakeDS(ks, self.imgs[self.drop:])

    def forward(self, model):
        return lambda items: model(torch.stack([it[0] for it in items]))


N = 11


def make_world(tmp, scale=1.0):
    """base ckpt, three arm ckpts (scorer-only changes), a pack made the way the label job made it (rounded 5/6/6 decimals, batch 4 per group)"""
    torch.manual_seed(1)
    base = Tiny(seed=0)
    imgs = torch.randn(N, 4)
    keys = [(f"log{i // 4}", f"tok{i:03d}", 0, 0) for i in range(N)]
    is_ho = np.array([i < 6 for i in range(N)])
    bs = base.state_dict()
    paths = {}

    def save(name, st, meta=None):
        p = os.path.join(tmp, f"{name}.pt")
        torch.save({"format": "refe-full-v1", "model": st, "meta": meta or {}}, p)
        paths[name] = p

    save("base", bs)
    for j, n in enumerate(("C0", "C0r", "S1E")):
        st = copy.deepcopy(bs)
        g = torch.Generator().manual_seed(10 + j)
        for k in st:
            if A.is_scorer_key(k):
                st[k] = st[k] + 0.05 * torch.randn(st[k].shape, generator=g)
        save(n, st)
    # the pack: run the base planner in the label job's batches
    xy, yw, lg = np.zeros((N, M, T, 2), np.float32), np.zeros((N, M, T), np.float32), np.zeros((N, M, 6), np.float32)
    base.eval()
    for b in F.plan_batches(is_ho, 4):
        with torch.no_grad():
            tr, sc = base(imgs[b])
        tr, sc = tr.numpy(), sc.numpy()
        xy[b] = np.round(tr[..., :2], 5)
        yw[b] = np.round(tr[..., 2], 6)
        lg[b] = np.round(sc, 6)
    pack = dict(log=np.array([k[0] for k in keys]), token=np.array([k[1] for k in keys]), step=np.zeros(N, np.int64), rank=np.zeros(N, np.int64),
                traj_xy=xy, yaw=yw, logits=lg, is_heldout=is_ho, pdm=np.random.RandomState(0).rand(N, M, 7).astype(np.float32),
                has_pdm=np.ones(N, bool), pdm_names=np.array(list("abcdefg")), logit_space=np.array("raw"))
    pp = os.path.join(tmp, "fit_pack.npz")
    np.savez(pp, **pack)
    return dict(paths=paths, pack_path=pp, pack=pack, keys=keys, imgs=imgs, base_state=bs)


def argv(w, out, *extra, arms=("C0", "C0r", "S1E")):
    a = ["--base-ckpt", w["paths"]["base"], "--out", out, "--pack", w["pack_path"], "--device", "cpu", "--workers", "0", "--chunk", "4",
         "--expect-base-sha256", ""]
    for n in arms:
        a += ["--arm", f"{n}={w['paths'][n]}"]
    return a + list(extra)


def run(w, out, *extra, seam=None, arms=("C0", "C0r", "S1E")):
    seam = seam or Seam(w["base_state"], w["keys"], w["imgs"])
    return F.main(argv(w, out, *extra, arms=arms), seam=seam)


@pytest.fixture()
def world(tmp_path):
    return make_world(str(tmp_path)), str(tmp_path / "out")


# ----------------------------------------------------------------------------------------------------------- the pass
def test_happy_path(world, capsys):
    w, out = world
    assert run(w, out) == 0
    assert "ZZFITARM_OK" in capsys.readouterr().out
    ctl = np.load(os.path.join(out, "P6_control", "fit_logits.npz"))
    assert ctl["logits"].shape == (N, M, 6) and ctl["logits"].dtype == np.float32
    assert list(ctl["token"]) == list(w["pack"]["token"])                 # the pack's set order
    assert np.abs(ctl["logits"] - w["pack"]["logits"]).max() <= 1e-6      # control reproduces the pack to its rounding
    for n in ("C0", "C0r", "S1E"):
        d = np.load(os.path.join(out, n, "fit_logits.npz"))
        assert d["logits"].shape == (N, M, 6)
        assert np.abs(d["logits"] - ctl["logits"]).max() > 1e-3           # an arm is not the control
        assert str(d["arm"]) == n
    assert len({tuple(np.load(os.path.join(out, n, "fit_logits.npz"))["logits"].ravel()[:50]) for n in ("C0", "C0r", "S1E")}) == 3
    import json
    rep = json.load(open(os.path.join(out, "report.json")))
    assert rep["controls"]["max_internal_diff"] == 0.0 and rep["verdict"] == "ZZFITARM_OK" and rep["n_sets"] == N


def test_arm_logits_equal_the_arm_run_alone(world):
    """the shared-forward trick: arm logits equal a model carrying that arm's scorer, run end to end"""
    w, out = world
    assert run(w, out) == 0
    st = torch.load(w["paths"]["C0"], weights_only=False)["model"]
    m = Tiny()
    m.load_state_dict(st)
    m.eval()
    with torch.no_grad():
        _, sc = m(w["imgs"][:4])
    got = np.load(os.path.join(out, "C0", "fit_logits.npz"))["logits"][:4]
    assert np.abs(got - sc.numpy()).max() < 1e-5


def test_resume_reuses_banked_chunks(world):
    w, out = world
    assert run(w, out) == 0
    first = np.load(os.path.join(out, "C0", "fit_logits.npz"))["logits"].copy()
    parts = sorted(os.listdir(os.path.join(out, "_progress")))
    os.remove(os.path.join(out, "_progress", [p for p in parts if p.startswith("part_")][-1]))
    assert run(w, out) == 0
    assert np.array_equal(first, np.load(os.path.join(out, "C0", "fit_logits.npz"))["logits"])


def test_limit_is_partial(world):
    w, out = world
    assert run(w, out, "--limit", "5") == 0
    assert np.load(os.path.join(out, "C0", "fit_logits.npz"))["logits"].shape[0] == 5


def test_RED_arm_with_moved_trunk_key(world, capsys):
    w, out = world
    st = torch.load(w["paths"]["C0"], weights_only=False)
    st["model"]["backbone.weight"] = st["model"]["backbone.weight"] + 1e-6
    torch.save(st, w["paths"]["C0"])
    assert run(w, out) == 1
    o = capsys.readouterr().out
    assert "ZZFITARM_FAIL" in o and "backbone.weight" in o and "ZZFITARM_OK" not in o


def test_RED_control_mismatch_above_tolerance(world, capsys):
    w, out = world
    p = dict(np.load(w["pack_path"]))
    p["logits"] = p["logits"] + 2e-3
    np.savez(w["pack_path"], **p)
    assert run(w, out) == 1
    assert "ZZFITARM_FAIL reference control: control logits" in capsys.readouterr().out


def test_control_mismatch_below_tolerance_passes_and_flag_moves_it(world, capsys):
    w, out = world
    p = dict(np.load(w["pack_path"]))
    p["logits"] = p["logits"] + 5e-4
    np.savez(w["pack_path"], **p)
    assert run(w, out) == 0
    assert run(w, out + "2", "--ref-tol", "1e-4") == 1


def test_RED_proposals_differ_from_pack(world, capsys):
    w, out = world
    p = dict(np.load(w["pack_path"]))
    p["traj_xy"] = p["traj_xy"].copy()
    p["traj_xy"][3, 5, 7, 1] += 1e-3
    np.savez(w["pack_path"], **p)
    assert run(w, out) == 1
    assert "proposal xy" in capsys.readouterr().out


def test_RED_yaw_differs_from_pack(world, capsys):
    w, out = world
    p = dict(np.load(w["pack_path"]))
    p["yaw"] = p["yaw"].copy()
    p["yaw"][2, 0, 0] += 1e-4
    np.savez(w["pack_path"], **p)
    assert run(w, out) == 1
    assert "proposal yaw" in capsys.readouterr().out


def test_RED_internal_control_not_exact(world, capsys):
    """a planner whose own score is not what the hook path reproduces (here: a bias the scorer modules do not carry)"""
    w, out = world
    seam = Seam(w["base_state"], w["keys"], w["imgs"], bias=1e-3)
    assert run(w, out, seam=seam) == 1
    assert "internal control" in capsys.readouterr().out


def test_RED_pack_set_without_dataset_row(world, capsys):
    w, out = world
    seam = Seam(w["base_state"], w["keys"], w["imgs"], drop=1)
    assert run(w, out, seam=seam) == 1
    assert "no dataset row" in capsys.readouterr().out


def test_RED_wrong_base_sha(world, capsys):
    w, out = world
    a = argv(w, out)
    a[a.index("--expect-base-sha256") + 1] = "0" * 64
    assert F.main(a, seam=Seam(w["base_state"], w["keys"], w["imgs"])) == 1
    assert "sha256" in capsys.readouterr().out


def test_RED_different_run_in_same_out(world, capsys):
    w, out = world
    assert run(w, out) == 0
    assert run(w, out, "--batch-size", "2") == 1
    assert "different run" in capsys.readouterr().out


def test_batch_plan_groups_and_limit():
    ho = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    b = F.plan_batches(ho, 4)
    assert b == [[0, 1, 2, 3], [4], [5, 6, 7, 8], [9, 10]]               # a batch never spans the two dump runs
    assert F.plan_batches(ho, 4, 6) == [[0, 1, 2, 3], [4], [5]]
    assert [len(c) for c in F.chunk_batches(b, 8, 4)] == [2, 2]


# ----------------------------------------------------------------------------------------------------------- make_arm_pack
def pack8(tmp, n=6):
    r = np.random.RandomState(0)
    d = dict(log=np.array([f"l{i}" for i in range(n)]), token=np.array([f"t{i}" for i in range(n)]), step=np.zeros(n, np.int64), rank=np.zeros(n, np.int64),
             traj_xy=r.rand(n, 4, 8, 2).astype(np.float32), yaw=r.rand(n, 4, 8).astype(np.float32), logits=r.rand(n, 4, 6).astype(np.float32),
             pdm=r.rand(n, 4, 7).astype(np.float32), has_pdm=np.ones(n, bool), is_heldout=np.zeros(n, bool), logit_space=np.array("x"))
    p = os.path.join(tmp, "p8.npz")
    np.savez(p, **d)
    return p, d


def armfile(tmp, d, order, name="a.npz", **kw):
    r = np.random.RandomState(5)
    x = dict(log=d["log"][order], token=d["token"][order], step=d["step"][order], rank=d["rank"][order], logits=r.rand(len(order), 4, 6).astype(np.float32))
    x.update(kw)
    p = os.path.join(tmp, name)
    np.savez(p, **x)
    return p, x


def test_make_pack_joins_by_key_and_replaces_only_logits(tmp_path):
    tmp = str(tmp_path)
    p8, d = pack8(tmp)
    order = np.array([3, 0, 5, 1, 4, 2])                                   # arm file in another order
    ap, x = armfile(tmp, d, order)
    out = os.path.join(tmp, "o.npz")
    sha, _ = MP.make(p8, ap, out)
    assert sha == MP.sha256_file(out)
    with np.load(out) as z:
        R = {k: z[k] for k in z.files}
    inv = np.argsort(order)
    assert np.array_equal(R["logits"], x["logits"][inv])
    for k in d:
        if k != "logits":
            assert np.array_equal(R[k], d[k]) and R[k].dtype == d[k].dtype
    assert MP.main(["--pack", p8, "--arm-logits", ap, "--out", out]) == 0


def test_RED_make_pack_missing_key(tmp_path):
    tmp = str(tmp_path)
    p8, d = pack8(tmp)
    ap, _ = armfile(tmp, d, np.array([0, 1, 2, 3, 4]))
    with pytest.raises(MP.PackRefused, match="missing"):
        MP.make(p8, ap, os.path.join(tmp, "o.npz"))
    assert MP.main(["--pack", p8, "--arm-logits", ap, "--out", os.path.join(tmp, "o.npz")]) == 1
    assert not os.path.exists(os.path.join(tmp, "o.npz"))


def test_RED_make_pack_extra_key(tmp_path):
    tmp = str(tmp_path)
    p8, d = pack8(tmp)
    ap, x = armfile(tmp, d, np.arange(6))
    x["token"] = np.array(list(x["token"][:5]) + ["zzz"])
    x["logits"] = x["logits"]
    np.savez(ap, **x)
    # one key swapped for a foreign one: a key is both missing and extra
    with pytest.raises(MP.PackRefused):
        MP.make(p8, ap, os.path.join(tmp, "o.npz"))
    d2 = {k: np.concatenate([v, v[:1]]) if k in ("log", "token", "step", "rank", "logits") else v for k, v in x.items()}
    d2["token"][-1] = "extra"
    np.savez(ap, **d2)
    with pytest.raises(MP.PackRefused, match="not in the pack|extra|missing"):
        MP.make(p8, ap, os.path.join(tmp, "o.npz"))


def test_RED_make_pack_changed_non_logit_array(tmp_path, monkeypatch):
    """the written file is verified by content: if any other array changed on the way (here forced by wrapping np.savez), it refuses"""
    tmp = str(tmp_path)
    p8, d = pack8(tmp)
    ap, _ = armfile(tmp, d, np.arange(6))
    real = np.savez

    def evil(path, **kw):
        kw = dict(kw)
        kw["pdm"] = kw["pdm"] + 1e-7
        return real(path, **kw)
    monkeypatch.setattr(MP.np, "savez", evil)
    with pytest.raises(MP.PackRefused, match="pdm"):
        MP.make(p8, ap, os.path.join(tmp, "o.npz"))
    assert not os.path.exists(os.path.join(tmp, "o.npz"))


def test_RED_make_pack_wrong_shape_and_nonfinite_and_pack_sha(tmp_path):
    tmp = str(tmp_path)
    p8, d = pack8(tmp)
    ap, x = armfile(tmp, d, np.arange(6), pack_sha256=np.array("abc"))
    with pytest.raises(MP.PackRefused, match="pack sha256"):
        MP.make(p8, ap, os.path.join(tmp, "o.npz"), expect_pack_sha="def")
    x["logits"] = x["logits"].copy()
    x["logits"][0, 0, 0] = np.nan
    np.savez(ap, **x)
    with pytest.raises(MP.PackRefused, match="finite"):
        MP.make(p8, ap, os.path.join(tmp, "o.npz"))
    x["logits"] = np.zeros((6, 4, 5), np.float32)
    np.savez(ap, **x)
    with pytest.raises(MP.PackRefused, match="shape"):
        MP.make(p8, ap, os.path.join(tmp, "o.npz"))


# ----------------------------------------------------------------------------------------------------------- the real C0 checkpoint, CPU
REAL_BASE = "F:/TanitAD_p6read_work/ckpt/run/model_final.pt"
REAL_C0 = "F:/tanitad_scratch/p6arms/c0/model_sft_B.pt"
REFE = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"


@pytest.mark.skipif(not (os.path.exists(REAL_BASE) and os.path.exists(REAL_C0) and os.path.isdir(REFE)), reason="local P6/C0 checkpoints absent")
def test_real_c0_checkpoint_loads_and_scores_random_tensors():
    sys.path.insert(0, REFE)
    import model as RM
    base_state, base_meta = A.load_ckpt(REAL_BASE)
    arm_state, arm_meta = A.load_ckpt(REAL_C0)
    A.verify_arm("C0", base_state, arm_state, arm_meta)                  # non-scorer keys bit-equal in the REAL pair
    scorer_changed = [k for k in base_state if A.is_scorer_key(k) and not torch.equal(base_state[k], arm_state[k])]
    assert scorer_changed, "C0 should differ from P6 in the scorer"
    cfg = RM.REFeConfig.for_backbone("vitl16")
    cfg.n_score_components = 6
    sdd = cfg.dec_depth if cfg.score_dec_depth is None else cfg.score_dec_depth

    class Holder(nn.Module):                                              # the three scorer modules, built as REFe.__init__ builds them
        def __init__(self):
            super().__init__()
            self.score_dec = nn.ModuleList([RM.CrossBlock(cfg.dec_width, cfg.dec_heads, cfg.dec_width) for _ in range(sdd)])
            self.score_q_mlp = nn.Sequential(nn.Linear(cfg.horizon_steps * cfg.traj_dim, cfg.dec_width), nn.GELU(), nn.Linear(cfg.dec_width, cfg.dec_width))
            self.score_head = nn.Linear(cfg.dec_width, cfg.n_score_components)
    h = Holder()
    ctl = A.ScorerArm("P6_control", h, {k: v for k, v in base_state.items() if A.is_scorer_key(k)})
    arm = A.ScorerArm("C0", h, {k: v for k, v in arm_state.items() if A.is_scorer_key(k)})
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, cfg.n_proposals, cfg.horizon_steps * cfg.traj_dim, generator=g)
    ctx = torch.randn(2, 12, cfg.dec_width, generator=g)
    a, c = arm(x, ctx), ctl(x, ctx)
    assert a.shape == (2, 64, 6) and c.shape == (2, 64, 6)
    assert torch.isfinite(a).all() and torch.isfinite(c).all()
    assert (a - c).abs().max() > 1e-4                                     # C0's scorer is not P6's
    assert "score_head.weight" in arm_state and arm_state["score_head.weight"].shape[0] == 6
