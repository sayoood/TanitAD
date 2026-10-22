"""PREREG_V8F_GRID: the stack's scene input at 8x20 tokens per frame instead of 4x10 (`--grid pool8x20`, ONE lever).

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): the 8x20 grid has existed in the trainer since v8F-mini's first
commit (`--grid`, `n_tok` 160) and in the cache since the encode, but nothing pinned that the flag REACHES the features the model trains on, nor that the eval
of the resulting checkpoint reads the same grid. Pinned here, hop by hop, as identities on a synthetic world whose two grids DIFFER:

  TRAIN  argv --grid pool8x20 -> make_config (n_tok 160, v8f_mini_train.py:160) -> declared_vs_built (the gate, :180) -> ClipStore(grid=...) (:337)
         -> V8FMiniWindows.item -> collate -> the `feats` the model's loss receives: EQUAL to the clip's `pool8x20` array, exactly (and the 4x10 control
         equals the `pool4x10` array);
         -> the BUILT parameter table: pos [160, d], the only per-token parameter (+120 d parameters, nothing else moves);
         -> a non-zero gradient on EVERY one of the 160 token-position rows.
  EVAL   the checkpoint the trainer saves carries `grid: pool8x20` and cfg.n_tok 160; v8f_mini_eval.py:268 reads the grid FROM the checkpoint (default
         pool4x10) and the model scores the cache's `pool8x20` array. A checkpoint whose n_tok disagrees with the features it is given is REFUSED: today the
         model's `adapter(feats) + pos` broadcast raises (MEASURED: "The size of tensor a (40) must match the size of tensor b (160)"), the eval writes no
         file, and no run is silently scored.

Each identity has a source mutation that turns it red: `raw/2026-10-22-grid/mutate_grid.py` (outputs beside it).
Everything that needs the v8F-mini research package (the reader, the queue stages, the chain) is below the reachability block and is SKIPPED, not
failing, in a checkout that has only `stack/` (the lander's probe).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_eval as EV  # noqa: E402

PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
HOLD = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_T = _load("test_v8f_mini_for_grid", Path(__file__).resolve().parent / "test_v8f_mini.py")
TR = _T.TR
torch.set_num_threads(1)
N_TOK = {"pool4x10": 40, "pool8x20": 160}
# the banked M2b (4x10) parameter counts, MEASURED: raw/full/runs/M2b_s0/summary.json (levers.params, n_params_trainable)
BANKED_M2B_PARAMS = {"adapter": 275712, "cond": 25088, "S": 1321737, "T": 1463868, "O": 2120216, "WM": 2007874}
BANKED_M2B_TOTAL = 7214495


# ---- a synthetic world whose two grids DIFFER ---------------------------------------------------------------------------------------------------
def two_grid_world(root, **kw):
    """`make_synthetic_world` + a `pool8x20` array [T, 160, 1024] in every clip, drawn independently of `pool4x10` (so a loader that returns the wrong
    grid is visible as a different array, not just a different shape). 4 train + 2 eval clips of 60 frames (the shape the trainer asks for)."""
    kw.update(n_train=4, n_eval=2, T=60)
    w = _ORIG_WORLD(root, **kw)
    rng = np.random.default_rng(99)
    for split in ("train", "eval"):
        for p in sorted((Path(root) / split / "clips").glob("*.pt")):
            o = torch.load(p, map_location="cpu", weights_only=True)
            T = o["pool4x10"].shape[0]
            o["pool8x20"] = torch.as_tensor(rng.normal(0, 1, (T, 160, D.D_FEAT)).astype(np.float32)).to(torch.float8_e4m3fn)
            torch.save(o, p)
    return w


_ORIG_WORLD = D.make_synthetic_world


@pytest.fixture(scope="module")
def world2(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn2g")
    w = two_grid_world(root)
    return {"root": root, **w}


def _clip_array(root, split, clip, grid):
    o = torch.load(Path(root) / split / "clips" / f"{clip}.pt", map_location="cpu", weights_only=True)
    return o[grid].to(torch.float16).numpy()


def _expected_feats(root, split, J, b, grid, hmax):
    """The feats the batch must hold: clip `ci`'s `grid` array at frames t-hmax+1..t, clamped at 0 (V8FMiniWindows.item's own indexing, restated)."""
    out = []
    for ci, t in zip(b["ci"].tolist(), b["t"].tolist()):
        arr = _clip_array(root, split, str(J["clip"][ci]), grid)
        out.append(arr[np.clip(np.arange(t - hmax + 1, t + 1), 0, len(arr) - 1)])
    return np.stack(out)


# ---- 1. argv -> config -> the gate -> the built model ----------------------------------------------------------------------------------------
def test_grid_flag_reaches_n_tok_the_gate_and_the_built_position_table(world2):
    J = D.load_join(Path(world2["root"]) / "join_train.npz")
    a8 = _T._args(["--grid", "pool8x20"])
    a4 = _T._args([])
    c8, c4 = TR.make_config(a8), TR.make_config(a4)
    assert (c4.n_tok, c8.n_tok) == (40, 160), "--grid reaches the config: 40 -> 160 (v8f_mini_train.py:160)"
    assert {k: v for k, v in c8.__dict__.items() if k != "n_tok"} == {k: v for k, v in c4.__dict__.items() if k != "n_tok"}, "ONE lever: nothing else in the config moves"
    assert a4.grid == "pool4x10" and a8.grid == "pool8x20" and TR.FLAG_CHECKS["grid"] == "cfg:grid"
    # the gate passes when built as declared, with the position table the declared size
    cfg, model, opt, info = _T._built(a8, J)
    assert model.pos.shape == (160, 32) and TR.declared_vs_built(a8, cfg, model, opt, info) == []
    cfg4, model4, opt4, info4 = _T._built(a4, J)
    assert model4.pos.shape == (40, 32) and TR.declared_vs_built(a4, cfg4, model4, opt4, info4) == []
    # ... and REFUSES each half of a lie (declared 8x20, built 4x10, and the reverse)
    cfg, model, opt, info = _T._built(a8, J, n_tok=40)
    assert any("grid pool8x20 declared but n_tok built 40" in m for m in TR.declared_vs_built(a8, cfg, model, opt, info))
    cfg, model, opt, info = _T._built(a4, J, n_tok=160)
    assert any("grid pool4x10 declared but n_tok built 160" in m for m in TR.declared_vs_built(a4, cfg, model, opt, info))


def test_grid_changes_exactly_the_position_table_at_the_real_size():
    """REAL SIZE (d 256, M2b): the parameter count moves by exactly (160 - 40) x d = 30,720 and only in the adapter group (pos lives there). The 4x10 counts equal
    the banked M2b's, MEASURED in raw/full/runs/M2b_s0/summary.json -- so 'the parameter count may change' (PREREG) is a known number, and the reader refuses any
    other delta as a second lever."""
    built = {}
    for g in ("pool4x10", "pool8x20"):
        a = TR.build_parser().parse_args(["--out", "x", "--arm", "M2b", "--grid", g, "--kin-mode", "yaw"])
        cfg = TR.make_config(a)
        assert cfg.n_tok == N_TOK[g] and cfg.d == 256
        built[g] = M.V8FMini(cfg, M.P10Prior.synthetic(0), None, None).built_levers()
    assert built["pool4x10"]["params"] == BANKED_M2B_PARAMS and built["pool4x10"]["n_params_trainable"] == BANKED_M2B_TOTAL
    delta = 120 * 256
    assert built["pool8x20"]["n_params_trainable"] == BANKED_M2B_TOTAL + delta == 7245215
    assert built["pool8x20"]["params"] == {**BANKED_M2B_PARAMS, "adapter": BANKED_M2B_PARAMS["adapter"] + delta}
    assert built["pool8x20"]["params"]["adapter"] == 306432


# ---- 2. the data hop: ClipStore(grid=...) -> item -> collate -------------------------------------------------------------------------------
@pytest.mark.parametrize("kin_mode", ["yaw", "steer"])
@pytest.mark.parametrize("grid", ["pool8x20", "pool4x10"])
def test_clipstore_grid_batch_feats_equal_the_clips_array_exactly(world2, grid, kin_mode):
    """Both ClipStore branches (the committed yaw mode, and the trainer's other --kin-mode) hand the model the requested grid's array."""
    root = Path(world2["root"])
    J = D.load_join(root / "join_train.npz")
    tracks = D.load_native_tracks(root / "native_tracks.pt") if kin_mode == "yaw" else None
    store = D.ClipStore(root, "train", lru=4, grid=grid, kin_mode=kin_mode, tracks=tracks)
    wins = D.train_windows(store, J, world2["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=10)
    rng = np.random.default_rng(3)
    items = [ds.item(*wins[i]) for i in rng.choice(len(wins), 12, replace=False)]
    b = D.collate(items)
    assert tuple(b["feats"].shape[1:]) == (10, N_TOK[grid], D.D_FEAT)
    want = _expected_feats(root, "train", J, b, grid, 10)
    assert np.array_equal(b["feats"].numpy(), want), f"the batch's feats are not the clip's {grid} array"
    other = "pool4x10" if grid == "pool8x20" else "pool8x20"
    a0, a1 = _clip_array(root, "train", world2["names"]["train"][0], grid), _clip_array(root, "train", world2["names"]["train"][0], other)
    assert a0.shape != a1.shape, "the world's two grids differ (shape), so the identity above cannot be met by the wrong grid"
    assert not np.array_equal(a0[:, :40], a1[:, :40]), "... and in content, on the first 40 tokens"


# ---- 3. the trainer, end to end: argv -> the batch the model sees ----------------------------------------------------------------------------
def _train(out, extra, monkeypatch, steps=4):
    """A real `TR.main` run on the two-grid synthetic world, with a spy on `V8FMini.loss` recording what the model RECEIVES."""
    cap = []
    orig_loss = M.V8FMini.loss

    def spy(self, b, *a, **k):
        cap.append({"feats": b["feats"].clone(), "ci": b["ci"].clone(), "t": b["t"].clone()})
        return orig_loss(self, b, *a, **k)
    monkeypatch.setattr(D, "make_synthetic_world", two_grid_world)
    monkeypatch.setattr(M.V8FMini, "loss", spy)
    common = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--steps", str(steps), "--log-every", "2", "--warmup", "2",
              "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5"]
    try:
        assert TR.main(["--out", str(out), *common, *extra]) == 0
    finally:
        monkeypatch.setattr(M.V8FMini, "loss", orig_loss)
        monkeypatch.setattr(D, "make_synthetic_world", _ORIG_WORLD)
    root = Path(out) / "_synthetic"
    ck = torch.load(Path(out) / "ckpt_last.pt", map_location="cpu", weights_only=False)
    return {"cap": cap, "root": root, "ck": ck, "sc": json.load(open(Path(out) / "selfcheck.json", encoding="utf-8")),
            "sm": json.load(open(Path(out) / "summary.json", encoding="utf-8")), "J": D.load_join(root / "join_train.npz")}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        base = tmp_path_factory.mktemp("grid_runs")
        out = {g: _train(base / g, ["--grid", g], mp) for g in ("pool8x20", "pool4x10")}
    finally:
        mp.undo()
    return out


@pytest.mark.parametrize("grid", ["pool8x20", "pool4x10"])
def test_trainer_argv_grid_is_the_array_the_model_trains_on(runs, grid):
    r = runs[grid]
    assert len(r["cap"]) >= 4, "the live check (3 batches) and the training loop both reached the loss"
    for c in r["cap"]:
        assert c["feats"].shape[2] == N_TOK[grid], f"the model received {c['feats'].shape[2]} tokens per frame, --grid {grid} declares {N_TOK[grid]}"
        want = _expected_feats(r["root"], "train", r["J"], c, grid, hmax=10)
        assert np.array_equal(c["feats"].numpy(), want), f"--grid {grid}: a batch the model received is not the clip's {grid} array (exact)"
    assert r["sc"]["declared_vs_built"] == "PASS" and r["sm"]["done"] is True


def test_trainer_8x20_run_records_the_grid_and_the_built_table_and_the_parameter_delta(runs):
    r8, r4 = runs["pool8x20"], runs["pool4x10"]
    ck = r8["ck"]
    assert ck["grid"] == "pool8x20" and ck["cfg"]["n_tok"] == 160 and tuple(ck["model"]["pos"].shape) == (160, 32), "the checkpoint carries the grid"
    assert runs["pool4x10"]["ck"]["grid"] == "pool4x10" and runs["pool4x10"]["ck"]["cfg"]["n_tok"] == 40
    lv8, lv4 = r8["sm"]["levers"], r4["sm"]["levers"]
    drop = lambda d: {k: v for k, v in d.items() if k != "residual_zero_init"}      # written BEFORE training (True) and AFTER it (False): the only expected difference
    assert drop(r8["sc"]["levers"]) == drop(lv8), "summary.json and selfcheck.json record the same built levers"
    assert lv8["n_params_trainable"] - lv4["n_params_trainable"] == 120 * 32 == lv8["params"]["adapter"] - lv4["params"]["adapter"], "only the position table moved"
    for k in ("cond", "S", "T", "O", "WM"):
        assert lv8["params"].get(k) == lv4["params"].get(k), k
    # the pipeline knob that follows the grid (REPORTED by the reader, not a lever): the LRU is sized from the token count
    assert r8["sc"]["info"]["lru"] < r4["sc"]["info"]["lru"]


def test_every_one_of_the_160_position_rows_gets_a_gradient_and_the_prior_control_reads_exactly_zero(world2):
    """Reachability of the NEW parameters: pos rows 40..159 exist only at 8x20. A gradient on the whole table could hide dead rows, so read it row by row; the
    frozen P10 prior (the gate's control) must read EXACTLY 0.0 -- without the control a zero reading would be indistinguishable from a broken probe."""
    root = Path(world2["root"])
    J = D.load_join(root / "join_train.npz")
    store = D.ClipStore(root, "train", lru=4, grid="pool8x20", kin_mode="yaw", tracks=D.load_native_tracks(root / "native_tracks.pt"))
    wins = D.train_windows(store, J, world2["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=10)
    b = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in range(0, 80, 5)])
    a = _T._args(["--grid", "pool8x20"])
    cfg, model, opt, info = _T._built(a, J)
    model.train()
    out = model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(0))
    out["terms"]["total"].backward()
    row = model.pos.grad.abs().sum(1)
    assert row.shape == (160,) and bool((row > 0).all()), f"token-position rows without a gradient: {(row == 0).nonzero().flatten().tolist()}"
    assert float(sum(p.grad.abs().sum() for p in model.prior.parameters() if p.grad is not None)) == 0.0
    live = TR.live_check(model.cpu(), b, cfg)                       # the trainer's own G-LIVE on the 8x20 batch: every group alive, prior control 0.0
    assert live["pass"] and live["control_prior_grad_sum"] == 0.0 and live["groups_grad"]["adapter"] > 0.0


# ---- 4. the eval hop: the checkpoint names the grid; the eval reads it; a disagreement is refused ----------------------------------------------
def _eval_args(root, ck, tag, out):
    root = Path(root)
    return ["run", "--ckpt", str(ck), "--tag", tag, "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"),
            "--eval-kin", str(root / "eval_kin.npz"), "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(out), "--n-boot", "20",
            "--batch", "16", "--lru", "4", "--tracks", str(root / "native_tracks.pt")]


def _spy_eval(monkeypatch):
    cap = {"grids": [], "feats": []}
    ci0 = D.ClipStore.__init__

    def init(self, cache_dir, split, lru=64, grid="pool4x10", **k):
        cap["grids"].append(grid)
        return ci0(self, cache_dir, split, lru=lru, grid=grid, **k)
    fi0 = M.V8FMini.forward_infer

    def fi(self, b, *a, **k):
        cap["feats"].append({"feats": b["feats"].clone(), "ci": b["ci"].clone(), "t": b["t"].clone()})
        return fi0(self, b, *a, **k)
    monkeypatch.setattr(D.ClipStore, "__init__", init)
    monkeypatch.setattr(M.V8FMini, "forward_infer", fi)
    return cap


def test_eval_of_an_8x20_checkpoint_reads_pool8x20_from_the_checkpoint(runs, tmp_path, monkeypatch):
    r = runs["pool8x20"]
    root = r["root"]
    cap = _spy_eval(monkeypatch)
    rc = EV.main(_eval_args(root, root.parent / "ckpt_last.pt", "g8", tmp_path))
    assert rc == 0 and (tmp_path / "eval_g8.json").exists() and (tmp_path / "eval_g8.npz").exists()
    assert cap["grids"] == ["pool8x20"], f"the eval built its ClipStore with {cap['grids']}; the checkpoint says pool8x20 (v8f_mini_eval.py:268)"
    EJ = D.load_join(root / "join_eval.npz")
    assert cap["feats"], "the model was run"
    for c in cap["feats"]:
        assert c["feats"].shape[2] == 160
        want = _expected_feats(root, "eval", EJ, c, "pool8x20", hmax=r["ck"]["cfg"]["hmax"])
        assert np.array_equal(c["feats"].numpy(), want), "a batch the eval scored is not the eval clip's pool8x20 array (exact)"
    z = np.load(tmp_path / "eval_g8.npz")
    assert np.isfinite(z["pred"]).all() and z["pred"].shape[1:] == (30, 2)


def test_eval_of_a_4x10_checkpoint_still_reads_pool4x10(runs, tmp_path, monkeypatch):
    r = runs["pool4x10"]
    cap = _spy_eval(monkeypatch)
    assert EV.main(_eval_args(r["root"], r["root"].parent / "ckpt_last.pt", "g4", tmp_path)) == 0
    assert cap["grids"] == ["pool4x10"] and all(c["feats"].shape[2] == 40 for c in cap["feats"])


@pytest.mark.parametrize("variant", ["no_grid_key_160", "grid_key_4x10_160", "grid_key_8x20_40"])
def test_eval_refuses_a_checkpoint_whose_n_tok_disagrees_with_its_features(runs, tmp_path, variant):
    """MEASURED today: the model's `adapter(feats) + pos` broadcast raises RuntimeError on any n_tok / feature-count disagreement, so nothing is scored silently.
    The three ways a disagreement can arise: a checkpoint with NO `grid` key (the eval's DEFAULT pool4x10 meets n_tok 160), a wrong key, and the reverse."""
    r8, r4 = runs["pool8x20"], runs["pool4x10"]
    ck = dict(r8["ck"] if variant != "grid_key_8x20_40" else r4["ck"])
    if variant == "no_grid_key_160":
        ck.pop("grid")
    elif variant == "grid_key_4x10_160":
        ck["grid"] = "pool4x10"
    else:
        ck["grid"] = "pool8x20"
    path = tmp_path / "bad.pt"
    torch.save(ck, path)
    root = (r8 if variant != "grid_key_8x20_40" else r4)["root"]
    out = tmp_path / "out"
    with pytest.raises((RuntimeError, SystemExit, KeyError)):
        EV.main(_eval_args(root, path, "bad", out))
    assert not list(out.glob("eval_bad*")) if out.exists() else True, "a refused checkpoint must leave no eval file behind"


def test_eval_does_not_score_a_cache_that_lacks_the_checkpoints_grid(runs, tmp_path):
    r8 = runs["pool8x20"]
    root = tmp_path / "cache4only"
    shutil.copytree(r8["root"], root)
    for p in (root / "eval" / "clips").glob("*.pt"):                  # an eval cache that was never encoded at 8x20
        o = torch.load(p, map_location="cpu", weights_only=True)
        o.pop("pool8x20")
        torch.save(o, p)
    with pytest.raises(KeyError, match="pool8x20"):
        EV.main(_eval_args(root, r8["root"].parent / "ckpt_last.pt", "nocache", tmp_path / "out"))


# =================================================================================================================================================
# THE READER (code/v8f_grid_read.py), THE CACHE CHECK, THE QUEUE STAGES AND THE CHAIN need the v8F-mini research package: skipped, not failing, in a checkout
# that has only `stack/`. The reader is tested on a synthetic straight-road world whose answers are analytic (test_v8f_dk.World), real files on disk and the
# real `main`; the real-data proofs (raw/2026-10-22-grid/real_proofs.py) run it unpatched on the banked M2b evals.
# =================================================================================================================================================
CODE = PKG / "code"
READ_P = CODE / "v8f_grid_read.py"
CACHE_P = CODE / "check_grid_cache.py"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
CHAIN = CODE / "box2_ops" / "chain_v8f_grid.sh"
needs_pkg = pytest.mark.skipif(not READ_P.exists(), reason="the v8F-mini package is not in this checkout")
R = _load("v8f_grid_read_under_test", READ_P) if READ_P.exists() else None            # importable without the package (the lander's stack-only probe)
_DK = _load("test_v8f_dk_for_grid", Path(__file__).resolve().parent / "test_v8f_dk.py") if READ_P.exists() else None
C0, NC = M.N_GOAL + M.N_LAT + M.N_LON, len(M.CONS_NAMES)
LATCONS = [C0 + i for i in range(12)]                                                  # CODE_GROUPS["latcons"] (pinned against the model below)
ADAPTER_B, TOTAL_B, D_POS = 275712, 7214495, 120 * 256
B_ARGV = ["--arm", "M2b", "--seed", "{s}", "--grid", "pool4x10", "--out", "D:\\runs\\{dir}", "--d", "256", "--kin-mode", "yaw", "--exclude", "540:600",
          "--batch", "64", "--steps-mult", "4", "--device", "cuda"]
PGT_FLAGS = ["--p-gt-start", "0.5", "--p-gt-end", "0.0"]
BASES = ["M2b", "M2b-pgt"]


def _sched(step, total, p0, p1):                                                       # independent of the reader's and the model's
    return p0 + (p1 - p0) * min(max(step / total, 0.0), 1.0)


def _ej(path, fn):
    p = Path(path)
    j = json.loads(p.read_text(encoding="utf-8"))
    r = fn(j)
    p.write_text(json.dumps(j if r is None else r), encoding="utf-8")


def _gworld(d, base="M2b", seeds=(0, 1), *, d_ade=-0.7, err_b=0.6, err_g=0.5, err_g1=None, sp_b=11.0, sp_g=10.4, gap_b=0.65, off_b=1.0, lam_b=0.27, lam_g=0.10, steps=19164,
            pgt_verdict=None, pgt_s0=None, states=None, adapter_g=None):
    """Synthetic evals + runs for B (`base`, both seeds, always) and G = B + --grid pool8x20 (`seeds`), plus a stand-in PGT read. Knobs: `d_ade` = ADE6(G) - ADE6(B)
    (the lateral offset beyond 2 s is SOLVED for it, so the K = 10 lead metrics do not move with it), `err_*` T's present-slot error (sd of the noise on every present
    constraint slot, latcons included, the SAME draws for both arms: a paired design), `sp_*` plan speed (-> the low-TTC tail), `gap_b` B's seed-1 offset (-> F)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    w = _DK.World(d, states=states)
    N, eid, gt = w.n, w.eid, w.gt
    lead = np.array([s == "LEAD" for s in w.states])
    rng = np.random.default_rng(5)
    g = np.zeros((N, M.CODE_DIM))
    g[:, :C0] = rng.normal(0, 1, (N, C0))
    present = rng.random((N, NC)) < 0.45
    z_all = np.random.default_rng(31).normal(0, 1, (N, NC))
    g[:, C0:] = np.where(present, rng.normal(0, 1, (N, NC)), 0.0)
    fb = np.zeros(N, bool)
    fb[:5] = True
    ev, ru = d / "eval", d / "runs"
    ev.mkdir(exist_ok=True)
    ru.mkdir(exist_ok=True)

    def ade(p):
        return np.linalg.norm(p.astype(np.float32).astype(np.float64) - gt, axis=-1).mean(1)

    def std_plan(sp, off, nz):
        p = _DK.path_at(np.full(N, sp), N)
        p[:, 10:, 1] += (off * (1.0 + 0.05 * nz))[:, None]
        return p

    def solve_off(sp, nz, target):
        lo, hi = 0.0, 40.0
        if ade(std_plan(sp, lo, nz)).mean() >= target:
            return lo
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ade(std_plan(sp, mid, nz)).mean() < target else (lo, mid)
        return 0.5 * (lo + hi)
    p0b, p1b = (0.5, 0.0) if base == "M2b-pgt" else (1.0, 0.2)
    B_RUN = lambda s: f"{base}_s{s}"                                                    # noqa: E731
    G_RUN = lambda s: f"{base}_s{s}_8x20"                                               # noqa: E731
    ade_b = {}
    for s in (0, 1):
        nz = np.random.default_rng(100 + s).normal(0, 1, N)
        for who in ("B", "G"):
            if who == "G" and s not in seeds:
                continue
            is_g = who == "G"
            tag = f"{base}-8x20_s{s}" if is_g else f"{base}_s{s}"
            pred = std_plan(sp_b, off_b + gap_b * s, nz) if not is_g else std_plan(sp_g, solve_off(sp_g, nz, ade_b[s] + d_ade), nz)
            if not is_g:
                ade_b[s] = ade(pred).mean()
            orc = _DK.path_at(np.full(N, 10.0), N)
            lam_w = (lam_g if is_g else lam_b) * (1.0 + 0.3 * np.random.default_rng(7 + s).normal(0, 1, N))
            lat = orc.copy()
            lat[:, :, 0] += np.where(lead, lam_w, 0.0)[:, None] * np.minimum(1.0, (np.arange(30) + 1) / 10.0)[None]
            err = (err_g1 if (s == 1 and err_g1 is not None) else err_g) if is_g else err_b
            tc = g.copy()
            tc[:, C0:] = np.where(present, g[:, C0:] + err * z_all, 0.0)
            ckpt = f"D:\\runs\\{G_RUN(s)}\\ckpt_last.pt" if is_g else None
            np.savez(ev / f"eval_{tag}.npz", pred=pred.astype(np.float32), ade6=ade(pred), eid=eid)
            ej = {"raw": {"traj": {"ade6": float(ade(pred).mean())}, "tactical": {"acc": 0.5}, "strategic": {"acc": 0.5}, "families": {}}}
            if ckpt:
                ej["ckpt"] = ckpt
            (ev / f"eval_{tag}.json").write_text(json.dumps(ej), encoding="utf-8")
            np.savez(ev / f"eval_{tag}-tacabs.npz", pred=pred.astype(np.float32), ade6=ade(pred), eid=eid, tac_code=tc, tac_oracle_fallback=fb)
            (ev / f"eval_{tag}-tacabs.json").write_text(json.dumps({"tac_manipulation": {"mode": "absmask"}, **({"ckpt": ckpt} if ckpt else {})}), encoding="utf-8")
            if (not is_g) and base == "M2b":                                           # the banked 2026-10-20 format: no tac_code (WC Amendment 1)
                np.savez(ev / f"eval_{tag}-tacorc.npz", pred=orc.astype(np.float32), ade6=ade(orc), eid=eid, tac_oracle_fallback=fb)
                ag = g.copy()
                ag[:, :22] = np.random.default_rng(99).normal(0, 1, (N, 22))
                np.savez(ev / f"eval_{tag}-tacorc-all-goal.npz", pred=orc.astype(np.float32), ade6=ade(orc), eid=eid, tac_code=ag, tac_oracle_fallback=fb)
            else:
                np.savez(ev / f"eval_{tag}-tacorc.npz", pred=orc.astype(np.float32), ade6=ade(orc), eid=eid, tac_code=g, tac_oracle_fallback=fb)
            (ev / f"eval_{tag}-tacorc.json").write_text(json.dumps({"tac_manipulation": {"mode": "oracle"}, **({"ckpt": ckpt} if ckpt else {})}), encoding="utf-8")
            lc = g.copy()
            lc[:, LATCONS] = tc[:, LATCONS] + 0.01
            np.savez(ev / f"eval_{tag}-tacorc-all-latcons.npz", pred=lat.astype(np.float32), ade6=ade(lat), eid=eid, tac_code=lc, tac_oracle_fallback=fb)
            (ev / f"eval_{tag}-tacorc-all-latcons.json").write_text(json.dumps({"tac_manipulation": {"mode": "oracle"}, **({"ckpt": ckpt} if ckpt else {})}), encoding="utf-8")
            # ---- the run
            rd = ru / (G_RUN(s) if is_g else B_RUN(s))
            rd.mkdir(parents=True, exist_ok=True)
            argv = [x.replace("{s}", str(s)).replace("{dir}", G_RUN(s) if is_g else B_RUN(s)) for x in B_ARGV]
            if base == "M2b-pgt":
                argv[2:2] = PGT_FLAGS
            if is_g:
                argv[argv.index("--grid") + 1] = "pool8x20"
            lev = {"wm": "b", "w": {"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1, "w_tac": 1.0, "w_cons": 0.5}, "residual_zero_init": False,
                   "params": {"adapter": ADAPTER_B + (D_POS if is_g else 0), "S": 11, "T": 22, "O": 33}, "n_params_trainable": TOTAL_B + (D_POS if is_g else 0),
                   "p_gt_start": p0b, "p_gt_end": p1b}
            if is_g and adapter_g is not None:
                lev["params"]["adapter"] = adapter_g
            sha = f"{'g' if is_g else 'b'}{s}{base}"
            sm = {"arm": "M2b", "seed": s, "steps": steps, "done": True, "n_clips": 4512, "train_sha256": "t" * 64, "excluded_sha256": HOLD, "argv_sha": sha,
                  "seconds": 26000.0 if is_g else 15700.0, "levers": lev}
            if not is_g and base == "M2b":
                sm["levers"] = {k: v for k, v in lev.items() if k not in ("p_gt_start", "p_gt_end")}        # banked runs predate the record
            (rd / "summary.json").write_text(json.dumps(sm), encoding="utf-8")
            (rd / "argv.json").write_text(json.dumps({"argv": argv, "sha": sha}), encoding="utf-8")
            sc = {"declared_vs_built": "PASS", "argv_sha": sha, "levers": dict(sm["levers"], residual_zero_init=True), "info": {"lru": 43 if is_g else 174}}
            (rd / "selfcheck.json").write_text(json.dumps(sc), encoding="utf-8")
            with open(rd / "train_log.jsonl", "w", encoding="utf-8") as f:
                for st in range(957, steps + 1, 957):
                    f.write(json.dumps({"step": st, "p_gt": _sched(st - 1, steps, p0b, p1b), "ade": 1.0}) + "\n")
            if is_g:
                torch.save({"model": {"pos": torch.zeros(160, 4)}, "cfg": {"n_tok": 160}, "grid": "pool8x20", "argv_sha": sha, "step": steps, "arm": "M2b", "seed": s},
                           rd / "ckpt_last.pt")
    # ---- the stand-in PGT read: the committed shapes of v8f_pgt_read.py's output (schema, stage, DECISION, VERDICT, rows.*.beside.ade6 of the PGT arms)
    pgt = d / "pgt"
    pgt.mkdir(exist_ok=True)
    verdict = pgt_verdict or ("SUCCESS" if base == "M2b-pgt" else "FAIL-NO-MECHANISM")
    mean = lambda tag: float(np.load(ev / f"eval_{tag}.npz")["ade6"].mean())           # noqa: E731
    rows = {f"s{s}": {"beside": {"ade6": ({"M2b": mean(f"M2b_s{s}"), "M2b-pgt": 9.9} if base == "M2b" else {"M2b": 9.9, "M2b-pgt": mean(f"M2b-pgt_s{s}")})}} for s in (0, 1)}
    if pgt_s0 != "none":
        (pgt / "pgt_read_s0.json").write_text(json.dumps({"schema": "v8f-pgt-read-v1", "stage": "s0", "DECISION": pgt_s0 or "CONTINUE", "VERDICT": None, "rows": {"s0": rows["s0"]}}),
                                              encoding="utf-8")
    if pgt_s0 != "stop_only":
        (pgt / "pgt_read.json").write_text(json.dumps({"schema": "v8f-pgt-read-v1", "stage": "final", "DECISION": None, "VERDICT": verdict, "rows": rows}), encoding="utf-8")
    np.save(d / "p10.npy", _DK.path_at(np.full(N, 10.0), N))
    w.write_gt()
    w.write_block()
    return {"d": d, "w": w, "eval": ev, "runs": ru, "pgt": pgt, "lead": lead, "base": base, "G_RUN": G_RUN, "B_RUN": B_RUN,
            "args": ["--base", base, "--pgt-dir", str(pgt), "--eval-dir", str(ev), "--runs-dir", str(ru), "--out", str(d / "grid.json"), "--n-boot", "300", "--dk-n-boot", "60",
                     "--no-families", "--no-k30", "--lead-block", str(d / "lead.npz"), "--p10-pred", str(d / "p10.npy"), "--gt-npz", str(d / "gt.npz")]}


def _go(f, stage="final", extra=()):
    rc = R.main(["--stage", stage, *f["args"], *extra])
    out = f["d"] / "grid.json"
    return rc, (json.loads(out.read_text(encoding="utf-8")) if rc == 0 else None)


@pytest.fixture
def no_known(monkeypatch):
    """The banked premises (F 0.1207, LAMBDA +0.2678 / +0.1667, S5 excess +4.42 / +2.50 pp) are the REAL banked M2b values; a synthetic world cannot hold them.
    The tests of everything else replace the check; `test_known_values_are_checked_for_base_m2b_only` and the real-data proofs run the real one."""
    monkeypatch.setattr(R.PGT, "known_values", lambda *a, **k: {"checked": False, "why": "synthetic test input"})


# ---- the base rule ------------------------------------------------------------------------------------------------------------------------------
def _pgt_files(d, *, s0=None, final=None):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    mk = lambda stage, kw: {"schema": "v8f-pgt-read-v1", "stage": stage, "rows": {}, **kw}      # noqa: E731
    if s0 is not None:
        (d / "pgt_read_s0.json").write_text(json.dumps(mk("s0", s0)), encoding="utf-8")
    if final is not None:
        (d / "pgt_read.json").write_text(json.dumps(mk("final", final)), encoding="utf-8")
    return d


@needs_pkg
@pytest.mark.parametrize("name,s0,final,base", [
    ("final SUCCESS after a CONTINUE", dict(DECISION="CONTINUE", VERDICT=None), dict(DECISION=None, VERDICT="SUCCESS"), "M2b-pgt"),
    ("final SUCCESS, no s0 file", None, dict(DECISION=None, VERDICT="SUCCESS"), "M2b-pgt"),
    ("final FAIL-MECHANISM-OK", dict(DECISION="CONTINUE", VERDICT=None), dict(DECISION=None, VERDICT="FAIL-MECHANISM-OK"), "M2b"),
    ("final FAIL-NO-MECHANISM", dict(DECISION="CONTINUE", VERDICT=None), dict(DECISION=None, VERDICT="FAIL-NO-MECHANISM"), "M2b"),
    ("seed-0 STOP, no final", dict(DECISION="STOP", VERDICT="FAIL-NO-MECHANISM"), None, "M2b"),
    ("seed-0 STOP with the mechanism", dict(DECISION="STOP", VERDICT="FAIL-MECHANISM-OK"), None, "M2b"),
])
def test_decide_base_rule_both_branches(tmp_path, name, s0, final, base):
    """B = M2b-pgt iff PREREG_V8F_PGT's FINAL read says VERDICT == "SUCCESS" (v8f_pgt_read.py: `VERDICT`, schema v8f-pgt-read-v1, stage final); else M2b."""
    d = _pgt_files(tmp_path / "p", s0=s0, final=final)
    assert R.decide_base(d)["base"] == base, name
    out = tmp_path / "base.json"
    assert R.main(["--decide-base", "--pgt-dir", str(d), "--out", str(out)]) == 0 and json.loads(out.read_text(encoding="utf-8"))["base"] == base


@needs_pkg
@pytest.mark.parametrize("name,s0,final,msg", [
    ("no read at all", None, None, "has no read"),
    ("seed 0 says CONTINUE but there is no final read", dict(DECISION="CONTINUE", VERDICT=None), None, "did not finish"),
    ("seed 0 STOPPED but a final read exists", dict(DECISION="STOP", VERDICT="FAIL-NO-MECHANISM"), dict(DECISION=None, VERDICT="SUCCESS"), "contradict"),
    ("an unknown verdict", dict(DECISION="CONTINUE", VERDICT=None), dict(DECISION=None, VERDICT="WIN"), "none of the three committed verdicts"),
    ("a final file that is not a final read", dict(DECISION="CONTINUE", VERDICT=None), dict(DECISION=None, VERDICT="SUCCESS", stage="s0"), "not a PGT final read"),
])
def test_decide_base_rule_refuses_everything_that_is_not_a_committed_shape(tmp_path, capsys, name, s0, final, msg):
    d = _pgt_files(tmp_path / "p", s0=s0, final=final)
    with pytest.raises(SystemExit) as e:
        R.decide_base(d)
    assert e.value.code == 2 and msg in capsys.readouterr().err, name
    assert R.main(["--decide-base", "--pgt-dir", str(d)]) == 2


@needs_pkg
def test_decide_base_rule_wrong_schema_and_the_exact_key_value():
    """The committed key and value are read out of v8f_pgt_read.py itself: `VERDICT` / `SUCCESS` are what its `verdict()` returns and its output dict names."""
    src = (CODE / "v8f_pgt_read.py").read_text(encoding="utf-8")
    assert 'schema="v8f-pgt-read-v1"' in src and 'DECISION=decision, VERDICT=vd' in src and '("SUCCESS" if ok else' in src
    assert R.PGT_SCHEMA == "v8f-pgt-read-v1" and R.PGT.verdict("final", {"s0": {"judge": {"passed": True, "mechanism": True}}, "s1": {"judge": {"passed": True, "mechanism": True}}}, (0, 1))[1] == "SUCCESS"


# ---- the decision table, both bases ----------------------------------------------------------------------------------------------------------------
@needs_pkg
@pytest.mark.parametrize("base", BASES)
def test_reader_decision_table_final_stage(tmp_path, no_known, base):
    rc, r = _go(_gworld(tmp_path / "ok", base))
    assert rc == 0 and r["VERDICT"] == "SUCCESS" and r["mechanism"] is True and r["DECISION"] is None and r["base"] == base, r["reading"]
    f = r["F"]
    ev = tmp_path / "ok" / "eval"
    d = np.load(ev / f"eval_{base}_s0.npz")["ade6"].mean() - np.load(ev / f"eval_{base}_s1.npz")["ade6"].mean()
    assert f == pytest.approx(abs(d), abs=1e-12) and r["two_F"] == pytest.approx(2 * f), f"F is B's own two seeds' gap ({base}), from BOTH evals"
    for s in ("s0", "s1"):
        row = r["rows"][s]
        assert row["primary"]["point"] < -2 * f and row["primary"]["separated"] and row["judge"]["passed"]
        assert row["mlat"]["fall"] == pytest.approx(1 - 0.5 / 0.6, abs=1e-9) and row["mlat"]["mse_delta"]["separated"] and row["judge"]["mlat"]["met"]
    # PRIMARY under 2F, mechanism met -> FAIL-MECHANISM-OK, the committed next lever named
    rc, r = _go(_gworld(tmp_path / "small", base, d_ade=-0.10))
    assert rc == 0 and r["VERDICT"] == "FAIL-MECHANISM-OK" and "LoRA on DINOv3" in r["next"] and r["mechanism"] is True
    assert r["rows"]["s0"]["primary"]["separated"] and 0 < abs(r["rows"]["s0"]["primary"]["point"]) < r["two_F"], "beyond 0, under 2F"
    # latcons accuracy did not fall 12.5 % -> FAIL-NO-MECHANISM, whatever the ADE did
    rc, r = _go(_gworld(tmp_path / "nomech", base, d_ade=-0.10, err_g=0.55))
    assert rc == 0 and r["VERDICT"] == "FAIL-NO-MECHANISM" and "more history frames" in r["next"] and r["mechanism"] is False
    # driving better beyond 2F at both seeds is SUCCESS even where the mechanism read is not met (the table: SUCCESS = both seeds pass)
    rc, r = _go(_gworld(tmp_path / "succ_nomech", base, err_g=0.6))
    assert rc == 0 and r["VERDICT"] == "SUCCESS" and r["mechanism"] is False
    # worse driving is never a pass
    rc, r = _go(_gworld(tmp_path / "worse", base, d_ade=0.8))
    assert rc == 0 and r["VERDICT"] != "SUCCESS" and not r["rows"]["s0"]["judge"]["passed"]
    # one seed is not enough
    f1 = _gworld(tmp_path / "oneseed", base)
    rc, r = _go(f1)
    assert r["VERDICT"] == "SUCCESS"
    g1 = _gworld(tmp_path / "oneseed2", base, d_ade=-0.7)
    _seed1_worse(g1, base)
    rc, r = _go(g1)
    assert rc == 0 and r["VERDICT"] != "SUCCESS" and r["rows"]["s0"]["judge"]["passed"] and not r["rows"]["s1"]["judge"]["passed"]


def _seed1_worse(f, base):
    """Make G's seed-1 ADE6 worse than B's by shifting its plan laterally beyond 2 s (the ade6 key is rewritten to match, as the eval would)."""
    ev = f["eval"]
    w = f["w"]
    for suf in ("", "-tacabs"):
        p = ev / f"eval_{base}-8x20_s1{suf}.npz"
        z = dict(np.load(p))
        z["pred"] = z["pred"].copy()
        z["pred"][:, 10:, 1] += 3.0
        z["ade6"] = np.linalg.norm(z["pred"].astype(np.float64) - w.gt, axis=-1).mean(1)
        np.savez(p, **z)


@needs_pkg
@pytest.mark.parametrize("base", BASES)
def test_reader_mlat_is_the_latcons_rmse_fall_with_a_12p5_percent_bar_and_needs_separation(tmp_path, no_known, base):
    # fall = 1 - err_g / err_b on the slot-pooled RMSE, exactly (the same noise draws, scaled): 10 % misses the bar, 15 % clears it, a RISE is never a pass
    for name, err_g, met in (("fall10", 0.54, False), ("fall15", 0.51, True), ("rise", 0.66, False), ("fall0", 0.60, False)):
        rc, r = _go(_gworld(tmp_path / name, base, d_ade=-0.10, err_g=err_g))
        j = r["rows"]["s0"]["judge"]["mlat"]
        assert rc == 0 and j["met"] is met and j["bar"] == 0.125, name
        assert r["rows"]["s0"]["mlat"]["fall"] == pytest.approx(1 - err_g / 0.6, abs=1e-9), name
    # the rise is separated (WORSE) and still not met: the sign matters, not only the size
    rc, r = _go(_gworld(tmp_path / "rise2", base, d_ade=-0.10, err_g=0.66))
    d = r["rows"]["s0"]["mlat"]["mse_delta"]
    assert d["point"] > 0 and d["separated"] and not r["rows"]["s0"]["judge"]["mlat"]["met"]
    # the reference RMSE is the one the PREREG quotes (slot-pooled, WC's rmse_group), the window-weighted one rides beside it
    row = r["rows"]["s0"]["mlat"]
    assert row["rmse_slot"]["B"] == pytest.approx(0.6 * 1.0, rel=0.25) and row["rmse_window"]["B"] > 0 and row["definitions_agree"] is True


@needs_pkg
def test_judge_applies_the_bar_inclusively_and_requires_a_separated_negative_mse_delta():
    mk = lambda fall, point=-0.1, sep=True, prim=-1.0, psep=True: {"primary": {"point": prim, "separated": psep},        # noqa: E731
                                                                  "mlat": {"fall": fall, "mse_delta": {"point": point, "separated": sep}}}
    assert R.judge(mk(0.125), 0.1)["mlat"]["met"] is True, "exactly 12.5 % clears the bar (>=)"
    assert R.judge(mk(0.1249), 0.1)["mlat"]["met"] is False
    assert R.judge(mk(0.30, sep=False), 0.1)["mlat"]["met"] is False, "a 30 % fall that is not separated is not met"
    assert R.judge(mk(0.30, point=0.05), 0.1)["mlat"]["met"] is False, "a RISE of the per-window MSE is not met even if the RMSE fall says 30 %"
    # PRIMARY: delta < 0, separated, |delta| > 2F (strict)
    assert R.judge(mk(0.2, prim=-0.21), 0.1)["passed"] is True
    assert R.judge(mk(0.2, prim=-0.20), 0.1)["passed"] is False, "|delta| == 2F does not pass"
    assert R.judge(mk(0.2, prim=-0.19), 0.1)["passed"] is False
    assert R.judge(mk(0.2, prim=-5.0, psep=False), 0.1)["passed"] is False
    assert R.judge(mk(0.2, prim=+5.0), 0.1)["passed"] is False


@needs_pkg
def test_mlat_two_rmse_definitions_can_disagree_and_the_reader_says_so():
    """Slot-pooled vs window-weighted: one window with 4 present latcons slots, one with 1. B's error is 1.0 everywhere; G removes it entirely on the 4-slot window and
    leaves the 1-slot window alone. Slot-pooled: sqrt(1/5) -> a 55.3 % fall; window-weighted: sqrt(1/2) -> a 29.3 % fall (here both clear the bar; the straddle is tested next)."""
    DK, WC = R._modules()
    cols = list(WC.GROUPS["latcons"])
    N = 6
    g = np.zeros((N, NC))
    g[0, cols[:4]] = 1.0
    g[1, cols[:1]] = 1.0
    has = np.ones(N, bool)
    mk = lambda e0, e1: {"t_cons": np.where(g != 0, g + np.array([e0, e1, 0, 0, 0, 0])[:, None], 0.0)}                 # noqa: E731
    B_, G_ = mk(1.0, 1.0), mk(0.2, 1.0)
    mb, nb = R.latcons_window_mse(B_, g, has, WC)
    mg, ng = R.latcons_window_mse(G_, g, has, WC)
    assert nb == ng == 5 and np.isclose(mb[0], 1.0) and np.isclose(mb[1], 1.0) and np.isnan(mb[2]) and np.isclose(mg[0], 0.04) and np.isclose(mg[1], 1.0)
    slot = lambda m: float(np.sqrt(np.where(g[:, cols] != 0, (m["t_cons"][:, cols] - g[:, cols]) ** 2, 0.0)[g[:, cols] != 0].mean()))     # noqa: E731
    fall_slot, fall_win = 1 - slot(G_) / slot(B_), 1 - np.sqrt(np.nanmean(mg)) / np.sqrt(np.nanmean(mb))
    assert fall_slot == pytest.approx(1 - np.sqrt((4 * 0.04 + 1) / 5), abs=1e-9) and fall_win == pytest.approx(1 - np.sqrt((0.04 + 1) / 2), abs=1e-9)
    assert fall_slot != pytest.approx(fall_win, abs=1e-3), "the two definitions are different numbers"
    # a straddle: G's slot-pooled fall 13 %, window-weighted fall 10 % -> the reader's `definitions_agree` is False
    ea = 1.0 - 0.13
    pe_b = {"rmse_group": {"latcons": 1.0}}
    pe_a = {"rmse_group": {"latcons": ea}}
    eid = np.arange(N)
    ref = {"has": has, "g_cons": g}
    # build arms whose window-weighted RMSE ratio is 0.90 while the slot-pooled ratio is 0.87
    A2 = {"t_cons": np.where(g != 0, g + np.array([0.8, 1.0, 0, 0, 0, 0])[:, None], 0.0)}
    row = R.mlat_row(lambda a, b, e, nb_, seed: {"delta": float(np.mean(a) - np.mean(b)), "lo": -1.0, "hi": -0.1, "separated": True}, A2, B_, pe_a, pe_b, ref, WC, eid, 10)
    assert row["fall"] == pytest.approx(0.13, abs=1e-12)
    expect_win = 1 - np.sqrt((1 * 0.64 + 1 * 1.0) / 2) / 1.0
    assert row["fall_window"] == pytest.approx(expect_win, abs=1e-9)
    assert row["definitions_agree"] == ((row["fall"] >= R.MLAT_BAR) == (row["fall_window"] >= R.MLAT_BAR))


@needs_pkg
def test_mlat_row_refuses_when_there_is_nothing_to_measure():
    DK, WC = R._modules()
    N = 4
    g = np.zeros((N, NC))                                           # no present latcons slot anywhere
    A = {"t_cons": g.copy()}
    pe = {"rmse_group": {"latcons": None}}
    with pytest.raises(SystemExit) as e:
        R.mlat_row(lambda *a: {}, A, A, pe, pe, {"has": np.ones(N, bool), "g_cons": g}, WC, np.arange(N), 10)
    assert e.value.code == 2


# ---- stages ----------------------------------------------------------------------------------------------------------------------------------------
@needs_pkg
@pytest.mark.parametrize("base", BASES)
def test_reader_stage_s0_continues_or_stops_on_seed_0_alone(tmp_path, no_known, base):
    rc, r = _go(_gworld(tmp_path / "go", base, seeds=(0,)), "s0")
    assert rc == 0 and r["DECISION"] == "CONTINUE" and r["VERDICT"] is None and "train seed 1" in r["next"] and list(r["rows"]) == ["s0"], r["reading"]
    assert r["two_F"] > 0, "F still comes from BOTH of B's seeds at stage s0"
    rc, r = _go(_gworld(tmp_path / "stop_mech", base, d_ade=-0.10, seeds=(0,)), "s0")
    assert rc == 0 and r["DECISION"] == "STOP" and r["VERDICT"] == "FAIL-MECHANISM-OK" and "LoRA" in r["next"]
    rc, r = _go(_gworld(tmp_path / "stop_nomech", base, d_ade=-0.10, err_g=0.58, seeds=(0,)), "s0")
    assert rc == 0 and r["DECISION"] == "STOP" and r["VERDICT"] == "FAIL-NO-MECHANISM" and "more history frames" in r["next"]
    rc, r = _go(_gworld(tmp_path / "stop_worse", base, d_ade=0.8, seeds=(0,)), "s0")
    assert rc == 0 and r["DECISION"] == "STOP"
    assert "s1" not in _go(_gworld(tmp_path / "both", base, seeds=(0, 1)), "s0")[1]["rows"], "seed 1 is not read at s0 even when its files are present"


@needs_pkg
def test_reader_f_is_the_bases_own_seed_floor_and_the_primary_bar_is_two_f(tmp_path, no_known):
    """F = |B_s0 - B_s1|: with gap_b large, F is large and a -0.7 m win no longer clears 2F; with gap_b ~ 0 the same win does. For B = M2b-pgt it is M2b-PGT's two seeds."""
    for base in BASES:
        rc, r = _go(_gworld(tmp_path / f"big_{base}", base, gap_b=2.0, d_ade=-0.7))
        assert r["F"] == pytest.approx(0.6, rel=0.15) and r["two_F"] > 0.7 and not r["rows"]["s0"]["judge"]["passed"] and r["VERDICT"] != "SUCCESS", base
        rc, r = _go(_gworld(tmp_path / f"small_{base}", base, gap_b=0.02, d_ade=-0.7))
        assert r["F"] < 0.1 and r["rows"]["s0"]["judge"]["passed"] and r["VERDICT"] == "SUCCESS", base
        ev = tmp_path / f"small_{base}" / "eval"
        want = abs(np.load(ev / f"eval_{base}_s0.npz")["ade6"].mean() - np.load(ev / f"eval_{base}_s1.npz")["ade6"].mean())
        assert r["F"] == pytest.approx(want, abs=1e-12)


@needs_pkg
@pytest.mark.parametrize("base", BASES)
def test_reader_deliberate_regression_reads_worse_and_fails(tmp_path, no_known, base):
    """A synthetic G whose plans are B's PLUS a deliberate lateral error beyond 2 s (d_ade > 0) and whose T is no more accurate than B's: PRIMARY fails, M-LAT not met."""
    rc, r = _go(_gworld(tmp_path / "reg", base, d_ade=1.0, err_g=0.6, lam_g=0.27 + 0.5, sp_g=11.6))
    assert rc == 0 and r["VERDICT"] == "FAIL-NO-MECHANISM"
    for s in ("s0", "s1"):
        row = r["rows"][s]
        assert row["primary"]["point"] > 0 and row["primary"]["separated"] and not row["judge"]["passed"] and not row["judge"]["mlat"]["met"]
        assert row["mleak"]["delta"]["point"] > 0 and row["mdk"]["delta"]["point"] > 0, "the beside reads move the same way"
    # and the success control: the SAME reader reads SUCCESS on the opposite knobs (it is not structurally unable to pass)
    assert _go(_gworld(tmp_path / "pos", base))[1]["VERDICT"] == "SUCCESS"


# ---- the refusals: every precondition of a G-vs-B read is a refusal, never a quiet pass ------------------------------------------------------------------
def _gr(f, s=0):
    return f["runs"] / f["G_RUN"](s)


def _br(f, s=0):
    return f["runs"] / f["B_RUN"](s)


def _geval(f, s, suf=""):
    return f["eval"] / f"eval_{f['base']}-8x20_s{s}{suf}"


def _argv_edit(path, fn):
    def go(j):
        j["argv"] = fn(list(j["argv"]))
    _ej(path, go)


def _set_flag(argv, flag, val):
    argv[argv.index(flag) + 1] = val
    return argv


def _ck_edit(f, fn, s=0):
    p = _gr(f, s) / "ckpt_last.pt"
    o = torch.load(p, weights_only=False)
    fn(o)
    torch.save(o, p)


def _lev(path, fn):
    def go(j):
        fn(j["levers"])
    _ej(path, go)


def _npz_edit(path, fn):
    z = dict(np.load(path))
    fn(z)
    np.savez(path, **z)


def _drop(f, s=0, name="summary.json"):
    (_gr(f, s) / name).unlink()


def _log_edit(f, fn):
    p = _gr(f) / "train_log.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    fn(rows)
    p.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")


def _shift_eid(z):
    z["eid"] = np.asarray(z["eid"]).copy()
    z["eid"][0] += 1


def _gt_codes_off(z):                                  # the constraint block (cols >= C0) is the GT every arm must carry identically
    z["tac_code"] = np.asarray(z["tac_code"]).copy()
    z["tac_code"][10:40, C0:] += 1.0


def _not_the_latcons_oracle(z):                        # outside the latcons columns the all-latcons oracle must still be GT
    z["tac_code"] = np.asarray(z["tac_code"]).copy()
    z["tac_code"][10:40, C0 + 12:] += 1.0


def _tie_off(f):
    _ej(f["pgt"] / "pgt_read.json", lambda j: j["rows"]["s0"]["beside"]["ade6"].update({f["base"]: j["rows"]["s0"]["beside"]["ade6"][f["base"]] + 0.01}))


def _fewer_steps(f, n):
    """A run that stopped at `n` steps, with its own consistent log (so the step-count check, not the log check, is what must refuse)."""
    p0, p1 = (0.5, 0.0) if f["base"] == "M2b-pgt" else (1.0, 0.2)
    _ej(_gr(f) / "summary.json", lambda j: j.update(steps=n))
    _log_edit(f, lambda r: r.__setitem__(slice(None), [dict(x, p_gt=_sched(x["step"] - 1, n, p0, p1)) for x in r if x["step"] <= n]))


MUT = {                                # name -> (mutator(f), a message fragment, the bases it is run for)
    "g summary missing": (lambda f: _drop(f, 0, "summary.json"), "is missing", BASES[:1]),
    "g checkpoint missing": (lambda f: _drop(f, 0, "ckpt_last.pt"), "is missing", BASES[:1]),
    "g not done": (lambda f: _ej(_gr(f) / "summary.json", lambda j: j.update(done=False)), "is not done", BASES[:1]),
    "g n_clips": (lambda f: _ej(_gr(f) / "summary.json", lambda j: j.update(n_clips=4000)), "n_clips 4000", BASES[:1]),
    "g withheld nothing": (lambda f: _ej(_gr(f) / "summary.json", lambda j: j.update(excluded_sha256="0" * 64)), "did not withhold", BASES[:1]),
    "g trained on another clip list": (lambda f: _ej(_gr(f) / "summary.json", lambda j: j.update(train_sha256="u" * 64)), "did not train on", BASES[:1]),
    "g trained fewer steps": (lambda f: _fewer_steps(f, 9000), "trained for 9000 steps", BASES[:1]),
    "g launch gate not PASS": (lambda f: _ej(_gr(f) / "selfcheck.json", lambda j: j.update(declared_vs_built="FAIL")), "declared_vs_built is not PASS", BASES[:1]),
    "g selfcheck from another run": (lambda f: _ej(_gr(f) / "selfcheck.json", lambda j: j.update(argv_sha="zzz")), "is not this run's", BASES[:1]),
    "g argv still says 4x10": (lambda f: _argv_edit(_gr(f) / "argv.json", lambda a: _set_flag(a, "--grid", "pool4x10")), "expected pool8x20", BASES[:1]),
    "g argv has a second lever": (lambda f: _argv_edit(_gr(f) / "argv.json", lambda a: a + ["--w-wm", "0.2"]), "not one lever", BASES),
    "g argv lost a flag": (lambda f: _argv_edit(_gr(f) / "argv.json", lambda a: [x for x in a if x not in ("--kin-mode", "yaw")]), "not one lever", BASES[:1]),
    "g argv is another arm": (lambda f: _argv_edit(_gr(f) / "argv.json", lambda a: _set_flag(a, "--arm", "M2")), "argv --arm", BASES[:1]),
    "b argv is an 8x20 run": (lambda f: _argv_edit(_br(f) / "argv.json", lambda a: _set_flag(a, "--grid", "pool8x20")), "B is not a 4x10 run", BASES[:1]),
    "g built another p_gt": (lambda f: _lev(_gr(f) / "summary.json", lambda l: l.update(p_gt_start=0.7)), "was built with p_gt", BASES),
    "g records no built p_gt": (lambda f: _lev(_gr(f) / "selfcheck.json", lambda l: l.pop("p_gt_end")), "record no built p_gt", BASES),
    "g built another weight": (lambda f: _lev(_gr(f) / "summary.json", lambda l: l["w"].update(w_pen=0.9)), "built levers", BASES[:1]),
    "g built another mask": (lambda f: _lev(_gr(f) / "selfcheck.json", lambda l: l.update(wm="a")), "built levers", BASES[:1]),
    "g has one parameter too many": (lambda f: _lev(_gr(f) / "summary.json", lambda l: l.update(n_params_trainable=l["n_params_trainable"] + 1)),
                                     "parameter delta", BASES),
    "g adapter delta is not the table": (lambda f: _lev(_gr(f) / "selfcheck.json", lambda l: l["params"].update(adapter=l["params"]["adapter"] - 256)),
                                         "parameter delta", BASES[:1]),
    "g grew a second group": (lambda f: _lev(_gr(f) / "summary.json", lambda l: l["params"].update(T=l["params"]["T"] + 5)), "group T", BASES[:1]),
    "g checkpoint is a 4x10 checkpoint": (lambda f: _ck_edit(f, lambda o: o.update(grid="pool4x10")), "carries grid", BASES[:1]),
    "g checkpoint built 40 tokens": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(n_tok=40)), "BUILT with n_tok 40", BASES[:1]),
    "g checkpoint has 40 position rows": (lambda f: _ck_edit(f, lambda o: o["model"].update(pos=torch.zeros(40, 4))), "position table has 40 rows", BASES[:1]),
    "g checkpoint from another argv": (lambda f: _ck_edit(f, lambda o: o.update(argv_sha="other")), "another argv", BASES[:1]),
    "g checkpoint at another step": (lambda f: _ck_edit(f, lambda o: o.update(step=100)), "checkpoint is at step 100", BASES[:1]),
    "g checkpoint of another seed": (lambda f: _ck_edit(f, lambda o: o.update(seed=1)), "checkpoint says arm", BASES[:1]),
    "g train log carries the wrong p_gt": (lambda f: _log_edit(f, lambda r: r[3].update(p_gt=0.99)), "p_gt", BASES),
    "g eval scored B's checkpoint": (lambda f: _ej(_geval(f, 0, ".json"), lambda j: j.update(ckpt=str(_br(f) / "ckpt_last.pt"))), "scored", BASES[:1]),
    "g oracle eval scored no checkpoint": (lambda f: _ej(_geval(f, 0, "-tacorc.json"), lambda j: j.pop("ckpt") and None), "scored None", BASES[:1]),
    "g eval is on another eval grid": (lambda f: _npz_edit(_geval(f, 0, ".npz"), _shift_eid), "eid", BASES[:1]),
    "g oracle eval has other GT codes": (lambda f: _npz_edit(_geval(f, 0, "-tacorc.npz"), _gt_codes_off), "the GT constraint block differs", BASES[:1]),
    "g all-latcons oracle is not GT outside latcons": (lambda f: _npz_edit(_geval(f, 0, "-tacorc-all-latcons.npz"), _not_the_latcons_oracle),
                                                      "not the all-latcons oracle", BASES[:1]),
    "g eval missing": (lambda f: _geval(f, 0, "-tacorc-all-latcons.npz").unlink(), "", BASES[:1]),
    "b evals are not the ones the PGT verdict read": (_tie_off, "does not match the PGT verdict", BASES),
}


@needs_pkg
@pytest.mark.parametrize("name", list(MUT))
def test_reader_refuses_rc2_with_the_reason(tmp_path, no_known, capsys, name):
    mut, frag, bases = MUT[name]
    for base in bases:
        f = _gworld(tmp_path / base.replace("-", "_"), base)
        mut(f)
        capsys.readouterr()
        rc = R.main(["--stage", "final", *f["args"]])
        err = capsys.readouterr().err
        assert rc == 2 and frag in err, f"{name} [{base}]: rc {rc}; stderr: {err[-400:]}"
        assert not (f["d"] / "grid.json").exists(), "a refused read must leave no result file for the next step to find"


@needs_pkg
def test_reader_refuses_a_base_that_the_pgt_verdict_does_not_decide(tmp_path, no_known, capsys):
    f = _gworld(tmp_path / "w", "M2b-pgt")                                              # the PGT final read says SUCCESS: B = M2b-pgt
    capsys.readouterr()
    assert R.main(["--stage", "final", *f["args"], "--base", "M2b"]) == 2
    assert "decides B = M2b-pgt" in capsys.readouterr().err
    f = _gworld(tmp_path / "w2", "M2b")                                                 # a FAIL verdict: B = M2b
    assert R.main(["--stage", "final", *f["args"], "--base", "M2b-pgt"]) == 2
    assert "decides B = M2b" in capsys.readouterr().err


@needs_pkg
def test_reader_refuses_a_final_read_without_seed_1_of_g(tmp_path, no_known, capsys):
    f = _gworld(tmp_path / "w", "M2b", seeds=(0,))
    assert R.main(["--stage", "final", *f["args"]]) == 2
    assert "is missing" in capsys.readouterr().err
    assert R.main(["--stage", "s0", *f["args"]]) == 0, "the same files are a complete stage-1 input"


@needs_pkg
def test_reader_refuses_a_pgt_b_without_the_pgt_flags(tmp_path, no_known, capsys):
    """B = M2b-pgt must itself be a pgt run: PGT's own run check (flags, built schedule, loop-logged p_gt) is applied to it, not assumed."""
    f = _gworld(tmp_path / "w", "M2b-pgt")
    _argv_edit(_br(f) / "argv.json", lambda a: [x for x in a if x not in ("--p-gt-start", "0.5", "--p-gt-end", "0.0")])
    capsys.readouterr()
    assert R.main(["--stage", "final", *f["args"]]) == 2
    assert "REFUSED" in capsys.readouterr().err


@needs_pkg
def test_reader_missing_inputs_are_rc2_not_a_traceback(tmp_path, no_known, capsys):
    assert R.main(["--stage", "final", "--base", "M2b", "--pgt-dir", str(tmp_path / "none"), "--eval-dir", str(tmp_path), "--runs-dir", str(tmp_path), "--out", str(tmp_path / "o.json")]) == 2
    assert "has no read" in capsys.readouterr().err
    assert R.main(["--stage", "final"]) == 2, "argparse: required arguments missing"


# ---- the pre-launch check of B's side ----------------------------------------------------------------------------------------------------------------
@needs_pkg
@pytest.mark.parametrize("base", BASES)
def test_check_inputs_reads_b_alone_and_the_identity_control_is_exactly_zero(tmp_path, no_known, capsys, base):
    f = _gworld(tmp_path / "c", base, seeds=())                                         # no G run exists yet: this is the check that runs BEFORE training
    assert R.main(["--check-inputs", *f["args"]]) == 0
    out = capsys.readouterr().out
    assert "GRID INPUTS OK" in out and f"base {base}" in out and "identity control exactly +0 [0, 0]" in out
    j = json.loads((f["d"] / "grid.json").read_text(encoding="utf-8"))
    assert j["base"] == base and j["F"] > 0 and j["identity_control"]["s0"]["mlat_fall"] == 0.0
    for s in ("s0", "s1"):
        cells = j["identity_control"][s]
        assert cells["mlat_mse"] == {"delta": 0.0, "lo": 0.0, "hi": 0.0, "point": 0.0}, "B against itself: exactly +0 [0, 0] on M-LAT's own cell"
        assert cells["latcons_rmse_slot"] > 0 and cells["latcons_rmse_window"] > 0
    assert j["b_identity_tied_to_pgt"]["checked"] is True and j["b_identity_tied_to_pgt"]["n_seed_arms"] == 2
    assert j["latcons_rmse_slot_b"]["0"] == pytest.approx(j["identity_control"]["s0"]["latcons_rmse_slot"])


@needs_pkg
def test_check_inputs_breaks_loudly_when_the_paired_instrument_is_broken(tmp_path, no_known, capsys, monkeypatch):
    """The identity control has teeth: a paired estimator that does not return exactly 0 on B-against-B stops the check (and the read)."""
    f = _gworld(tmp_path / "c", "M2b", seeds=())
    import taniteval.ci as ci
    orig = ci.paired_episode_cluster_bootstrap

    def biased(a, b, *r, **k):
        o = orig(a, b, *r, **k)
        if np.array_equal(np.asarray(a), np.asarray(b)):
            o = dict(o, delta=o["delta"] + 1e-9)
        return o
    monkeypatch.setattr(ci, "paired_episode_cluster_bootstrap", biased)
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2
    assert "IDENTITY CONTROL FAILED" in capsys.readouterr().err


@needs_pkg
def test_known_values_are_checked_for_base_m2b_only_and_refuse_a_world_that_does_not_hold_them(tmp_path, capsys):
    """The PREREG's premises (F 0.1207, LAMBDA, S5 excess) are the banked M2b evals' values: a synthetic world cannot reproduce them, so the real check REFUSES
    it for B = M2b, and does not apply to B = M2b-pgt (whose identity is tied to the PGT read instead)."""
    f = _gworld(tmp_path / "m", "M2b", seeds=())
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2
    assert "REFUSED" in capsys.readouterr().err
    f = _gworld(tmp_path / "p", "M2b-pgt", seeds=())
    assert R.main(["--check-inputs", *f["args"]]) == 0
    j = json.loads((f["d"] / "grid.json").read_text(encoding="utf-8"))
    assert j["known_values"]["checked"] is False and "M2b-pgt" in j["known_values"]["why"]


# ---- what the read reports beside the verdict ---------------------------------------------------------------------------------------------------------
@needs_pkg
@pytest.mark.parametrize("base", BASES)
def test_reader_reports_the_parameters_the_pipeline_and_the_mechanism_beside_the_verdict(tmp_path, no_known, base):
    rc, r = _go(_gworld(tmp_path / "w", base))
    assert rc == 0
    row = r["rows"]["s0"]
    be = row["beside"]
    assert be["parameters"]["expected_delta"] == 30720 and be["parameters"]["delta"] == 30720 and be["parameters"]["adapter"] == {"B": ADAPTER_B, "G": ADAPTER_B + 30720}
    assert be["pipeline"]["G"]["lru"] == 43 and be["pipeline"]["B"]["lru"] == 174 and be["pipeline"]["G"]["steps_per_s"] < be["pipeline"]["B"]["steps_per_s"]
    assert set(be["present_slot"]["rmse_group"]["B"]) >= {"latcons"} and be["present_slot"]["rmse"]["G"] < be["present_slot"]["rmse"]["B"]
    assert be["ade6"]["G"] < be["ade6"]["B"] and be["oracle_ade6_delta"]["delta"] == pytest.approx(0.0, abs=1e-9), "the GT-code oracle is the same plan for both arms"
    assert be["distance_keeping_k10"]["pair_vs_b"] is not None and be["distance_keeping_k10"]["pair_vs_p10"] is not None
    assert row["base"]["n_lead"] > 0 and r["strategic_family"].startswith("NOT PAUSED")
    assert r["schema"] == "v8f-grid-read-v1" and r["estimator"].startswith("paired episode-cluster bootstrap, B=300 (episodes)")
    assert "H-ESTIM-SEED-1" in r["estimator"], "the estimator names the variance its interval is blind to"
    assert "T1-class open loop" in r["tier"]


# ---- the cache check (code/check_grid_cache.py): every clip of box 2's caches must carry pool8x20 BEFORE a G arm trains -------------------------------------
CC = _load("check_grid_cache_under_test", CACHE_P) if CACHE_P.exists() else None
needs_cc = pytest.mark.skipif(CC is None, reason="the v8F-mini package is not in this checkout")


def _man_row(clip, n=60, grids=("4x10", "8x20"), s4=True, s8=True):
    shapes = {}
    if s4:
        shapes["pool4x10"] = [n, 40, 1024]
    if s8:
        shapes["pool8x20"] = [n, 160, 1024]
    return {"clip": clip, "n_frames": n, "grids": list(grids), "shapes": shapes}


def _man(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _cc_world(tmp_path, n_train=6, n_eval=3):
    tr = [f"t{i}" for i in range(n_train)]
    ev = [f"e{i}" for i in range(n_eval)]
    _man(tmp_path / "tr.jsonl", [_man_row(c) for c in tr])
    _man(tmp_path / "ev.jsonl", [_man_row(c) for c in ev] + [_man_row("extra_eval_clip_not_in_grid", grids=("4x10",), s8=False)])
    (tmp_path / "grid.json").write_text(json.dumps({"order": ev}), encoding="utf-8")
    return ["--train", str(tmp_path / "tr.jsonl"), "--eval", str(tmp_path / "ev.jsonl"), "--eval-grid", str(tmp_path / "grid.json"), "--n-train", str(n_train),
            "--out", str(tmp_path / "cc.json")]


@needs_cc
def test_cache_check_passes_a_complete_cache_and_ignores_eval_clips_the_eval_never_reads(tmp_path, capsys):
    a = _cc_world(tmp_path)
    assert CC.main(a) == 0 and "GRID CACHE OK: 6 train clips and the 3 eval-grid clips" in capsys.readouterr().out
    j = json.loads((tmp_path / "cc.json").read_text(encoding="utf-8"))
    assert j["ok"] is True and j["train"]["n_bad"] == 0 and j["eval"]["n_bad"] == 0 and j["eval"]["n_grid_clips"] == 3


@needs_cc
@pytest.mark.parametrize("name,edit,frag", [
    ("one train clip without 8x20 in grids", lambda tr, ev: tr.__setitem__(2, _man_row("t2", grids=("4x10",), s8=False)), "t2"),
    ("one train clip whose grids list only 4x10 while its shapes carry pool8x20", lambda tr, ev: tr.__setitem__(2, _man_row("t2", grids=("4x10",))), "does not list 8x20"),
    ("one train clip listing 8x20 but with no pool8x20 array", lambda tr, ev: tr.__setitem__(1, _man_row("t1", s8=False)), "shapes carries no pool8x20"),
    ("one train clip with a 4x10-shaped pool8x20", lambda tr, ev: tr.__setitem__(0, dict(_man_row("t0"), shapes={"pool4x10": [60, 40, 1024], "pool8x20": [60, 40, 1024]})),
     "expected [60, 160, 1024]"),
    ("pool8x20 with the wrong frame count", lambda tr, ev: tr.__setitem__(3, dict(_man_row("t3"), shapes={"pool4x10": [60, 40, 1024], "pool8x20": [59, 160, 1024]})),
     "expected [60, 160, 1024]"),
    ("a train clip without pool4x10 (B's own grid)", lambda tr, ev: tr.__setitem__(4, _man_row("t4", s4=False)), "shapes carries no pool4x10"),
    ("an eval-grid clip without 8x20", lambda tr, ev: ev.__setitem__(1, _man_row("e1", grids=("4x10",), s8=False)), "eval clip e1"),
    ("an eval-grid clip with no manifest row", lambda tr, ev: ev.pop(0), "the clip has no manifest row"),
    ("one train clip missing (the join must cover 4,572 -> here 6)", lambda tr, ev: tr.pop(), "not 6"),
    ("a row with no grids list", lambda tr, ev: tr.__setitem__(5, {"clip": "t5", "n_frames": 60}), "no `grids` list"),
])
def test_cache_check_refuses_every_clip_that_would_crash_a_grid_arm_hours_in(tmp_path, capsys, name, edit, frag):
    a = _cc_world(tmp_path)
    tr = [json.loads(x) for x in (tmp_path / "tr.jsonl").read_text(encoding="utf-8").splitlines()]
    ev = [json.loads(x) for x in (tmp_path / "ev.jsonl").read_text(encoding="utf-8").splitlines()]
    edit(tr, ev)
    _man(tmp_path / "tr.jsonl", tr)
    _man(tmp_path / "ev.jsonl", ev)
    capsys.readouterr()
    assert CC.main(a) == 2, name
    assert frag in capsys.readouterr().err, name
    assert json.loads((tmp_path / "cc.json").read_text(encoding="utf-8"))["ok"] is False, "the refusal is also written down"


@needs_cc
def test_cache_check_the_last_manifest_line_wins_like_the_builder_and_the_trainer(tmp_path, capsys):
    a = _cc_world(tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "tr.jsonl").read_text(encoding="utf-8").splitlines()]
    _man(tmp_path / "tr.jsonl", [_man_row("t0", grids=("4x10",), s8=False)] + rows)                       # superseded line: the file was re-encoded with 8x20
    assert CC.main(a) == 0
    assert json.loads((tmp_path / "cc.json").read_text(encoding="utf-8"))["train"]["n_superseded_lines_without_8x20"] == 1
    # the same clip, but the 8x20-less line is the LAST one: the effective row is bad
    _man(tmp_path / "tr.jsonl", rows + [_man_row("t0", grids=("4x10",), s8=False)])
    capsys.readouterr()
    assert CC.main(a) == 2 and "train clip t0" in capsys.readouterr().err
    # and the trainer's own reader agrees on which line is effective (n_frames of the LAST line)
    from tanitad.data.v8f_mini_data import read_manifest
    (tmp_path / "train").mkdir()
    _man(tmp_path / "train" / "manifest.jsonl", [_man_row("c", n=10), _man_row("c", n=20)])
    assert read_manifest(tmp_path, "train") == [("c", 20)]


@needs_cc
def test_cache_check_unreadable_inputs_are_rc2(tmp_path, capsys):
    a = _cc_world(tmp_path)
    (tmp_path / "tr.jsonl").write_text("{not json\n", encoding="utf-8")
    assert CC.main(a) == 2 and "is not a manifest row" in capsys.readouterr().err
    (tmp_path / "tr.jsonl").write_text("", encoding="utf-8")
    assert CC.main(a) == 2 and "holds no rows" in capsys.readouterr().err
    assert CC.main(["--train", str(tmp_path / "nope"), "--eval", str(tmp_path / "ev.jsonl"), "--eval-grid", str(tmp_path / "grid.json")]) == 2
    _cc_world(tmp_path)
    (tmp_path / "grid.json").write_text(json.dumps({"order": []}), encoding="utf-8")
    capsys.readouterr()
    assert CC.main(a) == 2 and "lists no clips" in capsys.readouterr().err


# =================================================================================================================================================
# THE QUEUE STAGES (code/v8f_mini_box2_queue.cmd: gategrid / traingrid / evalgrid): what is PRINTED is what RUNS, one lever, both bases
# =================================================================================================================================================
needs_queue = pytest.mark.skipif(not QUEUE.exists(), reason="the v8F-mini package is not in this checkout")
win = pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")
GSTAGES = ("gategrid", "traingrid", "evalgrid")


def _queue_text():
    return QUEUE.read_text(encoding="utf-8").replace("\r\n", "\n")


def _block(label):
    return _queue_text().split(f"\n:{label}\n", 1)[1].split("\nexit /b 0\n", 1)[0].split("\n")


def _code(lines):
    return [x for x in lines if x.strip() and not x.lower().startswith("rem ")]


def _argv_of(line):
    return [t.strip('"') for t in re.findall(r'"[^"]*"|\S+', line.strip())]


def _dry(args, env_extra=None, clean_paths=False):
    env = dict(os.environ, V8F_DRYRUN="1")
    if clean_paths:
        for k in ("REPO", "RUNSM", "CACHEFULL", "V9", "PY"):
            env.pop(k, None)
    env.update(env_extra or {})
    r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8")
    return r.returncode, [x for x in r.stdout.splitlines() if ".py" in x]


def _without(line, drop):
    t, out, i = _argv_of(line), [], 0
    while i < len(t):
        if t[i] in drop:
            i += 2
            continue
        out.append(t[i])
        i += 1
    return out


@needs_queue
def test_queue_grid_the_queue_is_crlf_and_the_dose_is_defined_once_and_only_gridbase_reads_it():
    raw = QUEUE.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 400, "a .cmd with bare LF endings misparses labels and `exit /b`"
    q = _queue_text()
    assert q.count('\nset "PGTS=0.5"\n') == 1 and q.count('\nset "PGTE=0.0"\n') == 1 and len(re.findall(r'^set "PGT[SE]=', q, re.M)) == 2
    for label in GSTAGES:
        body = "\n".join(_code(_block(label)))
        assert "p-gt" not in body and "0.5" not in body, f"{label}: the dose (or any literal copy of it) lives in :gridbase only"
        assert "call :gridbase" in body
    gb = "\n".join(_code(_block("gridbase")))
    assert 'set "GLEV= --p-gt-start %PGTS% --p-gt-end %PGTE%"' in gb and "p-gt-start 0.5" not in gb
    for st in GSTAGES:
        assert f'if /i "%~1"=="{st}" goto :{st}' in q, "a stage nobody dispatches never runs"
    if READ_P.exists():
        assert R.DOSE["M2b-pgt"] == (float(re.search(r'^set "PGTS=(.*)"$', q, re.M).group(1)), float(re.search(r'^set "PGTE=(.*)"$', q, re.M).group(1)))
        ap = TR.build_parser()
        assert R.DOSE["M2b"] == (1.0, 0.2) == (ap.get_default("p_gt_start"), ap.get_default("p_gt_end")), "M2b's dose is the trainer's default"


@needs_queue
def test_queue_grid_runs_the_command_it_prints_static():
    """A dry run can only print: the real line is checked statically -- the trainer / eval appear ONCE per command, inside the one string, and the real run executes that string."""
    for label, var in (("gategrid", "GG"), ("traingrid", "TGR")):
        code = _code(_block(label))
        assert [x for x in code if "%TRAIN%" in x] == [x for x in code if x.startswith(f'set "{var}=')], label
        assert any(x.startswith(f'set "{var}="%PY%" "%TRAIN%" --arm M2b%GLEV% ') and "--grid pool8x20" in x for x in code), label
        assert f"if defined V8F_DRYRUN (echo %{var}% & exit /b 0)" in code, label
        assert sum(x.startswith(f"%{var}% ") for x in code) == 1, label
        assert not any("pool4x10" in x for x in code), f"{label}: a 4x10 literal in a grid stage"
    code = _code(_block("evalgrid"))
    assert [x for x in code if "%EVAL%" in x] == [x for x in code if re.match(r'set "E[0-3]=', x)], "the eval script appears only inside the four strings"
    assert len([x for x in code if re.match(r'set "E[0-3]=', x)]) == 4
    assert "if defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & exit /b 0)" in code
    for k in range(4):
        assert sum(x.strip().startswith(f"%E{k}% >> ") for x in code) == 1, f"E{k} is executed exactly once"
    e = {k: next(x for x in code if x.startswith(f'set "E{k}=')) for k in range(4)}
    assert "--project --families" in e[0] and "--tac" not in e[0]
    assert "--tac absmask" in e[1] and "--tac oracle" in e[2] and "--tac-groups" not in e[2]
    assert "--tac oracle --tac-groups all-latcons --tac-alpha 1" in e[3]
    for k in range(4):
        assert '--ckpt "%OUT%\\ckpt_last.pt"' in e[k] and "%EC% %ER%" in e[k] and "--grid" not in e[k], "the eval reads the grid FROM the checkpoint; no flag can disagree with it"
    # every failure is noticed on its own line (flat `if errorlevel 1 (... & exit /b 1)`, no parenthesised multi-line block: MEASURED to return rc 0 from `cmd /c`)
    for label in GSTAGES:
        body = _block(label)
        assert not any(x.rstrip().endswith("(") for x in body), f"{label}: a multi-line parenthesised block"
        assert sum(1 for x in body if x.startswith("if errorlevel 1 (echo [FAIL]") and x.rstrip().endswith("exit /b 1)")) >= (1 if label != "evalgrid" else 4)


@needs_queue
@win
@pytest.mark.parametrize("base", BASES)
def test_queue_grid_stages_dry_both_bases_one_lever(tmp_path, base):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    lev = ["--p-gt-start", "0.5", "--p-gt-end", "0.0"] if base == "M2b-pgt" else []
    rc, ln = _dry(f"gategrid {base}", env)
    assert rc == 0 and len(ln) == 1 and "--selfcheck-only" in ln[0] and f"gate_{base}-8x20" in ln[0], ln
    t = _argv_of(ln[0])
    assert t[t.index("--arm"):t.index("--arm") + 2 + len(lev)] == ["--arm", "M2b", *lev] and t[t.index("--grid") + 1] == "pool8x20", ln
    rc, ln = _dry(f"traingrid 1 {base}", env)
    assert rc == 0 and len(ln) == 1, ln
    t = _argv_of(ln[0])
    assert t[t.index("--arm"):t.index("--arm") + 2 + len(lev)] == ["--arm", "M2b", *lev] and t[t.index("--seed") + 1] == "1", ln
    assert t[t.index("--grid") + 1] == "pool8x20" and t.count("--grid") == 1 and t[t.index("--out") + 1].endswith(f"{base}_s1_8x20"), ln
    assert "--exclude 540:600" in ln[0] and "--kin-mode yaw" in ln[0] and "--steps-mult 4" in ln[0] and "--batch 64" in ln[0], ln
    assert "--w-cons" not in ln[0] and "--w-tac" not in ln[0] and "--w-wm" not in ln[0], "ONE lever: no other weight on the line"
    rc, ev = _dry(f"evalgrid 0 {base}", env)
    assert rc == 0 and len(ev) == 4, ev
    tg = f"{base}-8x20_s0"
    for k, (suffix, frag) in enumerate((("", "--project --families"), ("-tacabs", "--tac absmask"), ("-tacorc", "--tac oracle"),
                                        ("-tacorc-all-latcons", "--tac oracle --tac-groups all-latcons --tac-alpha 1"))):
        assert f"--tag {tg}{suffix} " in ev[k] + " " and frag in ev[k], (k, ev[k])
        assert f"{base}_s0_8x20{os.sep}ckpt_last.pt" in ev[k], "every eval reads THIS run's last checkpoint (which names its own grid)"
        assert f"{os.sep}{base}_s0{os.sep}" not in ev[k], "never B's checkpoint"
    # ONE lever, from the queue's own lines: G's gate / train line is B's line with the grid (and the output dir) changed and NOTHING else
    drop = {"--grid", "--out"}
    b_train = _dry("trainpgt 1" if base == "M2b-pgt" else "train M2b 1", env)[1][0]
    assert "--grid pool4x10" in b_train and _without(b_train, drop) == _without(_dry(f"traingrid 1 {base}", env)[1][0], drop), "G's train line = B's + the grid"
    b_gate = _dry("gatepgt" if base == "M2b-pgt" else "gatearm M2b", env)[1][0]
    assert _without(b_gate, drop) == _without(_dry(f"gategrid {base}", env)[1][0], drop), "G's gate line = B's + the grid"


@needs_queue
@win
def test_queue_grid_refuses_a_base_that_is_not_m2b_or_m2b_pgt_and_missing_arguments(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    for args in ("gategrid", "gategrid M2", "gategrid M2b-pgt-x", "traingrid", "traingrid 0", "traingrid 0 M2", "traingrid 0 pgt", "evalgrid 0", "evalgrid", "evalgrid 0 M2b-8x20"):
        rc, ln = _dry(args, env)
        assert rc == 2 and ln == [], (args, rc, ln)
    rc, ln = _dry("gategrid m2b-PGT", env)                  # cmd's comparison is case-insensitive: the stage and the base normalise the same way
    assert rc == 0 and len(ln) == 1 and "--p-gt-start 0.5" in ln[0]


@needs_queue
@win
def test_queue_grid_leaves_every_other_stage_untouched(tmp_path):
    """The edit only ADDS stages: the neighbouring arms' lines (train M2b, trainpgt, trainwc, gatearm) read as they did, and none of them mentions the 8x20 grid."""
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    for args in ("train M2b 0", "trainpgt 0", "trainwc 0", "gatearm M2b", "gatepgt"):
        rc, ln = _dry(args, env)
        assert rc == 0 and len(ln) >= 1 and all("8x20" not in x for x in ln), (args, ln)
    assert "--grid pool4x10" in _dry("train M2b 0", env)[1][0] and "--grid pool4x10" in _dry("trainpgt 0", env)[1][0]


def _fake_tree(tmp):
    """A repo whose trainer / eval are FAKES that record the argv they are handed (and make the files the next stage waits for)."""
    repo = tmp / "repo"
    sc = repo / "stack" / "scripts"
    sc.mkdir(parents=True)
    rec = tmp / "argv.jsonl"
    (sc / "v8f_mini_train.py").write_text(
        "import json, os, sys\n"
        f"open({str(rec)!r}, 'a').write(json.dumps({{'who': 'train', 'argv': sys.argv[1:]}}) + '\\n')\n"
        "a = sys.argv; out = a[a.index('--out') + 1]\n"
        "os.makedirs(out, exist_ok=True)\n"
        "if os.environ.get('FAKE_TRAIN') == 'fail':\n"
        "    sys.exit(3)\n"
        "if '--selfcheck-only' not in a and os.environ.get('FAKE_TRAIN') != 'nosummary':\n"
        "    open(os.path.join(out, 'summary.json'), 'w').write('{}')\n", encoding="utf-8")
    (sc / "v8f_mini_eval.py").write_text(
        "import json, os, sys\n"
        f"open({str(rec)!r}, 'a').write(json.dumps({{'who': 'eval', 'argv': sys.argv[1:]}}) + '\\n')\n"
        "a = sys.argv; d = a[a.index('--out-dir') + 1]; t = a[a.index('--tag') + 1]\n"
        "if os.environ.get('FAKE_FAIL_TAG') == t:\n"
        "    sys.exit(1)\n"
        "os.makedirs(d, exist_ok=True)\n"
        "open(os.path.join(d, 'eval_' + t + '.npz'), 'wb').write(b'x')\n", encoding="utf-8")
    hj = repo / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "raw" / "p10_yaw4512"
    hj.mkdir(parents=True)
    (hj / "p10_prior.pt").write_bytes(b"p")
    (hj / "p10_eval_pred6.npy").write_bytes(b"p")
    cf = tmp / "cf"
    for sub in ("train", "eval"):
        (cf / sub).mkdir(parents=True)
        (cf / sub / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (cf / "native_tracks.pt").write_bytes(b"t")
    (tmp / "runsm").mkdir()
    return repo, rec


@needs_queue
@win
@pytest.mark.parametrize("base", BASES)
def test_queue_grid_real_lines_execute_exactly_the_printed_commands(tmp_path, base):
    repo, rec = _fake_tree(tmp_path)
    env = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    env.pop("V8F_DRYRUN", None)

    def real(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=120, encoding="utf-8")
        assert r.returncode == 0, (args, r.stdout, r.stderr)
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines()] if rec.exists() else []

    def printed(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(env, V8F_DRYRUN="1"), capture_output=True, text=True, timeout=60, encoding="utf-8")
        return [_argv_of(x)[2:] for x in r.stdout.splitlines() if ".py" in x]       # drop python.exe and the script: sys.argv[1:]
    want_gate, want_train, want_eval = printed(f"gategrid {base}"), printed(f"traingrid 1 {base}"), printed(f"evalgrid 1 {base}")        # BEFORE the runs: a finished stage prints [skip]
    assert len(want_gate) == 1 and len(want_train) == 1 and len(want_eval) == 4
    got = real(f"gategrid {base}")
    assert [g["argv"] for g in got] == want_gate and got[-1]["who"] == "train" and "--selfcheck-only" in got[-1]["argv"]
    n = len(got)
    got = real(f"traingrid 1 {base}")[n:]
    assert [g["argv"] for g in got] == want_train and len(got) == 1, "the REAL traingrid line hands the trainer the printed command"
    a = got[0]["argv"]
    assert a[a.index("--grid") + 1] == "pool8x20" and a[a.index("--arm") + 1] == "M2b" and a[a.index("--seed") + 1] == "1" and ("--p-gt-start" in a) == (base == "M2b-pgt")
    assert (tmp_path / "runsm" / f"{base}_s1_8x20" / "summary.json").exists() and not (tmp_path / "runsm" / f"{base}_s1").exists(), "G has its own run dir; B's is never touched"
    n += 1
    got = real(f"evalgrid 1 {base}")[n:]
    assert [g["argv"] for g in got] == want_eval and len(got) == 4 and all(g["who"] == "eval" for g in got), "the REAL evalgrid lines hand the eval the printed commands"
    tags = [g["argv"][g["argv"].index("--tag") + 1] for g in got]
    assert tags == [f"{base}-8x20_s1", f"{base}-8x20_s1-tacabs", f"{base}-8x20_s1-tacorc", f"{base}-8x20_s1-tacorc-all-latcons"]
    n += 4                                              # a finished stage is skipped, not re-run
    assert len(real(f"evalgrid 1 {base}")) == n and len(real(f"traingrid 1 {base}")) == n


@needs_queue
@win
def test_queue_grid_real_stages_notice_failures_and_skip_exactly_what_is_finished(tmp_path):
    repo, rec = _fake_tree(tmp_path)
    runsm = tmp_path / "runsm"
    ev = runsm / "eval"
    base = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(runsm), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    base.pop("V8F_DRYRUN", None)

    def stage(args, **extra):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(base, **extra), capture_output=True, text=True, timeout=120, encoding="utf-8")
        return r.returncode, r.stdout

    def evals():
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines() if '"eval"' in x] if rec.exists() else []
    rc, out = stage("gategrid M2b", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] gate M2b-8x20" in out
    rc, out = stage("traingrid 1 M2b", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] train M2b_s1_8x20 - see" in out
    rc, out = stage("traingrid 1 M2b", FAKE_TRAIN="nosummary")                 # a trainer that exits 0 WITHOUT a summary (a crash that swallowed its code)
    assert rc == 1 and "exited 0 without summary.json" in out
    rc, out = stage("evalgrid 1 M2b")                                          # evaluation refuses a run that has not finished training
    assert rc == 2 and "has not finished training" in out and not evals()
    assert stage("traingrid 1 M2b")[0] == 0
    rc, out = stage("traingrid 1 M2b")
    assert rc == 0 and "[skip] M2b_s1_8x20" in out, "a finished run is not re-trained"
    tags = ["M2b-8x20_s1", "M2b-8x20_s1-tacabs", "M2b-8x20_s1-tacorc", "M2b-8x20_s1-tacorc-all-latcons"]
    for tag in tags:
        rc, out = stage("evalgrid 1 M2b", FAKE_FAIL_TAG=tag)
        assert rc == 1 and f"[FAIL] eval {tag} - see" in out, (tag, rc, out)
        for f in list(ev.glob("eval_M2b-8x20_s1*.npz")):
            f.unlink()
    ev.mkdir(exist_ok=True)
    for n, done in enumerate(([0, 2], [1], [0], [2], [3], [0, 1, 2], [1, 3])):
        for f in list(ev.glob("eval_M2b-8x20_s1*")):
            f.rename(f.with_name(f"old{n}_" + f.name))
        rec.write_text("", encoding="utf-8")
        for k in done:
            (ev / f"eval_{tags[k]}.npz").write_bytes(b"x")
        assert stage("evalgrid 1 M2b")[0] == 0, done
        ran = [g["argv"][g["argv"].index("--tag") + 1] for g in evals()]
        assert ran == [tg for k, tg in enumerate(tags) if k not in done], (done, ran)


# =================================================================================================================================================
# THE CHAIN (code/box2_ops/chain_v8f_grid.sh): needs box 2 and ssh, so it cannot RUN here -- its order, its gate assertions (RUN against synthetic selfchecks),
# its --dry-run (== the queue's own dry-run lines) and every load-bearing line are pinned
# =================================================================================================================================================
def _git_bash():
    for p in (os.environ.get("GIT_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files (x86)/Git/bin/bash.exe"):
        if p and Path(p).exists():
            return p
    return None


needs_chain = pytest.mark.skipif(not CHAIN.exists(), reason="the v8F-mini package is not in this checkout")


@needs_chain
def test_chain_grid_static_ordering_the_wait_and_no_second_copy_of_the_dose():
    c = CHAIN.read_text(encoding="utf-8")
    assert b"\r\n" not in CHAIN.read_bytes(), "a bash script with CRLF endings does not run"
    assert "p-gt-start 0.5" not in c and not re.search(r"\bPGTS=0\.5\b", c) and "PGTS=$(sed -n" in c and "PGTE=$(sed -n" in c and "defines the dose more than once" in c
    # the order: predecessor -> refresh -> BASE decided -> check-inputs on the REAL B side -> cache check -> guard -> gate -> seed 0 -> read s0 -> (CONTINUE) seed 1 -> final read
    order = ["waiting for the predecessor chain", "fetch -q origin", '--decide-base --pgt-dir', 'v8f_grid_read.py" --check-inputs', 'check_grid_cache.py" --train',
             "v8f_guard.sh > ", "queue gategrid 120 gategrid $BASE", "runone 0\n", "gridread s0 grid_read_s0.json", "runone 1\n", "gridread final grid_read.json"]
    idx = [c.index(x) for x in order]
    assert idx == sorted(idx) and len(set(idx)) == len(idx), dict(zip(order, idx))
    assert 'queue "t_$g" $TRAIN_POLLS traingrid $s $BASE' in c and 'queue "e_$g" $EVAL_POLLS evalgrid $s $BASE' in c
    assert '[ "$(jget grid_read_s0.json DECISION)" = CONTINUE ] || finish' in c, "a STOP at seed 0 is a committed outcome (DONE), not a crash"
    assert c.index('check_grid_cache.py" --train') < c.index("launch telemetry_long") < c.index("queue gategrid 120"), "the cache is refused while box 2 is still idle and untouched"
    # the predecessor is PREREG_V8F_PGT's chain, by default; its markers are the shared ones
    assert "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_pgt.out}" in c and "ZZV8FMINI-STOP" in c and "ZZV8FMINI-DONE" in c and "chain_v8f_grid.out" in c
    assert 'grep -qF "\'^(ZZV8FMINI|" $B/v8f_guard.sh' in c
    # the training wait covers the longest run the guard lets live: TRAIN_POLLS x 30 s >= V8F_MAX_RUN_H (default) hours
    polls = int(re.search(r"^TRAIN_POLLS=(\d+)", c, re.M).group(1))
    maxh = float(re.search(r"V8F_MAX_RUN_H:-(\d+)", (CHAIN.parent / "v8f_guard.sh").read_text(encoding="utf-8")).group(1))
    assert polls * 30 / 3600 >= maxh, f"a run the guard lets live ({maxh} h) outlasts the wait ({polls * 30 / 3600} h)"
    # the B side per base: PGT's own pulled runs/evals for M2b-pgt, the banked M2b files for M2b; the P10 guard file and the check-inputs come BEFORE any training
    for x in ('$PGTD/runs/M2b-pgt_s$s/$f', '$PGTD/eval/eval_M2b-pgt_s$s$suf.$x', "$L/v8f_mini/runs/M2b_s$s/$f", "$L/v8f_cz/eval/$f", "$L/v8f_tacgrp2/eval/$f", "floors_perwindow.npz"):
        assert x in c, x
    assert c.index("floors_perwindow.npz") < c.index('v8f_grid_read.py" --check-inputs') < c.index("queue gategrid 120")
    # what a G run pulls: its selfcheck/summary/argv/log/ckpt (the reader reads the checkpoint's grid) and the four evals
    assert "for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull" in c
    assert 'for f in "" -tacabs -tacorc -tacorc-all-latcons; do for x in json npz log; do pull "eval/eval_$tg$f.$x"' in c
    arch = next(x for x in c.splitlines() if "archive --format=tar" in x)
    assert all(x in arch for x in ("stack", "taniteval", '"$HJ"', '"$PK"', '"$B1"')), "the reader, the cache checker and the lead block ride in the archive"
    # the reader files the chain insists the tip has
    for f in ("v8f_grid_read.py", "v8f_pgt_read.py", "v8f_dk_read.py", "v8f_wc_read.py", "v8f_tac_groups_read.py", "v8f_mini_full_report.py", "check_grid_cache.py"):
        assert f in c


@needs_chain
def test_chain_grid_syntax_and_dry_run_prints_the_queue_lines_for_both_bases(tmp_path):
    bash = _git_bash()
    if bash is None or sys.platform != "win32":
        pytest.skip("Git Bash on Windows is needed to run the chain's --dry-run")
    r = subprocess.run([bash, "-n", str(CHAIN)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    env = dict(os.environ, V8F_QUEUE=str(QUEUE), TMPDIR=tmp_path.as_posix())
    env.pop("V8F_DRYRUN", None)
    r = subprocess.run([bash, str(CHAIN), "--dry-run"], env=env, capture_output=True, text=True, timeout=180, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    stages = ["gategrid M2b", "gategrid M2b-pgt", "traingrid 0 M2b", "traingrid 1 M2b", "traingrid 0 M2b-pgt", "traingrid 1 M2b-pgt",
              "evalgrid 0 M2b", "evalgrid 1 M2b", "evalgrid 0 M2b-pgt", "evalgrid 1 M2b-pgt"]
    assert [x for x in out if x.startswith("== ")] == [f"== {s}" for s in stages]
    want = []                                               # the chain's printed lines ARE the queue's own dry-run lines at box 2's default paths
    for st in stages:
        want += _dry(st, clean_paths=True)[1]
    assert [x for x in out if ".py" in x] == want and len(want) == 2 + 4 + 16
    t0 = next(x for x in out if "--seed 0" in x and "v8f_mini_train.py" in x and "--p-gt-start" in x)
    assert "--arm M2b --p-gt-start 0.5 --p-gt-end 0.0 --seed 0 --grid pool8x20" in t0 and "D:\\tanitad\\repo" in t0 and "M2b-pgt_s0_8x20" in t0
    t1 = next(x for x in out if "--seed 1" in x and "v8f_mini_train.py" in x and "--p-gt-start" not in x)
    assert "--arm M2b --seed 1 --grid pool8x20" in t1 and "M2b_s1_8x20" in t1


def _gate_assertions():
    """The python block the chain runs on the PULLED gate selfcheck.json (its heredoc), as a program."""
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"<<'EOF' \|\| stop \"gate G assertions\"\n(.*?)\nEOF\n", c, re.S)
    assert m, "the chain's gate assertions are a python heredoc whose failure stops the chain"
    return m.group(1)


@needs_chain
@pytest.mark.parametrize("base", BASES)
def test_chain_grid_gate_assertions_pass_the_g_gate_and_refuse_every_way_it_can_be_wrong(tmp_path, base):
    """The chain's gate block is RUN against a synthetic selfcheck.json: a good G gate passes; each single-field corruption exits 1 and names the broken field (the
    schedule, the clip list, the holdout, the weights, the step count and the PARAMETER DELTA cannot drift under a green gate)."""
    import copy
    body = _gate_assertions()
    hold, b_sha = "f6a1dee0" + "0" * 56, "c" * 64
    p0, p1 = (0.5, 0.0) if base == "M2b-pgt" else (1.0, 0.2)
    pb = {"adapter": 275712, "cond": 25088, "S": 1321737, "T": 1463868, "O": 2120216, "WM": 2007874}
    D = 120 * 256
    good = {"declared_vs_built": "PASS", "live": {"pass": True, "control_prior_grad_sum": 0.0}, "steps_total": 19164,
            "info": {"n_clips": 4512, "n_excluded": 60, "n_excluded_declared": 60, "excluded_in_train": 0, "excluded_sha256": hold, "train_sha256": b_sha, "kin_mode": "yaw", "lru": 43},
            "levers": {"wm": "b", "navonly": False, "consistency": True, "cons_absent": "mask", "p_gt_start": p0, "p_gt_end": p1,
                       "w": {"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1, "w_tac": 1.0, "w_cons": 0.5},
                       "params": dict(pb, adapter=pb["adapter"] + D), "n_params_trainable": 7214495 + D}}
    bsum = tmp_path / "b_summary.json"
    bsum.write_text(json.dumps({"train_sha256": b_sha, "levers": {"w": {"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1}, "params": pb, "n_params_trainable": 7214495}}), encoding="utf-8")

    def run(mod=None, args=None):
        sc = copy.deepcopy(good)
        if mod:
            mod(sc)
        f = tmp_path / "sc.json"
        f.write_text(json.dumps(sc), encoding="utf-8")
        a = args or ["0.5", "0.0", base]
        r = subprocess.run([sys.executable, "-", str(f), hold, "19164", str(bsum), *a], input=body, capture_output=True, text=True, encoding="utf-8", timeout=60)
        return r.returncode, r.stdout + r.stderr
    rc, out = run()
    assert rc == 0 and "PASS" in out and f"built p_gt {p0} -> {p1}" in out and f"params {7214495 + D}" in out, out
    L = lambda k, v: (lambda s: s["levers"].__setitem__(k, v))                   # noqa: E731
    W = lambda k, v: (lambda s: s["levers"]["w"].__setitem__(k, v))              # noqa: E731
    I = lambda k, v: (lambda s: s["info"].__setitem__(k, v))                     # noqa: E731
    P = lambda k, v: (lambda s: s["levers"]["params"].__setitem__(k, v))         # noqa: E731
    cases = (("declared_vs_built", lambda s: s.__setitem__("declared_vs_built", "FAIL"), "declared_vs_built"),
             ("live", lambda s: s["live"].__setitem__("pass", False), "live"),
             ("control grad", lambda s: s["live"].__setitem__("control_prior_grad_sum", 1.0), "live"),
             ("n_clips", I("n_clips", 4000), "clips / holdout counts"), ("n_excluded", I("n_excluded", 59), "clips / holdout counts"),
             ("n_excluded_declared", I("n_excluded_declared", 59), "clips / holdout counts"), ("excluded_in_train", I("excluded_in_train", 1), "clips / holdout counts"),
             ("holdout", I("excluded_sha256", "0" * 64), "not the RA-1 holdout"), ("train list", I("train_sha256", "d" * 64), "TRAIN list"),
             ("kin_mode", I("kin_mode", "steer"), "kin_mode"), ("lru unrecorded", lambda s: s["info"].pop("lru"), "info.lru"),
             ("wm", L("wm", "a"), "levers"), ("navonly", L("navonly", True), "levers"), ("consistency", L("consistency", False), "levers"), ("cons_absent", L("cons_absent", "zero"), "levers"),
             ("p_gt start", L("p_gt_start", p0 + 0.1), "built p_gt"), ("p_gt end", L("p_gt_end", p1 + 0.1), "built p_gt"),
             ("p_gt missing", lambda s: (s["levers"].pop("p_gt_start"), s["levers"].pop("p_gt_end")) and None, "built p_gt None -> None"),
             ("w_pen", W("w_pen", 0.6), "w_pen"), ("w_pen_self", W("w_pen_self", 0.3), "w_pen_self"), ("w_wm", W("w_wm", 2.0), "w_wm"),
             ("w_tac", W("w_tac", 2.0), "w_tac / w_cons"), ("w_cons", W("w_cons", 1.0), "w_tac / w_cons"), ("steps", lambda s: s.__setitem__("steps_total", 9000), "steps 9000"),
             ("one parameter too many", L("n_params_trainable", 7214495 + D + 1), "n_params_trainable"),
             ("no position table (a 4x10 build)", lambda s: (s["levers"].__setitem__("n_params_trainable", 7214495), P("adapter", pb["adapter"])(s)) and None, "n_params_trainable"),
             ("a second group grew", P("T", pb["T"] + 5), "parameter group T"),
             ("the table landed in the wrong group", lambda s: (P("adapter", pb["adapter"])(s), P("O", pb["O"] + D)(s)) and None, "parameter group"),
             ("params unrecorded", lambda s: s["levers"].pop("params"), ""))
    for name, mod, msg in cases:
        rc, out = run(mod)
        assert rc != 0 and msg in out, (name, rc, out)
    # the lever's OTHER base: a G built with the OTHER base's schedule is refused (the base decides what B's schedule is)
    other = (1.0, 0.2) if base == "M2b-pgt" else (0.5, 0.0)
    rc, out = run(lambda s: s["levers"].update(p_gt_start=other[0], p_gt_end=other[1]))
    assert rc == 1 and f"built p_gt {other[0]} -> {other[1]}, expected {p0} -> {p1}" in out, out
    # the dose comes from the arguments (the queue's PGTS / PGTE), not from the block
    if base == "M2b-pgt":
        rc, out = run(args=["0.6", "0.0", base])
        assert rc == 1 and "expected 0.6 -> 0.0" in out, out
        assert run(L("p_gt_start", 0.6), args=["0.6", "0.0", base])[0] == 0
    else:
        assert run(args=["0.6", "0.0", base])[0] == 0, "M2b's schedule is the trainer's, not the queue's PGTS / PGTE"


@needs_chain
def test_chain_grid_load_bearing_lines_are_present():
    """Every line whose loss would let a wrong run through is pinned. Substrings, not whole lines: line ends are not load-bearing."""
    c = CHAIN.read_text(encoding="utf-8")
    lines = (
        r"""PGTS=$(sed -n 's/^set "PGTS=\(.*\)"$/\1/p' $QT); PGTE=$(sed -n 's/^set "PGTE=\(.*\)"$/\1/p' $QT)""",
        r"""[ -n "$PGTS" ] && [ -n "$PGTE" ] || stop "the tip's queue defines no PGTS / PGTE""",
        r"""[ "$(grep -c '^set "PGTS=' $QT)" = 1 ] && [ "$(grep -c '^set "PGTE=' $QT)" = 1 ] || stop "the tip's queue defines the dose more than once""",
        r"""for st in gategrid traingrid evalgrid; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done""",
        r"""grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start""",
        r"""grep -q '"p_gt_end": cfg.p_gt_end' || stop "the tip's built_levers does not record p_gt_end""",
        r"""git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block""",
        r"""--decide-base --pgt-dir "$(cygpath -m $PGTD)" --out "$(cygpath -m $E/base.json)" || stop "the base B cannot be decided from the PGT reads in $PGTD""",
        r"""case "$BASE" in M2b|M2b-pgt) ;; *) stop "the decided base '$BASE' is neither M2b nor M2b-pgt";; esac""",
        r"""|| stop "the reader cannot use the $BASE side (check-inputs)""",
        r"""|| stop "box 2's cache does not carry 8x20 for every clip (check_grid_cache)""",
        r"""--n-train 4572""",
        r"""|| stop "grid read $1""",
        r"""$PY "$RD/v8f_grid_read.py" --stage $1 --base $BASE --pgt-dir""",
        r"""$HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $E/runs/${BASE}_s0/summary.json)" $PGTS $PGTE $BASE <<'EOF' || stop "gate G assertions""",
        r"""pull "gate_${BASE}-8x20/selfcheck.json" "$E/gate_${BASE}-8x20""",
        r"""stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; exit 1; }""",
        r"""echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)""",
        r"""[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present""",
        r"""until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do""",
        r"""grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped""",
        r"""cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo""",
        r"""for f in eval_M2b_s$s-tacorc-all-goal.npz eval_M2b-s$s-tacorc-all-latcons.npz""".replace("M2b-s$s", "M2b_s$s"),
        r"""cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz""",
        r"""[ "$(jget grid_read_s0.json DECISION)" = CONTINUE ] || finish "$G1 -- seed 1 not trained (committed)""",
        r"""finish "$G1 | $(jget grid_read.json reading)""",
        r"""BASE=$(jget base.json base)""",
        r"""g="${BASE}_s${1}_8x20" tg="${BASE}-8x20_s${1}""",
        r"""grep ZZGUARD-STOP $L/v8fgrid_guard.out 2>/dev/null && stop "guard stop""",
        r"""V8F_CHAIN_OUT=$L/chain_v8f_grid.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh""",
        r"""[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry""")
    for ln in lines:
        assert ln in c, f"missing from the chain: {ln!r}"


@needs_pkg
def test_reader_layout_and_constants_match_the_model_the_trainer_and_the_eval():
    """The reader's constants are independent copies of the PREREG's numbers; here they are tied to the objects that actually build and score the arms."""
    assert R.N_TOK_G == N_TOK["pool8x20"] == 160 and R.N_TOK_B == N_TOK["pool4x10"] == 40
    assert (R.GRID_FLAG, R.GRID_VAL, R.B_GRID_VAL) == ("--grid", "pool8x20", "pool4x10")
    assert tuple(R.BASES) == tuple(BASES) and R.MLAT_BAR == 0.125 and R.SUF == "_8x20"
    assert R.g_name("M2b-pgt") == "M2b-pgt-8x20" and R.g_dir("M2b-pgt", 1) == "M2b-pgt_s1_8x20" and R.g_dir("M2b", 0) == "M2b_s0_8x20"
    assert tuple(R.EVAL_SUFFIXES) == ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")
    # the latcons columns of the code: the model's own group table, and the number of constraint slots the M-LAT RMSE is taken over
    _dk, WC = R._modules()
    assert [WC.C0 + i for i in WC.GROUPS["latcons"]] == LATCONS == list(M.CODE_GROUPS["latcons"])
    assert WC.C0 == C0 and WC.N_CONS == NC
    # the trainer's grid choices and the eval's default are the ones the reader refuses to confuse
    src = (REPO / "stack" / "scripts" / "v8f_mini_eval.py").read_text(encoding="utf-8")
    assert 'grid=ck.get("grid", "pool4x10")' in src, "the eval reads the grid FROM the checkpoint, default 4x10 (v8f_mini_eval.py:268)"


@needs_pkg
def test_reader_dose_arms_are_named_for_the_runs_the_queue_writes():
    """The run directory names the reader looks for are exactly the ones the queue's `traingrid` creates (no second spelling to drift)."""
    q = _queue_text() if QUEUE.exists() else ""
    if q:
        assert 'set "OUT=%RUNSM%\\%GBASE%_s%~2_8x20"' in q and 'set "TG=%GBASE%-8x20_s%~2"' in q
    assert R.g_dir("M2b", 1) == "M2b_s1_8x20" and R.g_name("M2b") + "_s1" == "M2b-8x20_s1"


@needs_pkg
@pytest.mark.parametrize("base", BASES)
def test_reader_mechanism_needs_every_seed_not_any(tmp_path, no_known, base):
    """PRIMARY fails at both seeds (-0.10 m is under 2F). The latcons fall holds at seed 0 only -> FAIL-NO-MECHANISM and `mechanism` False (every seed, not any);
    with the fall at both seeds the same ADE reads FAIL-MECHANISM-OK."""
    rc, r = _go(_gworld(tmp_path / "one", base, d_ade=-0.10, err_g1=0.6))
    assert rc == 0 and r["rows"]["s0"]["judge"]["mlat"]["met"] is True and r["rows"]["s1"]["judge"]["mlat"]["met"] is False, r["reading"]
    assert r["VERDICT"] == "FAIL-NO-MECHANISM" and r["mechanism"] is False
    rc, r = _go(_gworld(tmp_path / "both", base, d_ade=-0.10))
    assert rc == 0 and r["VERDICT"] == "FAIL-MECHANISM-OK" and r["mechanism"] is True


# ---- round 2 (closing the survivors of the first mutation round) -------------------------------------------------------------------------------------
def _all_g_evals_other_grid(f):
    for s in (0, 1):
        for suf in ("", "-tacabs", "-tacorc", "-tacorc-all-latcons"):
            _npz_edit(_geval(f, s, suf + ".npz"), _shift_eid)


def _b_seed1_gt_block_differs(f):
    _npz_edit(f["eval"] / "eval_M2b_s1-tacorc-all-goal.npz", _gt_codes_off)


def _no_present_latcons_slot(f):
    """Every present constraint slot of the GT block is the same in all arms; zero the latcons columns of the GT block in B's two seeds AND G's oracle so no window has one."""
    def zero(z):
        z["tac_code"] = np.asarray(z["tac_code"]).copy()
        z["tac_code"][:, LATCONS] = 0.0
    for s in (0, 1):
        for tag in (f["B_RUN"](s), f"{f['base']}-8x20_s{s}"):
            for suf in ("-tacorc.npz", "-tacorc-all-goal.npz", "-tacabs.npz", "-tacorc-all-latcons.npz"):
                p = f["eval"] / f"eval_{tag}{suf}"
                if p.exists() and "tac_code" in np.load(p).files:
                    _npz_edit(p, zero)


MUT2 = {                               # name -> (mutator(f), a message fragment, the bases it is run for)
    "g train log missing": (lambda f: _drop(f, 0, "train_log.jsonl"), "train_log", BASES),
    "g evals (all four, both seeds) on another eval grid than B's": (_all_g_evals_other_grid, "same eval grid as", BASES[:1]),
    "b seed 1 carries other GT codes than seed 0": (_b_seed1_gt_block_differs, "GT constraint block differs", BASES[:1]),
}


@needs_pkg
@pytest.mark.parametrize("name", list(MUT2))
def test_reader_refuses_rc2_with_the_reason_round2(tmp_path, no_known, capsys, name):
    mut, frag, bases = MUT2[name]
    for base in bases:
        f = _gworld(tmp_path / base.replace("-", "_"), base)
        mut(f)
        capsys.readouterr()
        rc = R.main(["--stage", "final", *f["args"]])
        err = capsys.readouterr().err
        assert rc == 2 and frag in err, f"{name} [{base}]: rc {rc}; stderr: {err[-400:]}"
        assert not (f["d"] / "grid.json").exists()


@needs_pkg
def test_check_inputs_refuses_banked_gt_codes_that_differ_between_the_two_b_seeds(tmp_path, no_known, capsys):
    f = _gworld(tmp_path / "c", "M2b", seeds=())
    _b_seed1_gt_block_differs(f)
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2
    assert "GT constraint block differs" in capsys.readouterr().err


@needs_pkg
def test_reader_final_stage_checks_the_known_values_too_for_base_m2b(tmp_path, capsys):
    """Not only --check-inputs: the stage read itself runs PGT's known-value check (F, LAMBDA, S5 excess) on B = M2b, and a synthetic world cannot hold them."""
    f = _gworld(tmp_path / "m", "M2b")
    capsys.readouterr()
    assert R.main(["--stage", "final", *f["args"]]) == 2
    assert "REFUSED" in capsys.readouterr().err


@needs_pkg
@pytest.mark.parametrize("what", ["mse", "fall"])
def test_identity_control_covers_m_lat_s_own_cells(tmp_path, no_known, capsys, monkeypatch, what):
    """B against itself must read exactly +0 [0, 0] on M-LAT's paired MSE and exactly 0 on both latcons-RMSE falls: a mlat_row that returns anything else for
    (B, B) stops the check-inputs and the read."""
    f = _gworld(tmp_path / "c", "M2b", seeds=())
    real = R.mlat_row

    def bent(pf, A, Bm, *a, **k):
        out = real(pf, A, Bm, *a, **k)
        if A is Bm:
            if what == "mse":
                out["mse_delta"] = dict(out["mse_delta"], delta=1e-9)
            else:
                out["fall"] = 1e-6
        return out
    monkeypatch.setattr(R, "mlat_row", bent)
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2
    assert "IDENTITY CONTROL FAILED" in capsys.readouterr().err


@needs_pkg
def test_mlat_is_undefined_not_zero_when_no_window_carries_a_present_latcons_slot(tmp_path, no_known, capsys):
    f = _gworld(tmp_path / "w", "M2b")
    _no_present_latcons_slot(f)
    capsys.readouterr()
    assert R.main(["--stage", "final", *f["args"]]) == 2
    assert "REFUSED" in capsys.readouterr().err


@needs_pkg
def test_reader_refuses_a_nan_in_ts_latcons_slots(tmp_path, no_known, capsys):
    """T's NaN would make G's per-window latcons MSE NaN where B's is finite: the present-slot sets differ and M-LAT is refused, never averaged around."""
    f = _gworld(tmp_path / "w", "M2b")

    def nan_slot(z):
        z["tac_code"] = np.asarray(z["tac_code"]).copy()
        i, j = np.argwhere(z["tac_code"][10:, LATCONS] != 0.0)[0]                    # past the first 5 windows (oracle fallbacks carry no label)
        z["tac_code"][10 + i, LATCONS[j]] = np.nan
    _npz_edit(_geval(f, 0, "-tacabs.npz"), nan_slot)
    capsys.readouterr()
    rc = R.main(["--stage", "final", *f["args"]])
    assert rc == 2 and "present-slot sets differ" in capsys.readouterr().err, "a NaN in T's code must not be scored"


@needs_queue
@win
def test_queue_grid_refuses_an_empty_seed_with_a_valid_base(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    for args in ('traingrid "" M2b', 'evalgrid "" M2b-pgt'):
        rc, ln = _dry(args, env)
        assert rc == 2 and ln == [], (args, rc, ln)


@needs_pkg
def test_mlat_bar_is_undefined_when_b_s_latcons_error_is_exactly_zero(tmp_path, no_known, capsys):
    """A relative fall of 12.5 % of nothing is not a bar: B's present-slot latcons RMSE of exactly 0 is refused, not divided by."""
    f = _gworld(tmp_path / "w", "M2b", err_b=0.0, err_g=0.0)
    capsys.readouterr()
    assert R.main(["--stage", "final", *f["args"]]) == 2
    assert "non-positive B RMSE" in capsys.readouterr().err
