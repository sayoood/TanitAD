"""Tests of tok_probe_g.py (card PREREG_V8F_PROOF_ADD41, controls g1 to g6, the grid lever, the pod script). CPU, synthetic worlds of the tree, no pod, no GPU.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider test_tok_probe_g.py --basetemp F:/TanitAD-scratch/v8proof/finegrid/tmp/pt/<new name>
The tool under test is ``tok_probe_g.py`` next to this file, or the file named by the environment variable TOK_PROBE_G_FILE (run_mutations_g.py points it at a mutated copy).
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
TREE = os.environ.get("FINEGRID_TREE", "F:/TanitAD-scratch/v8proof/finegrid/tmp/tree_leadsup")
READ = "F:/TanitAD-scratch/v8proof/spatial/leadsup/read"
BANKED = Path("F:/TanitAD-scratch/v8proof/tokprobe/tok_probe.py")
TOKFILE = Path(os.environ.get("TOK_PROBE_G_FILE", str(HERE / "tok_probe_g.py")))
for p in (READ,):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(4)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tp = load_module(TOKFILE, "tok_probe_g_under_test")
bk = load_module(BANKED, "tok_probe_banked")                                   # the banked tool of card ADD27, never edited
T = tp.use_tree(TREE)
bk.use_tree(TREE)
M, D, TR = T["M"], T["D"], T["TR"]
_tf = os.path.abspath(sys.modules["tanitad"].__file__).replace(os.sep, "/")
assert _tf.startswith(TREE.replace(os.sep, "/")), _tf                           # tanitad comes from the intended tree


# ---------------------------------------------------------------------------------------------- a synthetic world that also carries the 8 by 20 grid
def add_fine(root, independent=False, noise=0.05, seed=7):
    """Add ``pool8x20`` [T, 160, 1024] fp8 to every clip file: each 4 by 10 token repeated into its 2 by 2 block plus a little noise (row-major), or independent random tokens."""
    rng = np.random.default_rng(seed)
    for sp in ("train", "eval"):
        for f in sorted((Path(root) / sp / "clips").glob("*.pt")):
            d = torch.load(f)
            x = d["pool4x10"].float()
            Tn = x.shape[0]
            if independent:
                y = torch.as_tensor(rng.normal(size=(Tn, 160, 1024)).astype(np.float32))
            else:
                y = x.reshape(Tn, 4, 10, 1024).repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(Tn, 160, 1024)
                y = y + torch.as_tensor(rng.normal(0, noise, size=y.shape).astype(np.float32))
            d["pool8x20"] = y.to(torch.float8_e4m3fn)
            torch.save(d, f)


_orig_world = D.make_synthetic_world


def _world_with_fine(root, *a, **k):
    out = _orig_world(root, *a, **k)
    add_fine(root)
    return out


D.make_synthetic_world = _world_with_fine

SYN_ARGV = ("--arm M2b --seed 0 --grid pool4x10 --wm-state grid --wm-cell-dim 16 --lead-sup on --w-lead-sup 1.0 --lead-sup-detach on --synthetic --kin-mode yaw "
            "--batch 4 --steps-mult 1 --lr 3e-3 --warmup 20")


def _args(out, grid="pool4x10"):
    return TR.build_parser().parse_args(tp.trainer_argv(SYN_ARGV, str(out), grid))


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    a = _args(out, "pool8x20")
    cfg = TR.make_config(a)
    ds, wins, S, total, info = tp.build_training_data(a, cfg)
    idx = np.random.default_rng(0).choice(len(wins), 4, replace=False)
    batch = tp.to_device(D.collate([ds.item(*wins[i]) for i in idx]), torch.device("cpu"))
    return {"a": a, "cfg": cfg, "ds": ds, "wins": wins, "S": S, "total": total, "batch": batch, "info": info}


@pytest.fixture(scope="module")
def run160(tmp_path_factory):
    """A real short run through the CLI path on the 160-token synthetic world: two F arms (W 16 and W 64), grid pool8x20."""
    out = tmp_path_factory.mktemp("run160")
    rc = tp.main(["train", "--grid", "pool8x20", "--tree", TREE, "--out", str(out), "--trainer-argv", SYN_ARGV, "--arms", "F16_s0,F64_s1", "--steps", "120", "--device", "cpu", "--threads", "4"])
    return {"out": out, "rc": rc}


def fresh_module(mutate=False):
    lines = TOKFILE.read_text(encoding="utf-8").split("\n")
    hit = [i for i, ln in enumerate(lines) if ln.rstrip().endswith("# LOSS-LINE")]
    assert len(hit) == 1                                                       # the ONE line a mutation removes
    if mutate:
        ind = lines[hit[0]][:len(lines[hit[0]]) - len(lines[hit[0]].lstrip())]
        lines[hit[0]] = ind + "loss = 0.0 * out.sum()"
    mod = types.ModuleType("tp_mut")
    mod.__file__ = str(TOKFILE)
    exec(compile("\n".join(lines), str(TOKFILE), "exec"), mod.__dict__)
    mod.use_tree(TREE)
    return mod


def planted_batch(n_tok, seed=0, B=8, F=8, d=1024):
    rng = np.random.default_rng(seed)
    feats = torch.as_tensor(rng.normal(size=(B, F, n_tok, d)).astype(np.float32)).half()
    pres = torch.ones(B, F)
    gap = torch.as_tensor(rng.uniform(15, 45, size=(B, F)).astype(np.float32))
    clos = torch.as_tensor(rng.uniform(-3, 3, size=(B, F)).astype(np.float32))
    return {"feats": feats, "frame_ok": torch.ones(B, F, dtype=torch.bool), "lead_pres": pres, "lead_gapm": gap, "lead_clos": clos}


def p1_pass(mod, arm, grid, n_tok, seed=0):
    batch = planted_batch(n_tok, seed)
    p = mod.build_probe(arm, grid=grid)
    opt = mod.make_opt(p, 3e-4, 1e-2)
    _, _, gs = mod.step_probe(p, opt, batch, 3e-4, 1.0, 1.0, want_g=True)
    return gs["g_tok"] > 0 and gs["g_head"] > 0 and gs["g_ctrl"] == 0.0, gs


# ================================================================================================ g1: the default path is unchanged
def _ck(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_g1_default_grid_gives_bit_identical_probe_weights_to_the_banked_tool(tmp_path):
    out_b, out_g, out_d = tmp_path / "banked", tmp_path / "g", tmp_path / "g_nodefault"
    common = ["--tree", TREE, "--trainer-argv", SYN_ARGV, "--steps", "30", "--device", "cpu", "--threads", "4"]
    assert bk.main(["train", "--out", str(out_b)] + common) in (0, 2)
    assert tp.main(["train", "--grid", "pool4x10", "--out", str(out_g)] + common) in (0, 2)
    assert tp.main(["train", "--out", str(out_d)] + common) in (0, 2)             # the default is pool4x10
    for arm in bk.ARMS_DEFAULT:
        cb = _ck(out_b / arm / "ckpt_last.pt")
        for other in (out_g, out_d):
            cg = _ck(other / arm / "ckpt_last.pt")
            assert cb["state_dict"].keys() == cg["state_dict"].keys()
            assert all(torch.equal(cb["state_dict"][k], cg["state_dict"][k]) for k in cb["state_dict"]), arm
            assert cg["grid"] == "pool4x10" and cg["n_tok"] == 40 and cg["n_tok_seen"] == 40
    # the train logs carry the same numbers (the step-by-step losses), not only the final weights
    for arm in bk.ARMS_DEFAULT:
        lb = [json.loads(x) for x in open(out_b / arm / "train_log.jsonl", encoding="utf-8")][1:]
        lg = [json.loads(x) for x in open(out_g / arm / "train_log.jsonl", encoding="utf-8")][1:]
        assert [r["lead_sup"] for r in lb] == [r["lead_sup"] for r in lg]
    # extract on the banked checkpoint: the states and lead outputs of the two tools are equal array by array
    cache = out_b / "_synthetic"
    for name, mod, kw in (("b", bk, []), ("g", tp, ["--grid", "pool4x10"])):
        assert mod.main(["extract", "--tree", TREE, "--run-dir", str(out_b), "--arm", "TK16_s0", "--cache", str(cache), "--states-dir", str(tmp_path / ("st_" + name)),
                         "--eval-range", "0:2", "--train-range", "0:3", "--device", "cpu"] + kw) == 0
    for sp in ("eval", "train"):
        for suffix in ("", "_lead"):
            a = np.load(tmp_path / "st_b" / ("TK16_s0_%s%s.npz" % (sp, suffix)))
            b = np.load(tmp_path / "st_g" / ("TK16_s0_%s%s.npz" % (sp, suffix)))
            assert a.files == b.files and all(np.array_equal(a[k], b[k], equal_nan=(a[k].dtype.kind == "f")) for k in a.files), (sp, suffix)


# ================================================================================================ g4: gradients at 160 tokens (and the mutation of the loss line)
def test_g4_gradients_reach_the_probe_at_160_tokens_and_the_control_reads_exactly_zero():
    for arm in ("F16_s0", "F64_s1"):
        ok, gs = p1_pass(tp, arm, "pool8x20", 160)
        assert ok and gs["g_tok"] > 0 and gs["g_head"] > 0 and gs["g_ctrl"] == 0.0, (arm, gs)
    p = tp.build_probe("F64_s1")
    p.train()
    b = planted_batch(160)
    T_ = M.lead_sup_terms(p(b["feats"]), b, 5)
    T_["lead_sup"].backward()
    assert p.ctrl.grad is None and all(q.grad is not None and float(q.grad.abs().sum()) > 0 for q in p.head.parameters()) and p.head.frame.in_features == 160 * 64


def test_g4_the_mutation_that_removes_the_loss_line_makes_the_check_fail_at_160_tokens():
    ok, gs = p1_pass(fresh_module(mutate=True), "F16_s0", "pool8x20", 160)
    assert not ok and gs["g_tok"] == 0.0 and gs["g_head"] == 0.0
    assert p1_pass(fresh_module(), "F16_s0", "pool8x20", 160)[0]                # the unmutated one passes in the same breath


# ================================================================================================ g5: declared against built
def test_g5_batch_shape_refusals_and_arm_grid_refusals():
    ok = torch.zeros(4, 16, 160, 1024)
    assert tp.check_batch_shape(ok, 160, 4) == 160
    for bad, n, b in ((torch.zeros(4, 16, 40, 1024), 160, 4), (torch.zeros(4, 16, 160, 512), 160, 4), (torch.zeros(4, 160, 1024), 160, 4), (torch.zeros(8, 16, 160, 1024), 160, 4),
                      (torch.zeros(4, 3, 160, 1024), 160, 4), (torch.zeros(4, 16, 160, 1024), 40, 4)):
        with pytest.raises(SystemExit) as e:
            tp.check_batch_shape(bad, n, b)
        assert "REFUSED" in str(e.value)
    assert tp.check_batch_shape(torch.zeros(2, 16, 40, 1024), 40, 4) == 40       # a short last batch of a chunk is fine
    assert tp.parse_arm("F16SH_s0") == {"W": 16, "shuffle": True, "seed": 0, "family": "F", "grid": "pool8x20"}
    assert tp.parse_arm("TK64_s1")["grid"] == "pool4x10" and tp.parse_arm("F64_s1")["W"] == 64
    for bad in ("X16_s0", "F16_0", "TK16", "F_s0", "FTK16_s0"):
        with pytest.raises(SystemExit):
            tp.parse_arm(bad)
    for arm, grid in (("F16_s0", "pool4x10"), ("F64_s1", "pool4x10"), ("TK16_s0", "pool8x20"), ("TK64_s1", "pool8x20"), ("F16SH_s0", "pool4x10"), ("TK16SH_s0", "pool8x20")):
        with pytest.raises(SystemExit) as e:
            tp.check_arm_grid(arm, grid)
        assert "REFUSED" in str(e.value)
        with pytest.raises(SystemExit):
            tp.build_probe(arm, grid=grid)
    with pytest.raises(SystemExit):
        tp.check_arm_grid("F16_s0", "pool16x40")
    tp.check_arm_grid("F16_s0", "pool8x20")
    tp.check_arm_grid("TK16_s0", "pool4x10")


def test_g5_a_stream_with_the_wrong_token_count_is_refused_in_training(tmp_path):
    wrong = iter([planted_batch(40, i) for i in range(4)])                       # 40 tokens under an 8 by 20 run
    with pytest.raises(SystemExit) as e:
        tp.run_training(wrong, ["F16_s0"], 4, lambda s: 3e-3, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(tmp_path / "a"), grid="pool8x20")
    assert "REFUSED" in str(e.value) and "160" in str(e.value)
    wrong2 = iter([planted_batch(160, i) for i in range(4)])                     # 160 tokens under a 4 by 10 run
    with pytest.raises(SystemExit) as e:
        tp.run_training(wrong2, ["TK16_s0"], 4, lambda s: 3e-3, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(tmp_path / "b"), grid="pool4x10")
    assert "REFUSED" in str(e.value)
    with pytest.raises(SystemExit):                                             # an F arm handed to a 4 by 10 run
        tp.run_training(iter([planted_batch(40, 0)]), ["F16_s0"], 1, lambda s: 3e-3, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(tmp_path / "c"), grid="pool4x10")
    good = tp.run_training(iter([planted_batch(160, i) for i in range(3)]), ["F16_s0"], 3, lambda s: 3e-3, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(tmp_path / "d"), grid="pool8x20")
    assert good["F16_s0"].n_tok == 160
    ck = _ck(tmp_path / "d" / "F16_s0" / "ckpt_last.pt")
    assert ck["n_tok"] == 160 and ck["n_tok_seen"] == 160 and ck["grid"] == "pool8x20"


def test_g5_the_cli_run_on_the_160_token_world_records_160_tokens_everywhere(run160):
    out = run160["out"]
    assert run160["rc"] in (0, 2) and (out / "TRAIN_DONE.json").exists()
    for arm, W in (("F16_s0", 16), ("F64_s1", 64)):
        ck = _ck(out / arm / "ckpt_last.pt")
        assert ck["grid"] == "pool8x20" and ck["n_tok"] == 160 and ck["n_tok_seen"] == 160 and ck["W"] == W
        assert ck["state_dict"]["head.frame.weight"].shape[1] == 160 * W and ck["state_dict"]["tok.0.weight"].shape == (128, 1024)
        hdr = json.loads(open(out / arm / "train_log.jsonl", encoding="utf-8").readline())
        assert hdr["grid"] == "pool8x20" and hdr["n_tok"] == 160
    meta = json.load(open(out / "train_meta.json", encoding="utf-8"))
    assert meta["grid"] == "pool8x20" and meta["n_tok"] == 160 and meta["host_est"]["n_tok"] == 160 and meta["host_est"]["total_gb"] < tp.HOST_GB_MAX
    assert "--grid" in meta["trainer_argv"] and meta["trainer_argv"][meta["trainer_argv"].index("--grid") + 1] == "pool8x20"
    done = json.load(open(out / "TRAIN_DONE.json", encoding="utf-8"))
    assert done["grid"] == "pool8x20" and done["n_tok"] == 160 and sorted(done["arms"]) == ["F16_s0", "F64_s1"]


def test_cli_refusals_of_arm_and_grid(tmp_path):
    base = ["train", "--tree", TREE, "--trainer-argv", SYN_ARGV, "--steps", "4", "--device", "cpu"]
    with pytest.raises(SystemExit) as e:
        tp.main(base + ["--grid", "pool4x10", "--arms", "F16_s0", "--out", str(tmp_path / "a")])
    assert "REFUSED" in str(e.value) and "F16_s0" in str(e.value)
    with pytest.raises(SystemExit) as e:
        tp.main(base + ["--grid", "pool8x20", "--arms", "TK16_s0", "--out", str(tmp_path / "b")])
    assert "REFUSED" in str(e.value)
    with pytest.raises(SystemExit) as e:
        tp.main(base + ["--grid", "pool8x20", "--arms", "F16_s0,TK16_s0", "--out", str(tmp_path / "c")])
    assert "REFUSED" in str(e.value)
    assert not (tmp_path / "a" / "TRAIN_DONE.json").exists()


def test_the_default_arm_names_follow_the_grid_and_the_shuffle_arm_is_named_F16SH(tmp_path):
    assert tp.ARMS_BY_GRID["pool8x20"] == ("F16_s0", "F16_s1", "F64_s0", "F64_s1") and tp.ARMS_BY_GRID["pool4x10"] == tp.ARMS_DEFAULT
    out = tmp_path / "sh"
    rc = tp.main(["train", "--grid", "pool8x20", "--tree", TREE, "--out", str(out), "--trainer-argv", SYN_ARGV, "--steps", "6", "--shuffle-closing-labels", "--device", "cpu"])
    assert rc in (0, 2) and (out / "F16SH_s0" / "ckpt_last.pt").exists() and json.load(open(out / "train_meta.json", encoding="utf-8"))["shuffle_closing"]["n_shuffled"] > 0
    assert _ck(out / "F16SH_s0" / "ckpt_last.pt")["n_tok_seen"] == 160


def test_host_memory_estimate_scales_with_the_token_count_and_is_refused_above_the_cap(syn, monkeypatch):
    h160 = syn["info"]["host_est"]
    a40 = _args(syn["a"].out, "pool4x10")
    ds, wins, S, total, info40 = tp.build_training_data(a40, TR.make_config(a40))
    h40 = info40["host_est"]
    assert h160["n_tok"] == 160 and h40["n_tok"] == 40 and abs(h160["prefetch_gb"] / h40["prefetch_gb"] - 4.0) < 1e-9
    assert h160["lru_clips"] <= h40["lru_clips"] and h160["clip_store_gb"] <= 1.6 + 1e-6                  # the clip LRU shrinks with the tokens, the store stays about 1.5 GB
    monkeypatch.setattr(tp, "HOST_GB_MAX", 0.001)
    with pytest.raises(SystemExit) as e:
        tp.build_training_data(syn["a"], syn["cfg"])
    assert "REFUSED" in str(e.value)


# ================================================================================================ g6: forward path, noise labels, p5
def test_g6_the_forward_takes_the_token_array_only_and_labels_cannot_change_outputs_at_160_tokens():
    assert list(inspect.signature(tp.TokenProbe.forward).parameters) == ["self", "tokens"]
    p = tp.build_probe("F16_s0").eval()
    b = planted_batch(160)
    with torch.no_grad():
        o1 = p(b["feats"])
        b2 = dict(b)
        rng = torch.Generator().manual_seed(4)
        for k in ("lead_pres", "lead_gapm", "lead_clos"):
            b2[k] = torch.randn(b[k].shape, generator=rng) * 100
        o2 = p(b2["feats"])
    assert torch.equal(o1, o2)
    with pytest.raises(TypeError):
        p(b["feats"], b["lead_gapm"])


def test_g6_p5_reads_the_short_160_token_run_and_a_rising_loss_goes_red(run160, tmp_path):
    out = run160["out"]
    for arm in ("F16_s0", "F64_s1"):
        r = tp.check_log(str(out / arm), window=60)
        assert r["pass"], r
        assert r["min_n_gap"] > 0 and r["last_step"] == 120
        recs = [json.loads(ln) for ln in open(out / arm / "train_log.jsonl", encoding="utf-8")][1:]
        first = [r_ for r_ in recs if r_["first"]][0]
        assert first["g_tok"] > 0 and first["g_head"] > 0 and first["g_ctrl"] == 0.0 and all(r_["g_ctrl"] == 0.0 for r_ in recs)

    def wlog(path, losses):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"header": True, "arm": "F16_s0", "total": 3000}) + "\n")
            for i, l in enumerate(losses):
                fh.write(json.dumps({"step": 50 * (i + 1), "first": False, "lead_sup_mean": l, "n_gap": 10.0, "n_gap_min_interval": 10.0}) + "\n")
    good = np.linspace(2.0, 1.0, 60)
    wlog(tmp_path / "a.jsonl", good)
    assert tp.check_log(str(tmp_path / "a.jsonl"))["pass"]
    wlog(tmp_path / "b.jsonl", good[::-1])
    assert not tp.check_log(str(tmp_path / "b.jsonl"))["pass_lead_term_falls"]
    wlog(tmp_path / "c.jsonl", good[:40])
    assert not tp.check_log(str(tmp_path / "c.jsonl"))["pass_complete"]


def test_the_two_f_arms_differ_see_the_same_batches_and_the_same_seed_repeats(tmp_path, monkeypatch):
    seen = {}
    orig = tp.step_probe

    def spy(probe, opt, batch, *a, **k):
        seen.setdefault(id(probe), []).append(hashlib.sha256(batch["feats"].numpy().tobytes()).hexdigest()[:12])
        return orig(probe, opt, batch, *a, **k)

    monkeypatch.setattr(tp, "step_probe", spy)
    b = planted_batch(160, 3, B=4)
    arms = ["F16_s0", "F16_s1"]
    probes = tp.run_training(iter([dict(b) for _ in range(4)]), arms, 4, lambda s: 3e-3, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(tmp_path), grid="pool8x20")
    hs = list(seen.values())
    assert len(hs) == 2 and hs[0] == hs[1] and len(hs[0]) == 4
    assert not torch.equal(probes["F16_s0"].tok[0].weight, probes["F16_s1"].tok[0].weight)
    again = tp.run_training(iter([dict(b) for _ in range(4)]), ["F16_s0"], 4, lambda s: 3e-3, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(tmp_path / "again"), grid="pool8x20")
    assert torch.equal(again["F16_s0"].tok[0].weight, probes["F16_s0"].tok[0].weight)
    assert probes["F16_s0"].head.frame.in_features == 2560 and probes["F16_s0"].n_tok == 160


# ================================================================================================ g3: planted tokens at 160 tokens
F_, B_ = 8, 8


def planted(seed, n_clips, T_=24, d=1024, n_tok=160):
    rng = np.random.default_rng(seed)
    uv = np.random.default_rng(999)
    U = uv.normal(size=(n_tok, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = uv.normal(size=(n_tok, d)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    clips = []
    for _ in range(n_clips):
        present = rng.random() < 0.85
        g0, c = rng.uniform(15, 45), rng.uniform(-3, 3)
        gap = g0 + c * 0.2 * np.arange(T_)
        tok = rng.normal(size=(T_, n_tok, d)).astype(np.float32)
        if present:
            tok += 6.0 * ((gap - 30.0) / 10.0)[:, None, None] * U[None] + 6.0 * (c / 3.0) * V[None]
        clips.append({"tok": tok.astype(np.float16), "pres": np.full(T_, float(present), np.float32), "gap": np.where(present, gap, np.nan).astype(np.float32),
                      "clos": np.full(T_, c if present else np.nan, np.float32)})
    return clips


def batches_of(clips, seed):
    rng = np.random.default_rng(seed)
    while True:
        feats, pres, gap, clos = [], [], [], []
        for _ in range(B_):
            c = clips[int(rng.integers(len(clips)))]
            s = int(rng.integers(0, c["tok"].shape[0] - F_ + 1))
            feats.append(c["tok"][s:s + F_]); pres.append(c["pres"][s:s + F_]); gap.append(c["gap"][s:s + F_]); clos.append(c["clos"][s:s + F_])
        yield {"feats": torch.as_tensor(np.stack(feats)), "frame_ok": torch.ones(B_, F_, dtype=torch.bool), "lead_pres": torch.as_tensor(np.stack(pres)),
               "lead_gapm": torch.as_tensor(np.stack(gap)), "lead_clos": torch.as_tensor(np.stack(clos))}


def r2(y, p):
    return 1.0 - float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def heldout_r2(probe, clips):
    ys_g, ps_g, ys_c, ps_c = [], [], [], []
    with torch.no_grad():
        for c in clips:
            x = torch.as_tensor(c["tok"]).float()
            for s in range(0, x.shape[0] - F_ + 1, 2):
                o = probe(x[s:s + F_][None])[0]
                lab = slice(s + 4, s + F_)
                ok = np.isfinite(c["gap"][lab])
                ys_g += list(c["gap"][lab][ok]); ps_g += list(o[:, 1].numpy()[ok] * 50.0)
                ys_c += list(c["clos"][lab][ok]); ps_c += list(o[:, 2].numpy()[ok] * 10.0)
    return r2(np.array(ys_g), np.array(ps_g)), r2(np.array(ys_c), np.array(ps_c))


STEPS = 300


def test_g3_planted_tokens_at_160_tokens_are_read_back_above_0_9_on_held_out_clips(tmp_path):
    train, held = planted(1, 16), planted(2, 8)
    lr_at = TR.LRSchedule(3e-3, 50, STEPS)
    pr = tp.run_training(batches_of(train, 1), ["F16_s0"], STEPS, lr_at, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(tmp_path / "g3"), grid="pool8x20")["F16_s0"].eval()
    rg, rc = heldout_r2(pr, held)
    print("G3 held-out R2 at 160 tokens: gap %.4f closing %.4f" % (rg, rc))
    assert rg > 0.9, (rg, rc)
    assert pr.n_tok == 160 and pr.head.frame.in_features == 160 * 16


# ================================================================================================ extract at 160 tokens and the reader's record checks
def test_extract_at_160_tokens_writes_the_layouts_records_the_grid_and_the_reader_loads_them(run160, tmp_path, monkeypatch):
    import leadsup_read as LR
    import pod_lead_apply as PA
    root = run160["out"] / "_synthetic"
    st = tmp_path / "states"
    ckpt = run160["out"] / "F16_s0" / "ckpt_last.pt"
    argv = ["extract", "--grid", "pool8x20", "--tree", TREE, "--run-dir", str(run160["out"]), "--arm", "F16_s0", "--cache", str(root), "--states-dir", str(st), "--eval-range", "0:2", "--train-range", "0:6",
            "--device", "cpu"]
    assert tp.main(argv) == 0
    files = LR.arm_files(str(st), "F16_s0")
    rows, ck_sha = LR.check_records(files, plumbing=False, do_train=True)
    assert rows["pass"], rows
    assert ck_sha == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    for sp in ("eval", "train"):
        S = LR.load_states(files[sp + "_states"])
        assert S["Z"].dtype == np.float16 and S["Z"].shape[1] == 160 * 16 and S["clip_idx"].dtype == np.int32 and S["t"].dtype == np.int16
        n = 2 * 60 if sp == "eval" else 6 * 60
        assert S["Z"].shape[0] == n and len(S["clips"]) == (2 if sp == "eval" else 6)
        rec = json.load(open(files[sp + "_states"] + ".json", encoding="utf-8"))
        assert rec["grid"] == "pool8x20" and rec["n_tok"] == 160 and rec["n_tok_seen_in_training"] == 160 and rec["D"] == 2560 and rec["W"] == 16
        lead = np.load(files[sp + "_lead"], allow_pickle=False)
        head = LR.load_head(files[sp + "_lead"] + ".head.npz")
        idx, ok = PA.history_rows(S["clip_idx"], S["t"], head["hist"])
        idx2, ok2 = tp.history_rows(S["clip_idx"], S["t"], head["hist"])
        assert np.array_equal(idx, idx2) and np.array_equal(ok, ok2) and np.array_equal(ok, lead["has_history"])
        raw, okr = LR.row_outputs(head, S["Z"], idx)
        u = LR.head_out_units(raw[ok])
        assert np.abs(u["gap_m"] - lead["gap_m"][ok]).max() < 1e-4 and np.abs(u["closing_ms"] - lead["closing_ms"][ok]).max() < 1e-4
        assert np.abs(u["presence_logit"] - lead["presence_logit"][ok]).max() < 1e-4 and np.isnan(lead["gap_m"][~ok]).all()
    # g6 at read time: the label function replaced by noise changes nothing, and nothing calls it
    calls = []
    monkeypatch.setattr(D, "lead_sup_label_arrays", lambda J: calls.append(1) or {"pres": np.random.rand(10), "gap": np.random.rand(10), "clos": np.random.rand(10)})
    st2 = tmp_path / "states2"
    i = argv.index("--states-dir")
    assert tp.main(argv[:i + 1] + [str(st2)] + argv[i + 2:]) == 0
    assert not calls
    for sp in ("eval", "train"):
        a1, a2 = np.load(files[sp + "_lead"]), np.load(str(st2 / ("F16_s0_%s_lead.npz" % sp)))
        assert all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1.files)


def test_extract_refusals_grid_arm_checkpoint_and_a_cache_without_the_fine_key(run160, tmp_path):
    root = run160["out"] / "_synthetic"
    common = ["--tree", TREE, "--run-dir", str(run160["out"]), "--arm", "F16_s0", "--cache", str(root), "--states-dir", str(tmp_path / "s"), "--eval-range", "0:1", "--train-range", "0:1", "--device", "cpu"]
    with pytest.raises(SystemExit) as e:                                         # an F arm under the default grid
        tp.main(["extract"] + common)
    assert "REFUSED" in str(e.value)
    with pytest.raises(SystemExit) as e:                                         # a TK arm name against an F checkpoint
        tp.main(["extract", "--grid", "pool8x20", "--tree", TREE, "--ckpt", str(run160["out"] / "F16_s0" / "ckpt_last.pt"), "--arm", "TK16_s0", "--cache", str(root), "--states-dir", str(tmp_path / "s1"),
                 "--eval-range", "0:1", "--train-range", "0:1", "--device", "cpu"])
    assert "REFUSED" in str(e.value)
    # a checkpoint whose recorded token count seen is not 160 is refused at the 8 by 20 grid
    ck = _ck(run160["out"] / "F16_s0" / "ckpt_last.pt")
    ck["n_tok_seen"] = 40
    bad = tmp_path / "bad" / "F16_s0"
    bad.mkdir(parents=True)
    torch.save(ck, bad / "ckpt_last.pt")
    with pytest.raises(SystemExit) as e:
        tp.main(["extract", "--grid", "pool8x20", "--tree", TREE, "--ckpt", str(bad / "ckpt_last.pt"), "--arm", "F16_s0", "--cache", str(root), "--states-dir", str(tmp_path / "s2"), "--eval-range", "0:1",
                 "--train-range", "0:1", "--device", "cpu"])
    assert "REFUSED" in str(e.value) and "40" in str(e.value)
    # a cache that holds only the 4 by 10 grid
    import shutil
    c40 = tmp_path / "c40"
    shutil.copytree(root, c40)
    for f in (c40 / "eval" / "clips").glob("*.pt"):
        d = torch.load(f)
        d.pop("pool8x20")
        torch.save(d, f)
    with pytest.raises(SystemExit) as e:
        tp.main(["extract", "--grid", "pool8x20", "--tree", TREE, "--run-dir", str(run160["out"]), "--arm", "F16_s0", "--cache", str(c40), "--states-dir", str(tmp_path / "s3"), "--eval-range", "0:1",
                 "--train-range", "0:1", "--device", "cpu"])
    assert "REFUSED" in str(e.value) and "pool8x20" in str(e.value)


# ================================================================================================ g2 on synthetic caches
def test_g2_pool2x2_is_exact_and_the_control_reads_a_refinement_and_an_unrelated_grid(tmp_path):
    f = torch.randn(3, 40, 8)
    rep = f.reshape(3, 4, 10, 8).repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(3, 160, 8)
    assert torch.allclose(tp.pool2x2(rep), f, atol=1e-6)
    assert not torch.allclose(tp.pool2x2_colmajor(rep), f, atol=1e-3)           # the other layout reads differently
    world = tmp_path / "w"
    D.make_synthetic_world(world, n_train=6, n_eval=2, T=30, seed=1)              # patched: carries a pool8x20 that refines the pool4x10
    out = tmp_path / "g2.json"
    assert tp.main(["g2", "--tree", TREE, "--cache", str(world), "--out", str(out), "--n-clips", "5", "--frames", "10"]) == 0
    r = json.load(open(out, encoding="utf-8"))
    assert r["pass"] and r["corr"] >= 0.99 and r["n_clips_read"] == 5 and r["n_frames"] == 50 and r["median_abs_diff"] >= 0 and r["max_abs_diff"] >= r["median_abs_diff"]
    assert r["alt_colmajor_corr"] < r["corr"] and r["bar_corr"] == 0.99
    other = tmp_path / "o"
    D.make_synthetic_world(other, n_train=6, n_eval=2, T=30, seed=1)
    add_fine(other, independent=True)
    out2 = tmp_path / "g2b.json"
    assert tp.main(["g2", "--tree", TREE, "--cache", str(other), "--out", str(out2), "--n-clips", "5", "--frames", "10"]) == 0
    r2_ = json.load(open(out2, encoding="utf-8"))
    assert not r2_["pass"] and r2_["corr"] < 0.5
    for sp in ("train",):
        for f_ in sorted((other / sp / "clips").glob("*.pt")):
            d = torch.load(f_)
            d.pop("pool8x20")
            torch.save(d, f_)
    out3 = tmp_path / "g2c.json"
    assert tp.main(["g2", "--tree", TREE, "--cache", str(other), "--out", str(out3)]) == 0
    r3 = json.load(open(out3, encoding="utf-8"))
    assert r3["pass"] is False and "reason" in r3 and "corr" not in r3


# ================================================================================================ files and the pod script
BANNED = ["vo" + "id", "super" + "seded", "retr" + "acted", "form" + "erly", "corr" + "ection", "with" + "drawn", "no lon" + "ger", "closed" + " loop"]
OWN = ("tok_probe_g.py", "test_tok_probe_g.py", "run_mutations_g.py", "pod_finegrid_run.sh", "finegrid_read.py", "test_finegrid_read.py", "FINEGRID_DEFS.md", "HANDOFF.md")


def test_files_have_lf_line_endings_and_none_of_the_banned_words():
    for name in OWN:
        p = HERE / name
        if not p.exists():
            continue
        raw = p.read_bytes()
        assert b"\r" not in raw, name
        low = raw.decode("utf-8").lower()
        for w in BANNED:
            assert w not in low, (name, w)


def _sh(env, tmp_path, timeout=120):
    import shutil
    import subprocess
    bash = shutil.which("bash")
    e = dict(os.environ)
    e.update(env)
    return subprocess.run([bash, str(HERE / "pod_finegrid_run.sh")], env=e, capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def _fake_gpu_bin(tmp_path, n_lines):
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    body = "#!/usr/bin/env bash\ncase \"$*\" in\n  *compute-apps*) for i in $(seq 1 %d); do echo $((1000+i)); done;;\n  *) echo 'FakeGPU, 0 MiB, 1000 MiB';;\nesac\nexit 0\n" % n_lines
    (b / "nvidia-smi").write_bytes(body.encode("utf-8"))
    return b


def _fake_cg(tmp_path, free_gb, name="cg"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    maxb = 100 * 2 ** 30
    (d / "memory.max").write_text(str(maxb), encoding="utf-8")
    (d / "memory.stat").write_text("file 1\nanon %d\nkernel 3\n" % (maxb - free_gb * 2 ** 30), encoding="utf-8")
    return d


def test_pod_script_dry_run_prints_the_commands_and_the_guards_refuse(tmp_path):
    r = _sh({"DRY": "1"}, tmp_path)
    assert r.returncode == 0 and r.stdout.count("DRY: ") == 8 and "--expect-steps 19164" in r.stdout and "--shuffle-closing-labels" in r.stdout and "--arm F16SH_s0" in r.stdout
    assert r.stdout.count("--grid pool8x20") == 7 and "--input-norm ln" in r.stdout and " g2 " in r.stdout and "G2_GRIDS.json" in r.stdout
    out = tmp_path / "out"
    out.mkdir()
    (out / "FINEGRID_DONE.txt").write_text("done", encoding="utf-8")
    r = _sh({"OUT": str(out)}, tmp_path)
    assert r.returncode == 3 and "REFUSED" in r.stdout and not (out / "FINEGRID_STOP.txt").exists()
    out2 = tmp_path / "out2"
    r = _sh({"OUT": str(out2), "PB": str(tmp_path)}, tmp_path)                                                       # tok_probe_g.py is not in PB
    assert r.returncode == 1 and "missing" in (out2 / "FINEGRID_STOP.txt").read_text(encoding="utf-8")
    src = (HERE / "pod_finegrid_run.sh").read_text(encoding="utf-8")
    for needle in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.stat", '$1=="anon"', "MIN_FREE_GB:-28", "MAX_GPU_JOBS:-3", "PROBE_GAP_S:-45", "POLL_S:-60", "MAX_WAIT_S:-36000", "FINEGRID_DONE.txt",
                   "FINEGRID_STOP.txt", "/workspace/v8proof/runs/finegrid", "STATES=${STATES:-$PB/states}", "PB=${PB:-/workspace/v8proof/finegrid}", "/workspace/v8proof/tree_leadsup",
                   "/workspace/v8proof/data/cache_pod", "--query-compute-apps=pid --format=csv,noheader", "grep -c .", "finegrid.log", "--expect-steps"):
        assert needle in src, needle
    import re
    for banned in ("pgrep", "pkill", "ps -", "ps aux", "| grep -v grep"):
        assert banned not in src, banned
    assert not re.search(r"(^|[;&|(]|\s)rm\s+-?\w*\s", re.sub(r"--?[a-z-]*norm ", "", src)), "no rm command"
    assert "grep -c . ||" not in src


def test_pod_script_slot_wait_waits_on_three_gpu_jobs_and_stops_after_the_limit_without_starting(tmp_path):
    out = tmp_path / "out"
    pb = tmp_path / "pb"
    pb.mkdir()
    (pb / "tok_probe_g.py").write_text("print('must not run')\n", encoding="utf-8")
    tree = tmp_path / "tree"
    (tree / "stack").mkdir(parents=True)
    cache = tmp_path / "cache"
    (cache / "train" / "clips").mkdir(parents=True)
    (cache / "eval" / "clips").mkdir(parents=True)
    cg = _fake_cg(tmp_path, 60)
    env = {"OUT": str(out), "PB": str(pb), "TREE": str(tree), "CACHE": str(cache), "PATH": str(_fake_gpu_bin(tmp_path, 3)) + os.pathsep + os.environ["PATH"], "CG_MAX": str(cg / "memory.max"),
           "CG_STAT": str(cg / "memory.stat"), "PROBE_GAP_S": "0", "POLL_S": "1", "MAX_WAIT_S": "2", "PY": sys.executable}
    r = _sh(env, tmp_path)                                                                                             # three GPU jobs: no slot, ever
    assert r.returncode == 1 and "must not run" not in r.stdout
    stop = (out / "FINEGRID_STOP.txt").read_text(encoding="utf-8")
    assert "no free slot" in stop and "fourth GPU job" in stop and not (out / "G2_GRIDS.json").exists() and not (out / "train").exists()
    log = (out / "finegrid.log").read_text(encoding="utf-8")
    assert "gpu_jobs=3" in log and "waiting for a slot" in log and log.count("[20") >= 3                                  # time-stamped lines
    env2 = dict(env, PATH=str(_fake_gpu_bin(tmp_path, 0)) + os.pathsep + os.environ["PATH"], CG_MAX=str(_fake_cg(tmp_path, 20, "cg2") / "memory.max"), CG_STAT=str(tmp_path / "cg2" / "memory.stat"))
    r = _sh(env2, tmp_path)                                                                                            # no GPU job but only 20 GB free: still no slot
    assert r.returncode == 1 and "free=20 GB" in (out / "finegrid.log").read_text(encoding="utf-8") and not (out / "train").exists()
    assert (out / "FINEGRID_STOP.prev.").parent.exists() and len(list(out.glob("FINEGRID_STOP.prev.*"))) >= 1


class _SynthWrapper:
    """Text of a tok_probe_g.py stand-in for the end-to-end shell test: it patches the tree's synthetic world to carry the 8 by 20 grid, then runs the real file."""
    TEXT = (
        "import os, sys, runpy\n"
        "sys.dont_write_bytecode = True\n"
        "REAL = %r\n"
        "TREE = %r\n"
        "THIS = %r\n"
        "sys.path.insert(0, THIS)\n"
        "import importlib.util\n"
        "import test_tok_probe_g as TT\n"
        "spec = importlib.util.spec_from_file_location('tok_probe_g_real', REAL)\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['tok_probe_g_real'] = m; spec.loader.exec_module(m)\n"
        "m.use_tree(TREE)\n"
        "m.tree()['D'].make_synthetic_world = TT._world_with_fine\n"
        "sys.exit(m.main())\n")


def test_pod_script_end_to_end_on_the_synthetic_world_on_cpu(tmp_path):
    pb = tmp_path / "pb"
    pb.mkdir()
    (pb / "tok_probe_g.py").write_text(_SynthWrapper.TEXT % (str(TOKFILE), TREE, str(HERE)), encoding="utf-8")
    cache = tmp_path / "cache"
    D.make_synthetic_world(cache, n_train=6, n_eval=2, T=60, seed=0)                  # the same world the train step builds (same seed), with the 8 by 20 key
    out = tmp_path / "out"
    cg = _fake_cg(tmp_path, 60)
    argv = SYN_ARGV.replace("--batch 4", "--batch 4 --steps 6")
    env = {"OUT": str(out), "PB": str(pb), "TREE": TREE, "CACHE": str(cache), "PY": sys.executable, "DEVICE": "cpu", "STEPS_EXPECT": "0", "P3_STEPS": "4", "EVAL_RANGE": "0:2", "TRAIN_RANGE": "0:6",
           "G2_ARGS": "--n-clips 5 --frames 10", "TRAINER_ARGV": argv, "PATH": str(_fake_gpu_bin(tmp_path, 1)) + os.pathsep + os.environ["PATH"], "CG_MAX": str(cg / "memory.max"),
           "CG_STAT": str(cg / "memory.stat"), "PROBE_GAP_S": "0", "POLL_S": "1", "MAX_WAIT_S": "30", "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": "-1"}
    r = _sh(env, tmp_path, timeout=1500)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert (out / "FINEGRID_DONE.txt").exists() and not (out / "FINEGRID_STOP.txt").exists()
    g2 = json.load(open(out / "G2_GRIDS.json", encoding="utf-8"))
    assert g2["pass"] and g2["n_frames"] == 50
    states = pb / "states"
    for arm in ("F16_s0", "F16_s1", "F64_s0", "F64_s1", "F16SH_s0"):
        for sp in ("eval", "train"):
            assert (states / ("%s_%s.npz.json" % (arm, sp))).exists() and (states / ("%s_%s_lead.npz.head.npz" % (arm, sp))).exists()
    rec = json.load(open(states / "F64_s1_train.npz.json", encoding="utf-8"))
    assert rec["grid"] == "pool8x20" and rec["n_tok"] == 160 and rec["D"] == 160 * 64
    assert _ck(out / "train" / "F16_s1" / "ckpt_last.pt")["n_tok_seen"] == 160 and _ck(out / "p3" / "F16SH_s0" / "ckpt_last.pt")["n_tok_seen"] == 160
    log = (out / "finegrid.log").read_text(encoding="utf-8")
    assert log.count("run: ") >= 7 and "slot free on two probes" in log and "DONE:" in log
    done = (out / "FINEGRID_DONE.txt").read_text(encoding="utf-8")
    assert "tok_probe_g.py sha256" in done and done.count("ckpt F") == 5
    r2_ = _sh(env, tmp_path)                                                                                           # a finished marker refuses a second run
    assert r2_.returncode == 3
