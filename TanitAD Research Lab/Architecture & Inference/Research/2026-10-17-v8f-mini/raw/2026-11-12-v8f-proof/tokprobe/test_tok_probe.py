"""Tests of tok_probe.py (card PREREG_V8F_PROOF_ADD27, controls p1 to p5, extract layout). CPU, synthetic worlds of the tree, no pod, no GPU.

Run: python -m pytest -q test_tok_probe.py --basetemp F:/TanitAD-scratch/v8proof/tokprobe/tmp/pt/<new>   (parent folder must exist)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
TREE = "C:/Users/Admin/v8proof_c/leadsup/tree_patch"
READ = "F:/TanitAD-scratch/v8proof/spatial/leadsup/read"
for p in (str(HERE), READ):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(2)
import tok_probe as tp  # noqa: E402

T = tp.use_tree(TREE)
M, D, TR = T["M"], T["D"], T["TR"]
assert os.path.abspath(sys.modules["tanitad"].__file__).replace(os.sep, "/").startswith(TREE)
SYN_ARGV = ("--arm M2b --seed 0 --grid pool4x10 --wm-state grid --wm-cell-dim 16 --lead-sup on --w-lead-sup 1.0 --lead-sup-detach on --synthetic --kin-mode yaw "
            "--batch 8 --steps-mult 1 --lr 3e-3 --warmup 20")
TOKFILE = HERE / "tok_probe.py"


def _args(out):
    return TR.build_parser().parse_args(tp.trainer_argv(SYN_ARGV, str(out)))


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    a = _args(out)
    cfg = TR.make_config(a)
    ds, wins, S, total, info = tp.build_training_data(a, cfg)
    idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
    batch = tp.to_device(D.collate([ds.item(*wins[i]) for i in idx]), torch.device("cpu"))
    return {"a": a, "cfg": cfg, "ds": ds, "wins": wins, "S": S, "total": total, "batch": batch}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A real short run through the CLI path: four arms, 400 steps, synthetic world."""
    out = tmp_path_factory.mktemp("run")
    rc = tp.main(["train", "--tree", TREE, "--out", str(out), "--trainer-argv", SYN_ARGV, "--steps", "400", "--device", "cpu", "--threads", "2"])
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


def p1_pass(mod, batch):
    p = mod.build_probe("TK16_s0")
    opt = mod.make_opt(p, 3e-4, 1e-2)
    _, _, gs = mod.step_probe(p, opt, batch, 3e-4, 1.0, 1.0, want_g=True)
    return gs["g_tok"] > 0 and gs["g_head"] > 0 and gs["g_ctrl"] == 0.0, gs


# ------------------------------------------------------------------------------------------------ p1
def test_p1_gradients_reach_the_probe_and_the_control_reads_exactly_zero(syn):
    ok, gs = p1_pass(fresh_module(), syn["batch"])
    assert ok and gs["g_tok"] > 0 and gs["g_head"] > 0 and gs["g_ctrl"] == 0.0
    p = tp.build_probe("TK64_s1")
    p.train()
    T_ = M.lead_sup_terms(p(syn["batch"]["feats"]), syn["batch"], 5)
    T_["lead_sup"].backward()
    assert p.ctrl.grad is None and all(q.grad is not None and float(q.grad.abs().sum()) > 0 for q in p.head.parameters())


def test_p1_the_mutation_that_removes_the_loss_line_makes_the_check_fail(syn):
    ok, gs = p1_pass(fresh_module(mutate=True), syn["batch"])
    assert not ok and gs["g_tok"] == 0.0 and gs["g_head"] == 0.0           # the mutated step reads exactly 0.0 on the probe parameters: p1 goes red
    assert p1_pass(fresh_module(), syn["batch"])[0]                         # and the unmutated one passes in the same breath


def test_p1_a_loss_less_run_is_caught_in_the_log_and_the_log_of_the_real_run_has_the_first_step_grads(run):
    recs = [json.loads(ln) for ln in open(run["out"] / "TK16_s0" / "train_log.jsonl", encoding="utf-8")][1:]
    first = [r for r in recs if r["first"]][0]
    assert first["g_tok"] > 0 and first["g_head"] > 0 and first["g_ctrl"] == 0.0 and all(r["g_ctrl"] == 0.0 for r in recs)


# ------------------------------------------------------------------------------------------------ p4
def test_p4_the_forward_takes_the_token_array_only_and_labels_cannot_change_outputs(syn):
    assert list(__import__("inspect").signature(tp.TokenProbe.forward).parameters) == ["self", "tokens"]
    p = tp.build_probe("TK16_s0").eval()
    b = syn["batch"]
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


# ------------------------------------------------------------------------------------------------ p5 and the run
def test_run_completes_and_p5_reads_the_short_run(run):
    out = run["out"]
    assert (out / "TRAIN_DONE.json").exists() and (out / "train_meta.json").exists()
    for arm in tp.ARMS_DEFAULT:
        r = tp.check_log(str(out / arm), window=100)
        assert r["pass"], r
        assert r["min_n_gap"] > 0 and r["last_step"] == 400
        assert (out / arm / "ckpt_last.pt").exists()
    meta = json.load(open(out / "train_meta.json", encoding="utf-8"))
    assert meta["total"] == 400 and meta["lead_labels"]["n_gap"] > 0 and meta["lead_labels"]["n_clos"] > 0 and len(meta["lead_labels"]["sha256"]) == 64


def _write_log(path, losses, ngap=10.0):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"header": True, "arm": "X", "total": 3000}) + "\n")
        for i, l in enumerate(losses):
            st = 50 * (i + 1)
            fh.write(json.dumps({"step": st, "first": False, "lead_sup_mean": l, "n_gap": ngap, "n_gap_min_interval": ngap}) + "\n")


def test_p5_goes_red_on_a_rising_loss_a_zero_gap_count_and_an_unfinished_log(tmp_path):
    good = np.linspace(2.0, 1.0, 60)
    _write_log(tmp_path / "a.jsonl", good)
    assert tp.check_log(str(tmp_path / "a.jsonl"))["pass"]
    _write_log(tmp_path / "b.jsonl", good[::-1])
    assert not tp.check_log(str(tmp_path / "b.jsonl"))["pass_lead_term_falls"]
    _write_log(tmp_path / "c.jsonl", good, ngap=0.0)
    assert not tp.check_log(str(tmp_path / "c.jsonl"))["pass_gap_labels_every_step"]
    _write_log(tmp_path / "d.jsonl", good[:40])
    assert not tp.check_log(str(tmp_path / "d.jsonl"))["pass_complete"]


def test_the_four_probes_really_differ_and_see_the_same_batches(syn, tmp_path, monkeypatch):
    seen = {}
    orig = tp.step_probe

    def spy(probe, opt, batch, *a, **k):
        seen.setdefault(id(probe), []).append(hashlib.sha256(batch["feats"].numpy().tobytes()).hexdigest()[:12])
        return orig(probe, opt, batch, *a, **k)

    monkeypatch.setattr(tp, "step_probe", spy)
    b = syn["batch"]
    stream = iter([{k: v for k, v in b.items()} for _ in range(6)])
    arms = list(tp.ARMS_DEFAULT)
    probes = tp.run_training(stream, arms, 6, lambda s: 3e-3, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(tmp_path), log_every=50)
    hs = [v for v in seen.values()]
    assert len(hs) == 4 and all(h == hs[0] for h in hs) and len(hs[0]) == 6
    w = {a: probes[a].tok[0].weight.detach().clone() for a in arms}
    assert not torch.equal(w["TK16_s0"], w["TK16_s1"]) and not torch.equal(w["TK64_s0"], w["TK64_s1"])          # seeds differ -> weights differ, same batches
    assert w["TK16_s0"].shape == w["TK16_s1"].shape and probes["TK64_s0"].head.frame.in_features == 2560 and probes["TK16_s0"].head.frame.in_features == 640
    # same seed, same batches -> the same weights (the difference above is the seed and nothing else)
    again = tp.run_training(iter([dict(b) for _ in range(6)]), ["TK16_s0"], 6, lambda s: 3e-3, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(tmp_path / "again"))
    assert torch.equal(again["TK16_s0"].tok[0].weight, probes["TK16_s0"].tok[0].weight)


def test_the_probe_head_is_the_leadsuphead_and_the_trainer_schedule_and_groups_are_used(syn):
    p = tp.build_probe("TK16_s0")
    assert type(p.head) is M.LeadSupHead and p.head.hist == 5 and p.head.frame.in_features == 640
    opt = tp.make_opt(p, 3e-4, 1e-2)
    assert sorted(g["weight_decay"] for g in opt.param_groups) == [0.0, 1e-2] and all(id(p.ctrl) != id(q) for g in opt.param_groups for q in g["params"])
    s = TR.LRSchedule(3e-4, 200, 19164)
    assert abs(s(0) - 3e-4 / 200) < 1e-12 and abs(s(199) - 3e-4) < 1e-12 and abs(s(19163) - 3e-5) < 3e-7
    assert syn["a"].lr == 3e-3                                                                           # the parsed argv carries the optimiser settings, not typed copies
    d = TR.build_parser().parse_args(tp.trainer_argv(tp.DEFAULT_TRAINER_ARGV, "x"))
    assert (d.lr, d.wd, d.warmup, d.clip_grad, d.batch, d.steps_mult, d.exclude, d.seed) == (3e-4, 1e-2, 200, 1.0, 64, 4, "540:600", 0)


# ------------------------------------------------------------------------------------------------ p2 and p3 on planted tokens
F_, B_ = 8, 8


def planted(seed, n_clips, T_=24, d=1024):
    rng = np.random.default_rng(seed)
    uv = np.random.default_rng(999)
    U = uv.normal(size=(40, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = uv.normal(size=(40, d)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    clips = []
    for _ in range(n_clips):
        present = rng.random() < 0.85
        g0, c = rng.uniform(15, 45), rng.uniform(-3, 3)
        gap = g0 + c * 0.2 * np.arange(T_)
        tok = rng.normal(size=(T_, 40, d)).astype(np.float32)
        if present:
            tok += 6.0 * ((gap - 30.0) / 10.0)[:, None, None] * U[None] + 6.0 * (c / 3.0) * V[None]
        clips.append({"tok": tok.astype(np.float16), "pres": np.full(T_, float(present), np.float32), "gap": np.where(present, gap, np.nan).astype(np.float32),
                      "clos": np.full(T_, c if present else np.nan, np.float32)})
    return clips


def batches_of(clips, seed, clos_override=None):
    rng = np.random.default_rng(seed)
    while True:
        feats, pres, gap, clos = [], [], [], []
        for _ in range(B_):
            ci = int(rng.integers(len(clips)))
            s = int(rng.integers(0, clips[ci]["tok"].shape[0] - F_ + 1))
            c = clips[ci]
            feats.append(c["tok"][s:s + F_]); pres.append(c["pres"][s:s + F_]); gap.append(c["gap"][s:s + F_])
            clos.append((c["clos"] if clos_override is None else clos_override[ci])[s:s + F_])
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
                o = probe(x[s:s + F_][None])[0]                                  # [4, 3]
                lab = slice(s + 4, s + F_)
                ok = np.isfinite(c["gap"][lab])
                ys_g += list(c["gap"][lab][ok]); ps_g += list(o[:, 1].numpy()[ok] * 50.0)
                ys_c += list(c["clos"][lab][ok]); ps_c += list(o[:, 2].numpy()[ok] * 10.0)
    return r2(np.array(ys_g), np.array(ps_g)), r2(np.array(ys_c), np.array(ps_c))


STEPS = 500


def train_planted(clips, clos_override=None, arm="TK16_s0", out=None):
    lr_at = TR.LRSchedule(3e-3, 50, STEPS)
    pr = tp.run_training(batches_of(clips, 1, clos_override), [arm], STEPS, lr_at, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(out))
    return pr[arm].eval()


def test_p2_planted_tokens_are_read_back_above_0_9_and_p3_shuffled_labels_read_at_most_0_02(tmp_path):
    train, held = planted(1, 16), planted(2, 8)
    pr = train_planted(train, out=tmp_path / "p2")
    rg, rc = heldout_r2(pr, held)
    print("P2 held-out R2 gap %.4f closing %.4f" % (rg, rc))
    assert rg > 0.9 and rc > 0.9, (rg, rc)
    # p3: the closing labels permuted across rows (the flag's own function), the same tokens, the same steps
    class _DS:
        pass
    ds = _DS()
    flat = np.concatenate([c["clos"] for c in train])
    ds.lead_sup_labels = {"clos": flat}
    info = tp.shuffle_closing(ds)
    sh = ds.lead_sup_labels["clos"]
    assert info["n_shuffled"] == int(np.isfinite(flat).sum()) and np.array_equal(np.isfinite(sh), np.isfinite(flat))
    assert np.array_equal(np.sort(sh[np.isfinite(sh)]), np.sort(flat[np.isfinite(flat)])) and not np.array_equal(sh, flat, equal_nan=True)
    over = [sh[i * 24:(i + 1) * 24] for i in range(len(train))]
    prs = train_planted(train, clos_override=over, out=tmp_path / "p3")
    rg3, rc3 = heldout_r2(prs, held)
    print("P3 held-out R2 gap %.4f closing %.4f" % (rg3, rc3))
    assert rc3 <= 0.02 and rg3 > 0.9, (rg3, rc3)                                 # shuffled closing labels: closing not read; gap labels untouched: still read
    assert rc > rc3 + 0.5                                                      # the unshuffled arm reads it: the control separates


def test_p3_the_flag_on_the_trainer_data_path_permutes_and_names_the_arm(syn, tmp_path):
    ds = syn["ds"]
    before = np.array(ds.lead_sup_labels["clos"], copy=True)
    info = tp.shuffle_closing(ds)
    after = ds.lead_sup_labels["clos"]
    assert info["n_shuffled"] == int(np.isfinite(before).sum()) > 0 and np.array_equal(np.isfinite(after), np.isfinite(before)) and not np.array_equal(after, before, equal_nan=True)
    wins = syn["wins"]
    it = ds.item(*wins[5])                                                    # the item reads the permuted array
    rr = int(ds.J["off"][wins[5][0]]) + np.clip(np.arange(wins[5][1] - 15, wins[5][1] + 1), 0, None)
    assert np.array_equal(it["lead_clos"], after[rr], equal_nan=True)
    ds.lead_sup_labels["clos"] = before                                        # restore for later tests
    out = tmp_path / "sh"
    rc = tp.main(["train", "--tree", TREE, "--out", str(out), "--trainer-argv", SYN_ARGV, "--steps", "60", "--shuffle-closing-labels", "--device", "cpu"])
    assert (out / "TK16SH_s0" / "ckpt_last.pt").exists() and json.load(open(out / "train_meta.json", encoding="utf-8"))["shuffle_closing"]["n_shuffled"] > 0
    assert rc in (0, 2)


# ------------------------------------------------------------------------------------------------ extract and the reader
def test_extract_writes_the_layouts_and_the_reader_loads_and_reproduces_them(run, tmp_path, monkeypatch):
    import leadsup_read as LR
    import pod_lead_apply as PA
    root = run["out"] / "_synthetic"
    st = tmp_path / "states"
    ckpt = run["out"] / "TK16_s0" / "ckpt_last.pt"
    argv = ["extract", "--tree", TREE, "--run-dir", str(run["out"]), "--arm", "TK16_s0", "--cache", str(root), "--states-dir", str(st), "--eval-range", "0:2", "--train-range", "0:6", "--device", "cpu"]
    assert tp.main(argv) == 0
    files = LR.arm_files(str(st), "TK16_s0")
    rows, ck_sha = LR.check_records(files, plumbing=False, do_train=True)
    assert rows["pass"], rows
    assert ck_sha == hashlib.sha256(ckpt.read_bytes()).hexdigest() and rows["ckpt_sha256_identical_across_the_four_files"]
    for sp in ("eval", "train"):
        S = LR.load_states(files[sp + "_states"])
        assert S["Z"].dtype == np.float16 and S["Z"].shape[1] == 640 and S["clip_idx"].dtype == np.int32 and S["t"].dtype == np.int16
        n = 2 * 60 if sp == "eval" else 6 * 60
        assert S["Z"].shape[0] == n and len(S["clips"]) == (2 if sp == "eval" else 6)
        rec = json.load(open(files[sp + "_states"] + ".json", encoding="utf-8"))
        for k in ("ckpt", "ckpt_sha256", "step", "wm_state", "wm", "D", "n_frames", "n_clips", "split", "lo", "hi", "n_manifest", "out_sha256", "written_utc", "seconds"):
            assert k in rec, k
        lead = np.load(files[sp + "_lead"], allow_pickle=False)
        head = LR.load_head(files[sp + "_lead"] + ".head.npz")
        idx, ok = PA.history_rows(S["clip_idx"], S["t"], head["hist"])
        idx2, ok2 = tp.history_rows(S["clip_idx"], S["t"], head["hist"])
        assert np.array_equal(idx, idx2) and np.array_equal(ok, ok2) and np.array_equal(ok, lead["has_history"])           # the copy equals the original
        raw, okr = LR.row_outputs(head, S["Z"], idx)                          # the reader's own forward on the stored states
        u = LR.head_out_units(raw[ok])
        assert np.abs(u["gap_m"] - lead["gap_m"][ok]).max() < 1e-4 and np.abs(u["closing_ms"] - lead["closing_ms"][ok]).max() < 1e-4
        assert np.abs(u["presence_logit"] - lead["presence_logit"][ok]).max() < 1e-4 and np.isnan(lead["gap_m"][~ok]).all()
        assert int(ok.sum()) > 0 and int(lead["hist"]) == 5
    # the stored head tensors are the checkpoint's
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    hz = np.load(files["eval_lead"] + ".head.npz")
    assert np.array_equal(hz["frame_w"], sd["head.frame.weight"].numpy()) and np.array_equal(hz["mlp2_b"], sd["head.mlp.2.bias"].numpy())
    # p4 at read time: the label function replaced by noise changes nothing in the outputs (and nothing calls it)
    calls = []
    monkeypatch.setattr(D, "lead_sup_label_arrays", lambda J: calls.append(1) or {"pres": np.random.rand(10), "gap": np.random.rand(10), "clos": np.random.rand(10)})
    st2 = tmp_path / "states2"
    i = argv.index("--states-dir")
    assert tp.main(argv[:i + 1] + [str(st2)] + argv[i + 2:]) == 0
    assert not calls
    for sp in ("eval", "train"):
        a1, a2 = np.load(files[sp + "_lead"]), np.load(str(st2 / ("TK16_s0_%s_lead.npz" % sp)))
        assert all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1.files)
        assert np.array_equal(np.load(files[sp + "_states"])["Z"], np.load(str(st2 / ("TK16_s0_%s.npz" % sp)))["Z"])


def test_a_tampered_head_is_caught_by_the_reader_record_check(run, tmp_path):
    import leadsup_read as LR
    root = run["out"] / "_synthetic"
    st = tmp_path / "s"
    assert tp.main(["extract", "--tree", TREE, "--run-dir", str(run["out"]), "--arm", "TK64_s1", "--cache", str(root), "--states-dir", str(st), "--eval-range", "0:2", "--train-range", "0:3", "--device", "cpu"]) == 0
    files = LR.arm_files(str(st), "TK64_s1")
    assert LR.check_records(files)[0]["pass"]
    hp = files["eval_lead"] + ".head.npz"
    z = dict(np.load(hp))
    z["mlp2_b"] = z["mlp2_b"] + 1.0
    np.savez(hp, **z)
    assert not LR.check_records(files)[0]["pass"]
    assert json.load(open(files["train_states"] + ".json", encoding="utf-8"))["D"] == 2560


# ------------------------------------------------------------------------------------------------ files
BANNED = ["vo" + "id", "super" + "seded", "retr" + "acted", "form" + "erly", "corr" + "ection", "with" + "drawn", "no lon" + "ger", "closed" + " loop"]


def test_files_have_lf_line_endings_and_none_of_the_banned_words():
    for name in ("tok_probe.py", "TOKPROBE_DEFS.md", "pod_tokprobe_run.sh", "HANDOFF.md"):
        p = HERE / name
        if not p.exists():
            continue
        raw = p.read_bytes()
        assert b"\r" not in raw, name
        low = raw.decode("utf-8").lower()
        for w in BANNED:
            assert w not in low, (name, w)


# ------------------------------------------------------------------------------------------------ the pod script
def _sh(env, tmp_path):
    import shutil
    import subprocess
    bash = shutil.which("bash")
    e = dict(os.environ)
    e.update(env)
    return subprocess.run([bash, str(HERE / "pod_tokprobe_run.sh")], env=e, capture_output=True, text=True, encoding="utf-8", timeout=120)


def test_pod_script_dry_run_prints_the_commands_and_the_guards_refuse(tmp_path):
    r = _sh({"DRY": "1"}, tmp_path)
    assert r.returncode == 0 and r.stdout.count("DRY: ") == 7 and "--expect-steps 19164" in r.stdout and "--shuffle-closing-labels" in r.stdout and "--arm TK16SH_s0" in r.stdout
    out = tmp_path / "out"
    out.mkdir()
    (out / "TOKPROBE_DONE.txt").write_text("done", encoding="utf-8")
    r = _sh({"OUT": str(out)}, tmp_path)
    assert r.returncode == 3 and "REFUSED" in r.stdout and not (out / "TOKPROBE_STOP.txt").exists()                  # a finished marker: nothing runs
    out2 = tmp_path / "out2"
    r = _sh({"OUT": str(out2), "PB": str(tmp_path)}, tmp_path)                                                     # tok_probe.py is not in PB
    assert r.returncode == 1 and "missing" in (out2 / "TOKPROBE_STOP.txt").read_text(encoding="utf-8")
    src = (HERE / "pod_tokprobe_run.sh").read_text(encoding="utf-8")
    for needle in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.stat", '$1=="anon"', "MIN_FREE_GB:-20", "TOKPROBE_DONE.txt", "TOKPROBE_STOP.txt", "/workspace/v8proof/runs/tokprobe", "STATES=${STATES:-$PB/states}", "PB=${PB:-/workspace/v8proof/probe}"):
        assert needle in src, needle
