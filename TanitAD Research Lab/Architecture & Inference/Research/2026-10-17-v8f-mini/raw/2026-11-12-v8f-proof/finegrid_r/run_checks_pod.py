"""run_checks_pod.py -- the checks of tok_probe_gr.py that need torch (card PREREG_V8F_PROOF_ADD43). Plain python, no pytest, CPU only, tiny synthetic data. NOT RUN on the dev box (no torch there while the
other session holds the memory); the lead runs it ON THE POD before any real run:

  CUDA_VISIBLE_DEVICES=-1 /workspace/venv/bin/python run_checks_pod.py --tmp /workspace/v8proof/finegrid_r/checks_tmp [--tree /workspace/v8proof/tree_leadsup] [--pilot-tool /workspace/v8proof/finegrid/tok_probe_g.py]

--tmp must be a NEW folder (it is filled with four tiny synthetic worlds, about 70 MB, and checkpoints; the lead removes it afterwards). Last line: `ZZ CHECKS-POD DONE n_pass n_fail`. Exit 0 if no check failed.
Checks: g1 (default path bit-identical to the pilot tool: probe weights, step losses, extract outputs), r1 (batch log: length, equal for equal runs, differs when the chunk differs), chunk (the trainer's formula
reproduces the pilot's 43 clips / chunk 41 at 160 tokens and chunk 48 at 40 tokens; flag equal to the own grid = default), p5 (permuted-label arm read on its intact parts), g4/g3/g6 at 160 tokens (planted data, no world),
extract on the eval split only, refusals.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SYN_ARGV = ("--arm M2b --seed 0 --grid pool4x10 --wm-state grid --wm-cell-dim 16 --lead-sup on --w-lead-sup 1.0 --lead-sup-detach on --synthetic --kin-mode yaw "
            "--batch 4 --steps-mult 1 --lr 3e-3 --warmup 20")
NT, TT = 7, 48                                                                   # the tiny world: 7 train clips of 48 frames, 2 eval clips
torch.set_num_threads(4)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def ck(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def train(mod, out, extra, steps=30):
    return mod.main(["train", "--out", str(out), "--tree", A.tree, "--trainer-argv", SYN_ARGV, "--steps", str(steps), "--device", "cpu", "--threads", "4"] + extra)


def r1(out):
    return json.load(open(os.path.join(out, "r1_batches.json"), encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ checks
def check_g1():
    """The default path of the changed tool equals the pilot tool: probe weights bit-identical for the same seeds and steps, the step-by-step losses, and the extract outputs array by array."""
    ob, og = os.path.join(A.tmp, "g1_pilot"), os.path.join(A.tmp, "g1_new")
    assert train(PT, ob, ["--grid", "pool4x10"]) in (0, 2)
    assert train(TG, og, ["--grid", "pool4x10"]) in (0, 2)
    for arm in TG.ARMS_DEFAULT:
        cb, cg = ck(os.path.join(ob, arm, "ckpt_last.pt")), ck(os.path.join(og, arm, "ckpt_last.pt"))
        assert cb["state_dict"].keys() == cg["state_dict"].keys()
        assert all(torch.equal(cb["state_dict"][k], cg["state_dict"][k]) for k in cb["state_dict"]), arm
        lb = [json.loads(x) for x in open(os.path.join(ob, arm, "train_log.jsonl"), encoding="utf-8")][1:]
        lg = [json.loads(x) for x in open(os.path.join(og, arm, "train_log.jsonl"), encoding="utf-8")][1:]
        keys = ("lead_sup", "lead_sup_bce", "lead_sup_gap", "lead_sup_clos", "grad_norm", "g_tok", "g_head")
        assert [[r[k] for k in keys] for r in lb] == [[r[k] for k in keys] for r in lg], arm
    cache = os.path.join(ob, "_synthetic")
    for name, mod, out in (("p", PT, ob), ("n", TG, og)):
        assert mod.main(["extract", "--tree", A.tree, "--run-dir", out, "--arm", "TK16_s0", "--cache", cache, "--states-dir", os.path.join(A.tmp, "g1_st_" + name), "--splits", "eval",
                         "--eval-range", "0:2", "--device", "cpu"]) == 0
    a, b = np.load(os.path.join(A.tmp, "g1_st_p", "TK16_s0_eval.npz")), np.load(os.path.join(A.tmp, "g1_st_n", "TK16_s0_eval.npz"))
    assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
    la, lb2 = np.load(os.path.join(A.tmp, "g1_st_p", "TK16_s0_eval_lead.npz")), np.load(os.path.join(A.tmp, "g1_st_n", "TK16_s0_eval_lead.npz"))
    assert all(np.array_equal(la[k], lb2[k], equal_nan=(la[k].dtype.kind == "f")) for k in la.files)
    assert not os.path.exists(os.path.join(A.tmp, "g1_st_n", "TK16_s0_train.npz"))          # --splits eval writes no train state
    return "4 arms, 30 steps: weights, losses, extract outputs identical"


def check_r1_and_chunk():
    og = os.path.join(A.tmp, "g1_new")
    og_like = os.path.join(A.tmp, "r1_like_own")
    assert train(TG, og_like, ["--grid", "pool4x10", "--sampler-chunk-like", "pool4x10", "--arms", "TK16_s0"]) in (0, 2)
    a, b = r1(og), r1(og_like)
    assert a["n_batches"] == 30 == len(a["idx_hashes"]) == len(a["full_hashes"]) and a["idx_hashes"] == b["idx_hashes"] and a["sampler_chunk"] == b["sampler_chunk"]
    assert TG.r1_compare(os.path.join(og, "r1_batches.json"), os.path.join(og_like, "r1_batches.json"))["pass"]
    c0, c1 = ck(os.path.join(og, "TK16_s0", "ckpt_last.pt")), ck(os.path.join(og_like, "TK16_s0", "ckpt_last.pt"))
    assert all(torch.equal(c0["state_dict"][k], c1["state_dict"][k]) for k in c0["state_dict"])      # the flag equal to the own grid changes nothing
    TG.GRIDS["poolfake"] = 100000                                                    # a grid whose token count forces a smaller clip store, hence a smaller chunk
    try:
        og_fake = os.path.join(A.tmp, "r1_fake_sh")
        assert train(TG, og_fake, ["--grid", "pool4x10", "--sampler-chunk-like", "poolfake", "--shuffle-closing-labels", "--arms", "TK16SH_s2"]) in (0, 2)
    finally:
        del TG.GRIDS["poolfake"]
    f = r1(og_fake)
    assert f["sampler_chunk"] < a["sampler_chunk"] and f["idx_hashes"] != a["idx_hashes"], (f["sampler_chunk"], a["sampler_chunk"])
    assert not TG.r1_compare(os.path.join(og, "r1_batches.json"), os.path.join(og_fake, "r1_batches.json"))["pass"]
    assert TG.main(["r1", "--a", og, "--b", og_like, "--out", os.path.join(A.tmp, "r1_ok.json")]) == 0 and TG.main(["r1", "--a", og, "--b", og_fake, "--out", os.path.join(A.tmp, "r1_bad.json")]) == 2
    # the formula: the pilot's real numbers (longest clip 105 frames): 43 clips and chunk 41 at 160 tokens, 174 clips and chunk 48 at 40 tokens
    assert TG.trainer_lru(105, 160) == 43 and TG.sampler_chunk(43) == 41 and TG.trainer_lru(105, 40) == 174 and TG.sampler_chunk(174) == 48
    meta = json.load(open(os.path.join(og, "train_meta.json"), encoding="utf-8"))
    assert meta["sampler_chunk"] == TG.sampler_chunk(meta["lru_trainer"]) and meta["lru_trainer"] == TG.trainer_lru(meta["host_est"]["clip_frames_max"], 40)     # the mirror of the trainer line equals the trainer's own value
    done = json.load(open(os.path.join(og, "TRAIN_DONE.json"), encoding="utf-8"))
    assert done["sampler_chunk"] == meta["sampler_chunk"] and len(done["r1_sequence_sha256"]) == 64
    return "30 batches logged; equal runs equal; chunk %s vs %s differs and r1 fails; formula 43/41 and 174/48" % (a["sampler_chunk"], f["sampler_chunk"])


def check_p5_parts():
    og_fake = os.path.join(A.tmp, "r1_fake_sh")
    recs = [json.loads(x) for x in open(os.path.join(og_fake, "TK16SH_s2", "train_log.jsonl"), encoding="utf-8")]
    assert recs[0]["arm"] == "TK16SH_s2" and all(k in recs[1] for k in ("lead_sup_bce_mean", "lead_sup_gap_mean", "lead_sup_clos_mean"))       # the parts are in the log of the unchanged tool
    r = TG.check_log(os.path.join(og_fake, "TK16SH_s2"), window=10)
    assert r["shuffled_closing_arm"] is True and r["pass_intact_parts_fall"] is not None and "intact_parts_first_mean" in r and "permuted_part_clos_first_mean" in r
    assert r["pass"] == bool(r["pass_intact_parts_fall"] and r["pass_gap_labels_every_step"] and r["pass_complete"])

    def wlog(path, arm, bce, gap, clos):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"header": True, "arm": arm, "total": 3000}) + "\n")
            for i in range(len(bce)):
                fh.write(json.dumps({"step": 50 * (i + 1), "first": False, "lead_sup_mean": bce[i] + gap[i] + clos[i], "lead_sup_bce_mean": bce[i], "lead_sup_gap_mean": gap[i], "lead_sup_clos_mean": clos[i],
                                     "n_gap": 10.0, "n_gap_min_interval": 10.0}) + "\n")
    down, up = np.linspace(1.0, 0.5, 60), np.linspace(0.5, 1.0, 60)
    wlog(os.path.join(A.tmp, "p5_a.jsonl"), "F16SH_s2", down, down, up)
    a = TG.check_log(os.path.join(A.tmp, "p5_a.jsonl"))
    assert a["pass"] and not a["pass_lead_term_falls"] is None and a["pass_intact_parts_fall"] and a["permuted_part_clos_last_mean"] > a["permuted_part_clos_first_mean"]
    wlog(os.path.join(A.tmp, "p5_b.jsonl"), "F16_s2", down, down, up)                  # an ordinary arm keeps the check of card ADD27 (whole lead term)
    b = TG.check_log(os.path.join(A.tmp, "p5_b.jsonl"))
    assert b["shuffled_closing_arm"] is False and b["pass"] == b["pass_lead_term_falls"]
    wlog(os.path.join(A.tmp, "p5_c.jsonl"), "F16SH_s2", up, up, down)
    assert not TG.check_log(os.path.join(A.tmp, "p5_c.jsonl"))["pass"]
    return "intact parts read for SH arms, whole lead term for the others"


def planted_batch(n_tok, seed=0, B=8, F=8, d=1024):
    rng = np.random.default_rng(seed)
    feats = torch.as_tensor(rng.normal(size=(B, F, n_tok, d)).astype(np.float32)).half()
    return {"feats": feats, "frame_ok": torch.ones(B, F, dtype=torch.bool), "lead_pres": torch.ones(B, F), "lead_gapm": torch.as_tensor(rng.uniform(15, 45, size=(B, F)).astype(np.float32)),
            "lead_clos": torch.as_tensor(rng.uniform(-3, 3, size=(B, F)).astype(np.float32))}


def check_g4_g6():
    for arm, grid, n in (("F16_s2", "pool8x20", 160), ("TK64_s9", "pool4x10", 40)):
        p = TG.build_probe(arm, grid=grid)
        opt = TG.make_opt(p, 3e-4, 1e-2)
        _, _, gs = TG.step_probe(p, opt, planted_batch(n), 3e-4, 1.0, 1.0, want_g=True)
        assert gs["g_tok"] > 0 and gs["g_head"] > 0 and gs["g_ctrl"] == 0.0, (arm, gs)
    src = open(TG.__file__, encoding="utf-8").read().split("\n")
    hit = [i for i, ln in enumerate(src) if ln.rstrip().endswith("# LOSS-LINE")]
    assert len(hit) == 1
    src[hit[0]] = src[hit[0]][:len(src[hit[0]]) - len(src[hit[0]].lstrip())] + "loss = 0.0 * out.sum()"
    import types
    mut = types.ModuleType("tp_mut")
    mut.__file__ = TG.__file__
    exec(compile("\n".join(src), TG.__file__, "exec"), mut.__dict__)
    mut.use_tree(A.tree)
    p = mut.build_probe("F16_s2", grid="pool8x20")
    _, _, gs = mut.step_probe(p, mut.make_opt(p, 3e-4, 1e-2), planted_batch(160), 3e-4, 1.0, 1.0, want_g=True)
    assert gs["g_tok"] == 0.0 and gs["g_head"] == 0.0                                  # the mutated loss line reaches nothing: the check goes red
    assert list(inspect.signature(TG.TokenProbe.forward).parameters) == ["self", "tokens"]       # g6: tokens only
    p = TG.build_probe("F16_s2").eval()
    b = planted_batch(160)
    with torch.no_grad():
        o1 = p(b["feats"])
        b2 = dict(b)
        b2["lead_gapm"] = b["lead_gapm"] * 7.0
        assert torch.equal(o1, p(b2["feats"]))
    return "gradients non-zero, control 0.0, loss-line mutation reads 0.0, forward takes tokens only"


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


def check_g3():
    B_, F_, steps = 8, 8, 300
    train_c, held = planted(1, 16), planted(2, 8)

    def batches(clips, seed):
        rng = np.random.default_rng(seed)
        while True:
            f, p, g, c = [], [], [], []
            for _ in range(B_):
                cl = clips[int(rng.integers(len(clips)))]
                s = int(rng.integers(0, cl["tok"].shape[0] - F_ + 1))
                f.append(cl["tok"][s:s + F_]); p.append(cl["pres"][s:s + F_]); g.append(cl["gap"][s:s + F_]); c.append(cl["clos"][s:s + F_])
            yield {"feats": torch.as_tensor(np.stack(f)), "frame_ok": torch.ones(B_, F_, dtype=torch.bool), "lead_pres": torch.as_tensor(np.stack(p)), "lead_gapm": torch.as_tensor(np.stack(g)),
                   "lead_clos": torch.as_tensor(np.stack(c))}
    lr_at = A.TR.LRSchedule(3e-3, 50, steps)
    pr = TG.run_training(batches(train_c, 1), ["F16_s2"], steps, lr_at, lr=3e-3, wd=1e-2, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=os.path.join(A.tmp, "g3"), grid="pool8x20")["F16_s2"].eval()
    ys, ps = [], []
    with torch.no_grad():
        for cl in held:
            x = torch.as_tensor(cl["tok"]).float()
            for s in range(0, x.shape[0] - F_ + 1, 2):
                o = pr(x[s:s + F_][None])[0]
                lab = slice(s + 4, s + F_)
                ok = np.isfinite(cl["gap"][lab])
                ys += list(cl["gap"][lab][ok]); ps += list(o[:, 1].numpy()[ok] * 50.0)
    y, p = np.array(ys), np.array(ps)
    r2 = 1.0 - float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    assert r2 > 0.9, r2
    return "held-out gap R2 %.3f at 160 tokens (bar 0.9)" % r2


def check_refusals():
    for bad in (("F16_s2", "pool4x10"), ("TK16_s2", "pool8x20")):
        try:
            TG.check_arm_grid(*bad)
        except SystemExit as e:
            assert "REFUSED" in str(e)
        else:
            raise AssertionError(bad)
    try:
        train(TG, os.path.join(A.tmp, "ref"), ["--grid", "pool4x10", "--sampler-chunk-like", "pool9x9"])
    except SystemExit as e:                                                              # argparse (not in choices) or the tool's own refusal
        assert e.code not in (0, None)
    else:
        raise AssertionError("an unknown grid for --sampler-chunk-like was accepted")
    return "arm/grid pairing and an unknown chunk grid refused"


def main():
    global A, TG, PT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--tree", default="/workspace/v8proof/tree_leadsup")
    ap.add_argument("--pilot-tool", default="/workspace/v8proof/finegrid/tok_probe_g.py")
    ap.add_argument("--new-tool", default=os.path.join(HERE, "tok_probe_gr.py"))
    ap.add_argument("--controls-out", default="", help="write the dev-box controls file for the reader (g1, g3, g4, g6_forward + the tool sha256) here")
    A = ap.parse_args()
    if os.path.exists(A.tmp) and os.listdir(A.tmp):
        raise SystemExit("REFUSED: --tmp must be a new or empty folder")
    os.makedirs(A.tmp, exist_ok=True)
    TG, PT = load(A.new_tool, "tok_probe_gr_under_check"), load(A.pilot_tool, "tok_probe_g_pilot")
    T = TG.use_tree(A.tree)
    PT.use_tree(A.tree)
    A.TR, D = T["TR"], T["D"]
    orig = D.make_synthetic_world
    D.make_synthetic_world = lambda root, *a, **k: orig(root, n_train=NT, n_eval=2, T=TT, seed=k.get("seed", 0))
    checks = [("g1_default_path_bit_identical_to_the_pilot_tool", check_g1), ("r1_batch_log_and_sampler_chunk", check_r1_and_chunk), ("p5_intact_parts_for_permuted_label_arms", check_p5_parts),
              ("g4_g6_gradients_mutation_forward", check_g4_g6), ("g3_planted_tokens_at_160", check_g3), ("refusals", check_refusals)]
    npass = nfail = 0
    ok = {}
    for name, fn in checks:
        t0 = time.time()
        try:
            msg = fn()
            npass += 1
            ok[name] = True
            print("PASS %s (%.0f s): %s" % (name, time.time() - t0, msg), flush=True)
        except BaseException as e:  # a failed check must not stop the others
            nfail += 1
            ok[name] = False
            print("FAIL %s: %r" % (name, e), flush=True)
            traceback.print_exc()
    sha = hashlib.sha256(open(A.new_tool, "rb").read()).hexdigest()
    print("tool sha256: %s" % sha)
    if A.controls_out:
        ctl = {"g1": ok.get("g1_default_path_bit_identical_to_the_pilot_tool", False), "g3": ok.get("g3_planted_tokens_at_160", False), "g4": ok.get("g4_g6_gradients_mutation_forward", False),
               "g6_forward": ok.get("g4_g6_gradients_mutation_forward", False),
               "notes": {"written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "by": "run_checks_pod.py (synthetic data, CPU)", "code_sha256": {"tok_probe_gr.py": sha}, "checks": ok}}
        with open(A.controls_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(ctl, fh, indent=1)
    print("ZZ CHECKS-POD DONE %d %d" % (npass, nfail), flush=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
