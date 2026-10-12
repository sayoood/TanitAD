#!/usr/bin/env python3
"""Tests for the RL-S1 changes to refe/scorer_finetune.py (the u400 loading fix, build_parser(), the EPO scorer loss, the G11 preflight gate).
Run:  PYTHONIOENCODING=utf-8 python refe/test_rl_epo.py   ->   ZZTEST_PASS <n>   (prints the mutation table).   CPU only, no pod, no checkpoint.

WHAT IS PROVEN, AND HOW (CLAUDE.md "built, tested, and unreachable from its caller"):
  * The loss closures live inside main().  They are NOT imported: they are AST-extracted from the SOURCE TEXT of scorer_finetune.py and exec'd,
    so a mutation of the real caller is seen by every check below.  The suite is run on the unmutated source (must be green) and once per
    mutation (the snippet must occur EXACTLY once, else the test errors -- a vacuous mutation proves nothing; it must then report >= 1 failure).
  * DEFAULT-PRESERVING: the pre-RL-S1 closures (blob 92b52dac) are embedded below, sliced verbatim.  The OFF path is compared to them
    (a) structurally (AST equality) and (b) numerically: the loss values AND a multi-step AdamW parameter hash on tiny scorer modules, bit for bit,
    across b-modes listnet / compw / expreward, with and without the lane head.
"""
from __future__ import annotations

import ast
import copy
import contextlib
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import textwrap
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC_PATH = os.path.join(HERE, "scorer_finetune.py")

# ---- BASELINE: sliced verbatim from blob 92b52dac (scorer_finetune.py before RL-S1).  Do not edit; regenerate from the blob.

BASE_TOP_AGG_V1 = r'''def agg_v1(p):
    """planner.aggregate, rule navsim_v1, on [..., 6] probabilities OR labels: NC*DAC*(5EP+5TTC+2C)/12"""
    return p[..., 0] * p[..., 1] * (V1_W[0] * p[..., 2] + V1_W[1] * p[..., 3] + V1_W[2] * p[..., 4]) / sum(V1_W)
'''

BASE_TOP_EXPECTED_REWARD_LOSS = r'''def expected_reward_loss(sx, tg, tau_s, sel=None):
    """-E_{i ~ pi}[R_i], pi = softmax(log agg_v1(sigmoid(sx)) / tau_s) over each set, R_i = agg_v1(labels) (the true score
    of hypothesis i). A one-step choice whose outcome is KNOWN for every action: this is the policy-gradient objective in
    closed form -- no sampled action, so no sampling variance. As tau_s -> 0, pi -> the planner's argmax."""
    p = torch.sigmoid(sx)
    s = torch.log((agg_v1(p) if sel is None else sel(p)).clamp_min(1e-6))
    pi = torch.softmax(s / tau_s, dim=1)
    return -(pi * agg_v1(tg[..., :6])).sum(1).mean()
'''

BASE_NEST_LANE_BCE = r'''def lane_bce(sx, tg):
    if not a.lane_head:
        return torch.zeros((), device=sx.device)
    m = tg[..., 6] >= 0
    if not bool(m.any()):
        return torch.zeros((), device=sx.device)
    return F.binary_cross_entropy_with_logits(sx[..., 6][m], tg[..., 6][m])
'''

BASE_NEST_SEL_SCORE = r'''def sel_score(p):
    """the selection score: navsim_v1 over the six components, x the lane output when there is one"""
    return agg_v1(p[..., :6]) * (p[..., 6] if (a.lane_head and p.shape[-1] == 7) else 1.0)
'''

BASE_NEST_LOSSES = r'''def losses(sxA, sxB, tg, lam):
    if a.lane_head or tg.shape[-1] == 7:
        la, lb = a.lane_weight * lane_bce(sxA, tg), a.lane_weight * lane_bce(sxB, tg)
        sxA6, sxB6, tg6 = sxA[..., :6], sxB[..., :6], tg[..., :6]
        bA, bB, ln = losses6(sxA6, sxB6, tg6, lam, sxB)
        return bA + la, bB + lb, ln
    return losses6(sxA, sxB, tg, lam, sxB)
'''

BASE_NEST_LOSSES6 = r'''def losses6(sxA, sxB, tg, lam, sxB_full):
    bceA = F.binary_cross_entropy_with_logits(sxA, tg)
    if a.b_mode == "compw":                              # SFT-2: weighted per-component BCE, no ranking term
        w = compw.view(1, 1, 6)
        bceBw = (F.binary_cross_entropy_with_logits(sxB, tg, reduction="none") * w).sum(-1).mean() / w.sum()
        return bceA, bceBw, torch.zeros((), device=sxB.device)
    if a.b_mode == "expreward":                          # SFT-3: the selection's EXPECTED true score, all 64 outcomes known
        bceB = F.binary_cross_entropy_with_logits(sxB, tg)
        er = expected_reward_loss(sxB_full, tg, a.tau_s, sel=sel_score)
        return bceA, bceB + lam * er, er
    bceB = F.binary_cross_entropy_with_logits(sxB, tg)
    s = torch.log(agg_v1(torch.sigmoid(sxB)).clamp_min(1e-6))
    q = torch.softmax(agg_v1(tg) / a.tau_t, dim=1)
    listnet = -(q * torch.log_softmax(s, dim=1)).sum(1).mean()
    return bceA, bceB + lam * listnet, listnet
'''

BASE_PARSER_BLOCK = r'''ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--backbone", default="vitl16")
ap.add_argument("--targets", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--calib", required=True)
ap.add_argument("--onpolicy", required=True, help="the training on-policy sets dir")
ap.add_argument("--heldout", required=True, help="the held-out on-policy sets dir")
ap.add_argument("--heldout-logs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--lam", type=float, default=0.3, help="arm B: ListNet weight")
ap.add_argument("--tau-t", type=float, default=0.1, help="arm B: target temperature on the true navsim_v1 score")
ap.add_argument("--eval-every", type=int, default=400, help="optimizer updates between held-out evals")
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--preflight", action="store_true")
ap.add_argument("--max-updates", type=int, default=0)
# SFT-2 (LANE-1 follow-on); all default OFF = SFT-1 exactly
ap.add_argument("--lane-labels-train", default="", help="side files of onpolicy_relabel_lane.py for --onpolicy")
ap.add_argument("--lane-labels-heldout", default="", help="side files of onpolicy_relabel_lane.py for --heldout")
ap.add_argument("--b-mode", default="listnet", choices=("listnet", "compw", "expreward"),
                help="arm B: BCE + lam*ListNet (SFT-1) or per-component weighted BCE (--b-compw)")
ap.add_argument("--b-compw", default="1,1,1,1,1,1", help="arm B component weights NC,DAC,EP,TTC,C,DDC (b-mode compw)")
# SFT-4 (eval/PREREG_SFT4.md): the paper's PDM targets + the teacher's centre-line distance as a 7th output; default OFF
ap.add_argument("--pdm-labels-train", default="", help="side files of onpolicy_relabel_pdm.py for --onpolicy")
ap.add_argument("--pdm-labels-heldout", default="", help="side files of onpolicy_relabel_pdm.py for --heldout")
ap.add_argument("--require-pdm", action="store_true", help="train and evaluate ONLY on sets that carry PDM targets")
ap.add_argument("--teacher-lane-train", default="", help="side files of onpolicy_relabel_teacher_lane.py (train)")
ap.add_argument("--teacher-lane-heldout", default="", help="side files of onpolicy_relabel_teacher_lane.py (held-out)")
ap.add_argument("--lane-head", action="store_true", help="a 7th scorer output trained on the teacher lane label")
ap.add_argument("--lane-weight", type=float, default=1.0)
ap.add_argument("--tau-s", type=float, default=0.1,
                help="b-mode expreward: temperature of the selection policy softmax(log agg_v1(p) / tau_s)")
'''

BASE_LANE_BLOCK = r'''if a.lane_head:
    old = model.score_head
    new = nn.Linear(old.in_features, 7).to(dev)
    with torch.no_grad():
        new.weight.zero_(); new.bias.zero_()
        new.weight[:6].copy_(old.weight); new.bias[:6].copy_(old.bias)
        new.bias[6] = 1.5                                   # ~0.82: the mean of the soft lane target on held-out
        if os.environ.get("REFE_SFT4_MUTATE_G3") == "1":   # deliberate regression: a mis-copied row 0-5 must turn G3 RED
            new.weight[2, 0] += 1e-3
    model.score_head = new
    for p in model.score_head.parameters():
        p.requires_grad_(True)
'''



def load(src: str, name: str):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


def ast_eq(x: str, y: str) -> bool:
    return ast.dump(ast.parse(x)) == ast.dump(ast.parse(y))


class Ctx:
    """The source under test: parsed, imported as a module, and its main() closures extractable as text."""

    def __init__(self, src: str, tag: str):
        self.src, self.lines = src, src.split("\n")
        self.tree = ast.parse(src)
        self.main = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.mod = load(src, tag)
        self.nested = {n.name: n for n in self.main.body if isinstance(n, ast.FunctionDef)}

    def seg(self, node) -> str:
        return textwrap.dedent("\n".join(self.lines[node.lineno - 1:node.end_lineno])) + "\n"

    def nsrc(self, name: str) -> str:
        return self.seg(self.nested[name])

    def ns(self, names, **extra) -> dict:
        """a namespace holding the module's globals + `extra` (the closure variables) + the named main() closures exec'd from their SOURCE"""
        d = dict(vars(self.mod))
        d.update(extra)
        for n in names:
            exec(compile(self.nsrc(n), SRC_PATH, "exec"), d)
        return d

    def main_stmt(self, pred) -> str:
        """source of the first direct statement of main() satisfying pred(node)"""
        for n in self.main.body:
            if pred(n):
                return self.seg(n)
        raise KeyError("no such statement in main()")

    def main_has(self, pred) -> bool:
        return any(pred(n) for n in ast.walk(self.main))


def base_ns(**extra) -> dict:
    """the PRE-RL-S1 closures (blob 92b52dac), exec'd from the embedded verbatim text"""
    d = {"torch": torch, "F": F, "np": np, "nn": nn, "os": os, "V1_W": (5.0, 5.0, 2.0)}
    d.update(extra)
    for s in (BASE_TOP_AGG_V1, BASE_TOP_EXPECTED_REWARD_LOSS, BASE_NEST_LANE_BCE, BASE_NEST_SEL_SCORE, BASE_NEST_LOSSES, BASE_NEST_LOSSES6):
        exec(s, d)
    return d


# ------------------------------------------------------------------------------------------------ tiny scorer modules (the SCORER names)
class Tiny(nn.Module):
    """stand-in for the three scorer modules: same names (score_q_mlp, score_dec, score_head), same forward signature (trajs, sctx)"""

    def __init__(self, seed: int, n_out: int):
        super().__init__()
        self.score_q_mlp, self.score_dec, self.score_head = nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, n_out)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)

    def forward(self, cxy, sctx):
        return self.score_head(torch.tanh(self.score_dec(self.score_q_mlp(cxy) + sctx)))


def micro_data(n_out: int, n_micro: int, seed: int, b: int = 4, m: int = 8):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_micro):
        cxy, sctx = torch.randn(b, m, 4, generator=g), torch.randn(b, m, 8, generator=g)
        tg = torch.rand(b, m, 6, generator=g)
        tg[..., :2] = (tg[..., :2] > 0.3).float()                           # NC / DAC are hard 0 / 1 like the real labels
        if n_out == 7:
            lane = torch.rand(b, m, 1, generator=g)
            lane[torch.rand(b, m, 1, generator=g) < 0.3] = -1.0              # -1 = no teacher-lane label (masked)
            tg = torch.cat([tg, lane], -1)
        out.append((cxy, sctx, tg))
    return out


def phash(*mods) -> str:
    h = hashlib.sha256()
    for m in mods:
        for k, v in m.state_dict().items():
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def bits(*xs) -> tuple:
    return tuple(struct.pack("<f", float(x.detach() if torch.is_tensor(x) else x)).hex() for x in xs)


def run_arms(step_fn, n_out: int, steps: int = 3, accum: int = 2, seed: int = 0, lr: float = 1e-2):
    """main()'s training loop in miniature: arm A / arm B / a frozen base, AdamW + clip 1.0 every `accum` micro-batches.
    step_fn(sx0, sxA, sxB, tg, upd) -> (lA, lB, ln, ...).  Returns the per-micro-batch loss bit patterns and the parameter hash after the run."""
    torch.manual_seed(seed)
    A, B = Tiny(seed, n_out), Tiny(seed, n_out)
    base = copy.deepcopy(A)
    for p in base.parameters():
        p.requires_grad_(False)
    optA = torch.optim.AdamW(list(A.parameters()), lr=lr, weight_decay=0.01)
    optB = torch.optim.AdamW(list(B.parameters()), lr=lr, weight_decay=0.01)
    rec, upd = [], 0
    for i, (cxy, sctx, tg) in enumerate(micro_data(n_out, steps * accum, seed + 1)):
        sxA, sxB = A(cxy, sctx), B(cxy, sctx)
        with torch.no_grad():
            sx0 = base(cxy, sctx)
        out = step_fn(sx0, sxA, sxB, tg, upd)
        lA, lB, ln = out[0], out[1], out[2]
        ((lA + lB) / accum).backward()
        rec.append(bits(lA, lB, ln))
        if (i + 1) % accum == 0:
            for o, ps in ((optA, list(A.parameters())), (optB, list(B.parameters()))):
                torch.nn.utils.clip_grad_norm_(ps, 1.0)
                o.step()
                o.zero_grad(set_to_none=True)
            upd += 1
    return rec, phash(A, B), phash(A), phash(B)


COMPW = [1.0, 2.0, 1.0, 1.0, 0.5, 1.0]


def a_ns(b_mode: str, lane: bool, **kw):
    d = dict(b_mode=b_mode, lane_head=lane, lane_weight=0.7, tau_t=0.1, tau_s=0.1, lam=0.3, epo=False)
    d.update(kw)
    return types.SimpleNamespace(**d)


# ------------------------------------------------------------------------------------------------ SMOKE: the REAL main(), on CPU, with stubbed heavy parts
# The stubs replace only what needs a GPU / a 6 GB checkpoint / the nuPlan corpus: REFe (three scorer modules with the real NAMES, a frozen "trunk"),
# train.OnPolicyBank / TargetBank, ckpt_io's loader.  Everything else is scorer_finetune.py's own text -- the data section, the preflight (G1-G11),
# the training loop, the logs, save_full -- so a typo in glue no test above executes surfaces here, not on the pod.
class StubBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x, ctx):
        return torch.tanh(self.lin(x + ctx))


class StubREFe(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.trunk, self.traj_head = nn.Linear(3, 8), nn.Linear(8, 2)
        self.score_q_mlp, self.score_dec, self.score_head = nn.Linear(4, 8), nn.Sequential(StubBlock()), nn.Linear(8, cfg.n_score_components)

    def score_trajectories(self, trajs, sctx):                     # called as REFe.score_trajectories(ScorerCopy, ...) exactly like the real one
        return self.score_head(self.score_dec[0](self.score_q_mlp(trajs), sctx))

    def forward(self, img, ego, goal, calib=None, score_extra=None):
        sctx = self.trunk(img).detach()                            # the real scoring decoder reads the context DETACHED
        return self.traj_head(sctx), None, self.score_trajectories(score_extra, sctx)


class StubCfgFactory:
    @staticmethod
    def for_backbone(_b):
        return types.SimpleNamespace(n_score_components=6, nav_mode="goal", n_proposals=8, horizon_steps=4)


N_TRAIN, N_HELD, M_PROP = 100, 12, 8


class StubBank:
    def __init__(self, path, n_prop, horizon, lane_labels=None, pdm_labels=None, teacher_lane_labels=None):
        train = "train" in path
        g = np.random.default_rng(1 if train else 2)
        self.by = {}
        for i in range(N_TRAIN if train else N_HELD):
            tg = g.random((n_prop, 6)).astype(np.float32)
            tg[:, :2] = (tg[:, :2] > 0.3)
            self.by[(f"{'T' if train else 'H'}{i // 4}", f"tok{i}", i, 0)] = (5, None, tg, (5, 2), False)
        self.pdm_keys = set(self.by) if pdm_labels else set()
        self.n_pdm, self.n_pdm_stale, self.n_pdm_superseded = len(self.pdm_keys), 0, 0
        self.n_navsim_dac = self.n_lane = self.n_lane_stale = self.n_lane_superseded = 0
        self.lane_keep: dict = {}
        self.teacher_lane = {k: g.random(n_prop).astype(np.float32) for k in list(self.by)[::2]} if teacher_lane_labels else {}
        self.n_tlane, self.n_cols = len(self.teacher_lane), (7 if teacher_lane_labels else 6)


class StubTargets(torch.utils.data.Dataset):
    def __init__(self, targets, images, cfg, calib=None, onpolicy=None):
        self.onpolicy = onpolicy
        tr, ho = StubBank("train", 8, 4), StubBank("held", 8, 4)
        self.rows = [{"log_name": k[0], "token": k[1], "step": k[2], "rank": k[3]} for k in list(tr.by) + list(ho.by)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        k = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]))
        e = self.onpolicy.by[k]
        g = torch.Generator().manual_seed(1000 + i)
        tg = torch.as_tensor(e[2])
        if self.onpolicy.n_cols == 7:
            lane = self.onpolicy.teacher_lane.get(k)
            tg = torch.cat([tg, torch.as_tensor(lane if lane is not None else np.full(8, -1.0, np.float32))[:, None]], -1)
        return (torch.randn(8, 3, generator=g), torch.zeros(4), torch.zeros(2), torch.zeros(1), torch.randn(8, 4, generator=g), tg, torch.ones(8),
                torch.full((8,), 5), torch.zeros(0))


def smoke(src: str, td: str, *, epo: bool, epo_lam: float = 1.0, native7: bool = False, preflight_first: bool = True, terms=True, tag: str = "x") -> dict:
    """run the real main() (preflight, then train) on CPU; returns {exit codes, log rows, saved scorer tensors}"""
    import ckpt_io as real_ckpt_io
    csrc = src.replace('dev = "cuda"', 'dev = "cpu"').replace("pin_memory=True", "pin_memory=False").replace("num_workers=2", "num_workers=0").replace("n=10000, seed=20260927", "n=40, seed=20260927")
    csrc = csrc.replace('torch.autocast("cuda", dtype=torch.bfloat16)', 'torch.autocast("cpu", enabled=False)')
    sm = load(csrc, f"sft_smoke_{tag}")

    class StubCkptIO:
        FORMAT_FULL = "full"
        config_for_checkpoint = staticmethod(real_ckpt_io.config_for_checkpoint)

        @staticmethod
        def load_for_inference(model, path, map_location="cpu", backbone=None, meta_out=None):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(sd["model"], strict=True)
            if meta_out is not None:
                meta_out.update(sd.get("meta", {}) or {})
            return "full"

        @staticmethod
        def trainable_names(model):
            return {n for n, p in model.named_parameters() if p.requires_grad}

        @staticmethod
        def atomic_save(obj, path):
            torch.save(obj, path)
    sm.REFe, sm.REFeConfig, sm.ckpt_io, sm.T = StubREFe, StubCfgFactory, StubCkptIO, types.SimpleNamespace(OnPolicyBank=StubBank, TargetBank=StubTargets)
    cfg7 = types.SimpleNamespace(n_score_components=7 if native7 else 6)
    torch.manual_seed(0)
    ck = os.path.join(td, "ck.pt")
    torch.save({"format": "full", "model": StubREFe(cfg7).state_dict(), "meta": {"n_score_components": 7} if native7 else {}}, ck)
    out, tdir = os.path.join(td, "out"), os.path.join(td, "terms")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(td, "ho_logs.txt"), "w") as fh:
        fh.write(" ".join(f"H{i}" for i in range(4)))
    gt = np.random.default_rng(3)
    for nm, bank in (("train", StubBank("train", 8, 4)), ("held", StubBank("held", 8, 4))):
        os.makedirs(os.path.join(tdir, nm), exist_ok=True)
        with open(os.path.join(tdir, nm, "rl_terms_a.jsonl"), "w") as fh:
            for k in bank.by:
                fh.write(json.dumps({"kind": "rl_terms", "key": list(k), "ckpt_step": 5, "d": gt.random(8).round(3).tolist()}) + "\n")
    args = ["--ckpt", ck, "--targets", "t", "--images", "i", "--calib", "c", "--onpolicy", "train", "--heldout", "held", "--heldout-logs",
            os.path.join(td, "ho_logs.txt"), "--out", out, "--batch", "2", "--accum", "2", "--workers", "0", "--max-updates", "25", "--eval-every", "25",
            "--warmup", "2", "--lr", "1e-3", "--b-mode", "listnet", "--lam", "0", "--pdm-labels-train", "d", "--pdm-labels-heldout", "d", "--require-pdm"]
    if native7:
        args += ["--lane-head", "--teacher-lane-train", "x", "--teacher-lane-heldout", "y"]
    if epo:
        args += ["--epo", "--epo-ramp", "2", "--epo-lam", str(epo_lam), "--epo-terms-heldout", os.path.join(tdir, "held")]
        if terms is True:
            args += ["--epo-terms-train", os.path.join(tdir, "train")]
        elif terms is False:
            args += ["--epo-lam-d", "0"]                     # D named off: runs without side files
    saved_argv, saved_cuda = sys.argv, (torch.cuda.synchronize, torch.cuda.max_memory_allocated)
    torch.cuda.synchronize, torch.cuda.max_memory_allocated = (lambda *a, **k: None), (lambda *a, **k: 0)
    res: dict = {}
    try:
        for phase in (["--preflight"], []) if preflight_first else ([],):
            sys.argv = ["scorer_finetune.py"] + args + phase
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = sm.main()
            res["pre" if phase else "run"] = rc
            res["out_" + ("pre" if phase else "run")] = buf.getvalue()
    finally:
        sys.argv = saved_argv
        torch.cuda.synchronize, torch.cuda.max_memory_allocated = saved_cuda
    rows = {}
    for nm in ("preflight.jsonl", "sft.jsonl"):
        p = os.path.join(out, nm)
        rows[nm] = [json.loads(x) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []
    res["rows"] = rows
    res["out_exists"] = os.path.isdir(out)
    pf = os.path.join(out, "PREFLIGHT_PASS.json")
    res["pass_json"] = json.load(open(pf)) if os.path.exists(pf) else None
    for arm in ("A", "B"):
        p = os.path.join(out, f"model_sft_{arm}.pt")
        res[arm] = torch.load(p, weights_only=False)["model"] if os.path.exists(p) else None
    return res


def scorer_equal(x, y) -> bool:
    ks = [k for k in x if k.split(".")[0] in ("score_q_mlp", "score_dec", "score_head")]
    return bool(ks) and all(torch.equal(x[k], y[k]) for k in ks)


# ------------------------------------------------------------------------------------------------ the suite
def suite(C: Ctx) -> list:
    """Every check against the source in `C`.  Returns the list of FAILING check names (empty = green)."""
    bad: list = []

    def t(name, fn):
        try:
            ok = bool(fn())
        except Exception:                                                       # noqa: BLE001 -- an exception IS a failed check
            ok = False
        if not ok:
            bad.append(name)

    mod = C.mod
    # ======================================================================== PRIORITY 1: the u400 loading fix + build_parser(), default-preserving
    # ---- the pre-RL-S1 closures and helpers are unchanged, structurally
    t("p1/closures_verbatim", lambda: all(ast_eq(C.nsrc(n), s) for n, s in (
        ("lane_bce", BASE_NEST_LANE_BCE), ("sel_score", BASE_NEST_SEL_SCORE), ("losses", BASE_NEST_LOSSES), ("losses6", BASE_NEST_LOSSES6))))
    t("p1/top_helpers_verbatim", lambda: all(ast_eq(C.seg(next(n for n in C.tree.body if isinstance(n, ast.FunctionDef) and n.name == nm)), s)
                                              for nm, s in (("agg_v1", BASE_TOP_AGG_V1), ("expected_reward_loss", BASE_TOP_EXPECTED_REWARD_LOSS))))

    # ---- the OFF path is bit-identical to the baseline: loss values AND the parameter hash after a multi-step run
    for bm in ("listnet", "compw", "expreward"):
        for lane in (False, True):
            def off_identical(bm=bm, lane=lane):
                compw = torch.tensor(COMPW)
                an = a_ns(bm, lane)
                bns = base_ns(a=an, compw=compw)
                rns = C.ns(REAL_OFF_NAMES(C), a=an, compw=compw, rcfg=None, terms_train=None, ds=None, epo_last={})
                n_out = 7 if lane else 6
                want = run_arms(lambda sx0, sxA, sxB, tg, upd: bns["losses"](sxA, sxB, tg, an.lam), n_out)
                got = run_arms(REAL_STEP(rns, an), n_out)
                return want == got and len(set(want[0])) > 1                    # identical, and the run actually moved (not a constant)
            t(f"p1/off_bit_identical[{bm},{'lane' if lane else '6col'}]", off_identical)

    # ---- the parser: every pre-RL-S1 flag is unchanged in every attribute; new flags exist
    def parser_preserved():
        P0 = types.SimpleNamespace()
        d = {"argparse": __import__("argparse")}
        exec(BASE_PARSER_BLOCK, d)
        old, new = d["ap"], mod.build_parser()
        nm = {tuple(x.option_strings): x for x in new._actions}
        for x in old._actions:
            y = nm.get(tuple(x.option_strings))
            if y is None:
                return False
            for at in ("dest", "default", "type", "choices", "required", "nargs", "const", "help"):
                if getattr(x, at) != getattr(y, at):
                    return False
            if type(x) is not type(y):
                return False
        return True
    t("p1/parser_baseline_flags_unchanged", parser_preserved)
    t("p1/parser_returns_parser", lambda: isinstance(mod.build_parser(), __import__("argparse").ArgumentParser))
    t("p1/main_uses_build_parser", lambda: C.main_has(lambda n: isinstance(n, ast.Call) and getattr(n.func, "id", "") == "build_parser"))

    # ---- u400: the config is sized from the checkpoint (executes main()'s OWN `cfg = ...` statement)
    def cfg_stmt():
        return C.main_stmt(lambda n: isinstance(n, ast.Assign) and any(getattr(x, "id", "") == "cfg" for x in n.targets))

    class StubCfgFactory:
        @staticmethod
        def for_backbone(_b):
            return types.SimpleNamespace(n_score_components=6, nav_mode="goal", n_proposals=64)

    def cfg_from_ckpt(meta):
        import ckpt_io
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "ck.pt")
            torch.save({"format": "full", "model": {}, "meta": meta}, p)
            d = {"ckpt_io": ckpt_io, "REFeConfig": StubCfgFactory, "a": types.SimpleNamespace(backbone="vitl16", ckpt=p)}
            exec(cfg_stmt(), d)
            return d["cfg"]
    t("p1/u400_cfg_sized_from_checkpoint", lambda: cfg_from_ckpt({"n_score_components": 7}).n_score_components == 7)
    t("p1/cfg_unchanged_without_meta", lambda: cfg_from_ckpt({}).n_score_components == 6 and cfg_from_ckpt({}).nav_mode == "goal")
    t("p1/cfg_nav_mode_from_meta", lambda: cfg_from_ckpt({"nav_mode": "command"}).nav_mode == "command")

    # ---- the lane-head extension: a 6-row head is extended EXACTLY as before; a 7-row head is left alone (and consumes no RNG)
    def lane_ext_same_as_baseline(env=None):
        outs = []
        for use_new in (False, True):
            torch.manual_seed(11)
            lin = nn.Linear(8, 6)
            model = types.SimpleNamespace(score_head=lin)
            if use_new:
                mod.extend_lane_head(model, "cpu")
            else:
                exec(BASE_LANE_BLOCK, {"a": types.SimpleNamespace(lane_head=True), "model": model, "nn": nn, "torch": torch, "os": os, "dev": "cpu"})
            h = model.score_head
            outs.append((h.weight.detach().numpy().tobytes(), h.bias.detach().numpy().tobytes(), h.out_features,
                         all(p.requires_grad for p in h.parameters()), torch.get_rng_state().numpy().tobytes()))
        return outs[0] == outs[1] and outs[0][2] == 7
    t("p1/lane_extension_identical_to_baseline", lane_ext_same_as_baseline)

    def lane_skip_when_native7():
        torch.manual_seed(3)
        h7 = nn.Linear(8, 7)
        with torch.no_grad():
            h7.weight.mul_(3.0); h7.bias.add_(0.37)                          # a TRAINED head: not zero rows, not the 1.5 init
        w0, b0 = h7.weight.detach().clone(), h7.bias.detach().clone()
        model = types.SimpleNamespace(score_head=h7)
        rng0 = torch.get_rng_state().clone()
        out = mod.extend_lane_head(model, "cpu")
        return (out is h7 and model.score_head is h7 and torch.equal(h7.weight, w0) and torch.equal(h7.bias, b0)
                and torch.equal(torch.get_rng_state(), rng0))
    t("p1/lane_head_untouched_when_native7", lane_skip_when_native7)

    # ---- main()'s wiring (the hops from the caller)
    def calls_extend_under_lane_head():
        for n in ast.walk(C.main):
            if isinstance(n, ast.If) and ast.unparse(n.test) == "a.lane_head":
                if any(isinstance(m, ast.Call) and getattr(m.func, "id", "") == "extend_lane_head" for m in ast.walk(n)):
                    return True
        return False
    t("p1/main_calls_extend_lane_head_under_flag", calls_extend_under_lane_head)

    def refuses_native7_without_flag():
        for n in ast.walk(C.main):
            if isinstance(n, ast.If) and ast.unparse(n.test).replace(" ", "").replace("(", "").replace(")", "") == "native7andnota.lane_head":
                return any(isinstance(m, ast.Return) and getattr(m.value, "value", None) == 4 for m in n.body)
        return False
    t("p1/main_refuses_7row_head_without_lane_flag", refuses_native7_without_flag)
    t("p1/native7_from_the_loaded_head", lambda: any(isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "native7"
                                                       and "model.score_head.out_features" in ast.unparse(n.value) for n in C.main.body))

    # ======================================================================== PRIORITY 2: the EPO loss, default OFF
    import argparse
    import rl_reward as RW

    def raises(fn, exc=Exception):
        try:
            fn()
        except exc:
            return True
        return False

    CL = ["lane_bce", "sel_score", "losses", "losses6", "epo_term", "train_losses"]

    # ---- the maths of epo_loss against HAND-COMPUTED values.  rule = column 0 as the score, so A = sigmoid(sx) = (0.5, 0.25, 0.25), R = (1, 0, 0.5)
    rule0 = lambda p: p[..., 0]                                                                           # noqa: E731
    sx_of = lambda av: torch.logit(torch.tensor(av, dtype=torch.float32))[None, :, None]                  # noqa: E731  [1, M, 1] logits of probabilities av
    A3, R3 = [0.5, 0.25, 0.25], torch.tensor([[1.0, 0.0, 0.5]])
    close = lambda x, y, tol=2e-5: abs(float(x) - float(y)) < tol                                         # noqa: E731
    # T = 1: pi = A / sum(A) = (0.5, 0.25, 0.25) -> J = 0.5*1 + 0.25*0 + 0.25*0.5 = 0.625   (ref == policy -> KL = 0)
    t("p2/J_hand_T1", lambda: close(mod.epo_loss(sx_of(A3), sx_of(A3), R3, rule0, 1.0, 0.1, 1.0), -0.625))
    # T = 0.5: pi ~ A^2 = (0.25, 0.0625, 0.0625) / 0.375 = (2/3, 1/6, 1/6) -> J = 2/3 + 0 + 1/12 = 0.75
    t("p2/J_hand_T05", lambda: close(mod.epo_loss(sx_of(A3), sx_of(A3), R3, rule0, 0.5, 0.1, 1.0), -0.75))
    # KL(pi_SFT || pi): pi_SFT = (0.1, 0.3, 0.6), pi = (0.5, 0.25, 0.25):
    #   0.1 ln(0.2) + 0.3 ln(1.2) + 0.6 ln(2.4) = -0.16094379 + 0.05469647 + 0.52528124 = 0.41903392   (the REVERSE direction is 0.54027138)
    # loss = lam * (-J + beta * KL) = 2 * (-0.625 + 0.5 * 0.41903392) = -0.83096608
    SX_REF = sx_of([0.1, 0.3, 0.6])
    t("p2/kl_anchor_forward_hand", lambda: close(mod.epo_loss(sx_of(A3), SX_REF, R3, rule0, 1.0, 0.5, 2.0), -0.83096608))
    t("p2/parts_J_and_kl", lambda: (lambda pr: (mod.epo_loss(sx_of(A3), SX_REF, R3, rule0, 1.0, 0.5, 2.0, pr), close(pr["J"], 0.625), close(pr["kl"], 0.41903392))[1:])({}) == (True, True))

    def grad_hand():
        # dJ/dsx_j = (R_j - J) * A_j * (1 - A_j) / sum(A):  (0.375*0.25, -0.625*0.1875, -0.125*0.1875) = (0.09375, -0.1171875, -0.0234375); loss = -J
        sx = sx_of(A3).clone().requires_grad_(True)
        mod.epo_loss(sx, sx_of(A3), R3, rule0, 1.0, 0.1, 1.0).backward()
        return np.allclose(sx.grad.flatten().numpy(), [-0.09375, 0.1171875, 0.0234375], atol=2e-5)
    t("p2/gradient_hand", grad_hand)

    def lam0_exact():
        sx = sx_of(A3).clone().requires_grad_(True)
        v = mod.epo_loss(sx, SX_REF, R3, rule0, 0.5, 0.5, 0.0)
        v.backward()
        return float(v) == 0.0 and int(torch.count_nonzero(sx.grad)) == 0
    t("p2/lam_zero_value_and_grad_exactly_zero", lam0_exact)

    def no_grad_to_reward_or_reference():
        Rr = R3.clone().requires_grad_(True)
        ref = SX_REF.clone().requires_grad_(True)
        sx = sx_of(A3).clone().requires_grad_(True)
        mod.epo_loss(sx, ref, Rr, rule0, 1.0, 0.5, 1.0).backward()
        return Rr.grad is None and ref.grad is None and sx.grad is not None
    t("p2/reward_and_reference_receive_no_gradient", no_grad_to_reward_or_reference)
    t("p2/ramp_hand", lambda: (mod.epo_ramp(0, 200), mod.epo_ramp(100, 200), mod.epo_ramp(200, 200), mod.epo_ramp(400, 200), mod.epo_ramp(5, 0)) == (0.0, 0.5, 1.0, 1.0, 1.0))

    # ---- the reward plumbing: HAND-COMPUTED scene (rl_reward's own fixture): R = (1, 0.5116667, -0.15, 0.19) with lam_d = 0.15, D = (0, .2, 1, .4)
    TG = np.array([[1, 1, 1.0, 1, 1, 1], [1, 1, 0.5, 1, 1, 1], [0, 1, 0.0, 1, 1, 1], [1, 1, 0.2, 0, 1, 1]])
    D4 = np.array([0.0, 0.2, 1.0, 0.4])
    t("p2/epo_reward_hand", lambda: np.allclose(mod.epo_reward(TG[None], [{"d": D4}], RW.RewardCfg()), [[1.0, 0.51166667, -0.15, 0.19]], atol=1e-6, rtol=0))
    # lam_d = 0 named: R = PDMS (1, .7916667, 0, .25) with P1's guard penalty 0.25 (cap 0.5*(.7916667-.25) = .2708 does not bind) -> (1, .5416667, 0, .25)
    t("p2/epo_reward_lam_d_zero_hand", lambda: np.allclose(mod.epo_reward(TG[None], None, RW.RewardCfg(lam_d=0.0)), [[1.0, 0.54166667, 0.0, 0.25]], atol=1e-6, rtol=0))
    t("p2/epo_reward_missing_terms_raises", lambda: raises(lambda: mod.epo_reward(TG[None], None, RW.RewardCfg()))
       and raises(lambda: mod.epo_reward(TG[None], [None], RW.RewardCfg())))

    def reward_batch_matches_per_set():
        rng = np.random.default_rng(5)
        tgb = np.clip(rng.random((3, 4, 6)), 0, 1)
        tgb[..., :2] = (tgb[..., :2] > 0.3)
        recs = [{"d": rng.random(4).astype(np.float32), "k": rng.random(4).astype(np.float32), "lc_executed": rng.random(4) > 0.5,
                 "label_fires": bool(i % 2), "dither": (rng.random(4) > 0.7).astype(np.float32)} for i in range(3)]
        cfg = RW.RewardCfg(use_k=True, use_l=True)
        got = mod.epo_reward(tgb, recs, cfg)
        for i in range(3):
            want = RW.reward(tgb[i], cfg, d=recs[i]["d"], k=recs[i]["k"],
                             lane={"lc_executed": recs[i]["lc_executed"], "label_fires": recs[i]["label_fires"], "dither": recs[i]["dither"]})["R"]
            if not np.allclose(got[i], want, atol=1e-9, rtol=0):
                return False
        return True
    t("p2/epo_reward_batch_equals_per_set_reward_with_K_and_L", reward_batch_matches_per_set)

    # ---- the side-file loader (OnPolicyBank's own staleness rule) and the command-line refusals
    def loader_stats():
        key1, key2, key3, key9 = ("L", "t", 1, 0), ("L", "t", 2, 0), ("L", "t", 3, 0), ("L", "t", 9, 0)
        bank = {key1: (5,), key2: (5,), key3: (5,)}
        ok4 = [0.1] * 4

        def ln(key, ck=5, **kw):
            d = {"kind": "rl_terms", "key": list(key), "ckpt_step": ck, "d": ok4}
            d.update(kw)
            return json.dumps(d)
        lines = [ln(key1), ln(key2, ck=4), ln(key3, ck=6), ln(key9), ln(key1, d=[0.1] * 3), ln(key1, d=[float("nan")] * 4), ln(key1, d=[1.5] * 4),
                 "{not json", json.dumps({"kind": "other"}), ln(key1, d=[0.2] * 4), json.dumps({"kind": "rl_terms", "key": ["L"], "ckpt_step": 5, "d": ok4})]
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "rl_terms_a.jsonl"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
            terms, st = mod.load_rl_terms(td, bank, 4, True, False, False)
            _t2, st2 = mod.load_rl_terms(td, bank, 4, False, False, False)             # D not needed -> the d-range / length lines are not incomplete
        return (st == {"files": 1, "lines": 9, "matched": 1, "stale": 1, "superseded": 1, "incomplete": 4, "not_in_bank": 1, "bad_lines": 1, "duplicates": 1}
                and list(terms) == [key1] and np.allclose(terms[key1]["d"], 0.2) and st2["incomplete"] == 1 and st2["matched"] == 1)
    t("p2/terms_loader_stale_superseded_incomplete_duplicates", loader_stats)

    def loader_k_l_fields():
        key1 = ("L", "t", 1, 0)
        full = {"kind": "rl_terms", "key": list(key1), "ckpt_step": 5, "d": [0.1] * 2, "k": [0.2] * 2, "lc_executed": [True, False], "label_fires": True, "dither": [0, 1]}
        nolc = {k: v for k, v in full.items() if k != "lc_executed"}
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "rl_terms_a.jsonl"), "w", encoding="utf-8").write(json.dumps(full) + "\n")
            tf, sf = mod.load_rl_terms(td, {key1: (5,)}, 2, True, True, True)
            open(os.path.join(td, "rl_terms_a.jsonl"), "w", encoding="utf-8").write(json.dumps(nolc) + "\n")
            tn, sn = mod.load_rl_terms(td, {key1: (5,)}, 2, True, True, True)
        return sf["matched"] == 1 and tf[key1]["label_fires"] is True and tf[key1]["lc_executed"].tolist() == [True, False] and sn["matched"] == 0 and sn["incomplete"] == 1
    t("p2/terms_loader_k_and_lane_fields", loader_k_l_fields)

    REQ = ["--ckpt", "x", "--targets", "x", "--images", "x", "--calib", "x", "--onpolicy", "x", "--heldout", "x", "--heldout-logs", "x", "--out", "x"]
    EPO_OK = ["--epo", "--pdm-labels-train", "d", "--require-pdm", "--b-mode", "listnet", "--lam", "0", "--epo-terms-train", "t"]

    def refusals(extra, base=None):
        ap = mod.build_parser()
        return mod.epo_refusals(ap.parse_args(REQ + (EPO_OK if base is None else base) + extra), ap)
    t("p2/refusal_off_and_nothing_named_is_clean", lambda: mod.epo_refusals(mod.build_parser().parse_args(REQ), mod.build_parser()) == [])
    t("p2/refusal_levers_named_without_epo", lambda: len(refusals(["--epo-lam", "0.3"], base=[])) == 1 and "--epo is not" in refusals(["--epo-beta", "0.2"], base=[])[0])
    t("p2/refusal_complete_epo_argv_is_clean", lambda: refusals([]) == [])
    t("p2/refusal_no_terms_without_lam_d_zero",
      lambda: any("--epo-terms-train" in m for m in refusals([], base=[x for x in EPO_OK if x not in ("--epo-terms-train", "t")])))
    t("p2/refusal_lam_d_zero_named_allows_no_terms",
      lambda: refusals(["--epo-lam-d", "0"], base=[x for x in EPO_OK if x not in ("--epo-terms-train", "t")]) == [])
    t("p2/refusal_k_or_l_need_terms",
      lambda: len(refusals(["--epo-lam-d", "0", "--epo-k"], base=[x for x in EPO_OK if x not in ("--epo-terms-train", "t")])) == 1
      and len(refusals(["--epo-lam-d", "0", "--epo-l"], base=[x for x in EPO_OK if x not in ("--epo-terms-train", "t")])) == 1)
    t("p2/refusal_needs_pdm_labels", lambda: any("--pdm-labels-train" in m for m in refusals([], base=["--epo", "--b-mode", "listnet", "--lam", "0", "--epo-terms-train", "t"])))
    t("p2/refusal_needs_plain_bce_for_B", lambda: any("plain BCE" in m for m in refusals(["--lam", "0.3"])) and any("plain BCE" in m for m in refusals(["--b-mode", "expreward"])))
    t("p2/refusal_bad_values", lambda: len(refusals(["--epo-temp", "0"])) == 1 and len(refusals(["--epo-beta", "-1"])) == 1)

    # ---- the parser: every EPO lever exists with the registered default, and the switch is OFF
    def epo_defaults():
        a0 = mod.build_parser().parse_args(REQ)
        return ((a0.epo, a0.epo_temp, a0.epo_beta, a0.epo_ramp, a0.epo_lam, a0.epo_lam_d, a0.epo_w_prog, a0.epo_k, a0.epo_lam_k, a0.epo_l, a0.epo_lam_l,
                 a0.epo_terms_train, a0.epo_terms_heldout)
                == (False, 0.1, 0.1, 200, 1.0, 0.15, 0.5, False, 0.05, False, 0.05, "", ""))
    t("p2/parser_epo_flags_default_off", epo_defaults)

    # ---- row-key plumbing: only with --epo; both __getitem__ AND __getitems__ (the DataLoader fetcher uses the latter)
    class StubDS(torch.utils.data.Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, i):
            return tuple(torch.full((2,), float(i)) for _ in range(9))

    def indexed_subset():
        on = list(torch.utils.data.DataLoader(mod.make_train_set(StubDS(), [7, 2, 5, 3], True), batch_size=2, shuffle=False))
        off = list(torch.utils.data.DataLoader(mod.make_train_set(StubDS(), [7, 2, 5, 3], False), batch_size=2, shuffle=False))
        b9, rows = mod.split_rows(on[0], True)
        return (len(on[0]) == 10 and on[0][9].tolist() == [7, 2] and on[1][9].tolist() == [5, 3] and len(b9) == 9 and rows.tolist() == [7, 2]
                and len(off[0]) == 9 and type(mod.make_train_set(StubDS(), [1], False)) is torch.utils.data.Subset
                and mod.split_rows(off[0], False) == (off[0], None) and all(torch.equal(x, y) for x, y in zip(b9, off[0])))
    t("p2/rows_ride_the_dataloader_only_with_epo", indexed_subset)

    def same_order_on_off():
        g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
        ids = list(range(10))
        a_ = [b[0][:, 0].tolist() for b in torch.utils.data.DataLoader(mod.make_train_set(StubDS(), ids, True), batch_size=2, shuffle=True, generator=g1)]
        b_ = [b[0][:, 0].tolist() for b in torch.utils.data.DataLoader(mod.make_train_set(StubDS(), ids, False), batch_size=2, shuffle=True, generator=g2)]
        return a_ == b_
    t("p2/epo_on_off_see_the_same_sample_order", same_order_on_off)

    # ---- the assembly closures, EXEC'd from main()'s source text.  Fixture: 4 sets x 8 proposals; terms for dataset rows 0..3; the batch's rows are PERMUTED
    DS = types.SimpleNamespace(rows=[{"log_name": f"L{i}", "token": f"t{i}", "step": i, "rank": 0} for i in range(6)])
    rngd = np.random.default_rng(9)
    TERMS = {(f"L{i}", f"t{i}", i, 0): {"d": rngd.random(8).astype(np.float32)} for i in range(4)}
    ROWS = torch.tensor([3, 1, 0, 2])

    def mk(lane=False, **kw):
        base = dict(lam=0.0, epo=True, epo_lam=1.0, epo_ramp=2, epo_temp=0.1, epo_beta=0.1, accum=2)
        base.update(kw)
        an = a_ns("listnet", lane, **base)
        return an, C.ns(CL + ["epo_probe"], a=an, compw=torch.tensor(COMPW), rcfg=RW.RewardCfg(), terms_train=TERMS, ds=DS, epo_last={}, fwd=None)

    def one_microbatch(n_out):
        A, B = Tiny(0, n_out), Tiny(0, n_out)
        base = copy.deepcopy(A)
        cxy, sctx, tg = micro_data(n_out, 1, 1)[0]
        with torch.no_grad():
            sx0 = base(cxy, sctx)
        return A, B, base, (cxy, sctx, tg), sx0, A(cxy, sctx), B(cxy, sctx), tg

    def rule_v1(p):                                   # an INDEPENDENT statement of the base's rule: navsim_v1 over the six components, x the lane output when present
        v = p[..., 0] * p[..., 1] * (5.0 * p[..., 2] + 5.0 * p[..., 3] + 2.0 * p[..., 4]) / 12.0
        return v * p[..., 6] if p.shape[-1] == 7 else v

    def term_equals_reference(lane):
        n_out = 7 if lane else 6
        an, ns = mk(lane)
        A, B, base, data, sx0, sxA, sxB, tg = one_microbatch(n_out)
        ep = ns["epo_term"](sx0, sxB, tg, ROWS, 5)
        Rref = np.stack([RW.reward(tg[i, :, :6].numpy().astype(np.float64), RW.RewardCfg(), d=TERMS[(f"L{int(r)}", f"t{int(r)}", int(r), 0)]["d"])["R"]
                         for i, r in enumerate(ROWS)])
        want = mod.epo_loss(sxB, sx0, torch.as_tensor(Rref, dtype=torch.float32), rule_v1, 0.1, 0.1, 1.0)
        return abs(float(ep) - float(want)) < 1e-6 and float(ep) != 0.0
    t("p2/term_equals_independent_reference_6col", lambda: term_equals_reference(False))
    t("p2/term_equals_independent_reference_lane", lambda: term_equals_reference(True))

    def on_adds_exactly_epo_to_B_only(lane):
        n_out = 7 if lane else 6
        an, ns = mk(lane)
        _, _, _, _, sx0, sxA, sxB, tg = one_microbatch(n_out)
        on = ns["train_losses"](sx0, sxA, sxB, tg, ROWS, 5)
        an.epo = False
        off = ns["train_losses"](sx0, sxA, sxB, tg, None, 5)
        return (torch.equal(on[0], off[0]) and torch.equal(on[2], off[2]) and torch.equal(on[1], off[1] + on[3]) and on[3].requires_grad
                and float(on[3]) != 0.0 and float(off[3]) == 0.0 and not off[3].requires_grad)
    t("p2/on_adds_exactly_the_epo_term_to_B_and_not_to_A_6col", lambda: on_adds_exactly_epo_to_B_only(False))
    t("p2/on_adds_exactly_the_epo_term_to_B_and_not_to_A_lane", lambda: on_adds_exactly_epo_to_B_only(True))

    def epo_only_backward_reaches_every_B_module():
        an, ns = mk(False)
        A, B, base, _, sx0, sxA, sxB, tg = one_microbatch(6)
        ns["train_losses"](sx0, sxA, sxB, tg, ROWS, 5)[3].backward()
        gB = mod.module_grad_abs(B)
        return all(v > 0.0 and np.isfinite(v) for v in gB.values()) and all(p.grad is None for p in A.parameters()) and all(p.grad is None for p in base.parameters())
    t("p2/epo_only_backward_nonzero_on_each_B_module_and_none_on_A_or_base", epo_only_backward_reaches_every_B_module)

    def ramp_and_lam_in_the_term():
        an, ns = mk(False)
        _, _, _, _, sx0, sxA, sxB, tg = one_microbatch(6)
        e0, e1, e2, e9 = (float(ns["epo_term"](sx0, sxB, tg, ROWS, u)) for u in (0, 1, 2, 9))
        an.epo_lam = 2.0
        e9x2 = float(ns["epo_term"](sx0, sxB, tg, ROWS, 9))
        return e0 == 0.0 and abs(e1 - 0.5 * e2) < 1e-6 and e2 == e9 and abs(e9x2 - 2.0 * e9) < 1e-6 and e2 != 0.0
    t("p2/ramp_and_final_weight_scale_the_term", ramp_and_lam_in_the_term)

    # ---- K1: --epo-lam 0 reproduces arm A / the OFF run EXACTLY (loss bits AND parameter hash); a live EPO moves arm B only
    def k1_and_B_only():
        an0, ns0 = mk(False, epo=False)
        off = run_arms(lambda sx0, sxA, sxB, tg, upd: ns0["train_losses"](sx0, sxA, sxB, tg, None, upd), 6)
        anL, nsL = mk(False, epo_lam=0.0)
        lam0 = run_arms(lambda sx0, sxA, sxB, tg, upd: nsL["train_losses"](sx0, sxA, sxB, tg, ROWS, upd), 6)
        anO, nsO = mk(False)
        on = run_arms(lambda sx0, sxA, sxB, tg, upd: nsO["train_losses"](sx0, sxA, sxB, tg, ROWS, upd), 6)
        return off == lam0 and on[2] == off[2] and on[3] != off[3] and on[1] != off[1]
    t("p2/K1_lam0_equals_arm_A_and_live_EPO_moves_only_B", k1_and_B_only)

    # ---- the REAL training-loop body (main()'s own statements up to .backward()) reaches the gradient
    def loop_stmts():
        for n in ast.walk(C.main):
            if isinstance(n, ast.For) and ast.unparse(n.iter) == "dl":
                out = []
                for s in n.body:
                    out.append(C.seg(s))
                    if ".backward()" in ast.unparse(s):
                        return out
        raise KeyError("no training loop")

    def loop_grads(epo, lam):
        an, ns = mk(False, epo=epo, epo_lam=lam, epo_ramp=0)
        A, B, base, (cxy, sctx, tg), sx0, _a, _b, _t = one_microbatch(6)

        def fwd(_batch):
            with torch.no_grad():
                s0 = base(cxy, sctx)
            return s0, A(cxy, sctx), B(cxy, sctx), tg
        ns["fwd"], ns["upd"] = fwd, 3
        ns["batch"] = [torch.zeros(1)] * 9 + ([ROWS] if epo else [])
        for s in loop_stmts():
            exec(compile(s, SRC_PATH, "exec"), ns)
        return phash_grads(A), phash_grads(B)

    def phash_grads(m):
        h = hashlib.sha256()
        for p in m.parameters():
            h.update(b"none" if p.grad is None else p.grad.numpy().tobytes())
        return h.hexdigest()

    def loop_reaches():
        off_a, off_b = loop_grads(False, 1.0)
        on_a, on_b = loop_grads(True, 1.0)
        z_a, z_b = loop_grads(True, 0.0)
        return on_a == off_a and on_b != off_b and z_a == off_a and z_b == off_b
    t("p2/real_loop_body_flag_to_backward", loop_reaches)

    # ---- main()'s wiring, hop by hop (static)
    def hop_refusals():
        for i, n in enumerate(C.main.body):
            if isinstance(n, ast.Assign) and getattr(n.value, "func", None) is not None and getattr(n.value.func, "id", "") == "epo_refusals":
                nxt = C.main.body[i + 1]
                return isinstance(nxt, ast.If) and any(isinstance(m, ast.Return) and getattr(m.value, "value", None) == 4 for m in ast.walk(nxt))
        return False
    t("p2/hop_main_refuses_on_epo_refusals", hop_refusals)

    def hop_train_loop():
        for n in ast.walk(C.main):
            if isinstance(n, ast.For) and ast.unparse(n.iter) == "dl":
                segs = [C.seg(s) for s in n.body]
                return (ast_eq(segs[0], "batch, rows = split_rows(batch, a.epo)") and ast_eq(segs[1], "_s0, sA, sB, tg = fwd(batch)")
                        and ast_eq(segs[2], "lA, lB, ln, ep_t = train_losses(_s0, sA, sB, tg, rows, upd)")
                        and ast_eq(segs[3], "((lA + lB) / a.accum).backward()"))
        return False
    t("p2/hop_loop_split_rows_fwd_train_losses_backward", hop_train_loop)
    t("p2/hop_dataloader_built_by_make_train_set", lambda: any(isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "dl"
                                                              and "make_train_set(ds, tr_idx, a.epo)" in ast.unparse(n.value) for n in C.main.body))

    def hop_assembly():
        tl = ast.unparse(C.nested["train_losses"])
        et = ast.unparse(C.nested["epo_term"])
        return ("if a.epo:" in tl and "epo_term(" in tl and "lB = lB + ep" in tl and "lA = lA + " not in tl
                and "epo_loss(" in et and "sel_score" in et and "_pdm_aggregate" not in et)
    t("p2/hop_train_losses_calls_epo_term_with_sel_score", hop_assembly)
    t("p2/hop_g11_wired_in_preflight", lambda: "g11_epo_live_and_identity(armB" in C.src and 'gates["G11_epo_live_and_identity"]' in C.src
      and 'os.environ.get("REFE_RL_MUTATE_EPO_DETACH") != "1"' in C.src)
    t("p2/hop_train_index_restricted_to_sets_with_terms", lambda: any(isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "tr_idx"
                                                                     and "key(ds.rows[i]) in terms_train" in ast.unparse(n.value) for n in ast.walk(C.main)))
    t("p2/hop_terms_loaded_for_train_and_heldout", lambda: "load_rl_terms(a.epo_terms_train, op.by, cfg.n_proposals, a.epo_lam_d != 0.0, a.epo_k, a.epo_l)" in C.src
      and "load_rl_terms(a.epo_terms_heldout, ho.by, cfg.n_proposals, a.epo_lam_d != 0.0, a.epo_k, a.epo_l)" in C.src
      and "rl_reward.RewardCfg(lam_d=a.epo_lam_d, w_prog=a.epo_w_prog, guard=True, lam_k=a.epo_lam_k, use_k=a.epo_k," in C.src)
    t("p2/hop_speed_probe_uses_train_losses", lambda: "= train_losses(s0, sA, sB, t_, r_, 0, lam_pg=(1.0 if a.epo else None))" in C.src)

    # ---- G11: the preflight gate, run on the REAL closures
    def g11_run(mutate_env=False, tamper=None, probe_hook=None):
        an, ns = mk(False)
        A, B, base, (cxy, sctx, tg), _s0, _a, _b, _t = one_microbatch(6)
        if tamper:
            tamper(B)

        def fwd(_batch):
            with torch.no_grad():
                s0 = base(cxy, sctx)
            return s0, A(cxy, sctx), B(cxy, sctx), tg
        ns["fwd"] = fwd
        probe = lambda mode: ns["epo_probe"]([torch.zeros(1)] * 9, ROWS, mode)                    # noqa: E731
        if probe_hook:
            probe = probe_hook(probe)
        keep = os.environ.get("REFE_RL_MUTATE_EPO_DETACH")
        if mutate_env:
            os.environ["REFE_RL_MUTATE_EPO_DETACH"] = "1"
        try:
            return mod.g11_epo_live_and_identity(B, probe, kl_tol=0.0)
        finally:
            if keep is None:
                os.environ.pop("REFE_RL_MUTATE_EPO_DETACH", None)
            else:
                os.environ["REFE_RL_MUTATE_EPO_DETACH"] = keep
    t("p2/g11_green_on_a_live_term", lambda: (lambda g: g["ok"] and all(g["live"].values()) and g["controls_exactly_zero"]
                                              and g["kl_step0"] == 0.0 and all(v == 0.0 for v in g["per_module_lam0"].values()))(g11_run()))
    t("p2/g11_red_under_the_detach_regression", lambda: (lambda g: (not g["ok"]) and not any(g["live"].values()) and g["mutate_detach"])(g11_run(mutate_env=True)))

    def dead_module(B):
        for p in B.score_dec.parameters():
            p.requires_grad_(False)
    t("p2/g11_red_when_one_module_gets_no_gradient", lambda: (lambda g: (not g["ok"]) and g["live"]["score_dec"] is False and g["live"]["score_head"])(g11_run(tamper=dead_module)))

    def drift_B(B):
        with torch.no_grad():
            B.score_head.bias.add_(0.5)
    t("p2/g11_red_when_the_kl_anchor_is_not_at_zero_at_step0", lambda: (lambda g: (not g["ok"]) and g["kl_step0"] > 0.0 and not g["kl_ok"])(g11_run(tamper=drift_B)))
    t("p2/g11_red_when_a_control_reads_nonzero", lambda: (not g11_run(probe_hook=lambda pr: (lambda mode: pr("on")))["ok"]))

    # ======================================================================== the REAL main(), end to end on CPU (stubbed trunk / banks / checkpoint loader)
    S: dict = {}

    def sm(name, **kw):
        if name not in S:
            try:
                with tempfile.TemporaryDirectory() as td:
                    S[name] = smoke(C.src, td, tag=f"{name}_{len(S)}_{abs(hash(C.src)) % 10**6}", **kw)
            except Exception as e:                                                  # noqa: BLE001 -- a main() that crashes IS the failure
                S[name] = {"error": f"{type(e).__name__}: {e}"}
        return S[name]

    def trainrow(r):
        return [x for x in r["rows"]["sft.jsonl"] if x.get("event") == "train"][0]

    def dataev(r):
        return [x for x in r["rows"]["sft.jsonl"] if x.get("event") == "data"][0]

    def epo_run_ok(r, native7):
        g11 = r["pass_json"]["gates"]["G11_epo_live_and_identity"]
        tr, dv = trainrow(r), dataev(r)
        return (r["pre"] == 0 and r["run"] == 0 and g11["ok"] and all(g11["live"].values()) and g11["controls_exactly_zero"]
                and dv["epo"]["terms_train"]["matched"] == N_TRAIN and dv["train_sets"] == N_TRAIN and dv["epo"]["native7"] is native7
                and np.isfinite(tr["epo"]) and tr["epo"] != 0.0 and tr["epo_lam_pg"] == 1.0 and "epo_J" in tr and "epo_kl" in tr
                and r["A"] is not None and not scorer_equal(r["A"], r["B"]))
    t("smoke/main_epo_preflight_G11_then_train_6col", lambda: epo_run_ok(sm("s1", epo=True), False))
    t("smoke/main_epo_preflight_G11_then_train_native7_lane_head", lambda: epo_run_ok(sm("s2", epo=True, native7=True), True))

    def off_run_ok(r):
        tr, dv = trainrow(r), dataev(r)
        pre = [x for x in r["rows"]["preflight.jsonl"] if x.get("event") == "preflight"][-1]
        return (r["pre"] == 0 and r["run"] == 0 and "epo" not in dv and not any(k.startswith("epo") for k in tr)
                and "G11_epo_live_and_identity" not in pre["gates"] and pre["verdict"] == "PASS")
    t("smoke/main_off_path_has_no_epo_footprint_and_passes_preflight_at_lam0", lambda: off_run_ok(sm("s3", epo=False)))

    def k1_main_level():
        r1, r3, r4 = sm("s1", epo=True), sm("s3", epo=False), sm("s4", epo=True, epo_lam=0.0)
        return (scorer_equal(r3["A"], r3["B"])                                    # no EPO: B == A bit for bit (BCE-only B)
                and scorer_equal(r4["A"], r4["B"]) and scorer_equal(r4["A"], r3["A"]) and trainrow(r4)["epo"] == 0.0     # --epo --epo-lam 0 == arm A
                and scorer_equal(r1["A"], r3["A"]) and not scorer_equal(r1["B"], r3["B"]))                              # a live EPO moves B and never A
    t("smoke/main_K1_lam0_equals_arm_A_and_EPO_never_touches_A", k1_main_level)

    def refused_run():
        r = sm("s5", epo=True, terms=None, preflight_first=False)
        return r["run"] == 4 and "--epo-terms-train" in r["out_run"] and not r["out_exists"]
    t("smoke/main_refuses_epo_without_terms_before_touching_disk", refused_run)

    def lam_d_zero_runs():
        r = sm("s6", epo=True, terms=False)
        return (r["pre"] == 0 and r["run"] == 0 and dataev(r)["epo"]["terms_train"] is None and dataev(r)["epo"]["lam_d"] == 0.0
                and dataev(r)["train_sets"] == N_TRAIN and trainrow(r)["epo"] != 0.0)
    t("smoke/main_epo_runs_without_side_files_when_lam_d_zero_is_named", lam_d_zero_runs)

#@@MORE_CHECKS@@
    return bad


# ------------------------------------------------------------------------------------------------ what the OFF-path comparison runs
def REAL_OFF_NAMES(C: Ctx):
    return ["lane_bce", "sel_score", "losses", "losses6", "epo_term", "train_losses"]


def REAL_STEP(rns, an):
    return lambda sx0, sxA, sxB, tg, upd: rns["train_losses"](sx0, sxA, sxB, tg, None, upd)


# ------------------------------------------------------------------------------------------------ mutations: (name, group, snippet, replacement)
MUTATIONS = [
    ("cfg_without_checkpoint_sizing", "p1", "cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(a.backbone), a.ckpt)",
     "cfg = REFeConfig.for_backbone(a.backbone)"),
    ("lane_skip_removed", "p1", "    if int(old.out_features) == 7:\n        return old\n", ""),
    ("lane_row6_bias_changed", "p1", "new.bias[6] = 1.5 ", "new.bias[6] = 1.4 "),
    ("lam_default_changed", "p1", 'ap.add_argument("--lam", type=float, default=0.3', 'ap.add_argument("--lam", type=float, default=0.31'),
    ("expreward_perturbed", "p1", "return bceA, bceB + lam * er, er", "return bceA, bceB + lam * er * 1.0001, er"),
    ("lane_weight_dropped", "p1", "la, lb = a.lane_weight * lane_bce(sxA, tg), a.lane_weight * lane_bce(sxB, tg)",
     "la, lb = lane_bce(sxA, tg), lane_bce(sxB, tg)"),
    ("extend_not_called", "p1", "        extend_lane_head(model, dev)\n", "        pass\n"),
    ("native7_refusal_removed", "p1", "    if native7 and not a.lane_head:\n", "    if False and native7 and not a.lane_head:\n"),
    # ---- priority 2: the EPO loss
    ("epo_line_deleted", "p2", "            lB = lB + ep\n", "            pass\n"),
    ("epo_default_flipped_on", "p2", 'ap.add_argument("--epo", action="store_true",', 'ap.add_argument("--epo", action="store_true", default=True,'),
    ("policy_logits_detached", "p2", "logpi = epo_policy(sxB, rule_fn, T)", "logpi = epo_policy(sxB.detach(), rule_fn, T)"),
    ("kl_anchor_dropped", "p2", "return lam_pg * (-J + beta * kl).mean()", "return lam_pg * (-J).mean()"),
    ("pdm_aggregate_instead_of_sel_score", "p2", "torch.as_tensor(R, dtype=torch.float32, device=sxB.device), sel_score, a.epo_temp",
     "torch.as_tensor(R, dtype=torch.float32, device=sxB.device), T._pdm_aggregate, a.epo_temp"),
    ("ramp_ignored", "p2", "return 1.0 if ramp <= 0 else min(1.0, upd / ramp)", "return 1.0"),
    ("objective_sign_flipped", "p2", "return lam_pg * (-J + beta * kl).mean()", "return lam_pg * (J + beta * kl).mean()"),
    ("reward_not_detached", "p2", "J = (logpi.exp() * R.detach()).sum(1)", "J = (logpi.exp() * R).sum(1)"),
    ("d_term_ignored", "p2", 'd = np.stack([r["d"] for r in recs])', "d = np.zeros((len(recs), tg6.shape[1]))"),
    ("row_key_wrong", "p2", "recs = [terms_train.get(key(ds.rows[int(i)])) for i in rows]", "recs = [terms_train.get(key(ds.rows[0])) for i in rows]"),
    ("epo_leaks_into_arm_A", "p2", "            lB = lB + ep\n", "            lA = lA + ep\n            lB = lB + ep\n"),
    ("terms_refusal_removed", "p2", "if not a.epo_terms_train and a.epo_lam_d != 0.0:", "if False:"),
    ("epo_lam_ignored", "p2", "lam_pg = a.epo_lam * epo_ramp(upd, a.epo_ramp)", "lam_pg = 1.0 * epo_ramp(upd, a.epo_ramp)"),
    ("loop_bypasses_train_losses", "p2", "            lA, lB, ln, ep_t = train_losses(_s0, sA, sB, tg, rows, upd)\n",
     "            lA, lB, ln = losses(sA, sB, tg, a.lam)\n            ep_t = torch.zeros(())\n"),
    ("dataloader_without_rows", "p2", "dl = DataLoader(make_train_set(ds, tr_idx, a.epo),", "dl = DataLoader(Subset(ds, tr_idx),"),
    ("getitems_override_removed", "p2", "    def __getitems__(self, indices):\n        return [self[i] for i in indices]\n", ""),
    ("g11_never_backwards", "p2", "        if term.requires_grad:\n            term.backward()\n", "        if False:\n            term.backward()\n"),
    ("detach_regression_switch_removed", "p2", 'if os.environ.get("REFE_RL_MUTATE_EPO_DETACH") == "1":', "if False:"),
    ("named_levers_without_epo_allowed", "p2", "        if named:\n", "        if False:\n"),
    ("split_rows_wrong_slot", "p2", "return batch[:9], batch[9]", "return batch[:9], batch[0]"),
    ("stale_terms_accepted", "p2", "if ck != int(e[0]):", "if False:"),
    ("temperature_ignored", "p2", "torch.log(rule_fn(torch.sigmoid(sx)) + EPO_EPS) / T", "torch.log(rule_fn(torch.sigmoid(sx)) + EPO_EPS) / 1.0"),
    ("kl_direction_reversed", "p2", "kl = (logpi0.exp() * (logpi0 - logpi)).sum(1)", "kl = (logpi.exp() * (logpi - logpi0)).sum(1)"),
    ("loop_rows_not_split", "p2", "            batch, rows = split_rows(batch, a.epo)\n", "            rows = None\n"),
    ("g11_kl_check_bypassed", "p2", "kl_ok = kl0 is not None and math.isfinite(kl0) and abs(kl0) <= kl_tol", "kl_ok = True"),
    ("g11_controls_bypassed", "p2", 'zero_ok = all(v == 0.0 for r in ("lam0", "off") for v in reads[r].values()) and vals["lam0"] == 0.0 and vals["off"] == 0.0',
     "zero_ok = True"),
    ("g11_liveness_bypassed", "p2", 'reads["on"][m] > 0.0', 'reads["on"][m] >= 0.0'),
    ("train_index_not_restricted", "p2", "            tr_idx = [i for i in tr_idx if key(ds.rows[i]) in terms_train]\n", "            pass\n"),
    ("g4_liveness_demanded_at_lam0", "p3", "(float(bB) != float(bA) or a.lam == 0.0)", "float(bB) != float(bA)"),
    ("g11_not_in_gates", "p3", '            gates["G11_epo_live_and_identity"] = g11\n', "            pass\n"),
    ("terms_not_loaded", "p3", "            terms_train, st_tr = load_rl_terms(a.epo_terms_train, op.by, cfg.n_proposals, a.epo_lam_d != 0.0, a.epo_k, a.epo_l)\n",
     "            terms_train, st_tr = None, None\n"),
    ("epo_log_fields_dropped", "p3", '                    row.update({"epo": epo_acc["epo"] / ne,', '                    row.update({"epo_x": epo_acc["epo"] / ne,'),
    ("refusal_after_makedirs", "p3", "    bad_cli = epo_refusals(a, ap)", "    os.makedirs(a.out, exist_ok=True)\n    bad_cli = epo_refusals(a, ap)"),
]


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    n = 0
    real = suite(Ctx(src, "sft_real"))
    assert not real, f"the unmutated scorer_finetune.py FAILS its own checks: {real}"
    n += 1
    print(f"{'mutation':38s} {'grp':4s} caught-by (failing checks)")
    for i, (name, grp, old, new) in enumerate(MUTATIONS):
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        msrc = src.replace(old, new)
        try:
            ast.parse(msrc)
        except SyntaxError as e:
            raise AssertionError(f"mutation {name} does not compile: {e}")
        try:
            failing = suite(Ctx(msrc, f"sft_mut{i}"))
        except Exception as e:                                                   # noqa: BLE001 -- a mutated caller that breaks the harness is RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:70]}"]
        assert failing, f"mutation {name} ({grp}) stayed GREEN: the suite cannot see it"
        print(f"{name:38s} {grp:4s} {', '.join(failing[:3])}")
        n += 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
