"""tok_probe_gr.py -- made from tok_probe_g.py by make_tok_probe_gr.py (card PREREG_V8F_PROOF_ADD43, FINEGRID-R; definitions: FINEGRID_R_DEFS.md): adds the r1 batch log, the lever
--sampler-chunk-like and the p5 form for permuted-label arms; the default path is the one of tok_probe_g.py (control g1).
TOKPROBE with a grid lever, the probe tool of the frozen card PREREG_V8F_PROOF_ADD41 (FINEGRID; definitions: FINEGRID_DEFS.md).

A copy of tok_probe.py (card ADD27) with ONE added lever ``--grid pool4x10|pool8x20`` (default pool4x10) on ``train`` and ``extract``: 40 or 160 cached frozen camera tokens of width 1,024 per
frame, trained for the lead labels alone, read with the LeadSupHead of card ADD24. No training of the model. At the default grid the path is the one of tok_probe.py (control g1).
Tier T0 (a diagnostic on logged frames), never driving performance. STRATEGIC: NOT COMPUTED (a token probe reads no strategic quantity; n = 0). Runs on the pod GPU (train, extract); the dev box tests it on CPU with synthetic data.

  python tok_probe_g.py train   --grid pool8x20 --tree /workspace/v8proof/tree_leadsup --out DIR [--steps N] [--expect-steps 19164] [--shuffle-closing-labels] [--arms F16_s0,...]
  python tok_probe_g.py extract --grid pool8x20 --tree TREE --run-dir DIR --arm F16_s0 --cache /workspace/v8proof/data/cache_pod --states-dir DIR
  python tok_probe_g.py g2      --tree TREE --cache CACHE --out G2_GRIDS.json            (control g2: the 2 by 2 mean of the stored 8 by 20 tokens against the stored 4 by 10 tokens)
  python tok_probe_g.py check-log --run-dir DIR                                          (control p5)

Arm names: TK<W>[SH]_s<seed> belong to pool4x10, F<W>[SH]_s<seed> to pool8x20; the other pairing is refused. The probe's forward takes the token array ONLY (control p4).
Control g5: a batch that is not [batch, frames, n_tok, 1024] for the declared grid is refused, and the checkpoint records the token count actually seen.
Nothing here reads git, the pod, box 2 or the model's checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import shlex
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TREE = "/workspace/v8proof/tree_leadsup"
N_TOK, D_FEAT, HIDDEN, HIST = 40, 1024, 128, 5
GRIDS = {"pool4x10": 40, "pool8x20": 160}                   # grid key of the cache = trainer flag value -> tokens per frame
GRID_DEFAULT = "pool4x10"
ARMS_DEFAULT = ("TK16_s0", "TK16_s1", "TK64_s0", "TK64_s1")
ARMS_FINE = ("F16_s0", "F16_s1", "F64_s0", "F64_s1")
ARMS_BY_GRID = {"pool4x10": ARMS_DEFAULT, "pool8x20": ARMS_FINE}
SHUFFLE_ARM_BY_GRID = {"pool4x10": "TK16_s0", "pool8x20": "F16_s0"}
HOST_GB_MAX = 8.0                                           # the process stays under about 8 GB of host memory (estimate, refused above)
GAP_SCALE_M, CLOS_SCALE_MS = 50.0, 10.0                     # = M.LEAD_SUP_GAP_SCALE_M / LEAD_SUP_CLOS_SCALE_MS (asserted in use_tree)
LOG_EVERY = 50
SHUFFLE_RNG_SEED = 12345
R1_BATCHES = 200                                            # control r1: the window indices of the first 200 batches are hashed
TERMS = ("lead_sup", "lead_sup_bce", "lead_sup_gap", "lead_sup_clos")
HEAD_KEYS = {"frame_w": "head.frame.weight", "frame_b": "head.frame.bias", "mlp0_w": "head.mlp.0.weight",
             "mlp0_b": "head.mlp.0.bias", "mlp2_w": "head.mlp.2.weight", "mlp2_b": "head.mlp.2.bias"}
# the argv record of arm LSD_s0 (pod paths); --out and --device are replaced by this tool's own (TOKPROBE_DEFS A2)
DEFAULT_TRAINER_ARGV = (
    "--arm M2b --seed 0 --grid pool4x10 --wm-state grid --wm-cell-dim 16 --lead-sup on --w-lead-sup 1.0 --lead-sup-detach on "
    "--out X --cache-dir /workspace/v8proof/data/cache_pod --v9-train /workspace/v8proof/data/v9/v9_labels_train.npz "
    "--v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train /workspace/v8proof/data/v9/lead_train4572.npz "
    "--speed-train /workspace/v8proof/data/v9/refcv6_speed_max_v8_train.jsonl --join-train /workspace/v8proof/data/inputs/join_train.npz "
    "--p10 /workspace/v8proof/data/inputs/p10_prior.pt --kin-mode yaw --tracks /workspace/v8proof/data/inputs/native_tracks.pt "
    "--exclude 540:600 --batch 64 --steps-mult 4 --device cuda")

_TREE = {}


def use_tree(tree: str):
    """Import the tree's stack (model, data, trainer module) and check that tanitad comes from THAT tree. Returns the namespace {M, D, TR, tree}."""
    tree = os.path.abspath(tree)
    sp = os.path.join(tree, "stack")
    for p in (os.path.join(sp, "scripts"), sp):
        if p not in sys.path:
            sys.path.insert(0, p)
    from tanitad.data import v8f_mini_data as D
    from tanitad.models import v8f_mini as M
    import v8f_mini_train as TR
    tf = os.path.abspath(sys.modules["tanitad"].__file__).replace(os.sep, "/")
    if not tf.startswith(tree.replace(os.sep, "/")):
        raise SystemExit("REFUSED: tanitad is imported from %s, not from the tree %s" % (tf, tree))
    assert M.LEAD_SUP_GAP_SCALE_M == GAP_SCALE_M and M.LEAD_SUP_CLOS_SCALE_MS == CLOS_SCALE_MS
    _TREE.update({"M": M, "D": D, "TR": TR, "tree": tree})
    return _TREE


def tree():
    if not _TREE:
        raise RuntimeError("call use_tree(path) first")
    return _TREE


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def parse_arm(name):
    m = re.match(r"^(TK|F)(\d+)(SH)?_s(\d+)$", name)
    if not m:
        raise SystemExit("REFUSED: arm name %r is not TK<W>[SH]_s<seed> or F<W>[SH]_s<seed>" % name)
    grid = "pool4x10" if m.group(1) == "TK" else "pool8x20"
    return {"W": int(m.group(2)), "shuffle": bool(m.group(3)), "seed": int(m.group(4)), "family": m.group(1), "grid": grid}


def check_arm_grid(name, grid):
    """Refuse an F arm under pool4x10 and a TK arm under pool8x20."""
    if grid not in GRIDS:
        raise SystemExit("REFUSED: grid %r is not one of %s" % (grid, sorted(GRIDS)))
    ag = parse_arm(name)["grid"]
    if ag != grid:
        raise SystemExit("REFUSED: arm %s belongs to grid %s, the run declares %s" % (name, ag, grid))


def check_batch_shape(feats, n_tok, batch=None, hist=HIST):
    """Control g5: the batch the probe is trained on is [batch, frames, n_tok, 1024] for the declared grid (the sampler's last batch of a chunk can hold fewer rows than ``batch``, never more)."""
    s = tuple(feats.shape)
    if len(s) != 4 or s[2] != n_tok or s[3] != D_FEAT or s[1] < hist or (batch is not None and not 1 <= s[0] <= batch):
        raise SystemExit("REFUSED: a batch of tokens has shape %s, expected [%s, frames >= %d, %d, %d]" % (list(s), batch if batch is not None else "batch", hist, n_tok, D_FEAT))
    return s[2]


# ============================================================================================ the probe
class TokenProbe(nn.Module):
    """Per-token network 1,024 -> 128 (GELU) -> W with the same weights for every token, the 40 outputs concatenated to 40 * W per frame; then the LeadSupHead of card ADD24 on 5 frames.
    ``forward(tokens)`` takes the token array and NOTHING else (control p4). ``ctrl`` is a control parameter outside the graph (control p1): its gradient stays None, read as exactly 0.0."""

    def __init__(self, W: int, hist: int = HIST, n_tok: int = N_TOK, d_feat: int = D_FEAT, hidden: int = HIDDEN, input_norm: str = "ln"):
        super().__init__()
        if input_norm not in ("ln", "none"):
            raise ValueError("input_norm must be ln or none")
        self.W, self.hist, self.n_tok, self.d_feat, self.hidden, self.input_norm = int(W), int(hist), int(n_tok), int(d_feat), int(hidden), input_norm
        self.tok = nn.Sequential(nn.Linear(d_feat, hidden), nn.GELU(), nn.Linear(hidden, W))
        self.head = tree()["M"].LeadSupHead(n_tok * W, hist)
        self.ctrl = nn.Parameter(torch.zeros(1))

    def project(self, tokens):
        x = tokens.float()
        if self.input_norm == "ln":
            x = F.layer_norm(x, (x.shape[-1],))
        return self.tok(x).flatten(-2)                                          # [..., n_tok * W]

    def forward(self, tokens):
        return self.head(self.project(tokens))                                  # [B, F - hist + 1, 3]


def build_probe(arm: str, input_norm: str = "ln", grid: str | None = None):
    """The arm's probe; ``torch.manual_seed(seed)`` is called immediately before the construction, so the initial weights depend on seed and shape only.
    The grid is the arm family's (TK: pool4x10, F: pool8x20); a ``grid`` that disagrees with the arm name is refused."""
    p = parse_arm(arm)
    if grid is not None:
        check_arm_grid(arm, grid)
    torch.manual_seed(p["seed"])
    return TokenProbe(p["W"], n_tok=GRIDS[p["grid"]], input_norm=input_norm)


def make_opt(probe, lr, wd):
    """The trainer's two groups (TR:2843-2848): weight decay on ndim >= 2, none on the rest. ``ctrl`` is not given to the optimiser."""
    named = [(n, p) for n, p in probe.named_parameters() if p.requires_grad and n != "ctrl"]
    decay = [p for n, p in named if p.ndim >= 2]
    nodecay = [p for n, p in named if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}], lr=lr)


def grad_sums(probe):
    def s(ps):
        return float(sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None))
    return {"g_tok": s(probe.tok.parameters()), "g_head": s(probe.head.parameters()),
            "g_ctrl": 0.0 if probe.ctrl.grad is None else float(probe.ctrl.grad.abs().sum())}


def step_probe(probe, opt, batch, lr, clip_grad, w_loss, hist=HIST, want_g=False):
    """One training step of one probe on one batch: forward on the tokens, the lead term of the model's own function, backward, clip, step."""
    M = tree()["M"]
    for g in opt.param_groups:
        g["lr"] = lr
    out = probe(batch["feats"])
    T = M.lead_sup_terms(out, batch, hist)
    loss = w_loss * T["lead_sup"]  # LOSS-LINE
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gs = grad_sums(probe) if want_g else None
    gn = torch.nn.utils.clip_grad_norm_([p for n, p in probe.named_parameters() if p.requires_grad and n != "ctrl"], clip_grad)
    opt.step()
    return T, gn, gs


def to_device(b, dev):
    return {k: b[k].to(dev) for k in ("feats", "frame_ok", "lead_pres", "lead_gapm", "lead_clos")}


def run_training(stream, arms, total, lr_at, *, lr, wd, clip_grad, w_loss, device, out_dir, hist=HIST, log_every=LOG_EVERY, header=None, input_norm="ln", argv_sha="", grid=GRID_DEFAULT,
                 expect_batch=None):
    """Train the arms on the SAME batches of ``stream`` (an iterator of collated batches) for ``total`` steps in one process. Per arm: ``<out_dir>/<ARM>/train_log.jsonl`` and ``ckpt_last.pt``.
    ``grid`` is the declared grid: every arm must belong to it and every batch must be [batch, frames, n_tok, 1024] for it (control g5); the checkpoint records the token count seen."""
    dev = torch.device(device)
    probes = {a: build_probe(a, input_norm, grid).to(dev).train() for a in arms}
    n_tok = GRIDS[grid]
    n_tok_seen = None
    opts = {a: make_opt(p, lr, wd) for a, p in probes.items()}
    logs = {}
    for a, p in probes.items():
        os.makedirs(os.path.join(out_dir, a), exist_ok=True)
        logs[a] = open(os.path.join(out_dir, a, "train_log.jsonl"), "w", encoding="utf-8", newline="\n")
        logs[a].write(json.dumps({"header": True, "arm": a, "W": p.W, "seed": parse_arm(a)["seed"], "grid": grid, "n_tok": p.n_tok, "input_norm": input_norm, "total": total, "argv_sha": argv_sha,
                                  "n_params": {"tok": sum(q.numel() for q in p.tok.parameters()), "head": sum(q.numel() for q in p.head.parameters()), "ctrl": 1}, **(header or {})}) + "\n")
    acc = {a: {"sum": torch.zeros(4, device=dev), "n": 0, "ngap_min": torch.full((), float("inf"), device=dev)} for a in probes}
    t0 = time.time()
    step = 0
    for b in stream:
        if step >= total:
            break
        n_tok_seen = check_batch_shape(b["feats"], n_tok, expect_batch, hist)  # TOKCOUNT-REFUSAL
        bd = to_device(b, dev)
        log_now = (step + 1) % log_every == 0 or step == 0 or step + 1 == total
        for a, p in probes.items():
            T, gn, gs = step_probe(p, opts[a], bd, lr_at(step), clip_grad, w_loss, hist, want_g=log_now)
            ac = acc[a]
            ac["sum"] += torch.stack([T[k].detach() for k in TERMS])
            ac["n"] += 1
            ac["ngap_min"] = torch.minimum(ac["ngap_min"], T["lead_sup_n_gap"].detach())
            if log_now:
                if not math.isfinite(float(T["lead_sup"].detach())):
                    raise SystemExit("ZZNONFINITE-LOSSZZ arm %s step %d" % (a, step))
                mean = (ac["sum"] / max(ac["n"], 1)).tolist()
                rec = {"step": step + 1, "first": step == 0, "lr": lr_at(step), "grad_norm": float(gn), **{k: float(T[k].detach()) for k in TERMS},
                       **{k + "_mean": m for k, m in zip(TERMS, mean)}, "n_pres": float(T["lead_sup_n_pres"]), "n_gap": float(T["lead_sup_n_gap"]),
                       "n_clos": float(T["lead_sup_n_clos"]), "n_gap_min_interval": float(ac["ngap_min"]), **gs, "sec": time.time() - t0}
                logs[a].write(json.dumps(rec) + "\n")
                logs[a].flush()
                ac["sum"].zero_()
                ac["n"] = 0
                ac["ngap_min"] = torch.full((), float("inf"), device=dev)
        step += 1
        if log_now and step % (log_every * 10) == 0:
            print("[tok_probe %d/%d] %.0fs" % (step, total, time.time() - t0), flush=True)
    if step < total:
        raise SystemExit("REFUSED: the batch stream ended at step %d of %d" % (step, total))
    for a, p in probes.items():
        logs[a].close()
        save_probe(p, a, os.path.join(out_dir, a, "ckpt_last.pt"), step, argv_sha, grid=grid, n_tok_seen=n_tok_seen)
    return probes


def save_probe(p, arm, path, step, argv_sha="", grid=GRID_DEFAULT, n_tok_seen=None):
    blob = {"kind": "tokprobe", "arm": arm, "W": p.W, "seed": parse_arm(arm)["seed"], "grid": grid, "n_tok": p.n_tok, "d_feat": p.d_feat, "hidden": p.hidden, "hist": p.hist,
            "input_norm": p.input_norm, "step": int(step), "argv_sha": argv_sha, "state_dict": {k: v.detach().cpu() for k, v in p.state_dict().items()}}
    blob["n_tok_seen"] = None if n_tok_seen is None else int(n_tok_seen)  # CKPT-NTOK
    torch.save(blob, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_probe(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != "tokprobe":
        raise SystemExit("REFUSED: %s is not a tokprobe checkpoint" % path)
    grid = ck.get("grid", GRID_DEFAULT)                                       # a checkpoint of tok_probe.py has no grid key: it is a 4 by 10 one
    if GRIDS.get(grid) != ck["n_tok"]:
        raise SystemExit("REFUSED: %s records grid %s and n_tok %s" % (path, grid, ck["n_tok"]))
    p = TokenProbe(ck["W"], hist=ck["hist"], n_tok=ck["n_tok"], d_feat=ck["d_feat"], hidden=ck["hidden"], input_norm=ck["input_norm"])
    p.load_state_dict(ck["state_dict"])
    return p.to(device).eval(), ck


# ============================================================================================ train
def check_log(arm_dir_or_file, total=None, window=1000):
    """Control p5 on one arm's train log: the mean of the interval means of the lead term over the last ``window`` (1,000) steps below that of the first ``window`` steps, and n_gap above 0 at every record."""
    path = arm_dir_or_file if arm_dir_or_file.endswith(".jsonl") else os.path.join(arm_dir_or_file, "train_log.jsonl")
    recs = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    hdr, recs = recs[0], recs[1:]
    total = int(total or hdr["total"])
    body = [r for r in recs if not r["first"]]
    first = [r["lead_sup_mean"] for r in body if r["step"] <= window]
    last = [r["lead_sup_mean"] for r in body if r["step"] > total - window]
    ng = min(min(r["n_gap"] for r in recs), min(r["n_gap_min_interval"] for r in recs))
    out = {"arm": hdr["arm"], "total": total, "n_records": len(recs), "first_1000_mean": float(np.mean(first)) if first else None, "last_1000_mean": float(np.mean(last)) if last else None,
           "n_first": len(first), "n_last": len(last), "min_n_gap": ng, "last_step": recs[-1]["step"] if recs else None}
    out["pass_lead_term_falls"] = bool(first and last and out["last_1000_mean"] < out["first_1000_mean"])
    shuffled = bool(re.match(r"^(TK|F)\d+SH_s\d+$", str(hdr["arm"])))
    if all(k in r for r in body for k in ("lead_sup_bce_mean", "lead_sup_gap_mean", "lead_sup_clos_mean")):  # P5-PARTS: the parts of the lead term are in the log of tok_probe_g.py
        def part(keys, lo_hi):
            sel = [sum(r[k] for k in keys) for r in body if lo_hi(r["step"])]
            return float(np.mean(sel)) if sel else None
        fi, la = part(("lead_sup_bce_mean", "lead_sup_gap_mean"), lambda s: s <= window), part(("lead_sup_bce_mean", "lead_sup_gap_mean"), lambda s: s > total - window)
        out["intact_parts_first_mean"], out["intact_parts_last_mean"] = fi, la
        out["permuted_part_clos_first_mean"], out["permuted_part_clos_last_mean"] = part(("lead_sup_clos_mean",), lambda s: s <= window), part(("lead_sup_clos_mean",), lambda s: s > total - window)
        out["pass_intact_parts_fall"] = bool(fi is not None and la is not None and la < fi)
    else:
        out["pass_intact_parts_fall"] = None
    out["shuffled_closing_arm"] = shuffled
    out["pass_gap_labels_every_step"] = bool(ng > 0)
    out["pass_complete"] = bool(recs and recs[-1]["step"] == total)
    falls = out["pass_intact_parts_fall"] if shuffled else out["pass_lead_term_falls"]          # P5-FORM: an arm with permuted closing labels is read on its intact parts
    out["pass"] = bool(falls and out["pass_gap_labels_every_step"] and out["pass_complete"])
    return out


def build_training_data(a, cfg):
    """The trainer's data path: build_data (TR:2691) and the dataset of TR:2819-2821. Returns (ds, S, total, info)."""
    T = tree()
    D, TR = T["D"], T["TR"]
    store, J, wins, clips, info = TR.build_data(a, cfg)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start, lead_sup=cfg.lead_sup == "on")
    if not ds.lead_sup or not ds.lead_sup_labels:
        raise SystemExit("REFUSED: the built dataset carries no lead_sup labels")
    ll = ds.lead_sup_labels
    if min(ll["n_pres"], ll["n_gap"], ll["n_clos"]) <= 0:
        raise SystemExit("REFUSED: lead_sup labels: presence %s / gap %s / closing %s labelled rows" % (ll["n_pres"], ll["n_gap"], ll["n_clos"]))
    S = math.ceil(info["n_windows_all_clips"] / a.batch)
    total = a.steps or S * a.steps_mult
    n_tok = GRIDS[a.grid]
    nmax = int(J["n"].max())
    store_b, prefetch_b, base_b = info["lru"] * nmax * n_tok * D_FEAT * 2, 5 * a.batch * 16 * n_tok * D_FEAT * 2, int(1.5e9)
    info["host_est"] = {"n_tok": n_tok, "lru_clips": int(info["lru"]), "clip_frames_max": nmax, "clip_store_gb": store_b / 2 ** 30, "prefetch_gb": prefetch_b / 2 ** 30,
                        "other_gb": base_b / 2 ** 30, "total_gb": (store_b + prefetch_b + base_b) / 2 ** 30, "basis": "ESTIMATED: LRU clips x frames x tokens x 1024 x 2 B + 5 batches of 16 frames + 1.5 GB"}
    if info["host_est"]["total_gb"] > HOST_GB_MAX:
        raise SystemExit("REFUSED: estimated host memory %.1f GB is over %.1f GB (lru %d clips, %d tokens)" % (info["host_est"]["total_gb"], HOST_GB_MAX, info["lru"], n_tok))
    return ds, wins, S, total, info


def shuffle_closing(ds, seed=SHUFFLE_RNG_SEED):
    """Control p3: permute the finite closing-speed labels among the rows that have one (the mask, hence the label counts, stay)."""
    LS = ds.lead_sup_labels
    c = np.array(LS["clos"], copy=True)
    idx = np.flatnonzero(np.isfinite(c))
    c[idx] = c[idx][np.random.default_rng(seed).permutation(len(idx))]
    LS["clos"] = c
    return {"n_shuffled": int(len(idx)), "rng_seed": seed}


def trainer_lru(nmax, n_tok):
    """The trainer's clip-store size when --lru is 0 (v8f_mini_train.py:2736): max(8, 1.5e9 // (longest clip * tokens * 1024 * 2))."""
    return max(8, int(1.5e9 // (int(nmax) * int(n_tok) * 1024 * 2)))


def sampler_chunk(lru):
    """The trainer's sampler chunk (v8f_mini_train.py:2851): clips per pool, at most 48, at least 4, two below the clip-store size."""
    return max(4, min(48, int(lru) - 2))


def batch_hash(batch):
    """Hash of the window indices of one batch: the list of (clip, t, age_t, age_s) in order."""
    return hashlib.sha256(json.dumps([[int(x) for x in w] for w in batch]).encode()).hexdigest()[:16]


class R1Log:
    """Control r1 (card ADD43 section 4): records a hash per batch, in order, for the first ``n`` batches the sampler yields (logging only; the batches pass through unchanged)
    and writes <out>/r1_batches.json. ``idx`` hashes (clip, t) only, ``full`` hashes the whole item including the two age draws."""

    def __init__(self, path, n, meta=None):
        self.path, self.n, self.meta, self.idx, self.full, self.done = path, int(n), dict(meta or {}), [], [], False

    def watch(self, batches):
        for b in batches:
            if len(self.idx) < self.n:
                self.idx.append(batch_hash([w[:2] for w in b]))
                self.full.append(batch_hash(b))
                if len(self.idx) == self.n:
                    self.flush()
            yield b

    def flush(self):
        if self.done:
            return
        self.done = True
        rec = {"n_batches": len(self.idx), "idx_hashes": self.idx, "full_hashes": self.full, "sequence_sha256": hashlib.sha256("".join(self.full).encode()).hexdigest(), **self.meta}
        with open(self.path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rec, fh, indent=1)


def r1_compare(path_a, path_b):
    """Control r1: the two batch logs hold the same sequence of window indices (both the (clip, t) hashes and the full-item hashes), over at least one batch."""
    a, b = json.load(open(path_a, encoding="utf-8")), json.load(open(path_b, encoding="utf-8"))
    same_idx = a["idx_hashes"] == b["idx_hashes"]
    same_full = a["full_hashes"] == b["full_hashes"]
    mism = next((i for i, (x, y) in enumerate(zip(a["idx_hashes"], b["idx_hashes"])) if x != y), None)
    return {"pass": bool(same_idx and same_full and len(a["idx_hashes"]) > 0 and a["n_batches"] == b["n_batches"]), "n_a": a["n_batches"], "n_b": b["n_batches"], "idx_equal": same_idx, "full_equal": same_full,
            "first_mismatch_batch": mism, "sampler_chunk_a": a.get("sampler_chunk"), "sampler_chunk_b": b.get("sampler_chunk"), "sequence_sha256_a": a.get("sequence_sha256"), "sequence_sha256_b": b.get("sequence_sha256"),
            "files": [path_a, path_b]}


def cmd_r1(args):
    res = r1_compare(os.path.join(args.a, "r1_batches.json"), os.path.join(args.b, "r1_batches.json"))
    res["card"] = "PREREG_V8F_PROOF_ADD43 control r1"
    jdump(res, args.out)
    print("r1: pass %s (%d batches, chunks %s / %s) -> %s" % (res["pass"], res["n_a"], res["sampler_chunk_a"], res["sampler_chunk_b"], args.out))
    return 0 if res["pass"] else 2


def trainer_argv(text, out, grid=GRID_DEFAULT):
    """The argv handed to the trainer's own data builder: ``--out``, ``--device`` and ``--grid`` of the given text are replaced by this tool's own (the grid is the declared one)."""
    toks = shlex.split(text)
    res, i = [], 0
    while i < len(toks):
        if toks[i] in ("--out", "--device", "--grid"):
            i += 2
            continue
        res.append(toks[i])
        i += 1
    return res + ["--grid", grid, "--out", out, "--device", "cpu"]  # GRID-ARGV


def cmd_train(args):
    T = use_tree(args.tree)
    D, TR = T["D"], T["TR"]
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.grid not in GRIDS:
        raise SystemExit("REFUSED: grid %r is not one of %s" % (args.grid, sorted(GRIDS)))
    names = [n for n in (args.arms or (SHUFFLE_ARM_BY_GRID[args.grid] if args.shuffle_closing_labels else ",".join(ARMS_BY_GRID[args.grid]))).split(",") if n]
    if args.shuffle_closing_labels:
        names = [n if "SH_" in n else n.replace("_s", "SH_s", 1) for n in names]
    for n in names:
        check_arm_grid(n, args.grid)                                              # an F arm under pool4x10 or a TK arm under pool8x20 is refused
    os.makedirs(args.out, exist_ok=True)
    argv = trainer_argv(args.trainer_argv, args.out, args.grid)
    a = TR.build_parser().parse_args(argv)
    if args.steps:
        a.steps = args.steps
    cfg = TR.make_config(a)
    if a.grid != args.grid or cfg.n_tok != GRIDS[args.grid]:                      # declared against built (control g5)
        raise SystemExit("REFUSED: the run declares grid %s (%d tokens) but the trainer builds grid %s with n_tok %s" % (args.grid, GRIDS[args.grid], a.grid, cfg.n_tok))
    if cfg.lead_sup != "on" or cfg.lead_hist != HIST or not cfg.w_lead_sup > 0:
        raise SystemExit("REFUSED: the argv builds lead_sup %r / lead_hist %r / w_lead_sup %r" % (cfg.lead_sup, cfg.lead_hist, cfg.w_lead_sup))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    ds, wins, S, total, info = build_training_data(a, cfg)
    if args.expect_steps and total != args.expect_steps:
        raise SystemExit("REFUSED: the computed step count is %d, expected %d" % (total, args.expect_steps))
    shuf = shuffle_closing(ds) if args.shuffle_closing_labels else None
    argv_sha = hashlib.sha256(json.dumps(vars(args), sort_keys=True, default=str).encode()).hexdigest()[:16]
    like = args.sampler_chunk_like
    if like and like not in GRIDS:
        raise SystemExit("REFUSED: --sampler-chunk-like %r is not one of %s" % (like, sorted(GRIDS)))
    lru_like = (a.lru or trainer_lru(info["host_est"]["clip_frames_max"], GRIDS[like])) if like else info["lru"]
    chunk = sampler_chunk(lru_like)                                            # the default (no flag) is the original expression max(4, min(48, lru - 2))
    sampler = D.ChunkedSampler(wins, a.batch, seed=a.seed, chunk=chunk, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    r1 = R1Log(os.path.join(args.out, "r1_batches.json"), min(R1_BATCHES, total), {"grid": args.grid, "seed": a.seed, "batch": a.batch, "sampler_chunk": chunk, "sampler_chunk_like": like or None,
                                                                              "lru_trainer": info["lru"], "lru_used_for_chunk": lru_like, "total_steps": total})  # R1-LOG
    lr_at = TR.LRSchedule(a.lr, a.warmup, total)
    ll = ds.lead_sup_labels
    meta = {"S": S, "total": total, "n_windows": info["n_windows"], "n_windows_all_clips": info["n_windows_all_clips"], "n_clips": info["n_clips"], "train_sha256": info.get("train_sha256"),
            "train_window_list_sha256": info.get("train_window_list_sha256"), "excluded_sha256": info.get("excluded_sha256"),
            "lead_labels": {k: ll[k] for k in ("sha256", "n_rows", "n_pres", "n_gap", "n_clos")}, "shuffle_closing": shuf, "trainer_argv": argv, "batch": a.batch, "lr": a.lr, "wd": a.wd,
            "warmup": a.warmup, "clip_grad": a.clip_grad, "w_lead_sup": cfg.w_lead_sup, "hmax": cfg.hmax, "steps_override": args.steps or 0,
            "grid": args.grid, "n_tok": GRIDS[args.grid], "host_est": info["host_est"], "sampler_chunk": chunk, "sampler_chunk_like": like or None, "lru_trainer": info["lru"]}
    json.dump(meta, open(os.path.join(args.out, "train_meta.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=str)

    def stream():
        ep = 0
        while True:
            pf = TR.Prefetch(r1.watch(sampler.epoch(ep)), lambda idx: D.collate([ds.item(*w) for w in idx]), depth=3)
            for b in pf:
                yield b
            ep += 1

    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    run_training(stream(), names, total, lr_at, lr=a.lr, wd=a.wd, clip_grad=a.clip_grad, w_loss=cfg.w_lead_sup, device=dev, out_dir=args.out, hist=cfg.lead_hist, header=meta,
                 input_norm=args.input_norm, argv_sha=argv_sha, grid=args.grid, expect_batch=a.batch)
    r1.flush()
    res = {n: check_log(os.path.join(args.out, n)) for n in names}
    done = {"done": True, "steps": total, "arms": names, "grid": args.grid, "n_tok": GRIDS[args.grid], "seconds": time.time() - t0, "p5": res, "p5_all_pass": all(r["pass"] for r in res.values()), "input_norm": args.input_norm, "sampler_chunk": chunk, "r1_sequence_sha256": hashlib.sha256("".join(r1.full).encode()).hexdigest(),
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(done, open(os.path.join(args.out, "TRAIN_DONE.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print("TRAIN_DONE", json.dumps({"steps": total, "p5_all_pass": done["p5_all_pass"], "seconds": round(done["seconds"])}))
    return 0 if done["p5_all_pass"] else 2


# ============================================================================================ extract
def history_rows(clip_idx, t, hist):
    """Copy of pod_lead_apply.history_rows (PA:40-52): idx [n, hist] = row of the frames t-hist+1 .. t of the same clip (oldest first), -1 where any is absent."""
    clip_idx = np.asarray(clip_idx, np.int64)
    t = np.asarray(t, np.int64)
    key = {(int(c), int(g)): i for i, (c, g) in enumerate(zip(clip_idx, t))}
    idx = np.full((len(t), hist), -1, np.int64)
    for j in range(hist):
        off = hist - 1 - j
        for i, (c, g) in enumerate(zip(clip_idx, t)):
            idx[i, j] = key.get((int(c), int(g) - off), -1)
    ok = (idx >= 0).all(1)
    idx[~ok] = -1
    return idx, ok


def head_outputs_from_states(probe, Z, idx, ok, device, chunk=4096):
    """The probe's head on float16 states cast to float32: per-frame embeddings once, then the read-out of the gathered histories (the arithmetic of LeadSupHead.forward). NaN where no history."""
    head = probe.head
    raw = np.full((len(Z), 3), np.nan, np.float32)
    with torch.no_grad():
        E = np.empty((len(Z), head.frame.out_features), np.float32)
        for i in range(0, len(Z), chunk):
            E[i:i + chunk] = head.frame(torch.from_numpy(np.asarray(Z[i:i + chunk], np.float32)).to(device)).float().cpu().numpy()
        rows = np.flatnonzero(ok)
        for i in range(0, len(rows), chunk):
            r = rows[i:i + chunk]
            w = torch.from_numpy(E[idx[r]].reshape(len(r), -1)).to(device)
            raw[r] = head.mlp(w).float().cpu().numpy()
    return raw


def jdump(o, path):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(o, fh, indent=1)


def extract_split(probe, ck, ckpt_path, ck_sha, cache, split, lo, hi, out_states, device, chunk_frames=256, grid=GRID_DEFAULT):
    """States and lead outputs of one split in the layouts of extract_states.py and pod_lead_apply.py. Returns the two record dicts.
    The tokens are the cache key ``grid`` of every clip, [T, n_tok, 1024]; any other shape is refused (control g5). The state array is filled in place (one copy in memory, not two)."""
    D = tree()["D"]
    t0 = time.time()
    man = D.read_manifest(cache, split)
    nfr = dict(man)
    clips = sorted([c for c, _ in man], key=D.sha12)[lo:hi]
    total = int(sum(int(nfr[c]) for c in clips))
    Z = np.empty((total, probe.n_tok * probe.W), np.float16)
    ci, tt = np.empty(total, np.int32), np.empty(total, np.int16)
    o = 0
    with torch.no_grad():
        for j, c in enumerate(clips):
            d = torch.load(os.path.join(cache, split, "clips", c + ".pt"), map_location="cpu")
            if grid not in d:
                raise SystemExit("REFUSED: clip %s has no cache key %s (has %s)" % (c, grid, [k for k in d if str(k).startswith("pool")]))
            x = d[grid]                                                        # [T, n_tok, 1024]
            if x.ndim != 3 or x.shape[1] != probe.n_tok or x.shape[2] != D_FEAT or x.shape[0] != int(nfr[c]):
                raise SystemExit("REFUSED: clip %s: %s is %s, expected [%d, %d, %d]" % (c, grid, list(x.shape), int(nfr[c]), probe.n_tok, D_FEAT))
            T_ = x.shape[0]
            for i in range(0, T_, chunk_frames):
                Z[o + i:o + min(i + chunk_frames, T_)] = probe.project(x[i:i + chunk_frames].float().to(device)).cpu().to(torch.float16).numpy()
            ci[o:o + T_] = j
            tt[o:o + T_] = np.arange(T_, dtype=np.int16)
            o += T_
            del d, x
    assert o == total
    os.makedirs(os.path.dirname(os.path.abspath(out_states)), exist_ok=True)
    with open(out_states, "wb") as fh:
        np.savez(fh, Z=Z, clip_idx=ci, t=tt, clips=np.array(clips))
    st_sha = sha256_file(out_states)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec = {"ckpt": ckpt_path, "ckpt_sha256": ck_sha, "step": ck.get("step"), "wm_state": "tokprobe", "wm": "tokprobe", "D": int(Z.shape[1]), "n_frames": int(Z.shape[0]), "n_clips": len(clips),
           "split": split, "lo": lo, "hi": hi, "n_manifest": len(man), "out_sha256": st_sha, "written_utc": now, "seconds": round(time.time() - t0, 1),
           "arm": ck.get("arm"), "W": ck["W"], "probe": "tokprobe", "input_norm": ck["input_norm"], "grid": grid, "n_tok": probe.n_tok, "n_tok_seen_in_training": ck.get("n_tok_seen")}
    # the lead outputs from the STORED float16 states (the reader's forward on Z reproduces them)
    hist = int(probe.head.hist)
    idx, ok = history_rows(ci, tt, hist)
    raw = head_outputs_from_states(probe, Z, idx, ok, device)
    base = out_states[:-4] if out_states.endswith(".npz") else out_states
    lead_p = base + "_lead.npz"
    out = {"presence_logit": raw[:, 0].copy(), "gap_m": (raw[:, 1] * GAP_SCALE_M).astype(np.float32), "closing_ms": (raw[:, 2] * CLOS_SCALE_MS).astype(np.float32), "has_history": ok, "hist": np.int64(hist)}
    with open(lead_p, "wb") as fh:
        np.savez(fh, **out)
    sd = ck["state_dict"]
    tens = {nm: sd[key].detach().cpu().numpy().astype(np.float32) for nm, key in HEAD_KEYS.items()}
    hp = lead_p + ".head.npz"
    with open(hp, "wb") as fh:
        np.savez(fh, **tens, hist=np.int64(hist))
    lrec = {"ckpt": ckpt_path, "ckpt_sha256": ck_sha, "states": out_states, "states_sha256": st_sha, "states_record_ckpt_sha256": ck_sha, "out": lead_p, "out_sha256": sha256_file(lead_p),
            "head": hp, "head_sha256": sha256_file(hp), "n_rows": int(len(Z)), "n_applied": int(ok.sum()), "n_nan": int(len(Z) - ok.sum()), "hist": hist, "n_clips": len(clips),
            "cfg_lead": {"lead_sup": "on", "w_lead_sup": 1.0, "lead_hist": hist, "lead_sup_detach": "tokprobe"}, "step": ck.get("step"), "plumbing": False,
            "seconds": round(time.time() - t0, 1), "written_utc": now}
    jdump(lrec, lead_p + ".json")
    jdump(rec, out_states + ".json")                                          # the state record LAST (a finished split has it)
    return rec, lrec


def cmd_extract(args):
    use_tree(args.tree)
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or os.path.join(args.run_dir, args.arm, "ckpt_last.pt")
    probe, ck = load_probe(ckpt, dev)
    if args.arm and ck.get("arm") != args.arm:
        raise SystemExit("REFUSED: the checkpoint is arm %s, not %s" % (ck.get("arm"), args.arm))
    check_arm_grid(ck["arm"], args.grid)
    if ck.get("grid", GRID_DEFAULT) != args.grid:
        raise SystemExit("REFUSED: the checkpoint records grid %s, the extract declares %s" % (ck.get("grid", GRID_DEFAULT), args.grid))
    if args.grid != GRID_DEFAULT and ck.get("n_tok_seen") != GRIDS[args.grid]:
        raise SystemExit("REFUSED: the checkpoint saw %s tokens per frame in training, the grid %s has %d" % (ck.get("n_tok_seen"), args.grid, GRIDS[args.grid]))
    sha = sha256_file(ckpt)
    os.makedirs(args.states_dir, exist_ok=True)
    rng = {"eval": tuple(int(x) for x in args.eval_range.split(":")), "train": tuple(int(x) for x in args.train_range.split(":"))}
    for sp in args.splits.split(","):
        lo, hi = rng[sp]
        outp = os.path.join(args.states_dir, "%s_%s.npz" % (ck["arm"], sp))
        if os.path.exists(outp + ".json") and not args.force:
            print("extract: %s exists, skipped" % outp)
            continue
        rec, lrec = extract_split(probe, ck, ckpt, sha, args.cache, sp, lo, hi, outp, dev, grid=args.grid)
        print("extract %s %s: %d rows, %d with a %d-frame history" % (ck["arm"], sp, rec["n_frames"], lrec["n_applied"], lrec["hist"]))
    return 0


# ============================================================================================ control g2
def pool2x2(t160):
    """Row-major 8 by 20 tokens [..., 160, D] -> the 2 by 2 mean on the 4 by 10 layout [..., 40, D] (token index = row * 20 + column)."""
    s = t160.shape
    x = t160.reshape(*s[:-2], 4, 2, 10, 2, s[-1])
    return x.mean(dim=(-4, -2)).reshape(*s[:-2], 40, s[-1])


def pool2x2_colmajor(t160):
    """The same read with the token index = column * 8 + row (diagnostic beside, in case the grid was stored column-major)."""
    s = t160.shape
    x = t160.reshape(*s[:-2], 20, 8, s[-1]).transpose(-3, -2).reshape(*s[:-2], 160, s[-1])
    return pool2x2(x)


def _corr(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def cmd_g2(args):
    """Control g2 (card ADD41 section 4): the 2 by 2 mean of the stored 8 by 20 tokens against the stored 4 by 10 tokens, on ``--frames`` evenly spaced frames of the first ``--n-clips`` clips of the train
    manifest order (card: 20 clips, 10 frames each = 200 frames). Pooled Pearson correlation over every element, the median and the maximum absolute difference. Written to ``--out``; the exit code is 0 either way."""
    D = use_tree(args.tree)["D"]
    man = D.read_manifest(args.cache, args.split)
    clips = [c for c, _ in man][:args.n_clips]
    A, B, C = [], [], []
    per_clip, miss = [], []
    with torch.no_grad():
        for c in clips:
            d = torch.load(os.path.join(args.cache, args.split, "clips", c + ".pt"), map_location="cpu")
            if "pool8x20" not in d or "pool4x10" not in d:
                miss.append(c)
                continue
            T_ = d["pool4x10"].shape[0]
            fr = np.unique(np.round(np.linspace(0, T_ - 1, args.frames)).astype(int))
            fi = torch.as_tensor(fr)
            f8, f4 = d["pool8x20"].float()[fi], d["pool4x10"].float()[fi]  # cast first: the stored float8 dtype has no tensor-index kernel on the CPU (pod precheck of 2026-11-18)
            if tuple(f8.shape[1:]) != (160, D_FEAT) or tuple(f4.shape[1:]) != (40, D_FEAT):
                miss.append(c)
                continue
            p = pool2x2(f8)
            A.append(p.flatten())
            B.append(f4.flatten())
            C.append(pool2x2_colmajor(f8).flatten())
            per_clip.append({"clip": c, "frames": [int(x) for x in fr], "corr": _corr(p, f4), "dtype_8x20": str(d["pool8x20"].dtype), "dtype_4x10": str(d["pool4x10"].dtype)})
    out = {"card": "PREREG_V8F_PROOF_ADD41 control g2", "split": args.split, "cache": args.cache, "n_clips_asked": args.n_clips, "n_clips_read": len(per_clip), "clips_missing_a_grid": miss,
           "frames_per_clip_asked": args.frames, "layout": "row-major: 8 by 20 token index = row * 20 + column; 2 by 2 mean to 4 by 10 token index = row * 10 + column", "bar_corr": args.bar,
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if per_clip and not miss:
        a, b, c3 = torch.cat(A), torch.cat(B), torch.cat(C)
        ad = (a - b).abs()
        out.update({"n_frames": int(sum(len(x["frames"]) for x in per_clip)), "corr": _corr(a, b), "median_abs_diff": float(ad.median()), "max_abs_diff": float(ad.max()),
                    "mean_abs_4x10": float(b.abs().mean()), "min_clip_corr": min(x["corr"] for x in per_clip), "alt_colmajor_corr": _corr(c3, b), "per_clip": per_clip})
        out["pass"] = bool(out["corr"] >= args.bar)
    else:
        out.update({"pass": False, "reason": "a clip lacks a grid or has an unexpected shape" if miss else "no clip read"})
    jdump(out, args.out)
    print("g2: corr %s median_abs_diff %s pass %s -> %s" % (out.get("corr"), out.get("median_abs_diff"), out["pass"], args.out))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("g2")
    g.add_argument("--tree", default=DEFAULT_TREE)
    g.add_argument("--cache", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--split", default="train")
    g.add_argument("--n-clips", type=int, default=20)
    g.add_argument("--frames", type=int, default=10)
    g.add_argument("--bar", type=float, default=0.99)
    t = sub.add_parser("train")
    t.add_argument("--grid", choices=sorted(GRIDS), default=GRID_DEFAULT)
    t.add_argument("--tree", default=DEFAULT_TREE)
    t.add_argument("--out", required=True)
    t.add_argument("--trainer-argv", default=DEFAULT_TRAINER_ARGV)
    t.add_argument("--arms", default="")
    t.add_argument("--steps", type=int, default=0, help="the trainer's --steps override (short runs)")
    t.add_argument("--expect-steps", type=int, default=0)
    t.add_argument("--shuffle-closing-labels", action="store_true", help="control p3")
    t.add_argument("--input-norm", choices=("ln", "none"), default="ln")
    t.add_argument("--device", default="auto")
    t.add_argument("--threads", type=int, default=0)
    t.add_argument("--sampler-chunk-like", default="", help="take the sampler chunk the trainer would take for this grid (control r1); default: the run's own grid")
    e = sub.add_parser("extract")
    e.add_argument("--grid", choices=sorted(GRIDS), default=GRID_DEFAULT)
    e.add_argument("--tree", default=DEFAULT_TREE)
    e.add_argument("--run-dir", default="")
    e.add_argument("--arm", default="")
    e.add_argument("--ckpt", default="")
    e.add_argument("--cache", required=True)
    e.add_argument("--states-dir", required=True)
    e.add_argument("--splits", default="eval,train")
    e.add_argument("--eval-range", default="0:141")
    e.add_argument("--train-range", default="600:1800")
    e.add_argument("--device", default="auto")
    e.add_argument("--threads", type=int, default=0)
    e.add_argument("--force", action="store_true")
    c = sub.add_parser("check-log")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--arms", default="")
    r = sub.add_parser("r1")
    r.add_argument("--a", required=True)
    r.add_argument("--b", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "r1":
        return cmd_r1(a)
    if a.cmd == "g2":
        return cmd_g2(a)
    if a.cmd == "train":
        return cmd_train(a)
    if a.cmd == "extract":
        if not (a.ckpt or (a.run_dir and a.arm)):
            raise SystemExit("REFUSED: give --ckpt, or --run-dir and --arm")
        return cmd_extract(a)
    names = [n for n in (a.arms.split(",") if a.arms else sorted(d for d in os.listdir(a.run_dir) if os.path.isdir(os.path.join(a.run_dir, d)) and d.startswith(("TK", "F")))) if n]
    res = {n: check_log(os.path.join(a.run_dir, n)) for n in names}
    print(json.dumps(res, indent=1))
    return 0 if res and all(r["pass"] for r in res.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
