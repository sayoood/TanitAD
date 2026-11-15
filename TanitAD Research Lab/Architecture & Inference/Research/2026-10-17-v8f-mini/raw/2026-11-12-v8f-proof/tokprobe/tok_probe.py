"""tok_probe.py -- TOKPROBE, the probe tool of the frozen card PREREG_V8F_PROOF_ADD27 (definitions: TOKPROBE_DEFS.md).

A projection of the cached pooled frozen camera tokens (40 tokens of width 1,024 per frame), trained for the lead labels alone, read with the LeadSupHead of card ADD24. No training of the model.
Tier T0 (a diagnostic on logged frames), never driving performance. STRATEGIC: PAUSED (PI item 25). Runs on the pod GPU (train, extract); the dev box tests it on CPU with synthetic data.

  python tok_probe.py train   --tree /workspace/v8proof/tree_leadsup --out DIR [--steps N] [--expect-steps 19164] [--shuffle-closing-labels] [--arms TK16_s0,...]
  python tok_probe.py extract --tree TREE --run-dir DIR --arm TK16_s0 --cache /workspace/v8proof/data/cache_pod --states-dir /workspace/v8proof/probe/states
  python tok_probe.py check-log --run-dir DIR                       (control p5)

The probe's forward takes the token array ONLY (control p4). Nothing here reads git, the pod, box 2 or the model's checkpoints.
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
ARMS_DEFAULT = ("TK16_s0", "TK16_s1", "TK64_s0", "TK64_s1")
GAP_SCALE_M, CLOS_SCALE_MS = 50.0, 10.0                     # = M.LEAD_SUP_GAP_SCALE_M / LEAD_SUP_CLOS_SCALE_MS (asserted in use_tree)
LOG_EVERY = 50
SHUFFLE_RNG_SEED = 12345
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
    m = re.match(r"^TK(\d+)(SH)?_s(\d+)$", name)
    if not m:
        raise SystemExit("REFUSED: arm name %r is not TK<W>[SH]_s<seed>" % name)
    return {"W": int(m.group(1)), "shuffle": bool(m.group(2)), "seed": int(m.group(3))}


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


def build_probe(arm: str, input_norm: str = "ln"):
    """The arm's probe; ``torch.manual_seed(seed)`` is called immediately before the construction, so the initial weights depend on seed and shape only."""
    p = parse_arm(arm)
    torch.manual_seed(p["seed"])
    return TokenProbe(p["W"], input_norm=input_norm)


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


def run_training(stream, arms, total, lr_at, *, lr, wd, clip_grad, w_loss, device, out_dir, hist=HIST, log_every=LOG_EVERY, header=None, input_norm="ln", argv_sha=""):
    """Train the arms on the SAME batches of ``stream`` (an iterator of collated batches) for ``total`` steps in one process. Per arm: ``<out_dir>/<ARM>/train_log.jsonl`` and ``ckpt_last.pt``."""
    dev = torch.device(device)
    probes = {a: build_probe(a, input_norm).to(dev).train() for a in arms}
    opts = {a: make_opt(p, lr, wd) for a, p in probes.items()}
    logs = {}
    for a, p in probes.items():
        os.makedirs(os.path.join(out_dir, a), exist_ok=True)
        logs[a] = open(os.path.join(out_dir, a, "train_log.jsonl"), "w", encoding="utf-8", newline="\n")
        logs[a].write(json.dumps({"header": True, "arm": a, "W": p.W, "seed": parse_arm(a)["seed"], "input_norm": input_norm, "total": total, "argv_sha": argv_sha,
                                  "n_params": {"tok": sum(q.numel() for q in p.tok.parameters()), "head": sum(q.numel() for q in p.head.parameters()), "ctrl": 1}, **(header or {})}) + "\n")
    acc = {a: {"sum": torch.zeros(4, device=dev), "n": 0, "ngap_min": torch.full((), float("inf"), device=dev)} for a in probes}
    t0 = time.time()
    step = 0
    for b in stream:
        if step >= total:
            break
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
        save_probe(p, a, os.path.join(out_dir, a, "ckpt_last.pt"), step, argv_sha)
    return probes


def save_probe(p, arm, path, step, argv_sha=""):
    blob = {"kind": "tokprobe", "arm": arm, "W": p.W, "seed": parse_arm(arm)["seed"], "n_tok": p.n_tok, "d_feat": p.d_feat, "hidden": p.hidden, "hist": p.hist,
            "input_norm": p.input_norm, "step": int(step), "argv_sha": argv_sha, "state_dict": {k: v.detach().cpu() for k, v in p.state_dict().items()}}
    torch.save(blob, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_probe(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != "tokprobe":
        raise SystemExit("REFUSED: %s is not a tokprobe checkpoint" % path)
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
    out["pass_gap_labels_every_step"] = bool(ng > 0)
    out["pass_complete"] = bool(recs and recs[-1]["step"] == total)
    out["pass"] = bool(out["pass_lead_term_falls"] and out["pass_gap_labels_every_step"] and out["pass_complete"])
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
    return ds, wins, S, total, info


def shuffle_closing(ds, seed=SHUFFLE_RNG_SEED):
    """Control p3: permute the finite closing-speed labels among the rows that have one (the mask, hence the label counts, stay)."""
    LS = ds.lead_sup_labels
    c = np.array(LS["clos"], copy=True)
    idx = np.flatnonzero(np.isfinite(c))
    c[idx] = c[idx][np.random.default_rng(seed).permutation(len(idx))]
    LS["clos"] = c
    return {"n_shuffled": int(len(idx)), "rng_seed": seed}


def trainer_argv(text, out):
    toks = shlex.split(text)
    res, i = [], 0
    while i < len(toks):
        if toks[i] in ("--out", "--device"):
            i += 2
            continue
        res.append(toks[i])
        i += 1
    return res + ["--out", out, "--device", "cpu"]


def cmd_train(args):
    T = use_tree(args.tree)
    D, TR = T["D"], T["TR"]
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    argv = trainer_argv(args.trainer_argv, args.out)
    a = TR.build_parser().parse_args(argv)
    if args.steps:
        a.steps = args.steps
    cfg = TR.make_config(a)
    if cfg.lead_sup != "on" or cfg.lead_hist != HIST or not cfg.w_lead_sup > 0:
        raise SystemExit("REFUSED: the argv builds lead_sup %r / lead_hist %r / w_lead_sup %r" % (cfg.lead_sup, cfg.lead_hist, cfg.w_lead_sup))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    ds, wins, S, total, info = build_training_data(a, cfg)
    if args.expect_steps and total != args.expect_steps:
        raise SystemExit("REFUSED: the computed step count is %d, expected %d" % (total, args.expect_steps))
    shuf = None
    names = [n for n in (args.arms or ("TK16_s0" if args.shuffle_closing_labels else ",".join(ARMS_DEFAULT))).split(",") if n]
    if args.shuffle_closing_labels:
        shuf = shuffle_closing(ds)
        names = [n if "SH_" in n else n.replace("_s", "SH_s", 1) for n in names]
    for n in names:
        parse_arm(n)
    argv_sha = hashlib.sha256(json.dumps(vars(args), sort_keys=True, default=str).encode()).hexdigest()[:16]
    sampler = D.ChunkedSampler(wins, a.batch, seed=a.seed, chunk=max(4, min(48, info["lru"] - 2)), max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    lr_at = TR.LRSchedule(a.lr, a.warmup, total)
    ll = ds.lead_sup_labels
    meta = {"S": S, "total": total, "n_windows": info["n_windows"], "n_windows_all_clips": info["n_windows_all_clips"], "n_clips": info["n_clips"], "train_sha256": info.get("train_sha256"),
            "train_window_list_sha256": info.get("train_window_list_sha256"), "excluded_sha256": info.get("excluded_sha256"),
            "lead_labels": {k: ll[k] for k in ("sha256", "n_rows", "n_pres", "n_gap", "n_clos")}, "shuffle_closing": shuf, "trainer_argv": argv, "batch": a.batch, "lr": a.lr, "wd": a.wd,
            "warmup": a.warmup, "clip_grad": a.clip_grad, "w_lead_sup": cfg.w_lead_sup, "hmax": cfg.hmax, "steps_override": args.steps or 0}
    json.dump(meta, open(os.path.join(args.out, "train_meta.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=str)

    def stream():
        ep = 0
        while True:
            pf = TR.Prefetch(sampler.epoch(ep), lambda idx: D.collate([ds.item(*w) for w in idx]), depth=3)
            for b in pf:
                yield b
            ep += 1

    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    run_training(stream(), names, total, lr_at, lr=a.lr, wd=a.wd, clip_grad=a.clip_grad, w_loss=cfg.w_lead_sup, device=dev, out_dir=args.out, hist=cfg.lead_hist, header=meta,
                 input_norm=args.input_norm, argv_sha=argv_sha)
    res = {n: check_log(os.path.join(args.out, n)) for n in names}
    done = {"done": True, "steps": total, "arms": names, "seconds": time.time() - t0, "p5": res, "p5_all_pass": all(r["pass"] for r in res.values()), "input_norm": args.input_norm,
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


def extract_split(probe, ck, ckpt_path, ck_sha, cache, split, lo, hi, out_states, device, chunk_frames=256):
    """States and lead outputs of one split in the layouts of extract_states.py and pod_lead_apply.py. Returns the two record dicts."""
    D = tree()["D"]
    t0 = time.time()
    man = D.read_manifest(cache, split)
    clips = sorted([c for c, _ in man], key=D.sha12)[lo:hi]
    Zs, Ci, Ts = [], [], []
    with torch.no_grad():
        for j, c in enumerate(clips):
            d = torch.load(os.path.join(cache, split, "clips", c + ".pt"), map_location="cpu")
            x = d["pool4x10"].float()                                         # [T, 40, 1024]
            z = torch.cat([probe.project(x[i:i + chunk_frames].to(device)).cpu() for i in range(0, x.shape[0], chunk_frames)])
            Zs.append(z.to(torch.float16).numpy())
            Ci.append(np.full(z.shape[0], j, np.int32))
            Ts.append(np.arange(z.shape[0], dtype=np.int16))
            del d, x, z
    Z = np.concatenate(Zs)
    del Zs
    ci, tt = np.concatenate(Ci), np.concatenate(Ts)
    os.makedirs(os.path.dirname(os.path.abspath(out_states)), exist_ok=True)
    with open(out_states, "wb") as fh:
        np.savez(fh, Z=Z, clip_idx=ci, t=tt, clips=np.array(clips))
    st_sha = sha256_file(out_states)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec = {"ckpt": ckpt_path, "ckpt_sha256": ck_sha, "step": ck.get("step"), "wm_state": "tokprobe", "wm": "tokprobe", "D": int(Z.shape[1]), "n_frames": int(Z.shape[0]), "n_clips": len(clips),
           "split": split, "lo": lo, "hi": hi, "n_manifest": len(man), "out_sha256": st_sha, "written_utc": now, "seconds": round(time.time() - t0, 1),
           "arm": ck.get("arm"), "W": ck["W"], "probe": "tokprobe", "input_norm": ck["input_norm"]}
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
    sha = sha256_file(ckpt)
    os.makedirs(args.states_dir, exist_ok=True)
    rng = {"eval": tuple(int(x) for x in args.eval_range.split(":")), "train": tuple(int(x) for x in args.train_range.split(":"))}
    for sp in args.splits.split(","):
        lo, hi = rng[sp]
        outp = os.path.join(args.states_dir, "%s_%s.npz" % (ck["arm"], sp))
        if os.path.exists(outp + ".json") and not args.force:
            print("extract: %s exists, skipped" % outp)
            continue
        rec, lrec = extract_split(probe, ck, ckpt, sha, args.cache, sp, lo, hi, outp, dev)
        print("extract %s %s: %d rows, %d with a %d-frame history" % (ck["arm"], sp, rec["n_frames"], lrec["n_applied"], lrec["hist"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
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
    e = sub.add_parser("extract")
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
    a = ap.parse_args(argv)
    if a.cmd == "train":
        return cmd_train(a)
    if a.cmd == "extract":
        if not (a.ckpt or (a.run_dir and a.arm)):
            raise SystemExit("REFUSED: give --ckpt, or --run-dir and --arm")
        return cmd_extract(a)
    names = [n for n in (a.arms.split(",") if a.arms else sorted(d for d in os.listdir(a.run_dir) if os.path.isdir(os.path.join(a.run_dir, d)) and d.startswith("TK"))) if n]
    res = {n: check_log(os.path.join(a.run_dir, n)) for n in names}
    print(json.dumps(res, indent=1))
    return 0 if res and all(r["pass"] for r in res.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
