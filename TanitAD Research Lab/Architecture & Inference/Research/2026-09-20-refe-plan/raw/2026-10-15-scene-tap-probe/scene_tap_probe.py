#!/usr/bin/env python3
"""The scene-tap vs visual-tap probe for REFe's auxiliary drivable-area head (PREREG_SCENE_TAP_PROBE.md; zero A40).

QUESTION.  Can FROZEN features carry the road at P1 IoU [0, 40) m >= 0.80 (K2's bar), and through which tap: V = visual_ctx [7680, 256] (what S1 read: 0.496 / 0.521 / 0.486)
or S = scene_ctx [64, 256] (the registers' compression of the same tokens)?  A head-only fit on CACHED frozen features, 5 log-disjoint folds over the 1,058 navtest C3 tokens.

ARMS (all: the S1 head's own config, AuxDrivable depth 2 feed off, AdamW wd 0.01, lr 2e-4 peak, warmup 50 + cosine, w_aux 0.1, w_sdf 1.0, grad-clip 1.0, bf16 autocast in training)
    V  tap visual                         seeds 0 1 2
    S  tap scene (identical to V except the tap)   seeds 0 1 2
    C  V with the TRAIN-fold labels deranged across the train tokens (one fixed derangement per fold)   seed 0
EVERY (arm, seed, fold) unit trains on the 4 other folds for a FIXED number of updates at batch 8, then predicts its held-out fold once.  No early stopping, no selection on held-out data.

READ (reused, never reimplemented): `aux_s1_readout.frame_stats` (per-frame sufficient statistics against the navtest label store) and `aux_s1_readout.perception_reads` (P1 / P2 / K2),
the log-cluster bootstrap weights `aux_s1_readout.boot_weights` and `aux_s1_readout.by_log`.  `guard_p1_function` fails the run if either has been replaced.

    python scene_tap_probe.py folds                     # write folds.json (+ sha256) and print the instrument facts
    python scene_tap_probe.py selfcheck                 # real-universe guards (no training)
    python scene_tap_probe.py smoke  --out <dir>        # ~20 updates on a tiny subset: timing only, NO metric
    python scene_tap_probe.py run    --out <dir> [--updates 2000]      # the whole grid (resumable); writes PROBE_RESULT.json then PROBE_DONE | PROBE_FAILED
    python scene_tap_probe.py report --out <dir>        # re-assemble PROBE_RESULT.json from the unit files

TIER: T0-like (a head's perception on teacher-forced navtest frames); never driving performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))                  # .../2026-09-20-refe-plan
for _p in (os.path.join(PKG, "eval"), os.path.join(PKG, "refe")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402  (the head, its loss and the label geometry: the module the S1 trainer builds)
import aux_s1_readout as ASR  # noqa: E402  (the P1 / P2 / K2 readout and the bootstrap draw scheme)

TOOL = "scene_tap_probe/1"
TIER = "T0-like: head perception on teacher-forced navtest frames, never driving performance"

# ---- committed protocol (PREREG_SCENE_TAP_PROBE.md section 3) ---------------------------------------------------------------------------------------------------------
K_FOLDS = 5
PERM_SEED = 20260927                       # fold f's derangement uses seed PERM_SEED + f
SEED_BOOT = ASR.SEED                       # 20260927
N_BOOT = ASR.N_BOOT                        # 10000
N_UPDATES = 2000
BATCH = 8
LR_PEAK = 2e-4                             # --aux-drv-lr 2e-4 (s1aux_pod.sh arm_args)
WARMUP = 50                                # scorer_finetune.py --warmup default
WEIGHT_DECAY = 0.01                        # scorer_finetune.py:613
CLIP = 1.0                                 # scorer_finetune.py:1362
W_AUX = 0.1                                # --w-aux-drv 0.1
W_SDF = 1.0                                # AuxDrivableConfig default (the trainer does not override it)
DEPTH = 2
K2_BAR = ASR.K2_IOU_MIN                    # 0.80
K2_BAND_INDEX = 3                          # frame_stats' bands = BANDS (0-20, 20-40, 40-72) + K2_BAND (0-40) + all cells
SEEDS = {"V": (0, 1, 2), "S": (0, 1, 2), "C": (0,)}
ARMS = {"V": {"tap": "visual", "perm": False}, "S": {"tap": "scene", "perm": False}, "C": {"tap": "visual", "perm": True}}
UNIT_ORDER = ([("V", 0, f) for f in range(K_FOLDS)] + [("S", 0, f) for f in range(K_FOLDS)] + [("C", 0, f) for f in range(K_FOLDS)]
              + [("V", 1, f) for f in range(K_FOLDS)] + [("S", 1, f) for f in range(K_FOLDS)]
              + [("V", 2, f) for f in range(K_FOLDS)] + [("S", 2, f) for f in range(K_FOLDS)])

# ---- paths -------------------------------------------------------------------------------------------------------------------------------------------------------------
ART = "D:/Projects/TanitAD-artifacts"
C3_DIR = os.path.join(PKG, "raw", "2026-10-11-c3-selection-headroom")
TOKENS_JSON = os.path.join(C3_DIR, "tokens.json")
TABLE_NPZ = os.path.join(C3_DIR, "table.npz")
CTX_DIRS = {"visual": f"{ART}/aux_navtest_forward/c3_full_ctx", "scene": f"{ART}/aux_navtest_forward/c3_full_scene_ctx"}
CTX_LEN = {"visual": 7680, "scene": 64}
CTX_DIM = 256
ART_OUT = f"{ART}/scene_tap_probe"

# the mechanical outcome table (PREREG section 4, committed verbatim)
ROWS = {
    "ROW1_V_CARRIES": ("frozen VISUAL features carry the road; S1's 0.50 is a head-TRAINING limit (data / domain), not a feature limit",
                       "S1c: the visual head with more label logs (cost it); scene tap not needed"),
    "ROW2_S_CARRIES": ("the SCENE tap carries the road, the visual tap does not", "S1b: a scene-tap A40 arm (cost it)"),
    "ROW3_S_BETTER_NOT_ENOUGH": ("scene is the better frozen tap but not enough",
                                 "unfrozen-trunk graft = PI decision (scene tap preferred), plus the in-sample read"),
    "ROW4_SCENE_ELIMINATED": ("scene tap ELIMINATED as the K2 lever", "unfrozen-trunk graft = PI decision, plus the in-sample read"),
}


class GuardFailed(AssertionError):
    """A probe guard fired.  Under a deliberate mutation this is the EXPECTED outcome (the test turns RED = the guard is live)."""


def log(msg: str, fh=None) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def dump_json_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=1) + "\n").encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# =====================================================================================================================================================================
# the universe and the folds
# =====================================================================================================================================================================
def fold_of(log_id: str, k: int = K_FOLDS) -> int:
    """fold = int(sha256(log_id).hexdigest(), 16) % k, the log id exactly as `aux_s1_readout` keys it (the label-store shard / head_pred `log`)."""
    return int(hashlib.sha256(str(log_id).encode("utf-8")).hexdigest(), 16) % k


def folds_doc(tokens, token_log) -> dict:
    rows = [[t, token_log[t], fold_of(token_log[t])] for t in tokens]
    logs = sorted(set(token_log[t] for t in tokens))
    return {"tool": TOOL, "rule": "fold = int(sha256(log_id.encode('utf-8')).hexdigest(), 16) % 5", "k": K_FOLDS, "n_tokens": len(tokens), "n_logs": len(logs),
            "token_order": "table.npz['token'] order (raw/2026-10-11-c3-selection-headroom/table.npz)",
            "tokens_per_fold": {str(f): int(sum(1 for r in rows if r[2] == f)) for f in range(K_FOLDS)},
            "logs_per_fold": {str(f): int(sum(1 for lg in logs if fold_of(lg) == f)) for f in range(K_FOLDS)},
            "columns": ["token", "log", "fold"], "rows": rows}


class Universe:
    """tokens (table order), their log, fold and (log, token) keys."""

    def __init__(self, tokens, token_log):
        self.tokens = list(tokens)
        self.logs = np.asarray([token_log[t] for t in self.tokens])
        self.fold = np.asarray([fold_of(token_log[t]) for t in self.tokens], np.int64)
        self.keys = [(token_log[t], t) for t in self.tokens]
        self.n = len(self.tokens)


def load_universe() -> Universe:
    doc = json.load(open(TOKENS_JSON, encoding="utf-8"))
    tab = np.load(TABLE_NPZ, allow_pickle=False)
    tokens = [str(t) for t in tab["token"]]
    if set(tokens) != set(str(t) for t in doc["tokens"]) or len(set(tokens)) != len(tokens):
        raise GuardFailed("tokens.json and table.npz are not the same token set")
    return Universe(tokens, doc["token_log"])


def make_fold_split(fold_arr: np.ndarray, f: int):
    """-> (train_idx, held_idx) over token positions: held = the fold, train = the other four."""
    return np.flatnonzero(fold_arr != f), np.flatnonzero(fold_arr == f)


def guard_split(train_idx, held_idx, log_arr, n_total: int) -> None:
    """(a) log-disjointness: no token and no LOG on both sides, and the two sides cover every token exactly once."""
    ts, hs = set(int(i) for i in train_idx), set(int(i) for i in held_idx)
    if ts & hs:
        raise GuardFailed(f"{len(ts & hs)} tokens are on both the train and the held-out side (first {sorted(ts & hs)[:1]})")
    lt, lh = set(log_arr[np.asarray(list(ts), np.int64)].tolist()), set(log_arr[np.asarray(list(hs), np.int64)].tolist())
    if lt & lh:
        raise GuardFailed(f"{len(lt & lh)} logs are on both the train and the held-out side (first {sorted(lt & lh)[:1]})")
    if len(ts) + len(hs) != n_total:
        raise GuardFailed(f"train {len(ts)} + held-out {len(hs)} != {n_total}")


# =====================================================================================================================================================================
# the guards of the training path
# =====================================================================================================================================================================
def select_ctx(tap: str, ctxs: dict):
    """The tensor an arm's head reads: ctxs[tap], the very object (not a copy)."""
    return ctxs[tap]


def guard_tap(used, tap: str, ctxs: dict) -> None:
    """(b) the arm reads the RIGHT tap: same storage as the declared tensor (AUX.tap_is), not the other tap's, and the declared token count."""
    if not AUX.tap_is(used, ctxs[tap]):
        raise GuardFailed(f"arm tap {tap!r}: the head's input is not the {tap} ctx tensor")
    other = "scene" if tap == "visual" else "visual"
    if ctxs.get(other) is not None and AUX.tap_is(used, ctxs[other]):
        raise GuardFailed(f"arm tap {tap!r}: the head's input IS the {other} ctx tensor")
    if used.shape[1] != CTX_LEN[tap] and ctxs[tap].shape[1] == CTX_LEN[tap]:
        raise GuardFailed(f"arm tap {tap!r}: {used.shape[1]} ctx tokens, declared {CTX_LEN[tap]}")


def guard_p1_function() -> None:
    """(c) P1 IS `aux_s1_readout`'s own function: this module's reader and stats functions are the readout's objects, and the K2 band is the readout's."""
    if P1_READ is not ASR.perception_reads or STATS_FN is not ASR.frame_stats:
        raise GuardFailed("P1 is not aux_s1_readout's own function (perception_reads / frame_stats have been replaced)")
    if not (ASR.K2_BAND == (0.0, 40.0) and len(ASR.BANDS) == 3 and K2_BAND_INDEX == len(ASR.BANDS) and ASR.K2_IOU_MIN == K2_BAR == 0.80):
        raise GuardFailed("the K2 band / bar / band index differ from the readout's")


def permute_train_labels(labels: torch.Tensor, train_idx, fold: int, seed: int = PERM_SEED):
    """(d) arm C: a derangement of the TRAIN-fold labels across the train tokens, one fixed per fold (seed PERM_SEED + fold).  Held-out rows are never touched.
    -> (new labels [N,160,128] int8, src [n_train]: src[i] = the token whose label train_idx[i] now carries)."""
    tr = [int(i) for i in train_idx]
    src = AUX.derangement(list(tr), int(seed) + int(fold))
    new = labels.clone()
    new[torch.as_tensor(tr, dtype=torch.long)] = labels[torch.as_tensor(src, dtype=torch.long)]
    return new, np.asarray(src, np.int64)


def guard_label_perm(orig: torch.Tensor, new: torch.Tensor, src, train_idx, held_idx) -> None:
    """(d) the permutation acts within the train fold only: src is a fixed-point-free permutation OF the train tokens, the train rows carry exactly the labels src names,
    and every other row (the whole held-out fold included) is bit-identical to the true label."""
    tr = np.asarray(train_idx, np.int64)
    if sorted(src.tolist()) != sorted(tr.tolist()):
        raise GuardFailed("the label permutation is not a permutation of exactly the train tokens")
    if (src == tr).any():
        raise GuardFailed("the label permutation has a fixed point")
    if not torch.equal(new[torch.as_tensor(tr)], orig[torch.as_tensor(src)]):
        raise GuardFailed("a train row does not carry the label its source names")
    ho = torch.as_tensor(np.asarray(held_idx, np.int64))
    if not torch.equal(new[ho], orig[ho]):
        raise GuardFailed("a HELD-OUT label was changed by the permutation")
    mask = torch.ones(orig.shape[0], dtype=torch.bool)
    mask[torch.as_tensor(tr)] = False
    if not torch.equal(new[mask], orig[mask]):
        raise GuardFailed("a non-train label was changed by the permutation")


def draw_batches(train_idx, n_updates: int, batch: int, gen_seed: int):
    """DataLoader(shuffle=True, drop_last=True)-style epochs of a seeded randperm over the TRAIN tokens; yields LongTensor [batch], n_updates times."""
    g = torch.Generator()
    g.manual_seed(int(gen_seed))
    tr = torch.as_tensor(np.asarray(train_idx), dtype=torch.long)
    n = 0
    while n < n_updates:
        perm = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(perm) - batch + 1, batch):
            yield perm[i:i + batch]
            n += 1
            if n >= n_updates:
                return


def guard_batch(idx: torch.Tensor, held_mask: torch.Tensor) -> None:
    """(e) no held-out token is ever in a training batch."""
    if bool(held_mask[idx].any()):
        raise GuardFailed(f"a HELD-OUT token is in a training batch (positions {idx[held_mask[idx]].tolist()[:3]})")


# =====================================================================================================================================================================
# the readout (reused)
# =====================================================================================================================================================================
P1_READ = ASR.perception_reads            # P1 / P2 / K2 for one prediction set
STATS_FN = ASR.frame_stats                # per-frame inter / union / cells in every band + the SDF error sums


class LabelCache:
    """(log, token) -> the pre-decoded TRUE int8 label; the `labels(keys)` interface `aux_s1_readout.frame_stats` calls on its source."""

    def __init__(self, keys, labels: torch.Tensor):
        self.idx = {k: i for i, k in enumerate(keys)}
        self.lab = labels

    def labels(self, keys):
        sel = torch.as_tensor([self.idx[k] for k in keys], dtype=torch.long)
        return self.lab[sel].clone(), torch.ones(len(keys), dtype=torch.bool)


def pred_dict(keys, logit: np.ndarray, sdf: np.ndarray) -> dict:
    return {"keys": list(keys), "logit": np.asarray(logit, np.float32), "sdf": np.asarray(sdf, np.float32), "dropped_step_nonzero": 0,
            "duplicate_keys_identical": 0, "duplicate_keys_within_tol": 0, "duplicate_max_abs_diff": [0.0, 0.0]}


def pooled_p1(keys, logit, sdf, source, n_boot: int = N_BOOT):
    """-> (perception_reads result, per-frame stats).  `perception_reads` is the S1 readout's own P1; `n_boot` only sets how many resamples its CI uses."""
    pred = pred_dict(keys, logit, sdf)
    st = STATS_FN(pred, source)
    return P1_READ(pred, st, None, n_boot=n_boot, seed=SEED_BOOT), st


STAT_BANDS = ("0-20", "20-40", "40-72", "0-40", "all")


def stat_sums(st: dict) -> dict:
    """Sufficient statistics summed over a frame set (for the in-sample read): inter / union / cells per band + the SDF error sums."""
    out = {"n_frames": int(len(st["valid"])), "n_valid": int(st["valid"].sum())}
    for b, nm in enumerate(STAT_BANDS):
        out[f"inter_{nm}"], out[f"union_{nm}"], out[f"cells_{nm}"] = (float(st[f"{k}{b}"].sum()) for k in ("inter", "union", "cells"))
    for k in ("p2_abs", "p2_n", "all_abs", "all_n"):
        out[k] = float(st[k].sum())
    return out


# =====================================================================================================================================================================
# training and prediction
# =====================================================================================================================================================================
def lr_at(u: int, total: int, peak: float = LR_PEAK, warmup: int = WARMUP) -> float:
    """scorer_finetune.py:1331 shape (linear warm-up, then cosine to 0 at `total`), at the head's own peak (the trainer scales it by aux_lr / lr)."""
    return peak * (u + 1) / warmup if u < warmup else peak * 0.5 * (1 + math.cos(math.pi * (u - warmup) / max(total - warmup, 1)))


def make_box(tap: str, seed: int, device) -> "AUX.AuxDrivable":
    cfg = AUX.AuxDrivableConfig(tap=tap, depth=DEPTH, feed="off", w_aux=W_AUX, w_sdf=W_SDF, seed=int(seed))
    return AUX.AuxDrivable(cfg).to(device)


def train_unit(ctx_all: torch.Tensor, labels_train: torch.Tensor, train_idx, held_idx, tap: str, seed: int, fold: int, *, n_updates: int, batch: int = BATCH, device="cpu",
               amp: bool | None = None, draw=None):
    """One (arm, seed, fold) training: the S1 head, its loss, its optimiser, a FIXED number of updates at `batch`, on the train tokens only.
    -> (head, info).  Nothing here sees a held-out label; a held-out token in a batch raises (guard e)."""
    dev = torch.device(device)
    amp = (dev.type == "cuda") if amp is None else amp
    box = make_box(tap, seed, dev)
    opt = torch.optim.AdamW(list(box.head.parameters()), lr=LR_PEAK, weight_decay=WEIGHT_DECAY)
    held_mask = torch.zeros(ctx_all.shape[0], dtype=torch.bool)
    held_mask[torch.as_tensor(np.asarray(held_idx, np.int64))] = True
    drawn = torch.zeros(ctx_all.shape[0], dtype=torch.bool)
    valid = torch.ones(batch, dtype=torch.bool, device=dev)
    draw = draw or draw_batches
    sums, tail, t0 = np.zeros(2), [], time.time()
    gnorm = []
    box.head.train()
    for u, idx in enumerate(draw(train_idx, n_updates, batch, int(seed) * 100 + int(fold))):
        guard_batch(idx, held_mask)
        drawn[idx] = True
        ctx = ctx_all[idx].to(dev, non_blocking=True)
        lab = labels_train[idx].to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(u, n_updates)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            logit, sdf = box.predict(ctx)
        l_aux, bce, hub = box.loss(logit, sdf, lab, valid)
        tot = box.combine(torch.zeros((), device=dev), l_aux)
        tot.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(box.head.parameters()), CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sums += (float(bce), float(hub))
        if u >= n_updates - 100:
            tail.append((float(bce), float(hub)))
        if u % 50 == 0:
            gnorm.append(float(gn))
        if not math.isfinite(float(bce)):
            raise RuntimeError(f"non-finite loss at update {u}")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    tail = np.asarray(tail) if tail else np.zeros((1, 2))
    info = {"updates": int(n_updates), "batch": int(batch), "wall_s": round(time.time() - t0, 2), "distinct_train_tokens_drawn": int(drawn.sum()),
            "n_train_tokens": int(len(train_idx)), "loss_bce_mean_all": float(sums[0] / max(n_updates, 1)), "loss_huber_mean_all": float(sums[1] / max(n_updates, 1)),
            "loss_bce_last100": float(tail[:, 0].mean()), "loss_huber_last100": float(tail[:, 1].mean()), "grad_norm_every50_mean": float(np.mean(gnorm)) if gnorm else None,
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if dev.type == "cuda" else None}
    return box.head, info


@torch.no_grad()
def predict_head(head, ctx_all: torch.Tensor, idx, device="cpu", bs: int = 16):
    """fp32, eval mode (the readout's convention, A2.3): -> (logit [n,40,32], sdf_m [n,40,32]) float32 numpy in `idx` order."""
    dev = torch.device(device)
    head.eval()
    lg, sd = [], []
    idx = np.asarray(idx, np.int64)
    for s in range(0, len(idx), bs):
        ctx = ctx_all[torch.as_tensor(idx[s:s + bs])].to(dev)
        a, b = head(ctx.float())
        lg.append(a.float().cpu().numpy())
        sd.append(b.float().cpu().numpy())
    return np.concatenate(lg).astype(np.float32), np.concatenate(sd).astype(np.float32)


def config_dict(n_updates: int) -> dict:
    return {"k_folds": K_FOLDS, "fold_rule": "int(sha256(log).hexdigest(),16) % 5", "n_updates": int(n_updates), "batch": BATCH, "lr_peak": LR_PEAK, "warmup": WARMUP,
            "schedule": "linear warm-up then cosine to 0 at n_updates (scorer_finetune.py:1331)", "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY, "grad_clip": CLIP,
            "w_aux": W_AUX, "w_sdf": W_SDF, "depth": DEPTH, "feed": "off", "train_precision": "bf16 autocast on cuda, fp32 master weights", "predict_precision": "fp32",
            "ctx_dtype": "float32 (the planner's fp32 ctx, A2.3)", "perm_seed": PERM_SEED, "perm_rule": "AUX.derangement(train tokens, PERM_SEED + fold)",
            "batch_gen_seed": "seed*100 + fold", "head_init_seed": "seed (AuxDrivableConfig.seed)", "seeds": {k: list(v) for k, v in SEEDS.items()}, "arms": ARMS,
            "bootstrap": f"paired log-cluster, {N_BOOT} resamples, seed {SEED_BOOT}, ASR.boot_weights"}


def config_sha(n_updates: int) -> str:
    return sha256_bytes(dump_json_bytes(config_dict(n_updates)))


def unit_path(root: str, arm: str, seed: int, fold: int) -> str:
    return os.path.join(root, "units", f"{arm}_s{seed}_f{fold}.npz")


def init_path(root: str, tap: str, seed: int) -> str:
    return os.path.join(root, "units", f"INIT_{tap}_s{seed}.npz")


def run_unit(U: Universe, labels: torch.Tensor, ctxs: dict, arm: str, seed: int, fold: int, *, n_updates: int, device, source: LabelCache, root: str, fh=None) -> dict:
    spec = ARMS[arm]
    tap = spec["tap"]
    ctx_all = select_ctx(tap, ctxs)
    guard_tap(ctx_all, tap, ctxs)
    train_idx, held_idx = make_fold_split(U.fold, fold)
    guard_split(train_idx, held_idx, U.logs, U.n)
    labels_train = labels
    if spec["perm"]:
        labels_train, src = permute_train_labels(labels, train_idx, fold)
        guard_label_perm(labels, labels_train, src, train_idx, held_idx)
    head, info = train_unit(ctx_all, labels_train, train_idx, held_idx, tap, seed, fold, n_updates=n_updates, device=device)
    lg_h, sd_h = predict_head(head, ctx_all, held_idx, device)
    lg_t, sd_t = predict_head(head, ctx_all, train_idx, device)
    keys_t = [U.keys[i] for i in train_idx]
    _res, st_t = pooled_p1(keys_t, lg_t, sd_t, source, n_boot=2)          # the in-sample read, on this fold model's own training tokens (TRUE labels, also for C)
    ins = stat_sums(st_t)
    meta = {"arm": arm, "seed": seed, "fold": fold, "tap": tap, "config_sha256": config_sha(n_updates), "info": info, "insample": ins, "n_held": int(len(held_idx)),
            "tool": TOOL}
    p = unit_path(root, arm, seed, fold)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp.npz"
    np.savez(tmp, held_idx=np.asarray(held_idx, np.int64), logit=lg_h, sdf=sd_h, meta_json=np.asarray(json.dumps(meta, sort_keys=True)))
    os.replace(tmp, p)
    log(f"unit {arm} s{seed} f{fold}: {info['wall_s']} s, bce_last100 {info['loss_bce_last100']:.4f}, in-sample IoU0-40 "
        f"{ins['inter_0-40'] / max(ins['union_0-40'], 1):.4f} (train-side, not a held-out read)", fh)
    return meta


def run_init(U: Universe, ctxs: dict, tap: str, seed: int, *, device, root: str, fh=None) -> None:
    """The untrained-head floor: the head at init (seed), on every token, fp32."""
    box = make_box(tap, seed, torch.device(device))
    lg, sd = predict_head(box.head, ctxs[tap], np.arange(U.n), device)
    p = init_path(root, tap, seed)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp.npz"
    np.savez(tmp, logit=lg, sdf=sd, meta_json=np.asarray(json.dumps({"tap": tap, "seed": seed, "tool": TOOL}, sort_keys=True)))
    os.replace(tmp, p)
    log(f"init floor {tap} s{seed} written", fh)


# =====================================================================================================================================================================
# assembly: pooled held-out predictions, the estimator, the mechanical outcome row
# =====================================================================================================================================================================
def load_unit(path: str):
    z = np.load(path, allow_pickle=False)
    return z["held_idx"], z["logit"], z["sdf"], json.loads(str(z["meta_json"]))


def seed_floor(vals) -> float | None:
    """max pairwise |IoU_i - IoU_j| over an arm's seeds on the full set (None when the arm has < 2 seeds)."""
    v = [x for x in vals if x is not None]
    return None if len(v) < 2 else float(max(abs(a - b) for i, a in enumerate(v) for b in v[i + 1:]))


def contrast(reps_a: np.ndarray, reps_b: np.ndarray, point_a: float, point_b: float, floors: dict) -> dict:
    """A - B, paired (the same log resamples): the percentile CI of the resample differences, the point difference and the 'separated beyond the floor' rule."""
    reps = reps_a - reps_b
    ci, n_nan = ASR._ci(reps)
    point = float(point_a - point_b)
    lo, hi = ci
    excludes_zero = bool(math.isfinite(lo) and math.isfinite(hi) and (lo > 0 or hi < 0))
    fl = {k: v for k, v in floors.items() if v is not None}
    beyond = bool(all(abs(point) > v for v in fl.values()))
    return {"point": point, "ci95": ci, "n_nan_resamples": n_nan, "ci_excludes_zero": excludes_zero, "floors_used": fl, "floors_missing": [k for k, v in floors.items() if v is None],
            "abs_point_exceeds_all_floors": beyond, "separated_beyond_floor": bool(excludes_zero and beyond), "direction": "positive" if point > 0 else "negative" if point < 0 else "zero"}


def decide(v_iou: float, s_iou: float, sv: dict) -> dict:
    """The committed outcome table, mechanically (K2 rule = the S1 one: the POINT ESTIMATE against 0.80)."""
    sep_pos = bool(sv["separated_beyond_floor"] and sv["point"] > 0)
    if v_iou >= K2_BAR:
        row = "ROW1_V_CARRIES"
    elif s_iou >= K2_BAR and sep_pos:
        row = "ROW2_S_CARRIES"
    elif s_iou < K2_BAR and sep_pos:
        row = "ROW3_S_BETTER_NOT_ENOUGH"
    else:
        row = "ROW4_SCENE_ELIMINATED"
    return {"row": row, "reading": ROWS[row][0], "next_lever": ROWS[row][1], "v_heldout_iou_0_40": v_iou, "s_heldout_iou_0_40": s_iou, "bar": K2_BAR,
            "s_minus_v_separated_positive_beyond_floor": sep_pos}


def insample_read(v_held: float, v_train: float) -> dict:
    """D1 / D2 on arm V (applies in every row; only meaningful when held-out < 0.80)."""
    if v_held >= K2_BAR:
        return {"read": "NOT_APPLICABLE_V_HELDOUT_CLEARS_K2", "v_heldout": v_held, "v_insample": v_train}
    if v_train >= K2_BAR:
        return {"read": "D1", "meaning": "train-fold IoU >= 0.80 while held-out < 0.80: generalisation / data-limited, so more label logs is a lever", "v_heldout": v_held, "v_insample": v_train}
    return {"read": "D2", "meaning": "train-fold IoU < 0.80: the head cannot fit these frozen features even in-sample, so the tap or the trunk is the lever", "v_heldout": v_held,
            "v_insample": v_train}


def assemble(root: str, U: Universe, source: LabelCache, n_updates: int, fh=None) -> dict:
    """Pool every arm/seed's held-out predictions (each token exactly once), read P1 / P2 through the readout, bootstrap the contrasts, apply the table."""
    guard_p1_function()
    cfg_sha = config_sha(n_updates)
    logs_sorted = sorted(set(U.logs.tolist()))
    W = ASR.boot_weights(len(logs_sorted), N_BOOT, SEED_BOOT)
    arms: dict = {}
    reps_seed: dict = {}
    for arm in ("V", "S", "C"):
        arms[arm] = {"seeds": {}}
        reps_seed[arm] = {}
        for seed in SEEDS[arm]:
            lg = np.full((U.n, ASR.NCX, ASR.NCY), np.nan, np.float32)
            sd = np.full_like(lg, np.nan)
            cover = np.zeros(U.n, np.int64)
            ins_f = []
            for f in range(K_FOLDS):
                held, a, b, meta = load_unit(unit_path(root, arm, seed, f))
                if meta["config_sha256"] != cfg_sha:
                    raise GuardFailed(f"unit {arm} s{seed} f{f}: config_sha256 {meta['config_sha256'][:12]} != {cfg_sha[:12]}")
                if sorted(held.tolist()) != sorted(np.flatnonzero(U.fold == f).tolist()):
                    raise GuardFailed(f"unit {arm} s{seed} f{f}: held-out indices are not fold {f}")
                lg[held], sd[held] = a, b
                cover[held] += 1
                ins_f.append(meta["insample"])
            if not (cover == 1).all():
                raise GuardFailed(f"{arm} s{seed}: tokens predicted {sorted(set(cover.tolist()))} times (every token exactly once)")
            res, st = pooled_p1(U.keys, lg, sd, source)
            if not st["valid"].all():
                raise GuardFailed("an unlabelled frame in the pooled read")
            ul, (inter, union) = ASR.by_log(U.logs, st[f"inter{K2_BAND_INDEX}"], st[f"union{K2_BAND_INDEX}"])
            if ul != logs_sorted:
                raise GuardFailed("log order differs from the bootstrap's")
            with np.errstate(invalid="ignore", divide="ignore"):
                reps_seed[arm][seed] = (W @ inter) / (W @ union)
            per_fold = {}
            for f in range(K_FOLDS):
                m = U.fold == f
                per_fold[str(f)] = float(st[f"inter{K2_BAND_INDEX}"][m].sum() / max(st[f"union{K2_BAND_INDEX}"][m].sum(), 1.0))
            ins_iou = [i["inter_0-40"] / max(i["union_0-40"], 1.0) for i in ins_f]
            arms[arm]["seeds"][str(seed)] = {
                "heldout_p1": res["p1"]["all"], "heldout_p2": res["p2"], "k2": res["k2"], "heldout_iou_0_40": res["p1"]["all"]["0-40"]["est"],
                "heldout_iou_0_40_ci95_logs_only": res["p1"]["all"]["0-40"]["ci95"], "per_fold_heldout_iou_0_40": per_fold,
                "insample_iou_0_40_per_fold": ins_iou, "insample_iou_0_40_mean": float(np.mean(ins_iou)),
                "insample_iou_all_cells_mean": float(np.mean([i["inter_all"] / max(i["union_all"], 1.0) for i in ins_f])),
                "insample_sdf_mae_m_mean": float(np.mean([i["all_abs"] / max(i["all_n"], 1.0) for i in ins_f])),
                "train": [load_unit(unit_path(root, arm, seed, f))[3]["info"] for f in range(K_FOLDS)]}
        ious = [arms[arm]["seeds"][str(s)]["heldout_iou_0_40"] for s in SEEDS[arm]]
        arms[arm]["mean_seed_heldout_iou_0_40"] = float(np.mean(ious))
        arms[arm]["seed_ious"] = ious
        arms[arm]["seed_floor"] = seed_floor(ious)
        arms[arm]["mean_seed_insample_iou_0_40"] = float(np.mean([arms[arm]["seeds"][str(s)]["insample_iou_0_40_mean"] for s in SEEDS[arm]]))
        arms[arm]["mean_seed_heldout_overall_sdf_mae_m"] = float(np.mean([arms[arm]["seeds"][str(s)]["heldout_p2"]["overall_mae_m"]["est"] for s in SEEDS[arm]]))
        arms[arm]["mean_seed_heldout_near_boundary_sdf_mae_m"] = float(np.mean([arms[arm]["seeds"][str(s)]["heldout_p2"]["near_boundary_mae_m"]["est"] for s in SEEDS[arm]]))
        mean_reps = np.mean([reps_seed[arm][s] for s in SEEDS[arm]], axis=0)
        arms[arm]["mean_seed_heldout_iou_0_40_ci95_bootstrap"] = ASR._ci(mean_reps)[0]
        arms[arm]["_reps"] = mean_reps
    floors = {a: arms[a]["seed_floor"] for a in arms}
    contrasts = {}
    for name, (a, b) in {"S-V": ("S", "V"), "V-C": ("V", "C"), "S-C": ("S", "C")}.items():
        contrasts[name] = contrast(arms[a]["_reps"], arms[b]["_reps"], arms[a]["mean_seed_heldout_iou_0_40"], arms[b]["mean_seed_heldout_iou_0_40"], {a: floors[a], b: floors[b]})
    for a in arms:
        arms[a].pop("_reps")
    # the untrained-head floor
    init = {}
    for tap in ("visual", "scene"):
        init[tap] = {}
        for seed in (0, 1, 2):
            z = np.load(init_path(root, tap, seed), allow_pickle=False)
            res, _ = pooled_p1(U.keys, z["logit"], z["sdf"], source, n_boot=N_BOOT)
            init[tap][str(seed)] = {"iou_0_40": res["p1"]["all"]["0-40"]["est"], "ci95_logs_only": res["p1"]["all"]["0-40"]["ci95"], "p2": res["p2"]}
        init[tap]["mean_seed_iou_0_40"] = float(np.mean([init[tap][str(s)]["iou_0_40"] for s in (0, 1, 2)]))
    dec = decide(arms["V"]["mean_seed_heldout_iou_0_40"], arms["S"]["mean_seed_heldout_iou_0_40"], contrasts["S-V"])
    ins = insample_read(arms["V"]["mean_seed_heldout_iou_0_40"], arms["V"]["mean_seed_insample_iou_0_40"])
    prior = {"v_minus_c_separated_beyond_floor": contrasts["V-C"]["separated_beyond_floor"], "prior_dominated": not contrasts["V-C"]["separated_beyond_floor"],
             "note": "V-C is the features' contribution over the label marginal; C has ONE seed, so only V's seed floor is applied to V-C and S-C (floors_missing lists C)"}
    return {"tool": TOOL, "tier": TIER, "n_tokens": U.n, "n_logs": len(logs_sorted), "n_updates": int(n_updates), "config": config_dict(n_updates), "config_sha256": cfg_sha,
            "estimator": f"paired log-cluster bootstrap over {len(logs_sorted)} navtest logs, {N_BOOT} resamples, seed {SEED_BOOT} (ASR.boot_weights): per resample the pooled held-out "
                         f"IoU [0,40) per arm per seed, mean over seeds, then S-V / V-C / S-C. Answers 'another draw of LOGS' plus 'another TRAINING SEED' (the seed floor); NOT "
                         f"'another training corpus'. The 5 fold models are held fixed in the resamples.",
            "arms": arms, "contrasts": contrasts, "init_floor": init, "decision": dec, "insample_read": ins, "prior_control": prior}


def write_result(out: str, result: dict) -> None:
    with open(os.path.join(out, "PROBE_RESULT.json"), "wb") as f:
        f.write(dump_json_bytes(result))


# =====================================================================================================================================================================
# data loading
# =====================================================================================================================================================================
def free_ram_gb() -> float:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30
    except Exception:
        return float("nan")


def load_ctx(tap: str, tokens, fh=None) -> torch.Tensor:
    """[N, P, 256] float32 in token order, read once from the fp32 cache (no precision change)."""
    d, P = CTX_DIRS[tap], CTX_LEN[tap]
    arr = torch.empty((len(tokens), P, CTX_DIM), dtype=torch.float32)
    t0 = time.time()
    for i, t in enumerate(tokens):
        a = np.load(os.path.join(d, f"ctx_{t}.npy"))
        if a.shape != (P, CTX_DIM) or a.dtype != np.float32:
            raise GuardFailed(f"{tap} ctx for {t}: {a.dtype} {a.shape}, expected float32 {(P, CTX_DIM)}")
        arr[i] = torch.from_numpy(a)
        if i and i % 200 == 0:
            log(f"loaded {tap} ctx {i}/{len(tokens)} ({time.time() - t0:.0f} s)", fh)
    return arr


def load_true_labels(U: Universe):
    src = ASR.open_labels("navtest")                         # pinned store: sha256 + key count are checked by AuxLabelSource
    lab, valid = src.labels(U.keys)
    if not bool(valid.all()):
        raise GuardFailed(f"{int((~valid).sum())} of {U.n} tokens have no navtest label")
    return lab, LabelCache(U.keys, lab)


# =====================================================================================================================================================================
# the GPU lease (the shared dev-box lock of aux_navtest_forward.py, held in bounded stretches)
# =====================================================================================================================================================================
class Lease:
    def __init__(self, enabled: bool, max_hold_s: float = 1200.0, max_wait_s: float = 48 * 3600.0, fh=None):
        self.enabled, self.max_hold_s, self.max_wait_s, self.fh = enabled, max_hold_s, max_wait_s, fh
        self.body, self.t0, self.ANF = None, 0.0, None

    def ensure(self) -> None:
        if not self.enabled:
            return
        if self.body is not None and time.time() - self.t0 <= self.max_hold_s:
            return
        self.release()
        if self.ANF is None:
            import aux_navtest_forward as ANF
            self.ANF = ANF
        waited = time.time()
        while True:
            try:
                self.body = self.ANF.acquire_gpu_lock(wait_timeout_s=240.0)
                self.t0 = time.time()
                log(f"GPU lock taken after {time.time() - waited:.0f} s", self.fh)
                return
            except self.ANF.GpuBusy as e:
                if time.time() - waited > self.max_wait_s:
                    raise RuntimeError(f"GPU busy for {self.max_wait_s:.0f} s: {e}")
                log(f"GPU busy ({str(e)[:120]}); retry in 90 s", self.fh)
                time.sleep(90)

    def release(self) -> None:
        if self.enabled and self.body is not None:
            self.ANF.release_gpu_lock(self.ANF.LOCK, self.body)
            self.body = None


# =====================================================================================================================================================================
# commands
# =====================================================================================================================================================================
def cmd_folds(a) -> int:
    U = load_universe()
    doc = json.load(open(TOKENS_JSON, encoding="utf-8"))
    fd = folds_doc(U.tokens, doc["token_log"])
    b = dump_json_bytes(fd)
    out = os.path.join(a.out_dir or HERE, "folds.json")
    with open(out, "wb") as f:
        f.write(b)
    miss = [t for t in U.tokens if not os.path.exists(os.path.join(CTX_DIRS["visual"], f"ctx_{t}.npy"))]
    lab, _ = load_true_labels(U)
    print(json.dumps({"folds_json": out, "folds_sha256": sha256_bytes(b), "n_tokens": U.n, "n_logs": fd["n_logs"], "tokens_per_fold": fd["tokens_per_fold"],
                      "logs_per_fold": fd["logs_per_fold"], "missing_visual_ctx": len(miss), "n_labelled": int(lab.shape[0]),
                      "label_store_sha256": ASR.LABEL_PINS["navtest"][0], "label_store_keys": ASR.LABEL_PINS["navtest"][1]}, indent=1))
    return 0


def cmd_selfcheck(a) -> int:
    U = load_universe()
    guard_p1_function()
    for f in range(K_FOLDS):
        tr, ho = make_fold_split(U.fold, f)
        guard_split(tr, ho, U.logs, U.n)
    lab, src = load_true_labels(U)
    tr, ho = make_fold_split(U.fold, 0)
    new, s = permute_train_labels(lab, tr, 0)
    guard_label_perm(lab, new, s, tr, ho)
    print("SELFCHECK_OK", U.n, "tokens", len(set(U.logs.tolist())), "logs")
    return 0


def cmd_smoke(a) -> int:
    """Timing only.  ~20 updates of arm V (real cached ctx of a few tokens) and arm S (synthetic [64,256] ctx), NO metric is computed."""
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    os.makedirs(a.out, exist_ok=True)
    U = load_universe()
    lab, _src = load_true_labels(U)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(U.n, a.n_tokens, replace=False))
    t0 = time.time()
    sub_vis = torch.empty((len(pick), CTX_LEN["visual"], CTX_DIM), dtype=torch.float32)
    for j, i in enumerate(pick):
        sub_vis[j] = torch.from_numpy(np.load(os.path.join(CTX_DIRS["visual"], f"ctx_{U.tokens[i]}.npy")))
    read_s = time.time() - t0
    mean_file_s = read_s / len(pick)
    sub_scene = torch.randn((len(pick), CTX_LEN["scene"], CTX_DIM))
    sub_lab = lab[torch.as_tensor(pick)]
    held = np.arange(0, len(pick) // 4)
    train = np.arange(len(pick) // 4, len(pick))
    res = {"device": dev, "n_tokens": len(pick), "ctx_read_s_per_file": mean_file_s, "free_ram_gb": free_ram_gb(), "updates": a.updates}
    lease = Lease(enabled=(dev == "cuda" and not a.no_gpu_lock))
    try:
        lease.ensure()
        for name, ctx in (("V_visual_real_ctx", sub_vis), ("S_scene_SYNTHETIC_ctx", sub_scene)):
            tap = "visual" if name.startswith("V") else "scene"
            train_unit(ctx, sub_lab, train, held, tap, 0, 0, n_updates=5, device=dev)                  # warm-up (cuda context, kernels): not timed
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _h, info = train_unit(ctx, sub_lab, train, held, tap, 0, 0, n_updates=a.updates, device=dev)
            t1 = time.time()
            predict_head(_h, ctx, np.arange(len(pick)), dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            info["predict_s_per_token"] = (time.time() - t1) / len(pick)
            info["s_per_update"] = info["wall_s"] / a.updates
            res[name] = info
            log(f"smoke {name}: {info['wall_s']} s for {a.updates} updates = {info['s_per_update']:.4f} s/update; predict {info['predict_s_per_token']:.4f} s/token; "
                f"peak alloc {info['peak_alloc_gb']} GB; finite loss bce {info['loss_bce_mean_all']:.4f}")
    finally:
        lease.release()
    with open(os.path.join(a.out, "smoke_timing.json"), "wb") as f:
        f.write(dump_json_bytes(res))
    return 0


def cmd_run(a) -> int:
    out = a.out
    work = a.work or out                                    # unit files (per-fold held-out predictions) live in `work`; markers, probe.log and PROBE_RESULT.json in `out`
    os.makedirs(os.path.join(work, "units"), exist_ok=True)
    fh = open(os.path.join(out, "probe.log"), "a", encoding="utf-8")
    try:
        for mk in ("PROBE_DONE", "PROBE_FAILED"):
            if os.path.exists(os.path.join(out, mk)):
                os.remove(os.path.join(out, mk))
        U = load_universe()
        n_updates = int(a.updates)
        log(f"{TOOL}: {U.n} tokens, {len(set(U.logs.tolist()))} logs, updates {n_updates}, config {config_sha(n_updates)[:12]}", fh)
        guard_p1_function()
        for f in range(K_FOLDS):
            tr, ho = make_fold_split(U.fold, f)
            guard_split(tr, ho, U.logs, U.n)
        folds_b = dump_json_bytes(folds_doc(U.tokens, json.load(open(TOKENS_JSON, encoding="utf-8"))["token_log"]))
        want = a.folds_sha256
        if want and sha256_bytes(folds_b) != want:
            raise GuardFailed(f"folds.json sha256 {sha256_bytes(folds_b)} != the pre-registered {want}")
        for tap in ("visual", "scene"):
            miss = [t for t in U.tokens if not os.path.exists(os.path.join(CTX_DIRS[tap], f"ctx_{t}.npy"))]
            if miss:
                raise RuntimeError(f"{len(miss)} {tap} ctx files are missing from {CTX_DIRS[tap]} (first {miss[0]})")
        lab, source = load_true_labels(U)
        t_ram = time.time()
        while free_ram_gb() < 11.0 and time.time() - t_ram < 2 * 3600:
            log(f"free RAM {free_ram_gb():.1f} GB < 11 GB; waiting", fh)
            time.sleep(120)
        log(f"free RAM {free_ram_gb():.1f} GB; loading ctx", fh)
        ctxs = {"visual": load_ctx("visual", U.tokens, fh), "scene": load_ctx("scene", U.tokens, fh)}
        log(f"ctx loaded; free RAM {free_ram_gb():.1f} GB", fh)
        dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
        lease = Lease(enabled=(dev == "cuda" and not a.no_gpu_lock), max_hold_s=a.lease_s, fh=fh)
        try:
            for tap in ("visual", "scene"):
                for seed in (0, 1, 2):
                    if not os.path.exists(init_path(work, tap, seed)):
                        lease.ensure()
                        run_init(U, ctxs, tap, seed, device=dev, root=work, fh=fh)
            for (arm, seed, fold) in UNIT_ORDER:
                p = unit_path(work, arm, seed, fold)
                if os.path.exists(p):
                    if load_unit(p)[3]["config_sha256"] == config_sha(n_updates):
                        continue
                    raise GuardFailed(f"{p} exists with another config; refusing to mix")
                lease.ensure()
                run_unit(U, lab, ctxs, arm, seed, fold, n_updates=n_updates, device=dev, source=source, root=work, fh=fh)
        finally:
            lease.release()
        log("grid complete; assembling", fh)
        result = assemble(work, U, source, n_updates, fh)
        result["folds_sha256"] = sha256_bytes(folds_b)
        mp = os.path.join(CTX_DIRS["scene"], "scene_ctx_manifest.json")
        result["scene_ctx_manifest_sha256"] = sha256_file(mp) if os.path.exists(mp) else None
        sp = os.path.join(CTX_DIRS["scene"], "extract_summary.json")
        result["scene_ctx_extract_summary"] = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else None
        result["unit_files_dir"] = os.path.join(work, "units")
        write_result(out, result)
        with open(os.path.join(out, "PROBE_DONE"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"row": result["decision"]["row"], "v": result["decision"]["v_heldout_iou_0_40"], "s": result["decision"]["s_heldout_iou_0_40"],
                                "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        log(f"PROBE_DONE {result['decision']['row']}", fh)
        return 0
    except BaseException as e:                                                       # noqa: BLE001  (the marker must exist whatever killed the run)
        tb = traceback.format_exc()
        log(f"PROBE_FAILED {type(e).__name__}: {e}", fh)
        with open(os.path.join(out, "PROBE_FAILED"), "w", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {type(e).__name__}: {e}\n\n{tb}\n")
        return 1


def cmd_report(a) -> int:
    U = load_universe()
    lab, source = load_true_labels(U)
    result = assemble(a.work or a.out, U, source, int(a.updates))
    write_result(a.out, result)
    print(json.dumps(result["decision"], indent=1))
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("folds")
    p.add_argument("--out-dir", default=None)
    sub.add_parser("selfcheck")
    p = sub.add_parser("smoke")
    p.add_argument("--out", required=True)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--n-tokens", type=int, default=48)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-gpu-lock", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--updates", type=int, default=N_UPDATES)
    p.add_argument("--work", default=None, help="directory for units/ (default: --out)")
    p.add_argument("--folds-sha256", default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-gpu-lock", action="store_true")
    p.add_argument("--lease-s", type=float, default=1200.0)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--work", default=None)
    p.add_argument("--updates", type=int, default=N_UPDATES)
    a = ap.parse_args(argv)
    return {"folds": cmd_folds, "selfcheck": cmd_selfcheck, "smoke": cmd_smoke, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
