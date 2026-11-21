"""SH5-DAC-PROX: input prep (CPU), the LOCKSTEP multi-model trainer on cached `visual_ctx` (GPU, behind the dev-box lock), and the analysis (CPU).

    python sh5_train_eval.py prep    --out-dir D                      # CPU; builds D/sh5_inputs.npz (label-free) then D/sh5_labels.npz (opens the label arrays)
    python sh5_train_eval.py fit     --data D --ctx ctx.npy --seed 0 --stage A1 --out-dir R     # GPU (sh5_gpulock); never run by the build agent
    python sh5_train_eval.py analyze --data D --fits R --out-dir R                              # CPU

WHY LOCKSTEP.  ctx is 3.75 MiB per token (10.2 GiB for 2,782 tokens) on an HDD that reads ~24 MiB/s and an 8 GB GPU shared with the desktop.  Reading the ctx once per model per epoch would be
>100 GB of IO per model.  The trainer therefore streams a SHARD of tokens to the GPU once and trains EVERY learner (all folds x arms of the stage) on it.  A learner's optimizer step is taken per chunk
of `tokens_per_step` shard tokens restricted to the tokens OUTSIDE its held-out fold (so 8 x 0.8 tokens per step on average).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_lib as L                                                                                    # noqa: E402

PRIVGEOM = "C:/Users/Admin/par18/privgeom"
CENSUS_META = "C:/Users/Admin/par18/fastneg/run/inputs/meta.npz"
SETS_DIR = "C:/Users/Admin/srttc_s0/work/sets"
ARM_ORDER = ("S", "Snoprox", "Sshuf", "plant", "plantweak", "nullprox", "nulllabel")
STAGES = {                    # which learners a stage builds (prereg s9); 5 folds each
    "A1": [("S", None), ("Snoprox", None), ("Sshuf", None), ("plant", None)],
    "A2": [("plantweak", None)],
    "B": "null",              # nullprox reps r with r % 2 == seed; nulllabel reps r with r % 2 == seed
    "PRE": [("plant", None)],  # the plant-only pre-flight (one fold with --folds 0)
}


def now() -> str:
    import datetime
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# ======================================================================================================================================================================================
# prep (CPU).  Part 1 is LABEL-FREE: keys, folds, strata, P2's logits, the key-time points, the clearance targets (geometry, not the label).  Part 2 opens the label arrays.
# ======================================================================================================================================================================================
def build_inputs(out_path: str, pg: str = PRIVGEOM, meta: str = CENSUS_META, sets_dir: str = SETS_DIR, plan: str = None) -> dict:
    FL, PK = L.refs()
    z = np.load(f"{pg}/run/join/arrays.npz", allow_pickle=True)
    m = np.load(meta, allow_pickle=False)
    idx = z["idx"]
    key = [(str(m["key_log"][i]), str(m["key_token"][i]), int(m["key_step"][i]), int(m["key_rank"][i]), int(m["key_ckpt_step"][i])) for i in idx]
    N = len(idx)
    traj = np.zeros((N, 64, 20, 3), np.float32)
    by_log: dict = {}
    for n, k in enumerate(key):
        by_log.setdefault(k[0], []).append(n)
    for log, rows in by_log.items():
        st = FL.read_members(f"{sets_dir}/{log}.npz", ["token", "step", "rank", "ckpt_step", "traj3"])
        where = {(str(t), int(s), int(r), int(c)): i for i, (t, s, r, c) in enumerate(zip(st["token"], st["step"], st["rank"], st["ckpt_step"]))}
        for n in rows:
            _, tok, s, r, c = key[n]
            traj[n] = st["traj3"][where[(tok, s, r, c)]]
    toks = sorted({k[1] for k in key})
    trow = {t: i for i, t in enumerate(toks)}
    token_idx = np.array([trow[k[1]] for k in key], np.int64)
    tok_log = {}
    for k in key:
        tok_log[k[1]] = k[0]
    pts = L.keypoints(traj)
    g = z["g"]
    D = {"idx": idx, "token": np.array(toks), "token_log": np.array([tok_log[t] for t in toks]), "token_idx": token_idx, "fold": z["fold"], "log_idx": z["log_idx"], "code": z["code"],
         "z6": z["z6"], "ep_s0": z["ep_s0"], "ep_s1": z["ep_s1"], "pts": pts, "tgt": L.clearance_targets(g), "traj3": traj,
         "key_log": np.array([k[0] for k in key]), "key_token": np.array([k[1] for k in key]), "key_step": np.array([k[2] for k in key]), "key_rank": np.array([k[3] for k in key]),
         "key_ckpt_step": np.array([k[4] for k in key]), "log_names": z["log_names"]}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **D)
    return {"n_sets": N, "n_tokens": len(toks), "n_logs": int(len(set(tok_log.values()))), "out": out_path, "sha256": L.sha256_file(out_path)}


def attach_labels(inputs_path: str, out_path: str, pg: str = PRIVGEOM) -> dict:
    """Opens the label arrays (privgeom join/labels_sub.npy = the census labels6 of the retained sets).  Run ONLY after the prereg is sealed."""
    lab = np.load(f"{pg}/run/join/labels_sub.npy", allow_pickle=False)
    n = len(np.load(inputs_path, allow_pickle=True)["idx"])
    if lab.shape[0] != n:
        raise L.Sh5Error(f"labels {lab.shape} do not match {n} sets")
    g = np.load(f"{pg}/run/join/arrays.npz", allow_pickle=True)["g"]
    np.savez(out_path, lab=lab, sc_key=L.signed_key_clearance(g))         # the SIGNED key clearances are the label's own test: they live with the labels and feed ONLY the planted controls
    return {"out": out_path, "sha256": L.sha256_file(out_path), "shape": list(lab.shape)}


def load_data(inputs_path: str, labels_path: str = None, tokens_keep: np.ndarray = None) -> dict:
    """-> D with arrays; if `tokens_keep` (bool [T] over the inputs' tokens) is given, the analysis set is restricted to sets whose token is kept and everything is re-indexed
    (tokens 0..T'-1, logs 0..n-1 contiguous).  The census fold of each log is kept."""
    z = np.load(inputs_path, allow_pickle=True)
    D = {k: z[k] for k in z.files}
    if labels_path:
        zl = np.load(labels_path, allow_pickle=False)
        D["lab"], D["sc_key"] = zl["lab"], zl["sc_key"]
    if tokens_keep is not None:
        keep = tokens_keep[D["token_idx"]]
        remap = np.cumsum(tokens_keep) - 1
        D["token"], D["token_log"] = D["token"][tokens_keep], D["token_log"][tokens_keep]
        for k in ("idx", "fold", "log_idx", "code", "z6", "ep_s0", "ep_s1", "pts", "tgt", "sc_key", "traj3", "key_log", "key_token", "key_step", "key_rank", "key_ckpt_step", "token_idx", "lab"):
            if k in D:
                D[k] = D[k][keep]
        D["token_idx"] = remap[D["token_idx"]]
        u, loc = np.unique(D["log_idx"], return_inverse=True)
        D["log_idx"] = loc
        D["log_names_kept"] = D["log_names"][u] if len(D["log_names"]) > len(u) else D["log_names"]
    D["n_logs"] = int(D["log_idx"].max()) + 1
    D["N"] = len(D["fold"])
    D["T"] = len(D["token"])
    D["tok_fold"] = np.zeros(D["T"], np.int64)
    for t in range(D["T"]):
        D["tok_fold"][t] = D["fold"][np.flatnonzero(D["token_idx"] == t)[0]]
    D["tok_sets"] = [np.flatnonzero(D["token_idx"] == t) for t in range(D["T"])]
    return D


# ======================================================================================================================================================================================
# ctx providers
# ======================================================================================================================================================================================
class MemCtx:
    """In-memory ctx [T,P,D] (tests)."""

    def __init__(self, arr, device="cpu"):
        import torch
        self.t = torch.as_tensor(arr)
        self.device = device
        self.P, self.Dc = self.t.shape[1], self.t.shape[2]
        self.bytes_per_token = self.P * self.Dc * 2

    def load(self, rows):
        return self.t[list(rows)].to(self.device)


class NpyCtx:
    """The cached `visual_ctx` file ([T,7680,256] uint16 = bf16 bits, np.load(mmap_mode='r')).  `rows` = the ctx-file row of each analysis token (ctx_rows[token])."""

    def __init__(self, path, ctx_rows, device="cuda"):
        self.mm = np.load(path, mmap_mode="r")
        if self.mm.dtype != np.uint16:
            raise L.Sh5Error(f"{path}: dtype {self.mm.dtype}, want uint16 (bf16 bits)")
        self.rows = np.asarray(ctx_rows, np.int64)
        self.device = device
        self.P, self.Dc = self.mm.shape[1], self.mm.shape[2]
        self.bytes_per_token = self.P * self.Dc * 2

    def load(self, tokens):
        """-> bf16 [n, P, Dc] ON THE DEVICE.  Preallocated and filled one token at a time (Amendment 1): the sealed version built a list of the whole shard on the host and then `torch.stack`ed it, a
        host-RAM peak of TWO shards (~5.2 GiB for the 2.6 GiB default) on a box whose RAM gate is 3 GB."""
        import torch
        tokens = list(tokens)
        out = torch.empty((len(tokens), self.P, self.Dc), dtype=torch.bfloat16, device=self.device)
        for i, t in enumerate(tokens):
            raw = np.array(self.mm[self.rows[t]], dtype=np.uint16)
            out[i] = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16).to(self.device)
        return out


# ======================================================================================================================================================================================
# the learners and the lockstep trainer
# ======================================================================================================================================================================================
@dataclasses.dataclass
class Spec:
    arm: str
    fold: int
    seed: int
    rep: int = 0

    @property
    def name(self) -> str:
        return f"{self.arm}{'' if self.arm not in ('nullprox', 'nulllabel') else '_r' + str(self.rep)}_f{self.fold}_s{self.seed}"

    @property
    def group(self) -> str:
        return f"{self.arm}{'' if self.arm not in ('nullprox', 'nulllabel') else '_r' + str(self.rep)}_s{self.seed}"


def arm_w_reg(arm: str) -> float:
    return 0.0 if arm == "Snoprox" else L.W_REG


def arm_arrays(D: dict, arm: str, rep: int):
    """-> (y [N,64], tgt [N,64,10]) the TRAINING arrays of an arm.  nullprox: clearance targets permuted within token (prereg s8 C-NULL).  nulllabel: ALSO the DAC label permuted within set
    (a noise-trained branch: the machinery null).  Evaluation and the stacker always use the REAL label."""
    y = D["lab"][..., L.DAC_COL].astype(np.float32)
    tgt = D["tgt"]
    if arm == "nullprox":
        tgt = L.permute_within_token(tgt, D["token_idx"], L.NULL_SEED_BASE + L.NULL_SALT * rep)
    elif arm == "nulllabel":
        tgt = L.permute_within_token(tgt, D["token_idx"], L.NULL_SEED_BASE + L.NULL_SALT * rep)
        y = L.permute_within_set(y, L.NULL_SEED_BASE + 31 + L.NULL_SALT * rep)
    return y, tgt


def plant_rows(D: dict, sets: np.ndarray, arm: str, token_row: int, d_ctx: int, n_slots: int):
    """the planted block of ONE token (all its sets, all 64 proposals) -> float32 [n_entries, d_ctx]"""
    import sh5_branch as B
    sc = D["sc_key"][sets].astype(np.float64)
    if arm == "plantweak":
        sc = sc + L.plant_noise(str(D["token"][token_row]), sc.shape, L.PLANT_SIGMA_M)
    return B.plant_block(D["pts"][sets], sc, d_ctx, n_slots)


class Learner:
    def __init__(self, spec: Spec, cfg, D: dict, device: str):
        import torch
        import sh5_branch as B
        self.spec, self.device = spec, device
        torch.manual_seed(spec.seed * 1_000_003 + spec.fold * 101 + ARM_ORDER.index(spec.arm) * 7 + spec.rep * 13)
        self.model = B.SH5Branch(cfg).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=L.LR, weight_decay=L.WD, betas=L.BETAS, eps=L.EPS)
        self.step = 0
        self.w_reg = arm_w_reg(spec.arm)
        self.y, self.tgt = arm_arrays(D, spec.arm, spec.rep) if "lab" in D else (None, None)
        self.rng = np.random.default_rng(spec.seed * 7919 + spec.fold * 31 + ARM_ORDER.index(spec.arm) * 5 + spec.rep + 1)
        self.plant = spec.arm in ("plant", "plantweak")
        self.last_loss = float("nan")


@dataclasses.dataclass
class RunCfg:
    epochs_total: int = L.EPOCHS_TOTAL
    cycles: int = L.CYCLES_SHARDED
    inner: int = L.INNER_SHARDED
    p_sub: int = L.P_SUB
    tokens_per_step: int = L.TOKENS_PER_STEP
    shard_gib: float = 2.6
    device: str = "cuda"
    seed: int = 0
    min_avail_gb: float = L.MIN_AVAIL_GB_FIT
    autocast: bool = True
    epochs_override: int = None          # 0 = the C-ID pre-flight: no step is taken


class Trainer:
    def __init__(self, D: dict, specs: list, rc: RunCfg, provider, cfg=None, log=print, ram_guard=None):
        import sh5_branch as B
        self.D, self.rc, self.provider, self.log = D, rc, provider, log
        self.cfg = cfg or B.SH5Cfg(d_ctx=provider.Dc)
        self.ram_guard = ram_guard or (lambda: None)
        self.learners = [Learner(s, self.cfg, D, rc.device) for s in specs]
        tok = provider.bytes_per_token
        self.shard_tokens = max(1, int(rc.shard_gib * 2 ** 30 // tok))
        self.n_shards = math.ceil(D["T"] / self.shard_tokens)
        if self.n_shards == 1:
            self.cycles, self.inner = 1, rc.epochs_total
        else:
            self.cycles, self.inner = rc.cycles, rc.inner
        if rc.epochs_override is not None:
            self.cycles, self.inner = 1, rc.epochs_override
        self.t_start = time.time()
        self.n_steps_logged = 0

    # ---- a training step of one learner on a chunk ----------------------------------------------------------------------------------------------------------------------------
    def _ac(self):
        import torch
        dev = "cuda" if str(self.rc.device).startswith("cuda") else "cpu"
        return torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=bool(self.rc.autocast))      # cuda + autocast: bf16 (the real run); cpu + autocast: the bf16 EMULATION the tests use; autocast off: a no-op

    def _ctx_source(self, ln: Learner, t: int, ctx_pos: dict) -> int:
        """the token whose ctx the learner reads for token t.  Sshuf (the ctx-SHUFFLED arm, an ADDITION of s8): the NEXT token of the same fold that is resident in the current shard (cyclic, in token
        id order) - the branch then sees another scene's ctx with this token's proposals.  Every other arm reads its own token."""
        if ln.spec.arm != "Sshuf":
            return t
        same = sorted(u for u in ctx_pos if self.D["tok_fold"][u] == self.D["tok_fold"][t])
        if len(same) < 2:
            raise L.Sh5Error(f"Sshuf needs >= 2 resident tokens of fold {self.D['tok_fold'][t]} in the shard")
        return same[(same.index(t) + 1) % len(same)]

    def _batch(self, ln: Learner, toks: list, ctx_pos: dict, shard_ctx, train: bool):
        """-> pts [S,Pp,9,3], set_to_tok [S], ctx [n,P,Dc], y, tgt, sets"""
        import torch
        D = self.D
        sets = np.concatenate([D["tok_sets"][t] for t in toks])
        s2t = np.concatenate([np.full(len(D["tok_sets"][t]), i, np.int64) for i, t in enumerate(toks)])
        src = [self._ctx_source(ln, t, ctx_pos) for t in toks]
        ctx = shard_ctx[torch.as_tensor([ctx_pos[t] for t in src], device=shard_ctx.device)]
        if ln.plant:
            ctx = ctx.clone()
            for i, t in enumerate(toks):
                blk = plant_rows(D, D["tok_sets"][t], ln.spec.arm, t, ctx.shape[-1], ctx.shape[1])
                ctx[i, ctx.shape[1] - len(blk):] = torch.as_tensor(blk, dtype=ctx.dtype, device=ctx.device)
        pts = D["pts"][sets]
        if train:
            sel = np.argsort(ln.rng.random((len(sets), 64)), axis=1)[:, :self.rc.p_sub]
            r = np.arange(len(sets))[:, None]
            pts = pts[r, sel]
            y, tgt = ln.y[sets][r, sel], ln.tgt[sets][r, sel]
        else:
            y, tgt = None, None
        dev = self.rc.device
        f = lambda a: torch.as_tensor(a, device=dev)                                      # noqa: E731
        return f(pts), f(s2t), ctx, (None if y is None else f(y)), (None if tgt is None else f(tgt)), sets

    def _train_step(self, ln: Learner, toks: list, ctx_pos: dict, shard_ctx):
        import torch
        import sh5_branch as B
        pts, s2t, ctx, y, tgt, _ = self._batch(ln, toks, ctx_pos, shard_ctx, True)
        for g in ln.opt.param_groups:
            g["lr"] = L.LR * min(1.0, (ln.step + 1) / L.WARMUP)
        ln.model.train()
        with self._ac():
            kv = ln.model.kv(ctx)
            logit, reg = ln.model(pts, kv, s2t)
        lt = B.loss_terms(logit, reg, y, tgt, ln.w_reg)
        ln.opt.zero_grad(set_to_none=True)
        lt["total"].backward()
        ln.opt.step()
        ln.step += 1
        ln.last_loss = float(lt["total"].detach())

    @staticmethod
    def _sync(dev):
        import torch
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()

    # ---- the loop -------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _shards(self, cycle: int) -> list:
        order = np.random.default_rng(self.rc.seed * 100003 + cycle).permutation(self.D["T"])
        return [order[i * self.shard_tokens:(i + 1) * self.shard_tokens] for i in range(self.n_shards)]

    def train(self, ckpt_fn=None, start_cycle: int = 0):
        import torch
        D, rc = self.D, self.rc
        for cycle in range(start_cycle, self.cycles):
            for sh, toks_sh in enumerate(self._shards(cycle)):
                if self.inner == 0:
                    continue                          # C-ID pre-flight: no step is taken, so nothing is loaded
                self.ram_guard()
                t0 = time.time()
                shard_ctx = self.provider.load(toks_sh)
                self._sync(rc.device)
                ctx_pos = {int(t): i for i, t in enumerate(toks_sh)}
                self.log(f"  cycle {cycle + 1}/{self.cycles} shard {sh + 1}/{self.n_shards}: {len(toks_sh)} tokens loaded in {time.time() - t0:.0f} s")
                for ep in range(self.inner):
                    order = np.random.default_rng(rc.seed * 7_000_003 + cycle * 1009 + sh * 31 + ep).permutation(toks_sh)
                    for c0 in range(0, len(order), rc.tokens_per_step):
                        chunk = [int(t) for t in order[c0:c0 + rc.tokens_per_step]]
                        for ln in self.learners:
                            toks = [t for t in chunk if D["tok_fold"][t] != ln.spec.fold]
                            if not toks:
                                continue
                            self._train_step(ln, toks, ctx_pos, shard_ctx)
                        self.n_steps_logged += 1
                        if self.n_steps_logged == 20:
                            self._sync(rc.device)
                            dt = (time.time() - self.t_start) / 20
                            tot = self.cycles * self.inner * math.ceil(D["T"] / rc.tokens_per_step)
                            self.log(f"ZZTHROUGHPUT {dt:.2f} s/chunk-step (all {len(self.learners)} learners incl. the first shard load); projected training {dt * tot / 3600:.2f} h ESTIMATED")
                    self.log(f"    inner {ep + 1}/{self.inner} done; mean last loss {np.nanmean([ln.last_loss for ln in self.learners]):.4f}")
                del shard_ctx
                if str(rc.device).startswith("cuda"):
                    torch.cuda.empty_cache()
            if ckpt_fn is not None:
                ckpt_fn(cycle)

    # ---- held-out prediction: one more pass over the ctx --------------------------------------------------------------------------------------------------------------------------
    def predict(self) -> dict:
        """-> {group: {"z": [N,64] float32, "reg": [N,64,10] float16, "n_cover": [N] int}} with every set predicted by the learner of ITS fold."""
        import torch
        D, rc = self.D, self.rc
        out = {}
        for ln in self.learners:
            g = out.setdefault(ln.spec.group, {"z": np.full((D["N"], 64), np.nan, np.float32), "reg": np.zeros((D["N"], 64, L.N_REG), np.float16), "n_cover": np.zeros(D["N"], np.int64)})
            ln.model.eval()
        order = np.random.default_rng(rc.seed + 424242).permutation(D["T"])
        for sh in range(self.n_shards):
            toks_sh = order[sh * self.shard_tokens:(sh + 1) * self.shard_tokens]
            self.ram_guard()
            shard_ctx = self.provider.load(toks_sh)
            ctx_pos = {int(t): i for i, t in enumerate(toks_sh)}
            for t in toks_sh:
                t = int(t)
                for ln in self.learners:
                    if D["tok_fold"][t] != ln.spec.fold:
                        continue
                    pts, s2t, ctx, _, _, sets = self._batch(ln, [t], ctx_pos, shard_ctx, False)
                    with torch.no_grad(), self._ac():
                        kv = ln.model.kv(ctx)
                        logit, reg = ln.model(pts, kv, s2t)
                    g = out[ln.spec.group]
                    g["z"][sets] = logit.float().cpu().numpy()
                    g["reg"][sets] = reg.float().cpu().numpy().astype(np.float16)
                    g["n_cover"][sets] += 1
            del shard_ctx
        return out

    # ---- G-LIVE on the first real chunk -------------------------------------------------------------------------------------------------------------------------------------------
    def glive(self) -> dict:
        import torch
        import sh5_branch as B
        D, rc = self.D, self.rc
        toks = [int(t) for t in np.random.default_rng(rc.seed + 99).permutation(D["T"])[:rc.tokens_per_step]]
        shard_ctx = self.provider.load(toks)
        ctx_pos = {t: i for i, t in enumerate(toks)}
        res = {}
        for w in sorted({ln.w_reg for ln in self.learners}):
            ln = next(l for l in self.learners if l.w_reg == w)
            model = B.SH5Branch(self.cfg).to(rc.device)

            def make_batch(ln=ln):
                pts, s2t, ctx, y, tgt, _ = self._batch(ln, toks, ctx_pos, shard_ctx, True)
                return pts, s2t, ctx, y, tgt
            res[f"w_reg={w}"] = B.glive(model, make_batch, w, ac=self._ac)             # Amendment 1: the probe runs under the TRAINER's autocast (bf16 ctx)
        return {"ok": all(v["ok"] for v in res.values()), "by_w_reg": res}


# ======================================================================================================================================================================================
# assembling OOF arrays and the analysis
# ======================================================================================================================================================================================
def save_oof(out_dir: str, groups: dict, meta: dict, allow_partial: bool = False) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for g, v in groups.items():
        bad = (v["n_cover"] > 1).any() if allow_partial else (v["n_cover"] != 1).any()
        if bad:
            raise L.Sh5Error(f"group {g}: {int((v['n_cover'] != 1).sum())} sets are not covered exactly once by a held-out learner")
        p = f"{out_dir}/OOF_{g}.npz"
        np.savez(p, z=v["z"], reg=v["reg"])
        jdump({**meta, "group": g, "z_sha256": L.sha256_file(p)}, f"{out_dir}/OOF_{g}.json")
        paths.append(p)
    return paths


def _fl_pairs(z: np.ndarray, lab_dac: np.ndarray, code: np.ndarray) -> float:
    FL, _ = L.refs()
    st = FL.pair_stats(z, lab_dac == 0.0, lab_dac == 1.0, code, names=("ALLQ",))
    n, d = st["ALLQ"]
    return float(n.sum() / d.sum()) if d.sum() > 0 else float("nan")


def r2_reg(reg: np.ndarray, tgt: np.ndarray) -> float:
    """held-out R^2 of the 9 clearance outputs pooled (1 - SSE / SST about the per-output mean)."""
    r, t = reg.astype(np.float64).reshape(-1, reg.shape[-1]), tgt.astype(np.float64).reshape(-1, tgt.shape[-1])
    sst = ((t - t.mean(0)) ** 2).sum()
    return float(1.0 - ((r - t) ** 2).sum() / sst) if sst > 0 else float("nan")


def analyze(D: dict, fits_dir: str, gates: dict, out_dir: str, B: int = L.BOOT_B, log=print) -> dict:
    """The registered analysis (prereg s6 - s8).  `gates` = {"c_id": bool, "g_live": bool, "c_fid": bool, "red_table": bool} read from the gate files by main()."""
    FL, PK = L.refs()
    lab_dac = D["lab"][..., L.DAC_COL]
    an = L.Analyzer(lab_dac, D["code"], D["log_idx"], D["n_logs"], B=B)
    stats = PK.Stats(D["log_idx"], an.draws, FL, n_logs=D["n_logs"])
    st_cache: dict = {}

    def st(key, z):
        if key not in st_cache:
            st_cache[key] = an.stats(z)
        return st_cache[key]

    zL = D["z6"][..., L.DAC_COL]
    stL = st("L", zL)
    R = {"schema": L.SCHEMA, "created": now(), "n_sets": int(D["N"]), "n_tokens": int(D["T"]), "n_logs": int(D["n_logs"]), "gates": gates, "seeds": {}, "auc_L": {"F": an.auc(stL, "Q4"), "Q123": an.auc(stL, "Q123")}}
    oof = lambda g: np.load(f"{fits_dir}/OOF_{g}.npz")                                      # noqa: E731
    have = lambda g: os.path.exists(f"{fits_dir}/OOF_{g}.npz")                              # noqa: E731
    stacked: dict = {}

    def stack(g):
        if g not in stacked:
            zs = oof(g)["z"]
            stacked[g] = (zs, *L.oof_stack(zL, zs, lab_dac, D["fold"]))
        return stacked[g]

    calls, pg_seed = [], []
    for s in L.SEEDS:
        r = {}
        zS, zLS, par = stack(f"S_s{s}")
        stS, stLS = st(f"S_s{s}", zS), st(f"LS_s{s}", zLS)
        r["auc"] = {"S_alone": {"F": an.auc(stS, "Q4"), "Q123": an.auc(stS, "Q123")}, "L+S": {"F": an.auc(stLS, "Q4"), "Q123": an.auc(stLS, "Q123")}}
        r["delta_F"], r["delta_Q123"] = an.diff(stLS, stL, "Q4"), an.diff(stLS, stL, "Q123")
        r["S_alone_minus_L_F"] = an.diff(stS, stL, "Q4")
        r["stacker"] = [{"a": a, "b": b, "ok": ok} for a, b, ok in par]
        r["call"] = L.seed_call(r["delta_F"])
        calls.append(r["call"])
        if have(f"Snoprox_s{s}"):
            zN, zLN, parN = stack(f"Snoprox_s{s}")
            stN, stLN = st(f"Snoprox_s{s}", zN), st(f"LSn_s{s}", zLN)
            r["auc"]["Snoprox_alone"] = {"F": an.auc(stN, "Q4")}
            r["auc"]["L+Snoprox"] = {"F": an.auc(stLN, "Q4"), "Q123": an.auc(stLN, "Q123")}
            r["delta_F_noprox"] = an.diff(stLN, stL, "Q4")
            r["attribution_LS_minus_LSnoprox_F"] = an.diff(stLS, stLN, "Q4")
        if have(f"Sshuf_s{s}"):
            zH, zLH, _ = stack(f"Sshuf_s{s}")
            stH, stLH = st(f"Sshuf_s{s}", zH), st(f"LSh_s{s}", zLH)
            r["auc"]["Sshuf_alone"] = {"F": an.auc(stH, "Q4")}
            r["auc"]["L+Sshuf"] = {"F": an.auc(stLH, "Q4"), "Q123": an.auc(stLH, "Q123")}
            r["delta_F_shuf"] = an.diff(stLH, stL, "Q4")
            r["call_shuf"] = L.seed_call(r["delta_F_shuf"])
            r["ctx_specific_LS_minus_LSshuf_F"] = an.diff(stLS, stLH, "Q4")
        # ---- pick level (P2's own rule; secondary: the EP-led composite)
        pl = L.pick_level(D["z6"], D["lab"], zLS, D[f"ep_s{s}"], stats)
        r["pick_level"] = pl
        r["pick_gain_seed"] = L.pick_gain_seed(pl["rule[S]"]["vs_P2"]["d_pdms_x100"], pl["rule[S]"]["vs_P2"]["d_dac0"])
        pg_seed.append(r["pick_gain_seed"])
        r["reliability_at_picks"] = {"P2_row_at_P2_picks": L.reliability_at_picks(D["z6"], zL, lab_dac, stats, D["code"]),
                                    "LS_row_at_rule[S]_picks": L.reliability_at_picks(np.where(np.arange(6) == L.DAC_COL, zLS[..., None], D["z6"]), zLS, lab_dac, stats, D["code"])}
        # ---- C-REACH / C-POWER
        if have(f"plant_s{s}"):
            zP, zLP, _ = stack(f"plant_s{s}")
            r["c_reach"] = {"S_plant_alone_F": an.auc(st(f"plant_s{s}", zP), "Q4"), "R2_reg": r2_reg(oof(f"plant_s{s}")["reg"], D["tgt"]), "delta_F_LS_plant": an.diff(st(f"LSplant_s{s}", zLP), stL, "Q4")}
            r["c_reach"]["ok"] = bool(r["c_reach"]["S_plant_alone_F"]["lo"] >= L.REACH_MIN_AUC and r["c_reach"]["R2_reg"] >= L.REACH_MIN_R2)
        if have(f"plantweak_s{s}"):
            zW, zLW, _ = stack(f"plantweak_s{s}")
            r["c_power"] = {"delta_F_LS_plantweak": an.diff(st(f"LSplantweak_s{s}", zLW), stL, "Q4")}
        R["seeds"][f"s{s}"] = r
    R["raw_call"] = L.reach_call(calls)
    R["pick_gain"] = L.pick_gain_call(pg_seed)
    # ---- C-REACH, C-POWER, the nulls
    creach = [R["seeds"][f"s{s}"].get("c_reach", {}).get("ok") for s in L.SEEDS]
    R["c_reach_ok"] = bool(all(c is True for c in creach))
    pw = [R["seeds"][f"s{s}"].get("c_power", {}).get("delta_F_LS_plantweak") for s in L.SEEDS]
    R["c_power_ok"] = bool(all(p is not None for p in pw) and L.power_ok(pw))
    nulls = {"nullprox": [], "nulllabel": []}
    for arm in nulls:
        for rep in range(L.NULL_REPS if arm == "nullprox" else L.NULL_LABEL_REPS):
            s = rep % 2
            g = f"{arm}_r{rep}_s{s}"
            if not have(g):
                continue
            zq, zLq, _ = stack(g)
            stq, stLq = st(g, zq), st("LS" + g, zLq)
            row = {"rep": rep, "seed": s, "delta_F": an.diff(stLq, stL, "Q4")}
            if have(f"Snoprox_s{s}"):
                row["minus_LSnoprox_F"] = an.diff(stLq, st(f"LSn_s{s}", stack(f"Snoprox_s{s}")[1]), "Q4")
            row["fires"] = L.null_label_fires([row["delta_F"]])
            nulls[arm].append(row)
    R["nulls"] = nulls
    R["null_label_void"] = L.null_label_void([r["fires"] for r in nulls["nulllabel"]])
    R["null_complete"] = {"nullprox": len(nulls["nullprox"]) == L.NULL_REPS, "nulllabel": len(nulls["nulllabel"]) == L.NULL_LABEL_REPS}
    attrib = {}
    for s in L.SEEDS:
        a = R["seeds"][f"s{s}"].get("attribution_LS_minus_LSnoprox_F")
        if a is not None and nulls["nullprox"]:
            nv = [row["minus_LSnoprox_F"]["point"] for row in nulls["nullprox"] if "minus_LSnoprox_F" in row]
            attrib[f"s{s}"] = {"real": a, "null_points": nv, "n_null_ge_real": int(sum(v >= a["point"] for v in nv)), "p_emp": (1 + sum(v >= a["point"] for v in nv)) / (1 + len(nv)),
                               "p_resolution": 1 / (1 + len(nv))}
    R["attribution_vs_empirical_null"] = attrib
    gates_ok = all(bool(gates.get(k)) for k in ("c_id", "g_live", "c_fid", "red_table"))
    R["shuf_gain"] = bool(all(R["seeds"][f"s{s}"].get("call_shuf") == "REACH" for s in L.SEEDS))
    R["final_outcome"] = L.final_outcome(R["raw_call"], gates_ok, R["c_reach_ok"], R["null_label_void"], R["c_power_ok"], R["shuf_gain"])
    R["next_arm"] = L.next_arm(R["final_outcome"], R["pick_gain"]) if R["final_outcome"] in ("CTX-REACHABLE", "NOT-REACHABLE") else "NAMED-BY-REPORT"
    os.makedirs(out_dir, exist_ok=True)
    jdump(R, f"{out_dir}/results.json")
    return R


def fmt(ci: dict, k: int = 4) -> str:
    return f"{ci['point']:+.{k}f} [{ci['lo']:+.{k}f}, {ci['hi']:+.{k}f}]"


def write_tables(R: dict, path: str) -> None:
    ln = [f"# SH5-DAC-PROX tables (generated {R['created']}; {R['n_sets']} sets, {R['n_tokens']} tokens, {R['n_logs']} logs; paired log-cluster bootstrap)", "",
          f"**final: {R['final_outcome']}** (raw {R['raw_call']}; pick gain {R['pick_gain']}; C-REACH {R['c_reach_ok']}; C-POWER {R['c_power_ok']}; label-null void {R['null_label_void']}; gates {R['gates']}); next arm: {R['next_arm']}", ""]
    ln += ["| seed | AUC_F L | AUC_F S alone | AUC_F L+S | Delta_F (L+S - L) | call | Delta_Q1-3 | dPDMS x100 (rule[S] vs P2) | dDAC-unsafe |", "|---|---|---|---|---|---|---|---|---|"]
    for s in L.SEEDS:
        r = R["seeds"][f"s{s}"]
        pl = r["pick_level"]["rule[S]"]["vs_P2"]
        ln.append(f"| {s} | {R['auc_L']['F']['point']:.4f} | {r['auc']['S_alone']['F']['point']:.4f} | {r['auc']['L+S']['F']['point']:.4f} | {fmt(r['delta_F'])} | {r['call']} | {fmt(r['delta_Q123'])} | {fmt(pl['d_pdms_x100'], 3)} | {fmt(pl['d_dac0'])} |")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(ln) + "\n")


# ======================================================================================================================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--labels", action="store_true", help="also open the label arrays (only after the prereg is sealed)")
    f = sp.add_parser("fit")
    f.add_argument("--data", required=True, help="dir with sh5_inputs.npz + sh5_labels.npz")
    f.add_argument("--ctx", required=True)
    f.add_argument("--ctx-manifest", required=True, help="ctx_manifest.json of sh5_ctx_build.py (tokens in file order + fidelity)")
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--stage", required=True, choices=sorted(STAGES))
    f.add_argument("--folds", default="0,1,2,3,4")
    f.add_argument("--out-dir", required=True)
    f.add_argument("--epochs", type=int, default=None, help="0 = the C-ID pre-flight (no optimizer step)")
    f.add_argument("--shard-gib", type=float, default=2.6)
    f.add_argument("--max-wait", type=float, default=21600.0)
    f.add_argument("--min-avail-gb", type=float, default=L.MIN_AVAIL_GB_FIT)
    a = sp.add_parser("analyze")
    a.add_argument("--data", required=True)
    a.add_argument("--fits", required=True)
    a.add_argument("--gates", required=True, help="json {c_id, g_live, c_fid, red_table}")
    a.add_argument("--keep-tokens", default=None, help="the ctx manifest json (its done tokens are the analysis set)")
    a.add_argument("--out-dir", required=True)
    x = ap.parse_args(argv)
    import sh5_guard as G
    if x.cmd == "prep":
        if x.labels:
            G.require_sealed()                                     # the label arrays are opened only after the prereg is sealed
        rep = build_inputs(f"{x.out_dir}/sh5_inputs.npz")
        print(json.dumps(rep, indent=1))
        if x.labels:
            print(json.dumps(attach_labels(f"{x.out_dir}/sh5_inputs.npz", f"{x.out_dir}/sh5_labels.npz"), indent=1))
        return 0
    if x.cmd == "analyze":
        G.require_sealed()
        keep = None
        if x.keep_tokens:
            allt = np.load(f"{x.data}/sh5_inputs.npz", allow_pickle=True)["token"]
            keep, _ = ctx_keep(allt, x.keep_tokens)                 # the SAME analysis set as `fit`: the tokens whose ctx was built
        D = load_data(f"{x.data}/sh5_inputs.npz", f"{x.data}/sh5_labels.npz", keep)
        R = analyze(D, x.fits, json.load(open(x.gates, encoding="utf-8")), x.out_dir)
        write_tables(R, f"{x.out_dir}/RESULT_tables.md")
        print(R["final_outcome"], R["raw_call"], R["next_arm"])
        return 0
    return fit_main(x)


def ctx_keep(all_tokens, manifest_path: str):
    """-> (keep bool [len(all_tokens)], row_of {token: ctx-file row}).  The analysis set = the tokens whose ctx was BUILT (the manifest's `done` flags).  `fit` and `analyze` both read it from here (Amendment 1:
    `analyze` used the manifest's whole token list while `fit` used the done ones, so a partial build would have made them disagree on the analysis set)."""
    man = json.load(open(manifest_path, encoding="utf-8"))
    toks = [str(t) for t in man["tokens"]]
    done = np.load(man["done_file"]) if man.get("done_file") else np.ones(len(toks), bool)
    row_of = {t: i for i, t in enumerate(toks) if done[i]}
    return np.array([str(t) in row_of for t in all_tokens]), row_of


def fit_main(x) -> int:
    import sh5_gpulock as GL
    import sh5_guard as G
    G.require_sealed()
    allt = np.load(f"{x.data}/sh5_inputs.npz", allow_pickle=True)["token"]
    keep, row_of = ctx_keep(allt, x.ctx_manifest)
    D = load_data(f"{x.data}/sh5_inputs.npz", f"{x.data}/sh5_labels.npz", keep)
    ctx_rows = [row_of[str(t)] for t in D["token"]]
    folds = [int(v) for v in x.folds.split(",")]
    specs = []
    st = STAGES[x.stage]
    if st == "null":
        for rep in range(L.NULL_REPS):
            if rep % 2 == x.seed:
                specs += [Spec("nullprox", k, x.seed, rep) for k in folds]
        for rep in range(L.NULL_LABEL_REPS):
            if rep % 2 == x.seed:
                specs += [Spec("nulllabel", k, x.seed, rep) for k in folds]
    else:
        for arm, _ in st:
            specs += [Spec(arm, k, x.seed) for k in folds]
    print(f"ZZSH5FIT stage {x.stage} seed {x.seed}: {len(specs)} learners, {D['N']} sets, {D['T']} tokens, {D['n_logs']} logs; groups {sorted({s.group for s in specs})}", flush=True)
    rc = RunCfg(seed=x.seed, shard_gib=x.shard_gib, min_avail_gb=x.min_avail_gb, epochs_override=x.epochs)
    return GL.run_locked_inprocess(lambda: _fit_locked(D, ctx_rows, specs, rc, x), max_wait=x.max_wait, min_avail_gb=x.min_avail_gb, owner="sh5_dacprox", job=f"sh5-fit-{x.stage}-s{x.seed}")


def _fit_locked(D, ctx_rows, specs, rc, x) -> int:
    import torch
    import psutil
    log = lambda m: print(m, flush=True)                                                   # noqa: E731
    import sh5_gpulock as GL
    GL.require_locked(rc.device)
    provider = NpyCtx(x.ctx, ctx_rows, rc.device)

    def ram_guard():
        while psutil.virtual_memory().available / 2 ** 30 < 3.0:
            log("ZZRAMWAIT available RAM < 3 GB; pausing 20 s")
            time.sleep(20)
    tr = Trainer(D, specs, rc, provider, log=log, ram_guard=ram_guard)
    gl = tr.glive()
    jdump(gl, f"{x.out_dir}/g_live_{x.stage}_s{x.seed}.json")
    log(f"ZZGLIVE ok={gl['ok']}")
    if not gl["ok"]:
        log("ZZSH5FIT_FAIL G-LIVE failed: no optimizer step is taken")
        return 3
    tr.train(ckpt_fn=lambda c: log(f"  cycle {c + 1} done"))
    groups = tr.predict()
    partial = len({s.fold for s in specs}) < L.FOLDS
    paths = save_oof(x.out_dir, groups, {"stage": x.stage, "seed": x.seed, "epochs": rc.epochs_override if rc.epochs_override is not None else rc.epochs_total, "p_sub": rc.p_sub, "created": now(),
                                         "n_sets": D["N"], "partial_folds": sorted({s.fold for s in specs}) if partial else None}, allow_partial=partial)
    if partial:                                      # the plant-only PRE-FLIGHT: the DAC AUC and the regression R^2 on the covered sets only
        rep = {}
        for g_, v in groups.items():
            cov = v["n_cover"] == 1
            lab = D["lab"][..., L.DAC_COL][cov]
            st = _fl_pairs(v["z"][cov], lab, D["code"][cov])
            rep[g_] = {"n_sets": int(cov.sum()), "auc_all_strata": st, "r2_reg": r2_reg(v["reg"][cov], D["tgt"][cov])}
        jdump(rep, f"{x.out_dir}/preflight_{x.stage}_s{x.seed}.json")
        log(f"ZZPREFLIGHT {json.dumps(rep)}")
    log(f"ZZSH5FIT_OK {len(paths)} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
