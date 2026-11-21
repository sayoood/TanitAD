#!/usr/bin/env python3
"""Measure 5 EFFECTIVENESS (eval/PREREG_MEASURE5.md §3c-§5, blob c668b5b6): scorer-only fine-tune arms on the cached
snapshot-015 context, and the pre-registered readout on W3's 200 navtest tokens.

WHY SCORER-ONLY IS THE LIVE PATH: the scorer is detached from the trajectories AND the visual context
(model.py:746, :751), so the score loss reaches only score_q_mlp / score_dec / score_head -- the three modules trained
here, on the context the trunk produced. Declared simplifications: the cached trunk is snapshot 015's (the live trunk
keeps moving); the optimiser starts with fresh AdamW moments; on CPU the scorer trains in fp32 (the live run uses bf16
autocast; the CPU has no bf16 units -- MEASURED 3.0 s/step fp32 vs 56 s/step emulated bf16).

  data     training sets per sample from the labelled v4 lines (C2: 8 x 0.75x copies + STOP, --slow-frac 1.0), their
           side files and the props chunks: PURE = the 64 originals with their labels (exactly the v3 labeller's --
           proven by eval/validate_slow_labels_v4.py ref/reverify), SERVED = the v4 set; components by the trainer's own
           ScorerBank.components + its NAVSIM drivable-area override (train.py:318-346, :457-460)
  train    per seed, the arms in LOCKSTEP over one batch order (the same samples, the same order; only the label
           content differs): V3 (every set pure), V4 (served iff the labeller's own carries() at frac 0.5, else pure),
           V4-all (served always; reported). AdamW lr = the live lr at snapshot 015's step (read from the pod), wd 0.01,
           score weight 0.1, batch 16, 4 epochs, loss = train.py:1339-1342.
  eval     A0 (snapshot 015 unchanged) + every trained scorer on the navtest cache: the 64 alone (the shipped route),
           the 64 + 9 extras MASKED (the 64 keep their pure-set scores; an extra attends to the 64 and itself --
           stop_candidate_probe.score_masked) and PLAIN; the extras are snapshot 015's own top-8 at 0.75x (harness
           f075_r00..r07) + STOP (harness stopzeros), identical for every arm
  readout  E1 (primary), E3a, E3b, E2, E4; per-token V4 - V3 of seed means; paired log-cluster bootstrap over the 93
           logs (10,000, seed 20260927 -- eval/snapshot_pair_under_rule.py's draws, reproduced and self-tested); the seed
           floor; the pre-registered decision. -> eval/raw/m5_effectiveness/readout.json

    python eval/m5_finetune_eval.py data
    python eval/m5_finetune_eval.py train --seed 0 [--device cuda] [--arms V3,V4,V4all]
    python eval/m5_finetune_eval.py eval
    python eval/m5_finetune_eval.py readout
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REFE = os.path.join(os.path.dirname(HERE), "refe")
sys.path.insert(0, HERE)
sys.path.insert(0, REFE)

M5 = "D:/Projects/TanitAD/data/refe_m5"
CACHE = "D:/Projects/TanitAD/data/refe_trunk_cache/d7c59f4f"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"
NAV = "D:/Projects/TanitAD/data/refe_navtest"
TABLE = f"{NAV}/proptable/sub200_ep015/table.npz"
RANKS = f"{NAV}/proptable/sub200_ep015/slow_copies/ranks.npz"
BUILD = f"{NAV}/proptable/sub200_ep015/slow_copies/build.npz"
STOP_CSV = f"{NAV}/score/refe_sub200_ep015_stopzeros/refe_sub200_ep015_stopzeros.csv"
TOK = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
       "A1_sub200_tokens.json")
OUT = os.path.join(HERE, "raw", "m5_effectiveness")
LR = 1.0322678309765115e-04          # MEASURED: the live metrics.jsonl lr at step 4933 (snapshot 015), pod 2026-09-27
WD, SCORE_W, BATCH, EPOCHS = 0.01, 0.1, 16, 4
SEED = 20260927
ARMS = ("V3", "V4", "V4all")
N_SRC, FACTOR = 8, 0.75
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance")



def past(deadline: str) -> bool:
    """Local HH:MM reached? Clock times before 12:00 count as 'after midnight tonight' on BOTH sides, so a window that
    crosses midnight compares correctly (a plain string compare says "23:30" >= "04:28")."""
    def m(h, mi):
        v = h * 60 + mi
        return v + 1440 if v < 720 else v
    hh, mm = (int(x) for x in deadline.split(":"))
    lt = time.localtime()
    return m(lt.tm_hour, lt.tm_min) >= m(hh, mm)


def vram_cap(device) -> None:
    """M5_VRAM_CAP_MIB (set by eval/m5_gpu_chain.py): cap this process's torch allocator, so an overrun raises an OOM
    instead of the Windows sysmem fallback silently spilling into system RAM (MEASURED on this box 2026-09-27)."""
    cap = os.environ.get("M5_VRAM_CAP_MIB")
    if device.type == "cuda" and cap:
        tot = torch.cuda.get_device_properties(0).total_memory
        frac = min(1.0, max(0.05, float(cap) * 2 ** 20 / tot))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        print(f"  VRAM cap {float(cap):.0f} MiB = fraction {frac:.3f} of {tot / 2 ** 20:.0f} MiB", flush=True)


def reserve_vram(device) -> None:
    """M5_VRAM_RESERVE_MIB: take the stage's VRAM from the caching allocator at PROCESS START and keep it reserved
    (allocate, then free -- the caching allocator holds the segment), so every other job's start gate sees this
    stage's memory from its first second. MEASURED 2026-09-28 01:02-01:08: without it, a 301 s CPU preload looked like
    an idle GPU, another job's gate admitted an arm, and the total crossed the 7,300 MiB ceiling once the steps began."""
    mib = os.environ.get("M5_VRAM_RESERVE_MIB")
    if device.type == "cuda" and mib:
        x = torch.empty(int(float(mib) * 2 ** 20), dtype=torch.uint8, device=device)
        del x
        print(f"  VRAM reserved {float(mib):.0f} MiB (allocator reserved {torch.cuda.memory_reserved() / 2 ** 20:.0f} MiB)",
              flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def trainer_targets(t: dict) -> list:
    """What train.OnPolicyBank feeds the loss for one target dict (cross-checked against it in the validity run)."""
    import train as TR
    c = TR.ScorerBank.components(t)
    if isinstance(t, dict) and "navsim_dac.violation" in t:
        c[1] = 1.0 - min(max(float(t["navsim_dac.violation"]), 0.0), 1.0)
    return c


# ------------------------------------------------------------------------------------------ the scorer alone
class Scorer(nn.Module):
    """REFe's scoring branch, the SAME submodules (built by REFe.__init__, loaded from snapshot 015's partial state)."""

    def __init__(self, score_q_mlp, score_dec, score_head):
        super().__init__()
        self.score_q_mlp, self.score_dec, self.score_head = score_q_mlp, score_dec, score_head

    def forward(self, trajs, sctx):                       # == REFe.score_trajectories (model.py:672-682)
        s = self.score_q_mlp(trajs.flatten(2))
        for blk in self.score_dec:
            s = blk(s, sctx)
        return self.score_head(s)

    def masked(self, trajs, sctx, mask):                 # == stop_candidate_probe.score_masked
        s = self.score_q_mlp(trajs.flatten(2))
        for blk in self.score_dec:
            h = blk.n1(s)
            s = s + blk.self_attn(h, h, h, attn_mask=mask, need_weights=False)[0]
            h = blk.n2(s)
            s = s + blk.cross(h, sctx, sctx, need_weights=False)[0]
            s = s + blk.mlp(blk.n3(s))
        return self.score_head(s)


def load_scorer() -> Scorer:
    from model import REFe, REFeConfig
    cfg = REFeConfig.for_backbone("vitl16")
    full = REFe(cfg)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_partial"]
    want = {k: v for k, v in sd.items() if k.startswith(("score_q_mlp.", "score_dec.", "score_head."))}
    sc = Scorer(full.score_q_mlp, full.score_dec, full.score_head)
    sc.load_state_dict(want, strict=True)                  # strict: every scorer key present, none extra
    del full
    n_keys = sum(1 for k in sc.state_dict())
    if len(want) != n_keys:
        raise RuntimeError(f"scorer keys in the snapshot {len(want)} != the module's {n_keys}")
    return sc.eval()


# ------------------------------------------------------------------------------------------ cache access
class Cache:
    """key -> (shard, row) over <CACHE>/<split>/shard_*/ ; visual_ctx loaded by row (memmap)."""

    def __init__(self, split: str):
        self.dir = os.path.join(CACHE, split)
        self.shards, self.where, self.tags = [], {}, {}
        for s in sorted(d for d in os.listdir(self.dir) if d.startswith("shard_") and not d.endswith(".tmp")):
            p = os.path.join(self.dir, s)
            m = json.load(open(os.path.join(p, "meta.json"), encoding="utf-8"))
            if not (m.get("gate") or {}).get("pass"):
                raise RuntimeError(f"{p}: shard gate did not pass -- refusing to use it")
            keys = [str(k) for k in np.load(os.path.join(p, "key.npy"))]
            self.shards.append({"dir": p, "keys": keys, "tags": m["tags"],
                                "vc": np.load(os.path.join(p, "visual_ctx.npy"), mmap_mode="r")})
            for r, k in enumerate(keys):
                self.where[k] = (len(self.shards) - 1, r)

    def preload(self) -> None:
        """Every row of visual_ctx into RAM once (the arms x seeds x epochs then never touch the disk again)."""
        for sh in self.shards:
            sh["vc"] = np.array(sh["vc"], copy=True)

    def ctx(self, keys, device, dtype):
        out = []
        for k in keys:
            si, r = self.where[k]
            sh = self.shards[si]
            a = torch.from_numpy(np.array(sh["vc"][r], copy=True))
            a = a.view(torch.bfloat16) if sh["tags"]["visual_ctx"] == "bf16bits" else a
            out.append(a)
        return torch.stack(out).to(device=device, dtype=dtype)

    def array(self, name: str) -> np.ndarray:
        return np.concatenate([np.load(os.path.join(s["dir"], f"{name}.npy")) for s in self.shards])

    def keys(self):
        return [k for s in self.shards for k in s["keys"]]


# ------------------------------------------------------------------------------------------ data
def lines(pattern: str, kind: str) -> dict:
    out = {}
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == kind:
                    out[f"{r['log_name']}|{r['token']}|{int(r['step'])}|{int(r['rank'])}"] = r
    return out


def data(a) -> int:
    import onpolicy_label_v4 as L4
    sets = lines(os.path.join(a.m5, "sets", "onpolicy_r0_*.jsonl"), "onpolicy_set")
    aux = lines(os.path.join(a.m5, "sets", "slowaux_r0_*.jsonl"), "onpolicy_slowaux")
    props = lines(os.path.join(a.m5, "queue", "props_r0_*.jsonl*"), "onpolicy_props")
    spec_half = L4.SlowSpec(factors=(FACTOR,), frac=0.5)
    res = {}
    for split in ("train", "val"):
        cdir = os.path.join(CACHE, f"navtrain_{split}")
        if not os.path.isdir(cdir):
            res[split] = "no cache"
            continue
        C = Cache(f"navtrain_{split}")
        keys, P, TG, S, SG, car = [], [], [], [], [], []
        miss = {"no_set": 0, "no_aux": 0, "no_props": 0, "bad_spec": 0, "kept_mismatch": 0}
        for k in C.keys():
            r, x, pr = sets.get(k), aux.get(k), props.get(k)
            if r is None:
                miss["no_set"] += 1          # skipped by the labeller (ndiff < 3) or not labelled yet
                continue
            if x is None or pr is None:
                miss["no_aux" if x is None else "no_props"] += 1
                continue
            sl = r.get("slow") or {}
            if not sl.get("carries") or sorted(set(sl["factor"])) != [0.0, FACTOR] or len(sl["slots"]) != N_SRC + 1:
                miss["bad_spec"] += 1
                continue
            served = np.concatenate([np.asarray(r["traj"], np.float32), np.asarray(r["yaw"], np.float32)[..., None]], -1)
            # the v3 line's own rounding (onpolicy_label.worker: round(x, 5), round(y, 5), round(yaw, 6)), in PYTHON
            v3 = np.asarray([[[round(float(q[0]), 5), round(float(q[1]), 5), round(float(q[2]), 6)] for q in pk]
                             for pk in pr["props"]], np.float32)
            kept = [j for j in range(64) if j not in set(sl["slots"])]
            if not np.array_equal(served[kept], v3[kept]):
                miss["kept_mismatch"] += 1
                continue
            stg = [trainer_targets(t) for t in r["targets"]]
            ptg = list(stg)
            for slot, t in zip(x["slots"], x["dropped_targets"]):
                ptg[slot] = trainer_targets(t)
            keys.append(k)
            P.append(v3)
            TG.append(np.asarray(ptg, np.float32))
            S.append(served)
            SG.append(np.asarray(stg, np.float32))
            lg, tk, st, rk = k.split("|")
            car.append(L4.carries((lg, tk, int(st), int(rk)), spec_half))
        np.savez(os.path.join(a.m5, f"ft_data_{split}.npz"), key=np.array(keys), pure_traj=np.stack(P),
                 pure_tg=np.stack(TG), served_traj=np.stack(S), served_tg=np.stack(SG), carries=np.array(car))
        res[split] = {"cached": len(C.keys()), "usable": len(keys), "carrying_at_frac_0.5": int(np.sum(car)),
                      "excluded": miss}
        print(f"  {split}: {res[split]}", flush=True)
    json.dump(res, open(os.path.join(a.m5, "ft_data.json"), "w"), indent=1)
    print("ZZM5_DATA_OK")
    return 0


# ------------------------------------------------------------------------------------------ train
def arm_sets(arm: str, D: dict, idx: np.ndarray):
    if arm == "V3":
        return D["pure_traj"][idx], D["pure_tg"][idx]
    use = np.ones(len(idx), bool) if arm == "V4all" else D["carries"][idx]
    tr = np.where(use[:, None, None, None], D["served_traj"][idx], D["pure_traj"][idx])
    tg = np.where(use[:, None, None], D["served_tg"][idx], D["pure_tg"][idx])
    return tr, tg


def val_bce(model, D, C, device, dtype, amp) -> dict:
    """navtrain-val BCE per component on the PURE sets and on the COPY slots of the served sets (reported)."""
    if D is None:
        return {}
    keys = [str(k) for k in D["key"]]
    tot = {"pure": np.zeros(6), "copies": np.zeros(6)}
    n = {"pure": 0, "copies": 0}
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(keys), BATCH):
            idx = np.arange(b0, min(len(keys), b0 + BATCH))
            ctx = C.ctx([keys[i] for i in idx], device, dtype)
            for name, tr, tg in (("pure", D["pure_traj"][idx], D["pure_tg"][idx]),
                                 ("copies", D["served_traj"][idx], D["served_tg"][idx])):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    sx = model(torch.from_numpy(tr).to(device, dtype), ctx)
                bce = F.binary_cross_entropy_with_logits(sx.float(), torch.from_numpy(tg).to(device),
                                                         reduction="none").cpu().numpy()          # [b, 64, 6]
                if name == "pure":
                    tot["pure"] += bce.sum((0, 1))
                    n["pure"] += bce.shape[0] * bce.shape[1]
                else:
                    cp = np.array([[not np.array_equal(D["served_traj"][i][j], D["pure_traj"][i][j]) for j in range(64)]
                                   for i in idx])
                    tot["copies"] += bce[cp].sum(0)
                    n["copies"] += int(cp.sum())
    model.train()
    return {k: [round(float(v), 5) for v in tot[k] / max(n[k], 1)] for k in tot}


def train(a) -> int:
    device = torch.device(a.device)
    vram_cap(device)
    reserve_vram(device)
    amp = device.type == "cuda"
    dtype = torch.float32 if not amp else torch.bfloat16
    torch.set_num_threads(a.threads)
    D = dict(np.load(os.path.join(a.m5, "ft_data_train.npz")))
    pv = os.path.join(a.m5, "ft_data_val.npz")
    Dv = dict(np.load(pv)) if os.path.exists(pv) and os.path.isdir(os.path.join(CACHE, "navtrain_val")) else None
    C = Cache("navtrain_train")
    Cv = Cache("navtrain_val") if Dv is not None else None
    if a.preload:
        t = time.time()
        C.preload()
        print(f"  preloaded the train context in {time.time() - t:.0f} s", flush=True)
    keys = [str(k) for k in D["key"]]
    N = len(keys)
    arms = [x for x in a.arms.split(",") if x]
    base = load_scorer()
    for seed in [int(x) for x in a.seeds.split(",") if x != ""]:
        if os.path.exists(os.path.join(a.m5, "ft", f"trainlog_s{seed}.json")) and not a.force:
            print(f"  seed {seed}: already trained, skipping", flush=True)
            continue
        torch.manual_seed(SEED + seed)
        models, opts = {}, {}
        for arm in arms:
            m = Scorer(*[__import__("copy").deepcopy(x) for x in (base.score_q_mlp, base.score_dec, base.score_head)])
            models[arm] = m.to(device).train()
            opts[arm] = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
        log = {"seed": seed, "arms": arms, "N": N, "device": str(device), "amp_bf16": amp, "lr": LR, "wd": WD,
               "score_w": SCORE_W, "batch": BATCH, "epochs": EPOCHS, "steps": [], "val_bce": [], "order_sha": []}
        t0 = time.time()
        step = 0
        for ep in range(EPOCHS):
            g = torch.Generator().manual_seed(SEED * 1000 + seed * 100 + ep)
            perm = torch.randperm(N, generator=g).numpy()
            log["order_sha"].append(hashlib.sha256(perm.tobytes()).hexdigest()[:16])
            for b0 in range(0, N, BATCH):
                if a.deadline and past(a.deadline):
                    print(f"ZZM5_TRAIN_STOPPED deadline {a.deadline}: seed {seed} at epoch {ep} step {step} is NOT "
                          f"saved (a partial fine-tune is not the registered one)", flush=True)
                    return 2
                idx = perm[b0:b0 + BATCH]
                ctx = C.ctx([keys[i] for i in idx], device, dtype)
                rec = {"step": step, "epoch": ep}
                for arm in arms:
                    tr, tg = arm_sets(arm, D, idx)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        sx = models[arm](torch.from_numpy(tr).to(device, dtype), ctx)
                    per = F.binary_cross_entropy_with_logits(sx.float(), torch.from_numpy(tg).to(device),
                                                             reduction="none").mean(-1)             # [b, 64]
                    loss = per.sum() / float(64 * len(idx)) * SCORE_W                                # train.py:1339-1342
                    opts[arm].zero_grad(set_to_none=True)
                    loss.backward()
                    gn = torch.norm(torch.stack([p.grad.norm() for p in models[arm].parameters() if p.grad is not None]))
                    opts[arm].step()
                    rec[arm] = round(float(loss), 6)
                    rec[arm + "_gnorm"] = round(float(gn), 5)
                log["steps"].append(rec)
                step += 1
                if step % 50 == 0:
                    print(f"  seed {seed} ep {ep} step {step}: " + "  ".join(f"{arm} {rec[arm]:.5f}" for arm in arms)
                          + f"  {time.time() - t0:.0f} s  local {time.strftime('%H:%M:%S')}", flush=True)
            if Cv is not None:
                vb = {arm: val_bce(models[arm], Dv, Cv, device, dtype, amp) for arm in arms}
                log["val_bce"].append({"epoch": ep, **vb})
                print(f"  seed {seed} ep {ep} val BCE (pure): " + "  ".join(f"{arm} {np.mean(vb[arm]['pure']):.5f}"
                                                                          for arm in arms), flush=True)
        os.makedirs(os.path.join(a.m5, "ft"), exist_ok=True)
        for arm in arms:
            torch.save({k: v.detach().cpu() for k, v in models[arm].state_dict().items()},
                       os.path.join(a.m5, "ft", f"{arm}_s{seed}.pt"))
        log["seconds"] = round(time.time() - t0, 1)
        if device.type == "cuda":
            log["cuda_max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            log["cuda_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 2 ** 30, 3)
        json.dump(log, open(os.path.join(a.m5, "ft", f"trainlog_s{seed}.json"), "w"), indent=0)
        print(f"ZZM5_TRAIN_OK seed {seed} arms {arms} in {log['seconds']} s", flush=True)
    return 0


# ------------------------------------------------------------------------------------------ eval
def read_csv(p):
    rows = {}
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["token"] != "average":
                rows[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in SUB))
    return rows


def v1_agg(logits: np.ndarray) -> np.ndarray:
    """The planner's own navsim_v1 aggregate (REFePlanner.aggregate, called through the v4 labeller's shim)."""
    import onpolicy_label_v4 as L4
    flat = logits.reshape(-1, 6)
    return L4.v1_aggregate(flat).reshape(logits.shape[:-1])


def navtest_setup():
    import slow_copies as SC
    T = np.load(TABLE)
    RK = np.load(RANKS)
    B = np.load(BUILD, allow_pickle=False)
    C = Cache("navtest_sub200")
    toks = C.keys()
    ti = {str(t): i for i, t in enumerate(T["token"])}
    props = C.array("props")
    top = RK["top"]
    X, H, gate_copy = [], [], 0.0
    hr = {r: read_csv(f"{NAV}/score/refe_sub200_ep015_f075_r{r:02d}/refe_sub200_ep015_f075_r{r:02d}.csv")
          for r in range(N_SRC)}
    stop = read_csv(STOP_CSV)
    missing = 0
    for n, t in enumerate(toks):
        i = ti[t]
        ex = [SC.slow_copy(props[n, int(top[i, r])], FACTOR) for r in range(N_SRC)] + [np.zeros((20, 3), np.float32)]
        for r in range(N_SRC):
            gate_copy = max(gate_copy, float(np.abs(ex[r].astype(np.float64) -
                                                    B["nat_f075"][i, int(top[i, r])].astype(np.float64)).max()))
        X.append(np.stack(ex).astype(np.float32))
        h = [hr[r].get(t) for r in range(N_SRC)] + [stop.get(t)]
        missing += sum(1 for x in h if x is None or not x[0])
        H.append([T["pdms"][i, j] for j in range(64)] + [x[1] if x else np.nan for x in h])
    return {"C": C, "toks": toks, "tidx": np.array([ti[t] for t in toks]), "props": props, "X": np.stack(X),
            "pdms": np.asarray(H, np.float64), "T": T, "gate_copy_max_abs": gate_copy, "harness_missing": missing,
            "sub_extra": np.array([[hr[r].get(t, (0, 0, (np.nan,) * 6))[2] for r in range(N_SRC)]
                                   + [stop.get(t, (0, 0, (np.nan,) * 6))[2]] for t in toks], np.float64)}


def eval_model(model, E, device, bs: int = 8) -> dict:
    """logits of the 64 alone, of the 64 + extras MASKED, and PLAIN -- fp32, the planner's numerics."""
    model = model.to(device).eval()
    K = E["X"].shape[1]
    mask = torch.zeros(64 + K, 64 + K, dtype=torch.bool, device=device)
    mask[:64, 64:] = True
    mask[64:, 64:] = ~torch.eye(K, dtype=torch.bool, device=device)
    out = {"pure": [], "masked": [], "plain": []}
    with torch.no_grad():
        for b0 in range(0, len(E["toks"]), bs):
            ks = E["toks"][b0:b0 + bs]
            ctx = E["C"].ctx(ks, device, torch.float32)
            c64 = torch.from_numpy(E["props"][b0:b0 + bs]).to(device)
            full = torch.cat([c64, torch.from_numpy(E["X"][b0:b0 + bs]).to(device)], 1)
            out["pure"].append(model(c64, ctx).cpu().numpy())
            out["masked"].append(model.masked(full, ctx, mask).cpu().numpy())
            out["plain"].append(model(full, ctx).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def concordance(agg: np.ndarray, pdms: np.ndarray, involve: np.ndarray | None) -> float:
    """Fraction of candidate pairs the aggregate orders like the harness (ties in the aggregate 0.5), over pairs the
    harness does not tie; `involve` [n] bool: only pairs with >= 1 involved member (None = all pairs)."""
    ok = np.isfinite(pdms)
    a_, p_ = agg[ok], pdms[ok]
    inv = involve[ok] if involve is not None else None
    dh = p_[:, None] - p_[None, :]
    da = a_[:, None] - a_[None, :]
    iu = np.triu_indices(len(p_), 1)
    sel = np.abs(dh[iu]) > 1e-9
    if inv is not None:
        sel &= (inv[:, None] | inv[None, :])[iu]
    if not sel.any():
        return float("nan")
    s = np.sign(da[iu][sel]) * np.sign(dh[iu][sel])
    return float(np.mean(np.where(s > 0, 1.0, np.where(s == 0, 0.5, 0.0))))


def metrics(L: dict, E: dict) -> dict:
    """Per-token metrics from one model's logits."""
    n = len(E["toks"])
    K = E["X"].shape[1]
    extra = np.array([False] * 64 + [True] * K)
    agg_pure, agg_m, agg_p = v1_agg(L["pure"]), v1_agg(L["masked"]), v1_agg(L["plain"])
    P = E["pdms"]
    ar = np.arange(n)
    m = {"E1_masked": np.array([concordance(agg_m[t], P[t], extra) for t in ar]),
         "E1_plain": np.array([concordance(agg_p[t], P[t], extra) for t in ar]),
         "E3a": np.array([concordance(agg_pure[t], P[t, :64], None) for t in ar]),
         "E3b": 100.0 * P[ar, agg_pure.argmax(1)],
         "E2": 100.0 * P[ar, np.nan_to_num(agg_m, nan=-1).argmax(1)]}
    sig = 1.0 / (1.0 + np.exp(-L["masked"][:, 64:, :].astype(np.float64)))              # [n, K, 6]
    m["E4_pred_stop"] = sig[:, -1, :]
    m["E4_pred_copies"] = sig[:, :-1, :].mean(1)
    return m


def eval_(a) -> int:
    device = torch.device(a.device)
    vram_cap(device)
    reserve_vram(device)
    torch.set_num_threads(a.threads)
    E = navtest_setup()
    os.makedirs(os.path.join(a.m5, "evals"), exist_ok=True)
    base = load_scorer()
    models = {"A0": base}
    for p in sorted(glob.glob(os.path.join(a.m5, "ft", "*_s*.pt"))):
        name = os.path.basename(p)[:-3]
        m = Scorer(*[__import__("copy").deepcopy(x) for x in (base.score_q_mlp, base.score_dec, base.score_head)])
        m.load_state_dict(torch.load(p, map_location="cpu"))
        models[name] = m
    gates = {"copies_equal_harness_scored_max_abs": E["gate_copy_max_abs"], "harness_missing": E["harness_missing"]}
    np.savez(os.path.join(a.m5, "evals", "_harness_extra.npz"), token=np.array(E["toks"]), sub_extra=E["sub_extra"],
             pdms=E["pdms"])
    for name, m in models.items():
        t0 = time.time()
        L = eval_model(m, E, device)
        if name == "A0":
            tl = E["T"]["logits"][E["tidx"]].astype(np.float64)
            gates["A0_pure_vs_table_logits_max_abs"] = float(np.abs(L["pure"].astype(np.float64) - tl).max())
            gates["A0_pick_equals_table"] = int((v1_agg(L["pure"]).argmax(1) == E["T"]["pick"][E["tidx"]]).sum())
        gates.setdefault("masked_64_vs_pure_max_abs", 0.0)
        gates["masked_64_vs_pure_max_abs"] = max(gates["masked_64_vs_pure_max_abs"],
                                                 float(np.abs(L["masked"][:, :64] - L["pure"]).max()))
        mt = metrics(L, E)
        np.savez(os.path.join(a.m5, "evals", f"{name}.npz"), token=np.array(E["toks"]), **L, **mt)
        print(f"  {name}: E1 masked {np.nanmean(mt['E1_masked']):.4f} plain {np.nanmean(mt['E1_plain']):.4f} "
              f"E3a {np.nanmean(mt['E3a']):.4f} E3b {mt['E3b'].mean():.2f} E2 {mt['E2'].mean():.2f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    json.dump(gates, open(os.path.join(a.m5, "evals", "gates.json"), "w"), indent=1)
    print(json.dumps(gates))
    ok = (gates["copies_equal_harness_scored_max_abs"] == 0.0 and gates["harness_missing"] == 0
          and gates["A0_pure_vs_table_logits_max_abs"] <= 1e-3 and gates["A0_pick_equals_table"] == len(E["toks"])
          and gates["masked_64_vs_pure_max_abs"] <= 1e-4)
    print(f"ZZM5_EVAL_{'OK' if ok else 'GATE_FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------ readout
def cluster_draws(tokens: list, boot: int = 10000):
    """eval/snapshot_pair_under_rule.py's draws, verbatim: resample the LOGS with replacement, seed 20260927."""
    tl = json.load(open(TOK, encoding="utf-8"))["token_log"]
    logs = np.array([tl[t] for t in tokens])
    ul = np.unique(logs)
    idx = {l: np.where(logs == l)[0] for l in ul}
    rng = np.random.default_rng(20260927)
    return [np.concatenate([idx[l] for l in rng.choice(ul, size=len(ul), replace=True)]) for _ in range(boot)], len(ul)


def boot_ci(d: np.ndarray, draws) -> tuple:
    bs = []
    for ix in draws:
        v = d[ix]
        v = v[np.isfinite(v)]
        bs.append(v.mean() if len(v) else np.nan)
    bs = np.asarray(bs)
    return float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def estimator_selftest() -> dict:
    """Our draws reproduce snapshot_pair_under_rule.compare's published interval EXACTLY (ep014 -> ep015, navsim_v1)."""
    import snapshot_pair_under_rule as SPR
    ref = SPR.compare("sub200_ep014", "sub200_ep015", boot=10000)["rules"]["navsim_v1"]
    A, B = SPR.load("sub200_ep014"), SPR.load("sub200_ep015")
    ar = np.arange(len(A["token"]))
    d = (B["pdms"][ar, SPR.aggregate(B["logits"], B["order"], "navsim_v1").argmax(1)]
         - A["pdms"][ar, SPR.aggregate(A["logits"], A["order"], "navsim_v1").argmax(1)]) * 100
    draws, _ = cluster_draws(A["token"])
    lo, hi = boot_ci(d, draws)
    mine = [round(lo, 2), round(hi, 2)]
    return {"reference_ci": ref["ci95"], "ours": mine, "identical": mine == ref["ci95"]}


def readout(a) -> int:
    E = np.load(os.path.join(a.m5, "evals", "A0.npz"))
    toks = [str(t) for t in E["token"]]
    draws, n_logs = cluster_draws(toks)
    ev = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in glob.glob(os.path.join(a.m5, "evals", "*.npz"))
          if not os.path.basename(p).startswith("_")}
    HX = np.load(os.path.join(a.m5, "evals", "_harness_extra.npz"))
    seeds = sorted({int(k.split("_s")[1]) for k in ev if "_s" in k})
    res = {"_label": "PRE-REGISTERED readout, eval/PREREG_MEASURE5.md (blob c668b5b6, SPEC_PREREG_HASH 13:22)",
           "n_tokens": len(toks), "n_logs": n_logs, "seeds": seeds, "estimator": "paired log-cluster bootstrap over "
           "the logs, 10,000 resamples, 95 % percentile, seed 20260927 (eval/snapshot_pair_under_rule.py's draws)",
           "variance_answered": "EPISODES for the CI; TRAINING (batch order) by the seed floor; inference is "
                                "deterministic", "estimator_selftest": estimator_selftest(),
           "gates": json.load(open(os.path.join(a.m5, "evals", "gates.json")))}

    def arm_mean(arm, key):
        return np.nanmean(np.stack([ev[f"{arm}_s{s}"][key] for s in seeds]), 0)

    res["token_means"] = {name: {k: round(float(np.nanmean(v[k])), 5) for k in ("E1_masked", "E1_plain", "E3a", "E3b",
                                                                               "E2")} for name, v in sorted(ev.items())}
    cmp = {}
    for key in ("E1_masked", "E1_plain", "E3a", "E3b", "E2"):
        for pair, (x, y) in {"V4_minus_V3": ("V4", "V3"), "V4all_minus_V4": ("V4all", "V4"),
                             "V3_minus_A0": ("V3", "A0")}.items():
            try:
                xa = arm_mean(x, key) if x != "A0" else ev["A0"][key]
                ya = arm_mean(y, key) if y != "A0" else ev["A0"][key]
            except KeyError:
                continue
            d = xa - ya
            lo, hi = boot_ci(d, draws)
            cmp.setdefault(key, {})[pair] = {"diff": round(float(np.nanmean(d)), 5), "ci95": [round(lo, 5), round(hi, 5)]}
    res["comparisons"] = cmp
    floor = 0.0
    fl = {}
    for arm in ("V3", "V4"):
        tm = [float(np.nanmean(ev[f"{arm}_s{s}"]["E1_masked"])) for s in seeds if f"{arm}_s{s}" in ev]
        pairs = [abs(tm[i] - tm[j]) for i in range(len(tm)) for j in range(i + 1, len(tm))]
        fl[arm] = {"seed_token_means": [round(x, 5) for x in tm], "max_pair_abs": round(max(pairs), 5) if pairs else None}
        floor = max([floor] + pairs)
    res["seed_floor_E1"] = {"floor": round(floor, 5), "by_arm": fl}
    e1 = cmp["E1_masked"]["V4_minus_V3"]
    e3a = cmp["E3a"]["V4_minus_V3"]
    e3b = cmp["E3b"]["V4_minus_V3"]
    adopt = (e1["diff"] >= 0.03 and e1["ci95"][0] > 0 and e1["diff"] > floor and e3a["ci95"][0] > -0.01
             and e3b["ci95"][0] > -2.0)
    refuted = e1["ci95"][1] < 0
    # ⛔ the statistic is READ only when every gate passed and every pre-registered model exists
    g = res["gates"]
    gate_fail = [k for k, ok in (
        ("copies_equal_harness_scored", g.get("copies_equal_harness_scored_max_abs") == 0.0),
        ("harness_complete", g.get("harness_missing") == 0),
        ("A0_reproduces_table_logits", (g.get("A0_pure_vs_table_logits_max_abs") or 1.0) <= 1e-3),
        ("A0_reproduces_table_picks", g.get("A0_pick_equals_table") == len(toks)),
        ("masked_route_identity", (g.get("masked_64_vs_pure_max_abs") or 1.0) <= 1e-4),
        ("estimator_selftest", res["estimator_selftest"]["identical"])) if not ok]
    need = [f"{arm}_s{s_}" for arm in ("V3", "V4") for s_ in (0, 1, 2)]
    absent = [n for n in need if n not in ev]
    verdict = ("NO READOUT: gates failed " + ",".join(gate_fail)) if gate_fail else         (("NO READOUT: models absent " + ",".join(absent)) if absent else
         ("ADOPT" if adopt else ("REFUTED" if refuted else "NOT PROVEN")))
    res["decision"] = {"ADOPT": adopt and not gate_fail and not absent,
                       "REFUTED": refuted and not adopt and not gate_fail and not absent,
                       "verdict": verdict, "gates_failed": gate_fail, "models_absent": absent,
                       "rule": "ADOPT iff E1 V4-V3 >= +0.03, CI lo > 0, > seed floor; E3a CI lo > -0.01; E3b CI lo > "
                               "-2.0 PDMS points. REFUTED iff E1 CI hi < 0. Otherwise NOT PROVEN.",
                       "E1": e1, "E3a": e3a, "E3b": e3b, "seed_floor": round(floor, 5)}
    T = np.load(TABLE)
    E4 = {}
    for name in ("A0",) + tuple(f"{arm}_s{s}" for arm in ("V3", "V4") for s in seeds):
        if name in ev:
            E4[name] = {"pred_stop_NC_TTC_EP": [round(float(x), 4) for x in np.nanmean(ev[name]["E4_pred_stop"], 0)[[0, 3, 2]]],
                        "pred_copies_NC_TTC_EP": [round(float(x), 4) for x in np.nanmean(ev[name]["E4_pred_copies"], 0)[[0, 3, 2]]]}
    se = HX["sub_extra"]                                                     # [n, 9, 6] harness sub-scores
    res["E4_reported"] = {"harness_stop_NC_TTC_EP": [round(float(x), 4) for x in np.nanmean(se[:, -1, :], 0)[[0, 3, 2]]],
                          "harness_copies_NC_TTC_EP": [round(float(x), 4) for x in np.nanmean(se[:, :-1, :], (0, 1))[[0, 3, 2]]],
                          "models": E4}
    # the repo's results folder ONLY for the real data directory; a test run writes beside its own data
    out_dir = OUT if os.path.normcase(os.path.abspath(a.m5)) == os.path.normcase(os.path.abspath(M5))         else os.path.join(a.m5, "readout")
    os.makedirs(out_dir, exist_ok=True)
    res["written_to"] = out_dir
    json.dump(res, open(os.path.join(out_dir, "readout.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(res["decision"], indent=1))
    print(json.dumps(res["estimator_selftest"]))
    print(f"ZZM5_READOUT {res['decision']['verdict']}")
    return 0


def main() -> int:
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("data", "train", "eval", "readout"))
    ap.add_argument("--m5", default=M5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--preload", action="store_true", help="the whole train context into RAM once")
    ap.add_argument("--deadline", default="", help="local HH:MM: stop between steps; an unfinished seed is not saved")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--arms", default="V3,V4,V4all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cache", default=CACHE, help="the shared trunk cache root (a test cache for code-path runs)")
    a = ap.parse_args()
    CACHE = a.cache
    return {"data": data, "train": train, "eval": eval_, "readout": readout}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
