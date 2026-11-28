#!/usr/bin/env python3
"""The dev-box DECODER-SIDE PROXY TRAINER: REFe's trajectory decoder and scorer fine-tuned on the shared cache.

The encoder side is frozen and read from the cache (refe/proxy_cache.py): scene_ctx [64, 256] and visual_ctx [7680, 256]
per frame, bf16 bits. Everything that TRAINS is REFe's own modules, loaded from a live snapshot and run through the
lines of `REFe.forward` that follow `scene_proj`:
  trajectory  ego/goal token -> queries -> dec -> traj_head      (then the M4b hook, when installed)
  scorer      score_q_mlp(traj.detach()) -> score_dec over visual_ctx -> score_head, and
              REFe.score_trajectories(labelled set, visual_ctx) for the on-policy sets (train.py's on-policy branch)
LOSS / STEP -- train.py's on-policy loop line for line:
  l_traj   measures.wta_loss_yaw(mode) ('wrapped' IS model.wta_loss), or measures.anchored_wta_loss under M4b
  l_score  BCE per component on each covered sample's labelled set, / (cov_norm x batch) x score_w; uncovered
           micro-batches contribute score.sum() * 0 (no gradient)
  (l_traj + l_score) / accum -> backward; every accum micro-batches: clip (global 1.0, or M3 split), AdamW step
LABELS: train.OnPolicyBank on the extracted set lines, restricted to ckpt_step <= --max-ckpt-step (the newest version
per key wins, exactly as the live trainer). SAMPLER: train.SceneEpochSampler (one member per scene per epoch).
SCHEDULE: --zero-lr-steps optimiser steps at lr 0 (Adam moment re-estimation; weights provably unchanged), then
--steps at the LIVE cosine continued from --start-step: lr = 2e-4 (1 + cos(pi t / T_max)) / 2.

    python refe/proxy_train.py --cache <dir> --sets <dir> --snapshot <snap.pt> --out <arm dir>
        [--yaw-loss wrapped|plain] [--seed 0] [--steps 403] [--zero-lr-steps 50] [--batch 32 --accum 8]
        [--clip-split] [--slow-slots 8 --slow-slot-indices ...] [--twins 0.5 --twins-decel 4]
Writes <out>/{argv.json, config.json, metrics.jsonl, final.pt}; prints ZZPROXYTRAIN_OK.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measures as MS  # noqa: E402
import proxy_cache as PC  # noqa: E402
from model import REFe, REFeConfig, wta_loss  # noqa: E402

DEC = ("ego_enc.", "queries", "dec.", "traj_head.")
SCO = ("score_q_mlp.", "score_dec.", "score_head.")
LIVE_LR, LIVE_T_MAX = 2e-4, 10075          # the live run: lr 2e-4, cosine over 25 x 403 optimiser steps


def sha256_file(p) -> str:
    return PC.sha256_file(p)


def live_lr(t: int) -> float:
    return LIVE_LR * (1.0 + math.cos(math.pi * t / LIVE_T_MAX)) / 2.0


def build(snapshot: str, device: str):
    """REFe's decoder side with the snapshot's trained tensors. The ViT trunk is built with depth 0 (the cache stands
    in for it); every decoder/scorer tensor MUST come from the snapshot (refused otherwise)."""
    cfg = REFeConfig.for_backbone("vitl16")
    cfg.depth = 0
    model = REFe(cfg)
    part = torch.load(snapshot, map_location="cpu", weights_only=False)["model_partial"]
    want = {k: v for k, v in part.items() if k.startswith(DEC + SCO)}
    own = {k for k in model.state_dict() if k.startswith(DEC + SCO)}
    if set(want) != own:
        raise SystemExit(f"snapshot decoder keys != model decoder keys: missing {sorted(own - set(want))[:3]} "
                         f"extra {sorted(set(want) - own)[:3]}")
    model.load_state_dict(want, strict=False)
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith(DEC + SCO))
    return model.to(device), cfg, want


def decode(model, scene_ctx, ego, goal):
    """REFe.forward after scene_proj -- trajectory side (then the M4b hook, as the patched forward applies it)."""
    B = scene_ctx.shape[0]
    g = goal.unsqueeze(-1) * model.goal_freqs
    goal_feat = torch.cat([goal, g.sin().flatten(1), g.cos().flatten(1)], -1)
    ego_tok = model.ego_enc(torch.cat([ego, goal_feat], -1)).unsqueeze(1)
    q = model.queries.expand(B, -1, -1) + ego_tok
    for blk in model.dec:
        q = blk(q, scene_ctx)
    traj = model.traj_head(q).view(B, model.cfg.n_proposals, model.cfg.horizon_steps, model.cfg.traj_dim)
    slow = getattr(model, "slow_slots", None)
    return slow(traj, ego) if slow is not None else traj


def score(model, traj, visual_ctx):
    """REFe.forward's scorer on the proposals (the candidate trajectories detached, as the paper requires)."""
    s = model.score_q_mlp(traj.detach().flatten(2))
    for blk in model.score_dec:
        s = blk(s, visual_ctx)
    return model.score_head(s)


class Cache:
    """memory-mapped shards of the shared cache (bf16 bits as int16). A frame's visual_ctx sits at row `vindex` of its
    shard's visual file (absent = `index`; -1 = not stored, `proxy_cache.py --visual-for sets`)."""

    def __init__(self, d: Path):
        self.d = d
        self.mf = json.load(open(d / "manifest.json", encoding="utf-8"))
        self.frames = [json.loads(l) for l in open(d / "frames.jsonl", encoding="utf-8")]
        self.bf16 = self.mf["dtype"] == "bf16"
        dt = np.int16 if self.bf16 else np.float32
        self.mm = {}
        sizes, vsizes = {}, {}
        for fr in self.frames:
            sizes[fr["shard"]] = max(sizes.get(fr["shard"], 0), fr["index"] + 1)
            vsizes[fr["shard"]] = max(vsizes.get(fr["shard"], 0), fr.get("vindex", fr["index"]) + 1)
        for s, n in sizes.items():
            self.mm[s] = (np.memmap(d / f"shard_{s:04d}.scene.bin", dtype=dt, mode="r", shape=(n, 64, 256)),
                          np.memmap(d / f"shard_{s:04d}.visual.bin", dtype=dt, mode="r", shape=(vsizes[s], 7680, 256))
                          if vsizes[s] > 0 else None)
        self.pos = {tuple(fr["key"]): (fr["shard"], fr["index"], fr.get("vindex", fr["index"])) for fr in self.frames}

    def scene(self, key):
        s, i, _v = self.pos[key]
        return np.array(self.mm[s][0][i])

    def visual(self, key):
        s, _i, v = self.pos[key]
        return None if v < 0 else np.array(self.mm[s][1][v])

    def get(self, key):
        return self.scene(key), self.visual(key)

    def to_tensor(self, arr):
        t = torch.from_numpy(arr)
        return t.view(torch.bfloat16) if self.bf16 else t


def load_rows(cache_dir: Path, twins: list, twins_decel: float) -> list:
    rows = [json.loads(l) for l in open(cache_dir / "rows.jsonl", encoding="utf-8")]
    out = []
    for r in rows:
        out.append({"log_name": r["key"][0], "token": r["key"][1], "step": int(r["key"][2]), "rank": int(r["rank"]),
                    "ego": r["ego"], "goal": r["goal"], "traj": r["traj"]})
    if twins:
        import slow_twins as STW
        sc_sha = STW._sha256(Path(STW.SC.__file__))
        rmap = STW.rank_map(sorted({r["rank"] for r in out}), twins, 2)
        base = list(out)
        for r in base:
            for f in twins:
                tw, d = STW.make_twin(r, f, rmap[(r["rank"], float(f))], sc_sha, twins_decel)
                if d >= 0.3:
                    out.append(tw)
    return out


def make_batch(ids, rows, cache, op, n_goal_points: int):
    """EQUIVALENT COMPUTATION, not a change of loss: live, an uncovered sample's set is all zeros under mask 0 and
    the proposals' own scores never enter the on-policy loss. So only the COVERED samples' sets go through the
    scorer (theirs is the only visual_ctx read); the per-set BCE sum and the constant denominator
    (cov_norm x FULL batch) are train.py's, so loss and gradient are the live ones (up to float summation
    order)."""
    sc, eg, go, tg, vi, cx, cg = [], [], [], [], [], [], []
    for i in ids:
        r = rows[i]
        key = (r["log_name"], r["token"], r["step"])
        sc.append(cache.scene(key))
        eg.append(r["ego"])
        go.append(r["goal"][:2 * n_goal_points])
        tg.append(r["traj"])
        got = op.get(r["log_name"], r["token"], r["step"], MS.scorer_rank(r)) if op is not None else None
        if got is not None:
            v_ = cache.visual(key)
            if v_ is None:
                raise RuntimeError(f"a LABELLED sample has no cached visual_ctx: {key} -- rebuild the cache")
            vi.append(v_)
            cx.append(got[0])
            cg.append(got[1])
    n_cov = len(cx)
    return (cache.to_tensor(np.stack(sc)), torch.tensor(eg, dtype=torch.float32),
            torch.tensor(go, dtype=torch.float32), torch.tensor(tg, dtype=torch.float32),
            cache.to_tensor(np.stack(vi)) if n_cov else None,
            torch.from_numpy(np.stack(cx)) if n_cov else None,
            torch.from_numpy(np.stack(cg)) if n_cov else None)


def score_loss(sx, cg, batch: int, cov_norm: float, score_w: float, sco_params):
    """train.py's on-policy scorer loss (its lines `per = BCE(sx, cand_tg).mean(-1)`, `denom = max(cov_norm x
    batch, 1)`, `l_score = (per x cand_m).sum() / denom x score_w`) on the COVERED samples' logits sx [n_cov, M, 6];
    `batch` is the FULL micro-batch. sx None (no covered sample): live's `score.sum() * 0.0`, i.e. zero-valued
    gradients on every scorer tensor."""
    if sx is None:
        return sum(p_.sum() for p_ in sco_params) * 0.0
    per = F.binary_cross_entropy_with_logits(sx.float(), cg, reduction="none").mean(-1)       # [n_cov, M]
    return per.sum() / max(cov_norm * float(batch), 1.0) * score_w


def prep_sets(sets_dir: Path, max_ckpt: int, work: Path) -> Path:
    """extracted gz parts -> ONE onpolicy_proxy.jsonl holding only lines with ckpt_step <= max_ckpt (train.OnPolicyBank
    then keeps the newest (ckpt_step, label_version) per key, as live)"""
    work.mkdir(parents=True, exist_ok=True)
    outp = work / "onpolicy_proxy.jsonl"
    n_in = n_keep = 0
    with open(outp, "w", encoding="utf-8", newline="\n") as fo:
        parts = sorted(Path(sets_dir).glob("*.jsonl.gz"))
        if not parts:
            raise SystemExit(f"no *.jsonl.gz set parts in {sets_dir} -- the scorer would silently train on nothing")
        for p in parts:
            for line in gzip.open(p, "rt", encoding="utf-8"):
                n_in += 1
                try:
                    ck = int(json.loads(line)["ckpt_step"])
                except Exception:
                    continue
                if ck <= max_ckpt:
                    fo.write(line if line.endswith("\n") else line + "\n")
                    n_keep += 1
    print(f"  on-policy lines: {n_in:,} extracted from {len(parts)} parts, {n_keep:,} with ckpt_step <= {max_ckpt}",
          flush=True)
    if n_keep == 0:
        raise SystemExit("no on-policy line passes the ckpt_step filter -- refusing a scorer arm with no labels")
    return work


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--sets", default=None, help="dir of extracted onpolicy part_*.jsonl.gz (no scorer loss without)")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yaw-loss", choices=MS.YAW_MODES, default="wrapped")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=403)
    ap.add_argument("--zero-lr-steps", type=int, default=50)
    ap.add_argument("--start-step", type=int, default=4933)
    ap.add_argument("--max-ckpt-step", type=int, default=4933)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--score-w", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--lr-override", type=float, default=None, help="GATE G3 only: a constant lr (0 = identity test)")
    ap.add_argument("--clip-split", action="store_true", help="M3")
    ap.add_argument("--slow-slots", type=int, default=0, help="M4b")
    ap.add_argument("--slow-slot-indices", default="")
    ap.add_argument("--twins", default="", help="M4a factors, e.g. 0.5")
    ap.add_argument("--twins-decel", type=float, default=0.0)
    ap.add_argument("--deadline", default=None, help="local HH:MM: stop cleanly (the arm is then INCOMPLETE)")
    ap.add_argument("--sets-work", default=None, help="where the filtered set file goes (keep it OUT of the repo)")
    ap.add_argument("--dump-grads-after-accum", default=None,
                    help="TEST ONLY: save the first optimiser step's accumulated (unclipped) gradients here and exit")
    a = ap.parse_args()
    import train as T                                            # SceneEpochSampler, OnPolicyBank, onpolicy_cov_norm
    out = Path(a.out)
    if (out / "final.pt").exists():
        print(f"  REFUSING: {out}/final.pt exists -- an arm is never overwritten")
        return 4
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model, cfg, snap_state = build(a.snapshot, dev)
    slow_cfg = None
    if a.slow_slots:
        slow_cfg = MS.SlowSlotConfig.from_args(a.slow_slots, MS.DEFAULT_SLOW_PROFILES, cfg.n_proposals,
                                               indices=a.slow_slot_indices)
        MS.install_slow_slots(model, slow_cfg)
    cache = Cache(Path(a.cache))
    twins = [float(x) for x in a.twins.split(",") if x.strip()]
    rows = [r for r in load_rows(Path(a.cache), twins, a.twins_decel) if (r["log_name"], r["token"], r["step"]) in cache.pos]
    op = None
    if a.sets:
        wd = prep_sets(Path(a.sets), a.max_ckpt_step, Path(a.sets_work) if a.sets_work else out / "_sets")
        op = T.OnPolicyBank(str(wd), cfg.n_proposals, cfg.horizon_steps)
    # scene groups exactly as TargetBank builds them: (log, token, step) members ordered by rank
    idx = {}
    for i, r in enumerate(rows):
        idx.setdefault((r["log_name"], r["token"], r["step"]), []).append(i)
    groups = [sorted(v, key=lambda i: rows[i]["rank"]) for v in idx.values()]
    sampler = T.SceneEpochSampler(groups, a.seed, shuffle=True)

    class _DS:                                                    # onpolicy_cov_norm reads len(ds) and ds.rows
        def __len__(self):
            return len(self.rows)
    ds = _DS()
    ds.rows = rows
    cov_norm = T.onpolicy_cov_norm(ds, op, cfg.n_proposals) if op is not None else 1.0
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0, weight_decay=a.weight_decay)
    grps = MS.clip_groups(model) if a.clip_split else None
    code = {f: sha256_file(HERE / f) for f in ("proxy_train.py", "measures.py", "model.py", "train.py", "proxy_cache.py")}
    cfg_out = {"argv": sys.argv, "args": vars(a), "device": dev, "cuda": torch.cuda.get_device_name(0) if dev == "cuda"
               else None, "torch": torch.__version__, "snapshot_sha256": sha256_file(a.snapshot),
               "cache_manifest_sha256": sha256_file(Path(a.cache) / "manifest.json"), "code_sha256": code,
               "rows": len(rows), "scenes": len(groups), "trainable": sum(p.numel() for p in params),
               "onpolicy_sets": len(op.by) if op is not None else 0, "cov_norm": cov_norm,
               "started_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(sys.argv, open(out / "argv.json", "w"), indent=1)
    json.dump(cfg_out, open(out / "config.json", "w"), indent=1, default=str)
    print(f"  {len(rows):,} rows / {len(groups):,} scenes, sets {cfg_out['onpolicy_sets']:,}, cov_norm {cov_norm:.3f}, "
          f"trainable {cfg_out['trainable']:,}, device {dev}", flush=True)

    # ---- the sample stream (seeded, epoch after epoch), batches prepared by a background thread
    def order():
        ep = 0
        while True:
            sampler.set_epoch(ep)
            for i in sampler.order():
                yield i
            ep += 1
    stream = order()
    M = cfg.n_proposals

    total_steps = a.zero_lr_steps + a.steps
    n_micro = total_steps * a.accum
    q: queue.Queue = queue.Queue(maxsize=3)

    def producer():
        for _ in range(n_micro):
            q.put(make_batch([next(stream) for _ in range(a.batch)], rows, cache, op, cfg.n_goal_points))
        q.put(None)
    threading.Thread(target=producer, daemon=True).start()
    amp = a.amp == "bf16" and dev == "cuda"
    sco_params = [p_ for n_, p_ in model.named_parameters() if n_.startswith(SCO) and p_.requires_grad]
    n_cov_total = 0
    info: dict = {}
    t0, t_last = time.time(), time.time()
    mlog = open(out / "metrics.jsonl", "a", encoding="utf-8")
    stopped = None
    step = 0
    for step in range(total_steps):
        if PC.past(a.deadline):
            stopped = f"deadline {a.deadline} at step {step}"
            break
        lr = (a.lr_override if a.lr_override is not None else
              0.0 if step < a.zero_lr_steps else live_lr(a.start_step + step - a.zero_lr_steps))
        for g in opt.param_groups:
            g["lr"] = lr
        acc = {"l_traj": 0.0, "l_score": 0.0}
        cov_micro = []
        for _ in range(a.accum):
            b = q.get()
            sc, eg, go, tg, vi, cx, cg = (None if x is None else x.to(dev, non_blocking=True) for x in b)
            if not amp:                                          # fp32 path (CPU tests): bf16 bits widened exactly
                sc, vi = sc.float(), (None if vi is None else vi.float())
            op_live = cx is not None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                traj = decode(model, sc, eg, go)
                sx = model.score_trajectories(cx, vi) if op_live else None
            traj = traj.float()
            if slow_cfg is not None:
                l_traj, widx = MS.anchored_wta_loss(traj, tg, eg, slow_cfg, wta_fn=lambda t_, g_, yaw_w=0.1:
                                                    MS.wta_loss_yaw(t_, g_, yaw_w, a.yaw_loss, wta_fn=wta_loss),
                                                    info=info, raw_reserved=model.slow_slots.last_raw,
                                                    yaw_mode=a.yaw_loss)
            else:
                l_traj, widx = MS.wta_loss_yaw(traj, tg, mode=a.yaw_loss, wta_fn=wta_loss)
            l_score = score_loss(sx, cg, sc.shape[0], cov_norm, a.score_w, sco_params)
            n_cov_total += int(cx.shape[0]) if op_live else 0
            cov_micro.append(int(cx.shape[0]) if op_live else 0)
            ((l_traj + l_score) / a.accum).backward()
            acc["l_traj"] += float(l_traj) / a.accum
            acc["l_score"] += float(l_score) / a.accum
        if a.dump_grads_after_accum:                             # TEST ONLY: the accumulated, UNclipped gradients
            torch.save({"grads": {n: p_.grad.detach().clone() for n, p_ in model.named_parameters()
                                  if p_.requires_grad and p_.grad is not None},
                        "no_grad": [n for n, p_ in model.named_parameters() if p_.requires_grad and p_.grad is None],
                        "batch": a.batch, "accum": a.accum, "covered_per_micro_batch": cov_micro, "cov_norm": cov_norm,
                        "losses": acc}, a.dump_grads_after_accum)
            print("ZZDUMPGRADS_OK")
            return 0
        if a.clip_split:
            gt, gs = MS.clip_grad_norm_split(grps, 1.0, 1.0)
            gn = {"gn_traj": float(gt), "gn_score": float(gs)}
        else:
            gn = {"gn_all": float(torch.nn.utils.clip_grad_norm_(params, 1.0))}
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % a.log_every == 0 or step == total_steps - 1:
            with torch.no_grad():
                h = traj[..., 19, 2]
                wi = widx
                ar = torch.arange(traj.shape[0], device=traj.device)
                werr = MS.wrap_angle(traj[ar, wi, 19, 2] - tg[:, 19, 2]).abs()
            row = {"step": step, "lr": lr, **acc, **gn, "covered_samples_so_far": n_cov_total,
                   "frac_h19_beyond_pi": float((h.abs() > math.pi).float().mean()),
                   "winner_h19_err_median": float(werr.median()),
                   "cuda_max_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else None,
                   "sec_per_step": (time.time() - t_last) / max(
                       1, (a.log_every if step else 1)), "elapsed_s": time.time() - t0}
            t_last = time.time()
            mlog.write(json.dumps(row) + "\n")
            mlog.flush()
            print(f"    step {step:4d} lr {lr:.3e} l_traj {acc['l_traj']:.4f} l_score {acc['l_score']:.4f} "
                  f"{' '.join(f'{k} {v:.3f}' for k, v in gn.items())} h19>pi {row['frac_h19_beyond_pi']:.3f} "
                  f"werr19 {row['winner_h19_err_median']:.3f}  {row['elapsed_s']:.0f}s", flush=True)
    mlog.close()
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith(DEC + SCO)}
    done = stopped is None
    torch.save({"state": state, "complete": done, "steps_done": step + (1 if done else 0), "stopped": stopped,
                "config": cfg_out, "slow_slots": slow_cfg.to_meta() if slow_cfg else None, "yaw_loss": a.yaw_loss},
               out / ("final.pt" if done else "partial.pt"))
    changed = sum(int(not torch.equal(state[k], snap_state[k])) for k in state)
    json.dump({"complete": done, "steps_done": step + (1 if done else 0), "stopped": stopped, "tensors": len(state),
               "tensors_changed": changed, "covered_samples": n_cov_total,
               "cuda_max_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else None,
               "samples": (step + (1 if done else 0)) * a.accum * a.batch, "seconds": round(time.time() - t0, 1),
               "yaw_loss": a.yaw_loss, "finished_local": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(out / "summary.json", "w"), indent=1)
    print(f"  {'COMPLETE' if done else 'STOPPED (' + stopped + ')'}; {changed}/{len(state)} tensors differ from the "
          f"snapshot; {time.time() - t0:.0f} s")
    print("ZZPROXYTRAIN_OK" if done else "ZZPROXYTRAIN_STOPPED")
    return 0 if done else 5


if __name__ == "__main__":
    sys.exit(main())
