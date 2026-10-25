"""HiCAP training loop on cached features (paper sec:train, curriculum). Three stages; stage 2/3 mix 40 % stage-1 replay batches;
AdamW (no weight decay), lr = 2e-3 sqrt(1024/width) sqrt(batch/1024), one-cycle with 10 % warm-up per stage; seeds are explicit.
`step_s` logged is ACCUMULATED over log_every then divided (CLAUDE.md trap). Run: python -m tanitad.hicap.train --help"""
from __future__ import annotations
import argparse, json, math, os, time, hashlib
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from . import vocab as V
from .model import HiCAP, HiCAPConfig
from .losses import HiCAPLoss, VocabTensors, STAGES, lr_rule
from .data import HiCAPWindows, collate, targets_from_distances, compute_priors
from .full import HiCAPFull, FullLoss


@dataclass
class TrainCfg:
    steps: tuple = (2000, 1000, 1000)           # per stage
    batch: int = 256                            # paper: 2,048 on cached tensors; smaller here for RAM, lr rule rescales
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 4
    log_every: int = 50
    ckpt_dir: str = "runs/hicap"
    replay: float = 0.4
    sigma: float = 0.5
    clip_grad: float = 1.0
    amp: bool = False
    stages: tuple = (1, 2, 3)
    stage3_unfreeze_only: bool = False


def onecycle(step, total, lr, warm=0.1):
    w = max(1, int(total * warm))
    return lr * (step + 1) / w if step < w else lr * 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def batches(ds, batch, workers, seed, replay_ds=None, replay=0.0):
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers, collate_fn=collate, generator=g, persistent_workers=workers > 0)
    rl = DataLoader(replay_ds, batch_size=max(1, int(batch * replay)), shuffle=True, drop_last=True, num_workers=workers, collate_fn=collate, generator=g) if replay_ds is not None and replay > 0 else None
    rit = iter(rl) if rl else None
    while True:
        for b in dl:
            if rit is not None:
                try: r = next(rit)
                except StopIteration: rit = iter(rl); r = next(rit)
                n = int(len(b["k"]) * (1 - replay)); b = {k: torch.cat([b[k][:n], r[k][: len(b[k]) - n]]) for k in b}
            yield b


def train(model: HiCAP, Vt: VocabTensors, vocab: V.FactoredVocab, ds: HiCAPWindows, cfg: TrainCfg, priors: dict | None = None, log=print):
    seed_all(cfg.seed); dev = cfg.device
    model.to(dev); Vt.to(dev)
    crit = FullLoss() if isinstance(model, HiCAPFull) else HiCAPLoss()
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    hist = []
    stage1_ds = ds
    for stage, n_steps in zip(cfg.stages, cfg.steps):
        st = STAGES[stage]
        params = [p for p in model.parameters() if p.requires_grad]
        if stage < 3:                                  # re-ranker is trained only in stage 3
            for p in model.rerank.parameters(): p.requires_grad_(False)
        else:
            for p in model.rerank.parameters(): p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
        lr = lr_rule(model.c.d, cfg.batch)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        it = batches(ds, cfg.batch, cfg.workers, cfg.seed + stage, stage1_ds if st.replay > 0 else None, st.replay if st.replay > 0 else 0.0)
        t_acc = 0.0; model.train()
        for step in range(n_steps):
            t0 = time.time()
            for g in opt.param_groups: g["lr"] = onecycle(step, n_steps, lr)
            b = targets_from_distances(next(it), cfg.sigma, vocab=vocab if st.lam["ref"] > 0 else None)
            b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
            with torch.autocast(dev if dev != "cpu" else "cpu", dtype=torch.bfloat16, enabled=cfg.amp):
                total, parts = crit(model, Vt, b, stage=stage)
            opt.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad); opt.step()
            t_acc += time.time() - t0
            if (step + 1) % cfg.log_every == 0 or step == n_steps - 1:
                rec = dict(stage=stage, step=step + 1, loss=float(total), step_s=t_acc / cfg.log_every, **{k: float(v) for k, v in parts.items()})
                hist.append(rec); log(json.dumps(rec)); t_acc = 0.0
        ck = os.path.join(cfg.ckpt_dir, f"hicap_stage{stage}_seed{cfg.seed}.pt")
        torch.save(dict(model=model.state_dict(), cfg=asdict(model.c), train_cfg=asdict(cfg), stage=stage, hist=hist,
                        vocab_md5=hashlib.md5(vocab.paths.tobytes() + vocab.profiles.tobytes()).hexdigest()), ck)
        log(f"saved {ck}")
    return hist


def main():
    ap = argparse.ArgumentParser(description="HiCAP training on cached features (v9 release)")
    ap.add_argument("--cache-dir", required=True); ap.add_argument("--v9-npz", required=True); ap.add_argument("--vocab-npz", required=True,
        help="factored vocabulary built from TRAIN windows only (see build_hicap_vocab in scripts)")
    ap.add_argument("--text-npz", required=True, help="cached text-tower embeddings of node definitions")
    ap.add_argument("--out", default="runs/hicap"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs=3, default=[2000, 1000, 1000]); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d-v", type=int, default=2048); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from .data import V9Rows
    rows = V9Rows(a.v9_npz); vz = np.load(a.vocab_npz)
    vocab = V.FactoredVocab(vz["paths"], vz["path_cls"], vz["profiles"], vz["prof_cls"], vz["standstill"])
    ds = HiCAPWindows(a.cache_dir, rows, vocab, train=True)
    c = HiCAPConfig(d_v=a.d_v); model = HiCAP(c)
    t = np.load(a.text_npz)
    pf = np.stack([V.path_features(p) for p in vocab.paths]); qf = np.stack([V.profile_features(d) for d in vocab.profiles])
    pri = compute_priors(ds)
    Vt = VocabTensors(t["strat_text"], t["strat_args"], t["lat_text"], t["lat_args"], t["lon_text"], t["lon_args"], pf, vocab.path_cls, qf, vocab.prof_cls,
                      logq=pri, strat_mask=t["strat_mask"], cell_bias=t["cell_bias"])
    train(model, Vt, vocab, ds, TrainCfg(steps=tuple(a.steps), batch=a.batch, seed=a.seed, ckpt_dir=a.out, workers=a.workers), pri)


if __name__ == "__main__":
    main()
