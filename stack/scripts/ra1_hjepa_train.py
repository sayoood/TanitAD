#!/usr/bin/env python3
"""RA-1 trainer: ONE arm (R0 / A0 / A1 / A2 / A1x / A1h) on the pooled frozen-DINOv3 cache.  Spec: Project Steering/PREREG_HJ_STAGE0.md.

    python stack/scripts/ra1_hjepa_train.py --arm A1 --beta 0.08 --gamma 100 --seed 0 --grid 4x10 --holdout-n 60 \
        --cache D:/tanitad/cache --p10-prior <p10_prior.pt> --out D:/tanitad/runs/A1_b0.08_g100_s0 --steps 6000

HOLDOUT (PREREG_HJ_STAGE0 Amendment 1): `--holdout-n` is REQUIRED -- argparse refuses a run without it, because a flag whose default
would disable the holdout must be NAMED. The last N of the first 600 clips in sha12 order (`ra1_pooled.holdout_clips`) are withheld
from training for EVERY arm; the evaluator scores them and `verdict` selects the A1 cell on them -- never on the eval-141 windows
that score K. The clip list and its sha256 go into `<out>/meta.json`, `DONE` and every checkpoint.

LAUNCH-GATE SPIRIT (CLAUDE.md, 2026-09-26): before the first optimiser step the script runs
  G-HYG   `validate_config`: a flag whose default disables its mechanism must be NAMED (A1 without --gamma is refused, A0 with
          --gamma is refused, R0 with --beta is refused, --hist-drop outside A1h is refused);
  G-DVB   every declared lever read BACK from the built modules (`declared_vs_built`);
  G-LIVE  one REAL batch forward/backward: every declared loss term present and finite, the IDM term's gradient into the adapter
          non-zero in IDM arms and EXACTLY 0.0 in A0, no unexpected dead parameter, A1x really permutes;
and writes `launch_check.json`. Any failure => exit 2, no training. `--check-only` stops there.
After training, `ra1_hjepa_eval.py` re-audits the optimiser state (a parameter with no Adam state never received a gradient).

RESUMABLE: `latest.pt` every --ckpt-every steps (atomic), `final.pt` + `DONE` at the end. Batches, SIGReg directions, the A1x
permutation and the A1h draws are pure functions of (seed, step), so a resumed run is the uninterrupted run. A resume with a
different configuration is refused. CPU only when CUDA_VISIBLE_DEVICES=-1 (asserted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tanitad.data import ra1_pooled as RD  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import ra1_hjepa as R  # noqa: E402
from tanitad.models.v8f_mini import load_p10  # noqa: E402

NOT_IN_HASH = {"out", "resume", "log_every", "ckpt_every", "device", "check_only", "no_resume", "est_vram_only"}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=R.ARMS)
    ap.add_argument("--beta", type=float, default=None, help="LEVEL-1 SIGReg weight; required for every WM arm (A0/A1/A2/A1x/A1h), forbidden for R0")
    ap.add_argument("--gamma", type=float, default=None, help="LEVEL-1 IDM weight; required (>0) for A1/A2/A1x/A1h, absent/0 for A0, forbidden for R0")
    ap.add_argument("--beta-up", type=float, default=0.16, help="upper-level SIGReg weight (paper DROID level 2: 0.16)")
    ap.add_argument("--gamma-up", type=float, default=50.0, help="upper-level IDM weight (paper DROID level 2: 50); 0 forced in A0")
    ap.add_argument("--act-sigreg", type=float, default=0.005)
    ap.add_argument("--hist-drop", type=float, default=0.0, help="A1h ONLY: per-(window,position) history-dropout probability, in (0,1]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", default="4x10", choices=sorted(R.GRIDS))
    ap.add_argument("--cache", required=True, help="cache root holding train/ (and eval/) from pull_encode_pooled.py")
    ap.add_argument("--p10-prior", required=True, help="p10_prior.pt (v8f_mini_fit_p10.py, steer unit): the FROZEN residual prior")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clips", type=int, default=0, help="train clips = first N by sha12 (0 = all in the manifest), MINUS the holdout")
    ap.add_argument("--holdout-n", type=int, required=True,
                    help="REQUIRED (PREREG Amendment 1; the queue passes 60): the last N of the first --holdout-pool clips in sha12 order are "
                         "withheld from training for EVERY arm and are the A1 selection set")
    ap.add_argument("--holdout-pool", type=int, default=RD.STAGE_A_CLIPS,
                    help="the stage-A train pool the holdout is cut from (first N in sha12 order); 600 = PREREG, tests shrink it")
    ap.add_argument("--chunk-clips", type=int, default=0, help="0 = all clips resident; else a seeded subset resident, redrawn every --chunk-steps")
    ap.add_argument("--chunk-steps", type=int, default=500)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64, help="segments per step (SIGReg's N); each is --t1 frames")
    ap.add_argument("--t1", type=int, default=19, help="frames per segment: 19 = 3.6 s = 6 level-2 and 2 level-3 transitions of s=3")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--upper-lr-mult", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--w-traj", type=float, default=10.0)
    for k, v in (("d-tok", 64), ("adapter-hidden", 512), ("d1", 128), ("d-up", 128), ("macro-dim", 16), ("enc-hidden", 384),
                 ("pred-width", 192), ("pred-depth", 4), ("pred-heads", 6), ("context", 3), ("stride", 3),
                 ("readout-hidden", 256), ("sigreg-proj", 1024)):
        ap.add_argument(f"--{k}", type=int, default=v)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="run G-HYG/G-DVB/G-LIVE + the VRAM estimate and stop")
    ap.add_argument("--eval-cache-disjoint", default=None, help="eval cache root: assert train/eval clip-id disjointness")
    return ap


def config_from_args(a, d_feat: int) -> R.RA1Config:
    return R.RA1Config(arm=a.arm, d_feat=d_feat, n_tok=R.GRIDS[a.grid], grid=a.grid, d_tok=a.d_tok, adapter_hidden=a.adapter_hidden,
                       d1=a.d1, d_up=a.d_up, macro_dim=a.macro_dim, enc_hidden=a.enc_hidden, pred_width=a.pred_width,
                       pred_depth=a.pred_depth, pred_heads=a.pred_heads, context=a.context, stride=a.stride, t1=a.t1,
                       readout_hidden=a.readout_hidden, sigreg_proj=a.sigreg_proj, beta=a.beta, gamma=a.gamma,
                       beta_up=a.beta_up, gamma_up=a.gamma_up, act_sigreg=a.act_sigreg, w_traj=a.w_traj, hist_drop=a.hist_drop)


def args_hash(a) -> str:
    d = {k: v for k, v in sorted(vars(a).items()) if k not in NOT_IN_HASH}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def lr_factor(step: int, steps: int, warmup: int, min_frac: float) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(steps - warmup, 1)
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def save_atomic(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def pick_device(want: str):
    if want == "cpu":
        return torch.device("cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        assert not torch.cuda.is_available(), "CUDA_VISIBLE_DEVICES=-1 but CUDA is available"
        if want == "cuda":
            raise SystemExit("--device cuda requested but CUDA_VISIBLE_DEVICES=-1")
        return torch.device("cpu")
    if want == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_gen(device, seed, step, salt=0):
    return torch.Generator(device=device).manual_seed((seed * 1_000_003 + step * 7 + salt) % (2 ** 63))


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    device = pick_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t_boot = time.time()
    prior = load_p10(a.p10_prior)
    train = RD.list_clips(a.cache, "train")
    try:
        names, hold = RD.split_train_holdout([c for c, _ in train], a.n_clips, a.holdout_n, a.holdout_pool)
    except ValueError as e:                                   # G-HYG: holdout < 1 / holdout >= pool / pool larger than the manifest
        print(f"[G-HYG] REFUSED: {e}", flush=True)
        return 2
    holdout_rec = dict(n=a.holdout_n, pool=a.holdout_pool, clips=hold, sha256=RD.clips_sha256(hold), n_clips_flag=a.n_clips,
                       train_n=len(names), train_sha256=RD.clips_sha256(names))
    if a.eval_cache_disjoint:
        ev = [c for c, _ in RD.list_clips(a.eval_cache_disjoint, "eval")]
        print("[disjoint]", json.dumps(D.assert_disjoint(names, ev)), json.dumps(D.assert_disjoint(hold, ev)), flush=True)
    ahash = args_hash(a)
    latest = os.path.join(a.out, "latest.pt")
    ck = None
    if os.path.exists(latest) and not a.no_resume:
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if ck["args_hash"] != ahash:
            raise SystemExit(f"REFUSED: {latest} was written with a different configuration ({ck['args_hash']} != {ahash}). "
                             f"Use a new --out, or --no-resume to overwrite.")
        if ck.get("holdout") != holdout_rec:
            raise SystemExit(f"REFUSED: {latest} was trained on a different holdout/train split than the manifest gives now "
                             f"({(ck.get('holdout') or {}).get('sha256')} != {holdout_rec['sha256']}).")
    start = int(ck["step"]) if ck else 0
    if os.path.exists(os.path.join(a.out, "DONE")) and start >= a.steps:
        print(f"[ra1] {a.out} already DONE at step {start}")
        return 0

    # the run's record of what it withheld (PREREG Amendment 1): written BEFORE the gate, so even a refused launch names its split
    meta = dict(schema="ra1-run-meta-v1", arm=a.arm, seed=a.seed, args_hash=ahash, cache=str(a.cache),
                holdout=dict(holdout_rec, disjoint_from_train=not (set(hold) & set(names)),
                             rule="the last holdout_n of the first holdout_pool clips in sha12 order, withheld from training for EVERY arm "
                                  "(PREREG_HJ_STAGE0 Amendment 1); the A1 cell is selected on it, never on eval-141"),
                train=dict(n_clips_flag=a.n_clips, n=len(names), sha256=holdout_rec["train_sha256"]))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"[ra1] holdout: {len(hold)} clips withheld from training (sha256 {holdout_rec['sha256'][:12]}); "
          f"{len(names)} train clips (sha256 {holdout_rec['train_sha256'][:12]})", flush=True)

    bank = RD.ClipBank(a.cache, "train", prior, a.grid)
    resident = RD.chunk_clips(names, a.seed, start, a.chunk_clips, a.chunk_steps)
    bank.load(resident)
    d_feat = bank.clips[resident[0]]["pool"].shape[-1]
    cfg = config_from_args(a, d_feat)
    if ck:
        assert ck["cfg"] == cfg.to_json(), "checkpoint cfg != argv cfg"
        act_mean, act_std = ck["model"]["act_mean"], ck["model"]["act_std"]
    else:
        act_mean, act_std = bank.act_stats(resident)
    try:
        model = R.build_model(cfg, act_mean, act_std)          # G-HYG happens inside (validate_config)
    except ValueError as e:
        print(f"[G-HYG] REFUSED: {e}", flush=True)
        return 2
    model.to(device)
    if ck:
        model.load_state_dict(ck["model"])
    groups = R.optimizer_groups(model, a.lr, a.upper_lr_mult, a.wd)
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups)
    param_names = {id(p): n for n, p in model.named_parameters()}
    names_in_order = [param_names[id(p)] for g in opt.param_groups for p in g["params"]]
    if ck:
        opt.load_state_dict(ck["opt"])
        for g, saved in zip(opt.param_groups, groups):
            g["base_lr"] = saved["base_lr"]

    # ------------------------------------------------------------------ launch gate
    probe = RD.to_device(RD.sample_batch(bank, resident, a.seed, 10 ** 9, min(a.batch, 32), a.t1), device)
    dvb = R.declared_vs_built(cfg, model)
    live = R.live_report(model, probe, seed=a.seed)
    model.zero_grad(set_to_none=True)
    est = R.estimate_vram_gb(model, RD.to_device(RD.sample_batch(bank, resident, a.seed, 10 ** 9 + 1, a.batch, a.t1), device))
    gate = dict(cfg=cfg.to_json(), cfg_sha12=cfg.sha12(), args_hash=ahash, G_DVB=dvb, G_LIVE=live, vram_estimate=est,
                ok=bool(all(r["ok"] for r in dvb) and live["ok"]))
    json.dump(gate, open(os.path.join(a.out, "launch_check.json"), "w"), indent=1, default=str)
    print(f"[ra1] arm={a.arm} cfg={cfg.sha12()} params={est['params']:,} device={device} resident_clips={len(resident)} "
          f"tokens={bank.token_gb():.2f} GB(fp8)", flush=True)
    print(f"[ra1] VRAM ESTIMATE {est['estimate_gb']:.2f} GB (params {est['params_gb']:.2f} + activations {est['activations_gb']:.2f} "
          f"+ token buffers {est['token_buffers_gb']:.2f}, x{est['slack']} slack) -- an estimate; the measured peak is logged", flush=True)
    print(f"[ra1] launch gate: G-DVB {sum(r['ok'] for r in dvb)}/{len(dvb)}  G-LIVE ok={live['ok']} "
          f"(idm grad into adapter={live['idm_term_grad_into_adapter']}, unexpected dead={live['unexpected_dead']})", flush=True)
    if not gate["ok"]:
        print("[ra1] LAUNCH GATE FAILED -- not training:", json.dumps([r for r in dvb if not r["ok"]] + [live], default=str)[:1500], flush=True)
        return 2
    if a.check_only:
        print("[ra1] --check-only: gate passed")
        return 0

    # ------------------------------------------------------------------ train
    log_path = os.path.join(a.out, "train_log.jsonl")
    acc: dict = {}
    n_acc = 0
    t_log = time.time()
    model.train()
    for step in range(start, a.steps):
        if a.chunk_clips and step > start and step % a.chunk_steps == 0:
            resident = RD.chunk_clips(names, a.seed, step, a.chunk_clips, a.chunk_steps)
            bank.keep(resident)
            bank.load(resident)
        batch = RD.to_device(RD.sample_batch(bank, resident, a.seed, step, a.batch, a.t1), device)
        f = lr_factor(step, a.steps, a.warmup, a.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        model.set_generators(torch.Generator().manual_seed(a.seed * 999_983 + step))
        out = model.forward_batch(batch, generator=make_gen(device, a.seed, step),
                                  shuffle_generator=torch.Generator().manual_seed(a.seed * 1_299_709 + step))
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
        if not torch.isfinite(gn):
            raise SystemExit(f"non-finite gradient norm at step {step}: refusing to step")
        opt.step()
        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                acc[k] = acc.get(k, 0.0) + float(v.detach())
        acc["grad_norm"] = acc.get("grad_norm", 0.0) + float(gn)
        n_acc += 1
        done = step + 1
        if done % a.log_every == 0 or done == a.steps:
            row = {k: v / n_acc for k, v in acc.items()}
            row.update(step=done, lr=opt.param_groups[0]["lr"], s_per_step=(time.time() - t_log) / n_acc,
                       note="s_per_step is divided by the steps in the window (the trainer-log step_s trap)")
            if device.type == "cuda":
                row["cuda_peak_gb_MEASURED"] = torch.cuda.max_memory_allocated() / 2 ** 30
            with open(log_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print("[ra1] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                                       if k in ("step", "total", "traj", "ade_m", "idm_term/L1", "sigreg/L1", "pred/L1", "s_per_step",
                                                "cuda_peak_gb_MEASURED")), flush=True)
            acc, n_acc, t_log = {}, 0, time.time()
        if done % a.ckpt_every == 0 and done < a.steps:
            save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), step=done, cfg=cfg.to_json(), args_hash=ahash,
                             args=vars(a), param_names=names_in_order, holdout=holdout_rec), latest)
    final = dict(model=model.state_dict(), opt=opt.state_dict(), step=a.steps, cfg=cfg.to_json(), args_hash=ahash,
                 args=vars(a), param_names=names_in_order, holdout=holdout_rec)
    save_atomic(final, os.path.join(a.out, "final.pt"))
    save_atomic(final, latest)
    json.dump(dict(step=a.steps, cfg_sha12=cfg.sha12(), args_hash=ahash, seconds=time.time() - t_boot,
                   holdout_sha256=holdout_rec["sha256"], holdout_n=len(hold), train_sha256=holdout_rec["train_sha256"],
                   hist_drop_fraction=(None if not cfg.spec.hist_drop else
                                       sum(p.n_dropped for p in model.stack.predictors) / max(1, sum(p.n_positions for p in model.stack.predictors)))),
              open(os.path.join(a.out, "DONE"), "w"))
    print(f"[ra1] DONE {a.arm} step {a.steps} in {time.time() - t_boot:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
