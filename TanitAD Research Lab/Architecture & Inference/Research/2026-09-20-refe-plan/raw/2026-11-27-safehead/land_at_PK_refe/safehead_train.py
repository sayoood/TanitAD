#!/usr/bin/env python3
"""L2 SAFE-HEAD: the HEAD-ONLY trainer (PREREG_SAFEHEAD.md 2-3, PK/raw/2026-11-27-safehead/).  CPU, seconds per run, no GPU, no trunk.

It trains ONLY rows 0, 1, 3, 5 (NC, DAC, TTC, DDC) of C1's `score_head = nn.Linear(256, 6)` (model.py:563) on the cached post-decoder features `s` [64, 256]
(the exact input of the head, model.py:706-709), initialised from C1's own weights, with ONE loss: the quality-focal BCE (Li et al. 2020; for 0/1 labels it is the
focal loss) on the TRACKED NAVSIM v1.1 labels.  Rows 2 (EP) and 4 (C) are never parameters, and no other tensor of C1 is ever loaded.  At the end it applies the
card's pre-registered held-out rule (4.3) to the saved steps.  It never reads a navtest token.

    # a REAL run: refused unless TANITAD_SH_GATE_TOKEN names a PASS token that verifies for this exact argv, tree and data (CLAUDE.md, PI 2026-09-26)
    TANITAD_SH_GATE_TOKEN=<PASS_*.json> python refe/safehead_train.py --cache-index <run>/index_sealed.csv --init-from <C1 model_final.pt> \
        --heldout-logs splits/sft_heldout_logs.txt --navtest-logs <run>/navtest_logs.txt --out <run>/sh_s0 --seed 0 --steps 2000 --lr 0.0003 \
        --batch 32 --gamma 2.0 --rows 0,1,3,5 --save-steps 0,250,500,1000,2000
    # the gate's own runs (never a training launch: their heads are marked smoke and every read refuses them)
    python refe/safehead_train.py <the argv> --build-only        # G-DVB: build, write built.json, exit before step 1
    python refe/safehead_train.py <the argv> --smoke-steps 5     # G-LIVE / G-EVAL / G-CKPT
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import safehead_data as D  # noqa: E402

ARM_ENV = "TANITAD_GATE_ARM"                 # the launch gate's deliberate-regression arm (gate_checks.ARM_ENV); honoured ONLY in --build-only / --smoke-steps
TOKEN_ENV = "TANITAD_SH_GATE_TOKEN"
LOSS_NAME = "qfocal"
OPT = {"name": "Adam", "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}     # registered (card 3); not flags
LIVE_TERMS = ("qfl_nc", "qfl_dac", "qfl_ttc", "qfl_ddc")                            # one per trained row; their mean is the loss
TERM_OF_ROW = {0: "qfl_nc", 1: "qfl_dac", 3: "qfl_ttc", 5: "qfl_ddc"}
FEAT_MAX, FEAT_P99 = 1e-4, 1e-4         # CTRL-FEAT: C1's head on the cached float32 s reproduces the cached logits (the data agent MEASURED 7.6e-6; a wrong tensor is off by O(1))


@dataclasses.dataclass(slots=True)
class SafeHeadConfig:
    """Every lever of a run.  slots=True: an undeclared attribute is REFUSED (G-HYG)."""
    cache_index: str
    init_from: str
    heldout_logs: str
    navtest_logs: str
    out: str
    seed: int
    steps: int
    lr: float
    batch: int
    gamma: float
    rows: tuple
    save_steps: tuple
    permute_labels_seed: int
    smoke_steps: int
    build_only: bool


def _ints(s: str) -> tuple:
    return tuple(int(x) for x in str(s).split(",") if x.strip() != "")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-index", required=True, help="the SEALED index (safehead_data.py seal), never the raw one")
    ap.add_argument("--init-from", required=True, help="C1's model_final.pt (sha256 pinned by the profile)")
    ap.add_argument("--heldout-logs", required=True, help="splits/sft_heldout_logs.txt")
    ap.add_argument("--navtest-logs", required=True, help="the 136 navtest logs (safehead_read.py prep), for CTRL-LEAK")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--batch", type=int, required=True, help="tokens per step (each with all 64 proposals)")
    ap.add_argument("--gamma", type=float, required=True, help="the quality-focal exponent")
    ap.add_argument("--rows", required=True, help="the trained score_head rows, e.g. 0,1,3,5")
    ap.add_argument("--save-steps", required=True, help="steps whose head is saved; the held-out rule chooses among them")
    ap.add_argument("--permute-labels-seed", type=int, default=-1, help="CTRL-RED ONLY: permute the label rows across train tokens (-1 = off)")
    ap.add_argument("--smoke-steps", type=int, default=0, help="GATE ONLY (forbidden in a launch): train this many steps; heads marked smoke")
    ap.add_argument("--build-only", action="store_true", help="GATE ONLY (forbidden in a launch): build, write built.json, exit")
    return ap


def build_config(a) -> SafeHeadConfig:
    return SafeHeadConfig(cache_index=a.cache_index, init_from=a.init_from, heldout_logs=a.heldout_logs, navtest_logs=a.navtest_logs, out=a.out,
                          seed=a.seed, steps=a.steps, lr=a.lr, batch=a.batch, gamma=a.gamma, rows=_ints(a.rows), save_steps=_ints(a.save_steps),
                          permute_labels_seed=a.permute_labels_seed, smoke_steps=a.smoke_steps, build_only=bool(a.build_only))


def qfocal_terms(z, y, gamma: float):
    """Quality-focal BCE per output column: mean over (token, proposal) of |y - sigmoid(z)|^gamma * BCE(z, y).  [B, 64, R] -> [R]."""
    import torch
    import torch.nn.functional as F
    p = torch.sigmoid(z)
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    return ((y - p).abs().pow(gamma) * bce).mean(dim=(0, 1))


def require_token(argv: list) -> dict:
    """The trainer's own refusal: a real run starts only on a PASS token that verifies for THIS argv, tree and data (gate_token.verify_token)."""
    import gate_token as GT
    import safehead_gate as SHG
    tok = os.environ.get(TOKEN_ENV)
    if not tok:
        raise SystemExit(f"REFUSED: no launch-gate token ({TOKEN_ENV} unset). Run `launch_gate.py run --profile <refe_safehead_*>` first (CLAUDE.md, PI 2026-09-26)")
    prof = SHG.profile_name_for_argv(argv)
    key = GT.load_key(None, create=False)
    ok, reasons, info = GT.verify_token(tok, list(argv), HERE.parent, key=key, profile=prof, data_flags=SHG.DATA_FLAGS)
    if not ok:
        raise SystemExit("REFUSED: the launch-gate token does not verify: " + " | ".join(r[:200] for r in reasons[:4]))
    return {"token": tok, "token_sha256": D.sha256_file(tok), "profile": prof, "binding": info.get("binding")}


def run(argv: list) -> int:
    import torch
    a = build_parser().parse_args(argv)
    cfg = build_config(a)
    gate_mode = bool(cfg.smoke_steps or cfg.build_only)
    arm = os.environ.get(ARM_ENV) if gate_mode else None
    if not gate_mode:
        if os.environ.get(ARM_ENV):
            raise SystemExit(f"REFUSED: a regression arm is active ({os.environ[ARM_ENV]}): a training launch never runs under one")
        tok = require_token(argv)
    else:
        tok = None
    if sorted(cfg.rows) != sorted(set(cfg.rows)) or not set(cfg.rows) <= set(D.SAFETY_ROWS) or not cfg.rows:
        raise SystemExit(f"REFUSED: --rows {cfg.rows}: only the safety rows {D.SAFETY_ROWS} may train (EP/C rows {D.FROZEN_ROWS} are never parameters)")
    if 0 not in cfg.save_steps:
        raise SystemExit("REFUSED: --save-steps must include 0 (the untrained head: CTRL-ID and the held-out rule's step 0)")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"REFUSED: --out {out} is not empty (a run never writes over another)")
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    W0, b0 = D.c1_head_from_checkpoint(cfg.init_from)
    init_sha = D.sha256_file(cfg.init_from)
    c = D.load_cache(cfg.cache_index, splits=D.SPLITS_NAVTRAIN, require_sealed=True)
    leak = D.leak_problems(c, D.read_log_list(cfg.heldout_logs), D.read_log_list(cfg.navtest_logs))
    if leak:
        raise SystemExit("REFUSED (CTRL-LEAK): " + " | ".join(leak))
    dl = np.abs(D.direct_logits(c["s"], W0, b0) - c["logits"])
    feat = {"max_abs": float(dl.max()), "p99_abs": float(np.percentile(dl, 99)), "bar_max": FEAT_MAX, "bar_p99": FEAT_P99}
    if feat["max_abs"] > FEAT_MAX or feat["p99_abs"] > FEAT_P99:
        raise SystemExit(f"REFUSED (CTRL-FEAT): C1's head on the cached s is not the cached logits ({feat}): s is not the head's input, or init-from is not the cache's model")
    tr = np.where(c["split"] == "train")[0]
    ho = np.where(c["split"] == "heldout")[0]
    if len(tr) < cfg.batch or len(ho) == 0:
        raise SystemExit(f"REFUSED: {len(tr)} train / {len(ho)} held-out tokens (batch {cfg.batch})")
    rows = list(cfg.rows)
    Y = np.stack([c[D.ROW_LABEL[r]][tr] for r in rows], -1).astype(np.float32)                 # [Ntr, 64, R]
    n_fixed = None
    if cfg.permute_labels_seed >= 0:                                                            # CTRL-RED: every token gets another token's 64 label rows
        perm = np.random.default_rng(cfg.permute_labels_seed).permutation(len(tr))
        n_fixed = int((perm == np.arange(len(tr))).sum())
        Y = Y[perm]
    S = torch.from_numpy(np.ascontiguousarray(c["s"][tr]))
    YT = torch.from_numpy(Y)
    # the parameters: ONE weight row + ONE bias per trained row; the frozen rows are leaves OUTSIDE the optimizer (their grad is the zero control)
    params = {r: (torch.nn.Parameter(torch.from_numpy(W0[r].copy())), torch.nn.Parameter(torch.from_numpy(b0[r:r + 1].copy()))) for r in rows}
    frozen = {r: (torch.from_numpy(W0[r].copy()).requires_grad_(True), torch.from_numpy(b0[r:r + 1].copy()).requires_grad_(True))
              for r in range(D.N_OUT) if r not in rows}
    groups = [{"params": list(params[r]), "name": f"row{r}"} for r in rows]
    lr = cfg.lr * (2.0 if arm == "sh_dvb_lr_drift" else 1.0)
    if arm == "sh_reach_row_dropped":                                                           # deliberate regression: one trained row is not in the optimizer
        groups = [g for g in groups if g["name"] != f"row{rows[min(2, len(rows) - 1)]}"]
    opt = torch.optim.Adam(groups, lr=lr, betas=OPT["betas"], eps=OPT["eps"], weight_decay=OPT["weight_decay"])

    def head():
        Wr = [params[r][0] if r in params else frozen[r][0] for r in range(D.N_OUT)]
        br = [params[r][1] if r in params else frozen[r][1] for r in range(D.N_OUT)]
        return torch.stack(Wr), torch.cat(br)

    def head_np():
        with torch.no_grad():
            W, b = head()
            W, b = W.detach().numpy().copy(), b.detach().numpy().copy()
        if arm == "sh_ckpt_nondet":
            W[rows[0], 0] += np.float32(1e-3 * ((time.time_ns() // 1000) % 997 + 1))
        return W, b

    data_sha = {"init_sha256": init_sha, "index_sha256": D.sha256_file(cfg.cache_index), "heldout_logs_sha256": D.sha256_file(cfg.heldout_logs),
                "navtest_logs_sha256": D.sha256_file(cfg.navtest_logs)}
    built = {"seed": cfg.seed, "steps": cfg.steps, "lr": [g["lr"] for g in opt.param_groups], "batch": cfg.batch, "gamma": cfg.gamma,
             "rows": [int(g["name"][3:]) for g in opt.param_groups], "rows_declared": rows, "save_steps": list(cfg.save_steps),
             "permute_labels_seed": cfg.permute_labels_seed, "n_fixed_points": n_fixed, "optimizer": type(opt).__name__,
             "betas": [list(g["betas"]) for g in opt.param_groups], "eps": [g["eps"] for g in opt.param_groups],
             "weight_decay": [g["weight_decay"] for g in opt.param_groups], "loss": LOSS_NAME, "live_terms": [TERM_OF_ROW[r] for r in rows],
             "frozen_rows": sorted(frozen), "n_train": int(len(tr)), "n_heldout": int(len(ho)), "n_train_logs": int(len(set(c["log"][tr]))),
             "n_heldout_logs": int(len(set(c["log"][ho]))), "ctrl_feat": feat, "threads": torch.get_num_threads(), **data_sha,
             "gate_mode": "build" if cfg.build_only else ("smoke" if cfg.smoke_steps else None), "arm": arm}
    (out / "built.json").write_text(json.dumps(built, indent=1), encoding="utf-8")
    if cfg.build_only:
        print("ZZSH_BUILD_OK", flush=True)
        return 0
    n_steps = cfg.smoke_steps if cfg.smoke_steps else cfg.steps
    save = sorted({s for s in cfg.save_steps if s <= n_steps} | ({n_steps} if cfg.smoke_steps else set()))
    meta0 = {"rows": rows, "seed": cfg.seed, "init_sha256": init_sha, "index_sha256": data_sha["index_sha256"], "argv": list(argv),
             "smoke": bool(cfg.smoke_steps), "permute_labels_seed": cfg.permute_labels_seed, "loss": LOSS_NAME, "token": tok}
    shas = {}
    probe = {}

    def save_step(step):
        W, b = head_np()
        Wsave, bsave = W, (b0.copy() if arm == "sh_ckpt_bias_dropped" else b)
        shas[step] = D.save_head(out / f"head_step{step:05d}.npz", W=Wsave, b=bsave, W0=W0, b0=b0, meta={**meta0, "step": step})
        probe[step] = (W, b)
    if 0 in save:
        save_step(0)
    rng = np.random.default_rng(cfg.seed)
    order, pos = rng.permutation(len(tr)), 0
    mf = open(out / "metrics.jsonl", "w", encoding="utf-8")
    for step in range(1, n_steps + 1):
        if pos + cfg.batch > len(tr):
            order, pos = rng.permutation(len(tr)), 0
        idx = torch.from_numpy(order[pos:pos + cfg.batch])
        pos += cfg.batch
        W, b = head()
        z = S[idx] @ W.T + b                                                                     # [B, 64, 6]
        loss_rows = rows + ([2] if arm == "sh_reach_ep_in_loss" else [])
        yb = YT[idx]
        if arm == "sh_reach_ep_in_loss":
            yb = torch.cat([yb, torch.from_numpy(c["tr_ep"][tr])[idx].unsqueeze(-1).float()], -1)
        terms = qfocal_terms(z[..., loss_rows], yb, cfg.gamma)
        loss = terms.mean()
        opt.zero_grad(set_to_none=True)
        for w_, b_ in frozen.values():
            w_.grad, b_.grad = None, None
        loss.backward()
        rec = {"step": step, "loss": float(loss)}
        for i, r in enumerate(rows):
            if not (arm == "sh_live_term_dropped" and r == 3):
                rec[TERM_OF_ROW[r]] = float(terms[i])
        rec["grad_abs"] = {f"row{r}": float(params[r][0].grad.abs().sum() + params[r][1].grad.abs().sum()) for r in rows}
        rec["frozen_grad_abs"] = {f"row{r}": float((0.0 if w_.grad is None else w_.grad.abs().sum()) + (0.0 if b_.grad is None else b_.grad.abs().sum()))
                                  for r, (w_, b_) in frozen.items()}
        rec["finite"] = bool(np.isfinite(rec["loss"]) and all(np.isfinite(v) for v in rec["grad_abs"].values()))
        opt.step()
        mf.write(json.dumps(rec) + "\n")
        if not rec["finite"]:
            mf.close()
            raise SystemExit(f"FAILED: non-finite loss or gradient at step {step}")
        if step in save:
            save_step(step)
    mf.close()
    if cfg.smoke_steps:                                                                          # G-EVAL / G-CKPT probes: the trainer's own in-memory head and forward
        Wf, bf = probe[n_steps]
        k = min(8, len(ho))
        with torch.no_grad():
            zt = (torch.from_numpy(np.ascontiguousarray(c["s"][ho[:k]])) @ torch.from_numpy(Wf).T + torch.from_numpy(bf)).numpy()
        np.savez(out / "probe.npz", W=Wf, b=bf, ho_idx=ho[:k], torch_logits=zt, cached_logits=c["logits"][ho[:k]], s=c["s"][ho[:k]],
                 cached_pick=c["pick"][ho], ho_s=c["s"][ho], ho_logits=c["logits"][ho])
    sel = heldout_select(c, ho, out, save, W0, b0, rows)
    (out / "heads.sha256").write_text("".join(f"{shas[s]}  head_step{s:05d}.npz\n" for s in sorted(shas)), encoding="utf-8")
    (out / "run.json").write_text(json.dumps({"argv": list(argv), "built": built, "chosen_step": sel["chosen_step"], "seconds": round(time.time() - t0, 1),
                                              "token": tok, "evidence": "MEASURED (this run)"}, indent=1, default=str), encoding="utf-8")
    print(f"ZZSH_TRAIN_DONE steps {n_steps} chosen_step {sel['chosen_step']} heldout_safety_miss_rate {sel['per_step'][str(sel['chosen_step'])]['safety_miss_rate']:.4f} "
          f"({time.time() - t0:.1f} s)", flush=True)
    return 0


def heldout_metrics(c, ho, L) -> dict:
    """Held-out (navtrain, the 6 SFT held-out logs) read of logits L [Nho, 64, 6]: the pick by agg32 and its tracked labels."""
    k = D.select(L)
    ar = np.arange(len(ho))
    safe = (c["tr_nc"][ho] == 1) & (c["tr_dac"][ho] == 1) & (c["tr_ttc"][ho] == 1)
    miss = (~safe[ar, k]) & safe.any(1)
    m = {"safety_miss_rate": float(miss.mean()), "safety_misses": int(miss.sum())}      # THE selection metric: SAFETY labels only (NC, DAC, TTC)
    for nm in ("tr_nc", "tr_dac", "tr_ttc", "tr_c", "tr_ddc"):
        m[nm[3:]] = float(c[nm][ho][ar, k].mean() * 100)
    # descriptive ONLY, never fitted or selected on (the cache's EP / PDMS are a rebuilt metric cache; DATA_CONTRACT.md caveats):
    m["descriptive_cache_ep"] = float(c["tr_ep"][ho][ar, k].mean() * 100)
    m["descriptive_cache_pdms"] = float(c["tr_pdms"][ho][ar, k].mean() * 100)
    for r in (0, 1, 3, 5):
        m[f"auc_{D.ROW_NAMES[r]}"] = D.rank_auc(L[..., r], c[D.ROW_LABEL[r]][ho])
    m["pick"] = k
    return m


def heldout_select(c, ho, out: Path, save, W0, b0, rows) -> dict:
    """The card's pre-registered rule (4.3): the saved step with the LOWEST held-out SAFETY-MISS RATE (the pick fails NC, DAC or TTC while a proposal passing all three
    exists; safety labels only, never the cache's EP / PDMS); ties -> the EARLIEST step.  Step 0 must reproduce C1's cached picks on every held-out token (CTRL-ID,
    held-out side), or the run FAILS."""
    per, picks = {}, {}
    for s in save:
        h = D.load_head(out / f"head_step{s:05d}.npz")
        L = D.delta_logits(c["logits"][ho], c["s"][ho], h["W"], h["b"], W0, b0, rows)
        m = heldout_metrics(c, ho, L)
        picks[s] = m.pop("pick")
        per[str(s)] = m
    if 0 in picks and not np.array_equal(picks[0], c["pick"][ho]):
        raise SystemExit(f"FAILED (CTRL-ID held-out): the untrained head does not reproduce C1's pick on {int((picks[0] != c['pick'][ho]).sum())} held-out tokens")
    best = min(per[str(s)]["safety_miss_rate"] for s in save)
    chosen = min(s for s in save if per[str(s)]["safety_miss_rate"] == best)
    rep = {"rule": "argmin held-out safety-miss rate over --save-steps (safety labels only); ties -> earliest (PREREG_SAFEHEAD 4.3)", "n_heldout": int(len(ho)),
           "n_heldout_logs": int(len(set(c["log"][ho]))), "per_step": per, "chosen_step": int(chosen), "chosen_is_untrained": chosen == 0,
           "switch_rate_vs_c1": {str(s): float((picks[s] != c["pick"][ho]).mean()) for s in save}, "evidence": "MEASURED (held-out navtrain; NOT navtest)"}
    (out / "heldout_select.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def main(argv=None) -> int:
    return run(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
