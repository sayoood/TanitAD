#!/usr/bin/env python3
"""L2 SAFE-HEAD family: the HEAD-ONLY trainer (PREREG_SAFEHEAD.md 2-3; PREREG_SHMLP.md; PREREG_EPHEAD.md).  CPU, seconds per run, no GPU, no trunk.

SAFE-HEAD (the default flags) trains ONLY rows 0, 1, 3, 5 (NC, DAC, TTC, DDC) of C1's `score_head = nn.Linear(256, 6)` (model.py:563) on the cached post-decoder features
`s` [64, 256] (the exact input of the head, model.py:706-709), initialised from C1's own weights, with ONE loss: the quality-focal BCE (Li et al. 2020; for 0/1 labels it is
the focal loss) on the TRACKED NAVSIM v1.1 labels.  Rows 2 (EP) and 4 (C) are never parameters, and no other tensor of C1 is ever loaded.  At the end it applies the card's
pre-registered held-out rule (4.3) to the saved steps.  It never reads a navtest token.

Two flags added for the committed next arms (every default reproduces SAFE-HEAD exactly; each has its G-DVB entry in safehead_gate.py):
  --mlp-hidden 64   SH-MLP (PREREG_SHMLP, SAFE-HEAD card 8.1): the trained rows get a residual MLP 256 -> 64 -> 1 (GELU), z_r = C1_r(s) + MLP_r(s), LAST LAYER ZERO-INITIALISED
                    (step 0 = C1 bit for bit); C1's linear W, b are constants.  4 rows x (256*64 + 64 + 64 + 1) = 66,052 parameters.  0 (default) = the linear head.
  --select cv5      EP-HEAD (PREREG_EPHEAD): the saved step is chosen by 5-FOLD LOG-LEVEL CROSS-VALIDATION over ALL navtrain logs (folds = sorted log name, index mod 5; the
                    out-of-fold mean tracked PDMS of the agg32 pick, argmax, ties earliest) and the final head is trained on ALL logs; row 2 (EP) may then be the trained row
                    (--rows 2, SOFT label tr_ep).  heldout (default) = SAFE-HEAD's rule on the 6 SFT held-out logs (4.3).

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
import safehead_cvm as CVM  # noqa: E402

ARM_ENV = "TANITAD_GATE_ARM"                 # the launch gate's deliberate-regression arm (gate_checks.ARM_ENV); honoured ONLY in --build-only / --smoke-steps
TOKEN_ENV = "TANITAD_SH_GATE_TOKEN"
LOSS_NAME = "qfocal"
OPT = {"name": "Adam", "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}     # registered (card 3); not flags
TERM_OF_ROW = {0: "qfl_nc", 1: "qfl_dac", 2: "qfl_ep", 3: "qfl_ttc", 5: "qfl_ddc"}  # one term per trained row; their mean is the loss
LIVE_TERMS = ("qfl_nc", "qfl_dac", "qfl_ttc", "qfl_ddc")                            # the SAFE-HEAD arm's four terms (kept: safehead_gate.LIVE_TERMS)
FEAT_MAX, FEAT_P99 = 1e-4, 1e-4         # CTRL-FEAT: C1's head on the cached float32 s reproduces the cached logits (the data agent MEASURED 7.6e-6; a wrong tensor is off by O(1))
SELECTS = ("heldout", "cv5", "cv5m")
CV_SELECTS = ("cv5", "cv5m")                                                                   # the two 5-fold log-CV rules; cv5m = cv5 + the noise margin and guard (PREREG_DATASCALE 4.3)
N_FOLDS = 5


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
    mlp_hidden: int
    select: str
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
    ap.add_argument("--rows", required=True, help="the trained score_head rows, e.g. 0,1,3,5 (SAFE-HEAD, SH-MLP) or 2 (EP-HEAD, needs --select cv5)")
    ap.add_argument("--save-steps", required=True, help="steps whose head is saved; the selection rule chooses among them")
    ap.add_argument("--permute-labels-seed", type=int, default=-1, help="CTRL-RED ONLY: permute the label rows across train tokens (-1 = off)")
    ap.add_argument("--mlp-hidden", type=int, default=0, help="SH-MLP: hidden width of the residual MLP on each trained row (0 = the linear head, SAFE-HEAD)")
    ap.add_argument("--select", default="heldout", choices=SELECTS, help="heldout = SAFE-HEAD 4.3 (6 SFT held-out logs); cv5 = EP-HEAD 4.3 (5-fold log CV over all navtrain logs); cv5m = DATA-SCALE 4.3 (cv5 + a log-cluster CI margin and an OOF sub-score guard)")
    ap.add_argument("--smoke-steps", type=int, default=0, help="GATE ONLY (forbidden in a launch): train this many steps; heads marked smoke")
    ap.add_argument("--build-only", action="store_true", help="GATE ONLY (forbidden in a launch): build, write built.json, exit")
    return ap


def build_config(a) -> SafeHeadConfig:
    return SafeHeadConfig(cache_index=a.cache_index, init_from=a.init_from, heldout_logs=a.heldout_logs, navtest_logs=a.navtest_logs, out=a.out,
                          seed=a.seed, steps=a.steps, lr=a.lr, batch=a.batch, gamma=a.gamma, rows=_ints(a.rows), save_steps=_ints(a.save_steps),
                          permute_labels_seed=a.permute_labels_seed, mlp_hidden=a.mlp_hidden, select=a.select, smoke_steps=a.smoke_steps,
                          build_only=bool(a.build_only))


def qfocal_terms(z, y, gamma: float):
    """Quality-focal BCE per output column: mean over (token, proposal) of |y - sigmoid(z)|^gamma * BCE(z, y).  [B, 64, R] -> [R].  Soft labels (EP) are the quality form."""
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
        raise SystemExit(f"REFUSED: no launch-gate token ({TOKEN_ENV} unset). Run `launch_gate.py run --profile <refe_safehead_* | refe_shmlp_* | refe_ephead_* | refe_ds_*>` first (CLAUDE.md, PI 2026-09-26)")
    prof = SHG.profile_name_for_argv(argv)
    key = GT.load_key(None, create=False)
    ok, reasons, info = GT.verify_token(tok, list(argv), HERE.parent, key=key, profile=prof, data_flags=SHG.DATA_FLAGS)
    if not ok:
        raise SystemExit("REFUSED: the launch-gate token does not verify: " + " | ".join(r[:200] for r in reasons[:4]))
    return {"token": tok, "token_sha256": D.sha256_file(tok), "profile": prof, "binding": info.get("binding")}


# ============================================================================================ the model
class HeadModel:
    """The trainable part of a run.  Linear (mlp_hidden == 0): one weight row + one bias per trained row, in `params`.  MLP (> 0): per trained row a residual MLP
    256 -> H -> 1 (GELU) with a ZERO-INITIALISED last layer; C1's linear W, b are constants.  Rows outside `rows` are leaves OUTSIDE the optimiser (their gradient is the
    zero control).  `arm` is the launch gate's deliberate-regression arm (None in a launch)."""

    def __init__(self, rows, mlp_hidden, W0, b0, lr, arm, seed):
        import torch
        self.torch = torch
        self.rows, self.arm = list(rows), arm
        self.W0, self.b0 = W0, b0
        torch.manual_seed(seed)                                                                      # the MLP init is a function of the seed, identical in every fold
        self.mlp_hidden = 0 if arm == "sh_mlp_off" else int(mlp_hidden)                              # sh_mlp_off: the flag is ignored (G-DVB must catch it)
        self.params, self.frozen, self.layers = {}, {}, {}
        if self.mlp_hidden:
            H = self.mlp_hidden
            for r in self.rows:
                l1, l2 = torch.nn.Linear(D.MLP_IN, H), torch.nn.Linear(H, 1)
                if arm != "sh_mlp_nonzero_init":                                                     # THE zero init: step 0 is C1 exactly
                    torch.nn.init.zeros_(l2.weight)
                    torch.nn.init.zeros_(l2.bias)
                self.layers[r] = (l1, l2)
                self.params[r] = [l1.weight, l1.bias, l2.weight, l2.bias]
            self.Wc, self.bc = torch.from_numpy(W0.copy()), torch.from_numpy(b0.copy())              # C1's linear head: CONSTANTS on the trained rows
            self.frozen = {r: (torch.from_numpy(W0[r].copy()).requires_grad_(True), torch.from_numpy(b0[r:r + 1].copy()).requires_grad_(True))
                           for r in range(D.N_OUT) if r not in self.rows}
        else:
            self.params = {r: [torch.nn.Parameter(torch.from_numpy(W0[r].copy())), torch.nn.Parameter(torch.from_numpy(b0[r:r + 1].copy()))] for r in self.rows}
            self.frozen = {r: (torch.from_numpy(W0[r].copy()).requires_grad_(True), torch.from_numpy(b0[r:r + 1].copy()).requires_grad_(True))
                           for r in range(D.N_OUT) if r not in self.rows}
        groups = [{"params": list(self.params[r]), "name": f"row{r}"} for r in self.rows]
        self.lr = lr * (2.0 if arm == "sh_dvb_lr_drift" else 1.0)
        if arm == "sh_reach_row_dropped":                                                           # deliberate regression: one trained row is not in the optimizer
            groups = [g for g in groups if g["name"] != f"row{self.rows[min(2, len(self.rows) - 1)]}"]
            if not groups:                                                                          # a single-row arm (EP-HEAD): the optimiser keeps a dummy group so the build still runs
                groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "name": "row-1"}]
        self.opt = torch.optim.Adam(groups, lr=self.lr, betas=OPT["betas"], eps=OPT["eps"], weight_decay=OPT["weight_decay"])
        self.n_params = int(sum(p.numel() for g in self.opt.param_groups for p in g["params"]))
        self.last_layer_init_abs = float(sum(float(self.layers[r][1].weight.detach().abs().sum() + self.layers[r][1].bias.detach().abs().sum()) for r in self.rows)) if self.mlp_hidden else None

    def _lin(self):
        torch = self.torch
        if self.mlp_hidden:
            Wr = [self.frozen[r][0] if r in self.frozen else self.Wc[r] for r in range(D.N_OUT)]
            br = [self.frozen[r][1] if r in self.frozen else self.bc[r:r + 1] for r in range(D.N_OUT)]
        else:
            Wr = [self.params[r][0] if r in self.params else self.frozen[r][0] for r in range(D.N_OUT)]
            br = [self.params[r][1] if r in self.params else self.frozen[r][1] for r in range(D.N_OUT)]
        return torch.stack(Wr), torch.cat(br)

    def logits(self, Sb):
        """[B, 64, 256] -> [B, 64, 6]: the trainer's own forward (the head the checkpoint holds)."""
        torch = self.torch
        W, b = self._lin()
        z = Sb @ W.T + b
        if not self.mlp_hidden:
            return z
        cols = [z[..., r] for r in range(D.N_OUT)]
        for r in self.rows:
            l1, l2 = self.layers[r]
            cols[r] = cols[r] + (torch.nn.functional.gelu(Sb @ l1.weight.T + l1.bias) @ l2.weight[0] + l2.bias[0])
        return torch.stack(cols, -1)

    def state(self) -> dict:
        """numpy copies of what a head file holds: W, b and (MLP) the residual arrays."""
        torch = self.torch
        with torch.no_grad():
            if self.mlp_hidden:
                W, b = self.W0.copy(), self.b0.copy()
                mlp = {"mlp_w1": np.stack([self.layers[r][0].weight.detach().numpy() for r in self.rows]).copy(),
                       "mlp_b1": np.stack([self.layers[r][0].bias.detach().numpy() for r in self.rows]).copy(),
                       "mlp_w2": np.stack([self.layers[r][1].weight[0].detach().numpy() for r in self.rows]).copy(),
                       "mlp_b2": np.array([float(self.layers[r][1].bias[0]) for r in self.rows], np.float32)}
            else:
                W = np.stack([self.params[r][0].detach().numpy() if r in self.params else self.frozen[r][0].detach().numpy() for r in range(D.N_OUT)]).copy()
                b = np.concatenate([self.params[r][1].detach().numpy() if r in self.params else self.frozen[r][1].detach().numpy() for r in range(D.N_OUT)]).copy()
                mlp = None
        if self.arm == "sh_ckpt_nondet":
            W[self.rows[0], 0] += np.float32(1e-3 * ((time.time_ns() // 1000) % 997 + 1))
        return {"W": W, "b": b, "mlp": mlp}

    def grad_abs(self) -> dict:
        return {f"row{r}": float(sum(p.grad.abs().sum() for p in self.params[r])) for r in self.rows}

    def frozen_grad_abs(self) -> dict:
        return {f"row{r}": float((0.0 if w_.grad is None else w_.grad.abs().sum()) + (0.0 if b_.grad is None else b_.grad.abs().sum())) for r, (w_, b_) in self.frozen.items()}


def fit(M: HeadModel, S, Y, *, steps, batch, gamma, seed, save, on_save, metrics=None, extra_label=None):
    """The ONE training loop (the SAFE-HEAD one, unchanged in arithmetic).  S [N,64,256] and Y [N,64,R] torch tensors (labels of M.rows, in order), `extra_label` [N,64]
    (arm sh_reach_ep_in_loss: the label of a FROZEN row that wrongly enters the loss).  `on_save(step, state)` is called at every saved step (0 included);
    `metrics` is an open file for metrics.jsonl or None."""
    import torch
    n = S.shape[0]
    rows = M.rows
    extra_row = (2 if 2 not in rows else 0)
    if 0 in save:
        on_save(0, M.state())
    rng = np.random.default_rng(seed)
    order, pos = rng.permutation(n), 0
    for step in range(1, steps + 1):
        if pos + batch > n:
            order, pos = rng.permutation(n), 0
        idx = torch.from_numpy(order[pos:pos + batch])
        pos += batch
        z = M.logits(S[idx])                                                                     # [B, 64, 6]
        loss_rows = rows + ([extra_row] if M.arm == "sh_reach_ep_in_loss" else [])
        yb = Y[idx]
        if M.arm == "sh_reach_ep_in_loss":
            yb = torch.cat([yb, extra_label[idx].unsqueeze(-1).float()], -1)
        terms = qfocal_terms(z[..., loss_rows], yb, gamma)
        loss = terms.mean()
        M.opt.zero_grad(set_to_none=True)
        for w_, b_ in M.frozen.values():
            w_.grad, b_.grad = None, None
        loss.backward()
        rec = {"step": step, "loss": float(loss)}
        for i, r in enumerate(rows):
            if not (M.arm == "sh_live_term_dropped" and r == (3 if 3 in rows else rows[-1])):   # MM 2026-11-27 CODE AMENDMENT 1: TTC when trained (SAFE-HEAD,
                                                                                                # SH-MLP: unchanged), else the last trained row (EP-HEAD: the EP term)
                rec[TERM_OF_ROW[r]] = float(terms[i])
        rec["grad_abs"] = M.grad_abs()
        rec["frozen_grad_abs"] = M.frozen_grad_abs()
        rec["finite"] = bool(np.isfinite(rec["loss"]) and all(np.isfinite(v) for v in rec["grad_abs"].values()))
        M.opt.step()
        if metrics is not None:
            metrics.write(json.dumps(rec) + "\n")
        if not rec["finite"]:
            if metrics is not None:
                metrics.close()
            raise SystemExit(f"FAILED: non-finite loss or gradient at step {step}")
        if step in save:
            on_save(step, M.state())


def head_dict(st: dict, W0, b0, rows, mlp_hidden: int) -> dict:
    """A head in the form safehead_data.head_logits reads (the same dict load_head returns)."""
    meta = {"rows": list(rows)}
    if st["mlp"] is not None:
        meta["mlp_hidden"] = int(mlp_hidden)
    return {"W": st["W"], "b": st["b"], "W0": W0, "b0": b0, "meta": meta, "mlp": st["mlp"]}


# ============================================================================================ the run
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
    if sorted(cfg.rows) != sorted(set(cfg.rows)) or not set(cfg.rows) <= set(D.TRAINABLE_ROWS) or not cfg.rows:
        raise SystemExit(f"REFUSED: --rows {cfg.rows}: only the rows {D.TRAINABLE_ROWS} may train (the C row {4} is never a parameter)")
    if 2 in cfg.rows and cfg.select not in CV_SELECTS:
        raise SystemExit("REFUSED: --rows containing 2 (EP) needs --select cv5 or cv5m (the EP row has no held-out rule of the SAFE-HEAD card: its label is the rebuilt cache's soft EP)")
    if 2 in cfg.rows and len(cfg.rows) != 1:
        raise SystemExit("REFUSED: EP-HEAD trains row 2 ALONE (--rows 2)")
    if cfg.mlp_hidden < 0:
        raise SystemExit(f"REFUSED: --mlp-hidden {cfg.mlp_hidden}")
    if 0 not in cfg.save_steps:
        raise SystemExit("REFUSED: --save-steps must include 0 (the untrained head: CTRL-ID and the selection rule's step 0)")
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
    ho = np.where(c["split"] == "heldout")[0]
    cv = cfg.select in CV_SELECTS
    pool = np.arange(len(c["token"])) if cv else np.where(c["split"] == "train")[0]            # the tokens the head is FITTED on: all navtrain (cv5) | the 29 train logs
    if len(pool) < cfg.batch or len(ho) == 0:
        raise SystemExit(f"REFUSED: {len(pool)} train-pool / {len(ho)} held-out tokens (batch {cfg.batch})")
    rows = list(cfg.rows)
    Y = np.stack([c[D.ROW_LABEL_ALL[r]][pool] for r in rows], -1).astype(np.float32)           # [Npool, 64, R]
    n_fixed = None
    if cfg.permute_labels_seed >= 0:                                                            # CTRL-RED: every token gets another token's 64 label rows
        perm = np.random.default_rng(cfg.permute_labels_seed).permutation(len(pool))
        n_fixed = int((perm == np.arange(len(pool))).sum())
        Y = Y[perm]
    S = torch.from_numpy(np.ascontiguousarray(c["s"][pool]))
    YT = torch.from_numpy(Y)
    extra_row = 2 if 2 not in rows else 0
    EXTRA = torch.from_numpy(c[D.ROW_LABEL_ALL[extra_row]][pool].astype(np.float32))
    tok_fold = fold_of_log = None
    if cv:
        fold_of_log = D.cv_folds(c["log"][pool], N_FOLDS)
        if len(fold_of_log) < N_FOLDS:
            raise SystemExit(f"REFUSED: --select {cfg.select} needs >= {N_FOLDS} logs, the cache has {len(fold_of_log)}")
        tok_fold = (np.arange(len(pool)) % N_FOLDS) if arm == "sh_cv_token_folds" else np.array([fold_of_log[lg] for lg in c["log"][pool]])
        for k in range(N_FOLDS):
            if (tok_fold != k).sum() < cfg.batch or (tok_fold == k).sum() == 0:
                raise SystemExit(f"REFUSED: CV fold {k} has {(tok_fold == k).sum()} test / {(tok_fold != k).sum()} train tokens (batch {cfg.batch})")
    M = HeadModel(rows, cfg.mlp_hidden, W0, b0, cfg.lr, arm, cfg.seed)
    data_sha = {"init_sha256": init_sha, "index_sha256": D.sha256_file(cfg.cache_index), "heldout_logs_sha256": D.sha256_file(cfg.heldout_logs),
                "navtest_logs_sha256": D.sha256_file(cfg.navtest_logs)}
    built = {"seed": cfg.seed, "steps": cfg.steps, "lr": [g["lr"] for g in M.opt.param_groups], "batch": cfg.batch, "gamma": cfg.gamma,
             "rows": [int(g["name"][3:]) for g in M.opt.param_groups], "rows_declared": rows, "save_steps": list(cfg.save_steps),
             "permute_labels_seed": cfg.permute_labels_seed, "n_fixed_points": n_fixed, "optimizer": type(M.opt).__name__,
             "betas": [list(g["betas"]) for g in M.opt.param_groups], "eps": [g["eps"] for g in M.opt.param_groups],
             "weight_decay": [g["weight_decay"] for g in M.opt.param_groups], "loss": LOSS_NAME, "live_terms": [TERM_OF_ROW[r] for r in rows],
             "frozen_rows": sorted(M.frozen), "n_train": int(len(pool)), "n_heldout": int(len(ho)), "n_train_logs": int(len(set(c["log"][pool]))),
             "n_heldout_logs": int(len(set(c["log"][ho]))), "ctrl_feat": feat, "threads": torch.get_num_threads(), **data_sha,
             "mlp_hidden": int(M.mlp_hidden), "select": cfg.select, "n_trainable_params": M.n_params, "mlp_last_layer_abs_sum_init": M.last_layer_init_abs,
             "cv_fold_of_log": ({lg: sorted({int(f) for f in tok_fold[c["log"][pool] == lg]}) for lg in sorted(fold_of_log)} if cv else None),
             "gate_mode": "build" if cfg.build_only else ("smoke" if cfg.smoke_steps else None), "arm": arm}
    if cfg.select == "cv5m":                                                                    # only cv5m records its rule: cv5 / heldout built.json stay byte-identical
        built["cv_rule"] = CVM.rule_for_arm(arm)
    (out / "built.json").write_text(json.dumps(built, indent=1), encoding="utf-8")
    if cfg.build_only:
        print("ZZSH_BUILD_OK", flush=True)
        return 0
    n_steps = cfg.smoke_steps if cfg.smoke_steps else cfg.steps
    save = sorted({s for s in cfg.save_steps if s <= n_steps} | ({n_steps} if cfg.smoke_steps else set()))
    meta0 = {"rows": rows, "seed": cfg.seed, "init_sha256": init_sha, "index_sha256": data_sha["index_sha256"], "argv": list(argv),
             "smoke": bool(cfg.smoke_steps), "permute_labels_seed": cfg.permute_labels_seed, "loss": LOSS_NAME, "token": tok}
    if M.mlp_hidden:                                                                            # only a NON-default lever enters the head's meta (SAFE-HEAD heads stay byte-identical)
        meta0["mlp_hidden"] = int(M.mlp_hidden)
    if cv:
        meta0["select"] = cfg.select                                                            # "cv5" stays byte-identical to the parent heads
    sel_cv = cv_select(c, pool, tok_fold, S, YT, EXTRA, cfg, M, W0, b0, save, n_steps, arm) if cv else None
    if sel_cv is not None:
        (out / "cv_select.json").write_text(json.dumps(sel_cv, indent=1), encoding="utf-8")
        M = HeadModel(rows, cfg.mlp_hidden, W0, b0, cfg.lr, arm, cfg.seed)                      # the FINAL head: a fresh model, trained on ALL logs
    shas, probe = {}, {}

    def save_step(step, st):
        Wsave, bsave, msave = st["W"], st["b"], st["mlp"]
        if arm == "sh_ckpt_bias_dropped":                                                    # G-CKPT regression arm: the saved file is not the in-memory head
            if msave is None:
                bsave = b0.copy()                                                            # schema /1: the trained bias is replaced by C1's (SAFE-HEAD, unchanged)
            else:                                                                            # schema /2 (MM 2026-11-27 CODE AMENDMENT 1): W, b ARE C1's there, so dropping b
                msave = {**msave, "mlp_b2": np.zeros_like(np.asarray(msave["mlp_b2"]))}      # was a no-op; drop the trained MLP output bias instead
        shas[step] = D.save_head(out / f"head_step{step:05d}.npz", W=Wsave, b=bsave, W0=W0, b0=b0, meta={**meta0, "step": step}, mlp=msave)
        probe[step] = st
    mf = open(out / "metrics.jsonl", "w", encoding="utf-8")
    fit(M, S, YT, steps=n_steps, batch=cfg.batch, gamma=cfg.gamma, seed=cfg.seed, save=set(save), on_save=save_step, metrics=mf, extra_label=EXTRA)
    mf.close()
    if cfg.smoke_steps:                                                                          # G-EVAL / G-CKPT probes: the trainer's own in-memory head and forward
        st = probe[n_steps]
        k = min(8, len(ho))
        with torch.no_grad():
            zt = M.logits(torch.from_numpy(np.ascontiguousarray(c["s"][ho[:k]]))).numpy()
        extra = {} if st["mlp"] is None else dict(st["mlp"])
        np.savez(out / "probe.npz", W=st["W"], b=st["b"], ho_idx=ho[:k], torch_logits=zt, cached_logits=c["logits"][ho[:k]], s=c["s"][ho[:k]],
                 cached_pick=c["pick"][ho], ho_s=c["s"][ho], ho_logits=c["logits"][ho], **extra)
    if cv:
        sel = sel_cv
        line = (f"{cfg.select} chosen_step {sel['chosen_step']} oof_pdms {sel['per_step'][str(sel['chosen_step'])]['oof_pdms']:.4f} "
                f"(step 0: {sel['per_step']['0']['oof_pdms']:.4f})")
    else:
        sel = heldout_select(c, ho, out, save, W0, b0, rows)
        line = f"chosen_step {sel['chosen_step']} heldout_safety_miss_rate {sel['per_step'][str(sel['chosen_step'])]['safety_miss_rate']:.4f}"
    (out / "heads.sha256").write_text("".join(f"{shas[s]}  head_step{s:05d}.npz\n" for s in sorted(shas)), encoding="utf-8")
    (out / "run.json").write_text(json.dumps({"argv": list(argv), "built": built, "chosen_step": sel["chosen_step"], "seconds": round(time.time() - t0, 1),
                                              "token": tok, "evidence": "MEASURED (this run)"}, indent=1, default=str), encoding="utf-8")
    print(f"ZZSH_TRAIN_DONE steps {n_steps} {line} ({time.time() - t0:.1f} s)", flush=True)
    return 0


def cv_select(c, pool, tok_fold, S, YT, EXTRA, cfg, M0, W0, b0, save, n_steps, arm) -> dict:
    """PREREG_EPHEAD 4.3 (and, for --select cv5m, PREREG_DATASCALE 4.3 on top: the per-log table, the margin and the guard): 5-fold LOG-level cross-validation over ALL navtrain logs.  For each fold k a FRESH model is trained on the other 4 folds (same recipe, same seed,
    same steps), and at every saved step its head scores the fold-k tokens (the read's delta arithmetic, safehead_data.head_logits).  Per step the OUT-OF-FOLD pooled mean
    tracked PDMS of the agg32 pick is formed; the chosen step is the argmax, ties -> the EARLIEST; step 0 (C1's own head) is a column of the same table.  The cache's
    PDMS is the rebuilt metric cache (approximate, disclosed in the card): the rule only RANKS steps; every navtest number comes from the harness-exact table."""
    import torch
    rows = list(cfg.rows)
    n = len(pool)
    LG, SS, PK = c["logits"][pool], c["s"][pool], c["pick"][pool]
    oof = {s: np.zeros((n, D.N_PROP, D.N_OUT), np.float32) for s in save}
    for k in range(N_FOLDS):
        trk, tek = np.where(tok_fold != k)[0], np.where(tok_fold == k)[0]
        Mk = HeadModel(rows, cfg.mlp_hidden, W0, b0, cfg.lr, arm, cfg.seed)
        trk_t = torch.from_numpy(trk)

        def on_save(step, st, tek=tek):
            oof[step][tek] = D.head_logits(LG[tek], SS[tek], head_dict(st, W0, b0, rows, cfg.mlp_hidden))
        fit(Mk, S[trk_t], YT[trk_t], steps=n_steps, batch=cfg.batch, gamma=cfg.gamma, seed=cfg.seed, save=set(save), on_save=on_save, metrics=None, extra_label=EXTRA[trk_t])
    ar = np.arange(n)
    pdms, ep = c["tr_pdms"][pool].astype(np.float64), c["tr_ep"][pool].astype(np.float64)
    safe = (c["tr_nc"][pool] == 1) & (c["tr_dac"][pool] == 1) & (c["tr_ttc"][pool] == 1)
    logs = np.array(c["log"][pool])
    ulogs = sorted(set(logs))
    per, picks = {}, {}
    for s in save:
        k = D.select(oof[s])
        picks[s] = k
        p = pdms[ar, k]
        per[str(s)] = {"oof_pdms": float(p.mean() * 100), "per_fold_oof_pdms": [float(p[tok_fold == f].mean() * 100) for f in range(N_FOLDS)],
                       "oof_safety_miss_rate": float(((~safe[ar, k]) & safe.any(1)).mean()), "oof_ep": float(ep[ar, k].mean() * 100),
                       "oof_nc": float(c["tr_nc"][pool][ar, k].mean() * 100), "oof_dac": float(c["tr_dac"][pool][ar, k].mean() * 100),
                       "oof_ttc": float(c["tr_ttc"][pool][ar, k].mean() * 100), "oof_c": float(c["tr_c"][pool][ar, k].mean() * 100),
                       "switch_rate_vs_c1": float((k != PK).mean())}
    if 0 in picks and not np.array_equal(picks[0], PK):
        raise SystemExit(f"FAILED (CTRL-ID, out-of-fold): the untrained head does not reproduce C1's cached pick on {int((picks[0] != PK).sum())} of {n} tokens")
    p0 = pdms[ar, picks[0]]
    for s in save:
        d = pdms[ar, picks[s]] - p0
        per[str(s)]["oof_delta_vs_step0"] = float(d.mean() * 100)
        pl = np.array([d[logs == lg].mean() for lg in ulogs])
        per[str(s)]["descriptive_logs_improved_worsened_of"] = [int((pl > 0).sum()), int((pl < 0).sum()), int(len(ulogs))]
    best = max(per[str(s)]["oof_pdms"] for s in save)
    chosen = min(s for s in save if per[str(s)]["oof_pdms"] == best)
    if cfg.select == "cv5m":                                                                    # PREREG_DATASCALE 4.3: the per-log table, then the margin + guard decision FROM it
        k_of = {s: picks[s] for s in save}
        vals = {"pdms": pdms, "nc": c["tr_nc"][pool], "dac": c["tr_dac"][pool], "ttc": c["tr_ttc"][pool], "ep": ep, "c": c["tr_c"][pool]}
        tab = CVM.build_table(logs, k_of, vals)
        rule = CVM.rule_for_arm(arm)
        dec = CVM.decide(tab, save, rule)
        return {"rule": "cv5m (PREREG_DATASCALE 4.3): step s > 0 is eligible iff CI-lo(OOF gain) > 0 (paired log-cluster bootstrap, p5c_read.Boot, 10,000 draws, seed 20260927, "
                        "95 % percentile) AND no OOF sub-score delta (NC, DAC, TTC, EP, C) has CI-hi < 0; chosen = the eligible step with the largest OOF gain, ties earliest; "
                        "none eligible = step 0 (NULL-CV).  Folds = sorted log name, index mod 5, over ALL navtrain logs.  The decision is computed from `per_log`.",
                "select": "cv5m", "n_tokens": int(n), "n_logs": len(ulogs), "folds": {str(f): sorted(lg for lg in ulogs if D.cv_folds(ulogs, N_FOLDS)[lg] == f) for f in range(N_FOLDS)},
                "per_step": per, "per_log": tab, "margin_rule": rule, "margin": dec["per_step"], "eligible_steps": dec["eligible_steps"],
                "chosen_step": int(dec["chosen_step"]), "chosen_is_untrained": dec["chosen_step"] == 0, "no_margin_argmax_step": int(chosen),
                "no_margin_argmax_step_from_table": dec["no_margin_argmax_step_from_table"],
                "evidence": "MEASURED (out-of-fold navtrain, rebuilt-cache PDMS; NOT navtest)"}
    return {"rule": "argmax over --save-steps of the out-of-fold pooled mean tracked PDMS of the agg32 pick (cache tr_pdms, 5-fold log CV over ALL navtrain logs, folds = sorted "
                    "log name index mod 5); ties -> earliest; step 0 = C1 (PREREG_EPHEAD 4.3)",
            "n_tokens": int(n), "n_logs": len(ulogs), "folds": {str(f): sorted(lg for lg in ulogs if D.cv_folds(ulogs, N_FOLDS)[lg] == f) for f in range(N_FOLDS)},
            "per_step": per, "chosen_step": int(chosen), "chosen_is_untrained": chosen == 0,
            "evidence": "MEASURED (out-of-fold navtrain, rebuilt-cache PDMS; NOT navtest)"}


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
        L = D.head_logits(c["logits"][ho], c["s"][ho], h)
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
