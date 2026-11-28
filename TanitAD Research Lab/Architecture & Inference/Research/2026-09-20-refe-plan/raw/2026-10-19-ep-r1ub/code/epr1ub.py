"""EP-R1ub: the registered next arm of EP-R1 (MM_RULINGS_HEADROWS R7).  Two levers in ONE registration: the lr range {1e-3, 3e-3, 1e-2} (E1 -> EP-R1b) and an unsafe-proposal anchor with target 0
at weight u in {0.03, 0.1, 0.3} (D-LEAK -> EP-R1u).  Anchor fixed at 0.1.  Selection is fit-side OOF only: c* = argmax over the FEASIBLE configs of the mean-over-seeds OOF pick-level dPDMS,
feasible = the paired upper 95 % bound of the OOF unsafe-pick-rate delta vs P2 is <= 0 on BOTH seeds.

REGISTERED SOURCE: PREREG_EPR1UB.md (this package).  NEW FILE ONLY: the landed `epr1.py` (sha256 pinned below, verified before it is executed) is imported and called UNCHANGED; the pair term, the safe
anchor, the row head, the folds, the bank loader, the scoring and the bootstrap are all the landed ones.  What is new here: the u term, the 9-config grid, the all-config pick-level bootstrap, the
selection function, the identity precondition, the C2 stage for a u > 0 c*, and the derived `epr1_result.json` the frozen dev applier (`epr1_dev.py`) consumes.

STAGES
  run : (1) landed C1/C6 reachability + the UB reach proof on a real batch (aborts if RED)  (2) IDENTITY ARM: (lr 1e-3, anchor 0.1, u = 0) must reproduce the f10 fit bit for bit (aborts before the grid if not)
        (3) the 9 configs x seeds x 5 folds, each scored with the pick-level read and its paired log-cluster bootstrap  (4) selection  (5) FEASIBLE: refit c* per seed on all fit-set sets, write the rows,
        `epr1ub_result.json` and the landed-schema `epr1_result.json`; NONE-FEASIBLE: write `epr1ub_result.json` only.  Refuses to overwrite anything (c* is chosen once).
  c2  : the shuffled-label negative control at c* (EP labels shuffled among the SAFE proposals), the same fit incl. the u term.  Writes `epr1_c2_result.json` and the shuffled rows.

NEVER READS the dev-gate (P2's held-out table) or the navtest tables: every file access goes through the landed `_io_*` functions behind the landed PathGuard (a test enforces by AST that this file performs no
other file access; the one exception is the bootstrap read of the landed epr1.py inside `_load_parent`).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_EPR1_SHA256 = "a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8"      # the LANDED epr1.py (tip 1afc54f, CODE_SHA256.json)


def _load_parent():
    """The landed epr1.py, executed from bytes whose sha256 equals the pin (no check-then-use gap).  Registered as sys.modules['epr1'] so dataclasses and pickle-free code see one module."""
    p = os.path.join(HERE, "epr1.py")
    with open(p, "rb") as f:                                              # the only direct file read in this module (bootstrap of the verified parent)
        src = f.read()
    sha = hashlib.sha256(src).hexdigest()
    if sha != PARENT_EPR1_SHA256:
        raise RuntimeError(f"epr1.py in {HERE} has sha256 {sha}, the landed pin is {PARENT_EPR1_SHA256}: refusing to import unverified code")
    cur = sys.modules.get("epr1")
    if cur is not None:
        if os.path.normcase(os.path.abspath(getattr(cur, "__file__", "") or "")) == os.path.normcase(os.path.abspath(p)):
            return cur
        raise RuntimeError("a different 'epr1' module is already imported")
    mod = types.ModuleType("epr1")
    mod.__file__ = p
    sys.modules["epr1"] = mod
    exec(compile(src, p, "exec", dont_inherit=True), mod.__dict__)
    return mod


E = _load_parent()

SCHEMA = "epr1ub-v1"

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# REGISTERED constants (PREREG_EPR1UB.md; each changes the experiment if changed)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
UB_GRID_LR = (1e-3, 3e-3, 1e-2)               # s2: the lr extension (E1 -> EP-R1b)
UB_GRID_U = (0.03, 0.1, 0.3)                  # s2: the unsafe-anchor weight (D-LEAK -> EP-R1u)
UB_ANCHOR = 0.1                               # s2 / R7: the safe-anchor weight, fixed at the f10 c*'s value
UNSAFE_TARGET = 0.0                           # s3: the unsafe-anchor target is the constant 0
UNSAFE_NORM = "mean_over_unsafe_entries"      # s3: class-mean, denominator = unsafe entries of the minibatch (clamped at 1)
CRITERION_REG = "feasible_oof_dpdms"          # s5
CRITERIA = (CRITERION_REG, "pooled_oof")      # "pooled_oof" is the PARENT's criterion: selectable only so that the gate has something to REFUSE
SEL_UPPER_PCT = 97.5                          # s4/s5: the upper end of the landed paired_delta interval (percentile 2.5 / 97.5)
SEL_BOUND = 0.0                               # s5: feasible iff hi <= 0.0
SEL_SEEDS_RULE = "all"                        # s5: on BOTH seeds
SEL_OBJECTIVE = "mean_over_seeds_oof_pick_dpdms_x100"
SEL_TIE = "first_in_grid_order"
IDENT_LR, IDENT_U = 1e-3, 0.0                 # s4: the identity arm
F10_GRID_ID = "lr0.001_anchor0.1"             # the f10 grid row the identity arm reproduces
F10_RESULT_SHA256 = "eb573404bfd02bb8243cbb18cba954e2136bdcd0e3874cc21fb93e647e142463"       # raw/f10/epr1_result.sha256 at the tip
F10_ROW_SHA256 = {"0": "f775753f0817d53607b0044e9d33f6d2c5f5b01b73a8d20567d6394ca393fa95", "1": "636f6270f52ef462fb6a1a56a990aa8c683f3c30a3394091b77d10f439852fa8"}
F10_ROW_FILES = {"0": "ep_row_seed0.npy", "1": "ep_row_seed1.npy"}
LR_STEP_TOL = 0.01                            # s6: one Adam step moves the row by lr within 1 %
DEFAULT_F10_DIR = "C:/Users/Admin/epr1/run/f10"
DEFAULT_OUT_ROOT = "C:/Users/Admin/epr1ub/run"
OOF_KEYS = ("pooled", "fold_mean", "per_fold", "n_pairs", "oof_picks_changed", "n_oof_sets", "pick_level", "leak_c7")


class IdentityError(RuntimeError):
    """The u = 0 identity arm does not reproduce the f10 fit bit for bit: the instrument is wrong (outcome A)."""


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class CfgUB(E.Cfg):
    """The landed Cfg plus the unsafe-anchor weight u.  `parent_cfg()` is the landed object the frozen dev applier reconstructs from (lr, anchor, epochs, batch_sets)."""
    u: float = 0.0

    def id(self) -> str:
        return f"lr{self.lr:g}_u{self.u:g}"

    def parent_cfg(self) -> E.Cfg:
        return E.Cfg(lr=self.lr, anchor=self.anchor, epochs=self.epochs, batch_sets=self.batch_sets, k=self.k, pair_w=self.pair_w)

    def sha256(self) -> str:
        return E.sha256_bytes(E.canon_json({**dataclasses.asdict(self), "optimizer": "adam", "betas": list(E.ADAM_BETAS), "eps": E.ADAM_EPS, "weight_decay": 0.0, "row": E.EP_COL,
                                            "target": E.TARGET_ID, "safe_min": E.SAFE_MIN, "unsafe_anchor_target": UNSAFE_TARGET, "unsafe_anchor_norm": UNSAFE_NORM}))


def grid_ub(lr_grid=UB_GRID_LR, u_grid=UB_GRID_U, anchor: float = UB_ANCHOR, epochs: int = E.DEFAULT_EPOCHS, batch_sets: int = E.DEFAULT_BATCH_SETS) -> list:
    """The 9 configs in the registered order: lr ascending, then u ascending (ties go to the first)."""
    return [CfgUB(lr=lr, anchor=anchor, epochs=epochs, batch_sets=batch_sets, u=u) for lr in lr_grid for u in u_grid]


@dataclasses.dataclass
class OptionsUB(E.Options):
    criterion: str = CRITERION_REG
    lr_grid: tuple = UB_GRID_LR
    u_grid: tuple = UB_GRID_U
    anchor: float = UB_ANCHOR
    f10_dir: str = DEFAULT_F10_DIR
    ident_ref: dict | None = None          # API-only (tests): an injected identity reference; the CLI cannot reach it and the gate refuses it

    def resolved_out_dir(self) -> str:
        return self.out_dir or f"{DEFAULT_OUT_ROOT}/f{self.fraction}"

    def guard(self) -> E.PathGuard:
        files = [self.manifest, self.ranked_logs, self.p2_ckpt, os.path.abspath(E.__file__), os.path.abspath(__file__)]
        if self.s0_marker:
            files.append(self.s0_marker)
        if self.replacements_file:
            files.append(self.replacements_file)
        return E.PathGuard(read_dirs=[self.bank_dir, self.resolved_out_dir(), self.f10_dir], read_files=files, write_dirs=[self.resolved_out_dir()])

    def pins_overridden(self) -> list:
        out = super().pins_overridden()
        if tuple(self.lr_grid) != UB_GRID_LR:
            out.append("lr_grid")
        if tuple(self.u_grid) != UB_GRID_U:
            out.append("u_grid")
        if self.anchor != UB_ANCHOR:
            out.append("anchor")
        if self.criterion != CRITERION_REG:
            out.append("criterion")
        if os.path.normpath(self.f10_dir) != os.path.normpath(DEFAULT_F10_DIR):
            out.append("f10_dir")
        if self.ident_ref is not None:
            out.append("ident_ref")
        return out

    def grid(self) -> list:
        return grid_ub(self.lr_grid, self.u_grid, self.anchor, self.epochs, self.batch_sets)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the loss (the u term) and the fit path
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def unsafe_anchor_term(z6: torch.Tensor, safe: torch.Tensor) -> torch.Tensor:
    """BCE (with logits) of the EP logit against the CONSTANT 0 on the UNSAFE proposals only (all 64 positions of every set), mean over the unsafe entries.  = mean softplus(z_ep) on the unsafe set."""
    z = z6[..., E.EP_COL]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.full_like(z, UNSAFE_TARGET), reduction="none")        # MUTATION-TARGET-UB-TARGET
    m = (~safe).to(bce.dtype)                                                     # MUTATION-TARGET-UB-MASK: only unsafe proposals carry the anchor
    return (bce * m).sum() / m.sum().clamp(min=1.0)


def total_loss_ub(z6, y, safe, idx8, anchor_w: float, pair_w: float = 1.0, u: float = 0.0):
    """The landed loss plus u * (unsafe anchor).  u = 0 is the landed loss bit for bit (the term is still built)."""
    loss, pair, anchor = E.total_loss(z6, y, safe, idx8, anchor_w, pair_w)
    un = unsafe_anchor_term(z6, safe)
    loss = loss + u * un                    # MUTATION-TARGET-UB-TERM
    return loss, pair, anchor


def make_loss_fn(u: float):
    def loss_fn(z6, y, safe, idx8, anchor_w, pair_w=1.0):
        return total_loss_ub(z6, y, safe, idx8, anchor_w, pair_w, u)
    return loss_fn


def fit_ub(data, train_idx, cfg: CfgUB, seed: int, W6, b6) -> tuple:
    """The landed fit_row with the u loss routed in.  -> (row_w [256], row_b [1]) numpy."""
    return E.fit_row(data, train_idx, cfg, seed, W6, b6, loss_fn=make_loss_fn(cfg.u))       # MUTATION-TARGET-UB-REACH


def oof_run_ub(data, cfg: CfgUB, seed: int, W6, b6, train_pct: int = 100, fit_y=None) -> dict:
    """The landed `oof_run` body with `fit_ub` as the fit (so the u term reaches every fold's fit).  At u = 0 it equals the landed `oof_run` exactly (tested, and proved on the real bank by the identity arm)."""
    p2_w, p2_b = np.asarray(W6, np.float32)[E.EP_COL], np.asarray(b6, np.float32)[E.EP_COL:E.EP_COL + 1]
    train_data = data if fit_y is None else dataclasses.replace(data, y=fit_y)
    z = lambda: np.zeros(data.n_logs)                                          # noqa: E731
    num_log, den_log, n_log, pdn_log, pdp_log, un_log, up_log = z(), z(), z(), z(), z(), z(), z()
    per_fold, changed, n_oof = [], 0, 0
    pss = psn = pus = pun = 0.0
    for k in range(E.N_FOLDS):
        oof = np.nonzero(data.fold == k)[0]
        tr_logs = [i for i in np.argsort(data.log_rank, kind="stable") if data.log_fold[i] != k]
        if train_pct < 100:
            tr_logs = tr_logs[:max(1, math.ceil(train_pct / 100.0 * len(tr_logs)))]
        tr = np.nonzero(np.isin(data.log_idx, np.array(tr_logs, np.int64)))[0]
        w, b = fit_ub(train_data, tr, cfg, seed, W6, b6)
        st = E.score_sets(data, oof, w, b, p2_w, p2_b)
        num_log += E._bin(data, oof, st["num"])
        den_log += E._bin(data, oof, st["den"])
        n_log += E._bin(data, oof, np.ones(len(oof)))
        pdn_log += E._bin(data, oof, st["pd_new"])
        pdp_log += E._bin(data, oof, st["pd_p2"])
        un_log += E._bin(data, oof, st["unsafe_new"])
        up_log += E._bin(data, oof, st["unsafe_p2"])
        pss += float(st["p_safe_sum"].sum())
        psn += float(st["p_safe_n"].sum())
        pus += float(st["p_unsafe_sum"].sum())
        pun += float(st["p_unsafe_n"].sum())
        per_fold.append(float(st["num"].sum() / st["den"].sum()) if st["den"].sum() > 0 else float("nan"))
        changed += int((st["picks"] != st["pick_p2"]).sum())
        n_oof += len(oof)
    pick = E._pick_level(float(n_log.sum()), float(pdn_log.sum()), float(pdp_log.sum()), float(un_log.sum()), float(up_log.sum()))
    return {"pooled": float(num_log.sum() / den_log.sum()), "fold_mean": float(np.mean(per_fold)), "per_fold": per_fold, "num_log": num_log, "den_log": den_log,
            "n_pairs": float(den_log.sum()), "oof_picks_changed": changed, "n_oof_sets": n_oof, "n_log": n_log, "pdn_log": pdn_log, "pdp_log": pdp_log, "un_log": un_log, "up_log": up_log,
            "pick_level": pick, "leak_c7": {"mean_sigmoid_ep_on_safe": pss / max(psn, 1.0), "mean_sigmoid_ep_on_unsafe": pus / max(pun, 1.0)}}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the UB reach proof (PREREG_EPR1UB s6): non-zero where the lever acts, EXACTLY 0.0 where it must not
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _probe_ub(W6, b6, batch: dict, pair_w: float, anchor_w: float, u: float, y=None, safe=None, z6=None) -> dict:
    """One backward pass at a perturbed row with the UB loss.  -> grad report + the raw EP-row gradients and the loss value."""
    head = E.RowHead(W6, b6, probe=True, perturb=True)
    feat = batch["feat"].clone().float().requires_grad_(True)
    z6s = batch["z6"] if z6 is None else z6
    z = E.forward_z6(head, feat, z6s)
    loss, _p, _a = total_loss_ub(z, batch["y"] if y is None else y, batch["safe"] if safe is None else safe, batch["top8"], anchor_w, pair_w, u)
    if loss.requires_grad:
        loss.backward()
    rep = E._grad_report(head, feat)
    gw = head.row_w.grad if head.row_w.grad is not None else torch.zeros_like(head.row_w)
    gb = head.row_b.grad if head.row_b.grad is not None else torch.zeros_like(head.row_b)
    rep["_gw"], rep["_gb"], rep["_loss"] = gw.detach().clone(), gb.detach().clone(), float(loss.detach())
    return rep


def _same(a: dict, b: dict) -> bool:
    return bool(torch.equal(a["_gw"], b["_gw"]) and torch.equal(a["_gb"], b["_gb"]))


def reachability_check_ub(W6, b6, batch: dict, cfg: CfgUB) -> dict:
    """The u lever and the lr lever on a real batch {feat, z6, y, safe, top8}.  Raises ReachabilityError listing every failure; returns the report when GREEN."""
    fails: list = []
    s0, y0, z0 = batch["safe"], batch["y"], batch["z6"]
    n_unsafe, n_safe = int((~s0).sum()), int(s0.sum())
    if n_unsafe == 0 or n_safe == 0:
        raise E.ReachabilityError("the probe batch needs both safe and unsafe proposals to test the u term")
    u = cfg.u if cfg.u > 0.0 else 0.1
    rep: dict = {"n_unsafe_in_probe_batch": n_unsafe, "n_safe_in_probe_batch": n_safe, "u_probed": u}
    full = _probe_ub(W6, b6, batch, cfg.pair_w, cfg.anchor, u)
    uonly = _probe_ub(W6, b6, batch, 0.0, 0.0, 1.0)
    zero = _probe_ub(W6, b6, batch, 0.0, 0.0, 0.0)
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}      # noqa: E731
    rep.update({"full": strip(full), "u_only": strip(uonly), "zero_control": strip(zero)})
    for name, r in (("full", full), ("u_only", uonly)):
        if not (r["ep_row_w"] > 0.0 and math.isfinite(r["ep_row_w"]) and r["ep_row_b"] > 0.0 and math.isfinite(r["ep_row_b"])):
            fails.append(f"{name}: gradient on the EP row is ({r['ep_row_w']!r}, {r['ep_row_b']!r}); must be finite and > 0 (the u term does not reach the EP row)")
    for name, r in (("full", full), ("u_only", uonly), ("zero_control", zero)):
        if r["frozen_rows_w"] != 0.0 or r["frozen_rows_b"] != 0.0 or r["features"] != 0.0:
            fails.append(f"{name}: a frozen row or the features received a gradient (w {r['frozen_rows_w']!r}, b {r['frozen_rows_b']!r}, feat {r['features']!r}); must be EXACTLY 0.0")
    if zero["ep_row_w"] != 0.0 or zero["ep_row_b"] != 0.0:
        fails.append("zero_control: weights (0, 0, 0) still gave an EP-row gradient: the probe is not reading the loss")
    # the term IS mean softplus(z_ep) over the unsafe entries (target 0, class-mean denominator)
    zt = z0[..., E.EP_COL]
    err = float((unsafe_anchor_term(z0, s0) - torch.nn.functional.softplus(zt[~s0]).mean()).abs())
    rep["u_term_value_abs_err_vs_mean_softplus_on_unsafe"] = err
    if not err <= 1e-6:
        fails.append(f"u-term value differs from mean softplus(z_ep) over the unsafe entries by {err!r} (wrong target, mask or denominator)")
    # u = 0 is the landed loss, bit for bit (value and gradient)
    u0 = _probe_ub(W6, b6, batch, cfg.pair_w, cfg.anchor, 0.0)
    ref = E._probe_grads(W6, b6, batch, E.total_loss, cfg.pair_w, cfg.anchor)
    ident = bool(torch.equal(u0["_gw"], ref["_gw"]) and torch.equal(u0["_gb"], ref["_gb"]))
    rep["u0_gradient_bit_identical_to_landed_loss"] = ident
    if not ident:
        fails.append("u = 0: the gradient is NOT bit-identical to the landed total_loss (the u term perturbs the landed objective)")
    # u reaches the full loss
    rep["u_changes_full_gradient"] = bool(not _same(full, u0))
    if not rep["u_changes_full_gradient"]:
        fails.append("positive control: u > 0 did not change the full-loss gradient (the weight is not wired)")
    # the target is the constant 0: no EP label is read by the u term
    gen = torch.Generator().manual_seed(31)
    y_junk = torch.rand(y0.shape, generator=gen)
    rep["u_term_label_invariant_bit_identical"] = _same(_probe_ub(W6, b6, batch, 0.0, 0.0, 1.0, y=y_junk), uonly)
    if not rep["u_term_label_invariant_bit_identical"]:
        fails.append("u-term label invariance: replacing every EP label changed the u-only gradient (the target is not the constant 0)")
    # the u term acts on UNSAFE entries only
    z_junk = torch.where(s0, torch.randn(z0[..., E.EP_COL].shape, generator=gen), z0[..., E.EP_COL])
    z_safe_changed = z0.clone()
    z_safe_changed[..., E.EP_COL] = z_junk
    rep["u_term_safe_logit_invariant_bit_identical"] = _same(_probe_ub(W6, b6, batch, 0.0, 0.0, 1.0, z6=z_safe_changed), uonly)
    if not rep["u_term_safe_logit_invariant_bit_identical"]:
        fails.append("u-term mask: changing the stored EP logits of SAFE proposals changed the u-only gradient (the term acts on safe proposals)")
    z_unsafe_changed = z0.clone()
    z_unsafe_changed[..., E.EP_COL] = torch.where(~s0, torch.randn(z0[..., E.EP_COL].shape, generator=gen) * 3.0, z0[..., E.EP_COL])
    rep["u_term_unsafe_logit_changes_gradient"] = bool(not _same(_probe_ub(W6, b6, batch, 0.0, 0.0, 1.0, z6=z_unsafe_changed), uonly))
    if not rep["u_term_unsafe_logit_changes_gradient"]:
        fails.append("positive control: changing the stored EP logits of UNSAFE proposals did NOT change the u-only gradient")
    # no unsafe entry in the batch: the u term contributes EXACTLY 0.0
    nou = _probe_ub(W6, b6, batch, 0.0, 0.0, 1.0, safe=torch.ones_like(s0))
    rep["u_term_no_unsafe_entry_grad"] = {"w": nou["ep_row_w"], "b": nou["ep_row_b"]}
    if nou["ep_row_w"] != 0.0 or nou["ep_row_b"] != 0.0:
        fails.append(f"u-term with no unsafe entry: EP-row gradient ({nou['ep_row_w']!r}, {nou['ep_row_b']!r}) is not EXACTLY 0.0")
    # an all-unsafe batch: the safe terms contribute exactly 0.0, so the full-loss gradient IS the u-only gradient at the same u
    au_full = _probe_ub(W6, b6, batch, cfg.pair_w, cfg.anchor, u, safe=torch.zeros_like(s0))
    au_u = _probe_ub(W6, b6, batch, 0.0, 0.0, u, safe=torch.zeros_like(s0))
    rep["all_unsafe_full_equals_u_only_bit_identical"] = _same(au_full, au_u)
    if not rep["all_unsafe_full_equals_u_only_bit_identical"] or not au_u["ep_row_w"] > 0.0:
        fails.append("all-unsafe batch: the full-loss gradient is not bit-identical to the u-only gradient (a safe term still acts on unsafe proposals) or the u-only gradient is zero")
    # the real fit path: u reaches the optimiser, and lr is the step size
    nb = len(batch["feat"])
    mini = E.FitData(log_names=["probe"], log_rank=np.zeros(1, np.int64), log_fold=np.zeros(1, np.int64), log_idx=np.zeros(nb, np.int64), fold=np.zeros(nb, np.int64), feat=batch["feat"],
                     z6=z0, y=y0, safe=s0, pdms=torch.zeros_like(y0), top8=batch["top8"], provenance={})
    p2w = np.asarray(W6)[E.EP_COL]
    one = lambda c: fit_ub(mini, np.arange(nb), dataclasses.replace(c, epochs=1, batch_sets=nb), 0, W6, b6)      # noqa: E731
    w_u, _ = one(dataclasses.replace(cfg, u=u, lr=1e-3))
    w_0, _ = one(dataclasses.replace(cfg, u=0.0, lr=1e-3))
    rep["u_reaches_the_optimiser_rows_differ"] = bool(not np.array_equal(w_u, w_0))
    if not rep["u_reaches_the_optimiser_rows_differ"]:
        fails.append("the fitted row at u > 0 equals the row at u = 0 through fit_ub: the u term does not reach the optimiser")
    steps = {}
    for lr in tuple(UB_GRID_LR) + (0.0,):
        w, b = one(dataclasses.replace(cfg, u=u, lr=lr))
        moved = float(max(np.abs(w - p2w).max(), np.abs(b - np.asarray(b6)[E.EP_COL]).max()))
        steps[f"{lr:g}"] = moved
        if lr == 0.0:
            if moved != 0.0:
                fails.append(f"lr = 0 moved the row by {moved!r}: must be EXACTLY 0.0")
        elif not (lr * (1.0 - LR_STEP_TOL) <= moved <= lr * (1.0 + LR_STEP_TOL)):
            fails.append(f"lr {lr:g}: one Adam step moved the row by {moved:.6g}, expected {lr:g} +- {100 * LR_STEP_TOL:g} % (the fit path does not use this lr)")
    rep["one_step_max_abs_move_by_lr"] = steps
    if fails:
        raise E.ReachabilityError("UB reach RED: " + " | ".join(fails))
    return rep


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# selection (PREREG_EPR1UB s5)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def is_feasible(row: dict, seeds) -> bool:
    """Feasible iff the paired upper bound of the OOF unsafe-pick-rate delta vs P2 is <= SEL_BOUND on EVERY seed (a NaN bound is infeasible)."""
    his = [row["per_seed"][str(sd)]["boot"]["delta_unsafe_rate"]["hi"] for sd in seeds]
    return all(h <= SEL_BOUND for h in his)                                  # MUTATION-TARGET-SEL-FEAS


def dpdms_mean(row: dict, seeds) -> float:
    return float(np.mean([row["per_seed"][str(sd)]["boot"]["delta_pdms_x100"]["point"] for sd in seeds]))


def select_cstar(rows: list, seeds, criterion: str = CRITERION_REG) -> dict:
    """rows in GRID ORDER.  CRITERION_REG: argmax over the feasible configs of the mean-over-seeds OOF dPDMS (first in grid order on ties).  'pooled_oof' (the parent's criterion, unregistered here) is
    kept so the gate can refuse it.  -> {status, index, id, score, feasible_ids, tied_at_max}."""
    if criterion == "pooled_oof":
        sc = [float(np.mean([r["per_seed"][str(sd)]["pooled"] for sd in seeds])) for r in rows]
        i = int(np.argmax(sc))
        return {"status": "FEASIBLE", "index": i, "id": rows[i]["id"], "score": sc[i], "feasible_ids": [r["id"] for r in rows], "tied_at_max": [rows[k]["id"] for k in range(len(rows)) if sc[k] == sc[i]],
                "criterion": criterion}
    feas = [i for i, r in enumerate(rows) if is_feasible(r, seeds)]
    if not feas:
        return {"status": "NONE-FEASIBLE", "index": None, "id": None, "score": None, "feasible_ids": [], "tied_at_max": [], "criterion": criterion}
    sc = [dpdms_mean(rows[i], seeds) for i in feas]
    j = int(np.argmax(sc))                                                    # MUTATION-TARGET-SEL-ARGMAX: first maximum = grid order
    return {"status": "FEASIBLE", "index": feas[j], "id": rows[feas[j]]["id"], "score": sc[j], "feasible_ids": [rows[i]["id"] for i in feas],
            "tied_at_max": [rows[feas[k]]["id"] for k in range(len(feas)) if sc[k] == sc[j]], "criterion": criterion}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the identity arm (PREREG_EPR1UB s4)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _deep_diff(got, want, path: str, out: list) -> None:
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            out.append((path, "keys differ", sorted(got) if isinstance(got, dict) else repr(got), sorted(want)))
            return
        for k in sorted(want):
            _deep_diff(got[k], want[k], f"{path}/{k}", out)
    elif isinstance(want, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(got) != len(want):
            out.append((path, "length differs", repr(got)[:80], repr(want)[:80]))
            return
        for i, (g, w) in enumerate(zip(got, want)):
            _deep_diff(g, w, f"{path}[{i}]", out)
    else:
        if not (got == want):
            out.append((path, "value differs", repr(got), repr(want)))


def load_identity_ref(guard, opts: OptionsUB) -> dict:
    """The f10 fit as the identity reference: result sha256 and both row files' sha256 are checked against the registered pins BEFORE their content is used."""
    res_path = os.path.join(opts.f10_dir, "epr1_result.json")
    raw = E._io_read_bytes(guard, res_path)
    sha = E.sha256_bytes(raw)
    if sha != F10_RESULT_SHA256:
        raise IdentityError(f"{res_path} sha256 {sha} != the registered f10 result {F10_RESULT_SHA256}")
    doc = json.loads(raw.decode("utf-8"))
    if doc["c_star"]["id"] != F10_GRID_ID:
        raise IdentityError(f"the f10 c* is {doc['c_star']['id']!r}, expected {F10_GRID_ID!r}")
    grid_row = [g for g in doc["grid"] if g["id"] == F10_GRID_ID]
    if len(grid_row) != 1:
        raise IdentityError(f"the f10 grid has {len(grid_row)} rows named {F10_GRID_ID}")
    rows = {}
    for sd, name in F10_ROW_FILES.items():
        rsha = E.sha256_bytes(E._io_read_bytes(guard, os.path.join(opts.f10_dir, name)))
        if rsha != F10_ROW_SHA256[sd] or doc["final_rows"][sd]["sha256"] != rsha:
            raise IdentityError(f"{name}: sha256 {rsha} != the registered {F10_ROW_SHA256[sd]}")
        rows[sd] = rsha
    return {"source": "f10", "result_sha256": sha, "per_seed": grid_row[0]["per_seed"], "boot": doc["pick_level_at_c_star"], "row_sha256": rows}


def identity_arm(data, opts: OptionsUB, W6, b6, ref: dict, draws: np.ndarray) -> dict:
    """(lr 1e-3, anchor 0.1, u = 0) through the UB path vs the reference: OOF read, bootstrap and final-row sha256, per seed, EXACT.  -> report; raises IdentityError on ANY difference."""
    cfg = CfgUB(lr=IDENT_LR, anchor=opts.anchor, epochs=opts.epochs, batch_sets=opts.batch_sets, u=IDENT_U)
    diffs: list = []
    per_seed = {}
    for sd in opts.seeds:
        r = oof_run_ub(data, cfg, sd, W6, b6)
        got = {k: r[k] for k in OOF_KEYS}
        boot = E.boot_pick_level(r, draws)
        w, b = fit_ub(data, np.arange(data.n_sets), cfg, sd, W6, b6)
        rsha = E.sha256_bytes(E.row_bytes(w, b))
        _deep_diff(got, ref["per_seed"][str(sd)], f"seed{sd}/oof", diffs)
        _deep_diff(boot, ref["boot"][str(sd)], f"seed{sd}/boot", diffs)
        _deep_diff(rsha, ref["row_sha256"][str(sd)], f"seed{sd}/row_sha256", diffs)
        per_seed[str(sd)] = {"pooled": r["pooled"], "row_sha256": rsha, "boot": boot}
        E._log(f"identity seed {sd}: pooled {r['pooled']:.17g} row sha256 {rsha[:16]}")
    rep = {"config": cfg.id(), "config_u0_id": "lr0.001_anchor0.1", "reference": ref.get("source"), "reference_result_sha256": ref.get("result_sha256"), "n_differences": len(diffs),
           "bit_identical": not diffs, "per_seed": per_seed, "first_differences": [list(d) for d in diffs[:10]]}
    if diffs:
        raise IdentityError(f"the u = 0 arm does not reproduce the reference ({len(diffs)} differences), first: {diffs[:3]}")
    return rep


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _row_meta(w, b, name: str, sha: str, blob: bytes, W6, b6) -> dict:
    return {"file": name, "sha256": sha, "bytes": len(blob), "layout": "float32[257] = weight[2,:256] then bias[2], .npy",
            "max_abs_change_vs_p2": float(max(np.abs(w - W6[E.EP_COL]).max(), np.abs(b - b6[E.EP_COL]).max())), "fitted_on": "ALL fit-set sets of S_f"}


def _py(o):
    """numpy scalars -> Python (canon_json cannot serialise numpy)."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def build_compat_doc(res: dict, ub_sha: str) -> dict:
    """The landed-schema `epr1_result.json` the frozen `epr1_dev.py` consumes (schema epr1-v1, stage R1).  c*'s `config_sha256` is the LANDED Cfg's (what the applier recomputes from lr, anchor,
    epochs, batch_sets); u and the UB sha256 are extra keys; `code_sha256` is the landed epr1.py's (the applier pins it)."""
    cs = res["selection"]["c_star"]
    return _py({"schema": E.SCHEMA, "stage": "R1", "fraction_pct": res["fraction_pct"], "inputs": res["inputs"], "params": res["params"], "c1_c6_reachability": res["c1_c6_reachability"],
                "p2_row_reference_oof": res["p2_row_reference_oof"], "grid": res["grid"],
                "c_star": {"id": cs["id"], "lr": cs["lr"], "anchor": cs["anchor"], "u": cs["u"], "config_sha256": cs["config_sha256"], "ub_config_sha256": cs["ub_config_sha256"], "score": cs["score"],
                           "criterion": cs["criterion"], "tied_at_max": cs["tied_at_max"], "tie_break": SEL_TIE, "c_star_at_grid_edge": cs["c_star_at_grid_edge"]},
                "pick_level_at_c_star": res["selection"]["pick_level_at_c_star"], "final_rows": res["final_rows"], "code_sha256": res["code_sha256"],
                "epr1ub": {"derived_from": "epr1ub_result.json", "derived_from_sha256": ub_sha, "epr1ub_sha256": res["epr1ub_sha256"], "note": "derived view of the EP-R1ub result in the landed epr1-v1 schema so the "
                           "frozen epr1_dev.py accepts it; c*.config_sha256 is the landed Cfg's, the UB sha256 (with u) is c*.ub_config_sha256"},
                "run_meta": res["run_meta"]})


def _sidecar(guard, out: str, name: str, sha: str, side: str) -> None:
    E._io_write_bytes(guard, os.path.join(out, side), (sha + "  " + name + "\n").encode("utf-8"), exclusive=True)


def run_ub(opts: OptionsUB) -> dict:
    t0 = time.time()
    guard, out, manifest, man_sha, ranked_sha, selected, repl_rep, per_fold_logs, W6, b6, data = E._setup(opts)
    E._io_makedirs(guard, out)
    result_path = os.path.join(out, "epr1ub_result.json")
    compat_path = os.path.join(out, "epr1_result.json")
    stale = [result_path, compat_path] + [os.path.join(out, f"ep_row_seed{sd}.npy") for sd in opts.seeds]
    for p in stale:
        if E._io_exists(guard, p):
            raise E.BankError(f"{p} exists: c* is chosen ONCE and never re-drawn (move the directory away to rerun)")
    # --- 1. reach: the landed C1/C6 and the UB proof, on a real batch
    batch = E._probe_batch(data, opts.reach_n_sets)
    base = E.Cfg(lr=E.GRID_LR[-1], anchor=opts.anchor, epochs=opts.epochs, batch_sets=opts.batch_sets)
    reach = E.reachability_check(W6, b6, batch, base)
    reach_ub = reachability_check_ub(W6, b6, batch, CfgUB(lr=UB_GRID_LR[0], anchor=opts.anchor, epochs=opts.epochs, batch_sets=opts.batch_sets, u=UB_GRID_U[1]))
    E._log("C1/C6 + UB reach GREEN")
    draws = E.boot_draws(data.n_logs)
    # --- 2. the identity arm: a precondition of the grid
    ref = opts.ident_ref if opts.ident_ref is not None else load_identity_ref(guard, opts)
    try:
        ident = identity_arm(data, opts, W6, b6, ref, draws)                  # MUTATION-TARGET-IDENT
    except IdentityError as ex:
        E._io_write_bytes(guard, os.path.join(out, "epr1ub_ident_FAIL.json"), E.canon_json({"schema": SCHEMA, "stage": "IDENT-FAIL", "error": str(ex)[:2000]}), exclusive=False)
        raise
    E._log("IDENTITY ARM bit-identical to the reference")
    # --- 3. the grid
    y8, s8 = np.take_along_axis(data.y.numpy(), data.top8.numpy(), axis=1), np.take_along_axis(data.safe.numpy(), data.top8.numpy(), axis=1)
    pc = E.pair_counts_np(y8, s8)
    fold_table = [{"fold": k, "logs": per_fold_logs[k], "sets": int((data.fold == k).sum()), "contrast_sets": int(((data.fold == k) & (pc > 0)).sum()), "pairs": int(pc[data.fold == k].sum())}
                  for k in range(E.N_FOLDS)]
    ref_p2 = E.p2_reference(data, W6, b6)
    cfgs = opts.grid()
    rows = []
    for ci, cfg in enumerate(cfgs):
        per_seed = {}
        for sd in opts.seeds:
            r = oof_run_ub(data, cfg, sd, W6, b6)
            per_seed[str(sd)] = {**{k: r[k] for k in OOF_KEYS}, "boot": E.boot_pick_level(r, draws)}
        mean = {"pooled": float(np.mean([per_seed[str(s)]["pooled"] for s in opts.seeds])), "fold_mean": float(np.mean([per_seed[str(s)]["fold_mean"] for s in opts.seeds])),
                "dpdms_x100": dpdms_mean({"per_seed": per_seed}, opts.seeds),
                "d_unsafe_rate": float(np.mean([per_seed[str(s)]["boot"]["delta_unsafe_rate"]["point"] for s in opts.seeds]))}
        row = {"id": cfg.id(), "lr": cfg.lr, "u": cfg.u, "anchor": cfg.anchor, "config_sha256": cfg.sha256(), "per_seed": per_seed, "mean_over_seeds": mean}
        row["feasible_per_seed"] = {str(s): bool(per_seed[str(s)]["boot"]["delta_unsafe_rate"]["hi"] <= SEL_BOUND) for s in opts.seeds}
        row["feasible"] = bool(is_feasible(row, opts.seeds))
        rows.append(row)
        E._log(f"config {ci + 1}/{len(cfgs)} {cfg.id()} pooled {mean['pooled']:.5f} dPDMS {mean['dpdms_x100']:+.4f} unsafe hi "
               f"{[round(per_seed[str(s)]['boot']['delta_unsafe_rate']['hi'], 5) for s in opts.seeds]} feasible {row['feasible']}")
    # --- 4. selection
    sel = select_cstar(rows, opts.seeds, opts.criterion)                      # MUTATION-TARGET-SEL-CALL
    star = cfgs[sel["index"]] if sel["index"] is not None else None
    code_sha = E.sha256_bytes(E._io_read_bytes(guard, os.path.abspath(E.__file__)))
    ub_sha = E.sha256_bytes(E._io_read_bytes(guard, os.path.abspath(__file__)))
    selection = {"criterion": opts.criterion, "rule": "c* = argmax over FEASIBLE configs of the mean-over-seeds OOF pick-level dPDMS x100; feasible iff the paired upper 97.5th percentile of the OOF "
                 "unsafe-pick-rate delta vs P2 is <= 0.0 on BOTH seeds; ties go to the first in grid order", "status": sel["status"], "feasible_ids": sel["feasible_ids"],
                 "tied_at_max": sel["tied_at_max"], "n_feasible": len(sel["feasible_ids"]), "c_star": None, "f_weak": None, "pick_level_at_c_star": None,
                 "comparator_note": "P2 is IN-SAMPLE on these sets (it trained on them): the unsafe-rate and dPDMS deltas are biased in P2's favour"}
    final = {}
    if star is not None:
        i = sel["index"]
        pls = {str(sd): rows[i]["per_seed"][str(sd)]["boot"] for sd in opts.seeds}
        selection["pick_level_at_c_star"] = pls
        selection["f_weak"] = bool(any(pls[str(sd)]["delta_pdms_x100"]["point"] <= 0.0 for sd in opts.seeds) or rows[i]["mean_over_seeds"]["dpdms_x100"] <= 0.0)
        selection["c_star"] = {"id": star.id(), "lr": star.lr, "anchor": star.anchor, "u": star.u, "config_sha256": star.parent_cfg().sha256(), "ub_config_sha256": star.sha256(), "score": sel["score"],
                               "criterion": opts.criterion, "tied_at_max": sel["tied_at_max"], "c_star_at_grid_edge": bool(star.lr == max(opts.lr_grid)), "c_star_at_lower_lr_edge": bool(star.lr == min(opts.lr_grid))}
        for sd in opts.seeds:
            w, b = fit_ub(data, np.arange(data.n_sets), star, sd, W6, b6)
            blob = E.row_bytes(w, b)
            name = f"ep_row_seed{sd}.npy"
            sha = E._io_write_bytes(guard, os.path.join(out, name), blob, exclusive=True)
            final[str(sd)] = _row_meta(w, b, name, sha, blob, W6, b6)
    p = {"grid_lr": list(opts.lr_grid), "grid_u": list(opts.u_grid), "anchor": opts.anchor, "k": E.K, "seeds": list(opts.seeds), "epochs": opts.epochs, "batch_sets": opts.batch_sets,
         "optimizer": "adam", "betas": list(E.ADAM_BETAS), "eps": E.ADAM_EPS, "criterion": opts.criterion, "folds": E.N_FOLDS, "fold_rule": "rank_in_key_order mod 5", "target": E.TARGET_ID,
         "safe_min": E.SAFE_MIN, "unsafe_anchor_target": UNSAFE_TARGET, "unsafe_anchor_norm": UNSAFE_NORM, "threads": opts.threads, "torch": torch.__version__, "numpy": np.__version__}
    result = {"schema": SCHEMA, "stage": "UB", "fraction_pct": opts.fraction,
              "inputs": {"bank_dir": opts.bank_dir, "bank_manifest": opts.manifest, "bank_manifest_sha256": man_sha, "ranked_fit_logs_sha256": ranked_sha, "p2_ckpt": opts.p2_ckpt,
                         "p2_row_sha256": E.sha256_bytes(E.row_bytes(W6[E.EP_COL], b6[E.EP_COL:E.EP_COL + 1])), "bank_provenance": data.provenance, "replacements": repl_rep,
                         "n_logs": data.n_logs, "n_sets": data.n_sets, "fold_table": fold_table, "pins_overridden": opts.pins_overridden(), "f10_dir": opts.f10_dir},
              "params": p, "c1_c6_reachability": reach, "ub_reachability": reach_ub, "identity_arm": ident, "p2_row_reference_oof": ref_p2, "grid": rows, "selection": selection,
              "final_rows": final if final else None, "code_sha256": code_sha, "epr1ub_sha256": ub_sha,
              "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1)}}
    result = _py(result)
    blob = E.canon_json(result)
    sha = E._io_write_bytes(guard, result_path, blob, exclusive=True)
    _sidecar(guard, out, "epr1ub_result.json", sha, "epr1ub_result.sha256")
    if star is not None:
        cblob = E.canon_json(build_compat_doc(result, sha))
        csha = E._io_write_bytes(guard, compat_path, cblob, exclusive=True)
        _sidecar(guard, out, "epr1_result.json", csha, "epr1_result.sha256")
        E._log(f"c* = {star.id()} dPDMS {sel['score']:+.4f}; wrote {compat_path} sha256 {csha}")
    else:
        E._log("NONE-FEASIBLE: no config satisfies the unsafe-rate constraint on both seeds; no c*, no rows")
    E._log(f"wrote {result_path} sha256 {sha}")
    return result


def run_c2_ub(opts: OptionsUB) -> dict:
    """C2 at c*: the same fit (incl. the u term) with the EP labels shuffled among the SAFE proposals of each set; trained on shuffled, scored on real."""
    t0 = time.time()
    guard, out, manifest, man_sha, ranked_sha, selected, repl_rep, per_fold_logs, W6, b6, data = E._setup(opts)
    for stale in [os.path.join(out, "epr1_c2_result.json")] + [os.path.join(out, f"ep_row_c2shuffle_seed{sd}.npy") for sd in opts.seeds]:
        if E._io_exists(guard, stale):
            raise E.BankError(f"{stale} exists: the C2 read is written once (move the directory away to rerun)")
    doc = json.loads(E._io_read_bytes(guard, os.path.join(out, "epr1_result.json")).decode("utf-8"))
    ub_raw = E._io_read_bytes(guard, os.path.join(out, "epr1ub_result.json"))
    if doc["epr1ub"]["derived_from_sha256"] != E.sha256_bytes(ub_raw):
        raise E.BankError("epr1_result.json was not derived from the epr1ub_result.json on disk")
    if doc["inputs"]["bank_manifest_sha256"] != man_sha:
        raise E.BankError("the bank manifest changed since epr1_result.json was written")
    if doc["fraction_pct"] != opts.fraction:
        raise E.BankError(f"epr1_result.json is for f = {doc['fraction_pct']} %, not {opts.fraction} %")
    cs = doc["c_star"]
    star = CfgUB(lr=cs["lr"], anchor=cs["anchor"], u=cs["u"], epochs=doc["params"]["epochs"], batch_sets=doc["params"]["batch_sets"])
    if star.sha256() != cs["ub_config_sha256"] or star.parent_cfg().sha256() != cs["config_sha256"]:
        raise E.BankError("c* from epr1_result.json does not reproduce its recorded config sha256s")
    y_shuf = E.shuffle_safe_labels(data.y, data.safe, E.C2_SHUFFLE_SEED)
    per_seed, final = {}, {}
    for sd in opts.seeds:
        r = oof_run_ub(data, star, sd, W6, b6, fit_y=y_shuf)
        per_seed[str(sd)] = {k: r[k] for k in OOF_KEYS}
        E._log(f"C2 seed {sd} pooled OOF concordance (real labels) {r['pooled']:.5f}")
        w, b = fit_ub(dataclasses.replace(data, y=y_shuf), np.arange(data.n_sets), star, sd, W6, b6)
        blob = E.row_bytes(w, b)
        name = f"ep_row_c2shuffle_seed{sd}.npy"
        final[str(sd)] = {"file": name, "sha256": E._io_write_bytes(guard, os.path.join(out, name), blob, exclusive=True), "bytes": len(blob)}
    result = {"schema": E.SCHEMA, "stage": "C2", "fraction_pct": opts.fraction, "c_star": cs, "bank_manifest_sha256": man_sha, "shuffle_seed": E.C2_SHUFFLE_SEED,
              "shuffle": "EP labels permuted among the SAFE proposals of each set; trained on shuffled (with the u term), scored on real", "per_seed": _py(per_seed), "final_rows": final,
              "p2_row_reference_oof": doc["p2_row_reference_oof"], "epr1ub_sha256": E.sha256_bytes(E._io_read_bytes(guard, os.path.abspath(__file__))),
              "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1)}}
    E._io_write_bytes(guard, os.path.join(out, "epr1_c2_result.json"), E.canon_json(_py(result)), exclusive=True)
    return result


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _grid_str(v) -> str:
    return ",".join(f"{x:g}" for x in v)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EP-R1ub: lr extension + unsafe-proposal anchor, fit-side selection under a safety constraint. Never reads the dev-gate or navtest tables.")
    ap.add_argument("cmd", choices=("run", "c2"))
    ap.add_argument("--fraction", type=int, required=True, choices=E.FRACTIONS)
    ap.add_argument("--bank-dir", default=E.DEFAULT_BANK_DIR)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--ranked-logs", default=E.DEFAULT_RANKED)
    ap.add_argument("--p2-ckpt", default=E.DEFAULT_P2_CKPT)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--s0-marker", default=E.DEFAULT_S0_MARKER)
    ap.add_argument("--f10-dir", default=DEFAULT_F10_DIR)
    ap.add_argument("--seeds", default=",".join(str(s) for s in E.SEEDS))
    ap.add_argument("--epochs", type=int, default=E.DEFAULT_EPOCHS)
    ap.add_argument("--batch-sets", type=int, default=E.DEFAULT_BATCH_SETS)
    ap.add_argument("--criterion", choices=CRITERIA, default=CRITERION_REG)
    ap.add_argument("--lr-grid", default=_grid_str(UB_GRID_LR))
    ap.add_argument("--u-grid", default=_grid_str(UB_GRID_U))
    ap.add_argument("--anchor", type=float, default=UB_ANCHOR)
    ap.add_argument("--replacements-file", default=None)
    ap.add_argument("--threads", type=int, default=8)
    return ap


def options_from_args(a: argparse.Namespace) -> OptionsUB:
    return OptionsUB(fraction=a.fraction, bank_dir=a.bank_dir, manifest=a.manifest or (a.bank_dir.rstrip("/\\") + "/MANIFEST.sha256.json"), ranked_logs=a.ranked_logs, p2_ckpt=a.p2_ckpt,
                     out_dir=a.out_dir, s0_marker=a.s0_marker, seeds=tuple(int(s) for s in a.seeds.split(",")), epochs=a.epochs, batch_sets=a.batch_sets, criterion=a.criterion,
                     replacements_file=a.replacements_file, threads=a.threads, lr_grid=tuple(float(x) for x in a.lr_grid.split(",")), u_grid=tuple(float(x) for x in a.u_grid.split(",")),
                     anchor=a.anchor, f10_dir=a.f10_dir)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    opts = options_from_args(a)
    {"run": run_ub, "c2": run_c2_ub}[a.cmd](opts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
