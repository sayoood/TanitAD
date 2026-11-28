#!/usr/bin/env python3
"""v8f_mini_eval_speedavg.py -- the dev-box driver of PREREG_V8F_SPEEDAVG ("is T's speed error variance?"). Zero training, CPU.

WHY THIS FILE IS NAMED `v8f_mini_eval_*`: `box2_ops/v8f_guard.sh` stops "our" python on box 2 by matching `v8f_mini_train|v8f_mini_eval|...` on the command line. This driver is meant for the dev box
(CPU), but if it is ever run where that guard watches, the name keeps it inside the guard (the same reason `v8f_mini_eval_kindx.py` carries it). A test pins that the guard's regex matches.

WHAT IT DOES (PREREG_V8F_SPEEDAVG, verbatim): for seed s in {0, 1}, X_s's own forward, with T's FIVE speed cells (`lon_v_target_ms`, `v_a_ms`, `v_end_ms`, `v_lo_ms`, `v_hi_ms` = KIND's SPEED kind,
`v8f_mini.CODE_KINDS['speed']`) replaced by the MEAN of X_0's and X_1's T predictions of those cells on the SAME window. Every other cell, T's decisions, S and O are X_s's own. Inputs at inference
are both models' vision-only inputs (the evaluator hands both the same batch; nothing from the labels reaches either T except through the control modes below).

THE AVERAGING RULE (declared, fixed before any SPEEDAVG number): the arithmetic mean `(a + b) * 0.5` of the two models' T CODE at the five speed indices, i.e. of the STANDARDISED cells `cons_std` that O reads.
Both checkpoints must carry the same `cons_mu` / `cons_sd` (bitwise): then the mean of the standardised cells IS the standardised mean of the physical cells, and an averaged cell is on the scale O was trained on.
A pair whose standardisation differs is REFUSED (the code spaces would differ; the mean would not be one quantity).

MODES (one `v8f_mini_eval.py run` each, imported UNMODIFIED, tag `M2b-mixo_s<seed>-speedavg-<mode>`):
  own   X_s's own forward (`--tac pred`, the standard eval)                                CONTROL 3: reproduces the banked eval per window <= 1e-4 m
  self  X_s averaged WITH ITSELF                                                           CONTROL 1: reproduces `own` EXACTLY (delta 0.0 per window)
  gt    the five speed cells set to the GT (T's code elsewhere), where the window has a v9 row   CONTROL 2: reproduces KINDX's speed-only oracle per window <= 1e-4 m
  avg   X_s with the five speed cells replaced by the mean of X_0's and X_1's              THE ARM
`self`, `gt` and `avg` ride `v8f_mini_eval.py`'s own `--tac oracle --tac-groups speed` carrier path: the ONE hook is `v8f_mini.V8FMini.forward_diag` (wrapped for the duration of one eval and restored), which
recomputes X_s's own code, builds the replacement through `source_code` / `replace_cells`, and calls the original `forward_diag`. The eval's `tac` / `tac_manipulation` labels say "oracle" because that is the
carrier; this driver RELABELS them `speedavg-<mode>` in the json and adds a `speedavg` block (mode, seed, both checkpoints' sha256, the speed indices, the hook's own checks). A reader must read that block.

NON-FATAL HOOK CHECKS (recorded, never raised, so that a CONTROL is the thing that fails): `nonspeed_max_abs_dev` (every non-speed cell the eval's carrier handed to O equals X_s's own: must be exactly 0.0)
and `oracle_code_max_abs_dev` (gt mode: the code this driver built equals the eval's own oracle code for `--tac-groups speed`, cell for cell: must be exactly 0.0).

SIDE CHANNEL `speedavg_cells_<tag>.npz` (modes self / gt / avg): per window, the five speed cells of X_s (`own_std`), of the partner (`partner_std`, NaN for gt), of what O read (`used_std`), the GT in physical
units (`gt_phys`, NaN where the cell is absent) and `has` (the window has a v9 row), plus `cons_mu` / `cons_sd` of those five cells: the descriptive reads (RMSE to the GT before / after averaging, the two seeds' correlation).

`--limit N` (SMOKE ONLY): the first N windows of the eval grid, by wrapping `ctx_from_args` and `eval_grid_windows`; the json records `limit`. A reader refuses a limited run.

Evidence tier of the arm: T1-class open loop (own held outputs; no recorded future action). The `gt` mode is a T0 control, never a result.

SUBCOMMANDS
  evals   --seed S --x0 CKPT --x1 CKPT --expect-sha0 H --expect-sha1 H [--modes own,self,gt,avg] ...   run the modes for X_S (resumable: a mode whose npz, json (with the speedavg block) and cells exist is skipped);
          --dry-run prints the `v8f_mini_eval.py run` argvs and exits (no torch, no file).
  sha     print a file's sha256
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

SCHEMA = "v8f-speedavg-eval-v1"
#: the PREREG's five SPEED cells (KIND's SPEED kind), restated by NAME: `speed_idx` checks them against the model's own partition and refuses a model whose `CODE_KINDS['speed']` moved
SPEED_CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
MODES = ("own", "self", "gt", "avg")
TAGBASE = "M2b-mixo_s{seed}-speedavg"
TIER_ARM = "T1-class open loop (own held outputs; no recorded future action)"
TIER_GT = "T0 (GT speed cells): CONTROL 2 only, never a result"
TIERS = {"own": TIER_ARM, "self": TIER_ARM, "avg": TIER_ARM, "gt": TIER_GT}


def tag_of(mode: str, seed: int) -> str:
    return TAGBASE.format(seed=seed) + "-" + mode


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ the pure pieces (torch imported lazily)
def speed_idx(M) -> list:
    """The code indices of the five speed cells: from the PREREG's NAMES via the model's `CI`, checked index for index against `CODE_KINDS['speed']` and `code_group_index('speed')`. Refused if they disagree."""
    c0 = M.N_GOAL + M.N_LAT + M.N_LON
    idx = sorted(c0 + M.CI[n] for n in SPEED_CELLS)
    kinds = getattr(M, "CODE_KINDS", None)
    if not isinstance(kinds, dict) or sorted(kinds.get("speed", ())) != idx or list(M.code_group_index("speed")) != idx:
        raise SystemExit(f"REFUSED: the model's SPEED kind is {sorted((kinds or {}).get('speed', ()))}, the PREREG's five cells {SPEED_CELLS} are {idx}: a cell moved")
    if len(idx) != 5:
        raise SystemExit(f"REFUSED: the speed kind has {len(idx)} cells, the PREREG names five")
    return idx


def mean_cells(a, b):
    """The averaging rule: the arithmetic mean of two T codes, `(a + b) * 0.5`. Exact for a == b (a + a is exact in floating point, and so is * 0.5), so averaging a model with itself is the identity."""
    return (a + b) * 0.5


def replace_cells(code, src, idx):
    """`code` with the cells at `idx` taken from `src`; every other cell is `code`'s own (a copy: neither input is touched)."""
    out = code.clone()
    out[:, idx] = src[:, idx]
    return out


def source_code(mode: str, own, partner, gt, has):
    """The code the five speed cells are read from: gt -> the GT code where the window has a v9 row (T's own elsewhere: KINDX's oracle semantics), self / avg -> the mean of X_s's and the partner's code."""
    import torch
    if mode == "gt":
        return torch.where(has[:, None], gt, own)
    if mode in ("self", "avg"):
        return mean_cells(own, partner)
    raise ValueError(f"no source code for mode {mode!r}")


def partner_of(mode: str, model, other):
    """The model X_s is averaged with: ITSELF in `self` (control 1), the other seed's model in `avg`; none in `own` / `gt`."""
    if mode == "self":
        return model
    if mode == "avg":
        return other
    return None


class SpeedAvgHook:
    """Wraps `V8FMini.forward_diag` for ONE eval. Installed by `install`, undone by `restore`."""

    def __init__(self, M, mode: str, other, idx: list):
        self.M, self.mode, self.other, self.idx = M, mode, other, list(idx)
        self.ci = [i - (M.N_GOAL + M.N_LAT + M.N_LON) for i in self.idx]
        self.non = [i for i in range(M.CODE_DIM) if i not in set(self.idx)]
        self.orig = None
        self.rows = {k: [] for k in ("own_std", "partner_std", "used_std", "gt_phys", "has")}
        self.nonspeed_dev = 0.0
        self.oracle_dev = 0.0
        self.mu = self.sd = None
        self.n_calls = 0

    def __call__(self, model, batch, code_override):
        import torch
        with torch.no_grad():
            own = model.forward_infer(batch)["code"]                       # X_s's own T code, as the eval's first pass computed it
            has = batch["lab_t"]["has"].bool()
            gt = model._code_gt(batch["lab_t"]).to(own.dtype)
            part = None
            pm = partner_of(self.mode, model, self.other)
            if pm is not None:
                dev = own.device
                if next(pm.parameters()).device != dev:
                    pm.to(dev)
                part = pm.forward_infer(batch)["code"]                      # the partner's T code on the SAME batch (always recomputed, also for `self`)
            src = source_code(self.mode, own, part, gt, has)
            new = replace_cells(own, src, self.idx)
            self.nonspeed_dev = max(self.nonspeed_dev, float((code_override[:, self.non] - new[:, self.non]).abs().max()))
            self.oracle_dev = max(self.oracle_dev, float((code_override - new).abs().max()))
            self.rows["own_std"].append(own[:, self.idx].float().cpu().numpy())
            self.rows["partner_std"].append((part[:, self.idx] if part is not None else torch.full_like(own[:, self.idx], float("nan"))).float().cpu().numpy())
            self.rows["used_std"].append(new[:, self.idx].float().cpu().numpy())
            self.rows["gt_phys"].append(batch["lab_t"]["cons"][:, self.ci].float().cpu().numpy())
            self.rows["has"].append(has.cpu().numpy())
            if self.mu is None:
                self.mu, self.sd = model.cons_mu[self.ci].float().cpu().numpy(), model.cons_sd[self.ci].float().cpu().numpy()
            self.n_calls += 1
            return self.orig(model, batch, new)

    def install(self):
        self.orig = self.M.V8FMini.forward_diag
        hook = self
        self.M.V8FMini.forward_diag = lambda model, batch, code_override: hook(model, batch, code_override)
        return self

    def restore(self):
        if self.orig is not None:
            self.M.V8FMini.forward_diag = self.orig
            self.orig = None

    def cells(self) -> dict:
        import numpy as np
        out = {k: np.concatenate(v) for k, v in self.rows.items()}
        out["cons_mu"], out["cons_sd"] = self.mu, self.sd
        out["speed_idx"] = np.asarray(self.idx)
        return out

    def stats(self) -> dict:
        return {"mode": self.mode, "n_batches": self.n_calls, "nonspeed_max_abs_dev": self.nonspeed_dev,
                "oracle_code_max_abs_dev": self.oracle_dev if self.mode == "gt" else None}


# ------------------------------------------------------------------------------------------------ the pair of checkpoints
def built_cfg(cfg: dict) -> dict:
    """The config `v8f_mini_eval.py` builds from a checkpoint (`M.V8FMiniConfig(**ck["cfg"])`): the stored keys over the dataclass
    defaults. A key added after a checkpoint was written (e.g. PREREG_V8F_FLAT's `o_interface`, default "full" = every earlier arm bit
    for bit) is that default, so two runs of one model compare equal however old their checkpoints are."""
    import dataclasses
    from tanitad.models import v8f_mini as M
    out = {}
    for f in dataclasses.fields(M.V8FMiniConfig):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            out[f.name] = f.default_factory()
    out.update(cfg or {})
    return out


def pair_violations(ck0: dict, ck1: dict) -> list:
    """X_0 and X_1 must be two training runs of ONE model: same config, same grid, same state-dict layout, the SAME standardisation (cons_mu / cons_sd bitwise), different weights. [] = they are."""
    import torch
    bad = []
    b0, b1 = built_cfg(ck0.get("cfg", {})), built_cfg(ck1.get("cfg", {}))   # the configs the loader BUILDS: a key a checkpoint predates is its default
    if b0 != b1:
        diff = sorted(k for k in set(b0) | set(b1) if b0.get(k) != b1.get(k))
        bad.append(f"the two checkpoints' configs differ in {diff}")
    if ck0.get("grid") != ck1.get("grid"):
        bad.append(f"grid {ck0.get('grid')!r} != {ck1.get('grid')!r}")
    s0, s1 = ck0["model"], ck1["model"]
    if set(s0) != set(s1) or any(tuple(s0[k].shape) != tuple(s1[k].shape) for k in s0 if k in s1):
        bad.append("the two state dicts have different keys or shapes")
        return bad
    for k in ("cons_mu", "cons_sd"):
        if k not in s0 or not torch.equal(s0[k], s1[k]):
            bad.append(f"{k} differs between the two checkpoints: their code spaces differ, a mean of standardised cells would not be one quantity")
    if all(torch.equal(s0[k], s1[k]) for k in s0):
        bad.append("the two checkpoints hold identical weights: there is no second seed to average with")
    return bad


# ------------------------------------------------------------------------------------------------ argv of the eval each mode rides
def e_argv(mode: str, ckpt: str, tag: str, paths: dict, extra: list | None = None) -> list:
    """`v8f_mini_eval.py run ...`: KINDX's own / oracle(speed) argv (pinned equal by a test), without `--project --families` (the reader makes its own family blocks)."""
    common = ["run", "--ckpt", ckpt, "--tag", tag,
              "--v9-eval", paths["v9_eval"], "--lead-eval", paths["lead_eval"], "--speed-eval", paths["speed_eval"], "--kin-mode", "yaw", "--p10", paths["p10"],
              "--out-dir", paths["eval_dir"], "--eval-cache", paths["cache"], "--tracks", paths["tracks"]]
    for k, flag in (("eval_grid", "--eval-grid"), ("eval_kin", "--eval-kin"), ("eval_join", "--eval-join")):
        if paths.get(k):
            common += [flag, paths[k]]
    common += list(extra or [])
    if mode == "own":
        return common + ["--tac", "pred"]
    if mode in ("self", "gt", "avg"):
        return common + ["--tac", "oracle", "--tac-groups", "speed", "--tac-alpha", "1"]
    raise SystemExit(f"unknown mode {mode!r}")


def paths_of(a) -> dict:
    return {"v9_eval": a.v9_eval, "lead_eval": a.lead_eval, "speed_eval": a.speed_eval, "p10": a.p10, "eval_dir": a.eval_dir, "cache": a.eval_cache, "tracks": a.tracks,
            "eval_grid": a.eval_grid, "eval_kin": a.eval_kin, "eval_join": a.eval_join}


def extra_of(a) -> list:
    out = ["--n-boot", str(a.n_boot), "--batch", str(a.batch), "--lru", str(a.lru), "--device", a.device]
    return out


# ------------------------------------------------------------------------------------------------ --limit (SMOKE ONLY)
def install_limit(E, n: int):
    """The first `n` windows of the eval grid, end to end: the window list AND every per-window array of the scoring context. Returns the function that undoes it."""
    import numpy as np
    D = E.D
    orig_wins, orig_ctx = D.eval_grid_windows, E.ctx_from_args

    def wins(grid_json, J):
        return orig_wins(grid_json, J)[:n]

    def ctx(a, kin_mode=None):
        grid, J, X = orig_ctx(a, kin_mode)
        N = len(X["eid"])
        return grid, J, {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == N else v) for k, v in X.items()}

    D.eval_grid_windows, E.ctx_from_args = wins, ctx

    def undo():
        D.eval_grid_windows, E.ctx_from_args = orig_wins, orig_ctx
    return undo


# ------------------------------------------------------------------------------------------------ the evals
def _load_modules():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "..", "..", "taniteval"), os.path.join(here, "..")):
        if os.path.abspath(p) not in sys.path:
            sys.path.insert(0, os.path.abspath(p))
    from tanitad.models import v8f_mini as M          # noqa: N812
    spec = importlib.util.spec_from_file_location("v8f_mini_eval_for_speedavg", os.path.join(here, "v8f_mini_eval.py"))
    E = importlib.util.module_from_spec(spec)         # noqa: N806
    spec.loader.exec_module(E)
    return M, E


def build_model(M, ck: dict):
    """Exactly what `v8f_mini_eval.cmd_run` builds from a checkpoint."""
    import torch
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    return model.eval()


def finalize_json(js_path: str, block: dict, mode: str):
    """Relabel the eval's carrier fields and add the `speedavg` block (the eval wrote `tac: oracle` because that is the carrier path; in self / avg the cells are NOT the GT)."""
    with open(js_path, encoding="utf-8") as f:
        j = json.load(f)
    if mode != "own":
        j["tac_carrier_as_written"] = j.get("tac")
        j["tac"] = f"speedavg-{mode}"
        tm = j.get("tac_manipulation")
        if isinstance(tm, dict):
            tm["carrier_mode_as_written"] = tm.get("mode")
            tm["mode"] = f"speedavg-{mode}"
    j["speedavg"] = block
    with open(js_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(j, f, indent=1)


def eval_done(eval_dir: str, tag: str, mode: str) -> bool:
    npz, js = os.path.join(eval_dir, f"eval_{tag}.npz"), os.path.join(eval_dir, f"eval_{tag}.json")
    if not (os.path.exists(npz) and os.path.exists(js)):
        return False
    if mode != "own" and not os.path.exists(os.path.join(eval_dir, f"speedavg_cells_{tag}.npz")):
        return False
    try:
        with open(js, encoding="utf-8") as f:
            return "speedavg" in json.load(f)
    except (OSError, ValueError):
        return False


def run_evals(a, M=None, E=None, log=print):
    """The modes of X_<seed> in this process. M = the tip's `v8f_mini`, E = its `v8f_mini_eval` (injectable for the tests)."""
    import numpy as np
    import torch
    torch.set_num_threads(int(os.environ.get("SPEEDAVG_THREADS", "8")))
    ck_path = {0: a.x0, 1: a.x1}
    want = {0: a.expect_sha0, 1: a.expect_sha1}
    for s in (0, 1):
        if not os.path.exists(ck_path[s]):
            raise SystemExit(f"REFUSED: X_{s} checkpoint {ck_path[s]} does not exist")
        got = sha256_file(ck_path[s])
        if got != want[s]:
            raise SystemExit(f"REFUSED: X_{s} checkpoint sha256 {got} != the expected {want[s]}")
    if want[0] == want[1]:
        raise SystemExit("REFUSED: X_0 and X_1 are the same file: there is no second seed to average with")
    if M is None:
        M, E = _load_modules()                                           # noqa: N806
    idx = speed_idx(M)
    ck = {s: torch.load(ck_path[s], map_location="cpu", weights_only=False) for s in (0, 1)}
    bad = pair_violations(ck[0], ck[1])
    if bad:
        raise SystemExit("REFUSED: " + "; ".join(bad))
    other = build_model(M, ck[1 - a.seed]) if any(m in ("avg",) for m in a.modes) else None
    paths, extra = paths_of(a), extra_of(a)
    os.makedirs(a.eval_dir, exist_ok=True)
    n_run = 0
    for mode in a.modes:
        tag = tag_of(mode, a.seed)
        if eval_done(a.eval_dir, tag, mode):
            log(f"[skip] eval {tag}")
            continue
        argv = e_argv(mode, ck_path[a.seed], tag, paths, extra)
        log("EVAL " + " ".join(argv))
        hook = SpeedAvgHook(M, mode, other, idx).install() if mode != "own" else None
        undo = install_limit(E, a.limit) if a.limit else None
        try:
            rc = E.main(argv)
        finally:
            if hook is not None:
                hook.restore()
            if undo is not None:
                undo()
        if rc not in (0, None):
            raise SystemExit(f"FAIL: eval {tag} returned rc {rc}")
        npz, js = os.path.join(a.eval_dir, f"eval_{tag}.npz"), os.path.join(a.eval_dir, f"eval_{tag}.json")
        if not (os.path.exists(npz) and os.path.exists(js)):
            raise SystemExit(f"FAIL: eval {tag} returned 0 without writing its npz and json")
        block = {"schema": SCHEMA, "mode": mode, "seed": a.seed, "tag": tag, "tier": TIERS[mode], "limit": a.limit or None,
                 "x0_ckpt": ck_path[0], "x1_ckpt": ck_path[1], "x0_sha256": want[0], "x1_sha256": want[1], "base_ckpt": ck_path[a.seed],
                 "partner": {"own": None, "gt": None, "self": f"X_{a.seed} (itself)", "avg": f"X_{1 - a.seed}"}[mode],
                 "speed_cells": list(SPEED_CELLS), "speed_idx": list(idx), "averaging": "arithmetic mean (a + b) * 0.5 of the two T codes at the five speed indices (standardised cells; cons_mu / cons_sd bitwise equal)",
                 "carrier": None if mode == "own" else "v8f_mini_eval --tac oracle --tac-groups speed (carrier path only: the cells O read are NOT the GT unless mode == gt)"}
        if hook is not None:
            block["hook"] = hook.stats()
            np.savez_compressed(os.path.join(a.eval_dir, f"speedavg_cells_{tag}.npz"), **hook.cells())
        finalize_json(js, block, mode)
        log(f"[ok] eval {tag}")
        n_run += 1
    for s in (0, 1):
        after = sha256_file(ck_path[s])
        if after != want[s]:
            raise SystemExit(f"FAIL: the checkpoint X_{s} changed during the evals: {after} != {want[s]}")
    log(f"SPEEDAVG evals done for seed {a.seed}: {n_run} run, {len(a.modes) - n_run} skipped; both checkpoints' sha256 unchanged")
    return 0


def cmd_evals(a):
    if a.dry_run:                                                            # prints, imports no torch, touches no file
        for mode in a.modes:
            print("v8f_mini_eval.py " + " ".join(e_argv(mode, a.x0 if a.seed == 0 else a.x1, tag_of(mode, a.seed), paths_of(a), extra_of(a))))
        return 0
    if not (a.expect_sha0 and a.expect_sha1):
        raise SystemExit("REFUSED: --expect-sha0 and --expect-sha1 are required (the sha256 of both checkpoints as measured by the caller)")
    return run_evals(a)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("evals")
    p.add_argument("--seed", type=int, choices=(0, 1), required=True, help="which X_s is the base (its own forward); the other seed's model is the partner of `avg`")
    p.add_argument("--x0", required=True)
    p.add_argument("--x1", required=True)
    p.add_argument("--expect-sha0")
    p.add_argument("--expect-sha1")
    p.add_argument("--modes", type=lambda s: tuple(x for x in s.split(",") if x), default=MODES, help="comma list of own,self,gt,avg (default all four, in this order)")
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--eval-cache", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--v9-eval", required=True)
    p.add_argument("--lead-eval", required=True)
    p.add_argument("--speed-eval", required=True)
    p.add_argument("--p10", required=True)
    p.add_argument("--eval-grid", default=None)
    p.add_argument("--eval-kin", default=None)
    p.add_argument("--eval-join", default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lru", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0, help="SMOKE ONLY: the first N windows of the grid (the json records it; the reader refuses it)")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("sha")
    p.add_argument("path")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.mode == "sha":
        print(sha256_file(a.path))
        return 0
    bad = [m for m in a.modes if m not in MODES]
    if bad:
        raise SystemExit(f"unknown mode(s) {bad}: use {MODES}")
    return cmd_evals(a)


if __name__ == "__main__":
    sys.exit(main())
