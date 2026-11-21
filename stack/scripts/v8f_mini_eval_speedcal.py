#!/usr/bin/env python3
"""v8f_mini_eval_speedcal.py -- the dev-box driver of PREREG_V8F_SPEEDCAL ("is T's speed error a head bias?"). Zero training, CPU.

WHY THIS FILE IS NAMED `v8f_mini_eval_*`: `box2_ops/v8f_guard.sh` stops "our" python on box 2 by matching `v8f_mini_train|v8f_mini_eval|...` on the command line. This driver is meant for the dev box
(CPU); the name keeps it inside the guard if it is ever run where the guard watches (the same reason `v8f_mini_eval_speedavg.py` carries it). A test pins that the guard's regex matches.

WHAT IT DOES (PREREG_V8F_SPEEDCAL, verbatim): for seed s in {0, 1}, X_s = `M2b-mixo_s<s>`; for each of the FIVE SPEED cells c (`lon_v_target_ms`, `v_a_ms`, `v_end_ms`, `v_lo_ms`, `v_hi_ms` = KIND's SPEED
kind, the SAME cells as SPEEDAVG) an affine map  c' = a_{s,c} * c + b_{s,c}  in m/s, fitted by closed-form MASKED least squares on X_s's own T prediction of the cell on the TRAINING windows of HBLEND's fit set
(141 train clips) against the cell's v9 TRAIN target (the one T is trained against). The mask m_{w,c} drops the windows without a finite GT for that cell (and without a v9 row), exactly the trainer's own
mask (`cons_absent=mask`). NO EVAL WINDOW ENTERS THE FIT. At inference on eval-141, T's five speed cells are replaced by c'; every other cell, T's decisions, S and O are X_s's own, through the SAME hook
as SPEEDAVG (`v8f_mini_eval_speedavg.SpeedAvgHook`, subclassed: only `__call__` differs, so install / restore / the cells side channel are SPEEDAVG's).

THE MAP IS APPLIED IN PHYSICAL UNITS and written back into the STANDARDISED cell O reads:  std' = a * std + ((a - 1) * mu + b) / sd   (algebraically (a*(std*sd + mu) + b - mu) / sd), computed in float64
and cast back. With a = 1 and b = 0 this is `1.0 * std + 0.0` = std BIT FOR BIT, so CONTROL 1 (the identity map reproduces X_s's own eval, delta 0.0) needs no special case.

SUBCOMMANDS
  fit     --seed S --x CKPT --expect-sha H ...   X_s's T forward (the full `forward_infer`, no label read) on HBLEND's fit windows -> `fit_cells_<tag>.npz` (the arrays the controls re-fit) and `fit_<tag>.json`
          (a, b, n per cell, train RMSE before / after, the PERMUTED-target map of control 4). The window list is the trainer's own builder (`train_windows`, hist, window_start) at the eval's own
          deterministic age schedule (`--ages cycle`: a_t = i % 3, a_s = a_t + (i // 3) % 5), and is CHECKED equal to HBLEND's `fit_windows_*.npz` (ci, t, a_t, a_s) when `--hblend-windows` is given.
  evals   --seed S --x CKPT --expect-sha H --fit-json F ...   one `v8f_mini_eval.py run` per mode (imported UNMODIFIED), resumable (a finished mode is skipped):
            own       X_s's own forward (`--tac pred`, the standard eval)
            identity  the map a = 1, b = 0                                       CONTROL 1: reproduces `own` per window EXACTLY (delta 0.0)
            gt        the five speed cells set to the GT (T's code elsewhere)    CONTROL 3: reproduces KINDX's speed-only oracle per window <= 1e-4 m
            cal       the fitted map                                             THE ARM
            perm      the map fitted on TRAINING targets PERMUTED across windows CONTROL 4: must NOT be separated-better than X_s
            ceiling   the same affine form fitted on the EVAL GT itself          T0 DIAGNOSTIC, never a result: the best ANY affine map can do
          `--target b` runs the BASELINE B (`M2b_s<s>`, non-MIXO) with modes own / cal only: X_s's fitted map applied to B's cells (beside, descriptive).
  sha     print a file's sha256.

Evidence tier of the arm: T1-class open loop (own held outputs; no recorded future action). `gt` and `ceiling` are T0 diagnostics (they read the eval GT), never results. The map is a fixed function of T's own
output, so inference stays vision-only: no ego channel and no situation-classifier output in the goal path.

NON-FATAL HOOK CHECKS (recorded, never raised, so that a CONTROL is what fails): `nonspeed_max_abs_dev` (every non-speed cell the eval's carrier handed to O equals X_s's own: exactly 0.0) and
`oracle_code_max_abs_dev` (gt mode: the code built here equals the eval's own oracle code for `--tac-groups speed`: exactly 0.0).

`--limit N` (SMOKE ONLY): the first N windows of the eval grid (SPEEDAVG's `install_limit`); the json records it and the reader refuses a limited run. `--dry-run` prints the argvs and the plan, imports no torch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import v8f_mini_eval_speedavg as SA  # noqa: E402  (SPEEDAVG's driver: the hook, the pair rules, the argv, the limit)

SCHEMA = "v8f-speedcal-eval-v1"
FIT_SCHEMA = "v8f-speedcal-fit-v1"
#: the PREREG's five SPEED cells: SPEEDAVG's, restated by NAME and checked against the model's own partition by `SA.speed_idx`
SPEED_CELLS = SA.SPEED_CELLS
MODES = ("own", "identity", "gt", "cal", "perm", "ceiling")          # the order matters: `ceiling` is fitted on `identity`'s dump of the eval cells
B_MODES = ("own", "cal")
TAGBASE = {"x": "M2b-mixo_s{seed}-speedcal", "b": "M2b_s{seed}-speedcal"}
TIER_ARM = SA.TIER_ARM
TIER_T0 = "T0 (reads the eval GT): CONTROL 3 / CEILING diagnostic only, never a result"
TIERS = {"own": TIER_ARM, "identity": TIER_ARM, "cal": TIER_ARM, "perm": TIER_ARM, "gt": TIER_T0, "ceiling": TIER_T0}
#: control 4's permutation: a fixed seed, declared before any SPEEDCAL number
PERM_SEED = 20261025
#: a map fitted on fewer windows than this is refused (a sanity floor on degenerate input; the real fit set has thousands per cell)
MIN_FIT_N = 50
SHA256 = SA.sha256_file


def tag_of(mode: str, seed: int, target: str = "x") -> str:
    return TAGBASE[target].format(seed=seed) + "-" + mode


def fit_tag(seed: int) -> str:
    return TAGBASE["x"].format(seed=seed) + "-fit"


# ------------------------------------------------------------------------------------------------ the pure pieces (numpy only)
def masked_fit(x, y, m):
    """The closed-form least squares  argmin_{a,b} sum_{m} (a x + b - y)^2  in float64:  a = cov(x, y) / var(x),  b = mean(y) - a mean(x)  over the masked windows. -> (a, b, n)."""
    x = np.asarray(x, np.float64)[np.asarray(m, bool)]
    y = np.asarray(y, np.float64)[np.asarray(m, bool)]
    n = int(x.size)
    if n < 2:
        raise ValueError(f"a map cannot be fitted on {n} window(s)")
    xm, ym = x.mean(), y.mean()
    dx = x - xm
    sxx = float((dx * dx).sum())
    if not sxx > 0.0:
        raise ValueError("the predicted cell has no variance over the fit windows: the slope is undefined")
    a = float((dx * (y - ym)).sum() / sxx)
    return a, float(ym - a * xm), n


def fit_mask(x, y, has):
    """m_{w,c}: the window has a v9 row at T's tick AND a finite GT for cell c AND T's prediction is finite. [N, 5] bool."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return np.asarray(has, bool)[:, None] & np.isfinite(y) & np.isfinite(x)


def rmse(x, y, m, a=1.0, b=0.0):
    x, y, m = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(m, bool)
    return float(np.sqrt(np.mean((a * x[m] + b - y[m]) ** 2))) if m.any() else None


def fit_cells(x, y, has, min_n=MIN_FIT_N):
    """The five maps. x, y: [N, 5] physical units (m/s). -> dict(a, b, n [5 lists], rmse_before, rmse_after [5], mask [N, 5])."""
    mask = fit_mask(x, y, has)
    out = {"a": [], "b": [], "n": [], "rmse_before": [], "rmse_after": [], "mask": mask}
    for c in range(np.asarray(x).shape[1]):
        n_c = int(mask[:, c].sum())
        if n_c < min_n:
            raise ValueError(f"cell {c} has {n_c} fit windows with a GT (< {min_n}): refused")
        a, b, n = masked_fit(np.asarray(x)[:, c], np.asarray(y)[:, c], mask[:, c])
        out["a"].append(a)
        out["b"].append(b)
        out["n"].append(n)
        out["rmse_before"].append(rmse(np.asarray(x)[:, c], np.asarray(y)[:, c], mask[:, c]))
        out["rmse_after"].append(rmse(np.asarray(x)[:, c], np.asarray(y)[:, c], mask[:, c], a, b))
    return out


def permute_targets(y, mask, seed=PERM_SEED):
    """CONTROL 4's targets: per cell, the GT values of the masked windows PERMUTED across those windows (a fixed-seed permutation). The marginal of the target is kept and the pairing with T's
    prediction is destroyed, so a least-squares map on them collapses toward the mean speed (slope ~ 0)."""
    y = np.asarray(y, np.float64)
    out = y.copy()
    rng = np.random.default_rng(seed)
    for c in range(y.shape[1]):
        idx = np.flatnonzero(np.asarray(mask, bool)[:, c])
        out[idx, c] = y[rng.permutation(idx), c]
    return out


def std_to_phys(std, mu, sd):
    return np.asarray(std, np.float64) * np.asarray(sd, np.float64) + np.asarray(mu, np.float64)


def calibrate_std_np(std, a, b, mu, sd):
    """The map in the STANDARDISED cell O reads:  std' = a * std + ((a - 1) * mu + b) / sd  (float64, cast back to `std`'s dtype). a = 1, b = 0 -> std bit for bit."""
    std = np.asarray(std)
    a, b, mu, sd = (np.asarray(v, np.float64) for v in (a, b, mu, sd))
    k = ((a - 1.0) * mu + b) / sd
    return (a * std.astype(np.float64) + k).astype(std.dtype)


def map_block(fit: dict) -> dict:
    return {"a": [float(v) for v in fit["a"]], "b": [float(v) for v in fit["b"]], "n": [int(v) for v in fit["n"]]}


IDENTITY_MAP = {"a": [1.0] * 5, "b": [0.0] * 5}


# ------------------------------------------------------------------------------------------------ the hook (SPEEDAVG's, with another source for the five cells)
class SpeedCalHook(SA.SpeedAvgHook):
    """Wraps `V8FMini.forward_diag` for ONE eval. Same install / restore / cells / stats as SPEEDAVG's; only `__call__` differs: the five speed cells are X_s's own T cells passed through the affine map
    (modes identity / cal / perm / ceiling), or the GT where the window has a v9 row (mode gt, SPEEDAVG's own `source_code`)."""

    def __init__(self, M, mode: str, idx: list, a=None, b=None):
        super().__init__(M, mode, None, idx)
        if mode != "gt":
            if a is None or b is None or len(a) != len(idx) or len(b) != len(idx):
                raise ValueError(f"mode {mode!r} needs a map of {len(idx)} (a, b) pairs")
        self.a = None if a is None else np.asarray(a, np.float64)
        self.b = None if b is None else np.asarray(b, np.float64)

    def __call__(self, model, batch, code_override):
        import torch
        with torch.no_grad():
            own = model.forward_infer(batch)["code"]                       # X_s's own T code, exactly as the eval's first pass computed it
            has = batch["lab_t"]["has"].bool()
            gt = model._code_gt(batch["lab_t"]).to(own.dtype)
            if self.mode == "gt":
                src = SA.source_code("gt", own, None, gt, has)
            else:
                mu = model.cons_mu[self.ci].double().cpu().numpy()
                sd = model.cons_sd[self.ci].double().cpu().numpy()
                cal = calibrate_std_np(own[:, self.idx].cpu().numpy(), self.a, self.b, mu, sd)
                src = own.clone()
                src[:, self.idx] = torch.from_numpy(cal).to(device=own.device, dtype=own.dtype)
            new = SA.replace_cells(own, src, self.idx)
            self.nonspeed_dev = max(self.nonspeed_dev, float((code_override[:, self.non] - new[:, self.non]).abs().max()))
            self.oracle_dev = max(self.oracle_dev, float((code_override - new).abs().max()))
            self.rows["own_std"].append(own[:, self.idx].float().cpu().numpy())
            self.rows["partner_std"].append(torch.full_like(own[:, self.idx], float("nan")).float().cpu().numpy())
            self.rows["used_std"].append(new[:, self.idx].float().cpu().numpy())
            self.rows["gt_phys"].append(batch["lab_t"]["cons"][:, self.ci].float().cpu().numpy())
            self.rows["has"].append(has.cpu().numpy())
            if self.mu is None:
                self.mu, self.sd = model.cons_mu[self.ci].float().cpu().numpy(), model.cons_sd[self.ci].float().cpu().numpy()
            self.n_calls += 1
            return self.orig(model, batch, new)


# ------------------------------------------------------------------------------------------------ argv of the eval each mode rides
def carrier_mode(mode: str) -> str:
    """The `SA.e_argv` mode whose argv a SPEEDCAL mode rides: `own` is the standard eval (`--tac pred`); every hooked mode rides KINDX's `--tac oracle --tac-groups speed --tac-alpha 1` carrier."""
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}")
    return "own" if mode == "own" else "gt"


def e_argv(mode: str, ckpt: str, tag: str, paths: dict, extra: list | None = None) -> list:
    return SA.e_argv(carrier_mode(mode), ckpt, tag, paths, extra)


def extra_of(a) -> list:
    out = SA.extra_of(a)
    if not getattr(a, "no_families", False):
        out += ["--families"]            # the eval json then carries `families`, which `v8f_paired_families.py`'s reproduction control reads (SPEEDAVG's evals lacked it)
    return out


def file_sha_or_none(p):
    return SHA256(p) if p and os.path.exists(p) else None


# ------------------------------------------------------------------------------------------------ the fit
def window_ages(i: int) -> tuple:
    """The eval's own deterministic age schedule (`v8f_mini_eval.py run --ages cycle`) at window index i of the list."""
    a_t = i % 3
    return a_t, a_t + (i // 3) % 5


def run_fit(a, M=None, D=None, log=print):
    """X_s's T forward on the fit set's training windows, then the five masked least-squares maps and the permuted-target maps."""
    import time

    import torch
    torch.set_num_threads(int(os.environ.get("SPEEDCAL_THREADS", "8")))
    t0 = time.time()
    got = SHA256(a.x)
    if got != a.expect_sha:
        raise SystemExit(f"REFUSED: X_{a.seed} checkpoint sha256 {got} != the expected {a.expect_sha}")
    if M is None:
        M, _E = SA._load_modules()                                       # noqa: N806
        from tanitad.data import v8f_mini_data as D                      # noqa: N806
    idx = SA.speed_idx(M)
    ci_cells = [i - (M.N_GOAL + M.N_LAT + M.N_LON) for i in idx]
    ck = torch.load(a.x, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = SA.build_model(M, ck)
    with open(a.fit_clips, encoding="utf-8") as f:
        fc = json.load(f)
    clips = list(fc["clips"])
    sha_clips = hashlib.sha256("\n".join(clips).encode()).hexdigest()
    if sha_clips != fc.get("sha256_newline_joined"):
        raise SystemExit(f"REFUSED: the fit clip list hashes to {sha_clips}, its file says {fc.get('sha256_newline_joined')}")
    J = D.load_join(a.fit_join)
    tracks = D.load_native_tracks(a.tracks) if cfg.kin_mode == "yaw" else None
    store = D.ClipStore(a.fit_cache, "train", lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode=cfg.kin_mode, tracks=tracks)
    if a.limit:
        clips = clips[:max(1, a.limit)]
    wins = D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start)
    if a.limit:
        wins = wins[:a.limit]
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    N = len(wins)
    log(f"[fit s{a.seed}] clips {len(clips)} windows {N} (the trainer's own builder: train_windows, hist {cfg.hist}, start {ds.window_start}; ages cycle)")
    outs = {k: [] for k in ("own_std", "gt_phys", "has", "ci", "t", "a_t", "a_s")}
    with torch.no_grad():
        for b0 in range(0, N, a.batch):
            items = []
            for i in range(b0, min(b0 + a.batch, N)):
                ci, t, _, _ = wins[i]
                a_t, a_s = window_ages(i)
                items.append(ds.item(ci, t, a_t, a_s))
            b = D.collate(items)
            o = model.forward_infer(b)                                   # the full inference path: no label is read (the key guard); T's own cells are `code[:, idx]`
            outs["own_std"].append(o["code"][:, idx].float().numpy())
            outs["gt_phys"].append(b["lab_t"]["cons"][:, ci_cells].float().numpy())
            outs["has"].append(b["lab_t"]["has"].bool().numpy())
            for k in ("ci", "t", "a_t", "a_s"):
                outs[k].append(b[k].numpy())
            if (b0 // a.batch) % 20 == 0:
                log(f"[fit s{a.seed}] {b0 + len(items)}/{N}  {time.time() - t0:.0f}s")
    O = {k: np.concatenate(v) for k, v in outs.items()}
    hb = None
    if a.hblend_windows:
        z = np.load(a.hblend_windows)
        n_cmp = N if not a.limit else min(N, len(z["ci"]))
        bad = [k for k in ("ci", "t", "a_t", "a_s") if not np.array_equal(O[k][:n_cmp], z[k][:n_cmp])]
        if bad:
            raise SystemExit(f"REFUSED: the fit windows differ from HBLEND's {a.hblend_windows} in {bad}: not the same fit set")
        if not a.limit and len(z["ci"]) != N:
            raise SystemExit(f"REFUSED: {N} fit windows, HBLEND's list has {len(z['ci'])}")
        hb = {"file": a.hblend_windows, "sha256": SHA256(a.hblend_windows), "n_compared": int(n_cmp), "equal": True, "keys": ["ci", "t", "a_t", "a_s"]}
    mu = model.cons_mu[ci_cells].double().numpy()
    sd = model.cons_sd[ci_cells].double().numpy()
    x = std_to_phys(O["own_std"], mu, sd)
    y = O["gt_phys"].astype(np.float64)
    min_n = int(getattr(a, "min_fit_n", MIN_FIT_N))
    F = fit_cells(x, y, O["has"], min_n)
    y_perm = permute_targets(y, F["mask"], PERM_SEED)
    Fp = fit_cells(x, y_perm, O["has"], min_n)
    cells_path = os.path.join(a.out_dir, f"fit_cells_{fit_tag(a.seed)}.npz")
    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(cells_path, own_std=O["own_std"], gt_phys=O["gt_phys"], perm_gt_phys=y_perm.astype(np.float32), has=O["has"], mask=F["mask"], ci=O["ci"], t=O["t"], a_t=O["a_t"], a_s=O["a_s"],
                        cons_mu=model.cons_mu[ci_cells].float().numpy(), cons_sd=model.cons_sd[ci_cells].float().numpy(), speed_idx=np.asarray(idx))
    rec = {"schema": FIT_SCHEMA, "seed": a.seed, "tag": fit_tag(a.seed), "x_ckpt": a.x, "x_sha256": got, "speed_cells": list(SPEED_CELLS), "speed_idx": list(idx),
           "cons_mu": [float(v) for v in model.cons_mu[ci_cells].float()], "cons_sd": [float(v) for v in model.cons_sd[ci_cells].float()],
           "tier": "fit on TRAINING windows only; no eval window enters (the fit clips are train clips)",
           "fit_set": {"fit_clips": a.fit_clips, "fit_clips_sha256": fc.get("sha256_newline_joined"), "n_clips": len(clips), "clips": list(clips), "n_windows": int(N),
                       "window_builder": f"tanitad.data.v8f_mini_data.train_windows(hist={cfg.hist}, start={ds.window_start}) -- the trainer's; ages = the eval's own cycle schedule",
                       "join_train": a.fit_join, "join_train_sha256": SHA256(a.fit_join), "cache": a.fit_cache, "hblend_windows": hb, "label": "v9 TRAIN labels as the trainer's join (cons at T's tick frame, lab_t)",
                       "mask": "has (a v9 row at T's tick) & finite GT cell & finite prediction == the trainer's `cons_absent=mask`"},
           "map": {**map_block(F), "rmse_train_before": F["rmse_before"], "rmse_train_after": F["rmse_after"], "form": "c' = a*c + b in m/s; closed-form masked least squares, float64"},
           "perm": {"seed": PERM_SEED, **map_block(Fp), "rmse_train_before": Fp["rmse_before"], "rmse_train_after": Fp["rmse_after"],
                    "what": "the same fit on the cell's training targets PERMUTED across the masked windows (control 4)"},
           "per_cell": {nm: {"a": F["a"][c], "b": F["b"][c], "n": F["n"][c], "rmse_train_before": F["rmse_before"][c], "rmse_train_after": F["rmse_after"][c],
                             "mean_pred": float(x[F["mask"][:, c], c].mean()), "mean_gt": float(y[F["mask"][:, c], c].mean()),
                             "sd_pred": float(x[F["mask"][:, c], c].std()), "sd_gt": float(y[F["mask"][:, c], c].std())} for c, nm in enumerate(SPEED_CELLS)},
           "limit": a.limit or None, "seconds": time.time() - t0, "fit_cells_npz": cells_path}
    out = os.path.join(a.out_dir, f"fit_{fit_tag(a.seed)}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    log(f"[fit s{a.seed}] DONE {time.time() - t0:.0f}s a={['%.3f' % v for v in F['a']]} b={['%.3f' % v for v in F['b']]} n={F['n']}")
    return 0


# ------------------------------------------------------------------------------------------------ the evals
def finalize_json(js_path: str, block: dict, mode: str):
    """Relabel the eval's carrier fields (it wrote `tac: oracle` because that is the carrier path; in identity / cal / perm / ceiling the cells are NOT the GT) and add the `speedcal` block."""
    with open(js_path, encoding="utf-8") as f:
        j = json.load(f)
    if mode != "own":
        j["tac_carrier_as_written"] = j.get("tac")
        j["tac"] = f"speedcal-{mode}"
        tm = j.get("tac_manipulation")
        if isinstance(tm, dict):
            tm["carrier_mode_as_written"] = tm.get("mode")
            tm["mode"] = f"speedcal-{mode}"
    j["speedcal"] = block
    with open(js_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(j, f, indent=1)


def eval_done(eval_dir: str, tag: str, mode: str) -> bool:
    npz, js = os.path.join(eval_dir, f"eval_{tag}.npz"), os.path.join(eval_dir, f"eval_{tag}.json")
    if not (os.path.exists(npz) and os.path.exists(js)):
        return False
    if mode != "own" and not os.path.exists(os.path.join(eval_dir, f"speedcal_cells_{tag}.npz")):
        return False
    try:
        with open(js, encoding="utf-8") as f:
            return "speedcal" in json.load(f)
    except (OSError, ValueError):
        return False


def ceiling_map_from_cells(cells: dict, min_n=MIN_FIT_N) -> dict:
    """The CEILING: the same affine form fitted on the EVAL GT. `cells` = the identity eval's dump (own_std, gt_phys, has, cons_mu, cons_sd). T0: it reads the eval labels."""
    x = std_to_phys(cells["own_std"], cells["cons_mu"], cells["cons_sd"])
    y = np.asarray(cells["gt_phys"], np.float64)
    F = fit_cells(x, y, np.asarray(cells["has"], bool), min_n)
    return {**map_block(F), "rmse_eval_before": F["rmse_before"], "rmse_eval_after": F["rmse_after"]}


def load_fit(path: str, seed: int, idx: list) -> dict:
    with open(path, encoding="utf-8") as f:
        fit = json.load(f)
    if fit.get("schema") != FIT_SCHEMA:
        raise SystemExit(f"REFUSED: {path} is not a {FIT_SCHEMA} file")
    if fit.get("seed") != seed:
        raise SystemExit(f"REFUSED: {path} is the fit of seed {fit.get('seed')}, not {seed}")
    if list(fit.get("speed_idx", [])) != list(idx) or list(fit.get("speed_cells", [])) != list(SPEED_CELLS):
        raise SystemExit(f"REFUSED: {path} was fitted on cells {fit.get('speed_cells')} / {fit.get('speed_idx')}, the model's speed cells are {list(SPEED_CELLS)} / {list(idx)}")
    if fit.get("limit"):
        raise SystemExit(f"REFUSED: {path} is a LIMITED (smoke) fit: never a result")
    return fit


def run_evals(a, M=None, E=None, log=print):
    """The modes of X_<seed> (or of the baseline B with --target b) in this process. M = the tip's `v8f_mini`, E = its `v8f_mini_eval` (injectable for the tests)."""
    import torch
    torch.set_num_threads(int(os.environ.get("SPEEDCAL_THREADS", "8")))
    target = a.target
    allowed = MODES if target == "x" else B_MODES
    bad = [m for m in a.modes if m not in allowed]
    if bad:
        raise SystemExit(f"REFUSED: mode(s) {bad} are not available for --target {target} (allowed: {allowed})")
    order = [m for m in MODES if m in a.modes]
    if list(a.modes) != order:
        raise SystemExit(f"REFUSED: the modes must be given in the order {MODES} (ceiling is fitted on identity's dump); got {list(a.modes)}")
    if not os.path.exists(a.x):
        raise SystemExit(f"REFUSED: checkpoint {a.x} does not exist")
    got = SHA256(a.x)
    if got != a.expect_sha:
        raise SystemExit(f"REFUSED: checkpoint sha256 {got} != the expected {a.expect_sha}")
    if M is None:
        M, E = SA._load_modules()                                        # noqa: N806
    idx = SA.speed_idx(M)
    fit = None
    if any(m in ("cal", "perm") for m in a.modes):
        if not a.fit_json:
            raise SystemExit("REFUSED: --fit-json is required for the modes cal / perm")
        fit = load_fit(a.fit_json, a.seed, idx)
        if target == "x" and fit.get("x_sha256") != got:
            raise SystemExit(f"REFUSED: the fit was made on checkpoint {fit.get('x_sha256')}, this one is {got}")
    paths, extra = SA.paths_of(a), extra_of(a)
    os.makedirs(a.eval_dir, exist_ok=True)
    n_run = 0
    for mode in a.modes:
        tag = tag_of(mode, a.seed, target)
        if eval_done(a.eval_dir, tag, mode):
            log(f"[skip] eval {tag}")
            continue
        if mode == "identity":
            mp, msrc = IDENTITY_MAP, "identity (a = 1, b = 0)"
        elif mode == "cal":
            mp, msrc = fit["map"], f"fit json {os.path.basename(a.fit_json)} (sha256 {SHA256(a.fit_json)[:16]}), the TRAINING-window least squares"
        elif mode == "perm":
            mp, msrc = fit["perm"], f"fit json {os.path.basename(a.fit_json)}, the PERMUTED-target least squares (control 4)"
        elif mode == "ceiling":
            idtag = tag_of("identity", a.seed, target)
            cp = os.path.join(a.eval_dir, f"speedcal_cells_{idtag}.npz")
            if not os.path.exists(cp):
                raise SystemExit(f"REFUSED: mode ceiling needs the identity eval's cells ({cp}): run identity first")
            zc = np.load(cp)
            mp = ceiling_map_from_cells({k: zc[k] for k in zc.files}, int(getattr(a, "min_fit_n", MIN_FIT_N)))
            msrc = "fitted on the EVAL GT (T0 diagnostic) from the identity eval's cells"
            with open(os.path.join(a.eval_dir, f"ceiling_map_{tag}.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"seed": a.seed, "tag": tag, "source": msrc, **mp}, f, indent=1)
        else:
            mp, msrc = None, None
        argv = e_argv(mode, a.x, tag, paths, extra)
        log("EVAL " + " ".join(argv))
        hook = None if mode == "own" else SpeedCalHook(M, mode, idx, None if mode == "gt" else mp["a"], None if mode == "gt" else mp["b"]).install()
        undo = SA.install_limit(E, a.limit) if a.limit else None
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
        block = {"schema": SCHEMA, "mode": mode, "seed": a.seed, "target": target, "tag": tag, "tier": TIERS[mode], "limit": a.limit or None, "x_ckpt": a.x, "x_sha256": got,
                 "speed_cells": list(SPEED_CELLS), "speed_idx": list(idx), "form": "c' = a*c + b in m/s on T's own cell; written into the standardised cell O reads: std' = a*std + ((a-1)*mu + b)/sd",
                 "map": None if mp is None else {"a": [float(v) for v in mp["a"]], "b": [float(v) for v in mp["b"]], "source": msrc},
                 "fit_json": a.fit_json if mode in ("cal", "perm") else None, "fit_json_sha256": file_sha_or_none(a.fit_json) if mode in ("cal", "perm") else None,
                 "carrier": None if mode == "own" else "v8f_mini_eval --tac oracle --tac-groups speed (carrier path only: the cells O read are NOT the GT unless mode == gt)"}
        if hook is not None:
            block["hook"] = hook.stats()
            np.savez_compressed(os.path.join(a.eval_dir, f"speedcal_cells_{tag}.npz"), **hook.cells())
        finalize_json(js, block, mode)
        log(f"[ok] eval {tag}")
        n_run += 1
    after = SHA256(a.x)
    if after != got:
        raise SystemExit(f"FAIL: the checkpoint changed during the evals: {after} != {got}")
    log(f"SPEEDCAL evals done for seed {a.seed} target {target}: {n_run} run, {len(a.modes) - n_run} skipped; the checkpoint's sha256 unchanged")
    return 0


# ------------------------------------------------------------------------------------------------ CLI
def cmd_fit(a):
    if a.dry_run:
        print(f"fit seed {a.seed}: X_s's T forward on the fit windows of {a.fit_clips} ({a.fit_cache}, join {a.fit_join}), masked least squares per speed cell -> {a.out_dir}/fit_{fit_tag(a.seed)}.json")
        print(f"  windows: train_windows at the checkpoint's hist / window_start, ages cycle; checked equal to {a.hblend_windows}")
        print(f"  permuted-target map (control 4): seed {PERM_SEED}")
        return 0
    if not a.expect_sha:
        raise SystemExit("REFUSED: --expect-sha is required (the sha256 of the checkpoint as measured by the caller)")
    return run_fit(a)


def cmd_evals(a):
    if a.dry_run:
        for mode in a.modes:
            print("v8f_mini_eval.py " + " ".join(e_argv(mode, a.x, tag_of(mode, a.seed, a.target), SA.paths_of(a), extra_of(a))))
        return 0
    if not a.expect_sha:
        raise SystemExit("REFUSED: --expect-sha is required (the sha256 of the checkpoint as measured by the caller)")
    return run_evals(a)


def _common_data(p):
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


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--seed", type=int, choices=(0, 1), required=True)
    p.add_argument("--x", required=True, help="the checkpoint X_s = M2b-mixo_s<seed>")
    p.add_argument("--expect-sha")
    p.add_argument("--fit-clips", required=True, help="HBLEND's fit_clips.json (141 train clips)")
    p.add_argument("--fit-cache", required=True, help="the training feature cache root (contains train/clips/<clip>.pt)")
    p.add_argument("--fit-join", required=True, help="the trainer's join_train.npz (the v9 TRAIN labels on the cache grid)")
    p.add_argument("--tracks", required=True)
    p.add_argument("--hblend-windows", default=None, help="HBLEND's fit_windows_*.npz: the window list is CHECKED equal to it")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lru", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="SMOKE ONLY: the first N windows (the json records it; evals and the reader refuse a limited fit)")
    p.add_argument("--min-fit-n", type=int, default=MIN_FIT_N, help="refuse a cell with fewer fit windows with a GT (a sanity floor; the real fit set has thousands)")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("evals")
    p.add_argument("--seed", type=int, choices=(0, 1), required=True)
    p.add_argument("--target", choices=("x", "b"), default="x", help="x = X_s (the arm); b = the baseline B = M2b_s<seed> (modes own / cal only: X_s's map on B, beside)")
    p.add_argument("--x", required=True, help="the checkpoint to evaluate (X_s, or B with --target b)")
    p.add_argument("--expect-sha")
    p.add_argument("--fit-json", default=None)
    p.add_argument("--modes", type=lambda s: tuple(x for x in s.split(",") if x), default=MODES)
    _common_data(p)
    p.add_argument("--limit", type=int, default=0, help="SMOKE ONLY: the first N windows of the grid (the json records it; the reader refuses it)")
    p.add_argument("--min-fit-n", type=int, default=MIN_FIT_N, help="the CEILING fit's sanity floor per cell")
    p.add_argument("--no-families", action="store_true", help="do not add --families to the eval (the paired-families tool's reproduction control then cannot pass)")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("sha")
    p.add_argument("path")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.mode == "sha":
        print(SHA256(a.path))
        return 0
    if a.mode == "fit":
        return cmd_fit(a)
    bad = [m for m in a.modes if m not in MODES]
    if bad:
        raise SystemExit(f"unknown mode(s) {bad}: use {MODES}")
    return cmd_evals(a)


if __name__ == "__main__":
    sys.exit(main())
