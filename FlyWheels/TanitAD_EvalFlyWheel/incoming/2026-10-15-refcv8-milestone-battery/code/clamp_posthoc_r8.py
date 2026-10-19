"""clamp_posthoc_r8.py -- the POST-HOC plan_vmax_clamp read of the banked refcv8 milestone trees (AMENDMENT A-R8-CLAMP-1 section 2.3).  CPU only.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 \
    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
      python -X utf8 clamp_posthoc_r8.py vendor                # task 1: verify the vendored landed function + dependency blobs
      python -X utf8 clamp_posthoc_r8.py exactness             # task 2/3: source quotes (file:line) + per-tree empirical identity checks
      python -X utf8 clamp_posthoc_r8.py read --tree <tree>    # tasks 4/5: R1 gap + K1/K3 + the clamp-ON paired read, one tree, BOTH seeds
      python -X utf8 clamp_posthoc_r8.py assemble              # task 6: CLAMP_POSTHOC.json / .md

WHAT IT DOES.  The landed function `refcv6_selection.cap_plan_to_limit` (commit de4e15b, VENDORED byte-for-byte, NOT re-implemented) is applied
to the banked EMITTED plan `dump_s<seed>/refcv6_extras.npz :: nav_true.traj` [N, 8, 2] (the full 0.5-6 s plan = `out["traj"]` after E9), with the
per-window fed ceiling and v0 of the same dump and the horizons / anchor_dt / sel.accel_max of the run.  The clamped plan then REPLACES `os` through
the SAME substitution + `pair_read_r8.read_surface` path `nr_matched_instrument.py` uses (IMPORTED, not copied), so the four families and the SPEC
section 5 cells are the battery's own functions on a different plan.

PRE-REGISTERED CONTROLS (written before any clamp-ON metric was looked at; only R1-gap descriptive counts had been seen):
  K1  INERT LIMIT.  Limit = +inf on every row.  (a) the function returns the plan BIT-IDENTICALLY; (b) the full pipeline (extras -> substitution ->
      read_surface) reproduces EVERY banked PAIRED_READ cell for that tree / seed / surface exactly (delta, lo, hi, n_windows, n_episodes, separated,
      n_dropped_nonfinite, and the means).  Any miss => no clamp-ON number is written for that tree / seed / surface.
  K3  MISPAIRING.  The per-window limits rolled by 35 windows (about one episode block) must (a) change the binding set on >= 1 row at every
      tree / seed; (b) on the primary tree (step30000ema_unc, seed 0, S2 and S6) change >= 1 NR headline delta of the clamp-ON read.
  K2 / K4 need the GPU (flag-off bit parity of the clamp-carrying eval tree; post-hoc vs eval-forward clamp): NOT RUN HERE, listed as OWED.
  P   POSTCONDITION (extra, analytic): after the clamp, every waypoint interval speed <= max(limit, v0 - brake * t_mid) + 1e-4 m/s, and every
      non-binding row is bit-identical to the unclamped plan.

TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  ESTIMATOR / QUESTION: paired episode-cluster bootstrap over the 139
eval episodes (B 2000, seed 0, 95 % percentile) with the models FIXED -- "another draw of EPISODES" only; NOT another training run, NOT another
inference run.  Rates in the R1-gap tables are descriptive means with the same cluster bootstrap (mean of a 0/1 indicator vs zeros).  Single training seed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
OUT = PKG / "raw" / "clamp_posthoc"
ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
SCRATCH = os.environ.get("R8_CLAMP_SCRATCH",
                         "C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/"
                         "f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/clamp_scratch")
GD = "C:/Users/Admin/tanitad-push/.git"
LGS = "C:/lgs/Lfb47a58376"
CFG = "E:/refcv8_eval_kit/ckpt/config.json"
VENDORED = HERE / "_vendored_de4e15b_refcv6_selection.py"
SRC_COMMIT, SRC_PATH = "de4e15b", "stack/tanitad/refs/refcv6_selection.py"
SRC_BLOB = "b4e0e80439eda771b972d9556180feff55f4d66c"
HEADER_LINES = 5
PRIMARY = "step30000ema_unc"
TREES = [PRIMARY, "step20000_unc", "step20000ema_unc", "step15000_unc", "step15000ema_unc", "step10500_unc", "step10500ema_unc",
         "step5000_unc", "step5000ema_unc"]
SEEDS = (0, 1)
N_BOOT = 2000
MIS_ROLL = 35
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
QUESTION = ("episode-cluster bootstrap over the 139 eval episodes, models FIXED: answers 'another draw of EPISODES' only; "
            "NOT another training run, NOT another inference run")
TICK_S = 0.1          # refc.py:1748  self.anchor_dt = 0.1  (a decoder attribute, not a config key)
POST_TOL = 1e-4       # m/s, postcondition tolerance (fp32)
EXEMPT_TOL_COARSE = 0.01   # m/s: the "real excess" tolerance beside the strict > 0 one


# ------------------------------------------------------------------------------------------ small helpers
def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, p):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def blob_sha(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def git_show(rev: str, path: str) -> bytes:
    """READ-ONLY: git --git-dir=<push repo> show <rev>:<path> (the only way git objects are read here)."""
    r = subprocess.run(["git", f"--git-dir={GD}", "show", f"{rev}:{path}"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise SystemExit(f"[clamp_posthoc] git show {rev}:{path} failed (rc {r.returncode}): {r.stderr[:200]!r}")
    return r.stdout


def git_rev_parse(rev: str, path: str) -> str:
    r = subprocess.run(["git", f"--git-dir={GD}", "rev-parse", f"{rev}:{path}"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"UNAVAILABLE(rc {r.returncode})"


def find_line(text: str, needle: str, nth: int = 1) -> dict:
    """{line, text} of the nth line containing `needle` (1-based), or line None when absent."""
    k = 0
    for i, ln in enumerate(text.splitlines(), 1):
        if needle in ln:
            k += 1
            if k == nth:
                return {"line": i, "text": ln.strip()[:150], "needle": needle}
    return {"line": None, "text": None, "needle": needle}


def cluster_rate(x: np.ndarray, eid: np.ndarray) -> dict:
    """Mean of a 0/1 (or real) per-window indicator with the SAME cluster bootstrap (paired against zeros)."""
    from taniteval import ci
    x = np.asarray(x, dtype=np.float64)
    r = ci.paired_episode_cluster_bootstrap(x, np.zeros_like(x), list(map(int, eid)), n_boot=N_BOOT, seed=0)
    return {"mean": round(float(x.mean()), 6), "lo": r["lo"], "hi": r["hi"], "n": int(len(x)), "n_episodes": int(r["n_episodes"])}


# ------------------------------------------------------------------------------------------ task 1: the vendored function
def load_vendored():
    raw = VENDORED.read_bytes()
    nl = [i for i, b in enumerate(raw) if b == 0x0A][HEADER_LINES - 1]
    body = raw[nl + 1:]
    got = blob_sha(body)
    src = git_show(SRC_COMMIT, SRC_PATH)
    if got != SRC_BLOB or src != body or blob_sha(src) != SRC_BLOB:
        raise SystemExit(f"[clamp_posthoc] vendored body {got} != expected blob {SRC_BLOB} / git show differs: REFUSING")
    spec = importlib.util.spec_from_file_location("_vendored_de4e15b_refcv6_selection", str(VENDORED))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_vendored_de4e15b_refcv6_selection"] = mod
    spec.loader.exec_module(mod)
    for need in ("cap_plan_to_limit", "planned_max_speed", "step_durations_s"):
        if not hasattr(mod, need):
            raise SystemExit(f"[clamp_posthoc] vendored module lacks {need}")
    rec = {"vendored_file": str(VENDORED), "source_commit": SRC_COMMIT, "source_path": SRC_PATH,
           "git_rev_parse_source_blob": git_rev_parse(SRC_COMMIT, SRC_PATH), "expected_blob": SRC_BLOB,
           "body_blob_sha_recomputed": got, "body_equals_git_show_bytes": True, "header_lines": HEADER_LINES,
           "vendored_file_git_blob_sha": blob_sha(raw), "vendored_file_sha256": hashlib.sha256(raw).hexdigest(),
           "vendored_file_bytes": len(raw)}
    return mod, rec


def dependency_record() -> dict:
    """The vendored module imports tanitad.models.vocab_v7, tanitad.refs.refc_selector_targets, tanitad.refs.refcv6_tactical; the call site also uses
    refcv6_max_speed.  Resolved on PYTHONPATH against the fb47a58 tree -> compare their blobs to de4e15b's."""
    paths = ["stack/tanitad/models/vocab_v7.py", "stack/tanitad/refs/refc_selector_targets.py", "stack/tanitad/refs/refcv6_tactical.py",
             "stack/tanitad/refs/refcv6_max_speed.py"]
    out = {}
    for p in paths:
        wt = Path(LGS) / p
        out[p] = {"de4e15b": git_rev_parse("de4e15b", p), "fb47a58": git_rev_parse("fb47a58", p),
                  "fb47a58_worktree_C:/lgs": blob_sha(wt.read_bytes()) if wt.exists() else "ABSENT"}
        out[p]["identical_de4e15b_vs_fb47a58"] = bool(out[p]["de4e15b"] == out[p]["fb47a58"] and len(out[p]["de4e15b"]) == 40)
        out[p]["worktree_is_fb47a58"] = bool(out[p]["fb47a58_worktree_C:/lgs"] == out[p]["fb47a58"] and len(out[p]["fb47a58"]) == 40)
    for p in ("stack/tanitad/refs/refcv6_selection.py", "stack/tanitad/refs/refc_v3.py"):
        wt = Path(LGS) / p
        a, b = git_show("fb47a58", p), git_show("de4e15b", p)
        out[p] = {"de4e15b": git_rev_parse("de4e15b", p), "fb47a58": git_rev_parse("fb47a58", p),
                  "fb47a58_worktree_C:/lgs": blob_sha(wt.read_bytes()) if wt.exists() else "ABSENT",
                  "identical_de4e15b_vs_fb47a58": False,
                  "worktree_is_fb47a58": bool(wt.exists() and blob_sha(wt.read_bytes()) == git_rev_parse("fb47a58", p)),
                  "note": "differs ONLY by ADDED lines (the clamp): " + _added_only(a, b)}
    return out


def _added_only(old: bytes, new: bytes) -> str:
    import difflib
    o, n = old.decode("utf-8").splitlines(), new.decode("utf-8").splitlines()
    removed = added = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, o, n, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            removed += i2 - i1
        if tag in ("insert", "replace"):
            added += j2 - j1
    return f"removed {removed} lines, added {added} lines (fb47a58 -> de4e15b)"


# ------------------------------------------------------------------------------------------ inputs
def run_cfg() -> dict:
    c = jload(CFG)
    hz = [int(x) for x in c["horizons"]]
    return {"config_json": CFG, "horizons": hz, "tick_s": TICK_S, "brake_ms2": float(c["selection"]["sel_accel_max"]),
            "refcv8_speed_input": c["refcv8"].get("speed_input"), "refcv8_speed_enc8": bool(c["refcv8"].get("speed_enc8")),
            "argv_has_refcv7": bool([a for a in c["argv"] if a.startswith("--refcv7")]),
            "argv_has_sel_accel_max_flag": "--sel-accel-max" in c["argv"],
            "sources": {"horizons": "config.json['horizons'] == cfg.core.trajectory.horizons == decoder.anchor_horizons (refc.py:1769)",
                        "tick_s": "decoder.anchor_dt = 0.1 (refc.py:1748, a constant attribute)",
                        "brake_ms2": "config.json['selection']['sel_accel_max'] == decoder.sel.accel_max (refc.py:1067); the run argv carries no --sel-accel-max"}}


def fed_ceiling(vmax_ms: np.ndarray, valid: np.ndarray, rc: dict):
    """The value `plan_v_lim_ms` carries (de4e15b refc_v3.py:1950-1966 on the fb47a58 line numbering offset), derived with the MODULE'S OWN encoder:
         vmax_1h, over = MaxSpeedOneHotEncoder()(v_max_ms, v_max_valid);  v_lim = limit_ms_of_bin(argmax);  all-zero row -> +inf (X15);
         refcv8 speed_input='label': an `over` row takes the fed value itself (never below it); speed_enc8 False here."""
    import torch
    from tanitad.refs import refcv6_max_speed as v6ms
    enc = v6ms.MaxSpeedOneHotEncoder()
    vm, vv = torch.from_numpy(vmax_ms.astype(np.float32)), torch.from_numpy(valid.astype(np.float32))
    oh, over = enc(vm, vv)
    lim = v6ms.limit_ms_of_bin(oh.argmax(dim=-1)).to(torch.float32)
    known = oh.sum(dim=-1) > 0.5
    lim = torch.where(known, lim, torch.full_like(lim, float("inf")))
    if str(rc["refcv8_speed_input"] or ""):
        lim = torch.where(over.reshape(-1) & known, vm.reshape(-1).to(lim.dtype), lim)
        if rc["refcv8_speed_enc8"]:
            lim = torch.where(known, vm.reshape(-1).to(lim.dtype), torch.full_like(lim, float("inf")))
    # the bin must not depend on the model's compute dtype for these ladder values (fp32 vs bf16): checked, not assumed
    oh16, _ = enc(vm.bfloat16(), vv.bfloat16())
    return lim, {"n_rows": int(len(vm)), "n_known": int(known.sum()), "n_unknown_inf": int((~known).sum()), "n_over_ceiling": int(over.sum()),
                 "limit_values_ms": sorted({round(float(x), 6) for x in lim[torch.isfinite(lim)].unique().tolist()}),
                 "bins_equal_fp32_vs_bf16": bool((oh16.argmax(-1) == oh.argmax(-1)).all())}


def load_seed_inputs(tree: str, seed: int, rc: dict):
    d = f"{ARCHIVE}/{tree}/dump_s{seed}/refcv6_extras.npz"
    z = np.load(d)
    need = ["clip_index", "ws", "nav_true.traj", "nav_true.sel_idx", "v0", "v_max_ms", "v_max_valid"]
    miss = [k for k in need if k not in z.files]
    if miss:
        raise SystemExit(f"[clamp_posthoc] {d}: lacks {miss}")
    raw = {k: z[k] for k in need}
    raw["filter_off.traj"] = z["filter_off.traj"] if "filter_off.traj" in z.files else None
    raw["gt_max_speed_6s"] = z["gt_max_speed_6s"].astype(np.float64) if "gt_max_speed_6s" in z.files else None
    for k in ("v0", "v_max_ms", "v_max_valid", "nav_true.traj"):
        if not np.isfinite(raw[k]).all():
            raise SystemExit(f"[clamp_posthoc] {d}: non-finite {k}")
    lim, linfo = fed_ceiling(raw["v_max_ms"], raw["v_max_valid"], rc)
    return raw, lim, linfo, d


# ------------------------------------------------------------------------------------------ the clamp + the R1 gap
def apply_clamp(V, plan: np.ndarray, lim, v0: np.ndarray, rc: dict, brake: float | None = None):
    import torch
    t = torch.from_numpy(plan.astype(np.float32))
    out, diag = V.cap_plan_to_limit(t, lim, horizons=rc["horizons"], tick_s=rc["tick_s"], v0=torch.from_numpy(v0.astype(np.float32)),
                                    brake_ms2=float(rc["brake_ms2"] if brake is None else brake))
    return out, diag


def planned_speed(V, plan: np.ndarray, rc: dict, slots=None) -> np.ndarray:
    import torch
    t = torch.from_numpy(plan.astype(np.float32))
    if slots is not None:
        # a prefix grid [0..k) keeps the module's own derivation: horizons[:k]
        t = t[:, :len(slots)]
        hz = [rc["horizons"][i] for i in slots]
    else:
        hz = rc["horizons"]
    return V.planned_max_speed(t[:, None], horizons=hz, tick_s=rc["tick_s"])[:, 0].numpy()


def q3(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0, "median": None, "p90": None, "max": None}
    return {"n": int(len(x)), "median": round(float(np.median(x)), 4), "p90": round(float(np.quantile(x, 0.9)), 4), "max": round(float(x.max()), 4)}


def r1_gap(V, plan, lim, v0, eid, rc, mask=None, label="", gtmax=None) -> dict:
    """The R1 gap of an EMITTED (clamp-OFF) plan on the waypoint grid, with the module's own planned_max_speed."""
    import torch
    sel = np.ones(len(plan), bool) if mask is None else mask
    p, v, l, e = plan[sel], v0[sel], lim.numpy()[sel], eid[sel]
    pm = planned_speed(V, p, rc)
    fin = np.isfinite(l)
    exc = pm - l
    ex_strict = fin & (exc > 0)
    ex_coarse = fin & (exc > EXEMPT_TOL_COARSE)
    v0over = fin & (v > l)
    out_c, diag = apply_clamp(V, p, torch.from_numpy(l.astype(np.float32)), v, rc)
    bind = diag["plan_cap_binding"].numpy()
    resid = diag["plan_cap_excess_ms"].numpy()
    pm2 = planned_speed(V, p, rc, slots=[0, 1, 2, 3])
    ex2 = fin & ((pm2 - l) > 0)
    res = {"label": label, "n_windows": int(len(p)), "n_episodes": int(len(np.unique(e))), "n_finite_ceiling": int(fin.sum()),
           "exempt_rate_strict_full_plan_0_6s": cluster_rate(ex_strict.astype(float), e), "n_exempt_strict": int(ex_strict.sum()),
           "exempt_rate_gt_0p01_ms": cluster_rate(ex_coarse.astype(float), e), "n_exempt_gt_0p01": int(ex_coarse.sum()),
           "exempt_rate_strict_s2_slots_0_2s": cluster_rate(ex2.astype(float), e), "n_exempt_s2_slots": int(ex2.sum()),
           "n_episodes_with_an_exempt_row": int(len(np.unique(e[ex_strict]))),
           "excess_ms_on_exempt_rows": q3(exc[ex_strict]),
           "excess_ms_s2_slots_on_exempt_rows": q3((pm2 - l)[ex2]),
           "planned_max_speed_ms_on_exempt_rows": q3(pm[ex_strict]),
           "ceiling_ms_on_exempt_rows": {str(round(float(k), 4)): int(c) for k, c in zip(*np.unique(l[ex_strict], return_counts=True))},
           "v0_over_ceiling_rate": cluster_rate(v0over.astype(float), e), "n_v0_over": int(v0over.sum()),
           "exempt_and_v0_over": int((ex_strict & v0over).sum()), "exempt_and_v0_not_over": int((ex_strict & ~v0over).sum()),
           "clamp_binding_rate": cluster_rate(bind.astype(float), e), "n_binding": int(bind.sum()),
           "residual_above_ceiling_after_clamp_rate": cluster_rate((resid > EXEMPT_TOL_COARSE).astype(float), e), "n_residual": int((resid > EXEMPT_TOL_COARSE).sum()),
           "residual_excess_ms_after_clamp": q3(resid[resid > EXEMPT_TOL_COARSE]),
           "n_residual_strict_gt0_incl_fp_dust": int((resid > 0).sum()),
           "v0_minus_ceiling_ms_on_v0_over_rows": q3((v - l)[v0over]),
           **_gt_descriptors(gtmax, sel, l, fin, ex_strict, e),
           "note": ("exempt = planned max speed of the emitted plan on the 0.5-s/1-s waypoint grid (module planned_max_speed) > the fed ceiling, "
                    "strict (tol 0, as SpeedCeilingFilter's default); a +inf (unknown) ceiling is never exempt; 'residual' = rows still above the "
                    "ceiling AFTER the clamp (the v0-braking allowance max(limit, v0 - brake*t_mid) lets an over-ceiling ego decelerate gradually)")}
    return res


def _gt_descriptors(gtmax, sel, l, fin, ex_strict, e) -> dict:
    """Is the fed ceiling a ceiling for the EGO ITSELF?  gt_max_speed_6s = the max of the ego's own realised speed over the window's valid 6-s future
    (extras `_window`, r7_roll.py:213-215 neighbourhood).  A ceiling the logged ego itself exceeds is a label property, not a speed limit."""
    if gtmax is None:
        return {"gt_descriptors": "ABSENT (extras carry no gt_max_speed_6s)"}
    g = gtmax[sel]
    ok = fin & np.isfinite(g)
    over = ok & (g > l)
    out = {"gt_max_speed_exceeds_ceiling_rate_all_windows": cluster_rate(over.astype(float), e), "n_gt_exceeds": int(over.sum()),
           "gt_max_speed_exceeds_ceiling_on_exempt_rows": {"n_exempt": int(ex_strict.sum()), "n_gt_exceeds": int((ex_strict & over).sum())},
           "gt_max_minus_ceiling_ms_on_exempt_rows": q3((g - l)[ex_strict & ok])}
    return out


def clamp_checks(V, plan, lim, v0, rc, seed_tag="") -> dict:
    """Function-level controls on the banked plans: K1(a) inert limit bit-identity, P postcondition + non-binding bit-identity, K3(a) binding set."""
    import torch
    out_on, d_on = apply_clamp(V, plan, lim, v0, rc)
    inf = torch.full_like(lim, float("inf"))
    out_inf, d_inf = apply_clamp(V, plan, inf, v0, rc)
    p32 = plan.astype(np.float32)
    k1a = bool(np.array_equal(out_inf.numpy(), p32)) and int(d_inf["plan_cap_binding"].sum()) == 0
    bind = d_on["plan_cap_binding"].numpy()
    nonbind_identical = bool(np.array_equal(out_on.numpy()[~bind], p32[~bind]))
    # postcondition, derived independently of the function's own output scale: interval speeds of the clamped plan vs the allowance envelope
    dts = np.asarray(V.step_durations_s(rc["horizons"], rc["tick_s"]), np.float64)
    t_mid = np.cumsum(dts) - 0.5 * dts
    seq = np.concatenate([np.zeros_like(out_on.numpy()[:, :1], dtype=np.float64), out_on.numpy().astype(np.float64)], 1)
    spd = np.linalg.norm(seq[:, 1:] - seq[:, :-1], axis=-1) / dts[None, :]
    lm = lim.numpy().astype(np.float64)
    allow = np.maximum(lm[:, None], v0.astype(np.float64)[:, None] - rc["brake_ms2"] * t_mid[None, :])
    allow = np.where(np.isfinite(allow), allow, np.inf)
    viol = np.where(np.isfinite(allow), spd - allow, -np.inf).max()
    # K3(a): limits rolled by one episode block
    lim_roll = torch.from_numpy(np.roll(lim.numpy(), -MIS_ROLL))
    _, d_roll = apply_clamp(V, plan, lim_roll, v0, rc)
    brk = d_roll["plan_cap_binding"].numpy()
    # strict-brake (0) sensitivity: binding rate only
    _, d_strict = apply_clamp(V, plan, lim, v0, rc, brake=0.0)
    return {"K1a_inert_limit_bit_identical": k1a, "P_nonbinding_rows_bit_identical": nonbind_identical,
            "P_postcondition_max_violation_ms": round(float(viol), 7), "P_postcondition_pass": bool(viol <= POST_TOL),
            "K3a_binding_set_rows_changed_by_roll35": int((brk != bind).sum()), "K3a_pass": bool((brk != bind).sum() > 0),
            "n_binding": int(bind.sum()), "n_binding_rolled": int(brk.sum()),
            "sensitivity_brake0_binding_rows": int(d_strict["plan_cap_binding"].sum()),
            "ALL_function_level_PASS": bool(k1a and nonbind_identical and viol <= POST_TOL and (brk != bind).sum() > 0)}, out_on.numpy()


# ------------------------------------------------------------------------------------------ the paired read (imports nr_matched_instrument's path)
def get_readers():
    import nr_matched_instrument as NR
    import r8_battery_patch as BP
    L, P = BP.apply()
    ra = P.RR.ra3().ra
    import pair_read_r8 as PR
    return NR, BP, PR, ra, L.sha12


def trim_read(r: dict) -> dict:
    """Keep what a reader needs; drop the heavy component means."""
    return {"n_windows": r["n_windows"], "n_episodes": r["n_episodes"], "windows_check": r.get("windows_check"),
            "os_means": (r.get("means") or {}).get("os"),
            "rows": {k: {w: v.get(w) for w in ("os_minus_ha0_ext", "os_minus_refcv7_50400")} for k, v in r["rows"].items()}}


def s6_row_mask(NR, root, seed, ex_key_rows) -> np.ndarray:
    pdir = f"{root}/s6_s{seed}"
    man, files, ep_of = NR.panel_layout(pdir)
    m = np.zeros(len(ex_key_rows["ws"]), bool)
    for f in files:
        ws = np.load(f)["ws"]
        for r in NR.rows_of(ex_key_rows, ep_of, f, ws):
            m[r] = True
    return m


def read_tree(tree: str, force: bool) -> int:
    out_json = OUT / "trees" / f"{tree}.json"
    if out_json.exists() and not force:
        print(f"[{tree}] already read ({out_json}); --force to redo")
        return 0
    t0 = time.time()
    V, vrec = load_vendored()
    rc = run_cfg()
    if rc["horizons"] != [5, 10, 15, 20, 30, 40, 50, 60] or rc["argv_has_refcv7"]:
        raise SystemExit(f"[clamp_posthoc] unexpected run config: {rc}")
    NR, BP, PR, ra, sha12 = get_readers()
    root = f"{ARCHIVE}/{tree}"
    banked = jload(f"{root}/PAIRED_READ_{tree}.json")
    res = {"tool": "clamp_posthoc_r8.py", "tree": tree, "tier": TIER, "estimator": QUESTION, "n_boot": N_BOOT, "bootstrap_seed": PR.SEED,
           "generated": time.strftime("%FT%T"), "vendored": {k: vrec[k] for k in ("source_commit", "git_rev_parse_source_blob",
                                                                                  "vendored_file_git_blob_sha", "vendored_file_sha256")},
           "run_cfg": {k: rc[k] for k in ("horizons", "tick_s", "brake_ms2")},
           "banked_read": {"file": f"{root}/PAIRED_READ_{tree}.json", "finished": banked.get("finished"), "clock": banked.get("clock")},
           "seeds": {}, "controls": {}}
    reads_on, reads_k1 = {}, {}
    all_k1 = True
    for seed in SEEDS:
        raw, lim, linfo, dump = load_seed_inputs(tree, seed, rc)
        plan, v0 = raw["nav_true.traj"], raw["v0"]
        ex = NR.load_extras(os.path.dirname(dump))
        eid = raw["clip_index"]
        sd = {"extras": dump, "n_windows": int(len(plan)), "ceiling": linfo,
              "v0_ms": {"min": round(float(v0.min()), 3), "max": round(float(v0.max()), 3), "n_nan": 0}}
        s6m = s6_row_mask(NR, root, seed, ex)
        gtm = raw.get("gt_max_speed_6s")
        sd["r1_gap_emitted_plan"] = r1_gap(V, plan, lim, v0, eid, rc, label=f"{tree} s{seed} emitted plan, S2 window set", gtmax=gtm)
        sd["r1_gap_emitted_plan_s6_subset"] = r1_gap(V, plan, lim, v0, eid, rc, mask=s6m, label=f"{tree} s{seed} emitted plan, S6 window subset", gtmax=gtm)
        if raw["filter_off.traj"] is not None:
            sd["r1_gap_filter_off_arm"] = r1_gap(V, raw["filter_off.traj"], lim, v0, eid, rc, label=f"{tree} s{seed} filter_off arm (ceiling filter OFF)", gtmax=gtm)
        chk, plan_on = clamp_checks(V, plan, lim, v0, rc)
        sd["function_controls"] = chk
        ex["clamped_on"] = plan_on.astype(np.float32)
        import torch
        ex["clamped_inf"] = apply_clamp(V, plan, torch.full_like(lim, float("inf")), v0, rc)[0].numpy()
        lim_roll = torch.from_numpy(np.roll(lim.numpy(), -MIS_ROLL))
        ex["clamped_roll"] = apply_clamp(V, plan, lim_roll, v0, rc)[0].numpy()
        sd["surfaces"] = {}
        for surf, pat, bdump in (("S2", "panel_s{s}", "dump_s{s}"), ("S6", "s6_s{s}", "s6_s{s}")):
            pdir = f"{root}/{pat.format(s=seed)}"
            man, files, ep_of = NR.panel_layout(pdir)
            slots = list(man["grid"]["slots"])
            bw = BP.windows_of(f"{BP.R7_BASE_ROOT}/{bdump.format(s=seed)}")
            refarm = BP.R7_BASE_LABEL.format(s=seed)
            bseed = banked["surfaces"][surf]["seeds"][str(seed)]
            wd = f"{SCRATCH}/{tree}_s{seed}_{surf}"
            own = NR.own_os(files)
            ss = {"grid_slots": slots, "panel_dir": pdir, "os_vs_extras_nav_true_slots_max_abs": NR.max_abs(own, NR.plans_from_extras(ex, ep_of, files, "nav_true.traj", slots))}
            # K1(b): the inert-limit plan through the full substitution + read path reproduces the banked rows
            NR.write_variant(pdir, wd, files, NR.plans_from_extras(ex, ep_of, files, "clamped_inf", slots))
            rK1 = PR.read_surface(ra, sha12, wd, refarm, N_BOOT, bw)
            k1 = NR.compare_to_banked(rK1, bseed)
            ss["K1b_inert_limit_reproduces_banked_rows"] = k1
            reads_k1[(seed, surf)] = rK1
            all_k1 = all_k1 and bool(k1["exact"])
            if not k1["exact"]:
                ss["status"] = "K1 FAILED: no clamp-ON number written for this seed / surface"
                sd["surfaces"][surf] = ss
                continue
            # clamp-ON read
            NR.write_variant(pdir, wd, files, NR.plans_from_extras(ex, ep_of, files, "clamped_on", slots))
            rON = PR.read_surface(ra, sha12, wd, refarm, N_BOOT, bw)
            reads_on[(seed, surf)] = rON
            # paired ON - OFF on refcv8's own metric (same windows, same bootstrap draws)
            man_o, A_o, eid_o, keys_o = PR.load_panel_keyed(wd, sha12)
            man_n, A_n, eid_n, keys_n = PR.load_panel_keyed(pdir, sha12)
            if keys_o != keys_n:
                raise SystemExit(f"[{tree}] {surf} s{seed}: ON and banked panels' window order differ")
            dt = float(man_o["grid"]["dt_s"])
            c_on, c_off = PR.components(ra, A_o["os"], A_o["g"], dt), PR.components(ra, A_n["os"], A_n["g"], dt)
            ss["paired_ON_minus_OFF_refcv8_own_metric"] = {key: PR.paired_cell(c_on[key], c_off[key], eid_o, N_BOOT) for _f, key, *_r in PR.METRICS}
            if gtm is not None:
                rows_all = np.concatenate([np.asarray(NR.rows_of(ex, ep_of, f, np.load(f)["ws"])) for f in files])
                limf = lim.numpy()
                gx = np.isfinite(limf[rows_all]) & np.isfinite(gtm[rows_all]) & (gtm[rows_all] > limf[rows_all])
                dade = np.asarray(c_on["ade_m"], np.float64) - np.asarray(c_off["ade_m"], np.float64)
                chg = np.abs(A_o["os"].astype(np.float64) - A_n["os"].astype(np.float64)).reshape(len(dade), -1).max(1) > 0
                nW = len(dade)
                ss["ade_delta_attribution"] = {
                    "exact_decomposition_of": "mean(ADE_ON - ADE_OFF) over all windows of this surface (m); changed = os differs from the banked plan",
                    "mean_delta_all_windows": round(float(dade.mean()), 6), "n_windows": int(nW), "n_changed": int(chg.sum()),
                    "n_changed_gt_exceeds_ceiling": int((chg & gx).sum()), "n_changed_gt_within_ceiling": int((chg & ~gx).sum()),
                    "mean_delta_over_changed_windows_m": (round(float(dade[chg].mean()), 4) if chg.any() else None),
                    "mean_delta_over_changed_gt_exceeds_m": (round(float(dade[chg & gx].mean()), 4) if (chg & gx).any() else None),
                    "mean_delta_over_changed_gt_within_m": (round(float(dade[chg & ~gx].mean()), 4) if (chg & ~gx).any() else None),
                    "contribution_to_mean_delta_gt_exceeds_m": round(float(dade[chg & gx].sum() / nW), 6),
                    "contribution_to_mean_delta_gt_within_m": round(float(dade[chg & ~gx].sum() / nW), 6),
                    "n_changed_improved": int((chg & (dade < 0)).sum()), "n_changed_worsened": int((chg & (dade > 0)).sum())}
            ss["n_windows_where_os_changed"] = int((np.abs(A_o["os"].astype(np.float64) - A_n["os"].astype(np.float64)).reshape(len(eid_o), -1).max(1) > 0).sum())
            ss["read_ON"] = trim_read(rON)
            ss["rows_OFF_banked"] = {k: {w: v.get(w) for w in ("os_minus_ha0_ext", "os_minus_refcv7_50400")} for k, v in bseed["rows"].items()}
            ss["os_means_OFF_banked"] = (bseed.get("means") or {}).get("os")
            # K3(b): the primary tree, seed 0 only -- the limits rolled by one block change the clamp-ON read
            if tree == PRIMARY and seed == 0:
                NR.write_variant(pdir, wd, files, NR.plans_from_extras(ex, ep_of, files, "clamped_roll", slots))
                rR = PR.read_surface(ra, sha12, wd, refarm, N_BOOT, bw)
                hd_on = {key: rON["rows"][key]["os_minus_refcv7_50400"]["delta"] for fam, key, _l, _u, _lo, sf in PR.METRICS if surf in sf}
                hd_ro = {key: rR["rows"][key]["os_minus_refcv7_50400"]["delta"] for fam, key, _l, _u, _lo, sf in PR.METRICS if surf in sf}
                ch = {k: (hd_on[k] != hd_ro[k]) for k in hd_on}
                ss["K3b_rolled_limits_read"] = {"n_headline_cells": len(ch), "n_changed": int(sum(ch.values())), "changed": ch, "PASS": bool(sum(ch.values()) >= 1)}
            sd["surfaces"][surf] = ss
            print(f"[{tree}] s{seed} {surf} done {time.time() - t0:.0f}s", flush=True)
        res["seeds"][str(seed)] = sd
    # ---- bars: clamp-ON beside the banked clamp-OFF verdicts ------------------------------------------------------
    res["controls"]["K1_all_exact"] = bool(all_k1)
    res["controls"]["K1a_all"] = all(res["seeds"][str(s)]["function_controls"]["K1a_inert_limit_bit_identical"] for s in SEEDS)
    res["controls"]["P_all"] = all(res["seeds"][str(s)]["function_controls"]["P_postcondition_pass"]
                                   and res["seeds"][str(s)]["function_controls"]["P_nonbinding_rows_bit_identical"] for s in SEEDS)
    res["controls"]["K3a_all"] = all(res["seeds"][str(s)]["function_controls"]["K3a_pass"] for s in SEEDS)
    if tree == PRIMARY:
        k3b = [res["seeds"]["0"]["surfaces"][s].get("K3b_rolled_limits_read", {}).get("PASS") for s in ("S2", "S6")]
        res["controls"]["K3b_primary"] = {"S2": k3b[0], "S6": k3b[1], "PASS": bool(all(x is True for x in k3b))}
    res["controls"]["K2_K4"] = "OWED (need the GPU): flag-off bit parity of the clamp-carrying eval tree, and post-hoc == eval-forward clamp on >= 3 episodes"
    ok_ctl = res["controls"]["K1_all_exact"] and res["controls"]["K1a_all"] and res["controls"]["P_all"] and res["controls"]["K3a_all"] \
        and (tree != PRIMARY or res["controls"]["K3b_primary"]["PASS"])
    res["controls"]["ALL_PASS"] = bool(ok_ctl)
    if ok_ctl and len(reads_on) == 4:
        res["bars"] = bars_for(PR, reads_on, banked)
    else:
        res["bars"] = {"status": "WITHHELD: a control did not pass or a read is missing"}
    res["wall_s"] = round(time.time() - t0, 1)
    jdump(res, str(out_json))
    print(f"[{tree}] WROTE {out_json} controls_ALL_PASS={ok_ctl} {res['wall_s']}s", flush=True)
    return 0 if ok_ctl else 4


def bars_for(PR, reads_on: dict, banked: dict) -> dict:
    """BAR-R8-E1 / E3 / NR exactly as pair_read_r8.build() states them, on the clamp-ON cells (floor = the clamp-ON os seed-to-seed |mean diff|)."""
    cell = lambda surf, s, key, which: reads_on[(s, surf)]["rows"][key][which]
    floors_on, floors_off = {}, banked.get("inference_floor") or {}
    for surf in ("S2", "S6"):
        a, b = reads_on[(0, surf)]["means"]["os"]["ade_m"], reads_on[(1, surf)]["means"]["os"]["ade_m"]
        floors_on[surf] = abs(a - b)
    out = {"inference_floor_ADE_ON": floors_on, "inference_floor_ADE_OFF_banked": {s: (floors_off.get(s) or {}).get("ade_m") for s in ("S2", "S6")}}
    out["BAR-R8-E1_ON"] = PR.bar_E({s: cell("S2", s, "ade_m", "os_minus_ha0_ext") for s in SEEDS}, floors_on["S2"])
    out["BAR-R8-E3_ON"] = PR.bar_E({s: cell("S6", s, "ade_m", "os_minus_ha0_ext") for s in SEEDS}, floors_on["S6"])
    nr = {}
    for fam, key, label, unit, lower, surfs in PR.METRICS:
        for surf in surfs:
            nr[f"{surf}:{key}"] = {"family": fam, **PR.bar_NR_metric({s: cell(surf, s, key, "os_minus_refcv7_50400") for s in SEEDS})}
    out["BAR-R8-NR_ON"] = {"metrics": nr, "per_family": PR.rollup_NR(nr, {k: v["family"] for k, v in nr.items()})}
    out["BAR-R8-E1_OFF_banked"] = {"verdict": banked["bars"]["BAR-R8-E1"]["verdict"], "per_inference_seed": banked["bars"]["BAR-R8-E1"]["per_inference_seed"]}
    out["BAR-R8-E3_OFF_banked"] = {"verdict": banked["bars"]["BAR-R8-E3"]["verdict"], "per_inference_seed": banked["bars"]["BAR-R8-E3"]["per_inference_seed"]}
    out["BAR-R8-NR_OFF_banked"] = {"per_family": banked["bars"]["BAR-R8-NR"]["per_family"],
                                   "metrics": {k: {"verdict": v["verdict"], "per_inference_seed": v["per_inference_seed"]}
                                               for k, v in banked["bars"]["BAR-R8-NR"]["metrics"].items()}}
    out["per_family_changed_ON_vs_OFF"] = {f: (out["BAR-R8-NR_ON"]["per_family"].get(f), out["BAR-R8-NR_OFF_banked"]["per_family"].get(f))
                                           for f in sorted(set(out["BAR-R8-NR_ON"]["per_family"]) | set(out["BAR-R8-NR_OFF_banked"]["per_family"]))}
    out["metric_verdicts_changed"] = {k: (v["verdict"], out["BAR-R8-NR_OFF_banked"]["metrics"][k]["verdict"]) for k, v in nr.items()
                                      if v["verdict"].split(" ")[0] != out["BAR-R8-NR_OFF_banked"]["metrics"][k]["verdict"].split(" ")[0]}
    out["note"] = ("the NR REFERENCE side (refcv7-50,400, ceiling filter inert, no clamp) is NOT decided by the amendment (D-R8-NR-INSTRUMENT); the OFF verdicts are "
                   "the banked ones and cannot be overridden by the ON ones; E1/E3 'NOT PROVEN' uses the clamp-ON inference floor shown")
    return out


# ------------------------------------------------------------------------------------------ task 2: exactness
def exactness() -> int:
    V, vrec = load_vendored()
    rc = run_cfg()
    import nr_matched_instrument as NR
    deps = dependency_record()
    rv3_fb = Path(f"{LGS}/stack/tanitad/refs/refc_v3.py").read_text(encoding="utf-8")
    rv3_de = git_show("de4e15b", "stack/tanitad/refs/refc_v3.py").decode("utf-8")
    arm = Path(f"{LGS}/taniteval/tools/refcv3_arm.py").read_text(encoding="utf-8")
    rrp = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/r7_roll.py"
    r7roll = Path(f"{LGS}/{rrp}").read_text(encoding="utf-8")
    rc_py = Path(f"{LGS}/stack/tanitad/refs/refc.py").read_text(encoding="utf-8")
    ms = Path(f"{LGS}/stack/tanitad/refs/refcv6_max_speed.py").read_text(encoding="utf-8")
    man = jload(f"{ARCHIVE}/{PRIMARY}/panel_s0/manifest.json")
    q = {
        "de4e15b refc_v3.py (clamp call site)": {
            "E9 pick: traj = fan[arange, idx]": find_line(rv3_de, "traj = fan[torch.arange(b, device=fan.device), idx]"),
            "clamp gate (eval forward only)": find_line(rv3_de, 'if bool(getattr(self.cfg, "plan_vmax_clamp", False)) and not self.training:'),
            "clamp call": find_line(rv3_de, "traj, _cd = v6sel.cap_plan_to_limit("),
            "emitted plan assigned": find_line(rv3_de, 'out["traj"], out["sel_idx"] = traj, idx'),
            "limit exported by the scene hook": find_line(rv3_de, 'cache["plan_v_lim_ms"] = v_lim.detach()'),
            "limit = bin limit": find_line(rv3_de, "v_lim = v6ms.limit_ms_of_bin(vmax_1h.argmax(dim=-1)).to(dev)"),
            "unknown row -> +inf": find_line(rv3_de, "torch.full_like(v_lim, float(\"inf\")))"),
            "refcv8 label rule": find_line(rv3_de, 'if str(getattr(self.cfg.refcv8, "speed_input", "") or ""):'),
            "refcv7 scorer pick would replace out[traj] AFTER the clamp (call)": find_line(rv3_de, "self._refcv7_forward(out, v0, nav_cmd)"),
            "refcv7 heads only built under cfg.refcv7": find_line(rv3_de, 'if bool(getattr(cfg, "refcv7", False)):'),
            "refcv7_select replaces out[traj]": find_line(rv3_de, 'out["traj"] = pick'),
        },
        "fb47a58 refc_v3.py (the tree the batteries were rolled with; clamp absent)": {
            "E9 pick: traj = fan[arange, idx]": find_line(rv3_fb, "traj = fan[torch.arange(b, device=fan.device), idx]"),
            "emitted plan assigned": find_line(rv3_fb, 'out["traj"], out["sel_idx"] = traj, idx'),
            "refcv7 scorer pick call (only if refcv7_wta is built)": find_line(rv3_fb, "self._refcv7_forward(out, v0, nav_cmd)"),
            "refcv7 heads only built under cfg.refcv7": find_line(rv3_fb, 'if bool(getattr(cfg, "refcv7", False)):'),
            "refcv7_select replaces out[traj]": find_line(rv3_fb, 'out["traj"] = pick'),
        },
        "taniteval/tools/refcv3_arm.py (fb47a58)": {
            "window loop (one forward per logged window)": find_line(arm, "for i, (wi, e_i, t) in enumerate(sel):"),
            "v0 = measured scalar at t0": find_line(arm, "v0 = float(pose_last[3])"),
            "v0 broadcast to the batch": find_line(arm, "v0_t = torch.full((b,), v0, dtype=torch.float32, device=dev)"),
            "the model call (frames, nav_cmd, v0, steps, ego_state)": find_line(arm, "out = model(fr_b, nav_cmd=nav_t if nav_on else None,"),
            "os = out[traj] row 0 (nav_true)": find_line(arm, 'traj = out["traj"].float()'),
            "os slot selection": find_line(arm, 'acc["os"].append(traj[0:1, slots].cpu().numpy())'),
            "plan_full_nav_true = same tensor, every slot": find_line(arm, 'dec.setdefault(f"plan_full_{cname}", []).append('),
        },
        "r7_roll.py Refcv6Proxy (extras sidecar)": {
            "per-row extras recorded from out[...]": find_line(r7roll, "rec[c] = {k: out[k][r].detach().float().cpu().numpy() for k in EXTRA_ROW_KEYS"),
            "extras keys incl. traj": find_line(r7roll, 'EXTRA_ROW_KEYS = ("tacv6_goal_logits"'),
            "v_max_ms / v_max_valid / v0 per window": find_line(r7roll, '"v_max_ms": float(it["v_max_ms"]), "v_max_valid": float(it["v_max_valid"]),'),
            "v0 per window": find_line(r7roll, '"v0": float(v0.reshape(-1)[0]) if v0 is not None else None}'),
        },
        "refc.py (decoder attributes the clamp reads)": {
            "anchor_dt": find_line(rc_py, "self.anchor_dt = 0.1"),
            "anchor_horizons": find_line(rc_py, "self.anchor_horizons: tuple[int, ...] = tuple(int(k) for k in _h)"),
            "sel.accel_max from sel_accel_max": find_line(rc_py, "accel_max=self.sel_accel_max,"),
        },
        "refcv6_max_speed.py": {"limit_ms_of_bin": find_line(ms, "def limit_ms_of_bin(bin_idx: Tensor) -> Tensor:"),
                                "MaxSpeedOneHotEncoder": find_line(ms, "class MaxSpeedOneHotEncoder(nn.Module):")},
    }
    verdicts = {
        "C1_os_is_the_tensor_the_clamp_acts_on": {
            "verdict": "EXACT (conditional on the 3 empirical identities below holding for every tree / seed)",
            "reasoning": ("the clamp (de4e15b) sits between `traj = fan[arange, idx]` and `out[\"traj\"], out[\"sel_idx\"] = traj, idx`; fb47a58 has the same two lines "
                          "with nothing between them, so the banked `out[\"traj\"]` IS the clamp's input.  The only later writer of out[\"traj\"] is the refcv7 scorer pick "
                          "(`_refcv7_forward`, gated on `self.refcv7_wta is not None`), which exists only under `cfg.refcv7`; the run argv carries no --refcv7 flag "
                          f"(argv_has_refcv7 = {rc['argv_has_refcv7']}), and de4e15b refuses plan_vmax_clamp together with refcv7_select."),
            "fp32": "the extras' traj is not bf16- or fp16-representable (checked per tree), i.e. fp32 resident, as the clamp's input would be"},
        "C2_S2_and_S6_are_slots_of_one_plan": {
            "verdict": "EXACT (S2 slots [0,1,2,3] of the full plan; S6 slots [1,3,4,5,6,7]; S6 windows are a subset of S2's)",
            "evidence": {"S2_grid": {"slots": man["grid"]["slots"], "instants_s": man["grid"]["instants_s"]},
                         "S6_grid": jload(f"{ARCHIVE}/{PRIMARY}/s6_s0/manifest.json")["grid"]["slots"]},
            "caveat": ("the clamp acts on the FULL 8-slot plan; binding is decided over all 8 intervals and a binding row is re-integrated with cumsum, so even a "
                       "row binding only in 3-6 s changes slots 0-3 by fp32 rounding.  This read therefore clamps the full plan and slices, never the slice.")},
        "C3_no_emitted_plan_feeds_a_later_model_input": {
            "verdict": "EXACT for all 18 banked dump / panel sets (9 trees x 2 inference seeds): each is a single forward per logged window",
            "reasoning": ("refcv3_arm.py runs ONE forward per logged window from logged frames, the clip's nav token and the measured scalar v0 = pose_last[3] "
                          "(manifest t1_definition: 'consuming the observed frames, the clip's v7.2 nav token and the MEASURED v0 at t0 and nothing else'; "
                          "`no_closed_loop_because`: forward has NO action argument, no per-step decode, no state that advances).  `ego_state_fed` is false in the "
                          "manifest; `ego_actions` is a forward kwarg that refcv3_arm never passes; the proxy passes `ego_poses` = the OBSERVED pose history of the "
                          "window.  The sidecar only RECORDS out[...] rows.  The next window's inputs come from the dataset index over logged episodes."),
            "caveat": ("SeamState (`self._seam`) accumulates across forwards but is fed by the pre-pick blended-score norms, not by the emitted plan; it does not "
                       "read `out[\"traj\"]`."), "t1_definition": man["t1_definition"]["arm"] + ": " + man["t1_definition"]["_is"][:260]},
        "C4_numerics": {"verdict": "post-hoc is exact in DEFINITION; BIT-exactness against the eval-forward clamp is UNVERIFIED (K4 owed: CPU fp32 vs GPU kernels may differ in the last ulp)"},
    }
    emp = {}
    for tree in TREES:
        emp[tree] = {}
        for seed in SEEDS:
            raw, lim, linfo, dump = load_seed_inputs(tree, seed, rc)
            ex = NR.load_extras(os.path.dirname(dump))
            plan = raw["nav_true.traj"]
            import torch
            t = torch.from_numpy(plan)
            e = {"n_windows": int(len(plan)), "traj_bf16_exact": bool((t.to(torch.bfloat16).float() == t).all()),
                 "traj_fp16_exact": bool((t.half().float() == t).all()), "ceiling_values_ms": linfo["limit_values_ms"],
                 "n_unknown_ceiling": linfo["n_unknown_inf"], "n_over_ceiling_flag": linfo["n_over_ceiling"],
                 "bins_equal_fp32_vs_bf16": linfo["bins_equal_fp32_vs_bf16"]}
            for surf, pat in (("S2", "panel_s{s}"), ("S6", "s6_s{s}")):
                pdir = f"{ARCHIVE}/{tree}/{pat.format(s=seed)}"
                m, files, ep_of = NR.panel_layout(pdir)
                slots = list(m["grid"]["slots"])
                own = NR.own_os(files)
                on = NR.plans_from_extras(ex, ep_of, files, "nav_true.traj", slots)
                dfull, v0d, n_pf = 0.0, 0.0, 0
                for f in files:
                    z = np.load(f)
                    rows = NR.rows_of(ex, ep_of, f, z["ws"])
                    v0d = max(v0d, float(np.abs(z["v0"].astype(np.float64) - raw["v0"][rows].astype(np.float64)).max()))
                    dz = np.load(os.path.join(pdir, "decisions", os.path.basename(f)))
                    if "plan_full_nav_true" in dz.files:
                        n_pf += 1
                        dfull = max(dfull, float(np.abs(dz["plan_full_nav_true"].astype(np.float64) - ex["nav_true.traj"][rows].astype(np.float64)).max()))
                e[surf] = {"slots": slots, "n_windows": int(sum(len(a) for a in own)),
                           "panel_os_vs_extras_nav_true_slots_max_abs": NR.max_abs(own, on),
                           "decisions_plan_full_nav_true_vs_extras_max_abs": (dfull if n_pf == len(files) else f"ABSENT in {len(files) - n_pf} of {len(files)} episode files"),
                           "panel_v0_vs_extras_v0_max_abs": v0d}
            rows2, rows6 = set(), set()
            for surf, pat, acc in (("S2", "panel_s{s}", rows2), ("S6", "s6_s{s}", rows6)):
                pdir = f"{ARCHIVE}/{tree}/{pat.format(s=seed)}"
                m, files, ep_of = NR.panel_layout(pdir)
                for f in files:
                    acc.update(NR.rows_of(ex, ep_of, f, np.load(f)["ws"]))
            e["S2_panel_windows_equal_extras_rows"] = bool(len(rows2) == len(ex["ws"]) == len(plan))
            e["S6_windows_subset_of_S2"] = bool(rows6 <= rows2)
            e["n_S2_rows"], e["n_S6_rows"] = len(rows2), len(rows6)
            emp[tree][str(seed)] = e
            print(f"[exactness] {tree} s{seed} ok", flush=True)
    ident_ok = all(v["S2"]["panel_os_vs_extras_nav_true_slots_max_abs"] == 0.0 and v["S6"]["panel_os_vs_extras_nav_true_slots_max_abs"] == 0.0
                   and v["S2"]["panel_v0_vs_extras_v0_max_abs"] == 0.0 and v["S6"]["panel_v0_vs_extras_v0_max_abs"] == 0.0
                   and not v["traj_bf16_exact"] and not v["traj_fp16_exact"]
                   and v["S2_panel_windows_equal_extras_rows"] and v["S6_windows_subset_of_S2"]
                   for d in emp.values() for v in d.values())
    pf_ok = all(v[s]["decisions_plan_full_nav_true_vs_extras_max_abs"] == 0.0 for d in emp.values() for v in d.values() for s in ("S2", "S6"))
    res = {"tool": "clamp_posthoc_r8.py exactness", "generated": time.strftime("%FT%T"), "vendored": vrec, "dependency_blobs": deps, "run_cfg": rc,
           "source_quotes": q, "verdicts": verdicts, "empirical": emp,
           "empirical_summary": {"panel_os_equals_extras_slots_and_v0_equal_and_fp32_in_all_18_sets": bool(ident_ok),
                                 "decisions_plan_full_nav_true_equals_extras_in_all_18_sets_x_2_surfaces": bool(pf_ok)}}
    jdump(res, str(OUT / "EXACTNESS.json"))
    print("WROTE", OUT / "EXACTNESS.json", "ident_ok", ident_ok, "plan_full_ok", pf_ok)
    return 0 if (ident_ok and pf_ok) else 5


def vendor() -> int:
    V, vrec = load_vendored()
    deps = dependency_record()
    jdump({"vendored": vrec, "dependency_blobs": deps, "generated": time.strftime("%FT%T"),
           "import_resolution": ("tanitad.* imports of the vendored module resolve against " + LGS + " (fb47a58) on PYTHONPATH; "
                                 "vocab_v7 / refc_selector_targets / refcv6_tactical / refcv6_max_speed blobs listed with the comparison")},
          str(OUT / "VENDOR.json"))
    print(json.dumps(vrec, indent=1))
    bad = {p: v for p, v in deps.items() if "identical_de4e15b_vs_fb47a58" in v and v["identical_de4e15b_vs_fb47a58"] is False and "note" not in v}
    print("dependency differences (excluding the two files that differ by the added clamp lines):", bad or "none")
    return 0 if not bad else 6


# ------------------------------------------------------------------------------------------ task 6: assemble
def _c(c, sep_kind="worse") -> str:
    """'+0.0621 [+0.0267, +0.1022]' + a marker when separated in the direction that matters (sep_kind: 'worse' = lo > 0, 'better' = hi < 0)."""
    if not c or "delta" not in c:
        return "n/a"
    star = ""
    if sep_kind == "worse" and c["lo"] > 0:
        star = " **W**"
    if sep_kind == "better" and c["hi"] < 0:
        star = " **B**"
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{star}"


def _pct(r) -> str:
    return f"{100 * r['mean']:.2f} % [{100 * r['lo']:.2f}, {100 * r['hi']:.2f}]"


def r1_refresh() -> int:
    """Recompute the r1_gap blocks of every per-tree JSON with the CURRENT r1_gap (no bootstrap read is re-run; K1 / ON cells are untouched)."""
    V, vrec = load_vendored()
    rc = run_cfg()
    NR, BP, PR, ra, sha12 = get_readers()
    for tree in TREES:
        pj = OUT / "trees" / f"{tree}.json"
        r = jload(str(pj))
        for seed in SEEDS:
            raw, lim, linfo, dump = load_seed_inputs(tree, seed, rc)
            ex = NR.load_extras(os.path.dirname(dump))
            plan, v0, eid, gtm = raw["nav_true.traj"], raw["v0"], raw["clip_index"], raw.get("gt_max_speed_6s")
            s6m = s6_row_mask(NR, f"{ARCHIVE}/{tree}", seed, ex)
            sd = r["seeds"][str(seed)]
            sd["r1_gap_emitted_plan"] = r1_gap(V, plan, lim, v0, eid, rc, label=f"{tree} s{seed} emitted plan, S2 window set", gtmax=gtm)
            sd["r1_gap_emitted_plan_s6_subset"] = r1_gap(V, plan, lim, v0, eid, rc, mask=s6m, label=f"{tree} s{seed} emitted plan, S6 window subset", gtmax=gtm)
            if raw["filter_off.traj"] is not None:
                sd["r1_gap_filter_off_arm"] = r1_gap(V, raw["filter_off.traj"], lim, v0, eid, rc, label=f"{tree} s{seed} filter_off arm (ceiling filter OFF)", gtmax=gtm)
        r["r1_gap_refreshed"] = time.strftime("%FT%T") + " (residual tolerance %s m/s)" % EXEMPT_TOL_COARSE
        jdump(r, str(pj))
        print(f"[r1_refresh] {tree} ok", flush=True)
    return 0


def assemble() -> int:
    sys.path.insert(0, str(HERE))
    import pair_read_r8 as PR
    ex = jload(str(OUT / "EXACTNESS.json"))
    ven = jload(str(OUT / "VENDOR.json"))
    trees = {}
    missing = []
    for t in TREES:
        p = OUT / "trees" / f"{t}.json"
        if not p.exists():
            missing.append(t)
        else:
            trees[t] = jload(str(p))
    if missing:
        raise SystemExit(f"[assemble] per-tree reads missing: {missing}")
    bad = [t for t, r in trees.items() if not r["controls"].get("ALL_PASS")]
    if bad:
        raise SystemExit(f"[assemble] controls did not all pass for {bad}: refusing to assemble clamp-ON numbers")
    nr_keys = [f"{surf}:{key}" for fam, key, label, unit, lower, surfs in PR.METRICS for surf in surfs]
    summ = {"tool": "clamp_posthoc_r8.py assemble", "generated": time.strftime("%FT%T"), "tier": TIER, "estimator": QUESTION,
            "evidence_class": "MEASURED (ours): banked refcv8 dumps / panels under " + ARCHIVE + ", the landed function vendored from de4e15b; CPU only",
            "vendored": ven["vendored"], "dependency_blobs": ven["dependency_blobs"], "run_cfg": trees[PRIMARY]["run_cfg"],
            "exactness": {"verdicts": ex["verdicts"], "empirical_summary": ex["empirical_summary"], "source_quotes": ex["source_quotes"]},
            "controls": {t: r["controls"] for t, r in trees.items()},
            "owed": {"K2": "flag-off parity of the clamp-carrying eval tree on >= 3 episodes (GPU)",
                     "K4": "post-hoc clamp == eval-forward clamp on the same subset (GPU)"},
            "trees": {}}
    flips_all = []
    for t, r in trees.items():
        e = {"bars": r["bars"], "r1_gap": {}, "attribution": {}}
        for sd in SEEDS:
            sdr = r["seeds"][str(sd)]
            e["r1_gap"][str(sd)] = {k: sdr.get(k) for k in ("r1_gap_emitted_plan", "r1_gap_emitted_plan_s6_subset", "r1_gap_filter_off_arm") if sdr.get(k)}
            e["attribution"][str(sd)] = {surf: sdr["surfaces"][surf].get("ade_delta_attribution") for surf in ("S2", "S6")}
        e["paired_ON_minus_OFF"] = {str(sd): {surf: r["seeds"][str(sd)]["surfaces"][surf]["paired_ON_minus_OFF_refcv8_own_metric"] for surf in ("S2", "S6")} for sd in SEEDS}
        summ["trees"][t] = e
        for k, v in r["bars"]["metric_verdicts_changed"].items():
            flips_all.append({"tree": t, "metric": k, "ON": v[0], "OFF_banked": v[1]})
    summ["metric_verdict_changes_ON_vs_OFF"] = flips_all
    summ["family_verdict_changes_ON_vs_OFF"] = {
        t: {f: v for f, v in r["bars"]["per_family_changed_ON_vs_OFF"].items() if (v[0] or "").split(":")[0] != (v[1] or "").split(":")[0]}
        for t, r in trees.items()}
    jdump(summ, str(OUT / "CLAMP_POSTHOC.json"))
    # ---------------------------------------------------------------- markdown
    L = []
    w = L.append
    r0 = trees[PRIMARY]
    w("# CLAMP_POSTHOC -- the first clamp-ON numbers on real refcv8 windows (AMENDMENT A-R8-CLAMP-1 section 2.3)")
    w("")
    w(f"Generated {summ['generated']} by `code/clamp_posthoc_r8.py` (CPU only). **TIER:** {TIER}. **ESTIMATOR / QUESTION:** {QUESTION}; B {N_BOOT}, bootstrap seed 0, 95 % percentile; "
      "paired = refcv8 minus the reference on the same windows. Rates carry the same cluster bootstrap (mean of a 0/1 indicator vs zeros). "
      "**Single training seed; inference seeds 0 and 1 are the two banked rolls.** The post-hoc clamp is deterministic given the plan, so seeds 0/1 differ only through the plans.  "
      "`**W**` = CI lower > 0 on an error metric (separated worse); `**B**` = CI upper < 0 (separated better).")
    w("")
    w("**Evidence class:** MEASURED (ours) on `C:/Users/Admin/refcv8_eval_archive/battery/<tree>/{dump_s*,panel_s*,s6_s*}`; the clamp is the landed `cap_plan_to_limit`, vendored byte-for-byte. "
      "No clamp-ON number below is a GPU eval-forward number: K2/K4 are OWED, so every ON cell is a POST-HOC read on the banked plans.")
    w("")
    w("## 0. Headline")
    w("")
    p0 = r0["seeds"]["0"]["r1_gap_emitted_plan"]
    p1 = r0["seeds"]["1"]["r1_gap_emitted_plan"]
    a0 = r0["seeds"]["0"]["surfaces"]["S2"]["ade_delta_attribution"]
    w(f"* **Controls:** K1 (inert limit: function bit-identity AND full-pipeline reproduction of every banked cell), K3 (mispairing) and the analytic postcondition PASS on all {len(trees)} trees x 2 seeds; "
      "K2 / K4 are GPU-only and OWED.")
    w(f"* **R1 gap at step 30,000 EMA (S2 window set, n = {p0['n_windows']:,} windows / {p0['n_episodes']} episodes):** the emitted clamp-OFF plan exceeds the fed ceiling on **{_pct(p0['exempt_rate_strict_full_plan_0_6s'])}** "
      f"of windows (seed 0; seed 1 {_pct(p1['exempt_rate_strict_full_plan_0_6s'])}), median excess {p0['excess_ms_on_exempt_rows']['median']} m/s, p90 {p0['excess_ms_on_exempt_rows']['p90']}, max {p0['excess_ms_on_exempt_rows']['max']}; "
      f"**{p0['exempt_and_v0_over']} of {p0['n_exempt_strict']}** exempt rows are rows where v0 is already above the ceiling.")
    w(f"* **On every one of the {p0['n_exempt_strict']} exempt rows the logged ego's OWN 6-s max speed exceeds the fed ceiling too** "
      f"({p0['gt_max_speed_exceeds_ceiling_on_exempt_rows']['n_gt_exceeds']}/{p0['gt_max_speed_exceeds_ceiling_on_exempt_rows']['n_exempt']}); across all windows the logged ego exceeds the fed ceiling on "
      f"{_pct(p0['gt_max_speed_exceeds_ceiling_rate_all_windows'])}. At this tree the clamp binds ONLY where the logged-ego ground truth itself violates the label (all-tree counts in the next bullets).")
    w(f"* **Clamp-ON S2 ADE moves by {a0['mean_delta_all_windows']:+.4f} m** (seed 0; {a0['n_changed']} of {a0['n_windows']} windows changed, {a0['mean_delta_over_changed_windows_m']:+.2f} m each on average; "
      f"{a0['n_changed_worsened']} worsened / {a0['n_changed_improved']} improved), 100 % of it on windows where the ground truth exceeds the ceiling "
      f"(contribution on ceiling-respecting windows: {a0['contribution_to_mean_delta_gt_within_m']:+.4f} m).")
    b0 = r0["bars"]
    w(f"* **Bars at step 30,000 EMA, clamp-ON beside banked clamp-OFF:** BAR-R8-E1 {b0['BAR-R8-E1_ON']['verdict'].split(' (')[0]} (OFF {b0['BAR-R8-E1_OFF_banked']['verdict'].split(' (')[0]}); "
      f"BAR-R8-E3 {b0['BAR-R8-E3_ON']['verdict'].split(' (')[0]} (OFF {b0['BAR-R8-E3_OFF_banked']['verdict'].split(' (')[0]}); NR per family ON {json.dumps(b0['BAR-R8-NR_ON']['per_family'])} vs OFF {json.dumps(b0['BAR-R8-NR_OFF_banked']['per_family'])}.")
    w(f"* **Metric-level NR verdict flips ON vs OFF across the {len(trees)} trees:** {len(flips_all)} ({'; '.join(sorted({f['tree'] + ' ' + f['metric'] for f in flips_all})) or 'none'}).")
    resid = p0["residual_above_ceiling_after_clamp_rate"]
    w(f"* **The landed clamp does not make the ceiling hard where v0 is above it:** after the clamp {_pct(resid)} of windows ({p0['n_residual']}) are still above the ceiling "
      f"(median {p0['residual_excess_ms_after_clamp']['median']} m/s, max {p0['residual_excess_ms_after_clamp']['max']}), because the allowance is max(limit, v0 - brake * t_mid) with brake = {summ['run_cfg']['brake_ms2']} m/s^2.")
    # ---- across-tree bullets (computed, not typed)
    ex_rates = [trees[t]["seeds"]["0"]["r1_gap_emitted_plan"]["exempt_rate_strict_full_plan_0_6s"]["mean"] for t in trees]
    fo_rates = [trees[t]["seeds"]["0"]["r1_gap_filter_off_arm"]["exempt_rate_strict_full_plan_0_6s"]["mean"] for t in trees]
    tot = {(sd, sf): (sum(trees[t]["seeds"][str(sd)]["surfaces"][sf]["ade_delta_attribution"]["n_changed"] for t in trees),
                      sum(trees[t]["seeds"][str(sd)]["surfaces"][sf]["ade_delta_attribution"]["n_changed_gt_within_ceiling"] for t in trees))
           for sd in SEEDS for sf in ("S2", "S6")}
    w(f"* **Across the {len(trees)} trees (seed 0):** exempt rate {100 * min(ex_rates):.2f}-{100 * max(ex_rates):.2f} % with the ceiling filter acting; {100 * min(fo_rates):.1f}-{100 * max(fo_rates):.1f} % on the filter_off arm. "
      "Windows changed by the clamp (summed over trees) and, of those, windows where the logged ego stays WITHIN the ceiling: "
      + "; ".join(f"s{sd} {sf} {tot[(sd, sf)][1]} of {tot[(sd, sf)][0]}" for sd in SEEDS for sf in ("S2", "S6")) + ".")
    e1c = {sd: (r0["seeds"][str(sd)]["surfaces"]["S2"]["rows_OFF_banked"]["ade_m"]["os_minus_ha0_ext"], r0["seeds"][str(sd)]["surfaces"]["S2"]["read_ON"]["rows"]["ade_m"]["os_minus_ha0_ext"]) for sd in SEEDS}
    w("* **BAR-R8-E1 cell (os - ha0_ext, S2 ADE 0-2 s) at step 30,000 EMA, OFF -> ON:** " + "; ".join(f"seed {sd}: {_c(a, 'better')} -> {_c(b, 'better')}" for sd, (a, b) in e1c.items())
      + ". The bar needs CI upper < 0; it FAILED OFF (not separated) and FAILS ON (separated WORSE than the echo).")
    n_pf = sum(1 for f in flips_all if f["OFF_banked"].startswith("PASS") and f["ON"].startswith("FAIL"))
    n_fp = sum(1 for f in flips_all if f["OFF_banked"].startswith("FAIL") and f["ON"].startswith("PASS"))
    e13 = {t: (trees[t]["bars"]["BAR-R8-E1_OFF_banked"]["verdict"].split(" (")[0] + "->" + trees[t]["bars"]["BAR-R8-E1_ON"]["verdict"].split(" (")[0],
              trees[t]["bars"]["BAR-R8-E3_OFF_banked"]["verdict"].split(" (")[0] + "->" + trees[t]["bars"]["BAR-R8-E3_ON"]["verdict"].split(" (")[0]) for t in trees}
    w(f"* **Verdict changes ON vs OFF over all {len(trees)} trees:** BAR-R8-E1 {sorted({v[0] for v in e13.values()})}, BAR-R8-E3 {sorted({v[1] for v in e13.values()})} (no tree changes); "
      f"NR metric cells {n_pf} PASS->FAIL, {n_fp} FAIL->PASS, all on EMA trees ({sorted({f['tree'] for f in flips_all})}).")
    w("")
    w("## 1. Vendoring (task 1)")
    v = ven["vendored"]
    w("")
    w(f"* Source: commit `de4e15b`, `{SRC_PATH}`; `git rev-parse de4e15b:{SRC_PATH}` = `{v['git_rev_parse_source_blob']}`; body re-hashed = `{v['body_blob_sha_recomputed']}` (equal), body == `git show` bytes.")
    w(f"* Vendored file `code/_vendored_de4e15b_refcv6_selection.py`: 5-line header + the blob; file git-blob sha `{v['vendored_file_git_blob_sha']}`, sha256 `{v['vendored_file_sha256']}`, {v['vendored_file_bytes']} bytes. Loaded by `importlib` (`spec_from_file_location`); `clamp_posthoc_r8.py` refuses if the body hash drifts.")
    w("* Dependencies resolved against the fb47a58 tree (`C:/lgs/Lfb47a58376`, `PYTHONPATH`). Blob comparison de4e15b vs fb47a58:")
    w("")
    w("| file | de4e15b | fb47a58 | identical |")
    w("|---|---|---|---|")
    for pth, d in ven["dependency_blobs"].items():
        w(f"| `{pth}` | `{d['de4e15b'][:12]}` | `{d['fb47a58'][:12]}` | {'yes' if d['identical_de4e15b_vs_fb47a58'] else 'NO -- ' + d.get('note', '')} |")
    w("")
    w("No dependency differs. The two files that differ (`refcv6_selection.py`, `refc_v3.py`) differ ONLY by lines added for the clamp (0 lines removed), so every other line of the fb47a58 forward the batteries ran is the line the clamp would sit inside.")
    w("")
    w("## 2. Exactness from source (task 2) -- file:line")
    w("")
    for cid, vd in ex["verdicts"].items():
        w(f"### {cid}")
        w(f"* **{vd['verdict']}**")
        for k in ("reasoning", "caveat", "fp32", "t1_definition"):
            if k in vd:
                w(f"* {k}: {vd[k]}")
        if "evidence" in vd:
            w(f"* evidence: {json.dumps(vd['evidence'])}")
        w("")
    w("Source quotes (line numbers re-found by the script, not typed):")
    w("")
    w("| file | what | line | text |")
    w("|---|---|---|---|")
    for f, d in ex["source_quotes"].items():
        for k, q in d.items():
            w(f"| {f} | {k} | {q['line']} | `{(q['text'] or 'ABSENT').replace('|', '/')[:110]}` |")
    w("")
    w(f"Empirical identity checks over {len(ex['empirical'])} trees x 2 inference seeds ({ex['empirical_summary']}):")
    w("")
    w("| tree | seed | S2 n | S2 os vs extras slots (max abs) | S6 n | S6 os vs extras slots (max abs) | decisions plan_full vs extras (S2 / S6) | S6 subset of S2 | traj fp32-resident |")
    w("|---|---|---|---|---|---|---|---|---|")
    for t, d in ex["empirical"].items():
        for sd, e in d.items():
            w(f"| {t} | {sd} | {e['S2']['n_windows']} | {e['S2']['panel_os_vs_extras_nav_true_slots_max_abs']} | {e['S6']['n_windows']} | {e['S6']['panel_os_vs_extras_nav_true_slots_max_abs']} | "
              f"{e['S2']['decisions_plan_full_nav_true_vs_extras_max_abs']} / {e['S6']['decisions_plan_full_nav_true_vs_extras_max_abs']} | {e['S6_windows_subset_of_S2']} | {not e['traj_bf16_exact'] and not e['traj_fp16_exact']} |")
    w("")
    w("**Surface verdicts (amendment 2.3):** S2 = EXACT, S6 = EXACT, for all 9 trees and both seeds. No surface is NOT EXACT, so none is withheld.  Residual (NOT a surface defect): bit-exactness of a CPU fp32 post-hoc clamp against the GPU eval-forward clamp is unverified until K4.")
    w("")
    w("## 3. Inputs (task 3)")
    w("")
    rcf = summ["run_cfg"]
    c0 = r0["seeds"]["0"]
    w("* **Plan:** `dump_s<seed>/refcv6_extras.npz :: nav_true.traj` [N, 8, 2], full 8-slot 0.5-6 s plan, fp32, ego frame (== `plan_full_nav_true`, max abs 0.0). The clamp acts on all 8 slots; S2 / S6 are slices of the CLAMPED plan.")
    w(f"* **Fed ceiling per window (m/s):** derived with the model's own functions (`MaxSpeedOneHotEncoder` -> `limit_ms_of_bin`, all-zero row -> +inf, refcv8 label rule), from the dump's `v_max_ms` + `v_max_valid` (`r7_roll.py:213`), "
      f"the road-type step itself: {c0['ceiling']['limit_values_ms']} m/s (30 / 50 / 100 / 120 km/h); unknown (+inf) rows: {c0['ceiling']['n_unknown_inf']}; `over_ceiling` rows: {c0['ceiling']['n_over_ceiling']}; fp32 vs bf16 bins equal: {c0['ceiling']['bins_equal_fp32_vs_bf16']}. "
      "Units from source: `v_max_ms` in m/s (`SPEED_MAX_STEPS_MS_V6`), `limit_ms_of_bin` returns m/s; +inf = no label (X15).")
    w(f"* **v0 (m/s):** the dump's per-window `v0` (`r7_roll.py:215`; = `float(pose_last[3])`, `refcv3_arm.py:2225`); equals the panel's `v0` bit-for-bit (table above). Range {c0['v0_ms']['min']} .. {c0['v0_ms']['max']} m/s.")
    w(f"* **horizons** {rcf['horizons']} (ticks) from `config.json['horizons']`; **anchor_dt** {rcf['tick_s']} (`refc.py:1748`); **brake / sel.accel_max** {rcf['brake_ms2']} m/s^2 (`config.json['selection']['sel_accel_max']`, `refc.py:1067`; the run argv carries no `--sel-accel-max`). "
      "The model object itself was not rebuilt on CPU, so `decoder.sel.accel_max` is read from the run's recorded config, not from a live attribute (stated, not hidden).")
    w("")
    w("## 4. The R1 gap per tree / seed (task 4) -- clamp-OFF emitted plans")
    w("")
    w("Exempt = planned max speed of the emitted plan on the waypoint grid (module `planned_max_speed`) > the fed ceiling, strict. S2 window set = 4,754 windows / 139 episodes. "
      "Excess in m/s on exempt rows. 'v0>c' = rows whose v0 is above the ceiling. 'GT>c on exempt' = exempt rows where the logged ego's own 6-s max speed exceeds the ceiling.")
    w("")
    w("| tree | seed | exempt rate [cluster CI] (n) | excess med / p90 / max | v0>c | exempt AND v0>c | GT>c on exempt | binding | residual after clamp (n) med / max | filter_off exempt |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for t, r in trees.items():
        for sd in SEEDS:
            g = r["seeds"][str(sd)]["r1_gap_emitted_plan"]
            fo = r["seeds"][str(sd)].get("r1_gap_filter_off_arm")
            fos = f"{_pct(fo['exempt_rate_strict_full_plan_0_6s'])} ({fo['n_exempt_strict']})" if fo else "n/a (os-only roll)"
            w(f"| {t} | {sd} | {_pct(g['exempt_rate_strict_full_plan_0_6s'])} ({g['n_exempt_strict']}) | {g['excess_ms_on_exempt_rows']['median']} / {g['excess_ms_on_exempt_rows']['p90']} / {g['excess_ms_on_exempt_rows']['max']} | "
              f"{100 * g['v0_over_ceiling_rate']['mean']:.2f} % | {g['exempt_and_v0_over']} of {g['n_exempt_strict']} | {g['gt_max_speed_exceeds_ceiling_on_exempt_rows']['n_gt_exceeds']} of {g['gt_max_speed_exceeds_ceiling_on_exempt_rows']['n_exempt']} | "
              f"{_pct(g['clamp_binding_rate'])} ({g['n_binding']}) | {100 * g['residual_above_ceiling_after_clamp_rate']['mean']:.2f} % ({g['n_residual']}) {g['residual_excess_ms_after_clamp']['median']} / {g['residual_excess_ms_after_clamp']['max']} | {fos} |")
    w("")
    w("Same rates on the S6 window subset (3,668 windows) and on the 0-2 s slots only are in `CLAMP_POSTHOC.json` and the per-tree files (`r1_gap_emitted_plan_s6_subset`, `exempt_rate_strict_s2_slots_0_2s`).")
    w("")
    w("## 5. Controls (task 5)")
    w("")
    w("| tree | K1a function inert-limit bit-identical (s0 / s1) | K1b pipeline reproduces banked cells (cells exact / compared: S2 s0, S6 s0, S2 s1, S6 s1) | P postcondition max violation (m/s) s0 / s1 | K3a rows whose binding changes under the 35-window roll s0 / s1 | K3b (primary, s0) |")
    w("|---|---|---|---|---|---|")
    for t, r in trees.items():
        k1b = [r["seeds"][str(sd)]["surfaces"][sf]["K1b_inert_limit_reproduces_banked_rows"] for sd in SEEDS for sf in ("S2", "S6")]
        fc0, fc1 = r["seeds"]["0"]["function_controls"], r["seeds"]["1"]["function_controls"]
        k3b = r["controls"].get("K3b_primary")
        w(f"| {t} | {fc0['K1a_inert_limit_bit_identical']} / {fc1['K1a_inert_limit_bit_identical']} | "
          + ", ".join(f"{x['n_cells_compared'] - x['n_mismatched_fields']}/{x['n_cells_compared']}" + ("" if x["exact"] else " FAIL") for x in k1b)
          + f" (max |mean diff| {max(x['max_abs_mean_diff'] for x in k1b):.1e}) | {fc0['P_postcondition_max_violation_ms']} / {fc1['P_postcondition_max_violation_ms']} | "
          + f"{fc0['K3a_binding_set_rows_changed_by_roll35']} / {fc1['K3a_binding_set_rows_changed_by_roll35']} | "
          + (f"S2 {k3b['S2']}, S6 {k3b['S6']}" if k3b else "-") + " |")
    w("")
    w("K1b compares, per surface, the cells of 14 metrics x {os - ha0_ext, os - refcv7-50,400} (delta, lo, hi, n_windows, n_episodes, separated, n_dropped_nonfinite) plus the window means.")
    ks = r0["seeds"]["0"]["surfaces"]
    w("")
    w(f"K3b (primary, seed 0): rolling the limits by {MIS_ROLL} windows changed {ks['S2']['K3b_rolled_limits_read']['n_changed']} of {ks['S2']['K3b_rolled_limits_read']['n_headline_cells']} S2 NR headline deltas and "
      f"{ks['S6']['K3b_rolled_limits_read']['n_changed']} of {ks['S6']['K3b_rolled_limits_read']['n_headline_cells']} S6 ones (pre-registered pass: >= 1 changed).")
    w("")
    w("**Owed (GPU, not run here):** K2 flag-off bit parity of the clamp-carrying eval tree on >= 3 episodes; K4 post-hoc clamp == eval-forward clamp on the same subset. No clamp-ON number above is admissible as a decision number until K4.")
    w("")
    w("## 6. Clamp-ON beside clamp-OFF -- the bars")
    w("")
    w("OFF = the banked `PAIRED_READ_<tree>.json` verdict (the K1 re-read reproduces it exactly). ON floors = the clamp-ON |mean(os, s0) - mean(os, s1)|. The NR reference side (refcv7-50,400, ceiling filter inert, no clamp) is NOT decided here (`D-R8-NR-INSTRUMENT`).")
    w("")
    w("| tree | E1 OFF | E1 ON | E3 OFF | E3 ON | NR ADE OFF -> ON | NR LONG OFF -> ON | NR LAT OFF -> ON |")
    w("|---|---|---|---|---|---|---|---|")
    for t, r in trees.items():
        b = r["bars"]
        pf = b["per_family_changed_ON_vs_OFF"]
        w(f"| {t} | {b['BAR-R8-E1_OFF_banked']['verdict'].split(' (')[0]} | {b['BAR-R8-E1_ON']['verdict'].split(' (')[0]} | {b['BAR-R8-E3_OFF_banked']['verdict'].split(' (')[0]} | {b['BAR-R8-E3_ON']['verdict'].split(' (')[0]} | "
          + " | ".join(f"{(pf.get(f, [None, None])[1] or 'n/a')} -> {(pf.get(f, [None, None])[0] or 'n/a')}" for f in ("ADE", "LONGITUDINAL", "LATERAL")) + " |")
    w("")
    w("E1 / E3 cells (os - ha0_ext ADE, error metric; the bar needs CI upper < 0 at both seeds) and the paired ON - OFF on refcv8's own ADE:")
    w("")
    w("| tree | surface | seed | OFF banked | ON | paired ON - OFF |")
    w("|---|---|---|---|---|---|")
    for t, r in trees.items():
        for surf in ("S2", "S6"):
            for sd in SEEDS:
                sr = r["seeds"][str(sd)]["surfaces"][surf]
                w(f"| {t} | {surf} | {sd} | {_c(sr['rows_OFF_banked']['ade_m']['os_minus_ha0_ext'], 'better')} | {_c(sr['read_ON']['rows']['ade_m']['os_minus_ha0_ext'], 'better')} | {_c(sr['paired_ON_minus_OFF_refcv8_own_metric']['ade_m'], 'worse')} |")
    w("")
    w("NR headline cells (refcv8 - refcv7-50,400, error metric; FAIL iff CI lower > 0 at either seed):")
    w("")
    w("| tree | headline | OFF s0 | ON s0 | OFF s1 | ON s1 | verdict OFF -> ON |")
    w("|---|---|---|---|---|---|---|")
    for t, r in trees.items():
        for kk in nr_keys:
            surf, key = kk.split(":")
            cells = []
            for sd in SEEDS:
                sr = r["seeds"][str(sd)]["surfaces"][surf]
                cells += [_c(sr["rows_OFF_banked"][key]["os_minus_refcv7_50400"], "worse"), _c(sr["read_ON"]["rows"][key]["os_minus_refcv7_50400"], "worse")]
            vo = r["bars"]["BAR-R8-NR_OFF_banked"]["metrics"][kk]["verdict"].split(" (")[0]
            vn = r["bars"]["BAR-R8-NR_ON"]["metrics"][kk]["verdict"].split(" (")[0]
            w(f"| {t} | {kk} | " + " | ".join(cells) + f" | {vo} -> {vn}{' **CHANGED**' if vo != vn else ''} |")
    w("")
    w("## 7. The four families at the primary tree (step 30,000 EMA), both seeds, clamp-OFF (banked) beside clamp-ON")
    w("")
    w("Per surface and metric: `os - ha0_ext` and `os - refcv7-50,400` (OFF, ON), and the paired ON - OFF on refcv8's own metric. TACTICAL rows here are the TRAJECTORY-DERIVED decisions (they move with the plan); the tactical HEAD logits are not touched by the clamp. STRATEGIC: not applicable (`--no-strategic`, n = 0).")
    for sd in SEEDS:
        for surf in ("S2", "S6"):
            sr = r0["seeds"][str(sd)]["surfaces"][surf]
            w("")
            w(f"**{surf}, inference seed {sd}** ({sr['read_ON']['n_windows']} windows / {sr['read_ON']['n_episodes']} episodes; {sr['n_windows_where_os_changed']} windows changed; os mean ADE ON {sr['read_ON']['os_means']['ade_m']:.4f} m vs OFF {sr['os_means_OFF_banked']['ade_m']:.4f} m)")
            w("")
            w("| family | metric | os - ha0_ext OFF | os - ha0_ext ON | os - refcv7 OFF | os - refcv7 ON | paired ON - OFF |")
            w("|---|---|---|---|---|---|---|")
            for fam, key, label, unit, lower, surfs in PR.METRICS:
                off, on = sr["rows_OFF_banked"][key], sr["read_ON"]["rows"][key]
                kd = "worse" if lower else None
                w(f"| {fam} | {label} ({unit}) | {_c(off['os_minus_ha0_ext'], 'better' if lower else None)} | {_c(on['os_minus_ha0_ext'], 'better' if lower else None)} | "
                  f"{_c(off['os_minus_refcv7_50400'], kd)} | {_c(on['os_minus_refcv7_50400'], kd)} | {_c(sr['paired_ON_minus_OFF_refcv8_own_metric'][key], kd)} |")
    w("")
    w("## 8. Attribution of the ADE move (exact decomposition) -- every tree, S2 / S6, seed 0 / 1")
    w("")
    w("| tree | surface | seed | mean ADE ON - OFF (m) | windows changed | changed AND GT>ceiling | changed AND GT within ceiling | mean per changed window (m) | contribution from GT>ceiling windows (m) | contribution from GT-within windows (m) | worsened / improved |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, r in trees.items():
        for surf in ("S2", "S6"):
            for sd in SEEDS:
                a = r["seeds"][str(sd)]["surfaces"][surf]["ade_delta_attribution"]
                w(f"| {t} | {surf} | {sd} | {a['mean_delta_all_windows']:+.4f} | {a['n_changed']} | {a['n_changed_gt_exceeds_ceiling']} | {a['n_changed_gt_within_ceiling']} | {a['mean_delta_over_changed_windows_m']} | "
                  f"{a['contribution_to_mean_delta_gt_exceeds_m']:+.4f} | {a['contribution_to_mean_delta_gt_within_m']:+.4f} | {a['n_changed_worsened']} / {a['n_changed_improved']} |")
    w("")
    w("## 9. Caveats and what this read does NOT decide")
    w("")
    w("* POST-HOC on banked plans, CPU fp32. K2 and K4 (GPU) are owed, so these are not eval-forward numbers; they decide nothing until K4 passes.")
    w("* T1 open loop on logged frames: a clamp that makes a plan slower than the logged ego cannot score better on ADE against that ego. Section 8 shows the clamp changes only windows where the logged ego itself exceeds the fed ceiling, so the ADE read is structurally one-sided; it measures what compliance costs against a ground truth that does not comply, not driving quality.")
    w("* Single training seed; the interval answers 'another draw of EPISODES' only -- not another training run, not another inference run (the two inference seeds are shown separately).")
    w("* The 5k-30k banked verdicts are NOT moved (amendment section 4); the ON column is an additional read.")
    w("* The NR reference side is unchanged (refcv7-50,400 without clamp); `D-R8-NR-INSTRUMENT` is open.")
    sp = OUT / "SENSITIVITY_brake.md"
    if sp.exists():
        w("")
        w("## 10. Exploratory: the braking-allowance frontier (primary tree; NOT amendment-covered, NO verdict)")
        w("")
        L.extend(sp.read_text(encoding="utf-8").splitlines()[2:])
    open(str(OUT / "CLAMP_POSTHOC.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("WROTE", OUT / "CLAMP_POSTHOC.json", OUT / "CLAMP_POSTHOC.md", f"({len(L)} lines)")
    return 0


# ------------------------------------------------------------------------------------------ exploratory: the braking-allowance frontier
BRAKES = (0.0, 1.0, 2.5, 5.0, 10.0)


def sensitivity(tree: str = PRIMARY) -> int:
    """EXPLORATORY, NOT covered by the amendment, NO verdict.  The landed function takes `brake_ms2` (= decoder.sel.accel_max, 2.5 in this run).
    How do compliance and the ADE cost trade as that allowance moves?  Same function, same inputs, same substitution path; only brake changes.
    brake 2.5 must reproduce the main read's paired ON - OFF ADE exactly (the cross-check)."""
    V, vrec = load_vendored()
    rc = run_cfg()
    NR, BP, PR, ra, sha12 = get_readers()
    import torch
    root = f"{ARCHIVE}/{tree}"
    out = {"tool": "clamp_posthoc_r8.py sensitivity", "tree": tree, "tier": TIER, "estimator": QUESTION, "generated": time.strftime("%FT%T"),
           "status": "EXPLORATORY post-hoc; the landed brake is the run's sel.accel_max = %s; nothing here is a bar verdict" % rc["brake_ms2"],
           "brakes_ms2": list(BRAKES), "cells": {}}
    main = jload(str(OUT / "trees" / f"{tree}.json"))
    t0 = time.time()
    for seed in SEEDS:
        raw, lim, linfo, dump = load_seed_inputs(tree, seed, rc)
        plan, v0 = raw["nav_true.traj"], raw["v0"]
        ex = NR.load_extras(os.path.dirname(dump))
        eid = raw["clip_index"]
        for b in BRAKES:
            cl, diag = apply_clamp(V, plan, lim, v0, rc, brake=b)
            ex["cl"] = cl.numpy().astype(np.float32)
            bind, resid = diag["plan_cap_binding"].numpy(), diag["plan_cap_excess_ms"].numpy()
            rec = {"binding_rate": cluster_rate(bind.astype(float), eid), "n_binding": int(bind.sum()),
                   "residual_above_ceiling_rate": cluster_rate((resid > EXEMPT_TOL_COARSE).astype(float), eid), "n_residual": int((resid > EXEMPT_TOL_COARSE).sum()),
                   "residual_excess_ms": q3(resid[resid > EXEMPT_TOL_COARSE]), "n_residual_strict_gt0_incl_fp_dust": int((resid > 0).sum())}
            for surf, pat in (("S2", "panel_s{s}"), ("S6", "s6_s{s}")):
                pdir = f"{root}/{pat.format(s=seed)}"
                man, files, ep_of = NR.panel_layout(pdir)
                slots = list(man["grid"]["slots"])
                wd = f"{SCRATCH}/{tree}_sens_s{seed}_{surf}"
                NR.write_variant(pdir, wd, files, NR.plans_from_extras(ex, ep_of, files, "cl", slots))
                man_o, A_o, eid_o, keys_o = PR.load_panel_keyed(wd, sha12)
                man_n, A_n, eid_n, keys_n = PR.load_panel_keyed(pdir, sha12)
                dt = float(man_o["grid"]["dt_s"])
                c_on, c_off, c_ha = PR.components(ra, A_o["os"], A_o["g"], dt), PR.components(ra, A_n["os"], A_n["g"], dt), PR.components(ra, A_n["ha0_ext"], A_n["g"], dt)
                rec[surf] = {"n_windows_changed": int((np.abs(A_o["os"].astype(np.float64) - A_n["os"].astype(np.float64)).reshape(len(eid_o), -1).max(1) > 0).sum()),
                             "ade_ON_minus_OFF": PR.paired_cell(c_on["ade_m"], c_off["ade_m"], eid_o, N_BOOT),
                             "ade_ON_minus_ha0_ext": PR.paired_cell(c_on["ade_m"], c_ha["ade_m"], eid_o, N_BOOT),
                             "speed_mae_ON_minus_OFF": PR.paired_cell(c_on["LON_speed_mae_mps"], c_off["LON_speed_mae_mps"], eid_o, N_BOOT),
                             "ade_mean_ON": float(np.nanmean(c_on["ade_m"])), "ade_mean_OFF": float(np.nanmean(c_off["ade_m"]))}
            out["cells"][f"s{seed}_brake{b}"] = rec
            print(f"[sens] s{seed} brake {b} done {time.time() - t0:.0f}s", flush=True)
    # cross-check: brake 2.5 == the main read
    chk = {}
    for seed in SEEDS:
        for surf in ("S2", "S6"):
            a = out["cells"][f"s{seed}_brake{rc['brake_ms2']}"][surf]["ade_ON_minus_OFF"]
            m = main["seeds"][str(seed)]["surfaces"][surf]["paired_ON_minus_OFF_refcv8_own_metric"]["ade_m"]
            chk[f"s{seed}_{surf}"] = {"sens": [a["delta"], a["lo"], a["hi"]], "main": [m["delta"], m["lo"], m["hi"]], "equal": [a["delta"], a["lo"], a["hi"]] == [m["delta"], m["lo"], m["hi"]]}
    out["crosscheck_brake_landed_equals_main_read"] = chk
    out["crosscheck_PASS"] = bool(all(v["equal"] for v in chk.values()))
    jdump(out, str(OUT / "SENSITIVITY_brake.json"))
    L = ["# SENSITIVITY_brake -- EXPLORATORY (not covered by the amendment; no verdict)", "",
         f"Tree `{tree}`, the landed function with only `brake_ms2` varied (landed = {rc['brake_ms2']} m/s^2 = `sel.accel_max`). TIER: {TIER}. ESTIMATOR: {QUESTION}. "
         f"Cross-check (brake {rc['brake_ms2']} reproduces the main read's paired ON - OFF ADE exactly): **{out['crosscheck_PASS']}**.", "",
         "| seed | brake (m/s^2) | binding rate (n) | still above ceiling after clamp (n) / max excess m/s | S2 ADE ON - OFF | S2 os - ha0_ext (ON) | S6 ADE ON - OFF | S6 os - ha0_ext (ON) |", "|---|---|---|---|---|---|---|---|"]
    for seed in SEEDS:
        for b in BRAKES:
            r = out["cells"][f"s{seed}_brake{b}"]
            L.append(f"| {seed} | {b}{' (landed)' if b == rc['brake_ms2'] else ''} | {100 * r['binding_rate']['mean']:.2f} % ({r['n_binding']}) | {100 * r['residual_above_ceiling_rate']['mean']:.2f} % ({r['n_residual']}) / {r['residual_excess_ms']['max'] if r['residual_excess_ms']['max'] is not None else '-'} | "
                     f"{_c(r['S2']['ade_ON_minus_OFF'], 'worse')} | {_c(r['S2']['ade_ON_minus_ha0_ext'], 'better')} | {_c(r['S6']['ade_ON_minus_OFF'], 'worse')} | {_c(r['S6']['ade_ON_minus_ha0_ext'], 'better')} |")
    L += ["", "Reading: brake 0 is the STRICT clamp (the literal R1: the plan may never exceed the ceiling, even when v0 already does). For a positive brake the allowance is "
          "max(limit, v0 - brake * t_mid): a SMALLER brake leaves an over-ceiling ego more speed (looser), a LARGER brake forces it down to the limit sooner (stricter), and brake -> infinity "
          "converges to the strict case. 'still above ceiling' counts rows more than 0.01 m/s over the ceiling after the clamp (fp32 dust of a plan capped AT the ceiling is excluded; the strict > 0 count is in the JSON). "
          "The ADE column is a cost against the LOGGED ego, which itself exceeds the ceiling on every binding window (CLAMP_POSTHOC.md section 8); it does not say which point the programme should pick."]
    open(str(OUT / "SENSITIVITY_brake.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("WROTE", OUT / "SENSITIVITY_brake.json", "crosscheck", out["crosscheck_PASS"])
    return 0 if out["crosscheck_PASS"] else 7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["vendor", "exactness", "read", "assemble", "sens", "r1refresh"])
    ap.add_argument("--tree")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT / "trees", exist_ok=True)
    if a.stage == "vendor":
        return vendor()
    if a.stage == "exactness":
        return exactness()
    if a.stage == "read":
        if not a.tree:
            ap.error("--tree is required for read")
        return read_tree(a.tree, a.force)
    if a.stage == "sens":
        return sensitivity()
    if a.stage == "r1refresh":
        return r1_refresh()
    return assemble()


if __name__ == "__main__":
    sys.exit(main())
