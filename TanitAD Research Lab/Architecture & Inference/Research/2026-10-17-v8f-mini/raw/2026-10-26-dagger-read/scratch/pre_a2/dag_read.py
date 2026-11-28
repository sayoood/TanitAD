#!/usr/bin/env python3
"""dag_read.py -- the reader for PREREG_V8F_DAGGER (landed 2a78f059: the 2026-10-26 draft + Amendment 1 `A1-2026-10-26`, claim row H-V8F-DAGGER-1), adapted from the landed orc_read.py (Amendment 4).

    DAG (M2b fine-tuned 3,000 steps, a fraction p = DAG_P of eligible samples replaced by a route-paired OWN state from the DAgger bank) vs
    DAG-0 (the same fine-tune with --dagger-p 0), per seed s in {0, 1}:

      R-TLAT / R-TLON  the T head's lat / lon macro-F1 at ticks >= 1 of the T1-kin plain causal rollout, DAG_s - DAG-0_s, through
                       taniteval.ci.paired_episode_cluster_bootstrap (B 2000, seed 0, over episodes), on the MATCHED cells (scored in both runs).
                       A head CLEARS at seed s iff separated-better AND point >= bar(h, s) = max(2F_h, 2 * RF_h,s), inclusive.
      RF               |F1_h(DAG-0_s) - F1_h(ORC-0_s)| on the cells scored in both, through the same estimator (ORC-0 is a same-argv replicate of DAG-0).
      R-ADE            T1-kin ADE6, DAG - DAG-0, paired per window.        guard: not separated-worse
      R-T0             T0 own-code ADE6, DAG - DAG-0 point.                 guard: <= +F/2 = 0.0604 m

    Outcome order (PREREG section 7): a failed control = REFUSED (rc 2, NOTHING written); coverage < 0.5 = BLOCKED (rc 4, written, no arm is read);
    then a failed guard = HARMS; then the number of heads that clear at both seeds = ROBUST (2) / PARTIAL (1) / NO EFFECT (0).

    python dag_read.py --manifest runs.json --out DAG_READ.json --tool-root <stack+taniteval export> --eval-cache .. --tracks .. --eval-grid .. --eval-join ..
                       --m2b-ckpt 0=<M2b_s0/ckpt_last.pt> --m2b-ckpt 1=<M2b_s1/ckpt_last.pt> [--t1hd-read P --t1k-read P] [--prereg FILE] [--key-dagger-n K ..]

    rc 0 an outcome row (ROBUST / PARTIAL / NO EFFECT / HARMS) was written; rc 2 REFUSED: NOTHING is written; rc 3 UNPLACED (unreachable: the table is complete);
    rc 4 BLOCKED (written; no launch).

    The bank is read as `v8f_dagger_bank.py` writes it (PROVENANCE.json + bank.json + shard provenance + the shard npz headers; schema in the comment above `sha12`). K-B3 is THREE controls
    (Amendment 1 section 3): K-B3.1 bank clips subset of M2b's TRAIN clips by name, K-B3.2 the TRAIN window-list sha256 equals the trainer's, K-B3.3 bank clips and eval-141 are disjoint.
    WRITTEN since the K-B4/K-B5 build (2026-10-26; MEASURED on fixtures/dagger_smoke and fixtures/trainer_kb45): the generator's PROVENANCE.json `controls` K-B4 / K-B5 records and the trainer's
    TRAIN window-list sha key (train_window_list_sha256, summary.json beside train_sha256). STILL PROVISIONAL (read from the trainer's source, no run output exists; each is a CLI override): the built-record
    key names dagger_p / dagger_bank_sha256 (levers) and the totals dagger_n / dagger_eligible.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------- the PREREG (committed numbers)
SEEDS = (0, 1)
STEPS, LR, WARMUP, PGT_START, PGT_END = 3000, 1e-4, 100, 0.2, 0.2          # section 4: PREREG_V8F_ORECOVER Amendment 1, unchanged
# THE registered p of this run (section 4: p = 0.3 if O-RECOVER read NO EFFECT, p = 0.1 if it read HARMS). Amendment 1 section 0 (PREREG, landed) applies the mapping: O-RECOVER read HARMS => p = 0.1. ONE named constant.
DAG_P = 0.1
ARM_P = {"DAG": DAG_P, "DAG0": 0.0}                                        # named in every argv as --dagger-p
EGO_P = 0.0                                                                # section 4: --ego-perturb-p 0 is named on BOTH arms (the two levers are never combined)
DAG_ONLY_FLAGS = ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")   # section 4: DAG names all four dagger flags, DAG-0 only --dagger-p 0
LAW = ("--ego-perturb-vscale", "--ego-perturb-vmax", "--ego-perturb-adrop", "--ego-perturb-yawrate-std", "--ego-perturb-seed")   # inert at p = 0: named on neither arm
RATIO_TOL = 0.02                                                           # section 6: dagger_n / dagger_eligible in p +- 0.02
SEED_FLOOR = {"lat": 0.0124, "lon": 0.0073}                                # section 5: the T0 fresh values
TWO_F = {"lat": 0.0248, "lon": 0.0146}                                     # 2F, committed
F_ADE = 0.1207                                                             # F, the ADE6 seed floor
HALF_F = 0.0604                                                            # F/2 as the PREREG commits it (R-T0 point bar)
N_BOOT = 2000
N_WINDOWS, N_EPISODES = 2399, 141                                          # eval-141
COVERAGE_MIN = 0.5                                                         # section 6 / 7: the eligible fraction of TRAIN windows, inclusive
N_TRAIN_CLIPS, N_TRAIN_WINDOWS = 4512, 306566                              # section 1 (MODEL_REGISTRY section 17)
REPLAY_FRAC = 0.005                                                        # K-B1: a seeded 0.5 % of bank rollouts
KB5_TOL_M = 1e-4                                                           # K-B5: held-frame relative transforms preserved to <= 1e-4 m
HMAX = 16
MATCH_CONST_SHA_PREFIX = "438366d5"                                        # section 2: probe/THRESHOLDS_FIXED.txt sha256 438366d5... (only the prefix is in the PREREG)
BANK_ROLLOUT = {"replan_steps": 10, "limit": "causal", "nav": "deadreckon", "scene": "logged", "schedule": "periodic", "executor": "plan", "post_map": "none (PLAIN)"}   # section 1; the generator's own key spelling
TICKS = (1, 2)                                                             # section 1: the bank holds own states at ticks 1 and 2 only
MATCH_CONSTANTS = {"ARC_TIE_M": 0.05, "COVERAGE_MIN": 0.5, "END_EDGE_ROWS": 31, "OFF_ROUTE_HEAD_DEG": 30.0, "OFF_ROUTE_LAT_M": 3.0, "REPLAN_STEPS": 10, "TICKS": [1, 2]}   # section 2 (THRESHOLDS_FIXED.txt), as the generator records them
KB4_CASES = ("straight", "stationary_tie", "lateral_cut", "heading_cut", "end_edge", "loop")                                                           # section 6, K-B4
REASON_OK = "ok"
CUT_REASONS = ("off_lat", "off_head", "end_edge", "nonfinite")             # section 2 steps 5 and 6 (the generator's names); `nonfinite` is the generator's own fifth reason (its A8), not one of the PREREG's three cuts
REASONS = (REASON_OK,) + CUT_REASONS
KB_IDS = ("K-B0", "K-B1", "K-B2", "K-B3", "K-B4", "K-B5")
BANK_KIND, BANK_SCHEMA = "bank", "v8f-dagger-bank-v1"                    # PROVENANCE.json's kind / schema as v8f_dagger_bank.py writes them
TIER = ("T1 (T1-kin: self-action OPEN loop, plain causal bucket, nav dead-reckoned; never 'closed loop') for R-TLAT / R-TLON / R-ADE; "
        "T0 (teacher-forced own-code, eval-141) for R-T0")
BANKED_M2B_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8",
                  1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
PREREG_DRAFT_SHA256 = "719e5f153070e9c6986c11dac9250ea3f2a2363968c9ea65225d8b582b029374"                       # the draft as banked at ac139f71 / 07dcb513
PREREG_LANDED_SHA256 = "a4df4b8f44d9a322a1907338c990dee09d03fb35e1e136e6c60eb72a790f60f9"                      # the landed file (2a78f059): the draft's bytes + `## Amendment 1`
PREREG_ID = ("Project Steering/PREREG_V8F_DAGGER.md @ commit 2a78f059 (landed file sha256 " + PREREG_LANDED_SHA256 + " = the DRAFT 2026-10-26 (sha256 " + PREREG_DRAFT_SHA256
             + ") + Amendment 1 (A1-2026-10-26))")

# The section-7 table, VERBATIM, in the PREREG's order (a test compares every cell with the file; markdown bold markers are markup, not text).
OUTCOME_TABLE = (
    ("ROBUST",
     "R-TLAT and R-TLON both clear (§5) at both seeds; R-ADE not separated-worse at either seed; R-T0 ≤ +F/2 at both",
     "adopt DAG as the base. Re-read T1 dominance over P10 and FLAT's committed repeat on it. If the crawl beside rows (M1, L2) still fail, next is DAgger iteration 2 (DAG's own states), under its own read"),
    ("PARTIAL",
     "exactly one head clears at both seeds; guards held",
     "adopt for the head that moved. Next: iteration 2 for the other head, under its own read"),
    ("NO EFFECT",
     "no head clears at both seeds; guards held. A head separated-better but under its bar is NO EFFECT, reported and never called an effect",
     "own-state supervision does not repair T at T1-kin at this dose. With ORC also null or harmful, training-distribution levers on T's ego channel are ELIMINATED. Next: T-EGOCUT: T's ego token (`v8f_mini.py:686`) replaced by a constant at training AND inference, M2b init, 3,000 steps, both seeds, under its own PREREG. It decides whether T should read its own ego state at all"),
    ("HARMS",
     "R-ADE separated-worse at ≥ 1 seed, or R-T0 above +F/2 at ≥ 1 seed",
     "DAG is not adopted; reported with its rows. If it ran at p = 0.3: next is this PREREG's arm at p = 0.1. If it ran at p = 0.1: next is NO EFFECT's next (T-EGOCUT)"),
    ("BLOCKED",
     "COVERAGE < 0.5 (§6)",
     "no launch. The stats are reported. Next: a rejoin-target form (a synthesised return to the route for off-route own states, ChauffeurNet), under its own PREREG"),
    ("REFUSED",
     "any control fails",
     "fix and re-run"),
)
OUTCOME_NAMES = tuple(r[0] for r in OUTCOME_TABLE)
# The HARMS cell has two branches, chosen by the p this run ran at (a lookup, not an interpretation; both strings are substrings of the cell, a test pins that).
HARMS_BRANCH = {0.3: "this PREREG's arm at p = 0.1", 0.1: "NO EFFECT's next (T-EGOCUT)"}
UNPLACED_TEXT = ("UNPLACED: this reading matches no row of the PREREG's section-7 table. The table is stated complete, so this should be unreachable: a reader or "
                 "PREREG defect. No outcome word is invented; the PI / Master Mind decides, WITHOUT looking at which way the numbers fall.")


class Refusal(Exception):
    """A control failed or an input is unusable. ``msgs`` lists every reason found."""

    def __init__(self, msgs):
        self.msgs = list(msgs) if isinstance(msgs, (list, tuple)) else [str(msgs)]
        super().__init__("; ".join(self.msgs))


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True, file=sys.stderr)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise Refusal(f"cannot read {p}: {type(e).__name__}: {e}")


def jenc(x):
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def need(p, what):
    if not p or not os.path.exists(p):
        raise Refusal(f"{what}: {p!r} is missing (the run did not finish, or was not pulled)")
    return p


# ---------------------------------------------------------------------------------------------------------------- argv / records
def parse_flags(argv):
    """argv -> {flag: [values-of-occurrence-1, values-of-occurrence-2, ...]}; EVERY value after a flag is kept (the programme's ``v8f_pgt_read._flags`` keeps only the first)."""
    out, i, argv = {}, 0, list(argv)
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            j = i + 1
            vals = []
            while j < len(argv) and not argv[j].startswith("--"):
                vals.append(argv[j])
                j += 1
            out.setdefault(t, []).append(vals)
            i = j
        else:
            i += 1
    return out


def argv_sha16(argv):
    """The trainer's own argv hash (v8f_mini_train.argv_sha)."""
    return hashlib.sha256(" ".join(argv).encode()).hexdigest()[:16]


def gate_argv_sha256(argv):
    """launch_gate.argv_sha256: sha256 of the argv as an ORDERED JSON array."""
    return hashlib.sha256(json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def norm_path(p):
    return str(p).replace("\\", "/").rstrip("/")


def tail2(p):
    parts = norm_path(p).split("/")
    return "/".join(parts[-2:])


def dig(d, dotted):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def find_key(sources, key):
    """All places ``key`` (a dotted path) is recorded: [(source_name, value)]. Looked up bare and under ``totals.`` / ``levers.`` / ``built_levers.``."""
    found = []
    for name, d in sources:
        if not isinstance(d, dict):
            continue
        for pre in ("", "totals.", "levers.", "built_levers."):
            ok, v = dig(d, pre + key)
            if ok:
                found.append((f"{name}:{pre}{key}", v))
    return found


def same_float(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def is_hex(x, n=64):
    return isinstance(x, str) and len(x) == n and all(ch in "0123456789abcdef" for ch in x)


class Ctl:
    """The controls. ``add`` records PASS / FAIL with its detail; a control that raises is a FAIL carrying the exception, never a skip."""

    def __init__(self):
        self.items = {}
        self.notes = []

    def add(self, cid, ok, detail=None):
        self.items[cid] = {"PASS": bool(ok), "detail": detail}
        return bool(ok)

    def run(self, cid, fn):
        try:
            ok, detail = fn()
        except Refusal as e:
            ok, detail = False, "; ".join(e.msgs)
        except Exception as e:      # noqa: BLE001  -- a control that cannot be evaluated has not passed
            ok, detail = False, f"{type(e).__name__}: {e}"
        return self.add(cid, ok, detail)

    def failed(self):
        return {k: v for k, v in self.items.items() if not v["PASS"]}

    def flags(self):
        return {k: v["PASS"] for k, v in self.items.items()}


# ---------------------------------------------------------------------------------------------------------------- the programme's readers (imported, never copied)
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def pin_stack(tool_root):
    """taniteval's stack guard pins ONE stack (this venv's editable install points at G:); the stack in use is the tool root's, said explicitly."""
    os.environ["TANITEVAL_STACK_OVERRIDE"] = os.path.join(os.path.abspath(tool_root), "stack")


def load_t1hd(path, tool_root):
    """t1hd_read, imported as a module (as raw/2026-10-26-t1heads-read/probe_lag.py does), with its tool imports set up."""
    pin_stack(tool_root)
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    T = _load_module("t1hd_read", path)
    T.setup_imports(tool_root)
    return T


def load_t1k(path, root):
    """t1k_read is a SCRIPT (it reads sys.argv at import); its `--selftest` door defines `comps` / `paired` / `selftest`, prints the R-3 result and exits.
    The module object survives the SystemExit, so `comps` is the programme's own function, byte for byte."""
    pin_stack(root)
    old = sys.argv
    sys.argv = ["t1k_read.py", "--selftest", root]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            try:
                _load_module("t1k_read_for_dag", path)
            except SystemExit as e:
                if e.code not in (0, None):
                    raise
    finally:
        sys.argv = old
    m = sys.modules["t1k_read_for_dag"]
    if '"R-3": "PASS"' not in buf.getvalue() or not hasattr(m, "comps"):
        raise Refusal("t1k_read's R-3 self-test did not report PASS (the ADE6 geometry is not the programme's)")
    return m


# ---------------------------------------------------------------------------------------------------------------- per-run packs (what a T1 run yields)
class Pack:
    """One T1-kin run reduced to what the pairing needs. ``cells[hd]`` = {"true": [N,3], "pred": [N,3]} (true = -1 where unscored)."""

    def __init__(self, label, ci, t0, cells, tick0_f1, meta, controls, goal=None, extra=None):
        self.label, self.ci, self.t0, self.cells = label, np.asarray(ci), np.asarray(t0), cells
        self.tick0_f1, self.meta, self.controls, self.goal, self.extra = tick0_f1, meta, controls, goal or {}, extra or {}

    @property
    def N(self):
        return len(self.ci)


class RealBackend:
    """The real path: t1hd_read's own pipeline (`run_one`: tick file -> matching -> cells -> K1/K2/K3/K5/K6/K7) on a run directory."""

    def __init__(self, T, ns, n_boot, work_dir):
        self.T, self.a, self.n_boot, self.work = T, ns, n_boot, work_dir
        self.ctx = T.Ctx(ns)
        self._cache = {}

    def analyze_t1_light(self, entry, label, seed):
        """The K4 source run (--limit logged): only its own K2 / K3 (and the grid) are controls; its tick 0 feeds K4."""
        T = self.T
        tick_path = need(entry["ticks"], f"{label} ticks file")
        res, R, tt = T.run_one(self.ctx, self.a, tick_path, None, label, f"s{seed}", self.n_boot, self.a.banked_dir, full_rows=False, k7=False)
        grid = [(int(c), int(t)) for c, t in zip(tt["ci"], tt["t0"])] == [(int(w[0]), int(w[1])) for w in self.ctx.wins_grid]
        ctl = {"K2": (bool(res["K2"]["PASS"]), None), "K3": (bool(res["K3"]["PASS"]), None), "GRID": (bool(grid), {"n": int(tt["N"])})}
        self._cache[tick_path] = (R, tt)
        return Pack(label, tt["ci"], tt["t0"], {}, res["tick0"]["f1"], dict(tt["meta"]), ctl)

    def analyze_t1(self, entry, label, seed):
        T = self.T
        tick_path = need(entry["ticks"], f"{label} ticks file")
        dump = need(entry["dump_dir"], f"{label} dump dir")
        k7 = entry.get("k7") or {"mode": "x1"}
        mode = k7.get("mode", "x1")
        sd = f"s{seed}"
        res, R, tt = T.run_one(self.ctx, self.a, tick_path, dump, label, sd, self.n_boot, k7.get("banked_dir") or self.a.banked_dir, full_rows=True,
                               k7=mode in ("ref", "banked"), ref_dump=k7.get("ref_dump_dir") if mode == "ref" else None)
        ctl = {}
        for k in ("K1", "K2", "K3", "K5", "K6"):
            ctl[k] = (bool(res[k]["PASS"]), None)
        if mode in ("ref", "banked"):
            ctl["K7"] = (bool(res["K7"]["PASS"]), {"mode": mode})
        else:
            e = res["executed_track_rebuild_X1"]
            ok = e["max_cl_m"] is not None and e["max_cl_m"] <= 1e-9 and e["max_pose_tick_f64"] <= 1e-9
            ctl["K7"] = (bool(ok), {"mode": "x1 (no banked / flags-off reference for a plain fine-tuned arm): rebuilt executed track == dumped cl", "x1": e})
        grid = [(int(c), int(t)) for c, t in zip(tt["ci"], tt["t0"])] == [(int(w[0]), int(w[1])) for w in self.ctx.wins_grid]
        ctl["GRID"] = (bool(grid and tt["N"] == len(self.ctx.wins_grid)), {"n": int(tt["N"]), "grid_n": len(self.ctx.wins_grid)})
        cells = {hd: {"true": R["cells"][hd]["true"], "pred": R["cells"][hd]["pred"]} for hd in ("lat", "lon")}
        goal = res["TACTICAL"]["beside"]["goal macro-F1"]
        sp = {k: {x: v[x] for x in ("delta", "lo", "hi", "separated")} for k, v in res["STRATEGIC"]["decisive"].items()}
        crawl = {hd: res["TACTICAL"]["crawl"][hd]["frac_ticks_ge1_crawl"] for hd in ("lat", "lon")}
        pack = Pack(label, tt["ci"], tt["t0"], cells, res["tick0"]["f1"], dict(tt["meta"]), ctl, {"f1_ge1": goal["f1_ge1"], "f1_tick0": goal["f1_tick0"]},
                    {"S-POS vs CTRV (ticks>=1, within this run)": sp, "crawl_fraction_ticks_ge1": crawl, "off_route": res["off_route"]})
        self._cache[tick_path] = (R, tt)
        pack.ticks_path = tick_path
        return pack

    def k4(self, entry_logged, pack, t0_entry, seed):
        """K4 with the programme's own `k4_tick0_is_T0`, which reads `eval_M2b_<sd>-sfresh-fresh.{json,npz}` from a dir: the arm's OWN T0 eval is staged under that name."""
        T = self.T
        sd = f"s{seed}"
        stage = tempfile.mkdtemp(prefix="dag_k4_", dir=self.work)
        try:
            shutil.copyfile(need(t0_entry["json"], "T0 eval json"), os.path.join(stage, f"eval_M2b_{sd}-sfresh-fresh.json"))
            shutil.copyfile(need(t0_entry["npz"], "T0 eval npz"), os.path.join(stage, f"eval_M2b_{sd}-sfresh-fresh.npz"))
            if entry_logged is not None:
                R, tt = self._cache[entry_logged["ticks"]]
                src = "a --limit logged run's tick 0 (the T1-HEADS K4 configuration)"
            else:
                R, tt = self._cache[pack.ticks_path]
                src = "the decisive --limit causal run's own tick 0 (NO logged-bucket run supplied)"
            r = T.k4_tick0_is_T0(R, tt, tt["ci"], sd, stage)
            return bool(r["PASS"]), {"source": src, "abs_diff_vs_own_T0": r["abs_diff_vs_json"], "tolerance": r["tolerance"], "got": r["got"],
                                     "per_window": r["per_window"]}
        finally:
            shutil.rmtree(stage, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------- manifest
PATH_KEYS = {"train_dir", "token", "metrics", "ticks", "dump_dir", "json", "npz", "ref_dump_dir", "banked_dir", "report", "bank_file", "eval141_clip_ids", "train_clip_ids"}


def _resolve(x, base):
    if isinstance(x, dict):
        return {k: (os.path.normpath(os.path.join(base, v)) if k in PATH_KEYS and isinstance(v, str) and not os.path.isabs(v) else _resolve(v, base)) for k, v in x.items()}
    return x


def load_manifest(path):
    m = jload(path)
    m = _resolve(m, os.path.dirname(os.path.abspath(path)))
    if "arms" not in m or not isinstance(m["arms"], dict):
        raise Refusal("manifest has no 'arms'")
    return m


def entry(m, arm, seed):
    e = (m["arms"].get(arm) or {}).get(str(seed))
    if e is None:
        raise Refusal(f"manifest has no {arm} seed {seed}: the verdict needs BOTH seeds of DAG, DAG0 and the ORC0 replicate (a lever claim on one seed is what H-ESTIM-SEED-1 forbids)")
    return e


# ---------------------------------------------------------------------------------------------------------------- the training-run controls
def train_records(e, keys_dir):
    d = need(e.get("train_dir"), "train_dir")
    r = {"dir": d}
    for f in ("argv.json", "summary.json", "selfcheck.json"):
        r[f] = jload(need(os.path.join(d, f), f"{os.path.basename(d)}/{f}"))
    mp = e.get("metrics") or os.path.join(d, "metrics.json")
    r["metrics.json"] = jload(mp) if os.path.exists(mp) else None
    r["argv"] = list(r["argv.json"].get("argv", []))
    r["flags"] = parse_flags(r["argv"])
    r["sources"] = [("summary", r["summary.json"]), ("selfcheck", r["selfcheck.json"]), ("metrics", r["metrics.json"])]
    return r


def one_value(flags, name):
    occ = flags.get(name)
    if not occ or len(occ) != 1 or len(occ[0]) != 1:
        return False, None
    return True, occ[0][0]


def g_argv(tag, arm, seed, r):
    """The registered fine-tune, NAMED on the argv (the flags' defaults disable the lever, so a default is not a test of it). DAG names all four dagger flags,
    DAG-0 only --dagger-p 0 (the gate refuses declared-and-inert flags at p = 0); --ego-perturb-p 0 is named on both, and the ego-perturbation law on neither."""
    bad, fl = [], r["flags"]
    want = {"--steps": STEPS, "--lr": LR, "--warmup": WARMUP, "--p-gt-start": PGT_START, "--p-gt-end": PGT_END, "--ego-perturb-p": EGO_P, "--dagger-p": ARM_P[arm]}
    for name, w in want.items():
        ok, v = one_value(fl, name)
        if not ok:
            bad.append(f"{name} is not named exactly once with one value (argv: {fl.get(name)})")
        elif not same_float(v, w):
            bad.append(f"{name} {v}, registered {w}")
    ok, v = one_value(fl, "--init-ckpt")
    if not ok:
        bad.append("--init-ckpt is not named exactly once")
    elif not norm_path(v).endswith(f"M2b_s{seed}/ckpt_last.pt"):
        bad.append(f"--init-ckpt {v} is not .../M2b_s{seed}/ckpt_last.pt")
    ok, v = one_value(fl, "--seed")
    if not ok or str(v) != str(seed):
        bad.append(f"--seed {v}, expected the M2b checkpoint's seed {seed}")
    ok, v = one_value(fl, "--arm")
    if not ok or v != "M2b":
        bad.append(f"--arm {v}, expected M2b")
    for name in LAW:
        if name in fl:
            bad.append(f"{name} is named on {arm}: the ego-perturbation law is inert at --ego-perturb-p 0 and the two levers are never combined (section 4)")
    if arm == "DAG":
        ok, v = one_value(fl, "--dagger-bank")
        if not ok or not v:
            bad.append(f"--dagger-bank is not named exactly once with one value (argv: {fl.get('--dagger-bank')})")
        ok, v = one_value(fl, "--dagger-bank-sha256")
        if not ok or not is_hex(v):
            bad.append(f"--dagger-bank-sha256 is not named exactly once as a 64-hex sha256 (argv: {fl.get('--dagger-bank-sha256')})")
        ok, v = one_value(fl, "--dagger-seed")
        try:
            int(v)
            seed_ok = ok
        except (TypeError, ValueError):
            seed_ok = False
        if not seed_ok:
            bad.append(f"--dagger-seed is not named exactly once as an integer (argv: {fl.get('--dagger-seed')})")
    else:
        for name in DAG_ONLY_FLAGS:
            if name in fl:
                bad.append(f"{name} is named on DAG-0 (section 4: DAG-0 names only --dagger-p 0)")
    return not bad, bad or "argv names the registered fine-tune"


def g_summary(tag, seed, r):
    sm, sc, aj = r["summary.json"], r["selfcheck.json"], r["argv.json"]
    bad = []
    if sm.get("done") is not True:
        bad.append("summary.json done is not true")
    if sm.get("steps") != STEPS:
        bad.append(f"summary.json steps {sm.get('steps')}, registered {STEPS}")
    if sm.get("arm") != "M2b" or sm.get("seed") != seed:
        bad.append(f"summary.json says arm {sm.get('arm')} seed {sm.get('seed')}, expected M2b seed {seed}")
    shas = {"summary": sm.get("argv_sha"), "selfcheck": sc.get("argv_sha"), "argv.json": aj.get("sha"), "recomputed": argv_sha16(r["argv"])}
    if len(set(shas.values())) != 1:
        bad.append(f"argv_sha disagree: {shas}")
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"selfcheck declared_vs_built {sc.get('declared_vs_built')!r}, not PASS")
    return not bad, bad or {"argv_sha": shas["recomputed"], "steps": sm.get("steps")}


def init_sha_values(r, key):
    """The recorded init-checkpoint sha256 values: the lever record(s) (key CLI-overridable) and, if the argv path exists here, the file itself."""
    vals = [(w, v) for w, v in find_key(r["sources"], key)]
    ok, p = one_value(r["flags"], "--init-ckpt")
    if ok and os.path.isfile(p):
        vals.append((f"file:{p}", sha256_file(p)))
    return vals


def g_init(tag, seed, r, m2b_sha, key):
    vals = init_sha_values(r, key)
    if not vals:
        return False, f"no record of the init checkpoint's sha256 (looked for '{key}' in summary / selfcheck / metrics levers) and no readable file at --init-ckpt: the init is unproven"
    bad = [f"{w} = {str(v)[:16]}" for w, v in vals if str(v) != m2b_sha]
    return not bad, (f"init sha256 differs from the M2b checkpoint passed on the CLI ({m2b_sha[:16]}): " + "; ".join(bad)) if bad else {"init_sha256": m2b_sha, "sources": [w for w, _ in vals]}


def g_reached(tag, arm, r, kn, ke, kp):
    """The lever REACHED training (section 6): built p == argv p; DAG dagger_n / dagger_eligible within p +- 0.02; DAG-0 dagger_n exactly 0."""
    bad, det = [], {}
    bp = find_key(r["sources"], kp)
    if not bp:
        bad.append(f"no built value '{kp}' recorded (the lever's built_levers entry)")
    for w, v in bp:
        if not same_float(v, ARM_P[arm]):
            bad.append(f"{w} = {v}, argv names {ARM_P[arm]}")
    ns, es = find_key(r["sources"], kn), find_key(r["sources"], ke)
    if not ns:
        bad.append(f"no '{kn}' recorded by the trainer (the totals are a build item)")
    if len({json.dumps(v) for _, v in ns}) > 1:
        bad.append(f"'{kn}' recorded with conflicting values {ns}")
    if len({json.dumps(v) for _, v in es}) > 1:
        bad.append(f"'{ke}' recorded with conflicting values {es}")
    if ns:
        n = ns[0][1]
        det["dagger_n"] = n
        if arm == "DAG0":
            if not (isinstance(n, (int, float)) and n == 0):
                bad.append(f"DAG-0 {kn} = {n}, must be exactly 0")
        else:
            if not es:
                bad.append(f"no '{ke}' recorded")
            else:
                e = es[0][1]
                det["dagger_eligible"] = e
                if not (isinstance(n, (int, float)) and isinstance(e, (int, float)) and e > 0 and 0 <= n <= e):
                    bad.append(f"{kn} {n} / {ke} {e}: not a count pair with 0 <= n <= eligible, eligible > 0")
                else:
                    ratio = n / e
                    det["ratio"] = ratio
                    if not (DAG_P - RATIO_TOL - 1e-12 <= ratio <= DAG_P + RATIO_TOL + 1e-12):
                        bad.append(f"DAG replaced fraction {n}/{e} = {ratio:.4f} outside {DAG_P} +- {RATIO_TOL}")
    return not bad, bad or det


def read_token(path):
    return jload(need(path, "launch-gate token"))


def g_token(tag, r, tok):
    """A launch-gate PASS token exists, is a PASS, and binds THIS run's argv (commit / argv sha256 / data manifest present). The HMAC needs the launch host's
    key and is NOT verified here."""
    bad = []
    if tok.get("kind") != "launch-gate-token":
        bad.append(f"kind {tok.get('kind')!r}, not 'launch-gate-token'")
    if tok.get("verdict") != "PASS":
        bad.append(f"verdict {tok.get('verdict')!r}, not PASS")
    b = tok.get("binding") or {}
    argv = r["argv"]
    cands = {gate_argv_sha256(argv)}
    k = 0
    while k < len(argv) and not argv[k].startswith("--"):
        k += 1
        cands.add(gate_argv_sha256(argv[k:]))
    if b.get("argv_sha256") not in cands:
        bad.append(f"binding.argv_sha256 {str(b.get('argv_sha256'))[:16]} is not this run's argv (sha256 {gate_argv_sha256(argv)[:16]})")
    c = str(b.get("commit") or "")
    if len(c) != 40 or any(ch not in "0123456789abcdef" for ch in c):
        bad.append(f"binding.commit {c!r} is not a 40-hex commit")
    d = str(b.get("data_sha256") or "")
    if len(d) != 64:
        bad.append("binding.data_sha256 (the data manifest) is absent")
    return not bad, bad or {"commit": c, "argv_sha256": b.get("argv_sha256"), "data_sha256": d, "hmac": "present, NOT verified (key on the launch host)" if tok.get("hmac_sha256") else "absent"}


SUMMARY_SAME = ("steps", "n_clips", "n_windows", "train_sha256", "excluded_sha256", "seed")


def g_pair(seed, rd, r0, key):
    """DAG vs DAG-0 of one seed (section 4, the pair rule): argv identical except --out, --dagger-p and the three DAG-only flags; same init, steps, data, seed.
    The launch-gate data manifests are NOT required equal: DAG's includes the bank's sha256 (section 6) and DAG-0 names no bank."""
    bad = []
    drop = {"--out", "--dagger-p"} | set(DAG_ONLY_FLAGS)
    fd = {k: v for k, v in rd["flags"].items() if k not in drop}
    f0 = {k: v for k, v in r0["flags"].items() if k not in drop}
    if fd != f0:
        diff = sorted(k for k in set(fd) | set(f0) if fd.get(k) != f0.get(k))
        bad.append(f"argv differs in {diff} besides --out, --dagger-p and the DAG-only flags -- not one lever")
    sd, s0 = rd["summary.json"], r0["summary.json"]
    for k in SUMMARY_SAME:
        if sd.get(k) != s0.get(k):
            bad.append(f"summary {k}: DAG {sd.get(k)} vs DAG-0 {s0.get(k)}")
    id_, i0 = {str(v) for _, v in init_sha_values(rd, key)}, {str(v) for _, v in init_sha_values(r0, key)}
    if id_ != i0 or not id_:
        bad.append(f"init checkpoint sha256: DAG {sorted(id_)} vs DAG-0 {sorted(i0)}")
    return not bad, bad or "same init, steps, data, seed; argv identical except --out, --dagger-p and the DAG-only flags"


def g_replicate(seed, r0, ro, key):
    """Section 4: 'DAG-0's argv equals ORC-0's plus --dagger-p 0. ORC-0 is therefore a same-argv REPLICATE of DAG-0' -- the replicate floor RF is only a floor if that holds."""
    bad = []
    f0 = {k: v for k, v in r0["flags"].items() if k != "--out"}
    fo = {k: v for k, v in ro["flags"].items() if k != "--out"}
    if "--dagger-p" in fo:
        bad.append(f"ORC-0's argv already names --dagger-p {fo['--dagger-p']}: it is not the same argv minus that flag")
    fo = {**fo, "--dagger-p": [["0"]]}
    if f0 != fo:
        diff = sorted(k for k in set(f0) | set(fo) if f0.get(k) != fo.get(k))
        bad.append(f"DAG-0's argv is not ORC-0's plus --dagger-p 0: differs in {diff}")
    s0, so = r0["summary.json"], ro["summary.json"]
    for k in SUMMARY_SAME:
        if s0.get(k) != so.get(k):
            bad.append(f"summary {k}: DAG-0 {s0.get(k)} vs ORC-0 {so.get(k)}")
    i0, io_ = {str(v) for _, v in init_sha_values(r0, key)}, {str(v) for _, v in init_sha_values(ro, key)}
    if i0 != io_ or not io_:
        bad.append(f"init checkpoint sha256: DAG-0 {sorted(i0)} vs ORC-0 {sorted(io_)}")
    return not bad, bad or "DAG-0's argv equals ORC-0's plus --dagger-p 0, same init / steps / data / seed: ORC-0 is a same-argv replicate"


# ---------------------------------------------------------------------------------------------------------------- the bank (section 1, 2, 6; Amendment 1 section 3)
# The bank is what `stack/scripts/v8f_dagger_bank.py` WRITES (reconciled with the built generator and its smoke, A1-era; schema "v8f-dagger-bank-v1"). One bank directory:
#   PROVENANCE.json            {schema, kind "bank", bank_sha256, manifest "bank.json", shard_provenance [..], n_records, coverage{n_windows, n_eligible, fraction, min, blocked},
#                               reasons_by_tick{"1","2": {ok, off_lat, off_head, end_edge, nonfinite}}, partial, warnings}          <- the manifest's bank.{s}.report
#   bank.json                  {schema, n_shards, shards[{file, k, n_records, file_sha256, content_sha256}], common{ckpt_sha256, rollout, constants, thresholds_file_sha256, cfg{hmax},
#                               train{n_clips, n_windows, train_sha256}, window_list_sha256, eval{n_clips, clips, overlap}, partial, limit_clips, ..}, partial, n_records, coverage, reasons_by_tick}
#                               its sha256 IS the bank's sha256
#   shard_KKKofNNN.provenance.json   per shard: {plan{shard [k, n], window_list_sha256, ckpt_sha256 (None in gen mode), ..}, inputs{ckpt{sha256}}, kb0_tick0_identity{checked, mismatch},
#                               kb1_replay{chunks, windows, mismatch_chunks, frac, seed}, kb3_eval_disjoint{n_eval_clips, overlap}, coverage, reasons_by_tick, file_sha256, ..}
#   shard_KKKofNNN.npz         the records; its `header_json` member holds {"clips": {ci: clip NAME}} -- the ONLY place the bank's clips are listed by name
# PROVENANCE.json `controls` {K-B4{PASS, cases{..}, ..}, K-B5{PASS, max_pose_now_err_m, max_rel_transform_err_m, tol_m, ..}} is written by the generator since the K-B4/K-B5 build (fixtures/dagger_smoke); a bank from
# the older generator lacks it and the reader REFUSES it ("needs the bank generator to add"). On the TRAINER side the TRAIN window-list sha256 (windows_sha256(J["clip"], wins)) is `train_window_list_sha256`.
def sha12(c):
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]


def stable_sid(c):
    """v8f_mini_data.stable_episode_id (a test compares it with the tip's function)."""
    return int.from_bytes(hashlib.blake2b(c.encode("utf-8"), digest_size=8).digest(), "big") >> 1


def train_list_sha256(ids):
    """The trainer's own TRAIN CLIP-list hash (v8f_mini_train.py: sha256 of the clip ids sorted by sha12, newline-joined, recorded as `train_sha256`); a test compares it with the tip's source."""
    return hashlib.sha256(chr(10).join(sorted(ids, key=sha12)).encode("utf-8")).hexdigest()


def clip_overlap(bank_ids, eval_ids):
    """K-B3.3's disjointness, by clip id AND by stable sid (v8f_mini_data.assert_disjoint). Returns the problems found."""
    t, e = set(bank_ids), set(eval_ids)
    bad = []
    if not e:
        bad.append("the eval-141 clip-id list is empty: disjointness proves nothing")
    if t & e:
        bad.append(f"bank and eval-141 clip ids overlap: {sorted(t & e)[:3]} ({len(t & e)})")
    if {stable_sid(c) for c in t} & {stable_sid(c) for c in e}:
        bad.append("bank and eval-141 stable sids overlap")
    return bad


def train_subset_problems(bank_names, train_ids):
    """K-B3.1: every clip the bank holds is a clip of M2b's TRAIN list, matched BY NAME. The bank may hold FEWER (Amendment 1 section 3: equality is not required)."""
    t, bad = set(train_ids), []
    if not t:
        bad.append("the TRAIN clip list is empty: the subset proves nothing")
    if not bank_names:
        bad.append("the bank holds no clips")
    out = sorted(set(bank_names) - t)
    if out:
        bad.append(f"{len(out)} bank clip(s) are not in M2b's TRAIN clip list, e.g. {out[:3]}")
    return bad


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def is_count(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _shard_clip_names(path, what):
    """{ci: clip name} from the shard npz's `header_json` member (only that member is read)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            h = json.loads(bytes(z["header_json"]).decode("utf-8"))
        clips = h["clips"]
        if not isinstance(clips, dict) or not clips:
            raise ValueError("header_json['clips'] is not a non-empty mapping")
        return {str(k): str(v) for k, v in clips.items()}
    except Refusal:
        raise
    except Exception as e:      # noqa: BLE001
        raise Refusal(f"{what}: cannot read the clip names from {path}: {type(e).__name__}: {e}")


def load_bank(report_path, bank_file=None, what="bank report"):
    """Reads one bank directory as the generator wrote it. Raises Refusal only when a file is missing or unreadable; every consistency check is a CONTROL (g_bank_files / g_bank_config / ...)."""
    prov = jload(need(report_path, f"{what} (PROVENANCE.json)"))
    if not isinstance(prov, dict) or prov.get("kind") != BANK_KIND or prov.get("schema") != BANK_SCHEMA:
        raise Refusal(f"{what}: kind {prov.get('kind') if isinstance(prov, dict) else None!r} / schema {prov.get('schema') if isinstance(prov, dict) else None!r}, "
                      f"not the generator's {BANK_KIND!r} / {BANK_SCHEMA!r}")
    d = os.path.dirname(os.path.abspath(report_path))
    mp = need(bank_file or os.path.join(d, str(prov.get("manifest") or "bank.json")), f"{what}: the bank manifest (bank.json)")
    man = jload(mp)
    if not isinstance(man, dict) or not isinstance(man.get("shards"), list) or not man["shards"] or not isinstance(man.get("common"), dict):
        raise Refusal(f"{what}: bank.json has no 'shards' / 'common'")
    provs = [jload(need(os.path.join(d, nm), f"{what}: shard provenance {nm}")) for nm in (prov.get("shard_provenance") or [])]
    shards = []
    for s in man["shards"]:
        f = need(os.path.join(d, str(s.get("file"))), f"{what}: shard {s.get('file')}")
        shards.append({"file": s.get("file"), "sha256": sha256_file(f), "clips": _shard_clip_names(f, f"{what}: shard {s.get('file')}")})
    return {"dir": d, "prov": prov, "man": man, "man_sha256": sha256_file(mp), "shard_provs": provs, "shards": shards}


def common_of(B):
    return B["man"]["common"]


def bank_clip_names(B):
    return sorted({n for s in B["shards"] for n in s["clips"].values()})


def kb_record(B, kid):
    c = ((B["prov"].get("controls") or {}) if isinstance(B["prov"].get("controls"), dict) else {}).get(kid)
    if not isinstance(c, dict):
        raise Refusal(f"the bank's PROVENANCE.json carries no 'controls.{kid}' record (the generator does not write it: needs the bank generator to add it)")
    if c.get("PASS") is not True:
        raise Refusal(f"{kid} is recorded PASS = {c.get('PASS')!r}")
    return c


def g_bank_files(B):
    """The files are the ones the report names: bank.json hashes to the report's bank_sha256; every shard file to the manifest's; the shard provenance files belong to those shards; the counts add up."""
    prov, man, c, bad = B["prov"], B["man"], common_of(B), []
    if B["man_sha256"] != prov.get("bank_sha256"):                       # (a non-hex value in PROVENANCE can never equal a computed sha256: no separate format check)
        bad.append(f"bank.json hashes to {B['man_sha256'][:16]}, not PROVENANCE.bank_sha256 {str(prov.get('bank_sha256'))[:16]}")
    ms = man["shards"]
    if man.get("n_shards") != len(ms) or len(B["shard_provs"]) != len(ms):
        bad.append(f"n_shards {man.get('n_shards')!r} / manifest shards {len(ms)} / shard provenance files {len(B['shard_provs'])} disagree")
    for m_, sf in zip(ms, B["shards"]):
        if sf["sha256"] != m_.get("file_sha256"):
            bad.append(f"shard {m_.get('file')} hashes to {sf['sha256'][:16]}, not the manifest's {str(m_.get('file_sha256'))[:16]}")
    by_k = {(sp.get("plan") or {}).get("shard", [None])[0]: sp for sp in B["shard_provs"]}
    for m_ in ms:
        sp = by_k.get(m_.get("k"))
        if sp is None:
            bad.append(f"no shard provenance for shard k={m_.get('k')}")
            continue
        if sp.get("file_sha256") != m_.get("file_sha256") or sp.get("n_records") != m_.get("n_records"):
            bad.append(f"shard k={m_.get('k')}: its provenance (file_sha256 {str(sp.get('file_sha256'))[:16]}, n_records {sp.get('n_records')}) is not the manifest's")
        if (sp.get("plan") or {}).get("window_list_sha256") != c.get("window_list_sha256"):
            bad.append(f"shard k={m_.get('k')}: plan.window_list_sha256 differs from the manifest's common.window_list_sha256")
    tot = sum(m_.get("n_records") or 0 for m_ in ms)
    if not (tot == man.get("n_records") == prov.get("n_records")):
        bad.append(f"n_records: shards sum {tot}, bank.json {man.get('n_records')}, PROVENANCE {prov.get('n_records')}")
    if not (prov.get("partial") == man.get("partial") == c.get("partial")):
        bad.append(f"partial flag differs: PROVENANCE {prov.get('partial')!r}, bank.json {man.get('partial')!r}, common {c.get('partial')!r}")
    names = [n for s in B["shards"] for n in s["clips"].values()]
    cis = [k for s in B["shards"] for k in s["clips"]]
    if len(set(names)) != len(names) or len(set(cis)) != len(cis):
        bad.append("a clip (or a clip index) appears in more than one shard")
    return not bad, bad or {"bank_sha256": prov.get("bank_sha256"), "n_shards": len(ms), "n_records": tot, "n_bank_clips": len(names)}


def g_bank_config(B, eval_ids):
    """The bank's configuration equals section 1 / 2: complete (not --limit-clips), the rollout, the matching constants, hmax, the TRAIN plan size, the eval list the generator checked."""
    c, bad = common_of(B), []
    cst = c.get("constants") if isinstance(c.get("constants"), dict) else {}
    if c.get("schema") != BANK_SCHEMA:
        bad.append(f"common.schema {c.get('schema')!r}, expected {BANK_SCHEMA!r}")
    if c.get("partial") is not False or c.get("limit_clips") is not None:
        bad.append(f"the bank is PARTIAL (partial {c.get('partial')!r}, limit_clips {c.get('limit_clips')!r}): a --limit-clips bank is a smoke, never the bank an arm trains on")
    if c.get("rollout") != BANK_ROLLOUT:
        bad.append(f"common.rollout {c.get('rollout')!r}, registered {BANK_ROLLOUT}")
    for k, v in MATCH_CONSTANTS.items():
        if cst.get(k) != v:
            bad.append(f"common.constants.{k} {cst.get(k)!r}, registered {v!r} (section 2)")
    if cst.get("REASONS") != list(REASONS):
        bad.append(f"common.constants.REASONS {cst.get('REASONS')!r}, expected {list(REASONS)}")
    if not str(c.get("thresholds_file_sha256") or "").lower().startswith(MATCH_CONST_SHA_PREFIX):
        bad.append(f"common.thresholds_file_sha256 {str(c.get('thresholds_file_sha256'))[:16]!r} does not start with the PREREG's {MATCH_CONST_SHA_PREFIX} (probe/THRESHOLDS_FIXED.txt, section 2)")
    if (c.get("cfg") or {}).get("hmax") != HMAX:
        bad.append(f"common.cfg.hmax {(c.get('cfg') or {}).get('hmax')!r}, registered {HMAX}")
    tr = c.get("train") if isinstance(c.get("train"), dict) else {}
    if tr.get("n_clips") != N_TRAIN_CLIPS or tr.get("n_windows") != N_TRAIN_WINDOWS:
        bad.append(f"common.train n_clips {tr.get('n_clips')!r} / n_windows {tr.get('n_windows')!r}, registered {N_TRAIN_CLIPS} / {N_TRAIN_WINDOWS}")
    for k, v in (("ckpt_sha256", c.get("ckpt_sha256")), ("window_list_sha256", c.get("window_list_sha256")), ("train.train_sha256", tr.get("train_sha256"))):
        if not is_hex(v):
            bad.append(f"common.{k} is not a 64-hex sha256")
    ev = c.get("eval") if isinstance(c.get("eval"), dict) else {}
    if not eval_ids or set(ev.get("clips") or []) != set(eval_ids) or ev.get("overlap") != 0:
        bad.append(f"the eval clips the generator checked (n {len(ev.get('clips') or [])}, overlap {ev.get('overlap')!r}) are not the eval-141 list given to the reader (n {len(set(eval_ids))})")
    return not bad, bad or {"rollout": c.get("rollout"), "thresholds_file_sha256": c.get("thresholds_file_sha256"), "train": tr, "hmax": HMAX}


def g_kb0(B):
    nw = (B["prov"].get("coverage") or {}).get("n_windows")
    recs = [sp.get("kb0_tick0_identity") for sp in B["shard_provs"]]
    if not recs or any(not (isinstance(r, dict) and is_count(r.get("checked")) and is_count(r.get("mismatch"))) for r in recs):
        raise Refusal(f"K-B0 (tick-0 identity): a shard provenance carries no kb0_tick0_identity {{checked, mismatch}}: {recs}")
    chk, mm = sum(r["checked"] for r in recs), sum(r["mismatch"] for r in recs)
    if not (isinstance(nw, int) and nw > 0 and chk == nw and mm == 0):
        raise Refusal(f"K-B0 (tick-0 identity) must have checked == every window ({nw}) and mismatch == 0: checked {chk}, mismatch {mm}")
    return True, {"checked": chk, "n_windows": nw, "mismatch": 0, "verified_by_reader": False}


def g_kb1(B):
    nw = (B["prov"].get("coverage") or {}).get("n_windows")
    recs = [sp.get("kb1_replay") for sp in B["shard_provs"]]
    if not recs or any(not (isinstance(r, dict) and is_count(r.get("chunks")) and is_count(r.get("windows")) and is_count(r.get("mismatch_chunks")) and num(r.get("frac"))) for r in recs):
        raise Refusal(f"K-B1 (replay): a shard provenance carries no kb1_replay {{chunks, windows, mismatch_chunks, frac}}: {recs}")
    need_n = max(1, math.floor(REPLAY_FRAC * nw)) if isinstance(nw, int) else None
    w, ch, mm = sum(r["windows"] for r in recs), sum(r["chunks"] for r in recs), sum(r["mismatch_chunks"] for r in recs)
    low = [r["frac"] for r in recs if r["frac"] < REPLAY_FRAC]
    if not (need_n is not None and ch >= 1 and w >= need_n and mm == 0 and not low):
        raise Refusal(f"K-B1 (replay) must have mismatch_chunks == 0, a replay fraction >= {REPLAY_FRAC:.1%} in every shard (got {low or 'ok'}) and windows >= {need_n} "
                      f"(a seeded {REPLAY_FRAC:.1%} of {nw} rollouts): chunks {ch}, windows {w}, mismatch_chunks {mm}")
    return True, {"chunks": ch, "windows": w, "required": need_n, "mismatch_chunks": 0, "verified_by_reader": False}


def g_kb2(B, m2b_sha):
    """K-B2 (derived by the reader; the generator records the checkpoint sha256, not a PASS): the bank's generating checkpoint is the M2b checkpoint passed on the CLI."""
    g = common_of(B).get("ckpt_sha256")
    if g != m2b_sha:
        raise Refusal(f"K-B2 (on-policy): the bank was generated by checkpoint {str(g)[:16]}, not the M2b checkpoint passed on the CLI ({str(m2b_sha)[:16]})")
    odd = [((sp.get("inputs") or {}).get("ckpt") or {}).get("sha256") for sp in B["shard_provs"]]
    if any(o != g for o in odd):
        raise Refusal(f"K-B2: a shard's inputs.ckpt.sha256 {[str(o)[:12] for o in odd]} is not the manifest's common.ckpt_sha256 {g[:12]}")
    return True, {"generator_ckpt_sha256": g, "verified_by_reader": "common.ckpt_sha256 and every shard's inputs.ckpt.sha256 vs the CLI's M2b sha256"}


def g_kb31(B, train_ids):
    """K-B3.1 (Amendment 1): the bank's clips are a SUBSET of M2b's TRAIN clips, matched by name; fewer is allowed. Own deliberate regression: a bank with one non-TRAIN clip must be refused."""
    names = bank_clip_names(B)
    bad = train_subset_problems(names, train_ids)
    if not bad and not train_subset_problems(names + ["\x00not-a-train-clip"], train_ids):
        bad.append("the reader's subset check did not refuse a bank holding a clip that is not in TRAIN (the check is blind)")
    if bad:
        raise Refusal(bad)
    return True, {"n_bank_clips": len(names), "n_train_clips": len(set(train_ids)), "n_train_clips_without_bank_records": len(set(train_ids) - set(names)),
                  "deliberate_regression": "refused", "verified_by_reader": "clip names from the shard npz headers vs the TRAIN list"}


def g_kb33(B, eval_ids):
    """K-B3.3 (Amendment 1): bank clips ∩ eval-141 = ∅ (ids AND stable sids). Own deliberate regression: one eval clip injected must be refused. Also the generator's own record."""
    names = bank_clip_names(B)
    bad = clip_overlap(names, eval_ids)
    for sp in B["shard_provs"]:
        k3 = sp.get("kb3_eval_disjoint")
        if not (isinstance(k3, dict) and k3.get("overlap") == 0 and k3.get("n_eval_clips") == len(set(eval_ids))):
            bad.append(f"a shard's kb3_eval_disjoint {k3!r} is not {{overlap 0, n_eval_clips {len(set(eval_ids))}}}")
    if bad == [] and not clip_overlap(names + [sorted(eval_ids)[0]], eval_ids):
        bad.append("the reader's disjointness check did not refuse a bank holding one eval clip (the check is blind)")
    if bad:
        raise Refusal(bad)
    return True, {"n_bank_clips": len(names), "n_eval_clips": len(set(eval_ids)), "overlap": 0, "deliberate_regression": "refused", "verified_by_reader": "ids and stable sids vs the eval-141 list"}


def g_kb4(B):
    c = kb_record(B, "K-B4")
    cases = c.get("cases") if isinstance(c.get("cases"), dict) else {}
    miss = [k for k in KB4_CASES if cases.get(k) is not True]
    if miss:
        raise Refusal(f"K-B4 (the matcher's synthetic cases) must record every case True; not True / absent: {miss}")
    return True, {"cases": list(KB4_CASES), "verified_by_reader": False}


def g_kb5(B):
    c = kb_record(B, "K-B5")
    for k in ("max_pose_now_err_m", "max_rel_transform_err_m"):
        if not (num(c.get(k)) and c[k] <= KB5_TOL_M):
            raise Refusal(f"K-B5 (re-anchoring) must record {k} <= {KB5_TOL_M}: {c.get(k)!r}")
    return True, {"max_pose_now_err_m": c["max_pose_now_err_m"], "max_rel_transform_err_m": c["max_rel_transform_err_m"], "verified_by_reader": False}


def coverage_of(B):
    """The coverage, validated and RECOMPUTED from the counts: eligible fraction = n_eligible / n_windows; the states are summed from reasons_by_tick (ok = kept, the rest = cuts by reason)."""
    cv, rbt = B["prov"].get("coverage"), B["prov"].get("reasons_by_tick")
    if not isinstance(cv, dict):
        raise Refusal("the bank's PROVENANCE.json has no 'coverage' block")
    for k in ("n_windows", "n_eligible"):
        if not is_count(cv.get(k)):
            raise Refusal(f"coverage.{k} is not a non-negative integer: {cv.get(k)!r}")
    if cv["n_windows"] <= 0:
        raise Refusal("coverage.n_windows is 0")
    if not isinstance(rbt, dict) or any(not isinstance(v, dict) or any(not is_count(v.get(r)) for r in REASONS) for v in rbt.values()):
        raise Refusal(f"reasons_by_tick must count every reason {list(REASONS)} per tick (section 2: every cut is counted by reason): {rbt!r}")
    by_tick = {t: sum(v[r] for r in REASONS) for t, v in rbt.items()}
    cuts = {r: sum(v[r] for v in rbt.values()) for r in CUT_REASONS}
    kept = sum(v[REASON_OK] for v in rbt.values())
    return {"n_windows": cv["n_windows"], "n_eligible_windows": cv["n_eligible"], "eligible_fraction": cv["n_eligible"] / cv["n_windows"],
            "n_states_before_cuts": kept + sum(cuts.values()), "n_states_kept": kept, "cuts": cuts, "states_by_tick": by_tick, "stated_fraction": cv.get("fraction")}


def g_bank_accounting(B):
    """The arithmetic of section 1 / 2: every window has one own state at EACH of ticks 1 and 2 before cuts (2 x windows in all); cut records are stored (A11), so the states equal n_records;
    eligible windows <= kept states; the stated fraction / blocked / min are the recomputed ones."""
    cv, bad = coverage_of(B), []
    prov, man, c = B["prov"], B["man"], common_of(B)
    if c.get("partial") is False and cv["n_windows"] != (c.get("train") or {}).get("n_windows"):
        bad.append(f"coverage.n_windows {cv['n_windows']} != common.train.n_windows {(c.get('train') or {}).get('n_windows')}")
    if sorted(cv["states_by_tick"]) != [str(t) for t in TICKS]:
        bad.append(f"reasons_by_tick ticks {sorted(cv['states_by_tick'])}, expected {[str(t) for t in TICKS]}")
    for t, n in cv["states_by_tick"].items():
        if n != cv["n_windows"]:
            bad.append(f"tick {t}: {n} own states before cuts != n_windows {cv['n_windows']} (2 x n_windows = {2 * cv['n_windows']} in all: ticks 1 and 2 of every window)")
    if cv["n_states_before_cuts"] != prov.get("n_records") or cv["n_states_before_cuts"] != man.get("n_records"):
        bad.append(f"states before cuts {cv['n_states_before_cuts']} != n_records (PROVENANCE {prov.get('n_records')}, bank.json {man.get('n_records')}): a cut record is stored, never dropped")
    if cv["n_eligible_windows"] > min(cv["n_windows"], cv["n_states_kept"]):
        bad.append(f"n_eligible {cv['n_eligible_windows']} exceeds the windows / kept states")
    sf = cv["stated_fraction"]
    if not (num(sf) and abs(sf - cv["eligible_fraction"]) <= 1e-9):
        bad.append(f"the stated coverage.fraction {sf!r} != n_eligible / n_windows = {cv['eligible_fraction']}")
    if prov["coverage"].get("min") != COVERAGE_MIN or prov["coverage"].get("blocked") is not (cv["eligible_fraction"] < COVERAGE_MIN):
        bad.append(f"coverage.min {prov['coverage'].get('min')!r} / blocked {prov['coverage'].get('blocked')!r} vs the registered {COVERAGE_MIN} / {cv['eligible_fraction'] < COVERAGE_MIN}")
    if man.get("coverage") != prov.get("coverage") or man.get("reasons_by_tick") != prov.get("reasons_by_tick"):
        bad.append("bank.json's coverage / reasons_by_tick differ from PROVENANCE.json's")
    return not bad, bad or {k: cv[k] for k in ("n_windows", "n_eligible_windows", "eligible_fraction", "n_states_before_cuts", "n_states_kept", "cuts", "states_by_tick")}


def bank_stage(ctl, s, B, m2b_sha, eval_ids, train_ids):
    """Every bank control that needs no arm run (all of them can run before any training)."""
    ctl.run(f"BANK-FILES:s{s}", lambda: g_bank_files(B))
    ctl.run(f"BANK-CONFIG:s{s}", lambda: g_bank_config(B, eval_ids))
    ctl.run(f"K-B0:s{s}", lambda: g_kb0(B))
    ctl.run(f"K-B1:s{s}", lambda: g_kb1(B))
    ctl.run(f"K-B2:s{s}", lambda: g_kb2(B, m2b_sha))
    ctl.run(f"K-B3.1:s{s}", lambda: g_kb31(B, train_ids))
    ctl.run(f"K-B3.3:s{s}", lambda: g_kb33(B, eval_ids))
    ctl.run(f"K-B4:s{s}", lambda: g_kb4(B))
    ctl.run(f"K-B5:s{s}", lambda: g_kb5(B))
    ctl.run(f"BANK-ACCOUNTING:s{s}", lambda: g_bank_accounting(B))


def trainer_values(rd, r0, key, what):
    """The trainer's recorded value of `key` on DAG and DAG-0: ({arm: value}, problems). Missing, conflicting inside a run, or different between the arms are problems."""
    tw, bad = {}, []
    for nm, r in (("DAG", rd), ("DAG-0", r0)):
        vals = find_key(r["sources"], key)
        if not vals:
            bad.append(f"{nm}: the trainer's '{key}' is not recorded (needs the trainer to add it)")
        elif len({json.dumps(v) for _, v in vals}) > 1:
            bad.append(f"{nm}: '{key}' recorded with conflicting values {vals}")
        else:
            tw[nm] = vals[0][1]
    if len(set(tw.values())) > 1:
        bad.append(f"the trainer's '{key}' differs between DAG and DAG-0: {tw}")
    return tw, bad


def g_kb32(B, rd, r0, key_w):
    """K-B3.2 (Amendment 1): the TRAIN window-list sha256 the bank records (common.window_list_sha256 = windows_sha256(J["clip"], wins)) equals the trainer's, on BOTH arms."""
    bank = common_of(B).get("window_list_sha256")
    tw, bad = trainer_values(rd, r0, key_w, "window list")
    for nm, v in tw.items():
        if v != bank:
            bad.append(f"common.window_list_sha256 {str(bank)[:16]} != the trainer's ({nm}) {str(v)[:16]}")
    return not bad, bad or {"window_list_sha256": bank, "trainer": tw}


def g_train_list(B, train_ids, rd, r0, key_t):
    """What makes K-B3.1's TRAIN list M2b's: it hashes (the trainer's own formula) to the trainer's recorded clip-list sha256 on both arms, and to the TRAIN plan the generator recorded."""
    got = train_list_sha256(train_ids)
    tw, bad = trainer_values(rd, r0, key_t, "clip list")
    for nm, v in tw.items():
        if v != got:
            bad.append(f"the TRAIN clip list given to the reader hashes to {got[:16]}, not the trainer's ({nm}) '{key_t}' {str(v)[:16]}")
    gen = (common_of(B).get("train") or {}).get("train_sha256")
    if gen != got:
        bad.append(f"the generator planned over a TRAIN list hashing to common.train.train_sha256 {str(gen)[:16]}, not {got[:16]}")
    return not bad, bad or {"train_sha256": got, "trainer": tw}


def g_bank_declared(tag, seed, B, r, m2b_sha, key_init, key_bank):
    """The DAG arm trained on THIS bank (PREREG section 4: DAG names `--dagger-bank-sha256 <hex>`): (A) the argv value equals the trainer's RECORDED `levers.dagger_bank_sha256`, and (B) that record is
    the sha256 of the bank's bank.json (the trainer's definition of the bank's identity; hashed here, BANK-FILES ties it to PROVENANCE). Each comparison is its own message.
    K-B2 on the arm: the bank's generating checkpoint is the arm's --init-ckpt."""
    bsha, bad = B["man_sha256"], []
    ok, declared = one_value(r["flags"], "--dagger-bank-sha256")
    if not ok:
        bad.append(f"--dagger-bank-sha256 is not named exactly once with one value (argv: {r['flags'].get('--dagger-bank-sha256')})")
    built = find_key(r["sources"], key_bank)
    if not built:
        bad.append(f"no recorded bank sha256 '{key_bank}' (the lever's built_levers entry)")
    for w, v in built:
        if ok and str(v) != declared:
            bad.append(f"argv --dagger-bank-sha256 {str(declared)[:16]} != the recorded {w} = {str(v)[:16]}")
        if str(v) != bsha:
            bad.append(f"the recorded {w} = {str(v)[:16]} != the sha256 of the bank's bank.json {bsha[:16]} (the trainer's definition of the bank's identity)")
    inits = init_sha_values(r, key_init)
    if not inits:
        bad.append("no record of the arm's init checkpoint sha256: K-B2 cannot be shown on the arm")
    gen = common_of(B).get("ckpt_sha256")
    for w, v in inits:
        if str(v) != gen or str(v) != m2b_sha:
            bad.append(f"K-B2: {w} = {str(v)[:16]} but the bank was generated by {str(gen)[:16]} (M2b {m2b_sha[:16]})")
    return not bad, bad or {"bank_sha256": bsha, "built_records": [w for w, _ in built], "bank_manifest_hashed_here": True}


def header_summary(B):
    c = common_of(B)
    return {"bank_sha256": B["man_sha256"], "ckpt_sha256": c.get("ckpt_sha256"), "window_list_sha256": c.get("window_list_sha256"), "thresholds_file_sha256": c.get("thresholds_file_sha256"),
            "constants_sha256_generator_own": c.get("constants_sha256"), "rollout": c.get("rollout"), "hmax": (c.get("cfg") or {}).get("hmax"), "train": c.get("train"),
            "partial": c.get("partial"), "limit_clips": c.get("limit_clips"), "n_shards": B["man"].get("n_shards"), "n_bank_clips": len(bank_clip_names(B)),
            "n_records": B["man"].get("n_records"), "generator_warnings": B["prov"].get("warnings")}


# ---------------------------------------------------------------------------------------------------------------- T1 / T0 file controls
def load_dump(dump_dir):
    import glob
    fs = sorted(glob.glob(os.path.join(need(dump_dir, "dump dir"), "ep*.npz")))
    if not fs:
        raise Refusal(f"no ep*.npz in {dump_dir}")
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "fl", "eid", "clip_index", "ws")}


def g_t1cfg(label, hdr_, ckpt_tail, want_limit, ticks_meta):
    """The run is the registered rollout: plain causal bucket, m = 10, periodic, logged scene, dead-reckoned nav, heads dumped; it scored THIS arm's checkpoint."""
    h = hdr_.get("t1v8f") or {}
    bad = []
    for k, w in (("limit", want_limit), ("replan_steps", 10), ("schedule", "periodic"), ("scene", "logged"), ("nav", "deadreckon")):
        if h.get(k) != w:
            bad.append(f"t1v8f.{k} = {h.get(k)!r}, registered {w!r}")
    if h.get("ticks_heads") is not True:
        bad.append("t1v8f.ticks_heads is not true (--dump-heads)")
    if tail2(h.get("ckpt", "")) != ckpt_tail:
        bad.append(f"the run scored {h.get('ckpt')!r}, not .../{ckpt_tail}")
    if tail2(str(ticks_meta.get("ckpt", ""))) != ckpt_tail:
        bad.append(f"the ticks file says it scored {ticks_meta.get('ckpt')!r}, not .../{ckpt_tail}")
    if h.get("tag") != ticks_meta.get("tag"):
        bad.append(f"run json tag {h.get('tag')!r} != ticks file tag {ticks_meta.get('tag')!r}")
    return not bad, bad or {k: h.get(k) for k in ("limit", "replan_steps", "schedule", "scene", "nav", "ticks_heads", "kin_mode")}


def t0_arrays(t0e, ckpt_tail):
    js = jload(need(t0e["json"], "T0 eval json"))
    z = np.load(need(t0e["npz"], "T0 eval npz"))
    return js, {"ade6": np.asarray(z["ade6"], np.float64), "eid": np.asarray(z["eid"])}, tail2(str(js.get("ckpt", ""))) == ckpt_tail


# ---------------------------------------------------------------------------------------------------------------- the rows
def bar_for(hd, rf):
    """section 5: bar(h, s) = max(2F_h, 2 * RF_h,s). Without the replicate floor it is 2F; it can only rise, and only by the measured run-to-run gap."""
    return max(TWO_F[hd], 2.0 * rf)


def f1_pair_row(T, pa, pb, hd, n_boot, matched=True):
    """R-TLAT / R-TLON: macro-F1 at ticks >= 1, ``pa`` - ``pb`` (higher is better), the T1-HEADS slot layout (2 cells per window) and reducer. Matched: only cells
    scored in BOTH runs enter either side; unmatched: each run's own scored cells."""
    n = T.NCLS[hd]
    ea = T.E0._f1_encoded(pa.cells[hd]["true"], pa.cells[hd]["pred"], n)[:, 1:]
    eb = T.E0._f1_encoded(pb.cells[hd]["true"], pb.cells[hd]["pred"], n)[:, 1:]
    if matched:
        m = (ea >= 0) & (eb >= 0)
        ea, eb = np.where(m, ea, -1), np.where(m, eb, -1)
    a, b = ea.reshape(-1), eb.reshape(-1)
    eid2 = np.repeat(pa.ci, 2)
    red = T.f1_red(n)
    row = T.pbrow(f"R-T{hd.upper()}F1 ({pa.label} - {pb.label}, ticks>=1, {'matched' if matched else 'own'} cells)", a, b, eid2, n_boot, red, record=False,
                  lower_is_better=False, head=hd)
    row.update(n_cells_a=int((a >= 0).sum()), n_cells_b=int((b >= 0).sum()), n_cells_matched_pairs=int(((ea >= 0) & (eb >= 0)).sum()),
               f1_a=float(red(a)), f1_b=float(red(b)), two_F=TWO_F[hd], F=SEED_FLOOR[hd], matched=bool(matched))
    row["meets_2F"] = bool(row["better"] and row["delta"] >= TWO_F[hd])
    return row, (a, b, eid2, red)


def err_pair_row(T, name, a, b, eid, n_boot):
    """An error row ``a`` - ``b`` (lower is better) through the programme's own wrapper; ``point`` is the UNROUNDED difference of the means."""
    r = T.srow(name, a, b, eid, n_boot, record=False)
    r["point"] = None if r["mean_a"] is None else float(r["mean_a"] - r["mean_b"])
    return r


def k1_own(T, items, n_boot):
    """K1 for this reader's OWN decisive rows: each (a, a) pair through the same estimator must read exactly +0 [0, 0]."""
    bad = []
    for name, a, eid, red in items:
        r = T.PB(a, a, eid, n_boot=n_boot, seed=0, reduce=red)
        if not (r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0):
            bad.append({"row": name, "delta": r["delta"], "lo": r["lo"], "hi": r["hi"]})
    return not bad, bad or {"n_rows": len(items)}


# ---------------------------------------------------------------------------------------------------------------- the outcome (PREREG section 7)
def decide(rows):
    """``rows[seed]`` = {"R-TLAT": row, "R-TLON": row, "R-ADE": row, "R-T0": {"point": float}} with delta / separated per row, and RF on the two F1 rows. Returns the
    outcome, EVERY matching row in table order, and the per-seed flags. REFUSED and BLOCKED are not decided here (a failed control or a coverage failure stops the
    read before any row exists)."""
    if tuple(sorted(rows)) != tuple(SEEDS):
        raise Refusal(f"the outcome needs exactly seeds {SEEDS}, got {sorted(rows)}")
    hd_row = {"lat": "R-TLAT", "lon": "R-TLON"}

    def bar(s, hd):
        return bar_for(hd, rows[s][hd_row[hd]]["RF"])

    def sep_better(s, hd):
        r = rows[s][hd_row[hd]]
        return bool(r["separated"] and r["delta"] > 0)

    def clears(s, hd):                                              # separated-better AND point >= bar(h, s), inclusive
        r = rows[s][hd_row[hd]]
        return bool(sep_better(s, hd) and r["delta"] >= bar(s, hd))

    ade_worse = {s: bool(rows[s]["R-ADE"]["separated"] and rows[s]["R-ADE"]["delta"] > 0) for s in SEEDS}
    t0_above = {s: bool(rows[s]["R-T0"]["point"] > HALF_F) for s in SEEDS}
    guards_held = not any(ade_worse.values()) and not any(t0_above.values())
    clear_both = {hd: all(clears(s, hd) for s in SEEDS) for hd in ("lat", "lon")}
    sep_both = {hd: all(sep_better(s, hd) for s in SEEDS) for hd in ("lat", "lon")}
    n_clear = sum(clear_both.values())
    match = {
        "ROBUST": bool(n_clear == 2 and guards_held),
        "PARTIAL": bool(n_clear == 1 and guards_held),
        "NO EFFECT": bool(n_clear == 0 and guards_held),
        "HARMS": bool(any(ade_worse.values()) or any(t0_above.values())),
    }
    matching = [n for n in OUTCOME_NAMES if match.get(n)]
    flags = {"clears_per_seed": {hd: {f"s{s}": clears(s, hd) for s in SEEDS} for hd in ("lat", "lon")},
             "separated_better_per_seed": {hd: {f"s{s}": sep_better(s, hd) for s in SEEDS} for hd in ("lat", "lon")},
             "bar_per_seed": {hd: {f"s{s}": bar(s, hd) for s in SEEDS} for hd in ("lat", "lon")},
             "RF_per_seed": {hd: {f"s{s}": rows[s][hd_row[hd]]["RF"] for s in SEEDS} for hd in ("lat", "lon")},
             "R-ADE_separated_worse_per_seed": {f"s{s}": ade_worse[s] for s in SEEDS}, "R-T0_above_F_over_2_per_seed": {f"s{s}": t0_above[s] for s in SEEDS},
             "guards_held": guards_held, "head_clears_at_both_seeds": clear_both,
             "sub_bar_separated_better_at_both_seeds": {hd: bool(sep_both[hd] and not clear_both[hd]) for hd in ("lat", "lon")}}
    out = matching[0] if matching else "UNPLACED"
    return {"OUTCOME": out, "matching_rows_in_table_order": matching, "flags": flags}


def table_row(name):
    return next((c, n) for nm, c, n in OUTCOME_TABLE if nm == name)


def harms_next(p):
    """The HARMS cell's two branches, picked by the p this run ran at (a lookup, not an interpretation): at p = 0.3 the next is this PREREG's arm at p = 0.1;
    at p = 0.1 it is NO EFFECT's next (T-EGOCUT), whose committed text is carried."""
    br = HARMS_BRANCH.get(p)
    if br is None:
        return {"ran_at_p": p, "branch": None, "next": None, "note": "p is not one of the two registered values (0.3, 0.1): no branch applies"}
    return {"ran_at_p": p, "branch": br, "next": table_row("NO EFFECT")[1] if p == 0.1 else br}


def outcome_block(d):
    out = d["OUTCOME"]
    if out == "UNPLACED":
        return {**d, "condition": None, "committed_next": None, "note": UNPLACED_TEXT}
    cond, nxt = table_row(out)
    blk = {**d, "condition": cond, "committed_next": nxt}
    if out == "HARMS":
        blk["committed_next_for_this_run"] = harms_next(DAG_P)
    return blk


def blocked_block(cov):
    cond, nxt = table_row("BLOCKED")
    return {"OUTCOME": "BLOCKED", "matching_rows_in_table_order": ["BLOCKED"], "condition": cond, "committed_next": nxt,
            "coverage": {f"s{s}": cov[s] for s in SEEDS}, "coverage_min": COVERAGE_MIN,
            "blocked_seeds": [f"s{s}" for s in SEEDS if cov[s]["eligible_fraction"] < COVERAGE_MIN]}


def check_prereg_text(path):
    """Optional: every embedded table cell occurs verbatim in the PREREG file (markdown bold markers stripped)."""
    t = open(path, encoding="utf-8").read().replace("**", "")
    miss = [f"{nm}: {x}" for nm, c, n in OUTCOME_TABLE for x in (nm, c, n) if x not in t]
    return not miss, miss or "every outcome-table cell is verbatim in the PREREG file"


# ---------------------------------------------------------------------------------------------------------------- the read
def read(cfg, backend, T, K):
    """cfg: manifest (resolved), m2b_sha {seed: hex}, banked_sha, keys, n_boot, n_windows, n_episodes. Returns the result dict; raises Refusal(msgs)."""
    ctl = Ctl()
    m = cfg["manifest"]
    nb = cfg["n_boot"]
    kn, ke, kp, kb, ki, kw, kt = (cfg["keys"][k] for k in ("n", "eligible", "built_p", "bank_sha", "init_sha", "window_sha", "train_sha"))
    if sorted(int(s) for s in m.get("seeds", SEEDS)) != list(SEEDS):
        ctl.add("SEEDS", False, f"manifest seeds {m.get('seeds')}, the read needs {list(SEEDS)}")
    if cfg.get("prereg"):
        ctl.run("PREREG-TEXT", lambda: check_prereg_text(cfg["prereg"]))
    for s in SEEDS:
        ctl.run(f"M2B-CKPT:s{s}", lambda s=s: (cfg["m2b_sha"].get(s) == cfg["banked_sha"].get(s) and cfg["m2b_sha"].get(s) is not None,
                                              {"passed_on_cli": str(cfg["m2b_sha"].get(s))[:16], "banked_PREREG": str(cfg["banked_sha"].get(s))[:16]}))
    # ---------------- the bank stage: every bank control (section 6, before any training), then COVERAGE. Needs no arm run.
    eval_ids = jload(need(m.get("eval141_clip_ids"), "eval141_clip_ids (the eval-141 clip-id list K-B3 is checked against)"))
    if not isinstance(eval_ids, list) or not all(isinstance(x, str) for x in eval_ids):
        raise Refusal("eval141_clip_ids is not a JSON list of clip-id strings")
    train_ids = jload(need(m.get("train_clip_ids"), "train_clip_ids (M2b's TRAIN clip list K-B3.1 is checked against; it must hash to the trainer's recorded clip-list sha256)"))
    if not isinstance(train_ids, list) or not all(isinstance(x, str) for x in train_ids):
        raise Refusal("train_clip_ids is not a JSON list of clip-id strings")
    banks = {}
    for s in SEEDS:
        be = (m.get("bank") or {}).get(str(s))
        if be is None:
            raise Refusal(f"manifest has no bank for seed {s}: COVERAGE and the bank controls are read for BOTH seeds")
        banks[s] = {"B": load_bank(be.get("report"), be.get("bank_file"), what=f"bank seed {s} report"), "entry": be}
        bank_stage(ctl, s, banks[s]["B"], cfg["m2b_sha"].get(s, ""), eval_ids, train_ids)
    if ctl.failed():
        raise Refusal([f"{k}: {v['detail']}" for k, v in ctl.failed().items()])
    cov = {s: coverage_of(banks[s]["B"]) for s in SEEDS}
    if any(cov[s]["eligible_fraction"] < COVERAGE_MIN for s in SEEDS):       # BLOCKED: no launch; the stats are reported; no arm is read
        present = sorted(a for a in ("DAG", "DAG0") if m["arms"].get(a))
        blk = blocked_block(cov)
        blk["unverified"] = ["K-B3.2 (the bank's window-list sha256 vs the trainer's) and the TRAIN-LIST binding: read from the arm runs, none may exist under a BLOCK"]
        if present:
            blk["warning"] = f"arm entries {present} are in the manifest although COVERAGE < {COVERAGE_MIN} forbids the launch: they were NOT read"
        return {"rows": {}, "OUTCOME": blk, "controls": ctl.items, "controls_PASS": ctl.flags(),
                "bank": {f"s{s}": {"header": header_summary(banks[s]["B"]), "coverage": cov[s]} for s in SEEDS}, "runs": {}}
    runs = {}                                                       # (arm, seed) -> dict
    # ---------------- the files: training records, tokens, T1 json, dumps, T0
    arms = ["DAG", "DAG0", "ORC0"] + [a for a in ("ORC", "M2b") if a in m["arms"]]
    for arm in arms:
        for s in SEEDS:
            if arm in ("ORC", "M2b") and str(s) not in (m["arms"][arm] or {}):
                continue
            e = entry(m, arm, s)
            tag = f"{arm}_s{s}"
            rec = {"arm": arm, "seed": s, "tag": tag, "entry": e}
            if arm in ("DAG", "DAG0", "ORC0"):
                rec["train"] = train_records(e, ki)
                tr = rec["train"]
                ctl.run(f"SUMMARY:{tag}", lambda tr=tr, s=s, tag=tag: g_summary(tag, s, tr))
                ctl.run(f"INIT-CKPT:{tag}", lambda tr=tr, s=s, tag=tag: g_init(tag, s, tr, cfg["m2b_sha"].get(s, ""), ki))
            if arm in ("DAG", "DAG0"):
                rec["token"] = read_token(e.get("token"))
                tr = rec["train"]
                ctl.run(f"ARGV:{tag}", lambda tr=tr, arm=arm, s=s, tag=tag: g_argv(tag, arm, s, tr))
                ctl.run(f"DAGGER-REACHED:{tag}", lambda tr=tr, arm=arm, tag=tag: g_reached(tag, arm, tr, kn, ke, kp))
                ctl.run(f"GATE-TOKEN:{tag}", lambda tr=tr, rc=rec, tag=tag: g_token(tag, tr, rc["token"]))
            if arm == "DAG":
                ctl.run(f"BANK-DECLARED:{tag}", lambda tr=rec["train"], s=s, tag=tag: g_bank_declared(tag, s, banks[s]["B"], tr, cfg["m2b_sha"].get(s, ""), ki, kb))
            ckpt_tail = e.get("ckpt_tail") or (f"{os.path.basename(norm_path(e['train_dir']))}/ckpt_last.pt" if e.get("train_dir") else f"{tag}/ckpt_last.pt")
            rec["ckpt_tail"] = ckpt_tail
            t1 = e.get("t1")
            if not t1:
                raise Refusal(f"{tag}: manifest has no 't1' (the decisive --limit causal run)")
            rec["t1json"] = jload(need(t1["json"], f"{tag} t1v8f json"))
            rec["dump"] = load_dump(t1.get("dump_dir"))
            t0e = e.get("t0")
            if not t0e:
                raise Refusal(f"{tag}: manifest has no 't0' (the chain's standard T0 eval)")
            rec["t0json"], rec["t0arr"], t0_ok = t0_arrays(t0e, ckpt_tail)
            ctl.add(f"T0-CKPT:{tag}", t0_ok, f"T0 eval json ckpt {rec['t0json'].get('ckpt')!r} vs .../{ckpt_tail}")
            t0f = e.get("t0_fresh")                                 # A4 (inherited from ORC): K4's comparator is the arm's own T0 eval at --ages fresh, from its OWN key
            if not t0f:
                raise Refusal(f"{tag}: manifest has no 't0_fresh' (K4 compares tick 0 with the arm's own T0 eval at --ages fresh, ORC Amendment 4)")
            rec["t0fjson"] = jload(need(t0f["json"], f"{tag} fresh T0 eval json"))
            ctl.add(f"T0F-CKPT:{tag}", tail2(str(rec["t0fjson"].get("ckpt", ""))) == ckpt_tail,
                    f"fresh T0 eval json ckpt {rec['t0fjson'].get('ckpt')!r} vs .../{ckpt_tail}")
            ages = (rec["t0json"].get("ages"), rec["t0fjson"].get("ages"))     # R-T0 can never read the fresh eval, K4 never the standard one
            ctl.add(f"T0-AGES:{tag}", ages == ("cycle", "fresh"),
                    {"t0": ages[0], "t0_fresh": ages[1], "committed": ["cycle: box 2's standard eval (R-T0, T0-CKPT, the families)", "fresh: K4 only"]})
            runs[(arm, s)] = rec
    # ---------------- pair controls: DAG vs DAG-0 (the pair rule), DAG-0 vs ORC-0 (the replicate), the bank's window list vs the trainer's
    for s in SEEDS:
        rd, r0, ro = runs[("DAG", s)], runs[("DAG0", s)], runs[("ORC0", s)]
        ctl.run(f"PAIR:s{s}", lambda rd=rd, r0=r0, s=s: g_pair(s, rd["train"], r0["train"], ki))
        ctl.run(f"ORC0-REPLICATE:s{s}", lambda r0=r0, ro=ro, s=s: g_replicate(s, r0["train"], ro["train"], ki))
        ctl.run(f"K-B3.2:s{s}", lambda rd=rd, r0=r0, s=s: g_kb32(banks[s]["B"], rd["train"], r0["train"], kw))
        ctl.run(f"TRAIN-LIST:s{s}", lambda rd=rd, r0=r0, s=s: g_train_list(banks[s]["B"], train_ids, rd["train"], r0["train"], kt))
    # ---------------- the T1 runs through the backend
    for (arm, s), rec in runs.items():
        tag = rec["tag"]
        rec["pack"] = pk = backend.analyze_t1(rec["entry"]["t1"], rec["entry"]["t1"].get("tag") or tag, s)
        for k, (ok, det) in pk.controls.items():
            ctl.add(f"{k}:{tag}", ok, det)
        ctl.run(f"T1-CONFIG:{tag}", lambda rec=rec, pk=pk: g_t1cfg(rec["tag"], rec["t1json"], rec["ckpt_tail"], "causal", pk.meta))
        lg = rec["entry"].get("t1_logged")
        if lg is not None:
            lpk = backend.analyze_t1_light(lg, lg.get("tag") or f"{tag}-logged", s)
            for k, (ok, det) in lpk.controls.items():
                ctl.add(f"{k}-logged:{tag}", ok, det)
            lj = jload(need(lg["json"], f"{tag} logged t1v8f json"))
            ctl.run(f"T1-CONFIG-logged:{tag}", lambda rec=rec, lpk=lpk, lj=lj: g_t1cfg(rec["tag"], lj, rec["ckpt_tail"], "logged", lpk.meta))
        # K4: tick 0 equals the arm's OWN T0 eval (at --ages fresh) to 1e-3
        ok, det = backend.k4(lg, pk, rec["entry"]["t0_fresh"], s)
        if not ok and lg is None and isinstance(det, dict):
            det = {**det, "hint": "K4 is a LOGGED-bucket identity (the T1-HEADS PREREG ran it on a --limit logged run): a --limit causal run's tick 0 reads another speed bucket "
                                  "than the T0 eval, so it differs by construction. Supply a 't1_logged' run for this arm in the manifest."}
        ctl.add(f"K4:{tag}", ok, det)
        d = rec["dump"]
        ctl.add(f"DUMP-WINDOWS:{tag}", bool(len(d["ws"]) == pk.N and np.array_equal(d["clip_index"], pk.ci) and np.array_equal(d["ws"], pk.t0)),
                {"dump_n": int(len(d["ws"])), "ticks_n": int(pk.N), "note": "the dump's (clip_index, ws) are the tick file's (ci, t0), window for window"})
        ctl.add(f"GRID-COUNTS:{tag}", bool(pk.N == cfg["n_windows"] and len(np.unique(d["eid"])) == cfg["n_episodes"] and len(rec["t0arr"]["ade6"]) == cfg["n_windows"]),
                {"windows": int(pk.N), "episodes": int(len(np.unique(d["eid"]))), "t0_windows": int(len(rec["t0arr"]["ade6"])), "committed": [cfg["n_windows"], cfg["n_episodes"]]})
        # R-2: the reader's ADE6 is the run's own analysis
        js = (rec["t1json"].get("paired_decision_grade") or {}).get("paired_model_minus_floor_t1", {}).get("ade_m")

        def r2(rec=rec, d=d, js=js):
            if js is None:
                return False, "the run json carries no paired_model_minus_floor_t1.ade_m to reproduce"
            A, B = K.comps(d["cl"], d["g"])["ade6_m"], K.comps(d["fl"], d["g"])["ade6_m"]
            mk = np.isfinite(A) & np.isfinite(B)
            r = T.PB(A[mk], B[mk], d["eid"][mk], n_boot=nb, seed=0)
            return abs(r["delta"] - float(js["delta"])) <= 1e-6, {"reader_delta": r["delta"], "analysis_delta": js["delta"]}
        ctl.run(f"R2-ADE6:{tag}", r2)
    # ---------------- window identity between the arms of a seed (the pairing is per window: a different window set is not a pair)
    for s in SEEDS:
        base = runs[("DAG0", s)]
        for arm in arms:
            if arm == "DAG0" or (arm, s) not in runs:
                continue
            o = runs[(arm, s)]
            same = (np.array_equal(o["pack"].ci, base["pack"].ci) and np.array_equal(o["pack"].t0, base["pack"].t0)
                    and np.array_equal(o["dump"]["clip_index"], base["dump"]["clip_index"]) and np.array_equal(o["dump"]["ws"], base["dump"]["ws"])
                    and np.array_equal(o["dump"]["eid"], base["dump"]["eid"]) and np.array_equal(o["t0arr"]["eid"], base["t0arr"]["eid"])
                    and np.array_equal(o["dump"]["g"], base["dump"]["g"]))
            ctl.add(f"SAME-WINDOWS:{arm}-vs-DAG0:s{s}", same, "ticks (ci, t0), dumps (clip_index, ws, eid, g) and T0 eid are identical window for window")
    # ---------------- stop: any failed control => REFUSED, no row is computed
    if ctl.failed():
        raise Refusal([f"{k}: {v['detail']}" for k, v in ctl.failed().items()])
    # ---------------- the rows
    rows, items, beside = {}, [], {"per_seed": {}}
    for s in SEEDS:
        o, z, r0 = runs[("DAG", s)], runs[("DAG0", s)], runs[("ORC0", s)]
        sd = {}
        for hd, nm in (("lat", "R-TLAT"), ("lon", "R-TLON")):
            r, (a, b, eid2, red) = f1_pair_row(T, o["pack"], z["pack"], hd, nb, matched=True)
            rf_row, (ra, _rb, reid, rred) = f1_pair_row(T, z["pack"], r0["pack"], hd, nb, matched=True)       # RF: DAG-0 vs its same-argv replicate ORC-0
            rf = abs(float(rf_row["delta"]))
            r.update(RF=rf, RF_unrounded=abs(rf_row["f1_a"] - rf_row["f1_b"]), RF_n_cells_matched=rf_row["n_cells_matched_pairs"],
                     RF_f1_dag0=rf_row["f1_a"], RF_f1_orc0=rf_row["f1_b"], bar=bar_for(hd, rf), bar_binding="2RF" if 2.0 * rf > TWO_F[hd] else "2F")
            r["clears"] = bool(r["better"] and r["delta"] >= r["bar"])
            sd[nm] = r
            items.append((f"{nm} s{s}", a, eid2, red))
            items.append((f"RF {nm} s{s}", ra, reid, rred))
            ro, _ = f1_pair_row(T, o["pack"], z["pack"], hd, nb, matched=False)
            sd[nm + "_own_cells_beside"] = ro
        A, B = K.comps(o["dump"]["cl"], o["dump"]["g"])["ade6_m"], K.comps(z["dump"]["cl"], z["dump"]["g"])["ade6_m"]
        sd["R-ADE"] = err_pair_row(T, f"R-ADE T1-kin ADE6 (DAG - DAG-0) s{s}", A, B, o["dump"]["eid"], nb)
        mk = np.isfinite(A) & np.isfinite(B)
        items.append((f"R-ADE s{s}", A[mk], o["dump"]["eid"][mk], "mean"))
        sd["R-T0"] = err_pair_row(T, f"R-T0 T0 own-code ADE6 (DAG - DAG-0) s{s}", o["t0arr"]["ade6"], z["t0arr"]["ade6"], o["t0arr"]["eid"], nb)
        sd["R-T0"]["bar_point_le"] = HALF_F
        items.append((f"R-T0 s{s}", o["t0arr"]["ade6"], o["t0arr"]["eid"], "mean"))
        rows[s] = sd
        bs = {"tick0_f1": {"DAG": o["pack"].tick0_f1, "DAG0": z["pack"].tick0_f1, "ORC0": r0["pack"].tick0_f1},
              "goal_macro_f1": {"DAG": o["pack"].goal, "DAG0": z["pack"].goal},
              "per_run_t1heads": {"DAG": o["pack"].extra, "DAG0": z["pack"].extra, "ORC0": r0["pack"].extra}}
        for ref, key in (("ORC", "DAG_minus_ORC"), ("M2b", "DAG_minus_M2b")):
            mb = runs.get((ref, s))
            if mb is None:
                continue
            bs[key] = {"tick0_f1": {ref: mb["pack"].tick0_f1}}
            for hd, nm in (("lat", "T-LAT-F1"), ("lon", "T-LON-F1")):
                bs[key][nm] = f1_pair_row(T, o["pack"], mb["pack"], hd, nb, matched=True)[0]
            Am = K.comps(mb["dump"]["cl"], mb["dump"]["g"])["ade6_m"]
            bs[key]["T1-kin ADE6"] = err_pair_row(T, f"DAG - {ref} T1-kin ADE6 s{s}", A, Am, o["dump"]["eid"], nb)
            bs[key]["T0 ADE6"] = err_pair_row(T, f"DAG - {ref} T0 ADE6 s{s}", o["t0arr"]["ade6"], mb["t0arr"]["ade6"], o["t0arr"]["eid"], nb)
            if ref == "M2b":
                bs["DAG0_minus_M2b"] = {"T1-kin ADE6": err_pair_row(T, f"DAG-0 - M2b T1-kin ADE6 s{s}", B, Am, o["dump"]["eid"], nb)}
        beside["per_seed"][f"s{s}"] = bs
    beside["not_computed"] = {"per-class turn F1 (TURN_L / TURN_R)": "no instrument in the reader", "four families at T0 and T1": "no instrument in the reader (STRATEGIC is NOT paused)",
                              "R76 crawl rows M1 / L2": "carried per run (crawl_fraction_ticks_ge1), not paired", "R-TLAT / R-TLON re-scored at section 2's tie rule": "needs the bank matcher",
                              "T1 off-route fraction at tick 2": "carried per run (off_route), not paired"}
    ok1, det1 = k1_own(T, items, nb)
    ctl.add("K1-DAG-ROWS", ok1, det1)
    if not ok1:
        raise Refusal([f"K1-DAG-ROWS: {det1}"])
    dec = decide({s: {"R-TLAT": rows[s]["R-TLAT"], "R-TLON": rows[s]["R-TLON"], "R-ADE": rows[s]["R-ADE"], "R-T0": rows[s]["R-T0"]} for s in SEEDS})
    return {"rows": {f"s{s}": rows[s] for s in SEEDS}, "OUTCOME": outcome_block(dec), "controls": ctl.items, "controls_PASS": ctl.flags(), "beside": beside,
            "bank": {f"s{s}": {"header": header_summary(banks[s]["B"]), "coverage": cov[s]} for s in SEEDS},
            "runs": {r["tag"]: {"t1_ticks": r["entry"]["t1"].get("ticks"), "t1_tag": r["entry"]["t1"].get("tag"), "ckpt_tail": r["ckpt_tail"],
                                "ticks_sha256": (sha256_file(r["entry"]["t1"]["ticks"]) if r["entry"]["t1"].get("ticks") and os.path.exists(r["entry"]["t1"]["ticks"]) else None)}
                     for r in runs.values()}}


# ---------------------------------------------------------------------------------------------------------------- CLI
def parse_kv(items, what):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not v or not k.strip().lstrip("s").isdigit():
            raise SystemExit(f"{what}: expected SEED=VALUE, got {it!r}")
        out[int(k.strip().lstrip("s"))] = v
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, help="runs.json: arms DAG / DAG0 / ORC0 (required) and ORC / M2b (optional) x seeds, each with t1{tag,ticks,dump_dir,json,k7}, [t1_logged], "
                    "t0{json,npz} (box 2's standard eval), t0_fresh{json,npz} (the --ages fresh eval K4 reads); DAG / DAG0 / ORC0 also train_dir; DAG / DAG0 also token. "
                    "bank.{0,1}: {report = the bank directory's PROVENANCE.json (bank.json and the shards are read beside it), [bank_file = bank.json elsewhere]}; "
                    "eval141_clip_ids, train_clip_ids: paths to JSON lists of clip-id strings (M2b's TRAIN list must hash to the trainer's train_sha256)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tool-root", required=True, help="the pinned export (stack + taniteval); READ-ONLY")
    ap.add_argument("--t1hd-read")
    ap.add_argument("--t1k-read")
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--eval-grid", required=True)
    ap.add_argument("--eval-join", required=True)
    ap.add_argument("--banked-dir", default=".", help="t1hd_read's banked dir (only the K7 'banked' mode reads it)")
    ap.add_argument("--m2b-ckpt", action="append", help="SEED=path of the M2b checkpoint the fine-tune started from (sha256 is taken here)")
    ap.add_argument("--m2b-sha", action="append", help="SEED=sha256 instead of a path (the chain's measured value)")
    ap.add_argument("--banked-sha", action="append", help="SEED=sha256: override the PREREG's banked M2b checkpoint hashes (TEST ONLY)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-windows", type=int, default=N_WINDOWS)
    ap.add_argument("--n-episodes", type=int, default=N_EPISODES)
    ap.add_argument("--key-dagger-n", default="dagger_n", help="the run total of replaced samples (named in the PREREG, section 6)")
    ap.add_argument("--key-dagger-eligible", default="dagger_eligible", help="the run total of eligible samples (named in the PREREG, section 6)")
    ap.add_argument("--key-dagger-built-p", default="dagger_p", help="PROVISIONAL: the built lever value in the run's levers record")
    ap.add_argument("--key-dagger-bank-sha", default="dagger_bank_sha256", help="PROVISIONAL: the built bank sha256 in the run's levers record")
    ap.add_argument("--key-init-sha", default="init_ckpt.sha256", help="dotted key of the init checkpoint's sha256 in the levers record")
    ap.add_argument("--key-window-sha", default="train_window_list_sha256", help="the trainer's TRAIN WINDOW-list sha256 (G.windows_sha256(J['clip'], wins)); v8f_mini_train.py records it on every run, in summary.json beside train_sha256 (:1663)")
    ap.add_argument("--key-train-sha", default="train_sha256", help="the trainer's TRAIN CLIP-list sha256 (v8f_mini_train.py records it under this key)")
    ap.add_argument("--prereg", help="the PREREG file: every outcome-table cell embedded here must occur verbatim in it")
    ap.add_argument("--work-dir")
    return ap


def main(argv=None, backend_factory=None):
    a = build_parser().parse_args(argv)
    t_start = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    sib = os.path.normpath(os.path.join(here, "..", "2026-10-26-t1hdread", "t1hd_read.py"))
    sib_k = os.path.normpath(os.path.join(here, "..", "2026-10-26-t1kin-hacc", "t1k_read.py"))
    p_hd, p_k = a.t1hd_read or sib, a.t1k_read or sib_k
    out_dir = os.path.dirname(os.path.abspath(a.out))
    work = a.work_dir or os.path.join(out_dir, "_dag_work")
    try:
        for p, w in ((p_hd, "t1hd_read.py"), (p_k, "t1k_read.py"), (a.manifest, "manifest")):
            need(p, f"{w} (pass --t1hd-read / --t1k-read)" if "read" in w else w)
        os.makedirs(work, exist_ok=True)
        T = load_t1hd(p_hd, a.tool_root)
        K = load_t1k(p_k, a.tool_root)
        if T.SEED_FLOOR != SEED_FLOOR:
            raise Refusal(f"t1hd_read's seed floor {T.SEED_FLOOR} is not the PREREG's {SEED_FLOOR}")
        manifest = load_manifest(a.manifest)
        m2b_sha = {s: v.lower() for s, v in parse_kv(a.m2b_sha, "--m2b-sha").items()}
        for s, p in parse_kv(a.m2b_ckpt, "--m2b-ckpt").items():
            m2b_sha.setdefault(s, sha256_file(need(p, f"M2b seed {s} checkpoint")))
        banked = {**BANKED_M2B_SHA, **{s: v.lower() for s, v in parse_kv(a.banked_sha, "--banked-sha").items()}}
        if backend_factory is None:
            ns = T.parse(["--mode", "real", "--tool-root", a.tool_root, "--eval-cache", a.eval_cache, "--tracks", a.tracks, "--eval-grid", a.eval_grid,
                          "--eval-join", a.eval_join, "--banked-dir", a.banked_dir, "--out", os.path.join(work, "t1hd_unused.json")])
            backend = RealBackend(T, ns, a.n_boot, work)
        else:
            backend = backend_factory(T, a)
        cfg = {"manifest": manifest, "m2b_sha": m2b_sha, "banked_sha": banked, "n_boot": a.n_boot, "n_windows": a.n_windows, "n_episodes": a.n_episodes,
               "keys": {"n": a.key_dagger_n, "eligible": a.key_dagger_eligible, "built_p": a.key_dagger_built_p, "bank_sha": a.key_dagger_bank_sha,
                        "init_sha": a.key_init_sha, "window_sha": a.key_window_sha, "train_sha": a.key_train_sha}, "prereg": a.prereg}
        res = read(cfg, backend, T, K)
    except Refusal as e:
        print("DAG READ REFUSED (nothing written):", file=sys.stderr)
        for m_ in e.msgs:
            print("  -", m_, file=sys.stderr)
        return 2
    except (OSError, KeyError, ValueError) as e:
        print(f"DAG READ INPUT ERROR (nothing written): REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    finally:
        if a.work_dir is None:
            shutil.rmtree(work, ignore_errors=True)
    out = {"prereg": PREREG_ID, "claim_row": "H-V8F-DAGGER-1", "tier": TIER, "dagger_p": DAG_P,
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0, over episodes; DAG - DAG-0 compares two NETWORKS per seed on the same windows "
                        "(F1 rows: the T0 eval's F1 reducer on the matched cells; ADE rows: per-window mean). It answers 'would another draw of EPISODES say this?'; "
                        "the training-seed question is answered by the two seeds plus the replicate floor RF = |F1(DAG-0) - F1(ORC-0)| (ORC-0 is a same-argv replicate of DAG-0, "
                        "trained by another build), which raises the bar to max(2F, 2*RF); the planner is deterministic, so no inference-seed question arises.",
           "evidence_class": "MEASURED (ours; artifact paths in `runs` and the manifest)",
           "bars": {"2F": TWO_F, "F_ade6_m": F_ADE, "F_over_2_t0_point_m": HALF_F, "seed_floor": SEED_FLOOR, "bar_rule": "bar(h, s) = max(2F_h, 2 * RF_h,s), inclusive",
                    "coverage_min": COVERAGE_MIN},
           "strategic_family": "NOT PAUSED for v8F (PI ruling 2026-10-17, R73): reported BESIDE as S-POS@30/50/80 vs CTRV per run (unpaired); decides nothing",
           "n_boot": a.n_boot, "n_windows": a.n_windows, "n_episodes": a.n_episodes,
           "reader_sha256": sha256_file(os.path.abspath(__file__)), "t1hd_read_sha256": sha256_file(p_hd), "t1k_read_sha256": sha256_file(p_k),
           "manifest_sha256": sha256_file(a.manifest), "m2b_ckpt_sha256": {f"s{s}": v for s, v in m2b_sha.items()}, "wall_s": time.time() - t_start}
    if a.n_boot != N_BOOT:
        out["LABEL"] = f"TEST: B = {a.n_boot} != the registered {N_BOOT}; NOT A READ"
    out.update(res)
    ordered = {k: out[k] for k in ("OUTCOME", "controls_PASS", "rows") if k in out}
    ordered.update({k: v for k, v in out.items() if k not in ordered})
    os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ordered, f, indent=1, default=jenc)
    sha = sha256_file(a.out)
    oc = ordered["OUTCOME"]
    print(json.dumps({"OUTCOME": oc["OUTCOME"], "matching_rows": oc["matching_rows_in_table_order"], "committed_next": oc["committed_next"]}, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}  sha256 {sha}")
    return {"UNPLACED": 3, "BLOCKED": 4}.get(oc["OUTCOME"], 0)


if __name__ == "__main__":
    sys.exit(main())
