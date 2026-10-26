#!/usr/bin/env python3
"""orc_read.py -- the reader for Project Steering/PREREG_V8F_ORECOVER.md (blob at 28e28c88) incl. Amendment 1 (A1-2026-10-26), applied as written.

    ORC (M2b fine-tuned 3,000 steps WITH the ego perturbation) vs ORC-0 (the same fine-tune with --ego-perturb-p 0), per seed s in {0, 1}:

      R-TLAT / R-TLON  the T head's lat / lon macro-F1 at ticks >= 1 of the T1-kin plain causal rollout, ORC_s - ORC-0_s, through
                       taniteval.ci.paired_episode_cluster_bootstrap (B 2000, seed 0, over episodes) with the T0 eval's F1 reducer, on the MATCHED
                       cells (scored in both runs).   better if > 0; bars 2F = 0.0248 / 0.0146
      R-ADE            T1-kin ADE6, ORC - ORC-0, paired per window.                       guard: not separated-worse
      R-T0             T0 own-code ADE6, ORC - ORC-0 point.                               guard: <= +F/2 = 0.0604 m

    Nothing here re-implements the programme's instruments: the T head cells, the K1-K7 controls, the F1 reducer and the row wrapper are
    `t1hd_read` (imported as a module, like `probe_lag.py` does); the per-window ADE6 is `t1k_read.comps` (the module is loaded through its own
    `--selftest` door, which runs its R-3 self-test and exits after defining `comps` / `paired`); the bootstrap is `taniteval.ci`.

    python orc_read.py --manifest runs.json --out ORC_READ.json --tool-root <stack+taniteval export> --eval-cache .. --tracks .. --eval-grid .. --eval-join ..
                       --m2b-ckpt 0=<M2b_s0/ckpt_last.pt> --m2b-ckpt 1=<M2b_s1/ckpt_last.pt> [--t1hd-read P --t1k-read P] [--key-perturb-n K --key-perturb-eligible K ..]

    rc 0 an outcome row (ROBUST / PARTIAL / NO EFFECT / HARMS) was written; rc 2 REFUSED (a control failed, or an input is missing): NOTHING is written;
    rc 3 UNPLACED (the reading matches no row of the PREREG's table; written, no committed next -- the table needs an amendment, never a guess).
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
STEPS, LR, WARMUP, PGT_START, PGT_END = 3000, 1e-4, 100, 0.2, 0.2          # Amendment 1
PERT_P = {"ORC": 0.3, "ORC0": 0.0}                                         # section 1 / Amendment 1 (named in every argv)
RATIO_CENTRE, RATIO_TOL = 0.3, 0.02                                        # section 3: 0.3 +- 0.02
SEED_FLOOR = {"lat": 0.0124, "lon": 0.0073}                                # section 2: the T0 fresh values
TWO_F = {"lat": 0.0248, "lon": 0.0146}                                     # 2F, committed
F_ADE = 0.1207                                                             # F, the ADE6 seed floor
HALF_F = 0.0604                                                            # F/2 as the PREREG commits it (R-T0 point bar)
N_BOOT = 2000
N_WINDOWS, N_EPISODES = 2399, 141                                          # eval-141
TIER = ("T1 (T1-kin: self-action OPEN loop, plain causal bucket, nav dead-reckoned; never 'closed loop') for R-TLAT / R-TLON / R-ADE; "
        "T0 (teacher-forced own-code, eval-141) for R-T0")
BANKED_M2B_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8",
                  1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
PREREG_ID = "Project Steering/PREREG_V8F_ORECOVER.md @28e28c88 + Amendment 1 (A1-2026-10-26) + Amendment 2 (A2-2026-10-26)"
# Amendment 2 replaces the NO EFFECT condition (the as-written table had a hole); the as-written table above stays verbatim, this is the condition in force.
A2_NO_EFFECT = "no F1 row separated-better and ≥ 2F at both seeds, guards held"

# The section-4 table, VERBATIM, in the PREREG's order (a test compares every cell with the file; markdown bold markers are markup, not text).
OUTCOME_TABLE = (
    ("ROBUST", "R-TLAT and R-TLON separated-better and ≥ 2F, at both seeds; R-ADE not separated-worse; R-T0 within +F/2",
     "adopt ORC as the base. Re-read T1 dominance over P10 and the FLAT control on it (FLAT's committed repeat)"),
    ("PARTIAL", "at least one F1 row separated-better and ≥ 2F at both seeds, the other not, with the guards held",
     "adopt for the head that moved. The next arm targets the other head's inputs, under its own read"),
    ("NO EFFECT", "no F1 row separated-better at both seeds, guards held",
     "perturbing the inputs does not teach own-state decisions. Next: on-policy states, i.e. the model's own T1-kin states supervised with the human's "
     "labels at the matched route point (DAgger form), under its own PREREG"),
    ("HARMS", "R-ADE separated-worse at ≥ 1 seed, or R-T0 above +F/2",
     "ORC is not adopted. Reported with the rows; next is NO EFFECT's arm at p = 0.1"),
    ("REFUSED", "any control fails", "fix and re-run"),
)
OUTCOME_NAMES = tuple(r[0] for r in OUTCOME_TABLE)
UNPLACED_TEXT = ("UNPLACED: this reading matches no row of the PREREG's section-4 table (the table has a hole: a head separated-better at both seeds but below 2F "
                 "at one of them is neither NO EFFECT nor PARTIAL). No outcome word is invented. The table needs an amendment from the PI / Master Mind -- "
                 "written WITHOUT looking at which way the numbers fall, so it cannot move a goalpost -- before this reading is quoted.")


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
    """argv -> {flag: [values-of-occurrence-1, values-of-occurrence-2, ...]}; EVERY value after a flag is kept (``--ego-perturb-vscale 0.5 1.0`` has two --
    the programme's ``v8f_pgt_read._flags`` keeps only the first, so it cannot tell ``0.5 1.0`` from ``0.5 9.9``)."""
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
                _load_module("t1k_read_for_orc", path)
            except SystemExit as e:
                if e.code not in (0, None):
                    raise
    finally:
        sys.argv = old
    m = sys.modules["t1k_read_for_orc"]
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
        stage = tempfile.mkdtemp(prefix="orc_k4_", dir=self.work)
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
PATH_KEYS = {"train_dir", "token", "metrics", "ticks", "dump_dir", "json", "npz", "ref_dump_dir", "banked_dir"}


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
        raise Refusal(f"manifest has no {arm} seed {seed}: the verdict needs BOTH seeds of ORC and ORC-0 (a lever claim on one seed is what H-ESTIM-SEED-1 forbids)")
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
    """The registered fine-tune, NAMED on the argv (the flags' defaults disable the lever, so a default is not a test of it)."""
    bad, fl = [], r["flags"]
    want = {"--steps": STEPS, "--lr": LR, "--warmup": WARMUP, "--p-gt-start": PGT_START, "--p-gt-end": PGT_END, "--ego-perturb-p": PERT_P[arm]}
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


def g_pert(tag, arm, r, kn, ke, kp):
    """The perturbation REACHED training: built p == argv p; ORC n / eligible in 0.3 +- 0.02; ORC-0 n exactly 0."""
    bad, det = [], {}
    bp = find_key(r["sources"], kp)
    if not bp:
        bad.append(f"no built value '{kp}' recorded (the lever's built_levers entry)")
    for w, v in bp:
        if not same_float(v, PERT_P[arm]):
            bad.append(f"{w} = {v}, argv names {PERT_P[arm]}")
    ns, es = find_key(r["sources"], kn), find_key(r["sources"], ke)
    if not ns:
        bad.append(f"no '{kn}' recorded by the trainer (the totals are a build item)")
    if len({json.dumps(v) for _, v in ns}) > 1:
        bad.append(f"'{kn}' recorded with conflicting values {ns}")
    if len({json.dumps(v) for _, v in es}) > 1:
        bad.append(f"'{ke}' recorded with conflicting values {es}")
    if ns:
        n = ns[0][1]
        det["ego_perturb_n"] = n
        if arm == "ORC0":
            if not (isinstance(n, (int, float)) and n == 0):
                bad.append(f"ORC-0 {kn} = {n}, must be exactly 0")
        else:
            if not es:
                bad.append(f"no '{ke}' recorded")
            else:
                e = es[0][1]
                det["ego_perturb_eligible"] = e
                if not (isinstance(n, (int, float)) and isinstance(e, (int, float)) and e > 0 and 0 <= n <= e):
                    bad.append(f"{kn} {n} / {ke} {e}: not a count pair with 0 <= n <= eligible, eligible > 0")
                else:
                    ratio = n / e
                    det["ratio"] = ratio
                    if not (RATIO_CENTRE - RATIO_TOL - 1e-12 <= ratio <= RATIO_CENTRE + RATIO_TOL + 1e-12):
                        bad.append(f"ORC perturbed fraction {n}/{e} = {ratio:.4f} outside {RATIO_CENTRE} +- {RATIO_TOL}")
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


def g_pair(seed, ro, r0, to, t0_, key):
    """ORC vs ORC-0 of one seed: argv identical except --ego-perturb-p (and --out); same init, data, steps."""
    bad = []
    drop = {"--out", "--ego-perturb-p"}
    fo = {k: v for k, v in ro["flags"].items() if k not in drop}
    f0 = {k: v for k, v in r0["flags"].items() if k not in drop}
    if fo != f0:
        diff = sorted(k for k in set(fo) | set(f0) if fo.get(k) != f0.get(k))
        bad.append(f"argv differs in {diff} besides --ego-perturb-p and --out -- not one lever")
    so, s0 = ro["summary.json"], r0["summary.json"]
    for k in ("steps", "n_clips", "n_windows", "train_sha256", "excluded_sha256", "seed"):
        if so.get(k) != s0.get(k):
            bad.append(f"summary {k}: ORC {so.get(k)} vs ORC-0 {s0.get(k)}")
    io_, i0 = {str(v) for _, v in init_sha_values(ro, key)}, {str(v) for _, v in init_sha_values(r0, key)}
    if io_ != i0 or not io_:
        bad.append(f"init checkpoint sha256: ORC {sorted(io_)} vs ORC-0 {sorted(i0)}")
    do, d0 = (to.get("binding") or {}).get("data_sha256"), (t0_.get("binding") or {}).get("data_sha256")
    if do != d0:
        bad.append("launch-gate data manifests differ between ORC and ORC-0 (not the same data)")
    return not bad, bad or "same init, steps, data, seed; argv identical except --ego-perturb-p"


# ---------------------------------------------------------------------------------------------------------------- T1 / T0 file controls
def load_dump(dump_dir):
    import glob
    fs = sorted(glob.glob(os.path.join(need(dump_dir, "dump dir"), "ep*.npz")))
    if not fs:
        raise Refusal(f"no ep*.npz in {dump_dir}")
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "fl", "eid", "clip_index", "ws")}


def g_t1cfg(label, hdr, ckpt_tail, want_limit, ticks_meta):
    """The run is the registered rollout: plain causal bucket, m = 10, periodic, logged scene, dead-reckoned nav, heads dumped; it scored THIS arm's checkpoint."""
    h = hdr.get("t1v8f") or {}
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
def f1_pair_row(T, pa, pb, hd, n_boot, matched=True):
    """R-TLAT / R-TLON: macro-F1 at ticks >= 1, ``pa`` - ``pb`` (higher is better), the T1-HEADS slot layout (2 cells per window) and reducer. Matched: only cells
    scored in BOTH runs enter either side (the PREREG's 'matched cells'); unmatched: each run's own scored cells."""
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


# ---------------------------------------------------------------------------------------------------------------- the outcome (PREREG section 4)
def decide(rows):
    """``rows[seed]`` = {"R-TLAT": row, "R-TLON": row, "R-ADE": row, "R-T0": {"point": float}} with delta / separated per row. Returns the outcome, EVERY
    matching row in table order, and the per-seed flags. REFUSED is not decided here (a failed control stops the read before any row exists)."""
    if tuple(sorted(rows)) != tuple(SEEDS):
        raise Refusal(f"the outcome needs exactly seeds {SEEDS}, got {sorted(rows)}")
    hd_row = {"lat": "R-TLAT", "lon": "R-TLON"}

    def sep_better(s, hd):
        r = rows[s][hd_row[hd]]
        return bool(r["separated"] and r["delta"] > 0)

    def moved(s, hd):                                               # separated-better AND >= 2F
        r = rows[s][hd_row[hd]]
        return bool(sep_better(s, hd) and r["delta"] >= TWO_F[hd])

    ade_worse = {s: bool(rows[s]["R-ADE"]["separated"] and rows[s]["R-ADE"]["delta"] > 0) for s in SEEDS}
    t0_above = {s: bool(rows[s]["R-T0"]["point"] > HALF_F) for s in SEEDS}
    guards_held = not any(ade_worse.values()) and not any(t0_above.values())
    moved_both = {hd: all(moved(s, hd) for s in SEEDS) for hd in ("lat", "lon")}
    sep_both = {hd: all(sep_better(s, hd) for s in SEEDS) for hd in ("lat", "lon")}
    match = {
        "ROBUST": bool(moved_both["lat"] and moved_both["lon"] and guards_held),
        "PARTIAL": bool((moved_both["lat"] != moved_both["lon"]) and guards_held),
        "NO EFFECT": bool(not moved_both["lat"] and not moved_both["lon"] and guards_held),        # A2-2026-10-26 (as written: not sep_both)
        "HARMS": bool(any(ade_worse.values()) or any(t0_above.values())),
    }
    matching = [n for n in OUTCOME_NAMES if match.get(n)]
    flags = {"separated_better_and_ge_2F_per_seed": {hd: {f"s{s}": moved(s, hd) for s in SEEDS} for hd in ("lat", "lon")},
             "separated_better_per_seed": {hd: {f"s{s}": sep_better(s, hd) for s in SEEDS} for hd in ("lat", "lon")},
             "R-ADE_separated_worse_per_seed": {f"s{s}": ade_worse[s] for s in SEEDS}, "R-T0_above_F_over_2_per_seed": {f"s{s}": t0_above[s] for s in SEEDS},
             "guards_held": guards_held, "head_moved_at_both_seeds": moved_both,
             "sub_bar_separated_better_at_both_seeds": {hd: bool(sep_both[hd] and not moved_both[hd]) for hd in ("lat", "lon")}}
    out = matching[0] if matching else "UNPLACED"
    return {"OUTCOME": out, "matching_rows_in_table_order": matching, "flags": flags}


def outcome_block(d):
    out = d["OUTCOME"]
    if out == "UNPLACED":
        return {**d, "condition": None, "committed_next": None, "note": UNPLACED_TEXT}
    cond, nxt = next((c, n) for nm, c, n in OUTCOME_TABLE if nm == out)
    if out == "NO EFFECT":                                          # A2-2026-10-26: the condition in force; the as-written one is carried beside it
        return {**d, "condition": A2_NO_EFFECT, "condition_as_written": cond, "condition_amended_by": "A2-2026-10-26", "committed_next": nxt}
    return {**d, "condition": cond, "committed_next": nxt}


def check_prereg_text(path):
    """Optional: every embedded table cell occurs verbatim in the PREREG file (markdown bold markers stripped)."""
    t = open(path, encoding="utf-8").read().replace("**", "")
    miss = [f"{nm}: {x}" for nm, c, n in OUTCOME_TABLE for x in (nm, c, n) if x not in t]
    return not miss, miss or "every outcome-table cell is verbatim in the PREREG file"


# ---------------------------------------------------------------------------------------------------------------- the read
def sources_eq(vals):
    return len({json.dumps(v, sort_keys=True, default=str) for v in vals}) == 1


def read(cfg, backend, T, K):
    """cfg: manifest (resolved), m2b_sha {seed: hex}, banked_sha, keys, n_boot, n_windows, n_episodes. Returns the result dict; raises Refusal(msgs)."""
    ctl = Ctl()
    m = cfg["manifest"]
    nb = cfg["n_boot"]
    kn, ke, kp, ki = cfg["key_n"], cfg["key_eligible"], cfg["key_built_p"], cfg["key_init_sha"]
    if sorted(int(s) for s in m.get("seeds", SEEDS)) != list(SEEDS):
        ctl.add("SEEDS", False, f"manifest seeds {m.get('seeds')}, the read needs {list(SEEDS)}")
    if cfg.get("prereg"):
        ctl.run("PREREG-TEXT", lambda: check_prereg_text(cfg["prereg"]))
    for s in SEEDS:
        ctl.run(f"M2B-CKPT:s{s}", lambda s=s: (cfg["m2b_sha"].get(s) == cfg["banked_sha"].get(s) and cfg["m2b_sha"].get(s) is not None,
                                              {"passed_on_cli": str(cfg["m2b_sha"].get(s))[:16], "banked_PREREG": str(cfg["banked_sha"].get(s))[:16]}))
    runs = {}                                                       # (arm, seed) -> dict
    # ---------------- the files: training records, tokens, T1 json, dumps, T0
    arms = ["ORC", "ORC0"] + (["M2b"] if "M2b" in m["arms"] else [])
    for arm in arms:
        for s in SEEDS:
            if arm == "M2b" and str(s) not in (m["arms"]["M2b"] or {}):
                continue
            e = entry(m, arm, s)
            tag = f"{arm}_s{s}"
            rec = {"arm": arm, "seed": s, "tag": tag, "entry": e}
            if arm in ("ORC", "ORC0"):
                rec["train"] = train_records(e, ki)
                rec["token"] = read_token(e.get("token"))
                tr = rec["train"]
                ctl.run(f"ARGV:{tag}", lambda tr=tr, arm=arm, s=s, tag=tag: g_argv(tag, arm, s, tr))
                ctl.run(f"SUMMARY:{tag}", lambda tr=tr, s=s, tag=tag: g_summary(tag, s, tr))
                ctl.run(f"INIT-CKPT:{tag}", lambda tr=tr, s=s, tag=tag: g_init(tag, s, tr, cfg["m2b_sha"].get(s, ""), ki))
                ctl.run(f"PERTURB-REACHED:{tag}", lambda tr=tr, arm=arm, tag=tag: g_pert(tag, arm, tr, kn, ke, kp))
                ctl.run(f"GATE-TOKEN:{tag}", lambda tr=tr, rc=rec, tag=tag: g_token(tag, tr, rc["token"]))
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
            runs[(arm, s)] = rec
    # ---------------- pair controls (ORC vs ORC-0)
    for s in SEEDS:
        ro, r0 = runs[("ORC", s)], runs[("ORC0", s)]
        ctl.run(f"PAIR:s{s}", lambda ro=ro, r0=r0, s=s: g_pair(s, ro["train"], r0["train"], ro["token"], r0["token"], ki))
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
        # K4 (replaced for the fine-tuned arms): tick 0 equals the arm's OWN T0 eval to 1e-3
        ok, det = backend.k4(lg, pk, rec["entry"]["t0"], s)
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
        base = runs[("ORC0", s)]
        for arm in arms:
            if arm == "ORC0" or (arm, s) not in runs:
                continue
            o = runs[(arm, s)]
            same = (np.array_equal(o["pack"].ci, base["pack"].ci) and np.array_equal(o["pack"].t0, base["pack"].t0)
                    and np.array_equal(o["dump"]["clip_index"], base["dump"]["clip_index"]) and np.array_equal(o["dump"]["ws"], base["dump"]["ws"])
                    and np.array_equal(o["dump"]["eid"], base["dump"]["eid"]) and np.array_equal(o["t0arr"]["eid"], base["t0arr"]["eid"])
                    and np.array_equal(o["dump"]["g"], base["dump"]["g"]))
            ctl.add(f"SAME-WINDOWS:{arm}-vs-ORC0:s{s}", same, "ticks (ci, t0), dumps (clip_index, ws, eid, g) and T0 eid are identical window for window")
    # ---------------- stop: any failed control => REFUSED, no row is computed
    if ctl.failed():
        raise Refusal([f"{k}: {v['detail']}" for k, v in ctl.failed().items()])
    # ---------------- the rows
    rows, items, beside = {}, [], {"per_seed": {}}
    for s in SEEDS:
        o, z = runs[("ORC", s)], runs[("ORC0", s)]
        sd = {}
        for hd, nm in (("lat", "R-TLAT"), ("lon", "R-TLON")):
            r, (a, b, eid2, red) = f1_pair_row(T, o["pack"], z["pack"], hd, nb, matched=True)
            sd[nm] = r
            items.append((f"{nm} s{s}", a, eid2, red))
            ro, _ = f1_pair_row(T, o["pack"], z["pack"], hd, nb, matched=False)
            sd[nm + "_own_cells_beside"] = ro
        A, B = K.comps(o["dump"]["cl"], o["dump"]["g"])["ade6_m"], K.comps(z["dump"]["cl"], z["dump"]["g"])["ade6_m"]
        sd["R-ADE"] = err_pair_row(T, f"R-ADE T1-kin ADE6 (ORC - ORC-0) s{s}", A, B, o["dump"]["eid"], nb)
        mk = np.isfinite(A) & np.isfinite(B)
        items.append((f"R-ADE s{s}", A[mk], o["dump"]["eid"][mk], "mean"))
        sd["R-T0"] = err_pair_row(T, f"R-T0 T0 own-code ADE6 (ORC - ORC-0) s{s}", o["t0arr"]["ade6"], z["t0arr"]["ade6"], o["t0arr"]["eid"], nb)
        sd["R-T0"]["bar_point_le"] = HALF_F
        items.append((f"R-T0 s{s}", o["t0arr"]["ade6"], o["t0arr"]["eid"], "mean"))
        rows[s] = sd
        bs = {"tick0_f1": {"ORC": o["pack"].tick0_f1, "ORC0": z["pack"].tick0_f1},
              "goal_macro_f1": {"ORC": o["pack"].goal, "ORC0": z["pack"].goal},
              "per_run_t1heads": {"ORC": o["pack"].extra, "ORC0": z["pack"].extra}}
        mb = runs.get(("M2b", s))
        if mb is not None:
            bs["ORC_minus_M2b"] = {"tick0_f1": {"M2b": mb["pack"].tick0_f1}}
            for hd, nm in (("lat", "T-LAT-F1"), ("lon", "T-LON-F1")):
                bs["ORC_minus_M2b"][nm] = f1_pair_row(T, o["pack"], mb["pack"], hd, nb, matched=True)[0]
            Am = K.comps(mb["dump"]["cl"], mb["dump"]["g"])["ade6_m"]
            bs["ORC_minus_M2b"]["T1-kin ADE6"] = err_pair_row(T, f"ORC - M2b T1-kin ADE6 s{s}", A, Am, o["dump"]["eid"], nb)
            bs["ORC_minus_M2b"]["T0 ADE6"] = err_pair_row(T, f"ORC - M2b T0 ADE6 s{s}", o["t0arr"]["ade6"], mb["t0arr"]["ade6"], o["t0arr"]["eid"], nb)
            bs["ORC0_minus_M2b"] = {"T1-kin ADE6": err_pair_row(T, f"ORC-0 - M2b T1-kin ADE6 s{s}", B, Am, o["dump"]["eid"], nb)}
        beside["per_seed"][f"s{s}"] = bs
    ok1, det1 = k1_own(T, items, nb)
    ctl.add("K1-ORC-ROWS", ok1, det1)
    if not ok1:
        raise Refusal([f"K1-ORC-ROWS: {det1}"])
    dec = decide({s: {"R-TLAT": rows[s]["R-TLAT"], "R-TLON": rows[s]["R-TLON"], "R-ADE": rows[s]["R-ADE"], "R-T0": rows[s]["R-T0"]} for s in SEEDS})
    return {"rows": {f"s{s}": rows[s] for s in SEEDS}, "OUTCOME": outcome_block(dec), "controls": ctl.items, "controls_PASS": ctl.flags(), "beside": beside,
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
    ap.add_argument("--manifest", required=True, help="runs.json: arms ORC / ORC0 (and optional M2b) x seeds, each with train_dir, token, t1{tag,ticks,dump_dir,json,k7}, [t1_logged], t0{json,npz}")
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
    ap.add_argument("--key-perturb-n", default="ego_perturb_n")
    ap.add_argument("--key-perturb-eligible", default="ego_perturb_eligible")
    ap.add_argument("--key-built-p", default="ego_perturb_p", help="the built lever value in the run's levers record")
    ap.add_argument("--key-init-sha", default="init_ckpt.sha256", help="dotted key of the init checkpoint's sha256 in the levers record")
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
    work = a.work_dir or os.path.join(out_dir, "_orc_work")
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
               "key_n": a.key_perturb_n, "key_eligible": a.key_perturb_eligible, "key_built_p": a.key_built_p, "key_init_sha": a.key_init_sha, "prereg": a.prereg}
        res = read(cfg, backend, T, K)
    except Refusal as e:
        print("ORC READ REFUSED (nothing written):", file=sys.stderr)
        for m_ in e.msgs:
            print("  -", m_, file=sys.stderr)
        return 2
    except (OSError, KeyError, ValueError) as e:
        print(f"ORC READ INPUT ERROR (nothing written): REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    finally:
        if a.work_dir is None:
            shutil.rmtree(work, ignore_errors=True)
    out = {"prereg": PREREG_ID, "claim_row": "H-V8F-ORECOVER-1", "tier": TIER,
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0, over episodes; ORC - ORC-0 compares two NETWORKS per seed on the same windows "
                        "(F1 rows: the T0 eval's F1 reducer on the matched cells; ADE rows: per-window mean). It answers 'would another draw of EPISODES say this?'; the training-seed "
                        "question is answered only by the two seeds (H-ESTIM-SEED-1); a single ORC-0 per seed is the control for '3,000 more steps', not a replicate of the lever's noise.",
           "evidence_class": "MEASURED (ours; artifact paths in `runs` and the manifest)",
           "bars": {"2F": TWO_F, "F_ade6_m": F_ADE, "F_over_2_t0_point_m": HALF_F, "seed_floor": SEED_FLOOR},
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
    return 3 if oc["OUTCOME"] == "UNPLACED" else 0


if __name__ == "__main__":
    sys.exit(main())
