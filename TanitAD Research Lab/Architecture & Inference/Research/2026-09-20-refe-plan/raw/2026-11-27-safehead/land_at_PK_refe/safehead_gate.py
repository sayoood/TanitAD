#!/usr/bin/env python3
"""L2 SAFE-HEAD launch-gate profiles, refusals and checks (PREREG_SAFEHEAD.md 7; CLAUDE.md "NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS", PI 2026-09-26).

Registered into refe/launch_gate.py's PROFILES as NEW dicts (no existing profile moves), routed by `runner == "safehead"`:
  refe_safehead_s0   arm SH, seed 0 (the DECISION arm)          refe_safehead_s1   arm SH, seed 1 (the replicate R)
  refe_safehead_red  CTRL-RED: SH's recipe, seed 0, labels permuted across train tokens (--permute-labels-seed 20261127)
Checks: G-HYG, G-DVB, G-LIVE, G-EVAL, G-CKPT, G-SUITE (G-CLOCK: NA, a fixed sealed label cache has no label clock).  Each has a deliberate-regression arm
(TANITAD_GATE_ARM) that must make the gate FAIL (refe/test_safehead.py proves every one):
  G-HYG   sh_hyg_cfg_guard_removed (the config accepts an undeclared attribute) | sh_hyg_undeclared_read (the trainer reads a namespace attribute its parser lacks)
  G-DVB   sh_dvb_lr_drift (the optimiser is built at 2x the argv lr)          | sh_dvb_new_flag (a parser flag with no G-DVB entry)
  G-LIVE  sh_live_term_dropped (the TTC term is never emitted) | sh_reach_row_dropped (a trained row is not in the optimiser) | sh_reach_ep_in_loss (EP enters the loss:
          the frozen-row gradient control reads non-zero)
  G-EVAL  sh_eval_rowshift (the read loader applies the change to the wrong rows)
  G-CKPT  sh_ckpt_bias_dropped (the saved file is not the in-memory head)    | sh_ckpt_nondet (two identical runs differ)
  G-SUITE suite_failing_test (gate_checks.check_suite's own arm)

    python launch_gate.py refusals --profile refe_safehead_s0 --args "$ARGS"
    python launch_gate.py run --profile refe_safehead_s0 --pkg <pkg> --args "$ARGS" --out-dir <gate dir> --commit <landed 40-hex>
    python launch_gate.py verify --profile refe_safehead_s0 --token <PASS_*.json> --args "$ARGS" --pkg <pkg>
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import gate_checks as C  # noqa: E402
import safehead_data as D  # noqa: E402

C1_SHA256 = "f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4"     # MODEL_REGISTRY 14.6 C1 = launch_gate.SCORERFT_BASE_SHA256 (asserted in the test)
DATA_FLAGS = ("--cache-index", "--init-from", "--heldout-logs", "--navtest-logs")
OUTPUT_FLAGS = ("--out",)
CONTROL_FLAGS = ("--smoke-steps", "--build-only")
REGISTERED = {"--steps": "2000", "--lr": "0.0003", "--batch": "32", "--gamma": "2.0", "--rows": "0,1,3,5", "--save-steps": "0,250,500,1000,2000"}
RED_PERM_SEED = 20261127
ARMS = {"s0": {"seed": 0, "perm": None}, "s1": {"seed": 1, "perm": None}, "red": {"seed": 0, "perm": RED_PERM_SEED}}
PROFILE_NAME = {"s0": "refe_safehead_s0", "s1": "refe_safehead_s1", "red": "refe_safehead_red"}
CHECKS = ("G-HYG", "G-DVB", "G-LIVE", "G-EVAL", "G-CKPT", "G-SUITE")
LIVE_TERMS = ("qfl_nc", "qfl_dac", "qfl_ttc", "qfl_ddc")
SMOKE_STEPS = 5
EVAL_TOL = 1e-4
SUITE = [{"script": "test_safehead.py", "args": ["--fast"], "token": "ZZSAFEHEAD_PASS"},
         {"script": "test_launch_gate.py", "args": ["--static", "--no-banked"], "token": "ZZLG_STATIC_PASS"}]
# G-DVB: EVERY parser dest of safehead_train.py and how it is judged.  A dest that is not here is REFUSED (a new lever must arrive with its check).
#   built   = the argv value must equal what the trainer BUILT (built.json)     data = bound by sha256 in the token's data manifest (+ the pins below)
#   output  = where it writes (fresh)                                          control = gate-only switch, FORBIDDEN in a launch argv
DVB = {"cache_index": "data", "init_from": "data", "heldout_logs": "data", "navtest_logs": "data", "out": "output", "seed": "built", "steps": "built",
       "lr": "built", "batch": "built", "gamma": "built", "rows": "built", "save_steps": "built", "permute_labels_seed": "built",
       "smoke_steps": "control", "build_only": "control"}
REGISTERED_BUILT = {"optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "loss": "qfocal"}


def profiles() -> dict:
    out = {}
    for arm, spec in ARMS.items():
        req = {**{f: None for f in DATA_FLAGS + OUTPUT_FLAGS}, **REGISTERED, "--seed": str(spec["seed"])}
        if spec["perm"] is not None:
            req["--permute-labels-seed"] = str(spec["perm"])
        name = PROFILE_NAME[arm]
        out[name] = {"name": name, "runner": "safehead", "safehead_rules": True, "safehead_arm": arm, "trainer": "refe/safehead_train.py",
                     "parser_module": "safehead_train", "trainer_src": "safehead_train.py", "namespace_funcs": ("build_config", "run"),
                     "live_terms": LIVE_TERMS, "data_flags": DATA_FLAGS, "output_flags": OUTPUT_FLAGS,
                     "forbidden_flags": CONTROL_FLAGS + (() if spec["perm"] is not None else ("--permute-labels-seed",)),
                     "required_flags": req, "suite": [dict(e) for e in SUITE], "checks": CHECKS, "na_ok": ("G-CLOCK",),
                     "clock_na": "SAFE-HEAD has no label clock: its labels are a fixed, sealed cache bound by sha256 (--cache-index)",
                     "mintable": True, "require_commit": True, "rehearsal": False, "init_from_sha256": C1_SHA256, "smoke_steps": SMOKE_STEPS,
                     "card": "PK/raw/2026-11-27-safehead/PREREG_SAFEHEAD.md"}
    return out


def profile_name_for_argv(argv: list) -> str:
    import gate_token as GT
    perm = GT.flag_values(argv, "--permute-labels-seed")
    if perm and int(perm[0]) >= 0:
        return PROFILE_NAME["red"]
    seed = GT.flag_values(argv, "--seed")
    for arm, spec in ARMS.items():
        if spec["perm"] is None and seed and int(seed[0]) == spec["seed"]:
            return PROFILE_NAME[arm]
    raise SystemExit(f"REFUSED: the argv is no registered SAFE-HEAD arm (seed {seed}, permute {perm})")


def refusals(prof: dict, argv: list) -> list:
    """What `required_flags` cannot say: the EXACT flag set (no extra lever, none twice), every parser dest covered by G-DVB, a fresh --out."""
    import gate_token as GT
    T = importlib.import_module(prof.get("parser_module", "safehead_train"))
    parser = T.build_parser()
    dests = {x.dest for x in parser._actions if x.dest != "help"}
    out, name = [], prof["name"]
    no_dvb = sorted(dests - set(DVB))
    if no_dvb:
        out.append(f"{name}: parser flag(s) {no_dvb} have NO G-DVB entry (safehead_gate.DVB): a new lever without its check is refused")
    if argv and not str(argv[0]).startswith("--"):
        out.append(f"{name}: the argv starts with {argv[0]!r}, which belongs to no flag")
    seen: dict = {}
    for f, v in GT.flag_pairs(argv):
        seen[f] = seen.get(f, 0) + 1
    want = set(prof["required_flags"])
    for f in sorted(set(seen) - want):
        out.append(f"{name}: the argv carries {f}, which the arm does not register (its argv is exactly {sorted(want)})")
    for f, n in sorted(seen.items()):
        if n > 1:
            out.append(f"{name}: {f} is carried {n} times")
    for f in ("--rows", "--save-steps"):
        got = GT.flag_values(argv, f)
        if got and got[0] != REGISTERED[f]:
            out.append(f"{name} PINS {f} {REGISTERED[f]}; the argv has {f} {got[0]}")
    o = GT.flag_values(argv, "--out")
    if o and Path(o[0]).is_dir() and any(Path(o[0]).iterdir()):
        out.append(f"{name}: --out {o[0]} exists and is not empty (a run never writes over another)")
    return out


# ============================================================================================ the checks
def _trainer_env(arm) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")
    env.pop(C.ARM_ENV, None)
    env.pop("TANITAD_SH_GATE_TOKEN", None)
    if arm:
        env[C.ARM_ENV] = arm
    return env


def _derived(argv: list, out: Path, extra: list, drop=("--out", "--save-steps")) -> list:
    import gate_token as GT
    keep = []
    for f, v in GT.flag_pairs(argv):
        if f not in drop:
            keep += [f, *v]
    return keep + ["--out", str(out), *extra]


def _run_trainer(argv: list, arm) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(HERE / "safehead_train.py"), *argv], cwd=str(HERE), env=_trainer_env(arm), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def _tail(r) -> str:
    return ((r.stdout or "")[-400:] + " | " + (r.stderr or "")[-2000:]).replace("\n", " ")      # the evidence keeps the whole reason; the console line is cut at 200


def check_hyg(prof: dict, argv: list, arm=None, rehearsal=False) -> dict:
    T = importlib.import_module(prof["parser_module"])
    reasons, det = [], {}
    ns = T.build_parser().parse_args(argv)
    cls = T.SafeHeadConfig
    if C.arm_is("sh_hyg_cfg_guard_removed", arm):
        cls = dataclasses.make_dataclass("SafeHeadConfigUnguarded", [(f.name, f.type) for f in dataclasses.fields(T.SafeHeadConfig)])
    cfg = cls(**dataclasses.asdict(T.build_config(ns)))
    try:
        cfg._g_hyg_probe_undeclared = 1
        reasons.append("SafeHeadConfig ACCEPTED an undeclared attribute (the slots guard is absent)")
    except AttributeError:
        det["cfg_refuses_undeclared"] = True
    src = (HERE / prof["trainer_src"]).read_text(encoding="utf-8")
    if C.arm_is("sh_hyg_undeclared_read", arm):
        src += "\n\ndef build_config(a):\n    return a.undeclared_lever_g_hyg\n"
    dests = {x.dest for x in T.build_parser()._actions if x.dest != "help"}
    reads = C.namespace_reads(src, prof["namespace_funcs"])
    det["namespace_reads"] = sorted(reads)
    if reads - dests:
        reasons.append(f"the trainer reads namespace attributes its parser does not declare: {sorted(reads - dests)}")
    fields = {f.name for f in dataclasses.fields(T.SafeHeadConfig)}
    if fields != dests:
        reasons.append(f"config fields != parser dests: only-config {sorted(fields - dests)}, only-parser {sorted(dests - fields)}")
    return C.verdict("G-HYG", reasons, det, arm=arm, rehearsal=rehearsal)


def check_dvb(prof: dict, argv: list, work: Path, data: dict, arm=None, rehearsal=False) -> dict:
    T = importlib.import_module(prof["parser_module"])
    reasons, det = [], {}
    parser = T.build_parser()
    dests = {x.dest for x in parser._actions if x.dest != "help"}
    if C.arm_is("sh_dvb_new_flag", arm):
        dests.add("undeclared_new_lever")
    if dests - set(DVB):
        reasons.append(f"parser flag(s) {sorted(dests - set(DVB))} have no G-DVB entry: REFUSED (a new lever arrives with its check)")
    ns = parser.parse_args(argv)
    r = _run_trainer(_derived(argv, work / "build", ["--build-only"], drop=("--out",)), arm)
    det["build_rc"] = r.returncode
    bj = work / "build" / "built.json"
    if r.returncode != 0 or "ZZSH_BUILD_OK" not in r.stdout or not bj.is_file():
        reasons.append(f"the BUILD run failed (rc {r.returncode}): {_tail(r)}")
        return C.verdict("G-DVB", reasons, det, arm=arm, rehearsal=rehearsal)
    b = json.loads(bj.read_text(encoding="utf-8"))
    det["built"] = {k: b.get(k) for k in ("seed", "steps", "lr", "batch", "gamma", "rows", "save_steps", "permute_labels_seed", "optimizer", "loss")}
    want = {"seed": ns.seed, "steps": ns.steps, "batch": ns.batch, "gamma": ns.gamma, "permute_labels_seed": ns.permute_labels_seed,
            "save_steps": list(T._ints(ns.save_steps))}
    for k, v in want.items():
        if b.get(k) != v:
            reasons.append(f"G-DVB {k}: argv {v!r} != built {b.get(k)!r}")
    if not b.get("lr") or any(abs(x - ns.lr) > 0.0 for x in b["lr"]):
        reasons.append(f"G-DVB lr: argv {ns.lr!r} != the optimiser's {b.get('lr')!r}")
    rows = list(T._ints(ns.rows))
    if b.get("rows") != rows or b.get("rows_declared") != rows:
        reasons.append(f"G-DVB rows: argv {rows} != optimiser groups {b.get('rows')} / declared {b.get('rows_declared')}")
    if b.get("frozen_rows") != [r for r in range(D.N_OUT) if r not in rows]:
        reasons.append(f"G-DVB frozen rows {b.get('frozen_rows')} are not the complement of {rows}")
    if b.get("optimizer") != REGISTERED_BUILT["optimizer"] or b.get("loss") != REGISTERED_BUILT["loss"]:
        reasons.append(f"G-DVB optimiser/loss {b.get('optimizer')}/{b.get('loss')} != registered {REGISTERED_BUILT['optimizer']}/{REGISTERED_BUILT['loss']}")
    if any(list(x) != REGISTERED_BUILT["betas"] for x in b.get("betas", [None])) or any(x != REGISTERED_BUILT["eps"] for x in b.get("eps", [None])) \
            or any(x != REGISTERED_BUILT["weight_decay"] for x in b.get("weight_decay", [None])):
        reasons.append(f"G-DVB Adam constants {b.get('betas')}/{b.get('eps')}/{b.get('weight_decay')} != registered {REGISTERED_BUILT}")
    if b.get("live_terms") != [dict(zip((0, 1, 3, 5), LIVE_TERMS))[r] for r in rows]:
        reasons.append(f"G-DVB live terms {b.get('live_terms')} do not match the trained rows {rows}")
    if b.get("init_sha256") != prof["init_from_sha256"]:
        reasons.append(f"G-DVB init-from sha256 {str(b.get('init_sha256'))[:16]} is not C1's {prof['init_from_sha256'][:16]} (MODEL_REGISTRY 14.6)")
    for f, k in (("--cache-index", "index_sha256"), ("--heldout-logs", "heldout_logs_sha256"), ("--navtest-logs", "navtest_logs_sha256"), ("--init-from", "init_sha256")):
        if (data.get(f) or {}).get("sha256") != b.get(k):
            reasons.append(f"G-DVB {f}: the trainer read sha256 {str(b.get(k))[:16]}, the data manifest has {str((data.get(f) or {}).get('sha256'))[:16]}")
    return C.verdict("G-DVB", reasons, det, arm=arm, rehearsal=rehearsal, inputs_read={f: data[f] for f in DATA_FLAGS if f in data})


def _metrics(p: Path) -> list:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.is_file() else []


def check_live(prof: dict, argv: list, smoke: Path, r, data: dict, arm=None, rehearsal=False) -> dict:
    T = importlib.import_module(prof["parser_module"])
    reasons, det = [], {"smoke_rc": r.returncode}
    if r.returncode != 0:
        reasons.append(f"the real-data smoke failed (rc {r.returncode}): {_tail(r)}")
        return C.verdict("G-LIVE", reasons, det, arm=arm, rehearsal=rehearsal)
    rows = list(T._ints(T.build_parser().parse_args(argv).rows))
    terms = [dict(zip((0, 1, 3, 5), LIVE_TERMS))[x] for x in rows]
    M = _metrics(smoke / "metrics.jsonl")
    det["steps"] = len(M)
    if len(M) != SMOKE_STEPS:
        reasons.append(f"the smoke logged {len(M)} steps, not {SMOKE_STEPS}")
    for m in M:
        for t in terms + ["loss"]:
            if t not in m or not np.isfinite(m[t]):
                reasons.append(f"step {m.get('step')}: declared loss term {t!r} absent or non-finite")
        for g in rows:
            if not m["grad_abs"].get(f"row{g}", 0.0) > 0.0:
                reasons.append(f"step {m['step']}: trained row {g} ({D.ROW_NAMES[g]}) received NO gradient")
        for g, v in m["frozen_grad_abs"].items():
            if v != 0.0:
                reasons.append(f"step {m['step']}: frozen {g} received gradient {v} (must be EXACTLY 0.0: the zero control)")
    b = json.loads((smoke / "built.json").read_text(encoding="utf-8"))
    if b.get("rows") != rows:
        reasons.append(f"trained rows {rows} but the optimiser holds {b.get('rows')} (a trained row outside the optimiser never moves)")
    h0, hN = D.load_head(smoke / "head_step00000.npz"), D.load_head(smoke / f"head_step{SMOKE_STEPS:05d}.npz")
    moved = {D.ROW_NAMES[g]: float(np.abs(hN["W"][g] - h0["W"][g]).sum() + abs(hN["b"][g] - h0["b"][g])) for g in rows}
    det["row_moved_abs"] = moved
    for g, v in moved.items():
        if not v > 0.0:
            reasons.append(f"trained row {g} did not move in {SMOKE_STEPS} steps")
    fr = [g for g in range(D.N_OUT) if g not in rows]
    if not (np.array_equal(hN["W"][fr], h0["W"][fr]) and np.array_equal(hN["b"][fr], h0["b"][fr]) and np.array_equal(h0["W"], h0["W0"])):
        reasons.append("a frozen row (EP/C) or the step-0 head differs from C1's weights")
    det["ctrl_feat"] = b.get("ctrl_feat")
    return C.verdict("G-LIVE", reasons, det, arm=arm, rehearsal=rehearsal, inputs_read={f: data[f] for f in DATA_FLAGS if f in data})


def check_eval(prof: dict, smoke: Path, r, arm=None, rehearsal=False) -> dict:
    reasons, det = [], {}
    if r.returncode != 0 or not (smoke / "probe.npz").is_file():
        reasons.append(f"no smoke probe (rc {r.returncode})")
        return C.verdict("G-EVAL", reasons, det, arm=arm, rehearsal=rehearsal)
    p = np.load(smoke / "probe.npz", allow_pickle=False)
    hN, h0 = D.load_head(smoke / f"head_step{SMOKE_STEPS:05d}.npz"), D.load_head(smoke / "head_step00000.npz")
    rows = list(hN["meta"]["rows"])
    rows_read = [(g + 1) % D.N_OUT for g in rows] if C.arm_is("sh_eval_rowshift", arm) else rows
    if not (np.array_equal(hN["W"], p["W"]) and np.array_equal(hN["b"], p["b"])):
        reasons.append("the saved head is not the trainer's in-memory head")
    # (b) the loader builds the SAME FUNCTION as the trainer: the loaded head on the same s vs the trainer's own torch forward
    dd = float(np.abs(D.direct_logits(p["s"], hN["W"], hN["b"]) - p["torch_logits"]).max())
    det["max_abs_loaded_head_vs_trainer_forward"] = dd
    if not dd <= EVAL_TOL:
        reasons.append(f"the loaded head differs from the trainer's forward by {dd:.3g} > {EVAL_TOL}")
    # (c) the read's delta form adds exactly the head's CHANGE on the trained rows (like for like: the cached logits carry the forward's own precision of s)
    L = D.delta_logits(p["cached_logits"], p["s"], hN["W"], hN["b"], hN["W0"], hN["b0"], rows_read)
    chg = p["torch_logits"] - D.direct_logits(p["s"], hN["W0"], hN["b0"])
    d = float(np.abs((L - p["cached_logits"])[..., rows] - chg[..., rows]).max())
    fr = [g for g in range(D.N_OUT) if g not in rows]
    det["max_abs_read_delta_vs_trainer_change"] = d
    if not d <= EVAL_TOL:
        reasons.append(f"the read's change on the trained rows differs from the trainer's by {d:.3g} > {EVAL_TOL}")
    if not np.array_equal(L[..., fr], p["cached_logits"][..., fr]):
        reasons.append("the read changes an untrained row (EP/C)")
    L0 = D.delta_logits(p["ho_logits"], p["ho_s"], h0["W"], h0["b"], h0["W0"], h0["b0"], rows_read)
    det["step0_bitexact"] = bool(np.array_equal(L0, p["ho_logits"]))
    det["step0_pick_agree"] = int((D.select(L0) == p["cached_pick"]).sum())
    det["n_heldout"] = int(len(p["cached_pick"]))
    if not det["step0_bitexact"] or det["step0_pick_agree"] != det["n_heldout"]:
        reasons.append(f"CTRL-ID through the read: the untrained head gives bit-exact logits {det['step0_bitexact']}, picks {det['step0_pick_agree']}/{det['n_heldout']}")
    return C.verdict("G-EVAL", reasons, det, arm=arm, rehearsal=rehearsal)


def check_ckpt(prof: dict, argv: list, work: Path, arm=None, rehearsal=False) -> dict:
    reasons, det = [], {}
    runs = {}
    for nm, n, sv in (("A1", 4, "0,2,4"), ("A2", 4, "0,2,4"), ("B", 2, "0,2")):
        out = work / f"ckpt_{nm}"
        r = _run_trainer(_derived(argv, out, ["--smoke-steps", str(n), "--save-steps", sv]), arm)
        runs[nm] = out
        if r.returncode != 0:
            reasons.append(f"ckpt run {nm} failed (rc {r.returncode}): {_tail(r)}")
    if reasons:
        return C.verdict("G-CKPT", reasons, det, arm=arm, rehearsal=rehearsal)
    hA1, hA2 = D.load_head(runs["A1"] / "head_step00004.npz"), D.load_head(runs["A2"] / "head_step00004.npz")
    p = np.load(runs["A1"] / "probe.npz", allow_pickle=False)
    if not (np.array_equal(hA1["W"], p["W"]) and np.array_equal(hA1["b"], p["b"])):
        reasons.append("round trip: the loaded head is not the in-memory head that was saved")
    det["sha_A1"], det["sha_A2"] = D.head_arrays_sha256(hA1), D.head_arrays_sha256(hA2)
    if det["sha_A1"] != det["sha_A2"] or (runs["A1"] / "metrics.jsonl").read_bytes() != (runs["A2"] / "metrics.jsonl").read_bytes():
        reasons.append("determinism: two runs of the same argv differ")
    hp, hB = D.load_head(runs["A1"] / "head_step00002.npz"), D.load_head(runs["B"] / "head_step00002.npz")
    if D.head_arrays_sha256(hp) != D.head_arrays_sha256(hB):
        reasons.append("prefix: the step-2 head of a 4-step run is not the final head of a 2-step run")
    m = hA1["meta"]
    if m.get("init_sha256") != prof["init_from_sha256"] or m.get("smoke") is not True:
        reasons.append(f"head meta: init {str(m.get('init_sha256'))[:12]} smoke {m.get('smoke')}")
    return C.verdict("G-CKPT", reasons, det, arm=arm, rehearsal=rehearsal)


def run_checks(prof: dict, argv: list, pkg, out_dir: Path, *, want, emit, arm=None, rehearsal=False, data=None, suite=None) -> None:
    """launch_gate.run_checks's branch for runner == "safehead".  Writes evidence through `emit` (the gate's own writer)."""
    data = data or {}
    work = Path(out_dir) / "work" / f"safehead_{time.time_ns()}"
    work.mkdir(parents=True, exist_ok=True)
    if "G-HYG" in want:
        emit(check_hyg(prof, argv, arm=arm, rehearsal=rehearsal))
    if "G-DVB" in want:
        emit(check_dvb(prof, argv, work, data, arm=arm, rehearsal=rehearsal))
    smoke, r = work / "smoke", None
    if "G-LIVE" in want or "G-EVAL" in want:
        r = _run_trainer(_derived(argv, smoke, ["--smoke-steps", str(SMOKE_STEPS), "--save-steps", f"0,{SMOKE_STEPS}"]), arm)
    if "G-LIVE" in want:
        emit(check_live(prof, argv, smoke, r, data, arm=arm, rehearsal=rehearsal))
    if "G-EVAL" in want:
        emit(check_eval(prof, smoke, r, arm=arm, rehearsal=rehearsal))
    if "G-CKPT" in want:
        emit(check_ckpt(prof, argv, work, arm=arm, rehearsal=rehearsal))
    if "G-SUITE" in want:
        emit(suite() if suite is not None else C.check_suite(prof, arm=arm, rehearsal=rehearsal))
    if "G-CLOCK" in want:
        emit(C.evidence("G-CLOCK", "NA", [prof.get("clock_na", "no label clock")], {}, arm=arm, rehearsal=rehearsal))
