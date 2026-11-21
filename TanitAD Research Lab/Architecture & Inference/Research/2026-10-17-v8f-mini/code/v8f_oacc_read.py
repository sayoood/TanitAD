"""PREREG_V8F_OACC, committed before its data: does training O to match the human's ACCELERATION (X = B + `--w-acc <rule>`, ONE lever) make the deployed v8F output dominate its kinematic prior?

  python v8f_oacc_read.py --eval-dir <eval> --runs-dir <runs> --weights-dir <w> --wacc-json <w_acc.json> --ckpt-sha <ckpt_sha256.json> --seeds 0   --out oacc_read_s0.json   (the s0 read: gates seed 1)
  python v8f_oacc_read.py ... --seeds 0,1 --out oacc_read.json                                                                                         (the final read)
  python v8f_oacc_read.py --check-inputs --eval-dir <eval> --runs-dir <runs> --out x.json                                                              (the banked B side alone, before any training)

B = M2b (fixed), the banked runs `M2b_s{0,1}` and their standard evals. X = `M2b_s<SEED>_oacc`, eval tag `M2b-oacc_s<SEED>`. HX = X with HBLEND's per-step weights RE-FITTED on X's own training windows
(`v8f_mini_eval_oacc.py hx`, the deployed form): HX = P + w (.) (tau_X - P), P = the checkpoint's own frozen P10 (HBLEND's convention), w = `weights_M2b-oacc_s<SEED>.json`.
Tier: T1-class open loop (eval-141, 2,399 windows over 141 episodes). Estimator: taniteval.ci's PAIRED episode-cluster bootstrap, B = 2,000 (EPISODES): it answers "another draw of EPISODES"; training-seed
variance enters only through F = |ADE6(B_s0) - ADE6(B_s1)| (H-ESTIM-SEED-1). v8F-mini's planner is deterministic: no inference-seed floor applies.

THE PREREG'S READS (verbatim):
  PRIMARY-ACC    accel MAE X - B  < 0 and separated, at the seed run
  PRIMARY-KEEP   ADE6 X - B  upper 95 % bound <= +F/2 (F = 0.1207 m -> +0.060 m)
  DOMINANCE      HX - P10 (P10-yaw-4512): HSPEC's PRIMARY-DOM row verbatim = separated-better on ADE2, ADE6, accel MAE and curvature MAE; AND not separated-worse on speed, along, cross-track, heading,
                 yaw-rate, both plan-derived tactical agreements, CONSISTENCY(a) 'any', distance keeping (headway, time-gap, LEAD-VIOL at-risk share)
  beside         the four families for X and HX (LONGITUDINAL incl. distance keeping and LEAD-VIOL, LATERAL, TACTICAL, STRATEGIC -- switched on for v8F, PI-V8F-2026-10-17, R72), CONSISTENCY(a), HS on X (descriptive)
OUTCOMES (PREREG, in the PREREG's words):
  DOMINATES ITS PRIOR       DOMINANCE at both seeds (seed 1 runs on seed 0's PRIMARY-ACC and PRIMARY-KEEP pass)
  SMOOTHER, NOT DOMINANT    PRIMARY-ACC and PRIMARY-KEEP pass, DOMINANCE fails
  COSTS POSITION            PRIMARY-ACC passes, PRIMARY-KEEP fails
  NO EFFECT                 PRIMARY-ACC fails

DECLARED INTERPRETATIONS (the PREREG leaves them open; each is fixed HERE before any OACC number exists, in `OUTCOME_RULE` and pinned by tests):
  I1  F is the EXACT |ADE6(B_s0) - ADE6(B_s1)| from B's two banked evals (0.12074 m, HBLEND / HSPEC convention); the bar is F/2 = 0.06037 m; the PREREG's printed +0.060 m is reported BESIDE (`hi_le_printed_0.060`) and
      decides nothing. F must reproduce 0.1207 to 5e-4 or the read is refused.
  I2  The rows are evaluated per RUN seed; a statement about "PRIMARY-ACC" or "PRIMARY-KEEP" in the outcome table holds iff it holds at EVERY seed that was run. Seed 1 runs iff seed 0 passes BOTH (PREREG). Order of the
      table: PRIMARY-ACC fails at any run seed -> NO EFFECT; else PRIMARY-KEEP fails at any run seed -> COSTS POSITION; else (both pass at every run seed, so seed 1 ran) DOMINANCE at both seeds -> DOMINATES ITS PRIOR,
      otherwise SMOOTHER, NOT DOMINANT. That is a complete partition of (ACC, KEEP, DOM) over the run seeds: no cell is undefined (the lesson of FLAT's Amendment 1).
  I3  A seed-0-only read whose seed 0 passes ACC and KEEP is reported as PENDING SEED 1 (s1_required true, rc 0): it is not an outcome.
  I4  HSPEC's I5 is kept: "separated-better / worse" = `taniteval.v8f_beta_eval.paired_row`'s better / worse (the metric's direction; the tactical agreements are higher-is-better); distance keeping WORSE = headway or
      time-gap separated NEGATIVE, LEAD-VIOL at-risk share separated POSITIVE; the K = 30 (6 s) block, as HSPEC.
  I5  STRATEGIC: S is built, trained and read in X like in B (the interface is `full`), so the family is READ from X's eval and compared with B's; it is NOT identical to B's by construction (X is a new training run
      with a different loss, and O's gradient reaches S through the checkpoints). What IS identical by construction is HX = X's tactical / strategic blocks (HX only re-weights O's plan): checked and reported.
Refusals (rc 2): a missing file; B's / X's run records; the gate token of X (declared_vs_built PASS, G-LIVE pass with a frozen-prior control of exactly 0.0 and NO dead weight, acc_probe pass for X's w_acc); an X argv
that is not B's plus `--w-acc <the rule's value>` exactly once; built levers / checkpoint that do not say that w_acc; a different parameter count; checkpoint sha256s that are not the banked B's or that moved between training,
evals and the pulled file; HX weights not fitted on X's checkpoint; evals not on the committed 2,399 / 141 grid or not one grid; identity controls (HX at w = 1 reproduces X's eval block; at w = 0 the plan is P10 and
reads 2.4870 to 1e-3); F off its known value; a MISSING FAMILY; a pair of states the outcome partition cannot place."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
REPO = HERE.parents[5]
RAW = PKG / "raw"
ARM_B, ARM_X, SUF = "M2b", "M2b-oacc", "_oacc"
LEVER_FLAG = "--w-acc"
EVAL_WINDOWS, EVAL_EPISODES = 2399, 141
B_BOOT = 2000
F_PRINTED, F_TOL = 0.1207, 5e-4
P10_ADE6, P10_TOL = 2.4870, 1e-3
PRINTED_KEEP_BAR = 0.060
FAMILIES = ("longitudinal", "lateral", "tactical", "strategic")
BANKED_CKPT_SHA = {"M2b_s0": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8",
                   "M2b_s1": "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
# HSPEC's PRIMARY-DOM row, verbatim (raw/2026-10-25-hspec/v8f_hspec.py: DOM_BETTER / DOM_NOT_WORSE; a test compares these dicts with that file)
DOM_BETTER = {"ADE2": "ade2", "ADE6": "ade6", "accel MAE": "fw_lon_accel_mae", "curvature MAE": "fw_lat_curvature_mae_1pm"}
DOM_NOT_WORSE = {"speed MAE": "fw_lon_speed_mae", "along MAE": "fw_lon_along_mae", "cross-track MAE": "fw_lat_cross_mae", "heading MAE": "fw_lat_heading_mae_deg",
                 "yaw-rate MAE": "fw_lat_yaw_rate_mae_degps", "plan-derived lateral decision agreement": "fw_tac_lat_acc",
                 "plan-derived longitudinal decision agreement": "fw_tac_lon_acc"}
LOWER_IS_BETTER = {"tac_lat_acc": False, "tac_lon_acc": False}
CK = ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD", "self_viol_any")
# The committed next-step texts, VERBATIM from PREREG_V8F_OACC.md's outcome table (a test compares them with the file)
NEXT_TEXT = {"DOMINATES ITS PRIOR": "HX is the deployed v8F output and the register states dominance over the **kinematic floor** on every family; ⛔ never as state of the art. Next: the same row against FLAT",
             "SMOOTHER, NOT DOMINANT": "X becomes the base for later arms. The acceleration gap is quantified per horizon; next, the longitudinal decision (`PREREG_V8F_KHYP`)",
             "COSTS POSITION": "the trade-off is reported; the weight rule is not re-tuned post hoc",
             "NO EFFECT": "the acceleration error is not reachable through O's loss at this capacity; next, an O-capacity read"}
OUTCOMES = tuple(NEXT_TEXT)


def refuse(msg):
    print(f"OACC READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _jload(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        refuse(f"cannot read {p}: {type(e).__name__}: {e}")


def x_dir(s):
    return f"{ARM_B}_s{s}{SUF}"


def x_tag(s):
    return f"{ARM_X}_s{s}"


# ------------------------------------------------------------------------------------------------ the decision (pure; unit-tested over the whole partition)
def seed_pass(acc: dict, keep: dict, dom: dict) -> dict:
    """One RUN seed's three booleans from the rows. PRIMARY-ACC: mean < 0 and separated. PRIMARY-KEEP: the upper bound <= the bar. DOMINANCE: HSPEC's PRIMARY-DOM row."""
    return {"ACC": bool(acc["mean"] < 0 and acc["separated"]), "KEEP": bool(keep["hi"] <= keep["bar_hi_le"]), "DOM": bool(dom["pass"])}


def outcome(states: dict) -> tuple:
    """``states``: {seed: {ACC, KEEP, DOM}} for the seeds that were RUN ({0} or {0, 1}). -> (OUTCOME | 'PENDING SEED 1', s1_required). I2 / I3, a complete partition:
    ACC fails at any run seed -> NO EFFECT; else KEEP fails at any run seed -> COSTS POSITION; else seed 0 alone -> PENDING SEED 1; else DOM at both seeds -> DOMINATES ITS PRIOR; else SMOOTHER, NOT DOMINANT."""
    if sorted(states) not in ([0], [0, 1]):
        refuse(f"outcome: the run seeds {sorted(states)} are not [0] or [0, 1]")
    for s, v in states.items():
        if set(v) != {"ACC", "KEEP", "DOM"} or not all(isinstance(x, bool) for x in v.values()):
            refuse(f"outcome: seed {s}'s state {v!r} is not three booleans")
    if not all(v["ACC"] for v in states.values()):
        return "NO EFFECT", False
    if not all(v["KEEP"] for v in states.values()):
        return "COSTS POSITION", False
    if sorted(states) == [0]:
        return "PENDING SEED 1", True
    if all(v["DOM"] for v in states.values()):
        return "DOMINATES ITS PRIOR", False
    return "SMOOTHER, NOT DOMINANT", False


OUTCOME_RULE = outcome


def dk_flags(dk_pair: dict) -> dict:
    """HSPEC's, verbatim (I4): {name: {'delta','lo','hi','separated','worse','better'}} for headway, time-gap, LEAD-VIOL at-risk."""
    out = {}
    for nm, row, hi_is_worse in (("headway min (m)", dk_pair["dk"]["headway_min_m"], False), ("time-gap min (s)", dk_pair["dk"]["time_gap_min_s"], False),
                                 ("LEAD-VIOL at-risk share", dk_pair["lead_viol"]["at_risk"], True)):
        d, lo, hi, sep = row.get("delta"), row.get("lo"), row.get("hi"), bool(row.get("separated"))
        out[nm] = {"delta": d, "lo": lo, "hi": hi, "separated": sep, "worse": bool(sep and ((d > 0) if hi_is_worse else (d < 0))),
                   "better": bool(sep and ((d < 0) if hi_is_worse else (d > 0))), "worse_two_sided_sensitivity": sep}
    return out


def dom_row(paired_hx_p10: dict, cons_viol_any: dict, dkf: dict) -> dict:
    """HSPEC's PRIMARY-DOM row for HX - P10 at one seed (its `stage_read`, per seed, verbatim). ``paired_hx_p10``: the paired rows keyed by per-window metric; ``cons_viol_any``: the paired row of CONSISTENCY(a) 'any';
    ``dkf``: `dk_flags` of the HX - P10 distance-keeping pair."""
    rows, ok_better, ok_nw = {}, True, True
    for nm, key in DOM_BETTER.items():
        r = paired_hx_p10[key]
        rows[nm] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"], "need": "separated-better"}
        ok_better &= bool(r["better"])
    for nm, key in DOM_NOT_WORSE.items():
        r = paired_hx_p10[key]
        rows[nm] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"], "need": "not separated-worse"}
        ok_nw &= not bool(r["worse"])
    r = cons_viol_any
    rows["CONSISTENCY(a) 'any'"] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"], "need": "not separated-worse"}
    ok_nw &= not bool(r["worse"])
    for nm, f in dkf.items():
        rows[f"DK {nm}"] = {"mean": f["delta"], "lo": f["lo"], "hi": f["hi"], "separated": f["separated"], "better": f["better"], "worse": f["worse"], "need": "not separated-worse"}
        ok_nw &= not f["worse"]
    return {"pass": bool(ok_better and ok_nw), "all_better_ok": bool(ok_better), "all_not_worse_ok": bool(ok_nw), "rows": rows}


def keep_row(paired_ade6_x_minus_b: dict, F: float) -> dict:
    r = paired_ade6_x_minus_b
    return {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "bar_hi_le": F / 2, "pass": bool(r["hi"] <= F / 2), "hi_le_printed_0.060": bool(r["hi"] <= PRINTED_KEEP_BAR)}


# ------------------------------------------------------------------------------------------------ run records (pure over loaded dicts)
def _flags(argv):
    out, i = {}, 0
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[t] = argv[i + 1]
                i += 2
                continue
            out[t] = True
        i += 1
    return out


#: B's BUILT loss weights that its older record does not list (MEASURED 2026-10-26: the cfg of the banked M2b_s0 / M2b_s1 checkpoints; PREREG_V8F_OACC Amendment 2)
W_B_BUILT = {"w_tac": 1.0, "w_cons": 0.5}


def check_x_run(s, sm, argv, sc, rsm, rargv, w_acc, *, n_train=4512, holdout=None, lever_skip=("w_acc", "o_interface", "code_mix", "params", "n_params_trainable", "residual_zero_init", "w")) -> list:
    """The reasons seed ``s``'s X run is not B's plus ONE lever (empty = fine). ``rsm`` / ``rargv``: the banked B_s run's summary and argv. The launch-gate PASS token is selfcheck.json's."""
    bad, tag = [], x_tag(s)
    fl, rfl = _flags(argv), _flags(rargv)
    if sm.get("done") is not True:
        bad.append(f"{tag} is not done")
    if sm.get("arm") != "M2b" or sm.get("seed") != s:
        bad.append(f"{tag}/summary.json says {sm.get('arm')} seed {sm.get('seed')}, expected M2b seed {s}")
    if sm.get("n_clips") != n_train:
        bad.append(f"{tag}: n_clips {sm.get('n_clips')} != {n_train}")
    if holdout is not None and sm.get("excluded_sha256") != holdout:
        bad.append(f"{tag} did not withhold the RA-1 holdout")
    for k, what in (("train_sha256", "clip list"), ("steps", "step count"), ("n_windows", "window set")):
        if sm.get(k) != rsm.get(k):
            bad.append(f"{tag}: {what} ({k}) {str(sm.get(k))[:16]} != B's {str(rsm.get(k))[:16]}")
    # the launch-gate PASS token
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"{tag}: declared_vs_built is not PASS (no PASS token)")
    live = sc.get("live") or {}
    if live.get("pass") is not True or live.get("control_prior_grad_sum") != 0.0:
        bad.append(f"{tag}: G-LIVE is not a pass with a frozen-prior control of exactly 0.0 (no PASS token)")
    if live.get("dead_params") != [] or live.get("declared_dead") != []:
        bad.append(f"{tag}: G-LIVE's dead-weight set is {live.get('dead_params')} (declared {live.get('declared_dead')}); the lever kills no weight")
    if "acc" not in (live.get("terms") or {}):
        bad.append(f"{tag}: G-LIVE's terms carry no 'acc': the term is not in the loss dict")
    pr = sc.get("acc_probe")
    if not isinstance(pr, dict) or pr.get("pass") is not True:
        bad.append(f"{tag}: selfcheck.json carries no passing acc_probe (no PASS token)")
    else:
        if pr.get("w_acc_declared") != w_acc:
            bad.append(f"{tag}: acc_probe was run for w_acc {pr.get('w_acc_declared')!r}, the rule's value is {w_acc!r}")
        if not (isinstance(pr.get("head_o_grad_diff"), (int, float)) and pr["head_o_grad_diff"] > 0.0):
            bad.append(f"{tag}: acc_probe: O's head received no gradient from the term ({pr.get('head_o_grad_diff')})")
        if pr.get("control_prior_grad_diff") != 0.0 or pr.get("control_repeat_grad_diff") != 0.0 or pr.get("control_repeat_total_diff") != 0.0:
            bad.append(f"{tag}: an acc_probe control does not read exactly 0.0: {({k: pr.get(k) for k in ('control_prior_grad_diff', 'control_repeat_grad_diff', 'control_repeat_total_diff')})}")
        if not abs(float(pr.get("contribution", 0.0)) - float(pr.get("w_x_acc", 1.0))) <= 1e-3 * abs(float(pr.get("w_x_acc", 1.0))) + 1e-5:
            bad.append(f"{tag}: acc_probe: the total moved by {pr.get('contribution')}, w_acc x term is {pr.get('w_x_acc')}")
    if sc.get("argv_sha") != sm.get("argv_sha"):
        bad.append(f"{tag}: selfcheck.json (argv_sha {sc.get('argv_sha')}) is not this run's (summary.json {sm.get('argv_sha')})")
    # ONE lever: B's argv plus --w-acc <the rule's value> exactly once
    if fl.get("--arm") != "M2b":
        bad.append(f"{tag}: argv --arm {fl.get('--arm')}, expected M2b")
    if argv.count(LEVER_FLAG) != 1 or not _same(fl.get(LEVER_FLAG), w_acc):
        bad.append(f"{tag}: argv {LEVER_FLAG} {fl.get(LEVER_FLAG)} (named {argv.count(LEVER_FLAG)} times), expected {w_acc!r} exactly once")
    if LEVER_FLAG in rfl:
        bad.append(f"B_s{s}: argv names {LEVER_FLAG}: B is the banked M2b, which predates the flag")
    drop = {"--out", LEVER_FLAG}
    fx, fb = {k: x for k, x in fl.items() if k not in drop}, {k: x for k, x in rfl.items() if k not in drop}
    if fx != fb:
        bad.append(f"{tag}: argv differs from B_s{s}'s in {sorted(k for k in set(fx) | set(fb) if fx.get(k) != fb.get(k))} besides {LEVER_FLAG} and --out -- not one lever")
    for w, rec in (("summary", sm), ("selfcheck", sc)):
        lg, lb = rec.get("levers") or {}, rsm.get("levers") or {}
        if lg.get("w_acc") is None:
            bad.append(f"{tag}: {w}.json levers record no built w_acc: the lever is unproven")
        elif not _same(lg["w_acc"], w_acc):
            bad.append(f"{tag}: {w}.json built w_acc {lg['w_acc']!r}, the rule's value is {w_acc!r}")
        diff = sorted(k for k in set(lg) & set(lb) if k not in lever_skip and lg[k] != lb[k])
        if diff:
            bad.append(f"{tag}: {w}.json built levers {diff} differ from B_s{s}'s")
        # PREREG_V8F_OACC Amendment 2: `w` key by key; a weight B's older record does not carry must equal B's BUILT config (W_B_BUILT)
        wb, wx = lb.get("w") or {}, lg.get("w") or {}
        w_bad = sorted(k for k in set(wb) | set(wx) if wx.get(k) != (wb[k] if k in wb else W_B_BUILT.get(k, "absent from B's config")))
        if w_bad:
            bad.append(f"{tag}: {w}.json built loss weights {w_bad} differ from B_s{s}'s (a weight B's record lacks must equal B's built config {W_B_BUILT})")
        pb, px = lb.get("params") or {}, lg.get("params") or {}
        if not pb or pb != px or lg.get("n_params_trainable") != lb.get("n_params_trainable"):
            bad.append(f"{tag}: {w}.json parameter counts {px} / {lg.get('n_params_trainable')} are not B's {pb} / {lb.get('n_params_trainable')} (the lever adds no parameter)")
    return bad


def _same(got, want):
    try:
        return float(got) == float(want)
    except (TypeError, ValueError):
        return False


def ckpt_w_acc(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return (ck.get("cfg") or {}).get("w_acc"), ck.get("argv_sha"), ck.get("step"), ck.get("arm"), ck.get("seed")


def check_ckpt_sha(path, runs_dir, seeds):
    """B's two banked checkpoints, and each X checkpoint: one value at the end of training, after the evals and in the file pulled here."""
    if not path or not os.path.exists(path):
        refuse("--ckpt-sha is required and must exist: the chain's json of the checkpoint sha256 it measured on box 2")
    got, out = _jload(path), {}
    for tag, want in BANKED_CKPT_SHA.items():
        if got.get(tag) != want:
            refuse(f"the checkpoint {tag} has sha256 {str(got.get(tag))[:16]}..., the banked run's is {want[:16]}...: this is not the banked checkpoint")
        out[tag] = got[tag]
    for s in seeds:
        t = x_tag(s)
        a1, a2 = got.get(f"{t}_trained"), got.get(f"{t}_after_evals")
        loc = sha256_file(os.path.join(runs_dir, x_dir(s), "ckpt_last.pt"))
        if not (a1 and a2) or a1 != a2:
            refuse(f"{t}: the checkpoint sha256 at the end of training ({str(a1)[:16]}) and after the evals ({str(a2)[:16]}) are not one value")
        if a1 != loc:
            refuse(f"{t}: the checkpoint pulled into {runs_dir} has sha256 {loc[:16]}..., box 2 measured {str(a1)[:16]}...")
        out[t] = a1
    return out


# ------------------------------------------------------------------------------------------------ the instruments (the programme's own, imported)
class Instruments:
    def __init__(self):
        sys.path[:0] = [str(REPO / "stack"), str(REPO / "taniteval")]
        from taniteval import ci as CI                      # noqa: E402
        from taniteval import v8f_beta_eval as BE           # noqa: E402
        from tanitad.data import v8f_mini_data as D         # noqa: E402
        self.CI, self.BE, self.D = CI, BE, D
        self.ev = _load("v8f_mini_eval_for_oacc", REPO / "stack" / "scripts" / "v8f_mini_eval.py")
        self.H = _load("v8f_hblend_for_oacc_read", RAW / "2026-10-25-hblend" / "v8f_hblend.py")
        self.DK = _load("v8f_dk_read_for_oacc", HERE.parent / "v8f_dk_read.py")
        self.FR = _load("v8f_flat_read_for_oacc", HERE.parent / "v8f_flat_read.py")      # B's run loader, ckpt_facts, the programme's refusals routed to ONE message prefix
        self.FR.refuse = refuse
        self.FR.PGT.refuse = refuse
        self.FR.GR.refuse = refuse


def eval_world(I, a):
    D = I.D
    grid = json.load(open(a.eval_grid, encoding="utf-8"))
    J = D.load_join(a.eval_join)
    X = D.eval_context(grid, a.eval_kin, J, kin_mode="yaw")
    return grid, J, X


def load_eval(eval_dir, tag):
    npz, js = os.path.join(eval_dir, f"eval_{tag}.npz"), os.path.join(eval_dir, f"eval_{tag}.json")
    for p in (npz, js):
        if not os.path.exists(p):
            refuse(f"{p} is missing")
    return np.load(npz), _jload(js)


def prior_plan(I, ckpt_path, X):
    """P = the checkpoint's OWN frozen P10 on the eval kin features (HBLEND's convention; no image needed)."""
    import torch
    from tanitad.models import v8f_mini as M
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    with torch.no_grad():
        return model.prior(torch.as_tensor(X["kf8"], dtype=torch.float32), torch.as_tensor(X["base"], dtype=torch.float32)).numpy().astype(np.float64)


def paired_rows(I, pw_a, pw_b, eid, nb):
    return {k: I.BE.paired_row(pw_a[k], pw_b[k], eid, nb, lower_is_better=LOWER_IS_BETTER.get(k.replace("fw_", ""), True)) for k in pw_a}


def exact_F(zB0, zB1):
    return abs(float(zB0["ade6"].mean()) - float(zB1["ade6"].mean()))


def check_F(F):
    if abs(F - F_PRINTED) > F_TOL:
        refuse(f"F = |ADE6(B_s0) - ADE6(B_s1)| reads {F:.5f} m, the PREREG's known value is {F_PRINTED}: these are not B's banked evals")


# ------------------------------------------------------------------------------------------------ one seed
def read_seed(I, a, s, world, zB, bjB, F, w_acc, P10, fl, lead_block):
    ev, H, BE, D, DK = I.ev, I.H, I.BE, I.D, I.DK
    grid, J, X = world
    nb = a.n_boot
    eid, gt = X["eid"], X["gt"].astype(np.float64)
    zX, bjX = load_eval(a.eval_dir, x_tag(s))
    if (bjX.get("o_interface") or "full") != "full":
        refuse(f"{x_tag(s)}: the eval built o_interface {bjX.get('o_interface')!r}: X keeps B's full interface")
    ck_path = os.path.join(a.runs_dir, x_dir(s), "ckpt_last.pt")
    parts = str(bjX.get("ckpt") or "").replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 2 or parts[-2] != x_dir(s) or parts[-1] != "ckpt_last.pt":
        refuse(f"{x_tag(s)}: the eval scored {bjX.get('ckpt')!r}, not <runs>/{x_dir(s)}/ckpt_last.pt")
    if not np.array_equal(np.asarray(zX["eid"]), np.asarray(eid)) or not np.array_equal(np.asarray(zB["eid"]), np.asarray(eid)):
        refuse("the evals are not on the committed eval grid (eid differs)")
    n, ne = int(eid.size), int(np.unique(eid).size)
    if (n, ne) != (a.n_windows, a.n_episodes):
        refuse(f"the evals hold {n} windows over {ne} episodes, the committed eval-141 grid has {a.n_windows} over {a.n_episodes}")
    tauB, tauX = zB["pred"].astype(np.float64), zX["pred"].astype(np.float64)
    P = prior_plan(I, ck_path, X)
    # ---- the HX weights: fitted on THIS checkpoint (sha), on the TRAIN windows only
    W = _jload(os.path.join(a.weights_dir, f"weights_{x_tag(s)}.json"))
    if W.get("ckpt_sha256") != sha256_file(ck_path):
        refuse(f"{x_tag(s)}: the HX weights were fitted on checkpoint {str(W.get('ckpt_sha256'))[:16]}..., not on the pulled {x_dir(s)}/ckpt_last.pt")
    if not _same(W.get("w_acc_built"), w_acc):
        refuse(f"{x_tag(s)}: the HX weights' checkpoint was built with w_acc {W.get('w_acc_built')!r}, the rule's value is {w_acc!r}")
    w = np.stack([np.asarray(W["w_along_by_step"], np.float64), np.asarray(W["w_cross_by_step"], np.float64)], 1)
    if not (w.shape == (30, 2) and ((w >= 0) & (w <= 1)).all() and (np.diff(w, axis=0) >= -1e-12).all()):
        refuse(f"{x_tag(s)}: the HX weights are not in [0, 1] and monotone non-decreasing")
    ctl = {"P_vs_P10_file_maxabs_m": float(np.abs(P - P10).max())}
    HX, HX1, HX0 = H.apply_w(P, tauX, w), H.apply_w(P, tauX, 1.0), H.apply_w(P, tauX, 0.0)
    ctl["w1_vs_X_pred_maxabs_m"] = float(np.abs(HX1 - tauX).max())
    ctl["w0_ade6_mean"] = float(np.linalg.norm(HX0 - gt, axis=-1).mean())
    ctl["w0_ade6_vs_prereg_2.4870_absdiff"] = abs(ctl["w0_ade6_mean"] - P10_ADE6)
    ctl["X_banked_ade6_recomputed_maxabs"] = float(np.abs(np.linalg.norm(tauX - gt, axis=-1).mean(1) - zX["ade6"]).max())
    ctl["all_pass"] = bool(ctl["P_vs_P10_file_maxabs_m"] <= 1e-4 and ctl["w1_vs_X_pred_maxabs_m"] <= 1e-4 and ctl["w0_ade6_vs_prereg_2.4870_absdiff"] <= P10_TOL
                           and ctl["X_banked_ade6_recomputed_maxabs"] <= 1e-4)
    if not ctl["all_pass"]:
        refuse(f"{x_tag(s)}: identity controls failed: {json.dumps(ctl)}")
    # ---- per-window metric sets and the paired rows
    pwB, pwX, pwHX, pwP = (H.arm_pw(p, gt, BE) for p in (tauB, tauX, HX, P10))
    acc_row = BE.paired_row(pwX["fw_lon_accel_mae"], pwB["fw_lon_accel_mae"], eid, nb)
    keep = keep_row(BE.paired_row(pwX["ade6"], pwB["ade6"], eid, nb), F)
    p_hx_p10 = paired_rows(I, pwHX, pwP, eid, nb)
    exB, exX = H.extras_of(zB, D), H.extras_of(zX, D)
    resX, pwX_ev = ev.evaluate(tauX, X, exX, nb, with_families=True)
    resHX, pwHX_ev = ev.evaluate(HX, X, exX, nb, with_families=True)
    resHX1, _ = ev.evaluate(HX1, X, exX, nb, with_families=True)
    d1 = H.deep_diff(H.jround(resHX1), {k: bjX["raw"][k] for k in resHX1 if k in bjX["raw"]}, tol=1e-6)
    d1 = [m for m in d1 if not m.startswith("/traj/acts/zero_accel_frac")]            # HSPEC I7: a hard-threshold diagnostic that straddles under float32 rounding, not a reading quantity
    if d1:
        refuse(f"{x_tag(s)}: HX at w = 1 does not reproduce X's own eval block (the scoring route is not the eval's): {d1[:6]}")
    ctl["w1_full_block_vs_X_json_n_mismatches_excl_zero_accel_threshold_diagnostic"] = len(d1)
    viol = {"B": {k: zB[k].astype(bool) for k in CK}, "X": {k: pwX_ev[k].astype(bool) for k in CK}, "HX": {k: pwHX_ev[k].astype(bool) for k in CK}, "P10": {k: fl[f"P10_refit__{k}"].astype(bool) for k in CK[:5]}}
    cons_hx_p10 = BE.paired_row(viol["HX"]["viol_any"].astype(float), viol["P10"]["viol_any"].astype(float), eid, nb)
    # ---- distance keeping through the programme's own reader (K = 30, as HSPEC)
    order = list(grid["order"])
    preds = {f"B_s{s}": tauB, f"X_s{s}": tauX, f"HX_s{s}": HX, "P10": P10}
    pairs = [(f"HX_s{s}", "P10"), (f"X_s{s}", f"B_s{s}"), (f"HX_s{s}", f"B_s{s}"), (f"HX_s{s}", f"X_s{s}"), (f"X_s{s}", "P10"), (f"B_s{s}", "P10")]
    dk = DK.read_dk(preds=preds, gt=gt, eid=eid, clip=np.array([order[e] for e in eid]), frame=2 * np.asarray(X["t"], np.int64), v0=np.asarray(X["kin"][:, 0], np.float64), block=lead_block,
                    k=30, dt=0.2, n_boot=a.dk_n_boot, seed=0, saved_ade6={f"B_s{s}": zB["ade6"], f"X_s{s}": zX["ade6"]}, p10_floor_ade6=fl["P10_refit__ade6"], pairs=pairs)
    dkread = DK._clean(dk["read"])
    dkf = dk_flags(dkread["pairs"][f"HX_s{s}__minus__P10"])
    dom = dom_row(p_hx_p10, cons_hx_p10, dkf)
    st = seed_pass(acc_row, keep, dom)
    # ---- beside: the four families (never pooled), HS on X is descriptive and lives in the HSPEC package
    for arm, blk in (("B", bjB["raw"]), ("X", bjX["raw"])):
        for fam in ("tactical", "strategic"):
            if not blk.get(fam):
                refuse(f"{arm}'s eval json has no {fam} family (a missing family is refused, never dropped)")
    for fam in ("longitudinal", "lateral"):
        if not (resHX.get("families") or {}).get(fam):
            refuse(f"HX's evaluate block has no {fam} family")
    fam_pair = {"X_minus_B": {k: v for k, v in paired_rows(I, pwX, pwB, eid, nb).items() if k.startswith("fw_")},
                "HX_minus_B": {k: v for k, v in paired_rows(I, pwHX, pwB, eid, nb).items() if k.startswith("fw_")},
                "HX_minus_X": {k: v for k, v in paired_rows(I, pwHX, pwX, eid, nb).items() if k.startswith("fw_")}}
    cx, cb = np.asarray(zX["ck_pos_err"], np.float64), np.asarray(zB["ck_pos_err"], np.float64)       # [N, 3]: S's three checkpoints' position error per window
    with np.errstate(all="ignore"):
        strat_pair = {"mean_over_checkpoints": BE.paired_row(np.nanmean(cx, 1), np.nanmean(cb, 1), eid, nb), "per_checkpoint": [BE.paired_row(cx[:, j], cb[:, j], eid, nb) for j in range(cx.shape[1])]}
    res = {"seed": s, "tag": x_tag(s), "ckpt_sha256": sha256_file(ck_path), "state": st, "PRIMARY_ACC": {**{k: acc_row[k] for k in ("mean", "lo", "hi", "separated", "better", "worse")}, "pass": st["ACC"],
           "metric": "accel MAE (m/s^2), X - B, per-window family component fw_lon_accel_mae", "need": "< 0 and separated"},
           "PRIMARY_KEEP": keep, "DOMINANCE": dom, "identity_controls": ctl,
           "beside": {"ADE": {nm: {"X_minus_B": BE.paired_row(pwX[k], pwB[k], eid, nb), "HX_minus_B": BE.paired_row(pwHX[k], pwB[k], eid, nb), "HX_minus_P10": p_hx_p10[k]} for nm, k in
                              (("ADE2", "ade2"), ("ADE4", "ade4"), ("ADE6", "ade6"), ("FDE6", "fde6"))},
                      "families": {"longitudinal_lateral_paired_per_window": fam_pair, "X_eval_block": {k: resX.get(k) for k in ("families", "traj")},
                                   "HX_eval_block": {k: resHX.get(k) for k in ("families", "traj")}, "B_banked_block": {k: bjB["raw"].get(k) for k in ("families", "traj")},
                                   "tactical": {"B": bjB["raw"]["tactical"], "X": bjX["raw"]["tactical"], "HX": resHX.get("tactical"),
                                                "HX_equals_X_by_construction": H.jround(resHX.get("tactical")) == H.jround(resX.get("tactical"))},
                                   "strategic": {"status": "READ (switched on for v8F: PI-V8F-2026-10-17, R72)", "B": bjB["raw"]["strategic"], "X": bjX["raw"]["strategic"], "HX": resHX.get("strategic"),
                                                 "X_minus_B_ck_pos_err": strat_pair,
                                                 "identical_to_B": H.jround(bjB["raw"]["strategic"]) == H.jround(bjX["raw"]["strategic"]),
                                                 "identical_by_construction": False,
                                                 "why_not_by_construction": "X is a NEW training run with an extra loss term; O's gradient reaches S through the checkpoints in the full interface, so S is B's architecture, not B's weights",
                                                 "HX_equals_X_by_construction": H.jround(resHX.get("strategic")) == H.jround(resX.get("strategic"))},
                                   "distance_keeping_k30": dkread},
                      "consistency_a": {"rate": {k: {a_: float(v[a_].mean()) for a_ in v} for k, v in viol.items()}, "HX_minus_P10_any": cons_hx_p10},
                      "weights_HX": {"w_along_by_step": W["w_along_by_step"], "w_cross_by_step": W["w_cross_by_step"], "n_windows_fit": W.get("n_windows"), "fit_clips_sha256": W.get("fit_clips_sha256")}}}
    return res


def read_all(a, I):
    seeds = tuple(int(x) for x in a.seeds.split(","))
    if seeds not in ((0,), (0, 1)):
        refuse(f"--seeds {a.seeds!r}: expected 0 or 0,1")
    world = eval_world(I, a)
    wj = _jload(a.wacc_json)
    w_acc = wj["w_acc"]
    if not (isinstance(w_acc, (int, float)) and w_acc > 0 and wj.get("schema") == "v8f-oacc-wacc-v1"):
        refuse(f"{a.wacc_json}: not a v8f-oacc-wacc-v1 record with w_acc > 0")
    SB, _ = I.FR.load_b_runs(a.runs_dir)
    zB, bjB = {}, {}
    for s in (0, 1):
        zB[s], bjB[s] = load_eval(a.eval_dir, f"{ARM_B}_s{s}")
    F = exact_F(zB[0], zB[1])
    check_F(F)
    sha = check_ckpt_sha(a.ckpt_sha, a.runs_dir, seeds)
    fl = np.load(os.path.join(a.eval_dir, "floors_perwindow.npz")) if not a.floors else np.load(a.floors)
    P10 = np.load(a.p10_pred or str(I.DK.P10_DEFAULT)).astype(np.float64)
    lead_block = I.DK.load_block(a.lead_block_k30 or str(I.DK.LEAD_BLOCK_K30))
    out_seeds, bad = {}, []
    for s in seeds:
        d = os.path.join(a.runs_dir, x_dir(s))
        for f in ("summary.json", "argv.json", "selfcheck.json", "ckpt_last.pt"):
            if not os.path.exists(os.path.join(d, f)):
                refuse(f"{os.path.join(d, f)} is missing (the run did not finish, or was not pulled)")
        sm, argv, sc = (_jload(os.path.join(d, f)) for f in ("summary.json", "argv.json", "selfcheck.json"))
        ref = SB[(ARM_B, s)]
        bad += check_x_run(s, sm, argv.get("argv", []), sc, ref["summary.json"], ref["argv.json"].get("argv", []), w_acc, n_train=I.FR.PGT.N_TRAIN, holdout=I.FR.PGT.HOLDOUT_SHA)
        wa, ash, step, arm, sd = ckpt_w_acc(os.path.join(d, "ckpt_last.pt"))
        if not _same(wa, w_acc) or ash != sm.get("argv_sha") or step != sm.get("steps") or arm != "M2b" or sd != s:
            bad.append(f"{x_tag(s)}: the checkpoint says w_acc {wa!r} / argv_sha {ash} / step {step} / arm {arm} / seed {sd}; the run says w_acc {w_acc!r} / {sm.get('argv_sha')} / {sm.get('steps')} / M2b / {s}")
    if bad:
        refuse("; ".join(bad))
    for s in seeds:
        out_seeds[f"s{s}"] = read_seed(I, a, s, world, zB[s], bjB[s], F, w_acc, P10, fl, lead_block)
    states = {s: out_seeds[f"s{s}"]["state"] for s in seeds}
    oc, s1 = outcome(states)
    nxt = None if oc == "PENDING SEED 1" else f"{oc}: {NEXT_TEXT[oc]}."
    fmt = lambda f: " / ".join(f(out_seeds[f"s{s}"]) for s in seeds)      # noqa: E731
    reading = (f"PREREG_V8F_OACC: {oc}; w_acc {w_acc}; accel MAE X - B {fmt(lambda r: format(r['PRIMARY_ACC']['mean'], '+.4f') + ('' if r['PRIMARY_ACC']['separated'] else ' ns'))} m/s^2 (need < 0 separated); "
               f"ADE6 X - B upper {fmt(lambda r: format(r['PRIMARY_KEEP']['hi'], '+.4f'))} m (bar {F / 2:+.4f}); DOMINANCE HX - P10 {fmt(lambda r: 'PASS' if r['DOMINANCE']['pass'] else 'FAIL')}")
    return {"schema": "v8f-oacc-read-v1", "prereg": "PREREG_V8F_OACC.md", "base": ARM_B, "tier": "T1-class open loop (eval-141, 2,399 windows over 141 episodes)",
            "estimator": f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); answers 'another draw of EPISODES' only; training-seed variance enters only through F (H-ESTIM-SEED-1); planner deterministic",
            "n_windows": a.n_windows, "n_episodes": a.n_episodes, "w_acc": w_acc, "wacc_record": wj, "F": F, "keep_bar": F / 2, "keep_bar_printed": PRINTED_KEEP_BAR, "checkpoint_sha256": sha,
            "seeds_read": list(seeds), "per_seed": out_seeds, "seed_states": {f"s{s}": states[s] for s in seeds}, "s1_required": bool(s1), "OUTCOME": oc, "next": nxt,
            "strategic_family": "READ for v8F (switched on: PI-V8F-2026-10-17, R72); see per_seed.*.beside.families.strategic", "reading": reading}


def check_inputs(a, I):
    """The B side alone, before any training: B's runs, both banked evals on the committed grid, F reproducing its known value, P10's ADE6 reading 2.4870, the eval world loading."""
    world = eval_world(I, a)
    grid, J, X = world
    SB, _ = I.FR.load_b_runs(a.runs_dir)
    zB = {s: load_eval(a.eval_dir, f"{ARM_B}_s{s}") for s in (0, 1)}
    eid = X["eid"]
    n, ne = int(eid.size), int(np.unique(eid).size)
    if (n, ne) != (a.n_windows, a.n_episodes):
        refuse(f"the eval grid holds {n} windows over {ne} episodes, the committed one {a.n_windows} over {a.n_episodes}")
    for s in (0, 1):
        if not np.array_equal(np.asarray(zB[s][0]["eid"]), np.asarray(eid)):
            refuse(f"M2b_s{s}'s eval is not on the committed grid")
    F = exact_F(zB[0][0], zB[1][0])
    check_F(F)
    P10 = np.load(a.p10_pred or str(I.DK.P10_DEFAULT)).astype(np.float64)
    gt = X["gt"].astype(np.float64)
    p10_ade6 = float(np.linalg.norm(P10 - gt, axis=-1).mean())
    if abs(p10_ade6 - P10_ADE6) > P10_TOL:
        refuse(f"P10-yaw-4512 reads ADE6 {p10_ade6:.4f} on the eval grid, the PREREG's known value is {P10_ADE6}")
    out = {"base": ARM_B, "n_windows": n, "n_episodes": ne, "F": F, "keep_bar": F / 2, "P10_ade6": p10_ade6,
           "B_ade6": {f"s{s}": float(zB[s][0]["ade6"].mean()) for s in (0, 1)}, "B_accel_mae": {f"s{s}": float(np.nanmean(I.H.arm_pw(zB[s][0]["pred"].astype(np.float64), gt, I.BE)["fw_lon_accel_mae"])) for s in (0, 1)},
           "banked_ckpt_sha256": BANKED_CKPT_SHA, "expected_x_evals_per_seed": ["standard"]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-inputs", action="store_true")
    ap.add_argument("--eval-dir")
    ap.add_argument("--runs-dir")
    ap.add_argument("--weights-dir", help="the directory of weights_M2b-oacc_s<seed>.json (v8f_mini_eval_oacc.py hx)")
    ap.add_argument("--wacc-json", help="the rule's record (v8f_mini_eval_oacc.py wacc)")
    ap.add_argument("--ckpt-sha")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=B_BOOT)
    ap.add_argument("--dk-n-boot", type=int, default=B_BOOT)
    ap.add_argument("--n-windows", type=int, default=EVAL_WINDOWS)
    ap.add_argument("--n-episodes", type=int, default=EVAL_EPISODES)
    ap.add_argument("--floors", help="floors_perwindow.npz (default: <eval-dir>/floors_perwindow.npz, staged by the chain)")
    ap.add_argument("--p10-pred", default=None)
    ap.add_argument("--lead-block-k30", default=None)
    ap.add_argument("--eval-grid", default=str(RAW / "eval_grid_windows.json"))
    ap.add_argument("--eval-kin", default=str(RAW / "eval141_kin.npz"))
    ap.add_argument("--eval-join", default=str(RAW / "join_eval141.npz"))
    a = ap.parse_args(argv)
    try:
        need = ("eval_dir", "runs_dir", "out") if a.check_inputs else ("eval_dir", "runs_dir", "weights_dir", "wacc_json", "ckpt_sha", "out")
        for k in need:
            if getattr(a, k) is None:
                ap.error(f"--{k.replace('_', '-')} is required")
        I = Instruments()
        res = check_inputs(a, I) if a.check_inputs else read_all(a, I)
    except (OSError, KeyError, ValueError) as e:
        print(f"OACC READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    except Exception as e:                           # an inherited instrument's OWN Refusal (v8f_dk_read.Refusal): still a refusal, rc 2, never a traceback
        if type(e).__name__ != "Refusal":
            raise
        print(f"OACC READ REFUSED: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(I.DK._clean(res), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if a.check_inputs:
        print(f"OACC INPUTS OK: base {ARM_B}; F {res['F']:.5f} (bar F/2 {res['keep_bar']:.5f}); P10 ADE6 {res['P10_ade6']:.4f}; {res['n_windows']} windows over {res['n_episodes']} episodes")
    else:
        print(json.dumps({k: res[k] for k in ("reading", "OUTCOME", "s1_required", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
