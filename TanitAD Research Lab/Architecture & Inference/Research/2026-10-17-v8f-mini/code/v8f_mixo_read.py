"""PREREG_V8F_MIXO, committed before its data: does training O on codes mixed PER UNIT (`--code-mix kind`, ONE lever) teach it to use a partly right T code, and does that make
v8F-mini drive better?

  python v8f_mixo_read.py --stage s0|final --eval-dir <eval> --runs-dir <runs> --ckpt-sha ckpt_sha256.json --out mixo_read.json
  python v8f_mixo_read.py --check-inputs --eval-dir <eval> --runs-dir <runs> --out x.json      (the banked B side alone, before any training)

B = M2b (fixed in the PREREG), the banked runs `M2b_s{0,1}` and their evals. X = B + `--code-mix kind` and nothing else (run dir `M2b_s<SEED>_mixo`; eval tags `M2b-mixo_s<SEED>`,
suffixes "", "-tacabs", "-tacorc", "-tacorc-all-latcons", "-tacpred", "-tacorc-constraints", "-tacorc-constraints-a50"). The argv difference to B is exactly `--code-mix kind` (and `--out`).
O's teacher forcing draws an independent GT-or-own switch per sample AND per unit (the decisions + the six constraint kinds) at the same p_gt schedule (1.0 -> 0.2), so each cell's expected
GT fraction is B's and only the joint changes; the S-level switch, T's losses, the windows and the 19,164 steps are B's.
Eval: `v8f_mini_eval.py run` on eval-141 (2,399 windows, 141 episodes), "T1-class open loop". The mechanism and HARM reads use T0 oracle evals (they read the GT) only to score O.

  PRIMARY   per seed s: delta_s = ADE6(X_s) - ADE6(B_s), own code, paired over episodes, on ALL 2,399 windows (deciding). Seed s PASSES iff delta_s <= -2F, separated, with
            2F = 0.2415 m COMMITTED in the PREREG (F = |mean ADE6 B_s0 - mean ADE6 B_s1| = 0.1207 m; reproduced as a known value, never recomputed into the bar).
  D-M       (mechanism) the alpha-dose response r(0.5) of PREREG_V8F_TAC Amendment 2 (C), on X, with TAC_A2's own definition: r(0.5) = d_cons(0.5) / d_cons(1), where d_cons(a) is
            ADE6 of the constraints moved a fraction a of the way from T's own code to the GT, minus ADE6 of T's own code (`--tac pred`), paired over episodes. B's committed values are
            0.36 / 0.21 (M2b s0 / s1, RESULT_V8F_TAC_A2.md); the banked evals REPRODUCE them here (a known-value control: a different B is not the PREREG's B). X's O LEARNED iff
            r_X(0.5) - r_B(0.5) >= 0.10, against the COMMITTED r_B. Beside the PRIMARY, deciding nothing alone: it names the next lever on a FAIL.
  HARM      the GT-code oracle ADE6 on X against B, paired. Beside; O now sees fewer all-GT codes. Not deployable, never a bar.
  STAGES    s0: seed 0 only. CONTINUE iff seed 0 PASSES the PRIMARY (only then is seed 1 trained); otherwise STOP with the seed-0 verdict. final: both seeds; REFUSED unless seed 0 passed.
  OUTCOMES  PASS (PRIMARY met at both seeds) | FAIL-O-LEARNED (PRIMARY missed, r(0.5) rose by >= 0.10 over B's) | FAIL-O-DID-NOT-LEARN (PRIMARY missed, r(0.5) rose by < 0.10).
            The PREREG's table defines the O-LEARNED / DID-NOT-LEARN split at seed 0, where the only miss that can occur at stage s0 happens. A seed-1 miss after a seed-0 pass has no cell in the table;
            the SAME bar is applied to the seed that missed (against B's committed r for that seed, 0.21) and the output says `table_cell: extension`. Nothing else is moved.
Reported beside it, deciding nothing: the four families, per family and never pooled (LON incl. distance keeping K = 10 from v8f_dk_read.read_dk, LAT, TACTICAL and STRATEGIC -- NOT PAUSED for
v8F-mini --, and LAMBDA, the latcons leak); T's present-slot RMSE by group; both arms' parameter counts, step counts and steps per second.
Refusals (rc 2): a missing file; runs not on one clip list / not withholding the RA-1 holdout / not done / different step counts / different window counts; an X run whose argv is not B's plus
`--code-mix kind`, whose launch-gate record (selfcheck.json: declared_vs_built PASS, G-LIVE pass with a frozen-prior control of exactly 0.0, the code_mix probe that read a PARTLY-GT code) is not a
PASS, whose built levers / checkpoint do not say code_mix kind, whose parameter counts differ from B's by anything (the lever adds no parameter), or whose built p_gt schedule / loss weights /
hmax differ from B's (key by key: where B's old record lacks a key nothing is compared, "absent" is never "different"); a checkpoint sha256 that is not the banked B's, or an X checkpoint whose
sha256 differs between the end of training, the end of the evals and the file pulled here; evals not on one grid of windows (eid, or the committed 2,399 / 141); an X eval that did not score
<runs>/M2b_s<seed>_mixo/ckpt_last.pt or whose manipulation record is not the arm's; GT codes that differ between the arms; an identity control that misses exactly +0 [0, 0]; a B whose
r(0.5) does not reproduce the PREREG's; a MISSING FAMILY (tactical / strategic in an eval json, longitudinal / lateral / distance keeping / LAMBDA in the read); a final read after a seed-0 miss.
ADDED GUARDS (same direction; they refuse, they never decide): every X tacpred eval reproduces its standard eval per window within 1e-3 m (PREREG_V8F_TAC's own reproducibility check); the
loop-logged p_gt equals B's schedule at every logged step.
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (EPISODES); it answers "another draw of EPISODES", and training-seed variance enters only through F (H-ESTIM-SEED-1). The rig's
run-to-run floor is the seed-paired swing (0.102 m ADE6, R66): an effect below it is not reported as a lever effect, whatever its interval. v8F-mini's planner is deterministic, so no
inference-seed floor applies."""
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
RAW = PKG / "raw"
LEVER_FLAG, LEVER_VAL = "--code-mix", "kind"
ARM_B = "M2b"
ARM_X = "M2b-mixo"
SUF = "_mixo"
BAR_2F = 0.2415                                  # PREREG_V8F_MIXO: 2F, committed (F = 0.1207 m); a bar, not a recomputation
RISE_BAR = 0.10                                  # PREREG_V8F_MIXO: O LEARNED iff r(0.5) rises by >= 0.10 over B's
R_B_COMMITTED = {0: 0.36, 1: 0.21}               # RESULT_V8F_TAC_A2.md, row "(C) alpha = 0.50", M2b s0 / s1
R_B_TOL = 0.005                                  # those are rounded to 2 dp: the banked evals must reproduce them within half a unit of the last place
ALPHA_HALF = 0.5
EVAL_WINDOWS, EVAL_EPISODES = 2399, 141          # the committed eval-141 grid
TOL_REPRO = 1e-3                                 # PREREG_V8F_TAC's reproducibility tolerance (tacpred vs the standard eval, per window, m)
B_P_GT = (1.0, 0.2)                              # B's schedule (the trainer's default for M2b), which X inherits unchanged
B_GRID, N_TOK_B = "pool4x10", 40
B = 2000
X_SUFFIXES = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons", "-tacpred", "-tacorc-constraints", "-tacorc-constraints-a50")
DOSE_SUFFIXES = ("-tacpred", "-tacorc-constraints", "-tacorc-constraints-a50")
FAMILIES = ("longitudinal", "lateral", "tactical", "strategic")
# The banked B checkpoints (raw/full/MANIFEST.md): the chain asserts them on box 2 and writes the observed values to the `--ckpt-sha` json.
BANKED_CKPT_SHA = {"M2b_s0": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8",
                   "M2b_s1": "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
# The committed next-step texts, VERBATIM from PREREG_V8F_MIXO.md's outcome table (a test compares them with the file).
NEXT_TEXT = {"PASS": "X becomes the base; KIND is re-read on X",
             "FAIL-O-LEARNED": "O now consumes partial codes, but T is too wrong where O reads it: the T-side arm on the kind KIND ranks first",
             "FAIL-O-DID-NOT-LEARN": "O cannot use a mixed code at this capacity or with these inputs: an O-capacity or O-input arm, under its own PREREG (an architecture change: PI go)"}
NEXT = {None: "CONTINUE: seed 0 passes the PRIMARY; train seed 1 (the verdict needs both).",
        "PASS": "PASS: O trained on per-kind mixed codes drives better than B beyond 2F at both seeds. Next: " + NEXT_TEXT["PASS"] + ".",
        "FAIL-O-LEARNED": "FAIL, O LEARNED: the PRIMARY missed but r(0.5) rose by >= 0.10 over B's. Next: " + NEXT_TEXT["FAIL-O-LEARNED"] + ".",
        "FAIL-O-DID-NOT-LEARN": "FAIL, O DID NOT LEARN: the PRIMARY missed and r(0.5) rose by < 0.10. Next: " + NEXT_TEXT["FAIL-O-DID-NOT-LEARN"] + "."}


def refuse(msg):
    print(f"MIXO READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# The programme's own instruments, IMPORTED (never copied): PREREG_V8F_GRID's reader brings Amendment 2's `w_diff` / `B_UNRECORDED_W` (and, through it, the PGT reader: run identity, eval loading, the
# GT-code check, the K = 10 lead block, the paired contrasts, LAMBDA). Loaded as PRIVATE instances; their refusals are routed through THIS reader's refuse (one message prefix, one rc).
GR = _load("v8f_grid_read_for_mixo", HERE.parent / "v8f_grid_read.py")
GR.refuse = refuse
PGT = GR.PGT
PGT.refuse = refuse
w_diff, B_UNRECORDED_W, _SKIP_LEVERS = GR.w_diff, GR.B_UNRECORDED_W, GR._SKIP_LEVERS
_SKIP_LEVERS_X = tuple(_SKIP_LEVERS) + ("code_mix",)       # the lever itself: B's record predates the key and is compared to "whole" where it carries one


def _modules():
    return PGT._modules()


def x_dir(s):
    return f"{ARM_B}_s{s}{SUF}"


def _jload(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        refuse(f"cannot read {p}: {type(e).__name__}: {e}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ runs
def b_hmax(rfl, rsm):
    """The hmax the trainer's make_config gives B: `--hmax` when set and non-zero, else its auto rule (16 when the arm has a world-model auxiliary, else 10)."""
    h = rfl.get("--hmax")
    if h not in (None, True) and int(h) != 0:
        return int(h)
    wm = (rsm.get("levers") or {}).get("wm")
    if wm is None:
        refuse("B's record names no wm lever: B's built hmax (the trainer's auto rule: 16 with a world-model auxiliary, else 10) cannot be derived")
    return 16 if wm != "off" else 10


def ckpt_facts(path):
    """The BUILT model, read from the checkpoint the trainer saved."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, mdl = ck.get("cfg") or {}, ck.get("model") or {}
    tm, pos = mdl.get("time"), mdl.get("pos")
    return {"code_mix": cfg.get("code_mix"), "hist": cfg.get("hist"), "hmax": cfg.get("hmax"), "window_start": cfg.get("window_start"), "d": cfg.get("d"), "n_tok": cfg.get("n_tok"),
            "p_gt_start": cfg.get("p_gt_start"), "p_gt_end": cfg.get("p_gt_end"), "grid": ck.get("grid"),
            "time_rows": None if tm is None else int(tm.shape[0]), "pos_rows": None if pos is None else int(pos.shape[0]),
            "argv_sha": ck.get("argv_sha"), "step": ck.get("step"), "arm": ck.get("arm"), "seed": ck.get("seed")}


def check_params(lv_b, lv_x, tag):
    """The lever adds NO parameter: every group and the total equal B's. Any delta is a second lever."""
    pb, px = lv_b.get("params") or {}, lv_x.get("params") or {}
    bad = []
    if not pb or not px:
        bad.append("a run's levers carry no per-group parameter counts")
    else:
        for k in sorted(set(pb) | set(px)):
            if px.get(k, 0) != pb.get(k, 0):
                bad.append(f"group {k}: {px.get(k)} vs B's {pb.get(k)}")
    if (lv_x.get("n_params_trainable") or 0) != (lv_b.get("n_params_trainable") or 0):
        bad.append(f"n_params_trainable {lv_x.get('n_params_trainable')} vs B's {lv_b.get('n_params_trainable')}")
    if bad:
        refuse(f"{tag}: X's parameter counts are not B's (the per-unit mix adds no parameter): a second lever moved -- " + "; ".join(bad))
    return {"n_params_trainable": {"B": lv_b.get("n_params_trainable"), "X": lv_x.get("n_params_trainable")}, "groups": {"B": pb, "X": px}}


def load_b_runs(runs_dir):
    """B's two seeds, validated by PGT's own run check for the M2b arm (flags, built schedule, one clip list, holdout, done, steps)."""
    return PGT.load_runs(runs_dir, (0, 1), arms=(ARM_B,))


def check_gate_token(tag, sm, sc, bad):
    """The launch-gate PASS token of a run: selfcheck.json of the in-process gate the trainer runs before it trains (PREREG: no launch-gate PASS token => rc 2)."""
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"{tag}: the launch gate's declared_vs_built is not PASS (no PASS token)")
    lv_live = sc.get("live") or {}
    if lv_live.get("pass") is not True or lv_live.get("control_prior_grad_sum") != 0.0:
        bad.append(f"{tag}: the launch gate's G-LIVE is not a pass with a frozen-prior control of exactly 0.0 (no PASS token)")
    pr = sc.get("code_mix_probe")
    if not isinstance(pr, dict) or pr.get("pass") is not True:
        bad.append(f"{tag}: selfcheck.json carries no passing code_mix probe (the reachability check of the per-unit mix): no PASS token")
    else:
        if pr.get("code_mix") != LEVER_VAL:
            bad.append(f"{tag}: the code_mix probe was run for {pr.get('code_mix')!r}, not {LEVER_VAL!r}")
        if not (isinstance(pr.get("n_partial"), int) and pr["n_partial"] > 0):
            bad.append(f"{tag}: the code_mix probe saw no PARTLY-GT code ({pr.get('n_partial')}): the analytic check of the mix missed")
    if sc.get("argv_sha") != sm.get("argv_sha"):
        bad.append(f"{tag}: selfcheck.json (argv_sha {sc.get('argv_sha')}) is not this run's (summary.json {sm.get('argv_sha')})")


def load_x_runs(runs_dir, seeds, SB):
    S, logs, bad, pipe, params, built = {}, {}, [], {}, {}, {}
    p0, p1 = B_P_GT
    for s in seeds:
        d = os.path.join(runs_dir, x_dir(s))
        for f in ("summary.json", "argv.json", "selfcheck.json", "ckpt_last.pt"):
            if not os.path.exists(os.path.join(d, f)):
                refuse(f"{os.path.join(d, f)} is missing (the run did not finish, or was not pulled)")
        S[s] = {f: _jload(os.path.join(d, f)) for f in ("summary.json", "argv.json", "selfcheck.json")}
    for s in seeds:
        tag = f"{ARM_X}_s{s}"
        sm, argv, sc = S[s]["summary.json"], S[s]["argv.json"].get("argv", []), S[s]["selfcheck.json"]
        fl = PGT._flags(argv)
        ref = SB[(ARM_B, s)]
        rsm, rfl = ref["summary.json"], PGT._flags(ref["argv.json"].get("argv", []))
        if sm.get("done") is not True:
            bad.append(f"{tag} is not done")
        if sm.get("arm") != "M2b" or sm.get("seed") != s:
            bad.append(f"{tag}/summary.json says {sm.get('arm')} seed {sm.get('seed')}, expected M2b seed {s}")
        if sm.get("n_clips") != PGT.N_TRAIN:
            bad.append(f"{tag}: n_clips {sm.get('n_clips')} != {PGT.N_TRAIN}")
        if sm.get("excluded_sha256") != PGT.HOLDOUT_SHA:
            bad.append(f"{tag} did not withhold the RA-1 holdout")
        if sm.get("train_sha256") != rsm.get("train_sha256"):
            bad.append(f"{tag} did not train on {ARM_B}_s{s}'s clip list")
        if sm.get("steps") != rsm.get("steps"):
            bad.append(f"{tag} trained for {sm.get('steps')} steps, {ARM_B}_s{s} for {rsm.get('steps')}")
        if sm.get("n_windows") != rsm.get("n_windows"):
            bad.append(f"{tag} trained on {sm.get('n_windows')} windows, {ARM_B}_s{s} on {rsm.get('n_windows')}: the window set is not B's")
        check_gate_token(tag, sm, sc, bad)
        if fl.get("--arm") != "M2b":
            bad.append(f"{tag}: argv --arm {fl.get('--arm')}, expected M2b")
        if argv.count(LEVER_FLAG) != 1 or fl.get(LEVER_FLAG) != LEVER_VAL:
            bad.append(f"{tag}: argv {LEVER_FLAG} {fl.get(LEVER_FLAG)} (named {argv.count(LEVER_FLAG)} times), expected {LEVER_VAL} exactly once")
        if LEVER_FLAG in rfl:
            bad.append(f"{ARM_B}_s{s}: argv names {LEVER_FLAG}: B is the banked M2b, which predates the flag")
        drop = {"--out", LEVER_FLAG}
        fx, fb = {k: x for k, x in fl.items() if k not in drop}, {k: x for k, x in rfl.items() if k not in drop}
        if fx != fb:
            diff = sorted(k for k in set(fx) | set(fb) if fx.get(k) != fb.get(k))
            bad.append(f"{tag}: argv differs from {ARM_B}_s{s}'s in {diff} besides {LEVER_FLAG} and --out -- not one lever")
        hm_b = b_hmax(rfl, rsm)
        for w, rec in (("summary", sm), ("selfcheck", sc)):
            lg = rec.get("levers") or {}
            if lg.get("code_mix") is None:
                bad.append(f"{tag}: {w}.json levers record no built code_mix: the lever is unproven")
            elif lg.get("code_mix") != LEVER_VAL:
                bad.append(f"{tag}: {w}.json built code_mix {lg.get('code_mix')!r}, expected {LEVER_VAL!r}")
            for key, want in (("p_gt_start", p0), ("p_gt_end", p1)):
                if lg.get(key) is None:
                    bad.append(f"{tag}: {w}.json levers record no built {key}: the inherited schedule is unproven")
                elif not PGT._same_float(lg[key], want):
                    bad.append(f"{tag} was built with {key} {lg[key]} ({w}.json), B's is {want}")
            if lg.get("hmax") is not None and lg["hmax"] != hm_b:
                bad.append(f"{tag}: {w}.json built hmax {lg['hmax']}, B's is {hm_b}")
            lb = rsm.get("levers") or {}
            diff = sorted(k for k in set(lg) & set(lb) if k not in _SKIP_LEVERS_X and lg[k] != lb[k])
            diff += [f"w.{k}" for k in w_diff(lg.get("w") or {}, lb.get("w") or {}, rfl)]
            if diff:
                bad.append(f"{tag}: {w}.json built levers {diff} differ from {ARM_B}_s{s}'s")
        if (rsm.get("levers") or {}).get("code_mix", "whole") != "whole":
            bad.append(f"{ARM_B}_s{s}: summary.json built code_mix {rsm['levers']['code_mix']!r}: B is not the whole-code run the lever is defined against")
        d_model = int(fl.get("--d", 256))
        pc = {w: check_params(rsm.get("levers") or {}, rec.get("levers") or {}, f"{tag} ({w}.json)") for w, rec in (("summary", sm), ("selfcheck", sc))}
        params[s] = pc["summary"]
        ck = ckpt_facts(os.path.join(runs_dir, x_dir(s), "ckpt_last.pt"))
        if ck["code_mix"] != LEVER_VAL:
            bad.append(f"{tag}: the checkpoint was BUILT with code_mix {ck['code_mix']!r}, not {LEVER_VAL!r}")
        if ck["hist"] is not None and ck["hist"] != 4:
            bad.append(f"{tag}: the checkpoint was BUILT at hist {ck['hist']}, B's is 4")
        if ck["hmax"] != hm_b:
            bad.append(f"{tag}: the checkpoint was BUILT with hmax {ck['hmax']}, B's is {hm_b}")
        if ck["window_start"] not in (None, -1):
            bad.append(f"{tag}: the checkpoint was BUILT with window_start {ck['window_start']}: B's is absent (-1)")
        for key, want in (("p_gt_start", p0), ("p_gt_end", p1)):
            if ck[key] is not None and not PGT._same_float(ck[key], want):
                bad.append(f"{tag}: the checkpoint was BUILT with {key} {ck[key]}, B's is {want}")
        if ck["grid"] != B_GRID or ck["n_tok"] != N_TOK_B or ck["pos_rows"] != N_TOK_B:
            bad.append(f"{tag}: the checkpoint's grid {ck['grid']!r} / n_tok {ck['n_tok']} / pos rows {ck['pos_rows']} are not B's ({B_GRID!r} / {N_TOK_B} / {N_TOK_B})")
        if ck["argv_sha"] != sm.get("argv_sha"):
            bad.append(f"{tag}: the checkpoint was written by another argv (argv_sha {ck['argv_sha']} vs summary.json {sm.get('argv_sha')})")
        if ck["step"] != sm.get("steps"):
            bad.append(f"{tag}: the checkpoint is at step {ck['step']}, the run's {sm.get('steps')}")
        if ck["arm"] != "M2b" or ck["seed"] != s:
            bad.append(f"{tag}: the checkpoint says arm {ck['arm']} seed {ck['seed']}")
        built[s] = {"code_mix": ck["code_mix"], "hmax": ck["hmax"], "b_hmax": hm_b, "d": d_model}
        logs[s] = PGT._train_log_check(os.path.join(runs_dir, x_dir(s), "train_log.jsonl"), ARM_X, s, sm.get("steps"), p0, p1, True)
        pipe[s] = {"X": {"lru": (sc.get("info") or {}).get("lru"), "seconds": sm.get("seconds"), "steps": sm.get("steps"), "n_windows": sm.get("n_windows"), "S": sm.get("S"),
                         "steps_per_s": (sm["steps"] / sm["seconds"]) if sm.get("seconds") else None, "ckpt": ck},
                   "B": {"lru": (ref["selfcheck.json"].get("info") or {}).get("lru"), "seconds": rsm.get("seconds"), "steps": rsm.get("steps"),
                         "n_windows": rsm.get("n_windows"), "S": rsm.get("S"), "steps_per_s": (rsm["steps"] / rsm["seconds"]) if rsm.get("seconds") else None}}
    if bad:
        refuse("; ".join(bad))
    return S, logs, pipe, params, built


def check_ckpt_sha(path, runs_dir, seeds):
    """The checkpoints ARE the ones the PREREG names: B's two banked checkpoints (pinned here from raw/full/MANIFEST.md) as the chain measured them on box 2, and each X checkpoint the same at the end
    of training and at the end of its evals AND equal to the file pulled into --runs-dir (the checkpoint the reader inspected is the one that was scored)."""
    if not path:
        refuse("--ckpt-sha is required: the chain's json of the checkpoint sha256 it measured on box 2")
    if not os.path.exists(path):
        refuse(f"--ckpt-sha {path} does not exist")
    got = _jload(path)
    out = {}
    for tag, want in BANKED_CKPT_SHA.items():
        have = got.get(tag)
        if have != want:
            refuse(f"the checkpoint {tag} has sha256 {str(have)[:16]}..., the banked run's is {want[:16]}...: this is not the banked checkpoint")
        out[tag] = have
    for s in seeds:
        t = f"{ARM_X}_s{s}"
        a1, a2 = got.get(f"{t}_trained"), got.get(f"{t}_after_evals")
        loc = sha256_file(os.path.join(runs_dir, x_dir(s), "ckpt_last.pt"))
        if not (a1 and a2) or a1 != a2:
            refuse(f"{t}: the checkpoint sha256 at the end of training ({str(a1)[:16]}) and after the evals ({str(a2)[:16]}) are not one value: the scored checkpoint is not the trained one")
        if a1 != loc:
            refuse(f"{t}: the checkpoint pulled into {runs_dir} has sha256 {loc[:16]}..., box 2 measured {str(a1)[:16]}...: not the scored checkpoint")
        out[t] = a1
    return out


def check_eval_ckpts(eval_dir, seeds):
    """Every X eval names the mixo checkpoint it scored (the eval json's `ckpt`): an eval of B's checkpoint under X's tag is refused."""
    bad = []
    for s in seeds:
        for suf in X_SUFFIXES:
            p = os.path.join(eval_dir, f"eval_{ARM_X}_s{s}{suf}.json")
            if not os.path.exists(p):
                bad.append(f"{os.path.basename(p)} is missing")
                continue
            ck = _jload(p).get("ckpt")
            parts = str(ck or "").replace("\\", "/").rstrip("/").split("/")
            if len(parts) < 2 or parts[-2] != x_dir(s) or parts[-1] != "ckpt_last.pt":
                bad.append(f"{os.path.basename(p)} scored {ck!r}, not <runs>/{x_dir(s)}/ckpt_last.pt")
    if bad:
        refuse("; ".join(bad))


def check_counts(eid, a):
    n, ne = int(eid.size), int(np.unique(eid).size)
    if (n, ne) != (a.n_windows, a.n_episodes):
        refuse(f"the evals hold {n} windows over {ne} episodes, the committed eval-141 grid has {a.n_windows} over {a.n_episodes}")
    return n, ne


# ------------------------------------------------------------------------------------------------ the dose chain (D-M)
def _manip_ok(js, want_groups, want_alpha, alpha_optional=False):
    """The manipulation record is the arm's: an oracle on the constraints (26 code dims) at the wanted alpha. The Amendment-1 `constraints` file (B's alpha = 1 arm) predates the alpha record, so there
    an UNRECORDED alpha is accepted (PREREG_V8F_TAC's own reader does the same: `want[2] is None`), but a recorded one must equal 1."""
    m = js.get("tac_manipulation") or {}
    alpha = m.get("alpha")
    a_ok = (alpha_optional and alpha is None) or (alpha is not None and abs(float(alpha) - want_alpha) <= 1e-9)
    return (m.get("mode") == "oracle" and m.get("groups") == want_groups and m.get("n_code_dims_replaced") == 26 and a_ok), m


def load_dose(eval_dir, arm, s, eid):
    """One (arm, seed)'s three dose arrays: T's own code (tacpred), the constraints oracle at alpha 1 and at alpha 0.50 (PREREG_V8F_TAC Amendments 1 and 2). Every file on the eval grid; the
    oracle arms' manipulation records are the arm's; the two oracle arms agree about which windows have no v9 row."""
    t = f"{arm}_s{s}"
    out, fbs = {}, []
    for key, suf, want in (("pred", "-tacpred", None), ("full", "-tacorc-constraints", 1.0), ("half", "-tacorc-constraints-a50", ALPHA_HALF)):
        pz, pj = os.path.join(eval_dir, f"eval_{t}{suf}.npz"), os.path.join(eval_dir, f"eval_{t}{suf}.json")
        if not os.path.exists(pz):
            refuse(f"eval_{t}{suf}.npz is missing from {eval_dir}")
        z = np.load(pz)
        for k in ("eid", "ade6"):
            if k not in z.files:
                refuse(f"eval_{t}{suf}.npz carries no {k}")
        if not np.array_equal(np.asarray(z["eid"]), np.asarray(eid)):
            refuse(f"eval_{t}{suf}.npz is not on the same eval grid as the others (eid differs)")
        ade = np.asarray(z["ade6"], np.float64)
        if not np.isfinite(ade).all():
            refuse(f"eval_{t}{suf}.npz holds a non-finite ade6")
        if want is not None:
            if not os.path.exists(pj):
                refuse(f"eval_{t}{suf}.json is missing (the manipulation record)")
            ok, m = _manip_ok(_jload(pj), "constraints", want, alpha_optional=(key == "full"))
            if not ok:
                refuse(f"eval_{t}{suf}.json: manipulation {m.get('mode')} / {m.get('groups')} / {m.get('n_code_dims_replaced')} dims / alpha {m.get('alpha')}, "
                       f"expected oracle / constraints / 26 dims / alpha {want}")
            if "tac_oracle_fallback" in z.files:                   # a consistency cross-check only: an older file without the array is not refused for lacking a cross-check
                fbs.append(np.asarray(z["tac_oracle_fallback"], bool))
        elif os.path.exists(pj):
            js = _jload(pj)
            if js.get("tac", "pred") != "pred" or js.get("tac_manipulation"):
                refuse(f"eval_{t}{suf}.json is not T's own code (tac {js.get('tac')!r}, manipulation {js.get('tac_manipulation')!r}): the dose chain's base needs `--tac pred`")
        out[key] = ade
    if len(fbs) == 2 and not np.array_equal(fbs[0], fbs[1]):
        refuse(f"the constraints oracles of {t} disagree about which windows have no v9 row")
    return out


def check_repro(dose, std_ade, tag):
    """PREREG_V8F_TAC's reproducibility check: the tacpred run reproduces the standard eval per window (same weights, same windows)."""
    dev = float(np.abs(dose["pred"] - np.asarray(std_ade, np.float64)).max())
    if dev > TOL_REPRO:
        refuse(f"{tag}: the tacpred run differs from the standard eval by {dev:.3g} m per window (> {TOL_REPRO}): the dose chain's base is not the arm's own code")
    return dev


def dose_stats(pf, dose, eid, n_boot):
    """r(0.5) = d_cons(0.5) / d_cons(1), both paired against T's own code (TAC_A2's definition). `r` uses the UNROUNDED differences; `r_tac_a2` the library's 4-dp-rounded deltas exactly as
    v8f_tac_groups2_read does (beside, to show the two agree). r is None where d_cons(1) is not an improvement (nothing to take a fraction of)."""
    d1 = PGT.paired(pf, dose["full"], dose["pred"], eid, n_boot)
    dh = PGT.paired(pf, dose["half"], dose["pred"], eid, n_boot)
    ok = d1["point"] < 0
    return {"full": d1, "half": dh, "r": float(dh["point"] / d1["point"]) if ok else None,
            "r_tac_a2": float(dh["delta"] / d1["delta"]) if (ok and d1["delta"] != 0) else None,
            "r_defined": bool(ok), "ade6_pred": float(dose["pred"].mean())}


def dm_row(pf, doseX, doseB, eid, s, n_boot):
    """D-M at one seed: r_X(0.5) against B's COMMITTED r (0.36 / 0.21); O LEARNED iff the rise is >= 0.10. An undefined r_X (the full constraints oracle does not help X) is no rise."""
    xs, bs = dose_stats(pf, doseX, eid, n_boot), dose_stats(pf, doseB, eid, n_boot)
    rb = R_B_COMMITTED[s]
    if bs["r"] is None or abs(bs["r"] - rb) > R_B_TOL:
        refuse(f"B's banked evals read r(0.5) = {bs['r']} at seed {s}, the PREREG's (RESULT_V8F_TAC_A2.md) is {rb}: not the inputs the PREREG was written against")
    rise = None if xs["r"] is None else round(xs["r"] - rb, 12)
    learned = bool(rise is not None and rise >= RISE_BAR)
    return {"r_X": xs["r"], "r_B_committed": rb, "r_B_measured": bs["r"], "rise_over_committed": rise, "rise_over_measured": None if xs["r"] is None else xs["r"] - bs["r"],
            "bar": RISE_BAR, "learned": learned, "r_defined": xs["r_defined"], "X": xs, "B": bs}


# ------------------------------------------------------------------------------------------------ the decision
def judge(row):
    """The committed bars. PRIMARY: delta <= -2F (2F = 0.2415 m COMMITTED), separated, delta < 0. D-M: r_X(0.5) - r_B(0.5) >= 0.10."""
    p = row["primary"]
    return {"passed": bool(p["point"] <= -BAR_2F and p["separated"] and p["point"] < 0), "bar_2F_m": BAR_2F,
            "dm": {"learned": row["dm"]["learned"], "rise": row["dm"]["rise_over_committed"], "bar": RISE_BAR}}


def verdict(stage, rows, seeds):
    """s0: CONTINUE on a seed-0 PASS, else STOP with the seed-0 outcome. final: PASS iff every seed passed, else FAIL split by D-M at the seed(s) that MISSED."""
    passed = {s: rows[f"s{s}"]["judge"]["passed"] for s in seeds}
    if stage == "s0":
        if passed[0]:
            return "CONTINUE", None, "n/a"
        return "STOP", ("FAIL-O-LEARNED" if rows["s0"]["judge"]["dm"]["learned"] else "FAIL-O-DID-NOT-LEARN"), "defined"
    if all(passed.values()):
        return None, "PASS", "n/a"
    missed = [s for s in seeds if not passed[s]]
    learned = all(rows[f"s{s}"]["judge"]["dm"]["learned"] for s in missed)
    return None, ("FAIL-O-LEARNED" if learned else "FAIL-O-DID-NOT-LEARN"), ("defined" if missed == [0] else "extension")


# ------------------------------------------------------------------------------------------------ families (a missing family is refused)
def check_families(row, s, famfn_used):
    b = row["beside"]
    miss = []
    for arm in ("B", "X"):
        ej = b["eval_json"].get(arm) or {}
        for fam in ("tactical", "strategic"):
            if not ej.get(fam):
                miss.append(f"{arm}'s eval json has no {fam} family")
    dk = b["distance_keeping_k10"]
    if not (dk.get("pair_vs_b") and dk.get("arms")):
        miss.append("the K = 10 distance-keeping read has no X - B pair")
    if not (row.get("mleak") and row["mleak"].get("delta")):
        miss.append("LAMBDA (the latcons leak) is missing")
    if famfn_used:
        fam = b.get("families") or {}
        for f in ("longitudinal", "lateral"):
            if not fam.get(f):
                miss.append(f"the paired {f} family block is empty")
    if miss:
        refuse(f"seed {s}: a family is missing -- " + "; ".join(miss) + " (a missing family is refused, never dropped)")


# ------------------------------------------------------------------------------------------------ the read
def identity_control(pf, Zb_by_seed, pw, ctx, n_boot, WC, seeds, Zrel, fam, doseB):
    """PGT's identity control (PRIMARY, M-LEAK, M-DK, present-slot, oracle ADE6, every family cell: B against itself reads exactly +0 [0, 0]) plus this PREREG's own cells, the dose contrasts."""
    out = PGT.identity_control(pf, Zb_by_seed, pw, ctx, n_boot, WC, seeds, fam=fam)
    eid = ctx["eid"]
    for s in seeds:
        cells = {"dose_half": PGT.paired(pf, doseB[s]["half"], doseB[s]["half"], eid, n_boot), "dose_full": PGT.paired(pf, doseB[s]["full"], doseB[s]["full"], eid, n_boot)}
        for name, v in cells.items():
            if not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 and v["point"] == 0.0):
                refuse(f"IDENTITY CONTROL FAILED: B_s{s} - B_s{s} reads {v['delta']} [{v['lo']}, {v['hi']}] on {name}, not +0 [0, 0]: the paired instrument is broken")
            out[f"s{s}"][name] = {x: v[x] for x in ("delta", "lo", "hi", "point")}
    return out


def _common(a, WC):
    SB, logsB = load_b_runs(a.runs_dir)
    Zb, eid = PGT.load_all(a.eval_dir, (0, 1), (ARM_B,), WC)
    return SB, logsB, Zb, eid


def read(a, pairfn, famfn, DK, WC):
    seeds = (0,) if a.stage == "s0" else (0, 1)
    X = ARM_X
    SB, logsB, Zb, eid = _common(a, WC)
    SX, logsX, pipe, params, built = load_x_runs(a.runs_dir, seeds, SB)
    Zx, eid_x = PGT.load_all(a.eval_dir, seeds, (X,), WC)
    if not np.array_equal(np.asarray(eid_x), np.asarray(eid)):
        refuse(f"the {X} evals are not on the same eval grid as {ARM_B}'s (eid differs)")
    n_win, n_ep = check_counts(eid, a)
    check_eval_ckpts(a.eval_dir, seeds)
    sha = check_ckpt_sha(a.ckpt_sha, a.runs_dir, seeds)
    Zrel = {**Zb, **Zx}
    PGT.check_gt_codes(Zrel, seeds, WC)
    doseB = {s: load_dose(a.eval_dir, ARM_B, s, eid) for s in (0, 1)}
    doseX = {s: load_dose(a.eval_dir, X, s, eid) for s in seeds}
    repro = {f"{ARM_B}_s{s}": check_repro(doseB[s], Zb[(ARM_B, s)]["ade6"], f"{ARM_B}_s{s}") for s in (0, 1)}
    repro.update({f"{X}_s{s}": check_repro(doseX[s], Zx[(X, s)]["ade6"], f"{X}_s{s}") for s in seeds})
    F = abs(float(Zb[(ARM_B, 0)]["ade6"].mean()) - float(Zb[(ARM_B, 1)]["ade6"].mean()))
    ctx = PGT.build_ctx(a, DK, eid)
    Zdk = {**Zb, **Zx}
    pairs = [(f"{X}_s{s}", f"{ARM_B}_s{s}") for s in seeds] + [(f"{X}_s{s}", "P10") for s in seeds] + [(f"{ARM_B}_s{s}", "P10") for s in (0, 1)]
    res10 = PGT.dk_read(a, DK, ctx, Zdk, ctx["blk10"], 10, a.dk_n_boot, pairs)
    pw = res10["perwindow"]
    if not np.array_equal(np.asarray(pw["is_lead"], bool), ctx["lead"]):
        refuse("the LEAD windows of the distance-keeping read are not the lead block's join")
    base_st = {s: PGT.baseline_stats(pairfn, Zb[(ARM_B, s)], pw, ctx, a.n_boot) for s in (0, 1)}
    known = PGT.known_values(base_st, F, (0, 1))
    idc = identity_control(pairfn, {s: Zb[(ARM_B, s)] for s in seeds}, pw, ctx, a.n_boot, WC, seeds, Zrel, famfn, doseB)
    pe, _ = WC.present_error(Zrel, seeds)
    rows = {}
    for s in seeds:
        A, Bm = Zx[(X, s)], Zb[(ARM_B, s)]
        row = PGT.contrast(pairfn, A, Bm, pw, ctx, a.n_boot)
        row["base"] = {k: base_st[s][k] for k in ("lambda_m2b_m", "lambda_ci", "n_lead", "excess_p10", "excess_ci")}
        row["dm"] = dm_row(pairfn, doseX[s], doseB[s], eid, s, a.n_boot)
        row["judge"] = judge(row)
        ok = np.isfinite(pe[(ARM_B, s)]["mse"])
        row["beside"] = {
            "ade6": {"B": float(Bm["ade6"].mean()), "X": float(A["ade6"].mean())},
            "harm": {"oracle_ade6": {"B": float(Bm["orc_ade6"].mean()), "X": float(A["orc_ade6"].mean())},
                     "oracle_ade6_delta": PGT.paired(pairfn, A["orc_ade6"], Bm["orc_ade6"], eid, a.n_boot),
                     "note": "T0: both read the GT code; O has seen fewer all-GT codes under X. Not deployable, never a bar."},
            "present_slot": {"n_windows_present": int(ok.sum()), "rmse": {"B": float(np.sqrt(pe[(ARM_B, s)]["mse"][ok].mean())), "X": float(np.sqrt(pe[(X, s)]["mse"][ok].mean()))},
                             "rmse_group": {"B": pe[(ARM_B, s)]["rmse_group"], "X": pe[(X, s)]["rmse_group"]},
                             "mse_delta": PGT.paired(pairfn, pe[(X, s)]["mse"][ok], pe[(ARM_B, s)]["mse"][ok], eid[ok], a.n_boot)},
            "eval_json": {k: {kk: Z[(nm, s)]["json"].get(kk) for kk in ("traj", "families", "strategic", "tactical") if kk in Z[(nm, s)]["json"]}
                          for k, (nm, Z) in {"B": (ARM_B, Zb), "X": (X, Zx)}.items()},
            "distance_keeping_k10": {"pair_vs_b": res10["read"]["pairs"].get(f"{X}_s{s}__minus__{ARM_B}_s{s}"), "pair_vs_p10": res10["read"]["pairs"].get(f"{X}_s{s}__minus__P10"),
                                     "arms": {nm: res10["read"]["arms"][nm] for nm in (f"{X}_s{s}", f"{ARM_B}_s{s}", "P10")}},
            "parameters": params[s], "built": built[s], "pipeline": {k: {kk: vv for kk, vv in v.items() if kk != "ckpt"} for k, v in pipe[s].items()},
            "tacpred_repro_max_dev_m": {"B": repro[f"{ARM_B}_s{s}"], "X": repro[f"{X}_s{s}"]},
            "gt_source": {"B": Bm["gt_source"], "X": A["gt_source"]}}
        if famfn is not None:
            row["beside"]["families"] = famfn(A["pred"], Bm["pred"], eid)
        else:
            row["beside"]["families"] = {"status": "SKIPPED (--no-families, a TEST input; the chain never passes it)"}
        check_families(row, s, famfn is not None)
        rows[f"s{s}"] = row
    if a.stage == "final" and not rows["s0"]["judge"]["passed"]:
        refuse("--stage final, but seed 0 did NOT pass the PRIMARY: PREREG_V8F_MIXO runs seed 1 only after a seed-0 PASS, so a seed-1 run exists against the protocol")
    k30 = None
    if not a.no_k30:
        p30 = a.lead_block_k30 or str(DK.LEAD_BLOCK_K30)
        if os.path.exists(p30):
            k30 = PGT._trim_dk(PGT.dk_read(a, DK, ctx, Zdk, DK.load_block(p30), 30, a.dk_n_boot, pairs)["read"])
    decision, vd, cell = verdict(a.stage, rows, seeds)
    nxt = NEXT[None if decision == "CONTINUE" else vd]
    if cell == "extension":
        nxt += " (The PREREG's table defines the O-LEARNED / DID-NOT-LEARN split at seed 0; this is a seed-1 miss after a seed-0 pass, classified with the same bar against B's committed r(0.5) = 0.21: table_cell extension.)"
    fmt = lambda f: " / ".join(f(rows[f"s{s}"]) for s in seeds)           # noqa: E731
    prim = fmt(lambda r: f"{r['primary']['point']:+.3f}{'' if r['primary']['separated'] else ' ns'}")
    rr = fmt(lambda r: "undefined" if r["dm"]["r_X"] is None else f"{r['dm']['r_X']:.2f}")
    rb = fmt(lambda r: f"{r['dm']['r_B_committed']:.2f}")
    harm = fmt(lambda r: f"{r['beside']['harm']['oracle_ade6_delta']['point']:+.3f}")
    out = dict(schema="v8f-mixo-read-v1", prereg=f"PREREG_V8F_MIXO.md stage {a.stage}", base=ARM_B,
               tier="T1-class open loop (eval-141, 2,399 windows); the D-M and HARM reads use T0 oracle evals",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); F = |{ARM_B}_s0 - {ARM_B}_s1| = {F:.4f} m (B's own two seeds); bar 2F = {BAR_2F} m COMMITTED; "
                         f"answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)",
               n_windows=n_win, n_episodes=n_ep, n_lead_windows_k10=int(ctx["lead"].sum()), F=F, two_F_measured=2 * F, two_F_bar=BAR_2F, known_values=known,
               checkpoint_sha256=sha, identity_control=idc, train_log={f"{k}": v for k, v in logsX.items()},
               strategic_family="NOT PAUSED: un-paused for v8F-mini (PI ruling 2026-10-17); read from each run's eval json under rows.*.beside.eval_json",
               distance_keeping_k30=k30, stage=a.stage, rows=rows, DECISION=decision, VERDICT=vd, table_cell=cell, next=nxt)
    out["reading"] = (f"PREREG_V8F_MIXO stage {a.stage}: {decision or vd}; ADE6 X - B {prim} m (bar -2F = -{BAR_2F}); D-M r_X(0.5) {rr} against B's {rb} (bar +{RISE_BAR:.2f}); "
                      f"HARM (GT-code oracle, X - B) {harm} m." + ("" if vd is None else f" VERDICT {vd}."))
    return out


def check_inputs(a, DK, WC, pairfn, famfn=None):
    """The B side alone, before any training: PGT's banked-side check (the runs and evals of BOTH seeds load, the K = 10 lead context builds, F / LAMBDA / S5 known values reproduce, the paired
    instrument reads exactly 0 on M2b against itself) plus this PREREG's: B's dose files load and reproduce r(0.5) = 0.36 / 0.21, tacpred reproduces the standard eval, the grid is the committed
    2,399 / 141, the dose contrasts read exactly +0 [0, 0] on B against itself, and B's checkpoints are the banked ones when the chain's sha json is given."""
    banked = PGT.check_inputs(a, DK, WC, pairfn, famfn)
    Zb, eid = PGT.load_all(a.eval_dir, (0, 1), (ARM_B,), WC)
    n_win, n_ep = check_counts(eid, a)
    doseB = {s: load_dose(a.eval_dir, ARM_B, s, eid) for s in (0, 1)}
    repro = {f"{ARM_B}_s{s}": check_repro(doseB[s], Zb[(ARM_B, s)]["ade6"], f"{ARM_B}_s{s}") for s in (0, 1)}
    r = {}
    for s in (0, 1):
        st = dose_stats(pairfn, doseB[s], eid, a.n_boot)
        if st["r"] is None or abs(st["r"] - R_B_COMMITTED[s]) > R_B_TOL:
            refuse(f"B's banked evals read r(0.5) = {st['r']} at seed {s}, the PREREG's (RESULT_V8F_TAC_A2.md) is {R_B_COMMITTED[s]}: not the inputs the PREREG was written against")
        z = {"half": PGT.paired(pairfn, doseB[s]["half"], doseB[s]["half"], eid, a.n_boot), "full": PGT.paired(pairfn, doseB[s]["full"], doseB[s]["full"], eid, a.n_boot)}
        for name, v in z.items():
            if not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 and v["point"] == 0.0):
                refuse(f"IDENTITY CONTROL FAILED: B_s{s} - B_s{s} reads {v['delta']} [{v['lo']}, {v['hi']}] on dose_{name}, not +0 [0, 0]: the paired instrument is broken")
        r[f"s{s}"] = {"r_half": st["r"], "r_tac_a2": st["r_tac_a2"], "committed": R_B_COMMITTED[s], "d_cons_full": st["full"]["point"], "d_cons_half": st["half"]["point"]}
    F = abs(float(Zb[(ARM_B, 0)]["ade6"].mean()) - float(Zb[(ARM_B, 1)]["ade6"].mean()))
    out = {"base": ARM_B, "banked": banked, "n_windows": n_win, "n_episodes": n_ep, "F": F, "two_F_bar": BAR_2F, "dose_b": r, "tacpred_repro_max_dev_m": repro,
           "expected_x_suffixes": list(X_SUFFIXES), "expected_evals_per_seed": len(X_SUFFIXES), "banked_ckpt_sha256": BANKED_CKPT_SHA}
    if a.ckpt_sha:
        out["checkpoint_sha256"] = check_ckpt_sha(a.ckpt_sha, a.runs_dir, ())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("s0", "final"))
    ap.add_argument("--check-inputs", action="store_true", help="validate the B side alone (and the identity control) and exit")
    ap.add_argument("--eval-dir")
    ap.add_argument("--runs-dir")
    ap.add_argument("--out")
    ap.add_argument("--ckpt-sha", help="the chain's json {M2b_s0, M2b_s1, M2b-mixo_s<seed>_trained, M2b-mixo_s<seed>_after_evals: sha256} measured on box 2: required by the read")
    ap.add_argument("--n-boot", type=int, default=B, help="B of the deciding contrasts (default 2,000)")
    ap.add_argument("--dk-n-boot", type=int, default=B, help="B of the descriptive distance-keeping blocks (read_dk)")
    ap.add_argument("--no-families", action="store_true", help="TEST INPUT: skip the paired LON / LAT family blocks (the chain never passes it)")
    ap.add_argument("--no-k30", action="store_true", help="skip the 6 s distance-keeping block (descriptive only)")
    ap.add_argument("--n-windows", type=int, default=EVAL_WINDOWS, help="the committed eval grid's window count")
    ap.add_argument("--n-episodes", type=int, default=EVAL_EPISODES, help="the committed eval grid's episode count")
    ap.add_argument("--lead-block", help="the K = 10 B1 lead block (default: the banked 2026-09-02 block)")
    ap.add_argument("--lead-block-k30", help="the K = 30 block (default: raw/2026-10-22-dk/b1_eval_lead_block_k30.npz when present)")
    ap.add_argument("--p10-pred", default=None, help="P10-yaw-4512 eval predictions (default: the repo's)")
    ap.add_argument("--eval-grid", default=str(RAW / "eval_grid_windows.json"))
    ap.add_argument("--eval-kin", default=str(RAW / "eval141_kin.npz"))
    ap.add_argument("--eval-join", default=str(RAW / "join_eval141.npz"))
    ap.add_argument("--gt-npz", help="TEST INPUT: npz with gt, eid, clip, frame, v0, dt in place of the rebuild")
    a = ap.parse_args(argv)
    try:
        for need in ("eval_dir", "runs_dir", "out"):
            if getattr(a, need) is None:
                ap.error(f"--{need.replace('_', '-')} is required")
        DK, WC = _modules()
        a.p10_pred = a.p10_pred or str(DK.P10_DEFAULT)
        from taniteval.ci import paired_episode_cluster_bootstrap as pairfn
        if a.check_inputs:
            famfn = None if a.no_families else PGT._famfn(a)
            res = check_inputs(a, DK, WC, pairfn, famfn)
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(DK._clean(res), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            b = res["banked"]
            print(f"MIXO INPUTS OK: base {ARM_B}; F {res['F']:.4f} (2F bar {BAR_2F}); LAMBDA(B) {b['lambda_m2b'][0]:+.4f} / {b['lambda_m2b'][1]:+.4f}; r_B(0.5) "
                  f"{res['dose_b']['s0']['r_half']:.4f} / {res['dose_b']['s1']['r_half']:.4f} (committed {R_B_COMMITTED[0]} / {R_B_COMMITTED[1]}); {res['n_windows']} windows over {res['n_episodes']} episodes; "
                  f"expects {res['expected_evals_per_seed']} evals per seed; identity control exactly +0 [0, 0]")
            return 0
        if a.stage is None:
            ap.error("--stage is required unless --check-inputs")
        res = read(a, pairfn, None if a.no_families else PGT._famfn(a), DK, WC)
    except (OSError, KeyError, ValueError) as e:
        print(f"MIXO READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    except Exception as e:                           # an inherited instrument's OWN Refusal (v8f_dk_read.Refusal): still a refusal, rc 2, never a traceback
        if type(e).__name__ != "Refusal":
            raise
        print(f"MIXO READ REFUSED: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(DK._clean(res), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
