"""PREREG_V8F_FLAT, committed before its data: does v8F-mini's hierarchy beat its own capacity-matched FLAT control?  FLAT = M2b + `--o-interface cut` (ONE lever).

  python v8f_flat_read.py --eval-dir <eval> --runs-dir <runs> --ckpt-sha ckpt_sha256.json --out flat_read.json
  python v8f_flat_read.py --check-inputs --eval-dir <eval> --runs-dir <runs> --out x.json      (the banked B side alone, before any training)

B = M2b (fixed in the PREREG), the banked runs `M2b_s{0,1}` and their evals. X = FLAT = B + `--o-interface cut` and nothing else (run dir `M2b_s<SEED>_flat`; eval tags `M2b-flat_s<SEED>`,
suffixes "", "-tacabs", "-tacorc", "-tacorc-all-latcons"). The argv difference to B is exactly `--o-interface cut` (and `--out`). O reads a CONSTANT in place of T's code (zeros, after every
override) and of S's checkpoints (the nominal arcs, ok = 0); S and T are still built, trained on their own losses and run, and T still reads S. O keeps every parameter and every other
conditioning token (ego, nav, limit, age, the scene). Data, the 19,164 steps, the p_gt schedule, the WM auxiliary, the windows and the eval are B's.
Eval: `v8f_mini_eval.py run` on eval-141 (2,399 windows, 141 episodes), "T1-class open loop". Both seeds are ALWAYS trained and read: a hierarchy claim on one seed is what H-ESTIM-SEED-1 forbids.

  PRIMARY   per seed s: Delta_s = ADE6(B_s) - ADE6(FLAT_s), own code, paired over episodes, on ALL 2,399 windows (NEGATIVE = the hierarchy is better), with 2F = 0.2415 m COMMITTED in the PREREG
            (F = |mean ADE6 B_s0 - mean ADE6 B_s1| = 0.1207 m; reproduced as a known value, never recomputed into the bar). A seed reads
              LO   iff Delta_s <= -2F and the interval is separated below 0      HI   iff Delta_s >= +2F and the interval is separated above 0      MID  otherwise.
  OUTCOMES  the PREREG's table, verbatim:
              HIERARCHY BEATS FLAT  (LO, LO)                                  -> register C-V8F-HIER-1; the interface-quality arm (MIXO's outcome, then T-side) is built on it
              NOT ESTABLISHED       either seed misses -2F or is not separated, and neither seed reads >= +2F (HI) -> v8F is not validated against flat at 2F (said plainly)
              FLAT BETTER           (HI, HI)                                  -> an O-robustness arm (code dropout or MIXO-type mixing toward an empty code) under its own PREREG
              SEEDS DISAGREE        one seed LO, the other HI                 -> seed 2 of BOTH arms (B has no seed 2 banked), 2 x B's wall-clock on box 2
              FLAT LEANING          one seed HI, the other MID                -> reported to the PI as "v8F not validated against flat"; next is FLAT BETTER's committed arm
            AMENDMENT 1 (PREREG_V8F_FLAT.md, 2026-10-25, written before any FLAT data) added the last row. In its H / F / M letters (H = LO = Delta <= -2F separated, F = HI = Delta >= +2F
            separated, M = MID = anything else) the partition is COMPLETE: (H,H) HIERARCHY BEATS FLAT; (F,F) FLAT BETTER; (H,F) / (F,H) SEEDS DISAGREE; (F,M) / (M,F) FLAT LEANING (never reported
            as FLAT BETTER, which needs both seeds); (H,M) / (M,H) / (M,M) NOT ESTABLISHED. There is no gap: `verdict` maps the nine pairs through ONE table, and a pair that is not in it
            (an unknown seed state, or not two seeds) is a hard error, rc 2, never a verdict and never rc 0.
  IDENTITY  FLAT's GT-code oracle ADE6 minus its own ADE6 must read EXACTLY 0 at both seeds (per window, not on average), on an oracle that DID change the code offered to the interface
            (the eval's own `frac_code_changed` > 0). A structural control: a nonzero value voids the run (rc 2).
  STRATA    Delta_s within JUNCTION / CURVE / NO-TURN (raw/2026-10-25-kind/window_strata_eval141.npz, validated like the KIND reader does); a stratum with fewer than 10 episodes gets its n and a point
            estimate, no interval. Descriptive, no bar: it feeds nothing back (a test permutes the stratum labels and requires the outcome identical).
Reported beside it, deciding nothing: the four families, per family and never pooled (LON incl. distance keeping K = 10 from v8f_dk_read.read_dk, LAT, TACTICAL (T is still trained in FLAT, so its
decision metrics are read on both arms) and STRATEGIC -- NOT PAUSED for v8F-mini --), T's present-slot RMSE by group, the parameter counts, step counts and steps per second of both arms.
Refusals (rc 2): a missing file; runs not on one clip list / not withholding the RA-1 holdout / not done / different step counts / different window counts; an X run whose argv is not B's plus
`--o-interface cut`, whose launch-gate record (selfcheck.json: declared_vs_built PASS, G-LIVE pass with a frozen-prior control of exactly 0.0 and e_code.weight the ONLY declared-dead weight,
the o_interface probe that saw traj60 invariant to S and T under cut) is not a PASS, whose built levers / checkpoint do not say o_interface cut, whose parameter counts differ from B's by
anything (the lever adds no parameter), or whose built p_gt schedule / loss weights / hmax / code_mix differ from B's; a checkpoint sha256 that is not the banked B's, or an X checkpoint whose sha256
differs between the end of training, the end of the evals and the file pulled here; evals not on one grid of windows (eid, or the committed 2,399 / 141); an X eval that did not score
<runs>/M2b_s<seed>_flat/ckpt_last.pt or that did not build o_interface cut (its json says so); GT codes that differ between the arms; an identity control (B against itself) that misses exactly
+0 [0, 0]; FLAT's oracle identity that misses exactly 0; a B whose F / LAMBDA / S5 do not reproduce the PREREG's; a MISSING FAMILY (tactical / strategic in an eval json, longitudinal / lateral /
distance keeping in the read); a strata file that is not the eval grid's; a read with only one seed; a seed-state pair that is not one of the nine cells of the outcome table (a hard error, never an exit 0).
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
LEVER_FLAG, LEVER_VAL = "--o-interface", "cut"
ARM_B = "M2b"
ARM_X = "M2b-flat"
SUF = "_flat"
BAR_2F = 0.2415                                  # PREREG_V8F_FLAT: 2F, committed (F = 0.1207 m); a bar, not a recomputation
EVAL_WINDOWS, EVAL_EPISODES = 2399, 141          # the committed eval-141 grid
B_P_GT = (1.0, 0.2)                              # B's schedule (the trainer's default for M2b), which X inherits unchanged
B_GRID, N_TOK_B = "pool4x10", 40
B = 2000
X_SUFFIXES = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")
FAMILIES = ("longitudinal", "lateral", "tactical", "strategic")
STRATA_DEFAULT = RAW / "2026-10-25-kind" / "window_strata_eval141.npz"
# The banked B checkpoints (raw/full/MANIFEST.md): the chain asserts them on box 2 and writes the observed values to the `--ckpt-sha` json.
BANKED_CKPT_SHA = {"M2b_s0": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8",
                   "M2b_s1": "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
# The committed next-step texts, VERBATIM from PREREG_V8F_FLAT.md's outcome table (a test compares them with the file).
NEXT_TEXT = {"HIERARCHY BEATS FLAT": "register `C-V8F-HIER-1`; the interface-quality arm (MIXO's outcome, then T-side) is built on it",
             "NOT ESTABLISHED": "the interface is the defect. Run the lever MIXO names, then repeat this read on that arm",
             "FLAT BETTER": "an O-robustness arm (code dropout or MIXO-type mixing toward an empty code) under its own PREREG",
             "SEEDS DISAGREE": "seed 2 of BOTH arms (B has no seed 2 banked), 2 × B's wall-clock on box 2",
             # Amendment 1 (the PREREG's cell reads "reported to the PI as **v8F not validated against flat**; next is ..."; the bold markers are the file's markup, not part of the text)
             "FLAT LEANING": "reported to the PI as v8F not validated against flat; next is FLAT BETTER's committed arm (an O-robustness arm under its own PREREG)"}
MEANING = {"HIERARCHY BEATS FLAT": "S and T add at least 2F over a route-aware planner of the same capacity",
           "NOT ESTABLISHED": "the hierarchy, as trained, has not shown 2F of value over flat. Reported to the PI plainly: v8F is not validated against flat at 2F",
           "FLAT BETTER": "at own codes the interface HURTS: O leans on a code that is worse than nothing",
           "SEEDS DISAGREE": "the seed floor dominates",
           "FLAT LEANING": "the hierarchy is not validated, and one seed says the interface hurts. ⛔ It is never reported as FLAT BETTER, which needs both seeds"}   # verbatim; the JSON output is ascii-escaped
OUTCOMES = ("HIERARCHY BEATS FLAT", "NOT ESTABLISHED", "FLAT BETTER", "SEEDS DISAGREE", "FLAT LEANING")
NEXT = {vd: f"{vd}: {MEANING[vd]}. Next: {NEXT_TEXT[vd]}." for vd in OUTCOMES}
# Amendment 1: each seed is H (LO), F (HI) or M (MID), and ALL NINE pairs have exactly one row. This table is the whole decision; `verdict` reads nothing else.
HFM = {"LO": "H", "HI": "F", "MID": "M"}
OUTCOME_OF = {("H", "H"): "HIERARCHY BEATS FLAT",
              ("F", "F"): "FLAT BETTER",
              ("H", "F"): "SEEDS DISAGREE", ("F", "H"): "SEEDS DISAGREE",
              ("F", "M"): "FLAT LEANING", ("M", "F"): "FLAT LEANING",
              ("H", "M"): "NOT ESTABLISHED", ("M", "H"): "NOT ESTABLISHED", ("M", "M"): "NOT ESTABLISHED"}


def refuse(msg):
    print(f"FLAT READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# The programme's own instruments, IMPORTED (never copied): PREREG_V8F_GRID's reader brings Amendment 2's `w_diff` / `B_UNRECORDED_W` (and, through it, the PGT reader: run identity, eval loading, the
# GT-code check, the K = 10 lead block, the paired contrasts); the KIND reader brings the validated per-window strata. Loaded as PRIVATE instances; their refusals are routed through THIS reader's
# refuse (one message prefix, one rc).
GR = _load("v8f_grid_read_for_flat", HERE.parent / "v8f_grid_read.py")
GR.refuse = refuse
PGT = GR.PGT
PGT.refuse = refuse
KR = _load("v8f_kind_read_for_flat", HERE.parent / "v8f_kind_read.py")
KR.refuse = refuse
w_diff, B_UNRECORDED_W, _SKIP_LEVERS = GR.w_diff, GR.B_UNRECORDED_W, GR._SKIP_LEVERS
_SKIP_LEVERS_X = tuple(_SKIP_LEVERS) + ("o_interface", "code_mix")      # the lever itself: B's record predates the key and is compared to "full" where it carries one; code_mix is held at B's "whole" below


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
    return {"o_interface": cfg.get("o_interface"), "code_mix": cfg.get("code_mix"), "hist": cfg.get("hist"), "hmax": cfg.get("hmax"), "window_start": cfg.get("window_start"), "d": cfg.get("d"),
            "n_tok": cfg.get("n_tok"), "p_gt_start": cfg.get("p_gt_start"), "p_gt_end": cfg.get("p_gt_end"), "grid": ck.get("grid"),
            "time_rows": None if tm is None else int(tm.shape[0]), "pos_rows": None if pos is None else int(pos.shape[0]),
            "argv_sha": ck.get("argv_sha"), "step": ck.get("step"), "arm": ck.get("arm"), "seed": ck.get("seed")}


def check_params(lv_b, lv_x, tag):
    """The lever adds NO parameter: every group and the total equal B's (O keeps every parameter; S and T are still built). Any delta is a second lever."""
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
        refuse(f"{tag}: FLAT's parameter counts are not B's (the cut removes no parameter: the control is capacity-matched): a second lever moved -- " + "; ".join(bad))
    return {"n_params_trainable": {"B": lv_b.get("n_params_trainable"), "X": lv_x.get("n_params_trainable")}, "groups": {"B": pb, "X": px}}


def load_b_runs(runs_dir):
    """B's two seeds, validated by PGT's own run check for the M2b arm (flags, built schedule, one clip list, holdout, done, steps)."""
    return PGT.load_runs(runs_dir, (0, 1), arms=(ARM_B,))


def check_gate_token(tag, sm, sc, bad):
    """The launch-gate PASS token of a run: selfcheck.json of the in-process gate the trainer runs before it trains (PREREG: no launch-gate PASS token => rc 2)."""
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"{tag}: the launch gate's declared_vs_built is not PASS (no PASS token)")
    live = sc.get("live") or {}
    if live.get("pass") is not True or live.get("control_prior_grad_sum") != 0.0:
        bad.append(f"{tag}: the launch gate's G-LIVE is not a pass with a frozen-prior control of exactly 0.0 (no PASS token)")
    if live.get("declared_dead") != ["e_code.weight"] or live.get("dead_params") != ["e_code.weight"]:
        bad.append(f"{tag}: G-LIVE's dead-weight set is {live.get('dead_params')} (declared {live.get('declared_dead')}), expected exactly ['e_code.weight']: the cut left a wire into O, or killed another weight")
    pr = sc.get("o_interface_probe")
    if not isinstance(pr, dict) or pr.get("pass") is not True:
        bad.append(f"{tag}: selfcheck.json carries no passing o_interface probe (what O is handed under a perturbation of everything above it): no PASS token")
    else:
        if pr.get("o_interface") != LEVER_VAL:
            bad.append(f"{tag}: the o_interface probe was run for {pr.get('o_interface')!r}, not {LEVER_VAL!r}")
        perts = pr.get("perturbations") or {}
        need = {"donor_code", "gt_code", "random_code", "shifted_checkpoints"}
        if set(perts) != need:
            bad.append(f"{tag}: the o_interface probe perturbed {sorted(perts)}, expected {sorted(need)}")
        for name, v in perts.items():
            if v.get("max_abs_dtraj60") != 0.0 or v.get("o_read_code_abs_max") != 0.0 or v.get("o_read_ck_minus_nominal_abs_max") != 0.0 or v.get("o_read_ck_ok_abs_max") != 0.0:
                bad.append(f"{tag}: the o_interface probe under {name} reads {v}: O was not handed the constants / traj60 moved (max |d traj60| must be exactly 0.0)")
        sh = perts.get("shifted_checkpoints") or {}
        if not (isinstance(sh.get("t_reads_s_max_abs_dgoal_logits"), (int, float)) and sh["t_reads_s_max_abs_dgoal_logits"] > 0.0):
            bad.append(f"{tag}: the o_interface probe did not see T move with S's checkpoints (T must still read S)")
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
            if lg.get("o_interface") is None:
                bad.append(f"{tag}: {w}.json levers record no built o_interface: the lever is unproven")
            elif lg.get("o_interface") != LEVER_VAL:
                bad.append(f"{tag}: {w}.json built o_interface {lg.get('o_interface')!r}, expected {LEVER_VAL!r}")
            if lg.get("code_mix", "whole") != "whole":
                bad.append(f"{tag}: {w}.json built code_mix {lg.get('code_mix')!r}: FLAT is M2b's whole-code teacher forcing (ONE lever)")
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
        if (rsm.get("levers") or {}).get("o_interface", "full") != "full":
            bad.append(f"{ARM_B}_s{s}: summary.json built o_interface {rsm['levers']['o_interface']!r}: B is not the full-interface run the lever is defined against")
        d_model = int(fl.get("--d", 256))
        pc = {w: check_params(rsm.get("levers") or {}, rec.get("levers") or {}, f"{tag} ({w}.json)") for w, rec in (("summary", sm), ("selfcheck", sc))}
        params[s] = pc["summary"]
        ck = ckpt_facts(os.path.join(runs_dir, x_dir(s), "ckpt_last.pt"))
        if ck["o_interface"] != LEVER_VAL:
            bad.append(f"{tag}: the checkpoint was BUILT with o_interface {ck['o_interface']!r}, not {LEVER_VAL!r}")
        if ck["code_mix"] not in (None, "whole"):
            bad.append(f"{tag}: the checkpoint was BUILT with code_mix {ck['code_mix']!r}, B's is whole")
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
        built[s] = {"o_interface": ck["o_interface"], "hmax": ck["hmax"], "b_hmax": hm_b, "d": d_model}
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
    """Every X eval names the flat checkpoint it scored (the eval json's `ckpt`) and built o_interface cut (its json's `o_interface`, written by the eval from the checkpoint's cfg): an eval of B's
    checkpoint under X's tag, or of a checkpoint the eval built as full, is refused."""
    bad = []
    for s in seeds:
        for suf in X_SUFFIXES:
            p = os.path.join(eval_dir, f"eval_{ARM_X}_s{s}{suf}.json")
            if not os.path.exists(p):
                bad.append(f"{os.path.basename(p)} is missing")
                continue
            js = _jload(p)
            ck = js.get("ckpt")
            parts = str(ck or "").replace("\\", "/").rstrip("/").split("/")
            if len(parts) < 2 or parts[-2] != x_dir(s) or parts[-1] != "ckpt_last.pt":
                bad.append(f"{os.path.basename(p)} scored {ck!r}, not <runs>/{x_dir(s)}/ckpt_last.pt")
            if js.get("o_interface") != LEVER_VAL:
                bad.append(f"{os.path.basename(p)} built o_interface {js.get('o_interface')!r}, expected {LEVER_VAL!r} (the eval builds it from the checkpoint's cfg)")
    for s in seeds:
        p = os.path.join(eval_dir, f"eval_{ARM_B}_s{s}.json")
        if os.path.exists(p) and _jload(p).get("o_interface", "full") != "full":
            bad.append(f"{os.path.basename(p)} built o_interface {_jload(p).get('o_interface')!r}: B is the full-interface run")
    if bad:
        refuse("; ".join(bad))


def check_counts(eid, a):
    n, ne = int(eid.size), int(np.unique(eid).size)
    if (n, ne) != (a.n_windows, a.n_episodes):
        refuse(f"the evals hold {n} windows over {ne} episodes, the committed eval-141 grid has {a.n_windows} over {a.n_episodes}")
    return n, ne


# ------------------------------------------------------------------------------------------------ the identity read (FLAT's GT-code oracle minus own ADE6 is exactly 0)
def flat_identity(pf, Zx, eval_dir, seeds, eid, n_boot):
    """PREREG: FLAT's GT-code oracle ADE6 minus its own ADE6 must read EXACTLY 0 at both seeds -- per window (the arrays are equal, not just their means), on an oracle that DID change the code
    offered to the interface (`tac_manipulation.frac_code_changed` > 0 in the oracle eval's own json: otherwise a zero would say nothing), and so do the all-latcons oracle and the absmask eval
    (O reads constants whatever the evaluator writes). A nonzero value voids the run."""
    out = {}
    for s in seeds:
        Z = Zx[(ARM_X, s)]
        tag = f"{ARM_X}_s{s}"
        js = _jload(os.path.join(eval_dir, f"eval_{tag}-tacorc.json"))
        m = js.get("tac_manipulation") or {}
        if m.get("mode") != "oracle" or not (isinstance(m.get("frac_code_changed"), (int, float)) and m["frac_code_changed"] > 0.0):
            refuse(f"IDENTITY VOID: eval_{tag}-tacorc.json is not an oracle that changed the code offered to the interface (mode {m.get('mode')!r}, frac_code_changed {m.get('frac_code_changed')}): "
                   f"a zero read on it would say nothing")
        d = PGT.paired(pf, Z["orc_ade6"], Z["ade6"], eid, n_boot)
        worst = float(np.abs(Z["orc_ade6"] - Z["ade6"]).max())
        worst_pred = float(np.abs(Z["orc_pred"] - Z["pred"]).max())
        worst_lat = float(np.abs(Z["lat_pred"] - Z["pred"]).max())
        if not (np.array_equal(Z["orc_ade6"], Z["ade6"]) and np.array_equal(Z["orc_pred"], Z["pred"]) and np.array_equal(Z["lat_pred"], Z["pred"])
                and d["delta"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0 and d["point"] == 0.0):
            refuse(f"IDENTITY VOID: {tag}'s GT-code oracle minus its own ADE6 reads {d['delta']} [{d['lo']}, {d['hi']}] (point {d['point']}), max per-window |d ade6| {worst:.3g}, max |d pred| "
                   f"{worst_pred:.3g} (all-latcons oracle: {worst_lat:.3g}); it must be EXACTLY 0: O still reads something above it (or the eval did not build the cut)")
        out[f"s{s}"] = {"oracle_minus_own": {x: d[x] for x in ("delta", "lo", "hi", "point")}, "max_abs_per_window_dade6": worst, "max_abs_dpred_oracle": worst_pred,
                        "max_abs_dpred_all_latcons": worst_lat, "frac_code_changed": m["frac_code_changed"], "n_oracle_fallback_no_v9_row": m.get("n_oracle_fallback_no_v9_row")}
    return out


# ------------------------------------------------------------------------------------------------ the decision
def seed_state(p):
    """LO / HI / MID of one seed's Delta = ADE6(B) - ADE6(FLAT). The bars are the COMMITTED +-2F; `separated` is the library's (the CI excludes 0) AND on the bar's side."""
    pt = p["point"]
    if pt <= -BAR_2F and p["separated"] and p["hi"] < 0:
        return "LO"
    if pt >= BAR_2F and p["separated"] and p["lo"] > 0:
        return "HI"
    return "MID"


def verdict(states):
    """The PREREG's outcome table with Amendment 1 (all nine H / F / M pairs). `states` = (state of seed 0, state of seed 1). Returns (VERDICT, table_cell); table_cell is always "defined".
    A pair the table does not map (an unknown state, or not exactly two seeds) is a HARD ERROR, rc 2: the reader never exits 0 on a reading it cannot place."""
    if not isinstance(states, (tuple, list)) or len(states) != 2 or any(x not in HFM for x in states):
        refuse(f"verdict: seed states {states!r} are not a pair of LO / HI / MID; the nine-cell outcome table (PREREG_V8F_FLAT Amendment 1) cannot place them")
    key = (HFM[states[0]], HFM[states[1]])
    if key not in OUTCOME_OF:
        refuse(f"verdict: the seed pair {key} has no row in the nine-cell outcome table (PREREG_V8F_FLAT Amendment 1)")
    return OUTCOME_OF[key], "defined"


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
    if not (dk.get("pair_b_minus_x") and dk.get("arms")):
        miss.append("the K = 10 distance-keeping read has no B - FLAT pair")
    if famfn_used:
        fam = b.get("families") or {}
        for f in ("longitudinal", "lateral"):
            if not fam.get(f):
                miss.append(f"the paired {f} family block is empty")
    if miss:
        refuse(f"seed {s}: a family is missing -- " + "; ".join(miss) + " (a missing family is refused, never dropped)")


# ------------------------------------------------------------------------------------------------ strata (descriptive, no bar)
def strata_block(pf, Zb, Zx, seeds, eid, ctx, a):
    """Delta_s = ADE6(B) - ADE6(FLAT) within JUNCTION / CURVE / NO-TURN. The strata file is validated against the evals (eid, tick, label-derived stratum, v9-row presence == the oracle evals' own
    fallback mask, provenance) by the KIND reader's own loader. A stratum with < 10 episodes: n and the point, no interval. Nothing here is an input of `verdict`."""
    fbs = [~np.asarray(Zx[(ARM_X, s)]["has"], bool) for s in seeds] + [~np.asarray(Zb[(ARM_B, s)]["has"], bool) for s in (0, 1)]
    S = KR.load_strata(a.strata, eid, ctx["G"]["frame"], fbs)
    st = S["stratum"]
    out = {"definition": "Delta = ADE6(B) - ADE6(FLAT) within the stratum (negative = the hierarchy is better); paired episode-cluster bootstrap within the stratum's own episodes; "
                         f"fewer than {KR.MIN_STRATUM_EPISODES} episodes: n and the point, no interval. DESCRIPTIVE: no bar, no goalpost.",
           "file": os.path.basename(S["path"]), "release_md5": S["prov"].get("release_md5"), "strata": {}}
    for i, name in enumerate(KR.STRATA):
        m = st == i
        blk = {"n_windows": int(m.sum()), "n_episodes": int(np.unique(eid[m]).size), "seeds": {}}
        for s in seeds:
            if not m.any():
                blk["seeds"][f"s{s}"] = {"status": "NOT-DEFINED", "reason": "no window in the stratum"}
                continue
            x, b = Zx[(ARM_X, s)]["ade6"][m], Zb[(ARM_B, s)]["ade6"][m]
            if blk["n_episodes"] < KR.MIN_STRATUM_EPISODES:
                blk["seeds"][f"s{s}"] = {"point": float(b.mean() - x.mean()), "interval": f"NONE: fewer than {KR.MIN_STRATUM_EPISODES} episodes in the stratum"}
            else:
                r = PGT.paired(pf, b, x, eid[m], a.n_boot)
                blk["seeds"][f"s{s}"] = {k: r[k] for k in ("delta", "lo", "hi", "point", "separated")}
        out["strata"][name] = blk
    return out


# ------------------------------------------------------------------------------------------------ the read
def identity_control_b(pf, Zb_by_seed, pw, ctx, n_boot, WC, seeds, fam):
    """PGT's identity control: B against itself on PRIMARY, M-LEAK, M-DK, present-slot, oracle ADE6 and every family cell reads exactly +0 [0, 0]."""
    return PGT.identity_control(pf, Zb_by_seed, pw, ctx, n_boot, WC, seeds, fam=fam)


def _common(a, WC):
    SB, logsB = load_b_runs(a.runs_dir)
    Zb, eid = PGT.load_all(a.eval_dir, (0, 1), (ARM_B,), WC)
    return SB, logsB, Zb, eid


def read(a, pairfn, famfn, DK, WC):
    seeds = (0, 1)
    X = ARM_X
    SB, logsB, Zb, eid = _common(a, WC)
    for s in seeds:                                                          # BOTH seeds, always: a one-seed read is refused before anything is computed
        if not os.path.isdir(os.path.join(a.runs_dir, x_dir(s))):
            refuse(f"{x_dir(s)} is missing: the verdict needs BOTH seeds (a hierarchy claim on one seed is what H-ESTIM-SEED-1 forbids)")
    SX, logsX, pipe, params, built = load_x_runs(a.runs_dir, seeds, SB)
    Zx, eid_x = PGT.load_all(a.eval_dir, seeds, (X,), WC)
    if not np.array_equal(np.asarray(eid_x), np.asarray(eid)):
        refuse(f"the {X} evals are not on the same eval grid as {ARM_B}'s (eid differs)")
    n_win, n_ep = check_counts(eid, a)
    check_eval_ckpts(a.eval_dir, seeds)
    sha = check_ckpt_sha(a.ckpt_sha, a.runs_dir, seeds)
    Zrel = {**Zb, **Zx}
    PGT.check_gt_codes(Zrel, seeds, WC)
    F = abs(float(Zb[(ARM_B, 0)]["ade6"].mean()) - float(Zb[(ARM_B, 1)]["ade6"].mean()))
    ctx = PGT.build_ctx(a, DK, eid)
    pairs = [(f"{ARM_B}_s{s}", f"{X}_s{s}") for s in seeds] + [(f"{X}_s{s}", "P10") for s in seeds] + [(f"{ARM_B}_s{s}", "P10") for s in seeds]
    res10 = PGT.dk_read(a, DK, ctx, Zrel, ctx["blk10"], 10, a.dk_n_boot, pairs)
    pw = res10["perwindow"]
    if not np.array_equal(np.asarray(pw["is_lead"], bool), ctx["lead"]):
        refuse("the LEAD windows of the distance-keeping read are not the lead block's join")
    base_st = {s: PGT.baseline_stats(pairfn, Zb[(ARM_B, s)], pw, ctx, a.n_boot) for s in (0, 1)}
    known = PGT.known_values(base_st, F, (0, 1))
    idc = identity_control_b(pairfn, {s: Zb[(ARM_B, s)] for s in seeds}, pw, ctx, a.n_boot, WC, seeds, famfn)
    fid = flat_identity(pairfn, Zx, a.eval_dir, seeds, eid, a.n_boot)
    pe, _ = WC.present_error(Zrel, seeds)
    rows = {}
    for s in seeds:
        A, Bm = Zx[(X, s)], Zb[(ARM_B, s)]
        row = PGT.contrast(pairfn, Bm, A, pw, ctx, a.n_boot)                 # B - FLAT on every cell (negative = the hierarchy is better)
        row["base"] = {k: base_st[s][k] for k in ("lambda_m2b_m", "lambda_ci", "n_lead", "excess_p10", "excess_ci")}
        row["state"] = seed_state(row["primary"])
        ok = np.isfinite(pe[(ARM_B, s)]["mse"])
        row["beside"] = {
            "ade6": {"B": float(Bm["ade6"].mean()), "X": float(A["ade6"].mean())},
            "oracle_ade6": {"B": float(Bm["orc_ade6"].mean()), "X": float(A["orc_ade6"].mean())},
            "present_slot": {"n_windows_present": int(ok.sum()), "rmse": {"B": float(np.sqrt(pe[(ARM_B, s)]["mse"][ok].mean())), "X": float(np.sqrt(pe[(X, s)]["mse"][ok].mean()))},
                             "rmse_group": {"B": pe[(ARM_B, s)]["rmse_group"], "X": pe[(X, s)]["rmse_group"]},
                             "mse_delta_b_minus_x": PGT.paired(pairfn, pe[(ARM_B, s)]["mse"][ok], pe[(X, s)]["mse"][ok], eid[ok], a.n_boot),
                             "note": "T is trained in both arms; its accuracy is read on both, deciding nothing"},
            "eval_json": {k: {kk: Z[(nm, s)]["json"].get(kk) for kk in ("traj", "families", "strategic", "tactical") if kk in Z[(nm, s)]["json"]}
                          for k, (nm, Z) in {"B": (ARM_B, Zb), "X": (X, Zx)}.items()},
            "distance_keeping_k10": {"pair_b_minus_x": res10["read"]["pairs"].get(f"{ARM_B}_s{s}__minus__{X}_s{s}"), "pair_x_vs_p10": res10["read"]["pairs"].get(f"{X}_s{s}__minus__P10"),
                                     "arms": {nm: res10["read"]["arms"][nm] for nm in (f"{X}_s{s}", f"{ARM_B}_s{s}", "P10")}},
            "parameters": params[s], "built": built[s], "pipeline": {k: {kk: vv for kk, vv in v.items() if kk != "ckpt"} for k, v in pipe[s].items()},
            "gt_source": {"B": Bm["gt_source"], "X": A["gt_source"]}}
        if famfn is not None:
            row["beside"]["families"] = famfn(Bm["pred"], A["pred"], eid)      # B - FLAT, like the primary
        else:
            row["beside"]["families"] = {"status": "SKIPPED (--no-families, a TEST input; the chain never passes it)"}
        check_families(row, s, famfn is not None)
        rows[f"s{s}"] = row
    strata = strata_block(pairfn, Zb, Zx, seeds, eid, ctx, a)
    k30 = None
    if not a.no_k30:
        p30 = a.lead_block_k30 or str(DK.LEAD_BLOCK_K30)
        if os.path.exists(p30):
            k30 = PGT._trim_dk(PGT.dk_read(a, DK, ctx, Zrel, DK.load_block(p30), 30, a.dk_n_boot, pairs)["read"])
    states = tuple(rows[f"s{s}"]["state"] for s in seeds)
    vd, cell = verdict(states)
    fmt = lambda f: " / ".join(f(rows[f"s{s}"]) for s in seeds)               # noqa: E731
    prim = fmt(lambda r: f"{r['primary']['point']:+.3f}{'' if r['primary']['separated'] else ' ns'}")
    out = dict(schema="v8f-flat-read-v1", prereg="PREREG_V8F_FLAT.md", base=ARM_B, tier="T1-class open loop (eval-141, 2,399 windows); the identity read uses the T0 GT-code oracle eval (it reads the GT)",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); Delta = ADE6(B) - ADE6(FLAT); F = |{ARM_B}_s0 - {ARM_B}_s1| = {F:.4f} m (B's own two seeds); bar +-2F = "
                         f"{BAR_2F} m COMMITTED; answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)",
               n_windows=n_win, n_episodes=n_ep, n_lead_windows_k10=int(ctx["lead"].sum()), F=F, two_F_measured=2 * F, two_F_bar=BAR_2F, known_values=known,
               checkpoint_sha256=sha, identity_control=idc, flat_identity=fid, train_log={f"{k}": v for k, v in logsX.items()},
               strategic_family="NOT PAUSED: un-paused for v8F-mini (PI ruling 2026-10-17); read from each run's eval json under rows.*.beside.eval_json",
               distance_keeping_k30=k30, strata=strata, rows=rows, seed_states={f"s{s}": states[i] for i, s in enumerate(seeds)}, VERDICT=vd, table_cell=cell, next=NEXT[vd])
    out["reading"] = (f"PREREG_V8F_FLAT: {vd}; ADE6 M2b - FLAT {prim} m (bars -2F = -{BAR_2F}, +2F = +{BAR_2F}; seeds read {' / '.join(states)}); identity (FLAT GT-code oracle - own ADE6) "
                      f"{' / '.join(f'{fid[f's{s}']['oracle_minus_own']['delta']:+.4f}' for s in seeds)} m (exactly 0 required). {MEANING[vd]}")
    return out


def check_inputs(a, DK, WC, pairfn, famfn=None):
    """The B side alone, before any training: PGT's banked-side check (the runs and evals of BOTH seeds load, the K = 10 lead context builds, F / LAMBDA / S5 known values reproduce, the paired
    instrument reads exactly 0 on M2b against itself) plus this PREREG's: the grid is the committed 2,399 / 141, the strata file validates against B's own oracle evals, B's checkpoints are the
    banked ones when the chain's sha json is given."""
    banked = PGT.check_inputs(a, DK, WC, pairfn, famfn)
    Zb, eid = PGT.load_all(a.eval_dir, (0, 1), (ARM_B,), WC)
    n_win, n_ep = check_counts(eid, a)
    F = abs(float(Zb[(ARM_B, 0)]["ade6"].mean()) - float(Zb[(ARM_B, 1)]["ade6"].mean()))
    ctx = PGT.build_ctx(a, DK, eid)
    S = KR.load_strata(a.strata, eid, ctx["G"]["frame"], [~np.asarray(Zb[(ARM_B, s)]["has"], bool) for s in (0, 1)])
    st = S["stratum"]
    out = {"base": ARM_B, "banked": banked, "n_windows": n_win, "n_episodes": n_ep, "F": F, "two_F_bar": BAR_2F,
           "strata": {nm: {"n_windows": int((st == i).sum()), "n_episodes": int(np.unique(eid[st == i]).size)} for i, nm in enumerate(KR.STRATA)},
           "expected_x_suffixes": list(X_SUFFIXES), "expected_evals_per_seed": len(X_SUFFIXES), "banked_ckpt_sha256": BANKED_CKPT_SHA}
    if a.ckpt_sha:
        out["checkpoint_sha256"] = check_ckpt_sha(a.ckpt_sha, a.runs_dir, ())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-inputs", action="store_true", help="validate the B side alone (and the identity control) and exit")
    ap.add_argument("--eval-dir")
    ap.add_argument("--runs-dir")
    ap.add_argument("--out")
    ap.add_argument("--ckpt-sha", help="the chain's json {M2b_s0, M2b_s1, M2b-flat_s<seed>_trained, M2b-flat_s<seed>_after_evals: sha256} measured on box 2: required by the read")
    ap.add_argument("--n-boot", type=int, default=B, help="B of the deciding contrasts (default 2,000)")
    ap.add_argument("--dk-n-boot", type=int, default=B, help="B of the descriptive distance-keeping blocks (read_dk)")
    ap.add_argument("--no-families", action="store_true", help="TEST INPUT: skip the paired LON / LAT family blocks (the chain never passes it)")
    ap.add_argument("--no-k30", action="store_true", help="skip the 6 s distance-keeping block (descriptive only)")
    ap.add_argument("--n-windows", type=int, default=EVAL_WINDOWS, help="the committed eval grid's window count")
    ap.add_argument("--n-episodes", type=int, default=EVAL_EPISODES, help="the committed eval grid's episode count")
    ap.add_argument("--strata", default=str(STRATA_DEFAULT), help="the per-window stratum labels of the eval-141 grid (default: the repo's, built by mk_window_strata.py)")
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
            print(f"FLAT INPUTS OK: base {ARM_B}; F {res['F']:.4f} (2F bar {BAR_2F}); LAMBDA(B) {b['lambda_m2b'][0]:+.4f} / {b['lambda_m2b'][1]:+.4f}; {res['n_windows']} windows over {res['n_episodes']} episodes; "
                  f"strata {res['strata']}; expects {res['expected_evals_per_seed']} evals per seed; identity control exactly +0 [0, 0]")
            return 0
        res = read(a, pairfn, None if a.no_families else PGT._famfn(a), DK, WC)
    except (OSError, KeyError, ValueError) as e:
        print(f"FLAT READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    except Exception as e:                           # an inherited instrument's OWN Refusal (v8f_dk_read.Refusal): still a refusal, rc 2, never a traceback
        if type(e).__name__ != "Refusal":
            raise
        print(f"FLAT READ REFUSED: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(DK._clean(res), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
