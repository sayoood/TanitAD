"""PREREG_V8F_HIST, committed before its data: does feeding the stack's scene readers 10 history frames instead of 4 (`--hist 10`, ONE lever) make v8F-mini
drive better, and does it make T's LATERAL constraints (`latcons`, timings and geometry) more accurate?

  python v8f_hist_read.py --stage s0|final --eval-dir <eval> --runs-dir <runs> --out hist_read.json
  python v8f_hist_read.py --check-inputs --eval-dir ... --runs-dir ... --out x.json        (the B side alone, before any training)

B = M2b (fixed in the PREREG: PREREG_V8F_PGT and PREREG_V8F_GRID both FAILED, so neither is a base), the banked runs `M2b_s{0,1}` and their evals.
H = B + `--hist 10 --window-start 3` and nothing else (run dir `M2b_s<SEED>_h10`; eval tags `M2b-h10_s<SEED>`, suffixes "", "-tacabs", "-tacorc", "-tacorc-all-latcons").
AMENDMENT 1 (BUILD.md F-1): `--hist 10` ALONE starts every clip's training windows at t = 9 (train_windows' default start is hist - 1), so t = 3..8 are not trained on and the trainer's step
budget falls from 19,164 to 17,472. `--window-start 3` fixes BOTH the first training window and the held-age clamp's origin at B's value: ONE lever set by two flags, B's window list,
B's held ages, B's 306,566 windows and 19,164 steps; the readers see 10 frames, the ones before the clip padded. The argv difference to B is exactly `--hist 10 --window-start 3` (and `--out`).
Eval: `v8f_mini_eval.py run` on eval-141 (2,399 windows, 141 episodes), "T1-class open loop"; the mechanism read uses the absmask and GT-oracle evals (T0) only to score T.

  PRIMARY   per seed s: delta_s = ADE6(H_s) - ADE6(B_s), paired over episodes, on ALL 2,399 windows (deciding). Seed s PASSES iff delta_s < 0, separated and
            |delta_s| > 2F, F = |mean ADE6 B_s0 - mean ADE6 B_s1| (B's seed floor from its own two seeds: 0.1207 m for M2b, banked).
  M-LAT     per seed (decides the reading of a FAIL, never the verdict): T's slot-pooled present-slot RMSE on `latcons` falls by >= 12.5 % relative to B_s,
            separated (paired over episodes on the per-window MSE). The same machinery as PREREG_V8F_GRID (v8f_grid_read.mlat_row, IMPORTED): size on the
            slot-pooled RMSE (`rmse_group["latcons"]`, the 0.686 SD of M2b_s0), separation on the paired per-window latcons MSE, the window-weighted RMSE beside it.
  STAGES    s0: seed 0 only. CONTINUE iff seed 0 PASSES the PRIMARY (only then is seed 1 trained); otherwise STOP with the seed-0 verdict. final: both seeds.
  VERDICTS  SUCCESS (both seeds pass) | FAIL-MECHANISM-OK (latcons accuracy fell >= 12.5 % at every seed read; driving did not move beyond 2F) | FAIL-NO-MECHANISM.
Reported beside it, deciding nothing: the four families (LON incl. distance keeping, LAT, TACTICAL and STRATEGIC -- un-paused for v8F-mini); T's present-slot RMSE for
every group AND per latcons slot (the four timing / along-path slots turn_t_start_s, turn_t_end_s, turn_d_start_m, lc_t_cross_s pooled beside the eight geometry
slots); the latcons leak LAMBDA; the GT-code oracle ADE6; the PRIMARY on the t >= 11 subset (D4: 2,117 windows whose history is fully real at both hist values)
and on its complement; both arms' parameter counts, window counts, step counts and steps per second.
Refusals (rc 2): a missing file; runs not on one clip list / not withholding the RA-1 holdout / not done / different step counts / different window counts; an H run whose argv is not
B's plus `--hist 10 --window-start 3`, whose checkpoint was not built at hist 10 with window_start 3, whose `time` table does not have 10 rows, or whose parameter delta over B is not exactly
+1,536 in `adapter`; an H run whose run records (summary.json, selfcheck.json levers) do not say built hist 10 / window_start 3 / hmax = B's 16 (B's own old record, which predates the
keys, is compared KEY BY KEY: where it lacks one nothing is compared, "absent" is never "different"); an H run whose selfcheck is not PASS or whose built hmax differs from B's; any
loss weight or other built lever that differs from B's (key by key: PREREG_V8F_GRID
Amendment 2's `w_diff`, IMPORTED, with the same two proven keys); evals not on one grid of windows; GT codes that differ between the arms.
ADDED GUARDS (same direction; they refuse, they never decide): H's other built levers and its loop-logged p_gt equal B's; the checkpoint's grid and n_tok are B's; every H
eval names the hist-10 checkpoint it scored; the eval windows on the committed grid number 2,399 / 2,117 / 141 / 141 (t >= 11 / t = 3 / t = 7); the identity control (B
against itself reads exactly +0 [0, 0] on every contrast: M-LAT, the t >= 11 subset and the per-slot sets included).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (EPISODES); it answers "another draw of EPISODES", and training-seed variance enters only through F
(H-ESTIM-SEED-1). v8F-mini's planner is deterministic, so no inference-seed floor applies."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
RAW = PKG / "raw"
HIST_FLAG, HIST_VAL, B_HIST_VAL = "--hist", "10", "4"
WS_FLAG, WS_VAL = "--window-start", "3"          # Amendment 1: H's first training window AND held-age clamp origin = B's (hist 4 - 1)
B_WS_EFFECTIVE = 3                               # B's effective window start (hist - 1 = 3); B's old record does not say so, a new record does
H_ROWS, B_ROWS = 10, 4                           # the per-frame `time` table's rows
B_GRID = "pool4x10"                              # H keeps B's grid; the checkpoint must say so
N_TOK_B = 40
SUF = "_h10"
ARM_B = "M2b"
ARM_H = "M2b-h10"
MLAT_BAR = 0.125                                 # M-LAT: latcons RMSE must fall by at least this fraction of B's
T_SUB_MIN = 11                                   # D4: the eval windows whose history is fully real at BOTH hist values (t = 3 mod 4 on the grid, so t >= 9 <=> t >= 11)
WINDOWS_COMMITTED = {"total": 2399, "t_ge_11": 2117, "t3": 141, "t7": 141}      # PREREG D4 (MEASURED on raw/eval_grid_windows.json), asserted on the real grid
TIMING_SLOTS = ("turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "lc_t_cross_s")
B = 2000
EVAL_SUFFIXES = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")
# The committed next-step texts, VERBATIM from PREREG_V8F_HIST.md (a test compares them with the file).
NEXT_TEXT = {"SUCCESS": "the T1-kin read of H (`v8f_t1_eval.py`, m ∈ {30, 10}), then the next §13.6 lever on top (D1 rebalancing), under its own PREREG",
             "FAIL-MECHANISM-OK": "T is getting more accurate but not enough to convert: the encoder itself, LoRA on DINOv3 (needs the PI's AR-2 go)",
             "FAIL-NO-MECHANISM": "neither space nor time in the frozen tokens is T's limit: §13.6's D1 rebalancing (event-stratified batches; the clip list unchanged), "
                                  "under its own PREREG, with LoRA (AR-2) behind it"}
NEXT = {None: "CONTINUE: seed 0 passes the PRIMARY; train seed 1 (the verdict needs both).",
        "SUCCESS": "SUCCESS: the stack's scene readers at 10 history frames drive better than B beyond 2F at both seeds. Next: " + NEXT_TEXT["SUCCESS"] + ".",
        "FAIL-MECHANISM-OK": "FAIL with the mechanism in place: latcons accuracy fell >= 12.5 % but driving did not move beyond 2F. Next: " + NEXT_TEXT["FAIL-MECHANISM-OK"] + ".",
        "FAIL-NO-MECHANISM": "FAIL without the mechanism: latcons accuracy did not fall >= 12.5 %. Next: " + NEXT_TEXT["FAIL-NO-MECHANISM"] + "."}


def refuse(msg):
    print(f"HIST READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# The programme's own instruments, IMPORTED (never copied): PREREG_V8F_GRID's reader brings the M-LAT machinery, the identity control, the verdict rule and --
# the one the PREREG names -- Amendment 2's `w_diff` / `B_UNRECORDED_W`; it brings the PGT reader (run identity, eval loading, the GT-code check, the K = 10 lead
# block, the paired contrasts). Both are loaded as PRIVATE instances and their refusals are routed through THIS reader's refuse (one message prefix, one rc).
GR = _load("v8f_grid_read_for_hist", HERE.parent / "v8f_grid_read.py")
GR.refuse = refuse
PGT = GR.PGT
PGT.refuse = refuse
if GR.MLAT_BAR != MLAT_BAR:
    raise SystemExit(f"v8f_hist_read: GRID's M-LAT bar {GR.MLAT_BAR} != this PREREG's {MLAT_BAR}")
w_diff, B_UNRECORDED_W, _SKIP_LEVERS = GR.w_diff, GR.B_UNRECORDED_W, GR._SKIP_LEVERS
_SKIP_LEVERS_H = tuple(_SKIP_LEVERS) + ("hist",)  # the lever itself; hmax and window_start are compared with B's wherever B's record carries them (it may not: key by key)


def _modules():
    return PGT._modules()


def h_dir(s):
    return f"{ARM_B}_s{s}{SUF}"


def _jload(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        refuse(f"cannot read {p}: {type(e).__name__}: {e}")


# ------------------------------------------------------------------------------------------------ runs
def b_hmax(rfl, rsm):
    """The hmax the trainer's make_config gives B: `--hmax` when set and non-zero, else its auto rule (16 when the arm has a world-model auxiliary, else 10). B's
    banked record names its wm in levers; a record without it leaves the rule unprovable."""
    h = rfl.get("--hmax")
    if h not in (None, True) and int(h) != 0:
        return int(h)
    wm = (rsm.get("levers") or {}).get("wm")
    if wm is None:
        refuse("B's record names no wm lever: B's built hmax (the trainer's auto rule: 16 with a world-model auxiliary, else 10) cannot be derived")
    return 16 if wm != "off" else 10


def ckpt_facts(path):
    """The BUILT model, read from the checkpoint the trainer saved: the history it was built at, the `time` table's rows, hmax, the grid and n_tok."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, mdl = ck.get("cfg") or {}, ck.get("model") or {}
    tm, pos = mdl.get("time"), mdl.get("pos")
    return {"hist": cfg.get("hist"), "hmax": cfg.get("hmax"), "window_start": cfg.get("window_start"), "d": cfg.get("d"), "n_tok": cfg.get("n_tok"), "grid": ck.get("grid"),
            "time_rows": None if tm is None else int(tm.shape[0]), "time_dim": None if tm is None else int(tm.shape[-1]),
            "pos_rows": None if pos is None else int(pos.shape[0]),
            "argv_sha": ck.get("argv_sha"), "step": ck.get("step"), "arm": ck.get("arm"), "seed": ck.get("seed")}


def check_params(lv_b, lv_h, d, tag):
    """H adds EXACTLY six rows of the per-frame time table: (10 - 4) x d parameters, all in the adapter group (where `time` lives). Any other delta is a second lever."""
    want = (H_ROWS - B_ROWS) * d
    pb, ph = lv_b.get("params") or {}, lv_h.get("params") or {}
    bad = []
    if not pb or not ph:
        bad.append("a run's levers carry no per-group parameter counts")
    else:
        for k in sorted(set(pb) | set(ph)):
            dd = ph.get(k, 0) - pb.get(k, 0)
            if dd != (want if k == "adapter" else 0):
                bad.append(f"group {k}: {ph.get(k)} vs B's {pb.get(k)} (delta {dd}, expected {want if k == 'adapter' else 0})")
    dt = (lv_h.get("n_params_trainable") or 0) - (lv_b.get("n_params_trainable") or 0)
    if dt != want:
        bad.append(f"n_params_trainable delta {dt}, expected {want} = (10 - 4) x {d}")
    if bad:
        refuse(f"{tag}: H's parameter delta over B is not exactly the time table ((10 - 4) x d = {want}, all in adapter): a second lever moved -- " + "; ".join(bad))
    return {"d": d, "expected_delta": want, "delta": dt, "n_params_trainable": {"B": lv_b.get("n_params_trainable"), "H": lv_h.get("n_params_trainable")},
            "adapter": {"B": pb.get("adapter"), "H": ph.get("adapter")}}


def load_b_runs(runs_dir):
    """B's two seeds, validated by PGT's own run check for the M2b arm (flags, built schedule, one clip list, holdout, done, steps)."""
    return PGT.load_runs(runs_dir, (0, 1), arms=(ARM_B,))


def load_h_runs(runs_dir, seeds, SB):
    S, logs, bad, pipe, params, built = {}, {}, [], {}, {}, {}
    p0, p1 = 1.0, 0.2                                  # B's schedule (the trainer's default for M2b), which H inherits unchanged
    for s in seeds:
        d = os.path.join(runs_dir, h_dir(s))
        for f in ("summary.json", "argv.json", "selfcheck.json", "ckpt_last.pt"):
            if not os.path.exists(os.path.join(d, f)):
                refuse(f"{os.path.join(d, f)} is missing (the run did not finish, or was not pulled)")
        S[s] = {f: _jload(os.path.join(d, f)) for f in ("summary.json", "argv.json", "selfcheck.json")}
    for s in seeds:
        tag = f"{ARM_H}_s{s}"
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
            bad.append(f"{tag} trained on {sm.get('n_windows')} windows, {ARM_B}_s{s} on {rsm.get('n_windows')}: the window set is not B's (Amendment 1: {WS_FLAG} {WS_VAL})")
        if sc.get("declared_vs_built") != "PASS":
            bad.append(f"{tag}: the launch gate's declared_vs_built is not PASS (selfcheck is not PASS)")
        lv_live = sc.get("live") or {}
        if lv_live.get("pass") is not True or lv_live.get("control_prior_grad_sum") != 0.0:
            bad.append(f"{tag}: the launch gate's G-LIVE is not a pass with a frozen-prior control of exactly 0.0 (selfcheck is not PASS)")
        if sc.get("argv_sha") != sm.get("argv_sha"):
            bad.append(f"{tag}: selfcheck.json (argv_sha {sc.get('argv_sha')}) is not this run's (summary.json {sm.get('argv_sha')})")
        if fl.get("--arm") != "M2b":
            bad.append(f"{tag}: argv --arm {fl.get('--arm')}, expected M2b")
        if fl.get(HIST_FLAG) != HIST_VAL:
            bad.append(f"{tag}: argv {HIST_FLAG} {fl.get(HIST_FLAG)}, expected {HIST_VAL}")
        if fl.get(WS_FLAG) != WS_VAL:
            bad.append(f"{tag}: argv {WS_FLAG} {fl.get(WS_FLAG)}, expected {WS_VAL} (Amendment 1: without it --hist 10 starts the windows at t = 9 and trains 17,472 steps)")
        if rfl.get(HIST_FLAG, B_HIST_VAL) != B_HIST_VAL:
            bad.append(f"{ARM_B}_s{s}: argv {HIST_FLAG} {rfl.get(HIST_FLAG)}: B is not a 4-frame run")
        if WS_FLAG in rfl:
            bad.append(f"{ARM_B}_s{s}: argv names {WS_FLAG}: B is the banked M2b, which predates the flag")
        drop = {"--out", HIST_FLAG, WS_FLAG}
        fh, fb = {k: x for k, x in fl.items() if k not in drop}, {k: x for k, x in rfl.items() if k not in drop}
        if fh != fb:
            diff = sorted(k for k in set(fh) | set(fb) if fh.get(k) != fb.get(k))
            bad.append(f"{tag}: argv differs from {ARM_B}_s{s}'s in {diff} besides {HIST_FLAG}, {WS_FLAG} and --out -- not one lever")
        built_pg = {w: ((rec.get("levers") or {}).get("p_gt_start"), (rec.get("levers") or {}).get("p_gt_end")) for w, rec in (("summary", sm), ("selfcheck", sc))}
        for w, (b0, b1) in built_pg.items():
            if b0 is None or b1 is None:
                bad.append(f"{tag}: {w}.json levers record no built p_gt_start / p_gt_end: the inherited schedule is unproven")
            elif not (PGT._same_float(b0, p0) and PGT._same_float(b1, p1)):
                bad.append(f"{tag} was built with p_gt {b0} -> {b1} ({w}.json), B's is {p0} -> {p1}")
        hm_b = b_hmax(rfl, rsm)
        for w, rec in (("summary", sm), ("selfcheck", sc)):
            lg = rec.get("levers") or {}
            # Amendment 1 / F-3: the run's OWN record of what it built. Required of H (the trainer writes them at this commit; the gate asserted them before training) ...
            for key, want in (("hist", int(HIST_VAL)), ("window_start", int(WS_VAL)), ("hmax", hm_b)):
                if lg.get(key) is None:
                    bad.append(f"{tag}: {w}.json levers record no built {key}: the lever is unproven")
                elif lg.get(key) != want:
                    bad.append(f"{tag}: {w}.json built {key} {lg.get(key)}, expected {want}")
        for w, rec in (("summary", sm), ("selfcheck", sc)):
            lg, lb = rec.get("levers") or {}, rsm.get("levers") or {}
            diff = sorted(k for k in set(lg) & set(lb) if k not in _SKIP_LEVERS_H and lg[k] != lb[k])
            diff += [f"w.{k}" for k in w_diff(lg.get("w") or {}, lb.get("w") or {}, rfl)]
            if diff:
                bad.append(f"{tag}: {w}.json built levers {diff} differ from {ARM_B}_s{s}'s")
        # ... and of B only WHERE B's record has it (the banked M2b predates the keys: absent is not different), key by key
        for key, want in (("hist", int(B_HIST_VAL)), ("window_start", B_WS_EFFECTIVE), ("hmax", hm_b)):
            if key in (rsm.get("levers") or {}) and rsm["levers"][key] != want:
                bad.append(f"{ARM_B}_s{s}: summary.json built {key} {rsm['levers'][key]}, expected {want}: B is not the 4-frame run the lever is defined against")
        d_model = int(fl.get("--d", 256))
        pc = {w: check_params(rsm.get("levers") or {}, rec.get("levers") or {}, d_model, f"{tag} ({w}.json)") for w, rec in (("summary", sm), ("selfcheck", sc))}
        params[s] = pc["summary"]
        ck = ckpt_facts(os.path.join(runs_dir, h_dir(s), "ckpt_last.pt"))
        if ck["hist"] != int(HIST_VAL):
            bad.append(f"{tag}: the checkpoint was BUILT at hist {ck['hist']}, not {HIST_VAL}")
        if ck["time_rows"] != H_ROWS:
            bad.append(f"{tag}: the checkpoint's per-frame time table has {ck['time_rows']} rows, not {H_ROWS}")
        if ck["time_dim"] != d_model:
            bad.append(f"{tag}: the checkpoint's time table is {ck['time_dim']} wide, d is {d_model}")
        if ck["hmax"] != hm_b:
            bad.append(f"{tag}: the checkpoint was BUILT with hmax {ck['hmax']}, B's is {hm_b}")
        if ck["window_start"] != int(WS_VAL):
            bad.append(f"{tag}: the checkpoint was BUILT with window_start {ck['window_start']}, not {WS_VAL} (Amendment 1)")
        info_h = sc.get("info") or {}
        for key, what in (("window_start", "what train_windows was asked for"), ("window_t_min", "the first t of the training window list"),
                          ("ds_window_start", "the dataset's held-age clamp origin")):
            if key in info_h and info_h[key] != int(WS_VAL):                      # where present: provenance beside the levers (the gate asserted them before training)
                bad.append(f"{tag}: selfcheck.json info.{key} ({what}) is {info_h[key]}, not {WS_VAL}")
        if ck["grid"] != B_GRID or ck["n_tok"] != N_TOK_B or ck["pos_rows"] != N_TOK_B:
            bad.append(f"{tag}: the checkpoint's grid {ck['grid']!r} / n_tok {ck['n_tok']} / pos rows {ck['pos_rows']} are not B's ({B_GRID!r} / {N_TOK_B} / {N_TOK_B})")
        if ck["argv_sha"] != sm.get("argv_sha"):
            bad.append(f"{tag}: the checkpoint was written by another argv (argv_sha {ck['argv_sha']} vs summary.json {sm.get('argv_sha')})")
        if ck["step"] != sm.get("steps"):
            bad.append(f"{tag}: the checkpoint is at step {ck['step']}, the run's {sm.get('steps')}")
        if ck["arm"] != "M2b" or ck["seed"] != s:
            bad.append(f"{tag}: the checkpoint says arm {ck['arm']} seed {ck['seed']}")
        built[s] = {"hist": ck["hist"], "hmax": ck["hmax"], "window_start": ck["window_start"], "b_hmax": hm_b, "time_rows": ck["time_rows"]}
        logs[s] = PGT._train_log_check(os.path.join(runs_dir, h_dir(s), "train_log.jsonl"), ARM_H, s, sm.get("steps"), p0, p1, True)
        pipe[s] = {"H": {"lru": (sc.get("info") or {}).get("lru"), "seconds": sm.get("seconds"), "steps": sm.get("steps"), "n_windows": sm.get("n_windows"), "S": sm.get("S"),
                         "steps_per_s": (sm["steps"] / sm["seconds"]) if sm.get("seconds") else None, "ckpt": ck},
                   "B": {"lru": (ref["selfcheck.json"].get("info") or {}).get("lru"), "seconds": rsm.get("seconds"), "steps": rsm.get("steps"),
                         "n_windows": rsm.get("n_windows"), "S": rsm.get("S"), "steps_per_s": (rsm["steps"] / rsm["seconds"]) if rsm.get("seconds") else None}}
    if bad:
        refuse("; ".join(bad))
    return S, logs, pipe, params, built


def check_eval_ckpts(eval_dir, seeds):
    """Every H eval names the hist-10 checkpoint it scored (the eval json's `ckpt`): an eval of B's checkpoint under H's tag is refused."""
    bad = []
    for s in seeds:
        for suf in EVAL_SUFFIXES:
            p = os.path.join(eval_dir, f"eval_{ARM_H}_s{s}{suf}.json")
            ck = _jload(p).get("ckpt")
            parts = str(ck or "").replace("\\", "/").rstrip("/").split("/")
            if len(parts) < 2 or parts[-2] != h_dir(s) or parts[-1] != "ckpt_last.pt":
                bad.append(f"{os.path.basename(p)} scored {ck!r}, not <runs>/{h_dir(s)}/ckpt_last.pt")
    if bad:
        refuse("; ".join(bad))


# ------------------------------------------------------------------------------------------------ windows (D4)
def window_t(ctx):
    """The cache step t of every eval window: v8f_dk_read.ground_truth sets frame = 2t (RAW frames), and the --gt-npz test input follows the same convention."""
    fr = np.asarray(ctx["G"]["frame"], np.int64)
    if (fr % 2).any():
        refuse("the ground truth's frame is not 2t on every window: the eval windows' cache steps cannot be derived")
    return fr // 2


def window_counts(t):
    n = int(t.size)
    c = {"total": n, "t_ge_11": int((t >= T_SUB_MIN).sum()), "t_lt_11": int((t < T_SUB_MIN).sum()), "t3": int((t == 3).sum()), "t7": int((t == 7).sum())}
    if not c["t_ge_11"]:
        refuse("no eval window has t >= 11: the D4 subset is empty")
    c["asserted_against_prereg"] = bool(n == WINDOWS_COMMITTED["total"])
    if c["asserted_against_prereg"]:
        bad = [f"{k} {c[k]} != {WINDOWS_COMMITTED[k]}" for k in ("t_ge_11", "t3", "t7") if c[k] != WINDOWS_COMMITTED[k]]
        if bad:
            refuse("the eval grid's window counts are not the PREREG's D4 counts (" + "; ".join(bad) + "): the t >= 11 subset is not the committed 2,117")
    return c


# ------------------------------------------------------------------------------------------------ the per-slot beside-reads
def _pres(g_cons, has):
    return has[:, None] & (g_cons != 0.0)


def slot_names(D, WC):
    names = tuple(D.LAT_CONSTRAINTS)
    cols = list(WC.GROUPS["latcons"])
    if len(names) != len(cols) or any(x not in names for x in TIMING_SLOTS):
        refuse(f"the latcons slot names {names} do not match the WC reader's latcons columns {cols}")
    return names, cols


def slot_rmse(arm, g_cons, has, cols):
    """Per latcons slot and per slot SET: T's present-slot RMSE (SD units). Present = GT-present and GT != 0 (v8f_wc_read.present_error's own rule)."""
    pres = _pres(g_cons, has)
    e2 = np.where(pres, (arm["t_cons"] - g_cons) ** 2, 0.0)
    per = []
    for c in cols:
        n = int(pres[:, c].sum())
        per.append({"n_present": n, "rmse": float(np.sqrt(e2[:, c].sum() / n)) if n else None})
    return per, pres, e2


def set_rmse(e2, pres, cols):
    n = int(pres[:, cols].sum())
    return float(np.sqrt(e2[:, cols].sum() / n)) if n else None, n


def set_window_mse(arm, g_cons, has, cols):
    """Per window: the mean squared error of T's present slots within `cols`; NaN where the window has none."""
    pres = _pres(g_cons, has)[:, cols]
    e2 = np.where(pres, (arm["t_cons"][:, cols] - g_cons[:, cols]) ** 2, 0.0)
    n = pres.sum(1)
    return np.where(n > 0, e2.sum(1) / np.maximum(n, 1), np.nan)


def slot_block(pf, A, Bm, ref, WC, D, eid, n_boot):
    """latcons slot by slot (B vs H), and the four timing / along-path slots pooled beside the eight geometry slots. DESCRIPTIVE: it decides nothing."""
    names, cols = slot_names(D, WC)
    has, g = ref["has"], ref["g_cons"]
    per_a, pres, e2a = slot_rmse(A, g, has, cols)
    per_b, _, e2b = slot_rmse(Bm, g, has, cols)
    out = {"slots": {}, "sets": {}}
    for i, nm in enumerate(names):
        ra, rb = per_a[i]["rmse"], per_b[i]["rmse"]
        out["slots"][nm] = {"n_present": per_b[i]["n_present"], "rmse": {"B": rb, "H": ra}, "fall": (1.0 - ra / rb) if (ra is not None and rb) else None,
                            "kind": "timing" if nm in TIMING_SLOTS else "geometry"}
    timing = [cols[i] for i, nm in enumerate(names) if nm in TIMING_SLOTS]
    geom = [cols[i] for i, nm in enumerate(names) if nm not in TIMING_SLOTS]
    for key, cs in (("timing4", timing), ("geometry8", geom)):
        ra, na = set_rmse(e2a, pres, cs)
        rb, nb = set_rmse(e2b, pres, cs)
        ma, mb = set_window_mse(A, g, has, cs), set_window_mse(Bm, g, has, cs)
        ok = np.isfinite(mb)
        d = PGT.paired(pf, ma[ok], mb[ok], eid[ok], n_boot) if ok.any() else None
        out["sets"][key] = {"slots": [names[cols.index(c)] for c in cs], "n_slots_present": nb, "rmse": {"B": rb, "H": ra},
                            "fall": (1.0 - ra / rb) if (ra is not None and rb) else None, "n_windows_present": int(ok.sum()), "mse_delta": d}
    return out


# ------------------------------------------------------------------------------------------------ the read
def verdict(stage, rows, seeds):
    return GR.verdict(stage, rows, seeds)


def _common(a, WC):
    SB, logsB = load_b_runs(a.runs_dir)
    Zb, eid = PGT.load_all(a.eval_dir, (0, 1), (ARM_B,), WC)
    return SB, logsB, Zb, eid


def subset_rows(pf, A, Bm, sub, eid, n_boot):
    """The PRIMARY contrast on one subset of the eval windows (D4). Beside the deciding all-window PRIMARY, deciding nothing."""
    if not sub.any():
        return None
    return dict(PGT.paired(pf, A["ade6"][sub], Bm["ade6"][sub], eid[sub], n_boot), n_windows=int(sub.sum()), n_episodes=int(np.unique(eid[sub]).size))


def read(a, pairfn, famfn, DK, WC):
    seeds = (0,) if a.stage == "s0" else (0, 1)
    H = ARM_H
    SB, logsB, Zb, eid = _common(a, WC)
    SH, logsH, pipe, params, built = load_h_runs(a.runs_dir, seeds, SB)
    Zh, eid_h = PGT.load_all(a.eval_dir, seeds, (H,), WC)
    if not np.array_equal(np.asarray(eid_h), np.asarray(eid)):
        refuse(f"the {H} evals are not on the same eval grid as {ARM_B}'s (eid differs)")
    check_eval_ckpts(a.eval_dir, seeds)
    Zrel = {**Zb, **Zh}
    PGT.check_gt_codes(Zrel, seeds, WC)
    F = abs(float(Zb[(ARM_B, 0)]["ade6"].mean()) - float(Zb[(ARM_B, 1)]["ade6"].mean()))
    ctx = PGT.build_ctx(a, DK, eid)
    t = window_t(ctx)
    wc = window_counts(t)
    sub = t >= T_SUB_MIN
    Zdk = {**Zb, **Zh}
    pairs = [(f"{H}_s{s}", f"{ARM_B}_s{s}") for s in seeds] + [(f"{H}_s{s}", "P10") for s in seeds] + [(f"{ARM_B}_s{s}", "P10") for s in (0, 1)]
    res10 = PGT.dk_read(a, DK, ctx, Zdk, ctx["blk10"], 10, a.dk_n_boot, pairs)
    pw = res10["perwindow"]
    if not np.array_equal(np.asarray(pw["is_lead"], bool), ctx["lead"]):
        refuse("the LEAD windows of the distance-keeping read are not the lead block's join")
    base_st = {s: PGT.baseline_stats(pairfn, Zb[(ARM_B, s)], pw, ctx, a.n_boot) for s in (0, 1)}
    known = PGT.known_values(base_st, F, (0, 1))
    idc = identity_control(pairfn, {s: Zb[(ARM_B, s)] for s in seeds}, pw, ctx, a.n_boot, WC, seeds, Zrel, sub, None)
    pe, _ = WC.present_error(Zrel, seeds)
    ref = Zrel[(ARM_B, seeds[0])]
    D = _tanitad_data()
    rows = {}
    for s in seeds:
        A, Bm = Zh[(H, s)], Zb[(ARM_B, s)]
        row = PGT.contrast(pairfn, A, Bm, pw, ctx, a.n_boot)
        row["mlat"] = GR.mlat_row(pairfn, A, Bm, pe[(H, s)], pe[(ARM_B, s)], ref, WC, eid, a.n_boot)
        row["base"] = {k: base_st[s][k] for k in ("lambda_m2b_m", "lambda_ci", "n_lead", "excess_p10", "excess_ci")}
        row["judge"] = GR.judge(row, F)
        ok = np.isfinite(pe[(ARM_B, s)]["mse"])
        sub_p = subset_rows(pairfn, A, Bm, sub, eid, a.n_boot)
        sub_c = subset_rows(pairfn, A, Bm, ~sub, eid, a.n_boot)
        row["beside"] = {
            "ade6": {"B": float(Bm["ade6"].mean()), "H": float(A["ade6"].mean())},
            "primary_t_ge_11": dict(sub_p, passes_bar_if_it_were_deciding=bool(PGT._is_neg_sep(sub_p) and abs(sub_p["point"]) > 2 * F)),
            "primary_t_lt_11": sub_c,
            "oracle_ade6": {"B": float(Bm["orc_ade6"].mean()), "H": float(A["orc_ade6"].mean())},
            "oracle_ade6_delta": PGT.paired(pairfn, A["orc_ade6"], Bm["orc_ade6"], eid, a.n_boot),
            "present_slot": {"n_windows_present": int(ok.sum()), "rmse": {"B": float(np.sqrt(pe[(ARM_B, s)]["mse"][ok].mean())), "H": float(np.sqrt(pe[(H, s)]["mse"][ok].mean()))},
                             "rmse_group": {"B": pe[(ARM_B, s)]["rmse_group"], "H": pe[(H, s)]["rmse_group"]},
                             "mse_delta": PGT.paired(pairfn, pe[(H, s)]["mse"][ok], pe[(ARM_B, s)]["mse"][ok], eid[ok], a.n_boot)},
            "latcons_slots": slot_block(pairfn, A, Bm, ref, WC, D, eid, a.n_boot),
            "eval_json": {k: {kk: Z[(nm, s)]["json"].get(kk) for kk in ("traj", "families", "strategic", "tactical") if kk in Z[(nm, s)]["json"]}
                          for k, (nm, Z) in {"B": (ARM_B, Zb), "H": (H, Zh)}.items()},
            "distance_keeping_k10": {"pair_vs_b": res10["read"]["pairs"].get(f"{H}_s{s}__minus__{ARM_B}_s{s}"), "pair_vs_p10": res10["read"]["pairs"].get(f"{H}_s{s}__minus__P10"),
                                     "arms": {nm: res10["read"]["arms"][nm] for nm in (f"{H}_s{s}", f"{ARM_B}_s{s}", "P10")}},
            "parameters": params[s], "built": built[s], "pipeline": {k: {kk: vv for kk, vv in v.items() if kk != "ckpt"} for k, v in pipe[s].items()},
            "gt_source": {"B": Bm["gt_source"], "H": A["gt_source"]}}
        if famfn is not None:
            row["beside"]["families"] = famfn(A["pred"], Bm["pred"], eid)
        rows[f"s{s}"] = row
    k30 = None
    if not a.no_k30:
        p30 = a.lead_block_k30 or str(DK.LEAD_BLOCK_K30)
        if os.path.exists(p30):
            k30 = PGT._trim_dk(PGT.dk_read(a, DK, ctx, Zdk, DK.load_block(p30), 30, a.dk_n_boot, pairs)["read"])
    decision, vd, mech_all = verdict(a.stage, rows, seeds)
    nxt = NEXT[None if decision == "CONTINUE" else vd]
    fmt = lambda f: " / ".join(f(rows[f"s{s}"]) for s in seeds)
    prim = fmt(lambda r: f"{r['primary']['point']:+.3f}{'' if r['primary']['separated'] else ' ns'}")
    prim_sub = fmt(lambda r: f"{r['beside']['primary_t_ge_11']['point']:+.3f}{'' if r['beside']['primary_t_ge_11']['separated'] else ' ns'}")
    fall = fmt(lambda r: f"{100 * r['mlat']['fall']:+.1f}%{'' if r['mlat']['mse_delta']['separated'] else ' ns'}")
    disagree = any(not rows[f"s{s}"]["mlat"]["definitions_agree"] for s in seeds)
    out = dict(schema="v8f-hist-read-v1", prereg=f"PREREG_V8F_HIST.md stage {a.stage}", base=ARM_B,
               tier="T1-class open loop (eval-141, 2,399 windows); the mechanism read uses T0 absmask / GT-oracle evals",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); F = |{ARM_B}_s0 - {ARM_B}_s1| = {F:.4f} m (B's own two seeds); "
                         f"answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)",
               n_windows=int(eid.size), n_episodes=int(np.unique(eid).size), n_lead_windows_k10=int(ctx["lead"].sum()), F=F, two_F=2 * F, window_counts=wc,
               known_values=known, identity_control=idc, train_log={f"{k}": v for k, v in logsH.items()},
               strategic_family="NOT PAUSED: un-paused for v8F-mini (PI ruling 2026-10-17); read from each run's eval json under rows.*.beside.eval_json",
               distance_keeping_k30=k30, stage=a.stage, rows=rows, DECISION=decision, VERDICT=vd, mechanism=mech_all, next=nxt)
    out["reading"] = (f"PREREG_V8F_HIST stage {a.stage}: {decision or vd}; ADE6 H - B {prim} m (2F {2 * F:.4f}); on the t >= 11 subset {prim_sub} m; M-LAT latcons RMSE fall {fall} "
                      f"(bar {100 * MLAT_BAR:.1f}%; B {' / '.join(f'{rows[f's{s}']['mlat']['rmse_slot']['B']:.3f}' for s in seeds)} SD)"
                      f"{' [the slot-pooled and window-weighted definitions DISAGREE about the bar]' if disagree else ''}; mechanism {'MET' if mech_all else 'NOT met'}." +
                      ("" if vd is None else f" VERDICT {vd}."))
    return out


def _tanitad_data():
    from tanitad.data import v8f_mini_data as D                     # DK._paths() (called by PGT._modules) has put the stack on sys.path
    return D


def identity_control(pf, Zb_by_seed, pw, ctx, n_boot, WC, seeds, Zrel, sub, fam):
    """GRID's identity control (PRIMARY, M-LEAK, M-DK, present-slot, oracle ADE6, every family cell, M-LAT's cells: B against itself reads exactly +0 [0, 0]) plus this
    PREREG's own cells: the PRIMARY on the t >= 11 subset and on its complement, and the timing / geometry slot sets."""
    out = GR.identity_control(pf, Zb_by_seed, pw, ctx, n_boot, WC, seeds, Zrel, fam)
    eid = ctx["eid"]
    ref = Zrel[(ARM_B, seeds[0])]
    D = _tanitad_data()
    for s in seeds:
        A = Zb_by_seed[s]
        flat = {"primary_t_ge_11": subset_rows(pf, A, A, sub, eid, n_boot), "primary_t_lt_11": subset_rows(pf, A, A, ~sub, eid, n_boot)}
        sb = slot_block(pf, A, A, ref, WC, D, eid, n_boot)
        for key in ("timing4", "geometry8"):
            flat[f"slots_{key}_mse"] = sb["sets"][key]["mse_delta"]
            if sb["sets"][key]["fall"] not in (0.0, None):
                refuse(f"IDENTITY CONTROL FAILED: B_s{s} against itself reads a {key} RMSE fall of {sb['sets'][key]['fall']}, not exactly 0")
        for nm, v in sb["slots"].items():
            if v["fall"] not in (0.0, None):
                refuse(f"IDENTITY CONTROL FAILED: B_s{s} against itself reads a {nm} RMSE fall of {v['fall']}, not exactly 0")
        for name, v in flat.items():
            if v is None:
                continue
            if not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 and v["point"] == 0.0):
                refuse(f"IDENTITY CONTROL FAILED: B_s{s} - B_s{s} reads {v['delta']} [{v['lo']}, {v['hi']}] on {name}, not +0 [0, 0]: the paired instrument is broken")
            out[f"s{s}"][name] = {x: v[x] for x in ("delta", "lo", "hi", "point")}
    return out


def check_b_side(SB):
    """B's own facts the lever is defined against: a 4-frame run at the arm's auto hmax, the same at both seeds."""
    hm = {s: b_hmax(PGT._flags(SB[(ARM_B, s)]["argv.json"].get("argv", [])), SB[(ARM_B, s)]["summary.json"]) for s in (0, 1)}
    if len(set(hm.values())) != 1:
        refuse(f"B's two seeds disagree about hmax: {hm}")
    bh = {s: PGT._flags(SB[(ARM_B, s)]["argv.json"].get("argv", [])).get(HIST_FLAG, B_HIST_VAL) for s in (0, 1)}
    if set(bh.values()) != {B_HIST_VAL}:
        refuse(f"{ARM_B} is not a 4-frame run: argv {HIST_FLAG} {bh}")
    bw = {s: WS_FLAG in PGT._flags(SB[(ARM_B, s)]["argv.json"].get("argv", [])) for s in (0, 1)}
    if any(bw.values()):
        refuse(f"{ARM_B} names {WS_FLAG} in its argv {bw}: B is the banked M2b, which predates the flag")
    return hm[0]


def check_inputs(a, DK, WC, pairfn, famfn=None):
    """The B side alone, before any training: its runs and evals load, the PREREG's premises reproduce (B = M2b), the eval grid has the committed D4 counts, the paired
    instrument reads exactly 0 on B against itself, M-LAT, the t >= 11 subset and the per-slot sets included."""
    SB, logsB, Zb, eid = _common(a, WC)
    b_hm = check_b_side(SB)
    PGT.check_gt_codes(Zb, (0, 1), WC)
    F = abs(float(Zb[(ARM_B, 0)]["ade6"].mean()) - float(Zb[(ARM_B, 1)]["ade6"].mean()))
    ctx = PGT.build_ctx(a, DK, eid)
    t = window_t(ctx)
    wc = window_counts(t)
    sub = t >= T_SUB_MIN
    res10 = PGT.dk_read(a, DK, ctx, Zb, ctx["blk10"], 10, a.dk_n_boot, [])
    pw = res10["perwindow"]
    base_st = {s: PGT.baseline_stats(pairfn, Zb[(ARM_B, s)], pw, ctx, a.n_boot) for s in (0, 1)}
    known = PGT.known_values(base_st, F, (0, 1))
    idc = identity_control(pairfn, {s: Zb[(ARM_B, s)] for s in (0, 1)}, pw, ctx, a.n_boot, WC, (0, 1), Zb, sub, famfn)
    return {"base": ARM_B, "F": F, "b_hmax": b_hm, "b_hist": int(B_HIST_VAL), "window_counts": wc,
            "lambda_b": {s: base_st[s]["lambda_m2b_m"] for s in (0, 1)}, "excess_p10": {s: base_st[s]["excess_p10"] for s in (0, 1)}, "n_lead_k10": int(ctx["lead"].sum()),
            "latcons_rmse_slot_b": {s: idc[f"s{s}"]["latcons_rmse_slot"] for s in (0, 1)}, "latcons_rmse_window_b": {s: idc[f"s{s}"]["latcons_rmse_window"] for s in (0, 1)},
            "b_n_windows": {s: SB[(ARM_B, s)]["summary.json"].get("n_windows") for s in (0, 1)}, "b_steps": {s: SB[(ARM_B, s)]["summary.json"].get("steps") for s in (0, 1)},
            "gt_source": {f"{k[0]}_s{k[1]}": v["gt_source"] for k, v in Zb.items()}, "train_log": {f"{k[0]}_s{k[1]}": v for k, v in logsB.items()},
            "known_values": known, "identity_control": idc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("s0", "final"))
    ap.add_argument("--check-inputs", action="store_true", help="validate the B side alone (and the identity control) and exit")
    ap.add_argument("--eval-dir")
    ap.add_argument("--runs-dir")
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=B, help="B of the deciding contrasts (default 2,000)")
    ap.add_argument("--dk-n-boot", type=int, default=B, help="B of the descriptive distance-keeping blocks (read_dk)")
    ap.add_argument("--no-families", action="store_true", help="skip the descriptive family blocks (tests)")
    ap.add_argument("--no-k30", action="store_true", help="skip the 6 s distance-keeping block (descriptive only)")
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
                json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            wc = res["window_counts"]
            print(f"HIST INPUTS OK: base {ARM_B} (b_hist {res['b_hist']}, b_hmax {res['b_hmax']}); F {res['F']:.4f}; LAMBDA(B) {res['lambda_b'][0]:+.4f} / {res['lambda_b'][1]:+.4f}; "
                  f"latcons RMSE(B) {res['latcons_rmse_slot_b'][0]:.4f} / {res['latcons_rmse_slot_b'][1]:.4f} SD; windows {wc['total']} (t>=11 {wc['t_ge_11']}, "
                  f"t=3 {wc['t3']}, t=7 {wc['t7']}); B windows {res['b_n_windows'][0]}, steps {res['b_steps'][0]}; identity control exactly +0 [0, 0]")
            return 0
        if a.stage is None:
            ap.error("--stage is required unless --check-inputs")
        res = read(a, pairfn, None if a.no_families else PGT._famfn(a), DK, WC)
    except (OSError, KeyError, ValueError) as e:
        print(f"HIST READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    except Exception as e:                           # an inherited instrument's OWN Refusal (v8f_dk_read.Refusal, raised by DK.join_block): still a refusal, rc 2, never a traceback
        if type(e).__name__ != "Refusal":
            raise
        print(f"HIST READ REFUSED: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(DK._clean(res), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
