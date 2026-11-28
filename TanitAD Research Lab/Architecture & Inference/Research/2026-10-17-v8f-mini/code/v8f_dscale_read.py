"""PREREG_V8F_DSCALE, committed before its data: is v8F-mini's T (the tactical layer) DATA-LIMITED? Q = M2b trained on the first half of B's clips (`--data-frac 0.5`, ONE lever) at B's step budget.

  python v8f_dscale_read.py --eval-dir <eval> --runs-dir <runs> --train-manifest <cache train manifest.jsonl> --out dscale_read.json
  python v8f_dscale_read.py --check-inputs --eval-dir ... --runs-dir ... --train-manifest ... --out x.json        (the B side alone, before any training)

B = M2b, the banked runs `M2b_s{0,1}` and their evals (seed 1 only for the seed floor F). Q = B + `--data-frac 0.5` and nothing else (run dir `M2b_s0_df50`; eval tags `M2b-df50_s0`, suffixes
"", "-tacabs", "-tacorc", "-tacorc-all-latcons"). SEED 0 ONLY: the deciding read is on T, whose seed floor is 0.2 %, and its bar is about 20x that floor; ADE6 at one seed decides nothing (H-V8F-SEED-1).
`select_train_clips` takes the first round(0.5 x N) of B's clips in sha12 order (a nested half). The step budget is B's: S = ceil(n_windows_all_clips / batch) counts ALL clips, so Q trains 19,164 steps
and sees each of its windows about twice as often as B. B - Q is "doubling unique data at fixed steps". Eval: `v8f_mini_eval.py run` on eval-141 (2,399 windows, 141 episodes), T1-class open loop; the
T read uses the absmask and GT-oracle evals (T0), as in HIST.

  DECIDING READ (D-T)   r = RMSE(Q) / RMSE(B) - 1 of T's slot-pooled present-slot RMSE on `latcons` (the M-LAT instrument of GRID / HIST, IMPORTED: `v8f_grid_read.mlat_row`), paired episode-cluster
                        bootstrap (B = 2,000) on the per-window latcons MSE for the separation.
      DATA-LIMITED      r >= +5 %, MSE delta separated      -> D3 (more clips)
      NOT DATA-LIMITED  -5 % < r < +5 %, or the MSE delta not separated  -> LoRA on DINOv3 first
      EPOCH-LIMITED     r <= -5 %, MSE delta separated      -> more epochs at full data
  (a separated delta whose sign contradicts r -- the slot-pooled RMSE and the window-weighted MSE disagree -- counts as NOT separated in r's direction, and is reported.)
Reported beside it, deciding nothing: ADE6 Q_0 - B_0 with its CI against 2F; the four families; T's present-slot RMSE per group (latcons, lonspeed, stoplead) and per latcons slot; T's TRAINING constraint
loss (`cons`) over the last 10 % of steps, Q against B; the latcons leak LAMBDA; the GT-code oracle ADE6; both arms' parameter counts, window counts, steps.
Refusals (rc 2): a missing file; runs not on one clip list / not withholding the RA-1 holdout / not done / different step counts; Q's argv not B's plus `--data-frac 0.5`; the built clip count not
round(0.5 x N_B), or the clips not the first half of B's list in sha12 order (checked against the cache manifest, whose B list must itself reproduce B's recorded train_sha256); a step count not B's
19,164; a parameter count or any loss weight (`w_diff`, IMPORTED from GRID Amendment 2) different from B's; evals not on one grid; GT codes that differ. ADDED GUARDS (same direction; they refuse,
they never decide): Q's built hist / hmax / window_start and checkpoint are B's; Q's loop-logged p_gt is B's; every Q eval names Q's checkpoint; the identity control (B against itself reads exactly
+0 [0, 0] on every contrast, and r = 0 -> NOT DATA-LIMITED). Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (EPISODES): it answers "another draw of EPISODES"; the
training-seed question is answered by B's measured seed floor (H-ESTIM-SEED-1). v8F-mini's planner is deterministic, so no inference-seed floor applies."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
RAW = PKG / "raw"
DF_FLAG, DF_VAL, FRAC = "--data-frac", "0.5", 0.5
SUF = "_df50"
ARM_B, ARM_Q = "M2b", "M2b-df50"
STEPS_B = 19164                                  # the PREREG's budget, asserted against B's own record and against what the half-data run trained
BAR = 0.05                                       # D-T: |r| must reach 5 % (about 20 x B's 0.2 % seed floor, 4 x the largest change any one-lever arm produced)
TAIL = 0.10                                      # T's training constraint loss: the last 10 % of steps
HORIZON, B_START = 30, 3                         # windows per clip = len(range(3, n_frames - 30)): train_windows at B's hist 4 (start hist - 1 = 3); pinned to the data module by a test
B_HIST = 4
B = 2000
EVAL_SUFFIXES = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")
EVAL_EPISODES = 141
# The committed next-step cells, VERBATIM from PREREG_V8F_DSCALE.md's decision table (a test compares them with the file).
NEXT_TEXT = {"DATA-LIMITED": "**D3**: extend the training clips from the corpus (v9 labels from the ego log, the DINOv3 cache), and run M2b at 2× unique clips under its own PREREG, with the PRIMARY 2F bar at both seeds. D3's predicted effect on T is −r (ESTIMATED, log-linear, ≤ 2× extrapolation).",
             "NOT DATA-LIMITED": "halving the data does not move T in this range, so 2× data is not predicted to: **LoRA on DINOv3** first (the PI's AR-2 go), under its own PREREG, with D3 behind it",
             "EPOCH-LIMITED": "repeated windows help T more than new ones: **more epochs at full data** (B at 2× steps, the other half of `H-DATA-SCALE-1`), under its own PREREG"}
NEXT = {"DATA-LIMITED": "DATA-LIMITED: T's latcons RMSE rose by at least 5 % when the unique clips were halved at fixed steps, separated. Next: " + NEXT_TEXT["DATA-LIMITED"],
        "NOT DATA-LIMITED": "NOT DATA-LIMITED: T's latcons RMSE moved by less than 5 % (or not separably) when the unique clips were halved at fixed steps. Next: " + NEXT_TEXT["NOT DATA-LIMITED"],
        "EPOCH-LIMITED": "EPOCH-LIMITED: T's latcons RMSE FELL by at least 5 % when the unique clips were halved at fixed steps, separated. Next: " + NEXT_TEXT["EPOCH-LIMITED"]}


def refuse(msg):
    print(f"DSCALE READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# The programme's own instruments, IMPORTED (never copied). HIST's reader brings GRID's reader (the M-LAT machinery, `w_diff`, the identity control) and PGT's (run identity, eval loading, the GT-code
# check, the K = 10 lead block, the paired contrasts), plus the per-slot and checkpoint helpers. All are PRIVATE instances; their refusals are routed through THIS reader's refuse (one prefix, one rc).
HR = _load("v8f_hist_read_for_dscale", HERE.parent / "v8f_hist_read.py")
GR, PGT = HR.GR, HR.PGT
HR.refuse = GR.refuse = PGT.refuse = refuse
w_diff, B_UNRECORDED_W, _SKIP_LEVERS = GR.w_diff, GR.B_UNRECORDED_W, GR._SKIP_LEVERS


def _modules():
    return PGT._modules()


def q_dir(s=0):
    return f"{ARM_B}_s{s}{SUF}"


# ------------------------------------------------------------------------------------------------ the clip list (the cache's train manifest)
def sha12(clip):
    """tanitad.data.v8f_mini_data.sha12, restated so the reader needs no stack import; a test pins them equal."""
    return hashlib.sha256(clip.encode("utf-8")).hexdigest()[:12]


def sha_list(ids):
    """RA-1's clips digest as the trainer writes it into summary.json: newline-joined."""
    return hashlib.sha256(chr(10).join(ids).encode("utf-8")).hexdigest()


def half_n(n, frac=FRAC):
    """select_train_clips's count: round(frac x N), at least 1; all of them at frac >= 1."""
    return n if frac >= 1.0 else max(1, int(round(frac * n)))


def read_manifest(path):
    """The cache's train manifest -> [(clip, n_frames or None)] in file order (the last line per clip wins, as v8f_mini_data.read_manifest). A line without n_frames gives None."""
    ents = {}
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    d = json.loads(ln)
                    ents[d["clip"]] = int(d["n_frames"]) if d.get("n_frames") is not None else None
    except (OSError, ValueError, KeyError) as e:
        refuse(f"cannot read the train manifest {path}: {type(e).__name__}: {e}")
    if not ents:
        refuse(f"the train manifest {path} is empty")
    return list(ents.items())


def windows_of(ents, clips):
    """The trainer's window count for `clips` at B's hist 4: sum of len(range(3, n_frames - 30)); None when the manifest carries no frame counts."""
    nf = dict(ents)
    if any(nf[c] is None for c in clips):
        return None
    return int(sum(len(range(B_START, nf[c] - HORIZON)) for c in clips))


def expected_from_manifest(ents, exclude, frac=FRAC):
    """What the trainer's build_data would hand `select_train_clips` for B (frac 1) and for Q: B's list is the manifest minus `--exclude lo:hi` of its sha12 order; Q's is the first round(frac x N_B) of it."""
    clips_all = [c for c, _ in ents]
    excluded = []
    if exclude:
        lo, hi = (int(x) for x in str(exclude).split(":"))
        excluded = sorted(clips_all, key=sha12)[lo:hi]
        ex = set(excluded)
        clips_all = [c for c in clips_all if c not in ex]
    order = sorted(clips_all, key=sha12)
    n = half_n(len(order), frac)
    half = order[:n]
    return {"n_manifest": len(ents), "n_b": len(order), "n_half": n, "b_sha256": sha_list(order), "excluded_sha256": sha_list(excluded) if excluded else None, "half_sha256": sha_list(half),
            "first_sha12": sha12(half[0]), "last_sha12": sha12(half[-1]), "windows_b": windows_of(ents, order), "windows_q": windows_of(ents, half), "half": half, "frame_counts": all(x is not None for _, x in ents)}


def check_manifest_vs_b(exp, rsm, rfl):
    """The manifest must BE the list B trained on: its B list reproduces B's recorded train_sha256, clip count, holdout and (when it carries frame counts) window count -- else it proves nothing about Q's half."""
    bad = []
    if exp["b_sha256"] != rsm.get("train_sha256"):
        bad.append(f"the manifest's B list (sorted by sha12, minus --exclude {rfl.get('--exclude')}) hashes to {exp['b_sha256'][:16]}, B's recorded train_sha256 is {str(rsm.get('train_sha256'))[:16]}: not the list B trained on")
    if exp["n_b"] != rsm.get("n_clips"):
        bad.append(f"the manifest gives N_B = {exp['n_b']}, B trained on {rsm.get('n_clips')} clips")
    if exp["excluded_sha256"] != rsm.get("excluded_sha256"):
        bad.append("the manifest's excluded clips are not the holdout B withheld")
    if exp["windows_b"] is not None and exp["windows_b"] != rsm.get("n_windows"):
        bad.append(f"the manifest's frame counts give {exp['windows_b']} windows for B, B's record says {rsm.get('n_windows')}")
    if bad:
        refuse("; ".join(bad))


# ------------------------------------------------------------------------------------------------ runs
def check_b_side(SB):
    """B's own facts the lever is defined against: full data, the PREREG's step budget, a 4-frame run at the arm's auto hmax, the same at both seeds. Returns B's hmax."""
    hm = {s: HR.b_hmax(PGT._flags(SB[(ARM_B, s)]["argv.json"].get("argv", [])), SB[(ARM_B, s)]["summary.json"]) for s in (0, 1)}
    if len(set(hm.values())) != 1:
        refuse(f"B's two seeds disagree about hmax: {hm}")
    for s in (0, 1):
        fl, sm = PGT._flags(SB[(ARM_B, s)]["argv.json"].get("argv", [])), SB[(ARM_B, s)]["summary.json"]
        if DF_FLAG in fl and not PGT._same_float(fl[DF_FLAG], 1.0):
            refuse(f"{ARM_B}_s{s}: argv {DF_FLAG} {fl[DF_FLAG]}: B is the full-data run the lever is defined against")
        if not PGT._same_float(sm.get("data_frac", 1.0), 1.0):
            refuse(f"{ARM_B}_s{s}: summary.json data_frac {sm.get('data_frac')}: B is not a full-data run")
        if fl.get("--hist", str(B_HIST)) != str(B_HIST):
            refuse(f"{ARM_B}_s{s}: argv --hist {fl.get('--hist')}: B is not a {B_HIST}-frame run")
        if sm.get("steps") != STEPS_B:
            refuse(f"{ARM_B}_s{s} trained {sm.get('steps')} steps, the PREREG's budget is {STEPS_B}")
    return hm[0]


def load_b_runs(runs_dir):
    return HR.load_b_runs(runs_dir)


def check_q_params(lv_b, lv_q, tag):
    """Q has B's parameters EXACTLY, group by group: the lever moves data, never the model."""
    pb, pq = lv_b.get("params") or {}, lv_q.get("params") or {}
    bad = []
    if not pb or not pq:
        bad.append("a run's levers carry no per-group parameter counts")
    else:
        for k in sorted(set(pb) | set(pq)):
            if pq.get(k) != pb.get(k):
                bad.append(f"group {k}: {pq.get(k)} vs B's {pb.get(k)}")
    if lv_q.get("n_params_trainable") != lv_b.get("n_params_trainable"):
        bad.append(f"n_params_trainable {lv_q.get('n_params_trainable')} vs B's {lv_b.get('n_params_trainable')}")
    if bad:
        refuse(f"{tag}: Q's parameter count is not B's: a second lever moved -- " + "; ".join(bad))
    return {"n_params_trainable": {"B": lv_b.get("n_params_trainable"), "Q": lv_q.get("n_params_trainable")}, "groups": {"B": pb, "Q": pq}}


def load_q_run(runs_dir, SB, exp, b_hm, s=0):
    d = os.path.join(runs_dir, q_dir(s))
    for f in ("summary.json", "argv.json", "selfcheck.json", "ckpt_last.pt", "train_log.jsonl"):
        if not os.path.exists(os.path.join(d, f)):
            refuse(f"{os.path.join(d, f)} is missing (the run did not finish, or was not pulled)")
    S = {f: HR._jload(os.path.join(d, f)) for f in ("summary.json", "argv.json", "selfcheck.json")}
    tag = f"{ARM_Q}_s{s}"
    sm, argv, sc = S["summary.json"], S["argv.json"].get("argv", []), S["selfcheck.json"]
    fl = PGT._flags(argv)
    ref = SB[(ARM_B, s)]
    rsm, rfl = ref["summary.json"], PGT._flags(ref["argv.json"].get("argv", []))
    info, lv_live = sc.get("info") or {}, sc.get("live") or {}
    n_b = rsm.get("n_clips")
    want_n = half_n(n_b)
    bad = []
    if sm.get("done") is not True:
        bad.append(f"{tag} is not done")
    if sm.get("arm") != "M2b" or sm.get("seed") != s:
        bad.append(f"{tag}/summary.json says {sm.get('arm')} seed {sm.get('seed')}, expected M2b seed {s}")
    if sm.get("excluded_sha256") != PGT.HOLDOUT_SHA:
        bad.append(f"{tag} did not withhold the RA-1 holdout")
    if fl.get("--arm") != "M2b":
        bad.append(f"{tag}: argv --arm {fl.get('--arm')}, expected M2b")
    if fl.get(DF_FLAG) != DF_VAL:
        bad.append(f"{tag}: argv {DF_FLAG} {fl.get(DF_FLAG)}, expected {DF_VAL}")
    drop = {"--out", DF_FLAG}
    fq, fb = {k: x for k, x in fl.items() if k not in drop}, {k: x for k, x in rfl.items() if k not in drop}
    if fq != fb:
        diff = sorted(k for k in set(fq) | set(fb) if fq.get(k) != fb.get(k))
        bad.append(f"{tag}: argv differs from {ARM_B}_s{s}'s in {diff} besides {DF_FLAG} and --out -- not one lever")
    # ---- the data: the built clip count is round(0.5 x N_B), and the clips ARE the first half of B's list in sha12 order
    if not PGT._same_float(sm.get("data_frac"), FRAC):
        bad.append(f"{tag}: summary.json data_frac {sm.get('data_frac')}, expected {FRAC}")
    if sm.get("n_clips") != want_n or want_n != exp["n_half"]:
        bad.append(f"{tag} trained on {sm.get('n_clips')} clips, round({FRAC} x N_B = {n_b}) = {want_n} (the manifest's half: {exp['n_half']})")
    if info.get("n_clips") != want_n or info.get("n_clips_expected") != want_n or info.get("n_clips_all") != n_b:
        bad.append(f"{tag}: selfcheck.json built {info.get('n_clips')} clips of {info.get('n_clips_all')} (declared {info.get('n_clips_expected')}); expected {want_n} of {n_b}")
    if sm.get("train_sha256") != exp["half_sha256"] or (info.get("train_sha256") is not None and info.get("train_sha256") != exp["half_sha256"]):
        bad.append(f"{tag}: the clips are not the first {want_n} of B's list in sha12 order (train_sha256 {str(sm.get('train_sha256'))[:16]}, the half hashes to {exp['half_sha256'][:16]})")
    if sm.get("train_sha256") == rsm.get("train_sha256"):
        bad.append(f"{tag} trained on B's whole clip list: the lever did not reach the data")
    # ---- the budget: B's, because S counts the windows of ALL clips
    if sm.get("steps") != rsm.get("steps") or sm.get("steps") != STEPS_B:
        bad.append(f"{tag} trained for {sm.get('steps')} steps, {ARM_B}_s{s} for {rsm.get('steps')} (the PREREG's budget: {STEPS_B})")
    if sm.get("S") != rsm.get("S") or sm.get("steps_mult") != rsm.get("steps_mult"):
        bad.append(f"{tag}: S {sm.get('S')} x steps-mult {sm.get('steps_mult')} differ from B's {rsm.get('S')} x {rsm.get('steps_mult')}")
    if info.get("n_windows_all_clips") != rsm.get("n_windows"):
        bad.append(f"{tag}: the budget was sized from {info.get('n_windows_all_clips')} windows, B's {rsm.get('n_windows')}: S must count ALL clips")
    if sc.get("steps_total") != rsm.get("steps"):
        bad.append(f"{tag}: selfcheck.json steps_total {sc.get('steps_total')} != B's {rsm.get('steps')}")
    nq = sm.get("n_windows")
    if exp["windows_q"] is not None:
        if nq != exp["windows_q"]:
            bad.append(f"{tag} trained on {nq} windows, the half's clips give {exp['windows_q']} on the manifest's frame counts")
    elif not (isinstance(nq, int) and 0 < nq < (rsm.get("n_windows") or 0)):
        bad.append(f"{tag} trained on {nq} windows, B on {rsm.get('n_windows')}: a half of the clips must give fewer")
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"{tag}: the launch gate's declared_vs_built is not PASS (selfcheck is not PASS)")
    if lv_live.get("pass") is not True or lv_live.get("control_prior_grad_sum") != 0.0:
        bad.append(f"{tag}: the launch gate's G-LIVE is not a pass with a frozen-prior control of exactly 0.0 (selfcheck is not PASS)")
    if sc.get("argv_sha") != sm.get("argv_sha"):
        bad.append(f"{tag}: selfcheck.json (argv_sha {sc.get('argv_sha')}) is not this run's (summary.json {sm.get('argv_sha')})")
    # ---- the model and the losses are B's
    for w, rec in (("summary", sm), ("selfcheck", sc)):
        lg, lb = rec.get("levers") or {}, rsm.get("levers") or {}
        diff = sorted(k for k in set(lg) & set(lb) if k not in _SKIP_LEVERS and lg[k] != lb[k])
        diff += [f"w.{k}" for k in w_diff(lg.get("w") or {}, lb.get("w") or {}, rfl)]
        if diff:
            bad.append(f"{tag}: {w}.json built levers {diff} differ from {ARM_B}_s{s}'s")
        for key, want in (("hist", B_HIST), ("window_start", B_START), ("hmax", b_hm)):
            if lg.get(key) is None:
                bad.append(f"{tag}: {w}.json levers record no built {key}: the lever is unproven")
            elif lg.get(key) != want:
                bad.append(f"{tag}: {w}.json built {key} {lg.get(key)}, expected {want} (B's)")
        built_pg = (lg.get("p_gt_start"), lg.get("p_gt_end"))
        if None in built_pg:
            bad.append(f"{tag}: {w}.json levers record no built p_gt_start / p_gt_end: the inherited schedule is unproven")
        elif not (PGT._same_float(built_pg[0], PGT.M2B_START) and PGT._same_float(built_pg[1], PGT.M2B_END)):
            bad.append(f"{tag} was built with p_gt {built_pg[0]} -> {built_pg[1]} ({w}.json), B's is {PGT.M2B_START} -> {PGT.M2B_END}")
    params = {w: check_q_params(rsm.get("levers") or {}, rec.get("levers") or {}, f"{tag} ({w}.json)") for w, rec in (("summary", sm), ("selfcheck", sc))}
    ck = HR.ckpt_facts(os.path.join(d, "ckpt_last.pt"))
    if ck["hist"] != B_HIST or ck["time_rows"] != B_HIST:
        bad.append(f"{tag}: the checkpoint was BUILT at hist {ck['hist']} with a {ck['time_rows']}-row time table, B's is {B_HIST}")
    if ck["hmax"] != b_hm:
        bad.append(f"{tag}: the checkpoint was BUILT with hmax {ck['hmax']}, B's is {b_hm}")
    if ck["window_start"] not in (None, -1):
        bad.append(f"{tag}: the checkpoint was BUILT with window_start {ck['window_start']}: the lever is the data fraction alone")
    if ck["grid"] != HR.B_GRID or ck["n_tok"] != HR.N_TOK_B or ck["pos_rows"] != HR.N_TOK_B:
        bad.append(f"{tag}: the checkpoint's grid {ck['grid']!r} / n_tok {ck['n_tok']} / pos rows {ck['pos_rows']} are not B's ({HR.B_GRID!r} / {HR.N_TOK_B} / {HR.N_TOK_B})")
    if ck["argv_sha"] != sm.get("argv_sha"):
        bad.append(f"{tag}: the checkpoint was written by another argv (argv_sha {ck['argv_sha']} vs summary.json {sm.get('argv_sha')})")
    if ck["step"] != sm.get("steps"):
        bad.append(f"{tag}: the checkpoint is at step {ck['step']}, the run's {sm.get('steps')}")
    if ck["arm"] != "M2b" or ck["seed"] != s:
        bad.append(f"{tag}: the checkpoint says arm {ck['arm']} seed {ck['seed']}")
    if bad:
        refuse("; ".join(bad))
    log = PGT._train_log_check(os.path.join(d, "train_log.jsonl"), ARM_Q, s, sm.get("steps"), PGT.M2B_START, PGT.M2B_END, True)
    pipe = {"Q": {"lru": info.get("lru"), "seconds": sm.get("seconds"), "steps": sm.get("steps"), "n_windows": sm.get("n_windows"), "S": sm.get("S"), "n_clips": sm.get("n_clips"),
                  "steps_per_s": (sm["steps"] / sm["seconds"]) if sm.get("seconds") else None},
            "B": {"lru": (ref["selfcheck.json"].get("info") or {}).get("lru"), "seconds": rsm.get("seconds"), "steps": rsm.get("steps"), "n_windows": rsm.get("n_windows"), "S": rsm.get("S"),
                  "n_clips": rsm.get("n_clips"), "steps_per_s": (rsm["steps"] / rsm["seconds"]) if rsm.get("seconds") else None}}
    return S, log, pipe, params["summary"], {"n_clips": sm.get("n_clips"), "n_clips_all": info.get("n_clips_all"), "data_frac": sm.get("data_frac"), "first_sha12": exp["first_sha12"], "last_sha12": exp["last_sha12"],
                                            "train_sha256": sm.get("train_sha256"), "hist": ck["hist"], "hmax": ck["hmax"], "window_start": ck["window_start"]}


def check_eval_ckpts(eval_dir, s=0):
    """Every Q eval names the Q checkpoint it scored (the eval json's `ckpt`): an eval of B's checkpoint under Q's tag is refused."""
    bad = []
    for suf in EVAL_SUFFIXES:
        p = os.path.join(eval_dir, f"eval_{ARM_Q}_s{s}{suf}.json")
        ck = HR._jload(p).get("ckpt")
        parts = str(ck or "").replace("\\", "/").rstrip("/").split("/")
        if len(parts) < 2 or parts[-2] != q_dir(s) or parts[-1] != "ckpt_last.pt":
            bad.append(f"{os.path.basename(p)} scored {ck!r}, not <runs>/{q_dir(s)}/ckpt_last.pt")
    if bad:
        refuse("; ".join(bad))


# ------------------------------------------------------------------------------------------------ the deciding read
def decide(r, d, bar=BAR):
    """The PREREG's three-way table. `r` = RMSE(Q)/RMSE(B) - 1; `d` = the paired MSE delta (Q - B). A separated delta AGAINST r's direction does not support r."""
    sep = bool(d["separated"])
    if sep and r >= bar and d["point"] > 0:
        return "DATA-LIMITED"
    if sep and r <= -bar and d["point"] < 0:
        return "EPOCH-LIMITED"
    return "NOT DATA-LIMITED"


def dt_block(ml):
    """D-T from GRID's M-LAT row: Q is GRID's `G` slot there. r on the slot-pooled RMSE (the deciding definition), the window-weighted RMSE beside it."""
    rq, rb = ml["rmse_slot"]["G"], ml["rmse_slot"]["B"]
    wq, wb = ml["rmse_window"]["G"], ml["rmse_window"]["B"]
    r, rw = rq / rb - 1.0, wq / wb - 1.0
    if abs(r + ml["fall"]) > 1e-12:
        refuse(f"D-T: r {r} is not minus M-LAT's fall {ml['fall']}: the instrument and the read disagree")
    d = ml["mse_delta"]
    out, out_w = decide(r, d), decide(rw, d)
    return {"r": r, "r_percent": 100.0 * r, "bar": BAR, "rmse_slot": {"Q": rq, "B": rb}, "r_window": rw, "rmse_window": {"Q": wq, "B": wb}, "mse_delta": d,
            "n_windows_present": ml["n_windows_present"], "n_slots_present": ml["n_slots_present"], "outcome": out, "outcome_window_definition": out_w, "definitions_agree": bool(out == out_w)}


def tail_cons(path, steps, frac=TAIL):
    """T's TRAINING constraint loss (`cons`, the loop's own log) averaged over the records of the last `frac` of the steps."""
    if not os.path.exists(path):
        return {"status": "NOT-AVAILABLE", "reason": f"{path} is not staged"}
    recs = [json.loads(x) for x in open(path, encoding="utf-8").read().splitlines() if x.strip()]
    cut = (1.0 - frac) * steps
    tail = [r for r in recs if r["step"] > cut]
    if not tail or any("cons" not in r for r in tail):
        return {"status": "NOT-AVAILABLE", "reason": "no log record in the last 10 % of steps carries `cons`"}
    v = [float(r["cons"]) for r in tail]
    return {"status": "OK", "n_records": len(tail), "first_step": tail[0]["step"], "last_step": tail[-1]["step"], "cons_mean": float(np.mean(v)), "cons_sd": float(np.std(v))}


def train_cons_block(runs_dir, steps):
    q = tail_cons(os.path.join(runs_dir, q_dir(0), "train_log.jsonl"), steps)
    b = tail_cons(os.path.join(runs_dir, f"{ARM_B}_s0", "train_log.jsonl"), steps)
    out = {"window": f"last {int(100 * TAIL)} % of {steps} steps (records with step > {(1 - TAIL) * steps:.1f})", "Q": q, "B": b}
    if q["status"] == "OK" and b["status"] == "OK" and b["cons_mean"] > 0:
        out["Q_over_B_minus_1"] = q["cons_mean"] / b["cons_mean"] - 1.0
        out["Q_minus_B"] = q["cons_mean"] - b["cons_mean"]
    return out


def _rename_h(x):
    """HIST's slot_block names the treatment arm `H`; here it is `Q`."""
    if isinstance(x, dict):
        return {("Q" if k == "H" else k): _rename_h(v) for k, v in x.items()}
    return x


def group_block(pairfn, A, Bm, ref, WC, pe_q, pe_b, eid, n_boot):
    """T's present-slot RMSE per group (latcons, lonspeed, stoplead): Q against B, with the paired per-window MSE delta. DESCRIPTIVE."""
    out = {}
    has, g = ref["has"], ref["g_cons"]
    for name, cols in WC.GROUPS.items():
        cols = list(cols)
        rq, rb = pe_q["rmse_group"][name], pe_b["rmse_group"][name]
        ma, mb = HR.set_window_mse(A, g, has, cols), HR.set_window_mse(Bm, g, has, cols)
        ok = np.isfinite(mb)
        out[name] = {"rmse": {"B": rb, "Q": rq}, "r": (rq / rb - 1.0) if (rq is not None and rb) else None, "n_windows_present": int(ok.sum()),
                     "mse_delta": PGT.paired(pairfn, ma[ok], mb[ok], eid[ok], n_boot) if ok.any() else None}
    return out


def identity_control(pf, Zb_by_seed, pw, ctx, n_boot, WC, seeds, Zrel, fam):
    """GRID's identity control (PRIMARY, M-LEAK, M-DK, present-slot, oracle ADE6, every family cell, M-LAT's cells: B against itself reads exactly +0 [0, 0]) plus this PREREG's own cells: the per-slot
    sets, and D-T itself -- B against itself reads r = 0 exactly and the outcome NOT DATA-LIMITED."""
    out = GR.identity_control(pf, Zb_by_seed, pw, ctx, n_boot, WC, seeds, Zrel, fam)
    eid = ctx["eid"]
    ref = Zrel[(ARM_B, seeds[0])]
    D = HR._tanitad_data()
    for s in seeds:
        A = Zb_by_seed[s]
        sb = HR.slot_block(pf, A, A, ref, WC, D, eid, n_boot)
        for key in ("timing4", "geometry8"):
            v = sb["sets"][key]["mse_delta"]
            if sb["sets"][key]["fall"] not in (0.0, None):
                refuse(f"IDENTITY CONTROL FAILED: B_s{s} against itself reads a {key} RMSE fall of {sb['sets'][key]['fall']}, not exactly 0")
            if v is not None:
                if not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 and v["point"] == 0.0):
                    refuse(f"IDENTITY CONTROL FAILED: B_s{s} - B_s{s} reads {v['delta']} [{v['lo']}, {v['hi']}] on slots_{key}_mse, not +0 [0, 0]: the paired instrument is broken")
                out[f"s{s}"][f"slots_{key}_mse"] = {x: v[x] for x in ("delta", "lo", "hi", "point")}
        for nm, v in sb["slots"].items():
            if v["fall"] not in (0.0, None):
                refuse(f"IDENTITY CONTROL FAILED: B_s{s} against itself reads a {nm} RMSE fall of {v['fall']}, not exactly 0")
        pe, _ = WC.present_error({(ARM_B, s): A}, (s,))
        dt = dt_block(GR.mlat_row(pf, A, A, pe[(ARM_B, s)], pe[(ARM_B, s)], ref, WC, eid, n_boot))
        if dt["r"] != 0.0 or dt["r_window"] != 0.0 or dt["outcome"] != "NOT DATA-LIMITED":
            refuse(f"IDENTITY CONTROL FAILED: B_s{s} against itself reads D-T r = {dt['r']} (window {dt['r_window']}) -> {dt['outcome']}, not exactly 0 -> NOT DATA-LIMITED")
        out[f"s{s}"]["dt"] = {"r": dt["r"], "r_window": dt["r_window"], "outcome": dt["outcome"]}
    return out


def _common(a, WC):
    SB, logsB = load_b_runs(a.runs_dir)
    Zb, eid = PGT.load_all(a.eval_dir, (0, 1), (ARM_B,), WC)
    return SB, logsB, Zb, eid


def _manifest_facts(a, SB):
    rsm = SB[(ARM_B, 0)]["summary.json"]
    rfl = PGT._flags(SB[(ARM_B, 0)]["argv.json"].get("argv", []))
    exp = expected_from_manifest(read_manifest(a.train_manifest), rfl.get("--exclude"))
    check_manifest_vs_b(exp, rsm, rfl)
    if exp["n_half"] != half_n(rsm.get("n_clips")):
        refuse(f"the manifest's half ({exp['n_half']} clips) is not round({FRAC} x N_B = {rsm.get('n_clips')})")
    return exp


def read(a, pairfn, famfn, DK, WC):
    s = 0
    SB, logsB, Zb, eid = _common(a, WC)
    b_hm = check_b_side(SB)
    exp = _manifest_facts(a, SB)
    SQ, logQ, pipe, params, built = load_q_run(a.runs_dir, SB, exp, b_hm, s)
    Zq, eid_q = PGT.load_all(a.eval_dir, (s,), (ARM_Q,), WC)
    if not np.array_equal(np.asarray(eid_q), np.asarray(eid)):
        refuse(f"the {ARM_Q} evals are not on the same eval grid as {ARM_B}'s (eid differs)")
    check_eval_ckpts(a.eval_dir, s)
    Zrel = {**Zb, **Zq}
    PGT.check_gt_codes(Zrel, (s,), WC)
    F = abs(float(Zb[(ARM_B, 0)]["ade6"].mean()) - float(Zb[(ARM_B, 1)]["ade6"].mean()))
    ctx = PGT.build_ctx(a, DK, eid)
    wc = HR.window_counts(HR.window_t(ctx))
    n_ep = int(np.unique(eid).size)
    if eid.size == HR.WINDOWS_COMMITTED["total"] and n_ep != EVAL_EPISODES:
        refuse(f"the eval grid has {eid.size} windows over {n_ep} episodes, the PREREG's eval-141 has {EVAL_EPISODES}")
    pairs = [(f"{ARM_Q}_s{s}", f"{ARM_B}_s{s}"), (f"{ARM_Q}_s{s}", "P10"), (f"{ARM_B}_s0", "P10"), (f"{ARM_B}_s1", "P10")]
    res10 = PGT.dk_read(a, DK, ctx, {**Zb, **Zq}, ctx["blk10"], 10, a.dk_n_boot, pairs)
    pw = res10["perwindow"]
    if not np.array_equal(np.asarray(pw["is_lead"], bool), ctx["lead"]):
        refuse("the LEAD windows of the distance-keeping read are not the lead block's join")
    base_st = {x: PGT.baseline_stats(pairfn, Zb[(ARM_B, x)], pw, ctx, a.n_boot) for x in (0, 1)}
    known = PGT.known_values(base_st, F, (0, 1))
    idc = identity_control(pairfn, {s: Zb[(ARM_B, s)]}, pw, ctx, a.n_boot, WC, (s,), Zrel, None)
    pe, _ = WC.present_error(Zrel, (s,))
    ref = Zrel[(ARM_B, s)]
    D = HR._tanitad_data()
    A, Bm = Zq[(ARM_Q, s)], Zb[(ARM_B, s)]
    row = PGT.contrast(pairfn, A, Bm, pw, ctx, a.n_boot)
    ml = GR.mlat_row(pairfn, A, Bm, pe[(ARM_Q, s)], pe[(ARM_B, s)], ref, WC, eid, a.n_boot)
    dt = dt_block(ml)
    row["dt"] = dt
    row["base"] = {k: base_st[s][k] for k in ("lambda_m2b_m", "lambda_ci", "n_lead", "excess_p10", "excess_ci")}
    ok = np.isfinite(pe[(ARM_B, s)]["mse"])
    primary = row["primary"]
    row["beside"] = {
        "ade6": {"B": float(Bm["ade6"].mean()), "Q": float(A["ade6"].mean()), "delta": primary,
                 "separated_and_beyond_2F_if_it_were_deciding": bool(PGT._is_neg_sep(primary) and abs(primary["point"]) > 2 * F),
                 "note": "ADE6 at one seed decides nothing (H-V8F-SEED-1); a positive delta is the wrong direction for a data-limited T only if it is separated and beyond 2F"},
        "oracle_ade6": {"B": float(Bm["orc_ade6"].mean()), "Q": float(A["orc_ade6"].mean()), "delta": PGT.paired(pairfn, A["orc_ade6"], Bm["orc_ade6"], eid, a.n_boot)},
        "present_slot": {"n_windows_present": int(ok.sum()), "rmse": {"B": float(np.sqrt(pe[(ARM_B, s)]["mse"][ok].mean())), "Q": float(np.sqrt(pe[(ARM_Q, s)]["mse"][ok].mean()))},
                         "rmse_group": {"B": pe[(ARM_B, s)]["rmse_group"], "Q": pe[(ARM_Q, s)]["rmse_group"]},
                         "mse_delta": PGT.paired(pairfn, pe[(ARM_Q, s)]["mse"][ok], pe[(ARM_B, s)]["mse"][ok], eid[ok], a.n_boot)},
        "groups": group_block(pairfn, A, Bm, ref, WC, pe[(ARM_Q, s)], pe[(ARM_B, s)], eid, a.n_boot),
        "latcons_slots": _rename_h(HR.slot_block(pairfn, A, Bm, ref, WC, D, eid, a.n_boot)),
        "train_constraint_loss": train_cons_block(a.runs_dir, SB[(ARM_B, s)]["summary.json"]["steps"]),
        "eval_json": {k: {kk: Z[(nm, s)]["json"].get(kk) for kk in ("traj", "families", "strategic", "tactical") if kk in Z[(nm, s)]["json"]} for k, (nm, Z) in {"B": (ARM_B, Zb), "Q": (ARM_Q, Zq)}.items()},
        "distance_keeping_k10": {"pair_vs_b": res10["read"]["pairs"].get(f"{ARM_Q}_s{s}__minus__{ARM_B}_s{s}"), "pair_vs_p10": res10["read"]["pairs"].get(f"{ARM_Q}_s{s}__minus__P10"),
                                 "arms": {nm: res10["read"]["arms"][nm] for nm in (f"{ARM_Q}_s{s}", f"{ARM_B}_s{s}", "P10")}},
        "parameters": params, "built": built, "pipeline": pipe,
        "data": {"n_clips": {"B": exp["n_b"], "Q": exp["n_half"]}, "windows": {"B": SB[(ARM_B, s)]["summary.json"].get("n_windows"), "Q": SQ["summary.json"].get("n_windows")},
                 "steps": {"B": SB[(ARM_B, s)]["summary.json"].get("steps"), "Q": SQ["summary.json"].get("steps")}, "first_sha12": exp["first_sha12"], "last_sha12": exp["last_sha12"]},
        "gt_source": {"B": Bm["gt_source"], "Q": A["gt_source"]}}
    tc = row["beside"]["train_constraint_loss"]
    row["beside"]["generalisation_signature"] = {
        "reads": "a generalisation-limited T fits its smaller set better (train `cons` falls) at worse eval (r > 0)",
        "train_cons_fell": (tc["Q_over_B_minus_1"] < 0) if "Q_over_B_minus_1" in tc else None, "eval_rmse_rose": bool(dt["r"] > 0)}
    if famfn is not None:
        row["beside"]["families"] = famfn(A["pred"], Bm["pred"], eid)
    k30 = None
    if not a.no_k30:
        p30 = a.lead_block_k30 or str(DK.LEAD_BLOCK_K30)
        if os.path.exists(p30):
            k30 = PGT._trim_dk(PGT.dk_read(a, DK, ctx, {**Zb, **Zq}, DK.load_block(p30), 30, a.dk_n_boot, pairs)["read"])
    out_name = dt["outcome"]
    d = dt["mse_delta"]
    out = dict(schema="v8f-dscale-read-v1", prereg="PREREG_V8F_DSCALE.md", base=ARM_B, arm=ARM_Q, seed=0,
               tier="T1-class open loop (eval-141, 2,399 windows); the T read uses T0 absmask / GT-oracle evals",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); F = |{ARM_B}_s0 - {ARM_B}_s1| = {F:.4f} m (B's own two seeds); answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)",
               n_windows=int(eid.size), n_episodes=n_ep, n_lead_windows_k10=int(ctx["lead"].sum()), F=F, two_F=2 * F, window_counts=wc, manifest=expected_summary(exp),
               known_values=known, identity_control=idc, train_log={ARM_Q: logQ}, strategic_family="NOT PAUSED: un-paused for v8F-mini (PI ruling 2026-10-17); read from rows.s0.beside.eval_json",
               distance_keeping_k30=k30, rows={"s0": row}, DECISION=out_name, VERDICT=out_name, next=NEXT[out_name])
    p = row["primary"]
    out["reading"] = (f"PREREG_V8F_DSCALE: {out_name}; D-T latcons RMSE B {dt['rmse_slot']['B']:.4f} -> Q {dt['rmse_slot']['Q']:.4f} SD, r {dt['r_percent']:+.2f}% (bar +/-{100 * BAR:.0f}%), "
                      f"MSE delta {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] {'separated' if d['separated'] else 'NOT separated'}; "
                      f"ADE6 Q - B {p['point']:+.3f} m{'' if p['separated'] else ' ns'} (2F {2 * F:.4f}); Q trained {SQ['summary.json'].get('n_clips')} clips / {SQ['summary.json'].get('n_windows')} windows / "
                      f"{SQ['summary.json'].get('steps')} steps" + ("" if dt["definitions_agree"] else " [the slot-pooled and window-weighted definitions DISAGREE about the outcome]") + ".")
    return out


def expected_summary(exp):
    return {k: v for k, v in exp.items() if k != "half"}


def check_inputs(a, DK, WC, pairfn, famfn=None):
    """The B side alone, before any training: its runs and evals load, the PREREG's premises reproduce (B = M2b), the manifest IS B's list and gives the half Q must train on, the eval grid has the
    committed counts, the paired instrument reads exactly 0 on B against itself (D-T included)."""
    SB, logsB, Zb, eid = _common(a, WC)
    b_hm = check_b_side(SB)
    exp = _manifest_facts(a, SB)
    PGT.check_gt_codes(Zb, (0, 1), WC)
    F = abs(float(Zb[(ARM_B, 0)]["ade6"].mean()) - float(Zb[(ARM_B, 1)]["ade6"].mean()))
    ctx = PGT.build_ctx(a, DK, eid)
    wc = HR.window_counts(HR.window_t(ctx))
    n_ep = int(np.unique(eid).size)
    if eid.size == HR.WINDOWS_COMMITTED["total"] and n_ep != EVAL_EPISODES:
        refuse(f"the eval grid has {eid.size} windows over {n_ep} episodes, the PREREG's eval-141 has {EVAL_EPISODES}")
    res10 = PGT.dk_read(a, DK, ctx, Zb, ctx["blk10"], 10, a.dk_n_boot, [])
    pw = res10["perwindow"]
    base_st = {s: PGT.baseline_stats(pairfn, Zb[(ARM_B, s)], pw, ctx, a.n_boot) for s in (0, 1)}
    known = PGT.known_values(base_st, F, (0, 1))
    idc = identity_control(pairfn, {s: Zb[(ARM_B, s)] for s in (0, 1)}, pw, ctx, a.n_boot, WC, (0, 1), Zb, famfn)
    rsm = SB[(ARM_B, 0)]["summary.json"]
    return {"base": ARM_B, "F": F, "b_hmax": b_hm, "b_hist": B_HIST, "window_counts": wc, "n_episodes": n_ep, "manifest": expected_summary(exp),
            "expected_q": {"n_clips": exp["n_half"], "n_clips_all": exp["n_b"], "steps": rsm.get("steps"), "S": rsm.get("S"), "n_windows_all_clips": rsm.get("n_windows"), "n_windows_q": exp["windows_q"],
                           "train_sha256": exp["half_sha256"], "first_sha12": exp["first_sha12"], "last_sha12": exp["last_sha12"]},
            "lambda_b": {s: base_st[s]["lambda_m2b_m"] for s in (0, 1)}, "excess_p10": {s: base_st[s]["excess_p10"] for s in (0, 1)}, "n_lead_k10": int(ctx["lead"].sum()),
            "latcons_rmse_slot_b": {s: idc[f"s{s}"]["latcons_rmse_slot"] for s in (0, 1)}, "latcons_rmse_window_b": {s: idc[f"s{s}"]["latcons_rmse_window"] for s in (0, 1)},
            "b_n_windows": {s: SB[(ARM_B, s)]["summary.json"].get("n_windows") for s in (0, 1)}, "b_steps": {s: SB[(ARM_B, s)]["summary.json"].get("steps") for s in (0, 1)},
            "b_train_constraint_loss_tail": tail_cons(os.path.join(a.runs_dir, f"{ARM_B}_s0", "train_log.jsonl"), rsm.get("steps")),
            "gt_source": {f"{k[0]}_s{k[1]}": v["gt_source"] for k, v in Zb.items()}, "train_log": {f"{k[0]}_s{k[1]}": v for k, v in logsB.items()}, "known_values": known, "identity_control": idc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-inputs", action="store_true", help="validate the B side alone (and the identity control) and exit")
    ap.add_argument("--eval-dir")
    ap.add_argument("--runs-dir")
    ap.add_argument("--train-manifest", help="the cache's train/manifest.jsonl (clip, n_frames): the list B trained on, which the half must be the first round(0.5 x N_B) of")
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
        for need in ("eval_dir", "runs_dir", "out", "train_manifest"):
            if getattr(a, need) is None:
                ap.error(f"--{need.replace('_', '-')} is required")
        DK, WC = _modules()
        a.p10_pred = a.p10_pred or str(DK.P10_DEFAULT)
        from taniteval.ci import paired_episode_cluster_bootstrap as pairfn
        famfn = None if a.no_families else PGT._famfn(a)
        if a.check_inputs:
            res = check_inputs(a, DK, WC, pairfn, famfn)
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            wc, eq = res["window_counts"], res["expected_q"]
            print(f"DSCALE INPUTS OK: base {ARM_B} (b_hist {res['b_hist']}, b_hmax {res['b_hmax']}); F {res['F']:.4f}; LAMBDA(B) {res['lambda_b'][0]:+.4f} / {res['lambda_b'][1]:+.4f}; "
                  f"latcons RMSE(B) {res['latcons_rmse_slot_b'][0]:.4f} / {res['latcons_rmse_slot_b'][1]:.4f} SD; windows {wc['total']} over {res['n_episodes']} episodes; "
                  f"B clips {eq['n_clips_all']} windows {res['b_n_windows'][0]} steps {res['b_steps'][0]}; Q must be {eq['n_clips']} clips (first {eq['first_sha12']} .. last {eq['last_sha12']}), "
                  f"{eq['n_windows_q']} windows, {eq['steps']} steps; identity control exactly +0 [0, 0]")
            return 0
        res = read(a, pairfn, famfn, DK, WC)
    except (OSError, KeyError, ValueError) as e:
        print(f"DSCALE READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    except Exception as e:                           # an inherited instrument's OWN Refusal (v8f_dk_read.Refusal, raised by DK.join_block): still a refusal, rc 2, never a traceback
        if type(e).__name__ != "Refusal":
            raise
        print(f"DSCALE READ REFUSED: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(DK._clean(res), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
