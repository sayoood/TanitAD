"""PREREG_V8F_GRID, committed before its data: does feeding the stack 8x20 scene tokens per frame instead of 4x10 (`--grid pool8x20`, ONE lever) make v8F-mini
drive better, and does it make T's LATERAL constraints (`latcons`) more accurate?

  python v8f_grid_read.py --stage s0|final --base M2b|M2b-pgt --pgt-dir <v8f_pgt dir> --eval-dir <eval> --runs-dir <runs> --out grid_read.json
  python v8f_grid_read.py --decide-base --pgt-dir <v8f_pgt dir> [--out base.json]       (the chain's base rule; prints BASE=...)
  python v8f_grid_read.py --check-inputs --base M2b|M2b-pgt --pgt-dir ... --eval-dir ... --runs-dir ... --out x.json   (the B side alone, before any training)

The base B (fixed in the PREREG before PREREG_V8F_PGT had data): B = M2b-pgt (`--arm M2b --p-gt-start 0.5 --p-gt-end 0.0`) iff PREREG_V8F_PGT's FINAL read says
`VERDICT == "SUCCESS"` (v8f_pgt_read.py writes `schema v8f-pgt-read-v1`, `stage final`, `VERDICT`); otherwise (any FAIL, including a seed-0 STOP) B = M2b.
G = B + `--grid pool8x20` and nothing else (run dir `<B>_s<SEED>_8x20`; eval tags `<B>-8x20_s<SEED>`). Eval: `v8f_mini_eval.py run` on eval-141 (2,399 windows,
141 episodes), "T1-class open loop"; the mechanism read uses the absmask and GT-oracle evals (T0) only to score T.

  PRIMARY   per seed s: delta_s = ADE6(G_s) - ADE6(B_s), paired over episodes. Seed s PASSES iff delta_s < 0, separated and |delta_s| > 2F, F = |mean ADE6 B_s0 -
            mean ADE6 B_s1| (B's seed floor from its own two seeds: 0.1207 m for M2b, banked).
  M-LAT     per seed (decides the reading of a FAIL, never the verdict): T's present-slot RMSE on `latcons` falls by >= 12.5 % relative to B_s, separated.
              size       1 - RMSE_G / RMSE_B  >= 0.125, RMSE = the group RMSE v8f_wc_read.present_error reports (`rmse_group["latcons"]`, pooled over the present
                         latcons slots: the 0.686 SD the PREREG quotes for M2b_s0 -- the PREREG's own number fixes which RMSE the bar is on).
              separated  the paired per-window latcons MSE difference G - B (mean over the window's present latcons slots; GT-present windows of BOTH arms, which are the
                         same windows) is < 0 with an episode-cluster interval below 0 -- v8f_wc_read's present-slot error, restricted to latcons.
            Beside it, deciding nothing: the same fall on the window-weighted RMSE (sqrt of the mean per-window MSE; 0.468 at M2b_s0, a different number) and whether the
            two definitions AGREE about the 12.5 % line (they coincide for a uniform shrink of the error, which is the dose-curve premise of the bar; they may not
            for a non-uniform one).
  STAGES    s0: seed 0 only. CONTINUE iff seed 0 PASSES the PRIMARY (only then is seed 1 trained); otherwise STOP with the seed-0 verdict. final: both seeds.
  VERDICTS  SUCCESS (both seeds pass) | FAIL-MECHANISM-OK (latcons accuracy fell >= 12.5 % at every seed read; driving did not move beyond 2F) | FAIL-NO-MECHANISM.
Reported beside it, deciding nothing: the four families (LON incl. distance keeping, LAT, TACTICAL and STRATEGIC -- un-paused for v8F-mini, PI ruling 2026-10-17);
T's present-slot RMSE for every group; the latcons leak LAMBDA and the 2 s low-TTC tail (PREREG_V8F_PGT's M-LEAK / M-DK contrasts, G minus B); both arms' parameter
counts (G adds exactly the position table, (160 - 40) x d) and steps per second; the trainer's LRU / chunk (sized from the token count: a pipeline difference, not a lever).
Refusals (rc 2): a missing file; runs not on one clip list / not withholding the RA-1 holdout / not done / different step counts; a G run whose argv is not B's plus
`--grid pool8x20`, whose built n_tok is not 160, whose checkpoint does not carry `grid: pool8x20`, or whose selfcheck is not PASS; a B side whose identity does not
match the PGT verdict; evals not on one grid of windows; GT codes that differ between the arms.
ADDED GUARDS (same direction; they refuse, they never decide): the parameter delta of G over B is EXACTLY the position table and nothing else (a second lever moved
otherwise); G's other built levers and its loop-logged p_gt equal B's; every G eval names the 8x20 checkpoint it scored; B's evals are the ones the PGT verdict was
computed from (their ADE6 equals what the PGT read recorded); the identity control (B against itself reads exactly +0 [0, 0] on every contrast, M-LAT included).
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
BASES = ("M2b", "M2b-pgt")
GRID_FLAG, GRID_VAL, B_GRID_VAL = "--grid", "pool8x20", "pool4x10"
N_TOK_B, N_TOK_G = 40, 160
SUF = "_8x20"
MLAT_BAR = 0.125                                # M-LAT: latcons RMSE must fall by at least this fraction of B's
PGT_SCHEMA = "v8f-pgt-read-v1"
DOSE = {"M2b": (1.0, 0.2), "M2b-pgt": (0.5, 0.0)}      # (p_gt_start, p_gt_end): B's schedule, which G inherits unchanged
B = 2000
EVAL_SUFFIXES = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")
NEXT = {None: "CONTINUE: seed 0 passes the PRIMARY; train seed 1 (the verdict needs both).",
        "SUCCESS": "SUCCESS: the stack at 8x20 drives better than B beyond 2F at both seeds. Next: the T1-kin read of G (stack/scripts/v8f_t1_eval.py, m in {30, 10}), "
                   "then the next plan 13.6 lever on top (more history frames), under its own PREREG.",
        "FAIL-MECHANISM-OK": "FAIL with the mechanism in place: latcons accuracy fell >= 12.5 % but driving did not move beyond 2F. T is getting more accurate but not "
                             "enough to convert: the encoder itself, LoRA on DINOv3 (needs the PI's AR-2 go).",
        "FAIL-NO-MECHANISM": "FAIL without the mechanism: latcons accuracy did not fall >= 12.5 %, so spatial resolution is not T's limit. Next lever: plan 13.6's "
                             "more history frames (temporal context for turn timing), under its own PREREG."}


def refuse(msg):
    print(f"GRID READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# The programme's own PGT instrument, IMPORTED (never copied): run identity, eval loading, the GT-code check, the K = 10 lead block, the paired contrasts, the
# identity control. Its refusals are routed through THIS reader's refuse (one message prefix, one rc) BEFORE its modules are built.
PGT = _load("v8f_pgt_read_for_grid", HERE.parent / "v8f_pgt_read.py")
PGT.refuse = refuse


def _modules():
    return PGT._modules()


def g_name(base):
    return f"{base}-8x20"


def g_dir(base, s):
    return f"{base}_s{s}{SUF}"


# ------------------------------------------------------------------------------------------------ the base rule
def _jload(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError) as e:
        refuse(f"cannot read {p}: {type(e).__name__}: {e}")


def decide_base(pgt_dir):
    """B = M2b-pgt iff PREREG_V8F_PGT's FINAL read says VERDICT == SUCCESS; otherwise M2b. Everything that is not one of the three committed shapes is refused:
    no read at all, a read of the wrong schema or stage, a seed-0 STOP that coexists with a final read, a seed-0 CONTINUE with no final read."""
    d = Path(pgt_dir)
    pf, ps = d / "pgt_read.json", d / "pgt_read_s0.json"
    final = _jload(pf) if pf.exists() else None
    s0 = _jload(ps) if ps.exists() else None
    if final is None and s0 is None:
        refuse(f"PREREG_V8F_PGT has no read in {d} (neither pgt_read.json nor pgt_read_s0.json): the base B cannot be decided")
    for name, r, stage in (("pgt_read.json", final, "final"), ("pgt_read_s0.json", s0, "s0")):
        if r is not None and (r.get("schema") != PGT_SCHEMA or r.get("stage") != stage):
            refuse(f"{name} is not a PGT {stage} read (schema {r.get('schema')!r}, stage {r.get('stage')!r})")
    if final is not None:
        if s0 is not None and s0.get("DECISION") != "CONTINUE":
            refuse(f"pgt_read_s0.json says DECISION {s0.get('DECISION')!r} but a final read exists: the PGT files contradict each other")
        v = final.get("VERDICT")
        if v not in ("SUCCESS", "FAIL-MECHANISM-OK", "FAIL-NO-MECHANISM"):
            refuse(f"pgt_read.json VERDICT {v!r} is none of the three committed verdicts")
        base = "M2b-pgt" if v == "SUCCESS" else "M2b"
        return {"base": base, "why": f"PREREG_V8F_PGT final read VERDICT {v}", "pgt_verdict": v, "pgt_stage": "final", "source": str(pf)}
    if s0.get("DECISION") == "STOP":
        return {"base": "M2b", "why": f"PREREG_V8F_PGT seed-0 STOP (VERDICT {s0.get('VERDICT')}): seed 1 was never trained", "pgt_verdict": s0.get("VERDICT"),
                "pgt_stage": "s0", "source": str(ps)}
    refuse(f"pgt_read_s0.json says DECISION {s0.get('DECISION')!r} and there is no final read: the PGT chain did not finish")


# ------------------------------------------------------------------------------------------------ runs
def ckpt_facts(path):
    """The BUILT model, read from the checkpoint the trainer saved: the grid it names, the config's n_tok and the position table's rows."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    pos = ck["model"].get("pos")
    return {"grid": ck.get("grid"), "n_tok": (ck.get("cfg") or {}).get("n_tok"), "pos_rows": None if pos is None else int(pos.shape[0]),
            "argv_sha": ck.get("argv_sha"), "step": ck.get("step"), "arm": ck.get("arm"), "seed": ck.get("seed")}


def check_params(lv_b, lv_g, d, tag):
    """G adds EXACTLY the position table: (160 - 40) x d parameters, all in the adapter group (where `pos` lives). Any other delta is a second lever."""
    want = (N_TOK_G - N_TOK_B) * d
    pb, pg = lv_b.get("params") or {}, lv_g.get("params") or {}
    bad = []
    if not pb or not pg:
        bad.append("a run's levers carry no per-group parameter counts")
    else:
        for k in sorted(set(pb) | set(pg)):
            dd = pg.get(k, 0) - pb.get(k, 0)
            if dd != (want if k == "adapter" else 0):
                bad.append(f"group {k}: {pg.get(k)} vs B's {pb.get(k)} (delta {dd}, expected {want if k == 'adapter' else 0})")
    dt = (lv_g.get("n_params_trainable") or 0) - (lv_b.get("n_params_trainable") or 0)
    if dt != want:
        bad.append(f"n_params_trainable delta {dt}, expected {want} = (160 - 40) x {d}")
    if bad:
        refuse(f"{tag}: G's parameter delta over B is not exactly the position table ((160 - 40) x d = {want}): a second lever moved -- " + "; ".join(bad))
    return {"d": d, "expected_delta": want, "delta": dt, "n_params_trainable": {"B": lv_b.get("n_params_trainable"), "G": lv_g.get("n_params_trainable")},
            "adapter": {"B": pb.get("adapter"), "G": pg.get("adapter")}}


_SKIP_LEVERS = ("params", "n_params_trainable", "residual_zero_init", "w")  # params: the lever; residual_zero_init: written before (selfcheck) / after (summary) training;
                                                                                # w: compared key by key in w_diff (Amendment 2)
# PREREG_V8F_GRID Amendment 2 (written after G s0 trained, BEFORE any GRID number was read): M2b's banked lever record PREDATES these two keys.
# built_levers began recording them at 39e21e5 (PREREG_V8F_WC), after M2b trained. Both arms train at the trainer's defaults: their argv carries
# neither flag, and v8f_mini_train.py declares --w-tac 1.0 / --w-cons 0.5 at every commit (4658d5c .. c483a28) and passes both into the loss.
# A key B's record LACKS is read at exactly that proven value, only while B's argv does not set its flag; every other weight compares as recorded.
B_UNRECORDED_W = {"w_tac": (1.0, "--w-tac"), "w_cons": (0.5, "--w-cons")}


def w_diff(wg, wb, b_flags):
    """The loss weights where G's built record differs from B's: a key in B must equal G's; a key only in G must be one of B_UNRECORDED_W, at its
    proven value, with B's argv silent on its flag. Anything else (a key only in B included) is a difference."""
    out = []
    for k in sorted(set(wg) | set(wb)):
        if k in wb:
            if k not in wg or wg[k] != wb[k]:
                out.append(k)
        elif k not in B_UNRECORDED_W or wg[k] != B_UNRECORDED_W[k][0] or B_UNRECORDED_W[k][1] in b_flags:
            out.append(k)
    return out


def load_b_runs(runs_dir, base):
    """B's two seeds, validated by PGT's own run check for that arm (flags, built schedule, one clip list, holdout, done, steps; M2b-pgt also its loop-logged p_gt)."""
    return PGT.load_runs(runs_dir, (0, 1), arms=(base,))


def load_g_runs(runs_dir, base, seeds, SB, logsB):
    S, logs, bad, pipe, params = {}, {}, [], {}, {}
    p0, p1 = DOSE[base]
    for s in seeds:
        d = os.path.join(runs_dir, g_dir(base, s))
        for f in ("summary.json", "argv.json", "selfcheck.json", "ckpt_last.pt"):
            if not os.path.exists(os.path.join(d, f)):
                refuse(f"{os.path.join(d, f)} is missing (the run did not finish, or was not pulled)")
        S[s] = {f: _jload(os.path.join(d, f)) for f in ("summary.json", "argv.json", "selfcheck.json")}
    for s in seeds:
        tag = f"{g_name(base)}_s{s}"
        sm, argv, sc = S[s]["summary.json"], S[s]["argv.json"].get("argv", []), S[s]["selfcheck.json"]
        fl = PGT._flags(argv)
        ref = SB[(base, s)]
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
            bad.append(f"{tag} did not train on {base}_s{s}'s clip list")
        if sm.get("steps") != rsm.get("steps"):
            bad.append(f"{tag} trained for {sm.get('steps')} steps, {base}_s{s} for {rsm.get('steps')}")
        if sc.get("declared_vs_built") != "PASS":
            bad.append(f"{tag}: the launch gate's declared_vs_built is not PASS")
        if sc.get("argv_sha") != sm.get("argv_sha"):
            bad.append(f"{tag}: selfcheck.json (argv_sha {sc.get('argv_sha')}) is not this run's (summary.json {sm.get('argv_sha')})")
        if fl.get("--arm") != "M2b":
            bad.append(f"{tag}: argv --arm {fl.get('--arm')}, expected M2b")
        if fl.get(GRID_FLAG) != GRID_VAL:
            bad.append(f"{tag}: argv {GRID_FLAG} {fl.get(GRID_FLAG)}, expected {GRID_VAL}")
        if rfl.get(GRID_FLAG, B_GRID_VAL) != B_GRID_VAL:
            bad.append(f"{base}_s{s}: argv {GRID_FLAG} {rfl.get(GRID_FLAG)}: B is not a 4x10 run")
        drop = {"--out", GRID_FLAG}
        fg, fb = {k: x for k, x in fl.items() if k not in drop}, {k: x for k, x in rfl.items() if k not in drop}
        if fg != fb:
            diff = sorted(k for k in set(fg) | set(fb) if fg.get(k) != fb.get(k))
            bad.append(f"{tag}: argv differs from {base}_s{s}'s in {diff} besides --grid and --out -- not one lever")
        built = {w: ((rec.get("levers") or {}).get("p_gt_start"), (rec.get("levers") or {}).get("p_gt_end")) for w, rec in (("summary", sm), ("selfcheck", sc))}
        for w, (b0, b1) in built.items():
            if b0 is None or b1 is None:
                bad.append(f"{tag}: {w}.json levers record no built p_gt_start / p_gt_end: the inherited schedule is unproven")
            elif not (PGT._same_float(b0, p0) and PGT._same_float(b1, p1)):
                bad.append(f"{tag} was built with p_gt {b0} -> {b1} ({w}.json), B's is {p0} -> {p1}")
        for w, rec in (("summary", sm), ("selfcheck", sc)):
            lg, lb = rec.get("levers") or {}, rsm.get("levers") or {}
            diff = sorted(k for k in set(lg) & set(lb) if k not in _SKIP_LEVERS and lg[k] != lb[k])
            diff += [f"w.{k}" for k in w_diff(lg.get("w") or {}, lb.get("w") or {}, rfl)]
            if diff:
                bad.append(f"{tag}: {w}.json built levers {diff} differ from {base}_s{s}'s")
        pc = {w: check_params(rsm.get("levers") or {}, rec.get("levers") or {}, int(fl.get("--d", 256)), f"{tag} ({w}.json)") for w, rec in (("summary", sm), ("selfcheck", sc))}
        params[s] = pc["summary"]
        ck = ckpt_facts(os.path.join(runs_dir, g_dir(base, s), "ckpt_last.pt"))
        if ck["grid"] != GRID_VAL:
            bad.append(f"{tag}: the checkpoint carries grid {ck['grid']!r}, not {GRID_VAL!r}")
        if ck["n_tok"] != N_TOK_G:
            bad.append(f"{tag}: the checkpoint was BUILT with n_tok {ck['n_tok']}, not {N_TOK_G}")
        if ck["pos_rows"] != N_TOK_G:
            bad.append(f"{tag}: the checkpoint's position table has {ck['pos_rows']} rows, not {N_TOK_G}")
        if ck["argv_sha"] != sm.get("argv_sha"):
            bad.append(f"{tag}: the checkpoint was written by another argv (argv_sha {ck['argv_sha']} vs summary.json {sm.get('argv_sha')})")
        if ck["step"] != sm.get("steps"):
            bad.append(f"{tag}: the checkpoint is at step {ck['step']}, the run's {sm.get('steps')}")
        if ck["arm"] != "M2b" or ck["seed"] != s:
            bad.append(f"{tag}: the checkpoint says arm {ck['arm']} seed {ck['seed']}")
        logs[s] = PGT._train_log_check(os.path.join(runs_dir, g_dir(base, s), "train_log.jsonl"), g_name(base), s, sm.get("steps"), p0, p1, True)
        pipe[s] = {"G": {"lru": (sc.get("info") or {}).get("lru"), "seconds": sm.get("seconds"), "steps": sm.get("steps"),
                         "steps_per_s": (sm["steps"] / sm["seconds"]) if sm.get("seconds") else None, "ckpt": ck},
                   "B": {"lru": (ref["selfcheck.json"].get("info") or {}).get("lru"), "seconds": rsm.get("seconds"), "steps": rsm.get("steps"),
                         "steps_per_s": (rsm["steps"] / rsm["seconds"]) if rsm.get("seconds") else None}}
    if bad:
        refuse("; ".join(bad))
    return S, logs, pipe, params


def check_eval_ckpts(eval_dir, base, seeds):
    """Every G eval names the 8x20 checkpoint it scored (the eval json's `ckpt`): an eval of B's checkpoint under G's tag is refused."""
    bad = []
    for s in seeds:
        for suf in EVAL_SUFFIXES:
            p = os.path.join(eval_dir, f"eval_{g_name(base)}_s{s}{suf}.json")
            ck = _jload(p).get("ckpt")
            parts = str(ck or "").replace("\\", "/").rstrip("/").split("/")
            if len(parts) < 2 or parts[-2] != g_dir(base, s) or parts[-1] != "ckpt_last.pt":
                bad.append(f"{os.path.basename(p)} scored {ck!r}, not <runs>/{g_dir(base, s)}/ckpt_last.pt")
    if bad:
        refuse("; ".join(bad))


def tie_to_pgt(dec, Zb, base):
    """B's evals are the ones the PGT verdict was computed from: the mean ADE6 the PGT read recorded for this arm at each seed equals the one in the evals we hold."""
    if dec["pgt_stage"] not in ("s0", "final"):
        return {"checked": False}
    r = _jload(dec["source"])
    bad, n = [], 0
    for s in (0, 1):
        rec = (((r.get("rows") or {}).get(f"s{s}") or {}).get("beside") or {}).get("ade6") or {}
        if base in rec:
            n += 1
            got = float(Zb[(base, s)]["ade6"].mean())
            if abs(got - float(rec[base])) > 1e-9:
                bad.append(f"{base}_s{s}: the evals hold ADE6 {got:.6f}, the PGT read recorded {float(rec[base]):.6f}")
    if bad:
        refuse("a B side whose identity does not match the PGT verdict -- " + "; ".join(bad))
    return {"checked": True, "n_seed_arms": n, "source": dec["source"]}


# ------------------------------------------------------------------------------------------------ the M-LAT contrast
def latcons_window_mse(arm, g_cons, has, WC):
    """Per window: the mean squared error of T's PRESENT latcons slots (SD units); NaN where the window has none. v8f_wc_read.present_error's own formula
    (pres = GT-present & GT != 0; e2 = (T - GT)^2), restricted to the latcons columns."""
    cols = list(WC.GROUPS["latcons"])
    pres = has[:, None] & (g_cons != 0.0)
    pl = pres[:, cols]
    e2 = np.where(pl, (arm["t_cons"][:, cols] - g_cons[:, cols]) ** 2, 0.0)
    n = pl.sum(1)
    return np.where(n > 0, e2.sum(1) / np.maximum(n, 1), np.nan), int(pl.sum())


def mlat_row(pf, A, Bm, pe_a, pe_b, ref, WC, eid, n_boot):
    """The M-LAT contrast of arm A (G) against B at one seed. `ref` = the B-side arm that carries the GT block (the same labels on the same windows)."""
    has, g = ref["has"], ref["g_cons"]
    ma, na = latcons_window_mse(A, g, has, WC)
    mb, nb = latcons_window_mse(Bm, g, has, WC)
    ok = np.isfinite(mb)
    if not ok.any() or na != nb or not np.array_equal(np.isfinite(ma), ok):
        refuse("no window carries a present latcons slot, or the two arms' present-slot sets differ: the M-LAT contrast is undefined")
    ra, rb = pe_a["rmse_group"]["latcons"], pe_b["rmse_group"]["latcons"]
    if ra is None or rb is None or not rb > 0:
        refuse("the latcons group has no present slots or a non-positive B RMSE: the M-LAT bar is undefined")
    d = PGT.paired(pf, ma[ok], mb[ok], eid[ok], n_boot)
    win_a, win_b = float(np.sqrt(ma[ok].mean())), float(np.sqrt(mb[ok].mean()))
    fall = 1.0 - ra / rb
    fall_win = 1.0 - win_a / win_b
    return {"rmse_slot": {"G": ra, "B": rb}, "fall": fall, "rmse_window": {"G": win_a, "B": win_b}, "fall_window": fall_win,
            "definitions_agree": bool((fall >= MLAT_BAR) == (fall_win >= MLAT_BAR)), "n_windows_present": int(ok.sum()), "n_slots_present": na, "mse_delta": d}


def judge(row, F):
    """The committed bars. PRIMARY: delta < 0, separated, |delta| > 2F. M-LAT: separated fall of the per-window latcons MSE AND the latcons RMSE fell by >= 12.5 %."""
    p = row["primary"]
    ml = row["mlat"]
    d = ml["mse_delta"]
    return {"passed": bool(PGT._is_neg_sep(p) and abs(p["point"]) > 2 * F),
            "mlat": {"met": bool(PGT._is_neg_sep(d) and ml["fall"] >= MLAT_BAR), "bar": MLAT_BAR, "fall": ml["fall"], "separated": bool(d["separated"])}}


def verdict(stage, rows, seeds):
    mech_all = all(rows[f"s{s}"]["judge"]["mlat"]["met"] for s in seeds)
    if stage == "s0":
        go = rows["s0"]["judge"]["passed"]
        return ("CONTINUE" if go else "STOP"), (None if go else ("FAIL-MECHANISM-OK" if mech_all else "FAIL-NO-MECHANISM")), mech_all
    ok = all(rows[f"s{s}"]["judge"]["passed"] for s in seeds)
    return None, ("SUCCESS" if ok else ("FAIL-MECHANISM-OK" if mech_all else "FAIL-NO-MECHANISM")), mech_all


# ------------------------------------------------------------------------------------------------ the read
def _relabel(Zb, Zg, base, seeds):
    """v8f_wc_read.present_error / the GT-code check key the reference arm as ("M2b", seed): B (whatever it is) goes under that key, G under its own name."""
    Z = {("M2b", s): Zb[(base, s)] for s in (0, 1)}
    Z.update({(g_name(base), s): Zg[(g_name(base), s)] for s in seeds})
    return Z


def _common(a, WC):
    base = a.base
    dec = decide_base(a.pgt_dir)
    if base != dec["base"]:
        refuse(f"--base {base} but {dec['why']} decides B = {dec['base']}: a B side whose identity does not match the PGT verdict")
    SB, logsB = load_b_runs(a.runs_dir, base)
    Zb, eid = PGT.load_all(a.eval_dir, (0, 1), (base,), WC)
    tie = tie_to_pgt(dec, Zb, base)
    return dec, SB, logsB, Zb, eid, tie


def read(a, pairfn, famfn, DK, WC):
    seeds = (0,) if a.stage == "s0" else (0, 1)
    base, G = a.base, g_name(a.base)
    dec, SB, logsB, Zb, eid, tie = _common(a, WC)
    SG, logsG, pipe, params = load_g_runs(a.runs_dir, base, seeds, SB, logsB)
    Zg, eid_g = PGT.load_all(a.eval_dir, seeds, (G,), WC)
    if not np.array_equal(np.asarray(eid_g), np.asarray(eid)):
        refuse(f"the {G} evals are not on the same eval grid as {base}'s (eid differs)")
    check_eval_ckpts(a.eval_dir, base, seeds)
    Zrel = _relabel(Zb, Zg, base, seeds)
    PGT.check_gt_codes(Zrel, seeds, WC)
    F = abs(float(Zb[(base, 0)]["ade6"].mean()) - float(Zb[(base, 1)]["ade6"].mean()))
    ctx = PGT.build_ctx(a, DK, eid)
    Zdk = {**Zb, **Zg}
    pairs = [(f"{G}_s{s}", f"{base}_s{s}") for s in seeds] + [(f"{G}_s{s}", "P10") for s in seeds] + [(f"{base}_s{s}", "P10") for s in (0, 1)]
    res10 = PGT.dk_read(a, DK, ctx, Zdk, ctx["blk10"], 10, a.dk_n_boot, pairs)
    pw = res10["perwindow"]
    if not np.array_equal(np.asarray(pw["is_lead"], bool), ctx["lead"]):
        refuse("the LEAD windows of the distance-keeping read are not the lead block's join")
    base_st = {s: PGT.baseline_stats(pairfn, Zb[(base, s)], pw, ctx, a.n_boot) for s in (0, 1)}
    known = PGT.known_values(base_st, F, (0, 1)) if base == "M2b" else {"checked": False, "why": "B is M2b-pgt: the PREREG's banked premises are M2b's; B's identity is tied to the PGT read instead"}
    idc = identity_control(pairfn, {s: Zb[(base, s)] for s in seeds}, pw, ctx, a.n_boot, WC, seeds, Zrel, None)
    pe, _ = WC.present_error(Zrel, seeds)
    ref = Zrel[("M2b", seeds[0])]
    rows = {}
    for s in seeds:
        A, Bm = Zg[(G, s)], Zb[(base, s)]
        row = PGT.contrast(pairfn, A, Bm, pw, ctx, a.n_boot)
        row["mlat"] = mlat_row(pairfn, A, Bm, pe[(G, s)], pe[("M2b", s)], ref, WC, eid, a.n_boot)
        row["base"] = {k: base_st[s][k] for k in ("lambda_m2b_m", "lambda_ci", "n_lead", "excess_p10", "excess_ci")}
        row["judge"] = judge(row, F)
        ok = np.isfinite(pe[("M2b", s)]["mse"])
        row["beside"] = {
            "ade6": {"B": float(Bm["ade6"].mean()), "G": float(A["ade6"].mean())},
            "oracle_ade6": {"B": float(Bm["orc_ade6"].mean()), "G": float(A["orc_ade6"].mean())},
            "oracle_ade6_delta": PGT.paired(pairfn, A["orc_ade6"], Bm["orc_ade6"], eid, a.n_boot),
            "present_slot": {"n_windows_present": int(ok.sum()), "rmse": {"B": float(np.sqrt(pe[("M2b", s)]["mse"][ok].mean())), "G": float(np.sqrt(pe[(G, s)]["mse"][ok].mean()))},
                             "rmse_group": {"B": pe[("M2b", s)]["rmse_group"], "G": pe[(G, s)]["rmse_group"]},
                             "mse_delta": PGT.paired(pairfn, pe[(G, s)]["mse"][ok], pe[("M2b", s)]["mse"][ok], eid[ok], a.n_boot)},
            "eval_json": {k: {kk: Z[(nm, s)]["json"].get(kk) for kk in ("traj", "families", "strategic", "tactical") if kk in Z[(nm, s)]["json"]}
                          for k, (nm, Z) in {"B": (base, Zb), "G": (G, Zg)}.items()},
            "distance_keeping_k10": {"pair_vs_b": res10["read"]["pairs"].get(f"{G}_s{s}__minus__{base}_s{s}"), "pair_vs_p10": res10["read"]["pairs"].get(f"{G}_s{s}__minus__P10"),
                                     "arms": {nm: res10["read"]["arms"][nm] for nm in (f"{G}_s{s}", f"{base}_s{s}", "P10")}},
            "parameters": params[s], "pipeline": {k: {kk: vv for kk, vv in v.items() if kk != "ckpt"} for k, v in pipe[s].items()},
            "gt_source": {"B": Bm["gt_source"], "G": A["gt_source"]}}
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
    fall = fmt(lambda r: f"{100 * r['mlat']['fall']:+.1f}%{'' if r['mlat']['mse_delta']['separated'] else ' ns'}")
    disagree = any(not rows[f"s{s}"]["mlat"]["definitions_agree"] for s in seeds)
    out = dict(schema="v8f-grid-read-v1", prereg=f"PREREG_V8F_GRID.md stage {a.stage}", base=base, base_decision=dec, b_identity_tied_to_pgt=tie,
               tier="T1-class open loop (eval-141, 2,399 windows); the mechanism read uses T0 absmask / GT-oracle evals",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); F = |{base}_s0 - {base}_s1| = {F:.4f} m (B's own two seeds); "
                         f"answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)",
               n_windows=int(eid.size), n_episodes=int(np.unique(eid).size), n_lead_windows_k10=int(ctx["lead"].sum()), F=F, two_F=2 * F,
               known_values=known, identity_control=idc, train_log={f"{k}": v for k, v in logsG.items()},
               strategic_family="NOT PAUSED: un-paused for v8F-mini (PI ruling 2026-10-17); read from each run's eval json under rows.*.beside.eval_json",
               distance_keeping_k30=k30, stage=a.stage, rows=rows, DECISION=decision, VERDICT=vd, mechanism=mech_all, next=nxt)
    out["reading"] = (f"PREREG_V8F_GRID stage {a.stage}: {decision or vd}; base {base}; ADE6 G - B {prim} m (2F {2 * F:.4f}); M-LAT latcons RMSE fall {fall} "
                      f"(bar {100 * MLAT_BAR:.1f}%; B {' / '.join(f'{rows[f's{s}']['mlat']['rmse_slot']['B']:.3f}' for s in seeds)} SD)"
                      f"{' [the slot-pooled and window-weighted definitions DISAGREE about the bar]' if disagree else ''}; mechanism {'MET' if mech_all else 'NOT met'}." +
                      ("" if vd is None else f" VERDICT {vd}."))
    return out


def identity_control(pf, Zb_by_seed, pw, ctx, n_boot, WC, seeds, Zrel, fam):
    """PGT's identity control (PRIMARY, M-LEAK, M-DK, present-slot, oracle ADE6, every family cell: B against itself reads exactly +0 [0, 0]) plus M-LAT's own cells."""
    out = PGT.identity_control(pf, Zb_by_seed, pw, ctx, n_boot, WC, seeds, fam=fam)
    ref = Zrel[("M2b", seeds[0])]
    for s in seeds:
        A = Zb_by_seed[s]
        pe, _ = WC.present_error({("M2b", s): A}, (s,))
        row = mlat_row(pf, A, A, pe[("M2b", s)], pe[("M2b", s)], ref, WC, ctx["eid"], n_boot)
        cells = {"mlat_mse": row["mse_delta"]}
        for name, v in cells.items():
            if not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 and v["point"] == 0.0):
                refuse(f"IDENTITY CONTROL FAILED: B_s{s} - B_s{s} reads {v['delta']} [{v['lo']}, {v['hi']}] on {name}, not +0 [0, 0]: the paired instrument is broken")
        if row["fall"] != 0.0 or row["fall_window"] != 0.0:
            refuse(f"IDENTITY CONTROL FAILED: B_s{s} against itself reads a latcons RMSE fall of {row['fall']} / {row['fall_window']}, not exactly 0")
        out[f"s{s}"]["mlat_mse"] = {x: cells["mlat_mse"][x] for x in ("delta", "lo", "hi", "point")}
        out[f"s{s}"]["mlat_fall"] = row["fall"]
        out[f"s{s}"]["latcons_rmse_slot"] = row["rmse_slot"]["B"]
        out[f"s{s}"]["latcons_rmse_window"] = row["rmse_window"]["B"]
    return out


def check_inputs(a, DK, WC, pairfn, famfn=None):
    """The B side alone, before any training: its runs and evals load, the PGT identity of B holds, the PREREG's premises reproduce (B = M2b), the paired instrument
    reads exactly 0 on B against itself, M-LAT included."""
    base = a.base
    dec, SB, logsB, Zb, eid, tie = _common(a, WC)
    Zrel = {("M2b", s): Zb[(base, s)] for s in (0, 1)}
    PGT.check_gt_codes(Zrel, (0, 1), WC)
    F = abs(float(Zb[(base, 0)]["ade6"].mean()) - float(Zb[(base, 1)]["ade6"].mean()))
    ctx = PGT.build_ctx(a, DK, eid)
    res10 = PGT.dk_read(a, DK, ctx, Zb, ctx["blk10"], 10, a.dk_n_boot, [])
    pw = res10["perwindow"]
    base_st = {s: PGT.baseline_stats(pairfn, Zb[(base, s)], pw, ctx, a.n_boot) for s in (0, 1)}
    known = PGT.known_values(base_st, F, (0, 1)) if base == "M2b" else {"checked": False, "why": "B is M2b-pgt"}
    idc = identity_control(pairfn, {s: Zb[(base, s)] for s in (0, 1)}, pw, ctx, a.n_boot, WC, (0, 1), Zrel, famfn)
    return {"base": base, "base_decision": dec, "b_identity_tied_to_pgt": tie, "F": F, "lambda_b": {s: base_st[s]["lambda_m2b_m"] for s in (0, 1)},
            "excess_p10": {s: base_st[s]["excess_p10"] for s in (0, 1)}, "n_lead_k10": int(ctx["lead"].sum()),
            "latcons_rmse_slot_b": {s: idc[f"s{s}"]["latcons_rmse_slot"] for s in (0, 1)},
            "latcons_rmse_window_b": {s: idc[f"s{s}"]["latcons_rmse_window"] for s in (0, 1)},
            "gt_source": {f"{k[0]}_s{k[1]}": v["gt_source"] for k, v in Zb.items()}, "train_log": {f"{k[0]}_s{k[1]}": v for k, v in logsB.items()},
            "known_values": known, "identity_control": idc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("s0", "final"))
    ap.add_argument("--base", choices=BASES)
    ap.add_argument("--decide-base", action="store_true", help="print the base B decided by PREREG_V8F_PGT's reads and exit")
    ap.add_argument("--check-inputs", action="store_true", help="validate the B side alone (and the identity control) and exit")
    ap.add_argument("--pgt-dir", help="the PGT chain's output dir: pgt_read.json (final) and pgt_read_s0.json")
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
        if a.decide_base:
            if not a.pgt_dir:
                ap.error("--pgt-dir is required with --decide-base")
            dec = decide_base(a.pgt_dir)
            if a.out:
                with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(dec, f, indent=1)
            print(f"BASE={dec['base']} ({dec['why']})")
            return 0
        for need in ("base", "pgt_dir", "eval_dir", "runs_dir", "out"):
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
            print(f"GRID INPUTS OK: base {a.base} ({res['base_decision']['why']}); F {res['F']:.4f}; LAMBDA(B) {res['lambda_b'][0]:+.4f} / {res['lambda_b'][1]:+.4f}; "
                  f"latcons RMSE(B) {res['latcons_rmse_slot_b'][0]:.4f} / {res['latcons_rmse_slot_b'][1]:.4f} SD; identity control exactly +0 [0, 0]")
            return 0
        if a.stage is None:
            ap.error("--stage is required unless --check-inputs / --decide-base")
        res = read(a, pairfn, None if a.no_families else PGT._famfn(a), DK, WC)
    except (OSError, KeyError, ValueError) as e:
        print(f"GRID READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(DK._clean(res), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
