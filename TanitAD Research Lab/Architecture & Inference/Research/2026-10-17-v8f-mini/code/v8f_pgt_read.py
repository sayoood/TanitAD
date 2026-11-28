"""PREREG_V8F_PGT, committed before its data: does training the stack against its own upstream outputs (the scheduled-sampling floor,
`--p-gt-start 0.5 --p-gt-end 0.0`, ONE lever set by two flags) make v8F-mini drive better, and does it halve the two measured symptoms?

  python v8f_pgt_read.py --stage s0|final --eval-dir <eval> --runs-dir <runs> --out pgt_read.json
  python v8f_pgt_read.py --check-inputs --eval-dir <eval> --runs-dir <runs> --out x.json      (the banked M2b side alone, before any training)

Arms: M2b (banked) and M2b-pgt = `--arm M2b --p-gt-start 0.5 --p-gt-end 0.0` (queue stage `trainpgt`), the same 4,512 clips, the RA-1 holdout
withheld, M2b's 19,164 steps. Eval: `v8f_mini_eval.py run` on eval-141 (2,399 windows, 141 episodes), stamped "T1-class open loop"; the two
mechanism reads use T0 oracle evals (O reads the GT code), only to score a symptom.

  PRIMARY    per seed s: delta_s = ADE6(M2b-pgt_s) - ADE6(M2b_s), paired over episodes. Seed s PASSES iff delta_s < 0, separated and
             |delta_s| > 2F, F = |mean ADE6 M2b_s0 - M2b_s1| (M2b's seed floor, 0.1207 m, from BOTH banked M2b evals).
  M-LEAK     the latcons leak halves. The leak of an arm is  LAMBDA = along-track(2 s) of `eval_<c>-tacorc-all-latcons` minus that of
             `eval_<c>-tacorc`, per LEAD window of the banked K = 10 lead block (raw/2026-10-22-dk/latcons_mech.py's definition: pred[:, 9, 0]).
             PASS iff LAMBDA(M2b-pgt_s) - LAMBDA(M2b_s) < 0, separated, and |.| >= 0.5 LAMBDA(M2b_s). A difference of paired differences,
             bootstrapped over episodes on the same windows. M2b reads +0.2678 / +0.1667 m (banked).
  M-DK       the 2 s low-TTC tail halves. S5 = share of LEAD windows with min-TTC < 5 s at 2 s (the K = 10 block, v8f_dk_read.read_dk's per-window
             min_ttc_s). PASS iff S5(M2b-pgt_s) - S5(M2b_s) < 0, separated, and |.| >= 0.5 x M2b_s's excess over P10 (+4.42 / +2.50 pp banked).
             The pair is over the LEAD windows in which the lead is admissible for BOTH arms (>= 2 of the K = 10 steps inside the arm's own corridor,
             so min-TTC is finite; a window where the arm does not close gets the 30 s cap and stays in): the convention that reproduces the banked
             premises. Beside it, deciding nothing: `mdk.delta.n_dropped_no_lead_in_path` (windows that left the pair because an arm's path never
             has the lead in its corridor) and `mdk.delta.alt_all_lead` (the same contrast over EVERY LEAD window, such a window counted as not at
             risk). MEASURED on the banked evals: min-TTC is undefined on 38 / 21 of 743 LEAD windows for M2b (seeds 0 / 1) and 20 for P10.
  MECHANISM  met at a seed iff BOTH M-LEAK and M-DK pass there. It decides the reading of a FAIL, never the verdict.
  STAGES     s0: seed 0 only. CONTINUE iff seed 0 PASSES the PRIMARY (only then is seed 1 trained); otherwise STOP with the seed-0 verdict.
             final: both seeds.
  VERDICTS   SUCCESS (both seeds pass) | FAIL-MECHANISM-OK (the mechanism is met at every seed read; driving did not move beyond 2F) |
             FAIL-NO-MECHANISM (the leak or the tail did not halve).
Reported beside it, deciding nothing: the four families (LON incl. distance keeping from v8f_dk_read.read_dk -- headway, time-gap, min-TTC,
LEAD-VIOL with its progress control --, LAT, TACTICAL and STRATEGIC, un-paused for v8F-mini, PI ruling 2026-10-17); T's present-slot RMSE by
group; the GT-code oracle ADE6 of each arm.
Refusals (rc 2): a missing file; runs not on one clip list / not withholding the RA-1 holdout / not done / different step counts; an M2b-pgt run
whose argv is not `--arm M2b ... --p-gt-start 0.5 --p-gt-end 0.0`, whose built p_gt_start / p_gt_end differ (summary.json AND selfcheck.json),
or whose selfcheck is not PASS; an M2b run with either flag in its argv; evals not on one grid; GT codes that differ between the arms.
ADDED GUARDS (not in the PREREG's list, same direction; they refuse, they never decide): every other argv flag of an M2b-pgt run equals M2b's
(one lever); the loop's own train_log.jsonl p_gt equals the declared schedule at every logged step (the measured proof that the flags reached
the masks' input); an oracle eval is an oracle run (json mode); the banked M2b side reproduces the PREREG's known values (F, LAMBDA, the S5
excess) and the paired instrument reads EXACTLY +0 [0, 0] for an arm against itself (the identity control).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (EPISODES); it answers "another draw of EPISODES", and training-seed
variance enters only through F (H-ESTIM-SEED-1). v8F-mini's planner is deterministic, so no inference-seed floor applies."""
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
HOLDOUT_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
N_TRAIN = 4512
ARM_B, ARM_P = "M2b", "M2b-pgt"
ARMS = (ARM_B, ARM_P)
PGT_START, PGT_END = 0.5, 0.0                  # the ONE lever, set by two flags
M2B_START, M2B_END = 1.0, 0.2                  # M2b = the trainer's defaults
LEVER_FLAGS = ("--p-gt-start", "--p-gt-end")
IDX_2S = 9                                      # pred[:, 9] = t + 2.0 s (0.2 s steps); K = 10 block
TTC_THR_S = 5.0
B = 2000
HALF = 0.5
# the PREREG's banked premises (MEASURED: raw/2026-10-22-dk/latcons_mech.json, ttc_tail_ci.json; RESULT_V8F_WC.md for F). Reproduced on the real
# banked M2b side by --check-inputs and by every read: if they do not reproduce, the inputs are not the ones the PREREG was written against.
KNOWN = {"lambda": {0: 0.2678, 1: 0.1667}, "excess": {0: 0.0442, 1: 0.0250}, "F": 0.1207}
KNOWN_TOL = {"lambda": 6e-4, "excess": 6e-5, "F": 6e-5}
NEXT = {None: "CONTINUE: seed 0 passes the PRIMARY; train seed 1 (the verdict needs both).",
        "SUCCESS": "SUCCESS: M2b-pgt drives better than M2b beyond 2F at both seeds. Next: the T1-kin read of M2b-pgt "
                   "(stack/scripts/v8f_t1_eval.py, m in {30, 10}), then the pool8x20 input arm on top, under its own PREREG and a PI go.",
        "FAIL-MECHANISM-OK": "FAIL with the mechanism in place: the leak and the tail halved, but driving did not move beyond 2F. O's robustness to T is "
                             "not the ADE bottleneck: T's accuracy is. Next lever: the pool8x20 input arm (T's lateral geometry from a 4x finer "
                             "token grid), PI go.",
        "FAIL-NO-MECHANISM": "FAIL without the mechanism: the leak or the tail did not halve, so the coupling is structural, not exposure. Next lever: "
                             "separate O's longitudinal head from the lateral code slots (an architecture change, so a PI go)."}


def refuse(msg):
    print(f"PGT READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _modules():
    """The programme's own instruments, IMPORTED (never copied): the distance-keeping reader, the WC reader's present-slot machinery and the
    family block. Their refusals are routed through THIS reader's refuse (one message prefix, one rc)."""
    DK = _load("v8f_dk_read_for_pgt", HERE.parent / "v8f_dk_read.py")
    WC = _load("v8f_wc_read_for_pgt", HERE.parent / "v8f_wc_read.py")
    WC.refuse = refuse
    DK._paths()
    return DK, WC


# ------------------------------------------------------------------------------------------------ the schedule (independent of the model's)
def schedule(step, total, p0, p1):
    """The linear scheduled-sampling schedule, re-derived here on purpose: a mutated model.p_gt_schedule must not cancel against the reader."""
    return float(p0 + (p1 - p0) * min(max(step / max(total, 1), 0.0), 1.0))


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


def _same_float(got, want):
    try:
        return float(got) == float(want)
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------------------------ runs
def _train_log_check(path, arm, s, steps, p0, p1, required):
    if not os.path.exists(path):
        if required:
            refuse(f"{arm}_s{s}/train_log.jsonl is missing: the loop's own p_gt is the measured proof that the flags reached the masks")
        return {"status": "NOT-CHECKED", "reason": "no train_log.jsonl staged for this run"}
    recs = [json.loads(x) for x in open(path, encoding="utf-8").read().splitlines() if x.strip()]
    if not recs:
        refuse(f"{arm}_s{s}/train_log.jsonl is empty")
    bad = [r["step"] for r in recs if abs(float(r["p_gt"]) - schedule(int(r["step"]) - 1, steps, p0, p1)) > 1e-9]
    if bad:
        r = next(x for x in recs if x["step"] == bad[0])
        refuse(f"{arm}_s{s}: the loop's logged p_gt differs from the declared schedule {p0} -> {p1} at {len(bad)} logged steps (first: step "
               f"{bad[0]} logged {r['p_gt']:.6f}, schedule {schedule(bad[0] - 1, steps, p0, p1):.6f}): the flags did not reach the masks")
    return {"status": "OK", "n_records": len(recs), "first_p_gt": recs[0]["p_gt"], "last_p_gt": recs[-1]["p_gt"], "schedule": [p0, p1], "steps": steps}


def load_runs(runs_dir, seeds, arms=ARMS):
    S, bad = {}, []
    for arm in arms:
        for s in seeds:
            d = os.path.join(runs_dir, f"{arm}_s{s}")
            for f in ("summary.json", "argv.json", "selfcheck.json"):
                if not os.path.exists(os.path.join(d, f)):
                    refuse(f"{os.path.join(d, f)} is missing (the run did not finish, or was not pulled)")
            S[(arm, s)] = {f: json.load(open(os.path.join(d, f), encoding="utf-8")) for f in ("summary.json", "argv.json", "selfcheck.json")}
    sm = {k: v["summary.json"] for k, v in S.items()}
    if len({d["train_sha256"] for d in sm.values()}) != 1:
        bad.append("the runs did not train on one clip list")
    if {d.get("excluded_sha256") for d in sm.values()} != {HOLDOUT_SHA}:
        bad.append("a run did not withhold the RA-1 holdout")
    if {d["n_clips"] for d in sm.values()} != {N_TRAIN}:
        bad.append(f"n_clips {sorted({d['n_clips'] for d in sm.values()})} != {N_TRAIN}")
    if not all(d.get("done") is True for d in sm.values()):
        bad.append("a run is not done")
    if len({d["steps"] for d in sm.values()}) != 1:
        bad.append(f"the runs trained for different step counts: {sorted({d['steps'] for d in sm.values()})}")
    logs = {}
    for (arm, s), v in S.items():
        d, argv, sc = v["summary.json"], v["argv.json"].get("argv", []), v["selfcheck.json"]
        fl = _flags(argv)
        if d["arm"] != "M2b" or d["seed"] != s:
            bad.append(f"{arm}_s{s}/summary.json says {d['arm']} seed {d['seed']}, expected M2b seed {s}")
        if sc.get("declared_vs_built") != "PASS":
            bad.append(f"{arm}_s{s}: the launch gate's declared_vs_built is not PASS")
        if fl.get("--arm") != "M2b":
            bad.append(f"{arm}_s{s}: argv --arm {fl.get('--arm')}, expected M2b")
        built = {w: ((rec.get("levers") or {}).get("p_gt_start"), (rec.get("levers") or {}).get("p_gt_end")) for w, rec in (("summary", d), ("selfcheck", sc))}
        if arm == ARM_P:
            for flag, want in zip(LEVER_FLAGS, (PGT_START, PGT_END)):
                if not _same_float(fl.get(flag), want):
                    bad.append(f"{arm}_s{s}: argv {flag} {fl.get(flag)}, expected {want}")
            for w, (b0, b1) in built.items():
                if b0 is None or b1 is None:
                    bad.append(f"{arm}_s{s}: {w}.json levers record no built p_gt_start / p_gt_end (the run predates the record): the built schedule is unproven")
                elif not (_same_float(b0, PGT_START) and _same_float(b1, PGT_END)):
                    bad.append(f"{arm}_s{s} was built with p_gt {b0} -> {b1} ({w}.json), expected {PGT_START} -> {PGT_END}")
            ref = S.get((ARM_B, s))
            if ref is not None:                      # ONE lever: every other flag equals M2b's (the output directory is the only flag that may differ)
                drop = set(LEVER_FLAGS) | {"--out"}
                fm = {k: x for k, x in _flags(ref["argv.json"].get("argv", [])).items() if k != "--out"}
                fp = {k: x for k, x in fl.items() if k not in drop}
                if fp != fm:
                    diff = sorted(k for k in set(fp) | set(fm) if fp.get(k) != fm.get(k))
                    bad.append(f"{arm}_s{s}: argv differs from M2b_s{s}'s in {diff} besides the two lever flags -- not one lever")
            logs[(arm, s)] = _train_log_check(os.path.join(runs_dir, f"{arm}_s{s}", "train_log.jsonl"), arm, s, d["steps"], PGT_START, PGT_END, True)
        else:
            for flag in LEVER_FLAGS:
                if flag in fl:
                    bad.append(f"{arm}_s{s}: argv carries {flag} {fl[flag]}; M2b is the default {M2B_START} -> {M2B_END}")
            for w, (b0, b1) in built.items():     # runs built before the key existed are 1.0 -> 0.2 by construction
                if (b0 is not None and not _same_float(b0, M2B_START)) or (b1 is not None and not _same_float(b1, M2B_END)):
                    bad.append(f"{arm}_s{s} was built with p_gt {b0} -> {b1} ({w}.json), expected {M2B_START} -> {M2B_END}")
            logs[(arm, s)] = _train_log_check(os.path.join(runs_dir, f"{arm}_s{s}", "train_log.jsonl"), arm, s, d["steps"], M2B_START, M2B_END, False)
    if bad:
        refuse("; ".join(bad))
    return S, logs


# ------------------------------------------------------------------------------------------------ evals
def _mode(path):
    return ((json.load(open(path, encoding="utf-8")).get("tac_manipulation") or {}).get("mode"))


def load_arm(eval_dir, arm, s):
    """One (arm, seed): the standard eval, the absmask eval (T's own present slots), the GT-code oracle and the all-latcons oracle."""
    t = f"{arm}_s{s}"
    p = {k: os.path.join(eval_dir, f"eval_{t}{suf}") for k, suf in
         (("z", ".npz"), ("j", ".json"), ("abs", "-tacabs.npz"), ("absj", "-tacabs.json"), ("orc", "-tacorc.npz"), ("lat", "-tacorc-all-latcons.npz"))}
    for q in p.values():
        if not os.path.exists(q):
            refuse(f"{os.path.basename(q)} is missing from {eval_dir}")
    z, za, zo, zl = np.load(p["z"]), np.load(p["abs"]), np.load(p["orc"]), np.load(p["lat"])
    gsrc, zg = "-tacorc", zo
    if "tac_code" not in zo.files:                    # the 2026-10-20 GT oracle predates tac_code (WC Amendment 1): all-goal's constraint block IS the GT
        pg = os.path.join(eval_dir, f"eval_{t}-tacorc-all-goal.npz")
        if not os.path.exists(pg):
            refuse(f"eval_{t}-tacorc.npz has no tac_code (it predates it) and eval_{t}-tacorc-all-goal.npz is missing from {eval_dir}")
        gsrc, zg = "-tacorc-all-goal", np.load(pg)
        if "tac_code" not in zg.files:
            refuse(f"eval_{t}-tacorc-all-goal.npz has no tac_code either")
    if "tac_code" not in za.files:
        refuse(f"eval_{t}-tacabs.npz has no tac_code")
    eid = np.asarray(z["eid"])
    for name, zz in (("-tacabs", za), ("-tacorc", zo), ("-tacorc-all-latcons", zl), (gsrc, zg)):
        if not np.array_equal(np.asarray(zz["eid"]), eid):
            refuse(f"eval_{t}{name}.npz is not on the same eval grid as eval_{t}.npz (eid differs)")
    if _mode(p["absj"]) != "absmask":
        refuse(f"eval_{t}-tacabs.json is not an absmask run")
    for suf in ("-tacorc", "-tacorc-all-latcons"):                # an oracle is an oracle run (json present on every staged run; absent => unchecked)
        pj = os.path.join(eval_dir, f"eval_{t}{suf}.json")
        if os.path.exists(pj) and _mode(pj) != "oracle":
            refuse(f"eval_{t}{suf}.json is not an oracle run")
    for k in ("pred", "ade6"):
        for name, zz in (("", z), ("-tacorc", zo), ("-tacorc-all-latcons", zl)):
            if k not in zz.files:
                refuse(f"eval_{t}{name}.npz carries no {k}")
    C0, NC = _WC_LAYOUT
    tc = np.asarray(za["tac_code"], np.float64)[:, C0:C0 + NC]
    return {"arm": arm, "seed": s, "eid": eid, "ade6": np.asarray(z["ade6"], np.float64), "pred": np.asarray(z["pred"], np.float64),
            "t_cons": tc, "g_cons": np.asarray(zg["tac_code"], np.float64)[:, C0:C0 + NC], "has": ~np.asarray(zg["tac_oracle_fallback"], bool),
            "gt_source": gsrc, "orc_pred": np.asarray(zo["pred"], np.float64), "orc_ade6": np.asarray(zo["ade6"], np.float64),
            "lat_pred": np.asarray(zl["pred"], np.float64), "lat_code": np.asarray(zl["tac_code"], np.float64) if "tac_code" in zl.files else None,
            "json": json.load(open(p["j"], encoding="utf-8")).get("raw", {}), "tag": t, "dk": t}


_WC_LAYOUT = (22 + 5 + 7, 26)                       # N_GOAL + N_LAT + N_LON, N_CONS (checked against the WC reader's own constants in load_all)


def load_all(eval_dir, seeds, arms, WC):
    if _WC_LAYOUT != (WC.C0, WC.N_CONS):
        refuse(f"the constraint-block layout {_WC_LAYOUT} != the WC reader's {(WC.C0, WC.N_CONS)}")
    Z, eid = {}, None
    for arm in arms:
        for s in seeds:
            Z[(arm, s)] = load_arm(eval_dir, arm, s)
            if eid is None:
                eid = Z[(arm, s)]["eid"]
            elif not np.array_equal(Z[(arm, s)]["eid"], eid):
                refuse(f"eval_{arm}_s{s}.npz is not on the same eval grid as the others (eid differs)")
    return Z, eid


def check_gt_codes(Z, seeds, WC):
    """The GT constraint block of every arm and seed is identical (WC's check, called), AND every all-latcons oracle carries that same GT outside
    latcons: the eval that scores the leak read the labels it says it read."""
    WC.present_error({k: v for k, v in Z.items()}, seeds)
    lat_cols = [WC.C0 + i for i in WC.GROUPS["latcons"]]
    for (arm, s), v in Z.items():
        if v["lat_code"] is None:
            continue
        keep = [c for c in range(WC.C0, WC.C0 + WC.N_CONS) if c not in lat_cols]
        if not np.array_equal(v["lat_code"][v["has"]][:, keep].astype(np.float32), v["g_cons"][v["has"]][:, [c - WC.C0 for c in keep]].astype(np.float32)):
            refuse(f"{arm}_s{s}: eval_{arm}_s{s}-tacorc-all-latcons.npz does not carry the GT constraint slots outside latcons: it is not the all-latcons oracle")


# ------------------------------------------------------------------------------------------------ the world: ground truth, the K = 10 lead block, P10
def build_ctx(a, DK, eid, need_p10=True):
    G = DK.ground_truth(a)
    if not np.array_equal(np.asarray(G["eid"]), np.asarray(eid)):
        refuse("the rebuilt eval grid (eid) is not the evals' grid")
    blk10 = DK.load_block(a.lead_block or str(DK.LEAD_BLOCK_DEFAULT))
    if blk10["_k"] < 10:
        refuse(f"the lead block's K = {blk10['_k']} < 10: no 2 s horizon")
    J = DK.join_block(blk10, G["clip"], G["frame"], G["v0"], k=10, dt=G["dt"])
    p10 = None
    if need_p10:
        if not os.path.exists(a.p10_pred):
            refuse(f"the P10 predictions {a.p10_pred} do not exist")
        p10 = np.load(a.p10_pred).astype(np.float64)
    fl = os.path.join(a.eval_dir, "floors_perwindow.npz")
    floor_ade = np.load(fl)["P10_refit__ade6"] if os.path.exists(fl) else None
    return {"G": G, "blk10": blk10, "J": J, "lead": np.asarray(J["is_lead"], bool), "eid": np.asarray(G["eid"]), "p10": p10, "p10_floor": floor_ade}


def dk_read(a, DK, ctx, Z, block, k, n_boot, pairs):
    """v8f_dk_read.read_dk on the arms of Z + P10 at one horizon: the per-window min-TTC for M-DK, and the distance-keeping family beside it."""
    preds = {v["dk"]: v["pred"] for v in Z.values()}
    saved = {v["dk"]: v["ade6"] for v in Z.values()}
    preds["P10"] = ctx["p10"]
    try:
        return DK.read_dk(preds=preds, gt=ctx["G"]["gt"], eid=ctx["eid"], clip=ctx["G"]["clip"], frame=ctx["G"]["frame"], v0=ctx["G"]["v0"], block=block, k=k,
                          dt=ctx["G"]["dt"], n_boot=n_boot, seed=0, saved_ade6=saved, p10_floor_ade6=ctx["p10_floor"], pairs=pairs)
    except DK.Refusal as e:
        refuse(f"v8f_dk_read: {e}")


# ------------------------------------------------------------------------------------------------ the contrasts
def paired(pf, a, b, eid, n_boot):
    """a - b, paired over EPISODES. `point` is the UNROUNDED difference (the library rounds `delta` to 4 dp; a size bar must not)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    r = dict(pf(a, b, eid, n_boot, 0))
    r["point"] = float(a.mean() - b.mean())
    return r


def lam_of(v):
    """LAMBDA per window: along-track at 2 s of the all-latcons oracle minus that of the whole-GT-code oracle (latcons_mech.py's definition)."""
    return v["lat_pred"][:, IDX_2S, 0] - v["orc_pred"][:, IDX_2S, 0]


def s5_pair(pf, pw, lead, eid, na, nb, n_boot):
    ta, tb = pw[f"{na}__min_ttc_s"], pw[f"{nb}__min_ttc_s"]
    m = lead & np.isfinite(ta) & np.isfinite(tb)
    r = paired(pf, (ta[m] < TTC_THR_S).astype(np.float64), (tb[m] < TTC_THR_S).astype(np.float64), eid[m], n_boot)
    r.update(n_used=int(m.sum()), share_a=float((ta[m] < TTC_THR_S).mean()), share_b=float((tb[m] < TTC_THR_S).mean()))
    # DESCRIPTIVE, decides nothing. The deciding S5 pairs the LEAD windows in which the lead is admissible for BOTH arms (min-TTC finite: >= 2 steps inside the
    # arm's own corridor; the convention that reproduces the PREREG's banked premises). A window in which an arm's path never has the lead in its corridor
    # (it swerved clear, or went through it) leaves the pair, scored neither at risk nor safe. The same contrast over EVERY LEAD window, such a window counted
    # as not at risk (NaN < 5 is False), and how many windows the deciding one dropped:
    n_lead = int(lead.sum())
    r.update(n_lead=n_lead, n_dropped_no_lead_in_path=n_lead - int(m.sum()),
             alt_all_lead=paired(pf, (ta[lead] < TTC_THR_S).astype(np.float64), (tb[lead] < TTC_THR_S).astype(np.float64), eid[lead], n_boot))
    return r


def baseline_stats(pf, Zm2b, pw, ctx, n_boot):
    """M2b's own symptoms at one seed: LAMBDA(M2b) on the LEAD windows, and M2b's S5 excess over P10 (the bars' reference sizes)."""
    lead, eid = ctx["lead"], ctx["eid"]
    lam = lam_of(Zm2b)
    lam_ci = paired(pf, Zm2b["lat_pred"][lead][:, IDX_2S, 0], Zm2b["orc_pred"][lead][:, IDX_2S, 0], eid[lead], n_boot)
    ex = s5_pair(pf, pw, lead, eid, Zm2b["dk"], "P10", n_boot)
    return {"lambda_m2b_m": float(lam[lead].mean()), "lambda_ci": lam_ci, "n_lead": int(lead.sum()), "excess_p10": ex["point"], "excess_ci": ex}


def contrast(pf, A, Bm, pw, ctx, n_boot):
    """The treatment arm A minus the reference arm Bm at one seed: PRIMARY, M-LEAK, M-DK (no verdicts here: `judge` reads them)."""
    lead, eid = ctx["lead"], ctx["eid"]
    lam_a, lam_b = lam_of(A), lam_of(Bm)
    return {"primary": paired(pf, A["ade6"], Bm["ade6"], eid, n_boot),
            "mleak": {"lambda_a_m": float(lam_a[lead].mean()), "lambda_b_m": float(lam_b[lead].mean()),
                      "delta": paired(pf, lam_a[lead], lam_b[lead], eid[lead], n_boot)},
            "mdk": {"delta": s5_pair(pf, pw, lead, eid, A["dk"], Bm["dk"], n_boot)}}


def _is_neg_sep(r):
    return bool(r["point"] < 0 and r["separated"])


def judge(row, base, F):
    """The committed bars. PRIMARY: delta < 0, separated, |delta| > 2F. M-LEAK / M-DK: delta < 0, separated, |delta| >= half the M2b reference size.
    A reference size that is not positive leaves nothing to halve: the check is NOT-APPLICABLE and counts as NOT met (said, never silent)."""
    p = row["primary"]
    passed = bool(_is_neg_sep(p) and abs(p["point"]) > 2 * F)
    out = {"passed": passed}
    for key, ref, name in (("mleak", base["lambda_m2b_m"], "lambda_m2b_m"), ("mdk", base["excess_p10"], "excess_p10")):
        d = row[key]["delta"]
        if not ref > 0:
            out[key] = {"met": False, "status": "NOT-APPLICABLE", "reason": f"M2b's {name} = {ref:.4f} is not positive: there is no symptom to halve"}
        else:
            out[key] = {"met": bool(_is_neg_sep(d) and abs(d["point"]) >= HALF * ref), "half_of_m2b": HALF * ref, "ref": ref, "size": abs(d["point"])}
    out["mechanism"] = bool(out["mleak"]["met"] and out["mdk"]["met"])
    return out


def verdict(stage, rows, seeds):
    mech_all = all(rows[f"s{s}"]["judge"]["mechanism"] for s in seeds)
    if stage == "s0":
        go = rows["s0"]["judge"]["passed"]
        return ("CONTINUE" if go else "STOP"), (None if go else ("FAIL-MECHANISM-OK" if mech_all else "FAIL-NO-MECHANISM")), mech_all
    ok = all(rows[f"s{s}"]["judge"]["passed"] for s in seeds)
    return None, ("SUCCESS" if ok else ("FAIL-MECHANISM-OK" if mech_all else "FAIL-NO-MECHANISM")), mech_all


def known_values(base, F, seeds):
    """The banked premises must reproduce on the banked inputs (a known-value control, like the DK reader's kinematic contract)."""
    bad = []
    for s in seeds:
        for key, got in (("lambda", base[s]["lambda_m2b_m"]), ("excess", base[s]["excess_p10"])):
            if abs(got - KNOWN[key][s]) > KNOWN_TOL[key]:
                bad.append(f"M2b_s{s} {key} reads {got:+.4f}, the PREREG's banked value is {KNOWN[key][s]:+.4f}")
    if abs(F - KNOWN["F"]) > KNOWN_TOL["F"]:
        bad.append(f"F reads {F:.4f}, the banked value is {KNOWN['F']:.4f}")
    if bad:
        refuse("the banked M2b side does not reproduce the PREREG's known values (wrong eval, wrong lead block, or a changed grid): " + "; ".join(bad))
    return {"checked": True, "tolerances": KNOWN_TOL, "banked": KNOWN}


def identity_control(pf, Zm2b, pw, ctx, n_boot, WC, seeds, fam=None):
    """The paired instrument on an arm against ITSELF must read EXACTLY +0 [0, 0] on every contrast; anything else is a broken instrument, not a result."""
    out = {}
    for s in seeds:
        A = Zm2b[s]
        r = contrast(pf, A, A, pw, ctx, n_boot)
        flat = {"primary": r["primary"], "mleak": r["mleak"]["delta"], "mdk": r["mdk"]["delta"], "mdk_alt": r["mdk"]["delta"]["alt_all_lead"]}
        pe = WC.present_error({(ARM_B, s): {**A}}, (s,))[0][(ARM_B, s)]
        ok = np.isfinite(pe["mse"])
        flat["present_mse"] = paired(pf, pe["mse"][ok], pe["mse"][ok], ctx["eid"][ok], n_boot)
        flat["oracle_ade6"] = paired(pf, A["orc_ade6"], A["orc_ade6"], ctx["eid"], n_boot)
        if fam is not None:
            for cell, v in _iter_fam_cells(fam(A["pred"], A["pred"], ctx["eid"])):
                flat[f"family:{cell}"] = v
        for name, v in flat.items():
            if not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 and v["point"] == 0.0):
                refuse(f"IDENTITY CONTROL FAILED: M2b_s{s} - M2b_s{s} reads {v['delta']} [{v['lo']}, {v['hi']}] (point {v['point']}) on {name}, not +0 [0, 0]: "
                       f"the paired instrument is broken")
        out[f"s{s}"] = {k: {x: v[x] for x in ("delta", "lo", "hi", "point")} for k, v in flat.items()}
    return out


def _iter_fam_cells(block, path=""):
    """Every paired cell {delta, lo, hi} of a family block, flattened. A cell is a dict carrying delta / lo / hi."""
    if isinstance(block, dict):
        if all(k in block for k in ("delta", "lo", "hi")) and isinstance(block["delta"], (int, float)):
            if all(np.isfinite(float(block[k])) for k in ("delta", "lo", "hi")):          # an undefined cell (NaN) is not a contrast
                yield path, {"delta": float(block["delta"]), "lo": float(block["lo"]), "hi": float(block["hi"]), "point": float(block["delta"])}
        else:
            for k, v in block.items():
                yield from _iter_fam_cells(v, f"{path}/{k}" if path else str(k))


def _trim_dk(read):
    r = {k: v for k, v in read.items() if k not in ("ttc_stride2_sensitivity", "code")}
    r["arms"] = {nm: {k: v for k, v in ab.items() if k != "by_speed"} for nm, ab in read["arms"].items()}
    return r


# ------------------------------------------------------------------------------------------------ the read
def read(a, pairfn, famfn, DK, WC):
    seeds = (0,) if a.stage == "s0" else (0, 1)
    _, logs = load_runs(a.runs_dir, seeds)
    Zp, eid = load_all(a.eval_dir, seeds, ARMS, WC)
    Zb_all, _ = load_all(a.eval_dir, (0, 1), (ARM_B,), WC)          # the seed floor needs BOTH banked M2b evals at either stage
    if not np.array_equal(Zb_all[(ARM_B, 0)]["eid"], eid):
        refuse("the banked M2b evals are not on the same eval grid as the M2b-pgt evals (eid differs)")
    check_gt_codes(Zp, seeds, WC)
    F = abs(float(Zb_all[(ARM_B, 0)]["ade6"].mean()) - float(Zb_all[(ARM_B, 1)]["ade6"].mean()))
    ctx = build_ctx(a, DK, eid)
    Zdk = {k: v for k, v in {**Zb_all, **Zp}.items()}
    pairs = [(f"{ARM_P}_s{s}", f"{ARM_B}_s{s}") for s in seeds] + [(f"{ARM_P}_s{s}", "P10") for s in seeds] + [(f"{ARM_B}_s{s}", "P10") for s in seeds]
    res10 = dk_read(a, DK, ctx, Zdk, ctx["blk10"], 10, a.dk_n_boot, pairs)
    pw = res10["perwindow"]
    if not np.array_equal(np.asarray(pw["is_lead"], bool), ctx["lead"]):
        refuse("the LEAD windows of the distance-keeping read are not the lead block's join")
    base = {s: baseline_stats(pairfn, Zb_all[(ARM_B, s)], pw, ctx, a.n_boot) for s in (0, 1)}
    known = known_values(base, F, (0, 1))
    idc = identity_control(pairfn, {s: Zb_all[(ARM_B, s)] for s in seeds}, pw, ctx, a.n_boot, WC, seeds)
    rows = {}
    pe, _ = WC.present_error(Zp, seeds)
    for s in seeds:
        A, Bm = Zp[(ARM_P, s)], Zp[(ARM_B, s)]
        row = contrast(pairfn, A, Bm, pw, ctx, a.n_boot)
        row["base"] = {k: base[s][k] for k in ("lambda_m2b_m", "lambda_ci", "n_lead", "excess_p10", "excess_ci")}
        row["judge"] = judge(row, base[s], F)
        ok = np.isfinite(pe[(ARM_B, s)]["mse"])
        row["beside"] = {
            "ade6": {arm: float(Zp[(arm, s)]["ade6"].mean()) for arm in ARMS},
            "oracle_ade6": {arm: float(Zp[(arm, s)]["orc_ade6"].mean()) for arm in ARMS},
            "oracle_ade6_delta": paired(pairfn, A["orc_ade6"], Bm["orc_ade6"], eid, a.n_boot),
            "present_slot": {"n_windows_present": int(ok.sum()), "rmse": {arm: float(np.sqrt(pe[(arm, s)]["mse"][ok].mean())) for arm in ARMS},
                             "rmse_group": {arm: pe[(arm, s)]["rmse_group"] for arm in ARMS},
                             "mse_delta": paired(pairfn, pe[(ARM_P, s)]["mse"][ok], pe[(ARM_B, s)]["mse"][ok], eid[ok], a.n_boot),
                             "note": "T trains on S's own checkpoints more often under this arm and may get worse; reported, deciding nothing"},
            "eval_json": {arm: {k: Zp[(arm, s)]["json"].get(k) for k in ("traj", "families", "strategic", "tactical") if k in Zp[(arm, s)]["json"]} for arm in ARMS},
            "distance_keeping_k10": {"pair_vs_m2b": res10["read"]["pairs"].get(f"{ARM_P}_s{s}__minus__{ARM_B}_s{s}"),
                                     "pair_vs_p10": res10["read"]["pairs"].get(f"{ARM_P}_s{s}__minus__P10"),
                                     "arms": {nm: res10["read"]["arms"][nm] for nm in (f"{ARM_P}_s{s}", f"{ARM_B}_s{s}", "P10")}},
            "gt_source": {arm: Zp[(arm, s)]["gt_source"] for arm in ARMS}}
        if famfn is not None:
            row["beside"]["families"] = famfn(A["pred"], Bm["pred"], eid)
        rows[f"s{s}"] = row
    k30 = None
    if not a.no_k30:
        p30 = a.lead_block_k30 or str(DK.LEAD_BLOCK_K30)
        if os.path.exists(p30):
            r30 = dk_read(a, DK, ctx, Zdk, DK.load_block(p30), 30, a.dk_n_boot, pairs)
            k30 = _trim_dk(r30["read"])
    decision, vd, mech_all = verdict(a.stage, rows, seeds)
    nxt = NEXT[None if decision == "CONTINUE" else vd]
    leak = " / ".join(f"{rows[f's{s}']['mleak']['delta']['point']:+.4f}{'' if rows[f's{s}']['mleak']['delta']['separated'] else ' ns'}" for s in seeds)
    tail = " / ".join(f"{100 * rows[f's{s}']['mdk']['delta']['point']:+.2f}pp{'' if rows[f's{s}']['mdk']['delta']['separated'] else ' ns'}" for s in seeds)
    prim = " / ".join(f"{rows[f's{s}']['primary']['point']:+.3f}{'' if rows[f's{s}']['primary']['separated'] else ' ns'}" for s in seeds)
    out = dict(schema="v8f-pgt-read-v1", prereg=f"PREREG_V8F_PGT.md stage {a.stage}", tier="T1-class open loop (eval-141, 2,399 windows); mechanism reads use T0 oracle evals",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); F = |M2b_s0 - M2b_s1| = {F:.4f} m (both banked evals); "
                         f"answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)",
               n_windows=int(eid.size), n_episodes=int(np.unique(eid).size), n_lead_windows_k10=int(ctx["lead"].sum()), F=F, two_F=2 * F,
               dose={"M2b": [M2B_START, M2B_END], "M2b-pgt": [PGT_START, PGT_END]}, known_values=known, identity_control=idc,
               train_log={f"{k[0]}_s{k[1]}": v for k, v in logs.items()}, stage=a.stage, rows=rows,
               strategic_family="NOT PAUSED: un-paused for v8F-mini (PI ruling 2026-10-17); read from each run's eval json under rows.*.beside.eval_json",
               distance_keeping_k30=k30, DECISION=decision, VERDICT=vd, mechanism=mech_all, next=nxt)
    out["reading"] = (f"PREREG_V8F_PGT stage {a.stage}: {decision or vd}; ADE6 M2b-pgt - M2b {prim} m (2F {2 * F:.4f}); M-LEAK d-LAMBDA {leak} m "
                      f"(M2b {' / '.join(f'{base[s]['lambda_m2b_m']:+.4f}' for s in seeds)}); M-DK d-S5 {tail} (M2b excess over P10 "
                      f"{' / '.join(f'{100 * base[s]['excess_p10']:+.2f}pp' for s in seeds)}); mechanism {'MET' if mech_all else 'NOT met'}." +
                      ("" if vd is None else f" VERDICT {vd}."))
    return out


def check_inputs(a, DK, WC, pairfn, famfn=None):
    """The banked M2b side alone, before any training (the WC lesson: a reader tested only on synthetic inputs crashed on the real ones): the
    runs and evals of BOTH seeds load, the read inputs build, the PREREG's known values reproduce, the paired instrument reads exactly 0 on M2b
    against itself. Returns what it checked."""
    runs_ok, logs = load_runs(a.runs_dir, (0, 1), arms=(ARM_B,))
    Zb, eid = load_all(a.eval_dir, (0, 1), (ARM_B,), WC)
    check_gt_codes(Zb, (0, 1), WC)
    F = abs(float(Zb[(ARM_B, 0)]["ade6"].mean()) - float(Zb[(ARM_B, 1)]["ade6"].mean()))
    ctx = build_ctx(a, DK, eid)
    res10 = dk_read(a, DK, ctx, Zb, ctx["blk10"], 10, a.dk_n_boot, [])
    pw = res10["perwindow"]
    base = {s: baseline_stats(pairfn, Zb[(ARM_B, s)], pw, ctx, a.n_boot) for s in (0, 1)}
    known = known_values(base, F, (0, 1))
    idc = identity_control(pairfn, {s: Zb[(ARM_B, s)] for s in (0, 1)}, pw, ctx, a.n_boot, WC, (0, 1), fam=famfn)
    return {"F": F, "lambda_m2b": {s: base[s]["lambda_m2b_m"] for s in (0, 1)}, "excess_p10": {s: base[s]["excess_p10"] for s in (0, 1)},
            "n_lead_k10": int(ctx["lead"].sum()), "gt_source": {f"{k[0]}_s{k[1]}": v["gt_source"] for k, v in Zb.items()}, "train_log": {f"{k[0]}_s{k[1]}": v for k, v in logs.items()},
            "known_values": known, "identity_control": idc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("s0", "final"))
    ap.add_argument("--check-inputs", action="store_true", help="validate the banked M2b side alone (and the identity control) and exit")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
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
        DK, WC = _modules()
        a.p10_pred = a.p10_pred or str(DK.P10_DEFAULT)
        from taniteval.ci import paired_episode_cluster_bootstrap as pairfn
        if a.check_inputs:
            famfn = None if a.no_families else _famfn(a)
            res = check_inputs(a, DK, WC, pairfn, famfn)
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            print(f"PGT INPUTS OK: the banked M2b side loads; F {res['F']:.4f}; LAMBDA(M2b) "
                  f"{res['lambda_m2b'][0]:+.4f} / {res['lambda_m2b'][1]:+.4f}; S5 excess over P10 {res['excess_p10'][0]:+.4f} / {res['excess_p10'][1]:+.4f}; "
                  f"identity control exactly +0 [0, 0]; GT sources {res['gt_source']}")
            return 0
        if a.stage is None:
            ap.error("--stage is required unless --check-inputs")
        res = read(a, pairfn, None if a.no_families else _famfn(a), DK, WC)
    except (OSError, KeyError, ValueError) as e:
        print(f"PGT READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(DK._clean(res), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


def _famfn(a):
    R1 = _load("v8f_tac_groups_read_for_pgt", HERE.parent / "v8f_tac_groups_read.py")
    R1._paths()
    return R1.family_fn(a)


if __name__ == "__main__":
    sys.exit(main())
