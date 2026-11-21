"""S1 read (PREREG_V8F_PROOF_ADD12 note N8, "state-file mode"): the reads of stage 0e on a GIVEN trained state instead of states fitted by the tool.
Tier T0: a world-model diagnostic, never driving performance. CPU only. Choices in S1_DEFS.md (rules S1 .. S14). Reuses spatial/s0e/s0e_read.py by import
(loaders, folds, pair builder, probe subsample, bootstrap helpers), through it spatial/s0/s0_read.py and spatial/s0c/s0c_read.py. None of them is edited.
Subcommands:
    run    --states <npz> --tag TAG --expect-dim D --expect-state mean|grid [--ckpt-sha256 HEX]   -> RESULT_S1_<TAG>.json, VERDICT_S1_<TAG>.md, OOF_S1_<TAG>.npz
    pair   --a RESULT_S1_<A>.json --b RESULT_S1_<B>.json                                           -> PAIR_S1_<A>_minus_<B>.json/.md
    spread --results RESULT_S1_<..>.json ... [--name NAME]                                        -> SPREAD_S1_<NAME>.json/.md
    smoke  --states <npz> --expect-dim D --expect-state mean|grid [--clips N] [--folds K]          label blind: swapped label tracks, prints shapes, counts, timings, t1, t4
Exit code 2 = a control missed (numbers written, no label).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import hashlib
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ROOT = "F:/TanitAD-scratch/v8proof"
S0E_DIR = f"{ROOT}/spatial/s0e"
for _p in (S0E_DIR,):
    if _p not in sys.path:
        sys.path.append(_p)
import s0e_read as E  # noqa: E402  (puts s0 and s0c on the path)
import s0_read as S0  # noqa: E402
import s0c_read as S0C  # noqa: E402
RP = S0.RP

SEED, B_BOOT = S0.SEED, S0.B_BOOT
TIER, STATEMENT, FAMILIES = S0.TIER, S0.STATEMENT, S0.FAMILIES
PROBE_CAP = E.PROBE_CAP
ARMS3 = ("H", "N", "E")
ARM_KEY = {"H": "H-s", "N": "N-s", "E": "E"}        # arm_features of stage 0c reads a state by the name after the dash: St = {"s": Z}
E1_BAR = 0.02
T2_TOL = 0.005
REF_RESULT = f"{S0E_DIR}/RESULT_S0E.json"
CACHE_DIR = f"{HERE}/cache"
IN_SAMPLE = "in-sample for the model (train probe clips)"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def peak_ram_gb():
    return E.peak_ram_gb()


# ===================================================================================== state file (S1, S2)
def load_states(path):
    z = np.load(path, allow_pickle=False)
    need = {"Z", "clip_idx", "t", "clips"}
    if not need <= set(z.files):
        raise SystemExit(f"REFUSED: state file lacks {sorted(need - set(z.files))}")
    S = {"Z": z["Z"], "clip_idx": z["clip_idx"], "t": z["t"], "clips": [str(c) for c in z["clips"]]}
    rec_path = path + ".json"
    S["record"] = None
    if os.path.exists(rec_path):
        with open(rec_path, "r", encoding="utf-8") as fh:
            S["record"] = json.load(fh)
    return S


# ===================================================================================== controls t1 and t4 (S3, S4)
def check_t1(S, D_raw):
    """Alignment, exact: clip list, clip_idx and t equal the probe set's, rows equal, state finite, 2-D. D_raw = the probe set before any row is dropped."""
    Z = S["Z"]
    res = {"clips_equal": bool(list(S["clips"]) == list(D_raw["uep"])), "n_rows_state": int(len(Z)), "n_rows_probe": int(len(D_raw["epi"]))}
    same_n = bool(len(Z) == len(D_raw["epi"]) and len(S["clip_idx"]) == len(Z) and len(S["t"]) == len(Z))
    res["rows_equal"] = same_n
    res["clip_idx_equal"] = bool(same_n and np.array_equal(S["clip_idx"].astype(np.int64), np.asarray(D_raw["epi"], np.int64)))
    res["t_equal"] = bool(same_n and np.array_equal(S["t"].astype(np.int64), np.asarray(D_raw["bg"], np.int64)))
    res["z_is_2d"] = bool(Z.ndim == 2)
    res["z_finite"] = bool(Z.ndim == 2 and np.isfinite(Z).all())
    res["pass"] = bool(all(res[k] for k in ("clips_equal", "rows_equal", "clip_idx_equal", "t_equal", "z_is_2d", "z_finite")))
    res["bar"] = "clips, clip_idx and t equal the probe set exactly; state finite"
    return res


def check_t4(S, npz_path, expect_dim, expect_state, ckpt_sha256=None, expect_frames=None, expect_clips=None):
    """The state is what the arm declares: width, state mode, checkpoint sha256 against the record file written with the state."""
    rec = S["record"]
    Z = S["Z"]
    res = {"declared_dim": int(expect_dim), "declared_state": expect_state, "array_dim": int(Z.shape[1]) if Z.ndim == 2 else None}
    res["record_present"] = rec is not None
    res["dim_array"] = bool(Z.ndim == 2 and Z.shape[1] == expect_dim)
    res["dim_record"] = bool(rec is not None and rec.get("D") == expect_dim)
    res["state_mode_record"] = bool(rec is not None and rec.get("wm_state") == expect_state)
    res["mode_width_consistent"] = bool(expect_state != "grid" or expect_dim == 640)
    if rec is not None:
        res["record_wm_state"] = rec.get("wm_state")
        res["record_ckpt_sha256"] = rec.get("ckpt_sha256")
        res["record_counts_equal"] = bool(rec.get("n_frames") == len(Z) and (expect_clips is None or rec.get("n_clips") == expect_clips)
                                          and (expect_frames is None or rec.get("n_frames") == expect_frames))
        res["record_split_range"] = bool(rec.get("split") == "train" and rec.get("lo") == 600 and rec.get("hi") == 1800) if expect_clips == 1200 else True
        osha = rec.get("out_sha256")
        res["state_file_sha256_equals_record"] = bool(osha is not None and E.sha256_file(npz_path) == osha)
    else:
        res["record_counts_equal"] = res["record_split_range"] = res["state_file_sha256_equals_record"] = False
    if ckpt_sha256 is None:
        res["ckpt_sha256"] = "NOT CHECKED: --ckpt-sha256 not given (the record's value is printed above)"
        ck_ok = True
    else:
        ck_ok = bool(rec is not None and str(rec.get("ckpt_sha256", "")).lower() == ckpt_sha256.lower())
        res["ckpt_sha256"] = "equal to the record" if ck_ok else "DIFFERENT from the record"
    res["pass"] = bool(all(res[k] for k in ("record_present", "dim_array", "dim_record", "state_mode_record", "mode_width_consistent", "record_counts_equal",
                                            "record_split_range", "state_file_sha256_equals_record")) and ck_ok)
    res["bar"] = "array width, record width, state mode, counts and the state file's own sha256 equal the record; checkpoint sha256 equal to the one given"
    return res


# ===================================================================================== the reads, one fold (S5, S6, S7)
def ra_state_fold(f, Z, ego, y_gap, y_pres, epi, folds_b, out, lab):
    """r-a, one fold: lead gap (finite rows) and presence (all rows) from the state and from the ego vector (z-scored on the training rows). The call
    pattern is that of s0e_read.ra_fold: S0.fit_probe on at most PROBE_CAP training rows (seed 0 + fold). The state is used as given."""
    tr = np.flatnonzero(folds_b != f); te = np.flatnonzero(folds_b == f)
    mu, sd = ego[tr].mean(0), ego[tr].std(0) + 1e-9
    feats = {"s": Z, "ego": (ego - mu) / sd}
    info = {}
    for s, F in feats.items():
        for tg, y in (("gap", y_gap), ("pres", y_pres)):
            key = f"ra|{lab}|{s}|{tg}"
            if key not in out:
                out[key] = np.full(len(y), np.nan)
            vtr_all = tr[np.isfinite(y[tr])]
            vtr = E.sub_rows(vtr_all); vte = te[np.isfinite(y[te])]
            info[f"{tg}|train_rows_available"] = int(len(vtr_all)); info[f"{tg}|train_rows_used"] = int(len(vtr))
            if len(vtr) >= 12 and len(vte):
                out[key][vte] = S0.fit_probe(F[vtr], y[vtr], epi[vtr], SEED + f)(F[vte])
    return info


def change_state_fold(f, Z, Pr, dsets, out):
    """Change read, one fold: arms H (state at g and g-5 plus ego block), N (state at g plus ego block), E (ego block) by S0C.arm_features."""
    ri, rm = Pr["ri"], Pr["rm"]
    trp_all, tep = np.flatnonzero(Pr["fold_p"] != f), np.flatnonzero(Pr["fold_p"] == f)
    trp = E.sub_rows(trp_all)
    St = {"s": Z}
    for lab, d in dsets.items():
        for arm in ARMS3:
            F = S0C.arm_features(ARM_KEY[arm], St, Pr["Eb"], ri, rm, trp)
            key = f"arm|{lab}|{arm}"
            out[key] = np.full(len(ri), np.nan)
            if len(tep):
                out[key][tep] = S0C.arm_oof_fold(F, d, Pr["ep"], trp, tep, f)
            del F
    return {"pairs_train_available": int(len(trp_all)), "pairs_train_used": int(len(trp)), "pairs_test": int(len(tep))}


def process_fold(f, Z, D, Pr, labsets, log_=log, ra_only=False):
    t0 = time.time(); T = {}; out = {}; info = {}
    for lab, L in labsets.items():
        info[f"ra|{lab}"] = ra_state_fold(f, Z, D["kin"], L["gap"], L["pres"], D["epi"], Pr["folds_b"], out, lab)
    T["r_a"] = time.time() - t0
    log_(f"fold {f}: r-a done ({T['r_a']:.0f}s)")
    if not ra_only:
        t1 = time.time()
        info["change"] = change_state_fold(f, Z, Pr, {lab: L["d"] for lab, L in labsets.items()}, out)
        T["change"] = time.time() - t1
        log_(f"fold {f}: change read done ({T['change']:.0f}s)")
    T["fold_total"] = time.time() - t0
    return out, T, info


def compute_oof(Z, D, Pr, labsets, n_folds, stage_dir=None, sig=None, log_=log, ra_only=False):
    parts, times, infos = [], {}, {}
    for f in range(n_folds):
        sp = f"{stage_dir}/fold{f}.npz" if stage_dir else None
        if sp and os.path.exists(sp):
            z = np.load(sp, allow_pickle=True)
            if json.loads(str(z["sig"])) == sig:
                parts.append({k: z[k] for k in z.files if k not in ("sig", "times", "info")})
                times[f"fold{f}"] = json.loads(str(z["times"])); log_(f"fold {f}: resumed"); continue
        o, T, info = process_fold(f, Z, D, Pr, labsets, log_, ra_only)
        times[f"fold{f}"] = T; infos[f"fold{f}"] = info
        if sp:
            os.makedirs(stage_dir, exist_ok=True)
            np.savez(sp, sig=json.dumps(sig), times=json.dumps(T), info=json.dumps(info), **o)
        parts.append(o)
    return E.merge_oof(parts), times, infos


# ===================================================================================== analysis (S8)
def make_labsets(D, Pr):
    d, folds_b, fold_p = Pr["d"], Pr["folds_b"], Pr["fold_p"]
    perm = {"gap": S0.permute_within_fold(D["gap"], folds_b, 1), "pres": S0.permute_within_fold(D["pres"], folds_b, 2), "d": S0.permute_within_fold(d, fold_p, 1)}
    return {"main": {"gap": D["gap"], "pres": D["pres"], "d": d}, "e1": perm}


def analyse(D, Pr, oof, C, n_ep):
    epi, yg, yp, d, ep = D["epi"], D["gap"], D["pres"], Pr["d"], Pr["ep"]
    ra = {}
    for s in ("s", "ego"):
        ra[f"gap|{s}"] = E.single_r2c(epi, yg, oof[f"ra|main|{s}|gap"], C, n_ep)
        ra[f"pres|{s}"] = E.single_auc_c(epi, yp, oof[f"ra|main|{s}|pres"], C)
    P = lambda s, tg: oof[f"ra|main|{s}|{tg}"]
    dg = {"state minus ego": S0.paired(epi, yg, P("s", "gap"), P("ego", "gap"), C, n_ep)}
    dp = {"state minus ego": E.paired_auc_c(epi, yp, P("s", "pres"), P("ego", "pres"), C)}
    sk = {a: S0C.single_skill(ep, d, oof[f"arm|main|{a}"], C, n_ep) for a in ARMS3}
    pr = lambda a: oof[f"arm|main|{a}"]
    diffs = {"H minus E": S0C.paired_skill(ep, d, pr("H"), pr("E"), C, n_ep), "H minus N": S0C.paired_skill(ep, d, pr("H"), pr("N"), C, n_ep)}
    dstat = {"n_pairs": int(len(d)), "n_pair_clips": int(len(np.unique(ep))), "std_d_m": float(np.std(d)), "mean_d_m": float(np.mean(d)),
             "share_abs_d_above_2m": float((np.abs(d) > 2.0).mean())}
    return {"r_a": {"estimates": ra, "diffs_gap_r2": dg, "diffs_presence_auc": dp}, "change": {"arms": sk, "diffs": diffs, "d_stats": dstat}}


def t3_values(D, Pr, oof):
    """Control t3: every R2 (state, ego) and every change skill (H, N, E) on labels permuted within fold is at or below the bar; presence AUROC printed beside."""
    vals = {f"ra gap R2 {s}": RP.r2_pooled(D["gap"], oof[f"ra|e1|{s}|gap"]) for s in ("s", "ego")}
    vals.update({f"change skill {a}": S0C.skill(Pr["d_e1"], oof[f"arm|e1|{a}"]) for a in ARMS3})
    auc = {f"presence AUROC {s} (printed, no bar)": RP.auc_pooled(D["pres_e1"], oof[f"ra|e1|{s}|pres"]) for s in ("s", "ego")}
    return {"pass": bool(all(np.isfinite(v) and v <= E1_BAR for v in vals.values())), "max_value": float(max(vals.values())), "bar": E1_BAR, "values": vals,
            "presence_auc_printed": auc, "n_gap_rows": int(np.isfinite(D["gap"]).sum()), "n_pairs": int(len(Pr["d"])), "n_clips": int(len(D["uep"]))}


# ===================================================================================== control t2 (S9)
def sha_of(path):
    return E.sha256_file(path)


def t2_cache_key():
    return {"s0e_read_sha256": sha_of(f"{S0E_DIR}/s0e_read.py"), "index_sha256": sha_of(f"{E.DATA}/INDEX.json"), "s1_read_sha256": sha_of(f"{HERE}/s1_read.py")}


def t2_state_values(D, Pr, M, n_folds=5, log_=log):
    """The 64-number mean over the 40 projected cells as the 'state': pooled gap R2 and presence AUROC through the SAME code path as a state file."""
    labsets = {"main": {"gap": D["gap"], "pres": D["pres"], "d": Pr["d"]}}
    oof, _, _ = compute_oof(M, D, Pr, labsets, n_folds, log_=log_, ra_only=True)
    return {"gap_r2": float(RP.r2_pooled(D["gap"], oof["ra|main|s|gap"])), "pres_auc": float(RP.auc_pooled(D["pres"], oof["ra|main|s|pres"])),
            "n_gap_rows": int(np.isfinite(D["gap"]).sum()), "n_rows": int(len(D["gap"])), "n_clips": int(len(D["uep"]))}


def t2_reference(path=REF_RESULT):
    """Reads ONLY the two values of the mean state from RESULT_S0E.json (and its controls_pass flag)."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        r = json.load(fh)
    try:
        est = r["main"]["r_a"]["estimates"]
        return {"gap_r2": float(est["gap|m"]["r2"]), "pres_auc": float(est["pres|m"]["auc"]), "ref_controls_pass": bool(r.get("controls_pass"))}
    except (KeyError, TypeError):
        return {"missing": True, "ref_controls_pass": bool(r.get("controls_pass"))}


def check_t2(values, ref, tol=T2_TOL):
    res = {"state_mode_values": values, "tol": tol, "reference_file": REF_RESULT}
    if ref is None:
        res.update({"pass": False, "reason": "reference absent: RESULT_S0E.json does not exist yet"})
        return res
    if ref.get("missing"):                                    # note N9 (control f4): the two values are an identity reference of two code paths on ONE state; they do not need the stage 0e controls to pass
        res.update({"pass": False, "reason": "reference has no mean-state values (its layout differs)"})
        return res
    dg, dp = abs(values["gap_r2"] - ref["gap_r2"]), abs(values["pres_auc"] - ref["pres_auc"])
    res.update({"reference_values_read_at_run_time": {"gap_r2": ref["gap_r2"], "pres_auc": ref["pres_auc"]}, "abs_diff_gap_r2": float(dg), "abs_diff_pres_auc": float(dp),
                "pass": bool(dg <= tol and dp <= tol), "bar": f"both within {tol} of the stage 0e mean state"})
    return res


def get_t2_values(D, Pr, M_fn, cache_dir=CACHE_DIR, log_=log):
    """Cached by the sha256 of s0e_read.py, INDEX.json and s1_read.py: it runs once, not per state file. M_fn() returns the mean token [N, 64] (float32)."""
    key = t2_cache_key()
    name = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    p = f"{cache_dir}/t2_{name}.json"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            c = json.load(fh)
        if c.get("key") == key:
            log_(f"t2 values from cache {p}")
            return c["values"], True
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    v = t2_state_values(D, Pr, M_fn(), log_=log_)
    v["seconds"] = time.time() - t0
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"key": key, "values": v}, fh, indent=1)
    return v, False


# ===================================================================================== output
def ci_s(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def label_of(expect_state, expect_dim, tag):
    return (f"baseline (trained mean state), {tag}" if (expect_state == "mean" and expect_dim == 128) else f"trained state, {tag}")


def write_verdict(path, res):
    ctr, meta = res["controls"], res["meta"]
    L = [f"# VERDICT_S1_{meta['tag']}: PREREG_V8F_PROOF_ADD12 note N8, state-file mode"]
    L.append(f"Tier {TIER}: {STATEMENT}. Numbers are {IN_SAMPLE}. Estimator: paired episode-cluster bootstrap with clips as clusters ({meta['n_clips']} clips), B {meta['B']}, seed 0, one draw "
             f"shared by every number. {FAMILIES}")
    L.append("An interval over clips is blind to training variance (H-ESTIM-SEED-1): a difference between two trained arms needs the replicate spread (`spread`) beside it.")
    L.append("")
    L.append("## CONTROLS (first)")
    for k in ("t1", "t4", "t2", "t3"):
        if k not in ctr:
            L.append(f"- {k}: NOT RUN (an earlier control failed)")
            continue
        c = ctr[k]
        L.append(f"- {k}: {'PASS' if c['pass'] else 'FAIL'} :: " + json.dumps({a: b for a, b in c.items() if a != 'pass'}, default=float))
    L.append(f"- controls_pass: {res['controls_pass']}")
    L.append("")
    if not res["controls_pass"]:
        L.append("## LABEL WITHHELD: a control did not read its known value. Any numbers are in the RESULT file and are not quotable.")
    else:
        m = res["main"]; ra, ch = m["r_a"], m["change"]; n = meta["counts"]
        L.append(f"## Label: {res['label']}")
        L.append(f"No outcome name of the card is assigned by this tool. State width {meta['dim']}, mode {meta['expect_state']}, state file {meta['states_file_name']}, checkpoint sha256 {meta['ckpt_sha256_record']}.")
        L.append(f"## Reads (a): lead gap and presence from one frame ({TIER}; {STATEMENT}; {IN_SAMPLE})")
        L.append(f"Frames {n['bank_rows']}, frames with a lead gap in 0 to 60 m {n['gap_rows']}, clips {meta['n_clips']}; probe fit subsample at most {PROBE_CAP} training rows per fold.")
        for s, nm in (("s", "trained state"), ("ego", "ego vector")):
            a, b = ra["estimates"][f"gap|{s}"], ra["estimates"][f"pres|{s}"]
            L.append(f"- {nm}: gap R2 {a['r2']:+.3f} {ci_s(a['ci95'])} (n {a['n']} frames, {a['n_clips']} clips); presence AUROC {b['auc']:.3f} {ci_s(b['ci95'])} (n {b['n']} frames, {b['n_clips']} clips)")
        v, w = ra["diffs_gap_r2"]["state minus ego"], ra["diffs_presence_auc"]["state minus ego"]
        L.append(f"- state minus ego: gap R2 {v['diff']:+.3f} {ci_s(v['ci95'])}; presence AUROC {w['diff']:+.3f} {ci_s(w['ci95'])}")
        L.append(f"## Reads (b): change of the lead gap over 2.0 s, skill over persistence ({TIER}; {STATEMENT}; {IN_SAMPLE})")
        ds = ch["d_stats"]
        L.append(f"Pairs {ds['n_pairs']} of {ds['n_pair_clips']} clips; pairs removed by the continuity rule {n['pairs_removed_by_the_continuity_rule']}; std(d) {ds['std_d_m']:.3f} m, "
                 f"mean {ds['mean_d_m']:+.3f} m, share |d| above 2 m {ds['share_abs_d_above_2m']:.3f}; fit subsample at most {PROBE_CAP} training pairs per fold.")
        for a in ARMS3:
            r = ch["arms"][a]
            L.append(f"- {a}: skill {r['skill']:+.3f} {ci_s(r['ci95'])}; plain R2 {r['r2']:+.3f} {ci_s(r['r2_ci95'])}; n pairs {r['n_pairs']}, clips {r['n_episodes']}")
        for k, v in ch["diffs"].items():
            L.append(f"- {k}: {v['diff']:+.3f} {ci_s(v['ci95'])}")
    L.append("")
    L.append("Counts: " + json.dumps(meta["counts"]))
    L.append(f"Peak RAM {meta.get('peak_ram_gb')} GB; seconds {meta.get('total_seconds')}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ===================================================================================== driver: probe set
def load_probe(n_clips=None, swap=False):
    J = E.load_join()
    D = E.load_train(n_clips, join=J)
    if swap:
        D = E.block_swap_tracks(D, 0)
    return D


def prepare(D):
    """Rows with a non-finite kinematic value are dropped (S2); returns the keep mask, pairs, and the labels of the control t3."""
    keep = np.isfinite(D["kin"]).all(1)
    D = E.drop_nonfinite_kin(D)
    Pr = E.assemble(D)
    Pr["d_e1"] = S0.permute_within_fold(Pr["d"], Pr["fold_p"], 1)
    D["pres_e1"] = S0.permute_within_fold(D["pres"], Pr["folds_b"], 2)
    return D, Pr, keep


def counts_of(D, Pr):
    return {**Pr["counts"], "bank_rows": int(len(D["gap"])), "gap_rows": int(np.isfinite(D["gap"]).sum()), "rows_dropped_nonfinite_kin": int(D.get("n_dropped_nonfinite_kin", 0)),
            "pairs_per_fold": [int((Pr["fold_p"] == f).sum()) for f in range(5)]}


def save_oof(path, D, Pr, oof):
    keys = {k.replace("|", "__"): v for k, v in oof.items() if k.startswith("ra|main|") or k.startswith("arm|main|")}
    np.savez_compressed(path, gap=D["gap"], pres=D["pres"], epi=D["epi"].astype(np.int32), d=Pr["d"], ep=Pr["ep"].astype(np.int32), **keys)


def cmd_run(a, smoke=False):
    t_all = time.time()
    assert not S0.LEAK_PCA_FIT
    tag = a.tag if not smoke else "SMOKE"
    S = load_states(a.states)
    n_clips = a.clips if smoke else None
    D_raw = load_probe(n_clips, swap=smoke)
    t_load = time.time() - t_all
    ctr = {}
    ctr["t1"] = check_t1(S, D_raw)
    ctr["t4"] = check_t4(S, a.states, a.expect_dim, a.expect_state, a.ckpt_sha256, expect_clips=len(D_raw["uep"]) if not smoke else len(D_raw["uep"]))
    log("t1", ctr["t1"]["pass"], "t4", ctr["t4"]["pass"], "state", S["Z"].shape, "rows", len(D_raw["epi"]))
    out_dir = f"{HERE}/smoke" if smoke else HERE
    os.makedirs(out_dir, exist_ok=True)
    meta = {"tag": tag, "note": "PREREG_V8F_PROOF_ADD12_N8.md", "defs": "S1_DEFS.md", "tier": TIER, "statement": STATEMENT, "estimator": "paired episode-cluster bootstrap, clips as clusters",
            "B": 200 if smoke else B_BOOT, "n_clips": int(len(D_raw["uep"])), "dim": int(S["Z"].shape[1]) if S["Z"].ndim == 2 else None, "expect_state": a.expect_state,
            "states_file_name": os.path.basename(a.states), "ckpt_sha256_record": (S["record"] or {}).get("ckpt_sha256"), "in_sample": IN_SAMPLE}
    res = {"controls": ctr, "meta": meta}
    if not (ctr["t1"]["pass"] and ctr["t4"]["pass"]):
        res["controls_pass"] = False
        meta["total_seconds"] = round(time.time() - t_all, 1); meta["peak_ram_gb"] = round(peak_ram_gb(), 2)
        meta["counts"] = {}
        if not smoke:
            with open(f"{out_dir}/RESULT_S1_{tag}.json", "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=1, default=float)
            write_verdict(f"{out_dir}/VERDICT_S1_{tag}.md", res)
        log("a control missed (t1 or t4): nothing computed; exit 2")
        return 2
    keep = np.isfinite(D_raw["kin"]).all(1)
    Z = np.ascontiguousarray(S["Z"][keep]).astype(np.float32)
    D, Pr, _ = prepare(D_raw)
    n_ep = len(D["uep"])
    meta["counts"] = counts_of(D, Pr)
    nf = 5 if not smoke else a.folds
    # t2: cached; the mean token is only built when the cache is missing
    t_t2 = time.time()
    if smoke:
        ctr["t2"] = {"pass": None, "note": "not run in the smoke (it is the cached control of the real run)"}
    else:
        v, cached = get_t2_values(D, Pr, lambda: S0.mean_token(D["X"]))
        ctr["t2"] = check_t2(v, t2_reference()); ctr["t2"]["from_cache"] = cached
        log("t2", ctr["t2"]["pass"], v)
    D.pop("X", None)
    t_t2 = time.time() - t_t2
    labsets = make_labsets(D, Pr)
    sig = {"N": int(len(Z)), "P": int(len(Pr["ri"])), "cap": PROBE_CAP, "dim": int(Z.shape[1]), "state_sha256": E.sha256_file(a.states)[:16] if not smoke else "smoke"}
    stage_dir = None if smoke else f"{out_dir}/stages/{tag}"
    oof, times, infos = compute_oof(Z, D, Pr, labsets, nf, stage_dir, sig)
    if smoke:
        sec = {k: float(np.mean([t[k] for t in times.values()])) for k in next(iter(times.values()))}
        keep_out = {"state_shape": list(Z.shape), "t1": ctr["t1"], "t4": ctr["t4"], "counts": meta["counts"], "feature_shapes": {"H": [len(Pr["ri"]), 2 * Z.shape[1] + 9], "N": [len(Pr["ri"]), Z.shape[1] + 9], "E": [len(Pr["ri"]), 9]},
                    "folds_run": nf, "mean_seconds_per_fold_by_step": sec, "subsamples_fold0": infos.get("fold0"), "load_seconds": t_load, "total_seconds": time.time() - t_all,
                    "peak_ram_gb": peak_ram_gb(), "labels": "swapped tracks (label blind); nothing computed on them is stored or printed"}
        with open(f"{out_dir}/RESULT_S1_SMOKE_D{Z.shape[1]}.json", "w", encoding="utf-8") as fh:
            json.dump(keep_out, fh, indent=1, default=float)
        log("SMOKE", json.dumps(keep_out, default=float))
        return 0
    ctr["t3"] = t3_values(D, Pr, oof)
    res["controls_pass"] = bool(all(ctr[k]["pass"] for k in ("t1", "t2", "t3", "t4")))
    C = RP.boot_counts(n_ep, meta["B"], SEED)
    res["main"] = analyse(D, Pr, oof, C, n_ep)
    oof_path = f"{out_dir}/OOF_S1_{tag}.npz"
    save_oof(oof_path, D, Pr, oof)
    res["oof_file"] = os.path.basename(oof_path)
    meta["stage_seconds"] = times; meta["t2_seconds"] = t_t2; meta["load_seconds"] = t_load
    meta["total_seconds"] = round(time.time() - t_all, 1); meta["peak_ram_gb"] = round(peak_ram_gb(), 2)
    if res["controls_pass"]:
        res["label"] = label_of(a.expect_state, meta["dim"], tag)
    with open(f"{out_dir}/RESULT_S1_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    write_verdict(f"{out_dir}/VERDICT_S1_{tag}.md", res)
    log(f"done {meta['total_seconds']}s; controls_pass {res['controls_pass']}; peak RAM {meta['peak_ram_gb']} GB")
    return 0 if res["controls_pass"] else 2


# ===================================================================================== pair and spread (S11, S12)
def _load_run(path):
    p = path if os.path.isabs(path) or os.path.exists(path) else f"{HERE}/{path}"
    with open(p, "r", encoding="utf-8") as fh:
        r = json.load(fh)
    return r, os.path.dirname(os.path.abspath(p)).replace("\\", "/")


def paired_from(zA, zB, B=B_BOOT):
    """Differences A minus B from stored out-of-fold predictions; the labels of the two runs must be identical arrays. One bootstrap draw."""
    for k in ("gap", "pres", "d", "epi", "ep"):
        same = np.array_equal(zA[k], zB[k], equal_nan=True) if zA[k].dtype.kind == "f" else np.array_equal(zA[k], zB[k])
        if not same:
            raise SystemExit(f"REFUSED: the two runs carry different {k} arrays")
    epi, ep, yg, yp, d = zA["epi"].astype(np.int64), zA["ep"].astype(np.int64), zA["gap"], zA["pres"], zA["d"]
    n_ep = int(epi.max()) + 1
    C = RP.boot_counts(n_ep, B, SEED)
    g = lambda z, k: z[k]
    out = {"gap R2 (state)": S0.paired(epi, yg, g(zA, "ra__main__s__gap"), g(zB, "ra__main__s__gap"), C, n_ep),
           "presence AUROC (state)": E.paired_auc_c(epi, yp, g(zA, "ra__main__s__pres"), g(zB, "ra__main__s__pres"), C)}
    for arm in ARMS3:
        out[f"change skill {arm}"] = S0C.paired_skill(ep, d, g(zA, f"arm__main__{arm}"), g(zB, f"arm__main__{arm}"), C, n_ep)
    n = {"n_frames": int(np.isfinite(yg).sum()), "n_pairs": int(len(d)), "n_clips": n_ep, "B": B}
    return out, n


def cmd_pair(a):
    rA, dA = _load_run(a.a); rB, dB = _load_run(a.b)
    for r, nm in ((rA, "a"), (rB, "b")):
        if not r.get("controls_pass") or "main" not in r:
            print(f"REFUSED: run {nm} did not pass its controls"); return 2
    zA = np.load(f"{dA}/{rA['oof_file']}"); zB = np.load(f"{dB}/{rB['oof_file']}")
    zA = {k: zA[k] for k in zA.files}; zB = {k: zB[k] for k in zB.files}
    diffs, n = paired_from(zA, zB, a.B)
    tA, tB = rA["meta"]["tag"], rB["meta"]["tag"]
    res = {"a": tA, "b": tB, "diffs_a_minus_b": diffs, "n": n, "tier": TIER, "statement": STATEMENT, "in_sample": IN_SAMPLE,
           "estimator": "paired episode-cluster bootstrap, clips as clusters, one shared draw", "widths": [rA["meta"]["dim"], rB["meta"]["dim"]]}
    out = f"{HERE}/PAIR_S1_{tA}_minus_{tB}"
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    L = [f"# PAIR_S1: {tA} minus {tB}", f"Tier {TIER}: {STATEMENT}. {IN_SAMPLE}. Paired episode-cluster bootstrap, clips as clusters ({n['n_clips']} clips), B {n['B']}, seed 0, one draw. "
         f"Frames with a gap {n['n_frames']}, pairs {n['n_pairs']}. {FAMILIES}",
         "A difference is separated only if it holds at both seeds AND is larger than the spread of the HIER seeds (`spread`); an interval over clips is blind to training variance."]
    for k, v in diffs.items():
        L.append(f"- {k}: {v['diff']:+.4f} {ci_s(v['ci95'])}")
    with open(out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def metrics_of(r):
    m, e = r["main"]["r_a"]["estimates"], r["main"]["change"]["arms"]
    return {"gap R2 (state)": m["gap|s"]["r2"], "presence AUROC (state)": m["pres|s"]["auc"], "gap R2 (state minus ego)": r["main"]["r_a"]["diffs_gap_r2"]["state minus ego"]["diff"],
            "presence AUROC (state minus ego)": r["main"]["r_a"]["diffs_presence_auc"]["state minus ego"]["diff"], "change skill H": e["H"]["skill"], "change skill N": e["N"]["skill"],
            "change skill E": e["E"]["skill"], "H minus E": r["main"]["change"]["diffs"]["H minus E"]["diff"], "H minus N": r["main"]["change"]["diffs"]["H minus N"]["diff"]}


def spread_of(runs):
    tags = [r["meta"]["tag"] for r in runs]
    ms = [metrics_of(r) for r in runs]
    out = {}
    for k in ms[0]:
        v = np.array([m[k] for m in ms], np.float64)
        out[k] = {"min": float(v.min()), "max": float(v.max()), "range": float(v.max() - v.min()), "std_ddof1": float(v.std(ddof=1)), "values": {t: float(x) for t, x in zip(tags, v)}}
    return out


def cmd_spread(a):
    runs = []
    for p in a.results:
        r, _ = _load_run(p)
        if not r.get("controls_pass") or "main" not in r:
            print(f"REFUSED: {p} did not pass its controls"); return 2
        runs.append(r)
    if len(runs) < 2:
        print("REFUSED: a spread needs at least 2 runs"); return 2
    sp = spread_of(runs)
    res = {"runs": [r["meta"]["tag"] for r in runs], "n_runs": len(runs), "spread": sp, "tier": TIER, "statement": STATEMENT, "in_sample": IN_SAMPLE,
           "note": "point values of the runs; the replicate floor of this read (rule H-ESTIM-SEED-1); sample std with ddof 1 over n runs"}
    out = f"{HERE}/SPREAD_S1_{a.name}"
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    L = [f"# SPREAD_S1_{a.name}: {len(runs)} runs ({', '.join(res['runs'])})", f"Tier {TIER}: {STATEMENT}. {IN_SAMPLE}. Min, max, range and sample std (ddof 1) of each point value over the runs. {FAMILIES}"]
    for k, v in sp.items():
        L.append(f"- {k}: min {v['min']:+.4f}, max {v['max']:+.4f}, range {v['range']:.4f}, std {v['std_ddof1']:.4f}")
    with open(out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("run", "smoke"):
        p = sub.add_parser(nm)
        p.add_argument("--states", required=True)
        p.add_argument("--expect-dim", type=int, required=True)
        p.add_argument("--expect-state", choices=("mean", "grid"), required=True)
        p.add_argument("--ckpt-sha256", default=None)
        if nm == "run":
            p.add_argument("--tag", required=True)
        else:
            p.add_argument("--clips", type=int, default=300)
            p.add_argument("--folds", type=int, default=1)
    p = sub.add_parser("pair"); p.add_argument("--a", required=True); p.add_argument("--b", required=True); p.add_argument("--B", type=int, default=B_BOOT)
    p = sub.add_parser("spread"); p.add_argument("--results", nargs="+", required=True); p.add_argument("--name", default="HIER")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "smoke":
        return cmd_run(a, smoke=True)
    if a.cmd == "pair":
        return cmd_pair(a)
    return cmd_spread(a)


if __name__ == "__main__":
    sys.exit(main())
