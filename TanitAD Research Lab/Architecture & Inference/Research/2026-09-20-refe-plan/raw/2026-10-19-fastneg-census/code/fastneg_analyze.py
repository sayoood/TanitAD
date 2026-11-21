"""FASTNEG-CENSUS analysis: (a) support, (b) resolution, (c) refit learning curve, (c0) the powered null, the registered rule, the descriptive controls.  Reads the OOF tables the fit stage wrote
and the inputs; it can launch no fit, so it is NOT bound by the gate token (editing it does not invalidate the fits).  REGISTERED SOURCE: PREREG_FASTNEG_CENSUS.md + AMENDMENT1 (A1.3, A1.5, A1.6, A1.7)."""
from __future__ import annotations

import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastneg_lib as L                                                               # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# stage: analyze
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
WITHIN = ("S", "Q1", "Q2", "Q3", "Q4", "Q123")
CROSS = ("xQ4", "xQ123")


class Analyzer:
    def __init__(self, ctx: dict):
        self.c = ctx
        self.draws = L.boot_draws(ctx["n_logs"])
        nc, dac = ctx["labels6"][..., 0], ctx["labels6"][..., 1]
        self.lab = {"nc": nc, "dac": dac}
        self.strict = {r: (self.lab[r] == 0.0, self.lab[r] == 1.0) for r in L.ROWS}          # A1.3: unsafe = label == 0, safe = label == 1
        self.loose = {r: (self.lab[r] < 1.0, self.lab[r] == 1.0) for r in L.ROWS}           # descriptive sensitivity: unsafe = label < 1
        self.cache: dict = {}

    def stats(self, key: str, z: np.ndarray, row: str, names, loose: bool = False) -> dict:
        """{group: (num_log, den_log)} for one (source, row)."""
        ck = (key, row, loose)
        have = self.cache.setdefault(ck, {})
        need = [n for n in names if n not in have]
        if need:
            un, sf = (self.loose if loose else self.strict)[row]
            ps = L.pair_stats(z, un, sf, self.c["code"], names=set(need))
            for n in need:
                have[n] = (L.per_log(ps[n][0], self.c["log_idx"], self.c["n_logs"]), L.per_log(ps[n][1], self.c["log_idx"], self.c["n_logs"]))
        return {n: have[n] for n in names}

    def auc(self, st: dict, g: str) -> dict:
        return L.ratio_ci(st[g][0], st[g][1], self.draws)

    def diff(self, sa: dict, ga: str, sb: dict, gb: str) -> dict:
        return L.diff_ci(sa[ga][0], sa[ga][1], sb[gb][0], sb[gb][1], self.draws)


def analyze_support(an: Analyzer, strat_names=L.STRATA_NAMES) -> dict:
    c = an.c
    code, li, nl = c["code"], c["log_idx"], c["n_logs"]
    nc, dac = an.lab["nc"], an.lab["dac"]
    unsafe_any = (nc == 0.0) | (dac == 0.0)
    tot_unsafe = L.per_log(unsafe_any.sum(1).astype(np.float64), li, nl)
    out = {}
    for k, name in enumerate(strat_names):
        m = code == k
        rows = L.per_log(m.sum(1).astype(np.float64), li, nl)
        cnt = lambda x: L.per_log((x & m).sum(1).astype(np.float64), li, nl)                 # noqa: E731
        ua, n0, n05, d0 = cnt(unsafe_any), cnt(nc == 0.0), cnt(nc == 0.5), cnt(dac == 0.0)
        out[name] = {"rows": int(rows.sum()), "nc_eq_0": int(n0.sum()), "nc_eq_0.5": int(n05.sum()), "dac_eq_0": int(d0.sum()), "unsafe_any_nc0_or_dac0": int(ua.sum()),
                     "unsafe_share_of_rows": L.ratio_ci(ua, rows, an.draws), "nc0_share_of_rows": L.ratio_ci(n0, rows, an.draws), "dac0_share_of_rows": L.ratio_ci(d0, rows, an.draws),
                     "share_of_all_unsafe_rows": L.ratio_ci(ua, tot_unsafe, an.draws), "share_of_all_rows": L.ratio_ci(rows, L.per_log(np.full(len(code), float(code.shape[1])), li, nl), an.draws)}
    return out


def pack(st: dict, an: Analyzer, groups) -> dict:
    return {g: an.auc(st, g) for g in groups}


def analyze_all(a, S: dict, P: dict, ctx: dict) -> dict:
    an = Analyzer(ctx)
    res: dict = {"schema": L.SCHEMA, "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "strata_json_sha256": S["sha"], "strata": {
        "cut_points": S["doc"]["cut_points"], "rows_per_stratum": S["doc"]["rows_per_stratum"], "n_S_sets": S["doc"]["n_S_sets"], "n_sets": S["doc"]["n_sets"]}}
    # ---- (a)
    log("(a) label support")
    res["a_support"] = analyze_support(an)
    # ---- fit tables
    fits = {}
    for fn in sorted(os.listdir(P["fits"])) if os.path.isdir(P["fits"]) else []:
        if fn.endswith(".json"):
            tid = fn[:-5]
            if os.path.exists(f"{P['fits']}/{tid}.npy"):
                fits[tid] = np.load(f"{P['fits']}/{tid}.npy", mmap_mode="r")
    res["fit_tables_present"] = sorted(fits)
    # ---- (b)(i) P2's frozen rows
    log("(b)(i) P2 frozen rows")
    res["b_i_p2"] = {}
    for row in L.ROWS:
        z = ctx["z6"][..., L.ROW_COL[row]]
        st = an.stats("P2", z, row, WITHIN + CROSS)
        res["b_i_p2"][row] = {"auc": pack(st, an, WITHIN + CROSS), "F_minus_Q123": an.diff(st, "Q4", st, "Q123")}
        stl = an.stats("P2", z, row, ("Q4", "Q123", "S"), loose=True)
        res["b_i_p2"][row]["sensitivity_unsafe_lt_1"] = {"auc": pack(stl, an, ("Q4", "Q123", "S")), "F_minus_Q123": an.diff(stl, "Q4", stl, "Q123")}
    # ---- (b)(ii) HEADROW-PROBE's OOF tables
    res["b_ii_probe"] = {"status": "MISSING"}
    probe = None
    if os.path.exists(a.probe_oof):
        pz = np.load(a.probe_oof, allow_pickle=False)
        keys_ok = (np.array_equal(pz["token"].astype("U"), ctx["key_token"]) and np.array_equal(pz["log"].astype("U"), ctx["key_log"]) and np.array_equal(pz["step"].astype(np.int64), ctx["key_step"])
                   and np.array_equal(pz["rank"].astype(np.int64), ctx["key_rank"]) and np.array_equal(pz["ckpt_step"].astype(np.int64), ctx["key_ckpt_step"]))
        if not keys_ok:
            raise L.CensusError("HEADROW-PROBE's OOF file does not list the same sets in the same order as the bank: (b)(ii) STOPS")
        probe = {k: pz[k] for k in pz.files if k.startswith("oof_nc_") or k.startswith("oof_dac_")}
        res["b_ii_probe"] = {"status": "PRESENT", "file_sha256": L.sha256_file(a.probe_oof), "keys_equal_bank": True, "arms": {}}
        pm = {}
        for k in ("mean_speed_mps", "p2_pick_mean_speed_mps", "p2_pick_index", "labels6", "p2_logits6", "fold"):
            if k in pz.files:
                pm[k] = pz[k]
        res["b_ii_probe"]["cross_checks"] = {
            "mean_speed_max_abs_diff_vs_ours": float(np.abs(pm["mean_speed_mps"].astype(np.float64) - ctx["v"]).max()) if "mean_speed_mps" in pm else None,
            "pick_mean_speed_max_abs_diff_vs_ours": float(np.abs(pm["p2_pick_mean_speed_mps"].astype(np.float64) - ctx["v_b"]).max()) if "p2_pick_mean_speed_mps" in pm else None,
            "pick_index_equal": bool(np.array_equal(pm["p2_pick_index"].astype(np.int64), ctx["pick"])) if "p2_pick_index" in pm else None,
            "labels6_equal_bank": bool(np.array_equal(pm["labels6"], ctx["labels6"])) if "labels6" in pm else None,
            "p2_logits6_equal_bank": bool(np.array_equal(pm["p2_logits6"], ctx["z6"])) if "p2_logits6" in pm else None,
            "fold_equal_ours": bool(np.array_equal(pm["fold"].astype(np.int64), ctx["fold"])) if "fold" in pm else None}
        for arm in ("L", "M"):
            for s in L.SEEDS:
                for row in L.ROWS:
                    z = probe[f"oof_{row}_{arm}_s{s}"]
                    st = an.stats(f"probe_{arm}_s{s}", z, row, WITHIN + CROSS)
                    res["b_ii_probe"]["arms"].setdefault(f"{arm}_s{s}", {})[row] = {"auc": pack(st, an, WITHIN + CROSS), "F_minus_Q123": an.diff(st, "Q4", st, "Q123")}
    # ---- (c) our L refits
    res["c_refit"] = {"status": "MISSING"}
    have_primary = all(f"L_f{sz}_s{s}" in fits for sz in L.SIZES for s in L.SEEDS)
    if have_primary:
        log("(c) refits")
        res["c_refit"] = {"status": "PRESENT", "rows": {}}
        for row_i, row in enumerate(L.ROWS):
            res["c_refit"]["rows"][row] = {}
            for s in L.SEEDS:
                d = {}
                for sz in L.SIZES:
                    names = WITHIN + CROSS if sz == 100 else ("Q4", "Q123")
                    st = an.stats(f"L_f{sz}_s{s}", np.asarray(fits[f"L_f{sz}_s{s}"][row_i]), row, names)
                    d[f"f{sz}"] = {"auc": pack(st, an, names)}
                    d[f"f{sz}"]["_st"] = st
                st100, st25, st50 = d["f100"].pop("_st"), d["f25"].pop("_st"), d["f50"].pop("_st")
                d["F_minus_Q123_at_100"] = an.diff(st100, "Q4", st100, "Q123")
                d["slope_F_100_minus_25"] = an.diff(st100, "Q4", st25, "Q4")
                d["slope_F_100_minus_50"] = an.diff(st100, "Q4", st50, "Q4")
                d["slope_F_50_minus_25"] = an.diff(st50, "Q4", st25, "Q4")
                d["slope_Q123_100_minus_25"] = an.diff(st100, "Q123", st25, "Q123")
                stl = an.stats(f"L_f100_s{s}", np.asarray(fits[f"L_f100_s{s}"][row_i]), row, ("Q4", "Q123"), loose=True)
                d["sensitivity_unsafe_lt_1_F_minus_Q123_at_100"] = an.diff(stl, "Q4", stl, "Q123")
                d["deficit_call"] = L.deficit_call(d["F_minus_Q123_at_100"])
                d["slope_call"] = L.slope_call(d["slope_F_100_minus_25"])
                res["c_refit"]["rows"][row][f"s{s}"] = d
    # ---- the rule
    res["rule"] = {"status": "NOT_RUN"}
    if have_primary:
        classes, void_rows = {}, {}
        for row in L.ROWS:
            dd = [res["c_refit"]["rows"][row][f"s{s}"]["deficit_call"] for s in L.SEEDS]
            ss = [res["c_refit"]["rows"][row][f"s{s}"]["slope_call"] for s in L.SEEDS]
            classes[row] = {"deficit_by_seed": dd, "slope_by_seed": ss, "class": L.row_class(dd, ss)}
        res["rule"] = {"status": "RAN", "source": "our (c) 100 % L refit", "rows": classes}
        # robustness: the rule on HEADROW-PROBE's L OOF (deficit from the probe's L, slope from our refits)
        if probe is not None:
            rb = {}
            for row in L.ROWS:
                dd = [L.deficit_call(res["b_ii_probe"]["arms"][f"L_s{s}"][row]["F_minus_Q123"]) for s in L.SEEDS]
                ss = classes[row]["slope_by_seed"]
                rb[row] = {"deficit_by_seed": dd, "slope_by_seed": ss, "class": L.row_class(dd, ss)}
            res["rule"]["robustness_probe_L"] = rb
    # ---- (c0) the powered null
    res["c0_null"] = {"status": "MISSING"}
    have_null = all(f"null_r{r}{sd}_s{s}" in fits for r in range(1, L.NULL_REPS + 1) for sd in "AB" for s in L.SEEDS)
    if have_null:
        log("(c0) null")
        res["c0_null"] = {"status": "PRESENT", "rows": {}}
        fires_by_row = {}
        for row_i, row in enumerate(L.ROWS):
            reps, fires = [], []
            for r in range(1, L.NULL_REPS + 1):
                cis = []
                for s in L.SEEDS:
                    sa = an.stats(f"null_r{r}A_s{s}", np.asarray(fits[f"null_r{r}A_s{s}"][row_i]), row, ("Q4",))
                    sb = an.stats(f"null_r{r}B_s{s}", np.asarray(fits[f"null_r{r}B_s{s}"][row_i]), row, ("Q4",))
                    cis.append(an.diff(sa, "Q4", sb, "Q4"))
                fire = L.null_fires(cis)
                fires.append(bool(fire))
                reps.append({"rep": r, "ci_A_minus_B_by_seed": {f"s{s}": cis[i] for i, s in enumerate(L.SEEDS)}, "fires": bool(fire)})
            res["c0_null"]["rows"][row] = {"replicates": reps, "n_fire": int(sum(fires)), "fire_rate": sum(fires) / len(fires),
                                           "max_abs_point": float(max(abs(ci["point"]) for rp in reps for ci in rp["ci_A_minus_B_by_seed"].values()))}
            fires_by_row[row] = fires
        res["c0_null"]["void"] = bool(L.null_void(fires_by_row))
    # ---- the outcome
    if have_primary:
        void = bool(res["c0_null"].get("void", False)) if have_null else False
        out = L.census_outcome(res["rule"]["rows"]["nc"]["class"], res["rule"]["rows"]["dac"]["class"], void)
        out["null_status"] = "PRESENT" if have_null else "MISSING (outcome provisional)"
        res["outcome"] = out
        if probe is not None:
            o2 = L.census_outcome(res["rule"]["robustness_probe_L"]["nc"]["class"], res["rule"]["robustness_probe_L"]["dac"]["class"], void)
            res["outcome_robustness_probe_L"] = o2
            res["outcome_agrees_with_probe_L"] = bool(o2["outcome"] == out["outcome"])
    # ---- descriptive controls with no authority (placebo, cold)
    res["d_placebo"] = {"status": "MISSING"}
    if have_primary and all(f"placebo_q2_25_s{s}" in fits for s in L.SEEDS):
        res["d_placebo"] = {"status": "PRESENT", "rows": {}}
        for row_i, row in enumerate(L.ROWS):
            res["d_placebo"]["rows"][row] = {}
            for s in L.SEEDS:
                sp = an.stats(f"placebo_q2_25_s{s}", np.asarray(fits[f"placebo_q2_25_s{s}"][row_i]), row, ("Q4", "Q123"))
                s100 = an.stats(f"L_f100_s{s}", np.asarray(fits[f"L_f100_s{s}"][row_i]), row, ("Q4", "Q123"))
                s25 = an.stats(f"L_f25_s{s}", np.asarray(fits[f"L_f25_s{s}"][row_i]), row, ("Q4", "Q123"))
                res["d_placebo"]["rows"][row][f"s{s}"] = {"F_auc_placebo": an.auc(sp, "Q4"), "F_100_minus_placebo": an.diff(s100, "Q4", sp, "Q4"), "F_100_minus_F25_subsample": an.diff(s100, "Q4", s25, "Q4")}
    res["d_cold"] = {"status": "MISSING"}
    if all(f"cold_f{sz}_s{s}" in fits for sz in L.SIZES for s in L.SEEDS):
        log("cold")
        res["d_cold"] = {"status": "PRESENT", "rows": {}}
        for row_i, row in enumerate(L.ROWS):
            res["d_cold"]["rows"][row] = {}
            for s in L.SEEDS:
                st = {sz: an.stats(f"cold_f{sz}_s{s}", np.asarray(fits[f"cold_f{sz}_s{s}"][row_i]), row, ("Q4", "Q123")) for sz in L.SIZES}
                d = {f"f{sz}": pack(st[sz], an, ("Q4", "Q123")) for sz in L.SIZES}
                d["F_minus_Q123_at_100"] = an.diff(st[100], "Q4", st[100], "Q123")
                d["slope_F_100_minus_25"] = an.diff(st[100], "Q4", st[25], "Q4")
                d["would_read_deficit"] = L.deficit_call(d["F_minus_Q123_at_100"])
                d["would_read_slope"] = L.slope_call(d["slope_F_100_minus_25"])
                res["d_cold"]["rows"][row][f"s{s}"] = d
    # ---- controls
    res["controls"] = {}
    if have_primary and probe is not None:
        rp = {}
        for row_i, row in enumerate(L.ROWS):
            for s in L.SEEDS:
                mine = np.asarray(fits[f"L_f100_s{s}"][row_i])
                theirs = probe[f"oof_{row}_L_s{s}"]
                d = np.abs(mine.astype(np.float64) - theirs.astype(np.float64))
                rp[f"{row}_s{s}"] = {"max_abs_logit_diff": float(d.max()), "mean_abs_logit_diff": float(d.mean()), "fraction_bit_identical": float((mine == theirs).mean())}
        res["controls"]["C_REPRO_our_L100_vs_probe_L"] = rp
    rng = np.random.default_rng(20261019)
    cn = {}
    for row in L.ROWS:                                                                    # instrument controls on the real labels and strata: random scores -> 0.5, oracle scores -> 1.0
        zr = rng.normal(size=ctx["code"].shape).astype(np.float32)
        zo = (an.lab[row] + 0.001 * rng.random(ctx["code"].shape)).astype(np.float32)
        sr, so = an.stats("CTRL_random", zr, row, ("Q4", "Q123")), an.stats("CTRL_oracle", zo, row, ("Q4", "Q123"))
        cn[row] = {"random_scores_F": an.auc(sr, "Q4"), "random_scores_Q123": an.auc(sr, "Q123"), "oracle_scores_F": an.auc(so, "Q4")["point"], "oracle_scores_Q123": an.auc(so, "Q123")["point"]}
    res["controls"]["C_AUC_instrument"] = cn
    return res


def write_analysis(a, S: dict, P: dict, ctx: dict) -> int:
    os.makedirs(P["results"], exist_ok=True)
    res = analyze_all(a, S, P, ctx)
    data = L.canon_json(res)
    with open(f"{P['results']}/census.json", "wb") as f:
        f.write(data)
    log(f"analysis written: {P['results']}/census.json sha256 {L.sha256_bytes(data)}")
    if "outcome" in res:
        log(f"OUTCOME {res['outcome']['outcome']}  nc={res['outcome']['nc']} dac={res['outcome']['dac']} void={res['outcome']['void']}")
    return 0


