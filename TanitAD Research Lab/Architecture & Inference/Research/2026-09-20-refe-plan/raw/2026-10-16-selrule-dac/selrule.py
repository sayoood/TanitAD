#!/usr/bin/env python3
"""selrule.py -- the rules, the oracle controls, the fit and the confirm statistics of eval/PREREG_SELRULE_DAC.md (+ Amendment 1).

    python selrule.py fit --scores <scores_step0_base.npz> --pdm-tar <rlfan_pdm_val.tar.gz> --out FIT.json

THE CONTRACT IS THE PREREG (`PREREG_SELRULE_DAC.amended.md` in this package: sections 1-5 byte-identical to eval/PREREG_SELRULE_DAC.md).
Rules (selection only; the scorer is fixed):   B = argmax agg;   V(tau) = argmax agg over {p_DAC >= tau}, fall back to B when none qualifies;
K(k) = argmax of agg with its DAC and NC gate factors raised to the power k.   Oracles O-DAC / O-NC / O-both = B with the gate replaced by the
TRUE sub-score of the proposal (reported, never selected).   `agg` is rl_val_eval.select's navsim_v1 aggregate -- it is IMPORTED, never
re-implemented: every rule here edits the probability matrix and hands it to RV.select.

Pure functions (unit-tested in test_selrule.py): pick_B / pick_V / pick_K / pick_O, picks(), choose(), LogBoot, paired(), outcome().
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(os.path.dirname(HERE))                 # .../2026-09-20-refe-plan
sys.path.insert(0, os.path.join(PK, "refe"))
sys.path.insert(0, os.path.join(PK, "eval"))
import proxy_eval as PE  # noqa: E402  -- the estimator of RESULT_SFT4 / stage 3
import rl_reward as RR  # noqa: E402
import rl_val_eval as RV  # noqa: E402  -- loaders + the navsim_v1 aggregate

TOOL = "selrule/1"
B_BOOT, SEED = 10000, 20260927
TIE_TOL = 0.05                                    # PDMS x100 points (Amendment 1, A4)
COL = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3, "C": 4, "DDC": 5, "LANE": 6}
SUBS = ("NC", "DAC", "EP", "TTC", "C", "DDC")
GATE_COLS = (COL["NC"], COL["DAC"])               # the two gate factors of agg
ORACLE_COLS = {"DAC": (COL["DAC"],), "NC": (COL["NC"],), "both": (COL["NC"], COL["DAC"])}
GRID_TAU = (0.5, 0.6, 0.7, 0.8, 0.9)
GRID_K = (2, 3, 5, 8)
# registered order = "smaller change first, V before K" (A4).  B is NOT a candidate.
CANDIDATES = tuple([("V", t) for t in GRID_TAU] + [("K", k) for k in GRID_K])
SEL_CEILING_PTS = 3.5376954888978354              # D-REFE-ZEROTAIL-2 oracle-on-SEL gain, P2_MEASURED.json Q1_SEL_PROP.oracle_on_SEL_gain_pdms_pts.point
P2_BANKED_PDMS_X100, P2_BANKED_TOL = 88.82, 0.01  # P2's full-navtest score (zero_tail.EXPECT)


# ----------------------------------------------------------------------------------------------------------------- the rules
def name(spec):
    """('B',) -> 'B'; ('V', 0.5) -> 'V(0.5)'; ('K', 2) -> 'K(2)'; ('O', 'DAC') -> 'O-DAC'."""
    if spec[0] == "B":
        return "B"
    if spec[0] == "O":
        return f"O-{spec[1]}"
    return f"{spec[0]}({spec[1]:g})"


def _agg(p, lane):
    """The registered aggregate of ONE set: p [M, C] probabilities -> (argmax, selection score [M]).  RV.select, imported, not re-implemented."""
    return RV.select(p, "navsim_v1_lane" if lane else "navsim_v1", space="prob")


def pick_B(p, lane=False):
    return int(_agg(p, lane)[0])


def pick_V(p, tau, lane=False):
    """DAC veto: argmax agg over the proposals with p_DAC >= tau; if none qualifies, fall back to B."""
    b, sel = _agg(p, lane)
    ok = np.asarray(p)[:, COL["DAC"]] >= tau
    if not ok.any():
        return int(b)  # VETO-FALLBACK
    return int(np.argmax(np.where(ok, sel, -np.inf)))


def pick_K(p, k, lane=False):
    """Gate sharpening: agg with its DAC and NC gate factors raised to the power k (EP / TTC / C untouched)."""
    q = np.array(p, np.float64, copy=True)
    for c in GATE_COLS:
        q[:, c] = q[:, c] ** k  # K-GATES
    return int(_agg(q, lane)[0])


def pick_O(p, true_sub, which, lane=False):
    """Oracle control: B with the named gate(s) replaced by the TRUE sub-score of each proposal.  true_sub [M, >=6] in nc, dac, ep, ttc, c, ddc order."""
    q = np.array(p, np.float64, copy=True)
    ts = np.asarray(true_sub, np.float64)
    for c in ORACLE_COLS[which]:
        q[:, c] = ts[:, c]  # ORACLE-TRUE
    return int(_agg(q, lane)[0])


def picks(spec, P, tg=None, lane=False):
    """Picks [N] of one rule on a batch.  P [N, M, C] PROBABILITIES (sigmoid of the logits), tg [N, M, 6] true sub-scores (oracles only)."""
    out = np.empty(len(P), np.int64)
    for i in range(len(P)):
        if spec[0] == "B":
            out[i] = pick_B(P[i], lane)
        elif spec[0] == "V":
            out[i] = pick_V(P[i], spec[1], lane)
        elif spec[0] == "K":
            out[i] = pick_K(P[i], spec[1], lane)
        elif spec[0] == "O":
            out[i] = pick_O(P[i], tg[i], spec[1], lane)
        else:
            raise ValueError(f"unknown rule {spec!r}")
    return out


def choose(means, tol=TIE_TOL):
    """The registered choice: the highest mean PDMS (x100) over the 9 candidates; ties within `tol` go to the smaller change (the registered order,
    V before K).  -> (name, tied set in registered order)."""
    order = [name(c) for c in CANDIDATES]
    top = max(means[c] for c in order)
    tied = [c for c in order if top - means[c] <= tol + 1e-12]
    return tied[0], tied  # TIE-RULE


# ----------------------------------------------------------------------------------------------------------------- the estimator
class LogBoot:
    """proxy_eval.boot's paired log-cluster bootstrap, by cluster sums: the same draws (per-iteration rng.integers(0, G, G) from default_rng(seed)),
    the same group order (sorted log names), the same statistic (token-mean of the drawn logs' tokens), the same percentiles.  One instance = one
    set of draws, shared by every call, so every comparison on these tokens is paired by construction."""

    def __init__(self, logs, n=B_BOOT, seed=SEED):
        names, self.idx = np.unique(np.asarray([str(x) for x in logs]), return_inverse=True)
        self.names, self.G, self.n_boot = names, len(names), n
        rng = np.random.default_rng(seed)
        self.draws = np.stack([rng.integers(0, self.G, self.G) for _ in range(n)])
        self.cnt = np.bincount(self.idx, minlength=self.G).astype(np.float64)

    def ratio(self, num, den=None):
        """sum(num)/sum(den) over the drawn logs (den = 1 per token -> the token mean).  -> {'mean','ci95'}; a draw with sum(den) = 0 is dropped (counted)."""
        num = np.asarray(num, np.float64)
        s_n = np.bincount(self.idx, weights=num, minlength=self.G)
        s_d = self.cnt if den is None else np.bincount(self.idx, weights=np.asarray(den, np.float64), minlength=self.G)
        dall = s_d.sum()
        point = float(s_n.sum() / dall) if dall > 0 else float("nan")
        bd = s_d[self.draws].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(bd > 0, s_n[self.draws].sum(1) / bd, np.nan)
        ok = ~np.isnan(v)
        if not ok.any():
            return {"mean": point, "ci95": [float("nan"), float("nan")], "n_dropped_draws": int((~ok).sum())}
        return {"mean": point, "ci95": [float(np.percentile(v[ok], 2.5)), float(np.percentile(v[ok], 97.5))], "n_dropped_draws": int((~ok).sum())}


def paired(a, b, boot, scale=100.0):
    """mean(a - b) x scale with its paired log-cluster interval; better / worse / tied token counts.  adverse <=> the interval lies entirely below 0."""
    d = (np.asarray(a, np.float64) - np.asarray(b, np.float64)) * scale
    r = boot.ratio(d)
    r.update({"better": int((d > 0).sum()), "worse": int((d < 0).sum()), "tied": int((d == 0).sum()), "n": int(len(d)),
              "adverse": bool(r["ci95"][1] < 0), "separated": bool(r["ci95"][0] > 0 or r["ci95"][1] < 0)})
    return r


# ----------------------------------------------------------------------------------------------------------------- statistics at a pick
def at_pick(arr, pick):
    """arr [N, M] or [N, M, K] -> the value(s) at each set's pick."""
    return arr[np.arange(len(pick)), pick]


def summarise_pick(pdms, sub, pick):
    """Means (x100) of PDMS and the six sub-scores at the picks, and the zero / DAC = 0 / NC = 0 pick counts."""
    pp, ss = at_pick(pdms, pick), at_pick(sub, pick)
    out = {"PDMS_x100": float(100 * pp.mean()), "subscores_x100": {s: float(100 * ss[:, COL[s]].mean()) for s in SUBS},
           "zero_picks": int((pp == 0.0).sum()), "dac0_picks": int((ss[:, COL["DAC"]] == 0.0).sum()), "nc0_picks": int((ss[:, COL["NC"]] == 0.0).sum()), "n": int(len(pick))}
    return out


def selection_quality(pdms, pick):
    """TACTICAL leg (section 4): random / actual / best-of-64 on the same tokens, x100."""
    return {"random_x100": float(100 * pdms.mean()), "actual_x100": float(100 * at_pick(pdms, pick).mean()), "best_of_64_x100": float(100 * pdms.max(axis=1).mean())}


def compare(pdms, sub, pick_a, pick_b, boot):
    """Chosen/arm `a` against control `b` on identical tokens: DeltaPDMS and the six sub-score deltas (paired), and the zero-token movements."""
    pa, pb = at_pick(pdms, pick_a), at_pick(pdms, pick_b)
    sa, sb = at_pick(sub, pick_a), at_pick(sub, pick_b)
    res = {"PDMS": paired(pa, pb, boot), "subscores": {s: paired(sa[:, COL[s]], sb[:, COL[s]], boot) for s in SUBS}}
    res["adverse_subscores"] = sorted(s for s in SUBS if res["subscores"][s]["adverse"])
    res["picks_changed"] = int((pick_a != pick_b).sum())
    res["zero_tokens"] = {"b_zero": int((pb == 0).sum()), "a_zero": int((pa == 0).sum()), "rescued_b0_to_a_nonzero": int(((pb == 0) & (pa > 0)).sum()),
                          "new_zero_b_nonzero_to_a0": int(((pb > 0) & (pa == 0)).sum())}
    res["dac0"] = {"b": int((sb[:, COL["DAC"]] == 0).sum()), "a": int((sa[:, COL["DAC"]] == 0).sum())}
    res["nc0"] = {"b": int((sb[:, COL["NC"]] == 0).sum()), "a": int((sa[:, COL["NC"]] == 0).sum())}
    return res


def outcome(pdms_ci, adverse_sub, fam_adverse=None, fam_computed=True):
    """The committed outcome table (section 4 + Amendment 1 A7).  pdms_ci = the paired DeltaPDMS interval [lo, hi] (x100); adverse_sub = names of PDMS
    sub-scores whose paired interval lies entirely below 0; fam_adverse = names of longitudinal/lateral family components adverse (None when not computed)."""
    lo, hi = pdms_ci
    if hi < 0:
        return {"outcome": "REFUTED", "why": "upper bound of DeltaPDMS < 0", "adopt_if_subscore_leg_only": False}
    if not lo > 0:
        return {"outcome": "NOT PROVEN", "why": "the interval straddles 0 (lower bound <= 0 <= upper bound)", "adopt_if_subscore_leg_only": False}
    sub_only = not adverse_sub
    if adverse_sub:
        return {"outcome": "NOT ADOPTED (adverse component)", "why": f"lower bound > 0 but adverse PDMS sub-score(s): {', '.join(adverse_sub)}; no committed row: escalates (A7)",
                "adopt_if_subscore_leg_only": False}
    if not fam_computed or fam_adverse is None:
        return {"outcome": "UNRESOLVED (family leg not computed)", "why": "lower bound > 0 and no adverse PDMS sub-score, but the longitudinal/lateral family leg was not computed; never ADOPT",
                "adopt_if_subscore_leg_only": bool(sub_only)}
    if fam_adverse:
        return {"outcome": "NOT ADOPTED (adverse component)", "why": f"lower bound > 0, no adverse sub-score, but adverse family component(s): {', '.join(fam_adverse)}; no committed row: escalates (A7)",
                "adopt_if_subscore_leg_only": True}
    return {"outcome": "ADOPT", "why": "lower bound > 0, no PDMS sub-score adverse, no family component adverse", "adopt_if_subscore_leg_only": True}


# ----------------------------------------------------------------------------------------------------------------- the confirm
CMD_NAMES = {0: "left", 1: "straight", 2: "right", 3: "unknown"}


def confirm_compute(logs, cmd, pdms, sub, logits, chosen_spec, table_pick=None, n_boot=B_BOOT, seed=SEED):
    """The ONE confirm read on P2's table: B against the frozen rule on identical tokens, plus the oracle controls and the reported-never-gating blocks.
    logs [N] log name per token; cmd [N] int driving command (0 left, 1 straight, 2 right) or None; pdms [N, 64] the harness PDMS of every proposal;
    sub [N, 64, 6] its sub-scores (nc, dac, ep, ttc, c, ddc); logits [N, 64, 6] P2's own heads; chosen_spec = the rule frozen in FIT.json.
    -> a plain dict; contains NO decision (the caller adds the family leg, then outcome())."""
    P = RV.sigmoid(logits)
    boot = LogBoot(logs, n_boot, seed)
    B = picks(("B",), P)
    C = picks(tuple(chosen_spec), P)
    cmp_ = compare(pdms, sub, C, B, boot)
    res = {"n_tokens": int(len(B)), "n_logs": int(boot.G), "chosen_spec": list(chosen_spec),
           "B": summarise_pick(pdms, sub, B), "chosen": summarise_pick(pdms, sub, C), "compare": cmp_,
           "selection_quality": {"B": selection_quality(pdms, B), "chosen": selection_quality(pdms, C)},
           "ep_ttc_cost": {"EP": cmp_["subscores"]["EP"], "TTC": cmp_["subscores"]["TTC"],
                           "note": "the expected cost of re-weighting the gates (section 4: reported, never gating)"}}
    if table_pick is not None:
        tp = np.asarray(table_pick)
        res["B_vs_table_pick"] = {"equal": int((B == tp).sum()), "n": int(len(B)), "mean_PDMS_x100_at_table_pick": float(100 * at_pick(pdms, tp).mean())}
    ora = {}
    for w in ("DAC", "NC", "both"):
        pk = picks(("O", w), P, sub)
        d = paired(at_pick(pdms, pk), at_pick(pdms, B), boot)
        ora[f"O-{w}"] = {"summary": summarise_pick(pdms, sub, pk), "delta_vs_B": d, "picks_changed_vs_B": int((pk != B).sum()),
                         "share_of_SEL_ceiling_3.54": d["mean"] / SEL_CEILING_PTS, "ceiling_pts": SEL_CEILING_PTS}
    res["oracles"] = ora
    res["ceilings_x100"] = {"B": res["B"]["PDMS_x100"], "best_of_64": res["selection_quality"]["B"]["best_of_64_x100"],
                            "headroom_best_of_64_over_B": res["selection_quality"]["B"]["best_of_64_x100"] - res["B"]["PDMS_x100"]}
    if cmd is None:
        res["per_command"] = {"available": False, "reason": "no command field for these tokens: NOT COMPUTED"}
    else:
        cmd = np.asarray(cmd)
        dlt = (at_pick(pdms, C) - at_pick(pdms, B)) * 100.0
        pb, pc = at_pick(pdms, B), at_pick(pdms, C)
        out = {}
        for c, nm in CMD_NAMES.items():
            m = (cmd == c).astype(np.float64)
            if not m.any():
                continue
            out[nm] = {"tokens": int(m.sum()), "delta_PDMS_x100": boot.ratio(dlt * m, m), "B_PDMS_x100": boot.ratio(pb * m * 100.0, m), "chosen_PDMS_x100": boot.ratio(pc * m * 100.0, m),
                       "b_zero": int(((pb == 0) & (m > 0)).sum()), "chosen_zero": int(((pc == 0) & (m > 0)).sum())}
        res["per_command"] = {"available": True, "by_command": out, "tokens_covered": int(sum(v["tokens"] for v in out.values()))}
    res["picks"] = {"B_sha256": hashlib.sha256(B.astype("<i8").tobytes()).hexdigest(), "chosen_sha256": hashlib.sha256(C.astype("<i8").tobytes()).hexdigest()}
    return res, B, C


# ----------------------------------------------------------------------------------------------------------------- the fit
def sha256_file(p):
    return RV.sha256_file(p)


def load_rlval(scores_npz, pdm_dir):
    """-> keys, S [N, 64, C] logits, tg [N, 64, 6] true sub-scores, stats.  The join of rl_val_eval.run: same key AND same ckpt_step; no stale set is read."""
    sc = RV.load_scores(scores_npz)
    pdm, pst = RV.load_pdm(pdm_dir)
    keys = sorted(k for k in sc if k in pdm and pdm[k]["ckpt"] == sc[k]["ckpt"])
    if {sc[k]["space"] for k in keys} != {"logit"}:
        raise ValueError("the dump is not in logit space")
    S = np.stack([sc[k]["scores"] for k in keys])
    tg = np.stack([pdm[k]["tg"] for k in keys])
    st = {"scored_sets": len(sc), "label_sets": len(pdm), "joined": len(keys), "stale_or_missing": len(set(sc) | set(pdm)) - len(keys), "pdm_load": pst}
    return keys, S, tg, st


def labels_pdms(tg):
    s = RR.split_components(tg)
    return RR.pdms_v1(s["nc"], s["dac"], s["ep"], s["ttc"], s["c"])


def curves(P, tg, pdms, boot, lane):
    """B, the 9 candidates and the 3 oracles on RL-val: mean PDMS, delta vs B with its interval, sub-scores, zero counts, picks changed."""
    rules = [("B",)] + list(CANDIDATES) + [("O", w) for w in ("DAC", "NC", "both")]
    pk = {name(r): picks(r, P, tg, lane) for r in rules}
    out = {}
    for r in rules:
        n = name(r)
        row = summarise_pick(pdms, tg, pk[n])
        row["spec"] = list(r)
        row["picks_sha256"] = hashlib.sha256(pk[n].astype("<i8").tobytes()).hexdigest()
        if n != "B":
            cmp_ = compare(pdms, tg, pk[n], pk["B"], boot)
            row["delta_vs_B"] = {"mean": cmp_["PDMS"]["mean"], "ci95": cmp_["PDMS"]["ci95"], "better": cmp_["PDMS"]["better"], "worse": cmp_["PDMS"]["worse"]}
            row["picks_changed_vs_B"] = cmp_["picks_changed"]
            row["subscore_delta_vs_B_x100"] = {s: {"mean": cmp_["subscores"][s]["mean"], "ci95": cmp_["subscores"][s]["ci95"]} for s in SUBS}
        out[n] = row
    return out, pk


def identity_block(keys, S, pdms, pk_B, pk_Blane, pdm, sc_dict, banked_sft_step0):
    """B is rl_val_eval's own pick: the same set, the same rule, the same label join.  Hard gate."""
    n_mis = sum(int(RV.select(S[i], "navsim_v1")[0]) != int(pk_B[i]) for i in range(len(S)))
    n_mis_l = sum(int(RV.select(S[i], "navsim_v1_lane")[0]) != int(pk_Blane[i]) for i in range(len(S)))
    cfg = RR.RewardCfg(lam_d=0.0, w_prog=0.5, guard=True, use_k=False, use_l=False)
    res = {}
    for rule, pk in (("navsim_v1", pk_B), ("navsim_v1_lane", pk_Blane)):
        r = RV.run({"base": sc_dict}, pdm, None, None, rule, cfg, None, False, None, n_boot=200)
        mine = float(100 * at_pick(pdms, pk).mean())
        res[rule] = {"rl_val_eval_PDMS_x100": r["arms"]["base"]["PDMS_x100"], "mine_PDMS_x100": mine, "abs_diff": abs(r["arms"]["base"]["PDMS_x100"] - mine),
                     "rl_val_eval_zero_picks": r["arms"]["base"]["zero_picks"], "mine_zero_picks": int((at_pick(pdms, pk) == 0).sum()),
                     "rl_val_eval_sets_used": r["eligibility"]["sets_used"]}
    ok = (n_mis == 0 and n_mis_l == 0 and all(v["abs_diff"] < 1e-9 and v["rl_val_eval_zero_picks"] == v["mine_zero_picks"] and v["rl_val_eval_sets_used"] == len(keys) for v in res.values()))
    out = {"ok": bool(ok), "sets": len(keys), "B_pick_mismatches_vs_RV_select_navsim_v1": n_mis, "B_lane_pick_mismatches_vs_RV_select_navsim_v1_lane": n_mis_l, "rl_val_eval_run": res}
    if banked_sft_step0 is not None:
        out["banked_trainer_row_c0_sft_jsonl_step0_base_pick"] = {"value": banked_sft_step0, "mine_B_lane_x100": res["navsim_v1_lane"]["mine_PDMS_x100"],
                                                                  "abs_diff": abs(banked_sft_step0 - res["navsim_v1_lane"]["mine_PDMS_x100"]),
                                                                  "note": "the RL-S1 trainer's own number for the base scorer at step 0 (its rule is navsim_v1_lane); an INDEPENDENT code path"}
    return out


def extract_tar(tar_path, dest, expect_sha):
    got = sha256_file(tar_path)
    if got != expect_sha:
        raise SystemExit(f"{tar_path}: sha256 {got} != pinned {expect_sha}")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path) as t:
        names = t.getnames()
        t.extractall(dest, filter="data")
    return names


def fit(a):
    PIN = {"scores_step0_base.npz": "6332fcc6175805349dd84766d6e38eac79f3ab194588b541755c1cf6281fef63",
           "rlfan_pdm_val.tar.gz": "3dd6cba33cb1395f6c18b0204299a235384a160ea1456395f132de82fefbaae7"}
    sh_scores, sh_tar = sha256_file(a.scores), sha256_file(a.pdm_tar)
    if sh_scores != PIN["scores_step0_base.npz"]:
        raise SystemExit(f"scores sha256 {sh_scores} is not the pinned step0_base dump")
    names = extract_tar(a.pdm_tar, a.extract_dir, PIN["rlfan_pdm_val.tar.gz"])
    dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(a.extract_dir, "**", "pdm_*.jsonl"), recursive=True)})
    if len(dirs) != 1:
        raise SystemExit(f"expected exactly one directory of pdm_*.jsonl under {a.extract_dir}, found {dirs}")
    pdm_dir = dirs[0]
    keys, S, tg, st = load_rlval(a.scores, pdm_dir)
    if not (len(keys) == 3000 and len({k[0] for k in keys}) == 75 and S.shape == (3000, 64, 7) and tg.shape == (3000, 64, 6)):
        raise SystemExit(f"RL-val is not 3,000 sets / 75 logs / [64, 7]: {len(keys)} sets, S {S.shape}, tg {tg.shape}")
    pdm, _ = RV.load_pdm(pdm_dir)
    sc_dict = RV.load_scores(a.scores)
    pdms = labels_pdms(tg)
    P = RV.sigmoid(S)
    logs = [k[0] for k in keys]
    boot = LogBoot(logs, B_BOOT, SEED)
    banked = None
    if a.sft_jsonl and os.path.exists(a.sft_jsonl):
        for ln in open(a.sft_jsonl, encoding="utf-8"):
            d = json.loads(ln)
            if d.get("event") == "eval" and d.get("tag") == "step0":
                banked = float(d["base"]["pick"])
    # ---- PRIMARY: agg = navsim_v1 over columns 0-5 (A2).  The only family a choice is made from.
    cur, pk = curves(P, tg, pdms, boot, lane=False)
    # ---- SECONDARY (informational, never selected): the lane factor kept in agg (u400's own deployed rule)
    cur_l, pk_l = curves(P, tg, pdms, boot, lane=True)
    ident = identity_block(keys, S, pdms, pk["B"], pk_l["B"], pdm, sc_dict, banked)
    if not ident["ok"]:
        print("ZZSELRULE_IDENTITY_FAILED", json.dumps(ident)[:600], flush=True)
        raise SystemExit(7)
    means = {n: cur[n]["PDMS_x100"] for n in cur if n != "B" and cur[n]["spec"][0] in ("V", "K")}
    ch, tied = choose(means)
    means_l = {n: cur_l[n]["PDMS_x100"] for n in cur_l if n != "B" and cur_l[n]["spec"][0] in ("V", "K")}
    ch_l, tied_l = choose(means_l)
    best = float(100 * pdms.max(axis=1).mean())
    rand = float(100 * pdms.mean())
    b_pdms = cur["B"]["PDMS_x100"]
    oracles = {n: {**cur[n], "share_of_best_of_64_headroom_over_B": (cur[n]["PDMS_x100"] - b_pdms) / max(best - b_pdms, 1e-12)} for n in ("O-DAC", "O-NC", "O-both")}
    amended = os.path.join(HERE, "PREREG_SELRULE_DAC.amended.md")
    res = {
        "tool": TOOL, "frozen": True, "tier": "lookup of the TRUE NAVSIM v1.1 PDMS of the pick on RL-val sets (navtrain held-out, frozen pre-P2 fan); a FIT set, never an endpoint",
        "prereg": {"original_sha256": "48a6940f717e0e264930d788d4b33b3a71e8c063d2fc5957e17c5e6952b2c9e1", "original_path": "eval/PREREG_SELRULE_DAC.md @ origin 159c862",
                   "amended_copy": "PREREG_SELRULE_DAC.amended.md", "amended_sha256": sha256_file(amended), "selrule_py_sha256": sha256_file(os.path.join(HERE, "selrule.py"))},
        "scorer_identity": {"file": "Sayood/tanitad-refe-rls1-c0 final/scores_step0_base.npz", "sha256": sh_scores,
                            "is": "the untouched RL-S1 base scorer sx0 = SFT-4 arm B u400 (7 outputs) on the frozen PRE-P2 model_final fan; NOT P2 (6 outputs, sha ea31db22...)",
                            "warning": "the fit scorer is not the confirm scorer: transfer of tau / k is an ASSUMPTION (Amendment 1, A1)", "shape": list(S.shape)},
        "inputs": {"scores_sha256": sh_scores, "pdm_tar_sha256": sh_tar, "pdm_tar_members": len(names), "join": st},
        "rl_val": {"n_sets": len(keys), "n_logs": len(set(logs)), "logs": sorted(set(logs)), "ckpt_steps": sorted({int(v["ckpt"]) for v in pdm.values()})},
        "estimator": f"proxy_eval.boot's paired log-cluster bootstrap by cluster sums, {B_BOOT} resamples, 95 %, seed {SEED}, over {len(set(logs))} logs; it answers 'another draw of LOGS' only "
                     "(not another training run, not another inference run); u400 is one training run.",
        "identity": ident,
        "B": {**cur["B"], "agg": "navsim_v1 over columns 0-5 (P2's rule; the lane output ignored)"},
        "ceilings_x100": {"random": rand, "best_of_64": best, "B": b_pdms, "headroom_best_of_64_over_B": best - b_pdms},
        "curve": {n: cur[n] for n in cur if cur[n]["spec"][0] in ("V", "K")},
        "oracles": oracles,
        "chosen": {"name": ch, "spec": cur[ch]["spec"], "mean_PDMS_x100": cur[ch]["PDMS_x100"], "delta_vs_B": cur[ch]["delta_vs_B"], "picks_changed_vs_B": cur[ch]["picks_changed_vs_B"],
                   "subscore_delta_vs_B_x100": cur[ch]["subscore_delta_vs_B_x100"], "tied_set_in_registered_order": tied, "tie_tolerance_x100": TIE_TOL,
                   "registered_order": [name(c) for c in CANDIDATES], "beats_B_on_RL_val": bool(cur[ch]["PDMS_x100"] > b_pdms),
                   "delta_vs_B_separated_from_0": bool(cur[ch]["delta_vs_B"]["ci95"][0] > 0 or cur[ch]["delta_vs_B"]["ci95"][1] < 0)},
        "secondary_lane_aware_informational_only": {"agg": "navsim_v1_lane (u400's own deployed rule); NOT selected from, does not touch the frozen choice",
                                                    "B": cur_l["B"], "curve": {n: cur_l[n] for n in cur_l if cur_l[n]["spec"][0] in ("V", "K")},
                                                    "oracles": {n: cur_l[n] for n in ("O-DAC", "O-NC", "O-both")}, "would_have_chosen": ch_l, "tied_set": tied_l,
                                                    "agrees_with_primary_choice": bool(ch_l == ch)},
        "not_claimed": ["this is a FIT on the u400 scorer's heads; it says nothing about P2 on navtest until the one confirm is read",
                        "the interval does not measure training variance (u400 is one run) or inference variance (deterministic here)"],
    }
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    h = sha256_file(a.out)
    with open(a.out + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{h}  {os.path.basename(a.out)}\n")
    try:
        os.chmod(a.out, 0o444)
        os.chmod(a.out + ".sha256", 0o444)
    except OSError:
        pass
    print(f"ZZSELRULE_FIT_OK chosen {ch} (tied {tied}) B {b_pdms:.4f} chosen {cur[ch]['PDMS_x100']:.4f} sha256 {h}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--scores", default="C:/Users/Admin/rls1b_dl/c0/final/scores_step0_base.npz")
    f.add_argument("--pdm-tar", default="C:/Users/Admin/rls1b_dl/data/data/rlfan_pdm_val.tar.gz")
    f.add_argument("--extract-dir", required=True, help="a fresh, empty directory the label tarball is extracted into")
    f.add_argument("--sft-jsonl", default="C:/Users/Admin/rls1b_dl/c0/final/sft.jsonl")
    f.add_argument("--out", default=os.path.join(HERE, "FIT.json"))
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        if os.path.exists(a.out):
            raise SystemExit(f"{a.out} exists: FIT.json is write-once (the choice is frozen)")
        return fit(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
