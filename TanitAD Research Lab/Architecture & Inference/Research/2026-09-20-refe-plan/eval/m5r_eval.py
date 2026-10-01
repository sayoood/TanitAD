#!/usr/bin/env python3
"""Measure 5r (eval/PREREG_MEASURE5R.md, blob a8241136): the eval on the REPAIRED truth, the Stage-1 readout, families.

  eval     A0 + M5's V3_s*/V4_s* (<m5>/ft, reused iff the key lists match -- gate) + v5's V3r_s*/V4r_s* (<m5r>/ft, trained
           by `m5_finetune_eval.py train --m5 <m5r> --arms V3,V4`: on the v5 data V3 = the pure REPAIRED set, V4 = the
           served v5 set when the sample carries at frac 0.5) on W3's 200 navtest tokens. The scorer's inputs are
           M5's exactly (the unrepaired 64 + the unrepaired 0.75x copies of the top-8 + STOP; the planner scores
           unrepaired proposals). The TRUTH is the repaired one: the 64 from the repaired E-6 table, the copies from
           eval/m5_repaired_extras.py's harness runs, STOP unchanged.
           G-R6: repair(each model-scored copy) -> to_navsim == the harness-scored repaired copy, bit for bit; harness
           complete; A0 reproduces the table logits (<= 1e-3) and picks; masked identity (<= 1e-4).
  readout  Stage 1 of §6, exactly as registered (V4r - V3 gating; E1 >= +0.03, CI lo > 0, > seed floor; E3a lo > -0.01;
           E3b lo > -2.0; E2 lo > -2.0; REFUTED iff E1 hi < 0), read ONLY when G-R1..G-R6 all pass. Reported: V3r - V3,
           V4r - V3r, V4r - V4, V3 - A0; EP + the NAVSIM sub-scores of the picks (same estimator); the selection mix.
  families families6.py on seams of each model's E3b / E2 pick, REPAIRED poses (what the planner would execute).

    python eval/m5r_eval.py eval [--device cuda]
    python eval/m5r_eval.py readout
    python eval/m5r_eval.py families
"""
from __future__ import annotations

import argparse
import copy as _copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))
import m5_finetune_eval as FE  # noqa: E402

M5 = "D:/Projects/TanitAD/data/refe_m5"
M5R = "D:/Projects/TanitAD/data/refe_m5r"
TR = f"{FE.NAV}/proptable/sub200_ep015_repaired/table.npz"
REP_CSV = f"{FE.NAV}/score/refe_sub200_ep015_f075rep_r{{:02d}}/refe_sub200_ep015_f075rep_r{{:02d}}.csv"
REP_SEAM = f"{FE.NAV}/seams/proptable/sub200_ep015_m5rep/refe_sub200_ep015_f075rep_r{{:02d}}.npz"
SEAM0 = f"{FE.NAV}/seams/refe_sub200_ep015.npz"
OUTD = os.path.join(HERE, "raw", "m5r")
GATE_FILES = {"G-R1": ("selftest_repair.json", "G_R1_PASS"), "G-R2": ("gate_gr2.json", "pass"),
              "G-R3": ("gate_gr34.json", ("G_R3", "pass")), "G-R4": ("gate_gr34.json", ("G_R4", "pass")),
              "G-R5": ("gate_gr5.json", "pass")}


def setup():
    import refe_navtest_seam as SEAM
    import slow_copies as SC
    from planner import repair_last_heading
    T, Tr, RK = np.load(FE.TABLE), np.load(TR), np.load(FE.RANKS)
    if [str(t) for t in T["token"]] != [str(t) for t in Tr["token"]] or not np.array_equal(T["logits"], Tr["logits"]):
        raise SystemExit("the repaired table is not the E-6 table's snapshot/tokens")
    C = FE.Cache("navtest_sub200")
    toks = C.keys()
    ti = {str(t): i for i, t in enumerate(T["token"])}
    props, top = C.array("props"), RK["top"]
    hr = [FE.read_csv(REP_CSV.format(r, r)) for r in range(FE.N_SRC)]
    stop = FE.read_csv(FE.STOP_CSV)
    seams = [np.load(REP_SEAM.format(r))["poses"] for r in range(FE.N_SRC)]
    X, H, SX, gate, missing = [], [], [], 0.0, 0
    for n, t in enumerate(toks):
        i = ti[t]
        ex = [SC.slow_copy(props[n, int(top[i, r])], FE.FACTOR) for r in range(FE.N_SRC)] + [np.zeros((20, 3), np.float32)]
        for r in range(FE.N_SRC):
            rep = SEAM.to_navsim(repair_last_heading(np.asarray(ex[r]))).astype(np.float32)
            gate = max(gate, float(np.abs(rep.astype(np.float64) - seams[r][i].astype(np.float64)).max()))
        X.append(np.stack(ex).astype(np.float32))
        h = [hr[r].get(t) for r in range(FE.N_SRC)] + [stop.get(t)]
        missing += sum(1 for x in h if x is None or not x[0])
        H.append([Tr["pdms"][i, j] for j in range(64)] + [x[1] if x and x[0] else np.nan for x in h])
        SX.append([list(Tr["sub"][i, j]) for j in range(64)] + [list(x[2]) if x and x[0] else [np.nan] * 6 for x in h])
    return {"C": C, "toks": toks, "tidx": np.array([ti[t] for t in toks]), "props": props, "X": np.stack(X),
            "pdms": np.asarray(H, np.float64), "sub": np.asarray(SX, np.float64), "T": T, "Tr": Tr,
            "gate_copy_repaired_max_abs": gate, "harness_missing": missing}


def models():
    base = FE.load_scorer()
    out = {"A0": base}

    def mk(p):
        m = FE.Scorer(*[_copy.deepcopy(x) for x in (base.score_q_mlp, base.score_dec, base.score_head)])
        m.load_state_dict(torch.load(p, map_location="cpu"))
        return m
    for arm in ("V3", "V4"):
        for p in sorted(glob.glob(f"{M5}/ft/{arm}_s*.pt")):
            out[os.path.basename(p)[:-3]] = mk(p)
        for p in sorted(glob.glob(f"{M5R}/ft/{arm}_s*.pt")):
            out[os.path.basename(p)[:-3].replace(arm, arm + "r")] = mk(p)
    return out


def eval_(a) -> int:
    device = torch.device(a.device)
    FE.vram_cap(device)
    FE.reserve_vram(device)
    torch.set_num_threads(8)
    k5 = [str(k) for k in np.load(f"{M5}/ft_data_train.npz")["key"]]
    kr = [str(k) for k in np.load(f"{M5R}/ft_data_train.npz")["key"]]
    E = setup()
    os.makedirs(f"{M5R}/evals", exist_ok=True)
    gates = {"copies_repaired_equal_harness_scored_max_abs": E["gate_copy_repaired_max_abs"],
             "harness_missing": E["harness_missing"], "M5_models_reusable_same_key_order": k5 == kr}
    np.savez(f"{M5R}/evals/_truth.npz", token=np.array(E["toks"]), pdms=E["pdms"], sub=E["sub"])
    for name, m in models().items():
        t0 = time.time()
        L = FE.eval_model(m, E, device)
        if name == "A0":
            tl = E["T"]["logits"][E["tidx"]].astype(np.float64)
            gates["A0_pure_vs_table_logits_max_abs"] = float(np.abs(L["pure"].astype(np.float64) - tl).max())
            gates["A0_pick_equals_table"] = int((FE.v1_agg(L["pure"]).argmax(1) == E["T"]["pick"][E["tidx"]]).sum())
        gates["masked_64_vs_pure_max_abs"] = max(gates.get("masked_64_vs_pure_max_abs", 0.0),
                                                 float(np.abs(L["masked"][:, :64] - L["pure"]).max()))
        mt = FE.metrics(L, E)
        np.savez(f"{M5R}/evals/{name}.npz", token=np.array(E["toks"]), **L, **mt)
        print(f"  {name}: E1 masked {np.nanmean(mt['E1_masked']):.4f} E3a {np.nanmean(mt['E3a']):.4f} "
              f"E3b {mt['E3b'].mean():.2f} E2 {np.nanmean(mt['E2']):.2f} ({time.time() - t0:.0f} s)", flush=True)
    gates["pass"] = (gates["copies_repaired_equal_harness_scored_max_abs"] == 0.0 and gates["harness_missing"] == 0
                     and gates["A0_pure_vs_table_logits_max_abs"] <= 1e-3 and gates["A0_pick_equals_table"] == 200
                     and gates["masked_64_vs_pure_max_abs"] <= 1e-4 and gates["M5_models_reusable_same_key_order"])
    json.dump(gates, open(f"{M5R}/evals/gates.json", "w"), indent=1)
    os.makedirs(OUTD, exist_ok=True)
    json.dump(gates, open(os.path.join(OUTD, "gate_gr6_eval.json"), "w"), indent=1)
    print(json.dumps(gates))
    print(f"ZZM5R_EVAL_{'OK' if gates['pass'] else 'GATE_FAIL'}")
    return 0 if gates["pass"] else 1


def gate_status() -> dict:
    out = {}
    for g, (f, k) in GATE_FILES.items():
        try:
            d = json.load(open(os.path.join(OUTD, f), encoding="utf-8"))
            out[g] = bool(d[k[0]][k[1]] if isinstance(k, tuple) else d[k])
        except (OSError, KeyError, ValueError):
            out[g] = False
    try:
        out["G-R6"] = bool(json.load(open(os.path.join(OUTD, "gate_gr6_eval.json")))["pass"])
    except (OSError, KeyError, ValueError):
        out["G-R6"] = False
    return out


def readout(a) -> int:
    ev = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in glob.glob(f"{M5R}/evals/*.npz")
          if not os.path.basename(p).startswith("_")}
    TRU = np.load(f"{M5R}/evals/_truth.npz")
    toks = [str(t) for t in ev["A0"]["token"]]
    draws, n_logs = FE.cluster_draws(toks)
    seeds = [0, 1, 2]

    def arm(nm, key):
        if nm == "A0":
            return ev["A0"][key]
        return np.nanmean(np.stack([ev[f"{nm}_s{s}"][key] for s in seeds]), 0)
    res = {"_label": "PRE-REGISTERED Stage-1 readout, eval/PREREG_MEASURE5R.md (blob a8241136), REPAIRED truth, W3's "
                     "200 tokens", "n_tokens": len(toks), "n_logs": n_logs, "seeds": seeds,
           "estimator": "paired log-cluster bootstrap over the logs, 10,000 resamples, 95 % percentile, seed 20260927",
           "variance_answered": "EPISODES for the CI; TRAINING (batch order) by the seed floor; inference deterministic",
           "estimator_selftest": FE.estimator_selftest(), "gates": gate_status(),
           "token_means": {nm: {k: round(float(np.nanmean(v[k])), 5) for k in ("E1_masked", "E1_plain", "E3a", "E3b", "E2")}
                           for nm, v in sorted(ev.items())}}
    pairs = {"V4r_minus_V3": ("V4r", "V3"), "V3r_minus_V3": ("V3r", "V3"), "V4r_minus_V3r": ("V4r", "V3r"),
             "V4r_minus_V4": ("V4r", "V4"), "V3_minus_A0": ("V3", "A0"), "V4r_minus_A0": ("V4r", "A0")}
    cmp = {}
    for key in ("E1_masked", "E1_plain", "E3a", "E3b", "E2"):
        for pn, (x, y) in pairs.items():
            try:
                d = arm(x, key) - arm(y, key)
            except KeyError:
                continue
            lo, hi = FE.boot_ci(d, draws)
            cmp.setdefault(key, {})[pn] = {"diff": round(float(np.nanmean(d)), 5), "ci95": [round(lo, 5), round(hi, 5)]}
    res["comparisons"] = cmp
    fl, floor = {}, 0.0
    for nm in ("V3", "V4r"):
        tm = [float(np.nanmean(ev[f"{nm}_s{s}"]["E1_masked"])) for s in seeds if f"{nm}_s{s}" in ev]
        pr = [abs(tm[i] - tm[j]) for i in range(len(tm)) for j in range(i + 1, len(tm))]
        fl[nm] = {"seed_token_means": [round(x, 5) for x in tm], "max_pair_abs": round(max(pr), 5) if pr else None}
        floor = max([floor] + pr)
    res["seed_floor_E1"] = {"floor": round(floor, 5), "by_arm": fl}
    e1, e3a, e3b, e2 = (cmp[k]["V4r_minus_V3"] for k in ("E1_masked", "E3a", "E3b", "E2"))
    passed = (e1["diff"] >= 0.03 and e1["ci95"][0] > 0 and e1["diff"] > floor and e3a["ci95"][0] > -0.01
              and e3b["ci95"][0] > -2.0 and e2["ci95"][0] > -2.0)
    refuted = e1["ci95"][1] < 0
    need = [f"{a_}_s{s}" for a_ in ("V3", "V4r") for s in seeds]
    absent = [n for n in need if n not in ev]
    gfail = [g for g, ok in res["gates"].items() if not ok] + ([] if res["estimator_selftest"]["identical"]
                                                               else ["estimator_selftest"])
    verdict = (("NO READOUT: gates failed " + ",".join(gfail)) if gfail else
               ("NO READOUT: models absent " + ",".join(absent)) if absent else
               ("STAGE-1 PASS" if passed else ("REFUTED" if refuted else "NOT PROVEN")))
    res["decision"] = {"verdict": verdict, "gates_failed": gfail, "models_absent": absent,
                       "rule": "STAGE-1 PASS iff E1 V4r-V3 >= +0.03, CI lo > 0, > seed floor; E3a lo > -0.01; E3b lo "
                               "> -2.0; E2 lo > -2.0. REFUTED iff E1 CI hi < 0. Otherwise NOT PROVEN. ADOPT needs "
                               "Stage 2 (A5's 923 fresh tokens) on the same bars.",
                       "E1": e1, "E3a": e3a, "E3b": e3b, "E2": e2, "seed_floor": round(floor, 5)}
    # EP and the NAVSIM sub-scores of the picks (x100), same estimator
    S = TRU["sub"]
    ar = np.arange(len(toks))
    subs = {}
    for nm in ("A0", "V3", "V4", "V3r", "V4r"):
        names = ["A0"] if nm == "A0" else [f"{nm}_s{s}" for s in seeds]
        if any(n not in ev for n in names):
            continue
        per = {}
        for route in ("E3b", "E2"):
            vals = []
            for n in names:
                if route == "E3b":
                    k = FE.v1_agg(ev[n]["pure"]).argmax(1)
                else:
                    k = np.nan_to_num(FE.v1_agg(ev[n]["masked"]), nan=-1).argmax(1)
                vals.append(100.0 * S[ar, k])
            per[route] = np.nanmean(np.stack(vals), 0)
        subs[nm] = per
    sub_pairs = {}
    for route in ("E3b", "E2"):
        for pn, (x, y) in pairs.items():
            if x in subs and y in subs:
                d = subs[x][route] - subs[y][route]
                sub_pairs.setdefault(route, {})[pn] = {h: dict(zip(("diff", "ci95"), (
                    round(float(np.nanmean(d[:, j])), 3), [round(v, 3) for v in FE.boot_ci(d[:, j], draws)])))
                    for j, h in enumerate(("NC", "DAC", "EP", "TTC", "C", "DDC"))}
    res["pick_subscores_x100"] = {"arm_means": {nm: {r: [round(float(v), 3) for v in np.nanmean(subs[nm][r], 0)]
                                                     for r in subs[nm]} for nm in subs},
                                  "order": ["NC", "DAC", "EP", "TTC", "C", "DDC"], "pairs": sub_pairs}
    mix = {}
    for n, e in ev.items():
        k = np.nan_to_num(FE.v1_agg(e["masked"]), nan=-1).argmax(1)
        mix[n] = {"original": int((k < 64).sum()), "copy_075": int(((k >= 64) & (k < 72)).sum()), "stop": int((k == 72).sum())}
    res["E2_selection_mix"] = mix
    os.makedirs(OUTD, exist_ok=True)
    json.dump(res, open(os.path.join(OUTD, "readout_stage1.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(res["decision"], indent=1))
    print(f"ZZM5R_STAGE1 {verdict}")
    return 0


def families(a) -> int:
    import eval_checkpoint as EC
    import stop_candidate_probe as SCP
    ev = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in glob.glob(f"{M5R}/evals/*.npz")
          if not os.path.basename(p).startswith("_")}
    Tr = np.load(TR)
    S0 = np.load(SEAM0)
    toks = [str(t) for t in ev["A0"]["token"]]
    s0i = {str(t): i for i, t in enumerate(S0["token"])}
    tri = {str(t): i for i, t in enumerate(Tr["token"])}
    si, ri = np.array([s0i[t] for t in toks]), np.array([tri[t] for t in toks])
    seams = [np.load(REP_SEAM.format(r))["poses"] for r in range(FE.N_SRC)]
    C8 = np.concatenate([Tr["proposals"][ri].astype(np.float32),
                         np.stack([seams[r][ri] for r in range(FE.N_SRC)], 1).astype(np.float32),
                         np.zeros((len(toks), 1, 8, 3), np.float32)], 1)          # [n, 73, 8, 3] REPAIRED poses
    fam_dir = f"{M5R}/families"
    os.makedirs(fam_dir, exist_ok=True)
    ar = np.arange(len(toks))
    out = {"instrument": os.path.join(EC.EV6, "families6.py"), "poses": "REPAIRED (the executed plans)", "blocks": {}}
    for name, e in sorted(ev.items()):
        for route, k in (("E3b", FE.v1_agg(e["pure"]).argmax(1)),
                         ("E2", np.nan_to_num(FE.v1_agg(e["masked"]), nan=-1).argmax(1))):
            tag = f"{name}_{route}"
            sp, op, lg = (os.path.join(fam_dir, f"seam_{tag}.npz"), os.path.join(fam_dir, f"families_{tag}.json"),
                          os.path.join(fam_dir, f"families_{tag}.log"))
            if os.path.exists(op):
                os.remove(op)
            poses = S0["poses"].copy()
            poses[si] = C8[ar, k]
            np.savez(sp, token=S0["token"], fingerprint=S0["fingerprint"], poses=poses.astype(np.float32),
                     sampling=S0["sampling"], arm=np.array(f"REFe_m5r_{tag}"))
            rc, _ = EC.run([EC.TANITAD_PY, "families6.py", "--seam", sp, "--inputs", EC.EXPORT, "--stage", "1",
                            "--label", f"REFe-m5r-{tag}", "--out", op, "--n-boot", "2000"],
                           EC.EV6, dict(os.environ, PYTHONIOENCODING="utf-8"), lg)
            out["blocks"][tag] = SCP.summarize_families(op) if os.path.exists(op) else {"status": f"FAILED rc={rc}"}
    json.dump(out, open(os.path.join(OUTD, "families_stage1.json"), "w", encoding="utf-8"), indent=1)
    print(f"ZZM5R_FAMILIES_OK {sum(1 for v in out['blocks'].values() if 'status' not in v)}/{len(out['blocks'])}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("eval", "readout", "families"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    return {"eval": eval_, "readout": readout, "families": families}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
