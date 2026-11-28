#!/usr/bin/env python3
"""Measure 5r STAGE 2 (eval/PREREG_MEASURE5R.md §4/§6, blob a8241136) -- the confirmation on FRESH tokens.
⛔ RUNS ONLY AFTER A STAGE-1 PASS (eval/raw/m5r/readout_stage1.json verdict "STAGE-1 PASS"); every stage refuses otherwise.

Tokens: Amendment 5's 923 (43 logs, disjoint from W3's 200; eval/raw/a5_confirm/a5_confirm_tokens.json).
  gate     ADMISSIBILITY of M6's eval cache (refe_proxy/cache_eval_ep015: the planner's own input path, fp32) as the
           scorer context: its NAVSIM-grid proposals (to_navsim of the native 20-pose traj) equal
           proptable/a7confirm_ep015/proposals.npz bit for bit, and A0's logits re-computed on its visual_ctx reproduce
           that file's logits (<= 1e-3) and picks. Refuses to build anything on failure.
  seams    the REPAIRED truth's inputs: 64 per-slot seams (repair_last_heading(traj[:, k]) -> to_navsim), 8 copies of
           snapshot 015's own top-8 (the planner's v1 aggregate of those logits) at 0.75x -> repair -> to_navsim, and
           STOP (zeros; the repair is the identity).
  score    the 73 seams through the UNCHANGED harness (score_navtest_refe.py), --workers N (RAM rule applies).
  eval     A0 + V3_s* + V4r_s* (+ V3r_s*, V4_s*) on the 923 contexts; G-R6 on this truth.
  readout  §6 stage 2: ADOPT iff E1 V4r - V3 >= +0.03, CI lo > 0, > seed floor; E3a lo > -0.01; E3b lo > -2.0; E2 lo >
           -2.0 -- paired log-cluster bootstrap over the 43 logs; REFUTED iff E1 hi < 0; otherwise NOT PROVEN.

    python eval/m5r_stage2.py gate | seams | score [--workers 2] | eval [--device cuda] | readout
"""
from __future__ import annotations

import argparse
import copy as _copy
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))
import m5_finetune_eval as FE  # noqa: E402
import m5r_eval as R  # noqa: E402

TOK = os.path.join(HERE, "raw", "a5_confirm", "a5_confirm_tokens.json")
M6C = "D:/Projects/TanitAD/data/refe_proxy/cache_eval_ep015"
A7P = f"{FE.NAV}/proptable/a7confirm_ep015/proposals.npz"
SEAM0 = f"{FE.NAV}/seams/refe_a7confirm_ep015.npz"
SEAMD = f"{FE.NAV}/seams/proptable/a5_ep015_m5r2"
S2 = "D:/Projects/TanitAD/data/refe_m5r/stage2"
OUTD = R.OUTD
NAME = "a5_ep015_m5r2"


def require_stage1_pass():
    try:
        v = json.load(open(os.path.join(OUTD, "readout_stage1.json"), encoding="utf-8"))["decision"]["verdict"]
    except (OSError, KeyError, ValueError):
        v = "absent"
    if v != "STAGE-1 PASS":
        raise SystemExit(f"REFUSED: Stage 2 runs only after a STAGE-1 PASS (Stage 1 reads: {v!r})")


class M6Ctx:
    """(log|token) -> visual_ctx [7680, 256] fp32 from M6's shard .bin files (memmapped)."""

    def __init__(self):
        self.man = json.load(open(f"{M6C}/manifest.json", encoding="utf-8"))
        self.where, self.mm = {}, {}
        for line in open(f"{M6C}/frames.jsonl", encoding="utf-8"):
            r = json.loads(line)
            self.where[r["key"][1]] = (r["shard"], r["index"])
        self.fwd = {}

    def _vis(self, sh):
        if sh not in self.mm:
            p = os.path.join(M6C, self.man["shards"][sh]["visual"])
            n = os.path.getsize(p) // (7680 * 256 * 4)
            self.mm[sh] = np.memmap(p, dtype=np.float32, mode="r", shape=(n, 7680, 256))
        return self.mm[sh]

    def _fwd(self, sh):
        if sh not in self.fwd:
            self.fwd[sh] = dict(np.load(os.path.join(M6C, f"shard_{sh:04d}.fwd.npz")))
        return self.fwd[sh]

    def traj(self, tok):
        sh, i = self.where[tok]
        return self._fwd(sh)["traj"][i]

    def logits(self, tok):
        sh, i = self.where[tok]
        return self._fwd(sh)["logits"][i]

    def ctx(self, toks, device, dtype):
        import torch
        return torch.stack([torch.from_numpy(np.array(self._vis(self.where[t][0])[self.where[t][1]], copy=True))
                            for t in toks]).to(device=device, dtype=dtype)


def tokens():
    d = json.load(open(TOK, encoding="utf-8"))
    return [str(t) for t in d["tokens"]], d["token_log"]


def gate(_a) -> int:
    require_stage1_pass()
    import refe_navtest_seam as SEAM
    toks, _ = tokens()
    P = np.load(A7P)
    pi = {str(t): i for i, t in enumerate(P["token"])}
    C = M6Ctx()
    base = FE.load_scorer()
    import torch
    d_prop = d_log = 0.0
    picks_equal = 0
    for b0 in range(0, len(toks), 16):
        ks = toks[b0:b0 + 16]
        tr = np.stack([C.traj(t) for t in ks])
        nav = np.stack([[SEAM.to_navsim(x).astype(np.float32) for x in tr[j]] for j in range(len(ks))])
        ref = np.stack([P["proposals"][pi[t]] for t in ks])
        d_prop = max(d_prop, float(np.abs(nav.astype(np.float64) - ref.astype(np.float64)).max()))
        with torch.no_grad():
            lg = base(torch.from_numpy(tr).float(), C.ctx(ks, "cpu", torch.float32)).numpy()
        refl = np.stack([P["logits"][pi[t]] for t in ks])
        d_log = max(d_log, float(np.abs(lg.astype(np.float64) - refl.astype(np.float64)).max()))
        picks_equal += int((FE.v1_agg(lg).argmax(1) == np.array([P["pick"][pi[t]] for t in ks])).sum())
    res = {"proposals_vs_a7confirm_max_abs": d_prop, "A0_logits_vs_a7confirm_max_abs": d_log,
           "A0_picks_equal": picks_equal, "n": len(toks)}
    res["pass"] = d_prop == 0.0 and d_log <= 1e-3 and picks_equal == len(toks)
    json.dump(res, open(os.path.join(OUTD, "stage2_gate_context.json"), "w"), indent=1)
    print(json.dumps(res))
    return 0 if res["pass"] else 1


def seam_label(k):
    return f"refe_{NAME}_{k}"


def seams(_a) -> int:
    require_stage1_pass()
    if not json.load(open(os.path.join(OUTD, "stage2_gate_context.json")))["pass"]:
        raise SystemExit("REFUSED: the context admissibility gate did not pass")
    import refe_navtest_seam as SEAM
    import slow_copies as SC
    from planner import repair_last_heading
    toks, _ = tokens()
    S0 = np.load(SEAM0)
    si = {str(t): i for i, t in enumerate(S0["token"])}
    C = M6Ctx()
    P = np.load(A7P)
    pi = {str(t): i for i, t in enumerate(P["token"])}
    os.makedirs(SEAMD, exist_ok=True)
    n = len(S0["token"])
    rows = [si[t] for t in toks]
    tr = np.stack([C.traj(t) for t in toks])                                           # [n, 64, 20, 3]
    top = np.argsort(-np.stack([FE.v1_agg(P["logits"][pi[t]]) for t in toks]), axis=1, kind="stable")[:, :8]
    os.makedirs(S2, exist_ok=True)
    np.save(os.path.join(S2, "top8.npy"), top)

    def write(label, poses_tok):
        poses = S0["poses"].copy()
        poses[rows] = poses_tok
        np.savez(os.path.join(SEAMD, f"{label}.npz"), token=S0["token"], fingerprint=S0["fingerprint"],
                 poses=poses.astype(np.float32), sampling=S0["sampling"], arm=np.array(label.replace("refe_", "REFe_")))
    for k in range(64):
        write(seam_label(f"p{k:02d}"), np.stack([SEAM.to_navsim(repair_last_heading(tr[j, k])) for j in range(len(toks))]))
    for r in range(8):
        write(seam_label(f"c{r}"), np.stack([SEAM.to_navsim(repair_last_heading(SC.slow_copy(tr[j, top[j, r]], 0.75)))
                                             for j in range(len(toks))]))
    write(seam_label("stop"), np.zeros((len(toks), 8, 3), np.float32))
    print(f"ZZM5R2_SEAMS_OK {len(glob.glob(os.path.join(SEAMD, '*.npz')))} seams, {len(toks)} of {n} rows")
    return 0


def labels():
    return [seam_label(f"p{k:02d}") for k in range(64)] + [seam_label(f"c{r}") for r in range(8)] + [seam_label("stop")]


def score(a) -> int:
    """The 73 harness runs, RAM-aware (coordinator 2026-09-28): a run is STARTED only at >= --add-at-gb free and at
    most --workers at once; the newest run is KILLED (its own tree, explicit PID) and re-queued when free RAM stays
    < --retire-below-gb for 40 s. A run whose csv + PASS log exist is skipped (resumable)."""
    require_stage1_pass()
    import subprocess
    import eval_checkpoint as EC
    import slow_copy_probe as SCP
    import m5r_label as ML
    os.makedirs(f"{S2}/logs", exist_ok=True)

    def passed(lb):
        log, csvp = f"{S2}/logs/{lb}.log", f"{EC.DATA}/score/{lb}/{lb}.csv"
        if os.path.exists(csvp) and os.path.exists(log):
            st = SCP.status_of(open(log, encoding="utf-8", errors="replace").read())
            return st if st and st.get("status") == "PASS" else None
        return None
    todo = [lb for lb in labels() if not passed(lb)]
    live, res, t0, last, low = {}, {}, time.time(), 0.0, None
    tries = {}
    while todo or live:
        for lb, (p, fh, t_) in list(live.items()):
            if p.poll() is not None:
                fh.close()
                live.pop(lb)
                st = passed(lb)
                res[lb] = st
                print(f"  {time.strftime('%H:%M:%S')} {lb}: {'PASS' if st else 'FAILED rc %s' % p.returncode} "
                      f"({(time.time() - t_) / 60:.1f} min); {len(todo)} to start, {len(live)} live", flush=True)
                if not st and tries.get(lb, 0) < 2:
                    todo.append(lb)
        fg = ML.free_gb()
        now = time.time()
        if live and 0 <= fg < a.retire_below_gb:
            low = low or now
            if now - low >= 40:
                lb = max(live, key=lambda k: live[k][2])
                p, fh, _t = live.pop(lb)
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
                fh.close()
                todo.insert(0, lb)
                print(f"  {time.strftime('%H:%M:%S')} RAM {fg:.2f} GB < {a.retire_below_gb}: KILLED {lb}, re-queued",
                      flush=True)
                low, last = None, now
        else:
            low = None
        if todo and len(live) < a.workers and fg >= a.add_at_gb and now - last >= 60:
            lb = todo.pop(0)
            tries[lb] = tries.get(lb, 0) + 1
            fh = open(f"{S2}/logs/{lb}.log", "w", encoding="utf-8")
            p = subprocess.Popen([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", lb, "--seam",
                                  os.path.join(SEAMD, f"{lb}.npz"), "--tokens", TOK, "--out", f"{EC.DATA}/score"],
                                 cwd=HERE, env=dict(EC.env_driverl(), PYTHONIOENCODING="utf-8"), stdout=fh,
                                 stderr=subprocess.STDOUT)
            live[lb] = (p, fh, now)
            last = now
            print(f"  {time.strftime('%H:%M:%S')} START {lb} (free RAM {fg:.2f} GB; {len(live)} live)", flush=True)
        time.sleep(15)
    for lb in labels():
        res[lb] = passed(lb)
    bad = [k for k, v in res.items() if not (v and v.get("status") == "PASS")]
    json.dump({"runs": len(res), "failed": bad, "seconds": round(time.time() - t0, 1)},
              open(os.path.join(OUTD, "stage2_score_status.json"), "w"), indent=1)
    print(f"ZZM5R2_SCORE_{'OK' if not bad else 'FAIL'} {len(res) - len(bad)}/{len(res)}")
    return 0 if not bad else 1


def setup():
    import eval_checkpoint as EC
    import slow_copies as SC
    toks, _ = tokens()
    C = M6Ctx()
    P = np.load(A7P)
    pi = {str(t): i for i, t in enumerate(P["token"])}
    top = np.load(os.path.join(S2, "top8.npy"))
    props = np.stack([C.traj(t) for t in toks]).astype(np.float32)
    X = np.stack([np.stack([SC.slow_copy(props[j, top[j, r]], FE.FACTOR) for r in range(8)] +
                           [np.zeros((20, 3), np.float32)]).astype(np.float32) for j in range(len(toks))])
    H = [FE.read_csv(f"{EC.DATA}/score/{lb}/{lb}.csv") for lb in labels()]
    pd = np.array([[h[t][1] if h.get(t) and h[t][0] else np.nan for h in H] for t in toks], np.float64)
    sub = np.array([[list(h[t][2]) if h.get(t) and h[t][0] else [np.nan] * 6 for h in H] for t in toks], np.float64)
    T = {"logits": np.stack([P["logits"][pi[t]] for t in toks]), "pick": np.array([P["pick"][pi[t]] for t in toks])}
    return {"C": C, "toks": toks, "props": props, "X": X, "pdms": pd, "sub": sub, "T": T,
            "harness_missing": int(np.isnan(pd).sum())}


def eval_(a) -> int:
    require_stage1_pass()
    import torch
    device = torch.device(a.device)
    FE.vram_cap(device)
    FE.reserve_vram(device)
    E = setup()
    out = f"{S2}/evals"
    os.makedirs(out, exist_ok=True)
    gates = {"harness_missing": E["harness_missing"]}
    np.savez(f"{out}/_truth.npz", token=np.array(E["toks"]), pdms=E["pdms"], sub=E["sub"])
    for name, m in R.models().items():
        L = FE.eval_model(m, E, device)
        if name == "A0":
            gates["A0_pure_vs_a7confirm_logits_max_abs"] = float(np.abs(L["pure"].astype(np.float64) - E["T"]["logits"]).max())
            gates["A0_pick_equals"] = int((FE.v1_agg(L["pure"]).argmax(1) == E["T"]["pick"]).sum())
        gates["masked_64_vs_pure_max_abs"] = max(gates.get("masked_64_vs_pure_max_abs", 0.0),
                                                 float(np.abs(L["masked"][:, :64] - L["pure"]).max()))
        np.savez(f"{out}/{name}.npz", token=np.array(E["toks"]), **L, **FE.metrics(L, E))
        print(f"  {name} done", flush=True)
    gates["pass"] = (gates["harness_missing"] == 0 and gates["A0_pure_vs_a7confirm_logits_max_abs"] <= 1e-3
                     and gates["A0_pick_equals"] == len(E["toks"]) and gates["masked_64_vs_pure_max_abs"] <= 1e-4)
    json.dump(gates, open(os.path.join(OUTD, "stage2_gate_eval.json"), "w"), indent=1)
    print(json.dumps(gates))
    return 0 if gates["pass"] else 1


def readout(_a) -> int:
    require_stage1_pass()
    toks, token_log = tokens()
    ev = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in glob.glob(f"{S2}/evals/*.npz")
          if not os.path.basename(p).startswith("_")}
    logs = np.array([token_log[t] for t in toks])
    ul = np.unique(logs)
    idx = {l_: np.where(logs == l_)[0] for l_ in ul}
    rng = np.random.default_rng(20260927)
    draws = [np.concatenate([idx[l_] for l_ in rng.choice(ul, size=len(ul), replace=True)]) for _ in range(10000)]
    seeds = [0, 1, 2]

    def arm(nm, key):
        return ev["A0"][key] if nm == "A0" else np.nanmean(np.stack([ev[f"{nm}_s{s}"][key] for s in seeds]), 0)
    cmp = {}
    for key in ("E1_masked", "E3a", "E3b", "E2"):
        for pn, (x, y) in {"V4r_minus_V3": ("V4r", "V3"), "V3r_minus_V3": ("V3r", "V3"),
                           "V4r_minus_V3r": ("V4r", "V3r"), "V4r_minus_V4": ("V4r", "V4")}.items():
            try:
                d = arm(x, key) - arm(y, key)
            except KeyError:
                continue
            lo, hi = FE.boot_ci(d, draws)
            cmp.setdefault(key, {})[pn] = {"diff": round(float(np.nanmean(d)), 5), "ci95": [round(lo, 5), round(hi, 5)]}
    def seedpairs(nm):
        tm = [float(np.nanmean(ev[f"{nm}_s{s}"]["E1_masked"])) for s in seeds]
        return [abs(tm[i] - tm[j]) for i in range(3) for j in range(i + 1, 3)], tm
    gates = {f: json.load(open(os.path.join(OUTD, f))).get("pass") for f in ("stage2_gate_context.json",
                                                                              "stage2_gate_eval.json")}
    arms = {}
    for X in ("V4r", "V3r"):
        floor = max(seedpairs("V3")[0] + seedpairs(X)[0])
        e1, e3a, e3b, e2 = (cmp[k][f"{X}_minus_V3"] for k in ("E1_masked", "E3a", "E3b", "E2"))
        ok = (e1["diff"] >= 0.03 and e1["ci95"][0] > 0 and e1["diff"] > floor and e3a["ci95"][0] > -0.01
              and e3b["ci95"][0] > -2.0 and e2["ci95"][0] > -2.0)
        arms[X] = {"verdict": "PASS" if ok else ("REFUTED" if e1["ci95"][1] < 0 else "NOT PROVEN"),
                   "seed_floor": round(floor, 5), "seed_token_means_E1": seedpairs(X)[1],
                   "E1": e1, "E3a": e3a, "E3b": e3b, "E2": e2}
    vv = cmp["E1_masked"]["V4r_minus_V3r"]
    floor43 = max(seedpairs("V3r")[0] + seedpairs("V4r")[0])
    v4r_beats = vv["ci95"][0] > 0 and vv["diff"] > floor43
    pv, p3 = arms["V4r"]["verdict"] == "PASS", arms["V3r"]["verdict"] == "PASS"
    if not all(gates.values()):
        verdict = "NO READOUT: gates failed"
    elif pv and p3:
        verdict = "ADOPT V4r" if v4r_beats else "ADOPT V3r"
    elif p3:
        verdict = "ADOPT V3r"
    elif pv:
        verdict = "ADOPT V4r"
    else:
        verdict = f"NOTHING ADOPTED (V4r {arms['V4r']['verdict']}, V3r {arms['V3r']['verdict']})"
    res = {"_label": "PRE-REGISTERED Stage-2 readout (PREREG_MEASURE5R §6 + ADDENDUM 1, blob 2f32042b), A5's 923 "
                     "fresh tokens, 43 logs, REPAIRED truth", "n_tokens": len(toks), "n_logs": len(ul), "gates": gates,
           "estimator": "paired log-cluster bootstrap over the 43 logs, 10,000 resamples, 95 % percentile, seed 20260927",
           "comparisons": cmp, "arms": arms,
           "decision": {"verdict": verdict, "per_arm": {k: v["verdict"] for k, v in arms.items()},
                        "named_bar_E1_V4r_minus_V3r": vv, "named_bar_seed_floor": round(floor43, 5),
                        "V4r_beats_V3r_on_the_named_bar": v4r_beats,
                        "rule": "per arm X vs V3: E1 >= +0.03, CI lo > 0, > seed floor(V3, X); E3a lo > -0.01; E3b lo "
                                "> -2.0; E2 lo > -2.0; REFUTED iff E1 hi < 0. Both pass -> V3r unless E1 V4r-V3r CI "
                                "lo > 0 and > seed floor(V3r, V4r) -> V4r; one passes -> it; neither -> nothing."}}
    json.dump(res, open(os.path.join(OUTD, "readout_stage2.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(res["decision"], indent=1))
    print(f"ZZM5R_STAGE2 {verdict}")
    _subs(ev, toks, draws, res)
    return 0


def _subs(ev, toks, draws, res):
    """Reported: EP + the NAVSIM sub-scores (x100) of each arm's E3b / E2 pick, paired vs V3; the E2 selection mix."""
    S = np.load(f"{S2}/evals/_truth.npz")["sub"]
    ar = np.arange(len(toks))
    per = {}
    for nm in ("A0", "V3", "V3r", "V4", "V4r"):
        names = ["A0"] if nm == "A0" else [f"{nm}_s{s}" for s in (0, 1, 2)]
        if any(n not in ev for n in names):
            continue
        per[nm] = {r: np.nanmean(np.stack([100.0 * S[ar, (FE.v1_agg(ev[n]["pure"]).argmax(1) if r == "E3b" else
                                                         np.nan_to_num(FE.v1_agg(ev[n]["masked"]), nan=-1).argmax(1))]
                                           for n in names]), 0) for r in ("E3b", "E2")}
    out = {"order": ["NC", "DAC", "EP", "TTC", "C", "DDC"],
           "arm_means": {nm: {r: [round(float(v), 3) for v in np.nanmean(x, 0)] for r, x in d.items()}
                         for nm, d in per.items()}, "pairs": {}}
    for r in ("E3b", "E2"):
        for pn, (x, y) in {"V4r_minus_V3": ("V4r", "V3"), "V3r_minus_V3": ("V3r", "V3"),
                           "V4r_minus_V3r": ("V4r", "V3r")}.items():
            if x in per and y in per:
                d = per[x][r] - per[y][r]
                out["pairs"].setdefault(r, {})[pn] = {h: {"diff": round(float(np.nanmean(d[:, j])), 3),
                                                          "ci95": [round(v, 3) for v in FE.boot_ci(d[:, j], draws)]}
                                                      for j, h in enumerate(out["order"])}
    res["pick_subscores_x100"] = out
    res["E2_selection_mix"] = {n: {"original": int((k < 64).sum()), "copy_075": int(((k >= 64) & (k < 72)).sum()),
                                   "stop": int((k == 72).sum())}
                               for n, e in ev.items() for k in [np.nan_to_num(FE.v1_agg(e["masked"]), nan=-1).argmax(1)]}
    json.dump(res, open(os.path.join(OUTD, "readout_stage2.json"), "w", encoding="utf-8"), indent=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("gate", "seams", "score", "eval", "readout"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--add-at-gb", type=float, default=6.3)
    ap.add_argument("--retire-below-gb", type=float, default=4.7)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    return {"gate": gate, "seams": seams, "score": score, "eval": eval_, "readout": readout}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
