"""MM 2026-11-17: PDMS of the command-consistent re-rank (s3_rerank.py's rerank_p7_seam.npz) on full navtest, by scoring ONLY the tokens it changes
and splicing them into P7's banked per-token csv.  IDENTITY CONTROL first: P7's own pick, scored on the same tokens by the same harness, must reproduce
the banked csv's scores (max |diff| <= 1e-9) or the splice is refused.  Estimator: proxy_eval.boot (paired log-cluster, B 10,000, seed 20260927), on
per-token differences; answers "another draw of logs" only (one checkpoint, deterministic inference).  Tier T1 (NAVSIM v1 PDMS pseudo-sim).
usage: s4_pdms_rerank.py <work dir> ; run by run_s4.sh after the dev-box gate sweep."""
import gzip, json, os, subprocess, sys, time
import numpy as np

WORK = sys.argv[1]
PK = "F:/TanitAD_p7read_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
EV = PK + "/eval"
sys.path.insert(0, EV)
import proxy_eval as PE           # noqa: E402  (pinned; imported, never edited)
import command_adherence as CA    # noqa: E402

HOLD = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/adh_decomp"
SEAM_P7 = "E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7_full.npz"
SEAM_RR = HOLD + "/rerank_p7_seam.npz"
CSV_P7 = "E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p7_full/refe_stage2_p7_full.csv"
EXPORT = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]

a, b = np.load(SEAM_P7), np.load(SEAM_RR)
tok = [str(t) for t in a["token"]]
assert tok == [str(t) for t in b["token"]]
chg = sorted(t for t, d in zip(tok, np.abs(a["poses"] - b["poses"]).reshape(len(tok), -1).max(1)) if d > 0)
os.makedirs(WORK + "/score", exist_ok=True)
exp = CA.load_export(EXPORT)                                  # {token: agent input}: the module's own loader
cmd, logs = CA.commands_and_logs(tok, exp)
tl = dict(zip(tok, logs))
tc = dict(zip(tok, cmd))
NSH = 2                                                       # the harness puts every token AND log name on ONE Windows command line (32,767 chars): 2 shards
shards = [chg[i::NSH] for i in range(NSH)]
tfs = []
for i, sh in enumerate(shards):
    tfs.append(f"{WORK}/chg_tokens_s{i}.json")
    json.dump({"tokens": sh, "token_log": {t: tl[t] for t in sh}}, open(tfs[-1], "w"))
print("changed tokens", len(chg), "shards", [len(x) for x in shards], flush=True)


def score(label, seam):
    rows = {}
    for i, tf in enumerate(tfs):
        rows.update(PE.read_csv(score1(f"{label}_s{i}", seam, tf)))
    return rows


def score1(label, seam, tf):
    csv_p = f"{WORK}/score/{label}/{label}.csv"
    for att in range(1, 4):
        if os.path.exists(csv_p):
            return csv_p
        lp = f"{WORK}/{label}_score_{att}.log"
        env = dict(os.environ, PYTHONIOENCODING="utf-8", RUN_V1_FIX1_ALLOW_DRIFT="1", TANITAD_NAVSIM_ROOT="F:/Archive/devbox-C/navsim")
        rc = subprocess.call([PY_D, "score_e.py", "--label", label, "--seam", seam, "--tokens", tf, "--out", WORK + "/score"], cwd=EV + "/harness", env=env,
                             stdout=open(lp, "w"), stderr=subprocess.STDOUT)
        print(label, "attempt", att, "rc", rc, flush=True)
        if not os.path.exists(csv_p):
            time.sleep(120)
    raise SystemExit(f"REFUSING: {label} did not produce {csv_p}")


ctl = score("refe_p7pick_chg", SEAM_P7)
rr = score("refe_p7rerank_chg", SEAM_RR)
base = PE.read_csv(CSV_P7)
assert set(ctl) == set(chg) == set(rr), (len(ctl), len(chg), len(rr))
dmax = max(abs(float(ctl[t]["score"]) - float(base[t]["score"])) for t in chg)
res = {"instrument": "adh_decomp/scripts/s4_pdms_rerank.py", "tier": "T1 (NAVSIM v1 PDMS pseudo-sim, open-loop plan)", "estimator": "proxy_eval.boot paired log-cluster, B 10,000, seed 20260927 ('another draw of logs' only)",
       "n_changed": len(chg), "control_pick_rescored_max_abs_diff_vs_banked_csv": dmax}
if dmax > 1e-9:
    res["VERDICT"] = "INVALID: the identity control did not reproduce the banked csv; the splice is not quotable"
    json.dump(res, open(HOLD + "/adh_q6_pdms.json", "w"), indent=1)
    raise SystemExit(res["VERDICT"])
spl = {t: (rr[t] if t in rr else base[t]) for t in tok}


def blk(tset, key="score"):
    d = {t: 100.0 * (float(spl[t][key]) - float(base[t][key])) for t in tset}
    m, lo, hi = PE.boot(d, tl)
    return {"n": len(tset), "n_logs": len({tl[t] for t in tset}), "p7": 100.0 * float(np.mean([float(base[t][key]) for t in tset])),
            "rerank": 100.0 * float(np.mean([float(spl[t][key]) for t in tset])), "delta": m, "ci95": [lo, hi], "separated": bool(lo > 0 or hi < 0)}


turn = [t for t in tok if int(tc[t]) in (CA.LEFT, CA.RIGHT)]   # turn-active = LEFT u RIGHT commands (4,076)
assert len(turn) == 4076, len(turn)
res["PDMS_full_navtest"] = blk(tok)
res["PDMS_turn_active"] = blk(turn)
res["PDMS_changed_tokens"] = blk(chg)
res["subscores_changed_tokens"] = {k: blk(chg, k) for k in SUBS}
json.dump(res, open(HOLD + "/adh_q6_pdms.json", "w"), indent=1)
f = res["PDMS_full_navtest"]
print(f"ZZS4_DONE full navtest PDMS P7 {f['p7']:.4f} -> rerank {f['rerank']:.4f}, delta {f['delta']:+.4f} [{f['ci95'][0]:+.4f}, {f['ci95'][1]:+.4f}] (control max diff {dmax:.2e})")
