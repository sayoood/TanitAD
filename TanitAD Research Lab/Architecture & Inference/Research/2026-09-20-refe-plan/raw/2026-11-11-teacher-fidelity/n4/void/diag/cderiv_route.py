"""TF-N4 C-DERIV diagnosis (MM, 2026-11-13): does the DB-route ego reproduce the banked N2-R labels where the cache-route ego does not?"""
import json, os, sys
import numpy as np
RUN = "F:/tanitad_scratch/n4nt/run"
sys.path.insert(0, RUN + "/n4")
import n4_nt_build as B  # noqa: E402
import n4_geom as G  # noqa: E402
import transp_score as TS  # noqa: E402
CACHE = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
z = np.load(RUN + "/inputs/n2r/n2_plans_N2R.npz", allow_pickle=False)
toks, logs, plans = [str(t) for t in z["tokens"]], [str(x) for x in z["logs"]], np.asarray(z["plans20"], float)
ix = {t: i for i, t in enumerate(toks)}
rolls = B.load_rolls(RUN + "/inputs/n2r/rolls_RJ5/rows_s*of2.jsonl")
cder = sorted(toks, key=B.sha1_int)[:B.N_CDERIV]
dev = {}
for f in ("F:/tanitad_scratch/n4nt/run/n4run/build/derive_s0of2.jsonl", "F:/tanitad_scratch/n4nt/run/n4run/build/derive_s1of2.jsonl"):
    for ln in open(f, encoding="utf-8"):
        r = json.loads(ln)
        if "cderiv_maxdev_m" in r: dev[r["token"]] = r["cderiv_maxdev_m"]
worst = sorted(dev, key=lambda t: -dev[t])[:8]
pick = worst + [t for t in cder if t not in worst][:32]
by = {}
for t in pick: by.setdefault(logs[ix[t]], []).append(t)
out = []
for lg, tok, cdb in G.contexts(by):
    ccache = G.Ctx.from_parts(TS.load_cache_parts(CACHE, lg, tok))
    ds0 = float(np.abs(np.asarray(cdb.s0, float) - np.asarray(ccache.s0, float)).max())
    t20 = rolls[tok]; base = plans[ix[tok]]
    rdb = cdb.labels(t20)["R"]; rc = ccache.labels(t20)["R"]
    rec = {"token": tok, "ego_s0_maxdiff": ds0, "db_label_vs_bank_m": float(np.abs(rdb - base).max()),
           "cache_label_vs_bank_m": float(np.abs(rc - base).max()), "chain_cderiv_m": dev.get(tok)}
    out.append(rec); print(json.dumps(rec), flush=True)
a = np.array([r["db_label_vs_bank_m"] for r in out]); c = np.array([r["cache_label_vs_bank_m"] for r in out]); e = np.array([r["ego_s0_maxdiff"] for r in out])
summ = {"n": len(out), "db_route_max_m": float(a.max()), "db_route_n_le_1e9": int((a <= 1e-9).sum()), "cache_route_max_m": float(c.max()),
        "cache_route_n_le_1e9": int((c <= 1e-9).sum()), "ego_s0_maxdiff_max": float(e.max()), "ego_s0_maxdiff_median": float(np.median(e))}
print("ZZDIAG", json.dumps(summ))
json.dump({"summary": summ, "rows": out}, open("F:/tanitad_scratch/n4nt/diag/cderiv_route.json", "w"), indent=1)
