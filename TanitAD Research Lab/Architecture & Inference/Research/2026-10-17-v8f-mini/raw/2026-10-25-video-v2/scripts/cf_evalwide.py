"""cf_evalwide.py -- the OPTIONAL context card declared in SELECTION_V2.md s2: the same nav counterfactual over ALL 2,399 eval windows (no window drawn).
Per window D6 (headline) and lateral separation per injection point; T's lateral argmax vs the command; episode-cluster bootstrap (B = 2,000, 141 episodes).
CPU only; reads the 141-clip eval cache of SFRESH (read-only).  Writes OUT/data/cf_evalwide.npz + cf_evalwide.json."""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_lib import *  # noqa
from taniteval.ci import episode_cluster_bootstrap

t0 = time.time()
out_arr, out_js = {}, {}
for seed in SEEDS:
    model, cfg = load_model(seed)
    ew = EvalWorld(cfg, lru=8, cache=SFRESH_CACHE)
    N = ew.N
    D6 = {c: np.zeros(N) for c in CONFIGS}
    SEP = {c: np.zeros(N) for c in CONFIGS}
    LATOK = {c: np.zeros((N, 3), bool) for c in CONFIGS}
    own_ade6 = np.zeros(N)
    for b0 in range(0, N, 64):
        idx = list(range(b0, min(b0 + 64, N)))
        b = ew.batch(idx)
        for cname, where in CONFIGS.items():
            pl, la = {}, {}
            for cmd in CMDS:
                r = chain(model, b, nav=cmd_tensor(b, cmd), where=where)
                pl[cmd] = r["traj30"][:, 29].numpy().astype(np.float64)
                la[cmd] = r["lat_logits"].argmax(-1).numpy()
            f, l, rr = pl["FOLLOW"], pl["TURN_L"], pl["TURN_R"]
            D6[cname][idx] = 0.5 * (np.linalg.norm(l - f, axis=-1) + np.linalg.norm(rr - f, axis=-1))
            SEP[cname][idx] = np.abs(l[:, 1] - rr[:, 1])
            LATOK[cname][idx] = np.stack([la["FOLLOW"] == 0, la["TURN_L"] == 1, la["TURN_R"] == 2], 1)
        print(seed, b0, f"{time.time() - t0:.0f}s", flush=True)
    eid = ew.X["eid"]
    js = {}
    for cname in CONFIGS:
        js[cname] = {"D6": episode_cluster_bootstrap(D6[cname], eid, n_boot=2000, seed=0, dp=3),
                     "lat_sep_LR": episode_cluster_bootstrap(SEP[cname], eid, n_boot=2000, seed=0, dp=3),
                     "T_lat_argmax_follows_cmd": {c: float(LATOK[cname][:, k].mean()) for k, c in enumerate(CMDS)}}
        out_arr[f"{seed}/{cname}/D6"], out_arr[f"{seed}/{cname}/lat_sep"], out_arr[f"{seed}/{cname}/latok"] = D6[cname], SEP[cname], LATOK[cname]
    out_arr[f"{seed}/eid"] = eid
    out_js[seed] = js
    print(seed, json.dumps({c: (js[c]["D6"]["mean"], js[c]["D6"]["lo"], js[c]["D6"]["hi"]) for c in js}), flush=True)
np.savez_compressed(f"{OUT}/data/cf_evalwide.npz", **out_arr)
json.dump({"estimator": "episode_cluster_bootstrap, B=2000, seed 0, 141 episodes (taniteval/ci.py): answers 'another draw of EPISODES'; training seeds side by side",
           "n_windows": 2399, "tier": "T1-class open loop, DIAGNOSTIC (nav command injected at the named layer(s); not driving)", "seeds": out_js}, open(f"{OUT}/data/cf_evalwide.json", "w"), indent=1)
print("done", f"{time.time() - t0:.0f}s")
