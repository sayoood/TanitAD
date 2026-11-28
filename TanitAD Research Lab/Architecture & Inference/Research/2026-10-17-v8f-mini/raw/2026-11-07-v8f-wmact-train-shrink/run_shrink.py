"""AMENDMENT_WMACT_1 (frozen prefix 2,249 bytes, sha256 503ba1b8...): is level 1's predicted change aimed right but too large?
c_k is fitted on the TRAIN windows only and applied to eval-141. Writes RESULT_WMACT_SHRINK.json. Run with WMACT_CODE_ROOT set (see HANDOFF_WMACT.md)."""
import hashlib, json, time
import numpy as np, torch
import wmact_lib as W

AMEND, AMEND_N, AMEND_SHA = "../AMENDMENT_WMACT_1.md", 2249, "503ba1b881ce48715329bb6017d23f68477031c31d99e09b027e7cb8582afcc3"
assert hashlib.sha256(open(AMEND, "rb").read()[:AMEND_N]).hexdigest() == AMEND_SHA, "amendment prefix changed"
t0 = time.time()
EV, TR = torch.load("latents.pt", weights_only=False), torch.load("latents_train.pt", weights_only=False)
REF = json.load(open("RESULT_WMACT.json", encoding="utf-8"))


def pick(L):
    sel = np.flatnonzero(L["ok"].all(1).numpy())
    return sel, np.array(L["eid"])[sel]


sel_e, eid_e = pick(EV)
sel_t, _ = pick(TR)
uniq, Wt = W.boot_weights(eid_e)
W.check_draws_equal_ci(eid_e, uniq, Wt)
n, n_ep = len(sel_e), len(uniq)


@torch.no_grad()
def deltas(wm, L, sel):
    Z1, A_raw = L["Z1"][TAG][sel], L["A_raw"][sel]
    pk = W.level_pack(wm, Z1, wm.scale_actions(A_raw)[:, :-1])
    z = pk["true"][1].double()
    return {k: ((pk["pred"][(1, k)].double() - z[:, 0]), (z[:, k] - z[:, 0])) for k in W.KS[1]}


res = {"amendment": {"file": "AMENDMENT_WMACT_1.md", "prefix_bytes": AMEND_N, "sha256": AMEND_SHA}, "tier": "T0",
       "eval_windows": int(n), "eval_episodes": int(n_ep), "train_windows": int(len(sel_t)), "rows": {}, "controls": {}}
for TAG in ("M2b_s0", "M2b_s1"):
    m, cfg, ck = W.load_model(W.CKPT[TAG])
    dt, de = deltas(m.wm, TR, sel_t), deltas(m.wm, EV, sel_e)
    for k in W.KS[1]:
        ph_t, d_t = dt[k]
        c = float((ph_t * d_t).sum() / (ph_t * ph_t).sum())                       # fitted on TRAIN only
        ph, d = de[k]
        C = (d * d).sum(-1).numpy()
        cols = {"C": C}
        for nm, cc in (("shrunk", c), ("c1", 1.0), ("c0", 0.0)):
            cols[nm] = ((cc * ph - d) ** 2).sum(-1).numpy()
        es = W.ep_sums(cols, eid_e, uniq)
        row = {"c_train": c}
        for nm in ("shrunk", "c1", "c0"):
            row[f"SKILL_{nm}"] = W.ratio_row(es, es["C"], {"C": 1.0, nm: -1.0}, Wt, n, n_ep)
        dot, pp, dd = float((ph * d).sum()), float((ph * ph).sum()), float((d * d).sum())
        row["oracle_cos2_eval"] = dot * dot / (pp * dd)
        row["oracle_c_eval"] = dot / pp
        ref = REF["rows"][TAG][f"L1/k{k}"]["SKILL"]["point"]
        res["controls"][f"{TAG}/k{k}"] = {"c1_minus_record": row["SKILL_c1"]["point"] - ref, "c0_point": row["SKILL_c0"]["point"]}
        res["rows"][f"{TAG}/L1/k{k}"] = row
ok = all(abs(v["c1_minus_record"]) <= 1e-9 and v["c0_point"] == 0.0 for v in res["controls"].values())
res["controls_pass"] = ok
k1 = [res["rows"][f"{t}/L1/k1"]["SKILL_shrunk"] for t in ("M2b_s0", "M2b_s1")]
if not ok:
    res["outcome"] = "INVALID (a control failed)"
elif all(r["separated"] and r["point"] > 0 for r in k1):
    res["outcome"] = "L1-OVERSHOOTS"
elif not any(r["separated"] for r in k1):
    res["outcome"] = "L1-MISAIMED"
else:
    res["outcome"] = "MIXED"
json.dump(res, open("RESULT_WMACT_SHRINK.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
print("outcome", res["outcome"], "controls_pass", ok, round(time.time() - t0, 1), "s")
