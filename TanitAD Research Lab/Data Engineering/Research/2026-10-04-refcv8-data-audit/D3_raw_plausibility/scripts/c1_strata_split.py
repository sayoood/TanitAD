"""D3 check 1a/1c -- id-level split integrity (labels, cache views, sidecars) and strata-mix comparison train vs eval139.

Strata test: Pearson chi-square of the (train-cache clips x eval clips) contingency, with a MONTE-CARLO PERMUTATION p-value
(cells in the 139-clip eval are small, ~5.5 expected per country, so the asymptotic chi-square p is not trusted).
Control: the same statistic for a RANDOM 139-clip subset of train vs the rest must give p ~ U(0,1) -- we report its
p-values over 200 random splits (their median must be ~0.5 and the fraction < 0.05 must be ~5 %).
Ids are hashed (sha12); nothing here prints a raw clip id.
"""
import gzip, hashlib, json, sys, collections
import numpy as np
import torch
OUT = "D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D3_raw_plausibility/raw"
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]

def load(p):
    rows = {}
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l); rows[r["clip_id"]] = r
    return rows

LT = load("D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz")
LE = load("D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")
mt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
me = torch.load("D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt", map_location="cpu", weights_only=False)
tr_ids, ev_ids = list(mt["clip_id"]), list(me["clip_id"])
res = {}
# ---- ids -----------------------------------------------------------------------------------------
res["ids"] = {
  "n_train_cache": len(tr_ids), "n_train_cache_unique": len(set(tr_ids)), "n_eval_cache": len(ev_ids), "n_eval_cache_unique": len(set(ev_ids)),
  "train_cache_cap_eval_cache": len(set(tr_ids) & set(ev_ids)),
  "n_train_label_records": len(LT), "n_eval_label_records": len(LE),
  "train_labels_cap_eval_labels": len(set(LT) & set(LE)),
  "train_cache_cap_eval_labels": len(set(tr_ids) & set(LE)),
  "eval_cache_cap_train_labels": len(set(ev_ids) & set(LT)),
  "train_cache_not_in_train_labels": len(set(tr_ids) - set(LT)), "eval_cache_not_in_eval_labels": len(set(ev_ids) - set(LE)),
  "train_labels_not_in_cache": len(set(LT) - set(tr_ids)), "eval_labels_not_in_cache": len(set(LE) - set(ev_ids)),
  "episode_uid_train_unique": len(set(int(x) for x in mt["episode_uid"])), "episode_uid_eval_unique": len(set(int(x) for x in me["episode_uid"])),
  "episode_uid_overlap": len(set(int(x) for x in mt["episode_uid"]) & set(int(x) for x in me["episode_uid"])),
  "episode_id16_train_unique": len(set(int(x) for x in mt["episode_id"])), "episode_id16_overlap_train_eval": len(set(int(x) for x in mt["episode_id"]) & set(int(x) for x in me["episode_id"])),
  "split_field_train_records": dict(collections.Counter(r.get("split") for r in LT.values())),
  "split_field_eval_records": dict(collections.Counter(r.get("split") for r in LE.values())),
}
# 8-char / 12-char prefix collisions (logs print prefixes): train vs eval
res["ids"]["prefix8_overlap"] = len({c[:8] for c in tr_ids} & {c[:8] for c in ev_ids})
# ---- strata ----------------------------------------------------------------------------------------
def col(rows, ids, f):
    return [f(rows[c]) for c in ids if c in rows]
feats = {
 "country": lambda r: r["strata"]["country"], "daynight": lambda r: r["strata"]["daynight_clock"], "road_class": lambda r: r["strata"]["road_class"],
 "strata_cell": lambda r: r["strata"]["strata_cell"], "nav": lambda r: r["nav_command"]["token"],
 "a_tac_lat": lambda r: (r.get("a_tac") or {}).get("lat"), "a_tac_lon": lambda r: (r.get("a_tac") or {}).get("lon"),
}
rng = np.random.default_rng(11)
def chi2(counts_a, counts_b, cats):
    a = np.array([counts_a.get(c, 0) for c in cats], float); b = np.array([counts_b.get(c, 0) for c in cats], float)
    n = a.sum() + b.sum(); tot = a + b
    ea = tot * a.sum() / n; eb = tot * b.sum() / n
    m = tot > 0
    return float((((a - ea) ** 2 / np.where(ea > 0, ea, 1))[m]).sum() + (((b - eb) ** 2 / np.where(eb > 0, eb, 1))[m]).sum())
def perm_p(va, vb, nperm=4000):
    cats = sorted(set(va) | set(vb), key=str)
    ia = np.array([cats.index(x) for x in va]); ib = np.array([cats.index(x) for x in vb])
    allv = np.concatenate([ia, ib]); na = len(ia)
    obs = chi2(collections.Counter(va), collections.Counter(vb), cats)
    ge = 0
    for _ in range(nperm):
        rng.shuffle(allv)
        A = np.bincount(allv[:na], minlength=len(cats)); B = np.bincount(allv[na:], minlength=len(cats))
        s = chi2(dict(enumerate(A)), dict(enumerate(B)), list(range(len(cats))))
        ge += s >= obs
    return obs, (ge + 1) / (nperm + 1), len(cats)
res["strata"] = {}
for k, f in feats.items():
    vt = col(LT, tr_ids, f); ve = col(LE, ev_ids, f)
    obs, p, nc = perm_p([str(x) for x in vt], [str(x) for x in ve])
    ct, ce = collections.Counter(map(str, vt)), collections.Counter(map(str, ve))
    top = sorted(set(ct) | set(ce), key=lambda c: -ct.get(c, 0))
    res["strata"][k] = {"n_train": len(vt), "n_eval": len(ve), "n_categories": nc, "chi2": round(obs, 3), "perm_p": round(p, 4),
                        "table_pct": {c: [round(100 * ct.get(c, 0) / len(vt), 2), round(100 * ce.get(c, 0) / len(ve), 2)] for c in top[:40]}}
# control: random 139 of train vs rest -> p distribution
ps = []
vt_all = [str(x) for x in col(LT, tr_ids, feats["country"])]
for _ in range(120):
    idx = rng.permutation(len(vt_all)); a = [vt_all[i] for i in idx[:139]]; b = [vt_all[i] for i in idx[139:]]
    ps.append(perm_p(a, b, nperm=300)[1])
res["control_random_139_vs_rest_country"] = {"n_splits": len(ps), "median_p": float(np.median(ps)), "frac_p_lt_0.05": float(np.mean(np.array(ps) < 0.05))}
# recording span, stopped fraction
for k, f in {"recording_span_s": lambda r: r["horizon"]["recording_span_s"], "stopped_frame_frac_20s": lambda r: r["strata"]["stopped_frame_frac_20s"]}.items():
    a = np.array(col(LT, tr_ids, f), float); b = np.array(col(LE, ev_ids, f), float)
    res["strata"][k] = {"train": {q: round(float(np.quantile(a, q)), 3) for q in (0.05, 0.5, 0.95)}, "eval": {q: round(float(np.quantile(b, q)), 3) for q in (0.05, 0.5, 0.95)}}
# countries present only in one split
ct = collections.Counter(col(LT, tr_ids, feats["country"])); ce = collections.Counter(col(LE, ev_ids, feats["country"]))
res["countries"] = {"n_train": len(ct), "n_eval": len(ce), "eval_countries_absent_from_train": [c for c in ce if c not in ct],
                    "train_countries_absent_from_eval": [c for c in ct if c not in ce], "eval_per_country_min_max": [min(ce.values()), max(ce.values())]}
json.dump(res, open(OUT + "/c1_strata_split_result.json", "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ("ids", "countries", "control_random_139_vs_rest_country")}, indent=1, default=str))
for k, v in res["strata"].items():
    if "perm_p" in v: print(k, "chi2", v["chi2"], "p", v["perm_p"], "ncat", v["n_categories"])
    else: print(k, v)
