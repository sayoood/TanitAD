#!/usr/bin/env python3
"""M-SHUF tool (CPU, synthetic side files): derangement, preserved marginals, keys untouched, joins through load_beh_terms, determinism, refusal, rho_report flags a PDMS-function term.  ZZTEST_PASS n."""
import json, os, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mshuf_labels as MS
import scorer_finetune as S
fails = []
def check(n, ok):
    print(("PASS " if ok else "FAIL ") + n)
    if not ok: fails.append(n)
rng = np.random.default_rng(1)
M = 4
def rec(log, tok, step):
    q = rng.uniform(0.3, 1.0, (M, 5)).round(6); a = rng.uniform(size=(M, 5)) > 0.4
    return {"kind": "behaviour", "key": [log, tok, step, 0], "ckpt_step": 400, "label_version": 1, "q": q.tolist(), "act": a.astype(int).tolist(), "terms_version": "behaviour/1"}
recs = [rec(f"log{l}", f"t{l}_{i}", i) for l in range(5) for i in range(1 + (l % 3) * 2)]      # logs with 1, 3, 5, 1, 3 records
new, st = MS.shuffle_records(recs, 7)
same_keys = all(a["key"] == b["key"] and a["ckpt_step"] == b["ckpt_step"] for a, b in zip(recs, new))
check("keys / ckpt_step untouched", same_keys)
check("no record keeps its own payload (derangement inside logs, cyclic over singleton logs)", st["fixed_points"] == 0 and st["singleton_logs"] == 2)
inlog = all(sorted(map(json.dumps, [r["q"] for r in recs if r["key"][0] == lg])) == sorted(map(json.dumps, [r["q"] for r in new if r["key"][0] == lg])) for lg in ("log1", "log2", "log4"))
check("multi-record logs: the payload multiset of each log is preserved (permutation inside the log)", inlog)
check("marginals (activation rate, mean q on active) preserved EXACTLY", MS.marginals(recs) == MS.marginals(new))
check("deterministic for a seed, different for another", [r["q"] for r in MS.shuffle_records(recs, 7)[0]] == [r["q"] for r in new] and [r["q"] for r in MS.shuffle_records(recs, 8)[0]] != [r["q"] for r in new])
one, st1 = MS.shuffle_records(recs[:1], 7)
check("a single record overall stays and is COUNTED as a fixed point", st1["fixed_points"] == 1)
with tempfile.TemporaryDirectory(dir=os.environ.get("A2_TEST_TMP")) as td:
    a, b = os.path.join(td, "in"), os.path.join(td, "out"); os.makedirs(a)
    with open(os.path.join(a, "behaviour_onpolicy_x.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in recs)
    ns = type("N", (), dict(inp=a, out=b, seed=7))()
    rc = MS.cmd_shuffle(ns)
    man = json.load(open(os.path.join(b, "MSHUF_MANIFEST.json")))
    by = {tuple([r["key"][0], r["key"][1], r["key"][2], r["key"][3]]): (400,) for r in recs}
    t0, s0 = S.load_beh_terms(a, by, M); t1, s1 = S.load_beh_terms(b, by, M)
    differs = sum(1 for k in t0 if not np.array_equal(t0[k][0], t1[k][0]))
    check("CLI: rc 0, manifest says marginals_equal and 0 fixed points, the shuffled set joins load_beh_terms exactly like the true one, every record moved",
          rc == 0 and man["marginals_equal"] and man["fixed_points"] == 0 and s0["matched"] == s1["matched"] == len(recs) and differs == len(recs))
    try:
        MS.cmd_shuffle(type("N", (), dict(inp=a, out=a, seed=7))()); refused = False
    except SystemExit:
        refused = True
    check("--out == --in is refused", refused)
# rho_report: a term that is a monotone function of the PDM score is flagged, an independent one is not
terms, pdm = {}, {}
for i in range(60):
    s = rng.uniform(size=8); k = ("l", f"t{i}", 0, 0)
    q = np.stack([s, rng.uniform(size=8), rng.uniform(size=8), rng.uniform(size=8), rng.uniform(size=8)], -1).astype(np.float32)
    terms[k] = (q, np.ones((8, 5), bool)); pdm[k] = s
rep = MS.rho_report(terms, pdm.get)
check("rho_report: q_free (== PDMS) flagged, an independent term is not, q_prod is reported", rep["q_free"]["flag_function_of_pdms"] and not rep["q_gap"]["flag_function_of_pdms"] and rep["q_prod"]["sets"] == 60)
print(("ZZTEST_PASS " + str(len(fails))) if not fails else ("ZZTEST_FAIL " + str(fails)))
sys.exit(1 if fails else 0)
