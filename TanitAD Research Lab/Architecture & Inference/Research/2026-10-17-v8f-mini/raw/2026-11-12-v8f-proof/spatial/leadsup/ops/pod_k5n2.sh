cd /workspace/v8proof/runs || exit 1
/workspace/venv/bin/python - <<'PY'
import json, os
specs = [("leadsup_a","LS_s0"),("leadsup_a","LS_s1"),("leadsup_b","LSD_s0"),("leadsup_b","LSD_s1"),("leadsup_c","LSH_s0"),("leadsup_c","LSH_s1")]
m = lambda v: sum(v)/max(1,len(v))
for q,a in specs:
    rows = [json.loads(l) for l in open(os.path.join(q,a,"train_log.jsonl")) if l.strip().startswith("{")]
    rows = [r for r in rows if "lead_sup" in r]
    last = max(r["step"] for r in rows)
    F = [r for r in rows if r["step"] <= 1000]; L = [r for r in rows if r["step"] > last-1000]
    pa = all(r.get("lead_sup_n_gap",0) > 0 for r in rows)
    first_val = rows[0]["lead_sup"]; pb = first_val > m([r["lead_sup"] for r in L])
    gf, gl = m([r["lead_sup_gap"] for r in F]), m([r["lead_sup_gap"] for r in L]); pc = gl < gf
    print("K5N2 %s a=%s b=%s (step %d value %.4f, last1000 %.4f) c=%s (gap first1000 %.5f, last1000 %.5f) bce first1000 %.4f last1000 %.4f clos first1000 %.5f last1000 %.5f" % (
        a, pa, pb, rows[0]["step"], first_val, m([r["lead_sup"] for r in L]), pc, gf, gl,
        m([r["lead_sup_bce"] for r in F]), m([r["lead_sup_bce"] for r in L]), m([r["lead_sup_clos"] for r in F]), m([r["lead_sup_clos"] for r in L])))
PY
tail -n 4 /workspace/v8proof/distrule/queue.log | cut -c1-200
