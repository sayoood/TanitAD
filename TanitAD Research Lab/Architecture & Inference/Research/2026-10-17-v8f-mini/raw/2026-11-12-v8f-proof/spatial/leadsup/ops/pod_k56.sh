#!/usr/bin/env bash
# pod_k56.sh : runs ON THE POD (bash -s). Controls k5 and k6 of card PREREG_V8F_PROOF_ADD24 for the six arms. Reads logs and hashes only.
# k5: the lead term over the last 1,000 steps below the first 1,000 steps; the count of gap-labelled frames above 0 at every logged step.
# k6: the checkpoint the T1-kin dump names is the checkpoint of the arm (path, sha256, checkpoint older than the dump record).
cd /workspace/v8proof/runs || exit 1
for spec in "leadsup_a LS_s0" "leadsup_a LS_s1" "leadsup_b LSD_s0" "leadsup_b LSD_s1" "leadsup_c LSH_s0" "leadsup_c LSH_s1"; do
  set -- $spec; Q=$1; A=$2
  /workspace/venv/bin/python - "$Q" "$A" <<'PY'
import sys, json, os, glob, hashlib
q, a = sys.argv[1], sys.argv[2]
d = os.path.join(q, a)
rows = [json.loads(l) for l in open(os.path.join(d, "train_log.jsonl")) if l.strip().startswith("{")]
rows = [r for r in rows if "lead_sup" in r]
first = [r["lead_sup"] for r in rows if r["step"] <= 1000]
last_step = max(r["step"] for r in rows)
last = [r["lead_sup"] for r in rows if r["step"] > last_step - 1000]
ng = [r.get("lead_sup_n_gap") for r in rows]
ng_ok = all(x is not None and x > 0 for x in ng)
m = lambda v: sum(v) / max(1, len(v))
k5 = (m(last) < m(first)) and ng_ok
ck = os.path.join(d, "ckpt_last.pt")
h = hashlib.sha256()
with open(ck, "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""):
        h.update(b)
t1 = glob.glob(os.path.join(q, "t1", a, "t1v8f_%s_*.json" % a))
rec = json.load(open(t1[0])) if t1 else {}
named = rec.get("ckpt", "")
shas = {k: v for k, v in rec.items() if "sha" in k.lower() and isinstance(v, str)}
older = bool(t1) and os.path.getmtime(ck) <= os.path.getmtime(t1[0])
k6 = bool(t1) and os.path.abspath(named) == os.path.abspath(ck) and older and all(v[:64] == h.hexdigest() for k, v in shas.items() if "ckpt" in k.lower())
print("K56 %s last_step=%d n_rows=%d lead_first1000=%.4f lead_last1000=%.4f n_gap_min=%s k5=%s ckpt_sha=%s t1_names_ckpt=%s ckpt_older=%s t1_sha_keys=%s k6=%s" % (
    a, last_step, len(rows), m(first), m(last), min([x for x in ng if x is not None] or [None]), "PASS" if k5 else "FAIL",
    h.hexdigest()[:16], os.path.abspath(named) == os.path.abspath(ck), older, sorted(shas), "PASS" if k6 else "FAIL"))
PY
done
