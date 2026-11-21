import gzip, json
from collections import Counter
from pathlib import Path
A = Path(r"C:/Users/Admin/tanitad-review-20260906/art")
recs = [json.loads(l) for l in gzip.open(A/"s2_labels_parity-v7geom-1_train.jsonl.gz","rt",encoding="utf-8")]
print("records:", len(recs), " CONTROL keys[0]:", sorted(recs[0].keys())[:14])
lat = Counter(r["a_tac"]["lat"] for r in recs)
print("a_tac.lat census:", dict(lat))
turns = lat.get("TURN_L",0)+lat.get("TURN_R",0)
print("TURN_L+TURN_R = %d of %d = %.2f %% of the corpus" % (turns, len(recs), 100*turns/len(recs)))
# turn_suppression: present? does it ever fire on parity?
ts = [r.get("turn_suppression") for r in recs]
print("turn_suppression present on %d/%d records" % (sum(1 for x in ts if x is not None), len(recs)))
fired = [x for x in ts if isinstance(x, dict) and (x.get("applied") or x.get("suppressed") or x.get("fired"))]
print("turn_suppression FIRED on:", len(fired), " example:", json.dumps(ts[0])[:300] if ts[0] is not None else None)
# CoT presence
cot = sum(1 for r in recs if r.get("cot_tokens") or r.get("cot_source"))
print("records carrying CoT (the contesting evidence):", cot)
turns_no_cot = sum(1 for r in recs if r["a_tac"]["lat"] in ("TURN_L","TURN_R")
                   and not (r.get("cot_tokens") or r.get("cot_source")))
print("geometric turns WITHOUT CoT:", turns_no_cot)
print()
est = 30
print("THE DENOMINATOR QUESTION:")
print("  quoted:   ~%d of %d labels = %.2f %% of the CORPUS" % (est, len(recs), 100*est/len(recs)))
print("  but they are ALL in the turn class: %d of %d turns = %.2f %% of the TURN CLASS"
      % (est, turns, 100*est/turns))
print("  ratio: %.1fx" % ((est/turns)/(est/len(recs))))
print()
print("ONE LABEL PER CLIP:", len(recs), "records for", len({r["clip_id"] for r in recs}), "distinct clips")
print("  t0_s values:", Counter(r.get("t0_s") for r in recs))
print("  horizon values:", Counter(str(r.get("horizon")) for r in recs).most_common(3))
