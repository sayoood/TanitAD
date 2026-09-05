import gzip, json
from collections import Counter
from pathlib import Path
A = Path(r"C:/Users/Admin/tanitad-review-20260906/art")
recs = [json.loads(l) for l in gzip.open(A/"s2_labels_parity-v7geom-1_train.jsonl.gz","rt",encoding="utf-8")]
print("cot_source VALUES:", Counter(json.dumps(r.get("cot_source"))[:60] for r in recs).most_common(4))
print("cot_tokens type/len:", Counter("%s/%s" % (type(r.get("cot_tokens")).__name__,
      (len(r["cot_tokens"]) if isinstance(r.get("cot_tokens"), (list,str,dict)) else "-")) for r in recs).most_common(4))
print("alpamayo:", Counter(json.dumps(r.get("alpamayo"))[:30] for r in recs).most_common(3))
print()
print("=> a consumer testing TRUTHINESS of cot_source would read %d/%d as CoT-bearing"
      % (sum(1 for r in recs if r.get("cot_source")), len(recs)))
print("=> the sidecar declares perception coverage 201/2400 (absent on 2199)")
