"""X7 coverage probe (read-only). Intersects the banked B1 eval lead block with the eval139 clip set.
Inputs, both read from the tip 8522e49 with `git show` (GIT_DIR=/c/Users/Admin/tanitad-push/.git):
  TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz
  TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D1_label_census/tables/label_truth_eval139_clips.json
sha12 = sha256(clip_id)[:12] (stack/scripts/build_v9_labels.py:65-66)."""
import collections, hashlib, json, sys
import numpy as np
npz, lt = sys.argv[1], sys.argv[2]
ids139 = {r["sha12"] for r in json.load(open(lt, encoding="utf-8"))}
z = np.load(npz, allow_pickle=True)
cid = z["clip_id"].astype(str)
clips = sorted(set(cid))
sh = {c: hashlib.sha256(c.encode()).hexdigest()[:12] for c in clips}
in139 = [c for c in clips if sh[c] in ids139]
print("eval139 clips:", len(ids139), "| lead-block clips:", len(clips), "| intersection:", len(in139),
      "| eval139 missing from block:", len(ids139 - set(sh.values())), "| block clips outside eval139:", len(clips) - len(in139))
m = np.isin(cid, in139)
print("raw 10 Hz frame rows in eval139 clips:", int(m.sum()), dict(collections.Counter(z["state"][m].astype(str))))
print("block horizon: K=%d, ts_rel_s=%s" % (z["leads"].shape[1], np.round(z["ts_rel_s"], 2).tolist()))
