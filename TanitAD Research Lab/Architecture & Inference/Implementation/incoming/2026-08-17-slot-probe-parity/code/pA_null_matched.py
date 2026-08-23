"""PA — a random-latent null on the EXACT windows the real arm was scored on.

⛔ WHY THIS EXISTS. `spX_fake_cache.py` builds its own window list from the join,
so its null lands on 3,089 windows while the real caches land on 2,721. Comparing
"arm 7.169 vs null 6.338" across two different window sets is NOT a paired
comparison, and this report's most striking claim — that the trained latent is
WORSE than noise — would rest on it.

This takes the REAL cache and replaces ONLY the `cells` tensor with noise, keeping
every window, every target, every clip_id and the split. The comparison then runs
through `paired_episode_cluster_bootstrap` on identical windows, which is what the
estimator rule requires.
"""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
blob = torch.load(src, map_location="cpu", weights_only=False)
rows, meta = blob["rows"], dict(blob["meta"])
g = torch.Generator().manual_seed(0)
n_c, d_r = rows[0]["cells"].shape
for r in rows:
    r["cells"] = torch.randn(n_c, d_r, generator=g).to(torch.float16)
    r["tokens"] = None
meta["run_stamp"] = "RANDOM-LATENT-NULL-MATCHED@" + str(meta.get("step"))
meta["_evidence_class"] = ("SYNTHETIC (window-matched pipeline null; the REAL cache "
                           "with `cells` replaced by noise — same windows, same "
                           "targets, same split)")
meta["tokens_banked"] = False
torch.save({"rows": rows, "meta": meta}, dst)
print("wrote", dst, "rows", len(rows), "cells", (n_c, d_r))
