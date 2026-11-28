"""Discriminate: is the box head WEAK, or are its boxes in the WRONG PLACE (frame/units)?

AP 0.0038 against a uniform-random base rate of 0.124 is 32x BELOW chance. A merely untrained
head should sit NEAR chance. Two candidate explanations, and they differ in the geometry:
  (a) WEAK: predictions spread over a plausible region, GT spread too, they just miss.
  (b) MISPLACED: predictions clustered (e.g. all near the origin) or on a different scale/frame,
      so a uniform-random predictor beats them.
Prints the distributions of predicted and GT centres so the answer is read, not argued.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Projects\TanitAD\taniteval\tools")
import s1_pass as SP  # noqa: E402

A8 = r"C:\Users\Admin\tanitad-caches\a8-occupancy-5k-20260919\run"
corp = SP.Corpus(A8 + r"\config.json",
                 r"D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfB",
                 r"C:\Users\Admin\tanitad-caches\a7-imagenet-knockout-20260919\inputs\s2_labels_v8_eval.jsonl.gz",
                 r"D:\Projects\TanitAD-artifacts\b1-agent-join-3d-20260917\b1eval_agents_3d.jsonl.xz",
                 None)
SP.attach_agent_gt(corp, corp.targs)
mp = SP.ModelPass(corp, A8 + r"\ckpt_5000.pt", "cpu")
wis = [w for w in SP.trainer_windows(corp.ds, 1000) if corp.eligibility(w) is None][:4]
P, G, CONF = [], [], []
for wi in wis:
    item = corp.ds[wi]
    e_i, _ = corp.ds.index[wi]
    out = mp.forward(item, str(corp.clip_ids[e_i]))
    dec = {k: v[0].float().cpu() for k, v in out["perception"]["box_slots"].items()
           if torch.is_tensor(v) and v.dim() >= 2}
    P.append(dec["box"][:, :2].numpy())
    CONF.append(torch.sigmoid(dec["presence_logit"]).numpy())
    v = item["agent_valid"].bool()
    G.append(item["agent_box"][v][:, :2].numpy())
p, g, c = np.concatenate(P), np.concatenate(G), np.concatenate(CONF)


def desc(name, a):
    print("%-10s n=%-5d x: min %7.2f p50 %7.2f max %7.2f | y: min %7.2f p50 %7.2f max %7.2f"
          % (name, len(a), a[:, 0].min(), np.median(a[:, 0]), a[:, 0].max(),
             a[:, 1].min(), np.median(a[:, 1]), a[:, 1].max()))


print("windows:", len(wis), "(the head's decode ranges: x_fwd 60 m, y_half 16 m)")
desc("PRED", p)
desc("GT", g)
print("pred spread (std): x %.2f y %.2f | GT spread: x %.2f y %.2f"
      % (p[:, 0].std(), p[:, 1].std(), g[:, 0].std(), g[:, 1].std()))
print("presence: min %.4f p50 %.4f max %.4f | slots above 0.5: %d of %d"
      % (c.min(), np.median(c), c.max(), int((c > 0.5).sum()), len(c)))
# the discriminating number: nearest-GT distance for predictions vs for UNIFORM random points
d_pred = np.min(np.linalg.norm(p[:, None, :] - g[None, :, :], axis=-1), axis=1)
rng = np.random.default_rng(0)
u = np.stack([rng.uniform(0, 60, 4000), rng.uniform(-16, 16, 4000)], axis=-1)
d_rand = np.min(np.linalg.norm(u[:, None, :] - g[None, :, :], axis=-1), axis=1)
print("nearest-GT distance  PRED: p50 %.2f m, within 2 m: %.3f" % (np.median(d_pred),
                                                                   (d_pred <= 2).mean()))
print("nearest-GT distance  UNIFORM RANDOM: p50 %.2f m, within 2 m: %.3f"
      % (np.median(d_rand), (d_rand <= 2).mean()))
