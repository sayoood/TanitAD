#!/usr/bin/env python3
"""D-VT1 step 7 — CAN VISION SET THE TARGET SPEED? The one question §7.1 left open.

`vt_four_families.py` established that on **ego state alone** nothing beats the free
baseline: a CE classifier over the 23 VTARGET bands scores 0.2465 against 0.4066
for literally repeating `v0`'s band. That makes `refc1`'s `speed_cls` a dead
parameter *on ego inputs*. The remaining question — and the whole case for a
PREDICTED goal — is whether the **image** carries what the ego past does not.

This asks it directly, on REF-C-base's frozen fan, leave-one-episode-out, with the
band as the label because the band is what a conditioning input consumes.

| arm | features | inference-legal? |
|---|---|---|
| `identity_v0_band` | none — emit `vtarget_band(v0)` | ✅ free, 0 params |
| `majority` | none — emit the modal band | ✅ the floor |
| `v0` | ego speed | ✅ |
| `past` | causal ego speed over [t-0.7 s, t] | ✅ |
| `img` | `pooled` (704-d) | ✅ **the deployable question** |
| `img_v0` | `pooled` + `v0` | ✅ |

⛔ Frozen trunk, no `ego_dropout`. Same caveat as the F1 probe: this is a readout,
optimistic, and it is evidence for a GPU-day rather than a substitute for one.
⛔ Estimator: `paired_episode_cluster_bootstrap`, unit = val episode.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "stack"))

from taniteval.ci import paired_episode_cluster_bootstrap        # noqa: E402
from taniteval.lead_source import window_last_indices            # noqa: E402
from tanitad.lake.vocab import vtarget_band                       # noqa: E402
from tanitad.lake.vtarget import VT_GUARD_STEPS, vtarget_guarded   # noqa: E402

PAST_STEPS = 7
STEPS = 600
LR = 5e-2
BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 15.0),
         (15.0, np.inf)]


def loeo_clf(x, y, eid, n_cls, *, seed=0):
    torch.set_num_threads(1)
    out = np.empty(len(y), dtype=np.int64)
    for e in np.unique(eid):
        te = eid == e
        tr = ~te
        mu, sd = x[tr].mean(0), x[tr].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        xt = torch.tensor((x[tr] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y[tr])
        torch.manual_seed(seed)
        net = torch.nn.Linear(xt.shape[1], n_cls)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
        for _ in range(STEPS):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(xt), yt).backward()
            opt.step()
        with torch.no_grad():
            out[te] = net(torch.tensor((x[te] - mu) / sd,
                                       dtype=torch.float32)).argmax(-1).numpy()
    return out


def main(substrate: Path, poses: Path, out_json: Path):
    t0 = time.time()
    d = torch.load(substrate, map_location="cpu", weights_only=False)
    pooled = d["pooled"].numpy().astype(np.float64)
    v0 = d["v0"].numpy().astype(np.float64)
    eid = np.array(d["eid"])

    z = np.load(poses, allow_pickle=True)
    meta = json.loads(bytes(z["_meta_json"]).decode())
    P = {m["episode_id"]: z[m["file"].replace('.pt', '')
                            + "__poses"].astype(np.float64) for m in meta}
    vt = np.zeros(len(eid))
    ok = np.zeros(len(eid), bool)
    past = np.zeros((len(eid), PAST_STEPS + 1))
    align = 0.0
    for e in dict.fromkeys(eid.tolist()):
        sel = np.where(eid == e)[0]
        v = P[e][:, 3]
        last = window_last_indices(P[e].shape[0], stride=5)
        assert len(last) == len(sel)
        align = max(align, float(np.abs(v0[sel] - v[last]).max()))
        a, m, _l, _ = vtarget_guarded(v, last, guard_steps=VT_GUARD_STEPS,
                                      min_lookahead=50)
        vt[sel], ok[sel] = a, m
        for i, ell in enumerate(last):
            w = v[max(0, ell - PAST_STEPS):ell + 1]
            if w.size < PAST_STEPS + 1:
                w = np.concatenate([np.full(PAST_STEPS + 1 - w.size, w[0]), w])
            past[sel[i]] = w
    assert align == 0.0, f"substrate/pose join not bit-exact: {align}"

    sel = ok
    y_tok = np.array([vtarget_band(a) for a in vt[sel]])
    vocab = sorted(set(y_tok))
    idx = {t: i for i, t in enumerate(vocab)}
    y = np.array([idx[t] for t in y_tok])
    e_sel = eid[sel]
    pb = np.column_stack([past[sel], np.diff(past[sel], axis=1)])

    preds = {
        "identity_v0_band": np.array(
            [idx.get(vtarget_band(a), -1) for a in v0[sel]]),
        "majority": np.full(len(y), int(np.bincount(y).argmax())),
        "v0": loeo_clf(v0[sel][:, None], y, e_sel, len(vocab)),
        "past": loeo_clf(pb, y, e_sel, len(vocab)),
        "img": loeo_clf(pooled[sel], y, e_sel, len(vocab)),
        "img_v0": loeo_clf(np.column_stack([pooled[sel], v0[sel]]), y, e_sel,
                           len(vocab)),
    }
    hits = {k: (p == y).astype(float) for k, p in preds.items()}
    out = {
        "_what": ("can VISION set the target speed? band top-1 over the 23 "
                  "VTARGET tokens, leave-one-episode-out, REF-C-base frozen fan"),
        "_join_proof": {"max_abs_v0_minus_pose_speed": align},
        "_estimator": ("paired_episode_cluster_bootstrap (taniteval.ci), unit = "
                       "val episode, B=2000. NEVER overlapping_holdout_se"),
        "_frozen_trunk_caveat": ("readout on a FROZEN trunk with no ego_dropout — "
                                 "optimistic; evidence for a GPU-day, not a "
                                 "substitute for one"),
        "n_windows": int(sel.sum()), "n_windows_total": int(len(eid)),
        "n_episodes": int(len(set(e_sel.tolist()))),
        "n_bands_present": len(vocab),
        "band_top1": {k: round(float(v.mean()), 4) for k, v in hits.items()},
        "paired_vs_identity_v0_band": {
            k: paired_episode_cluster_bootstrap(v, hits["identity_v0_band"],
                                                e_sel, n_boot=2000, seed=0)
            for k, v in hits.items() if k != "identity_v0_band"},
        "by_speed": {},
        "_runtime_s": None,
    }
    for k, v in out["paired_vs_identity_v0_band"].items():
        v["_reads"] = ("delta > 0 => this arm beats the FREE baseline of "
                       "repeating the current speed's band. A goal head that "
                       "does not separate above 0 is a dead parameter.")
    vv = v0[sel]
    for lo, hi in BANDS:
        m = (vv >= lo) & (vv < hi)
        name = f"{lo:g}-{'inf' if np.isinf(hi) else f'{hi:g}'}"
        n_ep = len(set(e_sel[m].tolist()))
        if m.sum() < 30 or n_ep < 5:
            out["by_speed"][name] = {"status": "UNPOWERED", "n": int(m.sum()),
                                     "n_episodes": n_ep}
            continue
        out["by_speed"][name] = {
            "n": int(m.sum()), "n_episodes": n_ep,
            "band_top1": {k: round(float(v[m].mean()), 4)
                          for k, v in hits.items()}}
    out["_runtime_s"] = round(time.time() - t0, 1)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["band_top1"].items():
        ci = out["paired_vs_identity_v0_band"].get(k)
        tail = (f"  vs free: {ci['delta']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}] "
                f"sep={ci['separated']}") if ci else ""
        print(f"  {k:18s} band_top1={v:.4f}{tail}")
    print(f"wrote {out_json} in {out['_runtime_s']}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
