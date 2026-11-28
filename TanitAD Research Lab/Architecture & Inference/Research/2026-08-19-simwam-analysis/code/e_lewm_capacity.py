"""E-LEWM-1 CAPACITY CONTROL — can this harness produce a decodable latent AT ALL?

⛔ WHY THIS EXISTS. The pre-registered `lewm` gate failed TWICE:

    gate 1 (tick = 1 frame)   lead_gap -0.0130   b/w  3.22   sigreg 0.551
    gate 2 (tick = 10 frames) lead_gap -0.0257   b/w 16.25   sigreg 1.636

Both times SIGReg converged and NOTHING decoded, and the corrected tick made the
between/within-episode ratio WORSE. Two readings remain open and they demand
opposite responses:

  (A) the OBJECTIVE is the problem  — this encoder, on this data, CAN learn a
      decodable representation, but LeWM's objective does not induce one here.
  (B) the HARNESS is the problem    — 5.4 M params on 26 k frames of real
      driving simply cannot, and no objective would have worked. LeWM's Push-T /
      Reacher are visually trivial next to a road.

⭐ THE DISCRIMINATOR: train the SAME encoder with DIRECT SUPERVISION on the probe
targets. Nothing else changes — same architecture, same frames, same optimiser,
same steps, same episode-disjoint split.

  * it decodes  -> the harness has the capacity and the data. (A). The gate
                   failure is about the OBJECTIVE and the ablation is meaningful
                   once the replication is fixed.
  * it does NOT -> (B). The whole small-scale premise is wrong, E-LEWM-1 cannot
                   answer the question at this scale, and that must be reported
                   as such rather than dressed up as a finding about v6.

⚠️ THIS IS A CONTROL, NOT AN ARM. A supervised encoder is not a world model and
its number is never a WM result. It bounds what the harness could possibly show.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path.cwd() / "stack"))

import e_lewm_ablate as E  # noqa: E402
import e_trunk2_probe as P  # noqa: E402

#: supervised on the SAME quantities the probe reads, so the control answers
#: exactly the question the probe asks.
SUP = ("lead_gap_m", "left_occupied", "right_occupied", "ego_speed")


def main(steps=5000, batch=64, seed=0, dev=None) -> None:
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    frames = np.load(E.CACHE / "frames.npy", mmap_mode="r")
    clips = json.loads((E.CACHE / "clips.json").read_text(encoding="utf-8"))
    fmap = {(c["clip_id"], f): c["start"] + f for c in clips for f in range(c["n"])}

    tgt = {}
    for line in P.TARGETS.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            tgt[(r["clip_id"], int(r["frame_idx"]))] = r
    keys = [tuple(k) for k in
            json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    eg = P.ego_features(keys)
    for i, k in enumerate(keys):
        tgt.setdefault(k, {})["ego_speed"] = float(eg[i, 0])

    # rows + target matrix over frames that HAVE every supervised target
    rows, Y = [], []
    for k, gi in fmap.items():
        r = tgt.get(k)
        if not r:
            continue
        v = [r.get(t) for t in SUP]
        if any(x is None or not np.isfinite(x) for x in v):
            continue
        rows.append(gi); Y.append(v)
    rows = np.array(rows); Y = np.asarray(Y, dtype=np.float32)
    mu, sd = Y.mean(0), Y.std(0) + 1e-6
    Yn = torch.from_numpy((Y - mu) / sd)
    print(f"supervised control: {len(rows)} frames, targets {SUP}", flush=True)

    enc = E.Encoder(192).to(dev)
    head = torch.nn.Linear(192, len(SUP)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=3e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(steps):
        j = torch.randint(0, len(rows), (batch,), generator=g).numpy()
        o = torch.from_numpy(np.asarray(frames[rows[j]])).to(dev).float() / 255.0
        y = Yn[j].to(dev)
        loss = F.mse_loss(head(enc(o)), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"    [sup/s{seed}] step {step} loss {float(loss.detach()):.5f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    enc.eval()
    Z = np.zeros((len(frames), 192), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(frames), 256):
            b = torch.from_numpy(np.asarray(frames[i:i+256])).to(dev).float()/255.0
            Z[i:i+256] = enc(b).cpu().numpy()
    np.save(E.CACHE / f"z_supervised_s{seed}.npy", Z)
    rec = [{"arm": "supervised-CONTROL", "seed": seed,
            "n_params": sum(p.numel() for p in enc.parameters()),
            "config": {"d_latent": 192, "supervised_on": list(SUP)},
            "history": [{"step": steps - 1, "sigreg": float("nan"),
                         "pred": float(loss.detach())}],
            "latents": f"z_supervised_s{seed}.npy",
            "wall_s": round(time.time() - t0)}]
    prev = []
    tp = SP / "e_lewm_train.json"
    if tp.exists():
        prev = [r for r in json.loads(tp.read_text(encoding="utf-8"))
                if r["arm"] != "supervised-CONTROL"]
    tp.write_text(json.dumps(prev + rec, indent=1), encoding="utf-8")
    print(f"-> {tp}")


if __name__ == "__main__":
    main()
