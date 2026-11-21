"""E-SEED-2 stage 1 — the ENVIRONMENT bank, on the 24 UUID-matched val clips.

⛔ WHY THIS BANK EXISTS AND E-SEED-1's DOES NOT SUFFICE. E-SEED-1 scored `speed`
and `yaw_rate` -- BOTH EGO targets. E-DEC-17 (MEASURED, `frzrand`) established
that a frozen RANDOM encoder carries the BEST speed of three arms (+0.3552), so
`no ego number may be cited as evidence that an objective worked`. A seed-transfer
verdict resting on speed alone is therefore inadmissible as a statement about
SCENE content. This bank adds `n_agents`, the register's environment target.

⛔ THE JOIN, AND WHY IT IS SAFE HERE. The E-SEED-1 epcache
(`physicalai-val-bb543bdf7836`) stores `episode_id` as an INTEGER and carries no
clip UUID, so it cannot be joined to the obstacle labels at all. The v2 cache
`physicalai-val-w120-256x640cyl` stores `clip_id` as the UUID and its 24 clips are
a SUBSET of the 40 in `val40_agents.jsonl` (MEASURED: |val24 & val40| = 24).
The join is therefore by UUID and by frame index, and BOTH are asserted.

⚠️ The buffer is named `jpeg_buf`; its `codec` field says `png`. The codec field
decides the decoder (CLAUDE.md: trusting the NAME filled a 2.76 GB memmap with
zeros once already). Content is asserted, never the exit code.

`n_agents` DEFINITION (stated because it is not inherited): the number of
`obstacle.offline` cuboids present in that frame's record, ALL classes, no range
or occlusion filter. It is NOT asserted to match the definition behind the
register's E-DEC-8 +0.3274 -- cross-panel comparison is refused; only WITHIN-panel
arm contrasts are quoted.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

V2DIR = Path(r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901"
             r"\sp2\cache\physicalai-val-w120-256x640cyl")
AGENTS = Path(r"C:\Users\Admin\tanitad-caches\val40-obstacle-20260818"
              r"\join\val40_agents.jsonl")
OUT = Path(__file__).resolve().parent / "eseed2"
OUT.mkdir(exist_ok=True)

STRIDE = 2
DT = 0.1


def load_agents() -> dict:
    per = {}
    with open(AGENTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            per.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r["agents"]
    return per


def main() -> int:
    agents = load_agents()
    clips = sorted(p for p in V2DIR.glob("*.v2ep.pt"))
    print(f"[bank] {len(clips)} v2ep clips, {len(agents)} labelled clips")

    frames, rows, lab, covs = [], [], [], []
    for ci, p in enumerate(clips):
        d = torch.load(p, map_location="cpu", weights_only=False)
        cid = d["clip_id"]
        if d.get("codec") != "png":
            print(f"REFUSE: {cid} codec={d.get('codec')!r}, expected png")
            return 3
        if cid not in agents:
            print(f"REFUSE: {cid} has no agent labels")
            return 4
        buf = d["jpeg_buf"].numpy()
        ln = d["jpeg_len"].numpy()
        pos = d["poses"].numpy()
        T = len(ln)
        A = agents[cid]
        # ⛔ ALIGNMENT ASSERTION. The label stream is SHORTER than the cache
        # (MEASURED: 198/201 on clip 1, 173/201 on clip 17 -- the join
        # drops trailing frames). Rows are taken ONLY where a label exists, and
        # the coverage fraction is asserted so a badly-joined clip REFUSES
        # rather than contributing a handful of rows unnoticed.
        cov = len(A) / float(T)
        if cov < 0.75:
            print(f"REFUSE: {cid} label coverage {cov:.3f} ({len(A)}/{T})")
            return 5
        covs.append(cov)
        yaw = np.unwrap(pos[:, 2].astype(np.float64))
        yr = np.gradient(yaw, DT).astype(np.float32)
        v = pos[:, 3].astype(np.float32)
        offs = np.concatenate([[0], np.cumsum(ln)])
        for t in range(0, T, STRIDE):
            if t not in A:
                continue
            b = buf[offs[t]:offs[t + 1]].tobytes()
            im = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
            frames.append(np.ascontiguousarray(im.transpose(2, 0, 1)))
            rows.append((ci, t))
            lab.append((float(len(A[t])), v[t], yr[t]))
        del d, buf
        print(f"  ..{ci+1}/{len(clips)} {cid[:8]} T={T} rows={len(rows)}",
              flush=True)

    X = np.stack(frames).astype(np.uint8)
    L = np.asarray(lab, dtype=np.float32)
    R = np.asarray(rows, dtype=np.int32)

    fm, nz = float(X.mean()), float((X != 0).mean())
    print(f"[bank] frames {X.shape} mean={fm:.3f} nonzero_frac={nz:.4f}")
    if fm < 1.0 or nz < 0.5:
        print("REFUSE: frame bank looks zero-filled (the memmap trap)")
        return 6
    names = ("n_agents", "v_mps", "yaw_rate")
    for j, nm in enumerate(names):
        c = L[:, j]
        print(f"[bank] {nm:9s} n={len(c)} mean={c.mean():+.4f} "
              f"std={c.std():.4f} min={c.min():.3f} max={c.max():.3f} "
              f"finite={bool(np.isfinite(c).all())}")
        if not np.isfinite(c).all() or c.std() <= 0:
            print(f"REFUSE: label {nm} degenerate")
            return 7

    np.save(OUT / "frames_u8.npy", X)
    np.save(OUT / "labels.npy", L)
    np.save(OUT / "rows.npy", R)
    (OUT / "bank_meta.json").write_text(json.dumps(dict(
        n_rows=int(L.shape[0]), n_clips=len(clips), stride=STRIDE, dt=DT,
        v2dir=str(V2DIR), agents=str(AGENTS),
        clip_ids=[p.name.split(".v2ep")[0] for p in clips],
        label_coverage_min=float(min(covs)), label_coverage_mean=float(sum(covs)/len(covs)),
        label_cols=list(names), frame_geometry="256x640 cylindrical RGB",
        n_agents_definition=("count of obstacle.offline cuboids in the frame, "
                             "ALL classes, no range or occlusion filter"),
        frame_mean=fm, nonzero_frac=nz,
        evidence_class="MEASURED (ours; local v2 cache + val40 obstacle join)",
    ), indent=2), encoding="utf-8")
    print(f"[bank] wrote {OUT} ({X.nbytes/2**30:.2f} GiB frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
