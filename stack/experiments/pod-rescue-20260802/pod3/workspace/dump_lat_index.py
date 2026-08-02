"""Dump {tag -> episode_id, domain, T, src} for a latent cache dir.

Used to test whether the eval-pod v3 latent cache (/root/idm2/lat, 104 eps) and
the pod3 idm-proof latent cache (/workspace/tmp/idm/latents, 770 eps) refer to
the SAME underlying episodes. If pod3's set is a superset, the v3 val split can
be reconstructed on pod3 by episode_id and a paired read against A0 is possible.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch


def main(latdir: str, out: str) -> None:
    d = Path(latdir)
    recs = []
    for p in sorted(d.glob("*.pt")):
        try:
            o = torch.load(p, weights_only=False, map_location="cpu")
        except Exception as e:  # noqa: BLE001
            recs.append({"tag": p.stem, "error": repr(e)[:200]})
            continue
        z = o.get("z")
        rec = {
            "tag": p.stem,
            "episode_id": int(o["episode_id"]) if "episode_id" in o else None,
            "domain": o.get("domain"),
            "src": str(o.get("src", "")),
            "T": int(z.shape[0]) if z is not None and hasattr(z, "shape") else None,
            "zdim": int(z.shape[1]) if z is not None and getattr(z, "ndim", 0) == 2 else None,
            "keys": sorted(k for k in o.keys() if k != "z"),
        }
        recs.append(rec)
    Path(out).write_text(json.dumps({"latdir": latdir, "n": len(recs), "records": recs}, indent=1))
    doms: dict[str, int] = {}
    for r in recs:
        doms[str(r.get("domain"))] = doms.get(str(r.get("domain")), 0) + 1
    print("WROTE", out, "n=", len(recs), "domains=", doms, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
