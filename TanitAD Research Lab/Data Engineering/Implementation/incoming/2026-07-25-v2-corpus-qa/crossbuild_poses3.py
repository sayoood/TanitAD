"""Cross-build pose validation, EXACT clip_id keying (offset-free).

`ep_NNNNN.pt` carries only `episode_id = int.from_bytes(clip_id[:4])`, and the
file index is NOT the discovery index (measured offset +2 on this cache), so the
mapping is resolved the safe way: episode_id -> the clip_id among the 3,000
discovered parity clips that has that 4-char prefix, kept ONLY when that prefix
is UNIQUE in the discovered set. Matches are then exact full-clip_id equality
against the v2 manifest -- a shared prefix can never fake one. READ-ONLY.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--phase0-root", required=True)
ap.add_argument("--parity-cache", required=True)
ap.add_argument("--v2manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stack", default="/workspace/tmp/qa_stack")
a = ap.parse_args()
sys.path.insert(0, a.stack)
sys.path.insert(1, os.path.join(a.stack, "scripts"))
import refb_labels as rl                                                 # noqa: E402
from tanitad.data.physicalai import discover_r0_clips                    # noqa: E402

eid_of = lambda c: int.from_bytes(c.encode()[:4].ljust(4, b"\0"), "big")  # noqa: E731
disc = [c["clip_id"] for c in discover_r0_clips(a.phase0_root)]
by_eid: dict[int, list[str]] = {}
for c in disc:
    by_eid.setdefault(eid_of(c), []).append(c)
uniq_eid = {e: v[0] for e, v in by_eid.items() if len(v) == 1}

man = torch.load(a.v2manifest, map_location="cpu", weights_only=False)
v2 = {fn.split(".v2ep")[0]: man["poses"][i] for i, fn in enumerate(man["files"])}

R = {"parity_discovered": len(disc), "parity_eids_unique_prefix": len(uniq_eid),
     "v2_clips": len(v2), "parity_eps_on_disk": 0, "eid_ambiguous_skipped": 0,
     "resolved": 0, "shared_with_v2": 0, "compared": 0, "T_equal": 0,
     "poses_bit_identical": 0, "poses_allclose_1e3": 0,
     "maneuver_hist_identical": 0, "max_abs_dxy": 0.0, "max_abs_dyaw": 0.0,
     "max_abs_dv": 0.0, "diffs": []}
for p in sorted(glob.glob(os.path.join(a.parity_cache, "ep_*.pt"))):
    R["parity_eps_on_disk"] += 1
    d = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
    e = int(d["episode_id"])
    cid = uniq_eid.get(e)
    if cid is None:
        R["eid_ambiguous_skipped"] += 1
        continue
    R["resolved"] += 1
    pv = v2.get(cid)
    if pv is None:
        continue
    R["shared_with_v2"] += 1
    pp = d["poses"].float().clone()
    R["compared"] += 1
    if pp.shape != pv.shape:
        R["diffs"].append({"clip_id": cid, "ep": os.path.basename(p),
                           "T_parity": list(pp.shape), "T_v2": list(pv.shape)})
        continue
    R["T_equal"] += 1
    R["poses_bit_identical"] += bool(torch.equal(pp, pv))
    R["poses_allclose_1e3"] += bool(torch.allclose(pp, pv, atol=1e-3, rtol=0))
    dxy = float((pp[:, :2] - pv[:, :2]).abs().max())
    dyaw = float((pp[:, 2] - pv[:, 2]).abs().max())
    dv = float((pp[:, 3] - pv[:, 3]).abs().max())
    R["max_abs_dxy"] = max(R["max_abs_dxy"], dxy)
    R["max_abs_dyaw"] = max(R["max_abs_dyaw"], dyaw)
    R["max_abs_dv"] = max(R["max_abs_dv"], dv)
    hp = torch.bincount(rl.maneuver_labels(pp, 20), minlength=5)
    hv = torch.bincount(rl.maneuver_labels(pv, 20), minlength=5)
    same = bool(torch.equal(hp, hv))
    R["maneuver_hist_identical"] += same
    if not torch.equal(pp, pv):
        R["diffs"].append({"clip_id": cid, "ep": os.path.basename(p),
                           "max_dxy": round(dxy, 6), "max_dyaw": round(dyaw, 6),
                           "max_dv": round(dv, 6), "man_hist_same": same})
R["n_diffs"] = len(R["diffs"])
R["diffs"] = R["diffs"][:15]
json.dump(R, open(a.out, "w"), indent=2)
print(json.dumps(R, indent=2), flush=True)
