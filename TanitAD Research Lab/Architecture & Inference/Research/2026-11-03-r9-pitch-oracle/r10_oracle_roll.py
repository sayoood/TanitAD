#!/usr/bin/env python3
"""r10_oracle_roll.py -- r9_e1_dump.py with the R10 PITCH ORACLE on the BEV lift (T0 diagnostic; egomotion pitch at inference is NOT admissible).

    r10_oracle_roll.py --arm static|zero|H|C --table pitch_table_eval139.npz [--scale 1.0] <every r9_e1_dump.py argument>

``static``: the lift geometry is the tip's (the hook is never entered). ``zero``: every window is rebuilt through ``build_lift_geometry(..., pitch_delta_rad=0.0)`` and
asserted ``torch.equal`` to the tip's cached geometry (the bit-identity control inside the real model). ``H`` / ``C``: the NEWEST frame's pitch target (degrees, nose-down
positive; r10_pitch_table.py) is composed onto the clip's static extrinsic (bev_lift.compose_pitch_delta) before the ground intersection, per WINDOW. The map hires lift
is the ONE lift of the refcv8 forward (the near-lift geometry is derived from it), so the box path sees the same geometry. Nothing in r9_e1_dump.py is edited: the
collate and the loss entry are wrapped, and the model's ``_lift_bank_hires`` is replaced by an ``OracleBank`` that delegates everything else to the original.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve()
HERE = SCRIPT.parent
STACK = HERE.parents[3] / "stack"
for p in (str(STACK), str(STACK / "scripts"), str(STACK.parent / "taniteval")):
    if p not in sys.path:
        sys.path.insert(0, p)


class OracleBank:
    """Delegates to the original HiresLiftGeometryBank; for a window with a non-zero pitch delta it builds the geometry with the hook (uncached, 6 ms)."""

    def __init__(self, base, verify_zero=False):
        self.base, self.delta_rad, self.verify_zero = base, 0.0, bool(verify_zero)
        self.n_hook, self.n_off, self.n_verified = 0, 0, 0

    def __getattr__(self, name):
        return getattr(self.__dict__["base"], name)

    def __len__(self):
        return len(self.base)

    def for_episodes(self, ep_ids, device=None):
        from tanitad.models.bev_lift import build_lift_geometry
        ids = [int(x) for x in (ep_ids.tolist() if torch.is_tensor(ep_ids) else list(ep_ids))]
        if len(ids) != 1:
            raise SystemExit("[r10] the oracle bank is per WINDOW: batch size 1 only")
        b = self.base
        if self.delta_rad == 0.0 and not self.verify_zero:
            self.n_off += 1
            return b.for_episodes(ids, device=device)
        g = build_lift_geometry(b._extr[ids[0]], frame=b.frame, stride=b.stride, heights_m=b.heights_m, grid=b.grid_spec,
                                observed=b._observed, pitch_delta_rad=self.delta_rad)
        if self.delta_rad == 0.0:                                     # the bit-identity control
            g0, v0 = b.geometry(ids[0])
            assert torch.equal(g.grid, g0) and torch.equal(g.valid, v0), "pitch_delta_rad=0.0 is NOT bit-identical to the tip geometry"
            self.n_verified += 1
        else:
            self.n_hook += 1
        grid, valid = g.grid.unsqueeze(0), g.valid.unsqueeze(0)
        return (grid.to(device), valid.to(device)) if device is not None else (grid, valid)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=("static", "zero", "H", "C"))
    ap.add_argument("--table", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the table delta (1.0 = the oracle)")
    a, rest = ap.parse_known_args()
    import r9_e1_dump as D
    from tanitad.data.v2_dataset import load_or_build_manifest, stable_episode_id
    from tanitad.eval import refcv7_loader as L
    L.bootstrap()
    tr = L.trainer()
    tab = np.load(a.table, allow_pickle=False)
    table_sha = hashlib.sha256(Path(a.table).read_bytes()).hexdigest()
    manifest_ids = [str(c) for c in load_or_build_manifest(L.PATH_REMAP["--eval-cache"], verbose=False)["clip_id"]]
    sha_of_sid = {int(stable_episode_id(c)): L.sha12(c) for c in manifest_ids}
    st = {"bank": None, "n_windows": 0, "n_no_table": 0, "abs_delta_deg": [], "frames": 0}

    orig_build = L.build_model

    def build_model(*aa, **kk):
        out = orig_build(*aa, **kk)
        model = out[0]
        base = getattr(model, "_lift_bank_hires", None)
        if base is None:
            raise SystemExit("[r10] the model has no _lift_bank_hires: the oracle has nothing to hook")
        if a.arm != "static":
            st["bank"] = OracleBank(base, verify_zero=(a.arm == "zero"))
            model._lift_bank_hires = st["bank"]
        return out
    L.build_model = build_model

    orig_collate = D.collate

    def collate(e_ds, idx_list, law_idx):
        eb = orig_collate(e_ds, idx_list, law_idx)
        e_i, t = e_ds.index[idx_list[0]]
        eb["_r10_frame"] = int(t) + int(e_ds.window) - 1                # the NEWEST frame of the window (== the raster's ws)
        return eb
    D.collate = collate

    orig_cl = tr.compute_losses_v3

    def compute_losses_v3(model, eb, device, **kk):
        bank = st["bank"]
        if bank is not None:
            sid = int(eb["map_ep"].reshape(-1)[0])
            sha, fr = sha_of_sid[sid], int(eb["_r10_frame"])
            key = f"{sha}__{a.arm}_deg"
            d = 0.0
            if a.arm in ("H", "C"):
                if key not in tab.files:
                    st["n_no_table"] += 1
                else:
                    d = float(tab[key][fr]) * float(a.scale)
            bank.delta_rad = math.radians(d)
            st["n_windows"] += 1
            st["abs_delta_deg"].append(abs(d))
        return orig_cl(model, eb, device, **kk)
    tr.compute_losses_v3 = compute_losses_v3

    out_dir = None
    for i, x in enumerate(rest):
        if x == "--out":
            out_dir = Path(rest[i + 1])

    def record():
        if out_dir is None:
            return
        ad = np.asarray(st["abs_delta_deg"] or [0.0])
        rec = {"tool": "r10_oracle_roll.py", "arm": a.arm, "scale": a.scale, "table": str(a.table), "table_sha256": table_sha,
               "tier": "T0 diagnostic (egomotion pitch at inference is NOT admissible)", "n_windows_seen": st["n_windows"],
               "n_windows_clip_not_in_table": st["n_no_table"], "abs_delta_deg_mean_p95": [float(ad.mean()), float(np.percentile(ad, 95))],
               "bank": None if st["bank"] is None else {"n_hook": st["bank"].n_hook, "n_off": st["bank"].n_off, "n_zero_verified_bit_identical": st["bank"].n_verified},
               "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "r10_arm.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    atexit.register(record)
    return D.main(rest)


if __name__ == "__main__":
    sys.exit(main())
