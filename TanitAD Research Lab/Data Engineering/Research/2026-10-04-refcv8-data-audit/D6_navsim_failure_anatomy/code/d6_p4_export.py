"""D6 P4 -- GPU export of refcv7-50,400's OWN predicted drivable mask, from the SAME forward as the official navhard scoring.
TANITAD VENV.  Runs ONLY as the child of ``d6_p4_gpu_chain.py`` (itself the child of the battery's ``with_gpu_lock.py``).

    python d6_p4_export.py --mask-out <dir> -- <run_bridge7.py arguments, incl. --export-fan <dir>>

It imports the D6 bridge copy (``code/bridge_fix/run_bridge7.py`` + ``refcv7_bridge.py``, byte-for-byte what P1' ran) and adds three
OBSERVATION-ONLY hooks; no bridge file is edited and the forward is untouched:
  * ``R7.load_refcv7``         -> keeps a reference to the built model (to read its frozen class weights);
  * ``R7.scene_seed``          -> records the token of the scene about to be run (it is called with the token right before the forward);
  * ``R7.export_fan_from_out`` -> after the original export, reads ``out["perception"]["map_hires_logits"]`` [1, 8, 1000, 600] and writes
    ONE mask record for the token.

The record (``raw/p4_impl_choices.md`` s1-2): the calibrated drivable posterior ``p_hat = softmax(z - ln w)[:, 1]`` as an EXACT bitmask
``p_hat >= 0.5`` (the gate's input) and as uint8 ``round(255 p_hat)``; the raw-softmax drivable bitmask (DIAGNOSTIC ONLY); the
prior-corrected class decision (the module's own ``decide(..., "prior_corrected")``) as uint8 codes; the 0.25 m lift-valid bitmask.
Storage: ``<mask-out>/mask_<arm>.bin`` (zlib records, appended) + ``mask_<arm>.index.jsonl`` (offset, length, sha256 of the
compressed bytes, per-scene counts).  The index line is written AFTER the bytes are flushed; a torn tail is ignored on read.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bridge_fix"))

import numpy as np  # noqa: E402

import run_bridge7 as RB  # noqa: E402   (the D6 bridge copy; imports refcv7_bridge as R7)

R7 = RB.R7
STATE = {"model": None, "tok": None, "stage": None, "bin": None, "idx": None, "arm": None, "w": None, "n": 0}
H, W, NCLS, DRIVABLE = 1000, 600, 8, 1
LV_HW = (400, 240)


def _open(mask_out: str, arm: str):
    os.makedirs(mask_out, exist_ok=True)
    STATE["bin"] = open(os.path.join(mask_out, f"mask_{arm}.bin"), "ab")
    STATE["idx"] = open(os.path.join(mask_out, f"mask_{arm}.index.jsonl"), "a", encoding="utf-8")
    STATE["arm"] = arm


_orig_load = R7.load_refcv7


def _load_hook(*a, **k):
    res = _orig_load(*a, **k)
    STATE["model"] = res[0]
    return res


_orig_seed = R7.scene_seed


def _seed_hook(spec_seed, tok):
    STATE["tok"] = tok
    return _orig_seed(spec_seed, tok)


_orig_export = R7.export_fan_from_out


def _export_hook(out: dict, ds: dict) -> dict:
    fan = _orig_export(out, ds)
    _write_mask(out, fan)
    return fan


def _write_mask(out: dict, fan: dict) -> None:
    import torch
    from tanitad.models import map_head_hires as MH
    tok = STATE["tok"]
    if tok is None:
        raise RuntimeError("[p4] no token recorded before the forward -- refusing to write an unkeyed mask")
    per = out.get("perception")
    if not isinstance(per, dict) or per.get("map_hires_logits") is None:
        raise RuntimeError("[p4] out['perception']['map_hires_logits'] is missing -- the map head did not run")
    z = per["map_hires_logits"].float()
    if tuple(z.shape) != (1, NCLS, H, W):
        raise RuntimeError(f"[p4] map_hires_logits shape {tuple(z.shape)} != (1, {NCLS}, {H}, {W})")
    if STATE["w"] is None:
        w = STATE["model"]._map_hires_class_weight
        if w is None:
            raise RuntimeError("[p4] the model carries no frozen map class weights")
        STATE["w"] = w.detach().float().to(z.device)
        if tuple(STATE["w"].shape) != (NCLS,) or not bool((STATE["w"] > 0).all()):
            raise RuntimeError(f"[p4] class weights {STATE['w'].tolist()} are not 8 positive values")
    w = STATE["w"]
    with torch.no_grad():
        s = z - torch.log(w).view(1, -1, 1, 1)
        p_hat = torch.softmax(s, dim=1)[0, DRIVABLE]                      # calibrated posterior (the gate's probability)
        p_raw = torch.softmax(z, dim=1)[0, DRIVABLE]                      # raw softmax (DIAGNOSTIC ONLY)
        m_hat = (p_hat >= 0.5).cpu().numpy()
        m_raw = (p_raw >= 0.5).cpu().numpy()
        u8 = torch.round(p_hat * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        cls = MH.decide(z, "prior_corrected", class_weight=w)[0].to(torch.uint8).cpu().numpy()
        lv = per.get("map_hires_lift_valid")
        lv = (np.zeros(LV_HW, bool) if lv is None else lv[0].bool().cpu().numpy())
    if lv.shape != LV_HW:
        raise RuntimeError(f"[p4] lift-valid shape {lv.shape} != {LV_HW}")
    payload = b"".join([np.packbits(m_hat.reshape(-1)).tobytes(), np.packbits(m_raw.reshape(-1)).tobytes(),
                        u8.reshape(-1).tobytes(), cls.reshape(-1).tobytes(), np.packbits(lv.reshape(-1)).tobytes()])
    comp = zlib.compress(payload, 6)
    fb = STATE["bin"]
    fb.seek(0, 2)
    off = fb.tell()
    fb.write(comp)
    fb.flush()
    os.fsync(fb.fileno())
    rec = {"token": tok, "arm": STATE["arm"], "offset": off, "nbytes": len(comp),
           "sha256_16": hashlib.sha256(comp).hexdigest()[:16], "raw_nbytes": len(payload),
           "layout": "packbits(m_hat[1000x600]) | packbits(m_raw[1000x600]) | u8 p_hat[1000x600] | u8 cls_prior_corrected[1000x600] | packbits(lift_valid[400x240])",
           "n_drivable_hat": int(m_hat.sum()), "n_drivable_raw": int(m_raw.sum()), "mean_p_hat": float(u8.mean() / 255.0),
           "lift_valid_frac": float(lv.mean()), "sel_idx": int(fan["sel_idx"]), "n_cands": int(fan["cands_knots"].shape[0]),
           "class_weights": [float(x) for x in w.cpu().tolist()], "t": round(time.time(), 3)}
    STATE["idx"].write(json.dumps(rec) + "\n")
    STATE["idx"].flush()
    STATE["n"] += 1


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: d6_p4_export.py --mask-out <dir> -- <run_bridge7 args>")
    i = argv.index("--")
    mine, rest = argv[:i], argv[i + 1:]
    if len(mine) != 2 or mine[0] != "--mask-out":
        raise SystemExit("usage: d6_p4_export.py --mask-out <dir> -- <run_bridge7 args>")
    if "--export-fan" not in rest:
        raise SystemExit("[p4] the export needs --export-fan (the plans/ranking control reads the fan sidecar)")
    arms = rest[rest.index("--arms") + 1].split(",")
    if arms != ["R7_A1"]:
        raise SystemExit(f"[p4] only arm R7_A1 is registered for P4, got {arms}")
    _open(mine[1], "R7_A1")
    R7.load_refcv7 = _load_hook
    R7.scene_seed = _seed_hook
    R7.export_fan_from_out = _export_hook
    rc = RB.main(rest)
    print(f"[p4] mask records written this launch: {STATE['n']}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
