"""slim_rolling_ckpt_r8.py -- derive the milestone-layout checkpoints from a ROLLING `ckpt.pt` (CPU only, no model build).

WHY (MEASURED 2026-10-16): the refcv8 trainer writes `ckpt_<step>.pt` / `ckpt_ema_<step>.pt` ONLY at `MILESTONES = (5000, 15000, 20000,
30000)` (`stack/scripts/refc_v3_train.py:232`, `:10834-10839`); there is NO step-10,000 pair on Thor. The rolling `ckpt.pt` (every 500
steps, overwritten) carries everything the pair holds:
    ckpt_<S>.pt      = {"model": model.state_dict(), "step": S}                            == ck["model"], ck["step"]
    ckpt_ema_<S>.pt  = {"model": ema.model_state_dict(model), "step", "ema_decay", "num_updates", "kind": "ema_model_only"}
`WeightEMA.model_state_dict` (`stack/tanitad/train/weight_ema.py:229`): every shadowed entry replaced by its EMA value cast to the live
dtype, every other entry (frozen / integer buffers) taken from the live model, tied tensors aliased to the one shadow. This script does the
same from the checkpoint alone, `derive_ema_model` is pure and pinned by test_queue_after_5000.py against the REAL `WeightEMA`.

    python slim_rolling_ckpt_r8.py --src <full ckpt.pt> --out-dir E:/refcv8_eval_kit/ckpt --expect-step 10500 [--provenance k=v ...]

Writes `ckpt_<S>.pt`, `ckpt_ema_<S>.pt`, `ckpt_<S>.provenance.json` (atomic; refuses to overwrite an existing different file).
Prints ONE json line (`ok` true/false + reason). The derived files are NOT trainer-written milestone files and say so in the sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch


def _sig(t):
    return (t.untyped_storage().data_ptr(), int(t.storage_offset()), tuple(t.shape), tuple(t.stride()))


def derive_ema_model(model_sd: dict, ema_sd: dict) -> tuple[dict, dict]:
    """(model-only EMA state dict, stats). Pure; raises ValueError on any structural mismatch."""
    shadow = ema_sd["shadow"]
    missing = [n for n in shadow if n not in model_sd]
    if missing:
        raise ValueError(f"EMA shadow names {len(missing)} tensors absent from the model state dict, e.g. {missing[:3]}")
    covered = {_sig(model_sd[n]): n for n in shadow}
    out, n_direct, n_alias, n_live = {}, 0, 0, 0
    for k, t in model_sd.items():
        if not torch.is_tensor(t):
            out[k] = t
            n_live += 1
            continue
        if k in shadow:
            src, n_direct = k, n_direct + 1
        else:
            src = covered.get(_sig(t))
            if src is not None:
                n_alias += 1
        if src is None:
            out[k] = t.detach().clone()
            n_live += 1
        else:
            s = shadow[src]
            if tuple(s.shape) != tuple(t.shape):
                raise ValueError(f"shadow {src!r} shape {tuple(s.shape)} != live {tuple(t.shape)}")
            out[k] = s.detach().to(dtype=t.dtype).clone()
    return out, {"n_keys": len(out), "n_shadow": len(shadow), "n_direct": n_direct, "n_alias": n_alias, "n_live_copied": n_live}


def _md5(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def _atomic_save(obj, dest: Path, force: bool) -> str:
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} exists (pass --force to replace)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    torch.save(obj, tmp)
    os.replace(tmp, dest)
    return _md5(str(dest))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expect-step", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--provenance", nargs="*", default=[], help="k=v pairs copied into the sidecar (e.g. thor_md5=..., thor_mtime=...)")
    a = ap.parse_args()
    rec: dict = {"tool": "slim_rolling_ckpt_r8.py", "src": a.src, "started": time.strftime("%FT%T"), "ok": False}
    try:
        ck = torch.load(a.src, map_location="cpu", weights_only=False)
        for k in ("model", "step", "ema"):
            if k not in ck:
                raise ValueError(f"source has no {k!r} (keys: {sorted(ck)[:12]})")
        step = int(ck["step"])
        rec["step"] = step
        if a.expect_step is not None and step != a.expect_step:
            raise ValueError(f"source step is {step}, expected {a.expect_step}")
        ema = ck["ema"]
        if int(ema["num_updates"]) != step:
            raise ValueError(f"ema num_updates {ema['num_updates']} != step {step}")
        em, stats = derive_ema_model(ck["model"], ema)
        n_diff = sum(1 for k, v in em.items() if torch.is_tensor(v) and v.is_floating_point() and not torch.equal(v, ck["model"][k]))
        rec.update(ema_stats=stats, ema_decay=float(ema["decay"]), ema_num_updates=int(ema["num_updates"]),
                   n_tensors_ema_differs_from_live=n_diff)
        if n_diff == 0:
            raise ValueError("every EMA tensor equals the live weight: the derived EMA file would be pointless")
        out = Path(a.out_dir)
        raw_p, ema_p = out / f"ckpt_{step}.pt", out / f"ckpt_ema_{step}.pt"
        rec["raw"] = {"path": str(raw_p), "md5": _atomic_save({"model": ck["model"], "step": step}, raw_p, a.force)}
        rec["ema"] = {"path": str(ema_p), "md5": _atomic_save(
            {"model": em, "step": step, "ema_decay": float(ema["decay"]), "num_updates": int(ema["num_updates"]),
             "kind": "ema_model_only"}, ema_p, a.force)}
        rec["note"] = ("DERIVED on the dev box from a rolling ckpt.pt; NOT a trainer-written milestone file (the trainer writes those "
                       "only at MILESTONES=(5000,15000,20000,30000)). Same layout, same tensors.")
        rec["provenance"] = dict(kv.split("=", 1) for kv in a.provenance if "=" in kv)
        rec["src_md5"] = _md5(a.src)
        rec["ok"] = True
        (out / f"ckpt_{step}.provenance.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    except Exception as exc:                                    # noqa: BLE001 -- the json line IS the result
        rec["reason"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    print(json.dumps(rec, default=str))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
