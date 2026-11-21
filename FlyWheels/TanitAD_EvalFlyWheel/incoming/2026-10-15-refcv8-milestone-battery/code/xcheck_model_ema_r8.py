"""xcheck_model_ema_r8.py -- compare the TRAINER-WRITTEN `model_ema.pt` (final step) with the EMA file the chain DERIVED from `ckpt.pt['ema']`.  CPU only.  NEW file (2026-10-19, EvalFlyWheel).

    python xcheck_model_ema_r8.py --thor-ema E:/refcv8_eval_kit/ckpt/model_ema_48852_thor.pt --derived E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --step 48852 --out <json>

WHY.  At the final step the trainer writes NO `ckpt_ema_<step>.pt` (MILESTONES, refc_v3_train.py:232); the battery's EMA weights are therefore DERIVED by `extract_ema_r8.py` from the full `ckpt.pt['ema']`
(through the trainer's own WeightEMA / `_save_ema_model`, cross-checked there against a second derivation).  But the trainer ALSO writes `model_ema.pt` at `step == args.steps` (:10837-10841) straight from its
live EMA.  Those are two independent routes to the same weights; they must agree on EVERY tensor.  This script is that comparison: it is a check against a different CHANNEL, not a re-run of the derivation.
(When the run was resumed AT step == steps the loop body is skipped and model_ema.pt is never written; the chain then marks the check NOT APPLICABLE -- it does not call this script.)

Checks (all asserted on the loaded content, never on file names): both files load; meta keys == {model, step, ema_decay, num_updates, kind}; kind == 'ema_model_only'; step == --step and num_updates == --step
(the EMA has seen every optimiser step, extract_ema_r8.py:105); ema_decay equal; the key sets equal; every tensor equal in dtype, shape and VALUE (torch.equal) and every non-tensor equal.
The output records both md5s so the chain can pin the verdict to the exact files.  Peak memory ~2 x 0.4 GB; the chain memory-gates it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
META_KEYS = {"model", "step", "ema_decay", "num_updates", "kind"}


def _md5(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def compare(thor: dict, derived: dict, step: int) -> dict:
    """Pure comparison of two loaded payloads -> {ok, problems, n_keys, n_tensors, n_mismatch, mismatch_examples}."""
    import torch
    problems: list[str] = []
    for name, d in (("trainer model_ema.pt", thor), ("derived ckpt_ema", derived)):
        if not isinstance(d, dict) or set(d) != META_KEYS:
            problems.append(f"{name}: meta keys {sorted(d) if isinstance(d, dict) else type(d).__name__} != {sorted(META_KEYS)}")
            continue
        if d.get("kind") != "ema_model_only":
            problems.append(f"{name}: kind {d.get('kind')!r} != 'ema_model_only'")
        if int(d.get("step", -1)) != step:
            problems.append(f"{name}: step {d.get('step')} != {step}")
        if int(d.get("num_updates", -1)) != step:
            problems.append(f"{name}: num_updates {d.get('num_updates')} != {step}")
    if problems and not (isinstance(thor, dict) and isinstance(derived, dict) and "model" in thor and "model" in derived):
        return {"ok": False, "problems": problems, "n_keys": 0, "n_tensors": 0, "n_mismatch": 0, "mismatch_examples": []}
    if float(thor.get("ema_decay", -1)) != float(derived.get("ema_decay", -2)):
        problems.append(f"ema_decay differs: {thor.get('ema_decay')} vs {derived.get('ema_decay')}")
    a, b = thor["model"], derived["model"]
    if set(a) != set(b):
        problems.append(f"model key sets differ: only-in-trainer {sorted(set(a) - set(b))[:5]} only-in-derived {sorted(set(b) - set(a))[:5]}")
    n_t, bad = 0, []
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if torch.is_tensor(x) or torch.is_tensor(y):
            n_t += 1
            if not (torch.is_tensor(x) and torch.is_tensor(y)) or x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                bad.append(k)
        elif x != y:
            bad.append(k)
    if bad:
        problems.append(f"{len(bad)} entries differ between the trainer-written EMA and the derived EMA")
    if n_t == 0:
        problems.append("no tensors compared: refusing to call an empty comparison a match")
    return {"ok": not problems, "problems": problems, "n_keys": len(set(a) & set(b)), "n_tensors": n_t, "n_mismatch": len(bad), "mismatch_examples": bad[:8]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thor-ema", required=True)
    ap.add_argument("--derived", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec: dict = {"tool": "xcheck_model_ema_r8.py", "started": time.strftime("%FT%T"), "step": a.step, "ok": False,
                 "md5": {"thor": _md5(a.thor_ema), "derived": _md5(a.derived)}, "files": {"thor": a.thor_ema, "derived": a.derived}}
    try:
        import torch
        t = torch.load(a.thor_ema, map_location="cpu", weights_only=False)
        d = torch.load(a.derived, map_location="cpu", weights_only=False)
        rec.update(compare(t, d, a.step))
    except Exception as e:  # noqa: BLE001 -- an unreadable file is a FAIL with the reason, never a pass
        rec.update(ok=False, problems=[f"{type(e).__name__}: {e}"])
    rec["finished"] = time.strftime("%FT%T")
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, a.out)
    print(f"ZZXCHECK ok={rec['ok']} n_tensors={rec.get('n_tensors')} n_mismatch={rec.get('n_mismatch')} ZZ")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
