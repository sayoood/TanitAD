"""ema_check_r8.py -- CPU gate before an EMA-weights battery: is the EMA file (a) loadable STRICT through the same builder the battery uses
and (b) really DIFFERENT from the raw weights of the same step?  If the weights were identical the arm would be pointless.

    python ema_check_r8.py --raw E:/refcv8_eval_kit/ckpt/ckpt_5000.pt --ema E:/refcv8_eval_kit/ckpt/ckpt_ema_5000.pt \
        --config E:/refcv8_eval_kit/ckpt/config.json --out <EMA_CHECK_step5000.json> [--no-build]

Writes the json and prints ONE line `ema_check:ok` / `ema_check:fail:<reason>`.  ok = same key set / shapes / dtypes, same step,
>= 1 floating tensor differs, strict load missing == 0 and unexpected == 0 (skipped with --no-build: then NOT ok for the gate).
Assert on the json.  CPU only; ~1-2 min (the builder dominates).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def compare(raw_sd: dict, ema_sd: dict) -> dict:
    import torch
    rec = {"keys_equal": set(raw_sd) == set(ema_sd), "n_keys_raw": len(raw_sd), "n_keys_ema": len(ema_sd)}
    shape_bad, dtype_bad, n_float, n_diff, n_same_float = [], [], 0, 0, 0
    sq_d = sq_r = 0.0
    max_abs = 0.0
    top = []
    for k in sorted(set(raw_sd) & set(ema_sd)):
        a, b = raw_sd[k], ema_sd[k]
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            continue
        if tuple(a.shape) != tuple(b.shape):
            shape_bad.append(k)
            continue
        if a.dtype != b.dtype:
            dtype_bad.append(k)
            continue
        if not a.is_floating_point():
            continue
        n_float += 1
        d = (a.double() - b.double())
        if torch.count_nonzero(d) == 0:
            n_same_float += 1
            continue
        n_diff += 1
        m = float(d.abs().max())
        max_abs = max(max_abs, m)
        sq_d += float((d * d).sum())
        sq_r += float((a.double() ** 2).sum())
        top.append((m, k))
    top.sort(reverse=True)
    rec.update(n_shape_mismatch=len(shape_bad), n_dtype_mismatch=len(dtype_bad), n_float_tensors=n_float,
               n_float_differing=n_diff, n_float_identical=n_same_float, max_abs_diff=max_abs,
               rel_l2_distance=(math.sqrt(sq_d) / math.sqrt(sq_r)) if sq_r > 0 else None,
               top5_max_abs=[{"key": k, "max_abs": m} for m, k in top[:5]])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--ema", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rec: dict = {"tool": "ema_check_r8.py", "started": time.strftime("%FT%T"), "raw": a.raw, "ema": a.ema, "ok": False, "device": "cpu"}
    why: list[str] = []
    try:
        import torch
        torch.set_num_threads(4)
        raw = torch.load(a.raw, map_location="cpu", weights_only=False)
        ema = torch.load(a.ema, map_location="cpu", weights_only=False)
        rec["raw_step"], rec["ema_step"] = raw.get("step"), ema.get("step")
        rec["ema_meta"] = {k: ema.get(k) for k in ("ema_decay", "num_updates", "kind")}
        rec["md5"] = {}
        for nm, p in (("raw", a.raw), ("ema", a.ema)):
            h = hashlib.md5()
            with open(p, "rb") as f:
                for c in iter(lambda: f.read(1 << 22), b""):
                    h.update(c)
            rec["md5"][nm] = h.hexdigest()
        rec["tensors"] = compare(raw["model"], ema["model"])
        t = rec["tensors"]
        if rec["raw_step"] != rec["ema_step"]:
            why.append(f"step differs raw {rec['raw_step']} vs ema {rec['ema_step']}")
        if not (t["keys_equal"] and t["n_shape_mismatch"] == 0 and t["n_dtype_mismatch"] == 0):
            why.append("key set / shapes / dtypes differ between the raw and the EMA file")
        if t["n_float_differing"] == 0:
            why.append("IDENTICAL: no floating tensor differs between raw and EMA -- the EMA arm would be pointless (STOP and report)")
        del raw, ema
        if a.no_build:
            why.append("--no-build: the strict-load check was not run")
        else:
            import r8_repoint
            L = r8_repoint.apply()
            config = L.load_config(a.config)
            model, cfg, args, mrec = L.build_model(config, a.ema, "cpu")
            sd = mrec["state_dict"]
            rec["strict_load"] = {"step": sd.get("step"), "strict": sd.get("strict"), "missing": len(sd.get("missing") or []),
                                  "unexpected": len(sd.get("unexpected") or []), "n_keys": sd.get("n_keys"),
                                  "param_breakdown_equal": (mrec.get("param_breakdown") or {}).get("equal"), "build_s": mrec.get("build_s")}
            if rec["strict_load"]["missing"] or rec["strict_load"]["unexpected"]:
                why.append(f"strict load: missing {rec['strict_load']['missing']} unexpected {rec['strict_load']['unexpected']}")
    except Exception as exc:                                    # noqa: BLE001 -- the json is the result
        why.append(f"{type(exc).__name__}: {str(exc)[:300]}")
    rec["ok"] = not why
    rec["why_not_ok"] = why
    rec["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print("ema_check:ok" if rec["ok"] else "ema_check:fail:" + "; ".join(why))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
