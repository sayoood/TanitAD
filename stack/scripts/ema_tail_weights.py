"""Tail-EMA weights from two EMA snapshots of ONE run (E2-TAIL, ``PREREG_E2TAIL_2026-11-16.md`` sec. 1).

WHAT IT COMPUTES
    ``WeightEMA`` (``tanitad/train/weight_ema.py``) keeps ``ema_s = d * ema_(s-1) + (1 - d) * theta_s`` (``lerp``). Two saved
    snapshots ``ema_a`` (after ``num_updates_a`` updates) and ``ema_b`` (after ``num_updates_b``), same decay ``d``, give
    ``ema_b = d^n * ema_a + (1 - d^n) * W_tail`` with ``n = num_updates_b - num_updates_a``, so

        W_tail = (ema_b - d^n * ema_a) / (1 - d^n)  =  sum_{s=a+1..b} d^(b-s) * (1 - d) * theta_s / (1 - d^n)

    the exponentially weighted mean of the RAW weights of steps a+1..b alone, with the older history subtracted out. The
    arithmetic is fp64; each result is cast back to the tensor's STORED dtype.

WHAT IT DOES NOT DO
    * Tensors that are bit-identical in both files (frozen / not shadowed, and integer buffers) are COPIED, not computed.
    * It never writes into either input. The output is written to a temp name beside it and renamed.
    * The output has EXACTLY the structure of the second input (``model_ema.pt``): same top-level keys, same ``kind``
      (``ema_model_only``), ``step`` / ``num_updates`` of the later snapshot. No provenance key is added to it: the loaders
      record ``sorted(k for k in ck if k != "model")`` and ``r9_e1_dump`` reads ``kind``, so a new top-level key would
      change what every downstream record says about the file. Provenance goes to the sidecar ``<out>.provenance.json``.

REFUSES (exit 2, one line each): decays differ; ``num_updates`` missing / not an int; ``n <= 0``; tensor sets, shapes or
dtypes differ; a non-floating tensor that differs between the files (an EMA never averages those, so a difference means the
two files are not the same run); a non-finite value in the result; output equal to an input or already present.

Usage:  python ema_tail_weights.py --ema-a <run>/ckpt_ema_5000.pt --ema-b <run>/model_ema.pt --out <run>/model_ema_tail5k6k.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

__all__ = ["TailError", "tail_state", "tail_weights", "main"]


class TailError(ValueError):
    """The two files cannot give a tail average; the message says why."""


def _int_updates(ck: dict, label: str) -> int:
    v = ck.get("num_updates")
    if v is None:
        raise TailError(f"{label}: no 'num_updates' key (keys: {sorted(ck)})")
    if isinstance(v, bool) or not isinstance(v, int):
        raise TailError(f"{label}: num_updates {v!r} is not an int")
    return int(v)


def _decay(ck: dict, label: str) -> float:
    v = ck.get("ema_decay")
    if v is None:
        raise TailError(f"{label}: no 'ema_decay' key (keys: {sorted(ck)})")
    d = float(v)
    if not (0.0 < d < 1.0) or not math.isfinite(d):
        raise TailError(f"{label}: ema_decay {v!r} not in (0, 1)")
    return d


def tail_state(ck_a: dict, ck_b: dict) -> tuple:
    """The tail ``model`` state_dict from two loaded EMA files. Returns ``(state_dict, info)``."""
    for label, ck in (("ema-a", ck_a), ("ema-b", ck_b)):
        if not isinstance(ck, dict) or not isinstance(ck.get("model"), dict):
            raise TailError(f"{label}: not a model-only EMA file (no 'model' dict)")
        if ck.get("kind") != "ema_model_only":
            raise TailError(f"{label}: kind {ck.get('kind')!r} is not 'ema_model_only'")
    d_a, d_b = _decay(ck_a, "ema-a"), _decay(ck_b, "ema-b")
    if d_a != d_b:
        raise TailError(f"the two decays differ: ema-a {d_a!r} vs ema-b {d_b!r}")
    d = d_a
    nu_a, nu_b = _int_updates(ck_a, "ema-a"), _int_updates(ck_b, "ema-b")
    n = nu_b - nu_a
    if n <= 0:
        raise TailError(f"n = num_updates_b - num_updates_a = {nu_b} - {nu_a} = {n} must be > 0 (is ema-a later than ema-b?)")
    dn = d ** n                                         # python float == fp64
    denom = 1.0 - dn
    if not denom > 0.0:
        raise TailError(f"1 - d^n = {denom!r} is not positive (d={d}, n={n})")
    ma, mb = ck_a["model"], ck_b["model"]
    if set(ma) != set(mb):
        raise TailError(f"tensor sets differ: a-only e.g. {sorted(set(ma) - set(mb))[:3]}, b-only e.g. {sorted(set(mb) - set(ma))[:3]}")
    out, copied, computed, bad = {}, [], [], []
    for k in mb:                                        # b's order == the input file's order
        ta, tb = ma[k], mb[k]
        if not (torch.is_tensor(ta) and torch.is_tensor(tb)):
            raise TailError(f"{k}: not a tensor in both files")
        if tuple(ta.shape) != tuple(tb.shape):
            raise TailError(f"{k}: shape {tuple(ta.shape)} vs {tuple(tb.shape)}")
        if ta.dtype != tb.dtype:
            raise TailError(f"{k}: dtype {ta.dtype} vs {tb.dtype}")
        if torch.equal(ta, tb):
            out[k] = tb.clone()                         # bit for bit (an identical tensor is frozen / unshadowed)
            copied.append(k)
            if tb.is_floating_point() and not bool(torch.isfinite(tb).all()):
                bad.append(k)
            continue
        if not tb.is_floating_point():
            raise TailError(f"{k}: non-floating tensor ({tb.dtype}) differs between the files; an EMA does not average it, "
                            "so these are not two snapshots of one run")
        r = (tb.double() - dn * ta.double()) / denom
        if not bool(torch.isfinite(r).all()):
            bad.append(k)
            continue
        out[k] = r.to(tb.dtype)
        if not bool(torch.isfinite(out[k]).all()):      # fp32 overflow after the cast is also a refusal
            bad.append(k)
        computed.append(k)
    if bad:
        raise TailError(f"non-finite result in {len(bad)} tensor(s), e.g. {bad[:3]}")
    info = {"decay": d, "num_updates_a": nu_a, "num_updates_b": nu_b, "n": n, "d_pow_n": dn, "one_minus_d_pow_n": denom,
            "step_a": ck_a.get("step"), "step_b": ck_b.get("step"), "n_copied": len(copied), "n_computed": len(computed)}
    return out, info


def tail_weights(ck_a: dict, ck_b: dict) -> tuple:
    """The full output file dict: ``ck_b``'s structure, its ``model`` replaced by the tail state. Returns ``(dict, info)``."""
    state, info = tail_state(ck_a, ck_b)
    out = {k: v for k, v in ck_b.items()}               # same keys, same order, same kind / step / num_updates / ema_decay
    out["model"] = state
    return out, info


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ema-a", required=True, help="the EARLIER EMA snapshot, e.g. <run>/ckpt_ema_5000.pt")
    ap.add_argument("--ema-b", required=True, help="the LATER EMA snapshot, e.g. <run>/model_ema.pt")
    ap.add_argument("--out", required=True, help="output file (must not exist)")
    a = ap.parse_args(argv)
    pa, pb, po = Path(a.ema_a), Path(a.ema_b), Path(a.out)
    try:
        if not pa.is_file() or not pb.is_file():
            raise TailError(f"input missing: {pa if not pa.is_file() else pb}")
        if po.resolve() in (pa.resolve(), pb.resolve()):
            raise TailError("--out is one of the inputs")
        if po.exists():
            raise TailError(f"--out {po} already exists (refusing to overwrite)")
        ck_a = torch.load(str(pa), map_location="cpu", weights_only=False)
        ck_b = torch.load(str(pb), map_location="cpu", weights_only=False)
        out, info = tail_weights(ck_a, ck_b)
    except TailError as e:
        print(f"[ema-tail] REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    tmp = po.with_name(po.name + ".tmp")
    torch.save(out, str(tmp))
    os.replace(str(tmp), str(po))
    prov = dict(info, ema_a=str(pa), ema_b=str(pb), out=str(po), sha256_a=_sha256(pa), sha256_b=_sha256(pb),
                sha256_out=_sha256(po), top_level_keys=sorted(out), kind=out.get("kind"))
    Path(str(po) + ".provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True), encoding="utf-8")
    print(f"[ema-tail] wrote {po} (n={info['n']}, d={info['decay']}, computed {info['n_computed']}, copied {info['n_copied']}); "
          f"sha256 {prov['sha256_out']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
