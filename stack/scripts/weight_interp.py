"""Partial weight interpolation toward an init checkpoint (E2-WISE, ``PREREG_E2WISE_2026-11-17.md`` sec. 2 and 5).

WHAT IT COMPUTES
    For a run checkpoint ``run`` (e.g. ``model_ema_tail5k6k.pt``), its ``init`` (``--init-from`` file), a factor ``alpha`` in [0, 1]
    and a set S of module-name prefixes:

        theta = theta_init + alpha * (theta_run - theta_init)

    on every tensor whose name is under a prefix in S AND exists in the init with the SAME shape. The arithmetic is fp32 and the
    result is cast back to the run tensor's dtype. It is evaluated with ``torch.lerp``, which is the same line algebraically but
    is EXACT at both ends (alpha = 0 gives the init, alpha = 1 gives the run, bit for bit); the literal ``init + a * (run - init)``
    is not exact at alpha = 1 (a one-ulp rounding in ``run - init``), and the prereg requires bit-exactness there.

    Prefixes match WHOLE dotted components: ``core.decoder`` matches ``core.decoder`` and ``core.decoder.w`` and does NOT match
    ``core.decoderX.w``.

WHAT IT DOES NOT DO
    * Every other tensor is copied from the run file bit for bit (new modules, frozen tensors, everything outside S, and any tensor
      in S that is absent from the init, has another shape, or is not floating point; each of those is LISTED).
    * Non-tensor entries of the model dict and every top-level key other than ``model`` are copied unchanged.
    * It never writes into either input. The output is written to a temp name, RELOADED and checked, then renamed.

THE PERCEPTION GUARANTEE (prereg sec. 3.1)
    ``<out>.provenance.json`` lists ``perception_untouched``: every tensor OUTSIDE the prefixes, each re-read from the written
    file and confirmed bit-equal to the run file. The read uses it to reuse stage A. If any one differs, the tool refuses.

REFUSES (exit 2): a prefix that matches no tensor of the run file; no prefix; alpha not in [0, 1]; ``--out`` already existing or
equal to an input; run and init sharing no tensor name; a model that is not a dict; a non-finite value in an interpolated tensor.

Usage:  python weight_interp.py --run <tail.pt> --init <init.pt> --alpha 0.5 --prefixes core.decoder,scorer --out <out.pt>
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

__all__ = ["InterpError", "match_prefix", "interp_state", "interp_checkpoint", "main"]


class InterpError(ValueError):
    """The two files cannot be interpolated as asked; the message says why."""


def match_prefix(name: str, prefix: str) -> bool:
    """True when ``name`` is ``prefix`` or lies under it, matching WHOLE dotted components."""
    return name == prefix or name.startswith(prefix + ".")


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same dtype, shape and bytes (NaN-safe, unlike ``torch.equal``)."""
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if a.numel() == 0:
        return True
    return bool(a.detach().contiguous().reshape(-1).view(torch.uint8).equal(b.detach().contiguous().reshape(-1).view(torch.uint8)))


def _model(ck, label: str) -> dict:
    if not isinstance(ck, dict) or not isinstance(ck.get("model"), dict):
        raise InterpError(f"{label}: not a checkpoint with a 'model' dict")
    return ck["model"]


def interp_state(ck_run: dict, ck_init: dict, alpha: float, prefixes) -> tuple:
    """The interpolated ``model`` state dict. Returns ``(state_dict, info)``."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise InterpError(f"alpha {alpha!r} is not in [0, 1]")
    prefixes = [p for p in prefixes]
    if not prefixes or any((not isinstance(p, str)) or not p for p in prefixes):
        raise InterpError(f"--prefixes must be a non-empty list of non-empty names, got {prefixes!r}")
    mr, mi = _model(ck_run, "run"), _model(ck_init, "init")
    if not (set(k for k, v in mr.items() if torch.is_tensor(v)) & set(k for k, v in mi.items() if torch.is_tensor(v))):
        raise InterpError("run and init share no tensor name")
    for p in prefixes:
        if not any(torch.is_tensor(v) and match_prefix(k, p) for k, v in mr.items()):
            raise InterpError(f"prefix {p!r} matches no tensor in the run file")
    out, interpolated, copied_in_s, untouched, nontensor = {}, [], {}, [], []
    a = float(alpha)
    for k, tr in mr.items():                            # run's order == the output's order
        if not torch.is_tensor(tr):
            out[k] = tr
            nontensor.append(k)
            continue
        if not any(match_prefix(k, p) for p in prefixes):
            out[k] = tr.clone()                         # outside S: bit for bit
            untouched.append(k)
            continue
        ti = mi.get(k)
        if not torch.is_tensor(ti):
            out[k] = tr.clone()
            copied_in_s[k] = "absent from init"
        elif tuple(ti.shape) != tuple(tr.shape):
            out[k] = tr.clone()
            copied_in_s[k] = f"shape {tuple(tr.shape)} vs init {tuple(ti.shape)}"
        elif not (tr.is_floating_point() and ti.is_floating_point()):
            out[k] = tr.clone()
            copied_in_s[k] = f"non-floating ({tr.dtype} / init {ti.dtype})"
        else:
            r32, i32 = tr.detach().float(), ti.detach().float()
            res = torch.lerp(i32, r32, a)               # THE INTERPOLATION: init + a * (run - init), fp32
            if not bool(torch.isfinite(res).all()) and bool(torch.isfinite(r32).all()) and bool(torch.isfinite(i32).all()):
                raise InterpError(f"{k}: non-finite result from finite inputs")
            out[k] = res.to(tr.dtype)
            if bool(torch.isfinite(r32).all()) and not bool(torch.isfinite(out[k]).all()):
                raise InterpError(f"{k}: non-finite after the cast to {tr.dtype}")
            interpolated.append(k)
    info = {"alpha": a, "prefixes": list(prefixes), "n_interpolated": len(interpolated), "n_copied_in_S": len(copied_in_s),
            "n_untouched": len(untouched), "n_nontensor": len(nontensor), "interpolated": interpolated,
            "copied_in_S": copied_in_s, "untouched": untouched}
    return out, info


def interp_checkpoint(ck_run: dict, ck_init: dict, alpha: float, prefixes) -> tuple:
    """The full output file dict: the run's structure, its ``model`` replaced. Returns ``(dict, info)``."""
    state, info = interp_state(ck_run, ck_init, alpha, prefixes)
    out = {k: v for k, v in ck_run.items()}             # same keys, same order, every other entry unchanged
    out["model"] = state
    return out, info


def _hash(p: Path, name: str) -> str:
    h = hashlib.new(name)
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _state_digest(state: dict) -> str:
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode("utf-8"))
        if torch.is_tensor(v):
            h.update(str(v.dtype).encode() + str(tuple(v.shape)).encode())
            if v.numel():
                h.update(v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="the run checkpoint, e.g. <run>/model_ema_tail5k6k.pt")
    ap.add_argument("--init", required=True, help="the init checkpoint (the run's --init-from)")
    ap.add_argument("--alpha", required=True, type=float, help="0 = init on S, 1 = run")
    ap.add_argument("--prefixes", required=True, help="comma-separated module-name prefixes (whole dotted components)")
    ap.add_argument("--out", required=True, help="output file (must not exist)")
    a = ap.parse_args(argv)
    pr, pi, po = Path(a.run), Path(a.init), Path(a.out)
    prefixes = [p.strip() for p in a.prefixes.split(",")]
    tmp = po.with_name(po.name + ".tmp")
    try:
        if not pr.is_file() or not pi.is_file():
            raise InterpError(f"input missing: {pr if not pr.is_file() else pi}")
        if po.resolve() in (pr.resolve(), pi.resolve()):
            raise InterpError("--out is one of the inputs")
        if po.exists():
            raise InterpError(f"--out {po} already exists (refusing to overwrite)")
        ck_run = torch.load(str(pr), map_location="cpu", weights_only=False)
        ck_init = torch.load(str(pi), map_location="cpu", weights_only=False)
        out, info = interp_checkpoint(ck_run, ck_init, a.alpha, prefixes)
        torch.save(out, str(tmp))
        back = torch.load(str(tmp), map_location="cpu", weights_only=False)     # verify what is ON DISK, not what is in memory
        bm, rm = back["model"], ck_run["model"]
        if list(bm) != list(rm) or list(back) != list(ck_run):
            raise InterpError("written file differs in key structure from the run file")
        bad = [k for k in info["untouched"] if not _bit_equal(bm[k], rm[k])]
        bad += [k for k in info["copied_in_S"] if not _bit_equal(bm[k], rm[k])]
        if bad:
            raise InterpError(f"{len(bad)} copied tensor(s) not bit-equal to the run file after writing, e.g. {bad[:3]}")
    except InterpError as e:
        if tmp.exists():
            tmp.unlink()
        print(f"[weight-interp] REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    os.replace(str(tmp), str(po))
    prov = {"run": str(pr), "init": str(pi), "out": str(po), "sha256_run": _hash(pr, "sha256"), "sha256_init": _hash(pi, "sha256"),
            "sha256_out": _hash(po, "sha256"), "md5_out": _hash(po, "md5"), "state_dict_sha256_out": _state_digest(out["model"]),
            "alpha": info["alpha"], "prefixes": info["prefixes"], "n_interpolated": info["n_interpolated"],
            "n_copied_in_S": info["n_copied_in_S"], "n_untouched": info["n_untouched"], "interpolated": info["interpolated"],
            "copied_in_S": info["copied_in_S"], "perception_untouched": info["untouched"], "perception_untouched_all_bit_equal": True,
            "top_level_keys": sorted(out), "kind": out.get("kind")}
    Path(str(po) + ".provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True), encoding="utf-8")
    print(f"[weight-interp] wrote {po} (alpha={info['alpha']}, interpolated {info['n_interpolated']}, copied-in-S "
          f"{info['n_copied_in_S']}, untouched {info['n_untouched']}); sha256 {prov['sha256_out']} md5 {prov['md5_out']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
