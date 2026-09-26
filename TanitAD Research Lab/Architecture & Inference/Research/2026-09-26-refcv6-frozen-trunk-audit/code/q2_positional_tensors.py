"""Q2 (advisory class B) on the WHOLE refcv6 checkpoint, not only the backbone.

The 2026-09-22 review (TRUNK_INPUT_REVIEW F5) settled B1 for the resnet101 BACKBONE by parameter
arithmetic against the timm checkpoint (0 orphans). It did not cover the ~58 M parameters on top:
agent decoder, box decoder, BEV encoder, tactical decoder v6, the diffusion decoder. This
instrument answers the advisory's two questions over EVERY tensor of the trained run:

 (a) does any POSITIONAL / GEOMETRIC tensor exist that has no training-checkpoint counterpart,
     or sits in the checkpoint but never trains (the REFe 'random frozen table' shape)?
     -> list every state_dict key whose name or role is positional (pos / query / embed / level /
        row / col / source / anchor / grid ...), and read it at step 1000, 5000 and 30000 of the
        LIVE run. A learned table that is bit-identical across 29,000 steps is frozen in effect.
 (b) is there a non-persistent positional BUFFER (absent from state_dict, so it can never be
     checked against any checkpoint) -- found by building the model and diffing
     named_buffers() against state_dict() (done in q3_hooks_forward.py, which builds the model;
     this file only reads checkpoints, so it runs under the RAM floor).

All checkpoints are opened with mmap=True: pages are read on touch, so the ~400 MB files do not
become resident. Evidence class: MEASURED (the run's own checkpoints, md5 in ckpt/MD5SUMS).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.bootstrap()
import torch  # noqa: E402

CKPTS = {"step1000": C.KIT / "ckpt/ckpt_step1000.pt",
         "step5000": C.KIT / "ckpt/ckpt_5000.pt",
         "step30000": C.KIT / "ckpt/ckpt_30000.pt"}
POS_PAT = re.compile(r"(pos|query|queries|embed|level|row_|col_|source_code|anchor|grid|"
                     r"unobserved|register|token|slot|time|t_emb|timestep|coord|height)",
                     re.IGNORECASE)


def load_sd(p: Path) -> tuple[dict, dict]:
    ck = torch.load(str(p), map_location="cpu", weights_only=False, mmap=True)
    meta = {k: (v if not torch.is_tensor(v) else f"tensor{tuple(v.shape)}")
            for k, v in ck.items() if k != "model" and not isinstance(v, dict)}
    meta["top_keys"] = sorted(ck.keys())
    return ck["model"], meta


def main():
    if C.ram_available_gb() < 1.5:
        raise SystemExit("[audit:RAM] < 1.5 GB available even for a light job")
    sds, metas = {}, {}
    for k, p in CKPTS.items():
        sds[k], metas[k] = load_sd(p)
        print(k, "keys", len(sds[k]), "step", metas[k].get("step"), flush=True)
    keysets = {k: set(v) for k, v in sds.items()}
    same_keys = keysets["step1000"] == keysets["step5000"] == keysets["step30000"]
    ref = sds["step30000"]
    rows = []
    for name, t in ref.items():
        if not torch.is_tensor(t) or not t.is_floating_point():
            continue
        if not POS_PAT.search(name):
            continue
        a = sds["step1000"][name].float()
        b = sds["step5000"][name].float()
        c = t.float()
        rows.append({
            "key": name, "shape": list(c.shape), "numel": int(c.numel()),
            "norm_step1000": float(a.norm()), "norm_step30000": float(c.norm()),
            "rel_change_1000_to_5000": float((b - a).norm() / (a.norm() + 1e-12)),
            "rel_change_1000_to_30000": float((c - a).norm() / (a.norm() + 1e-12)),
            "bit_identical_1000_30000": bool(torch.equal(a, c)),
        })
    frozen = [r for r in rows if r["bit_identical_1000_30000"]]
    # every float tensor of the whole model: how many never moved between 1000 and 30000
    all_frozen = []
    n_float = 0
    for name, t in ref.items():
        if not torch.is_tensor(t) or not t.is_floating_point():
            continue
        n_float += 1
        if torch.equal(sds["step1000"][name], t):
            all_frozen.append({"key": name, "shape": list(t.shape)})
    # classify the never-moving tensors by module prefix
    by_prefix: dict = {}
    for r in all_frozen:
        pre = ".".join(r["key"].split(".")[:3])
        by_prefix[pre] = by_prefix.get(pre, 0) + 1
    out = {"what": "Q2: every positional/geometric-named tensor of the live run, read at steps "
                   "1000 / 5000 / 30000",
           "evidence_class": "MEASURED (the run's checkpoints; md5 in D:/refcv6_eval_kit/ckpt/MD5SUMS)",
           "ckpt_meta": metas, "same_keyset_all_three": same_keys,
           "n_keys": len(ref), "n_float_tensors": n_float,
           "n_positional_named": len(rows),
           "n_positional_named_bit_identical_1000_to_30000": len(frozen),
           "positional_named_bit_identical": frozen,
           "n_float_tensors_bit_identical_1000_to_30000": len(all_frozen),
           "bit_identical_by_prefix": dict(sorted(by_prefix.items())),
           "bit_identical_tensors": all_frozen,
           "positional_named": sorted(rows, key=lambda r: r["key"])}
    for r in out["positional_named"]:
        print("%-90s %-18s rel30k=%.4g ident=%s" % (r["key"], r["shape"],
              r["rel_change_1000_to_30000"], r["bit_identical_1000_30000"]))
    print("bit-identical float tensors 1000->30000:", len(all_frozen), "of", n_float)
    for k, v in out["bit_identical_by_prefix"].items():
        print("   ", k, v)
    C.write_json("q2_positional_tensors.json", out)


if __name__ == "__main__":
    main()
