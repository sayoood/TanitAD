#!/usr/bin/env python3
"""DINOv3 -> TanitAD ENCODER-SEED checkpoint converter (D-V7-DINO-SEED).

⛔ WHY THIS FILE EXISTS. The PI directed that v7 trains its trunk *from a DINOv3
initialisation*. Measured 2026-09-03 (`PREREG_V7F.md` §0 E1, two probes): **no
loader in this repo puts DINOv3 weights into ``ViTEncoder``/``ViT5Encoder``.**
DINOv3 appears only as O7's frozen teacher, as REF-A's precomputed feature bank
and in the fp8 shipper. ``--init-from`` loads a WHOLE-STACK checkpoint and
refuses a partial one. So "DINO as init" was not implementable at all.

This script closes exactly that gap: it reads a DINOv3 checkpoint and emits a
**seed checkpoint** carrying ONLY the encoder tensors, mapped onto
``tanitad.models.encoder.ViTEncoder``'s parameter names, with a provenance stamp
that a later audit can read.

⛔⛔ THE FAILURE MODE THIS FILE IS BUILT AGAINST IS A **SILENT PARTIAL SEED** —
a load that puts 60 % of the trunk in place, leaves the rest at random init, and
then looks *exactly* like a successful init in every log the run writes. That is
the `df` / cgroup / `step_s` family in a checkpoint costume: a wrong answer that
reports success. Therefore:

  * every source tensor is either MAPPED, or on an explicit ALLOW-LIST, or the
    conversion REFUSES and names the tensor;
  * every target tensor is either WRITTEN, or on the explicit LEFT-AT-INIT list,
    or the conversion REFUSES and names the tensor;
  * a shape disagreement REFUSES and names both shapes;
  * a geometry disagreement with the target config REFUSES and names the field;
  * the emitted table is printed, and the same table is stored in the seed's
    provenance stamp, so the audit does not depend on anyone having kept a log.

⚠️⚠️ **TWO DECLARED, UNAVOIDABLE LOSSES. READ THEM BEFORE QUOTING THIS SEED AS
"DINOv3".** They are recorded in the stamp under ``declared_losses`` and printed
by every run of this tool:

  1. **POSITIONAL INFORMATION DOES NOT TRANSFER.** DINOv3 is RoPE-only: its
     state dict contains **no** learned absolute-position table (verified on the
     ViT-L/16 snapshot: 415 tensors, none positional). ``ViTEncoder`` has the
     mirror image — a learned ``pos`` table and no RoPE. ``pos`` is therefore
     LEFT AT ITS OWN INIT and is the one target tensor on the left-at-init list.
     A seeded trunk is DINOv3's *content* at a *fresh* positional code.
  2. **CLS AND REGISTER TOKENS ARE DROPPED.** ``ViTEncoder`` has neither, so
     ``embeddings.cls_token`` / ``embeddings.register_tokens`` /
     ``embeddings.mask_token`` are allow-listed skips. DINOv3 was trained with
     those tokens participating in every attention; without them the function is
     not bit-identical to the published trunk.

⇒ ⛔ The Observer-Effect step-0 control (`PREREG_V7F.md` §6.2b) measures THIS
trunk, not the published one. Say so in any report that uses the ρ 0.91 figure.

⭐ WHAT *IS* EXACT. Everything else transfers without approximation, including
the two places a naive port would silently lose weights:

  * **qkv.** DINOv3 keeps q/k/v as three separate projections and has
    ``key_bias: false``; ``nn.MultiheadAttention`` wants one ``in_proj_weight``
    [3D, D] and one ``in_proj_bias`` [3D]. The bias is assembled as
    ``cat(q_bias, ZEROS(D), v_bias)`` — a zero bias is *exactly* "no bias", so
    this is a transfer, not an approximation.
  * **LayerScale.** DINOv3 computes ``x + ls * branch(norm(x))``; ``ViTEncoder``
    computes ``x + branch(norm(x))``. Since each branch ENDS in a linear map,
    ``ls`` folds into that map exactly: ``W <- diag(ls) @ W`` and ``b <- ls * b``
    on ``attn.out_proj`` and on ``mlp.2``. Folding is ON by default and recorded
    in the stamp; ``--no-fold-layer-scale`` refuses instead of dropping it.

⛔ ``--target vit5`` IS REFUSED ON PURPOSE. ``ViT5Encoder`` is RMSNorm (no bias),
bias-free qkv/proj and QK-Norm; a "port" onto it would have to DROP DINOv3's
LayerNorm biases and its q/v biases, which is `PREREG_V7F.md` §10 D1 option B —
*"lossy and unauditable"*, and the option the pre-registration rejects.

Usage (the source is a LOCAL snapshot path; this tool never downloads):

    python stack/scripts/dinov3_seed_checkpoint.py \
        --source <hf snapshot dir or a .safetensors/.pt file> \
        --out    <seed.pt> \
        --enc-dim 768 --enc-depth 12 --enc-heads 12 --patch 16 \
        --frame-h 256 --frame-w 640 --in-channels 3

Consumed by ``train_v6_staged.py --init-encoder-from <seed.pt>``.

Evidence class of every number this tool prints: MEASURED (ours; sha256 over the
source file and over the emitted tensors, shapes read from the tensors
themselves and from a really-constructed ``ViTEncoder``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
if str(_HERE.parents[1]) not in sys.path:            # stack/ on the path
    sys.path.insert(0, str(_HERE.parents[1]))

from tanitad.config import EncoderConfig             # noqa: E402
from tanitad.models.encoder import ViTEncoder        # noqa: E402

SEED_FORMAT = "tanitad-encoder-seed"
SEED_FORMAT_VERSION = 1
SEED_MARKER_KEY = "_tanitad_encoder_seed"

#: Source tensors that legitimately have no home in ``ViTEncoder``. ⛔ THIS LIST
#: IS THE WHOLE POINT: anything NOT on it and not mapped makes the conversion
#: REFUSE, because an unexplained leftover is how a partial seed passes for a
#: complete one. Extend it only with ``--allow-unmapped`` and a reason.
DINOV3_ALLOW_UNMAPPED: tuple[str, ...] = (
    "embeddings.cls_token",        # ViTEncoder emits patch tokens only
    "embeddings.register_tokens",  # ViTEncoder has no registers (ViT5Encoder does)
    "embeddings.mask_token",       # SSL-only; never used at inference
)

#: Target tensors that legitimately stay at their own init. ⚠️ `pos` is here
#: because DINOv3 carries NO absolute-position table at all (RoPE-only) — see
#: declared loss (1) in the module docstring. Nothing else may join this list.
TARGET_LEFT_AT_INIT: tuple[str, ...] = ("pos",)

#: Published DINOv3 geometries, used ONLY to name the variant in the stamp.
#: A geometry that matches none of these is stamped "custom" and still converts.
DINOV3_VARIANTS: dict[tuple[int, int, int, int], str] = {
    (384, 12, 6, 16): "vits16",
    (768, 12, 12, 16): "vitb16",
    (1024, 24, 16, 16): "vitl16",
    (1280, 32, 20, 16): "vith16plus",
}


class SeedError(RuntimeError):
    """A refusal. Every message names the tensor or the field that caused it."""


# ---------------------------------------------------------------------------
# source reading
# ---------------------------------------------------------------------------

def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_state(state: dict) -> str:
    """A content hash over a state dict that does NOT depend on dict order."""
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        h.update(k.encode("utf-8"))
        h.update(str(tuple(v.shape)).encode("utf-8"))
        h.update(str(v.dtype).encode("utf-8"))
        h.update(v.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()


def load_source(source) -> tuple[dict, dict, dict]:
    """``source`` (a dir, a .safetensors, or a torch file) -> (state, cfg, prov).

    ⛔ NEVER DOWNLOADS. HF quota is a hard ceiling and this tool runs on boxes
    behind a TLS proxy; a converter that silently reaches the network is a
    converter that fails in the one place it must not.
    """
    p = Path(source)
    if not p.exists():
        raise SeedError(f"--source {p} does not exist (this tool never "
                        f"downloads; pass a LOCAL snapshot path)")

    files: list[Path]
    if p.is_dir():
        cands = [p / "model.safetensors", p / "pytorch_model.bin",
                 p / "model.pt", p / "model.pth"]
        files = [c for c in cands if c.exists()]
        if not files:
            found = sorted(x.name for x in p.iterdir() if x.is_file())
            raise SeedError(
                f"--source {p} is a directory with no recognised weight file "
                f"(looked for model.safetensors / pytorch_model.bin / model.pt "
                f"/ model.pth). It contains: {found}")
        wfile = files[0]
        cfg_file = p / "config.json"
    else:
        wfile, cfg_file = p, p.parent / "config.json"

    if wfile.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(wfile))
    else:
        obj = torch.load(wfile, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            for k in ("state_dict", "model", "module"):
                if k in obj and isinstance(obj[k], dict):
                    obj = obj[k]
                    break
        if not isinstance(obj, dict):
            raise SeedError(f"{wfile} did not deserialise to a state dict "
                            f"(got {type(obj).__name__})")
        state = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}

    cfg = {}
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))

    prov = {
        "source_path": str(p),
        "source_weight_file": str(wfile),
        "source_sha256": _sha256(wfile),
        "source_bytes": wfile.stat().st_size,
        "source_config_file": str(cfg_file) if cfg_file.exists() else None,
        "source_config_sha256": _sha256(cfg_file) if cfg_file.exists() else None,
    }
    return state, cfg, prov


def strip_prefix(state: dict) -> tuple[dict, str]:
    """Drop a uniform wrapper prefix (``model.``, ``backbone.``, ...).

    ⚠️ Only a prefix shared by EVERY key is stripped. A partial prefix would be
    a different model shape, and guessing there is exactly the silent-partial
    failure this file exists to refuse."""
    for pre in ("model.", "backbone.", "module.", "dinov3."):
        if state and all(k.startswith(pre) for k in state):
            return {k[len(pre):]: v for k, v in state.items()}, pre
    return dict(state), ""


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"^layer\.(\d+)\.")


def infer_source_geometry(state: dict, cfg: dict) -> dict:
    """Geometry read from THE TENSORS, cross-checked against ``config.json``.

    ⛔ The tensors win. A config file is a claim about a checkpoint; the shapes
    are the checkpoint. Where the two disagree this REFUSES rather than picking
    one — a converter that quietly prefers the metadata is how a run ends up
    seeded from a different model than its record says."""
    w = state.get("embeddings.patch_embeddings.weight")
    if w is None:
        raise SeedError(
            "source has no 'embeddings.patch_embeddings.weight'. This tool "
            "reads the HuggingFace DINOv3ViTModel layout (the one "
            "`train_v6_staged.py`'s O7 teacher and `dino_precompute.py` both "
            f"load). Top-level keys seen: {sorted(state)[:8]}")
    if w.dim() != 4 or w.shape[2] != w.shape[3]:
        raise SeedError(f"patch-embed weight has shape {tuple(w.shape)}; "
                        f"expected [D, C, P, P]")
    d_model, in_ch, patch = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])

    depth = 0
    for k in state:
        m = _LAYER_RE.match(k)
        if m:
            depth = max(depth, int(m.group(1)) + 1)
    if depth == 0:
        raise SeedError("source has no 'layer.<i>.*' tensors — this is not a "
                        "HuggingFace DINOv3ViTModel state dict")

    inter = state.get("layer.0.mlp.up_proj.weight")
    if inter is None:
        raise SeedError("source has no 'layer.0.mlp.up_proj.weight'; the MLP "
                        "layout is not the one this tool maps")
    intermediate = int(inter.shape[0])

    n_heads = int(cfg.get("num_attention_heads", 0)) or 0
    geo = {"d_model": d_model, "depth": depth, "n_heads": n_heads,
           "patch_size": patch, "in_channels": in_ch,
           "intermediate_size": intermediate,
           "n_heads_source": "config.json" if n_heads else "UNKNOWN"}

    for field, cfg_key in (("d_model", "hidden_size"),
                           ("depth", "num_hidden_layers"),
                           ("patch_size", "patch_size"),
                           ("in_channels", "num_channels"),
                           ("intermediate_size", "intermediate_size")):
        if cfg_key in cfg and int(cfg[cfg_key]) != geo[field]:
            raise SeedError(
                f"source config.json says {cfg_key}={cfg[cfg_key]} but the "
                f"tensors say {field}={geo[field]}. The config and the weights "
                f"describe different models; refusing rather than guessing.")
    geo["variant"] = DINOV3_VARIANTS.get(
        (d_model, depth, n_heads, patch), "custom")
    return geo


def assert_target_geometry(src_geo: dict, enc_cfg: EncoderConfig) -> None:
    """REFUSE when the DINOv3 geometry does not match the target ViTEncoder."""
    checks = [
        ("d_model", src_geo["d_model"], enc_cfg.d_model, "--enc-dim"),
        ("depth", src_geo["depth"], enc_cfg.depth, "--enc-depth"),
        ("patch_size", src_geo["patch_size"], enc_cfg.patch_size, "--patch"),
        ("in_channels", src_geo["in_channels"], enc_cfg.in_channels,
         "--in-channels"),
    ]
    if src_geo.get("n_heads"):
        checks.append(("n_heads", src_geo["n_heads"], enc_cfg.n_heads,
                       "--enc-heads"))
    bad = [(f, s, t, flag) for f, s, t, flag in checks if int(s) != int(t)]
    if bad:
        lines = "\n".join(
            f"    {f}: DINOv3 has {s}, target has {t}   (set {flag} {s})"
            for f, s, t, flag in bad)
        raise SeedError(
            "⛔ GEOMETRY MISMATCH — the DINOv3 checkpoint and the target "
            "ViTEncoder are different models:\n" + lines +
            "\n  A seed built across a geometry mismatch is the silent-partial "
            "failure this tool exists to refuse.")
    exp_inter = int(4 * enc_cfg.d_model)
    if int(src_geo["intermediate_size"]) != exp_inter:
        raise SeedError(
            f"⛔ MLP WIDTH MISMATCH: DINOv3's intermediate_size is "
            f"{src_geo['intermediate_size']}, but ViTEncoder's Block is built "
            f"at mlp_ratio 4.0 == {exp_inter}. ViTEncoder does not expose "
            f"mlp_ratio, so this cannot be reconciled by a flag.")
    if enc_cfg.in_channels != 3:
        raise SeedError(
            f"⛔ --in-channels {enc_cfg.in_channels}: DINOv3's patch embed is "
            f"3-channel and this tool does NOT tile or rescale it. v7f's own "
            f"decision D2 is `--newest-frame-only --in-channels 3` precisely "
            f"because that is the only setting under which the transfer is "
            f"exact.")


# ---------------------------------------------------------------------------
# the mapping
# ---------------------------------------------------------------------------

def build_mapping(depth: int) -> list[dict]:
    """The complete DINOv3 -> ViTEncoder name-mapping table.

    One row per TARGET tensor. ``sources`` lists every source tensor consumed;
    ``transform`` names what is done to them. This table is printed by the tool
    and stored verbatim in the seed's provenance stamp."""
    rows: list[dict] = [
        {"target": "patch.weight",
         "sources": ["embeddings.patch_embeddings.weight"],
         "transform": "identity"},
        {"target": "patch.bias",
         "sources": ["embeddings.patch_embeddings.bias"],
         "transform": "identity"},
    ]
    for i in range(depth):
        s = f"layer.{i}."
        t = f"blocks.{i}."
        rows += [
            {"target": t + "norm1.weight", "sources": [s + "norm1.weight"],
             "transform": "identity"},
            {"target": t + "norm1.bias", "sources": [s + "norm1.bias"],
             "transform": "identity"},
            {"target": t + "attn.in_proj_weight",
             "sources": [s + "attention.q_proj.weight",
                         s + "attention.k_proj.weight",
                         s + "attention.v_proj.weight"],
             "transform": "qkv_row_concat"},
            {"target": t + "attn.in_proj_bias",
             "sources": [s + "attention.q_proj.bias",
                         s + "attention.k_proj.bias",   # OPTIONAL: key_bias=false
                         s + "attention.v_proj.bias"],
             "transform": "qkv_bias_concat_zero_fill_missing"},
            {"target": t + "attn.out_proj.weight",
             "sources": [s + "attention.o_proj.weight",
                         s + "layer_scale1.lambda1"],
             "transform": "layerscale_fold_rows"},
            {"target": t + "attn.out_proj.bias",
             "sources": [s + "attention.o_proj.bias",
                         s + "layer_scale1.lambda1"],
             "transform": "layerscale_fold_vector"},
            {"target": t + "norm2.weight", "sources": [s + "norm2.weight"],
             "transform": "identity"},
            {"target": t + "norm2.bias", "sources": [s + "norm2.bias"],
             "transform": "identity"},
            {"target": t + "mlp.0.weight", "sources": [s + "mlp.up_proj.weight"],
             "transform": "identity"},
            {"target": t + "mlp.0.bias", "sources": [s + "mlp.up_proj.bias"],
             "transform": "identity"},
            {"target": t + "mlp.2.weight",
             "sources": [s + "mlp.down_proj.weight", s + "layer_scale2.lambda1"],
             "transform": "layerscale_fold_rows"},
            {"target": t + "mlp.2.bias",
             "sources": [s + "mlp.down_proj.bias", s + "layer_scale2.lambda1"],
             "transform": "layerscale_fold_vector"},
        ]
    rows += [
        {"target": "norm.weight", "sources": ["norm.weight"],
         "transform": "identity"},
        {"target": "norm.bias", "sources": ["norm.bias"],
         "transform": "identity"},
    ]
    return rows


#: source tensors that a row may name but that need not exist (DINOv3 sets
#: ``key_bias: false``, so ``k_proj.bias`` is genuinely absent). Anything else
#: named by a row and missing is a REFUSAL.
_OPTIONAL_SOURCE_SUFFIX = ("attention.k_proj.bias",)


def _is_optional(key: str) -> bool:
    return any(key.endswith(s) for s in _OPTIONAL_SOURCE_SUFFIX)


def _apply_row(row: dict, state: dict, *, d_model: int,
               fold_layer_scale: bool) -> torch.Tensor:
    tr, src = row["transform"], row["sources"]
    if tr == "identity":
        return state[src[0]].clone().float()
    if tr == "qkv_row_concat":
        return torch.cat([state[k].float() for k in src], dim=0).clone()
    if tr == "qkv_bias_concat_zero_fill_missing":
        parts = []
        for k in src:
            v = state.get(k)
            if v is None:
                # a zero bias IS "no bias" — exact, not an approximation
                parts.append(torch.zeros(d_model, dtype=torch.float32))
            else:
                parts.append(v.float())
        return torch.cat(parts, dim=0).clone()
    if tr in ("layerscale_fold_rows", "layerscale_fold_vector"):
        base = state[src[0]].float().clone()
        ls = state.get(src[1])
        if ls is None:
            return base                       # no LayerScale in this checkpoint
        if not fold_layer_scale:
            raise SeedError(
                f"⛔ {src[1]} exists but --no-fold-layer-scale was passed. "
                f"ViTEncoder's Block has NO LayerScale parameter, so not "
                f"folding means silently DROPPING a trained per-channel gain "
                f"— the exact partial-seed failure this tool refuses. Either "
                f"fold (the default, and exact) or use a target that has "
                f"LayerScale.")
        ls = ls.float()
        return base * (ls[:, None] if tr == "layerscale_fold_rows" else ls)
    raise SeedError(f"unknown transform {tr!r} in the mapping table")


def build_seed(state: dict, enc_cfg: EncoderConfig, *,
               allow_unmapped: tuple[str, ...] = (),
               fold_layer_scale: bool = True,
               src_geo: dict | None = None) -> tuple[dict, dict]:
    """DINOv3 ``state`` -> (encoder state dict, report). Raises ``SeedError``.

    The target names and shapes come from a REALLY CONSTRUCTED ``ViTEncoder``,
    never from a hand-written list — so the mapping cannot drift away from the
    class it claims to target."""
    ref = ViTEncoder(enc_cfg)
    target = {k: tuple(v.shape) for k, v in ref.state_dict().items()}

    geo = src_geo or infer_source_geometry(state, {})
    rows = build_mapping(int(enc_cfg.depth))

    out: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    shape_errors: list[str] = []
    missing_sources: list[str] = []

    for row in rows:
        tgt = row["target"]
        if tgt not in target:
            raise SeedError(
                f"⛔ the mapping names target tensor {tgt!r}, which "
                f"ViTEncoder does not have. The mapping table and the class "
                f"have diverged — fix build_mapping(), never the class.")
        needed = [k for k in row["sources"] if not _is_optional(k)]
        absent = [k for k in needed if k not in state]
        if absent:
            missing_sources += absent
            continue
        try:
            val = _apply_row(row, state, d_model=int(enc_cfg.d_model),
                             fold_layer_scale=fold_layer_scale)
        except RuntimeError as exc:
            # ⚠️ A concat/broadcast that dies inside the transform IS a shape
            # mismatch. Letting torch's own message escape would name neither
            # the target tensor nor the source that is the wrong size.
            shapes = ", ".join(
                f"{k}{tuple(state[k].shape)}" for k in row["sources"]
                if k in state)
            shape_errors.append(
                f"    {tgt}: {row['transform']} failed on [{shapes}] "
                f"(ViTEncoder wants {target[tgt]}) — {exc}")
            continue
        if tuple(val.shape) != target[tgt]:
            shape_errors.append(
                f"    {tgt}: built {tuple(val.shape)} from "
                f"{row['sources']} ({row['transform']}), but ViTEncoder wants "
                f"{target[tgt]}")
            continue
        out[tgt] = val
        consumed.update(k for k in row["sources"] if k in state)

    if shape_errors:
        raise SeedError("⛔ SHAPE MISMATCH — refusing to write a seed:\n"
                        + "\n".join(shape_errors))
    if missing_sources:
        raise SeedError(
            "⛔ THE SOURCE IS MISSING TENSORS THE MAPPING NEEDS "
            f"({len(missing_sources)}): {sorted(set(missing_sources))[:8]}\n"
            "  A seed built around a hole is a partial seed. Refusing.")

    allowed = set(DINOV3_ALLOW_UNMAPPED) | set(allow_unmapped)
    leftover = sorted(k for k in state if k not in consumed and k not in allowed)
    if leftover:
        raise SeedError(
            f"⛔ {len(leftover)} SOURCE TENSOR(S) WERE NEITHER MAPPED NOR "
            f"ALLOW-LISTED: {leftover[:12]}\n"
            "  Every one of these is trained weight that would be thrown away "
            "silently. Either extend build_mapping(), or pass "
            "--allow-unmapped <key> for each one WITH a reason in the report.")

    left_at_init = sorted(k for k in target if k not in out)
    unexpected_init = [k for k in left_at_init if k not in TARGET_LEFT_AT_INIT]
    if unexpected_init:
        raise SeedError(
            f"⛔ {len(unexpected_init)} TARGET TENSOR(S) WOULD BE LEFT AT "
            f"RANDOM INIT AND ARE NOT ON THE LEFT-AT-INIT LIST: "
            f"{unexpected_init[:12]}\n"
            "  This is the silent partial seed. Refusing.")

    skipped = sorted(k for k in state if k in allowed and k not in consumed)
    report = {
        "n_source_tensors": len(state),
        "n_target_tensors": len(target),
        "n_mapped": len(out),
        "n_skipped_allowlist": len(skipped),
        "n_left_at_init": len(left_at_init),
        "mapped_keys": sorted(out),
        "skipped_source_keys": skipped,
        "left_at_init_keys": left_at_init,
        "consumed_source_keys": sorted(consumed),
        "mapping": rows,
        "layer_scale_folded": bool(
            fold_layer_scale and any(k.endswith("layer_scale1.lambda1")
                                     for k in state)),
        "source_geometry": geo,
        "target_geometry": {
            "class": "ViTEncoder",
            "d_model": int(enc_cfg.d_model), "depth": int(enc_cfg.depth),
            "n_heads": int(enc_cfg.n_heads),
            "patch_size": int(enc_cfg.patch_size),
            "in_channels": int(enc_cfg.in_channels),
            # ⚠️ RESOLVED, never the raw field. `image_width=None` and
            # `image_width == image_size` are the SAME geometry by
            # EncoderConfig.image_hw()'s own contract, and a loader comparing
            # the raw field refuses two identical geometries (measured here:
            # the converter writes None, `build_stack_from_args` writes 32).
            "image_size": int(enc_cfg.image_hw()[0]),
            "image_width": int(enc_cfg.image_hw()[1]),
            "n_tokens": int(ref.n_tokens),
            "token_grid": [int(ref.grid_h), int(ref.grid_w)],
        },
        "target_param_count": int(sum(v.numel()
                                      for v in ref.state_dict().values())),
        "seeded_param_count": int(sum(v.numel() for v in out.values())),
    }
    return out, report


DECLARED_LOSSES = [
    "POSITIONAL: DINOv3 is RoPE-only and carries NO learned absolute-position "
    "table; ViTEncoder's `pos` is LEFT AT ITS OWN INIT. The seeded trunk is "
    "DINOv3 content at a fresh positional code.",
    "TOKENS: DINOv3's CLS / register / mask tokens are dropped (ViTEncoder has "
    "none). DINOv3 was trained with those tokens attending, so the seeded "
    "forward is not bit-identical to the published trunk.",
    "⇒ The Observer-Effect step-0 control (PREREG_V7F §6.2b) therefore "
    "measures THIS trunk, not the published DINOv3. Any report that quotes the "
    "published rho 0.91 alongside this seed must say so.",
]


def make_provenance(src_prov: dict, report: dict, *, model_id: str | None,
                    argv: list[str], stripped_prefix: str,
                    encoder_sd: dict) -> dict:
    geo = report["source_geometry"]
    return {
        "format": SEED_FORMAT,
        "format_version": SEED_FORMAT_VERSION,
        "tool": "stack/scripts/dinov3_seed_checkpoint.py",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": list(argv),
        "dinov3_model_id": model_id,
        "dinov3_variant": geo.get("variant"),
        "source_prefix_stripped": stripped_prefix or None,
        **src_prov,
        "source_geometry": geo,
        "target_geometry": report["target_geometry"],
        "mapping": report["mapping"],
        "n_mapped": report["n_mapped"],
        "n_skipped_allowlist": report["n_skipped_allowlist"],
        "n_left_at_init": report["n_left_at_init"],
        "mapped_keys": report["mapped_keys"],
        "skipped_source_keys": report["skipped_source_keys"],
        "left_at_init_keys": report["left_at_init_keys"],
        "layer_scale_folded": report["layer_scale_folded"],
        "target_param_count": report["target_param_count"],
        "seeded_param_count": report["seeded_param_count"],
        "declared_losses": list(DECLARED_LOSSES),
        "seed_sha256": _sha256_state(encoder_sd),
        "_evidence_class": "MEASURED (ours; sha256 over the source file and "
                           "over the emitted tensors)",
    }


def format_table(report: dict, *, max_rows: int = 24) -> str:
    """The human-readable audit table. Printed by the tool AND reproducible
    from the stamp alone, so the audit never depends on a kept log."""
    g, t = report["source_geometry"], report["target_geometry"]
    out = [
        "",
        "=== DINOv3 -> ViTEncoder SEED =============================",
        f"  source geometry : d={g['d_model']} depth={g['depth']} "
        f"heads={g.get('n_heads') or '?'} patch={g['patch_size']} "
        f"in_ch={g['in_channels']} mlp={g['intermediate_size']} "
        f"variant={g.get('variant')}",
        f"  target geometry : d={t['d_model']} depth={t['depth']} "
        f"heads={t['n_heads']} patch={t['patch_size']} "
        f"in_ch={t['in_channels']} image={t['image_size']}x"
        f"{t['image_width'] or t['image_size']} "
        f"tokens={t['n_tokens']} grid={t['token_grid']}",
        f"  layer-scale fold: {report['layer_scale_folded']}",
        "",
        f"  MAPPED            : {report['n_mapped']:4d} / "
        f"{report['n_target_tensors']} target tensors",
        f"  SKIPPED (allowed) : {report['n_skipped_allowlist']:4d} source "
        f"tensors  {report['skipped_source_keys']}",
        f"  LEFT AT INIT      : {report['n_left_at_init']:4d} target "
        f"tensors  {report['left_at_init_keys']}",
        f"  source tensors    : {report['n_source_tensors']} "
        f"({len(report['consumed_source_keys'])} consumed + "
        f"{report['n_skipped_allowlist']} allow-listed)",
        f"  params seeded     : {report['seeded_param_count']:,} of "
        f"{report['target_param_count']:,} "
        f"({100.0 * report['seeded_param_count'] / max(1, report['target_param_count']):.2f} %)",
        "",
        "  name mapping (one row per target tensor; layer rows abridged):",
    ]
    shown = 0
    for row in report["mapping"]:
        if shown >= max_rows:
            out.append(f"    ... and {len(report['mapping']) - shown} more rows "
                       f"(the FULL table is in the seed's provenance stamp)")
            break
        srcs = ", ".join(row["sources"])
        out.append(f"    {row['target']:<32s} <- {srcs}   [{row['transform']}]")
        shown += 1
    out += ["", "  ⚠️ DECLARED LOSSES (recorded in the stamp):"]
    out += [f"    - {x}" for x in DECLARED_LOSSES]
    out += ["=" * 60, ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Convert a LOCAL DINOv3 checkpoint into an encoder-only "
                    "seed that train_v6_staged.py --init-encoder-from accepts.")
    ap.add_argument("--source", required=True,
                    help="LOCAL HF snapshot dir, or a .safetensors/.pt file. "
                         "⛔ never downloaded — HF quota is a hard ceiling")
    ap.add_argument("--out", required=True, help="seed checkpoint to write")
    ap.add_argument("--model-id", default=None,
                    help="e.g. facebook/dinov3-vitb16-pretrain-lvd1689m — "
                         "recorded in the stamp (inferred from the cache path "
                         "when it looks like an HF hub layout)")
    ap.add_argument("--target", choices=("vit", "vit5"), default="vit",
                    help="vit = ViTEncoder (the only faithful target). vit5 is "
                         "REFUSED: PREREG_V7F D1 option B, lossy+unauditable")
    # ---- target geometry (the ViTEncoder v7f will build) -------------------
    ap.add_argument("--enc-dim", type=int, default=768)
    ap.add_argument("--enc-depth", type=int, default=12)
    ap.add_argument("--enc-heads", type=int, default=12)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--frame-h", type=int, default=256)
    ap.add_argument("--frame-w", type=int, default=640)
    ap.add_argument("--in-channels", type=int, default=3)
    # ---- refusal knobs -----------------------------------------------------
    ap.add_argument("--allow-unmapped", action="append", default=[],
                    metavar="KEY",
                    help="extend the allow-list of source tensors that may go "
                         "unmapped. Repeatable. Each use belongs in the report "
                         "WITH a reason")
    ap.add_argument("--no-fold-layer-scale", action="store_true",
                    help="⛔ refuse instead of folding LayerScale into the "
                         "branch output projections (the fold is EXACT; this "
                         "exists so the choice is visible, not silent)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the table and write nothing")
    ap.add_argument("--json", default=None,
                    help="also write the provenance stamp as standalone JSON")
    return ap


def _infer_model_id(source: Path) -> str | None:
    """`.../models--facebook--dinov3-vitb16-.../snapshots/<sha>` -> the repo id."""
    for part in Path(source).resolve().parts:
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "/", 1).replace("--", "-")
    return None


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.target == "vit5":
        raise SystemExit(
            "⛔ --target vit5 is REFUSED. ViT5Encoder is RMSNorm (no bias), "
            "bias-free qkv/proj and QK-Norm, so a DINOv3 port onto it must "
            "DROP the LayerNorm biases and the q/v biases. That is "
            "PREREG_V7F.md §10 D1 option B — 'lossy and unauditable' — and it "
            "would make the Observer-Effect step-0 control measure something "
            "that is not DINOv3. Use --target vit (ViTEncoder), and launch v7f "
            "WITHOUT --vit5-encoder.")

    enc_cfg = EncoderConfig(
        in_channels=int(a.in_channels), image_size=int(a.frame_h),
        image_width=(None if int(a.frame_w) == int(a.frame_h)
                     else int(a.frame_w)),
        patch_size=int(a.patch), d_model=int(a.enc_dim),
        depth=int(a.enc_depth), n_heads=int(a.enc_heads))

    try:
        raw, cfg, src_prov = load_source(a.source)
        state, prefix = strip_prefix(raw)
        src_geo = infer_source_geometry(state, cfg)
        assert_target_geometry(src_geo, enc_cfg)
        encoder_sd, report = build_seed(
            state, enc_cfg, allow_unmapped=tuple(a.allow_unmapped),
            fold_layer_scale=not a.no_fold_layer_scale, src_geo=src_geo)
    except SeedError as e:
        raise SystemExit(f"[dinov3-seed] {e}") from None

    model_id = a.model_id or _infer_model_id(Path(a.source))
    prov = make_provenance(src_prov, report, model_id=model_id,
                           argv=list(argv if argv is not None else sys.argv[1:]),
                           stripped_prefix=prefix, encoder_sd=encoder_sd)
    prov["source_config"] = cfg
    print(format_table(report))
    print(f"[dinov3-seed] model_id={model_id} variant={src_geo.get('variant')} "
          f"source_sha256={src_prov['source_sha256'][:16]}… "
          f"seed_sha256={prov['seed_sha256'][:16]}…", flush=True)

    if a.dry_run:
        print("[dinov3-seed] --dry-run: nothing written", flush=True)
        return 0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({SEED_MARKER_KEY: SEED_FORMAT_VERSION,
                "encoder": encoder_sd,
                "_provenance": prov}, out)
    print(f"[dinov3-seed] wrote {out} "
          f"({out.stat().st_size / 1e6:.1f} MB, {len(encoder_sd)} tensors)",
          flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(prov, indent=2), encoding="utf-8")
        print(f"[dinov3-seed] stamp -> {a.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
