"""LC (the LoRA arm) for the T2 stage-5 driver: a STANDALONE fold of a v8F-mini LoRA state dict into the T2 loop's DINOv3-L q/v weights, on CPU.

    W_bf16  =  bf16( fp32(W)  +  ( fp32(B) @ fp32(A) ) * (alpha / r) )                     q and v of every one of the 24 blocks; k, o, biases untouched

This is the training side's own expression, op for op (`stack/tanitad/models/v8f_lora_encoder.py`: `LoRALinear.delta_weight` = `(B.float() @ A.float()) * scaling`, `scaling = float(alpha) / int(r)`; `_fold`:
`(W.float() + delta).to(W.dtype)`; `merge()`). It imports NOTHING from `tanitad` and nothing from the cache builder (`pull_encode_pooled`): only `torch`. The tests prove it equal to the training side's
`merged_encoder()` on the same CPU (`tests_lc/`).

Where it runs: INSIDE `t2a_v8f_encoder.InLoopEncoder.__init__`, after `AutoModel.from_pretrained(...)` returned a CPU bf16 model and BEFORE `.eval().to(device)`. `lora_fold_active` swaps `InLoopEncoder` for a subclass
for the duration of one `build_arm`; the subclass hooks `AutoModel.from_pretrained` for the duration of its `__init__` (exactly once, or it refuses). Two independent guards assert the pinned digest: in the hook (right
after the fold) and on the finished encoder (wherever its weights live). A fold that did not happen cannot reach a rollout.

The pin (`arm5.encoder`, built by `make_stage5.py --encoder lora`): per-seed source-checkpoint sha256, sha256 of the LoRA tensors, the merged q/v digest (one value + one per tensor, so a mismatch NAMES the tensors).
The digest is independent of the HF wrapper's parameter prefix (`model.layer.0...` vs `layer.0...`): names are normalised to `layer.<i>.attention.<q|v>_proj.weight`.

UNVERIFIED (needs box 2): the digest is a function of the CPU's fp32 matmul. Two CPUs with different BLAS kernels can differ in the last fp32 bit of `B @ A` and, in expectation well under one weight of 5e7,
in one bf16 ulp after the sum; the preflight (`check_l_ckpt.py --expect-merged-sha256` on box 2, or the driver's own preflight fold) refuses LOUDLY and names the tensors rather than running an encoder that is not the pinned one.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re

MID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
#: sha256 of the model.safetensors the cache builder encoded with and the LoRA trainer REFUSES to deviate from (`v8f_lora_encoder.BUILDER_MODEL_SHA256`).
BASE_WEIGHTS_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
KIND = "dinov3_lora_merged"
N_BLOCKS = 24
TARGETS = ("q", "v")
SHA_RE = re.compile(r"[0-9a-f]{64}")
_LORA_NAME = re.compile(r"(?:^|.*\.)layer\.(\d+)\.attention\.([qv])_proj\.lora_([AB])$")
_W_NAME = re.compile(r"(?:^|.*\.)layer\.(\d+)\.attention\.([qv])_proj\.weight$")
FORMULA = "W_bf16 = bf16( fp32(W) + (fp32(B) @ fp32(A)) * (alpha / r) ), q and v of every block, on CPU, before .to(device)"
MIRRORS = "stack/tanitad/models/v8f_lora_encoder.py LoRALinear.delta_weight + _fold + merge (tip a9da6e8f)"


def canon(layer: int, t: str) -> str:
    return f"layer.{int(layer)}.attention.{t}_proj.weight"


def expected_names(blocks: int = N_BLOCKS, targets=TARGETS) -> list:
    return [canon(i, t) for i in range(blocks) for t in targets]


# ------------------------------------------------------------------------------------------------------------------------------ hashing
def tensor_sha256(t) -> str:
    import torch
    x = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(f"{x.dtype}|{tuple(x.shape)}|".encode())
    h.update(x.view(torch.uint8).numpy().tobytes() if x.numel() else b"")
    return h.hexdigest()


def digest_of(per_tensor: dict) -> str:
    """The aggregate: sha256 of the sorted `name sha` lines (so the whole value is a function of every tensor and its name)."""
    return hashlib.sha256("\n".join(f"{k} {per_tensor[k]}" for k in sorted(per_tensor)).encode()).hexdigest()


def file_sha256(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


# ------------------------------------------------------------------------------------------------------------------------------ the blob
def load_blob(path) -> dict:
    """The L trainer's `ckpt_last.pt` (`v8f_mini_train.py` `save()`): `model` (the heads, B's layout), `opt`, `lora` ({meta, tensors}), `encoder_levers`, `cfg`, `step`, `seed`, `arm`, `argv_sha`, ..."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_lora(sd: dict, *, r: int = 32, alpha: float = 32.0, targets=TARGETS, blocks: int = N_BLOCKS) -> dict:
    """`lora_state_dict()` -> {(layer, target): (A [r, in], B [out, r])}, fp32. STRICT: the meta must equal the declared (r, alpha, targets, blocks), exactly `blocks x len(targets) x 2` tensors, no extra, no missing,
    every one finite fp32 of the right shape. Raises ValueError with a named reason."""
    import torch
    if not isinstance(sd, dict) or "meta" not in sd or "tensors" not in sd:
        raise ValueError("LORA_BAD_FORMAT the `lora` entry is not {meta, tensors}")
    mt = sd["meta"]
    for k, want in (("r", int(r)), ("alpha", float(alpha)), ("targets", list(targets)), ("blocks", int(blocks))):
        if mt.get(k) != want:
            raise ValueError(f"LORA_META_MISMATCH meta.{k} = {mt.get(k)!r}, declared {want!r}")
    out, seen = {}, {}
    for name, ten in sd["tensors"].items():
        m = _LORA_NAME.fullmatch(name)
        if not m:
            raise ValueError(f"LORA_UNKNOWN_TENSOR {name!r}")
        i, t, ab = int(m.group(1)), m.group(2), m.group(3)
        if t not in targets or not (0 <= i < blocks):
            raise ValueError(f"LORA_UNKNOWN_TENSOR {name!r} (layer {i}, target {t})")
        if (i, t, ab) in seen:
            raise ValueError(f"LORA_DUPLICATE_TENSOR {name!r} and {seen[(i, t, ab)]!r} normalise to the same slot (two different prefixes in one state dict)")
        seen[(i, t, ab)] = name
        if ten.dtype != torch.float32:
            raise ValueError(f"LORA_BAD_DTYPE {name!r} is {ten.dtype}, the trainer saves fp32")
        if not bool(torch.isfinite(ten).all()):
            raise ValueError(f"LORA_NONFINITE {name!r}")
        out.setdefault((i, t), {})[ab] = ten
    need = {(i, t, ab) for i in range(blocks) for t in targets for ab in "AB"}
    if set(seen) != need:
        raise ValueError(f"LORA_TENSOR_SET got {len(seen)} tensors, expected {len(need)}; missing {sorted(need - set(seen))[:4]}, extra {sorted(set(seen) - need)[:4]}")
    res = {}
    for (i, t), d in out.items():
        A, B = d["A"], d["B"]
        if A.dim() != 2 or B.dim() != 2 or A.shape[0] != int(r) or B.shape[1] != int(r):
            raise ValueError(f"LORA_BAD_SHAPE layer {i} {t}: A {tuple(A.shape)} B {tuple(B.shape)} for r={r}")
        res[(i, t)] = (A, B)
    return res


def lora_tensors_sha256(parsed: dict) -> str:
    return digest_of({f"{canon(i, t)}.{ab}": tensor_sha256(x) for (i, t), pr in parsed.items() for ab, x in zip("AB", pr)})


# ------------------------------------------------------------------------------------------------------------------------------ the model side
def target_params(model, blocks: int = N_BLOCKS, targets=TARGETS) -> dict:
    """{(layer, target): the q_proj / v_proj `.weight` Parameter} found by NAME SUFFIX, whatever the HF wrapper's prefix. Refuses a model that does not have exactly blocks x targets of them."""
    found = {}
    for name, p in model.named_parameters():
        m = _W_NAME.fullmatch(name)
        if m and m.group(2) in targets:
            key = (int(m.group(1)), m.group(2))
            if key in found:
                raise ValueError(f"MODEL_AMBIGUOUS_PARAM {name!r} and another parameter are both layer {key[0]} {key[1]}")
            found[key] = p
    want = {(i, t) for i in range(blocks) for t in targets}
    if set(found) != want:
        raise ValueError(f"MODEL_TARGET_SET found {len(found)} q/v weights, expected {len(want)}; missing {sorted(want - set(found))[:4]}")
    return found


def model_digest(model, blocks: int = N_BLOCKS, targets=TARGETS) -> dict:
    tp = target_params(model, blocks, targets)
    per = {canon(i, t): tensor_sha256(p.data) for (i, t), p in tp.items()}
    return {"merged_qv_sha256": digest_of(per), "tensor_sha256": per}


def fold_lora(model, parsed: dict, *, r: int = 32, alpha: float = 32.0) -> list:
    """Fold in place. The model must be on CPU and bf16 (the fold is the CPU fp32 expression; a CUDA model would be a different matmul). -> the folded names."""
    import torch
    scaling = float(alpha) / int(r)
    tp = target_params(model, N_BLOCKS, sorted({t for _, t in parsed}))
    done = []
    with torch.no_grad():
        for (i, t), (A, B) in sorted(parsed.items()):
            w = tp[(i, t)]
            if w.device.type != "cpu" or w.dtype != torch.bfloat16:
                raise RuntimeError(f"FOLD_WRONG_PLACE {canon(i, t)} is on {w.device} as {w.dtype}: the fold runs on CPU, in bf16 weights, BEFORE .to(device)")
            if tuple(w.shape) != (B.shape[0], A.shape[1]):
                raise ValueError(f"FOLD_SHAPE {canon(i, t)}: W {tuple(w.shape)} vs B @ A {(B.shape[0], A.shape[1])}")
            delta = (B.float() @ A.float()) * scaling                                          # LoRALinear.delta_weight, op for op
            w.data = (w.data.float() + delta).to(w.dtype)                                      # _fold, op for op
            done.append(canon(i, t))
    return done


def load_base_model(snapshot=None, dtype=None):
    """The T2 loop's own load: `AutoModel.from_pretrained(snapshot, local_files_only=True, dtype=bf16)` on CPU, eval. Offline. `snapshot` defaults to the newest local snapshot of the DINOv3-L model."""
    import torch
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModel
    if snapshot is None:
        import t2a_v8f_encoder as E
        snapshot = E.find_local_snapshot()
    return AutoModel.from_pretrained(snapshot, local_files_only=True, dtype=dtype or torch.bfloat16).eval()


def snapshot_safetensors_sha256(snapshot) -> str:
    return file_sha256(os.path.join(snapshot, "model.safetensors"))


def build_pin_for(blob_path, snapshot=None, *, r: int = 32, alpha: float = 32.0, targets=TARGETS, blocks: int = N_BLOCKS, base_model=None) -> dict:
    """For ONE blob: sha256 of the file, the LoRA tensors' sha256, its meta, and the merged q/v digest of the real base weights (`snapshot`, or an already loaded CPU bf16 `base_model` that is DEEP-COPIED)."""
    import copy
    blob = load_blob(blob_path)
    parsed = parse_lora(blob["lora"], r=r, alpha=alpha, targets=targets, blocks=blocks)
    if base_model is None:
        base_model = load_base_model(snapshot)
    m = copy.deepcopy(base_model)
    fold_lora(m, parsed, r=r, alpha=alpha)
    d = model_digest(m, blocks, targets)
    return {"source_ckpt_sha256": file_sha256(blob_path), "lora_tensors_sha256": lora_tensors_sha256(parsed), "lora_meta": dict(blob["lora"]["meta"]), "merged_qv_sha256": d["merged_qv_sha256"],
            "merged_qv_tensor_sha256": d["tensor_sha256"], "ckpt_step": blob.get("step"), "ckpt_seed": blob.get("seed"), "ckpt_arm": blob.get("arm"), "ckpt_argv_sha": blob.get("argv_sha"),
            "b_abs_max": float(max(float(B.abs().max()) for _, B in parsed.values())), "n_b_zero_modules": int(sum(1 for _, B in parsed.values() if float(B.abs().sum()) == 0.0)),
            "blob_keys": sorted(blob)}


# ------------------------------------------------------------------------------------------------------------------------------ the manifest block
def check_lora_encoder_block(enc: dict, arms, ckpt_sha256: dict) -> list:
    """The `arm5.encoder` block of a LoRA arm. -> [reason] (named, empty = valid). `arms` = the cells' arm names, `ckpt_sha256` = arm5.ckpt_sha256 (one file serves the heads AND the LoRA)."""
    bad = []
    if enc.get("kind") != KIND:
        return [f"ENCODER_UNSUPPORTED {enc.get('kind')!r}"]
    arms = list(arms)
    if not isinstance(enc.get("r"), int) or isinstance(enc.get("r"), bool) or enc["r"] <= 0:
        bad.append(f"LORA_BAD_FIELD r {enc.get('r')!r}")
    if not isinstance(enc.get("alpha"), (int, float)) or isinstance(enc.get("alpha"), bool) or enc["alpha"] <= 0:
        bad.append(f"LORA_BAD_FIELD alpha {enc.get('alpha')!r}")
    if enc.get("targets") != list(TARGETS):
        bad.append(f"LORA_BAD_FIELD targets {enc.get('targets')!r}: the fold supports {list(TARGETS)} (the PREREG_V8F_LORA targets)")
    if enc.get("blocks") != N_BLOCKS:
        bad.append(f"LORA_BAD_FIELD blocks {enc.get('blocks')!r}, DINOv3-L has {N_BLOCKS}")
    if enc.get("model_id") != MID:
        bad.append(f"LORA_BAD_FIELD model_id {enc.get('model_id')!r}")
    if enc.get("base_weights_sha256") != BASE_WEIGHTS_SHA256:
        bad.append(f"LORA_BAD_FIELD base_weights_sha256 {enc.get('base_weights_sha256')!r} is not the cache builder's {BASE_WEIGHTS_SHA256}")
    names = set(expected_names())
    for key in ("source_ckpt_sha256", "lora_tensors_sha256", "merged_qv_sha256", "merged_qv_tensor_sha256"):
        v = enc.get(key)
        if not isinstance(v, dict) or sorted(v) != sorted(arms):
            bad.append(f"LORA_BAD_FIELD {key} keys {sorted(v) if isinstance(v, dict) else v!r}, expected {sorted(arms)}")
            continue
        for a in arms:
            if key == "merged_qv_tensor_sha256":
                pt = v[a]
                if not isinstance(pt, dict) or set(pt) != names or not all(SHA_RE.fullmatch(str(s)) for s in pt.values()):
                    bad.append(f"LORA_BAD_FIELD {key}[{a}] is not the {len(names)} q/v tensors with a sha256 each")
                elif isinstance(enc.get("merged_qv_sha256"), dict) and digest_of(pt) != enc["merged_qv_sha256"].get(a):
                    bad.append(f"LORA_PIN_INCONSISTENT {a}: merged_qv_sha256 is not the aggregate of its own per-tensor digests")
            elif not SHA_RE.fullmatch(str(v[a])):
                bad.append(f"LORA_BAD_FIELD {key}[{a}] {v[a]!r}")
    src = enc.get("source_ckpt_sha256")
    if isinstance(src, dict) and src != dict(ckpt_sha256 or {}):
        bad.append(f"LORA_SOURCE_MISMATCH encoder.source_ckpt_sha256 {src} != arm5.ckpt_sha256 {dict(ckpt_sha256 or {})}: the heads and the LoRA must come from ONE file per seed")
    ms = enc.get("merged_qv_sha256")
    if isinstance(ms, dict) and len(ms) > 1 and len(set(ms.values())) != len(ms):
        bad.append("LORA_PIN_INCONSISTENT the two seeds have the same merged digest (the same LoRA twice)")
    return bad


def preflight(enc: dict, ckpt_paths: dict, snapshot) -> list:
    """The driver's preflight on box 2 (off the stub): the local DINOv3-L snapshot is the cache builder's file, and folding each seed's LoRA on THIS machine's CPU gives the digest the manifest pinned.
    -> [failures] (named; MERGED_SHA_MISMATCH lists the differing tensors). Nothing is written."""
    import copy
    got = snapshot_safetensors_sha256(snapshot)
    if got != enc["base_weights_sha256"]:
        return [f"LORA_BASE_MISMATCH {snapshot}/model.safetensors sha256 {got} != the pinned {enc['base_weights_sha256']}"]
    base, bad = load_base_model(snapshot), []
    for arm, path in sorted(ckpt_paths.items()):
        try:
            parsed = parse_lora(load_blob(path)["lora"], r=enc["r"], alpha=enc["alpha"], targets=enc["targets"], blocks=enc["blocks"])
            m = copy.deepcopy(base)
            fold_lora(m, parsed, r=enc["r"], alpha=enc["alpha"])
            assert_pin(m, enc, arm, "preflight fold on this machine's CPU")
        except (AssertionError, ValueError, RuntimeError, KeyError) as e:                  # noqa: PERF203
            bad.append(str(e) if str(e).startswith(("MERGED_SHA_MISMATCH", "LORA_", "FOLD_", "MODEL_")) else f"LORA_FOLD_FAILED {arm}: {e!r}")
    return bad


# ------------------------------------------------------------------------------------------------------------------------------ the install
def assert_pin(model, enc: dict, arm: str, where: str) -> dict:
    d = model_digest(model)
    want = enc["merged_qv_sha256"][arm]
    if d["merged_qv_sha256"] != want:
        pt = enc["merged_qv_tensor_sha256"][arm]
        differ = sorted(k for k in pt if d["tensor_sha256"].get(k) != pt[k])
        raise AssertionError(f"MERGED_SHA_MISMATCH ({where}) arm {arm}: merged q/v digest {d['merged_qv_sha256']} != manifest {want}; {len(differ)} of {len(pt)} tensors differ, first: {differ[:6]}")
    return d


@contextlib.contextmanager
def lora_fold_active(blob_path, enc: dict, arm: str):
    """For the duration of ONE `build_arm`: `t2a_v8f_encoder.InLoopEncoder` is a subclass that folds this arm's LoRA into the freshly loaded CPU bf16 weights (after `AutoModel.from_pretrained`, before
    `.to(device)`), asserts the pinned digest twice (right after the fold, and on the finished encoder), and carries `.lora_fold` (the report). Restores the class and `from_pretrained` on exit, always."""
    import t2a_v8f_encoder as E
    base_cls = E.InLoopEncoder
    got = {}

    class LoraInLoopEncoder(base_cls):
        lora_arm = arm

        def __init__(self, *a, **k):
            import transformers
            AM = transformers.AutoModel
            had = "from_pretrained" in vars(AM)
            saved = vars(AM).get("from_pretrained")
            orig = AM.from_pretrained
            calls = []

            def hooked(*aa, **kk):
                model = orig(*aa, **kk)
                calls.append(1)
                if len(calls) != 1:
                    raise RuntimeError("FOLD_NOT_APPLIED AutoModel.from_pretrained was called more than once inside InLoopEncoder.__init__")
                if file_sha256(blob_path) != enc["source_ckpt_sha256"][arm]:
                    raise AssertionError(f"HASH_MISMATCH_LORA_SOURCE {blob_path}: not the checkpoint the manifest pinned for {arm}")
                blob = load_blob(blob_path)
                parsed = parse_lora(blob["lora"], r=enc["r"], alpha=enc["alpha"], targets=enc["targets"], blocks=enc["blocks"])
                done = fold_lora(model, parsed, r=enc["r"], alpha=enc["alpha"])
                d = assert_pin(model, enc, arm, "after the fold, before .to(device)")
                got.update(arm=arm, folded=len(done), merged_qv_sha256=d["merged_qv_sha256"], fold_when="after AutoModel.from_pretrained, before .eval().to(device)")
                return model
            AM.from_pretrained = staticmethod(hooked)
            try:
                super().__init__(*a, **k)
            finally:
                if had:
                    AM.from_pretrained = saved
                else:
                    del AM.from_pretrained
            if len(calls) != 1 or not got:
                raise RuntimeError(f"FOLD_NOT_APPLIED the LoRA fold did not run for {arm} (from_pretrained calls: {len(calls)})")
            d = assert_pin(self.model, enc, arm, "on the finished encoder")
            self.lora_fold = dict(got, final_check=d["merged_qv_sha256"])

    E.InLoopEncoder = LoraInLoopEncoder
    try:
        yield LoraInLoopEncoder
    finally:
        E.InLoopEncoder = base_cls


def dumps(o) -> str:
    return json.dumps(o, indent=1, sort_keys=True)
