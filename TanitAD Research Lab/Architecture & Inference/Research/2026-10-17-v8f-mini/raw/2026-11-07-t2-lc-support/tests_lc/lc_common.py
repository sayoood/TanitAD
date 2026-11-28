"""Shared helpers of tests_lc/ (AGENT LC): paths with code_lc/ FIRST, the real DINOv3-L snapshot, one banked 256x640 cylinder frame, synthetic LoRA state dicts in the TRAINING side's own format, and the
training-side reference module (a verbatim copy of the tip's `v8f_lora_encoder.py` + the two cache-builder files, in tests_lc/ref/, loaded by file path).

Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest tests_lc/ -q      (-I: no PYTHONPATH; this file puts code_lc/ then code/ on sys.path)
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
LC, CODE, REF = f"{T2}/code_lc", f"{T2}/code", f"{HERE}/ref"
SCRATCH = "E:/Projects/TanitAD-artifacts/scratch_lc_t2v8f"
if sys.path[:1] != [LC]:
    sys.path.insert(0, LC)
if CODE not in sys.path:
    sys.path.append(CODE)

import torch  # noqa: E402

import t2a_v8f_encoder as E  # noqa: E402  (code/, the unchanged in-loop encoder)
import t2v8f_lora_merge as LM  # noqa: E402  (code_lc/)

assert os.path.dirname(os.path.abspath(LM.__file__)).replace("\\", "/") == LC
assert os.path.dirname(os.path.abspath(E.__file__)).replace("\\", "/") == CODE
FRAME_FILE = "E:/Projects/TanitAD-artifacts/refcv5cmp/data/eval/01be5919-6e61-45ba-a193-5f447efb4b51.v2ep.pt"
WSP = "E:/Projects/TanitAD-artifacts/v8f-wspedit"
HEAD_CKPT = {0: f"{WSP}/M2b_s0_wspedit_a10/ckpt_last.pt", 1: f"{WSP}/M2b_s1_wspedit_a10/ckpt_last.pt"}      # real trained heads (B's layout): the synthetic L blobs borrow them
ARMS = ("LC_s0", "LC_s1")
_C = {}


def snapshot() -> str:
    return E.find_local_snapshot()


def base_model():
    """The pristine CPU bf16 DINOv3-L (the T2 loop's own load). Shared and NEVER mutated: callers deep-copy."""
    if "base" not in _C:
        _C["base"] = LM.load_base_model(snapshot())
    return _C["base"]


def banked_frame(idx: int = 0) -> torch.Tensor:
    """Raw frame `idx` of a banked eval clip's v2ep (the cache's own 256x640 cylinder PNG) -> uint8 [3, 256, 640]."""
    key = ("frame", idx)
    if key not in _C:
        import torchvision
        o = torch.load(FRAME_FILE, map_location="cpu", weights_only=False)
        offs = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), o["jpeg_len"]]), 0)
        _C[key] = torchvision.io.decode_png(o["jpeg_buf"][int(offs[idx]):int(offs[idx + 1])]).contiguous()
    return _C[key]


# ------------------------------------------------------------------------------------------------------------------------------ the training side, as a reference
def ref_module():
    """`stack/tanitad/models/v8f_lora_encoder.py` at tip a9da6e8f, verbatim, loaded by file path; its builder files are the verbatim copies beside it."""
    if "ref" not in _C:
        os.environ["V8F_LORA_BUILDER"], os.environ["V8F_LORA_SHIP"] = f"{REF}/pull_encode_pooled.py", f"{REF}/dinov3_fp8_encode_ship.py"
        spec = importlib.util.spec_from_file_location("v8f_lora_encoder_ref", f"{REF}/v8f_lora_encoder.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["v8f_lora_encoder_ref"] = mod
        spec.loader.exec_module(mod)
        _C["ref"] = mod
    return _C["ref"]


def train_encoder(seed: int = 0):
    """A training-side `DinoV3LoRAEncoder` around a copy of the shared base (no second disk load): r 32, alpha 32, q/v, B = 0 (as built), A from the private generator."""
    R = ref_module()
    m, s = R.mean_std(torch.bfloat16, "cpu", from_processor=True)
    return R.DinoV3LoRAEncoder(copy.deepcopy(base_model()), m, s, r=32, alpha=32.0, targets=("q", "v"), seed=seed, grid="4x10", quantize=True)


def lora_sd(seed: int = 0, b_std: float = 0.0, a_seed: int | None = None) -> dict:
    """A `lora_state_dict()` in the TRAINING side's format (names from the real module, A from its private generator, B ~ N(0, b_std^2) from a private generator; b_std = 0 -> B = 0, the identity)."""
    enc = train_encoder(seed if a_seed is None else a_seed)
    sd = enc.lora_state_dict()
    g = torch.Generator().manual_seed(1000 + seed)
    for n, t in sd["tensors"].items():
        if "lora_B" in n:
            t.copy_(torch.randn(t.shape, generator=g) * b_std)
    sd["meta"]["weights_sha256"] = LM.BASE_WEIGHTS_SHA256
    return sd


def write_blob(path: str, sd: dict, head_seed: int = 0, step: int = 1000) -> str:
    """An L-trainer-shaped blob: the REAL trained heads of a banked checkpoint (`model`, `cfg`, ...) + a `lora` entry + `encoder_levers`. The heads' optimiser state is dropped to keep the fixture small."""
    ck = torch.load(HEAD_CKPT[head_seed], map_location="cpu", weights_only=False)
    ck.pop("opt", None)
    ck["lora"] = sd
    ck["encoder_levers"] = {"encoder": "lora", "lora_r": 32, "lora_alpha": 32.0, "lora_targets": ["q", "v"], "lora_blocks": 24, "lora_modules": 48, "lora_trainable": 3145728,
                            "weights_sha256": LM.BASE_WEIGHTS_SHA256, "backbone_dtype": "torch.bfloat16"}
    ck["step"], ck["arm"] = step, "L"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ck, path)
    return path


def make_block(blobs: dict, pins: dict | None = None) -> dict:
    """The `arm5.encoder` block the builder writes, from {arm: blob path}."""
    base = base_model()
    pins = pins or {a: LM.build_pin_for(p, base_model=base) for a, p in blobs.items()}
    arms = list(blobs)
    m0 = pins[arms[0]]["lora_meta"]
    blk = {"kind": LM.KIND, "r": int(m0["r"]), "alpha": float(m0["alpha"]), "targets": list(m0["targets"]), "blocks": int(m0["blocks"]), "model_id": LM.MID, "base_weights_sha256": LM.BASE_WEIGHTS_SHA256,
           "source_ckpt_sha256": {a: pins[a]["source_ckpt_sha256"] for a in arms}, "lora_tensors_sha256": {a: pins[a]["lora_tensors_sha256"] for a in arms},
           "merged_qv_sha256": {a: pins[a]["merged_qv_sha256"] for a in arms}, "merged_qv_tensor_sha256": {a: pins[a]["merged_qv_tensor_sha256"] for a in arms}}
    return blk


def fp8_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def rel_l2(a, b) -> float:
    return E.rel_l2(a.float(), b.float())


def sha_of_file(p) -> str:
    return LM.file_sha256(p)


def frozen_tokens(frame=None) -> torch.Tensor:
    """The unchanged `InLoopEncoder` on the banked frame -> fp8 [40, 1024] (cached)."""
    if "frozen_tok" not in _C:
        enc = E.InLoopEncoder(snapshot(), "bf16", "cpu")
        _C["frozen_tok"] = enc.tokens(banked_frame() if frame is None else frame, already_cyl=True)
    return _C["frozen_tok"]


def lora_tokens(blob_path: str, block: dict, arm: str, frame=None):
    """The LoRA-folded `InLoopEncoder` (under `lora_fold_active`) on the banked frame -> (fp8 [40, 1024], the encoder)."""
    with LM.lora_fold_active(blob_path, block, arm) as cls:
        enc = cls(snapshot(), "bf16", "cpu")
    return enc.tokens(banked_frame() if frame is None else frame, already_cyl=True), enc


def dump(o) -> str:
    return json.dumps(o, indent=1, sort_keys=True, default=str)
