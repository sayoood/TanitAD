"""wsp_common: shared loaders for the WSP-GAIN edit (PREREG_WSP_GAIN.md, sha256 92d61dcb...). Dev-box CPU, zero training.

Everything here is import-safe: no model forward, no data read at import time.
Run with:  CUDA_VISIBLE_DEVICES=-1  C:/Users/Admin/venvs/tanitad/Scripts/python.exe  (PYTHONPATH joined with ';')
"""
import os
import sys
import json
import hashlib
import importlib.util

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ART = "E:/Projects/TanitAD-artifacts"
TREE = ART + "/dagt1_tree_b43c7a9a"
for _p in (TREE + "/stack", TREE + "/taniteval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
HERE = ART + "/dagland/motion_c/theory/wsp_edit"
THEORY = ART + "/dagland/motion_c/theory"
RI_TOOL = ART + "/dagland/motion_c/ri/code/v8f_t1_eval_ri.py"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
DATA = TREE + "/" + PK
CARD = THEORY + "/PREREG_WSP_GAIN.md"
CARD_SHA = "92d61dcb6bec10f847ae23eecf2b4813076358abdfb9388c16c6016121a8e602"
CKPT_C0 = {0: ART + "/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt", 1: ART + "/v8f-recover/M2b_s1_recft_C0/ckpt_last.pt"}
EDIT_ROOT = ART + "/v8f-wspedit"
TRAIN_CACHE = "F:/TanitAD-scratch/speedread/cache"
TRAIN_MANIFEST = "F:/TanitAD-scratch/c_moved_20261103/box2_ship/logs/v8f_dscale/manifest_train.jsonl"
TRAIN_JOIN = "F:/TanitAD-scratch/aside_from_E/probe_scratch/trigger_probe_scratch/join_train_rebuilt.npz"
TRACKS = ART + "/sfresh_inputs/inputs/native_tracks.pt"
WINDOW_LIST_SHA = "e144adcd9b6273f2b44ad7dfa162931b90e159f7b3911e0893b25777b2ae8168"
BAND = (3.0, 8.0)
DT_SEC = 0.8
J_HI, J_LO = 11, 7


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def assert_card_frozen():
    got = sha256_file(CARD)
    # the card gains dated log entries below section 6 only; the frozen sha covers the text above the log marker
    txt = open(CARD, "rb").read()
    marker = b"## 6. Log (append-only, dated)"
    k = txt.find(marker)
    e = txt.find(b"\n", k)
    head = txt[:e + 1]
    head_sha = sha256_bytes(head)
    return got, head_sha


def load_tool():
    """The RI T1-kin tool module (same loader as p2b_check.py); asserts tanitad comes from the dagt1 tree."""
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(TREE), tanitad.__file__
    spec = importlib.util.spec_from_file_location("ri_tool", RI_TOOL)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    return T


def load_model(M, ckpt_path=None, ck=None):
    """Build V8FMini from a checkpoint dict (the ckpt cfg carries launch_intent_w 0.0, dropped after asserting it is 0.0, as in p2b_check.py)."""
    import torch
    if ck is None:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfgd = dict(ck["cfg"])
    assert cfgd.pop("launch_intent_w", 0.0) == 0.0
    cfg = M.V8FMiniConfig(**cfgd)
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, ck
