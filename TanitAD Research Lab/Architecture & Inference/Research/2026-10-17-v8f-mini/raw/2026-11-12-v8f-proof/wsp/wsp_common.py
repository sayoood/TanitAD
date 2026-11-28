"""wsp_common.py -- shared pieces of the v8F-PROOF package (H-V8F-PROOF-1 SOUP, -2 TASKVEC).  CPU only; imports tanitad from OUR tree and asserts it."""
import hashlib
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
WSP = "F:/TanitAD-scratch/v8proof/wsp"
TREE = WSP + "/tree"
PULL = "C:/Users/Admin/t2v8f_pulls"
CKPT = WSP + "/ckpt"
COMMIT = "3e264bbf3fa2f6f17bea448de09aa972faaf7725"
BANKS = {"NULL_s0": PULL + "/stopbal_BAL-NULL_s0", "NULL_rep": PULL + "/stopbal_BAL-NULL_rep", "G_s0": PULL + "/stopbal_BAL-G_s0", "G2_s0": PULL + "/stopbal_BAL-G2_s0"}
BUILT = ("SOUP2", "SOUPa0", "SOUPa1", "DOSE050", "DOSE100")
# group of a state-dict key (fixed BEFORE any number was computed).  Trainer naming (v8f_mini_train.py:1175): adapter, S=read_s, T=read_t, O=read_o, head_s, head_o, wm.
# Our additions: "cond" = the conditioning embeddings e_*, "head_T" = T's trunk and decision heads (trunk_t, h_goal, h_lat, h_lon, h_cons), "frozen" = prior / buffers.
def group_of(k):
    p = k.split(".")[0]
    if p in ("adapter", "pos", "time"):
        return "adapter"
    if p.startswith("e_"):
        return "cond"
    return {"read_s": "S", "read_t": "T", "read_o": "O", "head_s": "head_S", "head_o": "head_O", "trunk_t": "head_T", "h_goal": "head_T", "h_lat": "head_T",
            "h_lon": "head_T", "h_cons": "head_T", "wm": "WM"}.get(p, "other:" + p)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def use_tree():
    """put OUR tree first on sys.path and assert tanitad comes from it."""
    for p in (TREE + "/stack", TREE + "/taniteval"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(TREE), tanitad.__file__
    return tanitad


def load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model(ck):
    use_tree()
    import torch
    import tanitad.models.v8f_mini as M
    cfg = M.V8FMiniConfig(**ck["cfg"])
    m = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    m.load_state_dict(ck["model"], strict=True)
    return m


def trainable_names(model):
    """state-dict keys of the parameters with requires_grad (the optimizer's set, v8f_mini_train.py:1130)."""
    return [n for n, p in model.named_parameters() if p.requires_grad]


# ---- dump tags: name -> checkpoint (banked originals for the four trained arms, ours for the five built ones)
T1 = WSP + "/t1"
STOPDEC_OUT = WSP + "/stopdec"
DUMP_TAGS = {"NULL_s0": BANKS["NULL_s0"] + "/ckpt_last.pt", "NULL_rep": BANKS["NULL_rep"] + "/ckpt_last.pt",
             "SOUP2": CKPT + "/SOUP2/ckpt_last.pt", "SOUPa0": CKPT + "/SOUPa0/ckpt_last.pt", "SOUPa1": CKPT + "/SOUPa1/ckpt_last.pt",
             "G_s0": BANKS["G_s0"] + "/ckpt_last.pt", "G2_s0": BANKS["G2_s0"] + "/ckpt_last.pt", "DOSE050": CKPT + "/DOSE050/ckpt_last.pt", "DOSE100": CKPT + "/DOSE100/ckpt_last.pt"}
H1_TAGS = ("NULL_s0", "NULL_rep", "SOUP2", "SOUPa0", "SOUPa1")                 # T1 dumps read for H1
H2_TAGS = ("NULL_s0", "NULL_rep", "G_s0", "G2_s0", "DOSE050", "DOSE100")       # STOPDEC reads for H2 (each needs its T1 dump: stopdec_read loads the dump dir)
THREADS = 2                                                                    # ONE thread count for every dump (the OTBAND trap: dumps at other thread counts drift ~mm)
