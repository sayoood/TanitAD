"""Card PREREG_V8F_LEVER3_V2, control b2 for native_tracks_union.pt (content form).

PASS iff the union equals the expected object built from the two sha-verified inputs: tracks = the corpus tracks minus
the 60 holdout clips (corpus order), then the 10k tracks (10k order), every tensor equal in dtype, shape and bytes;
meta = the corpus meta plus the merge dict the tool writes; key order equal; file size equal to the pod record.
Planted arms that MUST fail: one flipped byte in one track, one track dropped, two tracks swapped in order, one meta
value changed. Positive control: the 10k input against its own re-save (different sha256, same content) must PASS.
"""
import hashlib, importlib.util, io, json, sys
import torch

B = "F:/TanitAD-scratch/rollhist/lever3/build"
U = "C:/Users/Admin/lever3/merged/native_tracks_union.pt"
C = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt"
T = "F:/TanitAD-scratch/rollhist/lever3/labels/in/tenk/labels/native_tracks_datascale10k.pt"
DEV_MAN = "F:/TanitAD-scratch/speedread/cache/train/manifest.jsonl"
POD_SIZE = 51494151
HOLD_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"

spec = importlib.util.spec_from_file_location("mlean", B + "/dscale10k_merge_lean.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
ids = M.holdout_ids(DEV_MAN)
assert M.holdout_digest(ids) == HOLD_SHA, "holdout digest"


def tb(t):
    t = t.contiguous().reshape(-1)
    return t.view(torch.uint8) if t.numel() else t


def eq_obj(a, b):
    """Strict equality: dict key ORDER, tensors by dtype/shape/bytes, everything else by ==."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and \
            tuple(a.shape) == tuple(b.shape) and bool(torch.equal(tb(a), tb(b)))
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(eq_obj(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(eq_obj(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


oc = torch.load(C, map_location="cpu", weights_only=True)
ot = torch.load(T, map_location="cpu", weights_only=True)
drop = set(ids)
keep = {c: v for c, v in oc["tracks"].items() if c not in drop}
meta = dict(oc.get("meta", {}))
meta["merge"] = {"tool": "dscale10k_merge.py", "n_corpus_kept": len(keep), "n_tenk": len(ot["tracks"]),
                 "n_holdout_dropped": len(oc["tracks"]) - len(keep)}
expected = {"tracks": {**keep, **ot["tracks"]}, "meta": meta}
u = torch.load(U, map_location="cpu", weights_only=True)
raw = open(U, "rb").read()
res = {"union_sha256": hashlib.sha256(raw).hexdigest(), "union_bytes": len(raw), "pod_bytes": POD_SIZE,
       "size_equal": len(raw) == POD_SIZE, "content_equal": eq_obj(u, expected), "n_tracks": len(u["tracks"])}

# planted arms (each must FAIL the check)
import copy
k0, k1 = list(u["tracks"])[0], list(u["tracks"])[-1]
m1 = {"tracks": dict(u["tracks"]), "meta": u["meta"]}
t = m1["tracks"][k0].clone(); bb = tb(t).clone(); bb[0] ^= 1; m1["tracks"][k0] = bb.view(t.dtype).reshape(t.shape)
m2 = {"tracks": {k: v for k, v in u["tracks"].items() if k != k1}, "meta": u["meta"]}
items = list(u["tracks"].items()); items[0], items[1] = items[1], items[0]
m3 = {"tracks": dict(items), "meta": u["meta"]}
m4 = {"tracks": u["tracks"], "meta": copy.deepcopy(u["meta"])}; m4["meta"]["merge"]["n_tenk"] += 1
res["planted_byte_fails"] = not eq_obj(m1, expected)
res["planted_drop_fails"] = not eq_obj(m2, expected)
res["planted_swap_fails"] = not eq_obj(m3, expected)
res["planted_meta_fails"] = not eq_obj(m4, expected)
# positive control: the 10k input vs its own re-save (pod-made file; the re-save differs in sha256)
buf = io.BytesIO(); torch.save(ot, buf)
ot2 = torch.load(io.BytesIO(buf.getvalue()), map_location="cpu", weights_only=True)
res["positive_control_sha_differs"] = hashlib.sha256(buf.getvalue()).hexdigest() != hashlib.sha256(open(T, "rb").read()).hexdigest()
res["positive_control_content_equal"] = eq_obj(ot, ot2)
res["PASS"] = all(res[k] for k in ("size_equal", "content_equal", "planted_byte_fails", "planted_drop_fails",
                                    "planted_swap_fails", "planted_meta_fails", "positive_control_sha_differs",
                                    "positive_control_content_equal"))
print(json.dumps(res, indent=1))
json.dump(res, open(B + "/B2_TRACKS_CONTENT.json", "w"), indent=1)
sys.exit(0 if res["PASS"] else 1)
