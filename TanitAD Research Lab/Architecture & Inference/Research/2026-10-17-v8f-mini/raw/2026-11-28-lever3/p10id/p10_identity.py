"""Which p10_prior.pt did the pod's UE_s2/A (and so RH_s2) use? Compare the frozen P10 tensors in
the checkpoint with each candidate file, tensor by tensor, bitwise. Read-only."""
import hashlib, json, os, sys
import torch

CKPT = sys.argv[1]
CANDS = sys.argv[2:]
OUT = os.environ.get("P10_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "P10_IDENTITY.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def flat(obj, pre=""):
    out = {}
    if isinstance(obj, torch.Tensor):
        out[pre] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat(v, f"{pre}.{k}" if pre else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flat(v, f"{pre}[{i}]"))
    return out


ck = torch.load(CKPT, map_location="cpu", weights_only=False)
top = sorted(ck.keys()) if isinstance(ck, dict) else type(ck).__name__
sd = None
for k in ("model", "state_dict", "model_state", "model_state_dict"):
    if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
        sd = ck[k]
        break
if sd is None:
    sd = ck
ckf = flat(sd)
p10keys = sorted(k for k in ckf if "p10" in k.lower() or "prior" in k.lower())
res = {"ckpt": CKPT, "ckpt_sha256": sha(CKPT), "ckpt_top_keys": top, "n_ckpt_tensors": len(ckf),
       "ckpt_p10_keys": p10keys[:200], "candidates": []}
for c in CANDS:
    cf = flat(torch.load(c, map_location="cpu", weights_only=False))
    rec = {"path": c, "sha256": sha(c), "n_tensors": len(cf), "keys": sorted(cf)[:50], "matches": []}
    # match each candidate tensor to a checkpoint p10 key by suffix, then bitwise
    for ck_ in sorted(cf):
        t = cf[ck_]
        hits = [k for k in p10keys if k.endswith(ck_) or k.endswith("." + ck_.split(".")[-1])]
        best = None
        for k in hits:
            u = ckf[k]
            if u.shape == t.shape and u.dtype == t.dtype:
                eq = bool(torch.equal(u, t))
                best = {"cand_key": ck_, "ckpt_key": k, "equal": eq,
                        "max_abs": float((u.double() - t.double()).abs().max()) if u.numel() else 0.0}
                if eq:
                    break
        rec["matches"].append(best or {"cand_key": ck_, "ckpt_key": None, "equal": False})
    rec["all_equal"] = bool(rec["matches"]) and all(m["equal"] for m in rec["matches"])
    res["candidates"].append(rec)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print(json.dumps({"ckpt_p10_keys": p10keys[:20], "n_p10": len(p10keys),
                  "cands": [(r["path"], r["sha256"][:16], r["n_tensors"], r["all_equal"]) for r in res["candidates"]]}, indent=1))
