"""Weights identity: (1) M2b HACC-baked wm.* == the source ckpt's wm.* (registry 17.1: bit for bit); (2) C0 wm.* vs M2b wm.*. Writes weights_check.json."""
import json, hashlib, torch
import wmact_lib as W

def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def cmp(a, b):
    ka, kb = sorted(a), sorted(b)
    same_keys = ka == kb
    eq = [k for k in ka if k in b and a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and torch.equal(a[k], b[k])]
    return {"n_a": len(ka), "n_b": len(kb), "same_keys": same_keys, "n_bit_equal": len(eq),
            "differing": [k for k in ka if k not in eq][:10]}

def wm_digest(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().contiguous().numpy().tobytes())
    return h.hexdigest()

out = {}
for s in (0, 1):
    hacc = load(W.CKPT[f"M2b_s{s}"])
    src_p = f"{W.SF}/inputs/M2b_s{s}/ckpt_last.pt"
    src = load(src_p)
    r = {"hacc_file": W.CKPT[f"M2b_s{s}"], "hacc_file_sha256": sha_file(W.CKPT[f"M2b_s{s}"]),
         "source_file": src_p, "source_file_sha256": sha_file(src_p),
         "hacc_declared_source_sha256": hacc["hacc"]["source_ckpt_sha256"],
         "wm_digest_hacc": wm_digest(W.wm_state(hacc)), "wm_digest_source": wm_digest(W.wm_state(src)),
         "wm_hacc_vs_source": cmp(W.wm_state(hacc), W.wm_state(src)),
         "adapter_pos_hacc_vs_source": cmp(W.adapter_state(hacc), W.adapter_state(src)),
         "all_hacc_vs_source": cmp(hacc["model"], src["model"])}
    c0p = W.CKPT_C0[f"C0_s{s}"]
    c0 = load(c0p)
    r["c0_file"] = c0p; r["c0_file_sha256"] = sha_file(c0p)
    r["wm_digest_c0"] = wm_digest(W.wm_state(c0))
    r["wm_c0_vs_m2b_hacc"] = cmp(W.wm_state(c0), W.wm_state(hacc))
    r["adapter_pos_c0_vs_m2b_hacc"] = cmp(W.adapter_state(c0), W.adapter_state(hacc))
    out[f"s{s}"] = r
    print(s, json.dumps({k: v for k, v in r.items() if k.startswith(("wm_", "adapter", "all_"))}, indent=0)[:1500])
json.dump(out, open("weights_check.json", "w"), indent=1)
