import sys, torch, json, hashlib
out = {}
for p in sys.argv[1:]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    if not isinstance(sd, dict) or not any(isinstance(v, torch.Tensor) for v in sd.values()):
        for k in ("model_state", "state_dict", "ema"):
            if k in ck: sd = ck[k]; break
    keys = [k for k in sd if ("fuse16" in k or "fuse32" in k)]
    rec = {"step": ck.get("step") if isinstance(ck, dict) else None, "keys": {}}
    for k in keys:
        w = sd[k].float()
        r = {"shape": list(w.shape)}
        if w.ndim == 4 and w.shape[2] == 1 and w.shape[3] == 1:
            co, ci = w.shape[0], w.shape[1]
            K = ci // co
            W = w[:, :, 0, 0]
            blocks = [W[:, j*co:(j+1)*co] for j in range(K)]
            I = torch.eye(co)
            r["K"] = K
            r["fro_per_frame_oldest_to_newest"] = [round(float(b.norm()), 4) for b in blocks]
            r["newest_minus_identity_fro"] = round(float((blocks[-1] - I).norm()), 4)
            r["identity_fro"] = round(float(I.norm()), 4)
            # control: identity-init state reads [0,...,0, sqrt(co)] and 0 deviation
        rec["keys"][k] = r
    out[p] = rec
print(json.dumps(out, indent=1))
