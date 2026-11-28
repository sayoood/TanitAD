import sys, torch
for p in sys.argv[1:]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    ks = [k for k in sd if "phi_tac" in k and k.endswith("weight") and sd[k].ndim == 3]
    print(p.split("/")[-1], "n_phi_tac_conv", len(ks))
    for k in sorted(ks):
        w = sd[k].float()
        taps = [round(float(w[:, :, j].norm()), 3) for j in range(w.shape[2])]
        print(" ", k, list(w.shape), "tap fro [t-2d, t-d, t] =", taps)
    # consumers of z_tac (sanity): list top-level modules whose names contain tac
    print("  tac-ish keys:", sorted({k.split('.')[0] for k in sd if 'tac' in k.split('.')[0]})[:20])
