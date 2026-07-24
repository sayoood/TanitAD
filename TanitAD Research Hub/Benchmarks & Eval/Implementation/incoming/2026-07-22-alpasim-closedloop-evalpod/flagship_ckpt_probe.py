import sys, torch
sys.path.insert(0, "/root/TanitAD/stack")
ck = torch.load("/root/models/flagship-speed/ckpt.pt", map_location="cpu", weights_only=False)
print("top-level keys:", list(ck.keys()) if isinstance(ck, dict) else type(ck))
if isinstance(ck, dict):
    print("step:", ck.get("step"))
    if "cfg" in ck:
        c = ck["cfg"]
        for k in ("tactical_policy", "tactical_pred", "v2_anchor_tactical", "v2_ego_to_planners"):
            print("cfg", k, "=", (c.get(k) if isinstance(c, dict) else getattr(c, k, "?")))
    sd = ck.get("model", ck)
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    print("n model keys:", len(keys))
    for pat in ("tactical_policy", "tactical_decoder", "tactical.", "tactical_pred",
                "anchor", "traj", "waypoint", "conf_head", "offset_head", "maneuver"):
        hits = [k for k in keys if pat in k]
        if hits:
            print(f"  [{pat}] {len(hits)} keys, e.g. {hits[:3]}")
    print("=== enc_readout.pt ===")
    er = torch.load("/root/models/flagship-speed/enc_readout.pt", map_location="cpu", weights_only=False)
    print("enc_readout keys:", list(er.keys()) if isinstance(er, dict) else type(er))
print("PROBE_DONE")
