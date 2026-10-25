import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
print("keys", sorted(a.keys())[:12])
ca, cb = a.get("cfg") or a.get("config"), b.get("cfg") or b.get("config")
ca = ca if isinstance(ca, dict) else vars(ca); cb = cb if isinstance(cb, dict) else vars(cb)
for k in sorted(set(ca) | set(cb)):
    if ca.get(k, "<MISSING>") != cb.get(k, "<MISSING>"): print("DIFF", k, repr(ca.get(k, "<MISSING>")), repr(cb.get(k, "<MISSING>")))
