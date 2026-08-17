import torch, sys, os, glob, warnings
warnings.filterwarnings("ignore")
root = r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
cands = []
for dp, dn, fn in os.walk(root):
    if ".git" in dp or "worktrees" in dp or "__pycache__" in dp:
        continue
    for f in fn:
        if f.endswith(".pt"):
            cands.append(os.path.join(dp, f))
print(f"TOTAL .pt candidates: {len(cands)}")
for p in sorted(cands):
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[SKIP] {os.path.relpath(p,root)}: {type(e).__name__} {str(e)[:60]}")
        continue
    if not isinstance(d, dict):
        print(f"[NONDICT] {os.path.relpath(p,root)}: {type(d).__name__}")
        continue
    # summarize
    items = []
    n881 = False
    for k, v in d.items():
        if hasattr(v, "shape"):
            items.append(f"{k}{tuple(v.shape)}")
            if v.shape and v.shape[0] == 881: n881 = True
        elif hasattr(v, "__len__"):
            items.append(f"{k}[len={len(v)}]")
            if len(v) == 881: n881 = True
        else:
            items.append(f"{k}={v}")
    flag = " <<< n=881" if n881 else ""
    print(f"{os.path.relpath(p,root)}{flag}\n    {', '.join(items[:16])}")
