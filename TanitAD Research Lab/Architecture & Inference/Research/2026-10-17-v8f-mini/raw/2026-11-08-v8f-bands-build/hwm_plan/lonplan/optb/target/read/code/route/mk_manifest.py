import hashlib, os, subprocess, sys
T = "F:/TanitAD-scratch/optb3_read/tree"
out = []
for sub in ("stack", "taniteval"):
    for dp, dn, fn in os.walk(os.path.join(T, sub)):
        dn[:] = [d for d in dn if d not in ("__pycache__", ".pytest_cache")]
        for f in fn:
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                out.append((hashlib.sha256(open(p, "rb").read()).hexdigest(), os.path.relpath(p, T).replace("\\", "/")))
out.sort(key=lambda x: x[1])
open("F:/TanitAD-scratch/optb3_read/route/tree_manifest.sha256", "w", encoding="utf-8", newline="\n").write("".join(f"{s}  {r}\n" for s, r in out))
# compare with git blobs of 0f471d57
bad = 0
for s, r in out:
    b = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "cat-file", "blob", f"0f471d57:{r}"], capture_output=True).stdout
    if hashlib.sha256(b).hexdigest() != s:
        bad += 1; print("DIFF", r)
print(len(out), "files; differ from 0f471d57 blobs:", bad)
