import hashlib, os, sys
root = os.path.dirname(os.path.abspath(__file__)); rows = []
for d, _, fs in os.walk(root):
    for f in sorted(fs):
        if f == "MANIFEST.txt": continue
        p = os.path.join(d, f); h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        rows.append(f"{os.path.relpath(p, root).replace(os.sep, '/')} | {os.path.getsize(p)} | {h}")
open(os.path.join(root, "MANIFEST.txt"), "w", newline="\n").write("\n".join(sorted(rows)) + "\n"); print(len(rows), "files")
