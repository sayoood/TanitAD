import json, os, shutil, subprocess
N = "F:/TanitAD-scratch/strat1/navrb2"
info = json.load(open(N + "/tmp/changed_files.json"))
pa = N + "/tmp/pa"
shutil.rmtree(pa, ignore_errors=True)
for rel in info["changed"] + info["new"]:
    os.makedirs(os.path.dirname(f"{pa}/b/{rel}"), exist_ok=True)
    shutil.copyfile(f"{N}/tree/{rel}", f"{pa}/b/{rel}")
    os.makedirs(os.path.dirname(f"{pa}/a/{rel}"), exist_ok=True)
    if rel in info["changed"]:
        data = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "show", f"4dedac45:{rel}"], capture_output=True).stdout
        open(f"{pa}/a/{rel}", "wb").write(data)
out = subprocess.run(["git", "-c", "core.autocrlf=false", "diff", "--no-index", "--no-color", "a", "b"], cwd=pa, capture_output=True).stdout
open(N + "/navrb2.patch", "wb").write(out.replace(b"\r\n", b"\n"))
print(len(out), out.count(b"diff --git"))
