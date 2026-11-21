"""Find every CLI-style invocation of refav1_arm.py at the tip and whether the same command block names --action-units.
Reads blobs from the mirror via `git cat-file --batch`; no worktree reads. ASCII output."""
import json, re, subprocess, sys
TIP = "5fb509a"
REPO = "C:/Users/Admin/tanitad-push"
def git(*a, inp=None):
    return subprocess.run(["git", "-C", REPO, "-c", "core.autocrlf=false", *a], input=inp, capture_output=True).stdout
files = git("grep", "-l", "-I", r"refav1_arm", TIP, "--", ".").decode("utf-8", "replace").split("\n")
files = [f.split(":", 1)[1] for f in files if f]
SKIP_EXT = (".json", ".jsonl", ".log", ".html", ".csv", ".npz", ".txt")
rows = []
inv = re.compile(r"refav1_arm(\.py)?")
imp = re.compile(r"^\s*(from\s+\S*refav1_arm\S*\s+import|import\s+\S*refav1_arm|.*importlib|.*spec_from_file_location|.*sys\.modules|.*#)")
for f in files:
    if f.lower().endswith(SKIP_EXT):
        continue
    raw = git("show", f"{TIP}:{f}")
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw.decode("latin-1")
    lines = txt.split("\n")
    for i, ln in enumerate(lines):
        if not inv.search(ln):
            continue
        # CLI-ish context: python/py launcher on the line, or the .py name inside a quoted argv element / path ending in refav1_arm.py
        cli = bool(re.search(r"(python[0-9.]*(\.exe)?|\bpy\b|\bpython3\b).*refav1_arm\.py", ln)) or \
              bool(re.search(r"[\"'][^\"']*refav1_arm\.py[\"']", ln)) or \
              bool(re.search(r"^\s*[\w./\:-]*refav1_arm\.py\b", ln)) or \
              bool(re.search(r"refav1_arm\.py\s+--", ln))
        if not cli:
            continue
        # window: lines i-3 .. i+40 until a blank-line gap after i (command block)
        lo = max(0, i - 6); hi = i + 1
        while hi < len(lines) and hi < i + 60:
            hi += 1
            if lines[hi - 1].strip() == "" and hi - 1 > i + 1 and not lines[hi - 2].rstrip().endswith(("\\", "`", "^", ",")):
                break
        block = "\n".join(lines[lo:hi])
        has = "action-units" in block or "action_units" in block
        rows.append({"file": f, "line": i + 1, "has_units_in_block": has, "text": ln.strip()[:160]})
json.dump(rows, open("C:/Users/Admin/hjunits2/scan/cli_scan.json", "w"), indent=1)
nf = len({r['file'] for r in rows})
print("candidate invocation lines:", len(rows), "in", nf, "files;", sum(1 for r in rows if not r['has_units_in_block']), "without --action-units in block")
