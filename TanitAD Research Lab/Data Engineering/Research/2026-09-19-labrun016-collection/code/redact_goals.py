"""Redact the raw clip id in the tip's GOALS_AND_CLAIMS.md -> sha12, tip-based, CAS write.
The id itself is never printed or placed on a command line."""
import hashlib
import json
import re
import subprocess
import tempfile

D, M = "D:/Projects/TanitAD", "C:/Users/Admin/tanitad-push"
P = "Project Steering/GOALS_AND_CLAIMS.md"
TIP = open("C:/Users/Admin/AppData/Local/Temp/claude/tip.txt").read().strip()
U = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
IDS = set(json.load(open("C:/Users/Admin/AppData/Local/Temp/claude/clip_universe.json"))["ids"])


def g(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True).stdout


def s12(u):
    return hashlib.sha256(u.encode()).hexdigest()[:12]


tip_blob = g(M, "rev-parse", "%s:%s" % (TIP, P)).decode().strip()
idx_blob = g(D, "ls-files", "--stage", "--", P).decode().split()[1]
wt_blob = g(D, "hash-object", "--", P).decode().strip()
assert len(tip_blob) == len(idx_blob) == len(wt_blob) == 40
assert tip_blob == idx_blob == wt_blob, "D:'s copy is not the tip's -- refusing (tip-based only)"

raw = open(D + "/" + P, "rb").read()                  # CRLF worktree bytes == tip content
txt = raw.decode("utf-8")
found = sorted(set(U.findall(txt)))
clips = [u for u in found if u in IDS]
other = [u for u in found if u not in IDS]
print("uuid-shaped distinct: %d | clip ids: %s | not clip ids: %s"
      % (len(found), [s12(u) for u in clips], [s12(u) for u in other]))

# where else on the tip does each clip id live? (pattern via a temp file, never argv)
for u in clips:
    with tempfile.NamedTemporaryFile("w", suffix=".pat", delete=False) as f:
        f.write(u + "\n")
        pat = f.name
    hits = g(M, "grep", "-c", "-F", "-f", pat, TIP).decode("utf-8", "replace").strip().split("\n")
    print("  sha12 %s appears on the tip in %d file(s):" % (s12(u), len([h for h in hits if h])))
    for h in hits:
        if h:
            path, n = h.rsplit(":", 1)
            print("     %3s  %s" % (n, U.sub("<id>", path.split(":", 1)[1])))

new = txt
for u in clips:
    new = new.replace(u, "sha12:" + s12(u))
changed = [(a, b) for a, b in zip(txt.split("\n"), new.split("\n")) if a != b]
assert len(txt.split("\n")) == len(new.split("\n"))
assert not any(u in new for u in clips), "a clip id survived"
assert all(u in new for u in other), "a non-clip uuid was altered"
print("lines changed: %d (expected = number of lines carrying a clip id)" % len(changed))
for a, b in changed:
    i = b.find("sha12:")
    print("   now: ...%s..." % b[max(0, i - 60):i + 40])

if g(D, "hash-object", "--", P).decode().strip() != wt_blob:   # compare-and-swap
    raise SystemExit("REFUSED: the file changed on disk")
open(D + "/" + P, "wb").write(new.encode("utf-8"))
print("written; new blob", g(D, "hash-object", "--", P).decode().strip()[:10])
