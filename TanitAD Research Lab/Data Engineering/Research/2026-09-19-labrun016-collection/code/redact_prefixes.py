"""Second pass on the register: 8-hex clip-id PREFIXES -> sha12, and mask them in the
banked exposure summary. No id or prefix is ever printed.

A token is rewritten only if ALL hold:
  * it is hex-bounded, 8 chars, and the prefix of EXACTLY ONE of the 306,152 clip ids;
  * it does NOT resolve as a git object in the push mirror (a commit hash is not a clip);
  * it is NOT inside a path (preceded by '/' '_' '.' or followed by '.' '/' '_'),
    because a path names a file that exists on the tip.
Proof obligation: mapping every `sha12:...` introduced here back to its original token
must reproduce the tip file BYTE FOR BYTE (the redaction-only test, applied to ourselves).
"""
import hashlib
import json
import re
import subprocess

D, M = "D:/Projects/TanitAD", "C:/Users/Admin/tanitad-push"
TIP = open("C:/Users/Admin/AppData/Local/Temp/claude/tip.txt").read().strip()
P = "Project Steering/GOALS_AND_CLAIMS.md"
PK = D + "/TanitAD Research Lab/Data Engineering/Research/2026-09-19-labrun016-collection"
ids = json.load(open("C:/Users/Admin/AppData/Local/Temp/claude/clip_universe.json"))["ids"]
by_pre = {}
for i in ids:
    by_pre.setdefault(i[:8], []).append(i)
H8 = re.compile(r"(?<![0-9a-f])([0-9a-f]{8})(?![0-9a-f])")
U = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def s12(u):
    return hashlib.sha256(u.encode()).hexdigest()[:12]


def is_git_object(tok):
    return subprocess.run(["git", "-C", M, "cat-file", "-e", tok + "^{object}"],
                          capture_output=True).returncode == 0


# ── the exposure summary: mask prefixes embedded in tip file names ─────────────
sp = PK + "/raw/clip_exposure_summary.json"
s = open(sp, encoding="utf-8").read()
s2 = H8.sub(lambda m: "<id8>" if m.group(1) in by_pre else m.group(1), s)
open(sp, "w", encoding="utf-8").write(s2)
print("exposure summary: %d prefix tokens masked"
      % sum(1 for m in H8.finditer(s) if m.group(1) in by_pre))

# ── the register ──────────────────────────────────────────────────────────────
raw = open(D + "/" + P, "rb").read()
txt = raw.decode("utf-8")
wt_blob = subprocess.run(["git", "-C", D, "hash-object", "--", P],
                         capture_output=True).stdout.decode().strip()
out, last, done, skipped = [], 0, [], {"git_object": 0, "in_path": 0, "ambiguous": 0}
for m in H8.finditer(txt):
    tok = m.group(1)
    if tok not in by_pre:
        continue
    before, after = txt[m.start() - 1:m.start()], txt[m.end():m.end() + 1]
    if len(by_pre[tok]) != 1:
        skipped["ambiguous"] += 1
        continue
    if before in "/_." or after in "./_":
        skipped["in_path"] += 1
        continue
    if is_git_object(tok):
        skipped["git_object"] += 1
        continue
    end = m.end()
    for tail in ("-…", "-..."):                       # a truncated id written as xxxxxxxx-…
        if txt.startswith(tail, end):
            end += len(tail)
            break
    rep = "sha12:" + s12(by_pre[tok][0])
    out.append(txt[last:m.start()])
    pos = sum(len(x) for x in out)            # exact span of this substitution in `new`
    out.append(rep)
    done.append((pos, pos + len(rep), txt[m.start():end]))
    last = end
out.append(txt[last:])
new = "".join(out)

# proof, POSITIONAL: invert exactly the spans we wrote. (A string-replace undo is
# ambiguous when a prefix maps to the same sha12 as a pass-1 full-id redaction --
# MEASURED: 1 such token here, and the replace-based proof failed on it.)
undo = new
for a, b, orig in reversed(done):
    undo = undo[:a] + orig + undo[b:]
assert undo == txt, "NOT redaction-only: inverting the spans does not reproduce the file"
# and the file we started from must itself be the tip plus pass 1 (full ids -> sha12)
tip = subprocess.run(["git", "-C", M, "show", "%s:%s" % (TIP, P)],
                     capture_output=True).stdout.decode("utf-8")
idset = {i for v in by_pre.values() for i in v}
tip_pass1 = U.sub(lambda m: "sha12:" + s12(m.group(0).lower())
                  if m.group(0).lower() in idset else m.group(0), tip)
assert txt.replace("\r\n", "\n") == tip_pass1.replace("\r\n", "\n"), "start file is not tip+pass1"
print("register: %d prefix references rewritten, skipped %s; inverse reproduces the tip: PASS"
      % (len(done), skipped))
if subprocess.run(["git", "-C", D, "hash-object", "--", P],
                  capture_output=True).stdout.decode().strip() != wt_blob:
    raise SystemExit("REFUSED: the register changed on disk")
open(D + "/" + P, "wb").write(new.encode("utf-8"))
left = sum(1 for m in H8.finditer(new) if m.group(1) in by_pre)
print("register written; clip-prefix tokens left: %d (all skipped by rule)" % left)
