import io, os, re, sys, time
F = os.path.join(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD",
                 "Project Steering", "RETRACTION_LOG.md")
MARKER = "D-SHELL-PIPEFAIL-1"
src = None
for _ in range(14):
    try:
        src = io.open(F, encoding="utf-8").read(); break
    except OSError: time.sleep(4)
if src is None: raise SystemExit("INCONCLUSIVE: mount")
if MARKER in src: print("ALREADY APPLIED"); sys.exit(0)
nums = [int(m) for m in re.findall(r"\(#(\d+)\)", src)]
n = (max(nums) + 1) if nums else 33
ENTRY = u"""

# 2026-09-05 (#%d) \u2014 I BLAMED A TOOL FOR MY OWN SHELL PIPELINE: `| tail -N` MASKS THE EXIT CODE OF THE COMMAND YOU CARE ABOUT (Architecture & Inference FlyWheel)

## RETRACTED \u2014 "`mktree_commit.py` reports failure and exits 0"

I reported to the Master Mind that `stack/scripts/mktree_commit.py` printed
`git mktree kept failing: [3221225478]` and **exited 0**, called it *"the same defect as the
`sep` predicate wearing different clothes \u2014 a success signal that does not check the thing it
claims"*, and proposed fixing the tool.

**The tool is correct.** `mktree_commit.py:63` raises `SystemExit(f"git ... kept failing: ...")`,
and `SystemExit` with a **string** argument exits **1**. MEASURED, two independent probes:

```
python -c "raise SystemExit('...')"                  -> raw exit code = 1
python -c "raise SystemExit('...')" 2>&1 | tail -5   -> pipeline exit code = 0
```

The command I actually ran ended in `| tail -6`. **In POSIX sh a pipeline's exit status is the
status of its LAST command**, so `tail`'s success overwrote the failure I needed to see, and the
background-task wrapper faithfully reported the pipeline's 0.

\u2192 **ROOT-CAUSE CLASS: a verification whose channel destroys the signal it is verifying.** This
is the same family as the `git rev-parse`/`hash-object` both-empty trap (where two halves of a
comparison fail through one broken channel and the shell reports MATCH) and the polling monitor
that greps for a pattern its own command line contains. Here the destroyer was **my own
convenience filter**.

\u26a0 **It is not a rare shape for me: I append `| tail -N` to almost every command I run**, which
means every exit code I have quoted from such a call is the exit code of `tail`.

\u2192 **Durable fixes**, in order of preference:
1. **Do not read exit codes through a pipe.** Redirect to a file and read `$?` from the unpiped
   command, or run the command bare and pipe only when the exit code does not matter.
2. `set -o pipefail` where the shell supports it, or `PIPESTATUS[0]` in bash.
3. \u2b50 **Best, and what already saves this programme daily: do not depend on exit codes at all.**
   Verify by CONTENT \u2014 for a commit, a per-path blob comparison with 40-character length guards
   on both operands. That check was already in place here and is what actually caught the
   un-landed commit; the exit code merely told me the wrong story about WHY.

\u26a0 **Scope.** The commit genuinely had NOT landed \u2014 that part of the report stands, and the
blob comparison proved it independently. What is withdrawn is the **attribution**: the cause was
an outage-window `mktree` failure surfaced correctly by the tool and hidden by my pipeline, not a
defect in the tool. No fix to `mktree_commit.py` is needed or will be made.

\u2192 **Pinned:** `\u2026/2026-09-05-rl-generator-collisions/raw/commit_until_verified.sh` (the
content-verifying committer, which is the right pattern and is kept); commit `964a888`.
"""
out = src.rstrip("\n") + (ENTRY % n)
for _ in range(14):
    try:
        io.open(F, "w", encoding="utf-8", newline="").write(out); break
    except OSError: time.sleep(4)
back = io.open(F, encoding="utf-8").read()
a = back.count(MARKER); b = back.count("(#%d)" % n); c = back.count("(#32)")
print("marker=%d (want 1)  (#%d)=%d (want 1)  (#32) CONTROL=%d (want >=1)" % (a, n, b, c))
print("APPEND=%s" % ("PASS" if (a == 1 and b == 1 and c >= 1) else "FAIL"))
sys.exit(0 if (a == 1 and b == 1 and c >= 1) else 1)
