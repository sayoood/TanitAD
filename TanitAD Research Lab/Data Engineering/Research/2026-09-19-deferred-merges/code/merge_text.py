"""Three-way merges of the deferred text files onto the CURRENT tip (2a88524).

Rule: each side's edits are computed against THAT SIDE'S OWN BASE and applied onto
the tip only where they touch DISJOINT regions. Two sides that both append at EOF
are ordered explicitly (by date). Any other overlap is REFUSED, never guessed.

Sources: Bp = d221843 (base of D:'s pre-sync backup), B = 57e2944 (the backup),
T0 = 37645fc (D:'s HEAD, base of today's worktree edits), T1 = 2a88524 (the tip),
W = D:'s current worktree.

Assertions per file (all must hold or nothing is written):
  * TIP SUPERSET — every tip line survives, except rows a side deliberately
    REPLACED (named per file and checked to be replaced by a longer/newer row);
  * every line a side ADDED is present;
  * no stray line: result lines are a subset of tip + backup + worktree lines.
Worktree writes are compare-and-swap: the file is re-hashed immediately before the
write and the write is refused if it changed since it was read.
Files are written CRLF, matching the worktree (core.autocrlf=true).

Usage: python merge_text.py <out_dir_for_json_and_proposals> [--write]
"""
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

R = "D:/Projects/TanitAD"
Bp, B, T0, T1 = "d221843", "57e2944", "37645fc", "2a88524"
OUT = Path(sys.argv[1])
WRITE = "--write" in sys.argv


def ref(r, p):
    b = subprocess.run(["git", "-C", R, "show", "%s:%s" % (r, p)], capture_output=True).stdout
    return b.replace(b"\r\n", b"\n").decode("utf-8").split("\n")


def wt_bytes(p):
    return (Path(R) / p).read_bytes()


def wt(p):
    return wt_bytes(p).replace(b"\r\n", b"\n").decode("utf-8").split("\n")


def edits(base, new, tag):
    return [(i1, i2, new[j1:j2], tag) for t, i1, i2, j1, j2
            in difflib.SequenceMatcher(None, base, new, autojunk=False).get_opcodes()
            if t != "equal"]


def apply(base, groups):
    """groups: edit lists in PRIORITY order (earlier = placed first at a shared EOF)."""
    allx = [e for g in groups for e in g]
    for a in range(len(allx)):
        for b in range(a + 1, len(allx)):
            (i1, i2, _, ta), (j1, j2, _, tb) = allx[a], allx[b]
            same_point = i1 == i2 == j1 == j2
            if same_point and i1 >= len(base) - 1:
                continue                        # both append at EOF: ordered by priority
            if same_point or (i1 < j2 and j1 < i2):
                raise SystemExit("REFUSED: overlapping edits %s@%d-%d and %s@%d-%d"
                                 % (ta, i1, i2, tb, j1, j2))
    out = list(base)
    # apply from the end; at a shared EOF point insert the LOWER-priority block (larger
    # k) first, so the higher-priority block, inserted after it at the same index,
    # lands BEFORE it. ⚠️ (i1, -k) gets this backwards -- asserted per file below.
    order = sorted(range(len(allx)), key=lambda k: (allx[k][0], k), reverse=True)
    for k in order:
        i1, i2, rep, _ = allx[k]
        out[i1:i2] = rep
    return out


def check(name, tip, result, added, replaced_ok):
    tipc = [l for l in tip if l.strip()]
    res = set(result)
    lost = [l for l in tipc if l not in res and l not in replaced_ok]
    miss = [l for l in added if l.strip() and l not in res]
    universe = set(tip) | set().union(*[set(x) for x in added_sources[name]])
    stray = [l for l in result if l not in universe]
    ok = not lost and not miss and not stray
    return {"tip_lines_lost": len(lost), "added_lines_missing": len(miss),
            "stray_lines": len(stray), "replaced_rows": len(replaced_ok), "pass": ok}


report, added_sources, results = {}, {}, {}

# ── 1. LAB_BACKLOG.md : base == tip; backup (vs Bp) + worktree (vs T0) ──────────
p = "TanitAD Research Lab/LAB_BACKLOG.md"
tip, bp, t0 = ref(T1, p), ref(Bp, p), ref(T0, p)
assert tip == bp == t0, "LAB_BACKLOG base is not the tip"
b, w = ref(B, p), wt(p)
eb, ew = edits(bp, b, "backup"), edits(t0, w, "worktree")
res = apply(tip, [eb, ew])                       # EOF: backup (09-13) before lab (09-19)
repl = [tip[i1] for i1, i2, _, _ in eb + ew if i2 == i1 + 1]
added_sources[p] = [b, w]
report[p] = check(p, tip, res, [l for e in eb + ew for l in e[2]], set(repl))
report[p]["replaced"] = [{"old_chars": len(tip[i1]), "new_chars": len(rep[0]), "side": tag,
                          "row": tip[i1].split("|")[1].strip()}
                         for i1, i2, rep, tag in eb + ew if i2 == i1 + 1 and len(rep) == 1]
results[p] = (res, w)

# ── 2. GOALS_AND_CLAIMS.md : tip == T0; worktree adds + restore LAB-RUN-012 heading ─
p = "Project Steering/GOALS_AND_CLAIMS.md"
tip, t0 = ref(T1, p), ref(T0, p)
assert tip == t0
w, b = wt(p), ref(B, p)
ew = edits(t0, w, "worktree")
assert all(i1 == i2 or not any(l.strip() for l in t0[i1:i2]) for i1, i2, _, _ in ew), \
    "the worktree DELETES tip content in GOALS"
hdr = "### 2026-09-13 — Research Lab (LAB-RUN-012) register updates"
mk = "<!-- LAB-RUN-012-2026-09-13 -->"
i = b.index(hdr)
blk = []
for l in b[i + 2:]:
    if not l.startswith("|"):
        break
    blk.append(l)
at = [k for k in range(len(tip)) if tip[k:k + len(blk)] == blk]
assert len(at) == 1, "the LAB-RUN-012 table is not uniquely located in the tip"
eh = [(at[0], at[0], [mk, "", hdr, ""], "backup-heading")]
res = apply(tip, [ew, eh])
added_sources[p] = [w, [mk, "", hdr, ""]]
report[p] = check(p, tip, res, [l for e in ew for l in e[2]] + [mk, hdr], set())
report[p]["lab_run_012_table_at_tip_line"] = at[0] + 1
results[p] = (res, w)

# ── 3. RETRACTION_LOG.md : base T0; tip adds + worktree adds ─────────────────
p = "Project Steering/RETRACTION_LOG.md"
tip, t0, w, b, bp = ref(T1, p), ref(T0, p), wt(p), ref(B, p), ref(Bp, p)
et, ew = edits(t0, tip, "tip"), edits(t0, w, "worktree")
res = apply(t0, [et, ew])                        # EOF: landed tip entry, then the lab's
backup_missing = [l for e in edits(bp, b, "backup") for l in e[2] if l.strip() and l not in set(res)]
added_sources[p] = [w]
report[p] = check(p, tip, res, [l for e in ew for l in e[2]], set())
report[p]["backup_edit_lines_not_in_result"] = len(backup_missing)
results[p] = (res, w)

# ── 4. PREREG : base T0; tip adds (E17/E18) + worktree adds (A7 amendment) ──────
p = "Project Steering/PREREG_REFCV6_DEVBOX_PREPARATION.md"
tip, t0, w = ref(T1, p), ref(T0, p), wt(p)
et, ew = edits(t0, tip, "tip"), edits(t0, w, "worktree")
res = apply(t0, [et, ew])
added_sources[p] = [w]
report[p] = check(p, tip, res, [l for e in ew for l in e[2]], set())
report[p]["NOT_WRITTEN_TO_WORKTREE"] = ("the added section is the Training FlyWheel's A7 "
                                        "amendment; A7 is on the do-not-touch list")
results[p] = (res, w)

# ── 5. ANCHOR_REPAIR.md : the backup is the later revision (judgment, see RESULT) ─
p = ("TanitAD Research Lab/Architecture & Inference/Implementation/incoming/"
     "2026-09-11-agentgt-anchor-repair/ANCHOR_REPAIR.md")
tip, b, w = ref(T1, p), ref(B, p), wt(p)
assert w == tip, "ANCHOR_REPAIR worktree moved"
report[p] = {"verdict": "BACKUP-REVISION", "tip_lines": len(tip), "result_lines": len(b),
             "pass": True}
results[p] = (b, w)

def first(lines, needle):
    return next(k for k, l in enumerate(lines) if needle in l)


# ⛔ ORDER at a shared EOF is invisible to the superset checks -- assert it explicitly
bl = results["TanitAD Research Lab/LAB_BACKLOG.md"][0]
assert first(bl, "LAB-RUN-012") < first(bl, "(LAB-RUN-016)"), "backlog EOF order"
rl = results["Project Steering/RETRACTION_LOG.md"][0]
assert first(rl, "RETR-2026-09-19-SOURCE-FRAMES-ABSENT") < \
    first(rl, "RETR-2026-09-19-POSTED-LIMIT-REFIND"), "retraction-log EOF order"
pr = results["Project Steering/PREREG_REFCV6_DEVBOX_PREPARATION.md"][0]
report["Project Steering/PREREG_REFCV6_DEVBOX_PREPARATION.md"]["order"] = (
    "tip E17/E18 before A7 amendment" if first(pr, "E17") < first(pr, "A7-AMENDMENT-BN")
    else "A7 amendment before tip E17/E18 (non-EOF placement)")
report["_order_assertions"] = "PASS: backup 09-13 block before lab 09-19 block; landed tip entry before the lab's"

print(json.dumps(report, indent=1, ensure_ascii=False))
assert all(v.get("pass") for k, v in report.items() if not k.startswith("_")), \
    "a merge failed its assertions"

OUT.mkdir(parents=True, exist_ok=True)
json.dump(report, open(OUT / "merge_report.json", "w", encoding="utf-8"), indent=1,
          ensure_ascii=False)
for p, (res, w_seen) in results.items():
    data = "\r\n".join(res).encode("utf-8")
    if p.endswith("PREREG_REFCV6_DEVBOX_PREPARATION.md"):
        (OUT / "PREREG_REFCV6_DEVBOX_PREPARATION.tip-based.md").write_bytes(
            "\n".join(res).encode("utf-8"))
        print("PROPOSAL (not written to the worktree):", p)
        continue
    if not WRITE:
        continue
    now = wt(p)
    if now != w_seen:                              # compare-and-swap
        raise SystemExit("REFUSED: %s changed on disk since it was read" % p)
    (Path(R) / p).write_bytes(data)
    print("written (CRLF):", p, "sha1-blob", subprocess.run(
        ["git", "-C", R, "hash-object", "--", p], capture_output=True).stdout.decode().strip()[:10])
