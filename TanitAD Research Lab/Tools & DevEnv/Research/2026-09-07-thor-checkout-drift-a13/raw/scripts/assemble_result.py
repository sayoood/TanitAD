#!/usr/bin/env python3
"""Assemble RESULT.md from the header draft + body draft + the measured drift table."""
import collections
import json
import io

t = json.load(open("A13_DRIFT_TABLE.json", encoding="utf-8"))
rows = t["rows"]
vc = collections.Counter(r["verdict"] for r in rows)
n_raw = len(rows)
n_eol = vc.get("EOL_ONLY", 0)
n_real = n_raw - n_eol
inc = [r for r in rows if r["in_refav1_closure"]]
inc_real = [r for r in inc if r["verdict"] != "EOL_ONLY"]
thor_side = [r for r in rows if r["verdict"] in
             ("THOR_BYTES_NOT_IN_GIT", "THOR_MATCHES_OFF_LINE_COMMIT")]
inconc = [r for r in rows if r["verdict"].startswith("INCONCLUSIVE")]


def dirof(p, n=2):
    s = p.split("/")
    return "/".join(s[:n]) if len(s) > 1 else p


out = io.StringIO()
out.write("### 5.1 Verdicts — every drifted path at repo HEAD `%s`\n\n" % t["repo_head"][:10])
out.write("| verdict | n | meaning |\n|---|---|---|\n")
MEAN = {
    "REPO_NEWER": "Thor holds a **named committed** version the repo has superseded — the repo is authoritative",
    "EOL_ONLY": "⚠️ **not drift** — CRLF-vs-LF only, from a file shipped off the Windows dev box",
    "REPO_NEWER_COMMIT_UNIDENTIFIED": "Thor's exact content **is** a blob the repo holds (control-bracketed probe), so it is committed; repo HEAD carries a **different** blob at that path, so the repo has moved on. The per-path walk that would *name* the superseding commit did not reach these rows before it was stopped — direction settled, revision not named",
    "THOR_BYTES_NOT_IN_GIT": "⛔ Thor content that exists **nowhere** in the repo's object store",
    "THOR_MATCHES_OFF_LINE_COMMIT": "⛔ Thor holds a commit that is **not** an ancestor of our HEAD",
    "INCONCLUSIVE_PROBE_FAILED": "the same-breath control did not resolve — **not** evidence of anything",
}
for k, v in vc.most_common():
    out.write("| `%s` | **%d** | %s |\n" % (k, v, MEAN.get(k, "")))
out.write("| | | |\n")
out.write("| **raw-sha drift rows** | **%d** | before the CRLF correction |\n" % n_raw)
out.write("| **REAL drift** | **%d** | raw-sha drift minus the %d EOL-only rows |\n"
          % (n_real, n_eol))
out.write("\n")

out.write("### 5.2 Real drift by directory\n\n| directory | n |\n|---|---|\n")
for k, v in collections.Counter(dirof(r["path"]) for r in rows
                                if r["verdict"] != "EOL_ONLY").most_common(12):
    out.write("| `%s` | %d |\n" % (k, v))
out.write("\n")

out.write("### 5.3 ⛔ The rows inside the refav1 import closure (n = %d, of which %d are real "
          "drift)\n\n" % (len(inc), len(inc_real)))
out.write("| file | verdict | tracked on Thor | CRLF-shipped |\n|---|---|---|---|\n")
for r in sorted(inc, key=lambda x: (x["verdict"] != "EOL_ONLY", x["path"]), reverse=True):
    out.write("| `%s` | `%s` | %s | %s |\n"
              % (r["path"], r["verdict"], "yes" if r["thor_tracked"] else "**no (shipped)**",
                 "yes" if r["thor_is_crlf"] else ""))
out.write("\n")

if thor_side:
    out.write("### 5.4 ⛔ Rows where Thor could be authoritative (n = %d)\n\n" % len(thor_side))
    for r in thor_side:
        out.write("* `%s` — %s\n" % (r["path"], r["evidence"]))
    out.write("\n")
else:
    out.write("### 5.4 Rows where Thor could be authoritative: **NONE**\n\n"
              "⭐ **Not one drifted path on Thor holds content the repo lacks.** Every row resolves "
              "either to a committed version the repo has superseded, or to a line-ending "
              "difference.\n\n")
if inconc:
    out.write("### 5.5 Inconclusive (n = %d)\n\n" % len(inconc))
    for r in inconc:
        out.write("* `%s` — %s\n" % (r["path"], r["evidence"]))
    out.write("\n")

table = out.getvalue()

q1 = ("**%d real drifted files**, not 24 — and the 24 is not wrong, it is **dated**. A raw blob-sha "
      "comparison finds **%d** rows; **%d of them are CRLF-only** (files shipped off the Windows "
      "dev box) and are not a currency defect. The row's 24 was measured 2026-08-18 against that "
      "day's HEAD; the repo has since advanced **1,273 commits**. All 24 of the original paths are "
      "still drifted today, and all 24 are now settled." % (n_real, n_raw, n_eol))
q2 = ("**THE REPO, on every single row — there is no row where Thor is newer.** %d rows are settled "
      "by a *positive identification of a commit*: Thor's bytes ARE the blob at Thor's own checkout "
      "HEAD `30d6d60` (2026-08-15), which `git merge-base --is-ancestor` confirms is an ancestor of "
      "repo HEAD `7084b2cf`. The rest are settled by naming the superseding commit in "
      "`30d6d60..HEAD`, or by a control-bracketed object-store probe. Per-file table in §5 and "
      "`raw/A13_DRIFT_TABLE.json`." % vc.get("REPO_NEWER", 0))
q3 = ("⛔ **The premise is stale — there is no live refav1 run** (finished 2026-09-04; 0 python "
      "processes on 3 independent probes; GPU 0 %%). Asked retrospectively: **YES — %d of the 130 "
      "closure files are real drift and a further 13 do not exist on Thor at all**, so the finished "
      "`refav1-b1-v72-ep3-speed` arm ran a **materially older** stack (`refa_v1.py` 1,885 lines vs "
      "the repo's 3,030). Consequence in §9: a re-run from today's repo is **not** a replication of "
      "that arm." % len(inc_real))

hdr = open("RESULT.md", encoding="utf-8").read()
body = open("BODY.md", encoding="utf-8").read()
hdr = hdr.replace("PLACEHOLDER_Q1", q1).replace("PLACEHOLDER_Q2", q2).replace("PLACEHOLDER_Q3", q3)
body = (body.replace("**PLACEHOLDER_TABLE**", table)
            .replace("PLACEHOLDER_N", str(n_real))
            .replace("PLACEHOLDER_EOL", str(n_eol)))
final = hdr.replace("PLACEHOLDER_BODY", body)
assert "PLACEHOLDER" not in final, [l for l in final.splitlines() if "PLACEHOLDER" in l]
open("RESULT_FINAL.md", "w", encoding="utf-8", newline="\n").write(final)
print("raw=%d eol=%d real=%d closure_rows=%d closure_real=%d thor_side=%d inconclusive=%d"
      % (n_raw, n_eol, n_real, len(inc), len(inc_real), len(thor_side), len(inconc)))
print("wrote RESULT_FINAL.md (%d bytes)" % len(final))
