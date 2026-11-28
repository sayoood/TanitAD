"""make_numbers_v7.py: for every numeric token (decimal or comma-formatted) on a line that v7 adds or changes against v6,
find the register row (H-V8F-PROOF-41 to 54, rows.txt) that contains it literally, and write NUMBERS_V7.tsv:
value, tex line, register row, artifact path named by the row, 80-character snippet. Unmatched tokens go to numbers_unmatched_v7.txt.
Adapted from make_numbers_v6.py (v6 files are not edited)."""
import difflib, re

V = "F:/TanitAD-scratch/v8proof/paper/"
a = open(V + "v6/tanitad_4b_paper.tex", encoding="utf8").read().split("\n")
b = open(V + "v7/tanitad_4b_paper.tex", encoding="utf8").read().split("\n")
new_lines = set()
old_of = {}
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
    if tag in ("replace", "insert"):
        new_lines.update(range(j1, j2))
        for j in range(j1, j2):
            old_of[j] = " ".join(a[i1:i2])

rows = {}
for ln in open(V + "v7/rows.txt", encoding="utf8").read().split("\n"):
    m = re.match(r"\| \*\*H-V8F-PROOF-(\d+) ", ln)
    if m:
        rows[int(m.group(1))] = ln
art = {}
for k, ln in rows.items():
    p = re.findall(r"`([^`]*/[^`]*)`", ln)
    art[k] = p[0] if p else ""

tok = re.compile(r"(?<![\w.])[+-]?\$?[+-]?\d[\d,]*\.\d+|(?<![\w.])\d{1,3}(?:,\d{3})+(?![\d])")
out, miss, seen = [], [], set()
for ln in sorted(new_lines):
    line = b[ln]
    if line.lstrip().startswith("%"):
        continue
    for m in tok.finditer(line):
        raw = m.group(0).replace("$", "")
        val = raw.lstrip("+-")
        if val in old_of.get(ln, ""):
            continue
        if (val, ln) in seen:
            continue
        seen.add((val, ln))
        parts = re.split(r"(?<=[a-z\)\}])\.\s", line)
        off, sent = 0, line
        for part in parts:
            if off <= m.start() < off + len(part) + 2:
                sent = part
                break
            off += len(part) + 2
        others = [t.group(0).replace("$", "").lstrip("+-") for t in tok.finditer(sent)]
        others = [o for o in others if o != val]
        best = (-1, None)
        for k, tx in rows.items():
            for mm in re.finditer(r"(?<![\d.])" + re.escape(val) + r"(?![\d])", tx):
                win = tx[max(0, mm.start() - 250):mm.end() + 250]
                sc = sum(1 for o in set(others) if o in win)
                if sc > best[0]:
                    snip = tx[max(0, mm.start() - 30):mm.end() + 30].replace("\n", " ").replace("\t", " ")
                    best = (sc, (k, snip[:80]))
        if best[1]:
            k, snip = best[1]
            out.append((val, str(ln + 1), "H-V8F-PROOF-%d" % k, art[k], snip))
        else:
            miss.append((val, ln + 1, line[:110]))

with open(V + "v7/NUMBERS_V7.tsv", "w", encoding="utf8", newline="\n") as f:
    f.write("value\ttex_line\tregister_row\tartifact_path_named_by_row\tsource_snippet\n")
    for r in out:
        f.write("\t".join(r) + "\n")
with open(V + "v7/numbers_unmatched_v7.txt", "w", encoding="utf8", newline="\n") as f:
    for m in miss:
        f.write("%s\t%d\t%s\n" % m)
print(len(out), "matched", len(miss), "unmatched")
