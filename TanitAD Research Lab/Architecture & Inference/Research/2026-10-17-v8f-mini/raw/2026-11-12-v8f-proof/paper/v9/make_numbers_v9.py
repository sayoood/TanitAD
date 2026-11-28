"""make_numbers_v9.py (style of make_numbers_v8.py): every numeric token (decimal or comma-formatted) on a line that v9 adds or changes
against v8 must be found literally in rows_84_91.txt (register rows H-V8F-PROOF-84, 86 to 91) or, as a rounded value, in the three
NAMES json files of the level study (register row 92 in preparation; ROW92.md did not exist). Tokens already on the old line are skipped.
Writes NUMBERS_V9.tsv and numbers_unmatched_v9.txt."""
import difflib, re, json

V = "F:/TanitAD-scratch/v8proof/paper/"
a = open(V + "v8/tanitad_4b_paper.tex", encoding="utf8").read().replace("\r\n", "\n").split("\n")
b = open(V + "v9/tanitad_4b_paper.tex", encoding="utf8").read().split("\n")
new_lines, old_of = set(), {}
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
    if tag in ("replace", "insert"):
        for j in range(j1, j2):
            new_lines.add(j)
            old_of[j] = " ".join(a[i1:i2])

rows = {}
for ln in open(V + "v9/rows_84_91.txt", encoding="utf8").read().split("\n"):
    m = re.match(r"\| \*\*H-V8F-PROOF-(\d+) ", ln)
    if m:
        rows[int(m.group(1))] = ln

floats = []
def walk(o):
    if isinstance(o, float):
        floats.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
for n in ("RLVa", "RLVb", "RLAh"):
    walk(json.load(open("F:/TanitAD-scratch/v8proof/actlevel/read/NAMES_%s.json" % n, encoding="utf8")))

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
        src = None
        for k, tx in rows.items():
            if re.search(r"(?<![\d.])" + re.escape(val) + r"(?![\d])", tx):
                src = "H-V8F-PROOF-%d" % k
                break
        if src is None and "." in val and "," not in val:
            d = len(val.split(".")[1])
            for f in floats:
                if ("%." + str(d) + "f") % abs(f) == val:
                    src = "NAMES json (rounded)"
                    break
        if src:
            out.append((val, str(ln + 1), src))
        else:
            miss.append((val, ln + 1, line[:110]))

with open(V + "v9/NUMBERS_V9.tsv", "w", encoding="utf8", newline="\n") as f:
    f.write("value\ttex_line\tsource\n")
    for r in out:
        f.write("\t".join(r) + "\n")
with open(V + "v9/numbers_unmatched_v9.txt", "w", encoding="utf8", newline="\n") as f:
    for m in miss:
        f.write("%s\t%d\t%s\n" % m)
print(len(out), "matched", len(miss), "unmatched")
