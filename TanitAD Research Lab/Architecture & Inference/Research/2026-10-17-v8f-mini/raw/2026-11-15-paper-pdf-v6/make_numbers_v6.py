"""make_numbers_v6.py: for every numeric token on a line that v6 adds or changes against v5, find a source file that contains it
(literally, or through a stated rounding) and write NUMBERS_V6.tsv: value, tex line, source file, source snippet (max 80 chars)."""
import difflib, re, glob, os

V = "F:/TanitAD-scratch/v8proof/"
a = open(V + "paper/v5/tanitad_4b_paper.tex", encoding="utf8").read().split("\n")
b = open(V + "paper/v6/tanitad_4b_paper.tex", encoding="utf8").read().split("\n")
new_lines = set()
old_of = {}
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
    if tag in ("replace", "insert"):
        new_lines.update(range(j1, j2))
        for j in range(j1, j2):
            old_of[j] = " ".join(a[i1:i2])

srcs = sorted(glob.glob(V + "GAC_ROWS_2*.md") + glob.glob(V + "GAC_ROWS_3*.md"))
srcs += [V + "lean/LEAN_RESULT_RULE.md", V + "spatial/q2/out_m/VERDICT_RB_PR_s0.md", V + "spatial/q2/out_t/VERDICT_RB_PE_s0.md",
         V + "spatial/q2/out_m/CHANGE_RB_PR_s0_P_s0.md", V + "spatial/q2/out_t/CHANGE_RB_PE_s0_P_s0.md",
         V + "spatial/q2/out_m/PAIR_RB_PR_s0_minus_P_s0.md", V + "spatial/q2/out_t/PAIR_RB_PE_s0_minus_P_s0.md",
         V + "GAC_ROWS_1.md", V + "GAC_ROWS_17.md", V + "GAC_ROWS_18.md", V + "page/RESULTS_BLOCK.md"]
texts = {s: open(s, encoding="utf8").read() for s in srcs if os.path.exists(s)}
# the six rounded values and the stated rounding (value in paper -> value as banked)
ROUND = {"0.55": "0.552", "0.54": "0.541", "0.24": "0.237", "0.21": "0.212", "0.36": "0.361", "0.31": "0.309",
         "0.79": "0.788", "0.80": "0.796", "0.78": "0.779", "0.72": "0.715", "1.23": "1.233", "1.25": "1.254"}
ROUNDFILE = {"0.55": 23, "0.54": 24, "0.24": 24, "0.21": 24, "0.36": 25, "0.31": 25, "0.79": 31, "0.80": 31, "0.78": 26, "0.72": 26, "1.23": 27, "1.25": 27}

tok = re.compile(r"(?<![\w.])[+-]?\$?[+-]?\d[\d,]*\.\d+|(?<![\w.])\d{1,3}(?:,\d{3})+(?![\d])")
NEWBLOCK_OK = set()
rows, miss = [], []
seen = set()
for ln in sorted(new_lines):
    line = b[ln]
    if line.lstrip().startswith("%"):
        continue
    for m in tok.finditer(line):
        raw = m.group(0).replace("$", "")
        val = raw.lstrip("+-")
        key = (val, ln)
        if val in old_of.get(ln, "") and ln not in NEWBLOCK_OK:
            continue
        if key in seen:
            continue
        seen.add(key)
        rounded = val in ROUND and not (val == "0.80" and ln + 1 != 78)
        look = ROUND[val] if rounded else val
        # a rounded value is only accepted for the paper values listed in ROUND; they are searched as banked
        found = None
        # context score: other numbers of the same sentence that sit within 250 characters of the hit in the source
        sent = None
        pos = m.start()
        parts = re.split(r"(?<=[a-z\)\}])\.\s", line)
        off = 0
        for part in parts:
            if off <= pos < off + len(part) + 2:
                sent = part
                break
            off += len(part) + 2
        sent = sent or line
        others = [t.group(0).replace("$", "").lstrip("+-") for t in tok.finditer(sent)]
        others = [o for o in others if o != val]
        best = (-1, None)
        for s, tx in texts.items():
            if rounded and val in ROUNDFILE and not s.endswith("GAC_ROWS_%d.md" % ROUNDFILE[val]):
                continue
            for mm in re.finditer(r"(?<![\d.])" + re.escape(look) + r"(?![\d])", tx):
                win = tx[max(0, mm.start() - 250):mm.end() + 250]
                sc = sum(1 for o in set(others) if o in win)
                if sc > best[0]:
                    snip = tx[max(0, mm.start() - 30):mm.end() + 30].replace(chr(10), " ").replace(chr(9), " ")
                    best = (sc, (s, snip[:80]))
        found = best[1]
        note = "" if look == val else " (rounds %s)" % look
        if found:
            rows.append((val + note, str(ln + 1), found[0].replace(V, ""), found[1]))
        else:
            miss.append((val, ln + 1, line[:100]))

with open(V + "paper/v6/NUMBERS_V6.tsv", "w", encoding="utf8", newline="\n") as f:
    f.write("value\ttex_line\tsource_file\tsource_snippet\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
with open(V + "paper/v6/numbers_unmatched.txt", "w", encoding="utf8", newline="\n") as f:
    for m in miss:
        f.write("%s\t%d\t%s\n" % m)
print(len(rows), "matched", len(miss), "unmatched")
