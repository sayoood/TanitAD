"""audit_cover.py -- lists every numeric token of RESULT_V8F_SPEEDSUB_T1.md / REGISTER_ROW.md that has no matching token in NUMBERS_AUDIT.md.
python -I -X utf8 audit_cover.py [--show]"""
import re
import sys
from decimal import Decimal

D = "E:/Projects/TanitAD-artifacts/dagland/speedsub_t1_result/"
TOK = re.compile(r"(?<![\w.-])(?:\d{4}-\d\d-\d\d(?!-[A-Za-z0-9])|\d{1,2}:\d\d(?::\d\d)?|(?!\d{4}-\d\d-\d\d)\d[\d,]*(?:\.\d+)?)")
HEX = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{6,}\b|\b[0-9a-f]{8}…")


def norm(t):
    if ":" in t or t.count("-") == 2:
        return t
    return format(Decimal(t.replace(",", "")).normalize(), "f")


STRUCT = [re.compile(r"(?im)^#+\s+[\d.]+"), re.compile(r"(?im)^\s*(?:>\s*)?\d+\.\s"),
          re.compile(r"(?i)\bsections?\s+[\d.]+(?:\s*\([a-d]\))?"), re.compile(r"(?i)\bpoint\s+\d\b"),
          re.compile(r"(?i)\bAmendments?\s+\d"), re.compile(r"\bA[12]\b"), re.compile(r"\(L\d+\)")]


def toks(text):
    text = HEX.sub(lambda m: " " * len(m[0]), text)
    for rx in STRUCT:
        text = rx.sub(lambda m: " " * len(m[0]), text)
    return [(norm(m[0]), m.start(), m[0]) for m in TOK.finditer(text)]


def main():
    audit = open(D + "NUMBERS_AUDIT.md", encoding="utf-8").read()
    head, _, rest = audit.partition("| section | rendered | source | arithmetic / note |")
    cov = {t for t, _, _ in toks(rest)}
    out = {}
    for f in ("RESULT_V8F_SPEEDSUB_T1.md", "REGISTER_ROW.md"):
        text = open(D + f, encoding="utf-8").read()
        un = []
        for n, pos, raw in toks(text):
            if n not in cov:
                un.append((n, text[max(0, pos - 45):pos + 30].replace("\n", " ")))
        out[f] = un
        print(f, "numeric tokens", len(toks(text)), "uncovered", len(un))
        if "--show" in sys.argv:
            for n, c in un:
                print("   ", n, "|", c)
    return out


if __name__ == "__main__":
    main()
