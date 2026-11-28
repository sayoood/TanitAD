"""DELIBERATE-REGRESSION ARM: re-introduce ONLY the defect line, nothing else.

`--on`  : replace the raise with the pre-2026-09-06 `return {}`
`--off` : restore the raise
ASCII-only stdout. Surgical on purpose -- if the whole file were reverted the
tests would fail on the missing scaffolding, which proves nothing about the
DEFECT. This mutation changes one branch and leaves every name in place.
"""
import io
import sys

P = ("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/"
     "stack/tanitad/data/alpamayo_records.py")
NL = "\r\n"

GOOD = NL.join([
    "        raise AlpamayoRecordsUnavailable(",
    '            "Alpamayo records parquet not found: %r. Set %s to the parquet "',
]) + NL
BAD = NL.join([
    "        return {}  # MUTATION: the pre-2026-09-06 silent empty",
    '        raise AlpamayoRecordsUnavailable(',
    '            "Alpamayo records parquet not found: %r. Set %s to the parquet "',
]) + NL

src = io.open(P, encoding="utf-8", newline="").read()
if "--on" in sys.argv:
    assert src.count(GOOD) == 1 and src.count(BAD) == 0, "already mutated?"
    src = src.replace(GOOD, BAD)
    state = "MUTATION ON (defect re-introduced)"
elif "--off" in sys.argv:
    assert src.count(BAD) == 1, "not mutated"
    src = src.replace(BAD, GOOD)
    state = "MUTATION OFF (fix restored)"
else:
    print("MUTATED" if BAD in src else "CLEAN")
    sys.exit(0)
io.open(P, "w", encoding="utf-8", newline="").write(src)
print(state)
