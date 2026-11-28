"""A2: rewrite the two BANK-CONFIG partial mutants (N42, N104) for the new predicate and add N145-N151 to mutate_dag.py. Run once, from a pristine mutate_dag.py (scratch/pre_a2/)."""
import pathlib

m = pathlib.Path("C:/Users/Admin/dagread/mutate_dag.py")
t = m.read_text(encoding="utf-8")
assert "\r\n" not in t
old = r'''    ("N42", "BANK-CONFIG accepts a PARTIAL (--limit-clips) bank", sub('if c.get("partial") is not False or c.get("limit_clips") is not None:', "if False:"), "bank_config"),
    ("N104", "BANK-CONFIG looks at the partial flag only (limit_clips is ignored)", sub(' or c.get("limit_clips") is not None:', ":"), "bank_config"),
'''
assert t.count(old) == 1
SEL = "bank_config or limit_clips"
new = r'''    # the A2 predicate (the generator's --limit-clips convention: 0 = no limit, partial = bool(limit_clips)); the pre-A2 text was `if c.get("partial") is not False or c.get("limit_clips") is not None:`
    ("N42", "BANK-CONFIG accepts a PARTIAL (--limit-clips 12, partial True) bank", sub('    elif part:\n        bad.append(f"the bank is PARTIAL', '    elif False:\n        bad.append(f"the bank is PARTIAL'), "@SEL@"),
    ("N104", "BANK-CONFIG ignores limit_clips (the partial flag alone decides)", sub('    part, lim = c.get("partial"), c.get("limit_clips")\n', '    part, lim = c.get("partial"), None\n'), "@SEL@"),
    ("N145", "BANK-CONFIG is back to the pre-A2 predicate: limit_clips must be None, so the generator's 0 (no limit) is read as a limit and every real bank is refused",
     sub('    if not (isinstance(part, bool) and (lim is None or is_count(lim))):\n        bad.append(f"the bank\'s partial / limit_clips are not the generator\'s types (partial must be a JSON bool, limit_clips a non-negative int or null): partial {part!r}, limit_clips {lim!r}")\n'
         '    elif part != bool(lim):\n        bad.append(f"the bank\'s partial flag and limit_clips contradict each other (partial {part!r}, limit_clips {lim!r}): the generator writes partial = bool(limit_clips), 0 meaning no limit, so this is not a document it wrote")\n'
         '    elif part:\n',
         '    if part is not False or lim is not None:\n'), "@SEL@"),
    ("N146", "BANK-CONFIG drops the partial check (the flag is derived from limit_clips, never read)", sub('    part, lim = c.get("partial"), c.get("limit_clips")\n', '    part, lim = bool(c.get("limit_clips")), c.get("limit_clips")\n'), "@SEL@"),
    ("N147", "BANK-CONFIG accepts limit_clips > 0 under partial False", sub("    elif part != bool(lim):", "    elif part and not bool(lim):"), "@SEL@"),
    ("N148", "BANK-CONFIG reports partial True with limit_clips 0 / null as a plain PARTIAL bank, not as the contradiction it is", sub("    elif part != bool(lim):", "    elif bool(lim) and not part:"), "@SEL@"),
    ("N149", "BANK-CONFIG does not check the JSON types of partial / limit_clips", sub("    if not (isinstance(part, bool) and (lim is None or is_count(lim))):", "    if False:"), "@SEL@"),
    ("N150", "BANK-CONFIG takes a JSON bool limit_clips for an int", sub("(lim is None or is_count(lim))", "(lim is None or isinstance(lim, int))"), "@SEL@"),
    ("N151", "BANK-CONFIG takes any value for the partial flag (it need not be a JSON bool)", sub("isinstance(part, bool) and (lim is None", "True and (lim is None"), "@SEL@"),
'''.replace("@SEL@", SEL)
m.write_text(t.replace(old, new), encoding="utf-8", newline="\n")
print("mutate_dag.py rewritten")
