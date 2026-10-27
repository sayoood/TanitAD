p = 'C:/Users/Admin/dagread/mutate_dag.py'
s = open(p, encoding='utf-8', newline='').read()
new = open('C:/Users/Admin/dagread/scratch/new_mutants.txt', encoding='utf-8', newline='').read()
a = s.index('    ("N24", "the eligible fraction counts kept states')
b = s.index('    # ------------------------------------------------------------------------------------------------ new: the outcome texts and labels')
s = s[:a] + new + s[b:]
old = '''    ("N64", "the PREREG id names another commit", sub("@ commit 07dcb513", "@ commit deadbeef"), "prereg_file"),'''
assert s.count(old) == 1
s = s.replace(old, '''    ("N64", "the PREREG id names another commit", sub("@ commit 2a78f059", "@ commit deadbeef"), "prereg_file"),
    ("N137", "the PREREG id drops Amendment 1", sub('") + Amendment 1 (A1-2026-10-26))")', '"))")'), "prereg_file"),
    ("N138", "the pinned sha256 of the landed PREREG is another file's", sub('PREREG_LANDED_SHA256 = "a4df4b8f', 'PREREG_LANDED_SHA256 = "a4df4b80'), "prereg_file"),''')
s = s.replace('''"""Mutation proof for dag_read.py''', '''"""Mutation proof for dag_read.py (A1 edition: the reader re-pinned to the landed PREREG + Amendment 1, K-B3 as three controls, the bank read in the generator's real layout)''', 1) if False else s
open(p, 'w', encoding='utf-8', newline='').write(s)
print("mutants:", s.count('\n    ("'))
