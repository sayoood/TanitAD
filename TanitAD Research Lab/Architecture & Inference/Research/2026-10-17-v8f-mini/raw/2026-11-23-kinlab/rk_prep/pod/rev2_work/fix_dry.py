p = "test_chain_dry_rk.py"
s = open(p, encoding="utf-8").read()
a = s.index("def sh(p):")
b = s.index("def plant(")
s = s[:a] + 'def sh(p):                      # forward-slash Windows form: valid for Git bash (which converts path arguments for native programs) and for the Windows python\n    return p.replace("\\\\", "/")\n\n\n' + s[b:]
s = s.replace(', DRY="1", MSYS_NO_PATHCONV="1")', ', DRY="1")')
open(p, "w", encoding="utf-8").write(s)
