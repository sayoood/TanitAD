import io, sys
P = r"C:\Users\Admin\par18\hnfp3\work\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\refe\test_hn_content_fp.py"
raw = open(P, "rb").read()
crlf = b"\r\n" in raw
t = raw.decode("utf-8")
assert not crlf, "file has CRLF"
def rep(old, new, n_expect):
    global t
    c = t.count(old)
    assert c == n_expect, (c, n_expect, old[:100])
    t = t.replace(old, new)
# (a) cmd_verify: the verify_token call now also carries ctx_anchor=pa.get("ctx_anchor")  [2 sites: M5, M6]
rep(r'''"commit=a.commit, content_flags=content_flags_of(prof) if prof else ())", "commit=a.commit)"''',
    r'''"commit=a.commit, content_flags=content_flags_of(prof) if prof else (), ctx_anchor=pa.get(\"ctx_anchor\"))", "commit=a.commit, ctx_anchor=pa.get(\"ctx_anchor\"))"''', 2)
# (b) finalize / run_checks: data_manifest call now also carries ctx_anchor=prof.get("ctx_anchor")
for stage, n in (("finalize", 2), ("run", 1)):
    rep(r'''timings=dtim, content_flags=content_flags_of(prof))\n    _log_manifest_timing(\"''' + stage + r'''\"''',
        r'''timings=dtim, content_flags=content_flags_of(prof), ctx_anchor=prof.get(\"ctx_anchor\"))\n    _log_manifest_timing(\"''' + stage + r'''\"''', n)
    rep(r'''timings=dtim)\n    _log_manifest_timing(\"''' + stage + r'''\"''',
        r'''timings=dtim, ctx_anchor=prof.get(\"ctx_anchor\"))\n    _log_manifest_timing(\"''' + stage + r'''\"''', n)
# (c) the symlink mutant: HNFP3's _stat_listing carries the SAME two statements, so the snippet is made unique by the lines that follow it in _content_listing (they end in `if rel == CONTENT_MANIFEST:`)
tail_old = r'''for d in dirnames if os.path.islink(os.path.join(dirpath, d))]\n",
                        "        pass  # MUTANT\n")],'''
assert t.count(tail_old) == 1
CTX = r'''        for fn in sorted(filenames):\n            full = os.path.join(dirpath, fn)\n            rel = os.path.relpath(full, rootp).replace(os.sep, \"/\")\n            if rel == CONTENT_MANIFEST:\n'''
t = t.replace(tail_old, r'''for d in dirnames if os.path.islink(os.path.join(dirpath, d))]\n''' + CTX + r'''",
                        "        pass  # MUTANT\n''' + CTX + r'''")],''')
# (d) fp_key: HNFP3 added a `dir-anchor` branch with the same guard line; the snippet now names the dir-content branch's return line too
rep(r'''[("        if fp.get(\"n_problems\") or not fp.get(\"content_sha256\"):\n", "        if False:\n")],''',
    r'''[("        if fp.get(\"n_problems\") or not fp.get(\"content_sha256\"):\n            return f\"dir-content-INVALID:", "        if False:\n            return f\"dir-content-INVALID:")],''', 1)
open(P, "wb").write(t.encode("utf-8"))
print("patched")
