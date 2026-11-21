import sys
p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


# run: the token binds the extra files that mint bound (evidence, dry record, merge record, releases, selfcheck); the runner must re-bind the same set
rep('    bad = token_problems(os.path.join(a.gate_dir, "gate_token.json"), a.arm, a.stage, toks, a.root, a.commit)\n',
    '    extra = {"--union-release": a.union_release, "tenk_v9": a.tenk_v9, "r_release": a.r_release, "merge_record": a.merge_record, "dry_record": a.dry, "evidence": a.evidence,\n'
    '             "selfcheck": getattr(a, "selfcheck", "")}                                  # gate patch 2026-11-17: the same extra files mint bound\n'
    '    bad = token_problems(os.path.join(a.gate_dir, "gate_token.json"), a.arm, a.stage, toks, a.root, a.commit, {k: v for k, v in extra.items() if v})\n')
rep('        if nm == "mint":\n            p.add_argument("--base-argv", required=True);',
    '        if nm == "run":\n'
    '            p.add_argument("--evidence", default=""); p.add_argument("--dry", default=""); p.add_argument("--merge-record", default=""); p.add_argument("--union-release", default="")\n'
    '            p.add_argument("--tenk-v9", default=""); p.add_argument("--r-release", default=""); p.add_argument("--selfcheck", default="")\n'
    '        if nm == "mint":\n            p.add_argument("--base-argv", required=True);')
rep('"dry_record": a.dry, "evidence": a.evidence}\n    b, bad = binding(arm, stage, toks, a.root, a.commit, {k: v for k, v in extra.items() if v})',
    '"dry_record": a.dry, "evidence": a.evidence,\n             "selfcheck": a.selfcheck}\n    b, bad = binding(arm, stage, toks, a.root, a.commit, {k: v for k, v in extra.items() if v})')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched2", p)
