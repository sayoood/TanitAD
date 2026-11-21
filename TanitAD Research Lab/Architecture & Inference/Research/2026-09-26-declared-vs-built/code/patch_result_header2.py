import sys
P = sys.argv[1]
s = open(P, encoding="utf-8").read()
reps = [
 ("**Scope:** FIX-3, FIX-4, FIX-5, G3 and the two gate modules G-HYG and G-DVB of `Project Steering/SPEC_REFCV7.md`\n(branch copy at `5de9363`, including amendment A1 §6).",
  "**Scope:** FIX-3, FIX-4, FIX-5, G3 and the two gate modules G-HYG and G-DVB of `Project Steering/SPEC_REFCV7.md`\n(amendments A1 §6 and A2 §7 — the PI's E1 ruling — included)."),
 ("`59f0d46`, `5de9363` and the current tip `da8400b` (blob comparison, both sides 40-char).",
  "`59f0d46`, `5de9363`, `da8400b` and the current tip `2c510fb` (blob comparison, both sides 40-char)."),
 ("a one-line `not self.training` guard at `:3337` would make it inference-only (not made — a design option).",
  "a one-line `not self.training` guard at `:3337` makes it inference-only — made after the PI's ruling (right)."),
 ("A refcv7 launch must call `declared_vs_built.check(model, args, build_parser(), forbid_kinds=(\"drivort\",))` — `train()` calls it WITHOUT forbidding DrivoR-T",
  "A refcv7 launch must call `declared_vs_built.check(model, args, build_parser(), forbid_kinds=(\"drivort\",))` AND `check_refcv7_required(model, args, tau_file=…/raw/nav_compliance_tau_train.json)` — `train()` calls the first WITHOUT forbidding DrivoR-T"),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(P, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
