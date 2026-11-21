"""SPEEDAVG pair check: compare the configs the loader BUILDS (missing keys = the V8FMiniConfig defaults, exactly as
`v8f_mini_eval.py` builds `M.V8FMiniConfig(**ck["cfg"])`), not the raw stored dicts. X_s0 was trained before `o_interface` existed,
so its stored cfg lacks the key and its loader builds the default "full"; X_s1 stores "full". Same model; the raw-dict compare refused it."""
import sys

R = sys.argv[1]
p = R + "/stack/scripts/v8f_mini_eval_speedavg.py"
s = open(p, encoding="utf-8", newline="").read()
a = ('    if ck0.get("cfg") != ck1.get("cfg"):\n'
     '        diff = sorted(k for k in set(ck0.get("cfg", {})) | set(ck1.get("cfg", {})) if ck0.get("cfg", {}).get(k) != ck1.get("cfg", {}).get(k))\n'
     '        bad.append(f"the two checkpoints\' configs differ in {diff}")\n')
b = ('    b0, b1 = built_cfg(ck0.get("cfg", {})), built_cfg(ck1.get("cfg", {}))   # the configs the loader BUILDS: a key a checkpoint predates is its default\n'
     '    if b0 != b1:\n'
     '        diff = sorted(k for k in set(b0) | set(b1) if b0.get(k) != b1.get(k))\n'
     '        bad.append(f"the two checkpoints\' configs differ in {diff}")\n')
assert s.count(a) == 1
s = s.replace(a, b)
anchor = "# ------------------------------------------------------------------------------------------------ the pair of checkpoints\n"
assert s.count(anchor) == 1
helper = (anchor +
          "def built_cfg(cfg: dict) -> dict:\n"
          "    \"\"\"The config `v8f_mini_eval.py` builds from a checkpoint (`M.V8FMiniConfig(**ck[\"cfg\"])`): the stored keys over the dataclass\n"
          "    defaults. A key added after a checkpoint was written (e.g. PREREG_V8F_FLAT's `o_interface`, default \"full\" = every earlier arm bit\n"
          "    for bit) is that default, so two runs of one model compare equal however old their checkpoints are.\"\"\"\n"
          "    import dataclasses\n"
          "    from tanitad.models import v8f_mini as M\n"
          "    out = {}\n"
          "    for f in dataclasses.fields(M.V8FMiniConfig):\n"
          "        if f.default is not dataclasses.MISSING:\n"
          "            out[f.name] = f.default\n"
          "        elif f.default_factory is not dataclasses.MISSING:\n"
          "            out[f.name] = f.default_factory()\n"
          "    out.update(cfg or {})\n"
          "    return out\n\n\n")
s = s.replace(anchor, helper)
open(p, "w", encoding="utf-8", newline="").write(s)

t = R + "/stack/tests/test_v8f_speedavg.py"
u = open(t, encoding="utf-8", newline="").read()
e = ('    c3 = ck(m1)\n'
     '    c3["model"].pop("pos")\n'
     '    assert any("keys or shapes" in x for x in SA.pair_violations(ck(m0), c3))\n')
f = (e +
     '    # a checkpoint that PREDATES a config key builds that key\'s default: X_s0 (MIXO s0) has no `o_interface`, X_s1 stores "full"\n'
     '    c4 = ck(m0)\n'
     '    c4["cfg"].pop("o_interface")\n'
     '    c5 = ck(m1)\n'
     '    c5["cfg"]["o_interface"] = "full"\n'
     '    assert SA.pair_violations(c4, c5) == [], "a missing key is its default: the same built model"\n'
     '    c6 = ck(m1)\n'
     '    c6["cfg"]["o_interface"] = "cut"\n'
     '    assert any("configs differ in [\'o_interface\']" in x for x in SA.pair_violations(c4, c6)), "a missing key whose default differs is refused"\n')
assert u.count(e) == 1
open(t, "w", encoding="utf-8", newline="").write(u.replace(e, f))
print("patched")
