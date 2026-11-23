p = "rev2_work/mk_test_gate.py"
s = open(p, encoding="utf-8").read()
old = '''rc, out = run(R, "precheck", "--tag", "RA_s0")'''
new = '''for what, key, val, why in (("K39 precheck: the CPU selfcheck names another cache path", "dagger_label_cache", "/elsewhere/c.npz", "dagger_label_cache ['/elsewhere/c.npz']"),
                            ("K40 precheck: the CPU selfcheck built label speed_v1", "dagger_label", "speed_v1", "dagger_label ['speed_v1']")):
    plant(R)
    spp = f"{R}/kin_cache/s0_selfcheck_cpu.json"
    scj = json.load(open(spp))
    scj["levers"]["dagger"][key] = val
    json.dump(scj, open(spp, "w"))
    rc, out = run(R, "precheck", "--tag", "RK_s0")
    report(what, rc != 0 and "REFUSED" in out and why in out, [l for l in out.splitlines() if l.startswith("PROBLEM")][:1])
rc, out = run(R, "precheck", "--tag", "RA_s0")'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
