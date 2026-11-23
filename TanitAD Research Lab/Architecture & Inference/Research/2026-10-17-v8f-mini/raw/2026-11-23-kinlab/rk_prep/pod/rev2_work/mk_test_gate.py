s = open("test_rk_gate_record.py.rev1", encoding="utf-8", newline="").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


sub('"""test_rk_gate_record.py -- KINLAB build. Dev-box test of rk_gate_record.py on planted files (stdlib only, no pod).',
    '"""test_rk_gate_record.py -- REVISION 2 (Amendment 1). Dev-box test of rk_gate_record.py on planted files (stdlib only, no pod).\nAll revision-1 cases stay (K1-K22, planted with the cache). New: K23 the cache file absent, K24 the cache sha mismatch (built levers read another sha), K25 the OTHER seed\'s cache named in the argv,\nK26 the CPU precompute\'s selfcheck missing, K27 its census differs from the bound selfcheck\'s, K28 its levers carry another sha, K29 the argv has no cache flag, K30 the built levers carry no cache sha,\nK31 the built levers carry another cache path, K32 the cache file changed after the mint (verify must refuse), K33 the cache file removed after the mint (verify must refuse), K34 the G-SUITE at 0ca6f49d (the old commit), K35 the record edited to drop its cache binding (verify must refuse).')
sub('COMMIT = "0ca6f49dd076c035c1cd42da5e4e89a6f6961b9a"', 'COMMIT = "1a8b11a1abd50c09636da8b7425e97573cdf1d04"')
sub('DF25 = "df250a0032e56aae89304421918ade5c583d53a0"', 'DF25 = "df250a0032e56aae89304421918ade5c583d53a0"\nOLD0CA = "0ca6f49dd076c035c1cd42da5e4e89a6f6961b9a"')
# argv: the cache flag
sub('''    if mut == "K14":
        argv += ["--dagger-mode", "add"]''', '''    if mut == "K14":
        argv += ["--dagger-mode", "add"]
    oseed = "1" if seed == "0" else "0"
    if mut != "K29":
        argv += ["--dagger-label-cache", f"{R}/kin_cache/s{oseed if mut == 'K25' else seed}_kin_v1.npz"]''')
sub('''    levers = {"dagger": {"dagger_bank_readout": "causal_g1x", "dagger_label": "speed_v1" if mut == "K3" else "kin_v1"}}
    w(f"{R}/runs/{tag}/selfcheck/selfcheck.json", {"declared_vs_built": "PASS", "acc_probe": {"pass": True}, "dagger_probe": probe, "levers": levers})''',
    '''    cpath = f"{R}/kin_cache/s{seed}_kin_v1.npz"
    if mut != "K23":
        w(f"{R}/kin_cache/s{seed}_kin_v1.npz", "planted cache bytes seed " + seed)
    csha = hashlib.sha256(("planted cache bytes seed " + seed).encode()).hexdigest()
    levers = {"dagger": {"dagger_bank_readout": "causal_g1x", "dagger_label": "speed_v1" if mut == "K3" else "kin_v1", "dagger_label_cache": "/elsewhere/c.npz" if mut == "K31" else cpath,
                         "dagger_label_cache_sha256": ("0" * 64 if mut == "K24" else None if mut == "K30" else csha)}}
    w(f"{R}/runs/{tag}/selfcheck/selfcheck.json", {"declared_vs_built": "PASS", "acc_probe": {"pass": True}, "dagger_probe": probe, "levers": levers})
    cprobe = {"kin_status_counts": counts_for(kin + 1, other - 1) if mut == "K27" else counts_for(kin, other)}
    clev = {"dagger": {"dagger_label": "kin_v1", "dagger_label_cache": cpath, "dagger_label_cache_sha256": "1" * 64 if mut == "K28" else csha}}
    if mut != "K26":
        w(f"{R}/kin_cache/s{seed}_selfcheck_cpu.json", {"levers": clev, "dagger_probe": cprobe})''')
sub('''    w(f"{R}/gsuite_result_rk.json", {"new_commit": DF25 if mut == "K8" else COMMIT,''', '''    w(f"{R}/gsuite_result_rk.json", {"new_commit": DF25 if mut == "K8" else OLD0CA if mut == "K34" else COMMIT,''')
sub('''       ("K17", "RK_s0", "rl_label_rows_changed [0]"), ("K18", "RK_s0", "not the RH bank"), ("K19", "RK_s0", "did not print ZZ ROLLHIST-PAIR PASS")]''',
    '''       ("K17", "RK_s0", "rl_label_rows_changed [0]"), ("K18", "RK_s0", "not the RH bank"), ("K19", "RK_s0", "did not print ZZ ROLLHIST-PAIR PASS"),
       ("K23", "RK_s0", "missing label_cache"), ("K24", "RK_s0", "the built levers carry dagger_label_cache_sha256"), ("K25", "RK_s0", "this seed's bank's cache"),
       ("K26", "RK_s0", "missing cache_selfcheck"), ("K27", "RK_s0", "the CPU precompute's kin_status_counts"), ("K28", "RK_s0", "it is not the run that wrote this cache"),
       ("K29", "RK_s0", "--dagger-label-cache is None"), ("K30", "RK_s0", "the built levers carry dagger_label_cache_sha256 ['None']"), ("K31", "RK_s0", "the built levers carry dagger_label_cache ['/elsewhere/c.npz']"),
       ("K34", "RK_s0", "G-SUITE new commit is '0ca6f49dd076c035c1cd42da5e4e89a6f6961b9a'")]''')
# K32 / K33: the cache file after the mint; and the seed-1 mutant of K25 too
sub('''os.remove(f"{R}/done_rk/tree.done")
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K20b tree marker removed after the mint", rc != 0 and "REFUSED" in out)''', '''os.remove(f"{R}/done_rk/tree.done")
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K20b tree marker removed after the mint", rc != 0 and "REFUSED" in out)

# K32 / K33: the label cache file changes / disappears after the mint (verify must refuse); K32b the CPU precompute's selfcheck edited
plant(R)
run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
w(f"{R}/kin_cache/s0_kin_v1.npz", "a different cache")
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K32 cache file changed after the mint", rc != 0 and "REFUSED" in out and "label_cache" in out)
plant(R)
run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
os.remove(f"{R}/kin_cache/s0_kin_v1.npz")
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K33 cache file removed after the mint", rc != 0 and "REFUSED" in out and "label_cache" in out)
plant(R)
run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
w(f"{R}/kin_cache/s0_selfcheck_cpu.json", {"edited": 1})
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K32b CPU precompute selfcheck changed after the mint", rc != 0 and "REFUSED" in out and "cache_selfcheck" in out)
# K35: the record is edited to drop its cache binding (every remaining bound file unchanged): verify must refuse
plant(R)
run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
rp2 = f"{R}/runs/RK_s0/gate/RH_GATE_RECORD.json"
rec2 = json.load(open(rp2))
del rec2["binds"]["label_cache"]
json.dump(rec2, open(rp2, "w"))
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K35 record without its cache binding", rc != 0 and "REFUSED" in out and "does not bind this seed's cache" in out)
# K36-K41: the pre-lock precheck of the cache pair (the chain's step `caches` and the DRY walk): well-formed OK; each absence / change REFUSES for its reason; it writes nothing
plant(R)
before = sorted(os.listdir(f"{R}/runs/RK_s0"))
rc, out = run(R, "precheck", "--tag", "RK_s0")
report("K36 precheck well-formed", rc == 0 and "ZZ RK-CACHE-PRECHECK OK RK_s0" in out and "census 0.6300" in out and sorted(os.listdir(f"{R}/runs/RK_s0")) == before and not os.path.exists(f"{R}/runs/RK_s0/gate"), out.strip().splitlines()[-2:])
plant(R, tag="RK_s1")
rc, out = run(R, "precheck", "--tag", "RK_s1")
report("K36b precheck seed 1", rc == 0)
for m, why in (("K23", "missing label_cache"), ("K26", "missing cache_selfcheck"), ("K24", None), ("K28", "the CPU precompute's selfcheck carries dagger_label_cache_sha256"), ("K6", "CENSUS BAR")):
    plant(R, mut=m)
    if m == "K24":                                                                   # the cache edited AFTER the precompute: the CPU selfcheck still reads the old sha
        w(f"{R}/kin_cache/s0_kin_v1.npz", "edited after the precompute")
        why = "the CPU precompute's selfcheck carries dagger_label_cache_sha256"
    rc, out = run(R, "precheck", "--tag", "RK_s0")
    report(f"K37 precheck {m}", rc != 0 and "REFUSED" in out and why in out, [l for l in out.splitlines() if l.startswith("PROBLEM")][:1])
for what, key, val, why in (("K39 precheck: the CPU selfcheck names another cache path", "dagger_label_cache", "/elsewhere/c.npz", "dagger_label_cache ['/elsewhere/c.npz']"),
                            ("K40 precheck: the CPU selfcheck built label speed_v1", "dagger_label", "speed_v1", "dagger_label ['speed_v1']")):
    plant(R)
    spp = f"{R}/kin_cache/s0_selfcheck_cpu.json"
    scj = json.load(open(spp))
    scj["levers"]["dagger"][key] = val
    json.dump(scj, open(spp, "w"))
    rc, out = run(R, "precheck", "--tag", "RK_s0")
    report(what, rc != 0 and "REFUSED" in out and why in out, [l for l in out.splitlines() if l.startswith("PROBLEM")][:1])
rc, out = run(R, "precheck", "--tag", "RA_s0")
report("K38 precheck refuses another tag", rc != 0 and "REFUSED" in out)
# K25b: seed 1 names seed 0's cache
plant(R, tag="RK_s1")
a1 = json.load(open(f"{R}/runs/RK_s1/argv.json"))
a1["argv"][-1] = f"{R}/kin_cache/s0_kin_v1.npz"
w(f"{R}/runs/RK_s1/argv.json", a1)
rc, out = run(R, "mint", "--tag", "RK_s1", "--readout", "causal_g1x")
report("K25b seed 1 argv names seed 0's cache", rc != 0 and "REFUSED" in out and "this seed's bank's cache" in out)''')
open("test_rk_gate_record.py", "w", encoding="utf-8", newline="").write(s)
