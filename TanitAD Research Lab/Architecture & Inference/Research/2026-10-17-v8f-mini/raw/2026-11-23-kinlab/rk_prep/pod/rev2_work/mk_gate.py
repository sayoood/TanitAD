s = open("rk_gate_record.py.rev1", encoding="utf-8", newline="").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


sub('"""rk_gate_record.py -- the RK chain\'s launch-gate record for ONE RK stage-B arm (RK_s0 | RK_s1), KINLAB (card PREREG_V8F_KINLAB, commit 0ca6f49d).',
    '"""rk_gate_record.py -- REVISION 2. The RK chain\'s launch-gate record for ONE RK stage-B arm (RK_s0 | RK_s1), KINLAB (card PREREG_V8F_KINLAB + Amendment 1, commit 1a8b11a1).\nRevision 2 adds, and only adds (the rev1 checks all stay): the LABEL-CACHE checks of Amendment 1 section 4 --\n  * the argv\'s `--dagger-label-cache` equals exactly $R/kin_cache/s<K>_kin_v1.npz (this seed\'s bank, never the other seed\'s);\n  * that file exists and is BOUND (sha256 re-hashed at `verify`), as is the precompute\'s selfcheck $R/kin_cache/s<K>_selfcheck_cpu.json;\n  * the selfcheck\'s BUILT levers `dagger_label_cache_sha256` equal the cache file\'s sha256 NOW (and `dagger_label_cache` equals the argv path); the CPU precompute\'s levers carry the same sha;\n  * the full-bank census read from the bound selfcheck equals the CPU precompute\'s census (same counts), so the bar of 0.5 is read from the bank the cache was computed on;\n  * the G-SUITE result and the record\'s commit are 1a8b11a1.')
sub('COMMIT = "0ca6f49dd076c035c1cd42da5e4e89a6f6961b9a"                                      # the KINLAB commit: 40 hex or this file refuses to run',
    'COMMIT = "1a8b11a1abd50c09636da8b7425e97573cdf1d04"                                      # the commit of record (KINLAB + the label cache): 40 hex or this file refuses to run')
sub('''    if tag.startswith("RK"):
        f["bank_json"]''', '''    if tag.startswith("RK"):
        f["label_cache"] = cache_path(seed)
        f["cache_selfcheck"] = f"{R}/kin_cache/s{seed}_selfcheck_cpu.json"
        f["bank_json"]''')
sub('''def files(tag):''', '''def cache_path(seed):
    return f"{R}/kin_cache/s{seed}_kin_v1.npz"


def files(tag):''')
sub('''    share, n_kin, n_ok, counts = census(sc)
    if share is None:''', '''    # -- Amendment 1 section 4: the label cache. The file is bound (a missing one refused above, as "missing label_cache"); the built levers must read ITS sha256 now
    want_cache = cache_path(seed)
    if m.get("--dagger-label-cache") != want_cache:
        bad.append(f"RK argv --dagger-label-cache is {m.get('--dagger-label-cache')!r}, not exactly {want_cache!r} (this seed's bank's cache)")
    csha = bind["label_cache"]["sha256"]
    lev_sha = [str(v) for v in find_key(sc.get("levers", {}), "dagger_label_cache_sha256")]
    if set(lev_sha) != {csha}:
        bad.append(f"the built levers carry dagger_label_cache_sha256 {sorted(set(lev_sha))}, not exactly the sha256 of {want_cache} now ({csha}): the cache read is not the file on disk")
    lev_path = [str(v) for v in find_key(sc.get("levers", {}), "dagger_label_cache")]
    if set(lev_path) != {want_cache}:
        bad.append(f"the built levers carry dagger_label_cache {sorted(set(lev_path))}, not exactly [{want_cache!r}]")
    csc = json.load(open(F["cache_selfcheck"]))
    cl = [str(v) for v in find_key(csc.get("levers", {}), "dagger_label_cache_sha256")]
    if set(cl) != {csha}:
        bad.append(f"the CPU precompute's selfcheck carries dagger_label_cache_sha256 {sorted(set(cl))}, not the cache file's {csha}: it is not the run that wrote this cache")
    share_c, _, n_ok_c, counts_c = census(csc)
    share, n_kin, n_ok, counts = census(sc)
    if counts_c != counts or share_c != share:
        bad.append(f"census: the CPU precompute's kin_status_counts {counts_c!r} differ from the bound selfcheck's {counts!r}")
    if share is None:''')
sub('"commit": COMMIT, "label": LABEL, "pass": not bad, "problems": bad, "binds": bind, "census": cen,', '"commit": COMMIT, "label": LABEL, "pass": not bad, "problems": bad, "binds": bind, "census": cen, "label_cache_sha256": (bind.get("label_cache") or {}).get("sha256"),')
sub('''    for k in bad:
        print("PROBLEM changed or missing since the mint:", k)''', '''    cp = (rec.get("binds", {}).get("label_cache") or {}).get("path")
    if not cp or cp != cache_path(a.tag.split("_s")[1]) or rec.get("label_cache_sha256") != (rec.get("binds", {}).get("label_cache") or {}).get("sha256"):
        bad.append("label cache: the record does not bind this seed's cache")
    for k in bad:
        print("PROBLEM changed or missing since the mint:", k)''')
open("rk_gate_record.py", "w", encoding="utf-8", newline="").write(s)

# ---- precheck (added after the first rev2 build): the pre-lock check of the cache pair, used by the chain and the DRY walk; writes nothing
s = open("rk_gate_record.py", encoding="utf-8", newline="").read()
sub('''def finish(tag, bad, bind, ro, cen):''', '''def precheck(a):
    """Pre-lock check of ONE seed's label cache (Amendment 1 section 3): the file and the CPU precompute's selfcheck exist; the selfcheck's built levers read THIS file's sha256 now and the cache's path; its census
    (the full bank's kin_status_counts) clears the bar. Writes nothing. The chain runs it before the lock; the gate record (mint) repeats every check on the selfcheck of the arm under the lock."""
    tag = a.tag
    if tag not in ("RK_s0", "RK_s1"):
        print("PROBLEM precheck serves RK_s0 / RK_s1 only")
        print("ZZ RK-CACHE-PRECHECK REFUSED", tag)
        return 1
    seed = tag.split("_s")[1]
    p, sp = cache_path(seed), f"{R}/kin_cache/s{seed}_selfcheck_cpu.json"
    bad = [f"missing {k}: {q}" for k, q in (("label_cache", p), ("cache_selfcheck", sp)) if not os.path.isfile(q)]
    if not bad:
        csha = sha(p)
        csc = json.load(open(sp))
        cl = [str(v) for v in find_key(csc.get("levers", {}), "dagger_label_cache_sha256")]
        if set(cl) != {csha}:
            bad.append(f"the CPU precompute's selfcheck carries dagger_label_cache_sha256 {sorted(set(cl))}, not the cache file's {csha} now: the file changed since the run that wrote it")
        lp = [str(v) for v in find_key(csc.get("levers", {}), "dagger_label_cache")]
        if set(lp) != {p}:
            bad.append(f"the CPU precompute's selfcheck carries dagger_label_cache {sorted(set(lp))}, not exactly [{p!r}]")
        lv = set(map(str, find_key(csc.get("levers", {}), "dagger_label")))
        if lv != {LABEL}:
            bad.append(f"the CPU precompute's built levers carry dagger_label {sorted(lv)}, not exactly [{LABEL!r}]")
        share, n_kin, n_ok, counts = census(csc)
        if share is None:
            bad.append(f"the CPU precompute's selfcheck has no usable kin_status_counts ({counts!r})")
        elif share < CENSUS_BAR:
            bad.append(f"CENSUS BAR: the relabelled share of ok records is {share:.4f} ({n_kin} of {n_ok}), below {CENSUS_BAR} (card section 4: KINLAB-BLOCKED, never tuned)")
        else:
            print(f"cache {p} sha256 {csha}; census {share:.4f} ({n_kin} of {n_ok}), bar {CENSUS_BAR}")
    for q in bad:
        print("PROBLEM", q)
    print("ZZ RK-CACHE-PRECHECK", "OK" if not bad else "REFUSED", tag)
    return 0 if not bad else 1


def finish(tag, bad, bind, ro, cen):''')
sub('''    v = sp.add_parser("verify")
    v.add_argument("--tag", required=True)
    v.set_defaults(f=verify)''', '''    v = sp.add_parser("verify")
    v.add_argument("--tag", required=True)
    v.set_defaults(f=verify)
    pc = sp.add_parser("precheck")
    pc.add_argument("--tag", required=True)
    pc.set_defaults(f=precheck)''')
sub("  verify --tag RK_s0                         re-hashes", "  precheck --tag RK_s0                       (revision 2) the pre-lock check of the label cache + the CPU precompute's selfcheck; writes nothing\n  verify --tag RK_s0                         re-hashes")
open("rk_gate_record.py", "w", encoding="utf-8", newline="").write(s)
