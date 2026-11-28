s = open("rk_argv.py.rev1", encoding="utf-8", newline="").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


sub('"""rk_argv.py -- the argv of the KINLAB arm RK (RL\'s argv with exactly `--dagger-label speed_v1` -> `kin_v1` and its own `--out`; SAME bank, SAME dose p = 0.1, SAME init, swap mode). stdlib only, CPU, sub-second. STAGED, NOT RUN.\nCard PREREG_V8F_KINLAB section 2: "RL_s0\'s argv with exactly `--dagger-label speed_v1` -> `kin_v1`, and `--out` changed".',
    '"""rk_argv.py -- REVISION 2. The argv of the KINLAB arm RK: RL\'s argv with EXACTLY THE THREE CHANGES of Amendment 1 section 2 (frozen, sha256 948670d9): `--dagger-label speed_v1` -> `kin_v1`, its own `--out`, and the added execution-only\ntoken pair `--dagger-label-cache <R>/kin_cache/s<K>_kin_v1.npz` appended LAST (after the label). SAME bank, SAME dose p = 0.1, SAME init, swap mode. A fourth difference is refused. stdlib only, CPU, sub-second.\nCard PREREG_V8F_KINLAB section 2 + Amendment 1 section 2.')
sub("EXACTLY two changes: the value of `--dagger-label` (speed_v1 -> kin_v1, same position, same token count) and `--out $R/runs/RK_s{k}/train`. Refuses unless",
    "EXACTLY three changes: the value of `--dagger-label` (speed_v1 -> kin_v1, same position), `--out $R/runs/RK_s{k}/train`, and the two appended tokens `--dagger-label-cache $R/kin_cache/s{k}_kin_v1.npz` (the cache of THIS seed's bank only). Refuses unless")
sub("the written argv differs from RL's in exactly {--dagger-label, --out} (token list diff, in order, same length);", "the written argv differs from RL's in exactly {--dagger-label, --dagger-label-cache, --out} (token list diff, in order: two values changed, two tokens appended at the END);")
sub("exactly the two differences above, nothing else. Refuses a hand-edited third difference.", "exactly the three differences above, nothing else. Refuses a hand-edited fourth difference.")
sub("  Last line: `ZZ RK-RENDER PASS` / `ZZ RK-RENDER FAIL`, or `ZZ RK-VERIFY PASS` / `ZZ RK-VERIFY FAIL`.",
    "  cpu-argv [--seeds 0 1] --out-dir DIR   for each seed k: the SAME argv as `render` would write (built in memory from RL_s{k}, checked; NOTHING is written under runs/), with `--device cuda` -> `cpu` (the ONLY difference), to\n                         DIR/s{k}_argv_cpu.json. This is the argv of the cache precompute (the trainer's --selfcheck-only on the CPU). Refuses if an existing RK_s{k}/argv.json differs from the in-memory argv.\n  Last line: `ZZ RK-RENDER PASS` / `ZZ RK-RENDER FAIL`, `ZZ RK-VERIFY PASS` / `ZZ RK-VERIFY FAIL`, or `ZZ RK-CPUARGV PASS` / `ZZ RK-CPUARGV FAIL`.")
sub('NEW_LABEL = "kin_v1"\n', 'NEW_LABEL = "kin_v1"\nCACHE_FLAG = "--dagger-label-cache"\n\n\ndef cache_path(root, seed):\n    """The cache of seed k\'s bank (Amendment 1): the only path the RK argv may name."""\n    return f"{root}/kin_cache/s{seed}_kin_v1.npz"\n')
sub('''    """RL argv vs RK argv (token lists): [problems]. RK must equal RL token for token except the label value and the --out value."""
    bad = []
    a, b = flags(rk), flags(rl)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if diff != ["--dagger-label", "--out"]:
        bad.append(f"RK differs from RL in {diff}, expected exactly ['--dagger-label', '--out']")''',
    '''    """RL argv vs RK argv (token lists): [problems]. RK must equal RL token for token except the label value, the --out value and the two tokens `--dagger-label-cache <path>` appended at the end (Amendment 1 section 2)."""
    bad = []
    a, b = flags(rk), flags(rl)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if diff != ["--dagger-label", CACHE_FLAG, "--out"]:
        bad.append(f"RK differs from RL in {diff}, expected exactly ['--dagger-label', '{CACHE_FLAG}', '--out']")
    want = cache_path(root, seed)
    if a.get(CACHE_FLAG) != [want]:
        bad.append(f"RK {CACHE_FLAG} reads {a.get(CACHE_FLAG)}, expected exactly [{want!r}]: the cache of THIS seed's bank (never the other seed's, never another path or kind)")
    if rk.count(CACHE_FLAG) != 1:
        bad.append(f"RK names {CACHE_FLAG} {rk.count(CACHE_FLAG)} times, expected once")''')
sub('''    if len(rk) != len(rl):
        bad.append(f"RK has {len(rk)} tokens, RL {len(rl)}: the label swap keeps the token count")
    else:                                      # token by token: the only positions that may differ are the two values
        pos = [i for i in range(len(rl)) if rl[i] != rk[i]]
        iL = rl.index("--dagger-label") + 1 if "--dagger-label" in rl else -1
        iO = rl.index("--out") + 1 if "--out" in rl else -1
        if any(i not in (iL, iO) for i in pos):
            bad.append(f"RK differs from RL at token positions {pos}, allowed only {[iL, iO]}")
    return bad''',
    '''    if len(rk) != len(rl) + 2:
        bad.append(f"RK has {len(rk)} tokens, RL {len(rl)}: the label swap keeps the token count and the cache adds exactly two ({CACHE_FLAG} and its path)")
    else:                                      # token by token: the only positions that may differ are the two values; the last two tokens are the cache pair
        pos = [i for i in range(len(rl)) if rl[i] != rk[i]]
        iL = rl.index("--dagger-label") + 1 if "--dagger-label" in rl else -1
        iO = rl.index("--out") + 1 if "--out" in rl else -1
        if any(i not in (iL, iO) for i in pos):
            bad.append(f"RK differs from RL at token positions {pos}, allowed only {[iL, iO]}")
        if rk[-2:] != [CACHE_FLAG, want]:
            bad.append(f"RK's last two tokens are {rk[-2:]}, expected [{CACHE_FLAG!r}, {want!r}]")
    return bad''')
sub('''    if "--dagger-label" in f:
        new = setf(new, "--dagger-label", [NEW_LABEL])
''', '''    if "--dagger-label" in f:
        new = setf(new, "--dagger-label", [NEW_LABEL])
    if CACHE_FLAG in f:
        bad.append(f"RL_s{seed} already names {CACHE_FLAG} {f[CACHE_FLAG]}: RK adds it, RL has none")
    new = new + [CACHE_FLAG, cache_path(root, seed)]
''')
sub('''def cmd_verify(a):''', '''def cmd_cpu_argv(a):
    ok = True
    for s in a.seeds:
        rlp = f"{R}/runs/RL_s{s}/argv.json"
        b0p = f"{R}/runs/B0_s{s}/argv.json"
        bad = [f"no {p}" for p in (rlp, b0p) if not os.path.isfile(p)]
        if not bad:
            new, bad = build(load(rlp), s, R)
            if flags(load(b0p)).get("--dagger-p") != ["0"]:
                bad.append(f"B0_s{s} --dagger-p is not 0")
            ex = f"{R}/runs/RK_s{s}/argv.json"
            if os.path.isfile(ex) and open(ex, encoding="utf-8").read() != json.dumps({"argv": new}, indent=1):
                bad.append(f"{ex} exists and differs from the argv this builds")
        if not bad:
            if flags(new).get("--device") != ["cuda"]:
                bad.append(f"RK argv --device reads {flags(new).get('--device')}, expected ['cuda'] (the CPU form changes only this)")
            else:
                cpu = setf(new, "--device", ["cpu"])
                pos = [i for i in range(len(new)) if new[i] != cpu[i]]
                if len(cpu) != len(new) or len(pos) != 1 or new[pos[0]] != "cuda" or cpu[pos[0]] != "cpu":
                    bad.append(f"the CPU argv differs from the RK argv at {pos}, not in exactly the --device value")
        if not bad:
            os.makedirs(a.out_dir, exist_ok=True)
            out = f"{a.out_dir}/s{s}_argv_cpu.json"
            text = json.dumps({"argv": cpu}, indent=1)
            open(out, "w", encoding="utf-8").write(text)
            print(f"s{s}: {out} argv_sha256={hashlib.sha256(text.encode()).hexdigest()} (RK argv sha256 {hashlib.sha256(json.dumps({'argv': new}, indent=1).encode()).hexdigest()})")
        for m in bad:
            print("PROBLEM", f"s{s}", m)
        ok &= not bad
    print("ZZ RK-CPUARGV", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_verify(a):''')
sub('''    for name, fn in (("render", cmd_render), ("verify", cmd_verify)):
        r = sp.add_parser(name)
        r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
        r.set_defaults(f=fn)''', '''    for name, fn in (("render", cmd_render), ("verify", cmd_verify), ("cpu-argv", cmd_cpu_argv)):
        r = sp.add_parser(name)
        r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
        if name == "cpu-argv":
            r.add_argument("--out-dir", required=True)
        r.set_defaults(f=fn)''')
open("rk_argv.py", "w", encoding="utf-8", newline="").write(s)
