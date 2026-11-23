p = "rk_argv.py"
s = open(p, encoding="utf-8", newline="").read()
def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a)); s = s.replace(a, b)
sub("  Last line: `ZZ RK-RENDER PASS` / `ZZ RK-RENDER FAIL`, `ZZ RK-VERIFY PASS` / `ZZ RK-VERIFY FAIL`, or `ZZ RK-CPUARGV PASS` / `ZZ RK-CPUARGV FAIL`.",
    "  show [--seeds 0 1]     like `render` but WRITES NOTHING: builds each RK argv in memory from RL_s{k}, checks it (and an existing RK_s{k}/argv.json, if any, must be identical) and prints its three differences from RL. Used by the chain's\n                         pre-lock validation and by DRY=1.\n  Last line: `ZZ RK-RENDER PASS` / `ZZ RK-RENDER FAIL`, `ZZ RK-VERIFY PASS` / `ZZ RK-VERIFY FAIL`, `ZZ RK-CPUARGV PASS` / `ZZ RK-CPUARGV FAIL`, or `ZZ RK-SHOW PASS` / `ZZ RK-SHOW FAIL`.")
sub("The stock pair check is run by rk_arm.sh on the filled argv minus the two label tokens, against B0, as for RL.", "The stock pair check is run by rk_arm.sh on the filled argv minus the four trailing tokens (label pair and cache pair), against B0.")
sub("def cmd_verify(a):", '''def cmd_show(a):
    ok = True
    for s in a.seeds:
        rlp, b0p = f"{R}/runs/RL_s{s}/argv.json", f"{R}/runs/B0_s{s}/argv.json"
        bad = [f"no {p}" for p in (rlp, b0p) if not os.path.isfile(p)]
        if not bad:
            rl = load(rlp)
            new, bad = build(rl, s, R)
            if flags(load(b0p)).get("--dagger-p") != ["0"]:
                bad.append(f"B0_s{s} --dagger-p is not 0")
            ex = f"{R}/runs/RK_s{s}/argv.json"
            if os.path.isfile(ex) and open(ex, encoding="utf-8").read() != json.dumps({"argv": new}, indent=1):
                bad.append(f"{ex} exists and differs from the argv this builds")
        if not bad:
            pos = [i for i in range(len(rl)) if rl[i] != new[i]]
            print(f"s{s}: RK argv = RL argv with {[(rl[i - 1], rl[i], new[i]) for i in pos]} and the appended {new[len(rl):]}; {len(new)} tokens; sha256 {hashlib.sha256(json.dumps({'argv': new}, indent=1).encode()).hexdigest()}")
        for m in bad:
            print("PROBLEM", f"s{s}", m)
        ok &= not bad
    print("ZZ RK-SHOW", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_verify(a):''')
sub('''("cpu-argv", cmd_cpu_argv)):''', '''("cpu-argv", cmd_cpu_argv), ("show", cmd_show)):''')
open(p, "w", encoding="utf-8", newline="").write(s)
