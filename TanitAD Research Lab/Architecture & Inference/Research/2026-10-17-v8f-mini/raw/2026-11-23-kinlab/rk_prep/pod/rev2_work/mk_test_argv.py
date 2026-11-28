s = open("test_rk_argv.py.rev1", encoding="utf-8", newline="").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


sub('"""test_rk_argv.py -- KINLAB build. Dev-box test of rk_argv.py on RL_s0\'s REAL argv (read from the pod, pasted below). stdlib only, no pod.\n\nThe well-formed render must differ from RL\'s in exactly the label value and --out (token for token) and be refused/accepted as below; each deliberate regression (a third difference, a wrong label, a wrong bank,\na wrong dose, a moved token) must REFUSE with the stated reason.',
    '"""test_rk_argv.py -- REVISION 2 (Amendment 1). Dev-box test of rk_argv.py on RL_s0\'s REAL argv (read from the pod, pasted below). stdlib only, no pod.\n\nThe well-formed render must differ from RL\'s in exactly the label value, --out and the two appended tokens `--dagger-label-cache <R>/kin_cache/s<K>_kin_v1.npz` (token for token) and be refused/accepted as below; each deliberate regression\n(a FOURTH difference, a wrong label, a wrong bank, a wrong dose, a moved token, the cache path of the OTHER seed\'s bank, the cache missing / moved / duplicated / of another kind) must REFUSE with the stated reason.')
sub('''rk, bad = M.build(RL, 0, "/root/rollhist", BSHA)
check("WELL-FORMED build", not bad and len(rk) == len(RL) and rk[-2:] == ["--dagger-label", "kin_v1"] and "/root/rollhist/runs/RK_s0/train" in rk, bad)
pos = [i for i in range(len(RL)) if RL[i] != rk[i]]
check("WELL-FORMED differs in exactly 2 token positions", len(pos) == 2 and RL[pos[0] - 1] == "--out" and RL[pos[1] - 1] == "--dagger-label", pos)''',
    '''CACHE0 = "/root/rollhist/kin_cache/s0_kin_v1.npz"
rk, bad = M.build(RL, 0, "/root/rollhist", BSHA)
check("WELL-FORMED build", not bad and len(rk) == len(RL) + 2 and rk[-2:] == ["--dagger-label-cache", CACHE0] and rk[-4:-2] == ["--dagger-label", "kin_v1"] and "/root/rollhist/runs/RK_s0/train" in rk, bad)
pos = [i for i in range(len(RL)) if RL[i] != rk[i]]
check("WELL-FORMED differs from RL in exactly 2 token positions plus the 2 appended cache tokens", len(pos) == 2 and RL[pos[0] - 1] == "--out" and RL[pos[1] - 1] == "--dagger-label" and rk[:len(RL)][:pos[0]] == RL[:pos[0]], pos)''')
sub('''check("WELL-FORMED seed-1 form (own init, bank, --out)", not M.build(''', '''rk1, bad1 = M.build([("/root/rollhist/bank/s1" if t == "/root/rollhist/bank/s0" else "/root/dscale10k/runs/UE_s1/A/ckpt_last.pt" if t == "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt" else "/root/rollhist/runs/RL_s1/train" if t == "/root/rollhist/runs/RL_s0/train" else t) for t in RL], 1, "/root/rollhist", BSHA)
check("WELL-FORMED seed-1 cache path is s1's", not bad1 and rk1[-2:] == ["--dagger-label-cache", "/root/rollhist/kin_cache/s1_kin_v1.npz"], rk1[-2:])
check("WELL-FORMED seed-1 form (own init, bank, --out)", not M.build(''')
sub('''    "V12 --out of RA": repl(rk, "/root/rollhist/runs/RK_s0/train", "/root/rollhist/runs/RA_s0/train"),
}''', '''    "V12 --out of RA": repl(rk, "/root/rollhist/runs/RK_s0/train", "/root/rollhist/runs/RA_s0/train"),
    # Amendment 1: the cache token pair
    "V13 cache path names the OTHER seed's bank (s1)": repl(rk, CACHE0, "/root/rollhist/kin_cache/s1_kin_v1.npz"),
    "V14 cache flag missing (RK = RL + two changes only)": rk[:-2],
    "V15 cache path of another kind (speed_v1)": repl(rk, CACHE0, "/root/rollhist/kin_cache/s0_speed_v1.npz"),
    "V16 cache path in another folder": repl(rk, CACHE0, "/tmp/s0_kin_v1.npz"),
    "V17 cache flag not last (a fourth flag after it)": rk + ["--detach-lower"],
    "V18 cache flag duplicated": rk + ["--dagger-label-cache", CACHE0],
    "V19 cache pair moved before the label (label not at its RL position)": rk[:-4] + ["--dagger-label-cache", CACHE0, "--dagger-label", "kin_v1"],
    "V20 cache path has no value": rk[:-1],
    "V21 cache path left as the other seed AND label speed_v1": repl(repl(rk, CACHE0, "/root/rollhist/kin_cache/s1_kin_v1.npz"), "kin_v1", "speed_v1"),
}''')
sub('''    "B7 RL readout is causal_g1": repl(RL, "causal_g1x", "causal_g1"),
}''', '''    "B7 RL readout is causal_g1": repl(RL, "causal_g1x", "causal_g1"),
    "B8 RL already names a cache": RL[:-2] + ["--dagger-label-cache", CACHE0, "--dagger-label", "speed_v1"],
}''')
V10old = '    "V10 label token moved (not last)": [t for t in rk[:-2]][:5] + ["--dagger-label", "kin_v1"] + rk[5:-2],'
sub(V10old, '    "V10 label token moved (not at its RL position)": rk[:5] + ["--dagger-label", "kin_v1"] + rk[5:-4] + ["--dagger-label-cache", CACHE0],')
open("test_rk_argv.py", "w", encoding="utf-8", newline="").write(s)
