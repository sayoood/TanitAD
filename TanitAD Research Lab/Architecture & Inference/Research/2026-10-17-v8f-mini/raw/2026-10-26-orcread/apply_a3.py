"""Bring orc_read.py, its tests and its mutants to PREREG_V8F_ORECOVER Amendment 3 (A3-2026-10-26, before any run): ORC-0 names ONLY
`--ego-perturb-p 0` from the ego-perturb family (the trainer's gate refuses inert law flags at p = 0); ORC names every law flag at its registered
value plus `--ego-perturb-seed <seed>`. Originals kept as *.pre_a3."""
import io
import shutil

H = "C:/Users/Admin/orcread/"


def rep(name, pairs):
    p = H + name
    shutil.copyfile(p, p + ".pre_a3")
    s = io.open(p, encoding="utf-8", newline="").read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:90])
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf-8", newline="").write(s)


rep("orc_read.py", [
    ('PERT_P = {"ORC": 0.3, "ORC0": 0.0}                                         # section 1 / Amendment 1 (named in every argv)',
     'PERT_P = {"ORC": 0.3, "ORC0": 0.0}                                         # section 1 / Amendment 1 (named in every argv)\n'
     '# Amendment 3: the perturbation LAW, named on ORC only (+ --ego-perturb-seed <seed>); ORC-0 names none of them (the trainer refuses inert law flags at p = 0)\n'
     'LAW = {"--ego-perturb-vscale": (0.5, 1.0), "--ego-perturb-vmax": (8.0,), "--ego-perturb-adrop": (1.5,), "--ego-perturb-yawrate-std": (2.0,)}\n'
     'LAW_SEED = "--ego-perturb-seed"'),
    ('''    ok, v = one_value(fl, "--arm")
    if not ok or v != "M2b":
        bad.append(f"--arm {v}, expected M2b")
    return not bad, bad or "argv names the registered fine-tune"''',
     '''    ok, v = one_value(fl, "--arm")
    if not ok or v != "M2b":
        bad.append(f"--arm {v}, expected M2b")
    if arm == "ORC":                                                # A3: the law is named, once, at its registered values
        for name, w in LAW.items():
            occ = fl.get(name) or []
            if len(occ) != 1 or len(occ[0]) != len(w) or not all(same_float(x, y) for x, y in zip(occ[0], w)):
                bad.append(f"{name} {occ}, registered {list(w)} (named once)")
        ok, v = one_value(fl, LAW_SEED)
        if not ok or str(v) != str(seed):
            bad.append(f"{LAW_SEED} {fl.get(LAW_SEED)}, registered {seed}")
    else:                                                           # A3: ORC-0 names no law flag (inert at p = 0, and refused by the trainer's gate)
        for name in list(LAW) + [LAW_SEED]:
            if name in fl:
                bad.append(f"{name} is named on ORC-0 (A3: ORC-0 names only --ego-perturb-p 0)")
    return not bad, bad or "argv names the registered fine-tune"'''),
    ('    drop = {"--out", "--ego-perturb-p"}\n',
     '    drop = {"--out", "--ego-perturb-p", LAW_SEED} | set(LAW)                 # A3: the law is ORC-only; g_argv checks it on each side\n'),
])

rep("test_orc_read.py", [
    ('''    return ["--arm", "M2b", "--seed", str(s), "--out", f"{root}/runs/{arm}_s{s}", "--init-ckpt", f"{root}/m2b/M2b_s{s}/ckpt_last.pt", "--steps", "3000", "--lr", "1e-4",
            "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", str(p), "--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-adrop", "1.5",
            "--ego-perturb-vmax", "8.0", "--ego-perturb-yawrate-std", "2.0"]''',
     '''    law = (["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-vmax", "8.0", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", str(s)]
           if arm == "ORC" else [])                                 # A3: ORC-0 names only --ego-perturb-p 0
    return ["--arm", "M2b", "--seed", str(s), "--out", f"{root}/runs/{arm}_s{s}", "--init-ckpt", f"{root}/m2b/M2b_s{s}/ckpt_last.pt", "--steps", "3000", "--lr", "1e-4",
            "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", str(p)] + law'''),
    ('''GOOD = ["--arm", "M2b", "--seed", "0", "--out", "/o", "--init-ckpt", "/x/M2b_s0/ckpt_last.pt", "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2",
        "--p-gt-end", "0.2", "--ego-perturb-p", "0.3"]''',
     '''GOOD0 = ["--arm", "M2b", "--seed", "0", "--out", "/o", "--init-ckpt", "/x/M2b_s0/ckpt_last.pt", "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2",
         "--p-gt-end", "0.2", "--ego-perturb-p", "0"]
LAW0 = ["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", "0"]
GOOD = [x if x != "0" or i != GOOD0.index("--ego-perturb-p") + 1 else "0.3" for i, x in enumerate(GOOD0)] + LAW0'''),
    ('''    a = list(GOOD)
    a[a.index("--ego-perturb-p") + 1] = "0"
    assert orc.g_argv("t", "ORC0", 0, fake_rec(a))[0] is True and not orc.g_argv("t", "ORC", 0, fake_rec(a))[0]''',
     '''    assert orc.g_argv("t", "ORC0", 0, fake_rec(GOOD0))[0] is True and not orc.g_argv("t", "ORC", 0, fake_rec(GOOD0))[0]


def test_g_argv_a3_law_on_orc_only():
    for i in range(0, len(LAW0), 1):                                # ORC-0 naming ANY law flag (or the law seed) is refused
        if LAW0[i].startswith("--"):
            j = i + 1
            while j < len(LAW0) and not LAW0[j].startswith("--"):
                j += 1
            ok, det = orc.g_argv("t", "ORC0", 0, fake_rec(GOOD0 + LAW0[i:j]))
            assert not ok and any(LAW0[i] in d for d in det), (LAW0[i], det)
            a = GOOD[:len(GOOD) - len(LAW0)] + LAW0[:i] + LAW0[j:]  # ORC missing that law flag is refused
            ok, det = orc.g_argv("t", "ORC", 0, fake_rec(a))
            assert not ok and any(LAW0[i] in d for d in det), (LAW0[i], det)
    for flag, vals in (("--ego-perturb-vscale", ["0.5", "0.9"]), ("--ego-perturb-vmax", ["9.0"]), ("--ego-perturb-adrop", ["1.0"]), ("--ego-perturb-yawrate-std", ["0"]),
                       ("--ego-perturb-seed", ["1"])):
        a = list(GOOD)
        k = a.index(flag)
        a[k + 1:k + 1 + len(vals)] = vals
        ok, det = orc.g_argv("t", "ORC", 0, fake_rec(a))
        assert not ok and any(flag in d for d in det), (flag, det)
    r_orc, r_0 = fake_rec(GOOD), fake_rec(GOOD0)                    # the pair: identical once --out, p and the ORC-only law are set aside
    assert {k: v for k, v in r_orc["flags"].items() if k not in {"--out", "--ego-perturb-p", orc.LAW_SEED} | set(orc.LAW)} == \\
           {k: v for k, v in r_0["flags"].items() if k not in {"--out", "--ego-perturb-p"}}'''),
])

rep("mutate_orc.py", [
    ('# id, what it breaks, edit, targeted -k\nMUTANTS = [\n',
     '# id, what it breaks, edit, targeted -k\nMUTANTS = [\n'
     '    ("M24", "A3: a law flag on ORC-0 is not refused", sub(\'                bad.append(f"{name} is named on ORC-0 (A3: ORC-0 names only --ego-perturb-p 0)")\', \'                pass\'), "a3_law"),\n'
     '    ("M25", "A3: ORC\'s law values are not checked", sub(\'    if arm == "ORC":                                                # A3: the law is named, once, at its registered values\', \'    if False:\'), "a3_law"),\n'),
])
print("ok")
