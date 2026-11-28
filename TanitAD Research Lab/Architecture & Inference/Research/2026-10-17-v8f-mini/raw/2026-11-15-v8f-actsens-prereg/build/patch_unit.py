import os, re
os.chdir("F:/TanitAD-scratch/actsens/box2_build/tests")
p = "test_actsens_check.py"
s = open(p, encoding="utf-8", newline="").read()


def sub(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old))
    s = s.replace(old, new)


def cut(start, end, new):
    global s
    i = s.index(start)
    j = s.index(end, i)
    s = s[:i] + new + s[j:]


sub('check("run order is DIV_s0, IDM_s0, DIV_s1, IDM_s1, NULL_s2", CK.RUN_ORDER == ["DIV_s0", "IDM_s0", "DIV_s1", "IDM_s1", "NULL_s2"])',
    'check("run order is IDM_s0, NULL_s2, DIV_s0, IDM_s1, DIV_s1, NULL_s3", CK.RUN_ORDER == ["IDM_s0", "NULL_s2", "DIV_s0", "IDM_s1", "DIV_s1", "NULL_s3"])')
cut('        check(f"{t}: --wm-speed is', '        check(f"{t}: the run names --wm-zcache',
    '''        lev = CK.TABLE[t]["lever"]
        check(f"{t}: lever {lev}: the card's one change from R35, --seed {CK.TABLE[t]['seed']}, init M2b_s{CK.TABLE[t]['init']}, zcache of the same seed as the init",
              (("--wm-idm-gamma" in tk and tk[tk.index("--wm-idm-gamma") + 1:tk.index("--wm-idm-gamma") + 4] == ["1000", "100", "100"]) == (lev == "idm")) and (("--wm-sampler" in tk and g("--wm-sampler") == "divrs") == (lev == "div"))
              and g("--seed") == str(CK.TABLE[t]["seed"]) and f"M2b_s{CK.TABLE[t]['init']}" in g("--init-ckpt") and f"M2b_s{CK.TABLE[t]['init']}" in g("--wm-zcache") and g("--wm-k") == "1")
''')
cut('    i = good.index("--wm-speed") + 1', '    check("a changed seed is refused"',
    '''    i = good.index("--wm-sampler") + 1
    bad = good[:i] + ["uniform"] + good[i + 1:]
    r = CK.audit_line("DIV_s0", " ".join(bad))
    check("a changed --wm-sampler is refused and the first differing token is named", len(r) == 1 and f"token {i} is 'uniform'" in r[0])
''')
sub('check("the NOW argv shipped for KIN is refused", CK.audit_line("DIV_s1", " ".join(CK.tokens("IDM_s1"))) != [])', 'check("the IDM argv shipped for DIV is refused", CK.audit_line("DIV_s1", " ".join(CK.tokens("IDM_s1"))) != [])')
cut('    check("KIN_rep_s2 is DIV_s0 apart', '    check("the frozen argv file hashes',
    '''    check("NULL_s2 is R35_s0 apart from the seed and --out; NULL_s3 is R35_s1 apart from the seed and --out", [(a, b) for a, b in zip(CK.tokens("NULL_s2"), CK.tokens("IDM_s0")) if a != b][:2] == [("2", "0"), (CK.out_decl("NULL_s2"), CK.out_decl("IDM_s0"))]
          and len(CK.tokens("NULL_s2")) + 3 == len(CK.tokens("IDM_s0")) and len(CK.tokens("NULL_s3")) + 3 == len(CK.tokens("IDM_s1")))
''')
# speed mutations -> lever mutations
i = s.index('        "speed lever absent"')
j = s.index('        "windows do not return the speed"')
j = s.index("\n", j) + 1
s = s[:i] + '''        "idm readback absent": lambda s: s["levers"].pop("wm_idm_gamma_readback"),
        "idm readback built at the old weights": lambda s: s["levers"]["wm_idm_gamma_readback"]["built"].__setitem__(0, 100.0),
        "idm readback built level 2 moved": lambda s: s["levers"]["wm_idm_gamma_readback"]["built"].__setitem__(1, 1000.0),
        "idm readback declared": lambda s: s["levers"]["wm_idm_gamma_readback"].__setitem__("declared", [100.0, 100.0, 100.0]),
        "idm readback on a non-idm arm": lambda s: s["levers"].__setitem__("wm_idm_gamma_readback", {"built": [1000.0, 100.0, 100.0]}) if "wm_idm_gamma_readback" not in s["levers"] else (_ for _ in ()).throw(KeyError()),
        "sampler record absent on a div arm": lambda s: s["levers"].pop("wm_sampler"),
        "sampler built uniform on a div arm": lambda s: s["levers"]["wm_sampler"].__setitem__("kind", "uniform"),
        "sampler edge drifted": lambda s: s["levers"]["wm_sampler"]["yaw_edges"].__setitem__(3, 0.25),
        "sampler cap drifted": lambda s: s["levers"]["wm_sampler"].__setitem__("cap_mult", 30.0),
        "sampler cell counts do not sum": lambda s: s["levers"]["wm_sampler"].__setitem__("cell_counts", [1] * 25),
        "sampler control 1 failed": lambda s: s["wm_sampler_probe"].__setitem__("pass", False),
        "sampler probe absent on a div arm": lambda s: s.pop("wm_sampler_probe"),
        "sampler probe on a non-div arm": lambda s: s.__setitem__("wm_sampler_probe", {"pass": True}) if "wm_sampler_probe" not in s else (_ for _ in ()).throw(KeyError()),
        "WMAux.gamma moved": lambda s: s["levers"].__setitem__("wm_idm_gamma", 1000.0),
        "n_windows moved": lambda s: s["info"].__setitem__("n_windows", 40000),
        "window list sha": lambda s: s["info"].__setitem__("train_window_list_sha256", "0" * 64),
        "C3 an unweighted term moved": lambda s: s["live"]["terms"].__setitem__("wm/pred/L2", s["live"]["terms"]["wm/pred/L2"] * 1.001) if "wm_idm_gamma_readback" in s["levers"] else (_ for _ in ()).throw(KeyError()),
        "C3 level/L1 not R35 + 900 x idm": lambda s: s["live"]["terms"].__setitem__("wm/level/L1", s["live"]["terms"]["wm/level/L1"] + 1.0) if "wm_idm_gamma_readback" in s["levers"] else (_ for _ in ()).throw(KeyError()),
''' + s[j:]
sub('"history K": lambda s: s["levers"].__setitem__("wm_k", 1),', '"history K": lambda s: s["levers"].__setitem__("wm_k", 3),')
sub('"WM parameter count": lambda s: s["levers"]["wm_only"].__setitem__("n_wm", 2143298),', '"WM parameter count": lambda s: s["levers"]["wm_only"].__setitem__("n_wm", 2143298 + 96),')
sub("check(\"the planted mutations ran on every run tag (>= 55 each)\", n_ref >= 55 * len(runnable))", "check(\"the planted mutations ran on every run tag (>= 40 each)\", n_ref >= 40 * len(runnable))")
cut("    # a mode-specific mutation: the KIN ramp flat", "    # ---- weights identity (pure)", "")
# weights identity
cut('    for i in range(3):\n        b[f"wm.stack.spd.{i}.weight"]', '    for k in [k for k in a if not k.startswith("wm.")]:',
    '''    ev, pr = CK.weights_identity(a, b)
    check("weights identity: non-WM equal, WM moved -> clean", pr == [] and ev["n_nonwm_equal"] == ev["n_nonwm"] and ev["n_wm_tensors"] > 0)
''')
i = s.index('    check("weights identity: speed channels still zero is refused"')
j = s.index('    check("weights identity: no WM tensor at all is refused"')
s = s[:i] + s[j:]
s = s.replace('    # ---- tree needles', '''    # ---- C2 (gamma from the log) and C5 (Adam state), both with a positive and a negative control
    def rows(g):
        out = []
        for st_ in (50, 100):
            r = {}
            for L in (1, 2, 3):
                r.update({f"wm/pred/L{L}": 0.2, f"wm/sigreg/L{L}": 3.0, f"wm/rollout/L{L}": 0.25, f"wm/idm/L{L}": 0.11 * L})
                if L >= 2:
                    r[f"wm/act_sigreg/L{L}"] = 8.0
                r[f"wm/level/L{L}"] = 0.2 + 0.25 + 0.24 + g[L - 1] * 0.11 * L + (0.04 if L >= 2 else 0.0)
            out.append(r)
        return out
    for tag_, g, okk in (("IDM_s0", [1000.0, 100.0, 100.0], True), ("IDM_s0", [100.0, 100.0, 100.0], False), ("NULL_s2", [100.0, 100.0, 100.0], True), ("DIV_s0", [1000.0, 100.0, 100.0], False), ("NULL_s3", [100.0, 100.0, 300.0], False)):
        lp = os.path.join(tmp, "gam.jsonl")
        open(lp, "w", encoding="utf-8").write("\\n".join(json.dumps(r) for r in rows(g)) + "\\n")
        check(f"C2 gamma-from-log {tag_} built {g} -> {'clean' if okk else 'refused'}", (CK.gamma_log_problems(tag_, lp)[1] == []) == okk)
    open(os.path.join(tmp, "gam0.jsonl"), "w").write("{}\\n")
    check("C2 gamma-from-log: a log without loss terms is refused", CK.gamma_log_problems("IDM_s0", os.path.join(tmp, "gam0.jsonl"))[1] != [])
    good_opt = {"opt": {"param_groups": [{"params": [0, 1, 2]}], "state": {0: {}, 1: {}, 2: {}}}}
    check("C5 Adam state: every parameter has state -> clean", CK.adam_state_problems(good_opt)[1] == [])
    check("C5 Adam state: a parameter without state is refused", CK.adam_state_problems({"opt": {"param_groups": [{"params": [0, 1, 2]}], "state": {0: {}, 1: {}}}})[1] != [])
    check("C5 Adam state: no optimiser groups is refused", CK.adam_state_problems({"opt": {}})[1] != [])
    # ---- tree needles''', 1)
open(p, "w", encoding="utf-8", newline="").write(s)
print("unit patched")
