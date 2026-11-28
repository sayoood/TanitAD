s = open("test/test_rk_checker.py.rev1", encoding="utf-8", newline="").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


sub('"""Tests of the RK kind of the study checker (dscale10k_check_rk.py = dscale10k_check_ra.py ba5dab8e + patch_rk.py): RH\'s argv + EXACTLY `--dagger-label kin_v1` + its own --out, the RH bank under bank/, dose 0.1 (KINLAB).',
    '"""REVISION 2. Tests of the RK kind of the study checker (dscale10k_check_rk.py = the revision-1 checker 273befbc + patch_rk2.py): RH\'s argv + EXACTLY `--dagger-label kin_v1` + the execution-only pair\n`--dagger-label-cache <R>/kin_cache/s<K>_kin_v1.npz` (Amendment 1) + its own --out, the RH bank under bank/, dose 0.1 (KINLAB). The cache pair is admitted once, for THIS seed only.')
sub('''def rk_toks(r, label=LABEL, p=None, extra=None):
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", f"RK_s{r.seed}", "train"))
    if label is not None:
        t = t + ["--dagger-label", label]''', '''def cache_of(r, seed=None, kind="kin_v1"):
    return os.path.join(r.tmp, "kin_cache", f"s{r.seed if seed is None else seed}_{kind}.npz")


def rk_toks(r, label=LABEL, p=None, extra=None, cache="default"):
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", f"RK_s{r.seed}", "train"))
    if label is not None:
        t = t + ["--dagger-label", label]
    if cache == "default":
        t = t + ["--dagger-label-cache", cache_of(r)]
    elif cache is not None:
        t = t + ["--dagger-label-cache", cache]''')
sub('''    assert sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)) == ["--dagger-label", "--out"] and a["--dagger-label"] == [LABEL] and len(toks) == len(r.rh_toks) + 2''',
    '''    assert sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)) == ["--dagger-label", "--dagger-label-cache", "--out"] and a["--dagger-label"] == [LABEL] and len(toks) == len(r.rh_toks) + 4
    assert a["--dagger-label-cache"] == [cache_of(r)] and toks[-2:] == ["--dagger-label-cache", cache_of(r)]''')
s += '''

# ---- revision 2: the cache token pair (Amendment 1)
def test_rk2_the_cache_pair_is_admitted_once_for_this_seed_only(tmp_path):
    for seed in (0, 1):
        r = Rig(tmp_path / f"c{seed}", seed=seed)
        pr, _ = mint_rk(r, rk_toks(r))
        assert pr == [], pr
        assert NEW.rh_argv_problems(json.load(open(r.base))["argv"], r.tmpl_toks, rk_toks(r), f"RK_s{seed}", r.gen) == []


def test_rk2_mutant_f_the_other_seeds_cache_is_refused(tmp_path):
    r = Rig(tmp_path / "mf0", seed=0)
    pr = _refused(r, rk_toks(r, cache=cache_of(r, seed=1)))
    assert "kin_cache" in "\\n".join(pr) and "s0_kin_v1.npz" in "\\n".join(pr)
    r1 = Rig(tmp_path / "mf1", seed=1)
    pr = _refused(r1, rk_toks(r1, cache=cache_of(r1, seed=0)))
    assert "s1_kin_v1.npz" in "\\n".join(pr)


def test_rk2_mutant_g_a_cache_of_another_kind_or_folder_is_refused(tmp_path):
    r = Rig(tmp_path / "mg2")
    pr = _refused(r, rk_toks(r, cache=cache_of(r, kind="speed_v1")))
    assert "kin_cache" in "\\n".join(pr)
    pr = _refused(r, rk_toks(r, cache=os.path.join(r.tmp, "elsewhere", f"s{r.seed}_kin_v1.npz")), tag="m2")
    assert "kin_cache" in "\\n".join(pr)


def test_rk2_mutant_h_the_cache_pair_missing_or_doubled_is_refused(tmp_path):
    r = Rig(tmp_path / "mh")
    pr = _refused(r, rk_toks(r, cache=None))
    assert "--dagger-label-cache" in "\\n".join(pr)
    t = rk_toks(r) + ["--dagger-label-cache", cache_of(r)]
    try:
        pr, _ = mint_rk(r, t, tag="m2")
    except ValueError as e:                                                                                   # the stock parser refuses a repeated flag outright
        assert "appears twice" in str(e)
    else:
        assert pr, "a doubled cache flag was MINTED"


def test_rk2_the_rh_and_b0_names_may_not_carry_the_cache_flag(tmp_path):
    r = Rig(tmp_path / "mi")
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", "RH_s0", "train")) + ["--dagger-label-cache", cache_of(r)]
    r.patch_heavy(NEW)
    rp, bp = r.argv_file("RH_s0_cache", t), r.argv_file("B0_s0_cache", r.b0_toks)
    a = r.mint_ns("RH_s0", rp, "RH_s0_cache", pair=bp, token=r.gen, selfcheck=r.sc_rh)
    pr = NEW.mint(a)
    assert pr and refused_by(pr, "C-ARGV"), pr
'''
open("test/test_rk_checker.py", "w", encoding="utf-8", newline="").write(s)
