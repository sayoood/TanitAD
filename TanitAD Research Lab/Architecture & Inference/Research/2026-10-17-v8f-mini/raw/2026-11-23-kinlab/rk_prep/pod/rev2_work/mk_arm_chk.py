def patch(src, dst, subs):
    s = open(src, encoding="utf-8", newline="").read()
    for a, b, n in subs:
        assert s.count(a) == n, (src, a[:70], s.count(a))
        s = s.replace(a, b)
    open(dst, "w", encoding="utf-8", newline="").write(s)


# ---- rk_arm.sh
patch("rk_arm.sh.rev1", "rk_arm.sh", [
    ("# rk_arm.sh TAG -- ONE stage-B arm of KINLAB (TAG = RK_s0 | RK_s1; card PREREG_V8F_KINLAB).",
     "# rk_arm.sh TAG -- REVISION 2 (commit 1a8b11a1, Amendment 1: the label cache). ONE stage-B arm of KINLAB (TAG = RK_s0 | RK_s1; card PREREG_V8F_KINLAB).\n"
     "# Revision 2 differences from revision 1, and only these: the argv carries the execution-only tokens `--dagger-label-cache $R/kin_cache/s<K>_kin_v1.npz` AFTER the label pair (rk_argv.py verify admits exactly three differences from RL);\n"
     "#   the pair check runs on the argv minus BOTH trailing pairs (label and cache); the cache file and the CPU precompute's selfcheck must exist BEFORE anything runs (written by the trainer's --selfcheck-only on the CPU, outside the lock);\n"
     "#   the selfcheck under the lock READS the cache (a hit); rk_gate_record.py binds the file's sha256 and requires the built levers to read it; shared pod files (rh_selfcheck.py) are read from $R, not from the kit folder.", 1),
    ("""[ -s $D/argv.json ] || afail "no $D/argv.json (rk_argv.py render first)\"""",
     """[ -s $D/argv.json ] || afail "no $D/argv.json (rk_argv.py render first)"
[ -s $KC/s${seed}_kin_v1.npz ] && [ -s $KC/s${seed}_selfcheck_cpu.json ] || afail "no label cache $KC/s${seed}_kin_v1.npz / $KC/s${seed}_selfcheck_cpu.json: the CPU precompute (Amendment 1 section 3) must run before the lock\"""", 1),
    ("""assert t[i + 1] == "kin_v1" and i + 2 == len(t)          # exactly the last two tokens""",
     """assert t[i + 1] == "kin_v1" and t[i + 2] == "--dagger-label-cache" and i + 4 == len(t)          # the label pair, then exactly the cache pair, as the LAST four tokens""", 1),
    ("RH_TREE=$TREE $PY $HERE/rh_selfcheck.py", "RH_TREE=$TREE $PY $R/rh_selfcheck.py", 1),
    ("`RK argv minus the last two tokens (--dagger-label kin_v1) vs the EXISTING B0_s$seed argv`", "`RK argv minus the last four tokens (--dagger-label kin_v1 --dagger-label-cache <path>) vs the EXISTING B0_s$seed argv`", 1),
])

# ---- chk_arm_rk.sh
patch("chk_arm_rk.sh.rev1", "chk_arm_rk.sh", [
    ("# chk_arm_rk.sh <mint|run> TAG   (TAG = RK_s0 | RK_s1)   Pod side.", "# chk_arm_rk.sh <mint|run> TAG   (TAG = RK_s0 | RK_s1)   Pod side. REVISION 2 (Amendment 1): the checker admits the cache token pair; it lives in the KIT folder ($HERE/dscale10k_check_rk.py), rh_evidence.py is read from $R.\n# (revision 1:)", 1),
    ("CHK=${DS_CHK:-$R/study/dscale10k_check_rk.py}", "CHK=${DS_CHK:-$HERE/dscale10k_check_rk.py}", 1),
    ("$PY $HERE/rh_evidence.py", "$PY $R/rh_evidence.py", 1),
])
print("ok")
