"""mk_flatdag_helpers.py -- derives flatdag_t1_pullcheck.py from the dag2t1 original (E:/Projects/TanitAD-artifacts/dag2t1/) by EXACT-COUNT substitutions / one marker splice.
Every substitution must occur exactly the stated number of times, or the derivation stops (the mk_dag2_helpers.py discipline).  The diffs against the originals are written to
pullcheck.diff and chain.diff next to this file, so a reviewer reads the change and not two 30 KB files.   python mk_flatdag_helpers.py"""
import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "E:/Projects/TanitAD-artifacts/dag2t1"


def rd(p):
    return open(p, encoding="utf-8", newline="").read()


def apply(t, subs, what):
    for old, new, n in subs:
        c = t.count(old)
        if c != n:
            sys.exit(f"{what}: anchor count {c} != {n} for: {old[:90]!r}")
        t = t.replace(old, new)
    return t


def splice(t, start, end, new, what):
    """Replace t[start:end) where start/end are unique markers (start included, end excluded)."""
    if t.count(start) != 1 or t.count(end) != 1:
        sys.exit(f"{what}: marker count {t.count(start)}/{t.count(end)} for {start[:50]!r} / {end[:50]!r}")
    i, j = t.index(start), t.index(end)
    if j <= i:
        sys.exit(f"{what}: end before start")
    return t[:i] + new + t[j:]


def write(name, text, orig, origname, difname):
    open(os.path.join(HERE, name), "w", encoding="utf-8", newline="").write(text)
    d = "".join(difflib.unified_diff(orig.splitlines(True), text.splitlines(True), "dag2t1/" + origname, name, n=0))
    open(os.path.join(HERE, difname), "w", encoding="utf-8", newline="").write(d)
    print(name, len(text), "bytes;", d.count("\n"), "diff lines")


# ------------------------------------------------------------------------------------------------ pullcheck
IDENTITY_NEW = '''FLAT_CKPT_SHA = {0: "8496934a2ce262c128a3e975da156de2f40e422925893c9979169a4c114493cc", 1: "6fdfc9b02ce87248c5315dd7604d3642a3b39f7ae0f593ec6b6ef5ef8df64c95"}   # v8f-mini-flat/M2b_s{S}_flat.pt (PREREG_V8F_FLATDAG section 1)


def identity_problems(arm: str, seed: int, d: str) -> list[str]:
    """The run's own records must say it is THIS arm and seed.  Called only once the files verify (so a half-pulled file cannot read as a contradiction).
    Identity only: the BUILD CHECK (the cut built, e_code dead, O invariant, the swap reached) is the READER's, because PREREG_V8F_FLATDAG section 4 routes a failing build check to BLOCKED, not to a wait or a STOP."""
    bad = []
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
        sm = json.load(f)
    with open(os.path.join(d, "argv.json"), encoding="utf-8") as f:
        aj = json.load(f)
    if sm.get("arm") != "M2b" or sm.get("seed") != seed:
        bad.append(f"summary.json says arm {sm.get('arm')!r} seed {sm.get('seed')!r}, this is M2b seed {seed}")
    argv = aj.get("argv")
    if not isinstance(argv, list) or not argv:
        return bad + ["argv.json carries no argv list"]
    fl = flags_of(argv)
    if one(fl, "--seed") != str(seed):
        bad.append(f"argv --seed {one(fl, '--seed')!r}, expected {seed}")
    ic = one(fl, "--init-ckpt")
    if ic is None or not norm(ic).endswith(f"M2b_s{seed}_flat/ckpt_last.pt"):
        bad.append(f"argv --init-ckpt {ic!r} is not .../M2b_s{seed}_flat/ckpt_last.pt (PREREG_V8F_FLATDAG section 1: the init is FLAT_s{seed})")
    out = one(fl, "--out")
    if out is None or os.path.basename(norm(out)) != run_name(arm, seed):
        bad.append(f"argv --out {out!r} does not end in {run_name(arm, seed)}: this directory holds another run's record")
    if one(fl, "--o-interface") != "cut":
        bad.append(f"argv --o-interface {fl.get('--o-interface')!r}: FLAT's cut is kept and NAMED exactly once (PREREG_V8F_FLATDAG section 1)")
    p = one(fl, "--dagger-p")
    try:
        pv = float(p)
    except (TypeError, ValueError):
        pv = None
    if pv is None:
        bad.append(f"argv names --dagger-p {p!r}, not exactly one number")
    elif arm == "flatdag0" and pv != 0.0:
        bad.append(f"flatdag0's argv names --dagger-p {p}: that is a FLAT-DAG run")
    elif arm == "flatdag" and not pv > 0.0:
        bad.append(f"flatdag's argv names --dagger-p {p}: that is a FLAT-DAG-0 run")
    if arm == "flatdag0":
        for n in DAG_ONLY:
            if n in fl:
                bad.append(f"flatdag0's argv names {n}: FLAT-DAG-0 names only --dagger-p 0")
    elif arm == "flatdag":
        for n in DAG_ONLY:
            if n not in fl:
                bad.append(f"flatdag's argv does not name {n}")
        bk = one(fl, "--dagger-bank")
        if bk is not None and not norm(bk).rstrip("/").endswith(f"dagger_bank_flat/s{seed}"):
            bad.append(f"flatdag's argv --dagger-bank {bk!r} is not .../dagger_bank_flat/s{seed}: FLAT's OWN bank (generator FLAT_s{seed}) is the only admissible one")
        if one(fl, "--dagger-seed") != str(seed):
            bad.append(f"flatdag's argv --dagger-seed {one(fl, '--dagger-seed')!r}, expected {seed}")
    lv = sm.get("levers") if isinstance(sm.get("levers"), dict) else {}
    if "dagger_p" in lv and pv is not None and float(lv["dagger_p"]) != pv:
        bad.append(f"summary.json built dagger_p {lv['dagger_p']} != the argv's {pv}")
    ics = (lv.get("init_ckpt") or {}).get("sha256") if isinstance(lv.get("init_ckpt"), dict) else None
    if ics is not None and ics != FLAT_CKPT_SHA[seed]:
        bad.append(f"summary.json levers.init_ckpt.sha256 {str(ics)[:16]} is not FLAT_s{seed}'s checkpoint {FLAT_CKPT_SHA[seed][:16]}")
    return bad


'''

pc = rd(f"{SRC}/dag2t1_pullcheck.py")
PC_SUBS = [
    ('"""dag2t1_pullcheck.py -- is one DAgger-2 run\'s pull (DAG2 = dag2, DAG2-0 = dag20), or one seed\'s iteration-2 bank, on the dev box COMPLETE?   (PREREG_V8F_DAGGER2 dev-box T1 chain; CPU only; read-only)\n'
     'DERIVED from dagt1_pullcheck.py by mk_dag2_helpers.py (pullcheck.diff).  Changes: the arm names, the init checkpoint (DAG_s<seed>, not M2b_s<seed>), the iteration-2 bank directory, the pin of the gate.',
     '"""flatdag_t1_pullcheck.py -- is one FLAT-DAG run\'s pull (FLAT-DAG = flatdag, FLAT-DAG-0 = flatdag0), or one seed\'s FLAT bank, on the dev box COMPLETE?   (PREREG_V8F_FLATDAG dev-box T1 chain; CPU only; read-only)\n'
     'DERIVED from dag2t1_pullcheck.py by mk_flatdag_helpers.py (pullcheck.diff).  Changes: the arm names, the 9-name sums (the GT-code oracle eval -tacorc.npz is listed), the init checkpoint (FLAT_s<seed>), --o-interface cut named,\n'
     'FLAT\'s own bank directory (dagger_bank_flat), the pin of the gate.', 1),
    ('ARMS = ("dag2", "dag20")', 'ARMS = ("flatdag", "flatdag0")', 1),
    ('    python dag2t1_pullcheck.py run  <ART> <dag2|dag20> <seed>', '    python flatdag_t1_pullcheck.py run  <ART> <flatdag|flatdag0> <seed>', 1),
    ('    python dagt1_pullcheck.py bank <ART> <seed> [--expect-shards N]', '    python flatdag_t1_pullcheck.py bank <ART> <seed> [--expect-shards N]', 1),
    ('<ART>/M2b_s<S>_<dag|dag0>/   ckpt_last.pt  eval_M2b-<arm>_s<S>.{json,npz}  selfcheck.json summary.json argv.json train_log.jsonl launch_gate_token.json   SHA256SUMS.txt = EXACTLY these 8 names',
     '<ART>/M2b_s<S>_<flatdag|flatdag0>/   ckpt_last.pt  eval_M2b-<arm>_s<S>.{json,npz}  eval_M2b-<arm>_s<S>-tacorc.npz  selfcheck.json summary.json argv.json train_log.jsonl launch_gate_token.json   SHA256SUMS.txt = EXACTLY these 9 names\n'
     '                                 (box 2 pulls eval_*-tacorc.json / .log and eval_*_proj.npz too, but they are not in the sums: the oracle-identity control reads the -tacorc.json when it is present)', 1),
    ('    """THE accepted set of a run\'s SHA256SUMS.txt: 8 names, no fresh-age eval (decision O1)."""\n    tg = eval_tag(arm, seed)\n'
     '    return ["ckpt_last.pt", f"eval_{tg}.json", f"eval_{tg}.npz", "selfcheck.json", "summary.json", "argv.json", "train_log.jsonl", "launch_gate_token.json"]',
     '    """THE accepted set of a run\'s SHA256SUMS.txt: 9 names (box 2\'s chain_v8f_flatdag.sh `sha256sum` line), no fresh-age eval (decision O1)."""\n    tg = eval_tag(arm, seed)\n'
     '    return ["ckpt_last.pt", f"eval_{tg}.json", f"eval_{tg}.npz", f"eval_{tg}-tacorc.npz", "selfcheck.json", "summary.json", "argv.json", "train_log.jsonl", "launch_gate_token.json"]', 1),
    ('pin = os.path.join(art, run_name("dag2", seed), "gate", "pin.txt")', 'pin = os.path.join(art, run_name("flatdag", seed), "gate", "pin.txt")', 1),
    ("({os.path.join(art, run_name('dag2', seed), 'gate', 'pin.txt')})", "({os.path.join(art, run_name('flatdag', seed), 'gate', 'pin.txt')})", 1),
    ('usage = "usage: dag2t1_pullcheck.py run <ART> <dag2|dag20> <seed>  |  bank <ART> <seed> [--expect-shards N]"',
     'usage = "usage: flatdag_t1_pullcheck.py run <ART> <flatdag|flatdag0> <seed>  |  bank <ART> <seed> [--expect-shards N]"', 1),
    ("the box-2 queue's bankdag runs {expect_shards}", "the box-2 queue's bankflat runs {expect_shards}", 1),
    ("PROVENANCE.json + bank.json (sha256-bound, == the gate's declared --dagger-bank-sha256)", "PROVENANCE.json + bank.json (sha256-bound, == the gate's declared --dagger-bank-sha256 of the flatdag run)", 1),
]
pc2 = apply(pc, PC_SUBS, "pullcheck")
pc2 = splice(pc2, "def identity_problems(", "def check(art: str", IDENTITY_NEW, "pullcheck identity")
write("flatdag_t1_pullcheck.py", pc2, pc, "dag2t1_pullcheck.py", "pullcheck.diff")

# the chain (flatdag_t1_chain.sh) is written by hand from dag2_t1_chain.sh's sections 0a-0d; chain.diff is produced by `diff` in the build log
