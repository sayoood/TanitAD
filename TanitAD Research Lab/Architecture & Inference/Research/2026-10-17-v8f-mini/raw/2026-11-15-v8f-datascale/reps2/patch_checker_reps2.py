"""patch_checker_reps2.py: blockreps1/study/dscale10k_check.py -> blockreps2/study/dscale10k_check.py (pod defaults; DS_PATCH_SRC / DS_PATCH_DST override for a local build).
The ONLY changes (each asserted by occurrence count; nothing else -- no bar, no pin, no list, no other arm -- is touched):
 1 ARMS: + UE_s2 (seed 2) and UE_s3 (seed 3): set U, stage A only, steps 'ue', exactly UE_s0 / UE_s1 except the seed.
 2 REP_ARMS = ("UE_s2", "UE_s3"), REP_REF_ARM = "UE_s0", and rep_ref_problems(toks, ref, arm): a REP arm's argv must equal UE_s0.A's token for token except --seed (pinned to the arm's own seed)
   and --out (pinned to UE_s0's --out with "/runs/UE_s0/" replaced by "/runs/<arm>/"). No reference given, a reference with a seed other than 0, a flag more or less, any other value: refused.
3 argv_diff_problems(..., rep_ref=None): a REP arm at stage B is refused; at stage A the problems of rep_ref_problems are appended to the usual B_s0-relative ones (which still run, unchanged).
4 mint / diff / run: a new optional --rep-ref-argv (UE_s0's argv file). mint feeds it to C-ARGV; mint and run BIND it into the token (extra file "rep_ref_argv", like ue_argv for UM), so the runner
  refuses when the reference file changed after the mint.
UM names are not added (UM_s2 -> 'unknown arm'); UE_s4 and above are not added. The seed-0 / seed-1 behaviour of every existing arm is identical: tests/test_reps2_checker.py compares both checkers."""
import os

SRC = os.environ.get("DS_PATCH_SRC", "/root/dscale10k/blockreps1/study/dscale10k_check.py")
DST = os.environ.get("DS_PATCH_DST", "/root/dscale10k/blockreps2/study/dscale10k_check.py")
src = open(SRC, encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (old, src.count(old))
    src = src.replace(old, new)


# 1 + 2 ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
rep('''    "B_s3": dict(set="B"''', '''    "UE_s2": dict(set="U", stage_b=False, seed=2, dagger_seed=None, steps="ue"),      # REPS2: the seed-2 replicate of UE_s0 (PI 2026-11-22 "Yes, two more seeds")
    "UE_s3": dict(set="U", stage_b=False, seed=3, dagger_seed=None, steps="ue"),      # REPS2: the seed-3 replicate of UE_s0
    "B_s3": dict(set="B"''')
rep('MIX_ARMS = ("UM_s0", "UM_s1")\n', '''MIX_ARMS = ("UM_s0", "UM_s1")
REP_ARMS = ("UE_s2", "UE_s3")                       # REPS2: UE_s0's stage-A argv with --seed and --out changed, nothing else (rep_ref_problems)
REP_REF_ARM = "UE_s0"
''')
rep('''def argv_diff_problems(base_toks, toks, arm: str, stage: str = "A", ue_steps: int | None = None) -> list:''',
    '''def rep_ref_problems(toks, ref_toks, arm: str) -> list:
    """REPS2: a REP arm's argv equals the reference (UE_s0.A's argv) token for token except ``--seed`` (= the arm's own seed) and ``--out`` (= the reference's with /runs/UE_s0/ -> /runs/<arm>/)."""
    if ref_toks is None:
        return [f"{arm}: no reference argv (--rep-ref-argv = {REP_REF_ARM}.A.json): the seed-only rule cannot be checked"]
    try:
        a, r = parse_argv(toks), parse_argv(ref_toks)
    except ValueError as e:
        return [f"{arm}: argv or reference does not parse: {e}"]
    bad = []
    if r.get("--seed") != ["0"]:
        bad.append(f"the reference argv has --seed {r.get('--seed')}, not ['0'] ({REP_REF_ARM})")
    ro = (r.get("--out") or [""])[0]
    if f"/runs/{REP_REF_ARM}/" not in ro:
        bad.append(f"the reference --out {ro!r} does not name /runs/{REP_REF_ARM}/")
    pinned = {"--seed": [str(ARMS[arm]["seed"])], "--out": [ro.replace(f"/runs/{REP_REF_ARM}/", f"/runs/{arm}/")]}
    for f in r:
        if f not in a:
            bad.append(f"{arm}: flag {f} of {REP_REF_ARM}'s argv is missing")
    for f in a:
        if f not in r:
            bad.append(f"{arm}: flag {f} {a[f]} is not in {REP_REF_ARM}'s argv (only --seed and --out may differ)")
        elif f in pinned:
            if a[f] != pinned[f]:
                bad.append(f"{arm}: flag {f} reads {a[f]}, pinned {pinned[f]}")
        elif a[f] != r[f]:
            bad.append(f"{arm}: flag {f}: {a[f]} differs from {REP_REF_ARM}'s {r[f]} (only --seed and --out may differ)")
    return bad


def argv_diff_problems(base_toks, toks, arm: str, stage: str = "A", ue_steps: int | None = None, rep_ref=None) -> list:''')
# 3 ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
rep('''        bad.append("a union arm that names the corpus --join-train path: it would read (or overwrite) the corpus join")
    return bad''', '''        bad.append("a union arm that names the corpus --join-train path: it would read (or overwrite) the corpus join")
    if arm in REP_ARMS:
        if stage != "A":
            bad.append(f"{arm} is a stage-A-only arm (stage {stage} refused)")
        bad += rep_ref_problems(toks, rep_ref, arm)
    return bad''')
# 4 ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
rep('''    put("C-ARGV", argv_diff_problems(base, toks, arm, stage, ue))''', '''    rep_ref = None
    if getattr(a, "rep_ref_argv", ""):
        rep_ref = json.load(open(a.rep_ref_argv, encoding="utf-8"))
        rep_ref = rep_ref["argv"] if isinstance(rep_ref, dict) else rep_ref
    put("C-ARGV", argv_diff_problems(base, toks, arm, stage, ue, rep_ref))''')
rep('''"ue_argv": getattr(a, "ue_argv", ""), "b_dry": getattr(a, "b_dry", "")}''', '''"ue_argv": getattr(a, "ue_argv", ""), "b_dry": getattr(a, "b_dry", ""),
             "rep_ref_argv": getattr(a, "rep_ref_argv", "")}''')
rep('''"ue_argv": g("ue_argv"), "b_dry": g("b_dry")}''', '''"ue_argv": g("ue_argv"), "b_dry": g("b_dry"), "rep_ref_argv": g("rep_ref_argv")}''')
rep('''p.add_argument("--b-dry", default="")''', '''p.add_argument("--b-dry", default=""); p.add_argument("--rep-ref-argv", default="")''', n=2)
rep('''p.add_argument("--base-argv", required=True); p.add_argument("--ue-steps", type=int)''', '''p.add_argument("--base-argv", required=True); p.add_argument("--ue-steps", type=int); p.add_argument("--rep-ref-argv", default="")''')
rep('''bad = argv_diff_problems(ld(a.base_argv), ld(a.argv), a.arm, a.stage, a.ue_steps)''', '''bad = argv_diff_problems(ld(a.base_argv), ld(a.argv), a.arm, a.stage, a.ue_steps, ld(a.rep_ref_argv) if a.rep_ref_argv else None)''')
os.makedirs(os.path.dirname(DST), exist_ok=True)
open(DST, "w", encoding="utf-8", newline="\n").write(src)
print("patched", DST, "bytes", len(src.encode("utf-8")))
