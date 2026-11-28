"""patch_checker_reps1.py (pod): blockum/study/dscale10k_check.py -> blockreps1/study/dscale10k_check.py, the ONLY changes (each asserted by occurrence count):
 1 ARMS: + UE_s1, UM_s1 (seed 1, set U, stage A only, steps 'ue'), exactly as UE_s0 / UM_s0 except the seed.
 2 MIX_ARMS = (UM_s0, UM_s1): every `arm == "UM_s0"` branch (argv_diff pinned TAC flags and allowed additions, required_checks, mint C-ARGV-MIX / G-LIVE) now reads `arm in MIX_ARMS`.
 3 mix_argv_problems: the expected --out is derived from the UE argv's --out ('/runs/UE_' -> '/runs/UM_'), which equals UM_OUT for the s0 pair and runs/UM_s1/A for the s1 pair.
 4 mix_regression takes `arm` (default UM_s0); G-RM2 mutates the seed to the OTHER value (1 for s0, 0 for s1), named G-RM2_seed_<value>.
 5 mint passes arm to mix_regression.
Nothing else (no bar, no pin, no list) is touched. The seed-0 behaviour is identical: python check_parity (reps1) compares both."""
import sys

src = open("/root/dscale10k/blockum/study/dscale10k_check.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (old, src.count(old))
    src = src.replace(old, new)


rep('''    "B_s3": dict(set="B"''', '''    "UE_s1": dict(set="U", stage_b=False, seed=1, dagger_seed=None, steps="ue"),      # REPS1: the seed-1 replicate of UE_s0 (Amendment UM-A2 DRAFT)
    "UM_s1": dict(set="U", stage_b=False, seed=1, dagger_seed=None, steps="ue"),      # REPS1: the seed-1 replicate of UM_s0
    "B_s3": dict(set="B"''')
rep('NCLIPS = {"B"', 'MIX_ARMS = ("UM_s0", "UM_s1")\nNCLIPS = {"B"')
rep('    if arm == "UM_s0":\n        allow_added |= set(TAC_FLAGS)', '    if arm in MIX_ARMS:\n        allow_added |= set(TAC_FLAGS)')
rep('    if arm == "UM_s0":\n        pinned.update', '    if arm in MIX_ARMS:\n        pinned.update')
rep('''    if a.get("--out") != [UM_OUT]:
        bad.append(f"--out reads {a.get('--out')}, expected {[UM_OUT]}")''', '''    exp_out = [(u.get("--out") or [""])[0].replace("/runs/UE_", "/runs/UM_")]
    if a.get("--out") != exp_out or exp_out == u.get("--out"):
        bad.append(f"--out reads {a.get('--out')}, expected {exp_out}")''')
rep('def mix_regression(base_toks, ue_toks, toks, raw: bytes, train_ids, b_ids, ue_steps) -> list:', 'def mix_regression(base_toks, ue_toks, toks, raw: bytes, train_ids, b_ids, ue_steps, arm="UM_s0") -> list:')
rep('argv_diff_problems(base_toks, t, "UM_s0", "A", ue_steps)', 'argv_diff_problems(base_toks, t, arm, "A", ue_steps)')
rep('("G-RM2_seed_1", full(setf(toks, "--seed", "1")))', '(f"G-RM2_seed_{ARMS[arm][\'seed\'] ^ 1}", full(setf(toks, "--seed", str(ARMS[arm]["seed"] ^ 1))))')
rep('(REQUIRED_MIX if arm == "UM_s0" else ())', '(REQUIRED_MIX if arm in MIX_ARMS else ())')
rep('    if arm == "UM_s0":\n        ue_t = json.load', '    if arm in MIX_ARMS:\n        ue_t = json.load')
rep('bdry.get("train_clips") or [], ue)', 'bdry.get("train_clips") or [], ue, arm)')
rep('    if arm == "UM_s0":\n        put("G-LIVE"', '    if arm in MIX_ARMS:\n        put("G-LIVE"')
assert 'arm == "UM_s0"' not in src
open("/root/dscale10k/blockreps1/study/dscale10k_check.py", "w", encoding="utf-8", newline="\n").write(src)
print("patched; remaining literal UM_s0 sites:", [i + 1 for i, l in enumerate(src.split("\n")) if "UM_s0" in l and not l.strip().startswith(("#", '"""'))])
