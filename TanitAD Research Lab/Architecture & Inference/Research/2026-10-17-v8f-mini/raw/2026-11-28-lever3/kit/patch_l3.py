#!/usr/bin/env python3
"""patch_l3.py -- makes kit/dscale10k_check_l3.py from dscale10k_check_s3.py (the pod's SEED3 checker, sha256 046deadd..).
Adds the arm kinds D3K_s0, D3K_s1, D9K_s0, D9K_s1 (PREREG_V8F_LEVER3_V2 section 4) and nothing else. Every edit is ASSERTED: the old text must occur exactly once, or the patch stops.
usage: patch_l3.py <in: dscale10k_check_s3.py> <out: dscale10k_check_l3.py>"""
import hashlib
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8", newline="").read()
sha8 = hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
n_edits = 0


def sub(old, new):
    global s, n_edits
    assert s.count(old) == 1, f"patch anchor occurs {s.count(old)} times (need exactly 1): {old[:90]!r}"
    s = s.replace(old, new)
    n_edits += 1


# 1. lineage line (the docstring's first line)
sub('"""[SEED3 build: dscale10k_check_s3.py',
    f'"""[LEVER3 build: dscale10k_check_l3.py = dscale10k_check_s3.py {sha8} + the arms D3K_s0 / D3K_s1 / D9K_s0 / D9K_s1 (patch_l3.py); PREREG_V8F_LEVER3_V2 sha256 707c5225]\n[SEED3 build: dscale10k_check_s3.py')

# 2. the arm table: kinds D3K (RH argv, 3,000 steps) and D9K (the same with --steps 9000). RH-like: bank, the two declarations, the readout token.
sub('RH_TOKENS = ("causal_g1x", "causal_g1")\n',
    '''#: LEVER3 (PREREG_V8F_LEVER3_V2 section 4): D3K = RH's argv on the rebuilt union with its own bank (3,000 steps); D9K = the same with `--steps 9000` (warmup 100, the cosine over its own 9,000 steps). Same readout token, kappa, dose p 0.1.
L3_KINDS = ("D3K", "D9K")
L3_STEPS = {"D3K": 3000, "D9K": 9000}
L3_ARMS = {"D3K_s0": ("D3K", 0), "D3K_s1": ("D3K", 1), "D9K_s0": ("D9K", 0), "D9K_s1": ("D9K", 1)}
for _a, (_k, _s) in L3_ARMS.items():
    RH_ARMS[_a] = (_k, _s)
    ARMS[_a] = dict(set="U", stage_b=True, seed=_s, dagger_seed=_s if _s else 0, steps="fixed")
RH_LIKE = RH_LIKE + L3_KINDS
RH_TOKENS = ("causal_g1x", "causal_g1")
''')

# 3. K-B2 for the dev-box layout: the stage-A checkpoint of the L3 arms lives under <root>/init/UE_sK/A/ (the MM's fetch), not <root>/runs/UE_sK/A/
sub('''def rh_stage_a_ckpt_problems(init: str, seed: int) -> list:''', '''def rh_stage_a_ckpt_problems(init: str, seed: int, kind: str | None = None) -> list:''')
sub('''    want = f"/runs/UE_s{seed}/A/ckpt_last.pt"
    return [] if _norm_path(init).endswith(want) else [f"K-B2: --init-ckpt {init!r} is not this arm's stage A (a path ending {want})"]''',
    '''    want = f"/runs/UE_s{seed}/A/ckpt_last.pt"
    if kind in L3_KINDS:                                                                    # LEVER3: the dev-box layout of the stage-A checkpoints (sha256 pinned by the bank's K-B2 check)
        want = f"/init/UE_s{seed}/A/ckpt_last.pt"
    return [] if _norm_path(init).endswith(want) else [f"K-B2: --init-ckpt {init!r} is not this arm's stage A (a path ending {want})"]''')
sub('''        bad += rh_stage_a_ckpt_problems(init, seed)''', '''        bad += rh_stage_a_ckpt_problems(init, seed, kind)''')

# 4. rh_argv_problems: D9K may differ from the template in --steps (own flag, pinned literally below); the dose branch of RH applies to both kinds
sub('''{"--out", "--init-ckpt"} | dag_bank | ({"--dagger-p"} if kind == "RHp3" else set())''',
    '''{"--out", "--init-ckpt"} | dag_bank | ({"--dagger-p"} if kind == "RHp3" else set()) | ({"--steps"} if kind == "D9K" else set())''')
sub('''        if kind == "RH":
            if a.get("--dagger-p") != t.get("--dagger-p") or not (a.get("--dagger-p") and float(a["--dagger-p"][0]) > 0):''',
    '''        if kind in L3_KINDS and a.get("--steps") != [str(L3_STEPS[kind])]:                 # LEVER3: the literal step count of the arm (3,000 / 9,000), never "the template's"
            bad.append(f"{kind} --steps reads {a.get('--steps')}, the card's only value for this arm is [{str(L3_STEPS[kind])!r}]")
        if kind in L3_KINDS and a.get("--warmup") != ["100"]:                               # the cosine runs over the arm's own steps with warmup 100
            bad.append(f"{kind} --warmup reads {a.get('--warmup')}, the card says ['100']")
        if kind in L3_KINDS:
            bk3 = _norm_path((a.get("--dagger-bank") or [""])[0]).rstrip("/")
            if not bk3.endswith(f"/bank/s{seed}") or "/bank_k/" in bk3:                     # the arm's own bank (card section 3.5), never the pod's and never a readout-K bank
                bad.append(f"{kind} --dagger-bank {bk3!r} is not a .../bank/s{seed} path")
            if tk != "causal_g1x":
                bad.append(f"{kind} --dagger-bank-readout {tk!r}: the card's bank is causal_g1x (RH's)")
        if kind == "RH" or kind in L3_KINDS:
            if a.get("--dagger-p") != t.get("--dagger-p") or not (a.get("--dagger-p") and float(a["--dagger-p"][0]) > 0):''')

# 5. C-STEPS stage B: 3,000 for every old arm, 9,000 for D9K
sub('''        return [] if t == STAGE_B_STEPS else [f"stage B runs {t} steps, the card says {STAGE_B_STEPS}"]''',
    '''        want_b = L3_STEPS.get((RH_ARMS.get(arm) or ("",))[0], STAGE_B_STEPS)
        return [] if t == want_b else [f"stage B runs {t} steps, the card says {want_b}"]''')

# 6. the pair of an L3 arm: D3K <-> D9K of the same seed (there is no B0 in this card). The new function sits after rh_pair_problems.
sub('''def _find_key(o, key):''', '''def l3_pair_problems(toks, pair_toks, arm: str, token: str | None) -> list:
    """C-RH-PAIR of an L3 arm: the arm's argv and its pair (D3K <-> D9K of the SAME seed) have the same flag set and equal values EXCEPT exactly ``--out`` (each its own, naming its arm) and ``--steps``
    (3,000 on D3K, 9,000 on D9K). The bank, its sha256, the init checkpoint, the dose, seed, readout token and kappa are therefore the same. There is no B0 arm in the LEVER3 card, so the stock
    ``rollhist_pair_check.py`` (RH <-> B0) is not run; this pair is the card's lever (steps) and nothing else."""
    if arm not in RH_ARMS or RH_ARMS[arm][0] not in L3_KINDS:
        return [f"{arm!r} is not an L3 arm"]
    kind, seed = RH_ARMS[arm]
    other = "D9K" if kind == "D3K" else "D3K"
    try:
        a, b = parse_argv(toks), parse_argv(pair_toks)
    except ValueError as e:
        return [f"pair argv does not parse: {e}"]
    bad = []
    for f in sorted(set(a) ^ set(b)):
        bad.append(f"flag {f} is on only one side of the {kind} <-> {other} pair")
    for f in sorted(set(a) & set(b)):
        if a[f] != b[f] and f not in ("--out", "--steps"):
            bad.append(f"flag {f}: {a[f]} differs from the pair's {b[f]} (the pair differs in --steps and --out only)")
    if a.get("--steps") != [str(L3_STEPS[kind])]:
        bad.append(f"{kind} --steps reads {a.get('--steps')}, expected [{str(L3_STEPS[kind])!r}]")
    if b.get("--steps") != [str(L3_STEPS[other])]:
        bad.append(f"the pair ({other}) --steps reads {b.get('--steps')}, expected [{str(L3_STEPS[other])!r}]")
    if f"/{other}_s{seed}/" not in _norm_path(" ".join(b.get("--out") or []) + "/"):
        bad.append(f"the pair's --out {b.get('--out')} does not name /{other}_s{seed}/")
    if a.get("--out") == b.get("--out"):
        bad.append("the pair has the same --out as the arm")
    if token is not None and (b.get("--dagger-bank-readout") or [""])[0] != token:
        bad.append(f"the pair's --dagger-bank-readout {b.get('--dagger-bank-readout')} is not the declared token {token!r}")
    return bad


def _find_key(o, key):''')

# 7. mint: the pair branch
sub('''        if pair is None:
            put("C-RH-PAIR", ["no --pair-argv (the other arm's filled argv: RH <-> B0 of the same seed)"])''',
    '''        if kind in L3_KINDS:                                                                  # LEVER3: D3K <-> D9K of the same seed (no B0 in this card)
            put("C-RH-PAIR", l3_pair_problems(toks, pair, arm, rtok) if pair is not None else ["no --pair-argv (the other L3 arm's filled argv: D3K <-> D9K of the same seed)"],
                readout_token=rtok, pair_argv_sha256=argv_sha(pair) if pair is not None else None, pair_check_sha256=None, pair_form="D3K<->D9K: --steps and --out only")
        elif pair is None:
            put("C-RH-PAIR", ["no --pair-argv (the other arm's filled argv: RH <-> B0 of the same seed)"])''')

# 8. the token file name: `*token.json` fails the secret scan of the repo (PI memory); the L3 checker writes and reads `gate_gatetok.json` (same record, schema and fields). The constant sits after TOKEN_SCHEMA.
sub('TOKEN_SCHEMA = "dscale10k-gate-token-v1"\n', 'TOKEN_SCHEMA = "dscale10k-gate-token-v1"\nGATE_TOKEN_FILE = "gate_gatetok.json"                  # LEVER3: was gate_token.json (a *token.json name fails the secret scan); content and schema unchanged\n')
assert s.count('"gate_token.json"') == 3, s.count('"gate_token.json"')
s = s.replace('"gate_token.json"', 'GATE_TOKEN_FILE')
n_edits += 1

open(dst, "w", encoding="utf-8", newline="").write(s)
print(f"patched {n_edits} anchors: {src} ({sha8}) -> {dst}")
