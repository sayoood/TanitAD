"""apply the P6-HN-S item-4 edits to refe/launch_gate.py in the edit tree (every anchor must occur exactly once)"""
from pathlib import Path
PK = Path("C:/Users/Admin/hn34/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
P = PK / "refe" / "launch_gate.py"
s = P.read_bytes().decode("utf-8")
assert "\r" not in s


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---- 1. the flags: the token fingerprints both inputs in EVERY scorer profile (the registry classifies them `data`); every profile except refe_hnscreen FORBIDS them
rep('''RLS1_FORBIDDEN = ("--preflight", "--lane-labels-train", "--lane-labels-heldout", "--b-compw", "--tau-s", "--tau-t") + AUX_HEAD_FLAGS      # RL-S1 is not the head's experiment''',
    '''HNSCREEN_FLAGS = ("--hn-bank", "--ctx-cache")                                           # P6-HN-S (PREREG_P6HN_SCREEN 4.2): the bank and the frozen-trunk context cache; ONLY refe_hnscreen accepts them
RLS1_FORBIDDEN = ("--preflight", "--lane-labels-train", "--lane-labels-heldout", "--b-compw", "--tau-s", "--tau-t") + AUX_HEAD_FLAGS + HNSCREEN_FLAGS      # RL-S1 is not the head's experiment, nor the hard-negative screen's''')
rep('''"--pdm-labels-train", "--pdm-labels-heldout", "--teacher-lane-train", "--teacher-lane-heldout", "--epo-terms-train", "--epo-terms-heldout", "--aux-drv-labels", "--epo-carve-logs")''',
    '''"--pdm-labels-train", "--pdm-labels-heldout", "--teacher-lane-train", "--teacher-lane-heldout", "--epo-terms-train", "--epo-terms-heldout", "--aux-drv-labels", "--epo-carve-logs",
             "--hn-bank", "--ctx-cache")''')
rep('''AUXS1_FORBIDDEN = ("--preflight", "--lane-labels-train", "--b-compw", "--tau-t", "--aux-drv-shuffle-seed") + AUXS1_EPO_FLAGS''',
    '''AUXS1_FORBIDDEN = ("--preflight", "--lane-labels-train", "--b-compw", "--tau-t", "--aux-drv-shuffle-seed") + AUXS1_EPO_FLAGS + HNSCREEN_FLAGS''')

# ---- 2. the hnscreen constants (after the aux S1c block, before the P2 constants)
HN_CONST = '''# ====================================================================================================================================================================================
# P6-HN-S (raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md 4.1 / 4.2 / 4.4): the offset hard-negative SCREEN.  refe_hnscreen = refe_rls1 (same runner, checks, required / forbidden flags, recipe, G-DVB,
# G-PREFLIGHT) with ITS arm table and the base argv PINNED.  A "pass" s trains A_s and B_s in ONE run (scorer_finetune trains both arms in a single process), so the arm is the pass: S0 S1 S2 (seeds 0 1 2);
# the extension passes S3 S4 are PI/rule-E arms (PREREG 7 rule E), not mintable until it fires.  The base argv is RL-S1's C1 argv token for token (eval/rls1b_arms.json); the ONLY allowed differences are
# --seed, --out, --dump-scores, the data paths re-rooted to the run box (their CONTENT fingerprints must equal C1's / C0's: `fingerprints_expected`), plus --hn-bank and --ctx-cache each exactly once.
HNSCREEN_ARM_SEEDS = {"S0": 0, "S1": 1, "S2": 2}
HNSCREEN_ARM_CLASS = {"S0": "C", "S1": "C", "S2": "C"}                                   # class C = no EPO lever: A_s and B_s differ by the hard-negative term alone
HNSCREEN_NEEDS_PI = {"S3": "S3 (seed 3) is an extension pass: it runs only under PREREG_P6HN_SCREEN section 7 rule E, which has not fired",
                     "S4": "S4 (seed 4) is an extension pass: it runs only under PREREG_P6HN_SCREEN section 7 rule E, which has not fired"}
HNSCREEN_PIN = {"--backbone": ("vitl16",), "--require-pdm": (), "--b-mode": ("listnet",), "--lam": ("0",), "--lr": ("1e-4",), "--max-updates": ("1000",), "--epochs": ("4",), "--workers": ("3",),
                "--eval-every": ("250",), "--lane-head": ()}      # C1's non-path tokens, value for value
HNSCREEN_PATHS = ("--ckpt", "--targets", "--images", "--calib", "--onpolicy", "--heldout", "--heldout-logs", "--out", "--pdm-labels-train", "--pdm-labels-heldout", "--dump-scores",
                  "--teacher-lane-train", "--teacher-lane-heldout") + HNSCREEN_FLAGS      # tokens whose VALUE may move (a re-rooted path) but which must be carried exactly once
HNSCREEN_TERMS = ("hn_rows", "hn_rows_expected", "hn_loss")                              # the trainer's train rows must emit these (D1: the rows that entered the term vs the rows the bank holds)
HNSCREEN_SUITE = RLS1_SUITE + [{"script": "test_hn_bank.py", "token": "ZZTEST_PASS"}, {"script": "test_ctx_cache.py", "token": "ZZTEST_PASS"},
                               {"script": "test_hn_screen_wiring.py", "token": "ZZTEST_PASS"}]


'''
rep('''# ====================================================================================================================================================================================
# P2 ''' if False else '''P2_RUN_DIR = "/workspace/data/refe_runs/p2_joint_pdm"''', HN_CONST + '''P2_RUN_DIR = "/workspace/data/refe_runs/p2_joint_pdm"''')

# ---- 3. the profile
rep('''# S1c = refe_aux_s1 (same runner, checks, data flags, forbidden flags, head levers, recipe except the budget) with ITS arm table and per-class budget.''',
    '''# P6-HN-S = refe_rls1 with ITS arm table (S0 S1 S2), the screen's own forbidden flags (the RL-S1 set minus --hn-bank / --ctx-cache, which it REQUIRES), `hn_rules` (hnscreen_refusals: the argv pin, G14 / G15
# are required by the argv, the trainer's own hn_refusals) and `schedule_reach` (G-DVB re-derives ceil(train_sets / batch x epochs / accum) against --max-updates 1000: PREREG 4.4's G-CLOCK).
PROFILES["refe_hnscreen"] = {**PROFILES["refe_rls1"], "name": "refe_hnscreen", "suite": HNSCREEN_SUITE, "arm_class": HNSCREEN_ARM_CLASS, "arm_seeds": HNSCREEN_ARM_SEEDS, "arms_need_pi": HNSCREEN_NEEDS_PI,
                             "forbidden_flags": tuple(f for f in RLS1_FORBIDDEN if f not in HNSCREEN_FLAGS), "hn_rules": True, "schedule_reach": True, "fingerprints_expected": {}}
# S1c = refe_aux_s1 (same runner, checks, data flags, forbidden flags, head levers, recipe except the budget) with ITS arm table and per-class budget.''')

# ---- 4. the refusals
HN_REF = '''# ============================================================================================ P6-HN-S arm rules (refe_hnscreen)
def _same_vals(a, b) -> bool:
    """two flag-value tuples are the same value: numerically when both parse as numbers ('1e-4' == '0.0001'), else as text"""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            if float(x) != float(y):
                return False
        except ValueError:
            if x != y:
                return False
    return True


def hnscreen_refusals(prof: dict, argv: list, mod, parser) -> tuple:
    """(refusals, extra live terms) of the screen's rules, ON TOP of rls1_refusals (class C, the recipe, the registered seed, the lane-head pairing, scorer_finetune's own epo refusals).  PREREG_P6HN_SCREEN 4.2 / 4.4:
    the argv is RL-S1's C1 argv plus --hn-bank and --ctx-cache, each exactly once -- ANY moved, added, dropped or doubled token is a refusal; --seed is named (the pass); path values may move (re-rooted) but their
    CONTENT fingerprints must equal the registered ones when the profile carries `fingerprints_expected`; the trainer's own `hn_refusals` (--hn-bank with --epo / --lam != 0 / --b-mode != listnet / --aux-drv,
    --ctx-cache with --aux-drv) are surfaced, not re-derived."""
    import gate_checks as C
    name, out = prof["name"], []
    got: dict = {}
    for f, v in GT.flag_pairs(list(argv)):
        got.setdefault(f, []).append(tuple(v))
    if argv and not str(argv[0]).startswith("--"):
        out.append(f"{name}: the argv starts with {argv[0]!r}, which belongs to no flag")
    allowed = set(HNSCREEN_PIN) | set(HNSCREEN_PATHS) | {"--seed"}
    for f in sorted(allowed):
        n = len(got.get(f, ()))
        if n == 0:
            out.append(f"{name}: {f} is DROPPED -- the screen's argv is RL-S1's C1 argv plus --hn-bank and --ctx-cache (PREREG_P6HN_SCREEN 4.2); every token is required")
        elif n > 1:
            out.append(f"{name}: {f} is carried {n} times (each token exactly once)")
    for f, want in HNSCREEN_PIN.items():
        if len(got.get(f, ())) == 1 and not _same_vals(got[f][0], want):
            out.append(f"{name}: {f} MOVED to {' '.join(got[f][0]) or '(no value)'}; the screen pins {f} {' '.join(want)} (RL-S1 C1's value)")
    for f in HNSCREEN_PATHS + ("--seed",):
        if len(got.get(f, ())) == 1 and len(got[f][0]) != 1:
            out.append(f"{name}: {f} needs exactly one value, the argv has {len(got[f][0])}")
    for f in sorted(set(got) - allowed):
        out.append(f"{name}: the argv ADDS {f} {' '.join(got[f][0])}, which the screen does not allow (allowed differences from C1: --seed, --out, --dump-scores, re-rooted data paths, and --hn-bank / --ctx-cache)")
    ns, err = C.parse_scorer_argv(parser, argv)
    if err:
        return out + [f"{name}: the trainer's parser rejects this argv ({err})"], ()
    out += [f"scorer_finetune refuses this argv: {m}" for m in mod.hn_refusals(ns, parser)]
    for f, want in sorted((prof.get("fingerprints_expected") or {}).items()):      # content fingerprints of the re-rooted data (UNVERIFIED offline: the registered values live on the run box)
        vals = got.get(f)
        if not vals or len(vals) != 1 or len(vals[0]) != 1:
            continue
        try:
            fk = GT.fp_key(GT.fingerprint(vals[0][0]))
        except (OSError, ValueError, KeyError) as e:
            out.append(f"{name}: cannot fingerprint {f} {vals[0][0]} ({e})")
            continue
        if fk != want:
            out.append(f"{name}: the content fingerprint of {f} is not the registered one (RL-S1 C1 / C0's token fingerprint)")
    return out, HNSCREEN_TERMS


def hnscreen_regression_argv(argv: list, arm: str) -> list:
    """The deliberate-regression arms of refe_hnscreen that act on the ARGV (PREREG_P6HN_SCREEN 4.4): `moved_lever` (a pinned token changed: --lr 2e-4) and `epo_with_hn` (an EPO lever beside the bank).
    Every other arm acts on the evidence (gate_checks.apply_hn_arm; check_dvb_scorer's arms) and returns the argv unchanged.  Each must be REFUSED by profile_refusals."""
    argv = list(argv)
    if arm == "moved_lever":
        return [("2e-4" if argv[i - 1] == "--lr" else t) for i, t in enumerate(argv)]
    if arm == "epo_with_hn":
        return argv + ["--epo", "--epo-terms-train", "/w/terms"]
    return argv


'''
rep('''# ============================================================================================ aux S1 arm rules (refe_aux_s1)
def auxs1_arm_class(argv: list) -> str:''', HN_REF + '''# ============================================================================================ aux S1 arm rules (refe_aux_s1)
def auxs1_arm_class(argv: list) -> str:''')
rep('''    elif prof.get("arm_rules") == "auxs1":
        rs, extra = auxs1_refusals(prof, argv, T, parser)
        out += rs
        terms += extra
    elif prof.get("recipe_rules") == "sft4r":''', '''    elif prof.get("arm_rules") == "auxs1":
        rs, extra = auxs1_refusals(prof, argv, T, parser)
        out += rs
        terms += extra
    if prof.get("hn_rules"):                                                # P6-HN-S: the screen's argv pin on top of the RL-S1 rules above
        rs, extra = hnscreen_refusals(prof, argv, T, parser)
        out += rs
        terms += extra
    if prof.get("recipe_rules") == "sft4r":''')

# ---- 5. the preflight consumer
rep('''    g12 = next((v for k, v in gates.items() if prefix[k] == "G12"), None)
    if g12 is not None:                                      # G12 (--aux-drv) is re-judged from its raw fields too: its own `ok` is a claim, the fields are the evidence
        reasons += C.judge_g12(g12, ns)''', '''    g12 = next((v for k, v in gates.items() if prefix[k] == "G12"), None)
    if g12 is not None:                                      # G12 (--aux-drv) is re-judged from its raw fields too: its own `ok` is a claim, the fields are the evidence
        reasons += C.judge_g12(g12, ns)
    g14 = next((v for k, v in gates.items() if prefix[k] == "G14"), None)
    g15 = next((v for k, v in gates.items() if prefix[k] == "G15"), None)
    if arm in C.HN_ARMS:                                     # P6-HN-S deliberate-regression arms: the same preflight, with the raw fields of the defect (each must turn a judge RED)
        g14, g15 = C.apply_hn_arm(g14, g15, arm)
    if g14 is not None:                                      # G14 (--hn-bank): the hard-negative term is live on arm B only, its controls read EXACTLY 0.0, the identities hold -- from the raw fields
        reasons += C.judge_g14(g14, ns)
    if g15 is not None:                                      # G15 (--ctx-cache): the cache IS the live trunk forward, bit for bit -- from the raw fields, and bound to the data event's built block
        reasons += C.judge_g15(g15, ns)
        ccb = (pf["data"] or {}).get("ctx_cache") if isinstance(pf.get("data"), dict) else None
        if isinstance(ccb, dict) and ccb.get("trunk_fingerprint") != g15.get("trunk_fingerprint_checkpoint"):
            reasons.append("G15: the data event's ctx_cache.trunk_fingerprint is not the checkpoint fingerprint G15 measured (the cache the trainer built is not the one the identity was proved on)")
        if isinstance(ccb, dict) and ccb.get("manifest_sha256") != g15.get("manifest_sha256"):
            reasons.append("G15: the data event's ctx_cache.manifest_sha256 is not the manifest G15 measured")''')

# ---- 6. the CLI: the arm names
rep('''p.add_argument("--arm-name", choices=sorted(set(RLS1_ARM_CLASS) | set(RLS1D_ARM_CLASS) | set(RLS1B_ARM_CLASS) | set(RLS1B_NEEDS_PI) | set(RLS1C_ARM_CLASS)) + sorted(AUXS1_ARM_CLASS) + sorted(AUXS1C_ARM_CLASS),''',
    '''p.add_argument("--arm-name", choices=sorted(set(RLS1_ARM_CLASS) | set(RLS1D_ARM_CLASS) | set(RLS1B_ARM_CLASS) | set(RLS1B_NEEDS_PI) | set(RLS1C_ARM_CLASS) | set(HNSCREEN_ARM_CLASS) | set(HNSCREEN_NEEDS_PI)) + sorted(AUXS1_ARM_CLASS) + sorted(AUXS1C_ARM_CLASS),''')
rep('''refe_aux_s1: N-s0..H2-s2; refe_aux_s1c: H1u10k-s0 | H1u2k-s0; "''', '''refe_aux_s1: N-s0..H2-s2; refe_aux_s1c: H1u10k-s0 | H1u2k-s0; refe_hnscreen: S0 S1 S2 (S3 S4 need rule E); "''')
rep('''(refe_rls1, refe_rls1b, refe_rls1c, refe_rls1d, refe_aux_s1, refe_aux_s1c), not {a.profile}")''', '''(refe_rls1, refe_rls1b, refe_rls1c, refe_rls1d, refe_aux_s1, refe_aux_s1c, refe_hnscreen), not {a.profile}")''')
P.write_bytes(s.encode("utf-8"))
print("ok", len(s))
