import re
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:90], s.count(old))
    s = s.replace(old, new)


def cut(start, end, new):
    """replace s[start .. end) where both are unique marker strings (end excluded)"""
    global s
    i = s.index(start)
    j = s.index(end, i)
    s = s[:i] + new + s[j:]


# ---- header
cut('"""ACTRSP launch-gate checker.', "  argv --tag T", '''"""ACTRSP launch-gate checker. Runs on box 2 (venv python; stdlib, numpy and torch only in the weights-identity load) and on the dev box. FORK of actsens_check.py (token schema unchanged: actsens-gate-token-v1; written_by actrsp_check.py gate).
Pattern: pin the argv, run the trainer with --selfcheck-only at the shipped commit, assert what it BUILT (the wm-only mode, the response weight and geometry read back from the BUILT loss, control C3 at step 0), write a per-run token (only here, never by hand), train under the token, post-check, bank.
Card: theory/actrsp/PREREG_V8F_ACTRSP_DRAFT.md (draft; the Master Mind freezes it). Frozen argvs: argvs_actrsp.json (sha256 pinned below). Code under test: the ACTRSP diff on ACTSENS commit 81cee6c1 (the landed commit is ACTRSP_COMMIT).
REFUSES an argv whose --wm-rsp-w is not named: every RSP arm must carry `--wm-rsp-w 1.0` exactly once, every NULL arm must not carry the flag (argv_rsp_problems, run by the gate before anything is built).

''')
rep("  argv --tag T [--pilot]", "  argv --tag T [--pilot]")

# ---- identities
cut("R35_TERMS_SHA256 =", "ARGVS_FILE =", "")
rep('ARGVS_SHA256 = "b0dd0af8fb7211af00de8e3c7291d9d8a3d6ee5316ff9665ad5487c10c8b72e5"', 'ARGVS_SHA256 = "@ARGVS_SHA256@"')
rep("# frozen/argvs_actrsp.json, built from theory/actrsp/argvs_actrsp.txt (MEASURED 2026-11-09)", "# argvs_actrsp.json, built by make_argvs_actrsp.py from actsens_NULL_s2 / NULL_s3 argv.txt (MEASURED 2026-11-16)")
rep('TRAINER_REL, MODEL_REL, DATA_REL, WINDOWS_REL = "stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/data/v8f_wm_windows.py"',
    'TRAINER_REL, MODEL_REL, DATA_REL, WINDOWS_REL, HJEPA_REL = "stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/data/v8f_wm_windows.py", "stack/tanitad/models/hjepa.py"')
cut("NEEDLES = {", "HEX64, HEX40", '''NEEDLES = {TRAINER_REL: ('"--wm-rsp-w"', "def bad_wm_rsp", "def wm_rsp_readback", "lever:wm_rsp_w", "wm_rsp_w=float(a.wm_rsp_w)", '"wm/rsp/L1"', "--wm-only"),
           MODEL_REL: ("wm_rsp_w", "def built_rsp", "RSP_DELTA"), HJEPA_REL: ("def response_loss", "def built_rsp", "RSP_K, RSP_STRIDE, RSP_CTX, RSP_S_MIN", "rsp_w: float = 0.0"), WINDOWS_REL: ("class V8FMiniWMWindows",)}
''')
cut("#: tag table.", "RUN_ORDER =", '''#: tag table. lever = rsp | null (the card's section 3); seed = the training seed; init = the M2b seed index of the start state, the zcache and the WM re-initialisation; order = the chain order (card section 3: pilot, RSP_s0, RSP_s2, NULL_s4, RSP_s1, RSP_s3, NULL_s5).
TABLE = {
    "RSP_s0":  {"lever": "rsp",  "seed": 0, "init": 0, "order": 1, "blocked": None},
    "RSP_s2":  {"lever": "rsp",  "seed": 2, "init": 0, "order": 2, "blocked": None},
    "NULL_s4": {"lever": "null", "seed": 4, "init": 0, "order": 3, "blocked": None},
    "RSP_s1":  {"lever": "rsp",  "seed": 1, "init": 1, "order": 4, "blocked": None},
    "RSP_s3":  {"lever": "rsp",  "seed": 3, "init": 1, "order": 5, "blocked": None},
    "NULL_s5": {"lever": "null", "seed": 5, "init": 1, "order": 6, "blocked": None},
}
PILOT_TAG, PILOT_BASE = "PILOT_RSP_s0", "RSP_s0"
''')
rep('GATE_AT_PILOT = ["IDM_s0", "DIV_s0"]', 'GATE_AT_PILOT = ["RSP_s0", "RSP_s2"]')

# ---- C3: read live.c3 (the trainer's own step-0 comparison on the same copy), no R35 reference file
cut("def r35_terms(k):", "def base_tag(tag):", '''RSP_W = 1.0


def argv_rsp_problems(tag, toks):
    """REFUSES an argv whose --wm-rsp-w is not named: an RSP arm carries `--wm-rsp-w 1.0` exactly once; a NULL arm does not carry the flag at all (C1: absent = the ACTSENS build)."""
    lever = TABLE[base_tag(tag)]["lever"]
    idx = [i for i, t in enumerate(toks) if t == "--wm-rsp-w"]
    if lever == "null":
        return [f"{tag}: a NULL arm names --wm-rsp-w (the flag must be absent)"] if idx else []
    if len(idx) != 1:
        return [f"{tag}: the argv names --wm-rsp-w {len(idx)} times; an RSP arm must name it exactly once as `--wm-rsp-w {RSP_W}` (an unnamed flag leaves the term off: a lever that is declared nowhere)"]
    if idx[0] + 1 >= len(toks) or toks[idx[0] + 1] != "1.0":
        return [f"{tag}: --wm-rsp-w is {toks[idx[0] + 1] if idx[0] + 1 < len(toks) else None!r}, the card fixes 1.0"]
    return []


def c3_problems(live, terms_first):
    """card C3 at step 0: the trainer compared the flag-on step-0 terms with the SAME copy (same weights, same batch, same seed) with the response term switched off: every unweighted term equal, levels 2 / 3 equal, level/L1 difference = 1.0 x rsp/L1."""
    c3 = live.get("c3") if isinstance(live.get("c3"), dict) else None
    if c3 is None:
        return ["C3: live.c3 is missing (the trainer did not run the step-0 identity)"]
    p = []
    if c3.get("pass") is not True:
        p.append(f"C3: the trainer's step-0 identity did not pass: {str(c3)[:300]}")
    if not (c3.get("unweighted_equal") is True and c3.get("levels_23_equal") is True and c3.get("ref_has_no_rsp_keys") is True and num(c3.get("n_unweighted")) and c3["n_unweighted"] >= 14):
        p.append(f"C3: unweighted terms / levels 2-3 not equal to the flag-off reference: {str(c3)[:300]}")
    try:
        d, want = c3["level_L1_diff"], RSP_W * terms_first["wm/rsp/L1"]
        if not (want > 0.0 and abs(d - want) <= max(1e-5 * want, c3.get("tol", 0.0)) and c3.get("rsp_w_x_rsp") == want):
            p.append(f"C3: level/L1 difference {d!r} != {RSP_W} x rsp/L1 {want!r}")
    except (KeyError, TypeError):
        p.append("C3: level_L1_diff or wm/rsp/L1 missing")
    return p


''')

# ---- gate_problems: the lever blocks
cut("    # ---- the two ACTRSP levers READ BACK", "    eq(\"info.n_windows (R35's, MEASURED)\"", '''    # ---- the ACTRSP lever READ BACK from the BUILT objects (levers.wm_rsp_readback: the BUILT LevelConfig and the BUILT loss's own measurement); the ACTSENS levers are absent
    rr = lv.get("wm_rsp_readback")
    if lever == "rsp":
        if not isinstance(rr, dict):
            p.append("levers.wm_rsp_readback is missing: the response weight was not read back from the built loss")
        else:
            eq("wm_rsp_readback.declared", rr.get("declared"), RSP_W)
            eq("wm_rsp_readback.built_config (the LevelConfig the loss multiplies by)", rr.get("built_config"), RSP_W)
            b = rr.get("built") if isinstance(rr.get("built"), dict) else {}
            w = b.get("w_in_loss_rsp")
            if not (finite(w) and abs(w - RSP_W) <= 1e-6):
                p.append(f"BUILT loss gives the response term the weight {w!r}, wanted {RSP_W} (1e-6)")
            for k, v in (("calls", 1), ("contrasts", 4), ("k", 10), ("ctx_passed", 1), ("anchors", 9)):
                eq(f"wm_rsp_readback.built.{k} (the card's geometry at hmax 91)", b.get(k), v)
            eq("wm_rsp_readback.hmax", rr.get("hmax"), HMAX)
    elif rr is not None:
        p.append(f"levers.wm_rsp_readback = {rr!r} on a {lever} arm: the flag must be absent")
    eq("levers.wm_idm_gamma_readback (ACTSENS lever, must be absent)", lv.get("wm_idm_gamma_readback"), None)
    eq("levers.wm_sampler (ACTSENS lever, must be absent)", lv.get("wm_sampler"), None)
    eq("built WMAux.gamma (lever wm_idm_gamma, the record value at every arm)", lv.get("wm_idm_gamma"), 100.0)
    rec = info.get("wm_sampler") if isinstance(info.get("wm_sampler"), dict) else {}
    eq("info.wm_sampler.kind (the uniform sampler of every earlier arm)", rec.get("kind"), "uniform")
''')
rep('''    if lever == "idm":
        p += c3_problems(init, terms)
''', '''    live_first = sc.get("live_first") if isinstance(sc.get("live_first"), dict) else None
    if lever == "rsp":
        if not (finite(terms.get("wm/rsp/L1")) and live_first is not None and finite(live_first.get("wm/rsp/L1"))):
            p.append("G-LIVE: wm/rsp/L1 is not among the live terms / live_first of an RSP arm")
        else:
            p += c3_problems(lp, live_first)
    else:
        if any("rsp" in k for k in terms) or live_first is not None or "c3" in lp:
            p.append("a NULL arm carries a response term / live_first / live.c3 in its selfcheck")
''')
rep('''    eq("declared_vs_built", sc.get("declared_vs_built"), "PASS")
''', '''    eq("declared_vs_built", sc.get("declared_vs_built"), "PASS")
    p += argv_rsp_problems(tag, toks)
''')

# ---- C2 identity: level-1 loss identity with the response term
cut("GAMMA_BETA, GAMMA_ROLLOUT_W", "def adam_state_problems", '''GAMMA_BETA, GAMMA_ROLLOUT_W, GAMMA_ACT_SIGREG, GAMMA_TOL = 0.08, 1.0, 0.005, 1e-4


def rsp_from_rows(rows, with_rsp):
    """card C2 (rsp_w_from_log.py): per logged step, level/L1 - pred - rollout_w x rollout - beta x sigreg - 100 x idm = rsp_w x rsp/L1 on an RSP arm (-> rsp_w), and the gamma identity of levels 2 / 3 (and of level 1 on a NULL arm) with the weights 100 / 100 / 100."""
    out = {"rsp_w": [], "g1": [], "g2": [], "g3": []}
    for r in rows:
        if "wm/level/L1" not in r:
            continue
        for L in (1, 2, 3):
            res = r[f"wm/level/L{L}"] - r[f"wm/pred/L{L}"] - GAMMA_ROLLOUT_W * r.get(f"wm/rollout/L{L}", 0.0) - GAMMA_BETA * r[f"wm/sigreg/L{L}"] - (GAMMA_ACT_SIGREG * r[f"wm/act_sigreg/L{L}"] if L >= 2 else 0.0)
            if L == 1 and with_rsp:
                res = res - 100.0 * r["wm/idm/L1"]
                if r.get("wm/rsp/L1"):
                    out["rsp_w"].append(res / r["wm/rsp/L1"])
            elif r[f"wm/idm/L{L}"]:
                out[f"g{L}"].append(res / r[f"wm/idm/L{L}"])
    return out


def gamma_log_problems(tag, log_path):
    """-> (evidence, problems): RSP arms recover rsp_w = 1.0 at every logged step and read gamma 100 at levels 2 / 3; NULL arms carry no `wm/rsp/L1` key and read gamma 100 / 100 / 100; each within 1e-4 relative at EVERY logged step."""
    rsp = TABLE[base_tag(tag)]["lever"] == "rsp"
    try:
        rows = [json.loads(ln) for ln in open(log_path, encoding="utf-8") if ln.strip()]
    except (OSError, ValueError):
        return {}, ["train_log.jsonl unreadable for the rsp-from-log identity"]
    ev, p = {}, []
    has = sum(1 for r in rows if "wm/rsp/L1" in r)
    ev["rows_with_rsp_key"], ev["rows"] = has, len(rows)
    if rsp and not has:
        p.append("rsp-from-log: no logged step carries wm/rsp/L1 on an RSP arm")
    if not rsp and has:
        p.append(f"rsp-from-log: {has} logged steps carry wm/rsp/L1 on a NULL arm (the flag must be absent)")
    g = rsp_from_rows(rows, rsp)
    want = {"g1": 100.0, "g2": 100.0, "g3": 100.0}
    if rsp:
        v = g["rsp_w"]
        ev["rsp_w"] = {"n": len(v), "min": min(v) if v else None, "max": max(v) if v else None, "want": RSP_W}
        if not v:
            p.append("rsp-from-log: no logged step carries the level-1 terms and rsp/L1")
        elif max(abs(x - RSP_W) for x in v) > GAMMA_TOL * RSP_W:
            p.append(f"rsp-from-log: recovered rsp_w in [{min(v):.6f}, {max(v):.6f}], wanted {RSP_W} within {GAMMA_TOL:g} relative at every step")
        want.pop("g1")
    for k, w in want.items():
        v = g[k]
        ev[k] = {"n": len(v), "min": min(v) if v else None, "max": max(v) if v else None, "want": w}
        if not v:
            p.append(f"rsp-from-log: no logged step carries the {k} loss terms")
        elif max(abs(x - w) for x in v) > GAMMA_TOL * w:
            p.append(f"rsp-from-log: {k} recovered in [{min(v):.6f}, {max(v):.6f}], wanted {w} within {GAMMA_TOL:g} relative at every step")
    return ev, p


''')
rep('t = tokens("DIV_s0")', 't = tokens("RSP_s0")')
rep('return PILOT_OUT_DECL + r"\\DIV_s0" if is_pilot(tag)', 'return PILOT_OUT_DECL + r"\\RSP_s0" if is_pilot(tag)')
s = s.replace("DIV_s0's", "RSP_s0's")
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p)
