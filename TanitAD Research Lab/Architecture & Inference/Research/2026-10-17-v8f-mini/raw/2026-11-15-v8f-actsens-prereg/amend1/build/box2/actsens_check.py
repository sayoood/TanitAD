"""ACTSENS launch-gate checker. Runs on box 2 (venv python; stdlib, numpy and torch only in the weights-identity load) and on the dev box.
Pattern: the ladder/ACTSENS checker: pin the argv, run the trainer with --selfcheck-only at the shipped commit, assert what it BUILT (the wm-only mode, the per-level IDM weights read back from the built LevelConfigs and the built sampler), write a per-run token (only here, never by hand), train under the token, post-check, bank.
Card: theory/actsens/PREREG_V8F_ACTSENS_DRAFT.md (frozen by the Master Mind 2026-11-15, sha256 b62227d9). Frozen argvs: frozen/argvs_actsens.json (sha256 pinned below). Code under test: the ACTSENS diff on ladder commit 1faa1a06 (the landed commit is ACTSENS_COMMIT).

  argv --tag T [--pilot] [--out-txt F]    dev box: the frozen argv of tag T (pilot: --out moved to the pilot dir, --steps 200 appended), one line
  render --template F --out F --set K=V   fill @K@ placeholders, write with CRLF line ends
  preflight                               disk floor, tree bound to ACTSENS_COMMIT (TIP_ACTSENS.txt, manifest members re-hashed, needles), start states and data files present
  stage --tag T --gid G [--pilot]         ONE box-2 job: preflight, pin, gate (selfcheck + BUILT-lever assertions + token), train, post; markers STAGE_<G>.done/.fail
  cleanup --tag T --gid G                 box 2: delete the run's own temporary gate dirs by literal path, ONLY after the dev box proved the bank (BANK_VERIFIED_<G>); never a checkpoint
  verify-token --token F --argv-file F --tag T --commit C
  project --pilot-dir D --out F --commit C   dev box: PILOT.json re-derived from the pulled pilot log
  permit --pilot F --commit C [--amendment F]   the 72 h rule over the planned runs
Test hook (dev box only): ACTSENS_ROOT maps every D:\\tanitad path to <root>; ACTSENS_FAKE_INIT_SHA_FILE replaces the start-state constants; ACTSENS_POLL_S sets the sampler period."""
import hashlib
import json
import ntpath
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- frozen identities ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ARGVS_SHA256 = "c111dc0372dd35b973adadd7a0bcd0d514df708b03aea03f3d7ad86796dc94e3"        # frozen/argvs_actsens.json, built from theory/actsens/argvs_actsens.txt (MEASURED 2026-11-09)
R35_TERMS_SHA256 = "3391b3b9b158c47ec5d7a2ed2053b7f1e2ca15e0a5877dda484482949672a2a9"      # frozen/r35_live_terms.json: live.terms of ladder_R35_s{0,1}/selfcheck.json (MEASURED on box 2; the file records each source's sha256)
R35_TERMS_FILE = os.path.join(HERE, "r35_live_terms.json") if os.path.exists(os.path.join(HERE, "r35_live_terms.json")) else os.path.join(HERE, "frozen", "r35_live_terms.json")
ARGVS_FILE = os.path.join(HERE, "argvs_actsens.json") if os.path.exists(os.path.join(HERE, "argvs_actsens.json")) else os.path.join(HERE, "frozen", "argvs_actsens.json")
#: M2b start states and the data identity are INHERITED from the ladder checker (same init files, same corpus, same exclusion); the ladder gate read them back from a real selfcheck, UNVERIFIED for this commit until its first real selfcheck is read.
INIT_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
V9_MD5 = "83e30bd3791f0767707eade5f4671759"
DATA = {"train_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1", "excluded_sha256": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "n_clips": 4512, "n_excluded": 60}
#: the full length of every run is the argv's own --steps (19,164 = 4 x 4791, the ladder's length); the pilot appends --steps 200.
RUN_STEPS, PILOT_STEPS, LIMIT_H = 19164, 200, 72.0
#: DISK REQUIREMENT (MM 2026-11-15): one run writes ckpt_last.pt, MEASURED 46,847,094 B on box 2 for this model class (R35H_s0, SLOW_s0), plus gate outputs (< 0.1 GB); box 2's D: read 5,338,267,648 B free before launch;
#: five checkpoints stay on box 2 after banking. A run starts only with NEED_GB_RUN free; the tree ship needs NEED_GB_SHIP. Below it the chain stops with exit 6 (stage rc 13).
NEED_GB_RUN, NEED_GB_SHIP = 2, 1
DECL_D = r"D:\tanitad"
HOME_DECL, REPO_DECL, M2B_DECL = DECL_D + r"\actsens", DECL_D + r"\actsens\repo", DECL_D + r"\runs\v8f_mini_full"
ZC_DECL = DECL_D + r"\zcache"
PILOT_OUT_DECL = HOME_DECL + r"\pilot_out"
TRAINER_REL, MODEL_REL, DATA_REL, WINDOWS_REL = "stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/data/v8f_wm_windows.py"
HMAX, CTX, WM_PARAMS_BASE = 91, (3, 3, 3), 2011970     # R35_s0 / R35_s1 selfcheck.json levers.wm_only.n_wm (MEASURED, ladder_R35_s{k}/selfcheck.json): wm_k 1, strides 3 5, hmax 91. Neither lever adds or removes a parameter (loss weight / window draw only).
N_WINDOWS, WINDOW_LIST_SHA = 40358, "1e4fc4c4401f89dcb94d7981bfb4907ac18f87e82b77ebd1d3783e080ebf6e2c"   # R35_s0 and R35_s1 selfcheck info (MEASURED, identical at both chassis)
NEEDLES = {TRAINER_REL: ('"--wm-idm-gamma"', '"live_first"', '"--wm-sampler"', "def bad_wm_idm_gamma", "def bad_wm_sampler", "lever:wm_idm_gamma", "data:wm_sampler", "def make_sampler", "def wm_sampler_probe"),
           MODEL_REL: ("wm_gamma_levels",), DATA_REL: ("class DiversitySampler", "def window_act_means", "def check_cell_shares"), WINDOWS_REL: ("class V8FMiniWMWindows",)}
HEX64, HEX40 = re.compile(r"^[0-9a-f]{64}$"), re.compile(r"^[0-9a-f]{40}$")
#: tag table. lever = idm | div | null (the card's section 2); seed = the training seed; init = the M2b seed index of the start state, the zcache and the WM re-initialisation (NULL_s2 keeps M2b_s0's).
TABLE = {
    "IDM_s0":  {"lever": "idm",  "seed": 0, "init": 0, "order": 1, "blocked": None},
    "NULL_s2": {"lever": "null", "seed": 2, "init": 0, "order": 2, "blocked": None},
    "DIV_s0":  {"lever": "div",  "seed": 0, "init": 0, "order": 3, "blocked": None},
    "IDM_s1":  {"lever": "idm",  "seed": 1, "init": 1, "order": 4, "blocked": None},
    "DIV_s1":  {"lever": "div",  "seed": 1, "init": 1, "order": 5, "blocked": None},
    "NULL_s3": {"lever": "null", "seed": 3, "init": 1, "order": 6, "blocked": None},
    # AMEND_ACTSENS_1 (A2): gate-only reference gates. R35_s{k}'s argv + `--wm-idm-gamma 100 100 100` (the default dose, NAMED so that the trainer records live_first). They never train and get no run token; order 0 keeps them out of RUN_ORDER.
    "REF_s0":  {"lever": "ref",  "seed": 0, "init": 0, "order": 0, "blocked": None},
    "REF_s1":  {"lever": "ref",  "seed": 1, "init": 1, "order": 0, "blocked": None},
}
PILOT_TAG, PILOT_BASE = "PILOT_DIV_s0", "DIV_s0"
RUN_ORDER = sorted((t for t, v in TABLE.items() if v["order"] and not v["blocked"]), key=lambda t: TABLE[t]["order"])
GATE_AT_PILOT = ["REF_s0", "IDM_s0", "DIV_s0"]   # the pilot job also gates these argvs IN THIS ORDER (REF_s0 first: IDM_s0's C3 reads its live_first), so run 1 starts under a token, steps_total is read before the 72 h decision and both speed modes are read back


# ---- environment ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fake():
    return os.environ.get("ACTSENS_ROOT")


def loc(p):
    r = fake()
    if r and p.lower().startswith(DECL_D.lower()):
        return os.path.join(r, *[x for x in p[len(DECL_D):].split("\\") if x])
    return p


def home():
    return loc(HOME_DECL)


def repo():
    return loc(REPO_DECL)


def commit():
    return os.environ.get("ACTSENS_COMMIT", "")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def init_sha(s):
    if fake() and os.environ.get("ACTSENS_FAKE_INIT_SHA_FILE"):
        return json.load(open(os.environ["ACTSENS_FAKE_INIT_SHA_FILE"]))[str(s)]
    return INIT_SHA[s]


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def finite(x):
    return num(x) and x == x and abs(x) != float("inf")


def write_text(p, s):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def tokens_sha(toks):
    return hashlib.sha256(json.dumps(list(toks), separators=(",", ":")).encode()).hexdigest()


def argv_sha16(toks):
    return hashlib.sha256(" ".join(toks).encode()).hexdigest()[:16]


# ---- the frozen argv ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def frozen_argvs():
    got = sha_file(ARGVS_FILE)
    if got != ARGVS_SHA256 and not os.environ.get("ACTSENS_TEST_ARGVS"):
        raise SystemExit(f"REFUSED: {ARGVS_FILE} sha256 {got} != the pinned {ARGVS_SHA256} (the frozen argv file changed)")
    return {k: v["argv"] for k, v in json.load(open(ARGVS_FILE, encoding="utf-8")).items()}


def r35_terms(k):
    got = sha_file(R35_TERMS_FILE)
    if got != R35_TERMS_SHA256 and not os.environ.get("ACTSENS_TEST_ARGVS"):
        raise SystemExit(f"REFUSED: {R35_TERMS_FILE} sha256 {got} != the pinned {R35_TERMS_SHA256}")
    return json.load(open(R35_TERMS_FILE, encoding="utf-8"))[f"s{k}"]["terms"]


UNWEIGHTED = ("wm/pred/L1", "wm/sigreg/L1", "wm/idm/L1", "wm/rollout/L1", "wm/rollout_valid/L1", "wm/rollout_masked/L1", "wm/pred/L2", "wm/sigreg/L2", "wm/idm/L2", "wm/act_sigreg/L2", "wm/rollout/L2",
              "wm/rollout_valid/L2", "wm/rollout_masked/L2", "wm/level/L2", "wm/pred/L3", "wm/sigreg/L3", "wm/idm/L3", "wm/act_sigreg/L3", "wm/rollout/L3", "wm/rollout_valid/L3", "wm/rollout_masked/L3", "wm/level/L3")


def ref_tag(init):
    return f"REF_s{init}"


def _first_problems(who, lf):
    """live_first (G-LIVE step 0, taken before any optimiser step) must carry every unweighted term the card compares, finite, and wm/level/L1 and wm/idm/L1."""
    if not isinstance(lf, dict) or not lf:
        return [f"{who}: live_first is missing (the trainer records it when --wm-idm-gamma is named)"]
    miss = [k for k in UNWEIGHTED + ("wm/level/L1", "wm/idm/L1") if not finite(lf.get(k))]
    return [f"{who}: live_first lacks finite {miss[:3]!r} ({len(miss)} terms)"] if miss else []


def c3_problems(init, live_first, ref_first):
    """card C3 as amended (AMEND_ACTSENS_1 A1): IDM_s{k}'s live_first (G-LIVE step 0: same init, same first batch) against REF_s{k}'s live_first (R35's argv + the default dose, same tree, read from REF's gate output on box 2):
    the unweighted terms equal bit for bit, level/L2 and level/L3 equal, and level/L1 difference = 900 x idm/L1 (1e-5 relative). The record is live_first, NEVER live.terms (the last of three AdamW steps)."""
    p = _first_problems(f"C3 IDM_s{init}", live_first) + _first_problems(f"C3 REF_s{init}", ref_first)
    if p:
        return p
    bad = [k for k in UNWEIGHTED if live_first[k] != ref_first[k]]
    if bad:
        p.append(f"C3: {len(bad)} unweighted live_first terms differ from REF_s{init}'s (first: {bad[0]} {live_first[bad[0]]!r} vs {ref_first[bad[0]]!r})")
    d, want = live_first["wm/level/L1"] - ref_first["wm/level/L1"], 900.0 * ref_first["wm/idm/L1"]
    if abs(d - want) > 1e-5 * abs(want):
        p.append(f"C3: level/L1 difference {d!r} != 900 x idm/L1 {want!r} (1e-5 relative)")
    return p


def ref_problems(init, terms, live_first):
    """AMEND_ACTSENS_1 A2, the REF gate: live.terms equal R35_s{k}'s banked terms on EVERY key, bit for bit (C1 on real data; also shows the NAMED default dose equals the absent flag); live_first present."""
    r, p = r35_terms(init), []
    if set(terms) != set(r):
        p.append(f"REF: live.terms keys differ from R35_s{init}'s: only here {sorted(set(terms) - set(r))[:3]}, only there {sorted(set(r) - set(terms))[:3]}")
    bad = [k for k in r if k in terms and terms[k] != r[k]]
    if bad:
        p.append(f"REF: {len(bad)} live.terms differ from R35_s{init}'s banked terms (first: {bad[0]} {terms[bad[0]]!r} vs {r[bad[0]]!r})")
    return p + _first_problems(f"REF_s{init}", live_first)


def base_tag(tag):
    return PILOT_BASE if tag == PILOT_TAG else tag


def is_pilot(tag):
    return tag == PILOT_TAG


def out_decl(tag):
    return PILOT_OUT_DECL + r"\DIV_s0" if is_pilot(tag) else frozen_argvs()[tag][frozen_argvs()[tag].index("--out") + 1]


def gate_out_decl(tag):
    return HOME_DECL + rf"\gate_out\{tag}"


def tokens(tag):
    """the shipped argv of a tag: the frozen tokens; the pilot = DIV_s0's with --out moved to the pilot dir and --steps 200 appended (nothing else changes)"""
    if tag != PILOT_TAG and tag not in TABLE:
        raise KeyError(tag)
    t = list(frozen_argvs()[base_tag(tag)])
    if is_pilot(tag):
        t[t.index("--out") + 1] = out_decl(tag)
        t += ["--steps", str(PILOT_STEPS)]
    return t


def audit_line(tag, line):
    """[] iff the shipped line equals the frozen argv of the tag token for token; else the first differing token, named"""
    want, got = tokens(tag), line.split(" ")
    if got == want:
        return []
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            return [f"{tag}: token {i} is {g!r}, the frozen argv has {w!r} (a changed token is refused)"]
    return [f"{tag}: the shipped argv has {len(got)} tokens, the frozen one {len(want)}"]


def out_of(toks):
    return toks[toks.index("--out") + 1]


def gate_argv(tag, toks):
    gt = list(toks)
    gt[gt.index("--out") + 1] = gate_out_decl(tag)
    gt.append("--selfcheck-only")
    return gt


def expected_steps(tag):
    return PILOT_STEPS if is_pilot(tag) else RUN_STEPS


# ---- BUILT-lever assertions (pure; the tests call it with planted selfchecks) -----------------------------------------------------------------------------------------------------
def gate_problems(tag, toks, sc, ref_sc=None):
    """Every assertion reads a key the ACTSENS trainer writes: declared_vs_built, argv_sha, steps_total, info.*, levers.{wm_only,wm_strides,wm_idm_gamma,wm_idm_gamma_readback,wm_sampler,wm_ladder_readback,params,w,recover,dagger_p,ego_perturb},
    live.{pass,missing_terms,dead_wm_groups,unexpected_dead,terms,control_adapter_grad_sum,control_nonwm_grad_sum,wm_groups_grad}, planner_identity_probe.{pass,n_tensors,n_equal}."""
    spec = TABLE[base_tag(tag)]
    init, lever, p = spec["init"], spec["lever"], []
    lv = sc.get("levers") if isinstance(sc.get("levers"), dict) else {}
    info = sc.get("info") if isinstance(sc.get("info"), dict) else {}

    def eq(name, got, want):
        if got != want:
            p.append(f"{name} = {got!r}, wanted {want!r}")
    if spec["blocked"]:
        p.append(f"{tag} is blocked: {spec['blocked']}")
    eq("declared_vs_built", sc.get("declared_vs_built"), "PASS")
    eq("selfcheck argv_sha (the trainer ran the pinned gate line)", sc.get("argv_sha"), argv_sha16(gate_argv(tag, toks)))
    eq("steps_total", sc.get("steps_total"), expected_steps(tag))
    for k, v in DATA.items():
        eq(f"info.{k}", info.get(k), v)
    eq("info.batch", info.get("batch"), 64)
    eq("info.md5_overridden", info.get("md5_overridden"), False)
    eq("info.excluded_in_train", info.get("excluded_in_train"), 0)
    for k, v in (("hmax", HMAX), ("wm", "b"), ("wm_idm_gamma", 100.0), ("wm_ch1_scale", 0.4), ("kin_mode", "yaw")):
        eq(f"built lever {k}", lv.get(k), v)
    eq("built w_wm", (lv.get("w") or {}).get("w_wm"), 0.1)
    # ---- the two ACTSENS levers READ BACK from the BUILT objects (levers.wm_idm_gamma_readback: the LevelConfigs the loss multiplies by; levers.wm_sampler + wm_sampler_probe: the BUILT sampler and card control 1)
    ir, smr = lv.get("wm_idm_gamma_readback"), lv.get("wm_sampler")
    if lever in ("idm", "ref"):
        want_g = [1000.0, 100.0, 100.0] if lever == "idm" else [100.0, 100.0, 100.0]
        if not isinstance(ir, dict):
            p.append("levers.wm_idm_gamma_readback is missing: the per-level IDM weight was not read back from the built stack")
        else:
            eq("wm_idm_gamma_readback.declared", ir.get("declared"), want_g)
            eq("wm_idm_gamma_readback.built (the weights the loss multiplies by, levels 1 / 2 / 3)", ir.get("built"), want_g)
    elif ir is not None:
        p.append(f"levers.wm_idm_gamma_readback = {ir!r} on a {lever} arm: the IDM flag must be absent")
    eq("built WMAux.gamma (lever wm_idm_gamma, the record value at every arm)", lv.get("wm_idm_gamma"), 100.0)
    rec = info.get("wm_sampler") if isinstance(info.get("wm_sampler"), dict) else {}
    if lever == "div":
        if not isinstance(smr, dict) or smr.get("kind") != "divrs":
            p.append(f"levers.wm_sampler = {smr!r}: the divrs sampler was not built")
        else:
            for k, v in (("yaw_edges", [0.02, 0.05, 0.10, 0.20]), ("acc_edges", [-1.0, -0.3, 0.3, 1.0]), ("cap_mult", 10.0), ("n_cells", 25)):
                eq(f"levers.wm_sampler.{k}", smr.get(k), v)
            eq("levers.wm_sampler.n_windows", smr.get("n_windows"), N_WINDOWS)
            if sum(smr.get("cell_counts") or []) != N_WINDOWS:
                p.append(f"levers.wm_sampler.cell_counts sum {sum(smr.get('cell_counts') or [])} != {N_WINDOWS} windows")
        sp = sc.get("wm_sampler_probe")
        if not isinstance(sp, dict) or sp.get("pass") is not True:
            p.append(f"wm_sampler_probe (card control 1 on the built sampler) did not pass: {str(sp)[:200]}")
    else:
        if smr is not None or sc.get("wm_sampler_probe") is not None:
            p.append(f"a {lever} arm carries levers.wm_sampler / wm_sampler_probe: the sampler flag must be absent")
        eq("info.wm_sampler.kind (the uniform sampler of every earlier arm)", rec.get("kind"), "uniform")
    eq("info.n_windows (R35's, MEASURED)", info.get("n_windows"), N_WINDOWS)
    eq("info.train_window_list_sha256 (R35's, MEASURED)", info.get("train_window_list_sha256"), WINDOW_LIST_SHA)
    # ---- the ladder levers READ BACK from the built model / dataset (levers.wm_ladder_readback)
    rb = lv.get("wm_ladder_readback")
    if not isinstance(rb, dict) or not isinstance(rb.get("built"), dict):
        return p + ["levers.wm_ladder_readback is missing: the ladder levers were not read back from the built model"]
    b = rb["built"]
    K, st, rk, rs = 1, (3, 5), (10, 10, 6), (5, 5, 1)
    eq("readback hmax", rb.get("hmax"), HMAX)
    eq("built rollout k per level", b.get("k"), list(rk))
    eq("built predictor context (pos rows) per level", b.get("ctx_built"), list(CTX))
    cum = [1, st[0], st[0] * st[1]]
    eq("built cumulative stride per level (strides)", b.get("cum_stride"), cum)
    eq("levers.wm_strides", lv.get("wm_strides"), list(st))
    states = b.get("states") or [None] * 3
    for i in range(3):
        eq(f"level {i + 1} anchors (stride {rs[i]})", (b.get("anchors") or [None] * 3)[i], len(range(0, (states[i] or 0) - 1, rs[i])) if states[i] else None)
        eq(f"level {i + 1} ctx handed to the free run", (b.get("ctx_passed") or [None] * 3)[i], CTX[i])
        w = (b.get("w_in_loss") or [None] * 3)[i]
        if not (finite(w) and abs(w - 1.0) <= 1e-4):
            p.append(f"level {i + 1} k-step weight in the built loss = {w!r}, wanted 1.0")
        if HMAX < rk[i] * cum[i] + 1:
            p.append(f"level {i + 1} reach {rk[i] * cum[i] + 1} frames exceeds the {HMAX}-frame window")
    wb = rb.get("windows_built") or {}
    for k, v in (("kind", "wmonly"), ("F", HMAX), ("stride", 1), ("pooled", True), ("t_min", HMAX - 1)):
        eq(f"built dataset {k}", wb.get(k), v)
    want_z = ZC_DECL + rf"\M2b_s{init}\train"
    if ntpath.normpath(str(wb.get("zcache", "")).replace("/", "\\")).lower() != ntpath.normpath(want_z).lower():
        p.append(f"built dataset reads the zcache {wb.get('zcache')!r}, wanted {want_z!r}")
    if wb.get("zcache_missing") not in (0, None):
        p.append(f"built dataset: {wb.get('zcache_missing')!r} clips have no zcache file")
    if not (num(wb.get("n_windows")) and wb["n_windows"] > 0):
        p.append(f"built dataset n_windows = {wb.get('n_windows')!r}")
    eq("built wm_k (absent at K = 1, as in R35)", lv.get("wm_k"), None)
    eq("built wm_k_levels (absent at K = 1)", lv.get("wm_k_levels"), None)
    # ---- wm-only: the trainable parameters are exactly the world model's: R35's count (neither lever adds a parameter)
    wo = lv.get("wm_only")
    if not isinstance(wo, dict) or wo.get("wm_only") is not True:
        p.append(f"levers.wm_only = {wo!r}: the WM-only mode was not built")
        wo = wo if isinstance(wo, dict) else {}
    eq("levers.wm_only.wm_k", wo.get("wm_k"), K)
    n = WM_PARAMS_BASE
    pr = lv.get("params") or {}
    if not (wo.get("n_wm") == n and wo.get("n_trainable") == n and lv.get("n_params_trainable") == n and pr.get("WM") == n):
        p.append(f"BUILT WM / trainable parameters = {wo.get('n_wm')!r}/{wo.get('n_trainable')!r}/{lv.get('n_params_trainable')!r} (params.WM {pr.get('WM')!r}), wanted {n} for all (R35's count: neither lever adds a parameter)")
    nz = {k: v for k, v in pr.items() if k != "WM" and v != 0}
    if nz or not pr:
        p.append(f"non-WM parameter groups are trainable: {nz or 'no params record'}")
    rec = info.get("wm_only") if isinstance(info.get("wm_only"), dict) else {}
    for k, v in (("sha256", init_sha(init)), ("applied", True), ("n_mismatch_after_load", 0), ("mode", "wm_only"), ("source_arm", "M2b"), ("source_seed", init)):
        eq(f"info.wm_only.{k}", rec.get(k), v)
    want_ic = M2B_DECL + rf"\M2b_s{init}\ckpt_last.pt"
    if ntpath.normpath(str(rec.get("path", "")).replace("/", "\\")).lower() != ntpath.normpath(want_ic).lower():
        p.append(f"info.wm_only.path {rec.get('path')!r} is not {want_ic}")
    eq("info.wm_only.wm_reinit_seed (the WM is re-initialised under the run's seed)", rec.get("wm_reinit_seed"), spec["seed"])
    # ---- G-LIVE
    lp = sc.get("live") if isinstance(sc.get("live"), dict) else {}
    eq("G-LIVE pass", lp.get("pass"), True)
    if lp.get("missing_terms") != [] or lp.get("dead_wm_groups") != [] or lp.get("unexpected_dead") != []:
        p.append(f"G-LIVE: missing {lp.get('missing_terms')!r} dead {lp.get('dead_wm_groups')!r} unexpected {lp.get('unexpected_dead')!r}")
    terms = lp.get("terms") if isinstance(lp.get("terms"), dict) else {}
    if not terms or any(not finite(v) for v in terms.values()) or not (finite(terms.get("wm")) and terms["wm"] > 0):
        p.append(f"G-LIVE: WM loss terms not finite and positive: {terms.get('wm')!r}")
    if not (finite(terms.get("wm/rollout/L1")) or finite(terms.get("wm/rollout"))):
        p.append("G-LIVE: no rollout term among the live loss terms although k > 1")
    if lever == "idm":                                                     # AMEND_ACTSENS_1 A1: C3 reads live_first (step 0) against REF_s{k}'s live_first, never live.terms (step 3)
        ref_first = (ref_sc or {}).get("live_first") if isinstance((ref_sc or {}).get("live_first"), dict) else None
        if ref_sc is None:
            p.append(f"C3: no {ref_tag(init)} gate output to compare with (the reference gate is missing or refused): the IDM arm is refused")
        else:
            p += c3_problems(init, sc.get("live_first"), ref_first)
    elif lever == "ref":
        p += ref_problems(init, terms, sc.get("live_first"))
    elif "live_first" in sc:
        p.append(f"a {lever} arm carries live_first: the IDM flag must be absent")
    if lp.get("control_adapter_grad_sum") != 0.0 or lp.get("control_nonwm_grad_sum") != 0.0:
        p.append(f"G-LIVE controls must read exactly 0.0: adapter {lp.get('control_adapter_grad_sum')!r}, non-WM {lp.get('control_nonwm_grad_sum')!r}")
    grp = lp.get("wm_groups_grad") if isinstance(lp.get("wm_groups_grad"), dict) else {}
    if not grp or any(not (num(g) and g > 0) for g in grp.values()):
        p.append("G-LIVE: a WM parameter group has zero gradient")
    pi = sc.get("planner_identity_probe") if isinstance(sc.get("planner_identity_probe"), dict) else {}
    if pi.get("pass") is not True or not (num(pi.get("n_tensors")) and pi["n_tensors"] > 0 and pi.get("n_equal") == pi.get("n_tensors")):
        p.append("planner identity probe did not pass")
    if (lv.get("recover") or {}).get("recover_p") != 0.0 or lv.get("dagger_p") != 0.0 or (lv.get("ego_perturb") or {}).get("ego_perturb_p") != 0.0:
        p.append("another lever (recover / dagger / ego-perturb) is on")
    return p


# ---- weights identity (the finished checkpoint against the M2b start state) --------------------------------------------------------------------------------------------------------
def load_ckpt(path):
    if fake() and os.environ.get("ACTSENS_CKPT_LOADER") == "pickle":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def same_bits(a, b):
    if hasattr(a, "detach"):
        import torch
        return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def weights_identity(init_sd, fin_sd):
    """-> (evidence, problems): every non-WM tensor of the finished checkpoint is bit-identical to the M2b start state (wm-only freezes them); the WM tensors exist"""
    na = {k: v for k, v in init_sd.items() if not k.startswith("wm.")}
    nb = {k: v for k, v in fin_sd.items() if not k.startswith("wm.")}
    p = []
    if not na:
        return {}, ["no non-WM tensor found in the init checkpoint (key names changed?)"]
    if set(na) != set(nb):
        p.append("non-WM tensor names differ from M2b's")
    differ = sorted(k for k in na if k in nb and not same_bits(na[k], nb[k]))
    if differ:
        p.append(f"{len(differ)} of {len(na)} non-WM tensor(s) differ from M2b's (first: {differ[:3]})")
    if not any(k.startswith("wm.") for k in fin_sd):
        p.append("no world-model tensor in the finished checkpoint")
    n_wm = sum(1 for k in fin_sd if k.startswith("wm."))
    if n_wm == 0:
        p.append("no world-model tensor in the finished checkpoint")
    return {"n_nonwm": len(na), "n_nonwm_equal": len(na) - len(differ), "n_wm_tensors": n_wm}, p


GAMMA_BETA, GAMMA_ROLLOUT_W, GAMMA_ACT_SIGREG, GAMMA_TOL = 0.08, 1.0, 0.005, 1e-4


def gamma_from_rows(rows):
    """card C2 (gamma_from_log.py): the BUILT IDM weight of each level recovered from every logged step by the loss identity level/L = pred + rollout_w x rollout + beta x sigreg (+ act_sigreg x asg at L >= 2) + gamma x idm."""
    g = {1: [], 2: [], 3: []}
    for r in rows:
        if "wm/level/L1" not in r:
            continue
        for L in (1, 2, 3):
            res = r[f"wm/level/L{L}"] - r[f"wm/pred/L{L}"] - GAMMA_ROLLOUT_W * r.get(f"wm/rollout/L{L}", 0.0) - GAMMA_BETA * r[f"wm/sigreg/L{L}"] - (GAMMA_ACT_SIGREG * r[f"wm/act_sigreg/L{L}"] if L >= 2 else 0.0)
            if r[f"wm/idm/L{L}"]:
                g[L].append(res / r[f"wm/idm/L{L}"])
    return g


def gamma_log_problems(tag, log_path):
    """-> (evidence, problems): IDM arms must read 1000 / 100 / 100 and every other arm 100 / 100 / 100, within 1e-4 relative at EVERY logged step."""
    want = {1: 1000.0 if TABLE[base_tag(tag)]["lever"] == "idm" else 100.0, 2: 100.0, 3: 100.0}
    try:
        rows = [json.loads(ln) for ln in open(log_path, encoding="utf-8") if ln.strip()]
    except (OSError, ValueError):
        return {}, ["train_log.jsonl unreadable for the gamma-from-log identity"]
    g = gamma_from_rows(rows)
    ev, p = {}, []
    for L in (1, 2, 3):
        v = g[L]
        ev[f"L{L}"] = {"n": len(v), "min": min(v) if v else None, "max": max(v) if v else None, "want": want[L]}
        if not v:
            p.append(f"gamma-from-log: no logged step carries the level-{L} loss terms")
        elif max(abs(x - want[L]) for x in v) > GAMMA_TOL * want[L]:
            p.append(f"gamma-from-log: level {L} recovered gamma in [{min(v):.6f}, {max(v):.6f}], wanted {want[L]} within {GAMMA_TOL:g} relative at every step")
    return ev, p


def adam_state_problems(blob):
    """card C5 (the probe_zerograd3 logic, read from the checkpoint's own optimiser): Adam allocates state lazily, so a parameter in an optimiser group with NO state never received a gradient. Every parameter of every group must have state."""
    opt = blob.get("opt") or {}
    ids = [i for g in opt.get("param_groups", []) for i in g.get("params", [])]
    st = opt.get("state") or {}
    miss = [i for i in ids if i not in st]
    ev = {"n_group_params": len(ids), "n_with_state": len(ids) - len(miss)}
    if not ids:
        return ev, ["no optimiser groups in the finished checkpoint: the zero-gradient probe cannot run"]
    return ev, ([f"{len(miss)} of {len(ids)} trainable parameters have NO Adam state (never received a gradient): first indices {miss[:5]}"] if miss else [])


def post_problems(tag, toks, out):
    """steps reached, checkpoint step, weights identity"""
    spec, p, ev = TABLE[base_tag(tag)], [], {}
    summ, ck_p = os.path.join(out, "summary.json"), os.path.join(out, "ckpt_last.pt")
    try:
        sm = json.load(open(summ, encoding="utf-8"))
    except (OSError, ValueError):
        return ev, ["summary.json unreadable"]
    if sm.get("steps") not in (None, expected_steps(tag)) or sm.get("steps_total") not in (None, expected_steps(tag)):
        p.append(f"summary steps {sm.get('steps')!r}/{sm.get('steps_total')!r} != {expected_steps(tag)}")
    if sm.get("argv_sha") not in (None, argv_sha16(toks)):
        p.append("summary argv_sha is not the shipped argv's")
    if is_pilot(tag):
        return ev, p
    gev, gp = gamma_log_problems(tag, os.path.join(out, "train_log.jsonl"))
    p += gp
    if not os.path.exists(ck_p):
        return ev, p + ["no ckpt_last.pt"]
    if fake() and os.environ.get("ACTSENS_SKIP_WEIGHTS") == "1":
        return ev, p
    ic = loc(M2B_DECL + rf"\M2b_s{spec['init']}\ckpt_last.pt")
    a, b = load_ckpt(ic), load_ckpt(ck_p)
    ev, fp = weights_identity(a["model"], b["model"])
    p += fp
    ev["gamma_from_log"] = gev
    ev2, fp2 = adam_state_problems(b)
    ev.update(ev2)
    p += fp2
    if b.get("step") not in (None, expected_steps(tag)):
        p.append(f"checkpoint step {b.get('step')!r}, wanted {expected_steps(tag)}")
    return ev, p


# ---- tree and data binding -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def tree_problems():
    p, c = [], commit()
    if not HEX40.match(c or ""):
        return [f"ACTSENS_COMMIT {c!r} is not a full 40-hex commit"]
    try:
        tip = open(os.path.join(repo(), "TIP_ACTSENS.txt"), encoding="utf-8").read().strip()
    except OSError:
        tip = None
    if tip != c:
        p.append(f"TIP_ACTSENS.txt reads {tip!r}, not ACTSENS_COMMIT {c} (the tree was not shipped at this commit)")
    try:
        man = json.load(open(os.path.join(home(), "tree_manifest.json"), encoding="utf-8"))
    except (OSError, ValueError) as e:
        return p + [f"tree_manifest.json unreadable: {type(e).__name__}"]
    if man.get("commit") != c:
        p.append(f"tree_manifest.json commit {man.get('commit')!r} != ACTSENS_COMMIT")
    mem = man.get("members") or {}
    bad = [rel for rel, want in sorted(mem.items()) if not os.path.exists(os.path.join(repo(), *rel.split("/"))) or sha_file(os.path.join(repo(), *rel.split("/"))) != want]
    if bad:
        p.append(f"{len(bad)} of {len(mem)} shipped files differ from the manifest (first: {bad[0]})")
    for rel, ns in NEEDLES.items():
        try:
            src = open(os.path.join(repo(), *rel.split("/")), encoding="utf-8", errors="replace").read()
        except OSError:
            p.append(f"{rel} unreadable")
            continue
        p += [f"the shipped {rel} lacks {n!r}" for n in ns if n not in src]
    return p


def data_paths():
    t = tokens("DIV_s0")
    g = lambda f: t[t.index(f) + 1]                                  # noqa: E731
    return {"v9_labels_train": g("--v9-train"), "lead_train": g("--lead-train"), "speed_train": g("--speed-train"), "join_train": g("--join-train"), "p10_prior": g("--p10"), "native_tracks": g("--tracks")}


def data_md5():
    out, bad = {}, []
    for k, pth in data_paths().items():
        f = loc(pth)
        if not os.path.isfile(f):
            bad.append(f"data file {k} missing: {pth}")
        else:
            out[k] = md5_file(f)
    want = os.environ.get("ACTSENS_FAKE_V9_MD5") if fake() and os.environ.get("ACTSENS_FAKE_V9_MD5") else V9_MD5
    if out.get("v9_labels_train") and out["v9_labels_train"] != want:
        bad.append(f"v9_labels_train md5 {out['v9_labels_train']} != the argv's --v9-train-md5")
    return out, bad


def disk_floor_problems():
    free = shutil.disk_usage(loc(DECL_D) if fake() else "D:\\").free
    floor = int(float(os.environ.get("ACTSENS_FLOOR_BYTES", "2e9"))) if fake() else int(2e9)
    return [f"disk floor: {free} bytes free < {floor}"] if free < floor else []


def run_dir(tag):
    return os.path.join(home(), "runs", tag)


def token_path(tag):
    return os.path.join(run_dir(tag), "gate", "gate_token.json")


def trainer_path():
    return os.path.join(repo(), *TRAINER_REL.split("/"))


def binding(tag, line):
    """everything the token binds, recomputed NOW"""
    md5s, bad = data_md5()
    try:
        tm = sha_file(os.path.join(home(), "tree_manifest.json"))
    except OSError:
        tm = None
    return {"tag": tag, "commit": commit(), "argv_sha256": tokens_sha(line.split(" ")), "data_md5": md5s, "tree_manifest_sha256": tm, "trainer_sha256": sha_file(trainer_path()) if os.path.exists(trainer_path()) else None}, bad


def token_ok(tag, line):
    """[] iff a token written by `gate` still binds this tag, commit, argv, data and shipped bytes"""
    try:
        tok = json.load(open(token_path(tag), encoding="utf-8"))
    except (OSError, ValueError):
        return ["no gate token"]
    want, bad = binding(tag, line)
    p = list(bad)
    if tok.get("written_by") != "actsens_check.py gate":
        p.append("token not written by the checker")
    for k, v in want.items():
        if tok.get(k) != v:
            p.append(f"token {k} differs from the current {k}")
    sc_p = os.path.join(run_dir(tag), "gate", "selfcheck.json")
    if not os.path.exists(sc_p) or sha_file(sc_p) != tok.get("selfcheck_sha256"):
        p.append("selfcheck.json differs from the token's")
    return p


# ---- trainer execution -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def poll_s():
    return float(os.environ.get("ACTSENS_POLL_S", "5"))


def gpu_sample():
    try:
        r = subprocess.run([os.environ.get("ACTSENS_NVSMI", "nvidia-smi"), "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, encoding="utf-8", errors="replace", timeout=20)
        a = r.stdout.strip().splitlines()[0].split(",")
        return float(a[0]), float(a[1])
    except Exception:                                                                    # noqa: BLE001  a missing sampler is recorded as no sample, never as a number
        return None


def run_trainer(toks, log_path, sample=False):
    """-> (rc, wall_s, gpu_samples). The trainer's output goes to log_path; the sampler only reads nvidia-smi."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0, samples = time.time(), []
    with open(log_path, "ab") as lg:
        lg.write(f"===== RUN-START {now()}\n".encode())
        pr = subprocess.Popen([sys.executable, trainer_path(), *toks], stdout=lg, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        while pr.poll() is None:
            if sample:
                g = gpu_sample()
                if g:
                    samples.append(g)
            time.sleep(poll_s())
        lg.write(f"===== RUN-END rc={pr.returncode}\n".encode())
    return pr.returncode, time.time() - t0, samples


def read_log(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if isinstance(r, dict) and isinstance(r.get("step"), int) and num(r.get("sec")):
                    recs.append((r["step"], float(r["sec"])))
    return recs


def time_pilot(log_path):
    """mean and median s/step over steps 50..200 from the cumulative `sec` column (start-up is inside the first 50 steps and excluded); ValueError when not one uninterrupted run"""
    recs = read_log(log_path)
    if not recs:
        raise ValueError("the pilot log holds no step/sec record")
    for (s0, t0), (s1, t1) in zip(recs, recs[1:]):
        if not (s1 > s0 and t1 > t0):
            raise ValueError(f"the pilot log is not one uninterrupted run (step {s0}->{s1}, sec {t0:.3f}->{t1:.3f})")
    by = dict(recs)
    pts = [s for s, _ in recs if 50 <= s <= PILOT_STEPS]
    if not pts or pts[0] != 50 or pts[-1] != PILOT_STEPS or len(pts) < 4:
        raise ValueError(f"the pilot log needs records at step 50 and {PILOT_STEPS} and at least 4 in between (saw {pts})")
    per = [(by[b] - by[a]) / (b - a) for a, b in zip(pts, pts[1:])]
    return {"mean_s_per_step": (by[PILOT_STEPS] - by[50]) / (PILOT_STEPS - 50), "median_s_per_step": statistics.median(per), "interval_s_per_step": per, "startup_s_first_50_steps": by[50], "interval_steps": pts}


def project(mean_s):
    """planned = the runs in RUN_ORDER (the 72 h basis). Every run is taken at the pilot's DIV_s0 step time (IDM and NULL have the same graph; DIV adds only a start-up the pilot also measures)."""
    n = len(RUN_ORDER)
    planned = mean_s * RUN_STEPS * n / 3600
    return {"n_runs_planned": n, "run_steps": RUN_STEPS, "steps_total": RUN_STEPS * n, "projected_total_h_planned": planned, "per_run_h": mean_s * RUN_STEPS / 3600, "limit_h": LIMIT_H, "within_limit_planned": planned <= LIMIT_H}


def classify(samples):
    util = [u for _, u in samples]
    if not util:
        return "unmeasured", None
    med = statistics.median(util)
    return ("compute-bound" if med >= 60 else "loader-bound"), med


# ---- the box-2 job -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def stage_fail(tag, gid, kind, why):
    write_text(os.path.join(run_dir(tag), f"STAGE_{gid}.fail"), f"{kind}: {why}"[:900] + "\n")
    print(f"ACTSENS-FAIL {tag} {kind}: {why}"[:1200])
    return {"preflight-fail": 3, "pin-fail": 3, "gate-fail": 4, "token-fail": 4, "train-fail": 5, "post-fail": 6, "disk-low": 9}.get(kind, 5)


def rm_own_temp(pth, parent_name):
    """delete one of this job's own temporary dirs by literal path; refuses anything not directly under a folder named parent_name. -> {path, bytes} or None"""
    if not os.path.isdir(pth) or os.path.basename(os.path.dirname(pth)) != parent_name:
        return None
    size = sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(pth) for f in fs)
    shutil.rmtree(pth)
    return {"path": pth, "bytes": size}


def do_gate(tag, line):
    """selfcheck-only run, BUILT-lever assertions, token. -> list of problems (empty = token written)"""
    toks = line.split(" ")
    rd = run_dir(tag)
    gd = os.path.join(rd, "gate")
    ref_sc, ref_sha = None, None
    if TABLE[base_tag(tag)]["lever"] == "idm":                              # AMEND_ACTSENS_1: the REF gate of the same chassis must have passed on THIS tree; its selfcheck is what C3 reads
        rt = ref_tag(TABLE[base_tag(tag)]["init"])
        rp = token_ok(rt, " ".join(tokens(rt)))
        rsp = os.path.join(run_dir(rt), "gate", "selfcheck.json")
        if rp or not os.path.exists(rsp):
            return [f"C3: the reference gate {rt} is missing or refused ({'; '.join(rp)[:300] or 'no selfcheck'}): the IDM arm is refused"]
        ref_sc, ref_sha = json.load(open(rsp, encoding="utf-8")), sha_file(rsp)
    if not token_ok(tag, line):
        print(f"gate-cached {tag}")
        return []
    if os.path.exists(gd):
        shutil.move(gd, gd + ".prev_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(gd)
    gout = loc(gate_out_decl(tag))
    rm_own_temp(gout, "gate_out")                                           # the selfcheck-only output of an earlier attempt: own temporary data, re-created by this run
    rc, _, _ = run_trainer(gate_argv(tag, toks), os.path.join(rd, "logs", "gate.log"))
    scp = os.path.join(gout, "selfcheck.json")
    if rc != 0 or not os.path.exists(scp):
        return [f"the trainer's --selfcheck-only exited rc={rc} and {'wrote' if os.path.exists(scp) else 'did not write'} selfcheck.json (log {rd}\\logs\\gate.log)"]
    shutil.copy2(scp, os.path.join(gd, "selfcheck.json"))
    sc = json.load(open(scp, encoding="utf-8"))
    bad = gate_problems(tag, toks, sc, ref_sc)
    if bad:
        write_text(os.path.join(rd, "RUN.FAILED"), json.dumps({"tag": tag, "at": now(), "problems": bad}, indent=1))
        return bad
    b, dbad = binding(tag, line)
    if dbad:
        return dbad
    write_text(token_path(tag), json.dumps({"schema": "actsens-gate-token-v1", "written_by": "actsens_check.py gate", "at": now(), "selfcheck_sha256": sha_file(os.path.join(gd, "selfcheck.json")), **({"c3_ref_selfcheck_sha256": ref_sha} if ref_sha else {}), **b}, indent=1))
    return []


def cmd_stage(tag, gid):
    if tag != PILOT_TAG and tag not in TABLE:
        return stage_fail(tag, gid, "pin-fail", f"unknown tag {tag}")
    if TABLE.get(tag, {}).get("blocked"):
        return stage_fail(tag, gid, "pin-fail", f"{tag} is blocked: {TABLE[tag]['blocked']}")
    if TABLE.get(tag, {}).get("lever") == "ref":
        return stage_fail(tag, gid, "pin-fail", f"{tag} is a gate-only reference: it never trains and gets no run token")
    rd = run_dir(tag)
    os.makedirs(rd, exist_ok=True)
    try:
        line = [l.strip() for l in open(os.path.join(rd, "argv.txt"), encoding="utf-8") if l.strip()]
    except OSError:
        return stage_fail(tag, gid, "pin-fail", "no shipped argv.txt")
    if len(line) != 1:
        return stage_fail(tag, gid, "pin-fail", f"argv.txt holds {len(line)} lines")
    line = line[0]
    bad = disk_floor_problems()
    if bad:
        return stage_fail(tag, gid, "disk-low", bad[0])
    bad = tree_problems()
    ic = loc(M2B_DECL + rf"\M2b_s{TABLE[base_tag(tag)]['init']}\ckpt_last.pt")
    if not os.path.exists(ic) or sha_file(ic) != init_sha(TABLE[base_tag(tag)]["init"]):
        bad.append(f"start state M2b_s{TABLE[base_tag(tag)]['init']} missing or its sha256 differs from the constant")
    zc_rel = "\\M2b_s%d\\train" % TABLE[base_tag(tag)]["init"]
    if not os.path.isdir(loc(ZC_DECL + zc_rel)):
        bad.append("the latent cache " + ZC_DECL + zc_rel + " is missing on this machine")
    if bad:
        return stage_fail(tag, gid, "preflight-fail", " | ".join(bad)[:800])
    bad = audit_line(tag, line)
    if bad:
        return stage_fail(tag, gid, "pin-fail", bad[0])
    jobs = [(tag, line)]
    if not is_pilot(tag) and TABLE[tag]["lever"] == "idm":     # AMEND_ACTSENS_1 A2: the reference gate of the chassis runs on this tree BEFORE the IDM gate (cached if the pilot job already passed it)
        rt = ref_tag(TABLE[tag]["init"])
        os.makedirs(run_dir(rt), exist_ok=True)
        rfull = " ".join(tokens(rt))
        write_text(os.path.join(run_dir(rt), "argv.txt"), rfull + "\n")
        jobs = [(rt, rfull)] + jobs
    if is_pilot(tag):                                          # the pilot also gates the FULL-length argvs of the first runs, so steps_total is read from a real selfcheck before the 72 h decision
        for t in GATE_AT_PILOT:
            os.makedirs(run_dir(t), exist_ok=True)
            full = " ".join(tokens(t))
            write_text(os.path.join(run_dir(t), "argv.txt"), full + "\n")
            jobs.append((t, full))
    for t, ln in jobs:
        bad = do_gate(t, ln)
        if bad:
            return stage_fail(tag, gid, "gate-fail", f"{t}: " + " | ".join(bad)[:800])
    bad = token_ok(tag, line)
    if bad:
        return stage_fail(tag, gid, "token-fail", bad[0])
    toks = line.split(" ")
    out = loc(out_of(toks))
    if is_pilot(tag):
        rm_own_temp(out, "pilot_out")                          # a pilot always starts from an empty out dir (a leftover is the pilot's own temporary data)
    rc, wall, samples = run_trainer(toks, os.path.join(rd, "logs", "train.log"), sample=is_pilot(tag))
    summ, logp = os.path.join(out, "summary.json"), os.path.join(out, "train_log.jsonl")
    if rc != 0 or not (os.path.exists(summ) and os.path.exists(logp)):
        return stage_fail(tag, gid, "train-fail", f"trainer rc={rc}; summary {os.path.exists(summ)}; log {os.path.exists(logp)}")
    ev, p = post_problems(tag, toks, out)
    if p:
        return stage_fail(tag, gid, "post-fail", " | ".join(p)[:800])
    if is_pilot(tag):
        return finish_pilot(tag, gid, line, out, wall, samples)
    ck = os.path.join(out, "ckpt_last.pt")
    write_text(os.path.join(rd, "RUN.done"), json.dumps({"tag": tag, "at": now(), "commit": commit(), "weights_identity": ev,
                                                         "evidence": {"ckpt_sha256": sha_file(ck), "summary_sha256": sha_file(summ), "train_log_sha256": sha_file(logp)}}, indent=1))
    write_text(os.path.join(rd, f"STAGE_{gid}.done"), "done\n")
    print(f"ACTSENS-OK {tag}")
    return 0


def finish_pilot(tag, gid, line, out, wall, samples):
    rd, pd = run_dir(tag), os.path.join(home(), "pilot")
    os.makedirs(pd, exist_ok=True)
    try:
        tm = time_pilot(os.path.join(out, "train_log.jsonl"))
    except ValueError as e:
        return stage_fail(tag, gid, "post-fail", str(e))
    for f in ("train_log.jsonl", "summary.json"):
        shutil.copy2(os.path.join(out, f), os.path.join(pd, f))
    shutil.copy2(os.path.join(rd, "gate", "selfcheck.json"), os.path.join(pd, "selfcheck.json"))
    shutil.copy2(token_path(tag), os.path.join(pd, "gate_token.json"))
    shutil.copy2(os.path.join(run_dir(ref_tag(0)), "gate", "selfcheck.json"), os.path.join(pd, "ref_selfcheck.json"))     # AMEND_ACTSENS_1: REF_s0's gate output (C1 on real data, and what IDM_s0's C3 was read against)
    bound, util = classify(samples)
    res = {"schema": "actsens-pilot-v1", "at": now(), "commit": commit(), "tag": tag, "argv_sha256": tokens_sha(line.split(" ")), "pilot_steps": PILOT_STEPS, "wall_s_total_incl_startup": wall, "timing": tm,
           "peak_gpu_mib": max((m for m, _ in samples), default=None), "n_gpu_samples": len(samples), "gpu_util_median_pct": util, "limit_by": bound,
           "train_log_sha256": sha_file(os.path.join(pd, "train_log.jsonl")), "projection": project(tm["mean_s_per_step"]),
           "full_run_gates_at_pilot": {t: os.path.exists(token_path(t)) for t in GATE_AT_PILOT}}
    write_text(os.path.join(pd, "PILOT.json"), json.dumps(res, indent=1))
    write_text(os.path.join(rd, f"STAGE_{gid}.done"), "done\n")
    print(f"ACTSENS-OK {tag} mean {tm['mean_s_per_step']:.4f} s/step, {res['projection']['projected_total_h_planned']:.1f} h planned")
    return 0


def cmd_cleanup(tag, gid):
    """box 2, AFTER the dev box pulled and sha-verified the bank and put BANK_VERIFIED_<gid>: delete this run's own temporary dirs by literal path; log (path, bytes). Never a checkpoint, never another run's folder."""
    rd = run_dir(tag)
    mark = os.path.join(rd, f"BANK_VERIFIED_{gid}")
    if not os.path.exists(mark):
        write_text(os.path.join(rd, f"CLEAN_{gid}.fail"), "no BANK_VERIFIED marker: the bank was not proven, nothing is deleted\n")
        print("ACTSENS-CLEAN-REFUSED no bank proof")
        return 6
    del_ = []
    for t in ([tag] + GATE_AT_PILOT) if is_pilot(tag) else ([tag, ref_tag(TABLE[tag]["init"])] if TABLE.get(tag, {}).get("lever") == "idm" else [tag]):
        d = rm_own_temp(loc(gate_out_decl(t)), "gate_out")
        if d:
            del_.append(d)
    if is_pilot(tag):
        d = rm_own_temp(loc(out_decl(tag)), "pilot_out")
        if d:
            del_.append(d)
    write_text(os.path.join(rd, "CLEANUP.json"), json.dumps({"tag": tag, "at": now(), "deleted_own_temp": del_}, indent=1))
    write_text(os.path.join(rd, f"CLEAN_{gid}.done"), "done\n")
    print(f"ACTSENS-CLEAN-OK {tag} deleted {len(del_)}")
    return 0


# ---- dev-box commands -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cmd_verify_token(a):
    tok = json.load(open(a["token"], encoding="utf-8"))
    line = [l.strip() for l in open(a["argv-file"], encoding="utf-8") if l.strip()][0]
    p = []
    if tok.get("written_by") != "actsens_check.py gate":
        p.append("not written by the checker")
    if tok.get("tag") != a["tag"] or tok.get("commit") != a["commit"] or tok.get("argv_sha256") != tokens_sha(line.split(" ")):
        p.append("token does not bind this tag, commit and argv")
    if audit_line(a["tag"], line):
        p.append("the argv is not the frozen one")
    if p:
        print("token refused: " + "; ".join(p))
        return 1
    print("token ok")
    return 0


def cmd_project(a):
    pd = a["pilot-dir"]
    res = json.load(open(os.path.join(pd, "PILOT.json"), encoding="utf-8"))
    tm = time_pilot(os.path.join(pd, "train_log.jsonl"))
    p = []
    if res.get("commit") != a["commit"]:
        p.append(f"the pilot timed commit {res.get('commit')}")
    if sha_file(os.path.join(pd, "train_log.jsonl")) != res.get("train_log_sha256"):
        p.append("train_log sha256 differs from PILOT.json")
    if abs(tm["mean_s_per_step"] - res["timing"]["mean_s_per_step"]) > 1e-9:
        p.append("mean s/step recomputed here differs from box 2's")
    pr = project(tm["mean_s_per_step"])
    if abs(pr["projected_total_h_planned"] - res["projection"]["projected_total_h_planned"]) > 1e-6:
        p.append("projection differs")
    if p:
        print("pilot refused: " + "; ".join(p))
        return 18
    res["verified_on_dev_box"] = {"at": now(), "mean_s_per_step": tm["mean_s_per_step"], "projection": pr}
    write_text(a["out"], json.dumps(res, indent=1))
    print(f"pilot verified: mean {tm['mean_s_per_step']:.4f} s/step; {pr['n_runs_planned']} runs {pr['projected_total_h_planned']:.1f} h; limit {LIMIT_H:.0f} h")
    return 0


def cmd_permit(a):
    res = json.load(open(a["pilot"], encoding="utf-8"))
    if res.get("commit") != a["commit"]:
        print(f"refused: PILOT.json is for commit {res.get('commit')}, not {a['commit']}")
        return 3
    total = (res.get("verified_on_dev_box") or {}).get("projection", res["projection"])["projected_total_h_planned"]
    if not finite(total):
        print("refused: the projection is not a number")
        return 3
    if total <= LIMIT_H:
        print(f"permit: projected {total:.1f} h <= {LIMIT_H:.0f} h")
        return 0
    am = a.get("amendment")
    if not am or not os.path.isfile(am) or os.path.getsize(am) == 0:
        print(f"refused: projected {total:.1f} h > {LIMIT_H:.0f} h and no amendment file (the card sends this case to the PI)")
        return 3
    txt = open(am, encoding="utf-8", errors="replace").read()
    psha = sha_file(a["pilot"])
    if f"pilot_sha256={psha}" not in txt or f"commit={a['commit']}" not in txt:
        print(f"refused: the amendment {am} must contain 'pilot_sha256={psha}' and 'commit={a['commit']}'")
        return 3
    print(f"permit: projected {total:.1f} h > {LIMIT_H:.0f} h, amendment {am} binds this pilot and commit")
    return 0


def cmd_render(a):
    t = open(a["template"], encoding="utf-8").read()
    for _ in range(3):                                           # a value may itself carry @ROOT@
        for k, v in a["set"].items():
            t = t.replace(f"@{k}@", v)
    left = re.findall(r"@[A-Z_]+@", t)
    if left:
        print(f"render: unfilled placeholders {sorted(set(left))}")
        return 2
    with open(a["out"], "w", encoding="utf-8", newline="") as f:
        f.write(t.replace("\r\n", "\n").replace("\n", "\r\n"))
    return 0


def parse(argv):
    a, i, sets = {}, 0, {}
    while i < len(argv):
        x = argv[i]
        if x == "--set":
            k, v = argv[i + 1].split("=", 1)
            sets[k] = v
            i += 2
        elif x.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                a[x[2:]], i = argv[i + 1], i + 2
            else:
                a[x[2:]], i = True, i + 1
        else:
            i += 1
    a["set"] = sets
    return a


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, a = argv[0], parse(argv[1:])
    if cmd == "argv":
        tag = PILOT_TAG if a.get("pilot") else a.get("tag")
        if tag != PILOT_TAG and tag not in TABLE:
            print("unknown tag")
            return 2
        if TABLE.get(tag, {}).get("blocked"):
            print(f"REFUSED: {tag} is blocked: {TABLE[tag]['blocked']}")
            return 23
        line = " ".join(tokens(tag))
        if "UNVERIFIED" in line:
            print("REFUSED: the argv carries a placeholder path")
            return 23
        if a.get("out-txt"):
            write_text(a["out-txt"], line + "\n")
        else:
            print(line)
        return 0
    if cmd == "render":
        return cmd_render(a)
    if cmd == "preflight":
        bad = disk_floor_problems() + tree_problems()
        print("preflight OK" if not bad else "ACTSENS-PREFLIGHT-FAIL " + " | ".join(bad))
        return 0 if not bad else 3
    if cmd == "stage":
        return cmd_stage(a["tag"], a["gid"])
    if cmd == "cleanup":
        return cmd_cleanup(a["tag"], a["gid"])
    if cmd == "verify-token":
        return cmd_verify_token(a)
    if cmd == "project":
        return cmd_project(a)
    if cmd == "permit":
        return cmd_permit(a)
    print(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
