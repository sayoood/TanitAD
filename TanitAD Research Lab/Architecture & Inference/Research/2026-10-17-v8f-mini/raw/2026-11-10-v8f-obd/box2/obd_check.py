"""OBD launch-gate checker. Runs on box 2 (venv python; stdlib, numpy and torch only in the moved-identity load) and on the dev box.
Pattern: the OTBAND checker (theory/otband/box2/otband_check.py, itself the ladder checker): pin the argv, run the trainer with --selfcheck-only at the shipped commit, assert what it BUILT, write a per-run token (only here, never by hand),
train under the token, post-check, bank.   Card: theory/obd/PREREG_V8F_OBD_DRAFT.md (draft). Frozen argvs: frozen/argvs_obd.json (sha256 pinned below). Code under test: commit 202849da013b5d74c0b0086dd9865fcda6836018 or its successor
(the successor is REQUIRED while the purity probe does not read the acceleration term: PURITY_ACC_FINDING.md).

  argv --tag T [--pilot] [--out-txt F]    dev box: the frozen argv of tag T (pilot: --out moved to the pilot dir, --steps 200 appended), one line
  render --template F --out F --set K=V   fill @K@ placeholders, write with CRLF line ends
  preflight                               disk floor, tree bound to OBD_COMMIT (TIP_OBD.txt, manifest members re-hashed, needles), start states and data files present
  stage --tag T --gid G [--pilot]         ONE box-2 job: preflight, pin, gate (selfcheck + BUILT-lever assertions + token), train, post; markers STAGE_<G>.done/.fail
  cleanup --tag T --gid G                 box 2: delete the run's own temporary gate dirs by literal path, ONLY after the dev box proved the bank (BANK_VERIFIED_<G>); never a checkpoint
  verify-token --token F --argv-file F --tag T --commit C
  project --pilot-dir D --out F --commit C   dev box: PILOT.json re-derived from the pulled pilot log
  permit --pilot F --commit C [--amendment F]   the 12 h rule over the seven planned runs
  parity --bank-ckpt F --sums F --out F [--dagoacc-ckpt F] [--basis file|tensors]   dev box: NULL parity of OBD-NULL_s0 against DAG-OACC s0 (card, Controls item 1) -> NULL_PARITY.json
Test hook (dev box only): OBD_ROOT maps every D:\\tanitad path to <root>; OBD_FAKE_INIT_SHA_FILE / OBD_FAKE_BANK_SHA_FILE replace the start-state / DAgger-bank constants; OBD_POLL_S sets the sampler period."""
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
ARGVS_SHA256 = "346a48ed7b070fffacd22bf280e72c42831126fb906c231a7ac24301e624a424"        # frozen/argvs_obd.json == theory/obd/argvs_obd.json (make_argvs_obd.py, MEASURED 2026-11-10)
ARGVS_FILE = os.path.join(HERE, "argvs_obd.json") if os.path.exists(os.path.join(HERE, "argvs_obd.json")) else os.path.join(HERE, "frozen", "argvs_obd.json")
#: M2b start states: read from DAG-OACC's own selfcheck.json (v8f-dagoacc/M2b_s{k}_dagoacc/selfcheck.json, info.init_ckpt.sha256) and the card; same files, same corpus, same exclusion.
INIT_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
#: DAgger banks (sha256 of bank.json): DAG-OACC's argv.json (--dagger-bank-sha256), equal to its selfcheck levers.dagger_bank_sha256 (MEASURED from those files).
BANK_SHA = {0: "197a68bb540be5a7d157942e9157e1c8e60ca60d82b1ca4fb4fcb68edccfaa8f", 1: "36c7aea964dafe9423253aa479c2fc6cdfc93848395b593315cba812f5f04d4d"}
V9_MD5 = "83e30bd3791f0767707eade5f4671759"
DATA = {"train_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1", "excluded_sha256": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "n_clips": 4512, "n_excluded": 60}
#: DAG-OACC's selfcheck (both seeds, MEASURED from the files): S steps per epoch 4791, steps_total 3000, trainable parameters per group (NO group is frozen), p_gt 0.2, w_acc 49.6969, DAgger p 0.1.
S_PER_EPOCH = 4791
RUN_STEPS, PILOT_STEPS, LIMIT_H = 3000, 200, 12.0
PARAMS = {"adapter": 275712, "cond": 25088, "S": 1321737, "T": 1463868, "O": 2120216, "WM": 2007874}
N_PARAMS_TRAINABLE = 7214495
W_ACC, DAGGER_P, P_GT = 49.6969, 0.1, 0.2
#: the DAG-OACC s0 checkpoint sha256 the NULL parity control compares with (v8f-dagoacc/M2b_s0_dagoacc/SHA256SUMS.txt, line `*ckpt_last.pt`)
DAGOACC_S0_CKPT_SHA = "67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f"
#: the launch purity probe must read the acceleration term (card, Controls item 3). At 202849da it does not (PURITY_ACC_FINDING.md), so with this True every CTL / BAND gate refuses until a commit with the probe change is shipped.
ACC_PROBE_REQUIRED = True
#: DISK REQUIREMENT (a stated constant, nothing measured): box 2's D: read 11 GB free recently. One run writes ckpt_last.pt (model + AdamW state, ESTIMATED <= 1.5 GB) plus gate outputs (< 0.1 GB);
#: seven (eight) checkpoints stay on box 2 after banking. A run starts only with NEED_GB_RUN free; the tree ship needs NEED_GB_SHIP. Below it the chain stops with exit 6 (stage rc 13).
NEED_GB_RUN, NEED_GB_SHIP = 5, 1
DECL_D = r"D:\tanitad"
HOME_DECL, REPO_DECL, M2B_DECL = DECL_D + r"\obd", DECL_D + r"\obd\repo", DECL_D + r"\runs\v8f_mini_full"
BANK_DECL = DECL_D + r"\runs\v8f_mini_full\dagger_bank"
PILOT_OUT_DECL = HOME_DECL + r"\pilot_out"
TRAINER_REL, MODEL_REL, DATA_REL = "stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py"
NEEDLES = {TRAINER_REL: ('"--w-op"', '"--w-tac-traj"', '"--w-acc"', '"--dagger-p"', '"--detach-upper"', "def bad_oband", "def oband_probe", "def acc_probe", 'keys = (*keys, "acc")'),
           MODEL_REL: ("w_tac_traj", "def accel_term"), DATA_REL: ("def attach_stok",)}
HEX64, HEX40 = re.compile(r"^[0-9a-f]{64}$"), re.compile(r"^[0-9a-f]{40}$")
#: tag table. kind null = DAG-OACC's recipe unchanged; ctl = + --detach-upper; band = + --detach-upper and the band loss. init = the M2b seed index (and the DAgger bank / --dagger-seed); seed = the training seed.
#: order None = not in the planned seven (OBD-NULL_s1 is the CONDITIONAL arm: the chain appends it only when the NULL parity control fails).
TABLE = {
    "OBD-NULL_s0":  {"kind": "null", "seed": 0, "init": 0, "order": 1, "blocked": None},
    "OBD-NULL_rep": {"kind": "null", "seed": 2, "init": 0, "order": 2, "blocked": None},
    "OBD-CTL_s0":   {"kind": "ctl",  "seed": 0, "init": 0, "order": 3, "blocked": None},
    "OBD-BAND_s0":  {"kind": "band", "seed": 0, "init": 0, "order": 4, "blocked": None},
    "OBD-CTL_s1":   {"kind": "ctl",  "seed": 1, "init": 1, "order": 5, "blocked": None},
    "OBD-BAND_s1":  {"kind": "band", "seed": 1, "init": 1, "order": 6, "blocked": None},
    "OBD-BAND_rep": {"kind": "band", "seed": 2, "init": 0, "order": 7, "blocked": None},
    "OBD-NULL_s1":  {"kind": "null", "seed": 1, "init": 1, "order": None, "blocked": None, "conditional": True},
}
PILOT_TAG, PILOT_BASE = "PILOT_OBD-BAND_s0", "OBD-BAND_s0"
CONDITIONAL_TAG, PARITY_TAG = "OBD-NULL_s1", "OBD-NULL_s0"
W_OP, W_TAC = 0.3333333, 0.6666667
RUN_ORDER = sorted((t for t, v in TABLE.items() if v["order"] and not v["blocked"]), key=lambda t: TABLE[t]["order"])
GATE_AT_PILOT = ["OBD-BAND_s0", "OBD-CTL_s0", "OBD-NULL_s0"]          # the pilot job also gates these full-length argvs, so the first runs start under a token and steps_total is read before the 12 h decision
MOVED_PREFIXES = {"S": ("read_s", "head_s"), "T": ("read_t", "trunk_t", "h_goal", "h_lat", "h_lon", "h_cons"), "O": ("read_o", "head_o")}      # tensors that must move in every arm (nothing is frozen)

# ---- environment ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fake():
    return os.environ.get("OBD_ROOT")


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
    return os.environ.get("OBD_COMMIT", "")


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
    if fake() and os.environ.get("OBD_FAKE_INIT_SHA_FILE"):
        return json.load(open(os.environ["OBD_FAKE_INIT_SHA_FILE"]))[str(s)]
    return INIT_SHA[s]


def bank_sha(s):
    if fake() and os.environ.get("OBD_FAKE_BANK_SHA_FILE"):
        return json.load(open(os.environ["OBD_FAKE_BANK_SHA_FILE"]))[str(s)]
    return BANK_SHA[s]


def bank_file(s):
    return loc(BANK_DECL + rf"\s{s}\bank.json")


def bank_problems(init):
    """the DAgger bank file on box 2 hashes to DAG-OACC's (the trainer re-hashes it too: --dagger-bank-sha256)"""
    f = bank_file(init)
    if not os.path.isfile(f):
        return [f"DAgger bank s{init} missing: {f}"]
    got = sha_file(f)
    return [] if got == bank_sha(init) else [f"DAgger bank s{init} sha256 {got} != DAG-OACC's {bank_sha(init)}"]


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
    if got != ARGVS_SHA256 and not os.environ.get("OBD_TEST_ARGVS"):
        raise SystemExit(f"REFUSED: {ARGVS_FILE} sha256 {got} != the pinned {ARGVS_SHA256} (the frozen argv file changed)")
    return {k: v["argv"] for k, v in json.load(open(ARGVS_FILE, encoding="utf-8")).items()}


def base_tag(tag):
    return PILOT_BASE if tag == PILOT_TAG else tag


def is_pilot(tag):
    return tag == PILOT_TAG


def out_decl(tag):
    return PILOT_OUT_DECL + r"\OBD-BAND_s0" if is_pilot(tag) else frozen_argvs()[tag][frozen_argvs()[tag].index("--out") + 1]


def gate_out_decl(tag):
    return HOME_DECL + rf"\gate_out\{tag}"


def tokens(tag):
    """the shipped argv of a tag: the frozen tokens; the pilot = OBD-BAND_s0's with --out moved to the pilot dir and --steps 200 appended (nothing else changes)"""
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
def flag_value(toks, name):
    """the single value after `name` in an argv token list, or None"""
    return toks[toks.index(name) + 1] if name in toks and toks.index(name) + 1 < len(toks) else None


def purity_keys(kind):
    """the O terms of an arm's own loss whose purity the launch probe must read (card, Controls item 3): ade / fde without a band weight, op / tac with one; the acceleration term when ACC_PROBE_REQUIRED"""
    base = ("op", "tac") if kind == "band" else ("ade", "fde")
    return (*base, "acc") if ACC_PROBE_REQUIRED else base


def gate_problems(tag, toks, sc):
    """Every assertion reads a key the trainer at 202849da writes: declared_vs_built, argv_sha, steps_total, S_steps_per_epoch_100pct, info.*, levers.{params,n_params_trainable,w_acc,w_op,w_tac_traj,stok_dim,init_ckpt,recover,
    dagger_p,dagger_bank_sha256,dagger_bank_sha256_declared,dagger,ego_perturb,ego_perturb_with_dagger,launch_intent,p_gt_*}, live.{pass,missing_terms,unexpected_dead,terms,control_prior_grad_sum,head_o_grad_step0,groups_grad},
    acc_probe, dagger_probe, otband_probe.{pass,otband,detach_upper,contribution,purity,code_edge_max_dknot}. The selfcheck does NOT record the training seed or (for NULL) the detach: the seed is bound through
    argv_sha (the pinned gate line carries --seed) and read here from the pinned tokens; detach is read from otband_probe.detach_upper (CTL / BAND) and from the ABSENCE of that probe (NULL); the post check reads both from the checkpoint."""
    spec = TABLE[base_tag(tag)]
    kind, init, p = spec["kind"], spec["init"], []
    lv = sc.get("levers") if isinstance(sc.get("levers"), dict) else {}
    info = sc.get("info") if isinstance(sc.get("info"), dict) else {}

    def eq(name, got, want):
        if got != want:
            p.append(f"{name} = {got!r}, wanted {want!r}")

    def near(name, got, want, tol):
        if not (finite(got) and abs(got - want) <= tol):
            p.append(f"{name} = {got!r}, wanted {want!r} (tol {tol})")
    if spec["blocked"]:
        p.append(f"{tag} is blocked: {spec['blocked']}")
    eq("declared_vs_built", sc.get("declared_vs_built"), "PASS")
    eq("selfcheck argv_sha (the trainer ran the pinned gate line)", sc.get("argv_sha"), argv_sha16(gate_argv(tag, toks)))
    eq("steps_total", sc.get("steps_total"), expected_steps(tag))
    eq("S_steps_per_epoch_100pct", sc.get("S_steps_per_epoch_100pct"), S_PER_EPOCH)
    eq("pinned argv --seed (the training seed of this tag)", flag_value(toks, "--seed"), str(spec["seed"]))
    eq("pinned argv --dagger-seed (the init's index)", flag_value(toks, "--dagger-seed"), str(init))
    eq("pinned argv --w-acc", flag_value(toks, "--w-acc"), str(W_ACC))
    eq("pinned argv --detach-upper present", "--detach-upper" in toks, kind != "null")
    for k, v in DATA.items():
        eq(f"info.{k}", info.get(k), v)
    eq("info.batch", info.get("batch"), 64)
    eq("info.md5_overridden", info.get("md5_overridden"), False)
    eq("info.excluded_in_train", info.get("excluded_in_train"), 0)
    eq("built lever kin_mode", lv.get("kin_mode"), "yaw")
    # start state: the M2b checkpoint, applied, read back equal; no new keys (DAG-OACC's architecture)
    ic = info.get("init_ckpt") if isinstance(info.get("init_ckpt"), dict) else {}
    eq("info.init_ckpt.sha256 (start state M2b)", ic.get("sha256"), init_sha(init))
    eq("info.init_ckpt.applied", ic.get("applied"), True)
    eq("info.init_ckpt.n_mismatch_after_load", ic.get("n_mismatch_after_load"), 0)
    eq("info.init_ckpt.source_arm", ic.get("source_arm"), "M2b")
    eq("info.init_ckpt.source_seed", ic.get("source_seed"), init)
    want_ic = M2B_DECL + rf"\M2b_s{init}\ckpt_last.pt"
    if ntpath.normpath(str(ic.get("path", "")).replace("/", "\\")).lower() != ntpath.normpath(want_ic).lower():
        p.append(f"info.init_ckpt.path {ic.get('path')!r} is not {want_ic}")
    if ic.get("declared_new_keys"):
        p.append(f"info.init_ckpt.declared_new_keys = {ic.get('declared_new_keys')!r}: no arm adds keys")
    # NOTHING is frozen: the trainable parameter count of every group is DAG-OACC's (read back from the BUILT model's requires_grad counts)
    pr = lv.get("params") if isinstance(lv.get("params"), dict) else {}
    for g, n in PARAMS.items():
        eq(f"BUILT trainable params of group {g} (DAG-OACC's count; none frozen)", pr.get(g), n)
    extra = sorted(set(pr) - set(PARAMS))
    if extra:
        p.append(f"BUILT parameter groups {extra} are not in DAG-OACC's model (a module was added)")
    eq("BUILT n_params_trainable", lv.get("n_params_trainable"), N_PARAMS_TRAINABLE)
    # the DAG-OACC recipe levers
    eq("built lever w_acc", lv.get("w_acc"), W_ACC)
    eq("built lever p_gt_start", lv.get("p_gt_start"), P_GT)
    eq("built lever p_gt_end", lv.get("p_gt_end"), P_GT)
    eq("built lever dagger_p", lv.get("dagger_p"), DAGGER_P)
    eq("built lever dagger_bank_sha256 (read back from the built swap)", lv.get("dagger_bank_sha256"), bank_sha(init))
    eq("built lever dagger_bank_sha256_declared (the argv's)", lv.get("dagger_bank_sha256_declared"), BANK_SHA[init])
    dg = lv.get("dagger") if isinstance(lv.get("dagger"), dict) else {}
    eq("built dagger.dagger_p", dg.get("dagger_p"), DAGGER_P)
    eq("built dagger.dagger_seed", dg.get("dagger_seed"), init)
    eq("built dagger.dagger_bank_sha256", dg.get("dagger_bank_sha256"), bank_sha(init))
    want_bank = BANK_DECL + rf"\s{init}\bank.json"
    if ntpath.normpath(str(dg.get("dagger_bank", "")).replace("/", "\\")).lower() != ntpath.normpath(want_bank).lower():
        p.append(f"built dagger.dagger_bank {dg.get('dagger_bank')!r} is not {want_bank}")
    idg = info.get("dagger") if isinstance(info.get("dagger"), dict) else {}
    eq("info.dagger.bank_sha256", idg.get("bank_sha256"), bank_sha(init))
    eq("info.dagger.coverage.blocked", (idg.get("coverage") or {}).get("blocked"), False)
    dpr = sc.get("dagger_probe") if isinstance(sc.get("dagger_probe"), dict) else {}
    eq("dagger_probe.pass", dpr.get("pass"), True)
    eq("dagger_probe.p_declared", dpr.get("p_declared"), DAGGER_P)
    if not (num(dpr.get("n_selected_at_p1")) and dpr["n_selected_at_p1"] > 0):
        p.append(f"dagger_probe.n_selected_at_p1 = {dpr.get('n_selected_at_p1')!r}, wanted > 0 (a swapped row exists on the probe batch)")
    for k in ("control_repeat_loss_diff", "control_repeat_grad_diff", "control_p0_loss_diff", "control_p0_grad_diff", "control_prior_grad_diff"):
        eq(f"dagger_probe.{k} (control: exactly 0.0)", dpr.get(k), 0.0)
    ap = sc.get("acc_probe") if isinstance(sc.get("acc_probe"), dict) else {}
    eq("acc_probe.pass", ap.get("pass"), True)
    eq("acc_probe.w_acc_declared", ap.get("w_acc_declared"), W_ACC)
    if not (finite(ap.get("acc")) and ap["acc"] > 0.0):
        p.append(f"acc_probe.acc = {ap.get('acc')!r}, wanted finite > 0")
    elif not (finite(ap.get("contribution")) and abs(ap["contribution"] - W_ACC * ap["acc"]) <= 1e-3 * abs(W_ACC * ap["acc"]) + 1e-6):
        p.append(f"acc_probe.contribution = {ap.get('contribution')!r}, w_acc x acc = {W_ACC * ap['acc']!r} (the weight does not reach the total)")
    if not (num(ap.get("head_o_grad_diff")) and ap["head_o_grad_diff"] > 0.0):
        p.append(f"acc_probe.head_o_grad_diff = {ap.get('head_o_grad_diff')!r}, wanted > 0 (the term gives O's head gradient)")
    for k in ("control_repeat_total_diff", "control_repeat_grad_diff", "control_prior_grad_diff"):
        eq(f"acc_probe.{k} (control: exactly 0.0)", ap.get(k), 0.0)
    # the levers that are NOT on
    eq("built lever ego_perturb_p", (lv.get("ego_perturb") or {}).get("ego_perturb_p"), 0.0)
    eq("built lever recover_p", (lv.get("recover") or {}).get("recover_p"), 0.0)
    eq("built lever ego_perturb_with_dagger", lv.get("ego_perturb_with_dagger"), False)
    eq("built lever launch_intent_w", (lv.get("launch_intent") or {}).get("launch_intent_w", 0.0), 0.0)
    for k in ("wm_only", "level_head", "lead_head", "stok_dim"):
        if k in lv:
            p.append(f"levers.{k} is present: a lever outside DAG-OACC's recipe was built")
    if "STOK" in pr:
        p.append("a strategic-token embedding was built")

    def purity_by_term(ob, keys):
        pu = ob.get("purity") if isinstance(ob.get("purity"), dict) else {}
        by = pu.get("by_term") if isinstance(pu.get("by_term"), dict) else {}
        eq("otband_probe.purity.by_term keys (every O term of this arm's loss, the acceleration term included)", sorted(by), sorted(keys))
        for k in keys:
            v = by.get(k) if isinstance(by.get(k), dict) else {}
            eq(f"otband_probe.purity.by_term.{k}.upper_grad_abs_sum (exactly 0.0 on S and T)", v.get("upper_grad_abs_sum"), 0.0)
            if not (num(v.get("head_o")) and v["head_o"] > 0.0):
                p.append(f"otband_probe.purity.by_term.{k}.head_o = {v.get('head_o')!r}, wanted > 0 (O's head receives gradient)")
    ob = sc.get("otband_probe") if isinstance(sc.get("otband_probe"), dict) else None
    if kind == "null":
        for k in ("w_op", "w_tac_traj"):
            if k in lv:
                p.append(f"null arm: built lever {k} = {lv[k]!r} is present (a band lever was built)")
        if ob is not None and (ob.get("detach_upper") is True or ob.get("otband") not in (None, "off")):
            p.append(f"null arm: otband_probe is present and says {ob.get('otband')!r} detach_upper {ob.get('detach_upper')!r} (the detach or a band lever was built)")
    elif kind == "ctl":
        for k in ("w_op", "w_tac_traj"):
            if k in lv:
                p.append(f"control arm: built lever {k} = {lv[k]!r} is present (a band lever was built)")
        ob = ob or {}
        eq("otband_probe.pass (control)", ob.get("pass"), True)
        eq("otband_probe.otband (control: the detach probe only)", ob.get("otband"), "detach-only")
        eq("otband_probe.detach_upper (the BUILT config's detach)", ob.get("detach_upper"), True)
        purity_by_term(ob, purity_keys(kind))
    else:
        near("built lever w_op", lv.get("w_op"), W_OP, 1e-9)
        near("built lever w_tac_traj", lv.get("w_tac_traj"), W_TAC, 1e-9)
        ob = ob or {}
        eq("otband_probe.pass", ob.get("pass"), True)
        eq("otband_probe.otband", ob.get("otband"), "on")
        eq("otband_probe.detach_upper (the BUILT config's detach)", ob.get("detach_upper"), True)
        for wname, key, want in (("w_op", "op", W_OP), ("w_tac_traj", "tac", W_TAC)):
            c = (ob.get("contribution") or {}).get(wname)
            if not isinstance(c, dict):
                p.append(f"otband_probe.contribution.{wname} is missing")
                continue
            near(f"otband_probe.contribution.{wname}.weight", c.get("weight"), want, 1e-9)
            if not (finite(c.get("term")) and c["term"] > 0.0):
                p.append(f"otband_probe.contribution.{wname}.term = {c.get('term')!r}, wanted > 0")
            elif not (finite(c.get("moved_total_by")) and abs(c["moved_total_by"] - c["weight"] * c["term"]) <= 1e-3 * abs(c["weight"] * c["term"]) + 1e-6):
                p.append(f"band contribution {wname}: the total moved by {c.get('moved_total_by')!r}, weight x term = {c['weight'] * c['term']!r} (the weight does not reach the total)")
        pu = ob.get("purity") if isinstance(ob.get("purity"), dict) else {}
        eq("otband_probe.purity.upper_grad_abs_sum (O's terms on S and T: exactly 0.0)", pu.get("upper_grad_abs_sum"), 0.0)
        if not (num(pu.get("head_o")) and pu["head_o"] > 0.0):
            p.append(f"otband_probe.purity.head_o = {pu.get('head_o')!r}, wanted > 0 (O's head receives gradient)")
        purity_by_term(ob, purity_keys(kind))
        if not (finite(ob.get("code_edge_max_dknot")) and ob["code_edge_max_dknot"] > 0.0):
            p.append(f"otband_probe.code_edge_max_dknot = {ob.get('code_edge_max_dknot')!r}, wanted > 0 (the T -> O edge)")
    if kind == "ctl":
        pu = ob.get("purity") if isinstance(ob.get("purity"), dict) else {}
        eq("otband_probe.purity.upper_grad_abs_sum (control: exactly 0.0)", pu.get("upper_grad_abs_sum"), 0.0)
    # G-LIVE: every term finite, every group (none frozen) has gradient
    lp = sc.get("live") if isinstance(sc.get("live"), dict) else {}
    eq("live.pass", lp.get("pass"), True)
    if lp.get("missing_terms") != [] or lp.get("unexpected_dead") != []:
        p.append(f"G-LIVE: missing {lp.get('missing_terms')!r} unexpected_dead {lp.get('unexpected_dead')!r}")
    terms = lp.get("terms") if isinstance(lp.get("terms"), dict) else {}
    if not terms or any(not finite(v) for v in terms.values()):
        p.append("G-LIVE: a loss term is missing or not finite")
    for k in ("ade", "fde", "acc"):
        if not (finite(terms.get(k)) and terms[k] > 0.0):
            p.append(f"G-LIVE: term {k} = {terms.get(k)!r}, wanted finite > 0")
    bands = ("ade_op", "fde_op", "ade_tac", "fde_tac", "op", "tac")
    if kind != "band":
        if any(k in terms for k in bands):
            p.append(f"{kind} arm: G-LIVE carries band terms {[k for k in bands if k in terms]}")
    else:
        for k in bands:
            if not (finite(terms.get(k)) and terms[k] > 0.0):
                p.append(f"G-LIVE: band term {k} = {terms.get(k)!r}, wanted finite > 0")
    eq("live.control_prior_grad_sum (the control must read exactly 0.0)", lp.get("control_prior_grad_sum"), 0.0)
    if not (num(lp.get("head_o_grad_step0")) and lp["head_o_grad_step0"] > 0.0):
        p.append(f"G-LIVE: head_o_grad_step0 = {lp.get('head_o_grad_step0')!r}, wanted > 0")
    gg = lp.get("groups_grad") if isinstance(lp.get("groups_grad"), dict) else {}
    for g in ("adapter", "S", "T", "O", "head_s", "head_o", "wm"):
        if not (num(gg.get(g)) and gg[g] > 0.0):
            p.append(f"G-LIVE: group {g} has gradient {gg.get(g)!r}, wanted > 0 (no group is frozen)")
    return p


# ---- moved identity (pure over two state dicts; the loader is separate) -------------------------------------------------------------------------------------------------------------
def same_bits(a, b):
    if hasattr(a, "detach"):
        import torch
        return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def is_group_key(k, prefixes):
    return any(k == f or k.startswith(f + ".") for f in prefixes)


def moved_identity(init_sd, fin_sd):
    """-> (evidence, problems): nothing is frozen, so at least one tensor of each of S, T and O of the finished checkpoint differs from the M2b init"""
    p, ev = [], {}
    for g, pre in MOVED_PREFIXES.items():
        keys = sorted(k for k in init_sd if is_group_key(k, pre))
        if not keys:
            p.append(f"no {g} tensor found in the init checkpoint (key names changed?)")
            continue
        miss = [k for k in keys if k not in fin_sd]
        moved = [k for k in keys if k in fin_sd and not same_bits(init_sd[k], fin_sd[k])]
        ev[g] = {"n_tensors": len(keys), "n_moved": len(moved)}
        if miss:
            p.append(f"{len(miss)} {g} tensor(s) missing from the finished checkpoint (first: {miss[0]})")
        if not moved:
            p.append(f"no {g} tensor moved from the init: group {g} was not trained")
    return ev, p


def load_ckpt(path):
    if fake() and os.environ.get("OBD_CKPT_LOADER") == "pickle":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def post_problems(tag, toks, out):
    """steps reached, checkpoint, S / T / O moved, and the BUILT config the checkpoint carries (cfg: detach_upper, w_traj, w_op, w_tac_traj, w_acc; seed; the DAgger swap; the init)"""
    spec, p, ev = TABLE[base_tag(tag)], [], {}
    summ, ck_p = os.path.join(out, "summary.json"), os.path.join(out, "ckpt_last.pt")
    try:
        sm = json.load(open(summ, encoding="utf-8"))
    except (OSError, ValueError):
        return ev, ["summary.json unreadable"]
    if sm.get("steps") != expected_steps(tag) or sm.get("done") is not True:
        p.append(f"summary steps {sm.get('steps')!r} done {sm.get('done')!r}, wanted {expected_steps(tag)} and done true")
    if sm.get("argv_sha") != argv_sha16(toks):
        p.append("summary argv_sha is not the shipped argv's")
    if sm.get("seed") != spec["seed"]:
        p.append(f"summary seed {sm.get('seed')!r}, wanted {spec['seed']}")
    if not os.path.exists(ck_p):
        return ev, p + ["no ckpt_last.pt"]
    ic = loc(M2B_DECL + rf"\M2b_s{spec['init']}\ckpt_last.pt")
    if fake() and os.environ.get("OBD_SKIP_WEIGHTS") == "1":
        return ev, p
    a, b = load_ckpt(ic), load_ckpt(ck_p)
    ev, fp = moved_identity(a["model"], b["model"])
    p += fp
    if b.get("step") != expected_steps(tag):
        p.append(f"checkpoint step {b.get('step')!r}, wanted {expected_steps(tag)}")
    if b.get("seed") != spec["seed"]:
        p.append(f"checkpoint seed {b.get('seed')!r}, wanted {spec['seed']}")
    if b.get("arm") != "M2b":
        p.append(f"checkpoint arm {b.get('arm')!r}, wanted 'M2b'")
    if b.get("argv_sha") != argv_sha16(toks):
        p.append("checkpoint argv_sha is not the shipped argv's")
    cfg = b.get("cfg") if isinstance(b.get("cfg"), dict) else {}
    kind = spec["kind"]
    want = {"detach_upper": kind != "null", "w_traj": 0.0 if kind == "band" else 1.0, "w_op": W_OP if kind == "band" else 0.0, "w_tac_traj": W_TAC if kind == "band" else 0.0, "w_acc": W_ACC}
    for k, v in want.items():
        if not (cfg.get(k) == v or (num(v) and num(cfg.get(k)) and abs(cfg[k] - v) <= 1e-9)):
            p.append(f"checkpoint cfg.{k} = {cfg.get(k)!r}, wanted {v!r}")
    dg = b.get("dagger") if isinstance(b.get("dagger"), dict) else {}
    if dg.get("dagger_p") != DAGGER_P or dg.get("dagger_bank_sha256") != bank_sha(spec["init"]) or dg.get("dagger_seed") != spec["init"]:
        p.append(f"checkpoint dagger record {({k: dg.get(k) for k in ('dagger_p', 'dagger_seed', 'dagger_bank_sha256')})!r} is not p {DAGGER_P}, seed {spec['init']}, bank {bank_sha(spec["init"])[:8]}")
    ick = b.get("init_ckpt") if isinstance(b.get("init_ckpt"), dict) else {}
    if ick.get("sha256") != init_sha(spec["init"]):
        p.append("checkpoint init_ckpt.sha256 is not the M2b start state")
    return ev, p

# ---- tree and data binding -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def tree_problems():
    p, c = [], commit()
    if not HEX40.match(c or ""):
        return [f"OBD_COMMIT {c!r} is not a full 40-hex commit"]
    try:
        tip = open(os.path.join(repo(), "TIP_OBD.txt"), encoding="utf-8").read().strip()
    except OSError:
        tip = None
    if tip != c:
        p.append(f"TIP_OBD.txt reads {tip!r}, not OBD_COMMIT {c} (the tree was not shipped at this commit)")
    try:
        man = json.load(open(os.path.join(home(), "tree_manifest.json"), encoding="utf-8"))
    except (OSError, ValueError) as e:
        return p + [f"tree_manifest.json unreadable: {type(e).__name__}"]
    if man.get("commit") != c:
        p.append(f"tree_manifest.json commit {man.get('commit')!r} != OBD_COMMIT")
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
    t = tokens("OBD-BAND_s0")
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
    want = os.environ.get("OBD_FAKE_V9_MD5") if fake() and os.environ.get("OBD_FAKE_V9_MD5") else V9_MD5
    if out.get("v9_labels_train") and out["v9_labels_train"] != want:
        bad.append(f"v9_labels_train md5 {out['v9_labels_train']} != the argv's --v9-train-md5")
    return out, bad


def disk_floor_problems():
    free = shutil.disk_usage(loc(DECL_D) if fake() else "D:\\").free
    floor = int(float(os.environ.get("OBD_FLOOR_BYTES", "4e9"))) if fake() else int(4e9)
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
    init = TABLE[base_tag(tag)]["init"]
    bad = list(bad) + bank_problems(init)
    return {"tag": tag, "commit": commit(), "argv_sha256": tokens_sha(line.split(" ")), "data_md5": md5s, "dagger_bank_sha256": sha_file(bank_file(init)) if os.path.isfile(bank_file(init)) else None, "tree_manifest_sha256": tm, "trainer_sha256": sha_file(trainer_path()) if os.path.exists(trainer_path()) else None}, bad


def token_ok(tag, line):
    """[] iff a token written by `gate` still binds this tag, commit, argv, data and shipped bytes"""
    try:
        tok = json.load(open(token_path(tag), encoding="utf-8"))
    except (OSError, ValueError):
        return ["no gate token"]
    want, bad = binding(tag, line)
    p = list(bad)
    if tok.get("written_by") != "obd_check.py gate":
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
    return float(os.environ.get("OBD_POLL_S", "5"))


def gpu_sample():
    try:
        r = subprocess.run([os.environ.get("OBD_NVSMI", "nvidia-smi"), "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, encoding="utf-8", errors="replace", timeout=20)
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
    """planned = the seven runs in RUN_ORDER (the 12 h basis); with_null_s1 adds the conditional OBD-NULL_s1 (information only). Every run is taken at the pilot's OBD-BAND step time (CTL and NULL compute less)."""
    n = len(RUN_ORDER)
    planned = mean_s * RUN_STEPS * n / 3600
    return {"n_runs_planned": n, "run_steps": RUN_STEPS, "steps_total": RUN_STEPS * n, "projected_total_h_planned": planned, "projected_total_h_with_null_s1": mean_s * RUN_STEPS * (n + 1) / 3600,
            "per_run_h": mean_s * RUN_STEPS / 3600, "limit_h": LIMIT_H, "within_limit_planned": planned <= LIMIT_H}


def classify(samples):
    util = [u for _, u in samples]
    if not util:
        return "unmeasured", None
    med = statistics.median(util)
    return ("compute-bound" if med >= 60 else "loader-bound"), med


# ---- the box-2 job -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def stage_fail(tag, gid, kind, why):
    write_text(os.path.join(run_dir(tag), f"STAGE_{gid}.fail"), f"{kind}: {why}"[:900] + "\n")
    print(f"OBD-FAIL {tag} {kind}: {why}"[:1200])
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
    bad = gate_problems(tag, toks, sc)
    if bad:
        write_text(os.path.join(rd, "RUN.FAILED"), json.dumps({"tag": tag, "at": now(), "problems": bad}, indent=1))
        return bad
    b, dbad = binding(tag, line)
    if dbad:
        return dbad
    write_text(token_path(tag), json.dumps({"schema": "obd-gate-token-v1", "written_by": "obd_check.py gate", "at": now(), "selfcheck_sha256": sha_file(os.path.join(gd, "selfcheck.json")), **b}, indent=1))
    return []


def cmd_stage(tag, gid):
    if tag != PILOT_TAG and tag not in TABLE:
        return stage_fail(tag, gid, "pin-fail", f"unknown tag {tag}")
    if TABLE.get(tag, {}).get("blocked"):
        return stage_fail(tag, gid, "pin-fail", f"{tag} is blocked: {TABLE[tag]['blocked']}")
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
    bad += bank_problems(TABLE[base_tag(tag)]["init"])
    if bad:
        return stage_fail(tag, gid, "preflight-fail", " | ".join(bad)[:800])
    bad = audit_line(tag, line)
    if bad:
        return stage_fail(tag, gid, "pin-fail", bad[0])
    jobs = [(tag, line)]
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
    write_text(os.path.join(rd, "RUN.done"), json.dumps({"tag": tag, "at": now(), "commit": commit(), "moved_identity": ev,
                                                         "evidence": {"ckpt_sha256": sha_file(ck), "summary_sha256": sha_file(summ), "train_log_sha256": sha_file(logp)}}, indent=1))
    write_text(os.path.join(rd, f"STAGE_{gid}.done"), "done\n")
    print(f"OBD-OK {tag}")
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
    bound, util = classify(samples)
    res = {"schema": "obd-pilot-v1", "at": now(), "commit": commit(), "tag": tag, "argv_sha256": tokens_sha(line.split(" ")), "pilot_steps": PILOT_STEPS, "wall_s_total_incl_startup": wall, "timing": tm,
           "peak_gpu_mib": max((m for m, _ in samples), default=None), "n_gpu_samples": len(samples), "gpu_util_median_pct": util, "limit_by": bound,
           "train_log_sha256": sha_file(os.path.join(pd, "train_log.jsonl")), "projection": project(tm["mean_s_per_step"]),
           "full_run_gates_at_pilot": {t: os.path.exists(token_path(t)) for t in GATE_AT_PILOT}}
    write_text(os.path.join(pd, "PILOT.json"), json.dumps(res, indent=1))
    write_text(os.path.join(rd, f"STAGE_{gid}.done"), "done\n")
    print(f"OBD-OK {tag} mean {tm['mean_s_per_step']:.4f} s/step, {res['projection']['projected_total_h_planned']:.1f} h planned")
    return 0


def cmd_cleanup(tag, gid):
    """box 2, AFTER the dev box pulled and sha-verified the bank and put BANK_VERIFIED_<gid>: delete this run's own temporary dirs by literal path; log (path, bytes). Never a checkpoint, never another run's folder."""
    rd = run_dir(tag)
    mark = os.path.join(rd, f"BANK_VERIFIED_{gid}")
    if not os.path.exists(mark):
        write_text(os.path.join(rd, f"CLEAN_{gid}.fail"), "no BANK_VERIFIED marker: the bank was not proven, nothing is deleted\n")
        print("OBD-CLEAN-REFUSED no bank proof")
        return 6
    del_ = []
    for t in ([tag] + GATE_AT_PILOT) if is_pilot(tag) else [tag]:
        d = rm_own_temp(loc(gate_out_decl(t)), "gate_out")
        if d:
            del_.append(d)
    if is_pilot(tag):
        d = rm_own_temp(loc(out_decl(tag)), "pilot_out")
        if d:
            del_.append(d)
    write_text(os.path.join(rd, "CLEANUP.json"), json.dumps({"tag": tag, "at": now(), "deleted_own_temp": del_}, indent=1))
    write_text(os.path.join(rd, f"CLEAN_{gid}.done"), "done\n")
    print(f"OBD-CLEAN-OK {tag} deleted {len(del_)}")
    return 0


# ---- dev-box commands -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cmd_verify_token(a):
    tok = json.load(open(a["token"], encoding="utf-8"))
    line = [l.strip() for l in open(a["argv-file"], encoding="utf-8") if l.strip()][0]
    p = []
    if tok.get("written_by") != "obd_check.py gate":
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
    print(f"pilot verified: mean {tm['mean_s_per_step']:.4f} s/step; {pr['n_runs_planned']} runs {pr['projected_total_h_planned']:.1f} h (with the conditional OBD-NULL_s1 {pr['projected_total_h_with_null_s1']:.1f} h); limit {LIMIT_H:.0f} h")
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


def sums_sha(sums_path, name="ckpt_last.pt"):
    """the sha256 a SHA256SUMS.txt (`<64 hex> *<name>`) holds for `name`, or None"""
    for line in open(sums_path, encoding="utf-8", errors="replace"):
        m = re.match(r"^([0-9a-fA-F]{64}) [ *]?(\S+)\s*$", line.strip())
        if m and m.group(2) == name:
            return m.group(1).lower()
    return None


def tensor_parity(a_path, b_path):
    """-> (equal, detail): every tensor of the two checkpoints' `model` state dicts has the same keys, shape, dtype and bytes. Metadata (argv_sha, out path, opt state) is NOT compared."""
    a, b = load_ckpt(a_path)["model"], load_ckpt(b_path)["model"]
    ka, kb = sorted(a), sorted(b)
    differ = [k for k in ka if k in b and not same_bits(a[k], b[k])]
    detail = {"n_keys_a": len(ka), "n_keys_b": len(kb), "only_in_a": [k for k in ka if k not in b][:5], "only_in_b": [k for k in kb if k not in a][:5], "n_tensors_differ": len(differ), "first_differ": differ[:5]}
    return ka == kb and not differ, detail


def cmd_parity(a):
    """NULL parity (card, Controls item 1): OBD-NULL_s0's ckpt_last.pt sha256 against DAG-OACC s0's (SHA256SUMS.txt). rc 0 = the control RAN (see `append_obd_null_s1` in the output file); rc 2 = an input is unreadable.
    basis `file` (default, the card's text): the whole-file sha256. NOTE (IMPL_CHOICES_OBD 6): the checkpoint embeds argv_sha, which contains --out, and --out differs by construction, so the file sha256 cannot match; the tensor-level
    comparison (basis `tensors`, needs --dagoacc-ckpt) is recorded beside it in every case."""
    bank, sums, out = a.get("bank-ckpt"), a.get("sums"), a.get("out")
    basis = a.get("basis", "file") if isinstance(a.get("basis", "file"), str) else "file"
    if basis not in ("file", "tensors") or not (bank and sums and out):
        print("parity refused: need --bank-ckpt, --sums, --out and --basis file|tensors")
        return 2
    want = sums_sha(sums) if os.path.isfile(sums) else None
    if not want or not HEX64.match(want):
        print(f"parity refused: no `ckpt_last.pt` sha256 in {sums}")
        return 2
    pin = os.environ.get("OBD_TEST_DAGOACC_SHA") or DAGOACC_S0_CKPT_SHA              # the env var is a TEST hook (the dry run's stand-in checkpoint is not DAG-OACC's); a real launch never sets it
    if want != pin:
        print(f"parity refused: {sums} holds {want} for ckpt_last.pt, the pinned DAG-OACC s0 value is {pin} (the reference moved)")
        return 2
    if not os.path.isfile(bank):
        print(f"parity refused: {bank} does not exist")
        return 2
    got = sha_file(bank)
    res = {"schema": "obd-null-parity-v1", "at": now(), "bank_ckpt": bank, "bank_ckpt_sha256": got, "dagoacc_s0_ckpt_sha256_from_sums": want, "file_sha_equal": got == want, "basis": basis,
           "tensor_equal": None, "tensor_detail": None}
    dag = a.get("dagoacc-ckpt")
    if isinstance(dag, str) and os.path.isfile(dag):
        if sha_file(dag) != want:
            print(f"parity refused: {dag} does not hash to the SUMS value")
            return 2
        eq_, det = tensor_parity(bank, dag)
        res["tensor_equal"], res["tensor_detail"] = eq_, det
    if basis == "tensors" and res["tensor_equal"] is None:
        print("parity refused: basis tensors needs --dagoacc-ckpt (the DAG-OACC s0 checkpoint)")
        return 2
    res["append_obd_null_s1"] = (not res["file_sha_equal"]) if basis == "file" else (not res["tensor_equal"])
    write_text(out, json.dumps(res, indent=1))
    print(f"PARITY file_sha_equal={res['file_sha_equal']} tensor_equal={res['tensor_equal']} basis={basis} append_obd_null_s1={res['append_obd_null_s1']}")
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
        print("preflight OK" if not bad else "OBD-PREFLIGHT-FAIL " + " | ".join(bad))
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
    if cmd == "parity":
        return cmd_parity(a)
    print(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
