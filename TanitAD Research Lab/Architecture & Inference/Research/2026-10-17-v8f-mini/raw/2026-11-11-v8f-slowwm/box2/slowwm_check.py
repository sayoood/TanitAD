"""SLOWWM launch-gate checker (box 2 and dev box; stdlib, torch only in the weights-identity load). Pattern: theory/wmpool/box2/wmpool_check.py with the pool replaced by the basis lever.
Card theory/slowwm/PREREG_V8F_SLOWWM.md (sha256 7c2775e3...94ca). Code of record: commit f7e790ab57e10a9004b04b206195a0ae5a5295b8.
  basis --gid G    ONE box-2 job: slow_basis.py fit (train, --expect-windows 40358, --exclude 540:600, --spool), cache (slow, pca) x (train, eval), check3 once per split of clips8.json (eval 6 + train 2 clips, combined into one record; AMEND_SLOWWM_2), counts -> BASIS.json (basis caches are never deleted)
  argv --tag T [--pilot] [--out-txt F] | stage --tag T --gid G [--pilot] | cleanup --tag T --gid G | verify-basis | verify-token | project | permit | render | preflight   (as WMPOOL's)
Test hooks (dev box only, refused by the chain unless SLOWWM_DRY=1): SLOWWM_ROOT maps D:\\tanitad to a folder; SLOWWM_FAKE_INIT_SHA_FILE; SLOWWM_TEST_BASIS (a stand-in for slow_basis.py); SLOWWM_TEST_COUNTS; SLOWWM_FAKE_FREE_GB."""
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
ARGVS_SHA256 = "192168f16d52bcc6af12e1f28c180600bcbb47629d13340229641c835b64ac8f"            # frozen/argvs_slowwm.json (make_argvs_slowwm.py from R35H s0/s1 argv.txt, MEASURED 2026-11-11)
ARGVS_FILE = os.path.join(HERE, "frozen", "argvs_slowwm.json")
CARD_SHA256 = "7c2775e39f9b1271342a0c588780d3da8b4d63dfb56d06de9fef4c50bc8594ca"
#: pinned inputs of the basis stage, shipped from the dev box and re-hashed on box 2 (the builder and the 8-clip list live beside the card; nothing is copied into this package)
BASIS_CODE_SHA256 = {"slow_basis.py": "d6778c8a831266a65c40f66ca125f656b3c10593a21dd90b07462eca4e89add4", "clips8.json": "f4a034b7a51d52d2609912a575af6637f3da44ee699f2e2970c7f4d3a67421f3"}
INIT_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}     # INHERITED from wmpool_check.py
V9_MD5 = "83e30bd3791f0767707eade5f4671759"
DATA = {"train_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1", "excluded_sha256": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "n_clips": 4512, "n_excluded": 60}
RUN_STEPS, PILOT_STEPS, LIMIT_H, EXPECT_WINDOWS = 19164, 200, 72.0, 40358
NEED_GB_RUN, NEED_GB_SHIP, NEED_GB_BASIS = 1, 1, 6          # a run writes ~47 MB (MEASURED, EGOWM banks); basis: spool ~1.9 GB + 2 kinds x 2 splits ~1.0 GB (ESTIMATED, HANDOFF_SLOWWM.md) -> 6 asked
# Basis disk rule (2026-11-11 ~22:00, after the first basis run): the 6 GB counts bytes the stage would ALLOCATE. A re-run reads the spool files that exist (slow_basis.py Source.inputs) and replaces every
# cache and fit file in place (a .tmp then os.replace, one file at a time), so the bytes already under D:\tanitad\zbasis are not new allocation. MEASURED on box 2 after the first run: 2.77 GiB in
# 14,152 files (spool 1.82, slow 0.46, pca 0.46, fit 0.03 GiB). Need = max(1 GB, 6 GB - held bytes), the same rule on box 2 (cmd_basis) and on the dev box (`disk-basis`, slowwm_basis.sh).
DECL_D = r"D:\tanitad"
HOME_DECL, REPO_DECL, M2B_DECL = DECL_D + r"\slowwm", DECL_D + r"\slowwm\repo", DECL_D + r"\runs\v8f_mini_full"
ZB_DECL, FIT_DECL, SPOOL_DECL = DECL_D + r"\zbasis", DECL_D + r"\zbasis\fit", DECL_D + r"\zbasis\spool"
CODE_DECL, BASIS_DIR_DECL, PILOT_OUT_DECL = HOME_DECL + r"\basiscode", HOME_DECL + r"\basis", HOME_DECL + r"\pilot_out"
CACHE_ROOT, TRACKS = DECL_D + r"\cache_full", DECL_D + r"\cache_full\native_tracks.pt"
COUNTS = {"train": 4572, "eval": 141}
KINDS = ("slow", "pca")
TRAINER_REL, MODEL_REL, DATA_REL, WINDOWS_REL = "stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/data/v8f_wm_windows.py"
HMAX, CTX = 91, (3, 3, 3)
NEEDLES = {TRAINER_REL: ('"--wm-basis"', "def bad_wm_basis", "def wm_basis_readback", "wm_basis=a.wm_basis"), MODEL_REL: ("def built_basis", "wm_basis"), WINDOWS_REL: ("zbasis", "norm_mean_tokens")}
HEX64, HEX40 = re.compile(r"^[0-9a-f]{64}$"), re.compile(r"^[0-9a-f]{40}$")
TABLE = {"SLOW_s0": {"seed": 0, "init": 0, "kind": "slow", "order": 1}, "PCA-B_s0": {"seed": 0, "init": 0, "kind": "pca", "order": 2}, "SLOW_s1": {"seed": 1, "init": 1, "kind": "slow", "order": 3},
         "PCA-B_s1": {"seed": 1, "init": 1, "kind": "pca", "order": 4}, "SLOW_rep": {"seed": 2, "init": 0, "kind": "slow", "order": 5}}
for _v in TABLE.values():
    _v["blocked"] = None
PILOT_TAG, PILOT_BASE = "PILOT_SLOW_s0", "SLOW_s0"
RUN_ORDER = sorted(TABLE, key=lambda t: TABLE[t]["order"])
GATE_AT_PILOT = list(RUN_ORDER)             # the pilot job also gates all five full-length argvs (a real selfcheck each, before the 72 h decision)


def fake():
    return os.environ.get("SLOWWM_ROOT")


def loc(p):
    r = fake()
    return os.path.join(r, *[x for x in p[len(DECL_D):].split("\\") if x]) if r and p.lower().startswith(DECL_D.lower()) else p


def home():
    return loc(HOME_DECL)


def repo():
    return loc(REPO_DECL)


def commit():
    return os.environ.get("SLOWWM_COMMIT", "")


def dry():
    return os.environ.get("SLOWWM_DRY") == "1"


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
    return json.load(open(os.environ["SLOWWM_FAKE_INIT_SHA_FILE"]))[str(s)] if fake() and os.environ.get("SLOWWM_FAKE_INIT_SHA_FILE") else INIT_SHA[s]


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


def frozen_argvs():
    got = sha_file(ARGVS_FILE)
    if got != ARGVS_SHA256 and not os.environ.get("SLOWWM_TEST_ARGVS"):
        raise SystemExit(f"REFUSED: {ARGVS_FILE} sha256 {got} != the pinned {ARGVS_SHA256}")
    return {k: v["argv"] for k, v in json.load(open(ARGVS_FILE, encoding="utf-8")).items()}


def base_tag(tag):
    return PILOT_BASE if tag == PILOT_TAG else tag


def is_pilot(tag):
    return tag == PILOT_TAG


def out_decl(tag):
    a = frozen_argvs()[base_tag(tag)]
    return PILOT_OUT_DECL + r"\SLOW_s0" if is_pilot(tag) else a[a.index("--out") + 1]


def gate_out_decl(tag):
    return HOME_DECL + rf"\gate_out\{tag}"


def tokens(tag):
    if tag != PILOT_TAG and tag not in TABLE:
        raise KeyError(tag)
    t = list(frozen_argvs()[base_tag(tag)])
    if is_pilot(tag):
        t[t.index("--out") + 1] = out_decl(tag)
        t += ["--steps", str(PILOT_STEPS)]
    return t


def audit_line(tag, line):
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
    return gt + ["--selfcheck-only"]


def expected_steps(tag):
    return PILOT_STEPS if is_pilot(tag) else RUN_STEPS


def zb_train_decl(kind):
    return ZB_DECL + "\\" + kind + r"\train"


def read_basis_manifest(kind):
    """the cache manifest ON DISK (box 2) -> dict, or {} when unreadable"""
    try:
        return json.load(open(os.path.join(loc(zb_train_decl(kind)), "BASIS_MANIFEST.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def basis_record():
    try:
        return json.load(open(os.path.join(loc(BASIS_DIR_DECL), "BASIS.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def gate_problems(tag, toks, sc, manifest=None, record=None):
    """BUILT-lever assertions on a real selfcheck (pure; tests plant selfchecks). Keys: declared_vs_built, argv_sha, steps_total, info.*, levers.{wm_only,wm_k,wm_k_levels,wm_strides,wm_basis_readback,wm_ladder_readback,params,w,recover,dagger_p,ego_perturb},
    live.{...}, planner_identity_probe.{pass,n_tensors,n_equal}. manifest / record default to the box-2 files (the cache manifest of the kind, BASIS.json)."""
    spec = TABLE[base_tag(tag)]
    init, kind, p = spec["init"], spec["kind"], []
    manifest = read_basis_manifest(kind) if manifest is None else manifest
    record = basis_record() if record is None else record
    lv = sc.get("levers") if isinstance(sc.get("levers"), dict) else {}
    info = sc.get("info") if isinstance(sc.get("info"), dict) else {}

    def eq(name, got, want):
        if got != want:
            p.append(f"{name} = {got!r}, wanted {want!r}")
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
    # ---- the basis READ BACK from the built module, the built dataset and the cache manifest (levers.wm_basis_readback)
    rb_ = lv.get("wm_basis_readback")
    if not isinstance(rb_, dict):
        p.append("levers.wm_basis_readback is missing: the basis lever was not read back from the BUILT model (declared vs built)")
        rb_ = {}
    eq("wm_basis_readback.declared", rb_.get("declared"), kind)
    bb = rb_.get("built") if isinstance(rb_.get("built"), dict) else {}
    eq("wm_basis_readback.built.mode", bb.get("mode"), kind)
    eq("wm_basis_readback.built.in_dim", bb.get("in_dim"), 256)
    eq("wm_basis_readback.built.basis_path_refused (the basis path is accepted)", bb.get("basis_path_refused"), False)
    eq("wm_basis_readback.built.mean_path_refused", bb.get("mean_path_refused"), True)
    eq("wm_basis_readback.built.token_path_refused", bb.get("token_path_refused"), True)
    eq("wm_basis_readback.built.basis_path_is_proj_tail", bb.get("basis_path_is_proj_tail"), True)
    eq("wm_basis_readback.config", rb_.get("config"), kind)
    eq("wm_basis_readback.dataset_basis (the BUILT dataset serves the basis)", rb_.get("dataset_basis"), kind)
    eq("wm_basis_readback.kind (the cache manifest the BUILT dataset read)", rb_.get("kind"), kind)
    eq("wm_basis_readback.dim", rb_.get("dim"), 256)
    eq("wm_basis_readback.dataset_zbasis_shape (a real item)", rb_.get("dataset_zbasis_shape"), [HMAX, 256])
    eq("wm_basis_readback.cache_problems", rb_.get("cache_problems"), [])
    bsha = rb_.get("basis_sha256")
    if not (isinstance(bsha, str) and HEX64.match(bsha)):
        p.append(f"wm_basis_readback.basis_sha256 {bsha!r} is not a sha256")
    if bsha != manifest.get("basis_sha256"):
        p.append(f"wm_basis_readback.basis_sha256 {str(bsha)[:12]} != the cache manifest's {str(manifest.get('basis_sha256'))[:12]} (the basis built is not the basis on disk)")
    if bsha != (record.get("basis_sha256") or {}).get(kind):
        p.append(f"wm_basis_readback.basis_sha256 {str(bsha)[:12]} != BASIS.json's fit record for {kind} (the basis is not the one fitted and checked)")
    if ntpath.normpath(str(rb_.get("zcache", "")).replace("/", "\\")).lower() != ntpath.normpath(zb_train_decl(kind)).lower():
        p.append(f"wm_basis_readback.zcache {rb_.get('zcache')!r} is not the {kind} basis cache {zb_train_decl(kind)!r}")
    # ---- the ladder levers READ BACK (levers.wm_ladder_readback), as WMPOOL's gate
    rb = lv.get("wm_ladder_readback")
    if not isinstance(rb, dict) or not isinstance(rb.get("built"), dict):
        return p + ["levers.wm_ladder_readback is missing: the ladder levers were not read back from the built model"]
    b = rb["built"]
    st, rk, rs = (3, 5), (10, 10, 6), (5, 5, 1)
    eq("readback hmax", rb.get("hmax"), HMAX)
    eq("built rollout k per level", b.get("k"), list(rk))
    eq("built predictor context per level", b.get("ctx_built"), list(CTX))
    cum = [1, st[0], st[0] * st[1]]
    eq("built cumulative stride per level", b.get("cum_stride"), cum)
    eq("levers.wm_strides", lv.get("wm_strides"), list(st))
    states = b.get("states") or [None] * 3
    for i in range(3):
        eq(f"level {i + 1} anchors", (b.get("anchors") or [None] * 3)[i], len(range(0, (states[i] or 0) - 1, rs[i])) if states[i] else None)
        eq(f"level {i + 1} ctx handed to the free run", (b.get("ctx_passed") or [None] * 3)[i], CTX[i])
        w = (b.get("w_in_loss") or [None] * 3)[i]
        if not (finite(w) and abs(w - 1.0) <= 1e-4):
            p.append(f"level {i + 1} k-step weight in the built loss = {w!r}, wanted 1.0")
        if HMAX < rk[i] * cum[i] + 1:
            p.append(f"level {i + 1} reach {rk[i] * cum[i] + 1} frames exceeds the {HMAX}-frame window")
    wb = rb.get("windows_built") or {}
    for k, v in (("kind", "wmonly"), ("F", HMAX), ("stride", 1), ("pooled", True), ("t_min", HMAX - 1)):
        eq(f"built dataset {k}", wb.get(k), v)
    if ntpath.normpath(str(wb.get("zcache", "")).replace("/", "\\")).lower() != ntpath.normpath(zb_train_decl(kind)).lower():
        p.append(f"built dataset reads {wb.get('zcache')!r}, wanted {zb_train_decl(kind)!r}")
    if wb.get("zcache_missing") not in (0, None):
        p.append(f"built dataset: {wb.get('zcache_missing')!r} clips have no cache file")
    if not (num(wb.get("n_windows")) and wb["n_windows"] > 0):
        p.append(f"built dataset n_windows = {wb.get('n_windows')!r}")
    eq("built wm_k", lv.get("wm_k"), 3)
    eq("built wm_k_levels", lv.get("wm_k_levels"), [3] * 3)
    # ---- wm-only: trainable parameters are exactly the world model's (the count is read, not pinned: UNVERIFIED for a 256-input proj until the first selfcheck)
    wo = lv.get("wm_only")
    if not isinstance(wo, dict) or wo.get("wm_only") is not True:
        p.append(f"levers.wm_only = {wo!r}: the WM-only mode was not built")
        wo = wo if isinstance(wo, dict) else {}
    eq("levers.wm_only.wm_k", wo.get("wm_k"), 3)
    pr = lv.get("params") or {}
    n = pr.get("WM")
    if not (num(n) and n > 0 and wo.get("n_wm") == n and wo.get("n_trainable") == n and lv.get("n_params_trainable") == n):
        p.append(f"BUILT WM / trainable parameters = {wo.get('n_wm')!r}/{wo.get('n_trainable')!r}/{lv.get('n_params_trainable')!r} (params.WM {n!r}): not one equal positive count")
    nz = {k: v for k, v in pr.items() if k != "WM" and v != 0}
    if nz or not pr:
        p.append(f"non-WM parameter groups are trainable: {nz or 'no params record'}")
    rec = info.get("wm_only") if isinstance(info.get("wm_only"), dict) else {}
    for k, v in (("sha256", init_sha(init)), ("applied", True), ("n_mismatch_after_load", 0), ("mode", "wm_only"), ("source_arm", "M2b"), ("source_seed", init), ("wm_reinit_seed", spec["seed"])):
        eq(f"info.wm_only.{k}", rec.get(k), v)
    want_ic = M2B_DECL + rf"\M2b_s{init}\ckpt_last.pt"
    if ntpath.normpath(str(rec.get("path", "")).replace("/", "\\")).lower() != ntpath.normpath(want_ic).lower():
        p.append(f"info.wm_only.path {rec.get('path')!r} is not {want_ic}")
    lp = sc.get("live") if isinstance(sc.get("live"), dict) else {}
    eq("G-LIVE pass", lp.get("pass"), True)
    if lp.get("missing_terms") != [] or lp.get("dead_wm_groups") != [] or lp.get("unexpected_dead") != []:
        p.append(f"G-LIVE: missing {lp.get('missing_terms')!r} dead {lp.get('dead_wm_groups')!r} unexpected {lp.get('unexpected_dead')!r}")
    terms = lp.get("terms") if isinstance(lp.get("terms"), dict) else {}
    if not terms or any(not finite(v) for v in terms.values()) or not (finite(terms.get("wm")) and terms["wm"] > 0):
        p.append(f"G-LIVE: WM loss terms not finite and positive: {terms.get('wm')!r}")
    if lp.get("control_adapter_grad_sum") != 0.0 or lp.get("control_nonwm_grad_sum") != 0.0:
        p.append(f"G-LIVE controls must read exactly 0.0: adapter {lp.get('control_adapter_grad_sum')!r}, non-WM {lp.get('control_nonwm_grad_sum')!r}")
    grp = lp.get("wm_groups_grad") if isinstance(lp.get("wm_groups_grad"), dict) else {}
    if not grp or any(not (num(g) and g > 0) for g in grp.values()):
        p.append("G-LIVE: a WM parameter group has zero gradient")
    if not (finite(grp.get("proj")) and grp.get("proj") > 0) and "proj" in grp:
        p.append("G-LIVE: wm.proj has no gradient")
    pi = sc.get("planner_identity_probe") if isinstance(sc.get("planner_identity_probe"), dict) else {}
    if pi.get("pass") is not True or not (num(pi.get("n_tensors")) and pi["n_tensors"] > 0 and pi.get("n_equal") == pi.get("n_tensors")):
        p.append("planner identity probe did not pass")
    if (lv.get("recover") or {}).get("recover_p") != 0.0 or lv.get("dagger_p") != 0.0 or (lv.get("ego_perturb") or {}).get("ego_perturb_p") != 0.0:
        p.append("another lever (recover / dagger / ego-perturb) is on")
    return p


def load_ckpt(path):
    if fake() and os.environ.get("SLOWWM_CKPT_LOADER") == "pickle":
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
    """every non-WM tensor of the finished checkpoint is bit-identical to the M2b start state; WM tensors exist"""
    na = {k: v for k, v in init_sd.items() if not k.startswith("wm.")}
    nb = {k: v for k, v in fin_sd.items() if not k.startswith("wm.")}
    if not na:
        return {}, ["no non-WM tensor found in the init checkpoint (key names changed?)"]
    p = [] if set(na) == set(nb) else ["non-WM tensor names differ from M2b's"]
    differ = sorted(k for k in na if k in nb and not same_bits(na[k], nb[k]))
    if differ:
        p.append(f"{len(differ)} of {len(na)} non-WM tensor(s) differ from M2b's (first: {differ[:3]})")
    if not any(k.startswith("wm.") for k in fin_sd):
        p.append("no world-model tensor in the finished checkpoint")
    return {"n_nonwm": len(na), "n_nonwm_equal": len(na) - len(differ)}, p


def post_problems(tag, toks, out):
    spec, p, ev = TABLE[base_tag(tag)], [], {}
    try:
        sm = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return ev, ["summary.json unreadable"]
    if sm.get("steps") not in (None, expected_steps(tag)) or sm.get("steps_total") not in (None, expected_steps(tag)):
        p.append(f"summary steps {sm.get('steps')!r}/{sm.get('steps_total')!r} != {expected_steps(tag)}")
    if sm.get("argv_sha") not in (None, argv_sha16(toks)):
        p.append("summary argv_sha is not the shipped argv's")
    if is_pilot(tag):
        return ev, p
    ck_p = os.path.join(out, "ckpt_last.pt")
    if not os.path.exists(ck_p):
        return ev, p + ["no ckpt_last.pt"]
    if fake() and os.environ.get("SLOWWM_SKIP_WEIGHTS") == "1":
        return ev, p
    a, b = load_ckpt(loc(M2B_DECL + rf"\M2b_s{spec['init']}\ckpt_last.pt")), load_ckpt(ck_p)
    ev, fp = weights_identity(a["model"], b["model"])
    if b.get("step") not in (None, expected_steps(tag)):
        fp.append(f"checkpoint step {b.get('step')!r}, wanted {expected_steps(tag)}")
    return ev, p + fp


def tree_problems():
    p, c = [], commit()
    if not HEX40.match(c or ""):
        return [f"SLOWWM_COMMIT {c!r} is not a full 40-hex commit"]
    try:
        tip = open(os.path.join(repo(), "TIP_SLOWWM.txt"), encoding="utf-8").read().strip()
    except OSError:
        tip = None
    if tip != c:
        p.append(f"TIP_SLOWWM.txt reads {tip!r}, not SLOWWM_COMMIT {c}")
    try:
        man = json.load(open(os.path.join(home(), "tree_manifest.json"), encoding="utf-8"))
    except (OSError, ValueError) as e:
        return p + [f"tree_manifest.json unreadable: {type(e).__name__}"]
    if man.get("commit") != c:
        p.append(f"tree_manifest.json commit {man.get('commit')!r} != SLOWWM_COMMIT")
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
    t = tokens("SLOW_s0")
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
    if out.get("v9_labels_train") and out["v9_labels_train"] != (os.environ.get("SLOWWM_FAKE_V9_MD5") if fake() and os.environ.get("SLOWWM_FAKE_V9_MD5") else V9_MD5):
        bad.append("v9_labels_train md5 differs from the argv's --v9-train-md5")
    return out, bad


def free_bytes():
    f = shutil.disk_usage(loc(DECL_D) if fake() else "D:\\").free
    return float(os.environ["SLOWWM_FAKE_FREE_GB"]) * 1e9 if fake() and dry() and os.environ.get("SLOWWM_FAKE_FREE_GB") else f


def disk_floor_problems():
    floor = int(float(os.environ.get("SLOWWM_FLOOR_BYTES", "4e9"))) if fake() else int(4e9)
    return [f"disk floor: {free_bytes()} bytes free < {floor}"] if free_bytes() < floor else []


def held_basis_bytes(root=None):
    """Bytes of every file under the basis root (D:\\tanitad\\zbasis): the files a basis re-run reuses (spool) or replaces in place (caches, fit)."""
    r = root or loc(ZB_DECL)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(r) for f in fs) if os.path.isdir(r) else 0


def basis_need_bytes(held):
    return max(NEED_GB_RUN * 1e9, NEED_GB_BASIS * 1e9 - max(0, held))


def basis_disk_problems(free, held):
    need = basis_need_bytes(held)
    return [f"D: free {free / 1e9:.2f} GB < {need / 1e9:.2f} GB needed ({NEED_GB_BASIS} GB less {held / 1e9:.2f} GB of basis files reused or replaced in place) (disk-low)"] if free < need else []


def run_dir(tag):
    return os.path.join(home(), "runs", tag)


def token_path(tag):
    return os.path.join(run_dir(tag), "gate", "gate_token.json")


def trainer_path():
    return os.path.join(repo(), *TRAINER_REL.split("/"))


def binding(tag, line):
    md5s, bad = data_md5()
    try:
        tm = sha_file(os.path.join(home(), "tree_manifest.json"))
    except OSError:
        tm = None
    br = basis_record()
    return {"tag": tag, "commit": commit(), "argv_sha256": tokens_sha(line.split(" ")), "data_md5": md5s, "tree_manifest_sha256": tm, "trainer_sha256": sha_file(trainer_path()) if os.path.exists(trainer_path()) else None,
            "basis_sha256": (br.get("basis_sha256") or {}).get(TABLE[base_tag(tag)]["kind"])}, bad


def token_ok(tag, line):
    try:
        tok = json.load(open(token_path(tag), encoding="utf-8"))
    except (OSError, ValueError):
        return ["no gate token"]
    want, bad = binding(tag, line)
    p = list(bad)
    if tok.get("written_by") != "slowwm_check.py gate":
        p.append("token not written by the checker")
    p += [f"token {k} differs from the current {k}" for k, v in want.items() if tok.get(k) != v]
    sc_p = os.path.join(run_dir(tag), "gate", "selfcheck.json")
    if not os.path.exists(sc_p) or sha_file(sc_p) != tok.get("selfcheck_sha256"):
        p.append("selfcheck.json differs from the token's")
    return p


def poll_s():
    return float(os.environ.get("SLOWWM_POLL_S", "5"))


def gpu_sample():
    try:
        r = subprocess.run([os.environ.get("SLOWWM_NVSMI", "nvidia-smi"), "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, encoding="utf-8", errors="replace", timeout=20)
        a = r.stdout.strip().splitlines()[0].split(",")
        return float(a[0]), float(a[1])
    except Exception:                                                                    # noqa: BLE001  a missing sampler is no sample, never a number
        return None


def run_trainer(toks, log_path, sample=False):
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


def time_pilot(log_path):
    """mean / median s/step over steps 50..200 from the cumulative `sec` column; ValueError when not one uninterrupted run"""
    recs = []
    for line in open(log_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if isinstance(r, dict) and isinstance(r.get("step"), int) and num(r.get("sec")):
                recs.append((r["step"], float(r["sec"])))
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
    n = len(RUN_ORDER)
    planned = mean_s * RUN_STEPS * n / 3600
    return {"n_runs_planned": n, "run_steps": RUN_STEPS, "steps_total": RUN_STEPS * n, "projected_total_h_planned": planned, "per_run_h": mean_s * RUN_STEPS / 3600, "limit_h": LIMIT_H, "within_limit_planned": planned <= LIMIT_H}


def rm_own_temp(pth, parent_name):
    """delete one of this job's own temporary dirs by literal path; only directly under a folder named parent_name -> {path, bytes} or None"""
    if not os.path.isdir(pth) or os.path.basename(os.path.dirname(pth)) != parent_name:
        return None
    size = sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(pth) for f in fs)
    shutil.rmtree(pth)
    return {"path": pth, "bytes": size}


# ---- the basis stage (card control 3 on real train frames) --------------------------------------------------------------------------------------------------------------------------------------
def basis_dir():
    return loc(BASIS_DIR_DECL)


def basis_counts():
    t = os.environ.get("SLOWWM_TEST_COUNTS")
    return {k: int(v) for k, v in (x.split("=") for x in t.split(","))} if dry() and t else dict(COUNTS)


def basis_count_problems(root):
    p, found = [], {}
    for k in KINDS:
        for split, n in basis_counts().items():
            d = os.path.join(root, k, split)
            names = os.listdir(d) if os.path.isdir(d) else []
            found[f"{k}/{split}"] = sum(1 for x in names if x.endswith(".npz") and x != "basis.npz")
            if found[f"{k}/{split}"] != n:
                p.append(f"basis {k}/{split}: {found[f'{k}/{split}']} clip files, wanted {n}")
            if any(x.endswith(".tmp") for x in names):
                p.append(f"basis {k}/{split}: stray .tmp file")
            if not os.path.isfile(os.path.join(d, "BASIS_MANIFEST.json")):
                p.append(f"basis {k}/{split}: BASIS_MANIFEST.json missing")
    return found, p


#: AMENDMENT 2 (AMEND_SLOWWM_2.md): check 3 runs once per split of clips8.json (eval 6 clips, train 2 clips) and the stage combines the two records, so the identity covers 8 DISTINCT clips;
#: SLOW's cov check is the identity its normalisation implies, V^T (C + ridge I) V = I (equivalently cov(u) = I - ridge V^T V) within 1e-3; PCA-B keeps cov(u) = I within 1e-3.
CHECK3_SCHEMA, CHECK3_SPLITS, N_CLIPS, COV_TOL = "slowwm-check3-v2", ("eval", "train"), 8, 1e-3


def num_ok(x, lim):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf") and 0 <= x <= lim


def cov_identity(basis_dir_path):
    """From the SHIPPED basis_slow.npz / basis_pca.npz and fit_stats.npz (keys: proj, mean, meta_json['ridge']; C): SLOW's V^T (C + ridge I) V - I and cov(u) - (I - ridge V^T V) (the same quantity, two routes),
    PCA-B's cov(u) - I, and SLOW's per-direction deviation of cov(u) from I (beside number). Needs numpy; the dev box never calls it (it reads the numbers in the record)."""
    import numpy as np
    with np.load(os.path.join(basis_dir_path, "basis_slow.npz"), allow_pickle=False) as z:
        V, ridge = np.asarray(z["proj"], np.float64), float(json.loads(str(z["meta_json"]))["ridge"])
    with np.load(os.path.join(basis_dir_path, "basis_pca.npz"), allow_pickle=False) as z:
        P = np.asarray(z["proj"], np.float64)
    with np.load(os.path.join(basis_dir_path, "fit_stats.npz")) as z:
        C = np.asarray(z["C"], np.float64)
    k, I = V.shape[1], np.eye(V.shape[1])
    cs, cp = V.T @ C @ V, P.T @ C @ P
    vbv = np.abs(V.T @ (C + ridge * np.eye(C.shape[0])) @ V - I).max()
    via_cov = np.abs(cs - (I - ridge * (V.T @ V))).max()
    dev = np.abs(np.diag(cs) - 1.0)
    return {"tol": COV_TOL, "ridge": ridge, "k": int(k), "slow_identity_max_dev": float(max(vbv, via_cov)), "slow_identity_vbv_max_dev": float(vbv), "slow_identity_via_cov_max_dev": float(via_cov),
            "slow_identity_pass": bool(max(vbv, via_cov) <= COV_TOL), "pca_max_dev": float(np.abs(cp - I).max()), "pca_pass": bool(np.abs(cp - I).max() <= COV_TOL),
            "slow_cov_u_dev_from_I_per_direction": [float(x) for x in dev], "slow_cov_u_dev_from_I_diag_max": float(dev.max()), "slow_cov_u_dev_from_I_offdiag_max": float(np.abs(cs - np.diag(np.diag(cs))).max()),
            "note": "beside number: SLOW's cov(u) is I - ridge V^T V by the card's normalisation (AMEND_SLOWWM_2 B); it is reported, not pass-or-fail"}


def _same_part(ev, tr, key):
    return json.dumps(ev.get(key), sort_keys=True) == json.dumps(tr.get(key), sort_keys=True)


def _body_problems(res):
    """every refusal of a combined CHECK3 record except the consistency of its own overall `pass`"""
    p = []
    if res.get("schema") != CHECK3_SCHEMA:
        p.append(f"check3 record schema {res.get('schema')!r} is not {CHECK3_SCHEMA!r} (one record per split, combined)")
    if res.get("SMOKE"):
        p.append("check3 is a SMOKE record")
    sp, clips, union = res.get("splits") or {}, res.get("clips") or [], []
    for s in CHECK3_SPLITS:
        r = sp.get(s)
        if not isinstance(r, dict):
            p.append(f"check3 has no record for split {s}")
            continue
        union += list(r.get("clips") or [])
        if not r.get("clips"):
            p.append(f"check3 split {s} ran on no clips")
        if r.get("pass") is not True:
            p.append(f"check3 split {s}: the per-split record's pass is not true")
        if r.get("SMOKE"):
            p.append(f"check3 split {s} is a SMOKE record")
        for k in KINDS:
            i = (r.get("identity") or {}).get(k) or {}
            if i.get("pass") is not True or i.get("wrong_kind_fails_identity") is not True:
                p.append(f"check3 identity {k} on split {s}: {i}")
    if len(set(union)) < N_CLIPS or len(set(clips)) < N_CLIPS:
        p.append(f"check3 ran on fewer than {N_CLIPS} clips ({len(set(union))} distinct in the splits, {len(set(clips))} distinct in the record)")
    elif len(union) != len(set(union)) or sorted(union) != sorted(clips):
        p.append("check3 clips: the splits overlap or do not add up to the record's clip list")
    for k in KINDS:
        i = (res.get("identity") or {}).get(k) or {}
        if i.get("pass") is not True or i.get("wrong_kind_fails_identity") is not True:
            p.append(f"check3 identity {k} (combined): {i}")
    c = res.get("cov_identity")
    if not isinstance(c, dict):
        p.append("check3 has no cov_identity block (SLOW's V^T (C + ridge I) V = I and PCA-B's cov(u) = I)")
    else:
        if not num_ok(c.get("slow_identity_max_dev"), COV_TOL):
            p.append(f"check3 SLOW cov identity V^T (C + ridge I) V = I: max deviation {c.get('slow_identity_max_dev')!r} > {COV_TOL}")
        if not num_ok(c.get("pca_max_dev"), COV_TOL):
            p.append(f"check3 PCA-B cov(u) = I: max deviation {c.get('pca_max_dev')!r} > {COV_TOL}")
        if c.get("slow_identity_pass") is not True or c.get("pca_pass") is not True:
            p.append(f"check3 cov flags: slow_identity_pass={c.get('slow_identity_pass')!r} pca_pass={c.get('pca_pass')!r}")
        d = c.get("slow_cov_u_dev_from_I_per_direction")
        if not (isinstance(d, list) and d and all(num_ok(x, 1e9) for x in d)):
            p.append("check3 lacks SLOW's per-direction deviation of cov(u) from I (beside number)")
    if not (isinstance(res.get("slow_vs_pca"), dict) and res["slow_vs_pca"].get("pass") is True):
        p.append("check3 SLOW-vs-PCA autocorrelation comparison did not pass")
    if not (isinstance(res.get("shuffled_time_basis"), dict) and res["shuffled_time_basis"].get("pass") is False):
        p.append("check3 shuffled-time basis did not fail as planted")
    return p


def check3_problems(res):
    if not isinstance(res, dict):
        return ["CHECK3.json unreadable"]
    p = _body_problems(res)
    if res.get("pass") is not True:
        p.append("check3 pass is not true (identity on 8 clips, cov identities, SLOW-vs-PCA autocorrelation, or the shuffled-time basis did not fail as planted)")
    elif p:
        p.append("check3 overall pass is true while a check above fails: the record is refused")
    return p


def build_check3(ev, tr, cov):
    """combine the two slow_basis.py check3 records (split eval, split train) and the cov identity block into the ONE CHECK3 record; its `pass` is derived here, from the same predicate the checker re-applies"""
    sp = {"eval": ev, "train": tr}
    clips = [c for s in CHECK3_SPLITS for c in (sp[s].get("clips") or [])]
    res = {"schema": CHECK3_SCHEMA, "tol_identity": ev.get("tol_identity"), "tol_cov": COV_TOL, "SMOKE": bool(ev.get("SMOKE") or tr.get("SMOKE")), "clips": clips, "splits": {s: {k: v for k, v in sp[s].items()} for s in CHECK3_SPLITS},
           "identity": {}, "cov_identity": cov, "slow_vs_pca": ev.get("slow_vs_pca"), "shuffled_time_basis": ev.get("shuffled_time_basis"), "cov_u_fit_frames_raw": ev.get("cov_u_fit_frames"),
           "pass_cov_slow_card_tolerance_raw": ev.get("pass_cov_slow_card_tolerance")}
    for k in KINDS:
        ii = [(sp[s].get("identity") or {}).get(k) or {} for s in CHECK3_SPLITS]
        res["identity"][k] = {"pass": all(i.get("pass") is True for i in ii), "wrong_kind_fails_identity": all(i.get("wrong_kind_fails_identity") is True for i in ii),
                              "max_abs_diff": max((i.get("max_abs_diff", float("inf")) for i in ii), default=float("inf")), "wrong_kind_basis_max_abs_diff": min((i.get("wrong_kind_basis_max_abs_diff", 0.0) for i in ii), default=0.0)}
    res["pass"] = True
    body = _body_problems(res)
    if not _same_part(ev, tr, "slow_vs_pca") or not _same_part(ev, tr, "shuffled_time_basis"):
        body.append("the eval and train records disagree on the basis-level comparisons")
    res["pass"] = not body
    if body:
        res["refusals"] = body[:20]
    return res


def fit_problems(rec):
    if not isinstance(rec, dict):
        return ["FIT_RECORD.json unreadable"]
    m, p = rec.get("meta") or {}, []
    if m.get("n_windows_ok") != EXPECT_WINDOWS:
        p.append(f"fit n_windows_ok {m.get('n_windows_ok')!r} != {EXPECT_WINDOWS}")
    if m.get("smoke"):
        p.append("the fit is a SMOKE")
    if m.get("card_sha256") != CARD_SHA256:
        p.append("fit meta.card_sha256 is not the frozen card's")
    if not all(isinstance(rec.get("basis_sha256", {}).get(k), str) and HEX64.match(rec["basis_sha256"][k]) for k in KINDS):
        p.append("fit basis_sha256 missing")
    return p


def basis_fail(gid, why):
    write_text(os.path.join(basis_dir(), f"BASIS_{gid}.fail"), why[:900] + "\n")
    print(f"SLOWWM-BASIS-FAIL {why}"[:1200])
    return 5


def run_py(args, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", ZC_REPO=repo())
    with open(log_path, "ab") as lg:
        lg.write(f"===== {now()} {' '.join(args[1:3])}\n".encode())
        return subprocess.run([sys.executable, *args], stdout=lg, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env).returncode


def cmd_basis(gid):
    os.makedirs(basis_dir(), exist_ok=True)
    cc = loc(CODE_DECL)
    bad = basis_disk_problems(free_bytes(), held_basis_bytes()) + tree_problems()
    for k in (0, 1):
        ic = loc(M2B_DECL + rf"\M2b_s{k}\ckpt_last.pt")
        if not os.path.exists(ic) or sha_file(ic) != init_sha(k):
            bad.append(f"start state M2b_s{k} missing or its sha256 differs from the constant")
    for f, want in BASIS_CODE_SHA256.items():
        fp = os.path.join(cc, f)
        if not os.path.isfile(fp):
            bad.append(f"{f} is not in {CODE_DECL}: run the ship stage first")
        elif sha_file(fp) != want and not (dry() and os.environ.get("SLOWWM_TEST_BASIS")):
            bad.append(f"{f} in {CODE_DECL} differs from the pinned sha256")
    if bad:
        return basis_fail(gid, "preflight: " + " | ".join(bad)[:800])
    sb = os.environ["SLOWWM_TEST_BASIS"] if dry() and os.environ.get("SLOWWM_TEST_BASIS") else os.path.join(cc, "slow_basis.py")
    root, tracks, zb, fit, spool, lg = loc(CACHE_ROOT), loc(TRACKS), loc(ZB_DECL), loc(FIT_DECL), loc(SPOOL_DECL), os.path.join(basis_dir(), "basis.log")
    steps = [("fit", [sb, "fit", "--root", root, "--split", "train", "--tracks", tracks, "--exclude", "540:600", "--expect-windows", str(EXPECT_WINDOWS), "--out", fit, "--spool", spool])]
    for k in KINDS:
        for split in ("train", "eval"):
            steps.append((f"cache-{k}-{split}", [sb, "cache", "--root", root, "--split", split, "--tracks", tracks, "--basis", os.path.join(fit, f"basis_{k}.npz"), "--out", zb, "--spool", spool, "--commit", commit()]))
    c3 = os.path.join(basis_dir(), "CHECK3.json")
    c3s = {s_: os.path.join(basis_dir(), f"CHECK3.{s_}.json") for s_ in CHECK3_SPLITS}
    for s_ in CHECK3_SPLITS:                    # AMENDMENT 2 A: one check3 per split of clips8.json (each against that split's cache), combined below
        steps.append((f"check3-{s_}", [sb, "check3", "--root", root, "--split", s_, "--tracks", tracks, "--basis-dir", fit, "--cache-dir", zb, "--clips-file", os.path.join(cc, "clips8.json"), "--out-json", c3s[s_]]))
    for name, args in steps:
        rc = run_py(args, lg)
        if rc != 0:
            return basis_fail(gid, f"slow_basis.py {name} exited rc={rc} (log {lg})")
    try:
        parts, rec = {s_: json.load(open(c3s[s_], encoding="utf-8")) for s_ in CHECK3_SPLITS}, json.load(open(os.path.join(fit, "FIT_RECORD.json"), encoding="utf-8"))
        res = build_check3(parts["eval"], parts["train"], cov_identity(fit))
        write_text(c3, json.dumps(res, indent=1))
    except (OSError, ValueError, KeyError, ImportError) as e:
        return basis_fail(gid, f"check3 / fit record not written or not combinable ({type(e).__name__}: {str(e)[:120]})")
    bad = check3_problems(res) + fit_problems(rec)
    if bad:
        return basis_fail(gid, "check3-refused: " + " | ".join(bad)[:800])
    found, bad = basis_count_problems(zb)
    if bad:
        return basis_fail(gid, "counts: " + " | ".join(bad)[:800])
    mans = {}
    for k in KINDS:
        for split in basis_counts():
            mp = os.path.join(zb, k, split, "BASIS_MANIFEST.json")
            m = json.load(open(mp, encoding="utf-8"))
            if m.get("basis_sha256") != rec["basis_sha256"][k] or m.get("kind") != k or m.get("smoke"):
                return basis_fail(gid, f"counts: manifest {k}/{split} is not the fitted {k} basis")
            mans[f"{k}/{split}"] = {"basis_sha256": m.get("basis_sha256"), "n_clips": m.get("n_clips"), "manifest_sha256": sha_file(mp)}
    for f in ("FIT_RECORD.json", "SPECTRA.json"):                                       # copies beside CHECK3.json so the dev box pulls one folder
        shutil.copy2(os.path.join(fit, f), os.path.join(basis_dir(), f))
    write_text(os.path.join(basis_dir(), "BASIS.json"), json.dumps({"schema": "slowwm-basis-v1", "at": now(), "commit": commit(), "counts": found, "expected_counts": basis_counts(), "manifests": mans, "basis_sha256": rec["basis_sha256"],
               "fit_record_sha256": sha_file(os.path.join(basis_dir(), "FIT_RECORD.json")), "spectra_sha256": sha_file(os.path.join(basis_dir(), "SPECTRA.json")), "check3_sha256": sha_file(c3), "n_windows_ok": rec["meta"]["n_windows_ok"]}, indent=1))
    write_text(os.path.join(basis_dir(), f"BASIS_{gid}.done"), "done\n")
    print("SLOWWM-OK basis")
    return 0


def cmd_verify_basis(a):
    """dev box: re-judge the pulled BASIS.json, FIT_RECORD.json, CHECK3.json (sha256 chain, counts, windows, check3, commit)"""
    try:
        rec, fit, c3 = (json.load(open(a[k], encoding="utf-8")) for k in ("basis-json", "fit-json", "check3-json"))
    except (OSError, ValueError) as e:
        print(f"basis refused: unreadable ({type(e).__name__})")
        return 11
    p = check3_problems(c3) + fit_problems(fit)
    if rec.get("commit") != a["commit"]:
        p.append(f"BASIS.json is for commit {rec.get('commit')!r}")
    for key, f in (("fit_record_sha256", "fit-json"), ("check3_sha256", "check3-json")):
        if rec.get(key) != sha_file(a[f]):
            p.append(f"{f} differs from the sha256 BASIS.json recorded")
    if rec.get("basis_sha256") != fit.get("basis_sha256"):
        p.append("BASIS.json and FIT_RECORD.json carry different basis sha256")
    want = {f"{k}/{sp}": n for k in KINDS for sp, n in basis_counts().items()}
    if rec.get("counts") != want:
        p.append(f"counts {rec.get('counts')!r}, wanted {want!r}")
    for k in KINDS:
        for sp in basis_counts():
            if (rec.get("manifests") or {}).get(f"{k}/{sp}", {}).get("basis_sha256") != (rec.get("basis_sha256") or {}).get(k):
                p.append(f"manifest {k}/{sp} does not carry the fitted {k} basis sha256")
    if len(set((rec.get("basis_sha256") or {}).values())) != 2:
        p.append("slow and pca carry the same basis sha256")
    if p:
        print("basis refused: " + "; ".join(p))
        return 11
    print("basis ok")
    return 0


# ---- the box-2 job (stage / pilot / cleanup) and the dev-box commands: as WMPOOL's, with the basis in place of the cells -------------------------------------------------------------------
def stage_fail(tag, gid, kind, why):
    write_text(os.path.join(run_dir(tag), f"STAGE_{gid}.fail"), f"{kind}: {why}"[:900] + "\n")
    print(f"SLOWWM-FAIL {tag} {kind}: {why}"[:1200])
    return {"preflight-fail": 3, "pin-fail": 3, "gate-fail": 4, "token-fail": 4, "train-fail": 5, "post-fail": 6, "disk-low": 9}.get(kind, 5)


def do_gate(tag, line):
    toks, rd = line.split(" "), run_dir(tag)
    gd = os.path.join(rd, "gate")
    if not token_ok(tag, line):
        print(f"gate-cached {tag}")
        return []
    if os.path.exists(gd):
        shutil.move(gd, gd + ".prev_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(gd)
    gout = loc(gate_out_decl(tag))
    rm_own_temp(gout, "gate_out")
    rc, _, _ = run_trainer(gate_argv(tag, toks), os.path.join(rd, "logs", "gate.log"))
    scp = os.path.join(gout, "selfcheck.json")
    if rc != 0 or not os.path.exists(scp):
        return [f"the trainer's --selfcheck-only exited rc={rc} and {'wrote' if os.path.exists(scp) else 'did not write'} selfcheck.json (log {rd}\\logs\\gate.log)"]
    shutil.copy2(scp, os.path.join(gd, "selfcheck.json"))
    bad = gate_problems(tag, toks, json.load(open(scp, encoding="utf-8")))
    if bad:
        write_text(os.path.join(rd, "RUN.FAILED"), json.dumps({"tag": tag, "at": now(), "problems": bad}, indent=1))
        return bad
    b, dbad = binding(tag, line)
    if dbad:
        return dbad
    write_text(token_path(tag), json.dumps({"schema": "slowwm-gate-token-v1", "written_by": "slowwm_check.py gate", "at": now(), "selfcheck_sha256": sha_file(os.path.join(gd, "selfcheck.json")), **b}, indent=1))
    return []


def cmd_stage(tag, gid):
    if tag != PILOT_TAG and tag not in TABLE:
        return stage_fail(tag, gid, "pin-fail", f"unknown tag {tag}")
    rd, spec = run_dir(tag), TABLE[base_tag(tag)]
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
    ic = loc(M2B_DECL + rf"\M2b_s{spec['init']}\ckpt_last.pt")
    if not os.path.exists(ic) or sha_file(ic) != init_sha(spec["init"]):
        bad.append(f"start state M2b_s{spec['init']} missing or its sha256 differs from the constant")
    zd = loc(zb_train_decl(spec["kind"]))
    if not os.path.isfile(os.path.join(zd, "BASIS_MANIFEST.json")):
        bad.append(f"the basis cache {zb_train_decl(spec['kind'])} is missing or has no BASIS_MANIFEST.json (run the basis stage first)")
    if not os.path.isfile(os.path.join(basis_dir(), "BASIS.json")):
        bad.append("no BASIS.json: the basis stage (fit, cache, check3) has not passed on this machine")
    if bad:
        return stage_fail(tag, gid, "preflight-fail", " | ".join(bad)[:800])
    bad = audit_line(tag, line)
    if bad:
        return stage_fail(tag, gid, "pin-fail", bad[0])
    jobs = [(tag, line)]
    if is_pilot(tag):
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
        rm_own_temp(out, "pilot_out")
    rc, wall, samples = run_trainer(toks, os.path.join(rd, "logs", "train.log"), sample=is_pilot(tag))
    summ, logp = os.path.join(out, "summary.json"), os.path.join(out, "train_log.jsonl")
    if rc != 0 or not (os.path.exists(summ) and os.path.exists(logp)):
        return stage_fail(tag, gid, "train-fail", f"trainer rc={rc}; summary {os.path.exists(summ)}; log {os.path.exists(logp)}")
    ev, p = post_problems(tag, toks, out)
    if p:
        return stage_fail(tag, gid, "post-fail", " | ".join(p)[:800])
    if is_pilot(tag):
        pd = os.path.join(home(), "pilot")
        try:
            tm = time_pilot(logp)
        except ValueError as e:
            return stage_fail(tag, gid, "post-fail", str(e))
        os.makedirs(pd, exist_ok=True)
        for f in ("train_log.jsonl", "summary.json"):
            shutil.copy2(os.path.join(out, f), os.path.join(pd, f))
        shutil.copy2(os.path.join(rd, "gate", "selfcheck.json"), os.path.join(pd, "selfcheck.json"))
        shutil.copy2(token_path(tag), os.path.join(pd, "gate_token.json"))
        util = [u for _, u in samples]
        res = {"schema": "slowwm-pilot-v1", "at": now(), "commit": commit(), "tag": tag, "argv_sha256": tokens_sha(line.split(" ")), "pilot_steps": PILOT_STEPS, "wall_s_total_incl_startup": wall, "timing": tm,
               "peak_gpu_mib": max((m for m, _ in samples), default=None), "gpu_util_median_pct": statistics.median(util) if util else None, "train_log_sha256": sha_file(os.path.join(pd, "train_log.jsonl")),
               "projection": project(tm["mean_s_per_step"]), "full_run_gates_at_pilot": {t: os.path.exists(token_path(t)) for t in GATE_AT_PILOT}}
        write_text(os.path.join(pd, "PILOT.json"), json.dumps(res, indent=1))
        write_text(os.path.join(rd, f"STAGE_{gid}.done"), "done\n")
        print(f"SLOWWM-OK {tag} mean {tm['mean_s_per_step']:.4f} s/step, {res['projection']['projected_total_h_planned']:.1f} h planned")
        return 0
    write_text(os.path.join(rd, "RUN.done"), json.dumps({"tag": tag, "at": now(), "commit": commit(), "weights_identity": ev, "basis_sha256": binding(tag, line)[0]["basis_sha256"],
               "evidence": {"ckpt_sha256": sha_file(os.path.join(out, "ckpt_last.pt")), "summary_sha256": sha_file(summ), "train_log_sha256": sha_file(logp)}}, indent=1))
    write_text(os.path.join(rd, f"STAGE_{gid}.done"), "done\n")
    print(f"SLOWWM-OK {tag}")
    return 0


def cmd_cleanup(tag, gid):
    """AFTER the dev box proved the bank (BANK_VERIFIED_<gid>): delete this run's own gate dirs by literal path; log (path, bytes). Never a checkpoint, never a basis cache."""
    rd = run_dir(tag)
    if not os.path.exists(os.path.join(rd, f"BANK_VERIFIED_{gid}")):
        write_text(os.path.join(rd, f"CLEAN_{gid}.fail"), "no BANK_VERIFIED marker: the bank was not proven, nothing is deleted\n")
        print("SLOWWM-CLEAN-REFUSED no bank proof")
        return 6
    del_ = [d for t in (([tag] + GATE_AT_PILOT) if is_pilot(tag) else [tag]) for d in [rm_own_temp(loc(gate_out_decl(t)), "gate_out")] if d]
    if is_pilot(tag):
        d = rm_own_temp(loc(out_decl(tag)), "pilot_out")
        del_ += [d] if d else []
    write_text(os.path.join(rd, "CLEANUP.json"), json.dumps({"tag": tag, "at": now(), "deleted_own_temp": del_}, indent=1))
    write_text(os.path.join(rd, f"CLEAN_{gid}.done"), "done\n")
    print(f"SLOWWM-CLEAN-OK {tag} deleted {len(del_)}")
    return 0


def cmd_verify_token(a):
    tok = json.load(open(a["token"], encoding="utf-8"))
    line = [l.strip() for l in open(a["argv-file"], encoding="utf-8") if l.strip()][0]
    p = [] if tok.get("written_by") == "slowwm_check.py gate" else ["not written by the checker"]
    if tok.get("tag") != a["tag"] or tok.get("commit") != a["commit"] or tok.get("argv_sha256") != tokens_sha(line.split(" ")):
        p.append("token does not bind this tag, commit and argv")
    if audit_line(a["tag"], line):
        p.append("the argv is not the frozen one")
    print("token refused: " + "; ".join(p) if p else "token ok")
    return 1 if p else 0


def cmd_project(a):
    pd = a["pilot-dir"]
    res = json.load(open(os.path.join(pd, "PILOT.json"), encoding="utf-8"))
    tm = time_pilot(os.path.join(pd, "train_log.jsonl"))
    p = [f"the pilot timed commit {res.get('commit')}"] if res.get("commit") != a["commit"] else []
    if sha_file(os.path.join(pd, "train_log.jsonl")) != res.get("train_log_sha256"):
        p.append("train_log sha256 differs from PILOT.json")
    pr = project(tm["mean_s_per_step"])
    if abs(tm["mean_s_per_step"] - res["timing"]["mean_s_per_step"]) > 1e-9 or abs(pr["projected_total_h_planned"] - res["projection"]["projected_total_h_planned"]) > 1e-6:
        p.append("mean s/step or projection recomputed here differs from box 2's")
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
    if f"pilot_sha256={sha_file(a['pilot'])}" not in txt or f"commit={a['commit']}" not in txt:
        print(f"refused: the amendment {am} must contain 'pilot_sha256={sha_file(a['pilot'])}' and 'commit={a['commit']}'")
        return 3
    print(f"permit: projected {total:.1f} h > {LIMIT_H:.0f} h, amendment {am} binds this pilot and commit")
    return 0


def cmd_render(a):
    t = open(a["template"], encoding="utf-8").read()
    for _ in range(3):
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
            sets[k], i = v, i + 2
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
        line = " ".join(tokens(tag))
        write_text(a["out-txt"], line + "\n") if a.get("out-txt") else print(line)
        return 0
    table = {"render": lambda: cmd_render(a), "verify-basis": lambda: cmd_verify_basis(a), "basis": lambda: cmd_basis(a["gid"]), "stage": lambda: cmd_stage(a["tag"], a["gid"]),
             "cleanup": lambda: cmd_cleanup(a["tag"], a["gid"]), "verify-token": lambda: cmd_verify_token(a), "project": lambda: cmd_project(a), "permit": lambda: cmd_permit(a)}
    if cmd == "disk-basis":               # dev box, before the basis job: --free-bytes and --held-bytes read from box 2; 0 = enough, 13 = disk-low, 19 = unreadable
        try:
            if not all(isinstance(a.get(k), str) and a[k].isdigit() for k in ("free-bytes", "held-bytes")):
                raise ValueError("not a whole number of bytes")
            free, held = float(a["free-bytes"]), float(a["held-bytes"])
        except (KeyError, TypeError, ValueError):
            print("disk-basis: --free-bytes / --held-bytes missing or not a number")
            return 19
        bad = basis_disk_problems(free, held)
        print(f"disk-basis OK: free {free / 1e9:.2f} GB >= {basis_need_bytes(held) / 1e9:.2f} GB needed (held {held / 1e9:.2f} GB)" if not bad else bad[0])
        return 0 if not bad else 13
    if cmd == "preflight":
        bad = disk_floor_problems() + tree_problems()
        print("preflight OK" if not bad else "SLOWWM-PREFLIGHT-FAIL " + " | ".join(bad))
        return 0 if not bad else 3
    if cmd in table:
        return table[cmd]()
    print(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
