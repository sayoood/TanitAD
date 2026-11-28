"""n1dm_dumps.py -- STAGED. T1-kin eval-141 dumps for the N1DM card (PREREG_V8F_N1DM.md sha256 N1DM-DRAFT, section 3 guard): N1DM_s{0,1} and N1D_s{0,1} (each with its zero twin Z) under the executor of record
T1-kin/hb0.2 (stack/scripts/v8f_t1_eval_hb.py --kappa-max 0.2, sha256 74fd8445 = blob at bcf267ce), and N1DM_s{0,1} under the old executor (--kappa-max none), dev box CPU, 2 threads, zero training.
Adapted from n1dw_dumps.py (N1DW, ran 2026-11-20): the arm list, the tags (N1DM-<ARM>[Z]), the paths, and the executor routing change; the pinned evaluator (v8f_t1_eval.py sha256 180044eb), the
thread count, the grid, the join and every --nav-rb flag are the N1DW dump's.  Routing, as the pod's hb_dump_n1dm.sh does for D1-HO: argv[1] = tree/stack/scripts/v8f_t1_eval_hb.py with
`--kappa-max K --inner v8f_t1_eval_rbd --hb-log <out>/hb_<tag>.json` ahead of `run ...`; the inner T1 tool is the study wrapper v8f_t1_eval_rbd.py of the the-N1DM-commit tree (the tool the N1DW dumps used).
Env: N1DM_KAPPA = 0.2 (default, record) | none (old executor, OUT = dumps141_old unless N1DM_D_141 is set).
Usage: python n1dm_dumps.py TAG [TAG ...]   TAG in N1DM-<ARM> / N1DM-<ARM>Z for ARM in N1DM_s0 N1DM_s1 N1D_s0 N1D_s1  (Z = the same checkpoint with --nav-rb-variant zero).
Checkpoints: ckpts/<ARM>/ckpt_last.pt (pod runs, pulled), checked against ckpts/SHA256_PODS.txt (the pod's own sha256sum)."""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _p in ("F:/TanitAD-scratch/strat1/navrb2/read", "F:/TanitAD-scratch/strat1/read"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import navrb2_lib as NL  # noqa: E402

sys.path.insert(0, NL.SR)
sys.path.insert(0, NL.SR + "/theory/otband/read")
import strat1_lib as L  # noqa: E402
import run_dumps as RD  # noqa: E402  (module level is definitions only)

ROOT = "F:/TanitAD-scratch/strat1/navrbd"      # NAVRBD tables: b2/tables/navtext_nav_rbd.npz (sha 298049df), the file N1D eval-141 dumps read
KAPPA = os.environ.get("N1DM_KAPPA", "0.2")
OUT = os.environ.get("N1DM_D_141", HERE + ("/dumps141_hb02" if KAPPA == "0.2" else "/dumps141_old" if KAPPA == "none" else "/dumps141_k" + KAPPA))
TREE = os.environ.get("N1DM_TREE", HERE + "/tree")          # git -c core.autocrlf=false archive the-N1DM-commit stack taniteval, plus stack/scripts/v8f_t1_eval_hb.py (blob bcf267ce)
CK = os.environ.get("N1DM_CKPTS", HERE + "/ckpts")                      # N1DM_s0/s1 (pulled from the pod when ARM_DONE) + SHA256_PODS.txt
CK_N1D = os.environ.get("N1DM_CKPTS_N1D", "F:/TanitAD-scratch/n1dc/read/eval141/ckpts")      # N1D_s0/s1 + SHA256_PODS.txt, already on the dev box (pulled by the N1DH chain, sha256 = the pod's own)


def ck_root(arm):
    return CK_N1D if arm.startswith("N1D_") else CK
RB_NPZ = ROOT + "/b2/tables/navtext_nav_rbd.npz"
RB_SHA256 = "298049df71dc469874dd2429781c9019c4727b917bc3bd2233314542f6bbc82a"
SMOKE_WINDOWS = int(os.environ.get("N1DM_SMOKE_WINDOWS", "0"))      # >0: --max-windows N (a code check, never a result)
D_ARMS = ("N1DM_s0", "N1DM_s1", "N1D_s0", "N1D_s1")
PFX = "N1DM-"
TAGS = tuple(f"{PFX}{a}{z}" for a in D_ARMS for z in ("", NL.Z))
def arm_of(t):
    a = t[len(PFX):]
    return a[:-1] if a.endswith(NL.Z) else a


def _pod_sha(arm):
    """the pod's own sha256sum line for <arm>/ckpt_last.pt in ckpts/SHA256_PODS.txt (step ckpt_sha of run_n1dm141.sh already checked the file against it)"""
    for ln in open(f"{ck_root(arm)}/SHA256_PODS.txt", encoding="utf-8").read().splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[1] == f"{arm}/ckpt_last.pt" and len(parts[0]) == 64:
            return parts[0]
    raise RuntimeError(f"no pod sha256 for {arm} in {ck_root(arm)}/SHA256_PODS.txt")


def _src(arm):
    # MM fix 2026-11-20: run_dumps.run() reads the expected sha256 unconditionally, so a missing BANK_MANIFEST.json raised FileNotFoundError.
    # The manifest is used when present; otherwise the pod's sha256sum record is the expected value.
    mp = f"{ck_root(arm)}/{arm}/BANK_MANIFEST.json"
    if os.path.isfile(mp):
        return ("bank", mp)
    try:
        return ("sha", _pod_sha(arm))
    except (OSError, RuntimeError):
        return ("sha", None)          # N1DM checkpoints are pulled when the pod's arm is ARM_DONE; until then refusal_ck says so (never a dump from a checkpoint with no recorded sha256)


ARMS = {t: (f"{ck_root(arm_of(t))}/{arm_of(t)}/ckpt_last.pt", _src(arm_of(t))) for t in TAGS}
for _t, _v in ARMS.items():
    RD.ARMS[_t] = _v
RD.OUT, RD.TREE = OUT, TREE
RD.INPUTS["--eval-grid"] = NL.SR + "/inputs/eval_grid_windows.json"
RD.INPUTS["--eval-join"] = NL.SR + "/inputs/join_eval141.npz"
if os.environ.get("N1DM_SMOKE_GRID"):
    RD.INPUTS["--eval-grid"] = os.environ["N1DM_SMOKE_GRID"]


def rb_npz_checked():
    got = NL.sha256_file(RB_NPZ)
    if got != RB_SHA256:
        raise RuntimeError(f"navtext_nav_rbd.npz sha256 {got[:12]} != pinned {RB_SHA256[:12]}")
    return RB_NPZ


def tree_problem():
    """the tree must hold the hb wrapper (sha256 74fd8445), the study wrapper and equal the STRAT1 pins for the two pinned evaluators and ci.py (NL.tree_problem), and v8f_t1_eval_rbd.py"""
    for f_ in ("v8f_t1_eval_rbd.py", "v8f_t1_eval_hb.py"):
        if not os.path.isfile(TREE + "/stack/scripts/" + f_):
            return f"tree file missing: {TREE}/stack/scripts/{f_} (export the-N1DM-commit to N1DM_TREE and add the hb wrapper)"
    if NL.sha256_file(TREE + "/stack/scripts/v8f_t1_eval_hb.py")[:8] != "74fd8445":
        return "v8f_t1_eval_hb.py sha256 is not 74fd8445"
    if KAPPA != "none" and not (KAPPA.replace(".", "", 1).isdigit() and float(KAPPA) > 0):
        return f"N1DM_KAPPA must be a positive number or none, got {KAPPA}"
    return NL.tree_problem(TREE)


def refusal_ck(arm, ck_root_=None):
    """-> reason or None: the checkpoint exists, a BANK_MANIFEST.json (when present) names its sha256 and it matches"""
    tag = arm[:-1] if arm.endswith(NL.Z) else arm
    root = ck_root_ or globals()["ck_root"](tag)
    if tag not in D_ARMS:
        return f"unknown arm {arm}"
    ck, mp = f"{root}/{tag}/ckpt_last.pt", f"{root}/{tag}/BANK_MANIFEST.json"
    if not os.path.isfile(ck):
        return f"REFUSED: checkpoint missing ({ck})"
    if not os.path.isfile(mp) and ARMS.get(f"{PFX}{tag}", (None, ("sha", None)))[1][1] is None:
        return f"REFUSED: no sha256 record for {tag} (neither {mp} nor a line in {root}/SHA256_PODS.txt)"
    if os.path.isfile(mp):
        want = ((json.load(open(mp, encoding="utf-8")).get("files") or {}).get("ckpt_last.pt") or {}).get("sha256")
        if not want or NL.sha256_file(ck) != want:
            return "REFUSED: checkpoint sha256 differs from (or is absent in) its bank manifest"
    return None


def variant_of(tag):
    return "zero" if tag.endswith(NL.Z) else "follow"


def extra_args(tag):
    """-> the arguments appended to the driver's argv (pure given files): text nav + rb flags; the Z twin names --nav-rb-variant zero"""
    npz = L.pinned("navtext_nav.npz", NL.SR + "/inputs")
    a = ["--nav-text-npz", npz, "--nav-text-sha256", L.PINS["navtext_nav.npz"], "--dump-nav",
         "--nav-rb", "on", "--nav-rb-feat", "exitdist", "--nav-rb-npz", rb_npz_checked(), "--nav-rb-sha256", RB_SHA256, "--nav-rb-variant", variant_of(tag)]
    return a + (["--max-windows", str(SMOKE_WINDOWS)] if SMOKE_WINDOWS else [])


class _Shim:
    """stands in for the `subprocess` module inside run_dumps"""
    cur = None
    PIPE, STDOUT = subprocess.PIPE, subprocess.STDOUT

    @classmethod
    def run(cls, argv, **kw):
        argv = list(argv)
        assert argv[1].replace("\\", "/").endswith("/stack/scripts/v8f_t1_eval.py"), argv[1]
        argv[1] = argv[1].replace("v8f_t1_eval.py", "v8f_t1_eval_hb.py")
        os.makedirs(OUT, exist_ok=True)
        argv = argv[:2] + ["--kappa-max", KAPPA, "--inner", "v8f_t1_eval_rbd", "--hb-log", f"{OUT}/hb_{cls.cur}.json"] + argv[2:]
        argv = argv + extra_args(cls.cur)
        env = dict(kw.get("env") or os.environ)
        env["PYTHONPATH"] = ";".join(os.path.normpath(p) for p in (TREE + "/stack", TREE + "/taniteval"))
        kw["env"] = env
        os.makedirs(OUT, exist_ok=True)
        with open(f"{OUT}/argv_{cls.cur}_m10_causal_heads.txt", "a", encoding="utf-8") as f:
            f.write("N1DM extra: " + " ".join(argv[-12:]) + "\n")
        return subprocess.run(argv, **kw)


RD.subprocess = _Shim


def refusal(tag):
    if tag not in ARMS:
        return f"unknown tag {tag}"
    return refusal_ck(arm_of(tag))


def run(tag):
    why = refusal(tag)
    if why:
        return {"tag": tag, "status": why}
    _Shim.cur = tag
    os.makedirs(OUT, exist_ok=True)
    return RD.run(tag)


def main():
    tags = sys.argv[1:]
    assert tags and all(t in ARMS for t in tags), f"tags must be among {list(ARMS)}"
    tp = tree_problem() or NL.import_check(RD.PY, TREE)
    if tp:
        raise SystemExit("REFUSED: " + tp)
    for k, v in RD.INPUTS.items():
        assert os.path.exists(v), (k, v)
    rep = {"tree": TREE, "out": OUT, "threads": 2, "smoke_windows": SMOKE_WINDOWS,
           "inputs_sha256": {k: (RD.sha(v) if os.path.isfile(v) else "directory") for k, v in RD.INPUTS.items()}, "runs": []}
    rc = 0
    for t in tags:
        r = run(t)
        print(time.strftime("%H:%M:%S"), json.dumps(r), flush=True)
        rep["runs"].append(r)
        if r.get("status") not in ("ok", "skip (done file exists)"):
            rc = 1
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(os.path.join(OUT, f"DUMPS_N1DM_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
