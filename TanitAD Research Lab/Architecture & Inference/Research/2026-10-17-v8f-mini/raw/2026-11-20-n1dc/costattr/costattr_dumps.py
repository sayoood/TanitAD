"""n1dc_dumps.py -- STAGED. T1-kin eval-141 dumps for the N1DC card (PREREG_V8F_N1DC.md sha256 2eae7964, section 3 guard): N1DC_s{0,1} and N1D_s{0,1} (each with its zero twin Z) under the executor of record
T1-kin/hb0.2 (stack/scripts/v8f_t1_eval_hb.py --kappa-max 0.2, sha256 74fd8445 = blob at bcf267ce), and N1DC_s{0,1} under the old executor (--kappa-max none), dev box CPU, 2 threads, zero training.
Adapted from n1dw_dumps.py (N1DW, ran 2026-11-20): the arm list, the tags (N1DC-<ARM>[Z]), the paths, and the executor routing change; the pinned evaluator (v8f_t1_eval.py sha256 180044eb), the
thread count, the grid, the join and every --nav-rb flag are the N1DW dump's.  Routing, as the pod's hb_dump_n1dc.sh does for D1-HO: argv[1] = tree/stack/scripts/v8f_t1_eval_hb.py with
`--kappa-max K --inner v8f_t1_eval_rbd --hb-log <out>/hb_<tag>.json` ahead of `run ...`; the inner T1 tool is the study wrapper v8f_t1_eval_rbd.py of the bcf3ba3f tree (the tool the N1DW dumps used).
Env: N1DC_KAPPA = 0.2 (default, record) | none (old executor, OUT = dumps141_old unless N1DC_D_141 is set).
Usage: python n1dc_dumps.py TAG [TAG ...]   TAG in N1DC-<ARM> / N1DC-<ARM>Z for ARM in N1DC_s0 N1DC_s1 N1D_s0 N1D_s1  (Z = the same checkpoint with --nav-rb-variant zero).
Checkpoints: ckpts/<ARM>/ckpt_last.pt (pod runs, pulled), checked against ckpts/SHA256_PODS.txt (the pod's own sha256sum)."""
import json
import os
import subprocess
import sys
import time

HERE = "F:/TanitAD-scratch/n1dc/costattr"
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
KAPPA = os.environ.get("N1DC_KAPPA", "0.2")
OUT = os.environ.get("N1DC_D_141", HERE + ("/dumps141_hb02" if KAPPA == "0.2" else "/dumps141_old" if KAPPA == "none" else "/dumps141_k" + KAPPA))
TREE = os.environ.get("N1DC_TREE", "F:/TanitAD-scratch/n1dc/read/eval141/tree")          # git -c core.autocrlf=false archive bcf3ba3f stack taniteval, plus stack/scripts/v8f_t1_eval_hb.py (blob bcf267ce)
CK = os.environ.get("N1DC_CKPTS", HERE + "/ckpts")
RB_NPZ = ROOT + "/b2/tables/navtext_nav_rbd.npz"
RB_SHA256 = "298049df71dc469874dd2429781c9019c4727b917bc3bd2233314542f6bbc82a"
SMOKE_WINDOWS = int(os.environ.get("N1DC_SMOKE_WINDOWS", "0"))      # >0: --max-windows N (a code check, never a result)
D_ARMS = ("N1DC_s0U", "N1DC_s1U")
PFX = "N1DC-"
TAGS = tuple(f"{PFX}{a}{z}" for a in D_ARMS for z in ("",))
def arm_of(t):
    return t[len(PFX):]
def cdir(arm):
    return arm[:-1] + "_zuse"


def _pod_sha(arm):
    """the pod's own sha256sum line for <arm>/ckpt_last.pt in ckpts/SHA256_PODS.txt (step ckpt_sha of run_n1dc141.sh already checked the file against it)"""
    for ln in open(f"{CK}/SHA256_PODS.txt", encoding="utf-8").read().splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[1] == f"{cdir(arm)}/ckpt_last.pt" and len(parts[0]) == 64:
            return parts[0]
    raise RuntimeError(f"no pod sha256 for {arm} in {CK}/SHA256_PODS.txt")


def _src(arm):
    # MM fix 2026-11-20: run_dumps.run() reads the expected sha256 unconditionally, so a missing BANK_MANIFEST.json raised FileNotFoundError.
    # The manifest is used when present; otherwise the pod's sha256sum record is the expected value.
    mp = f"{CK}/{cdir(arm)}/BANK_MANIFEST.json"
    return ("bank", mp) if os.path.isfile(mp) else ("sha", _pod_sha(arm))


ARMS = {t: (f"{CK}/{cdir(arm_of(t))}/ckpt_last.pt", _src(arm_of(t))) for t in TAGS}
for _t, _v in ARMS.items():
    RD.ARMS[_t] = _v
RD.OUT, RD.TREE = OUT, TREE
RD.INPUTS["--eval-grid"] = NL.SR + "/inputs/eval_grid_windows.json"
RD.INPUTS["--eval-join"] = NL.SR + "/inputs/join_eval141.npz"
if os.environ.get("N1DC_SMOKE_GRID"):
    RD.INPUTS["--eval-grid"] = os.environ["N1DC_SMOKE_GRID"]


def rb_npz_checked():
    got = NL.sha256_file(RB_NPZ)
    if got != RB_SHA256:
        raise RuntimeError(f"navtext_nav_rbd.npz sha256 {got[:12]} != pinned {RB_SHA256[:12]}")
    return RB_NPZ


def tree_problem():
    """the tree must hold the hb wrapper (sha256 74fd8445), the study wrapper and equal the STRAT1 pins for the two pinned evaluators and ci.py (NL.tree_problem), and v8f_t1_eval_rbd.py"""
    for f_ in ("v8f_t1_eval_rbd.py", "v8f_t1_eval_hb.py"):
        if not os.path.isfile(TREE + "/stack/scripts/" + f_):
            return f"tree file missing: {TREE}/stack/scripts/{f_} (export bcf3ba3f to N1DC_TREE and add the hb wrapper)"
    if NL.sha256_file(TREE + "/stack/scripts/v8f_t1_eval_hb.py")[:8] != "74fd8445":
        return "v8f_t1_eval_hb.py sha256 is not 74fd8445"
    if KAPPA != "none" and not (KAPPA.replace(".", "", 1).isdigit() and float(KAPPA) > 0):
        return f"N1DC_KAPPA must be a positive number or none, got {KAPPA}"
    return NL.tree_problem(TREE)


def refusal_ck(arm, ck_root=None):
    """-> reason or None: the checkpoint exists, a BANK_MANIFEST.json (when present) names its sha256 and it matches"""
    root = ck_root or CK
    tag = arm
    if tag not in D_ARMS:
        return f"unknown arm {arm}"
    ck, mp = f"{root}/{cdir(tag)}/ckpt_last.pt", f"{root}/{cdir(tag)}/BANK_MANIFEST.json"
    if not os.path.isfile(ck):
        return f"REFUSED: checkpoint missing ({ck})"
    if os.path.isfile(mp):
        want = ((json.load(open(mp, encoding="utf-8")).get("files") or {}).get("ckpt_last.pt") or {}).get("sha256")
        if not want or NL.sha256_file(ck) != want:
            return "REFUSED: checkpoint sha256 differs from (or is absent in) its bank manifest"
    return None


def variant_of(tag):
    return "follow"


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
            f.write("N1DC extra: " + " ".join(argv[-12:]) + "\n")
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
    json.dump(rep, open(os.path.join(OUT, f"DUMPS_COSTATTR_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
