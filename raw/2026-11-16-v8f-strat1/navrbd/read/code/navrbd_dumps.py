"""navrbd_dumps.py -- T1-kin dumps of the N1D arms on EVAL-141 (the guard cells, Z-EVAL, Z-USE beside), dev box CPU, 2 threads, zero training.
Usage: python navrbd_dumps.py TAG [TAG ...]   TAG in NAVRBD-N1D_s0 NAVRBD-N1D_s1 and their zero-channel twins NAVRBD-N1D_s0Z NAVRBD-N1D_s1Z (the same checkpoint with --nav-rb-variant zero).
A thin wrapper over the UNEDITED otband run_dumps.py (as navrb2_dumps.py): the shim swaps v8f_t1_eval.py for the study wrapper v8f_t1_eval_rbd.py of the NAVRBD tree (env NAVRBD_TREE, an export of the
build commit; the pod tree is 0c3b374f), appends --nav-text-npz / --dump-nav / --nav-rb on --nav-rb-feat exitdist --nav-rb-npz <navtext_nav_rbd.npz> --nav-rb-sha256 <pin> --nav-rb-variant follow|zero.
Checkpoints: ckpts/<ARM>/ckpt_last.pt (pulled from the pod runs/<ARM>/train/ckpt_last.pt); an optional ckpts/<ARM>/BANK_MANIFEST.json is checked (sha256); the checkpoint sha256 is recorded in
the DUMPS json either way.  One environment, 2 threads."""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrbd_read as NR  # noqa: E402
import navrb2_lib as NL  # noqa: E402

sys.path.insert(0, NL.SR)
sys.path.insert(0, NL.SR + "/theory/otband/read")
import strat1_lib as L  # noqa: E402
import run_dumps as RD  # noqa: E402  (module level is definitions only)

OUT = os.environ.get("NAVRBD_D_141", NR.HERE + "/dumps141")
TREE = os.environ.get("NAVRBD_TREE", NR.ROOT + "/tree")
CK = os.environ.get("NAVRBD_CKPTS", NR.HERE + "/ckpts")
RB_NPZ = NR.ROOT + "/b2/tables/navtext_nav_rbd.npz"
RB_SHA256 = "298049df71dc469874dd2429781c9019c4727b917bc3bd2233314542f6bbc82a"
SMOKE_WINDOWS = int(os.environ.get("NAVRBD_SMOKE_WINDOWS", "0"))      # >0: --max-windows N (a code check, never a result)
D_ARMS = ("N1D_s0", "N1D_s1")
TAGS = tuple(f"NAVRBD-{a}{z}" for a in D_ARMS for z in ("", NL.Z))
ARMS = {t: (f"{CK}/{t[7:].rstrip(NL.Z)}/ckpt_last.pt", ("bank", f"{CK}/{t[7:].rstrip(NL.Z)}/BANK_MANIFEST.json")) for t in TAGS}
for _t, _v in ARMS.items():
    RD.ARMS[_t] = _v
RD.OUT, RD.TREE = OUT, TREE
RD.INPUTS["--eval-grid"] = NL.SR + "/inputs/eval_grid_windows.json"
RD.INPUTS["--eval-join"] = NL.SR + "/inputs/join_eval141.npz"
if os.environ.get("NAVRBD_SMOKE_GRID"):
    RD.INPUTS["--eval-grid"] = os.environ["NAVRBD_SMOKE_GRID"]


def rb_npz_checked():
    got = NL.sha256_file(RB_NPZ)
    if got != RB_SHA256:
        raise RuntimeError(f"navtext_nav_rbd.npz sha256 {got[:12]} != pinned {RB_SHA256[:12]}")
    return RB_NPZ


def tree_problem():
    """the NAVRBD tree must hold the study wrapper and equal the STRAT1 pins for the two pinned evaluators and ci.py (NL.tree_problem), and v8f_t1_eval_rbd.py"""
    if not os.path.isfile(TREE + "/stack/scripts/v8f_t1_eval_rbd.py"):
        return f"tree file missing: {TREE}/stack/scripts/v8f_t1_eval_rbd.py (export the build commit to NAVRBD_TREE)"
    return NL.tree_problem(TREE)


def refusal_ck(arm, ck_root=None):
    """-> reason or None: the checkpoint exists, a BANK_MANIFEST.json (when present) names its sha256 and it matches"""
    root = ck_root or CK
    tag = arm[:-1] if arm.endswith(NL.Z) else arm
    if tag not in D_ARMS:
        return f"unknown arm {arm}"
    ck, mp = f"{root}/{tag}/ckpt_last.pt", f"{root}/{tag}/BANK_MANIFEST.json"
    if not os.path.isfile(ck):
        return f"REFUSED: checkpoint missing ({ck})"
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
        argv[1] = argv[1].replace("v8f_t1_eval.py", "v8f_t1_eval_rbd.py")
        argv = argv + extra_args(cls.cur)
        env = dict(kw.get("env") or os.environ)
        env["PYTHONPATH"] = ";".join(os.path.normpath(p) for p in (TREE + "/stack", TREE + "/taniteval"))
        kw["env"] = env
        os.makedirs(OUT, exist_ok=True)
        with open(f"{OUT}/argv_{cls.cur}_m10_causal_heads.txt", "a", encoding="utf-8") as f:
            f.write("NAVRBD extra: " + " ".join(argv[-12:]) + "\n")
        return subprocess.run(argv, **kw)


RD.subprocess = _Shim


def refusal(tag):
    if tag not in ARMS:
        return f"unknown tag {tag}"
    return refusal_ck(tag[7:])


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
    json.dump(rep, open(os.path.join(OUT, f"DUMPS_NAVRBD_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
