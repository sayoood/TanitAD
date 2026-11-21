"""navrb2_dumps.py -- T1-kin dumps of the four NAVRB2 R arms (CPU, 2 threads, zero training).  Usage: python navrb2_dumps.py TAG [TAG ...]   (TAG in N1R_s0 S1R_s0 N1R_s1 S1R_s1 and their zero-channel
twins N1R_s0Z ...: the same checkpoint dumped with --nav-rb-variant zero, the Z-EVAL / Z-USE control).

A thin wrapper over the UNEDITED otband run_dumps.py (the pinned STRAT1 dump driver): its argv is the DAG-OACC baseline's with --ckpt / --tag / --out-dir replaced; here the shim (a) swaps the script
v8f_t1_eval.py for the study wrapper v8f_t1_eval_rb.py of the e71782a3 tree (which pins the tool's sha256 and refuses on a difference), (b) appends --nav-text-npz / --nav-text-sha256 (the pinned
navtext_nav.npz: R arms are text-nav), --dump-nav, and --nav-rb on --nav-rb-npz <table> --nav-rb-sha256 <pin> --nav-rb-variant follow|zero, (c) pins the child's PYTHONPATH to the e71782a3 tree.
An arm is REFUSED (nothing launched) when its checkpoint / bank manifest is missing, names another tag or commit than e71782a3, or the checkpoint sha256 differs; the tree is checked first (tree_problem,
import_check).  Dumps of one read are made in ONE environment and ONE thread count (2), as the STRAT1 dumps.  Writes DUMPS_NAVRB2_<stamp>.json under out/."""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrb2_lib as NL  # noqa: E402

sys.path.insert(0, NL.SR)
sys.path.insert(0, NL.SR + "/theory/otband/read")
import strat1_lib as L  # noqa: E402
import run_dumps as RD  # noqa: E402  (module level is definitions only)

OUT = NL.T1
SMOKE_WINDOWS = int(os.environ.get("NAVRB2_SMOKE_WINDOWS", "0"))      # >0: --max-windows N (a code check, never a result)
TAGS = tuple(f"NAVRB2-{a}{z}" for a in NL.R_ARMS for z in ("", NL.Z))
ARMS = {t: (f"{NL.CK}/{t[7:].rstrip(NL.Z)}/ckpt_last.pt", ("bank", f"{NL.CK}/{t[7:].rstrip(NL.Z)}/BANK_MANIFEST.json")) for t in TAGS}
for _t, _v in ARMS.items():
    RD.ARMS[_t] = _v
RD.OUT, RD.TREE = OUT, NL.TREE
RD.INPUTS["--eval-grid"] = NL.SR + "/inputs/eval_grid_windows.json"
RD.INPUTS["--eval-join"] = NL.SR + "/inputs/join_eval141.npz"
if os.environ.get("NAVRB2_SMOKE_GRID"):
    RD.INPUTS["--eval-grid"] = os.environ["NAVRB2_SMOKE_GRID"]


def variant_of(tag):
    return "zero" if tag.endswith(NL.Z) else "follow"


def extra_args(tag):
    """-> the arguments appended to the driver's argv (pure given files): text nav + rb flags; the Z twin names --nav-rb-variant zero"""
    npz = L.pinned("navtext_nav.npz", NL.SR + "/inputs")
    a = ["--nav-text-npz", npz, "--nav-text-sha256", L.PINS["navtext_nav.npz"], "--dump-nav",
         "--nav-rb", "on", "--nav-rb-npz", NL.rb_npz_checked(), "--nav-rb-sha256", NL.RB_SHA256, "--nav-rb-variant", variant_of(tag)]
    return a + (["--max-windows", str(SMOKE_WINDOWS)] if SMOKE_WINDOWS else [])


class _Shim:
    """stands in for the `subprocess` module inside run_dumps"""
    cur = None
    PIPE, STDOUT = subprocess.PIPE, subprocess.STDOUT

    @classmethod
    def run(cls, argv, **kw):
        argv = list(argv)
        assert argv[1].replace("\\", "/").endswith("/stack/scripts/v8f_t1_eval.py"), argv[1]
        argv[1] = argv[1].replace("v8f_t1_eval.py", "v8f_t1_eval_rb.py")
        argv = argv + extra_args(cls.cur)
        env = dict(kw.get("env") or os.environ)
        env["PYTHONPATH"] = ";".join(os.path.normpath(p) for p in (NL.TREE + "/stack", NL.TREE + "/taniteval"))
        kw["env"] = env
        os.makedirs(OUT, exist_ok=True)
        with open(f"{OUT}/argv_{cls.cur}_m10_causal_heads.txt", "a", encoding="utf-8") as f:
            f.write("NAVRB2 extra: " + " ".join(argv[-12:]) + "\n")
        return subprocess.run(argv, **kw)


RD.subprocess = _Shim


def refusal(tag):
    if tag not in ARMS:
        return f"unknown tag {tag}"
    return NL.refusal(tag[7:])


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
    tp = NL.tree_problem() or NL.import_check(RD.PY)
    if tp:
        raise SystemExit("REFUSED: " + tp)
    for k, v in RD.INPUTS.items():
        assert os.path.exists(v), (k, v)
    rep = {"tree": NL.TREE, "out": OUT, "commit": NL.COMMIT, "threads": 2, "smoke_windows": SMOKE_WINDOWS,
           "inputs_sha256": {k: (RD.sha(v) if os.path.isfile(v) else "directory") for k, v in RD.INPUTS.items()}, "runs": []}
    rc = 0
    for t in tags:
        r = run(t)
        print(time.strftime("%H:%M:%S"), json.dumps(r), flush=True)
        rep["runs"].append(r)
        if r.get("status") not in ("ok", "skip (done file exists)"):
            rc = 1
    os.makedirs(NL.OUT, exist_ok=True)
    json.dump(rep, open(os.path.join(NL.OUT, f"DUMPS_NAVRB2_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
