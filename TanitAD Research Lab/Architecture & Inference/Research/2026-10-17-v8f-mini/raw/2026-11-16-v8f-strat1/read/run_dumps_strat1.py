"""run_dumps_strat1.py -- T1-kin dumps for the STRAT1 read (CPU, zero training).  Usage: python run_dumps_strat1.py TAG [TAG ...]   (TAG in ARMS, e.g. STRAT1-S1_s0)

A thin wrapper over theory/otband/read/run_dumps.py (UNEDITED; sha256 in READ_PINS): its tag table is extended with the seven STRAT1 tags, its module paths are redirected
(dump folder, tree, input files) and its own run() does the work (the DAG-OACC baseline's argv with --ckpt / --tag / --out-dir / --dump-ticks replaced; CPU, 2 threads;
checkpoint sha256 checked before AND after; waits for free memory; writes the done file).  Added here, READ_PLAN 1 and 2.1:
  * an arm is REFUSED (nothing launched) when its ckpt_last.pt or BANK_MANIFEST.json is missing, the manifest names another tag or another commit than e81bde01, or the
    checkpoint sha256 differs from the manifest;
  * the nav arguments come from the checkpoint's own cfg.nav_source: text -> --nav-text-npz / --nav-text-sha256 (the pinned navtext_nav.npz), v9 -> NONE.  A v9 (C0) arm
    given an npz is refused, a text arm without one is refused (nav_args());
  * every arm is dumped with --dump-nav (the nav fed at each tick; READ_PLAN 3 item 2) on top of the driver's --dump-heads;
  * the child's PYTHONPATH is the read tree's stack and taniteval, ';'-joined Windows paths, and import_check() asserts, in a child with that very environment, that tanitad.__file__ lies under the read tree (memory: a ':'-joined path or -I imports the G: editable install).
All seven dumps of one read are made in ONE environment and ONE thread count (2).  Writes DUMPS_STRAT1_<stamp>.json beside this file.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/theory/otband/read")
import strat1_lib as L  # noqa: E402
import run_dumps as RD  # noqa: E402  (module level is definitions only; its main() is not called)

OUT = os.environ.get("STRAT1_T1", "F:/TanitAD-scratch/strat1/read/t1")
SMOKE_WINDOWS = int(os.environ.get("STRAT1_SMOKE_WINDOWS", "0"))      # >0: --max-windows N (a code check, never a result)
TAGS = tuple(f"STRAT1-{a}" for a in L.ARMS)
ARMS = {t: (f"{L.CK}/{t[7:]}/ckpt_last.pt", ("bank", f"{L.CK}/{t[7:]}/BANK_MANIFEST.json")) for t in TAGS}
for _t, _v in ARMS.items():
    RD.ARMS[_t] = _v
RD.OUT, RD.TREE = OUT, L.TREE
RD.INPUTS["--eval-grid"] = L.INP + "/eval_grid_windows.json"        # byte-identical to the OTBAND copies (sha256 pinned in strat1_lib.PINS)
RD.INPUTS["--eval-join"] = L.INP + "/join_eval141.npz"
if os.environ.get("STRAT1_SMOKE_GRID"):          # a reduced grid json for the code check (never a result)
    RD.INPUTS["--eval-grid"] = os.environ["STRAT1_SMOKE_GRID"]


def nav_args(arm, nav_source, npz, sha):
    """-> extra argv for the evaluators.  text <-> npz, v9 <-> no npz; anything else raises ValueError (the evaluators refuse the same, BUILD_SPEC 7)."""
    if nav_source == "text":
        if not (npz and sha):
            raise ValueError(f"{arm}: a text-nav checkpoint needs --nav-text-npz and --nav-text-sha256")
        return ["--nav-text-npz", npz, "--nav-text-sha256", sha]
    if nav_source == "v9":
        if npz or sha:
            raise ValueError(f"{arm}: a v9-nav checkpoint must not be given a text-nav file")
        return []
    raise ValueError(f"{arm}: unknown nav_source {nav_source!r}")


def nav_source_of(ckpt):
    import torch
    return torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]["nav_source"]


def extra_args(tag, nav_source=None):
    """the arguments appended to the OTBAND driver's argv for this tag"""
    arm = tag[7:]
    src = nav_source or nav_source_of(ARMS[tag][0])
    want_text = arm in L.TEXT_ARMS
    if (src == "text") != want_text:
        raise ValueError(f"{tag}: the checkpoint's nav_source is {src!r}, the card's arm is {'text' if want_text else 'v9'}")
    npz = L.pinned("navtext_nav.npz") if want_text else ""
    return nav_args(arm, src, npz, L.PINS["navtext_nav.npz"] if want_text else "") + ["--dump-nav"] + (["--max-windows", str(SMOKE_WINDOWS)] if SMOKE_WINDOWS else [])


class _Shim:
    """stands in for the `subprocess` module inside run_dumps: appends the STRAT1 arguments to the driver's argv and pins the child's import path"""
    cur = None
    PIPE, STDOUT = subprocess.PIPE, subprocess.STDOUT

    @classmethod
    def run(cls, argv, **kw):
        argv = list(argv) + extra_args(cls.cur)
        env = dict(kw.get("env") or os.environ)
        env["PYTHONPATH"] = ";".join(os.path.normpath(p) for p in (L.TREE + "/stack", L.TREE + "/taniteval"))
        kw["env"] = env
        with open(f"{OUT}/argv_{cls.cur}_m10_causal_heads.txt", "a", encoding="utf-8") as f:
            f.write("STRAT1 extra: " + " ".join(argv[-8:]) + "\n")
        return subprocess.run(argv, **kw)


RD.subprocess = _Shim


def refusal(tag):
    """-> reason or None.  Files only; launches nothing."""
    if tag not in ARMS:
        return f"unknown tag {tag}"
    ck, (_, man_p) = ARMS[tag]
    if not os.path.isfile(ck):
        return f"REFUSED: checkpoint missing ({ck})"
    if not os.path.isfile(man_p):
        return f"REFUSED: BANK_MANIFEST.json missing ({man_p})"
    man = json.load(open(man_p, encoding="utf-8"))
    want = ((man.get("files") or {}).get("ckpt_last.pt") or {}).get("sha256")
    if not want:
        return "REFUSED: BANK_MANIFEST.json holds no ckpt_last.pt sha256"
    if man.get("tag") != tag[7:]:
        return f"REFUSED: the bank manifest names tag {man.get('tag')}, not {tag[7:]}"
    if man.get("commit") != L.COMMIT:
        return f"REFUSED: the bank was trained at commit {man.get('commit')}, the code of record is {L.COMMIT}"
    if RD.sha(ck) != want:
        return "REFUSED: checkpoint sha256 differs from its bank manifest"
    return None


def tree_problem():
    """the read tree must be the e81bde01 export: the files the dump touches are re-hashed against READ_PINS.json (written by make_pins.py from `git show e81bde01:<path>`)"""
    p = HERE + "/READ_PINS.json"
    if not os.path.isfile(p):
        return "READ_PINS.json missing"
    for rel, want in json.load(open(p, encoding="utf-8"))["tree_sha256"].items():
        q = f"{L.TREE}/{rel}"
        if not os.path.isfile(q):
            return f"tree file missing: {q}"
        if RD.sha(q) != want:
            return f"tree file differs from READ_PINS.json: {rel}"
    return None


def import_check():
    """-> problem string or None: tanitad and taniteval must import from the read tree in a child with the dump's environment"""
    env = dict(os.environ, PYTHONPATH=";".join(os.path.normpath(p) for p in (L.TREE + "/stack", L.TREE + "/taniteval")), CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([RD.PY, "-c", "import tanitad, taniteval; print(tanitad.__file__); print(taniteval.__file__)"], env=env, capture_output=True, encoding="utf-8")
    root = os.path.normpath(L.TREE).lower()
    paths = [x.strip() for x in r.stdout.splitlines()]
    if r.returncode or len(paths) != 2 or not all(os.path.normpath(x).lower().startswith(root) for x in paths):
        return f"tanitad / taniteval do not import from the read tree: {paths} {r.stderr[-200:]}"
    return None


def run(tag):
    why = refusal(tag)
    if why:
        return {"tag": tag, "status": why}
    try:
        extra_args(tag)
    except ValueError as e:
        return {"tag": tag, "status": f"REFUSED: {e}"}
    _Shim.cur = tag
    os.makedirs(OUT, exist_ok=True)
    return RD.run(tag)


def main():
    tags = sys.argv[1:]
    assert tags and all(t in ARMS for t in tags), f"tags must be among {list(ARMS)}"
    tp = tree_problem() or import_check()
    if tp:
        raise SystemExit("REFUSED: " + tp)
    for k, v in RD.INPUTS.items():
        assert os.path.exists(v), (k, v)
    rep = {"tree": L.TREE, "out": OUT, "commit": L.COMMIT, "threads": 2, "smoke_windows": SMOKE_WINDOWS,
           "inputs_sha256": {k: (RD.sha(v) if os.path.isfile(v) else "directory") for k, v in RD.INPUTS.items()}, "runs": []}
    rc = 0
    for t in tags:
        r = run(t)
        print(time.strftime("%H:%M:%S"), json.dumps(r), flush=True)
        rep["runs"].append(r)
        if r.get("status") not in ("ok", "skip (done file exists)"):
            rc = 1
    json.dump(rep, open(os.path.join(HERE, "out", f"DUMPS_STRAT1_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
