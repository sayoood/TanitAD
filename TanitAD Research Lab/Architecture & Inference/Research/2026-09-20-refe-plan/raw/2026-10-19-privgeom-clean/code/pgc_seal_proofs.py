"""PRIVGEOM-CLEAN seal PROOFS (LABEL-FREE; run AFTER the seal, before the read).  Writes run_clean/seal_proofs/seal_proofs.json.

    python -B -W ignore pgc_seal_proofs.py            # the proofs; exit 0 iff every arm behaved as committed

A. THE REAL ENTRY POINT REFUSES.  A mirror of the package (code + ref + the sealed prereg / .sha256 / SEALED_AT.txt / ruling) is attacked and the REAL `pgc_run.py` is run as a subprocess on each copy:
     genuine seal-check ................ ACCEPTED   (the control: the same entry point, the same stage)
     one-byte-edited prereg ............ REFUSED     (every stage: seal-check, preflight, select, join, gate, fit, analyze)
     the pre-revision DRAFT prereg ..... REFUSED     (with a freshly written .sha256 and SEALED_AT that agree with it: only the STATUS guard can refuse)
     a draft-STATUS copy of the sealed . REFUSED     (same isolation)
     a one-byte-edited pinned code file  REFUSED
     PGC_REHEARSAL=1 .................... seal-check and preflight REFUSE (exit 3): the rehearsal bypass is not a seal
   and no output file is created by a refused stage.
B. THE SEAL CHECK OPENS NO LABEL: in-process, on the REAL root, with `numpy.load` and `builtins.open` spied: zero `np.load` calls; the only opens are the sealed small text files and byte-hashing of the pinned / input files (no member is parsed).
C. A PHASE-A GATE FAILURE OPENS NO LABEL: the real `stage_gate` (REHEARSAL mode, on the banked SEEN join, a copy of the output dir) with an injected branch-table failure returns 5, writes `labels_opened: false`, and never calls `np.load` on a label file;
   the control proves the spy WOULD see a label load.
"""
from __future__ import annotations

import builtins
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import pgc_seal as S                                                                     # noqa: E402

PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
OUT = f"{ROOT}/run_clean/seal_proofs"
STAGES = ("seal-check", "preflight", "select", "join", "gate", "fit", "analyze")


def env(**kw):
    e = {k: v for k, v in os.environ.items() if not k.startswith("PGC_")}
    e.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    e.update(kw)
    return e


def run_stage(mirror: str, stage: str, **envkw) -> dict:
    r = subprocess.run([PY, "-B", "-W", "ignore", f"{mirror}/code/pgc_run.py", stage], capture_output=True, text=True, env=env(**envkw), cwd=f"{mirror}/code", stdin=subprocess.DEVNULL)
    return {"stage": stage, "rc": r.returncode, "tail": (r.stdout + r.stderr).strip()[-400:]}


def make_mirror(td: str, name: str) -> str:
    m = f"{td}/{name}"
    shutil.copytree(f"{ROOT}/code", f"{m}/code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(f"{ROOT}/ref", f"{m}/ref", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for n in (S.PREREG_NAME, "PREREG_PRIVGEOM_CLEAN.sha256", S.SEALED_AT_NAME, S.RULING_NAME):
        shutil.copyfile(f"{ROOT}/{n}", f"{m}/{n}")
    return m


def rehash(m: str, text: str) -> None:
    """write `text` as the prereg and make the .sha256 AND SEALED_AT agree with it, so that ONLY a content guard (status) can refuse it."""
    with open(f"{m}/{S.PREREG_NAME}", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    h = S.sha_file(f"{m}/{S.PREREG_NAME}")
    with open(f"{m}/PREREG_PRIVGEOM_CLEAN.sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{h}  {S.PREREG_NAME}\n")
    sa = json.load(open(f"{m}/{S.SEALED_AT_NAME}", encoding="utf-8"))
    sa["prereg_sha256"] = h
    with open(f"{m}/{S.SEALED_AT_NAME}", "w", encoding="utf-8", newline="\n") as f:
        json.dump(sa, f, indent=1)


def arm(rep: list, name: str, res: dict, want_refused: bool, needle: str = "") -> bool:
    ok = (res["rc"] != 0) if want_refused else (res["rc"] == 0)
    if want_refused and needle:
        ok = ok and needle in res["tail"]
    rep.append({"arm": name, "expected": "REFUSED" if want_refused else "ACCEPTED", "rc": res["rc"], "stage": res["stage"], "needle": needle, "ok": bool(ok), "tail": res["tail"][-260:]})
    return ok


def proofs_A(td: str) -> list:
    rep = []
    text = S.read_text(f"{ROOT}/{S.PREREG_NAME}")
    # ---- control: genuine
    m = make_mirror(td, "genuine")
    arm(rep, "genuine seal: seal-check", run_stage(m, "seal-check"), False)
    # ---- one-byte-edited prereg, through EVERY stage
    m = make_mirror(td, "edited")
    b = bytearray(open(f"{m}/{S.PREREG_NAME}", "rb").read())
    i = b.index(b"PRIVGEOM")
    b[i] ^= 0x20                                                                          # ONE byte (case flip)
    open(f"{m}/{S.PREREG_NAME}", "wb").write(bytes(b))
    for st in STAGES:
        arm(rep, f"one-byte-edited prereg: {st}", run_stage(m, st), True, "does not hash to its recorded sha256")
    rep.append({"arm": "refused stages create no output", "ok": not os.path.exists(f"{m}/run_clean"), "outputs": sorted(os.listdir(m))})
    # ---- the real pre-revision DRAFT, made self-consistent (hash + SEALED_AT agree): only the STATUS guard remains
    m = make_mirror(td, "draft_real")
    rehash(m, S.read_text(f"{ROOT}/draft/PREREG_PRIVGEOM_CLEAN.DRAFT_pre_revision1.md"))
    arm(rep, "real pre-revision DRAFT prereg (self-consistent hash): seal-check", run_stage(m, "seal-check"), True, "status is 'DRAFT'")
    arm(rep, "real pre-revision DRAFT prereg (self-consistent hash): select", run_stage(m, "select"), True, "status is 'DRAFT'")
    # ---- a draft-STATUS copy of the SEALED text
    m = make_mirror(td, "draft_flipped")
    flipped = text.replace("**STATUS: SEALED", "**STATUS: DRAFT", 1)
    assert flipped != text
    rehash(m, flipped)
    for st in ("seal-check", "gate"):
        arm(rep, f"draft-STATUS copy of the sealed text: {st}", run_stage(m, st), True, "status is 'DRAFT'")
    # ---- the real draft against the REAL sealed .sha256 (the plain stale-hash case)
    m = make_mirror(td, "draft_stale")
    shutil.copyfile(f"{ROOT}/draft/PREREG_PRIVGEOM_CLEAN.DRAFT_pre_revision1.md", f"{m}/{S.PREREG_NAME}")
    arm(rep, "real DRAFT prereg against the sealed .sha256: seal-check", run_stage(m, "seal-check"), True, "does not hash")
    # ---- a one-byte-edited pinned code file
    m = make_mirror(td, "code_edit")
    with open(f"{m}/code/pgc_lib.py", "ab") as f:
        f.write(b"#")
    arm(rep, "one-byte-edited pinned code file (pgc_lib.py): seal-check", run_stage(m, "seal-check"), True, "pinned file(s) differ")
    arm(rep, "one-byte-edited pinned code file (pgc_lib.py): fit", run_stage(m, "fit"), True, "pinned file(s) differ")
    # ---- the rehearsal bypass is not a seal
    m = make_mirror(td, "rehearsal_env")
    for st in ("seal-check", "preflight"):
        arm(rep, f"PGC_REHEARSAL=1: {st}", run_stage(m, st, PGC_REHEARSAL="1"), True, "")
    return rep


def proofs_B() -> dict:
    """in-process, on the REAL root: the seal check calls np.load zero times."""
    import numpy as np
    import pgc_run
    loads, opens = [], []
    real_load, real_open = np.load, builtins.open
    np.load = lambda *a, **k: (loads.append(str(a[0]) if a else ""), real_load(*a, **k))[1]
    builtins.open = lambda f, mode="r", *a, **k: (opens.append((os.path.abspath(str(f)), mode)), real_open(f, mode, *a, **k))[1]
    try:
        info = pgc_run.require_prereg(inputs=True)
    finally:
        np.load, builtins.open = real_load, real_open
    names = sorted({os.path.basename(p) for p, m in opens})
    parsed = [p for p, m in opens if "b" not in m and p.endswith((".npz", ".npy"))]
    return {"ok": bool(len(loads) == 0 and not parsed and "prereg" in info), "np_load_calls": len(loads), "opened_as_text_and_is_npz_or_npy": parsed, "n_opens": len(opens), "opened_basenames": names, "info": info,
            "note": "meta.npz / feat_f10.npy are opened in 'rb' mode ONLY to sha256 their bytes (the seal pins them); no member is parsed and no array is loaded."}


def proofs_C() -> dict:
    """a child process (REHEARSAL on the SEEN join, a scratch output dir) so that the env is set before pgc_run is imported."""
    td = tempfile.mkdtemp(prefix="pgc_phaseA_", dir=OUT)
    try:
        r = subprocess.run([PY, "-B", "-W", "ignore", f"{HERE}/pgc_seal_proofs.py", "_phaseA_child"], capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=HERE,
                           env=env(PGC_REHEARSAL="1", PGC_RUN=td, PGC_JOIN_DIR="C:/Users/Admin/par18/privgeom/run/join"))
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("PHASEA_RESULT ")]
        res = json.loads(line[-1][len("PHASEA_RESULT "):]) if line else {"ok": False, "error": (r.stdout + r.stderr)[-600:]}
        res["child_rc"] = r.returncode
        return res
    finally:
        shutil.rmtree(td, ignore_errors=True)


def phaseA_child() -> int:
    import subprocess as sp
    import numpy as np
    import pgc_lib
    import pgc_run
    loaded = []
    real_load = np.load
    np.load = lambda *a, **k: (loaded.append(str(a[0]) if a else ""), real_load(*a, **k))[1]
    pgc_run.subprocess.run = lambda *a, **k: sp.CompletedProcess(a, 0, "", "")           # the suites are not what is under test here
    real_bt = pgc_lib.branch_table_failures
    pgc_lib.branch_table_failures = lambda mod=None: [("injected failure", "x", "y")]
    pgc_run.C.branch_table_failures = pgc_lib.branch_table_failures
    rc = pgc_run.stage_gate(None)
    gp = json.load(open(f"{pgc_run.RUN}/gate_report.json", encoding="utf-8"))
    label_loads = [p for p in loaded if p.replace("\\", "/").endswith(("labels_sub.npy", "meta.npz"))]
    label_file_written = os.path.exists(f"{pgc_run.RUN}/join_labels_sub.npy")
    # control: the spy WOULD see a label load
    np.load(f"{pgc_run.paths()['join']}/labels_sub.npy", allow_pickle=False)
    control_seen = any(p.replace("\\", "/").endswith("labels_sub.npy") for p in loaded)
    pgc_lib.branch_table_failures = real_bt
    ok = bool(rc == 5 and gp.get("labels_opened") is False and gp.get("stopped_before_labels") is True and not label_loads and not label_file_written and control_seen and len(real_bt()) == 0)
    print("PHASEA_RESULT " + json.dumps({"ok": ok, "gate_rc": rc, "labels_opened": gp.get("labels_opened"), "stopped_before_labels": gp.get("stopped_before_labels"), "label_loads_before_stop": label_loads,
                                          "label_file_written": label_file_written, "spy_control_sees_a_label_load": control_seen, "phase_A_flags": {k: gp[k].get("pass") if isinstance(gp.get(k), dict) else gp.get(k) for k in ("C-JOIN", "C-ID-EXTRACT", "C-GRAD-REAL")},
                                          "branch_table_failures": gp.get("C-RED_branch_table_failures")}), flush=True)
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_phaseA_child":
        return phaseA_child()
    os.makedirs(OUT, exist_ok=True)
    out = {"schema": "privgeom-clean-seal-proofs-v1"}
    td = tempfile.mkdtemp(prefix="pgc_mirror_", dir=OUT)
    try:
        out["seal_holds_on_the_real_root"] = S.check_seal(ROOT, inputs=True)
        out["A_entry_point_refuses"] = proofs_A(td)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    out["B_seal_check_opens_no_label"] = proofs_B()
    out["C_phase_A_failure_opens_no_label"] = proofs_C()
    out["A_all_ok"] = all(r["ok"] for r in out["A_entry_point_refuses"])
    out["all_ok"] = bool(out["A_all_ok"] and out["B_seal_check_opens_no_label"]["ok"] and out["C_phase_A_failure_opens_no_label"].get("ok"))
    with open(f"{OUT}/seal_proofs.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({"all_ok": out["all_ok"], "A": [(r["arm"], r["ok"]) for r in out["A_entry_point_refuses"]], "B": out["B_seal_check_opens_no_label"]["ok"], "C": out["C_phase_A_failure_opens_no_label"].get("ok")}, indent=1))
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
