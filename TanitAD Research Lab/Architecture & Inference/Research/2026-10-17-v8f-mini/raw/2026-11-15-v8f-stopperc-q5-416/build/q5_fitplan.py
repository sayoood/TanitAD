"""q5_fitplan.py -- the fit plan the Master Mind runs UNCHANGED after the pod build (AMENDMENT 7 section 2.5 steps 3-5).  CPU only; s0p_fit.py is called, never edited.

Layout under --base (everything the plan names is derived from it):
  pod_out/{S1,S2}_{416,256p}/   `q5_frames_pod.py gather` output pulled from the pod (v40 v160 mean y clip ws eid .npy)       pod_out/eval_{416,256p}/  same, eval-141
  c0/{S1,S2}/                   C0's S0-P-format train dir for that sample (E y clip ws eid, v40/v160/mean of the 256 route, meta.json, sample.json)
  c0/s0p_eval.npz               C0's eval npz (s0p_extract_eval.py)
  assembled/{S}_{416,256p}/     OUTPUT of `assemble`: E y clip ws eid copied bitwise from C0, v40 v160 mean from the pod      assembled/eval_{416,256p}.npz
  work/{S}_{416,256p}/          s0p_fit work dirs (proj_*.npy, fit_<arm>_<model>.{json,npy})

  python q5_fitplan.py assemble --base B          (asserts the pod's y / clip / ws / eid equal C0's, then q5_now_encode.write_s0p_dir / write_s0p_npz)
  python q5_fitplan.py plan     --base B [--python PY] [--samples S1,S2] [--out B/fitplan.sh]   (writes the command list and B/read_cfg.json)
  python q5_fitplan.py run      --base B [--only-prep]                                          (executes the plan's lines in order, in-process)
Fit calls per (sample, resolution): prep once; 416: E, E-limit, M+E, V40+E, V160+E x {linear, mlp} plus q5_fit_extra (V40+E-limit) x {linear, mlp}; 256p: M+E, V40+E, V160+E x {linear, mlp}.
E / E-limit are fitted ONCE per sample (in the 416 work dir): their inputs are bit-identical in both dirs (C3e), so a 256p copy would be the same fit.
"""
import argparse
import json
import os
import shlex
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
S0P = os.environ.get("Q5_S0P", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/s0p")
PY_DEFAULT = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TAGS = ("416", "256p")
ARMS_416 = ("E", "E-limit", "M+E", "V40+E", "V160+E")
ARMS_256P = ("M+E", "V40+E", "V160+E")
MODELS = ("linear", "mlp")
EXTERNAL_CONTROLS = ("c0_differing_bytes", "c2_auroc", "c2_outcome", "c4_E_bitwise_equal_C0", "c6a_max_abs_diff", "c6b_mae_255", "c6c_bitwise", "c7_ok", "c8_bitwise", "c9_counts")


def paths(base, samples):
    j = lambda *p: os.path.join(base, *p).replace("\\", "/")                    # noqa: E731
    P = {"base": base, "samples": {}}
    for s in samples:
        P["samples"][s] = {"c0": j("c0", s), **{f"pod{t}": j("pod_out", f"{s}_{t}") for t in TAGS}, **{f"train{t}": j("assembled", f"{s}_{t}") for t in TAGS},
                           **{f"work{t}": j("work", f"{s}_{t}") for t in TAGS}}
    P["eval"] = {"c0": j("c0", "s0p_eval.npz"), **{f"pod{t}": j("pod_out", f"eval_{t}") for t in TAGS}, **{f"eval{t}": j("assembled", f"eval_{t}.npz") for t in TAGS}}
    return P


def commands(base, samples, py=PY_DEFAULT):
    """The ordered command list: [(label, argv list)].  Every command is one `s0p_fit.py` or `q5_fit_extra.py` call."""
    P = paths(base, samples)
    fit, extra = f"{S0P}/s0p_fit.py", f"{HERE.replace(chr(92), '/')}/q5_fit_extra.py"
    out = []
    for s in samples:
        d = P["samples"][s]
        for t in TAGS:
            ev = P["eval"][f"eval{t}"]
            out.append((f"{s} {t} prep", [py, fit, "prep", "--train", d[f"train{t}"], "--eval", ev, "--work", d[f"work{t}"]]))
            for arm in (ARMS_416 if t == "416" else ARMS_256P):
                for m in MODELS:
                    out.append((f"{s} {t} fit {arm} {m}", [py, fit, "fit", "--train", d[f"train{t}"], "--eval", ev, "--work", d[f"work{t}"], "--arm", arm, "--model", m]))
            if t == "416":
                for m in MODELS:
                    out.append((f"{s} 416 fit V40+E-limit {m}", [py, extra, "--train", d["train416"], "--eval", ev, "--work", d["work416"], "--model", m]))
    return out


def read_cfg(base, samples, tree="F:/TanitAD-scratch/stopbal_read/tree"):
    P = paths(base, samples)
    return {"tree": tree, "eval": P["eval"]["eval416"], "eval416": P["eval"]["eval416"], "eval256p": P["eval"]["eval256p"],
            "sidecar_limit_eval": os.path.join(base, "c0", "sidecar_limit_eval.npy").replace("\\", "/"),
            "out": os.path.join(base, "RESULT_Q5.json").replace("\\", "/"),
            "samples": {s: {"work416": P["samples"][s]["work416"], "work256p": P["samples"][s]["work256p"], "train416": P["samples"][s]["train416"], "train256p": P["samples"][s]["train256p"],
                            "work256b": os.path.join(base, "banked256b", s).replace("\\", "/")} for s in samples},
            "controls": {k: None for k in EXTERNAL_CONTROLS},
            "_note": "controls left null REFUSE the read (fail closed): fill them from the controls ledger (C0, C2 planted fits, C4, C6 from the pod ledger / hashes, C7, C8 from test_q5 and smoke_c8.json, C9 from the sample + eval_windows.json). "
                     "C1, C3a/b/c/d/e and C5 are computed by q5_read from the dirs. work256b / work416pb / work416tok are the beside arms' inputs; an absent dir is reported BLOCKED, never invented."}


def assemble(base, samples, log=print):
    sys.path.insert(0, HERE)
    import q5_now_encode as Q
    P = paths(base, samples)
    os.makedirs(os.path.join(base, "assembled"), exist_ok=True)
    for s in samples:
        d = P["samples"][s]
        for t in TAGS:
            pod = d[f"pod{t}"]
            for k in ("y", "clip", "ws", "eid"):
                a, b = np.load(os.path.join(pod, k + ".npy")), np.load(os.path.join(d["c0"], k + ".npy"))
                assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), f"{s} {t}: pod {k} differs from C0's"
            Q.write_s0p_dir(d["c0"], d[f"train{t}"], *(np.load(os.path.join(pod, k + ".npy"), mmap_mode="r") for k in ("v40", "v160", "mean")), extra_meta={"sample": s, "res": t})
            log(f"assembled {d[f'train{t}']}")
    for t in TAGS:
        pod = P["eval"][f"pod{t}"]
        z = np.load(P["eval"]["c0"], allow_pickle=False)
        for k in ("y", "clip", "ws", "eid"):
            a = np.load(os.path.join(pod, k + ".npy"))
            assert a.dtype == z[k].dtype and a.shape == z[k].shape and a.tobytes() == z[k].tobytes(), f"eval {t}: pod {k} differs from C0's"
        Q.write_s0p_npz(P["eval"]["c0"], P["eval"][f"eval{t}"], *(np.load(os.path.join(pod, k + ".npy")) for k in ("v40", "v160", "mean")), extra_meta={"res": t})
        log(f"assembled {P['eval'][f'eval{t}']}")


def run_plan(cmds, only_prep=False, log=print):
    """Execute the plan's lines in order, IN-PROCESS (the same argv the shell would get; a failure raises)."""
    import q5_controls as C
    F, _ = C.s0p_modules()
    import q5_fit_extra as X
    for label, argv in cmds:
        if only_prep and "prep" not in label:
            continue
        log(f"[{label}]")
        script, args = os.path.basename(argv[1]), argv[2:]
        if script == "s0p_fit.py":
            F.main(args)
        elif script == "q5_fit_extra.py":
            X.main(args)
        else:
            raise ValueError(script)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("assemble", "plan", "run"))
    ap.add_argument("--base", required=True)
    ap.add_argument("--samples", default="S1,S2")
    ap.add_argument("--python", default=PY_DEFAULT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only-prep", action="store_true")
    a = ap.parse_args(argv)
    samples = [s for s in a.samples.split(",") if s]
    if a.cmd == "assemble":
        assemble(a.base, samples)
    elif a.cmd == "plan":
        cmds = commands(a.base, samples, a.python)
        out = a.out or os.path.join(a.base, "fitplan.sh")
        lines = ["#!/usr/bin/env bash", "# generated by q5_fitplan.py plan; OMP_NUM_THREADS 2; independent lines after each prep may run in parallel, at most 2 threads in total", "export OMP_NUM_THREADS=2", "set -e"]
        lines += [f"# {lab}\n" + " ".join(shlex.quote(x) for x in c) for lab, c in cmds]
        lines.append(f"{shlex.quote(a.python)} {HERE.replace(chr(92), '/')}/q5_read.py {os.path.join(a.base, 'read_cfg.json').replace(chr(92), '/')}")
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(a.base, "read_cfg.json"), "w", encoding="utf-8") as f:
            json.dump(read_cfg(a.base, samples), f, indent=1)
        print(f"{len(cmds)} fit-side calls -> {out}; read cfg -> {os.path.join(a.base, 'read_cfg.json')}")
    else:
        run_plan(commands(a.base, samples, a.python), a.only_prep)


if __name__ == "__main__":
    main()
