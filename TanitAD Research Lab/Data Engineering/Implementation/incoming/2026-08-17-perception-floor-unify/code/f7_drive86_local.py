"""STEP 7 (DEV BOX GPU) — drive the 86-clip re-run on the dev box's own GPU.

⛔ THIS FILE CONTAINS NO EXTRACTION. It is `f2_drive86.py`'s local twin: the same
population computation, the same content-based resume, the same completion
predicate — with the Colab transport (ship / upload todo / exec / pull tar)
replaced by a subprocess. **The detection is `code/f1_run86.py`, unchanged and
unread by this file**, which became host-agnostic on 2026-08-17 (every
`/content/...` literal is now an `F86_*` env var defaulting to the Colab value).

⚠️ A SECOND RUNNER WOULD HAVE BEEN THE DEFECT, ONE LEVEL UP. Two extractions
produce records that are structurally identical and differ only in what is
ABSENT — invisible, exactly like the mixed floor this package exists to remove.
Likewise the resume: `s2_lab_lib.content_census_local` is imported, never
re-implemented, because two answers to *"is this clip done?"* is how a corpus
goes mixed while both checks report green.

⚠️ WHY LOCAL AT ALL: free-Colab T4 assignment answered 503 Service Unavailable on
43+ attempts across ~90 min and this account is entitled to no other accelerator
(`PERCEPTION_FLOOR_UNIFY.md` §5.1). ⛔ Thor is off limits (live 30k S-W).

⛔ THE PROOF-GATE COMES FIRST. `--proof` runs ONE clip and refuses the batch
unless that record carries a NON-ZERO road/sky liveness count and zero errors.
The C77 dtype fix was developed on a T4 (sm_75); this GPU is Ada (sm_89) and "no
traceback" is not evidence it applied — 115 structurally perfect EMPTY records is
what C77 actually banked. Proof is a detection, never an absent error.

⛔ IT DOES NOT PUSH TO HUGGINGFACE. HF is READ (v2 labels, w120 shards,
records.parquet); the bank is local, with per-clip md5s in `_md5.json`.

usage:
  python f7_drive86_local.py --aug120 <dir> --v2-dir <dir> --bank <dir>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.join(HERE, "f1_run86.py")
sys.path.insert(0, os.path.join(REPO, "colab"))
import s2_lab_lib as L                                            # noqa: E402

CONF, SCHEMA = 0.25, 2


def census(bank, want):
    """⛔ THE SHARED PREDICATE, NEVER A LOCAL LOOKALIKE (f2's rule verbatim):
    liveness control present AND zero errors AND schema >= 2 AND floor == 0.25."""
    return L.content_census_local(bank, want=set(want),
                                  require_schema=SCHEMA, require_conf=CONF)


PRELOAD = '''"""Preload pyarrow/pandas BEFORE torch, sam3 and huggingface_hub load.

⛔ WITHOUT THIS THE RUN DIES WITH A WINDOWS ACCESS VIOLATION AND NO TRACEBACK.
MEASURED 2026-08-17 (dev box, Win11, py3.13, torch 2.11.0+cu128): with torch +
sam3 + triton + huggingface_hub already resident, the LAZY `import
pyarrow.dataset` that `pandas.read_parquet` performs inside
`v2_to_pilot.pick_clips` kills the interpreter at 0xC0000005 — faulthandler puts
the fault exactly on that import line. A/B, both outcomes fixed in advance:

    A  pyarrow imported AFTER torch+sam3   -> rc 139 (segfault)
    B  pyarrow imported FIRST (this file)  -> rc 0, read_parquet -> 23 644 rows

It is a DLL load-ORDER conflict, not a data, GPU or model fault. Fixing it here
rather than in `f1_run86.py` keeps the ONE extraction path byte-identical on
Colab, where the conflict does not exist (Linux, and pyarrow is already
imported by the runtime).
"""
try:
    import pyarrow            # noqa: F401
    import pyarrow.dataset    # noqa: F401
    import pyarrow.parquet    # noqa: F401
    import pandas             # noqa: F401
except Exception:             # never break an interpreter that lacks them
    pass
'''


def write_preload(root: str) -> str:
    """Materialise the `sitecustomize` shim the child interpreter picks up.

    `site` imports `sitecustomize` at startup from the first sys.path entry
    that has it, and PYTHONPATH entries are on sys.path by then — so this runs
    BEFORE a single line of `f1_run86.py`, which is the whole point."""
    d = os.path.join(root, "_preload")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "sitecustomize.py")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(PRELOAD)
    return p


def run_f1(todo, root, out, work, chunk, timeout):
    """Invoke the ONE extraction path as a subprocess, host configured by env.

    ⚠️ `F86_TAR=""` skips the tar — it is TRANSPORT for pulling a bank off an
    ephemeral VM, and here the bank is already durable.

    ⛔ `HF_TOKEN` IS REQUIRED IN THE ENV AND IT IS NOT OPTIONAL HYGIENE.
    `facebook/sam3` is a GATED repo, and the weights are fetched by the VENDOR:
    `sam3.model_builder.download_ckpt_from_hf` calls `hf_hub_download(repo_id,
    filename)` with NO token argument, so it authenticates only from the
    ambient environment. Every one of OUR HF reads goes through
    `L.hf_download`, which passes the token explicitly — so the run gets all
    the way through the v2 labels and the shard index and dies only at the
    weights. MEASURED here 2026-08-17: `GatedRepoError: 401 ... Access to model
    facebook/sam3 is restricted`. Colab hid this because the notebook
    environment already carried HF_TOKEN. ⚠️ The token is read IN PLACE from
    Keys.txt by `L.get_hf_token()` and handed to the child through the
    environment — never printed, never written to a file, never in argv."""
    env = dict(os.environ,
               F86_ROOT=root, F86_REPO=REPO, F86_OUT=out, F86_WORK=work,
               F86_TAR="", F86_CHUNK=str(chunk),
               F86_TODO=os.path.join(root, "_no_todo_file.json"),
               F86_TODO_INLINE=json.dumps(list(todo)),
               HF_TOKEN=L.get_hf_token(),
               # ⚠️ PRECAUTION, NOT A DIAGNOSED FIX — stated that way on
               # purpose. MEASURED 2026-08-17: the first attempt died with rc
               # 3221225477 (0xC0000005 ACCESS_VIOLATION) while fetching the
               # 3.45 GB `facebook/sam3` checkpoint, and produced NO output at
               # all (see PYTHONUNBUFFERED below), so WHERE it died is not
               # known. What IS known: the Rust Xet client had host RSS at
               # 5.1 GB for a 3.45 GB file and left the on-disk `.incomplete`
               # reading 0 B for minutes (it buffers in RAM, flushes in
               # bursts); and afterwards the file WAS complete and correct —
               # a re-fetch returned it from cache in 1 s, and
               # `build_processor` then loaded it in 8 s at peak 3.575 GB with
               # the C77 dtype fix applied. ⛔ So do NOT record "xet segfaults"
               # as a finding: the download had in fact succeeded. Xet is
               # disabled here only because it costs nothing once the weights
               # are cached and it removes one suspect from the next run.
               HF_HUB_DISABLE_XET="1",
               # ⛔ WITHOUT THIS A CRASH IS INVISIBLE. The child's stdout is a
               # PIPE, so Python block-buffers it; only `[bank]` lines carry
               # flush=True. MEASURED: the 0xC0000005 above produced ZERO
               # output — `[cfg]`, `[v2]`, `[assets]`, `[sam3] up` were all
               # still sitting in an 8 KB buffer that never flushed, and the
               # log looked like the run had not started. A monitor that
               # cannot see the failure is the trap CLAUDE.md logs for
               # self-matching filters, in a buffering costume.
               PYTHONUNBUFFERED="1",
               PYTHONUTF8="1",
               # ⛔ THE PRELOAD IS LOad-ORDER SURGERY AND IT IS LOAD-BEARING —
               # see `write_preload` for the measurement. It goes FIRST on
               # PYTHONPATH so `site` picks up our `sitecustomize`.
               PYTHONPATH=os.pathsep.join(
                   [os.path.join(root, "_preload"), os.path.join(REPO, "stack")]),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "6"))
    p = subprocess.Popen([sys.executable, F1], env=env, cwd=root,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace",
                         bufsize=1)
    lines = []
    t0 = time.time()
    for line in p.stdout:                       # stream, so a long run is visible
        lines.append(line.rstrip("\n"))
        print(line.rstrip("\n"), flush=True)
        if time.time() - t0 > timeout:
            p.kill()
            lines.append(f"[drive] KILLED after {timeout}s")
            break
    p.wait()
    return p.returncode, lines


def peak_gb(lines):
    """⛔ `torch.cuda.max_memory_allocated` is the ONLY admissible device-memory
    figure (CLAUDE.md Thor trap) — `mem_get_info` free-space arithmetic is not.
    f1 already emits it via `L.gpu_mem_report`; parse, do not re-probe."""
    vals = [float(m) for m in
            re.findall(r"max_memory_allocated = ([\d.]+) GB", "\n".join(lines))]
    return max(vals) if vals else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug120", required=True)
    ap.add_argument("--v2-dir", required=True, help="the banked 115")
    ap.add_argument("--bank", required=True, help="local durable bank for the 86")
    ap.add_argument("--root", default=os.path.join(HERE, "..", "_work86"),
                    help="F86_ROOT — becomes cwd; gets the CLIP BPE vocab copy")
    ap.add_argument("--proof", action="store_true", default=True,
                    help="run 1 clip first and refuse the batch unless it is LIVE")
    ap.add_argument("--no-proof", dest="proof", action="store_false")
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--max-stalls", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "raw",
                                                  "f7_run86_local.json"))
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    work = os.path.join(root, "bf86")
    os.makedirs(a.bank, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    print(f"[preload] {write_preload(root)}")

    # ⚠️ find_bpe() globs the CLIP vocab RELATIVE TO CWD, and f1 chdirs to
    # F86_ROOT. open_clip's copy is found first on this box, but seeding ROOT
    # makes the run independent of which site-packages happen to be importable.
    try:
        import open_clip
        src = os.path.join(os.path.dirname(open_clip.__file__),
                           "bpe_simple_vocab_16e6.txt.gz")
        dst = os.path.join(root, "bpe_simple_vocab_16e6.txt.gz")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
    except Exception as e:                                       # noqa: BLE001
        print(f"[bpe] seed skipped: {type(e).__name__}: {e}")

    # ---- the population, recomputed rather than inherited (f2's asserts) ---- #
    cohort = {c["clip_id"] for c in json.load(
        open(os.path.join(a.aug120, "merged", "ph0_v2.json"),
             encoding="utf-8"))["clips"]}
    have_v2 = {os.path.basename(p)[:-5]
               for p in glob.glob(os.path.join(a.v2_dir, "*.json"))}
    the86 = sorted(cohort - have_v2)
    assert len(cohort) == 201, f"cohort moved: {len(cohort)}"
    assert len(have_v2) == 115, f"v2 leg moved: {len(have_v2)}"
    assert len(the86) == 86, f"residual is {len(the86)}, not 86"
    assert not (set(the86) & have_v2), "residual overlaps the v2 leg"
    print(f"[pop] cohort {len(cohort)} · v2 {len(have_v2)} · residual "
          f"{len(the86)} (recomputed, not read from a manifest)")

    import torch
    L.assert_cuda_conv()                    # a REAL conv2d, not is_available()
    cap = torch.cuda.get_device_capability(0)
    dev = torch.cuda.get_device_name(0)
    tot = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[gpu] {dev} sm_{cap[0]}{cap[1]} · {tot:.2f} GB · torch "
          f"{torch.__version__}")

    log, stalls, peaks = [], 0, []
    t_run = time.time()
    proof: dict = {"ran": False}
    while True:
        cen = census(a.bank, the86)
        done = set(cen["complete_clips"])
        todo = [c for c in the86 if c not in done]
        print(f"[resume] complete {len(done)}/86 · todo {len(todo)} · "
              f"det {cen['n_det_total']} scene {cen['n_scene_det_total']} "
              f"errors {sum(cen['error_census'].values())}")
        if not todo:
            print("F7_ALL_COMPLETE")
            break

        # ⛔ PROOF FIRST: one clip, and the batch does not start unless it is LIVE.
        n = 1 if (a.proof and not proof["ran"] and not done) else len(todo)
        chunk = todo[:n]
        print(f"[exec] {'PROOF (1 clip)' if n == 1 else f'batch {len(chunk)} clips'}"
              f" · f1_run86.py · timeout {a.timeout:.0f}s", flush=True)
        t0 = time.time()
        rc, lines = run_f1(chunk, root, a.bank, work, len(log), a.timeout)
        pk = peak_gb(lines)
        if pk:
            peaks.append(pk)

        if n == 1:
            proof["ran"] = True
            p = os.path.join(a.bank, f"{chunk[0]}.json")
            rec = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
            lvn = ((rec.get("liveness") or {}).get("n_det")) or {}
            proof.update({
                "clip_id": chunk[0], "rc": rc,
                "liveness_n_det": lvn,
                "n_det_total": rec.get("n_det_total"),
                "n_scene_det_total": rec.get("n_scene_det_total"),
                "n_err_total": rec.get("n_err_total"),
                "per_concept_hits": {k: v for k, v in
                                     (rec.get("per_concept_hits") or {}).items()
                                     if v},
                "per_scene_hits": {k: v for k, v in
                                   (rec.get("per_scene_hits") or {}).items() if v},
                "engine": rec.get("engine"),
                "peak_gb": pk, "wall_s": round(time.time() - t0, 1)})
            live = sum(int(v) for v in lvn.values()) > 0
            if not (live and int(rec.get("n_err_total") or 0) == 0):
                proof["PASS"] = False
                # ⚠️ NAME THE RIGHT FAILURE. "No record at all" (the extraction
                # crashed) and "a record whose control is dead" (the C77 shape)
                # are different diagnoses, and printing the second for the first
                # is exactly the "symptom read as its own root cause" trap
                # CLAUDE.md logs for cuDNN/df/cgroup. Keep them apart.
                tail = "\n".join(lines[-25:])
                proof["tail"] = tail
                json.dump({"class": "MEASURED", "proof": proof},
                          open(a.out, "w", encoding="utf-8"), indent=1)
                if not rec:
                    raise SystemExit(
                        f"[proof] EXTRACTION PRODUCED NO RECORD for "
                        f"{chunk[0]} (rc={rc}) — this is a RUN failure, not a "
                        "dead liveness control. Last output:\n" + tail)
                raise SystemExit(
                    "[proof] REFUSING THE BATCH — first clip came back with "
                    f"liveness {lvn} and {rec.get('n_err_total')} errors. On "
                    "Ada (sm_89), not the T4 the C77 dtype fix was developed "
                    "on, a dead road/sky control is exactly what an unapplied "
                    "fix banks: 86 structurally perfect EMPTY records. Fix the "
                    "engine before spending the GPU.")
            proof["PASS"] = True
            print(f"[proof] LIVE — road/sky {lvn} · {rec['n_det_total']} agent "
                  f"+ {rec['n_scene_det_total']} scene detections · 0 errors · "
                  f"peak {pk} GB. Batch may proceed.")

        cen2 = census(a.bank, the86)
        gained = cen2["n_complete"] - len(done)
        log.append({"call": len(log), "requested": len(chunk),
                    "wall_s": round(time.time() - t0, 1), "rc": rc,
                    "peak_gb": pk, "complete_after": cen2["n_complete"],
                    "gained": gained})
        print(f"[call {len(log)-1}] +{gained} complete -> "
              f"{cen2['n_complete']}/86 in {time.time()-t0:.0f}s (rc={rc})")
        if gained <= 0:
            stalls += 1
            print(f"⚠️ STALL {stalls}/{a.max_stalls} — call added no complete "
                  f"clips (rc={rc})")
            if stalls >= a.max_stalls:
                print("F7_STALLED")
                break
        else:
            stalls = 0

    # ---- md5 sidecar: the bank is local, so it carries its own integrity ---- #
    mpath = os.path.join(a.bank, "_md5.json")
    clips = {}
    for p in sorted(glob.glob(os.path.join(a.bank, "*.json"))):
        stem = os.path.basename(p)[:-5]
        if stem.startswith("_"):
            continue
        b = open(p, "rb").read()
        clips[stem] = {"md5": hashlib.md5(b).hexdigest(), "bytes": len(b)}
    json.dump({"class": "MEASURED", "pushed_to_hf": False, "n": len(clips),
               "clips": clips}, open(mpath, "w", encoding="utf-8"), indent=1)

    cen = census(a.bank, the86)
    zero = cen["zero_det_clips"]
    out = {
        "class": "MEASURED", "n_target": 86,
        "host": "dev box local GPU (Colab T4 capacity was 503)",
        "device": dev, "sm": f"sm_{cap[0]}{cap[1]}", "torch": torch.__version__,
        "extraction": {"path": "code/f1_run86.py (UNCHANGED, host-agnostic via "
                               "F86_* env)", "md5": hashlib.md5(
                                   open(F1, "rb").read()).hexdigest()},
        "wall_s_total": round(time.time() - t_run, 1),
        "peak_gb_max": max(peaks) if peaks else None,
        "calls": log, "proof": proof, "pushed_to_hf": False,
        "completion_rule": "liveness control present AND zero errors AND "
                           "schema>=2 AND engine.confidence_threshold==0.25 "
                           "— never a file count (C77)",
        "census": {k: v for k, v in cen.items()
                   if k not in ("zero_det_clips", "complete_clips")},
        "zero_split": {
            "empty_scene_control_live":
                sum(1 for z in zero if z["liveness_live"]),
            "dead_control": sum(1 for z in zero if not z["liveness_live"]),
            "clips": zero},
        "residual": sorted(set(the86) - set(cen["complete_clips"])),
        "PASS": bool(cen["pass_"] and cen["n_complete"] == 86),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"[f7] complete {cen['n_complete']}/86 · det {cen['n_det_total']} · "
          f"scene {cen['n_scene_det_total']} · live {cen['liveness_live']} · "
          f"errors {sum(cen['error_census'].values())} · "
          f"peak {out['peak_gb_max']} GB · PASS {out['PASS']}")
    print("F7_PASS" if out["PASS"] else "F7_INCOMPLETE")
    print("F7_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
