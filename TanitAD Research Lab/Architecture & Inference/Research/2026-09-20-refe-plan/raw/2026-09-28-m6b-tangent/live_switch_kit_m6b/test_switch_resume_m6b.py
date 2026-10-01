#!/usr/bin/env python3
"""Does the LIVE command line + `--yaw-loss plain_tangent --tan-w 0.1 --declare-change yaw_loss` resume cleanly from a
checkpoint the pod's CURRENT trainer wrote -- on the EXACT bytes code/switch_tangent.sh ships? Modelled on
raw/2026-09-27-training-measures/live_switch_kit/test_switch_resume.py (the M6 kit's test), with two differences that
make it stricter: the patched side runs the SHIPPED FILES THEMSELVES (pod_refe/*.py, md5-asserted), not a fresh
application of the patch; and the unpatched side is asserted to be the pod's live bytes (train.py 47dd5b7a, model.py
719e98b5 -- coordinator 2026-09-28 00:14). Real `train.main` on temporary copies (tiny shim, CPU), real resume path.

ROUTE A -- the live run is still on pod_train_v2.sh (heading loss "wrapped"):
  A1  POD trainer, fixed scorer, --grow: starts, checkpoints, halts (rc 7)
  A2  POD trainer resumes with the live OP_FLAGS (--scorer-mode onpolicy ... --declare-change scorer_mode): the
      2026-09-26 switch replayed (rc 7, scorer_mode onpolicy in the checkpoint) -- the live run's state
  A3  KIT --preflight: identity differs ONLY by the declared yaw_loss ('wrapped' -> 'plain_tangent'; tan_w is NOT a
      change: its legacy default is 0.1); writes nothing (rc 0, PREFLIGHT_OK)
  A4  KIT WITHOUT --declare-change yaw_loss: refused (rc 4) -- the negative control
  A5  KIT with the declares: resumes (rc 7 at the test halt); "MEASURE M6 ... plain_tangent" printed; ONE new declared
      change naming only --yaw-loss; the checkpoint's meta carries measures.yaw_loss plain_tangent and its identity
      args tan_w 0.1
  A6  MUTATION: KIT resume from A5's checkpoint with --tan-w 0.2 (undeclared): REFUSED (rc 4), nothing written -- the
      term's weight is identity-tracked, so a silent weight change cannot slip into the live run
  A7  KIT resume again (no declare needed any more): identity matches, completes (rc 0), final meta plain_tangent
ROUTE B -- the live run was first switched to M6 plain (pod_train_v3.sh, the M6 kit's files e963dd12 / c7c602b7 /
d41a4b48):
  B1-B2 as A1-A2; B3 the M6 KIT with `--yaw-loss plain --declare-change yaw_loss` (rc 7); B4 the M6b KIT with the
  declares: rc 7, the new declared change is exactly --yaw-loss 'plain' -> 'plain_tangent'; B5 resume again rc 0.
    python test_switch_resume_m6b.py            -> ZZSWITCH_RESUME_M6B_OK / ZZSWITCH_RESUME_M6B_FAIL  (+ .json record)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[2]
REFE = PKG / "refe"
KIT = HERE / "pod_refe"
M6KIT = PKG / "raw" / "2026-09-27-training-measures" / "live_switch_kit" / "pod_refe"
sys.path.insert(0, str(REFE))
import selftest_measures as SM  # noqa: E402  make_bank / targs / the child trainer shim

POD_LIVE_MD5 = {"train.py": "47dd5b7ae37921bc4e301f75d9389fb8", "model.py": "719e98b54e9b50dc206dc0121f1da8e1"}
KIT_MD5 = {"train.py": "b986a5c70f4b1e0b2a44c8bce5a16bba", "model.py": "c7c602b783bc989e13b4843a8fba8d54",
           "measures.py": "27564fe06595181973b1a34761aa489b"}
M6KIT_MD5 = {"train.py": "e963dd12e99105f219a5547bef639166", "model.py": "c7c602b783bc989e13b4843a8fba8d54",
             "measures.py": "d41a4b48cf0738a241f42417b4154644"}


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main() -> int:
    import torch
    tmp = Path(tempfile.mkdtemp(prefix="switch_resume_m6b_"))
    pod, kit, m6 = tmp / "pod", tmp / "kit", tmp / "m6kit"
    for d, over in ((pod, None), (kit, KIT), (m6, M6KIT)):
        d.mkdir()
        for f in REFE.glob("*.py"):
            shutil.copy2(f, d / f.name)
        if over is not None:
            for f in ("train.py", "model.py", "measures.py"):
                shutil.copy2(over / f, d / f)
    md5s = {"pod": {f: md5(pod / f) for f in POD_LIVE_MD5}, "kit": {f: md5(kit / f) for f in KIT_MD5},
            "m6kit": {f: md5(m6 / f) for f in M6KIT_MD5}}
    bank, op, empty = tmp / "bank", tmp / "op", tmp / "empty"
    empty.mkdir()
    SM.make_bank(bank, 8, [8.0, 12.0, 15.0], [0.0, 0.02, -0.03], rank1_every=2, onpolicy=op)
    grow = ["--grow", "--epochs", "3", "--grow-scenes", "8", "--ckpt-every-steps", "1"]
    OPF = ["--scorer-mode", "onpolicy", "--onpolicy-targets", str(op), "--declare-change", "scorer_mode"]
    M6B = ["--yaw-loss", "plain_tangent", "--tan-w", "0.1", "--declare-change", "yaw_loss"]
    M6 = ["--yaw-loss", "plain", "--declare-change", "yaw_loss"]
    n = [0]

    def child(root, argv):
        n[0] += 1
        spec = tmp / f"spec{n[0]}.json"
        out = tmp / f"child{n[0]}.out"
        spec.write_text(json.dumps({"root": str(root), "out": str(out), "argv": argv}))
        lf = tmp / f"child{n[0]}.log"
        with open(lf, "w", encoding="utf-8") as fh:
            subprocess.run([sys.executable, str(REFE / "selftest_measures.py"), "--child", "trainer", "--spec", str(spec)],
                           stdout=fh, stderr=subprocess.STDOUT, cwd=str(root),
                           env=dict(os.environ, OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8"))
        rc = torch.load(out, weights_only=False)["rc"] if out.exists() else None
        return rc, lf.read_text(encoding="utf-8", errors="replace")

    def ck(run):
        return torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)

    def decl(run):
        ev = [json.loads(l) for l in open(run / "metrics.jsonl", encoding="utf-8")]
        return [e["changes"] for e in ev if e.get("event") == "declared_change"]

    def stamp(run):
        return {p.name: p.stat().st_mtime_ns for p in run.iterdir()}

    # ---------------------------------------------------------------- ROUTE A: from the v2 (wrapped) live state
    ra = tmp / "run_a"

    def aa(*extra):
        return SM.targs(bank, ra, empty, *grow, *extra, batch=2, accum=1)
    res = {}
    res["A1"] = child(pod, aa("--halt-after-steps", "1"))
    res["A2"] = child(pod, aa("--resume", *OPF, "--halt-after-steps", "2"))
    mode_a2 = ck(ra)["identity"]["args"].get("scorer_mode")
    b3 = stamp(ra)
    res["A3"] = child(kit, aa("--resume", *OPF, *M6B, "--preflight"))
    a3 = stamp(ra)
    res["A4"] = child(kit, aa("--resume", *OPF, "--yaw-loss", "plain_tangent", "--tan-w", "0.1"))
    nd4 = len(decl(ra))
    res["A5"] = child(kit, aa("--resume", *OPF, *M6B, "--halt-after-steps", "4"))
    dA = decl(ra)
    ck5 = ck(ra)
    b6 = stamp(ra)
    res["A6"] = child(kit, aa("--resume", *OPF, "--yaw-loss", "plain_tangent", "--tan-w", "0.2"))
    a6 = stamp(ra)
    res["A7"] = child(kit, aa("--resume", *OPF, "--yaw-loss", "plain_tangent", "--tan-w", "0.1"))
    finA = torch.load(ra / "model_final.pt", map_location="cpu", weights_only=False) if (ra / "model_final.pt").exists() \
        else {"meta": {}}
    # ---------------------------------------------------------------- ROUTE B: from a v3 (M6 plain) live state
    rb = tmp / "run_b"

    def bb(*extra):
        return SM.targs(bank, rb, empty, *grow, *extra, batch=2, accum=1)
    res["B1"] = child(pod, bb("--halt-after-steps", "1"))
    res["B2"] = child(pod, bb("--resume", *OPF, "--halt-after-steps", "2"))
    res["B3"] = child(m6, bb("--resume", *OPF, *M6, "--halt-after-steps", "3"))
    y_b3 = (ck(rb)["meta"].get("measures") or {}).get("yaw_loss")
    res["B4"] = child(kit, bb("--resume", *OPF, *M6B, "--halt-after-steps", "5"))
    dB = decl(rb)
    y_b4 = (ck(rb)["meta"].get("measures") or {}).get("yaw_loss")
    res["B5"] = child(kit, bb("--resume", *OPF, "--yaw-loss", "plain_tangent", "--tan-w", "0.1"))
    finB = torch.load(rb / "model_final.pt", map_location="cpu", weights_only=False) if (rb / "model_final.pt").exists() \
        else {"meta": {}}

    WT = "--yaw-loss: 'wrapped' -> 'plain_tangent'"
    PT = "--yaw-loss: 'plain' -> 'plain_tangent'"
    checks = {
        "0 the unpatched side IS the pod's live bytes; the patched side IS the shipped kit (md5)":
            md5s["pod"] == POD_LIVE_MD5 and md5s["kit"] == KIT_MD5 and md5s["m6kit"] == M6KIT_MD5,
        "A1 pod trainer start halts (rc 7)": res["A1"][0] == 7,
        "A2 pod on-policy switch replayed (rc 7, scorer_mode onpolicy in the checkpoint)":
            res["A2"][0] == 7 and mode_a2 == "onpolicy",
        "A3 KIT preflight OK; ONLY the declared yaw_loss differs (no tan_w line); nothing written":
            res["A3"][0] == 0 and "PREFLIGHT_OK" in res["A3"][1] and WT in res["A3"][1]
            and "--tan-w:" not in res["A3"][1] and b3 == a3,
        "A4 KIT without --declare-change yaw_loss: REFUSED (rc 4), no new declared change":
            res["A4"][0] == 4 and "REFUSING TO RESUME" in res["A4"][1] and len(decl(ra)) >= nd4,
        "A5 KIT switch resumes (rc 7), MEASURE M6 names plain_tangent, the new declared change is ONLY yaw_loss":
            res["A5"][0] == 7 and "MEASURE M6" in res["A5"][1] and "--yaw-loss plain_tangent" in res["A5"][1]
            and len(dA) >= 2 and dA[-1] == [WT],
        "A5 the switched checkpoint: meta measures.yaw_loss plain_tangent, identity tan_w 0.1":
            (ck5["meta"].get("measures") or {}).get("yaw_loss") == "plain_tangent"
            and ck5["identity"]["args"].get("tan_w") == 0.1,
        "A6 MUTATION: an undeclared --tan-w 0.2 is REFUSED (rc 4) and writes nothing":
            res["A6"][0] == 4 and "REFUSING TO RESUME" in res["A6"][1] and b6 == a6,
        "A7 resume again needs no declare, completes (rc 0), final meta plain_tangent":
            res["A7"][0] == 0 and (finA["meta"].get("measures") or {}).get("yaw_loss") == "plain_tangent",
        "B1-B3 pod start, on-policy replay, M6 plain switch (rc 7 each; checkpoint yaw_loss plain)":
            res["B1"][0] == 7 and res["B2"][0] == 7 and res["B3"][0] == 7 and y_b3 == "plain",
        "B4 KIT from a v3 (plain) checkpoint: rc 7, the new declared change is exactly plain -> plain_tangent":
            res["B4"][0] == 7 and len(dB) >= 3 and dB[-1] == [PT] and y_b4 == "plain_tangent",
        "B5 resume again completes (rc 0), final meta plain_tangent":
            res["B5"][0] == 0 and (finB["meta"].get("measures") or {}).get("yaw_loss") == "plain_tangent",
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  (route A declared changes: {dA}; route B: {dB}; rcs {({k: v[0] for k, v in res.items()})}; temp {tmp})")
    ok = all(checks.values())
    rec = {"checks": checks, "rc": {k: v[0] for k, v in res.items()}, "declared_A": dA, "declared_B": dB, "md5": md5s,
           "tested_sha256": {f"{side}/{f}": hashlib.sha256(Path(d / f).read_bytes()).hexdigest()
                             for side, d in (("kit", kit), ("pod", pod)) for f in ("train.py", "model.py")}
           | {"kit/measures.py": hashlib.sha256((kit / "measures.py").read_bytes()).hexdigest(),
              "selftest_measures.py": hashlib.sha256((REFE / "selftest_measures.py").read_bytes()).hexdigest()},
           "failed": [k for k, v in checks.items() if not v]}
    json.dump(rec, open(HERE / "test_switch_resume_m6b.json", "w"), indent=1)
    if not ok:
        for k in ("A3", "A5", "A6", "B4"):
            print(f"---- {k} tail ----\n{res[k][1][-1500:]}")
    if ok:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ZZSWITCH_RESUME_M6B_OK" if ok else "ZZSWITCH_RESUME_M6B_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
