#!/usr/bin/env python3
"""POSITIVE per-stage camera-frame census for a NavSim two-stage split.

⛔ WHY THIS EXISTS (W7, 2026-09-20). The bridge's stage-1 path is a DECLARED CV
stand-in, hard-coded from E2's warmup finding (0/192 stage-1 jpgs). A hard-coded
stand-in is a claim about warmup that would score silently on navhard, so this
probe re-establishes the fact on THIS split, by COUNT, before anything is scored:
an arm whose stage 1 is a CV stand-in has NO official two-stage EPDMS.

⭐ THE DISCRIMINATING CONTROL. A "0 hits" from a filesystem probe is a claim about
the PROBE unless something that must read non-zero read non-zero in the same pass
(CLAUDE.md: a count of 0 from a file that could not be READ is indistinguishable
from a genuine absence). Every stage is therefore counted against EVERY root, and
the report carries both stages side by side: stage 2 is the same-breath control
for stage 1, and if BOTH read zero the verdict is INCONCLUSIVE, never "absent".

    python probe_stage_frames.py --inputs <export json> --out raw/stage_frame_census.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

CAMS = ("cam_l0", "cam_f0", "cam_r0")
DEFAULT_ROOTS = (
    "C:/Users/Admin/navsim-crun/data/openscene/navhard_two_stage/sensor_blobs",
    "C:/Users/Admin/navsim-crun/data/openscene/warmup_two_stage/sensor_blobs",
    "C:/Users/Admin/navsim-crun/data/openscene/sensor_blobs/test",
    "C:/Users/Admin/navsim/data/openscene/navhard_two_stage/sensor_blobs",
    "C:/Users/Admin/navsim/data/openscene/warmup_two_stage/sensor_blobs",
    "C:/Users/Admin/navsim/data/openscene/sensor_blobs/test",
    "C:/Users/Admin/navsim/data/openscene/openscene-v1.1/sensor_blobs",
    "D:/Archive/devbox-C/navsim/sensor_blobs",
)


def census(inputs: str, roots, sample_read: int = 8) -> dict:
    doc = json.load(open(inputs, encoding="utf-8"))
    toks = doc["tokens"]
    roots = [(r, pathlib.Path(r)) for r in roots]
    out = {"_what": ("per-stage camera-frame census: for EVERY scorer token, EVERY history frame, "
                     "EVERY camera, does the source jpg EXIST on disk under each root"),
           "inputs": inputs, "roots": [r for r, _ in roots],
           "root_exists": {r: p.exists() for r, p in roots},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": {}}
    for stage in (1, 2):
        sel = {t: r for t, r in toks.items() if r["stage"] == stage}
        expected = sum(len(r["cams"][c]) for r in sel.values() for c in CAMS)
        per_root, first_missing, first_present = {}, None, None
        for rname, rp in roots:
            n = 0
            for t, r in sel.items():
                for c in CAMS:
                    for rel in r["cams"][c]:
                        p = rp / rel
                        try:
                            ok = p.is_file()
                        except OSError:
                            ok = False
                        if ok:
                            n += 1
                            if first_present is None:
                                first_present = str(p)
                        elif first_missing is None:
                            first_missing = str(p)
            per_root[rname] = n
        # any-root union (a frame reachable from at least one root)
        union = 0
        for t, r in sel.items():
            for c in CAMS:
                for rel in r["cams"][c]:
                    if any((rp / rel).is_file() for _, rp in roots):
                        union += 1
        # BYTE-LEVEL control: a file that "exists" but cannot be READ is not a frame.
        read_ok, read_fail, read_tried = 0, [], 0
        for t, r in sorted(sel.items()):
            if read_tried >= sample_read:
                break
            for c in CAMS:
                rel = r["cams"][c][-1]
                for _, rp in roots:
                    p = rp / rel
                    if p.is_file():
                        read_tried += 1
                        try:
                            with open(p, "rb") as fh:
                                head = fh.read(4)
                            if head[:2] == b"\xff\xd8":
                                read_ok += 1
                            else:
                                read_fail.append({"path": str(p), "why": f"not a JPEG (magic {head[:2]!r})"})
                        except Exception as e:                         # noqa: BLE001
                            read_fail.append({"path": str(p), "why": f"{type(e).__name__}: {e}"})
                        break
                if read_tried >= sample_read:
                    break
        out["stages"][f"stage_{stage}"] = {
            "n_tokens": len(sel), "cam_files_expected": expected,
            "cam_files_present_per_root": per_root,
            "cam_files_present_any_root": union,
            "fraction_present": round(union / expected, 6) if expected else None,
            "example_missing": first_missing, "example_present": first_present,
            "byte_read_control": {"tried": read_tried, "jpeg_ok": read_ok, "failures": read_fail[:5]},
        }
    s1 = out["stages"]["stage_1"]["cam_files_present_any_root"]
    s2 = out["stages"]["stage_2"]["cam_files_present_any_root"]
    if s1 == 0 and s2 == 0:
        v = ("INCONCLUSIVE: BOTH stages read zero — with no stage that must read non-zero, a zero is a "
             "claim about this probe, not about the corpus")
    elif s1 == 0:
        v = ("STAGE_1_CAMERA_FRAMES_ABSENT (discriminating control PASSED: stage 2 read "
             f"{s2}/{out['stages']['stage_2']['cam_files_expected']} in the same pass, so the probe can "
             "see files) => a CAMERA arm's stage 1 can only be a DECLARED stand-in and its official "
             "two-stage EPDMS is UNDEFINED")
    elif s1 < out["stages"]["stage_1"]["cam_files_expected"]:
        v = f"STAGE_1_PARTIAL: {s1} of {out['stages']['stage_1']['cam_files_expected']}"
    else:
        v = "STAGE_1_COMPLETE: a camera arm can answer both stages"
    out["verdict"] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", default=",".join(DEFAULT_ROOTS))
    a = ap.parse_args(argv)
    rep = census(a.inputs, [r for r in a.roots.split(",") if r.strip()])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({"verdict": rep["verdict"],
                      "stage_1": {k: rep["stages"]["stage_1"][k] for k in
                                  ("n_tokens", "cam_files_expected", "cam_files_present_any_root")},
                      "stage_2": {k: rep["stages"]["stage_2"][k] for k in
                                  ("n_tokens", "cam_files_expected", "cam_files_present_any_root")}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
