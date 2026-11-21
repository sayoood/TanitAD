#!/usr/bin/env python3
"""Targeted extraction of NAVSIM v2 `navhard_two_stage` STAGE-1 camera frames from the already
downloaded, sha256-verified NAVSIM v1 `navtest` camera archives.

WHY THIS EXISTS (MEASURED 2026-09-20)
-------------------------------------
navhard stage 1 needs 5,400 camera jpgs over 450 scorer tokens / 76 logs. NONE are unpacked on this
box (W7's census: stage 1 0/5,400 present, stage 2 65,544/65,544 as the same-breath control), so a
camera arm's stage 1 can only be a declared CV stand-in and the official two-stage EPDMS is
UNDEFINED. But the frames are NOT absent from the box — every one of navhard's 76 logs is inside
navtest's 136-log universe, and the navtest camera archives (32 shards, 127,882,665,618 B, sha256
== HF X-Linked-ETag, all_verified) store WHOLE LOGS, including frames navtest's own scorer index
never references. W7 confirmed on shard 6: 2 of the 76 navhard logs, 120/120 of their stage-1 files
present, with a stage-2 control of 0/1,185.

⇒ The unlock is an EXTRACTION, not a download.

WHAT THIS REFUSES TO DO
-----------------------
⛔ It never reports success on a count it did not assert BY NAME. The failure this guards against is
the poisoned-bank class already in CLAUDE.md: a pre-allocated output that looks complete because its
SIZE is right while its CONTENT is wrong or partial. Here a silent partial extraction would produce
a stage-1 arm scored on fewer frames than the scorer asks for — a quiet, plausible, wrong number.

Therefore every extracted file is asserted on CONTENT (JPEG magic `\xff\xd8\xff`, non-zero size),
the per-shard receipt records exactly which members matched, and `--verify` re-checks the FULL
5,400-name set against disk and prints one of three verdicts and nothing else:

    COMPLETE      all 5,400 present and content-valid
    INCOMPLETE    names missing (they are listed)
    INCONCLUSIVE  the probe itself could not read (never reported as absence)

⚠️ INCONCLUSIVE is a distinct outcome on purpose. `CLAUDE.md`: *a count of 0 from a file that could
not be READ is indistinguishable from a genuine absence*. A run that cannot open the destination
tree reports INCONCLUSIVE, never INCOMPLETE.

Streaming (`r|gz`) is single-pass and cannot seek, which is the right mode for a 6 GB member-filtered
read and keeps RSS flat. D: is exFAT and shared with training reads, so shards are processed one at
a time and the tool is resumable: a file already on disk with valid content is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time

HF_PREFIX = "openscene-v1.1/sensor_blobs/test/"
JPEG_MAGIC = b"\xff\xd8\xff"


def load_need(inputs_json: str) -> dict:
    """The stage-1 (log, CAM_DIR, basename) set, read from the suite's own agent-inputs export.

    The tokens live under `d["tokens"]`; the top level carries metadata. Reading the top level
    instead yields 0 tokens and looks exactly like an empty corpus — that mistake was made once
    already this session, so the count is asserted below rather than trusted.
    """
    with open(inputs_json, encoding="utf-8") as f:
        d = json.load(f)
    toks = d.get("tokens")
    if not isinstance(toks, dict) or not toks:
        raise SystemExit(f"⛔ INCONCLUSIVE: no 'tokens' dict in {inputs_json}")
    need: dict[tuple[str, str, str], None] = {}
    n_tok = 0
    for _t, v in toks.items():
        if not isinstance(v, dict) or v.get("stage") != 1:
            continue
        n_tok += 1
        log = v.get("log_name")
        for cam, lst in (v.get("cams") or {}).items():
            for p in (lst or []):
                if isinstance(p, str) and p:
                    need[(log, cam.upper(), os.path.basename(p))] = None
    if not need:
        raise SystemExit("⛔ INCONCLUSIVE: 0 stage-1 camera files derived — schema changed?")
    print(f"[need] stage-1 tokens={n_tok} logs={len({k[0] for k in need})} files={len(need)}",
          flush=True)
    return need


def dest_for(root: str, log: str, cam: str, base: str) -> str:
    return os.path.join(root, log, cam, base)


def valid(path: str) -> bool:
    """CONTENT assertion, not existence: a zero-byte or non-JPEG file is NOT a hit."""
    try:
        if os.path.getsize(path) <= 0:
            return False
        with open(path, "rb") as f:
            return f.read(3) == JPEG_MAGIC
    except OSError:
        return False


def extract_shard(tgz: str, need: dict, root: str, receipt_dir: str) -> dict:
    si = os.path.basename(tgz).rsplit("_", 1)[-1].split(".")[0]
    t0 = time.time()
    wanted = {}
    for (log, cam, base) in need:
        wanted[f"{HF_PREFIX}{log}/{cam}/{base}"] = (log, cam, base)
    got, skipped, bad = 0, 0, []
    os.makedirs(root, exist_ok=True)
    with tarfile.open(tgz, mode="r|gz") as tf:          # streaming: no seeking, flat RSS
        for m in tf:
            if not m.isfile():
                continue
            hit = wanted.get(m.name)
            if hit is None:
                continue
            log, cam, base = hit
            out = dest_for(root, log, cam, base)
            if valid(out):
                skipped += 1
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                bad.append({"member": m.name, "reason": "extractfile returned None"})
                continue
            data = src.read()
            if not data.startswith(JPEG_MAGIC):
                bad.append({"member": m.name, "reason": f"not JPEG (first3={data[:3]!r})"})
                continue
            tmp = out + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, out)
            got += 1
    rep = {"shard": si, "tgz": tgz, "archive_bytes": os.path.getsize(tgz),
           "extracted": got, "already_present": skipped, "rejected": bad[:40],
           "n_rejected": len(bad), "wall_s": round(time.time() - t0, 1)}
    os.makedirs(receipt_dir, exist_ok=True)
    with open(os.path.join(receipt_dir, f"shard_{si}.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(f"[shard {si}] extracted={got} already={skipped} rejected={len(bad)} "
          f"wall={rep['wall_s']}s", flush=True)
    return rep


def verify(need: dict, root: str) -> int:
    """Assert the FULL name set against disk. Three verdicts; absence is never inferred."""
    if not os.path.isdir(root):
        print(f"INCONCLUSIVE: destination root unreadable: {root}")
        return 3
    probe_ok = 0
    missing = []
    for (log, cam, base) in need:
        p = dest_for(root, log, cam, base)
        if valid(p):
            probe_ok += 1
        else:
            missing.append(f"{log}/{cam}/{base}")
    total = len(need)
    if probe_ok == 0 and total > 0:
        # cannot distinguish "nothing extracted yet" from "cannot read" — say so
        print(f"INCONCLUSIVE: 0/{total} readable; the probe demonstrated no successful read at all")
        return 3
    if missing:
        print(f"INCOMPLETE: {probe_ok}/{total} present; {len(missing)} missing")
        for m in missing[:20]:
            print("   missing:", m)
        return 1
    print(f"COMPLETE: {probe_ok}/{total} stage-1 camera files present and content-valid")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="navsim_agent_inputs.json for navhard_two_stage")
    ap.add_argument("--archives", required=True, help="dir holding openscene_sensor_test_camera_*.tgz")
    ap.add_argument("--dest", required=True, help="navhard sensor_blobs root to populate")
    ap.add_argument("--shards", default="", help="comma list, e.g. 6 or 0-31 (default: none)")
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--verify", action="store_true", help="verify the full set and exit")
    a = ap.parse_args()

    need = load_need(a.inputs)
    if a.verify:
        return verify(need, a.dest)

    ids: list[int] = []
    for part in filter(None, a.shards.split(",")):
        if "-" in part:
            lo, hi = part.split("-")
            ids += list(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    if not ids:
        print("nothing to do: pass --shards or --verify")
        return 0
    for i in ids:
        tgz = os.path.join(a.archives, f"openscene_sensor_test_camera_{i}.tgz")
        if not os.path.exists(tgz):
            print(f"⛔ shard {i}: archive absent {tgz}")
            return 2
        extract_shard(tgz, need, a.dest, a.receipts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
