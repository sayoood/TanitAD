"""THE ARM-INDEPENDENT CONTROL, run on warmup with ZERO new scoring.

The prereg (2fc5bcc) commits to `f_slow > f_fast`: the <= 5 m clause must fire MORE on
slow-start scenes than on fast-start ones, or start speed is not what drives it and the
stratification separates nothing.

That control needs no arm and no GPU -- warmup's per-scene EP is already banked (3b02a41) and
the clause's signature is arm-INDEPENDENCE (every arm reads EP exactly 1.0 on a fired scene,
established at d86dccb). All that was missing is the join key, which is what this builds.

⛔ This tests MY DERIVATION, not the hypothesis. The 5.0 m/s threshold was fixed at 2fc5bcc
from the scoring rule; if the clause does not concentrate in the SLOW stratum, the derivation
in that prereg is wrong and the navhard arms must not be run on it.

⚠️ Warmup is a DIFFERENT venue from navhard and its scores are already known, so nothing here
is a score-blind test of the hypothesis -- it is a test of the threshold's MECHANISM, which is
a property of the scenes and the reference planner, not of any arm.
"""
from __future__ import annotations

import csv
import io
import json
import pathlib
import re
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from peek import WinUnpickler  # noqa: E402

TAR = "C:/Users/Admin/navsim/data/navsim-v2/navsim_v2.2_warmup_two_stage.tar.gz"
RAW = pathlib.Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                   "2026-09-19-navsim-refcv4b-bridge/raw")
HERE = pathlib.Path(__file__).resolve().parent
HEX = re.compile(r"^[0-9a-f]{8,}$")
ARMS = {"STOP": "score_STOP_zero.csv", "CV": "score_CV_official.csv",
        "A1": "score_A1_ego_cmd.csv", "ECHO": "score_ECHO_ha0_ext.csv",
        "A2": "score_A2_vision_pure.csv", "A4": "score_A4_blind_ego_cmd.csv"}
COL = "ego_progress_stage_two"

THRESHOLD_MS = 5.0          # fixed at 2fc5bcc, from the rule -- NOT refitted here


def read_arm(fn):
    out = {}
    for r in csv.DictReader(open(RAW / fn, newline="", encoding="utf-8")):
        t = (r.get("token") or "")
        if not HEX.match(t):
            continue
        try:
            out[t] = float(r.get(COL, ""))
        except (TypeError, ValueError):
            pass
    return out


def warmup_speeds():
    """scene_token -> |v0| for every warmup scene the archive carries."""
    rows, members = {}, 0
    with tarfile.open(TAR, "r:gz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".pkl")):
                continue
            members += 1
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                o = WinUnpickler(io.BytesIO(f.read())).load()
                # ⛔ initial_token, NOT scene_token. MEASURED: scene_token joins 0 of 204
                # against the scorer's CSVs; initial_token joins 204/204 (which_token.py).
                tok = o["scene_metadata"]["initial_token"]
                v = o["frames"][-1]["ego_status"]["ego_velocity"]
                rows[tok] = round((float(v[0]) ** 2 + float(v[1]) ** 2) ** 0.5, 6)
            except Exception:
                pass
            if members % 200 == 0:
                print(f"  {members} members, {len(rows)} scenes", flush=True)
    print(f"  pkl members={members} scenes_read={len(rows)}")
    return rows


def main() -> int:
    eps = {a: read_arm(fn) for a, fn in ARMS.items()}
    toks = set(eps["STOP"])
    for a in eps:
        toks &= set(eps[a])
    print(f"tokens present in ALL {len(ARMS)} arms: {len(toks)}")

    # the clause's signature, re-derived here rather than inherited: EP == 1.0 for EVERY arm
    fired = {t for t in toks if all(eps[a][t] == 1.0 for a in eps)}
    print(f"clause fired (all arms EP==1.0): {len(fired)} / {len(toks)}"
          f" = {len(fired)/len(toks):.4f}")

    speeds = warmup_speeds()
    joined = sorted(toks & set(speeds))
    print(f"JOINED scored<->speed: {len(joined)} of {len(toks)} scored tokens")
    if not joined:
        print("ZZABORT: the join is empty -- the CSV token is not scene_token")
        return 3

    fast = [t for t in joined if speeds[t] >= THRESHOLD_MS]
    slow = [t for t in joined if speeds[t] < THRESHOLD_MS]
    ff = sum(1 for t in fast if t in fired)
    fs = sum(1 for t in slow if t in fired)
    assert fast and slow, "control: a stratum is empty"

    f_fast = ff / len(fast)
    f_slow = fs / len(slow)
    out = {
        "_what": "arm-independent control for the 5.0 m/s fast-start threshold, on WARMUP",
        "_threshold_provenance": "fixed at 2fc5bcc from pdm_scorer.py:232-237; NOT refitted here",
        "_clause_detection": "EP == 1.0 for ALL SIX arms on the same scene (arm-independence "
                             "is the clause's signature; established d86dccb)",
        "threshold_ms": THRESHOLD_MS,
        "n_joined": len(joined), "n_fast": len(fast), "n_slow": len(slow),
        "fired_total": len(fired), "fired_fast": ff, "fired_slow": fs,
        "f_fast": round(f_fast, 4), "f_slow": round(f_slow, 4),
        "control_f_slow_gt_f_fast": bool(f_slow > f_fast),
        "ratio_slow_over_fast": (round(f_slow / f_fast, 3) if f_fast else None),
    }
    print(json.dumps(out, indent=1))
    verdict = ("CONTROL PASSES -- the clause concentrates in SLOW starts"
               if f_slow > f_fast else
               "CONTROL FAILS -- start speed does not drive the clause; the 2fc5bcc "
               "derivation is WRONG and the navhard arms must not run on this stratum")
    out["_verdict"] = verdict
    print(verdict)
    (HERE / "warmup_control.json").write_bytes(json.dumps(out, indent=1).encode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
