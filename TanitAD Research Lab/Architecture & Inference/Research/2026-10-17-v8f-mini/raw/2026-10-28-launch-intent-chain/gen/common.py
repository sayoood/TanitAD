#!/usr/bin/env python3
"""Shared constants and helpers of the LAUNCH-INTENT box-2 chain generators (PREREG_V8F_LAUNCH_INTENT, arm A at both seeds).

Nothing here touches box 2, the installed C:/Users/Admin/box2_ship, or git beyond read-only `git show` / `rev-parse` / `ls-remote` on the mirror.
Every pinned value is COMPUTED from a banked artifact (never retyped); a value that cannot be computed aborts the generator.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent                                                  # .../dagland/launch_intent_chain
SHIP = OUT / "box2_ship"
QSTAGE = OUT / "queue_stage"
ART = Path("E:/Projects/TanitAD-artifacts")
LI = ART / "dagland" / "launch_intent"                             # the draft PREREG + the staged trainer build
STAGE = LI / "stage"
R2 = ART / "dagorc_r2"                                             # the model: the DAG-ORC r2 chain and its generators
R2SHIP = R2 / "box2_ship"
OACC = ART / "v8f-dagoacc"                                         # DAG-OACC's banked runs (the CTRL and the one-lever reference)
REPDIR = ART / "v8f-dag-orc-oacc"                                  # DAG-ORC r2's pulls (REP lives here)
DAG1 = ART / "v8f-dagger"
INSTALLED = Path("C:/Users/Admin/box2_ship")                       # READ ONLY: this package never installs into it
MIRROR = "C:/Users/Admin/tanitad-push"
PRE_TIP = "94a47f4b"                                               # the PRE-LEVER tip: proves each NEW lever grep discriminates (the lever is absent from its blobs); the mirror's HEAD is stale: never used
LANDED_COMMIT = "b09ac89fc9d6e81baf3977954ae638919b979e71"         # the LANDED LAUNCH-INTENT flag (coordinator message 2026-10-28, ls-remote verified: agent/arch-inf-20260803). INHERITED, then RE-MEASURED by make_chain_li.py (blob sha256 below vs `git show`)
LANDED_TRAINER_SHA = "0d708b3cb841bc1f1832b09fe61e3d8421a4d3f59769f4435c0b22bb77e268ce"      # stack/scripts/v8f_mini_train.py at LANDED_COMMIT (== the staged trainer)
LANDED_MODEL_SHA = "b337ccecd3abfa4250a77e233502d9d3464a280dbafdc86b1e94ecbeef682721"        # stack/tanitad/models/v8f_mini.py at LANDED_COMMIT (== the staged model)
TIP = LANDED_COMMIT[:8]                                            # the commit the queue stage, the export and the dry run are built on: the landed one (its queue blob a45dedf2 == the pre-lever tip's)
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
QUEUE_PATH = f"{PK}/code/v8f_mini_box2_queue.cmd"
SCRATCH = Path("F:/TanitAD-scratch/launch_intent_chain")      # the ONLY place a self-test step may write (sealed.py asserts it in the same shell as every step)
PYV = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BASH = "C:/Program Files/Git/bin/bash.exe"

LI_ARM = "launchintent"
LI_W = "1.58926"                                                   # W_A (PREREG section 3, arm A, share 1/3); arm B is W 0.726175 and is NOT in this chain
LI_W_B = "0.726175"
LI_W_HI = "4.17852"
WACC = "49.6969"


def git(*args, binary: bool = False, check: bool = True):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "-c", "core.autocrlf=false", *args], capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: rc {r.returncode}: {r.stderr.decode('utf-8', 'replace')[:300]}")
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def tip_full(ref: str | None = None) -> str:
    t = git("rev-parse", f"{ref or TIP}^{{commit}}").strip()
    assert re.fullmatch(r"[0-9a-f]{40}", t), t
    return t


QUEUE_TIP_FILE = QSTAGE / f"queue_tip_{TIP}.cmd"                   # the landed commit's queue blob, kept beside the staged queue for `diff`
LI_ARMS = ("launchintent", "launchintentrep")                      # A_s (trainer seed s) and A_rep,s (trainer seed 10 + s): the queue's arm names
RUN_ORDER = (("launchintent", 0), ("launchintent", 1), ("launchintentrep", 0), ("launchintentrep", 1))      # MM_REDESIGN section 3: A_0, A_1, A_rep,0, A_rep,1 (the main pair first: a killed chain still yields it)
RUN_LABEL = {("launchintent", 0): "A_0", ("launchintent", 1): "A_1", ("launchintentrep", 0): "A_rep,0", ("launchintentrep", 1): "A_rep,1"}


def tseed(arm: str, seed: int) -> int:
    """The trainer's --seed of a run: the init seed S for A, 10 + S for A_rep (REP's rule: DAG-ORC r2's M2b_s<S>_dagoaccorcrep)."""
    return seed + (10 if arm == "launchintentrep" else 0)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sums(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        h, n = line.split(" ", 1)
        out[n.strip().lstrip("*")] = h
    return out


def rep(t, old, new, n: int = 1):
    """Exact-count replacement on str or bytes: aborts unless ``old`` occurs exactly ``n`` times."""
    if isinstance(t, bytes):
        o = old if isinstance(old, bytes) else old.encode("utf-8")
        assert t.count(o) == n, (old[:90], t.count(o), n)
        return t.replace(o, new if isinstance(new, bytes) else new.encode("utf-8"))
    assert t.count(old) == n, (old[:90], t.count(old), n)
    return t.replace(old, new)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def w_from_rule(share_num: int, share_den: int, f: float, digits: int = 6) -> str:
    """PREREG section 3: 1 + W = s (1 - f) / ((1 - s) f), 6 significant digits; returns W as the string the argv carries."""
    s = share_num / share_den
    w1 = s * (1 - f) / ((1 - s) * f)
    return f"{w1 - 1.0:.{digits}g}"


def f_measured() -> float:
    j = load(LI / "f_measure.json")
    assert j["n_launch"] == 49618 and j["n_windows"] == 306566 and j["n_gt_valid"] == 306566, j
    f = j["n_launch"] / j["n_windows"]
    assert abs(f - j["f"]) < 1e-12
    return f


def write_lf(path: Path, text: str) -> None:
    assert "\r" not in text, f"{path}: CR in an LF file"
    path.write_bytes(text.encode("utf-8"))
    print("wrote", path, len(text.encode("utf-8")), "bytes (LF)")
