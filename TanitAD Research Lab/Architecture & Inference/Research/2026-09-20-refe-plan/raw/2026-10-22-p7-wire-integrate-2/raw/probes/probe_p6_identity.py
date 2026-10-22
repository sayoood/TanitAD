"""P7-WIRE-INTEGRATE-2: is the refe_p6 DEFAULT path byte-identical?  Run once per tree (tip, overlay); the printed lines must be equal.
  * sha256 of refe_p6's profile (the head package's anchor, test_p7_head_wiring.py P6_PROFILE_SHA256) and of the P6 `expect_schedule`
  * sha256 of the JSON of `gate_checks.check_clock` on the P6 profile, for a clean build capture, a wrong-count capture and an unregistered-bank argv (the evidence a P6 launch would write, minus nothing)
  * the files P6 resumes from: train.py, model.py, session_reserve.py, p6_pod.sh, p6_prep.py (sha256)
usage: python probe_p6_identity.py <refe dir>"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REFE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REFE))
import gate_checks as C  # noqa: E402
import launch_gate as L  # noqa: E402

P6 = L.PROFILES["refe_p6"]
print("refe_p6 profile sha256        ", hashlib.sha256(repr(sorted(P6.items())).encode()).hexdigest())
print("refe_p6 expect_schedule sha256", hashlib.sha256(repr(sorted(P6["expect_schedule"].items())).encode()).hexdigest(), "| keys", sorted(P6["expect_schedule"]))


def fake_b(n):
    return SimpleNamespace(rc="STOPPED_AT_BUILD", args=SimpleNamespace(epochs=25), steps=10075, samplers=[{"n": 103037}], bank_len={"TargetBank": {"n_tuples": n, "n_scenes": 103037}})


for label, argv, n in (("clean 3-C bank", ["--targets", "/w/data/refe_navtrain/train_grow_cmd_3c"], 167300), ("wrong count", ["--targets", "/w/data/refe_navtrain/train_grow_cmd_3c"], 167301),
                       ("unregistered bank", ["--targets", "/w/data/refe_navtrain/other_bank"], 167300)):
    ev = C.check_clock(argv, fake_b(n), P6)
    print(f"check_clock(refe_p6, {label:18s}) {ev['status']:4s} sha256 {hashlib.sha256(json.dumps(ev, sort_keys=True, default=str).encode()).hexdigest()[:32]} keys {sorted(ev['details'])}")
for f in ("train.py", "model.py", "session_reserve.py", "p6_pod.sh", "p6_prep.py"):
    print(f"{f:20s} sha256", hashlib.sha256((REFE / f).read_bytes()).hexdigest())
