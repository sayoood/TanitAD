#!/usr/bin/env python3
"""P7-READ-YAW: stage2_sft.py's seam command under --ego-yaw-table (CPU, no GPU, no navtest read). Known-value controls + RED arms.
   python eval/test_p7_read_yaw.py  ->  ZZP7YAW PASS n/n"""
import gzip, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "refe"))
import test_p6_read_gaps as T  # noqa: E402  (its stage2 capture harness: every side effect stubbed)

res = []


def check(name):
    def d(fn):
        try:
            fn(); res.append((True, name)); print("PASS", name)
        except Exception as e:  # noqa: BLE001
            res.append((False, name)); print("FAIL", name, "<--", type(e).__name__, str(e)[:240])
        return fn
    return d


def _ckpt(td, yaw: bool):
    import torch
    p = Path(td) / ("yaw.pt" if yaw else "noyaw.pt")
    meta = {"ego_yaw": "nuplan_angular_rate_z", "ego_yaw_scale": 10.0, "ego_yaw_slot": 4} if yaw else {}
    torch.save({"model": {}, "meta": meta}, p)
    return str(p)


def _table(td, payload: bytes):
    p = Path(td) / "t.json.gz"
    p.write_bytes(payload)
    return str(p)


def _run(argv, ckpt):
    S2 = T._load_stage2(HERE / "stage2_sft.py", "stage2_sft_yaw_%d" % len(res))
    orig = T.HP.require
    T.HP.require = lambda root, pins=None, caller=None: orig(root, pins, T._caller_ok_path())
    old_argv = list(sys.argv)
    try:
        cap = None
        # run_stage2_capture hard-codes the ckpt argument; pass ours through argv (argparse: the last --ckpt wins)
        cap = T.run_stage2_capture(S2, ["--ckpt", ckpt, *argv])
    finally:
        T.HP.require = orig
        sys.argv = old_argv
    return cap, S2


with tempfile.TemporaryDirectory() as td:
    noyaw, yaw = _ckpt(td, False), _ckpt(td, True)

    @check("Y1 no flag, non-yaw checkpoint: the seam command is the OLD one (no launcher)")
    def _():
        cap, S2 = _run([], noyaw)
        cmd = [c for c in cap["calls"] if "refe_navtest_seam.py" in c["cmd"]][0]["cmd"]
        assert cmd[1] == "refe_navtest_seam.py" and "ego_yaw.py" not in " ".join(cmd), cmd

    @check("Y2 a yaw-trained checkpoint WITHOUT the table is REFUSED before the lock (SystemExit, no seam call)")
    def _():
        try:
            _run([], yaw)
        except SystemExit as e:
            assert "REFUSING" in str(e) and "ego-yaw-table" in str(e), str(e)
            return
        raise AssertionError("a yaw checkpoint with no table ran")

    @check("Y3 a table that is not the registered navtest table (sha256 prefix) is REFUSED")
    def _():
        try:
            _run(["--ego-yaw-table", _table(td, b"not the table")], yaw)
        except SystemExit as e:
            assert "not the registered navtest table" in str(e), str(e)
            return
        raise AssertionError("a wrong table ran")

    @check("Y4 the registered table: the seam runs under ego_yaw.py with the SAME seam arguments, in order, and result.json records the table sha256")
    def _():
        payload = b"stand-in table bytes"
        pref = hashlib.sha256(payload).hexdigest()[:8]
        S2 = T._load_stage2(HERE / "stage2_sft.py", "stage2_sft_yaw_ok")
        S2.NAVTEST_YAW_SHA_PREFIX = pref                      # the stub table's own prefix stands in for f5ebe24f (the real table is not read here)
        orig = T.HP.require
        T.HP.require = lambda root, pins=None, caller=None: orig(root, pins, T._caller_ok_path())
        try:
            t = _table(td, payload)
            cap = T.run_stage2_capture(S2, ["--ckpt", yaw, "--ego-yaw-table", t])
        finally:
            T.HP.require = orig
        cmd = [c for c in cap["calls"] if "refe_navtest_seam.py" in c["cmd"]][0]["cmd"]
        assert cmd[1].replace("\\", "/").endswith("refe/ego_yaw.py") and cmd[2:5] == ["--table", t, "--"] and cmd[5] == "refe_navtest_seam.py", cmd
        assert cmd[6:8] == ["--ckpt", yaw], cmd
        assert cap["result"]["ego_yaw"]["sha256"].startswith(pref) and cap["result"]["ego_yaw"]["checkpoint_wants_yaw"] is True

    @check("Y5 RED: the pre-patch stage2_sft.py (no yaw flag) fails Y2 (it would have read the yaw checkpoint with a yaw of 0.0)")
    def _():
        src = (HERE / "stage2_sft.py").read_text(encoding="utf-8")
        assert src.count("if EY.wants_yaw(meta) and not a.ego_yaw_table:") == 1
        mut = src.replace("if EY.wants_yaw(meta) and not a.ego_yaw_table:", "if False:")
        p = Path(td) / "stage2_sft_mut.py"
        p.write_text(mut, encoding="utf-8")
        S2 = T._load_stage2(p, "stage2_sft_yaw_mut")
        orig = T.HP.require
        T.HP.require = lambda root, pins=None, caller=None: orig(root, pins, T._caller_ok_path())
        try:
            T.run_stage2_capture(S2, ["--ckpt", yaw])
            ran = True
        except SystemExit:
            ran = False
        finally:
            T.HP.require = orig
        assert ran, "the mutant was still refused: Y2 would not detect a dropped guard"

n = sum(1 for ok, _ in res if ok)
print(f"ZZP7YAW {'PASS' if n == len(res) else 'FAIL'} {n}/{len(res)}")
sys.exit(0 if n == len(res) else 1)
