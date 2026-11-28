"""Mutation runs of ACTLEVEL (control v4): six deliberate regressions, each by an exact string replacement (asserted to match exactly once) in ONE mutation copy of the tree
(work/tree_mut); the two changed files are restored by copying the pristine ones (tree_level) over them after every mutation. The unmutated tree must be green before and after.
Each run is the serial pytest of test_v8f_act_level.py and test_v8f_act_cons.py. Usage: run_mutations.py   ->  MUTATIONS_ACTLEVEL.json"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time

ROOT = "F:/TanitAD-scratch/v8proof/actlevel"
PRISTINE = f"{ROOT}/work/tree_level"
MUT = f"{ROOT}/work/tree_mut"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
FILES = ["stack/tanitad/models/act_cons.py", "stack/scripts/v8f_mini_train.py"]
TESTS = ["stack/tests/test_v8f_act_level.py", "stack/tests/test_v8f_act_cons.py"]

MUTATIONS = [
    ("m1_level_add_removed", FILES[1], 'T["total"] = T["total"] + a.w_act_level * ac_r["level"]', 'T["total"] = T["total"]'),
    ("m2_reference_read_detached", FILES[0], "g_lvl = _gap_m(head, params, ref[:, :, lo:hi])", "g_lvl = _gap_m(head, params, ref[:, :, lo:hi].detach())"),
    ("m3_label_at_frame_s_plus_k_minus_1", FILES[0], "            f = s + k\n", "            f = s + k - 1\n"),
    ("m4_head_params_attached_in_level_read", FILES[0], "g_lvl = _gap_m(head, params, ref[:, :, lo:hi])", "g_lvl = _gap_m(head, {n: p for n, p in head.named_parameters()}, ref[:, :, lo:hi])"),
    ("m5_valid_label_mask_dropped", FILES[0], "            ok = (pres > 0.5) & torch.isfinite(gap) & (gap >= GAP_LO_M) & (gap <= GAP_HI_M) & batch[\"frame_ok\"][:, f].bool()\n",
     "            ok = torch.ones_like(gap, dtype=torch.bool)\n"),
    ("m6_refusal_V_with_W0_removed", FILES[1], "if a.w_act_level > 0.0 and a.w_act_cons == 0.0:", "if False:"),
]


def free_commit_mb() -> int:
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", "$o = Get-CimInstance Win32_OperatingSystem; [int]($o.FreeVirtualMemory / 1024)"], capture_output=True, text=True)
    return int(r.stdout.strip())


def wait_memory():
    t0 = time.time()
    while free_commit_mb() < 5632:
        if time.time() - t0 > 1200:
            raise SystemExit("commit memory below 5632 MB for 20 minutes")
        time.sleep(60)


def pytest_run(tag: str) -> dict:
    wait_memory()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               TMP=f"{ROOT}/work/tmp", TEMP=f"{ROOT}/work/tmp", PYTHONPATH=";".join([f"{MUT}/stack", f"{MUT}/taniteval", f"{MUT}/stack/scripts"]))
    chk = subprocess.run([PY, "-c", "import tanitad,os;f=os.path.abspath(tanitad.__file__).lower().replace(chr(92),'/');assert f.startswith('f:/tanitad-scratch/v8proof/actlevel/work/tree_mut'),f"], env=env, capture_output=True, text=True)
    assert chk.returncode == 0, chk.stderr[-300:]
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS, f"--basetemp={ROOT}/work/pt"], cwd=MUT, env=env, capture_output=True, text=True, encoding="utf-8")
    out = (r.stdout or "") + (r.stderr or "")
    failed = sorted(set(re.findall(r"^FAILED (\S+)", out, re.M)))
    m = re.search(r"(\d+) passed", out)
    f = re.search(r"(\d+) failed", out)
    return {"tag": tag, "rc": r.returncode, "passed": int(m.group(1)) if m else 0, "failed": int(f.group(1)) if f else 0, "failed_tests": failed, "tail": out[-300:].replace("\n", " | ")}


def restore():
    for rel in FILES:
        shutil.copyfile(f"{PRISTINE}/{rel}", f"{MUT}/{rel}")


def main():
    if not os.path.isdir(MUT):
        for sub in ("stack", "taniteval"):
            shutil.copytree(f"{PRISTINE}/{sub}", f"{MUT}/{sub}", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    restore()
    res = {"before": pytest_run("unmutated_before"), "mutations": []}
    for name, rel, old, new in MUTATIONS:
        src = open(f"{MUT}/{rel}", "rb").read().decode("utf-8")
        assert src.count(old) == 1, (name, src.count(old))
        open(f"{MUT}/{rel}", "wb").write(src.replace(old, new).encode("utf-8"))
        try:
            r = pytest_run(name)
        finally:
            restore()
        r["file"] = rel
        r["red"] = bool(r["failed"] > 0)
        res["mutations"].append(r)
        print(name, "RED" if r["red"] else "NOT RED", r["failed_tests"], flush=True)
    res["after"] = pytest_run("unmutated_after")
    for rel in FILES:
        assert open(f"{PRISTINE}/{rel}", "rb").read() == open(f"{MUT}/{rel}", "rb").read()
    res["red_count"] = sum(1 for m in res["mutations"] if m["red"])
    res["total"] = len(res["mutations"])
    res["unmutated_green_before_and_after"] = bool(res["before"]["rc"] == 0 and res["after"]["rc"] == 0)
    res["PASS"] = bool(res["red_count"] == res["total"] and res["unmutated_green_before_and_after"])
    with open(f"{ROOT}/MUTATIONS_ACTLEVEL.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print("MUTATIONS", res["red_count"], "/", res["total"], "green before/after", res["unmutated_green_before_and_after"])


if __name__ == "__main__":
    main()
