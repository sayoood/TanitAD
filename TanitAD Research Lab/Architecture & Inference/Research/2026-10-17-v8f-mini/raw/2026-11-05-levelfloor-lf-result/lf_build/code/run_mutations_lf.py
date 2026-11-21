"""run_mutations_lf.py -- the mutation log of tests/test_level_floor.py: a CLEAN GREEN run first, then each mutation of level_floor.py (OPERATOR mutations m*) and of the patched tool's floor branch (TOOL
mutations t*: F before S, F skipped in the early-return branch, F after the projection, plan_raw not kept, v0 read from the wrong history row) -- a one-line edit of a COPY, asserted to match exactly once --
must turn at least one test RED.  Writes logs/MUTATIONS_LF.json and logs/MUTATIONS_LF.txt.  usage: python -I run_mutations_lf.py"""
import json
import os
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SCR = "F:/TanitAD-scratch/motion_c/lf_read"
MUT = f"{SCR}/mut"
MUTATIONS = [
    ("m1_delete_the_max", "    s_new = np.maximum(s_k, s_f)\n", "    s_new = s_f\n"),
    ("m2_drop_the_lead_cap", "        capped = np.where(lead, np.minimum(v_f, cap_speed), v_f)\n", "        capped = v_f\n"),
    ("m3_ignore_kappa", "        v_f = kappa * v_hat\n", "        v_f = v_hat\n"),
    ("m4_drop_the_3ms_band", "        band = np.isfinite(v_hat) & (v_hat >= V_BAND_MIN)\n", "        band = np.isfinite(v_hat)\n"),
    ("m5_extrapolate_along_x", "    return seg[j] / ln[j], bool(j != len(ln) - 1)\n", "    return FWD.copy(), False\n"),
    ("m6_ignore_v0", "    v0p = np.maximum(np.nan_to_num(np.atleast_1d(np.asarray(v0, np.float64)), nan=0.0, posinf=0.0, neginf=0.0), 0.0)[:, None]\n", "    v0p = np.zeros_like(v_f)\n"),
    ("m7_instant_ramp", "    t_star = np.maximum(v_f - v0p, 0.0) / a_f\n", "    t_star = np.zeros_like(v_f)\n"),
    ("m8_presence_strict", "(pi >= PI_MIN)", "(pi > PI_MIN)"),
]
TOOL_MUTATIONS = [
    ("t1_F_before_S", "            plan_f, plan.floor_info = floor(batch, plan_s, v0)\n", "            plan_f, plan.floor_info = floor(batch, raw, v0)\n"),
    ("t2_early_return_skips_F", "                return feasible_plan(plan_f, feasible, v0=v0)\n            return plan_f\n", "                return feasible_plan(plan_f, feasible, v0=v0)\n            return plan_s\n"),
    ("t3_F_after_the_projection", "                return feasible_plan(plan_f, feasible, v0=v0)\n", "                return floor(batch, feasible_plan(plan_s, feasible, v0=v0), v0)[0]\n"),
    ("t4_plan_raw_not_kept", "            plan.plan_raw = raw\n            plan_s = smooth_plan(raw,", "            plan_s = smooth_plan(raw,"),
    ("t5_v0_from_the_wrong_row", "            v0 = batch[\"p10f\"][:, -1, 0].double().cpu().numpy()          # the measured ego speed (== TickRecorder's ``v0``)\n", "            v0 = batch[\"p10f\"][:, -2, 0].double().cpu().numpy()\n"),
]


def run(op_path, tag, tool_path=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", LF_OPERATOR=op_path, TMP=f"{SCR}/pt", TEMP=f"{SCR}/pt", PYTHONDONTWRITEBYTECODE="1")
    if tool_path:
        env["LF_TOOL"] = tool_path
    r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", f"{SCR}/pt/mut_{tag}", f"{PK}/tests/test_level_floor.py"], env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    failed = sorted(set(re.findall(r"FAILED [^:]+::(\S+)", out)))
    tail = [l for l in out.splitlines() if re.search(r"\d+ (passed|failed)", l)]
    return {"rc": r.returncode, "failed_tests": failed, "summary": tail[-1] if tail else out[-300:]}


def main():
    os.makedirs(MUT, exist_ok=True)
    os.makedirs(f"{SCR}/pt", exist_ok=True)
    src = open(f"{PK}/code/level_floor.py", encoding="utf-8", newline="").read()
    res = {"clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "operator": f"{PK}/code/level_floor.py", "tool": f"{PK}/tool/v8f_t1_eval_ri_lf.py", "clean": run(f"{PK}/code/level_floor.py", "clean"), "mutations": {}}
    assert res["clean"]["rc"] == 0, ("CLEAN RUN NOT GREEN", res["clean"])
    for name, old, new in MUTATIONS:
        assert src.count(old) == 1, (name, src.count(old))
        p = f"{MUT}/level_floor_{name}.py"
        open(p, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        r = run(p, name)
        r["RED"] = bool(r["rc"] != 0 and r["failed_tests"])
        res["mutations"][name] = r
    tool_src = open(f"{PK}/tool/v8f_t1_eval_ri_lf.py", encoding="utf-8", newline="").read()
    for name, old, new in TOOL_MUTATIONS:
        assert tool_src.count(old) == 1, (name, tool_src.count(old))
        d = f"{MUT}/tool_{name}"
        os.makedirs(d, exist_ok=True)
        open(f"{d}/v8f_t1_eval_ri.py", "w", encoding="utf-8", newline="\n").write(tool_src.replace(old, new))
        open(f"{d}/level_floor.py", "w", encoding="utf-8", newline="\n").write(src)
        r = run(f"{PK}/code/level_floor.py", name, f"{d}/v8f_t1_eval_ri.py")
        r["RED"] = bool(r["rc"] != 0 and r["failed_tests"])
        res["mutations"][name] = r
    res["all_red"] = all(v["RED"] for v in res["mutations"].values())
    json.dump(res, open(f"{PK}/logs/MUTATIONS_LF.json", "w", encoding="utf-8", newline="\n"), indent=1)
    lines = [f"clean: {res['clean']['summary']}"] + [f"{k}: {'RED' if v['RED'] else 'STILL GREEN'}  {v['summary']}  failed: {', '.join(v['failed_tests'])}" for k, v in res["mutations"].items()]
    lines.append("ALL MUTATIONS RED" if res["all_red"] else "SOME MUTATION STAYED GREEN")
    open(f"{PK}/logs/MUTATIONS_LF.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if res["all_red"] else 2)


if __name__ == "__main__":
    main()
