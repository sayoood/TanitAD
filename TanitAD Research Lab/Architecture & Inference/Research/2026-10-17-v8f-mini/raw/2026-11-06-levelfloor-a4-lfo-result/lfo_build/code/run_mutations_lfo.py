"""run_mutations_lfo.py -- the mutation log of tests/test_lfo.py: a CLEAN GREEN run first, then each one-line mutation of a COPY of the LFO tool (asserted to match exactly once) must turn at least one test RED.
The two mutations the A4 card asks for are m1 (delete the single line that assigns the oracle v_hat) and m2 (drop the gt_ok guard); both must fail KA-O3 (the tests named test_ka_o3_*). Per mutation the log records the failing
tests and the pass / fail counts of the KA-O3 tests and of the whole file. Writes logs/MUTATIONS_LFO.json and logs/MUTATIONS_LFO.txt.   usage: python -I run_mutations_lfo.py"""
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
SCR = "F:/TanitAD-scratch/motion_c/lfo_build"
MUT = f"{SCR}/mut"
TREE = "F:/TanitAD-scratch/motion_c/lfo_read/tree"
TOOL_SRC = f"{PK}/tool/v8f_t1_eval_ri_lfo.py"
OPERATOR = f"{TREE}/stack/scripts/level_floor.py"
MUTATIONS = [
    ("m1_delete_the_oracle_assignment", "            v_hat = self.oracle_vhat(len(v_heads))\n", "            pass\n", True),
    ("m2_drop_the_gt_ok_guard", "    ok = np.asarray(gt_ok, bool) & np.isfinite(v)\n", "    ok = np.isfinite(v)\n", True),
    ("m3_knots_1_to_5", "LOGGED_KNOTS = 6 ", "LOGGED_KNOTS = 5 ", True),
    ("m4_oracle_reads_the_next_tick", "gt, ok = gt_in_frame(store.get(clip)[\"pose\"], pose_now, st.j)\n            gts.append(gt)", "gt, ok = gt_in_frame(store.get(clip)[\"pose\"], pose_now, st.j + 1)\n            gts.append(gt)", True),
    ("m5_rollout_does_not_hand_over_the_states", "            planner.states = states\n", "            pass\n", True),
    ("m6_planner_does_not_hand_over_the_states", "                floor.states = plan.states\n", "                pass\n", True),
    ("m7_heads_mode_takes_the_oracle_branch", "        if self.vhat_source == \"logged\":                                                             # LFO, CEILING: the level input is the logged route speed\n", "        if True:\n", False),
    ("m8_vhat_heads_not_kept", "            info[\"vhat_heads\"] = np.asarray(v_heads, np.float64)\n", "            pass\n", False),
    ("m9_no_ceiling_label_in_the_header", "            header[\"label\"] = \"CEILING\"\n", "            pass\n", False),
    ("m10_refusal_without_the_floor_dropped", "        if src_ is not None:\n            raise SystemExit(f\"REFUSED: --floor-vhat-source", "        if False:\n            raise SystemExit(f\"REFUSED: --floor-vhat-source", False),
    ("m11_default_source_is_logged", "        a.floor_vhat_source = \"heads\"\n", "        a.floor_vhat_source = \"logged\"\n", False),
    ("m12_meta_carries_no_label", "**({\"label\": \"CEILING\"} if self.vhat_source == \"logged\" else {})", "", False),
]


def run(tool_path, tag):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", LFO_TOOL=tool_path, TMP=f"{SCR}/pt", TEMP=f"{SCR}/pt", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-rA", "--tb=no", "-p", "no:cacheprovider", "--basetemp", f"{SCR}/pt/mut_{tag}", f"{PK}/tests/test_lfo.py"], env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    passed = sorted(set(re.findall(r"^PASSED [^:]+::(\S+)", out, re.M)))
    failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) [^:]+::(\S+)", out, re.M)))
    ka = [t for t in passed + failed if t.startswith("test_ka_o3")]
    return {"rc": r.returncode, "passed": len(passed), "failed": len(failed), "failed_tests": failed, "ka_o3_total": len(ka), "ka_o3_passed": sum(t in passed for t in ka), "ka_o3_failed": sum(t in failed for t in ka),
            "summary": [l for l in out.splitlines() if re.search(r"\d+ (passed|failed)", l)][-1:] or [out[-200:]]}


def main():
    os.makedirs(MUT, exist_ok=True)
    os.makedirs(f"{SCR}/pt", exist_ok=True)
    src = open(TOOL_SRC, encoding="utf-8", newline="").read()
    op = open(OPERATOR, encoding="utf-8", newline="").read()
    res = {"clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tool": TOOL_SRC, "clean": run(TOOL_SRC, "clean") if False else None, "mutations": {}}
    d0 = f"{MUT}/tool_clean"
    os.makedirs(d0, exist_ok=True)
    open(f"{d0}/v8f_t1_eval_ri.py", "w", encoding="utf-8", newline="\n").write(src)
    open(f"{d0}/level_floor.py", "w", encoding="utf-8", newline="\n").write(op)
    res["clean"] = run(f"{d0}/v8f_t1_eval_ri.py", "clean")
    assert res["clean"]["rc"] == 0 and res["clean"]["failed"] == 0, ("CLEAN RUN NOT GREEN", res["clean"])
    for name, old, new, required in MUTATIONS:
        assert src.count(old) == 1, (name, src.count(old))
        d = f"{MUT}/tool_{name}"
        os.makedirs(d, exist_ok=True)
        open(f"{d}/v8f_t1_eval_ri.py", "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        open(f"{d}/level_floor.py", "w", encoding="utf-8", newline="\n").write(op)
        r = run(f"{d}/v8f_t1_eval_ri.py", name)
        r["RED"] = bool(r["rc"] != 0 and r["failed_tests"])
        r["required_by_the_card"] = bool(name in ("m1_delete_the_oracle_assignment", "m2_drop_the_gt_ok_guard"))
        r["must_fail_ka_o3"] = required
        r["ka_o3_RED"] = bool(r["ka_o3_failed"] > 0)
        res["mutations"][name] = r
    res["all_red"] = all(v["RED"] for v in res["mutations"].values())
    res["card_mutations_fail_ka_o3"] = all(v["ka_o3_RED"] for v in res["mutations"].values() if v["required_by_the_card"])
    res["ka_o3_mutations_all_fail_ka_o3"] = all(v["ka_o3_RED"] for v in res["mutations"].values() if v["must_fail_ka_o3"])
    json.dump(res, open(f"{PK}/logs/MUTATIONS_LFO.json", "w", encoding="utf-8", newline="\n"), indent=1)
    c = res["clean"]
    lines = [f"clean: {c['passed']} passed, {c['failed']} failed (KA-O3 tests: {c['ka_o3_passed']} of {c['ka_o3_total']} pass)"]
    for k, v in res["mutations"].items():
        lines.append(f"{k}: {'RED' if v['RED'] else 'STILL GREEN'}  file {v['passed']} passed / {v['failed']} failed; KA-O3 {v['ka_o3_passed']} pass / {v['ka_o3_failed']} fail of {v['ka_o3_total']}; failed: {', '.join(v['failed_tests'])}")
    lines.append("MUTATIONS ALL RED" if res["all_red"] else "SOME MUTATION STAYED GREEN")
    lines.append("CARD MUTATIONS (m1, m2) FAIL KA-O3" if res["card_mutations_fail_ka_o3"] else "A CARD MUTATION DID NOT FAIL KA-O3")
    open(f"{PK}/logs/MUTATIONS_LFO.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if (res["all_red"] and res["card_mutations_fail_ka_o3"] and res["ka_o3_mutations_all_fail_ka_o3"]) else 2)


if __name__ == "__main__":
    main()
