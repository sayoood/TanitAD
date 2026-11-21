"""REWARD-2 chain (detached). Markers in F:/tanitad_scratch/mgym6/progress.txt; the LAST line is ALL_DONE or an ABORT line.
usage: chain_v6.py [--dry] [--workers N]      (N <= 4; --dry uses F:/tanitad_scratch/mgym6dry and STOPS BEFORE THE DRAW)
Order (PREREG_REWARD_2 + brief):
 PREREG_OK (sha256 of PREREG_REWARD_2.md == PREREG_REWARD_2.sha256)
 -> PIN_OK (every pinned file equals CODE_PIN.json, no unpinned code file; the chain REFUSES on a mismatch)
 -> TESTS_GREEN -> EXCL (section 0 exclusion list frozen with its own sha256; does not touch the seed)
 -> R123_PASS (R1, R2, R3 on used states 3, 11, 17) -> R4_PASS (R4 + G-REPRO on the 39 used eligible states)
 -> GATE_WRITTEN -> DRAWN (seed 20261111, once) -> STATES_PINNED
 -> MAIN_DONE (20 full-arm states) -> LADDER_lo_hi_DONE (+ blocks 160:200, 200:240 only while < 5 eligible, never a new seed) -> DET_DONE -> TTCOFF_DONE -> SUMMARY_DONE -> ALL_DONE.
Any red test, R or G-REPRO failure => a FAILED marker file + `ABORT FAILED ...` and exit; the seed is then UNDRAWN. A dry chain ends with DRY_STOP_BEFORE_DRAW and can never draw
(MGYM6_DRY is set for the process and every child except the draw stage, and draw_gate refuses a guarded seed in that mode).
CPU only (CUDA_VISIBLE_DEVICES=-1), <= 4 workers, a new state / test group starts only while available RAM >= 7 GiB (ramguard; every wait is logged)."""
import hashlib, json, os, shutil, subprocess, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ramguard

PY = r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
V6 = os.path.dirname(os.path.abspath(__file__))              # .../pilot_v6/scorer_gym
P6 = os.path.dirname(V6)
FLOOR, ARC = "0.20", "0.05"
STAGES_REAL = ["PREREG_OK", "PIN_OK", "TESTS", "EXCL", "R123", "R4", "GATE", "DRAW", "STATES_PINNED", "MAIN", "LADDER", "DET", "TTCOFF", "SUMMARY", "COPY", "ALL_DONE"]
STAGES_DRY = ["PREREG_OK", "PIN_OK", "TESTS", "EXCL", "R123", "R4", "DRY_STOP_BEFORE_DRAW"]


def plan(dry):
    """the stage order; a dry chain has no DRAW (tests mutate this)"""
    return list(STAGES_DRY if dry else STAGES_REAL)


def test_groups():
    """(script, [test names]) per test process; the chain runs them <= workers at a time"""
    T4, T5, T6 = V6 + r"\tests\test_v4.py", V6 + r"\tests\test_v5.py", V6 + r"\tests\test_v6.py"
    TC = P6 + r"\tests\test_curve_speed_v6.py"
    groups = [
        (T4, ["test_fix1_oracle_lexicographic_and_regression_arm", "test_fix1_control_identical_when_argmax_arm_is_clean", "test_fix4_all_tie_cell_label_free_and_regression_arm",
              "test_fix6_ladder_known_function_and_regression_arms", "test_v4_state_selection_seeded_disjoint_navtest_free"]),
        (T4, ["test_fix3_map_prescreen_flags_main10_and_regression_arm", "test_fix3_prescreen_does_not_flag_clean_state"]),
        (T4, ["test_fix2_ttc_corridor_clears_stall15_false_alarms_and_regression_arm", "test_fix2_corridor_keeps_stall10_and_stall7_collisions_flagged"]),
        (T4, ["test_control_fixes_off_reproduces_pilot_v2_returns", "test_fix6_ladder_policy_reproduces_pilot_v3_q3_when_fixes_off", "test_v4_deterministic_same_process_fixes_on"]),
        (T5, ["test_card_constants", "test_qfree_floor_value_and_regression_arm", "test_eligibility_rule_and_regression_arms", "test_b3_verdict_and_regression_arms",
              "test_block_extension_and_regression_arms"]),
        (T5, ["test_floor_reachable_from_flag_to_crawl_return_and_regression_arm"]),
        (T5, ["test_selection_v5_structure_with_dummy_seed"]),
        (TC, ["fix_on_source_mutant_red", "off_identical", "off_identical_mutant_red", "off_reproduces_stored", "fix_on", "fix_on_mutant_red"]),
        (TC, ["reach_off_matches_stored", "reach_on", "reach_mutant_red"]),
        (T6, ["test_card_constants_v6", "test_draw_gate_dry_and_mutants", "test_draw_v6_and_chain_cannot_draw_in_dry", "test_ramguard_and_mutants", "test_presim_detector_and_mutants", "test_free_road_class_and_mutants",
              "test_kinematic_crawl_share_and_mutants", "test_b3f_verdict_and_mutants", "test_pin_check_and_mutant", "test_flag_to_evaluator_and_run_meta", "test_chain_test_groups_name_real_tests"]),
        (T6, ["test_r_arms_judges_and_mutants"]),
        (T6, ["test_selection_v6_capacity_with_dummy_seed"]),
    ]
    return groups


class Chain:
    def __init__(self, dry, workers):
        assert 1 <= workers <= ramguard.MAX_WORKERS
        self.dry, self.workers = dry, workers
        self.W = r"F:\tanitad_scratch\mgym6dry" if dry else r"F:\tanitad_scratch\mgym6"
        self.PROG = self.W + r"\progress.txt"
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", MGYM6_DRY="1")      # test mode for everything except the draw stage
        for k in ("MGYM6_DRAW_STAGE",):
            self.env.pop(k, None)
        if dry:
            os.environ["MGYM6_DRY"] = "1"
            self.env["MGYM6_DRY_CHAIN"] = "1"
            os.environ["MGYM6_DRY_CHAIN"] = "1"
        else:
            for k in ("MGYM6_MIN_RAM_GIB", "MGYM6_DRY_CHAIN"):
                self.env.pop(k, None)
                os.environ.pop(k, None)

    # ------------------------------------------------------------------ helpers
    def mark(self, s):
        with open(self.PROG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}\n")

    def run(self, args, out, env=None):
        return subprocess.Popen([PY, "-I"] + args, stdout=open(out, "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env or self.env)

    def abort(self, why):
        self.mark("ABORT " + why)
        sys.exit(1)

    def fail(self, which, why):
        """R / G-REPRO / test failure: FAILED marker, seed stays undrawn"""
        open(self.W + r"\FAILED", "w", encoding="utf-8").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {which}: {why}\n")
        if not self.dry:
            os.makedirs(os.path.join(P6, "raw", "reward2"), exist_ok=True)
            shutil.copy(self.W + r"\FAILED", os.path.join(P6, "raw", "reward2", "FAILED"))
        self.abort(f"FAILED {which}: {why} (seed 20261111 UNDRAWN)")

    def stage(self, args, out, name, env=None):
        p = self.run(args, out, env)
        p.wait()
        if p.returncode != 0:
            self.abort(f"{name} rc={p.returncode} (see {out})")
        self.mark(f"{name} rc=0")

    def gate_ram(self, label):
        ramguard.wait_for_ram(label, self.mark)

    # ------------------------------------------------------------------ stages
    def s_prereg(self):
        want = open(os.path.join(P6, "PREREG_REWARD_2.sha256"), encoding="utf-8").read().split()[0]
        got = sha(os.path.join(P6, "PREREG_REWARD_2.md"))
        if got != want:
            self.abort(f"PREREG_SHA_MISMATCH file={got} frozen={want}")
        self.mark("PREREG_OK sha=" + got)

    def s_pin(self):
        bad = verify_pin(P6)
        if bad:
            self.abort("PIN_MISMATCH " + "; ".join(bad[:8]) + (f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""))
        self.mark("PIN_OK sha(CODE_PIN.json)=" + sha(os.path.join(P6, "CODE_PIN.json")))

    def s_tests(self, groups=None):
        groups = groups if groups is not None else test_groups()
        pending, running, outs = list(enumerate(groups)), [], {}
        while pending or running:
            if pending and len(running) < self.workers:
                i, (f, names) = pending[0]
                self.gate_ram(f"test group {i}")
                pending.pop(0)
                outs[i] = self.W + rf"\tests_v6_g{i}.out"
                running.append((self.run([f] + names, outs[i]), i))
                time.sleep(15)
                continue
            time.sleep(2)
            running = [(p, i) for p, i in running if p.poll() is None]
        def green(i):
            txt = open(outs[i], errors="ignore", encoding="utf-8").read()
            n_pass = sum(1 for l in txt.splitlines() if l.startswith("PASS "))
            return "TESTS_DONE fails=0" in txt and n_pass == len(groups[i][1])        # a misspelled test name would otherwise be skipped silently
        bad = [i for i in outs if not green(i)]
        if bad:
            self.fail("TESTS", f"groups {bad} not green (see {self.W}\\tests_v6_g*.out)")
        self.mark(f"TESTS_GREEN groups={len(groups)}")

    def s_excl(self):
        self.stage([V6 + r"\draw_v6.py", "freeze"], self.W + r"\excl.out", "EXCL_FROZEN")
        self.mark("EXCL_SHA " + sha(os.path.join(P6, "PREREG_REWARD_2_EXCLUDE.json")))

    def s_r123(self):
        self.gate_ram("R1-R3")
        p = self.run([V6 + r"\grepro_v6.py", self.W + r"\grepro_v6.json", "--workers", str(min(self.workers, 3))], self.W + r"\grepro_v6.out")
        p.wait()
        self.copy_raw(["grepro_v6.json", "grepro_v6.out"])
        if p.returncode != 0:
            self.fail("R1-R3", f"rc={p.returncode} (see raw/reward2/grepro_v6.out)")
        self.mark("R123_PASS")

    def s_r4(self):
        self.gate_ram("R4")
        p = self.run([V6 + r"\r4_v6.py", self.W + r"\r4_v6.json", "--workers", str(self.workers)], self.W + r"\r4_v6.out")
        p.wait()
        self.copy_raw(["r4_v6.json", "r4_v6.out"])
        if p.returncode != 0:
            self.fail("R4/G-REPRO", f"rc={p.returncode} (see raw/reward2/r4_v6.out)")
        self.mark("R4_PASS")

    def copy_raw(self, files):
        if self.dry:
            return
        os.makedirs(os.path.join(P6, "raw", "reward2"), exist_ok=True)
        for f in files:
            if os.path.exists(self.W + "\\" + f):
                shutil.copy(self.W + "\\" + f, os.path.join(P6, "raw", "reward2", f))

    def s_gate(self):
        assert not self.dry
        g1 = json.load(open(self.W + r"\grepro_v6.json", encoding="utf-8"))
        g4 = json.load(open(self.W + r"\r4_v6.json", encoding="utf-8"))
        gate = dict(r_all_pass=bool(g1["PASS"] and g4["PASS"]), R1=bool(g1["R1"]), R2=bool(g1["R2"]), R3=bool(g1["R3"]["known"]), R4=bool(g4["R4"]), G_REPRO=bool(g4["G_REPRO"]),
                    prereg_sha=sha(os.path.join(P6, "PREREG_REWARD_2.md")), code_pin_sha=sha(os.path.join(P6, "CODE_PIN.json")),
                    exclusion_sha=sha(os.path.join(P6, "PREREG_REWARD_2_EXCLUDE.json")), written=time.strftime("%Y-%m-%d %H:%M:%S"))
        if not gate["r_all_pass"]:
            self.fail("GATE", "R1-R4/G-REPRO not all recorded PASS")
        json.dump(gate, open(self.W + r"\DRAW_GATE.json", "w", encoding="utf-8"), indent=1)
        self.mark("GATE_WRITTEN")

    def s_draw(self):
        assert not self.dry, "a dry chain can never draw"
        states = os.path.join(P6, "PREREG_REWARD_2_STATES.json")
        if os.path.exists(states):
            self.mark("DRAW_REUSED (list already drawn and pinned; never redrawn)")
        else:
            env = {k: v for k, v in self.env.items() if k not in ("MGYM6_DRY", "MGYM6_DRY_CHAIN")}
            env["MGYM6_DRAW_STAGE"] = "1"
            self.stage([V6 + r"\draw_v6.py", "draw"], self.W + r"\draw_v6.out", "DRAWN", env)
        want = open(states.replace(".json", ".sha256"), encoding="utf-8").read().split()[0]
        if sha(states) != want:
            self.abort("STATES_SHA_MISMATCH")
        self.mark("STATES_PINNED sha=" + want)

    def s_runs(self):
        W, A = self.W, ["--floor", FLOOR, "--curve-min-seg-m", ARC, "--workers", str(self.workers)]
        if not os.path.exists(W + r"\free_presim.json"):       # AMENDMENT 1: the F class, written before the first rollout
            self.stage([V6 + r"\presim_v6.py", W + r"\free_presim.json", "fresh", "--workers", str(self.workers)], W + r"\presim.log", "FREE_PRESIM_DONE")
        self.stage([V6 + r"\run_v6.py", "main"] + A + ["--out", W + r"\run_main"], W + r"\run_main.log", "MAIN_DONE")
        lad, lo, hi = [], 20, 160
        while True:
            d = W + rf"\ladder_{lo}_{hi}"
            self.stage([V6 + r"\run_v6.py", "ladder"] + A + ["--range", f"{lo}:{hi}", "--out", d], d + ".log", f"LADDER_{lo}_{hi}_DONE")
            lad.append(d)
            q = W + r"\next_block.out"
            self.stage([V6 + r"\next_block_v6.py", W + r"\run_main"] + lad, q, "NEXT_BLOCK_CHECK")
            info = json.loads([l for l in open(q, encoding="utf-8") if l.startswith("JSON")][0][4:])
            self.mark(f"B3_SO_FAR {json.dumps(info)}")
            if info["next_block"] is None:
                break
            lo, hi = info["next_block"]
        self.lad = lad
        d = []
        for i in (1, 2):
            self.gate_ram(f"determinism child {i}")
            d.append(self.run([V6 + r"\determinism_child_v6.py", "0", FLOOR, ARC], W + rf"\det{i}.txt"))
        for q_ in d:
            q_.wait()
        self.mark("DET_DONE")
        self.stage([V6 + r"\run_v6.py", "ttcoff"] + A + ["--out", W + r"\run_ttcoff"], W + r"\run_ttcoff.log", "TTCOFF_DONE")
        self.stage([V6 + r"\summarize_v6.py", W + r"\run_main", W + r"\run_ttcoff", W + r"\results_v6.json", W + r"\det1.txt", W + r"\det2.txt"] + lad, W + r"\summarize_v6.out", "SUMMARY_DONE")

    def s_copy(self):
        dst = os.path.join(P6, "raw", "reward2")
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(self.W)):
            p = self.W + "\\" + f
            if os.path.isfile(p) and f != "DRAW_GATE.json" and os.path.getsize(p) < 5_000_000:
                shutil.copy(p, os.path.join(dst, f))
        for d in sorted(os.listdir(self.W)):
            p = self.W + "\\" + d
            if os.path.isdir(p):
                os.makedirs(os.path.join(dst, d), exist_ok=True)
                for f in ("run_meta.json", "errors.jsonl", "states.jsonl.gz"):
                    q = p + "\\" + f
                    if os.path.exists(q) and os.path.getsize(q) < 5_000_000:
                        shutil.copy(q, os.path.join(dst, d, f))
        shutil.copy(os.path.join(P6, "PREREG_REWARD_2_EXCLUDE.json"), dst)
        self.mark("COPIED raw/reward2")

    # ------------------------------------------------------------------ main
    def main(self):
        os.makedirs(self.W, exist_ok=True)
        if os.path.exists(self.W + r"\FAILED"):
            self.abort("A_FAILED_MARKER_EXISTS (a previous R/G-REPRO/test failure; the Master Mind decides before any relaunch)")
        open(self.W + r"\chain.pid", "w", encoding="utf-8").write(str(os.getpid()))
        self.mark(f"CHAIN_START pid={os.getpid()} dry={self.dry} workers={self.workers} curve_min_seg_m={ARC} floor={FLOOR} min_ram_gib={ramguard.min_gib_now()} plan={plan(self.dry)}")
        for st in plan(self.dry):
            if st == "PREREG_OK": self.s_prereg()
            elif st == "PIN_OK": self.s_pin()
            elif st == "TESTS": self.s_tests()
            elif st == "EXCL": self.s_excl()
            elif st == "R123": self.s_r123()
            elif st == "R4": self.s_r4()
            elif st == "GATE": self.s_gate()
            elif st == "DRAW": self.s_draw()
            elif st == "STATES_PINNED": pass
            elif st in ("MAIN", "LADDER", "DET", "TTCOFF", "SUMMARY"):
                if st == "MAIN":
                    self.s_runs()
            elif st == "COPY": self.s_copy()
            elif st == "ALL_DONE": self.mark("ALL_DONE")
            elif st == "DRY_STOP_BEFORE_DRAW": self.mark("DRY_STOP_BEFORE_DRAW (seed 20261111 untouched; ALL_DONE is not written in a dry chain)")
            else: raise AssertionError(st)


def sha(f):
    return hashlib.sha256(open(f, "rb").read()).hexdigest()


def pinned_code_files(p6):
    """every code file the chain runs, relative to pilot_v6 with forward slashes (the set CODE_PIN.json must cover)"""
    out = []
    for sub in ("scorer_gym", "scorer_gym/tests", "tests"):
        d = os.path.join(p6, *sub.split("/"))
        if os.path.isdir(d):
            out += [f"{sub}/{f}" for f in sorted(os.listdir(d)) if f.endswith(".py")]
    return out


def verify_pin(p6, pin_path=None):
    """returns the list of mismatches (empty = OK): a pinned file with a different sha256 or missing, or a code file the chain could run that is not pinned"""
    pin = json.load(open(pin_path or os.path.join(p6, "CODE_PIN.json"), encoding="utf-8"))
    bad = []
    for rel, want in pin["files"].items():
        f = os.path.normpath(os.path.join(p6, rel))
        if not os.path.exists(f):
            bad.append(f"MISSING {rel}")
        elif sha(f) != want:
            bad.append(f"SHA {rel}")
    for rel in pinned_code_files(p6):
        if rel not in pin["files"]:
            bad.append(f"UNPINNED {rel}")
    return bad


if __name__ == "__main__":
    ap_dry = "--dry" in sys.argv
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    c = Chain(ap_dry, workers)
    try:
        c.main()
    except SystemExit:
        raise
    except Exception:
        c.mark("ABORT exception " + traceback.format_exc().replace("\n", " | ")[-600:])
        sys.exit(2)
