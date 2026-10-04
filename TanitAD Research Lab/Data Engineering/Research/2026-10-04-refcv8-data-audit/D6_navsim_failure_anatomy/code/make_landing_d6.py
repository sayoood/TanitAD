"""D6 -- (re)write ../../LANDING_READY_D6.txt from the package folder: md5 of every file for git, size+location of the not-for-git ones. Re-run after every change."""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                       # .../D6_navsim_failure_anatomy
ROOT = os.path.dirname(PKG)                       # .../2026-10-04-refcv8-data-audit
PREFIX = "TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/"
NOTE = {
    "RESULT.md": "the deliverable: bottom line, controls, literals, attribution table, costed probes, section 12 (P3 results, registered SPEC + A1, queue state), Appendix A (T1-T10) and B (P3 tables)",
    "SPEC_P1P2.md": "pre-registration of P1 (candidate fan) and P2 (nav-withhold); REGISTERED by the Master Mind sha256 4eaf655f8e000f5ae06a4b0ba93ff05be04a037064d1c16d108e7c11c1f5f4bd, 2026-10-04T12:00:42Z",
    "SPEC_P1P2_A1.md": "amendment A1 (no WTA/r7 heads in the as-launched model: universe = FAN117); REGISTERED sha256 def9b4c24222d47dc49de0174690279c4819a7da1dc896954d7c6b582f1392bb, 12:09:57Z",
}
GIT_DIRS = ("code", "raw")
NOT_GIT_SUFFIX = (".jsonl",)
NOT_GIT_NAMES = {"geom_all.stdout.txt", "rescore_A.stdout.txt", "rescore_B.stdout.txt", "p3_navtest.stdout.txt", "p3_snap.stdout.txt", "p3_speed_s0.stdout.txt",
                 "p3_speed_s1.stdout.txt", "gate.log", "gpu_wait.log", "cpu_chain.log", "launch_pids.json", "chain.log"}
NOT_GIT_NOTE = {
    "d6_scene_table_step30000.csv": "intermediate (part 1); subsumed by d6_scene_table_FULL_step30000.csv",
    "d6_scene_geometry_step30000.csv": "intermediate (part 2); subsumed by the FULL table",
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    git, ng = [], []
    for top in ("RESULT.md", "SPEC_P1P2.md", "SPEC_P1P2_A1.md"):
        git.append((top, NOTE[top]))
    for d in GIT_DIRS:
        for dp, dn, fn in os.walk(os.path.join(PKG, d)):
            dn[:] = [x for x in dn if x != "__pycache__"]
            for f in sorted(fn):
                rel = os.path.relpath(os.path.join(dp, f), PKG).replace("\\", "/")
                if f.endswith(NOT_GIT_SUFFIX) or f in NOT_GIT_NAMES or f in NOT_GIT_NOTE:
                    ng.append((rel, NOT_GIT_NOTE.get(f, "bulk evidence / live log, re-derivable or transient")))
                elif f.endswith(".pyc"):
                    continue
                else:
                    kind = ("analysis / harness code" if rel.startswith("code/") else "aggregate result / input list / registered-hash record")
                    if rel.startswith("code/bridge_fix/"):
                        kind = "bridge COPY (insert-only vs the .base files; base blob shas in BASE_BLOBS.txt)" if not rel.endswith(".base") else "unmodified copy of the live base file (needed by test_default_identical.py)"
                    git.append((rel, kind))
    L = ["## D6", "",
         "LANDING_READY -- refcv8 data audit, stream D6 (NAVSIM FAILURE ANATOMY: refcv7 navhard DAC/NC zeros + P3 oracles + P1/P2 SPEC + bridge export), Data Engineering, 2026-10-04",
         "Hand-over: NOTHING staged or committed by this agent (no git add / commit / push). The Master Mind lands these paths, BY PATH.",
         "Paths are repo-relative to D:/Projects/TanitAD. md5 = md5 of the file bytes as written on D: at hand-over (re-run code/make_landing_d6.py to refresh).", "",
         "md5                                path -- note"]
    for rel, note in git:
        p = os.path.join(PKG, rel)
        L.append(f"{md5(p)}   {PREFIX}D6_navsim_failure_anatomy/{rel} -- {note}")
    L += ["", "Re-run: see RESULT.md section 11 (tanitad venv for parts 1-6 and the analyses; navsim venv C:/Users/Admin/navsim-crun/venv for d6_rescore.py / d6_fan_score.py / d6_p3_oracles.py / d6_fanbench.py; navsim-1.1 tree for d6_p3_navtest.py).", "",
          "Not for git", "-----------"]
    for rel, note in ng:
        p = os.path.join(PKG, rel)
        L.append(f"{os.path.getsize(p):>12,d} B   {PREFIX}D6_navsim_failure_anatomy/{rel} -- {note}  [location: D: dev box, in the package folder]")
    L += ["", "Thor-only state", "---------------", "None. Thor was not touched; no process was started there; no Thor job to end.", "",
          "Dev-box state at hand-over (2026-10-04, local Berlin time)", "---------------------------------------------------------",
          "* All CPU analysis / oracle processes have EXITED. THREE detached processes are intentionally alive, all CPU-only until the battery's lock frees:",
          "  - the GATE (code/run_p1p2_gate.py; pid 51304, python 49820): saw raw/SPEC_A1_REGISTERED.txt at 14:10:21 and started the battery's with_gpu_lock.py (pid 34564), which is WAITING for the dev-box GPU lock",
          "    (raw/gpu_wait.log: holder refcv7-milestone-step50400, acquired 13:20:15). It NEVER touches a held lock; the GPU chain (code/run_p1p2_chain.py) runs only once the wrapper holds the lock. Kill by explicit PID only.",
          "  - the CPU waiter (code/run_p1p2_cpu.py; pid 14864, python 53644): polls raw/p1p2_gpu_done.json, then scores P1 (3 shards) and P2 on the CPU (RAM-gated, <= 3 processes).",
          "* The GPU, devbox_gpu.lock and every other process were not touched by this stream. The live navsim/code files were not edited (code/bridge_fix/ is a copy).", "",
          "Notes for the Master Mind", "-------------------------",
          "* P1/P2 numbers do NOT exist yet. After the chain and the CPU stage finish: run code/d6_p1_analyze.py and code/d6_p2_analyze.py (controls K1-K6 / K7-K8 must PASS before any reading is quoted), then add raw/d6_p1.json, raw/d6_p2.json and the jsonl evidence to this list.",
          "* Registered hashes are in raw/SPEC_SHA256.txt (written by the Master Mind). The SPEC files must not be edited.",
          "* GOALS_AND_CLAIMS / MODEL_REGISTRY were not edited by this stream."]
    open(os.path.join(ROOT, "LANDING_READY_D6.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print(len(git), "git paths;", len(ng), "not-for-git")


if __name__ == "__main__":
    main()
