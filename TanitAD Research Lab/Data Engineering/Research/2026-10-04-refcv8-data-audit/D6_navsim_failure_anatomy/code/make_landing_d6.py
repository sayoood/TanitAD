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
    "SPEC_P1P2_A2_P1PRIME.md": "amendment A2 (P1' on step 50,400, K4s state-level control, bar 0.10); REGISTERED sha256 0ccd6c5aa4de3d56abd0c30907da9fbb509dcd347d209259aece891fe7f64705, 2026-10-04T15:14:40Z",
    "SPEC_P1P2_A1.md": "amendment A1 (no WTA/r7 heads in the as-launched model: universe = FAN117); REGISTERED sha256 def9b4c24222d47dc49de0174690279c4819a7da1dc896954d7c6b582f1392bb, 12:09:57Z",
}
GIT_DIRS = ("code", "raw")
NOT_GIT_SUFFIX = (".jsonl",)
NOT_GIT_PREFIX = ("cpu_",)
NOT_GIT_NAMES = {"d6_p1_WITHHELD_controls_failed.json", "gate_p1x.log", "gpu_wait_p1x.log", "gpu_wait_p2.log", "chain_P1.log", "chain_P2.log", "chain_K7.log", "chain_P2_run1_FOREIGN_COMPUTE_ABORT.log", "chain_K7_run1_FOREIGN_COMPUTE_ABORT.log", "launch_pids_p2.json", "launch_pids_p1x.json", "gpu_rec.json", "gpu_rec_p2.json", "geom_all.stdout.txt", "rescore_A.stdout.txt", "rescore_B.stdout.txt", "p3_navtest.stdout.txt", "p3_snap.stdout.txt", "p3_speed_s0.stdout.txt",
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
    for top in ("RESULT.md", "SPEC_P1P2.md", "SPEC_P1P2_A1.md", "SPEC_P1P2_A2_P1PRIME.md"):
        git.append((top, NOTE[top]))
    for d in GIT_DIRS:
        for dp, dn, fn in os.walk(os.path.join(PKG, d)):
            dn[:] = [x for x in dn if x != "__pycache__"]
            for f in sorted(fn):
                rel = os.path.relpath(os.path.join(dp, f), PKG).replace("\\", "/")
                if f.endswith(NOT_GIT_SUFFIX) or f in NOT_GIT_NAMES or f in NOT_GIT_NOTE or f.startswith(NOT_GIT_PREFIX):
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
          "* P1 (30k) and P2/K7 are COMPLETE (P1 INCONCLUSIVE, K4; P2 reported, K8 failed and the SPEC rows conflict). FOUR detached CPU/queue processes are intentionally alive for P1' (SPEC_P1P2_A2_P1PRIME):",
          "  - the P1' GATE (code/run_p1p2_gate.py; pid 50748) already found raw/SPEC_A2_REGISTERED.txt and raw/P1X_GPU_GO.txt and started the battery's with_gpu_lock.py (pid 48020), which is WAITING for the dev-box lock (raw/gpu_wait_p1x.log; the lock is held by refcv7-a6text-step50400_a6text). It never touches a held lock. Stage P1X exports all 5,912 scenes on ckpt_50400.",
          "  - the P1' CPU waiter (code/run_p1x_cpu.py; pid 52336 + child): waits for raw/p1x_gpu_done.json AND the OFFICIAL 50,400 R7_A1 navhard scoring (counts.json PASS, 5,912 rows), builds the token sets by rule (raw/spec_p1x_token_sets.json), then WAITS for raw/P1X_SETS_ACK.txt before scoring any fan; gate: free commit >= 6 GB, <= 2 scorers.",
          "  Kill by explicit PID only (raw/launch_pids_p1x.json). The GPU, devbox_gpu.lock and every other process were not touched by this stream; the live navsim/code files were not edited (code/bridge_fix/ is a copy).", "",
          "Notes for the Master Mind", "-------------------------",
          "* P1' numbers do NOT exist yet. When raw/p1x_cpu_done.json appears run: python code/d6_p1_analyze.py --p1x (writes raw/d6_p1x.json; withholds every read unless K1,K2,K3,K4s,K5,K6 pass) and re-run code/make_landing_d6.py. The --p1x analysis path was compile-checked but NOT run on synthetic data (free commit was below the 6 GB line), so expect to fix a path bug if one exists.",
          "* Handshake still owed to the Master Mind: when raw/p1x_sets_ready.json appears, send raw/spec_p1x_token_sets.json, then create raw/P1X_SETS_ACK.txt (the CPU waiter scores nothing before it).",
          "* Registered hashes are in raw/SPEC_SHA256.txt (written by the Master Mind). The SPEC files must not be edited.",
          "* GOALS_AND_CLAIMS / MODEL_REGISTRY were not edited by this stream."]
    open(os.path.join(ROOT, "LANDING_READY_D6.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print(len(git), "git paths;", len(ng), "not-for-git")


if __name__ == "__main__":
    main()
