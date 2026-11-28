"""P1 agent-conditioning gate -- arm launcher WITH a one-variable preflight.

The preflight is not advisory: it DIFFS THE ACTUAL ARGV LISTS that will be
handed to `refc_v3_train.py` and refuses to run when an arm differs from the
control in anything other than its DECLARED delta. A row-bank arm once silently
multiplied an effective lambda by ~n/24 and invalidated two sweeps; the intent
was one variable there too.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

PY = r"C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
STACK = r"C:/Users/Admin/tanitad-p1gate/stack"
TRAINER = STACK + r"/scripts/refc_v3_train.py"
ROOT = r"C:/Users/Admin/p1gate"
JOIN = r"C:/Users/Admin/tanitad-caches/b1-agent-join-20260906/b1eval_agents.jsonl.xz"
LABELS = r"C:/Users/Admin/tanitad-refcv4b-analyze/labels/s2_labels_v7.2_eval.jsonl.gz"
ANCHORS = r"C:/Users/Admin/refcv4b_final/anchors.pt"


def base(out: str, steps: int, seed: int) -> list[str]:
    """Everything HELD CONSTANT across every arm."""
    return [
        "--arm", "hier", "--size", "tiny",
        "--v2-cache", ROOT + "/corpus/train",
        "--v7-labels", LABELS,
        "--eval-cache", ROOT + "/corpus/eval",
        "--eval-labels", LABELS,
        "--eval-every", str(max(steps // 2, 1)), "--eval-batches", "8",
        "--image-hw", "256", "640",
        "--steps", str(steps), "--batch", "8", "--workers", "0",
        "--prefetch-factor", "1", "--v2-lru", "8",
        "--lr", "1e-4", "--warmup", str(max(steps // 10, 1)), "--seed", str(seed),
        "--log-every", "25", "--save-every", str(steps),
        "--nav-from-v7", "--u8-batches",
        "--anchors", ANCHORS, "--n-anchors", "117",
        "--anchor-v0-conditioned", "--anchor-control-units", "alat",
        "--sel-accel-max", "2.0", "--goal-str",
        "--ego-state-inject", "--ego-dropout", "0.5",
        "--sampler", "ddim", "--w-u0", "0.5",
        "--out", out,
    ]


#: arm -> (declared delta appended to base, human reason)
ARMS: dict[str, tuple[list[str], str]] = {
    # CONTROL: the recipe every banked refcv4b/refcv5 arm was trained under.
    "off": (["--agents", "off"], "control: no agent seam, no graph built"),
    # TREATMENT: the ONE variable.
    "head": (["--agents", "head", "--w-agent", "1.0", "--agent-join", JOIN,
              "--agent-join-allow-legacy-ids"],
             "treatment: DD-faithful learned agent tokens, real join"),
    # DELIBERATE REGRESSION: identical to `head` in every flag; the JOIN FILE
    # is the shuffled one, so the graph is built, the detection loss is
    # computed, and the tokens carry NO information about THIS clip.
    "shuf": (["--agents", "head", "--w-agent", "1.0",
              "--agent-join", ROOT + "/joins/b1eval_agents_SHUFFLED.jsonl.xz",
              "--agent-join-allow-legacy-ids"],
             "deliberate regression: agents wired, clip->agents permuted"),
}


def preflight(arms: list[str], steps: int, seed: int) -> dict:
    """Refuse if any arm differs from the control by more than its delta."""
    ctrl = base(ROOT + f"/runs/off_s{seed}", steps, seed) + ARMS["off"][0]
    report = {"control_argv": ctrl, "arms": {}}
    ok = True
    for a in arms:
        argv = base(ROOT + f"/runs/{a}_s{seed}", steps, seed) + ARMS[a][0]
        # normalise the one flag that MUST differ (the out dir names the arm)
        def norm(v):
            return ["<OUT>" if "/runs/" in x else x for x in v]
        c, t = norm(ctrl), norm(argv)
        # the declared delta: everything from "--agents" onward
        ci, ti = c.index("--agents"), t.index("--agents")
        head_same = c[:ci] == t[:ti]
        report["arms"][a] = {
            "argv": argv,
            "declared_delta": ARMS[a][0],
            "held_constant_prefix_identical": head_same,
            "prefix_len": len(c[:ci]),
        }
        if not head_same:
            ok = False
            diff = [(i, x, y) for i, (x, y) in
                    enumerate(zip(c[:ci], t[:ti])) if x != y]
            report["arms"][a]["UNDECLARED_DIFF"] = diff
    report["PREFLIGHT"] = "PASS" if ok else "REFUSED"
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["off", "head", "shuf"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args()

    rep = preflight(a.arms, a.steps, a.seed)
    os.makedirs(ROOT + "/runs", exist_ok=True)
    with open(ROOT + f"/runs/preflight_s{a.seed}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    print("[preflight]", rep["PREFLIGHT"])
    for k, v in rep["arms"].items():
        print(f"  {k}: prefix_identical={v['held_constant_prefix_identical']} "
              f"held={v['prefix_len']} delta={' '.join(v['declared_delta'])}")
    if rep["PREFLIGHT"] != "PASS":
        print("[preflight] REFUSING TO LAUNCH -- arms differ in more than the "
              "one declared variable")
        return 3
    if a.preflight_only:
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = STACK + ";" + r"C:/Users/Admin/tanitad-p1gate/taniteval"
    env["PYTHONIOENCODING"] = "utf-8"
    for arm in a.arms:
        out = ROOT + f"/runs/{arm}_s{a.seed}"
        os.makedirs(out, exist_ok=True)
        argv = [PY, "-u", TRAINER] + rep["arms"][arm]["argv"]
        t0 = time.time()
        print(f"[launch] {arm} seed={a.seed} -> {out}", flush=True)
        with open(out + "/train.log", "w", encoding="utf-8") as lg:
            rc = subprocess.call(argv, env=env, stdout=lg,
                                 stderr=subprocess.STDOUT, cwd=out)
        print(f"[done] {arm} rc={rc} {time.time()-t0:.0f}s", flush=True)
        if rc != 0:
            print(f"[FAIL] {arm} rc={rc}; tail:")
            with open(out + "/train.log", encoding="utf-8",
                      errors="replace") as lg:
                print("".join(lg.readlines()[-40:]))
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
