#!/usr/bin/env python3
"""TF-N2 navtest prep (run BEFORE the card is frozen; it rolls and scores nothing and opens no navtest value).

  python n2_nt_prep.py --run-dir F:/tanitad_scratch/tfn2_nt/run --mirror C:/Users/Admin/tanitad-push/.git --tip <sha-or-ref> [--hold <this dir>]

Builds a FRESH run tree from the git tip (git archive; autocrlf is irrelevant) and from the hold, then pins it:
  w/pk/refe/, w/pk/code/, w/pk/eval/ (top-level .py, instruments/, harness/)   <- tip PK/refe, PK/code, PK/eval
  w/pk/refe/build_teacher_rollouts.py                              <- tip TF/n2/reroll/ (the patched engine-knob hook; overwrites the tip's PK/refe copy)
  w/pk/eval/score_p4_a1.py, n2/tf_n1_analyze_a1.py                 <- tip TF/n1/a1/ (TF-N1-A1's harness route and estimator, unchanged)
  w/p5r/p5r_labels.py                                              <- tip raw/2026-11-12-p5r/ (P5-R, lambda* 3.0)
  repo/ (taniteval + stack/tanitad code, for families_run)         <- tip
  inputs/rollouts/rows_s{0..3}of4.jsonl.xz, refe_p4_teacher_navtest.{npz,csv}, refe_p4_human_navtest.csv, frame_control_navtest_full.json,
  inputs/n1_csv/refe_a1_tfn1_c.csv                                  <- tip (banked inputs; N1's run dir was deleted, its CSVs are in the repo)
  n2/tfn2_nt_{build,analyze,chain}.py, w/pk/eval/tfn2_nt_roll.py   <- the hold
Then: harness_pins.verify on the exported harness; refuses unless the E: copy of score_navtest_refe.py (which harness/score_e.py loads by an absolute
path) equals the run tree's; writes n2/N2_PINS.json (sha256 of every .py/.json/.yaml under w/, repo/, n2/ plus every input and the absolute devkit /
token-export files) and prints its sha256, which the card quotes. Hashing reads BYTES only, never values.
"""
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile

PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TF = f"{PK}/raw/2026-11-11-teacher-fidelity"
P5R = f"{PK}/raw/2026-11-12-p5r/p5r_labels.py"
DEVKIT = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
TOKEN_LOG = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
DEVKIT_FILES = ["navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
                "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
                "navsim/evaluate/pdm_score.py", "navsim/planning/metric_caching/metric_cache_processor.py"]
WATCH_EXT = (".py", ".json", ".yaml", ".yml")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_lf(p):
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


class Git:
    def __init__(self, mirror, tip):
        self.m, self.tip = mirror, tip

    def run(self, *args):
        return subprocess.run(["git", "--git-dir", self.m, "-c", "core.autocrlf=false", *args], check=True, capture_output=True).stdout

    def files(self, path):
        out = self.run("ls-tree", "-r", "--name-only", self.tip, "--", path).decode("utf-8").splitlines()
        return [x.strip('"') for x in out if x]

    def tar(self, paths):
        return tarfile.open(fileobj=io.BytesIO(self.run("archive", self.tip, *paths)))


def put(run, rel, data):
    out = os.path.join(run, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(data)


def export_tree(g, run, src, dst, keep=lambda rel: True):
    """every file under tip:<src>/ -> <run>/<dst>/<rel>"""
    n = 0
    with g.tar([src]) as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.startswith(src + "/"):
                continue
            rel = m.name[len(src) + 1:]
            if rel.startswith("/") or ".." in rel.split("/"):
                sys.exit(f"REFUSING: odd tar member {m.name}")
            if keep(rel):
                put(run, f"{dst}/{rel}", tf.extractfile(m).read())
                n += 1
    return n


def export_file(g, run, src, dst):
    put(run, dst, g.run("show", f"{g.tip}:{src}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mirror", required=True, help="the .git dir of the repo mirror (read only)")
    ap.add_argument("--tip", required=True)
    ap.add_argument("--hold", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    run = a.run_dir
    if os.path.exists(run) and os.listdir(run):
        sys.exit(f"REFUSING: {run} exists and is not empty (prep only makes a fresh tree; a resume uses the existing one)")
    g = Git(a.mirror, a.tip)
    print("tip", a.tip, "=", g.run("rev-parse", a.tip).decode().strip())
    n = export_tree(g, run, f"{PK}/refe", "w/pk/refe")
    n += export_tree(g, run, f"{PK}/code", "w/pk/code")          # route_rank.py imports route_lane_rank_patch from ../code
    n += export_tree(g, run, f"{PK}/eval/instruments", "w/pk/eval/instruments")
    n += export_tree(g, run, f"{PK}/eval/harness", "w/pk/eval/harness")
    for f in g.files(f"{PK}/eval"):
        rel = f[len(PK) + len("/eval/"):]
        if "/" not in rel and rel.endswith(".py"):
            export_file(g, run, f, f"w/pk/eval/{rel}")
            n += 1
    print("code files exported:", n)
    export_tree(g, run, "taniteval", "repo/taniteval", lambda r: r.endswith((".py", ".json", ".yaml", ".yml")) and not r.startswith(("results/", "tests/")))
    export_tree(g, run, "stack/tanitad", "repo/stack/tanitad", lambda r: r.endswith((".py", ".json", ".yaml", ".yml")))
    export_file(g, run, f"{TF}/n2/reroll/build_teacher_rollouts.py", "w/pk/refe/build_teacher_rollouts.py")
    export_file(g, run, f"{TF}/n1/a1/score_p4_a1.py", "w/pk/eval/score_p4_a1.py")
    export_file(g, run, f"{TF}/n1/a1/tf_n1_analyze_a1.py", "n2/tf_n1_analyze_a1.py")
    export_file(g, run, P5R, "w/p5r/p5r_labels.py")
    if "TEACHER_OVERRIDES" not in open(os.path.join(run, "w/pk/refe/build_teacher_rollouts.py"), encoding="utf-8").read():
        sys.exit("REFUSING: build_teacher_rollouts.py has no TEACHER_OVERRIDES hook (the TF-N2 re-roll patch is not at the tip)")
    # ---- banked inputs, all from the tip
    T5 = f"{PK}/raw/2026-10-05-teacher-navtest"
    for i in range(4):
        export_file(g, run, f"{T5}/full/rows_s{i}of4.jsonl.xz", f"inputs/rollouts/rows_s{i}of4.jsonl.xz")
    export_file(g, run, f"{T5}/refe_p4_teacher_navtest.npz", "inputs/refe_p4_teacher_navtest.npz")
    export_file(g, run, f"{T5}/csv/refe_p4_teacher_navtest.csv", "inputs/refe_p4_teacher_navtest.csv")
    export_file(g, run, f"{T5}/csv/refe_p4_human_navtest.csv", "inputs/refe_p4_human_navtest.csv")
    export_file(g, run, f"{PK}/raw/2026-10-04-navtest-final/frame_control_navtest_full.json", "inputs/frame_control_navtest_full.json")
    export_file(g, run, f"{TF}/n1/run_a1/csv/refe_a1_tfn1_c.csv", "inputs/n1_csv/refe_a1_tfn1_c.csv")
    # ---- the hold's scripts
    for f in ("tfn2_nt_build.py", "tfn2_nt_analyze.py", "tfn2_nt_chain.py"):
        put(run, f"n2/{f}", open(os.path.join(a.hold, f), "rb").read())
    put(run, "w/pk/eval/tfn2_nt_roll.py", open(os.path.join(a.hold, "tfn2_nt_roll.py"), "rb").read())
    # ---- the harness is the pinned one
    sys.path.insert(0, os.path.join(run, "w", "pk", "eval"))
    import harness_pins
    v = harness_pins.verify(root=harness_pins.Path(os.path.join(run, "w/pk/eval/harness")),
                            caller=harness_pins.Path(run) / "w" / "pk" / "eval" / "score_navtest_refe.py")
    if not v.get("ok"):
        sys.exit("REFUSING: exported harness does not match HARNESS_PINS: " + json.dumps(v, default=str)[:600])
    print("harness_pins.verify OK:", v["n_files"], "files, caller variant", v["caller"]["variant"])
    src = open(os.path.join(run, "w/pk/eval/harness/score_e.py"), encoding="utf-8").read()
    m = re.search(r'^PKG = "([^"]+)"', src, re.M)
    ec = m.group(1) + "/eval/score_navtest_refe.py"
    mine = os.path.join(run, "w/pk/eval/score_navtest_refe.py")
    if not os.path.isfile(ec) or sha_lf(ec) != sha_lf(mine):
        sys.exit(f"REFUSING: score_e.py loads {ec}, which is missing or differs from {mine}")
    print("score_e.py caller identical to the run tree's:", ec)
    # ---- pins
    files = {}
    for top in ("w", "repo", "n2"):
        for d, _, fs in os.walk(os.path.join(run, top)):
            if "__pycache__" in d:
                continue
            for f in fs:
                if f.endswith(WATCH_EXT) and f != "N2_PINS.json":
                    files[os.path.relpath(os.path.join(d, f), run).replace(os.sep, "/")] = sha(os.path.join(d, f))
    for d, _, fs in os.walk(os.path.join(run, "inputs")):
        for f in fs:
            files[os.path.relpath(os.path.join(d, f), run).replace(os.sep, "/")] = sha(os.path.join(d, f))
    for f in DEVKIT_FILES:
        files[f"{DEVKIT}/{f}"] = sha(f"{DEVKIT}/{f}")
    files[TOKEN_LOG] = sha(TOKEN_LOG)
    out = os.path.join(run, "n2", "N2_PINS.json")
    json.dump({"card": "PREREG_TF_N2.md C-PIN", "tip": a.tip, "files": dict(sorted(files.items()))}, open(out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"ZZN2_PINS {len(files)} files -> {out}\nN2_PINS.json sha256 {sha(out)}")


if __name__ == "__main__":
    main()
