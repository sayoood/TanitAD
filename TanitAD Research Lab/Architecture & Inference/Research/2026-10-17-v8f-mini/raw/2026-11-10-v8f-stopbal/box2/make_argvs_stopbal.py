r"""Builds the STOPBAL retrain argvs (PREREG_V8F_STOPBAL, FROZEN 2026-11-10): each arm = DAG-OACC's argv of its init (v8f-dagoacc/M2b_s{k}_dagoacc/argv.json) VERBATIM, with ONLY
  --out D:\tanitad\runs\v8f_stopbal\<tag>, [--seed 2 for the _rep arm], and the card's one flag (BAL-G: --goal-pos-weight-stop 4.5616; BAL-L: --lon-class-weight-stop 4.6991; BAL-NULL: nothing).
Checks that every argv parses with the trainer at the code of record (export dir STOPBAL_TREE) and that each diff against the DAG-OACC argv is exactly the declared one. Writes frozen/argvs_stopbal.json,
argvs_stopbal.txt and argvs_stopbal.meta.json beside this file. Nothing is launched.
Usage: C:/Users/Admin/venvs/tanitad/Scripts/python.exe make_argvs_stopbal.py   (env STOPBAL_TREE = export of commit 2bca1c52, default F:/TanitAD-scratch/stopbal_build/tree)"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "E:/Projects/TanitAD-artifacts/v8f-dagoacc"
TREE = os.environ.get("STOPBAL_TREE", "F:/TanitAD-scratch/stopbal_build/tree")
COMMIT = "2bca1c52abbc2346d7380d33ae78b324cbda9df7"
TRAINER_SHA256_AT_COMMIT = "1ae4ef3b0617e8d523889251811cb4de029b9c379e9aafc9f8ad5d2d5f35a119"      # sha256 of `git show 2bca1c52:stack/scripts/v8f_mini_train.py` (MEASURED 2026-11-10)
OUT_ROOT = r"D:\tanitad\runs\v8f_stopbal" + "\\"
W_GOAL, W_LON = "4.5616", "4.6991"                                                                  # the card: square-root rule, MEASURED prevalence (prevalence_stopbal.json)
# arm -> (init seed k, extra flags appended, training seed override or None, what)
ARMS = {
    "BAL-NULL_s0": (0, [], None, "DAG-OACC s0's argv, only --out (same-code reference at the code of record)"),
    "BAL-NULL_rep": (0, [], 2, "BAL-NULL s0's argv at --seed 2 (the run-to-run floor)"),
    "BAL-G_s0": (0, ["--goal-pos-weight-stop", W_GOAL], None, "DAG-OACC s0 + --goal-pos-weight-stop 4.5616"),
    "BAL-L_s0": (0, ["--lon-class-weight-stop", W_LON], None, "DAG-OACC s0 + --lon-class-weight-stop 4.6991"),
    "BAL-NULL_s1": (1, [], None, "DAG-OACC s1's argv, only --out"),
    "BAL-G_s1": (1, ["--goal-pos-weight-stop", W_GOAL], None, "DAG-OACC s1 + --goal-pos-weight-stop 4.5616"),
    "BAL-L_s1": (1, ["--lon-class-weight-stop", W_LON], None, "DAG-OACC s1 + --lon-class-weight-stop 4.6991"),
}


def load(k):
    return json.load(open(f"{SRC}/M2b_s{k}_dagoacc/argv.json", encoding="utf-8"))


def main():
    sys.path[:0] = [TREE + "/stack", TREE + "/stack/scripts"]
    import v8f_mini_train as TR  # noqa: E402
    trainer_sha = hashlib.sha256(open(TREE + "/stack/scripts/v8f_mini_train.py", "rb").read()).hexdigest()
    assert trainer_sha == TRAINER_SHA256_AT_COMMIT, f"the export at STOPBAL_TREE is not the unmodified trainer of {COMMIT[:8]} (sha256 {trainer_sha})"
    base = {k: load(k) for k in (0, 1)}
    for k in (0, 1):
        assert TR.argv_sha(base[k]["argv"]) == base[k]["sha"], f"DAG-OACC s{k}: argv_sha at the commit {TR.argv_sha(base[k]['argv'])} != recorded {base[k]['sha']}"
    out, lines = {}, []
    for tag, (k, extra, seed, what) in ARMS.items():
        av = list(base[k]["argv"])
        i = av.index("--out")
        av[i + 1] = OUT_ROOT + tag
        if seed is not None:
            j = av.index("--seed")
            av[j + 1] = str(seed)
        av += extra
        b = base[k]["argv"]
        diffs = [n for n in range(len(b)) if av[n] != b[n]]
        want = [i + 1] + ([av.index("--seed") + 1] if seed is not None else [])
        assert diffs == sorted(want), (tag, diffs, want)
        assert av[len(b):] == extra, tag
        ns = TR.build_parser().parse_args(av)
        assert (ns.goal_pos_weight_stop, ns.lon_class_weight_stop) == (float(W_GOAL) if "--goal-pos-weight-stop" in extra else 1.0, float(W_LON) if "--lon-class-weight-stop" in extra else 1.0), tag
        out[tag] = {"what": what, "init_seed": k, "argv": av}
        lines.append(f"{tag}: " + " ".join(av))
    os.makedirs(HERE + "/frozen", exist_ok=True)
    json.dump(out, open(HERE + "/frozen/argvs_stopbal.json", "w", encoding="utf-8", newline="\n"), indent=1)
    open(HERE + "/argvs_stopbal.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    meta = {"commit": COMMIT, "trainer_sha256_at_commit": trainer_sha, "dagoacc_argv_sha": {str(k): base[k]["sha"] for k in base},
            "argvs_json_sha256": hashlib.sha256(open(HERE + "/frozen/argvs_stopbal.json", "rb").read()).hexdigest()}
    json.dump(meta, open(HERE + "/argvs_stopbal.meta.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print(len(out), "argvs parse; trainer sha256", trainer_sha, "; argvs sha256", meta["argvs_json_sha256"])


if __name__ == "__main__":
    main()
