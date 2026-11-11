r"""Builds the STOPBAL second-dose argvs (AMENDMENT_3_DOSE2.md, sha256 460fd548...): each arm = DAG-OACC's argv of its init (v8f-dagoacc/M2b_s{k}_dagoacc/argv.json) VERBATIM, with ONLY
  --out D:\tanitad\runs\v8f_stopbal\<tag> and the one flag (BAL-G2: --goal-pos-weight-stop 20.81; BAL-L2: --lon-class-weight-stop 22.08).
Checks that every argv parses with the trainer at the code of record 715ef8fc (export dir DOSE2_TREE), that each diff against the DAG-OACC argv is exactly the declared one, and that each equals the
FIRST dose's frozen argv of the same lever and seed with only the flag value changed (4.5616 -> 20.81, 4.6991 -> 22.08). Writes frozen/argvs_dose2.json, argvs_dose2.txt, argvs_dose2.meta.json. Nothing is launched.
Usage: C:/Users/Admin/venvs/tanitad/Scripts/python.exe make_argvs_dose2.py   (env DOSE2_TREE = export of 715ef8fc, default F:/TanitAD-scratch/dose2_build/tree)"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "E:/Projects/TanitAD-artifacts/v8f-dagoacc"
FIRST = os.path.join(os.path.dirname(HERE), "box2", "frozen", "argvs_stopbal.json")
TREE = os.environ.get("DOSE2_TREE", "F:/TanitAD-scratch/dose2_build/tree")
COMMIT = "715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf"
TRAINER_SHA256 = "1ae4ef3b0617e8d523889251811cb4de029b9c379e9aafc9f8ad5d2d5f35a119"      # `git show 715ef8fc:stack/scripts/v8f_mini_train.py` (MEASURED; equal to the file at 2bca1c52)
OUT_ROOT = r"D:\tanitad\runs\v8f_stopbal" + "\\"
W_GOAL, W_LON = "20.81", "22.08"                                                          # AMENDMENT_3_DOSE2.md: full balance, n_neg / n_pos
# arm -> (init seed, flag, value, first-dose twin tag, first-dose value)
ARMS = {
    "BAL-G2_s0": (0, "--goal-pos-weight-stop", W_GOAL, "BAL-G_s0", "4.5616"),
    "BAL-L2_s0": (0, "--lon-class-weight-stop", W_LON, "BAL-L_s0", "4.6991"),
    "BAL-G2_s1": (1, "--goal-pos-weight-stop", W_GOAL, "BAL-G_s1", "4.5616"),
    "BAL-L2_s1": (1, "--lon-class-weight-stop", W_LON, "BAL-L_s1", "4.6991"),
}


def main():
    sys.path[:0] = [TREE + "/stack", TREE + "/stack/scripts"]
    import v8f_mini_train as TR  # noqa: E402
    tsha = hashlib.sha256(open(TREE + "/stack/scripts/v8f_mini_train.py", "rb").read()).hexdigest()
    assert tsha == TRAINER_SHA256, f"the export at DOSE2_TREE is not the unmodified trainer of {COMMIT[:8]} ({tsha})"
    base = {k: json.load(open(f"{SRC}/M2b_s{k}_dagoacc/argv.json", encoding="utf-8")) for k in (0, 1)}
    first = json.load(open(FIRST, encoding="utf-8"))
    out, lines = {}, []
    for tag, (k, flag, val, twin, twin_val) in ARMS.items():
        assert TR.argv_sha(base[k]["argv"]) == base[k]["sha"], k
        av = list(base[k]["argv"])
        av[av.index("--out") + 1] = OUT_ROOT + tag
        av += [flag, val]
        b = base[k]["argv"]
        diffs = [n for n in range(len(b)) if av[n] != b[n]]
        assert diffs == [b.index("--out") + 1] and av[len(b):] == [flag, val], tag
        ft = list(first[twin]["argv"])
        ft[ft.index("--out") + 1] = OUT_ROOT + tag
        assert ft[-1] == twin_val and ft[-2] == flag, twin
        ft[-1] = val
        assert ft == av, f"{tag} differs from the first dose's {twin} beyond the flag value"
        ns = TR.build_parser().parse_args(av)
        want = (float(val) if flag == "--goal-pos-weight-stop" else 1.0, float(val) if flag == "--lon-class-weight-stop" else 1.0)
        assert (ns.goal_pos_weight_stop, ns.lon_class_weight_stop) == want, tag
        out[tag] = {"what": f"DAG-OACC s{k} + {flag} {val}", "init_seed": k, "argv": av}
        lines.append(f"{tag}: " + " ".join(av))
    os.makedirs(HERE + "/frozen", exist_ok=True)
    json.dump(out, open(HERE + "/frozen/argvs_dose2.json", "w", encoding="utf-8", newline="\n"), indent=1)
    open(HERE + "/argvs_dose2.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    meta = {"commit": COMMIT, "trainer_sha256_at_commit": tsha, "dagoacc_argv_sha": {str(k): base[k]["sha"] for k in base},
            "argvs_json_sha256": hashlib.sha256(open(HERE + "/frozen/argvs_dose2.json", "rb").read()).hexdigest()}
    json.dump(meta, open(HERE + "/argvs_dose2.meta.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print(len(out), "argvs parse; each equals the first dose's twin with only the flag value changed; argvs sha256", meta["argvs_json_sha256"])


if __name__ == "__main__":
    main()
