"""Builds the OTBAND retrain argvs (box 2 paths as in theory/ladder/IMPL_CHOICES_LADDER.md section 2) and checks that every one parses with the OTBAND trainer's parser.
Writes argvs_otband.json (arm -> list of tokens) and argvs_otband.txt (one line per run). Nothing is launched. Usage: python make_argvs.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = "F:/TanitAD-scratch/hwm_bands/b/tree"
sys.path[:0] = [TREE + "/stack", TREE + "/stack/scripts", TREE + "/taniteval"]
import v8f_mini_train as TR  # noqa: E402

RUNS = r"D:\tanitad\runs"
TAIL = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--exclude", "540:600", "--device", "cuda", "--batch", "64",
        "--cache-dir", r"D:\tanitad\cache_full", "--v9-train", r"D:\tanitad\data\v9\v9_labels_train.npz", "--v9-train-md5", "83e30bd3791f0767707eade5f4671759",
        "--lead-train", r"D:\tanitad\data\v9\lead_train4572.npz", "--speed-train", r"D:\tanitad\data\v9\refcv6_speed_max_v8_train.jsonl",
        "--join-train", RUNS + r"\v8f_mini_full\join_train.npz", "--p10", RUNS + r"\v8f_mini_full\p10\p10_prior.pt", "--tracks", r"D:\tanitad\cache_full\native_tracks.pt",
        "--steps-mult", "2"]
BANDS = ["--w-traj", "0", "--w-op", "0.3333333", "--w-tac-traj", "0.6666667"]            # the knot-count fractions 10/30 and 20/30: where the bands agree the loss equals the 30-knot ADE weighting
STOK = r"D:\tanitad\data\stok\STREAM_C_DIR_UNVERIFIED"
FREEZE = ["--freeze", "S", "adapter", "cond"]                                              # S's output stays M2b's (the group AND the shared adapter / embeddings frozen); T trainable in every arm
DET = ["--detach-upper"]                                                                    # SPEC section 9: O's loss puts 0.0 gradient on T (default off; M2b ran without it), named in every arm
ARMS = {
    "OB-BAND-TOK": ("band loss (w_op 1/3, w_tac_traj 2/3, w_traj 0) with the detached strategic token into T", [*BANDS, *DET, *FREEZE, "--t-stok-dir", STOK]),
    "OB-BAND": ("band loss, no token (the matched control of the token arm)", [*BANDS, *DET, *FREEZE]),
    "OB-CTL": ("today's loss (w_traj 1, no band weight), no token, same detach and freeze (the control of the band loss)", [*DET, *FREEZE]),
}
out, lines = {}, []
for arm, (what, flags) in ARMS.items():
    for s in (0, 1):
        name = f"{arm}_s{s}"
        argv = [*TAIL, "--seed", str(s), *flags, "--init-ckpt", RUNS + rf"\v8f_mini_full\M2b_s{s}\ckpt_last.pt", "--out", RUNS + rf"\v8f_otband\{name}"]
        out[name] = {"what": what, "argv": argv}
name = "OB-BAND_rep"                                    # the replicate: OB-BAND's flags, init M2b_s0, another training seed
out[name] = {"what": "replicate of OB-BAND s0: the same flags and init, training seed 2 (the run-to-run floor of the retrain, H-ESTIM-SEED-1)",
             "argv": [*TAIL, "--seed", "2", *BANDS, *DET, *FREEZE, "--init-ckpt", RUNS + r"\v8f_mini_full\M2b_s0\ckpt_last.pt", "--out", RUNS + rf"\v8f_otband\{name}"]}
for name, d in out.items():
    TR.build_parser().parse_args(d["argv"])                 # every token is a flag of the OTBAND trainer, every value parses
    lines.append(f"{name}: " + " ".join(d["argv"]))
json.dump(out, open(HERE + "/argvs_otband.json", "w", encoding="utf-8"), indent=1)
open(HERE + "/argvs_otband.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print(len(out), "argvs parse")
