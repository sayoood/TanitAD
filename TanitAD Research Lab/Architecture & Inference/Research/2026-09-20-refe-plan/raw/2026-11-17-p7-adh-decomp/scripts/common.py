"""shared setup: imports the pinned instruments (never copies/edits them). Code exported from mirror FETCH_HEAD into SCR/code."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ["TANITAD_REPO"] = "E:/Projects/TanitAD"
SCR = "F:/tanitad_scratch/adhdecomp"
HOLD = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/adh_decomp"
sys.path.insert(0, SCR + "/code")
import numpy as np
import command_adherence as CA
SEAMS = "E:/Projects/TanitAD/data/refe_navtest/seams"
DUMP = SEAMS + "/stage2_p7_full.proposals.npz"
SEAM = SEAMS + "/stage2_p7_full.npz"
CSV = "E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p7_full/refe_stage2_p7_full.csv"
def setup():
    exp = CA.load_export()
    seam = CA.load_seam(SEAM)
    tokens = sorted(seam["tokens"])
    cmd, logs = CA.commands_and_logs(tokens, exp)
    poses = CA.align(seam, tokens)
    z = np.load(DUMP, allow_pickle=False)
    idx = {str(t): i for i, t in enumerate(z["token"])}
    order = np.array([idx[t] for t in tokens])
    return exp, tokens, cmd, logs, poses, z, order
