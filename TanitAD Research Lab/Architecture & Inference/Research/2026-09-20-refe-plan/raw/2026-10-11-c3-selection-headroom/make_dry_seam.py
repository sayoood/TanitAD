"""DRY RUN helper: the BANKED u400 picks restricted to the C3 population, as a seam. Scoring it through the subset harness
checks (i) the Hydra token-list command line fits Windows' limit at 1,058 tokens and (ii) the subset harness reproduces the
banked per-token score for every population token -- the same check the pipeline's G2 makes on the re-run picks."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
toks = json.load(open(os.path.join(HERE, "tokens.json")))["tokens"]
B = np.load("D:/Projects/TanitAD/data/refe_navtest/seams/stage2_sft4_B_u400_navtest_full.npz")
ix = {str(t): i for i, t in enumerate(B["token"])}
sel = [ix[t] for t in toks]
out = "D:/Projects/TanitAD/data/refe_navtest/seams/c3/dry/refe_c3_u400_dry_pick.npz"
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, token=B["token"][sel], fingerprint=B["fingerprint"][sel], poses=B["poses"][sel], sampling=B["sampling"], arm=np.array("REFe_c3_dry_banked_pick"))
print(len(sel), out)
