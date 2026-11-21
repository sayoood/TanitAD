"""Known-value control for the BANKED harness (G3): a 24-token subset of u400's banked full-navtest seam, to be re-scored by the banked harness and
compared with u400's banked per-token CSV (bit-exact expected). No P6 checkpoint, no GPU."""
import csv, gzip, json, sys
import numpy as np
D = "E:/Projects/TanitAD/data/refe_navtest"
EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
minus = json.load(open(f"{D}/navtest_full_minus_c1_tokens.json", encoding="utf-8"))
ok = set(minus["tokens"] if isinstance(minus, dict) else minus)
exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
z = np.load(f"{D}/seams/stage2_sft4_B_u400_navtest_full.npz")
tok = [str(t) for t in z["token"]]
by = {}
for i, t in enumerate(tok):
    if t in ok:
        by.setdefault(exp[t]["log_name"], []).append(i)
logs = sorted(l for l, v in by.items() if len(v) >= 12)[:2]
idx = sorted(i for l in logs for i in by[l][:12])
sub = f"C:/Users/Admin/par18/p6gaps/raw/harness_ctl/ctl_seam.npz"
np.savez(sub, token=z["token"][idx], fingerprint=z["fingerprint"][idx], poses=z["poses"][idx], sampling=z["sampling"], arm=z["arm"])
toks = [tok[i] for i in idx]
json.dump({"tokens": toks, "token_log": {t: exp[t]["log_name"] for t in toks}}, open("C:/Users/Admin/par18/p6gaps/raw/harness_ctl/ctl_tokens.json", "w"), indent=0)
print(len(toks), logs)
