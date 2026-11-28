import sys, json, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import common2 as C
fs = [f"{C.OLDRAW}/{n}" for n in ("windows_train.npz","sample_pstop.npz","sample_pmove.npz","eval141_features.npz","results_eval141.json","pca_V_pstop.npz","pca_V_pmove.npz","pca_Z_pstop.npz","pca_Z_pmove.npz",
      "oof_pstop_V_linear.npz","oof_pmove_V_linear.npz","oof_pstop_Z_linear.npz","oof_pmove_Z_linear.npz","oof_pstop_V_mlp.npz","oof_pmove_V_mlp.npz","oof_pstop_Z_mlp.npz","oof_pmove_Z_mlp.npz")]
fs += [f"{C.OLDSCR}/{n}" for n in ("join_train_rebuilt.npz","V_pstop.npy","V_pmove.npy","Z_pstop.npy","Z_pmove.npy","V_eval141.npy","pcacomp_V_pstop_all.npz","pcacomp_V_pmove_all.npz","pcacomp_Z_pstop_all.npz","pcacomp_Z_pmove_all.npz")]
fs += [C.EVAL_JOIN, C.CKPT]
out = {}
t0 = time.time()
for f in fs:
    out[f] = {"sha256": C.sha256_file(f), "bytes": os.path.getsize(f)}
out["_ckpt_sha_expected"] = C.CKPT_SHA
assert out[C.CKPT]["sha256"] == C.CKPT_SHA, "checkpoint sha mismatch"
json.dump(out, open(f"{C.RAW}/INPUTS_SHA256.json", "w"), indent=1)
print("ok", len(fs), f"{time.time()-t0:.0f}s")
