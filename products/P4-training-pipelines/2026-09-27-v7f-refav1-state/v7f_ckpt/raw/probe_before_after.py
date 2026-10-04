"""Before/after probe on the unit rig (CPU, tiny S-W, synthetic corpus), MEASURED:
 (1) FRESH-run numerics: the pre-F3 merge trainer (sha 8c212e5f) vs the F3 trainer -- per-step loss, draws,
     final model + optimiser state must be bit-identical (the capture is a pure read; nothing else runs on a
     fresh launch);
 (2) the RESUME: the pre-F3 trainer on the interrupted-run protocol reproduces F3 (the first batches) and F3b
     (the lr factor), the F3 trainer is exact.
"""
import hashlib, importlib.util, json, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, "C:/Users/Admin/v7f_ckpt/stack/tests")
import test_v6_exact_resume as X  # noqa: E402

torch = X.torch
PRE = Path("C:/Users/Admin/v7f_ckpt/pristine/train_v6_staged.py")
assert hashlib.sha256(PRE.read_bytes()).hexdigest().startswith("8c212e5fe2b61237"), "not the pre-F3 merge trainer"
spec = importlib.util.spec_from_file_location("train_v6_staged_preF3", PRE)
pre = importlib.util.module_from_spec(spec)
sys.modules["train_v6_staged_preF3"] = pre
spec.loader.exec_module(pre)
post = X.T
out = {"pre_sha256": hashlib.sha256(PRE.read_bytes()).hexdigest(),
       "post_sha256": hashlib.sha256(Path(post.__file__).read_bytes()).hexdigest(),
       "torch": torch.__version__, "fresh": {}, "resume": {}}

for name, extra in (("nav_o4", X.NAV), ("nonav_o4", X.NO_NAV), ("nav_uniform", X.NAV + ("--o4-alpha", "0")),
                    ("nav_sigreg_bank", X.NAV + ("--sigreg-accum", "3"))):
    tmp = Path(tempfile.mkdtemp(prefix=f"f3probe_{name}_"))
    cache = tmp / "empty"
    cache.mkdir()
    r_pre = X.run_trainer(X._argv(tmp / "pre", cache, *extra), module=pre)
    r_post = X.run_trainer(X._argv(tmp / "post", cache, *extra), module=post)
    lp = {s: r["loss"] for s, r in X._train_rows(tmp / "pre").items()}
    lq = {s: r["loss"] for s, r in X._train_rows(tmp / "post").items()}
    lrp = {s: r["lr"] for s, r in X._train_rows(tmp / "pre").items()}
    lrq = {s: r["lr"] for s, r in X._train_rows(tmp / "post").items()}
    cp, cq = X._load(tmp / "pre" / "ckpt.pt"), X._load(tmp / "post" / "ckpt.pt")
    out["fresh"][name] = {
        "rc": [r_pre["rc"], r_post["rc"]], "draws_equal": r_pre["draws"] == r_post["draws"],
        "n_draws": len(r_pre["draws"]), "loss_equal": lp == lq, "lr_equal": lrp == lrq, "losses": lq,
        "model_differ": X._tensor_diff(cp["stack"], cq["stack"]), "n_model_tensors": len(cq["stack"]),
        "opt_state_differ": X._tensor_diff(cp["opt"]["state"], cq["opt"]["state"]),
        "opt_groups_equal": cp["opt"]["param_groups"] == cq["opt"]["param_groups"],
        "ckpt_keys_pre": sorted(cp), "ckpt_keys_post": sorted(cq)}
    print("FRESH", name, json.dumps({k: v for k, v in out["fresh"][name].items() if k != "losses"}), flush=True)

for label, mod in (("pre_F3", pre), ("post_F3", post)):
    t0 = time.time()
    reasons, det = X.abc(Path(tempfile.mkdtemp(prefix=f"f3probe_abc_{label}_")), *X.NAV, module=mod)
    out["resume"][label] = {"reasons": reasons, "elapsed_s": round(time.time() - t0, 1),
                            **{k: det[k] for k in ("draws_resumed", "draws_uninterrupted_same_steps",
                                                   "draws_uninterrupted_first_steps", "loss_resumed",
                                                   "loss_uninterrupted", "lr_resumed", "lr_uninterrupted",
                                                   "final_step", "opt_groups_equal", "n_model_tensors")},
                            "n_model_differ": len(det["model_differ"]),
                            "n_opt_state_differ": len(det["opt_state_differ"]),
                            "config_resume_rng": det["config_resume_rng"]}
    if label == "pre_F3":
        out["resume"][label]["lr_ratio_step4"] = det["lr_resumed"][4] / det["lr_uninterrupted"][4]
    print("RESUME", label, json.dumps(out["resume"][label], default=str)[:1500], flush=True)

Path("C:/Users/Admin/v7f_ckpt/work/probe_before_after.json").write_text(json.dumps(out, indent=1, default=str))
print("WROTE probe_before_after.json")
