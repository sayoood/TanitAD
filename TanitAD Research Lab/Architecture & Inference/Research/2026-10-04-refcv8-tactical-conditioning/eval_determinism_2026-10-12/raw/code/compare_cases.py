import json, sys, torch
def eq_maps(a, b):
    ks = sorted(set(a) | set(b))
    return [k for k in ks if k not in a or k not in b or a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
            or not torch.equal(a[k], b[k])]
rep = {}
for e in (0, 1):
    o = torch.load(f"case_tree_emit{e}.pt", weights_only=False)
    n = torch.load(f"case_tree_new_emit{e}.pt", weights_only=False)
    nb = o["n_base"]
    r = {}
    for k in ("train1", "train2"):
        r[k] = {"outputs_differ_old_vs_new": eq_maps(o[k]["out"], n[k]["out"]),
                "grads_differ_old_vs_new": eq_maps(o[k]["grads"], n[k]["grads"]),
                "n_outputs": len(o[k]["out"]), "n_grads": len(o[k]["grads"]),
                "loss_equal": bool(torch.equal(o[k]["loss"], n[k]["loss"])),
                "r8_gen_state_equal": bool(torch.equal(o[k]["gen_state"], n[k]["gen_state"]))}
    r["eval_repeat_differ_OLD"] = eq_maps(o["eval_a"]["out"], o["eval_b"]["out"])
    r["eval_repeat_differ_NEW"] = eq_maps(n["eval_a"]["out"], n["eval_b"]["out"])
    r["eval_seed777_vs_1234_differ_NEW"] = eq_maps(n["eval_a"]["out"], n["eval_c_seed777"]["out"])
    r["r8_gen_state_after_eval_equal_old_vs_new"] = bool(torch.equal(o["eval_b"]["gen_state"], n["eval_b"]["gen_state"]))
    r["global_rng_end_equal_old_vs_new"] = bool(torch.equal(o["global_state_end"], n["global_state_end"]))
    # the BASE candidates (columns :nb of every per-candidate tensor) and every non-candidate output, old vs new eval
    base_diff, cand_keys = [], []
    for k, v in o["eval_a"]["out"].items():
        w = n["eval_a"]["out"][k]
        ncand = o["eval_a"]["out"]["out.r8_cand_src"].shape[1]
        if v.dim() >= 2 and v.shape[1] == ncand and ncand > nb:
            cand_keys.append(k)
            if not torch.equal(v[:, :nb], w[:, :nb]):
                base_diff.append(k)
        elif not torch.equal(v, w):
            base_diff.append(k + " (non-candidate tensor)")
    r["eval_old_vs_new_base_columns_or_noncandidate_differ"] = base_diff
    r["eval_per_candidate_keys"] = cand_keys
    rep[f"emit_live_{bool(e)}"] = r
json.dump(rep, open("compare_cases.json", "w"), indent=1)
print(json.dumps(rep, indent=1)[:6000])
