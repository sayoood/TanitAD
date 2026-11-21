"""Diagnosis of the O-RECOVER gate refusal (2026-10-26 12:20, box 2): "ego probe: base is unchanged on 1 of 29 selected rows".
Runs the trainer UNCHANGED (the tip's file) on the gate's own argv with only --out moved, and wraps ego_perturb_probe: if it refuses, every
selected row is dumped -- kin3 before / after, |base| before / after, whether base changed -- then the refusal is re-raised. Writes nothing
else; the probe batch is the gate's (first batch of sampler.epoch(0)) and the perturbation draws are deterministic in (seed, PROBE_STEP).
usage: python diag_egoprobe.py <gate line file> <out dir> <dump json>"""
import json
import os
import sys

line_file, out_dir, dump = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, r"D:\tanitad\repo\stack\scripts")
sys.path.insert(0, r"D:\tanitad\repo\stack")
import torch  # noqa: E402
import v8f_mini_train as T  # noqa: E402

toks, cur, q = [], "", False
for ch in open(line_file, encoding="utf-8").read().strip():
    if ch == '"':
        q = not q
    elif ch == " " and not q:
        if cur:
            toks.append(cur)
        cur = ""
    else:
        cur += ch
if cur:
    toks.append(cur)
i = [k for k, t in enumerate(toks) if t.endswith("v8f_mini_train.py")][0]
argv = toks[i + 1:]
argv[argv.index("--out") + 1] = out_dir
assert "--selfcheck-only" in argv, "not a gate line"
orig = T.ego_perturb_probe


def wrapped(model, batch, cfg, ego):
    try:
        return orig(model, batch, cfg, ego)
    except AssertionError as e:
        pert = ego.apply(batch, ego.PROBE_STEP, p=1.0)
        sel = ego.last["sel"]
        rows = []
        for r in torch.nonzero(sel).flatten().tolist():
            b0, b1 = batch["base"][r].double(), pert["base"][r].double()
            rows.append({"row": r, "base_changed": bool((b0 != b1).any()), "base_absmax_before": float(b0.abs().max()),
                         "base_absmax_after": float(b1.abs().max()), "kin3_before": [float(x) for x in batch["kin3"][r].reshape(-1)[:3]],
                         "kin3_after": [float(x) for x in pert["kin3"][r].reshape(-1)[:3]]})
        json.dump({"refusal": str(e), "n_selected": len(rows), "unchanged": [x for x in rows if not x["base_changed"]], "rows": rows},
                  open(dump, "w"), indent=1)
        print("DIAG dumped", dump)
        raise


T.ego_perturb_probe = wrapped
print("DIAG argv:", " ".join(argv)[:400])
sys.exit(T.main(argv))
