"""X3 finding -> the next lever, MEASURED without editing any file.

The v7f dry S-T records `tactical_to_below` = [nav.embed.weight, nav.arg_proj.weight, nav.arg_proj.bias]:
the tactical layer's nav term is gate * layer_proj.tactical(embed(tok) + arg_proj(args)), and the SHARED
code is grouped predictor_op (BELOW layer_tac). Lever: detach the shared code on the NON-operative
layers (the operative path -- which is what trains the shared code in S-W -- is untouched).

Arms, same argv (the dry ladder's S-T line), same seed, in-process through the REAL trainer main():
  A  as merged                 -> expect the 3 violations
  B  detach on tac/str (patch) -> expect 0 violations AND bit-identical S-T loss rows (the shared code is
                                  FROZEN in S-T, so no trainable parameter's gradient can change)
usage: python x3_nav_lever.py <dry_transcript.json> <out_dir>
"""
import json
import sys
from pathlib import Path

tr, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
STACK = Path("<V7F_CHAIN>/stack")
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))
import train_v6_staged as T                                     # noqa: E402
from tanitad.models import nav_conditioning as NC                # noqa: E402

st = next(r for r in tr["steps"] if r["step"] == "S-T")
sw = next(r for r in tr["steps"] if r["step"] == "S-W")


def with_out(argv, d):
    a = list(argv)
    a[a.index("--out") + 1] = str(d)
    return a


orig_fwd = NC.NavConditioner.forward


def detached_fwd(self, token_id, args, layer):
    code = self.encode(token_id, args)
    if layer != "operative":
        code = code.detach()
    return self.gate[layer] * self.layer_proj[layer](code)


res = {}
for arm, fwd in (("A_as_merged", orig_fwd), ("B_detach_shared_code_on_tac_str", detached_fwd)):
    NC.NavConditioner.forward = fwd
    try:
        for stage, argv in (("S-W", sw["argv"]), ("S-T", st["argv"])):
            d = out / arm / stage
            import torch
            torch.manual_seed(20260927)   # the SAME global RNG state for both arms' builds
            rc = T.main(with_out(argv, d))
            dr = json.loads((d / "dry_run.json").read_text())
            res.setdefault(arm, {})[stage] = {
                "rc": rc, "isolation_pass": dr["isolation"]["pass"],
                "violations": dr["isolation"].get("violations", {}),
                "loss_rows": [r.get("loss") for r in dr["steps"]],
                "t1_rows": [r.get("t1_latent") for r in dr["steps"]],
                "gate_X3": json.loads((d / "stage_gate.json").read_text())["probes"]
                .get("X3_isolation", {}).get("pass")}
    finally:
        NC.NavConditioner.forward = orig_fwd
for stage in ("S-W", "S-T"):
    a, b = res["A_as_merged"][stage], res["B_detach_shared_code_on_tac_str"][stage]
    res[f"{stage}_loss_rows_bit_identical"] = a["loss_rows"] == b["loss_rows"]
(out / "x3_nav_lever.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
