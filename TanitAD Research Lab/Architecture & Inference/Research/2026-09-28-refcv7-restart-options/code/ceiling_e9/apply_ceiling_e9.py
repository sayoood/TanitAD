"""SPEC_REFCV7 A2 / PI R1 fix: the speed ceiling must reach the EMITTED plan.

MEASURED 2026-09-28: the NavSim bridge on refcv7 step 1,500 gave bit-identical plans with the
ceiling filter ON and OFF on 204/204 warmup scenes, with 2 emitted plans over the ceiling on
CUDA. Source: the decoder masks only its LOCAL `rank` (refc.py, SpeedCeilingFilter) and returns
`score` unmasked. RefCV3Model's E9 goal selection then re-ranks the fan from that unmasked score
with the reach mask only, so the ceiling never reaches `out["traj"]`.

Fix, inference-only like the filter itself:
- the decoder EXPORTS the mask it applied, as `out["ceil_keep"]` (only when the filter ran, i.e.
  never in training while `speed_ceiling_in_training` is False);
- E9 ranks through `e9_rank(blended, out)`, which applies reach_keep AND ceil_keep. A row with no
  survivor under both keeps its reach-only ranking and is counted in `e9_ceil_dead_frac`.

usage: apply_ceiling_e9.py <tree root containing stack/>. Every replacement must match exactly
once, or the tool refuses and writes nothing. It works in LF space and writes each file back in
its own EOL.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
F_DEC = root / "stack/tanitad/refs/refc.py"
F_V3 = root / "stack/tanitad/refs/refc_v3.py"

DEC = [
    ("        rank = score\n        reach_keep = None\n",
     "        rank = score\n        reach_keep = None\n        ceil_keep = None      # SPEC_REFCV7 A2: exported so E9 applies the same ceiling\n"),
    ("            rank = rank.masked_fill(~_keep, float('-inf'))\n            tele.update(_st)\n",
     "            rank = rank.masked_fill(~_keep, float('-inf'))\n            ceil_keep = _keep\n            tele.update(_st)\n"),
    ("            out[\"reach_keep\"] = reach_keep\n        return out\n",
     "            out[\"reach_keep\"] = reach_keep\n"
     "        if ceil_keep is not None:\n"
     "            # ⛔ SPEC_REFCV7 A2 / PI R1: the set speed is a HARD cap on the EMITTED plan. The\n"
     "            # argmax above is not the emitted plan on a v3 build: RefCV3Model's E9 goal\n"
     "            # selection re-ranks the fan from `score`, which is returned UNMASKED. Without\n"
     "            # this export the ceiling never reached `out[\"traj\"]` (MEASURED 2026-09-28:\n"
     "            # filter ON == OFF on 204/204 NavSim warmup scenes at refcv7 step 1,500).\n"
     "            out[\"ceil_keep\"] = ceil_keep\n"
     "        return out\n"),
    ("        \"prefinal_logits\", \"reach_keep\", \"layer_u0_hat\", \"layer_logits\",\n",
     "        \"prefinal_logits\", \"reach_keep\", \"layer_u0_hat\", \"layer_logits\",\n"
     "        # SPEC_REFCV7 A2 / PI R1: the ceiling mask the decoder's argmax used, so E9 can apply\n"
     "        # it to the EMITTED plan. Present only when the filter ran (inference).\n"
     "        \"ceil_keep\",\n"),
]

V3 = [
    ("        rank = blended\n"
     "        if \"reach_keep\" in out:            # post-guard mask (dead rows full)\n"
     "            rank = blended.masked_fill(~out[\"reach_keep\"], float(\"-inf\"))\n"
     "        idx = rank.argmax(dim=1)\n",
     "        rank, e9_tele = e9_rank(blended, out)\n"
     "        out.update(e9_tele)\n"
     "        idx = rank.argmax(dim=1)\n"),
    ("class RefCV3Model(nn.Module):\n",
     "def e9_rank(blended: Tensor, out: dict) -> tuple[Tensor, dict]:\n"
     "    \"\"\"E9's selection mask: the decoder's reach mask AND its speed-ceiling mask.\n"
     "\n"
     "    ⛔ SPEC_REFCV7 A2 / PI R1: the fed set speed is a HARD cap on the EMITTED plan.\n"
     "    `ceil_keep` exists only when the decoder's ceiling filter ran (inference; never in\n"
     "    training while `speed_ceiling_in_training` is False), so training is unchanged.\n"
     "    A row with no candidate under both masks keeps its reach-only ranking (the decoder's\n"
     "    own empty-row rule: an unsatisfiable window is a measurement failure, not a licence\n"
     "    to emit nothing) and is COUNTED, so obedience below 1.0 is always explained.\n"
     "    \"\"\"\n"
     "    rank = blended\n"
     "    tele = {}\n"
     "    if \"reach_keep\" in out:            # post-guard mask (dead rows full)\n"
     "        rank = blended.masked_fill(~out[\"reach_keep\"], float(\"-inf\"))\n"
     "    ck = out.get(\"ceil_keep\")\n"
     "    if ck is not None:\n"
     "        both = rank.masked_fill(~ck, float(\"-inf\"))\n"
     "        dead = ~torch.isfinite(both).any(dim=1)\n"
     "        rank = torch.where(dead[:, None], rank, both)\n"
     "        tele[\"e9_ceil_dead_frac\"] = dead.float().mean().detach()\n"
     "    return rank, tele\n"
     "\n"
     "\n"
     "class RefCV3Model(nn.Module):\n"),
]


def patch(path, reps):
    raw = path.read_bytes()
    crlf = b"\r\n" in raw
    s = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in reps:
        n = s.count(old)
        if n != 1:
            sys.exit(f"REFUSED {path.name}: {n} matches for {old[:60]!r}")
        s = s.replace(old, new)
    return s.replace("\n", "\r\n") if crlf else s, crlf


out = [(F_DEC, *patch(F_DEC, DEC)), (F_V3, *patch(F_V3, V3))]
for path, text, crlf in out:          # write only after BOTH files patched cleanly
    path.write_bytes(text.encode("utf-8"))
    print("PATCHED", path.relative_to(root), "crlf" if crlf else "lf")
