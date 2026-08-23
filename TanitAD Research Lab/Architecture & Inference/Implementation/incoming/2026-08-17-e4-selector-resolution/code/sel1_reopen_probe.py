"""Does SEL-1's PRE-REGISTERED reopening path work AT ALL? Executed, not read.

Plant sigma = 0.30 m -- FAR below the FUNDED line of 0.80 -- run the real
estimator, write the real artifact, and hand it to the real reader.
"""
import pyarrow  # noqa: F401  (must precede torch on this box)
import json, sys, pathlib, tempfile
REPO = pathlib.Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "tests"))
import torch  # noqa
from test_e_wc2_sigma_star import make_dump
import e_wc2_sigma_star as E
import v6_chain as C

tmp = pathlib.Path(tempfile.mkdtemp(prefix="e4reopen"))
PLANTED = 0.30
d = make_dump(sigma2=PLANTED, sigma6=2 * PLANTED, seed=3)
res = E.run(d, features=["pooled", "ctx"], n_boot=0)

top = sorted(res.keys())
ratios = res.get("references_and_ratios", {})

# write it EXACTLY where the chain expects the admission artifact
root = tmp / "experiments"
sw = root / "v6F-SW-30k"
sw.mkdir(parents=True)
art = sw / C.SW_LATENT_ADMISSION["artifact"]
art.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

cfg = C.ChainConfig()
cfg.root = str(root).replace("\\", "/")
adm = C.read_sw_admission(cfg)

# and what the chain then DOES with it
step = C.ChainStep(key="S-T:goal", stage="S-T", arm="goal",
                   out=cfg.path("v6F-ST-10k"), steps=10, lr=1e-4,
                   selector="goal", w_select=1.0,
                   init_from_key="S-W", prev_gate_key="S-W", max_horizon=60)
try:
    C.assert_selector_admissible(step, cfg)
    refusal = None
except C.ChainRefusal as e:
    refusal = str(e).splitlines()[0]

print(json.dumps({
    "planted_sigma_perax_m": PLANTED,
    "estimator_recovered": ratios.get("sigma_perax_2s_m"),
    "estimator_top_level_keys": top,
    "reader_expects_field": C.SW_LATENT_ADMISSION["field"],
    "field_present_at_top_level": C.SW_LATENT_ADMISSION["field"] in res,
    "field_present_anywhere": any(
        C.SW_LATENT_ADMISSION["field"] in (v if isinstance(v, dict) else {})
        for v in res.values()),
    "read_sw_admission": {k: adm.get(k) for k in
                          ("present", "verdict", "_read")},
    "selector_launch_refused": refusal is not None,
    "refusal_first_line": refusal,
}, indent=1))
