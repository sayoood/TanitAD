"""Dump the derived channel sets and requirement maps, for the package record.

⛔ Reads the models' OWN signatures. Nothing here is typed by hand.
"""
import inspect
import json
import sys

from tanitad.channel_admissibility import excluded_channels
from tanitad.refs import refc
from tanitad.refs import refc_v3 as v3
from tanitad.rl import refc_adapter as ra
from tanitad.rl import refcv3_adapter as ad

out = {}

sig_refc = list(inspect.signature(refc.RefCModel.forward).parameters)
sig_v3 = list(inspect.signature(v3.RefCV3Model.forward).parameters)
out["signatures"] = {
    "refc.RefCModel.forward": sig_refc,
    "refc_v3.RefCV3Model.forward": sig_v3,
    "only_on_RefCModel": sorted(set(sig_refc) - set(sig_v3)),
    "only_on_RefCV3Model": sorted(set(sig_v3) - set(sig_refc)),
}
out["excluded_by_admissibility"] = sorted(excluded_channels())

m_refc = refc.RefCModel(refc.refc_smoke_config())
cfg_v0 = refc.refc_smoke_config()
cfg_v0.sel_reach_clamp = True
m_refc_v0 = refc.RefCModel(cfg_v0)
m_v3 = v3.RefCV3Model(v3.refc_v3_smoke_config(hier=False))

out["derived_channels"] = {
    "refc.RefCModel (the PILOT's family)":
        list(ad.forward_conditioning_channels(m_refc)),
    "refc_v3.RefCV3Model (this adapter's test caller)":
        list(ad.forward_conditioning_channels(m_v3)),
    "refc_adapter.FORWARD_KEYS (hand-kept, RefCV3Model only)":
        list(ra.FORWARD_KEYS),
}
out["plumbed_by_this_adapter"] = list(ad.PLUMBED_CHANNELS)

out["requirement_maps"] = {
    "refc_smoke_config() — the pilot's config shape, defaults":
        ad.conditioning_requirements(m_refc),
    "refc_smoke_config() + sel_reach_clamp=True":
        ad.conditioning_requirements(m_refc_v0),
    "refc_v3_smoke_config() — core.sel_reach_clamp=True at refc_v3.py:751":
        ad.conditioning_requirements(m_v3),
}

# ⛔ THE REPOINT, EXECUTED — the proposal this package rejected.
try:
    ra.conditioning_requirements(m_refc_v0)
    out["repoint_refc_adapter_onto_RefCModel"] = "⛔ DID NOT RAISE — re-derive!"
except Exception as exc:
    out["repoint_refc_adapter_onto_RefCModel"] = {
        "raised": type(exc).__name__, "message": str(exc)[:400]}

out["declarations"] = ad.requirement_report()
out["declarations_in_scope_for_the_pilots_family"] = [
    d["channel"] for d in ad.requirement_report(m_refc)]

json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1,
          ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items()
                  if k != "declarations"}, indent=1, ensure_ascii=False))
