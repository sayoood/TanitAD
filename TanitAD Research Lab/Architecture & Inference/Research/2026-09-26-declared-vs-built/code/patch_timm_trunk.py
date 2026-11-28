"""FIX-5 (per-stage pretrained fingerprint) applied to the LF working copy of
stack/tanitad/models/timm_trunk.py. Exact-match edits; refuses if an anchor moved."""
import sys

P = sys.argv[1]
s = open(P, encoding="utf-8", newline="").read()
assert "\r" not in s


def edit(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[patch] {tag}: anchor found {n} times (need 1)")
    s = s.replace(old, new)


edit('''_PINNED_STATS: dict[str, float] = {
    "resnet34.a1_in1k": IMAGENET_CONV1_ABS_SUM,     # MEASURED 2026-09-16
    "resnet101.a1_in1k": 1402.778076171875,         # MEASURED 2026-09-16
}
''', '''_PINNED_STATS: dict[str, float] = {
    "resnet34.a1_in1k": IMAGENET_CONV1_ABS_SUM,     # MEASURED 2026-09-16
    "resnet101.a1_in1k": 1402.778076171875,         # MEASURED 2026-09-16
}
#: ⛔⛔ FIX-5 (A16 audit Q6, G2 regression R3, 2026-09-26): THE STEM WAS THE WHOLE CHECK, so a
#: trunk with the ImageNet stem and a RANDOM layer4 (a partial load) PASSED. Every STAGE is now
#: fingerprinted: sha256[:16] of the fp32 bytes of the FIRST conv of layer1..layer4, MEASURED
#: 2026-09-26 on the cached timm checkpoints (the built tensor is bit-equal to the file's; banked
#: in `.../2026-09-26-declared-vs-built/raw/stage_fingerprints.json`).
#: ⚠️ WHY A HASH AND NOT A |w|.sum() BAND: on resnet101 `layer1.0.conv1` a He re-init reads
#: 1.04-1.05x the pretrained |w|.sum() -- a 2 % band would pass a random stage on another seed.
#: A backbone with no entry is compared stage by stage against the CHECKPOINT ON DISK instead.
_PINNED_STAGE_SHA16: dict[str, dict[str, str]] = {
    "resnet34.a1_in1k": {"layer1.0.conv1": "f6daa1fb2b054df6",
                         "layer2.0.conv1": "3c633611cfc77356",
                         "layer3.0.conv1": "237623d9929e51c1",
                         "layer4.0.conv1": "b0e7e9ad75f71c5d"},
    "resnet101.a1_in1k": {"layer1.0.conv1": "3cd8d151e1cbfa2e",
                          "layer2.0.conv1": "8cc9f36651fa491e",
                          "layer3.0.conv1": "5c5edf2459e14471",
                          "layer4.0.conv1": "a6c9af2e13eeea37"},
}
''', "pinned stage sha")

edit('''        if not (lo <= got <= hi):
            raise RuntimeError(
                f"refcv6 trunk {model_name!r}: {name}.weight |w|.sum() = "
                f"{got:.4f} is outside [{lo:.4f}, {hi:.4f}] — the ImageNet "
                f"weights were NOT loaded (a He init reads ~875). Refusing to "
                f"train a trunk that claims a prior it does not have.")
        return
''', '''        if not (lo <= got <= hi):
            raise RuntimeError(
                f"refcv6 trunk {model_name!r}: {name}.weight |w|.sum() = "
                f"{got:.4f} is outside [{lo:.4f}, {hi:.4f}] — the ImageNet "
                f"weights were NOT loaded (a He init reads ~875). Refusing to "
                f"train a trunk that claims a prior it does not have.")
        _assert_stages_pretrained(net, model_name)       # FIX-5: every stage, not the stem only
        return
''', "stem pinned branch")

edit('''    if not torch.allclose(stem.weight.detach().float().cpu(), ref.float(),
                          atol=1e-6):
        raise RuntimeError(
            f"refcv6 trunk {model_name!r}: the built {name}.weight does not "
            f"match the pretrained checkpoint on disk — `pretrained=True` did "
            f"not take effect.")
''', '''    if not torch.allclose(stem.weight.detach().float().cpu(), ref.float(),
                          atol=1e-6):
        raise RuntimeError(
            f"refcv6 trunk {model_name!r}: the built {name}.weight does not "
            f"match the pretrained checkpoint on disk — `pretrained=True` did "
            f"not take effect.")
    _assert_stages_pretrained(net, model_name)           # FIX-5: every stage, not the stem only


def _stage_first_convs(net: nn.Module) -> dict[str, tuple[str, nn.Conv2d]]:
    """``{stage: (dotted_name, conv)}`` -- the FIRST conv of every stage, found by SEARCH
    (ResNet ``layerN.``; ConvNeXt ``stages_N.`` / ``stages.N.``), never by a hard-coded path."""
    import re as _re
    out: dict[str, tuple[str, nn.Conv2d]] = {}
    for n, m in net.named_modules():
        g = _re.match(r"^(layer\\d+|stages[._]\\d+)\\.", n)
        if isinstance(m, nn.Conv2d) and g and g.group(1) not in out:
            out[g.group(1)] = (n, m)
    return out


def stage_fingerprints(net: nn.Module) -> dict[str, str]:
    """``{conv_name: sha256[:16] of its fp32 weight bytes}`` for the first conv of each stage."""
    import hashlib
    return {n: hashlib.sha256(m.weight.detach().float().contiguous().cpu().numpy().tobytes()
                              ).hexdigest()[:16]
            for n, m in _stage_first_convs(net).values()}


def _assert_stages_pretrained(net: nn.Module, model_name: str) -> None:
    """⛔ FIX-5: FAIL LOUD unless EVERY stage carries the pretrained weights.

    A pinned backbone is checked by exact fingerprint (:data:`_PINNED_STAGE_SHA16`); any other
    is compared tensor-by-tensor, BY NAME, against the checkpoint timm would load. A backbone
    with no stage found, or whose reference cannot be read, is REFUSED -- a stage that cannot be
    verified is not verified.
    """
    got = stage_fingerprints(net)
    if not got:
        raise RuntimeError(
            f"refcv6 trunk {model_name!r}: no stage convolution found (searched `layerN.` / "
            f"`stagesN.`), so the per-stage pretrained check cannot run. Refusing rather than "
            f"verifying the stem alone -- the partial-load blind spot FIX-5 closes.")
    pinned = _PINNED_STAGE_SHA16.get(str(model_name))
    if pinned is not None:
        bad = {k: (got.get(k), v) for k, v in pinned.items() if got.get(k) != v}
        if bad:
            raise RuntimeError(
                f"refcv6 trunk {model_name!r}: stage weights do NOT match the pinned ImageNet "
                f"fingerprints {bad} ((built, pinned) sha256[:16]). The stem may be right while "
                f"a stage is random -- a PARTIAL load. Refusing to train a trunk that claims a "
                f"prior it only partly has.")
        return
    ref = _reference_state_dict(model_name)
    if ref is None:
        raise RuntimeError(
            f"refcv6 trunk {model_name!r}: no pinned per-stage fingerprint and the reference "
            f"checkpoint could not be read, so the STAGES cannot be verified. Add a measured "
            f"entry to `_PINNED_STAGE_SHA16`, or pass `verify_imagenet_stats=False` and accept "
            f"that the run record cannot claim a prior.")
    for n, m in _stage_first_convs(net).values():
        t = ref.get(n + ".weight")
        if t is None or tuple(t.shape) != tuple(m.weight.shape) or not torch.equal(
                m.weight.detach().float().cpu(), t.detach().float().cpu()):
            raise RuntimeError(
                f"refcv6 trunk {model_name!r}: stage conv {n!r} does not match the pretrained "
                f"checkpoint on disk -- a PARTIAL load (FIX-5).")


def _reference_state_dict(model_name: str) -> dict | None:
    """The state_dict ``timm`` would load for ``model_name``, or ``None``."""
    try:
        import timm
        from timm.models._hub import load_state_dict_from_hf
        cfg = timm.get_pretrained_cfg(model_name)
        hf_id = getattr(cfg, "hf_hub_id", None)
        if not hf_id:
            return None
        return load_state_dict_from_hf(hf_id)
    except Exception:                                      # pragma: no cover
        return None
''', "reference branch + stage check")

open(P, "w", encoding="utf-8", newline="").write(s)
print("timm_trunk patched")
