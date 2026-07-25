"""TanitEval — data layer.

Loads raw-frame validation episodes (canonical epcache format) and prepares the
per-architecture model inputs:
  * flagship / refb : raw frame stacks  [T, 9, 256, 256] uint8
  * refa-*          : frozen-encoder token features [T, 256, d] fp16, computed
                      on the fly with the EXACT extraction the training features
                      used (dinov2: torch.hub get_intermediate_layers @224;
                      ijepa: transformers last_hidden_state @224) and cached to
                      disk so repeat runs are free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/TanitAD/stack")
from tanitad.data.mixing import load_episode  # noqa: E402

FEATCACHE = Path("/root/featcache")

# Canonical validation splits (parity is sacred). The CLEAN held-out split is the
# only admissible decision-grade physicalai val set; the f1b378 split LEAKS ~78 %
# of its episodes into the parity TRAIN corpus (physicalai-train-e438721ae894),
# so any number computed on it is train-contaminated. ``list_val_episodes``
# refuses the leaky split by default — this is the single chokepoint every
# decision-grade entrypoint (runner, closedloop, hierarchy, pathspeed,
# efficiency, refc_rerank, planning, planner_p2, bench, generalization) routes
# through, so the refusal covers the whole harness in one place.
CLEAN_VAL = "physicalai-val-0c5f7dac3b11"
LEAKY_VAL = "physicalai-val-f1b378f295ae"

#: Provenance of the LAST ``list_val_episodes`` call, so an emitter can stamp
#: *which* val cache produced a number into its own results JSON. Read by
#: ``bench.run`` consumers via :func:`last_val_parity`.
_LAST_VAL_PARITY: dict = {}


def last_val_parity() -> dict:
    """The integrity record of the most recent val listing (empty before any).

    The blast-radius lesson, mechanised: for every number this program has
    published, "which val cache, how many episodes" had to be RECONSTRUCTED from
    ``n_windows == 881`` because no result JSON recorded it. From now on it is
    recorded."""
    return dict(_LAST_VAL_PARITY)


def _parity_module():
    """``tanitad.data.parity``, imported late and refusing LOUDLY if absent.

    Late because a bare ImportError at module load would take down ``--help``
    and every non-val code path on a pod whose ``stack/`` checkout is behind.
    Refusing because an eval that cannot verify its val cache must not quietly
    produce a number — that is the exact failure this guard exists to close."""
    try:
        from tanitad.data import parity                      # noqa: PLC0415
    except Exception as ex:                                   # pragma: no cover
        raise RuntimeError(
            f"REFUSING to evaluate: the shared parity guard "
            f"(tanitad.data.parity) is not importable ({type(ex).__name__}: "
            f"{ex}). A val cache that cannot be integrity-checked produces a "
            f"plausible-looking WRONG ADE and nothing downstream can detect it. "
            f"Sync stack/ to this machine (PYTHONPATH must include "
            f"<repo>/stack) and re-run.") from ex
    return parity


def list_val_episodes(val_dir: str, n: int | None = None,
                      allow_leaky: bool = False, allow_partial: bool = False,
                      label: str | None = None):
    """Sorted ``ep_*.pt`` files under ``val_dir`` (first ``n`` if given).

    THE val-side integrity chokepoint. Before any episode is read it asserts,
    via the ONE shared guard ``tanitad.data.parity`` (same module the trainers
    use — there is deliberately no second implementation):

      * the split is not the known-leaky ``f1b378`` corpus;
      * the cache's episode count is a **registered deployment** of the clean
        val split (600 full build / 40 canonical TanitEval — manifest
        ``known_deployments``), so a truncated or partial val cache is refused
        instead of silently rescoring the benchmark on fewer episodes;
      * the uid ``sha256`` matches the committed manifest **when the manifest
        carries one**. It does not yet: no committed artifact enumerates the val
        uids and the Wave-1 B agent correctly refused to invent a digest. The
        one command that upgrades this to a content check, run on a pod whose
        ``compute_skipset.py`` has just printed ``VERDICT MATCH``::

            PYTHONPATH=/workspace/TanitAD/stack python3 \\
              stack/scripts/make_parity_manifest.py --record --split val \\
              --cache-dir <epcache>/physicalai-val-0c5f7dac3b11

        then bring the changed ``parity_manifest.json`` back and stage it. Until
        that lands this is COUNT + CACHE-IDENTITY, and it says so in its own log
        line.
      * ``n`` episodes were actually available (asking for 40 from a 12-episode
        deployment used to return 12 and score it, silently).

    ``allow_leaky=True`` — ONLY for a deliberate label/leakage audit.
    ``allow_partial=True`` — ONLY for a deliberate non-decision-grade probe on a
    partial deployment; the run is stamped ``decision_grade: False``.
    """
    global _LAST_VAL_PARITY
    if LEAKY_VAL in str(val_dir) and not allow_leaky:
        raise RuntimeError(
            f"REFUSED leaky val split {LEAKY_VAL!r}: ~78 % of its episodes leak "
            f"into the parity train corpus e438721ae894, so every number on it "
            f"is train-contaminated. Use the CLEAN held-out split {CLEAN_VAL!r} "
            f"(taniteval.data.CLEAN_VAL). Pass allow_leaky=True only for a "
            f"deliberate leakage/label audit.")
    lbl = label or f"val {Path(val_dir).name}"
    if LEAKY_VAL in str(val_dir):        # allow_leaky: audit path, stays loud
        print(f"[parity] ⚠ {lbl}: LEAKY split {LEAKY_VAL} accepted because "
              f"allow_leaky=True. Admissible ONLY as a label/leakage audit — "
              f"NEVER as a decision-grade number.", flush=True)
        _LAST_VAL_PARITY = {"checked": False, "corpus_key": LEAKY_VAL,
                            "cache_dir": str(val_dir), "leaky": True,
                            "decision_grade": False, "label": lbl}
    else:
        _LAST_VAL_PARITY = _parity_module().assert_val_cache(
            val_dir, label=lbl, requested=n,
            decision_grade=not allow_partial)
    files = sorted(Path(val_dir).glob("ep_*.pt"))
    _LAST_VAL_PARITY["episodes_listed"] = len(files[:n] if n else files)
    return files[:n] if n else files


def load_raw(files):
    """Episodes with raw frames (flagship / refb path)."""
    return [load_episode(str(f), mmap=True) for f in files]


class FeatEp:
    """Episode view exposing frozen-encoder features as .feats (gate EpWrap style)."""

    def __init__(self, feats, ep, eid):
        self.feats = feats                       # [T, 256, d] fp16 (cpu)
        self.actions = ep.actions.float()
        self.poses = ep.poses.float()
        self.episode_id = eid


def _imagenet(latest, size):
    if size != latest.shape[-1]:
        latest = F.interpolate(latest, size=(size, size), mode="bilinear",
                               align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (latest - mean) / std


class FrozenEncoder:
    """Lazy frozen encoder matching the training feature extraction exactly."""

    def __init__(self, kind: str, device="cuda"):
        assert kind in ("dinov2", "ijepa")
        self.kind, self.device, self._m = kind, device, None

    def _model(self):
        if self._m is None:
            if self.kind == "dinov2":
                m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            else:
                from transformers import IJepaModel
                m = IJepaModel.from_pretrained("facebook/ijepa_vith14_1k")
            self._m = m.to(self.device).eval()
        return self._m

    @torch.no_grad()
    def encode_episode(self, ep, batch=16):
        latest = ep.frames[:, -3:].float().div(255.0)      # current RGB [T,3,S,S]
        latest = _imagenet(latest, 224)
        m, toks = self._model(), []
        for i in range(0, latest.shape[0], batch):
            x = latest[i:i + batch].to(self.device)
            if self.kind == "dinov2":
                out = m.get_intermediate_layers(x, n=1)[0]          # [B,256,768]
            else:
                out = m(pixel_values=x).last_hidden_state           # [B,256,1280]
            toks.append(out.half().cpu())
        return torch.cat(toks)

    def free(self):
        self._m = None
        torch.cuda.empty_cache()


def load_features(files, kind: str, device="cuda", verbose=True):
    """FeatEps for the given val files, via the disk cache when possible."""
    cache = FEATCACHE / kind
    enc, out = FrozenEncoder(kind, device), []
    for i, f in enumerate(files):
        # Namespace the cache by the corpus dir: ep_*.pt filenames COLLIDE
        # across corpora (physicalai/comma/cosmos all start at ep_00000.pt),
        # so a bare-filename key would serve physicalai features for comma/
        # cosmos frames. Keying on the parent (val-root) dir disambiguates.
        cf = cache / f.parent.name / f.name
        cf.parent.mkdir(parents=True, exist_ok=True)
        ep = load_episode(str(f), mmap=True)
        if cf.exists():
            feats = torch.load(cf, map_location="cpu", mmap=True,
                               weights_only=True)
        else:
            feats = enc.encode_episode(ep)
            torch.save(feats, cf)
            if verbose and i % 10 == 0:
                print(f"[feat:{kind}] encoded {i}/{len(files)}", flush=True)
        out.append(FeatEp(feats, ep, i))
    enc.free()
    return out


class RawEp:
    """Raw-frame episode view for frame-input models (.feats = frames)."""

    def __init__(self, ep, eid):
        self.feats = ep.frames                   # [T, 9, S, S] uint8 (mmap)
        self.actions = ep.actions.float()
        self.poses = ep.poses.float()
        self.episode_id = eid


def load_frames(files):
    return [RawEp(load_episode(str(f), mmap=True), i)
            for i, f in enumerate(files)]
