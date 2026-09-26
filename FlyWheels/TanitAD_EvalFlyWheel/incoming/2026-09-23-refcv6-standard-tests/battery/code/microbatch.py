"""Run ONE `model(...)` call over B rows as several sub-calls, and hand the caller ONE merged `out`.

Why: the trainer's in-run eval computes `compute_losses_v3` over batch 16. On the 8 GB dev-box card a
16-row refcv6 forward does not fit (frames 1.96 GB fp32 before the trunk normalises them twice), but
every LOSS REDUCTION must still run over the same 16 rows or the batch-level ratios
(`sum(err*sv)/sum(sv)`, masked CEs, set-loss normalisers) are not the numbers the run logged. So the
split happens INSIDE the forward: `compute_losses_v3` is unmodified and sees one 16-row `out`.

Merge rules (recorded per key path, so a reviewer can audit them):
  * tensor whose dim 0 == its OWN micro-batch size in EVERY sub-call   -> torch.cat on dim 0
    (the sizes are deliberately UNEQUAL, e.g. 3,3,3,3,4, so a fixed-size non-batch tensor can
    never pass this test by coincidence);
  * 0-dim tensor / python float                                          -> row-weighted mean
    (exact for a per-row mean; FLAGGED as `wmean` so any other statistic is visible);
  * everything else (non-batch tensors, ints, bools, strings)           -> taken from sub-call 0,
    and FLAGGED `shared-DIFFER` if the sub-calls disagree;
  * dict / list / tuple                                                  -> recursive.
The per-call ego window (`core.set_ego_window`, a one-shot POP channel) is split the same way and
passed explicitly as `ego_poses`/`ego_n_past`, which `refc.py:4409-4412` treats identically.
"""
from __future__ import annotations

import torch

_BATCH_KW = ("frames", "nav_cmd", "v0", "lan", "nav_known", "ego_state", "withheld_speed",
             "gp_point", "gp_valid", "nav_args", "v_max_ms", "v_max_valid", "ego_poses",
             "perception_grid", "perception_valid")


def _slice(x, sl):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x[sl]
    if isinstance(x, dict):
        return {k: _slice(v, sl) for k, v in x.items()}
    raise TypeError(f"cannot slice {type(x)}")


class MicroBatchForward:
    def __init__(self, model, sizes):
        self.model = model
        self.sizes = [int(s) for s in sizes]
        if len(set(self.sizes)) < 2 and len(self.sizes) > 1:
            raise ValueError("micro-batch sizes must not all be equal (batch-dim detection)")
        self.rules: dict = {}
        self.n_calls = 0
        self._orig = None

    # -- install / remove ------------------------------------------------------------- #
    def install(self):
        if self._orig is not None:
            return self
        self._orig = self.model.forward          # bound method of the class
        self.model.forward = self._forward       # instance attribute: nn.Module.__call__ uses it
        return self

    def remove(self):
        if self._orig is not None:
            del self.model.forward
            self._orig = None

    # -- the split forward ------------------------------------------------------------ #
    def _forward(self, frames, *args, **kw):
        if args:
            raise TypeError("MicroBatchForward: pass everything but `frames` by keyword")
        b = int(frames.shape[0])
        if sum(self.sizes) != b:
            raise ValueError(f"micro sizes {self.sizes} sum to {sum(self.sizes)} != batch {b}")
        self.n_calls += 1
        core = self.model.core
        ego = None
        if getattr(core, "ego_hist", None) is not None and kw.get("ego_poses") is None:
            ego = getattr(core, "_ego_window", None)
            if ego is None:
                raise ValueError("ego-history build but no ego window was set before the forward")
            core._ego_window = None          # consumed here, exactly as the core would POP it
        parts, starts = [], []
        s0 = 0
        for m in self.sizes:
            sl = slice(s0, s0 + m)
            sub = {}
            for k, v in kw.items():
                sub[k] = _slice(v, sl) if (k in _BATCH_KW or k == "agent_gt") else v
            if ego is not None:
                sub["ego_poses"] = ego[0][sl]
                sub["ego_n_past"] = ego[1]
            parts.append(self._orig(frames[sl], **sub))
            starts.append(s0)
            s0 += m
        return self._merge(parts, "out")

    def _merge(self, parts, path):
        first = parts[0]
        if torch.is_tensor(first):
            if first.dim() >= 1 and all(torch.is_tensor(p) and p.dim() >= 1 and p.shape[0] == m
                                        for p, m in zip(parts, self.sizes)):
                self.rules[path] = "cat"
                return torch.cat(parts, 0)
            if first.dim() == 0:
                self.rules[path] = "wmean"
                w = torch.tensor(self.sizes, dtype=torch.float64)
                vals = torch.stack([p.detach().to(torch.float64).cpu() for p in parts])
                return (vals * w).sum().div(w.sum()).to(first.dtype).to(first.device)
            same = all(torch.is_tensor(p) and p.shape == first.shape and torch.equal(p, first)
                       for p in parts[1:])
            self.rules[path] = "shared" if same else "shared-DIFFER"
            return first
        if isinstance(first, dict):
            keys = list(first.keys())
            for p in parts[1:]:
                if set(p.keys()) != set(keys):
                    self.rules[path] = "dict-KEYS-DIFFER"
            return {k: self._merge([p.get(k) for p in parts], f"{path}.{k}") for k in keys}
        if isinstance(first, (list, tuple)):
            n = len(first)
            if any(len(p) != n for p in parts):
                self.rules[path] = "seq-LEN-DIFFER"
                return first
            out = [self._merge([p[i] for p in parts], f"{path}[{i}]") for i in range(n)]
            return type(first)(out) if isinstance(first, tuple) else out
        if isinstance(first, float) and not isinstance(first, bool):
            self.rules[path] = "wmean"
            tot = sum(float(p) * m for p, m in zip(parts, self.sizes))
            return tot / float(sum(self.sizes))
        if first is None:
            self.rules[path] = "none"
            return None
        same = all(p == first for p in parts[1:])
        self.rules[path] = "shared" if same else "shared-DIFFER"
        return first

    def flagged(self) -> dict:
        return {k: v for k, v in self.rules.items() if v not in ("cat", "none", "shared")}
