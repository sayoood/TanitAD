"""PREREG_T2_DAGGER (DT2): the T2 own-state bank and the swap that draws WHOLE bank samples into a training batch.

A bank row is a whole sample (card log 2026-11-07 14:33, reading 1): the inputs the planner consumed at one plan tick of a stage-4 T2 rollout (``INPUT_KEYS``: the tokens of the window's frames
rendered at the EXECUTED poses, the executed ego history, nav, limit, ages, poses) plus the ingredients of RECOVER-FT's target law (``gt_logged`` = the logged future in the logged NOW frame, ``dy`` / ``dpsi`` =
the executed pose in the logged frame), and the provenance (scene id, fold, seed, start, tick).
THE TARGET is ``R.rejoin_target(gt_logged, dy, dpsi, mode, steps)``: the SAME function RECOVER-FT calls, so there is one target law. It is the ONLY supervision on a bank row (reading 2): the swap sets
``gt_ok`` True and masks every other term -- ``lab`` / ``lab_t`` / ``lab_s`` ``has`` False (ckpt, goal, lat, lon, cons, teacher forcing), ``rc_ok`` False, ``launch`` / ``level_ok`` / ``lead_has`` /
``lead_near`` False when present -- and adds ``aux_ok`` [B] bool (False on bank rows), which ``V8FMini.loss`` reads to gate acc, the consistency penalties and the WM auxiliary.
``p == 0``: nothing is read, nothing is drawn, ``apply`` returns the SAME batch object. DETERMINISM: a private generator seeded from a domain-separated hash of (seed, step).
On disk: ``bank.json`` (manifest: schema, n, shard file name + sha256, per-row meta) + ``rows.pt`` (torch.save of ``{key: stacked tensor}``). The bank's identity is the manifest's sha256.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

SCHEMA = "t2bank-v1"
#: what the planner consumed at the tick (``V8FMini.INFERENCE_INPUT_KEYS``) -- stacked, one row per plan tick
INPUT_KEYS = ("feats", "frame_ok", "p10f", "act", "nav", "pose", "limit", "limit_ok", "base", "a_t", "a_s", "pose_now", "pose_t", "pose_s", "kin3", "p10_t")
TARGET_KEYS = ("gt_logged", "dy", "dpsi")
ROW_KEYS = INPUT_KEYS + TARGET_KEYS
T2_DEFAULTS = {"p": 0.0, "seed": 0, "target": "rejoin", "rejoin_steps": 10}
#: the label masks a bank row clears (batch key -> sub key or None), when the key is in the batch
MASKED_LABELS = (("lab", "has"), ("lab_t", "has"), ("lab_s", "has"), ("lab_s", "rc_ok"))
MASKED_FLAGS = ("launch", "level_ok", "lead_has", "lead_near")
SHARD = "rows.pt"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class BankRefused(Exception):
    pass


class T2Bank:
    """The loaded bank: ``a`` = {key: tensor [N, ...]}, ``meta`` = list of per-row dicts (scene_id, fold, seed, f0, tick ...), ``sha256`` = the manifest's, ``path`` = the manifest path."""

    def __init__(self, a: dict, meta: list, *, sha256=None, path=None):
        miss = [k for k in ROW_KEYS if k not in a]
        if miss:
            raise BankRefused(f"t2 bank lacks {miss}")
        n = int(a["feats"].shape[0])
        if n == 0 or any(int(v.shape[0]) != n for v in a.values()) or len(meta) != n:
            raise BankRefused(f"t2 bank is empty or its columns / meta disagree on the row count ({n})")
        if any("scene_id" not in m for m in meta):
            raise BankRefused("t2 bank meta lacks scene_id on some row")
        self.a, self.meta, self.sha256, self.path = a, list(meta), sha256, path

    def __len__(self):
        return int(self.a["feats"].shape[0])

    @property
    def scene_ids(self) -> list:
        return sorted({m["scene_id"] for m in self.meta})

    def save(self, out_dir) -> str:
        import torch
        os.makedirs(out_dir, exist_ok=True)
        sp = os.path.join(out_dir, SHARD)
        torch.save({k: v.contiguous() for k, v in self.a.items()}, sp)
        man = {"schema": SCHEMA, "n": len(self), "shard": SHARD, "shard_sha256": sha256_file(sp), "row_keys": list(ROW_KEYS), "scene_ids": self.scene_ids, "rows": self.meta}
        mp = os.path.join(out_dir, "bank.json")
        with open(mp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        self.path, self.sha256 = mp, sha256_file(mp)
        return self.sha256

    @classmethod
    def load(cls, out_dir, expect_sha256=None) -> "T2Bank":
        import torch
        mp = os.path.join(out_dir, "bank.json")
        if not os.path.exists(mp):
            raise BankRefused(f"{out_dir}: no bank.json (a bare shard is not a bank: the identity is the manifest's sha256)")
        sha = sha256_file(mp)
        if expect_sha256 and sha != str(expect_sha256).lower():
            raise BankRefused(f"{mp}: sha256 {sha} != declared {expect_sha256}")
        man = json.load(open(mp, encoding="utf-8"))
        if man.get("schema") != SCHEMA:
            raise BankRefused(f"{mp}: schema {man.get('schema')!r} is not {SCHEMA!r}")
        sp = os.path.join(out_dir, man["shard"])
        if sha256_file(sp) != man["shard_sha256"]:
            raise BankRefused(f"{sp}: shard sha256 differs from the manifest's (a file that exists with changed bytes)")
        a = torch.load(sp, map_location="cpu", weights_only=True)
        if len(man["rows"]) != int(man["n"]):
            raise BankRefused(f"{mp}: n {man['n']} != {len(man['rows'])} meta rows")
        return cls(a, man["rows"], sha256=sha, path=mp)


class T2BankSwap:
    PROBE_STEP = -1

    def __init__(self, p=0.0, bank: T2Bank | None = None, target="rejoin", rejoin_steps=10, seed=0, bank_path=None, bank_sha256=None):
        self.p, self.bank, self.target = float(p), bank, str(target)
        self.rejoin_steps, self.seed = int(rejoin_steps), int(seed)
        self.bank_path = None if bank_path is None else str(bank_path)
        self.bank_sha256 = None if bank_sha256 is None else str(bank_sha256).lower()
        self.last = None
        self.opened = {}                         # scene_id -> number of times a row of that scene was drawn into a batch (the read refuses a scene in here)

    @property
    def active(self) -> bool:
        return self.p > 0.0

    def built_levers(self) -> dict:
        """What the BUILT swap does (the G-DVB 'built' side): the sha256 is the LOADED manifest's, not the declared string."""
        return {"t2_bank_p": self.p, "t2_bank": self.bank_path, "t2_bank_sha256": (None if self.bank is None or self.bank.sha256 is None else str(self.bank.sha256).lower()),
                "t2_bank_seed": self.seed, "t2_bank_target": self.target, "t2_bank_rejoin_steps": self.rejoin_steps, "masked_labels": [list(x) for x in MASKED_LABELS], "masked_flags": list(MASKED_FLAGS)}

    def problems(self) -> list:
        bad = []
        if not (math.isfinite(self.p) and 0.0 <= self.p <= 1.0):
            bad.append(f"t2_bank_p {self.p!r} is not in [0, 1]")
        if self.seed < 0:
            bad.append(f"t2_bank_seed {self.seed!r} is negative")
        if self.active:
            if self.bank is None or len(self.bank) == 0:
                bad.append("t2_bank_p > 0 but no (or an empty) bank: a lever that is declared and has nothing to draw")
            if self.bank_sha256 is None:
                bad.append("t2_bank_p > 0 needs --t2-bank-sha256")
        elif self.bank is not None or self.bank_path is not None or self.bank_sha256 is not None or self.seed != T2_DEFAULTS["seed"]:
            bad.append("t2 bank flags are set while t2_bank_p is 0: a lever that is declared and does nothing")
        return bad

    def _seed_of(self, step: int) -> int:
        h = hashlib.sha256(f"t2bank/{self.seed}/{int(step)}".encode("ascii")).digest()
        return int.from_bytes(h[:8], "little") % (1 << 62)

    def apply(self, batch: dict, step: int, *, p=None) -> dict:
        """The batch with a share ``p`` of its rows replaced by whole bank samples, or ``batch`` itself when p is 0. ``p`` overrides for this call only (the gate's probes)."""
        import torch
        from tanitad.data import v8f_recover as R
        p = self.p if p is None else float(p)
        B = int(batch["feats"].shape[0])
        if not p > 0.0:
            self.last = {"step": int(step), "p": 0.0, "n_rows": B, "n_selected": 0}
            return batch
        saved, self.p = self.p, p
        try:
            bad = self.problems()
        finally:
            self.p = saved
        if bad:
            raise ValueError("t2 bank swap REFUSED: " + "; ".join(bad))
        a, N = self.bank.a, len(self.bank)
        for k in INPUT_KEYS:
            if tuple(a[k].shape[1:]) != tuple(batch[k].shape[1:]):
                raise ValueError(f"t2 bank swap REFUSED: bank {k} {tuple(a[k].shape[1:])} != batch {tuple(batch[k].shape[1:])} (a bank of another hmax / grid)")
        g = torch.Generator().manual_seed(self._seed_of(step))
        u = torch.rand(B, generator=g, dtype=torch.float64)
        r = torch.clamp((torch.rand(B, generator=g, dtype=torch.float64) * N).long(), max=N - 1)
        sel = (u < p).to(batch["feats"].device)
        rows = sel.nonzero(as_tuple=True)[0]
        idx = r[rows.cpu()]
        self.last = {"step": int(step), "p": p, "n_rows": B, "n_selected": int(rows.numel()), "sel": sel.detach().cpu().clone(), "bank_idx": idx.clone()}
        out = dict(batch)
        out["aux_ok"] = ~sel
        if not rows.numel():
            return out
        for k in INPUT_KEYS:
            x = batch[k].clone()
            x[rows] = a[k][idx].to(device=x.device, dtype=x.dtype)
            out[k] = x
        dt = batch["gt"].dtype
        gt = batch["gt"].clone()
        gt[rows] = R.rejoin_target(a["gt_logged"][idx].double(), a["dy"][idx].double(), a["dpsi"][idx].double(), mode=self.target, steps=self.rejoin_steps).to(device=gt.device, dtype=dt)
        out["gt"] = gt
        gok = batch["gt_ok"].clone()
        gok[rows] = True
        out["gt_ok"] = gok
        for top, sub in MASKED_LABELS:
            d = dict(out[top])
            x = d[sub].clone()
            x[rows] = False
            d[sub] = x
            out[top] = d
        for k in MASKED_FLAGS:
            if k in out:
                x = out[k].clone()
                x[rows] = False
                out[k] = x
        for i in idx.tolist():
            sid = self.bank.meta[i]["scene_id"]
            self.opened[sid] = self.opened.get(sid, 0) + 1
        return out

    def opened_record(self) -> dict:
        return {"schema": SCHEMA, "bank_sha256": self.built_levers()["t2_bank_sha256"], "n_scenes_opened": len(self.opened), "scene_ids": sorted(self.opened), "draws": dict(sorted(self.opened.items()))}
