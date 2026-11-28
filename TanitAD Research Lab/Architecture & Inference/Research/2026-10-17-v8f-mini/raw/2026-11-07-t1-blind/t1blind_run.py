"""t1blind_run: the T1-kin rollout of PREREG_T1_BLIND.md (arm BLIND1_s), a THIN WRAPPER around the pinned RIC0 tool `v8f_t1_eval_ri.py` (sha256 0417c838...). The tool is loaded BY FILE PATH after its sha256 holds
and is not copied or edited; the wrapper patches two names in the loaded module object and nothing else:
  1. `build_item` (the tool's item builder, a module global that `rollout` calls): the ORIGINAL runs first (so `--scene reindexed` and `--limit causal` are the tool's own), then, in mode `blind`, ONLY the key `feats` is
     re-assigned: every row of the item's per-frame scene tokens := the pinned BLIND mean (eval-141 cache mean, fp16 [40, 1024], sha256 cd6a7ab5...), repeated over the hmax rows. `frame_ok`, `p10f`, `nav`, `base`, `limit`, `pose*`, ages
     (the ego inputs) are the very objects the original built; the wrapper counts any item whose other keys changed identity (G-EGO) and refuses to continue.
  2. `V8FMini.forward_infer` (class level): a RECEIPT at the point where the policy RECEIVES its batch, before the model reads it. Per call it compares every received `feats` row to the pinned mean BITWISE (int16 view; G-MEAN) and
     digests every received non-`feats` input per row (G-EGO). It then calls the original unchanged.
Modes:  off = the replacement is switched off (the receipt still runs): the rollout must reproduce RIC0_s bit for bit (G-ID).   blind = BLIND1.
        mutant = the wrapper path runs but LEAVES THE TOKENS UNCHANGED (the card's G-MEAN mutant: the gate must FAIL on its receipt).   mutant-ego = replaces the tokens AND perturbs one ego input (p10f speed channel by +1e-3):
        the G-EGO mutant (the gate must FAIL on its tick-0 digests).
Everything else is the tool's CLI: `t1blind_run.py --blind-mode blind --receipt-dir D -- <the tool's `run` flags>`.   A receipt is written even when the run is a smoke (`--max-windows`).
Tier T1-kin. Nothing here trains or reads a GPU (CUDA hidden)."""
import os
import sys
import json
import hashlib
import argparse
import importlib.util

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
TOOL = "F:/TanitAD-scratch/motion_c/ri/tree_ri/stack/scripts/v8f_t1_eval_ri.py"
TOOL_SHA = "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7"
MEAN_PT = f"{HERE}/blind_mean_pinned.pt"
MEAN_SHA = "cd6a7ab5e3c5450ec3528889b04e06e5605fea30ea9ff02a0e36d9ba47a2fcb9"       # PINS_E4: sha256 of the fp16 [40, 1024] tensor BYTES (not of the .pt file)
MODES = ("off", "blind", "mutant", "mutant-ego")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_mean(path=MEAN_PT):
    import torch
    m = torch.load(path, map_location="cpu")
    assert tuple(m.shape) == (40, 1024) and m.dtype == torch.float16, (tuple(m.shape), m.dtype)
    got = hashlib.sha256(m.numpy().tobytes()).hexdigest()
    if got != MEAN_SHA:
        raise SystemExit(f"REFUSED: the BLIND mean tensor is {got}, not the pinned {MEAN_SHA}")
    return m


def load_tool():
    got = sha256_file(TOOL)
    if got != TOOL_SHA:
        raise SystemExit(f"REFUSED: the tool is {got}, not the pinned {TOOL_SHA}")
    spec = importlib.util.spec_from_file_location("v8f_t1_eval_ri_pinned", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v8f_t1_eval_ri_pinned"] = mod
    spec.loader.exec_module(mod)
    return mod


class Receipt:
    """What the policy received, counted at the forward_infer entry (bitwise) and at the item builder (identity of the non-feats keys)."""

    def __init__(self, mode, mean_t, hm_expected):
        self.mode, self.mean_t = mode, mean_t
        self.mean_i16 = mean_t.view(__import__("torch").int16)
        self.c = {"forward_calls": 0, "rows_received": 0, "frames_received": 0, "frames_equal_mean_bitwise": 0, "frames_not_equal_mean": 0,
                  "items_built": 0, "items_replaced": 0, "items_other_keys_changed": 0, "items_keyset_changed": 0}
        self.tick, self.win, self.dig, self.fdig = [], [], [], []
        self.chunk_base, self.rows_in_run = 0, 0
        self.ticks_per_chunk = None

    def digest_rows(self, batch):
        """Per-row sha256 over every received input except `feats` (sorted key order): the ego inputs, the validity mask, the nav, the ages."""
        import torch
        B = int(batch["feats"].shape[0])
        keys = sorted(k for k in batch if k != "feats")
        cols = []
        for k in keys:
            t = batch[k]
            t = t if torch.is_tensor(t) else torch.as_tensor(np.asarray(t))
            t = t.detach().cpu().contiguous()
            cols.append((k, t))
        out = []
        for b in range(B):
            h = hashlib.sha256()
            for k, t in cols:
                h.update(k.encode()); h.update(np.ascontiguousarray(t[b].numpy()).tobytes())
            out.append(h.hexdigest())
        return out

    def on_forward(self, batch):
        import torch
        f = batch["feats"]
        B, F = int(f.shape[0]), int(f.shape[1])
        eq = (f.view(torch.int16) == self.mean_i16[None, None]).flatten(2).all(-1)                 # [B, F] bitwise
        n_eq = int(eq.sum())
        self.c["forward_calls"] += 1
        self.c["rows_received"] += B
        self.c["frames_received"] += B * F
        self.c["frames_equal_mean_bitwise"] += n_eq
        self.c["frames_not_equal_mean"] += B * F - n_eq
        n = self.c["forward_calls"] - 1
        d = self.digest_rows(batch)
        self.tick.extend([n % self.ticks_per_chunk] * B)
        if n % self.ticks_per_chunk == 0:
            self.chunk_base = self.rows_in_run
            self.rows_in_run += B
        self.win.extend([self.chunk_base + b for b in range(B)])
        self.dig.extend(d)

    def save(self, path_json, path_npz, extra):
        os.makedirs(os.path.dirname(path_json), exist_ok=True)
        json.dump({**extra, "mode": self.mode, **self.c}, open(path_json, "w", encoding="utf-8", newline="\n"), indent=1)
        np.savez(path_npz, tick=np.asarray(self.tick, np.int16), win=np.asarray(self.win, np.int32), digest=np.asarray(self.dig, "S64"))


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--blind-mode", choices=MODES, required=True)
    ap.add_argument("--receipt-dir", required=True)
    ap.add_argument("--mean", default=MEAN_PT)
    a, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    mean_t = load_mean(a.mean)
    tool = load_tool()
    import torch
    mean_np = mean_t.numpy()
    mode = a.blind_mode
    tag = rest[rest.index("--tag") + 1] if "--tag" in rest else "run"
    steps = int(rest[rest.index("--replan-steps") + 1]) if "--replan-steps" in rest else 1
    R = Receipt(mode, mean_t, None)
    R.ticks_per_chunk = tool.H // steps
    orig_build = tool.build_item

    def build_item(ds, proxy, st, tick, a_t, a_s, scene, limit, stats=None):
        it = orig_build(ds, proxy, st, tick, a_t, a_s, scene, limit, stats)
        R.c["items_built"] += 1
        ids0 = {k: id(v) for k, v in it.items()}
        if mode in ("blind", "mutant-ego"):
            f = it["feats"]
            assert f.dtype == np.float16 and f.shape[1:] == (40, 1024), (f.dtype, f.shape)
            it["feats"] = np.repeat(mean_np[None], f.shape[0], axis=0)                      # <<REPLACE>> every row of the item's scene tokens := the pinned mean
            R.c["items_replaced"] += 1
        if mode == "mutant-ego":                                                            # the G-EGO mutant: one ego input perturbed
            p = np.array(it["p10f"], copy=True); p[..., 0] += 1e-3; it["p10f"] = p
        ids1 = {k: id(v) for k, v in it.items()}
        if set(ids0) != set(ids1):
            R.c["items_keyset_changed"] += 1
        chg = [k for k in ids0 if k != "feats" and ids0[k] != ids1.get(k)]
        if chg:
            R.c["items_other_keys_changed"] += 1
            if mode != "mutant-ego":
                raise RuntimeError(f"REFUSED: the wrapper changed non-feats keys {chg}")
        return it

    tool.build_item = build_item
    orig_fi = tool.M.V8FMini.forward_infer

    def forward_infer(self, batch):
        R.on_forward(batch)
        return orig_fi(self, batch)

    tool.M.V8FMini.forward_infer = forward_infer
    rc = tool.main(["run"] + rest)
    pj, pn = f"{a.receipt_dir}/receipt_{tag}.json", f"{a.receipt_dir}/ego_digests_{tag}.npz"
    R.save(pj, pn, {"tag": tag, "tool": TOOL, "tool_sha256": TOOL_SHA, "mean_pt": a.mean, "mean_tensor_sha256": MEAN_SHA, "wrapper_sha256": sha256_file(__file__), "argv": rest, "tool_rc": rc})
    print(f"RECEIPT {tag} mode={mode}: frames_received {R.c['frames_received']} equal_mean {R.c['frames_equal_mean_bitwise']} items_replaced {R.c['items_replaced']} other_keys_changed {R.c['items_other_keys_changed']}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
