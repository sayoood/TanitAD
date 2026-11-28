#!/usr/bin/env python3
"""VDAC A5 Stage 0 FORWARD PRODUCER (READINESS item S0 / H6): writes the npz files `vdac_stage0.py` (read, s1f) consumes.  A MEASURING tool, not a training run: needs no launch-gate token and is forbidden in a
launch argv.  It trains nothing, writes nothing into the run directory and modifies no pinned file (scorer_finetune.py, launch_gate.py, gate_checks.py, model.py, train.py are only IMPORTED).

WHAT IT DOES.  Rebuilds P6 + the drivable head EXACTLY as the HA-s0 run did, forwards frames through the trainer's own `fwd()` path (bf16 autocast, eval mode, no grad) and writes the frozen reader's npz schema
(`log, token, step, logit, sdf_m`; logit / sdf_m float32 [N,40,32]; step all 0).  `fwd()` is a closure inside the trainer's main() and cannot be imported, so the head-relevant part of it is replicated here:
`model(img, ego, goal, calib=..., score_extra=cxy)`, the context the scoring decoder reads (a forward pre-hook on `model.score_dec[0]`, the trainer's own tap) and `aux.predict(sctx)`.

  --mode head   the TRAINED head (`aux_drv.pt`, the missing loader):  in-sample frames (the make-list frame list, TRAIN bank rows, step 0 / rank 0)  -> preds_insample.npz
                                                                      the 256 C0-ID frames (stage0_repro256.tsv, HELD-OUT bank rows)             -> preds_repro256.npz
  --mode init   the seed-0 INITIAL head, rebuilt by `build_head(seed 0)` (no weights file):  the pod's head_pred_step0.npz frames (HELD-OUT bank) -> head_init0_rebuilt_a.npz  (s1f --init-rebuilt-a)
                                                                                           the pod's head_pred_h48x.npz frames (TRAIN bank)   -> head_init0_b.npz           (s1f --init-rebuilt-b)
  --mode both   both, ONE model load, every distinct frame forwarded through the trunk once.
  bank-keys     (first argument) the P6 TRAIN bank's (log, token, step, rank) rows as the TSV `vdac_stage0.py make-list --bank-keys` takes; CPU, no model; checked against the run's data event.

REFUSES (no fallback) when: any input is missing; a sha256 differs from the declared one; the run's config cannot be reproduced (the recorded argv does not parse with the trainer's own parser, differs from the
`args` recorded in model_sft_A.pt, or disagrees with aux_drv.pt's recorded head config / the run's data event); the head's strict load fails; the loaded head equals the initial head in EVERY tensor; a frame has no
(or more than one) bank row; the label store / H48 / SFT lists do not verify the frame list; an output already exists; or the GPU is in use by ANY compute process (a P7 trainer) unless --allow-shared-gpu.
Reproduction checks (reported, and rc 3 when they fail, AFTER the files are written): the 256 re-forwarded frames vs the banked head_pred_update10000.npz, and the rebuilt initial head vs the pod's head_pred_step0.npz
(mask-sign agreement >= 99.9 % on each, the consumer's own bars).  rc: 0 ok, 1 refused (SystemExit text), 3 reproduction check failed.
--dry-run validates every input and prints the plan; it never touches CUDA and writes nothing.  --dry-run --with-banks also builds the model (CPU) and the banks and resolves every frame to its row.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFE_DIR = os.path.abspath(os.environ.get("REFE_DIR") or HERE)
for _p in (REFE_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _need in ("model.py", "ckpt_io.py", "aux_drivable_head.py", "scorer_finetune.py", "train.py", "vdac_stage0.py"):
    if not os.path.isfile(os.path.join(REFE_DIR, _need)) and not os.path.isfile(os.path.join(HERE, _need)):
        raise SystemExit(f"REFUSED: {_need} not found in {REFE_DIR} (set REFE_DIR to the refe/ package of the run): no fallback")

import torch  # noqa: E402

import aux_drivable_head as AUX  # noqa: E402
import ckpt_io  # noqa: E402
import vdac_stage0 as VS  # noqa: E402  (the consumer: its list reader / verifier / npz validator are used as is, never copied)
from model import REFe, REFeConfig  # noqa: E402

TOOL = "vdac_stage0_forward/1"
ID_SIGN_AGREE = VS.ID_SIGN_AGREE            # 0.999: the consumer's own C0-ID / S1-F bars
INIT_SIGN_AGREE = VS.INIT_SIGN_AGREE
SKIP_ARGS = ("out", "dump_scores")            # paths that may legitimately differ between the recorded namespace and a re-derived argv
MUTATIONS = ("skip_head_load", "no_guard", "y_flip_output", "misalign_keys", "init_seed1", "skip_list_sha", "skip_gpu_guard")      # deliberate regressions the tests switch on (VDAC_FWD_MUTATE=a,b); a real run never does


def mutation(name):
    return name in os.environ.get("VDAC_FWD_MUTATE", "").split(",")


def refuse(msg):
    raise SystemExit("REFUSED: " + msg)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def need_file(p, what):
    if not p or not os.path.isfile(p):
        refuse(f"{what}: {p!r} is not a file (no fallback)")
    return p


def check_sha(path, declared, what, full=False):
    """declared = the sha256 the operator froze.  full: exactly 64 hex (frame lists); else a prefix of >= 12 hex."""
    d = (declared or "").strip().lower()
    if not d or any(c not in "0123456789abcdef" for c in d) or (len(d) != 64 if full else len(d) < 12):
        refuse(f"{what}: the declared sha256 {declared!r} is not {'64' if full else '>= 12'} hex characters")
    got = sha256_file(path)
    if not got.startswith(d):
        refuse(f"{what}: sha256 {got} does not start with the declared {d}")
    return got


# ======================================================================== the GPU guard
def parse_nvsmi(stdout, rc):
    """-> (apps | None, why).  None = cannot tell (a refusal for a CUDA run: an unverifiable GPU is not an idle one)."""
    if rc != 0:
        return None, f"nvidia-smi rc {rc}"
    return [ln.strip() for ln in stdout.splitlines() if ln.strip()], ""


def gpu_apps():
    exe = os.environ.get("VDAC_NVSMI") or shutil.which("nvidia-smi")
    if not exe:
        return None, "nvidia-smi not found"
    try:
        r = subprocess.run([exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], capture_output=True, text=True, encoding="utf-8", timeout=60)
    except Exception as e:                                                 # noqa: BLE001
        return None, repr(e)
    return parse_nvsmi(r.stdout, r.returncode)


GPU_PROBE = gpu_apps                                                       # tests replace this


def require_gpu_free(device, allow_shared, where):
    """default = REFUSE when any compute process holds the GPU (the P7 trainer).  Returns the apps list seen (recorded in the meta)."""
    if device != "cuda" or mutation("skip_gpu_guard"):
        return []
    apps, why = GPU_PROBE()
    if apps is None:
        if not allow_shared:
            refuse(f"gpu-unverifiable ({where}): {why}; an unverifiable GPU is not an idle one (pass --allow-shared-gpu only if the pod is known idle)")
        return None
    if apps and not allow_shared:
        refuse(f"gpu-busy ({where}): {len(apps)} compute process(es) hold the GPU, e.g. {apps[0]!r}: pause P7 first (refe/vdac_pause_p7.sh pattern) or pass --allow-shared-gpu")
    return apps


# ======================================================================== config reproduction
def parse_run_argv(text):
    """the run's argv, parsed by the TRAINER'S OWN parser (so every default is the trainer's)."""
    import scorer_finetune as SF
    toks = shlex.split(text.strip())
    if not toks:
        refuse("the argv file is empty")
    ap = SF.build_parser()
    try:
        ra = ap.parse_args(toks)
    except SystemExit:
        refuse("the recorded argv does not parse with the trainer's own parser (scorer_finetune.build_parser)")
    bad = AUX.aux_refusals(ra)
    if bad:
        refuse("the recorded argv is inconsistent: " + "; ".join(bad))
    if not ra.aux_drv:
        refuse("the recorded argv has no --aux-drv: this is not a head run")
    if ra.aux_drv_feed != "off" or ra.aux_drv_shuffle_seed is not None:
        refuse("the recorded argv is a feed-on or label-shuffled arm: Stage 0 is for HA-s0 (feed off, true labels)")
    if ra.split_half != -1 or ra.lane_head:
        refuse("the recorded argv carries --split-half or --lane-head: the row selection / model of such a run is not reproduced by this tool")
    return ra


def recorded_args(model_sft_path):
    """vars(a) of the run, as written by the trainer into model_sft_A.pt's meta (only the meta is read: memory-mapped)."""
    need_file(model_sft_path, "the run's model_sft_A.pt (its meta holds the recorded argument namespace)")
    sd = torch.load(model_sft_path, map_location="cpu", weights_only=False, mmap=True)
    try:
        return dict(sd["meta"]["scorer_finetune"]["args"])
    except (KeyError, TypeError):
        refuse(f"{model_sft_path} carries no meta.scorer_finetune.args: the run's recorded namespace is unavailable")


def diff_args(ra, rec):
    a = vars(ra)
    out = []
    for k in sorted(set(a) | set(rec)):
        if k in SKIP_ARGS:
            continue
        if k not in a or k not in rec:
            out.append(f"{k}: {'absent from the re-parsed argv' if k not in a else 'absent from the recorded namespace'}")
        elif a[k] != rec[k]:
            out.append(f"{k}: argv {a[k]!r} != recorded {rec[k]!r}")
    return out


def expected_aux_cfg(ra, seed=None):
    return AUX.AuxDrivableConfig(tap=ra.aux_drv_tap, depth=ra.aux_drv_depth, feed=ra.aux_drv_feed, w_aux=ra.w_aux_drv, seed=int(ra.seed if seed is None else seed))


def read_events(run_dir):
    for nm in ("sft.jsonl", "preflight.jsonl"):
        p = os.path.join(run_dir, nm)
        if os.path.isfile(p):
            return nm, [json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip().startswith("{")]
    refuse(f"{run_dir} holds neither sft.jsonl nor preflight.jsonl: the run's data event (built facts) is unavailable")


def built_facts(aux):
    f = aux.built_facts()
    return {k: f[k] for k in ("depth", "cell", "nx", "ny", "n_params", "ctx_dim", "head_built", "feed_built")}


def load_trained_aux(path, ra, run_dir, ev_name, events):
    """aux_drv.pt -> (AuxDrivable with the saved head loaded STRICTLY, the file's dict).  Every recorded fact is checked against the re-derived config."""
    need_file(path, "aux_drv.pt")
    z = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("format", "head", "feed", "update", "meta"):
        if k not in z:
            refuse(f"aux_drv.pt lacks {k!r}")
    if z["format"] != "aux_drv/1":
        refuse(f"aux_drv.pt format {z['format']!r} is not 'aux_drv/1' (this tool reads HA arms; HB's aux_drv/2 needs its own loader)")
    if z["feed"] is not None:
        refuse("aux_drv.pt carries a feed projection (H2): not an HA head")
    cfg = expected_aux_cfg(ra)
    rec_cfg = z["meta"].get("config")
    if rec_cfg != dataclasses.asdict(cfg):
        refuse(f"aux_drv.pt's recorded head config {rec_cfg} != the config derived from the argv {dataclasses.asdict(cfg)}")
    if z["meta"].get("shuffle_seed") is not None:
        refuse("aux_drv.pt records a label-shuffled head (HA-SHUF): not the head Stage 0 reads")
    if int(z["update"]) != int(ra.max_updates):
        refuse(f"aux_drv.pt is from update {z['update']}, the argv names --max-updates {ra.max_updates}: not the final head")
    man = os.path.join(ra.aux_drv_labels, "manifest.json")
    need_file(man, "the label store manifest named by --aux-drv-labels")
    store_sha = json.load(open(man, encoding="utf-8")).get("store_sha256")
    if store_sha != z["meta"].get("store_sha256"):
        refuse(f"the label store's store_sha256 {store_sha} != the one aux_drv.pt was trained on {z['meta'].get('store_sha256')}")
    aux = AUX.AuxDrivable(cfg)
    if not mutation("skip_head_load"):
        try:
            aux.head.load_state_dict(z["head"], strict=True)
        except RuntimeError as e:
            refuse(f"aux_drv.pt's head does not load strictly into the rebuilt head: {str(e)[:300]}")
    facts = built_facts(aux)
    data_ev = [e for e in events if e.get("event") == "data" and "aux_drv" in e]
    if len(data_ev) != 1:
        refuse(f"{ev_name} holds {len(data_ev)} data events with an aux_drv block (expected exactly 1)")
    rec = data_ev[0]["aux_drv"]
    for k, v in facts.items():
        if k in rec and rec[k] != v:
            refuse(f"the run's data event says aux_drv.{k} = {rec[k]!r}, the rebuilt head has {v!r}")
    for k, want in (("tap", cfg.tap), ("feed", cfg.feed), ("store_sha256", store_sha), ("shuffle_seed", None)):
        if rec.get(k, want) != want:
            refuse(f"the run's data event says aux_drv.{k} = {rec.get(k)!r}, expected {want!r}")
    init = AUX.AuxDrivable(cfg)                                          # a fresh seed build, to prove the loaded head is not still the initial one
    same = [n for (n, p), (_, q) in zip(aux.head.named_parameters(), init.head.named_parameters()) if torch.equal(p, q)]
    total = len(list(aux.head.named_parameters()))
    if len(same) == total and not mutation("no_guard"):
        refuse(f"the loaded head is bit-identical to a freshly built seed-{cfg.seed} head in ALL {total} tensors: the weights were not loaded (or the run never trained)")
    return aux, z, {"tensors_identical_to_init": len(same), "tensors": total, "facts": facts}


def build_init_aux(ra):
    """the seed-0 INITIAL head: the recorded config with seed 0, from build_head's own forked RNG stream (no weights file)."""
    if int(ra.seed) != 0:
        refuse(f"the run's seed is {ra.seed}: its initial head is not the seed-0 head S1-F names")
    return AUX.AuxDrivable(expected_aux_cfg(ra, seed=1 if mutation("init_seed1") else 0))


# ======================================================================== frames
def read_lists_and_verify(a, n_pc):
    excl, h48, sft = VS.excluded_logs_from(a.h48_logs, a.sft_heldout)
    store = VS.Store(a.store, a.store_sha256_prefix)
    out = {"sft": sft, "store": store}
    if a.mode in ("head", "both"):
        p = need_file(a.insample_list, "the in-sample frame list")
        if mutation("skip_list_sha"):
            got = sha256_file(p)
        else:
            got = check_sha(p, a.insample_list_sha256, "the in-sample frame list", full=True)
        rows = VS.read_list(p)
        VS.verify_list(rows, store, excl, n_pc)
        out["insample"] = {"rows": rows, "keys": [(r[0], r[1]) for r in rows], "sha256": got}
        pr = need_file(a.repro_list, "the C0-ID frame list (stage0_repro256.tsv)")
        got_r = check_sha(pr, a.repro_list_sha256, "the C0-ID frame list", full=True)
        L = open(pr, encoding="utf-8").read().split("\n")
        if not L or L[0] != "log\ttoken\tsha256key":
            refuse("the C0-ID frame list header is not log<TAB>token<TAB>sha256key")
        rrows = [tuple(x.split("\t")) for x in L[1:] if x]
        if len({(r[0], r[1]) for r in rrows}) != len(rrows) or not rrows:
            refuse("the C0-ID frame list is empty or holds duplicates")
        for lg, tk, h in rrows:
            if h != VS.key_hash(lg, tk):
                refuse(f"the C0-ID list's stored hash of ({lg}, {tk}) is not sha256 of '<log>|<token>'")
            if lg not in set(sft):
                refuse(f"the C0-ID frame ({lg}, {tk}) is outside the 24 SFT held-out logs")
        out["repro"] = {"keys": [(r[0], r[1]) for r in rrows], "sha256": got_r}
    if a.mode in ("init", "both"):
        out["pod"] = {}
        for tag, p, sh in (("step0", a.pod_step0, a.pod_step0_sha256), ("h48x", a.pod_h48x, a.pod_h48x_sha256)):
            need_file(p, f"the pod's head_pred_{tag}.npz")
            got = check_sha(p, sh, f"the pod's head_pred_{tag}.npz")
            z = np.load(p, allow_pickle=False)
            if any(k not in z.files for k in ("log", "token", "step", "logit")):
                refuse(f"the pod's head_pred_{tag}.npz lacks log/token/step/logit")
            keys = list(zip([str(x) for x in z["log"]], [str(x) for x in z["token"]]))
            if len(set(keys)) != len(keys) or not keys:
                refuse(f"the pod's head_pred_{tag}.npz is empty or holds duplicate keys")
            out["pod"][tag] = {"keys": keys, "steps": [int(s) for s in z["step"]], "logit": np.asarray(z["logit"], np.float32), "sha256": got}
    if a.banked_pred and a.mode in ("head", "both"):
        need_file(a.banked_pred, "the banked head_pred_update10000.npz")
        out["banked_sha256"] = check_sha(a.banked_pred, a.banked_sha256, "the banked head_pred_update10000.npz")
        bz = np.load(a.banked_pred, allow_pickle=False)
        bk = list(zip([str(x) for x in bz["log"]], [str(x) for x in bz["token"]]))
        pos = {k: i for i, k in enumerate(bk)}
        miss = [k for k in out["repro"]["keys"] if k not in pos]
        if miss:
            refuse(f"{len(miss)} C0-ID frame(s) are not in the banked head_pred_update10000.npz, e.g. {miss[0]}")
        out["banked_logit"] = np.asarray(bz["logit"], np.float32)[[pos[k] for k in out["repro"]["keys"]]]
    elif a.mode in ("head", "both"):
        refuse("--banked-pred is required in head mode (the C0-ID reference: a tool that cannot reproduce S1a's rows has no authority over the in-sample number)")
    return out


def bank_key(r):
    return (r.get("log_name", ""), r.get("token", ""), int(r.get("step", 0)), int(r.get("rank", 0)))


def bank_indices(rows, in_bank, *, heldout, ho_logs=(), require_pdm=False, pdm_keys=()):
    """the trainer's tr_idx / ho_idx, line for line (scorer_finetune.py:1771-1776): train = key in the bank and log not held out (and PDM if required); held-out = key in the held-out bank (and PDM)."""
    out = []
    for i, r in enumerate(rows):
        k = bank_key(r)
        if k not in in_bank or (require_pdm and k not in pdm_keys):
            continue
        if not heldout and r.get("log_name") in ho_logs:
            continue
        out.append(i)
    return out


def locate(rows, idx, wanted, pick, what):
    """wanted [(log, token, step, rank|None)] -> row indices.  pick 'exact': rank must match (list frames: step 0 rank 0).  pick 'first': the first row in idx order for (log, token, step) - the trainer's own
    h48x de-duplication (scorer_finetune.py:1828-1835) - used for the pod's reference files, which carry no rank.  A frame with no row is a REFUSAL."""
    by = {}
    for i in idx:
        r = rows[i]
        by.setdefault((r.get("log_name", ""), r.get("token", ""), int(r.get("step", 0))), []).append((int(r.get("rank", 0)), i))
    out = []
    for lg, tk, st, rk in wanted:
        c = by.get((lg, tk, st))
        if not c:
            refuse(f"{what}: no bank row for ({lg}, {tk}, step {st})")
        if pick == "exact":
            m = [i for r_, i in c if r_ == rk]
            if len(m) != 1:
                refuse(f"{what}: ({lg}, {tk}, step {st}) has {len(m)} bank rows at rank {rk} (need exactly 1)")
            out.append(m[0])
        else:
            out.append(c[0][1])
    return out


# ======================================================================== the forward (the trainer's fwd(), head-relevant part)
@torch.no_grad()
def forward_pass(model, cap, heads, dataset, order, need, device, batch, workers, autocast_on):
    """order: row indices (ascending); need {row: set(head names)}; a head runs on a whole batch when any row of it needs it.  -> {(head, row): (logit [40,32], sdf [40,32])} float32 CPU.
    Mirrors scorer_finetune.fwd(): model(img, ego, goal, calib, score_extra=cxy) under bf16 autocast, sctx = the scoring decoder's own context tap, aux.predict(sctx)."""
    from torch.utils.data import DataLoader, Subset
    out = {}
    pos = 0
    dl = DataLoader(Subset(dataset, list(order)), batch_size=batch, shuffle=False, num_workers=workers)
    for b in dl:
        img, ego, goal, _tgt, cxy, _ctg, _cm, _ci, cal = b
        n = img.shape[0]
        rows_b = list(order[pos:pos + n])
        pos += n
        img, ego, goal, cxy = (x.to(device, non_blocking=True) for x in (img, ego, goal, cxy))
        names = sorted(set().union(*[need[i] for i in rows_b]))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_on):
            model(img, ego, goal, calib=cal if cal.numel() else None, score_extra=cxy)         # `cal` stays where the loader put it, as in the trainer
            sctx = cap["sctx"]
            res = {nm: heads[nm].predict(sctx) for nm in names}
        for nm in names:
            lg, sd = res[nm][0].detach().float().cpu().numpy(), res[nm][1].detach().float().cpu().numpy()
            for j, i in enumerate(rows_b):
                if nm in need[i]:
                    out[(nm, i)] = (lg[j], sd[j])
    if pos != len(order):
        raise RuntimeError(f"forward: {pos} outputs for {len(order)} rows")
    return out


def write_pred_npz(path, keys, logits, sdfs):
    if mutation("y_flip_output"):
        logits, sdfs = logits[:, :, ::-1], sdfs[:, :, ::-1]
    if mutation("misalign_keys"):
        keys = keys[1:] + keys[:1]
    tmp = path + ".tmp.npz"
    np.savez(tmp, log=np.asarray([k[0] for k in keys]), token=np.asarray([k[1] for k in keys]), step=np.zeros(len(keys), np.int64),
             logit=np.ascontiguousarray(logits, np.float32), sdf_m=np.ascontiguousarray(sdfs, np.float32))
    os.replace(tmp, path)
    VS.load_pred(path, keys, what=os.path.basename(path))                  # the consumer's own validator: never leave behind a file `vdac_stage0.py` would refuse


def real_banks(ra):
    """the two banks (scorer_finetune.py:1771-1776, 1788-1795): CPU only, no model.  NOT exercised by the CPU tests (it needs the pod's data): `bank-keys` and `--dry-run --with-banks` are its checks."""
    import copy
    import train as T
    cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(ra.backbone), ra.ckpt)             # reads only the checkpoint's meta (memory-mapped)
    op = T.OnPolicyBank(ra.onpolicy, cfg.n_proposals, cfg.horizon_steps, lane_labels=ra.lane_labels_train or None, pdm_labels=ra.pdm_labels_train or None, teacher_lane_labels=ra.teacher_lane_train or None)
    ho = T.OnPolicyBank(ra.heldout, cfg.n_proposals, cfg.horizon_steps, lane_labels=ra.lane_labels_heldout or None, pdm_labels=ra.pdm_labels_heldout or None, teacher_lane_labels=ra.teacher_lane_heldout or None)
    ds = T.TargetBank(ra.targets, ra.images, cfg, calib=ra.calib, onpolicy=op)
    ds_ho = copy.copy(ds)
    ds_ho.onpolicy = ho
    ho_logs = set(open(ra.heldout_logs, encoding="utf-8").read().split())
    tr_idx = bank_indices(ds.rows, op.by, heldout=False, ho_logs=ho_logs, require_pdm=ra.require_pdm, pdm_keys=op.pdm_keys)
    ho_idx = bank_indices(ds.rows, ho.by, heldout=True, require_pdm=ra.require_pdm, pdm_keys=ho.pdm_keys)
    if not tr_idx or not ho_idx:
        refuse(f"{len(tr_idx)} training and {len(ho_idx)} held-out rows survive the trainer's filters")
    return {"train": (ds, tr_idx), "heldout": (ds_ho, ho_idx)}


def real_context(ra, device):
    """the model (P6, built and loaded exactly as the trainer: scorer_finetune.py:1736-1760) and the two banks."""
    cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(ra.backbone), ra.ckpt)
    model = REFe(cfg)
    meta: dict = {}
    ckpt_io.load_for_inference(model, ra.ckpt, map_location="cpu", backbone=ra.backbone, meta_out=meta)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if int(model.score_head.out_features) == 7:
        refuse("the checkpoint's score head has 7 outputs (lane output): not P6")
    cap: dict = {}
    model.score_dec[0].register_forward_pre_hook(lambda mod, args: cap.__setitem__("sctx", args[1]))
    return {"model": model, "cap": cap, **real_banks(ra), "p6_meta": meta}


def main_bank_keys(argv, banks_factory=None):
    """`bank-keys`: the P6 TRAIN bank's rows (log, token, step, rank) as the TSV `vdac_stage0.py make-list --bank-keys` takes (keys only; no model, no GPU).  The rows are the trainer's own tr_idx; with --run-dir the
    count must equal the run's data event `train_sets` (a different bank than HA-s0 trained on is refused)."""
    ap = argparse.ArgumentParser(prog="vdac_stage0_forward.py bank-keys")
    ap.add_argument("--argv-file", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ra = parse_run_argv(open(need_file(a.argv_file, "the argv file"), encoding="utf-8").read())
    ev_name, events = read_events(a.run_dir)
    data = [e for e in events if e.get("event") == "data"]
    if len(data) != 1 or "train_sets" not in data[0]:
        refuse(f"{ev_name} holds {len(data)} data events (expected exactly 1, with train_sets)")
    ds, idx = (banks_factory or real_banks)(ra)["train"]
    if len(idx) != int(data[0]["train_sets"]):
        refuse(f"the rebuilt train index holds {len(idx)} sets, the run's data event says train_sets = {data[0]['train_sets']}: not the bank HA-s0 trained on")
    txt = "log" + chr(9) + "token" + chr(9) + "step" + chr(9) + "rank" + chr(10) + "".join(
        chr(9).join([r.get("log_name", ""), r.get("token", ""), str(int(r.get("step", 0))), str(int(r.get("rank", 0)))]) + chr(10) for r in (ds.rows[i] for i in idx))
    sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    print("PLAN", json.dumps({"bank_rows": len(idx), "matches_run_train_sets": int(data[0]["train_sets"]), "bank_keys_sha256": sha}))
    if a.dry_run:
        print("DRY-RUN: nothing written")
        return 0
    p = os.path.join(a.out_dir, "bank_keys.tsv")
    if os.path.exists(p):
        refuse(f"{p} exists: nothing is overwritten")
    os.makedirs(a.out_dir, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline=chr(10)) as f:
        f.write(txt)
    print("WROTE", p, "sha256", sha)
    return 0


# ======================================================================== plan + run
def build_cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=("head", "init", "both"))
    ap.add_argument("--run-dir", required=True, help="the HA-s0 run directory (aux_drv.pt, model_sft_A.pt, sft.jsonl)")
    ap.add_argument("--argv-file", required=True, help="the run's argv, one line: `ARM=HA-s0 MODE=argv bash vdac_pod.sh` output")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--aux-sha256", required=True, help="sha256 (>= 12 hex) of aux_drv.pt (HA-s0: LFS 281320071fea...)")
    ap.add_argument("--p6-sha256", required=True, help="sha256 (>= 12 hex) of --ckpt (P6 model_final.pt: 968a469d...)")
    ap.add_argument("--store", required=True, help="the aux label store dir (verifies the frame list exactly as `vdac_stage0.py read` does)")
    ap.add_argument("--h48-logs", required=True)
    ap.add_argument("--sft-heldout", required=True)
    ap.add_argument("--store-sha256-prefix", default=VS.R.STORE_SHA_PREFIX)
    ap.add_argument("--insample-list", default="")
    ap.add_argument("--insample-list-sha256", default="")
    ap.add_argument("--repro-list", default="")
    ap.add_argument("--repro-list-sha256", default="")
    ap.add_argument("--banked-pred", default="", help="HA-s0's head_pred_update10000.npz (C0-ID reference)")
    ap.add_argument("--banked-sha256", default="")
    ap.add_argument("--pod-step0", default="", help="the pod's head_pred_step0.npz (1,705 frames, HELD-OUT bank)")
    ap.add_argument("--pod-step0-sha256", default="")
    ap.add_argument("--pod-h48x", default="", help="the pod's head_pred_h48x.npz (1,502 frames, TRAIN bank)")
    ap.add_argument("--pod-h48x-sha256", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-banks", action="store_true", help="with --dry-run: also build the model (CPU) and the banks and resolve every frame")
    ap.add_argument("--allow-shared-gpu", action="store_true", help="DEFAULT IS TO REFUSE when any process holds the GPU")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--autocast", default="bf16", choices=("bf16", "off"), help="bf16 = the trainer's fwd(); off only for CPU tests")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--per-city-for-tests", type=int, default=0, help=argparse.SUPPRESS)
    return ap


def validate_cli(a):
    need = {"head": ("insample_list", "insample_list_sha256", "repro_list", "repro_list_sha256", "banked_pred", "banked_sha256"),
            "init": ("pod_step0", "pod_step0_sha256", "pod_h48x", "pod_h48x_sha256")}
    for m, names in need.items():
        if a.mode in (m, "both"):
            miss = [n for n in names if not getattr(a, n)]
            if miss:
                refuse(f"--mode {a.mode} needs " + ", ".join("--" + n.replace("_", "-") for n in miss))
    for nm in ("run_dir", "store"):
        if not os.path.isdir(getattr(a, nm)):
            refuse(f"--{nm.replace('_', '-')} {getattr(a, nm)!r} is not a directory")


def out_names(mode):
    n = []
    if mode in ("head", "both"):
        n += ["preds_insample.npz", "preds_repro256.npz"]
    if mode in ("init", "both"):
        n += ["head_init0_rebuilt_a.npz", "head_init0_b.npz"]
    return n


def reproduction(name, ref, mine, bar):
    ag = VS.sign_agreement(ref, mine)
    return {"check": name, "frames": int(len(ref)), "sign_agreement": ag, "bar": bar, "max_abs_logit_diff": float(np.abs(np.asarray(ref, np.float64) - np.asarray(mine, np.float64)).max()), "ok": bool(ag >= bar)}


def main(argv=None, factory=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "bank-keys":
        return main_bank_keys(argv[1:], factory)
    a = build_cli().parse_args(argv)
    validate_cli(a)
    t0 = time.time()
    n_pc = a.per_city_for_tests or VS.PER_CITY
    # ---- 1. the run's configuration, reproduced from two independent records (argv, model_sft_A.pt meta) and checked against aux_drv.pt and the data event
    ra = parse_run_argv(open(need_file(a.argv_file, "the argv file"), encoding="utf-8").read())
    if os.path.normpath(ra.out) != os.path.normpath(a.run_dir):
        refuse(f"the argv's --out {ra.out} is not --run-dir {a.run_dir}: the argv file is not this run's")
    diffs = diff_args(ra, recorded_args(os.path.join(a.run_dir, "model_sft_A.pt")))
    if diffs:
        refuse(f"the re-parsed argv differs from the namespace recorded in model_sft_A.pt in {len(diffs)} key(s): " + " | ".join(diffs[:8]))
    aux_sha = check_sha(os.path.join(a.run_dir, "aux_drv.pt"), a.aux_sha256, "aux_drv.pt")
    p6_sha = check_sha(need_file(ra.ckpt, "the P6 checkpoint named by --ckpt"), a.p6_sha256, "the P6 checkpoint")
    ev_name, events = read_events(a.run_dir)
    trained, aux_file, head_info = (None, None, None)
    if a.mode in ("head", "both"):
        trained, aux_file, head_info = load_trained_aux(os.path.join(a.run_dir, "aux_drv.pt"), ra, a.run_dir, ev_name, events)
    init = build_init_aux(ra) if a.mode in ("init", "both") else None
    # ---- 2. the frames
    fr = read_lists_and_verify(a, n_pc)
    names = out_names(a.mode)
    clash = [n for n in names + ["stage0_forward.json"] if os.path.exists(os.path.join(a.out_dir, n))]
    if clash:
        refuse(f"--out-dir already holds {clash}: nothing is overwritten")
    if a.dry_run:
        apps, why = GPU_PROBE() if a.device == "cuda" else ([], "")
        print("GPU_STATE", json.dumps({"device": a.device, "compute_processes": apps, "unverifiable": why, "a_real_run_would": "refuse" if (apps or apps is None) and not a.allow_shared_gpu else "proceed"}))
    else:
        require_gpu_free(a.device, a.allow_shared_gpu, "start")
    plan = {"tool": TOOL, "mode": a.mode, "run_dir": a.run_dir, "aux_sha256": aux_sha, "p6_sha256": p6_sha, "seed": int(ra.seed), "outputs": names,
            "frames": {k: len(fr[k]["keys"]) for k in ("insample", "repro") if k in fr} | {f"pod_{k}": len(v["keys"]) for k, v in fr.get("pod", {}).items()},
            "head_load": None if head_info is None else {k: head_info[k] for k in ("tensors_identical_to_init", "tensors")}, "device": a.device, "autocast": a.autocast,
            "argv_vs_recorded_namespace": "identical (all keys except out, dump_scores)", "event_file": ev_name}
    print("PLAN", json.dumps(plan))
    if a.dry_run and not a.with_banks:
        print("DRY-RUN: every input is present, hashed and consistent; no bank, no model, no CUDA, nothing written")
        return 0
    # ---- 3. model + banks, resolved to rows BEFORE any GPU use
    factory = factory or (lambda ra_, dev_: real_context(ra_, dev_))
    ctx = factory(ra, a.device)
    model, cap = ctx["model"], ctx["cap"]
    banks = {"train": ctx["train"], "heldout": ctx["heldout"]}
    jobs = []        # (out file, bank, head, keys, wanted rows spec)
    if a.mode in ("head", "both"):
        jobs.append(("preds_insample.npz", "train", "trained", fr["insample"]["keys"], [(k[0], k[1], 0, 0) for k in fr["insample"]["keys"]], "exact"))
        jobs.append(("preds_repro256.npz", "heldout", "trained", fr["repro"]["keys"], [(k[0], k[1], 0, 0) for k in fr["repro"]["keys"]], "exact"))
    if a.mode in ("init", "both"):
        p0, ph = fr["pod"]["step0"], fr["pod"]["h48x"]
        jobs.append(("head_init0_rebuilt_a.npz", "heldout", "init", p0["keys"], [(k[0], k[1], s, None) for k, s in zip(p0["keys"], p0["steps"])], "first"))
        jobs.append(("head_init0_b.npz", "train", "init", ph["keys"], [(k[0], k[1], s, None) for k, s in zip(ph["keys"], ph["steps"])], "first"))
    resolved = []
    for fname, bank, head, keys, wanted, pick in jobs:
        ds, idx = banks[bank]
        rows_i = locate(ds.rows, idx, wanted, pick, f"{fname} ({bank} bank)")
        if any(int(ds.rows[i].get("step", 0)) != 0 for i in rows_i):
            refuse(f"{fname}: a resolved bank row has step != 0 (the reader's schema is step 0)")
        resolved.append((fname, bank, head, keys, rows_i))
    print("RESOLVED", json.dumps({f: len(r) for f, _, _, _, r in resolved}))
    if a.dry_run:
        print("DRY-RUN: model and banks built on CPU, every frame resolved to exactly one bank row; nothing written")
        return 0
    # ---- 4. the GPU is checked AGAIN right before it is used (loading the banks took minutes; P7 may have come back), then everything is moved and run
    gpu_seen = require_gpu_free(a.device, a.allow_shared_gpu, "before the forward")
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        refuse("--device cuda but torch.cuda.is_available() is False (no silent CPU fallback)")
    model.to(device)
    heads = {}
    if trained is not None:
        heads["trained"] = trained.to(device).eval()
    if init is not None:
        heads["init"] = init.to(device).eval()
    for h in heads.values():
        if any(isinstance(m, torch.nn.Dropout) for m in h.modules()):
            refuse("the head holds a Dropout: eval mode would differ from the trainer's train-mode forward")
    os.makedirs(a.out_dir, exist_ok=True)
    results = {}
    t_fwd = time.time()
    for bank in ("train", "heldout"):
        ds, _idx = banks[bank]
        need = {}
        for fname, b, head, keys, rows_i in resolved:
            if b == bank:
                for i in rows_i:
                    need.setdefault(i, set()).add(head)
        if not need:
            continue
        order = sorted(need)         # natural row order, batches of 16: the SAME composition the trainer's own dumps used (evaluate(), the h48x forward: batch_size=16, in index order) wherever the frame set is the same
        results[bank] = forward_pass(model, cap, heads, ds, order, need, device, a.batch, a.workers, a.autocast == "bf16")
        print("FORWARD", json.dumps({"bank": bank, "distinct_frames": len(order), "elapsed_s": round(time.time() - t_fwd, 1)}), flush=True)
    wall_fwd = time.time() - t_fwd
    written, loaded = {}, {}
    for fname, bank, head, keys, rows_i in resolved:
        lg = np.stack([results[bank][(head, i)][0] for i in rows_i])
        sd = np.stack([results[bank][(head, i)][1] for i in rows_i])
        write_pred_npz(os.path.join(a.out_dir, fname), keys, lg, sd)
        written[fname] = {"sha256": sha256_file(os.path.join(a.out_dir, fname)), "frames": len(keys)}
        loaded[fname] = (keys, lg)
    # ---- 5. reproduction reports (the consumer's own bars; reported here so a pause is not spent finding out later)
    repro = []
    if "preds_repro256.npz" in loaded:
        repro.append(reproduction("C0-ID (256 frames vs the banked head_pred_update10000.npz)", fr["banked_logit"], loaded["preds_repro256.npz"][1], ID_SIGN_AGREE))
    if "head_init0_rebuilt_a.npz" in loaded:
        p0 = fr["pod"]["step0"]
        pos = {k: i for i, k in enumerate(loaded["head_init0_rebuilt_a.npz"][0])}
        repro.append(reproduction("S1-F init (rebuilt seed-0 head vs the pod's head_pred_step0.npz)", p0["logit"], loaded["head_init0_rebuilt_a.npz"][1][[pos[k] for k in p0["keys"]]], INIT_SIGN_AGREE))
    meta = {"tool": TOOL, "utc": utc(), "status": "OK" if all(r["ok"] for r in repro) else "REPRODUCTION_FAILED", "tool_sha256": sha256_file(os.path.abspath(__file__)),
            "consumer_sha256": sha256_file(VS.__file__), "plan": plan, "outputs": written, "reproduction": repro, "head_load": None if head_info is None else head_info,
            "inputs": {"argv_file_sha256": sha256_file(a.argv_file), "insample_list_sha256": fr.get("insample", {}).get("sha256"), "repro_list_sha256": fr.get("repro", {}).get("sha256"),
                       "pod_step0_sha256": fr.get("pod", {}).get("step0", {}).get("sha256"), "pod_h48x_sha256": fr.get("pod", {}).get("h48x", {}).get("sha256"), "banked_sha256": fr.get("banked_sha256")},
            "env": {"torch": torch.__version__, "device": a.device, "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "autocast": a.autocast, "batch": a.batch},
            "gpu_apps_seen_before_forward": gpu_seen, "wall_s": {"forward": round(wall_fwd, 1), "total": round(time.time() - t0, 1)}}
    with open(os.path.join(a.out_dir, "stage0_forward.json"), "w", encoding="utf-8", newline="\n") as f_meta:
        json.dump(meta, f_meta, indent=1, default=float)
    for r in repro:
        print("REPRO", json.dumps(r))
    print("STAGE0_FORWARD", meta["status"], json.dumps({k: v["frames"] for k, v in written.items()}))
    return 0 if meta["status"] == "OK" else 3


if __name__ == "__main__":
    sys.exit(main())
