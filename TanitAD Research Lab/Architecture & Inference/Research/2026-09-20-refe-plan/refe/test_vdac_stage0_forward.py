#!/usr/bin/env python3
"""Tests of vdac_stage0_forward.py on SYNTHETIC fixtures only: a TINY REFe (built with the real class), the REAL head class (2,440,450 params), a synthetic label store and banks.  No real data, no pod, no GPU.
    CUDA_VISIBLE_DEVICES=-1  python test_vdac_stage0_forward.py
Layout needed (the repo layout): this file and vdac_stage0_forward.py in refe/ beside model.py, train.py, scorer_finetune.py, aux_drivable_head.py, ckpt_io.py, vdac_stage0.py, test_vdac_stage0.py (its synthetic
store `Fx` is reused); ../eval/s1_read_v2.py, the frozen s1_read.py and ../splits/h48_logs.txt as for test_vdac_stage0.py.  REFE_DIR overrides where the refe/ package is read from."""
import dataclasses
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_vdac_stage0 as TS  # noqa: E402  (the consumer's synthetic store)
import vdac_stage0 as VS  # noqa: E402
import vdac_stage0_forward as F  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402
import ckpt_io  # noqa: E402
from model import REFe, REFeConfig  # noqa: E402

PER_CITY, REPRO = 8, 6
# the REAL HA-s0 argv: `ARM=HA-s0 MODE=argv VDAC_WS=/workspace bash vdac_pod.sh` at git tip 2207ef03 (MEASURED 2026-11-12)
REAL_ARGV = ("--ckpt /workspace/data/refe_runs/p6_paper_faithful/model_final.pt --targets /workspace/data/refe_navtrain/train_grow_cmd_3c_p7 --images /workspace/data/navtrain_pixels "
             "--calib /workspace/data/refe_navtrain/train_grow/calib_table.json --onpolicy /workspace/data/p6base_labels/train_sets --heldout /workspace/data/p6base_labels/heldout_sets "
             "--heldout-logs /workspace/data/p6base_labels/logs/heldout_logs.txt --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --pdm-labels-heldout /workspace/data/p6base_labels/heldout_pdm "
             "--pdm-labels-train /workspace/data/p6base_labels/train_pdm --require-pdm --navtest-logs /workspace/data/refe_scorer1p6/navtest_ids/navtest_logs.json "
             "--navtest-tokens /workspace/data/refe_scorer1p6/navtest_ids/navtest_tokens.txt.gz --max-updates 10000 --batch 8 --accum 1 --epochs 2 --eval-every 5000 --seed 0 --aux-drv --w-aux-drv 0.1 "
             "--aux-drv-lr 0.0002 --aux-drv-tap visual --aux-drv-depth 2 --aux-drv-labels /workspace/data/aux_drivable/label_store --aux-drv-exclude-logs /workspace/data/refe_vdac/h48_logs.txt "
             "--aux-drv-exclude-sha256 2ae13935a3ead7f30b015f33b0bebc7a35e70eb2a4194161f8cbb66ac89b8da8 --dump-scores /workspace/data/refe_vdac/ha-s0/scores --out /workspace/data/refe_vdac/ha-s0")
TINY = dict(backbone="vits16", width=32, depth=1, heads=2, img_h=32, img_w=32, n_cameras=1, cameras=("CAM_F0",), dec_depth=1, n_proposals=4, horizon_steps=4, reg_heads=2, n_registers=2, lora_rank=2)


def fs(p):
    """forward slashes: the argv is parsed with shlex (POSIX), as on the pod; a Windows sandbox path must not carry backslashes into it"""
    return p.replace("\\", "/")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class SynthDS(torch.utils.data.Dataset):
    """a bank-shaped dataset: .rows (log_name, token, step, rank) and the trainer's 9-tuple per item.  The image / ego / goal are a function of (log, token) only (a decoy row of another step / rank carries the same
    frame); the proposals cxy differ per row."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        g = torch.Generator().manual_seed(int(hashlib.sha256(f"{r['log_name']}|{r['token']}".encode()).hexdigest()[:8], 16))
        img, ego, goal = torch.rand(3, 32, 32, generator=g), torch.rand(7, generator=g), torch.rand(4, generator=g)
        g2 = torch.Generator().manual_seed(i)
        cxy = torch.rand(5, 4, 3, generator=g2)
        return img, ego, goal, torch.zeros(4, 3), cxy, torch.zeros(5, 6), torch.ones(5, dtype=torch.bool), 0, torch.zeros(0)


def make_model(seed=7):
    torch.manual_seed(seed)
    model = REFe(REFeConfig(**TINY)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cap = {}
    model.score_dec[0].register_forward_pre_hook(lambda mod, args: cap.__setitem__("sctx", args[1]))
    return model, cap


class Env:
    """one synthetic run: store, banks, a trained head saved in the trainer's aux_drv/1 format, the recorded namespace, the data event, the frame lists and the pod's reference files."""

    def __init__(self, td):
        self.td = td
        self.fx = fx = TS.Fx(td)
        self.run = os.path.join(td, "run", "ha-s0")
        os.makedirs(os.path.join(self.run, "scores"))
        # rows of ONE target dir: every labelled frame at (step 0, rank 0) plus decoys at step 1 and rank 3 (as the consumer's fixture); SFT logs are the held-out bank
        self.rows = []
        for (lg, t) in sorted(fx.labels):
            for st, rk in ((0, 0), (1, 0), (0, 3)):
                self.rows.append({"log_name": lg, "token": t, "step": st, "rank": rk})
        self.ds = SynthDS(self.rows)
        self.ho_logs = set(fx.sft)
        allkeys = {F.bank_key(r) for r in self.rows}
        self.tr_idx = F.bank_indices(self.rows, allkeys, heldout=False, ho_logs=self.ho_logs)
        self.ho_idx = F.bank_indices(self.rows, {F.bank_key(r) for r in self.rows if r["log_name"] in self.ho_logs}, heldout=True)
        self.model, self.cap = make_model()
        self.p6 = os.path.join(td, "p6_model_final.pt")
        open(self.p6, "wb").write(b"synthetic P6 checkpoint bytes")
        # argv = the REAL one with the paths re-pointed at this sandbox
        self.argv_text = (REAL_ARGV.replace("/workspace/data/refe_runs/p6_paper_faithful/model_final.pt", fs(self.p6)).replace("/workspace/data/aux_drivable/label_store", fs(fx.root))
                          .replace("--out /workspace/data/refe_vdac/ha-s0", "--out " + fs(self.run)).replace("--max-updates 10000", "--max-updates 10"))
        self.argv_file = os.path.join(td, "argv.txt")
        open(self.argv_file, "w").write(self.argv_text + "\n")
        self.ra = F.parse_run_argv(self.argv_text)
        self.cfg = F.expected_aux_cfg(self.ra)
        # the trained head: the seed-0 head, every tensor moved (as 10,000 Adam updates would)
        self.init_aux = AUX.AuxDrivable(self.cfg)
        self.trained = AUX.AuxDrivable(self.cfg)
        g = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for p in self.trained.head.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
        self.store_sha = json.load(open(os.path.join(fx.root, "manifest.json")))["store_sha256"]
        self.write_aux()
        torch.save({"model": {}, "meta": {"scorer_finetune": {"arm": "A", "base_ckpt": self.p6, "args": dict(vars(self.ra))}}}, os.path.join(self.run, "model_sft_A.pt"))
        self.write_events()
        # frame lists, through the consumer's own make-list
        self.list_dir = os.path.join(td, "lists")
        bank_pred0 = os.path.join(td, "banked_for_list.npz")
        sft_keys = sorted(k for k in fx.labels if k[0] in self.ho_logs)
        TS.write_pred(bank_pred0, sft_keys, np.zeros((len(sft_keys), 40, 32), np.float32))
        rc = VS.main(["make-list", "--out-dir", self.list_dir, "--bank-keys", fx.bank_path, "--store", fx.root, "--h48-logs", fx.h48_path, "--sft-heldout", fx.sft_path, "--banked-pred", bank_pred0,
                      "--store-sha256-prefix", fx.prefix, "--per-city-for-tests", str(PER_CITY), "--repro-n-for-tests", str(REPRO)])
        assert rc == 0
        self.list_p = os.path.join(self.list_dir, "stage0_frames.tsv")
        self.repro_p = os.path.join(self.list_dir, "stage0_repro256.tsv")
        meta = json.load(open(os.path.join(self.list_dir, "stage0_frames.json")))
        self.list_sha, self.repro_sha = meta["frame_list_sha256"], meta["repro256_sha256"]
        self.list_keys = [(r[0], r[1]) for r in VS.read_list(self.list_p)]
        self.repro_keys = [tuple(x.split("\t")[:2]) for x in open(self.repro_p).read().split("\n")[1:-1]]
        # the pod's reference files, made by an INDEPENDENT forward (a plain loop, fp32, batch 1, the reference head objects)
        self.step0_keys = sft_keys                                                    # the held-out bank's frames (SFT logs)
        self.h48x_keys = sorted(k for k in fx.labels if k[0] in set(fx.h48))[:12]
        banked_keys = self.repro_keys + [k for k in self.step0_keys if k not in set(self.repro_keys)][:4]
        self.banked = os.path.join(self.run, "scores", "head_pred_update10000.npz")
        TS.write_pred(self.banked, banked_keys, self.ref(self.trained, banked_keys)[0])
        self.pod_step0 = os.path.join(self.run, "scores", "head_pred_step0.npz")
        TS.write_pred(self.pod_step0, self.step0_keys, self.ref(self.init_aux, self.step0_keys)[0])
        self.pod_h48x = os.path.join(self.run, "head_pred_h48x.npz")
        TS.write_pred(self.pod_h48x, self.h48x_keys, self.ref(self.init_aux, self.h48x_keys)[0])

    def write_aux(self, mutate=None):
        z = {"format": "aux_drv/1", "head": self.trained.head.state_dict(), "feed": None, "opt": {}, "update": 10,
             "meta": {"config": dataclasses.asdict(self.cfg), "store_sha256": self.store_sha, "frames": 1, "shuffle_seed": None, "head_file": "aux_drv.pt"}}
        if mutate:
            mutate(z)
        ckpt_io.atomic_save(z, os.path.join(self.run, "aux_drv.pt"))
        self.aux_sha = sha(os.path.join(self.run, "aux_drv.pt"))

    def write_events(self, mutate=None):
        ev = {"event": "data", "train_sets": len(self.tr_idx), "aux_drv": {**self.init_aux.built_facts(), "tap": "visual", "feed": "off", "lr": 0.0002, "store_sha256": self.store_sha, "frames": 1, "shuffle_seed": None}}
        if mutate:
            mutate(ev)
        with open(os.path.join(self.run, "sft.jsonl"), "w") as f:
            f.write(json.dumps({"event": "start"}) + "\n" + json.dumps(ev) + "\n")

    def ref(self, aux, keys):
        """independent reference: one frame at a time, fp32, the model + the given head object."""
        row = {(r["log_name"], r["token"]): i for i, r in enumerate(self.rows) if r["step"] == 0 and r["rank"] == 0}
        L, S = [], []
        with torch.no_grad():
            for k in keys:
                img, ego, goal, _t, cxy, _c, _m, _i, cal = self.ds[row[k]]
                self.model(img[None], ego[None], goal[None], calib=None, score_extra=cxy[None])
                lg, sd = aux.predict(self.cap["sctx"])
                L.append(lg[0].numpy())
                S.append(sd[0].numpy())
        return np.stack(L).astype(np.float32), np.stack(S).astype(np.float32)

    def factory(self, ra, dev):
        return {"model": self.model, "cap": self.cap, "train": (self.ds, self.tr_idx), "heldout": (self.ds, self.ho_idx)}

    def argv(self, out, mode="both", extra=()):
        a = ["--mode", mode, "--run-dir", self.run, "--argv-file", self.argv_file, "--out-dir", out, "--aux-sha256", self.aux_sha[:16], "--p6-sha256", sha(self.p6)[:16],
             "--store", self.fx.root, "--h48-logs", self.fx.h48_path, "--sft-heldout", self.fx.sft_path, "--store-sha256-prefix", self.fx.prefix, "--per-city-for-tests", str(PER_CITY),
             "--device", "cpu", "--autocast", "off", "--batch", "5", "--workers", "0"]
        if mode in ("head", "both"):
            a += ["--insample-list", self.list_p, "--insample-list-sha256", self.list_sha, "--repro-list", self.repro_p, "--repro-list-sha256", self.repro_sha,
                  "--banked-pred", self.banked, "--banked-sha256", sha(self.banked)[:16]]
        if mode in ("init", "both"):
            a += ["--pod-step0", self.pod_step0, "--pod-step0-sha256", sha(self.pod_step0)[:16], "--pod-h48x", self.pod_h48x, "--pod-h48x-sha256", sha(self.pod_h48x)[:16]]
        return a + list(extra)

    def run_main(self, out, mode="both", extra=(), env=None):
        old = os.environ.get("VDAC_FWD_MUTATE")
        if env is not None:
            os.environ["VDAC_FWD_MUTATE"] = env
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                rc = F.main(self.argv(out, mode, extra), factory=self.factory)
        finally:
            if env is not None:
                if old is None:
                    os.environ.pop("VDAC_FWD_MUTATE", None)
                else:
                    os.environ["VDAC_FWD_MUTATE"] = old
        return rc, buf.getvalue()


def refused(testcase, fn, *frag):
    with testcase.assertRaises(SystemExit) as cm:
        fn()
    msg = str(cm.exception.code)
    testcase.assertTrue(msg.startswith("REFUSED"), msg)
    for f in frag:
        testcase.assertIn(f, msg)
    return msg


_COUNTER = 0


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.E = Env(cls.td.name)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def out(self):
        global _COUNTER
        _COUNTER += 1
        return os.path.join(self.td.name, f"out_{_COUNTER}")


class TLoader(Base):
    def test_loader_round_trips_a_saved_head_bit_for_bit(self):
        E = self.E
        aux, z, info = F.load_trained_aux(os.path.join(E.run, "aux_drv.pt"), E.ra, E.run, *F.read_events(E.run))
        sd, want = aux.head.state_dict(), E.trained.head.state_dict()
        self.assertEqual(list(sd), list(want))
        for k in want:
            self.assertTrue(torch.equal(sd[k], want[k]), k)                 # bit for bit, not allclose
        self.assertEqual(info["tensors_identical_to_init"], 0)
        self.assertEqual(info["facts"]["n_params"], 2440450)

    def test_loader_refusals(self):
        E = self.E
        path = os.path.join(E.run, "aux_drv.pt")
        ev = F.read_events(E.run)

        def fail(mut, frag, ev_mut=None):
            E.write_aux(mut)
            events = ev
            if ev_mut:
                E.write_events(ev_mut)
                events = F.read_events(E.run)
            try:
                refused(self, lambda: F.load_trained_aux(path, E.ra, E.run, *events), frag)
            finally:
                E.write_aux()
                E.write_events()

        fail(lambda z: z.__setitem__("format", "aux_drv/2"), "aux_drv/2")
        fail(lambda z: z.__setitem__("feed", {"proj.weight": torch.zeros(1)}), "feed")
        fail(lambda z: z["meta"]["config"].__setitem__("depth", 1), "head config")
        fail(lambda z: z.__setitem__("update", 5), "final head")
        fail(lambda z: z["meta"].__setitem__("store_sha256", "0" * 64), "store_sha256")
        fail(lambda z: z["meta"].__setitem__("shuffle_seed", 0), "label-shuffled")
        fail(lambda z: z["head"].pop(next(iter(z["head"]))), "strictly")
        fail(lambda z: None, "n_params", ev_mut=lambda e: e["aux_drv"].__setitem__("n_params", 1))
        fail(lambda z: None, "aux_drv.tap", ev_mut=lambda e: e["aux_drv"].__setitem__("tap", "scene"))
        # a head that is still the initial one (never loaded / never trained) is refused
        fail(lambda z: z.__setitem__("head", E.init_aux.head.state_dict()), "bit-identical")
        with open(path, "ab") as f:
            f.write(b"x")
        refused(self, lambda: F.check_sha(path, E.aux_sha[:16], "aux_drv.pt"), "does not start with")
        E.write_aux()
        refused(self, lambda: F.load_trained_aux(os.path.join(E.run, "absent.pt"), E.ra, E.run, *ev), "not a file")

    def test_init_head_is_seed0_deterministic_and_differs_from_seed1(self):
        E = self.E
        a = F.build_init_aux(E.ra)
        torch.manual_seed(12345)
        torch.rand(100)                                                      # the global stream moves between the builds: build_head forks its own
        b = F.build_init_aux(E.ra)
        for (k, p), (_, q) in zip(a.head.state_dict().items(), b.head.state_dict().items()):
            self.assertTrue(torch.equal(p, q), k)
        c = AUX.AuxDrivable(dataclasses.replace(E.cfg, seed=1))
        self.assertFalse(all(torch.equal(p, q) for p, q in zip(a.head.state_dict().values(), c.head.state_dict().values())))
        ref = AUX.build_head(dataclasses.replace(E.cfg, seed=0))
        for (k, p), q in zip(a.head.state_dict().items(), ref.state_dict().values()):
            self.assertTrue(torch.equal(p, q), k)
        ra1 = F.parse_run_argv(E.argv_text.replace("--seed 0", "--seed 1"))
        refused(self, lambda: F.build_init_aux(ra1), "seed-0")


class TConfig(Base):
    def test_the_real_ha_s0_argv_parses_with_the_trainers_own_parser(self):
        ra = F.parse_run_argv(REAL_ARGV)
        self.assertEqual((ra.seed, ra.max_updates, ra.aux_drv_depth, ra.aux_drv_tap, ra.aux_drv_feed, ra.w_aux_drv, ra.backbone, ra.require_pdm, ra.split_half),
                         (0, 10000, 2, "visual", "off", 0.1, "vitl16", True, -1))
        cfg = F.expected_aux_cfg(ra)
        self.assertEqual((cfg.tap, cfg.depth, cfg.feed, cfg.seed), ("visual", 2, "off", 0))

    def test_argv_refusals(self):
        refused(self, lambda: F.parse_run_argv(""), "empty")
        refused(self, lambda: F.parse_run_argv("--not-a-flag 1"), "does not parse")
        refused(self, lambda: F.parse_run_argv(REAL_ARGV.replace("--aux-drv ", "")), "aux")
        refused(self, lambda: F.parse_run_argv(REAL_ARGV + " --aux-drv-shuffle-seed 0"), "label-shuffled")
        refused(self, lambda: F.parse_run_argv(REAL_ARGV + " --split-half 0"), "split-half")

    def test_recorded_namespace_must_equal_the_argv_and_run_dir(self):
        E = self.E
        rec = F.recorded_args(os.path.join(E.run, "model_sft_A.pt"))
        self.assertEqual(F.diff_args(E.ra, rec), [])
        rec2 = dict(rec, lr=3e-4)
        self.assertTrue(any(d.startswith("lr:") for d in F.diff_args(E.ra, rec2)))
        ra2 = F.parse_run_argv(E.argv_text.replace("--batch 8", "--batch 4"))
        self.assertTrue(any(d.startswith("batch:") for d in F.diff_args(ra2, rec)))
        # through main(): a different argv than the recorded one is refused before any model is built
        other = os.path.join(self.td.name, "argv_other.txt")
        open(other, "w").write(E.argv_text.replace("--batch 8", "--batch 4"))
        a = E.argv(self.out(), "head")
        a[a.index("--argv-file") + 1] = other
        refused(self, lambda: F.main(a, factory=E.factory), "differs from the namespace recorded")
        wrong = os.path.join(self.td.name, "argv_wrong_out.txt")
        open(wrong, "w").write(E.argv_text.replace("--out " + fs(E.run), "--out /somewhere/else"))
        a = E.argv(self.out(), "head")
        a[a.index("--argv-file") + 1] = wrong
        refused(self, lambda: F.main(a, factory=E.factory), "not this run's")
        os.remove(other)
        os.remove(wrong)


class TFrames(Base):
    def test_bank_indices_and_locate(self):
        rows = [{"log_name": "A", "token": "t1", "step": 0, "rank": 0}, {"log_name": "A", "token": "t1", "step": 0, "rank": 3}, {"log_name": "A", "token": "t1", "step": 1, "rank": 0},
                {"log_name": "H", "token": "t2", "step": 0, "rank": 0}, {"log_name": "B", "token": "t3", "step": 0, "rank": 0}]
        bank = {F.bank_key(r) for r in rows if r["token"] != "t3"}              # t3 is not in the bank
        self.assertEqual(F.bank_indices(rows, bank, heldout=False, ho_logs={"H"}), [0, 1, 2])
        self.assertEqual(F.bank_indices(rows, bank, heldout=True), [0, 1, 2, 3])
        pdm = {F.bank_key(rows[0]), F.bank_key(rows[3])}
        self.assertEqual(F.bank_indices(rows, bank, heldout=True, require_pdm=True, pdm_keys=pdm), [0, 3])
        idx = [0, 1, 2, 3]
        self.assertEqual(F.locate(rows, idx, [("A", "t1", 0, 0)], "exact", "x"), [0])
        self.assertEqual(F.locate(rows, idx, [("A", "t1", 0, 3)], "exact", "x"), [1])
        self.assertEqual(F.locate(rows, idx, [("A", "t1", 1, 0)], "exact", "x"), [2])
        self.assertEqual(F.locate(rows, idx, [("A", "t1", 0, None)], "first", "x"), [0])      # the trainer's own first-row de-duplication
        refused(self, lambda: F.locate(rows, idx, [("A", "t1", 0, 5)], "exact", "x"), "0 bank rows at rank 5")
        refused(self, lambda: F.locate(rows, idx, [("B", "t3", 0, 0)], "exact", "x"), "no bank row")
        refused(self, lambda: F.locate(rows + [dict(rows[0])], idx + [5], [("A", "t1", 0, 0)], "exact", "x"), "2 bank rows")

    def test_frame_list_sha_and_structure_refusals(self):
        E = self.E
        a = E.argv(self.out(), "head")

        def with_(flag, val):
            b = list(a)
            b[b.index(flag) + 1] = val
            return b
        refused(self, lambda: F.main(with_("--insample-list-sha256", "0" * 64), factory=E.factory), "does not start with")
        refused(self, lambda: F.main(with_("--insample-list-sha256", E.list_sha[:20]), factory=E.factory), "64 hex")
        refused(self, lambda: F.main(with_("--repro-list-sha256", "1" * 64), factory=E.factory), "does not start with")
        # a doctored list whose sha is declared correctly is still refused by the consumer's verifier
        doc = os.path.join(self.td.name, "doctored.tsv")
        txt = open(E.list_p, encoding="utf-8").read().split("\n")
        txt[1] = txt[1].replace(txt[1].split("\t")[1], txt[2].split("\t")[1], 1)       # a duplicate frame
        open(doc, "w", encoding="utf-8", newline="\n").write("\n".join(txt))
        b = with_("--insample-list", doc)
        b = list(b)
        b[b.index("--insample-list-sha256") + 1] = sha(doc)
        refused(self, lambda: F.main(b, factory=E.factory))
        os.remove(doc)
        # the list sha is checked on a mutant that skips it: the SAME check must go RED (see TMutations)

    def test_missing_inputs_are_refused(self):
        E = self.E
        for flag in ("--pod-step0", "--banked-pred", "--insample-list"):
            a = E.argv(self.out(), "both")
            a[a.index(flag) + 1] = os.path.join(self.td.name, "absent")
            refused(self, lambda: F.main(a, factory=E.factory), "not a file")
        a = E.argv(self.out(), "head")
        i = a.index("--banked-pred")
        del a[i:i + 4]
        refused(self, lambda: F.main(a, factory=E.factory), "--banked-pred")
        refused(self, lambda: F.main(E.argv(self.out(), "init")[:-2] + [], factory=E.factory))      # --pod-h48x-sha256 dropped


class TBankKeys(Base):
    def banks(self):
        return lambda ra: {"train": (self.E.ds, self.E.tr_idx)}

    def test_bank_keys_feed_the_consumers_make_list_and_reproduce_the_same_list(self):
        E = self.E
        out = os.path.join(self.td.name, "bk_out")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = F.main_bank_keys(["--argv-file", E.argv_file, "--run-dir", E.run, "--out-dir", out], self.banks())
        self.assertEqual(rc, 0, buf.getvalue())
        txt = open(os.path.join(out, "bank_keys.tsv"), encoding="utf-8").read()
        TAB, NL = chr(9), chr(10)
        self.assertTrue(txt.startswith(TAB.join(["log", "token", "step", "rank"]) + NL))
        self.assertEqual(txt.count(NL), 1 + len(E.tr_idx))
        self.assertNotIn(chr(13), txt)
        self.assertFalse(any(r[0] in E.ho_logs for r in (x.split(TAB) for x in txt.split(NL)[1:-1])), "a held-out log is in the train bank keys")
        lst = os.path.join(self.td.name, "bk_list")
        rc = VS.main(["make-list", "--out-dir", lst, "--bank-keys", os.path.join(out, "bank_keys.tsv"), "--store", E.fx.root, "--h48-logs", E.fx.h48_path, "--sft-heldout", E.fx.sft_path,
                      "--store-sha256-prefix", E.fx.prefix, "--per-city-for-tests", str(PER_CITY)])
        self.assertEqual(rc, 0)
        self.assertEqual(json.load(open(os.path.join(lst, "stage0_frames.json")))["frame_list_sha256"], E.list_sha)      # the same draw as the fixture's own bank

    def test_bank_keys_refuse_a_bank_that_is_not_the_one_the_run_trained_on(self):
        E = self.E
        E.write_events(lambda e: e.__setitem__("train_sets", len(E.tr_idx) + 1))
        try:
            refused(self, lambda: F.main_bank_keys(["--argv-file", E.argv_file, "--run-dir", E.run, "--out-dir", os.path.join(self.td.name, "bk_bad")], self.banks()), "train_sets")
        finally:
            E.write_events()
        out = os.path.join(self.td.name, "bk_out2")
        os.makedirs(out)
        open(os.path.join(out, "bank_keys.tsv"), "w").close()
        refused(self, lambda: F.main_bank_keys(["--argv-file", E.argv_file, "--run-dir", E.run, "--out-dir", out], self.banks()), "exists")
        self.assertEqual(F.main(["bank-keys", "--argv-file", E.argv_file, "--run-dir", E.run, "--out-dir", os.path.join(self.td.name, "bk_dry"), "--dry-run"], self.banks()), 0)
        self.assertFalse(os.path.exists(os.path.join(self.td.name, "bk_dry")))


class TGpu(Base):
    def setUp(self):
        self.old = F.GPU_PROBE

    def tearDown(self):
        F.GPU_PROBE = self.old

    def test_parse_nvsmi(self):
        self.assertEqual(F.parse_nvsmi("", 0), ([], ""))
        self.assertEqual(F.parse_nvsmi("1234, python, 20000 MiB\n", 0)[0], ["1234, python, 20000 MiB"])
        self.assertIsNone(F.parse_nvsmi("", 9)[0])

    def test_default_is_refuse_when_the_gpu_is_in_use(self):
        F.GPU_PROBE = lambda: (["4321, train.py, 31000 MiB"], "")
        refused(self, lambda: F.require_gpu_free("cuda", False, "t"), "gpu-busy")
        self.assertEqual(F.require_gpu_free("cuda", True, "t"), ["4321, train.py, 31000 MiB"])      # explicit opt-in only
        F.GPU_PROBE = lambda: ([], "")
        self.assertEqual(F.require_gpu_free("cuda", False, "t"), [])
        F.GPU_PROBE = lambda: (None, "nvidia-smi not found")
        refused(self, lambda: F.require_gpu_free("cuda", False, "t"), "gpu-unverifiable")
        self.assertEqual(F.require_gpu_free("cpu", False, "t"), [])

    def test_main_refuses_before_any_model_is_built_when_busy(self):
        E = self.E
        F.GPU_PROBE = lambda: (["1, trainer, 1 MiB"], "")
        built = []

        def factory(ra, dev):
            built.append(1)
            return E.factory(ra, dev)
        a = E.argv(self.out(), "head")
        a[a.index("cpu")] = "cuda"
        refused(self, lambda: F.main(a, factory=factory), "gpu-busy")
        self.assertEqual(built, [])

    def test_gpu_checked_again_right_before_the_forward(self):
        E = self.E
        seq = iter([([], ""), (["9, p7, 1 MiB"], "")])                     # idle at start, busy when the banks have loaded
        F.GPU_PROBE = lambda: next(seq)
        a = E.argv(self.out(), "head")
        a[a.index("cpu")] = "cuda"
        refused(self, lambda: F.main(a, factory=E.factory), "gpu-busy (before the forward)")

    def test_dry_run_never_initialises_cuda_and_reports_the_state(self):
        E = self.E
        F.GPU_PROBE = lambda: (["1, trainer, 1 MiB"], "")
        out = self.out()
        a = E.argv(out, "both", ["--dry-run"])
        a[a.index("cpu")] = "cuda"
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(F.main(a, factory=E.factory), 0)
        self.assertIn('"a_real_run_would": "refuse"', buf.getvalue())
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(os.path.exists(out))


class TRun(Base):
    def run_ok(self, mode="both", **kw):
        out = self.out()
        rc, txt = self.E.run_main(out, mode, **kw)
        return out, rc, txt

    def chk_trained_predictions(self, env=None):
        """the written in-sample / C0-ID predictions equal an independent forward of the TRAINED head, and are not the initial head's."""
        E = self.E
        out, rc, txt = self.run_ok("head", env=env)
        self.assertEqual(rc, 0, txt)
        for fname, keys in (("preds_insample.npz", E.list_keys), ("preds_repro256.npz", E.repro_keys)):
            lg, sd = VS.load_pred(os.path.join(out, fname), keys, what=fname)
            ref_l, ref_s = E.ref(E.trained, keys)
            self.assertLess(float(np.abs(lg - ref_l).max()), 1e-4, fname)
            self.assertLess(float(np.abs(sd - ref_s).max()), 1e-4, fname)
            init_l, _ = E.ref(E.init_aux, keys)
            self.assertGreater(float(np.abs(lg - init_l).max()), 1e-3, fname + ": equals the initial head")

    def test_trained_predictions_match_an_independent_forward(self):
        self.chk_trained_predictions()

    def test_both_mode_writes_every_file_and_the_consumer_accepts_them(self):
        E = self.E
        out, rc, txt = self.run_ok("both")
        self.assertEqual(rc, 0, txt)
        self.assertEqual(sorted(os.listdir(out)), ["head_init0_b.npz", "head_init0_rebuilt_a.npz", "preds_insample.npz", "preds_repro256.npz", "stage0_forward.json"])
        meta = json.load(open(os.path.join(out, "stage0_forward.json")))
        self.assertEqual(meta["status"], "OK")
        self.assertEqual({r["check"][:5] for r in meta["reproduction"]}, {"C0-ID", "S1-F "})
        for r in meta["reproduction"]:
            self.assertEqual(r["sign_agreement"], 1.0)
            self.assertLess(r["max_abs_logit_diff"], 1e-4)
        for k, v in meta["outputs"].items():
            self.assertEqual(sha(os.path.join(out, k)), v["sha256"])
        # the consumer's `read`: C0-ID (its OWN reproduction control) passes on my files
        rd = os.path.join(self.td.name, "out_read" + os.path.basename(out))
        rc2 = VS.main(["read", "--out-dir", rd, "--frame-list", E.list_p, "--frame-list-sha256", E.list_sha, "--preds-insample", os.path.join(out, "preds_insample.npz"), "--banked-pred", E.banked,
                       "--preds-repro", os.path.join(out, "preds_repro256.npz"), "--repro-n", str(REPRO), "--store", E.fx.root, "--h48-logs", E.fx.h48_path, "--sft-heldout", E.fx.sft_path,
                       "--n-boot", "50", "--store-sha256-prefix", E.fx.prefix, "--per-city-for-tests", str(PER_CITY)])
        self.assertEqual(rc2, 0)
        res = json.load(open(os.path.join(rd, "stage0_read.json")))
        self.assertTrue(res["controls"]["C0-ID"]["ok"], res["controls"]["C0-ID"])
        self.assertEqual(res["controls"]["C0-ID"]["n_frames"], REPRO)
        self.assertEqual(res["n_frames"], 4 * PER_CITY)
        # the consumer's `s1f --dry-run`: the rebuilt initial head reproduces the pod's step 0 (agreement 1.0) and the manifest builds from my files
        sj = os.path.join(self.td.name, "in_sample_iou_" + os.path.basename(out) + ".json")
        json.dump({"row": "S0-WEAK", "in_sample_iou": 0.5}, open(sj, "w"))
        bm = os.path.join(self.td.name, "base_manifest_" + os.path.basename(out) + ".json")
        json.dump({"files": {}}, open(bm, "w"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc3 = VS.main(["s1f", "--out-dir", os.path.join(out, "s1f"), "--base-manifest", bm, "--in-sample-json", sj, "--init-pod-step0", E.pod_step0,
                           "--init-rebuilt-a", os.path.join(out, "head_init0_rebuilt_a.npz"), "--init-rebuilt-b", os.path.join(out, "head_init0_b.npz"), "--dry-run"])
        self.assertEqual(rc3, 0)
        self.assertIn('"ok": true', buf.getvalue())
        # the init files carry the SEED-0 head's predictions on the right bank (the h48x frames come from the TRAIN bank, the step-0 frames from the held-out bank)
        lg, _ = VS.load_pred(os.path.join(out, "head_init0_b.npz"), E.h48x_keys, what="b")
        self.assertLess(float(np.abs(lg - E.ref(E.init_aux, E.h48x_keys)[0]).max()), 1e-4)

    def test_forward_is_order_and_batch_invariant(self):
        E = self.E
        o1, _, _ = self.run_ok("head", extra=["--batch", "2"])
        o2, _, _ = self.run_ok("head", extra=["--batch", "9"])
        a, _ = VS.load_pred(os.path.join(o1, "preds_insample.npz"), E.list_keys, what="a")
        b, _ = VS.load_pred(os.path.join(o2, "preds_insample.npz"), E.list_keys, what="b")
        self.assertLess(float(np.abs(a - b).max()), 1e-4)
        self.assertEqual(VS.sign_agreement(a, b), 1.0)

    def test_the_run_is_deterministic(self):
        o1, _, _ = self.run_ok("both")
        o2, _, _ = self.run_ok("both")
        for f in ("preds_insample.npz", "preds_repro256.npz", "head_init0_rebuilt_a.npz", "head_init0_b.npz"):
            z1, z2 = np.load(os.path.join(o1, f)), np.load(os.path.join(o2, f))
            for k in z1.files:
                self.assertTrue((z1[k] == z2[k]).all(), f + ":" + k)

    def test_reproduction_failure_gives_rc_3_after_writing(self):
        E = self.E
        bad = os.path.join(self.td.name, "banked_wrong.npz")
        TS.write_pred(bad, E.repro_keys, -E.ref(E.trained, E.repro_keys)[0])
        out = self.out()
        rc, txt = E.run_main(out, "head", extra=["--banked-pred", bad, "--banked-sha256", sha(bad)[:16]])
        self.assertEqual(rc, 3, txt)
        self.assertEqual(json.load(open(os.path.join(out, "stage0_forward.json")))["status"], "REPRODUCTION_FAILED")
        self.assertTrue(os.path.isfile(os.path.join(out, "preds_insample.npz")))

    def test_dry_run_writes_nothing_and_with_banks_resolves_every_frame(self):
        E = self.E
        out = self.out()
        rc, txt = E.run_main(out, "both", extra=["--dry-run"])
        self.assertEqual(rc, 0, txt)
        self.assertIn("DRY-RUN", txt)
        self.assertNotIn("RESOLVED", txt)
        rc, txt = E.run_main(out, "both", extra=["--dry-run", "--with-banks"])
        self.assertEqual(rc, 0, txt)
        self.assertIn("RESOLVED", txt)
        self.assertFalse(os.path.exists(out), "a dry run writes nothing")

    def test_existing_outputs_are_never_overwritten(self):
        out, rc, _ = self.run_ok("head")
        self.assertEqual(rc, 0)
        a = self.E.argv(out, "head")
        refused(self, lambda: F.main(a, factory=self.E.factory), "already holds")

    def test_a_frame_with_no_bank_row_is_refused_before_the_forward(self):
        E = self.E
        keep = E.tr_idx
        E.tr_idx = [i for i in keep if (E.rows[i]["log_name"], E.rows[i]["token"]) != E.list_keys[0]]
        try:
            out = self.out()
            with self.assertRaises(SystemExit) as cm:
                E.run_main(out, "head")
            self.assertIn("no bank row", str(cm.exception.code))
            self.assertFalse(os.path.exists(os.path.join(out, "preds_insample.npz")))
        finally:
            E.tr_idx = keep

    def test_head_run_never_writes_into_the_run_dir(self):
        before = sorted(os.listdir(self.E.run))
        self.run_ok("both")
        self.assertEqual(sorted(os.listdir(self.E.run)), before)


class TMutations(Base):
    """each deliberate regression must make its check fail (AssertionError) or make the tool refuse (SystemExit): the checks bite."""

    def red(self, fn, env):
        old = os.environ.get("VDAC_FWD_MUTATE")
        os.environ["VDAC_FWD_MUTATE"] = env
        try:
            with self.assertRaises((AssertionError, SystemExit), msg=f"mutation {env} was NOT caught"):
                fn()
        finally:
            if old is None:
                os.environ.pop("VDAC_FWD_MUTATE", None)
            else:
                os.environ["VDAC_FWD_MUTATE"] = old

    def run_check(self):
        t = TRun("test_trained_predictions_match_an_independent_forward")
        t.td, t.E = self.td, self.E
        return t

    def test_skip_head_load_is_refused_by_the_tool_and_by_the_prediction_check_when_the_guard_is_off(self):
        t = self.run_check()
        self.red(lambda: t.chk_trained_predictions(), "skip_head_load")                      # the in-tool guard: loaded head == initial head
        self.red(lambda: t.chk_trained_predictions(), "skip_head_load,no_guard")             # guard off: predictions equal the initial head's -> the test's own assertion goes RED

    def test_y_flipped_output_is_caught(self):
        self.red(lambda: self.run_check().chk_trained_predictions(), "y_flip_output")

    def test_key_misalignment_is_caught(self):
        self.red(lambda: self.run_check().chk_trained_predictions(), "misalign_keys")

    def test_init_seed1_is_caught(self):
        E = self.E

        def chk():
            out = os.path.join(self.td.name, "o_init_" + str(id(chk)) + str(os.environ.get("VDAC_FWD_MUTATE")))
            rc, txt = E.run_main(out, "init")
            self.assertEqual(rc, 0, txt)                                                     # the S1-F reproduction against the pod's seed-0 step 0 must hold
        chk()                                                                                # clean: passes
        self.red(chk, "init_seed1")                                                          # seed 1: reproduction fails (rc 3) -> RED

    def test_skipping_the_list_sha_check_is_caught(self):
        E = self.E

        def chk():
            a = E.argv(os.path.join(self.td.name, "o_sha_" + str(os.environ.get("VDAC_FWD_MUTATE"))), "head")
            a[a.index("--insample-list-sha256") + 1] = "0" * 64
            refused(self, lambda: F.main(a, factory=E.factory), "does not start with")
        chk()
        self.red(chk, "skip_list_sha")

    def test_skipping_the_gpu_guard_is_caught(self):
        old = F.GPU_PROBE
        F.GPU_PROBE = lambda: (["1, trainer, 1 MiB"], "")
        try:
            def chk():
                refused(self, lambda: F.require_gpu_free("cuda", False, "t"), "gpu-busy")
            chk()
            self.red(chk, "skip_gpu_guard")
        finally:
            F.GPU_PROBE = old

    def test_mutation_names_are_the_declared_ones(self):
        src = open(os.path.join(HERE, "vdac_stage0_forward.py"), encoding="utf-8").read()
        for m in F.MUTATIONS:
            self.assertIn(f'mutation("{m}")', src, m)


class TNoPinnedEdits(unittest.TestCase):
    def test_this_tool_imports_the_pinned_files_but_never_writes_them(self):
        src = open(os.path.join(HERE, "vdac_stage0_forward.py"), encoding="utf-8").read()
        for pinned in ("scorer_finetune.py", "launch_gate.py", "gate_checks.py"):
            self.assertNotIn(f'open(os.path.join(REFE_DIR, "{pinned}")', src)
        self.assertEqual(src.count(".write("), 1)                                                # the only text writer is the bank-keys TSV (f.write(txt)); the rest is np.savez / json.dump / os.replace on --out-dir files


if __name__ == "__main__":
    unittest.main(verbosity=2)
