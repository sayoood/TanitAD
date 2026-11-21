r"""Tests of hierrep_gaps.py on SYNTHETIC dumps only (no HIERREP data, no model).
Run: CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe test_hierrep_gaps.py
(taniteval is imported from TANITEVAL_ROOT through hierrep_read; PYTHONPATH entries are joined with ';' on Windows.)"""
import argparse
import os
import shutil
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import hierrep_gaps as HG  # noqa: E402

TMP = HERE + "/_tmp_test"
C = 3                       # episodes
WS = (3, 7)


def make_tables():
    n_raw = 200
    lead = {"clip": np.array([f"c{i}" for i in range(C)]), "has": np.ones((C, n_raw), np.int8), "lead": np.ones((C, n_raw), np.int8),
            "gap": np.full((C, n_raw), 20.0), "v": np.full((C, n_raw), 5.0)}
    n = 101
    gy = np.zeros(C * n, np.uint32)
    s_idx = np.tile(np.arange(n), C)
    gy[s_idx % 2 == 0] = 1 << 6
    lab = {"clip": lead["clip"], "off": np.arange(C) * n, "n": np.full(C, n), "has": np.ones(C * n, bool), "goal_y": gy, "goal_w": np.full(C * n, 1 << 6, np.uint32),
           "lon_cls": np.where(s_idx % 2 == 0, 2, 3).astype(np.int16)}
    return {"lead_table": lead, "label_table": lab}


def make_dump(d, offset=(0.0, 0.0), n_files=C):
    os.makedirs(d, exist_ok=True)
    k = np.arange(1, 31)[:, None]
    g1 = np.concatenate([k * 1.0, np.zeros((30, 1))], axis=1)           # straight along +x at 5 m/s
    for c in range(n_files):
        g = np.tile(g1, (len(WS), 1, 1))
        np.savez(f"{d}/ep{c:04d}.npz", g=g, cl=g + np.asarray(offset), ws=np.array(WS), eid=np.full(len(WS), c), clip_index=np.full(len(WS), c), v0=np.full(len(WS), 5.0))


def make_ticks(path, flip=()):
    rows = []
    for c in range(C):
        for w, ws in enumerate(WS):
            for t in range(3):
                rows.append((c, ws + 10 * t, t, t))
    ci, j, age, tick = (np.array(x) for x in zip(*rows))
    s = j - age
    lab = s % 2 == 0
    p = np.where(lab, 0.9, 0.1)
    for r in flip:
        p[r] = 1.0 - p[r]
    tgp = np.full((len(ci), 22), 0.1)
    tgp[:, 6] = p
    lon = np.zeros((len(ci), 7))
    lon[np.arange(len(ci)), np.where(p >= 0.5, 2, 3)] = 5.0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, ci=ci, j=j, age_t=age, tick=tick, t_goal_prob=tgp, t_lon_logits=lon)
    return lab


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)
        os.makedirs(TMP)
        cls.tab = make_tables()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_g2_identity_is_exactly_zero(self):
        make_dump(TMP + "/id")
        out, meta = HG.run_arm(TMP + "/id", None, self.tab, expect_files=C, identity=True)
        for k in HG.A_KEYS:
            self.assertEqual(float(np.abs(out["A_" + k + "_sum"]).max()), 0.0)
        self.assertEqual(int(out["A_gap_abs_cnt"].sum()), C * len(WS) * 30)

    def test_planted_offset_reads_2m(self):
        make_dump(TMP + "/off", offset=(2.0, 0.0))                       # rolled ego 2.0 m ahead along the path
        out, _ = HG.run_arm(TMP + "/off", None, self.tab, expect_files=C)
        n = out["A_gap_abs_cnt"].sum()
        self.assertAlmostEqual(out["A_gap_abs_sum"].sum() / n, 2.0, places=9)
        self.assertAlmostEqual(out["A_gap_signed_sum"].sum() / n, -2.0, places=9)      # closer
        self.assertEqual(int(out["A_n_cross2"].sum()), 0)
        make_dump(TMP + "/offy", offset=(0.0, 2.5))                       # lateral offset: no gap error, counted as cross-path
        o2, _ = HG.run_arm(TMP + "/offy", None, self.tab, expect_files=C)
        self.assertEqual(float(np.abs(o2["A_gap_abs_sum"]).max()), 0.0)
        self.assertEqual(int(o2["A_n_cross2"].sum()), C * len(WS) * 30)

    def test_time_gap_share_planted(self):
        # logged: gap 20 m at 5 m/s -> tg 4.0 s; rolled 19.5 m further ahead -> gap 0.5 m -> tg 0.1 s < 1.0 s
        make_dump(TMP + "/tgp", offset=(19.5, 0.0))
        out, _ = HG.run_arm(TMP + "/tgp", None, self.tab, expect_files=C)
        self.assertEqual(out["A_tg_share_sum"].sum(), out["A_tg_share_cnt"].sum())
        self.assertEqual(out["A_tg_share_cnt"].sum(), C * len(WS) * 30)

    def test_planted_decision_flip_reads_in_confusion(self):
        make_dump(TMP + "/b")
        lab = make_ticks(TMP + "/tk_good.npz")
        with np.load(TMP + "/tk_good.npz") as z:
            tk = {k: z[k] for k in z.files}
        ep_ids = np.arange(C)
        good = HG.part_b(tk, self.tab["label_table"], ep_ids)
        npos, nneg = int(lab.sum()), int((~lab).sum())
        self.assertEqual(good["stop_all"].sum(0).tolist(), [npos, 0, 0, nneg])
        self.assertEqual(float(HG.ba_stat(good["stop_all"].sum(0))), 1.0)
        make_ticks(TMP + "/tk_flip.npz", flip=(0,))                       # one decision flipped
        with np.load(TMP + "/tk_flip.npz") as z:
            tk2 = {k: z[k] for k in z.files}
        fl = HG.part_b(tk2, self.tab["label_table"], ep_ids)
        r0_pos = bool(lab[0])
        exp = [npos - 1, 1, 0, nneg] if r0_pos else [npos, 0, 1, nneg - 1]
        self.assertEqual(fl["stop_all"].sum(0).tolist(), exp)
        self.assertLess(float(HG.ba_stat(fl["stop_all"].sum(0))), 1.0)
        # AUROC: perfect separation = 1.0, full flip = 0.0
        P = HG.auc_prep(good["row_score"], good["row_label"], good["row_ep"])
        self.assertEqual(float(HG.auc_w(P, np.ones(len(P["y"])))), 1.0)
        Pf = HG.auc_prep(1.0 - good["row_score"], good["row_label"], good["row_ep"])
        self.assertEqual(float(HG.auc_w(Pf, np.ones(len(Pf["y"])))), 0.0)
        # sample-and-hold label step: the row of tick 2 reads cache step j - age_t, not j
        self.assertEqual(int(good["stop_t0"].sum()), C * len(WS))

    def test_missing_episode_refuses(self):
        for arm in ("HIER", "FLAT"):
            for s in (0, 1):
                d = f"{TMP}/lay/{arm}_s{s}"
                make_dump(f"{d}/dump_{arm}_s{s}_m10_causal_heads", n_files=(C - 1 if (arm, s) == ("FLAT", 1) else C))
                make_ticks(f"{d}/ticks/ticks_{arm}_s{s}_m10_causal_heads.npz")
        os.makedirs(TMP + "/tabs", exist_ok=True)
        np.savez(TMP + "/tabs/lead_table.npz", **self.tab["lead_table"])
        np.savez(TMP + "/tabs/label_table.npz", **self.tab["label_table"])
        a = argparse.Namespace(t1_root=TMP + "/lay", tables_dir=TMP + "/tabs", out_dir=TMP + "/out", pairs="HIER:FLAT", seeds="0,1", map="", runs_root="", n_boot=50,
                               n_boot_control=20, expect_files=C, no_card_counts=True)
        self.assertEqual(HG.cmd_read(a), 3)
        with open(TMP + "/out/VERDICT_HIERGAP.md", encoding="utf-8") as f:
            self.assertIn("READ NOT VALID", f.read())

    def test_read_runs_end_to_end_and_g2_holds(self):
        for arm in ("HIER", "FLAT"):
            for s in (0, 1):
                d = f"{TMP}/lay2/{arm}_s{s}"
                make_dump(f"{d}/dump_{arm}_s{s}_m10_causal_heads", offset=((0.5 if arm == "FLAT" else 0.0), 0.0))
                make_ticks(f"{d}/ticks/ticks_{arm}_s{s}_m10_causal_heads.npz")
        os.makedirs(TMP + "/tabs2", exist_ok=True)
        np.savez(TMP + "/tabs2/lead_table.npz", **self.tab["lead_table"])
        np.savez(TMP + "/tabs2/label_table.npz", **self.tab["label_table"])
        a = argparse.Namespace(t1_root=TMP + "/lay2", tables_dir=TMP + "/tabs2", out_dir=TMP + "/out2", pairs="HIER:FLAT", seeds="0,1", map="", runs_root="", n_boot=50,
                               n_boot_control=20, expect_files=C, no_card_counts=True)
        rc = HG.cmd_read(a)
        import json
        with open(TMP + "/out2/RESULT_HIERGAP.json", encoding="utf-8") as f:
            r = json.load(f)
        self.assertTrue(r["controls"]["g2_identity"]["PASS"])
        self.assertTrue(r["controls"]["g3_completeness"]["PASS"])
        c = r["pairs"]["HIER_vs_FLAT"]["cells"]["A:gap_abs"]
        self.assertAlmostEqual(c["D"], -0.5, places=9)                    # HIER has zero error, FLAT 0.5 m: hier minus flat = -0.5
        self.assertIn(rc, (0, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
