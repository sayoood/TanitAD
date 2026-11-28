# ============================================================ G2: selection quality (random / actual / best-of-64)
import selection_quality as SQ  # noqa: E402
import csv as csv_mod  # noqa: E402

P2_TABLE = Path(os.environ.get("P2_FULL_TABLE", "C:/Users/Admin/p2full_work/final/table_p2_full.npz"))
# raw/2026-10-16-p2-full-table/anatomy.json (MEASURED 2026-10-18, zero_tail.Boot, B 10000, seed 20260927): the banked KNOWN VALUES of P2's own 64-proposal table
P2_ANATOMY = {"actual": (88.8184088897293, 87.86107970695247, 89.77011829946386), "best_of_64": (98.23109369448527, 97.81901516296715, 98.61925747503028),
              "random": (73.00314921968497, 71.48988586296944, 74.5661639319074)}
P2_ANATOMY_COUNTS = {"n_pick_zero": 509, "n_best_zero": 64, "n_best_strictly_above_pick": 8292}


def selq_vs_anatomy(z=None, tol=1e-9):
    z = z if z is not None else dict(np.load(P2_TABLE, allow_pickle=False))
    r = SQ.read_selection_quality(z["pdms"], z["pick"], [str(t) for t in z["token"]], [str(l) for l in z["log"]])
    for k, (m, lo, hi) in P2_ANATOMY.items():
        got = r["levels_x100"][k]
        assert abs(got["mean"] - m) < tol and abs(got["ci95"][0] - lo) < tol and abs(got["ci95"][1] - hi) < tol, (k, got, (m, lo, hi))
    for k, v in P2_ANATOMY_COUNTS.items():
        assert r[k] == v, (k, r[k], v)
    return r


@check("G2 selection quality: P2's banked 64-proposal navtest table reproduces anatomy.json (pick 88.8184 / best-of-64 98.2311 / random 73.0031, CIs, counts) to 1e-9")
def t_g2_selq_known_value():
    if not P2_TABLE.exists():
        raise AssertionError(f"{P2_TABLE} absent: the known-value control cannot run")
    r = selq_vs_anatomy()
    assert r["random_control"]["actual_beats_random"] is True and "INVALID" not in r


@check("G2 RED: a shuffled pick / a duplicated proposal column / a token-shuffled table each FAIL the known-value control")
def t_g2_selq_known_value_red():
    z = dict(np.load(P2_TABLE, allow_pickle=False))
    rng = np.random.default_rng(3)
    a = dict(z)
    a["pick"] = rng.integers(0, 64, len(z["pick"]))                                 # the pick no longer the scorer's
    must_fail(selq_vs_anatomy, a)
    b = dict(z)
    b["pdms"] = z["pdms"].copy()
    b["pdms"][:, 63] = z["pdms"][:, 0]                                              # 63 distinct proposals: random and best move
    must_fail(selq_vs_anatomy, b)
    c = dict(z)
    c["pdms"] = z["pdms"][rng.permutation(len(z["pdms"]))]                          # token -> scores mapping broken
    must_fail(selq_vs_anatomy, c)


def _synthetic(n_tok=40, n_log=5, seed=11):
    rng = np.random.default_rng(seed)
    pdms = rng.uniform(0, 1, (n_tok, 64))
    pick = np.argmax(pdms, 1)                                                       # a perfect scorer on half the tokens, random on the rest
    pick[::2] = rng.integers(0, 64, len(pick[::2]))
    toks = [f"tok{i:03d}" for i in range(n_tok)]
    logs = [f"log{i % n_log}" for i in range(n_tok)]
    return pdms, pick, toks, logs


@check("G2 selection quality on a synthetic world with analytic answers: actual = the pick's PDMS, random = the mean over 64, best = the max over 64; differences consistent")
def t_g2_selq_synthetic():
    pdms, pick, toks, logs = _synthetic()
    r = SQ.read_selection_quality(pdms, pick, toks, logs, n=200)
    L = r["levels_x100"]
    assert L["actual"]["mean"] == float((pdms[np.arange(len(pick)), pick] * 100).mean())
    assert abs(L["random"]["mean"] - float(pdms.mean() * 100)) < 1e-9
    assert abs(L["best_of_64"]["mean"] - float(pdms.max(1).mean() * 100)) < 1e-9
    D = r["paired_differences_x100"]
    assert abs(D["best_minus_actual"]["mean"] - (L["best_of_64"]["mean"] - L["actual"]["mean"])) < 1e-9
    assert abs(D["actual_minus_random"]["mean"] - (L["actual"]["mean"] - L["random"]["mean"])) < 1e-9
    assert L["random"]["mean"] < L["actual"]["mean"] < L["best_of_64"]["mean"]
    bad = pdms.copy()
    bad[3, 5] = np.nan
    must_fail(SQ.read_selection_quality, bad, pick, toks, logs, 50)                # NaN is not a read
    must_fail(SQ.read_selection_quality, pdms[:, :63], pick, toks, logs, 50)       # 63 proposals is not a read


@check("G2 selection quality: random >= actual is flagged INVALID (an instrument failure, C3's rule), never reported as a result")
def t_g2_selq_invalid_flag():
    pdms, _pick, toks, logs = _synthetic()
    worst = np.argmin(pdms, 1)                                                      # a scorer that picks the WORST proposal
    with tempfile.TemporaryDirectory() as td:
        tb = Path(td) / "t.npz"
        np.savez(tb, pdms=pdms, pick=worst, token=np.array(toks), log=np.array(logs))
        r = SQ.read_table(tb)
        assert "INVALID" in r and r["random_control"]["random_ge_actual_point"] is True
        np.savez(tb, pdms=pdms, pick=np.argmax(pdms, 1), token=np.array(toks), log=np.array(logs))
        assert "INVALID" not in SQ.read_table(tb)


def _fake_dump(td: Path, n=6):
    rng = np.random.default_rng(5)
    props = rng.uniform(10, 90, (n, 64, 8, 3)).astype(np.float32)
    toks = np.array([f"tok{i}" for i in range(n)])
    pick = rng.integers(0, 64, n)
    p = td / "dump.npz"
    np.savez(p, token=toks, fingerprint=np.array([f"fp{i}" for i in range(n)]), proposals=props, logits=np.zeros((n, 64, 6), np.float32), pick=pick, sampling=np.array([8, 0.5]))
    return p, toks, props, pick


def _fake_runner_factory(calls, fail_k=None):
    def runner(label, seam, tokens, out, log):
        calls.append(label)
        k = int(label.rsplit("_p", 1)[1])
        z = np.load(seam)
        sc = np.clip(z["poses"][:, -1, 0] / 100.0, 0, 1)
        d = Path(out) / label
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{label}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv_mod.writer(f)
            w.writerow(["token", "valid", "score", *SQ.SUB])
            for t, s in zip(z["token"], sc):
                w.writerow([str(t), "True", repr(float(s)), *["1.0"] * 6])
            w.writerow(["average", "True", "0.5", *["1.0"] * 6])
        st = {"label": label, "status": "FAIL" if k == fail_k else "PASS", "csv_valid_rows": len(sc)}
        txt = json.dumps(st) + "\n"
        Path(log).write_text(txt, encoding="utf-8")
        return txt
    return runner


@check("G2 the 64-column chain (seams -> harness columns -> table -> read) on a synthetic dump with a fake runner: gates PASS, resumable, and every RED arm fails")
def t_g2_selq_chain():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dump, toks, props, pick = _fake_dump(td)
        root = td / "data"
        n = SQ.write_column_seams(dump, "x", root)
        assert n == 6 and all(SQ.seam_k("x", k, root).exists() for k in range(64))
        assert np.array_equal(np.load(SQ.seam_k("x", 7, root))["poses"], props[:, 7])      # column k = proposals[:, k]
        calls = []
        res = SQ.score_columns("x", n, workers=3, runner=_fake_runner_factory(calls), root=root)
        assert res["ok"] and len(calls) == 64
        calls2 = []
        res2 = SQ.score_columns("x", n, workers=3, runner=_fake_runner_factory(calls2), root=root)
        assert res2["ok"] and calls2 == [], "a PASSed column was re-scored: not resumable"
        # the arm's own csv = the harness score of the picked proposal
        arm = td / "arm.csv"
        with open(arm, "w", newline="", encoding="utf-8") as f:
            w = csv_mod.writer(f)
            w.writerow(["token", "valid", "score", *SQ.SUB])
            for i, t in enumerate(toks):
                w.writerow([str(t), "True", repr(float(np.clip(props[i, pick[i], -1, 0] / 100.0, 0, 1))), *["1.0"] * 6])
            w.writerow(["average", "True", "0.5", *["1.0"] * 6])
        tl = {str(t): f"log{i % 2}" for i, t in enumerate(toks)}
        g = SQ.assemble_table(dump, "x", arm, td / "table.npz", tl, root)
        assert g["ok"] and g["G2_pdms_at_pick_equals_arm_csv"]["n_mismatch"] == 0, g
        r = SQ.read_table(td / "table.npz")
        z = np.load(td / "table.npz")
        assert abs(r["levels_x100"]["best_of_64"]["mean"] - float(z["pdms"].max(1).mean() * 100)) < 1e-9
        # RED 1: the arm's csv disagrees with the table at the pick on one token -> G2 fails (the dump is not the run that was scored)
        bad = td / "arm_bad.csv"
        bad.write_text(arm.read_text(encoding="utf-8").replace(repr(float(np.clip(props[2, pick[2], -1, 0] / 100.0, 0, 1))), "0.123456", 1), encoding="utf-8")
        assert SQ.assemble_table(dump, "x", bad, td / "t2.npz", tl, root)["ok"] is False
        # RED 2: one column FAILs the harness -> score_columns not ok, names the column
        root2 = td / "data2"
        SQ.write_column_seams(dump, "y", root2)
        r2 = SQ.score_columns("y", n, workers=2, runner=_fake_runner_factory([], fail_k=9), root=root2)
        assert r2["ok"] is False and r2["failed"] == [9]
        # RED 3: the column holding the pick scored from ANOTHER proposal's poses -> the table at the pick no longer equals the arm's csv -> G2 fails
        d = np.load(dump)
        np.savez(SQ.seam_k("x", int(pick[0]), root), token=d["token"], fingerprint=d["fingerprint"], poses=props[:, (int(pick[0]) + 1) % 64], sampling=np.array([8, 0.5]), arm=np.array("m"))
        for k in range(64):
            shutil.rmtree(SQ.csv_k("x", k, root).parent, ignore_errors=True)
        for p in (root / "score" / "_selq_x_logs").glob("*.log"):
            p.unlink()
        SQ.score_columns("x", n, workers=3, runner=_fake_runner_factory([]), root=root)
        assert SQ.assemble_table(dump, "x", arm, td / "t3.npz", tl, root)["ok"] is False


