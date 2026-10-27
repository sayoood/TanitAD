FIXTURES = HERE / "fixtures" / "dagger_smoke"                              # the generator's REAL smoke output (PROVENANCE.json, bank.json, one shard provenance): the templates of every synthetic bank
WIN_SHA = hashlib.sha256(b"synthetic TRAIN window list").hexdigest()       # what the trainer would record as the TRAIN window-list sha256
THR_SHA = "438366d5" + "ab" * 28                                           # sha256 of probe/THRESHOLDS_FIXED.txt starts with the PREREG's 438366d5
TICK1 = {"ok": 270000, "off_lat": 3000, "off_head": 1000, "end_edge": 32566, "nonfinite": 0}        # 306,566 own states at tick 1
TICK2 = {"ok": 160000, "off_lat": 100000, "off_head": 20000, "end_edge": 26561, "nonfinite": 5}     # 306,566 own states at tick 2
KB45 = {"K-B4": {"PASS": True, "cases": {k: True for k in ("straight", "stationary_tie", "lateral_cut", "heading_cut", "end_edge", "loop")}},
        "K-B5": {"PASS": True, "max_pose_now_err_m": 0.0, "max_rel_transform_err_m": 2e-7}}              # the `controls` block the generator does NOT write yet (needs the bank generator to add)


def real_doc(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def npz_bytes(clips):
    import io
    b = io.BytesIO()
    np.savez(b, header_json=np.frombuffer(json.dumps({"clips": clips, "schema": "v8f-dagger-bank-v1"}, sort_keys=True).encode("utf-8"), np.uint8), ci=np.zeros(3, np.int32))
    return b.getvalue()


def finalize(B):
    """Recompute every hash a bank document carries from its current content (so a test edit changes only what it means to change)."""
    for i, sh in enumerate(B["shards"]):
        sh["bytes"] = npz_bytes(sh["clips"])
        sh["sha256"] = hashlib.sha256(sh["bytes"]).hexdigest()
        B["man"]["shards"][i]["file_sha256"] = sh["sha256"]
        B["shard_provs"][i]["file_sha256"] = sh["sha256"]
    B["man_bytes"] = (json.dumps(B["man"], sort_keys=True, indent=1) + "\n").encode("utf-8")
    B["man_sha256"] = hashlib.sha256(B["man_bytes"]).hexdigest()
    B["prov"]["bank_sha256"] = B["man_sha256"]
    return B


def make_bank(m2b_sha, n_elig=250000, clips=None):
    """A good COMPLETE bank in the generator's real layout, built from its real smoke documents as templates (so the key structure IS the generator's).
    The bank holds `clips` (default: all 4,512 TRAIN clips), split over two shards."""
    ids = list(CLIPS if clips is None else clips)
    prov, man, sp0 = real_doc("PROVENANCE.json"), real_doc("bank.json"), real_doc("shard_000of001.provenance.json")
    half = N_WIN // 2
    cov = {"n_windows": N_WIN, "n_eligible": n_elig, "fraction": n_elig / N_WIN, "min": 0.5, "blocked": n_elig / N_WIN < 0.5}
    rbt = {"1": dict(TICK1), "2": dict(TICK2)}
    c = man["common"]
    c.update(ckpt_sha256=m2b_sha, partial=False, limit_clips=None, window_list_sha256=WIN_SHA, thresholds_file_sha256=THR_SHA)
    c["train"] = {"exclude": "540:600", "n_clips": 4512, "n_excluded": 0, "n_windows": N_WIN, "train_sha256": TRAIN_SHA}
    c["eval"] = {**c["eval"], "clips": sorted(EVAL_IDS), "n_clips": 141, "overlap": 0}
    man.update(n_shards=2, partial=False, n_records=2 * N_WIN, coverage=dict(cov), reasons_by_tick=json.loads(json.dumps(rbt)))
    man["shards"] = [{"file": f"shard_{k:03d}of002.npz", "k": k, "n_records": 2 * half, "file_sha256": "", "content_sha256": f"{k:02x}" * 32} for k in range(2)]
    shard_provs = []
    for k in range(2):
        sp = json.loads(json.dumps(sp0))
        sp["plan"].update(shard=[k, 2], window_list_sha256=WIN_SHA, train_sha256=TRAIN_SHA, n_train_clips=4512, n_train_windows=N_WIN, limit_clips=None, partial=False, shard_windows=half)
        sp["inputs"]["ckpt"]["sha256"] = m2b_sha
        sp.update(n_records=2 * half, kb0_tick0_identity={"checked": half, "mismatch": 0}, kb1_replay={"chunks": 24, "windows": 768, "mismatch_chunks": 0, "frac": 0.005, "seed": 0},
                  kb3_eval_disjoint={"n_eval_clips": 141, "overlap": 0})
        shard_provs.append(sp)                                                                        # a shard's own coverage / reasons are not read by the reader
    prov.update(n_records=2 * N_WIN, coverage=dict(cov), reasons_by_tick=json.loads(json.dumps(rbt)), partial=False, warnings=[],
                shard_provenance=[f"shard_{k:03d}of002.provenance.json" for k in range(2)], controls=json.loads(json.dumps(KB45)))
    B = {"dir": None, "prov": prov, "man": man, "shard_provs": shard_provs,
         "shards": [{"file": man["shards"][k]["file"], "clips": {str(i): ids[i] for i in range(len(ids)) if i % 2 == k}} for k in range(2)]}
    return finalize(B)


def write_bank(B, d):
    d = pathlib.Path(d)
    d.mkdir(parents=True, exist_ok=True)
    jdump(d / "PROVENANCE.json", B["prov"])
    (d / "bank.json").write_bytes(B["man_bytes"])
    for sh in B["shards"]:
        (d / sh["file"]).write_bytes(sh["bytes"])
    for nm, sp in zip(B["prov"]["shard_provenance"], B["shard_provs"]):
        jdump(d / nm, sp)
    return str(d / "PROVENANCE.json")
