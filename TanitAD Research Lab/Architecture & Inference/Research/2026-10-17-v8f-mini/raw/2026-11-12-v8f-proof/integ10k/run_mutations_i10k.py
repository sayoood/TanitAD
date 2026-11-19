"""run_mutations_i10k.py -- counted mutation list of the tools of card PREREG_V8F_PROOF_ADD46 (numpy only). Each mutation is a text edit of a COPY of this folder (the old text must occur
exactly once); test_integ10k.py is run in the copy and must turn RED; the unmutated folder is run first and last and must be GREEN. Result: MUTATIONS_INTEG10K.json.
The four mutations the card names: min_dropped, reader_fed_future_frame, reader_of_the_other_seed, heldout_split_ignored (the rest are further).
Run: python run_mutations_i10k.py   (serial, CUDA_VISIBLE_DEVICES=-1; temp copies under work/mut, basetemp under work/bt/m)
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MUT = os.path.join(HERE, "work", "mut")
BT = os.path.join(HERE, "work", "bt", "m")
DIST = os.path.normpath(os.path.join(HERE, "..", "distrule"))
FILES = ["tok_reader.py", "qmap_fit.py", "qcal_map.py", "reader_controls.py", "read_i10k.py", "v8f_t1_eval_reader_rule.py", "make_tok_probe_lr.py", "pod_integ10k_queue.sh", "test_integ10k.py", "tok_probe_lr.py"]
M = [
    ("min_dropped", "v8f_t1_eval_reader_rule.py", "used = np.minimum(g, q_raw(g, xs, ys))", "used = q_raw(g, xs, ys)", "card"),
    ("reader_fed_future_frame", "tok_reader.py", "E[idx - (reader.hist - 1 - k)]", "E[np.minimum(idx + k, T - 1)]", "card"),
    ("reader_of_the_other_seed", "pod_integ10k_queue.sh", 'rollout I10RR RR "$k" "$k"', 'rollout I10RR RR "$k" "$((1 - k))"', "card"),
    ("heldout_split_ignored", "make_tok_probe_lr.py", "'    clips = [c for c in clips if not is_heldout(c)]\\n'", "'    clips = [c for c in clips]\\n'", "card"),
    ("heldout_rule_mod_20", "tok_reader.py", "return clip_hash_value(clip_id) % 10 == 0", "return clip_hash_value(clip_id) % 20 == 0", "further"),
    ("fit_part_equals_heldout", "tok_reader.py", "return clip_hash_value(clip_id) % 20 == 0", "return clip_hash_value(clip_id) % 10 == 0", "further"),
    ("none_counts_as_pass", "read_i10k.py", 'all(v.get("PASS") is True for v in res["controls"].values())', 'all(v.get("PASS") is not False for v in res["controls"].values())', "further"),
    ("cut_needs_only_one_seed", "read_i10k.py", 'all(d < 0 for d in e1["d_s"]) and e1["A_excludes_0"] and crossed_below', 'any(d < 0 for d in e1["d_s"]) and e1["A_excludes_0"] and crossed_below', "further"),
    ("share_bar_lowered", "read_i10k.py", "np.isfinite(s) and s >= 0.5", "np.isfinite(s) and s >= 0.05", "further"),
    ("map_band_opened", "qmap_fit.py", "BAND = (0.15, 0.25)", "BAND = (0.0, 1.0)", "further"),
    ("RRid_constant_not_1e9", "v8f_t1_eval_reader_rule.py", '"RRid": 1e9', '"RRid": 1.0', "further"),
    ("presence_threshold_zero", "v8f_t1_eval_reader_rule.py", "P_MIN = 0.5", "P_MIN = 0.0", "further"),
    ("crossed_run_uses_own_reader", "pod_integ10k_queue.sh", "rollout I10RRx RR 0 1", "rollout I10RRx RR 0 0", "further"),
    ("slot_wait_shortened", "pod_integ10k_queue.sh", "      sleep 45\n", "      sleep 0\n", "further"),
    ("k5_tolerance_ignored", "read_i10k.py", 'd1 <= tol and d2 <= tol and sha_ok', 'sha_ok', "further"),
    ("k9_band_dropped", "read_i10k.py", "0.15 <= qm[k][\"coverage\"][\"share_below_Q\"] <= 0.25", "True", "further"),
]


def run(folder, tag):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", DISTRULE_DIR=DIST, TOKPROBE_GR_SRC=os.path.normpath(os.path.join(HERE, "..", "finegrid_r", "tok_probe_gr.py")))
    bt = os.path.join(BT, tag)
    os.makedirs(os.path.dirname(bt), exist_ok=True)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", bt, os.path.join(folder, "test_integ10k.py")], capture_output=True, env=env, cwd=folder, timeout=600)
    out = p.stdout.decode("utf-8", "replace")
    return p.returncode, out.strip().splitlines()[-1] if out.strip() else ""


def main():
    res = {"card": "PREREG_V8F_PROOF_ADD46", "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "control_before": None, "control_after": None, "mutations": []}
    base = os.path.join(MUT, "base")
    os.makedirs(base, exist_ok=True)
    for f in FILES:
        shutil.copy2(os.path.join(HERE, f), os.path.join(base, f))
    rc, last = run(base, "c0")
    res["control_before"] = {"rc": rc, "last": last}
    for i, (mid, f, old, new, kind) in enumerate(M):
        d = os.path.join(MUT, "m%02d" % i)
        os.makedirs(d, exist_ok=True)
        for g in FILES:
            shutil.copy2(os.path.join(HERE, g), os.path.join(d, g))
        t = open(os.path.join(d, f), encoding="utf-8", newline="").read()
        n = t.count(old)
        if n != 1:
            res["mutations"].append({"id": mid, "file": f, "kind": kind, "error": "old text occurs %d times" % n, "red": False})
            continue
        with open(os.path.join(d, f), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(t.replace(old, new))
        rc, last = run(d, "m%02d" % i)
        res["mutations"].append({"id": mid, "file": f, "kind": kind, "red": rc != 0, "last": last})
        print(mid, "RED" if rc != 0 else "STILL GREEN", last[:100], flush=True)
    rc, last = run(base, "c1")
    res["control_after"] = {"rc": rc, "last": last}
    res["n_mutations"] = len(M)
    res["n_red"] = sum(1 for m in res["mutations"] if m["red"])
    res["n_card_named"] = sum(1 for m in res["mutations"] if m["kind"] == "card")
    res["n_card_named_red"] = sum(1 for m in res["mutations"] if m["kind"] == "card" and m["red"])
    res["all_red_and_controls_green"] = bool(res["n_red"] == len(M) and res["control_before"]["rc"] == 0 and res["control_after"]["rc"] == 0)
    with open(os.path.join(HERE, "MUTATIONS_INTEG10K.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print("controls", res["control_before"]["rc"], res["control_after"]["rc"], "red", res["n_red"], "of", len(M))
    return 0 if res["all_red_and_controls_green"] else 1


if __name__ == "__main__":
    sys.exit(main())
