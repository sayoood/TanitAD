"""Patch 4: the preflight WRITES its partial report on any gate failure (a raise that
discards the measurement is how gate 5 fired with no number attached), and gate 5
records the live-vs-live repeat (the harness's own noise floor) and the window ids
beside the live-vs-reference divergence."""
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
D = os.path.join(ROOT, "stack/scripts/rl_refcv3_min.py")
s = open(D, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, f"expected {count}, found {n}: {old[:70]!r}"
    s = s.replace(old, new)


# gate 5: the noise floor + window ids, and the numbers in the raise message
rep('''    with torch.no_grad():
        live = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=int(prov["decoder_steps"]))
        ref = reference(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=int(prov["decoder_steps"]))
    div = trajectory_divergence(live["anchor_traj"], ref["anchor_traj"])
    rep["anchor_step0"] = {"max_divergence_m2": float(div.abs().max()), "n_frozen": n_frozen,
                           "PASS": float(div.abs().max()) == 0.0}
    if not rep["anchor_step0"]["PASS"]:
        raise SystemExit("[preflight] ⛔ step-0 divergence != 0 — a mode mismatch (TRAIN-C5 family)")
''', '''    with torch.no_grad():
        live = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=int(prov["decoder_steps"]))
        live2 = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=int(prov["decoder_steps"]))
        ref = reference(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=int(prov["decoder_steps"]))
    div = trajectory_divergence(live["anchor_traj"], ref["anchor_traj"])
    div_ll = trajectory_divergence(live["anchor_traj"], live2["anchor_traj"])   # the harness's own noise floor
    rep["anchor_step0"] = {"max_divergence_m2": float(div.abs().max()),
                           "live_vs_live_repeat_m2": float(div_ll.abs().max()),
                           "logits_live_vs_ref_maxabs": float((live["anchor_logits"] - ref["anchor_logits"]).abs().max()),
                           "wis": [int(w) for w in b["wis"]], "n_frozen": n_frozen,
                           "PASS": float(div.abs().max()) == 0.0}
    if not rep["anchor_step0"]["PASS"]:
        raise SystemExit(f"[preflight] ⛔ step-0 divergence != 0 — a mode mismatch (TRAIN-C5 family): "
                         f"live-vs-ref {rep['anchor_step0']['max_divergence_m2']:.6g} m², "
                         f"live-vs-live {rep['anchor_step0']['live_vs_live_repeat_m2']:.6g} m² on windows "
                         f"{rep['anchor_step0']['wis']}")
''')

# the partial report is WRITTEN on any failure: wrap the body via a helper
rep('''def mode_preflight(a) -> int:
    t_all = time.time()
    device = a.device if torch.cuda.is_available() else "cpu"
    rep: dict = {"_what": "D-RL-REFCV3-MIN preflight — NOTHING trained, no checkpoint written",
''', '''def mode_preflight(a) -> int:
    """Wrapper: the partial report is WRITTEN whenever a gate raises, so a failure
    carries its measurement instead of discarding it (2026-09-05: gate 5 fired with
    no number attached, and the cause had to be re-measured by a separate probe)."""
    rep: dict = {}
    try:
        return _preflight_body(a, rep)
    except BaseException as ex:                                      # noqa: BLE001
        rep["PASS"] = False
        rep["failure"] = f"{type(ex).__name__}: {ex}"
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=1, default=str)
            _p(f"[preflight] partial report written to {a.out} (PASS=False)")
        raise


def _preflight_body(a, rep: dict) -> int:
    t_all = time.time()
    device = a.device if torch.cuda.is_available() else "cpu"
    rep.update({"_what": "D-RL-REFCV3-MIN preflight — NOTHING trained, no checkpoint written",
''')
# close the dict literal that used to start with `rep: dict = {` : find its end
rep('''                 "imports": list(IMPORT_LOG), "device": device,
                 "torch": torch.__version__, "_evidence_class": "MEASURED (ours)"}
    model, cfg, targs, prov = load(a, device)
''', '''                 "imports": list(IMPORT_LOG), "device": device,
                 "torch": torch.__version__, "_evidence_class": "MEASURED (ours)"})
    model, cfg, targs, prov = load(a, device)
''')
open(D, "w", encoding="utf-8", newline="\n").write(s)
print("patch 4 applied")
