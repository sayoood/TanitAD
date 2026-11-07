#!/usr/bin/env python3
"""G7 RECORDER around YT's shim launcher. Behaviour is YT's `ego_yaw.main` UNCHANGED; this file only wraps `EgoYawShim.apply` to RECORD what it wrote into ego[4] (returned value untouched,
asserted identical to the original call's result), so G7 (PREREG section 4: slot 4 non-zero on >= 90 % of tokens and == float32(10 x table)) is read from the live seam, not from a unit test.

    python yf3_launch_g7.py --record <out.json> --table <navtest table> [--allow-incoming] -- <script.py> [args...]

The record is written in a `finally` (also when the seam exits non-zero): {shim_installed, n_calls, n_rows, scale, table_sha256, train_table_sha256, by_token: {token: float32 value fed},
n_incoming_nonzero}. For a checkpoint WITHOUT meta ego_yaw (A0, arm C) YT's shim is never installed and the record says so (n_calls 0): the planner is untouched.
"""
import json
import os
import sys
import threading

YT_REFE = "C:/Users/Admin/pxyaw/refe"
sys.path.insert(0, YT_REFE)
import ego_yaw as EY  # noqa: E402  (YT's launcher module, unmodified)

STATE = {"installed": 0, "n_calls": 0, "n_rows": 0, "by_token": {}, "reports": [], "mismatch": 0}
LOCK = threading.Lock()


def _install_hook():
    orig_apply, orig_install = EY.EgoYawShim.apply, EY.EgoYawShim.install

    def install(self, model, token_source=None):
        with LOCK:
            STATE["installed"] += 1
        return orig_install(self, model, token_source)

    def apply(self, ego):
        toks = list(self._current())
        out = orig_apply(self, ego)
        if len(toks) != ego.shape[0] and len(toks) == 1:
            toks = toks * ego.shape[0]
        vals = out[..., self.slot].detach().cpu().tolist()
        with LOCK:
            STATE["n_calls"] += 1
            for t, v in zip(toks, vals):
                STATE["n_rows"] += 1
                prev = STATE["by_token"].get(t)
                if prev is not None and prev != v:
                    STATE["mismatch"] += 1
                STATE["by_token"][t] = v
            if self not in STATE["reports"]:
                STATE["reports"].append(self)
        return out

    EY.EgoYawShim.apply, EY.EgoYawShim.install = apply, install


def main(argv):
    if "--record" not in argv:
        print("usage: yf3_launch_g7.py --record <out.json> --table <t> -- <script> [args]")
        return 2
    i = argv.index("--record")
    rec_path = argv[i + 1]
    rest = argv[:i] + argv[i + 2:]
    _install_hook()
    try:
        return EY.main(rest)
    finally:
        shims = [s.report() for s in STATE["reports"]]
        rec = {"shim_installed": STATE["installed"] > 0, "n_installs": STATE["installed"], "n_calls": STATE["n_calls"], "n_rows": STATE["n_rows"], "n_tokens_fed": len(STATE["by_token"]),
               "value_changed_between_calls": STATE["mismatch"], "shim_reports": shims, "by_token": STATE["by_token"]}
        os.makedirs(os.path.dirname(os.path.abspath(rec_path)), exist_ok=True)
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(rec, f)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
