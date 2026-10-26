"""build_labels.py -- PREREG_LONHEAD_C s2: the oracle-arm label npz for the LHS and DER rolls = the FROZEN head's argmax (LON6 -> v7 id) on every
one of the 4,754 eval139 roll windows, from the L1 eval z_tac bank (SELCOR variant Z) + the bank's fed v0 (`v0_in`).  CPU only.  NEW file.
Format == refcv9_l2/selfsharp/labels_selfsharp_s0.npz (clip_index int64, ws int64, lon_v7 float32), key set asserted equal to it.
z_tac is bit-identical across inference seeds 0 / 1 (asserted here), so ONE label file serves both LHS rolls (as SELFSHARP's s1 roll did).
No clip id is printed or written (clip_index is the battery's integer episode index).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import k_head as K  # noqa: E402  (assert_registered, softmax, LF, load_bank)

SS_LABELS = Path("C:/Users/Admin/refcv8_pf/refcv9_l2/selfsharp/labels_selfsharp_s0.npz")
OUT = ROOT / "labels"


def main() -> int:
    hashes = K.assert_registered()
    kh = json.loads((ROOT / "result" / "K_HEAD.json").read_text(encoding="utf-8"))
    assert kh["go"] is True and kh["controls"]["ALL_PASS"] is True, "K-HEAD did not GO -- no labels, no rolls"
    b0 = K.load_bank(K.L1_BANK / "ztac_eval_s0.npz")
    b1 = K.load_bank(K.L1_BANK / "ztac_eval_s1.npz")
    assert len(b0["ws"]) == 4754 and b0["meta"].get("final") is True
    j1 = np.array([b1["idx"][(int(c), int(w))] for c, w in zip(b0["e_i"], b0["ws"])])
    z_eq = float(np.abs(b0["z"] - b1["z"][j1]).max())
    v_eq = float(np.abs(np.asarray(b0["v0_in"], np.float64) - np.asarray(b1["v0_in"], np.float64)[j1]).max())
    assert z_eq == 0.0 and v_eq == 0.0, (z_eq, v_eq)
    ss = np.load(SS_LABELS)
    ks_ss = set(zip(ss["clip_index"].tolist(), ss["ws"].tolist()))
    ks_b = list(zip(b0["e_i"].tolist(), b0["ws"].tolist()))
    assert ks_ss == set(ks_b) and len(ks_b) == 4754, "bank key set != SELFSHARP label key set"
    L6 = np.asarray(K.LF.LON6_V7, np.int64)
    native = None
    rec = {"z_s0_vs_s1_maxabs": z_eq, "v0_in_s0_vs_s1_maxabs": v_eq, "n": len(ks_b), "registered_sha256": hashes}
    OUT.mkdir(exist_ok=True)
    lines = []
    for tag, fn in (("LHS", "LH_FROZEN.pt"), ("DER", "DER_FROZEN.pt")):
        blob = K.LF.load_frozen(ROOT / fn, K.REG[fn])
        lg = K.LF.predict_logits6(blob, b0["z"], np.asarray(b0["v0_in"], np.float64))
        lab = L6[lg.argmax(1)]
        assert np.isin(lab, L6).all()
        p = OUT / f"labels_{tag}.npz"
        np.savez(p, clip_index=b0["e_i"].astype(np.int64), ws=b0["ws"].astype(np.int64), lon_v7=lab.astype(np.float32))
        z = np.load(p)
        assert np.array_equal(z["lon_v7"].astype(np.int64), lab) and z["lon_v7"].dtype == ss["lon_v7"].dtype
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h} *labels/{p.name}")
        # agreement with the SELFSHARP labels (the native head's LON6 argmax) on the same keys -- descriptive
        ssmap = {(int(c), int(w)): int(v) for c, w, v in zip(ss["clip_index"], ss["ws"], ss["lon_v7"])}
        nat = np.array([ssmap[k] for k in ks_b])
        u, c = np.unique(lab, return_counts=True)
        rec[tag] = {"file": str(p), "sha256": h, "class_counts_v7": {int(a): int(b) for a, b in zip(u, c)},
                    "agree_with_selfsharp_native_argmax": float((nat == lab).mean()), "n_changed_vs_selfsharp": int((nat != lab).sum())}
        print(f"[labels] {tag} {h[:12]} counts {rec[tag]['class_counts_v7']} agree-native {rec[tag]['agree_with_selfsharp_native_argmax']:.4f}", flush=True)
    (OUT / "labels.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "LABELS_INFO.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
