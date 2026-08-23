### Panel gate

admitted: `['ego_progress', 'recovery']`  ·  dropped: `['comfort']`

* `comfort` — INADMISSIBLE for cv_holdv0, v4_blind, v4_oracle -> dropped from EVERY arm so the composite is the same object on both sides of every paired delta

### Arm scores

| arm | n evals | n eps | goal provenance | `ego_progress` | `recovery` | **PSS_recovery_progress** |
|---|---:|---:|---|---|---|---|
| `cv_holdv0` | 15981 | 40 | none | 0.9407 [0.9169, 0.9610] | 0.0776 [0.0604, 0.0959] | **0.5705 [0.5558, 0.5844]** |
| `v4_oracle` | 15981 | 40 | ORACLE | 0.9462 [0.9320, 0.9591] | 0.0629 [0.0449, 0.0816] | **0.5622 [0.5496, 0.5725]** |
| `refc_xl_produced` | 15981 | 40 | none | 0.9438 [0.9279, 0.9585] | 0.0259 [0.0150, 0.0378] | **0.5499 [0.5421, 0.5566]** |
| `v1_tactical_follow` | 15981 | 40 | CONSTANT `follow` | 0.9081 [0.8871, 0.9279] | 0.0785 [0.0598, 0.0985] | **0.5471 [0.5340, 0.5595]** |
| `v1_tactical_oracle` | 15981 | 40 | ORACLE nav command | 0.9047 [0.8834, 0.9247] | 0.0817 [0.0621, 0.1024] | **0.5467 [0.5338, 0.5591]** |
| `refc_small_produced` | 15981 | 40 | none | 0.9315 [0.9120, 0.9485] | 0.0296 [0.0184, 0.0421] | **0.5444 [0.5360, 0.5514]** |
| `refc_base_produced` | 15981 | 40 | none | 0.9317 [0.9112, 0.9493] | 0.0293 [0.0175, 0.0422] | **0.5439 [0.5345, 0.5519]** |
| `nospeed_tactical_oracle` | 15981 | 40 | ORACLE nav command | 0.8961 [0.8727, 0.9177] | 0.0800 [0.0610, 0.1004] | **0.5394 [0.5242, 0.5540]** |
| `v4_blind` | 15981 | 40 | ORACLE | 0.5999 [0.4857, 0.7050] | 0.1159 [0.0913, 0.1426] | **0.3749 [0.3076, 0.4368]** |
| `stand_still` | 15981 | 40 | none | 0.0000 [0.0000, 0.0000] | — | **—** |

### Diagnostics

| arm | recovery defined frac | mean along-track end (m) | mean cross-track end (m) | wallclock s | planner calls | rollout steps |
|---|---:|---:|---:|---:|---:|---:|
| `cv_holdv0` | 0.8203 | 24.944 | 3.448 | 232.2 | 15981 | 0 |
| `v4_oracle` | 0.8377 | 24.859 | 3.531 | 3137.8 | 15981 | 0 |
| `refc_xl_produced` | 0.8256 | 24.957 | 4.264 | 3470.3 | 15981 | 0 |
| `v1_tactical_follow` | 0.8423 | 26.356 | 3.604 | 1973.1 | 15981 | 0 |
| `v1_tactical_oracle` | 0.8419 | 26.041 | 3.567 | 2705.5 | 15981 | 0 |
| `refc_small_produced` | 0.8253 | 24.775 | 4.257 | 1901.3 | 15981 | 0 |
| `refc_base_produced` | 0.8250 | 24.758 | 4.257 | 2606.5 | 15981 | 0 |
| `nospeed_tactical_oracle` | 0.8451 | 26.060 | 3.563 | 2853.7 | 15981 | 0 |
| `v4_blind` | 0.5748 | 21.092 | 3.192 | 3304.5 | 15981 | 0 |
| `stand_still` | 0.0000 | 0.000 | 1.007 | 262.5 | 15981 | 0 |

### Paired vs `v4_oracle` (episode-cluster bootstrap, B=2000, identical rows)

| arm | Δ `ego_progress` | Δ `recovery` | **Δ PSS** |
|---|---|---|---|
| `cv_holdv0` − `v4_oracle` | **-0.0055** [-0.0205, +0.0081] n.s. | **+0.0168** [+0.0008, +0.0332] ⭐ SEP | **+0.0034** [-0.0078, +0.0138] n.s. |
| `refc_xl_produced` − `v4_oracle` | **-0.0024** [-0.0114, +0.0061] n.s. | **-0.0360** [-0.0481, -0.0263] ⭐ SEP | **-0.0169** [-0.0238, -0.0105] ⭐ SEP |
| `v1_tactical_follow` − `v4_oracle` | **-0.0381** [-0.0625, -0.0142] ⭐ SEP | **+0.0152** [+0.0016, +0.0300] ⭐ SEP | **-0.0140** [-0.0268, -0.0019] ⭐ SEP |
| `v1_tactical_oracle` − `v4_oracle` | **-0.0415** [-0.0660, -0.0172] ⭐ SEP | **+0.0181** [+0.0048, +0.0331] ⭐ SEP | **-0.0147** [-0.0274, -0.0028] ⭐ SEP |
| `refc_small_produced` − `v4_oracle` | **-0.0147** [-0.0266, -0.0029] ⭐ SEP | **-0.0322** [-0.0450, -0.0214] ⭐ SEP | **-0.0216** [-0.0284, -0.0149] ⭐ SEP |
| `refc_base_produced` − `v4_oracle` | **-0.0145** [-0.0269, -0.0027] ⭐ SEP | **-0.0319** [-0.0443, -0.0216] ⭐ SEP | **-0.0217** [-0.0293, -0.0141] ⭐ SEP |
| `nospeed_tactical_oracle` − `v4_oracle` | **-0.0502** [-0.0771, -0.0243] ⭐ SEP | **+0.0164** [+0.0037, +0.0304] ⭐ SEP | **-0.0201** [-0.0345, -0.0068] ⭐ SEP |
| `v4_blind` − `v4_oracle` | **-0.3464** [-0.4593, -0.2437] ⭐ SEP | **+0.0564** [+0.0314, +0.0811] ⭐ SEP | **-0.1882** [-0.2557, -0.1240] ⭐ SEP |
| `stand_still` − `v4_oracle` | **-0.9462** [-0.9591, -0.9320] ⭐ SEP | — | — |

### Paired vs `cv_holdv0` (episode-cluster bootstrap, B=2000, identical rows)

| arm | Δ `ego_progress` | Δ `recovery` | **Δ PSS** |
|---|---|---|---|
| `v4_oracle` − `cv_holdv0` | **+0.0055** [-0.0081, +0.0205] n.s. | **-0.0168** [-0.0332, -0.0008] ⭐ SEP | **-0.0034** [-0.0138, +0.0078] n.s. |
| `refc_xl_produced` − `cv_holdv0` | **+0.0031** [-0.0058, +0.0131] n.s. | **-0.0528** [-0.0711, -0.0352] ⭐ SEP | **-0.0203** [-0.0303, -0.0097] ⭐ SEP |
| `v1_tactical_follow` − `cv_holdv0` | **-0.0326** [-0.0633, -0.0013] ⭐ SEP | **-0.0017** [-0.0156, +0.0127] n.s. | **-0.0178** [-0.0368, +0.0022] n.s. |
| `v1_tactical_oracle` − `cv_holdv0` | **-0.0360** [-0.0666, -0.0052] ⭐ SEP | **+0.0017** [-0.0132, +0.0167] n.s. | **-0.0182** [-0.0373, +0.0019] n.s. |
| `refc_small_produced` − `cv_holdv0` | **-0.0092** [-0.0177, +0.0003] n.s. | **-0.0497** [-0.0672, -0.0323] ⭐ SEP | **-0.0255** [-0.0358, -0.0146] ⭐ SEP |
| `refc_base_produced` − `cv_holdv0` | **-0.0090** [-0.0160, -0.0007] ⭐ SEP | **-0.0490** [-0.0674, -0.0307] ⭐ SEP | **-0.0252** [-0.0349, -0.0150] ⭐ SEP |
| `nospeed_tactical_oracle` − `cv_holdv0` | **-0.0447** [-0.0753, -0.0129] ⭐ SEP | **-0.0004** [-0.0134, +0.0138] n.s. | **-0.0235** [-0.0430, -0.0037] ⭐ SEP |
| `v4_blind` − `cv_holdv0` | **-0.3409** [-0.4501, -0.2420] ⭐ SEP | **+0.0403** [+0.0211, +0.0589] ⭐ SEP | **-0.1939** [-0.2595, -0.1332] ⭐ SEP |
| `stand_still` − `cv_holdv0` | **-0.9407** [-0.9610, -0.9169] ⭐ SEP | — | — |

### Paired vs `v1_tactical_oracle` (episode-cluster bootstrap, B=2000, identical rows)

| arm | Δ `ego_progress` | Δ `recovery` | **Δ PSS** |
|---|---|---|---|
| `cv_holdv0` − `v1_tactical_oracle` | **+0.0360** [+0.0052, +0.0666] ⭐ SEP | **-0.0017** [-0.0167, +0.0132] n.s. | **+0.0182** [-0.0019, +0.0373] n.s. |
| `v4_oracle` − `v1_tactical_oracle` | **+0.0415** [+0.0172, +0.0660] ⭐ SEP | **-0.0181** [-0.0331, -0.0048] ⭐ SEP | **+0.0147** [+0.0028, +0.0274] ⭐ SEP |
| `refc_xl_produced` − `v1_tactical_oracle` | **+0.0391** [+0.0144, +0.0652] ⭐ SEP | **-0.0543** [-0.0722, -0.0393] ⭐ SEP | **-0.0019** [-0.0157, +0.0120] n.s. |
| `v1_tactical_follow` − `v1_tactical_oracle` | **+0.0034** [+0.0008, +0.0069] ⭐ SEP | **-0.0032** [-0.0056, -0.0013] ⭐ SEP | **+0.0004** [-0.0008, +0.0018] n.s. |
| `refc_small_produced` − `v1_tactical_oracle` | **+0.0268** [+0.0005, +0.0545] ⭐ SEP | **-0.0515** [-0.0684, -0.0367] ⭐ SEP | **-0.0072** [-0.0210, +0.0075] n.s. |
| `refc_base_produced` − `v1_tactical_oracle` | **+0.0270** [+0.0002, +0.0550] ⭐ SEP | **-0.0504** [-0.0676, -0.0352] ⭐ SEP | **-0.0067** [-0.0208, +0.0078] n.s. |
| `nospeed_tactical_oracle` − `v1_tactical_oracle` | **-0.0086** [-0.0209, +0.0027] n.s. | **-0.0019** [-0.0093, +0.0038] n.s. | **-0.0055** [-0.0130, +0.0011] n.s. |
| `v4_blind` − `v1_tactical_oracle` | **-0.3048** [-0.4229, -0.1949] ⭐ SEP | **+0.0349** [+0.0153, +0.0569] ⭐ SEP | **-0.1744** [-0.2440, -0.1080] ⭐ SEP |
| `stand_still` − `v1_tactical_oracle` | **-0.9047** [-0.9247, -0.8834] ⭐ SEP | — | — |

### Pre-registered gates
```
{
  "G1_instrument_sensitivity": {
    "question": "does the protocol separate an arm that CAN see the perturbation from the IDENTICAL arm that cannot?",
    "v4_oracle_minus_v4_blind_PSS": {
      "delta": 0.1882,
      "lo": 0.124,
      "hi": 0.2557,
      "separated": true,
      "_sign_flipped": true
    },
    "published_reference": "+0.1882 [+0.1240, +0.2557] SEPARATED",
    "PASS": true,
    "if_fail": "NO ARM SCORE IS ADMISSIBLE; the panel reports the failure instead of a ranking."
  },
  "G2_port_fidelity": {
    "v4_oracle_minus_cv_holdv0_PSS": {
      "delta": -0.0034,
      "lo": -0.0138,
      "hi": 0.0078,
      "separated": false,
      "_sign_flipped": true
    },
    "v4_oracle_minus_cv_holdv0_recovery": {
      "delta": -0.0168,
      "lo": -0.0332,
      "hi": -0.0008,
      "separated": true
    },
    "published_reference_PSS": "-0.0034 [-0.0138, +0.0078] n.s.",
    "published_reference_recovery": "-0.0168 [-0.0332, -0.0008] SEPARATED",
    "PASS_pss_ns": true,
    "PASS_recovery_sep_neg": true
  },
  "G5_standing_still_adversary": {
    "recovery_defined_fraction": 0.0,
    "recovery_mean_where_defined": null,
    "frac_scored_1p0": null,
    "PASS": true,
    "rule": "a plan that does not move must be EXCLUDED (NaN), never scored 1.0. The pre-fix metric put this shape +0.597 ABOVE a sighted arm."
  },
  "D_DISCRIMINATIVE_POWER": {
    "definition": "D-NULL iff EVERY pairwise paired PSS contrast among the non-blind arms straddles zero => the instrument separates sighted from blind but CANNOT RANK OUR ARMS, and NO ranking is published.",
    "n_contrasts": 28,
    "n_separated": 12,
    "verdict": "D-PARTIAL",
    "contrasts": [
      {
        "pair": "cv_holdv0 - nospeed_tactical_oracle",
        "delta": 0.0235,
        "lo": 0.0037,
        "hi": 0.043,
        "separated": true,
        "_sign_flipped": false
      },
      {
        "pair": "cv_holdv0 - refc_base_produced",
        "delta": 0.0252,
        "lo": 0.015,
        "hi": 0.0349,
        "separated": true,
        "_sign_flipped": false
      },
      {
        "pair": "cv_holdv0 - refc_small_produced",
        "delta": 0.0255,
        "lo": 0.0146,
        "hi": 0.0358,
        "separated": true,
        "_sign_flipped": false
      },
      {
        "pair": "cv_holdv0 - refc_xl_produced",
        "delta": 0.0203,
        "lo": 0.0097,
        "hi": 0.0303,
        "separated": true,
        "_sign_flipped": false
      },
      {
        "pair": "cv_holdv0 - v1_tactical_follow",
        "delta": 0.0178,
        "lo": -0.0022,
        "hi": 0.0368,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "cv_holdv0 - v1_tactical_oracle",
        "delta": 0.0182,
        "lo": -0.0019,
        "hi": 0.0373,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "cv_holdv0 - v4_oracle",
        "delta": 0.0034,
        "lo": -0.0078,
        "hi": 0.0138,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "nospeed_tactical_oracle - refc_base_produced",
        "delta": 0.0015,
        "lo": -0.0142,
        "hi": 0.0163,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "nospeed_tactical_oracle - refc_small_produced",
        "delta": 0.0016,
        "lo": -0.0136,
        "hi": 0.0163,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "nospeed_tactical_oracle - refc_xl_produced",
        "delta": -0.0033,
        "lo": -0.0186,
        "hi": 0.0116,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "nospeed_tactical_oracle - v1_tactical_follow",
        "delta": -0.0058,
        "lo": -0.0128,
        "hi": 0.0008,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "nospeed_tactical_oracle - v1_tactical_oracle",
        "delta": -0.0055,
        "lo": -0.013,
        "hi": 0.0011,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "nospeed_tactical_oracle - v4_oracle",
        "delta": -0.0201,
        "lo": -0.0345,
        "hi": -0.0068,
        "separated": true,
        "_sign_flipped": false
      },
      {
        "pair": "refc_base_produced - refc_small_produced",
        "delta": 0.0002,
        "lo": -0.0024,
        "hi": 0.0025,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "refc_base_produced - refc_xl_produced",
        "delta": -0.005,
        "lo": -0.0082,
        "hi": -0.0023,
        "separated": true,
        "_sign_flipped": false
      },
      {
        "pair": "refc_base_produced - v1_tactical_follow",
        "delta": -0.0072,
        "lo": -0.0213,
        "hi": 0.0073,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "refc_base_produced - v1_tactical_oracle",
        "delta": -0.0067,
        "lo": -0.0208,
        "hi": 0.0078,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "refc_base_produced - v4_oracle",
        "delta": -0.0217,
        "lo": -0.0293,
        "hi": -0.0141,
        "separated": true,
        "_sign_flipped": false
      },
      {
        "pair": "refc_small_produced - refc_xl_produced",
        "delta": -0.0055,
        "lo": -0.0084,
        "hi": -0.0027,
        "separated": true,
        "_sign_flipped": false
      },
      {
        "pair": "refc_small_produced - v1_tactical_follow",
        "delta": -0.0077,
        "lo": -0.0215,
        "hi": 0.0071,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "refc_small_produced - v1_tactical_oracle",
        "delta": -0.0072,
        "lo": -0.021,
        "hi": 0.0075,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "refc_small_produced - v4_oracle",
        "delta": -0.0216,
        "lo": -0.0284,
        "hi": -0.0149,
        "separated": true,
        "_sign_flipped": false
      },
      {
        "pair": "refc_xl_produced - v1_tactical_follow",
        "delta": -0.0024,
        "lo": -0.0158,
        "hi": 0.0115,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "refc_xl_produced - v1_tactical_oracle",
        "delta": -0.0019,
        "lo": -0.0157,
        "hi": 0.012,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "refc_xl_produced - v4_oracle",
        "delta": -0.0169,
        "lo": -0.0238,
        "hi": -0.0105,
        "separated": true,
        "_sign_flipped": false
      },
      {
        "pair": "v1_tactical_follow - v1_tactical_oracle",
        "delta": 0.0004,
        "lo": -0.0008,
        "hi": 0.0018,
        "separated": false,
        "_sign_flipped": false
      },
      {
        "pair": "v1_tactical_follow - v4_oracle",
        "delta": -0.014,
        "lo": -0.0268,
        "hi": -0.0019,
        "separated": true,
        "_sign_flipped": false
      },
      {
        "pair": "v1_tactical_oracle - v4_oracle",
        "delta": -0.0147,
        "lo": -0.0274,
        "hi": -0.0028,
        "separated": true,
        "_sign_flipped": false
      }
    ]
  }
}
```
