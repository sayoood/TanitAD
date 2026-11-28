# READ_IP1 (card PREREG_V8F_PROOF_ADD44, INTEG-PRE)

Tier: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance

STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument)

Bases read: [0, 1, 2, 3]. Missing: {}

## Controls
* c1_heads: PASS
* c2_label_rule_cuts_U: PASS
* c3_identity_on_C_base0: PASS
* c4_counts: PASS
* c5_reader_controls: PASS
* c6_R0_equals_banked_IPo: PASS

## Names
```
{
 "cut": "INTEG-PRE-CUTS-UNSAFE",
 "reaches_label_rule": "INTEG-PRE-REACHES-LABEL-RULE",
 "levers": "INTEG-PRE-LEVERS-COMPATIBLE",
 "price": "INTEG-PRE-COSTS-LONGITUDINAL INTEG-PRE-COSTS-TACTICAL",
 "price_final_flag": true
}
```
## Differences (U, per base | pooled | two-level interval | same-sign count)
* D1_Cq_minus_Kr0: {"per_base": [-0.035587, -0.078292, -0.067616, -0.074733], "pooled": -0.064057, "interval": [-0.095381, -0.036091], "excludes_0": true, "n_same_sign_as_pooled": 4}
* D2_Cq_minus_Cr0: {"per_base": [-0.046263, -0.081851, -0.076512, -0.078292], "pooled": -0.07073, "interval": [-0.101398, -0.042677], "excludes_0": true, "n_same_sign_as_pooled": 4}
* D3_Cq_minus_Clabel: {"per_base": [0.021352, 0.014235, 0.012456, 0.019573], "pooled": 0.016904, "interval": [0.005989, 0.029045], "excludes_0": true, "n_same_sign_as_pooled": 4}
* D4_Cq_minus_Kq: {"per_base": [-0.001779, -0.003559, -0.005338, 0.019573], "pooled": 0.002224, "interval": [-0.00782, 0.015345], "excludes_0": false, "n_same_sign_as_pooled": 1}
* D5_Cr0_minus_Kr0: {"per_base": [0.010676, 0.003559, 0.008897, 0.003559], "pooled": 0.006673, "interval": [-0.002233, 0.015562], "excludes_0": false, "n_same_sign_as_pooled": 4}
* share_reached_per_base: {"values": [0.6842, 0.8519, 0.86, 0.8], "bases": ["s0", "s1", "s2", "s3"], "n_at_least_0.5": 4}
* contact_Cq_minus_Clabel: {"C": {"per_base": [0.021807, 0.015576, 0.015576, 0.007788], "pooled": 0.015187, "interval": [0.006309, 0.027161], "excludes_0": true, "n_same_sign_as_pooled": 4}}
## Families
```
{
 "C_RQ_against_K_R0": {
  "Q3": {
   "name_by_rule": "COST-LONGITUDINAL COST-TACTICAL",
   "final": true,
   "name": "COST-LONGITUDINAL COST-TACTICAL",
   "cost_families": [
    "LONGITUDINAL",
    "TACTICAL"
   ],
   "gain_families": []
  },
  "ade6_m": {
   "per_base": [
    0.570287,
    0.636388,
    0.563622,
    0.439796
   ],
   "pooled": 0.552523,
   "interval": [
    0.374784,
    0.743941
   ],
   "excludes_0": true,
   "n_same_sign_as_pooled": 4
  }
 },
 "C_RQ_against_C_R0": {
  "Q3": {
   "name_by_rule": "COST-LONGITUDINAL COST-TACTICAL",
   "final": true,
   "name": "COST-LONGITUDINAL COST-TACTICAL",
   "cost_families": [
    "LONGITUDINAL",
    "TACTICAL"
   ],
   "gain_families": []
  },
  "ade6_m": {
   "per_base": [
    0.561508,
    0.616202,
    0.546896,
    0.461338
   ],
   "pooled": 0.546486,
   "interval": [
    0.374024,
    0.735509
   ],
   "excludes_0": true,
   "n_same_sign_as_pooled": 4
  }
 }
}
```
