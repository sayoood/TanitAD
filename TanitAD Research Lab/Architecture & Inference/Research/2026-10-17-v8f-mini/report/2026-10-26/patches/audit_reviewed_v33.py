# Hand review of the B, C and D tokens of audit_v33.py (exec'd by it). Key = the signed token as the audit prints it; value = the reviewer's reason.
# A token that is not here and is not in class A stops the audit. A key here that does not occur in B, C or D also stops it.
REVIEWED = {
    "7.24": "section reference (the status line of v33 and the 7.24 / 7.26 update sentences say 'section 7.24' for the SPEEDREAD read); structural, a heading number of this report, also present in v32",
    "7.25": "section reference (the status line names the DAGGER2 section); structural, a heading number of this report, also present in v32",
    "7.26": "section reference (the DAG-ORC section; the status line, the update sentences and the 7.29 prose point to it); structural, a heading number of this report, also present in v32",
    "8.1": "section reference (the established-claims list of the Discussion, named by a 7.29 sentence); structural, a heading number of this report, also present in v32",
    "7.27": "section reference (heading number of the SPEEDSUB section, named in the status line, the abstract, the 7.28 and 7.29 prose and the 8.4 rows); structural, found in no source because it is this report's own numbering",
    "7.28": "section reference (heading number of the SPEEDSUB-T1 section, named in the status line, the abstract, the 7.29 prose and the 8.4 rows); structural, found in no source because it is this report's own numbering",
    "7.29": "section reference (heading number of the LAUNCH-INTENT section, named in the title line, the status line, the abstract, the 7.26 update sentence and the 8.x text); structural, found in no source because it is this report's own numbering",
    "+28.7": "sign added in the prose range 'from -30.2 % to +28.7 %' (and 'between -3.0 % and +28.7 %'); the value 28.7 is the share closed in the s1 main curvature cell (RESULT table line 123, 28.7 %); the range and its ends are recomputed from the point deltas in section H",
    "+73.6": "sign added in the prose range 'from -23.4 % to +73.6 %' over the eight main-pair cells; 73.6 is the improvement / bar of the s1 main curvature cell (RESULT table lines 98 and 349, 73.6 %); the range is recomputed in section H",
    "+158.1": "sign added in the prose range 'from -130.9 % to +158.1 %' over all 16 cells; 158.1 is the improvement / bar of the s0 replicate L2 cell (RESULT table lines 95 and 348, 158.1 %); the range is recomputed in section H",
    "76": "derived count, 19 rows x 4 pair-seed cells (X - Y), named in the floors and the numbers prose; section H asserts 76 cells with 16 separated-better, 7 separated-worse and 53 not separated. The v32 occurrence is an unrelated 76",
    "29.3": "derived value, the twelve T1-kin step times 1756 s / 60 = 29.3 min; the sum and the quotient are recomputed in section H from the relaunch's chain log",
}
# Section-scoped exceptions: 'section:token' -> reason. A token of the prose of 7.27 / 7.28 / 7.29 that is not found in that section's own sources must be explained here.
_REF = "section reference to another section of this report (a heading number), structural; the target section exists in this report (checked by checks_v33.py)"
SCOPE_OK = {
    "7.27:7.27": "the section's own heading number; structural",
    "7.27:7.24": _REF,
    "7.27:7.28": _REF,
    "7.28:7.28": "the section's own heading number; structural",
    "7.28:7.27": _REF,
    "7.28:7.26": _REF,
    "7.28:7.29": _REF,
    "7.29:7.29": "the section's own heading number; structural",
    "7.29:7.26": _REF,
    "7.29:7.28": _REF,
    "7.29:8.1": _REF,
    "7.29:29.3": "derived: the twelve T1-kin step times sum to 1756 s, 1756 / 60 = 29.3 min; recomputed in section H from the relaunch's chain log (the sources print the 12 step times, not the quotient)",
    "7.29:76": "derived: 19 rows x 4 pair-seed cells; section H asserts the 76 cells and their 16 / 7 / 53 split from the read JSON",
    "7.29:+28.7": "the sign of the upper end of the prose range of the share of the Y - P10 gap closed; the unsigned value 28.7 is in the RESULT's table; the range is recomputed in section H",
    "7.29:+73.6": "the sign of the upper end of the prose range of improvement / bar over the eight main-pair cells; the unsigned value is in the RESULT's table; recomputed in section H",
    "7.29:+158.1": "the sign of the upper end of the prose range of improvement / bar over all 16 cells; the unsigned value is in the RESULT's table; recomputed in section H",
}
