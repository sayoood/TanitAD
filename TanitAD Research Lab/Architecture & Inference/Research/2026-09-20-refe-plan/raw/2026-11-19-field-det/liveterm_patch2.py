H = 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_p7/det/land/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe'
p = H + '/test_field_p7_profile.py'; s = open(p, 'rb').read().decode('utf-8')
o1 = ("# 2026-11-19 (deterministic field-loss lookup): the ONLY change to either profile dict is its field_loss_sha256 value (e29d91c3 -> da7b4303, one occurrence each); substituting the old value back\n"
      "# reproduces the e9ecd885 hashes 68f73903 (F) and 258b53f9 (C) exactly, so these two are those hashes re-computed with the new loss pin.\n"
      "E9_PROFILE_SHA256 = {\"F\": 'e795f43cd511751100bcd587df31f732ff826a4c30b4a5c7d2fcd1830b554c22', \"C\": 'f4b05b5c91367f5c80e607f315a59db174174100600eca34691323cfeb6a634f'}\n")
n1 = ("# 2026-11-19, two pod-measured fixes, and NOTHING else in either profile dict: (1) the deterministic field-loss lookup moves field_loss_sha256 (e29d91c3 -> da7b4303) in\n"
      "# both; (2) C no longer declares field_L as a live term (C's pod gate FAILED G-LIVE on it: the trainer logs field_L only when --w-field > 0). chk_unchanged undoes\n"
      "# exactly those two and must reproduce the e9ecd885 hashes (E9_ORIG_SHA256) bit for bit, so no other key can have moved.\n"
      "E9_PROFILE_SHA256 = {\"F\": 'e795f43cd511751100bcd587df31f732ff826a4c30b4a5c7d2fcd1830b554c22', \"C\": '2944fef660e56ef04266ce09a0df37da2405d345471ba1f0125c2a936019b1df'}\n"
      "E9_ORIG_SHA256 = {\"F\": '68f739037368a42867cd420c002dd4b85572c170bf2f46ea3ca6089996d12013', \"C\": '258b53f9c1602f788f7c3c466c476079d107e960c9a0e45dd4955a7bc7492088'}\n"
      "E9_LOSS_SHA256 = 'e29d91c349539ba366c73739fd2fe7fc6f1e7a6a015c4c0749df5f15df36c994'\n")
o2 = ("        ok(hashlib.sha256(json.dumps(L.PROFILES[nm], sort_keys=True, default=str).encode()).hexdigest() == E9_PROFILE_SHA256[arm], (nm, \"profile moved since e9ecd885\"))\n")
n2 = o2 + ("        back = dict(L.PROFILES[nm], field_loss_sha256=E9_LOSS_SHA256)\n"
           "        if arm == \"C\":\n"
           "            back[\"live_terms\"] = tuple(back[\"live_terms\"]) + (\"field_L\",)\n"
           "        ok(hashlib.sha256(json.dumps(back, sort_keys=True, default=str).encode()).hexdigest() == E9_ORIG_SHA256[arm], (nm, \"undoing the two 2026-11-19 fixes must give the e9ecd885 profile exactly\"))\n")
for o, n in ((o1, n1), (o2, n2)):
    assert s.count(o) == 1, o[:80]
    s = s.replace(o, n)
open(p, 'wb').write(s.encode('utf-8')); print('edited test_field_p7_profile.py')
