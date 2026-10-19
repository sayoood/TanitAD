| gate | poison (A, B, A2 + a global decoy) | deleted dir (A gone, B identical) | failed load leaves nothing |
|---|---|---|---|
| gate in mixed | RED | RED | RED |

- gate in mixed / poison (A, B, A2 + a global decoy): unexpectedly identical: <module 'srttc_r1__c2g_b7464f950bc0' from 'C:\\Users\\Admin\\AppData\\Local\\Temp\\c2g_sibling_bbehjb4e\\pa_A_2b0724edb79a\\srttc_r1.py'
- gate in mixed / deleted dir (A gone, B identical): True is not false : B was handed A's module object (its state is visible)
- gate in mixed / failed load leaves nothing: RuntimeError not raised
