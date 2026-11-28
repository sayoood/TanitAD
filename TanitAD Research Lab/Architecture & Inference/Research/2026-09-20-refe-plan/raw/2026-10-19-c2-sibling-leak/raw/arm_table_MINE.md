| gate | poison (A, B, A2 + a global decoy) | deleted dir (A gone, B identical) | failed load leaves nothing |
|---|---|---|---|
| gate in mine | GREEN | GREEN | GREEN |
| sha-keyed sys.modules lookup (old loaders) | RED | RED | GREEN |
| sibling import via sys.modules / sys.path (old global binding) | RED | RED | GREEN |
| failed load stays registered (no poison guard) | GREEN | GREEN | RED |

- sha-keyed sys.modules lookup (old loaders) / poison (A, B, A2 + a global decoy): unexpectedly identical: <module 'srttc_r1__c2g_73df054d5d71' from 'C:\\Users\\Admin\\AppData\\Local\\Temp\\c2g_sibling_utwdjppz\\pa_A_2064a9340f55\\srttc_r1.py'
- sha-keyed sys.modules lookup (old loaders) / deleted dir (A gone, B identical): True is not false : B was handed A's module object (its state is visible)
- sibling import via sys.modules / sys.path (old global binding) / poison (A, B, A2 + a global decoy): <module 'srttc_r1'> is not <module 'srttc_r1__c2g_7f471c2621a5_n3' from 'C:\\Users\\Admin\\AppData\\Local\\Temp\\c2g_sibling_utwdjppz\\pa_A_e40277f69449\\srttc_
- sibling import via sys.modules / sys.path (old global binding) / deleted dir (A gone, B identical): 'b841347d90c298ec597f10afbbd284d68258df55c98cbac8cbd9a0a444cc5d0b' != 'f6a94984d72c1892ca979d3364f2ca625bc4a326d7f4dc4191ca96a0a0198e7a'
- failed load stays registered (no poison guard) / failed load leaves nothing: Items in the first set but not the second:
