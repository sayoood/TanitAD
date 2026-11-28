"""MM 2026-11-22: fix ra_env.sh's commit check (the hand-written case pattern had 38 hex classes, so no 40-hex commit could pass) and the comment the fill's global replace garbled."""
p = 'F:/TanitAD-scratch/rolladd/pod/ra_env.sh'
s = open(p, encoding='utf-8', newline='').read()
assert '\r' not in s
lines = s.split('\n')
i = [k for k, l in enumerate(lines) if l.startswith('case "$COMMIT" in ')]
assert len(i) == 1, i
assert lines[i[0]].count('[0-9a-f]') == 38
lines[i[0]] = ('[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ra_env.sh: COMMIT is \'$COMMIT\', not a filled 40-hex commit (fill_ra_commit.sh)"; exit 1; }'
               '      # MM 2026-11-22: was a case pattern of 38 hex classes, which refused every 40-hex commit')
j = [k for k, l in enumerate(lines) if 'is a placeholder: fill_ra_commit.sh' in l]
assert len(j) == 1, j
lines[j[0]] = '# COMMIT below was a placeholder until fill_ra_commit.sh (MM, after landing) wrote the 40-hex commit; every script that sources this file refuses unless it is 40 lowercase hex.'
open(p, 'w', encoding='utf-8', newline='').write('\n'.join(lines))
print('fixed')
