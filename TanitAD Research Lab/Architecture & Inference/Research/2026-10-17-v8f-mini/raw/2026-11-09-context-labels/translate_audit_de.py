"""German display text for a built pilot audit page (PI request, 2026-11-08). Only what is SHOWN changes: the stored answer values stay the English class names, so scoring and export are unaffected.
Usage: python translate_audit_de.py <PILOT_AUDIT.html>   (rewrites the file in place; each replacement must match exactly once)."""
import sys

DE = ('const DE = {clear:"klar", rain:"Regen", snow:"Schnee", fog:"Nebel", dry:"trocken", wet:"nass", snow_ice:"Schnee/Eis", good:"gut", reduced:"eingeschränkt", poor:"schlecht", '
      'dark:"dunkel", lit_night:"Nacht, beleuchtet", twilight:"Dämmerung", day:"Tag", glare:"Blendung", night:"Nacht", dawn_dusk:"Morgen-/Abenddämmerung"};\n'
      'const de = (x) => DE[x] || x;\n')

R = [
    ('<html lang="en">', '<html lang="de">'),
    ('<title>Pilot audit (local, private)</title>', '<title>Pilot-Prüfung (lokal, privat)</title>'),
    ('const FIELD_TITLE = {weather:"Weather", surface:"Road surface", visibility:"Visibility", illumination:"Illumination", tod:"Rough time of day"};',
     'const FIELD_TITLE = {weather:"Wetter", surface:"Fahrbahnzustand", visibility:"Sichtweite", illumination:"Beleuchtung", tod:"Ungefähre Tageszeit"};'),
    ('const CANT = "cant_tell";\n', 'const CANT = "cant_tell";\n' + DE),
    ('<span>Frame ${i+1} / ${N}</span><span>${nDone(st)} done</span>', '<span>Bild ${i+1} / ${N}</span><span>${nDone(st)} fertig</span>'),
    ('alt="frame ${i+1}"', 'alt="Bild ${i+1}"'),
    ('${c===CANT?"can\'t tell":c.replace("_"," ")}', '${c===CANT?"nicht erkennbar":de(c)}'),
    ('<button class="nav">Previous</button>', '<button class="nav">Zurück</button>'),
    ('${i===N-1?"Last":"Next"}', '${i===N-1?"Letztes":"Weiter"}'),
    ('>Reveal results (${nDone(st)}/${N} done)</button>', '>Ergebnisse anzeigen (${nDone(st)}/${N} fertig)</button>'),
    ('confirm("Reveal shows the model labels. Answers are frozen afterwards. Continue?")', 'confirm("Jetzt werden die Modell-Labels angezeigt. Danach können die Antworten nicht mehr geändert werden. Fortfahren?")'),
    ('const ex=$(`<button>Export answers (JSON)</button>`); ex.onclick=doExport;\n  fin.appendChild', 'const ex=$(`<button>Antworten exportieren (JSON)</button>`); ex.onclick=doExport;\n  fin.appendChild'),
    ('<small>No model output is shown until all ${N} frames are answered and you press reveal. Progress is kept in this browser.</small>',
     '<small>Modell-Ergebnisse werden erst angezeigt, wenn alle ${N} Bilder beantwortet sind und Sie auf „Ergebnisse anzeigen“ klicken. Der Fortschritt wird in diesem Browser gespeichert.</small>'),
    ('<h2>Results: PI against each source</h2>', '<h2>Ergebnisse: Ihre Antworten im Vergleich zu jeder Quelle</h2>'),
    ('<small>Agreement excludes "can\'t tell". Sources: siglip = SigLIP2 zero-shot, qwen = Qwen2.5-VL-3B at 1 Hz, rule = image-only illumination rule (uncalibrated), sun = clock-based sun band estimate (cross-check only).</small>',
     '<small>Die Übereinstimmung zählt „nicht erkennbar“ nicht mit. Quellen: siglip = SigLIP2 (Zero-Shot), qwen = Qwen2.5-VL-3B mit 1 Hz, rule = Beleuchtungsregel nur aus dem Bild (unkalibriert), sun = Schätzung des Sonnenstands aus der Uhrzeit (nur zur Gegenprüfung).</small>'),
    ('<th>PI said</th><th>agreed</th>', '<th>Ihre Antwort</th><th>Übereinstimmung</th>'),
    ('<td><b>all</b></td><td>${r.n} <small>(can\'t tell: ${r.cant})</small></td>', '<td><b>gesamt</b></td><td>${r.n} <small>(nicht erkennbar: ${r.cant})</small></td>'),
    ('<td></td><td>${c}</td>', '<td></td><td>${de(c)}</td>'),
    ('<summary>${r.dis.length} disagreeing frames (${s})</summary>', '<summary>${r.dis.length} Bilder mit abweichender Antwort (${s})</summary>'),
    ('<span>#${x.i+1}: PI ${x.pi}, ${s} ${x.model}</span>', '<span>#${x.i+1}: Sie: ${de(x.pi)}, ${s}: ${de(x.model)}</span>'),
    ('const ex=$(`<button>Export answers (JSON)</button>`); ex.onclick=doExport; app.appendChild(ex);', 'const ex=$(`<button>Antworten exportieren (JSON)</button>`); ex.onclick=doExport; app.appendChild(ex);'),
    ('function pct(k,n){return n? (100*k/n).toFixed(0)+"% ("+k+"/"+n+")" : "n/a";}', 'function pct(k,n){return n? (100*k/n).toFixed(0)+" % ("+k+"/"+n+")" : "k. A.";}'),
]

p = sys.argv[1]
s = open(p, encoding="utf8", newline="").read().replace("\r\n", "\n")
for a, b in R:
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
open(p, "w", encoding="utf8", newline="").write(s)
print("translated", len(R), "strings")
