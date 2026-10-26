// Render the HiCAP A0 poster: hicap_poster.html -> hicap_poster.pdf (A0 portrait, vector),
// hicap_poster.png (~3000 px wide) and hicap_poster_preview.png (~1400 px wide).
// Usage: node Paper/HiCAP/poster/render_poster.js        (needs the `playwright` package + chromium)
// The script also runs a layout lint and prints every overflowing box, every text element below the
// legibility floor (18.67 px = 14 pt at A0 scale) and every element that leaves the page box.
// Exit code 2 if the lint finds a problem, so a broken layout is loud.
const { chromium } = require('playwright');
const path = require('path');

const DIR = __dirname;
const SRC = 'file://' + path.join(DIR, 'hicap_poster.html');
const PX_PER_MM = 96 / 25.4;
const W = Math.round(841 * PX_PER_MM);   // 3179 CSS px
const H = Math.round(1189 * PX_PER_MM);  // 4494 CSS px

async function lint(page) {
  return page.evaluate(() => {
    const out = { overflow: [], small: [], outside: [], overlap: [] };
    const pg = document.querySelector('.page').getBoundingClientRect();
    const name = (el) => (el.id ? '#' + el.id : '') + (el.className && typeof el.className === 'string' ? '.' + el.className.trim().split(/\s+/).join('.') : '') + ' "' + (el.textContent || '').trim().slice(0, 50).replace(/\s+/g, ' ') + '"';
    // 1. boxes whose content overflows (only for elements that clip or have fixed size)
    document.querySelectorAll('.node, .panel, .card, .fixed').forEach((el) => {
      if (el.scrollHeight > el.clientHeight + 2 || el.scrollWidth > el.clientWidth + 2)
        out.overflow.push(`${name(el)}  content ${el.scrollWidth}x${el.scrollHeight} > box ${el.clientWidth}x${el.clientHeight}`);
    });
    // 2. font-size floor for every element that directly holds text (HTML and SVG)
    const walker = document.createTreeWalker(document.querySelector('.page'), NodeFilter.SHOW_TEXT);
    const seen = new Set();
    while (walker.nextNode()) {
      const t = walker.currentNode; if (!t.textContent.trim()) continue;
      const el = t.parentElement; if (seen.has(el)) continue; seen.add(el);
      const cs = getComputedStyle(el); let fs = parseFloat(cs.fontSize);
      // SVG text inside a scaled <svg> renders at fontSize * scale
      const svg = el.closest('svg');
      if (svg && svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width) {
        const r = svg.getBoundingClientRect(); fs = fs * (r.width / svg.viewBox.baseVal.width);
      }
      if (fs < 18.6) out.small.push(`${fs.toFixed(1)}px ${name(el)}`);
    }
    // 3. anything outside the page box
    document.querySelectorAll('.page *').forEach((el) => {
      const r = el.getBoundingClientRect(); if (!r.width || !r.height) return;
      if (r.right > pg.right + 1 || r.bottom > pg.bottom + 1 || r.left < pg.left - 1 || r.top < pg.top - 1)
        out.outside.push(`${name(el)} [${Math.round(r.left)},${Math.round(r.top)},${Math.round(r.right)},${Math.round(r.bottom)}]`);
    });
    // 4. hero nodes that overlap each other
    const nodes = [...document.querySelectorAll('.hero .node')];
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i].getBoundingClientRect(), b = nodes[j].getBoundingClientRect();
      if (nodes[i].contains(nodes[j]) || nodes[j].contains(nodes[i])) continue;
      if (a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1)
        out.overlap.push(`${name(nodes[i])}  X  ${name(nodes[j])}`);
    }
    return out;
  });
}

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.goto(SRC, { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);

  const L = await lint(page);
  let bad = 0;
  for (const [k, v] of Object.entries(L)) {
    console.log(`lint ${k}: ${v.length}`);
    v.slice(0, 60).forEach((s) => console.log('   ' + s));
    bad += v.length;
  }

  await page.pdf({ path: path.join(DIR, 'hicap_poster.pdf'), width: '841mm', height: '1189mm',
    printBackground: true, preferCSSPageSize: true, margin: { top: 0, right: 0, bottom: 0, left: 0 } });

  for (const [file, px] of [['hicap_poster.png', 3000], ['hicap_poster_preview.png', 1400]]) {
    const c = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: px / W });
    const p = await c.newPage();
    await p.goto(SRC, { waitUntil: 'load' });
    await p.evaluate(() => document.fonts.ready);
    await p.locator('.page').screenshot({ path: path.join(DIR, file) });
    await c.close();
  }
  await browser.close();
  console.log(`rendered: hicap_poster.pdf (841x1189 mm), hicap_poster.png, hicap_poster_preview.png; lint problems: ${bad}`);
  process.exit(bad ? 2 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
