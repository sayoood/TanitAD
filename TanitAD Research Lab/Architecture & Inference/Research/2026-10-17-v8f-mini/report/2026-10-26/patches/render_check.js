// Offline render check for the report page: the page template's own pre-processing (code, then math) and a local marked, no network.
// node render_check.js <md or html file> [path to a marked install]
// For an .html argument the md-src block is extracted first. Prints table / h3 / math counts, ragged tables, rendered slots, stray placeholders.
const fs = require("fs");
const path = require("path");
const markedPath = process.argv[3] || "C:/Users/Admin/AppData/Local/hermes/hermes-agent/node_modules/marked";
const marked = require(markedPath);
function render(file) {
  let src = fs.readFileSync(file, "utf8");
  if (file.endsWith(".html")) {
    const a = src.indexOf('<script type="text/plain" id="md-src">');
    const b = src.indexOf("</script>", a);
    src = src.slice(a + '<script type="text/plain" id="md-src">'.length, b);
  }
  const code = [];
  src = src.replace(/```[\s\S]*?```/g, m => { code.push(m); return "@@CODE" + (code.length - 1) + "@@"; });
  src = src.replace(/`[^`\n]+`/g, m => { code.push(m); return "@@CODE" + (code.length - 1) + "@@"; });
  const math = [];
  src = src.replace(/\$\$([\s\S]+?)\$\$/g, (m, b) => { math.push("$$" + b + "$$"); return "@@MATH" + (math.length - 1) + "@@"; });
  src = src.replace(/(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$/g, (m, pre, b) => { math.push("\\(" + b + "\\)"); return pre + "@@MATH" + (math.length - 1) + "@@"; });
  src = src.replace(/@@CODE(\d+)@@/g, (m, i) => code[+i]);
  let html = marked.parse(src, { gfm: true });
  html = html.replace(/@@MATH(\d+)@@/g, (m, i) => math[+i]);  // the template restores these after the parse
  return { html, math: math.length };
}
const p = process.argv[2];
const { html, math } = render(p);
const tables = (html.match(/<table/g) || []).length;
const h3 = (html.match(/<h3/g) || []).length;
let bad = 0;
for (const t of html.split("<table").slice(1)) {
  const rows = t.split("</table>")[0].split("<tr").slice(1);
  const counts = rows.map(r => (r.match(/<t[dh][ >]/g) || []).length);
  if (new Set(counts).size !== 1) bad++;
}
const pre = (html.match(/<pre/g) || []).length;
const slots = (html.match(/@@DAGORC_RESULT@@/g) || []).length;
const stray = (html.match(/@@(CODE|MATH)\d+@@/g) || []).length;
console.log(JSON.stringify({ file: path.basename(p), tables, h3, math, tables_with_ragged_rows: bad, pre_blocks: pre, slots_rendered: slots, stray_placeholders: stray }));
