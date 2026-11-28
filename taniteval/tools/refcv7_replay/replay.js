/* refcv7 replay tool -- static, opens from disk (file://), no server, no fetch.
 *
 * Data arrives as <script src> files: data/manifest.js and data/clip_NN.js (one per clip), plus <img> JPEGs.
 * Every overlay is drawn client-side from METRIC coordinates; the camera overlays go through a JavaScript port of
 * the model's own RigCamera cylindrical projection, checked against Python values by the "self-test" button.
 *
 * ONE COLOUR RULE EVERYWHERE:  GT = GREEN, dashed or outline-only, tagged "GT".
 *                              MODEL = ORANGE (the emitted plan) / BLUE (candidates, decoder pick, detections), solid / filled.
 */
(function () {
'use strict';
const RV7 = window.RV7 = window.RV7 || {};
const M = RV7.manifest;
RV7.clips = RV7.clips || {};
const $ = (id) => document.getElementById(id);
const clamp = (x, a, b) => Math.min(Math.max(x, a), b);

// ------------------------------------------------------------------------------------------------ constants
const GT_C = [110, 231, 138], PLAN_C = [249, 115, 22], DEC_C = [59, 130, 246], BOX_C = [56, 132, 255], GOAL_C = [232, 121, 249];
const FAN_HI = [165, 225, 255], FAN_LO = [88, 70, 214];
const rgb = (c, a) => a === undefined ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const lerpC = (a, b, t) => [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * clamp(t, 0, 1)));
// 8 lat / 8 lon class colours: distinct, NO green (GT) and NO orange (plan)
const LAT_COL = ['#64748b', '#38bdf8', '#818cf8', '#a78bfa', '#22d3ee', '#c084fc', '#f472b6', '#fb7185'];
const LON_COL = ['#94a3b8', '#64748b', '#a78bfa', '#f43f5e', '#38bdf8', '#475569', '#e879f9', '#facc15'];
const GRID_X = 1000, GRID_Y = 600, BEV_W = 480, BEV_H = 800, X_MAX = 100, Y_HALF = 30;
const SX = BEV_W / (2 * Y_HALF), SY = BEV_H / X_MAX;       // 8 px / m
const m2p = (x, y) => [(Y_HALF - y) * SX, (X_MAX - x) * SY];  // forward is up, LEFT is left
const HZ = [5, 10, 15, 20, 30, 40, 50, 60];
const CUBOID_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
const CEIL_STAMP = M.ceil_stamp;

// ------------------------------------------------------------------------------------------------ state
const S = {
  rank: null, cd: null, i: 0, playing: false, lastTs: 0, acc: 0,
  L: {
    view: 'both',
    gt_path: true, gt_boxes: true, gt_other: false, gt_goals: true, gt_map: true,
    plan: true, dec: true, fan: true, fan_k: 3, fan_src: 'e9', fan_col: 'rank', f_reach: false, f_ceil: false, f_nav: false, fan_lbl: true,
    boxes: true, thr: M.thr_box, nms: true, nms_r: 2.0, goals: true, pred_map: true,
    map_alpha: 0.7, layout: 'side', dim: true, cls: M.mapcls.map(() => true),
  },
};
const imgCache = new Map();      // 'rank:i' -> HTMLImageElement
const mapCache = new Map();      // 'rank:i' -> {pred, gt} | 'pending'
let rafQueued = false;

// ------------------------------------------------------------------------------------------------ camera model (port of RigCamera.project, cylindrical)
function Cam(c) {
  this.W = c.W; this.H = c.H; this.f = c.f; this.R = c.R; this.t = c.t;
  this.cx = (c.W - 1) / 2; this.cy = (c.H - 1) / 2;
}
Cam.prototype.project = function (X, Y, Z) {                // rig point -> [col, row, ok]
  const R = this.R, dx = X - this.t[0], dy = Y - this.t[1], dz = Z - this.t[2];
  const xc = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz;     // p_cam = R^T (p - t)
  const yc = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz;
  const zc = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz;
  const rho = Math.sqrt(xc * xc + zc * zc), phi = Math.atan2(xc, zc);
  const col = this.cx + this.f * phi, row = this.cy + this.f * yc / Math.max(rho, 1e-9);
  const ok = rho > 1e-9 && col >= -1e-6 && col <= this.W - 1 + 1e-6 && row >= -1e-6 && row <= this.H - 1 + 1e-6;
  return [col, row, ok, zc, rho];
};
// ground point exactly as render_refcv3_video.CylProjector drops them: behind the plane or < 0.5 m radially -> null
Cam.prototype.ground = function (x, y) {
  const p = this.project(x, y, 0);
  if (p[3] <= 0 || p[4] < 0.5 || !p[2]) return null;
  return [p[0], p[1]];
};
function densify(path, n) {                                    // arc-length linear interpolation, origin prepended
  const p = [[0, 0]].concat(path);
  const s = [0];
  for (let i = 1; i < p.length; i++) s.push(s[i - 1] + Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]));
  const tot = s[s.length - 1];
  if (tot <= 1e-6) return p;
  const out = [];
  let j = 0;
  for (let k = 0; k < n; k++) {
    const q = tot * k / (n - 1);
    while (j < s.length - 2 && s[j + 1] < q) j++;
    const seg = s[j + 1] - s[j], u = seg > 0 ? (q - s[j]) / seg : 0;
    out.push([p[j][0] + (p[j + 1][0] - p[j][0]) * u, p[j][1] + (p[j + 1][1] - p[j][1]) * u]);
  }
  return out;
}
function cuboidCorners(cx, cy, cz, l, w, h, yaw) {
  const c = Math.cos(yaw), s = Math.sin(yaw), hl = l / 2, hw = w / 2;
  const loc = [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]];
  const fp = loc.map(([a, b]) => [c * a - s * b + cx, s * a + c * b + cy]);
  const z0 = cz - h / 2, z1 = cz + h / 2;
  return fp.map((p) => [p[0], p[1], z0]).concat(fp.map((p) => [p[0], p[1], z1]));
}
function footprint(b, yaw) {
  const c = Math.cos(yaw), s = Math.sin(yaw), hl = b[2] / 2, hw = b[3] / 2;
  return [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]].map(([a, d]) => [c * a - s * d + b[0], s * a + c * d + b[1]]);
}

// ------------------------------------------------------------------------------------------------ data helpers
function b64Int16(b64) {
  const bin = atob(b64), n = bin.length >> 1, out = new Float32Array(n);
  const dv = new DataView(new ArrayBuffer(bin.length));
  for (let i = 0; i < bin.length; i++) dv.setUint8(i, bin.charCodeAt(i));
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(2 * i, true) / 100;
  return out;
}
function fanOf(w) {                                           // Float32Array [N*8*2] metres
  if (!w._fan) w._fan = b64Int16(w.fan);
  return w._fan;
}
function candPath(w, k) {
  const f = fanOf(w), out = [];
  for (let s = 0; s < 8; s++) out.push([f[(k * 8 + s) * 2], f[(k * 8 + s) * 2 + 1]]);
  return out;
}
function fanList(w) {
  const N = w.e9.length, L = S.L, src = L.fan_src === 'dec' ? w.dc : w.e9;
  const idx = [];
  // the decoder's own empty-survivor rule: when NO candidate satisfies the ceiling the whole fan is kept
  let useCeil = L.f_ceil;
  S.ceilFallback = false;
  if (useCeil) {
    let any = false;
    for (let k = 0; k < N && !any; k++) if ((!L.f_reach || w.rch[k] === '1') && w.ceil[k] === '1') any = true;
    if (!any) { useCeil = false; S.ceilFallback = true; }
  }
  for (let k = 0; k < N; k++) {
    if (L.f_reach && w.rch[k] !== '1') continue;
    if (useCeil && w.ceil[k] !== '1') continue;
    if (L.f_nav && w.nci && w.navc[k] !== '1') continue;
    idx.push(k);
  }
  idx.sort((a, b) => src[b] - src[a]);
  return idx.slice(0, L.fan_k);
}
function fanColour(rank, k, score, smin, smax) {
  let t;
  if (S.L.fan_col === 'score') t = smax > smin ? 1 - (score - smin) / (smax - smin) : 0;
  else t = k > 1 ? rank / (k - 1) : 0;
  return { c: lerpC(FAN_HI, FAN_LO, t), a: 0.95 - 0.55 * t };
}
function greedyMatch(dxy, dp, gxy, gpos, gign, dist) {        // box3d_match_rows, numpy-identical
  const gi = [];
  for (let j = 0; j < gxy.length; j++) if (gpos[j] || gign[j]) gi.push(j);
  const taken = gi.map(() => false), order = dp.map((p, i) => i).sort((a, b) => dp[b] - dp[a] || a - b), rows = [];
  for (const i of order) {
    if (!gi.length) { rows.push([i, -1, 0]); continue; }
    let bj = -1, bd = Infinity;
    for (let q = 0; q < gi.length; q++) {
      if (taken[q]) continue;
      const d = Math.hypot(gxy[gi[q]][0] - dxy[i][0], gxy[gi[q]][1] - dxy[i][1]);
      if (d < bd) { bd = d; bj = q; }
    }
    if (bj >= 0 && bd <= dist) { taken[bj] = true; const g = gi[bj]; rows.push([i, g, gpos[g] ? 1 : -1]); }
    else rows.push([i, -1, 0]);
  }
  return rows;
}
function bevNms(dxy, dp, r) {
  const order = dp.map((p, i) => i).sort((a, b) => dp[b] - dp[a] || a - b), kept = [];
  for (const i of order) {
    let ok = true;
    for (const k of kept) if (Math.hypot(dxy[i][0] - dxy[k][0], dxy[i][1] - dxy[k][1]) <= r) { ok = false; break; }
    if (ok) kept.push(i);
  }
  return kept;
}
function boxState(w) {                                         // the model boxes that are SHOWN, and their match to GT
  const L = S.L, bx = w.bx, n = bx.p.length;
  let idx = [];
  for (let j = 0; j < n; j++) if (bx.p[j] >= L.thr - 1e-12) idx.push(j);
  if (L.nms && idx.length) {
    const kept = bevNms(idx.map((j) => bx.b[j]), idx.map((j) => bx.p[j]), L.nms_r);
    idx = kept.map((k) => idx[k]);
  }
  const res = { idx, tp: 0, fp: 0, dc: 0, link: {}, label: w.dt.lab, npos: w.dt.np, kind: {} };
  if (w.dt.lab && idx.length) {
    const rows = greedyMatch(idx.map((j) => bx.b[j]), idx.map((j) => bx.p[j]), w.gb.b, w.gb.pos, w.gb.ign, M.match_dist);
    for (const [i, g, k] of rows) {
      const j = idx[i];
      res.kind[j] = k;
      if (k === 1) { res.tp++; res.link[j] = g; } else if (k === 0) res.fp++; else res.dc++;
    }
  }
  return res;
}

// ------------------------------------------------------------------------------------------------ images
function camKey(rank, i) { return rank + ':' + i; }
function camImage(rank, i) {
  const key = camKey(rank, i);
  let im = imgCache.get(key);
  if (!im) {
    im = new Image();
    im.onload = () => { if (rank === S.rank && i === S.i) queueRender(); };
    im.src = M.clips[rank - 1].cam_dir + 'w' + String(i).padStart(4, '0') + '.jpg';
    imgCache.set(key, im);
    if (imgCache.size > 80) imgCache.delete(imgCache.keys().next().value);
  }
  return im;
}
const decCanvas = document.createElement('canvas'); decCanvas.width = GRID_Y; decCanvas.height = GRID_X;
const decCtx = decCanvas.getContext('2d', { willReadFrequently: true });
async function decodePng(b64) {
  const im = new Image();
  im.src = 'data:image/png;base64,' + b64;
  await im.decode();
  decCtx.clearRect(0, 0, GRID_Y, GRID_X);
  decCtx.drawImage(im, 0, 0);
  const d = decCtx.getImageData(0, 0, GRID_Y, GRID_X).data, out = new Uint8Array(GRID_X * GRID_Y);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4];
  return out;
}
function getMaps(rank, i) {
  const key = camKey(rank, i), c = mapCache.get(key);
  if (c && c !== 'pending') return c;
  if (c === 'pending') return null;
  const cd = RV7.clips[rank];
  if (!cd) return null;
  mapCache.set(key, 'pending');
  Promise.all([decodePng(cd.mp[i][0]), decodePng(cd.mp[i][1])]).then(([p, g]) => {
    mapCache.set(key, { pred: p, gt: g });
    if (mapCache.size > 70) { for (const k of mapCache.keys()) { if (k !== key && mapCache.get(k) !== 'pending') { mapCache.delete(k); break; } } }
    if (rank === S.rank && i === S.i) queueRender();
  });
  return null;
}
function prefetch() {
  const cd = S.cd; if (!cd) return;
  for (let d = 1; d <= 8; d++) {
    const j = S.i + d; if (j >= cd.w.length) break;
    camImage(S.rank, j);
    if (d <= 3) getMaps(S.rank, j);
  }
}

// ------------------------------------------------------------------------------------------------ rendering: BEV maps
const offA = document.createElement('canvas'), offB = document.createElement('canvas');
[offA, offB].forEach((c) => { c.width = GRID_Y; c.height = GRID_X; });
const imgA = offA.getContext('2d').createImageData(GRID_Y, GRID_X), imgB = offB.getContext('2d').createImageData(GRID_Y, GRID_X);
function composeMap(arr, kind, out, outlineOnly) {
  const d = out.data, en = S.L.cls, al = S.L.map_alpha, dim = S.L.dim, PAL = M.mapcol;
  for (let r = 0; r < GRID_X; r++) {
    const gr = GRID_X - 1 - r;
    for (let c = 0; c < GRID_Y; c++) {
      const gc = GRID_Y - 1 - c, v = arr[gr * GRID_Y + gc], cls = v & 7, ns = (v >> 3) & 1, li = (v >> 4) & 1, o = (r * GRID_Y + c) * 4;
      let vis, a;
      if (kind === 'pred') { vis = en[cls] && !li; a = al * ((ns && dim) ? 0.38 : 1); }
      else { vis = en[cls] && !ns; a = al * ((li && dim) ? 0.38 : 1); }
      if (vis && outlineOnly) {                                  // GT as an outline: only boundary cells
        let b = false;
        const nb = [gr > 0 ? arr[(gr - 1) * GRID_Y + gc] : v, gr < GRID_X - 1 ? arr[(gr + 1) * GRID_Y + gc] : v,
                    gc > 0 ? arr[gr * GRID_Y + gc - 1] : v, gc < GRID_Y - 1 ? arr[gr * GRID_Y + gc + 1] : v];
        for (let q = 0; q < 4; q++) if (((nb[q] & 7) !== cls) || (((nb[q] >> 3) & 1) !== ns)) { b = true; break; }
        if (!b) vis = false;
        a = 0.95;
      }
      if (!vis) { d[o + 3] = 0; continue; }
      const col = outlineOnly ? GT_C : PAL[cls];
      d[o] = col[0]; d[o + 1] = col[1]; d[o + 2] = col[2]; d[o + 3] = Math.round(a * 255);
    }
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ rendering: shared overlay helpers
function dashed(ctx, pts, dash, gap) { ctx.setLineDash([dash, gap]); strokePath(ctx, pts); ctx.setLineDash([]); }
function strokePath(ctx, pts) {
  ctx.beginPath(); let started = false;
  for (const p of pts) { if (!p) { started = false; continue; } if (!started) { ctx.moveTo(p[0], p[1]); started = true; } else ctx.lineTo(p[0], p[1]); }
  ctx.stroke();
}
function chip(ctx, text, x, y, fg, bg, font) {
  ctx.font = font || 'bold 12px system-ui,sans-serif';
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = bg; ctx.fillRect(x - 3, y - 11, tw + 6, 15);
  ctx.fillStyle = fg; ctx.fillText(text, x, y);
  return tw + 6;
}
function showFam() { const v = S.L.view; return { gt: v !== 'model', model: v !== 'gt' }; }
function planStamp(w) {
  const ex = w.pex;
  return (ex ? 'plan peak ' + w.pvm.toFixed(1) + ' m/s > ceiling ' + (w.vl === null ? '–' : w.vl.toFixed(1)) + ' m/s · ' : '') + CEIL_STAMP;
}

// ------------------------------------------------------------------------------------------------ rendering: BEV
function drawBEV(ctx, w, panel) {
  const L = S.L, fam = showFam(), maps = getMaps(S.rank, S.i);
  ctx.fillStyle = '#06090d'; ctx.fillRect(0, 0, BEV_W, BEV_H);
  const overlay = L.layout === 'overlay';
  ctx.imageSmoothingEnabled = true;
  if (maps) {
    if (panel === 'A') {
      if (L.pred_map && (fam.model)) { composeMap(maps.pred, 'pred', imgA, false); offA.getContext('2d').putImageData(imgA, 0, 0); ctx.drawImage(offA, 0, 0, BEV_W, BEV_H); }
      if (overlay && L.gt_map && fam.gt) { composeMap(maps.gt, 'gt', imgB, true); offB.getContext('2d').putImageData(imgB, 0, 0); ctx.drawImage(offB, 0, 0, BEV_W, BEV_H); }
    } else if (L.gt_map && fam.gt) {
      composeMap(maps.gt, 'gt', imgB, false); offB.getContext('2d').putImageData(imgB, 0, 0); ctx.drawImage(offB, 0, 0, BEV_W, BEV_H);
    }
  } else { ctx.fillStyle = '#556'; ctx.font = '12px system-ui'; ctx.fillText('decoding map…', 10, 20); }
  // range rings + guides
  const [ex, ey] = m2p(0, 0);
  ctx.lineWidth = 1;
  for (let r = 10; r <= 100; r += 10) {
    ctx.strokeStyle = r % 50 ? 'rgba(255,255,255,0.16)' : 'rgba(255,255,255,0.30)';
    ctx.beginPath(); ctx.ellipse(ex, ey, r * SX, r * SY, 0, Math.PI, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = 'rgba(210,218,230,0.8)'; ctx.font = '13px system-ui';
    if (r % 20 === 0) ctx.fillText(r + ' m', BEV_W - 34, Math.max(m2p(r, 0)[1] - 2, 10));
  }
  for (const yl of [-20, -10, 0, 10, 20]) { const [px] = m2p(0, yl); ctx.strokeStyle = yl ? 'rgba(255,255,255,0.09)' : 'rgba(255,255,255,0.2)'; ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, BEV_H); ctx.stroke(); }
  drawBevOverlays(ctx, w, fam);
  // ego
  ctx.fillStyle = '#f1f5f9'; ctx.beginPath(); ctx.moveTo(ex - 7, ey); ctx.lineTo(ex + 7, ey); ctx.lineTo(ex, ey - 15); ctx.closePath(); ctx.fill();
  if ((L.plan || L.dec || L.fan) && fam.model) {
    ctx.font = '13px system-ui'; const t = planStamp(w), tw = ctx.measureText(t).width;
    ctx.fillStyle = 'rgba(9,12,17,0.82)'; ctx.fillRect(2, BEV_H - 19, tw + 8, 17); ctx.fillStyle = w.pex ? '#fca5a5' : '#f5b45a'; ctx.fillText(t, 6, BEV_H - 6);
  }
}
function drawBevOverlays(ctx, w, fam) {
  const L = S.L;
  const poly = (pts) => pts.map(([x, y]) => m2p(x, y));
  // --- MODEL fan (under everything)
  if (L.fan && fam.model) {
    const list = fanList(w), src = L.fan_src === 'dec' ? w.dc : w.e9, sc = list.map((k) => src[k]);
    const smin = Math.min(...sc), smax = Math.max(...sc);
    for (let r = list.length - 1; r >= 0; r--) {
      const k = list[r], col = fanColour(r, list.length, src[k], smin, smax);
      ctx.strokeStyle = rgb(col.c, col.a); ctx.lineWidth = r === 0 ? 3 : 2.2;
      strokePath(ctx, poly([[0, 0]].concat(candPath(w, k))));
      if (L.fan_lbl && list.length <= 8) {
        const e = m2p(...candPath(w, k)[7]); ctx.fillStyle = rgb(col.c, 1); ctx.font = 'bold 13px system-ui'; ctx.fillText('#' + (r + 1), clamp(e[0] + 3, 2, BEV_W - 20), clamp(e[1] - 3, 10, BEV_H - 4));
      }
    }
  }
  // --- GT boxes (outline only), then MODEL boxes (filled)
  const g = w.gb;
  if (fam.gt && (L.gt_boxes || L.gt_other)) {
    for (let i = 0; i < g.b.length; i++) {
      const b = g.b[i]; if (!(b[0] >= 0 && b[0] <= X_MAX && Math.abs(b[1]) <= Y_HALF)) continue;
      const main = g.pos[i], ign = g.ign[i];
      if (main && !L.gt_boxes) continue;
      if (!main && !L.gt_other) continue;
      const pts = poly(footprint(b, g.y[i])); pts.push(pts[0]);
      ctx.strokeStyle = rgb(GT_C, main ? 1 : (ign ? 0.9 : 0.45)); ctx.lineWidth = main ? 2 : 1.4;
      if (main) strokePath(ctx, pts); else dashed(ctx, pts, 3, 2.5);
    }
  }
  const bs = (fam.model && L.boxes) ? boxState(w) : null;
  if (bs) {
    const bx = w.bx;
    for (const j of bs.idx) {
      const pts = poly(footprint(bx.b[j], bx.y[j])); pts.push(pts[0]);
      ctx.beginPath(); pts.forEach((p, q) => q ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath();
      ctx.fillStyle = rgb(BOX_C, 0.48); ctx.fill(); ctx.strokeStyle = bs.kind[j] === 1 ? '#fff' : rgb(BOX_C, 1); ctx.lineWidth = 1.8; ctx.stroke();
      const c = m2p(bx.b[j][0], bx.b[j][1]); ctx.fillStyle = '#dbeafe'; ctx.font = 'bold 13px system-ui'; ctx.fillText(bx.p[j].toFixed(2), c[0] + 6, c[1] - 5);
      if (bs.kind[j] === 1 && fam.gt && L.gt_boxes) {
        const gb = g.b[bs.link[j]]; const q = m2p(gb[0], gb[1]);
        ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(c[0], c[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
      }
    }
  }
  // --- GT path (dashed green, 1 s dots)
  if (L.gt_path && fam.gt && w.gtd.length) {
    ctx.strokeStyle = rgb(GT_C, 1); ctx.lineWidth = 4.2; dashed(ctx, poly([[0, 0]].concat(w.gtd)), 10, 7);
    ctx.fillStyle = rgb(GT_C, 1);
    for (let k = 9; k < w.gtd.length; k += 10) { const p = m2p(...w.gtd[k]); ctx.beginPath(); ctx.arc(p[0], p[1], 3.4, 0, 6.3); ctx.fill(); }
    const e = m2p(...w.gtd[w.gtd.length - 1]); chip(ctx, 'GT path', clamp(e[0] + 5, 2, BEV_W - 60), clamp(e[1] + 3, 12, BEV_H - 4), '#0b1a10', rgb(GT_C, 0.95), 'bold 14px system-ui');
  }
  // --- decoder pick (when different), then the emitted plan
  if (L.dec && fam.model && w.core !== null && w.core !== w.sel) {
    ctx.strokeStyle = rgb(DEC_C, 1); ctx.lineWidth = 3; strokePath(ctx, poly([[0, 0]].concat(candPath(w, w.core))));
    const e = m2p(...candPath(w, w.core)[7]); chip(ctx, 'MODEL decoder pick #' + w.core, clamp(e[0] + 5, 2, BEV_W - 130), clamp(e[1] + 14, 12, BEV_H - 4), '#fff', rgb(DEC_C, 0.95), 'bold 14px system-ui');
  }
  if (L.plan && fam.model) {
    ctx.strokeStyle = rgb(PLAN_C, 1); ctx.lineWidth = 4.4; strokePath(ctx, poly([[0, 0]].concat(w.tr)));
    w.tr.forEach(([x, y], q) => {
      const p = m2p(x, y);
      if (HZ[q] % 10 === 0) { ctx.fillStyle = rgb(PLAN_C, 1); ctx.beginPath(); ctx.arc(p[0], p[1], 4.4, 0, 6.3); ctx.fill(); ctx.strokeStyle = '#141414'; ctx.lineWidth = 1; ctx.stroke(); }
      else { ctx.strokeStyle = rgb(PLAN_C, 1); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p[0], p[1], 3.4, 0, 6.3); ctx.stroke(); }
    });
    const e = m2p(...w.tr[7]); chip(ctx, 'MODEL plan #' + w.sel + ' (E9)', clamp(e[0] + 7, 2, BEV_W - 125), clamp(e[1] - 6, 12, BEV_H - 4), '#fff', rgb(PLAN_C, 0.95), 'bold 14px system-ui');
  }
  // --- tactical geometric goals: MODEL g_tac (fuchsia diamonds) and the GT label (green rings)
  if (L.goals && fam.model) {
    w.gtac.forEach((g4, q) => {
      const p = m2p(g4[0], g4[1]), r = 7;
      ctx.fillStyle = rgb(GOAL_C, 0.9); ctx.strokeStyle = '#2a0b2f'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(p[0], p[1] - r); ctx.lineTo(p[0] + r, p[1]); ctx.lineTo(p[0], p[1] + r); ctx.lineTo(p[0] - r, p[1]); ctx.closePath(); ctx.fill(); ctx.stroke();
      const hx = p[0] - Math.sin(g4[2]) * 14, hy = p[1] - Math.cos(g4[2]) * 14;     // heading (0 = forward/up, +yaw = left)
      ctx.strokeStyle = rgb(GOAL_C, 1); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(hx, hy); ctx.stroke();
      ctx.fillStyle = rgb(GOAL_C, 1); ctx.font = 'bold 13px system-ui'; ctx.fillText('MODEL g' + (2 * (q + 1)) + 's ' + g4[3].toFixed(1) + 'm/s', clamp(p[0] + 9, 2, BEV_W - 92), clamp(p[1] + 4, 10, BEV_H - 4));
    });
  }
  if (L.gt_goals && fam.gt) {
    w.gtacg.forEach((g4, q) => {
      if (!w.gtacv[q]) return;
      const p = m2p(g4[0], g4[1]);
      ctx.strokeStyle = rgb(GT_C, 1); ctx.lineWidth = 2.2; ctx.setLineDash([3, 2]); ctx.beginPath(); ctx.arc(p[0], p[1], 8, 0, 6.3); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = rgb(GT_C, 1); ctx.font = 'bold 13px system-ui'; ctx.fillText('GT ' + (2 * (q + 1)) + 's', clamp(p[0] + 10, 2, BEV_W - 40), clamp(p[1] - 6, 10, BEV_H - 4));
    });
  }
}

// ------------------------------------------------------------------------------------------------ rendering: camera
function drawCamera(ctx, w, cam, im) {
  const L = S.L, fam = showFam();
  ctx.clearRect(0, 0, cam.W, cam.H);
  if (im && im.complete && im.naturalWidth) ctx.drawImage(im, 0, 0); else { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, cam.W, cam.H); ctx.fillStyle = '#667'; ctx.fillText('loading frame…', 10, 20); }
  const path = (pts, n) => densify(pts, n || 64).map(([x, y]) => cam.ground(x, y));
  // fan
  if (L.fan && fam.model) {
    const list = fanList(w), src = L.fan_src === 'dec' ? w.dc : w.e9, sc = list.map((k) => src[k]);
    const smin = Math.min(...sc), smax = Math.max(...sc);
    for (let r = list.length - 1; r >= 0; r--) {
      const k = list[r], col = fanColour(r, list.length, src[k], smin, smax);
      ctx.strokeStyle = rgb(col.c, col.a); ctx.lineWidth = r === 0 ? 3.5 : 2.4;
      const pp = path(candPath(w, k)); strokePath(ctx, pp);
      if (L.fan_lbl && list.length <= 8) {
        const last = pp.filter((p) => p).pop();
        if (last) { ctx.fillStyle = rgb(col.c, 1); ctx.font = 'bold 12px system-ui'; ctx.fillText('#' + (r + 1), clamp(last[0] + 4, 2, cam.W - 24), clamp(last[1] - 3, 12, cam.H - 4)); }
      }
    }
  }
  // boxes (GT outlines first, then the model's)
  const labels = [], placed = [];
  const edges = (cor, colStyle, lw, dash) => {
    ctx.strokeStyle = colStyle; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash);
    for (const [a, b] of CUBOID_EDGES) {
      const run = []; let started = false;
      ctx.beginPath();
      for (let s = 0; s <= 12; s++) {
        const u = s / 12, P = [cor[a][0] + (cor[b][0] - cor[a][0]) * u, cor[a][1] + (cor[b][1] - cor[a][1]) * u, cor[a][2] + (cor[b][2] - cor[a][2]) * u];
        const p = cam.project(P[0], P[1], P[2]);
        if (p[2]) { if (!started) { ctx.moveTo(p[0], p[1]); started = true; } else ctx.lineTo(p[0], p[1]); } else started = false;
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
  };
  const g = w.gb;
  if (fam.gt && (L.gt_boxes || L.gt_other)) {
    for (let i = 0; i < g.b.length; i++) {
      const main = g.pos[i], ign = g.ign[i];
      if (!(main || ign)) continue;                                 // hidden / out-of-field GT is BEV-only
      if (main && !L.gt_boxes) continue;
      if (!main && !L.gt_other) continue;
      const hz = g.zh[i], b = g.b[i];
      const cor = cuboidCorners(b[0], b[1], hz ? g.z[i] : 0, b[2], b[3], hz ? g.h[i] : 0, g.y[i]);
      edges(cor, rgb(GT_C, main ? 1 : 0.8), main ? 2.4 : 1.6, main ? null : [5, 4]);
      const top = cam.project(b[0], b[1], (hz ? g.z[i] + g.h[i] / 2 : 0));
      if (top[2]) labels.push({ x: top[0], y: top[1], t: 'GT ' + M.short_cls[g.c[i]], fg: rgb(GT_C, 1), bg: 'rgba(5,20,10,0.55)', pr: 1 });
    }
  }
  const bs = (fam.model && L.boxes) ? boxState(w) : null;
  if (bs) {
    const bx = w.bx;
    for (const j of bs.idx) {
      const b = bx.b[j], cor = cuboidCorners(b[0], b[1], bx.z[j], b[2], b[3], bx.h[j], bx.y[j]);
      // faint filled footprint + top face
      for (const face of [[0, 1, 2, 3], [4, 5, 6, 7]]) {
        const pts = face.map((q) => cam.project(cor[q][0], cor[q][1], cor[q][2]));
        if (pts.every((p) => p[2])) { ctx.beginPath(); pts.forEach((p, q) => q ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.fillStyle = rgb(BOX_C, 0.22); ctx.fill(); }
      }
      edges(cor, bs.kind[j] === 1 ? '#fff' : rgb(BOX_C, 1), 2.6, null);
      edges(cor, rgb(BOX_C, 1), 1.4, null);
      const top = cam.project(b[0], b[1], bx.z[j] + bx.h[j] / 2);
      if (top[2]) labels.push({ x: top[0], y: top[1], t: M.short_cls[bx.c[j]] + ' ' + bx.p[j].toFixed(2) + (bs.kind[j] === 1 ? ' TP' : ''), fg: '#fff', bg: rgb(BOX_C, 0.92), pr: 2 + bx.p[j] });
    }
  }
  labels.sort((a, b) => b.pr - a.pr);
  ctx.font = 'bold 11px system-ui,sans-serif';
  for (const lb of labels) {
    const tw = ctx.measureText(lb.t).width;
    for (const dy of [-16, -30, 2, -44, 16]) {
      const x = clamp(lb.x - tw / 2, 2, cam.W - tw - 6), y = lb.y + dy;
      if (y < 14 || y > cam.H - 4) continue;
      const rc = [x - 2, y - 11, x + tw + 3, y + 3];
      if (placed.every((q) => rc[2] < q[0] || rc[0] > q[2] || rc[3] < q[1] || rc[1] > q[3])) {
        placed.push(rc); ctx.fillStyle = lb.bg; ctx.fillRect(rc[0], rc[1], rc[2] - rc[0], rc[3] - rc[1]); ctx.fillStyle = lb.fg; ctx.fillText(lb.t, x, y); break;
      }
    }
  }
  // GT path (dashed green, wide), decoder pick, then the emitted plan on top
  if (L.gt_path && fam.gt && w.gtd.length) {
    const pp = path(w.gtd, 90);
    ctx.strokeStyle = rgb(GT_C, 1); ctx.lineWidth = 6; ctx.setLineDash([14, 9]); strokePath(ctx, pp); ctx.setLineDash([]);
    ctx.fillStyle = rgb(GT_C, 1);
    for (let k = 9; k < w.gtd.length; k += 10) { const q = cam.ground(w.gtd[k][0], w.gtd[k][1]); if (q) { ctx.beginPath(); ctx.arc(q[0], q[1], 4.6, 0, 6.3); ctx.fill(); } }
    const last = pp.filter((p) => p).pop(); if (last) chip(ctx, 'GT path', clamp(last[0] + 6, 2, cam.W - 70), clamp(last[1] + 14, 14, cam.H - 4), '#0b1a10', rgb(GT_C, 0.95));
  }
  if (L.dec && fam.model && w.core !== null && w.core !== w.sel) {
    const pp = path(candPath(w, w.core)); ctx.strokeStyle = rgb(DEC_C, 1); ctx.lineWidth = 5; strokePath(ctx, pp);
    const last = pp.filter((p) => p).pop(); if (last) chip(ctx, 'MODEL decoder pick #' + w.core, clamp(last[0] + 6, 2, cam.W - 190), clamp(last[1] + 30, 14, cam.H - 4), '#fff', rgb(DEC_C, 0.95));
  }
  if (L.plan && fam.model) {
    const pp = path(w.tr); ctx.strokeStyle = rgb(PLAN_C, 1); ctx.lineWidth = 4.6; strokePath(ctx, pp);
    w.tr.forEach(([x, y], q) => {
      const p = cam.ground(x, y); if (!p) return;
      if (HZ[q] % 10 === 0) { ctx.fillStyle = rgb(PLAN_C, 1); ctx.beginPath(); ctx.arc(p[0], p[1], 5.6, 0, 6.3); ctx.fill(); ctx.strokeStyle = '#141414'; ctx.lineWidth = 1; ctx.stroke(); }
      else { ctx.strokeStyle = rgb(PLAN_C, 1); ctx.lineWidth = 2.2; ctx.beginPath(); ctx.arc(p[0], p[1], 4.4, 0, 6.3); ctx.stroke(); }
    });
    const last = pp.filter((p) => p).pop(); if (last) chip(ctx, 'MODEL plan #' + w.sel + ' (E9)', clamp(last[0] + 8, 2, cam.W - 150), clamp(last[1] - 6, 14, cam.H - 4), '#fff', rgb(PLAN_C, 0.95));
  }
  // tactical goal points on the ground
  if (L.goals && fam.model) {
    w.gtac.forEach((g4, q) => {
      const p = cam.ground(g4[0], g4[1]); if (!p) return;
      ctx.fillStyle = rgb(GOAL_C, 0.95); ctx.strokeStyle = '#2a0b2f'; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(p[0], p[1] - 9); ctx.lineTo(p[0] + 9, p[1]); ctx.lineTo(p[0], p[1] + 9); ctx.lineTo(p[0] - 9, p[1]); ctx.closePath(); ctx.fill(); ctx.stroke();
      chip(ctx, 'g' + (2 * (q + 1)), p[0] + 11, p[1] + 4, '#2a0b2f', rgb(GOAL_C, 0.95), 'bold 11px system-ui');
    });
  }
  if (L.gt_goals && fam.gt) {
    w.gtacg.forEach((g4, q) => {
      if (!w.gtacv[q]) return;
      const p = cam.ground(g4[0], g4[1]); if (!p) return;
      ctx.strokeStyle = rgb(GT_C, 1); ctx.lineWidth = 2.6; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.arc(p[0], p[1], 10, 0, 6.3); ctx.stroke(); ctx.setLineDash([]);
      chip(ctx, 'GT' + (2 * (q + 1)), p[0] + 13, p[1] - 8, '#0b1a10', rgb(GT_C, 0.95), 'bold 11px system-ui');
    });
  }
  if ((L.plan || L.dec || L.fan) && fam.model) {
    ctx.font = 'bold 12px system-ui'; const t = 'MODEL plan: ' + planStamp(w), tw = ctx.measureText(t).width;
    ctx.fillStyle = 'rgba(9,12,17,0.82)'; ctx.fillRect(4, cam.H - 20, tw + 10, 17); ctx.fillStyle = w.pex ? '#fca5a5' : '#f5b45a'; ctx.fillText(t, 9, cam.H - 7);
  }
  // corner tags
  ctx.font = 'bold 12px system-ui';
  let x0 = 6;
  if (fam.model) x0 += chip(ctx, 'MODEL', x0, 18, '#fff', rgb(PLAN_C, 0.95)) + 4;
  if (fam.gt) chip(ctx, 'GT', x0, 18, '#0b1a10', rgb(GT_C, 0.95));
}

// ------------------------------------------------------------------------------------------------ DOM panels
function pct(x) { return (100 * x).toFixed(x >= 0.995 ? 0 : 1) + '%'; }
function barRows(names, probs, predI, gtI, cols, opts) {
  opts = opts || {};
  let h = '<div class="bars">';
  const order = opts.sort ? probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]) : probs.map((p, i) => i);
  for (const i of order) {
    const top = !opts.noTop && i === predI;
    const isGt = gtI !== null && gtI !== undefined && (Array.isArray(gtI) ? gtI.indexOf(i) >= 0 : gtI === i);
    const unsc = opts.scored && opts.scored[i] !== '1' && opts.labelled;
    let mk = '';
    if (isGt) mk += '<span class="g">GT</span> ';
    if (top) mk += '<span class="m">MODEL</span>';
    const tip = opts.conf ? `${names[i]}: validity ${pct(probs[i])} · confidence ${pct(opts.conf[i])}${unsc ? ' · UNSCORED token (no supervised negatives / below the n floor)' : ''}` : names[i];
    h += `<div class="r ${top ? 'top' : ''} ${isGt ? 'gtrow' : ''} ${unsc ? 'unscored' : ''}" title="${tip}"><span class="nm">${names[i]}</span>` +
      `<span class="bar"><b style="width:${(100 * probs[i]).toFixed(1)}%;${cols && !top ? '' : ''}"></b>${opts.thr ? `<span class="thr" style="left:${50}%"></span>` : ''}</span>` +
      `<span class="pv">${pct(probs[i])}</span><span class="mk">${mk}</span></div>`;
  }
  return h + '</div>';
}
function updatePanels(w) {
  const cd = S.cd;
  // inputs / given
  const ceil = w.ck === null ? 'no valid fed max speed (ceiling inert)' : `fed max speed ${w.vmr.toFixed(1)} m/s → ceiling step ${w.ck} km/h = ${w.vl.toFixed(1)} m/s`;
  const plan = `plan peak ${w.pvm.toFixed(1)} m/s ` + (w.vl === null ? '' : (w.pex ? `<b style="color:#fca5a5">EXCEEDS the ceiling ${w.vl.toFixed(1)}</b>` : `within the ceiling`));
  $('inputs').innerHTML = `<span class="given-chip">nav = ${w.nav.toUpperCase()} <span class="dim">(GIVEN input)</span></span> <span class="given-chip">ego speed v0 ${w.v0.toFixed(2)} m/s (${(3.6 * w.v0).toFixed(0)} km/h)</span> <span class="given-chip">${ceil}</span> <span class="plan-chip">MODEL ${plan}</span> <span class="stamp">${CEIL_STAMP}</span>`;
  // agreement
  const p3 = w.p3, lat = M.lat[w.latp.indexOf(Math.max(...w.latp))], lon = M.lon[w.lonp.indexOf(Math.max(...w.lonp))];
  const LAT3 = ['lane_keep', 'turn_left', 'turn_right'], LON3 = ['brake_stop', 'steady', 'accelerate'];
  const tri = (v, yes, no, na) => v === null || v === undefined ? `<span class="ag na">${na}</span>` : (v ? `<span class="ag yes">✓ ${yes}</span>` : `<span class="ag no">✗ ${no}</span>`);
  $('agree').innerHTML = '<div class="chipline">' +
    tri(p3.navok, `plan agrees with NAV ${w.nav.toUpperCase()}`, `plan does NOT follow NAV ${w.nav.toUpperCase()}`, 'nav: n/a') +
    tri(p3.latok, `plan agrees with tactical lat (${lat} ↔ ${LAT3[p3.lat3]})`, `plan DISAGREES with tactical lat (decoder ${lat}, plan ${LAT3[p3.lat3]})`, `tactical lat: n/a (${lat} has no turn counterpart)`) +
    tri(p3.lonok, `plan agrees with tactical lon (${lon} ↔ ${LON3[p3.lon3]})`, `plan DISAGREES with tactical lon (decoder ${lon}, plan ${LON3[p3.lon3]})`, 'tactical lon: n/a') +
    '</div><div class="small dim">DERIVED by this tool: the plan\'s first 2 s read as (lat3, lon3) by the programme\'s factor_from_kinematics v2 rule; nav = the programme\'s compliance predicate (left/right) or "does not turn" (follow). Coarse by construction. KNOWN-VALUE CONTROL: the same rule on the GT path agrees with the GT v7 labels on lat ' + M.control.lat[0] + '/' + M.control.lat[1] + ' and lon ' + M.control.lon[0] + '/' + M.control.lon[1] + ' labelled windows (all 12 clips); the lon reading is the weaker one.</div>';
  // lat / lon bars
  const latGt = w.latg, lonGt = w.long_;
  const gtNote = (g) => g === null ? '<div class="small dim">GT: not labelled at this instant (outside the label band)</div>' : '';
  $('latBox').innerHTML = '<div class="sub">LATERAL action <span class="dim">posterior (8)</span></div>' + barRows(M.lat, w.latp, w.latp.indexOf(Math.max(...w.latp)), latGt) + gtNote(latGt);
  $('lonBox').innerHTML = '<div class="sub">LONGITUDINAL action <span class="dim">posterior (8)</span></div>' + barRows(M.lon, w.lonp, w.lonp.indexOf(Math.max(...w.lonp)), lonGt) + gtNote(lonGt);
  const labelled = w.gsc.indexOf('1') >= 0;
  $('goalBox').innerHTML = barRows(M.goal, w.gp, -1, labelled ? w.ggt : null, null, { sort: true, noTop: true, scored: w.gsc, labelled, thr: true, conf: w.gcf }) +
    (labelled ? '' : '<div class="small dim">GT goal tokens: not labelled at this instant</div>');
  let gh = '<div class="small"><table class="fan"><tr><th></th><th>x m</th><th>y m</th><th>heading °</th><th>speed m/s</th></tr>';
  w.gtac.forEach((g4, q) => {
    const gg = w.gtacg[q], v = w.gtacv[q];
    gh += `<tr class="mrow"><td style="color:#e879f9"><b>MODEL</b> ${2 * (q + 1)} s</td><td>${g4[0].toFixed(1)}</td><td>${g4[1].toFixed(1)}</td><td>${(g4[2] * 57.2958).toFixed(1)}</td><td>${g4[3].toFixed(1)}</td></tr>` +
      `<tr class="grow"><td style="color:#6ee78a"><b>GT</b> ${2 * (q + 1)} s</td>` + (v ? `<td>${gg[0].toFixed(1)}</td><td>${gg[1].toFixed(1)}</td><td>${(gg[2] * 57.2958).toFixed(1)}</td><td>${gg[3].toFixed(1)}</td>` : '<td colspan="4" class="dim">not valid (clip ends)</td>') + '</tr>';
  });
  $('gtacBox').innerHTML = gh + '</table></div>';
  // metrics
  const m = w.m, f = (v, d) => v === null || v === undefined ? '–' : v.toFixed(d === undefined ? 2 : d);
  const ious = M.mapcls.map((nm, c) => ({ nm, c, u: m.un[c], v: m.un[c] > 0 ? m.it[c] / m.un[c] : null }));
  const bs = boxState(w), fam = showFam();
  let mh = `<div class="box"><h3>THIS FRAME <span class="tag model">MODEL</span> <span class="dim">vs</span> <span class="tag gt">GT</span></h3><div class="kv">` +
    `<div><b>${f(m.ade)}</b><span>ADE 8 slots (m)</span></div><div><b>${f(m.fde)}</b><span>FDE @ 6 s (m)</span></div><div><b>${f(m.spd)}</b><span>speed err 0–2 s (m/s)</span></div>` +
    `<div><b>${f(m.hd, 1)}</b><span>heading err 0–2 s (°)</span></div><div><b>${f(m.cv, 4)}</b><span>curvature err 0–2 s (1/m)</span></div></div>` +
    `<div class="small dim" style="margin-top:4px">ADE/FDE/speed/heading/curvature: the EMITTED plan (E9 pick #${w.sel}) against the GT path. ${w.sel !== w.core ? `The decoder's own pick would be #${w.core}.` : 'The decoder\'s own pick is the same candidate.'}</div></div>`;
  mh += `<div class="box"><h3>BOXES <span class="tag model">MODEL</span> p ≥ ${S.L.thr.toFixed(3)}${S.L.nms ? ' + NMS ' + S.L.nms_r.toFixed(1) + ' m' : ''} <span class="dim">vs</span> <span class="tag gt">GT</span> VIS-1</h3>` +
    (w.dt.lab ? `<div>shown <b>${bs.idx.length}</b> · TP <b>${bs.tp}</b> · FP <b>${bs.fp}</b> · DontCare ${bs.dc} · GT positives <b>${w.dt.np}</b> · recall ${w.dt.np ? (bs.tp / w.dt.np).toFixed(2) : '–'} · precision ${bs.idx.length ? (bs.tp / Math.max(bs.tp + bs.fp, 1)).toFixed(2) : '–'}</div>` +
      `<div class="small dim">greedy 2 m, positives ∪ IGNORE (the programme's rule, ported to JS and checked against Python by the self-test). Python reference at p ≥ ${M.thr_box.toFixed(4)}, no NMS: TP ${w.dt.tp0} / det ${w.dt.n0}.</div>` :
      '<div class="small" style="color:#f5b45a">no agent label on this frame — matches undefined</div>') + '</div>';
  mh += '<div class="box iou"><h3>MAP IoU THIS FRAME <span class="dim">(prior_corrected, scored cells; per class)</span></h3>';
  for (const r of ious) {
    mh += `<div class="r"><span>${r.nm}</span><span class="bar"><b style="width:${r.v === null ? 0 : (100 * r.v).toFixed(1)}%;background:rgb(${M.mapcol[r.c].join(',')})"></b></span><span>${r.v === null ? '— (absent)' : r.v.toFixed(2) + ' (∪ ' + r.u.toLocaleString() + ')'}</span></div>`;
  }
  $('metrics').innerHTML = mh + '</div>';
  // fan table
  const list = fanList(w), src = S.L.fan_src === 'dec' ? w.dc : w.e9;
  const flag = (b) => b === '1' ? '✓' : '<span style="color:#f87171">✗</span>';
  let th = '<table class="fan"><tr><th>#</th><th>cand</th><th>E9</th><th>dec</th><th>goal d</th><th>reach</th><th>ceil</th><th>nav</th></tr>';
  list.slice(0, 12).forEach((k, r) => {
    th += `<tr><td>${r + 1}</td><td>${k}${k === w.sel ? ' <b style="color:#f97316">plan</b>' : ''}${k === w.core && k !== w.sel ? ' <b style="color:#3b82f6">dec</b>' : (k === w.core ? '' : '')}</td><td>${w.e9[k].toFixed(2)}</td><td>${w.dc[k].toFixed(2)}</td><td>${w.gd[k].toFixed(1)}</td><td>${flag(w.rch[k])}</td><td>${flag(w.ceil[k])}</td><td>${w.nci ? flag(w.navc[k]) : '–'}</td></tr>`;
  });
  th += '</table>';
  const nk = (s) => s.split('').filter((c) => c === '1').length;
  const spread = (ids) => {                                   // mean pairwise distance of the 6 s endpoints
    const f = fanOf(w), e = ids.map((k) => [f[(k * 8 + 7) * 2], f[(k * 8 + 7) * 2 + 1]]);
    if (e.length < 2) return 0; let s_ = 0;
    for (let a = 0; a < e.length; a++) for (let b = a + 1; b < e.length; b++) s_ += Math.hypot(e[a][0] - e[b][0], e[a][1] - e[b][1]);
    return 2 * s_ / (e.length * (e.length - 1));
  };
  const allIds = w.e9.map((v, k) => k), reachIds = allIds.filter((k) => w.rch[k] === '1'), ceilIds = reachIds.filter((k) => w.ceil[k] === '1');
  $('fanTable').innerHTML = th + `<div class="small dim">fan: ${w.e9.length} candidates · kept by reach ${nk(w.rch)} · by ceiling ${nk(w.ceil)} · complying with nav ${w.nci ? nk(w.navc) : 'n/a (nav has no side)'} · ranked by ${S.L.fan_src === 'dec' ? "the decoder's score" : 'the E9 score (final)'}</div>` +
    (S.ceilFallback ? '<div class="small" style="color:#f5b45a">no candidate satisfies the ceiling in this window: the decoder keeps the whole fan (empty-survivor rule), so the ceiling filter is not applied here.</div>' : '') +
    `<div class="small dim">how messy: mean pairwise distance of the 6 s endpoints — all ${allIds.length}: <b>${spread(allIds).toFixed(1)} m</b> · reach-kept ${reachIds.length}: <b>${spread(reachIds).toFixed(1)} m</b> · also under the ceiling ${ceilIds.length}: <b>${spread(ceilIds).toFixed(1)} m</b> · shown top-${list.length}: <b>${spread(list).toFixed(1)} m</b></div>`;
}

// ------------------------------------------------------------------------------------------------ strip
let stripBase = null, stripGeom = null;
function buildStrip() {
  const cd = S.cd, n = cd.w.length, cv = $('strip'), W = cv.width, H = cv.height;
  const left = 140, right = 8, cw = (W - left - right) / n, rowH = 14, gap = 2, top = 4;
  stripGeom = { left, right, cw, n, rowH, top, gap };
  const off = document.createElement('canvas'); off.width = W; off.height = H;
  const c = off.getContext('2d');
  c.fillStyle = '#0a0e14'; c.fillRect(0, 0, W, H);
  const rows = [['MODEL', 'lat', 'm'], ['GT', 'lat', 'g'], ['MODEL', 'lon', 'm'], ['GT', 'lon', 'g'], ['MODEL', 'plan vs tactical', 'a'], ['MODEL', 'plan vs nav', 'n']];
  rows.forEach((r, ri) => {
    const y = top + ri * (rowH + gap);
    c.font = 'bold 11px system-ui'; c.fillStyle = r[0] === 'GT' ? rgb(GT_C, 1) : rgb(PLAN_C, 1); c.fillText(r[0], 4, y + 11);
    c.fillStyle = '#9aa7b8'; c.font = '11px system-ui'; c.fillText(r[1], r[0] === 'GT' ? 30 : 52, y + 11);
    for (let i = 0; i < n; i++) {
      const w = cd.w[i], x = left + i * cw;
      let col = null, extra = null;
      if (r[2] === 'm') col = (r[1] === 'lat' ? LAT_COL : LON_COL)[(r[1] === 'lat' ? w.latp : w.lonp).indexOf(Math.max(...(r[1] === 'lat' ? w.latp : w.lonp)))];
      else if (r[2] === 'g') { const g = r[1] === 'lat' ? w.latg : w.long_; col = g === null ? null : (r[1] === 'lat' ? LAT_COL : LON_COL)[g]; }
      else if (r[2] === 'a') { const a = w.p3.latok, b = w.p3.lonok; if (a === null && b === null) col = '#2a3342'; else col = (a === false || b === false) ? '#ef4444' : '#3b82f6'; }
      else { const a = w.p3.navok; col = a === null ? '#2a3342' : (a ? '#3b82f6' : '#ef4444'); }
      if (col === null) { c.fillStyle = '#141b26'; c.fillRect(x, y, Math.ceil(cw), rowH); c.fillStyle = '#1d2736'; for (let q = 0; q < rowH; q += 4) c.fillRect(x, y + q, Math.ceil(cw), 1); }
      else { c.fillStyle = col; c.fillRect(x, y, Math.ceil(cw), rowH); }
      if (r[2] === 'g' && col !== null) {                              // a GT cell where the MODEL disagrees gets a red cap on the MODEL row
        const mi = (r[1] === 'lat' ? w.latp : w.lonp).indexOf(Math.max(...(r[1] === 'lat' ? w.latp : w.lonp)));
        const g = r[1] === 'lat' ? w.latg : w.long_;
        if (mi !== g) { c.fillStyle = '#ff2d2d'; c.fillRect(x, y - 3, Math.ceil(cw), 3); }
      }
    }
    c.strokeStyle = r[0] === 'GT' ? rgb(GT_C, 0.9) : 'rgba(249,115,22,0.8)'; c.lineWidth = 1; c.strokeRect(left - 0.5, y - 0.5, cw * n + 1, rowH + 1);
  });
  // time axis: every 5 s
  c.fillStyle = '#8c98a8'; c.font = '10px system-ui';
  const t0 = cd.w[0].t, t1 = cd.w[n - 1].t, ya = top + rows.length * (rowH + gap) + 12;
  for (let t = Math.ceil(t0 / 5) * 5; t <= t1; t += 5) {
    const i = Math.round((t - t0) / ((t1 - t0) / Math.max(n - 1, 1))), x = left + (i + 0.5) * cw;
    c.fillRect(x, ya - 12, 1, 5); c.fillText(t.toFixed(0) + ' s', x - 8, ya);
  }
  stripBase = off;
  $('stripLegend').innerHTML = '<span style="flex-basis:100%">red cap on a MODEL row = MODEL decision ≠ GT label · hatched = not labelled at that instant · plan rows: blue = agrees, red = disagrees, grey = n/a · click to seek</span><span>lat:</span>' + M.lat.map((n_, i) => `<span><i style="background:${LAT_COL[i]}"></i>${n_}</span>`).join('') +
    '<span style="margin-left:14px">lon:</span>' + M.lon.map((n_, i) => `<span><i style="background:${LON_COL[i]}"></i>${n_}</span>`).join('');
}
function drawStrip() {
  if (!stripBase) return;
  const cv = $('strip'), c = cv.getContext('2d');
  c.drawImage(stripBase, 0, 0);
  const g = stripGeom, x = g.left + (S.i + 0.5) * g.cw;
  c.strokeStyle = '#fff'; c.lineWidth = 2; c.beginPath(); c.moveTo(x, 0); c.lineTo(x, g.top + 6 * (g.rowH + g.gap)); c.stroke();
}

// ------------------------------------------------------------------------------------------------ main render
function render() {
  rafQueued = false;
  const cd = S.cd; if (!cd) return;
  const w = cd.w[S.i], L = S.L, fam = showFam();
  const cam = cd.camObj;
  $('grid').classList.toggle('v-gt', L.view === 'gt'); $('grid').classList.toggle('v-model', L.view === 'model');
  drawCamera($('cam').getContext('2d'), w, cam, camImage(S.rank, S.i));
  const overlay = L.layout === 'overlay';
  $('bevPanelA').classList.toggle('hidden', !overlay && L.view === 'gt');
  $('bevPanelB').classList.toggle('hidden', overlay || L.view === 'model');
  $('bevNameA').textContent = overlay ? 'predicted 10 cm map (fill) + SAM3 GT map (green outline)' : 'predicted 10 cm map';
  $('bevTagA').textContent = overlay ? 'MODEL + GT' : 'MODEL';
  if (!$('bevPanelA').classList.contains('hidden')) drawBEV($('bevA').getContext('2d'), w, 'A');
  if (!$('bevPanelB').classList.contains('hidden')) drawBEV($('bevB').getContext('2d'), w, 'B');
  drawStrip();
  $('scrub').value = S.i;
  $('timeLbl').textContent = 't = ' + w.t.toFixed(1) + ' s';
  $('frameLbl').textContent = 'frame ' + (S.i + 1) + ' / ' + cd.w.length;
  updatePanels(w);
  prefetch();
}
function queueRender() {
  if (rafQueued) return; rafQueued = true;
  const run = () => { if (!rafQueued) return; render(); };            // render() clears the flag; the slower of rAF / timeout then no-ops
  requestAnimationFrame(run); setTimeout(run, 60);                     // the timeout keeps a hidden / throttled tab alive
}

// ------------------------------------------------------------------------------------------------ clip loading
function loadClip(rank, then) {
  S.rank = rank; S.i = 0; setPlaying(false);
  $('clipSel').value = rank;
  const go = () => {
    S.cd = RV7.clips[rank]; S.cd.camObj = new Cam(S.cd.cam);
    // keep only the two most recently visited clips in memory (a clip is ~10 MB of text, ~40 MB parsed)
    S.visited = (S.visited || []).filter((r) => r !== rank); S.visited.push(rank);
    while (S.visited.length > 2) {
      const old = S.visited.shift(); delete RV7.clips[old];
      for (const k of [...mapCache.keys()]) if (k.startsWith(old + ':')) mapCache.delete(k);
      for (const k of [...imgCache.keys()]) if (k.startsWith(old + ':')) imgCache.delete(k);
    }
    $('scrub').max = S.cd.w.length - 1;
    $('camLabel').textContent = M.clips[rank - 1].why + ' · sha12 ' + S.cd.sha12;
    buildStrip(); queueRender(); if (then) then();
  };
  if (RV7.clips[rank]) { go(); return; }
  $('camLabel').textContent = 'loading clip ' + rank + '…';
  const s = document.createElement('script');
  s.src = M.clips[rank - 1].file; s.onload = go;
  s.onerror = () => { $('camLabel').textContent = 'FAILED to load ' + M.clips[rank - 1].file; };
  document.head.appendChild(s);
}
function seek(i) { if (!S.cd) return; S.i = clamp(Math.round(i), 0, S.cd.w.length - 1); queueRender(); }
function setPlaying(p) {
  S.playing = p; S.acc = 0; S.lastTs = 0; $('bPlay').textContent = p ? '⏸ pause' : '▶ play';
  clearInterval(S.timer); S.timer = null;
  if (p) S.timer = setInterval(tick, 20);                               // a timer, not rAF: it keeps running in a throttled tab
}
function tick() {
  if (!S.playing || !S.cd) return;
  const ts = performance.now();
  if (!S.lastTs) S.lastTs = ts;
  S.acc += (ts - S.lastTs) * parseFloat($('speedSel').value); S.lastTs = ts;
  const step = 100;                                                     // 10 fps = 0.1 s of recorded time per frame
  let moved = false;
  while (S.acc >= step) {
    S.acc -= step; moved = true;
    if (S.i < S.cd.w.length - 1) S.i++;
    else if ($('loopChk').checked) S.i = 0;
    else { setPlaying(false); break; }
  }
  if (moved) queueRender();
}

// ------------------------------------------------------------------------------------------------ controls
function el(tag, attrs, html) { const e = document.createElement(tag); Object.assign(e, attrs || {}); if (html) e.innerHTML = html; return e; }
function addCheck(parent, key, label, sub, swatch) {
  const l = el('label'); if (sub) l.className = 'sub';
  const i = el('input', { type: 'checkbox', checked: !!S.L[key] }); i.onchange = () => { S.L[key] = i.checked; queueRender(); };
  l.appendChild(i); l.insertAdjacentHTML('beforeend', (swatch || '') + '<span>' + label + '</span>'); parent.appendChild(l); return i;
}
function addRange(parent, key, label, min, max, step, fmt, onchange) {
  const wrap = el('div'); const row = el('div', { className: 'row' });
  const lab = el('span', { className: 'dim' }, label); const v = el('span', { className: 'v mono' });
  const i = el('input', { type: 'range', min, max, step, value: S.L[key] });
  const show = () => { v.textContent = fmt(S.L[key]); };
  i.oninput = () => { S.L[key] = parseFloat(i.value); show(); if (onchange) onchange(); queueRender(); };
  show(); row.appendChild(lab); row.appendChild(v); wrap.appendChild(row); wrap.appendChild(i); parent.appendChild(wrap); return i;
}
function addSelect(parent, key, label, opts) {
  const row = el('div', { className: 'row' }); row.appendChild(el('span', { className: 'dim' }, label));
  const s = el('select'); opts.forEach(([v, t]) => s.appendChild(el('option', { value: v }, t))); s.value = S.L[key];
  s.onchange = () => { S.L[key] = s.value; queueRender(); }; row.appendChild(s); parent.appendChild(row);
}
function buildControls() {
  const gt = '<span class="sw dash" style="border-color:#6ee78a"></span>';
  const lb = $('layerBox'); lb.innerHTML = '<h3>LAYERS</h3>';
  lb.appendChild(el('h4', null, '<span class="tag gt">GT</span> ground truth — green'));
  addCheck(lb, 'gt_path', 'GT path (dashed, dots = 1 s)', false, gt);
  addCheck(lb, 'gt_boxes', 'GT boxes — VIS-1 positives (outline)', false, '<span class="sw fill" style="border-color:#6ee78a"></span>');
  addCheck(lb, 'gt_other', 'GT boxes — IGNORE / hidden (dashed)', true);
  addCheck(lb, 'gt_goals', 'GT tactical goals 2 / 4 / 6 s (rings)', false, '<span class="sw" style="border-color:#6ee78a"></span>');
  addCheck(lb, 'gt_map', 'GT map (SAM3, 10 cm)', false);
  lb.appendChild(el('h4', null, '<span class="tag model">MODEL</span> model output — orange / blue'));
  addCheck(lb, 'plan', 'emitted plan = E9 pick (orange)', false, '<span class="sw" style="border-color:#f97316"></span>');
  addCheck(lb, 'dec', "decoder's own pick, when different (blue)", false, '<span class="sw" style="border-color:#3b82f6"></span>');
  addCheck(lb, 'boxes', 'detections (blue, filled)', false, '<span class="sw fill" style="border-color:#3884ff;background:rgba(56,132,255,.5)"></span>');
  addRange(lb, 'thr', 'score threshold (default = the TRAIN P=R gate)', M.thr_min, 0.9, 0.001, (v) => v.toFixed(3));
  const nmsRow = el('div'); lb.appendChild(nmsRow);
  addCheck(nmsRow, 'nms', 'NMS (BEV centre distance)', true);
  addRange(nmsRow, 'nms_r', 'NMS radius (m)', 0.5, 6, 0.1, (v) => v.toFixed(1) + ' m');
  addCheck(lb, 'goals', 'tactical geometric goals g_tac 2 / 4 / 6 s (diamonds)', false, '<span class="sw" style="border-color:#e879f9"></span>');
  addCheck(lb, 'pred_map', 'predicted map (10 cm)', false);
  const fb = $('fanBox'); fb.innerHTML = '<h3>MODEL FAN — 117 candidates</h3>';
  addCheck(fb, 'fan', 'show the fan (blue → indigo by rank)', false, '<span class="sw" style="border-color:#7aa7ff"></span>');
  addRange(fb, 'fan_k', 'top-k candidates shown', 1, 117, 1, (v) => String(v));
  addSelect(fb, 'fan_src', 'rank by', [['e9', 'E9 score (final pick)'], ['dec', "decoder's own score"]]);
  addSelect(fb, 'fan_col', 'colour by', [['rank', 'rank'], ['score', 'score']]);
  fb.appendChild(el('div', { className: 'small dim' }, 'only candidates kept by:'));
  addCheck(fb, 'f_reach', 'reachability mask', true);
  addCheck(fb, 'f_ceil', 'speed-ceiling mask (planned peak ≤ ceiling)', true);
  addCheck(fb, 'f_nav', 'complies with nav (left / right only)', true);
  addCheck(fb, 'fan_lbl', 'rank labels (k ≤ 8)', true);
  fb.appendChild(el('div', { id: 'fanTable' }));
  const mb = $('mapBox'); mb.innerHTML = '<h3>MAPS</h3>';
  addRange(mb, 'map_alpha', 'map opacity', 0, 1, 0.05, (v) => v.toFixed(2));
  addSelect(mb, 'layout', 'layout', [['side', 'side by side (MODEL | GT)'], ['overlay', 'overlay (MODEL fill + GT outline)']]);
  addCheck(mb, 'dim', 'dim cells that are not scored', false);
  const chips = el('div', { className: 'chips' });
  M.mapcls.forEach((nm, c) => {
    const ch = el('span', { className: 'chip' }, `<i style="background:rgb(${M.mapcol[c].join(',')})"></i>${nm}`);
    ch.onclick = () => { S.L.cls[c] = !S.L.cls[c]; ch.classList.toggle('off', !S.L.cls[c]); queueRender(); };
    chips.appendChild(ch);
  });
  mb.appendChild(chips);
  mb.appendChild(el('div', { className: 'small dim' }, 'A map cell is SCORED when SAM3 saw it AND the lift reaches it now; other cells are dimmed (drawn, not scored). The prediction is map_head_hires.decide(prior_corrected).'));
  const lg = $('legendBox'); lg.innerHTML = '<h3>LEGEND — one rule</h3>' +
    `<div><span class="sw dash" style="border-color:#6ee78a"></span> <b style="color:#6ee78a">GT</b> path (dashed)</div>` +
    `<div><span class="sw fill" style="border-color:#6ee78a;background:transparent"></span> <b style="color:#6ee78a">GT</b> boxes (outline, "GT car")</div>` +
    `<div><span class="sw" style="border-color:#f97316"></span> <b style="color:#f97316">MODEL</b> emitted plan (E9 pick)</div>` +
    `<div><span class="sw" style="border-color:#3b82f6"></span> <b style="color:#3b82f6">MODEL</b> decoder's pick (if different)</div>` +
    `<div><span class="sw" style="border-color:#a5e1ff"></span><span class="sw" style="border-color:#5846d6"></span> <b style="color:#7aa7ff">MODEL</b> fan, rank 1 → k</div>` +
    `<div><span class="sw fill" style="border-color:#3884ff;background:rgba(56,132,255,.5)"></span> <b style="color:#7aa7ff">MODEL</b> boxes (filled, score)</div>` +
    `<div><span style="color:#e879f9">◆</span> <b style="color:#e879f9">MODEL</b> g_tac goal · <span style="color:#6ee78a">◯</span> <b style="color:#6ee78a">GT</b> goal label</div>` +
    `<div class="small stamp" style="margin-top:6px">${CEIL_STAMP}</div>` +
    `<div class="small dim">OPEN-LOOP replay of recorded held-out windows — not closed-loop driving. One DDIM draw per window (seed 0).</div>`;
}

// ------------------------------------------------------------------------------------------------ self-test
RV7.selfTest = function () {
  const out = [], cd = S.cd;
  if (!cd) return 'no clip loaded';
  // 1. projection vs the Python RigCamera values
  const tv = cd.tv; let md = 0, okMis = 0;
  tv.xyz.forEach((p, k) => {
    const q = cd.camObj.project(p[0], p[1], p[2]), r = tv.cro[k];
    if (!!q[2] !== !!r[2]) okMis++; else if (q[2]) md = Math.max(md, Math.abs(q[0] - r[0]), Math.abs(q[1] - r[1]));
  });
  out.push(`projection: ${tv.xyz.length} 3-D points vs the model's RigCamera (Python): max |Δ| = ${md.toExponential(2)} px, valid-flag mismatches = ${okMis}`);
  // 2. greedy matcher vs the stored Python pairs at the P=R gate, no NMS, whole clip
  let nw = 0, bad = 0, tie = 0;
  cd.w.forEach((w) => {
    if (!w.dt.lab) return; nw++;
    const idx = []; w.bx.p.forEach((p, j) => { if (p >= M.thr_box - 1e-12) idx.push(j); });
    const rows = greedyMatch(idx.map((j) => w.bx.b[j]), idx.map((j) => w.bx.p[j]), w.gb.b, w.gb.pos, w.gb.ign, M.match_dist);
    const mine = rows.map(([i, g, k]) => [w.bx.i[idx[i]], g, k]);
    if (JSON.stringify(mine) === JSON.stringify(w.dt.pairs)) return;
    const key = (r) => JSON.stringify(r), a_ = mine.map(key).sort(), b_ = w.dt.pairs.map(key).sort();
    if (JSON.stringify(a_) === JSON.stringify(b_)) tie++; else bad++;          // same TP / FP / DontCare, only the order of tied scores differs
  });
  out.push(`box matcher: ${nw} agent-labelled windows of this clip, JS greedy vs Python pairs at p >= ${M.thr_box.toFixed(4)}: windows with a different TP / FP / DontCare set = ${bad}; same set but tied scores listed in another order = ${tie}`);
  // 3. fan decode: the E9 pick's candidate equals the emitted plan (cm quantisation)
  let fd = 0; cd.w.forEach((w) => { const p = candPath(w, w.sel); for (let s = 0; s < 8; s++) fd = Math.max(fd, Math.abs(p[s][0] - w.tr[s][0]), Math.abs(p[s][1] - w.tr[s][1])); });
  out.push(`fan: decoded candidate #sel vs the emitted plan over ${cd.w.length} windows: max |Δ| = ${fd.toFixed(4)} m (int16 centimetre quantisation: <= 0.005)`);
  // 4. picks: argmax of the exported scores
  let pk = 0, pk2 = 0; cd.w.forEach((w) => {
    let b = -1, bv = -Infinity; w.e9.forEach((v, k) => { if (w.rch[k] === '1' && v > bv) { bv = v; b = k; } }); if (b !== w.sel) pk++;
    let c = -1, cv = -Infinity, any = false; w.dc.forEach((v, k) => { if (w.rch[k] === '1' && w.ceil[k] === '1') any = true; });
    w.dc.forEach((v, k) => { if (w.rch[k] === '1' && (w.ceil[k] === '1' || !any) && v > cv) { cv = v; c = k; } }); if (w.core !== null && c !== w.core) pk2++;
  });
  out.push(`picks: E9 pick != argmax(E9 score | reach) in ${pk} windows; decoder pick != argmax(decoder score | reach & ceiling) in ${pk2} windows (of ${cd.w.length})`);
  // 5. maps: scored-cell count vs the trainer's, current window
  const maps = mapCache.get(camKey(S.rank, S.i)); const w = cd.w[S.i];
  if (maps && maps !== 'pending') {
    let ns = 0; for (let k = 0; k < maps.gt.length; k++) { const v = maps.gt[k]; if (!((v >> 3) & 1) && !((v >> 4) & 1)) ns++; }
    let ps = 0; for (let k = 0; k < maps.pred.length; k++) { const v = maps.pred[k]; if (!((v >> 3) & 1) && !((v >> 4) & 1)) ps++; }
    out.push(`maps (frame ${S.i + 1}): scored cells GT-png ${ns.toLocaleString()} · pred-png ${ps.toLocaleString()} · trainer n_supervised ${w.m.nsc.toLocaleString()}  ${ns === w.m.nsc && ps === w.m.nsc ? 'OK' : 'MISMATCH'}`);
  } else out.push('maps: not decoded yet — press self-test again');
  const el_ = $('selftest'); el_.textContent = 'SELF-TEST (clip ' + S.rank + ')\n' + out.join('\n'); el_.classList.remove('hidden'); el_.onclick = () => el_.classList.add('hidden');
  return out.join('\n');
};

// ------------------------------------------------------------------------------------------------ boot
function boot() {
  $('runlabel').textContent = M.run; $('stepinfo').textContent = `final checkpoint · step ${M.step.toLocaleString()} · ckpt md5 ${M.ckpt_md5}`;
  $('tiernote').textContent = 'OPEN-LOOP replay of recorded held-out eval139 windows — NOT closed-loop driving. ' + CEIL_STAMP + '.';
  document.title = 'refcv7 replay · step ' + M.step;
  const sel = $('clipSel');
  M.clips.forEach((c) => sel.appendChild(el('option', { value: c.rank }, `${c.rank}. ${c.slot} · nav ${c.nav.toUpperCase()} · ${c.n} windows · ${c.sha12}`)));
  sel.onchange = () => loadClip(parseInt(sel.value, 10));
  $('bPlay').onclick = () => setPlaying(!S.playing);
  $('bBack').onclick = () => { setPlaying(false); seek(S.i - 1); };
  $('bFwd').onclick = () => { setPlaying(false); seek(S.i + 1); };
  $('bPrevClip').onclick = () => loadClip(S.rank > 1 ? S.rank - 1 : M.clips.length);
  $('bNextClip').onclick = () => loadClip(S.rank < M.clips.length ? S.rank + 1 : 1);
  $('scrub').oninput = () => seek(parseInt($('scrub').value, 10));
  $('bHelp').onclick = () => $('help').classList.toggle('hidden');
  $('bSelfTest').onclick = () => RV7.selfTest();
  document.querySelectorAll('input[name=view]').forEach((r) => { r.onchange = () => { if (r.checked) { S.L.view = r.value; queueRender(); } }; });
  const strip = $('strip');
  const stripSeek = (ev) => { const r = strip.getBoundingClientRect(), x = (ev.clientX - r.left) * strip.width / r.width; if (stripGeom) seek(Math.floor((x - stripGeom.left) / stripGeom.cw)); };
  strip.onclick = (ev) => { setPlaying(false); stripSeek(ev); };
  strip.onmousemove = (ev) => {
    if (!S.cd || !stripGeom) return;
    const r = strip.getBoundingClientRect(), x = (ev.clientX - r.left) * strip.width / r.width, i = Math.floor((x - stripGeom.left) / stripGeom.cw);
    if (i < 0 || i >= S.cd.w.length) { strip.title = ''; return; }
    const w = S.cd.w[i], mi = (a) => a.indexOf(Math.max(...a));
    strip.title = `frame ${i + 1}  t=${w.t.toFixed(1)} s\nMODEL lat ${M.lat[mi(w.latp)]} · lon ${M.lon[mi(w.lonp)]}\nGT lat ${w.latg === null ? '–' : M.lat[w.latg]} · lon ${w.long_ === null ? '–' : M.lon[w.long_]}`;
  };
  document.addEventListener('keydown', (ev) => {
    const tg = ev.target, tn = tg && tg.tagName;
    const k = ev.key;
    if (tg && /SELECT|TEXTAREA/.test(tn) && (k === ' ' || k.startsWith('Arrow'))) return;
    if (tg && tn === 'INPUT' && tg.type === 'range' && k.startsWith('Arrow')) return;   // the control keeps its own arrows
    if (tg && tn === 'INPUT' && tg.type !== 'range' && k === ' ') return;               // a checkbox keeps its space
    if (tg && tn === 'BUTTON') tg.blur();
    if (k === ' ') { ev.preventDefault(); setPlaying(!S.playing); }
    else if (k === 'ArrowRight') { ev.preventDefault(); setPlaying(false); seek(S.i + (ev.shiftKey ? 10 : 1)); }
    else if (k === 'ArrowLeft') { ev.preventDefault(); setPlaying(false); seek(S.i - (ev.shiftKey ? 10 : 1)); }
    else if (k === '[') $('bPrevClip').click();
    else if (k === ']') $('bNextClip').click();
    else if (k === 'b' || k === 'g' || k === 'm') { const v = { b: 'both', g: 'gt', m: 'model' }[k]; S.L.view = v; document.querySelector(`input[name=view][value=${v}]`).checked = true; queueRender(); }
  });
  // optional deep-link overrides, e.g. ?clip=7&frame=60&L.layout=overlay&L.view=model&L.fan_k=8 (used for screenshots)
  const q0 = new URLSearchParams(location.search);
  q0.forEach((v, k) => {
    if (!k.startsWith('L.') || !(k.slice(2) in S.L)) return;
    const key = k.slice(2), cur = S.L[key];
    S.L[key] = typeof cur === 'boolean' ? (v === '1' || v === 'true') : (typeof cur === 'number' ? parseFloat(v) : v);
  });
  const vr = document.querySelector(`input[name=view][value=${S.L.view}]`); if (vr) vr.checked = true;
  buildControls();
  document.addEventListener('click', (ev) => { if (ev.target && ev.target.tagName === 'BUTTON') ev.target.blur(); });
  const q = new URLSearchParams(location.search);
  loadClip(parseInt(q.get('clip') || '1', 10), () => { if (q.get('frame')) seek(parseInt(q.get('frame'), 10)); });
}
RV7.state = S; RV7.render = render; RV7.seek = seek; RV7.loadClip = loadClip;
boot();
})();
