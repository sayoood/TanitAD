/* TanitResim SPA — vanilla JS + canvas, no build step.
 *
 * Two views: (1) HOME = scenario cards, one per episode of the selected
 * session; (2) SESSION = one column per arm (camera fan + steer/accel charts
 * + head readouts) over a shared BEV + error-strip master panel with a
 * scrubber. Everything is legended. URL hash carries session/episode/step so
 * a view is shareable: #/s/<id>/e/<ep>/t/<step>. */
(function () {
  "use strict";

  var GT = "#eef2f7";
  var EGO_DIM = "#63728c";

  var S = {
    sessions: [],        // /api/sessions summaries
    id: null,            // current session id
    sess: null,          // full session.json
    view: "home",
    ep: 0, step: 0,
    showGT: true,
    playing: false, timer: null,
    imgs: {},            // frame filename -> Image (per loaded session)
    bevRange: {},        // ep idx -> {fwd, lat}
    dom: null,           // cached session-view element refs
  };

  // ---------- utilities -------------------------------------------------
  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }
  function getJSON(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(url + " -> " + r.status);
      return r.json();
    });
  }
  function fmt(v, nd) {
    if (v == null || isNaN(v)) return "–";
    return Number(v).toFixed(nd == null ? 3 : nd);
  }
  function armColor(name) {
    var a = (S.sess && S.sess.meta.arms || []).find(function (x) { return x.name === name; });
    return a ? a.color : "#9aa7bd";
  }
  function armNames() {
    return (S.sess ? S.sess.meta.arms : []).map(function (a) { return a.name; });
  }
  function curEpisode() { return S.sess.episodes[S.ep]; }
  function curStep() { return curEpisode().steps[S.step]; }

  // ---------- routing ---------------------------------------------------
  function readHash() {
    var h = location.hash.replace(/^#\/?/, "");
    var p = h.split("/").filter(Boolean);   // ["s",<id>,"e",<ep>,"t",<step>]
    var out = { id: null, view: "home", ep: 0, step: 0 };
    for (var i = 0; i < p.length; i += 2) {
      if (p[i] === "s") out.id = decodeURIComponent(p[i + 1] || "");
      else if (p[i] === "e") { out.ep = +p[i + 1] || 0; out.view = "session"; }
      else if (p[i] === "t") out.step = +p[i + 1] || 0;
    }
    return out;
  }
  function writeHash() {
    var h = "#/";
    if (S.id) h += "s/" + encodeURIComponent(S.id);
    if (S.view === "session") h += "/e/" + S.ep + "/t/" + S.step;
    if (location.hash !== h) history.replaceState(null, "", h);
  }

  // ---------- boot ------------------------------------------------------
  function boot() {
    getJSON("/api/sessions").then(function (list) {
      S.sessions = list;
      var r = readHash();
      var id = r.id || (list[0] && list[0].id);
      if (!id) { document.getElementById("app").innerHTML =
        '<div class="loading">No session bundles found. Export one with ' +
        '<code>replay_app.py --mode export</code>.</div>'; return; }
      loadSession(id, r);
    }).catch(function (e) {
      document.getElementById("app").innerHTML =
        '<div class="loading">Failed to load sessions: ' + e.message + '</div>';
    });
    window.addEventListener("hashchange", onHashChange);
    window.addEventListener("keydown", onKey);
  }

  function onHashChange() {
    var r = readHash();
    var id = r.id || S.id;
    if (id !== S.id) { loadSession(id, r); return; }
    S.view = r.view; S.ep = clamp(r.ep, 0, S.sess.episodes.length - 1);
    S.step = clamp(r.step, 0, curEpisode().steps.length - 1);
    render();
  }

  function loadSession(id, r) {
    stopPlay();
    getJSON("/api/session/" + encodeURIComponent(id)).then(function (sess) {
      S.id = id; S.sess = sess; S.imgs = {}; S.bevRange = {}; S.dom = null;
      S.view = (r && r.view) || "home";
      S.ep = clamp((r && r.ep) || 0, 0, sess.episodes.length - 1);
      S.step = clamp((r && r.step) || 0, 0, sess.episodes[S.ep].steps.length - 1);
      preloadEpisode(S.ep);
      render();
    });
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function preloadEpisode(ep) {
    var steps = S.sess.episodes[ep].steps;
    steps.forEach(function (st) {
      if (S.imgs[st.frame]) return;
      var img = new Image();
      img.onload = function () { if (S.view === "session" && S.ep === ep) redrawStep(); };
      img.src = "/frames/" + encodeURIComponent(S.id) + "/" + st.frame;
      S.imgs[st.frame] = img;
    });
  }

  // ---------- top bar ---------------------------------------------------
  function renderTopbar() {
    var c = document.getElementById("topbar-controls");
    c.innerHTML = "";
    if (S.sessions.length > 1) {
      var sel = el("select");
      S.sessions.forEach(function (s) {
        var o = el("option", null, s.session_name + "  (" + s.n_episodes + " ep)");
        o.value = s.id; if (s.id === S.id) o.selected = true; sel.appendChild(o);
      });
      sel.onchange = function () { location.hash = "#/s/" + encodeURIComponent(sel.value); };
      c.appendChild(sel);
    }
    if (S.view === "session") {
      var tg = el("label", "toggle");
      var cb = el("input"); cb.type = "checkbox"; cb.checked = S.showGT;
      cb.onchange = function () { S.showGT = cb.checked; redrawStep(); };
      tg.appendChild(cb); tg.appendChild(el("span", null, "Ground truth overlay"));
      c.appendChild(tg);
      var home = el("button", null, "◂ Scenarios");
      home.onclick = function () { S.view = "home"; stopPlay(); writeHash(); render(); };
      c.appendChild(home);
    }
  }

  // ---------- legend fragment ------------------------------------------
  function legendEl(includeGT, includeEgo) {
    var lg = el("div", "legend");
    armNames().forEach(function (n) {
      var chip = el("span", "chip");
      var dot = el("span", "dot"); dot.style.background = armColor(n);
      chip.appendChild(dot); chip.appendChild(el("span", null, n.toUpperCase()));
      lg.appendChild(chip);
    });
    if (includeGT) {
      var g = el("span", "chip");
      g.appendChild(el("span", "dash")); g.appendChild(el("span", null, "GT"));
      lg.appendChild(g);
    }
    if (includeEgo) {
      var e = el("span", "chip");
      var d = el("span", "dot"); d.style.background = EGO_DIM;
      e.appendChild(d); e.appendChild(el("span", null, "ego"));
      lg.appendChild(e);
    }
    return lg;
  }

  // ---------- HOME view -------------------------------------------------
  function render() {
    renderTopbar();
    writeHash();
    if (S.view === "session") renderSession();
    else renderHome();
  }

  function renderHome() {
    S.dom = null;
    var app = document.getElementById("app");
    app.innerHTML = "";
    var meta = S.sess.meta;

    // summary strip
    var sum = el("div", "summary");
    var left = el("div");
    left.appendChild(el("h1", null, meta.session_name));
    left.appendChild(el("div", "sub",
      meta.corpora.join(" · ") + " · " + meta.episodes.length + " episodes · " +
      totalSteps() + " steps"));
    sum.appendChild(left);
    sum.appendChild(legendEl(true, false));
    var stats = el("div", "stats");
    meta.arms.forEach(function (a) {
      var st = el("div", "stat");
      st.appendChild(el("div", "k", a.name.toUpperCase() + " ADE"));
      var v = el("div", "v", fmt(a.ade, 2) + " m"); v.style.color = a.color;
      st.appendChild(v);
      st.appendChild(el("div", "k", "p50 " + fmt(a.latency_p50, 1) + " ms"));
      stats.appendChild(st);
    });
    sum.appendChild(stats);
    app.appendChild(sum);

    // cards
    var maxAde = Math.max(0.01, Math.max.apply(null, meta.episodes.map(function (e) {
      return Math.max.apply(null, Object.values(e.per_arm_ade).concat([0]));
    })));
    var grid = el("div", "cards");
    meta.episodes.forEach(function (e) {
      grid.appendChild(scenarioCard(e, maxAde));
    });
    app.appendChild(grid);
  }

  function totalSteps() {
    return S.sess.meta.episodes.reduce(function (a, e) { return a + e.n_steps; }, 0);
  }

  function scenarioCard(e, maxAde) {
    var card = el("div", "card");
    var thumb = el("div", "thumb");
    var img = el("img"); img.loading = "lazy";
    img.src = "/frames/" + encodeURIComponent(S.id) + "/" + e.thumb;
    img.alt = "episode " + e.idx;
    thumb.appendChild(img);
    thumb.appendChild(el("span", "tag", e.corpus_tag));
    thumb.appendChild(el("span", "worst", "worst " + fmt(e.worst_ade, 2) + " m"));
    thumb.appendChild(el("span", "epno", "Episode " + e.idx));
    card.appendChild(thumb);

    var body = el("div", "body");
    armNames().forEach(function (n) {
      var ade = e.per_arm_ade[n];
      var row = el("div", "barrow");
      row.appendChild(el("div", "nm", n.toUpperCase()));
      var track = el("div", "track");
      var fill = el("div", "fill");
      fill.style.width = (ade == null ? 0 : clamp(ade / maxAde, 0, 1) * 100) + "%";
      fill.style.background = armColor(n);
      track.appendChild(fill); row.appendChild(track);
      row.appendChild(el("div", "val", ade == null ? "–" : fmt(ade, 2)));
      body.appendChild(row);
    });
    card.appendChild(body);
    card.onclick = function () {
      location.hash = "#/s/" + encodeURIComponent(S.id) + "/e/" + e.idx + "/t/0";
    };
    return card;
  }

  // ---------- SESSION view (build once, update per step) ----------------
  function renderSession() {
    if (!S.dom || S.dom.ep !== S.ep) buildSessionDOM();
    updateSession();
  }

  function buildSessionDOM() {
    stopPlay();
    preloadEpisode(S.ep);
    S.bevRange[S.ep] = S.bevRange[S.ep] || computeBevRange(curEpisode());
    var app = document.getElementById("app");
    app.innerHTML = "";
    var dom = { ep: S.ep, cams: {}, charts: {}, heads: {}, headHost: {} };

    // header
    var head = el("div", "session-head");
    var title = el("div", null, S.sess.meta.session_name);
    title.style.fontWeight = 700; title.style.fontSize = "17px";
    head.appendChild(title);
    var nav = el("div", "epnav");
    var prev = el("button", null, "◂ ep");
    prev.onclick = function () { gotoEp(S.ep - 1); };
    var next = el("button", null, "ep ▸");
    next.onclick = function () { gotoEp(S.ep + 1); };
    dom.epind = el("span", "epind");
    nav.appendChild(prev); nav.appendChild(dom.epind); nav.appendChild(next);
    head.appendChild(nav);
    head.appendChild(legendEl(true, false));
    app.appendChild(head);

    // arm columns
    var cols = el("div", "arm-cols");
    armNames().forEach(function (n) {
      cols.appendChild(buildArmColumn(n, dom));
    });
    app.appendChild(cols);

    // master panel
    app.appendChild(buildMaster(dom));

    S.dom = dom;
  }

  function buildArmColumn(name, dom) {
    var color = armColor(name);
    var col = el("div", "arm-col");
    col.style.setProperty("--col", color);

    var hd = el("div", "hd");
    var dot = el("span", "dot"); dot.style.background = color;
    hd.appendChild(dot);
    hd.appendChild(el("span", "nm", name));
    var metrics = el("div", "metrics");
    var mAde = el("div", "m"); mAde.appendChild(el("div", "k", "ADE"));
    var vAde = el("div", "v"); vAde.style.color = color;
    var epMeta = S.sess.meta.episodes[S.ep].per_arm_ade[name];
    vAde.textContent = fmt(epMeta, 2) + " m"; mAde.appendChild(vAde);
    var armMeta = S.sess.meta.arms.find(function (a) { return a.name === name; });
    var mLat = el("div", "m"); mLat.appendChild(el("div", "k", "p50"));
    var vLat = el("div", "v", fmt(armMeta.latency_p50, 1) + " ms"); mLat.appendChild(vLat);
    metrics.appendChild(mAde); metrics.appendChild(mLat);
    hd.appendChild(metrics);
    col.appendChild(hd);

    // camera
    col.appendChild(el("div", "cap", "Camera + trajectory fan"));
    var cw = el("div", "canvas-wrap");
    var cam = el("canvas");
    cw.appendChild(cam); col.appendChild(cw);
    var ml = el("div", "mini-legend");
    var lg = el("div", "legend");
    var chip = el("span", "chip"); var cd = el("span", "dot"); cd.style.background = color;
    chip.appendChild(cd); chip.appendChild(el("span", null, name.toUpperCase() + " fan"));
    lg.appendChild(chip);
    var g = el("span", "chip"); g.appendChild(el("span", "dash")); g.appendChild(el("span", null, "GT"));
    lg.appendChild(g); ml.appendChild(lg); col.appendChild(ml);
    dom.cams[name] = cam;

    // charts
    col.appendChild(el("div", "cap", "Control readout vs GT"));
    var c2 = el("div", "charts2");
    dom.charts[name] = {};
    ["steer", "accel"].forEach(function (kind) {
      var box = el("div", "chart-box");
      var lbl = el("div", "lbl");
      var title = el("span", null, kind + (kind === "steer" ? " (rad)" : " (m/s²)"));
      var leg = el("span");
      leg.innerHTML = '<span style="color:' + color + '">●</span> ' + name +
        ' &nbsp;<span style="color:' + GT + '">⋯</span> GT';
      leg.style.fontSize = "9.5px"; leg.style.color = "var(--ink-faint)";
      lbl.appendChild(title); lbl.appendChild(leg);
      box.appendChild(lbl);
      var cv = el("canvas"); box.appendChild(cv);
      c2.appendChild(box);
      dom.charts[name][kind] = cv;
    });
    col.appendChild(c2);

    // heads
    col.appendChild(el("div", "cap", "Head readouts"));
    var host = el("div", "heads");
    col.appendChild(host);
    dom.headHost[name] = host;
    return col;
  }

  function buildMaster(dom) {
    var m = el("div", "master");

    var bevSide = el("div", "bev-side");
    var bt = el("div", "panel-title");
    bt.appendChild(el("span", null, "BEV — all arms (metres, forward = up)"));
    bt.appendChild(legendEl(true, true));
    bevSide.appendChild(bt);
    var bev = el("canvas"); bevSide.appendChild(bev);
    dom.bev = bev;
    m.appendChild(bevSide);

    var errSide = el("div", "err-side");
    var et = el("div", "panel-title");
    et.appendChild(el("span", null, "Per-step waypoint ADE — error strip"));
    et.appendChild(legendEl(false, false));
    errSide.appendChild(et);
    var err = el("canvas"); errSide.appendChild(err);
    dom.err = err;
    err.onclick = function (ev) {
      var r = err.getBoundingClientRect();
      var n = curEpisode().steps.length;
      var i = Math.round((ev.clientX - r.left) / r.width * (n - 1));
      seek(clamp(i, 0, n - 1));
    };
    errSide.appendChild(el("div", "err-hint", "Click the strip to seek to a spike."));
    m.appendChild(errSide);

    var scr = el("div", "scrubber");
    dom.play = el("button", "playbtn", "▶");
    dom.play.onclick = togglePlay;
    scr.appendChild(dom.play);
    dom.range = el("input"); dom.range.type = "range"; dom.range.min = 0;
    dom.range.max = curEpisode().steps.length - 1; dom.range.step = 1;
    dom.range.oninput = function () { seek(+dom.range.value); };
    scr.appendChild(dom.range);
    dom.ind = el("span", "ind");
    scr.appendChild(dom.ind);
    m.appendChild(scr);
    return m;
  }

  // ---------- per-step update ------------------------------------------
  function updateSession() {
    var dom = S.dom;
    dom.epind.textContent = "Episode " + curEpisode().idx + " / " +
      (S.sess.episodes.length - 1) + "  ·  " + curEpisode().corpus_tag;
    dom.range.value = S.step;
    var st = curStep();
    dom.ind.textContent = "step " + (S.step + 1) + "/" + curEpisode().steps.length +
      "  (global " + st.step + ")   v=" + fmt(st.ego.speed, 1) + " m/s  ψ̇=" +
      fmt(st.ego.yaw_rate, 2) + " rad/s";
    redrawStep();
  }

  function redrawStep() {
    if (!S.dom || S.view !== "session") return;
    var st = curStep();
    armNames().forEach(function (n) {
      drawCamera(S.dom.cams[n], st, n);
      drawChart(S.dom.charts[n].steer, n, "steer");
      drawChart(S.dom.charts[n].accel, n, "accel");
      renderHeads(S.dom.headHost[n], n, st.arms[n]);
    });
    drawBEV(S.dom.bev, st);
    drawErrorStrip(S.dom.err);
  }

  // ---------- canvas helpers -------------------------------------------
  function fit(canvas, cssH) {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth || canvas.parentElement.clientWidth || 300;
    canvas.style.height = cssH + "px";
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(cssH * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, cssH);
    return { ctx: ctx, w: w, h: cssH };
  }

  function polyline(ctx, pts, color, width, dashed) {
    if (!pts || pts.length < 2) return;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (dashed) ctx.setLineDash([6, 5]);
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.restore();
  }

  // ---------- camera ----------------------------------------------------
  function drawCamera(canvas, st, name) {
    var img = S.imgs[st.frame];
    var natW = (img && img.naturalWidth) || 256;
    var natH = (img && img.naturalHeight) || 192;
    var cw = canvas.clientWidth || 300;
    var cssH = Math.round(cw * natH / natW);
    var f = fit(canvas, cssH);
    var ctx = f.ctx;
    var sx = f.w / natW, sy = f.h / natH;
    if (img && img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, f.w, f.h);
    else { ctx.fillStyle = "#0a0f18"; ctx.fillRect(0, 0, f.w, f.h); }

    function toPx(p) { return [p[0] * sx, p[1] * sy]; }
    var arm = st.arms[name];
    if (S.showGT && st.gt_wp_img) {
      var gp = st.gt_wp_img.map(toPx);
      polyline(ctx, gp, GT, 2, true);
      dot(ctx, gp[gp.length - 1], GT, 3);
    }
    if (arm && arm.wp_img) {
      var ap = arm.wp_img.map(toPx);
      polyline(ctx, ap, armColor(name), 2.6, false);
      ap.slice(1).forEach(function (p) { dot(ctx, p, armColor(name), 3); });
    }
  }
  function dot(ctx, p, color, r) {
    ctx.save(); ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 6.2832); ctx.fill();
    ctx.restore();
  }

  // ---------- steer/accel charts ---------------------------------------
  function drawChart(canvas, name, kind) {
    var steps = curEpisode().steps;
    var arm = steps.map(function (s) { return s.arms[name] ? s.arms[name][kind] : null; });
    var gt = steps.map(function (s) { return s.gt_action[kind]; });
    var all = arm.concat(gt).filter(function (v) { return v != null && !isNaN(v); });
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    if (!isFinite(lo)) { lo = -1; hi = 1; }
    var pad = (hi - lo) * 0.15 || 0.1; lo -= pad; hi += pad;
    var f = fit(canvas, 74), ctx = f.ctx;
    var mL = 4, mR = 4, mT = 4, mB = 10;
    var pw = f.w - mL - mR, ph = f.h - mT - mB;
    var n = steps.length;
    function X(i) { return mL + (n <= 1 ? 0 : i / (n - 1) * pw); }
    function Y(v) { return mT + (hi === lo ? ph / 2 : (hi - v) / (hi - lo) * ph); }
    // zero line
    if (lo < 0 && hi > 0) {
      ctx.strokeStyle = "#ffffff18"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(mL, Y(0)); ctx.lineTo(mL + pw, Y(0)); ctx.stroke();
    }
    // current marker
    ctx.strokeStyle = "#ffffff30"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(S.step), mT); ctx.lineTo(X(S.step), mT + ph); ctx.stroke();
    // series
    polyline(ctx, gt.map(function (v, i) { return [X(i), Y(v)]; }), GT, 1.4, true);
    polyline(ctx, arm.map(function (v, i) { return v == null ? null : [X(i), Y(v)]; })
      .filter(Boolean), armColor(name), 1.8, false);
    // current dots
    if (arm[S.step] != null) dot(ctx, [X(S.step), Y(arm[S.step])], armColor(name), 2.4);
    dot(ctx, [X(S.step), Y(gt[S.step])], GT, 2.2);
    // y range labels
    ctx.fillStyle = "#63728c"; ctx.font = "9px monospace";
    ctx.fillText(hi.toFixed(2), mL + 1, mT + 8);
    ctx.fillText(lo.toFixed(2), mL + 1, mT + ph - 1);
  }

  // ---------- BEV -------------------------------------------------------
  function computeBevRange(ep) {
    var fwd = 20, lat = 8;
    ep.steps.forEach(function (s) {
      var paths = [s.gt_wp_bev];
      Object.keys(s.arms).forEach(function (n) { if (s.arms[n].wp_bev) paths.push(s.arms[n].wp_bev); });
      paths.forEach(function (p) {
        p.forEach(function (xy) {
          fwd = Math.max(fwd, xy[0]); lat = Math.max(lat, Math.abs(xy[1]));
        });
      });
    });
    return { fwd: Math.ceil((fwd + 2) / 5) * 5, lat: Math.ceil((lat + 1) / 5) * 5 };
  }

  function drawBEV(canvas, st) {
    var f = fit(canvas, 320), ctx = f.ctx;
    var R = S.bevRange[S.ep];
    var mL = 34, mR = 10, mT = 10, mB = 22;
    var pw = f.w - mL - mR, ph = f.h - mT - mB;
    var oX = mL + pw / 2, oY = mT + ph;
    var scale = Math.min(pw / (2 * R.lat), ph / R.fwd);
    function P(x, y) { return [oX - y * scale, oY - x * scale]; }

    // grid — forward lines
    ctx.font = "9px monospace"; ctx.textBaseline = "middle";
    for (var d = 0; d <= R.fwd; d += 5) {
      var y = oY - d * scale;
      ctx.strokeStyle = d === 0 ? "#ffffff30" : "#ffffff12"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(mL, y); ctx.lineTo(mL + pw, y); ctx.stroke();
      ctx.fillStyle = "#63728c"; ctx.textAlign = "right";
      ctx.fillText(d + " m", mL - 4, y);
    }
    // grid — lateral lines
    ctx.textBaseline = "top"; ctx.textAlign = "center";
    for (var lx = -R.lat; lx <= R.lat; lx += 5) {
      var x = oX - lx * scale;
      ctx.strokeStyle = lx === 0 ? "#ffffff30" : "#ffffff12"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, mT); ctx.lineTo(x, mT + ph); ctx.stroke();
      ctx.fillStyle = "#63728c";
      ctx.fillText((lx > 0 ? "+" : "") + lx, x, mT + ph + 4);
    }
    // axis captions
    ctx.fillStyle = "#97a6bd"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText("lateral (m, + = left)", oX, mT + ph + 12);

    // scale bar (5 m)
    var barLen = 5 * scale, bx = mL + pw - barLen - 6, by = mT + 10;
    ctx.strokeStyle = "#eef2f7"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + barLen, by); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(bx, by - 3); ctx.lineTo(bx, by + 3);
    ctx.moveTo(bx + barLen, by - 3); ctx.lineTo(bx + barLen, by + 3); ctx.stroke();
    ctx.fillStyle = "#eef2f7"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.font = "9px monospace"; ctx.fillText("5 m", bx + barLen / 2, by - 4);

    // paths
    if (S.showGT && st.gt_wp_bev) {
      polyline(ctx, st.gt_wp_bev.map(function (p) { return P(p[0], p[1]); }), GT, 2, true);
    }
    armNames().forEach(function (n) {
      var a = st.arms[n]; if (!a || !a.wp_bev) return;
      var pts = a.wp_bev.map(function (p) { return P(p[0], p[1]); });
      polyline(ctx, pts, armColor(n), 2.4, false);
      dot(ctx, pts[pts.length - 1], armColor(n), 3);
    });
    // ego marker
    var e = P(0, 0);
    ctx.fillStyle = EGO_DIM; ctx.strokeStyle = "#0a0f18"; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(e[0], e[1] - 6); ctx.lineTo(e[0] - 4, e[1] + 4);
    ctx.lineTo(e[0] + 4, e[1] + 4); ctx.closePath(); ctx.fill(); ctx.stroke();

    // legend box
    bevLegend(ctx, f.w - mR - 96, mT + 6);
  }

  function bevLegend(ctx, x, y) {
    ctx.save();
    var rows = armNames().map(function (n) { return { c: armColor(n), t: n.toUpperCase(), dash: false }; });
    rows.push({ c: GT, t: "GT", dash: true });
    var w = 90, h = rows.length * 15 + 8;
    ctx.fillStyle = "#0e1420cc"; ctx.strokeStyle = "#ffffff20";
    roundRect(ctx, x, y, w, h, 6); ctx.fill(); ctx.stroke();
    ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.font = "10px system-ui, sans-serif";
    rows.forEach(function (r, i) {
      var ry = y + 12 + i * 15;
      ctx.strokeStyle = r.c; ctx.lineWidth = 2.4;
      if (r.dash) ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(x + 8, ry); ctx.lineTo(x + 24, ry); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#c7d2e2"; ctx.fillText(r.t, x + 30, ry);
    });
    ctx.restore();
  }
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  // ---------- error strip ----------------------------------------------
  function drawErrorStrip(canvas) {
    var steps = curEpisode().steps;
    var f = fit(canvas, 150), ctx = f.ctx;
    var mL = 32, mR = 8, mT = 8, mB = 16;
    var pw = f.w - mL - mR, ph = f.h - mT - mB;
    var n = steps.length;
    var hi = 0.1;
    armNames().forEach(function (nm) {
      steps.forEach(function (s) { var a = s.arms[nm]; if (a && a.ade != null) hi = Math.max(hi, a.ade); });
    });
    hi = Math.ceil(hi * 1.1 * 10) / 10;
    function X(i) { return mL + (n <= 1 ? 0 : i / (n - 1) * pw); }
    function Y(v) { return mT + (hi <= 0 ? ph : (hi - v) / hi * ph); }
    // grid + y labels
    ctx.font = "9px monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (var g = 0; g <= 3; g++) {
      var v = hi * g / 3, y = Y(v);
      ctx.strokeStyle = "#ffffff12"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(mL, y); ctx.lineTo(mL + pw, y); ctx.stroke();
      ctx.fillStyle = "#63728c"; ctx.fillText(v.toFixed(1), mL - 4, y);
    }
    ctx.save(); ctx.translate(9, mT + ph / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = "#97a6bd"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.font = "9.5px system-ui, sans-serif"; ctx.fillText("ADE (m)", 0, 0); ctx.restore();
    // current marker
    ctx.strokeStyle = "#f5b30188"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(S.step), mT); ctx.lineTo(X(S.step), mT + ph); ctx.stroke();
    // lines
    armNames().forEach(function (nm) {
      var pts = steps.map(function (s, i) {
        var a = s.arms[nm]; return (a && a.ade != null) ? [X(i), Y(a.ade)] : null;
      }).filter(Boolean);
      polyline(ctx, pts, armColor(nm), 1.8, false);
    });
    // current dots
    armNames().forEach(function (nm) {
      var a = steps[S.step].arms[nm];
      if (a && a.ade != null) dot(ctx, [X(S.step), Y(a.ade)], armColor(nm), 2.6);
    });
  }

  // ---------- head readouts --------------------------------------------
  function renderHeads(host, name, arm) {
    host.innerHTML = "";
    var col = armColor(name);
    var h = arm ? arm.heads : null;
    if (!h || Object.keys(h).length === 0) {
      host.appendChild(el("div", "none", "No monitor heads for this arm.")); return;
    }
    if (h.imag_rel) {
      host.appendChild(el("div", "sub", "imagination error (imag_rel, per horizon)"));
      Object.keys(h.imag_rel).sort(function (a, b) { return +a - +b; }).forEach(function (k) {
        host.appendChild(bar("k" + k, h.imag_rel[k], 2.0, col, fmt(h.imag_rel[k], 2)));
      });
    }
    if (h.sigma != null) {
      host.appendChild(el("div", "sub", "belief σ (H15)"));
      host.appendChild(bar("sigma", h.sigma, Math.max(1, h.sigma * 1.3), col, fmt(h.sigma, 3)));
    }
    if (h.conf != null || h.ood != null) {
      host.appendChild(el("div", "sub", "confidence / OOD"));
      if (h.conf != null) host.appendChild(bar("conf", h.conf, Math.max(1, h.conf * 1.3), "#7fd18b", fmt(h.conf, 3)));
      if (h.ood != null) host.appendChild(bar("ood", h.ood, Math.max(1, h.ood * 1.3), "#e88f6a", fmt(h.ood, 3)));
    }
    if (h.maneuver_probs) {
      var names = (S.sess.meta.maneuver_classes) ||
        h.maneuver_probs.map(function (_, i) { return "m" + i; });
      var gt = h.maneuver_gt;
      var argmax = h.maneuver_probs.indexOf(Math.max.apply(null, h.maneuver_probs));
      host.appendChild(el("div", "sub", "maneuver distribution" +
        (gt != null ? "  (GT: " + (names[gt] || gt) + ")" : "")));
      var man = el("div", "man");
      h.maneuver_probs.forEach(function (p, i) {
        var label = (names[i] || ("m" + i)) + (i === gt ? " ✓" : "");
        var color = i === argmax ? "#f5b301" : "#5b6b86";
        var row = bar(label, p, 1.0, color, (p * 100).toFixed(0) + "%");
        if (i === gt) row.querySelector(".hk").classList.add("gtmark");
        man.appendChild(row);
      });
      host.appendChild(man);
    }
    if (h.nav_cmd != null) {
      var navNames = ["straight", "left", "right"];
      host.appendChild(el("div", "sub", "nav command: " + (navNames[h.nav_cmd] || h.nav_cmd)));
    }
  }
  function bar(k, v, max, color, valText) {
    var row = el("div", "hrow");
    row.appendChild(el("div", "hk", k));
    var track = el("div", "htrack");
    var fill = el("div", "hfill");
    fill.style.width = clamp((v || 0) / (max || 1), 0, 1) * 100 + "%";
    fill.style.background = color;
    track.appendChild(fill); row.appendChild(track);
    row.appendChild(el("div", "hv", valText));
    return row;
  }

  // ---------- navigation / playback ------------------------------------
  function seek(i) {
    S.step = clamp(i, 0, curEpisode().steps.length - 1);
    writeHash(); updateSession();
  }
  function gotoEp(i) {
    i = clamp(i, 0, S.sess.episodes.length - 1);
    if (i === S.ep) return;
    stopPlay(); S.ep = i; S.step = 0; writeHash();
    buildSessionDOM(); updateSession();
  }
  function togglePlay() { S.playing ? stopPlay() : startPlay(); }
  function startPlay() {
    S.playing = true; if (S.dom) S.dom.play.textContent = "⏸";
    S.timer = setInterval(function () {
      if (S.step >= curEpisode().steps.length - 1) { stopPlay(); return; }
      seek(S.step + 1);
    }, 100);
  }
  function stopPlay() {
    S.playing = false; if (S.timer) clearInterval(S.timer); S.timer = null;
    if (S.dom && S.dom.play) S.dom.play.textContent = "▶";
  }
  function onKey(e) {
    if (S.view !== "session") return;
    if (e.key === "ArrowRight") { seek(S.step + 1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { seek(S.step - 1); e.preventDefault(); }
    else if (e.key === "ArrowUp") { gotoEp(S.ep + 1); e.preventDefault(); }
    else if (e.key === "ArrowDown") { gotoEp(S.ep - 1); e.preventDefault(); }
    else if (e.key === " ") { togglePlay(); e.preventDefault(); }
  }

  window.addEventListener("resize", function () { if (S.view === "session") redrawStep(); });
  boot();
})();
