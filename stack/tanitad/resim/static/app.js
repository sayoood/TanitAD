/* TanitResim SPA — vanilla JS + canvas, no build step.
 *
 * Two views: (1) HOME = scenario cards, one per episode of the selected
 * session; (2) SESSION = one column per arm (camera fan + a decoded-intent
 * text HUD + steer/accel charts + head readouts) over a shared master panel:
 * BEV, error strip, a MANEUVER band (per-window kinematic class: badge + color
 * timeline), an ACTION panel (executed GT steer/accel: gauges + synced
 * time-series) and a scrubber. The camera HUD mirrors THE STANDARD
 * (taniteval.corpus_overlay): each arm's decoded tactical maneuver + strategic
 * route/goal + ADE + v0, with a BEV-only fallback note when a step's camera
 * calibration is unrecoverable. Everything is legended. URL hash carries
 * session/episode/step so a view is shareable: #/s/<id>/e/<ep>/t/<step>. */
(function () {
  "use strict";

  var GT = "#eef2f7";
  var EGO_DIM = "#63728c";

  // Maneuver-class palette, keyed by canonical refb_labels class name (see
  // scripts/refb_labels.py MANEUVER_CLASSES). Distinct hues inside the TanitAD
  // language; an unknown/absent class -> neutral slate.
  var MAN_COLORS = {
    lane_keep: "#5b83b0",     // steady slate-blue
    turn_left: "#22d3ee",     // cyan
    turn_right: "#e35ce0",    // magenta
    accelerate: "#63d29a",    // green
    brake_stop: "#f2765f",    // red-orange
  };
  var MAN_NEUTRAL = "#3a4a66";
  var MAN_LABELS = {          // compact badge labels
    lane_keep: "LANE KEEP", turn_left: "TURN L", turn_right: "TURN R",
    accelerate: "ACCEL", brake_stop: "BRAKE / STOP",
  };
  // Executed-action signal palette (GT control) — kept clear of arm colors.
  var ACT_COLORS = { steer: "#9db8ff", accel: "#ffc073" };
  var ACT_UNITS = { steer: "rad", accel: "m/s²" };

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

  // ---------- maneuver helpers ------------------------------------------
  function maneuverClasses() {
    return (S.sess && S.sess.meta.maneuver_classes) || null;
  }
  function maneuverName(id) {
    if (id == null) return null;
    var c = maneuverClasses();
    return (c && c[id] != null) ? c[id] : ("m" + id);
  }
  function maneuverColor(id) {
    if (id == null) return MAN_NEUTRAL;
    return MAN_COLORS[maneuverName(id)] || MAN_NEUTRAL;
  }
  function maneuverLabel(id) {
    var nm = maneuverName(id);
    if (nm == null) return "NO DATA";
    return MAN_LABELS[nm] || nm.replace(/_/g, " ").toUpperCase();
  }
  function episodeHasManeuvers() {
    return curEpisode().steps.some(function (s) { return s.maneuver != null; });
  }

  // ---------- decoded-intent helpers (THE STANDARD's text HUD) -----------
  // Strategic route/goal command names, indexed by ArmOutput.nav_cmd. Data-
  // driven from the bundle (export.NAV_COMMANDS) so the label is correct; the
  // canonical tuple is the fallback for older bundles (the previous hard-coded
  // ["straight","left","right"] mislabelled follow/straight).
  function navCommands() {
    return (S.sess && S.sess.meta.nav_commands) ||
      ["follow", "left", "right", "straight"];
  }
  function navName(id) {
    if (id == null) return null;
    var c = navCommands();
    return (c && c[id] != null) ? c[id] : ("cmd" + id);
  }
  // The arm's DECODED tactical maneuver = argmax of its maneuver_probs head
  // (the same intent taniteval's HUD prints as "tactical: <man>"); null if the
  // arm has no maneuver head.
  function decodedManeuver(arm) {
    var p = arm && arm.heads && arm.heads.maneuver_probs;
    if (!p || !p.length) return null;
    var mi = 0;
    for (var i = 1; i < p.length; i++) if (p[i] > p[mi]) mi = i;
    return mi;
  }
  // A step whose camera geometry was not recoverable: the exporter nulls its
  // image-plane paths (uncalibrated corpus) -> BEV-only fallback.
  function stepIsUncalibrated(st) {
    return !!st && st.gt_wp_img == null;
  }
  // Dark or light ink for readable text on a solid maneuver-color badge.
  function contrastInk(hex) {
    var m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
    if (!m) return GT;
    var n = parseInt(m[1], 16);
    var lum = (0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) +
      0.114 * (n & 255)) / 255;
    return lum > 0.55 ? "#0a0f18" : GT;
  }
  // A padded value range that always includes zero (signed action signals).
  function niceRange(vals) {
    var v = vals.filter(function (x) { return x != null && !isNaN(x); });
    var lo = v.length ? Math.min.apply(null, v) : -1;
    var hi = v.length ? Math.max.apply(null, v) : 1;
    lo = Math.min(lo, 0); hi = Math.max(hi, 0);
    if (hi === lo) { hi += 1; lo -= 1; }
    var pad = (hi - lo) * 0.12 || 0.1;
    return { lo: lo - pad, hi: hi + pad };
  }

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
    // maneuver mix ribbon (episode's kinematic-class distribution)
    var mc = e.maneuver_counts;
    if (mc && Object.keys(mc).length) {
      var total = Object.keys(mc).reduce(function (a, k) { return a + mc[k]; }, 0);
      body.appendChild(el("div", "man-ribbon-cap", "maneuver mix"));
      var ribbon = el("div", "man-ribbon");
      Object.keys(mc).forEach(function (k) {
        var seg = el("div", "seg");
        seg.style.width = (total ? mc[k] / total * 100 : 0) + "%";
        seg.style.background = MAN_COLORS[k] || MAN_NEUTRAL;
        seg.title = k.replace(/_/g, " ") + ": " + mc[k];
        ribbon.appendChild(seg);
      });
      body.appendChild(ribbon);
    }
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

    // Phase-0 GO verdict banner (shared formal-gate suite)
    if (S.sess.meta.gates) app.appendChild(buildGoBanner(S.sess.meta.gates));

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

  // --- formal-gate UI (D1-D3 + Phase-0 GO verdict) ------------------------
  function gateBadge(label, status) {
    var s = (status || "N/A").toString().toUpperCase();
    var cls = "gate-badge gate-" +
      (s === "PASS" ? "pass" : s === "FAIL" ? "fail" :
        s === "BLOCKED" ? "blocked" : "na");
    var b = el("span", cls);
    b.appendChild(el("span", "gb-k", label));
    b.appendChild(el("span", "gb-v", s));
    return b;
  }

  function gateNum(k, v) {
    var m = el("div", "m");
    m.appendChild(el("div", "k", k));
    m.appendChild(el("div", "v", v));
    return m;
  }

  function buildGatesPanel(name, gm) {
    var box = el("div", "gates-panel");
    box.appendChild(el("div", "cap", "Formal gates (D1–D3, necessary-not-sufficient)"));
    var badges = el("div", "gate-badges");
    badges.appendChild(gateBadge("D1", gm.D1));
    badges.appendChild(gateBadge("D2", gm.D2));
    badges.appendChild(gateBadge("D3", gm.D3));
    box.appendChild(badges);
    var nums = el("div", "metrics gate-metrics");
    nums.appendChild(gateNum("D1 ADE", fmt(gm.d1_ade_0_2s, 3) + " m"));
    nums.appendChild(gateNum("oracle", fmt(gm.oracle_ceiling_ade_0_2s, 3) + " m"));
    if (gm.grounded_ade_0_2s != null)
      nums.appendChild(gateNum("grounded", fmt(gm.grounded_ade_0_2s, 3) + " m"));
    if (gm.d2_dir_acc != null)
      nums.appendChild(gateNum("D2 dir-acc", fmt(gm.d2_dir_acc, 2)));
    if (gm.maneuver_balacc != null)
      nums.appendChild(gateNum("maneuver", fmt(gm.maneuver_balacc, 2) + " bal-acc"));
    if (gm.route_balacc != null)
      nums.appendChild(gateNum("route", fmt(gm.route_balacc, 2) + " bal-acc"));
    box.appendChild(nums);
    return box;
  }

  function buildGoBanner(gates) {
    var box = el("div", "go-banner");
    var v = gates.verdict || {};
    var pm = (v.per_metric || {}).d1_decode_ade_0_2s || {};
    var winner = pm.winner_lowest || "—";
    var bl = gates.baselines || {};
    var txt = "Phase-0 gates — CV floor " +
      fmt(bl.constant_velocity, 2) + " m · D1 decode winner: " + winner;
    var edge = v.hierarchy_edge_necessary_conditions;
    if (edge) {
      txt += " · flagship beats refs (D1): " +
        edge.flagship_beats_refs_on_d1_decode +
        " · grounded>CV: " + edge.flagship_grounded_beats_cv_floor;
    }
    var main = el("div", "go-main", txt);
    box.appendChild(main);
    var sub = el("div", "go-sub",
      "Necessary, not sufficient — closed-loop D4–D6 arbitrate (see phase0_go_criteria.md). " +
      "Gated on " + (gates.n_val_episodes || "?") + " val episodes / " +
      (gates.n_windows || "?") + " windows.");
    box.appendChild(sub);
    return box;
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

    // formal gates (D1-D3 + oracle ceiling) — the shared compare_arms suite
    if (armMeta.gates) col.appendChild(buildGatesPanel(name, armMeta.gates));

    // camera
    col.appendChild(el("div", "cap", "Camera + trajectory fan"));
    var cw = el("div", "canvas-wrap");
    var cam = el("canvas");
    cw.appendChild(cam);
    // decoded-intent text HUD overlaid top-left on the frame (THE STANDARD:
    // tactical maneuver + strategic route/goal + ADE + v0), populated per step.
    var hud = el("div", "cam-hud");
    cw.appendChild(hud);
    dom.camHud = dom.camHud || {};
    dom.camHud[name] = hud;
    col.appendChild(cw);
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

    // maneuver band (only when the episode carries kinematic labels) + the
    // executed-action panel — both full-width rows below the BEV/error row.
    if (episodeHasManeuvers()) m.appendChild(buildManeuverBand(dom));
    m.appendChild(buildActionPanel(dom));

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
      updateCamHud(n, st);
      drawChart(S.dom.charts[n].steer, n, "steer");
      drawChart(S.dom.charts[n].accel, n, "accel");
      renderHeads(S.dom.headHost[n], n, st.arms[n]);
    });
    drawBEV(S.dom.bev, st);
    drawErrorStrip(S.dom.err);
    drawManeuver(S.dom);
    drawAction(S.dom);
  }

  // ---------- canvas helpers -------------------------------------------
  // `cssW` (optional) pins the CSS width explicitly. The camera passes the
  // wrap's width so sizing never reads the canvas's OWN transient clientWidth
  // (which, feeding back into its height, produced the stretched-frame bug).
  function fit(canvas, cssH, cssW) {
    var dpr = window.devicePixelRatio || 1;
    var w = cssW || canvas.clientWidth || canvas.parentElement.clientWidth || 300;
    if (cssW) canvas.style.width = cssW + "px";   // pin width (else CSS 100%)
    canvas.style.height = cssH + "px";
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(cssH * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, cssH);
    return { ctx: ctx, w: w, h: cssH };
  }

  // Contain-fit a frame of (fw x fh) inside a (cw x ch) panel: the frame keeps
  // its TRUE aspect and is centered, with dark letterbox/pillarbox bars filling
  // the remainder — never stretched. Overlays reuse the returned rect so the
  // trajectory fan stays locked to the frame pixels it was projected against.
  function containRect(cw, ch, fw, fh) {
    var scale = Math.min(cw / fw, ch / fh);
    var dw = fw * scale, dh = fh * scale;
    return { scale: scale, ox: (cw - dw) / 2, oy: (ch - dh) / 2, dw: dw, dh: dh };
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
  // Camera panel side cap: a single-arm session gives a very wide column, so
  // cap the (square) panel and center it rather than letting it balloon;
  // narrower multi-arm columns just use their full width.
  var CAM_MAX = 460;

  function drawCamera(canvas, st, name) {
    var img = S.imgs[st.frame];
    var loaded = img && img.complete && img.naturalWidth;
    // Frame's TRUE pixel size = the space wp_img was projected in (export uses
    // the written JPEG's w,h). Default square 256 (our frames) until it loads.
    var fw = loaded ? img.naturalWidth : 256;
    var fh = loaded ? img.naturalHeight : 256;
    // Panel side from the WRAP width (stable, layout-driven) — NOT the canvas's
    // own clientWidth, whose transient value used to feed back into its height
    // and stretch the frame. Square panel, centered in the column.
    var wrapW = canvas.parentElement.clientWidth || canvas.clientWidth || 300;
    var side = Math.min(wrapW, CAM_MAX);
    canvas.style.display = "block"; canvas.style.margin = "0 auto";
    var f = fit(canvas, side, side);
    var ctx = f.ctx;

    // Letterbox: dark backdrop, then the frame contain-fit at its true aspect.
    ctx.fillStyle = "#0a0f18"; ctx.fillRect(0, 0, f.w, f.h);
    var r = containRect(f.w, f.h, fw, fh);
    if (loaded) ctx.drawImage(img, r.ox, r.oy, r.dw, r.dh);

    // Overlay maps frame-pixel (u,v) through the SAME contain rect (equal x/y
    // scale => no axis distortion; forward points stay below the horizon).
    function toPx(p) { return [r.ox + p[0] * r.scale, r.oy + p[1] * r.scale]; }
    ctx.save();
    ctx.beginPath(); ctx.rect(r.ox, r.oy, r.dw, r.dh); ctx.clip();  // stay on frame
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
    ctx.restore();
  }
  function dot(ctx, p, color, r) {
    ctx.save(); ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 6.2832); ctx.fill();
    ctx.restore();
  }

  // ---------- decoded-intent HUD (per-arm camera text overlay) ----------
  // Mirrors taniteval.corpus_overlay's text HUD, per arm and co-located with
  // the camera: the arm's decoded TACTICAL maneuver + STRATEGIC route/goal +
  // ADE + v0, and the BEV-only fallback note on an uncalibrated step.
  function hudRow(k, valEl) {
    var row = el("div", "hud-intent");
    row.appendChild(el("span", "hud-k", k));
    row.appendChild(valEl);
    return row;
  }
  function updateCamHud(name, st) {
    var hud = S.dom.camHud && S.dom.camHud[name];
    if (!hud) return;
    hud.innerHTML = "";
    var arm = st.arms[name];
    var manId = decodedManeuver(arm);
    var heads = (arm && arm.heads) || {};

    if (manId != null) {                    // tactical: <decoded maneuver>
      var mv = el("span", "hud-v", maneuverLabel(manId));
      mv.style.color = maneuverColor(manId);
      var row = hudRow("tactical", mv);
      var gt = heads.maneuver_gt;
      if (gt != null && gt === manId) row.appendChild(el("span", "hud-ok", "✓"));
      else if (gt != null) row.appendChild(el("span", "hud-gt",
        "gt " + maneuverLabel(gt).toLowerCase()));
      hud.appendChild(row);
    }
    if (heads.nav_cmd != null) {            // strategic: route <goal>
      hud.appendChild(hudRow("strategic",
        el("span", "hud-v", "route " + (navName(heads.nav_cmd) || heads.nav_cmd))));
    }
    // metrics: per-arm ADE + ego v0 (always shown — the honest scalar read)
    var ade = arm ? arm.ade : null;
    hud.appendChild(el("div", "hud-metrics",
      "ADE " + (ade == null ? "–" : fmt(ade, 2) + " m") +
      "  ·  v " + fmt(st.ego.speed, 1) + " m/s"));
    if (manId == null && heads.nav_cmd == null && ade == null)
      hud.lastChild.textContent = "no decoded intent";
    // BEV-only fallback note when the camera geometry was unrecoverable
    if (stepIsUncalibrated(st)) {
      hud.appendChild(el("div", "hud-fallback",
        "camera overlay disabled — calibration unverified · see BEV"));
    }
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

  // ---------- maneuver band (badge + per-window timeline strip) --------
  function buildManeuverBand(dom) {
    var band = el("div", "man-band");
    var title = el("div", "panel-title");
    title.appendChild(el("span", null,
      "Maneuver — kinematic class (2 s lookahead)"));
    var lg = el("div", "legend man-legend");
    (maneuverClasses() || []).forEach(function (nm) {
      var chip = el("span", "chip");
      var d = el("span", "dot"); d.style.background = MAN_COLORS[nm] || MAN_NEUTRAL;
      chip.appendChild(d);
      chip.appendChild(el("span", null,
        MAN_LABELS[nm] || nm.replace(/_/g, " ").toUpperCase()));
      lg.appendChild(chip);
    });
    title.appendChild(lg);
    band.appendChild(title);

    var row = el("div", "man-row");
    dom.manBadge = el("div", "man-badge");
    row.appendChild(dom.manBadge);
    var wrap = el("div", "man-strip-wrap");
    dom.manStrip = el("canvas");
    dom.manStrip.onclick = function (ev) {
      var r = dom.manStrip.getBoundingClientRect();
      var n = curEpisode().steps.length;
      seek(clamp(Math.round((ev.clientX - r.left) / r.width * (n - 1)), 0, n - 1));
    };
    wrap.appendChild(dom.manStrip);
    row.appendChild(wrap);
    band.appendChild(row);
    band.appendChild(el("div", "err-hint",
      "Each cell = one window's maneuver; white marker = current. Click to seek."));
    return band;
  }

  function drawManeuver(dom) {
    if (!dom || !dom.manStrip) return;              // hidden when no labels
    var id = curStep().maneuver;
    var color = maneuverColor(id);
    dom.manBadge.textContent = maneuverLabel(id);
    dom.manBadge.style.background = id == null ? "#1b2740" : color;
    dom.manBadge.style.color = id == null ? "var(--ink-dim)" : contrastInk(color);
    dom.manBadge.style.borderColor = id == null ? "var(--line)" : color;
    drawManeuverStrip(dom.manStrip);
  }

  function drawManeuverStrip(canvas) {
    var steps = curEpisode().steps;
    var f = fit(canvas, 30), ctx = f.ctx;
    var n = steps.length, bw = f.w / n;
    for (var i = 0; i < n; i++) {
      var mid = steps[i].maneuver;
      ctx.fillStyle = mid == null ? MAN_NEUTRAL : maneuverColor(mid);
      ctx.fillRect(i * bw, 0, Math.ceil(bw) + 0.6, f.h);
    }
    // current-step marker: dark halo + white line + top caret.
    var cx = (S.step + 0.5) * bw;
    ctx.strokeStyle = "#0a0f18"; ctx.lineWidth = 3.5;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, f.h); ctx.stroke();
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, f.h); ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.beginPath(); ctx.moveTo(cx - 4, 0); ctx.lineTo(cx + 4, 0);
    ctx.lineTo(cx, 5); ctx.closePath(); ctx.fill();
  }

  // ---------- action panel (executed GT steer/accel) -------------------
  function buildActionPanel(dom) {
    var panel = el("div", "act-panel");
    var title = el("div", "panel-title");
    title.appendChild(el("span", null,
      "Action — executed control (ground truth)"));
    var lg = el("div", "legend");
    ["steer", "accel"].forEach(function (k) {
      var chip = el("span", "chip");
      var d = el("span", "dot"); d.style.background = ACT_COLORS[k];
      chip.appendChild(d);
      chip.appendChild(el("span", null, k + " (" + ACT_UNITS[k] + ")"));
      lg.appendChild(chip);
    });
    title.appendChild(lg);
    panel.appendChild(title);

    var grid = el("div", "act-grid");
    dom.actGauge = {}; dom.actSeries = {};
    ["steer", "accel"].forEach(function (kind) {
      var rowEl = el("div", "act-row");
      var gcell = el("div", "act-gauge");
      var lab = el("div", "glabel");
      lab.appendChild(el("span", "gname", kind));
      var gval = el("span", "gval", "–");
      gval.style.color = ACT_COLORS[kind];
      lab.appendChild(gval);
      gcell.appendChild(lab);
      var gbar = el("canvas", "gbar"); gcell.appendChild(gbar);
      gcell.appendChild(el("div", "gunit", ACT_UNITS[kind]));
      rowEl.appendChild(gcell);
      var scell = el("div", "act-series-wrap");
      var series = el("canvas"); scell.appendChild(series);
      rowEl.appendChild(scell);
      grid.appendChild(rowEl);
      dom.actGauge[kind] = { bar: gbar, val: gval };
      dom.actSeries[kind] = series;
    });
    panel.appendChild(grid);
    return panel;
  }

  function drawAction(dom) {
    if (!dom || !dom.actSeries) return;
    var steps = curEpisode().steps;
    ["steer", "accel"].forEach(function (kind) {
      var vals = steps.map(function (s) {
        return s.gt_action ? s.gt_action[kind] : null;
      });
      var rng = niceRange(vals);
      var cur = vals[S.step];
      var g = dom.actGauge[kind];
      g.val.textContent = (cur == null || isNaN(cur)) ? "–"
        : Number(cur).toFixed(kind === "steer" ? 3 : 2);
      drawGauge(g.bar, cur, rng, ACT_COLORS[kind]);
      drawActionSeries(dom.actSeries[kind], vals, rng, ACT_COLORS[kind]);
    });
  }

  // Compact signed bar gauge: track, zero tick, colored bar from zero to the
  // current value, and a knob — a quick sign/magnitude read of the control.
  function drawGauge(canvas, value, rng, color) {
    var f = fit(canvas, 26), ctx = f.ctx;
    var mL = 7, mR = 7, pw = f.w - mL - mR, cy = f.h / 2;
    function X(v) { return mL + (v - rng.lo) / (rng.hi - rng.lo) * pw; }
    ctx.lineCap = "round";
    ctx.strokeStyle = "#ffffff14"; ctx.lineWidth = 6;
    ctx.beginPath(); ctx.moveTo(mL, cy); ctx.lineTo(mL + pw, cy); ctx.stroke();
    var zx = X(0);
    ctx.strokeStyle = "#ffffff40"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(zx, cy - 9); ctx.lineTo(zx, cy + 9); ctx.stroke();
    if (value != null && !isNaN(value)) {
      ctx.strokeStyle = color; ctx.lineWidth = 6;
      ctx.beginPath(); ctx.moveTo(zx, cy); ctx.lineTo(X(value), cy); ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(X(value), cy, 4.5, 0, 6.2832); ctx.fill();
    }
  }

  // Executed-action time-series over the episode: zero baseline + y labels,
  // a soft area fill, the GT line, a current-step marker and value dot.
  function drawActionSeries(canvas, vals, rng, color) {
    var f = fit(canvas, 66), ctx = f.ctx;
    var mL = 32, mR = 6, mT = 6, mB = 4;
    var pw = f.w - mL - mR, ph = f.h - mT - mB;
    var n = vals.length;
    function X(i) { return mL + (n <= 1 ? 0 : i / (n - 1) * pw); }
    function Y(v) {
      return mT + (rng.hi === rng.lo ? ph / 2
        : (rng.hi - v) / (rng.hi - rng.lo) * ph);
    }
    ctx.strokeStyle = "#ffffff20"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(mL, Y(0)); ctx.lineTo(mL + pw, Y(0)); ctx.stroke();
    ctx.fillStyle = "#63728c"; ctx.font = "8.5px monospace";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText("0", mL - 4, Y(0));
    ctx.textBaseline = "top"; ctx.fillText(rng.hi.toFixed(2), mL - 4, mT);
    ctx.textBaseline = "bottom"; ctx.fillText(rng.lo.toFixed(2), mL - 4, mT + ph);
    // current-step marker
    ctx.strokeStyle = "#ffffff30"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(S.step), mT); ctx.lineTo(X(S.step), mT + ph);
    ctx.stroke();
    var pts = vals.map(function (v, i) {
      return (v == null || isNaN(v)) ? null : [X(i), Y(v)];
    }).filter(Boolean);
    if (pts.length > 1) {
      ctx.save();
      var grad = ctx.createLinearGradient(0, mT, 0, mT + ph);
      grad.addColorStop(0, color + "55"); grad.addColorStop(1, color + "08");
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.moveTo(pts[0][0], Y(0));
      pts.forEach(function (p) { ctx.lineTo(p[0], p[1]); });
      ctx.lineTo(pts[pts.length - 1][0], Y(0)); ctx.closePath(); ctx.fill();
      ctx.restore();
    }
    polyline(ctx, pts, color, 1.8, false);
    var cv = vals[S.step];
    if (cv != null && !isNaN(cv)) dot(ctx, [X(S.step), Y(cv)], color, 2.8);
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
      host.appendChild(el("div", "sub", "strategic route / goal (nav command): " +
        (navName(h.nav_cmd) || h.nav_cmd)));
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
