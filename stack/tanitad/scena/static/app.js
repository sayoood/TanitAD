/* TanitScena SPA — vanilla JS + canvas, no build step, no CDN.
 *
 * Two views: (1) HOME = a search-first scenario browser — a prominent local
 * semantic search box (queries /api/search), filter chips (lifecycle, evidence
 * label, ★ rating, data-source kind), and a responsive grid of scenario cards;
 * (2) DETAIL = one scenario in full — opponent evidence, description +
 * correct-behavior, a schematic top-down BEV canvas glyph for the scenario
 * family (legended), metric hooks, a lifecycle stepper, and the dataset-links
 * section. URL hash routes: #/s/<id> (detail), #/q/<query> (search). */
(function () {
  "use strict";

  var STAGES = ["catalogued", "spec-drafted", "data-sourced",
                "oracle-tested", "live-measured", "excellence-proven"];
  var STAGE_COLOR = {
    "catalogued": "#97a6bd", "spec-drafted": "#7c9cff",
    "data-sourced": "#22d3ee", "oracle-tested": "#4fd1c5",
    "live-measured": "#f5b301", "excellence-proven": "#7fd18b",
  };
  var EGO = "#f5b301", CONE = "#f5893a", HAZARD = "#ff6a8a",
      SIGHT = "#22d3ee", MAGENTA = "#e35ce0", WHITEISH = "#eef2f7";

  var S = {
    scenarios: [], byId: {}, detail: {}, meta: null,
    view: "home", curId: null, query: "", results: null,
    filters: { stage: new Set(), label: new Set(), stars: new Set(), kind: new Set() },
    searchSeq: 0, dom: null, searchTimer: null,
  };

  // ---------- utilities ----------
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
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function stageIndex(st) { var i = STAGES.indexOf(st); return i < 0 ? 0 : i; }

  // ---------- routing ----------
  function readHash() {
    var h = location.hash.replace(/^#\/?/, "");
    var p = h.split("/").filter(Boolean);
    if (p[0] === "s" && p[1]) return { view: "detail", id: decodeURIComponent(p[1]) };
    if (p[0] === "q") return { view: "home", query: decodeURIComponent(p.slice(1).join("/") || "") };
    return { view: "home", query: "" };
  }
  function writeHash() {
    var h = "#/";
    if (S.view === "detail" && S.curId) h += "s/" + encodeURIComponent(S.curId);
    else if (S.query) h += "q/" + encodeURIComponent(S.query);
    if (location.hash !== h) history.replaceState(null, "", h);
  }

  // ---------- boot ----------
  function boot() {
    Promise.all([getJSON("/api/scenarios"), getJSON("/api/meta").catch(function () { return null; })])
      .then(function (res) {
        S.scenarios = res[0];
        S.meta = res[1];
        S.byId = {};
        S.scenarios.forEach(function (s) { S.byId[s.id] = s; });
        route();
      })
      .catch(function (e) {
        document.getElementById("app").innerHTML =
          '<div class="err">Failed to load scenarios: ' + e.message + "</div>";
      });
    window.addEventListener("hashchange", route);
    window.addEventListener("keydown", onKey);
    document.getElementById("home-link").onclick = function () { location.hash = "#/"; };
    window.addEventListener("resize", function () {
      if (S.view === "detail" && S.curId && S.detail[S.curId]) {
        var c = document.getElementById("schematic");
        if (c) drawSchematic(c, S.detail[S.curId]);
      }
    });
  }

  function route() {
    var r = readHash();
    if (r.view === "detail") {
      S.view = "detail"; S.curId = r.id;
      renderTopbar();                            // topbar reads S.view — set first
      renderDetail(r.id);
    } else {
      S.view = "home"; S.curId = null;
      renderTopbar();
      if (r.query !== S.query) { S.query = r.query || ""; runSearch(S.query, true); }
      renderHome();
    }
  }

  // ---------- top bar ----------
  function renderTopbar() {
    var c = document.getElementById("topbar-controls");
    c.innerHTML = "";
    if (S.meta) {
      var note = el("span", "embnote");
      note.appendChild(el("span", "pip"));
      note.appendChild(el("span", null, "embedder: " + (S.meta.embedder || "lazy") +
        " · " + S.meta.n + " scenarios"));
      c.appendChild(note);
    }
    if (S.view === "detail") {
      var home = el("button", null, "◂ All scenarios");
      home.onclick = function () { location.hash = "#/"; };
      c.appendChild(home);
    }
  }

  // ---------- HOME ----------
  function renderHome() {
    writeHash();
    var app = document.getElementById("app");
    app.innerHTML = "";

    var hero = el("div", "hero");
    hero.appendChild(el("h1", null, "Opponent-weakness scenario database"));
    hero.appendChild(el("div", "sub",
      "Search SC-01..SC-" + pad2(S.scenarios.length) + " by meaning — construction " +
      "cones, occluded pedestrians, stop-arms, degraded visibility. Local vector " +
      "search, no network."));

    var box = el("div", "searchbox");
    box.appendChild(el("span", "mag", "⌕"));
    var input = el("input");
    input.type = "text"; input.value = S.query;
    input.placeholder = "e.g. construction lane closure, hidden pedestrian, red light…";
    input.setAttribute("aria-label", "semantic search");
    input.oninput = function () { onSearchInput(input.value); };
    box.appendChild(input);
    var kbd = el("span", "kbd", "/"); box.appendChild(kbd);
    var clearBtn = el("button", "clearbtn", "Clear");
    clearBtn.onclick = function () { input.value = ""; onSearchInput(""); input.focus(); };
    box.appendChild(clearBtn);
    hero.appendChild(box);
    var status = el("div", "search-status");
    hero.appendChild(status);
    app.appendChild(hero);

    var filters = el("div", "filters");
    app.appendChild(filters);

    var cards = el("div", "cards");
    app.appendChild(cards);

    S.dom = { input: input, status: status, filters: filters, cards: cards };
    buildFilters();
    updateHomeCards();
    if (S.query) { input.focus(); input.setSelectionRange(input.value.length, input.value.length); }
  }
  function pad2(n) { return (n < 10 ? "0" : "") + n; }

  function onSearchInput(v) {
    S.query = v.trim();
    writeHash();
    if (S.searchTimer) clearTimeout(S.searchTimer);
    if (!S.query) { S.results = null; updateHomeCards(); return; }
    S.searchTimer = setTimeout(function () { runSearch(S.query, false); }, 170);
  }

  function runSearch(q, sync) {
    if (!q) { S.results = null; if (!sync) updateHomeCards(); return; }
    var seq = ++S.searchSeq;
    getJSON("/api/search?q=" + encodeURIComponent(q) + "&k=14").then(function (data) {
      if (seq !== S.searchSeq) return;            // stale response, ignore
      S.results = data.results || [];
      if (S.meta && data.embedder) S.meta.embedder = data.embedder;
      if (!sync) updateHomeCards();
    }).catch(function () { if (seq === S.searchSeq && !sync) { S.results = []; updateHomeCards(); } });
  }

  function passesFilters(s) {
    var F = S.filters;
    if (F.stage.size && !F.stage.has(s.lifecycle_stage)) return false;
    if (F.label.size && !F.label.has(s.evidence_label || "none")) return false;
    if (F.stars.size && !F.stars.has(String(s.stars || 0))) return false;
    if (F.kind.size) {
      var ks = s.data_source_kinds || [];
      if (!ks.some(function (k) { return F.kind.has(k); })) return false;
    }
    return true;
  }

  function activeFilterCount() {
    var F = S.filters;
    return F.stage.size + F.label.size + F.stars.size + F.kind.size;
  }

  function buildFilters() {
    var host = S.dom.filters;
    host.innerHTML = "";
    var stagesPresent = STAGES.filter(function (st) {
      return S.scenarios.some(function (s) { return s.lifecycle_stage === st; });
    });
    var labels = ["FACT", "CLAIM", "INFER"].filter(function (l) {
      return S.scenarios.some(function (s) { return s.evidence_label === l; });
    });
    var starsPresent = uniq(S.scenarios.map(function (s) { return s.stars || 0; }))
      .sort(function (a, b) { return b - a; });
    var kinds = uniq([].concat.apply([], S.scenarios.map(function (s) {
      return s.data_source_kinds || [];
    }))).sort();

    host.appendChild(filterRow("Lifecycle", "stage", stagesPresent, function (v) {
      var c = STAGE_COLOR[v] || "#97a6bd";
      return { label: v, dot: c };
    }));
    host.appendChild(filterRow("Evidence", "label", labels, function (v) {
      return { label: v, dot: v === "FACT" ? "#7fd18b" : v === "CLAIM" ? "#f5b301" : "#97a6bd" };
    }));
    host.appendChild(filterRow("Rating", "stars", starsPresent.map(String), function (v) {
      var n = +v; return { label: n ? "★".repeat(n) : "unrated", dot: n ? "#f5b301" : "#63728c" };
    }));
    host.appendChild(filterRow("Data source", "kind", kinds, function (v) {
      return { label: v, dot: "#22d3ee" };
    }));
  }

  function filterRow(label, group, values, styler) {
    var row = el("div", "filter-row");
    row.appendChild(el("span", "flabel", label));
    values.forEach(function (v) {
      var st = styler(v);
      var count = S.scenarios.filter(function (s) {
        if (group === "stage") return s.lifecycle_stage === v;
        if (group === "label") return (s.evidence_label || "none") === v;
        if (group === "stars") return String(s.stars || 0) === v;
        if (group === "kind") return (s.data_source_kinds || []).indexOf(v) >= 0;
      }).length;
      var chip = el("span", "fchip");
      var dot = el("span", "fdot"); dot.style.background = st.dot; dot.style.color = st.dot;
      chip.appendChild(dot);
      chip.appendChild(el("span", null, st.label));
      chip.appendChild(el("span", "fx", String(count)));
      if (S.filters[group].has(v)) chip.classList.add("on");
      chip.onclick = function () {
        if (S.filters[group].has(v)) S.filters[group].delete(v);
        else S.filters[group].add(v);
        chip.classList.toggle("on");
        updateHomeCards();
      };
      row.appendChild(chip);
    });
    if (label === "Data source" && activeFilterCount()) {
      var clear = el("span", "fchip fclear", "clear filters ✕");
      clear.onclick = function () {
        S.filters = { stage: new Set(), label: new Set(), stars: new Set(), kind: new Set() };
        buildFilters(); updateHomeCards();
      };
      row.appendChild(clear);
    }
    return row;
  }

  function updateHomeCards() {
    var base = S.results ? S.results.slice() : S.scenarios.slice();
    var vis = base.filter(passesFilters);
    var maxScore = 0;
    if (S.results) S.results.forEach(function (r) { maxScore = Math.max(maxScore, r.score || 0); });

    var cards = S.dom.cards;
    cards.innerHTML = "";
    if (!vis.length) {
      cards.appendChild(el("div", "empty",
        S.results ? "No scenarios match this search + filters." : "No scenarios match these filters."));
    } else {
      vis.forEach(function (s) { cards.appendChild(scenarioCard(s, maxScore)); });
    }

    var st = S.dom.status; st.innerHTML = "";
    if (S.results) {
      st.appendChild(el("span", null, vis.length + " of " + S.results.length + " results for"));
      st.appendChild(el("span", "badge", "“" + S.query + "”"));
      st.appendChild(el("span", null, "· ranked by local " +
        ((S.meta && S.meta.embedder) || "vector") + " similarity"));
    } else {
      st.appendChild(el("span", null, vis.length + " scenario" + (vis.length === 1 ? "" : "s") +
        (activeFilterCount() ? " (filtered)" : " · type to search by meaning")));
    }
    // refresh the clear-filters affordance state
    if (S.dom && S.dom.filters) {
      var hasClear = S.dom.filters.querySelector(".fclear");
      if (activeFilterCount() && !hasClear) buildFilters();
      if (!activeFilterCount() && hasClear) buildFilters();
    }
  }

  function tag(cls, txt) { return el("span", "tag " + cls, txt); }
  function evChip(label) {
    var l = label || "none";
    return tag("ev-" + l, label || "unlabelled");
  }
  function stageChip(stage) {
    var t = tag("stage", stage || "—");
    var c = STAGE_COLOR[stage] || "#97a6bd";
    t.style.background = c + "22"; t.style.color = c; t.style.borderColor = c + "55";
    return t;
  }

  function scenarioCard(s, maxScore) {
    var card = el("div", "card");
    if (s.score != null) {
      var sb = el("div", "scorebar");
      var f = el("div", "fill");
      f.style.width = clamp(maxScore ? s.score / maxScore : 0, 0, 1) * 100 + "%";
      sb.appendChild(f); card.appendChild(sb);
    }
    var body = el("div", "body");
    var idrow = el("div", "idrow");
    idrow.appendChild(el("span", "scid", s.id));
    if (s.stars) idrow.appendChild(el("span", "stars", "★".repeat(s.stars)));
    if (s.score != null) idrow.appendChild(el("span", "score", s.score.toFixed(3)));
    body.appendChild(idrow);
    body.appendChild(el("div", "title", s.title || "(untitled)"));
    if (s.description) body.appendChild(el("div", "desc", s.description));

    var chips = el("div", "chiprow");
    chips.appendChild(evChip(s.evidence_label));
    chips.appendChild(stageChip(s.lifecycle_stage));
    if (s.w_code) chips.appendChild(tag("wtag", s.w_code + (s.family ? " fam" : "")));
    if (s.headline) chips.appendChild(tag("headline", "headline"));
    if (s.is_new) chips.appendChild(tag("newtag", "new"));
    body.appendChild(chips);

    if (s.data_source_kinds && s.data_source_kinds.length) {
      var kinds = el("div", "kinds");
      s.data_source_kinds.forEach(function (k) { kinds.appendChild(tag("kind", k)); });
      body.appendChild(kinds);
    }
    card.appendChild(body);
    card.onclick = function () { location.hash = "#/s/" + encodeURIComponent(s.id); };
    return card;
  }

  // ---------- DETAIL ----------
  function renderDetail(id) {
    writeHash();
    var app = document.getElementById("app");
    app.innerHTML = '<div class="loading">Loading ' + id + "…</div>";
    var have = S.detail[id];
    var p = have ? Promise.resolve(have)
      : getJSON("/api/scenario/" + encodeURIComponent(id)).then(function (f) {
        S.detail[id] = f; return f;
      });
    p.then(paintDetail).catch(function () {
      app.innerHTML = '<div class="err">Scenario ' + id + " not found. " +
        '<a href="#/" style="color:var(--refa)">Back to all scenarios</a>.</div>';
    });
  }

  function paintDetail(s) {
    var app = document.getElementById("app");
    app.innerHTML = "";

    // header
    var head = el("div", "detail-head");
    var back = el("button", "back", "◂ Back");
    back.onclick = function () { location.hash = "#/"; };
    head.appendChild(back);
    var ht = el("div", "htext");
    ht.appendChild(el("div", "scid", s.id + (s.w_code ? "  ·  " + s.w_code + (s.family ? " family" : "") : "")));
    ht.appendChild(el("h1", null, s.title || "(untitled)"));
    var chips = el("div", "chiprow");
    if (s.stars) { var st = el("span", "stars", "★".repeat(s.stars)); chips.appendChild(st); }
    chips.appendChild(evChip(s.evidence_label));
    chips.appendChild(stageChip(s.lifecycle_stage));
    if (s.headline) chips.appendChild(tag("headline", "headline"));
    if (s.is_new) chips.appendChild(tag("newtag", "new"));
    ht.appendChild(chips);
    head.appendChild(ht);
    app.appendChild(head);

    // opponent evidence panel
    var ev = el("div", "panel");
    ev.appendChild(ptitle("Opponent evidence", evChip(s.evidence_label)));
    var evline = el("div", "evidence-line body-txt");
    evline.appendChild(el("span", "txt", s.opponent_evidence || "—"));
    ev.appendChild(evline);
    if (s.evidence_links && s.evidence_links.length) {
      var links = el("div", "evidence-links");
      s.evidence_links.forEach(function (u) {
        var a = el("a", null, u); a.href = u; a.target = "_blank"; a.rel = "noopener";
        links.appendChild(a);
      });
      ev.appendChild(links);
    }
    app.appendChild(ev);

    // grid: schematic + description
    var grid = el("div", "detail-grid");

    var schematicPanel = el("div", "panel");
    schematicPanel.appendChild(ptitle("Scenario schematic (top-down BEV)", null));
    var wrap = el("div", "schematic-wrap");
    var canvas = el("canvas"); canvas.id = "schematic";
    wrap.appendChild(canvas);
    schematicPanel.appendChild(wrap);
    schematicPanel.appendChild(el("div", "schematic-cap",
      "Hand-authored glyph for the " + (familyLabel(familyOf(s.id))) + " family — schematic, not to scale."));
    grid.appendChild(schematicPanel);

    var textPanel = el("div", "panel");
    textPanel.appendChild(ptitle("Description", null));
    textPanel.appendChild(el("div", "body-txt", s.description || "—"));
    if (s.correct_behavior) {
      var cb = el("div", "cb");
      cb.appendChild(el("div", "k", "Correct behavior"));
      cb.appendChild(el("div", null, s.correct_behavior));
      textPanel.appendChild(cb);
    }
    if (s.mechanism) {
      var mech = el("div", "mech");
      mech.appendChild(el("div", "k", "TanitAD mechanism"));
      mech.appendChild(el("div", null, s.mechanism));
      textPanel.appendChild(mech);
    }
    grid.appendChild(textPanel);
    app.appendChild(grid);

    // lifecycle stepper
    var life = el("div", "panel");
    life.appendChild(ptitle("Lifecycle", el("span", "embnote",
      "current: " + (s.lifecycle_stage || "—"))));
    life.appendChild(buildStepper(s.lifecycle_stage));
    if (s.status_text) life.appendChild(el("div", "hooks-text", s.status_text));
    app.appendChild(life);

    // metric hooks
    var mh = el("div", "panel");
    mh.appendChild(ptitle("Metric hooks", null));
    if (s.metric_hooks && s.metric_hooks.length) {
      var hooks = el("div", "hooks");
      s.metric_hooks.forEach(function (h) { hooks.appendChild(el("span", "hook", h)); });
      mh.appendChild(hooks);
    } else {
      mh.appendChild(el("div", "body-txt", "—"));
    }
    if (s.metric_hooks_text) mh.appendChild(el("div", "hooks-text", s.metric_hooks_text));
    app.appendChild(mh);

    // dataset links
    var ds = el("div", "panel");
    ds.appendChild(ptitle("Dataset links", el("span", "embnote",
      (s.data_sources ? s.data_sources.length : 0) + " source" +
      ((s.data_sources && s.data_sources.length === 1) ? "" : "s"))));
    if (s.data_sources && s.data_sources.length) {
      var dsGrid = el("div", "datasets");
      s.data_sources.forEach(function (d) { dsGrid.appendChild(datasetCard(d)); });
      ds.appendChild(dsGrid);
    } else {
      ds.appendChild(el("div", "body-txt", "No data sources catalogued yet."));
    }
    app.appendChild(ds);

    drawSchematic(canvas, s);
  }

  function ptitle(text, extra) {
    var t = el("div", "ptitle");
    t.appendChild(el("span", null, text));
    if (extra) t.appendChild(extra);
    return t;
  }

  function datasetCard(d) {
    var card = el("div", "ds-card");
    var top = el("div", "ds-top");
    top.appendChild(el("span", "ds-kind k-" + (d.kind || "other"), d.kind || "other"));
    card.appendChild(top);
    card.appendChild(el("div", "ds-ref", d.ref || ""));
    if (d.status) card.appendChild(el("div", "ds-status", d.status));
    var actions = el("div", "ds-actions");
    if (d.link) {
      var a = el("a", "ds-link", "get dataset ▸");
      a.href = d.link; a.target = "_blank"; a.rel = "noopener";
      actions.appendChild(a);
    } else {
      actions.appendChild(el("span", "ds-nolink", "no public link"));
    }
    if (d.replay) {
      var r = el("a", "ds-replay", "open in replay ▸");
      r.href = d.link || "#"; r.target = "_blank"; r.rel = "noopener";
      actions.appendChild(r);
    }
    card.appendChild(actions);
    return card;
  }

  function buildStepper(current) {
    var wrap = el("div", "stepper");
    var curIdx = stageIndex(current);
    STAGES.forEach(function (st, i) {
      var step = el("div", "step" + (i < curIdx ? " done" : i === curIdx ? " cur" : ""));
      step.appendChild(el("div", "connector"));
      step.appendChild(el("div", "node", i < curIdx ? "✓" : String(i + 1)));
      step.appendChild(el("div", "slabel", st));
      wrap.appendChild(step);
    });
    return wrap;
  }

  // ---------- schematic canvas ----------
  function familyOf(id) {
    return ({
      "SC-01": "cone", "SC-09": "cone",
      "SC-02": "occlusion", "SC-03": "occlusion",
      "SC-04": "stoparm", "SC-05": "visibility", "SC-06": "emergency",
      "SC-07": "mrm", "SC-08": "stall", "SC-10": "atypical",
      "SC-11": "wrongside", "SC-12": "officer", "SC-13": "lead", "SC-14": "redlight",
    })[id] || "generic";
  }
  function familyLabel(fam) {
    return ({
      cone: "work-zone / cone-taper", occlusion: "occluded-actor",
      stoparm: "stop-arm gate", visibility: "degraded-visibility",
      emergency: "emergency-vehicle", mrm: "post-incident MRM",
      stall: "frozen-vehicle / stall", atypical: "atypical-vehicle",
      wrongside: "wrong-side / oncoming", officer: "traffic-officer gesture",
      lead: "stationary lead", redlight: "signal-phase", generic: "generic scenario",
    })[fam] || fam;
  }

  function fitCanvas(canvas, cssW, cssH) {
    var dpr = window.devicePixelRatio || 1;
    canvas.style.height = cssH + "px";
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    return ctx;
  }

  function drawSchematic(canvas, s) {
    var W = canvas.clientWidth || (canvas.parentElement && canvas.parentElement.clientWidth) || 440;
    var H = 300;
    var ctx = fitCanvas(canvas, W, H);
    var road = roadBase(ctx, W, H);
    var fam = familyOf(s.id);
    (GLYPHS[fam] || GLYPHS.generic)(ctx, W, H, road);
  }

  function roadBase(ctx, W, H) {
    var rl = W * 0.28, rr = W * 0.72, top = 14, bot = H - 14;
    var cx = (rl + rr) / 2;
    // grass/verge
    ctx.fillStyle = "#0c121d"; ctx.fillRect(0, 0, W, H);
    // asphalt
    ctx.fillStyle = "#141c2b"; ctx.fillRect(rl, top, rr - rl, bot - top);
    // edge lines
    ctx.strokeStyle = "#3a4a66"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(rl, top); ctx.lineTo(rl, bot);
    ctx.moveTo(rr, top); ctx.lineTo(rr, bot); ctx.stroke();
    // dashed centre divider
    ctx.strokeStyle = "#5b6b86"; ctx.lineWidth = 2; ctx.setLineDash([9, 9]);
    ctx.beginPath(); ctx.moveTo(cx, top); ctx.lineTo(cx, bot); ctx.stroke();
    ctx.setLineDash([]);
    return { rl: rl, rr: rr, top: top, bot: bot, cx: cx,
             leftLane: (rl + cx) / 2, rightLane: (cx + rr) / 2, laneW: (rr - rl) / 2 };
  }

  function ego(ctx, x, y, color) {
    color = color || EGO;
    ctx.save();
    ctx.fillStyle = color; ctx.strokeStyle = "#0a0f18"; ctx.lineWidth = 1.5;
    roundRectPath(ctx, x - 9, y - 14, 18, 28, 5); ctx.fill(); ctx.stroke();
    // heading nub (forward = up)
    ctx.fillStyle = "#0a0f18";
    ctx.beginPath(); ctx.moveTo(x, y - 16); ctx.lineTo(x - 4, y - 9); ctx.lineTo(x + 4, y - 9); ctx.closePath(); ctx.fill();
    ctx.restore();
  }
  function carRect(ctx, x, y, w, h, color, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha == null ? 1 : alpha;
    ctx.fillStyle = color; ctx.strokeStyle = "#0a0f18"; ctx.lineWidth = 1.5;
    roundRectPath(ctx, x - w / 2, y - h / 2, w, h, 4); ctx.fill(); ctx.stroke();
    ctx.restore();
  }
  function coneGlyph(ctx, x, y) {
    ctx.save();
    ctx.fillStyle = CONE;
    ctx.beginPath(); ctx.moveTo(x, y - 7); ctx.lineTo(x - 5, y + 5); ctx.lineTo(x + 5, y + 5); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#ffe0c2"; ctx.fillRect(x - 3.4, y - 1, 6.8, 2.2);
    ctx.restore();
  }
  function dotMark(ctx, x, y, color, r) {
    ctx.save(); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r || 6, 0, 6.2832); ctx.fill();
    ctx.strokeStyle = "#0a0f18"; ctx.lineWidth = 1; ctx.stroke(); ctx.restore();
  }
  function pathLine(ctx, pts, color, dashed, width) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width || 2.4; ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (dashed) ctx.setLineDash([7, 6]);
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke(); ctx.restore();
  }
  function hatch(ctx, poly, color) {
    ctx.save();
    ctx.beginPath(); ctx.moveTo(poly[0][0], poly[0][1]);
    for (var i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
    ctx.closePath();
    ctx.fillStyle = color + "18"; ctx.fill();
    ctx.clip();
    ctx.strokeStyle = color + "55"; ctx.lineWidth = 1.5;
    var minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
    poly.forEach(function (p) { minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]); minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]); });
    for (var d = minY - (maxX - minX); d < maxY; d += 9) {
      ctx.beginPath(); ctx.moveTo(minX, d); ctx.lineTo(maxX, d + (maxX - minX)); ctx.stroke();
    }
    ctx.restore();
  }
  function stopLine(ctx, x0, x1, y) {
    ctx.save(); ctx.strokeStyle = WHITEISH; ctx.lineWidth = 4; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.restore();
  }
  function signalHead(ctx, x, y, on) {
    ctx.save();
    ctx.fillStyle = "#0a0f18"; ctx.strokeStyle = "#3a4a66"; ctx.lineWidth = 1.5;
    roundRectPath(ctx, x - 7, y - 18, 14, 36, 4); ctx.fill(); ctx.stroke();
    var cols = ["#ff5a5a", "#f5b301", "#7fd18b"];
    for (var i = 0; i < 3; i++) {
      ctx.beginPath(); ctx.arc(x, y - 11 + i * 11, 4, 0, 6.2832);
      ctx.fillStyle = (on === i) ? cols[i] : "#20293a"; ctx.fill();
      if (on === i) { ctx.shadowColor = cols[i]; ctx.shadowBlur = 8; ctx.fill(); ctx.shadowBlur = 0; }
    }
    ctx.restore();
  }
  function roundRectPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  function legendBox(ctx, x, y, rows) {
    ctx.save();
    ctx.font = "11px system-ui, sans-serif";
    var wMax = 0; rows.forEach(function (r) { wMax = Math.max(wMax, ctx.measureText(r.t).width); });
    var bw = wMax + 44, lh = 16, bh = rows.length * lh + 12;
    ctx.fillStyle = "#0b1120e0"; ctx.strokeStyle = "#ffffff22"; ctx.lineWidth = 1;
    roundRectPath(ctx, x, y, bw, bh, 7); ctx.fill(); ctx.stroke();
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    rows.forEach(function (r, i) {
      var ry = y + 11 + i * lh;
      swatch(ctx, x + 8, ry, r);
      ctx.fillStyle = "#c7d2e2"; ctx.fillText(r.t, x + 28, ry);
    });
    ctx.restore();
  }
  function swatch(ctx, x, y, r) {
    ctx.save();
    if (r.type === "line" || r.type === "dash") {
      ctx.strokeStyle = r.c; ctx.lineWidth = 2.6;
      if (r.type === "dash") ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 15, y); ctx.stroke();
    } else if (r.type === "rect") {
      ctx.fillStyle = r.c; roundRectPath(ctx, x + 1, y - 5, 14, 10, 2); ctx.fill();
    } else if (r.type === "cone") {
      coneGlyph(ctx, x + 8, y + 1);
    } else if (r.type === "hatch") {
      ctx.fillStyle = r.c + "33"; ctx.fillRect(x + 1, y - 5, 14, 10);
      ctx.strokeStyle = r.c + "aa"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x + 1, y + 5); ctx.lineTo(x + 15, y - 5); ctx.stroke();
    } else if (r.type === "tri") {
      ctx.fillStyle = r.c; ctx.beginPath();
      ctx.moveTo(x + 8, y - 6); ctx.lineTo(x + 2, y + 5); ctx.lineTo(x + 14, y + 5); ctx.closePath(); ctx.fill();
    } else {
      dotMark(ctx, x + 8, y, r.c, 5);
    }
    ctx.restore();
  }

  // ---------- glyph family drawings ----------
  var GLYPHS = {
    cone: function (ctx, W, H, r) {
      // right lane closed by a cone taper; ego merges left
      var pts = [];
      for (var i = 0; i <= 4; i++) {
        var t = i / 4;
        pts.push([r.rr - 6 - t * (r.rightLane - r.rl - 6 + r.laneW * 0.1), r.bot - 60 - t * (r.bot - r.top - 120)]);
      }
      var closed = [[r.rr, r.top + 10]].concat(pts.slice().reverse()).concat([[r.rr, r.bot - 40]]);
      hatch(ctx, closed, "#f5b301");
      pathLine(ctx, [[r.rightLane, r.bot - 20], [r.rightLane, r.bot - 70], [r.leftLane, r.bot - 130], [r.leftLane, r.top + 20]], EGO, true);
      pts.forEach(function (p) { coneGlyph(ctx, p[0], p[1]); });
      ego(ctx, r.rightLane, r.bot - 18);
      legendBox(ctx, 10, 10, [
        { t: "cones (taper)", type: "cone" },
        { t: "closed area", type: "hatch", c: "#f5b301" },
        { t: "ego merge path", type: "dash", c: EGO },
        { t: "ego", type: "tri", c: EGO },
      ]);
    },
    occlusion: function (ctx, W, H, r) {
      var vanY = r.top + 96;
      carRect(ctx, r.rightLane, vanY, r.laneW * 0.7, 46, "#6b7890");   // parked occluder
      // occluded wedge from ego to behind van
      ctx.save(); ctx.fillStyle = SIGHT + "1c";
      ctx.beginPath(); ctx.moveTo(r.leftLane, r.bot - 30);
      ctx.lineTo(r.rightLane - 14, vanY - 26); ctx.lineTo(r.rr, vanY - 40); ctx.lineTo(r.rr, vanY + 30);
      ctx.lineTo(r.rightLane - 14, vanY + 26); ctx.closePath(); ctx.fill(); ctx.restore();
      dotMark(ctx, r.rightLane + 4, vanY + 40, MAGENTA, 6);            // hidden pedestrian
      pathLine(ctx, [[r.leftLane, r.bot - 18], [r.leftLane, r.top + 40]], EGO, true);
      ego(ctx, r.leftLane, r.bot - 18);
      legendBox(ctx, 10, 10, [
        { t: "occluder (van)", type: "rect", c: "#6b7890" },
        { t: "hidden actor", type: "dot", c: MAGENTA },
        { t: "occluded sector", type: "hatch", c: SIGHT },
        { t: "ego (cautious)", type: "tri", c: EGO },
      ]);
    },
    stoparm: function (ctx, W, H, r) {
      var busY = r.top + 92;
      carRect(ctx, r.leftLane, busY, r.laneW * 0.72, 88, "#f5c542");   // school bus (left lane)
      // stop-arm out toward ego side
      ctx.save(); ctx.fillStyle = "#ff5a5a";
      roundRectPath(ctx, r.leftLane + r.laneW * 0.34, busY - 6, 16, 16, 3); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = "bold 9px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("STOP", r.leftLane + r.laneW * 0.34 + 8, busY + 2); ctx.restore();
      dotMark(ctx, r.cx + 4, busY - 54, MAGENTA, 6);                   // child crossing in front
      var lineY = r.bot - 96;
      stopLine(ctx, r.cx, r.rr, lineY);
      ego(ctx, r.rightLane, lineY + 26);                              // ego stopped before line
      legendBox(ctx, 10, 10, [
        { t: "school bus", type: "rect", c: "#f5c542" },
        { t: "stop-arm", type: "rect", c: "#ff5a5a" },
        { t: "stop line", type: "dash", c: WHITEISH },
        { t: "occluded child", type: "dot", c: MAGENTA },
        { t: "ego (hard stop)", type: "tri", c: EGO },
      ]);
    },
    visibility: function (ctx, W, H, r) {
      var g = ctx.createLinearGradient(0, r.top, 0, r.bot);
      g.addColorStop(0, "#c9d2e0cc"); g.addColorStop(0.55, "#c9d2e030"); g.addColorStop(1, "#c9d2e000");
      ctx.fillStyle = g; ctx.fillRect(r.rl, r.top, r.rr - r.rl, (r.bot - r.top) * 0.62);
      carRect(ctx, r.rightLane, r.top + 54, r.laneW * 0.6, 34, "#8a94a6", 0.5);  // barely-visible hazard
      ctx.save(); ctx.strokeStyle = SIGHT; ctx.setLineDash([6, 5]); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(r.rl, r.top + (r.bot - r.top) * 0.58); ctx.lineTo(r.rr, r.top + (r.bot - r.top) * 0.58); ctx.stroke(); ctx.restore();
      pathLine(ctx, [[r.cx, r.bot - 18], [r.cx, r.top + (r.bot - r.top) * 0.55]], EGO, true);
      ego(ctx, r.cx, r.bot - 18);
      legendBox(ctx, 10, 10, [
        { t: "degraded visibility", type: "hatch", c: "#c9d2e0" },
        { t: "sightline limit", type: "dash", c: SIGHT },
        { t: "obscured hazard", type: "rect", c: "#8a94a6" },
        { t: "ego (slowed)", type: "tri", c: EGO },
      ]);
    },
    emergency: function (ctx, W, H, r) {
      var evY = r.top + 74;
      carRect(ctx, r.cx, evY, r.laneW * 0.8, 46, "#e14b4b");
      ctx.save(); dotMark(ctx, r.cx - 6, evY - 26, "#5a9bff", 4); dotMark(ctx, r.cx + 6, evY - 26, "#ff5a5a", 4); ctx.restore();
      pathLine(ctx, [[r.cx, r.bot - 18], [r.rightLane, r.bot - 90], [r.rr - 8, r.bot - 150]], EGO, true);
      ego(ctx, r.cx, r.bot - 18);
      legendBox(ctx, 10, 10, [
        { t: "emergency vehicle", type: "rect", c: "#e14b4b" },
        { t: "lights", type: "dot", c: "#5a9bff" },
        { t: "yield / clear path", type: "dash", c: EGO },
        { t: "ego", type: "tri", c: EGO },
      ]);
    },
    mrm: function (ctx, W, H, r) {
      var aY = r.bot - 96;
      // anomaly star at ego front
      ctx.save(); ctx.strokeStyle = HAZARD; ctx.lineWidth = 2.4;
      for (var k = 0; k < 8; k++) { var a = k / 8 * 6.2832; ctx.beginPath(); ctx.moveTo(r.cx, aY); ctx.lineTo(r.cx + Math.cos(a) * 13, aY + Math.sin(a) * 13); ctx.stroke(); }
      ctx.restore();
      ctx.save(); ctx.strokeStyle = SIGHT; ctx.setLineDash([5, 5]); ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(r.cx, r.bot - 18, 46, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke(); ctx.restore();
      ego(ctx, r.cx, r.bot - 18);
      ctx.save(); ctx.fillStyle = EGO; ctx.font = "bold 10px system-ui"; ctx.textAlign = "center";
      ctx.fillText("freeze-in-place MRM", r.cx, r.bot - 44); ctx.restore();
      legendBox(ctx, 10, 10, [
        { t: "anomaly / collision", type: "dot", c: HAZARD },
        { t: "corridor check", type: "dash", c: SIGHT },
        { t: "ego (MRM stop)", type: "tri", c: EGO },
      ]);
    },
    stall: function (ctx, W, H, r) {
      carRect(ctx, r.cx, r.top + 96, r.laneW * 0.9, 46, "#6b7890");
      ctx.save(); ctx.fillStyle = HAZARD; ctx.font = "bold 14px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("✕", r.cx, r.top + 96); ctx.restore();
      pathLine(ctx, [[r.cx, r.bot - 18], [r.rightLane, r.bot - 80], [r.rr - 6, r.bot - 120]], EGO, true);
      // safe-stop marker on shoulder
      ctx.save(); ctx.strokeStyle = "#7fd18b"; ctx.lineWidth = 2; ctx.setLineDash([]);
      roundRectPath(ctx, r.rr - 16, r.bot - 138, 18, 26, 4); ctx.stroke(); ctx.restore();
      ego(ctx, r.cx, r.bot - 18);
      legendBox(ctx, 10, 10, [
        { t: "blocked vehicle", type: "rect", c: "#6b7890" },
        { t: "chosen safe-stop", type: "line", c: "#7fd18b" },
        { t: "stop-placement path", type: "dash", c: EGO },
        { t: "ego", type: "tri", c: EGO },
      ]);
    },
    atypical: function (ctx, W, H, r) {
      var vY = r.top + 92;
      carRect(ctx, r.cx, vY, r.laneW * 0.8, 40, "#b98cff");
      ctx.save(); ctx.fillStyle = "#fff"; ctx.font = "bold 12px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("↑?", r.cx, vY); ctx.restore();
      pathLine(ctx, [[r.cx, vY + 22], [r.cx - 10, vY + 70]], "#8a94a6", true, 2);   // class-prior (wrong)
      pathLine(ctx, [[r.cx, vY - 22], [r.cx + 8, vY - 66]], HAZARD, false, 2.4);     // observed motion
      ego(ctx, r.cx, r.bot - 18);
      legendBox(ctx, 10, 10, [
        { t: "atypical vehicle", type: "rect", c: "#b98cff" },
        { t: "class-prior (wrong)", type: "dash", c: "#8a94a6" },
        { t: "observed motion", type: "line", c: HAZARD },
        { t: "ego", type: "tri", c: EGO },
      ]);
    },
    wrongside: function (ctx, W, H, r) {
      carRect(ctx, r.rightLane, r.top + 92, r.laneW * 0.7, 40, "#6b7890");   // obstruction in ego lane
      carRect(ctx, r.leftLane, r.top + 46, r.laneW * 0.6, 34, MAGENTA, 0.4);  // imagined oncoming
      pathLine(ctx, [[r.rightLane, r.bot - 18], [r.cx, r.bot - 80], [r.leftLane + 6, r.top + 140], [r.rightLane, r.top + 70]], EGO, true);
      ego(ctx, r.rightLane, r.bot - 18);
      legendBox(ctx, 10, 10, [
        { t: "obstruction", type: "rect", c: "#6b7890" },
        { t: "imagined oncoming", type: "rect", c: MAGENTA },
        { t: "bounded excursion", type: "dash", c: EGO },
        { t: "ego", type: "tri", c: EGO },
      ]);
    },
    officer: function (ctx, W, H, r) {
      signalHead(ctx, r.rr - 2, r.top + 40, -1);   // dead signal (none lit)
      ctx.save(); ctx.strokeStyle = "#ff5a5a"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(r.rr - 8, r.top + 34); ctx.lineTo(r.rr + 4, r.top + 46);
      ctx.moveTo(r.rr + 4, r.top + 34); ctx.lineTo(r.rr - 8, r.top + 46); ctx.stroke(); ctx.restore();
      // officer figure
      var ox = r.cx, oy = r.top + 96;
      ctx.save(); ctx.strokeStyle = "#ffd766"; ctx.fillStyle = "#ffd766"; ctx.lineWidth = 2;
      dotMark(ctx, ox, oy - 10, "#ffd766", 5);
      ctx.beginPath(); ctx.moveTo(ox, oy - 5); ctx.lineTo(ox, oy + 10);
      ctx.moveTo(ox, oy - 2); ctx.lineTo(ox + 12, oy - 8);
      ctx.moveTo(ox, oy - 2); ctx.lineTo(ox - 10, oy + 2);
      ctx.moveTo(ox, oy + 10); ctx.lineTo(ox - 5, oy + 20); ctx.moveTo(ox, oy + 10); ctx.lineTo(ox + 5, oy + 20); ctx.stroke(); ctx.restore();
      pathLine(ctx, [[r.cx, r.bot - 18], [r.cx, r.bot - 70]], EGO, true);
      ego(ctx, r.cx, r.bot - 18);
      legendBox(ctx, 10, 10, [
        { t: "dead signal", type: "dot", c: "#20293a" },
        { t: "directing officer", type: "dot", c: "#ffd766" },
        { t: "ego (defer / creep)", type: "dash", c: EGO },
      ]);
    },
    lead: function (ctx, W, H, r) {
      var leadY = r.top + 92;
      carRect(ctx, r.cx, leadY, r.laneW * 0.85, 44, "#6b7890");
      // following-gap bracket
      ctx.save(); ctx.strokeStyle = "#7fd18b"; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(r.cx + 30, leadY + 24); ctx.lineTo(r.cx + 30, r.bot - 44);
      ctx.moveTo(r.cx + 24, leadY + 24); ctx.lineTo(r.cx + 36, leadY + 24);
      ctx.moveTo(r.cx + 24, r.bot - 44); ctx.lineTo(r.cx + 36, r.bot - 44); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle = "#7fd18b"; ctx.font = "10px system-ui"; ctx.textAlign = "left";
      ctx.fillText("safe gap", r.cx + 40, (leadY + r.bot - 20) / 2); ctx.restore();
      pathLine(ctx, [[r.cx, r.bot - 18], [r.cx, r.bot - 44]], EGO, false);
      ego(ctx, r.cx, r.bot - 18);
      legendBox(ctx, 10, 10, [
        { t: "stopped lead", type: "rect", c: "#6b7890" },
        { t: "following gap", type: "dash", c: "#7fd18b" },
        { t: "ego (early braking)", type: "tri", c: EGO },
      ]);
    },
    redlight: function (ctx, W, H, r) {
      signalHead(ctx, r.cx, r.top + 34, 0);          // red lit
      var lineY = r.top + 120;
      stopLine(ctx, r.rl, r.rr, lineY);
      pathLine(ctx, [[r.cx, r.bot - 18], [r.cx, lineY + 22]], EGO, false);
      ego(ctx, r.cx, lineY + 24);
      legendBox(ctx, 10, 10, [
        { t: "red signal", type: "dot", c: "#ff5a5a" },
        { t: "stop line", type: "dash", c: WHITEISH },
        { t: "ego (hard stop)", type: "tri", c: EGO },
      ]);
    },
    generic: function (ctx, W, H, r) {
      ctx.save(); ctx.fillStyle = HAZARD; ctx.font = "bold 22px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("⚠", r.cx, r.top + 92); ctx.restore();
      ctx.save(); ctx.fillStyle = "#63728c"; ctx.font = "11px system-ui"; ctx.textAlign = "center";
      ctx.fillText("scenario hazard ahead", r.cx, r.top + 116); ctx.restore();
      pathLine(ctx, [[r.cx, r.bot - 18], [r.cx, r.top + 130]], EGO, true);
      ego(ctx, r.cx, r.bot - 18);
      legendBox(ctx, 10, 10, [
        { t: "hazard", type: "dot", c: HAZARD },
        { t: "ego path", type: "dash", c: EGO },
        { t: "ego", type: "tri", c: EGO },
      ]);
    },
  };

  // ---------- misc ----------
  function uniq(arr) { var out = [], seen = {}; arr.forEach(function (x) { if (!seen[x]) { seen[x] = 1; out.push(x); } }); return out; }

  function onKey(e) {
    if (e.key === "/" && S.view === "home" && document.activeElement !== (S.dom && S.dom.input)) {
      if (S.dom && S.dom.input) { S.dom.input.focus(); e.preventDefault(); }
    } else if (e.key === "Escape") {
      if (S.view === "detail") location.hash = "#/";
      else if (S.dom && S.dom.input && S.dom.input.value) { S.dom.input.value = ""; onSearchInput(""); }
    }
  }

  boot();
})();
