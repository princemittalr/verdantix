// Verdantix dashboard: live twin over WebSocket, 3D campus map, operations panel.
// Backend base URL: "" when this page is served by the FastAPI server, else the tunnel URL from config.js.
const API = String(window.VX_API || "").replace(/\/$/, "");
const media = (p) => (p && p.startsWith("/") ? API + p : p);
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const fmt = (n) => (n == null ? "--" : Number(n).toLocaleString("en-IN"));
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const COLORS = { ok: "#1fa971", warn: "#e8962b", crit: "#e5484d" };
// Building "clay": healthy buildings stay porcelain; only problems and the selection get colour.
const CLAY = { ok: "#f7f8fb", warn: "#f2b25a", crit: "#ec6b6f", sel: "#5672f5" };
const OFM = "https://tiles.openfreemap.org";
// Light basemap built on OpenFreeMap vector tiles (free, no key). If tiles fail, the porcelain ground still renders.
function baseStyle() {
  return {
    version: 8,
    glyphs: `${OFM}/fonts/{fontstack}/{range}.pbf`,
    sources: {
      sat: { type: "raster", tileSize: 256, attribution: "Imagery © Esri", maxzoom: 19,
             tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"] },
    },
    layers: [
      { id: "ground", type: "background", paint: { "background-color": "#e9ebf1" } },
      { id: "sat", type: "raster", source: "sat", layout: { visibility: "none" }, paint: { "raster-saturation": -0.2, "raster-opacity": 0.95 } },
    ],
  };
}
// Roads, greenery and water from OpenFreeMap, added after load so the twin still works if the tiles are unreachable.
function addBasemap() {
  try {
    map.addSource("omt", { type: "vector", url: `${OFM}/planet` });
    const L = [
      { id: "green", type: "fill", source: "omt", "source-layer": "landcover", filter: ["in", ["get", "class"], ["literal", ["grass", "wood", "farmland"]]],
        paint: { "fill-color": "#dbeadf", "fill-opacity": 0.8 } },
      { id: "park", type: "fill", source: "omt", "source-layer": "park", paint: { "fill-color": "#d5e8da", "fill-opacity": 0.7 } },
      { id: "water", type: "fill", source: "omt", "source-layer": "water", paint: { "fill-color": "#d2e1f5" } },
      { id: "road-case", type: "line", source: "omt", "source-layer": "transportation", filter: ["!=", ["get", "class"], "path"],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#d9dce5", "line-width": ["interpolate", ["exponential", 1.6], ["zoom"], 13, 1.5, 18, 22] } },
      { id: "road", type: "line", source: "omt", "source-layer": "transportation", filter: ["!=", ["get", "class"], "path"],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#ffffff", "line-width": ["interpolate", ["exponential", 1.6], ["zoom"], 13, 0.8, 18, 18] } },
      { id: "path", type: "line", source: "omt", "source-layer": "transportation", filter: ["==", ["get", "class"], "path"],
        paint: { "line-color": "#ffffff", "line-width": 1.4, "line-dasharray": [2, 1.5] } },
      { id: "osm-bld", type: "fill", source: "omt", "source-layer": "building", paint: { "fill-color": "#dfe2ea" } },
    ];
    L.forEach((l) => map.addLayer(l, "sat"));
  } catch (e) { console.warn("basemap unavailable", e); }
}
let S = null;              // latest twin state
let campus = null;         // campus geojson
let map, bldChart, weekChart;
let selected = null;       // selected building id
let pin = null;            // student report location
let forecastInfo = null;   // day-ahead forecast for the selected building
const markers = {};
const chatHistory = [];

function toast(msg, ms = 3500) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}

// Operator passcode lives only in this browser tab (sessionStorage), never in the code.
const opToken = () => { try { return sessionStorage.getItem("vx-op") || ""; } catch { return ""; } };
async function api(path, opts = {}) {
  let r;
  const headers = { ...(opts.headers || {}) };
  if (opToken()) headers["X-Operator"] = opToken();
  try { r = await fetch(API + path, { ...opts, headers }); }
  catch { throw new Error(API ? "the live twin backend isn't reachable yet" : "server not reachable"); }
  if (r.status === 401) { openSignin(); throw new Error("Operator sign-in needed for this action"); }
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}
// Buttons that change things report their errors instead of failing silently.
const safely = (fn) => async (...a) => { try { return await fn(...a); } catch (e) { toast(e.message, 5000); } };
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

// ---------------- map ----------------
const FLOOR_H = 3.6;            // metres per storey
const M_LAT = 1 / 111320;       // degrees per metre (latitude)
const mLng = (lat) => 1 / (111320 * Math.cos((lat * Math.PI) / 180));

// Convex hull (monotone chain) of [lng, lat] points.
function hull(pts) {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo.at(-2), lo.at(-1), q) <= 0) lo.pop(); lo.push(q); }
  for (const q of p.slice().reverse()) { while (up.length >= 2 && cross(up.at(-2), up.at(-1), q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
// Campus boundary: hull of every mapped feature, pushed out by `pad` metres.
function campusBoundary(pad = 70) {
  const pts = [];
  campus.features.forEach((f) => (f.geometry.type === "Point" ? [f.geometry.coordinates] : f.geometry.coordinates[0]).forEach((c) => pts.push(c)));
  const h = hull(pts);
  const cx = h.reduce((a, c) => a + c[0], 0) / h.length, cy = h.reduce((a, c) => a + c[1], 0) / h.length;
  const ring = h.map(([x, y]) => {
    const dx = (x - cx) / mLng(cy), dy = (y - cy) / M_LAT, d = Math.hypot(dx, dy) || 1;
    return [x + (dx / d) * pad * mLng(cy), y + (dy / d) * pad * M_LAT];
  });
  return { ring: [...ring, ring[0]], center: [cx, cy] };
}
// Each storey as its own slab with a thin gap, so buildings read as real floors in 3D.
function floorSlabs() {
  const out = [];
  campus.features.filter((f) => f.geometry.type === "Polygon" && f.properties.type !== "sports").forEach((f) => {
    const n = Math.max(f.properties.floors || 1, 1);
    for (let i = 0; i < n; i++)
      out.push({ type: "Feature", geometry: f.geometry,
                 properties: { fid: `${f.properties.id}#${i}`, id: f.properties.id, name: f.properties.name,
                               base: i * FLOOR_H, top: i * FLOOR_H + FLOOR_H - 0.45, roof: i === n - 1 } });
  });
  return { type: "FeatureCollection", features: out };
}
// Tall translucent beam over every open alert, visible from across the campus.
function beaconShape(lng, lat, r = 9, n = 20) {
  const ring = [];
  for (let i = 0; i <= n; i++) {
    const a = (2 * Math.PI * i) / n;
    ring.push([lng + Math.cos(a) * r * mLng(lat), lat + Math.sin(a) * r * M_LAT]);
  }
  return { type: "Polygon", coordinates: [ring] };
}
let orbiting = false, is3D = true, campusView = null;

async function initMap() {
  campus = campus || await api("/api/campus");
  const center = [77.4500, 12.6605];
  map = new maplibregl.Map({
    container: "map",
    style: baseStyle(),
    attributionControl: { compact: true, customAttribution: "© OpenFreeMap © OpenStreetMap" },
    center, zoom: 16.4, pitch: 55, bearing: -20, maxZoom: 19.5,
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "bottom-right");
  map.on("load", () => {
    addBasemap();
    // Campus-only view: everything outside the DSU boundary is dimmed and the camera stays on campus.
    const { ring, center: cc } = campusBoundary(70);
    const xs = ring.map((c) => c[0]), ys = ring.map((c) => c[1]);
    const bounds = [[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]];
    const padLng = (bounds[1][0] - bounds[0][0]) * 0.9, padLat = (bounds[1][1] - bounds[0][1]) * 0.9;
    map.setMaxBounds([[bounds[0][0] - padLng, bounds[0][1] - padLat], [bounds[1][0] + padLng, bounds[1][1] + padLat]]);
    map.fitBounds(bounds, { padding: { top: 90, bottom: 70, left: 430, right: 280 }, pitch: 0, bearing: -20, duration: 0 });
    map.setMinZoom(Math.max(map.getZoom() - 0.8, 14));
    map.jumpTo({ pitch: 58 });
    campusView = { center: map.getCenter(), zoom: map.getZoom(), pitch: 58, bearing: -20 };
    map.setLight({ anchor: "map", position: [1.4, 215, 35], intensity: 0.32, color: "#ffffff" });

    map.addSource("mask", { type: "geojson", data: { type: "Feature", properties: {},
      geometry: { type: "Polygon", coordinates: [[[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]], ring.slice().reverse()] } } });
    map.addLayer({ id: "mask", type: "fill", source: "mask", paint: { "fill-color": "#e9ebf1", "fill-opacity": 0.72 } });
    map.addSource("boundary", { type: "geojson", data: { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: ring } } });
    map.addLayer({ id: "boundary-glow", type: "line", source: "boundary", paint: { "line-color": "#3e5bf2", "line-width": 12, "line-opacity": 0.10, "line-blur": 8 } });
    map.addLayer({ id: "boundary", type: "line", source: "boundary", paint: { "line-color": "#3e5bf2", "line-width": 1.4, "line-opacity": 0.6, "line-dasharray": [2, 2] } });

    map.addSource("campus", { type: "geojson", data: campus, promoteId: "id" });
    map.addLayer({ id: "sports", type: "fill", source: "campus", filter: ["==", ["get", "type"], "sports"],
                   paint: { "fill-color": "#bfe5cf", "fill-opacity": 0.85, "fill-outline-color": "#9fd4b6" } });
    map.addLayer({ id: "footprint", type: "line", source: "campus",
                   filter: ["all", ["==", ["geometry-type"], "Polygon"], ["!=", ["get", "type"], "sports"]],
                   paint: { "line-color": "#c9cdd9", "line-width": 1, "line-opacity": 0.8 } });
    map.addSource("floors", { type: "geojson", data: floorSlabs(), promoteId: "fid" });
    const statusColor = ["match", ["coalesce", ["feature-state", "status"], "ok"], "crit", CLAY.crit, "warn", CLAY.warn, CLAY.ok];
    map.addLayer({ id: "bld", type: "fill-extrusion", source: "floors",
                   paint: {
                     "fill-extrusion-color": ["case", ["boolean", ["feature-state", "sel"], false],
                       ["case", ["boolean", ["get", "roof"], false], "#7d93f8", CLAY.sel],
                       ["boolean", ["get", "roof"], false], ["interpolate", ["linear"], 0.45, 0, statusColor, 1, "#ffffff"], statusColor],
                     "fill-extrusion-base": ["get", "base"],
                     "fill-extrusion-height": ["get", "top"],
                     "fill-extrusion-opacity": 0.96,
                     "fill-extrusion-vertical-gradient": true,
                   } });
    map.addSource("beacons", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: "beacons", type: "fill-extrusion", source: "beacons",
                   paint: { "fill-extrusion-color": ["match", ["get", "severity"], "crit", COLORS.crit, COLORS.warn],
                            "fill-extrusion-height": ["get", "h"], "fill-extrusion-base": 0, "fill-extrusion-opacity": 0.45 } });
    let t0 = performance.now();
    setInterval(() => {  // gentle pulse on the alert beams
      if (!map.getLayer("beacons")) return;
      const k = (performance.now() - t0) / 1000;
      map.setPaintProperty("beacons", "fill-extrusion-opacity", 0.5 + 0.25 * Math.sin(k * 2.2));
    }, 120);
    map.addSource("pts", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: "pts", type: "circle", source: "pts",
                   paint: { "circle-radius": ["match", ["get", "type"], "bin", 7, 5],
                            "circle-color": ["match", ["get", "type"], "bin", ["get", "color"], "water_tank", ["coalesce", ["get", "color"], "#5fa8ff"], "camera", "#161a2c", "water_meter", "#3e6fe0", "#9aa0b4"],
                            "circle-stroke-color": "#ffffff", "circle-stroke-width": 2 } });
    map.addSource("route", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: "route", type: "line", source: "route", layout: { visibility: "none" },
                   paint: { "line-color": "#3e5bf2", "line-width": 4, "line-dasharray": [1.5, 1] } });
    map.addSource("hot", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: "hot", type: "circle", source: "hot", layout: { visibility: "none" },
                   paint: { "circle-radius": ["*", ["get", "weight"], 6], "circle-color": "#e5484d", "circle-opacity": 0.16,
                            "circle-stroke-color": "#e5484d", "circle-stroke-width": 1.5 } });

    map.on("click", "bld", (e) => selectBuilding(e.features[0].properties.id));
    map.on("click", "pts", (e) => {
      const p = e.features[0].properties;
      new maplibregl.Popup().setLngLat(e.lngLat).setHTML(`<b>${p.name}</b><br>${p.info || p.type}`).addTo(map);
    });
    map.on("click", (e) => {
      if ($("#tab-students").classList.contains("on")) {
        pin = e.lngLat;
        $("#repZone").value = "";
        if (markers._pin) markers._pin.remove();
        markers._pin = new maplibregl.Marker({ color: "#9fd3ff" }).setLngLat(pin).addTo(map);
        toast("Report location set from map");
      }
    });
    ["bld", "pts"].forEach((l) => {
      map.on("mouseenter", l, () => (map.getCanvas().style.cursor = "pointer"));
      map.on("mouseleave", l, () => (map.getCanvas().style.cursor = ""));
    });
    map.on("mousedown", () => orbiting && toggleOrbit(false));
    map.on("touchstart", () => orbiting && toggleOrbit(false));
    if (S) render(S);
  });
}

// ---------------- walk-in mode: real 360° imagery + live twin HUD ----------------
let walkId = null;
const TOUR = Object.assign({ default: "https://www.dsu.edu.in/virtual-tour/" }, window.VX_TOUR || {});
function openWalk(id = selected || Object.keys(S?.buildings || {})[0]) {
  if (!id) return;
  if (orbiting) toggleOrbit(false);
  walkId = id;
  const url = TOUR[id] || TOUR.default;
  if ($("#walkFrame").dataset.url !== url) { $("#walkFrame").src = url; $("#walkFrame").dataset.url = url; }
  $("#walkBld").value = id;
  $("#walkNote").textContent = TOUR[id] ? "360° view of this building" : "Campus-wide 360° tour: no building-specific scene linked yet";
  $("#walkOpen").href = url;
  $("#walk").hidden = false;
  renderWalk(S);
}
function closeWalk() { $("#walk").hidden = true; walkId = null; }
function renderWalk(st) {
  if (!walkId || !st) return;
  const b = st.buildings[walkId];
  if (!b) return;
  const al = st.alerts.filter((a) => (a.status === "open" || a.status === "assigned") && a.asset.split("/")[0] === walkId);
  const pct = (v, base) => (base ? Math.round((100 * (v - base)) / base) : 0);
  const dp = pct(b.power_kw, b.baseline_kw), dw = pct(b.water_lph, b.water_baseline_lph);
  const tank = Object.values(st.tanks || {}).find((t) => t.feeds.includes(walkId));
  $("#walkHud").className = `walkhud ${b.status}`;
  $("#walkHud").innerHTML = `
    <div class="wh-top"><span class="dot ${b.status}"></span><b>${esc(b.name)}</b><span class="grade ${b.grade}">${b.grade}</span></div>
    <div class="m">LIVE TWIN · ${new Date(st.time).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}${b.live ? ' · <span class="live-tag">IoT</span>' : ""}</div>
    <div class="wh-grid">
      <div><span>Power</span><b>${fmt(b.power_kw)} kW</b><em class="${dp > 15 ? "bad" : ""}">${dp >= 0 ? "+" : ""}${dp}% vs normal</em></div>
      <div><span>Water</span><b>${fmt(b.water_lph)} L/h</b><em class="${dw > 50 ? "bad" : ""}">${dw >= 0 ? "+" : ""}${dw}% vs normal</em></div>
      <div><span>Green score</span><b>${b.score}/100</b></div>
      ${tank ? `<div><span>${esc(tank.name)}</span><b>${tank.level_pct}%</b><em class="${tank.overflow_lph ? "bad" : ""}">pump ${tank.pump_on ? "ON" : "off"}${tank.overflow_lph ? " · spilling" : ""}</em></div>` : ""}
    </div>
    ${al.length ? al.map((a) => `<div class="wh-alert ${a.severity}"><b>${a.id}</b> ${esc(a.title)}<br><small>₹${fmt(a.inr_per_day)}/day · ${a.status}${a.work_order ? " · " + a.work_order : ""}</small>
        ${a.status === "open" ? `<button data-assign="${a.id}">Assign</button>` : ""}</div>`).join("")
      : `<div class="wh-ok">No open issues in this building</div>`}`;
}

function toggleOrbit(on = !orbiting) {
  orbiting = on;
  $("#orbitBtn").classList.toggle("on", on);
  if (!on) { map.stop(); return; }
  if (!is3D) toggle3D(true);
  const spin = () => {
    if (!orbiting) return;
    map.easeTo({ bearing: map.getBearing() + 40, duration: 8000, easing: (t) => t });
    map.once("moveend", () => orbiting && spin());
  };
  spin();
}
function toggle3D(on = !is3D) {
  is3D = on;
  $("#viewBtn").querySelector("span").textContent = on ? "2D" : "3D";
  if (!on && orbiting) toggleOrbit(false);
  map.easeTo({ pitch: on ? 58 : 0, bearing: on ? -20 : 0, duration: 900 });
}
function resetView() {
  if (orbiting) toggleOrbit(false);
  if (campusView) map.flyTo({ ...campusView, speed: 0.8 });
}

const bmarks = {};
function updateBadges(st) {
  for (const f of campus.features) {
    const id = f.properties.id, b = st.buildings[id];
    if (!b) continue;
    if (!bmarks[id]) {
      const el = document.createElement("div");
      el.className = "bmark";
      el.innerHTML = `<b></b><span>${esc(b.name)}</span>`;
      el.title = b.name;
      el.onclick = (e) => { e.stopPropagation(); selectBuilding(id); };
      bmarks[id] = { el, m: new maplibregl.Marker({ element: el, anchor: "bottom" }).setLngLat(f.properties.center).addTo(map) };
    }
    const { el } = bmarks[id];
    el.classList.toggle("warn", b.status === "warn");   // classList, not className: keep MapLibre's own marker classes
    el.classList.toggle("crit", b.status === "crit");
    el.classList.toggle("sel", id === selected);
    el.querySelector("b").textContent = b.score;
  }
}

function updateMap(st) {
  if (!map || !map.getSource("pts")) return;
  updateBadges(st);
  const floorsOf = Object.fromEntries(campus.features.map((f) => [f.properties.id, Math.max(f.properties.floors || 1, 1)]));
  for (const [id, b] of Object.entries(st.buildings)) {
    map.setFeatureState({ source: "campus", id }, { status: b.status, sel: id === selected });
    for (let i = 0; i < (floorsOf[id] || 1); i++) map.setFeatureState({ source: "floors", id: `${id}#${i}` }, { status: b.status, sel: id === selected });
  }
  const openA = st.alerts.filter((a) => a.status === "open" || a.status === "assigned");
  map.getSource("beacons")?.setData({ type: "FeatureCollection", features: openA.map((a) => ({
    type: "Feature", geometry: beaconShape(a.lng, a.lat), properties: { severity: a.severity, h: a.severity === "crit" ? 95 : 60 } })) });
  const pts = campus.features.filter((f) => f.geometry.type === "Point").map((f) => {
    const p = { ...f.properties };
    const bin = st.bins[p.id];
    if (bin) { p.color = COLORS[bin.status]; p.info = `Fill ${bin.fill}% · full in ${bin.eta_full_h} h · ${bin.source}`; }
    const tank = st.tanks?.[p.id];
    if (tank) { p.color = tank.status === "ok" ? "#3e6fe0" : COLORS[tank.status]; p.info = `Level ${tank.level_pct}% · pump ${tank.pump_on ? "ON" : "off"}${tank.overflow_lph ? ` · overflowing ${fmt(tank.overflow_lph)} L/h` : ""}`; }
    return { type: "Feature", geometry: f.geometry, properties: p };
  });
  map.getSource("pts").setData({ type: "FeatureCollection", features: pts });

  const open = st.alerts.filter((a) => a.status === "open" || a.status === "assigned");
  const keep = new Set(open.map((a) => a.id));
  for (const id of Object.keys(markers)) if (id !== "_pin" && !keep.has(id)) { markers[id].remove(); delete markers[id]; }
  for (const a of open) {
    if (markers[a.id]) continue;
    const el = document.createElement("div");
    el.className = `pulse ${a.severity}`;
    el.title = a.title;
    markers[a.id] = new maplibregl.Marker({ element: el, offset: [0, 18] }).setLngLat([a.lng, a.lat])
      .setPopup(new maplibregl.Popup({ offset: 16 }).setHTML(`<b>${a.id}: ${esc(a.title)}</b><br>${esc(a.detail)}<br>₹${fmt(a.inr_per_day)} a day`))
      .addTo(map);
  }
}

// ---------------- rendering ----------------
function render(st) {
  S = st;
  const t = new Date(st.time);
  $("#simTime").textContent = t.toLocaleString("en-IN", { weekday: "short", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
  const k = st.kpis;
  $("#kScore").textContent = k.green_score;
  $("#ring").style.setProperty("--p", k.green_score);
  $("#ring").style.setProperty("--ring", k.green_score >= 85 ? COLORS.ok : k.green_score >= 70 ? "#7fb81f" : k.green_score >= 55 ? COLORS.warn : COLORS.crit);
  $("#kPower").textContent = fmt(k.power_kw);
  $("#kEnergy").textContent = fmt(k.energy_today_kwh);
  $("#kCost").textContent = fmt(k.cost_today_inr);
  $("#kCo2").textContent = fmt(k.co2_today_kg);
  $("#kWater").textContent = fmt(k.water_today_kl);
  $("#kAlerts").textContent = k.open_alerts;
  $("#kWaste").textContent = k.waste_inr_per_day ? `₹${fmt(k.waste_inr_per_day)}/day at risk` : "";
  const tb = st.tariff || {};
  $("#tariff").textContent = tb.band ? `${tb.band[0].toUpperCase() + tb.band.slice(1)} tariff ₹${tb.inr_per_kwh}/kWh` : "--";
  $("#tariff").className = `tariff ${tb.band || ""}`;
  $("#iot").hidden = !st.live_sensors;
  $("#iot").textContent = `${st.live_sensors} live sensor${st.live_sensors === 1 ? "" : "s"}`;
  announce(st);
  renderBoard(st);
  renderAlerts(st);
  renderBuilding(st);
  updateMap(st);
  renderWalk(st);
}

// Toast each new critical alert once (not the ones already open when the page loads).
const seenAlerts = new Set();
let firstRender = true;
function announce(st) {
  for (const a of st.alerts) {
    if (seenAlerts.has(a.id)) continue;
    seenAlerts.add(a.id);
    if (!firstRender && a.status === "open" && (a.severity === "crit" || a.source === "student")) {
      toast(`New ${a.severity === "crit" ? "critical " : ""}alert ${a.id}: ${a.title}`, 5000);
      beep();
    }
  }
  firstRender = false;
}
function beep() {
  try {
    const ctx = beep.ctx || (beep.ctx = new AudioContext());
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = 880; g.gain.value = 0.04;
    o.connect(g).connect(ctx.destination); o.start(); o.stop(ctx.currentTime + 0.12);
  } catch { /* audio blocked until the user interacts; that's fine */ }
}
function flyTo(lng, lat, zoom = 17.6) { if (map) map.flyTo({ center: [lng, lat], zoom, pitch: 60, speed: 0.8 }); }

function renderBoard(st) {
  const rows = Object.entries(st.buildings).sort((a, b) => b[1].score - a[1].score);
  $("#board").innerHTML = `<h4>Greenest buildings</h4>` + rows.map(([id, b], i) =>
    `<div class="r" data-bld="${id}"><span>${i + 1}</span><span>${esc(b.name)}</span><span>${b.score}</span><span class="grade ${b.grade}">${b.grade}</span></div>`).join("");
}

const PROV = {  // where each incident's evidence comes from
  "sensor-ml": ["Dataset replay + ML", "replay"], camera: ["AI-inferred (camera)", "ai"], student: ["Manual report, AI-verified", "manual"],
  "student-unverified": ["Manual report, unverified", "manual"], iot: ["Measured (sensor)", "measured"], "demo-scenario": ["Simulated demo", "sim"],
};
const prov = (src) => PROV[src] || [src, "replay"];
const TEAMS = { water: "Plumbing", energy: "Electrical", waste: "Housekeeping", resource: "Facilities" };
const OPEN = (a) => a.status === "open" || a.status === "assigned";
const isClosed = (a) => ["resolved", "self-cleared", "dismissed", "rejected"].includes(a.status);
const when = (iso) => (iso ? iso.replace("T", " ").slice(5, 16) : "");
const filt = { kind: "all", status: "open", source: "all" };

function urgency(a, o) {
  const sev = a.severity === "crit" ? 3 : 1;
  return sev * (1 + (a.inr_per_day || 0) / 1000) * (o?.overdue ? 2 : 1) * (a.source === "student-unverified" ? 0.6 : 1);
}
function nextAction(a, o) {
  if (!o) return ["Assign to " + (TEAMS[a.kind] || "Facilities"), `data-assign="${a.id}"`];
  if (o.status === "assigned" || o.status === "reopened") return ["Acknowledge", `data-ostatus="${o.id}:acknowledged"`];
  if (o.status === "acknowledged") return ["Start work", `data-ostatus="${o.id}:in_progress"`];
  return ["Upload after-fix photo", `data-proofpick="${o.id}"`];
}
function renderPriority(st) {
  const items = st.alerts.filter(OPEN).map((a) => ({ a, o: a.work_order ? st.orders.find((x) => x.id === a.work_order) : null }))
    .sort((x, y) => urgency(y.a, y.o) - urgency(x.a, x.o)).slice(0, 3);
  $("#priority").innerHTML = items.length ? `<h3>Priority actions</h3>` + items.map(({ a, o }) => {
    const [label, attr] = nextAction(a, o);
    const [pv] = prov(a.source);
    return `<div class="prio ${a.severity}" data-fly="${a.lng},${a.lat}">
      <div class="t"><span class="aid">${a.id}</span>${esc(a.title)}</div>
      <div class="m"><span>about ₹${fmt(a.inr_per_day)} a day (estimated)</span><span class="pill prov">${pv}</span>
        <span>${o ? `${o.team}, due ${when(o.deadline)}${o.overdue ? ' <b class="late">overdue</b>' : ""}` : `${TEAMS[a.kind]} not assigned yet`}</span></div>
      <button class="dark" ${attr}>${label}</button></div>`;
  }).join("") : "";
}
function timeline(h) {
  if (!h?.length) return "";
  return `<details class="tl"><summary>Timeline (${h.length})</summary><ol>${h.slice().reverse().map((e) =>
    `<li><b>${esc(e.event)}</b> <span>${when(e.at)} · ${esc(e.by)}</span>${e.detail ? `<em>${esc(e.detail)}</em>` : ""}</li>`).join("")}</ol></details>`;
}
function feedbackBtns(a) {
  if (a.feedback) return `<div class="m">Labelled <b>${a.feedback.verdict.replace("_", " ")}</b> by ${esc(a.feedback.by)}${a.feedback.reason ? `: ${esc(a.feedback.reason)}` : ""}</div>`;
  if (a.source === "demo-scenario") return "";
  return `<div class="fb">Was this right? <button data-fb="${a.id}:correct">Correct</button><button data-fb="${a.id}:false_positive">False alarm</button>
    <button data-fb="${a.id}:duplicate">Duplicate</button><button data-fb="${a.id}:uncertain">Needs inspection</button></div>`;
}
function alertCard(a) {
  const o = a.work_order ? S.orders.find((x) => x.id === a.work_order) : null;
  const [pv, pc] = prov(a.source);
  const score = a.evidence?.score != null ? `<span>model score ${Number(a.evidence.score).toFixed(2)}${a.evidence.threshold ? ` (threshold ${a.evidence.threshold})` : ""}</span>` : "";
  return `<div class="card clickable ${isClosed(a) ? "ok faded" : a.severity}" data-fly="${a.lng},${a.lat}">
    <div class="t"><span class="aid">${a.id}</span>${esc(a.title)}</div>
    <div class="d">${esc(a.detail)}</div>
    <div class="m"><span class="pill prov ${pc}">${pv}</span><span>detected ${when(a.created)}</span>${a.updated !== a.created ? `<span>updated ${when(a.updated)}</span>` : ""}</div>
    <div class="m"><span>about ₹${fmt(a.inr_per_day)}/day, ${a.co2_kg_per_day} kg CO₂/day (estimated)</span>${score}</div>
    <div class="m"><span class="st st-${a.status}">${a.status.replace("-", " ")}</span>${o ? `<span>${o.id}: ${o.team}, ${o.status.replace(/_/g, " ")}, due ${when(o.deadline)}${o.overdue ? ' <b class="late">overdue</b>' : ""}</span>` : ""}</div>
    ${a.evidence?.image ? `<img src="${media(a.evidence.image)}" alt="Evidence photo" loading="lazy">` : ""}
    ${a.status === "open" ? `<div class="act"><button data-assign="${a.id}">Assign to ${TEAMS[a.kind] || "Facilities"}</button>
      <button class="ghost" data-dismiss="${a.id}">Dismiss</button></div>` : ""}
    ${isClosed(a) ? `<div class="act"><button class="ghost" data-reopen="${a.id}">Reopen</button></div>` : ""}
    ${feedbackBtns(a)}
    ${timeline(a.history)}
  </div>`;
}

function orderCard(o) {
  const flow = { assigned: ["acknowledged", "Acknowledge"], reopened: ["acknowledged", "Acknowledge"], acknowledged: ["in_progress", "Start work"] }[o.status];
  const live = !["done", "cancelled"].includes(o.status);
  return `<div class="card ${o.status === "done" ? "ok" : o.status === "cancelled" ? "faded" : o.overdue ? "crit" : "warn"}">
    <div class="t"><span class="aid">${o.id}</span>${o.team}: ${esc(o.title)}</div>
    <div class="m"><span class="st st-${o.status}">${o.status.replace(/_/g, " ")}</span><span>due ${when(o.deadline)}</span>${o.overdue ? '<b class="late">overdue, escalated to supervisor</b>' : ""}</div>
    ${o.status === "done" ? `<div class="m">Fix verified from photo (model score ${o.proof?.issue_score})</div>` : ""}
    ${live ? `<div class="act">${flow ? `<button data-ostatus="${o.id}:${flow[0]}">${flow[1]}</button>` : ""}
      ${["in_progress", "acknowledged", "assigned", "reopened"].includes(o.status) ? `<button class="ghost" data-proofpick="${o.id}">Upload after-fix photo</button>` : ""}
      <button class="ghost" data-comment="${o.id}">Comment</button><button class="ghost" data-ostatus="${o.id}:cancelled">Cancel</button></div>` : ""}
    ${o.status === "done" ? `<div class="act"><button class="ghost" data-ostatus="${o.id}:reopened">Reopen</button></div>` : ""}
    ${o.comments?.length ? `<div class="m">${o.comments.slice(-2).map((c) => `“${esc(c.text)}”`).join(" ")}</div>` : ""}
    ${timeline(o.history)}
  </div>`;
}

function renderAlerts(st) {
  renderPriority(st);
  let list = st.alerts.filter((a) => (filt.status === "all" || (filt.status === "open" ? OPEN(a) : isClosed(a)))
    && (filt.kind === "all" || a.kind === filt.kind) && (filt.source === "all" || prov(a.source)[1] === filt.source));
  const openList = $("#alertList").querySelectorAll("details[open]");
  const keepOpen = new Set([...openList].map((d) => d.closest("[data-fly]")?.querySelector(".aid")?.textContent));
  $("#alertList").innerHTML = list.length ? list.slice(0, 40).map(alertCard).join("")
    : `<div class="sub">${filt.status === "open" ? "No open issues match. The twin is watching every meter, camera and report." : "Nothing matches these filters."}</div>`;
  $("#alertList").querySelectorAll(".card").forEach((c) => { if (keepOpen.has(c.querySelector(".aid")?.textContent)) c.querySelector("details")?.setAttribute("open", ""); });
  const ordersOpen = new Set([...$("#orderList").querySelectorAll("details[open]")].map((d) => d.closest(".card")?.querySelector(".aid")?.textContent));
  $("#orderList").innerHTML = st.orders.length ? st.orders.slice(0, 30).map(orderCard).join("") : `<div class="sub">No work orders yet. Assign an alert to create one.</div>`;
  $("#orderList").querySelectorAll(".card").forEach((c) => { if (ordersOpen.has(c.querySelector(".aid")?.textContent)) c.querySelector("details")?.setAttribute("open", ""); });
}
$("#alertFilters").addEventListener("change", (e) => { filt[e.target.name] = e.target.value; if (S) renderAlerts(S); });

async function selectBuilding(id) {
  selected = id;
  $("#bldSelect").value = id;
  if (S && map) updateMap(S);
  switchTab("building");
  const s = await api(`/api/building/${id}/series`);
  const pad = (n) => Array(n).fill(null);
  const fc = s.forecast || [];
  const data = { labels: [...s.t, ...(s.t_fc || [])], datasets: [
    { label: "Actual", data: [...s.kwh, ...pad(fc.length)], borderColor: "#161a2c", backgroundColor: "rgba(22,26,44,.06)", fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2 },
    { label: "Normal", data: [...s.baseline, ...pad(fc.length)], borderColor: "#9aa0b4", borderDash: [4, 4], pointRadius: 0, tension: 0.35, borderWidth: 1.5 },
    { label: "AI forecast", data: [...pad(s.kwh.length - 1), s.kwh[s.kwh.length - 1], ...fc], borderColor: "#3e5bf2", backgroundColor: "rgba(62,91,242,.08)", fill: true, borderDash: [5, 3], pointRadius: 0, tension: 0.35, borderWidth: 2 },
  ] };
  if (bldChart) { bldChart.data = data; bldChart.update(); }
  else bldChart = new Chart($("#bldChart"), { type: "line", data, options: chartOpts("kWh per hour: last 24 h and next 24 h") });
  forecastInfo = s.forecast ? { id, kwh: s.forecast_kwh_24h, peak: s.forecast_peak_kw, at: s.forecast_peak_at, mape: s.forecast_mape_pct } : null;
  if (S) renderBuilding(S);
}

function chartOpts(title) {
  const f = { family: "Geist, system-ui, sans-serif", size: 11 };
  return { responsive: true, animation: false,
           plugins: { legend: { labels: { color: "#3a3f57", boxWidth: 10, boxHeight: 10, usePointStyle: true, font: f } },
                      title: { display: !!title, text: title, color: "#6e7389", font: { ...f, size: 12, weight: "600" }, align: "start" } },
           scales: { x: { ticks: { color: "#6e7389", maxTicksLimit: 8, font: f }, grid: { display: false }, border: { display: false } },
                     y: { ticks: { color: "#6e7389", font: f, maxTicksLimit: 5 }, grid: { color: "#e4e6ee" }, border: { display: false } } } };
}

async function loadSensors() {
  try {
    const r = await api("/api/sensors");
    $("#sensorList").innerHTML = r.sensors.length ? r.sensors.map((x) => `<div class="card ${x.status === "fresh" ? "ok" : x.status === "stale" ? "warn" : "crit"}">
      <div class="t">${esc(x.sensor)} <span class="pill prov ${x.status === "fresh" ? "measured" : "stale"}">${x.status}</span></div>
      <div class="m"><span>${esc(x.asset || "")} ${esc(x.metric || "")} = ${x.value ?? "–"}</span><span>last reading ${x.age_s == null ? "never" : x.age_s + " s ago"}</span>
      <span>${x.readings} ok, ${x.rejected} rejected${x.last_error ? ` (${esc(x.last_error)})` : ""}</span></div></div>`).join("")
      : `<div class="sub">No real sensors connected yet. Buildings show dataset replay. ESP32 nodes post to /api/ingest; a sensor that stops reporting shows here as stale or offline, never as healthy.</div>`;
  } catch { /* offline */ }
}

function renderBuilding(st) {
  const id = selected || Object.keys(st.buildings)[0];
  const b = st.buildings[id];
  if (!b) return;
  $("#bldInfo").innerHTML = [
    ["Power", `${fmt(b.power_kw)} kW`], ["Expected", `${fmt(b.baseline_kw)} kW`], ["Score", `${b.score} (${b.grade})`],
    ["Water", `${fmt(b.water_lph)} L/h`], ["Water normal", `${fmt(b.water_baseline_lph)} L/h`], ["Alerts", b.alerts.length],
    ...(forecastInfo && forecastInfo.id === id ? [["Next 24 h (AI)", `${fmt(forecastInfo.kwh)} kWh`], ["Forecast peak", `${fmt(forecastInfo.peak)} kW`], ["Peak at", forecastInfo.at]] : []),
  ].map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("") +
    `<div class="sub srcline">${b.live ? '<span class="pill prov measured">Measured</span> power from a live sensor'
      : b.stale_sensor ? '<span class="pill prov stale">Sensor stale</span> no recent reading, showing dataset replay'
      : '<span class="pill prov replay">Dataset replay</span> BDG2 meter rescaled to this building'}; time ${S ? when(S.time) : ""} (twin clock).
      Score: 100 minus penalties for load and water above normal and for open alerts. Forecast: model, not measured.</div>`;
  $("#tankList").innerHTML = Object.entries(st.tanks || {}).map(([tid, t]) => `<div class="card clickable ${t.status}" data-fly="${t.lng},${t.lat}">
      <div class="tank"><div class="tankviz ${t.status}"><i style="height:${t.level_pct}%"></i></div>
      <div><div class="t">${t.name} <span class="pill prov ${t.live ? "measured" : "sim"}">${t.live ? "Measured level" : "Simulated"}</span></div>
      <div class="m"><span>${t.level_pct}% · ${t.level_kl}/${t.capacity_kl} kL</span><span>pump ${t.pump_on ? "ON" : "off"}</span>
      <span>use ${fmt(t.demand_lph)} L/h</span>${t.overflow_lph ? `<span style="color:var(--crit)">spilling ${fmt(t.overflow_lph)} L/h</span>` : ""}</div>
      <div class="m"><span>peak-hour pumping today: ${t.peak_pump_min_today} min</span></div></div></div></div>`).join("") || `<div class="sub">No tanks mapped.</div>`;
  const waste = new Set(st.alerts.filter((a) => a.rule === "empty-room" && a.status !== "resolved" && a.status !== "self-cleared").map((a) => a.asset));
  $("#roomGrid").innerHTML = Object.entries(st.rooms).map(([r, v]) => `<div class="room ${waste.has(r) ? "waste" : ""}">
      <b>${r.split("/")[1]}</b>AC ${v.ac_kw} kW<br>Light ${v.light_kw} kW<br>${v.people != null ? "Camera: " + v.people + " people" : "occ " + Math.round(v.occ_prob * 100) + "%"}<br>${v.temp_c}°C</div>`).join("");
  $("#binList").innerHTML = Object.entries(st.bins).sort((a, b) => b[1].fill - a[1].fill).map(([id, v]) => `<div class="card ${v.status}">
      <div class="t">${id} <span class="pill prov ${({ iot: "measured", camera: "ai", student: "manual", model: "sim" })[v.source] || "sim"}">${({ iot: "Measured", camera: "AI-inferred", student: "Manual report", model: "Simulated", "demo-scenario": "Simulated demo" })[v.source] || v.source}</span></div>
      <div class="bar"><i class="${v.status}" style="width:${v.fill}%"></i></div>
      <div class="m"><span>${v.fill}% full</span><span>${v.fill >= 100 ? "overflowing" : "full in ~" + v.eta_full_h + " h"}</span></div></div>`).join("");
}

// ---------------- tabs + controls ----------------
function switchTab(name) {
  $$("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("on", t.id === `tab-${name}`));
  if (name === "tour" && !$("#tourFrame").src) $("#tourFrame").src = "https://www.dsu.edu.in/virtual-tour/";
}
$("#tabs").addEventListener("click", (e) => e.target.dataset.tab && switchTab(e.target.dataset.tab));
$("#speedSeg").addEventListener("click", async (e) => {
  if (!e.target.dataset.speed) return;
  $$("#speedSeg button").forEach((b) => b.classList.toggle("on", b === e.target));
  try { await post("/api/clock", { speed: Number(e.target.dataset.speed) }); } catch (err) { toast(err.message); }
});
$("#jumpNight").onclick = safely(async () => { await post("/api/clock", { jump: "night" }); toast("Jumped to 2 AM: watch night-time water and after-hours power"); });
$("#jumpMorning").onclick = safely(async () => { await post("/api/clock", { jump: "morning" }); toast("Jumped to 9 AM"); });
$$(".demo button").forEach((b) => (b.onclick = safely(async () => {
  const r = await post(`/api/scenario/${b.dataset.scn}`);
  toast(b.dataset.scn === "clear" ? "Demo scenarios cleared" : `Simulated ${r.kind.replace("_", " ")} at ${r.target} (labelled as a demo)`);
})));
$("#bldSelect").onchange = (e) => selectBuilding(e.target.value);

const refresh = async () => { try { render(await api("/api/state")); } catch { /* the socket will catch up */ } };
document.addEventListener("click", safely(async (e) => {
  const t = e.target.closest?.("button, a, summary, details, input, label");
  const d = t?.dataset || {};
  const card = e.target.closest?.("[data-fly]");
  if (card && !t) {
    const [lng, lat] = card.dataset.fly.split(",").map(Number);
    flyTo(lng, lat);
  }
  if (!t) return;
  if (d.alertlink) e.preventDefault();
  if (d.assign) { const o = await post(`/api/alerts/${d.assign}/order`); toast(`${o.id} assigned to ${o.team}, due ${when(o.deadline)}`); refresh(); }
  if (d.dismiss) { const why = prompt("Why dismiss this alert? (recorded in the audit log)", ""); if (why === null) return; await post(`/api/alerts/${d.dismiss}/dismiss`, { note: why }); toast(`${d.dismiss} dismissed`); refresh(); }
  if (d.reopen) { await post(`/api/alerts/${d.reopen}/reopen`); toast(`${d.reopen} reopened`); refresh(); }
  if (d.fb) {
    const [aid, verdict] = d.fb.split(":");
    const reason = verdict === "correct" ? "" : prompt("Reason (optional, recorded):", "") ?? null;
    if (reason === null) return;
    await post(`/api/alerts/${aid}/feedback`, { verdict, reason }); toast(`${aid} labelled ${verdict.replace("_", " ")}`); refresh();
  }
  if (d.ostatus) {
    const [oid, status] = d.ostatus.split(":");
    const note = status === "cancelled" || status === "reopened" ? prompt(`Reason to ${status === "cancelled" ? "cancel" : "reopen"} ${oid}:`, "") : "";
    if (note === null) return;
    await post(`/api/orders/${oid}/status`, { status, note: note || "" }); toast(`${oid}: ${status.replace("_", " ")}`); refresh();
  }
  if (d.comment) { const text = prompt(`Comment on ${d.comment}:`, ""); if (!text) return; await post(`/api/orders/${d.comment}/comment`, { note: text }); refresh(); }
  if (d.proofpick) { proofFor = d.proofpick; $("#proofFile").click(); }
  if (d.alertlink) { const a = S?.alerts.find((x) => x.id === d.alertlink); if (a) { switchTab("alerts"); filt.status = "all"; $("#alertFilters").elements.status.value = "all"; renderAlerts(S); flyTo(a.lng, a.lat); } }
}));
let proofFor = null;

// ---------------- operator sign-in ----------------
function openSignin() { if (!$("#signin").open) { $("#opPass").value = ""; $("#opMsg").textContent = ""; $("#signin").showModal(); } }
async function refreshOperator() {
  try {
    const w = await api("/api/whoami");
    const on = w.operator && (w.auth_enabled ? !!opToken() : true);
    document.body.classList.toggle("is-op", on);
    $("#opBtn").querySelector("span").textContent = on ? (w.auth_enabled ? "Operator" : "Operator (auth off)") : "Operator sign-in";
    $("#opBtn").classList.toggle("on", on);
  } catch { /* offline */ }
}
$("#provMore").onclick = (e) => { e.preventDefault(); $("#aboutBtn").click(); };
$("#opBtn").onclick = () => {
  if (document.body.classList.contains("is-op") && opToken()) {
    if (confirm("Sign out of operator mode?")) { try { sessionStorage.removeItem("vx-op"); } catch {} refreshOperator(); }
    return;
  }
  openSignin();
};
$("#signinForm").onsubmit = async (e) => {
  e.preventDefault();
  const pass = $("#opPass").value.trim();
  try {
    const r = await fetch(API + "/api/signin", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ passcode: pass }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Wrong passcode");
    try { sessionStorage.setItem("vx-op", pass); } catch {}
    $("#signin").close(); toast("Signed in as operator. Changes are recorded in the audit log."); refreshOperator();
  } catch (err) { $("#opMsg").textContent = err.message; }
};
$("#proofFile").addEventListener("change", safely(async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f || !proofFor) return;
  const fd = new FormData();
  fd.append("file", f);
  toast("Checking the after-fix photo…", 8000);
  const r = await api(`/api/orders/${proofFor}/proof`, { method: "POST", body: fd });
  toast(r.message, 6000);
  refresh();
}));
$("#viewBtn").onclick = () => toggle3D();
$("#showSat").onchange = (e) => {
  if (!map?.getLayer("sat")) return;
  map.setLayoutProperty("sat", "visibility", e.target.checked ? "visible" : "none");
  map.setPaintProperty("mask", "fill-opacity", e.target.checked ? 0.55 : 0.72);
};
$("#board").addEventListener("click", (e) => { const r = e.target.closest("[data-bld]"); if (r) selectBuilding(r.dataset.bld); });
$("#walkBtn").onclick = () => openWalk();
$("#bldWalk").onclick = () => openWalk(selected);
$("#walkBack").onclick = () => { const id = walkId; closeWalk(); if (id) selectBuilding(id); };
$("#walkBld").onchange = (e) => { selectBuilding(e.target.value); openWalk(e.target.value); };
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && walkId) closeWalk(); });
$("#orbitBtn").onclick = () => toggleOrbit();
$("#homeBtn").onclick = () => resetView();
$("#showRoute").onchange = async (e) => {
  map.setLayoutProperty("route", "visibility", e.target.checked ? "visible" : "none");
  if (e.target.checked) {
    const r = await api("/api/route");
    map.getSource("route").setData({ type: "Feature", geometry: { type: "LineString", coordinates: r.coords } });
    toast(`Pickup route: ${r.stops.length} bins, ${fmt(r.distance_m)} m`);
  }
};
$("#showHot").onchange = async (e) => {
  map.setLayoutProperty("hot", "visibility", e.target.checked ? "visible" : "none");
  if (e.target.checked) {
    const hs = await api("/api/hotspots");
    map.getSource("hot").setData({ type: "FeatureCollection", features: hs.map((h) => ({ type: "Feature", geometry: { type: "Point", coordinates: [h.lng, h.lat] }, properties: { weight: Math.min(h.weight, 8) } })) });
    toast(hs.length ? `${hs.length} hotspots. Top: ${hs[0].assets.join(", ")} (${hs[0].priority}) — ${hs[0].suggestion}` : "No hotspots yet", 6000);
  }
};

// ---------------- vision ----------------
let vTimer = null, vStream = null;
const vVideo = $("#vVideo"), vImg = $("#vImg"), vCanvas = $("#vCanvas");
$("#vSource").onchange = async (e) => {
  stopLive();
  if (e.target.value === "webcam") return startWebcam();
  if (vStream) { vStream.getTracks().forEach((t) => t.stop()); vStream = null; }
  $("#vFile").accept = e.target.value === "video" ? "video/*" : "image/*";
  $("#vFile").click();
};
$("#vFile").onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  if (f.type.startsWith("video")) { vImg.hidden = true; vVideo.hidden = false; vVideo.srcObject = null; vVideo.src = url; }
  else { vVideo.hidden = true; vImg.hidden = false; vImg.src = url; }
};
async function startWebcam() {
  try {
    vStream = await navigator.mediaDevices.getUserMedia({ video: { width: 960 } });
    vImg.hidden = true; vVideo.hidden = false; vVideo.src = ""; vVideo.srcObject = vStream;
  } catch { toast("Camera not available. Pick a demo video or image instead."); }
}
function frameBlob() {
  const src = vImg.hidden ? vVideo : vImg;
  const w = src.videoWidth || src.naturalWidth, h = src.videoHeight || src.naturalHeight;
  if (!w) return null;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(src, 0, 0, w, h);
  return new Promise((res) => c.toBlob(res, "image/jpeg", 0.8)).then((b) => ({ blob: b, w, h }));
}
async function analyseOnce() {
  const f = await frameBlob();
  if (!f) return toast("No frame yet. Start the webcam or load a video/image.");
  const fd = new FormData();
  fd.append("file", f.blob, "frame.jpg");
  const cam = $("#vCam").value;
  if (cam) {   // CCTV mode: the server decides and raises/clears alerts at the camera's location
    const r = await api(`/api/cctv/${cam}`, { method: "POST", body: fd });
    drawBoxes({ objects: r.objects, size: r.size }, f.w, f.h);
    const name = $("#vCam").selectedOptions[0].textContent;
    $("#vOut").innerHTML = `<div class="cctvres ${r.action.startsWith("alert") ? "hit" : ""}"><b>${esc(name)}</b>:
        ${r.score >= r.threshold ? `sees <b>${esc(r.issue)}</b>` : "nothing wrong in view"}: model score ${r.score.toFixed(2)} (raise at ${r.threshold}), ${r.people} people</div>
      <div>${r.top.map((t) => `<span class="tag">${esc(t.label)}</span>`).join("")}</div>
      ${r.action !== "none" ? `<div><b>${r.action.startsWith("alert") ? "Reported automatically as " + r.action.split(" ")[1] : r.action.startsWith("cleared") ? "Problem gone: closed " + r.action.split(" ")[1] : esc(r.action)}</b></div>` : ""}
      <div class="sub">Model scores are not calibrated probabilities. One incident per camera and problem; it opens after 2 detections and closes after 3 clean frames.</div>`;
    return;
  }
  fd.append("room", $("#vRoom").value);
  fd.append("bin", $("#vBin").value);
  const r = await api("/api/vision/analyze", { method: "POST", body: fd });
  drawBoxes(r, f.w, f.h);
  const c = r.clip;
  $("#vOut").innerHTML = `<div><b>${r.people}</b> people · ${r.bin ? `bin: <b>${r.bin.label}</b> (${Math.round(r.bin.conf * 100)}%)` : "no bin model hit"} · ${r.ms} ms</div>
    <div>Most likely issue: <b>${esc(c.issue)}</b>, model score ${c.issue_score.toFixed(2)}${c.issue_score >= (r.threshold || 0.6) ? " (above threshold)" : " (below threshold)"}</div>
    <div>${c.top.map((t) => `<span class="tag">${esc(t.label)} ${Number(t.p).toFixed(2)}</span>`).join("")}</div>
    <div class="sub">Scores are uncalibrated model outputs, not probabilities. Mode: ${esc(r.models?.clip || "")}</div>
    ${r.events.length ? `<div>Twin updated: ${r.events.join(" · ")}</div>` : ""}`;
}
function drawBoxes(r, w, h) {
  const box = vCanvas.getBoundingClientRect();
  vCanvas.width = box.width; vCanvas.height = box.height;
  const s = Math.min(box.width / r.size[0], box.height / r.size[1]);
  const ox = (box.width - r.size[0] * s) / 2, oy = (box.height - r.size[1] * s) / 2;
  const g = vCanvas.getContext("2d");
  g.clearRect(0, 0, box.width, box.height);
  g.lineWidth = 2; g.font = "12px JetBrains Mono, monospace";
  for (const o of r.objects) {
    const [x1, y1, x2, y2] = o.box;
    const col = o.label.startsWith("bin:") ? (o.label.toLowerCase().includes("overflow") ? "#ff6b5e" : "#f0a948") : o.label === "person" ? "#4cc79e" : "#9fd3ff";
    g.strokeStyle = col; g.fillStyle = col;
    g.strokeRect(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s);
    g.fillText(`${o.label} ${Math.round(o.conf * 100)}%`, ox + x1 * s + 3, oy + y1 * s + 13);
  }
}
function stopLive() { clearInterval(vTimer); vTimer = null; $("#vStart").textContent = "Start live analysis"; }
$("#vStart").onclick = () => {
  if (vTimer) return stopLive();
  if (!vStream && vImg.hidden && !vVideo.src) startWebcam();
  let busy = false;
  vTimer = setInterval(async () => { if (busy) return; busy = true; try { await analyseOnce(); } catch (e) { toast(e.message); } busy = false; }, 1500);
  $("#vStart").textContent = "Stop live analysis";
};
$("#vOnce").onclick = () => analyseOnce().catch((e) => toast(e.message));

// ---------------- copilot ----------------
function md(text) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const lines = esc(text || "").split("\n");
  let html = "", inList = false, inTable = false;
  for (let l of lines) {
    l = l.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
    if (/^\|/.test(l)) {
      if (/^\|[\s\-:|]+\|$/.test(l)) continue;
      if (!inTable) { html += "<table>"; inTable = true; }
      html += "<tr>" + l.split("|").slice(1, -1).map((c) => `<td>${c.trim()}</td>`).join("") + "</tr>";
      continue;
    } else if (inTable) { html += "</table>"; inTable = false; }
    if (/^\s*[-*] /.test(l)) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${l.replace(/^\s*[-*] /, "")}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    if (/^#{1,3} /.test(l)) html += `<h3>${l.replace(/^#+ /, "")}</h3>`;
    else if (l.trim()) html += `<p>${l}</p>`;
  }
  return html + (inList ? "</ul>" : "") + (inTable ? "</table>" : "");
}
const linkAlerts = (html) => html.replace(/\b(A\d{4}|WO\d{4})\b/g, (m) => m.startsWith("A") ? `<a href="#" data-alertlink="${m}">${m}</a>` : m);
function addMsg(text, me, meta) {
  const d = document.createElement("div");
  d.className = `msg ${me ? "me" : ""}`;
  d.innerHTML = me ? text.replace(/</g, "&lt;") : linkAlerts(md(text)) + (meta ? `<small>${meta}</small>` : "");
  $("#chat").appendChild(d);
  d.scrollIntoView({ behavior: "smooth" });
  return d;
}
async function ask(q) {
  addMsg(q, true);
  const wait = addMsg("Checking the twin…", false);
  try {
    const r = await post("/api/copilot", { question: q, history: chatHistory, lang: $("#askLang").value });
    wait.remove();
    addMsg(r.answer, false, `${r.provider}${r.tools.length ? " · tools: " + r.tools.map((t) => t.tool).join(", ") : ""}`);
    chatHistory.push({ role: "user", content: q }, { role: "assistant", content: r.answer });
  } catch (e) { wait.innerHTML = `Couldn't reach the copilot: ${e.message}`; }
}
$("#askForm").onsubmit = (e) => { e.preventDefault(); const q = $("#askInput").value.trim(); if (q) { $("#askInput").value = ""; ask(q); } };
$$(".chips button").forEach((b) => (b.onclick = () => ask(b.textContent)));
// Voice: browser speech recognition (Chrome/Edge), in the selected answer language.
(() => {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn = $("#micBtn");
  if (!SR) { btn.hidden = true; return; }
  let rec = null;
  btn.onclick = () => {
    if (rec) { rec.stop(); return; }
    rec = new SR();
    rec.lang = { en: "en-IN", kn: "kn-IN", hi: "hi-IN" }[$("#askLang").value] || "en-IN";
    rec.interimResults = true;
    btn.classList.add("on");
    rec.onresult = (e) => { $("#askInput").value = [...e.results].map((r) => r[0].transcript).join(""); };
    rec.onend = () => { btn.classList.remove("on"); const q = $("#askInput").value.trim(); rec = null; if (q) { $("#askInput").value = ""; ask(q); } };
    rec.onerror = (e) => toast(`Voice input: ${e.error}`);
    rec.start();
  };
})();

// ---------------- what-if ----------------
async function runSim() {
  const p = { ac_setpoint_delta: +$("#sAc").value, after_hours_cut: $("#sAh").value / 100, solar_kwp: +$("#sSol").value,
              pickups_per_day: +$("#sPk").value, led_retrofit: $("#sLed").checked, fix_leaks: $("#sLeak").checked,
              smart_pump_schedule: $("#sPump").checked };
  $("#vAc").textContent = `${p.ac_setpoint_delta} °C`; $("#vAh").textContent = `${Math.round(p.after_hours_cut * 100)}%`;
  $("#vSol").textContent = `${p.solar_kwp} kWp`; $("#vPk").textContent = p.pickups_per_day;
  const r = await post("/api/simulate", p);
  lastSim = { p, r };
  const lo = (v) => fmt(Math.round(v * r.uncertainty.low_factor)), hi = (v) => fmt(Math.round(v * r.uncertainty.high_factor));
  $("#simOut").innerHTML = `<div class="savings-kind"><b>Potential savings (modelled)</b>, not realised or measured.
      Range ₹${lo(r.inr_saved_year)} to ₹${hi(r.inr_saved_year)} a year (±25%: ${esc(r.uncertainty.why)}).</div>
    <div class="big">
      <div class="stat"><span>Potential per year</span><b>₹${fmt(r.inr_saved_year)}</b></div>
      <div class="stat"><span>CO₂ avoided</span><b>${r.co2_tonnes_saved_year} t</b></div>
      <div class="stat"><span>Energy saved</span><b>${fmt(r.kwh_saved_year)} kWh</b></div>
      <div class="stat"><span>Of campus energy</span><b>${r.pct_energy_saved}%</b></div></div>
    ${paybackTable(r)}
    <div class="list">${Object.entries(r.breakdown_kwh).map(([k, v]) => `<div class="m"><span style="width:110px">${k.replace("_", " ")}</span>
      <div class="bar" style="flex:1"><i style="width:${r.kwh_saved_year ? (100 * v) / r.kwh_saved_year : 0}%"></i></div><span>${fmt(v)} kWh</span></div>`).join("")}</div>
    <div class="sub">Water saved: ${fmt(r.water_kl_saved_year)} kL/yr · bin overflow-hours/yr: ${fmt(r.bin_overflow_hours_year)}${r.bin_opex_inr_year > 0 ? ` (extra pickups cost ₹${fmt(r.bin_opex_inr_year)}/yr)` : ""} ·
      baseline ${fmt(r.baseline_kwh_year)} kWh/yr → projected ${fmt(r.baseline_kwh_year - r.kwh_saved_year)} kWh/yr.
      Method: full year of replayed hourly data; AC share ${Math.round(r.assumptions.ac_share * 100)}% of load (CU-BEMS rooms), BEE ${r.assumptions.ac_saving_per_deg * 100}% per °C,
      tariff ₹${r.assumptions.tariff}/kWh (assumed), grid ${r.assumptions.grid_factor} kg CO₂/kWh (CEA), solar ${r.assumptions.solar_yield} kWh/kWp/day.</div>
    <div class="row"><button class="soft" id="saveScn">Save scenario to compare</button><button class="soft" id="csvScn">Export CSV</button></div>
    ${scenarios.length ? `<table class="payback"><tr><th>Saved scenario</th><th>₹/yr</th><th>t CO₂</th><th>Payback</th></tr>${scenarios.map((x) =>
      `<tr><td>${esc(x.name)}</td><td>₹${fmt(x.r.inr_saved_year)}</td><td>${x.r.co2_tonnes_saved_year}</td><td>${x.r.payback_total_years ?? "–"} yr</td></tr>`).join("")}</table>` : ""}`;
  $("#saveScn").onclick = () => { scenarios.push({ name: describeScn(p), r }); runSim(); };
  $("#csvScn").onclick = () => exportCsv();
}
let lastSim = null;
const scenarios = [];
function describeScn(p) {
  return [p.ac_setpoint_delta && `AC +${p.ac_setpoint_delta}°C`, p.after_hours_cut && `after-hours −${Math.round(p.after_hours_cut * 100)}%`,
          p.led_retrofit && "LED", p.solar_kwp && `${p.solar_kwp} kWp solar`, p.fix_leaks && "fix leaks", p.smart_pump_schedule && "night pumping"]
    .filter(Boolean).join(", ") || "no change";
}
function exportCsv() {
  const all = [...scenarios, ...(lastSim ? [{ name: describeScn(lastSim.p) + " (current)", r: lastSim.r }] : [])];
  const head = "scenario,category,inr_per_year,inr_low,inr_high,kwh_per_year,co2_t_per_year,water_kl_per_year,capex_inr,payback_years,tariff_inr_kwh,grid_kg_co2_kwh";
  const rows = all.map(({ name, r }) => [JSON.stringify(name), "potential (modelled)", r.inr_saved_year, Math.round(r.inr_saved_year * 0.75), Math.round(r.inr_saved_year * 1.25),
    r.kwh_saved_year, r.co2_tonnes_saved_year, r.water_kl_saved_year, r.capex_total_inr, r.payback_total_years ?? "", r.assumptions.tariff, r.assumptions.grid_factor].join(","));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([[head, ...rows].join("\n")], { type: "text/csv" }));
  a.download = "verdantix-what-if.csv"; a.click();
}
function paybackTable(r) {
  const names = { ac_setpoint: "AC setpoint", after_hours: "After-hours cut", led_retrofit: "LED retrofit", solar: "Rooftop solar", fix_leaks: "Fix leaks", smart_pump: "Smart pumps" };
  const rows = Object.keys(names).filter((k) => r.inr_breakdown[k] > 0);
  if (!rows.length) return `<div class="sub">Move a slider to see savings, cost and payback.</div>`;
  const yrs = (v) => (v == null ? "–" : v === 0 ? "instant" : v < 1 ? `${Math.round(v * 12)} mo` : `${v} yr`);
  return `<table class="payback"><tr><th>Measure</th><th>₹ saved/yr</th><th>Capex</th><th>Payback</th></tr>
    ${rows.map((k) => `<tr><td>${names[k]}</td><td>₹${fmt(r.inr_breakdown[k])}</td><td>₹${fmt(r.capex_inr[k])}</td><td>${yrs(r.payback_years[k])}</td></tr>`).join("")}
    <tr><th>Total</th><th>₹${fmt(r.inr_saved_year)}</th><th>₹${fmt(r.capex_total_inr)}</th><th>${yrs(r.payback_total_years)}</th></tr></table>`;
}
["#sAc", "#sAh", "#sSol", "#sPk", "#sLed", "#sLeak", "#sPump"].forEach((s) => $(s).addEventListener("input", runSim));

// ---------------- report ----------------
async function loadLedger() {
  try {
    const l = await api("/api/ledger");
    $("#ledger").innerHTML = `<h3>Impact ledger <small>fixed incidents</small></h3>
      <p class="sub">${esc(l.note)}</p>` + (l.rows.length ? `<table class="payback"><tr><th>Incident</th><th>Fixed in</th><th>Loss rate</th><th>Est. avoided / yr</th><th>Measured</th></tr>
      ${l.rows.map((x) => `<tr><td><a href="#" data-alertlink="${x.incident}">${x.incident}</a> ${esc(x.title)}</td><td>${x.hours_to_fix ?? "–"} h</td><td>₹${fmt(x.loss_rate_inr_per_day)}/day</td>
        <td>₹${fmt(x.est_avoided_inr_per_year)}, ${x.est_avoided_co2_t_per_year} t CO₂</td><td>needs meter data</td></tr>`).join("")}</table>`
      : `<div class="sub">No fixes verified yet. When a work order closes with a verified after-fix photo, it appears here.</div>`);
  } catch { /* offline */ }
}
$("#genReport").onclick = safely(async () => {
  $("#reportOut").innerHTML = "<div class='sub'>Pulling a week of twin data and writing the report…</div>";
  const r = await post("/api/report/weekly");
  $("#pdfLink").href = API + "/api/report/weekly.pdf";
  $("#pdfLink").hidden = false;
  $("#reportOut").innerHTML = md(r.narrative) + `<div class="sub">Written by ${r.provider} from twin data.</div>`;
  const h = await api("/api/history?days=7");
  const data = { labels: h.daily_kwh.map((_, i) => `Day ${i + 1}`), datasets: [{ label: "Campus kWh per day", data: h.daily_kwh, backgroundColor: "#3e5bf2", borderRadius: 6 }] };
  if (weekChart) { weekChart.data = data; weekChart.update(); } else weekChart = new Chart($("#weekChart"), { type: "bar", data, options: chartOpts("Campus kWh per day (dataset replay)") });
  loadLedger();
});

// ---------------- students ----------------
$("#repForm").onsubmit = async (e) => {
  e.preventDefault();
  const f = $("#repFile").files[0];
  const cat = $("#repCat").value;
  if (!f && !cat) return toast("Pick what you found, or add a photo.");
  let loc = pin;
  if ($("#repZone").value) {
    const feat = campus.features.find((x) => x.properties.id === $("#repZone").value);
    const c = feat.geometry.type === "Point" ? feat.geometry.coordinates : feat.properties.center;
    loc = { lng: c[0], lat: c[1] };
  }
  if (!loc) return toast("Pick a place from the list or click the map.");
  const fd = new FormData();
  if (f) fd.append("file", f);
  fd.append("category", cat); fd.append("lat", loc.lat); fd.append("lng", loc.lng); fd.append("note", $("#repNote").value);
  fd.append("reporter", $("#repName").value.trim());
  $("#repResult").innerHTML = `<div class='sub'>${f ? "Checking location, duplicates and photo…" : "Sending…"}</div>`;
  let r;
  try { r = await api("/api/reports", { method: "POST", body: fd }); }
  catch (err) { $("#repResult").innerHTML = `<div class="card warn"><div class="t">Couldn't send</div><div class="d">${esc(err.message)}. Your details are still here; send again.</div></div>`; return; }
  $("#repResult").innerHTML = r.ok
    ? `<div class="card ok"><div class="t">${r.verified ? "Verified" : "Sent, unverified"}: ${r.report.id}, alert ${r.alert}</div>
       <div class="d">${r.report.clip ? `AI sees ${esc(r.report.clip.issue)} (model score ${Number(r.report.clip.issue_score).toFixed(2)})` : "No photo: confirmed when others report the same thing nearby"}${r.credits != null ? `. ${r.credits} green credits` : ""}</div></div>`
    : `<div class="card warn"><div class="t">${r.check === "repeat" ? "Already reported" : "Not sent"}</div><div class="d">${esc(r.message)}</div></div>`;
  if (r.ok) { $("#repForm").reset(); }
  loadReports();
};
async function loadReports() {
  const board = await api("/api/reports/leaderboard");
  $("#repBoard").innerHTML = board.length ? board.map((b, i) => `<div class="m"><span>${i + 1}.</span><span style="flex:1">${esc(b.name)}</span><b>${b.credits}</b></div>`).join("")
    : `<div class="sub">Add a name to earn credits: 10 per verified report, 5 unverified, 2 per confirmation, once per person. Names show as first name and initial.</div>`;
  const rs = await api("/api/reports");
  $("#repList").innerHTML = rs.length ? rs.map((r) => `<div class="card clickable ${r.kind === "water" ? "crit" : "warn"}" data-fly="${r.lng},${r.lat}"><div class="t">${r.id} · ${r.kind} · ${esc(r.zone)} <small class="d">by ${esc(r.reporter || "Anonymous")}</small></div>
    ${r.image ? `<img src="${media(r.image)}" alt="" style="max-width:100%">` : ""}${r.verified === false ? '<div class="m"><span class="pill src-student-unverified">unverified</span></div>' : ""}<div class="m"><span>${r.created.replace("T", " ")}</span><span>+${r.confirmations} confirmations</span><span class="st st-${r.status}">${(r.status || "open").replace("_", " ")}${r.resolution ? ": " + esc(r.resolution) : ""}</span><a href="#" data-alertlink="${r.alert}">${r.alert}</a></div></div>`).join("")
    : `<div class="sub">No student reports yet.</div>`;
}

// ---------------- about ----------------
$("#aboutBtn").onclick = async () => {
  const a = await api("/api/about");
  const m = a.metrics;
  $("#aboutBody").innerHTML = `<h2>Data &amp; models</h2>
    <p class="sub">Real open datasets mapped onto DSU's buildings. Live sensors plug into the same ingest API.</p>
    <h3>Model performance (held-out data)</h3>
    <table><tr><th>Model</th><th>Dataset</th><th>Result</th></tr>
      <tr><td>Occupancy (Random Forest)</td><td>${m.occupancy.dataset}</td><td>accuracy ${m.occupancy.accuracy}, F1 ${m.occupancy.f1} on ${fmt(m.occupancy.test_rows)} rows</td></tr>
      <tr><td>Leak detector (${m.leak.method})</td><td>${m.leak.dataset}</td><td>${m.leak.events_caught}/${m.leak.leak_events} bursts caught within ${m.leak.catch_window_days} days (median delay ${m.leak.median_delay_days} d); precision ${m.leak.precision} over ${m.leak.flagged_nights} flagged nights</td></tr>
      <tr><td>Energy anomalies</td><td>BDG2 hourly meters</td><td>robust z-score vs hour-of-week baseline + Isolation Forest confirmation</td></tr>
      ${m.forecast ? `<tr><td>Day-ahead load forecast (${m.forecast.method})</td><td>${m.forecast.dataset}</td><td>MAPE ${m.forecast.mape_pct}% vs ${m.forecast.naive_mape_pct}% for same-hour-last-week, over ${m.forecast.buildings} buildings</td></tr>` : ""}
      <tr><td>Vision</td><td colspan="2">${Object.values(a.models).join(" · ")}</td></tr></table>
    <h3>Which dataset feeds which DSU building</h3>
    <table><tr><th>DSU building</th><th>Source meter</th><th>Scale</th><th>Water</th></tr>
      ${Object.entries(a.mapping.buildings).map(([k, v]) => `<tr><td>${k}</td><td>${v.source}</td><td>×${v.scale}</td><td>${v.water_source}</td></tr>`).join("")}</table>
    <p class="sub">Rooms: CU-BEMS 2019 (Chulalongkorn). Water network: BattLeDIM 2018 L-Town. Occupancy: UCI. Bins: Roboflow garbage_fill_level (CC BY 4.0).</p>
    <h3>Assumptions</h3>
    <table>${Object.entries(a.assumptions).map(([k, v]) => `<tr><td>${k}</td><td>${typeof v === "object" ? JSON.stringify(v) : v}</td></tr>`).join("")}</table>
    <p class="sub">GenAI: ${Object.entries(a.llm).map(([k, v]) => `${k} ${v ? "on" : "off"}`).join(", ")}; offline rule-based fallback always on.</p>
    <h3>Where each number comes from</h3>
    <table>${Object.entries(a.data_sources).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</table>
    <h3>Vision health</h3>
    <table><tr><td>Mode</td><td>${esc(a.vision_health.mode)}</td></tr><tr><td>Provider</td><td>${esc(a.vision_health.provider)}</td></tr>
      <tr><td>Last success</td><td>${a.vision_health.last_ok || "none yet"}</td></tr><tr><td>Last error</td><td>${esc(a.vision_health.last_error || "none")}</td></tr>
      <tr><td>Calls</td><td>${a.vision_health.calls}</td></tr></table>
    <p class="sub">Vision scores are uncalibrated model scores, not probabilities. Full mode (YOLO11n + CLIP) and lite mode (cloud vision API) can disagree; neither has been validated on DSU camera footage yet.</p>
    <h3>Operator-labelled accuracy</h3>
    ${Object.keys(a.feedback).length ? `<table><tr><th>Source</th><th>Labelled</th><th>Correct</th><th>False alarm</th><th>Precision</th></tr>${Object.entries(a.feedback).map(([k, v]) =>
      `<tr><td>${esc(k)}</td><td>${v.labelled}</td><td>${v.correct}</td><td>${v.false_positive}</td><td>${v.precision ?? "–"}</td></tr>`).join("")}</table>`
      : `<p class="sub">No labels yet. Operators mark incidents Correct, False alarm or Duplicate; precision per source appears here.</p>`}
    <h3>Security and storage</h3>
    <p class="sub">Operator sign-in: ${a.security.operator_auth ? "required for changes" : "OFF (set OPERATOR_TOKEN)"}. Sensors: ${a.security.sensor_auth ? "token required" : "open"}.
      Uploads up to ${a.security.max_upload_mb} MB; rate limited per device. Records are saved to SQLite and restored on restart (restored: ${a.restored.alerts} alerts, ${a.restored.orders} orders, ${a.restored.reports} reports). Server started ${esc(a.started)}.</p>`;
  $("#about").showModal();
};

// ---------------- QR for the mobile reporter ----------------
function reportUrl() { return `${location.origin}/report`; }
function drawQR(el, size) {
  el.innerHTML = "";
  if (window.QRCode) new QRCode(el, { text: reportUrl(), width: size, height: size, correctLevel: QRCode.CorrectLevel.M });
  else el.innerHTML = `<a class="btnlink" href="/report" target="_blank">Open the reporter</a>`;
}
$("#qrBtn").onclick = () => {
  drawQR($("#qrBig"), 260);
  const local = ["localhost", "127.0.0.1"].includes(location.hostname);
  $("#qrUrl").textContent = reportUrl() + (local ? "  (localhost: phones need the live https link)" : "");
  $("#qrDlg").showModal();
};

// ---------------- guided demo (autopilot) ----------------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitAlert(pred, ms = 9000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    const a = S?.alerts.find((x) => (x.status === "open" || x.status === "assigned") && pred(x));
    if (a) return a;
    await sleep(400);
  }
  return null;
}
const overview = () => map && campusView && map.flyTo({ ...campusView, speed: 0.7 });
const STEPS = [
  ["Verdantix is a live digital twin of DSU Harohalli. Every building is coloured by its live energy and water status.",
    async () => { await post("/api/scenario/clear"); switchTab("alerts"); overview(); }],
  ["Night falls. At 2 AM the campus should be nearly idle, so water flow and power draw are suspicious.",
    () => post("/api/clock", { jump: "night" })],
  ["A pipe bursts near the Boys Hostel. Minimum-night-flow spots the leak; the pressure drop points to the zone.",
    async () => { await post("/api/scenario/leak"); const a = await waitAlert((x) => x.rule === "main-leak"); if (a) { flyTo(a.lng, a.lat); markers[a.id]?.togglePopup(); } }],
  ["One click sends it to Plumbing with a 4-hour deadline. It only closes when CLIP verifies an 'after fix' photo.",
    async () => { const a = await waitAlert((x) => x.rule === "main-leak" && x.status === "open", 2000); if (a) await post(`/api/alerts/${a.id}/order`); switchTab("alerts"); }],
  ["The SOE overhead tank's float switch fails: the pump never stops and clean water spills off the roof.",
    async () => { await post("/api/scenario/tank_overflow"); const a = await waitAlert((x) => x.rule === "tank-overflow"); await selectBuilding("soe-a-block"); $("#tankList").scrollIntoView({ behavior: "smooth" }); if (a) flyTo(a.lng, a.lat); }],
  ["An AC is left on in an empty classroom after hours. The camera sees nobody, so it is flagged with its rupee cost.",
    async () => { await post("/api/scenario/ac_night"); $("#roomGrid").scrollIntoView({ behavior: "smooth" }); }],
  ["Bins: YOLO reads fill level from cameras, the twin forecasts time-to-full and plans an optimised pickup route.",
    async () => { await post("/api/scenario/bin_overflow"); await sleep(1500); const c = $("#showRoute"); if (!c.checked) { c.checked = true; c.onchange({ target: c }); } overview(); }],
  ["The GenAI copilot reads the live twin through tools and ranks what to fix first, citing alert IDs.",
    async () => { switchTab("copilot"); ask("What should facilities fix first today?"); }],
  ["What-if: raise AC by 1 °C, cut after-hours load by half, retrofit LEDs and shift pumping off-peak. Savings, capex and payback over a full year.",
    async () => { switchTab("whatif"); $("#sAc").value = 1; $("#sAh").value = 50; $("#sLed").checked = true; $("#sPump").checked = true; $("#sLeak").checked = true; await runSim(); }],
  ["Students are sensors too. Scan this, snap a photo, and AI checks location, duplicates and the image before it lands on the twin.",
    async () => { $("#qrBtn").onclick(); }],
  ["Detect → Verify → Assign → Prove. Open data, ₹0 running cost, and ready for real ESP32 sensors through the ingest API.",
    async () => { if ($("#qrDlg").open) $("#qrDlg").close(); switchTab("alerts"); overview(); }],
];
let auto = null;
async function runStep(i) {
  if (!auto) return;
  clearTimeout(auto.timer);
  if (i >= STEPS.length) return stopAuto();
  auto.i = i;
  $("#capStep").textContent = `Guided demo, step ${i + 1} of ${STEPS.length}`;
  $("#capText").textContent = STEPS[i][0];
  $("#capNext").textContent = i === STEPS.length - 1 ? "Finish" : "Next";
  try { await STEPS[i][1](); } catch (e) { console.warn(e); }
  if (auto && auto.i === i) auto.timer = setTimeout(() => runStep(i + 1), 11000);
}
function stopAuto() { if (auto) clearTimeout(auto.timer); auto = null; $("#caption").hidden = true; $("#autoBtn").innerHTML = '<svg><use href="#i-play"/></svg><span>Guided demo</span>'; }
$("#autoBtn").onclick = () => {
  if (auto) return stopAuto();
  auto = { i: 0 };
  $("#caption").hidden = false;
  $("#autoBtn").innerHTML = '<svg><use href="#i-stop"/></svg><span>Stop demo</span>';
  runStep(0);
};
$("#capNext").onclick = () => auto && runStep(auto.i + 1);
$("#capStop").onclick = stopAuto;
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && auto) stopAuto(); if (e.key === "ArrowRight" && auto) runStep(auto.i + 1); });

// ---------------- boot ----------------
let booted = false;
async function boot() {
  if (booted) return;
  $("#pdfLink").href = API + "/api/report/weekly.pdf";
  if (!campus) campus = await api("/api/campus");
  const st = await api("/api/state");
  booted = true;
  await initMap();
  const blds = Object.entries(st.buildings);
  $("#bldSelect").innerHTML = blds.map(([id, b]) => `<option value="${id}">${b.name}</option>`).join("");
  $("#walkBld").innerHTML = $("#bldSelect").innerHTML;
  $("#vRoom").innerHTML += Object.keys(st.rooms).map((r) => `<option value="${r}">${r.split("/")[1]}</option>`).join("");
  $("#vBin").innerHTML += Object.keys(st.bins).map((b) => `<option value="${b}">${b}</option>`).join("");
  $("#vCam").innerHTML += campus.features.filter((f) => f.properties.type === "camera")
    .map((f) => `<option value="${f.properties.id}">${esc(f.properties.name)}</option>`).join("");
  $("#vCam").onchange = () => { $("#vLinks").hidden = !!$("#vCam").value; };
  $("#repZone").innerHTML = `<option value="">(click map)</option>` + campus.features.map((f) => `<option value="${f.properties.id}">${f.properties.name}</option>`).join("");
  render(st);
  selectBuilding(blds[0][0]);
  switchTab("alerts");
  runSim();
  loadReports();
  loadLedger();
  loadSensors();
  setInterval(() => { if ($("#tab-building").classList.contains("on")) loadSensors(); }, 10000);
  refreshOperator();
  drawQR($("#qrSmall"), 96);
  setInterval(() => { if ($("#tab-students").classList.contains("on")) loadReports().catch(() => {}); }, 8000);
  connect();
}
let lastMsg = Date.now();
setInterval(() => {   // data freshness: say so when the dashboard stops updating
  const age = Math.round((Date.now() - lastMsg) / 1000);
  const el = $("#fresh");
  if (!el || el.classList.contains("down")) return;
  el.textContent = age < 6 ? "Live" : `Last update ${age} s ago`;
  el.classList.toggle("stale", age >= 15);
}, 1000);
function connect() {
  const base = API ? new URL(API) : location;
  const ws = new WebSocket(`${base.protocol === "https:" ? "wss" : "ws"}://${base.host}/ws`);
  ws.onmessage = (e) => { lastMsg = Date.now(); render(JSON.parse(e.data)); };
  ws.onopen = () => { $("#fresh").className = "fresh"; };
  ws.onclose = () => { $("#fresh").className = "fresh down"; $("#fresh").textContent = "Reconnecting…"; setTimeout(connect, 2000); };
}
// The free cloud backend may be asleep: keep retrying for up to ~3 minutes while it wakes up.
(async function start(tries = 0) {
  try { await boot(); }
  catch (e) {
    if (tries < 36) { toast(`Waking up the live twin… (${e.message})`, 5200); setTimeout(() => start(tries + 1), 5000); }
    else toast(`Startup failed: ${e.message}`, 10000);
  }
})();
