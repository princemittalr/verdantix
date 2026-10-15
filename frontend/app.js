// Verdantix dashboard: live twin over WebSocket, 3D campus map, operations panel.
// Backend base URL: "" when this page is served by the FastAPI server, else the tunnel URL from config.js.
const API = String(window.VX_API || "").replace(/\/$/, "");
const media = (p) => (p && p.startsWith("/") ? API + p : p);
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const fmt = (n) => (n == null ? "--" : Number(n).toLocaleString("en-IN"));
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const COLORS = { ok: "#3fbf7f", warn: "#f0a948", crit: "#ff6b5e" };
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

async function api(path, opts = {}) {
  let r;
  try { r = await fetch(API + path, opts); }
  catch { throw new Error(API ? "the live twin backend isn't reachable yet" : "server not reachable"); }
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}
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
    style: {
      version: 8,
      glyphs: "https://protomaps.github.io/basemaps-assets/fonts/{fontstack}/{range}.pbf",
      sources: {
        sat: { type: "raster", tileSize: 256, attribution: "Imagery © Esri",
               tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"] },
      },
      layers: [{ id: "sat", type: "raster", source: "sat", paint: { "raster-saturation": -0.35, "raster-brightness-max": 0.75 } }],
    },
    center, zoom: 16.4, pitch: 55, bearing: -20, maxZoom: 19.5,
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
  map.on("load", () => {
    // Campus-only view: everything outside the DSU boundary is dimmed and the camera stays on campus.
    const { ring, center: cc } = campusBoundary(70);
    const xs = ring.map((c) => c[0]), ys = ring.map((c) => c[1]);
    const bounds = [[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]];
    const padLng = (bounds[1][0] - bounds[0][0]) * 0.9, padLat = (bounds[1][1] - bounds[0][1]) * 0.9;
    map.setMaxBounds([[bounds[0][0] - padLng, bounds[0][1] - padLat], [bounds[1][0] + padLng, bounds[1][1] + padLat]]);
    map.fitBounds(bounds, { padding: { top: 30, bottom: 60, left: 230, right: 20 }, pitch: 0, bearing: -20, duration: 0 });
    map.setMinZoom(Math.max(map.getZoom() - 0.8, 14));
    map.jumpTo({ zoom: map.getZoom() + 0.4, pitch: 58 });
    campusView = { center: map.getCenter(), zoom: map.getZoom(), pitch: 58, bearing: -20 };
    map.setLight({ anchor: "map", position: [1.4, 215, 35], intensity: 0.45, color: "#ffffff" });

    map.addSource("mask", { type: "geojson", data: { type: "Feature", properties: {},
      geometry: { type: "Polygon", coordinates: [[[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]], ring.slice().reverse()] } } });
    map.addLayer({ id: "mask", type: "fill", source: "mask", paint: { "fill-color": "#050a08", "fill-opacity": 0.78 } });
    map.addSource("boundary", { type: "geojson", data: { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: ring } } });
    map.addLayer({ id: "boundary-glow", type: "line", source: "boundary", paint: { "line-color": "#4cc79e", "line-width": 10, "line-opacity": 0.18, "line-blur": 6 } });
    map.addLayer({ id: "boundary", type: "line", source: "boundary", paint: { "line-color": "#4cc79e", "line-width": 1.6, "line-dasharray": [3, 2] } });

    map.addSource("campus", { type: "geojson", data: campus, promoteId: "id" });
    map.addLayer({ id: "sports", type: "fill", source: "campus", filter: ["==", ["get", "type"], "sports"],
                   paint: { "fill-color": "#3fbf7f", "fill-opacity": 0.22, "fill-outline-color": "#7ee2b4" } });
    map.addLayer({ id: "footprint", type: "line", source: "campus",
                   filter: ["all", ["==", ["geometry-type"], "Polygon"], ["!=", ["get", "type"], "sports"]],
                   paint: { "line-color": "#e3ece8", "line-width": 1, "line-opacity": 0.5 } });
    map.addSource("floors", { type: "geojson", data: floorSlabs(), promoteId: "fid" });
    const statusColor = ["match", ["coalesce", ["feature-state", "status"], "ok"], "crit", COLORS.crit, "warn", COLORS.warn, COLORS.ok];
    map.addLayer({ id: "bld", type: "fill-extrusion", source: "floors",
                   paint: {
                     "fill-extrusion-color": ["case", ["boolean", ["feature-state", "sel"], false],
                       ["interpolate", ["linear"], 0.5, 0, statusColor, 1, "#ffffff"],
                       ["boolean", ["get", "roof"], false], ["interpolate", ["linear"], 0.35, 0, statusColor, 1, "#ffffff"], statusColor],
                     "fill-extrusion-base": ["get", "base"],
                     "fill-extrusion-height": ["get", "top"],
                     "fill-extrusion-opacity": 0.88,
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
    map.addLayer({ id: "bld-label", type: "symbol", source: "campus", filter: ["==", ["geometry-type"], "Polygon"],
                   layout: { "text-field": ["get", "name"], "text-size": 12, "text-font": ["Noto Sans Regular"], "symbol-placement": "point" },
                   paint: { "text-color": "#fff", "text-halo-color": "#000", "text-halo-width": 1.2 } });
    map.addSource("pts", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: "pts", type: "circle", source: "pts",
                   paint: { "circle-radius": ["match", ["get", "type"], "bin", 7, 5],
                            "circle-color": ["match", ["get", "type"], "bin", ["get", "color"], "water_tank", ["coalesce", ["get", "color"], "#5fa8ff"], "camera", "#9fd3ff", "water_meter", "#5fa8ff", "#ccc"],
                            "circle-stroke-color": "#000", "circle-stroke-width": 1.5 } });
    map.addSource("route", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: "route", type: "line", source: "route", layout: { visibility: "none" },
                   paint: { "line-color": "#ffd166", "line-width": 4, "line-dasharray": [2, 1] } });
    map.addSource("hot", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: "hot", type: "circle", source: "hot", layout: { visibility: "none" },
                   paint: { "circle-radius": ["*", ["get", "weight"], 6], "circle-color": "#ff6b5e", "circle-opacity": 0.25,
                            "circle-stroke-color": "#ff6b5e", "circle-stroke-width": 1 } });

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
  $("#viewBtn").textContent = on ? "2D" : "3D";
  if (!on && orbiting) toggleOrbit(false);
  map.easeTo({ pitch: on ? 58 : 0, bearing: on ? -20 : 0, duration: 900 });
}
function resetView() {
  if (orbiting) toggleOrbit(false);
  if (campusView) map.flyTo({ ...campusView, speed: 0.8 });
}

function updateMap(st) {
  if (!map || !map.getSource("pts")) return;
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
    if (tank) { p.color = tank.status === "ok" ? "#5fa8ff" : COLORS[tank.status]; p.info = `Level ${tank.level_pct}% · pump ${tank.pump_on ? "ON" : "off"}${tank.overflow_lph ? ` · overflowing ${fmt(tank.overflow_lph)} L/h` : ""}`; }
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
    markers[a.id] = new maplibregl.Marker({ element: el }).setLngLat([a.lng, a.lat])
      .setPopup(new maplibregl.Popup({ offset: 12 }).setHTML(`<b>${a.id} · ${esc(a.title)}</b><br>${esc(a.detail)}<br>₹${fmt(a.inr_per_day)}/day`))
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
  $("#kGrade").textContent = "/100";
  $("#kPower").textContent = fmt(k.power_kw);
  $("#kEnergy").textContent = fmt(k.energy_today_kwh);
  $("#kCost").textContent = fmt(k.cost_today_inr);
  $("#kCo2").textContent = fmt(k.co2_today_kg);
  $("#kWater").textContent = fmt(k.water_today_kl);
  $("#kAlerts").textContent = k.open_alerts;
  $("#kWaste").textContent = k.waste_inr_per_day ? `₹${fmt(k.waste_inr_per_day)}/day at risk` : "";
  const tb = st.tariff || {};
  $("#tariff").textContent = tb.band ? `${tb.band} tariff · ₹${tb.inr_per_kwh}/kWh` : "--";
  $("#tariff").className = `badge ${tb.band || ""}`;
  $("#iot").hidden = !st.live_sensors;
  $("#iot").textContent = `IoT live · ${st.live_sensors}`;
  announce(st);
  renderBoard(st);
  renderAlerts(st);
  renderBuilding(st);
  updateMap(st);
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
  $("#board").innerHTML = `<h4>Green score leaderboard</h4>` + rows.map(([id, b], i) =>
    `<div class="r"><span>${i + 1}</span><span>${b.name}</span><span>${b.score}</span><span class="grade ${b.grade}">${b.grade}</span></div>`).join("");
}

function alertCard(a) {
  const order = a.work_order ? S.orders.find((o) => o.id === a.work_order) : null;
  const faded = a.status === "resolved" || a.status === "self-cleared";
  return `<div class="card clickable ${faded ? "ok faded" : a.severity}" data-fly="${a.lng},${a.lat}" title="Click to locate on the map">
    <div class="t">${a.id} · ${esc(a.title)}</div>
    <div class="d">${esc(a.detail)}</div>
    <div class="m"><span class="pill">${a.kind}</span><span class="pill src-${a.source}">${a.source}</span>
      <span>₹${fmt(a.inr_per_day)}/day</span><span>${a.co2_kg_per_day} kg CO₂/day</span><span>${a.status}${order ? " · " + order.id : ""}</span></div>
    ${a.evidence?.image ? `<img src="${media(a.evidence.image)}" alt="" style="max-width:100%;border-radius:6px">` : ""}
    ${a.status === "open" ? `<div class="act"><button data-assign="${a.id}">Assign to ${({ water: "Plumbing", energy: "Electrical", waste: "Housekeeping" })[a.kind] || "Facilities"}</button>
      <button class="ghost" data-dismiss="${a.id}">Dismiss</button></div>` : ""}
  </div>`;
}

function renderAlerts(st) {
  const open = st.alerts.filter((a) => a.status === "open" || a.status === "assigned");
  const done = st.alerts.filter((a) => a.status === "resolved").slice(0, 5);
  $("#alertList").innerHTML = open.length || done.length ? [...open, ...done].map(alertCard).join("") : `<div class="sub">No open issues. The twin is watching every meter, camera and report.</div>`;
  $("#orderList").innerHTML = st.orders.length ? st.orders.map((o) => `<div class="card ${o.status === "done" ? "ok" : o.status === "cancelled" ? "faded" : o.status === "overdue" ? "crit" : "warn"}">
      <div class="t">${o.id} · ${o.team}</div><div class="d">${esc(o.title)}</div>
      <div class="m"><span>${o.status}</span><span>due ${o.deadline.replace("T", " ")}</span></div>
      ${o.status === "cancelled" ? `<div class="m">Demo scenario cleared</div>` : o.status !== "done" ? `<div class="act"><label class="btnlink" style="cursor:pointer">Upload proof photo<input type="file" accept="image/*" data-proof="${o.id}" hidden></label></div>` : `<div class="m">Verified by CLIP · issue score ${o.proof?.issue_score}</div>`}
    </div>`).join("") : `<div class="sub">No work orders yet. Assign an alert to create one.</div>`;
}

async function selectBuilding(id) {
  selected = id;
  $("#bldSelect").value = id;
  switchTab("building");
  const s = await api(`/api/building/${id}/series`);
  const pad = (n) => Array(n).fill(null);
  const fc = s.forecast || [];
  const data = { labels: [...s.t, ...(s.t_fc || [])], datasets: [
    { label: "Actual kWh", data: [...s.kwh, ...pad(fc.length)], borderColor: "#4cc79e", backgroundColor: "rgba(76,199,158,.15)", fill: true, tension: 0.3, pointRadius: 0 },
    { label: "Expected", data: [...s.baseline, ...pad(fc.length)], borderColor: "#8fa39b", borderDash: [4, 4], pointRadius: 0, tension: 0.3 },
    { label: "AI forecast (next 24 h)", data: [...pad(s.kwh.length - 1), s.kwh[s.kwh.length - 1], ...fc], borderColor: "#c99bff", borderDash: [6, 3], pointRadius: 0, tension: 0.3 },
  ] };
  if (bldChart) { bldChart.data = data; bldChart.update(); }
  else bldChart = new Chart($("#bldChart"), { type: "line", data, options: chartOpts("kWh per hour · last 24 h and forecast") });
  forecastInfo = s.forecast ? { id, kwh: s.forecast_kwh_24h, peak: s.forecast_peak_kw, at: s.forecast_peak_at, mape: s.forecast_mape_pct } : null;
  if (S) renderBuilding(S);
}

function chartOpts(title) {
  return { responsive: true, animation: false, plugins: { legend: { labels: { color: "#8fa39b", boxWidth: 12 } }, title: { display: !!title, text: title, color: "#8fa39b" } },
           scales: { x: { ticks: { color: "#8fa39b", maxTicksLimit: 8 }, grid: { color: "#1c2a25" } }, y: { ticks: { color: "#8fa39b" }, grid: { color: "#1c2a25" } } } };
}

function renderBuilding(st) {
  const id = selected || Object.keys(st.buildings)[0];
  const b = st.buildings[id];
  if (!b) return;
  $("#bldInfo").innerHTML = [
    ["Power", `${fmt(b.power_kw)} kW`], ["Expected", `${fmt(b.baseline_kw)} kW`], ["Score", `${b.score} (${b.grade})`],
    ["Water", `${fmt(b.water_lph)} L/h`], ["Water normal", `${fmt(b.water_baseline_lph)} L/h`], ["Alerts", b.alerts.length],
    ...(forecastInfo && forecastInfo.id === id ? [["Next 24 h (AI)", `${fmt(forecastInfo.kwh)} kWh`], ["Forecast peak", `${fmt(forecastInfo.peak)} kW`], ["Peak at", forecastInfo.at]] : []),
  ].map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("") + (b.live ? `<div class="sub"><span class="live-tag">LIVE IoT</span> readings from a real sensor override the replayed meter.</div>` : "");
  $("#tankList").innerHTML = Object.entries(st.tanks || {}).map(([tid, t]) => `<div class="card clickable ${t.status}" data-fly="${t.lng},${t.lat}">
      <div class="tank"><div class="tankviz ${t.status}"><i style="height:${t.level_pct}%"></i></div>
      <div><div class="t">${t.name}${t.live ? '<span class="live-tag">LIVE</span>' : ""}</div>
      <div class="m"><span>${t.level_pct}% · ${t.level_kl}/${t.capacity_kl} kL</span><span>pump ${t.pump_on ? "ON" : "off"}</span>
      <span>use ${fmt(t.demand_lph)} L/h</span>${t.overflow_lph ? `<span style="color:var(--crit)">spilling ${fmt(t.overflow_lph)} L/h</span>` : ""}</div>
      <div class="m"><span>peak-hour pumping today: ${t.peak_pump_min_today} min</span></div></div></div></div>`).join("") || `<div class="sub">No tanks mapped.</div>`;
  const waste = new Set(st.alerts.filter((a) => a.rule === "empty-room" && a.status !== "resolved" && a.status !== "self-cleared").map((a) => a.asset));
  $("#roomGrid").innerHTML = Object.entries(st.rooms).map(([r, v]) => `<div class="room ${waste.has(r) ? "waste" : ""}">
      <b>${r.split("/")[1]}</b>AC ${v.ac_kw} kW<br>Light ${v.light_kw} kW<br>${v.people != null ? "📷 " + v.people + " people" : "occ " + Math.round(v.occ_prob * 100) + "%"}<br>${v.temp_c}°C</div>`).join("");
  $("#binList").innerHTML = Object.entries(st.bins).sort((a, b) => b[1].fill - a[1].fill).map(([id, v]) => `<div class="card ${v.status}">
      <div class="t">${id} <small class="d">${v.source}</small></div>
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
  await post("/api/clock", { speed: Number(e.target.dataset.speed) });
});
$("#jumpNight").onclick = async () => { await post("/api/clock", { jump: "night" }); toast("Jumped to 2 AM: watch night-time water and after-hours power"); };
$("#jumpMorning").onclick = async () => { await post("/api/clock", { jump: "morning" }); toast("Jumped to 9 AM"); };
$$(".demo button").forEach((b) => (b.onclick = async () => {
  const r = await post(`/api/scenario/${b.dataset.scn}`);
  toast(b.dataset.scn === "clear" ? "Demo scenarios cleared" : `Injected ${r.kind} at ${r.target}`);
}));
$("#bldSelect").onchange = (e) => selectBuilding(e.target.value);

document.addEventListener("click", async (e) => {
  const assign = e.target.dataset.assign, dismiss = e.target.dataset.dismiss;
  const card = e.target.closest?.("[data-fly]");
  if (card && !assign && !dismiss && e.target.tagName !== "INPUT" && e.target.tagName !== "LABEL") {
    const [lng, lat] = card.dataset.fly.split(",").map(Number);
    flyTo(lng, lat);
  }
  if (assign) { const o = await post(`/api/alerts/${assign}/order`); toast(`${o.id} assigned to ${o.team}, due ${o.deadline.replace("T", " ")}`); }
  if (dismiss) { await post(`/api/alerts/${dismiss}/dismiss`); }
});
document.addEventListener("change", async (e) => {
  const oid = e.target.dataset.proof;
  if (!oid || !e.target.files[0]) return;
  const fd = new FormData();
  fd.append("file", e.target.files[0]);
  toast("Verifying proof photo with CLIP…");
  const r = await api(`/api/orders/${oid}/proof`, { method: "POST", body: fd });
  toast(r.message, 5000);
});
$("#viewBtn").onclick = () => toggle3D();
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
  fd.append("room", $("#vRoom").value);
  fd.append("bin", $("#vBin").value);
  const r = await api("/api/vision/analyze", { method: "POST", body: fd });
  drawBoxes(r, f.w, f.h);
  const c = r.clip;
  $("#vOut").innerHTML = `<div><b>${r.people}</b> people · ${r.bin ? `bin: <b>${r.bin.label}</b> (${Math.round(r.bin.conf * 100)}%)` : "no bin model hit"} · ${r.ms} ms</div>
    <div>CLIP issue: <b>${c.issue}</b> ${Math.round(c.issue_score * 100)}% ${c.issue_score >= 0.5 ? "⚠️" : ""}</div>
    <div>${c.top.map((t) => `<span class="tag">${t.label} ${Math.round(t.p * 100)}%</span>`).join("")}</div>
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
function addMsg(text, me, meta) {
  const d = document.createElement("div");
  d.className = `msg ${me ? "me" : ""}`;
  d.innerHTML = me ? text.replace(/</g, "&lt;") : md(text) + (meta ? `<small>${meta}</small>` : "");
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
  $("#simOut").innerHTML = `<div class="big">
      <div class="stat"><span>Saved per year</span><b>₹${fmt(r.inr_saved_year)}</b></div>
      <div class="stat"><span>CO₂ avoided</span><b>${r.co2_tonnes_saved_year} t</b></div>
      <div class="stat"><span>Energy saved</span><b>${fmt(r.kwh_saved_year)} kWh</b></div>
      <div class="stat"><span>Of campus energy</span><b>${r.pct_energy_saved}%</b></div></div>
    ${paybackTable(r)}
    <div class="list">${Object.entries(r.breakdown_kwh).map(([k, v]) => `<div class="m"><span style="width:110px">${k.replace("_", " ")}</span>
      <div class="bar" style="flex:1"><i style="width:${r.kwh_saved_year ? (100 * v) / r.kwh_saved_year : 0}%"></i></div><span>${fmt(v)} kWh</span></div>`).join("")}</div>
    <div class="sub">Water saved: ${fmt(r.water_kl_saved_year)} kL/yr · bin overflow-hours/yr: ${fmt(r.bin_overflow_hours_year)}${r.bin_opex_inr_year > 0 ? ` (extra pickups cost ₹${fmt(r.bin_opex_inr_year)}/yr)` : ""} ·
      assumes AC share ${Math.round(r.assumptions.ac_share * 100)}% of load, ₹${r.assumptions.tariff}/kWh, ${r.assumptions.grid_factor} kg CO₂/kWh</div>`;
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
$("#genReport").onclick = async () => {
  $("#reportOut").innerHTML = "<div class='sub'>Pulling a week of twin data and writing the report…</div>";
  const r = await post("/api/report/weekly");
  $("#pdfLink").hidden = false;
  $("#reportOut").innerHTML = md(r.narrative) + `<div class="sub">Written by ${r.provider} from twin data.</div>`;
  const h = await api("/api/history?days=7");
  const data = { labels: h.daily_kwh.map((_, i) => `Day ${i + 1}`), datasets: [{ label: "Campus kWh per day", data: h.daily_kwh, backgroundColor: "#4cc79e" }] };
  if (weekChart) { weekChart.data = data; weekChart.update(); } else weekChart = new Chart($("#weekChart"), { type: "bar", data, options: chartOpts() });
};

// ---------------- students ----------------
$("#repForm").onsubmit = async (e) => {
  e.preventDefault();
  const f = $("#repFile").files[0];
  let loc = pin;
  if ($("#repZone").value) {
    const feat = campus.features.find((x) => x.properties.id === $("#repZone").value);
    const c = feat.geometry.type === "Point" ? feat.geometry.coordinates : feat.properties.center;
    loc = { lng: c[0], lat: c[1] };
  }
  if (!loc) return toast("Pick a place from the list or click the map.");
  const fd = new FormData();
  fd.append("file", f); fd.append("lat", loc.lat); fd.append("lng", loc.lng); fd.append("note", $("#repNote").value);
  fd.append("reporter", $("#repName").value.trim());
  $("#repResult").innerHTML = "<div class='sub'>Checking location, duplicates and photo…</div>";
  const r = await api("/api/reports", { method: "POST", body: fd });
  $("#repResult").innerHTML = r.ok
    ? `<div class="card ok"><div class="t">Accepted as ${r.report.id} → alert ${r.alert}${r.credits != null ? ` · ${r.credits} green credits` : ""}</div><div class="d">CLIP: ${r.report.clip.issue} (${Math.round(r.report.clip.issue_score * 100)}%)</div></div>`
    : `<div class="card warn"><div class="t">Not accepted (${r.check} check)</div><div class="d">${r.message}</div></div>`;
  loadReports();
};
async function loadReports() {
  const board = await api("/api/reports/leaderboard");
  $("#repBoard").innerHTML = board.length ? board.map((b, i) => `<div class="m"><span>${i + 1}.</span><span style="flex:1">${esc(b.name)}</span><b>${b.credits}</b></div>`).join("")
    : `<div class="sub">Add your name to a report to earn credits (10 per accepted report, 2 per confirmation).</div>`;
  const rs = await api("/api/reports");
  $("#repList").innerHTML = rs.length ? rs.map((r) => `<div class="card clickable ${r.kind === "water" ? "crit" : "warn"}" data-fly="${r.lng},${r.lat}"><div class="t">${r.id} · ${r.kind} · ${esc(r.zone)} <small class="d">by ${esc(r.reporter || "anonymous")}</small></div>
    <img src="${media(r.image)}" alt="" style="max-width:100%;border-radius:6px"><div class="m"><span>${r.created.replace("T", " ")}</span><span>+${r.confirmations} confirmations</span><span>${r.alert}</span></div></div>`).join("")
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
    <p class="sub">GenAI: ${Object.entries(a.llm).map(([k, v]) => `${k} ${v ? "✓" : "✗"}`).join(" · ")} · offline rule-based fallback always on.</p>`;
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
  $("#capStep").textContent = `Guided demo · ${i + 1} / ${STEPS.length}`;
  $("#capText").textContent = STEPS[i][0];
  $("#capNext").textContent = i === STEPS.length - 1 ? "Finish" : "Next ▸";
  try { await STEPS[i][1](); } catch (e) { console.warn(e); }
  if (auto && auto.i === i) auto.timer = setTimeout(() => runStep(i + 1), 11000);
}
function stopAuto() { if (auto) clearTimeout(auto.timer); auto = null; $("#caption").hidden = true; $("#autoBtn").textContent = "▶ Guided demo"; }
$("#autoBtn").onclick = () => {
  if (auto) return stopAuto();
  auto = { i: 0 };
  $("#caption").hidden = false;
  $("#autoBtn").textContent = "■ Stop demo";
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
  $("#vRoom").innerHTML += Object.keys(st.rooms).map((r) => `<option value="${r}">${r.split("/")[1]}</option>`).join("");
  $("#vBin").innerHTML += Object.keys(st.bins).map((b) => `<option value="${b}">${b}</option>`).join("");
  $("#repZone").innerHTML = `<option value="">(click map)</option>` + campus.features.map((f) => `<option value="${f.properties.id}">${f.properties.name}</option>`).join("");
  render(st);
  selectBuilding(blds[0][0]);
  switchTab("alerts");
  runSim();
  loadReports();
  drawQR($("#qrSmall"), 96);
  setInterval(() => { if ($("#tab-students").classList.contains("on")) loadReports().catch(() => {}); }, 8000);
  connect();
}
function connect() {
  const base = API ? new URL(API) : location;
  const ws = new WebSocket(`${base.protocol === "https:" ? "wss" : "ws"}://${base.host}/ws`);
  ws.onmessage = (e) => render(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 2000);
}
// The free cloud backend may be asleep: keep retrying for up to ~3 minutes while it wakes up.
(async function start(tries = 0) {
  try { await boot(); }
  catch (e) {
    if (tries < 36) { toast(`Waking up the live twin… (${e.message})`, 5200); setTimeout(() => start(tries + 1), 5000); }
    else toast(`Startup failed: ${e.message}`, 10000);
  }
})();
