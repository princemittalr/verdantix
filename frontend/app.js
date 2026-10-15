// Verdantix dashboard: live twin over WebSocket, 3D campus map, operations panel.
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const fmt = (n) => (n == null ? "--" : Number(n).toLocaleString("en-IN"));
const COLORS = { ok: "#3fbf7f", warn: "#f0a948", crit: "#ff6b5e" };
let S = null;              // latest twin state
let campus = null;         // campus geojson
let map, bldChart, weekChart;
let selected = null;       // selected building id
let pin = null;            // student report location
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
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

// ---------------- map ----------------
async function initMap() {
  campus = await api("/api/campus");
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
    const xs = [], ys = [];
    campus.features.forEach((f) => (f.geometry.type === "Point" ? [f.geometry.coordinates] : f.geometry.coordinates[0]).forEach(([x, y]) => { xs.push(x); ys.push(y); }));
    map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: { top: 40, bottom: 40, left: 220, right: 20 }, pitch: 50, bearing: -20, duration: 0 });
    map.setZoom(map.getZoom() + 0.5);
    map.addSource("campus", { type: "geojson", data: campus, promoteId: "id" });
    map.addLayer({ id: "sports", type: "fill", source: "campus", filter: ["==", ["get", "type"], "sports"],
                   paint: { "fill-color": "#3fbf7f", "fill-opacity": 0.18, "fill-outline-color": "#3fbf7f" } });
    map.addLayer({ id: "bld", type: "fill-extrusion", source: "campus",
                   filter: ["all", ["==", ["geometry-type"], "Polygon"], ["!=", ["get", "type"], "sports"]],
                   paint: {
                     "fill-extrusion-color": ["case", ["boolean", ["feature-state", "sel"], false], "#e6fff4",
                       ["match", ["coalesce", ["feature-state", "status"], "ok"], "crit", COLORS.crit, "warn", COLORS.warn, COLORS.ok]],
                     "fill-extrusion-height": ["*", ["max", ["get", "floors"], 1], 3.6],
                     "fill-extrusion-opacity": 0.82,
                   } });
    map.addLayer({ id: "bld-label", type: "symbol", source: "campus", filter: ["==", ["geometry-type"], "Polygon"],
                   layout: { "text-field": ["get", "name"], "text-size": 12, "text-font": ["Noto Sans Regular"], "symbol-placement": "point" },
                   paint: { "text-color": "#fff", "text-halo-color": "#000", "text-halo-width": 1.2 } });
    map.addSource("pts", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: "pts", type: "circle", source: "pts",
                   paint: { "circle-radius": ["match", ["get", "type"], "bin", 7, 5],
                            "circle-color": ["match", ["get", "type"], "bin", ["get", "color"], "camera", "#9fd3ff", "water_tank", "#5fa8ff", "water_meter", "#5fa8ff", "#ccc"],
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
    if (S) render(S);
  });
}

function updateMap(st) {
  if (!map || !map.getSource("pts")) return;
  for (const [id, b] of Object.entries(st.buildings)) map.setFeatureState({ source: "campus", id }, { status: b.status, sel: id === selected });
  const pts = campus.features.filter((f) => f.geometry.type === "Point").map((f) => {
    const p = { ...f.properties };
    const bin = st.bins[p.id];
    if (bin) { p.color = COLORS[bin.status]; p.info = `Fill ${bin.fill}% · full in ${bin.eta_full_h} h · ${bin.source}`; }
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
      .setPopup(new maplibregl.Popup({ offset: 12 }).setHTML(`<b>${a.id} · ${a.title}</b><br>${a.detail}<br>₹${fmt(a.inr_per_day)}/day`))
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
  renderBoard(st);
  renderAlerts(st);
  renderBuilding(st);
  updateMap(st);
}

function renderBoard(st) {
  const rows = Object.entries(st.buildings).sort((a, b) => b[1].score - a[1].score);
  $("#board").innerHTML = `<h4>Green score leaderboard</h4>` + rows.map(([id, b], i) =>
    `<div class="r"><span>${i + 1}</span><span>${b.name}</span><span>${b.score}</span><span class="grade ${b.grade}">${b.grade}</span></div>`).join("");
}

function alertCard(a) {
  const order = a.work_order ? S.orders.find((o) => o.id === a.work_order) : null;
  const faded = a.status === "resolved" || a.status === "self-cleared";
  return `<div class="card ${faded ? "ok faded" : a.severity}">
    <div class="t">${a.id} · ${a.title}</div>
    <div class="d">${a.detail}</div>
    <div class="m"><span class="pill">${a.kind}</span><span class="pill src-${a.source}">${a.source}</span>
      <span>₹${fmt(a.inr_per_day)}/day</span><span>${a.co2_kg_per_day} kg CO₂/day</span><span>${a.status}${order ? " · " + order.id : ""}</span></div>
    ${a.evidence?.image ? `<img src="${a.evidence.image}" alt="" style="max-width:100%;border-radius:6px">` : ""}
    ${a.status === "open" ? `<div class="act"><button data-assign="${a.id}">Assign to ${({ water: "Plumbing", energy: "Electrical", waste: "Housekeeping" })[a.kind] || "Facilities"}</button>
      <button class="ghost" data-dismiss="${a.id}">Dismiss</button></div>` : ""}
  </div>`;
}

function renderAlerts(st) {
  const open = st.alerts.filter((a) => a.status === "open" || a.status === "assigned");
  const done = st.alerts.filter((a) => a.status === "resolved").slice(0, 5);
  $("#alertList").innerHTML = open.length || done.length ? [...open, ...done].map(alertCard).join("") : `<div class="sub">No open issues. The twin is watching every meter, camera and report.</div>`;
  $("#orderList").innerHTML = st.orders.length ? st.orders.map((o) => `<div class="card ${o.status === "done" ? "ok" : o.status === "overdue" ? "crit" : "warn"}">
      <div class="t">${o.id} · ${o.team}</div><div class="d">${o.title}</div>
      <div class="m"><span>${o.status}</span><span>due ${o.deadline.replace("T", " ")}</span></div>
      ${o.status !== "done" ? `<div class="act"><label class="btnlink" style="cursor:pointer">Upload proof photo<input type="file" accept="image/*" data-proof="${o.id}" hidden></label></div>` : `<div class="m">Verified by CLIP · issue score ${o.proof?.issue_score}</div>`}
    </div>`).join("") : `<div class="sub">No work orders yet. Assign an alert to create one.</div>`;
}

async function selectBuilding(id) {
  selected = id;
  $("#bldSelect").value = id;
  switchTab("building");
  const s = await api(`/api/building/${id}/series`);
  const data = { labels: s.t, datasets: [
    { label: "Actual kWh", data: s.kwh, borderColor: "#4cc79e", backgroundColor: "rgba(76,199,158,.15)", fill: true, tension: 0.3, pointRadius: 0 },
    { label: "Expected", data: s.baseline, borderColor: "#8fa39b", borderDash: [4, 4], pointRadius: 0, tension: 0.3 },
  ] };
  if (bldChart) { bldChart.data = data; bldChart.update(); }
  else bldChart = new Chart($("#bldChart"), { type: "line", data, options: chartOpts("kWh per hour") });
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
  ].map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("");
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
    const r = await post("/api/copilot", { question: q, history: chatHistory });
    wait.remove();
    addMsg(r.answer, false, `${r.provider}${r.tools.length ? " · tools: " + r.tools.map((t) => t.tool).join(", ") : ""}`);
    chatHistory.push({ role: "user", content: q }, { role: "assistant", content: r.answer });
  } catch (e) { wait.innerHTML = `Couldn't reach the copilot: ${e.message}`; }
}
$("#askForm").onsubmit = (e) => { e.preventDefault(); const q = $("#askInput").value.trim(); if (q) { $("#askInput").value = ""; ask(q); } };
$$(".chips button").forEach((b) => (b.onclick = () => ask(b.textContent)));

// ---------------- what-if ----------------
async function runSim() {
  const p = { ac_setpoint_delta: +$("#sAc").value, after_hours_cut: $("#sAh").value / 100, solar_kwp: +$("#sSol").value,
              pickups_per_day: +$("#sPk").value, led_retrofit: $("#sLed").checked, fix_leaks: $("#sLeak").checked };
  $("#vAc").textContent = `${p.ac_setpoint_delta} °C`; $("#vAh").textContent = `${Math.round(p.after_hours_cut * 100)}%`;
  $("#vSol").textContent = `${p.solar_kwp} kWp`; $("#vPk").textContent = p.pickups_per_day;
  const r = await post("/api/simulate", p);
  $("#simOut").innerHTML = `<div class="big">
      <div class="stat"><span>Saved per year</span><b>₹${fmt(r.inr_saved_year)}</b></div>
      <div class="stat"><span>CO₂ avoided</span><b>${r.co2_tonnes_saved_year} t</b></div>
      <div class="stat"><span>Energy saved</span><b>${fmt(r.kwh_saved_year)} kWh</b></div>
      <div class="stat"><span>Of campus energy</span><b>${r.pct_energy_saved}%</b></div></div>
    <div class="list">${Object.entries(r.breakdown_kwh).map(([k, v]) => `<div class="m"><span style="width:110px">${k.replace("_", " ")}</span>
      <div class="bar" style="flex:1"><i style="width:${r.kwh_saved_year ? (100 * v) / r.kwh_saved_year : 0}%"></i></div><span>${fmt(v)} kWh</span></div>`).join("")}</div>
    <div class="sub">Water saved: ${fmt(r.water_kl_saved_year)} kL/yr · bin overflow-hours/yr: ${fmt(r.bin_overflow_hours_year)} ·
      assumes AC share ${Math.round(r.assumptions.ac_share * 100)}% of load, ₹${r.assumptions.tariff}/kWh, ${r.assumptions.grid_factor} kg CO₂/kWh</div>`;
}
["#sAc", "#sAh", "#sSol", "#sPk", "#sLed", "#sLeak"].forEach((s) => $(s).addEventListener("input", runSim));

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
  $("#repResult").innerHTML = "<div class='sub'>Checking location, duplicates and photo…</div>";
  const r = await api("/api/reports", { method: "POST", body: fd });
  $("#repResult").innerHTML = r.ok
    ? `<div class="card ok"><div class="t">Accepted as ${r.report.id} → alert ${r.alert}</div><div class="d">CLIP: ${r.report.clip.issue} (${Math.round(r.report.clip.issue_score * 100)}%)</div></div>`
    : `<div class="card warn"><div class="t">Not accepted (${r.check} check)</div><div class="d">${r.message}</div></div>`;
  loadReports();
};
async function loadReports() {
  const rs = await api("/api/reports");
  $("#repList").innerHTML = rs.length ? rs.map((r) => `<div class="card ${r.kind === "water" ? "crit" : "warn"}"><div class="t">${r.id} · ${r.kind} · ${r.zone}</div>
    <img src="${r.image}" alt="" style="max-width:100%;border-radius:6px"><div class="m"><span>${r.created.replace("T", " ")}</span><span>+${r.confirmations} confirmations</span><span>${r.alert}</span></div></div>`).join("")
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

// ---------------- boot ----------------
async function boot() {
  await initMap();
  const st = await api("/api/state");
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
  connect();
}
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (e) => render(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 2000);
}
boot().catch((e) => toast(`Startup failed: ${e.message}`, 8000));
