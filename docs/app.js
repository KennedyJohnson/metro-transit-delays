/* global featureRow, TreeModel */
const $ = (id) => document.getElementById(id);
const getJSON = (u) => fetch(u, { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
const WINDOW_BEFORE = 15, WINDOW_COUNT = 8;
const WEATHER_VARS = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"];

let meta, routes, cal, models = null, weather = null, showAll = false;
const routeById = {}, bodies = {};
const sel = { r: null, d: null, s: null };

// ---------- formatting ----------
function clock(min) {
  const m = ((min % 1440) + 1440) % 1440, h = Math.floor(m / 60), mm = String(m % 60).padStart(2, "0");
  return `${h % 12 || 12}:${mm} ${h < 12 ? "AM" : "PM"}`;
}
const pct = (p) => `${Math.round(p * 100)}%`;
function delayText(m) {
  if (m <= -0.5) return `${Math.round(-m)} min early`;
  if (m < 1) return "on time";
  return `${Math.round(m)} min late`;
}
function localISO(d) { return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; }
function routeLabel(r) { return r.short && r.long ? `${r.short} – ${r.long}` : r.short || r.long || r.id; }
function el(tag, props = {}, ...kids) {
  const n = Object.assign(document.createElement(tag), props);
  n.append(...kids.filter((k) => k != null));
  return n;
}

// ---------- data ----------
function loadModels() {
  models ??= Promise.all(["median", "late", "early"].map((k) => getJSON(`data/model_${k}.json`)))
    .then(([m, l, e]) => (m && l && e ? { median: new TreeModel(m), late: new TreeModel(l), early: new TreeModel(e) } : null));
  return models;
}
function loadWeather() {
  const q = "latitude=44.98&longitude=-93.27&timezone=America%2FChicago&forecast_days=8&hourly=" + WEATHER_VARS.join(",");
  weather ??= getJSON(`https://api.open-meteo.com/v1/forecast?${q}`).then((j) => {
    if (!j?.hourly) return null;
    const out = {};
    j.hourly.time.forEach((t, i) => { out[t] = Object.fromEntries(WEATHER_VARS.map((k) => [k, j.hourly[k][i]])); });
    return out;
  });
  return weather;
}
function routeBody(id) {
  bodies[id] ??= getJSON(`data/routes/${routeById[id].file}.json`);
  return bodies[id];
}

// ---------- picker ----------
function fillRoutes() {
  const list = [...routes].sort((a, b) => (a.short || a.long).localeCompare(b.short || b.long, undefined, { numeric: true }));
  $("route").replaceChildren(el("option", { value: "", textContent: "Choose a route" }),
    ...list.map((r) => el("option", { value: r.id, textContent: routeLabel(r) })));
}

function fillDays() {
  const now = new Date();
  const opts = [];
  for (let i = 0; i < 7; i++) {
    const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() + i);
    const iso = localISO(d);
    if (!cal.services[iso]) continue;
    const name = i === 0 ? "Today" : i === 1 ? "Tomorrow" : d.toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" });
    opts.push(el("option", { value: iso, textContent: name }));
  }
  $("day").replaceChildren(...opts);
  const t = new Date(now.getTime() + 5 * 60e3);
  $("time").value = `${String(t.getHours()).padStart(2, "0")}:${String(Math.floor(t.getMinutes() / 5) * 5).padStart(2, "0")}`;
}

async function chooseRoute(id, dir, stop) {
  sel.r = id || null;
  $("route").value = id || "";
  $("dir-field").hidden = $("stop-field").hidden = $("when-field").hidden = $("results").hidden = true;
  if (!id) return;
  const body = await routeBody(id);
  if (!body || sel.r !== id) return;
  const dirs = Object.entries(body.dirs);
  $("dirs").replaceChildren(...dirs.map(([d, v]) => el("button", { type: "button", role: "radio", textContent: v.name, onclick: () => chooseDir(d) })));
  $("dir-field").hidden = false;
  await chooseDir(dir != null && body.dirs[dir] ? String(dir) : dirs.length === 1 ? dirs[0][0] : null, stop);
}

async function chooseDir(d, stop) {
  sel.d = d;
  const body = await routeBody(sel.r);
  const keys = Object.keys(body.dirs);
  [...$("dirs").children].forEach((b, i) => b.setAttribute("aria-checked", String(keys[i] === d)));
  if (d == null) { $("stop-field").hidden = $("when-field").hidden = $("results").hidden = true; return; }
  const stops = body.dirs[d].stops;
  $("stop").replaceChildren(el("option", { value: "", textContent: "Choose your stop" }),
    ...stops.map((s, i) => el("option", { value: s[0], textContent: `${i + 1}. ${s[1]}` })));
  $("stop-field").hidden = false;
  chooseStop(stop && stops.some((s) => s[0] === stop) ? stop : null);
}

function chooseStop(s) {
  sel.s = s;
  $("stop").value = s || "";
  $("when-field").hidden = !s;
  $("results").hidden = !s;
  if (!s) return;
  history.replaceState(null, "", `#r=${encodeURIComponent(sel.r)}&d=${sel.d}&s=${encodeURIComponent(s)}`);
  store.set("last", { ...sel });
  render();
}

// ---------- predictions ----------
function verdict(p) {
  if (p == null) return ["none", "Scheduled"];
  if (p.late >= 0.45) return ["bad", "Often late"];
  if (p.early >= 0.2) return ["bad", "Often leaves early"];
  if (p.early >= 0.1) return ["warn", "Can leave early"];
  const mins = Math.round(p.median);
  if (mins < 2 && p.late < 0.2) return ["good", "Usually on time"];
  if (mins < 2) return ["warn", "Sometimes late"];
  return ["warn", `Usually ~${mins} min late`];
}

async function render() {
  const token = {};
  render.token = token;
  const body = await routeBody(sel.r);
  const dirBody = body.dirs[sel.d];
  const col = dirBody.stops.findIndex((s) => s[0] === sel.s);
  const day = $("day").value;
  const running = new Set(cal.services[day] || []);
  const deps = dirBody.trips.filter((t) => t.t[col] != null && running.has(body.svc[t.s])).sort((a, b) => a.t[col] - b.t[col]);
  const dayName = $("day").selectedOptions[0]?.textContent || day;
  $("results-title").textContent = `${routeLabel(routeById[sel.r])} · ${dirBody.name} · ${dirBody.stops[col][1]}`;
  updateSaveButton();

  const [hh, mm] = ($("time").value || "07:00").split(":").map(Number);
  const from = hh * 60 + mm - WINDOW_BEFORE;
  let shown = showAll ? deps : deps.filter((t) => t.t[col] >= from).slice(0, WINDOW_COUNT);
  if (!showAll && !shown.length) shown = deps.slice(-WINDOW_COUNT);
  $("more").textContent = showAll ? "Show fewer" : "Show the whole day";
  $("more").hidden = deps.length <= WINDOW_COUNT;

  const m = meta.model ? await loadModels() : null;
  const wx = m ? await loadWeather() : null;
  if (render.token !== token) return;
  const dir = routeById[sel.r].dirs.find((x) => String(x.id) === sel.d) || {};
  const now = new Date();
  const nowMin = day === localISO(now) ? now.getHours() * 60 + now.getMinutes() : -1;
  const preds = shown.map((trip) => {
    if (!m) return null;
    const row = featureRow({ dateStr: day, dir, trip, col, stopMean: dirBody.stops[col][2], meta, holidays: cal.holidays, weather: wx });
    return { median: m.median.predict(row), late: m.late.predict(row), early: m.early.predict(row), known: !!trip.h };
  });

  $("deps").replaceChildren(...shown.map((trip, i) => {
    const p = preds[i];
    const [cls, text] = verdict(p);
    const t = trip.t[col];
    const detail = p ? el("div", { className: "detail" },
      `Typically ${delayText(p.median)} · 5+ min late ${pct(p.late)} of the time`,
      p.early >= 0.1 ? el("div", { className: "early", textContent: `Leaves early ${pct(p.early)} of the time. Get to the stop a few minutes before ${clock(t)}.` }) : null,
      p.known ? null : el("div", { textContent: "Limited history for this exact trip; based on the route and stop." })) : null;
    return el("li", { className: "dep" + (t < nowMin ? " past" : "") },
      el("div", { className: "time" }, clock(t), t >= 1440 ? el("small", { textContent: "after midnight" }) : null),
      el("div", {}, el("span", { className: `pill ${cls}`, textContent: text })),
      detail);
  }));
  if (!deps.length) {
    $("deps").replaceChildren(el("li", { className: "muted", textContent: `No scheduled departures from this stop ${day === localISO(now) ? "today" : "on " + dayName}.` }));
  }

  // actionable tip: the most and least reliable upcoming departures in view, when the choice matters
  const upcoming = preds.map((p, i) => [p, shown[i]]).filter(([p, t]) => p && t.t[col] >= nowMin);
  const tip = $("tip");
  tip.hidden = true;
  if (upcoming.length >= 2) {
    const risk = ([p]) => p.late + 2 * p.early;
    const best = upcoming.reduce((a, b) => (risk(b) < risk(a) ? b : a));
    const worst = upcoming.reduce((a, b) => (risk(b) > risk(a) ? b : a));
    if (risk(worst) - risk(best) >= 0.15) {
      const early = (p) => (p.early >= 0.05 ? `, leaves early ${pct(p.early)}` : "");
      tip.hidden = false;
      tip.textContent = `Most reliable here: the ${clock(best[1].t[col])} (5+ min late ${pct(best[0].late)} of the time${early(best[0])}). ` +
        `Least reliable: the ${clock(worst[1].t[col])} (late ${pct(worst[0].late)}${early(worst[0])}).`;
    }
  }
}

// ---------- saved trips ----------
const saved = () => store.get("trips", []);
const sameTrip = (a) => a.r === sel.r && a.d === sel.d && a.s === sel.s;
function updateSaveButton() { $("save").textContent = saved().some(sameTrip) ? "★ Saved" : "☆ Save trip"; }
function renderSaved() {
  const trips = saved();
  $("saved-wrap").hidden = !trips.length;
  $("saved").replaceChildren(...trips.map((t, i) => el("button", {
    type: "button", title: "Show this trip",
    onclick: (e) => {
      if (e.target.classList.contains("x")) { store.set("trips", trips.filter((_, j) => j !== i)); renderSaved(); updateSaveButton(); return; }
      showAll = false;
      chooseRoute(t.r, t.d, t.s);
    },
  }, t.label, el("span", { className: "x", textContent: "×", title: "Remove" }))));
}
$("save").addEventListener("click", async () => {
  const trips = saved();
  if (trips.some(sameTrip)) store.set("trips", trips.filter((t) => !sameTrip(t)));
  else {
    const body = await routeBody(sel.r);
    const stop = body.dirs[sel.d].stops.find((s) => s[0] === sel.s);
    const r = routeById[sel.r];
    store.set("trips", [...trips, { ...sel, label: `${r.short || r.long} ${body.dirs[sel.d].name.split(" ")[0]} · ${stop[1]}` }]);
  }
  renderSaved();
  updateSaveButton();
});

// ---------- accuracy ----------
function renderAccuracy(m, live) {
  if (!m) return;
  $("bt-text").textContent = `Backtest on ${m.test_days[0]} to ${m.test_days[1]}: ${m.n.toLocaleString()} observed departures the model didn't train on. ` +
    "Typical error is the average gap between predicted and actual delay, in minutes.";
  const rows = [["Assume every bus is on time", m.mae_on_time, null], ["The route's average", m.mae_route_history, null],
    ["That trip's own average", m.mae_trip_history, m.late_auc_trip_history], ["This model", m.mae, m.late_auc, true]];
  $("bt").innerHTML = "<tr><th>Predictor</th><th>Typical error (min)</th><th>Spotting 5+ min late (AUC)</th></tr>" +
    rows.map(([n, e, a, hl]) => `<tr class="${hl ? "hl" : ""}"><td>${n}</td><td>${e.toFixed(2)}</td><td>${a == null ? "–" : a.toFixed(3)}</td></tr>`).join("");
  if (live?.length) {
    $("live").innerHTML = "<tr><th>Day</th><th>Departures</th><th>Typical error</th><th>5+ min late: predicted / actual</th></tr>" +
      live.slice(-14).reverse().map((d) => `<tr><td>${d.date}</td><td>${d.n.toLocaleString()}</td><td>${d.mae.toFixed(2)} min</td>` +
        `<td>${d.late_mean_pred == null ? "–" : pct(d.late_mean_pred)} / ${pct(d.late_share)}</td></tr>`).join("");
  } else {
    $("live").outerHTML = '<p class="small muted">Daily scores start the day after the first model is published.</p>';
  }
  $("accuracy").hidden = false;
}

// ---------- init ----------
async function init() {
  const [m, r, c, metrics, live] = await Promise.all(["meta", "routes", "calendar", "metrics", "live"].map((n) => getJSON(`data/${n}.json`)));
  meta = m; routes = r?.routes; cal = c;
  const notice = $("notice");
  if (!meta || !routes || !cal) {
    notice.hidden = false;
    notice.textContent = "Schedules haven't been published yet. Check back soon.";
    $("route").replaceChildren(el("option", { textContent: "Not available yet" }));
    return;
  }
  if (!meta.model) {
    notice.hidden = false;
    notice.textContent = `Still learning: ${meta.days_collected} of ${meta.min_days} days of Metro Transit data collected. ` +
      "You can browse schedules now; delay predictions appear once there's enough history.";
  } else if (meta.stale_since) {
    notice.hidden = false;
    notice.textContent = "Data collection paused recently, so predictions may be a little out of date.";
  }
  $("updated").textContent = `${meta.observations.toLocaleString()} observed departures over ${meta.days_collected} days` +
    (meta.first_day ? ` since ${meta.first_day}` : "") + `. Updated ${meta.updated.replace("T", " ")}.`;
  routes.forEach((x) => { routeById[x.id] = x; });
  fillRoutes();
  fillDays();
  renderSaved();
  renderAccuracy(meta.model ? metrics : null, live);

  $("route").addEventListener("change", (e) => { showAll = false; chooseRoute(e.target.value); });
  $("stop").addEventListener("change", (e) => { showAll = false; chooseStop(e.target.value || null); });
  $("day").addEventListener("change", () => sel.s && render());
  $("time").addEventListener("change", () => { showAll = false; if (sel.s) render(); });
  $("more").addEventListener("click", () => { showAll = !showAll; render(); });

  const h = new URLSearchParams(location.hash.slice(1));
  const start = h.get("r") ? { r: h.get("r"), d: h.get("d"), s: h.get("s") } : store.get("last", null);
  if (start?.r && routeById[start.r]) chooseRoute(start.r, start.d, start.s);
}

init();
