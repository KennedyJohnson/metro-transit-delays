const $ = (id) => document.getElementById(id);
const getJSON = (u) => fetch(u, { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
const fmt = (m) => (m == null ? "–" : `${m >= 0 ? "" : "−"}${Math.abs(m).toFixed(1)} min`);
const hourLabel = (h) => (h % 12 || 12) + (h < 12 ? "a" : "p");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function chart(hours, pred, typical) {
  const W = Math.max(300, Math.min(760, $("chart").clientWidth || 760)), H = W < 500 ? 220 : 260, L = 44, R = 12, T = 12, B = 28;
  const vals = [...pred, ...typical].filter((v) => v != null);
  const lo = Math.min(0, ...vals), hi = Math.max(1, ...vals) * 1.1;
  const x = (i) => L + (i * (W - L - R)) / (hours.length - 1);
  const y = (v) => T + ((hi - v) * (H - T - B)) / (hi - lo);
  const path = (arr) => arr.map((v, i) => (v == null ? null : `${x(i)},${y(v)}`)).filter(Boolean).join(" ");
  const step = hi - lo > 8 ? 2 : 1;
  let grid = "";
  for (let v = Math.ceil(lo); v <= hi; v += step) {
    grid += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  }
  const xl = hours.map((h, i) => (i % (W < 500 ? 4 : 3) === 0 ? `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${hourLabel(h)}</text>` : "")).join("");
  return `<svg viewBox="0 0 ${W} ${H}" aria-label="Forecast minutes late by hour">${grid}${xl}
    <polyline fill="none" stroke="var(--typ)" stroke-width="2" stroke-dasharray="4 3" points="${path(typical)}"/>
    <polyline fill="none" stroke="var(--accent)" stroke-width="2.5" points="${path(pred)}"/></svg>`;
}

function daytimeMean(fc, arr) {
  const v = fc.hours.map((h, i) => (h >= 6 && h <= 19 ? arr[i] : null)).filter((x) => x != null);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

function showRoute(fc, id) {
  const r = fc.routes[id];
  $("route").value = id;
  $("chart-title").textContent = r.name;
  $("chart-sub").textContent = `Forecast for ${fc.date}, issued ${fc.issued.replace("T", " ")}. Minutes behind schedule; positive = late.`;
  $("chart").innerHTML = chart(fc.hours, r.pred, r.typical);
}

async function init() {
  const [status, fc, m, live] = await Promise.all(["status", "forecast", "metrics", "live"].map((n) => getJSON(`data/${n}.json`)));
  if (status) $("updated").textContent = `Data: ${status.observations.toLocaleString()} trip snapshots over ${status.days_collected} days since ${status.first_day}. Updated ${status.updated.replace("T", " ")}.`;
  if (!status || status.collecting || !fc) {
    $("collecting").hidden = false;
    $("collecting-text").textContent = status
      ? `The collector has ${status.days_collected} day${status.days_collected === 1 ? "" : "s"} of Metro Transit data so far. Forecasts start once there are ${status.min_days} days to learn from.`
      : "The collector hasn't recorded any data yet.";
    return;
  }

  const ids = Object.keys(fc.routes).sort((a, b) => fc.routes[a].name.localeCompare(fc.routes[b].name, undefined, { numeric: true }));
  $("route").replaceChildren(...ids.map((id) => Object.assign(document.createElement("option"), { value: id, textContent: fc.routes[id].name })));
  const ranked = ids.map((id) => [id, daytimeMean(fc, fc.routes[id].pred), daytimeMean(fc, fc.routes[id].typical)])
    .filter(([, p]) => p != null).sort((a, b) => b[1] - a[1]);
  $("rank").innerHTML = `<thead><tr><th>Route</th><th>Forecast today</th><th>Typical</th></tr></thead><tbody>` +
    ranked.slice(0, 15).map(([id, p, t]) => `<tr data-id="${esc(id)}"><td>${esc(fc.routes[id].name)}</td><td>${fmt(p)}</td><td>${fmt(t)}</td></tr>`).join("") + "</tbody>";
  $("rank").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (tr) { showRoute(fc, tr.dataset.id); $("route").scrollIntoView({ behavior: "smooth", block: "center" }); }
  });
  $("route").addEventListener("change", (e) => showRoute(fc, e.target.value));
  $("today").hidden = false; // before drawing, so the chart can measure its width
  showRoute(fc, ranked[0]?.[0] ?? ids[0]);

  if (m) {
    $("bt-text").textContent = `Backtest on ${m.test_days[0]} to ${m.test_days[1]} (${m.n.toLocaleString()} route-hours the model didn't train on). Average absolute error, in minutes:`;
    $("bt").innerHTML = `<tr><th>Predictor</th><th>Error (lower is better)</th></tr>` +
      [["Same hour last week", m.baseline_last_week_mae], ["14-day average for that route and hour", m.baseline_14day_mean_mae], ["This model", m.model_mae, true]]
        .map(([n, v, hl]) => `<tr class="${hl ? "hl" : ""}"><td>${n}</td><td>${v.toFixed(2)}</td></tr>`).join("");
  }
  if (live?.length) {
    $("live").innerHTML = `<tr><th>Day</th><th>Route-hours</th><th>Model error</th><th>Last-week baseline</th></tr>` +
      live.slice(-14).reverse().map((d) => `<tr><td>${d.date}</td><td>${d.n}</td><td>${d.model_mae.toFixed(2)}</td><td>${d.baseline_last_week_mae.toFixed(2)}</td></tr>`).join("");
  } else {
    $("live").outerHTML = `<p class="small muted">The first daily scores appear the day after the first forecast.</p>`;
  }
  $("accuracy").hidden = false;
}

init();
