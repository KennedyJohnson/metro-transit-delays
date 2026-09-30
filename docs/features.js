// Mirror of model/features.py build() for one departure. tests/test_parity.py checks the two agree.
const WX = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"];

function daytype(dow) { return dow < 5 ? "wk" : dow === 5 ? "sat" : "sun"; }

// dateStr "YYYY-MM-DD" (service day); dir: routes.json dir entry; trip: route-file trip; col: stop column
// weather: {"YYYY-MM-DDTHH:00": {temperature_2m, ...}} or null
function featureRow({ dateStr, dir, trip, col, stopMean, meta, holidays, weather }) {
  const [y, m, d] = dateStr.split("-").map(Number);
  const dow = (new Date(Date.UTC(y, m - 1, d)).getUTCDay() + 6) % 7;
  const g = meta.global;
  const idx = trip.t.map((v, i) => (v == null ? -1 : i)).filter((i) => i >= 0);
  const stopIdx = idx.indexOf(col);
  const schedMin = trip.t[col];
  const rdMean = dir.mean ?? g.delay_min, rdLate = dir.late ?? g.late, rdEarly = dir.early ?? g.early;
  const h = trip.h?.[daytype(dow)];
  const row = {
    hour: (schedMin / 60) % 24,
    dow,
    holiday: holidays.includes(dateStr) ? 1 : 0,
    progress: stopIdx / Math.max(idx.length - 1, 1),
    stop_idx: stopIdx,
    start_hour: trip.start / 60,
    rd_mean: rdMean, rd_late: rdLate, rd_early: rdEarly,
    trip_mean: h ? h[0] : rdMean, trip_late: h ? h[1] : rdLate, trip_early: h ? h[2] : rdEarly,
    trip_n: Math.log1p(h ? h[3] : 0),
    stop_mean: stopMean ?? rdMean,
    recent7: dir.recent7 ?? NaN,
  };
  const t = new Date(Date.UTC(y, m - 1, d) + Math.floor(schedMin / 60) * 3600e3).toISOString().slice(0, 13) + ":00";
  const w = weather?.[t];
  for (const k of WX) row[k] = w && w[k] != null ? w[k] : NaN;
  return row;
}

if (typeof module !== "undefined") module.exports = { featureRow, daytype };
