"""Small synthetic static GTFS feed for tests."""
import zipfile

ROUTES = {"21": ("21", "Lake St / Selby Av"), "902": ("Green", "METRO Green Line")}


def trips():
    """(trip_id, route, direction, service, start minute) - every 20 min 5:00-23:00 each way, weekday + weekend."""
    out = []
    for route in ROUTES:
        for d in (0, 1):
            for svc in ("WK", "WE"):
                for m in range(300, 1380, 20):
                    out.append((f"{route}-{d}-{svc}-{m}", route, d, svc, m))
    return out


STOPS = [f"S{i}" for i in range(10)]


def write_gtfs(path, extra_stop_times=""):
    st = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence"]
    for tid, route, d, svc, m in trips():
        stops = STOPS if d == 0 else STOPS[::-1]
        for k, s in enumerate(stops):
            t = (m + 4 * k) * 60
            hh = f"{t // 3600:02d}:{t % 3600 // 60:02d}:00"
            st.append(f"{tid},{hh},{hh},{s},{k + 1}")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("routes.txt", "route_id,route_short_name,route_long_name,route_type\n"
                   + "".join(f"{r},{s},{l},3\n" for r, (s, l) in ROUTES.items()))
        z.writestr("trips.txt", "route_id,service_id,trip_id,direction_id,trip_headsign\n"
                   + "".join(f"{r},{svc},{t},{d},{'East' if d == 0 else 'West'} to S{9 if d == 0 else 0}\n" for t, r, d, svc, _ in trips()))
        z.writestr("stops.txt", "stop_id,stop_name\n" + "".join(f"{s},Stop {s}\n" for s in STOPS))
        z.writestr("stop_times.txt", "\n".join(st) + "\n" + extra_stop_times)
        z.writestr("calendar.txt", "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
                   "WK,1,1,1,1,1,0,0,20260101,20271231\nWE,0,0,0,0,0,1,1,20260101,20271231\n")
    return path
