"""Persistence: operational records survive restarts.

SQLite (one file, no server). Alerts, work orders, problem reports, green credits and
confirmations are snapshotted every few seconds when they change; the audit log is
written immediately and is append-only. On start-up the twin reloads them.

Path: VERDANTIX_DB (default backend/data/state/verdantix.db). On a host whose disk is
wiped on redeploy (e.g. Render free), point it at a persistent disk if you have one.
Backup: copy the .db file (safe while running: WAL mode), e.g.  sqlite3 verdantix.db ".backup b.db"
"""
import json
import sqlite3
import threading
import time
from dataclasses import asdict
from datetime import datetime

SCHEMA_VERSION = 1


class Store:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        self._lock = threading.Lock()
        with self._lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS snapshot (key TEXT PRIMARY KEY, json TEXT NOT NULL, updated TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, actor TEXT, ip TEXT,
                    action TEXT NOT NULL, target TEXT, detail TEXT);
                CREATE INDEX IF NOT EXISTS audit_target ON audit(target);
            """)
            row = self.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:   # migrations go here when the schema changes
                self.db.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            self.db.commit()
        self._last = {}

    # ----- snapshots -----
    def save(self, key, obj):
        js = json.dumps(obj, default=str, separators=(",", ":"))
        if self._last.get(key) == js:
            return False
        with self._lock:
            self.db.execute("INSERT INTO snapshot VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET json=excluded.json, updated=excluded.updated",
                            (key, js, datetime.now().isoformat(timespec="seconds")))
            self.db.commit()
        self._last[key] = js
        return True

    def load(self, key, default=None):
        with self._lock:
            row = self.db.execute("SELECT json FROM snapshot WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        self._last[key] = row[0]
        return json.loads(row[0])

    # ----- audit log -----
    def audit(self, action, target="", actor="", ip="", detail=""):
        with self._lock:
            self.db.execute("INSERT INTO audit (at, actor, ip, action, target, detail) VALUES (?, ?, ?, ?, ?, ?)",
                            (datetime.now().isoformat(timespec="seconds"), actor, ip, action, target,
                             detail if isinstance(detail, str) else json.dumps(detail, default=str)))
            self.db.commit()

    def audit_list(self, limit=100, target=None):
        q, args = "SELECT id, at, actor, ip, action, target, detail FROM audit", []
        if target:
            q += " WHERE target=?"
            args.append(target)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self.db.execute(q, args).fetchall()
        return [dict(zip(("id", "at", "actor", "ip", "action", "target", "detail"), r)) for r in rows]


# ---------- twin <-> store ----------
def snapshot(twin, desk):
    with twin.lock:
        alerts = [asdict(a) for a in twin.alerts.values()]
        orders = [asdict(o) for o in twin.orders.values()]
    reports = []
    for r in desk.reports:
        d = {k: v for k, v in r.items() if k not in ("_hash", "_emb", "_confirmers")}
        d["_hash"] = str(r["_hash"]) if r.get("_hash") is not None else None
        d["_emb"] = [round(float(x), 5) for x in r["_emb"]] if r.get("_emb") is not None else None
        d["_confirmers"] = sorted(r.get("_confirmers", ()))
        reports.append(d)
    return {"alerts": alerts, "orders": orders, "reports": reports, "points": desk.points}


def restore(store, twin, desk):
    """Load saved records into a fresh twin. Returns counts."""
    import itertools

    import numpy as np
    from twin.engine import Alert, WorkOrder
    alerts = store.load("alerts", []) or []
    orders = store.load("orders", []) or []
    reports = store.load("reports", []) or []
    fields_a = Alert.__dataclass_fields__
    fields_o = WorkOrder.__dataclass_fields__
    with twin.lock:
        for d in alerts:
            a = Alert(**{k: v for k, v in d.items() if k in fields_a})
            if a.source == "demo-scenario" and a.status in ("open", "assigned"):
                a.status = "self-cleared"   # demo injections don't survive a restart
                twin.log(a, "self-cleared", "system", "server restarted; demo scenario ended")
            twin.alerts[a.id] = a
        for d in orders:
            o = WorkOrder(**{k: v for k, v in d.items() if k in fields_o})
            twin.orders[o.id] = o
        nums = [int(x.id.lstrip("AWO")) for x in list(twin.alerts.values()) + list(twin.orders.values()) if x.id.lstrip("AWO").isdigit()]
        if nums:
            twin._ids = itertools.count(max(nums) + 1)
    try:
        import imagehash
    except ImportError:
        imagehash = None
    for d in reports:
        d["_hash"] = imagehash.hex_to_hash(d["_hash"]) if d.get("_hash") and imagehash else None
        d["_emb"] = np.array(d["_emb"], dtype=np.float32) if d.get("_emb") else None
        d["_confirmers"] = set(d.get("_confirmers") or [])
        desk.reports.append(d)
    if reports:
        desk._ids = itertools.count(len(reports) + 1)
    desk.points.update(store.load("points", {}) or {})
    return {"alerts": len(alerts), "orders": len(orders), "reports": len(reports)}


def start_autosave(store, twin, desk, every_s=4.0):
    def loop():
        while True:
            time.sleep(every_s)
            try:
                snap = snapshot(twin, desk)
                for k, v in snap.items():
                    store.save(k, v)
            except Exception as e:  # never kill the server over a save
                print("autosave failed:", e)
    threading.Thread(target=loop, daemon=True, name="autosave").start()


def flush(store, twin, desk):
    for k, v in snapshot(twin, desk).items():
        store.save(k, v)
