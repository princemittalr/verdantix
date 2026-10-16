"""Access control, rate limiting and upload limits.

Roles
- public:   view the twin, submit problem reports, ask the copilot, run what-if (rate limited)
- maintainer: maintenance staff (MAINTAINER_TOKEN). Work orders only: acknowledge, start, comment, upload proof.
- operator: facilities staff / administrator. Clock and demo controls, assign/dismiss/label alerts, work orders, CCTV feeds.
            Sent as the X-Operator header (set OPERATOR_TOKEN on the server; never in frontend code).
- sensor:   IoT devices posting to /api/ingest or /api/cctv (X-Token = INGEST_TOKEN, or the operator token).
If OPERATOR_TOKEN is not set, operator checks are OFF (local development) and /api/about says so.
"""
import hmac
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

import config as C


def _eq(a: str, b: str) -> bool:
    return bool(a) and bool(b) and hmac.compare_digest(a.encode(), b.encode())


def auth_misconfigured() -> bool:
    """Production with no operator token: privileged actions are refused (fail closed)."""
    return C.PRODUCTION and not C.OPERATOR_TOKEN


def is_operator(token: str) -> bool:
    if not C.OPERATOR_TOKEN:
        return not C.PRODUCTION   # open only in local development
    return _eq(token, C.OPERATOR_TOKEN)


def require_operator(x_operator: str = Header("")):
    if auth_misconfigured():
        raise HTTPException(503, "Operator actions are disabled: OPERATOR_TOKEN is not configured on this server.")
    if not is_operator(x_operator):
        raise HTTPException(401, "Operator sign-in needed for this action.")
    return "operator" if C.OPERATOR_TOKEN else "operator (auth off, local dev)"


def role_of(token: str) -> str:
    """Server-side role from the passcode: 'admin', 'maintainer' or 'public'. The client's claim is never trusted."""
    if is_operator(token):
        return "admin"
    if C.MAINTAINER_TOKEN and _eq(token, C.MAINTAINER_TOKEN):
        return "maintainer"
    return "public"


def require_maintainer(x_operator: str = Header("")):
    """Work-order actions: maintainers and administrators."""
    if auth_misconfigured():
        raise HTTPException(503, "Staff actions are disabled: OPERATOR_TOKEN is not configured on this server.")
    r = role_of(x_operator)
    if r == "public":
        raise HTTPException(401, "Maintainer or operator sign-in needed for this action.")
    return "operator" if r == "admin" else "maintainer"


def require_sensor(x_token: str = Header(""), x_operator: str = Header("")):
    sensor_key = C.INGEST_TOKEN or C.OPERATOR_TOKEN
    if not sensor_key:
        if C.PRODUCTION:
            raise HTTPException(503, "Sensor ingestion is disabled: INGEST_TOKEN is not configured on this server.")
        return "sensor (auth off, local dev)"
    if not (_eq(x_token, sensor_key) or _eq(x_operator, C.OPERATOR_TOKEN)):
        raise HTTPException(401, "Missing or wrong X-Token header.")
    return "sensor"


def client_ip(request: Request) -> str:
    """Client address for rate limiting, from headers the client can't forge where possible (see config)."""
    h = request.headers
    direct = request.client.host if request.client else "unknown"
    if C.TRUST_EDGE_HEADERS:
        edge = h.get("cf-connecting-ip") or h.get("true-client-ip")
        if edge:
            return edge.strip()[:64]
    parts = [p.strip() for p in h.get("x-forwarded-for", "").split(",") if p.strip()]
    if C.TRUSTED_PROXY_HOPS > 0 and len(parts) >= C.TRUSTED_PROXY_HOPS:
        return parts[-C.TRUSTED_PROXY_HOPS][:64]
    if C.TRUST_EDGE_HEADERS and parts:
        return parts[0][:64]   # Render puts the real client first; spoofable only when the edge headers are missing
    return direct[:64]


class RateLimiter:
    """Sliding-window limit per (bucket, client IP)."""

    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, bucket: str, ip: str, limit: int, window_s: float = 60.0):
        now = time.monotonic()
        with self._lock:
            q = self._hits[(bucket, ip)]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                wait = int(window_s - (now - q[0])) + 1
                raise HTTPException(429, f"Too many requests. Try again in {wait} s.")
            q.append(now)
            if len(self._hits) > 20000:   # forget idle clients
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > window_s]:
                    self._hits.pop(k, None)


limiter = RateLimiter()
LIMITS = {  # requests per minute per IP
    "report": 6, "vision": 30, "cctv": 40, "copilot": 12, "weekly": 4, "simulate": 90, "ingest": 240, "signin": 8,
}


SCALE = float(os.getenv("RATE_LIMIT_SCALE", "1"))   # raise for load tests; never set below 1 in production


def limit(bucket: str):
    def dep(request: Request):
        limiter.check(bucket, client_ip(request), max(1, int(LIMITS[bucket] * SCALE)))
    return dep


async def read_upload(file, max_mb: float | None = None) -> bytes:
    """Read an upload without letting it grow past the size limit."""
    max_bytes = int((max_mb or C.MAX_UPLOAD_MB) * 1024 * 1024)
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File is too large. The limit is {max_mb or C.MAX_UPLOAD_MB:g} MB.")
    if not data:
        raise HTTPException(400, "The file is empty.")
    return data
