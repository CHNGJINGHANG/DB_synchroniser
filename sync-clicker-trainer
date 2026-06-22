"""
Sync Clicker Trainer
=====================
A real-time multi-user synchronisation training tool.

ARCHITECTURE:
- A single Flask process serves both the plain HTML pages (join screen +
  live room) AND the JSON API (/api/state, /api/click) on one port.
  The canvas lives directly in the page -- no iframes, no client-side
  framework, just plain HTML/CSS/JS talking to a small Flask backend.
- Shared room/click state lives in a process-level dict guarded by a
  lock, polled by the page's own JS rather than via any server push.

PERFORMANCE NOTES (these matter a lot for a page with a live canvas --
without them, the browser tab can end up redrawing far more than it
needs to and noticeably heat up the machine):
- The draw loop is capped to TARGET_FPS instead of running at the
  display's raw refresh rate, and is fully paused when the tab is
  hidden (Page Visibility API).
- The canvas's pixel backing store is capped at MAX_DPR (1.5) instead
  of using the raw devicePixelRatio (2-3 on Retina/4K screens), since
  canvas rendering cost scales with total pixel count -- an uncapped
  DPR can mean rendering up to ~9x more pixels per frame than needed.
- The stats panel (strokes/min boxes) is DOM-updated (innerHTML) only
  a few times a second and skipped entirely if unchanged, since DOM
  rebuilds are far more expensive than a canvas paint.
- Click history is pruned to the visible window so the per-frame pulse
  computation stays bounded instead of growing over a long session.

RUN:
    pip install flask
    python sync_clicker_trainer.py
    -> open http://localhost:8765  (and that same URL on other devices
       on your network, using your machine's LAN IP instead of localhost)
"""

import os
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock

from flask import Flask, request, jsonify, render_template_string

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
PORT = int(os.environ.get("PORT", 8765))  # hosting platforms set PORT themselves
WINDOW_SECONDS = 15.0          # how much time history is visible on screen
PULSE_A = 5.0                  # pulse amplitude scale
PULSE_K = 8.0                  # pulse decay rate
SPS_WINDOW = 1.0               # sliding window (seconds) used to compute strokes-per-minute
TARGET_FPS = 30                # cap the canvas redraw rate (display refresh rate is overkill)
CURVE_SAMPLES = 90             # points per curve -- visually smooth, cheap to compute
MAX_DPR = 1.5                  # cap canvas pixel density (uncapped DPR can be 2-3x on Retina/4K)
POLL_MS_ACTIVE = 300           # how often the page polls /api/state while visible
POLL_MS_HIDDEN = 2000          # how often it polls while the tab is hidden/backgrounded
COLOR_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bfef45", "#fabed4", "#469990",
]

app = Flask(__name__)


# --------------------------------------------------------------------------
# SHARED ROOM STORE  (process-level, guarded by a lock)
# --------------------------------------------------------------------------
@dataclass
class RoomState:
    users: dict = field(default_factory=dict)  # user_id -> {"name", "color", "clicks": [...]}
    created_at: float = field(default_factory=time.time)
    admin_id: str | None = None     # first person to join a room becomes its admin
    started: bool = False           # clicking is only allowed once the admin starts the session
    started_at: float | None = None


_rooms: dict[str, RoomState] = {}
_lock = Lock()


def get_or_create_room(room_id: str) -> RoomState:
    with _lock:
        if room_id not in _rooms:
            _rooms[room_id] = RoomState()
        return _rooms[room_id]


def join_room(room_id: str, name: str):
    """Registers a user in a room. The first person to join becomes admin.
    Returns (user_id, is_admin)."""
    room = get_or_create_room(room_id)
    user_id = str(uuid.uuid4())[:8]
    with _lock:
        is_admin = room.admin_id is None
        if is_admin:
            room.admin_id = user_id
        color = COLOR_PALETTE[len(room.users) % len(COLOR_PALETTE)]
        room.users[user_id] = {"name": name, "color": color, "clicks": []}
    return user_id, is_admin


def start_room(room_id: str, user_id: str) -> bool:
    """Only the room's admin can start the session. Returns True on success,
    False if the caller isn't the admin."""
    room = get_or_create_room(room_id)
    with _lock:
        if room.admin_id != user_id:
            return False
        room.started = True
        room.started_at = time.time()
        return True


def register_click(room_id: str, user_id: str) -> bool:
    """Returns False (and does nothing) if the session hasn't been started
    yet -- clicking is gated server-side, not just hidden in the UI."""
    room = get_or_create_room(room_id)
    with _lock:
        if not room.started:
            return False
        if user_id in room.users:
            room.users[user_id]["clicks"].append(time.time())
            _prune_clicks_locked(room, time.time())
            return True
    return False


def _prune_clicks_locked(room: RoomState, now: float):
    """Drop clicks older than the visible window. Caller must hold _lock.
    Keeps per-frame pulse computation bounded instead of growing forever
    over a long session."""
    cutoff = now - WINDOW_SECONDS - 1.0
    for u in room.users.values():
        clicks = u["clicks"]
        if clicks and clicks[0] < cutoff:
            u["clicks"] = [c for c in clicks if c >= cutoff]


# --------------------------------------------------------------------------
# JSON API
# --------------------------------------------------------------------------
@app.get("/api/state")
def api_state():
    room_id = request.args.get("room", "")
    room = get_or_create_room(room_id)
    now = time.time()
    with _lock:
        _prune_clicks_locked(room, now)
        payload = {
            "serverNow": now,
            "adminId": room.admin_id,
            "started": room.started,
            "users": [
                {"id": uid, "name": u["name"], "color": u["color"], "clicks": u["clicks"]}
                for uid, u in room.users.items()
            ],
        }
    return jsonify(payload)


@app.post("/api/click")
def api_click():
    data = request.get_json(silent=True) or {}
    room_id = data.get("room")
    user_id = data.get("user_id")
    if not room_id or not user_id:
        return jsonify({"ok": False, "error": "missing room or user_id"}), 400
    accepted = register_click(room_id, user_id)
    if not accepted:
        return jsonify({"ok": False, "error": "session has not been started yet"}), 403
    return jsonify({"ok": True})


@app.post("/api/start")
def api_start():
    data = request.get_json(silent=True) or {}
    room_id = data.get("room")
    user_id = data.get("user_id")
    if not room_id or not user_id:
        return jsonify({"ok": False, "error": "missing room or user_id"}), 400
    ok = start_room(room_id, user_id)
    if not ok:
        return jsonify({"ok": False, "error": "only the room's admin can start the session"}), 403
    return jsonify({"ok": True, "started": True})


@app.post("/api/join")
def api_join():
    data = request.get_json(silent=True) or {}
    room_id = (data.get("room") or "").strip()
    name = (data.get("name") or "").strip()
    if not room_id or not name:
        return jsonify({"error": "room and name are required"}), 400
    user_id, is_admin = join_room(room_id, name)
    return jsonify({"user_id": user_id, "room_id": room_id, "name": name, "is_admin": is_admin})


# --------------------------------------------------------------------------
# PAGES
# --------------------------------------------------------------------------
JOIN_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sync Clicker Trainer</title>
  <style>
    body { background:#0e1117; color:#eee; font-family:sans-serif; display:flex;
           align-items:center; justify-content:center; height:100vh; margin:0; }
    .card { background:#161a23; padding:32px; border-radius:12px; width:360px; }
    h1 { font-size:22px; margin-top:0; }
    label { display:block; font-size:13px; color:#aaa; margin:14px 0 4px; }
    input { width:100%; box-sizing:border-box; padding:10px; border-radius:6px;
            border:1px solid #333; background:#0e1117; color:#eee; font-size:14px; }
    button { width:100%; margin-top:18px; padding:12px; border:none; border-radius:6px;
              background:#ff4b4b; color:white; font-weight:bold; font-size:15px; cursor:pointer; }
    button.secondary { background:#333; margin-top:8px; }
    #err { color:#ff8080; font-size:13px; margin-top:10px; min-height:16px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🎯 Sync Clicker Trainer</h1>
    <label>Your name</label>
    <input id="name" placeholder="e.g. Alex">
    <label>Room ID (share this with others)</label>
    <input id="room" value="room1">
    <button id="joinBtn">Join Room</button>
    <button id="genBtn" class="secondary">Generate Random Room ID</button>
    <div id="err"></div>
  </div>
  <script>
    document.getElementById("genBtn").addEventListener("click", function() {
        document.getElementById("room").value = Math.random().toString(16).slice(2, 8);
    });
    document.getElementById("joinBtn").addEventListener("click", async function() {
        const name = document.getElementById("name").value.trim();
        const room = document.getElementById("room").value.trim();
        const err = document.getElementById("err");
        if (!name || !room) { err.textContent = "Enter both a name and a room ID."; return; }
        try {
            const res = await fetch("/api/join", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({name: name, room: room})
            });
            const data = await res.json();
            if (!res.ok) { err.textContent = data.error || "Could not join."; return; }
            const params = new URLSearchParams({room: data.room_id, user: data.user_id, name: data.name});
            window.location.href = "/room?" + params.toString();
        } catch (e) {
            err.textContent = "Connection error: " + e.message;
        }
    });
  </script>
</body>
</html>
"""

ROOM_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sync Clicker Trainer — {{ room_id }}</title>
  <style>
    body { background:#0e1117; color:#eee; font-family:sans-serif; margin:0; padding:20px; }
    .topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }
    a.leave { color:#aaa; font-size:13px; text-decoration:none; }
    .panel { background:#0e1117; border-radius:8px; padding:8px; }
    .head { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
    #statusText { color:#888; font-size:12px; }
    #clickBtn { background:#ff4b4b; color:white; border:none; border-radius:6px;
                padding:14px 28px; font-size:16px; font-weight:bold; cursor:pointer; }
    #clickBtn:disabled { background:#3a2f33; color:#888; cursor:not-allowed; }
    canvas { width:100%; height:430px; display:block; border-radius:6px; }
    #spsRow { display:flex; gap:14px; margin-top:14px; flex-wrap:wrap; }
    .cap { color:#777; font-size:12px; margin-top:14px; }
  </style>
</head>
<body>
  <div class="topbar">
    <h2 style="margin:0;">Room: <code>{{ room_id }}</code> · You: <b>{{ user_name }}</b></h2>
    <a class="leave" href="/">👋 Leave room</a>
  </div>

  <div class="panel">
    <div class="head">
      <div id="statusText">connecting...</div>
      <div style="display:flex; gap:10px;">
        <button id="startBtn" style="display:none; background:#3cb44b; color:white; border:none;
                border-radius:6px; padding:14px 24px; font-size:15px; font-weight:bold; cursor:pointer;">
          ▶️ Start Session
        </button>
        <button id="clickBtn" disabled>🖱️ CLICK / TAP</button>
      </div>
    </div>
    <div id="waitBanner" style="display:none; background:#1c2333; border:1px solid #2c3650; border-radius:8px;
         padding:14px 18px; margin-bottom:10px; color:#aac; font-size:14px;">
      ⏳ Waiting for the room admin to start the session...
    </div>
    <canvas id="syncCanvas"></canvas>
    <div id="spsRow"></div>
  </div>

  <p class="cap">Tip: open this same room ID (and this same URL's host) on another device or tab to add a second user.</p>

  <script>
    const cfg = {
        roomId: {{ room_id|tojson }},
        userId: {{ user_id|tojson }},
        userName: {{ user_name|tojson }},
        windowSeconds: {{ window_seconds }},
        A: {{ pulse_a }},
        k: {{ pulse_k }},
        spsWindow: {{ sps_window }},
        targetFps: {{ target_fps }},
        curveSamples: {{ curve_samples }},
        maxDpr: {{ max_dpr }},
        pollMsActive: {{ poll_ms_active }},
        pollMsHidden: {{ poll_ms_hidden }},
    };

    const canvas = document.getElementById("syncCanvas");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("statusText");
    const spsRow = document.getElementById("spsRow");
    const clickBtn = document.getElementById("clickBtn");
    const startBtn = document.getElementById("startBtn");
    const waitBanner = document.getElementById("waitBanner");

    // Admin status and "started" state always come from the server's live
    // poll response, never just trusted from the URL -- so a user can't
    // fake being admin by editing the page, and everyone reacts the moment
    // the real admin clicks Start.
    let isAdmin = false;
    let sessionStarted = false;

    function applyAccessState() {
        if (isAdmin) {
            startBtn.style.display = sessionStarted ? "none" : "inline-block";
        } else {
            startBtn.style.display = "none";
        }
        clickBtn.disabled = !sessionStarted;
        waitBanner.style.display = (!sessionStarted && !isAdmin) ? "block" : "none";
    }
    applyAccessState();

    startBtn.addEventListener("click", async function() {
        startBtn.disabled = true;
        try {
            const res = await fetch("/api/start", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({room: cfg.roomId, user_id: cfg.userId})
            });
            const data = await res.json();
            if (!res.ok) { statusEl.textContent = data.error || "Could not start session."; startBtn.disabled = false; }
        } catch (e) {
            statusEl.textContent = "Start failed: " + e.message;
            startBtn.disabled = false;
        }
    });

    // Cap the effective pixel ratio used for the canvas backing store.
    // Uncapped devicePixelRatio (2-3 on Retina/4K screens) multiplies the
    // actual pixel count up to ~9x, making every frame's clear+redraw+
    // composite far more expensive than it needs to be for this chart.
    const effectiveDpr = Math.min(window.devicePixelRatio || 1, cfg.maxDpr);

    function resize() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * effectiveDpr;
        canvas.height = rect.height * effectiveDpr;
    }
    resize();
    window.addEventListener("resize", resize);

    let latestUsers = [];
    let serverNowAtLastFetch = Date.now() / 1000;
    let clientMsAtLastFetch = performance.now();

    statusEl.textContent = "connecting...";

    let pollTimer = null;
    async function pollState() {
        if (!document.hidden) {
            try {
                const res = await fetch("/api/state?room=" + encodeURIComponent(cfg.roomId));
                if (!res.ok) throw new Error("HTTP " + res.status);
                const data = await res.json();
                latestUsers = data.users;
                serverNowAtLastFetch = data.serverNow;
                clientMsAtLastFetch = performance.now();

                isAdmin = (data.adminId === cfg.userId);
                sessionStarted = !!data.started;
                applyAccessState();  // cheap (a few style/disabled flips), fine every poll tick

                statusEl.textContent = latestUsers.length + " user(s) connected — live" +
                    (isAdmin ? " (you're the admin)" : "");
            } catch (e) {
                statusEl.textContent = "connection error: " + e.message;
            }
        }
        pollTimer = setTimeout(pollState, document.hidden ? cfg.pollMsHidden : cfg.pollMsActive);
    }
    pollState();

    clickBtn.addEventListener("click", function() {
        fetch("/api/click", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({room: cfg.roomId, user_id: cfg.userId})
        }).catch(function(e) { statusEl.textContent = "click failed: " + e.message; });
    });
    canvas.addEventListener("click", function() { clickBtn.click(); });
    window.addEventListener("keydown", function(e) {
        if (e.code === "Space") { e.preventDefault(); clickBtn.click(); }
    });

    function currentServerTime() {
        return serverNowAtLastFetch + (performance.now() - clientMsAtLastFetch) / 1000.0;
    }
    function pulse(t, A, k) {
        if (t < 0) return 0;
        return A * t * Math.exp(-k * t);
    }
    function strokesPerMinute(clicks, now) {
        let count = 0;
        for (const c of clicks) { if (now - c <= cfg.spsWindow) count++; }
        return (count / cfg.spsWindow) * 60.0;
    }

    // Draw loop: capped to cfg.targetFps and fully paused while the tab is
    // hidden, instead of redrawing at full display refresh rate forever.
    const FRAME_INTERVAL = 1000 / cfg.targetFps;
    let lastFrameTime = 0;
    const N = cfg.curveSamples;

    // pulse(t) = A * t * exp(-k*t) decays fast for k=8; beyond this many
    // seconds past a click, its contribution is visually/numerically
    // negligible, so we never need to touch samples further out than this.
    const SUPPORT_SECONDS = 5 / cfg.k;

    // DOM updates (innerHTML rebuilds) are far more expensive than a canvas
    // paint, so the stats panel refreshes on its own slower cadence and
    // skips the write entirely if nothing changed.
    const STATS_INTERVAL = 250; // ms
    let lastStatsTime = 0;
    let lastStatsHtml = "";

    function draw(ts) {
        requestAnimationFrame(draw);
        if (document.hidden) return;
        if (ts - lastFrameTime < FRAME_INTERVAL) return;
        lastFrameTime = ts;

        const dpr = effectiveDpr;
        const W = canvas.width, H = canvas.height;
        const now = currentServerTime();

        ctx.fillStyle = "#0e1117";
        ctx.fillRect(0, 0, W, H);

        const padL = 50 * dpr, padR = 20 * dpr, padT = 20 * dpr, padB = 35 * dpr;
        const plotW = W - padL - padR, plotH = H - padT - padB;
        const winSec = cfg.windowSeconds;
        const yMax = cfg.A * 0.4;

        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const x = padL + plotW * (i / 5);
            ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
        }
        for (let i = 0; i <= 4; i++) {
            const y = padT + plotH * (i / 4);
            ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
        }

        ctx.fillStyle = "#cccccc";
        ctx.font = (12 * dpr) + "px sans-serif";
        ctx.fillText("-" + winSec.toFixed(0) + "s", padL, H - 8 * dpr);
        ctx.fillText("now", padL + plotW - 20 * dpr, H - 8 * dpr);

        const wantStats = (ts - lastStatsTime >= STATS_INTERVAL);
        const spsHtml = [];

        latestUsers.forEach(function(u, idx) {
            // Splat each click's pulse only into the handful of nearby
            // samples where it's non-negligible (pulse decays to ~0 within
            // SUPPORT_SECONDS), instead of summing over all N samples for
            // every click. Cost is now ~clicks x support_samples, NOT
            // samples x clicks -- so raising N (resolution) is free.
            const buf = new Float32Array(N + 1);
            const dt = winSec / N;
            const supportSamples = Math.max(1, Math.ceil(SUPPORT_SECONDS / dt));

            for (const ct of u.clicks) {
                const age = now - ct;
                if (age < -0.05 || age > cfg.windowSeconds + SUPPORT_SECONDS) continue;
                const clickRel = -age;                       // click's position on the -winSec..0 axis
                const centerIdx = (clickRel + winSec) / dt;   // fractional sample index of the click
                const iStart = Math.max(0, Math.floor(centerIdx));
                const iEnd = Math.min(N, iStart + supportSamples);
                for (let i = iStart; i <= iEnd; i++) {
                    const tRel = -winSec + dt * i;
                    const v = pulse(tRel - clickRel, cfg.A, cfg.k);
                    if (v !== 0) buf[i] += v;
                }
            }

            ctx.beginPath();
            for (let i = 0; i <= N; i++) {
                const x = padL + plotW * (i / N);
                const y = padT + plotH * (1 - Math.min(buf[i], yMax) / yMax);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = u.color;
            ctx.lineWidth = 2.5 * dpr;
            ctx.stroke();

            const label = u.name + (u.id === cfg.userId ? " (you)" : "");
            ctx.fillStyle = u.color;
            ctx.fillRect(padL + 8 * dpr, padT + 8 * dpr + idx * 18 * dpr, 10 * dpr, 10 * dpr);
            ctx.fillStyle = "#eeeeee";
            ctx.font = (12 * dpr) + "px sans-serif";
            ctx.fillText(label, padL + 22 * dpr, padT + 17 * dpr + idx * 18 * dpr);

            if (wantStats) {
                const spm = strokesPerMinute(u.clicks, now);
                spsHtml.push(
                    '<div style="border:2px solid ' + u.color + ';border-radius:10px;' +
                    'padding:8px 18px;min-width:140px;text-align:center;background:rgba(255,255,255,0.03);">' +
                    '<div style="color:' + u.color + ';font-size:13px;font-weight:600;margin-bottom:2px;">' + label + '</div>' +
                    '<div style="color:#ffffff;font-size:34px;font-weight:800;line-height:1;">' + spm.toFixed(0) + '</div>' +
                    '<div style="color:#999;font-size:11px;">strokes / min</div>' +
                    '</div>'
                );
            }
        });

        if (wantStats) {
            lastStatsTime = ts;
            const html = spsHtml.join("");
            if (html !== lastStatsHtml) {
                spsRow.innerHTML = html;
                lastStatsHtml = html;
            }
        }
    }
    requestAnimationFrame(draw);

    document.addEventListener("visibilitychange", function() {
        if (!document.hidden) {
            lastFrameTime = 0;
            clearTimeout(pollTimer);
            pollState();
        }
    });
  </script>
</body>
</html>
"""


@app.get("/")
def join_page():
    return render_template_string(JOIN_PAGE)


@app.get("/room")
def room_page():
    room_id = request.args.get("room", "")
    user_id = request.args.get("user", "")
    user_name = request.args.get("name", "")
    if not room_id or not user_id:
        return ("", 302, {"Location": "/"})
    # Make sure the user actually exists in the room (e.g. after a server
    # restart); if not, bounce back to the join page.
    room = get_or_create_room(room_id)
    if user_id not in room.users:
        return ("", 302, {"Location": "/"})
    return render_template_string(
        ROOM_PAGE,
        room_id=room_id,
        user_id=user_id,
        user_name=user_name,
        window_seconds=WINDOW_SECONDS,
        pulse_a=PULSE_A,
        pulse_k=PULSE_K,
        sps_window=SPS_WINDOW,
        target_fps=TARGET_FPS,
        curve_samples=CURVE_SAMPLES,
        max_dpr=MAX_DPR,
        poll_ms_active=POLL_MS_ACTIVE,
        poll_ms_hidden=POLL_MS_HIDDEN,
    )


if __name__ == "__main__":
    print(f"Sync Clicker Trainer running at http://localhost:{PORT}")
    print("Other devices on your network can join at http://<your-LAN-ip>:" + str(PORT))
    # threaded=True lets multiple users' requests (state polling, clicks)
    # be handled concurrently instead of queueing behind each other.
    app.run(host="0.0.0.0", port=PORT, threaded=True)
