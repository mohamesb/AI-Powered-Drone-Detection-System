/* ============================================================
   STENDR-DEMO frontend
   Connects to /ws/swarm (5 Hz) and /ws/camera (~video fps),
   renders the tactical map and the camera feed with detections.
   ============================================================ */

const OSLO = [59.9139, 10.7522];

let map = null;
let droneLayer = null;          // Leaflet layer group of drone markers
const droneMarkers = {};        // id → Leaflet marker
const droneRoutes = {};         // id → Leaflet polyline of recent positions
const droneTrail = {};          // id → array of [lat, lon]
let selectedDroneId = null;
let lastFpsTs = 0;
let frameCount = 0;
let fps = 0;

// ---------- Map setup ----------
function initMap() {
  map = L.map("map", {
    center: OSLO,
    zoom: 13,
    zoomControl: false,
    attributionControl: true,
  });

  // CartoDB dark tiles — free, no API key needed
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; OpenStreetMap &copy; CartoDB',
    subdomains: "abcd",
    maxZoom: 19,
  }).addTo(map);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png", {
    subdomains: "abcd",
    maxZoom: 19,
    pane: "shadowPane",
  }).addTo(map);

  droneLayer = L.layerGroup().addTo(map);
}

// ---------- Drone markers ----------
function buildDroneIcon(droneId, isSelected) {
  const cls = "drone-marker" + (isSelected ? " selected" : "");
  return L.divIcon({
    className: "",
    html: `<div class="${cls}"><div class="drone-label">${droneId}</div></div>`,
    iconSize: [12, 12],
    iconAnchor: [6, 6],
  });
}

function updateDronePositions(snap) {
  const { drones, selected_drone_id, min_pair_distance_m } = snap;
  selectedDroneId = selected_drone_id;

  // Status bar
  document.getElementById("status-swarm").textContent = drones.length;
  document.getElementById("status-minsep").textContent =
      min_pair_distance_m + " m";
  document.getElementById("status-selected").textContent = selected_drone_id || "—";

  // Tint min-sep amber/red if drones are getting close
  const sepEl = document.getElementById("status-minsep");
  if (min_pair_distance_m < 80) sepEl.style.color = "var(--accent-bad)";
  else if (min_pair_distance_m < 150) sepEl.style.color = "var(--accent-warn)";
  else sepEl.style.color = "var(--text)";

  for (const d of drones) {
    const isSel = d.id === selected_drone_id;
    if (droneMarkers[d.id]) {
      droneMarkers[d.id].setLatLng([d.lat, d.lon]);
      droneMarkers[d.id].setIcon(buildDroneIcon(d.id, isSel));
    } else {
      const m = L.marker([d.lat, d.lon], {
        icon: buildDroneIcon(d.id, isSel),
      }).addTo(droneLayer);
      m.on("click", () => selectDrone(d.id));
      droneMarkers[d.id] = m;
    }

    // Track the recent flight path
    if (!droneTrail[d.id]) droneTrail[d.id] = [];
    droneTrail[d.id].push([d.lat, d.lon]);
    if (droneTrail[d.id].length > 50) droneTrail[d.id].shift();

    if (droneRoutes[d.id]) {
      droneRoutes[d.id].setLatLngs(droneTrail[d.id]);
      droneRoutes[d.id].setStyle({
        color: isSel ? "#ffb347" : "#7dffb6",
        opacity: isSel ? 0.6 : 0.25,
      });
    } else {
      droneRoutes[d.id] = L.polyline(droneTrail[d.id], {
        color: "#7dffb6",
        weight: 1,
        opacity: 0.25,
      }).addTo(droneLayer);
    }
  }
}

async function selectDrone(droneId) {
  await fetch(`/api/drone/${droneId}/select`, { method: "POST" });
  document.getElementById("feed-drone-id").textContent = droneId;
}

// ---------- Swarm WebSocket ----------
function connectSwarmWs() {
  const ws = new WebSocket(`ws://${location.host}/ws/swarm`);
  ws.onopen = () => {
    document.getElementById("status-link").classList.remove("disconnected");
  };
  ws.onclose = () => {
    document.getElementById("status-link").classList.add("disconnected");
    setTimeout(connectSwarmWs, 1500);
  };
  ws.onmessage = (e) => {
    const snap = JSON.parse(e.data);
    updateDronePositions(snap);
  };
}

// ---------- Camera WebSocket ----------
function connectCameraWs() {
  const ws = new WebSocket(`ws://${location.host}/ws/camera`);
  const img = document.getElementById("camera-img");
  const canvas = document.getElementById("detection-canvas");
  const emptyMsg = document.getElementById("camera-empty");
  const ctx = canvas.getContext("2d");

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === "error") {
      emptyMsg.style.display = "flex";
      return;
    }

    if (msg.type === "summary") {
      showSummary(msg.stats);
      return;
    }

    if (msg.type !== "frame") return;
    emptyMsg.style.display = "none";

    // Show the frame
    img.src = `data:image/jpeg;base64,${msg.jpeg_b64}`;
    document.getElementById("feed-drone-id").textContent =
        msg.selected_drone_id || "—";

    // Inference timing badge
    document.getElementById("status-infer").textContent =
        msg.result.inference_ms + " ms";

    // FPS counter (smoothed)
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTs > 1000) {
      fps = (frameCount / ((now - lastFpsTs) / 1000)).toFixed(1);
      document.getElementById("feed-fps").textContent = fps + " fps";
      frameCount = 0;
      lastFpsTs = now;
    }

    // Draw bounding boxes once the image actually loads & we know its size
    img.onload = () => {
      // Match canvas to the displayed image area, not raw frame size
      const rect = img.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      drawDetections(ctx, msg.result, canvas.width, canvas.height);
    };

    // Detection summary text
    const dets = msg.result.detections;
    if (dets.length) {
      document.getElementById("detection-summary").textContent =
          `${dets.length} detection${dets.length > 1 ? "s" : ""} · ` +
          dets.map(d => `${d.class.toUpperCase()} ${(d.confidence*100).toFixed(0)}%`).join("  ·  ");
    } else {
      document.getElementById("detection-summary").textContent = "no detections";
    }

    // Live stats panel
    updateStats(msg.live_stats);
  };

  ws.onclose = () => setTimeout(connectCameraWs, 1500);
}

// ---------- Detection drawing ----------
function drawDetections(ctx, result, w, h) {
  ctx.clearRect(0, 0, w, h);
  if (!result.detections.length) return;

  // The image is "contain"-fitted inside its box, so we need to compute the
  // actual letterboxed region the frame occupies.
  const fw = result.frame_w;
  const fh = result.frame_h;
  const scale = Math.min(w / fw, h / fh);
  const drawW = fw * scale;
  const drawH = fh * scale;
  const offX = (w - drawW) / 2;
  const offY = (h - drawH) / 2;

  for (const d of result.detections) {
    const [x1n, y1n, x2n, y2n] = d.bbox_norm;
    const x = offX + x1n * drawW;
    const y = offY + y1n * drawH;
    const bw = (x2n - x1n) * drawW;
    const bh = (y2n - y1n) * drawH;

    const isDrone = /drone/i.test(d.class);
    const colour = isDrone ? "#7dffb6" : "#ffb347";

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = colour;
    ctx.strokeRect(x, y, bw, bh);

    // Corner ticks for that targeting-system feel
    const tick = 8;
    ctx.beginPath();
    ctx.moveTo(x, y + tick); ctx.lineTo(x, y); ctx.lineTo(x + tick, y);
    ctx.moveTo(x + bw - tick, y); ctx.lineTo(x + bw, y); ctx.lineTo(x + bw, y + tick);
    ctx.moveTo(x, y + bh - tick); ctx.lineTo(x, y + bh); ctx.lineTo(x + tick, y + bh);
    ctx.moveTo(x + bw - tick, y + bh); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw, y + bh - tick);
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label
    const label = `${d.class.toUpperCase()} ${(d.confidence * 100).toFixed(0)}%`;
    ctx.font = "10px JetBrains Mono, monospace";
    const metrics = ctx.measureText(label);
    ctx.fillStyle = "rgba(10,13,15,0.85)";
    ctx.fillRect(x, y - 14, metrics.width + 8, 14);
    ctx.fillStyle = colour;
    ctx.fillText(label, x + 4, y - 4);
  }
}

// ---------- Stats panel ----------
function updateStats(s) {
  if (!s) return;
  setStat("stat-rate", (s.detection_rate * 100).toFixed(1) + "%", s.detection_rate > 0);
  setStat("stat-frames", s.frames_processed);
  setStat("stat-peak", s.peak_concurrent_drones, s.peak_concurrent_drones > 0);
  setStat("stat-conf", s.avg_confidence ? s.avg_confidence.toFixed(2) : "—",
          s.avg_confidence > 0);
  setStat("stat-fps", s.effective_fps);
  setStat("stat-p95", s.p95_inference_ms + " ms");
}

function setStat(id, value, accent = false) {
  const el = document.getElementById(id);
  el.textContent = value;
  el.classList.toggle("has-detection", accent);
}

// ---------- End-of-video summary ----------
function showSummary(stats) {
  const modal = document.getElementById("summary-modal");
  const body = document.getElementById("modal-body");
  const lines = [
    `Session duration     ${stats.session_seconds} s`,
    `Frames processed     ${stats.frames_processed}`,
    `Frames with drone    ${stats.frames_with_drone}`,
    `Detection rate       ${(stats.detection_rate * 100).toFixed(1)} %`,
    `Total detections     ${stats.total_detections}`,
    `Peak concurrent      ${stats.peak_concurrent_drones}`,
    `Avg confidence       ${stats.avg_confidence}`,
    `Avg inference        ${stats.avg_inference_ms} ms`,
    `p95 inference        ${stats.p95_inference_ms} ms`,
    `Effective FPS        ${stats.effective_fps}`,
    ``,
    `Class breakdown:`,
    ...Object.entries(stats.class_breakdown).map(([k, v]) =>
      `  ${k.padEnd(18)} ${v}`),
  ];
  body.textContent = lines.join("\n");
  modal.classList.remove("hidden");
}

document.getElementById("modal-close").addEventListener("click", () => {
  document.getElementById("summary-modal").classList.add("hidden");
});

// ---------- Boot ----------
window.addEventListener("DOMContentLoaded", () => {
  initMap();
  connectSwarmWs();
  connectCameraWs();
  lastFpsTs = performance.now();
});
