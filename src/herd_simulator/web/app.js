"use strict";

var BEHAVIOURS = {
  0: "Resting", 1: "Grazing", 2: "Ruminating",
  3: "Walking", 4: "Restless", 5: "Unknown"
};

var GEO_LABELS = {0: "Inside", 1: "Warning", 2: "Breach"};

var COLORS = {
  green:  "#22c55e",
  yellow: "#eab308",
  red:    "#ef4444",
  grey:   "#6b7280"
};

var map, polygonLayer, markers = {}, pollTimer;
var pollInterval = 2000;
var connected = true;

// ---- Initialisation ----------------------------------------------------

function init() {
  map = L.map("map", {zoomControl: true});
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap"
  }).addTo(map);
  map.setView([12.9712, 79.1595], 17);

  var toast = document.createElement("div");
  toast.id = "toast";
  document.body.appendChild(toast);

  poll();
}

// ---- Polling -----------------------------------------------------------

function poll() {
  clearTimeout(pollTimer);

  Promise.all([
    fetchJSON("/api/state"),
    fetchJSON("/api/queue")
  ]).then(function(results) {
    setConnected(true);
    var state = results[0];
    var queue = results[1];

    updateSimInfo(state);
    updateEnv(state);
    updatePolygon(state.pasture_polygon);
    updateMarkers(state.animals || []);
    updateTable(state.animals || []);
    updateEvents(state.active_events || {});
    updateQueue(queue);
  }).catch(function() {
    setConnected(false);
  }).finally(function() {
    pollTimer = setTimeout(poll, pollInterval);
  });
}

function fetchJSON(url) {
  return fetch(url).then(function(r) {
    if (!r.ok) throw new Error(r.status);
    return r.json();
  });
}

function setConnected(ok) {
  connected = ok;
  var dot = document.getElementById("connection-status");
  dot.className = "status-dot " + (ok ? "connected" : "disconnected");
  dot.title = ok ? "Connected" : "Disconnected";
}

// ---- Sim info ----------------------------------------------------------

function updateSimInfo(state) {
  document.getElementById("sim-mode").textContent = state.sim_mode || "--";
  document.getElementById("sim-second").textContent = "t=" + state.sim_second;
  if (state.uptime_seconds !== undefined) {
    document.getElementById("uptime").textContent = "up " + formatDuration(state.uptime_seconds);
  }
}

function formatDuration(s) {
  s = Math.round(s);
  if (s < 60) return s + "s";
  if (s < 3600) return Math.floor(s / 60) + "m " + (s % 60) + "s";
  var h = Math.floor(s / 3600);
  var m = Math.floor((s % 3600) / 60);
  return h + "h " + m + "m";
}

// ---- Environment -------------------------------------------------------

function updateEnv(state) {
  setText("env-temp", val(state.ambient_temp_c, 1) + " °C");
  setText("env-hum", val(state.humidity_pct, 1) + " %");
  setText("env-thi", val(state.thi, 1));
}

// ---- Map: polygon ------------------------------------------------------

function updatePolygon(polygon) {
  if (!polygon || !polygon.length) return;
  if (polygonLayer) return;

  var latlngs = polygon.map(function(p) { return [p[0], p[1]]; });
  polygonLayer = L.polygon(latlngs, {
    color: "#22c55e", weight: 2, fillOpacity: 0.08, dashArray: "6 4"
  }).addTo(map);
  map.fitBounds(polygonLayer.getBounds().pad(0.3));
}

// ---- Map: markers ------------------------------------------------------

function updateMarkers(animals) {
  animals.forEach(function(a) {
    var id = a.animal_id;
    var color = a.dropped_out ? COLORS.grey : (COLORS[a.alert_band] || COLORS.green);
    var latlng = [a.latitude, a.longitude];

    if (markers[id]) {
      markers[id].setLatLng(latlng);
      markers[id].setStyle({color: color, fillColor: color});
    } else {
      markers[id] = L.circleMarker(latlng, {
        radius: 7, weight: 2, color: color,
        fillColor: color, fillOpacity: 0.8
      }).addTo(map);
    }

    var tip = "ID " + id +
      " | " + BEHAVIOURS[a.behaviour] +
      " | " + val(a.body_temp_c, 1) + "°C" +
      " | Risk " + a.risk_score +
      " | Batt " + val(a.battery_pct, 0) + "%";
    if (a.event_codes && a.event_codes.length) {
      tip += " | " + a.event_codes.join(", ");
    }
    markers[id].bindTooltip(tip);
  });
}

// ---- Table -------------------------------------------------------------

function updateTable(animals) {
  var tbody = document.getElementById("animal-tbody");
  var sorted = animals.slice().sort(function(a, b) { return a.animal_id - b.animal_id; });

  var rows = sorted.map(function(a) {
    var band = a.dropped_out ? "grey" : a.alert_band;
    return '<tr class="band-' + band + '">' +
      "<td>" + a.animal_id + (a.is_physical ? " ⭐" : "") + "</td>" +
      "<td>" + val(a.body_temp_c, 1) + "</td>" +
      "<td>" + val(a.thi, 1) + "</td>" +
      "<td>" + (BEHAVIOURS[a.behaviour] || "?") + "</td>" +
      "<td>" + a.risk_score + "</td>" +
      "<td>" + (GEO_LABELS[a.geofence_status] || "?") + "</td>" +
      "<td>" + val(a.battery_pct, 0) + "%</td>" +
      "<td>" + (a.event_codes.length ? a.event_codes.join(", ") : "—") + "</td>" +
      "</tr>";
  });

  tbody.innerHTML = rows.join("");
}

// ---- Events ------------------------------------------------------------

function updateEvents(evts) {
  var container = document.getElementById("events-list");
  var keys = Object.keys(evts);
  if (!keys.length) {
    container.innerHTML = "<em>None</em>";
    return;
  }

  container.innerHTML = keys.map(function(eid) {
    var e = evts[eid];
    return '<div class="event-item">' +
      '<span class="evt-info">' + eid + " | ID " + e.animal_id +
      " | " + e.event_type + " (" + e.elapsed_s + "s)</span>" +
      '<button onclick="clearEvent(\'' + eid + '\')">Clear</button>' +
      "</div>";
  }).join("");
}

// ---- Queue -------------------------------------------------------------

function updateQueue(q) {
  setText("q-priority", (q.priority_queue && q.priority_queue.length)
    ? q.priority_queue.join(", ") : "—");
  setText("q-rr", (q.rr_next_5 && q.rr_next_5.length)
    ? q.rr_next_5.join(", ") : "—");
  setText("q-writes", q.total_writes || 0);
  setText("q-sweeps", q.sweeps_completed || 0);
}

// ---- Event injection ---------------------------------------------------

function injectEvent() {
  var type = document.getElementById("inject-type").value;
  var animalId = parseInt(document.getElementById("inject-id").value, 10);
  if (isNaN(animalId) || animalId < 1 || animalId > 20) {
    showToast("Invalid animal ID");
    return;
  }

  fetch("/api/events", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({animal_id: animalId, type: type})
  }).then(function(r) { return r.json().then(function(d) { return {ok: r.ok, data: d}; }); })
    .then(function(res) {
      if (res.ok) {
        showToast("Injected " + type + " on ID " + animalId + " (" + res.data.event_id + ")");
      } else {
        showToast("Error: " + (res.data.message || "unknown"));
      }
    })
    .catch(function() { showToast("Request failed"); });
}

function clearEvent(eventId) {
  fetch("/api/events/" + eventId, {method: "DELETE"})
    .then(function(r) {
      if (r.status === 204) {
        showToast("Cleared " + eventId);
      } else {
        return r.json().then(function(d) { showToast("Error: " + d.message); });
      }
    })
    .catch(function() { showToast("Request failed"); });
}

// ---- Helpers -----------------------------------------------------------

function val(v, decimals) {
  if (v === null || v === undefined) return "--";
  return Number(v).toFixed(decimals);
}

function setText(id, text) {
  var el = document.getElementById(id);
  if (el) el.textContent = text;
}

function showToast(msg) {
  var t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(function() { t.classList.remove("show"); }, 2500);
}

// ---- Start -------------------------------------------------------------

document.addEventListener("DOMContentLoaded", init);
