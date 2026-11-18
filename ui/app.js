// PathSnap evaluator UI (laptop 2). Talks only to the API; it never sees reference routes.
(function () {
  const $ = (id) => document.getElementById(id);
  const state = { api: "", partition: null, trip: null, layers: [], done: {} };

  try { const saved = localStorage.getItem("pathsnapApi"); if (saved) $("apiUrl").value = saved; } catch (e) { /* storage unavailable */ }

  const map = L.map("map", { preferCanvas: true }).setView([20, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);
  const legend = L.control({ position: "bottomright" });
  legend.onAdd = () => {
    const d = L.DomUtil.create("div", "legend");
    d.innerHTML = '<div><i style="background:#1f4fd1"></i>GPS observations</div>' +
                  '<div><i style="background:#d62728"></i>Predicted route (classifier + HMM)</div>';
    return d;
  };
  legend.addTo(map);

  function setStatus(text, cls) { const s = $("status"); s.textContent = text; s.className = "status " + cls; }
  function log(msg) {
    const li = document.createElement("li");
    li.textContent = new Date().toLocaleTimeString() + "  " + msg;
    $("log").prepend(li);
  }
  function pct(v) { return v === null || v === undefined ? "n/a" : (100 * v).toFixed(2) + " %"; }
  function num(v) { return v === null || v === undefined ? "n/a" : Number(v).toLocaleString(); }
  function frac(c, n) { return n ? " (" + num(c) + " / " + num(n) + ")" : ""; }
  function table(id, rows) {
    const body = rows.map((row) => row[0] === "head"
      ? '<tr class="head"><th></th><th>' + row[1] + "</th><th>" + row[2] + "</th></tr>"
      : "<tr><th>" + row[0] + '</th><td class="num">' + row[1] + '</td><td class="num">' + (row[2] || "") + "</td></tr>");
    $(id).innerHTML = "<tbody>" + body.join("") + "</tbody>";
  }

  async function call(path, options) {
    const res = await fetch(state.api + path, options);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
      throw new Error(res.status + " " + detail);
    }
    return res.json();
  }

  $("connect").onclick = async () => {
    state.api = $("apiUrl").value.trim().replace(/\/+$/, "");
    try { localStorage.setItem("pathsnapApi", state.api); } catch (e) { /* ignore */ }
    setStatus("connecting...", "busy");
    try {
      const h = await call("/health");
      const info = await call("/model-info");
      renderModel(info);
      const sel = $("partition");
      sel.innerHTML = h.partitions.map((p) => '<option value="' + p + '">' + p + "</option>").join("");
      sel.disabled = h.partitions.length === 0;
      setStatus("connected - " + h.model, "ok");
      log("connected; test sets: " + h.partitions.join(", "));
      if (h.partitions.length) await loadTrips(h.partitions[0]);
    } catch (e) {
      setStatus("connection failed", "err");
      log("connect failed: " + e.message);
    }
  };

  $("partition").onchange = (e) => loadTrips(e.target.value);

  async function loadTrips(partition) {
    state.partition = partition;
    state.trip = null;
    $("run").disabled = true;
    $("selected").textContent = "No trip selected.";
    try {
      const t = await call("/trips?partition=" + encodeURIComponent(partition));
      log("GET /trips " + partition + " -> " + t.trips.length + " trips");
      renderTrips(t.trips);
      renderOverall(state.done[partition] || null);
    } catch (e) {
      log("loading trips failed: " + e.message);
    }
  }

  function renderTrips(ids) {
    const box = $("trips");
    box.innerHTML = "";
    ids.forEach((id) => {
      const b = document.createElement("button");
      b.className = "trip";
      b.textContent = id;
      b.onclick = () => {
        state.trip = id;
        [...box.children].forEach((c) => c.classList.toggle("active", c === b));
        $("selected").textContent = "Selected " + state.partition + " / " + id;
        $("run").disabled = false;
      };
      box.appendChild(b);
    });
  }

  $("run").onclick = async () => {
    if (!state.trip) return;
    $("run").disabled = true;
    setStatus("predicting " + state.trip + "...", "busy");
    log("POST /predict " + state.partition + " / " + state.trip);
    const t0 = performance.now();
    try {
      const r = await call("/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trip_id: state.trip, partition: state.partition })
      });
      log("response for " + r.trip_id + " in " + ((performance.now() - t0) / 1000).toFixed(1) + " s");
      draw(r);
      renderMetrics(r);
      state.done[r.partition] = r.overall;
      renderOverall(r.overall);
      setStatus("done - " + r.trip_id, "ok");
    } catch (e) {
      setStatus("prediction failed", "err");
      log("predict failed: " + e.message);
    } finally {
      $("run").disabled = false;
    }
  };

  function draw(r) {
    state.layers.forEach((l) => map.removeLayer(l));
    state.layers = [];
    const gps = L.layerGroup(r.gps.map((p) => L.circleMarker(p, {
      radius: 2.5, color: "#1f4fd1", weight: 0, fillOpacity: 0.75
    })));
    const route = L.layerGroup(r.predicted_route.map((run) => L.polyline(run, {
      color: "#d62728", weight: 4, opacity: 0.9
    })));
    gps.addTo(map); route.addTo(map);
    state.layers.push(gps, route);
    if (r.gps.length) map.fitBounds(L.latLngBounds(r.gps), { padding: [30, 30] });
  }

  function renderModel(info) {
    const rows = [["Classifier", (info.model || "n/a") + (info.family ? " (" + info.family + ")" : ""), ""],
                  ["Decoder", "HMM / Viterbi, beta " + info.decoder.beta_m + " m, w " + info.decoder.emission_weight, ""]];
    const d = info.development_accuracy;
    if (d) {
      rows.push(["head", "Classifier only", "Classifier + HMM"]);
      rows.push(["Student-test top-1", pct(d.classifier_only.top1), pct(d.classifier_hmm.top1)]);
      rows.push(["Student-test continuity", pct(d.classifier_only.continuity), pct(d.classifier_hmm.continuity)]);
    }
    table("model", rows);
  }

  function renderMetrics(r) {
    const m = r.metrics;
    const rows = [["Trip", r.partition + " / " + r.trip_id, ""], ["GPS observations", num(r.n_observations), ""]];
    if (m.available) {
      const n = m.n_valid_labels;
      rows.push(["Points with a reference label", num(n), ""]);
      rows.push(["head", "Classifier only", "Classifier + HMM"]);
      rows.push(["Top-1 accuracy", pct(m.top1_classifier) + frac(m.n_correct_classifier, n),
                 pct(m.top1_hmm) + frac(m.n_correct_hmm, n)]);
      rows.push(["Route continuity (strict)", pct(m.continuity_classifier), pct(m.continuity_hmm)]);
      rows.push(["Route continuity (3 hops)", pct(m.continuity_3hop_classifier), pct(m.continuity_3hop_hmm)]);
      rows.push(["Discontinuities", num(m.discontinuities_classifier), num(m.discontinuities_hmm)]);
      rows.push(["Arc-set F1", pct(m.arc_f1_classifier), pct(m.arc_f1_hmm)]);
    } else {
      rows.push(["Reference route", "not available on server", ""]);
    }
    rows.push(["HMM chain breaks", num(r.hmm_breaks), ""]);
    rows.push(["Server time", r.timings_s.request_total + " s", ""]);
    table("metrics", rows);
  }

  function renderOverall(o) {
    if (!o || !o.trips_run) { $("overall").innerHTML = '<tbody><tr><td class="hint">No trips run yet on this test set.</td></tr></tbody>'; return; }
    table("overall", [
      ["Test set", o.partition, ""],
      ["Trips run", num(o.trips_run), ""],
      ["Points scored", num(o.n_valid), ""],
      ["head", "Classifier only", "Classifier + HMM"],
      ["Top-1 accuracy", pct(o.top1_classifier) + frac(o.n_correct_classifier, o.n_valid),
       pct(o.top1_hmm) + frac(o.n_correct_hmm, o.n_valid)],
    ]);
  }
})();
