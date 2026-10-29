// LungLens in the browser: ONNX Runtime Web runs the trained model on this device.
// The X-ray is drawn to a 384x384 canvas, turned to grayscale in [0, 1] (the same preprocessing as
// predict.py), and the model returns the three class probabilities plus a 12x12 heatmap per class.

const SIZE = 384;
const CLASSES = ["healthy", "sick_but_not_tb", "tb"];
const NAMES = { healthy: "Healthy", sick_but_not_tb: "Sick, but not TB", tb: "TB" };
let session = null, card = null, samples = [], current = null, lastImg = null, lastMeta = null;
const sessions = {}; // "main" = deployed model, "alt" = the other DenseNet (loaded on demand)
let which = "main";

const $ = (s) => document.querySelector(s);
const base = $("#xray"), heat = $("#heat"), boxes = $("#boxes");

async function init() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  try {
    [card, samples] = await Promise.all([
      fetch("model_card.json").then((r) => r.json()),
      fetch("samples/samples.json").then((r) => r.json()),
    ]);
    buildGallery();
    setStatus("Loading the model (28 MB, once)…");
    session = sessions.main = await ort.InferenceSession.create("model/lunglens.onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    setupModelSwitch();
    setStatus("Model ready. Upload an X-ray or tap a sample.");
    $("#run-state").textContent = "ready";
  } catch (e) {
    console.error(e);
    setStatus("Could not load the model in this browser: " + e.message);
  }
}

function setStatus(t) { $("#status").textContent = t; }

function buildGallery() {
  const g = $("#gallery");
  samples.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.title = `Sample ${i + 1}: true label ${NAMES[s.class]}`;
    b.innerHTML = `<img src="samples/${s.file}" alt="Sample chest X-ray, true label ${NAMES[s.class]}" loading="lazy"><span>${NAMES[s.class]}</span>`;
    b.addEventListener("click", () => {
      document.querySelectorAll("#gallery button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      loadURL(`samples/${s.file}`, s);
    });
    g.appendChild(b);
  });
}

function loadURL(url, meta) {
  const img = new Image();
  img.onload = () => analyse(img, meta || null);
  img.src = url;
}

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return setStatus("Please choose an image file (PNG or JPG).");
  const url = URL.createObjectURL(file);
  document.querySelectorAll("#gallery button").forEach((x) => x.classList.remove("active"));
  loadURL(url, null);
}

function toTensor(img) {
  const c = document.createElement("canvas");
  c.width = c.height = SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, SIZE, SIZE);
  const d = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const x = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) x[i] = (0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]) / 255;
  return new ort.Tensor("float32", x, [1, 1, SIZE, SIZE]);
}

function threshold() {
  return which === "alt" ? card.alternative_model.threshold : card.screening.threshold;
}

function setupModelSwitch() {
  const alt = card.alternative_model;
  const box = $("#model-switch");
  if (!alt || !box) return;
  box.hidden = false;
  $("#m-main").textContent = card.architecture + " (deployed)";
  $("#m-alt").textContent = alt.architecture;
  box.querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
    which = b.dataset.which;
    box.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    if (!sessions[which]) {
      setStatus("Loading the second model (28 MB, once)…");
      sessions[which] = await ort.InferenceSession.create("model/lunglens_alt.onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    }
    session = sessions[which];
    if (lastImg) analyse(lastImg, lastMeta); else setStatus("Model switched. Upload an X-ray or tap a sample.");
  }));
}

async function analyse(img, meta) {
  if (!session) return setStatus("The model is still loading…");
  lastImg = img; lastMeta = meta;
  $("#empty").hidden = true;
  const W = 768;
  [base, heat, boxes].forEach((c) => { c.width = c.height = W; });
  base.getContext("2d").drawImage(img, 0, 0, W, W);
  setStatus("Analysing…");
  const t0 = performance.now();
  const out = await session.run({ xray: toTensor(img) });
  const ms = performance.now() - t0;
  const p = Array.from(out.probs.data);
  const n = out.cam.dims[2] * out.cam.dims[3];
  const camTB = Array.from(out.cam.data.slice(2 * n, 3 * n));
  current = { p, camTB, side: out.cam.dims[3], meta };
  drawHeat();
  drawBoxes();
  showResult(p, ms, meta);
  setStatus(`Done in ${Math.round(ms)} ms on this device. Nothing was uploaded.`);
}

function jet(v) {
  const c = (x) => Math.max(0, Math.min(1, x));
  return [c(1.5 - Math.abs(4 * v - 3)), c(1.5 - Math.abs(4 * v - 2)), c(1.5 - Math.abs(4 * v - 1))].map((x) => Math.round(x * 255));
}

function drawHeat() {
  const ctx = heat.getContext("2d");
  ctx.clearRect(0, 0, heat.width, heat.height);
  if (!current || !$("#show-heat").checked) return;
  const { camTB, side } = current;
  const lo = Math.min(...camTB), hi = Math.max(...camTB);
  const small = document.createElement("canvas");
  small.width = small.height = side;
  const sctx = small.getContext("2d");
  const im = sctx.createImageData(side, side);
  camTB.forEach((v, i) => {
    const t = (v - lo) / (hi - lo + 1e-8);
    const [r, g, b] = jet(t);
    im.data.set([r, g, b, 255], 4 * i);
  });
  sctx.putImageData(im, 0, 0);
  ctx.globalAlpha = Number($("#opacity").value);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(small, 0, 0, heat.width, heat.height);
}

function drawBoxes() {
  const ctx = boxes.getContext("2d");
  ctx.clearRect(0, 0, boxes.width, boxes.height);
  const m = current && current.meta;
  const has = m && m.boxes && m.boxes.length;
  $("#box-toggle").hidden = !has;
  if (!has || !$("#show-boxes").checked) return;
  const k = boxes.width / 512;
  ctx.strokeStyle = "#39ff6a";
  ctx.lineWidth = 4;
  m.boxes.forEach(([x1, y1, x2, y2]) => ctx.strokeRect(x1 * k, y1 * k, (x2 - x1) * k, (y2 - y1) * k));
}

function showResult(p, ms, meta) {
  const thr = threshold();
  const tb = p[2], top = CLASSES[p.indexOf(Math.max(...p))];
  const v = $("#verdict");
  if (tb >= thr) {
    v.className = "verdict tb";
    v.innerHTML = `Refer for a TB lab test<small>TB probability ${(tb * 100).toFixed(1)}%, above the screening cut-off of ${(thr * 100).toFixed(1)}% (set so only 5% of people without TB are referred). Confirm with sputum / GeneXpert.</small>`;
  } else if (top === "sick_but_not_tb") {
    v.className = "verdict sick";
    v.innerHTML = `Signs of a lung problem, but not TB<small>Below the TB threshold. The X-ray still looks abnormal: see a doctor.</small>`;
  } else {
    v.className = "verdict ok";
    v.innerHTML = `No TB signs found<small>TB probability ${(tb * 100).toFixed(1)}%, below the screening cut-off of ${(thr * 100).toFixed(1)}%. The model is very sure when an X-ray is clear, so the cut-off is low.</small>`;
  }
  const bars = $("#bars");
  bars.innerHTML = "";
  CLASSES.forEach((c, i) => {
    const mark = c === "tb" ? `<div class="mark" style="left:${thr * 100}%" title="screening threshold"></div>` : "";
    bars.insertAdjacentHTML("beforeend",
      `<div class="bar ${c === "tb" ? "tb" : ""}"><div class="label"><span>${NAMES[c]}</span><b>${(p[i] * 100).toFixed(1)}%</b></div>
       <div class="track"><div class="fill" style="width:${p[i] * 100}%"></div>${mark}</div></div>`);
  });
  $("#which").textContent = which === "alt" ? card.alternative_model.architecture : card.architecture + " (deployed)";
  $("#truth").textContent = meta ? `True label (from TBX11K): ${NAMES[meta.class]}${meta.boxes && meta.boxes.length ? ` · ${meta.boxes.length} TB box(es) drawn by doctors` : ""}.` : "Your own image: no true label.";
}

$("#file").addEventListener("change", (e) => loadFile(e.target.files[0]));
const drop = $("#drop");
drop.addEventListener("click", () => $("#file").click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file").click(); } });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
$("#show-heat").addEventListener("change", drawHeat);
$("#opacity").addEventListener("input", drawHeat);
$("#show-boxes").addEventListener("change", drawBoxes);
$("#try-sample").addEventListener("click", () => {
  const tb = [...document.querySelectorAll("#gallery button")].find((b, i) => samples[i].class === "tb");
  document.querySelector("#demo").scrollIntoView();
  (tb || document.querySelector("#gallery button")).click();
});
init();
