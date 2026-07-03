// ---------- Image source tabs (upload vs URL) ----------
const tabBtns = document.querySelectorAll(".tab-btn");
const panels = document.querySelectorAll(".image-input");

tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabBtns.forEach((b) => b.classList.remove("active"));
    panels.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.querySelector(`.image-input[data-panel="${btn.dataset.tab}"]`).classList.add("active");
  });
});

// ---------- Backend health check ----------
const statusEl = document.getElementById("backend-status");

async function checkBackend() {
  statusEl.textContent = "checking backend…";
  statusEl.className = "checking";
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { method: "GET" });
    if (res.ok) {
      statusEl.textContent = `backend online — ${BACKEND_URL}`;
      statusEl.className = "ok";
    } else {
      throw new Error("bad response");
    }
  } catch (e) {
    statusEl.textContent = `backend unreachable — ${BACKEND_URL}`;
    statusEl.className = "down";
  }
}
checkBackend();

// ---------- Form submit ----------
const form = document.getElementById("predict-form");
const errorEl = document.getElementById("form-error");
const submitBtn = document.getElementById("submit-btn");

const idleEl = document.getElementById("receipt-idle");
const loadingEl = document.getElementById("receipt-loading");
const resultEl = document.getElementById("receipt-result");

function showState(state) {
  idleEl.classList.add("hidden");
  loadingEl.classList.add("hidden");
  resultEl.classList.add("hidden");
  if (state === "idle") idleEl.classList.remove("hidden");
  if (state === "loading") loadingEl.classList.remove("hidden");
  if (state === "result") resultEl.classList.remove("hidden");
}

function formatMoney(n) {
  return `$${Number(n).toFixed(2)}`;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.textContent = "";

  const itemName = document.getElementById("item_name").value.trim();
  const imageFileInput = document.getElementById("image_file");
  const imageUrl = document.getElementById("image_url").value.trim();
  const activeTab = document.querySelector(".tab-btn.active").dataset.tab;

  if (!itemName) {
    errorEl.textContent = "Item name is required.";
    return;
  }
  if (activeTab === "upload" && (!imageFileInput.files || imageFileInput.files.length === 0)) {
    errorEl.textContent = "Please upload an image, or switch to Image URL.";
    return;
  }
  if (activeTab === "url" && !imageUrl) {
    errorEl.textContent = "Please paste an image URL, or switch to Upload file.";
    return;
  }

  const fd = new FormData();
  fd.append("item_name", itemName);
  fd.append("brand_name", document.getElementById("brand_name").value.trim());
  fd.append("bullet_points", document.getElementById("bullet_points").value);
  fd.append("product_description", document.getElementById("product_description").value);
  fd.append("unit", document.getElementById("unit").value.trim());

  const valueRaw = document.getElementById("value").value;
  if (valueRaw !== "") fd.append("value", valueRaw);

  if (activeTab === "upload") {
    fd.append("image", imageFileInput.files[0]);
  } else {
    fd.append("image_url", imageUrl);
  }

  submitBtn.disabled = true;
  showState("loading");

  try {
    const res = await fetch(`${BACKEND_URL}/predict_fields`, {
      method: "POST",
      body: fd,
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    document.getElementById("r-item-name").textContent = (data.parsed.item_name || itemName).toUpperCase();
    document.getElementById("r-brand-name").textContent = (data.parsed.brand_name || "").toUpperCase();
    document.getElementById("r-price").textContent = formatMoney(data.predicted_price);
    document.getElementById("r-lgb").textContent = formatMoney(data.lgb_price);
    document.getElementById("r-cat").textContent = formatMoney(data.catboost_price);

    showState("result");
  } catch (err) {
    errorEl.textContent = err.message || "Something went wrong. Is the backend running?";
    showState("idle");
  } finally {
    submitBtn.disabled = false;
  }
});
