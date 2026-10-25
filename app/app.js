/* ============================================================
   Vietnam Travel Recommender — Frontend JS
   Gọi FastAPI tại /api/v1/...
============================================================ */

const API = "";  // Cùng origin, không cần prefix

// ─── State ───────────────────────────────────────────────────
let destinations = [];          // [{destination_id, name, location, category, rating}]
let userRatings  = {};          // {destination_id: rating} cho Tab 2

// ─── Khởi tạo ────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  await Promise.all([checkHealth(), loadDestinations()]);
});

// ─── Tabs ─────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    });
  });
}

// ─── Health Check ─────────────────────────────────────────────
async function checkHealth() {
  const badge = document.getElementById("status-badge");
  try {
    const res  = await fetch(`${API}/health`);
    const data = await res.json();
    const knn  = data.components?.knn_recommender === "ready";
    const rag  = data.components?.rag_retriever   === "ready";
    if (knn && rag) {
      badge.textContent = "✓ Hệ thống sẵn sàng";
      badge.className   = "badge badge-ok";
    } else {
      badge.textContent = "⚠ Một số thành phần chưa sẵn sàng";
      badge.className   = "badge badge-err";
    }
  } catch {
    badge.textContent = "✗ Không kết nối được API";
    badge.className   = "badge badge-err";
  }
}

// ─── Load danh sách địa điểm ──────────────────────────────────
async function loadDestinations() {
  try {
    const res  = await fetch(`${API}/api/v1/recommend/destinations`);
    const data = await res.json();
    destinations = data.destinations || [];
    populateSelects();
  } catch (e) {
    showError("similar-result", "Không thể tải danh sách địa điểm. Hãy chắc chắn API server đang chạy.");
    showError("user-result",    "Không thể tải danh sách địa điểm.");
  }
}

function populateSelects() {
  const selSimilar = document.getElementById("dest-select");
  const selUser    = document.getElementById("user-dest-select");
  const opts = destinations.map(d =>
    `<option value="${d.destination_id}">${d.name} — ${d.location}</option>`
  ).join("");

  selSimilar.innerHTML = `<option value="">— Chọn địa điểm —</option>` + opts;
  selUser.innerHTML    = `<option value="">— Chọn địa điểm —</option>` + opts;
}

// ─── TAB 1: Gợi ý tương tự ────────────────────────────────────
async function getSimilar() {
  const destId = document.getElementById("dest-select").value;
  const k      = parseInt(document.getElementById("similar-k").value) || 5;
  const result = document.getElementById("similar-result");

  if (!destId) {
    showError("similar-result", "Vui lòng chọn một địa điểm.");
    return;
  }

  const dest = destinations.find(d => d.destination_id === destId);
  showSkeleton("similar-result", k);
  setLoading("btn-similar", true);

  try {
    const res  = await fetch(`${API}/api/v1/recommend/similar`, {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({destination_id: destId, k}),
    });
    const data = await res.json();

    if (!res.ok) {
      showError("similar-result", data.detail || "Đã xảy ra lỗi.");
      return;
    }

    const html = `
      <div class="result-header">
        <span>Gợi ý <strong>${data.total}</strong> địa điểm tương tự với</span>
        <strong>${dest?.name || destId}</strong>
      </div>
      <div class="result-grid">
        ${data.results.map(r => cardHTML(r)).join("")}
      </div>`;
    result.innerHTML = html;
    result.classList.remove("hidden");
  } catch {
    showError("similar-result", "Lỗi kết nối tới API server.");
  } finally {
    setLoading("btn-similar", false);
  }
}

// ─── TAB 2: Gợi ý cá nhân ─────────────────────────────────────
function addRating() {
  const sel    = document.getElementById("user-dest-select");
  const destId = sel.value;
  const rating = parseFloat(document.getElementById("user-rating").value) || 4.0;

  if (!destId) { alert("Vui lòng chọn địa điểm."); return; }
  if (userRatings[destId] !== undefined) { alert("Bạn đã thêm địa điểm này rồi."); return; }

  userRatings[destId] = rating;
  sel.value = "";
  renderRatingsList();
}

function removeRating(destId) {
  delete userRatings[destId];
  renderRatingsList();
}

function clearRatings() {
  userRatings = {};
  renderRatingsList();
}

function renderRatingsList() {
  const list  = document.getElementById("ratings-list");
  const items = document.getElementById("ratings-items");
  const ids   = Object.keys(userRatings);

  if (ids.length === 0) {
    list.classList.add("hidden");
    return;
  }
  list.classList.remove("hidden");

  items.innerHTML = ids.map(id => {
    const name = destinations.find(d => d.destination_id === id)?.name || id;
    const stars = "★".repeat(Math.round(userRatings[id])) + "☆".repeat(5 - Math.round(userRatings[id]));
    return `
      <div class="rating-item">
        <span class="rating-item-name">${name}</span>
        <span class="rating-item-score">${userRatings[id].toFixed(1)} ${stars}</span>
        <button class="rating-item-del" onclick="removeRating('${id}')" title="Xóa">✕</button>
      </div>`;
  }).join("");
}

async function getUserRecs() {
  const k      = parseInt(document.getElementById("user-k").value) || 5;
  const result = document.getElementById("user-result");

  if (Object.keys(userRatings).length === 0) {
    showError("user-result", "Vui lòng thêm ít nhất một địa điểm bạn đã ghé thăm.");
    return;
  }

  showSkeleton("user-result", k);
  setLoading("btn-user", true);

  try {
    const res  = await fetch(`${API}/api/v1/recommend/for-user`, {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({user_ratings: userRatings, k}),
    });
    const data = await res.json();

    if (!res.ok) {
      showError("user-result", data.detail || "Đã xảy ra lỗi.");
      return;
    }

    const html = `
      <div class="result-header">
        Gợi ý <strong>${data.total}</strong> địa điểm phù hợp với sở thích của bạn
      </div>
      <div class="result-grid">
        ${data.results.map(r => cardHTML(r)).join("")}
      </div>`;
    result.innerHTML = html;
    result.classList.remove("hidden");
  } catch {
    showError("user-result", "Lỗi kết nối tới API server.");
  } finally {
    setLoading("btn-user", false);
  }
}

// ─── TAB 3: Chat ──────────────────────────────────────────────
async function sendChat(event) {
  event.preventDefault();
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question) return;

  input.value = "";
  appendMessage("user", question);
  const typingEl = appendTyping();
  setLoading("btn-chat", true);

  try {
    const res  = await fetch(`${API}/api/v1/advise`, {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({question, top_k: 5}),
    });
    const data = await res.json();
    typingEl.remove();

    if (!res.ok) {
      appendMessage("bot", `⚠️ Lỗi: ${data.detail || "Đã xảy ra lỗi."}`);
      return;
    }

    appendMessage("bot", data.answer, data.sources || []);
  } catch {
    typingEl.remove();
    appendMessage("bot", "⚠️ Không thể kết nối tới API server. Hãy kiểm tra lại.");
  } finally {
    setLoading("btn-chat", false);
    input.focus();
  }
}

function fillChat(text) {
  document.getElementById("chat-input").value = text;
  document.getElementById("chat-input").focus();
}

function appendMessage(role, text, sources = []) {
  const container = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = `message message--${role}`;

  // Format markdown xuống dòng cơ bản
  const formatted = text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br/>");

  const cleanSrc = s => {
    let n = s.replace(".txt", "").replace(/_/g, " ");
    if (n.startsWith("a ")) n = "Đà " + n.slice(2);
    return n.charAt(0).toUpperCase() + n.slice(1);
  };
  const sourcesHTML = sources.length > 0
    ? `<div class="chat-sources"><span>📚 Cẩm nang:</span> ${sources.map(cleanSrc).join(" · ")}</div>`
    : "";


  div.innerHTML = `<div class="message-bubble">${formatted}${sourcesHTML}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function appendTyping() {
  const container = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = "message message--bot";
  div.innerHTML = `<div class="message-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

// ─── Helpers ──────────────────────────────────────────────────
function cardHTML(r) {
  const dest   = destinations.find(d => d.destination_id === r.destination_id) || {};
  const pct    = Math.round(r.similarity_score * 100);
  const stars  = dest.rating ? `⭐ ${parseFloat(dest.rating).toFixed(1)}` : "";
  return `
    <div class="dest-card">
      <div class="dest-card-rank">#${r.rank}</div>
      <div class="dest-card-name">${r.name}</div>
      <div class="dest-card-meta">${dest.location || ""} · ${dest.category || ""} ${stars}</div>
      <span class="dest-card-score">Tương đồng: ${pct}%</span>
      <div class="score-bar-wrap"><div class="score-bar" style="width:${pct}%"></div></div>
    </div>`;
}

function showSkeleton(containerId, count) {
  const el = document.getElementById(containerId);
  el.innerHTML = `<div class="skeleton-grid">${Array(Math.min(count, 6)).fill('<div class="skeleton"></div>').join("")}</div>`;
  el.classList.remove("hidden");
}

function showError(containerId, msg) {
  const el = document.getElementById(containerId);
  el.innerHTML = `<div class="alert alert-err">⚠️ ${msg}</div>`;
  el.classList.remove("hidden");
}

function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.disabled = loading;
  if (loading) {
    btn.dataset.originalText = btn.textContent;
    btn.textContent = "Đang xử lý…";
  } else {
    btn.textContent = btn.dataset.originalText || btn.textContent;
  }
}
