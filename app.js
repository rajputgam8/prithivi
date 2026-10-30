"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  token: "",
  busy: false,
  assistant: null,
  confirm: null,
};

function readTokenFromHash() {
  const raw = location.hash.startsWith("#") ? location.hash.slice(1) : "";
  const params = new URLSearchParams(raw);
  const token = params.get("token") || "";
  if (token) {
    sessionStorage.setItem("prithivi.token", token);
    history.replaceState(null, "", location.pathname + location.search);
  }
  return token || sessionStorage.getItem("prithivi.token") || "";
}

function lock() {
  sessionStorage.removeItem("prithivi.token");
  state.token = "";
  $("#app").hidden = true;
  $("#auth").hidden = false;
  $("#token-input").value = "";
  $("#token-input").focus();
}

function toast(message, ms = 2600) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { el.hidden = true; }, ms);
}

async function api(path, options = {}) {
  const headers = Object.assign({}, options.headers || {});
  headers.Authorization = `Bearer ${state.token}`;
  if (options.body !== undefined && typeof options.body !== "string") {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, Object.assign({}, options, { headers }));
  if (response.status === 401) {
    lock();
    throw new Error("token rejected");
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.error) detail = body.error;
    } catch (_) { /* not JSON */ }
    throw new Error(detail);
  }
  return response;
}

async function jsonApi(path, options) {
  const response = await api(path, options);
  return response.json();
}

// ---------------------------------------------------------------- auth
$("#auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const token = $("#token-input").value.trim();
  if (!token) return;
  state.token = token;
  try {
    await jsonApi("/api/status");
    sessionStorage.setItem("prithivi.token", token);
    $("#auth").hidden = true;
    $("#app").hidden = false;
    await boot();
  } catch (error) {
    $("#auth-error").textContent = error.message;
  }
});

$("#lock-button").addEventListener("click", lock);

// ---------------------------------------------------------------- tabs
const titles = { chat: "Chat", memory: "Memory", knowledge: "Knowledge", status: "Status" };
$$(".nav-tab").forEach((button) => {
  button.addEventListener("click", () => {
    $$(".nav-tab").forEach((b) => b.classList.toggle("active", b === button));
    $$(".tab-section").forEach((panel) => { panel.classList.toggle("active", panel.id === `tab-${button.dataset.tab}`); });
    if (button.dataset.tab === "memory") loadMemory();
    if (button.dataset.tab === "knowledge") loadKnowledge();
    if (button.dataset.tab === "status") loadStatus();
    if (button.dataset.tab === "chat") $("#chat-input").focus();
  });
});

// ---------------------------------------------------------------- chat
const input = $("#chat-input");
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});

$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || state.busy) return;
  input.value = "";
  input.style.height = "auto";
  send(message);
});

$$(".quick-btn").forEach((button) => {
  button.addEventListener("click", () => {
    if (state.busy) return;
    send(button.textContent);
  });
});

// ---------------------------------------------------------------- voice
const GREETING = "Hello sir, how can I help you?";
const synth = "speechSynthesis" in window ? window.speechSynthesis : null;
const voice = {
  enabled: localStorage.getItem("prithivi.voice") !== "off",
  greeted: false,
  listening: false,
  transcript: "",
  error: false,
};

function pickVoice() {
  if (!synth) return null;
  const voices = synth.getVoices();
  if (!voices.length) return null;
  const preferred = [
    /Google UK English Female/i,
    /Google US English/i,
    /Microsoft (Aria|Jenny|Zira)/i,
    /Samantha/i,
  ];
  for (const pattern of preferred) {
    const hit = voices.find((item) => pattern.test(item.name));
    if (hit) return hit;
  }
  return voices.find((item) => item.lang === "en-US")
    || voices.find((item) => (item.lang || "").startsWith("en"))
    || voices[0];
}

function cleanForSpeech(text) {
  return String(text || "")
    .replace(/```[\s\S]*?```/g, " code block omitted. ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[*_#>~|]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function speak(text) {
  if (!synth || !voice.enabled) return;
  const clean = cleanForSpeech(text);
  if (!clean) return;
  synth.cancel();
  const utterance = new SpeechSynthesisUtterance(clean);
  utterance.lang = "en-US";
  utterance.rate = 1;
  utterance.pitch = 1;
  const chosen = pickVoice();
  if (chosen) utterance.voice = chosen;
  synth.speak(utterance);
}

function stopSpeaking() {
  if (synth) synth.cancel();
}

function renderVoiceToggle() {
  const button = $("#voice-toggle");
  if (!button) return;
  button.classList.toggle("muted", !voice.enabled);
  button.setAttribute("aria-pressed", String(voice.enabled));
  button.title = voice.enabled ? "Voice replies on" : "Voice replies off";
}

function greet() {
  if (voice.greeted || !voice.enabled) return;
  voice.greeted = true;
  speak(GREETING);
}

function startVoiceGreeting() {
  renderVoiceToggle();
  if (!synth) {
    const button = $("#voice-toggle");
    if (button) button.hidden = true;
    return;
  }
  synth.getVoices();
  const activated = !navigator.userActivation || navigator.userActivation.hasBeenActive;
  if (activated) greet();
  else document.addEventListener("pointerdown", greet, { once: true });
}

if (synth) {
  $("#voice-toggle").addEventListener("click", () => {
    voice.enabled = !voice.enabled;
    localStorage.setItem("prithivi.voice", voice.enabled ? "on" : "off");
    if (!voice.enabled) stopSpeaking();
    else greet();
    renderVoiceToggle();
    toast(voice.enabled ? "Voice replies on." : "Voice replies off.");
  });
}

const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
const micButton = $("#mic-button");
let recognition = null;

function setMicState(listening) {
  voice.listening = listening;
  micButton.classList.toggle("listening", listening);
  micButton.setAttribute("aria-label", listening ? "Stop voice input" : "Start voice input");
  input.placeholder = listening ? "Listening…" : "Type your message...";
}

if (!SpeechRecognitionCtor) {
  micButton.hidden = true;
  micButton.title = "Voice input is not supported by this browser";
} else {
  recognition = new SpeechRecognitionCtor();
  recognition.lang = "en-US";
  recognition.interimResults = true;
  recognition.continuous = false;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    voice.transcript = "";
    voice.error = false;
    stopSpeaking();
    setMicState(true);
  };

  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const piece = event.results[i][0].transcript;
      if (event.results[i].isFinal) voice.transcript += piece;
      else interim += piece;
    }
    input.value = `${voice.transcript}${interim}`.trim();
    input.dispatchEvent(new Event("input"));
  };

  recognition.onerror = (event) => {
    voice.error = true;
    const messages = {
      "not-allowed": "Microphone blocked. Allow mic access, or open http://127.0.0.1:8000/ on the device running Prithivi.",
      "service-not-allowed": "Speech recognition is blocked by this browser.",
      "no-speech": "I didn't hear anything.",
      "network": "Speech recognition needs a network connection.",
      "audio-capture": "No microphone was found.",
    };
    toast(messages[event.error] || `Voice input failed: ${event.error}`);
  };

  recognition.onend = () => {
    setMicState(false);
    const text = voice.transcript.trim() || input.value.trim();
    if (!voice.error && text && !state.busy) {
      input.value = "";
      input.style.height = "auto";
      send(text);
    }
  };

  micButton.addEventListener("click", () => {
    if (voice.listening) {
      recognition.stop();
      return;
    }
    try {
      recognition.start();
    } catch (_) {
      toast("Voice input is already starting.");
    }
  });
}

function clearEmptyState() {
  const empty = $(".empty-state", $("#messages"));
  if (empty) empty.remove();
}

function addMessage(role, text) {
  clearEmptyState();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  const meta = document.createElement("div");
  meta.className = "meta";
  wrap.append(bubble, meta);
  $("#messages").append(wrap);
  scrollDown();
  return { wrap, bubble, meta };
}

function scrollDown() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

function ensureAssistant() {
  if (!state.assistant) {
    state.assistant = addMessage("assistant", "");
    state.assistant.bubble.classList.add("streaming");
  }
  return state.assistant;
}

async function send(message) {
  state.busy = true;
  stopSpeaking();
  $("#send-button").disabled = true;
  state.assistant = null;
  addMessage("user", message);
  const pending = addMessage("assistant", "…");
  pending.bubble.classList.add("streaming");
  state.assistant = pending;

  try {
    const response = await api("/api/chat", { method: "POST", body: { message } });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        handleEvent(block);
      }
    }
  } catch (error) {
    if (state.assistant) {
      state.assistant.bubble.textContent = `Could not reach Prithivi: ${error.message}`;
      state.assistant.bubble.classList.remove("streaming");
    }
  } finally {
    state.busy = false;
    $("#send-button").disabled = false;
    if (state.assistant) state.assistant.bubble.classList.remove("streaming");
    state.assistant = null;
    refreshStatusLite();
  }
}

function handleEvent(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  let payload = {};
  if (dataLines.length) {
    try { payload = JSON.parse(dataLines.join("\n")); } catch (_) { payload = {}; }
  }
  if (event === "delta") {
    const node = ensureAssistant();
    if (node.bubble.textContent === "…") node.bubble.textContent = "";
    node.bubble.textContent += payload.text || "";
    scrollDown();
  } else if (event === "confirm") {
    showConfirm(payload);
  } else if (event === "final") {
    renderFinal(payload);
  } else if (event === "error") {
    const node = ensureAssistant();
    node.bubble.textContent = payload.message || "Something went wrong.";
    node.bubble.classList.remove("streaming");
  }
}

function renderFinal(result) {
  const node = ensureAssistant();
  node.bubble.textContent = result.text || "(no answer)";
  node.bubble.classList.remove("streaming");

  node.meta.textContent = "";
  for (const chip of result.provenance || []) {
    const span = document.createElement("span");
    span.className = "chip";
    span.textContent = chip;
    node.meta.append(span);
  }
  if (result.blocked) {
    const span = document.createElement("span");
    span.className = "chip";
    span.textContent = "needs permission";
    node.meta.append(span);
  }
  for (const tool of result.tools || []) {
    const line = document.createElement("div");
    line.className = "tool-line";
    const mark = document.createElement("span");
    mark.className = tool.success ? "ok" : "fail";
    mark.textContent = tool.success ? "ok" : "fail";
    line.append(mark, document.createTextNode(` ${tool.tool} — ${tool.summary}`));
    node.wrap.append(line);
  }
  const timing = document.createElement("div");
  timing.className = "meta";
  timing.textContent = `${result.iterations || 0} step(s) · ${result.elapsed_ms || 0} ms${result.error ? ` · ${result.error}` : ""}`;
  node.wrap.append(timing);
  if (!result.error && result.text) speak(result.text);
  scrollDown();
}

// ---------------------------------------------------------- confirmations
function showConfirm(request) {
  state.confirm = request;
  $("#confirm-title").textContent = `${request.level_name}: ${request.tool}`;
  $("#confirm-consequence").textContent = request.consequence || "Run this action?";
  $("#confirm-args").textContent = request.args_preview || "";
  const phraseWrap = $("#confirm-phrase-wrap");
  if (request.typed_phrase) {
    phraseWrap.hidden = false;
    $("#confirm-phrase").textContent = request.typed_phrase;
    $("#confirm-phrase-input").value = "";
    $("#confirm-phrase-input").focus();
  } else {
    phraseWrap.hidden = true;
  }
  $("#confirm").hidden = false;
}

async function answerConfirm(approve) {
  const request = state.confirm;
  if (!request) return;
  const phrase = $("#confirm-phrase-input").value.trim();
  if (approve && request.typed_phrase && phrase.toLowerCase() !== request.typed_phrase.toLowerCase()) {
    toast("The phrase does not match.");
    return;
  }
  $("#confirm").hidden = true;
  state.confirm = null;
  try {
    await jsonApi("/api/confirm", { method: "POST", body: { token: request.token, approve, phrase } });
  } catch (error) {
    toast(`Could not answer: ${error.message}`);
  }
}

$("#confirm-approve").addEventListener("click", () => answerConfirm(true));
$("#confirm-decline").addEventListener("click", () => answerConfirm(false));
$("#confirm-phrase-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); answerConfirm(true); }
});

// ---------------------------------------------------------------- memory
$("#memory-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("#memory-text").value.trim();
  if (!text) return;
  try {
    await jsonApi("/api/memory", {
      method: "POST",
      body: { text, pinned: $("#memory-pinned").checked },
    });
    $("#memory-text").value = "";
    toast("Remembered.");
    loadMemory();
  } catch (error) {
    toast(error.message);
  }
});

$("#memory-toggle").addEventListener("change", async (event) => {
  try {
    await jsonApi("/api/memory/toggle", { method: "POST", body: { enabled: event.target.checked } });
    toast(event.target.checked ? "Memory enabled." : "Memory disabled. Nothing is stored or recalled.");
    loadMemory();
  } catch (error) {
    toast(error.message);
    event.target.checked = !event.target.checked;
  }
});

async function loadMemory() {
  try {
    const data = await jsonApi("/api/memory");
    $("#memory-toggle").checked = !!data.enabled;
    const stats = data.stats || {};
    $("#memory-stats").textContent = data.enabled
      ? `${stats.count || 0} item(s) · ${Object.entries(stats.by_type || {}).map(([k, v]) => `${v} ${k}`).join(", ") || "empty"}`
      : "Memory is off. Nothing will be stored or recalled.";
    const list = $("#memory-list");
    list.textContent = "";
    for (const memory of data.memories || []) {
      const li = document.createElement("li");
      const grow = document.createElement("div");
      grow.className = "grow";
      const title = document.createElement("div");
      title.className = "title";
      title.textContent = memory.text;
      const sub = document.createElement("div");
      sub.className = "sub";
      sub.textContent = `${memory.type} · ${memory.key}${memory.pinned ? " · pinned" : ""}`;
      grow.append(title, sub);
      const del = document.createElement("button");
      del.className = "ghost";
      del.textContent = "Forget";
      del.addEventListener("click", async () => {
        try {
          await jsonApi(`/api/memory/${encodeURIComponent(memory.key)}`, { method: "DELETE" });
          loadMemory();
        } catch (error) { toast(error.message); }
      });
      li.append(grow, del);
      list.append(li);
    }
    if (!list.children.length) {
      const li = document.createElement("li");
      li.className = "muted";
      li.textContent = "Nothing saved yet.";
      list.append(li);
    }
  } catch (error) {
    toast(error.message);
  }
}

// ------------------------------------------------------------- knowledge
async function loadKnowledge() {
  try {
    const data = await jsonApi("/api/knowledge");
    const stats = data.stats || {};
    $("#knowledge-stats").textContent = `${stats.documents || 0} document(s), ${stats.chunks || 0} chunk(s)`;
    const list = $("#knowledge-list");
    list.textContent = "";
    for (const doc of data.documents || []) {
      const li = document.createElement("li");
      const grow = document.createElement("div");
      grow.className = "grow";
      const title = document.createElement("div");
      title.className = "title";
      title.textContent = doc.title || doc.uri;
      const sub = document.createElement("div");
      sub.className = "sub";
      sub.textContent = `${doc.chunk_count || 0} chunk(s) · ${doc.status} · ${doc.source_type} · ${doc.id.slice(0, 10)}`;
      grow.append(title, sub);
      const del = document.createElement("button");
      del.className = "ghost";
      del.textContent = "Delete";
      del.addEventListener("click", async () => {
        if (!confirm(`Delete ${doc.title || doc.uri} and its index?`)) return;
        try {
          await jsonApi(`/api/knowledge/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
          loadKnowledge();
        } catch (error) { toast(error.message); }
      });
      li.append(grow, del);
      list.append(li);
    }
    if (!list.children.length) {
      const li = document.createElement("li");
      li.className = "muted";
      li.textContent = "No documents yet.";
      list.append(li);
    }
  } catch (error) {
    toast(error.message);
  }
}

$("#ingest-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const path = $("#ingest-path").value.trim();
  const text = $("#ingest-text").value.trim();
  const title = $("#ingest-title").value.trim();
  if (!path && !text) { toast("Give a path or paste some text."); return; }
  $("#ingest-status").textContent = "Indexing…";
  try {
    const result = await jsonApi("/api/knowledge/ingest", {
      method: "POST",
      body: { path, text, title },
    });
    $("#ingest-status").textContent = `${result.duplicate ? "Already indexed" : "Indexed"}: ${result.chunk_count} chunk(s).`;
    $("#ingest-path").value = "";
    $("#ingest-text").value = "";
    $("#ingest-title").value = "";
    loadKnowledge();
  } catch (error) {
    $("#ingest-status").textContent = error.message;
  }
});

// ---------------------------------------------------------------- status
async function loadStatus() {
  try {
    const data = await jsonApi("/api/status");
    $("#session-label").textContent = data.session_id || "";
    const grid = $("#status-grid");
    grid.textContent = "";
    const stats = [
      ["State", data.state],
      ["Model", `${data.provider} · ${data.model}`, true],
      ["Turns", String(data.turns || 0)],
      ["Tools", String(data.tools || 0)],
      ["Memories", String((data.memory || {}).count || 0)],
      ["Documents", String((data.knowledge || {}).documents || 0)],
      ["Audit chain", data.audit_ok === null ? "n/a" : data.audit_ok ? "intact" : "BROKEN"],
    ];
    for (const [key, value, small] of stats) {
      const box = document.createElement("div");
      box.className = "stat";
      const k = document.createElement("div");
      k.className = "k";
      k.textContent = key;
      const v = document.createElement("div");
      v.className = `v${small ? " small" : ""}`;
      v.textContent = value;
      box.append(k, v);
      grid.append(box);
    }
  } catch (error) {
    toast(error.message);
  }

  try {
    const data = await jsonApi("/api/tools");
    const list = $("#tool-list");
    list.textContent = "";
    for (const tool of data.tools || []) {
      const li = document.createElement("li");
      const grow = document.createElement("div");
      grow.className = "grow";
      const title = document.createElement("div");
      title.className = "title";
      title.textContent = tool.name;
      const sub = document.createElement("div");
      sub.className = "sub";
      sub.textContent = tool.description || "";
      grow.append(title, sub);
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = `L${tool.level}${tool.confirm ? " · confirm" : ""}`;
      li.append(grow, badge);
      list.append(li);
    }
  } catch (error) {
    toast(error.message);
  }
}

async function refreshStatusLite() {
  try {
    const data = await jsonApi("/api/status");
    $("#session-label").textContent = data.session_id || "";
    $$(".status-dot").forEach((dot) => dot.classList.remove("off"));
  } catch (_) {
    $$(".status-dot").forEach((dot) => dot.classList.add("off"));
  }
}

// ---------------------------------------------------------------- boot
async function boot() {
  $("#app").hidden = false;
  $("#auth").hidden = true;
  startVoiceGreeting();
  await refreshStatusLite();
  await loadMemory();
  await loadKnowledge();
  $("#chat-input").focus();
}

(async function start() {
  state.token = readTokenFromHash();
  if (!state.token) {
    $("#auth").hidden = false;
    $("#token-input").focus();
    return;
  }
  try {
    await jsonApi("/api/status");
    await boot();
  } catch (_) {
    lock();
  }
})();
