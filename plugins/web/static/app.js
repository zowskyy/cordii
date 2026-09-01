const state = {
    sessionId: null,
    eventSource: null,
};

const els = {
    connectionStatus: document.getElementById("connection-status"),
    activeSession: document.getElementById("active-session"),
    sessionList: document.getElementById("session-list"),
    refreshSessions: document.getElementById("refresh-sessions"),
    modelList: document.getElementById("model-list"),
    metrics: document.getElementById("metrics"),
    chatMessages: document.getElementById("chat-messages"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
};

function setStatus(text, kind = "") {
    els.connectionStatus.textContent = text;
    els.connectionStatus.className = "badge " + kind;
}

function appendMessage(role, content) {
    const node = document.createElement("div");
    node.className = "message " + role;
    node.innerHTML = `<div class="meta">${role}</div><div class="content">${escapeHtml(String(content ?? ""))}</div>`;
    els.chatMessages.appendChild(node);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

async function loadSessions() {
    const res = await fetch("/api/sessions");
    const data = await res.json();
    els.sessionList.innerHTML = "";
    for (const session of data) {
        const item = document.createElement("div");
        item.className = "session-item" + (session.session_id === state.sessionId ? " active" : "");
        const label = document.createElement("span");
        label.textContent = session.session_id;
        const actions = document.createElement("div");
        const selectBtn = document.createElement("button");
        selectBtn.textContent = "Open";
        selectBtn.onclick = () => selectSession(session.session_id);
        const deleteBtn = document.createElement("button");
        deleteBtn.textContent = "Delete";
        deleteBtn.onclick = async () => {
            await fetch(`/api/sessions/${encodeURIComponent(session.session_id)}`, { method: "DELETE" });
            if (state.sessionId === session.session_id) {
                state.sessionId = null;
                els.chatMessages.innerHTML = '<div class="empty">Select or create a session to begin.</div>';
                els.activeSession.textContent = "no session";
            }
            loadSessions();
        };
        actions.appendChild(selectBtn);
        actions.appendChild(deleteBtn);
        item.appendChild(label);
        item.appendChild(actions);
        els.sessionList.appendChild(item);
    }
}

async function loadModels() {
    const res = await fetch("/api/models");
    const data = await res.json();
    els.modelList.innerHTML = "";
    for (const model of data) {
        const item = document.createElement("div");
        item.className = "model-item";
        const label = document.createElement("span");
        label.textContent = model.id;
        const actions = document.createElement("div");
        const switchBtn = document.createElement("button");
        switchBtn.textContent = "Switch";
        switchBtn.onclick = async () => {
            await fetch(`/api/models/${encodeURIComponent(model.id)}/switch`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
            loadModels();
        };
        actions.appendChild(switchBtn);
        item.appendChild(label);
        item.appendChild(actions);
        els.modelList.appendChild(item);
    }
}

async function loadMetrics() {
    const res = await fetch("/api/metrics");
    const data = await res.json();
    els.metrics.textContent = JSON.stringify(data, null, 2);
}

async function selectSession(sessionId) {
    state.sessionId = sessionId;
    els.activeSession.textContent = sessionId;
    els.chatMessages.innerHTML = "";
    loadSessions();
    await streamSession(sessionId);
}

async function streamSession(sessionId) {
    if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
    }
    const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/events`);
    if (!res.body) {
        return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
        const { value, done } = await reader.read();
        if (done) {
            break;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith("data:")) {
                continue;
            }
            const payload = trimmed.slice(5).trim();
            if (!payload || payload === "[DONE]") {
                continue;
            }
            try {
                const event = JSON.parse(payload);
                renderEvent(event);
            } catch {
                // ignore parse errors from keep-alive frames
            }
        }
    }
}

function renderEvent(event) {
    if (!event || !event.type) {
        return;
    }
    if (event.type === "user.message") {
        appendMessage("user", event.payload?.content ?? event.payload);
    } else if (event.type === "assistant.message") {
        appendMessage("assistant", event.payload?.content ?? event.payload);
    } else if (event.type.startsWith("tool.")) {
        appendMessage("system", `${event.type}: ${JSON.stringify(event.payload).slice(0, 500)}`);
    } else if (event.type === "turn.start" || event.type === "turn.end") {
        appendMessage("system", event.type);
    }
}

async function runCurrentSession(input) {
    if (!state.sessionId) {
        return;
    }
    appendMessage("user", input);
    const res = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input }),
    });
    const data = await res.json();
    if (data.error) {
        appendMessage("system", "Error: " + data.error);
        return;
    }
    if (data.result) {
        appendMessage("assistant", data.result);
    }
}

els.refreshSessions.onclick = loadSessions;

els.chatForm.onsubmit = async (ev) => {
    ev.preventDefault();
    const text = els.chatInput.value.trim();
    if (!text) {
        return;
    }
    els.chatInput.value = "";
    if (!state.sessionId) {
        const res = await fetch("/api/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input: text }) });
        const data = await res.json();
        if (data.session_id) {
            await selectSession(data.session_id);
            if (data.result) {
                appendMessage("assistant", data.result);
            }
            return;
        }
    }
    await runCurrentSession(text);
};

async function init() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        setStatus(data.status === "ok" ? "online" : "degraded", data.status === "ok" ? "ok" : "danger");
    } catch {
        setStatus("offline", "danger");
    }
    loadSessions();
    loadModels();
    loadMetrics();
    setInterval(loadMetrics, 5000);
}

init();
