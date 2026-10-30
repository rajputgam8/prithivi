"use strict";

/* GitHub Pages preview only. No backend is reachable from a static site, so
   every /api request below is answered with clearly-labelled demo data. */

(function () {
  const banner = document.createElement("div");
  banner.id = "preview-banner";
  banner.innerHTML =
    "<strong>UI PREVIEW</strong> · demo data only — run Prithivi locally for the real AI backend";
  document.body.appendChild(banner);

  const style = document.createElement("style");
  style.textContent = `
    #preview-banner {
      position: fixed; left: 0; right: 0; bottom: 0; z-index: 99999;
      padding: 9px 14px; text-align: center;
      font: 500 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
      color: #f4e8c1; background: rgba(24, 18, 8, 0.94);
      border-top: 1px solid rgba(255, 196, 87, 0.45);
      letter-spacing: 0.02em;
    }
    #preview-banner strong { color: #ffc457; letter-spacing: 0.08em; }
    body { padding-bottom: 40px; }
  `;
  document.head.appendChild(style);

  sessionStorage.setItem("prithivi.token", "preview-token");

  const json = (data, status = 200) =>
    new Response(JSON.stringify(data), {
      status,
      headers: { "Content-Type": "application/json" },
    });

  const status = {
    session_id: "preview-session",
    state: "idle",
    turns: 3,
    provider: "preview",
    model: "static demo (no backend)",
    memory: { enabled: true, count: 2, by_type: { preference: 1, fact: 1 } },
    knowledge: { documents: 2, chunks: 14 },
    tools: 8,
    audit_ok: true,
  };

  let memoryEnabled = true;
  const memories = [
    {
      key: "preferred_name",
      text: "The user prefers to be called by their first name.",
      type: "preference",
      value: null,
      pinned: true,
      source: "user_stated",
      confidence: 1,
    },
    {
      key: "timezone",
      text: "The user is in India Standard Time.",
      type: "fact",
      value: null,
      pinned: false,
      source: "user_stated",
      confidence: 0.9,
    },
  ];

  const documents = [
    {
      id: "doc_preview_notes",
      title: "Project notes.pdf",
      uri: "file:///preview/project-notes.pdf",
      status: "indexed",
      source_type: "pdf",
      chunk_count: 9,
    },
    {
      id: "doc_preview_readme",
      title: "README.md",
      uri: "file:///preview/README.md",
      status: "indexed",
      source_type: "markdown",
      chunk_count: 5,
    },
  ];

  const tools = [
    { name: "get_time", description: "Return the current local time.", level: 0, confirm: false },
    { name: "calculate", description: "Evaluate a maths expression.", level: 0, confirm: false },
    { name: "search_web", description: "Search the public web with citations.", level: 1, confirm: false },
    { name: "web_fetch", description: "Read one public web page.", level: 1, confirm: false },
    { name: "remember_memory", description: "Store a durable fact or preference.", level: 1, confirm: false },
    { name: "recall_memory", description: "Recall stored facts and preferences.", level: 1, confirm: false },
    { name: "search_knowledge", description: "Search the local knowledge base.", level: 1, confirm: false },
    { name: "open_url", description: "Open a URL in the device browser.", level: 2, confirm: true },
  ];

  function sse(events) {
    const encoder = new TextEncoder();
    return new Response(
      new ReadableStream({
        start(controller) {
          let index = 0;
          const push = () => {
            if (index >= events.length) {
              controller.close();
              return;
            }
            const [event, data] = events[index++];
            controller.enqueue(
              encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
            );
            setTimeout(push, event === "delta" ? 420 : 120);
          };
          push();
        },
      }),
      { status: 200, headers: { "Content-Type": "text/event-stream" } }
    );
  }

  const chatReply =
    "This is the static UI preview. The chat, memory, and knowledge panels are wired to demo data, not to a live model. Run Prithivi on your phone to connect the real backend.";

  const originalFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const rawUrl = typeof input === "string" ? input : input.url;
    const path = new URL(rawUrl, location.href).pathname;
    if (!path.startsWith("/api/")) return originalFetch(input, init);

    const method = String(
      (init && init.method) || (typeof input !== "string" && input.method) || "GET"
    ).toUpperCase();
    let body = {};
    if (init && typeof init.body === "string") {
      try {
        body = JSON.parse(init.body);
      } catch (_) {
        body = {};
      }
    }

    if (path === "/api/status") return Promise.resolve(json(status));

    if (path === "/api/memory" && method === "GET") {
      return Promise.resolve(
        json({
          enabled: memoryEnabled,
          memories: memoryEnabled ? memories : [],
          stats: {
            count: memoryEnabled ? memories.length : 0,
            by_type: { preference: 1, fact: 1 },
          },
        })
      );
    }
    if (path === "/api/memory" && method === "POST") {
      const record = {
        key: body.key || `note_${memories.length + 1}`,
        text: body.text || "Remembered item",
        type: body.type || "fact",
        value: null,
        pinned: !!body.pinned,
        source: "user_stated",
        confidence: 1,
      };
      memories.unshift(record);
      return Promise.resolve(json(record, 201));
    }
    if (path === "/api/memory/toggle") {
      memoryEnabled = !!body.enabled;
      return Promise.resolve(json({ enabled: memoryEnabled }));
    }
    if (path.startsWith("/api/memory/") && method === "DELETE") {
      return Promise.resolve(json({ deleted: true }));
    }

    if (path === "/api/knowledge" && method === "GET") {
      return Promise.resolve(
        json({
          documents,
          stats: {
            documents: documents.length,
            chunks: documents.reduce((sum, doc) => sum + doc.chunk_count, 0),
          },
        })
      );
    }
    if (path === "/api/knowledge/ingest") {
      const doc = {
        id: `doc_preview_${Date.now()}`,
        title: body.title || "Preview note",
        uri: body.uri || "note://preview",
        status: "indexed",
        source_type: "note",
        chunk_count: 3,
      };
      documents.unshift(doc);
      return Promise.resolve(json(Object.assign({ duplicate: false }, doc), 201));
    }
    if (path.startsWith("/api/knowledge/") && method === "DELETE") {
      return Promise.resolve(json({ deleted: true }));
    }

    if (path === "/api/tools") return Promise.resolve(json({ tools }));
    if (path === "/api/sessions") return Promise.resolve(json({ sessions: [] }));
    if (path === "/api/confirm") return Promise.resolve(json({ resolved: true }));

    if (path === "/api/chat") {
      return Promise.resolve(
        sse([
          ["delta", { text: "This is the static UI preview. " }],
          ["delta", { text: "The chat, memory, and knowledge panels are wired to demo data, " }],
          ["delta", { text: "not to a live model. " }],
          ["delta", { text: "Run Prithivi on your phone to connect the real backend." }],
          [
            "final",
            {
              text: chatReply,
              provenance: ["preview · demo data"],
              tools: [],
              blocked: false,
              iterations: 1,
              elapsed_ms: 42,
            },
          ],
          ["done", {}],
        ])
      );
    }

    return Promise.resolve(json({ error: "not found in preview" }, 404));
  };
})();
