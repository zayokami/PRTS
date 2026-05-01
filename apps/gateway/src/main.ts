import { randomUUID } from "node:crypto";
import Fastify from "fastify";
import websocket from "@fastify/websocket";

const PORT = Number(process.env.GATEWAY_PORT ?? 4787);
const AGENT_URL = process.env.AGENT_URL ?? `http://127.0.0.1:${process.env.AGENT_PORT ?? 4788}`;

const app = Fastify({ logger: true });

await app.register(websocket);

app.get("/health", async () => ({
  service: "prts-gateway",
  ok: true,
  agent_url: AGENT_URL,
  ts: new Date().toISOString(),
}));

// 透传到 agent: GET /agent/v1/sessions/:id/history
app.get<{ Params: { id: string } }>("/sessions/:id/history", async (req, reply) => {
  const resp = await fetch(`${AGENT_URL}/agent/v1/sessions/${encodeURIComponent(req.params.id)}/history`);
  reply.code(resp.status).type("application/json").send(await resp.text());
});

// 透传到 agent: GET /agent/v1/skills
app.get("/skills", async (_req, reply) => {
  const resp = await fetch(`${AGENT_URL}/agent/v1/skills`);
  reply.code(resp.status).type("application/json").send(await resp.text());
});

interface InboundUserFrame {
  type: "user";
  content: string;
}

type OutboundFrame =
  | { type: "ready"; session_id: string }
  | { type: "token"; text: string }
  | { type: "tool_call"; id: string; name: string; arguments: unknown }
  | { type: "tool_result"; id: string; name: string; result?: unknown; error?: unknown }
  | { type: "notify"; message: string; kind?: string; payload?: unknown }
  | { type: "done"; stop_reason?: string }
  | { type: "error"; message: string };

/** 解析 SSE: `event: x\ndata: {...}\n\n` —— 上游 sse-starlette 用 CRLF,调用方需先规范化。 */
function* parseSseChunks(buffer: string): Generator<{ event: string; data: string }, string> {
  let cursor = 0;
  while (true) {
    const sep = buffer.indexOf("\n\n", cursor);
    if (sep < 0) break;
    const block = buffer.slice(cursor, sep);
    cursor = sep + 2;
    let event = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    yield { event, data: dataLines.join("\n") };
  }
  return buffer.slice(cursor);
}

// 4 MiB:把单条 user 消息打到这里基本就是恶意了,SSE 上游也会被这种行为拖住。
const MAX_USER_CONTENT_BYTES = 4 * 1024 * 1024;

app.get<{ Querystring: { session_id?: string } }>(
  "/ws/chat",
  { websocket: true },
  (socket, req) => {
    const requested = typeof req.query.session_id === "string" ? req.query.session_id : null;
    const activeSessionId =
      requested && /^[\w-]{1,64}$/.test(requested) ? requested : randomUUID();
    let busy = false;
    // WS 关闭时把进行中的 agent fetch 一并中断 —— 否则 agent 会继续算下去,
    // 浪费 LLM token 也吃 SQLite 写盘。
    let inflight: AbortController | null = null;

    const send = (frame: OutboundFrame) => {
      if (socket.readyState !== socket.OPEN) return;
      try { socket.send(JSON.stringify(frame)); } catch (e) { app.log.warn({ e }, "ws send failed"); }
    };

    send({ type: "ready", session_id: activeSessionId });

    socket.on("close", () => {
      if (inflight) {
        try { inflight.abort(); } catch { /* ignore */ }
        inflight = null;
      }
    });

    socket.on("message", async (raw: Buffer) => {
      if (busy) {
        send({ type: "error", message: "上一轮还在生成中,请等待" });
        return;
      }

      if (raw.byteLength > MAX_USER_CONTENT_BYTES) {
        send({ type: "error", message: `消息过大 (${raw.byteLength} bytes > ${MAX_USER_CONTENT_BYTES})` });
        return;
      }

      let frame: InboundUserFrame;
      try {
        frame = JSON.parse(raw.toString()) as InboundUserFrame;
      } catch {
        send({ type: "error", message: "bad json" });
        return;
      }
      if (frame.type !== "user" || typeof frame.content !== "string") {
        send({ type: "error", message: "expected {type:'user', content:string}" });
        return;
      }

      busy = true;
      let sawDone = false;
      const ac = new AbortController();
      inflight = ac;

      try {
        const resp = await fetch(`${AGENT_URL}/agent/v1/converse`, {
          method: "POST",
          headers: { "content-type": "application/json", accept: "text/event-stream" },
          body: JSON.stringify({
            session_id: activeSessionId,
            content: frame.content,
            channel: "web",
          }),
          signal: ac.signal,
        });

        if (!resp.ok || !resp.body) {
          const text = await resp.text().catch(() => "");
          send({ type: "error", message: `agent ${resp.status}: ${text.slice(0, 200)}` });
          return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
          const gen = parseSseChunks(buffer);
          let result = gen.next();
          while (!result.done) {
            const { event, data } = result.value;
            try {
              if (event === "token") {
                const { text } = JSON.parse(data) as { text: string };
                send({ type: "token", text });
              } else if (event === "tool_call") {
                const { id, name, arguments: args } = JSON.parse(data) as {
                  id: string; name: string; arguments: unknown;
                };
                send({ type: "tool_call", id, name, arguments: args });
              } else if (event === "tool_result") {
                const { id, name, result: r, error } = JSON.parse(data) as {
                  id: string; name: string; result?: unknown; error?: unknown;
                };
                send({ type: "tool_result", id, name, result: r, error });
              } else if (event === "notify") {
                const { message, kind, payload } = JSON.parse(data) as {
                  message: string; kind?: string; payload?: unknown;
                };
                send({ type: "notify", message, kind, payload });
              } else if (event === "error") {
                const { message } = JSON.parse(data) as { message: string };
                send({ type: "error", message });
              } else if (event === "done") {
                let stop_reason: string | undefined;
                try {
                  stop_reason = (JSON.parse(data) as { stop_reason?: string }).stop_reason;
                } catch { /* keep undefined */ }
                send({ type: "done", stop_reason });
                sawDone = true;
              }
            } catch (e) {
              app.log.warn({ e, event, data }, "malformed sse chunk");
              // 上游帧损坏也告诉前端,不要让 UI 留在 streaming 转圈状态。
              send({
                type: "error",
                message: `上游 SSE 帧解析失败 (event=${event})`,
              });
            }
            result = gen.next();
          }
          buffer = result.value as string;
        }

        // 上游流自然结束但没发 done(网关被踢、agent 提前 close 等) ——
        // 给前端一个兜底 done,让 UI 能从 streaming 退出去。
        if (!sawDone) {
          send({ type: "done", stop_reason: "stream_closed" });
        }
      } catch (err) {
        if ((err as { name?: string }).name === "AbortError") {
          app.log.info({ session: activeSessionId }, "fetch aborted (client disconnected)");
          return;
        }
        app.log.error({ err }, "agent bridge failed");
        send({ type: "error", message: err instanceof Error ? err.message : String(err) });
        if (!sawDone) send({ type: "done", stop_reason: "error" });
      } finally {
        if (inflight === ac) inflight = null;
        busy = false;
      }
    });
  },
);

try {
  await app.listen({ port: PORT, host: "127.0.0.1" });
} catch (err) {
  app.log.error(err);
  process.exit(1);
}
