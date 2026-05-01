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

interface InboundUserFrame {
  type: "user";
  content: string;
}

interface OutboundFrame {
  type: "ready" | "token" | "done" | "error";
  text?: string;
  message?: string;
  session_id?: string;
}

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

app.get<{ Querystring: { session_id?: string } }>(
  "/ws/chat",
  { websocket: true },
  (socket, req) => {
    const requested = typeof req.query.session_id === "string" ? req.query.session_id : null;
    const activeSessionId =
      requested && /^[\w-]{1,64}$/.test(requested) ? requested : randomUUID();
    let busy = false;

    const send = (frame: OutboundFrame) => {
      try { socket.send(JSON.stringify(frame)); } catch (e) { app.log.warn({ e }, "ws send failed"); }
    };

    send({ type: "ready", session_id: activeSessionId });

    socket.on("message", async (raw: Buffer) => {
      if (busy) {
        send({ type: "error", message: "上一轮还在生成中,请等待" });
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

      try {
        const resp = await fetch(`${AGENT_URL}/agent/v1/converse`, {
          method: "POST",
          headers: { "content-type": "application/json", accept: "text/event-stream" },
          body: JSON.stringify({
            session_id: activeSessionId,
            content: frame.content,
            channel: "web",
          }),
        });

        if (!resp.ok || !resp.body) {
          const text = await resp.text().catch(() => "");
          send({ type: "error", message: `agent ${resp.status}: ${text.slice(0, 200)}` });
          busy = false;
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
            if (event === "token") {
              try {
                const { text } = JSON.parse(data) as { text: string };
                send({ type: "token", text });
              } catch { /* skip malformed */ }
            } else if (event === "error") {
              try {
                const { message } = JSON.parse(data) as { message: string };
                send({ type: "error", message });
              } catch {
                send({ type: "error", message: data });
              }
            } else if (event === "done") {
              send({ type: "done" });
            }
            result = gen.next();
          }
          buffer = result.value as string;
        }
      } catch (err) {
        app.log.error({ err }, "agent bridge failed");
        send({ type: "error", message: err instanceof Error ? err.message : String(err) });
      } finally {
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
