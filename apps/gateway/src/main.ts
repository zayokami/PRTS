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

type ChatRole = "system" | "user" | "assistant";
interface ChatMessage {
  role: ChatRole;
  content: string;
}

interface InboundUserFrame {
  type: "user";
  content: string;
}

interface OutboundFrame {
  type: "ready" | "token" | "done" | "error" | "echo";
  text?: string;
  message?: string;
  session_id?: string;
}

/** 解析 sse-starlette 输出的 `event: x\ndata: {...}\n\n` 文本块。 */
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

app.get("/ws/chat", { websocket: true }, (socket) => {
  const sessionId = randomUUID();
  const history: ChatMessage[] = [];
  let busy = false;

  const send = (frame: OutboundFrame) => {
    try { socket.send(JSON.stringify(frame)); } catch (e) { app.log.warn({ e }, "ws send failed"); }
  };

  send({ type: "ready", session_id: sessionId });

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

    history.push({ role: "user", content: frame.content });
    busy = true;

    try {
      const resp = await fetch(`${AGENT_URL}/agent/v1/converse`, {
        method: "POST",
        headers: { "content-type": "application/json", accept: "text/event-stream" },
        body: JSON.stringify({ session_id: sessionId, messages: history }),
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
      let assistant = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        // sse-starlette 用 CRLF,统一规范化
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        const gen = parseSseChunks(buffer);
        let result = gen.next();
        while (!result.done) {
          const { event, data } = result.value;
          if (event === "token") {
            try {
              const { text } = JSON.parse(data) as { text: string };
              assistant += text;
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

      if (assistant) history.push({ role: "assistant", content: assistant });
    } catch (err) {
      app.log.error({ err }, "agent bridge failed");
      send({ type: "error", message: err instanceof Error ? err.message : String(err) });
    } finally {
      busy = false;
    }
  });
});

try {
  await app.listen({ port: PORT, host: "127.0.0.1" });
} catch (err) {
  app.log.error(err);
  process.exit(1);
}
