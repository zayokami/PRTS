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

// P1 阶段填充:WS chat 转发 / SSE 流
app.get("/ws/chat", { websocket: true }, (socket) => {
  socket.send(JSON.stringify({ type: "hello", from: "gateway", note: "P0 stub — 聊天逻辑在 P1 阶段实现" }));
  socket.on("message", (raw) => {
    socket.send(JSON.stringify({ type: "echo", payload: raw.toString() }));
  });
});

try {
  await app.listen({ port: PORT, host: "127.0.0.1" });
} catch (err) {
  app.log.error(err);
  process.exit(1);
}
