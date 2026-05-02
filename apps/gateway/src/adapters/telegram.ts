import { Bot, webhookCallback } from "grammy";
import type { FastifyReply, FastifyRequest } from "fastify";

import { telegramSessionId } from "../session/id.js";

/** 从 Agent SSE 流聚合出的单个事件。 */
interface AgentEvent {
  event: string;
  data: Record<string, unknown>;
}

/** 把文本截断到 max 字符,用于 tool_result 摘要。 */
function summarize(value: unknown, max = 240): string {
  if (value === undefined || value === null) return "";
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return text.length > max ? text.slice(0, max) + "…" : text;
}

/** 消费完整的 SSE 响应体,解析为事件列表。
 *
 * Telegram 不需要像 WebSocket 那样逐帧实时推送,而是等 Agent 流结束后
 * 聚合为一条消息回复。因此这里一次性读完全部 body 再解析。
 */
async function consumeSse(resp: Response): Promise<AgentEvent[]> {
  if (!resp.body) return [];
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
  }

  const events: AgentEvent[] = [];
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
    try {
      events.push({ event, data: JSON.parse(dataLines.join("\n")) });
    } catch {
      // 跳过无法解析的帧
    }
  }
  return events;
}

/** 向 Agent 发 converse 请求,返回聚合后的 Markdown 回复文本。 */
async function fetchAgentReply(
  agentUrl: string,
  chatId: number,
  text: string,
  ac: AbortController,
): Promise<string> {
  const sessionId = telegramSessionId(chatId);
  const resp = await fetch(`${agentUrl}/agent/v1/converse`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "text/event-stream",
    },
    body: JSON.stringify({
      session_id: sessionId,
      content: text,
      channel: "telegram",
      user_ref: String(chatId),
    }),
    signal: ac.signal,
  });

  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`agent ${resp.status}: ${body.slice(0, 200)}`);
  }

  const events = await consumeSse(resp);

  // 聚合 token + 内联 tool 痕迹
  const parts: string[] = [];
  for (const evt of events) {
    if (evt.event === "token") {
      const t = evt.data.text;
      if (typeof t === "string") parts.push(t);
    } else if (evt.event === "tool_call") {
      const name = String(evt.data.name ?? "");
      const args = summarize(evt.data.arguments, 120);
      parts.push(`\n\n→ **${name}**${args ? `(${args})` : ""}`);
    } else if (evt.event === "tool_result") {
      const name = String(evt.data.name ?? "");
      const isErr = evt.data.error !== undefined && evt.data.error !== null;
      const body = summarize(isErr ? evt.data.error : evt.data.result, 200);
      parts.push(`\n← **${name}**: ${isErr ? "❌ " : ""}${body}`);
    } else if (evt.event === "error") {
      const msg = String(evt.data.message ?? "");
      parts.push(`\n\n⚠️ *错误*: ${msg}`);
    }
    // done / notify 不追加到文本(notify 在 Telegram 里没有对应 UI)
  }

  return parts.join("").trim() || "(无回复)";
}

/** 创建一个配置好的 grammY Bot 实例。
 *
 * 每个 chat 维护自己的 AbortController:如果用户在新消息到达时上一轮
 * 还没回复完,旧请求会被取消,避免串台和重复回复。
 */
export function createTelegramBot(agentUrl: string, token: string): Bot {
  const bot = new Bot(token);

  // chatId → 当前进行中的 AbortController
  const inflight = new Map<number, AbortController>();

  bot.on("message:text", async (ctx) => {
    const chatId = ctx.chat.id;
    const text = ctx.message.text;

    // 取消上一轮(如果还在跑)
    const old = inflight.get(chatId);
    if (old) {
      try {
        old.abort();
      } catch {
        /* ignore */
      }
    }

    const ac = new AbortController();
    inflight.set(chatId, ac);

    try {
      const reply = await fetchAgentReply(agentUrl, chatId, text, ac);
      await ctx.reply(reply, { parse_mode: "Markdown" });
    } catch (err) {
      const name = (err as { name?: string }).name;
      if (name === "AbortError") {
        // 被新消息取消 —— 静默,不做任何回复
        return;
      }
      const msg = err instanceof Error ? err.message : String(err);
      await ctx.reply(`PRTS 处理出错: ${msg}`);
    } finally {
      if (inflight.get(chatId) === ac) {
        inflight.delete(chatId);
      }
    }
  });

  return bot;
}

/** polling 模式启动 bot。 */
export async function startTelegramBot(bot: Bot): Promise<void> {
  await bot.init();
  const me = bot.botInfo;
  console.log(`[telegram] polling as @${me.username}`);
  bot.start();
}

/** 为 Fastify webhook 模式创建 route handler。 */
export function createWebhookHandler(
  bot: Bot,
): (req: FastifyRequest, reply: FastifyReply) => Promise<unknown> {
  // grammY 内置 Fastify adapter,直接返回兼容的 handler
  return webhookCallback(bot, "fastify");
}
