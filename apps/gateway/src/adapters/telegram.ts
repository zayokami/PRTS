import { Bot, webhookCallback } from "grammy";
import type { FastifyReply, FastifyRequest } from "fastify";
import { createWriteStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

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

const TEXT_EXTENSIONS = new Set([
  ".txt", ".md", ".markdown", ".json", ".js", ".ts", ".jsx", ".tsx",
  ".py", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
  ".rb", ".php", ".sh", ".bash", ".zsh", ".ps1", ".csv", ".log",
  ".yaml", ".yml", ".xml", ".html", ".htm", ".css", ".scss", ".sass",
  ".ini", ".toml", ".cfg", ".conf", ".sql", ".gitignore",
]);

function isTextFile(filename: string): boolean {
  const ext = path.extname(filename).toLowerCase();
  return TEXT_EXTENSIONS.has(ext);
}

/** 读取文本文件前 maxBytes 字节,返回 UTF-8 字符串(超长截断)。 */
async function readTextPreview(filePath: string, maxBytes = 50_000): Promise<string | null> {
  try {
    const handle = await fs.open(filePath, "r");
    try {
      const buf = Buffer.alloc(maxBytes);
      const { bytesRead } = await handle.read(buf, 0, maxBytes, 0);
      let text = buf.toString("utf-8", 0, bytesRead);
      if (bytesRead === maxBytes) text += "\n…(截断)";
      return text;
    } finally {
      await handle.close();
    }
  } catch {
    return null;
  }
}

/** 从 Telegram CDN 下载文件到本地 destPath,返回最终路径。 */
async function downloadTelegramFile(
  bot: Bot,
  token: string,
  fileId: string,
  destPath: string,
): Promise<string | null> {
  try {
    const file = await bot.api.getFile(fileId);
    if (!file.file_path) return null;
    const url = `https://api.telegram.org/file/bot${token}/${file.file_path}`;
    const resp = await fetch(url);
    if (!resp.ok) return null;
    await fs.mkdir(path.dirname(destPath), { recursive: true });
    const body = resp.body;
    if (!body) return null;
    await pipeline(Readable.fromWeb(body as any), createWriteStream(destPath));
    return destPath;
  } catch (err) {
    console.error("[telegram] download failed:", err);
    return null;
  }
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

/**
 * 为 Telegram 消息构造统一文本描述(包含附件元信息)。
 *
 * 图片:保留 caption,附加本地保存路径。
 * 语音:转录文本(如有)或标记为语音消息。
 * 文件:如果是文本文件则内联内容预览,否则仅保留文件名和路径。
 */
async function buildMediaContent(
  bot: Bot,
  token: string,
  workspaceDir: string,
  chatId: number,
  message: any,
): Promise<string> {
  const ts = Date.now();
  const baseDir = path.join(workspaceDir, "telegram", "media", String(chatId));

  // Photo
  if (message.photo && message.photo.length > 0) {
    const photo = message.photo[message.photo.length - 1]; // 最大尺寸
    const fileName = `${ts}_photo.jpg`;
    const dest = path.join(baseDir, fileName);
    const saved = await downloadTelegramFile(bot, token, photo.file_id, dest);
    const caption = message.caption || "";
    const lines = caption ? [caption] : ["[用户发送了一张图片]"];
    if (saved) lines.push(`[文件: ${saved}]`);
    return lines.join("\n");
  }

  // Voice
  if (message.voice) {
    const ext = message.voice.mime_type?.includes("ogg") ? ".ogg" : ".audio";
    const fileName = `${ts}_voice${ext}`;
    const dest = path.join(baseDir, fileName);
    const saved = await downloadTelegramFile(bot, token, message.voice.file_id, dest);
    const caption = message.caption || "";
    const lines = caption ? [caption] : ["[用户发送了一段语音消息]"];
    if (saved) lines.push(`[文件: ${saved}]`);
    return lines.join("\n");
  }

  // Audio (music files etc.)
  if (message.audio) {
    const fileName = message.audio.file_name || `${ts}_audio.${message.audio.mime_type?.split("/")[1] || "bin"}`;
    const dest = path.join(baseDir, fileName);
    const saved = await downloadTelegramFile(bot, token, message.audio.file_id, dest);
    const caption = message.caption || "";
    const title = message.audio.title || message.audio.performer || "";
    const lines = caption ? [caption] : [title ? `[音频] ${title}` : "[用户发送了一个音频文件]"];
    if (saved) lines.push(`[文件: ${saved}]`);
    return lines.join("\n");
  }

  // Document
  if (message.document) {
    const fileName = message.document.file_name || `${ts}_document.bin`;
    const dest = path.join(baseDir, fileName);
    const saved = await downloadTelegramFile(bot, token, message.document.file_id, dest);
    const caption = message.caption || "";
    const lines = caption ? [caption] : [];

    if (saved) {
      if (isTextFile(fileName)) {
        const preview = await readTextPreview(saved, 20_000);
        if (preview !== null) {
          lines.push(`[文件: ${saved}]\n\`\`\`\n${preview}\n\`\`\``);
        } else {
          lines.push(`[文件: ${saved}]`);
        }
      } else {
        lines.push(`[文件: ${saved}]`);
      }
    }
    if (lines.length === 0) lines.push("[用户发送了一个文件]");
    return lines.join("\n");
  }

  // Video / VideoNote
  if (message.video || message.video_note) {
    const vid = message.video || message.video_note;
    const fileName = message.video?.file_name || `${ts}_video.mp4`;
    const dest = path.join(baseDir, fileName);
    const saved = await downloadTelegramFile(bot, token, vid.file_id, dest);
    const caption = message.caption || "";
    const lines = caption ? [caption] : [message.video ? "[用户发送了一个视频]" : "[用户发送了一个视频笔记]"];
    if (saved) lines.push(`[文件: ${saved}]`);
    return lines.join("\n");
  }

  return "[未知消息类型]";
}

/** 创建一个配置好的 grammY Bot 实例。
 *
 * 每个 chat 维护自己的 AbortController:如果用户在新消息到达时上一轮
 * 还没回复完,旧请求会被取消,避免串台和重复回复。
 *
 * @param workspaceDir 本地工作区目录,用于保存下载的 Telegram 媒体文件。
 */
export function createTelegramBot(agentUrl: string, token: string, workspaceDir: string): Bot {
  const bot = new Bot(token);

  // chatId → 当前进行中的 AbortController
  const inflight = new Map<number, AbortController>();

  async function handleMessage(chatId: number, content: string) {
    const old = inflight.get(chatId);
    if (old) {
      try { old.abort(); } catch { /* ignore */ }
    }

    const ac = new AbortController();
    inflight.set(chatId, ac);

    try {
      const reply = await fetchAgentReply(agentUrl, chatId, content, ac);
      await bot.api.sendMessage(chatId, reply, { parse_mode: "Markdown" });
    } catch (err) {
      const name = (err as { name?: string }).name;
      if (name === "AbortError") return;
      const msg = err instanceof Error ? err.message : String(err);
      await bot.api.sendMessage(chatId, `PRTS 处理出错: ${msg}`);
    } finally {
      if (inflight.get(chatId) === ac) {
        inflight.delete(chatId);
      }
    }
  }

  bot.on("message:text", async (ctx) => {
    await handleMessage(ctx.chat.id, ctx.message.text);
  });

  bot.on("message:photo", async (ctx) => {
    const content = await buildMediaContent(bot, token, workspaceDir, ctx.chat.id, ctx.message);
    await handleMessage(ctx.chat.id, content);
  });

  bot.on("message:voice", async (ctx) => {
    const content = await buildMediaContent(bot, token, workspaceDir, ctx.chat.id, ctx.message);
    await handleMessage(ctx.chat.id, content);
  });

  bot.on("message:audio", async (ctx) => {
    const content = await buildMediaContent(bot, token, workspaceDir, ctx.chat.id, ctx.message);
    await handleMessage(ctx.chat.id, content);
  });

  bot.on("message:document", async (ctx) => {
    const content = await buildMediaContent(bot, token, workspaceDir, ctx.chat.id, ctx.message);
    await handleMessage(ctx.chat.id, content);
  });

  bot.on("message:video", async (ctx) => {
    const content = await buildMediaContent(bot, token, workspaceDir, ctx.chat.id, ctx.message);
    await handleMessage(ctx.chat.id, content);
  });

  bot.on("message:video_note", async (ctx) => {
    const content = await buildMediaContent(bot, token, workspaceDir, ctx.chat.id, ctx.message);
    await handleMessage(ctx.chat.id, content);
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
