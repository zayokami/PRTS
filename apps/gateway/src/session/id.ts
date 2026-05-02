/** Session ID 生成与规范化。
 *
 * 不同渠道(session_id 空间)必须隔离,否则同一用户通过 Web 和 Telegram 会
 * 看到彼此的历史,违反 P5 "不串台" 要求。
 */
import { randomUUID } from "node:crypto";

const SESSION_ID_RE = /^[\w-]{1,64}$/;

/** 为 Web Dashboard 客户端生成新 session ID。 */
export function generateWebSessionId(): string {
  // UUID v4 去掉 `-` 后 32 字符,符合 SESSION_ID_RE
  return randomUUID().replace(/-/g, "");
}

/** 把 Telegram chatId 映射为固定 session_id。
 *
 * 固定映射的好处:重启 Gateway 后同一 Telegram 用户仍落在同一会话,
 * 历史记录不丢。加 `tg-` 前缀避免与 Web UUID 空间冲突。
 */
export function telegramSessionId(chatId: number | string): string {
  return `tg-${chatId}`;
}

/** 校验 session_id 是否合法(供路由参数校验用)。 */
export function isValidSessionId(id: string): boolean {
  return SESSION_ID_RE.test(id);
}
