import { useEffect, useRef, useState, type FormEvent } from "react";

type Role = "user" | "assistant" | "system";
interface Msg {
  role: Role;
  content: string;
  kind?: "text" | "tool" | "notify";
}

type Frame =
  | { type: "ready"; session_id: string }
  | { type: "token"; text: string }
  | { type: "tool_call"; id: string; name: string; arguments: unknown }
  | { type: "tool_result"; id: string; name: string; result?: unknown; error?: unknown }
  | { type: "notify"; message: string; kind?: string; payload?: unknown }
  | { type: "done"; stop_reason?: string }
  | { type: "error"; message: string }
  | { type: "ping" }
  | { type: "pong" };

type ConnStatus = "connecting" | "ready" | "streaming" | "reconnecting" | "disconnected";

const SESSION_KEY = "prts.session_id";
const PING_INTERVAL_MS = 30000;
const INITIAL_RETRY_MS = 1000;
const MAX_RETRY_MS = 30000;

function summarize(value: unknown, max = 240): string {
  if (value === undefined || value === null) return "";
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return text.length > max ? text.slice(0, max) + "…" : text;
}

/** 指数退避计算下一次重连延迟 */
function nextRetryDelay(attempt: number): number {
  const jitter = Math.random() * 0.3 + 0.85; // 0.85–1.15
  return Math.min(INITIAL_RETRY_MS * Math.pow(2, attempt), MAX_RETRY_MS) * jitter;
}

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<ConnStatus>("connecting");
  const [sessionId, setSessionId] = useState<string>("");
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryCountRef = useRef(0);
  const shouldReconnectRef = useRef(true);

  const connect = () => {
    if (!shouldReconnectRef.current) return;

    const stored = localStorage.getItem(SESSION_KEY);
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const qs = stored ? `?session_id=${encodeURIComponent(stored)}` : "";
    const ws = new WebSocket(`${proto}://${location.host}/ws/chat${qs}`);
    wsRef.current = ws;

    ws.onopen = () => {
      retryCountRef.current = 0;
      setReconnectAttempt(0);
      // 心跳:每 30s 发 ping,如果 pong 没回来 server 会踢
      if (pingTimerRef.current) clearInterval(pingTimerRef.current);
      pingTimerRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, PING_INTERVAL_MS);
    };

    ws.onmessage = (ev) => {
      let frame: Frame;
      try {
        frame = JSON.parse(ev.data);
      } catch {
        return;
      }

      // 心跳 pong —— 不处理
      if (frame.type === "ping" || frame.type === "pong") return;

      if (frame.type === "ready") {
        const sid = frame.session_id;
        setSessionId(sid);
        localStorage.setItem(SESSION_KEY, sid);
        fetch(`/api/sessions/${encodeURIComponent(sid)}/history`)
          .then((r) => (r.ok ? r.json() : { messages: [] }))
          .then((j: { messages: { role: Role; content: string }[] }) => {
            const seed = (j.messages ?? []).filter((m) => m.role === "user" || m.role === "assistant");
            setMessages(seed.map((m) => ({ role: m.role, content: m.content, kind: "text" })));
            setStatus("ready");
          })
          .catch(() => setStatus("ready"));
      } else if (frame.type === "token") {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === "assistant" && last.kind === "text") {
            next[next.length - 1] = { ...last, content: last.content + frame.text };
          } else {
            next.push({ role: "assistant", content: frame.text, kind: "text" });
          }
          return next;
        });
      } else if (frame.type === "tool_call") {
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            kind: "tool",
            content: `→ 调用 ${frame.name}(${summarize(frame.arguments, 160)})`,
          },
        ]);
      } else if (frame.type === "tool_result") {
        const isErr = frame.error !== undefined && frame.error !== null;
        const label = isErr ? `✗ ${frame.name} 失败` : `← ${frame.name} 返回`;
        const body = summarize(isErr ? frame.error : frame.result, 320);
        setMessages((prev) => [
          ...prev,
          { role: "system", kind: "tool", content: `${label}: ${body}` },
        ]);
      } else if (frame.type === "notify") {
        setMessages((prev) => [
          ...prev,
          { role: "system", kind: "notify", content: `🔔 ${frame.message}` },
        ]);
      } else if (frame.type === "done") {
        setStatus("ready");
      } else if (frame.type === "error") {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `[error] ${frame.message}`, kind: "text" },
        ]);
        setStatus("ready");
      }
    };

    ws.onclose = () => {
      if (pingTimerRef.current) {
        clearInterval(pingTimerRef.current);
        pingTimerRef.current = null;
      }
      // 只有"当前活跃的"那个 ws 关闭了才触发重连逻辑
      if (wsRef.current !== ws) return;

      setStatus("disconnected");
      wsRef.current = null;

      if (!shouldReconnectRef.current) return;

      // 指数退避重连
      const delay = nextRetryDelay(retryCountRef.current);
      retryCountRef.current += 1;
      setReconnectAttempt(retryCountRef.current);
      setStatus("reconnecting");

      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(() => {
        connect();
      }, delay);
    };

    ws.onerror = () => {
      // onerror 后一定会跟着 onclose,重连逻辑在 onclose 里统一处理
      if (wsRef.current === ws) setStatus("disconnected");
    };
  };

  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();

    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (pingTimerRef.current) clearInterval(pingTimerRef.current);
      if (wsRef.current) {
        try { wsRef.current.close(); } catch { /* ignore */ }
        wsRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const send = (e: FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || status !== "ready") return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    ws.send(JSON.stringify({ type: "user", content: text }));
    setMessages((prev) => [...prev, { role: "user", content: text, kind: "text" }]);
    setInput("");
    setStatus("streaming");
  };

  const newSession = () => {
    localStorage.removeItem(SESSION_KEY);
    location.reload();
  };

  const statusLabel = {
    connecting: "连接中…",
    ready: "就绪",
    streaming: "生成中…",
    reconnecting: `重连中 (${reconnectAttempt})…`,
    disconnected: "已断开",
  }[status];

  const statusColor = {
    connecting: "#58a6ff",
    ready: "#3fb950",
    streaming: "#d29922",
    reconnecting: "#d29922",
    disconnected: "#f85149",
  }[status];

  return (
    <main style={S.main}>
      <header style={S.header}>
        <h1 style={S.title}>PRTS</h1>
        <span style={{ ...S.meta, color: statusColor }}>
          ● {statusLabel} {sessionId && `· ${sessionId.slice(0, 8)}`}
        </span>
        <button type="button" onClick={newSession} style={S.newBtn} title="开新会话">
          新会话
        </button>
      </header>

      <div ref={listRef} style={S.list}>
        {messages.length === 0 && <div style={S.hint}>博士,有什么需要协助的?</div>}
        {messages.map((m, i) => {
          const rowStyle =
            m.role === "user"
              ? { ...S.row, ...S.rowUser }
              : m.role === "system"
              ? { ...S.row, ...S.rowSystem }
              : { ...S.row, ...S.rowAssistant };
          const label =
            m.role === "user"
              ? "博士"
              : m.role === "system"
              ? m.kind === "notify"
                ? "通知"
                : "工具"
              : "PRTS";
          const contentStyle = m.role === "system" ? { ...S.content, ...S.contentSystem } : S.content;
          return (
            <div key={i} style={rowStyle}>
              <span style={S.role}>{label}</span>
              <pre style={contentStyle}>{m.content}</pre>
            </div>
          );
        })}
      </div>

      <form onSubmit={send} style={S.form}>
        <input
          autoFocus
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            status === "ready"
              ? "输入消息,回车发送"
              : status === "streaming"
              ? "PRTS 正在思考…"
              : status === "reconnecting"
              ? `重连中,请稍候 (${reconnectAttempt})…`
              : "连接断开,正在重连…"
          }
          disabled={status !== "ready"}
          style={S.input}
        />
        <button type="submit" disabled={status !== "ready" || !input.trim()} style={S.btn}>
          发送
        </button>
      </form>
    </main>
  );
}

const S: Record<string, React.CSSProperties> = {
  main: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    fontFamily: "ui-monospace, monospace",
    background: "#0d1117",
    color: "#e6edf3",
  },
  header: {
    display: "flex",
    alignItems: "baseline",
    gap: 12,
    padding: "12px 20px",
    borderBottom: "1px solid #30363d",
  },
  title: { margin: 0, fontSize: 18, letterSpacing: 2 },
  meta: { fontSize: 12, opacity: 0.9, flex: 1, fontWeight: 500 },
  newBtn: {
    fontSize: 12,
    padding: "4px 10px",
    background: "transparent",
    color: "#e6edf3",
    border: "1px solid #30363d",
    borderRadius: 4,
    cursor: "pointer",
    fontFamily: "inherit",
  },
  list: {
    flex: 1,
    overflowY: "auto",
    padding: "16px 20px",
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  hint: { opacity: 0.4, textAlign: "center", marginTop: 40 },
  row: { display: "flex", flexDirection: "column", gap: 4, maxWidth: "80ch" },
  rowUser: { alignSelf: "flex-end", alignItems: "flex-end" },
  rowAssistant: { alignSelf: "flex-start" },
  rowSystem: { alignSelf: "center", alignItems: "center", opacity: 0.7, maxWidth: "70ch" },
  role: { fontSize: 11, opacity: 0.5 },
  content: {
    margin: 0,
    padding: "8px 12px",
    background: "#161b22",
    border: "1px solid #30363d",
    borderRadius: 6,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
    fontFamily: "inherit",
    fontSize: 14,
  },
  contentSystem: {
    background: "#10171f",
    borderColor: "#1f2a36",
    fontSize: 12,
    color: "#9aa7b3",
  },
  form: {
    display: "flex",
    gap: 8,
    padding: "12px 20px",
    borderTop: "1px solid #30363d",
  },
  input: {
    flex: 1,
    padding: "8px 12px",
    background: "#161b22",
    color: "inherit",
    border: "1px solid #30363d",
    borderRadius: 6,
    fontFamily: "inherit",
    fontSize: 14,
  },
  btn: {
    padding: "8px 16px",
    background: "#238636",
    color: "white",
    border: 0,
    borderRadius: 6,
    cursor: "pointer",
    fontFamily: "inherit",
  },
};
