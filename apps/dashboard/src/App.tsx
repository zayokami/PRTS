import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  Plus,
  MessageSquare,
  Settings,
  ChevronDown,
  ArrowUp,
  Paperclip,
  Mic,
  PanelLeftClose,
  PanelLeftOpen,
  Copy,
  Check,
  RotateCcw,
} from "lucide-react";

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
const MAX_MESSAGES = 500;

function summarize(value: unknown, max = 240): string {
  if (value === undefined || value === null) return "";
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return text.length > max ? text.slice(0, max) + "…" : text;
}

function nextRetryDelay(attempt: number): number {
  const jitter = Math.random() * 0.3 + 0.85;
  return Math.min(INITIAL_RETRY_MS * Math.pow(2, attempt), MAX_RETRY_MS) * jitter;
}

/* ------------------------------------------------------------------ */
/*  Sidebar                                                           */
/* ------------------------------------------------------------------ */
function Sidebar({
  isOpen,
  onToggle,
  onNewChat,
  currentSessionId,
}: {
  isOpen: boolean;
  onToggle: () => void;
  onNewChat: () => void;
  currentSessionId: string;
}) {
  return (
    <aside
      style={{
        width: isOpen ? 260 : 0,
        minWidth: isOpen ? 260 : 0,
        transition: "width 0.2s ease, min-width 0.2s ease",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        background: "#f9f9f9",
        borderRight: "1px solid #e5e5e5",
        height: "100vh",
      }}
    >
      <div style={{ padding: "12px", display: "flex", alignItems: "center", gap: 8 }}>
        <button
          onClick={onToggle}
          style={{
            padding: 8,
            borderRadius: 8,
            border: "none",
            background: "transparent",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#666",
          }}
          title="收起侧边栏"
        >
          <PanelLeftClose size={18} />
        </button>
        <button
          onClick={onNewChat}
          style={{
            flex: 1,
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #e5e5e5",
            background: "#fff",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontSize: 14,
            color: "#1a1a1a",
          }}
        >
          <Plus size={16} />
          新对话
        </button>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "0 12px" }}>
        <div
          style={{
            padding: "10px 12px",
            borderRadius: 8,
            background: "#ececf1",
            fontSize: 14,
            color: "#1a1a1a",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginBottom: 4,
          }}
        >
          <MessageSquare size={16} />
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {currentSessionId ? currentSessionId.slice(0, 16) : "当前会话"}
          </span>
        </div>
      </div>

      <div style={{ padding: 12, borderTop: "1px solid #e5e5e5" }}>
        <button
          style={{
            width: "100%",
            padding: "10px 12px",
            borderRadius: 8,
            border: "none",
            background: "transparent",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 10,
            fontSize: 14,
            color: "#666",
          }}
        >
          <Settings size={16} />
          设置
        </button>
      </div>
    </aside>
  );
}

/* ------------------------------------------------------------------ */
/*  Markdown-like simple renderer                                     */
/* ------------------------------------------------------------------ */
function ChatMessageContent({ content }: { content: string }) {
  // Simple newline-to-line rendering (avoid <p> nesting issues)
  const lines = content.split("\n");
  return (
    <div style={{ lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
      {lines.map((line, i) => (
        <div key={i} style={{ margin: "2px 0", minHeight: "1.2em" }}>
          {line || " "}
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  App                                                               */
/* ------------------------------------------------------------------ */
export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<ConnStatus>("connecting");
  const [sessionId, setSessionId] = useState<string>("");
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [copiedId, setCopiedId] = useState<number | null>(null);
  const [hoveredMsgId, setHoveredMsgId] = useState<number | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryCountRef = useRef(0);
  const shouldReconnectRef = useRef(true);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const pushMessage = (msg: Msg) => {
    setMessages((prev) => {
      const next = [...prev, msg];
      if (next.length > MAX_MESSAGES) {
        return next.slice(next.length - MAX_MESSAGES);
      }
      return next;
    });
  };

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

      if (frame.type === "ping" || frame.type === "pong") return;

      if (frame.type === "ready") {
        const sid = frame.session_id;
        setSessionId(sid);
        localStorage.setItem(SESSION_KEY, sid);
        fetch(`/api/sessions/${encodeURIComponent(sid)}/history`)
          .then((r) => (r.ok ? r.json() : { messages: [] }))
          .then((j: { messages: { role: Role; content: string }[] }) => {
            const seed = (j.messages ?? []).filter(
              (m) => m.role === "user" || m.role === "assistant"
            );
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
        pushMessage({
          role: "system",
          kind: "tool",
          content: `调用 ${frame.name}(${summarize(frame.arguments, 160)})`,
        });
      } else if (frame.type === "tool_result") {
        const isErr = frame.error !== undefined && frame.error !== null;
        const label = isErr ? `${frame.name} 失败` : `${frame.name} 返回`;
        const body = summarize(isErr ? frame.error : frame.result, 320);
        pushMessage({ role: "system", kind: "tool", content: `${label}: ${body}` });
      } else if (frame.type === "notify") {
        pushMessage({ role: "system", kind: "notify", content: frame.message });
      } else if (frame.type === "done") {
        setStatus("ready");
      } else if (frame.type === "error") {
        pushMessage({ role: "assistant", content: `[error] ${frame.message}`, kind: "text" });
        setStatus("ready");
      }
    };

    ws.onclose = () => {
      if (pingTimerRef.current) {
        clearInterval(pingTimerRef.current);
        pingTimerRef.current = null;
      }
      if (wsRef.current !== ws) return;
      setStatus("disconnected");
      wsRef.current = null;

      if (!shouldReconnectRef.current) return;

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
        try {
          wsRef.current.close();
        } catch {
          /* ignore */
        }
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
    pushMessage({ role: "user", content: text, kind: "text" });
    setInput("");
    setStatus("streaming");
    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const newSession = () => {
    localStorage.removeItem(SESSION_KEY);
    location.reload();
  };

  const copyMessage = (idx: number, text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(idx);
      setTimeout(() => setCopiedId(null), 2000);
    });
  };

  const isEmpty = messages.length === 0;

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif", background: "#fff" }}>
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen((s) => !s)}
        onNewChat={newSession}
        currentSessionId={sessionId}
      />

      {/* Main */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", position: "relative", overflow: "hidden" }}>
        {/* Top Nav */}
        <header
          style={{
            height: 48,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderBottom: "1px solid #e5e5e5",
            padding: "0 16px",
            position: "relative",
          }}
        >
          {!sidebarOpen && (
            <button
              onClick={() => setSidebarOpen(true)}
              style={{
                position: "absolute",
                left: 12,
                top: "50%",
                transform: "translateY(-50%)",
                padding: 6,
                borderRadius: 6,
                border: "none",
                background: "transparent",
                cursor: "pointer",
                color: "#666",
                display: "flex",
                alignItems: "center",
              }}
            >
              <PanelLeftOpen size={18} />
            </button>
          )}

          <button
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 14,
              fontWeight: 500,
              color: "#1a1a1a",
              background: "transparent",
              border: "none",
              cursor: "pointer",
              padding: "4px 8px",
              borderRadius: 6,
            }}
          >
            PRTS Agent
            <ChevronDown size={14} />
          </button>

          {/* Connection status dot */}
          <div
            style={{
              position: "absolute",
              right: 16,
              top: "50%",
              transform: "translateY(-50%)",
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12,
              color: status === "ready" ? "#10a37f" : status === "streaming" ? "#f59e0b" : "#ef4444",
            }}
          >
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: status === "ready" ? "#10a37f" : status === "streaming" ? "#f59e0b" : "#ef4444",
              }}
            />
            {status === "ready"
              ? "就绪"
              : status === "streaming"
              ? "生成中…"
              : status === "reconnecting"
              ? `重连中 (${reconnectAttempt})`
              : status === "connecting"
              ? "连接中"
              : "已断开"}
          </div>
        </header>

        {/* Chat Area */}
        <div
          ref={listRef}
          style={{
            flex: 1,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {isEmpty ? (
            /* Empty state */
            <div
              style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                padding: "0 20px 120px",
              }}
            >
              <h1
                style={{
                  fontSize: 32,
                  fontWeight: 600,
                  color: "#1a1a1a",
                  marginBottom: 40,
                  textAlign: "center",
                }}
              >
                你好，我是PRTS。
              </h1>

              {/* Suggestion chips */}
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 12,
                  justifyContent: "center",
                  maxWidth: 600,
                  marginBottom: 40,
                }}
              >
                {[
                  "帮我写一段 Python 代码",
                  "解释一下量子计算",
                  "今天天气怎么样",
                  "翻译这段文字",
                ].map((s) => (
                  <button
                    key={s}
                    onClick={() => {
                      setInput(s);
                      textareaRef.current?.focus();
                    }}
                    style={{
                      padding: "10px 16px",
                      borderRadius: 12,
                      border: "1px solid #e5e5e5",
                      background: "#fff",
                      fontSize: 14,
                      color: "#666",
                      cursor: "pointer",
                      transition: "all 0.15s",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = "#d1d1d1";
                      e.currentTarget.style.background = "#fafafa";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = "#e5e5e5";
                      e.currentTarget.style.background = "#fff";
                    }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Messages */
            <div style={{ flex: 1, padding: "20px 0" }}>
              {messages.map((m, i) => {
                const isUser = m.role === "user";
                const isSystem = m.role === "system";

                if (isSystem) {
                  return (
                    <div
                      key={i}
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        padding: "8px 20px",
                      }}
                    >
                      <div
                        style={{
                          padding: "8px 16px",
                          borderRadius: 8,
                          background: m.kind === "tool" ? "#f0f9ff" : "#fef3c7",
                          border: `1px solid ${m.kind === "tool" ? "#bae6fd" : "#fde68a"}`,
                          fontSize: 13,
                          color: "#666",
                          maxWidth: "80%",
                          display: "flex",
                          alignItems: "center",
                          gap: 6,
                        }}
                      >
                        {m.kind === "tool" ? (
                          <RotateCcw size={14} style={{ color: "#0284c7" }} />
                        ) : (
                          <MessageSquare size={14} style={{ color: "#d97706" }} />
                        )}
                        {m.content}
                      </div>
                    </div>
                  );
                }

                return (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      justifyContent: isUser ? "flex-end" : "flex-start",
                      padding: "12px 20px",
                    }}
                  >
                    <div
                      style={{
                        maxWidth: "80%",
                        display: "flex",
                        gap: 12,
                        flexDirection: isUser ? "row-reverse" : "row",
                      }}
                    >
                      {/* Avatar */}
                      <div
                        style={{
                          width: 28,
                          height: 28,
                          borderRadius: "50%",
                          background: isUser ? "#10a37f" : "#e5e5e5",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          flexShrink: 0,
                          fontSize: 12,
                          fontWeight: 600,
                          color: isUser ? "#fff" : "#666",
                        }}
                      >
                        {isUser ? "你" : "AI"}
                      </div>

                      {/* Bubble */}
                      <div
                        style={{
                          background: isUser ? "#f5f5f5" : "#fff",
                          border: isUser ? "none" : "1px solid #e5e5e5",
                          borderRadius: 16,
                          padding: "12px 16px",
                          position: "relative",
                        }}
                        onMouseEnter={() => setHoveredMsgId(i)}
                        onMouseLeave={() => setHoveredMsgId(null)}
                      >
                        <ChatMessageContent content={m.content} />

                        {/* Actions */}
                        <div
                          style={{
                            display: "flex",
                            gap: 8,
                            marginTop: 8,
                            opacity: hoveredMsgId === i ? 1 : 0,
                            transition: "opacity 0.15s",
                          }}
                        >
                          <button
                            onClick={() => copyMessage(i, m.content)}
                            style={{
                              padding: "4px 8px",
                              borderRadius: 6,
                              border: "1px solid #e5e5e5",
                              background: "#fff",
                              cursor: "pointer",
                              display: "flex",
                              alignItems: "center",
                              gap: 4,
                              fontSize: 12,
                              color: "#666",
                            }}
                            title="复制"
                          >
                            {copiedId === i ? <Check size={12} /> : <Copy size={12} />}
                            {copiedId === i ? "已复制" : "复制"}
                          </button>
                          {!isUser && (
                            <button
                              onClick={() => {
                                /* regenerate - would need backend support */
                              }}
                              style={{
                                padding: "4px 8px",
                                borderRadius: 6,
                                border: "1px solid #e5e5e5",
                                background: "#fff",
                                cursor: "pointer",
                                display: "flex",
                                alignItems: "center",
                                gap: 4,
                                fontSize: 12,
                                color: "#666",
                              }}
                              title="重新生成"
                            >
                              <RotateCcw size={12} />
                              重新生成
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Input Area */}
        <div
          style={{
            padding: "16px 20px 24px",
            background: "#fff",
            borderTop: isEmpty ? "none" : "1px solid #e5e5e5",
          }}
        >
          <form
            onSubmit={send}
            style={{
              maxWidth: 768,
              margin: "0 auto",
              position: "relative",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "flex-end",
                gap: 8,
                padding: "10px 14px",
                borderRadius: 20,
                border: "1px solid #e5e5e5",
                background: "#fff",
                boxShadow: "0 2px 6px rgba(0,0,0,0.05)",
              }}
            >
              <button
                type="button"
                style={{
                  padding: 6,
                  borderRadius: "50%",
                  border: "none",
                  background: "transparent",
                  cursor: "pointer",
                  color: "#666",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
                title="附件"
              >
                <Paperclip size={18} />
              </button>

              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  e.target.style.height = "auto";
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send(e);
                  }
                }}
                placeholder="输入以开始对话……"
                disabled={status !== "ready"}
                rows={1}
                style={{
                  flex: 1,
                  border: "none",
                  outline: "none",
                  background: "transparent",
                  fontSize: 15,
                  lineHeight: 1.5,
                  color: "#1a1a1a",
                  resize: "none",
                  maxHeight: 200,
                  padding: "4px 0",
                }}
              />

              <button
                type="button"
                style={{
                  padding: 6,
                  borderRadius: "50%",
                  border: "none",
                  background: "transparent",
                  cursor: "pointer",
                  color: "#666",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
                title="语音输入"
              >
                <Mic size={18} />
              </button>

              <button
                type="submit"
                disabled={status !== "ready" || !input.trim()}
                style={{
                  padding: "6px 8px",
                  borderRadius: "50%",
                  border: "none",
                  background: status === "ready" && input.trim() ? "#1a1a1a" : "#e5e5e5",
                  color: "#fff",
                  cursor: status === "ready" && input.trim() ? "pointer" : "not-allowed",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transition: "background 0.15s",
                }}
              >
                <ArrowUp size={18} />
              </button>
            </div>

            <div
              style={{
                textAlign: "center",
                marginTop: 8,
                fontSize: 12,
                color: "#999",
              }}
            >
              PRTS生成的内容仅供参考，请自行验证。
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
