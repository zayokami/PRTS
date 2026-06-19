import { memo, useCallback, useEffect, useRef, useState, type FormEvent } from "react";
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
  Trash2,
  Pencil,
} from "lucide-react";

type Role = "user" | "assistant" | "system";
interface Msg {
  role: Role;
  content: string;
  kind?: "text" | "tool" | "notify";
  ts: number;
}

interface SessionInfo {
  id: string;
  channel: string;
  user_ref: string | null;
  title: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
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

function relativeTime(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  const now = Date.now();
  const diff = now - d.getTime();
  const min = Math.floor(diff / 60000);
  if (min < 1) return "刚刚";
  if (min < 60) return `${min} 分钟前`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr} 小时前`;
  const day = Math.floor(hr / 24);
  if (day < 30) return `${day} 天前`;
  return d.toLocaleDateString("zh-CN");
}

/* ------------------------------------------------------------------ */
/*  Sidebar                                                           */
/* ------------------------------------------------------------------ */
function Sidebar({
  isOpen,
  onToggle,
  onNewChat,
  sessions,
  currentSessionId,
  onSwitchSession,
  onDeleteSession,
  onRenameSession,
}: {
  isOpen: boolean;
  onToggle: () => void;
  onNewChat: () => void;
  sessions: SessionInfo[];
  currentSessionId: string;
  onSwitchSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, title: string) => void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const startEdit = (s: SessionInfo) => {
    setEditingId(s.id);
    setEditTitle(s.title || s.id.slice(0, 8));
  };

  const commitEdit = () => {
    if (editingId && editTitle.trim()) {
      onRenameSession(editingId, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle("");
  };

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
        {sessions.length === 0 && (
          <div style={{ padding: "20px 12px", fontSize: 13, color: "#999", textAlign: "center" }}>
            暂无历史会话
          </div>
        )}
        {sessions.map((s) => {
          const isCurrent = s.id === currentSessionId;
          const isEditing = editingId === s.id;
          return (
            <div
              key={s.id}
              onMouseEnter={() => setHoveredId(s.id)}
              onMouseLeave={() => setHoveredId(null)}
              onClick={() => !isEditing && onSwitchSession(s.id)}
              style={{
                padding: "10px 12px",
                borderRadius: 8,
                background: isCurrent ? "#ececf1" : "transparent",
                fontSize: 14,
                color: "#1a1a1a",
                cursor: isEditing ? "default" : "pointer",
                display: "flex",
                alignItems: "center",
                gap: 10,
                marginBottom: 2,
                position: "relative",
                transition: "background 0.1s",
              }}
            >
              <MessageSquare size={16} style={{ flexShrink: 0, opacity: 0.6 }} />
              {isEditing ? (
                <input
                  autoFocus
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onBlur={commitEdit}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitEdit();
                    if (e.key === "Escape") {
                      setEditingId(null);
                      setEditTitle("");
                    }
                  }}
                  onClick={(e) => e.stopPropagation()}
                  style={{
                    flex: 1,
                    border: "1px solid #10a37f",
                    borderRadius: 4,
                    padding: "2px 6px",
                    fontSize: 13,
                    background: "#fff",
                    color: "#1a1a1a",
                    outline: "none",
                  }}
                />
              ) : (
                <span
                  style={{
                    flex: 1,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    startEdit(s);
                  }}
                >
                  {s.title || s.id.slice(0, 16)}
                </span>
              )}

              {!isEditing && hoveredId === s.id && (
                <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      startEdit(s);
                    }}
                    style={{
                      padding: 2,
                      border: "none",
                      background: "transparent",
                      cursor: "pointer",
                      color: "#999",
                      display: "flex",
                    }}
                    title="重命名"
                  >
                    <Pencil size={13} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm(`删除会话「${s.title || s.id.slice(0, 16)}」？`)) {
                        onDeleteSession(s.id);
                      }
                    }}
                    style={{
                      padding: 2,
                      border: "none",
                      background: "transparent",
                      cursor: "pointer",
                      color: "#999",
                      display: "flex",
                    }}
                    title="删除"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              )}

              {!isEditing && hoveredId !== s.id && (
                <span style={{ fontSize: 11, color: "#999", flexShrink: 0 }}>
                  {relativeTime(s.updated_at)}
                </span>
              )}
            </div>
          );
        })}
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
/*  Message content renderer                                          */
/* ------------------------------------------------------------------ */
const ChatMessageContent = memo(function ChatMessageContent({ content }: { content: string }) {
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
});

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
  const [sessions, setSessions] = useState<SessionInfo[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryCountRef = useRef(0);
  const shouldReconnectRef = useRef(true);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const historyLoadingRef = useRef(false);

  const fetchSessions = useCallback(async () => {
    try {
      const r = await fetch("/api/sessions?limit=50");
      if (r.ok) {
        const j = await r.json();
        setSessions(j.sessions || []);
      }
    } catch {
      /* ignore */
    }
  }, []);

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
        setMessages([]);
        historyLoadingRef.current = true;
        fetch(`/api/sessions/${encodeURIComponent(sid)}/history`)
          .then((r) => (r.ok ? r.json() : { messages: [] }))
          .then((j: { messages: { role: Role; content: string }[] }) => {
            const seed = (j.messages ?? []).filter(
              (m) => m.role === "user" || m.role === "assistant"
            );
            setMessages(seed.map((m) => ({ role: m.role, content: m.content, kind: "text", ts: Date.now() })));
            historyLoadingRef.current = false;
            setStatus("ready");
            fetchSessions();
          })
          .catch(() => {
            historyLoadingRef.current = false;
            setStatus("ready");
          });
      } else if (frame.type === "token") {
        if (historyLoadingRef.current) return;
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === "assistant" && last.kind === "text") {
            next[next.length - 1] = { ...last, content: last.content + frame.text };
          } else {
            next.push({ role: "assistant", content: frame.text, kind: "text", ts: Date.now() });
          }
          return next;
        });
      } else if (frame.type === "tool_call") {
        pushMessage({
          role: "system",
          kind: "tool",
          content: `调用 ${frame.name}(${summarize(frame.arguments, 160)})`,
          ts: Date.now(),
        });
      } else if (frame.type === "tool_result") {
        const isErr = frame.error !== undefined && frame.error !== null;
        const label = isErr ? `${frame.name} 失败` : `${frame.name} 返回`;
        const body = summarize(isErr ? frame.error : frame.result, 320);
        pushMessage({ role: "system", kind: "tool", content: `${label}: ${body}`, ts: Date.now() });
      } else if (frame.type === "notify") {
        pushMessage({ role: "system", kind: "notify", content: frame.message, ts: Date.now() });
      } else if (frame.type === "done") {
        setStatus("ready");
        fetchSessions();
      } else if (frame.type === "error") {
        pushMessage({ role: "assistant", content: `[error] ${frame.message}`, kind: "text", ts: Date.now() });
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
    fetchSessions();
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
    pushMessage({ role: "user", content: text, kind: "text", ts: Date.now() });
    setInput("");
    setStatus("streaming");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const newSession = () => {
    localStorage.removeItem(SESSION_KEY);
    setMessages([]);
    setSessionId("");
    shouldReconnectRef.current = false;
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore */ }
      wsRef.current = null;
    }
    shouldReconnectRef.current = true;
    connect();
  };

  const switchSession = (id: string) => {
    if (id === sessionId) return;
    localStorage.setItem(SESSION_KEY, id);
    setMessages([]);
    setSessionId(id);
    shouldReconnectRef.current = false;
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore */ }
      wsRef.current = null;
    }
    shouldReconnectRef.current = true;
    setStatus("connecting");
    connect();
  };

  const deleteSession = async (id: string) => {
    try {
      await fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (id === sessionId) {
        newSession();
      }
      fetchSessions();
    } catch {
      /* ignore */
    }
  };

  const renameSession = async (id: string, title: string) => {
    try {
      await fetch(`/api/sessions/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ title }),
      });
      fetchSessions();
    } catch {
      /* ignore */
    }
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
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen((s) => !s)}
        onNewChat={newSession}
        sessions={sessions}
        currentSessionId={sessionId}
        onSwitchSession={switchSession}
        onDeleteSession={deleteSession}
        onRenameSession={renameSession}
      />

      <main style={{ flex: 1, display: "flex", flexDirection: "column", position: "relative", overflow: "hidden" }}>
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
                你今天在想些什么？
              </h1>

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
            <div style={{ flex: 1, padding: "20px 0" }}>
              {messages.map((m, i) => {
                const isUser = m.role === "user";
                const isSystem = m.role === "system";
                const msgKey = `${m.role}-${m.ts}-${i}`;

                if (isSystem) {
                  return (
                    <div
                      key={msgKey}
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
                    key={msgKey}
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
                placeholder="有问题，尽管问"
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
              PRTS 可能会生成不准确的信息，请验证重要信息。
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
