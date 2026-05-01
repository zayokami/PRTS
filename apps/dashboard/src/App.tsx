import { useEffect, useRef, useState, type FormEvent } from "react";

type Role = "user" | "assistant";
interface Msg {
  role: Role;
  content: string;
}

type Frame =
  | { type: "ready"; session_id: string }
  | { type: "token"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<"connecting" | "ready" | "streaming" | "closed">("connecting");
  const [sessionId, setSessionId] = useState<string>("");
  const wsRef = useRef<WebSocket | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/chat`);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      let frame: Frame;
      try { frame = JSON.parse(ev.data); } catch { return; }

      if (frame.type === "ready") {
        setSessionId(frame.session_id);
        setStatus("ready");
      } else if (frame.type === "token") {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === "assistant") {
            next[next.length - 1] = { ...last, content: last.content + frame.text };
          } else {
            next.push({ role: "assistant", content: frame.text });
          }
          return next;
        });
      } else if (frame.type === "done") {
        setStatus("ready");
      } else if (frame.type === "error") {
        setMessages((prev) => [...prev, { role: "assistant", content: `[error] ${frame.message}` }]);
        setStatus("ready");
      }
    };
    ws.onclose = () => setStatus("closed");
    ws.onerror = () => setStatus("closed");

    return () => ws.close();
  }, []);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [messages]);

  const send = (e: FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || status !== "ready") return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    ws.send(JSON.stringify({ type: "user", content: text }));
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setStatus("streaming");
  };

  return (
    <main style={S.main}>
      <header style={S.header}>
        <h1 style={S.title}>PRTS</h1>
        <span style={S.meta}>
          {status} {sessionId && `· ${sessionId.slice(0, 8)}`}
        </span>
      </header>

      <div ref={listRef} style={S.list}>
        {messages.length === 0 && <div style={S.hint}>博士,有什么需要协助的?</div>}
        {messages.map((m, i) => (
          <div key={i} style={{ ...S.row, ...(m.role === "user" ? S.rowUser : S.rowAssistant) }}>
            <span style={S.role}>{m.role === "user" ? "博士" : "PRTS"}</span>
            <pre style={S.content}>{m.content}</pre>
          </div>
        ))}
      </div>

      <form onSubmit={send} style={S.form}>
        <input
          autoFocus
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={status === "ready" ? "输入消息,回车发送" : status}
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
  main: { display: "flex", flexDirection: "column", height: "100vh", fontFamily: "ui-monospace, monospace", background: "#0d1117", color: "#e6edf3" },
  header: { display: "flex", alignItems: "baseline", gap: 12, padding: "12px 20px", borderBottom: "1px solid #30363d" },
  title: { margin: 0, fontSize: 18, letterSpacing: 2 },
  meta: { fontSize: 12, opacity: 0.6 },
  list: { flex: 1, overflowY: "auto", padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 },
  hint: { opacity: 0.4, textAlign: "center", marginTop: 40 },
  row: { display: "flex", flexDirection: "column", gap: 4, maxWidth: "80ch" },
  rowUser: { alignSelf: "flex-end", alignItems: "flex-end" },
  rowAssistant: { alignSelf: "flex-start" },
  role: { fontSize: 11, opacity: 0.5 },
  content: { margin: 0, padding: "8px 12px", background: "#161b22", border: "1px solid #30363d", borderRadius: 6, whiteSpace: "pre-wrap", wordBreak: "break-word", fontFamily: "inherit", fontSize: 14 },
  form: { display: "flex", gap: 8, padding: "12px 20px", borderTop: "1px solid #30363d" },
  input: { flex: 1, padding: "8px 12px", background: "#161b22", color: "inherit", border: "1px solid #30363d", borderRadius: 6, fontFamily: "inherit", fontSize: 14 },
  btn: { padding: "8px 16px", background: "#238636", color: "white", border: 0, borderRadius: 6, cursor: "pointer", fontFamily: "inherit" },
};
