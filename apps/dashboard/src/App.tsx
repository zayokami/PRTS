import { useEffect, useState } from "react";

export default function App() {
  const [gatewayHealth, setGatewayHealth] = useState<string>("…");

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((j) => setGatewayHealth(JSON.stringify(j)))
      .catch((e) => setGatewayHealth(`error: ${e}`));
  }, []);

  return (
    <main style={{ fontFamily: "monospace", padding: 24 }}>
      <h1>PRTS</h1>
      <p>博士,欢迎回来。</p>
      <pre>gateway /health: {gatewayHealth}</pre>
      <p style={{ opacity: 0.6 }}>P0 骨架 — 聊天 UI 在 P1 阶段实现。</p>
    </main>
  );
}
