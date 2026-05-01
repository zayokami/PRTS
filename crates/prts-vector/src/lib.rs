//! PRTS 向量检索引擎。P0 占位,P7 阶段实现。
//!
//! 计划:
//! - 后端使用 sqlite-vec(brute-force,100% 召回,zerocopy + AVX2/NEON)
//! - 通过 rmcp 暴露为独立 MCP server,Agent 经 stdio 连接

pub fn hello() -> &'static str {
    "prts-vector v0.1.0 (P0 stub)"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert!(hello().starts_with("prts-vector"));
    }
}
