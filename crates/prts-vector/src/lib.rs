//! PRTS 向量检索引擎。
//!
//! 后端使用 sqlite-vec(编译进二进制,零外部依赖),
//! 通过 rmcp 暴露为 MCP server。

use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{Connection, ffi::sqlite3_auto_extension, params};
use serde_json::Value;

/// 把 `Vec<f32>` 转成 SQLite BLOB 接受的 `&[u8]`(零拷贝)。
fn vec_as_bytes(v: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            v.as_ptr() as *const u8,
            v.len() * std::mem::size_of::<f32>(),
        )
    }
}

/// 核心向量存储。
pub struct VectorStore {
    conn: Connection,
    dim: usize,
}

impl VectorStore {
    /// 打开(或创建)向量数据库。
    ///
    /// `dim` 是向量维度(如 1536 for OpenAI text-embedding-3-small)。
    /// sqlite-vec 的虚拟表必须在建表时声明维度,因此这里写死一张表,
    /// 而不是按集合动态建表 —— MVP 足够。
    pub fn open(path: &Path, dim: usize) -> Result<Self> {
        // 注册 sqlite-vec 为 SQLite auto-extension,这样每个新连接自动加载。
        unsafe {
            let init: unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut i8,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> i32 = std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
            sqlite3_auto_extension(Some(init));
        }

        let conn = Connection::open(path)
            .with_context(|| format!("open vec db {}", path.display()))?;

        // 建虚拟表:embedding 是 float[DIM] 向量列,+id / +payload 是普通文本列。
        let sql = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{dim}], +id TEXT, +payload TEXT)"
        );
        conn.execute(&sql, []).context("create vec0 virtual table")?;

        Ok(Self { conn, dim })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 插入或覆盖一条向量。
    ///
    /// `id` 是业务唯一键;`payload` 以 JSON 文本形式存到普通列(可选)。
    pub fn upsert(&self, id: &str, vec: &[f32], payload: Option<Value>) -> Result<()> {
        if vec.len() != self.dim {
            anyhow::bail!(
                "vector dim mismatch: expected {}, got {}",
                self.dim,
                vec.len()
            );
        }
        let blob = vec_as_bytes(vec);
        let payload_text = payload.map(|p| p.to_string());

        self.conn
            .execute(
                "INSERT OR REPLACE INTO vec_items(id, embedding, payload) VALUES (?, ?, ?)",
                params![id, blob, payload_text.as_deref().unwrap_or("")],
            )
            .context("vec upsert")?;
        Ok(())
    }

    /// 最近邻搜索。
    ///
    /// 返回 `[(id, distance, payload), ...]` 按距离升序(L2;对归一化向量等价于余弦距离)。
    pub fn search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<(String, f64, Option<String>)>> {
        if query_vec.len() != self.dim {
            anyhow::bail!(
                "query dim mismatch: expected {}, got {}",
                self.dim,
                query_vec.len()
            );
        }
        let blob = vec_as_bytes(query_vec);
        let limit = top_k as i64;

        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, distance, payload FROM vec_items \
                 WHERE embedding MATCH ? \
                 ORDER BY distance \
                 LIMIT ?",
            )
            .context("prepare search")?;

        let rows = stmt
            .query_map(params![blob, limit], |row| {
                let id: String = row.get(0)?;
                let distance: f64 = row.get(1)?;
                let payload: Option<String> = row.get(2)?;
                Ok((id, distance, payload))
            })
            .context("search query")?;

        let mut out = Vec::new();
        for r in rows {
            out.push(r.context("row decode")?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_and_search() {
        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("test.db");
        let store = VectorStore::open(&db, 4).unwrap();

        store
            .upsert("a", &[1.0, 0.0, 0.0, 0.0], Some(serde_json::json!({"tag": "a"})))
            .unwrap();
        store.upsert("b", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        store.upsert("c", &[0.0, 0.0, 1.0, 0.0], None).unwrap();

        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 < results[1].1);
        assert_eq!(
            results[0].2,
            Some(r#"{"tag":"a"}"#.to_string()),
            "payload 应被写回"
        );
    }
}
