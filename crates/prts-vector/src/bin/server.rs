//! PRTS Vector MCP Server —— stdio transport.
//!
//! 启动方式(被 Agent MCPManager spawn):
//!     prts-vector --db ~/.prts/vector.db --dim 1536

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::Parser;
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::{ProtocolVersion, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::io::stdio,
};
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::info;

use prts_vector::VectorStore;

#[derive(Parser, Debug)]
#[command(name = "prts-vector", about = "PRTS vector search MCP server")]
struct Args {
    #[arg(long, default_value = "~/.prts/vector.db")]
    db: PathBuf,

    #[arg(long, default_value_t = 1536)]
    dim: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct UpsertParams {
    id: String,
    vector: Vec<f32>,
    payload: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchParams {
    query_vector: Vec<f32>,
    #[schemars(range(min = 1, max = 100))]
    top_k: u32,
}

#[derive(Clone)]
struct VectorServer {
    store: Arc<Mutex<VectorStore>>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl VectorServer {
    fn new(store: VectorStore) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Store or overwrite a vector by ID")]
    fn upsert(&self, Parameters(params): Parameters<UpsertParams>) -> String {
        let guard = self.store.lock().unwrap();
        match guard.upsert(&params.id, &params.vector, params.payload) {
            Ok(()) => serde_json::json!({"ok": true}).to_string(),
            Err(e) => serde_json::json!({"ok": false, "error": e.to_string()}).to_string(),
        }
    }

    #[tool(description = "Search nearest vectors by L2 distance")]
    fn search(&self, Parameters(params): Parameters<SearchParams>) -> String {
        let guard = self.store.lock().unwrap();
        match guard.search(&params.query_vector, params.top_k as usize) {
            Ok(rows) => {
                let results: Vec<_> = rows
                    .into_iter()
                    .map(|(id, distance, payload)| {
                        let mut obj = serde_json::json!({"id": id, "distance": distance});
                        if let Some(p) = payload {
                            obj["payload"] = serde_json::Value::String(p);
                        }
                        obj
                    })
                    .collect();
                serde_json::json!({"ok": true, "results": results}).to_string()
            }
            Err(e) => serde_json::json!({"ok": false, "error": e.to_string()}).to_string(),
        }
    }
}

#[tool_handler]
impl ServerHandler for VectorServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            instructions: Some(
                "PRTS vector store. Upsert vectors and search by L2 distance.".into(),
            ),
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let db_path = if let Some(rest) = args.db.to_str().and_then(|s| s.strip_prefix("~")) {
        dirs::home_dir()
            .expect("home dir")
            .join(rest)
    } else {
        args.db
    };

    info!("opening vec db={} dim={}", db_path.display(), args.dim);
    let store = VectorStore::open(&db_path, args.dim)?;
    let server = VectorServer::new(store);

    let service = server.serve(stdio()).await?;
    info!("prts-vector ready");
    service.waiting().await?;
    Ok(())
}
