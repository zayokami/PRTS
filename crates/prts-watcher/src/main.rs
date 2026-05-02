//! PRTS Watcher —— 文件监听 + cron 调度守护进程。
//!
//! 启动时从 Agent 拉取当前 @task 列表,然后:
//! - 用 notify 监听 workspace/skills/ 目录,文件变化时 POST /events/fs 触发热加载
//! - 用 tokio-cron-scheduler 按 cron 表达式调度,触发时 POST /events/cron 执行 task
//!
//! 用法:
//!     prts-watcher --workspace ~/.prts/workspace --agent http://127.0.0.1:4788

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use notify::{Event, RecursiveMode, Watcher};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(name = "prts-watcher", about = "PRTS file & cron watcher daemon")]
struct Args {
    /// 用户工作区目录(监听 skills/ 子目录)
    #[arg(long, default_value = "")]
    workspace: String,

    /// Agent HTTP 地址
    #[arg(long, default_value = "http://127.0.0.1:4788")]
    agent: String,

    /// 首次同步前等待 Agent 启动的秒数
    #[arg(long, default_value_t = 5)]
    startup_delay: u64,
}

/// Agent /tasks 端点返回的 task 信息。
#[derive(Debug, Clone, Deserialize)]
struct TaskInfo {
    name: String,
    cron: Option<String>,
    #[allow(dead_code)]
    on: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TasksResponse {
    tasks: Vec<TaskInfo>,
}

#[derive(Debug, Serialize)]
struct FsEventRequest {
    changed_files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FsEventResponse {
    reloaded: bool,
    tasks: Vec<TaskInfo>,
    #[allow(dead_code)]
    errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CronEventRequest {
    task_name: String,
}

#[derive(Debug, Deserialize)]
struct CronEventResponse {
    ok: bool,
    #[allow(dead_code)]
    result: Option<serde_json::Value>,
    #[allow(dead_code)]
    error: Option<String>,
}

/// 共享状态:当前已注册的 task 列表 + reqwest client。
struct WatcherState {
    client: Client,
    agent_url: String,
    tasks: Vec<TaskInfo>,
}

impl WatcherState {
    fn new(agent_url: String) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client build"),
            agent_url,
            tasks: Vec::new(),
        }
    }

    /// 从 Agent 拉取最新 task 列表。
    async fn sync_tasks(&mut self) -> Result<Vec<TaskInfo>> {
        let url = format!("{}/agent/v1/tasks", self.agent_url);
        let resp = self.client.get(&url).send().await.with_context(|| {
            format!("GET {url} failed")
        })?;
        if !resp.status().is_success() {
            anyhow::bail!("GET {url} returned {}", resp.status());
        }
        let body: TasksResponse = resp.json().await.context("decode /tasks response")?;
        self.tasks = body.tasks.clone();
        info!("synced {} task(s) from agent", body.tasks.len());
        Ok(body.tasks)
    }

    /// 通知 Agent 文件变化,触发 skill 热加载。
    async fn notify_fs_event(&self, changed: Vec<String>) -> Result<FsEventResponse> {
        let url = format!("{}/agent/v1/events/fs", self.agent_url);
        let resp = self
            .client
            .post(&url)
            .json(&FsEventRequest { changed_files: changed })
            .send()
            .await
            .with_context(|| format!("POST {url} failed"))?;
        if !resp.status().is_success() {
            anyhow::bail!("POST {url} returned {}", resp.status());
        }
        let body: FsEventResponse = resp.json().await.context("decode /events/fs response")?;
        Ok(body)
    }

    /// 触发指定 task 的 cron 执行。
    async fn trigger_cron(&self, task_name: &str) -> Result<()> {
        let url = format!("{}/agent/v1/events/cron", self.agent_url);
        let resp = self
            .client
            .post(&url)
            .json(&CronEventRequest {
                task_name: task_name.to_string(),
            })
            .send()
            .await
            .with_context(|| format!("POST {url} for task {task_name} failed"))?;
        if !resp.status().is_success() {
            anyhow::bail!("POST {url} returned {}", resp.status());
        }
        let body: CronEventResponse = resp.json().await.context("decode /events/cron response")?;
        if body.ok {
            info!("task {} executed successfully", task_name);
        } else {
            error!("task {} execution failed: {:?}", task_name, body.error);
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // workspace 目录解析
    let workspace: PathBuf = if args.workspace.is_empty() {
        dirs::home_dir()
            .context("cannot determine home dir")?
            .join(".prts")
            .join("workspace")
    } else {
        PathBuf::from(&args.workspace)
    };
    let skills_dir = workspace.join("skills");
    if !skills_dir.is_dir() {
        warn!("skills dir does not exist yet: {}", skills_dir.display());
    }

    // 等待 Agent 就绪
    if args.startup_delay > 0 {
        info!("waiting {}s for agent to boot...", args.startup_delay);
        tokio::time::sleep(Duration::from_secs(args.startup_delay)).await;
    }

    let state = Arc::new(Mutex::new(WatcherState::new(args.agent.clone())));

    // 首次同步 task 列表
    {
        let mut s = state.lock().await;
        if let Err(e) = s.sync_tasks().await {
            error!("initial task sync failed: {e:?}");
        }
    }

    // 启动 cron scheduler
    let scheduler = JobScheduler::new().await.context("create scheduler")?;
    sync_cron_jobs(&scheduler, state.clone()).await?;

    // 启动文件 watcher
    let fs_state = state.clone();
    let _watcher = spawn_fs_watcher(&skills_dir, fs_state).await?;

    // scheduler 需要显式启动后台轮询
    scheduler.start().await?;
    info!("prts-watcher ready | workspace={} agent={}", workspace.display(), args.agent);

    // 保持进程存活
    tokio::signal::ctrl_c().await?;
    info!("shutting down");
    Ok(())
}

/// 根据当前 task 列表创建 / 更新 cron jobs。
async fn sync_cron_jobs(
    scheduler: &JobScheduler,
    state: Arc<Mutex<WatcherState>>,
) -> Result<()> {
    let tasks = {
        let s = state.lock().await;
        s.tasks.clone()
    };

    for t in tasks {
        let Some(cron_expr) = t.cron else { continue };
        let name = t.name.clone();
        let st = state.clone();

        let job = Job::new_async(cron_expr.as_str(), move |_uuid, _l| {
            let task_name = name.clone();
            let st2 = st.clone();
            Box::pin(async move {
                let guard = st2.lock().await;
                if let Err(e) = guard.trigger_cron(&task_name).await {
                    error!("cron trigger for {} failed: {e:?}", task_name);
                }
            })
        });

        match job {
            Ok(j) => {
                scheduler.add(j).await?;
                info!("registered cron job: {} ('{}')", t.name, cron_expr);
            }
            Err(e) => {
                error!("invalid cron expression for task {}: {e}", t.name);
            }
        }
    }
    Ok(())
}

/// 启动 notify 文件 watcher。
async fn spawn_fs_watcher(
    skills_dir: &Path,
    state: Arc<Mutex<WatcherState>>,
) -> Result<impl Watcher> {
    let dir = skills_dir.to_path_buf();
    let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
        let Ok(event) = res else { return };
        // 只关心 .py 文件的写入/创建/重命名
        let has_py = event.paths.iter().any(|p| {
            p.extension()
                .map(|e| e == "py")
                .unwrap_or(false)
        });
        if !has_py {
            return;
        }
        match event.kind {
            notify::EventKind::Create(_)
            | notify::EventKind::Modify(_)
            | notify::EventKind::Remove(_) => {
                let changed: Vec<String> = event
                    .paths
                    .iter()
                    .filter(|p| {
                        p.extension()
                            .map(|e| e == "py")
                            .unwrap_or(false)
                    })
                    .map(|p| p.to_string_lossy().to_string())
                    .collect();
                if changed.is_empty() {
                    return;
                }
                info!("detected file change: {:?}", changed);
                let st = state.clone();
                tokio::spawn(async move {
                    let guard = st.lock().await;
                    match guard.notify_fs_event(changed).await {
                        Ok(resp) => {
                            if resp.reloaded {
                                info!("skill reload succeeded, {} task(s) now", resp.tasks.len());
                            } else {
                                error!("skill reload failed: {:?}", resp.errors);
                            }
                        }
                        Err(e) => {
                            error!("fs event notify failed: {e:?}");
                        }
                    }
                });
            }
            _ => {}
        }
    })?;

    if dir.is_dir() {
        watcher.watch(&dir, RecursiveMode::Recursive)?;
    }
    Ok(watcher)
}
