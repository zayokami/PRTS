//! PRTS 文件 / cron 守护进程。P0 占位,P6 阶段实现。
//!
//! 计划:
//! - notify crate 监听 workspace 目录,事件 POST 给 Agent /agent/v1/events/fs
//! - tokio-cron-scheduler 触发 @task(cron=...) 注册的任务

pub fn hello() -> &'static str {
    "prts-watcher v0.1.0 (P0 stub)"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert!(hello().starts_with("prts-watcher"));
    }
}
