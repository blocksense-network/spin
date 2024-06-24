//! The Spin runtime running in the same process as the test

use anyhow::Context as _;
use test_environment::{
    http::{Request, Response},
    services::ServicesConfig,
    Runtime, TestEnvironment, TestEnvironmentConfig,
};

/// An instance of Spin running in the same process as the tests instead of as a separate process
///
/// Use `runtimes::spin_cli::SpinCli` if you'd rather use Spin as a separate process
pub struct InProcessSpin {
    trigger: spin_trigger_http::HttpTrigger,
}

impl InProcessSpin {
    /// Configure a new instance of Spin running in the same process as the tests
    pub fn config(
        services_config: ServicesConfig,
        preboot: impl FnOnce(&mut TestEnvironment<InProcessSpin>) -> anyhow::Result<()> + 'static,
    ) -> TestEnvironmentConfig<Self> {
        TestEnvironmentConfig {
            services_config,
            create_runtime: Box::new(|env| {
                preboot(env)?;
                tokio::runtime::Runtime::new()
                    .context("failed to start tokio runtime")?
                    .block_on(async { initialize_trigger(env).await })
            }),
        }
    }

    /// Create a new instance of Spin running in the same process as the tests
    pub fn new(trigger: spin_trigger_http::HttpTrigger) -> Self {
        Self { trigger }
    }

    /// Make an HTTP request to the Spin instance
    pub fn make_http_request(&self, req: Request<'_, &[u8]>) -> anyhow::Result<Response> {
        tokio::runtime::Runtime::new()?.block_on(async {
            let method: reqwest::Method = req.method.into();
            let req = http::request::Request::builder()
                .method(method)
                .uri(req.path)
                // TODO(rylev): convert headers and body as well
                .body(spin_http::body::empty())
                .unwrap();
            let response = self
                .trigger
                .handle(
                    req,
                    http::uri::Scheme::HTTP,
                    (std::net::Ipv4Addr::LOCALHOST, 3000).into(),
                    (std::net::Ipv4Addr::LOCALHOST, 7000).into(),
                )
                .await?;
            use http_body_util::BodyExt;
            let status = response.status().as_u16();
            let body = response.into_body();
            let chunks = body
                .collect()
                .await
                .context("could not get runtime test HTTP response")?
                .to_bytes()
                .to_vec();
            Ok(Response::full(status, Default::default(), chunks))
        })
    }
}

impl Runtime for InProcessSpin {
    fn error(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Initialize the trigger for the Spin instance inside the environment
async fn initialize_trigger(
    env: &mut TestEnvironment<InProcessSpin>,
) -> anyhow::Result<InProcessSpin> {
    use spin_trigger::{
        loader::TriggerLoader, HostComponentInitData, RuntimeConfig, TriggerExecutorBuilder,
    };
    use spin_trigger_http::HttpTrigger;

    // Create the locked app and write it to a file
    let locked_app = spin_loader::from_file(
        env.path().join("spin.toml"),
        spin_loader::FilesMountStrategy::Direct,
        None,
    )
    .await?;
    let json = locked_app.to_json()?;
    std::fs::write(env.path().join("locked.json"), json)?;

    // Create a loader and trigger builder
    let loader = TriggerLoader::new(env.path().join(".working_dir"), false);
    let mut builder = TriggerExecutorBuilder::<HttpTrigger>::new(loader);
    builder.hooks(spin_trigger::network::Network::default());

    // Build the trigger
    let trigger = builder
        .build(
            format!("file:{}", env.path().join("locked.json").display()),
            RuntimeConfig::default(),
            HostComponentInitData::default(),
        )
        .await?;
    Ok(InProcessSpin::new(trigger))
}
