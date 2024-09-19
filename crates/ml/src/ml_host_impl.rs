use anyhow::{anyhow, Context};

use spin_world::v2 as ml_wit;

use ml_wit::errors::ErrorCode;
use ml_wit::graph::{ExecutionTarget, Graph, GraphBuilder, GraphEncoding};
use ml_wit::inference::GraphExecutionContext;
use ml_wit::{errors, graph, inference, tensor};

use spin_core::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;

use spin_core::wasmtime::component::Resource;

use crate::backend::{BackendExecutionContext, BackendGraph};
use crate::backend::{BackendInner, TensorId};
use crate::model_files::ModelFiles;

pub struct GraphInternalData(pub Box<dyn BackendGraph>);

pub struct TensorInternalData {
    pub tensor_dimensions: tensor::TensorDimensions,
    pub tensor_type: tensor::TensorType,
    pub tensor_data: tensor::TensorData,
}

pub struct ErrorInternalData {
    code: errors::ErrorCode,
    message: String,
}

pub struct ExecutionContext(pub Box<dyn BackendExecutionContext>);

#[derive(Default)]
pub struct MLHostImpl {
    pub state_dir: Option<PathBuf>,
    pub graphs: table::Table<GraphInternalData>,
    pub tensors: table::Table<TensorInternalData>,
    pub errors: table::Table<ErrorInternalData>,

    pub executions: table::Table<ExecutionContext>,
    pub backends: Vec<Box<dyn BackendInner>>,
    pub model_files: HashMap<String, ModelFiles>,
}

impl MLHostImpl {
    pub fn new_error(
        errors: &mut table::Table<ErrorInternalData>,
        code: ErrorCode,
        message: String,
    ) -> Resource<errors::Error> {
        errors
            .push(ErrorInternalData { code, message })
            .map(Resource::<errors::Error>::new_own)
            .expect("Can't allocate error")
    }

    fn new_graph(
        graphs: &mut table::Table<GraphInternalData>,
        errors: &mut table::Table<ErrorInternalData>,
        graph_internal_data: GraphInternalData,
    ) -> Result<Result<Resource<Graph>, Resource<errors::Error>>, anyhow::Error> {
        Ok(match graphs.push(graph_internal_data) {
            Ok(graph_rep) => Ok(Resource::<Graph>::new_own(graph_rep)),
            Err(err) => Err(MLHostImpl::new_error(
                errors,
                ErrorCode::RuntimeError,
                format!("{:?}", err),
            )),
        })
    }
}

#[async_trait]
impl graph::HostGraph for MLHostImpl {
    async fn init_execution_context(
        &mut self,
        graph: Resource<Graph>,
    ) -> Result<
        Result<Resource<inference::GraphExecutionContext>, Resource<errors::Error>>,
        anyhow::Error,
    > {
        if let Some(graph) = self.graphs.get_mut(graph.rep()) {
            match graph.0.init_execution_context() {
                Ok(execution_context) => {
                    return Ok(self
                        .executions
                        .push(execution_context)
                        .map(Resource::<inference::GraphExecutionContext>::new_own)
                        .map_err(|_| {
                            MLHostImpl::new_error(
                                &mut self.errors,
                                ErrorCode::RuntimeError,
                                "Can't create graph execution context".to_string(),
                            )
                        }));
                }
                Err(err) => {
                    return Ok(Err(MLHostImpl::new_error(
                        &mut self.errors,
                        ErrorCode::RuntimeError,
                        err.to_string(),
                    )));
                }
            }
        }
        Err(anyhow!(
            "[graph::HostGraph] fn init_execution_context -> Not implemented"
        ))
    }

    fn drop(&mut self, graph: Resource<Graph>) -> Result<(), anyhow::Error> {
        self.graphs
            .remove(graph.rep())
            .context(format!("Can't find graph with ID = {}", graph.rep()))?;
        Ok(())
    }
}

#[async_trait]
impl errors::HostError for MLHostImpl {
    async fn new(
        &mut self,
        code: errors::ErrorCode,
        data: String,
    ) -> Result<Resource<errors::Error>, anyhow::Error> {
        Ok(MLHostImpl::new_error(&mut self.errors, code, data))
    }

    fn drop(&mut self, error: Resource<errors::Error>) -> Result<(), anyhow::Error> {
        self.errors
            .remove(error.rep())
            .ok_or(anyhow!(format!(
                "Can't find error with ID = {}",
                error.rep()
            )))
            .map(|_| ())
    }

    async fn code(&mut self, error: Resource<errors::Error>) -> Result<ErrorCode, anyhow::Error> {
        self.errors
            .get(error.rep())
            .ok_or(anyhow!(format!(
                "Can't find error with ID = {}",
                error.rep()
            )))
            .map(|e| e.code)
    }

    async fn data(&mut self, error: Resource<errors::Error>) -> Result<String, anyhow::Error> {
        self.errors
            .get(error.rep())
            .ok_or(anyhow!(format!(
                "Can't find error with ID = {}",
                error.rep()
            )))
            .map(|e| e.message.clone())
    }
}

#[async_trait]
impl tensor::HostTensor for MLHostImpl {
    async fn new(
        &mut self,
        tensor_dimensions: tensor::TensorDimensions,
        tensor_type: tensor::TensorType,
        tensor_data: tensor::TensorData,
    ) -> Result<Resource<tensor::Tensor>, anyhow::Error> {
        let tensor = TensorInternalData {
            tensor_dimensions,
            tensor_type,
            tensor_data,
        };
        self.tensors
            .push(tensor)
            .map(Resource::<tensor::Tensor>::new_own)
            .map_err(|_| anyhow!("Can't allocate tensor"))
    }
    async fn dimensions(
        &mut self,
        tensor: Resource<tensor::Tensor>,
    ) -> Result<Vec<u32>, anyhow::Error> {
        self.tensors
            .get(tensor.rep())
            .ok_or(anyhow!(format!(
                "Can't find tensor with ID = {}",
                tensor.rep()
            )))
            .map(|t| t.tensor_dimensions.clone())
    }
    async fn ty(
        &mut self,
        tensor: Resource<tensor::Tensor>,
    ) -> Result<tensor::TensorType, anyhow::Error> {
        self.tensors
            .get(tensor.rep())
            .ok_or(anyhow!(format!(
                "Can't find tensor with ID = {}",
                tensor.rep()
            )))
            .map(|t| t.tensor_type)
    }
    async fn data(
        &mut self,
        tensor: Resource<tensor::Tensor>,
    ) -> Result<tensor::TensorData, anyhow::Error> {
        self.tensors
            .get(tensor.rep())
            .ok_or(anyhow!(format!(
                "Can't find tensor with ID = {}",
                tensor.rep()
            )))
            .map(|t| t.tensor_data.clone())
    }
    fn drop(&mut self, tensor: Resource<tensor::Tensor>) -> Result<(), anyhow::Error> {
        self.tensors
            .remove(tensor.rep())
            .context(format!("Can't find tensor with ID = {}", tensor.rep()))?;
        Ok(())
    }
}

#[async_trait]
impl inference::HostGraphExecutionContext for MLHostImpl {
    async fn set_input(
        &mut self,
        graph_execution_context: Resource<GraphExecutionContext>,
        input_name: String,
        tensor: Resource<tensor::Tensor>,
    ) -> Result<Result<(), Resource<errors::Error>>, anyhow::Error> {
        let execution_context = self
            .executions
            .get_mut(graph_execution_context.rep())
            .context(format!(
                "Can't find graph execution context with ID = {}",
                graph_execution_context.rep()
            ))?;

        let tensor = self
            .tensors
            .get(tensor.rep())
            .context(format!("Can't find tensor with ID = {}", tensor.rep()))?;

        let tensor_id = TensorId::new(&input_name);

        Ok(execution_context
            .0
            .set_input(&tensor_id, tensor)
            .map_err(|err| {
                MLHostImpl::new_error(&mut self.errors, ErrorCode::RuntimeError, err.to_string())
            }))
    }

    async fn compute(
        &mut self,
        graph_execution_context: Resource<GraphExecutionContext>,
    ) -> Result<Result<(), Resource<errors::Error>>, anyhow::Error> {
        let graph_execution = self
            .executions
            .get_mut(graph_execution_context.rep())
            .ok_or(anyhow!(format!(
                "Can't find graph execution context with ID = {}",
                graph_execution_context.rep()
            )))?;

        Ok(graph_execution.0.compute().map_err(|err| {
            MLHostImpl::new_error(
                &mut self.errors,
                ErrorCode::RuntimeError,
                format!("Inference error = {:?}", err.to_string()),
            )
        }))
    }

    async fn get_output(
        &mut self,
        graph_execution_context: Resource<GraphExecutionContext>,
        input_name: String,
    ) -> Result<Result<Resource<tensor::Tensor>, Resource<errors::Error>>, anyhow::Error> {
        let graph_execution = self
            .executions
            .get_mut(graph_execution_context.rep())
            .ok_or(anyhow!(format!(
                "Can't find graph execution context with ID = {}",
                graph_execution_context.rep()
            )))?;

        let tensor_id = TensorId::new(&input_name);

        let res = graph_execution.0.get_output(&tensor_id).map_err(|err| {
            MLHostImpl::new_error(&mut self.errors, ErrorCode::RuntimeError, err.to_string())
        });
        match res {
            Ok(tensor) => {
                match self
                    .tensors
                    .push(tensor)
                    .map(Resource::<tensor::Tensor>::new_own)
                {
                    Ok(t) => return Ok(Ok(t)),
                    Err(_) => {
                        return Ok(Err(self
                            .errors
                            .push(ErrorInternalData {
                                code: ErrorCode::RuntimeError,
                                message: "Can't create tensor for get_output".to_string(),
                            })
                            .map(Resource::<errors::Error>::new_own)
                            .map_err(|_| anyhow!("Can't allocate error"))?));
                    }
                }
            }
            Err(err) => {
                return Ok(Err(err));
            }
        }
    }

    fn drop(&mut self, execution: Resource<GraphExecutionContext>) -> Result<(), anyhow::Error> {
        let id = execution.rep();
        self.executions
            .remove(id)
            .context("{Can't drow GraphExecutionContext with id = {id}")?;
        Ok(())
    }
}

#[async_trait]
impl errors::Host for MLHostImpl {}

#[async_trait]
impl graph::Host for MLHostImpl {
    async fn load(
        &mut self,
        _builders: Vec<GraphBuilder>,
        graph_encoding: GraphEncoding,
        _target: ExecutionTarget,
    ) -> Result<Result<Resource<Graph>, Resource<errors::Error>>, anyhow::Error> {
        /*for backend in self.backends.iter_mut() {
            if backend.encoding() == graph_encoding {
                match backend.load(builders, target, graph_encoding, None) {
                    Ok(graph_internal_data) => {
                        return MLHostImpl::new_graph(
                            &mut self.graphs,
                            &mut self.errors,
                            graph_internal_data,
                        );
                    }
                    Err(err) => {
                        return Ok(Err(MLHostImpl::new_error(
                            &mut self.errors,
                            ErrorCode::RuntimeError,
                            format!("Can't load model error = {:?}", err),
                        )));
                    }
                }
            }
        }*/
        Err(anyhow!(
            "[graph::Host] fn load -> graph_encoding = {graph_encoding:?} is not supported "
        ))
    }

    async fn load_by_name(
        &mut self,
        model_name: String,
    ) -> Result<Result<Resource<Graph>, Resource<errors::Error>>, anyhow::Error> {
        let parts: Vec<_> = model_name.split(':').map(|x| x.to_string()).collect();
        let target = if parts.len() > 1 {
            if let Some(target) = map_string_to_execution_target(&parts[parts.len() - 1]) {
                target
            } else {
                ExecutionTarget::Cpu
            }
        } else {
            ExecutionTarget::Cpu
        };

        if parts.len() > 0 {
            let model_name = &parts[0];
            println!("Searching for model = `{model_name}`");
            if let Some(model) = self.model_files.get(model_name) {
                if let Some(backend) = self
                    .backends
                    .iter_mut()
                    .find(|b| b.encoding() == model.encoding)
                {
                    match backend.load(model, target) {
                        Ok(graph_internal_data) => {
                            return MLHostImpl::new_graph(
                                &mut self.graphs,
                                &mut self.errors,
                                graph_internal_data,
                            );
                        }
                        Err(err) => {
                            return Ok(Err(MLHostImpl::new_error(
                                &mut self.errors,
                                ErrorCode::RuntimeError,
                                format!("Can't load model '{model_name}' error = {err:?}"),
                            )));
                        }
                    }
                }
            }
        }
        Err(anyhow!(
            "[graph::Host] fn load_by_name -> model not supported "
        ))
    }

    async fn register_by_name(
        &mut self,
        model_name: String,
        encoding: GraphEncoding,
        files: Vec<String>,
        sources: Vec<Vec<String>>,
        hashes: Vec<String>,
    ) -> Result<Result<(), Resource<errors::Error>>, anyhow::Error> {
        println!("Registering model `{model_name}`");
        let _r = self.model_files.insert(
            model_name.clone(),
            ModelFiles {
                name: model_name.clone(),
                encoding,
                files,
                sources,
                hashes,
            },
        );
        Ok(Ok(()))
    }
}

impl inference::Host for MLHostImpl {}
impl tensor::Host for MLHostImpl {}

fn map_string_to_graph_encoding(target: &str) -> Option<GraphEncoding> {
    match target {
        "openvino" => Some(GraphEncoding::Openvino),
        "llm" => Some(GraphEncoding::Ggml),
        _ => None,
    }
}

/// Return the execution target string expected by OpenVINO from the
/// `ExecutionTarget` enum provided by wasi-nn.
fn map_string_to_execution_target(target: &str) -> Option<ExecutionTarget> {
    match target {
        "CPU" => Some(ExecutionTarget::Cpu),
        "GPU" => Some(ExecutionTarget::Gpu),
        "TPU" => Some(ExecutionTarget::Tpu),
        _ => None,
    }
}
