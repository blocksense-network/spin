use anyhow::{anyhow, Context};

use crate::test_host_components::ml_backend;
use crate::test_host_components::ml_wit::test::test as ml_wit;

use ml_wit::errors::ErrorCode;
use ml_wit::graph::{ExecutionTarget, Graph, GraphBuilder, GraphEncoding};
use ml_wit::inference::GraphExecutionContext;
use ml_wit::{errors, graph, inference, tensor};

use spin_core::wasmtime::component::Resource;

use ml_backend::{BackendExecutionContext, BackendGraph, BackendInner, TensorId};

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
    pub graphs: table::Table<GraphInternalData>,
    pub tensors: table::Table<TensorInternalData>,
    pub errors: table::Table<ErrorInternalData>,

    pub executions: table::Table<ExecutionContext>,
    pub backends: Vec<Box<dyn BackendInner>>,
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

impl graph::HostGraph for MLHostImpl {
    fn init_execution_context(
        &mut self,
        graph: Resource<Graph>,
    ) -> Result<Resource<inference::GraphExecutionContext>, Resource<errors::Error>> {
        if let Some(graph) = self.graphs.get_mut(graph.rep()) {
            match graph.0.init_execution_context() {
                Ok(execution_context) => {
                    return self
                        .executions
                        .push(execution_context)
                        .map(Resource::<inference::GraphExecutionContext>::new_own)
                        .map_err(|_| {
                            MLHostImpl::new_error(
                                &mut self.errors,
                                ErrorCode::RuntimeError,
                                "Can't create graph execution context".to_string(),
                            )
                        });
                }
                Err(err) => {
                    return Err(MLHostImpl::new_error(
                        &mut self.errors,
                        ErrorCode::RuntimeError,
                        err.to_string(),
                    ));
                }
            }
        }
        panic!("[graph::HostGraph] fn init_execution_context -> Not implemented")
    }

    fn drop(&mut self, graph: Resource<Graph>) -> Result<(), anyhow::Error> {
        self.graphs
            .remove(graph.rep())
            .context(format!("Can't find graph with ID = {}", graph.rep()))?;
        Ok(())
    }
}

impl errors::HostError for MLHostImpl {
    fn new(&mut self, code: errors::ErrorCode, data: String) -> Resource<errors::Error> {
        MLHostImpl::new_error(&mut self.errors, code, data)
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

    fn code(&mut self, error: Resource<errors::Error>) -> ErrorCode {
        self.errors.get(error.rep()).unwrap().code
    }

    fn data(&mut self, error: Resource<errors::Error>) -> String {
        self.errors.get(error.rep()).unwrap().message.clone()
    }
}

impl tensor::HostTensor for MLHostImpl {
    fn new(
        &mut self,
        tensor_dimensions: tensor::TensorDimensions,
        tensor_type: tensor::TensorType,
        tensor_data: tensor::TensorData,
    ) -> Resource<tensor::Tensor> {
        let tensor = TensorInternalData {
            tensor_dimensions,
            tensor_type,
            tensor_data,
        };
        self.tensors
            .push(tensor)
            .map(Resource::<tensor::Tensor>::new_own)
            .map_err(|_| anyhow!("Can't allocate tensor"))
            .unwrap()
    }
    fn dimensions(&mut self, tensor: Resource<tensor::Tensor>) -> Vec<u32> {
        self.tensors
            .get(tensor.rep())
            .ok_or(anyhow!(format!(
                "Can't find tensor with ID = {}",
                tensor.rep()
            )))
            .map(|t| t.tensor_dimensions.clone())
            .unwrap()
    }
    fn ty(&mut self, tensor: Resource<tensor::Tensor>) -> tensor::TensorType {
        self.tensors
            .get(tensor.rep())
            .ok_or(anyhow!(format!(
                "Can't find tensor with ID = {}",
                tensor.rep()
            )))
            .map(|t| t.tensor_type)
            .unwrap()
    }
    fn data(&mut self, tensor: Resource<tensor::Tensor>) -> tensor::TensorData {
        self.tensors
            .get(tensor.rep())
            .ok_or(anyhow!(format!(
                "Can't find tensor with ID = {}",
                tensor.rep()
            )))
            .map(|t| t.tensor_data.clone())
            .unwrap()
    }
    fn drop(&mut self, tensor: Resource<tensor::Tensor>) -> Result<(), anyhow::Error> {
        self.tensors
            .remove(tensor.rep())
            .context(format!("Can't find tensor with ID = {}", tensor.rep()))?;
        Ok(())
    }
}

impl inference::HostGraphExecutionContext for MLHostImpl {
    fn set_input(
        &mut self,
        graph_execution_context: Resource<GraphExecutionContext>,
        input_name: String,
        tensor: Resource<tensor::Tensor>,
    ) -> Result<(), Resource<errors::Error>> {
        let execution_context = self
            .executions
            .get_mut(graph_execution_context.rep())
            .context(format!(
                "Can't find graph execution context with ID = {}",
                graph_execution_context.rep()
            ))
            .unwrap();

        let tensor = self
            .tensors
            .get(tensor.rep())
            .context(format!("Can't find tensor with ID = {}", tensor.rep()))
            .unwrap();

        let index = input_name
            .parse::<usize>()
            .context("Can't parse {} to usize for input_name={input_name}")
            .unwrap();
        let tensor_id = TensorId::Index(index as u32);

        execution_context
            .0
            .set_input(&tensor_id, tensor)
            .map_err(|err| {
                MLHostImpl::new_error(&mut self.errors, ErrorCode::RuntimeError, err.to_string())
            })
    }

    fn compute(
        &mut self,
        graph_execution_context: Resource<GraphExecutionContext>,
    ) -> Result<(), Resource<errors::Error>> {
        let graph_execution = self
            .executions
            .get_mut(graph_execution_context.rep())
            .ok_or(anyhow!(format!(
                "Can't find graph execution context with ID = {}",
                graph_execution_context.rep()
            )))
            .unwrap();

        graph_execution.0.compute().map_err(|err| {
            MLHostImpl::new_error(
                &mut self.errors,
                ErrorCode::RuntimeError,
                format!("Inference error = {:?}", err.to_string()),
            )
        })
    }

    fn get_output(
        &mut self,
        graph_execution_context: Resource<GraphExecutionContext>,
        input_name: String,
    ) -> Result<Resource<tensor::Tensor>, Resource<errors::Error>> {
        let graph_execution = self
            .executions
            .get_mut(graph_execution_context.rep())
            .ok_or(anyhow!(format!(
                "Can't find graph execution context with ID = {}",
                graph_execution_context.rep()
            )))
            .unwrap();

        let index = input_name
            .parse::<usize>()
            .context("Can't parse {} to usize for input_name={input_name}")
            .unwrap();
        let tensor_id = TensorId::Index(index as u32);

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
                    Ok(t) => Ok(t),
                    Err(_) => Err(self
                        .errors
                        .push(ErrorInternalData {
                            code: ErrorCode::RuntimeError,
                            message: "Can't create tensor for get_output".to_string(),
                        })
                        .map(Resource::<errors::Error>::new_own)
                        .map_err(|_| anyhow!("Can't allocate error"))
                        .unwrap()),
                }
            }
            Err(err) => Err(err),
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

impl errors::Host for MLHostImpl {}

impl graph::Host for MLHostImpl {
    fn load(
        &mut self,
        builders: Vec<GraphBuilder>,
        graph_encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Resource<Graph>, Resource<errors::Error>> {
        for backend in self.backends.iter_mut() {
            if backend.encoding() == graph_encoding {
                match backend.load(builders, target, graph_encoding, None) {
                    Ok(graph_internal_data) => {
                        return MLHostImpl::new_graph(
                            &mut self.graphs,
                            &mut self.errors,
                            graph_internal_data,
                        )
                        .unwrap();
                    }
                    Err(err) => {
                        return Err(MLHostImpl::new_error(
                            &mut self.errors,
                            ErrorCode::RuntimeError,
                            format!("Can't load model error = {:?}", err),
                        ));
                    }
                }
            }
        }
        panic!("[graph::Host] fn load -> graph_encoding = {graph_encoding:?} is not supported")
    }

    fn load_by_name(
        &mut self,
        model_name: String,
    ) -> Result<Resource<Graph>, Resource<errors::Error>> {
        let parts: Vec<_> = model_name.split(':').map(|x| x.to_string()).collect();
        if parts.len() > 1 {
            if let Some(graph_encoding) = map_string_to_graph_encoding(&parts[0]) {
                for backend in self.backends.iter_mut() {
                    if backend.encoding() == graph_encoding {
                        match backend.load_by_name(model_name.clone()) {
                            Ok(graph_internal_data) => {
                                return MLHostImpl::new_graph(
                                    &mut self.graphs,
                                    &mut self.errors,
                                    graph_internal_data,
                                )
                                .unwrap();
                            }
                            Err(err) => {
                                return Err(MLHostImpl::new_error(
                                    &mut self.errors,
                                    ErrorCode::RuntimeError,
                                    format!("Can't load model '{model_name}' error = {err:?}"),
                                ));
                            }
                        }
                    }
                }
            }
        }
        panic!("[graph::Host] fn load_by_name -> model not supported ")
    }
}

impl inference::Host for MLHostImpl {}
impl tensor::Host for MLHostImpl {}

fn map_string_to_graph_encoding(target: &str) -> Option<GraphEncoding> {
    match target {
        "openvino" => Some(GraphEncoding::Openvino),
        _ => None,
    }
}
