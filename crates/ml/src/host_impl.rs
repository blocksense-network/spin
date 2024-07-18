use crate::imagenet_download::imagenet_check_models;
use crate::imagenet_download::{imagenet_download, OpenvinoModel};
use anyhow::{anyhow, Context};
use ml_wit::errors::ErrorCode;
use ml_wit::graph::{ExecutionTarget, Graph, GraphBuilder, GraphEncoding};
use ml_wit::inference::GraphExecutionContext;
use ml_wit::{errors, graph, inference, tensor};
use spin_core::async_trait;
use spin_world::v2 as ml_wit;
use std::path::PathBuf;

use spin_core::wasmtime::component::Resource;
use tokio::sync::Mutex;

use crate::backend::BackendInner;

use openvino::{Core, Layout, Precision, TensorDesc};

#[derive(Debug)]
pub struct GraphInternalData {
    pub xml: Vec<u8>,
    pub weights: Vec<u8>,
    pub target: ExecutionTarget,
}

pub struct GraphExecutionContextInternalData {
    pub cnn_network: openvino::CNNNetwork,
    pub executable_network: Mutex<openvino::ExecutableNetwork>,
    pub infer_request: openvino::InferRequest,
}

pub struct TensorInternalData {
    tensor_dimensions: tensor::TensorDimensions,
    tensor_type: tensor::TensorType,
    tensor_data: tensor::TensorData,
}

pub struct ErrorInternalData {
    code: errors::ErrorCode,
    message: String,
}

#[derive(Default)]
pub struct MLHostImpl {
    pub state_dir: Option<PathBuf>,
    pub openvino: Option<openvino::Core>,
    pub graphs: table::Table<GraphInternalData>,
    pub executions: table::Table<GraphExecutionContextInternalData>,
    pub tensors: table::Table<TensorInternalData>,
    pub errors: table::Table<ErrorInternalData>,

    pub backends: Vec<Box<dyn BackendInner>>,
}

impl MLHostImpl {
    fn loeaded_to_graph(
        &mut self,
        model: OpenvinoModel,
        target: ExecutionTarget,
    ) -> Result<Result<Resource<Graph>, Resource<errors::Error>>, anyhow::Error> {
        let graph_internal_data = GraphInternalData {
            xml: model.xml,
            weights: model.weights,
            target,
        };
        MLHostImpl::new_graph(&mut self.graphs, &mut self.errors, graph_internal_data)
    }

    fn load_imagenet(
        &mut self,
        target: ExecutionTarget,
    ) -> Result<Result<Resource<Graph>, Resource<errors::Error>>, anyhow::Error> {
        if let Some(dir) = &self.state_dir {
            match imagenet_check_models(dir) {
                Ok(model) => self.loeaded_to_graph(model, target),
                Err(_) => {
                    imagenet_download(dir)?;
                    let model = imagenet_check_models(dir).map_err(|e| anyhow!("{:?}", e))?;
                    self.loeaded_to_graph(model, target)
                }
            }
        } else {
            Err(anyhow!(
                "state_dir is not set, therefore there is no place to download models"
            ))
        }
    }
    fn new_error(
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

    fn new_execution_context(
        openvino: &mut Core,
        graph: &GraphInternalData,
    ) -> Result<GraphExecutionContextInternalData, String> {
        let mut cnn_network = openvino
            .read_network_from_buffer(&graph.xml, &graph.weights)
            .map_err(|e| format!("Can't create graph execution context, err=r {e:?}"))?;
        for i in 0..cnn_network.get_inputs_len().unwrap() {
            let name = cnn_network.get_input_name(i).map_err(|e| e.to_string())?;
            cnn_network
                .set_input_layout(&name, Layout::NHWC)
                .map_err(|e| e.to_string())?;
        }

        let mut exec_network: openvino::ExecutableNetwork = openvino
            .load_network(&cnn_network, map_execution_target_to_string(graph.target))
            .map_err(|e| {
                format!(
                    "Can't create graph execution context for target {:?}, error {e:?}",
                    graph.target
                )
            })?;
        let infer_request = exec_network
            .create_infer_request()
            .map_err(|e| format!("Can't create InferRequest, errpr = {e:?}"))?;
        let graph_execution_context = GraphExecutionContextInternalData {
            cnn_network,
            executable_network: Mutex::new(exec_network),
            infer_request,
        };
        Ok(graph_execution_context)
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
        if let Some(graph) = self.graphs.get(graph.rep()) {
            Ok(
                match MLHostImpl::new_execution_context(self.openvino.as_mut().expect(""), graph) {
                    Ok(graph_execution_context) => self
                        .executions
                        .push(graph_execution_context)
                        .map(Resource::<inference::GraphExecutionContext>::new_own)
                        .map_err(|_| {
                            MLHostImpl::new_error(
                                &mut self.errors,
                                ErrorCode::RuntimeError,
                                "Can't create graph execution context".to_string(),
                            )
                        }),
                    Err(message) => Err(MLHostImpl::new_error(
                        &mut self.errors,
                        ErrorCode::RuntimeError,
                        message,
                    )),
                },
            )
        } else {
            Err(anyhow!(
                "[graph::HostGraph] fn init_execution_context -> Not implemented"
            ))
        }
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
        let index = input_name
            .parse()
            .context("Can't parse {} to usize for input_name")?;
        // Construct the blob structure. TODO: there must be some good way to
        // discover the layout here; `desc` should not have to default to NHWC.
        let tensor_resource = self
            .tensors
            .get(tensor.rep())
            .context(format!("Can't find tensor with ID = {}", tensor.rep()))?;
        let precision = map_tensor_type_to_precision(tensor_resource.tensor_type);
        let dimensions = tensor_resource
            .tensor_dimensions
            .iter()
            .map(|&d| d as usize)
            .collect::<Vec<_>>();
        let desc = TensorDesc::new(Layout::NHWC, &dimensions, precision);
        let blob = openvino::Blob::new(&desc, &tensor_resource.tensor_data)?;
        let execution_context: &mut GraphExecutionContextInternalData = self
            .executions
            .get_mut(graph_execution_context.rep())
            .context(format!(
                "Can't find graph execution context with ID = {}",
                graph_execution_context.rep()
            ))?;
        let input_name = execution_context
            .cnn_network
            .get_input_name(index)
            .context(format!("Can't find input with name = {}", index))?;
        let res = execution_context
            .infer_request
            .set_blob(&input_name, &blob)
            .map_err(|err| {
                MLHostImpl::new_error(
                    &mut self.errors,
                    ErrorCode::RuntimeError,
                    format!("Inference error = {:?}", err.to_string()),
                )
            });
        Ok(res)
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
        Ok(graph_execution.infer_request.infer().map_err(|err| {
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
        let index = input_name
            .parse::<usize>()
            .context("Can't parse {} to usize for input_name")?;
        let graph_execution = self
            .executions
            .get_mut(graph_execution_context.rep())
            .ok_or(anyhow!(format!(
                "Can't find graph execution context with ID = {}",
                graph_execution_context.rep()
            )))?;
        let output_name = graph_execution
            .cnn_network
            .get_output_name(index)
            .context("Can't find output name for ID = {index}")?;
        let blob = graph_execution
            .infer_request
            .get_blob(&output_name)
            .context("Can't get blob for output name = {output_name}")?;
        let tensor_desc = blob.tensor_desc().context("Can't get blob description")?;
        let buffer = blob.buffer().context("Can't get blob buffer")?.to_vec();
        let tensor_dimensions = tensor_desc
            .dims()
            .iter()
            .map(|&d| d as u32)
            .collect::<Vec<_>>();

        let tensor = TensorInternalData {
            tensor_dimensions,
            tensor_type: map_precision_to_tensor_type(tensor_desc.precision()),
            tensor_data: buffer,
        };
        Ok(
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
                    .map_err(|_| anyhow!("Can't allocate error"))?),
            },
        )
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
        graph: Vec<GraphBuilder>,
        graph_encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Result<Resource<Graph>, Resource<errors::Error>>, anyhow::Error> {
        if graph.len() != 2 {
            return Err(anyhow!("Expected 2 elements in graph builder vector"));
        }
        if graph_encoding != GraphEncoding::Openvino {
            return Err(anyhow!("Only OpenVINO encoding is supported"));
        }
        // Read the guest array.
        let graph_internal_data = GraphInternalData {
            xml: graph[0].clone(),
            weights: graph[1].clone(),
            target,
        };
        MLHostImpl::new_graph(&mut self.graphs, &mut self.errors, graph_internal_data)
    }

    async fn load_by_name(
        &mut self,
        model_name: String,
    ) -> Result<Result<Resource<Graph>, Resource<errors::Error>>, anyhow::Error> {
        let parts: Vec<_> = model_name.split(':').map(|x| x.to_string()).collect();
        if parts.len() > 1 {
            if let Some(graph_encoding) = map_string_to_graph_encoding(&parts[0]) {
                for backend in self.backends.iter() {
                    if backend.encoding() == graph_encoding {
                        //return backend.load_by_name(model_name);

                        if model_name == "imagenet" {
                            return self.load_imagenet(ExecutionTarget::Gpu);
                        }

                        if parts.len() == 3 {
                            if let Some(target) = map_string_to_execution_target(&parts[2]) {
                                let model_name = &parts[1];
                                if model_name == "imagenet" {
                                    return self.load_imagenet(target);
                                }
                            }
                        }
                    }
                }
            }
        }

        Err(anyhow!(
            "[graph::Host] fn load_by_name -> model not supported "
        ))
    }
}

impl inference::Host for MLHostImpl {}
impl tensor::Host for MLHostImpl {}

/// Return the execution target string expected by OpenVINO from the
/// `ExecutionTarget` enum provided by wasi-nn.
fn map_execution_target_to_string(target: ExecutionTarget) -> &'static str {
    match target {
        ExecutionTarget::Cpu => "CPU",
        ExecutionTarget::Gpu => "GPU",
        ExecutionTarget::Tpu => {
            unimplemented!("OpenVINO does not support TPU execution targets")
        }
    }
}

fn map_string_to_graph_encoding(target: &str) -> Option<GraphEncoding> {
    match target {
        "openvino" => Some(GraphEncoding::Openvino),
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

/// Return OpenVINO's precision type for the `TensorType` enum provided by
/// wasi-nn.
fn map_tensor_type_to_precision(tensor_type: tensor::TensorType) -> openvino::Precision {
    match tensor_type {
        tensor::TensorType::Fp16 => Precision::FP16,
        tensor::TensorType::Fp32 => Precision::FP32,
        tensor::TensorType::Fp64 => Precision::FP64,
        tensor::TensorType::U8 => Precision::U8,
        tensor::TensorType::I32 => Precision::I32,
        tensor::TensorType::I64 => Precision::I64,
        tensor::TensorType::Bf16 => todo!("not yet supported in `openvino` bindings"),
    }
}
fn map_precision_to_tensor_type(precision: openvino::Precision) -> tensor::TensorType {
    match precision {
        Precision::FP16 => tensor::TensorType::Fp16,
        Precision::FP32 => tensor::TensorType::Fp32,
        Precision::FP64 => tensor::TensorType::Fp64,
        Precision::U8 => tensor::TensorType::U8,
        Precision::I32 => tensor::TensorType::I32,
        Precision::I64 => tensor::TensorType::I64,
        _ => todo!("not yet supported in `openvino` bindings"),
    }
}
