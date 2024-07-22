use spin_world::v2 as ml_wit;

use ml_wit::graph::{ExecutionTarget, GraphBuilder, GraphEncoding};
use ml_wit::tensor::TensorType;

use crate::backend::tensor;

use super::{BackendInner, BackendExecutionContext};
use crate::imagenet_download::{imagenet_check_models, imagenet_download, OpenvinoModel};
use openvino::{Core, Layout, Precision, TensorDesc};
use std::path::PathBuf;

use crate::host_impl::{GraphInternalData, TensorInternalData, ExecutionContext};
use anyhow::{anyhow, Context};
use tokio::sync::Mutex;


pub struct OpenvinoBackend {
    pub openvino: openvino::Core,
    pub state_dir: Option<PathBuf>,
}
unsafe impl Send for OpenvinoBackend {}
unsafe impl Sync for OpenvinoBackend {}

impl BackendInner for OpenvinoBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Openvino
    }

    fn load(
        &mut self,
        builders: Vec<GraphBuilder>,
        target: ExecutionTarget,
        encoding: GraphEncoding,
        name: Option<String>,
    ) -> Result<GraphInternalData, anyhow::Error> {
        if builders.len() != 2 {
            return Err(anyhow!("Expected 2 elements in graph builder vector"));
        }
        if encoding != GraphEncoding::Openvino {
            return Err(anyhow!("Only OpenVINO encoding is supported"));
        }
        // Read the guest array.
        let graph_internal_data = GraphInternalData {
            builders,
            target,
            encoding,
            name,
        };
        Ok(graph_internal_data)
    }

    fn load_by_name(&mut self, model_name: String) -> Result<GraphInternalData, anyhow::Error> {
        let parts: Vec<_> = model_name.split(':').map(|x| x.to_string()).collect();
        if parts.len() == 3 {
            if let Some(target) = map_string_to_execution_target(&parts[2]) {
                let model_name = &parts[1];
                if model_name == "imagenet" {
                    return self.load_imagenet(target);
                }
            }
        }
        Err(anyhow!("not implemented"))
    }

    fn init_execution_context(
        &mut self,
        graph: &GraphInternalData,
    ) -> Result<ExecutionContext, anyhow::Error> {
        Ok(ExecutionContext(Box::new(
            OpenvinoBackend::new_execution_context(&mut self.openvino, graph)
                .map_err(|message| anyhow!("{}", message))?),
        ))
    }
}

pub struct OpenvinoExecutionContext {
    pub cnn_network: openvino::CNNNetwork,
    pub executable_network: Mutex<openvino::ExecutableNetwork>,
    pub infer_request: openvino::InferRequest,
}

unsafe impl Send for OpenvinoExecutionContext {}
unsafe impl Sync for OpenvinoExecutionContext {}

impl BackendExecutionContext for OpenvinoExecutionContext {
    fn set_input(
        &mut self,
        input_name: String,
        tensor: &TensorInternalData,
    ) -> Result<(), anyhow::Error> {
        let index = input_name
            .parse()
            .context("Can't parse {} to usize for input_name")?;
        // Construct the blob structure. TODO: there must be some good way to
        // discover the layout here; `desc` should not have to default to NHWC.
        let precision = map_tensor_type_to_precision(tensor.tensor_type);
        let dimensions = tensor
            .tensor_dimensions
            .iter()
            .map(|&d| d as usize)
            .collect::<Vec<_>>();
        let desc = TensorDesc::new(Layout::NHWC, &dimensions, precision);
        let blob = openvino::Blob::new(&desc, &tensor.tensor_data)?;

        let input_name = self
            .cnn_network
            .get_input_name(index)
            .context(format!("Can't find input with name = {}", index))?;
        self.infer_request
            .set_blob(&input_name, &blob)
            .map_err(|err| anyhow!("Inference error = {:?}", err.to_string()))
    }

    fn compute(&mut self) -> Result<(), anyhow::Error> {
        self.infer_request
            .infer()
            .map_err(|err| anyhow!("Inference error = {:?}", err.to_string()))
    }

    fn get_output(&mut self, input_name: String) -> Result<TensorInternalData, anyhow::Error> {
        let index = input_name
            .parse::<usize>()
            .context("Can't parse {} to usize for input_name")?;

        let output_name = self
            .cnn_network
            .get_output_name(index)
            .context("Can't find output name for ID = {index}")?;
        let blob = self
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
        Ok(tensor)
    }
}

impl OpenvinoBackend {
    fn load_imagenet(
        &mut self,
        target: ExecutionTarget,
    ) -> Result<GraphInternalData, anyhow::Error> {
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

    fn loeaded_to_graph(
        &mut self,
        model: OpenvinoModel,
        target: ExecutionTarget,
    ) -> Result<GraphInternalData, anyhow::Error> {
        let builders = vec![model.xml, model.weights];
        let graph_internal_data = GraphInternalData {
            builders,
            target,
            encoding: GraphEncoding::Openvino,
            name: Some(model.name),
        };
        Ok(graph_internal_data)
    }

    fn new_execution_context(
        openvino: &mut Core,
        graph: &GraphInternalData,
    ) -> Result<OpenvinoExecutionContext, String> {
        let mut cnn_network = openvino
            .read_network_from_buffer(&graph.builders[0], &graph.builders[1])
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
        let graph_execution_context = OpenvinoExecutionContext {
            cnn_network,
            executable_network: Mutex::new(exec_network),
            infer_request,
        };
        Ok(graph_execution_context)
    }
}

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
///
fn map_tensor_type_to_precision(tensor_type: tensor::TensorType) -> openvino::Precision {
    //use openvino::Precision;

    match tensor_type {
        TensorType::Fp16 => Precision::FP16,
        TensorType::Fp32 => Precision::FP32,
        TensorType::Fp64 => Precision::FP64,
        TensorType::U8 => Precision::U8,
        TensorType::I32 => Precision::I32,
        TensorType::I64 => Precision::I64,
        TensorType::Bf16 => todo!("not yet supported in `openvino` bindings"),
    }
}
fn map_precision_to_tensor_type(precision: openvino::Precision) -> tensor::TensorType {
    //use openvino::Precision;
    match precision {
        Precision::FP16 => TensorType::Fp16,
        Precision::FP32 => TensorType::Fp32,
        Precision::FP64 => TensorType::Fp64,
        Precision::U8 => TensorType::U8,
        Precision::I32 => TensorType::I32,
        Precision::I64 => TensorType::I64,
        _ => todo!("not yet supported in `openvino` bindings"),
    }
}
