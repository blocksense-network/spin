use spin_world::v2 as ml_wit;

use ml_wit::graph::{ExecutionTarget, GraphBuilder, GraphEncoding};
use ml_wit::tensor;
use ml_wit::tensor::TensorType;

use crate::backend::BackendGraph;

use super::{BackendExecutionContext, BackendInner, TensorId};

use crate::model_files::{self, ModelFiles};
use openvino::{DeviceType, ElementType, Shape, Tensor as OvTensor};

use std::path::PathBuf;

use crate::ml_host_impl::{ExecutionContext, GraphInternalData, TensorInternalData};
use anyhow::{anyhow, Context};
use std::sync::{Arc, Mutex};

pub struct OpenvinoBackend {
    pub openvino: openvino::Core,
    pub state_dir: Option<PathBuf>,
}

struct OpenvinoGraph(Arc<Mutex<openvino::CompiledModel>>);

unsafe impl Send for OpenvinoGraph {}
unsafe impl Sync for OpenvinoGraph {}

unsafe impl Send for OpenvinoBackend {}
unsafe impl Sync for OpenvinoBackend {}

impl BackendInner for OpenvinoBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Openvino
    }

    fn load(
        &mut self,
        model_files: &ModelFiles,
        target: ExecutionTarget,
    ) -> Result<GraphInternalData, anyhow::Error> {
        let state_dir = self
            .state_dir
            .clone()
            .context("state_dir is not set, therefore there is no place to download models")?;
        let builders = model_files.builders(&state_dir)?;
        if builders.len() != 2 {
            return Err(anyhow!("Expected 2 elements in graph builder vector"));
        }
        if model_files.encoding != GraphEncoding::Openvino {
            return Err(anyhow!("Only OpenVINO encoding is supported"));
        }

        // Read the guest array.
        let xml = &builders[0];
        let weights = &builders[1];

        // Construct a new tensor for the model weights.
        let shape = Shape::new(&[1, weights.len() as i64 / 4])?;
        let mut weights_tensor = OvTensor::new(ElementType::F32, &shape)?;
        let buffer = weights_tensor.get_raw_data_mut()?;
        buffer.copy_from_slice(weights);

        // Construct OpenVINO graph structures: `model` contains the graph
        // structure, `compiled_model` can perform inference.

        let model = self
            .openvino
            .read_model_from_buffer(xml, Some(&weights_tensor))?;

        let compiled_model = self
            .openvino
            .compile_model(&model, map_execution_target_to_string(target))?;
        Ok(GraphInternalData(Box::new(OpenvinoGraph(Arc::new(
            Mutex::new(compiled_model),
        )))))
    }
}

impl BackendGraph for OpenvinoGraph {
    fn init_execution_context(&mut self) -> Result<ExecutionContext, anyhow::Error> {
        let mut compiled_model = self.0.lock().unwrap();
        let infer_request = compiled_model.create_infer_request()?;
        Ok(ExecutionContext(Box::new(OpenvinoExecutionContext {
            infer_request,
        })))
    }
}

pub struct OpenvinoExecutionContext {
    pub infer_request: openvino::InferRequest,
}

unsafe impl Send for OpenvinoExecutionContext {}
unsafe impl Sync for OpenvinoExecutionContext {}

impl BackendExecutionContext for OpenvinoExecutionContext {
    fn set_input(
        &mut self,
        tensor_id: &TensorId,
        tensor: &TensorInternalData,
    ) -> Result<(), anyhow::Error> {
        // Construct the tensor.
        let precision = map_tensor_type_to_element_type(&tensor.tensor_type);
        let dimensions = tensor
            .tensor_dimensions
            .iter()
            .map(|&d| d as i64)
            .collect::<Vec<_>>();
        let shape = Shape::new(&dimensions)?;
        let mut new_tensor = OvTensor::new(precision, &shape)?;
        let buffer = new_tensor.get_raw_data_mut()?;
        buffer.copy_from_slice(&tensor.tensor_data);

        // Assign the tensor to the request.
        match tensor_id {
            TensorId::Index(i) => self
                .infer_request
                .set_input_tensor_by_index(*i as usize, &new_tensor)?,
            TensorId::Name(name) => self.infer_request.set_tensor(name, &new_tensor)?,
        };

        Ok(())
    }

    fn compute(&mut self) -> Result<(), anyhow::Error> {
        self.infer_request
            .infer()
            .map_err(|err| anyhow!("Inference error = {:?}", err.to_string()))
    }

    fn get_output(&mut self, tensor_id: &TensorId) -> Result<TensorInternalData, anyhow::Error> {
        let output_tensor = match tensor_id {
            TensorId::Index(i) => self.infer_request.get_output_tensor_by_index(*i as usize)?,
            TensorId::Name(name) => self.infer_request.get_tensor(name)?,
        };
        let dimensions = output_tensor
            .get_shape()?
            .get_dimensions()
            .iter()
            .map(|&dim| dim as u32)
            .collect::<Vec<u32>>();
        let element_type = output_tensor
            .get_element_type()
            .map_err(|err| anyhow!("Inference error = {err:?}"))?;
        let data = output_tensor.get_raw_data()?.to_vec();
        Ok(TensorInternalData {
            tensor_dimensions: dimensions,
            tensor_type: map_precision_to_tensor_type(element_type),
            tensor_data: data,
        })
    }
}

/// Return the execution target string expected by OpenVINO from the
/// `ExecutionTarget` enum provided by wasi-nn.
fn map_execution_target_to_string(target: ExecutionTarget) -> DeviceType<'static> {
    match target {
        ExecutionTarget::Cpu => DeviceType::CPU,
        ExecutionTarget::Gpu => DeviceType::GPU,
        ExecutionTarget::Tpu => {
            unimplemented!("OpenVINO does not support TPU execution targets")
        }
    }
}

fn map_precision_to_tensor_type(precision: openvino::ElementType) -> tensor::TensorType {
    //use openvino::Precision;
    match precision {
        ElementType::F16 => TensorType::Fp16,
        ElementType::F32 => TensorType::Fp32,
        ElementType::F64 => TensorType::Fp64,
        ElementType::U8 => TensorType::U8,
        ElementType::I32 => TensorType::I32,
        ElementType::I64 => TensorType::I64,
        _ => todo!("not yet supported in `openvino` bindings"),
    }
}

fn map_tensor_type_to_element_type(tensor_type: &TensorType) -> ElementType {
    match tensor_type {
        TensorType::Fp16 => ElementType::F16,
        TensorType::Fp32 => ElementType::F32,
        TensorType::Fp64 => ElementType::F64,
        TensorType::U8 => ElementType::U8,
        TensorType::I32 => ElementType::I32,
        TensorType::I64 => ElementType::I64,
        TensorType::Bf16 => ElementType::Bf16,
    }
}
