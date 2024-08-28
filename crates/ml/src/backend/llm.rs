
use std::path::PathBuf;
use std::sync::Arc;

use llm::{InferenceSession, InferenceSessionConfig, Model, OutputRequest};
use spin_world::v2 as ml_wit;

use ml_wit::graph::{ExecutionTarget, GraphBuilder, GraphEncoding};
use ml_wit::tensor;
use ml_wit::tensor::TensorType;

use crate::backend::BackendGraph;

use super::{BackendExecutionContext, BackendInner, TensorId};
use super::GraphInternalData;
use crate::ml_host_impl::ExecutionContext;
use crate::ml_host_impl::TensorInternalData;
use std::sync::Mutex;

use anyhow::{anyhow, Ok};



pub struct  RustformersLLMBackend {
    pub state_dir: Option<PathBuf>,
}

pub struct  RustformersLLMGraph {
    model: Arc<dyn llm::Model>, 
}

pub struct  LLMExecutionContext {
    model:  Arc<dyn llm::Model>, 
    inference_session: InferenceSession,
    output_request: llm::OutputRequest
}

impl BackendInner for RustformersLLMBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Openvino
    }

    fn load(
        &mut self,
        builders: Vec<GraphBuilder>,
        _target: ExecutionTarget,
        _encoding: GraphEncoding,
        _name: Option<String>,
    ) -> Result<GraphInternalData, anyhow::Error> {

        //Err(anyhow!("not implemented"))
        Err(anyhow!("not implemented"))

    }

    fn load_by_name(&mut self, _model_name: String) -> Result<GraphInternalData, anyhow::Error> {
        let model_architecture = llm::ModelArchitecture::Llama;
        let tokenizer_source = llm::TokenizerSource::Embedded;
        let model_path = self.state_dir.clone().unwrap().join("models").join("open_llama_3b-f16.bin");
        let model = llm::load_dynamic(
            Some(model_architecture),
            &model_path,
            tokenizer_source,
            Default::default(),
            llm::load_progress_callback_stdout,
        )
        .map_err(|err| {
            anyhow!("Failed to load {model_architecture} model from {model_path:?}: {err}")
        })?;
        let res = RustformersLLMGraph {
            model: Arc::<dyn Model>::from(model),
        };
        Ok(GraphInternalData(Box::new(res)))

    }
}

impl BackendGraph for RustformersLLMGraph {
    fn init_execution_context(&mut self) -> Result<ExecutionContext, anyhow::Error> {
        let inference_session = self.model.as_ref().start_session(InferenceSessionConfig::default());
        let embeddings = vec![];
        let output_request = OutputRequest {all_logits:None, embeddings: Some(embeddings)};
        Ok(ExecutionContext(Box::new(LLMExecutionContext {
            model: self.model.clone(),
            inference_session,
            output_request,
        })))
    }
    
}

unsafe impl Send for LLMExecutionContext {}
unsafe impl Sync for LLMExecutionContext {}

impl BackendExecutionContext for LLMExecutionContext {
    fn set_input(
        &mut self,
        tensor_id: &TensorId,
        tensor: &TensorInternalData,
    ) -> Result<(), anyhow::Error> {
        //self.
        // Construct the tensor.

    }

    fn compute(&mut self) -> Result<(), anyhow::Error> {
        self.model.evaluate(&mut self.inference_session, &query_token_ids, &mut self.output_request);
        Ok(())

    }
    fn get_output(&mut self, tensor_id: &TensorId) -> Result<crate::ml_host_impl::TensorInternalData, anyhow::Error> {
        
    }
}