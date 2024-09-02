use std::path::PathBuf;
use std::sync::Arc;

use llm::{InferenceSession, InferenceSessionConfig, Model, OutputRequest};
use spin_world::v2 as ml_wit;

use ml_wit::graph::{ExecutionTarget, GraphBuilder, GraphEncoding};
use ml_wit::tensor::TensorType;

use crate::backend::BackendGraph;

use super::GraphInternalData;
use super::{BackendExecutionContext, BackendInner, TensorId};
use crate::ml_host_impl::ExecutionContext;
use crate::ml_host_impl::TensorInternalData;

use anyhow::{anyhow, Context, Ok};
use sha1::{Digest, Sha1};

pub struct RustformersLLMBackend {
    pub state_dir: Option<PathBuf>,
}

pub struct RustformersLLMGraph {
    model: Arc<dyn llm::Model>,
}

pub struct LLMExecutionContext {
    model: Arc<dyn llm::Model>,
    inference_session: InferenceSession,
    output_request: llm::OutputRequest,
    query_token_ids: Vec<u32>,
}

impl BackendInner for RustformersLLMBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Ggml
    }

    fn load(
        &mut self,
        _builders: Vec<GraphBuilder>,
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
        let model_path = self
            .state_dir
            .clone()
            .unwrap()
            .join("models")
            .join("open_llama_3b-f16.bin");

        let file_data = std::fs::read(&model_path).unwrap();
        let file_data_size = file_data.len();
        let mut hasher = Sha1::new();
        hasher.update(file_data);
        let sha1_hash = hasher.finalize();
        println!("{model_path:?} hash = {sha1_hash:x} size = {file_data_size}");

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
        let inference_session = self
            .model
            .as_ref()
            .start_session(InferenceSessionConfig::default());
        let output_request = OutputRequest {
            all_logits: Some(vec![]),
            embeddings: Some(vec![]),
        };
        Ok(ExecutionContext(Box::new(LLMExecutionContext {
            model: self.model.clone(),
            inference_session,
            output_request,
            query_token_ids: Default::default(),
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
        match tensor_id {
            TensorId::Name(name) => {
                match name.as_str() {
                    "query" => {
                        let query = std::str::from_utf8(&tensor.tensor_data).unwrap();
                        let vocab = self.model.tokenizer();
                        let beginning_of_sentence = true;
                        self.query_token_ids = vocab
                            .tokenize(query, beginning_of_sentence)
                            .unwrap()
                            .iter()
                            .map(|(_, tok)| *tok)
                            .collect::<Vec<_>>();
                        return Ok(())
                    },
                    "token_ids" => {
                        let mut query_token_ids = vec![];
                        let vocab = self.model.tokenizer();
                        let num_tokens: u32 = vocab.len().try_into().context("Only vocabs with num tokens less then 32 unsigned bits are supprted")?;
                        for i in 0..(tensor.tensor_data.len()/4) {
                            let offset = i * 4;
                            let v = u32::from_le_bytes(
                                tensor.tensor_data[offset..offset + 4]
                                    .try_into()
                                    .expect("Needed 4 bytes for a float"),
                            );
                            if v < num_tokens {
                                query_token_ids.push(v);
                            } else {
                                return Err(anyhow!("Unexpected token with number {v}, which is greater the number of tokens {num_tokens}"));
                            }
                        }
                        self.query_token_ids = query_token_ids;
                        return Ok(());
                    },
                    _ => Err(anyhow!("Unknown output with name {name}. Supported names are `embeddings` and `all_logits`")),
                }
            }
            TensorId::Index(_) => {
                Err(anyhow!("Input as index is not supported. Supported TensorIDs `query` and `token_ids`"))
            }
        }
    }

    fn compute(&mut self) -> Result<(), anyhow::Error> {
        self.model.evaluate(
            &mut self.inference_session,
            &self.query_token_ids,
            &mut self.output_request,
        );
        Ok(())
    }

    fn get_output(&mut self, tensor_id: &TensorId) -> Result<TensorInternalData, anyhow::Error> {
        match tensor_id {
            TensorId::Name(name) => {
                match name.as_str() {
                    "embeddings" => self.get_embeddings(),
                    "all_logits" => self.get_all_logits(),
                    _ => Err(anyhow!("Unknown output with name {name}. Supported names are `embeddings` and `all_logits`")),
                }
            }
            TensorId::Index(i) => {
                Err(anyhow!("Unknown output with index {i}. Supported indexes are names are `embeddings` and `all_logits`"))
            }
        }
    }
}

impl LLMExecutionContext {
    fn get_tensor_data(data: &[f32]) -> TensorInternalData {
        let tensor_data = data
            .iter()
            .copied()
            .flat_map(|x| f32::to_le_bytes(x).into_iter())
            .collect();
        let tensor_type = TensorType::Fp32;
        let tensor_dimensions: Vec<u32> = vec![data.len() as u32];
        TensorInternalData {
            tensor_data,
            tensor_dimensions,
            tensor_type,
        }
    }

    fn get_embeddings(&mut self) -> Result<TensorInternalData, anyhow::Error> {
        if let Some(tensor_data_f32) = &self.output_request.embeddings {
            Ok(Self::get_tensor_data(tensor_data_f32))
        } else {
            Err(anyhow!("Мissing embeddings in this model"))
        }
    }

    fn get_all_logits(&mut self) -> Result<TensorInternalData, anyhow::Error> {
        if let Some(tensor_data_f32) = &self.output_request.all_logits {
            Ok(Self::get_tensor_data(tensor_data_f32))
        } else {
            Err(anyhow!("Мissing all logits in this model"))
        }
    }
}
