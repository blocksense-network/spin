use std::path::PathBuf;
use std::sync::Arc;

use llm::{InferenceSession, InferenceSessionConfig, Model, OutputRequest, TokenUtf8Buffer};
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
    token_utf8_buf: TokenUtf8Buffer,
    response: String,
    n_past: usize,
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
            token_utf8_buf: TokenUtf8Buffer::new(),
            response: Default::default(),
            n_past: 0,
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
                    "next_token" => {
                        if tensor.tensor_type != TensorType::I32 || tensor.tensor_dimensions != vec![1, 1] || tensor.tensor_data.len() != 4 {
                            return Err(anyhow!("Expected tensor data is tensor_type = I32 and tensor_dimensions = [1,1]"));
                        }
                        let next_token = u32::from_le_bytes(tensor.tensor_data[0..4].try_into().context("This should never happen!")?);
                        //println!("NEXT TOKEN = {next_token}");
                        let vocab = self.model.tokenizer();
                        let num_tokens: u32 = vocab.len().try_into().context("Only vocabs with num tokens less then 32 unsigned bits are supprted")?;

                        //println!("NUM TOKENS = {num_tokens}");
                        if next_token >= num_tokens {
                            return Err(anyhow!("Unexpected token with number {next_token}, which is greater the number of tokens {num_tokens}"));
                        }
                        //println!("EOT TOKEN ID = {}", self.model.eot_token_id());
                        if next_token == self.model.eot_token_id() {
                            return Err(anyhow!("End of sequence token passed as {next_token}"));
                        }
                        //println!("n_past = {}, model_size = {}", self.n_past, self.model.context_size());
                        if self.n_past + 1 >= self.model.context_size() {
                            return Err(anyhow!("Exceeded maximunim number of model context size = {}", self.model.context_size()));
                        }
                        self.n_past += 1;                
                        let token = vocab.token(next_token as usize);
                        if let Some(tokens) = self.token_utf8_buf.push(&token) {
                            self.response.push_str(&tokens);

                        }
                        self.query_token_ids = vec![next_token];
                        return Ok(());
                    },
                    _ => Err(anyhow!("Unknown input with name {name}. Supported names are `query` and `next_token`")),
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
                    "last_logits" => self.get_all_logits(),
                    "response" => self.get_response(),
                    "eos_token_id" => self.get_eos_token_id(),
                    _ => Err(anyhow!("Unknown output with name {name}. Supported names are `embeddings`, `response` and `last_logits`")),
                }
            }
            TensorId::Index(i) => {
                Err(anyhow!("Unknown output with index {i}. Supported indexes are names are `embeddings`, `response` and `last_logits`"))
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
            let vocab = self.model.tokenizer();
            let num_tokens: u32 = vocab.len().try_into().context("Only vocabs with num tokens less then 32 unsigned bits are supprted")?;
            let l = tensor_data_f32.len();
            Ok(Self::get_tensor_data(&tensor_data_f32[l - num_tokens as usize .. l]))
        } else {
            Err(anyhow!("Мissing all logits in this model"))
        }
    }

    fn get_response(&mut self) -> Result<TensorInternalData, anyhow::Error> {
        let tensor_data = self.response.clone().into_bytes();
        let tensor_type = TensorType::U8;
        let tensor_dimensions: Vec<u32> = vec![tensor_data.len() as u32];
        Ok(TensorInternalData {
            tensor_data,
            tensor_dimensions,
            tensor_type,
        })
    }

    fn get_eos_token_id(&mut self) -> Result<TensorInternalData, anyhow::Error> {
        let tensor_data = self.model.eot_token_id().to_le_bytes().to_vec();
        let tensor_type = TensorType::I32;
        let tensor_dimensions: Vec<u32> = vec![1];
        Ok(TensorInternalData {
            tensor_data,
            tensor_dimensions,
            tensor_type,
        })
    }
}