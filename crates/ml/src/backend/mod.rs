use spin_world::v2 as ml_wit;

use ml_wit::graph::{ExecutionTarget, GraphBuilder, GraphEncoding};

use crate::ml_host_impl::{ExecutionContext, GraphInternalData, TensorInternalData};

#[cfg(feature = "openvino")]
pub mod openvino;

pub mod llm;

/// A [Backend] contains the necessary state to load [Graph]s.
pub trait BackendInner: Send + Sync {
    fn encoding(&self) -> GraphEncoding;
    fn load(
        &mut self,
        builders: Vec<GraphBuilder>,
        target: ExecutionTarget,
        encoding: GraphEncoding,
        name: Option<String>,
    ) -> Result<GraphInternalData, anyhow::Error>;

    fn load_by_name(&mut self, model_name: String) -> Result<GraphInternalData, anyhow::Error>;
}

/// A [BackendGraph] can create [BackendExecutionContext]s; this is the backing
/// implementation for the user-facing graph.
pub trait BackendGraph: Send + Sync {
    fn init_execution_context(&mut self) -> Result<ExecutionContext, anyhow::Error>;
}

pub trait BackendExecutionContext: Send + Sync {
    fn set_input(
        &mut self,
        tensor_id: &TensorId,
        tensor: &TensorInternalData,
    ) -> Result<(), anyhow::Error>;

    fn compute(&mut self) -> Result<(), anyhow::Error>;

    fn get_output(&mut self, tensor_id: &TensorId) -> Result<TensorInternalData, anyhow::Error>;
}

/// An identifier for a tensor in a [Graph].
#[derive(Debug)]
pub enum TensorId {
    Index(u32),
    Name(String),
}
impl TensorId {
    pub fn new(input_name: &String) -> Self {
        match input_name.parse::<u32>() {
            Ok(index) => TensorId::Index(index),
            Err(_) => TensorId::Name(input_name.to_string()),
        }
    }
    pub fn index(&self) -> Option<u32> {
        match self {
            TensorId::Index(i) => Some(*i),
            TensorId::Name(_) => None,
        }
    }
    pub fn name(&self) -> Option<&str> {
        match self {
            TensorId::Index(_) => None,
            TensorId::Name(n) => Some(n),
        }
    }
}

/*
/// Errors returned by a backend; [BackendError::BackendAccess] is a catch-all
/// for failures interacting with the ML library.
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Failed while accessing backend")]
    BackendAccess(#[from] anyhow::Error),
    #[error("Failed while accessing guest module")]
    GuestAccess(#[from] GuestError),
    #[error("The backend expects {0} buffers, passed {1}")]
    InvalidNumberOfBuilders(usize, usize),
    #[error("Not enough memory to copy tensor data of size: {0}")]
    NotEnoughMemory(usize),
    #[error("Unsupported tensor type: {0}")]
    UnsupportedTensorType(String),
}
*/
