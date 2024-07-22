//#[cfg(feature = "openvino")]
pub mod openvino;
//pub mod onnx;

use spin_world::v2 as ml_wit;

use ml_wit::graph::{ExecutionTarget, GraphBuilder, GraphEncoding};
use ml_wit::inference::GraphExecutionContext;
use ml_wit::tensor;

use crate::host_impl::{GraphInternalData, TensorInternalData, ExecutionContext};


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
    //fn as_dir_loadable<'a>(&'a mut self) -> Option<&'a mut dyn BackendFromDir>;

    fn init_execution_context(
        &mut self,
        graph: &GraphInternalData,
    ) -> Result<ExecutionContext, anyhow::Error>;
}

/// A [BackendGraph] can create [BackendExecutionContext]s; this is the backing
/// implementation for the user-facing graph.
pub trait BackendGraph: Send + Sync {
    fn init_execution_context(&self) -> Result<ExecutionContext, anyhow::Error>;
}

pub trait BackendExecutionContext: Send + Sync {
    fn set_input(
        &mut self,
        input_name: String,
        tensor: &TensorInternalData,
    ) -> Result<(), anyhow::Error>;

    fn compute(&mut self) -> Result<(), anyhow::Error>;

    fn get_output(&mut self, input_name: String) -> Result<TensorInternalData, anyhow::Error>;
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
