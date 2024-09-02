use crate::test_host_components::ml_host_impl;
use crate::test_host_components::ml_wit::test::test as ml_wit;

use ml_host_impl::{ExecutionContext, GraphInternalData, TensorInternalData};
use ml_wit::graph::{ExecutionTarget, GraphBuilder, GraphEncoding};

#[cfg(feature = "openvino")]
pub mod openvino;

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
}
