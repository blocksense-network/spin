use spin_world::v2::graph::{ExecutionTarget, GraphEncoding};
//use ml_wit::graph::{, Graph, GraphBuilder, GraphEncoding};

use super::BackendInner;
use crate::backend::Graph;

#[derive(Default)]
pub struct OpenvinoBackend {
    pub openvino: Option<openvino::Core>,
}
unsafe impl Send for OpenvinoBackend {}
unsafe impl Sync for OpenvinoBackend {}

impl BackendInner for OpenvinoBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Openvino
    }

    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, String> {
        Err("not implemented".to_owned())
    }
}
