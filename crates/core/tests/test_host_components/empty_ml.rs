#[allow(clippy::module_inception)]
pub mod ml {
    wasmtime::component::bindgen!("ml" in "tests/core-wasi-test/wit");

    use spin_core::HostComponent;

    use anyhow::{anyhow, Context};
    use test::test as ml_wit;

    use ml_wit::{errors, graph, inference, tensor};

    use ml_wit::errors::ErrorCode;
    use ml_wit::graph::{ExecutionTarget, Graph, GraphBuilder, GraphEncoding};
    use ml_wit::inference::GraphExecutionContext;

    use wasmtime::component::Resource;

    #[derive(Clone)]
    pub struct MLHostComponent;

    impl HostComponent for MLHostComponent {
        type Data = MLHostImpl;

        fn add_to_linker<T: Send>(
            linker: &mut spin_core::Linker<T>,
            get: impl Fn(&mut spin_core::Data<T>) -> &mut Self::Data + Send + Sync + Copy + 'static,
        ) -> anyhow::Result<()> {
            Ml::add_to_linker(linker, get)
        }

        fn build_data(&self) -> Self::Data {
            MLHostImpl {
                ..Default::default()
            }
        }
    }

    pub struct ErrorInternalData {
        code: errors::ErrorCode,
        message: String,
    }

    #[derive(Default)]
    pub struct MLHostImpl {
        pub errors: table::Table<ErrorInternalData>,
    }

    impl MLHostImpl {
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
    }

    impl graph::HostGraph for MLHostImpl {
        fn init_execution_context(
            &mut self,
            _graph: Resource<Graph>,
        ) -> Result<Resource<inference::GraphExecutionContext>, Resource<errors::Error>> {
            Err(MLHostImpl::new_error(
                &mut self.errors,
                ErrorCode::UnsupportedOperation,
                "Not compliled with openvino support".to_string(),
            ))
        }

        fn drop(&mut self, _graph: Resource<Graph>) -> Result<(), anyhow::Error> {
            Err(anyhow!("Not compliled with openvino support"))
        }
    }

    impl errors::HostError for MLHostImpl {
        fn new(&mut self, code: errors::ErrorCode, message: String) -> Resource<errors::Error> {
            MLHostImpl::new_error(&mut self.errors, code, message)
        }

        fn drop(
            &mut self,
            error: Resource<errors::Error>,
        ) -> std::result::Result<(), anyhow::Error> {
            self.errors
                .remove(error.rep())
                .context(format!("Can't find error with ID = {}", error.rep()))
                .map(|_| ())
        }

        fn code(&mut self, error: Resource<errors::Error>) -> ErrorCode {
            if let Some(e) = self.errors.get(error.rep()) {
                return e.code;
            }
            panic!("Can't find error with ID = {}", error.rep());
        }

        fn data(&mut self, error: Resource<errors::Error>) -> String {
            if let Some(e) = self.errors.get(error.rep()) {
                return e.message.clone();
            }
            panic!("Can't find error with ID = {}", error.rep());
        }
    }
    impl tensor::HostTensor for MLHostImpl {
        fn new(
            &mut self,
            _tensor_dimensions: tensor::TensorDimensions,
            _tensor_type: tensor::TensorType,
            _tensor_data: tensor::TensorData,
        ) -> Resource<tensor::Tensor> {
            panic!("Not compliled with openvino support");
        }
        fn dimensions(&mut self, tensor: Resource<tensor::Tensor>) -> Vec<u32> {
            panic!("Can't find tensor with ID = {}", tensor.rep());
        }

        fn ty(&mut self, tensor: Resource<tensor::Tensor>) -> tensor::TensorType {
            panic!("Can't find tensor with ID = {}", tensor.rep());
        }

        fn data(&mut self, tensor: Resource<tensor::Tensor>) -> tensor::TensorData {
            panic!("Can't find tensor with ID = {}", tensor.rep());
        }
        fn drop(
            &mut self,
            tensor: Resource<tensor::Tensor>,
        ) -> std::result::Result<(), anyhow::Error> {
            panic!("Can't find tensor with ID = {}", tensor.rep());
        }
    }

    impl inference::HostGraphExecutionContext for MLHostImpl {
        fn set_input(
            &mut self,
            _graph_execution_context: Resource<GraphExecutionContext>,
            _input_name: String,
            tensor: Resource<tensor::Tensor>,
        ) -> Result<(), Resource<errors::Error>> {
            panic!("Can't find tensor with ID = {}", tensor.rep());
        }

        fn compute(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
        ) -> Result<(), Resource<errors::Error>> {
            Err(MLHostImpl::new_error(
                &mut self.errors,
                ErrorCode::RuntimeError,
                format!(
                    "Can't find graph execution context with ID = {}",
                    graph_execution_context.rep()
                ),
            ))
        }

        fn get_output(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
            _input_name: String,
        ) -> Result<Resource<tensor::Tensor>, Resource<errors::Error>> {
            Err(MLHostImpl::new_error(
                &mut self.errors,
                ErrorCode::RuntimeError,
                format!(
                    "Can't find graph execution context with ID = {}",
                    graph_execution_context.rep()
                ),
            ))
        }

        fn drop(
            &mut self,
            execution: Resource<GraphExecutionContext>,
        ) -> std::result::Result<(), anyhow::Error> {
            Err(anyhow!(
                "Can't drow GraphExecutionContext with id = {}",
                execution.rep()
            ))
        }
    }

    impl errors::Host for MLHostImpl {}
    impl graph::Host for MLHostImpl {
        fn load(
            &mut self,
            _graph: Vec<GraphBuilder>,
            _graph_encoding: GraphEncoding,
            _target: ExecutionTarget,
        ) -> Result<Resource<Graph>, Resource<errors::Error>> {
            Err(MLHostImpl::new_error(
                &mut self.errors,
                ErrorCode::RuntimeError,
                "Expected 2 elements in graph builder vector".to_string(),
            ))
        }
        fn load_by_name(
            &mut self,
            _graph: String,
        ) -> Result<Resource<Graph>, Resource<errors::Error>> {
            panic!("[graph::Host] fn load_by_name -> Not implemented");
        }
    }

    impl inference::Host for MLHostImpl {}
    impl tensor::Host for MLHostImpl {}
}
