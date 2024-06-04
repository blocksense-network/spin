pub mod Ml {
    wasmtime::component::bindgen!("ml" in "tests/core-wasi-test/wit");

    use spin_core::HostComponent;

    use anyhow::anyhow;
    use test::test::errors;
    use test::test::graph;
    use test::test::inference;
    use test::test::tensor;

    use test::test::errors::ErrorCode;
    use test::test::graph::{Error, ExecutionTarget, Graph, GraphBuilder, GraphEncoding};
    use test::test::inference::GraphExecutionContext;
    use wasmtime::component::Resource;

    #[derive(Clone)]
    pub struct NNHostComponent;

    impl HostComponent for NNHostComponent {
        type Data = MLHostImpl;

        fn add_to_linker<T: Send>(
            linker: &mut spin_core::Linker<T>,
            get: impl Fn(&mut spin_core::Data<T>) -> &mut Self::Data + Send + Sync + Copy + 'static,
        ) -> anyhow::Result<()> {
            Ml::add_to_linker(linker, get)
        }

        fn build_data(&self) -> Self::Data {
            MLHostImpl {}
        }
    }

    pub struct MLHostImpl {}

    impl graph::HostGraph for MLHostImpl {
        fn init_execution_context(
            &mut self,
            graph: Resource<Graph>,
        ) -> Result<
            Result<Resource<inference::GraphExecutionContext>, Resource<errors::Error>>,
            anyhow::Error,
        > {
            Err(anyhow!("Not implemented"))
        }
        fn drop(&mut self, graph: Resource<Graph>) -> Result<(), anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
    }

    impl errors::HostError for MLHostImpl {
        fn new(
            &mut self,
            code: errors::ErrorCode,
            data: String,
        ) -> Result<Resource<errors::Error>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }

        fn drop(&mut self, err: Resource<errors::Error>) -> Result<(), anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }

        fn code(&mut self, err: Resource<errors::Error>) -> Result<ErrorCode, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }

        fn data(&mut self, err: Resource<errors::Error>) -> Result<String, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
    }
    impl tensor::HostTensor for MLHostImpl {
        fn new(
            &mut self,
            tensor_dimentions: Vec<u32>,
            tensor_type: tensor::TensorType,
            tensor_data: Vec<u8>,
        ) -> Result<Resource<tensor::Tensor>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn dimensions(
            &mut self,
            tensor: Resource<tensor::Tensor>,
        ) -> Result<Vec<u32>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn ty(
            &mut self,
            tensor: Resource<tensor::Tensor>,
        ) -> Result<tensor::TensorType, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn data(&mut self, tensor: Resource<tensor::Tensor>) -> Result<Vec<u8>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn drop(&mut self, tensor: Resource<tensor::Tensor>) -> Result<(), anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
    }

    impl inference::HostGraphExecutionContext for MLHostImpl {
        fn set_input(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
            name: String,
            val2: Resource<tensor::Tensor>,
        ) -> Result<Result<(), Resource<errors::Error>>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn compute(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
        ) -> Result<Result<(), Resource<errors::Error>>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn get_output(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
            name: String,
        ) -> Result<Result<Resource<tensor::Tensor>, Resource<errors::Error>>, anyhow::Error>
        {
            Err(anyhow!("Not implemented"))
        }
        fn drop(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
        ) -> Result<(), anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
    }

    impl errors::Host for MLHostImpl {}
    impl graph::Host for MLHostImpl {
        fn load(
            &mut self,
            graph: Vec<Vec<u8>>,
            graph_encoding: GraphEncoding,
            target: ExecutionTarget,
        ) -> Result<Result<Resource<Graph>, Resource<errors::Error>>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn load_by_name(
            &mut self,
            graph: String,
        ) -> Result<Result<Resource<Graph>, Resource<errors::Error>>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
    }

    impl inference::Host for MLHostImpl {}
    impl tensor::Host for MLHostImpl {}
}
