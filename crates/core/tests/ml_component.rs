
pub mod ml {
    wasmtime::component::bindgen!("ml" in "tests/core-wasi-test/wit");

    use spin_core::HostComponent;

    use anyhow::anyhow;
    use test::test::errors;
    use test::test::graph;
    use test::test::inference;
    use test::test::tensor;

    use test::test::errors::ErrorCode;
    use test::test::graph::{ExecutionTarget, Graph, GraphBuilder, GraphEncoding};
    use test::test::inference::GraphExecutionContext;
    use wasmtime::component::Resource;
    use tokio::sync::Mutex;
    
    use table;
    use openvino::{Core, InferenceError, Layout, Precision, SetupError, TensorDesc};

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
            MLHostImpl { ..Default::default()}
        }
    }

    pub struct GraphInternalData {
        pub xml: Vec<u8>,
        pub weights: Vec<u8>,
        pub target: ExecutionTarget,
    }


    pub struct GraphExecutionContextInternalData {
        cnn_network: openvino::CNNNetwork,
        executable_network: Mutex<openvino::ExecutableNetwork>,
    }

    #[derive(Default)]
    pub struct MLHostImpl {
        pub openvino: Option<openvino::Core>,
        pub graphs: table::Table<GraphInternalData>,
        pub execution_contexts: table::Table<GraphExecutionContextInternalData>,
    }

    impl graph::HostGraph for MLHostImpl {
        fn init_execution_context(
            &mut self,
            graph: Resource<Graph>,
        ) -> Result<
            Result<Resource<inference::GraphExecutionContext>, Resource<errors::Error>>,
            anyhow::Error,
        > {
            // Construct the context if none is present; this is done lazily (i.e.
            // upon actually loading a model) because it may fail to find and load
            // the OpenVINO libraries. The laziness limits the extent of the error
            // only to wasi-nn users, not all WASI users.
            if self.openvino.is_none() {
                self.openvino.replace(openvino::Core::new(None)?);
            }
            if self.openvino.is_some() { 
                if let Some(graph) = self.graphs.get(graph.rep()) {
                    let mut cnn_network = self.openvino.as_mut().expect("").read_network_from_buffer(&graph.xml, &graph.weights)?;
    
                // Construct OpenVINO graph structures: `cnn_network` contains the graph
                // structure, `exec_network` can perform inference.
                //let core = self
                //    .0
                //    .as_mut()
                //    .expect("openvino::Core was previously constructed");
                //let mut cnn_network = core.read_network_from_buffer(&xml, &weights)?;
    
                // TODO: this is a temporary workaround. We need a more elegant way to
                // specify the layout in the long run. However, without this newer
                // versions of OpenVINO will fail due to parameter mismatch.
                for i in 0..cnn_network.get_inputs_len().unwrap() {
                    let name = cnn_network.get_input_name(i)?;
                    cnn_network.set_input_layout(&name, Layout::NHWC)?;
                };
    
                let exec_network = self.openvino.as_mut().expect("").load_network(&cnn_network, map_execution_target_to_string(graph.target))?;

                let graph_execution_context = GraphExecutionContextInternalData {
                        cnn_network: cnn_network, 
                        executable_network: Mutex::new(exec_network)
                };

                if let Ok(graph_execution_context_id) = self.execution_contexts.push(graph_execution_context).map(Resource::<inference::GraphExecutionContext>::new_own){
                    println!("{:?}", graph_execution_context_id);
                    return Ok(Ok(graph_execution_context_id))
                };
                
                //.ok_or(sqlite::Error::InvalidConnection);
            /*let box_: Box<dyn BackendGraph> = Box::new(OpenvinoGraph(
                Arc::new(cnn_network),
                Arc::new(Mutex::new(exec_network)),
            ));
            Ok(box_.into())
            */
                }
            }
            Err(anyhow!("[graph::HostGraph] fn init_execution_context -> Not implemented"))
        }
        fn drop(&mut self, _graph: Resource<Graph>) -> Result<(), anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
    }

    impl errors::HostError for MLHostImpl {
        fn new(
            &mut self,
            _code: errors::ErrorCode,
            _data: String,
        ) -> Result<Resource<errors::Error>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }

        fn drop(&mut self, _error: Resource<errors::Error>) -> Result<(), anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }

        fn code(&mut self, _error: Resource<errors::Error>) -> Result<ErrorCode, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }

        fn data(&mut self, _error: Resource<errors::Error>) -> Result<String, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
    }
    impl tensor::HostTensor for MLHostImpl {
        fn new(
            &mut self,
            _tensor_dimentions: tensor::TensorDimensions,
            _tensor_type: tensor::TensorType,
            _tensor_data: tensor::TensorData,
        ) -> Result<Resource<tensor::Tensor>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn dimensions(
            &mut self,
            _tensor: Resource<tensor::Tensor>,
        ) -> Result<Vec<u32>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn ty(
            &mut self,
            _tensor: Resource<tensor::Tensor>,
        ) -> Result<tensor::TensorType, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn data(&mut self, _tensor: Resource<tensor::Tensor>) -> Result<tensor::TensorData, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn drop(&mut self, _tensor: Resource<tensor::Tensor>) -> Result<(), anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
    }

    impl inference::HostGraphExecutionContext for MLHostImpl {
        fn set_input(
            &mut self,
            _graph_execution_context: Resource<GraphExecutionContext>,
            _input_name: String,
            _tensor: Resource<tensor::Tensor>,
        ) -> Result<Result<(), Resource<errors::Error>>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn compute(
            &mut self,
            _graph_execution_context: Resource<GraphExecutionContext>,
        ) -> Result<Result<(), Resource<errors::Error>>, anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
        fn get_output(
            &mut self,
            _graph_execution_context: Resource<GraphExecutionContext>,
            _input_name: String,
        ) -> Result<Result<Resource<tensor::Tensor>, Resource<errors::Error>>, anyhow::Error>
        {
            Err(anyhow!("Not implemented"))
        }
        fn drop(
            &mut self,
            _graph_execution_context: Resource<GraphExecutionContext>,
        ) -> Result<(), anyhow::Error> {
            Err(anyhow!("Not implemented"))
        }
    }

    impl errors::Host for MLHostImpl {}
    impl graph::Host for MLHostImpl {
        fn load(
            &mut self,
            graph: Vec<GraphBuilder>,
            graph_encoding: GraphEncoding,
            target: ExecutionTarget,
        ) -> Result<Result<Resource<Graph>, Resource<errors::Error>>, anyhow::Error> {
            println!("[graph::Host] load entry point");
            if graph.len() != 2 {
                return Err(anyhow!("Expected 2 elements in graph builder vector"))
            }
            if graph_encoding != GraphEncoding::Openvino {
                return Err(anyhow!("Only OpenVINO encoding is supported"))
            }
            // Read the guest array.
            let graph_internal_data = GraphInternalData { xml: graph[0].clone(), weights: graph[1].clone(), target: target };
            if let Ok(graph_id) = self.graphs.push(graph_internal_data).map(Resource::<Graph>::new_own){
                println!("{:?}", graph_id);
                return Ok(Ok(graph_id))
            };//.ok_or(sqlite::Error::InvalidConnection);
            
            Err(anyhow!("[graph::Host] fn load -> Not implemented"))
        }
        fn load_by_name(
            &mut self,
            _graph: String,
        ) -> Result<Result<Resource<Graph>, Resource<errors::Error>>, anyhow::Error> {
            Err(anyhow!("[graph::Host] fn load_by_name -> Not implemented"))
        }
    }

    impl inference::Host for MLHostImpl {}
    impl tensor::Host for MLHostImpl {}



    /// Return the execution target string expected by OpenVINO from the
    /// `ExecutionTarget` enum provided by wasi-nn.
    fn map_execution_target_to_string(target: ExecutionTarget) -> &'static str {
        match target {
            ExecutionTarget::Cpu => "CPU",
            ExecutionTarget::Gpu => "GPU",
            ExecutionTarget::Tpu => unimplemented!("OpenVINO does not support TPU execution targets"),
        }
    }

    /// Return OpenVINO's precision type for the `TensorType` enum provided by
    /// wasi-nn.
    fn map_tensor_type_to_precision(tensor_type: tensor::TensorType) -> openvino::Precision {
        match tensor_type {
            tensor::TensorType::Fp16 => Precision::FP16,
            tensor::TensorType::Fp32 => Precision::FP32,
            tensor::TensorType::Fp64 => Precision::FP64,
            tensor::TensorType::U8 => Precision::U8,
            tensor::TensorType::I32 => Precision::I32,
            tensor::TensorType::I64 => Precision::I64,
            tensor::TensorType::Bf16 => todo!("not yet supported in `openvino` bindings"),
        }
    }
}
