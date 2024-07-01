pub mod ml {
    wasmtime::component::bindgen!("ml" in "tests/core-wasi-test/wit");

    use spin_core::HostComponent;

    use anyhow::{anyhow, Context};
    use test::test::errors;
    use test::test::errors::HostError;
    use test::test::graph;
    use test::test::inference;
    use test::test::tensor;

    use test::test::errors::ErrorCode;
    use test::test::graph::{ExecutionTarget, Graph, GraphBuilder, GraphEncoding};
    use test::test::inference::GraphExecutionContext;
    use wasmtime::component::Resource;

    use openvino::{Layout, Precision, TensorDesc};

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
    #[derive(Debug)]
    pub struct GraphInternalData {
        pub xml: Vec<u8>,
        pub weights: Vec<u8>,
        pub target: ExecutionTarget,
    }

    pub struct GraphExecutionContextInternalData {
        pub cnn_network: openvino::CNNNetwork,
        //pub executable_network: Mutex<openvino::ExecutableNetwork>,
        pub infer_request: openvino::InferRequest,
    }

    pub struct TensorInternalData {
        tensor_dimensions: tensor::TensorDimensions,
        tensor_type: tensor::TensorType,
        tensor_data: tensor::TensorData,
    }

    pub struct ErrorInternalData {
        code: errors::ErrorCode,
        message: String,
    }

    #[derive(Default)]
    pub struct MLHostImpl {
        pub openvino: Option<openvino::Core>,
        pub graphs: table::Table<GraphInternalData>,
        pub executions: table::Table<GraphExecutionContextInternalData>,
        pub tensors: table::Table<TensorInternalData>,
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

        fn init_execution_context_internal(
            graph: &GraphInternalData,
            openvino: &mut Option<openvino::Core>,
            executions: &mut table::Table<GraphExecutionContextInternalData>,
        ) -> Result<Resource<inference::GraphExecutionContext>, anyhow::Error> {
            if openvino.is_none() {
                openvino.replace(openvino::Core::new(None)?);
            }
            if openvino.is_some() {
                let mut cnn_network = openvino
                    .as_mut()
                    .context("Can't create openvino graph without backend")?
                    .read_network_from_buffer(&graph.xml, &graph.weights)?;

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
                }

                let mut exec_network = openvino
                    .as_mut()
                    .expect("")
                    .load_network(&cnn_network, map_execution_target_to_string(graph.target))?;
                let infer_request = exec_network
                    .create_infer_request()
                    .context("Can't create InferRequest")?;
                let graph_execution_context = GraphExecutionContextInternalData {
                    cnn_network,
                    //executable_network: Mutex::new(exec_network),
                    infer_request,
                };
                return executions
                    .push(graph_execution_context)
                    .map(Resource::<inference::GraphExecutionContext>::new_own)
                    .map_err(|_| anyhow!("Can't store execution context"));
            }
            Err(anyhow!("Can't create openvino backend"))
        }

        fn get_output_internal(
            graph_execution: &mut GraphExecutionContextInternalData,
            input_name: String,
        ) -> Result<TensorInternalData, String> {
            let index = input_name.parse::<usize>().map_err(|err| {
                format!(
                    "Can't parse {} to usize for input_name, err = {err}",
                    input_name
                )
            })?;
            let output_name = graph_execution
                .cnn_network
                .get_output_name(index)
                .map_err(|err| format!("Can't find output name for ID = {index}, err = {err}"))?;

            let blob = graph_execution
                .infer_request
                .get_blob(&output_name)
                .map_err(|err| {
                    format!("Can't get blob for output name = {output_name}, err = {err}")
                })?;
            let tensor_desc = blob
                .tensor_desc()
                .map_err(|err| format!("Can't get blob description, err = {err}"))?;
            let buffer = blob
                .buffer()
                .map_err(|err| format!("Can't get blob buffer, error = {err}"))?
                .to_vec();
            let tensor_dimensions = tensor_desc
                .dims()
                .iter()
                .map(|&d| d as u32)
                .collect::<Vec<_>>();
            let tensor = TensorInternalData {
                tensor_dimensions,
                tensor_type: map_precision_to_tensor_type(tensor_desc.precision()),
                tensor_data: buffer,
            };
            Ok(tensor)
        }
    }

    impl graph::HostGraph for MLHostImpl {
        fn init_execution_context(
            &mut self,
            graph: Resource<Graph>,
        ) -> Result<Resource<inference::GraphExecutionContext>, Resource<errors::Error>> {
            let res = match self.graphs.get(graph.rep()) {
                Some(graph) => MLHostImpl::init_execution_context_internal(
                    graph,
                    &mut self.openvino,
                    &mut self.executions,
                )
                .map_err(|err| ErrorInternalData {
                    code: ErrorCode::RuntimeError,
                    message: err.to_string(),
                }),
                None => Err(ErrorInternalData {
                    code: ErrorCode::RuntimeError,
                    message: "Can't create graph execution context".to_string(),
                }),
            };
            match res {
                Ok(res) => Ok(res),
                Err(e) => Err(MLHostImpl::new_error(&mut self.errors, e.code, e.message)),
            }
        }

        fn drop(&mut self, graph: Resource<Graph>) -> Result<(), anyhow::Error> {
            self.graphs
                .remove(graph.rep())
                .context(format!("Can't find graph with ID = {}", graph.rep()))
                .map(|_| ())
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
            tensor_dimensions: tensor::TensorDimensions,
            tensor_type: tensor::TensorType,
            tensor_data: tensor::TensorData,
        ) -> Resource<tensor::Tensor> {
            let tensor = TensorInternalData {
                tensor_dimensions,
                tensor_type,
                tensor_data,
            };
            self.tensors
                .push(tensor)
                .map(Resource::<tensor::Tensor>::new_own)
                .expect("Can't allocate tensor")
        }
        fn dimensions(&mut self, tensor: Resource<tensor::Tensor>) -> Vec<u32> {
            if let Some(t) = self.tensors.get(tensor.rep()) {
                return t.tensor_dimensions.clone();
            }
            panic!("Can't find tensor with ID = {}", tensor.rep());
        }

        fn ty(&mut self, tensor: Resource<tensor::Tensor>) -> tensor::TensorType {
            if let Some(t) = self.tensors.get(tensor.rep()) {
                return t.tensor_type;
            }
            panic!("Can't find tensor with ID = {}", tensor.rep());
        }

        fn data(&mut self, tensor: Resource<tensor::Tensor>) -> tensor::TensorData {
            if let Some(t) = self.tensors.get(tensor.rep()) {
                return t.tensor_data.clone();
            }
            panic!("Can't find tensor with ID = {}", tensor.rep());
        }
        fn drop(
            &mut self,
            tensor: Resource<tensor::Tensor>,
        ) -> std::result::Result<(), anyhow::Error> {
            self.tensors
                .remove(tensor.rep())
                .context(format!("Can't find tensor with ID = {}", tensor.rep()))
                .map(|_| ())
        }
    }

    impl inference::HostGraphExecutionContext for MLHostImpl {
        fn set_input(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
            input_name: String,
            tensor: Resource<tensor::Tensor>,
        ) -> Result<(), Resource<errors::Error>> {
            let index = input_name
                .parse()
                .expect("Can't parse {} to usize for input_name");
            // Construct the blob structure. TODO: there must be some good way to
            // discover the layout here; `desc` should not have to default to NHWC.
            let tensor_resource = self
                .tensors
                .get(tensor.rep())
                .unwrap_or_else(|| panic!("Can't find tensor with ID = {}", tensor.rep()));
            let precision = map_tensor_type_to_precision(tensor_resource.tensor_type);
            let dimensions = tensor_resource
                .tensor_dimensions
                .iter()
                .map(|&d| d as usize)
                .collect::<Vec<_>>();
            let desc = TensorDesc::new(Layout::NHWC, &dimensions, precision);
            let blob = openvino::Blob::new(&desc, &tensor_resource.tensor_data)
                .expect("Error in Blob::new");
            let execution_context: &mut GraphExecutionContextInternalData = self
                .executions
                .get_mut(graph_execution_context.rep())
                .unwrap_or_else(|| panic!("Can't find tensor with ID = {}", tensor.rep()));
            let input_name = execution_context
                .cnn_network
                .get_input_name(index)
                .unwrap_or_else(|_| panic!("Can't find input with name = {}", index));
            match execution_context.infer_request.set_blob(&input_name, &blob) {
                Ok(res) => Ok(res),
                Err(err) => Err(self.new(
                    ErrorCode::RuntimeError,
                    format!("Inference error = {:?}", err.to_string()),
                )),
            }
        }

        fn compute(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
        ) -> Result<(), Resource<errors::Error>> {
            let graph_execution = self
                .executions
                .get_mut(graph_execution_context.rep())
                .ok_or(MLHostImpl::new_error(
                    &mut self.errors,
                    ErrorCode::RuntimeError,
                    format!(
                        "Can't find graph execution context with ID = {}",
                        graph_execution_context.rep()
                    ),
                ))?;
            match graph_execution.infer_request.infer() {
                Ok(..) => Ok(()),
                Err(err) => Err(MLHostImpl::new_error(
                    &mut self.errors,
                    ErrorCode::RuntimeError,
                    format!("Inference error = {:?}", err.to_string()),
                )),
            }
        }

        fn get_output(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
            input_name: String,
        ) -> Result<Resource<tensor::Tensor>, Resource<errors::Error>> {
            let graph_execution = self
                .executions
                .get_mut(graph_execution_context.rep())
                .ok_or(format!(
                    "Can't find graph execution context with ID = {}",
                    graph_execution_context.rep()
                ))
                .unwrap();

            match MLHostImpl::get_output_internal(graph_execution, input_name) {
                Ok(tensor) => self
                    .tensors
                    .push(tensor)
                    .map(Resource::<tensor::Tensor>::new_own)
                    .map_err(|_| {
                        MLHostImpl::new_error(
                            &mut self.errors,
                            ErrorCode::RuntimeError,
                            "Can't create tensor for get_output".to_string(),
                        )
                    }),
                Err(err) => Err(MLHostImpl::new_error(
                    &mut self.errors,
                    ErrorCode::RuntimeError,
                    err,
                )),
            }
        }

        fn drop(
            &mut self,
            execution: Resource<GraphExecutionContext>,
        ) -> std::result::Result<(), anyhow::Error> {
            let id = execution.rep();
            self.executions
                .remove(id)
                .context("{Can't drow GraphExecutionContext with id = {id}")
                .map(|_| ())
        }
    }

    impl errors::Host for MLHostImpl {}
    impl graph::Host for MLHostImpl {
        fn load(
            &mut self,
            graph: Vec<GraphBuilder>,
            graph_encoding: GraphEncoding,
            target: ExecutionTarget,
        ) -> Result<Resource<Graph>, Resource<errors::Error>> {
            if graph.len() != 2 {
                return Err(MLHostImpl::new_error(
                    &mut self.errors,
                    ErrorCode::RuntimeError,
                    "Expected 2 elements in graph builder vector".to_string(),
                ));
            }
            if graph_encoding != GraphEncoding::Openvino {
                return Err(MLHostImpl::new_error(
                    &mut self.errors,
                    ErrorCode::RuntimeError,
                    "Only OpenVINO encoding is supported".to_string(),
                ));
            }
            // Read the guest array.
            let graph_internal_data = GraphInternalData {
                xml: graph[0].clone(),
                weights: graph[1].clone(),
                target,
            };
            match self.graphs.push(graph_internal_data) {
                Ok(graph_rep) => Ok(Resource::<Graph>::new_own(graph_rep)),
                Err(err) => {
                    match self.errors.push(ErrorInternalData {
                        code: ErrorCode::RuntimeError,
                        message: format!("{:?}", err),
                    }) {
                        Ok(error_rep) => Err(Resource::<errors::Error>::new_own(error_rep)),
                        Err(err) => {
                            panic!("Can't create internal error for {:?}", err);
                        }
                    }
                }
            }
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

    /// Return the execution target string expected by OpenVINO from the
    /// `ExecutionTarget` enum provided by wasi-nn.
    fn map_execution_target_to_string(target: ExecutionTarget) -> &'static str {
        match target {
            ExecutionTarget::Cpu => "CPU",
            ExecutionTarget::Gpu => "GPU",
            ExecutionTarget::Tpu => {
                unimplemented!("OpenVINO does not support TPU execution targets")
            }
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
    fn map_precision_to_tensor_type(precision: openvino::Precision) -> tensor::TensorType {
        match precision {
            Precision::FP16 => tensor::TensorType::Fp16,
            Precision::FP32 => tensor::TensorType::Fp32,
            Precision::FP64 => tensor::TensorType::Fp64,
            Precision::U8 => tensor::TensorType::U8,
            Precision::I32 => tensor::TensorType::I32,
            Precision::I64 => tensor::TensorType::I64,
            _ => todo!("not yet supported in `openvino` bindings"),
        }
    }
}
