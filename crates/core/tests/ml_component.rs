
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
    use tokio::sync::Mutex;
    
    use table;
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
            MLHostImpl { ..Default::default()}
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
        pub executable_network: Mutex<openvino::ExecutableNetwork>,
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
    
                    let mut exec_network = self.openvino.as_mut().expect("").load_network(&cnn_network, map_execution_target_to_string(graph.target))?;
                    let infer_request = exec_network.create_infer_request().expect("Can't create InferRequest");
                    let graph_execution_context = GraphExecutionContextInternalData {
                        cnn_network: cnn_network, 
                        executable_network: Mutex::new(exec_network),
                        infer_request: infer_request,
                    };

                    match self.executions.push(graph_execution_context).map(Resource::<inference::GraphExecutionContext>::new_own) {
                        Ok(res) => {
                            return Ok(Ok(res));
                        }
                        Err(_) => {
                            match self.errors.push(ErrorInternalData { code: ErrorCode::RuntimeError, message: "Can't create graph execution context".to_string()}){
                                Ok(id) => {
                                    return Ok(Err(Resource::<errors::Error>::new_own(id)));
                                }
                                Err(_) => {
                                    return Err(anyhow!("Can't allocate error"));
                                }
                            }
                        }
                    }
                
                }
            }
            Err(anyhow!("[graph::HostGraph] fn init_execution_context -> Not implemented"))
        }
        fn drop(&mut self, graph: Resource<Graph>) -> Result<(), anyhow::Error> {
            let _v = self.graphs.remove(graph.rep());
            Ok(())
        }
    }

    impl errors::HostError for MLHostImpl {
        fn new(
            &mut self,
            code: errors::ErrorCode,
            data: String,
        ) -> Result<Resource<errors::Error>, anyhow::Error> {
            self.errors.push(ErrorInternalData { code: code, message:data}).map(Resource::<errors::Error>::new_own).map_err(|_|anyhow!("Can't allocate error"))
        }

        fn drop(&mut self, error: Resource<errors::Error>) -> Result<(), anyhow::Error> {
            self.errors.remove(error.rep()).ok_or(anyhow!(format!("Can't find error with ID = {}", error.rep()))).map(|_| ())
        }

        fn code(&mut self, error: Resource<errors::Error>) -> Result<ErrorCode, anyhow::Error> {
            self.errors.get(error.rep()).ok_or(anyhow!(format!("Can't find error with ID = {}", error.rep()))).map(|e| e.code)
        }

        fn data(&mut self, error: Resource<errors::Error>) -> Result<String, anyhow::Error> {
            self.errors.get(error.rep()).ok_or(anyhow!(format!("Can't find error with ID = {}", error.rep()))).map(|e| e.message.clone())

        }
    }
    impl tensor::HostTensor for MLHostImpl {
        fn new(
            &mut self,
            tensor_dimensions: tensor::TensorDimensions,
            tensor_type: tensor::TensorType,
            tensor_data: tensor::TensorData,
        ) -> Result<Resource<tensor::Tensor>, anyhow::Error> {
            let tensor = TensorInternalData {
                tensor_dimensions: tensor_dimensions,
                tensor_type: tensor_type,
                tensor_data: tensor_data,
            };
            self.tensors.push(tensor).map(Resource::<tensor::Tensor>::new_own).map_err(|_|anyhow!("Can't allocate tensor"))
        }
        fn dimensions(
            &mut self,
            tensor: Resource<tensor::Tensor>,
        ) -> Result<Vec<u32>, anyhow::Error> {
            self.tensors.get(tensor.rep()).ok_or(anyhow!(format!("Can't find tensor with ID = {}", tensor.rep()))).map(|t| t.tensor_dimensions.clone())
        }
        fn ty(
            &mut self,
            tensor: Resource<tensor::Tensor>,
        ) -> Result<tensor::TensorType, anyhow::Error> {
            self.tensors.get(tensor.rep()).ok_or(anyhow!(format!("Can't find tensor with ID = {}", tensor.rep()))).map(|t| t.tensor_type)
        }
        fn data(&mut self, tensor: Resource<tensor::Tensor>) -> Result<tensor::TensorData, anyhow::Error> {
            self.tensors.get(tensor.rep()).ok_or(anyhow!(format!("Can't find tensor with ID = {}", tensor.rep()))).map(|t| t.tensor_data.clone())
        }
        fn drop(&mut self, tensor: Resource<tensor::Tensor>) -> Result<(), anyhow::Error> {
            self.tensors.remove(tensor.rep()).ok_or(anyhow!(format!("Can't find tensor with ID = {}", tensor.rep()))).map(|_| ())
        }
    }

    impl inference::HostGraphExecutionContext for MLHostImpl {
        fn set_input(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
            input_name: String,
            tensor: Resource<tensor::Tensor>,
        ) -> Result<Result<(), Resource<errors::Error>>, anyhow::Error> {
            let index = input_name.parse().context("Can't parse {} to usize for input_name")?;
            // Construct the blob structure. TODO: there must be some good way to
            // discover the layout here; `desc` should not have to default to NHWC.
            let tensor_resource = self.tensors.get(tensor.rep()).ok_or(anyhow!(format!("Can't find tensor with ID = {}", tensor.rep())))?;
            let precision = map_tensor_type_to_precision(tensor_resource.tensor_type);
            let dimensions = tensor_resource.tensor_dimensions
                .iter()
                .map(|&d| d as usize)
                .collect::<Vec<_>>();
            let desc = TensorDesc::new(Layout::NHWC, &dimensions, precision);
            let blob = openvino::Blob::new(&desc, &tensor_resource.tensor_data)?;
            let execution_context: &mut GraphExecutionContextInternalData = self.executions.get_mut(graph_execution_context.rep()).context(format!("Can't find graph execution context with ID = {}", graph_execution_context.rep()))?;
            let input_name = execution_context.cnn_network.get_input_name(index).context(format!("Can't find input with name = {}", index))?;
            let res = match execution_context.infer_request.set_blob(&input_name, &blob) {
                Ok(res) => {
                    Ok(res)
                }
                Err(err) => {
                    Err(self.new(ErrorCode::RuntimeError,  format!("Inference error = {:?}", err.to_string()))?)
                }
            };
            Ok(res)
        }

        fn compute(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
        ) -> Result<Result<(), Resource<errors::Error>>, anyhow::Error> {
            let graph_execution = self.executions
                .get_mut(graph_execution_context.rep())
                .ok_or(anyhow!(format!("Can't find graph execution context with ID = {}", graph_execution_context.rep())))?;
            match graph_execution.infer_request.infer() {
                Ok(..) => { Ok(Ok(())) }
                Err(err) => {
                    Ok(Err(self.new(ErrorCode::RuntimeError, format!("Inference error = {:?}", err.to_string()))?))
                }
            }
        }

        fn get_output(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
            input_name: String,
        ) -> Result<Result<Resource<tensor::Tensor>, Resource<errors::Error>>, anyhow::Error>
        {
            let index = input_name.parse::<usize>().context("Can't parse {} to usize for input_name")?;
            let graph_execution = self.executions
                .get_mut(graph_execution_context.rep())
                .ok_or(anyhow!(format!("Can't find graph execution context with ID = {}", graph_execution_context.rep())))?;
            let output_name = graph_execution.cnn_network.get_output_name(index).context("Can't find output name for ID = {index}")?;
            let blob = graph_execution.infer_request.get_blob(&output_name).context("Can't get blob for output name = {output_name}")?;
            let tensor_desc = blob.tensor_desc().context("Can't get blob description")?;
            let buffer = blob.buffer()
            .context("Can't get blob buffer")?.iter()
            .map(|&d| d as u8)
            .collect::<Vec<_>>();
            let tensor_dimensions = tensor_desc.dims().iter()
            .map(|&d| d as u32)
            .collect::<Vec<_>>();

            let tensor = TensorInternalData {
                tensor_dimensions:tensor_dimensions,
                tensor_type: map_precision_to_tensor_type(tensor_desc.precision()),
                tensor_data: buffer,
            };
            Ok(match self.tensors.push(tensor).map(Resource::<tensor::Tensor>::new_own) {
                Ok(t) => {
                    Ok(t)
                }
                Err(_) => {
                    Err(self.new(ErrorCode::RuntimeError,  format!("Can't create tensor for get_output"))?)
                }
            })
        }

        fn drop(
            &mut self,
            graph_execution_context: Resource<GraphExecutionContext>,
        ) -> Result<(), anyhow::Error> {
            let id = graph_execution_context.rep();
            self.graphs.remove(id).context("{Can't drow GraphExecutionContext with id = {id}")?;
            Ok(())
            
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
            match self.graphs.push(graph_internal_data) {
                Ok(graph_rep) => {
                    return Ok(Ok(Resource::<Graph>::new_own(graph_rep)));
                },
                Err(err) => {
                    match self.errors.push(ErrorInternalData { code: ErrorCode::RuntimeError, message: format!("{:?}", err) }) {
                        Ok(error_rep) => {
                            return Ok(Err(Resource::<errors::Error>::new_own(error_rep)));
                        }
                        Err(err) => {
                            return  Err(anyhow!("Can't create internal error for {:?}", err));
                        }
                    }
                }
            }
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
