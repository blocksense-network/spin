/*use crate::ml::Ml::test::test::graph::Host as GraphHost;
use crate::ml::Ml::test::test::graph::GraphBuilder;
use crate::ml::Ml::test::test::graph::GraphEncoding;
use crate::ml::Ml::test::test::graph::ExecutionTarget;
use crate::ml::Ml::test::test::graph::Graph;// as GraphGraph;

use crate::ml::Ml::test::test::errors::Error;


pub mod Ml {
    wasmtime::component::bindgen!("ml" in "tests/core-wasi-test/wit");

    use spin_core::HostComponent;

    //use crate::nn_prototype::nn_prototype::test::test::errors::Host as ErrorHost;
    //use crate::nn_prototype::nn_prototype::test::test::graph::Host as GraphHost;
    //use crate::nn_prototype::nn_prototype::test::test::errors::ErrorCode as ErrorCode;


    
    use crate::ml::GraphHost;
    use crate::ml::GraphBuilder;
    use crate::ml::GraphEncoding;
    use crate::ml::ExecutionTarget;
    use crate::ml::Graph;
    use crate::ml::Error;
    

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

    impl GraphHost for MLHostImpl {
        fn load(builder: &GraphBuilder, 
            encoding: &GraphEncoding, 
            target: &ExecutionTarget) -> wasmtime::Result<Graph, Error> {
            
        }
        //load: func(builder: list<graph-builder>, encoding: graph-encoding, target: execution-target) -> result<graph, error>;
    }

    //impl Ml::error::ErrorHost for MLHostImpl {

    //}
    /*impl ErrorHost for MLHostImpl {
        fn code(&mut self) -> wasmtime::Result<ErrorCode> {
            Ok(self.code)
        }
        /*fn say_hello(&mut self, x: String) -> wasmtime::Result<String> {
            Ok(format!("Hello bace {x}!"))
        }*/
    }
    */
}*/