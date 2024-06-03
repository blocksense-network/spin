pub mod hello {
    wasmtime::component::bindgen!("hello" in "tests/core-wasi-test/wit");

    use spin_core::HostComponent;
    use anyhow::anyhow;
    use test::test::gggg2;
    use test::test::kotenca;
    use test::test::kotenca::{ErrorCode};


    #[derive(Clone)]
    pub struct HelloHostComponent;

    impl HostComponent for HelloHostComponent {
        type Data = HelloHostImpl;

        fn add_to_linker<T: Send>(
            linker: &mut spin_core::Linker<T>,
            get: impl Fn(&mut spin_core::Data<T>) -> &mut Self::Data + Send + Sync + Copy + 'static,
        ) -> anyhow::Result<()> {
            Hello::add_to_linker(linker, get)
        }

        fn build_data(&self) -> Self::Data {
            HelloHostImpl {}
        }
    }

    pub struct HelloHostImpl {}

    impl gggg2::Host for HelloHostImpl {
        fn say_hello(&mut self, x: String) -> wasmtime::Result<String> {
            Ok(format!("Hello bace {x}!"))
        }
    }

    impl kotenca::HostError for HelloHostImpl {
        fn new(&mut self, code: kotenca::ErrorCode, data: String) -> Result<wasmtime::component::Resource<kotenca::Error>, anyhow::Error> {
            //Ok(kotenca::Error(code, data))
            Err(anyhow!("Not implemented"))
        }

        fn drop(&mut self, val: wasmtime::component::Resource<kotenca::Error>) -> Result<(), anyhow::Error>{
            Ok(())
        }

        fn code(&mut self, val: wasmtime::component::Resource<kotenca::Error>) -> Result<ErrorCode, anyhow::Error> {
            println!("{:?}", &val);
            //Ok(val.code)
            Err(anyhow!("Not implemented"))
        }

        fn data(&mut self, val: wasmtime::component::Resource<kotenca::Error>) -> Result<String, anyhow::Error> {
            println!("{:?}", &val);

            //Ok(val.data)
            Err(anyhow!("Not implemented"))
        }
    
    }
    
    impl kotenca::Host for HelloHostImpl {}


}