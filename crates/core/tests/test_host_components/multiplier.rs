use spin_core::HostComponent;

// Simple test HostComponent; multiplies the input by the configured factor
#[derive(Clone)]
pub struct MultiplierHostComponent;

mod multiplier {
    wasmtime::component::bindgen!("multiplier" in "tests/core-wasi-test/wit");
}

impl HostComponent for MultiplierHostComponent {
    type Data = Multiplier;

    fn add_to_linker<T: Send>(
        linker: &mut spin_core::Linker<T>,
        get: impl Fn(&mut spin_core::Data<T>) -> &mut Self::Data + Send + Sync + Copy + 'static,
    ) -> anyhow::Result<()> {
        multiplier::imports::add_to_linker(linker, get)
    }

    fn build_data(&self) -> Self::Data {
        Multiplier(2)
    }
}

pub struct Multiplier(pub i32);

impl multiplier::imports::Host for Multiplier {
    fn multiply(&mut self, a: i32) -> i32 {
        self.0 * a
    }
}