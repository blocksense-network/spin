use spin_core::HostComponent;

use crate::test_host_components::backend;
use crate::test_host_components::host_impl::MLHostImpl;
use crate::test_host_components::ml_wit::test::test as ml_wit;

use backend::openvino::OpenvinoBackend;
use backend::BackendInner;

pub struct MLHostComponent {}

impl HostComponent for MLHostComponent {
    type Data = MLHostImpl;

    fn add_to_linker<T: Send>(
        linker: &mut spin_core::Linker<T>,
        get: impl Fn(&mut spin_core::Data<T>) -> &mut Self::Data + Send + Sync + Copy + 'static,
    ) -> anyhow::Result<()> {
        ml_wit::graph::add_to_linker(linker, get)?;
        ml_wit::inference::add_to_linker(linker, get)?;
        ml_wit::errors::add_to_linker(linker, get)?;
        ml_wit::tensor::add_to_linker(linker, get)
    }

    fn build_data(&self) -> Self::Data {
        let mut backends: Vec<Box<dyn BackendInner>> = vec![];
        if let Ok(openvino) = openvino::Core::new() {
            backends.push(Box::new(OpenvinoBackend { openvino }));
        }

        MLHostImpl {
            backends,
            ..Default::default()
        }
    }
}
