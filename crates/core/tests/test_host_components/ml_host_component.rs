use spin_core::HostComponent;

use crate::test_host_components::ml_backend;
use crate::test_host_components::ml_host_impl::MLHostImpl;
use crate::test_host_components::ml_wit::test::test as ml_wit;
use ml_backend::BackendInner;

#[cfg(feature = "openvino")]
use ml_backend::openvino::OpenvinoBackend;

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
        let backends: Vec<Box<dyn BackendInner>> = vec![
            #[cfg(feature = "openvino")]
            Box::new(OpenvinoBackend {
                openvino: openvino::Core::new().unwrap(),
            }),
        ];

        MLHostImpl {
            backends,
            ..Default::default()
        }
    }
}
