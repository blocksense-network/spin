use spin_core::HostComponent;
use spin_world::v2 as ml_wit;

use crate::host_impl::MLHostImpl;

#[derive(Clone)]
pub struct MLHostComponent;

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
        MLHostImpl {
            ..Default::default()
        }
    }
}