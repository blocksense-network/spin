use std::path::PathBuf;

use spin_world::v2 as ml_wit;

use crate::{backend::BackendInner, host_impl::MLHostImpl};

use spin_app::{AppComponent, DynamicHostComponent};
use spin_core::HostComponent;

#[cfg(feature = "openvino")]
use crate::backend::openvino::OpenvinoBackend;

pub struct MLHostComponent {
    pub state_dir: Option<PathBuf>,
}

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
                state_dir: self.state_dir.clone(),
            }),
        ];

        MLHostImpl {
            state_dir: self.state_dir.clone(),
            backends,
            ..Default::default()
        }
    }
}

impl DynamicHostComponent for MLHostComponent {
    fn update_data(&self, _data: &mut Self::Data, _component: &AppComponent) -> anyhow::Result<()> {
        /*let hosts = component
            .get_metadata(ALLOWED_HOSTS_KEY)?
            .unwrap_or_default();
        data.allowed_hosts = AllowedHostsConfig::parse(&hosts, self.resolver.get().unwrap())?;*/
        Ok(())
    }
}
