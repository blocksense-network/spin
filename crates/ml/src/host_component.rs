use std::path::PathBuf;

use spin_app::DynamicHostComponent;
use spin_core::HostComponent;
use spin_world::v2 as ml_wit;

use crate::backend;
use crate::backend::openvino::OpenvinoBackend;
use crate::{backend::BackendInner, host_impl::MLHostImpl};
//#[derive(Clone)]
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
        let mut backends: Vec<Box<dyn BackendInner>> = vec![];
        backends.push(Box::new(OpenvinoBackend {
            openvino: openvino::Core::new(None).ok(),
        }));

        MLHostImpl {
            state_dir: self.state_dir.clone(),
            openvino: openvino::Core::new(None).ok(),
            backends,
            ..Default::default()
        }
    }
}

impl DynamicHostComponent for MLHostComponent {
    fn update_data(
        &self,
        _data: &mut Self::Data,
        _component: &spin_app::AppComponent,
    ) -> anyhow::Result<()> {
        /*let hosts = component
            .get_metadata(ALLOWED_HOSTS_KEY)?
            .unwrap_or_default();
        data.allowed_hosts = AllowedHostsConfig::parse(&hosts, self.resolver.get().unwrap())?;*/
        Ok(())
    }
}
