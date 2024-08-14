pub mod ml_backend;

//#[cfg(not(feature = "openvino"))]
//pub mod empty_ml;
//#[cfg(feature = "openvino")]
//pub mod openvino;
//pub mod ml;

pub mod ml_wit {
    wasmtime::component::bindgen!("ml" in "tests/core-wasi-test/wit");
}

pub mod host_component;
pub mod host_impl;

pub mod multiplier;
