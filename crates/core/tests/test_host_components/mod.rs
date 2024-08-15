pub mod ml_backend;
pub mod ml_wit {
    wasmtime::component::bindgen!("ml" in "tests/core-wasi-test/wit");
}

pub mod host_component;
pub mod host_impl;

pub mod multiplier;
