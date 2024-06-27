use spin_sdk::http::{IntoResponse, Response};
use spin_sdk::http_component;

mod ml {
    wit_bindgen::generate!({
        world: "ml",
        path: "wit/ml.wit"
    });
}

mod imagenet;
mod imagenet_classes;
use crate::imagenet::imagenet_openvino_test;

/// A simple Spin HTTP component.
#[http_component]
fn hello_world(_req: http::Request<()>) -> anyhow::Result<impl IntoResponse> {
    let _ = imagenet_openvino_test(".".to_string(), "GPU".to_string(), "images/image0.jpg".to_string());
    Ok(Response::new(200, "Hello, world from imagenet demo  !"))
}
