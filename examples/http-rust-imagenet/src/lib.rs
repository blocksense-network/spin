use http::Method;
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
fn imagenet_handler(req: http::Request<()>) -> anyhow::Result<impl IntoResponse> {

    match req.method() {
        &Method::POST => {
            match imagenet_openvino_test(".".to_string(), "GPU".to_string(), "image0.jpg".to_string()) {
                Ok(_) => {Ok(Response::new(200, "Hello, world from imagenet demo !"))}
                Err(e) => {
                    let message = e.to_string();
                    Ok(Response::new(200, message))
                }
            }
            
        }
        _ => {
            Ok(Response::new(200, "Hello, world from non post method!"))
        }
    }   

}
