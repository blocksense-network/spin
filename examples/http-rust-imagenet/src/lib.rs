use http::Method;
use ml::fermyon::spin::graph::load_by_name;
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
use crate::imagenet::imagenet_infer;

use image2tensor::convert_image_bytes_to_tensor_bytes;

use std::borrow::BorrowMut;
use std::env;
use std::fs;
use std::path::PathBuf;

use std::io::Write;
//use std::sync::{Mutex, OnceLock};
use tokio::sync::Mutex;

use once_cell::sync::Lazy;

use spin_sdk::http::{Body, Request, ResponseBuilder, StatusCode};
use std::io::Bytes;

use crate::ml::fermyon::spin::graph;

/// A simple Spin HTTP component.
#[http_component]
async fn imagenet_handler(req: http::Request<Vec<u8>>) -> anyhow::Result<impl IntoResponse> {
    let imagenet_graph = load_by_name("imagenet").unwrap();
    let context = graph::Graph::init_execution_context(&imagenet_graph).unwrap();
    match req.method() {
        &Method::POST => {
            //println!("req.body = {:?}", req.body());

            let image = req.body();

            println!("body = {:?}", String::from_utf8_lossy(image));
            imagenet_infer(&context, image).unwrap();
        }
        _ => {}
    }
    let form = r#"
    <!-- make sure the attribute enctype is set to multipart/form-data -->
    <form action="/hello" method="post" enctype="multipart/form-data">
        <!-- upload of a single file -->
        <p>
            <label>Add file (single): </label><br/>
            <input type="file" name="example1"/>
        </p>
        <p>
            <input type="submit"/>
        </p>
    </form>
    "#;
    let response = Response::builder()
        .header("Foo", "Bar")
        .status(200)
        .body(form)
        .build();
    Ok(response)
    //Ok(Response::new(200, form.to_string()))
}
