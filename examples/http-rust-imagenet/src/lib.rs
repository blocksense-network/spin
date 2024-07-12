use http::{HeaderValue, Method};
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
mod file_server;
use crate::imagenet::imagenet_infer;



use std::io::Read;


use crate::ml::fermyon::spin::graph;
use http::HeaderMap;
use multipart::server::Multipart;

fn parse_content_type(headers: &HeaderMap<HeaderValue>) -> Option<mime::Mime> {
    headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|val| val.to_str().ok())
        .and_then(|val| val.parse::<mime::Mime>().ok())
}

/// A simple Spin HTTP component.
#[http_component]
async fn imagenet_handler(req: http::Request<Vec<u8>>) -> anyhow::Result<impl IntoResponse> {
    println!("METHOD = {:?}", &req.method());
    println!("URI = {:?}", req.uri());
    let imagenet_graph = load_by_name("imagenet").unwrap();

    let context = graph::Graph::init_execution_context(&imagenet_graph).unwrap();
    match req.method() {
        &Method::POST => {

            let (parts, body) = req.into_parts();
            let x = parse_content_type(&parts.headers).unwrap();
            let boundary = x.get_param("boundary").unwrap();
            let mut mp = Multipart::with_body(&*body, boundary.as_str());
       
            while let Some(mut field) = mp.read_entry().unwrap() {
                //while field.data.read_to_endhas_data_left().expect("Bace Bace") {
                    let mut file_content: Vec<u8> = vec![];
                    //let data = field.data.fill_buf().unwrap();
                    let bytes_read = field.data.read_to_end(&mut file_content).unwrap();
                    //let s = String::from_utf8_lossy(data);
                    println!("headers: {:?}, bytes_read = {}", field.headers, bytes_read);
                    imagenet_infer(&context, &file_content).unwrap();

            }
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
