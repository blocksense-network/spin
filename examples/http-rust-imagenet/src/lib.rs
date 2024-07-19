use anyhow::Ok;
use http::{HeaderMap, HeaderValue, Method};
use multipart::server::Multipart;
use std::io::Read;

use ml::fermyon::spin::graph::load_by_name;

use spin_sdk::http::{IntoResponse, Response};
use spin_sdk::http_component;
use spin_sdk::key_value::Store;

mod ml {
    wit_bindgen::generate!({
        world: "ml",
        path: "wit/ml.wit"
    });
}

mod imagenet;
mod imagenet_classes;

use crate::imagenet::elapsed_to_string;
use crate::imagenet::imagenet_infer;
use crate::ml::fermyon::spin::graph;

fn parse_content_type(headers: &HeaderMap<HeaderValue>) -> Option<mime::Mime> {
    headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|val| val.to_str().ok())
        .and_then(|val| val.parse::<mime::Mime>().ok())
}

fn store_handler(req: http::Request<Vec<u8>>) -> anyhow::Result<Vec<u8>> {
    let path = req.uri().path();
    let path_parts: Vec<_> = path.split('/').map(|x| x.to_string()).collect();
    if path_parts.len() > 2 {
        let store = Store::open_default()?;
        let key = path_parts[2].clone();
        match store.get_json::<Vec<u8>>(key)? {
            Some(value) => {
                return Ok(value);
            }
            None => {}
        }
    }
    Err(anyhow::anyhow!("not found"))
}

fn imagenet_handler(req: http::Request<Vec<u8>>) -> anyhow::Result<String> {
    let res = match req.method() {
        &Method::POST => {
            let (parts, body) = req.into_parts();
            let x = parse_content_type(&parts.headers).unwrap();
            let boundary = x.get_param("boundary").unwrap();
            let mut mp = Multipart::with_body(&*body, boundary.as_str());
            println!("parts = {parts:?}");

            // FORM DATA
            let mut image_key = "".to_owned();
            let mut target = "CPU".to_string();
            let mut file_content: Vec<u8> = vec![];

            while let Some(mut field) = mp.read_entry().unwrap() {
                match field.headers.name.as_ref().to_owned().as_str() {
                    "image" => {
                        let _bytes_read = field.data.read_to_end(&mut file_content).unwrap();
                        let store = Store::open_default()?;
                        if let Some(filename) = field.headers.filename.clone() {
                            image_key = filename;
                            let _ = store.set_json(image_key.clone(), &file_content)?;
                        }
                    }
                    "target" => {
                        target = "".to_string();
                        let _bytes_read = field.data.read_to_string(&mut target).unwrap();
                    }
                    _ => {}
                }
            }

            use core::result::Result::Ok;
            let imagenet_name = format!("openvino:imagenet:{}", target);
            match load_by_name(&imagenet_name) {
                Ok(imagenet_graph) => match graph::Graph::init_execution_context(&imagenet_graph) {
                    Ok(context) => {
                        if file_content.len() > 0 && image_key.len() > 0 {
                            match imagenet_infer(&context, &file_content) {
                                Ok(res) => {
                                    let mut b = "".to_string();
                                    let filename = image_key;
                                    b.push_str(&format!(
                                        r#"<img src="/store/{filename}" width="33%"></img>"#
                                    ));
                                    b.push_str(
                                        r#"<table>
                                        <thead>
                                        <tr>
                                            <th>Class</th>
                                            <th>Weight</th>
                                        </tr>
                                        </thead>"#,
                                    );
                                    for x in &res {
                                        b.push_str(&format!(
                                            "<tr><td>{}</td> <td>{:.2}</td></tr>",
                                            x.class, x.weight,
                                        ));
                                    }

                                    let caption = elapsed_to_string(
                                        "Inference time",
                                        res.first().unwrap().inference_time_in_ns,
                                    );
                                    b.push_str(format!("<caption>{}</caption>", caption).as_str());
                                    b.push_str("</table>");

                                    b.to_owned()
                                }
                                Err(e) => e.to_string(),
                            }
                        } else {
                            "Nothing to compute".to_string()
                        }
                    }
                    Err(err) => err.data(),
                },
                Err(err) => err.data(),
            }
        }
        _ => "".to_string(),
    };
    Ok(format!("<div>{res}</div>"))
}

fn add_form(mut html_body: String) -> String {
    let form = r#"
    <!-- make sure the attribute enctype is set to multipart/form-data -->
    <form action="/imagenet" method="post" enctype="multipart/form-data">



        <!-- upload of a single file -->
        <p>
            <label>Add file (single): </label><br/>
            <input type="file" name="image"/>
        </p>
        <p>
            <label for="target">Choose a inference target:</label>
            <select name="target" id="target">
                <option value="CPU">CPU</option>
                <option value="GPU">GPU</option>
            </select> 
        </p>
        <p>
            <input type="submit"/>
        </p>
    </form>
    "#;
    html_body.push_str(form);
    html_body
}

/// A simple Spin HTTP component.
#[http_component]
async fn imagenet_demo_handler(req: http::Request<Vec<u8>>) -> anyhow::Result<impl IntoResponse> {
    //println!("METHOD = {:?}", &req.method());
    //println!("URI = {:?}", &req.uri().path_and_query());
    let path = req.uri().path();
    let path_parts: Vec<_> = path.split('/').map(|x| x.to_string()).collect();
    if path_parts.len() > 1 {
        match path_parts[1].as_str() {
            "imagenet" => {
                let html_body = add_form(imagenet_handler(req)?);
                let response = Response::builder()
                    .header("Foo", "Bar")
                    .status(200)
                    .body(html_body)
                    .build();
                return Ok(response);
            }
            "store" => {
                //println!("CALL STORE");
                let contents = store_handler(req)?;
                let response = Response::builder()
                    .header("Foo", "Bar")
                    .status(404)
                    .body(contents)
                    .build();
                return Ok(response);
            }
            _ => {}
        }
    }

    let response = Response::builder()
        .status(404)
        .body("404 - Not found".to_owned())
        .build();
    return Ok(response);
}
