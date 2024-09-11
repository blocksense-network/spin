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

mod llama;

use crate::llama::{llama_infer, session_handler, history_handler, download_handler};
use crate::ml::fermyon::spin::graph;

fn parse_content_type(headers: &HeaderMap<HeaderValue>) -> Option<mime::Mime> {
    headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|val| val.to_str().ok())
        .and_then(|val| val.parse::<mime::Mime>().ok())
}

/// A simple Spin HTTP component.
#[http_component]
async fn llama_demo_handler(req: http::Request<Vec<u8>>) -> anyhow::Result<impl IntoResponse> {
    //println!("METHOD = {:?}", &req.method());
    //println!("URI = {:?}", &req.uri().path_and_query());
    let path = req.uri().path();
    let path_parts: Vec<_> = path.split('/').map(|x| x.to_string()).collect();
    if path_parts.len() > 1 {
        match path_parts[1].as_str() {
            "llama" => {
                let html_body = llama_handler(req)?;
                let response = Response::builder().status(200).body(html_body).build();
                return Ok(response);
            }
            "session" => {
                let contents = session_handler(req)?;
                let response = Response::builder().status(200).body(contents).build();
                return Ok(response);
            }
            "history" => {
                let contents = history_handler(req)?;
                let response = Response::builder().status(200).body(contents).build();
                return Ok(response);
            }
            "download" => {
                if path_parts.len() > 2 {
                    let contents = download_handler(req)?;
                    let key = path_parts[2].clone();
                    let response = Response::builder()
                        .header(http::header::CONTENT_DISPOSITION.to_string(),
                                format!("attachment; filename={}.json", key.as_str()))
                        .status(200)
                        .body(contents).build();
                    return Ok(response);
                }
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

fn llama_handler(req: http::Request<Vec<u8>>) -> anyhow::Result<String> {
    let res = match req.method() {
        &Method::POST => {
            let (parts, body) = req.into_parts();
            let x = parse_content_type(&parts.headers).unwrap();
            let boundary = x.get_param("boundary").unwrap();
            let mp = Multipart::with_body(&*body, boundary.as_str());

            let form_data = llama_process_form(mp).unwrap();

            use core::result::Result::Ok;
            let model_name = format!("llm:{}", form_data.model);

            match load_by_name(&model_name) {
                Ok(llama_graph) => match graph::Graph::init_execution_context(&llama_graph) {
                    Ok(context) => llama_infer(&context, &form_data.promt, model_name, form_data.rng_seed).unwrap(),
                    Err(err) => err.data(),
                },
                Err(err) => err.data(),
            }
        }
        _ => {
            "".to_string()
        }
    };
    Ok(lamma_add_form(format!("<div>{res}</div>")))
}

fn lamma_add_form(mut html_body: String) -> String {
    let form = r#"
    <!-- make sure the attribute enctype is set to multipart/form-data -->
    <form action="/llama" method="post" enctype="multipart/form-data">
        <h2>
            Enter text to process by llama
        </h2>
        <p>
            <label>Submit text to llama </label><br/>
            <input type="text" name="promt"/>
        </p>
        <p>
            <label>Random seed for sampling</label><br/>
            <input type="text" name="rng_seed"/>
        </p>
        <p>
            <label for="target">Choose a inference target:</label>
            <select name="target" id="target">
                <option value="CPU">CPU</option>
                <option value="GPU">GPU</option>
            </select> 
        </p>
        <p>
        <label for="model">Choose a inference network:</label>
        <select name="model" id="model">
            <option value="open_llama_3b-f16">open_llama_3b-f16</option>
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

#[derive(Debug)]
struct LlamaFormData {
    promt: String,
    model: String,
    target: String,
    rng_seed: u64,
}

fn llama_process_form(mut mp: Multipart<&[u8]>) -> Result<LlamaFormData, anyhow::Error> {
    // FORM DATA
    let mut promt = "".to_owned();
    let mut target = "CPU".to_string();
    let mut model = "".to_owned();
    let mut rng_seed = 1337;

    while let Some(mut field) = mp.read_entry().unwrap() {
        match field.headers.name.as_ref().to_owned().as_str() {
            "promt" => {
                promt = "".to_string();
                let _bytes_read = field.data.read_to_string(&mut promt).unwrap();
            }
            "model" => {
                model = "".to_string();
                let _bytes_read = field.data.read_to_string(&mut model).unwrap();
            }
            "rng_seed" => {
                let mut x = "".to_string();
                let _bytes_read = field.data.read_to_string(&mut x).unwrap();
                rng_seed = x.parse::<u64>()?;
            }
            "target" => {
                target = "".to_string();
                let _bytes_read = field.data.read_to_string(&mut target).unwrap();
            }
            _ => {}
        }
    }

    Ok(LlamaFormData {
        promt,
        model,
        target,
        rng_seed,
    })
}
