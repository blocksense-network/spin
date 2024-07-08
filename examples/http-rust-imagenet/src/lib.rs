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
use crate::imagenet::imagenet_openvino_test;

use std::borrow::BorrowMut;
use std::env;
use std::fs;
use std::path::PathBuf;

use std::io::Write;
//use std::sync::{Mutex, OnceLock};
use tokio::sync::Mutex;

use once_cell::sync::Lazy;

use spin_sdk::http::{Request, RequestBuilder};

#[derive(Debug)]
struct MLContext {
    v: i32,
}

impl MLContext {
    fn inc(&mut self) -> i32 {
        println!(
            "inc(mut self) pointer 1 => {:x}",
            self as *mut MLContext as u64
        );
        //println!("INC!!! {:?}", *self);
        self.v = self.v + 1;
        self.v
    }
}

static ML_CONTEXT: Lazy<Mutex<MLContext>> = Lazy::new(|| {
    println!("New lazy !!");
    Mutex::new(MLContext { v: 0 })
});

/*fn main() {
    let base_url = "https://raw.githubusercontent.com/blocksense-network/imagenet_openvino/db44329b8e2b3398c9cc34dd56d94f3ce6fd6e21/"; //images/0.jpg

    let imagenet_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-programs/imagenet");
    let images_dir = imagenet_path.join("images");
    fs::create_dir_all(images_dir).unwrap();
    let files = ["model.xml", "model.bin", "images/0.jpg", "images/1.jpg"];
    for file in files {
        try_download(&(base_url.to_owned() + file), &imagenet_path.join(file)).unwrap();
    }

    println!("cargo:rerun-if-changed=build.rs");
}*/
/*
fn try_download(url: &str, filename: &PathBuf) -> Result<(), anyhow::Error> {
    let mut easy = Easy::new();
    easy.url(url)
        .map_err(|e| anyhow::anyhow!("Error {} when downloading {}", e.to_string(), url))?;

    let mut dst = Vec::new();
    {
        let mut transfer = easy.transfer();
        transfer
            .write_function(|data| {
                dst.extend_from_slice(data);
                Ok(data.len())
            })
            .unwrap();
        transfer
            .perform()
            .map_err(|e| anyhow::anyhow!("Error {} when downloading {}", e.to_string(), url))?;
    }
    {
        let mut file = std::fs::File::create(filename)?;
        file.write_all(dst.as_slice())?;
    }
    Ok(())
}
*/

/*
fn array() -> &'static Mutex<Vec<u8>> {
    static ARRAY: OnceLock<Mutex<Vec<u8>>> = OnceLock::new();
    ARRAY.get_or_init(|| Mutex::new(vec![]))
}

fn do_a_call() {
    array().lock().await().push(1);
}
*/
/*fn main() {
    do_a_call();
    do_a_call();
    do_a_call();

    println!("called {}", array().lock().unwrap().len());
}*/

use std::collections::HashMap;
use std::sync::OnceLock;

fn hashmap() -> &'static HashMap<u32, &'static str> {
    static HASHMAP: OnceLock<HashMap<u32, &str>> = OnceLock::new();
    HASHMAP.get_or_init(|| {
        println!("Initializing hashmap");
        let mut m = HashMap::new();
        m.insert(0, "foo");
        m.insert(1, "bar");
        m.insert(2, "baz");
        m
    })
}

/// A simple Spin HTTP component.
#[http_component]
async fn imagenet_handler(req: http::Request<()>) -> anyhow::Result<impl IntoResponse> {
    let mut ml_context = ML_CONTEXT.lock().await; //.expect("ML context is not initialized");
    println!("v = {}", &ml_context.v);
    println!("h = {:?}", hashmap());
    //  do_a_call();
    let x = load_by_name("imagenet"); //.expect("msg");
    match req.method() {
        &Method::POST => {
            match imagenet_openvino_test(
                ".".to_string(),
                "GPU".to_string(),
                "image0.jpg".to_string(),
            ) {
                Ok(_) => Ok(Response::new(200, "Hello, world from imagenet demo !")),
                Err(e) => {
                    let message = e.to_string();
                    Ok(Response::new(200, message))
                }
            }
        }
        _ => {
            let v = ml_context.inc();
            //        let x = array().lock().unwrap().len();
            Ok(Response::new(
                200,
                format!("Loading please wait! v = {}, x = {:?}", v, x),
            ))
        }
    }
}
