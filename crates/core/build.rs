
use std::env;
use std::fs;
use std::path::PathBuf;
extern crate curl;

use std::io::Write;
use curl::easy::Easy;

fn main() {
    let base_url = "https://raw.githubusercontent.com/blocksense-network/imagenet_openvino/db44329b8e2b3398c9cc34dd56d94f3ce6fd6e21/";//images/0.jpg

    let imagenet_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-programs/imagenet");
    let images_dir = imagenet_path.join("images");
    fs::create_dir_all(images_dir).unwrap();
    let files = [
        "model.xml",
        "model.bin",
        "images/0.jpg",
        "images/1.jpg",
    ];
    for file in files {
        try_download(&(base_url.to_owned()+file), &imagenet_path.join(file)).unwrap();
    }
    
    println!("cargo:rerun-if-changed=build.rs");
}

fn try_download(url: &str, filename: &PathBuf) -> Result<(), anyhow::Error> {
    let mut easy = Easy::new();
    easy.url(url).map_err(|e| anyhow::anyhow!("Error {} when downloading {}", e.to_string(), url))?;

    let mut dst = Vec::new();
    {
        let mut transfer = easy.transfer();
        transfer.write_function(|data| {
            dst.extend_from_slice(data);
            Ok(data.len())
        }).unwrap();
        transfer.perform().map_err(|e| anyhow::anyhow!("Error {} when downloading {}", e.to_string(), url))?;
    }
    {
        let mut file = std::fs::File::create(filename)?;
        file.write_all(dst.as_slice())?;
    }
    Ok(())
}