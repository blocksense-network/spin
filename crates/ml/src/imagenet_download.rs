use curl::easy::Easy;
use hex_literal::hex;
use sha1::{Digest, Sha1};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn try_download(url: &str, filename: &PathBuf) -> Result<(), anyhow::Error> {
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

pub struct OpenvinoModel {
    pub xml: Vec<u8>,
    pub weights: Vec<u8>,
    pub name: String,
}

pub fn check_file_hash(file_data: &[u8], expected_hash: &[u8; 20]) -> std::io::Result<()> {
    let mut hasher = Sha1::new();
    hasher.update(file_data);
    let sha1_hash = hasher.finalize();
    let check = sha1_hash == (*expected_hash).into();
    if check {
        Ok(())
    } else {
        let message = format!("Expected sha1 hash = {:x}", sha1_hash);
        let e = std::io::Error::new(std::io::ErrorKind::InvalidData, message);
        Err(e)
    }
}

pub fn imagenet_check_models(base_path: &Path) -> std::io::Result<OpenvinoModel> {
    let imagenet_path = base_path.join("imagenet");
    let model_xml = fs::read(imagenet_path.join("model.xml"))?;
    check_file_hash(
        &model_xml,
        &hex!("380a4621bf51ae357cb0eaafab203f214dbb036c"),
    )?;
    let model_weights = fs::read(imagenet_path.join("model.bin"))?;
    check_file_hash(
        &model_weights,
        &hex!("a50b3bbd47369e306002193fd18847a186c0bcf4"),
    )?;
    Ok(OpenvinoModel {
        xml: model_xml,
        weights: model_weights,
        name: "imagenet".to_owned(),
    })
}

pub fn imagenet_download(base_path: &Path) -> Result<(), anyhow::Error> {
    let base_url = "https://raw.githubusercontent.com/blocksense-network/imagenet_openvino/db44329b8e2b3398c9cc34dd56d94f3ce6fd6e21/";
    let imagenet_path = base_path.join("imagenet");
    fs::create_dir_all(imagenet_path.clone()).unwrap();
    let files = ["model.xml", "model.bin"];
    for file in files {
        try_download(&(base_url.to_owned() + file), &imagenet_path.join(file))?;
    }
    Ok(())
}
