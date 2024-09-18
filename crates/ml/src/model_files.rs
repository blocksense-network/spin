use curl::easy::Easy;
use curl::easy::WriteError;

use hex::decode;
use sha1::{Digest, Sha1};

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::vec;

use ml_wit::graph::GraphEncoding;
use spin_world::v2 as ml_wit;

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
                Result::<usize, WriteError>::Ok(data.len())
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



pub struct ModelFiles {
    pub name: String, // openvino:imagenet
    pub encoding: GraphEncoding,
    pub files: Vec<String>,   // ["model.xml", "model.bin"]
    pub sources: Vec<String>, // [
    // "https://raw.githubusercontent.com/blocksense-network/imagenet_openvino/db44329b8e2b3398c9cc34dd56d94f3ce6fd6e21/model.xml",
    // "https://raw.githubusercontent.com/blocksense-network/imagenet_openvino/db44329b8e2b3398c9cc34dd56d94f3ce6fd6e21/model.bin",
    // ]
    pub hashes: Vec<String>, // [ "sha1:380a4621bf51ae357cb0eaafab203f214dbb036c", "sha1:a50b3bbd47369e306002193fd18847a186c0bcf4" ]
}

impl ModelFiles {
    pub fn check(&self, base_path: &Path) -> Result<(), anyhow::Error> {
        let model_directory = self.model_directory(base_path);
        for i in 0..self.files.len() {
            let filename = model_directory.join(self.files[i].clone());
            let file_content = fs::read(filename)?;
            ModelFiles::check_file_hash(&file_content, &self.hashes[i])?;
        }
        Ok(())
    }

    pub fn builders(&self, base_path: &Path) -> Result<Vec<Vec<u8>>, anyhow::Error> {
        let mut res = vec![];
        let model_directory = self.model_directory(base_path);
        fs::create_dir_all(model_directory.clone())?;
        for i in 0..self.files.len() {
            let filename = model_directory.join(self.files[i].clone());
            let file_content = match fs::read(&filename) {
                std::io::Result::Ok(file_content) => file_content,
                std::io::Result::Err(_e) => {
                    try_download(&self.sources[i], &filename)?;
                    let file_content = fs::read(filename)?;
                    file_content
                }
            };
            ModelFiles::check_file_hash(&file_content, &self.hashes[i])?;
            res.push(file_content);
        }
        Ok(res)
    }


    pub fn model_directory(&self, base_path: &Path) -> PathBuf {
        let imagenet_path = base_path.join(&self.name);
        imagenet_path
    }

    pub fn check_file_hash(file_data: &[u8], expected_hash: &String) -> std::io::Result<()> {
        //let hash = self.hashes[i].into_bytes();
        //let expected_hash: [u8; 20] = hash.try_into().unwrap();
        let mut parts = expected_hash.split(':');
        if let Some(sha) = parts.next() {
            match sha {
                "sha1" => {
                    if let Some(hash_str) = parts.next() {
                        let x = decode(&hash_str).unwrap();
                        let x2: [u8; 20] = x.as_slice().try_into().unwrap();

                        let mut hasher = Sha1::new();
                        hasher.update(file_data);
                        let sha1_hash = hasher.finalize();

                        let check = sha1_hash == (x2).into();
                        if check {
                            return std::io::Result::Ok(());
                        } else {
                            let message = format!("Expected sha1 hash = {:x}", sha1_hash);
                            let e = std::io::Error::new(std::io::ErrorKind::InvalidData, message);
                            return Err(e);
                        }
                    }
                    let message = format!("Wrong hash format, use for example `sha1:380..b036c`");
                    let e = std::io::Error::new(std::io::ErrorKind::InvalidData, message);
                    return Err(e);
                }
                "sha256" => {
                    let message = format!("SHA265 is not implemented");
                    let e = std::io::Error::new(std::io::ErrorKind::InvalidData, message);
                    return Err(e);
                }
                _ => {
                    let message = format!("Unknown hashing algorithm {}", sha);
                    let e = std::io::Error::new(std::io::ErrorKind::InvalidData, message);
                    return Err(e);
                }
            }
        } else {
            let message =
                format!("Mising hashing algorithm prefix, use for example `sha1:380..b036c`");
            let e = std::io::Error::new(std::io::ErrorKind::InvalidData, message);
            return Err(e);
        }
    }
}

pub fn check_file_hash(file_data: &[u8], expected_hash: &[u8; 20]) -> std::io::Result<()> {
    let mut hasher = Sha1::new();
    hasher.update(file_data);
    let sha1_hash = hasher.finalize();
    let check = sha1_hash == (*expected_hash).into();
    if check {
        std::io::Result::Ok(())
    } else {
        let message = format!("Expected sha1 hash = {:x}", sha1_hash);
        let e = std::io::Error::new(std::io::ErrorKind::InvalidData, message);
        Err(e)
    }
}

