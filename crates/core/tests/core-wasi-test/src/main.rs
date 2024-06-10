//! This test program takes argument(s) that determine which WASI feature to
//! exercise and returns an exit code of 0 for success, 1 for WASI interface
//! failure (which is sometimes expected in a test), and some other code on
//! invalid argument(s).

use std::time::Duration;
mod multiplier {
    wit_bindgen::generate!({
        world: "multiplier",
        path: "wit/multiplier.wit"
    });
}
mod hello {
    wit_bindgen::generate!({
        world: "hello",
        path: "wit/hello.wit"
    });
}

mod ml {
    wit_bindgen::generate!({
        world: "ml",
        path: "wit/ml.wit"
    });
}
use crate::ml::test::test::graph::ExecutionTarget;
use crate::ml::test::test::graph::GraphEncoding;
use std::path::Path;

use crate::hello::test::test::gggg2::say_hello;

use image2tensor;
use image2tensor::convert_image_to_tensor_bytes;

type Result = std::result::Result<(), Box<dyn std::error::Error>>;

fn main() -> Result {
    let mut args = std::env::args();
    let cmd = args.next().expect("cmd");

    match cmd.as_str() {
        "noop" => (),
        "echo" => {
            eprintln!("echo");
            std::io::copy(&mut std::io::stdin(), &mut std::io::stdout())?;
        }
        "alloc" => {
            let size: usize = args.next().expect("size").parse().expect("size");
            eprintln!("alloc {size}");
            let layout = std::alloc::Layout::from_size_align(size, 8).expect("layout");
            unsafe {
                let p = std::alloc::alloc(layout);
                if p.is_null() {
                    return Err("allocation failed".into());
                }
                // Force allocation to actually happen
                p.read_volatile();
            }
        }
        "read" => {
            let path = args.next().expect("path");
            eprintln!("read {path}");
            std::fs::read(path)?;
        }
        "write" => {
            let path = args.next().expect("path");
            eprintln!("write {path}");
            std::fs::write(path, "content")?;
        }
        "multiply" => {
            let input: i32 = args.next().expect("input").parse().expect("i32");
            eprintln!("multiply {input}");
            let output = multiplier::imports::multiply(input);
            println!("{output}");
        }
        "hello" => {
            let input: String = args.next().expect("input").parse().expect("String");
            let output = say_hello(&input);
            println!("{output}");
        }
        "imagenet" => {
            let path_as_string = args.next().expect("path");
            let path = Path::new(&path_as_string);
            let model: Vec<u8> = std::fs::read(&path.join("model.xml"))?;
            println!("Loaded model from xml len = {}", model.len());
            let weights = std::fs::read(&path.join("model.bin"))?;
            println!("Loaded weigths with len = {}", weights.len());
            let imagenet_graph = ml::test::test::graph::load(&[model, weights], GraphEncoding::Openvino, ExecutionTarget::Cpu).unwrap();
            println!("Loaded graph into wasi-nn with ID: {:?}", imagenet_graph);
            let context = ml::test::test::graph::Graph::init_execution_context(&imagenet_graph).unwrap();
            println!("Created context with ID: {:?}", context);
            let tensor_dimensions:Vec<u32> = vec![1, 3, 224, 224];
            let tensor_data = convert_image_to_tensor_bytes(
                "images/0.jpg",
                tensor_dimensions[2],
                tensor_dimensions[3],
                image2tensor::TensorType::F32,
                image2tensor::ColorOrder::BGR,
            )
            .or_else(|e| Err(e))
            .unwrap();
            

            
            
            let tensor_type = ml::test::test::tensor::TensorType::Fp32;
            let tensor_id = ml::test::test::tensor::Tensor::new(&tensor_dimensions, tensor_type, &tensor_data);
            println!("Created tensor with ID: {:?}", tensor_id);

            let set_input_result = ml::test::test::inference::GraphExecutionContext::set_input(&context, "0", tensor_id).unwrap();
            println!("Input set with ID: {:?}", set_input_result);

            let infered_result = ml::test::test::inference::GraphExecutionContext::compute(&context).unwrap();
            println!("Executed graph inference");
            let output_result_id = ml::test::test::inference::GraphExecutionContext::get_output(&context, "0").unwrap();

            let output_result = ml::test::test::tensor::Tensor::data(&output_result_id);
            println!("output = {:?}", &output_result);


            println!("Kuku!");
        }
        "sleep" => {
            let duration =
                Duration::from_millis(args.next().expect("duration_ms").parse().expect("u64"));
            eprintln!("sleep {duration:?}");
            std::thread::sleep(duration);
        }
        "panic" => {
            eprintln!("panic");
            panic!("intentional panic");
        }
        cmd => panic!("unknown cmd {cmd}"),
    };
    Ok(())
}
