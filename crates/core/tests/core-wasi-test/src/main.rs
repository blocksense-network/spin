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
mod imagenet_classes;
use crate::imagenet_classes::{InferenceResult, sort_results};


use crate::ml::test::test::graph::ExecutionTarget;
use crate::ml::test::test::graph::GraphEncoding;
use crate::ml::test::test::tensor::TensorType;
use std::path::Path;

use crate::hello::test::test::gggg2::say_hello;

use image2tensor;
use image2tensor::convert_image_to_tensor_bytes;

type Result = std::result::Result<(), Box<dyn std::error::Error>>;

fn elapsed_to_string(fn_name: &str, elapsed: u128) -> String {
    if elapsed < 1000 {
        format!("`{}` took {} ns", fn_name, elapsed)
    } else if elapsed < 1000 * 1000 {
        format!("`{}` took {:.2} µs", fn_name, elapsed as f64 / 1000.0)
    } else {
        format!("`{}` took {:.2} ms", fn_name, elapsed as f64 / 1000.0 / 1000.0)
    }
}
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
            eprintln!("Loaded model from xml len = {}", model.len());
            let weights = std::fs::read(&path.join("model.bin"))?;
            eprintln!("Loaded weigths with len = {}", weights.len());
            let imagenet_graph = ml::test::test::graph::load(&[model, weights], GraphEncoding::Openvino, ExecutionTarget::Gpu).unwrap();
            eprintln!("Loaded graph into wasi-nn with ID: {:?}", imagenet_graph);
            
            let context = ml::test::test::graph::Graph::init_execution_context(&imagenet_graph).unwrap();
            
            eprintln!("Created context with ID: {:?}", context);
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
            eprintln!("Created tensor with ID: {:?}", tensor_id);

            let set_input_result = ml::test::test::inference::GraphExecutionContext::set_input(&context, "0", tensor_id).unwrap();
            eprintln!("Input set with ID: {:?}", set_input_result);
            let start_for_elapsed_macro = std::time::Instant::now();
            let infered_result = ml::test::test::inference::GraphExecutionContext::compute(&context).unwrap();
            let elapsed = start_for_elapsed_macro.elapsed().as_nanos();
            eprintln!("Executed graph inference. {}", elapsed_to_string("compute", elapsed));
            let output_result_id = ml::test::test::inference::GraphExecutionContext::get_output(&context, "0").unwrap();

            let output_data = ml::test::test::tensor::Tensor::data(&output_result_id);
            let output_dimensions = ml::test::test::tensor::Tensor::dimensions(&output_result_id);
            let output_type = ml::test::test::tensor::Tensor::ty(&output_result_id);
            if output_dimensions.len() == 2 && output_dimensions[0] == 1 && output_dimensions[1] == 1001 && output_type == TensorType::Fp32 {
                let output_vec_f32 = unsafe { std::slice::from_raw_parts(output_data.as_ptr() as *const f32,  1001) };
                let results = sort_results(&output_vec_f32);
                for i in 0..3 {
                    println!(
                        "{:.2} -> {}",
                        results[i].weight,
                        imagenet_classes::IMAGENET_CLASSES[results[i].index],
                    );
                }
            } else {
                eprintln!("Output not as expected, output = {:?} {:?}", &output_dimensions, &output_type);
            }
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