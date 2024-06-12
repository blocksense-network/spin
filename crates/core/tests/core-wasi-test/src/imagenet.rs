

//use crate::ml::test::test::graph::ExecutionTarget;
//use crate::ml::test::test::graph::GraphEncoding;

//use crate::ml::test::test::tensor::TensorType;
//use crate::ml::test::test::tensor::Tensor;
//use crate::ml::test::test::inference::Tensor;
use crate::ml::test::test::{graph, tensor, inference};


use image2tensor;
use image2tensor::convert_image_to_tensor_bytes;

use crate::Path;
use crate::imagenet_classes::IMAGENET_CLASSES;
use crate::imagenet_classes;

pub fn elapsed_to_string(fn_name: &str, elapsed: u128) -> String {
    if elapsed < 1000 {
        format!("`{}` took {} ns", fn_name, elapsed)
    } else if elapsed < 1000 * 1000 {
        format!("`{}` took {:.2} µs", fn_name, elapsed as f64 / 1000.0)
    } else {
        format!("`{}` took {:.2} ms", fn_name, elapsed as f64 / 1000.0 / 1000.0)
    }
}

pub fn imagenet_openvino_test(path_as_string: String) -> std::result::Result<(), Box<dyn std::error::Error>> {
    
    let path = Path::new(&path_as_string);
    let model: Vec<u8> = std::fs::read(&path.join("model.xml"))?;
    eprintln!("Loaded model from xml len = {}", model.len());
    let weights = std::fs::read(&path.join("model.bin"))?;
    eprintln!("Loaded weigths with len = {}", weights.len());
    let imagenet_graph = graph::load(&[model, weights], graph::GraphEncoding::Openvino, graph::ExecutionTarget::Gpu).unwrap();
    eprintln!("Loaded graph into wasi-nn with ID: {:?}", imagenet_graph);
    
    let context = graph::Graph::init_execution_context(&imagenet_graph).unwrap();
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
    
    
    
    let tensor_type = tensor::TensorType::Fp32;
    let tensor_id = tensor::Tensor::new(&tensor_dimensions, tensor_type, &tensor_data);
    eprintln!("Created tensor with ID: {:?}", tensor_id);

    let set_input_result = inference::GraphExecutionContext::set_input(&context, "0", tensor_id).unwrap();
    eprintln!("Input set with ID: {:?}", set_input_result);
    let start_for_elapsed_macro = std::time::Instant::now();
    let infered_result = inference::GraphExecutionContext::compute(&context).unwrap();
    let elapsed = start_for_elapsed_macro.elapsed().as_nanos();
    eprintln!("Executed graph inference. {}", elapsed_to_string("compute", elapsed));
    let output_result_id = inference::GraphExecutionContext::get_output(&context, "0").unwrap();

    let output_data = tensor::Tensor::data(&output_result_id);
    let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
    let output_type = tensor::Tensor::ty(&output_result_id);
    if output_dimensions.len() == 2 && output_dimensions[0] == 1 && output_dimensions[1] == 1001 && output_type == tensor::TensorType::Fp32 {
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
    Ok(())
}

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.
pub fn sort_results(buffer: &[f32]) -> Vec<InferenceResult> {
    let mut results: Vec<InferenceResult> = buffer
        .iter()
        .skip(1)
        .enumerate()
        .map(|(c, p)| InferenceResult{ index: c, weight: *p})
        .collect();
    results.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
    results
}

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
pub struct InferenceResult{
    pub index: usize,
    pub weight: f32,
}
