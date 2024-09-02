use crate::ml::test::test::{graph, inference, tensor, errors};
use image2tensor::convert_image_bytes_to_tensor_bytes;

use crate::imagenet_classes;
use crate::Path;
use std::time::Duration;

pub fn elapsed_to_string(fn_name: &str, elapsed: &Duration) -> String {
    let elapsed_ns = elapsed.as_nanos();
    if elapsed_ns < 1000 {
        format!("`{}` took {} ns", fn_name, elapsed_ns)
    } else if elapsed_ns < 1000 * 1000 {
        format!("`{}` took {:.2} µs", fn_name, elapsed_ns as f64 / 1000.0)
    } else {
        format!(
            "`{}` took {:.2} ms",
            fn_name,
            elapsed_ns as f64 / 1000.0 / 1000.0
        )
    }
}

pub fn bytes_to_string(b: usize) -> String {
    if b < 1024 {
        format!("{} Bytes", b)
    } else if b < 1024 * 1024 {
        format!("{:.2} kB", b as f64 / 1024.0)
    } else {
        format!("{:.2} MB", b as f64 / 1024.0 / 1024.0)
    }
}

/// Return the execution target type from string
fn map_string_to_execution_target(target: &str) -> Result<graph::ExecutionTarget, String> {
    match target {
        "CPU" => Ok(graph::ExecutionTarget::Cpu),
        "GPU" => Ok(graph::ExecutionTarget::Gpu),
        "TPU" => Ok(graph::ExecutionTarget::Tpu),
        _ => Err(format!("Unknown execution targer = {}", target)),
    }
}


pub fn preprocess_image_for_imagenet(
    image_file_data: &[u8],
    tensor_dimensions:&[u32],
) -> Vec<u8> {
    let tensor_data = convert_image_bytes_to_tensor_bytes(
        image_file_data,
        tensor_dimensions[2],
        tensor_dimensions[3],
        image2tensor::TensorType::F32,
        image2tensor::ColorOrder::BGR,
    )
    .unwrap();

    let mut new_tensor_data = Vec::<f32>::new();

    let num_colors = tensor_dimensions[1] as usize;
    let height = tensor_dimensions[2] as usize;
    let width = tensor_dimensions[3] as usize;

    for c in 0..num_colors {
        for y in 0..height {
            for x in 0..width {
                let offset = ((y * width + x) * 3 + c) * 4;
                let v = f32::from_le_bytes(
                    tensor_data[offset..offset + 4]
                        .try_into()
                        .expect("Needed 4 bytes for a float"),
                );
                new_tensor_data.push(v);
            }
        }
    }
    let (_head, body, _tail) = unsafe { new_tensor_data.align_to::<u8>() };
    body.to_vec()
}

pub fn imagenet_openvino_test(
    path_as_string: String,
    target_as_string: String,
    image_file: String,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(&path_as_string);
    let target = map_string_to_execution_target(&target_as_string)?;
    let model = {
        let start_for_elapsed_macro = std::time::Instant::now();
        let model: Vec<u8> = std::fs::read(path.join("model.xml"))?;
        let elapsed = start_for_elapsed_macro.elapsed();
        eprintln!(
            "Loaded model from xml {} {}",
            bytes_to_string(model.len()),
            elapsed_to_string("fs::read", &elapsed)
        );
        model
    };
    let weights = {
        let start_for_elapsed_macro = std::time::Instant::now();
        let weights = std::fs::read(path.join("model.bin"))?;
        let elapsed = start_for_elapsed_macro.elapsed();
        eprintln!(
            "Loaded weigths {} {}",
            bytes_to_string(weights.len()),
            elapsed_to_string("fs::read", &elapsed)
        );
        weights
    };
    let imagenet_graph = {
        let start_for_elapsed_macro = std::time::Instant::now();
        let imagenet_graph =
            graph::load(&[model, weights], graph::GraphEncoding::Openvino, target).unwrap();
        let elapsed = start_for_elapsed_macro.elapsed();
        eprintln!("---- {:?} ----", target);
        eprintln!(
            "Loaded graph with ID: {:?} {}",
            imagenet_graph,
            elapsed_to_string("graph::load", &elapsed)
        );
        imagenet_graph
    };
    let context = {
        let start_for_elapsed_macro = std::time::Instant::now();
        let context = graph::Graph::init_execution_context(&imagenet_graph).map_err(|e| errors::Error::data(&e) ).expect("XXXXXXXX -> init_execution_context failed");
        let elapsed = start_for_elapsed_macro.elapsed();
        eprintln!(
            "Created context with ID: {:?} {}",
            context,
            elapsed_to_string("Graph::init_execution_context", &elapsed)
        );
        context
    };

    let tensor_dimensions: Vec<u32> = vec![1, 3, 224, 224];
    let image_file_bytes = std::fs::read(image_file).unwrap();
    let tensor_data = preprocess_image_for_imagenet(&image_file_bytes, &tensor_dimensions);

    let tensor_id = {
        let start_for_elapsed_macro = std::time::Instant::now();
        let tensor_type = tensor::TensorType::Fp32;
        let tensor_id = tensor::Tensor::new(&tensor_dimensions, tensor_type, &tensor_data);
        let elapsed = start_for_elapsed_macro.elapsed();
        eprintln!(
            "Created tensor with ID: {:?} {}",
            tensor_id,
            elapsed_to_string("Tensor::new", &elapsed)
        );
        tensor_id
    };
    let input_name = "0";
    {
        let start_for_elapsed_macro = std::time::Instant::now();
        inference::GraphExecutionContext::set_input(&context, input_name, tensor_id).unwrap();
        let elapsed = start_for_elapsed_macro.elapsed();
        eprintln!(
            "Input set {}",
            elapsed_to_string("GraphExecutionContext::set_input", &elapsed)
        );
    }
    {
        let start_for_elapsed_macro = std::time::Instant::now();
        inference::GraphExecutionContext::compute(&context).unwrap();
        let elapsed = start_for_elapsed_macro.elapsed();
        eprintln!(
            "Executed graph inference. {}",
            elapsed_to_string("GraphExecutionContext::compute", &elapsed)
        );
    }
    let output_result_id = {
        let start_for_elapsed_macro = std::time::Instant::now();
        let output_result_id =
            inference::GraphExecutionContext::get_output(&context, input_name).unwrap();
        let elapsed = start_for_elapsed_macro.elapsed();
        eprintln!(
            "Obtaining output {}",
            elapsed_to_string("GraphExecutionContext::get_output", &elapsed)
        );
        output_result_id
    };
    let (output_data, output_dimensions, output_type) = {
        let start_for_elapsed_macro = std::time::Instant::now();
        let output_data = tensor::Tensor::data(&output_result_id);
        let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
        let output_type = tensor::Tensor::ty(&output_result_id);
        let elapsed = start_for_elapsed_macro.elapsed();
        eprintln!(
            "Copying data from tensor. {}",
            elapsed_to_string("Tensor::data+dimensions+type", &elapsed)
        );
        (output_data, output_dimensions, output_type)
    };
    if output_dimensions.len() == 2
        && output_dimensions[0] == 1
        && output_dimensions[1] == 1001
        && output_type == tensor::TensorType::Fp32
    {
        let output_vec_f32 =
            unsafe { std::slice::from_raw_parts(output_data.as_ptr() as *const f32, 1001) };
        let results = sort_results(output_vec_f32);
        for res in results.iter().take(3) {
            println!(
                "{:.2} -> {}",
                res.weight,
                imagenet_classes::IMAGENET_CLASSES[res.index],
            );
        }
    } else {
        eprintln!(
            "Output not as expected, output = {:?} {:?}",
            &output_dimensions, &output_type
        );
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
        .map(|(c, p)| InferenceResult {
            index: c,
            weight: *p,
        })
        .collect();
    results.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
    results
}

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
pub struct InferenceResult {
    pub index: usize,
    pub weight: f32,
}
