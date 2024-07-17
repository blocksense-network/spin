use crate::imagenet_classes;
use crate::ml::fermyon::spin::inference::GraphExecutionContext;
use crate::ml::fermyon::spin::{inference, tensor};
use image2tensor::convert_image_bytes_to_tensor_bytes;

pub fn elapsed_to_string(fn_name: &str, elapsed: u128) -> String {
    if elapsed < 1000 {
        format!("`{}` took {} ns", fn_name, elapsed)
    } else if elapsed < 1000 * 1000 {
        format!("`{}` took {:.2} µs", fn_name, elapsed as f64 / 1000.0)
    } else {
        format!(
            "`{}` took {:.2} ms",
            fn_name,
            elapsed as f64 / 1000.0 / 1000.0
        )
    }
}


#[derive(Debug)]
pub struct DescriptiveInferenceResult {
    pub weight: f32,
    pub class: String,
    pub inference_time_in_ns: u128,
}

pub fn imagenet_infer(
    context: &GraphExecutionContext,
    image_file_data: &[u8],
) -> std::result::Result<Vec<DescriptiveInferenceResult>, Box<dyn std::error::Error>> {
    let tensor_dimensions: Vec<u32> = vec![1, 3, 224, 224];

    let tensor_data = convert_image_bytes_to_tensor_bytes(
        image_file_data,
        tensor_dimensions[2],
        tensor_dimensions[3],
        image2tensor::TensorType::F32,
        image2tensor::ColorOrder::BGR,
    )
    .or_else(|e| Err(e))
    .unwrap();

    let tensor_id = {
        let start_for_elapsed_macro = std::time::Instant::now();
        let tensor_type = tensor::TensorType::Fp32;
        let tensor_id = tensor::Tensor::new(&tensor_dimensions, tensor_type, &tensor_data);
        let elapsed = start_for_elapsed_macro.elapsed().as_nanos();
        eprintln!(
            "Created tensor with ID: {:?} {}",
            tensor_id,
            elapsed_to_string("Tensor::new", elapsed)
        );
        tensor_id
    };
    let input_name = "0";
    {
        inference::GraphExecutionContext::set_input(&context, input_name, tensor_id).unwrap();
    }
    let inference_time_in_ns = {
        let start_for_elapsed_macro = std::time::Instant::now();
        let _infered_result = inference::GraphExecutionContext::compute(&context).unwrap();
        start_for_elapsed_macro.elapsed().as_nanos()
    };
    let output_result_id = {
        let start_for_elapsed_macro = std::time::Instant::now();
        let output_result_id =
            inference::GraphExecutionContext::get_output(&context, input_name).unwrap();
        let elapsed = start_for_elapsed_macro.elapsed().as_nanos();
        eprintln!(
            "Obtaining output {}",
            elapsed_to_string("GraphExecutionContext::get_output", elapsed)
        );
        output_result_id
    };
    let (output_data, output_dimensions, output_type) = {
        let start_for_elapsed_macro = std::time::Instant::now();
        let output_data = tensor::Tensor::data(&output_result_id);
        let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
        let output_type = tensor::Tensor::ty(&output_result_id);
        let elapsed = start_for_elapsed_macro.elapsed().as_nanos();
        eprintln!(
            "Copying data from tensor. {}",
            elapsed_to_string("Tensor::data+dimensions+type", elapsed)
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
        let results = sort_results(&output_vec_f32);
        let mut res: Vec<DescriptiveInferenceResult> = vec![];
        for i in 0..3 {
            println!(
                "{:.2} -> {}",
                results[i].weight,
                imagenet_classes::IMAGENET_CLASSES[results[i].index],
            );
            res.push(DescriptiveInferenceResult {
                weight: results[i].weight,
                class: imagenet_classes::IMAGENET_CLASSES[results[i].index].to_string(),
                inference_time_in_ns,
            })
        }
        return Ok(res);
    } else {
        eprintln!(
            "Output not as expected, output = {:?} {:?}",
            &output_dimensions, &output_type
        );
    }
    Err("Unknown error".into())
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
