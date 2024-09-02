/*use crate::graph::GraphExecutionContext;
use crate::tensor;
use crate::inference;
use crate::tensor::TensorType;
*/

use crate::ml::fermyon::spin::inference::GraphExecutionContext;
use crate::ml::fermyon::spin::{inference, tensor};


pub fn llama_embeddings(
    context: &GraphExecutionContext,
    promt: &str,
) -> std::result::Result<String, Box<dyn std::error::Error>> {

    let tensor_data = promt.to_owned().into_bytes();
    let tensor_type = tensor::TensorType::U8;
    let tensor_dimensions: Vec<u32> = vec![1, tensor_data.len() as u32];
    let tensor_id = tensor::Tensor::new(&tensor_dimensions, tensor_type, &tensor_data);
    let input_name = "0";

    inference::GraphExecutionContext::set_input(&context, input_name, tensor_id).unwrap();
    inference::GraphExecutionContext::compute(&context).unwrap();
    
    
    let output_result_id = inference::GraphExecutionContext::get_output(&context, "embeddings").unwrap();

    let output_data = tensor::Tensor::data(&output_result_id);
    let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
    let output_type = tensor::Tensor::ty(&output_result_id);
   

    let res = if output_dimensions.len() == 1
        && output_dimensions[0] == 3200
        && output_type == tensor::TensorType::Fp32
    {
        let output_vec_f32 =
            unsafe { std::slice::from_raw_parts(output_data.as_ptr() as *const f32, output_dimensions[0] as usize) };
        
        let res = format!("dim = {output_dimensions:?} type = {output_type:?} data = {output_vec_f32:.2?}");  
        res
    } else {
        return Err(format!("Output mismatch found dim = {output_dimensions:?} type = {output_type:?}").into());
    };
    


    let output_result_id = inference::GraphExecutionContext::get_output(&context, "all_logits").unwrap();
    let output_data = tensor::Tensor::data(&output_result_id);
    let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
    let output_type = tensor::Tensor::ty(&output_result_id);
    
    let res2 = if output_dimensions.len() == 1
        && output_dimensions[0] == 96000
        && output_type == tensor::TensorType::Fp32
    {
        let output_vec_f32 =
            unsafe { std::slice::from_raw_parts(output_data.as_ptr() as *const f32, output_dimensions[0] as usize) };
        
        let res = format!("dim = {output_dimensions:?} type = {output_type:?} data = {output_vec_f32:.2?}");  
        res
    } else {
        return Err(format!("Output mismatch found dim = {output_dimensions:?} type = {output_type:?}").into());
    };
    
    Ok(format!("</br> all logits = {res2} </br> </br> embeddings = {res}"))
}