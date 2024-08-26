use crate::graph::GraphExecutionContext;

pub fn llama_infer(
    context: &GraphExecutionContext,
    promt: &str,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    Ok("llama result".to_string())
}