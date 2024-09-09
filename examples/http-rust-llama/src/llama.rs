/*use crate::graph::GraphExecutionContext;
use crate::tensor;
use crate::inference;
use crate::tensor::TensorType;
*/

use crate::ml::fermyon::spin::inference::GraphExecutionContext;
use crate::ml::fermyon::spin::{inference, tensor, errors};

use rand_chacha;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::rand_core::RngCore;

#[derive(Debug, Clone, PartialEq)]
/// An individual logit with some additional metadata for use by the samplers.
pub struct Logit {
    /// The token id.
    pub token_id: u32,
    /// The logit value.
    pub logit: f32,
    /// Computed probability.
    pub prob: f32,
}

#[derive(Debug, Clone, Default)]
/// A collection of [Logit]s. You normally will need to build this from the result of
/// evaluating the LLM.
///
/// For convenience, this can [Deref] to the internal [Vec].
pub struct Logits {
    sorted: bool,
    has_softmax: bool,
    logits: Vec<Logit>,
}


impl Logits {
    /// Make a new [Logits] from an iterator of `f32`. We'd like to
    /// write this as [TryFrom] but unfortunately the types make this impossible.
    pub fn try_from_iter<I: IntoIterator<Item = f32>>(it: I) -> Result<Self, anyhow::Error> {
        let mut tid = 0;
        Ok(Self {
            sorted: false,
            has_softmax: false,
            logits: it
                .into_iter()
                .enumerate()
                .map(|(idx, logit)| {
                    if logit.is_nan() {
                        Err(anyhow::anyhow!("InvalidLogit {idx}"))?
                    }
                    let result = Logit {
                        token_id: tid,
                        logit,
                        prob: 0f32,
                    };
                    tid += 1;
                    Ok(result)
                })
                .collect::<Result<Vec<_>, anyhow::Error>>()?,
        })
    }

    /// Make a new [Logits] from an iterator of `f32` while only keeping the top `k`
    /// values and maintaining sorted order. This may be faster than building the
    /// full logits and then later sorting/pruning them. Set `k` high enough that
    /// the logits it prunes aren't ones that would be considered with normal
    /// sampling. Something like 500 to 2,000 is probably reasonable.
    ///
    /// Note: Infinite and NaN values will also be filtered.
    pub fn try_from_iter_top_k<I: IntoIterator<Item = f32>>(
        it: I,
        k: usize,
    ) -> Result<Self,  anyhow::Error> {
        if k == 0 {
            return Ok(Self::default());
        }

        Ok(Logits {
            sorted: true,
            has_softmax: false,
            logits: (0u32..)
                .zip(it)
                .filter(|(_tid, logit)| logit.is_finite())
                .fold(Vec::with_capacity(k), |mut logits, (tid, logit)| {
                    if logits.len() == k {
                        // The Vec is guaranteed not to be empty at this point.
                        if logit > unsafe { logits.last().unwrap_unchecked().logit } {
                            logits.truncate(k - 1);
                        } else {
                            return logits;
                        }
                    }
                    logits.insert(
                        logits.partition_point(|l| logit < l.logit),
                        Logit {
                            token_id: tid,
                            logit,
                            prob: 0f32,
                        },
                    );
                    logits
                }),
        })
    }
}

impl TryFrom<Vec<f32>> for Logits {
    type Error = anyhow::Error;

    fn try_from(value: Vec<f32>) -> Result<Self, Self::Error> {
        Self::try_from_iter(value)
    }
}

impl Logits {
    /// Get the sorted flag.
    pub fn get_sorted(&self) -> bool {
        self.sorted
    }

    /// Set the sorted flag.
    pub fn set_sorted(&mut self, is_sorted: bool) -> &mut Self {
        self.sorted = is_sorted;
        self
    }

    /// Get the softmax flag.
    pub fn get_softmax(&self) -> bool {
        self.has_softmax
    }

    /// Set the softmax flag.
    pub fn set_softmax(&mut self, has_softmax: bool) -> &mut Self {
        self.has_softmax = has_softmax;
        self
    }

    /// Ensure the [Logits] are sorted. Generally not necessary to call this directly.
    pub fn ensure_sorted(&mut self) -> Result<&mut Self, anyhow::Error> {
        if self.get_sorted() {
            return Ok(self);
        }

        let mut sort_err = Ok(());
        self.logits.as_mut_slice().sort_by(|a, b| {
            b.logit.partial_cmp(&a.logit).unwrap_or_else(|| {
                sort_err = Err(anyhow::anyhow!("Impossible: logit comparison failed?"));
                std::cmp::Ordering::Less
            })
        });
        sort_err?;
        self.set_sorted(true);
        Ok(self)
    }

    /// Ensure the softmax function has been applied to the [Logits].
    pub fn ensure_softmax(&mut self) -> Result<&mut Self, anyhow::Error> {
        if self.logits.is_empty() || self.has_softmax {
            self.has_softmax = true;
            self.sorted = true;
            return Ok(self);
        }
        self.ensure_sorted()?;
        let max_l = self.logits[0].logit;
        let cum_sum = self.logits.iter_mut().fold(0f32, |cs, l| {
            l.prob = (l.logit - max_l).exp();
            cs + l.prob
        });
        self.logits.iter_mut().for_each(|l| l.prob /= cum_sum);
        self.has_softmax = true;
        Ok(self)
    }

}


fn extract_embeddings(context: &GraphExecutionContext) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let output_result_id = inference::GraphExecutionContext::get_output(&context, "embeddings").unwrap();
    let output_data = tensor::Tensor::data(&output_result_id);
    let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
    let output_type = tensor::Tensor::ty(&output_result_id);
  

    if output_dimensions.len() == 1
        && output_dimensions[0] == 3200
        && output_type == tensor::TensorType::Fp32
    {
        let output_vec_f32 =
            unsafe { std::slice::from_raw_parts(output_data.as_ptr() as *const f32, output_dimensions[0] as usize) };
        
        let res = format!("dim = {output_dimensions:?} type = {output_type:?} data = {output_vec_f32:.2?}");  
        Ok(res)
    } else {
        Err(format!("Output mismatch found dim = {output_dimensions:?} type = {output_type:?}").into())
    }
}

fn extract_response(context: &GraphExecutionContext) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let output_result_id = inference::GraphExecutionContext::get_output(&context, "response").unwrap();
    let output_data = tensor::Tensor::data(&output_result_id);
    let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
    let output_type = tensor::Tensor::ty(&output_result_id);
    if output_dimensions.len() == 1
        && output_type == tensor::TensorType::U8
    {
        let output_vec =
            unsafe { std::slice::from_raw_parts(output_data.as_ptr() as *const u8, output_dimensions[0] as usize) }.to_vec();   
        Ok(String::from_utf8(output_vec)?)
    } else {
        Err(format!("Unexpected response tensor format").into())
    }
}

fn sample_next_token(context: &GraphExecutionContext, rnd_num: f32) -> Result<u32, Box<dyn std::error::Error>> {
    let output_result_id = inference::GraphExecutionContext::get_output(&context, "all_logits").unwrap();
    let output_data = tensor::Tensor::data(&output_result_id);
    let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
    let output_type = tensor::Tensor::ty(&output_result_id);
    if output_dimensions.len() == 1 && output_type == tensor::TensorType::Fp32 {
        let output_vec_f32 =
            unsafe { std::slice::from_raw_parts(output_data.as_ptr() as *const f32, output_dimensions[0] as usize) };
        let mut logits = Logits::try_from(output_vec_f32.to_vec())?;
        logits.ensure_softmax()?;
        //let mut res = format!("dim = {output_dimensions:?} type = {output_type:?}");  
        let mut cdf = 0f32;
        let k = 40;
        for i in 0..k {
            cdf += logits.logits[i].prob;
            //let f = format!("</br> token_id = {} -> prob {:.3?} cdf = {:.3?}", logits.logits[i].token_id, logits.logits[i].prob, cdf);
            //res.push_str(&f)
        }
        let mut s = 0f32;
        let rnd_num = rnd_num * cdf;//rng.next_u32() as f32 * 2.0f32.powf(-32.0f32) * cdf;
        let mut selected = k - 1;
        for i in 0..k {
            s += logits.logits[i].prob;
            if s >= rnd_num {
                selected = i;
                break;
            }
        }
        Ok(logits.logits[selected].token_id)
    } else {
        Err(format!("Unexpected all_logits tensor format").into())
    }
}


pub fn llama_infer(
    context: &GraphExecutionContext,
    promt: &str,
) -> std::result::Result<String, Box<dyn std::error::Error>> {

    let query_tensor_data = promt.to_owned().into_bytes();
    let query_tensor_type = tensor::TensorType::U8;
    let query_tensor_dimensions: Vec<u32> = vec![1, query_tensor_data.len() as u32];
    let query_tensor_id = tensor::Tensor::new(&query_tensor_dimensions, query_tensor_type, &query_tensor_data);
    let query_input_name = "query";



    //let input_name = "token_ids";

    inference::GraphExecutionContext::set_input(&context, query_input_name, query_tensor_id).unwrap();
    inference::GraphExecutionContext::compute(&context).unwrap();
    
    let seed = 32;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..32 {
        let _ = rng.next_u32();
    }

    let mut res = "".to_string();

    for _ in 0..100 {
        let r = rng.next_u32() as f32 * 2.0f32.powf(-32.0f32);
        match sample_next_token(context, r) {
            Ok(token_id) => {
                let token_tensor_data = token_id.to_le_bytes().to_vec();
                let token_tensor_type = tensor::TensorType::I32;
                let token_tensor_dimensions: Vec<u32> = vec![1, 1];
                let token_tensor_id = tensor::Tensor::new(&token_tensor_dimensions, token_tensor_type, &token_tensor_data);
                let token_input_name = "next_token";
                println!("next token = {token_id} tensor_data = {token_tensor_data:?}");
                if let Err(err) = inference::GraphExecutionContext::set_input(&context, token_input_name, token_tensor_id) {
                    
                    eprintln!("Error err = {err:?}");
                    return Ok(res);
                }
                inference::GraphExecutionContext::compute(&context).unwrap();
                let response = extract_response(&context)?;
                res.push_str(format!("<br> token_id = {token_id} response = {response}").as_str());
            }
            Err(e) => {
                res.push_str(format!("<br> error = {e}").as_str());
            }
        }
    }

    Ok(res)
 
    /*

    let output_result_id = inference::GraphExecutionContext::get_output(&context, "all_logits").unwrap();
    let output_data = tensor::Tensor::data(&output_result_id);
    let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
    let output_type = tensor::Tensor::ty(&output_result_id);

    let res2 = if output_dimensions.len() == 1
        && output_type == tensor::TensorType::Fp32
    {
        let output_vec_f32 =
            unsafe { std::slice::from_raw_parts(output_data.as_ptr() as *const f32, output_dimensions[0] as usize) };
        let mut logits = Logits::try_from(output_vec_f32.to_vec())?;
        logits.ensure_softmax()?;
        let mut res = format!("dim = {output_dimensions:?} type = {output_type:?}");  
        let mut cdf = 0f32;
        let k = 40;
        for i in 0..k {
            cdf += logits.logits[i].prob;
            let f = format!("</br> token_id = {} -> prob {:.3?} cdf = {:.3?}", logits.logits[i].token_id, logits.logits[i].prob, cdf);
            res.push_str(&f)
        }
        let mut s = 0f32;
        let rnd_num = rng.next_u32() as f32 * 2.0f32.powf(-32.0f32) * cdf;
        let mut selected = k - 1;
        for i in 0..k {
            s += logits.logits[i].prob;
            if s >= rnd_num {
                selected = i;
                break;
            }
        }
        res.push_str(&format!("<br>Selected token_id = {} -> prob {:.3?} rnd = {:.3?}", logits.logits[selected].token_id, logits.logits[selected].prob, rnd_num));
        res
    } else {
        return Err(format!("Output mismatch found dim = {output_dimensions:?} type = {output_type:?}").into());
    };
    Ok(format!("</br> all logits = {res2} </br> </br> embeddings = {res}"))

    */
    
}