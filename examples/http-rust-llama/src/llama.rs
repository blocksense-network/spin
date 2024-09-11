/*use crate::graph::GraphExecutionContext;
use crate::tensor;
use crate::inference;
use crate::tensor::TensorType;
*/

use crate::ml::fermyon::spin::inference::GraphExecutionContext;
use crate::ml::fermyon::spin::{errors, inference, tensor};

use crate::Store;
use rand_chacha;
use rand_chacha::rand_core::RngCore;
use rand_chacha::rand_core::SeedableRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
    ) -> Result<Self, anyhow::Error> {
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

fn extract_embeddings(
    context: &GraphExecutionContext,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let output_result_id =
        inference::GraphExecutionContext::get_output(&context, "embeddings").unwrap();
    let output_data = tensor::Tensor::data(&output_result_id);
    let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
    let output_type = tensor::Tensor::ty(&output_result_id);

    if output_dimensions.len() == 1
        && output_dimensions[0] == 3200
        && output_type == tensor::TensorType::Fp32
    {
        let output_vec_f32 = unsafe {
            std::slice::from_raw_parts(
                output_data.as_ptr() as *const f32,
                output_dimensions[0] as usize,
            )
        };

        let res = format!(
            "dim = {output_dimensions:?} type = {output_type:?} data = {output_vec_f32:.2?}"
        );
        Ok(res)
    } else {
        Err(
            format!("Output mismatch found dim = {output_dimensions:?} type = {output_type:?}")
                .into(),
        )
    }
}

fn extract_response(
    context: &GraphExecutionContext,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let output_result_id =
        inference::GraphExecutionContext::get_output(&context, "response").unwrap();
    let output_data = tensor::Tensor::data(&output_result_id);
    let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
    let output_type = tensor::Tensor::ty(&output_result_id);
    if output_dimensions.len() == 1 && output_type == tensor::TensorType::U8 {
        let output_vec = unsafe {
            std::slice::from_raw_parts(
                output_data.as_ptr() as *const u8,
                output_dimensions[0] as usize,
            )
        }
        .to_vec();
        Ok(String::from_utf8(output_vec)?)
    } else {
        Err(format!("Unexpected response tensor format").into())
    }
}

fn extract_eos_token_id(
    context: &GraphExecutionContext,
) -> std::result::Result<u32, Box<dyn std::error::Error>> {
    let output_result_id =
        inference::GraphExecutionContext::get_output(&context, "eos_token_id").unwrap();
    let output_data = tensor::Tensor::data(&output_result_id);
    let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
    let output_type = tensor::Tensor::ty(&output_result_id);
    if output_dimensions.len() == 1 && output_type == tensor::TensorType::I32 {
        let output_vec = unsafe {
            std::slice::from_raw_parts(
                output_data.as_ptr() as *const u32,
                output_dimensions[0] as usize,
            )
        }
        .to_vec();
        if output_vec.len() == 1 {
            Ok(output_vec[0])
        } else {
            Err(format!("Unexpected `eos_token_id` tensor dimensions").into())
        }
    } else {
        Err(format!("Unexpected response tensor format").into())
    }
}

fn sample_next_token(
    context: &GraphExecutionContext,
    rnd_num: f32,
) -> Result<(u32, Vec<TokenWithProbabily>), Box<dyn std::error::Error>> {
    let output_result_id =
        inference::GraphExecutionContext::get_output(&context, "last_logits").unwrap();
    let output_data = tensor::Tensor::data(&output_result_id);
    let output_dimensions = tensor::Tensor::dimensions(&output_result_id);
    let output_type = tensor::Tensor::ty(&output_result_id);
    if output_dimensions.len() == 1 && output_type == tensor::TensorType::Fp32 {
        let output_vec_f32 = unsafe {
            std::slice::from_raw_parts(
                output_data.as_ptr() as *const f32,
                output_dimensions[0] as usize,
            )
        };
        let mut logits = Logits::try_from(output_vec_f32.to_vec())?;
        logits.ensure_softmax()?;

        let mut cdf = 0f32;
        let k = 40;
        for i in 0..k {
            cdf += logits.logits[i].prob;
        }
        let mut s = 0f32;
        let rnd_num = rnd_num * cdf;
        let mut selected = k - 1;
        for i in 0..k {
            s += logits.logits[i].prob;
            if s >= rnd_num {
                selected = i;
                break;
            }
        }

        let mut res: Vec<TokenWithProbabily> = vec![];
        for i in 0..k {
            res.push(TokenWithProbabily {
                token_id: logits.logits[i].token_id,
                prob: logits.logits[i].prob,
            });
        }
        Ok((logits.logits[selected].token_id, res))
    } else {
        Err(format!("Unexpected `last_logits` tensor format").into())
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct TokenWithProbabily {
    token_id: u32,
    prob: f32,
}

#[derive(Serialize, Deserialize)]
struct InferenceSession {
    model_name: String,
    query: String,
    response: String,
    rng_seed: u64,
    sampled_tokens: Vec<u32>,
    top_logits: Vec<TokenWithProbabily>,
    hash_sha256: String,
}

impl InferenceSession {
    pub fn new(model_name: &String, query: &str, rng_seed: u64) -> Self {
        Self {
            model_name: model_name.clone(),
            query: query.to_string(),
            response: "".to_string(),
            rng_seed: rng_seed,
            sampled_tokens: Vec::new(),
            top_logits: Vec::new(),
            hash_sha256: "".to_string(),
        }
    }

    pub fn add_token(&mut self, token_id: u32, top_tokens: Vec<TokenWithProbabily>) {
        self.sampled_tokens.push(token_id);
        self.top_logits.extend_from_slice(&top_tokens);
    }

    pub fn finish(&mut self) -> String {
        let token_ids: Vec<u8> = self
            .top_logits
            .iter()
            .copied()
            .flat_map(|x| u32::to_le_bytes(x.token_id).into_iter())
            .collect();
        let token_probs: Vec<u8> = self
            .top_logits
            .iter()
            .copied()
            .flat_map(|x| f32::to_le_bytes(x.prob).into_iter())
            .collect();

        let mut hasher = Sha256::new();
        hasher.update(self.model_name.clone().into_bytes());
        hasher.update(self.query.clone().into_bytes());
        hasher.update(self.response.clone().into_bytes());
        hasher.update(u64::to_le_bytes(self.rng_seed));
        hasher.update(token_ids);
        hasher.update(token_probs);
        // read hash digest and consume hasher
        let result = hasher.finalize();
        self.hash_sha256 = hex::encode(result);
        self.hash_sha256.clone()
    }

    pub fn generate_key(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.model_name.clone().into_bytes());
        hasher.update(self.query.clone().into_bytes());
        hasher.update(u64::to_le_bytes(self.rng_seed));
        let result = hex::encode(hasher.finalize());
        result[0..8].to_string()
    }
}

pub fn llama_infer(
    context: &GraphExecutionContext,
    promt: &str,
    model_name: String,
    seed: u64, 
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let query_tensor_data = promt.to_owned().into_bytes();
    let query_tensor_type = tensor::TensorType::U8;
    let query_tensor_dimensions: Vec<u32> = vec![1, query_tensor_data.len() as u32];
    let query_tensor_id = tensor::Tensor::new(
        &query_tensor_dimensions,
        query_tensor_type,
        &query_tensor_data,
    );
    let query_input_name = "query";
    inference::GraphExecutionContext::set_input(&context, query_input_name, query_tensor_id)
        .unwrap();
    inference::GraphExecutionContext::compute(&context).unwrap();

    let mut session = InferenceSession::new(&model_name, promt, seed);

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..32 {
        let _ = rng.next_u32();
    }

    let mut res = "".to_string();
    let eos_token_id = extract_eos_token_id(&context)?;

    let store = Store::open_default()?;

    let key = session.generate_key();
    println!("Session with key = {key}");

    for _ in 0..2048 {
        let r = rng.next_u32() as f32 * 2.0f32.powf(-32.0f32);
        match sample_next_token(context, r) {
            Ok((token_id, top_tokens)) => {
                if token_id != eos_token_id {
                    let token_tensor_data = token_id.to_le_bytes().to_vec();
                    let token_tensor_type = tensor::TensorType::I32;
                    let token_tensor_dimensions: Vec<u32> = vec![1, 1];
                    let token_tensor_id = tensor::Tensor::new(
                        &token_tensor_dimensions,
                        token_tensor_type,
                        &token_tensor_data,
                    );
                    let token_input_name = "next_token";
                    println!("next token = {token_id} tensor_data = {token_tensor_data:?}");
                    inference::GraphExecutionContext::set_input(
                        &context,
                        token_input_name,
                        token_tensor_id,
                    )
                    .unwrap();
                    inference::GraphExecutionContext::compute(&context).unwrap();
                    let response = extract_response(&context)?;
                    res.push_str(
                        format!("<br> token_id = {token_id} response = {response}").as_str(),
                    );
                    session.add_token(token_id, top_tokens);
                    session.response = response;
                    //let j = serde_json::json!(&session);
                    //println!("{}", j.to_string());
                    let v = store.set_json::<InferenceSession>(key.clone(), &session);
                    println!("updated {:?}", v);
                } else {
                    session.finish();
                    res.push_str(format!("<br> hash_sha256 = {}", session.hash_sha256).as_str());
                    let v = store.set_json::<InferenceSession>(key.clone(), &session);
                    println!("finished {:?}", v);
                    break;
                }
            }
            Err(e) => {
                res.push_str(format!("<br> error = {e}").as_str());
            }
        }
    }

    Ok(res)
}


fn render_session_public_to_http(session: &InferenceSession) -> String {
    let query = &session.query;
    let model = &session.model_name;
    let response = &session.response;
    let hash = &session.hash_sha256;
    let rng_seed = &session.rng_seed;
    let mut res = format!(
        "<div>
        <table>
        <tr>
            <td>Query:</td>
            <td>{query}</td>
        </tr>
        <tr>
            <td>model:</td>
            <td>{model}</td>
        </tr>
        <tr>
            <td>Response:</td>
            <td>{response}</td>
        </tr>
        <tr>
            <td>hash_sha256:</td>
            <td>{hash}</td>
        </tr>
        <tr>
            <td>Rnd seed:</td>
            <td>{rng_seed}</td>
        </tr>
        </table>
        </div>"
    );
    res
}

fn render_session_private_to_http(session: &InferenceSession) -> String {
    let query = &session.query;
    let model = &session.model_name;
    let response = &session.response;
    let hash = &session.hash_sha256;
    let rng_seed = &session.rng_seed;
    let mut res = "<div><table><thead><th>Iteration</th></thead>".to_string();
    let mut offset = 0;
    let mut it = 0;
    let k = session.top_logits.len() / session.sampled_tokens.len();
    for t in &session.sampled_tokens {
        res.push_str("<tr>");
        res.push_str(format!("<td><b>{it}</b></td>").as_str());
        for i in 0..k {
            let token_id = session.top_logits[offset + i].token_id;
            if token_id != *t {
                res.push_str(format!("<td>{token_id}</td>").as_str());
            } else {
                res.push_str(format!("<td><b>{token_id}</b></td>").as_str());
            }
        }
        res.push_str("</tr>");

        res.push_str("<tr>");
        res.push_str(format!("<td> prob[%]</td>").as_str());
        for i in 0..k {
            let prob = session.top_logits[offset + i].prob * 100.0f32;
            res.push_str(format!("<td>{prob:.2}</td>").as_str());
        }
        res.push_str("</tr>");

        it = it + 1;
        offset = offset + k;
    }
    res.push_str("</table></div>");
    res
}



pub fn session_handler(req: http::Request<Vec<u8>>) -> anyhow::Result<Vec<u8>> {
    let path = req.uri().path();
    let path_parts: Vec<_> = path.split('/').map(|x| x.to_string()).collect();
    if path_parts.len() > 2 {
        let store = Store::open_default()?;
        let key = path_parts[2].clone();
        match store.get_json::<InferenceSession>(key)? {
            Some(value) => {
                //return Ok(serde_json::json!(&value).to_string().into());
                let public = render_session_public_to_http(&value);
                let private = render_session_private_to_http(&value);
                let res = format!("{public}</br>{private}");
                return Ok(res.into());
                //return Ok(value);
            }
            None => {}
        }
    }
    Err(anyhow::anyhow!("not found"))
}


pub fn history_handler(req: http::Request<Vec<u8>>) -> anyhow::Result<String> {
    let store = Store::open_default()?;
    let keys = store.get_keys()?;
    let mut res = "".to_string();
    res.push_str("<div>Previous sessions:<br>");
    res.push_str("<table>");
    res.push_str("<thead>");
    res.push_str("<th> Query </th>");
    res.push_str("<th> Seed </th>");
    res.push_str("<th> Used tokens </th>");
    res.push_str("<th> Hash </th>");
    res.push_str("</thead>");

    for key in keys {
        match store.get_json::<InferenceSession>(key.clone())? {
            Some(value) => {
                res.push_str("<tr>");
                res.push_str(format!("<td><a href=\"/session/{}\" > {}</a></td>", &key, &value.query).as_str());
                res.push_str(format!("<td>{}</td>", &value.rng_seed).as_str());
                res.push_str(format!("<td>{}</td>", &value.sampled_tokens.len()).as_str());
                res.push_str(format!("<td>{}</td>", &value.hash_sha256).as_str());
                res.push_str(format!("<td><a href=\"/download/{}\" > Download</a></td>", &key).as_str());
                res.push_str("</tr>");
            }
            _ => {

            }
        }

    }
    res.push_str("</table></div>");
    Ok(res)
}

pub fn download_handler(req: http::Request<Vec<u8>>) -> anyhow::Result<String> {
    let path = req.uri().path();
    let path_parts: Vec<_> = path.split('/').map(|x| x.to_string()).collect();
    if path_parts.len() > 2 {
        let store = Store::open_default()?;
        let key = path_parts[2].clone();
        match store.get_json::<InferenceSession>(key)? {
            Some(value) => {
                return Ok(serde_json::json!(&value).to_string().into());
            }
            None => {}
        }
    }
    Err(anyhow::anyhow!("not found"))

}