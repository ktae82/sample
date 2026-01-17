use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub weight_input_hidden: Vec<f64>,  // Input to hidden weights
    pub bias_hidden: Vec<f64>,          // Hidden bias
    pub weight_hidden_output: Vec<f64>, // Hidden to output weights
    pub bias_output: f64,               // Output bias
}

impl Model {
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let f = std::fs::File::open(path)?;
        let model: Model = serde_json::from_reader(f)?;
        Ok(model)
    }

    pub fn get_weights(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        (
            self.weight_input_hidden.clone(),
            self.bias_hidden.clone(),
            self.weight_hidden_output.clone(),
            self.bias_output,
        )
    }
}
