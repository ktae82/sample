use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub w_ih: Vec<f64>, // Input to hidden weights
    pub b_h: Vec<f64>,  // Hidden bias
    pub w_ho: Vec<f64>, // Hidden to output weights
    pub b_o: f64,       // Output bias
}

impl Model {
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let f = std::fs::File::open(path)?;
        let model: Model = serde_json::from_reader(f)?;
        Ok(model)
    }

    pub fn get_weights(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        (
            self.w_ih.clone(),
            self.b_h.clone(),
            self.w_ho.clone(),
            self.b_o,
        )
    }
}
