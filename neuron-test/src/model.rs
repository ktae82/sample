use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub w1: f64,
    pub w2: f64,
    pub b: f64,
}

impl Model {
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let f = std::fs::File::open(path)?;
        let model: Model = serde_json::from_reader(f)?;
        Ok(model)
    }

    pub fn get_weights(&self) -> (f64, f64, f64) {
        (self.w1, self.w2, self.b)
    }
}
