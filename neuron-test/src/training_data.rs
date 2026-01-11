pub struct TrainingData {
    x1: f64,
    x2: f64,
    target: f64,
}

impl TrainingData {
    pub fn new(x1: f64, x2: f64, target: f64) -> Self {
        Self { x1, x2, target }
    }

    pub fn get_data(&self) -> (f64, f64) {
        (self.x1, self.x2)
    }

    pub fn get_target(&self) -> f64 {
        self.target
    }
}
