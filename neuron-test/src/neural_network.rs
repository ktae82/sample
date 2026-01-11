use rand::Rng;

pub struct NeuralNetwork {
    weight1: f64,
    weight2: f64,
    bias: f64,
    learning_rate: f64, // learning rate
}

impl NeuralNetwork {
    pub fn new(learning_rate: f64) -> Self {
        let mut rng = rand::rng();
        Self {
            weight1: rng.random_range(-1.0..1.0),
            weight2: rng.random_range(-1.0..1.0),
            bias: rng.random_range(-1.0..1.0),
            learning_rate,
        }
    }

    // forward
    pub fn predict(&self, x1: f64, x2: f64) -> f64 {
        self.weight1 * x1 + self.weight2 * x2 + self.bias
    }

    // training
    pub fn train(&mut self, x1: f64, x2: f64, target: f64) {
        let y = self.predict(x1, x2);
        let error = y - target;

        self.apply_gradient(self.learning_rate, error, x1, x2);
    }

    // get weights
    pub fn get_weights(&self) -> (f64, f64, f64) {
        (self.weight1, self.weight2, self.bias)
    }

    /// Construct a network from explicit weights/bias and learning rate.
    pub fn from_weights(w1: f64, w2: f64, b: f64, learning_rate: f64) -> Self {
        Self {
            weight1: w1,
            weight2: w2,
            bias: b,
            learning_rate,
        }
    }

    // gradient descent
    fn apply_gradient(&mut self, lr: f64, err: f64, x1: f64, x2: f64) {
        self.weight1 -= lr * err * x1;
        self.weight2 -= lr * err * x2;
        self.bias -= lr * err;
    }
}
