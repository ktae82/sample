use rand::Rng;

const HIDDEN_SIZE: usize = 8;

pub struct NeuralNetwork {
    // Input to hidden layer weights (2 x 8)
    w_ih: [f64; HIDDEN_SIZE * 2],
    // Hidden layer bias (8)
    b_h: [f64; HIDDEN_SIZE],
    // Hidden to output layer weights (8 x 1)
    w_ho: [f64; HIDDEN_SIZE],
    // Output layer bias (1)
    b_o: f64,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(learning_rate: f64) -> Self {
        let mut rng = rand::rng();
        let mut w_ih = [0.0; HIDDEN_SIZE * 2];
        let mut b_h = [0.0; HIDDEN_SIZE];
        let mut w_ho = [0.0; HIDDEN_SIZE];

        for i in 0..w_ih.len() {
            w_ih[i] = rng.random_range(-1.0..1.0);
        }
        for i in 0..b_h.len() {
            b_h[i] = rng.random_range(-1.0..1.0);
        }
        for i in 0..w_ho.len() {
            w_ho[i] = rng.random_range(-1.0..1.0);
        }
        let b_o = rng.random_range(-1.0..1.0);

        Self {
            w_ih,
            b_h,
            w_ho,
            b_o,
            learning_rate,
        }
    }

    // Sigmoid activation function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    // Sigmoid derivative
    fn sigmoid_derivative(y: f64) -> f64 {
        y * (1.0 - y)
    }

    // Forward pass
    pub fn predict(&self, x1: f64, x2: f64) -> f64 {
        // Input to hidden
        let mut hidden = [0.0; HIDDEN_SIZE];
        for j in 0..HIDDEN_SIZE {
            let z = self.w_ih[j * 2] * x1 + self.w_ih[j * 2 + 1] * x2 + self.b_h[j];
            hidden[j] = Self::sigmoid(z);
        }

        // Hidden to output
        let mut z_o = self.b_o;
        for j in 0..HIDDEN_SIZE {
            z_o += self.w_ho[j] * hidden[j];
        }

        z_o
    }

    // Training with backpropagation
    pub fn train(&mut self, x1: f64, x2: f64, target: f64) {
        // Forward pass with intermediate values
        let mut hidden_z = [0.0; HIDDEN_SIZE];
        let mut hidden_a = [0.0; HIDDEN_SIZE];

        // Input to hidden
        for j in 0..HIDDEN_SIZE {
            hidden_z[j] = self.w_ih[j * 2] * x1 + self.w_ih[j * 2 + 1] * x2 + self.b_h[j];
            hidden_a[j] = Self::sigmoid(hidden_z[j]);
        }

        // Hidden to output
        let mut z_o = self.b_o;
        for j in 0..HIDDEN_SIZE {
            z_o += self.w_ho[j] * hidden_a[j];
        }

        // Backward pass
        let output_error = z_o - target;

        // Output layer gradients
        for j in 0..HIDDEN_SIZE {
            let dw_ho = output_error * hidden_a[j];
            self.w_ho[j] -= self.learning_rate * dw_ho;
        }
        self.b_o -= self.learning_rate * output_error;

        // Hidden layer gradients
        for j in 0..HIDDEN_SIZE {
            let hidden_error = output_error * self.w_ho[j] * Self::sigmoid_derivative(hidden_a[j]);

            // Update weights from input to hidden
            self.w_ih[j * 2] -= self.learning_rate * hidden_error * x1;
            self.w_ih[j * 2 + 1] -= self.learning_rate * hidden_error * x2;

            // Update hidden layer bias
            self.b_h[j] -= self.learning_rate * hidden_error;
        }
    }

    // Get weights for serialization
    pub fn get_weights(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        (
            self.w_ih.to_vec(),
            self.b_h.to_vec(),
            self.w_ho.to_vec(),
            self.b_o,
        )
    }

    /// Construct a network from explicit weights and learning rate.
    pub fn from_weights(
        w_ih: Vec<f64>,
        b_h: Vec<f64>,
        w_ho: Vec<f64>,
        b_o: f64,
        learning_rate: f64,
    ) -> Self {
        let mut w_ih_arr = [0.0; HIDDEN_SIZE * 2];
        let mut b_h_arr = [0.0; HIDDEN_SIZE];
        let mut w_ho_arr = [0.0; HIDDEN_SIZE];

        w_ih_arr.copy_from_slice(&w_ih);
        b_h_arr.copy_from_slice(&b_h);
        w_ho_arr.copy_from_slice(&w_ho);

        Self {
            w_ih: w_ih_arr,
            b_h: b_h_arr,
            w_ho: w_ho_arr,
            b_o,
            learning_rate,
        }
    }
}
