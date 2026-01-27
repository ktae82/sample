use rand::Rng;

const HIDDEN_SIZE: usize = 8;

pub struct NeuralNetwork {
    // Input to hidden layer weights (2 x 8)
    weight_input_hidden: [f64; HIDDEN_SIZE * 2],
    // Hidden layer bias (8)
    bias_hidden: [f64; HIDDEN_SIZE],
    // Hidden to output layer weights (8 x 1)
    weight_hidden_output: [f64; HIDDEN_SIZE],
    // Output layer bias (1)
    bias_output: f64,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(learning_rate: f64) -> Self {
        let mut rng = rand::rng();
        let mut weight_input_hidden = [0.0; HIDDEN_SIZE * 2];
        let mut bias_hidden = [0.0; HIDDEN_SIZE];
        let mut weight_hidden_output = [0.0; HIDDEN_SIZE];

        for item in &mut weight_input_hidden {
            *item = rng.random_range(-1.0..1.0);
        }
        for item in &mut bias_hidden {
            *item = rng.random_range(-1.0..1.0);
        }
        for item in &mut weight_hidden_output {
            *item = rng.random_range(-1.0..1.0);
        }
        let bias_output = rng.random_range(-1.0..1.0);

        Self {
            weight_input_hidden,
            bias_hidden,
            weight_hidden_output,
            bias_output,
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
    pub fn predict(&self, input1: f64, input2: f64) -> f64 {
        // Input to hidden
        let mut hidden = [0.0; HIDDEN_SIZE];
        for (neuron_index, hidden_value) in hidden.iter_mut().enumerate() {
            let activation_input = self.weight_input_hidden[neuron_index * 2] * input1
                + self.weight_input_hidden[neuron_index * 2 + 1] * input2
                + self.bias_hidden[neuron_index];
            *hidden_value = Self::sigmoid(activation_input);
        }

        // Hidden to output
        let mut output_activation_input = self.bias_output;
        for (neuron_index, hidden_value) in hidden.iter().enumerate() {
            output_activation_input += self.weight_hidden_output[neuron_index] * hidden_value;
        }

        output_activation_input
    }

    // Training with backpropagation
    pub fn train(&mut self, input1: f64, input2: f64, target: f64) {
        // Forward pass with intermediate values
        let mut hidden_activation_input = [0.0; HIDDEN_SIZE];
        let mut hidden_activation = [0.0; HIDDEN_SIZE];

        // Input to hidden
        for (neuron_index, activation_input_value) in hidden_activation_input.iter_mut().enumerate()
        {
            *activation_input_value = self.weight_input_hidden[neuron_index * 2] * input1
                + self.weight_input_hidden[neuron_index * 2 + 1] * input2
                + self.bias_hidden[neuron_index];
            hidden_activation[neuron_index] = Self::sigmoid(*activation_input_value);
        }

        // Hidden to output
        let mut output_activation_input = self.bias_output;
        for (neuron_index, activation_value) in hidden_activation.iter().enumerate() {
            output_activation_input += self.weight_hidden_output[neuron_index] * activation_value;
        }

        // Backward pass
        let output_error = output_activation_input - target;

        // Output layer gradients
        for (neuron_index, weight) in self.weight_hidden_output.iter_mut().enumerate() {
            let delta_weight_hidden_output = output_error * hidden_activation[neuron_index];
            *weight -= self.learning_rate * delta_weight_hidden_output;
        }
        self.bias_output -= self.learning_rate * output_error;

        // Hidden layer gradients
        for (neuron_index, weight) in self.weight_hidden_output.iter().enumerate() {
            let hidden_error =
                output_error * weight * Self::sigmoid_derivative(hidden_activation[neuron_index]);

            // Update weights from input to hidden
            self.weight_input_hidden[neuron_index * 2] -=
                self.learning_rate * hidden_error * input1;
            self.weight_input_hidden[neuron_index * 2 + 1] -=
                self.learning_rate * hidden_error * input2;

            // Update hidden layer bias
            self.bias_hidden[neuron_index] -= self.learning_rate * hidden_error;
        }
    }

    // Get weights for serialization
    pub fn get_weights(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        (
            self.weight_input_hidden.to_vec(),
            self.bias_hidden.to_vec(),
            self.weight_hidden_output.to_vec(),
            self.bias_output,
        )
    }

    /// Construct a network from explicit weights and learning rate.
    pub fn from_weights(
        weight_input_hidden: Vec<f64>,
        bias_hidden: Vec<f64>,
        weight_hidden_output: Vec<f64>,
        bias_output: f64,
        learning_rate: f64,
    ) -> Self {
        let mut weight_input_hidden_arr = [0.0; HIDDEN_SIZE * 2];
        let mut bias_hidden_arr = [0.0; HIDDEN_SIZE];
        let mut weight_hidden_output_arr = [0.0; HIDDEN_SIZE];

        weight_input_hidden_arr.copy_from_slice(&weight_input_hidden);
        bias_hidden_arr.copy_from_slice(&bias_hidden);
        weight_hidden_output_arr.copy_from_slice(&weight_hidden_output);

        Self {
            weight_input_hidden: weight_input_hidden_arr,
            bias_hidden: bias_hidden_arr,
            weight_hidden_output: weight_hidden_output_arr,
            bias_output,
            learning_rate,
        }
    }
}
