use std::fs::File;

use crate::model::Model;
use crate::neural_network::NeuralNetwork;
use crate::training_data::TrainingData;

pub struct Trainer {
    pub epochs: usize,
    pub learning_rate: f64,
    pub out: String,
}

impl Trainer {
    pub fn new(epochs: usize, learning_rate: f64, out: impl Into<String>) -> Self {
        Self {
            epochs,
            learning_rate,
            out: out.into(),
        }
    }

    pub fn train_and_save_model(&self, training_datas: &Vec<TrainingData>) {
        // Train the neural network
        let mut nn = NeuralNetwork::new(self.learning_rate);

        for epoch in 0..self.epochs {
            let mut total_error = 0.0;

            for data_point in training_datas {
                let (x1, x2) = data_point.get_data();
                let target = data_point.get_target();
                let pred = nn.predict(x1, x2);
                let error = (pred - target).abs();
                total_error += error;
                nn.train(x1, x2, target);
            }

            // Print average error every 10% of epochs
            let avg_error = total_error / training_datas.len() as f64;
            if (epoch + 1) % (self.epochs / 10).max(1) == 0 {
                println!(
                    "Epoch {}/{}: avg error = {:.20}",
                    epoch + 1,
                    self.epochs,
                    avg_error
                );
            }
        }

        let (w1, w2, b) = nn.get_weights();

        // Save the model to a file
        let model = Model { w1, w2, b };
        let file = File::create(&self.out).expect("failed to create output file");
        serde_json::to_writer_pretty(&file, &model).expect("failed to write json model");
        println!("Model saved to {} (json)", self.out);
    }
}
