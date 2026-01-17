use clap::{Parser, Subcommand};

mod model;
mod neural_network;
mod trainer;
mod training_data;

use crate::model::Model;
use crate::neural_network::NeuralNetwork;
use crate::trainer::Trainer;
use crate::training_data::TrainingData;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the network and save learned weights to a file
    Train {
        /// Number of epochs
        #[arg(short, long, default_value_t = 5000)]
        epochs: usize,

        /// Learning rate
        #[arg(short, long, default_value_t = 0.01)]
        learning_rate: f64,

        /// Output file for weights
        #[arg(short, long, default_value = "model.json")]
        output: String,
    },

    /// Load a saved model and run inference
    Predict {
        /// model file to load
        #[arg(short, long, default_value = "model.json")]
        model: String,

        /// input1 value
        #[arg(short = 'a', long)]
        input1: Option<f64>,

        /// input2 value
        #[arg(short = 'b', long)]
        input2: Option<f64>,
    },
}

fn train_and_save_model(epochs: usize, learning_rate: f64, output: &str) {
    let training_data = vec![
        TrainingData::new(1.0, 2.0, 3.0),
        TrainingData::new(2.0, 3.0, 5.0),
        TrainingData::new(3.0, 4.0, 7.0),
        TrainingData::new(4.0, 5.0, 9.0),
        TrainingData::new(5.0, 6.0, 11.0),
        TrainingData::new(6.0, 7.0, 13.0),
        TrainingData::new(7.0, 8.0, 15.0),
        TrainingData::new(8.0, 9.0, 17.0),
        TrainingData::new(9.0, 10.0, 19.0),
        TrainingData::new(10.0, 11.0, 21.0),
    ];

    Trainer::new(epochs, learning_rate, output).train_and_save_model(&training_data);
}

fn predict_from_model(path: &str, input1_option: Option<f64>, input2_option: Option<f64>) {
    let input1 = input1_option.unwrap_or(3.0);
    let input2 = input2_option.unwrap_or(5.0);

    let model = Model::load_from_file(path).expect("failed to load model from file");
    let (weight_input_hidden, bias_hidden, weight_hidden_output, bias_output) = model.get_weights();

    let neural_network = NeuralNetwork::from_weights(
        weight_input_hidden,
        bias_hidden,
        weight_hidden_output,
        bias_output,
        0.0,
    );
    let result = neural_network.predict(input1, input2);
    println!("Loaded model with 1 hidden layer (8 neurons)");
    println!("{} + {} ≈ {}", input1, input2, result);
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train {
            epochs,
            learning_rate,
            output,
        } => train_and_save_model(epochs, learning_rate, &output),
        Commands::Predict {
            model,
            input1,
            input2,
        } => predict_from_model(&model, input1, input2),
    }
}
