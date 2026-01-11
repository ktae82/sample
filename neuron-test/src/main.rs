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
        lr: f64,

        /// Output file for weights
        #[arg(short, long, default_value = "model.json")]
        out: String,
    },

    /// Load a saved model and run inference
    Predict {
        /// model file to load
        #[arg(short, long, default_value = "model.json")]
        model: String,

        /// x1 value
        #[arg(short = 'a', long)]
        x1: Option<f64>,

        /// x2 value
        #[arg(short = 'b', long)]
        x2: Option<f64>,
    },
}

fn train_and_save_model(epochs: usize, lr: f64, out: &str) {
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

    Trainer::new(epochs, lr, out).train_and_save_model(&training_data);
}

fn predict_from_model(path: &str, x1_opt: Option<f64>, x2_opt: Option<f64>) {
    let x1 = x1_opt.unwrap_or(3.0);
    let x2 = x2_opt.unwrap_or(5.0);

    let model = Model::load_from_file(path).expect("failed to load model from file");
    let (w1, w2, b) = model.get_weights();

    let nn = NeuralNetwork::from_weights(w1, w2, b, 0.0);
    let result = nn.predict(x1, x2);
    println!(
        "Loaded model: w1={:.6}, w2={:.6}, b={:.6}",
        model.w1, model.w2, model.b
    );
    println!("{} + {} ≈ {}", x1, x2, result);
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train { epochs, lr, out } => train_and_save_model(epochs, lr, &out),
        Commands::Predict { model, x1, x2 } => predict_from_model(&model, x1, x2),
    }
}
