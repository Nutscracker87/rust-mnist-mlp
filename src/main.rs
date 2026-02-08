//! MNIST digit classifier: loads data, trains a small MLP, and runs inference on a custom image.
//! Entry point wires together data loading (data_loader), training (network), and console output.

mod data_loader;
mod network;

use ndarray::Array1;

use crate::data_loader::MnistDataSet;
use crate::network::Network;

fn main() {
    // --- Load data ---
    // MNIST: 60k training + 10k test images, each 28×28, with labels 0–9.
    let data_set = MnistDataSet::new();

    // Load a custom image (e.g. hand-drawn "7") and preprocess to 28×28, same format as MNIST.
    let my_digit = data_loader::create_from_img("seven.png");

    // Optional: grab a random "7" from the training set so we can compare it to our image visually.
    let digit_from_train = data_set
        .get_random_digit(7)
        .expect("Digit 7 exists in training set");

    // Print both as ASCII art (your image vs a training "7").
    visualise_number(&my_digit);
    visualise_number(&digit_from_train);

    // --- Convert to ndarray for the network ---
    // The network expects Array1<f32>; data_loader gives Vec<f32>. Convert and keep (input, target) pairs.
    println!("Preparing training and test data...");
    let training_data: Vec<(Array1<f32>, Array1<f32>)> = data_set
        .training_data
        .into_iter()
        .map(|(img, lbl)| (Array1::from_vec(img), Array1::from_vec(lbl)))
        .collect();

    let test_data: Vec<(Array1<f32>, u8)> = data_set
        .test_data
        .into_iter()
        .map(|(img, lbl)| (Array1::from_vec(img), lbl))
        .collect();

    println!("Data prepared; starting training...");
    let my_digit = Array1::from_vec(my_digit);

    // --- Build and train the network ---
    // Architecture: 784 inputs → 30 hidden neurons → 10 outputs (one per digit). Learning rate 3.0.
    let mut net = Network::new(&[784, 30, 10], 3.0);
    // Train for 30 epochs, mini-batch size 32; evaluate on test_data after each epoch.
    net.sgd(training_data, 30, 32, Some(&test_data));

    // --- Run inference on the custom image ---
    // predict() returns 10 activations (scores for digits 0–9).
    let prediction = net.predict(my_digit.view());

    // The predicted digit is the index of the output with the highest score.
    let result = Network::prediction_to_digit(prediction.view());

    println!("Network predicts: {}", result);
    println!("Output scores (per class): {:?}", prediction);
}

/// Prints a 28×28 digit (784 f32 in [0, 1], row-major) as ASCII art with a border.
/// Pixels: >0.5 = full block, >0.1 = light block, else space. Helps compare your image to training samples.
fn visualise_number(digit: &[f32]) {
    println!("┌{}┐", "──".repeat(28));

    for y in 0..28 {
        print!("│");
        for x in 0..28 {
            let val = digit[y * 28 + x];
            if val > 0.5 {
                print!("██");
            } else if val > 0.1 {
                print!("░░");
            } else {
                print!("  ");
            }
        }
        println!("│");
    }

    println!("└{}┘", "──".repeat(28));
}
