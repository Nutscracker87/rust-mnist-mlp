//! MNIST digit classifier: loads data, trains a small MLP, and runs inference on a custom image.
//! Entry point wires together data loading (data_loader), training (network), and console output.

mod data_loader;
mod network;

use ndarray::Array1;

use crate::data_loader::MnistDataSet;
use crate::network::Network;

fn main() {
    // Load MNIST and build (input, target) pairs for training and test
    let data_set = MnistDataSet::new();

    // Load and preprocess a custom digit image (e.g. hand-drawn) to match MNIST 28×28 format
    let my_digit = data_loader::create_from_img("seven.png");

    // Optional: show a sample digit from the training set for comparison
    let digit_from_train = data_set
        .get_random_digit(7)
        .expect("Digit 7 exists in training set");

    visualise_number(&my_digit);
    visualise_number(&digit_from_train);

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

    print!("The program reached here");    
    let my_digit = Array1::from_vec(my_digit);

    // Build network (784 → 30 → 10) and train with mini-batch SGD
    let mut net = Network::new(&[784, 30, 10], 2.0);
    net.sgd(training_data, 30, 10, Some(&test_data));

    // Predict class for the custom image; output is 10 class scores
    let prediction = net.predict(my_digit.view());

    // Predicted digit = index of the output neuron with highest activation
    let result = Network::prediction_to_digit(prediction.view());

    println!("Network predicts: {}", result);
    println!("Output scores (per class): {:?}", prediction);
}

/// Prints a 28×28 digit (slice of 784 f32 in [0,1]) as ASCII art with a simple border.
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
