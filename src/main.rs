mod network;

// use neural_network::Perceptron;
use mnist::MnistBuilder;

use crate::network::Network;

fn main() {
    let mnist = MnistBuilder::new()
        .base_path("data")
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    let train_images = mnist.trn_img; // Vec<u8> of 60000 * 784 pixels
    let train_labels = mnist.trn_lbl; // Vec<u8> of 60000 labels
    let _test_images = mnist.tst_img;   // Vec<u8> of 10000 * 784 pixels
    let _test_labels = mnist.tst_lbl;   // Vec<u8> of 10000 labels

    let mut training_data: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

    train_images
        .chunks(784)
        .zip(train_labels.iter())
        .for_each(|(train_image, &label)| {
            let img_to_255: Vec<f32> = train_image.iter().map(|&x| x as f32 / 255.0).collect();

            training_data.push((img_to_255, vectorized_result(label)));
        });

    let test_data: Vec<(Vec<f32>, u8)> = _test_images
        .chunks(784)
        .zip(_test_labels.iter())
        .map(|(img, &lbl)| {
            let img_f32 = img.iter().map(|&x| x as f32 / 255.0).collect();
            (img_f32, lbl)
        })
        .collect();

    let mut net = Network::new(&[784, 30, 10], 3.0);
    net.sgd(training_data, 30, 10, Some(&test_data));
}

// 1. Turn (labels) у "one-hot encoding" (vector with 10 elements)
// Example: number 3 turns into[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
fn vectorized_result(y: u8) -> Vec<f32> {
    let mut e = vec![0.0; 10];
    e[y as usize] = 1.0;
    e
}
