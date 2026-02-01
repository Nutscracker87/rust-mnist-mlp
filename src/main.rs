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

    let train_images = mnist.trn_img;  // Vec<u8> of 60000 * 784 pixels
    let train_labels = mnist.trn_lbl;  // Vec<u8> of 60000 labels
    let _test_images = mnist.tst_img;   // Vec<u8> of 10000 * 784 pixels
    let _test_labels = mnist.tst_lbl;   // Vec<u8> of 10000 labels

    let nn = Network::new(&[784, 30, 10], 0.5);
    let img_u8 = &train_images[0..784];
    let label = &train_labels[0];

    let img_f32: Vec<f32> = img_u8.iter().map(|&x| (x as f32)/255.0 ).collect();

    let prediction = nn.predict(&img_f32);

    println!("Реальна цифра: {}", label);
    println!("Прогноз мережі (10 імовірностей):");
    for (digit, confidence) in prediction.iter().enumerate() {
        println!("{}: {:.4}", digit, confidence);
    }

    // Display first 5 digits as ASCII art
    // for i in 1..15 {
    //     let label = train_labels[i];
    //     let image = &train_images[i * 784..(i + 1) * 784];
        
    //     println!("\n=== Digit: {} (index {}) ===", label, i);
    //     for row in 0..28 {
    //         for col in 0..28 {
    //             let pixel = image[row * 28 + col];
    //             // Map pixel intensity to ASCII characters
    //             let ch = match pixel {
    //                 0..=50 => ' ',
    //                 51..=100 => '.',
    //                 101..=150 => '+',
    //                 151..=200 => '*',
    //                 201..=255 => '#',
    //             };
    //             print!("{}", ch);
    //         }
    //         println!();
    //     }
    // }
}
