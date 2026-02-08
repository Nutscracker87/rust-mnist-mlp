//! MNIST data loading and image preprocessing.
//! Single place for: loading MNIST from disk, normalizing to f32, one-hot labels,
//! and converting user images (e.g. hand-drawn) into MNIST-style 28×28 format.

use image::{imageops, DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma};
use mnist::MnistBuilder;
use rand::seq::IndexedRandom;

/// In-memory MNIST dataset: training pairs (input, one-hot target) and test pairs (input, label).
pub struct MnistDataSet {
    pub training_data: Vec<(Vec<f32>, Vec<f32>)>,
    pub test_data: Vec<(Vec<f32>, u8)>,
}

impl MnistDataSet {
    /// Loads MNIST from `data/`, normalizes pixels to [0, 1], and builds training/test vectors.
    pub fn new() -> Self {
        let mnist = MnistBuilder::new()
            .base_path("data")
            .label_format_digit()
            .training_set_length(60_000)
            .test_set_length(10_000)
            .finalize();

        let train_images = mnist.trn_img;
        let train_labels = mnist.trn_lbl;
        let test_images = mnist.tst_img;
        let test_labels = mnist.tst_lbl;

        let mut training_data: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

        for (train_image, &label) in train_images.chunks(784).zip(train_labels.iter()) {
            let img_f32: Vec<f32> = train_image.iter().map(|&x| x as f32 / 255.0).collect();
            training_data.push((img_f32, vectorized_result(label)));
        }

        let test_data: Vec<(Vec<f32>, u8)> = test_images
            .chunks(784)
            .zip(test_labels.iter())
            .map(|(img, &lbl)| {
                let img_f32: Vec<f32> = img.iter().map(|&x| x as f32 / 255.0).collect();
                (img_f32, lbl)
            })
            .collect();

        Self {
            training_data,
            test_data,
        }
    }

    /// Returns one random training sample with the given digit label (0–9), or None if none exist.
    pub fn get_random_digit(&self, searching_digit_label: u8) -> Option<Vec<f32>> {
        let target_idx = searching_digit_label as usize;
        let candidates: Vec<&Vec<f32>> = self
            .training_data
            .iter()
            .filter(|(_pixels, label_vec)| {
                label_vec.get(target_idx).map_or(false, |&val| val > 0.9)
            })
            .map(|(pixels, _label_vec)| pixels)
            .collect();

        let mut rng = rand::rng();
        candidates.choose(&mut rng).map(|&pixels| pixels.clone())
    }
}

/// One-hot encode a digit label: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
pub fn vectorized_result(y: u8) -> Vec<f32> {
    let mut e = vec![0.0; 10];
    e[y as usize] = 1.0;
    e
}

/// Crops image to the bounding box of pixels brighter than a noise threshold (30).
pub fn crop_to_content(img: &GrayImage) -> image::SubImage<&GrayImage> {
    let (w, h) = img.dimensions();
    let mut min_x = w;
    let mut max_x = 0;
    let mut min_y = h;
    let mut max_y = 0;
    let mut found = false;

    for (x, y, p) in img.enumerate_pixels() {
        if p.0[0] > 30 {
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
            found = true;
        }
    }

    if !found {
        image::imageops::crop_imm(img, 0, 0, w, h)
    } else {
        let x = min_x.saturating_sub(2);
        let y = min_y.saturating_sub(2);
        let width = (max_x + 2).min(w - 1) - x;
        let height = (max_y + 2).min(h - 1) - y;
        image::imageops::crop_imm(img, x, y, width, height)
    }
}

/// Converts a user image (e.g. hand-drawn digit) to MNIST-like 28×28 grayscale:
/// invert, crop, resize to fit in 20×20, center on 28×28, slight blur.
pub fn prepare_mnist_image(img: &DynamicImage) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let mut gray_img = img.to_luma8();
    imageops::invert(&mut gray_img);

    let cropped = crop_to_content(&gray_img);
    let (cw, ch) = cropped.dimensions();

    // Scale to fit within 20x20
    let (new_w, new_h) = if cw > ch {
        (20, (20 * ch / cw).max(1))
    } else {
        ((20 * cw / ch).max(1), 20)
    };

    let mut resized = imageops::resize(
        &cropped.to_image(),
        new_w,
        new_h,
        imageops::FilterType::Lanczos3,
    );
    resized = imageops::blur(&resized, 0.5);

    // --- NEW: CENTER OF MASS CALCULATION ---
    let mut total_mass = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;

    for y in 0..new_h {
        for x in 0..new_w {
            let pixel_val = resized.get_pixel(x, y).0[0] as f32;
            sum_x += x as f32 * pixel_val;
            sum_y += y as f32 * pixel_val;
            total_mass += pixel_val;
        }
    }

    // Default to geometric center if image is empty
    let (com_x, com_y) = if total_mass > 0.0 {
        (sum_x / total_mass, sum_y / total_mass)
    } else {
        (new_w as f32 / 2.0, new_h as f32 / 2.0)
    };

    // Calculate offsets to place Center of Mass at (14, 14)
    let x_offset = (14.0 - com_x).round() as i32;
    let y_offset = (14.0 - com_y).round() as i32;

    // --- CREATE CANVAS ---
    let mut canvas = ImageBuffer::new(28, 28);

    for y in 0..new_h {
        for x in 0..new_w {
            let p = resized.get_pixel(x, y);
            // Boost signal to ensure it's not too faint
            let enhanced_val = (p.0[0] as f32 * 1.5).min(255.0) as u8;

            let target_x = x as i32 + x_offset;
            let target_y = y as i32 + y_offset;

            // Only draw if within 28x28 bounds
            if target_x >= 0 && target_x < 28 && target_y >= 0 && target_y < 28 {
                canvas.put_pixel(target_x as u32, target_y as u32, Luma([enhanced_val]));
            }
        }
    }
    canvas
}

/// Load and preprocess a custom digit image (e.g. hand-drawn) to match MNIST 28×28 format
pub fn create_from_img(path: &str) -> Vec<f32> {
    let img = image::open(path).expect("image should exist");
    let my_digit_prep = prepare_mnist_image(&img);
    let my_digit: Vec<f32> = my_digit_prep
        .pixels()
        .map(|p| p.0[0] as f32 / 255.0)
        .collect();

    my_digit
}
