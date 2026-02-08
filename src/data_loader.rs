//! MNIST data loading and image preprocessing.
//! Single place for: loading MNIST from disk, normalizing to f32, one-hot labels,
//! and converting user images (e.g. hand-drawn) into MNIST-style 28×28 format.

use image::{imageops, DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma};
use mnist::MnistBuilder;
use rand::seq::IndexedRandom;

/// In-memory MNIST dataset.
///
/// - `training_data`: each element is (pixel vector, one-hot label). Used for training the network.
/// - `test_data`: each element is (pixel vector, digit label 0–9). Used to measure accuracy.
pub struct MnistDataSet {
    /// (784 pixels as f32 in [0,1], one-hot vector of length 10)
    pub training_data: Vec<(Vec<f32>, Vec<f32>)>,
    /// (784 pixels as f32 in [0,1], digit label 0–9)
    pub test_data: Vec<(Vec<f32>, u8)>,
}

impl MnistDataSet {
    /// Loads MNIST from `data/`, normalizes pixels to [0, 1], and builds training/test vectors.
    pub fn new() -> Self {
        // Load raw MNIST files from disk (images + labels for train and test).
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

        // Each MNIST image is 28×28 = 784 pixels. We process them in chunks of 784.
        for (train_image, &label) in train_images.chunks(784).zip(train_labels.iter()) {
            // Normalize pixel values from 0..255 to 0.0..1.0 (neural net works better with small numbers).
            let img_f32: Vec<f32> = train_image.iter().map(|&x| x as f32 / 255.0).collect();
            // One-hot encode the label so the network can learn: e.g. 3 → [0,0,0,1,0,0,0,0,0,0].
            training_data.push((img_f32, vectorized_result(label)));
        }

        // Test set: same pixel normalization, but keep label as a single digit (no one-hot).
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
    /// Useful for demo: e.g. "show me a random 7 from the training set".
    pub fn get_random_digit(&self, searching_digit_label: u8) -> Option<Vec<f32>> {
        let target_idx = searching_digit_label as usize;
        // Collect all training samples whose one-hot label has a 1 at position target_idx.
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

/// One-hot encode a digit label for training: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
/// The network has 10 outputs; we want the correct one to be 1.0 and the rest 0.0.
pub fn vectorized_result(y: u8) -> Vec<f32> {
    let mut e = vec![0.0; 10];
    e[y as usize] = 1.0;
    e
}

/// Crops the image to the smallest rectangle that contains all "content" pixels.
/// Content = pixels brighter than 30 (on 0–255 scale), i.e. the digit, not the background.
/// Returns a view into the original image (SubImage); no copy of pixel data.
pub fn crop_to_content(img: &GrayImage) -> image::SubImage<&GrayImage> {
    // Image size: (width, height) in pixels.
    let (w, h) = img.dimensions();
    // We will track the bounding box of all bright pixels: left, right, top, bottom.
    let mut min_x = w;
    let mut max_x = 0;
    let mut min_y = h;
    let mut max_y = 0;
    let mut found = false;

    // Walk every pixel. p.0[0] is the single grayscale value (0 = black, 255 = white).
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
        // No bright pixels: return the whole image instead of an empty crop.
        image::imageops::crop_imm(img, 0, 0, w, h)
    } else {
        // Add 2 pixels margin on each side so the digit isn't flush against the crop edge.
        // saturating_sub: subtract 2 but never go below 0.
        let x = min_x.saturating_sub(2);
        let y = min_y.saturating_sub(2);
        // Extend right/bottom by 2, but clamp to image bounds (width/height must stay valid).
        let width = (max_x + 2).min(w - 1) - x;
        let height = (max_y + 2).min(h - 1) - y;
        image::imageops::crop_imm(img, x, y, width, height)
    }
}

/// Converts a user image (e.g. hand-drawn digit) to MNIST-like 28×28 grayscale.
/// Pipeline: grayscale → invert (white digit on black) → crop to digit → resize to fit in 20×20
/// → blur → center by center-of-mass on 28×28 canvas.
pub fn prepare_mnist_image(img: &DynamicImage) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    // Step 1: Grayscale only (no color). Then invert so we have white digit on black, like MNIST.
    let mut gray_img = img.to_luma8();
    imageops::invert(&mut gray_img);

    // Step 2: Crop to the rectangle that contains the digit (removes empty borders).
    let cropped = crop_to_content(&gray_img);
    // Dimensions of the cropped region: we need these to scale proportionally.
    let (cw, ch) = cropped.dimensions();

    // Step 3: Scale to fit inside a 20×20 box while keeping aspect ratio (no squashing).
    // If wider than tall: width=20, height = 20*ch/cw. If taller: height=20, width = 20*cw/ch.
    let (new_w, new_h) = if cw > ch {
        (20, (20 * ch / cw).max(1))
    } else {
        ((20 * cw / ch).max(1), 20)
    };
    // .max(1) ensures we never get 0 width or height (would break resize).

    let mut resized = imageops::resize(
        &cropped.to_image(),
        new_w,
        new_h,
        imageops::FilterType::Lanczos3,
    );
    resized = imageops::blur(&resized, 0.5);

    // Step 4: Compute center of mass of the digit (bright pixels "weigh" more).
    // This gives us the visual center so we can place it in the middle of the 28×28 canvas.
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

    // Center of mass = (sum_x / total_mass, sum_y / total_mass). If no bright pixels, use geometric center.
    let (com_x, com_y) = if total_mass > 0.0 {
        (sum_x / total_mass, sum_y / total_mass)
    } else {
        (new_w as f32 / 2.0, new_h as f32 / 2.0)
    };

    // Step 5: We want the digit's center of mass at (14, 14) on the 28×28 canvas (middle).
    let x_offset = (14.0 - com_x).round() as i32;
    let y_offset = (14.0 - com_y).round() as i32;

    // Step 6: Create black 28×28 canvas and draw the resized digit, shifted by the offsets.
    let mut canvas = ImageBuffer::new(28, 28);

    for y in 0..new_h {
        for x in 0..new_w {
            let p = resized.get_pixel(x, y);
            // Slight brightness boost so faint strokes still register well.
            let enhanced_val = (p.0[0] as f32 * 1.5).min(255.0) as u8;

            let target_x = x as i32 + x_offset;
            let target_y = y as i32 + y_offset;

            // Only write pixels that fall inside the 28×28 canvas (some may shift off the edge).
            if target_x >= 0 && target_x < 28 && target_y >= 0 && target_y < 28 {
                canvas.put_pixel(target_x as u32, target_y as u32, Luma([enhanced_val]));
            }
        }
    }
    canvas
}

/// Loads an image from disk and converts it to the same format as MNIST training data:
/// 784 floats in [0, 1], row-major order (same as the network expects).
pub fn create_from_img(path: &str) -> Vec<f32> {
    let img = image::open(path).expect("image should exist");
    let my_digit_prep = prepare_mnist_image(&img);
    // Flatten 28×28 to 784 values and normalize to [0, 1] like training data.
    let my_digit: Vec<f32> = my_digit_prep
        .pixels()
        .map(|p| p.0[0] as f32 / 255.0)
        .collect();

    my_digit
}
