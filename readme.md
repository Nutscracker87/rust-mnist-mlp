# rust-mnist-mlp

MNIST digit classifier using a multilayer perceptron (MLP) and backpropagation. Implemented in Rust with sigmoid activation and mini-batch stochastic gradient descent (SGD).

---

## Program flow at a glance

The program runs in four main stages:

1. **Prepare custom image** — Turn your image (e.g. hand-drawn digit) into MNIST-style 28×28 grayscale.
2. **Load MNIST data** — Read training/test images from disk, normalize pixels, one-hot encode labels.
3. **Train the network** — Mini-batch SGD: shuffle data, split into batches, backprop + gradient update per batch, optionally evaluate on test set each epoch.
4. **Inference** — Run the trained network on your custom image and print the predicted digit.

Below, each stage is broken down into concrete steps with code references.

---

## Quick start

**Prerequisites:** Rust toolchain (e.g. `rustup`), MNIST data in the `data/` folder.

1. **Get MNIST data** (if not already present):
   - Place the four IDX files in `data/`: `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`.
   - Or download and extract from [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist) (or a mirror) into `data/`.

2. **Run the program:**
   ```bash
   cargo run
   ```
   The program loads MNIST, trains the network (784 → 30 → 10) for 30 epochs, then runs inference on a custom image `seven.png`. Put a 28×28-friendly digit image at `seven.png` (or change the path in `main.rs`).

3. **Optional:** To try your own digit image, use the same name in code or call `data_loader::create_from_img("your_image.png")` to preprocess it to MNIST-style 28×28 grayscale.

---

## Project structure and workflow

- **Data** — Load MNIST, normalize pixels to [0,1], one-hot encode labels. → `data_loader::MnistDataSet::new()`, `vectorized_result()`
- **Custom image** — Load and preprocess a hand-drawn digit to 28×28. → `data_loader::create_from_img()`, `prepare_mnist_image()`
- **Build network** — Create MLP with given layer sizes and learning rate. → `Network::new(&[784, 30, 10], learning_rate)`
- **Training** — Shuffle data, split into mini-batches, backprop + gradient step per batch. → `Network::sgd()`, which calls `update_mini_batch()`
- **Single batch step** — For each sample: forward → deltas → gradients; then average and update weights. → `update_mini_batch()` → `backprop()` → `forward()`, `compute_deltas()`, `compute_gradients()`
- **Inference** — Forward pass only; predicted digit = argmax of output. → `Network::predict()`, `Network::prediction_to_digit()`
- **Evaluation** — Count correct predictions on test set. → `Network::evaluate()`

**Loss:** MSE per output, $C = \frac{1}{2} \sum (y - t)^2$.  
**Update rule:** $w \leftarrow w - (\eta/|\text{batch}|) \sum \nabla w$, $b \leftarrow b - (\eta/|\text{batch}|) \sum \nabla b$.

---

## Detailed workflow by stage

Step-by-step logic for each part of the program. Code locations are in `src/` (e.g. `data_loader.rs`, `network.rs`, `main.rs`).

---

### Stage 1: Preparing a custom image

Goal: convert an arbitrary image (e.g. photo or drawing of a digit) into the same format as MNIST — 28×28 grayscale, white digit on black background, digit centered — so the network can process it.

1. Load the image from disk (e.g. PNG). → `create_from_img(path)`, `image::open(path)`
2. Convert to grayscale (single channel 0–255). MNIST has no colour. → `prepare_mnist_image()`, `img.to_luma8()`
3. Invert colours so the digit is white on black (MNIST convention). → `imageops::invert(&mut gray_img)`
4. **Crop to content:** find the smallest rectangle that contains all "bright" pixels (value > 30). This removes empty borders. Add a 2-pixel margin on each side. → `crop_to_content()` in `data_loader.rs`
5. Get the cropped region's width and height. Scale so the image **fits inside 20×20** while keeping aspect ratio (no squashing). E.g. 100×50 → 20×10. → `cropped.dimensions()`, then `(new_w, new_h)` with 20×20 fit logic
6. Resize the cropped image to that size; apply a slight blur so edges look more like MNIST. → `imageops::resize()`, `imageops::blur()`
7. Compute the **center of mass** of the digit (bright pixels "weigh" more). We want to place this point at the centre of the 28×28 canvas. → Loop over pixels: `sum_x += x * val`, same for y; `com = (sum_x, sum_y) / total_mass`
8. Create a black 28×28 canvas. Draw the resized image on it so that its center of mass sits at (14, 14). Pixels that would fall outside 28×28 are clipped. Optionally boost brightness slightly. → `ImageBuffer::new(28, 28)`, loop with `target_x = x + x_offset`, bounds check, `put_pixel`
9. Flatten the 28×28 image to 784 floats and normalize to [0, 1] (divide by 255). This is the same format as MNIST training vectors. → `create_from_img()` → `pixels().map(|p| p.0[0] as f32 / 255.0).collect()`

**Entry point:** `data_loader::create_from_img("path.png")` returns `Vec<f32>` of length 784.

---

### Stage 2: Loading MNIST data

Goal: load the official MNIST train/test sets from disk and convert them into in-memory vectors the network can use (normalized pixels + one-hot labels for training, digit labels for test).

1. Read the four IDX files from `data/`: train images, train labels, test images, test labels. → `MnistBuilder::new().base_path("data").finalize()`
2. Training images: each image is 28×28 = 784 bytes. Process in chunks of 784. Convert each pixel from 0..255 to 0.0..1.0. → `train_images.chunks(784)`, `map(|x| x as f32 / 255.0)`
3. For each training sample, one-hot encode the label: digit 3 → `[0,0,0,1,0,0,0,0,0,0]`. Store pairs `(pixel_vec, one_hot_vec)`. → `vectorized_result(label)` in `MnistDataSet::new()`
4. Test images: same pixel normalization. Keep the label as a single digit (0–9) for accuracy evaluation. Store pairs `(pixel_vec, label)`. → `test_data` in `MnistDataSet::new()`
5. In `main`, convert `Vec<f32>` to `ndarray::Array1<f32>` for the network. Training: `Vec<(Array1, Array1)>`; test: `Vec<(Array1, u8)>`. → `map(|(img, lbl)| (Array1::from_vec(img), ...))`

**Entry point:** `MnistDataSet::new()` populates `training_data` and `test_data`. `main.rs` then converts them to `Array1` and passes to the network.

---

### Stage 3: Training the network

Goal: adjust weights and biases so that the network’s outputs (10 scores) match the one-hot targets on the training set. We use mini-batch SGD: small batches of samples, gradient computed per sample and averaged, then one update per batch.

#### 3.1 Build the network

1. Define architecture by layer sizes, e.g. `[784, 30, 10]`: 784 inputs (pixels), 30 hidden neurons, 10 outputs (one per digit). → `Network::new(&[784, 30, 10], learning_rate)`
2. For each consecutive pair of sizes, create one layer: random weights and biases in [-1, 1). So we get two weight matrices: 784→30 and 30→10. → `Layer::new(sizes[i], sizes[i+1])` in a loop

#### 3.2 One epoch of training

One epoch = one full pass over the training set, in mini-batches.

1. Shuffle the training data so batches are different every epoch. → `training_data.shuffle(&mut rng)` in `sgd()`
2. Split the training data into chunks of size `mini_batch_size` (e.g. 32). Each chunk is one mini-batch. → `training_data.chunks(mini_batch_size)`
3. For **each mini-batch**, run a single gradient step (see "One mini-batch step" below). → `update_mini_batch(batch)`
4. (Optional) After the epoch, run the network on the test set and count how many samples are classified correctly (prediction = argmax of output). Print the count. → `evaluate(test_data)` in `sgd()`

#### 3.3 One mini-batch step

For one batch of (input, target) pairs:

1. For **each sample** in the batch, run backpropagation: forward pass (cache activations and weighted sums), compute deltas δ for each layer, then compute gradients ∂C/∂w and ∂C/∂b. This can be done in parallel over samples. → `batch.par_iter().map(|(input, target)| self.backprop(...))`
2. Sum the weight gradients and bias gradients over all samples in the batch. → Loop over `delta_grads`, add each layer's gradients to `total_grad_w`, `total_grad_b`
3. Update every weight and bias: subtract `(learning_rate / batch_len) * total_gradient`. So we use the *average* gradient over the batch. → `weights.scaled_add(-step, &total_grad_w[l])`, same for biases

#### 3.4 Backpropagation (per sample)

For one (input, target) pair, `backprop()` does:

1. **Forward:** Feed input through each layer; store each layer's output **a** and pre-sigmoid sum **z**. → `forward(network_input)` → `all_layers_outputs`, `all_weighted_sums`
2. **Deltas:** Output layer: δ = (a − target) ⊙ σ'(z). Hidden layers (from last hidden backward): δ^l = (W^{l+1}ᵀ δ^{l+1}) ⊙ σ'(z^l). → `compute_deltas(...)`
3. **Gradients:** ∂C/∂b = δ; ∂C/∂w = δ ⊗ a_prev (outer product). Return gradients for all layers. → `compute_gradients(layers_deltas, all_layers_outputs)`

**Entry point:** `Network::sgd(training_data, epochs, mini_batch_size, Some(&test_data))` in `main.rs`.

---

### Stage 4: Inference

Goal: given a single image (e.g. your custom 28×28 vector), run the network and get the predicted digit 0–9.

1. Run a forward pass through all layers (no caching, no backward). Output is a vector of 10 activations (scores for digits 0–9). → `Network::predict(input.view())`
2. The predicted digit is the **index** of the output with the highest value. E.g. if output[7] is largest, prediction is 7. → `Network::prediction_to_digit(prediction.view())`
3. (Optional) Print the raw 10 scores and the predicted digit. → `main.rs`: `println!("Network predicts: {}", result)`

**Entry point:** In `main.rs`, after training: `net.predict(my_digit.view())` and `Network::prediction_to_digit(prediction.view())`.

---

### Code map

- **Custom image prep** — `data_loader.rs`: `create_from_img()`, `prepare_mnist_image()`, `crop_to_content()`
- **MNIST loading** — `data_loader.rs`: `MnistDataSet::new()`, `vectorized_result()`
- **Network build & training** — `network.rs`: `Network::new()`, `sgd()`, `update_mini_batch()`, `backprop()`, `forward()`, `compute_deltas()`, `compute_gradients()`
- **Inference & main flow** — `main.rs`: `main()`, `visualise_number()`; calls into `data_loader` and `Network`

---

## Notation

- **L** — number of weight layers in the network.
- **$w^{(l)}_{jk}$** — weight from neuron **k** in layer **l−1** to neuron **j** in layer **l** (in code: `weights[j][k]` for that layer).
- **$b^{(l)}_{j}$** — bias of neuron **j** in layer **l**.
- **$z^{(l)}_{j}$** — pre-activation (weighted sum) of neuron **j** in layer **l**.
- **$a^{(l)}_{j}$** — activation (output) of neuron **j** in layer **l**; **$a^{(0)}$** is the network input.
- **$\sigma(z)$** — Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- **$\sigma'(z)$** — derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
- **$t$** (or **$y$** in target) — target output (one-hot for training).

---

## Stage 1: Forward pass

For each layer **l** from 1 to L we compute pre-activation and activation; these are cached for the backward pass.

**Weighted sum (pre-activation):**
$$
z^{(l)}_{j} = \sum_{k} w^{(l)}_{jk} \cdot a^{(l-1)}_{k} + b^{(l)}_{j}
$$

**Activation:**
$$
a^{(l)}_{j} = \sigma(z^{(l)}_{j})
$$

**In code:** `Network::forward()` (used inside `backprop()`). It returns all $a$ (including input) and all $z$ per layer. Single-layer forward is `Layer::calculate_output()`; full inference without cache is `Network::predict()`.

---

## Stage 2: Backward pass (deltas)

We compute the error signal $\delta$ for each layer, from output back to input.

### Output layer ($\delta^{(L)}$)

$$
\delta^{(L)}_{j} = (a^{(L)}_{j} - t_{j}) \cdot \sigma'(z^{(L)}_{j})
$$

$t_j$ is the target (one-hot) for output $j$. This is the derivative of the MSE loss with respect to $z^{(L)}_j$ via the chain rule.

### Hidden layers ($\delta^{(l)}$)

Error is propagated backward using the weights of the *next* layer:

$$
\delta^{(l)}_{j} = \left( \sum_{k} w^{(l+1)}_{kj} \cdot \delta^{(l+1)}_{k} \right) \cdot \sigma'(z^{(l)}_{j})
$$

So we combine the deltas of layer $l+1$, weighted by the connections back to neuron $j$, then multiply by $\sigma'(z^{(l)}_j)$.

**In code:** `Network::compute_deltas()`. It takes the cached `all_layers_outputs` and `all_weighted_sums` from the forward pass, plus the target, and returns a `Vec<Vec<f32>>` of deltas (one `Vec<f32>` per layer).

---

## Stage 3: Gradients

Using the cached $a$ and the computed $\delta$, we get the gradients of the cost with respect to weights and biases.

**Bias gradient:**
$$
\frac{\partial C}{\partial b^{(l)}_{j}} = \delta^{(l)}_{j}
$$

**Weight gradient:**
$$
\frac{\partial C}{\partial w^{(l)}_{jk}} = a^{(l-1)}_{k} \cdot \delta^{(l)}_{j}
$$

**In code:** `Network::compute_gradients()`. It takes `layers_deltas` and `all_layers_outputs` and returns weight gradients and bias gradients (same shapes as the network parameters).

---

## Full training step (per sample and per batch)

1. **Forward:** Compute and store all $z$ and $a$ → `forward()`.
2. **Backward:** Compute $\delta$ for every layer → `compute_deltas()`.
3. **Gradients:** Compute $\partial C/\partial w$ and $\partial C/\partial b$ → `compute_gradients()`.
4. **Batch:** For a mini-batch, sum the gradients over all samples, then update:
   - $w \leftarrow w - (\eta/n) \sum \nabla w$, $b \leftarrow b - (\eta/n) \sum \nabla b$,
   where $n$ is the batch size → done inside `update_mini_batch()` after calling `backprop()` for each sample and accumulating.

**In code:** `backprop()` orchestrates steps 1–3. `update_mini_batch()` loops over the batch, calls `backprop()` for each (input, target), accumulates gradients, then applies the update with step size $\eta/n$.

---

## Summary

| Theory | Code |
|--------|------|
| Forward: $z^l = W^l a^{l-1} + b^l$, $a^l = \sigma(z^l)$ | `forward()`, `Layer::calculate_output()` |
| Output delta: $\delta^L = (a^L - t) \odot \sigma'(z^L)$ | `compute_deltas()` (output layer) |
| Hidden delta: $\delta^l = (W^{l+1})^T \delta^{l+1} \odot \sigma'(z^l)$ | `compute_deltas()` (hidden loop) |
| $\partial C/\partial b = \delta$, $\partial C/\partial w = \delta \cdot a^{\mathrm{in}}$ | `compute_gradients()` |
| Mini-batch SGD update | `update_mini_batch()`, `sgd()` |

---

## Implementation note

Weight indexing in code: `weights[neuron_j][input_k]` corresponds to $w^{(l)}_{jk}$ (connection from input index $k$ to neuron index $j$). This matches the formula $z_j = \sum_k w_{jk} a_k + b_j$.
